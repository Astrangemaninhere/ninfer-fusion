#pragma once

// W13 P0: weight host-offload skeleton (layer/expert granularity, byte mirror).
//
// The artifact reader already mmaps the whole file (src/artifact/reader.cpp:201,
// PROT_READ MAP_PRIVATE), so the weight SOURCE is a pageable host mapping: no new
// streaming layer. Offload therefore means: decide at load time which weight spans
// stay device-resident and which fault in on use, then page the offloaded spans
// from the mmap -> pinned staging -> H2D. The fetch cache is modelled on
// src/ops/ple/ple_table.h (bounded pinned LRU + epoch-exempt eviction for
// in-flight pointers); the pinned allocator is INJECTED so this header stays
// host-only and unit-testable (the engine injects cudaHostAlloc/writeCombined).
//
// P0 SCOPE (this header): residency classes, the plan carrier, classification,
// and the fault cache. The per-GEMM hot-path hook is deliberately NOT here:
// see _collab/A_s32_w13_p0.md for the specified hook point, per-GEMM cost, and
// locking analysis (PleTable's lock-free current-epoch exemption is the
// precedent). Until that hook lands, --weight-host-bytes fails loudly at CLI
// parse (serve_options.cpp) -- an offload flag that silently keeps everything
// resident would be a silent no-op, which W13 must never be.
//
// Budget policy (W13 x KV-cold-tier coexistence): TWO separate budgets.
// cold_host_bytes (EngineOptions) stays scoped to the KV cold tier; weights get
// their own budget. Lifetimes and failure modes differ (KV cold pages age per
// request; weight spans are static after load), and a shared pool would let KV
// pressure evict weight staging invisibly. Exhaustion semantics: pinned-budget
// LRU eviction is NORMAL operation (bandwidth only, never precision -- spans are
// byte mirrors); registering a span larger than the whole pinned budget throws;
// exhausting the device page arena at fault time (P1) must throw with the span
// id. Never silently degrade.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::product {

// Residency class for one weight span, decided at load time.
enum class WeightResidency : std::uint8_t {
    Resident, // device arena slot, copied once at load (today's behaviour)
    Host,     // no permanent device slot; mmap -> pinned -> H2D on use
    Disk,     // reserved (explicit tiering); P0 never emits it
};

// One addressable weight span. Dense models use expert = -1; MoE targets use
// (layer, expert). artifact_offset/bytes identify the span inside the mmap.
struct WeightSpan {
    std::uint32_t layer = 0;
    std::int32_t expert = -1;
    WeightResidency residency = WeightResidency::Resident;
    std::uint64_t artifact_offset = 0;
    std::uint64_t bytes = 0;

    [[nodiscard]] std::string label() const {
        return "L" + std::to_string(layer) +
               (expert < 0 ? std::string(":dense") : ":e" + std::to_string(expert));
    }
};

// The plan carrier (mirrors how layer_residual / layer_sliding_windows ride the
// KV plan): one entry per addressable weight span, in artifact order.
struct WeightResidencyPlan {
    std::vector<WeightSpan> spans;

    [[nodiscard]] std::uint64_t bytes_by(WeightResidency want) const {
        std::uint64_t total = 0;
        for (const WeightSpan& s : spans) {
            if (s.residency == want) { total += s.bytes; }
        }
        return total;
    }
    [[nodiscard]] std::uint64_t resident_bytes() const {
        return bytes_by(WeightResidency::Resident);
    }
    [[nodiscard]] std::uint64_t host_bytes() const { return bytes_by(WeightResidency::Host); }
};

// Load-time classification. host_budget_bytes = how much weight payload may
// leave the device arena (EngineOptions.weight_host_offload_bytes). Policy P0:
// experts first (per-token sparse coverage), then DENSE layers DEEPEST-FIRST
// (mirror of the KV deep-protection instinct: shallow layers fault least often
// under prefill-heavy traffic). Deterministic, byte-lossless either way.
inline void classify_weight_residency(std::vector<WeightSpan>& spans,
                                      std::uint64_t host_budget_bytes) {
    std::vector<std::size_t> order(spans.size());
    for (std::size_t i = 0; i < order.size(); ++i) { order[i] = i; }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const bool ea = spans[a].expert >= 0;
        const bool eb = spans[b].expert >= 0;
        if (ea != eb) { return ea; }               // experts first
        if (spans[a].layer != spans[b].layer) {    // then deepest first
            return spans[a].layer > spans[b].layer;
        }
        return a < b;                              // stable tie-break
    });
    std::uint64_t remaining = host_budget_bytes;
    for (const std::size_t i : order) {
        if (remaining == 0) { break; }
        if (spans[i].bytes <= remaining) {
            spans[i].residency = WeightResidency::Host;
            remaining -= spans[i].bytes;
        }
        // A span larger than the remaining budget stays Resident; a span larger
        // than the WHOLE budget can never be offloaded by this policy (the P1
        // fault cache would throw on registration -- loud, not silent).
    }
}

// Bounded pinned LRU fault cache (PleTable pattern, host-only, injectable
// allocators). The engine-side H2D step is one cudaMemcpyAsync at the consumer
// (documented in A_s32_w13_p0.md): fault() makes the span's bytes available at
// a pinned host pointer; the consumer enqueues the copy to the span's fixed
// device page slot.
class WeightPageCache {
public:
    using PinnedAlloc = std::function<void*(std::size_t)>;
    using PinnedFree = std::function<void(void*)>;

    WeightPageCache(std::uint64_t pinned_budget_bytes, PinnedAlloc alloc, PinnedFree free_fn)
        : pinned_budget_(pinned_budget_bytes), alloc_(std::move(alloc)),
          free_(std::move(free_fn)) {}

    ~WeightPageCache() {
        for (auto& [id, entry] : entries_) {
            if (entry.pinned != nullptr) { free_(entry.pinned); }
        }
    }

    WeightPageCache(const WeightPageCache&) = delete;
    WeightPageCache& operator=(const WeightPageCache&) = delete;

    // Registers a fault source span (bytes live in the artifact mmap at src).
    // A span larger than the whole pinned budget is rejected loudly: it could
    // never be faulted.
    void register_span(std::uint32_t layer, std::int32_t expert, const void* src,
                       std::uint64_t bytes) {
        if (bytes == 0) { throw std::invalid_argument("weight span is empty"); }
        if (bytes > pinned_budget_) {
            throw std::invalid_argument("weight span L" + std::to_string(layer) +
                                        " exceeds the whole pinned budget");
        }
        Entry entry{};
        entry.layer = layer;
        entry.expert = expert;
        entry.src = static_cast<const std::byte*>(src);
        entry.bytes = bytes;
        entries_.emplace(next_id_++, entry);
    }

    [[nodiscard]] std::size_t span_count() const noexcept { return entries_.size(); }
    [[nodiscard]] std::uint64_t pinned_bytes_used() const noexcept { return used_; }

    // Fault epoch (PleTable gather_epoch_ analogue): pointers handed out during
    // the current epoch are exempt from eviction until a NEW epoch begins (the
    // consumer may have already recorded them for an in-flight H2D/kernel);
    // begin_fault_epoch() both closes the previous epoch and opens the next.
    // Single-threaded in P0; the P1 hot-path hook adds a mutex on the miss path
    // only (hit path = atomic epoch read, PleTable precedent).
    void begin_fault_epoch() noexcept { ++epoch_; }

    // Returns the pinned pointer for span_id, faulting from the mmap source on
    // a miss (memcpy -- the mmap is the byte-exact mirror). The miss path makes
    // room FIRST: if eviction cannot free enough (only in-flight spans), it
    // throws BEFORE allocating anything, so a failed fault leaves the cache
    // state unchanged (exception safety found by the S32 host check).
    void* fault(std::size_t span_id) {
        auto it = entries_.find(span_id);
        if (it == entries_.end()) { throw std::out_of_range("unknown weight span id"); }
        Entry& entry = it->second;
        if (entry.pinned == nullptr) {
            make_room(entry.bytes);
            void* pinned = alloc_(static_cast<std::size_t>(entry.bytes));
            if (pinned == nullptr) {
                throw std::runtime_error("weight pinned allocation failed for span " +
                                         std::to_string(span_id));
            }
            std::memcpy(pinned, entry.src, static_cast<std::size_t>(entry.bytes));
            entry.pinned = pinned;
            used_ += entry.bytes;
        }
        entry.last_use_seq = ++use_seq_;
        entry.last_epoch = epoch_.load();
        return entry.pinned;
    }

    // Evicts LRU entries (by monotonic use sequence, deterministic under ties)
    // until used_ + incoming fits the budget. Entries faulted during the
    // CURRENT epoch are exempt (their pointers may be in flight).
    void make_room(std::uint64_t incoming_bytes) {
        const std::uint64_t current = epoch_.load();
        while (used_ + incoming_bytes > pinned_budget_) {
            std::size_t victim = static_cast<std::size_t>(-1);
            std::uint64_t oldest = UINT64_MAX;
            for (auto& [id, entry] : entries_) {
                if (entry.pinned == nullptr) { continue; }
                if (entry.last_epoch >= current) { continue; } // in-flight exemption
                if (entry.last_use_seq < oldest) {
                    oldest = entry.last_use_seq;
                    victim = id;
                }
            }
            if (victim == static_cast<std::size_t>(-1)) {
                throw std::runtime_error(
                    "weight pinned budget exhausted with only in-flight spans; "
                    "increase the budget");
            }
            Entry& entry = entries_.at(victim);
            free_(entry.pinned);
            used_ -= entry.bytes;
            entry.pinned = nullptr;
        }
    }

private:
    struct Entry {
        std::uint32_t layer = 0;
        std::int32_t expert = -1;
        const std::byte* src = nullptr;
        std::uint64_t bytes = 0;
        void* pinned = nullptr;
        std::uint64_t last_use_seq = 0;
        std::uint64_t last_epoch = 0;
    };

    std::uint64_t pinned_budget_;
    PinnedAlloc alloc_;
    PinnedFree free_;
    std::unordered_map<std::size_t, Entry> entries_;
    std::size_t next_id_ = 0;
    std::uint64_t used_ = 0;
    std::uint64_t use_seq_ = 0;
    std::atomic<std::uint64_t> epoch_{0};
};

} // namespace ninfer::product
