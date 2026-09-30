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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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


// ===========================================================================
// W13 P1 -- the offload runtime: layer-grouped classification, a cyclic device
// arena with STABLE per-layer addresses, and the layer-ordered fetch schedule.
//
// Why this shape and not an LRU page cache: every target caches weight POINTERS
// at bind time (TypedBinding -> Weight/Tensor views, and the decode path is
// captured into a CUDA graph whose memcpy/compute nodes freeze those pointers).
// A pointer that can move is therefore not a loading detail, it is a rewrite of
// every GEMM entry. The arena below gives each offloaded LAYER a fixed offset
// `slot(L) = (L - first_offloaded_layer) % arena_layers`, so the address a span
// is bound to is a pure function of the layer index and NEVER changes. What
// cycles is the CONTENT of the slot, not the pointer.
//
// Budget split (three pools, deliberately not one):
//   * host_pinned_bytes      -- the pinned host mirror of the offloaded weights.
//                               Allocated once at load, never evicted: the
//                               mirror is what the prefetch reads from.
//   * device_arena_bytes     -- the device working set, `arena_layers` layer
//                               strips. THIS is the number that must be smaller
//                               than the offloaded byte count, and
//                               `device_bytes_freed = offloaded - arena`.
//   * everything else        -- untouched (KV cold tiers keep their own pools;
//                               see kv_cold_tier_budget.h's note that W13 follows
//                               the same "make the contradiction loud" rule).


// ===========================================================================
// W13 P1 -- classification, the offload plan, and the fetch schedule.
//
// Nothing below reads a compiled-in list of layers, experts or models. The span
// set and its (layer, expert) grouping come from the artifact's own object names
// (the targets' bind loops build them as "<prefix>layers/<N>/..."), and the sizes
// come from the artifact descriptors.
// ===========================================================================

// One device weight object as the artifact describes it. `name` is the artifact
// object name (structure, not a model fact); the two numbers are its span inside
// the mmap.
struct WeightSpanSource {
    std::string name;
    std::uint64_t artifact_offset = 0;
    std::uint64_t bytes           = 0;
};

// Where a span lives in the model's execution order. expert = -1 for dense.
struct WeightSpanLocation {
    std::uint32_t layer = 0;
    std::int32_t expert = -1;
};

namespace detail {

// Reads the first "<key>/<digits>/" segment of an artifact object name, or
// "<key>/<digits>" when it ends the name.
[[nodiscard]] inline bool parse_indexed_segment(std::string_view name, std::string_view key,
                                                std::uint32_t& out) noexcept {
    const std::string needle = "/" + std::string(key) + "/";
    std::size_t cursor       = std::string_view::npos;
    std::size_t at           = name.find(needle);
    if (at != std::string_view::npos) {
        cursor = at + needle.size();
    } else {
        const std::string bare = std::string(key) + "/";
        if (name.size() > bare.size() && name.compare(0, bare.size(), bare) == 0) {
            cursor = bare.size();
        }
    }
    if (cursor == std::string_view::npos) { return false; }
    std::uint64_t value = 0;
    std::size_t digits  = 0;
    while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
        value = value * 10 + static_cast<std::uint64_t>(name[cursor] - '0');
        ++digits;
        ++cursor;
    }
    if (digits == 0) { return false; }
    if (cursor < name.size() && name[cursor] != '/') { return false; }
    out = static_cast<std::uint32_t>(value);
    return true;
}

// The one artifact root whose "/layers/" segment IS a layer key.
//
// Every other user of the same spelling belongs to no layer -- the vision tower's
// vision/layers/N/... and the draft heads' dflash/layers/N/... and dflash2/layers/N/...
// are exactly the objects the contract below lists as layer-less -- yet "/layers/"
// matched at ANY root, so vision layer N and text layer N were ONE key: 324 of one real
// artifact's 333 vision objects carry "/layers/", and they shared text layer N's arena
// strip. The collision was only unreachable because the default span floor
// (binder.cpp: max(1 MiB, device_capacity_bytes / 4096)) happened to sit above the
// 2,679,744 B vision spans on a 16 GiB arena -- a derived number, not an invariant; an
// arena under 4 GiB or any --weight-span-floor-bytes <= 2,679,744 makes it reachable.
[[nodiscard]] inline bool has_layer_root(std::string_view name) noexcept {
    constexpr std::string_view root = "text/";
    return name.size() > root.size() && name.compare(0, root.size(), root) == 0;
}

} // namespace detail

// (layer, expert) for an artifact weight object, or nullopt for an object that
// belongs to no layer (token embedding, final norm, output head, MTP/DFlash
// heads, vision tower). Those stay resident: they are read every step and are
// small, so offloading them would cost PCIe bandwidth for no device saving.
[[nodiscard]] inline std::optional<WeightSpanLocation>
weight_span_location(std::string_view object_name) noexcept {
    // The key is ROOT-QUALIFIED: "/layers/" denotes a layer only under a root that has
    // one. A vision or draft-head object stays resident with the other layer-less
    // objects instead of being grouped into a text layer's offload strip.
    if (!detail::has_layer_root(object_name)) { return std::nullopt; }
    std::uint32_t layer = 0;
    if (!detail::parse_indexed_segment(object_name, "layers", layer)) { return std::nullopt; }
    WeightSpanLocation out;
    out.layer            = layer;
    std::uint32_t expert = 0;
    if (detail::parse_indexed_segment(object_name, "experts", expert)) {
        out.expert = static_cast<std::int32_t>(expert);
    }
    return out;
}

// Operator knobs. Defaults are constructed by the caller from measured numbers,
// never picked here.
struct WeightOffloadLimits {
    std::uint64_t host_pinned_bytes  = 0; // 0 = offload disabled
    std::uint64_t device_arena_bytes = 0; // 0 = derive from prefetch_layers
    std::uint32_t prefetch_layers    = 2; // layers looked ahead (>= 2, see slot rule)
    // Objects smaller than this are not worth a slot (norms, biases, a_log...).
    // Derived by the caller from the artifact, not typed in here.
    std::uint64_t min_span_bytes = 0;
    // THE RESIDENCY CONTRACT. The caller asserts that EVERY pass over the layer
    // order enters every offloaded layer through note_layer() before its weights
    // are used -- prefill chunks, verify/decode steps and MTP rounds alike.
    //
    // It defaults to FALSE because that is the engine's truth today: the only hook
    // is prefill-only (src/targets/qwen3_6/impl/runtime/text_context_impl.h, the
    // `ph == Phase::Prefill` gate), so a plan admitted with this false would be
    // read stale for every decode step. See build_weight_offload_plan()'s refusal
    // for the measured consequence.
    bool fetch_per_layer_entry = false;
    // THE OTHER HALF OF THE SAME CONTRACT (notehook), and it is not an opinion: on this tree
    // `use_cuda_graph` gates every CUDA-graph capture of the layer walk
    // (src/targets/qwen3_6/impl/runtime/decode_impl.h:85, mtp_impl.h:420, dflash_impl.h:615),
    // and a graph REPLAYS the slot addresses the arena handed out at CAPTURE time. An arena
    // that rotates can therefore only be correct while somebody re-fetches between replays,
    // and nothing does: the fetch is a HOST-side cudaMemcpyAsync on the transfer stream,
    // which is illegal to issue while the consuming stream is being captured
    // (src/ops/stream_capture.h). So a ROTATING arena plus a captured decode is the
    // silent-stale case W13 exists to refuse, and the caller states it HERE rather than
    // letting the plan builder guess it from a flag it cannot see.
    bool decode_graph_captured = false;
};

// F1059/streamfix -- THE ONE KNOB THAT KEEPS THE CAPTURE REFUSAL ALIVE WHILE IT IS MEASURED.
//
// The plan-time refusal below exists because a ROTATING arena read by a captured decode has no
// re-fetch between replays. `fetch_enroll()` is that re-fetch: the copies are captured as graph
// nodes, and the slot address of a layer is `layer_index(layer) % arena_layers`
// (WeightOffloadPlan::arena_slot_of), computed ONCE in the runtime's constructor and never
// recomputed -- so a replay writes each layer into the SAME strip the graph was captured with.
// Rotation is therefore reproducible at capture time, which is the condition the refusal was
// standing in for.
//
// The refusal is NOT deleted here. It fires unless the operator asks for the capturable path
// explicitly, so a tree in which `fetch_enroll()` is not implemented still refuses rather than
// reading stale bytes in silence; and the coordinator's rule -- a guard is not removed before the
// captured arm has been MEASURED byte-identical to its resident twin -- is obeyed by construction.
// DEFAULTS TO TRUE, and that default is the measurement rather than the intent: with the fetch
// enrolled, the captured arm is byte-identical to the resident arm of the same graph-on
// configuration (41 B / 0474f45fad71764945050fde3d0340065d0d86c74b35243444b7da5c6dd9b96e on 8
// tokens, 168 B / 994134e6fe9a53c6b9d8e8ab6a80ec315db84989714d65f214cbddb3a0e83923 on 32) --
// offloaded == resident, which is the gate. So the capturable path is what an operator gets, and
// `NINFER_W13_CAPTURABLE_FETCH=0` is the OPT-OUT that restores the pre-F1059 refusal.
//
// WHY THE OPT-OUT IS KEPT AT ALL. A refusal that can no longer be reached is a refusal that can no
// longer be read, and the guard's message is the only place a reader learns why a rotating arena
// under capture is dangerous. It also keeps this plan builder honest for a backend that does NOT
// implement fetch_enroll(): that configuration must still refuse rather than read stale bytes.
[[nodiscard]] inline bool capturable_fetch_requested() noexcept {
    const char* raw = std::getenv("NINFER_W13_CAPTURABLE_FETCH");
    if (raw == nullptr) { return true; }
    return *raw != '\0' && *raw != '0';
}

// The result of classification: which spans leave the device arena, how the
// cyclic arena is laid out, and what was actually freed.
struct WeightOffloadPlan {
    std::vector<WeightSpan> spans;          // offloaded spans, ARTIFACT order
    std::vector<std::uint64_t> layer_offsets; // byte offset inside the layer strip
    std::vector<std::size_t> source_indices;  // index into the sources vector
    std::vector<std::uint32_t> layers;        // offloaded layers, ascending
    std::uint32_t layer_stride = 0;
    std::uint32_t arena_layers = 0;
    std::uint64_t offloaded_bytes    = 0;
    std::uint64_t device_arena_bytes = 0;
    std::uint64_t device_bytes_freed = 0;

    [[nodiscard]] bool empty() const noexcept { return spans.empty(); }
    [[nodiscard]] std::uint32_t first_layer() const { return layers.front(); }
    [[nodiscard]] std::uint32_t last_layer() const { return layers.back(); }
    [[nodiscard]] bool owns_layer(std::uint32_t layer) const noexcept {
        return !layers.empty() && layer >= layers.front() && layer <= layers.back();
    }
    [[nodiscard]] std::uint32_t layer_index(std::uint32_t layer) const {
        if (!owns_layer(layer)) { throw std::out_of_range("weight offload: layer not offloaded"); }
        const std::uint32_t at = layer - layers.front();
        if (layers[at] != layer) {
            throw std::out_of_range("weight offload: offloaded layers are not contiguous");
        }
        return at;
    }
    // slot(L) = (L - first) % arena_layers. Two layers can only collide if they are
    // `arena_layers` apart, and the schedule never keeps more than `arena_layers`
    // layers live, so a live layer's strip is never overwritten by a prefetch.
    [[nodiscard]] std::uint32_t arena_slot_of(std::uint32_t layer) const {
        return layer_index(layer) % arena_layers;
    }
    // STABLE device byte offset of a span: its arena layer strip plus its offset
    // inside that strip. Invariant across every fetch -- this is the property that
    // keeps the offload invisible to the GEMMs and to the captured decode graph.
    [[nodiscard]] std::uint64_t device_offset_of(std::size_t span_index) const {
        if (span_index >= spans.size()) {
            throw std::out_of_range("weight offload span index out of range");
        }
        return static_cast<std::uint64_t>(arena_slot_of(spans[span_index].layer)) * layer_stride +
               layer_offsets[span_index];
    }
};

// Classification. Policy, all of it derived from the artifacts + the budget:
//   1. drop objects outside any layer, and objects below min_span_bytes
//   2. group the survivors by layer
//   3. take layer groups DEEPEST-FIRST until the pinned budget would be exceeded
//      (greedy over an ascending list, so the selection is a contiguous SUFFIX --
//      which is what makes slot(L) one cyclic function)
//   4. size the arena at `prefetch_layers` strips of the widest selected layer
// Throws when the budget admits nothing offloadable, and when the arena is not
// strictly smaller than what is offloaded: that would be a false promise of freed
// memory, and W13 must never degrade silently.
[[nodiscard]] inline WeightOffloadPlan
build_weight_offload_plan(const std::vector<WeightSpanSource>& sources,
                          const WeightOffloadLimits& limits) {
    WeightOffloadPlan plan;
    // =======================================================================================
    // [F-1019] THE VALIDATION IS MOVED ABOVE THE GATE. THIS IS THE WHOLE EDIT.
    // =======================================================================================
    // THE DEFECT, EXACTLY. `prefetch_layers` had exactly two consumers in this file, both BELOW
    // the `host_pinned_bytes == 0` early return (`:448` and `:563` in the pre-image). So when the
    // host mirror is not funded, the parameter was neither USED nor CHECKED: `--weight-prefetch-layers
    // 0` and `--weight-prefetch-layers 1` -- the two values this function's own contract calls
    // illegal, "would let the arena slot a layer is being computed from be overwritten" -- were
    // accepted without a word, and the run continued with a flag the operator believes is in force.
    // That is the "accepted and ignored" shape the record calls this project's worst outcome.
    //
    // WHY THIS IS THE COMPLEMENT OF F-958 AND NOT A SUPERSESSION OF IT. F-958 (`dl/knobproof`)
    // compiled and two-state-exercised a REFUSAL at the FRONT DOOR -- `src/serve/serve_options.cpp`
    // `if` :994 / `throw` :995, copied into `apps/cli/options.cpp` :1175/:1176 and
    // `apps/perplexity/main.cpp` :270/:271 -- and measured `--weight-prefetch-layers 4` ALONE going
    // from ACCEPTED-AND-STORED to REFUSED. That refusal closes the CLI spelling. It does NOT close
    // this function, which is the HOME of the parameter and is reachable from any caller that builds
    // `WeightOffloadLimits` directly (`src/artifact/binder.cpp:198` is the only one on this tree).
    // A front-door refusal plus an unguarded home is the same defect one layer down; this edit makes
    // the home refuse values it already declared illegal, whatever door was used.
    //
    // WHAT IT DELIBERATELY DOES NOT DO. It does NOT make `prefetch_layers` a reason to build a plan:
    // with `host_pinned_bytes == 0` there is no arena to prefetch into, so an empty plan remains the
    // CORRECT answer and the gate stays. It also does NOT turn the early return into a throw: a
    // caller with no host budget is a legitimate configuration, not an error, and F-958's arm A6
    // ("`4` BEFORE `--weight-host-bytes 4G`" stays legal) is the measured proof that the ordering of
    // argv must not change legality. Only the PARAMETER'S OWN CONTRACT is enforced here.
    if (limits.prefetch_layers < 2) {
        throw std::invalid_argument("W13 prefetch depth below 2 would let the arena slot a "
                                    "layer is being computed from be overwritten");
    }
    if (limits.host_pinned_bytes == 0) {
        // The gate, now AFTER the parameter's contract is settled. Named as a reason rather than
        // left as a bare `return plan;`, so a reader of the empty plan knows which of the two
        // possible empties this is: no host mirror was funded, so nothing is offloadable.
        return plan; // no host mirror funded: the correct plan is empty, and `prefetch_layers`
                     // has still been validated above.
    }

    struct Grouped {
        std::uint32_t layer = 0;
        std::vector<std::size_t> sources;
        std::uint64_t bytes = 0;
    };
    std::vector<Grouped> groups;
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const WeightSpanSource& source = sources[i];
        if (source.bytes == 0 || source.bytes < limits.min_span_bytes) { continue; }
        const std::optional<WeightSpanLocation> at = weight_span_location(source.name);
        if (!at.has_value()) { continue; }
        auto it = std::find_if(groups.begin(), groups.end(),
                               [&](const Grouped& g) { return g.layer == at->layer; });
        if (it == groups.end()) {
            groups.push_back(Grouped{at->layer, {}, 0});
            it = groups.end() - 1;
        }
        it->sources.push_back(i);
        it->bytes += source.bytes;
    }
    if (groups.empty()) {
        throw std::invalid_argument("--weight-host-bytes is set but no layer-scoped weight "
                                    "object reached the span floor; nothing could be offloaded");
    }
    std::sort(groups.begin(), groups.end(),
              [](const Grouped& a, const Grouped& b) { return a.layer < b.layer; });

    std::uint64_t used = 0;
    std::size_t first  = groups.size();
    for (std::size_t i = groups.size(); i-- > 0;) {
        if (used + groups[i].bytes > limits.host_pinned_bytes) { break; }
        used += groups[i].bytes;
        first = i;
    }
    if (first == groups.size()) {
        throw std::invalid_argument("--weight-host-bytes does not admit a single layer group; "
                                    "increase it or lower the span floor");
    }
    for (std::size_t g = first; g < groups.size(); ++g) {
        plan.layers.push_back(groups[g].layer);
    }

    // Byte offset of each span inside its layer strip, accumulated in artifact
    // order so the layout is deterministic for a given artifact.
    std::unordered_map<std::uint32_t, std::uint64_t> strip_cursor;
    // EVERY span start is aligned, not just the strip base. `device_offset_of()`
    // hands the GEMM `arena + slot*stride + layer_offsets[i]`, and the resident
    // path aligns every object the same way (Binder::materialize_on_device ->
    // align_up(capacity, tensor_alignment(layout)), tensor_alignment == 256).
    // Packing spans back to back is what made offload-only runs fail: an NVFP4
    // payload is align_up(n*k/2,256) + n*k/16 + sizeof(float), which is ALWAYS
    // 4 (mod 16), so the second and later NVFP4 span of a strip started at
    // %16==4 and validate_nvfp4_weight refused it ("algn=0/0") -- a refusal only
    // the offload path could produce, because the resident path aligns. The
    // padding lives only in the device arena: `offloaded_bytes` stays the pure
    // sum of the span bytes, which is the pinned-RAM bill and the H2D traffic.
    constexpr std::uint64_t kStripAlignment = 256; // == tensor_alignment(layout)
    std::uint64_t widest = 0;
    for (std::size_t g = first; g < groups.size(); ++g) {
        for (const std::size_t source_index : groups[g].sources) {
            WeightSpan span;
            span.layer           = groups[g].layer;
            span.expert          = -1;
            span.residency       = WeightResidency::Host;
            span.artifact_offset = sources[source_index].artifact_offset;
            span.bytes           = sources[source_index].bytes;
            plan.spans.push_back(span);
            plan.source_indices.push_back(source_index);
            std::uint64_t& cursor = strip_cursor[groups[g].layer];
            cursor = (cursor + kStripAlignment - 1) / kStripAlignment * kStripAlignment;
            plan.layer_offsets.push_back(cursor);
            cursor += span.bytes;
            plan.offloaded_bytes += span.bytes;
        }
        // `widest` is read from the ALIGNED cursor: a span has to fit inside its
        // own strip, or the cyclic arena slot would spill into the next layer.
        widest = std::max(widest, strip_cursor[groups[g].layer]);
    }
    for (const auto& [layer, bytes] : strip_cursor) {
        if (bytes > widest) { throw std::logic_error("weight offload strip overflow"); }
        (void)layer;
    }

    plan.layer_stride = static_cast<std::uint32_t>(
        (widest + kStripAlignment - 1) / kStripAlignment * kStripAlignment);

    // Post-condition, and the machine check for the bug this block exists for: if
    // the per-span alignment above is ever dropped, one of these fires HERE
    // instead of surfacing much later as a refused NVFP4 weight ("algn=0/0") or,
    // worse, as a silently wrong GEMM. Both halves are demonstrably reachable:
    // tests/test_weight_residency.cpp::test_span_starts_are_strip_aligned proves
    // the fixture is a misalignment case before requiring the plan to be aligned.
    for (std::size_t s = 0; s < plan.spans.size(); ++s) {
        if (plan.layer_offsets[s] % kStripAlignment != 0) {
            throw std::logic_error("weight offload span is not strip-aligned");
        }
        if (plan.layer_offsets[s] + plan.spans[s].bytes > plan.layer_stride) {
            throw std::logic_error("weight offload span escapes its layer strip");
        }
    }
    plan.arena_layers = static_cast<std::uint32_t>(plan.layers.size());
    if (limits.device_arena_bytes != 0) {
        if (limits.device_arena_bytes < plan.layer_stride) {
            // F1, MADE ACTIONABLE (notehook). The refusal stays -- an arena that cannot hold one
            // strip cannot hold a layer, and a value silently raised to the minimum would be a
            // flag accepted and ignored -- but it now NAMES the number it needs instead of
            // making the operator bisect for it. NOTE the second half, because it is the part
            // that changed: a 1-STRIP cyclic arena IS sufficient once the per-layer fetch is
            // asserted, because every layer is entered through note_layer() before its weights
            // are read and the join makes the consuming stream wait for THAT layer's H2D.
            throw std::invalid_argument(
                "--weight-device-arena-bytes cannot hold one layer strip of the offloaded set: "
                "the widest offloaded layer strip is " + std::to_string(plan.layer_stride) +
                " B and the arena is " + std::to_string(limits.device_arena_bytes) +
                " B. Raise --weight-device-arena-bytes to at least " +
                std::to_string(plan.layer_stride) +
                " B (exactly one strip is CORRECT with the per-layer fetch asserted), or lower "
                "--weight-host-bytes so a shallower tail fits under the arena available");
        }
        const std::uint32_t asked = static_cast<std::uint32_t>(limits.device_arena_bytes /
                                                              plan.layer_stride);
        plan.arena_layers = std::min(asked, plan.arena_layers);
    } else {
        plan.arena_layers = std::min(limits.prefetch_layers, plan.arena_layers);
    }
    plan.device_arena_bytes = static_cast<std::uint64_t>(plan.layer_stride) * plan.arena_layers;
    if (plan.device_arena_bytes >= plan.offloaded_bytes) {
        throw std::invalid_argument(
            "W13 would free no device memory: the offload arena (" +
            std::to_string(plan.device_arena_bytes) + " B) is not smaller than the offloaded "
            "weight bytes (" + std::to_string(plan.offloaded_bytes) +
            " B). Lower --weight-device-arena-bytes or raise --weight-host-bytes");
    }
    plan.device_bytes_freed = plan.offloaded_bytes - plan.device_arena_bytes;

    // THE REFUSAL THIS MECHANISM EXISTS TO BE ABLE TO MAKE.
    //
    // `device_arena_bytes < offloaded_bytes` was just proved above, and the arena is
    // cyclic -- so it holds `arena_layers` strips while `layers.size()` layers were
    // taken out of the device weight arena. At any instant
    // `layers.size() - arena_layers` offloaded layers have a strip that belongs to a
    // DIFFERENT layer. A reader that reaches such a layer without going through
    // note_layer() therefore computes with another layer's weights: a wrong answer,
    // silently attributed to the model. That is the one failure mode W13 forbids.
    //
    // Measured on this tree (RTX 5090 D, 31.8 GiB; qwen3_8_27b_nvfp4.ninfer;
    // engine sha256 eedfb8aa...): `--weight-host-bytes 6442450944
    // --weight-device-arena-bytes 1073741824` STARTED, exited rc=0, and printed 64
    // generated ids that were all 0, against the resident arm's 59 real ids -- i.e.
    // the silent case, not a refusal. (That run's plan: 20 layers, arena_layers=2,
    // stride=382853120 B, offloaded 6199873632 B.)
    //
    // So the plan refuses unless the caller asserts the contract. Nothing is
    // approximated and nothing is dropped: the run stops, naming the layer range,
    // the arena, and the hook that is missing.
    if (!limits.fetch_per_layer_entry) {
        const std::uint64_t stale = plan.layers.size() > plan.arena_layers
                                        ? plan.layers.size() - plan.arena_layers
                                        : 0;
        // (notehook) THE MESSAGE MOVED WITH THE MECHANISM. It used to say the hook "runs in the
        // Prefill phase only"; that is no longer true of this tree -- run_layers now enters
        // every offloaded layer through note_layer() in every phase that is not CUDA-graph
        // captured, and the H2D is joined to the consuming stream by an event. What is still
        // missing when this fires is the CALLER'S ASSERTION: a front end that has not read this
        // contract must say so before it plans an offload, because the plan cannot see whether
        // the run it belongs to ever calls the hook.
        throw std::invalid_argument(
            "weight offload cannot be numerically transparent for these layers: L" +
            std::to_string(plan.layers.front()) + "..L" + std::to_string(plan.layers.back()) +
            " (" + std::to_string(plan.layers.size()) +
            " layers) are offloaded into a cyclic arena of " + std::to_string(plan.arena_layers) +
            " strip(s), so " + std::to_string(stale) +
            " of them have no strip of their own outside the pass that fetched them, and the "
            "caller did NOT assert the per-layer residency contract. The mechanism this plan "
            "needs IS landed on this tree (note_layer() in every phase that is not CUDA-graph "
            "captured, with the H2D joined to the consuming stream by an event -- see "
            "WeightResidencyRuntime::note_layer and WeightResidencyDevice's event quartet), so "
            "what is missing is the ASSERTION, not the fetch: set "
            "WeightOffloadLimits::fetch_per_layer_entry from the run's own phase structure "
            "(src/targets/registry.cpp does), or clear --weight-host-bytes to run with every "
            "weight resident");
    }
    if (limits.decode_graph_captured && plan.arena_layers < plan.layers.size() &&
        !capturable_fetch_requested()) {
        // The arena rotates, and a captured decode replays the slot addresses it was captured
        // with. REFUSED rather than run, because the wrong answer would be silent and
        // attributed to the model -- the one failure mode W13 exists to make impossible.
        throw std::invalid_argument(
            "weight offload cannot be numerically transparent under CUDA-graph capture: L" +
            std::to_string(plan.layers.front()) + "..L" + std::to_string(plan.layers.back()) +
            " (" + std::to_string(plan.layers.size()) +
            " layers) are offloaded into a cyclic arena of " + std::to_string(plan.arena_layers) +
            " strip(s), so the arena ROTATES, and a captured decode graph replays the slot "
            "addresses it was captured with while nothing re-fetches between replays -- the "
            "offloaded layers would be read stale, silently. Re-run with --no-cuda-graph (the "
            "layer walk is then launched eagerly and note_layer() runs on every pass), or set "
            "NINFER_W13_CAPTURABLE_FETCH=1 to use the capturable fetch (the H2D on the offload's "
            "own stream, forked into the capture by fetch_enroll() so each replay re-fills the "
            "arena), or clear --weight-host-bytes to run with every weight resident");
    }
    return plan;
}

// ---------------------------------------------------------------------------
// The device/host backend. Abstract so the whole schedule is testable on the CPU
// with plain g++: the engine injects the CUDA implementation (materializer.cpp),
// the unit test injects a counting one. One call per primitive is deliberate --
// the runtime owns WHEN a copy is enqueued, because that is the design.
// ---------------------------------------------------------------------------
class WeightResidencyDevice {
public:
    virtual ~WeightResidencyDevice() = default;
    virtual void* device_alloc(std::uint64_t bytes)       = 0;
    virtual void device_free(void* slot) noexcept         = 0;
    virtual void* pinned_alloc(std::uint64_t bytes)       = 0;
    virtual void pinned_free(void* pinned) noexcept       = 0;
    virtual void enqueue_h2d(void* device_slot, const void* pinned, std::uint64_t bytes) = 0;
    virtual void synchronize()                            = 0;
    virtual std::uint64_t now_ns() const noexcept         = 0;
    // THE JOIN (notehook). The H2D above is issued on the TRANSFER stream, and before this the
    // consuming stream had no edge to it at all: note_layer() enqueued the copy and returned,
    // so a layer's GEMMs could be submitted while its bytes were still in flight. That is a
    // silent-precision hazard of the same kind W13's refusal is about, and it was present on
    // the PREFILL path too -- the path that already ran.
    //
    // The shape is an EVENT, not a synchronize(): fetch_event_record() after a layer's spans
    // are enqueued, consumer_wait() on the consuming stream before that layer's weights are
    // read. Both are stream operations and neither is a host wait, so the prefetch overlap the
    // arena exists for is preserved -- the wait on layer L is free once the copy issued
    // `arena_layers - 1` layers earlier has landed.
    //
    // PURE VIRTUAL ON PURPOSE. A defaulted no-op would let a backend -- and therefore a front
    // end -- be accepted and silently unjoined, which is the one outcome this project forbids.
    virtual void* fetch_event_create()                      = 0;
    virtual void  fetch_event_record(void* event)           = 0;
    virtual void  consumer_wait(void* event)                = 0;
    virtual void  fetch_event_destroy(void* event) noexcept = 0;
    // THE ENROLMENT (F1059/streamfix), ONE CALL PER LAYER, issued BEFORE that layer's spans are
    // enqueued. Its whole job is to make the fetch legal and visible when the CONSUMING stream is
    // under `cudaStreamBeginCapture`: the backend forks its own fetch stream into that capture --
    // publish the consumer's position with an event, make the fetch stream wait on it -- and
    // `cudaStreamWaitEvent` PROPAGATES the capture to the stream it waits from, so every copy
    // enqueued afterwards becomes a node of the captured graph. Without it a HOST-side fetch is
    // simply not in the graph, which is why notehook had to refuse a rotating arena under capture.
    //
    // A DEFAULTED NO-OP would be the same failure mode the quartet above refuses by being pure
    // virtual: a backend accepted and silently unenrolled. It is pure virtual for that reason.
    //
    // UNCONDITIONAL, and that is load-bearing rather than tidy: the wait it contributes is what
    // orders the prefetch's WRITE against the consuming stream's READ of the strip it overwrites
    // (slot(index + arena_layers - 1) == slot(index - 1)), and that hazard exists with no graph
    // anywhere in the picture. Measured: a private fetch stream WITHOUT this fork left the
    // offloaded arm byte-identical to the broken one.
    virtual void fetch_enroll()                             = 0;
};

// Greppable counters, PleForensics style (NINFER_W13_STATS=1 prints them).
struct WeightResidencyCounters {
    std::uint64_t layers_entered    = 0;
    std::uint64_t prefetches_issued = 0;
    std::uint64_t prefetch_hits     = 0; // layer was already resident when entered
    std::uint64_t faults            = 0; // prefetch did not land -> synchronous fetch
    std::uint64_t h2d_bytes         = 0;
    std::uint64_t stall_ns          = 0; // time spent inside a synchronous fault
    std::uint64_t layer_joins       = 0; // note_layer() entries that joined consumer<-fetch

    [[nodiscard]] std::uint64_t h2d_bytes_per_token(std::uint64_t tokens) const noexcept {
        return tokens == 0 ? 0 : h2d_bytes / tokens;
    }
    [[nodiscard]] std::string describe() const {
        return "[weight-offload] layers=" + std::to_string(layers_entered) +
               " prefetch=" + std::to_string(prefetches_issued) + " hits=" +
               std::to_string(prefetch_hits) + " faults=" + std::to_string(faults) +
               " joins=" + std::to_string(layer_joins) +
               " h2d=" + std::to_string(h2d_bytes) + " B stall=" + std::to_string(stall_ns) +
               " ns";
    }
};

// The runtime. One instance per loaded artifact, owned by MaterializedArtifact.
//
// Fetch schedule: note_layer(L) is the only entry point. On it the runtime
//   1. verifies layer L is resident (a prefetch scheduled `arena_layers - 1`
//      layers earlier should have covered it; a miss is counted as a fault and
//      fetched now, which is a bandwidth event and never a precision event),
//   2. issues the prefetch for layer L + arena_layers - 1, whose arena slot is
//      exactly the one that layer is about to vacate.
// So the arena is a sliding window over the layer order, and every H2D is issued
// at least `arena_layers - 1` layers before it is needed.
class WeightResidencyRuntime {
public:
    WeightResidencyRuntime(WeightOffloadPlan plan, WeightResidencyDevice* device)
        : plan_(std::move(plan)), device_(device) {
        if (device_ == nullptr) { throw std::invalid_argument("weight residency needs a device"); }
        if (plan_.empty()) { return; }
        arena_ = device_->device_alloc(plan_.device_arena_bytes);
        if (arena_ == nullptr) {
            throw std::runtime_error("weight offload arena allocation failed");
        }
        resident_.assign(plan_.layers.size(), false);
        slot_.resize(plan_.layers.size());
        for (std::size_t i = 0; i < plan_.layers.size(); ++i) {
            slot_[i] = static_cast<std::byte*>(arena_) +
                       static_cast<std::uint64_t>(plan_.arena_slot_of(plan_.layers[i])) *
                           plan_.layer_stride;
        }
        pinned_.assign(plan_.spans.size(), nullptr);
        // One event per offloaded layer, created HERE so the hot path never allocates. An
        // allocation failure is cleaned up rather than thrown through: the arena is held by a
        // raw pointer and this constructor's failure means no destructor runs.
        fetch_events_.assign(plan_.layers.size(), nullptr);
        for (std::size_t i = 0; i < fetch_events_.size(); ++i) {
            fetch_events_[i] = device_->fetch_event_create();
            if (fetch_events_[i] == nullptr) {
                for (void* event : fetch_events_) {
                    if (event != nullptr) { device_->fetch_event_destroy(event); }
                }
                device_->device_free(arena_);
                arena_ = nullptr;
                throw std::runtime_error("weight offload fetch-event allocation failed");
            }
        }
    }

    ~WeightResidencyRuntime() {
        // NINFER_W13_STATS=1: the counters below had NO reader anywhere in the tree.
        // docs/cli.md names this variable and WeightResidencyCounters' own comment
        // promises it, but `git grep --untracked NINFER_W13` was two hits and both were
        // comments -- no getenv. Teardown is the one point every front end reaches, so
        // the line lands here instead of in three callers; and because a run that
        // offloaded nothing still prints, the line answers "did the knob do anything at
        // all" as well as "how often did we fault in steady state".
        if (const char* raw = std::getenv("NINFER_W13_STATS");
            raw != nullptr && *raw != '\0' && *raw != '0') {
            std::fprintf(stderr, "%s\n", counters_.describe().c_str());
        }
        for (void* event : fetch_events_) {
            if (event != nullptr) { device_->fetch_event_destroy(event); }
        }
        for (void* block : pinned_) {
            if (block != nullptr) { device_->pinned_free(block); }
        }
        if (arena_ != nullptr) { device_->device_free(arena_); }
    }

    WeightResidencyRuntime(const WeightResidencyRuntime&)            = delete;
    WeightResidencyRuntime& operator=(const WeightResidencyRuntime&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return !plan_.empty(); }
    [[nodiscard]] const WeightOffloadPlan& plan() const noexcept { return plan_; }
    [[nodiscard]] const WeightResidencyCounters& counters() const noexcept { return counters_; }
    [[nodiscard]] const void* device_arena_base() const noexcept { return arena_; }

    // Binds one offloaded span to its pinned mirror (copied once, here) and returns
    // the STABLE device address the object must be materialized at. Called once
    // per span by the load path, before any note_layer().
    void* adopt_span(std::size_t span_index, const void* host_source_bytes) {
        if (span_index >= plan_.spans.size()) {
            throw std::out_of_range("weight offload span index out of range");
        }
        if (pinned_[span_index] != nullptr) {
            throw std::logic_error("weight offload span adopted twice");
        }
        if (host_source_bytes == nullptr) {
            throw std::invalid_argument("weight offload span has no host source");
        }
        const WeightSpan& span = plan_.spans[span_index];
        void* block            = device_->pinned_alloc(span.bytes);
        if (block == nullptr) {
            throw std::runtime_error("weight offload pinned allocation failed for " + span.label());
        }
        std::memcpy(block, host_source_bytes, static_cast<std::size_t>(span.bytes));
        pinned_[span_index] = block;
        return device_address(span_index);
    }

    // The device address a span is bound to. Valid BEFORE any fault and never
    // changes: this is what makes the offload invisible to the GEMMs and to the
    // captured decode graph.
    [[nodiscard]] void* device_address(std::size_t span_index) const {
        if (arena_ == nullptr) { throw std::logic_error("weight offload is not enabled"); }
        return static_cast<std::byte*>(arena_) + plan_.device_offset_of(span_index);
    }

    // The one hot-path entry point. Cheap in steady state: an index computation and
    // one bool read on the host; no device call, no lock, no allocation.
    void note_layer(std::uint32_t layer) {
        if (plan_.empty()) { return; }
        ++counters_.layers_entered;
        if (!plan_.owns_layer(layer)) { return; }
        const std::uint32_t index = plan_.layer_index(layer);
        if (resident_[index]) {
            ++counters_.prefetch_hits;
        } else {
            const std::uint64_t start = device_->now_ns();
            fetch_layer(index);
            counters_.stall_ns += device_->now_ns() - start;
            ++counters_.faults;
        }
        // THE JOIN (notehook). The consuming stream waits on the event recorded when THIS
        // layer's spans were last enqueued -- whether that was a moment ago (a fault) or
        // `arena_layers - 1` layers ago (the ordinary prefetch hit). Waiting on an event that
        // has already fired is free, so this costs one record/wait pair per layer and closes the
        // race the old code left open: `resident_` is set at ENQUEUE time, not at completion
        // time, so without this wait a prefetched layer could be computed while still in flight.
        device_->consumer_wait(fetch_events_[index]);
        ++counters_.layer_joins;
        const std::size_t ahead = static_cast<std::size_t>(index) + plan_.arena_layers - 1;
        if (ahead < plan_.layers.size() && !resident_[ahead]) { fetch_layer(ahead); }
    }

    // Bytes that will cross PCIe for one full pass over the layer order.
    [[nodiscard]] std::uint64_t bytes_per_round() const noexcept { return plan_.offloaded_bytes; }

    // Pinned host mirror actually held (the number the operator must be able to
    // afford in RAM; it is unswappable).
    [[nodiscard]] std::uint64_t pinned_bytes() const noexcept { return plan_.offloaded_bytes; }

private:
    void fetch_layer(std::size_t index) {
        // The arena is cyclic, so a fetch into this slot EVICTS whichever layer was
        // here before. Forgetting that would leave a stale "resident" bit and the next
        // pass over that layer would read the wrong layer's bytes -- a silent
        // precision bug, which is the one failure mode W13 must never have.
        const std::uint32_t slot = plan_.arena_slot_of(plan_.layers[index]);
        for (std::size_t other = 0; other < plan_.layers.size(); ++other) {
            if (other == index) { continue; }
            if (plan_.arena_slot_of(plan_.layers[other]) == slot) { resident_[other] = false; }
        }
        // F1059: BEFORE the first span is enqueued, and the order matters twice over: the copies
        // and their record have to be INSIDE the capture when one is running (a fork issued after
        // them would enrol the stream too late to capture them), and in eager mode the wait is
        // what keeps this copy out of the strip the consumer is still reading.
        device_->fetch_enroll();
        std::size_t slots = 0;
        for (std::size_t s = 0; s < plan_.spans.size(); ++s) {
            if (plan_.spans[s].layer != plan_.layers[index]) { continue; }
            device_->enqueue_h2d(slot_[index] + plan_.layer_offsets[s], pinned_[s],
                                 plan_.spans[s].bytes);
            counters_.h2d_bytes += plan_.spans[s].bytes;
            ++slots;
        }
        if (slots == 0) { throw std::logic_error("weight offload layer has no spans"); }
        // Record ONCE, after EVERY span of this layer has been enqueued on the transfer stream:
        // the event is what the consuming stream waits on in note_layer(), so it has to cover
        // the whole layer and not just its first span.
        device_->fetch_event_record(fetch_events_[index]);
        resident_[index] = true;
        ++counters_.prefetches_issued;
    }

    WeightOffloadPlan plan_;
    WeightResidencyDevice* device_ = nullptr;
    void* arena_                   = nullptr;
    std::vector<std::byte*> slot_;
    std::vector<bool> resident_;
    std::vector<void*> pinned_;
    std::vector<void*> fetch_events_; // one per offloaded layer, the join in note_layer()
    WeightResidencyCounters counters_;
};

} // namespace ninfer::product
