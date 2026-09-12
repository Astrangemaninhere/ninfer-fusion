#pragma once

// Cold Host tier: the eviction consumer behind `--cold-policy host`.
//
// WHAT IT CONSUMES  device KV pages that no attention kernel can read any more
//                   (`cold_host_page_is_read_free`), one pinned Host extent per
//                   page, bounded by `--cold-host-bytes`.
// WHAT IT PRODUCES  the release of those pages' device replicas (real device
//                   memory back into the pool) and, consequently, a page whose
//                   only replica is on the host.
// HOW IT RESTORES   through the engine's existing Host->Device materialization
//                   path. A page with no device replica and a current Host
//                   replica is exactly the "HostOnly replica" state that
//                   materialization already plans (program_impl.h:
//                   prepare_kv_restores counts `pages.host_resident`) and
//                   executes, so the tier needs no read-back path, no sentinel
//                   block-table entry and no device cold slot.
//
// WHY READ-FREE IS THE GATE, not page age (which is what the slot tiers use):
// the Window/Disk cold pools stay READABLE -- attention decodes their pages
// inline from device cold slots (ops/softmax_attention/dense/causal_cache/
// small_t.cu reads cache.cold_slots + cache.cold_slot_valid), which is why they
// reserve device slots at all. A Host page has no device slot, so it may only be
// evicted when no kernel will ever look at it again. On a full-attention layer
// every committed token is re-read on every decode round, so no page is ever
// read-free; with a sliding window W on every layer, page p (tokens
// [p*P, (p+1)*P)) is unread once (p+1)*P + W <= frontier. A stack with any
// full-attention layer therefore has no read-free page, and the tier must say so
// loudly instead of reserving host memory it can never use -- which is the real
// reason `effective_cold_pages(Host)` returns 0 (layouts_impl.h:124-141): the
// device cold-slot pool is the *slot* tiers' resource, and 0 is the correct
// device-side consequence of a Host tier, not a placeholder.
//
// GRANULARITY: one logical page ACROSS ALL TEXT LAYERS, never a slice of one. A
// page is the block table's addressing unit and the Host replica's unit
// (HostKVPageLayout packs every text layer's planes for one page end to end), and
// the tiers below follow the same rule: a cold-slot sentinel encodes ONE slot base
// and the attention kernels index that slot number in every layer's cold_slots at
// once, so a device page can only be demoted whole-slot x all layers (H1). The
// tier therefore has no per-layer and no per-head eviction notion at all: one page
// is one residency bit, which is exactly the state materialization restores.
//
// The byte budget itself (the cap, the ladder, the page arithmetic) lives in
// product/kv_cold_tier_budget.h; this class only owns the tier's resources and
// its counters, so there is exactly one definition of "full".

#include "core/host_kv_arena.h"
#include "product/kv_cold_tier_budget.h"
#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// The tier's unit is one Paged-KV page, the same unit the slot tiers count (the
// keep-tokens arithmetic in program_impl.h and the DP's cold capacity both use
// it). Named here so the read-free predicate's `page_tokens` argument cannot be
// silently fed a different granularity.
inline constexpr std::uint32_t kColdHostPageTokens = 64;
static_assert(kColdHostPageTokens == static_cast<std::uint32_t>(kPagedKVPageSize),
              "the Cold Host tier must use the Paged KV page as its unit");

// A page is read-free when every layer's sliding window has already moved past
// it: layer L attends [frontier+1-W_L, frontier], so page p is unread by L iff
// (p+1)*P + W_L <= frontier. `layer_windows` holds the first `layers` entries of
// the decoder's per-layer table (the remaining entries of that fixed-size array
// are padding and must not be passed here). A layer with window 0 is full
// attention and reads every committed token, so it makes the predicate false for
// every page. The `+ W` form keeps one token of slack: eviction has to be
// impossible, not merely unlikely.
[[nodiscard]] inline bool
cold_host_page_is_read_free(std::uint32_t page, std::uint32_t page_tokens,
                            std::uint32_t frontier,
                            std::span<const std::uint32_t> layer_windows) noexcept {
    if (page_tokens == 0 || layer_windows.empty()) { return false; }
    const std::uint64_t page_end = (static_cast<std::uint64_t>(page) + 1U) * page_tokens;
    for (const std::uint32_t window : layer_windows) {
        if (window == 0) { return false; }
        if (page_end + window > frontier) { return false; }
    }
    return true;
}

// True when the tier can ever admit a page on this model. A false answer means
// --cold-policy host (and any --cold-host-bytes) is structurally inert, which the
// program reports once at construction.
[[nodiscard]] inline bool
cold_host_layers_are_windowed(std::span<const std::uint32_t> layer_windows) noexcept {
    if (layer_windows.empty()) { return false; }
    return std::all_of(layer_windows.begin(), layer_windows.end(),
                       [](std::uint32_t window) { return window != 0; });
}

class ColdHostTier {
public:
    // `budget.host_bytes` is the tier's whole memory: the arena is created with
    // exactly that capacity, so the cap holds at the allocator as well as in the
    // arithmetic. `layouts` are the page-store layouts this tier serves.
    ColdHostTier(product::ColdTierBudget budget, std::span<const HostKVPageLayout> layouts)
        : budget_(budget) {
        if (!budget_.host_tier_enabled || budget_.host_bytes == 0 || layouts.empty()) { return; }
        std::size_t minimum_stride = layouts.front().page_stride;
        for (const HostKVPageLayout& layout : layouts) {
            minimum_stride = std::min(minimum_stride, layout.page_stride);
        }
        if (minimum_stride == 0) { return; }
        arena_ = std::make_unique<HostKVArena>(budget_.host_bytes, layouts);
        const std::size_t extent_capacity = budget_.host_bytes / minimum_stride;
        if (extent_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Cold Host tier extent capacity exceeds uint32");
        }
        if (extent_capacity == 0) { return; }
        extents_ = std::make_unique<HostKVExtentStore>(
            *arena_, static_cast<std::uint32_t>(extent_capacity));
    }

    // A tier is usable only when it owns both the arena and a non-empty descriptor
    // pool: an enabled policy with a budget that cannot hold one page stays inert
    // (and reports it) rather than failing at the first eviction.
    [[nodiscard]] bool enabled() const noexcept {
        return budget_.host_tier_enabled && extents_ != nullptr;
    }

    [[nodiscard]] HostKVExtentStore* extents() noexcept { return extents_.get(); }

    [[nodiscard]] const product::ColdTierBudget& budget() const noexcept { return budget_; }

    [[nodiscard]] std::uint32_t capacity_pages() const noexcept {
        return product::cold_tier_page_capacity(budget_.host_bytes, budget_.host_page_bytes);
    }

    [[nodiscard]] std::uint32_t used_pages() const noexcept { return usage_.host_pages; }

    [[nodiscard]] product::ColdTierDecision admit() const noexcept {
        return product::admit_cold_page(budget_, usage_);
    }

    void note_admission(product::ColdTier tier) noexcept {
        product::reserve_cold_page(usage_, tier);
        switch (tier) {
        case product::ColdTier::Host: ++counters_.host_admissions; break;
        case product::ColdTier::Disk: ++counters_.disk_admissions; break;
        case product::ColdTier::StayHot: break;
        }
    }

    // The page could not leave the device: counted, never thrown.
    void note_refused() noexcept { ++counters_.stays_hot; }

    // A page's Host replica was dropped (teardown sweep): the bytes are back.
    void note_page_released(std::uint32_t pages) noexcept {
        for (std::uint32_t i = 0; i < pages; ++i) {
            product::release_cold_page(usage_, product::ColdTier::Host);
        }
        counters_.releases += pages;
    }

    [[nodiscard]] const product::ColdTierCounters& counters() const noexcept { return counters_; }

    [[nodiscard]] std::string description() const {
        return product::describe_cold_tier_budget(budget_, capacity_pages(),
                                                  product::cold_tier_page_capacity(
                                                      budget_.disk_bytes, budget_.disk_page_bytes));
    }

    [[nodiscard]] std::string counters_line() const { return counters_.describe(); }

private:
    product::ColdTierBudget budget_{};
    product::ColdTierUsage usage_{};
    product::ColdTierCounters counters_{};
    std::unique_ptr<HostKVArena> arena_;
    std::unique_ptr<HostKVExtentStore> extents_;
};

} // namespace ninfer::targets::qwen3_6::detail
