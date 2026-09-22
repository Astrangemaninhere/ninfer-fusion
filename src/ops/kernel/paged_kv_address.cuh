#pragma once

#include "core/paged_kv_cache.h"
#include "ninfer/ops/softmax_attention.h"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kPagedKVPageShift = 6;
inline constexpr int kPagedKVPageMask  = kPagedKVPageSize - 1;

static_assert(kPagedKVPageSize == (1 << kPagedKVPageShift));

// Maximum logical pages one attention split can span, for a maximum-visible-key envelope W and
// the split count S the launch geometry guarantees:
//     pages_per_split <= ceil(W / (kPagedKVPageSize * S))
// The attention kernels size their shared page-id table from page_ids() and the launchers must
// never dispatch a split that breaks the bound; the two producers of active_split_count are
// causal_attention_split_capacity / causal_small_t_launch_capacity (dense cache) and
// gqa_small_t_active_splits (decode).
//
// kGqaGuaranteedSplitCount is the 85 of CausalAttentionGeometry::SmallTMaximumSplits
// (= 85 * SmallTSplitScale, scale >= 1), the floor both landed worst cases already assume:
//     kCausalAttentionMaximumVisibleKeys     / (64 * 85) ->  49 pages  (dense causal cache)
//     kCausalAttentionMaximumVisibleKeysYarn / (64 * 85) -> 193 pages  (decode kernels)
inline constexpr int kGqaGuaranteedSplitCount = 85;

[[nodiscard]] __host__ __device__ constexpr int paged_kv_split_page_span(std::uint32_t max_visible_keys) {
    const std::uint64_t per_split = static_cast<std::uint64_t>(kPagedKVPageSize) *
                                    static_cast<std::uint64_t>(kGqaGuaranteedSplitCount);
    return static_cast<int>((static_cast<std::uint64_t>(max_visible_keys) + per_split - 1) /
                            per_split);
}

// Rounded up to a power of two on purpose: that is what both kernel families already declare
// (49 -> 64 dense, 193 -> 256 decode), so this reproduces the landed sizes exactly while
// leaving the envelope as the only thing that has to move if the context budget grows.
[[nodiscard]] __host__ __device__ constexpr int paged_kv_page_ids(std::uint32_t max_visible_keys) {
    int bound = 1;
    while (bound < paged_kv_split_page_span(max_visible_keys)) { bound <<= 1; }
    return bound;
}

static_assert(paged_kv_page_ids(kCausalAttentionMaximumVisibleKeys) == 64,
              "dense causal cache page-id table size changed: re-check the page-id fill bound");
static_assert(paged_kv_page_ids(kCausalAttentionMaximumVisibleKeysYarn) == 256,
              "decode page-id table size changed: re-check the page-id fill bound");

__device__ __forceinline__ std::int32_t paged_kv_physical_page(const std::int32_t* block_table,
                                                               std::int32_t position) {
    return block_table[position >> kPagedKVPageShift];
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t paged_kv_page_head_offset(std::int32_t physical_page,
                                                                  std::int32_t head) {
    return static_cast<std::int64_t>(LeadingExtent) * kPagedKVPageSize *
           (static_cast<std::int64_t>(head) +
            static_cast<std::int64_t>(HeadExtent) * physical_page);
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(std::int32_t physical_page, std::int32_t head, std::int32_t page_offset,
                        std::int32_t leading) {
    return paged_kv_page_head_offset<LeadingExtent, HeadExtent>(physical_page, head) +
           static_cast<std::int64_t>(LeadingExtent) * page_offset + leading;
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(const std::int32_t* block_table, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return paged_kv_element_offset<LeadingExtent, HeadExtent>(
        paged_kv_physical_page(block_table, position), head, position & kPagedKVPageMask, leading);
}

} // namespace ninfer::ops
