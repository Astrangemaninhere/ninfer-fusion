// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : cometkim/ninfer
// Branch      : feat/1m-context
// Commit      : 2cc56b5db39f951c1e91b2caec8b39274d795682
// Source path : src/core/kv_ring_bits.h
// sha256(src) : b1df327d1549d69fcc13f4133447baf0cf9c3081ce805e6706339def598c932b
// Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits (rk4v4-e8 = rotated K 4b, V 4b, rk4v4 in the mix), which is a DIFFERENT axis from src/kvcfg/kv_formats.h's tier vocabulary (hot/tail/cold, Rk4v4).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer {

// One slot row's hq residual-window validity words: [0, kKvRingWords) are the recent-ring bits
// (bit r = ring slot r holds the row the next fetch will name); word kKvRingWords bit 0 marks the
// sink rows as populated for the owning history. Passed by value so callers can hand host-computed
// masks straight to the kernel. kKvRingWords matches kCausalHqRecentKeys / 32 (asserted at the
// owning cache, which sees both definitions).
inline constexpr int kKvRingWords = 16;
inline constexpr int kKvSideWords = kKvRingWords + 1;

struct KvRingWords {
    std::uint32_t w[kKvSideWords] = {};
};

// Applies word masks to one slot row's validity words:
// words[i] = (words[i] & and_mask.w[i]) | or_mask.w[i] for i < word_count.
// Async, word_count threads.
void apply_kv_ring_valid_words(std::uint32_t* words, const KvRingWords& and_mask,
                               const KvRingWords& or_mask, int word_count, cudaStream_t stream);

// Copy every layer's side-plane rows for one slot row to another (prefix-fork inheritance when
// the source row still owns coherent rows). residual_* are the full per-cache planes
// [head_dim, kv_heads, side_rows, layers * table_rows]; slot_stride_bytes is one dim-3 element.
void copy_kv_residual_slot(const void* residual_k, const void* residual_v,
                           std::int64_t slot_stride_bytes, std::int32_t layer_count,
                           std::int32_t table_rows, std::int32_t source_slot,
                           std::int32_t destination_slot, cudaStream_t stream);

} // namespace ninfer
