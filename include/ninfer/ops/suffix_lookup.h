#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/**
 * Suffix lookup (LABD lookup draft stage, first landed kernel).
 *
 * Per batch row b, given the accepted token history ids[0, len[b]) and the
 * query suffix ids[starts[b], starts[b]+Q) (the newest Q tokens), find the
 * longest earlier occurrence of the query tail:
 *
 *     best[b] = argmax over 0 <= o < suffix_lookup_scan_limit(starts[b], len[b], Q, K) of
 *                   L(o) = max L <= Q such that
 *                       ids[o+Q-L .. o+Q) == ids[starts[b]+Q-L .. starts[b]+Q)
 *               ties prefer the LATEST occurrence (largest o), and only
 *               matches with L >= min_len participate.
 *
 * A candidate window must be strictly earlier than the query window and must leave the whole
 * K-token continuation inside the history (2026-09-03 anti-self-match fix: an overlapping
 * window's "continuation" is the query itself, which is not new information). That bound is
 * stated exactly once, in suffix_lookup_scan_limit below; the device kernel, the host
 * reference in src/spec/lookup_fuse.h and both test mirrors scan
 * o in [0, suffix_lookup_scan_limit(...)).
 *
 * Outputs per row: best_len[b] (I32, 0 when nothing matched) and
 * continuation[b][k] = ids[best_offset[b]+Q+k] for k < K (I32, -1 padded when
 * out of range). The engine draft chain reads `continuation` as its proposal
 * tokens; repeated calls with an advancing query form the n-gram chain.
 *
 * `ids` is contiguous I32 [H], `starts/len/best_len` are I32 [B], and
 * `continuation` is contiguous I32 [B,K]. No workspace; no state change.
 */
namespace ninfer::ops {

// The one definition of "which candidate offsets may participate": strictly earlier
// (o + Q < starts[b]) and inside the history with room for the full continuation
// (o + Q + K < len[b]). The kernel scans [0, suffix_lookup_scan_limit(...)); the host
// reference and the mirrors must ask exactly the same question.
[[nodiscard]] __host__ __device__ constexpr std::int32_t
suffix_lookup_scan_limit(std::int32_t start, std::int32_t length, std::int32_t query,
                         std::int32_t continuation_tokens) {
    const std::int32_t by_history = length - query - continuation_tokens;
    const std::int32_t by_start   = start - query;
    return by_history < by_start ? by_history : by_start;
}

void suffix_lookup(const void* ids, std::int32_t history,
                   const std::int32_t* starts, const std::int32_t* lengths,
                   std::int32_t batch, std::int32_t query, std::int32_t min_len,
                   std::int32_t continuation_tokens, std::int32_t* best_len,
                   std::int32_t* best_offset, std::int32_t* continuation,
                   cudaStream_t stream);

} // namespace ninfer::ops
