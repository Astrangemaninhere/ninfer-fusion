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
 *     best[b] = argmax over 0 <= o < starts[b] of
 *                   L(o) = max L <= Q such that
 *                       ids[o+Q-L .. o+Q) == ids[starts[b]+Q-L .. starts[b]+Q)
 *               ties prefer the LATEST occurrence (largest o), and only
 *               matches with L >= min_len and o+Q+K <= len[b] participate.
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

void suffix_lookup(const void* ids, std::int32_t history,
                   const std::int32_t* starts, const std::int32_t* lengths,
                   std::int32_t batch, std::int32_t query, std::int32_t min_len,
                   std::int32_t continuation_tokens, std::int32_t* best_len,
                   std::int32_t* best_offset, std::int32_t* continuation,
                   cudaStream_t stream);

} // namespace ninfer::ops
