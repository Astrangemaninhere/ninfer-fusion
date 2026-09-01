#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: DFlash2 candidate selector with a greedy path walk
 *
 * `unary_logits` is contiguous BF16 [V,S*B]: full-vocab base logits of S
 * proposal positions for B batch rows, token-major (each column holds one
 * token's V logits). `projected_hidden` is contiguous BF16 [R,S*B]: the
 * candidate selector's hidden projection of the same proposal columns.
 * `predecessor_codebook` and `successor_codebook` are BF16 [V,R] Weights.
 * `anchors` is contiguous I32 [B] and holds the previous target token per row.
 *
 * `candidates` (I32), `unary` (F32), and `scores` (F32) are caller-owned
 * scratch with shapes [B,S,K], [B,S,K], and [B,S,K,K] respectively; all are
 * completely overwritten. `drafts` is contiguous I32 [S*B] and receives the
 * selected token for every proposal position.
 *
 * Selection: per row/step the top-K base logits (higher value first, lower
 * token id breaking ties) form the candidate set. Each candidate c at step s
 * receives the pair score
 *
 *   score(s,p,c) = unary[s,c]
 *                + sum_r projected[r,s] * predecessor_codebook[pred,r]
 *                                      * successor_codebook[candidate[s,c],r],
 *
 * where pred is the anchor token at s=0 and candidate[s-1,p] afterwards; only
 * p=0 is admitted at s=0. The walk starts at previous candidate index 0 and
 * greedily picks the highest-score current candidate at every step (lowest
 * candidate index breaking ties).
 *
 * The registered domain is V=248320, R=256, S=7, B=1..8, K=16. Inputs and
 * weights are unchanged and the Op owns no workspace or persistent state.
 */
void dflash2_selector(const Tensor& unary_logits, const Tensor& projected_hidden,
                      const Weight& predecessor_codebook, const Weight& successor_codebook,
                      const Tensor& anchors, Tensor& candidates, Tensor& unary, Tensor& scores,
                      Tensor& drafts, std::int32_t steps, std::int32_t top_k, cudaStream_t stream);

} // namespace ninfer::ops
