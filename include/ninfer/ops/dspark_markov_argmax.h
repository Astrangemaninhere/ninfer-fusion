#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: DSpark sequential Markov-head argmax with SVIP dynamic extent
 *
 * `logits` is contiguous BF16 [V,K*B]: the base draft logits of K proposal
 * positions for B batch rows, with token columns grouped per batch row.
 * `anchors` is contiguous I32 [B] and holds the previous target token per row.
 * `markov_w1` and `markov_w2` are BF16 [V,256] weights. `drafts` is contiguous
 * I32 [K*B] and is completely overwritten for positions up to the SVIP extent.
 * `best_value` and `best_index` are caller-owned contiguous I32 [B] scratch
 * buffers. `extents` is contiguous I32 [B]; it must enter with every element
 * equal to the host budget (1..K) and receives min(budget, j+1) for the first
 * position whose base-logit entropy H satisfies H > threshold^2 (the budget
 * when none stops).
 *
 * For each batch row b and each position j in order, the previous token p
 * (the anchor for j=0, otherwise drafts[j-1]) defines the transition bias
 *
 *   bias(v) = sum_r markov_w1[p,r] * markov_w2[v,r],
 *
 * and drafts[b*K+j] = argmax_v(logits[b*K+j, v] + bias(v)). After position j
 * is sampled, the base-logit entropy of the next distribution is evaluated and
 * the row stops early when sqrt(entropy) > threshold. Positions beyond the
 * extent are not evaluated. The registered domain is V=248320, K=1..7,
 * B=1..8. Inputs and weights are unchanged; best_value and extents are
 * overwritten and the Op owns no workspace or persistent state.
 */
void dspark_markov_argmax(const Tensor& logits, const Weight& markov_w1,
                          const Weight& markov_w2, const Tensor& anchors, Tensor& drafts,
                          Tensor& best_value, Tensor& best_index, Tensor& extents,
                          float entropy_threshold, cudaStream_t stream);

} // namespace ninfer::ops
