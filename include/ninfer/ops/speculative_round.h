#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Caller-owned transient capacity for every draft-count and batch-size pair in the inclusive
// domains. token_domain is the fixed sampling profile; invalid domains throw.
[[nodiscard]] std::size_t speculative_accept_greedy_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch);

/**
 * Op: speculative_prepare_verify_inputs
 *
 * Math / indexing:
 *   For row b and 0<=j<=K:
 *     verify_ids[j,b] = anchors[b]                         when j=0
 *                       drafts[j-1,b]                     when 0<j<=Pcur[b]
 *                       anchors[b]                        otherwise;
 *     positions[j,b]  = base_positions[b] + min(j,Pcur[b]).
 *
 * Logical shapes:
 *   All tensors are contiguous I32. anchors/base_positions/current_extents are [B], drafts is
 *   [K,B] with K>=1 and B>=1, and verify_ids/positions are [K+1,B]. Each current extent is in
 *   [0,K]. Inputs and outputs do not overlap.
 *
 * Effects:
 *   Writes every physical output element, including safe invalid-tail values. Inputs remain
 *   unchanged.
 *
 * Workspace:
 *   None.
 */
void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions, const Tensor& current_extents,
                                       Tensor& verify_ids, Tensor& positions, cudaStream_t stream);

/**
 * Prepare only the target verification ids when the caller already owns the matching position
 * matrix. Shapes and id semantics are identical to speculative_prepare_verify_inputs; the
 * existing positions remain untouched.
 */
void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    cudaStream_t stream);

/**
 * Op: speculative_accept_greedy_drafts
 *
 * Algorithm:
 *   Independently for each row b, greedy mode accepts the longest available draft prefix matching
 *   the per-column penalty-adjusted argmax and commits that argmax at the first mismatch (or the
 *   bonus column). With both penalties disabled and no allowed_tokens mask, target_tokens is
 *   the exact raw-logit fast path. A mask constrains all valid columns before selection.
 *   Sampling mode applies configs[b] to each valid verification column, accepts draft i with
 *   target probability p_i(draft_i), samples from the residual distribution on first rejection,
 *   and samples a bonus from column Pcur[b] when every available draft is accepted. The draft
 *   proposal distribution is one-hot at each greedy draft token; when `draft_ids`/`draft_probs`
 *   are non-empty (I32/FP32 [16,K,B], the DFlash2 selector's per-step top-K candidates and
 *   softmax scores) the accept test becomes u < min(1, p_i/q_i) with q_i the draft's own
 *   proposal mass, and a rejection resamples from max(p - q, 0).
 *   RNG domains are the speculative accept/correction/bonus SamplePurpose values and logical
 *   positions derived from the old length.
 *
 *   TREE ROUND (--draft-tree L,d with L > 1: `column_masks` is a non-empty I64 [K+1,B] whose
 *   column j lists the columns column j descends from, bit 0 being the anchor column). The greedy
 *   test is the SAME test read off the parent column: node column j (1..extent) is accepted iff
 *   its own draft token drafts[j-1] equals the verifier's argmax at its parent column and its
 *   parent is accepted (a depth-1 node's parent is column 0, the committed anchor). A node's depth
 *   is popcount(mask) - 1, so the accepted count is the deepest accepted node's depth and
 *   accepted_columns[b] is that node's COLUMN. licensed_tokens holds the accepted CHAIN's own
 *   tokens in depth order, so licensed_tokens[0..A-1] is the chain and licensed_tokens[A] the
 *   accepted column's argmax. With the chain spelling of the mask, (1 << (j+1)) - 1, every one of
 *   these equals the rule above value for value.
 *   `accepted_columns` is OPTIONAL and is written by the greedy path whenever the round publishes
 *   a column, i.e. whenever `column_masks` is bound (a tree round), where it is the deepest
 *   accepted node's column. For a chain round the value would be the accepted count A -- exactly
 *   the column the current code already selects the continuation hidden from -- and a chain caller
 *   may leave the argument unbound, in which case the Op writes nothing: a tree round can gate on
 *   `column_masks` being non-empty and use it while every --draft-tokens round keeps the tensor it
 *   uses today (the sampling and penalty routes keep writing `accepted` only, and no tree round can
 *   reach them). A BOUND `accepted_columns` is I32 [B], and it is REQUIRED to be bound on a tree
 *   round -- the only readers of the column are tree-gated, so the check lives in that gate.
 *   A tree round is greedy-only: the sampling and penalty routes keep their own definitions and
 *   refuse `column_masks` (the runtime rejects a tree round on those routes before it launches).
 *
 * Logical shapes:
 *   All Tensor storage is contiguous. target_tokens/licensed_tokens are I32 [K+1,B], drafts is
 *   I32 [K,B], logits is BF16 [physical_rows,K+1,B], and current_extents/lengths/anchors/
 *   licensed_counts/accepted are I32 [B]. token_domain is in [1,physical_rows], K>=1, B>=1, and
 *   configs points to a device-resident SamplingConfig[B]. Tensor arguments, configs, and
 *   configs[b].token_counts do not overlap except for the explicitly mutated objects.
 *
 * Numeric:
 *   Sampling masks, filtering, penalties, normalization, and RNG semantics are those of sampling.h.
 *
 * Effects:
 *   For each row, let A be the accepted draft count and L=A+1. licensed_tokens[0:A,b] receives
 *   accepted drafts, licensed_tokens[A,b] receives the correction/bonus token, and the remaining
 *   physical slots are zero. licensed_counts[b]=L; accepted[b]=A; anchors[b] becomes the
 *   correction/bonus token; lengths[b]+=L. In every mode, each produced token increments
 *   configs[b].token_counts when that pointer is non-null. current_extents and all other inputs
 *   remain unchanged. Request statistics are
 *   deliberately outside this Op.
 *
 * Workspace:
 *   Caller-owned transient storage reported by
 *   speculative_accept_greedy_drafts_workspace_capacity_bytes().
 */
void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& current_extents,
                                      const Tensor& column_masks, Tensor& lengths, Tensor& anchors,
                                      Tensor& licensed_tokens, Tensor& licensed_counts,
                                      Tensor& accepted, Tensor& accepted_columns,
                                      std::int32_t token_domain, const SamplingConfig* configs,
                                      const Tensor& draft_ids, const Tensor& draft_probs,
                                      WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: speculative_select_accepted_hidden
 *
 * Math / indexing:
 *   out[:,b] = hidden[:,selectors[b],b].
 *
 * Shape / numeric / effects:
 *   hidden is contiguous BF16 [D,T,B], selectors is contiguous I32 [B] with every value in [0,T),
 *   and out is distinct contiguous BF16 [D,B]. The Op exactly copies BF16 bits, writes all of out,
 *   and uses no workspace or other state.
 */
void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        cudaStream_t stream);

/**
 * Op: proposal_remap_token_ids
 *
 * Math / indexing:
 *   proposal_tokens[i]' = id_map[proposal_tokens[i]] for every proposal token.
 *
 * Effects:
 *   Updates the contiguous non-empty I32 proposal_tokens vector in place; every input id is in
 *   [0,count), and id_map is a distinct device I32 array [count]. There is no workspace or other
 *   state side effect.
 */
void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map,
                              std::int32_t count, cudaStream_t stream);

} // namespace ninfer::ops
