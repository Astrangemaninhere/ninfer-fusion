#pragma once

#include "core/tensor.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: mtp_prepare_next_round
 *
 * Math / indexing:
 *   For T=K+1 and each row b with A=accepted[b], L=licensed_counts[b]:
 *     alignment_ids[j,b] = verify_ids[j+1,b]  for 0<=j<A
 *                          next_anchors[b]     otherwise;
 *     remaining_after = max(remaining_budgets[b]-L,0);
 *     context_after   = max(max_context-updated_frontiers[b]-1,0);
 *     next_extents[b] = min(K,max(remaining_after-1,0),context_after);
 *     For S=max(K-1,1) and 0<=s<S:
 *       ar_positions[b,s]      = updated_frontiers[b]+s;
 *       ar_rope_positions[b,s] = ar_positions[b,s]+rope_deltas[b];
 *       ar_valid_columns[b,s]  = (s+1 < next_extents[b]).
 *
 * Logical shapes / effects:
 *   verify_ids/alignment_ids are distinct contiguous I32 [K+1,B]. ar_positions,
 *   ar_rope_positions, and ar_valid_columns are I32 [B,max(K-1,1)] with contiguous rows and one
 *   shared step stride at least B; this permits an exact-B prefix of a fixed-capacity frame. All
 *   other tensors are contiguous I32 [B]. B>=1, 1<=K<=5, 0<=accepted[b]<=K,
 *   licensed_counts[b]=accepted[b]+1, updated_frontiers and remaining_budgets are non-negative,
 *   and max_context is positive. The Op writes every output slot, including safe invalid-tail
 *   values. Inputs remain unchanged. No workspace or other state is used.
 */
// Adaptive draft-window cap (auto mode): the window grows while the live window is fully
// accepted and collapses just past the last accepted draft otherwise.
void mtp_adaptive_extents(const Tensor& accepted, const Tensor& current_extents, Tensor& cuts,
                          std::int32_t k_max, cudaStream_t stream);

void mtp_prepare_next_round(const Tensor& verify_ids, const Tensor& next_anchors,
                            const Tensor& accepted, const Tensor& updated_frontiers,
                            const Tensor& remaining_budgets, const Tensor& licensed_counts,
                            const Tensor& rope_deltas, Tensor& alignment_ids, Tensor& next_extents,
                            Tensor& ar_positions, Tensor& ar_rope_positions,
                            Tensor& ar_valid_columns, std::int32_t max_context,
                            cudaStream_t stream, const Tensor* svip_cuts = nullptr);

// mtp_tree_commit_history's per-row diagnostics, written into `commit_flags`. Every non-zero bit
// says the TREE METADATA the draft side published for that row is inconsistent, so the round's
// accepted chain cannot be trusted: the runtime turns a non-zero word into a hard error instead of
// committing a history that came from a guess. Zero is the only accepted value.
//
//   kMtpTreeFlagColumnOutOfRange : accepted_columns[b] is not a column of this frame
//   kMtpTreeFlagMaskMissingSelf  : column_masks[c] does not have its own bit c set
//   kMtpTreeFlagDepthMismatch    : popcount(column_masks[c]) - 1 != column_depths[c]
//   kMtpTreeFlagChainOrder       : the accepted chain's deepest column is not c
//   kMtpTreeFlagPositionOutOfRange: frontier + A is outside the mapped logical cache
inline constexpr std::int32_t kMtpTreeFlagColumnOutOfRange  = 1 << 0;
inline constexpr std::int32_t kMtpTreeFlagMaskMissingSelf   = 1 << 1;
inline constexpr std::int32_t kMtpTreeFlagDepthMismatch     = 1 << 2;
inline constexpr std::int32_t kMtpTreeFlagChainOrder        = 1 << 3;
inline constexpr std::int32_t kMtpTreeFlagPositionOutOfRange = 1 << 4;

// The verify width a tree round can carry, and with it the ancestor-mask width of
// qwen3_6::kMtpDecodeMaximumWidth. The Op asserts it instead of overrunning a chain buffer.
inline constexpr std::int32_t kMtpTreeMaximumWidth = 16;

/**
 * Op: mtp_tree_commit_history
 *
 * TREE ROUNDS ONLY (--draft-tree L,d with L > 1). Moves the accepted chain's K/V to the positions
 * the NEXT round reads as its history, and publishes the column map it used.
 *
 * Why it is required. A verify column's K/V lands at ITS OWN cache position: the bf16 small-T
 * kernel writes cache_k/cache_v at gqa_cache_index(..., p_tok) with p_tok = pos[column] =
 * frontier + column (src/ops/kernel/gqa_attention_decode_bf16.cuh:173-193), and the round's
 * positions come from speculative_prepare_verify_inputs (base_positions + min(j, Pcur)). The
 * accepted node of a TREE round is an arbitrary deep column c, not the A-th column, while the next
 * round reads its history as the position PREFIX frontier+1..frontier+A. Without this Op the next
 * round therefore reads the K/V of rejected branches and siblings -- and nothing throws, because
 * every one of those positions was written by this round itself (it is a silent wrong answer, not
 * a missing one). For a chain the accepted set IS the prefix, which is why --draft-tokens has never
 * needed this step.
 *
 * Math / indexing. With c = accepted_columns[b] and C = the set bit positions of
 * column_masks[c, b] (bit 0 is the anchor column and is always set; every other bit is an accepted
 * node), C in ASCENDING column order is the accepted chain in depth order, because a node's
 * ancestors all have a smaller column index (include/ninfer/ops/dflash2_ddtree.h:56-64). So
 *
 *   chain_sources[i, b] = the (i+1)-th smallest set bit of column_masks[c, b],  i = 0..A-1,
 *   A = popcount(column_masks[c, b]) - 1  (equivalently column_depths[c, b])
 *
 * and every other chain_sources slot is -1. For each i with chain_sources[i, b] = s the K/V row of
 * logical position frontier+s is copied, bf16 bit-for-bit, to position frontier+i+1 (the node at
 * depth i+1). s > i+1 always holds, so applying the moves in ascending i is safe in place: a write
 * can only land on a source an earlier step already read.
 *
 * L = 1 (--draft-tokens) degeneracy: the mask is the prefix (1 << (j+1)) - 1, so
 * chain_sources[i, b] == i+1, every copy is position-to-itself and ZERO bytes move; the runtime
 * does not even call the Op on a chain round.
 *
 * Effects. Overwrites only logical positions frontier+1..frontier+A of the given layer's K/V
 * planes; every other position keeps its bytes. Publishes chain_sources [W,B] and, per row, the
 * kMtpTreeFlag* word. Inputs are unchanged, and the Op decides no frontier and keeps no state.
 *
 * Logical shapes. column_masks I64 [W,B] and column_depths I32 [W,B] (the round's ingress), then
 * accepted_columns / base_frontiers / table_rows / commit_flags I32 [B] and chain_sources I32
 * [W,B]; W = the round's verify width with 2 <= W <= kMtpTreeMaximumWidth, B >= 1. `cache` is one
 * layer's BF16 paged batch view; block_tables is addressed as
 * block_tables + table_rows[b] * block_tables.ne[0].
 */
void mtp_tree_commit_history(const Tensor& column_masks, const Tensor& column_depths,
                             const Tensor& accepted_columns, const Tensor& base_frontiers,
                             const Tensor& table_rows, PagedKVBatchLayerView cache,
                             Tensor& chain_sources, Tensor& commit_flags, cudaStream_t stream);

/**
 * Op: mtp_svip_entropy_extents
 *
 * SVIP self-verification cap. logits are the BF16 target verify logits shaped
 * [vocab, cols, batch]; accepted is I32 [batch]. For each row the kernel finds
 * the first verify column c > accepted[row] whose softmax entropy (nats)
 * exceeds `threshold` and atomically writes cuts[row] = c - accepted[row] - 1
 * (initialized by the caller to the maximum draft count). Passing the result as
 * svip_cuts to mtp_prepare_next_round caps the next round's draft extent.
 */
void mtp_svip_entropy_extents(const Tensor& logits, const Tensor& accepted, Tensor& cuts,
                              float threshold, cudaStream_t stream);

/**
 * Op: mtp_draft_align_hidden
 *
 * TREE ROUNDS ONLY (--draft-tree L,d with L > 1). Re-indexes the verify hidden from VERIFY COLUMN
 * order to ACCEPTED-CHAIN DEPTH order, so that the MTP head can be driven by the chain the round
 * actually committed.
 *
 * Why it is required. mtp_prepare_next_round publishes `alignment_ids[j] = verify_ids[j+1]`, and
 * mtp_forward_decode_batch pairs index j with `hidden[:,j]` at cache position `positions[j]`. That
 * pairing is depth-correct only while the accepted nodes are the COLUMN PREFIX -- which is what a
 * chain round has and what a tree round does NOT have. A tree round's accepted chain sits at the
 * arbitrary columns published as chain_sources, so index j would carry column j's hidden together
 * with column j+1's token, i.e. the MTP head (and the MTP cache it writes, which the next round
 * reads) would be driven by rejected branches and siblings. Nothing throws: every one of those
 * columns was produced by this round itself.
 *
 * Math / indexing. W = width, B = batch, `chain_sources` I32 [W,B] holding, in row i, the verify
 * COLUMN of the accepted node at depth i+1 (0 <= chain_sources[i,b] < W; every slot beyond
 * A = popcount(mask)-1 is -1, see mtp_tree_commit_history), and `valid_counts` I32 [B] = the
 * round's licensed_counts = A+1:
 *
 *   out[:,j,b] = hidden[:,j,b]                            for j == 0 or j >= valid_counts[b]
 *                hidden[:,chain_sources[j-1,b],b]         for 1 <= j < valid_counts[b]
 *
 * Depth 0 keeps the anchor column (chain_sources has no depth-0 row); depth j > 0 takes the column
 * of the depth-j node. Indices j and j > 0 of a chain are the same node, and column order equals
 * depth order, so a chain is the identity.
 *
 * L = 1 degeneracy: a chain's chain_sources[i,b] == i+1, so out[:,j,b] == hidden[:,j,b] for every j
 * and the Op is a bit-for-bit copy of `hidden`; the runtime does not call it on a chain round at
 * all.
 *
 * Effects / safety. Writes every element of `out`. A chain_sources value outside [0,W) is CLAMPED
 * to column 0 instead of being used as an offset, so a malformed tree table can produce a wrong
 * gather but never an out-of-range read; that table's own validation is mtp_tree_commit_history's
 * kMtpTreeFlag* word, which the runtime turns into a hard error instead of committing a history
 * that came from a guess. `out` must not alias `hidden`, and inputs remain unchanged.
 *
 * Logical shapes. hidden and out are distinct contiguous BF16 [D,W,B] with the same D, W and B;
 * 2 <= W <= kMtpTreeMaximumWidth; B >= 1; chain_sources I32 [W,B]; valid_counts I32 [B].
 * No workspace or other state is used.
 */
void mtp_draft_align_hidden(const Tensor& hidden, const Tensor& chain_sources,
                            const Tensor& valid_counts, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
