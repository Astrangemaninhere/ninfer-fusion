#pragma once

// mtp_proposal_topk: per-token top-L rows of a row-major 2-D bf16 table.
// Raw-pointer API (self-contained; adapter at call sites), the same convention as
// include/ninfer/ops/vocab_topk16.h.
//
// WHY IT EXISTS
//   The MTP proposal head materializes one full proposal row per draft depth and per batch
//   lane and then keeps only its argmax (TextContext::proposal_argmax,
//   src/targets/qwen3_6/impl/runtime/text_context_impl.h:690-710). A `--draft-tree L,d` round
//   (L > 1) needs the L best rows of that same row, because every depth's proposal row becomes
//   L sibling tree nodes (src/targets/qwen3_6/impl/runtime/mtp_tree_produce.h). This op is that
//   extraction and nothing else: it invents no score, it runs no extra forward, and it touches
//   no state -- so an L <= 1 round that never calls it is byte-identical.
//
// SCOPE (honest)
//   It is ADDITIVE and, as landed, NOT YET CALLED: the proposal loop that must call it is the
//   remaining half (see REPORT.md). The closest precedent in this tree is
//   include/ninfer/ops/dflash2_tree_walk.h -- "ADDITIVE and DEFAULT OFF ... Nothing calls it
//   yet" -- landed ahead of its consumer for the same reason.
//
// INPUT
//   logits_bf16  [rows, tokens] row-major: logits_bf16[r * tokens + t] (bf16 bit pattern).
//                For the FULL proposal head `rows` is the text output domain and the returned
//                row index IS the global token id. For the SHORTLIST head `rows` is the
//                shortlist width and the index must be remapped through the shortlist table;
//                the planner refuses a tree round on a shortlist head (layouts_impl.h) rather
//                than let a shortlist index be published as if it were a token id.
//   rows         1 .. (no bound here; the kernel strides)
//   tokens       1 .. (one block per token)
//   top_l        1..16 (== ops::ddtree::kTopK); clamped to `rows` when rows < top_l.
//
// OUTPUT (fully overwritten: every entry, including the tail)
//   ids_out I32 [top_l, tokens]: ids_out[i * tokens + t] is the row index of the (i+1)-th best
//   row for token t, ties broken by the LOWEST row index. When fewer than `top_l` distinct rows
//   exist the tail repeats the last chosen index, so a reader that only wants the first `paths`
//   entries never reads an uninitialised value.
//
// The op owns no workspace and keeps no persistent state.

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

void mtp_proposal_topk(const void* logits_bf16, std::int32_t rows, std::int32_t tokens,
                       std::int32_t top_l, std::int32_t* ids_out, cudaStream_t stream);

} // namespace ninfer::ops
