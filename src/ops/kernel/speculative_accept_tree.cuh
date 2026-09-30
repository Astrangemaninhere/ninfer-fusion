#pragma once

// ============================================================================================
// F904 / line subagentfix -- THE TREE ACCEPT, AS THE OP'S OWN CONTRACT STATES IT, ON THE ROUTE
// THAT DID NOT IMPLEMENT IT.
//
// THE DEFECT THIS FILE EXISTS FOR, AT ITS OWN file:line
//   include/ninfer/ops/speculative_round.h:74-88 defines ONE rule for a tree round:
//     "node column j (1..extent) is accepted iff its own draft token drafts[j-1] equals the
//      verifier's argmax at its PARENT column and its parent is accepted ... A node's depth is
//      popcount(mask) - 1, so the accepted count is the deepest accepted node's depth and
//      accepted_columns[b] is that node's COLUMN ... accepted_columns ... is REQUIRED to be bound
//      on a tree round."
//   The Op has TWO routes. `speculative_accept_greedy_drafts_kernel`
//   (src/ops/kernel/speculative_round.cuh:112-215, the single-block route) branches on
//   `column_masks` and implements exactly that rule. The multiblock route's
//   `speculative_sampling_group_finalize_kernel` (`:460-505`) takes NEITHER `column_masks` NOR
//   `accepted_columns` and its greedy branch runs
//       while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
//   -- the CHAIN prefix test -- over a DEPTH-MAJOR tree frame. On the sibling lattice
//   (src/targets/qwen3_6/impl/runtime/mtp_tree_produce.h:209-223) column 1 + s*L + i carries node
//   (s,i) whose parent is column 1 + (s-1)*L, so that test compares a column against a SIBLING's
//   token, stops at the first non-spine column, and never publishes `accepted_columns`.
//
// WHY THIS FILE AND NOT AN EDIT TO THAT KERNEL -- A MEASURED BLOCKER, NOT A PREFERENCE
//   `src/ops/launcher/speculative_round.cu` is the ONLY file that includes
//   ops/kernel/speculative_round.cuh (grep, one includer) and it is a DEVICE TU compiled with
//   `-rdc=true`, so an edit there forces the device link to be re-run. `dl/musefix` (F-897)
//   measured that a rebuilt object's own fatbin cannot match the prebuilt
//   `build/src/CMakeFiles/ninfer_ops.dir/cmake_device_link.o`, so the `ar r` recipe `dl/allons`
//   used for a HOST TU does not apply. THIS LINE RE-RAN THE DEVICE LINK TWICE, WITH THE BUILD'S OWN
//   COMMAND (build/src/CMakeFiles/ninfer_ops.dir/dlink.txt, 251 objects), AND THE WSL SESSION DIED
//   BOTH TIMES with `Wsl/Service/E_UNEXPECTED` inside 60 s (logs/build_step3.txt,
//   logs/build_step3b.txt). The remedy is therefore not to re-run it: this kernel is a NEW,
//   SELF-CONTAINED device TU compiled with -rdc=false, which needs NO device link at all, and it is
//   launched from the WRAPPER -- a HOST TU, exactly the case allons' recipe covers.
//
// WHAT THIS KERNEL IS, PRECISELY, SO IT IS NOT MISREAD AS MORE THAN IT IS
//   It RECOMPUTES the round's accept with the contract's rule and OVERWRITES the values the launch
//   published (licensed_counts, accepted, anchors, licensed_tokens, accepted_columns, lengths, and
//   the per-token credit in cfg.token_counts), on the SAME stream and therefore between the accept
//   launch and every reader of them, INCLUDING the acceptlog dump, which synchronizes that stream.
//   It does NOT touch the multiblock finalize kernel, which keeps its chain rule on disk. The
//   published round is the contract's; the file that got it wrong is still wrong in source. That is
//   stated here rather than smoothed, and the kernel-level edit is the DEFERRED half.
//
// IT IS IDEMPOTENT ON THE PATHS THAT WERE ALREADY RIGHT. The wrapper calls it only when
// `column_masks.data != nullptr` (the contract's own gate: a tree round, the only round whose
// consumer reads a column) and only when the frame fits `kTreeAcceptMaximumWidth`. On the
// single-block route, which already applied this exact rule, the recomputed values equal the
// published ones, so the length delta is zero and the token_counts undo/redo cancels -- measured,
// not assumed: it is the no-op control this line's two-state reading includes.
//
// AND IT RELIES ON ONE THING IT CANNOT SEE. `column_masks` bound is taken to imply a GREEDY round,
// because that is what the tree's own contract says ("A tree round is greedy-only: the sampling and
// penalty routes ... refuse column_masks (the runtime rejects a tree round on those routes before it
// launches)", speculative_round.h:88-91), and the runtime's refusal is enforced before the launch
// (program_impl.h:17910-17918, re-read). A mask-bound non-greedy round would be a NEW defect and
// this kernel would be wrong on it; it is named rather than guarded by a second spelling of a
// temperature test this file cannot read.
// ============================================================================================

#include "ninfer/ops/sampling.h"
#include "ops/launcher/speculative_accept_tree.h"

#include <cstdint>

namespace ninfer::ops {

// ONE THREAD PER ROW. The accept is a sequential decision over at most fifteen columns, exactly as
// the single-block route's `tid == 0` branch and the multiblock route's `col == 0 && group == 0`
// thread already are, so a one-thread block is not a new shape -- it is the same work.
__global__ void speculative_accept_tree_greedy_kernel(
    const std::int32_t* __restrict__ target_tokens, const std::int32_t* __restrict__ drafts,
    const std::int32_t* __restrict__ current_extents,
    const std::uint64_t* __restrict__ column_masks, std::int32_t* __restrict__ lengths,
    std::int32_t* __restrict__ anchors, std::int32_t* __restrict__ licensed_tokens,
    std::int32_t* __restrict__ licensed_counts, std::int32_t* __restrict__ accepted,
    std::int32_t* __restrict__ accepted_columns, const SamplingConfig* __restrict__ configs,
    std::int32_t cols, std::int32_t k) {
    const int row = static_cast<int>(blockIdx.x);
    int extent    = current_extents[row];
    extent        = extent < 0 ? 0 : (extent > k ? k : extent);
    if (cols > detail::kTreeAcceptMaximumWidth) {
        // The same loud refusal the single-block route states, and it is unreachable on the
        // multiblock route (sampler_multiblock_ok admits cols <= kSamplerMaxColumns == 16). It
        // writes no length credit, exactly as the single-block refusal does not.
        licensed_counts[row]  = 0;
        accepted[row]         = -1;
        accepted_columns[row] = -1;
        return;
    }
    const std::int32_t* row_targets =
        target_tokens + static_cast<std::int64_t>(row) * cols;
    const std::int32_t* row_drafts   = drafts + static_cast<std::int64_t>(row) * k;
    const std::uint64_t* row_masks   = column_masks + static_cast<std::int64_t>(row) * cols;
    std::int32_t* row_tokens         = licensed_tokens + static_cast<std::int64_t>(row) * cols;

    // WHAT THE ROUTE PUBLISHED, read BEFORE it is overwritten: the length credit it already added
    // and the tokens it already credited. The undo below is what makes this an OVERWRITE and not a
    // second commit.
    const int old_produced = licensed_counts[row];

    // ---- the contract's rule, character for character the same decision the single-block route
    // ---- makes, so the two routes cannot spell one rule two ways.
    bool node_ok[detail::kTreeAcceptMaximumWidth]              = {};
    std::int32_t node_depth[detail::kTreeAcceptMaximumWidth]   = {};
    node_ok[0] = true;
    int best   = -1;
    for (int j = 1; j <= extent; ++j) {
        const std::uint64_t mask = row_masks[j];
        const std::uint64_t anc  = mask & ~(std::uint64_t{1} << j);
        const int parent         = anc == 0 ? 0 : (63 - __clzll(static_cast<long long>(anc)));
        node_depth[j]            = static_cast<std::int32_t>(__popcll(mask)) - 1;
        node_ok[j]               = node_ok[parent] && (row_targets[parent] == row_drafts[j - 1]);
        if (node_ok[j] && (best < 0 || node_depth[j] > node_depth[best])) { best = j; }
    }
    const int a        = best < 0 ? 0 : node_depth[best];
    const int take_col = best < 0 ? 0 : best;
    const int t_star   = row_targets[take_col];

    const SamplingConfig cfg = configs[row];
    if (cfg.token_counts != nullptr) {
        for (int i = 0; i < old_produced && i <= k; ++i) {
            atomicSub(&cfg.token_counts[row_tokens[i]], 1);
        }
    }
    for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
    // The tree's committed stream is the ACCEPTED CHAIN's own tokens in depth order: depth i + 1 is
    // the accepted node whose mask carries exactly i + 1 bits.
    int placed = 0;
    for (int j = 1; j <= extent && placed < a; ++j) {
        if (!node_ok[j] || node_depth[j] != placed + 1) { continue; }
        row_tokens[placed] = row_drafts[j - 1];
        ++placed;
    }
    row_tokens[a] = t_star;

    const int produced    = a + 1;
    licensed_counts[row]  = produced;
    accepted[row]         = a;
    accepted_columns[row] = take_col;
    anchors[row]          = t_star;
    lengths[row] += produced - old_produced;
    if (cfg.token_counts != nullptr) {
        for (int i = 0; i < produced; ++i) { atomicAdd(&cfg.token_counts[row_tokens[i]], 1); }
    }
}

} // namespace ninfer::ops
