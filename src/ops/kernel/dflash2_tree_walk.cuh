#pragma once

// Implements: include/ninfer/ops/dflash2_tree_walk.h
// Match: the selector's own candidate/score layouts, and the ONE beam core shared
// with the host test (include/ninfer/ops/dflash2_ddtree_beam.h).

#include "ninfer/ops/dflash2_ddtree_beam.h"
#include "ops/launcher/dflash2_tree_walk.h"

#include <cstdint>

namespace ninfer::ops {

// One thread block per batch row; lane 0 owns the row's beam search. That is
// deliberate for this additive, default-off op: the whole row is
// steps * kTopK * kTopK <= 15 * 256 = 3840 score floats and the beam keeps at most
// kBeamMaxPaths * kBeamTopK = 256 extensions per step, i.e. the work is a few
// hundred dependent loads and a 7-times-repeated top-L insertion -- far below what a
// per-row warp split would have to synchronise for. A later slice can split a row
// across the warp if this shows up in a profile.
__global__ void dflash2_tree_walk_kernel(const std::int32_t* __restrict__ candidates,
                                         const float* __restrict__ scores, std::int32_t batch,
                                         std::int32_t steps, std::int32_t node_budget,
                                         std::int32_t max_nodes,
                                         std::int32_t* __restrict__ node_rank,
                                         std::int32_t* __restrict__ node_parent,
                                         std::int32_t* __restrict__ node_depth,
                                         std::int32_t* __restrict__ node_token,
                                         std::int64_t* __restrict__ node_ancestors,
                                         std::int32_t* __restrict__ node_count) {
    const int b = static_cast<int>(blockIdx.x);
    if (b >= batch) { return; }
    if (static_cast<int>(threadIdx.x) != 0) { return; }

    // The engine's flat layouts put the batch stride OUTSIDE the (s,p,c) block:
    //   scores[b][s][p][c]    = b + batch * (s + steps*(p + K*c))
    //   candidates[b][s][c]   = b + batch * (s + steps*c)
    // (src/ops/kernel/dflash2_selector.cuh:18-27), so one row is `base + b` and the
    // in-row offsets are exactly ddtree::beam_score_offset / beam_candidate_offset.
    const float* row_scores           = scores + b;
    const std::int32_t* row_candidates = candidates + b;

    __shared__ ddtree::BeamScratch scratch;
    __shared__ ddtree::BeamNode shared_nodes[detail::kDflash2TreeMaximumNodes];

    const std::int32_t paths = ddtree::beam_build(row_scores, steps, node_budget, scratch);
    if (paths <= 0) {
        // Unreachable for the validated argument domain; be loud rather than publish
        // a silently empty tree (a negative count is outside every valid contract).
        node_count[b] = -1;
        return;
    }
    const std::int32_t count =
        ddtree::beam_nodes(scratch, steps, detail::kDflash2TreeMaximumNodes, shared_nodes);
    if (count < 0) {
        node_count[b] = -1;
        return;
    }

    const std::int64_t base = static_cast<std::int64_t>(b) * max_nodes;
    for (std::int32_t i = 0; i < count; ++i) {
        const ddtree::BeamNode& node = shared_nodes[i];
        node_rank[base + i]          = node.rank;
        node_parent[base + i]        = node.parent;
        node_depth[base + i]         = node.depth;
        // Token id of this node's candidate: the column the verify pass feeds. Depth
        // and rank are the node's identity; rows (p) are irrelevant here.
        node_token[base + i] = row_candidates[ddtree::beam_candidate_offset(0, 1, node.depth,
                                                                           steps, node.rank)];
        node_ancestors[base + i] =
            static_cast<std::int64_t>(ddtree::beam_node_ancestors(shared_nodes, i));
    }
    node_count[b] = count;
}

} // namespace ninfer::ops
