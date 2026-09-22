#include "ops/launcher/dflash2_tree_walk.h"

#include "core/device.h"
#include "ops/kernel/dflash2_tree_walk.cuh"
#include "ops/launcher/dflash2_selector.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void dflash2_tree_walk_launch(const Tensor& candidates, const Tensor& scores, Tensor& node_rank,
                              Tensor& node_parent, Tensor& node_depth, Tensor& node_token,
                              Tensor& node_ancestors, Tensor& node_count, std::int32_t steps,
                              std::int32_t top_k, std::int32_t node_budget,
                              cudaStream_t stream) {
    // Same registered domain as the selector: S = 1..15, K = 16
    // (src/ops/launcher/dflash2_selector.cu:21-25). The node budget must be able to
    // hold one chain and must fit the 64-bit ancestor mask.
    if (steps < 1 || steps > kDflash2TreeMaximumSteps || top_k != kDflash2SelectorTopK) {
        throw std::invalid_argument("dflash2_tree_walk: registered domain is S=1..15, K=16");
    }
    if (node_budget < steps || node_budget > kDflash2TreeMaximumNodes) {
        throw std::invalid_argument(
            "dflash2_tree_walk: node_budget must be in [steps, 63] (64-bit ancestor mask)");
    }
    const std::int32_t batch = candidates.ne[0];
    // The kernel writes into fixed-size shared memory and rejects an overflowing
    // tree, so the caller's row capacity has to be able to hold the budget.
    if (node_rank.ne[1] < node_budget) {
        throw std::invalid_argument(
            "dflash2_tree_walk: the node arrays must be at least node_budget wide");
    }

    dflash2_tree_walk_kernel<<<batch, 32, 0, stream>>>(
        static_cast<const std::int32_t*>(candidates.data), static_cast<const float*>(scores.data),
        batch, steps, node_budget, node_rank.ne[1], static_cast<std::int32_t*>(node_rank.data),
        static_cast<std::int32_t*>(node_parent.data), static_cast<std::int32_t*>(node_depth.data),
        static_cast<std::int32_t*>(node_token.data),
        static_cast<std::int64_t*>(node_ancestors.data),
        static_cast<std::int32_t*>(node_count.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
