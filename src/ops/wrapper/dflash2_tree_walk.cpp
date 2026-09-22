#include "ninfer/ops/dflash2_tree_walk.h"

#include "ops/launcher/dflash2_selector.h"
#include "ops/launcher/dflash2_tree_walk.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

void dflash2_tree_walk(const Tensor& candidates, const Tensor& scores, Tensor& node_rank,
                       Tensor& node_parent, Tensor& node_depth, Tensor& node_token,
                       Tensor& node_ancestors, Tensor& node_count, std::int32_t steps,
                       std::int32_t top_k, std::int32_t node_budget, cudaStream_t stream) {
    const auto invalid = [](const char* message) { throw std::invalid_argument(message); };
    if (candidates.dtype != DType::I32 || scores.dtype != DType::FP32 ||
        node_rank.dtype != DType::I32 || node_parent.dtype != DType::I32 ||
        node_depth.dtype != DType::I32 || node_token.dtype != DType::I32 ||
        node_ancestors.dtype != DType::I64 || node_count.dtype != DType::I32) {
        invalid("dflash2_tree_walk: candidates I32, scores F32, node ids/depths I32, "
                "ancestor masks I64");
    }
    const auto contiguous = [](const Tensor& tensor) {
        return tensor.is_contiguous() && tensor.data != nullptr;
    };
    if (!contiguous(candidates) || !contiguous(scores) || !contiguous(node_rank) ||
        !contiguous(node_parent) || !contiguous(node_depth) || !contiguous(node_token) ||
        !contiguous(node_ancestors) || !contiguous(node_count)) {
        invalid("dflash2_tree_walk: tensors must be contiguous and non-null");
    }
    if (steps < 1 || steps > detail::kDflash2TreeMaximumSteps ||
        top_k != detail::kDflash2SelectorTopK) {
        invalid("dflash2_tree_walk: registered domain is S=1..15, K=16");
    }
    const std::int32_t batch = candidates.ne[0];
    if (batch < 1 || batch > 8) {
        invalid("dflash2_tree_walk: batch must be 1..8 (the registered selector domain)");
    }
    if (candidates.ne[1] != steps || candidates.ne[2] != top_k) {
        invalid("dflash2_tree_walk: candidates must be I32 [B,S,K]");
    }
    // The engine publishes scores as one [B,S,K,K] block per batch element whose flat
    // layout is b + batch * (s + steps*(p + K*c)); see
    // src/ops/kernel/dflash2_selector.cuh:24-27. Requiring the exact shape is what
    // lets the kernel treat `scores + b` as one row.
    if (scores.ne[0] != batch || scores.ne[1] != steps || scores.ne[2] != top_k ||
        scores.ne[3] != top_k) {
        invalid("dflash2_tree_walk: scores must be F32 [B,S,K,K] in the selector layout");
    }
    if (node_budget < steps) {
        invalid("dflash2_tree_walk: node_budget must be able to hold one chain (>= steps)");
    }
    if (node_budget > detail::kDflash2TreeMaximumNodes) {
        invalid("dflash2_tree_walk: node_budget must fit the 64-bit ancestor mask (<= 63)");
    }
    if (node_rank.ne[0] != batch || node_rank.ne[1] < node_budget ||
        node_rank.ne[1] > detail::kDflash2TreeMaximumNodes) {
        invalid("dflash2_tree_walk: node arrays must be I32 [B,max_nodes] with "
                "node_budget <= max_nodes <= 63");
    }
    // The per-node arrays share one row width so the kernel can use one stride.
    const auto same_width = [&](const Tensor& tensor) {
        return tensor.ne[0] == batch && tensor.ne[1] == node_rank.ne[1];
    };
    if (!same_width(node_parent) || !same_width(node_depth) || !same_width(node_token) ||
        !same_width(node_ancestors)) {
        invalid("dflash2_tree_walk: every node array must be [B,max_nodes] with the same width");
    }
    if (node_count.ne[0] != batch) {
        invalid("dflash2_tree_walk: node_count must be I32 [B]");
    }

    detail::dflash2_tree_walk_launch(candidates, scores, node_rank, node_parent, node_depth,
                                     node_token, node_ancestors, node_count, steps, top_k,
                                     node_budget, stream);
}

} // namespace ninfer::ops
