#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Registered selector geometry, S = 1..15 (include/ninfer/ops/dflash2_selector.h:60
// and the launcher's own check in src/ops/launcher/dflash2_selector.cu:21-25).
inline constexpr int kDflash2TreeMaximumSteps = 15;
// Upper bound on tree nodes for one batch row: one bit per verify column in a
// 64-bit ancestor mask, so 63 nodes plus the anchor column (see
// include/ninfer/ops/dflash2_ddtree_beam.h, kBeamMaxNodes).
inline constexpr int kDflash2TreeMaximumNodes = 63;

// No rank/token module state: the op owns no workspace and keeps no persistent
// state, exactly like dflash2_selector_launch.
void dflash2_tree_walk_launch(const Tensor& candidates, const Tensor& scores, Tensor& node_rank,
                              Tensor& node_parent, Tensor& node_depth, Tensor& node_token,
                              Tensor& node_ancestors, Tensor& node_count, std::int32_t steps,
                              std::int32_t top_k, std::int32_t node_budget,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
