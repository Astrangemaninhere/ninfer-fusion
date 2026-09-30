#pragma once

// F904 / line subagentfix -- the HOST declaration of the tree-accept overwrite. See
// src/ops/kernel/speculative_accept_tree.cuh for the defect, the contract citation and the measured
// reason this is a new self-contained TU rather than an edit to the multiblock finalize kernel.

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The verify width a tree round can carry: the Op's own bound, spelled in
// include/ninfer/ops/mtp_round.h:67 as kMtpTreeMaximumWidth and mirrored in both accept routes as a
// local `kAcceptMaximumWidth`. Declared ONCE, here, in the namespace both the wrapper (which gates on
// it) and the kernel (which indexes with it) can see, so the two cannot drift apart.
inline constexpr int kTreeAcceptMaximumWidth = 16;

// Recomputes ONE tree round's accept with the rule include/ninfer/ops/speculative_round.h:74-88
// states and overwrites every value the launch published from it, on `stream`, so the call is
// ordered between the accept launch and every reader of those values.
//
// PRECONDITIONS, all checked by the caller and not re-derived here:
//   * column_masks is bound, I64 [K+1,B] -- the contract's own tree gate;
//   * K+1 <= kTreeAcceptMaximumWidth;
//   * the round is greedy, which the runtime guarantees before the launch for a mask-bound round.
void speculative_accept_tree_greedy_overwrite(const Tensor& target_tokens, const Tensor& drafts,
                                              const Tensor& current_extents,
                                              const Tensor& column_masks, Tensor& lengths,
                                              Tensor& anchors, Tensor& licensed_tokens,
                                              Tensor& licensed_counts, Tensor& accepted,
                                              Tensor& accepted_columns, std::int32_t token_domain,
                                              const SamplingConfig* configs, cudaStream_t stream);

} // namespace ninfer::ops::detail
