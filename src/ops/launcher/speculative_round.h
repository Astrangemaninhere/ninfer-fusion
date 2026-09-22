#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void speculative_prepare_verify_inputs_launch(const Tensor& anchors, const Tensor& drafts,
                                              const Tensor& base_positions,
                                              const Tensor& current_extents, Tensor& verify_ids,
                                              Tensor& positions, cudaStream_t stream);
void speculative_prepare_verify_ids_launch(const Tensor& anchors, const Tensor& drafts,
                                           const Tensor& current_extents, Tensor& verify_ids,
                                           cudaStream_t stream);

// column_masks is I64 [K+1,B] or empty: non-empty selects the greedy TREE accept (and the depth /
// column it publishes into accepted_columns). The stochastic routes keep the public Op's separate
// definitions; the runtime refuses a tree round on them before this launch.
void speculative_accept_greedy_drafts_launch(const Tensor& target_tokens, const Tensor& logits,
                                             const Tensor& drafts, const Tensor& current_extents,
                                             const Tensor& column_masks, Tensor& lengths,
                                             Tensor& anchors, Tensor& licensed_tokens,
                                             Tensor& licensed_counts, Tensor& accepted,
                                             Tensor& accepted_columns, std::int32_t token_domain,
                                             const SamplingConfig* configs,
                                             const Tensor& draft_ids, const Tensor& draft_probs,
                                             DeviceSpan workspace, cudaStream_t stream);

void speculative_select_accepted_hidden_launch(const Tensor& hidden, const Tensor& selectors,
                                               Tensor& out, cudaStream_t stream);

void proposal_remap_token_ids_launch(Tensor& proposal_tokens, const std::int32_t* id_map,
                                     std::int32_t n, cudaStream_t stream);

} // namespace ninfer::ops::detail
