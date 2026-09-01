#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kDflash2SelectorRank = 256;
inline constexpr int kDflash2SelectorTopK = 16;

void dflash2_selector_launch(const Tensor& unary_logits, const Tensor& projected_hidden,
                             const Weight& predecessor_codebook,
                             const Weight& successor_codebook, const Tensor& anchors,
                             Tensor& candidates, Tensor& unary, Tensor& scores, Tensor& drafts,
                             std::int32_t steps, std::int32_t top_k, cudaStream_t stream);

} // namespace ninfer::ops::detail
