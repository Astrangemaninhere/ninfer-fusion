#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kDsparkMarkovRank = 256;

void dspark_markov_argmax_launch(const Tensor& logits, const Weight& markov_w1,
                                 const Weight& markov_w2, const Tensor& anchors, Tensor& drafts,
                                 Tensor& best_value, Tensor& best_index, Tensor& extents,
                                 std::int32_t k, float entropy_threshold, cudaStream_t stream);

} // namespace ninfer::ops::detail
