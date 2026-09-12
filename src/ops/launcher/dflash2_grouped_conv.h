#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void dflash2_grouped_conv_launch(const Tensor& hidden, const Tensor& delta, const Weight& base,
                                 std::int32_t block_size, std::int32_t group_size,
                                 std::int32_t taps, std::int32_t side, Tensor& out,
                                 cudaStream_t stream);

} // namespace ninfer::ops::detail
