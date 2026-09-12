#include "ops/launcher/dflash2_grouped_conv.h"

#include "core/device.h"
#include "ops/kernel/dflash2_grouped_conv.cuh"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

void dflash2_grouped_conv_launch(const Tensor& hidden, const Tensor& delta, const Weight& base,
                                 std::int32_t block_size, std::int32_t group_size,
                                 std::int32_t taps, std::int32_t side, Tensor& out,
                                 cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int32_t hidden_size = hidden.ne[0];
    const std::int32_t tokens      = hidden.ne[1];
    const std::int32_t groups      = hidden_size / group_size;
    const int grid                 = static_cast<int>(
        std::max<std::int64_t>(1, (static_cast<std::int64_t>(hidden_size) + kBlock - 1) / kBlock));
    dflash2_grouped_conv_kernel<kBlock><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const __nv_bfloat16*>(delta.data),
        static_cast<const __nv_bfloat16*>(base.qdata), static_cast<__nv_bfloat16*>(out.data),
        hidden_size, tokens, taps, groups, group_size, block_size, side);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
