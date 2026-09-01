#include "ops/launcher/entropy_cold_requant.h"

#include "core/device.h"
#include "ops/kernel/entropy_cold_requant_kernels.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void entropy_cold_requant_raw_launch(const std::uint8_t* src_codes,
                                     const std::uint8_t* src_scales, ColdRequantSource mode,
                                     int kv_heads, int page_count, std::uint8_t* dst_codes,
                                     std::uint8_t* dst_scales, cudaStream_t stream) {
    const dim3 grid(kv_heads, page_count);
    entropy_cold_requant_kernel<<<grid, 256, 0, stream>>>(src_codes, src_scales, mode, kv_heads,
                                                          dst_codes, dst_scales);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
