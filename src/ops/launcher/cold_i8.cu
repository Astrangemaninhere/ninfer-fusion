#include "ops/launcher/cold_i8.h"

#include "core/device.h"
#include "ops/kernel/cold_i8_kernels.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Kernel definitions live in this single TU: the shared header only
// declares them so attention kernels can include its device helpers
// without duplicate device-link definitions.
__global__ void cold_i8_slot_pack_kernel(const std::uint8_t* __restrict__ src_codes,
                                         const std::uint8_t* __restrict__ src_scales,
                                         int kv_heads,
                                         std::uint8_t* __restrict__ slots,
                                         std::int32_t* __restrict__ slot_valid,
                                         int slot_bytes) {
    const int head = static_cast<int>(blockIdx.x);
    const int page = static_cast<int>(blockIdx.y);
    const std::int64_t plane = static_cast<std::int64_t>(head) +
                               static_cast<std::int64_t>(kv_heads) * page;
    const std::uint8_t* src_c = src_codes + plane * kColdI8SlotCodeBytes;
    const std::uint8_t* src_s = src_scales + plane * kColdI8SlotScaleBytes;
    std::uint8_t* slot = slots + plane * slot_bytes;
    if (threadIdx.x == 0) {
        *reinterpret_cast<std::uint32_t*>(slot) = kColdI8SlotMagic;
        *reinterpret_cast<std::uint16_t*>(slot + 4) = 1; // version
        *reinterpret_cast<std::uint16_t*>(slot + 6) = 1; // flags: valid
    }
    __syncthreads();
    for (int i = static_cast<int>(threadIdx.x); i < kColdI8SlotCodeBytes; i += 256) {
        slot[kColdI8SlotHeaderBytes + i] = src_c[i];
    }
    for (int i = static_cast<int>(threadIdx.x); i < kColdI8SlotScaleBytes; i += 256) {
        slot[kColdI8SlotHeaderBytes + kColdI8SlotCodeBytes + i] = src_s[i];
    }
    if (threadIdx.x == 0) {
        slot_valid[plane] = 1;  // fixed layout: always valid, no overflow path
    }
}

__global__ void cold_i8_slot_restore_kernel(const std::uint8_t* __restrict__ slots,
                                            int kv_heads, std::int8_t* __restrict__ dst_codes,
                                            __half* __restrict__ dst_scales,
                                            int slot_bytes) {
    const int head = static_cast<int>(blockIdx.x);
    const int page = static_cast<int>(blockIdx.y);
    const std::int64_t plane = static_cast<std::int64_t>(head) +
                               static_cast<std::int64_t>(kv_heads) * page;
    const std::uint8_t* slot = slots + plane * slot_bytes;
    std::int8_t* codes = dst_codes + plane * (64 * 256);
    __half* scales     = dst_scales + plane * (64 * 4);
    const int row0     = static_cast<int>(threadIdx.x) >> 2;  // 64 rows
    const int lane     = static_cast<int>(threadIdx.x) & 3;   // 4 quarter-rows
    if (lane != 0) { return; }
    std::int8_t row_codes[256];
    __half row_scales[4];
    cold_i8_decode_row(slot, row0, row_codes, row_scales);
#pragma unroll
    for (int g = 0; g < 4; ++g) { scales[row0 * 4 + g] = row_scales[g]; }
#pragma unroll
    for (int d = 0; d < 256; ++d) { codes[row0 * 256 + d] = row_codes[d]; }
}


void cold_i8_slot_pack_launch(const std::uint8_t* src_codes, const std::uint8_t* src_scales,
                              int kv_heads, int page_count, std::uint8_t* slots,
                              std::int32_t* slot_valid, int slot_bytes, cudaStream_t stream) {
    const dim3 grid(kv_heads, page_count);
    cold_i8_slot_pack_kernel<<<grid, 256, 0, stream>>>(src_codes, src_scales, kv_heads, slots,
                                                       slot_valid, slot_bytes);
    CUDA_CHECK(cudaGetLastError());
}

void cold_i8_slot_restore_launch(const std::uint8_t* slots, int kv_heads, int page_count,
                                 std::int8_t* dst_codes, void* dst_scales_fp16,
                                 int slot_bytes, cudaStream_t stream) {
    const dim3 grid(kv_heads, page_count);
    cold_i8_slot_restore_kernel<<<grid, 256, 0, stream>>>(
        slots, kv_heads, dst_codes, static_cast<__half*>(dst_scales_fp16), slot_bytes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
