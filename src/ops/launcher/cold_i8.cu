#include "ops/launcher/cold_i8.h"

#include "core/device.h"
#include "ops/kernel/cold_i8_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

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


// BF16-COLD-LAND A5. The BF16 tier's inverse. Row geometry is the raw slot's: `row`
// addresses 128 B of packed E2M1 nibbles and 16 B of E4M3 group-16 scales, and the
// destination is 256 bf16 (512 B) per row. Decoding nibble * scale and converting to
// bf16 is exactly what the bf16 decode body does when it reads a still-cold page
// (gqa_attention_decode_bf16.cuh), so a warmed page and a cold page hold the same
// values -- the tile the next decode stages is the same either way.
__global__ void cold_i8_slot_restore_bf16_kernel(const std::uint8_t* __restrict__ slots,
                                                 int kv_heads,
                                                 __nv_bfloat16* __restrict__ dst_bf16) {
    constexpr int kRows = kColdI8SlotCodeBytes / 128; // 64 token rows per page
    static_assert(kRows == 64, "the raw slot row geometry is 64 rows x 128 B of nibbles");
    const int head = static_cast<int>(blockIdx.x);
    const int page = static_cast<int>(blockIdx.y);
    const std::int64_t plane = static_cast<std::int64_t>(head) +
                               static_cast<std::int64_t>(kv_heads) * page;
    const std::uint8_t* slot = slots + plane * kColdI8SlotBytes;
    __nv_bfloat16* dst       = dst_bf16 + plane * (kRows * 256);
    const int row  = static_cast<int>(threadIdx.x) >> 2;  // 64 rows
    const int lane = static_cast<int>(threadIdx.x) & 3;   // 4 quarter-rows
    if (lane != 0) { return; }
    const std::uint8_t* row_codes  = cold_i8_slot_codes(slot) + row * 128;
    const std::uint8_t* row_scales = cold_i8_slot_scales(slot) + row * 16;
    __nv_bfloat16* out             = dst + row * 256;
#pragma unroll 4
    for (int d = 0; d < 256; ++d) {
        const std::uint8_t byte = row_codes[d >> 1];
        const std::uint8_t nib  = (d & 1) != 0 ? static_cast<std::uint8_t>(byte >> 4)
                                               : static_cast<std::uint8_t>(byte & 0x0F);
        const float value = gqa_kv_nvfp4_e2m1_to_f32(nib) *
                            gqa_kv_nvfp4_e4m3_to_f32(row_scales[d >> 4]);
        out[d] = __float2bfloat16(value);
    }
}

// BF16-COLD-LAND E5. The E8 tier's inverse: verbatim code copy back into the packed
// 4-bit code plane, and the fp16 g64 scales rebuilt from the first E4M3 byte of each
// group of four in the slot's g16 tail. Plane geometry is the e8 write path's:
// gqa_kv_i4_code_index is paged_kv_element_offset<128, KVHeads> and
// gqa_kv_quant_scale_index is paged_kv_element_offset<4, KVHeads>, so a (head, page)
// plane is 8192 B of codes and 512 B (64 x 4) of fp16 scales.
__global__ void cold_i8_slot_restore_rk4v4_kernel(const std::uint8_t* __restrict__ slots,
                                               int kv_heads,
                                               std::int8_t* __restrict__ dst_codes,
                                               __half* __restrict__ dst_scales) {
    const int head = static_cast<int>(blockIdx.x);
    const int page = static_cast<int>(blockIdx.y);
    const std::int64_t plane = static_cast<std::int64_t>(head) +
                               static_cast<std::int64_t>(kv_heads) * page;
    const std::uint8_t* slot = slots + plane * kColdI8SlotBytes;
    std::int8_t* codes = dst_codes + plane * kColdI8SlotCodeBytes;
    __half* scales     = dst_scales + plane * (64 * 4);
    const int row0 = static_cast<int>(threadIdx.x) >> 2;  // 64 rows
    const int lane = static_cast<int>(threadIdx.x) & 3;   // 4 quarter-rows
    if (lane != 0) { return; }
    const std::uint8_t* row_codes  = cold_i8_slot_codes(slot) + row0 * 128;
    const std::uint8_t* row_scales = cold_i8_slot_scales(slot) + row0 * 16;
#pragma unroll
    for (int i = 0; i < 128; ++i) {
        codes[row0 * 128 + i] = static_cast<std::int8_t>(row_codes[i]);
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        scales[row0 * 4 + g] =
            __float2half_rn(gqa_kv_nvfp4_e4m3_to_f32(row_scales[g * 4]));
    }
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

// BF16-COLD-LAND A5. Reading a bf16 page back is only defined against the fixed raw
// record (decoder_state.cpp cold_slot_stride_for(DType::BF16) is static_asserted to be
// ops::kColdI8SlotBytes), so a caller that hands over any other record width is refused
// by name rather than decoded off the end of its slot.
void cold_i8_slot_restore_bf16_launch(const std::uint8_t* slots, int kv_heads, int page_count,
                                      __nv_bfloat16* dst_bf16, int slot_bytes,
                                      cudaStream_t stream) {
    if (slot_bytes != kColdI8SlotBytes) {
        throw std::invalid_argument(
            "bf16 cold restore needs the fixed int8 raw record (" +
            std::to_string(kColdI8SlotBytes) + " B) but this layer's record is " +
            std::to_string(slot_bytes) + " B");
    }
    const dim3 grid(kv_heads, page_count);
    cold_i8_slot_restore_bf16_kernel<<<grid, 256, 0, stream>>>(slots, kv_heads, dst_bf16);
    CUDA_CHECK(cudaGetLastError());
}

// BF16-COLD-LAND E5. Same fixed-record precondition as the bf16 inverse above.
void cold_i8_slot_restore_rk4v4_launch(const std::uint8_t* slots, int kv_heads, int page_count,
                                    std::int8_t* dst_codes, void* dst_scales_fp16,
                                    int slot_bytes, cudaStream_t stream) {
    if (slot_bytes != kColdI8SlotBytes) {
        throw std::invalid_argument(
            "e8 cold restore needs the fixed int8 raw record (" +
            std::to_string(kColdI8SlotBytes) + " B) but this layer's record is " +
            std::to_string(slot_bytes) + " B");
    }
    const dim3 grid(kv_heads, page_count);
    cold_i8_slot_restore_rk4v4_kernel<<<grid, 256, 0, stream>>>(
        slots, kv_heads, dst_codes, static_cast<__half*>(dst_scales_fp16));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
