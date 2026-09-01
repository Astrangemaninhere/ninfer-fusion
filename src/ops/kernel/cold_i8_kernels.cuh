#pragma once

// ninfer::ops::detail - raw cold-slot codec for INT8-tier KV pages (v2b).
//
// The INT8 tier's requantized E2M1 codes are near-uniform (measured H
// 3.6-3.9 on real planes), so rANS gains nothing; this slot format stores
// the requant output verbatim with a fixed layout and no overflow path:
//
//   [ 16 B header | 8192 B packed E2M1 nibbles | 1024 B E4M3 g16 scales ]
//
// The nibble/scale planes use the same page-major geometry the NVFP4 tier
// stores (codes [128, 64, kv_heads, pages], scales [16, 64, kv_heads,
// pages]), produced by entropy_cold_requant's Int8G64 mode. Restore
// converts a slot back into the INT8 tier's native planes (int8 codes +
// fp16 group-64 scales) with the upper-bound group scale, adding <0.4%
// quantization noise on top of the requant's measured 0.012 NMSE.
//
// Kernel DEFINITIONS live only in ops/launcher/cold_i8.cu; this header
// declares them plus the shared device helpers so attention kernels can
// include it without duplicate device-link definitions.

#include "ninfer/ops/cold_i8.h"
#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kColdI8SlotHeaderBytes = 16;
inline constexpr int kColdI8SlotCodeBytes   = 8192;   // 64 rows x 128 B nibbles
inline constexpr int kColdI8SlotScaleBytes  = 1024;   // 64 rows x 16 B E4M3
static_assert(kColdI8SlotHeaderBytes + kColdI8SlotCodeBytes + kColdI8SlotScaleBytes ==
              ninfer::ops::kColdI8SlotBytes);
inline constexpr std::uint32_t kColdI8SlotMagic = 0x49384352u; // "RC8I"

// Pack one requantized (page, head, plane) into a raw slot. src uses the
// nvfp4 page-major geometry entropy_cold_requant emits; slots layout is
// [slot_bytes, kv_heads, 2, pages] with V one nb[2] step past K.
__global__ void cold_i8_slot_pack_kernel(const std::uint8_t* __restrict__ src_codes,
                                         const std::uint8_t* __restrict__ src_scales,
                                         int kv_heads,
                                         std::uint8_t* __restrict__ slots,
                                         std::int32_t* __restrict__ slot_valid);

// Warm restore: unpack one (page, head, plane) slot into the INT8 tier's
// native planes (int8 codes [256,64,kv_heads,pages], fp16 scales
// [4,64,kv_heads,pages]). One block per (head, page); 256 threads split
// the 64 rows.
__global__ void cold_i8_slot_restore_kernel(const std::uint8_t* __restrict__ slots,
                                            int kv_heads, std::int8_t* __restrict__ dst_codes,
                                            __half* __restrict__ dst_scales);

// Slot region accessors for producers.
__device__ __forceinline__ const std::uint8_t*
cold_i8_slot_scales(const std::uint8_t* slot) {
    return slot + kColdI8SlotHeaderBytes + kColdI8SlotCodeBytes;
}

__device__ __forceinline__ const std::uint8_t*
cold_i8_slot_codes(const std::uint8_t* slot) {
    return slot + kColdI8SlotHeaderBytes;
}

// Decode one key row of a raw slot into INT8-tier native form: 256 int8
// codes plus one fp16 scale per 64-channel group. The group scale is the
// upper bound 6*max(e4m3 sub-scales)/127 so no amax scan of the decoded
// values is needed; codes clamp at 127 so fp16 rounding-down is safe.
// Used by the warm-restore kernel and the attention producers' cold staging.
__device__ __forceinline__ void cold_i8_decode_row(const std::uint8_t* slot, int row,
                                                   std::int8_t* codes_out,   // 256, d-major
                                                   __half* scales_out) {     // 4 groups
    const std::uint8_t* row_codes  = cold_i8_slot_codes(slot) + row * 128;
    const std::uint8_t* row_scales = cold_i8_slot_scales(slot) + row * 16;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        float mx = 0.0f;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            mx = fmaxf(mx, gqa_kv_nvfp4_e4m3_to_f32(row_scales[g * 4 + s]));
        }
        const float scale = mx * 6.0f / 127.0f;
        scales_out[g]     = __float2half(scale);
        const float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
#pragma unroll
        for (int i = 0; i < 64; i += 2) {
            const int d          = g * 64 + i;
            const std::uint8_t b = row_codes[d >> 1];
            const float v0 = gqa_kv_nvfp4_e2m1_to_f32(b & 0x0F) *
                             gqa_kv_nvfp4_e4m3_to_f32(row_scales[d >> 4]);
            const float v1 = gqa_kv_nvfp4_e2m1_to_f32(b >> 4) *
                             gqa_kv_nvfp4_e4m3_to_f32(row_scales[(d + 1) >> 4]);
            int c0         = __float2int_rn(v0 * inv);
            int c1         = __float2int_rn(v1 * inv);
            c0             = max(-127, min(127, c0));
            c1             = max(-127, min(127, c1));
            codes_out[d]     = static_cast<std::int8_t>(c0);
            codes_out[d + 1] = static_cast<std::int8_t>(c1);
        }
    }
}

} // namespace ninfer::ops::detail
