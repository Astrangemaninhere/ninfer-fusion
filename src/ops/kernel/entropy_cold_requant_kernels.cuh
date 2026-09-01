#pragma once

// ninfer::ops::detail - cold-page requantization kernel for the entropy slot
// codec (revision 2).
//
// The revision-1 slot codec rANS-encoded the stored group-16 NVFP4 code
// nibbles directly; on real Qwen3.8 pages those codes are near-uniform
// (~4.0 bits/nibble) so the fixed-slot encoder fell back to the uncompressed
// plane and the cold pool saved nothing. Requantizing the same values with
// one E4M3 scale per 64 channels skews the code distribution enough for the
// unchanged order-0 rANS to compress it (measured 2.0-2.6 bits/code on
// captured .kvc frames; 3-bit signed requant was measured worse: 0.04-0.13
// NMSE vs 0.011-0.022 for g64 E2M1).
//
// The kernel reads one page plane in its stored format and writes fresh
// NVFP4 planes: packed E2M1 codes plus one E4M3FN scale per 64-channel group
// replicated into the four group-16 scale slots it covers. Downstream slot
// encode, decode, scale scatter, and attention producers stay byte-identical
// to revision 1; only the codes fed into the rANS change.

#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
// ISO3 = sign-magnitude INT3: low 3 bits magnitude 0..7, bit 3 sign.
// Inlined here to keep the op header dependency-free.
__device__ __forceinline__ std::uint8_t gqa_iso3_nibble(float value, float scale) {
    float mag = roundf(fabsf(value) / scale);
    if (mag > 7.0f) { mag = 7.0f; }
    if (mag < 0.0f) { mag = 0.0f; }
    std::uint8_t code = static_cast<std::uint8_t>(mag);
    if (value < 0.0f && code != 0) { code |= 0x08u; }
    return code;
}
__device__ __forceinline__ float gqa_iso3_decode(std::uint8_t code) {
    const float mag = static_cast<float>(code & 0x07u);
    return (code & 0x08u) != 0 ? -mag : mag;
}
#include "ops/launcher/entropy_cold_requant.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One block per (kv_head, page); 256 threads = 64 token rows x 4 groups of
// 64 channels. dst planes use the nvfp4 page-major layout the slot encoder
// expects: codes [128, 64, kv_heads, pages], scales [16, 64, kv_heads, pages].
__global__ void entropy_cold_requant_kernel(const std::uint8_t* __restrict__ src_codes,
                                            const std::uint8_t* __restrict__ src_scales,
                                            ColdRequantSource mode, int kv_heads,
                                            std::uint8_t* __restrict__ dst_codes,
                                            std::uint8_t* __restrict__ dst_scales) {
    const int head  = static_cast<int>(blockIdx.x);
    const int page  = static_cast<int>(blockIdx.y);
    const int token = static_cast<int>(threadIdx.x) >> 2;
    const int group = static_cast<int>(threadIdx.x) & 3;
    const int lane0 = group * 64;

    // Row strides in bytes: nvfp4 codes 256/2, nvfp4 scales 256/16,
    // int8 codes 256, int8 scales 4 fp16 = 8.
    const std::int64_t page_rows = static_cast<std::int64_t>(kPagedKVPageSize);
    const std::int64_t head_off =
        page_rows * (static_cast<std::int64_t>(head) +
                     static_cast<std::int64_t>(kv_heads) * static_cast<std::int64_t>(page));

    float vals[64];
    if (mode == ColdRequantSource::Nvfp4G16) {
        const std::uint8_t* codes   = src_codes + 128 * head_off + 128 * token;
        const std::uint8_t* scales = src_scales + 16 * head_off + 16 * token;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            const int          d    = lane0 + i;
            const std::uint8_t byte = codes[d >> 1];
            const std::uint8_t nib  = (d & 1) != 0 ? static_cast<std::uint8_t>(byte >> 4)
                                                   : static_cast<std::uint8_t>(byte & 0x0F);
            vals[i] = gqa_kv_nvfp4_e2m1_to_f32(nib) * gqa_kv_nvfp4_e4m3_to_f32(scales[d >> 4]);
        }
    } else if (mode == ColdRequantSource::Iso3VG16) {
        // The global NVFP4 tier stores V as ISO3 sign-magnitude INT3 nibbles in
        // the same two-per-byte plane geometry. Requant keeps the native ISO3
        // nibble semantics so the warm producers' dequant path is unchanged;
        // only the scales are re-derived per 64 channels.
        const std::uint8_t* codes   = src_codes + 128 * head_off + 128 * token;
        const std::uint8_t* scales = src_scales + 16 * head_off + 16 * token;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            const int          d    = lane0 + i;
            const std::uint8_t byte = codes[d >> 1];
            const std::uint8_t nib  = (d & 1) != 0 ? static_cast<std::uint8_t>(byte >> 4)
                                                   : static_cast<std::uint8_t>(byte & 0x0F);
            vals[i] = gqa_iso3_decode(nib) * gqa_kv_nvfp4_e4m3_to_f32(scales[d >> 4]);
        }
    } else {
        const std::int8_t* codes = reinterpret_cast<const std::int8_t*>(src_codes) + 256 * head_off +
                                   256 * token;
        const __half* scales = reinterpret_cast<const __half*>(src_scales + 8 * head_off +
                                                              8 * token);
        const float s = __half2float(scales[group]);
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            vals[i] = static_cast<float>(codes[lane0 + i]) * s;
        }
    }

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 64; ++i) { amax = fmaxf(amax, fabsf(vals[i])); }
    const bool iso3_out = mode == ColdRequantSource::Iso3VG16;
    const std::uint8_t scale_byte =
        gqa_kv_nvfp4_fp32_to_e4m3(fmaxf(amax / (iso3_out ? 7.0f : 6.0f), 0x1p-9f));
    const float s = gqa_kv_nvfp4_e4m3_to_f32(scale_byte);

    std::uint8_t* dst_c = dst_codes + 128 * head_off + 128 * token;
#pragma unroll
    for (int i = 0; i < 64; i += 2) {
        std::uint8_t lo;
        std::uint8_t hi;
        if (iso3_out) {
            lo = gqa_iso3_nibble(vals[i], s);
            hi = gqa_iso3_nibble(vals[i + 1], s);
        } else {
            lo = gqa_kv_nvfp4_e2m1_nibble(vals[i] / s);
            hi = gqa_kv_nvfp4_e2m1_nibble(vals[i + 1] / s);
        }
        dst_c[(lane0 + i) >> 1] = static_cast<std::uint8_t>(lo | (hi << 4));
    }
    std::uint8_t* dst_s = dst_scales + 16 * head_off + 16 * token;
    dst_s[group * 4 + 0] = scale_byte;
    dst_s[group * 4 + 1] = scale_byte;
    dst_s[group * 4 + 2] = scale_byte;
    dst_s[group * 4 + 3] = scale_byte;
}

} // namespace ninfer::ops::detail
