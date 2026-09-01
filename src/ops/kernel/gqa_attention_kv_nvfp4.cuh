#pragma once

// ninfer::ops - packed E2M1 NVFP4, per-token 16-channel-scale KV cache codec.
//
// The cache stores two planes per K/V tensor:
//   * code  plane: two 4-bit E2M1 words per byte, d-contiguous, leading
//     extent = head_dim / 2 bytes per token row;
//   * scale plane: one E4M3FN byte per contiguous 16-channel group, leading
//     extent = head_dim / 16 bytes per token row.
//
// Append quantizes BF16 source values x as
//     s        = max(E4M3_RNE(max_i |x_i| / 6), 2^-9)
//     code[i]  = E2M1_round_to_nearest(x_i / s)
//     decode   = E2M1(code[i]) * s.
// K may carry the per-4-channel orthogonal rotation applied by the caller;
// the codec itself is rotation-agnostic.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaKvNvfp4HeadDim  = 256;
inline constexpr int kGqaKvNvfp4Group    = 16;
inline constexpr int kGqaKvNvfp4Groups   = kGqaKvNvfp4HeadDim / kGqaKvNvfp4Group;
inline constexpr int kGqaKvNvfp4CodeLead = kGqaKvNvfp4HeadDim / 2;
inline constexpr int kGqaKvNvfp4ScaleLead = kGqaKvNvfp4Groups;

__device__ __forceinline__ std::uint8_t gqa_kv_nvfp4_e2m1_nibble(float x) {
    const float a = fabsf(x);
    std::uint8_t c;
    if (a < 0.25f) { c = 0; }
    else if (a < 0.75f) { c = 1; }
    else if (a < 1.25f) { c = 2; }
    else if (a < 1.75f) { c = 3; }
    else if (a < 2.5f) { c = 4; }
    else if (a < 3.5f) { c = 5; }
    else if (a < 5.0f) { c = 6; }
    else { c = 7; }
    if (x < 0.0f) { c |= 0x08u; }
    return c;
}

__device__ __forceinline__ float gqa_kv_nvfp4_e2m1_to_f32(std::uint8_t code) {
    const std::uint8_t mag = code & 0x07u;
    float magnitude;
    if (mag == 0) { magnitude = 0.0f; }
    else if (mag == 1) { magnitude = 0.5f; }
    else if (mag == 2) { magnitude = 1.0f; }
    else if (mag == 3) { magnitude = 1.5f; }
    else if (mag == 4) { magnitude = 2.0f; }
    else if (mag == 5) { magnitude = 3.0f; }
    else if (mag == 6) { magnitude = 4.0f; }
    else { magnitude = 6.0f; }
    return (code & 0x08u) != 0 ? -magnitude : magnitude;
}

// Round-to-nearest-even E4M3FN byte. Values below the smallest normal roll up
// through the denormal mantissa; zero stays zero.
__device__ __forceinline__ std::uint8_t gqa_kv_nvfp4_fp32_to_e4m3(float x) {
    if (!(x > 0.0f)) { return 0; }
    const std::uint32_t bits = __float_as_uint(x);
    const std::uint32_t sign = (bits >> 24) & 0x80u;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 7;
    if (exponent >= 15) { return static_cast<std::uint8_t>(sign | (15u << 3) | 7u); }
    if (exponent <= 0) {
        // E4M3FN denormals decode as mantissa / 512 (mantissa * 2^-9), so
        // the encoder must quantize x * 512, not x * 64.
        int mantissa = static_cast<int>(roundf(x * 512.0f));
        if (mantissa <= 0) { return static_cast<std::uint8_t>(sign); }
        if (mantissa >= 8) { return static_cast<std::uint8_t>(sign | (1u << 3)); }
        return static_cast<std::uint8_t>(sign | mantissa);
    }
    std::uint32_t mantissa = (bits >> 20) & 0x7u;
    const std::uint32_t guard  = (bits >> 19) & 1u;
    const std::uint32_t sticky = bits & 0x7ffffu;
    if (guard && (sticky || (mantissa & 1u))) {
        mantissa += 1;
        if (mantissa > 7) {
            mantissa = 0;
            exponent += 1;
            if (exponent >= 15) { return static_cast<std::uint8_t>(sign | (15u << 3) | 7u); }
        }
    }
    return static_cast<std::uint8_t>(sign | (exponent << 3) | mantissa);
}

__device__ __forceinline__ float gqa_kv_nvfp4_e4m3_to_f32(std::uint8_t byte) {
    const int exponent = (byte >> 3) & 0x0F;
    const int mantissa = byte & 0x07;
    if (exponent == 0) { return static_cast<float>(mantissa) / 512.0f; }
    return ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_nvfp4_code_index(int physical_page, int kv_head,
                                                                int d, int page_offset) {
    return paged_kv_element_offset<kGqaKvNvfp4CodeLead, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, d >> 1);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_nvfp4_scale_index(int physical_page, int kv_head,
                                                                 int group, int page_offset) {
    return paged_kv_element_offset<kGqaKvNvfp4ScaleLead, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_nvfp4_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaKvNvfp4HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

// Dequantize 8 consecutive E2M1 codes (dims [d, d+8), inside one 16-group)
// with the group's E4M3 scale into 8 BF16 packed as an int4. codes8 points
// at the four packed bytes.
__device__ __forceinline__ int4 gqa_kv_dequant_nvfp4x8_from(const std::uint8_t* codes8, float scale) {
    const int raw = load_vec<int>(codes8);
    const std::uint8_t* c = reinterpret_cast<const std::uint8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = gqa_kv_nvfp4_e2m1_to_f32(c[i] & 0x0Fu) * scale;
        const float x1 = gqa_kv_nvfp4_e2m1_to_f32(c[i] >> 4) * scale;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

} // namespace ninfer::ops
