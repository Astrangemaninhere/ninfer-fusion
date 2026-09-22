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

// S36: 256-reference constant. Geometry-generic kernels MUST use Geometry::HeadDim;
// this value is only an upper bound / legacy reference (see _collab/A_s36_muse_decode.md).
inline constexpr int kGqaKvNvfp4HeadDim  = 256;
inline constexpr int kGqaKvNvfp4Group    = 16;
// S36: 256-reference constant. Geometry-generic kernels MUST use Geometry::HeadDim;
// this value is only an upper bound / legacy reference (see _collab/A_s36_muse_decode.md).
inline constexpr int kGqaKvNvfp4Groups   = kGqaKvNvfp4HeadDim / kGqaKvNvfp4Group;
// S36: 256-reference constant. Geometry-generic kernels MUST use Geometry::HeadDim;
// this value is only an upper bound / legacy reference (see _collab/A_s36_muse_decode.md).
inline constexpr int kGqaKvNvfp4CodeLead = kGqaKvNvfp4HeadDim / 2;
// S36: 256-reference constant. Geometry-generic kernels MUST use Geometry::HeadDim;
// this value is only an upper bound / legacy reference (see _collab/A_s36_muse_decode.md).
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

// ---- E4M3FN lattice ------------------------------------------------------
//
// `gqa_kv_nvfp4_e4m3_magnitude_bits` is the magnitude lattice. It is a verbatim
// lift of what the SCALE encoder has always computed, so that every existing
// consumer of an E4M3 scale plane (nvfp4, iso3, cold-i8, entropy requant, and
// the fp8 scale plane itself) keeps byte-identical behaviour: the scale path
// treats 0x7F as "saturated scale" and reads it back through
// `gqa_kv_nvfp4_e4m3_to_f32` as ldexpf(1.875f, 8) = 480.0f.
__device__ __forceinline__ std::uint8_t gqa_kv_nvfp4_e4m3_magnitude_bits(float x) {
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

// Round-to-nearest-even E4M3FN byte. Values below the smallest normal roll up
// through the denormal mantissa; zero stays zero.
// SCALE PLANE ONLY -- the "positive scale path". Two of its properties are
// load-bearing for every packed tier's scale plane and are therefore frozen:
// (a) any x <= 0 encodes to 0x00, and (b) the top octave saturates to 0x7F,
// which its matching reader below decodes to 480.0f. Data planes use
// gqa_kv_nvfp4_data_fp32_to_e4m3 / gqa_kv_nvfp4_data_e4m3_to_f32 instead.
__device__ __forceinline__ std::uint8_t gqa_kv_nvfp4_fp32_to_e4m3(float x) {
    return gqa_kv_nvfp4_e4m3_magnitude_bits(x);
}

__device__ __forceinline__ float gqa_kv_nvfp4_e4m3_to_f32(std::uint8_t byte) {
    const int exponent = (byte >> 3) & 0x0F;
    const int mantissa = byte & 0x07;
    if (exponent == 0) { return static_cast<float>(mantissa) / 512.0f; }
    return ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

// ---- DATA plane: a real E4M3FN pair --------------------------------------
//
// Two defects made the data plane unusable, and they were a MATCHED PAIR:
//   (1) the writer was the scale encoder above, whose first line is
//       `if (!(x > 0.0f)) { return 0; }`: every K/V element with x <= 0 was
//       stored as +0. Measured on 176 real .kvc frames, token-weighted K NMSE
//       is 0.55 for that writer (about "half the elements annihilated").
//   (2) the reader took the exponent as `(byte >> 3) & 0x0F`, so bit 7 never
//       participated: decode(0x80 | b) == decode(b) for 256/256 bytes. Fixing
//       only (1) therefore makes things WORSE -- the writer starts emitting
//       0x8X and the reader flips it to a positive -- measured host NMSE
//       0.5605 (as written) -> 2.6090 (writer only) -> 0.1212 (both, top
//       octave still private). The two halves must change in one edit.
//
// The data lattice is also the FULL E4M3FN top octave: 448 (0x7E) is the
// largest finite value, the encoder admits [256,448] instead of clamping at
// 248 (the size of that error band: the group peak is clamped for 22-24% of
// the groups in the real corpus, and the clamped byte used to read back as
// 480.0 = 1.07x..1.94x the group's own amax), and 0x7F / 0xFF are the
// format's NaN codes, which this writer never emits and this reader never
// returns: a stored byte must not be able to inject a NaN into attention.
__device__ __forceinline__ std::uint8_t gqa_kv_nvfp4_data_fp32_to_e4m3(float x) {
    if (x != x) { return 0; }               // a NaN must not become a large FINITE value
    const std::uint32_t bits = __float_as_uint(x);
    const std::uint32_t sign = (bits >> 24) & 0x80u;
    const float ax           = fabsf(x);
    const std::uint32_t abits = __float_as_uint(ax);
    int exponent = static_cast<int>((abits >> 23) & 0xffu) - 127 + 7;
    std::uint32_t code;
    if (exponent >= 16) {
        code = 0x7Eu;                       // |x| >= 512: saturate to the format max, 448
    } else if (exponent <= 0) {
        // Same 2^-9 denormal grid as the scale path, but ties-to-even: the
        // scale path uses roundf() (ties away from zero), which contradicts
        // the "Round-to-nearest-even" contract the normal branch honours.
        int mantissa = static_cast<int>(rintf(ax * 512.0f));
        if (mantissa <= 0) { code = 0x0u; }
        else if (mantissa >= 8) { code = 0x08u; }        // 2^-6, the smallest normal
        else { code = static_cast<std::uint32_t>(mantissa); }
    } else {
        std::uint32_t mantissa = (abits >> 20) & 0x7u;
        const std::uint32_t guard  = (abits >> 19) & 1u;
        const std::uint32_t sticky = abits & 0x7ffffu;
        if (guard && (sticky || (mantissa & 1u))) {
            mantissa += 1;
            if (mantissa > 7) {
                mantissa = 0;
                exponent += 1;
            }
        }
        if (exponent >= 16) { code = 0x7Eu; }
        else {
            code = (static_cast<std::uint32_t>(exponent) << 3) | mantissa;
            if (code >= 0x7Fu) { code = 0x7Eu; }         // 0x7F is a NaN code: clamp to 448
        }
    }
    return static_cast<std::uint8_t>(sign | code);
}

__device__ __forceinline__ float gqa_kv_nvfp4_data_e4m3_to_f32(std::uint8_t byte) {
    const std::uint32_t mag = byte & 0x7Fu;              // bit 7 IS the sign here
    const int exponent = static_cast<int>((mag >> 3) & 0x0Fu);
    const int mantissa = static_cast<int>(mag & 0x07u);
    float value;
    if (exponent == 0) { value = static_cast<float>(mantissa) / 512.0f; }
    else if (exponent == 15 && mantissa == 7) { value = 448.0f; }   // NaN code -> format max
    else { value = ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7); }
    return (byte & 0x80u) != 0u ? -value : value;
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_nvfp4_code_index(int physical_page, int kv_head,
                                                                int d, int page_offset) {
    return paged_kv_element_offset<(Geometry::HeadDim / 2), Geometry::KVHeads>(
        physical_page, kv_head, page_offset, d >> 1);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_nvfp4_scale_index(int physical_page, int kv_head,
                                                                 int group, int page_offset) {
    return paged_kv_element_offset<(Geometry::HeadDim / kGqaKvNvfp4Group), Geometry::KVHeads>(
        physical_page, kv_head, page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_nvfp4_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(Geometry::HeadDim) *
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
