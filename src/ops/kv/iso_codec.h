#pragma once
// iso_codec.h — ISO 系 KV 量化编解码 (v1 契约, 标量参考实现; GPU 内核照此语义).
//
// 布局 (与 tools/convert/kv_iso_ref.py 对拍, 改动必须同步):
//   iso4: 组 = 16 连续元素; 组头 1x fp16 scale s = max|v|/7;
//         每元素 4bit 有符号码 q = round(clamp(v/s, -7, 7))  (two's complement 存);
//         打包: 每字节 2 元素 (低 4bit 先)。 解码: v' = q * s。
//   iso3: 组 = 8 连续元素; 组头 1x fp16 scale s = max|v|/3;
//         每元素 3bit: 1bit 符号 + 2bit 幅值 |q| in 0..3;
//         打包: 8 元素 = 24bit = 3 字节 (小端位序, 元素 0 在最低位)。
//         解码: v' = sign * |q| * s。
// 组头 fp16 序列紧随 codes 之前? 否 —— 本 v1: 输出两块独立缓冲
// (codes 字节流 + scales fp16 数组), 由调用方决定放置 (与 E8/int8 fill 同款
// 双缓冲习惯)。每层行数对齐: 元素数必须被组大小整除 (调用方保证)。
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ninfer::ops::kv {

constexpr int kIso4Group = 16;
constexpr int kIso3Group = 8;

inline __host__ __device__ std::uint16_t fp16_bits(float v) {
    // 标量 fp32 -> fp16 位 (参考用; 引擎用 __float2half_rn)
    // device 端 std::bit_cast 不可靠 -> CUDA 内建; host 端 memcpy
    std::uint32_t b;
#if defined(__CUDA_ARCH__)
    b = __float_as_uint(v);
#else
    std::memcpy(&b, &v, 4);
#endif
    const std::uint32_t sign = (b >> 16) & 0x8000u;
    const std::int32_t exp = static_cast<std::int32_t>((b >> 23) & 0xff) - 127 + 15;
    const std::uint32_t mant = (b >> 13) & 0x3ffu;
    if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }   // inf
    if (exp <= 0) {
        if (exp < -10) { return static_cast<std::uint16_t>(sign); }          // 0
        const std::uint32_t m = mant | 0x400u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
        return static_cast<std::uint16_t>(sign | (m >> shift));
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | mant);
}

inline __host__ __device__ float fp16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1fu;
    const std::uint32_t mant = h & 0x3ffu;
    std::uint32_t b;
    if (exp == 0) {
        b = sign | (mant << 13);
    } else if (exp == 31) {
        b = sign | 0x7f800000u | (mant << 13);
    } else {
        b = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
#if defined(__CUDA_ARCH__)
    return __uint_as_float(b);
#else
    float f;
    std::memcpy(&f, &b, 4);
    return f;
#endif
}

// ---- iso4 ----
inline std::size_t iso4_codes_bytes(std::size_t n) { return (n + 1) / 2; }

inline void iso4_encode(const float* v, std::size_t n, std::uint8_t* codes,
                        std::uint16_t* scales) {
    for (std::size_t g = 0; g < n; g += kIso4Group) {
        float mx = 0.0f;
        for (int i = 0; i < kIso4Group; ++i) { mx = std::max(mx, std::fabs(v[g + i])); }
        const float s = mx / 7.0f;
        scales[g / kIso4Group] = fp16_bits(s > 0.0f ? s : 1.0f);
        const float ss = s > 0.0f ? s : 1.0f;
        for (int i = 0; i < kIso4Group; i += 2) {
            auto q = [&](float x) -> std::uint8_t {
                const int c = static_cast<int>(std::lround(x / ss));
                return static_cast<std::uint8_t>(std::clamp(c, -7, 7) & 0xf);
            };
            codes[(g + i) / 2] = static_cast<std::uint8_t>(q(v[g + i]) |
                                                           (q(v[g + i + 1]) << 4));
        }
    }
}

inline void iso4_decode(const std::uint8_t* codes, const std::uint16_t* scales,
                        std::size_t n, float* out) {
    for (std::size_t g = 0; g < n; g += kIso4Group) {
        const float s = fp16_to_f32(scales[g / kIso4Group]);
        for (int i = 0; i < kIso4Group; ++i) {
            const std::uint8_t byte = codes[(g + i) / 2];
            const std::uint8_t code = (i & 1) ? (byte >> 4) : (byte & 0xf);
            std::int8_t q = static_cast<std::int8_t>(code & 0x8 ? code | 0xf0 : code);
            out[g + i] = static_cast<float>(q) * s;
        }
    }
}

// ---- iso3 (符号 + 2bit 幅值, 8 元素 = 3 字节, 元素 0 在最低位) ----
inline std::size_t iso3_codes_bytes(std::size_t n) { return (n / kIso3Group) * 3; }

inline void iso3_encode(const float* v, std::size_t n, std::uint8_t* codes,
                        std::uint16_t* scales) {
    for (std::size_t g = 0; g < n; g += kIso3Group) {
        float mx = 0.0f;
        for (int i = 0; i < kIso3Group; ++i) { mx = std::max(mx, std::fabs(v[g + i])); }
        const float s = mx / 3.0f;
        scales[g / kIso3Group] = fp16_bits(s > 0.0f ? s : 1.0f);
        const float ss = s > 0.0f ? s : 1.0f;
        std::uint32_t packed = 0;
        for (int i = 0; i < kIso3Group; ++i) {
            const float x = v[g + i];
            const int mag = std::clamp(
                static_cast<int>(std::lround(std::fabs(x) / ss)), 0, 3);
            const std::uint32_t bits = static_cast<std::uint32_t>(
                (x < 0.0f ? 1u : 0u) << 2 | mag);
            packed |= bits << (3 * i);
        }
        codes[(g / kIso3Group) * 3 + 0] = packed & 0xff;
        codes[(g / kIso3Group) * 3 + 1] = (packed >> 8) & 0xff;
        codes[(g / kIso3Group) * 3 + 2] = (packed >> 16) & 0xff;
    }
}

inline void iso3_decode(const std::uint8_t* codes, const std::uint16_t* scales,
                        std::size_t n, float* out) {
    for (std::size_t g = 0; g < n; g += kIso3Group) {
        const float s = fp16_to_f32(scales[g / kIso3Group]);
        const std::uint8_t* c = codes + (g / kIso3Group) * 3;
        const std::uint32_t packed = c[0] | (std::uint32_t(c[1]) << 8) |
                                     (std::uint32_t(c[2]) << 16);
        for (int i = 0; i < kIso3Group; ++i) {
            const std::uint32_t bits = (packed >> (3 * i)) & 0x7u;
            const float mag = static_cast<float>(bits & 0x3u);
            out[g + i] = (bits & 0x4u) ? -mag * s : mag * s;
        }
    }
}

} // namespace ninfer::ops::kv
