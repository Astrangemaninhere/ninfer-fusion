#include "ninfer/ops/cold_i8.h"
#include "ninfer/ops/entropy_cold_requant.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeadDim    = 256;
constexpr int kPageRows   = 64;
constexpr int kKvHeads    = 4;
constexpr int kI8CodeB    = kHeadDim * kPageRows;      // 16384 per head
constexpr int kI8ScaleB   = kHeadDim / 64 * 2 * kPageRows; // 512 fp16 bytes
constexpr int kNvCodeB    = kHeadDim / 2 * kPageRows;  // 8192
constexpr int kNvScaleB   = kHeadDim / 16 * kPageRows; // 1024

std::uint8_t e4m3_rne(float x) {
    if (!(x > 0.0f)) { return 0; }
    std::uint32_t bits;
    std::memcpy(&bits, &x, 4);
    const std::uint32_t sign = (bits >> 24) & 0x80u;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 7;
    if (exponent >= 15) { return static_cast<std::uint8_t>(sign | (15u << 3) | 7u); }
    if (exponent <= 0) {
        int mantissa = static_cast<int>(std::nearbyint(x * 512.0f));
        if (mantissa <= 0) { return static_cast<std::uint8_t>(sign); }
        if (mantissa >= 8) { return static_cast<std::uint8_t>(sign | (1u << 3)); }
        return static_cast<std::uint8_t>(sign | mantissa);
    }
    std::uint32_t mantissa = (bits >> 20) & 0x7u;
    const std::uint32_t guard = (bits >> 19) & 1u;
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

float e4m3_to_f32(std::uint8_t byte) {
    const int e = (byte >> 3) & 0xF;
    const int m = byte & 0x7;
    if (e == 0) { return static_cast<float>(m) / 512.0f; }
    return (1.0f + static_cast<float>(m) / 8.0f) * std::pow(2.0f, static_cast<float>(e - 7));
}

float e2m1_to_f32(std::uint8_t code) {
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[code & 0x7];
    return (code & 0x8) != 0 ? -v : v;
}

std::uint8_t e2m1_code(float x) {
    const float a = std::fabs(x);
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

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h >> 15) & 1u;
    const std::uint32_t exp  = (h >> 10) & 0x1Fu;
    const std::uint32_t man  = h & 0x3FFu;
    float out;
    if (exp == 0) {
        out = std::ldexp(static_cast<float>(man), -24);
    } else {
        out = std::ldexp(1024.0f + static_cast<float>(man), static_cast<int>(exp) - 25);
    }
    return sign != 0 ? -out : out;
}

std::uint16_t float_to_half(float f) {
    if (f <= 0.0f) { return 0; }
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    int exponent              = static_cast<int>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mantissa    = (x >> 13) & 0x3FFu;
    if (exponent <= 0) {
        const std::uint32_t man_full = (x & 0x7FFFFFu) | 0x800000u;
        const int shift              = 14 - exponent + 1;
        mantissa                     = man_full >> shift;
        const std::uint32_t round_bit = (man_full >> (shift - 1)) & 1u;
        if (round_bit != 0) { mantissa += 1; }
        exponent = 0;
    } else {
        const std::uint32_t round_bit = (x >> 12) & 1u;
        const std::uint32_t sticky    = x & 0xFFFu;
        if (round_bit != 0 && (sticky != 0 || (mantissa & 1u) != 0)) {
            mantissa += 1;
            if (mantissa > 0x3FFu) {
                mantissa = 0;
                exponent += 1;
            }
        }
    }
    if (exponent >= 31) { return static_cast<std::uint16_t>(sign | (31u << 10)); }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) |
                                      mantissa);
}

} // namespace

int main() {
    std::mt19937 rng(20260830);
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        if (!ok) {
            std::printf("FAIL: %s\n", what);
            ++failures;
        }
    };

    // 1) synthetic int8 page per head
    std::vector<std::int8_t> src_codes(static_cast<std::size_t>(kKvHeads) * kI8CodeB);
    std::vector<std::uint16_t> src_scales(static_cast<std::size_t>(kKvHeads) * kPageRows * 4);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::uniform_real_distribution<float> level(0.01f, 50.0f);
    for (int head = 0; head < kKvHeads; ++head) {
        for (int row = 0; row < kPageRows; ++row) {
            for (int g = 0; g < 4; ++g) {
                const float amp = level(rng);
                float amax      = 1e-6f;
                float vals[64];
                for (int i = 0; i < 64; ++i) {
                    float v = amp * noise(rng);
                    if (((row * 7 + i) % 251) == 0) { v *= 32.0f; }
                    vals[i] = v;
                    amax    = std::fmax(amax, std::fabs(v));
                }
                const float s      = std::fmax(amax / 127.0f, 1e-30f);
                const std::uint16_t sb = float_to_half(s);
                src_scales[(static_cast<std::size_t>(head) * kPageRows + row) * 4 + g] = sb;
                const float sh = half_to_float(sb);
                for (int i = 0; i < 64; ++i) {
                    float q = std::nearbyint(vals[i] / sh);
                    q = std::fmin(127.0f, std::fmax(-127.0f, q));
                    src_codes[static_cast<std::size_t>(head) * kI8CodeB + row * kHeadDim +
                              g * 64 + i] = static_cast<std::int8_t>(q);
                }
            }
        }
    }

    // 2) requant + pack on device
    GuardedDeviceBuffer d_src_codes(static_cast<std::size_t>(kKvHeads) * kI8CodeB);
    GuardedDeviceBuffer d_src_scales(static_cast<std::size_t>(kKvHeads) * kI8ScaleB);
    GuardedDeviceBuffer d_rq_codes(static_cast<std::size_t>(kKvHeads) * kNvCodeB);
    GuardedDeviceBuffer d_rq_scales(static_cast<std::size_t>(kKvHeads) * kNvScaleB);
    GuardedDeviceBuffer d_slots(static_cast<std::size_t>(kKvHeads) * 2 * ops::kColdI8SlotBytes);
    GuardedDeviceBuffer d_valid(static_cast<std::size_t>(kKvHeads) * 2 * sizeof(std::int32_t));
    GuardedDeviceBuffer d_out_codes(static_cast<std::size_t>(kKvHeads) * kI8CodeB);
    GuardedDeviceBuffer d_out_scales(static_cast<std::size_t>(kKvHeads) * kI8ScaleB);

    d_src_codes.copy_from_host(src_codes.data(), d_src_codes.bytes());
    d_src_scales.copy_from_host(src_scales.data(), d_src_scales.bytes());

    ops::entropy_cold_requant_raw(
        static_cast<const std::uint8_t*>(d_src_codes.data()),
        static_cast<const std::uint8_t*>(d_src_scales.data()),
        ops::EntropyColdRequantMode::Int8G64, kKvHeads, 1,
        static_cast<std::uint8_t*>(d_rq_codes.data()),
        static_cast<std::uint8_t*>(d_rq_scales.data()), nullptr);
    cuda_synchronize();
    auto* valid_k = static_cast<std::int32_t*>(d_valid.data());
    auto* valid_v = valid_k + kKvHeads;
    ops::cold_i8_slot_pack_raw(static_cast<const std::uint8_t*>(d_rq_codes.data()),
                               static_cast<const std::uint8_t*>(d_rq_scales.data()), kKvHeads, 1,
                               static_cast<std::uint8_t*>(d_slots.data()), valid_k, nullptr);
    ops::cold_i8_slot_pack_raw(
        static_cast<const std::uint8_t*>(d_rq_codes.data()) + static_cast<std::size_t>(kKvHeads) * kNvCodeB,
        static_cast<const std::uint8_t*>(d_rq_scales.data()) + static_cast<std::size_t>(kKvHeads) * kNvScaleB,
        kKvHeads, 1,
        static_cast<std::uint8_t*>(d_slots.data()) + static_cast<std::size_t>(kKvHeads) * ops::kColdI8SlotBytes,
        valid_v, nullptr);
    cuda_synchronize();
    ops::cold_i8_slot_restore_raw(static_cast<const std::uint8_t*>(d_slots.data()), kKvHeads, 1,
                                  static_cast<std::int8_t*>(d_out_codes.data()),
                                  d_out_scales.data(), nullptr);
    cuda_synchronize();

    std::vector<std::int8_t> got_codes(src_codes.size());
    std::vector<std::uint16_t> got_scales(src_scales.size());
    std::vector<std::int32_t> got_valid(static_cast<std::size_t>(kKvHeads) * 2);
    d_out_codes.copy_to_host(got_codes.data(), d_out_codes.bytes());
    d_out_scales.copy_to_host(got_scales.data(), d_out_scales.bytes());
    d_valid.copy_to_host(got_valid.data(), d_valid.bytes());
    for (std::size_t i = 0; i < got_valid.size(); ++i) {
        check(got_valid[i] == 1, "raw slot valid flag");
    }

    // 3) host oracle: decode -> requant g64 -> decode -> int8 re-encode with
    //    upper-bound group scale (mirror cold_i8_decode_row)
    double num = 0.0, den = 0.0;
    int byte_mismatch = 0;
    int scale_mismatch = 0;
    for (int head = 0; head < kKvHeads; ++head) {
        for (int row = 0; row < kPageRows; ++row) {
            for (int g = 0; g < 4; ++g) {
                const float sh = half_to_float(
                    src_scales[(static_cast<std::size_t>(head) * kPageRows + row) * 4 + g]);
                float vals[64];
                for (int i = 0; i < 64; ++i) {
                    vals[i] = static_cast<float>(
                                  src_codes[static_cast<std::size_t>(head) * kI8CodeB +
                                            row * kHeadDim + g * 64 + i]) * sh;
                }
                // requant g64 (matches entropy_cold_requant oracle)
                float amax = 0.0f;
                for (int i = 0; i < 64; ++i) { amax = std::fmax(amax, std::fabs(vals[i])); }
                const std::uint8_t sb = e4m3_rne(std::fmax(amax / 6.0f, 0x1p-9f));
                const float s = e4m3_to_f32(sb);
                float dec[64];
                for (int i = 0; i < 64; ++i) {
                    dec[i] = e2m1_to_f32(e2m1_code(vals[i] / s)) * s;
                }
                // upper-bound int8 re-encode (mirror device)
                float mx = 0.0f;
                for (int sub = 0; sub < 4; ++sub) {
                    // g64 group covers scale slots [g*4, g*4+4) of the row's
                    // g16 layout; requant wrote the same sb replicated.
                    mx = std::fmax(mx, e4m3_to_f32(sb));
                }
                const float scale = mx * 6.0f / 127.0f;
                const std::uint16_t want_scale = float_to_half(scale);
                const std::uint16_t got_scale =
                    got_scales[(static_cast<std::size_t>(head) * kPageRows + row) * 4 + g];
                if (want_scale != got_scale) { ++scale_mismatch; }
                const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
                const float got_sh = half_to_float(got_scale);
                for (int i = 0; i < 64; ++i) {
                    int c = static_cast<int>(std::nearbyint(dec[i] * inv));
                    c = std::max(-127, std::min(127, c));
                    const std::int8_t got = got_codes[static_cast<std::size_t>(head) * kI8CodeB +
                                                      row * kHeadDim + g * 64 + i];
                    if (got != static_cast<std::int8_t>(c)) { ++byte_mismatch; }
                    const float restored = static_cast<float>(got) * got_sh;
                    num += static_cast<double>(restored - vals[i]) * (restored - vals[i]);
                    den += static_cast<double>(vals[i]) * vals[i];
                }
            }
        }
    }
    const double nmse = den > 0.0 ? num / den : 0.0;
    check(byte_mismatch == 0, "restore codes match oracle");
    check(scale_mismatch == 0, "restore scales match oracle");
    check(nmse < 0.05, "roundtrip NMSE bound");  // synthetic spiky data; real planes measured 0.012
    std::printf("cold_i8 roundtrip: byte_mismatch=%d scale_mismatch=%d NMSE=%.5f\n",
                byte_mismatch, scale_mismatch, nmse);

    if (failures == 0) { std::printf("cold_i8: all tests passed\n"); }
    return failures == 0 ? 0 : 1;
}
