#include "ninfer/ops/entropy_cold_requant.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeadDim     = 256;
constexpr int kPageRows    = 64;
constexpr int kKvHeads     = 4;
constexpr int kNvfp4CodeB  = kHeadDim / 2 * kPageRows;      // 8192 per head-page
constexpr int kNvfp4ScaleB = kHeadDim / 16 * kPageRows;     // 1024
constexpr int kInt8CodeB   = kHeadDim * kPageRows;          // 16384
constexpr int kInt8ScaleB  = kHeadDim / 64 * 2 * kPageRows; // 512 (fp16)

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

float iso3_to_f32(std::uint8_t code) {
    const float mag = static_cast<float>(code & 0x7);
    return (code & 0x8) != 0 ? -mag : mag;
}

std::uint8_t iso3_code(float value, float scale) {
    float mag = std::roundf(std::fabs(value) / scale);  // device uses roundf
    if (mag > 7.0f) { mag = 7.0f; }
    if (mag < 0.0f) { mag = 0.0f; }
    std::uint8_t code = static_cast<std::uint8_t>(mag);
    if (value < 0.0f && code != 0) { code |= 0x08u; }
    return code;
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

struct PageValues {
    std::vector<float> values;
    std::vector<std::uint8_t> nvfp4_codes;
    std::vector<std::uint8_t> nvfp4_scales;
    std::vector<std::uint8_t> iso3_codes;
    std::vector<std::uint8_t> iso3_scales;
    std::vector<std::int8_t> int8_codes;
    std::vector<std::uint16_t> int8_scales;
};

PageValues make_page(std::mt19937& rng) {
    PageValues page;
    page.values.resize(static_cast<std::size_t>(kKvHeads) * kPageRows * kHeadDim);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::uniform_real_distribution<float> level(0.001f, 8.0f);
    for (int head = 0; head < kKvHeads; ++head) {
        for (int row = 0; row < kPageRows; ++row) {
            const float amp = level(rng);
            for (int d = 0; d < kHeadDim; ++d) {
                float v = amp * noise(rng);
                if (((head * 131 + row * 17 + d) % 4096) == 0) { v *= 64.0f; }
                if (head == kKvHeads - 1 && row < 2) { v = 0.0f; }
                page.values[(static_cast<std::size_t>(head) * kPageRows + row) * kHeadDim + d] = v;
            }
        }
    }

    page.nvfp4_codes.assign(static_cast<std::size_t>(kKvHeads) * kNvfp4CodeB, 0);
    page.nvfp4_scales.assign(static_cast<std::size_t>(kKvHeads) * kNvfp4ScaleB, 0);
    page.iso3_codes.assign(static_cast<std::size_t>(kKvHeads) * kNvfp4CodeB, 0);
    page.iso3_scales.assign(static_cast<std::size_t>(kKvHeads) * kNvfp4ScaleB, 0);
    page.int8_codes.assign(static_cast<std::size_t>(kKvHeads) * kInt8CodeB, 0);
    page.int8_scales.assign(static_cast<std::size_t>(kKvHeads) * kPageRows * (kHeadDim / 64), 0);

    for (int head = 0; head < kKvHeads; ++head) {
        for (int row = 0; row < kPageRows; ++row) {
            const std::size_t vrow =
                (static_cast<std::size_t>(head) * kPageRows + row) * kHeadDim;
            for (int g = 0; g < kHeadDim / 16; ++g) {
                float amax = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    amax = std::fmax(amax, std::fabs(page.values[vrow + g * 16 + i]));
                }
                const std::uint8_t kb = e4m3_rne(std::fmax(amax / 6.0f, 0x1p-9f));
                page.nvfp4_scales[static_cast<std::size_t>(head) * kNvfp4ScaleB +
                                  row * (kHeadDim / 16) + g] = kb;
                const float ks = e4m3_to_f32(kb);
                const std::uint8_t vb = e4m3_rne(std::fmax(amax / 7.0f, 0x1p-9f));
                page.iso3_scales[static_cast<std::size_t>(head) * kNvfp4ScaleB +
                                 row * (kHeadDim / 16) + g] = vb;
                const float vs = e4m3_to_f32(vb);
                for (int i = 0; i < 16; i += 2) {
                    page.nvfp4_codes[static_cast<std::size_t>(head) * kNvfp4CodeB +
                                     row * (kHeadDim / 2) + g * 8 + i / 2] =
                        static_cast<std::uint8_t>(
                            e2m1_code(page.values[vrow + g * 16 + i] / ks) |
                            (e2m1_code(page.values[vrow + g * 16 + i + 1] / ks) << 4));
                    page.iso3_codes[static_cast<std::size_t>(head) * kNvfp4CodeB +
                                    row * (kHeadDim / 2) + g * 8 + i / 2] =
                        static_cast<std::uint8_t>(
                            iso3_code(page.values[vrow + g * 16 + i], vs) |
                            (iso3_code(page.values[vrow + g * 16 + i + 1], vs) << 4));
                }
            }
            for (int g = 0; g < kHeadDim / 64; ++g) {
                float amax = 0.0f;
                for (int i = 0; i < 64; ++i) {
                    amax = std::fmax(amax, std::fabs(page.values[vrow + g * 64 + i]));
                }
                const float s = std::fmax(amax / 127.0f, 1e-30f);
                const std::uint16_t sb = float_to_half(s);
                page.int8_scales[(static_cast<std::size_t>(head) * kPageRows + row) *
                                     (kHeadDim / 64) + g] = sb;
                const float sh = half_to_float(sb);
                for (int i = 0; i < 64; ++i) {
                    float q = std::nearbyint(page.values[vrow + g * 64 + i] / sh);
                    if (q > 127.0f) { q = 127.0f; }
                    if (q < -127.0f) { q = -127.0f; }
                    page.int8_codes[static_cast<std::size_t>(head) * kInt8CodeB +
                                    row * kHeadDim + g * 64 + i] = static_cast<std::int8_t>(q);
                }
            }
        }
    }
    return page;
}

void requant_oracle(const PageValues& page, int mode, std::vector<std::uint8_t>& out_codes,
                    std::vector<std::uint8_t>& out_scales) {
    // mode: 0 = nvfp4 K source, 1 = int8 source, 2 = iso3 V source
    const bool iso3_out = mode == 2;
    const float scale_div = iso3_out ? 7.0f : 6.0f;
    out_codes.assign(static_cast<std::size_t>(kKvHeads) * kNvfp4CodeB, 0);
    out_scales.assign(static_cast<std::size_t>(kKvHeads) * kNvfp4ScaleB, 0);
    std::vector<float> dec(static_cast<std::size_t>(kKvHeads) * kPageRows * kHeadDim);
    for (int head = 0; head < kKvHeads; ++head) {
        for (int row = 0; row < kPageRows; ++row) {
            for (int g64 = 0; g64 < kHeadDim / 64; ++g64) {
                if (mode == 1) {
                    const std::uint16_t sb =
                        page.int8_scales[(static_cast<std::size_t>(head) * kPageRows + row) *
                                             (kHeadDim / 64) + g64];
                    const float s = half_to_float(sb);
                    for (int i = 0; i < 64; ++i) {
                        const std::int8_t c =
                            page.int8_codes[static_cast<std::size_t>(head) * kInt8CodeB +
                                            row * kHeadDim + g64 * 64 + i];
                        dec[(static_cast<std::size_t>(head) * kPageRows + row) * kHeadDim +
                            g64 * 64 + i] = static_cast<float>(c) * s;
                    }
                } else {
                    for (int i = 0; i < 64; ++i) {
                        const int d = g64 * 64 + i;
                        const std::uint8_t byte =
                            (mode == 0 ? page.nvfp4_codes : page.iso3_codes)
                                [static_cast<std::size_t>(head) * kNvfp4CodeB +
                                 row * (kHeadDim / 2) + (d >> 1)];
                        const std::uint8_t nib =
                            (d & 1) != 0 ? static_cast<std::uint8_t>(byte >> 4)
                                         : static_cast<std::uint8_t>(byte & 0x0F);
                        const float s = e4m3_to_f32(
                            (mode == 0 ? page.nvfp4_scales : page.iso3_scales)
                                [static_cast<std::size_t>(head) * kNvfp4ScaleB +
                                 row * (kHeadDim / 16) + (d >> 4)]);
                        dec[(static_cast<std::size_t>(head) * kPageRows + row) * kHeadDim + d] =
                            (mode == 0 ? e2m1_to_f32(nib) : iso3_to_f32(nib)) * s;
                    }
                }
            }
        }
    }
    for (int head = 0; head < kKvHeads; ++head) {
        for (int row = 0; row < kPageRows; ++row) {
            for (int g64 = 0; g64 < kHeadDim / 64; ++g64) {
                float amax = 0.0f;
                for (int i = 0; i < 64; ++i) {
                    amax = std::fmax(
                        amax,
                        std::fabs(dec[(static_cast<std::size_t>(head) * kPageRows + row) *
                                          kHeadDim + g64 * 64 + i]));
                }
                const std::uint8_t sb = e4m3_rne(std::fmax(amax / scale_div, 0x1p-9f));
                const float s = e4m3_to_f32(sb);
                for (int i = 0; i < 64; i += 2) {
                    const int d0 = g64 * 64 + i;
                    const float v0 =
                        dec[(static_cast<std::size_t>(head) * kPageRows + row) * kHeadDim + d0];
                    const float v1 = dec[(static_cast<std::size_t>(head) * kPageRows + row) *
                                             kHeadDim + d0 + 1];
                    const std::uint8_t lo = iso3_out ? iso3_code(v0, s) : e2m1_code(v0 / s);
                    const std::uint8_t hi = iso3_out ? iso3_code(v1, s) : e2m1_code(v1 / s);
                    out_codes[static_cast<std::size_t>(head) * kNvfp4CodeB +
                              row * (kHeadDim / 2) + (d0 >> 1)] =
                        static_cast<std::uint8_t>(lo | (hi << 4));
                }
                for (int r = 0; r < 4; ++r) {
                    out_scales[static_cast<std::size_t>(head) * kNvfp4ScaleB +
                               row * (kHeadDim / 16) + g64 * 4 + r] = sb;
                }
            }
        }
    }
}

int check(bool ok, const char* what, int& failures) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    std::mt19937 rng(20260830);
    int failures = 0;
    const char* tags[3] = {"nvfp4-g16 K", "int8-g64", "iso3-g16 V"};

    for (int mode = 0; mode < 3; ++mode) {
        PageValues page = make_page(rng);

        std::vector<float> stored(static_cast<std::size_t>(kKvHeads) * kPageRows * kHeadDim);
        for (int head = 0; head < kKvHeads; ++head) {
            for (int row = 0; row < kPageRows; ++row) {
                if (mode == 1) {
                    for (int g = 0; g < kHeadDim / 64; ++g) {
                        const float s = half_to_float(
                            page.int8_scales[(static_cast<std::size_t>(head) * kPageRows + row) *
                                                 (kHeadDim / 64) + g]);
                        for (int i = 0; i < 64; ++i) {
                            stored[(static_cast<std::size_t>(head) * kPageRows + row) * kHeadDim +
                                   g * 64 + i] =
                                static_cast<float>(
                                    page.int8_codes[static_cast<std::size_t>(head) * kInt8CodeB +
                                                    row * kHeadDim + g * 64 + i]) * s;
                        }
                    }
                    continue;
                }
                for (int g = 0; g < kHeadDim / 16; ++g) {
                    const std::vector<std::uint8_t>& codes =
                        mode == 0 ? page.nvfp4_codes : page.iso3_codes;
                    const std::vector<std::uint8_t>& scales =
                        mode == 0 ? page.nvfp4_scales : page.iso3_scales;
                    const float s = e4m3_to_f32(
                        scales[static_cast<std::size_t>(head) * kNvfp4ScaleB +
                               row * (kHeadDim / 16) + g]);
                    for (int i = 0; i < 16; ++i) {
                        const int d = g * 16 + i;
                        const std::uint8_t byte =
                            codes[static_cast<std::size_t>(head) * kNvfp4CodeB +
                                  row * (kHeadDim / 2) + (d >> 1)];
                        const std::uint8_t nib =
                            (d & 1) != 0 ? static_cast<std::uint8_t>(byte >> 4)
                                         : static_cast<std::uint8_t>(byte & 0x0F);
                        stored[(static_cast<std::size_t>(head) * kPageRows + row) * kHeadDim + d] =
                            (mode == 0 ? e2m1_to_f32(nib) : iso3_to_f32(nib)) * s;
                    }
                }
            }
        }

        std::vector<std::uint8_t> expect_codes, expect_scales;
        requant_oracle(page, mode, expect_codes, expect_scales);

        const bool int8_source = mode == 1;
        GuardedDeviceBuffer dsrc_codes(int8_source
                                           ? static_cast<std::size_t>(kKvHeads) * kInt8CodeB
                                           : static_cast<std::size_t>(kKvHeads) * kNvfp4CodeB);
        GuardedDeviceBuffer dsrc_scales(int8_source
                                            ? static_cast<std::size_t>(kKvHeads) * kInt8ScaleB
                                            : static_cast<std::size_t>(kKvHeads) * kNvfp4ScaleB);
        GuardedDeviceBuffer ddst_codes(static_cast<std::size_t>(kKvHeads) * kNvfp4CodeB);
        GuardedDeviceBuffer ddst_scales(static_cast<std::size_t>(kKvHeads) * kNvfp4ScaleB);

        if (int8_source) {
            dsrc_codes.copy_from_host(page.int8_codes.data(), dsrc_codes.bytes());
            dsrc_scales.copy_from_host(page.int8_scales.data(), dsrc_scales.bytes());
        } else if (mode == 0) {
            dsrc_codes.copy_from_host(page.nvfp4_codes.data(), dsrc_codes.bytes());
            dsrc_scales.copy_from_host(page.nvfp4_scales.data(), dsrc_scales.bytes());
        } else {
            dsrc_codes.copy_from_host(page.iso3_codes.data(), dsrc_codes.bytes());
            dsrc_scales.copy_from_host(page.iso3_scales.data(), dsrc_scales.bytes());
        }

        ops::entropy_cold_requant_raw(
            static_cast<const std::uint8_t*>(dsrc_codes.data()),
            static_cast<const std::uint8_t*>(dsrc_scales.data()),
            mode == 0   ? ops::EntropyColdRequantMode::Nvfp4G16
            : mode == 1 ? ops::EntropyColdRequantMode::Int8G64
                        : ops::EntropyColdRequantMode::Iso3VG16,
            kKvHeads, 1, static_cast<std::uint8_t*>(ddst_codes.data()),
            static_cast<std::uint8_t*>(ddst_scales.data()), nullptr);
        cuda_synchronize();

        std::vector<std::uint8_t> got_codes(static_cast<std::size_t>(kKvHeads) * kNvfp4CodeB);
        std::vector<std::uint8_t> got_scales(static_cast<std::size_t>(kKvHeads) * kNvfp4ScaleB);
        ddst_codes.copy_to_host(got_codes.data(), ddst_codes.bytes());
        ddst_scales.copy_to_host(got_scales.data(), ddst_scales.bytes());

        const char* tag = tags[mode];
        const bool codes_ok  = got_codes == expect_codes;
        const bool scales_ok = got_scales == expect_scales;
        check(codes_ok, (std::string(tag) + " requant codes match oracle").c_str(), failures);
        check(scales_ok, (std::string(tag) + " requant scales match oracle").c_str(), failures);

        double num = 0.0, den = 0.0;
        const bool iso3_out = mode == 2;
        for (std::size_t i = 0; i < stored.size(); ++i) {
            const int head = static_cast<int>(i / (static_cast<std::size_t>(kPageRows) * kHeadDim));
            const int row  = static_cast<int>((i / kHeadDim) % kPageRows);
            const int d    = static_cast<int>(i % kHeadDim);
            const std::uint8_t byte =
                got_codes[static_cast<std::size_t>(head) * kNvfp4CodeB + row * (kHeadDim / 2) +
                          (d >> 1)];
            const std::uint8_t nib =
                (d & 1) != 0 ? static_cast<std::uint8_t>(byte >> 4)
                             : static_cast<std::uint8_t>(byte & 0x0F);
            const float s = e4m3_to_f32(
                got_scales[static_cast<std::size_t>(head) * kNvfp4ScaleB + row * (kHeadDim / 16) +
                           (d >> 4)]);
            const float out = iso3_out ? iso3_to_f32(nib) * s : e2m1_to_f32(nib) * s;
            num += static_cast<double>(out - stored[i]) * (out - stored[i]);
            den += static_cast<double>(stored[i]) * stored[i];
        }
        const double nmse = den > 0.0 ? num / den : 0.0;
        check(nmse < 0.03, (std::string(tag) + " requant NMSE bound").c_str(), failures);
        std::printf("[%s] requant NMSE vs stored = %.5f (codes_ok=%d scales_ok=%d)\n", tag, nmse,
                    codes_ok ? 1 : 0, scales_ok ? 1 : 0);
    }

    if (failures == 0) { std::printf("entropy_cold_requant: all tests passed\n"); }
    return failures == 0 ? 0 : 1;
}
