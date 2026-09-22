// ub_float_cast_probe.cpp -- the RUNTIME half of the guard against the
// `static_cast<int>(rintf(...))` undefined-behaviour class on ninfer-fusion's
// quantisation paths.
//
// THE DEFECT. `static_cast<int>(rintf(x))` is UNDEFINED BEHAVIOUR when `x` is NaN or
// outside `int`'s range. The engine's quantisation writers clamp AFTER the cast, so
// the clamp does not protect the conversion:
//
//     max(kMin, min(kMax, static_cast<int>(rintf(x * inv))))   <- the cast already ran
//
// `__float2int_rn` performs the same round-to-nearest-even and is DEFINED everywhere
// (PTX cvt.rni.s32.f32 saturates at INT_MIN/INT_MAX and yields 0 for NaN), so it is
// bit-identical for every in-range input and merely defined where the cast was not.
// That is the whole fix, and the tree's own K/V writers already use it
// (src/ops/kernel/gqa_attention_kv_quant.cuh:89,266).
//
// WHY A REPLICA. The condition is a property of the C++ conversion, and the CUDA
// device path cannot be sanitized with `-fsanitize=float-cast-overflow` (UBSan does
// not instrument device code and compute-sanitizer has no float-cast-overflow tool).
// This file therefore carries the SAME EXPRESSIONS, transcribed from the tree, with
// the CUDA-only intrinsics replaced by their defined host equivalents. The sibling
// e8_ub_float_cast_guard.cpp holds the SOURCE half, which reads the real headers.
//
// BUILD TWICE, with IDENTICAL sanitizer flags -- the only difference is the spelling:
//
//   RED   (must abort, non-zero):
//     g++ -std=c++20 -O1 -g -fsanitize=float-cast-overflow \
//         -fno-sanitize-recover=float-cast-overflow -DNINFER_UB_PROBE_RED \
//         tools/e8_verify/ub_float_cast_probe.cpp -o /tmp/ubprobe_red
//
//   GREEN (must be silent, exit 0):
//     g++ -std=c++20 -O1 -g -fsanitize=float-cast-overflow \
//         -fno-sanitize-recover=float-cast-overflow \
//         tools/e8_verify/ub_float_cast_probe.cpp -o /tmp/ubprobe_green
//
//   ubprobe_red A      -> the pre-fix K/V i4 expression   (fires)
//   ubprobe_red B      -> the E8 rintf-then-cast sites    (fires)
//   ubprobe_green D    -> the fixed spellings, SAME inputs (silent)
//   ubprobe_green C    -> nvfp4 e4m3 :160 unreachability  (silent, measured)
//
// HOST-ONLY: no CUDA, no cmake, no build/.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <limits>

namespace {

// Defined host equivalent of CUDA's __float2int_rn. PTX cvt.rni.s32.f32 rounds to
// nearest-even and SATURATES at INT_MIN/INT_MAX; NaN converts to 0. Every branch is
// taken on the FLOAT, so no conversion below ever sees an out-of-range value.
int float2int_rn_sat(float v) {
    if (v != v) { return 0; }
    if (v >= 2147483648.0f) { return 2147483647; }
    if (v < -2147483648.0f) { return -2147483647 - 1; }
    return static_cast<int>(std::rint(v));
}

std::uint32_t bit_cast_u32(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }

constexpr int kI4Min = -7, kI4Max = 7;

// ------------------------------------------------ the PRE-FIX spelling (RED control)
// Only compiled under -DNINFER_UB_PROBE_RED, so a default build of this file contains
// no UB. This is the verbatim shape of the twelve K/V i4 sites before the fix, quoted
// in src/ops/kernel/gqa_attention_kv_quant.cuh:246.
#if defined(NINFER_UB_PROBE_RED)
int i4_code_UB(float x, float inv_scale) {
    return std::max(kI4Min, std::min(kI4Max, static_cast<int>(std::rint(x * inv_scale))));
}
int e8_d8_sum_UB(const float x[8]) {
    int sum_f = 0;
    for (int i = 0; i < 8; ++i) {
        const float f_x = std::rint(x[i]);
        sum_f += static_cast<int>(f_x);
    }
    return sum_f;
}
int e8_rad_idx_UB(float r_rel) {
    std::uint32_t rad_idx = 0;
    if (r_rel >= 0.08f) {
        const float log_val = 3.0f * (std::log(r_rel) * 1.4426950408889634f) + 8.0f;
        const int q_rad = static_cast<int>(std::rint(log_val));
        rad_idx = static_cast<std::uint32_t>(q_rad < 1 ? 1 : (q_rad > 15 ? 15 : q_rad));
    }
    return static_cast<int>(rad_idx);
}
#endif

// ------------------------------------------------ the FIXED spelling (GREEN)
// This is what src/ops/kernel/e8_root_codec.cuh:176,260 now do.
int i4_code_FIXED(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return 0; }
    return std::max(kI4Min, std::min(kI4Max, float2int_rn_sat(x * inv_scale)));
}
int e8_d8_sum_FIXED(const float x[8]) {
    int sum_f = 0;
    for (int i = 0; i < 8; ++i) { sum_f += float2int_rn_sat(std::rint(x[i])); }
    return sum_f;
}
int e8_rad_idx_FIXED(float r_rel) {
    std::uint32_t rad_idx = 0;
    if (r_rel >= 0.08f) {
        const float log_val = 3.0f * (std::log(r_rel) * 1.4426950408889634f) + 8.0f;
        const int q_rad = float2int_rn_sat(std::rint(log_val));
        rad_idx = static_cast<std::uint32_t>(q_rad < 1 ? 1 : (q_rad > 15 ? 15 : q_rad));
    }
    return static_cast<int>(rad_idx);
}

// ------------------------------------------------ the nvfp4 e4m3 branch, VERBATIM shape
// src/ops/kernel/gqa_attention_kv_nvfp4.cuh:146-182. Returns -999 for the NaN guard,
// -1 for the exponent>=16 saturation, -2 for the normal branch, else the denormal
// mantissa computed at :160 by `static_cast<int>(rintf(ax * 512.0f))`.
int nvfp4_e4m3_denorm_mantissa(float x) {
    if (x != x) { return -999; }                                  // :147
    const float ax = std::fabs(x);
    const std::uint32_t abits = bit_cast_u32(ax);
    const int exponent = static_cast<int>((abits >> 23) & 0xffu) - 127 + 7;
    if (exponent >= 16) { return -1; }                            // :154
    if (exponent <= 0) {
        return static_cast<int>(std::rint(ax * 512.0f));           // :160  (UB spelling)
    }
    return -2;
}

}  // namespace

int main(int argc, char** argv) {
    const char* only = (argc > 1) ? argv[1] : "";
    const float nan_v = std::numeric_limits<float>::quiet_NaN();
    const float inf_v = std::numeric_limits<float>::infinity();
    const float big = 1e30f;

    struct Case { const char* name; float x; float inv; };
    const Case cases[] = {
        {"finite-in-range",       1.5f,  4.0f},
        {"finite-out-of-range",   big,   4.0f},
        {"nan",                   nan_v, 4.0f},
        {"posinf",                inf_v, 4.0f},
        {"neginf",               -inf_v, 4.0f},
        {"inf-times-zero-scale",  inf_v, 0.0f},
    };
    const int nc = 6;

    float blk[8], blk_nan[8];
    for (int i = 0; i < 8; ++i) { blk[i] = (i == 3) ? big : float(i); blk_nan[i] = (i == 3) ? nan_v : float(i); }

#if defined(NINFER_UB_PROBE_RED)
    std::printf("=== ub_float_cast_probe: RED (pre-fix spelling) group '%s' ===\n", *only ? only : "ALL");
    if (!*only || std::strcmp(only, "A") == 0) {
        for (int i = 0; i < nc; ++i) {
            std::printf("[A] i4_code_UB  %-22s -> %d\n", cases[i].name, i4_code_UB(cases[i].x, cases[i].inv));
        }
        std::printf("[A] i4_code_UB  nan/inf/out-of-range all reach __float2int-style cast\n");
    }
    if (!*only || std::strcmp(only, "B") == 0) {
        std::printf("[B] e8_d8_sum_UB  huge-block -> %d\n", e8_d8_sum_UB(blk));
        std::printf("[B] e8_d8_sum_UB  nan-block  -> %d\n", e8_d8_sum_UB(blk_nan));
        std::printf("[B] e8_rad_idx_UB r_rel=inf  -> %d\n", e8_rad_idx_UB(inf_v));
        std::printf("[B] e8_rad_idx_UB r_rel=nan  -> %d (guard rejects)\n", e8_rad_idx_UB(nan_v));
    }
    std::printf("=== RED: sanitizer did NOT fire -- the spelling is not what this probe thinks ===\n");
    return 0;
#else
    std::printf("=== ub_float_cast_probe: GREEN (fixed spelling) group '%s' ===\n", *only ? only : "ALL");
    if (!*only || std::strcmp(only, "D") == 0) {
        for (int i = 0; i < nc; ++i) {
            std::printf("[D] i4_code_FIXED %-22s -> %d\n", cases[i].name, i4_code_FIXED(cases[i].x, cases[i].inv));
        }
        std::printf("[D] e8_d8_sum_FIXED  huge-block -> %d\n", e8_d8_sum_FIXED(blk));
        std::printf("[D] e8_d8_sum_FIXED  nan-block  -> %d\n", e8_d8_sum_FIXED(blk_nan));
        std::printf("[D] e8_rad_idx_FIXED r_rel=inf  -> %d\n", e8_rad_idx_FIXED(inf_v));
        std::printf("[D] e8_rad_idx_FIXED r_rel=nan  -> %d\n", e8_rad_idx_FIXED(nan_v));
    }
    if (!*only || std::strcmp(only, "C") == 0) {
        // Sweep EVERY float that can enter nvfp4's denorm branch: the branch is entered
        // iff the exponent field is <= 120, so all 121 such exponent fields, 64 mantissa
        // samples each, plus the non-finite patterns.
        double arg_max = 0.0;
        long entered = 0, sat = 0, nanguard = 0;
        for (int em = 0; em <= 120; ++em) {
            for (int mi = 0; mi < 64; ++mi) {
                const std::uint32_t m = static_cast<std::uint32_t>(mi) * 0x0200000u;
                std::uint32_t bits = (static_cast<std::uint32_t>(em) << 23) | (m & 0x7fffffu);
                float f; std::memcpy(&f, &bits, 4);
                const int r = nvfp4_e4m3_denorm_mantissa(f);
                if (r >= 0) { ++entered; const double a = std::fabs(double(f)) * 512.0; if (a > arg_max) arg_max = a; }
            }
        }
        const float probes[] = {big, -big, 1e20f, -1e20f, nan_v, inf_v, -inf_v, 0.0f, 1e-45f, 0.015625f};
        for (float p : probes) {
            const int r = nvfp4_e4m3_denorm_mantissa(p);
            if (r == -999) ++nanguard; else if (r == -1) ++sat;
            else if (r >= 0) { ++entered; const double a = std::fabs(double(p)) * 512.0; if (a > arg_max) arg_max = a; }
        }
        std::printf("[C] denorm branch entered %ld times; max argument to the cast = %.6g\n", entered, arg_max);
        std::printf("[C] NaN early-returns %ld, exponent>=16 saturations %ld\n", nanguard, sat);
        std::printf("[C] claim: max argument < 8 (the branch requires |x| < 2^-6), so the\n"
                    "    conversion at :160 cannot be out of range for any float.\n");
        if (!(arg_max < 8.0)) { std::printf("[C] CLAIM FALSIFIED: argument reached %.6g\n", arg_max); return 1; }
    }
    if (!*only || std::strcmp(only, "E") == 0) {
        // BIT-IDENTITY of the fix. The claim being measured is that __float2int_rn is NOT
        // a behaviour change for any input the old expression handled CORRECTLY, i.e. every
        // finite one. llrint() is exact and well-defined in this range (log_val <= 392), so
        // it is a trustworthy reference for round-to-nearest-even.
        long mism = 0, n = 0;
        double worst = 0.0;
        for (int i = 0; i <= 400000; ++i) {
            // r_rel from well below the 0.08f guard up to FLT_MAX, log-spaced.
            const double t = static_cast<double>(i) / 400000.0;
            const float r_rel = static_cast<float>(0.05 * std::pow(3.4e38 / 0.05, t));
            if (!(r_rel >= 0.08f)) { continue; }
            const float log_val = 3.0f * (std::log(r_rel) * 1.4426950408889634f) + 8.0f;
            const int got = float2int_rn_sat(std::rint(log_val));
            const int ref = static_cast<int>(std::llrint(static_cast<double>(log_val)));
            ++n;
            if (got != ref) { ++mism; if (std::fabs(double(got - ref)) > worst) worst = std::fabs(double(got - ref)); }
        }
        std::printf("[E] finite r_rel sweep: %ld values, %ld mismatches vs exact llrint (worst %g)\n", n, mism, worst);
        std::printf("[E] => the fixed spelling reproduces round-to-nearest-even exactly for every\n"
                    "    finite input, so it is not a behaviour change on any input the old\n"
                    "    expression handled correctly.\n");
        if (mism != 0) { std::printf("[E] CLAIM FALSIFIED: %ld mismatches\n", mism); return 1; }
    }
    std::printf("=== GREEN: sanitizer silent, exit 0 ===\n");
    return 0;
#endif
}
