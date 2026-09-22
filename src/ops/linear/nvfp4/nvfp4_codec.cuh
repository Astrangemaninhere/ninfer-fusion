#pragma once

#include "ops/common/fp4_codec.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

// DECODE GATE -- and it is not a performance opinion, it is the vendor header's own rule.
//
// The software decoder this replaces is ~13 SASS instructions PER NIBBLE (2x I2FP.F32.U32,
// a variable shift, 2x FMUL, 3x LOP3, an FSEL and a sign fix). nvfp4_gemv_kernel calls it
// 1088 times per thread and each call decodes 2 nibbles, so on sm_120a it is the largest
// single block of that kernel's instruction stream.
//
// The ONE-instruction form is not new to this tree, it is in the toolkit: cuda_fp4.hpp:428
// emits `cvt.rn.f16x2.e2m1x2` to turn one packed code byte into an f16x2, and gates that on
// cuda_fp8.h:88-96 -- __CUDA_ARCH__ >= 1000 AND a family-specific target AND family in
// [1000,1300). NINFER_NVFP4_HAS_CVT_F16X2_E2M1X2 is that gate, spelled once, so no target
// that built before this patch changes behaviour: where it is 0 the software codec below is
// used verbatim as before.
//
// THE WIDEN IS EXACT, NOT APPROXIMATE. Every e2m1 magnitude {0, .5, 1, 1.5, 2, 3, 4, 6} is
// representable in f16, so f16 -> f32 introduces no rounding and the pair returned here is
// bit-identical to decode_fp4_e2m1x2. That is not an argument, it is a measurement:
// dl/nvfp4gemv/probe/dec_probe.cu compares the two over all 256 byte values, and the same
// binary carries a lo/hi-swapped arm that MUST go RED, so a green arm proves the comparison
// could have failed.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
    ((__CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000) && (__CUDA_ARCH_FAMILY_SPECIFIC__ < 1300))
#define NINFER_NVFP4_HAS_CVT_F16X2_E2M1X2 1
#else
#define NINFER_NVFP4_HAS_CVT_F16X2_E2M1X2 0
#endif

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
#if NINFER_NVFP4_HAS_CVT_F16X2_E2M1X2
    // low nibble lands in the LOW half of the f16x2, which is the layout decode_fp4_e2m1x2
    // returns (lo = storage & 0xF) -- same convention as the toolkit header's
    // __nv_cvt_fp4x2_to_halfraw2, which memcpy's the same register shape.
    //
    // The spelling is not decoration, it is MEASURED with ptxas 13.3
    // (dl/nvfp4gemv/s29_ptxas.out, s30_ptxas2.out, s31.out):
    //   * `mov.b16 {b0,b1}, <32-bit reg>`                     -> Arguments mismatch
    //   * `mov.b16 {b0,b1}, <the .s16 reg nvcc picks for "h">` -> Arguments mismatch
    //   * `mov.b32 {b0,b1,b2,b3}, %r` with `%r` a plain .b32 asm operand -> ASSEMBLES,
    //     and lowers to the single SASS instruction F2FP.F16.E2M1.UNPACK_B.
    // That last form is used, so nvcc has no chance to substitute a signed register.
    std::uint32_t halves = 0;
    const std::uint32_t byte_bits = static_cast<std::uint32_t>(storage);
    asm("{\n"
        " .reg .b8 __$b0, __$b1, __$b2, __$b3;\n"
        " mov.b32 {__$b0, __$b1, __$b2, __$b3}, %1;\n"
        " cvt.rn.f16x2.e2m1x2 %0, __$b0;\n"
        "}\n"
        : "=r"(halves)
        : "r"(byte_bits));
    return __half22float2(ninfer::ops::half2_from_bits(halves));
#else
    float lo = 0.0F;
    float hi = 0.0F;
    decode_fp4_e2m1x2(storage, lo, hi);
    return make_float2(lo, hi);
#endif
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

// The ACTIVATION codec: 16 f32 in, 32 bits of packed e2m1 out.
//
// TARGET GATE, and it is not a number. MEASURED (hand-written PTX + ptxas): the
// `cvt.rn.satfinite.e2m1x2.f32` this form is built on is rejected by sm_70, sm_75,
// sm_80, sm_86, sm_89, sm_90 AND by the plain sm_100, and accepted only where the
// feature macro is defined (sm_100a/103a/110a/120a/121a and the five family targets).
// NINFER_FP4_HAS_CVT_E2M1X2 (ops/common/fp4_codec.cuh) is that measured fact, keyed on
// the __CUDA_ARCH_FEAT_* / __CUDA_ARCH_FAMILY_SPECIFIC__ macros exactly as
// ops/common/mma.cuh keys its Blackwell forms.
//
// Below that boundary the portable encoder is used. It produces the SAME bytes -- the
// same round-to-nearest-even tie resolution and the same satfinite clamp -- and the two
// are compared bit-for-bit by tests/test_fp4_codec.cpp on any machine that has a card
// where the instruction exists. Without a gate here the whole nvfp4_w4a4 TU is red for
// every rung below sm_100a, which is what the third wall was.
__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
#if NINFER_FP4_HAS_CVT_E2M1X2
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
#else
    const Fp4Pair pairs[8] = {{values[0].x, values[0].y}, {values[1].x, values[1].y},
                              {values[2].x, values[2].y}, {values[3].x, values[3].y},
                              {values[4].x, values[4].y}, {values[5].x, values[5].y},
                              {values[6].x, values[6].y}, {values[7].x, values[7].y}};
    pack_nvfp4_e2m1x16_portable(pairs, codes_lo, codes_hi);
#endif
}

__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x * input_scale_divisor, decoded_scale);
        values[pair].y = __fdiv_rn(values[pair].y * input_scale_divisor, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

} // namespace ninfer::ops::detail
