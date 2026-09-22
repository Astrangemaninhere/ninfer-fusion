#pragma once

// ---------------------------------------------------------------------------
// fp4_codec.cuh -- the NON-BLACKWELL fp4 (e2m1) codec.
//
// WHY THIS EXISTS
// ---------------
// MEASURED (ptxas, both toolkits, hand-written PTX so the verdict is the assembler's
// and not a compiler's -- see the report's instruction x target matrix):
//
//   cvt.rn.satfinite.e2m1x2.f32   sm_70 X  sm_75 X  sm_80 X  sm_86 X  sm_89 X
//                                 sm_90 X  sm_100 X  sm_100a OK  sm_120a OK
//                                 (+ the 100f/103f/110f/120f/121f family targets)
//
// So the engine's ONLY fp4 ACTIVATION codec -- the eight `cvt.rn.satfinite.e2m1x2.f32`
// in ops/linear/nvfp4/nvfp4_codec.cuh:45-52 -- cannot be assembled for any target
// below sm_100a. That is the whole of the third wall: with the mma.cuh/memory.cuh
// guards in place, `ops/linear/nvfp4/nvfp4_w4a4.cu` on sm_75 still failed, and its
// 5744 ptxas errors collapsed to 49 lines of exactly ONE kind, that instruction.
//
// The replacement below is the same conversion written in fp arithmetic and integer
// bit work, with NO cvt.e2m1x2, so it assembles everywhere. It is deliberately
// `__host__ __device__`: the host side is what makes the codec testable on a machine
// with no GPU of the relevant capability, and it is what lets tests/test_fp4_codec.cpp
// check the portable encoder against the ISA's OWN lattice (an independent source)
// rather than against itself.
//
// WHY ROUND-TO-NEAREST-EVEN, AND HOW IT RELATES TO `cvt`
// -----------------------------------------------------
// `cvt.rn.satfinite.e2m1x2.f32` is round-to-nearest-EVEN on the code, saturating to the
// largest representable magnitude (6.0) instead of producing inf, and clamping NaN to
// that same magnitude. The portable form reproduces both properties by construction:
// the thresholds sit exactly on the midpoints of the e2m1 lattice, each tie resolves to
// the EVEN code, and the ladder saturates at code 7.
// THAT IT MATCHES IS TESTED, NOT ASSUMED: tests/test_fp4_codec.cpp pins the decode
// against the printed lattice and pins the encode at all seven midpoints. On a machine
// WITH an sm_100a/120a card the same test compares the two bit-for-bit; where no such
// card is present it reports that half as PENDING HARDWARE rather than passing silently
// (SKIP_RETURN_CODE 77 in tests/CMakeLists.txt).
//
// NOTE ON WHAT THE QPN FALLBACK ROUTE ACTUALLY NEEDS
// --------------------------------------------------
// The QPN W4A16 route consumes ALREADY-PACKED e2m1 weight codes and an fp16 activation,
// so it needs the DECODE half only: decode_fp4_e2m1x2(). The ENCODE half is needed by
// the W4A4 routes (activation quantisation), and it is what closes the sm_75/sm_86 wall
// for ops/linear/nvfp4/nvfp4_w4a4.cu. Both halves live here because a codec with one
// direction only is how the two drift apart.
// ---------------------------------------------------------------------------

#include <cstdint>

// A host-only compiler (for example the g++ pass over tests/test_fp4_codec.cpp, which
// includes this header precisely BECAUSE the codec is host-callable) has no __host__ /
// __device__ keywords. They degrade to nothing there, so the same source is usable from
// a GPU-less test binary and from device code, with one definition of the arithmetic.
#if !defined(__CUDACC__) && !defined(__CUDA_ARCH__)
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace ninfer::ops::detail {

// ---------------------------------------------------------------------------
// The e2m1 lattice, named once
// ---------------------------------------------------------------------------
//
// NVFP4 e2m1: bit 3 = sign, bits [2:1] = exponent (bias 1), bit 0 = mantissa.
//   exp 0 -> subnormal : value = 0.5 * mantissa        -> {0, 0.5}
//   exp 1 -> value = (1 + 0.5*m) * 2^0                 -> {1, 1.5}
//   exp 2 -> value = (1 + 0.5*m) * 2^1                 -> {2, 3}
//   exp 3 -> value = (1 + 0.5*m) * 2^2                 -> {4, 6}
//
// This array IS the independent reference tests/test_fp4_codec.cpp asserts against: it
// is the ISA's printed lattice, not a second copy of the decoder below.
inline constexpr float kFp4E2m1Lattice[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
inline constexpr float kFp4E2m1Max         = 6.0F;

// The signs of the raw float bits. A union is used rather than std::signbit so the
// function exists identically on the host and on the device without a library call, and
// rather than `1.0F/x` so its answer cannot depend on the compiler's treatment of -0.0F.
[[nodiscard]] __host__ __device__ inline bool fp4_sign_negative(float value) noexcept {
    union {
        float as_float;
        std::uint32_t as_bits;
    } pun;
    pun.as_float = value;
    return (pun.as_bits & 0x80000000u) != 0u;
}

// Decode one 4-bit e2m1 code (0..15; bit 3 is the sign) to float.
// Bit arithmetic only: no cvt instruction, no cuda_fp4.h, no table memory.
[[nodiscard]] __host__ __device__ inline float decode_fp4_e2m1(unsigned code) noexcept {
    const unsigned magnitude = code & 0x7u;
    const unsigned exponent  = (magnitude >> 1) & 0x3u;
    const unsigned mantissa  = magnitude & 0x1u;
    const float value = (exponent == 0u)
                            ? (0.5F * static_cast<float>(mantissa))
                            : ((1.0F + 0.5F * static_cast<float>(mantissa)) *
                               static_cast<float>(1u << (exponent - 1u)));
    return (code & 0x8u) != 0u ? -value : value;
}

// Decode the low and high nibbles of one storage byte the way the engine's packed
// layout defines them (low nibble = even k; see ops/linear/qpn/qpn_kernels.cuh's
// layout comment and ops/linear/nvfp4/nvfp4_w4a4_mma.cuh's fragment map).
__host__ __device__ inline void
decode_fp4_e2m1x2(std::uint8_t storage, float& lo, float& hi) noexcept {
    lo = decode_fp4_e2m1(static_cast<unsigned>(storage & 0x0Fu));
    hi = decode_fp4_e2m1(static_cast<unsigned>(storage >> 4));
}

// Encode one float to a 4-bit e2m1 code: round-to-nearest-EVEN on the code, saturating
// to the largest representable magnitude. NaN and +-inf map to the saturated magnitude,
// which is what `.satfinite` does.
//
// The seven comparisons below are the seven MIDPOINTS of the e2m1 lattice, written so
// that each tie goes to the EVEN code:
//   0.25 -> 0 (even) | 0.75 -> 2 | 1.25 -> 2 | 1.75 -> 4 | 2.5 -> 4 | 3.5 -> 6 | 5.0 -> 6
[[nodiscard]] __host__ __device__ inline unsigned encode_fp4_e2m1(float value) noexcept {
    const bool negative = fp4_sign_negative(value);
    float magnitude     = value < 0.0F ? -value : value;
    // NaN fails every ordered comparison, so it must be caught explicitly: `!(a <= b)`
    // is true for NaN, and a saturating convert returns the max magnitude for it.
    if (!(magnitude <= kFp4E2m1Max)) { magnitude = kFp4E2m1Max; }
    unsigned code;
    if (magnitude <= 0.25F) {
        code = 0u;
    } else if (magnitude < 0.75F) {
        code = 1u;
    } else if (magnitude <= 1.25F) {
        code = 2u;
    } else if (magnitude < 1.75F) {
        code = 3u;
    } else if (magnitude <= 2.5F) {
        code = 4u;
    } else if (magnitude < 3.5F) {
        code = 5u;
    } else if (magnitude <= 5.0F) {
        code = 6u;
    } else {
        code = 7u;
    }
    return negative ? (code | 0x8u) : code;
}

// Pack a (lo, hi) pair into one byte: lo in the LOW nibble, hi in the HIGH nibble.
// SAME OPERAND ORDER as the instruction this replaces:
//   asm: cvt.rn.satfinite.e2m1x2.f32 b, <hi>, <lo>
// (ops/linear/nvfp4/nvfp4_codec.cuh passes (values[i].y, values[i].x), so x lands in the
// low nibble -- which is what the packed layout's "low nibble = even k" requires.)
[[nodiscard]] __host__ __device__ inline std::uint8_t pack_fp4_e2m1x2(float lo,
                                                                     float hi) noexcept {
    return static_cast<std::uint8_t>((encode_fp4_e2m1(hi) << 4) | encode_fp4_e2m1(lo));
}

// The 16-value / 32-bit pack, drop-in for nvfp4_codec.cuh's pack_nvfp4_e2m1x16():
// same byte order, same result.
__host__ __device__ inline void
pack_nvfp4_e2m1x16_portable(const float (&values)[16], std::uint32_t& codes_lo,
                            std::uint32_t& codes_hi) noexcept {
    std::uint8_t bytes[8];
    for (unsigned pair = 0; pair < 8u; ++pair) {
        bytes[pair] = pack_fp4_e2m1x2(values[2u * pair], values[2u * pair + 1u]);
    }
    codes_lo = static_cast<std::uint32_t>(bytes[0]) |
               (static_cast<std::uint32_t>(bytes[1]) << 8) |
               (static_cast<std::uint32_t>(bytes[2]) << 16) |
               (static_cast<std::uint32_t>(bytes[3]) << 24);
    codes_hi = static_cast<std::uint32_t>(bytes[4]) |
               (static_cast<std::uint32_t>(bytes[5]) << 8) |
               (static_cast<std::uint32_t>(bytes[6]) << 16) |
               (static_cast<std::uint32_t>(bytes[7]) << 24);
}

// The (x, y) pair shape nvfp4_codec.cuh already has in hand at its call site, so the swap
// there is a name change and not a rewrite. x is the LOW nibble (even k), y the high.
struct Fp4Pair {
    float x;
    float y;
};

__host__ __device__ inline void
pack_nvfp4_e2m1x16_portable(const Fp4Pair (&values)[8], std::uint32_t& codes_lo,
                            std::uint32_t& codes_hi) noexcept {
    std::uint8_t bytes[8];
    for (unsigned pair = 0; pair < 8u; ++pair) {
        bytes[pair] = pack_fp4_e2m1x2(values[pair].x, values[pair].y);
    }
    codes_lo = static_cast<std::uint32_t>(bytes[0]) |
               (static_cast<std::uint32_t>(bytes[1]) << 8) |
               (static_cast<std::uint32_t>(bytes[2]) << 16) |
               (static_cast<std::uint32_t>(bytes[3]) << 24);
    codes_hi = static_cast<std::uint32_t>(bytes[4]) |
               (static_cast<std::uint32_t>(bytes[5]) << 8) |
               (static_cast<std::uint32_t>(bytes[6]) << 16) |
               (static_cast<std::uint32_t>(bytes[7]) << 24);
}

// ---------------------------------------------------------------------------
// Which form a build may use
// ---------------------------------------------------------------------------
// MEASURED: cvt.e2m1x2.f32 assembles on sm_100a/103a/110a/120a/121a and on the five
// family targets; it is REJECTED by 70/75/80/86/89/90 and -- the part that is easy to
// get wrong -- by the plain (non-'a') sm_100 as well. So the gate keys on the FEATURE
// macros, exactly like ops/common/mma.cuh's NINFER_MMA_HAS_KIND_F8F6F4, and never on the
// number. NINFER_FP4_HAS_CVT_E2M1X2 is 1 only where the instruction is known to assemble.
#if defined(__CUDA_ARCH__) &&                                                                      \
    (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) ||                  \
     defined(__CUDA_ARCH_FEAT_SM110_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) ||                  \
     defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
#define NINFER_FP4_HAS_CVT_E2M1X2 1
#else
#define NINFER_FP4_HAS_CVT_E2M1X2 0
#endif

} // namespace ninfer::ops::detail
