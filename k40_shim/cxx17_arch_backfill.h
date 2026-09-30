// k40names prototype (d): the arch-name back-fill, at the NAME level, for both families.
//
// Injected with `-include k40_shim/cxx17_arch_backfill.h` (next to the C++17 `std::span`
// shim), so no tree source changes.  Nothing in here is a "fix one function" patch: it
// re-declares, for the architectures whose CUDA headers omit them, the exact overload
// set the 11.8 front end reports undefined for this tree.
//
// WHY IT IS NEEDED (measured, not assumed):
//   CUDA 11.8 cuda_fp16.h keeps
//     #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)          line 1964 .. 3735
//   open and declares 73 functions inside it; 6 of them are used by src/.
//   CUDA 11.8 cuda_bf16.h keeps
//     #if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
//                                                                    line  319 .. 3737
//   open and declares 155 functions inside it; 10 of them are used by src/.
//   CUDA 12.8 declares the same fp16 names with NO arch gate at all (line 2783 vs the
//   1964/2944 gates), and reduces the bf16 gate to __hfma2 alone (2750..2796) -- so the
//   same source builds on 12.8 at compute_52 and dies on 11.8 at compute_35.  That is the
//   header-vintage wall, and it is why the guard below keys on the toolchain as well as
//   the arch.
//
// WHY THE SET IS CLOSED (this is the point of doing it by name): the 16 names below are
//   the complete intersection of
//     (a) the two gate blocks above, and
//     (b) every half/bf16-family identifier that appears in ANY file under src/ or
//         include/ (static scan with comments and string literals blanked, 1340 files),
//   NOT just the names some red TU happened to reach before EDG gave up.  The 184-TU
//   census stops early in 136 of 172 red TUs, so "the names I saw in error messages"
//   would be an undercount by construction; the static scan is what closes the set.
//
// WHY NOT "ALL 73 + ALL 155": because those blocks also *declare* names that exist
//   elsewhere in the headers (`__ldg`, `__shfl_sync`, `__stcs`, `__stcg`, ... sit inside
//   the bf16 block purely as a layout accident).  Re-declaring one of those is a
//   redefinition error, so a mechanical "everything inside the gate" pass is wrong.
//
// NEUTRALITY: every branch here is compiled only in a device pass for a toolchain whose
//   headers gate the name AND an arch below the gate.  Under the tree's own 12.8 the whole
//   file expands to nothing, so the token stream reaching the front end is identical to a
//   build without it -- measured byte-for-byte on PTX and SASS in
//   dl/k40names/logs/ (see REPORT section 4).
//
// EXACTNESS (the standard the previous line set with `bits << 16` was "exact, not
//   approximate").  Each body below is annotated with its argument, and the fp16
//   arithmetic ones use fp64 *deliberately*: every half is exact in fp64, and the exact
//   difference/product of two halves needs at most 41 significand bits, so the fp64
//   intermediate is EXACT and the single RN narrowing is bit-identical to the hardware
//   instruction.  The one place where that argument does not close completely is __hfma2
//   (an fp16 fma can need up to 62 bits); it is stated inline and no live call site below
//   sm_70 exists, see the report.
#ifndef K40_ARCH_BACKFILL_H
#define K40_ARCH_BACKFILL_H

#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ < 12)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer_k40_backfill {

// Representation-independent bit moves.  __half/__half2/__nv_bfloat16/__nv_bfloat162 are
// structs whose member names changed between CUDA releases, so build them from bits
// rather than from members.
template <class To, class From>
__device__ __forceinline__ To bit_copy(const From& from) {
    static_assert(sizeof(To) == sizeof(From), "bit_copy: size mismatch");
    To to;
    __builtin_memcpy(&to, &from, sizeof(To));
    return to;
}

// ---- exact bf16 <-> fp32, the same integer spelling as
//      src/ops/common/math.cuh's pack_bf16x2 / k40_bf16_bits_to_float ----
__device__ __forceinline__ float bf16_bits_to_f32(std::uint16_t b) {
    return __uint_as_float(static_cast<std::uint32_t>(b) << 16);
}

// Round-to-nearest-even on the top 16 bits, signaling NaN preserved (the second clause is
// what keeps a NaN a NaN; without it the carry out of the low half lands in the
// exponent and 0x7f800001 becomes Inf).
__device__ __forceinline__ std::uint16_t f32_to_bf16_bits(float v) {
    unsigned b = __float_as_uint(v);
    if (~b & 0x7f800000u) {
        b += 0x7fffu + ((b >> 16) & 1u);
    } else if (b & 0xffffu) {
        b |= 0x10000u;
    }
    return static_cast<std::uint16_t>(b >> 16);
}

__device__ __forceinline__ __nv_bfloat16 bf16_from_bits(std::uint16_t b) {
    return bit_copy<__nv_bfloat16>(b);
}
__device__ __forceinline__ std::uint16_t bf16_to_bits(__nv_bfloat16 v) {
    return bit_copy<std::uint16_t>(v);
}
__device__ __forceinline__ __nv_bfloat162 bf162_from_bits(std::uint32_t u) {
    return bit_copy<__nv_bfloat162>(u);
}
__device__ __forceinline__ std::uint32_t bf162_to_bits(__nv_bfloat162 v) {
    return bit_copy<std::uint32_t>(v);
}
// low half = .x, high half = .y (CUDA's own layout; the previous line's arm reads it the
// same way at src/ops/common/math.cuh:136).
__device__ __forceinline__ std::uint32_t bf162_pack(std::uint16_t lo, std::uint16_t hi) {
    return static_cast<std::uint32_t>(lo) | (static_cast<std::uint32_t>(hi) << 16);
}
__device__ __forceinline__ std::uint32_t half2_pack(std::uint16_t lo, std::uint16_t hi) {
    return static_cast<std::uint32_t>(lo) | (static_cast<std::uint32_t>(hi) << 16);
}
__device__ __forceinline__ __half2 half2_from_bits(std::uint32_t u) {
    return bit_copy<__half2>(u);
}
__device__ __forceinline__ __half half_from_bits(std::uint16_t u) {
    return bit_copy<__half>(u);
}
__device__ __forceinline__ std::uint16_t half_to_bits(__half h) {
    return bit_copy<std::uint16_t>(h);
}

}  // namespace ninfer_k40_backfill

// ===========================================================================
// fp16 half2 arithmetic -- the 6 names 11.8 gates below sm_53
// (cuda_fp16.h 1964..3735: __hsub2 2236, __hsub2_rn 2297, __hmul2 2256, __hmul 2548,
//  __hfma2 2436, __hgt 3111).  Every other fp16 name the tree uses (__half2half2,
//  __halves2half2, __highs2half2, __high2half, __low2half, __half22float2,
//  __floats2half2_rn, __half2float, __float2half, __double2half, __ushort_as_half,
//  __low2float, __high2float) is declared OUTSIDE that block and is left alone --
//  measured rc=0 at sm_35 with real ptxas, see logs/n08_probe2.log.
// ===========================================================================
#if (__CUDA_ARCH__ < 530)

// EXACT: the product of two halves is exact in fp32 (<= 22 significand bits), so widening
// both operands, multiplying in fp32 and narrowing once with __floats2half2_rn is the
// same single rounding that mul.rn.f16x2 performs -- subnormals and Inf included.
__device__ __forceinline__ __half2 __hmul2(__half2 a, __half2 b) {
    const float2 fa = __half22float2(a);
    const float2 fb = __half22float2(b);
    return __floats2half2_rn(fa.x * fb.x, fa.y * fb.y);
}
__device__ __forceinline__ __half2 __hmul2_rn(__half2 a, __half2 b) { return __hmul2(a, b); }

__device__ __forceinline__ __half __hmul(__half a, __half b) {
    return __float2half(__half2float(a) * __half2float(b));
}

// EXACT, and this is where fp32 would NOT have been enough: the exact difference of two
// halves needs up to 41 significand bits (fp16's exponent span is 39, plus an 11-bit
// significand), which fp32's 24 bits cannot hold -- a fp32 intermediate rounds and the
// subsequent narrowing can then double-round across an fp16 midpoint.  fp64 holds every
// such difference EXACTLY, so one rounding via __double2half is the hardware result.
__device__ __forceinline__ __half2 __hsub2(__half2 a, __half2 b) {
    const float2 fa = __half22float2(a);
    const float2 fb = __half22float2(b);
    return ninfer_k40_backfill::half2_from_bits(ninfer_k40_backfill::half2_pack(
        ninfer_k40_backfill::half_to_bits(
            __double2half(static_cast<double>(fa.x) - static_cast<double>(fb.x))),
        ninfer_k40_backfill::half_to_bits(
            __double2half(static_cast<double>(fa.y) - static_cast<double>(fb.y)))));
}
__device__ __forceinline__ __half2 __hsub2_rn(__half2 a, __half2 b) { return __hsub2(a, b); }

__device__ __forceinline__ __half __hsub(__half a, __half b) {
    return __double2half(static_cast<double>(__half2float(a)) -
                         static_cast<double>(__half2float(b)));
}

// a*b is exact in fp64 (<= 22 bits); the sum is exact whenever the addend's exponent gap
// is <= 31 (22 + 31 == 53).  An fp16 fma can in principle need 62 bits, so this arm is
// exact-except-for-gap>31 rather than exact -- stated, not hidden.  It also has no live
// instantiation below sm_70 in this tree: both call sites (qpn_kernels.cuh:238 and :300)
// sit in the wmma path that NINFER_QPN_HAS_WMMA (>= 700) switches off.
__device__ __forceinline__ __half2 __hfma2(__half2 a, __half2 b, __half2 c) {
    const float2 fa = __half22float2(a);
    const float2 fb = __half22float2(b);
    const float2 fc = __half22float2(c);
    return __floats2half2_rn(
        static_cast<float>(static_cast<double>(fa.x) * static_cast<double>(fb.x) +
                           static_cast<double>(fc.x)),
        static_cast<float>(static_cast<double>(fa.y) * static_cast<double>(fb.y) +
                           static_cast<double>(fc.y)));
}

// EXACT: widening a half to fp32 is exact for every input (subnormal/Inf/NaN included),
// and fp32 compares in the same order, so the predicate is the hardware's.  NaN compares
// false on both sides, which is what __hgt is defined to do.
__device__ __forceinline__ bool __hgt(__half a, __half b) {
    return __half2float(a) > __half2float(b);
}

#endif  // __CUDA_ARCH__ < 530

// ===========================================================================
// bf16 -- the 10 names 11.8 gates below sm_80 (cuda_bf16.h 319..3737: __bfloat1622float2
// 355, __halves2bfloat162 1403, __bfloat16_as_ushort 1468, __ushort_as_bfloat16 1500,
// __hsub2 2123, __hsub2_rn 2183, __hmul2 2143, __hmul 2434, __hfma2 2322, __hgt 2997).
// __bfloat162float (246), __float2bfloat16 (169), __float2bfloat16_rn (184),
// __low2float (301), __high2float (317) and __floats2bfloat162_rn are NOT in the block and
// are left alone -- measured rc=0 at sm_35 with real ptxas.
//
// The arithmetic bodies are the spelling CUTLASS uses (and the one this tree already
// carries at src/ops/common/math.cuh:100-104 for the same reason): exact integer
// widening/shrinking on the top 16 bits.  Unlike fp16, fp64 cannot rescue the
// add/sub case -- bf16's exponent field is as wide as fp32's, so an exact difference can
// need ~262 bits -- so the boundary is stated per name instead.
// ===========================================================================
#if (__CUDA_ARCH__ < 800)

// EXACT: bf16 keeps fp32's 8-bit exponent field, so `bits << 16` widens every input
// exactly -- subnormal, Inf and NaN included.  Same argument as math.cuh:127-133.
__device__ __forceinline__ float2 __bfloat1622float2(__nv_bfloat162 v) {
    const std::uint32_t u = ninfer_k40_backfill::bf162_to_bits(v);
    float2 out;
    out.x = ninfer_k40_backfill::bf16_bits_to_f32(static_cast<std::uint16_t>(u & 0xffffu));
    out.y = ninfer_k40_backfill::bf16_bits_to_f32(static_cast<std::uint16_t>(u >> 16));
    return out;
}

// EXACT: pure bit moves, no arithmetic and no rounding at any point.
__device__ __forceinline__ __nv_bfloat16 __ushort_as_bfloat16(unsigned short int u) {
    return ninfer_k40_backfill::bf16_from_bits(u);
}
__device__ __forceinline__ unsigned short int __bfloat16_as_ushort(__nv_bfloat16 v) {
    return ninfer_k40_backfill::bf16_to_bits(v);
}
// low half = a, high half = b: CUDA 12.8 (which does NOT gate this name) defines it as
// `__nv_bfloat162_raw v = { a.__x, b.__x }`, i.e. the same packing order, so this is a
// transcription of the current header rather than a guess.
__device__ __forceinline__ __nv_bfloat162 __halves2bfloat162(__nv_bfloat16 a,
                                                            __nv_bfloat16 b) {
    return ninfer_k40_backfill::bf162_from_bits(
        ninfer_k40_backfill::bf162_pack(ninfer_k40_backfill::bf16_to_bits(a),
                                        ninfer_k40_backfill::bf16_to_bits(b)));
}

// EXACT: 8-bit bf16 significands make the product exact in fp32 (16 <= 24 bits); one
// round-to-nearest-even pack back is the hardware's single rounding.
__device__ __forceinline__ __nv_bfloat162 __hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {
    const float2 fa = __bfloat1622float2(a);
    const float2 fb = __bfloat1622float2(b);
    return ninfer_k40_backfill::bf162_from_bits(ninfer_k40_backfill::bf162_pack(
        ninfer_k40_backfill::f32_to_bf16_bits(fa.x * fb.x),
        ninfer_k40_backfill::f32_to_bf16_bits(fa.y * fb.y)));
}
__device__ __forceinline__ __nv_bfloat16 __hmul(__nv_bfloat16 a, __nv_bfloat16 b) {
    return ninfer_k40_backfill::bf16_from_bits(ninfer_k40_backfill::f32_to_bf16_bits(
        __bfloat162float(a) * __bfloat162float(b)));
}

// "exact when the fp32 difference is exact": with 8-bit significands that holds whenever
// the operands' exponent gap is <= 16.  Every live call site in this tree satisfies it and
// the argument is per site, not per name:
//   q4_rowsplit_storage.cuh:64   __hsub2(bf162(128 + nibble), bf162(136.0))
//   w8_small_t_mma.cuh:51        __hsub2_rn(bf162(128 +/- code), bf162(128.0 | signs))
//   w8_gdn_input_gemm_splitk.cu:67  the same biased-integer pattern
// all three are small integers plus/minus a small integer, where fp32 is exact, and the
// result is an integer in [-8, 7] that bf16 represents exactly.
__device__ __forceinline__ __nv_bfloat162 __hsub2(__nv_bfloat162 a, __nv_bfloat162 b) {
    const float2 fa = __bfloat1622float2(a);
    const float2 fb = __bfloat1622float2(b);
    return ninfer_k40_backfill::bf162_from_bits(ninfer_k40_backfill::bf162_pack(
        ninfer_k40_backfill::f32_to_bf16_bits(fa.x - fb.x),
        ninfer_k40_backfill::f32_to_bf16_bits(fa.y - fb.y)));
}
__device__ __forceinline__ __nv_bfloat162 __hsub2_rn(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hsub2(a, b);
}
__device__ __forceinline__ __nv_bfloat16 __hsub(__nv_bfloat16 a, __nv_bfloat16 b) {
    return ninfer_k40_backfill::bf16_from_bits(ninfer_k40_backfill::f32_to_bf16_bits(
        __bfloat162float(a) - __bfloat162float(b)));
}

// Same boundary as the fp16 arm: a*b is exact in fp32; the addend may round.  No live
// instantiation below sm_70 (qpn_kernels.cuh:238/:300 are inside NINFER_QPN_HAS_WMMA).
__device__ __forceinline__ __nv_bfloat162 __hfma2(__nv_bfloat162 a, __nv_bfloat162 b,
                                                  __nv_bfloat162 c) {
    const float2 fa = __bfloat1622float2(a);
    const float2 fb = __bfloat1622float2(b);
    const float2 fc = __bfloat1622float2(c);
    return ninfer_k40_backfill::bf162_from_bits(ninfer_k40_backfill::bf162_pack(
        ninfer_k40_backfill::f32_to_bf16_bits(fmaf(fa.x, fb.x, fc.x)),
        ninfer_k40_backfill::f32_to_bf16_bits(fmaf(fa.y, fb.y, fc.y))));
}

// EXACT: exact widening plus fp32 ordering; NaN is false on both sides.
__device__ __forceinline__ bool __hgt(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __bfloat162float(a) > __bfloat162float(b);
}

#endif  // __CUDA_ARCH__ < 800

// ---------------------------------------------------------------------------
// K40 gap 3: __maxnreg__(n) is a CUDA 12.4+ declaration attribute; it appears 0 times in
// 11.8's crt/host_defines.h and 2 times (lines 101, 140) in 12.8's.  The tree uses it on 5
// files.  On sm_3x..sm_8x the attribute has no meaning anyway (PTX .maxnreg / the register
// cap is a sm_90 feature), so the honest 11.x spelling is "expand to nothing", which is what
// the empty body below does.  NOTE: this DROPS the register cap on the K40 path -- it makes
// the TU compile, it does not preserve the cap.  Recorded, not hidden.
#ifndef __maxnreg__
#define __maxnreg__(n)
#endif

#endif  // device pass, toolchain whose headers gate these names
#endif  // K40_ARCH_BACKFILL_H
