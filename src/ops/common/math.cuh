#pragma once

#include "ops/common/math.h"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

// SiLU, with the exponential folded onto the side that cannot overflow:
// `expf(-x)` reaches +inf for x below -88.72 and `x / inf` then silently
// returns -0, while SiLU is still a normal bf16 there (-2.6e-37 at that
// edge).  Written this way the divisor stays in (1, 2] for every finite x,
// and the only subnormal the form can produce, `e`, enters a multiply
// instead of the divide.  Algebraically identical to x / (1.0f + expf(-x)).
__device__ __forceinline__ float silu(float x) {
    const float e = expf(-fabsf(x));
    return (x >= 0.0f ? x : x * e) / (1.0f + e);
}

// Same divisor, same overflow; sigmoid's limit at -inf is 0 so the zero is
// less obviously wrong, but for x in (-88.72, -92.88] the true value is still
// a nonzero bf16 (3.2e-39 = 32 x the bf16 min subnormal at the edge).
__device__ __forceinline__ float sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return (x >= 0.0f ? 1.0f : e) / (1.0f + e);
}

__device__ __forceinline__ float softplus(float x) { return (x > 20.0f) ? x : log1pf(expf(x)); }

__device__ __forceinline__ float exp2_approx(float x) {
    float y;
// PARAMXLATE — the AMD arm of this construct, with every parameter named.
//   PTX  ex2.approx.f32 d, a                    AMDGCN  v_exp_f32 d, a
//   d/a  : 32-bit float.  NVPTX constraints "=f"/"f"  ->  AMDGPU "v"/"v".
//          MEASURED: on amdgcn the only constraint letter that names a register usable
//          here is "v"; "f", "l", "h", "n" are rejected ("invalid output constraint"),
//          and "r"/"s" do not name a plain VGPR.  v_exp_f32 also accepts an SGPR source
//          (measured: `v_exp_f32 v5, s7` assembles).
//   .approx: not a modifier on this target.  AMD's v_exp_f32 IS the hardware exponent
//          unit, so the qualifier is the instruction, not a suffix to translate.  The
//          call site passes no .ftz/.sat/rounding qualifier, so none is emitted.
//   MEASURED (this line): encodes on gfx900..gfx1201; gfx906 `v_exp_f32_e32 v5, v7` =
//          7E0A4107, and a device-linked gfx906 object carries `v_exp_f32_e32 v1, v1`.
//   NOT CLAIMED: that the two approximations agree numerically at the edges.  There is
//          no AMD silicon on this box; that comparison is EXTERNAL-UNPROBED.
#if defined(__AMDGCN__) || defined(__HIP_PLATFORM_AMD__)
    asm("v_exp_f32 %0, %1;" : "=v"(y) : "v"(x));
#else
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
#endif
    return y;
}

// Pack two f32 into one bf16x2 register: bf16(hi) in the high half, bf16(lo) in the low half.
//
// PREVOLTA-ATTN: `cvt.rn.bf16x2.f32` is an sm_80 instruction -- measured per rung (real ptxas,
// nvcc 12.8): sm_50/52/61/70 reject `.bf16x2` outright ("Feature '.bf16x2' requires .target
// sm_75 or higher") and sm_75 rejects the conversion ("cvt.bf16x2.f32 requires .target sm_80 or
// higher"). Because this helper was the ONLY emitter of that instruction in src/ and it emitted
// it unconditionally, every one of its eight consumers was unbuildable below sm_80 -- including
// src/ops/launcher/gqa_attention_prefill.cu, whose 7129 ptxas errors for sm_52 were entirely
// this one line (dl/oldnvidia/REPORT.md section 2.1). The software arm below is the same
// round-to-nearest-even conversion done on the top 16 bits with integer ops (the spelling
// CUTLASS's software path uses), so:
//   * sm_80 and up: byte-identical to before, the asm arm is taken;
//   * below sm_80 : the TU now BUILDS instead of failing, and the values are the ones that
//     arch can produce. This is a build-wall fix, not a performance or a support claim.
// A NaN input stays a NaN (the exponent field stays all-ones); its payload bits are not the
// ones cvt would have produced, which is the one difference, and no caller feeds this NaN.
// PARAMXLATE — on an AMD device target the FIRST arm below (the integer one) is the
// correct arm, and it is the arm this guard now selects.  Measured, and why:
//   * `cvt.rn.bf16x2.f32` has no AMDGCN spelling on gfx906 (nor gfx908/90a/942/1030/
//     1100/1200/1201): LLVM 21 answers "instruction not supported on this GPU".  The
//     only tested arch that accepts `v_cvt_pk_bf16_f32` is gfx950.
//   * forcing the feature on (`-mattr=+bf16-cvt-insts`) does NOT make it available: it
//     assembles to an encoding the same arch's own disassembler reads as
//     `v_cndmask_b32_e32` -- a silent wrong, which is why the arm below is not used.
//   * the integer arm's rounding is byte-for-byte AMD's own: hip/amd_detail/
//     amd_hip_bfloat16.h's float_to_bfloat16 is `u.int32 += 0x7fff + ((u.int32 >> 16)
//     & 1); ... u.int32 >> 16` -- round-to-nearest-even on the top 16 bits, the same
//     expression as this arm, and this arm now carries that function's OTHER clause too
//     (the signaling-NaN preservation).  There is no remaining difference: the two
//     clauses below ARE float_to_bfloat16, and the arm is bit-identical to it on every
//     one of the 2^32 float inputs.
//   * the operand order is unchanged: this arm puts `hi` in the high half and `lo` in
//     the low half, which is what the `cvt.rn.bf16x2.f32 d, hi, lo` arm emits.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)) || defined(__AMDGCN__) || \
    defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
    unsigned lo_bits = __float_as_uint(lo);
    unsigned hi_bits = __float_as_uint(hi);
    // AMD's own float_to_bfloat16, clause for clause (see the PARAMXLATE note above):
    // round to nearest-even when the exponent bits are not all ones, and PRESERVE the NaN
    // otherwise.  The rounding clause alone turns a signaling NaN into Inf -- 0x7f800001
    // becomes 0x7f80 -- because the carry out of the low half lands in the exponent.
    if (~lo_bits & 0x7f800000u) lo_bits += 0x7fffu + ((lo_bits >> 16) & 1u);
    else if (lo_bits & 0xffffu) lo_bits |= 0x10000u;  // Preserve signaling NaN
    if (~hi_bits & 0x7f800000u) hi_bits += 0x7fffu + ((hi_bits >> 16) & 1u);
    else if (hi_bits & 0xffffu) hi_bits |= 0x10000u;  // Preserve signaling NaN
    return (hi_bits & 0xffff0000u) | (lo_bits >> 16);
}
#else
__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
    std::uint32_t out;
    const std::uint32_t lo_bits = __float_as_uint(lo);
    const std::uint32_t hi_bits = __float_as_uint(hi);
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "r"(hi_bits), "r"(lo_bits));
    return out;
}
#endif

__device__ __forceinline__ float2 bf16x2_to_float2(__nv_bfloat162 value) {
    return __bfloat1622float2(value);
}

__device__ __forceinline__ float2 bf16x2_bits_to_float2(std::uint32_t bits) {
    return bf16x2_to_float2(load_vec<__nv_bfloat162>(&bits));
}

__device__ __forceinline__ __half2 half2_from_bits(std::uint32_t bits) {
    return load_vec<__half2>(&bits);
}

} // namespace ninfer::ops
