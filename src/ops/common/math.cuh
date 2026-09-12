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
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
    std::uint32_t out;
    const std::uint32_t lo_bits = __float_as_uint(lo);
    const std::uint32_t hi_bits = __float_as_uint(hi);
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "r"(hi_bits), "r"(lo_bits));
    return out;
}

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
