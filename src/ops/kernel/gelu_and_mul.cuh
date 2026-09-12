#pragma once

// ninfer::ops — gelu_mul kernel: out = gelu(gate) * up, elementwise, exact erf form.
// The activation is gelu_one<false> from ops/kernel/gelu.cuh, so this Op and ops::gelu
// share one exact-erf definition (erff, not a polynomial fit). Vectorized over bf16
// pairs; included only by its launcher.
// See docs/maintainer/op-development.md §6 (no math approximation).

#include "ops/common/bf16_vector.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/gelu.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGeluAndMulPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 gelu_mul_pair(__nv_bfloat162 g, __nv_bfloat162 u) {
    const float r0 = gelu_one<false>(__low2float(g)) * __low2float(u);
    const float r1 = gelu_one<false>(__high2float(g)) * __high2float(u);
    return __floats2bfloat162_rn(r0, r1);
}

__launch_bounds__(256) __global__ void gelu_and_mul_scalar_kernel(const __nv_bfloat16* gate,
                                                                 const __nv_bfloat16* up,
                                                                 __nv_bfloat16* out,
                                                                 std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        out[i] = __float2bfloat16_rn(gelu_one<false>(__bfloat162float(gate[i])) *
                                     __bfloat162float(up[i]));
    }
}

__launch_bounds__(256) __global__ void gelu_and_mul_bf16x8_kernel(const Bf16x8Pack* gate,
                                                                 const Bf16x8Pack* up,
                                                                 Bf16x8Pack* out,
                                                                 std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        const Bf16x8Pack g = load_vec<Bf16x8Pack>(gate + i);
        Bf16x8Pack value   = load_vec<Bf16x8Pack>(up + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            value.pair[pair] = gelu_mul_pair(g.pair[pair], value.pair[pair]);
        }
        store_vec(out + i, value);
    }
}

// Strided read-only inputs (e.g. slices of one packed gate/up matrix); out stays contiguous.
__launch_bounds__(256) __global__ void gelu_and_mul_strided_input_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, std::int64_t n,
    std::int32_t ne0, std::int32_t ne1, std::int32_t ne2, std::int64_t gnb0, std::int64_t gnb1,
    std::int64_t gnb2, std::int64_t gnb3, std::int64_t unb0, std::int64_t unb1, std::int64_t unb2,
    std::int64_t unb3) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const auto* gate_bytes    = reinterpret_cast<const unsigned char*>(gate);
    const auto* up_bytes      = reinterpret_cast<const unsigned char*>(up);
    for (std::int64_t i = start; i < n; i += stride) {
        std::int64_t rem = i;
        const auto d0    = static_cast<std::int32_t>(rem % ne0);
        rem /= ne0;
        const auto d1 = static_cast<std::int32_t>(rem % ne1);
        rem /= ne1;
        const auto d2 = static_cast<std::int32_t>(rem % ne2);
        const auto d3 = static_cast<std::int32_t>(rem / ne2);

        const std::int64_t goff =
            static_cast<std::int64_t>(d0) * gnb0 + static_cast<std::int64_t>(d1) * gnb1 +
            static_cast<std::int64_t>(d2) * gnb2 + static_cast<std::int64_t>(d3) * gnb3;
        const std::int64_t uoff =
            static_cast<std::int64_t>(d0) * unb0 + static_cast<std::int64_t>(d1) * unb1 +
            static_cast<std::int64_t>(d2) * unb2 + static_cast<std::int64_t>(d3) * unb3;
        const auto gv = *reinterpret_cast<const __nv_bfloat16*>(gate_bytes + goff);
        const auto uv = *reinterpret_cast<const __nv_bfloat16*>(up_bytes + uoff);
        out[i]        = __float2bfloat16_rn(gelu_one<false>(__bfloat162float(gv)) *
                                            __bfloat162float(uv));
    }
}

} // namespace ninfer::ops
