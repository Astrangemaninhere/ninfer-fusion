#pragma once

// Implements: include/ninfer/ops/sigmoid_mul.h
// Match: contiguous BF16 inputs. Registered aligned/eight-element domains use
// one 16-byte pack per thread; BF16x2 and scalar routes preserve correctness for
// smaller alignments and odd tails. Sigmoid remains FP32 expf, not a fit.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kSigmoidGateMulPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 sigmoid_gate_mul_pair(__nv_bfloat162 gate,
                                                                __nv_bfloat162 x) {
    const float r0 = __low2float(x) * sigmoid(__low2float(gate));
    const float r1 = __high2float(x) * sigmoid(__high2float(gate));
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void sigmoid_gate_mul_scalar_kernel(const __nv_bfloat16* gate, __nv_bfloat16* x,
                                               std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) * sigmoid(__bfloat162float(gate[i])));
    }
}

__launch_bounds__(256) __global__
    void sigmoid_gate_mul_bf16x8_kernel(const Bf16x8Pack* gate, Bf16x8Pack* x, std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        const Bf16x8Pack gv = load_vec<Bf16x8Pack>(gate + i);
        Bf16x8Pack xv       = load_vec<Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            xv.pair[pair] = sigmoid_gate_mul_pair(gv.pair[pair], xv.pair[pair]);
        }
        store_vec(x + i, xv);
    }
}

// Headwise scalar gate. `x` is [D,H,T] contiguous and `gate` is [H,T] contiguous, so the
// gate index of element (d,h,t) is h + H*t. Every 8-element pack lies inside one head when
// D % 8 == 0, so the pack index divided by (D/8) is exactly that gate index: one sigmoid
// per pack, no per-element gate load.
__launch_bounds__(256) __global__ void headwise_sigmoid_gate_mul_bf16x8_kernel(
    const __nv_bfloat16* gate, Bf16x8Pack* x, std::int64_t packs,
    std::int32_t packs_per_head) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        const float g = sigmoid(__bfloat162float(gate[i / packs_per_head]));
        Bf16x8Pack value = load_vec<Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            value.pair[pair] = __floats2bfloat162_rn(
                __low2float(value.pair[pair]) * g, __high2float(value.pair[pair]) * g);
        }
        store_vec(x + i, value);
    }
}

// Any head_dim: one gate scalar per (head, token) with an exact index divide.
__launch_bounds__(256) __global__ void headwise_sigmoid_gate_mul_scalar_kernel(
    const __nv_bfloat16* gate, __nv_bfloat16* x, std::int64_t n, std::int32_t head_dim) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const float g = sigmoid(__bfloat162float(gate[i / head_dim]));
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) * g);
    }
}

__launch_bounds__(256) __global__
    void sigmoid_gate_mul_bf16x2_kernel(const __nv_bfloat16* gate, __nv_bfloat16* x,
                                        std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kSigmoidGateMulPairsPerThread;
    const std::int64_t n2 = n / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    auto* x2          = reinterpret_cast<__nv_bfloat162*>(x);
    for (std::int64_t j = tid * kSigmoidGateMulPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 g0 = gate2[j];
        const __nv_bfloat162 x0 = x2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 g1  = gate2[j + 1];
            const __nv_bfloat162 x1  = x2[j + 1];
            const __nv_bfloat162 g2  = gate2[j + 2];
            const __nv_bfloat162 x2v = x2[j + 2];
            const __nv_bfloat162 g3  = gate2[j + 3];
            const __nv_bfloat162 x3  = x2[j + 3];
            x2[j]                    = sigmoid_gate_mul_pair(g0, x0);
            x2[j + 1]                = sigmoid_gate_mul_pair(g1, x1);
            x2[j + 2]                = sigmoid_gate_mul_pair(g2, x2v);
            x2[j + 3]                = sigmoid_gate_mul_pair(g3, x3);
        } else {
            x2[j] = sigmoid_gate_mul_pair(g0, x0);
            if (j + 1 < n2) { x2[j + 1] = sigmoid_gate_mul_pair(gate2[j + 1], x2[j + 1]); }
            if (j + 2 < n2) { x2[j + 2] = sigmoid_gate_mul_pair(gate2[j + 2], x2[j + 2]); }
        }
    }

    if (tid == 0 && (n & 1) != 0) {
        const std::int64_t i = n - 1;
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) * sigmoid(__bfloat162float(gate[i])));
    }
}

} // namespace ninfer::ops
