#pragma once

// Implements: include/ninfer/ops/logit_policy.h
// Match: contiguous BF16, in-place. Registered aligned domains use one 16-byte
// pack per thread; BF16x2 and scalar routes preserve correctness for smaller
// alignments and odd tails. FP32 tanhf; a nonpositive cap disables the tanh
// stage, and multiplier = 1 / cap = 0 leaves every value bit-exact.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kLogitPolicyPairsPerThread = 4;

__device__ __forceinline__ float logit_policy_value(float v, float multiplier,
                                                    float cap) {
    v = v * multiplier;
    if (cap > 0.0f) { v = cap * tanhf(v / cap); }
    return v;
}

__device__ __forceinline__ __nv_bfloat162
logit_policy_pair(__nv_bfloat162 x, float multiplier, float cap) {
    const float r0 = logit_policy_value(__low2float(x), multiplier, cap);
    const float r1 = logit_policy_value(__high2float(x), multiplier, cap);
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void logit_policy_scalar_kernel(__nv_bfloat16* x, float multiplier,
                                           float cap, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(
            logit_policy_value(__bfloat162float(x[i]), multiplier, cap));
    }
}

__launch_bounds__(256) __global__
    void logit_policy_bf16x8_kernel(Bf16x8Pack* x, float multiplier, float cap,
                                    std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        Bf16x8Pack xv = load_vec<Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            xv.pair[pair] = logit_policy_pair(xv.pair[pair], multiplier, cap);
        }
        store_vec(x + i, xv);
    }
}

__launch_bounds__(256) __global__
    void logit_policy_bf16x2_kernel(__nv_bfloat16* x, float multiplier,
                                    float cap, std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kLogitPolicyPairsPerThread;
    const std::int64_t n2 = n / 2;

    auto* x2 = reinterpret_cast<__nv_bfloat162*>(x);
    for (std::int64_t j = tid * kLogitPolicyPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 x0 = x2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 x1  = x2[j + 1];
            const __nv_bfloat162 x2v = x2[j + 2];
            const __nv_bfloat162 x3  = x2[j + 3];
            x2[j]                    = logit_policy_pair(x0, multiplier, cap);
            x2[j + 1]                = logit_policy_pair(x1, multiplier, cap);
            x2[j + 2]                = logit_policy_pair(x2v, multiplier, cap);
            x2[j + 3]                = logit_policy_pair(x3, multiplier, cap);
        } else {
            x2[j] = logit_policy_pair(x0, multiplier, cap);
            if (j + 1 < n2) {
                x2[j + 1] = logit_policy_pair(x2[j + 1], multiplier, cap);
            }
            if (j + 2 < n2) {
                x2[j + 2] = logit_policy_pair(x2[j + 2], multiplier, cap);
            }
        }
    }

    if (tid == 0 && (n & 1) != 0) {
        const std::int64_t i = n - 1;
        x[i] = __float2bfloat16_rn(
            logit_policy_value(__bfloat162float(x[i]), multiplier, cap));
    }
}

} // namespace ninfer::ops
