// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : MirkoCovizzi/ninfer-rtx5090-mobile
// Branch      : feat/kvarn-production
// Commit      : 750aac554d52b758c3ee33eff81f08bf1dceb201
// Source path : src/ops/kvarn/hadamard.cuh
// sha256(src) : dd6f05fd15f4bf286716c8986fe9e5f06bec392d6c5b41c60ffdb160cd061088
// Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits (rk4v4-e8 = rotated K 4b, V 4b, rk4v4 in the mix), which is a DIFFERENT axis from src/kvcfg/kv_formats.h's tier vocabulary (hot/tail/cold, Rk4v4).
#pragma once

#include "ops/kvarn/config.cuh"

#include <cuda_runtime.h>

namespace ninfer::ops::kvarn::detail {

__device__ __forceinline__ float hadamard_block(float value, float (&stage)[2][D], int d) {
#pragma unroll
    for (int span = 1; span < 32; span <<= 1) {
        const float other = __shfl_xor_sync(0xffffffffU, value, span);
        value             = (d & span) == 0 ? value + other : other - value;
    }
    stage[0][d] = value;
    __syncthreads();
    int current = 0;
    for (int span = 32; span < D; span <<= 1) {
        const int group         = d / (2 * span);
        const int lane          = d & (span - 1);
        const int left          = group * 2 * span + lane;
        const int right         = left + span;
        const float left_value  = stage[current][left];
        const float right_value = stage[current][right];
        value = (d & span) == 0 ? left_value + right_value : left_value - right_value;
        stage[current ^ 1][d] = value;
        current ^= 1;
        __syncthreads();
    }
    return value * 0.0625F;
}

__device__ __forceinline__ void hadamard_warp(float (&values)[D / 32], int lane) {
#pragma unroll
    for (int span = 1; span < 32; span <<= 1) {
#pragma unroll
        for (int segment = 0; segment < D / 32; ++segment) {
            const float other = __shfl_xor_sync(0xffffffffU, values[segment], span);
            values[segment] =
                (lane & span) == 0 ? values[segment] + other : other - values[segment];
        }
    }
#pragma unroll
    for (int span = 1; span < D / 32; span <<= 1) {
#pragma unroll
        for (int block = 0; block < D / 32; block += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float left              = values[block + offset];
                const float right             = values[block + span + offset];
                values[block + offset]        = left + right;
                values[block + span + offset] = left - right;
            }
        }
    }
#pragma unroll
    for (float& value : values) { value *= 0.0625F; }
}

} // namespace ninfer::ops::kvarn::detail
