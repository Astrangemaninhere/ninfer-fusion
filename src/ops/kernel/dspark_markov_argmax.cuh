#pragma once

// Implements: include/ninfer/ops/dspark_markov_argmax.h
// Match: contiguous BF16 base logits and BF16 low-rank Markov weights.

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

#include "ops/launcher/dspark_markov_argmax.h"

namespace ninfer::ops {

inline constexpr int kDsparkMarkovTiles = 128;

template <int Block>
__launch_bounds__(Block) __global__ void dspark_markov_argmax_kernel(
    const __nv_bfloat16* __restrict__ logits, const __nv_bfloat16* __restrict__ markov_w1,
    const __nv_bfloat16* __restrict__ markov_w2, const std::int32_t* __restrict__ anchors,
    const std::int32_t* __restrict__ drafts, std::int32_t* __restrict__ best_value,
    std::int32_t* __restrict__ best_index, const std::int32_t* __restrict__ extents,
    std::int32_t vocab, std::int32_t k, std::int32_t position) {
    static_assert(Block % 32 == 0);
    const std::int32_t row  = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t tile = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t tid  = static_cast<std::int32_t>(threadIdx.x);
    if (extents[row] <= position) { return; }
    const std::int32_t previous =
        position == 0 ? anchors[row] : drafts[static_cast<std::int64_t>(row) * k + position - 1];
    if (previous < 0 || previous >= vocab) {
        if (tid == 0) {
            best_index[row] = previous;
            best_value[row] = position;
        }
        return;
    }

    __shared__ float previous_embedding[detail::kDsparkMarkovRank];
    const __nv_bfloat16* previous_row =
        markov_w1 + static_cast<std::int64_t>(previous) * detail::kDsparkMarkovRank;
    for (std::int32_t r = tid; r < detail::kDsparkMarkovRank; r += Block) {
        previous_embedding[r] = __bfloat162float(previous_row[r]);
    }
    __syncthreads();

    const std::int32_t tile_tokens = (vocab + kDsparkMarkovTiles - 1) / kDsparkMarkovTiles;
    const std::int32_t token_begin = tile * tile_tokens;
    const std::int32_t token_end   = min(token_begin + tile_tokens, vocab);
    const std::int64_t column_base =
        (static_cast<std::int64_t>(row) * k + position) * static_cast<std::int64_t>(vocab);
    const __nv_bfloat16* column = logits + column_base;

    float best               = -CUDART_INF_F;
    std::int32_t best_token   = 0;
    for (std::int32_t v = token_begin + tid; v < token_end; v += Block) {
        const __nv_bfloat16* w2_row =
            markov_w2 + static_cast<std::int64_t>(v) * detail::kDsparkMarkovRank;
        float value = __bfloat162float(column[v]);
#pragma unroll 8
        for (std::int32_t r = 0; r < detail::kDsparkMarkovRank; ++r) {
            value += previous_embedding[r] * __bfloat162float(w2_row[r]);
        }
        if (value > best) {
            best       = value;
            best_token = v;
        }
    }

    __shared__ float tile_values[Block];
    __shared__ std::int32_t tile_indices[Block];
    tile_values[tid]  = best;
    tile_indices[tid] = best_token;
    __syncthreads();
    for (std::int32_t stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (tile_values[tid + stride] > tile_values[tid]) {
                tile_values[tid]  = tile_values[tid + stride];
                tile_indices[tid] = tile_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0 && token_end > token_begin) {
        const int bits = __float_as_int(tile_values[0]);
        const unsigned int key =
            (bits & 0x80000000) ? static_cast<unsigned int>(~bits)
                                : (static_cast<unsigned int>(bits) | 0x80000000u);
        const unsigned int old = atomicMax(reinterpret_cast<unsigned int*>(best_value + row), key);
        if (key > old) {
            atomicExch(best_index + row, tile_indices[0]);
        }
    }
}

template <int Block>
__launch_bounds__(Block) __global__ void dspark_svip_entropy_kernel(
    const __nv_bfloat16* __restrict__ logits, std::int32_t* __restrict__ extents,
    std::int32_t vocab, std::int32_t k, std::int32_t position, float threshold_squared) {
    static_assert(Block % 32 == 0);
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
    if (extents[row] <= position) { return; }
    const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
    const __nv_bfloat16* column =
        logits + (static_cast<std::int64_t>(row) * k + position) * vocab;

    __shared__ float shared_max;
    __shared__ float shared_sum;
    __shared__ float shared_entropy;
    __shared__ float shared_values[Block];

    float local_max = -CUDART_INF_F;
    for (std::int32_t v = tid; v < vocab; v += Block) {
        local_max = fmaxf(local_max, __bfloat162float(column[v]));
    }
    shared_values[tid] = local_max;
    __syncthreads();
    for (std::int32_t stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { shared_values[tid] = fmaxf(shared_values[tid], shared_values[tid + stride]); }
        __syncthreads();
    }
    shared_max = shared_values[0];
    __syncthreads();

    float local_sum = 0.0f;
    float local_h   = 0.0f;
    for (std::int32_t v = tid; v < vocab; v += Block) {
        const float probability = expf(__bfloat162float(column[v]) - shared_max);
        local_sum += probability;
        local_h -= probability * logf(probability);
    }
    shared_values[tid] = local_sum;
    __syncthreads();
    for (std::int32_t stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { shared_values[tid] += shared_values[tid + stride]; }
        __syncthreads();
    }
    shared_sum = shared_values[0];
    shared_values[tid] = local_h;
    __syncthreads();
    for (std::int32_t stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { shared_values[tid] += shared_values[tid + stride]; }
        __syncthreads();
    }
    shared_entropy = shared_values[0];
    __syncthreads();

    if (tid == 0) {
        const float entropy = shared_entropy / shared_sum + logf(shared_sum);
        if (entropy > threshold_squared) {
            extents[row] = position + 1;
        }
    }
}

} // namespace ninfer::ops