#pragma once

// Implements: include/ninfer/ops/dflash2_selector.h
// Match: token-major BF16 logits/projection and the documented scratch layouts.

#include "ops/launcher/dflash2_selector.h"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ int dflash2_selector_candidate_offset(int b, int batch, int s,
                                                                 int steps, int c) {
    return b + batch * (s + steps * c);
}

__device__ __forceinline__ int dflash2_selector_score_offset(int b, int batch, int s, int steps,
                                                             int p, int c, int top_k) {
    return b + batch * (s + steps * (p + top_k * c));
}

template <int Block, int K>
__launch_bounds__(Block) __global__ void dflash2_selector_topk_kernel(
    const __nv_bfloat16* __restrict__ logits, std::int32_t* __restrict__ candidates,
    float* __restrict__ unary, int vocab, int batch, int steps, int columns) {
    const int column = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    if (column >= columns) { return; }

    float local_value[K];
    std::int32_t local_index[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        local_value[k] = -CUDART_INF_F;
        local_index[k] = vocab;
    }
    const __nv_bfloat16* col_logits = logits + static_cast<std::int64_t>(column) * vocab;
    for (int v = tid; v < vocab; v += Block) {
        const float value = __bfloat162float(col_logits[v]);
        // Insertion into the sorted list while keeping (value, -id) ordering.
        for (int k = 0; k < K; ++k) {
            const bool better =
                value > local_value[k] || (value == local_value[k] && v < local_index[k]);
            if (better) {
                for (int tail = K - 1; tail > k; --tail) {
                    local_value[tail] = local_value[tail - 1];
                    local_index[tail] = local_index[tail - 1];
                }
                local_value[k] = value;
                local_index[k] = v;
                break;
            }
        }
    }

    __shared__ float shared_value[Block][K];
    __shared__ std::int32_t shared_index[Block][K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        shared_value[tid][k] = local_value[k];
        shared_index[tid][k] = local_index[k];
    }
    __syncthreads();

    if (tid == 0) {
        float merged_value[K];
        std::int32_t merged_index[K];
#pragma unroll
        for (int k = 0; k < K; ++k) {
            merged_value[k] = -CUDART_INF_F;
            merged_index[k] = vocab;
        }
        for (int thread = 0; thread < Block; ++thread) {
            for (int k = 0; k < K; ++k) {
                const float value = shared_value[thread][k];
                const std::int32_t index = shared_index[thread][k];
                for (int slot = 0; slot < K; ++slot) {
                    const bool better =
                        value > merged_value[slot] ||
                        (value == merged_value[slot] && index < merged_index[slot]);
                    if (better) {
                        for (int tail = K - 1; tail > slot; --tail) {
                            merged_value[tail] = merged_value[tail - 1];
                            merged_index[tail] = merged_index[tail - 1];
                        }
                        merged_value[slot] = value;
                        merged_index[slot] = index;
                        break;
                    }
                }
            }
        }
        const int b = column / steps;
        const int s = column - b * steps;
        for (int k = 0; k < K; ++k) {
            const int offset =
                dflash2_selector_candidate_offset(b, batch, s, steps, k);
            candidates[offset] = merged_index[k];
            unary[offset]      = merged_value[k];
        }
    }
}

template <int Block>
__launch_bounds__(Block) __global__ void dflash2_selector_scores_kernel(
    const std::int32_t* __restrict__ candidates, const float* __restrict__ unary,
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ predecessor_codebook,
    const __nv_bfloat16* __restrict__ successor_codebook,
    const std::int32_t* __restrict__ anchors, float* __restrict__ scores, int vocab, int batch,
    int steps, int top_k, int columns) {
    constexpr int K   = detail::kDflash2SelectorTopK;
    constexpr int R   = detail::kDflash2SelectorRank;
    const int flat    = static_cast<int>(blockIdx.x);
    const int tid     = static_cast<int>(threadIdx.x);
    if (flat >= columns || tid >= K * K) { return; }
    const int b = flat / steps;
    const int s = flat - b * steps;
    const int p = tid / K;
    const int c = tid - p * K;

    __shared__ std::int32_t shared_candidates[K];
    __shared__ float shared_unary[K];
    if (tid < K) {
        const int offset = dflash2_selector_candidate_offset(b, batch, s, steps, tid);
        shared_candidates[tid] = candidates[offset];
        shared_unary[tid]      = unary[offset];
    }
    __syncthreads();

    if (s == 0 && p > 0) {
        scores[dflash2_selector_score_offset(b, batch, s, steps, p, c, top_k)] = -CUDART_INF_F;
        return;
    }
    const std::int32_t predecessor =
        s == 0 ? anchors[b]
               : candidates[dflash2_selector_candidate_offset(b, batch, s - 1, steps, p)];
    const std::int32_t successor = shared_candidates[c];
    const __nv_bfloat16* hidden = projected + static_cast<std::int64_t>(R) * flat;
    const __nv_bfloat16* predecessor_row =
        predecessor_codebook + static_cast<std::int64_t>(predecessor) * R;
    const __nv_bfloat16* successor_row =
        successor_codebook + static_cast<std::int64_t>(successor) * R;
    float pair = 0.0f;
#pragma unroll 8
    for (int r = 0; r < R; ++r) {
        pair += __bfloat162float(hidden[r]) * __bfloat162float(predecessor_row[r]) *
                __bfloat162float(successor_row[r]);
    }
    scores[dflash2_selector_score_offset(b, batch, s, steps, p, c, top_k)] =
        pair + shared_unary[c];
}

__global__ void dflash2_selector_walk_kernel(const std::int32_t* __restrict__ candidates,
                                             const float* __restrict__ scores,
                                             std::int32_t* __restrict__ drafts,
                                             const SamplingConfig* __restrict__ configs,
                                             std::int32_t* __restrict__ out_ids,
                                             float* __restrict__ out_probs, int batch,
                                             int steps, int top_k) {
    constexpr int K = detail::kDflash2SelectorTopK;
    const int b     = static_cast<int>(blockIdx.x);
    const int lane  = static_cast<int>(threadIdx.x);
    constexpr unsigned Mask = 0xffffffffu;
    const SamplingConfig cfg = configs != nullptr ? configs[b] : SamplingConfig{};
    const bool sample        = cfg.temperature > 0.0f;
    int previous             = 0;
    for (int s = 0; s < steps; ++s) {
        const int pred = previous;
        const auto row_score = [&](int c) {
            return scores[dflash2_selector_score_offset(b, batch, s, steps, pred, c, top_k)];
        };

        // Raw row maximum (used for softmax normalization and greedy tie-break).
        float raw = lane < K ? row_score(lane) : -CUDART_INF_F;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            raw = fmaxf(raw, __shfl_xor_sync(Mask, raw, offset));
        }
        const float raw_max = raw;

        // Sampled walk: Gumbel-max over the row (noise keyed by candidate
        // rank); greedy walk: plain argmax, lowest rank breaking ties.
        float value = -CUDART_INF_F;
        if (lane < K) {
            if (sample) {
                const float u = sampling_uniform(cfg.seed, s + 1,
                                                 kSamplePurposeDFlash2Selector, lane);
                const float g = -__logf(-__logf(fmaxf(u, 1.0e-12f)));
                value         = row_score(lane) + g;
            } else {
                value = row_score(lane);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_xor_sync(Mask, value, offset));
        }
        const float best              = __shfl_sync(Mask, value, 0);
        const bool equal              = lane < K && value == best;
        const std::int32_t candidate_rank = equal ? lane : K;
        std::int32_t chosen = candidate_rank;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            chosen = min(chosen, __shfl_xor_sync(Mask, chosen, offset));
        }
        drafts[b * steps + s] =
            candidates[dflash2_selector_candidate_offset(b, batch, s, steps, chosen)];
        previous = chosen;

        // Publish this step's candidate distribution: softmax over the raw row
        // of the *pre-walk* predecessor rank (the row the accept side reuses).
        if (out_probs != nullptr && out_ids != nullptr) {
            float e = lane < K ? __expf(row_score(lane) - raw_max) : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                e = e + __shfl_xor_sync(Mask, e, offset);
            }
            const float norm = __shfl_sync(Mask, e, 0);
            if (lane < K) {
                const int base  = lane + K * (s + steps * b);
                out_probs[base] = norm > 0.0f ? __expf(row_score(lane) - raw_max) / norm : 0.0f;
                out_ids[base] =
                    candidates[dflash2_selector_candidate_offset(b, batch, s, steps, lane)];
            }
        }
    }
}

} // namespace ninfer::ops
