#pragma once

// Implements: include/ninfer/ops/dflash2_selector.h
// Match: token-major BF16 logits/projection and the documented scratch layouts.

#include "ops/launcher/dflash2_selector.h"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

// Sentinel for an unfilled shortlist slot: every real global token id is below it.
inline constexpr std::int32_t kDflash2NoToken = 0x7fffffff;

__device__ __forceinline__ int dflash2_selector_candidate_offset(int b, int batch, int s,
                                                                 int steps, int c) {
    return b + batch * (s + steps * c);
}

__device__ __forceinline__ int dflash2_selector_score_offset(int b, int batch, int s, int steps,
                                                             int p, int c, int top_k) {
    return b + batch * (s + steps * (p + top_k * c));
}

__device__ __forceinline__ void dflash2_selector_merge16(
    const float* __restrict__ a, const std::int32_t* __restrict__ ai, int na,
    const float* __restrict__ b, const std::int32_t* __restrict__ bi, int nb,
    float* __restrict__ out, std::int32_t* __restrict__ outi, int k) {
    // Merge two descending (value, -id) lists, keeping the top k entries.
    int ia = 0;
    int ib = 0;
    for (int slot = 0; slot < k; ++slot) {
        bool take_a;
        if (ia >= na) {
            take_a = false;
        } else if (ib >= nb) {
            take_a = true;
        } else {
            take_a = a[ia] > b[ib] || (a[ia] == b[ib] && ai[ia] < bi[ib]);
        }
        if (take_a) {
            out[slot]  = a[ia];
            outi[slot] = ai[ia];
            ++ia;
        } else {
            out[slot]  = b[ib];
            outi[slot] = bi[ib];
            ++ib;
        }
    }
}

template <int Block, int K>
__launch_bounds__(Block) __global__ void dflash2_selector_topk_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const std::int32_t* __restrict__ row_to_global_ids,
    std::int32_t* __restrict__ candidates, float* __restrict__ unary, int vocab, int domain,
    int batch, int steps, int columns) {
    const int column = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    if (column >= columns) { return; }

    // Per-thread top-K over a strided slice of the head rows. `row_to_global_ids`
    // maps a shortlist row to the global token id used for the draft output and for
    // score ties (contract 6.1); a null map is the identity, i.e. the full head.
    float local_value[K];
    std::int32_t local_index[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        local_value[k] = -CUDART_INF_F;
        local_index[k] = kDflash2NoToken;
    }
    const __nv_bfloat16* col_logits = logits + static_cast<std::int64_t>(column) * vocab;
    // Contract: only rows [0, domain) may enter the top-K. A full head additionally
    // excludes ids at or above the token domain (248077) as unaddressable results.
    for (int v = tid; v < domain; v += Block) {
        const float value = __bfloat162float(col_logits[v]);
        const std::int32_t id =
            row_to_global_ids != nullptr ? row_to_global_ids[v] : static_cast<std::int32_t>(v);
        // Fast reject: below the current K-th best (with id tie-break).
        if (value < local_value[K - 1] ||
            (value == local_value[K - 1] && id >= local_index[K - 1])) {
            continue;
        }
        for (int k = 0; k < K; ++k) {
            const bool better =
                value > local_value[k] || (value == local_value[k] && id < local_index[k]);
            if (better) {
                for (int tail = K - 1; tail > k; --tail) {
                    local_value[tail] = local_value[tail - 1];
                    local_index[tail] = local_index[tail - 1];
                }
                local_value[k] = value;
                local_index[k] = id;
                break;
            }
        }
    }

    // Parallel tree reduction of the Block per-thread top-K lists.
    __shared__ float shared_value[Block][K];
    __shared__ std::int32_t shared_index[Block][K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        shared_value[tid][k] = local_value[k];
        shared_index[tid][k] = local_index[k];
    }
    __syncthreads();

    for (int stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float merged_value[K];
            std::int32_t merged_index[K];
            dflash2_selector_merge16(shared_value[tid], shared_index[tid], K,
                                     shared_value[tid + stride], shared_index[tid + stride], K,
                                     merged_value, merged_index, K);
#pragma unroll
            for (int k = 0; k < K; ++k) {
                shared_value[tid][k] = merged_value[k];
                shared_index[tid][k] = merged_index[k];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int b = column / steps;
        const int s = column - b * steps;
        for (int k = 0; k < K; ++k) {
            const int offset = dflash2_selector_candidate_offset(b, batch, s, steps, k);
            candidates[offset] = shared_index[0][k];
            unary[offset]      = shared_value[0][k];
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
    int steps, int top_k, int columns, float pair_scale) {
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
        pair * pair_scale + shared_unary[c];
}

__global__ void dflash2_selector_walk_kernel(const std::int32_t* __restrict__ candidates,
                                             const float* __restrict__ scores,
                                             const float* __restrict__ unary,
                                             std::int32_t* __restrict__ drafts,
                                             const SamplingConfig* __restrict__ configs,
                                             std::int32_t* __restrict__ out_ids,
                                             float* __restrict__ out_probs, int batch,
                                             int steps, int top_k, int debug) {
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
        // Keep this lane's own score: the reduction below overwrites `value`, and
        // comparing the reduced partial maxima lets lane 0 win almost every step, which
        // silently discards the edge term and degenerates the walk into a per-step unary
        // argmax.
        const float own_value = value;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_xor_sync(Mask, value, offset));
        }
        const float best              = __shfl_sync(Mask, value, 0);
        const bool equal              = lane < K && own_value == best;
        const std::int32_t candidate_rank = equal ? lane : K;
        std::int32_t chosen = candidate_rank;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            chosen = min(chosen, __shfl_xor_sync(Mask, chosen, offset));
        }
        drafts[b * steps + s] =
            candidates[dflash2_selector_candidate_offset(b, batch, s, steps, chosen)];
        previous = chosen;

        // Env-gated (NINFER_DF2SEL): decompose the winner into unary and edge, with
        // the unary row's span as the scale to judge the edge term against.
        if (debug != 0 && b == 0 && lane == 0) {
            const float e_chosen = scores[dflash2_selector_score_offset(b, batch, s, steps,
                                                                       pred, chosen, top_k)];
            const float u_chosen =
                unary[dflash2_selector_candidate_offset(b, batch, s, steps, chosen)];
            float u_hi = -CUDART_INF_F;
            float u_lo = CUDART_INF_F;
            float pair_hi = -CUDART_INF_F;
            float pair_lo = CUDART_INF_F;
            float e_hi = -CUDART_INF_F;
            int e_arg = -1;
            int u_arg = -1;
            for (int c = 0; c < K; ++c) {
                const float u =
                    unary[dflash2_selector_candidate_offset(b, batch, s, steps, c)];
                const float pair_c =
                    scores[dflash2_selector_score_offset(b, batch, s, steps, pred, c, top_k)] - u;
                u_lo = fminf(u_lo, u);
                pair_hi = fmaxf(pair_hi, pair_c);
                pair_lo = fminf(pair_lo, pair_c);
                if (c == 0 || u > u_hi) {
                    u_hi = u;
                    u_arg = c;
                }
                if (c == 0 || u + pair_c > e_hi) {
                    e_hi = u + pair_c;
                    e_arg = c;
                }
            }
            printf("[df2sel] s=%d pred=%d chosen=%d tok=%d E=%.3f u=%.3f pair=%.3f "
                   "uspan=%.3f pairspan=%.3f Earg=%d Uarg=%d\n",
                   s, pred, chosen,
                   candidates[dflash2_selector_candidate_offset(b, batch, s, steps, chosen)],
                   e_chosen, u_chosen, e_chosen - u_chosen, u_hi - u_lo, pair_hi - pair_lo,
                   e_arg, u_arg);
            // 逐步取证（NINFER_DF2SEL 门控）：把本步的 top-K 候选与其 unary 全量打印，
            // 供离线按列归因：正确 token 是否在候选里（头的问题）/ 在却选错（walk 的问题）。
            printf("[df2cand] s=%d pred=%d chosen=%d k=%d", s, pred, chosen, (int)K);
            for (int c = 0; c < K; ++c) {
                printf(" %d:%.2f",
                       candidates[dflash2_selector_candidate_offset(b, batch, s, steps, c)],
                       unary[dflash2_selector_candidate_offset(b, batch, s, steps, c)]);
            }
            printf("\n");
        }

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
