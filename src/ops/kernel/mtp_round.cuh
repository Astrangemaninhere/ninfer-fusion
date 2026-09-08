#pragma once

// Implements: include/ninfer/ops/mtp_round.h
// Match: request-major fixed K=1..5 autoregressive MTP round transition.

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void mtp_prepare_next_round_kernel(
    const std::int32_t* verify_ids, const std::int32_t* next_anchors, const std::int32_t* accepted,
    const std::int32_t* updated_frontiers, const std::int32_t* remaining_budgets,
    const std::int32_t* licensed_counts, const std::int32_t* rope_deltas,
    std::int32_t* alignment_ids, std::int32_t* next_extents, std::int32_t* ar_positions,
    std::int32_t* ar_rope_positions, std::int32_t* ar_valid_columns, std::int32_t k,
    std::int32_t ar_step_stride, std::int32_t max_context,
    const std::int32_t* __restrict__ svip_cuts) {
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int a         = accepted[row];
    a             = a < 0 ? 0 : (a > k ? k : a);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        alignment_ids[row * T + j] = j < a ? verify_ids[row * T + j + 1] : next_anchors[row];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int licensed       = licensed_counts[row];
        const int remaining      = remaining_budgets[row] - licensed;
        const int budget_extent  = remaining > 1 ? remaining - 1 : 0;
        const int context_extent = max_context - updated_frontiers[row] - 1;
        int next                 = budget_extent < context_extent ? budget_extent : context_extent;
        next                     = next < 0 ? 0 : (next > k ? k : next);
        if (svip_cuts != nullptr) {
            const int cut = svip_cuts[row];
            next = cut < next ? (cut < 0 ? 0 : cut) : next;
        }
        next_extents[row]        = next;
        const int steps          = k > 1 ? k - 1 : 1;
        for (int s = 0; s < steps; ++s) {
            const int offset          = s * ar_step_stride + row;
            const int position        = updated_frontiers[row] + s;
            ar_positions[offset]      = position;
            ar_rope_positions[offset] = position + rope_deltas[row];
            ar_valid_columns[offset]  = s + 1 < next ? 1 : 0;
        }
    }
}


// SVIP: entropy-based draft length cap. One block per row scans the verify
// columns col = accepted+1 .. k (the positions that predict next round's
// drafts) and caps the draft count before the first column whose softmax
// entropy (nats) exceeds the threshold. A fully accepted row keeps the
// caller-extent behavior (cap = k).
__global__ void mtp_svip_entropy_extents_kernel(
    const __nv_bfloat16* __restrict__ logits, const std::int32_t* __restrict__ accepted,
    std::int32_t* __restrict__ cuts, int vocab, int cols, int batch, float threshold) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= batch) { return; }
    const int tid       = static_cast<int>(threadIdx.x);
    const int a         = accepted[row];
    const std::int64_t v_stride = static_cast<std::int64_t>(cols) * batch;
    const __nv_bfloat16* row_base = logits + row;  // layout [vocab][cols][batch]
    const int k = cols - 1;
    cuts[row] = k;

    for (int col = a + 1; col < cols; ++col) {
        const __nv_bfloat16* col_base = row_base + static_cast<std::int64_t>(col) * batch;
        float m = -CUDART_INF_F;
        for (int v = tid; v < vocab; v += blockDim.x) {
            m = fmaxf(m, __bfloat162float(col_base[static_cast<std::int64_t>(v) * v_stride]));
        }
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        }
        __shared__ float s_max;
        if (tid == 0) { s_max = m == -CUDART_INF_F ? 0.0F : m; }
        __syncthreads();
        m = s_max;

        float sum = 0.0F;
        float sln = 0.0F;
        for (int v = tid; v < vocab; v += blockDim.x) {
            const float x = __bfloat162float(col_base[static_cast<std::int64_t>(v) * v_stride]);
            const float e = __expf(x - m);
            sum += e;
            sln += e * (x - m);
        }
        __shared__ float s_sum[8];
        __shared__ float s_sln[8];
        const int warp = tid >> 5;
        const int lane = tid & 31;
        for (int off = 16; off > 0; off >>= 1) {
            sum = sum + __shfl_xor_sync(0xffffffffu, sum, off);
            sln = sln + __shfl_xor_sync(0xffffffffu, sln, off);
        }
        if (lane == 0) { s_sum[warp] = sum; s_sln[warp] = sln; }
        __syncthreads();
        if (warp == 0) {
            float tsum = lane < (blockDim.x >> 5) ? s_sum[lane] : 0.0F;
            float tsln = lane < (blockDim.x >> 5) ? s_sln[lane] : 0.0F;
            for (int off = 16; off > 0; off >>= 1) {
                tsum = tsum + __shfl_xor_sync(0xffffffffu, tsum, off);
                tsln = tsln + __shfl_xor_sync(0xffffffffu, tsln, off);
            }
            if (lane == 0) {
                const float entropy = logf(tsum) - tsln / tsum;
                if (entropy > threshold) {
                    cuts[row] = col - a - 1;
                }
            }
        }
        __syncthreads();
        if (cuts[row] != k) { break; }  // cut found by this or an earlier column
    }
}

} // namespace ninfer::ops
