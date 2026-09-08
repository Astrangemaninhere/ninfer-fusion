// Implements: include/ninfer/ops/vocab_topk16.h
#pragma once

#include <cuda_bf16.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops {
namespace {
constexpr int kVtkK = 16;
}

template <int Block>
__launch_bounds__(Block) __global__ void vocab_topk16_kernel(
    const __nv_bfloat16* __restrict__ logits, int vocab, int tokens,
    std::int32_t* __restrict__ ids_out, __nv_bfloat16* __restrict__ vals_out) {
    const int t = static_cast<int>(blockIdx.x);
    if (t >= tokens) { return; }
    const int tid = static_cast<int>(threadIdx.x);
    float lv[kVtkK];
    int li[kVtkK];
#pragma unroll
    for (int k = 0; k < kVtkK; ++k) { lv[k] = -1e30f; li[k] = vocab; }
    for (int v = tid; v < vocab; v += Block) {
        const float x = __bfloat162float(logits[static_cast<std::int64_t>(v) * tokens + t]);
        if (x < lv[kVtkK - 1]) { continue; }
        for (int k = 0; k < kVtkK; ++k) {
            if (x > lv[k] || (x == lv[k] && v < li[k])) {
                for (int j = kVtkK - 1; j > k; --j) { lv[j] = lv[j - 1]; li[j] = li[j - 1]; }
                lv[k] = x;
                li[k] = v;
                break;
            }
        }
    }
    __shared__ float sv[Block][kVtkK];
    __shared__ int si[Block][kVtkK];
#pragma unroll
    for (int k = 0; k < kVtkK; ++k) { sv[tid][k] = lv[k]; si[tid][k] = li[k]; }
    __syncthreads();
    for (int stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            // merge sv[tid] and sv[tid+stride] (descending, id tiebreak low first)
            float ov[kVtkK];
            int oi[kVtkK];
            int a = 0, b = 0;
            for (int slot = 0; slot < kVtkK; ++slot) {
                bool take_a;
                if (a >= kVtkK) take_a = false;
                else if (b >= kVtkK) take_a = true;
                else take_a = sv[tid][a] > sv[tid + stride][b] ||
                              (sv[tid][a] == sv[tid + stride][b] &&
                               si[tid][a] < si[tid + stride][b]);
                if (take_a) { ov[slot] = sv[tid][a]; oi[slot] = si[tid][a]; ++a; }
                else { ov[slot] = sv[tid + stride][b]; oi[slot] = si[tid + stride][b]; ++b; }
            }
#pragma unroll
            for (int k = 0; k < kVtkK; ++k) { sv[tid][k] = ov[k]; si[tid][k] = oi[k]; }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int k = 0; k < kVtkK; ++k) {
            ids_out[t * kVtkK + k] = si[0][k];
            vals_out[t * kVtkK + k] = __float2bfloat16(sv[0][k]);
        }
    }
}

} // namespace ninfer::ops
