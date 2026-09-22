#pragma once

// Kernel for ops::mtp_proposal_topk (include/ninfer/ops/mtp_proposal_topk.h).
//
// One block per token. Each thread keeps its own top_l of the strided slice of the row it owns
// in shared memory; one thread then merges the per-thread candidates by (value desc, row index
// asc). The rows of a proposal head are long (the text output vocabulary) and top_l is small,
// so a per-thread insertion into a fixed top_l array is cheaper than a sort.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

template <int kBlock>
__global__ void mtp_proposal_topk_kernel(const __nv_bfloat16* __restrict__ logits,
                                         int rows, int tokens, int top_l,
                                         int* __restrict__ ids) {
    const int t = static_cast<int>(blockIdx.x);
    if (t >= tokens) { return; }
    const int tid = static_cast<int>(threadIdx.x);

    // Per-thread top_l, kept sorted descending with the tie-break already applied inside the
    // thread by scanning its own slice in INCREASING row order and inserting only on a strict
    // improvement (so an earlier, equal-valued row is never displaced).
    extern __shared__ unsigned char smem_raw[];
    float* s_val = reinterpret_cast<float*>(smem_raw);
    int* s_idx   = reinterpret_cast<int*>(s_val + static_cast<std::size_t>(kBlock) * top_l);

    const int mine = (rows + kBlock - 1) / kBlock;
    for (int slot = 0; slot < top_l; ++slot) {
        s_val[static_cast<std::size_t>(tid) * top_l + slot] = -INFINITY;
        s_idx[static_cast<std::size_t>(tid) * top_l + slot] = -1;
    }
    for (int step = 0; step < mine; ++step) {
        const int r = tid + step * kBlock;
        if (r >= rows) { break; }
        const float v = __bfloat162float(logits[static_cast<std::size_t>(r) * tokens + t]);
        if (v == -INFINITY) { continue; }
        // Insert into the thread-local top_l if it beats the current worst.
        if (v > s_val[static_cast<std::size_t>(tid) * top_l + top_l - 1]) {
            int at = top_l - 1;
            while (at > 0 && v > s_val[static_cast<std::size_t>(tid) * top_l + at - 1]) { --at; }
            for (int k = top_l - 1; k > at; --k) {
                s_val[static_cast<std::size_t>(tid) * top_l + k] =
                    s_val[static_cast<std::size_t>(tid) * top_l + k - 1];
                s_idx[static_cast<std::size_t>(tid) * top_l + k] =
                    s_idx[static_cast<std::size_t>(tid) * top_l + k - 1];
            }
            s_val[static_cast<std::size_t>(tid) * top_l + at] = v;
            s_idx[static_cast<std::size_t>(tid) * top_l + at] = r;
        }
    }
    __syncthreads();

    // One thread merges the kBlock * top_l candidates. Comparing (value, index) lexicographically
    // is exactly "value descending, then the lowest row index wins".
    if (tid == 0) {
        const int total = kBlock * top_l;
        for (int out = 0; out < top_l; ++out) {
            int best = -1;
            for (int c = 0; c < total; ++c) {
                if (s_idx[c] < 0) { continue; }
                if (best < 0) { best = c; continue; }
                if (s_val[c] > s_val[best] ||
                    (s_val[c] == s_val[best] && s_idx[c] < s_idx[best])) {
                    best = c;
                }
            }
            if (best < 0) {
                // Fewer distinct rows than top_l: repeat the last chosen index.
                ids[static_cast<std::size_t>(out) * tokens + t] =
                    out == 0 ? 0 : ids[static_cast<std::size_t>(out - 1) * tokens + t];
            } else {
                ids[static_cast<std::size_t>(out) * tokens + t] = s_idx[best];
                s_idx[best] = -1; // consume it
            }
        }
    }
}

} // namespace ninfer::ops
