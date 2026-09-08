// Implements: include/ninfer/ops/suffix_lookup.h
// One block per batch row; threads scan candidate offsets in parallel and a
// block-wide reduction keeps the (longest, latest) tail match.

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops {

namespace {
constexpr int kSuffixLookupBlock = 256;
}

template <int Block>
__launch_bounds__(Block) __global__ void suffix_lookup_kernel(
    const std::int32_t* __restrict__ ids, const std::int32_t* __restrict__ starts,
    const std::int32_t* __restrict__ lengths, std::int32_t batch, std::int32_t query,
    std::int32_t min_len, std::int32_t continuation_tokens, std::int32_t* __restrict__ best_len,
    std::int32_t* __restrict__ best_offset, std::int32_t* __restrict__ continuation) {
    const int b = static_cast<int>(blockIdx.x);
    if (b >= batch) { return; }
    const int tid   = static_cast<int>(threadIdx.x);
    const int start = starts[b];
    const int len   = lengths[b];
    // 只搜严格更早的非重叠窗口: o+query <= start 且给续写留空间
    const int limit_a = len - query - continuation_tokens;
    const int limit_b = start - query;
    const int limit = limit_a < limit_b ? limit_a : limit_b;   // device 端 min
    // (match_len, offset) 联合极值: 长匹配优先, 同长取最新 (offset 大)。
    int best_l = 0;
    int best_o = -1;
    for (int o = tid; o < limit; o += Block) {
        // 从尾部逐 token 对齐 (查询 = ids[start .. start+query))
        int l = 0;
        for (int q = 0; q < query; ++q) {
            if (ids[start + query - 1 - q] == ids[o + query - 1 - q]) {
                l = q + 1;
            } else {
                break;
            }
        }
        if (l >= min_len && (l > best_l || (l == best_l && o > best_o))) {
            best_l = l;
            best_o = o;   // 候选窗口起点; 匹配段起点 = o + query - l
        }
    }
    // 块内归约
    __shared__ int shared_l[Block];
    __shared__ int shared_o[Block];
    shared_l[tid] = best_l;
    shared_o[tid] = best_o;
    __syncthreads();
    for (int stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const int ol = shared_l[tid + stride];
            const int oo = shared_o[tid + stride];
            if (ol > shared_l[tid] || (ol == shared_l[tid] && oo > shared_o[tid])) {
                shared_l[tid] = ol;
                shared_o[tid] = oo;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        const int l = shared_l[0];
        const int o = shared_o[0];
        best_len[b]   = l;
        best_offset[b] = o;
        const int cbase = b * continuation_tokens;
        if (l >= min_len) {
            for (int k = 0; k < continuation_tokens; ++k) {
                const int pos = o + query + k;
                continuation[cbase + k] = (pos >= 0 && pos < len) ? ids[pos] : -1;
            }
        } else {
            for (int k = 0; k < continuation_tokens; ++k) {
                continuation[cbase + k] = -1;
            }
        }
    }
}

} // namespace ninfer::ops
