// ninfer::ops - split-KV verify 算子对实现 (v1 正确性优先, 标量核;
// 数学与 split_attention_ref.h CPU 参考一致, 供对拍)。
#include "ninfer/ops/split_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer::ops {
namespace {

// 每 (b, hq, tq) 一个线程: 局部 softmax partials (Kahan 不用, 标量足够)。
__global__ void split_attention_local_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    int batch, int query_heads, int kv_heads, int kv_group, int t_local, int dim,
    float scale, float* __restrict__ local_max, float* __restrict__ local_sum,
    float* __restrict__ pv, int q_rows, int q_base, int kv_offset) {
    const int total = batch * query_heads * q_rows;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total) { return; }
    const int tq = row % q_rows;
    const int hq = (row / q_rows) % query_heads;
    const int b = row / (q_rows * query_heads);
    // GQA: hq 属于组 g = hq / kv_group; 该组 kv 头 = g
    const int g = hq / kv_group;
    const float* qrow = q + (static_cast<std::int64_t>(b * query_heads + hq) * q_rows + tq) * dim;
    float m = -1e30f;
    float s = 0.0f;
    // pv 就地累加 (标量循环)
    float* pvrow = pv + (static_cast<std::int64_t>(row)) * dim;
    for (int d = 0; d < dim; ++d) { pvrow[d] = 0.0f; }
    const int causal_end = q_base + tq - kv_offset + 1;
    for (int tk = 0; tk < t_local; ++tk) {
        if (tk >= causal_end) { break; }
        const float* krow =
            k + (static_cast<std::int64_t>(b * kv_heads + g) * t_local + tk) * dim;
        float acc = 0.0f;
        for (int d = 0; d < dim; ++d) { acc += qrow[d] * krow[d]; }
        acc *= scale;
        // 因果: tq 只看 tk <= tq 的局部位置 (split-KV 分片按位置连续时,

        const float m_new = acc > m ? acc : m;
        const float corr = std::exp(m - m_new);
        s = s * corr + std::exp(acc - m_new);
        const float* vrow =
            v + (static_cast<std::int64_t>(b * kv_heads + g) * t_local + tk) * dim;
        for (int d = 0; d < dim; ++d) { pvrow[d] = pvrow[d] * corr + std::exp(acc - m_new) * vrow[d]; }
        m = m_new;
    }
    local_max[row] = m;
    local_sum[row] = s;
}

__global__ void split_attention_combine_kernel(
    const float* const* __restrict__ lmax, const float* const* __restrict__ lsum,
    const float* const* __restrict__ pvs, int shards, int rows, int dim,
    float* __restrict__ out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    float m = -1e30f;
    float s = 0.0f;
    float acc[128];   // dim <= 128 (v1 约束, wrapper 校验)
    for (int d = 0; d < dim; ++d) { acc[d] = 0.0f; }
    for (int r = 0; r < shards; ++r) {
        const float rm = lmax[r][row];
        const float m_new = rm > m ? rm : m;
        const float corr = std::exp(m - m_new);
        s = s * corr + std::exp(rm - m_new) * lsum[r][row];
        const float* pvrow = pvs[r] + static_cast<std::int64_t>(row) * dim;
        for (int d = 0; d < dim; ++d) { acc[d] = acc[d] * corr + std::exp(rm - m_new) * pvrow[d]; }
        m = m_new;
    }
    float* outrow = out + static_cast<std::int64_t>(row) * dim;
    for (int d = 0; d < dim; ++d) { outrow[d] = (s > 0.0f ? acc[d] / s : 0.0f); }
}

} // namespace

void split_attention_local(const float* q, const float* k, const float* v,
                           std::int32_t batch, std::int32_t query_heads,
                           std::int32_t kv_heads, std::int32_t kv_group,
                           std::int32_t t_local, std::int32_t dim, float scale,
                           float* local_max, float* local_sum, float* pv,
                           cudaStream_t stream, std::int32_t q_rows,
                           std::int32_t q_base, std::int32_t kv_offset) {
    if (q_rows <= 0) { q_rows = t_local; }
    if (dim > 128 || batch <= 0 || query_heads <= 0 || kv_heads <= 0) { return; }
    const int total = batch * query_heads * q_rows;
    const int block = 256;
    split_attention_local_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        q, k, v, batch, query_heads, kv_heads, kv_group, t_local, dim, scale, local_max,
        local_sum, pv, q_rows, q_base, kv_offset);
}

void split_attention_combine(const float* const* local_max, const float* const* local_sum,
                             const float* const* pv, std::int32_t shards,
                             std::int32_t rows, std::int32_t dim, float* out,
                             cudaStream_t stream) {
    const int block = 128;
    split_attention_combine_kernel<<<(rows + block - 1) / block, block, 0, stream>>>(
        local_max, local_sum, pv, shards, rows, dim, out);
}

} // namespace ninfer::ops
