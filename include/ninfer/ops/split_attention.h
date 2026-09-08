#pragma once
// ninfer::ops - split-KV verify (1Cat/多卡 verify 注意力的算子对, v1 落地).
//
// 语义 (vLLM split-KV verify / 分布式注意力):
//   长上下文 KV 按注意力头分片到 R 个 rank (或按位置分片)。每个分片独立算
//   局部注意力并输出 *partials* (flash 在线 softmax 的可合并统计), 再由
//   合并核/宿主把各分片合成完整 softmax 输出 —— 全程只交换 O(T) 统计量,
//   不交换 KV。
//
// 算子 1: split_attention_local
//   输入: q[B,Hq,D] (D=query 维), k/v 局部片 [B, Hk_local, T_local, D],
//         mask 可选 (I32 -inf 语义由调用方预乘; v1 不做 mask, 位置填充走
//         pad 输入), scale = D^-0.5
//   输出 per (b,hq,tq): local_max[B*Hq*Tq], local_sum[...], pv[B*Hq*Tq,D]
//   (Hq 每头只 attend 其 GQA 组内局部 kv 头)
//
// 算子 2: split_attention_combine
//   输入: 各分片 partials (R 组 local_max/sum/pv, 行主序同形状)
//   输出: out[B*Hq*Tq, D] = softmax 合并结果 (在线 softmax 精确合并)
//   combine 是纯宿主可测的标量数学, 提供 CPU 参考 (split_attention_ref.h)。
#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops {

// 局部注意力: 每个 (b, hq, tq) 一个线程块内线程? v1 正确性优先: 每
// (b,hq,tq) 一个线程, 串行扫 T_local (后续按 flash/mma 优化, 语义不变)。
void split_attention_local(const float* q, const float* k, const float* v,
                           std::int32_t batch, std::int32_t query_heads,
                           std::int32_t kv_heads, std::int32_t kv_group,
                           std::int32_t t_local, std::int32_t dim, float scale,
                           float* local_max, float* local_sum, float* pv,
                           cudaStream_t stream, std::int32_t q_rows = 0,
                           std::int32_t q_base = 0, std::int32_t kv_offset = 0);

// 合并 R 组分片 partials -> 完整输出 (每行 T 个 query 位置独立)。
void split_attention_combine(const float* const* local_max, const float* const* local_sum,
                             const float* const* pv, std::int32_t shards,
                             std::int32_t rows, std::int32_t dim, float* out,
                             cudaStream_t stream);

} // namespace ninfer::ops
