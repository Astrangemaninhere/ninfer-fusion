// muse128_repro.cu — GqaMuseGeometry small_t 最小复现 (32q/2kv/128d bf16).
// 定位 warmup 死循环: 哪个 kernel、哪一行。
#include "ops/launcher/gqa_attention.h"

#include <cuda_bf16.h>

#include <cstdio>
#include <vector>

int main() {
    using namespace ninfer;
    const int D = 128, QH = 32, KVH = 2, T = 1, batch = 1;
    const float scale = 0.342f;
    // q: [D, QHeads, width, batch] bf16; k/v: [D, KVHeads, ...]
    std::vector<__nv_bfloat16> q(D * QH * T * batch, __float2bfloat16(0.1f));
    std::vector<__nv_bfloat16> k(D * KVH * T * batch, __float2bfloat16(0.2f));
    std::vector<__nv_bfloat16> v(D * KVH * T * batch, __float2bfloat16(0.3f));
    std::vector<std::int32_t> pos{0}, table_rows{0};

    // cache: 一页 64 tok × KVH × D bf16 (k/v 各一)
    const std::size_t plane = 64ULL * KVH * D;
    std::vector<__nv_bfloat16> k_page(plane), v_page(plane);
    __nv_bfloat16 *d_kp, *d_vp;
    cudaMalloc(&d_kp, plane * 2);
    cudaMalloc(&d_vp, plane * 2);
    cudaMemcpy(d_kp, k_page.data(), plane * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vp, v_page.data(), plane * 2, cudaMemcpyHostToDevice);
    std::vector<std::int32_t> bt{0};  // block table: 逻辑页0 -> 物理页0
    std::int32_t* d_bt;
    cudaMalloc(&d_bt, 4);
    cudaMemcpy(d_bt, bt.data(), 4, cudaMemcpyHostToDevice);

    ninfer::PagedKVBatchLayerView cache{};
    cache.k_pages = Tensor(d_kp, DType::BF16, {D, KVH, 64});
    cache.v_pages = Tensor(d_vp, DType::BF16, {D, KVH, 64});
    cache.block_tables = Tensor(d_bt, DType::I32, {64, 1});
    cache.head_dim = D;
    cache.num_kv_heads = KVH;
    cache.layer_index = 0;
    cache.dtype = DType::BF16;

    Tensor tq(q.data(), DType::BF16, {D, QH, T, batch});
    Tensor tk(k.data(), DType::BF16, {D, KVH, T, batch});
    Tensor tv(v.data(), DType::BF16, {D, KVH, T, batch});
    Tensor tpos(pos.data(), DType::I32, {batch});
    Tensor trows(table_rows.data(), DType::I32, {batch});
    __nv_bfloat16 *d_q, *d_k, *d_v, *d_out;
    std::int32_t *d_pos, *d_rows;
    cudaMalloc(&d_q, q.size() * 2); cudaMalloc(&d_k, k.size() * 2);
    cudaMalloc(&d_v, v.size() * 2); cudaMalloc(&d_out, q.size() * 2);
    cudaMalloc(&d_pos, 4); cudaMalloc(&d_rows, 4);
    cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k.data(), k.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v.data(), v.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, pos.data(), 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, table_rows.data(), 4, cudaMemcpyHostToDevice);
    Tensor dq(d_q, DType::BF16, {D, QH, T, batch});
    Tensor dk(d_k, DType::BF16, {D, KVH, T, batch});
    Tensor dv(d_v, DType::BF16, {D, KVH, T, batch});
    Tensor dpos(d_pos, DType::I32, {batch});
    Tensor drows(d_rows, DType::I32, {batch});
    Tensor dout(d_out, DType::BF16, {D, QH, T, batch});

    // partials: splits 由 helper 内部 envelope 决定; 给富裕容量
    const int splits = 85;  // DecodeSplits = 85 * 1
    float *d_pa, *d_pm, *d_pl;
    cudaMalloc(&d_pa, (std::size_t)D * QH * T * batch * splits * 4);
    cudaMalloc(&d_pm, (std::size_t)QH * T * batch * splits * 4);
    cudaMalloc(&d_pl, (std::size_t)QH * T * batch * splits * 4);
    Tensor pa(d_pa, DType::FP32, {D, QH, T * splits});
    Tensor pm(d_pm, DType::FP32, {QH, T * splits});
    Tensor pl(d_pl, DType::FP32, {QH, T * splits});

    ops::GqaExecutionEnvelope env{};  // 默认: 全窗口
    printf("launching gqa_attention_small_t_muse...\n"); fflush(stdout);
    ops::detail::gqa_attention_small_t_muse(
        dq, dk, dv, dpos, Tensor{}, drows, scale, cache, env,
        0, T, pa, pm, pl, dout, 0);
    printf("launched, syncing...\n"); fflush(stdout);
    cudaError_t e = cudaDeviceSynchronize();
    printf("sync=%s\n", cudaGetErrorString(e));
    std::vector<__nv_bfloat16> out(q.size());
    cudaMemcpy(out.data(), d_out, q.size() * 2, cudaMemcpyDeviceToHost);
    printf("out[0..3]=%.4f %.4f %.4f %.4f\n", __bfloat162float(out[0]),
           __bfloat162float(out[1]), __bfloat162float(out[2]), __bfloat162float(out[3]));
    return 0;
}
