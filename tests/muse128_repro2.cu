// muse128_repro2.cu — Muse small_t warmup 形状复现: 满 envelope + 141-token cache + graph capture.
#include "ops/launcher/gqa_attention.h"

#include <cuda_bf16.h>

#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    using namespace ninfer;
    const int D = 128, QH = 32, KVH = 2;
    const int T = 1, batch = 1;
    const int tokens_ctx = argc > 1 ? atoi(argv[1]) : 141;  // 预填上下文
    const bool use_graph = argc > 2 ? atoi(argv[2]) != 0 : true;
    const float scale = 0.342f;

    std::vector<__nv_bfloat16> q(D * QH * T * batch, __float2bfloat16(0.1f));
    std::vector<__nv_bfloat16> k(D * KVH * T * batch, __float2bfloat16(0.2f));
    std::vector<__nv_bfloat16> v(D * KVH * T * batch, __float2bfloat16(0.3f));
    std::vector<std::int32_t> pos{tokens_ctx}, table_rows{0};

    // cache: 3 页 (64*3=192 >= 141)
    const int pages = (tokens_ctx + 63) / 64;
    const std::size_t plane = 64ULL * KVH * D * pages;
    __nv_bfloat16 *d_kp, *d_vp;
    cudaMalloc(&d_kp, plane * 2);
    cudaMalloc(&d_vp, plane * 2);
    cudaMemset(d_kp, 0x3c, plane * 2);
    cudaMemset(d_vp, 0x3c, plane * 2);
    std::vector<std::int32_t> bt(pages);
    for (int i = 0; i < pages; ++i) { bt[i] = i; }
    std::int32_t* d_bt;
    cudaMalloc(&d_bt, pages * 4);
    cudaMemcpy(d_bt, bt.data(), pages * 4, cudaMemcpyHostToDevice);

    ninfer::PagedKVBatchLayerView cache{};
    cache.k_pages = Tensor(d_kp, DType::BF16, {D, KVH, 64, pages});
    cache.v_pages = Tensor(d_vp, DType::BF16, {D, KVH, 64, pages});
    cache.block_tables = Tensor(d_bt, DType::I32, {64, 1});
    cache.head_dim = D;
    cache.num_kv_heads = KVH;
    cache.dtype = DType::BF16;

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

    const int splits = 85;
    float *d_pa, *d_pm, *d_pl;
    cudaMalloc(&d_pa, (std::size_t)D * QH * T * batch * splits * 4);
    cudaMalloc(&d_pm, (std::size_t)QH * T * batch * splits * 4);
    cudaMalloc(&d_pl, (std::size_t)QH * T * batch * splits * 4);
    Tensor pa(d_pa, DType::FP32, {D, QH, T * splits});
    Tensor pm(d_pm, DType::FP32, {QH, T * splits});
    Tensor pl(d_pl, DType::FP32, {QH, T * splits});

    // 满 envelope: warmup 的 graph replay interval 形状
    ops::GqaExecutionEnvelope env{1u, 2048u};

    auto launch = [&] {
        ops::detail::gqa_attention_small_t_launch(
            dq, dk, dv, dpos, Tensor{}, drows, scale, cache, env,
            0, T, pa, pm, pl, dout, 0);
    };

    printf("tokens_ctx=%d graph=%d launching...\n", tokens_ctx, use_graph);
    printf("splits(capacity)=%d\n",
           ops::detail::gqa_attention_split_capacity(QH, T, DType::BF16, env));
    fflush(stdout);
    if (use_graph) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);
        launch();
        cudaStreamEndCapture(0, &g);
        cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
        cudaGraphLaunch(ge, 0);
        printf("graph launched\n");
    } else {
        launch();
        printf("eager launched\n");
    }
    fflush(stdout);
    cudaError_t e = cudaDeviceSynchronize();
    printf("sync=%s\n", cudaGetErrorString(e));
    return 0;
}
