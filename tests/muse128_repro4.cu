// muse128_repro4.cu — W1 minimal repro: Muse nvfp4(K)+iso3(V) small_t decode.
// serve crashes on first generation with sticky cudaErrorInvalidValue surfacing
// at the cudaFuncSetAttribute line. bf16 repro passes, so pin the fault to this
// cache format path. Eager, no graph, tiny context.
#include "ops/launcher/gqa_attention.h"

#include <cuda_bf16.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    using namespace ninfer;
    const int D = 128, QH = 32, KVH = 2;
    const int T = (argc > 1 ? atoi(argv[1]) : 1);
    const int batch = 1;
    const int tokens_ctx = argc > 2 ? atoi(argv[2]) : 141;
    const float scale = 0.342f;

    std::vector<__nv_bfloat16> q(D * QH * T * batch, __float2bfloat16(0.1f));
    std::vector<__nv_bfloat16> k(D * KVH * T * batch, __float2bfloat16(0.2f));
    std::vector<__nv_bfloat16> v(D * KVH * T * batch, __float2bfloat16(0.3f));
    std::vector<std::int32_t> pos{tokens_ctx}, table_rows{0};

    // nvfp4 cache planes: codes U8 [D/2, KVH, 64, pages], scales FP8 [D/16, KVH, 64, pages]
    const int pages = (tokens_ctx + 63) / 64;
    const std::size_t codes_plane = (D / 2) * KVH * 64 * pages;
    const std::size_t scale_plane = (D / 16) * KVH * 64 * pages;
    std::uint8_t *d_kc, *d_vc, *d_ks, *d_vs;
    cudaMalloc(&d_kc, codes_plane); cudaMalloc(&d_vc, codes_plane);
    cudaMalloc(&d_ks, scale_plane); cudaMalloc(&d_vs, scale_plane);
    cudaMemset(d_kc, 0x22, codes_plane); cudaMemset(d_vc, 0x22, codes_plane);
    cudaMemset(d_ks, 0x38, scale_plane); cudaMemset(d_vs, 0x38, scale_plane);  // 1.0 scale
    std::vector<std::int32_t> bt(pages);
    for (int i = 0; i < pages; ++i) { bt[i] = i; }
    std::int32_t* d_bt;
    cudaMalloc(&d_bt, pages * 4);
    cudaMemcpy(d_bt, bt.data(), pages * 4, cudaMemcpyHostToDevice);

    PagedKVBatchLayerView cache{};
    cache.k_pages = Tensor(d_kc, DType::U8, {D / 2, KVH, 64, pages});
    cache.v_pages = Tensor(d_vc, DType::U8, {D / 2, KVH, 64, pages});
    cache.k_scale_pages = Tensor(d_ks, DType::FP8_E4M3FN, {D / 16, KVH, 64, pages});
    cache.v_scale_pages = Tensor(d_vs, DType::FP8_E4M3FN, {D / 16, KVH, 64, pages});
    cache.block_tables = Tensor(d_bt, DType::I32, {64, 1});
    cache.head_dim = D;
    cache.num_kv_heads = KVH;
    cache.dtype = DType::NVFP4;
    cache.v_dtype = DType::ISO3;
    cache.quant_group = 16;
    cache.v_quant_group = 16;

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

    const int splits = 32;
    float *d_pa, *d_pm, *d_pl;
    cudaMalloc(&d_pa, (std::size_t)D * QH * T * batch * splits * 4);
    cudaMalloc(&d_pm, (std::size_t)QH * T * batch * splits * 4);
    cudaMalloc(&d_pl, (std::size_t)QH * T * batch * splits * 4);
    Tensor pa(d_pa, DType::FP32, {D, QH, T * splits});
    Tensor pm(d_pm, DType::FP32, {QH, T * splits});
    Tensor pl(d_pl, DType::FP32, {QH, T * splits});

    ops::GqaExecutionEnvelope env{1u, 2048u};

    printf("T=%d ctx=%d nvfp4+iso3 launching...\n", T, tokens_ctx);
    fflush(stdout);
    ops::detail::gqa_attention_small_t_launch(
        dq, dk, dv, dpos, Tensor{}, drows, scale, cache, env,
        0, T, pa, pm, pl, dout, 0);
    cudaError_t le = cudaGetLastError();
    printf("launch err=%s\n", cudaGetErrorString(le));
    fflush(stdout);
    cudaError_t e = cudaDeviceSynchronize();
    printf("sync=%s\n", cudaGetErrorString(e));
    return 0;
}
