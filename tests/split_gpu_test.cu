// split_gpu_test.cu — split_attention v2 对拍: q 草稿窗不分片 + kv 位置分片,
// 全局因果 (q_base/kv_offset); 金标准 = attention_mono 全长因果取尾窗行。
#include "ninfer/ops/split_attention.h"
#include "ops/split_attention_ref.h"

#include <cstdio>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const int batch = 2, qheads = 4, kvheads = 2, kgroup = 2;
    const int T = 32, D = 16;
    const float scale = 1.0f / 8.0f;
    std::vector<float> q(batch * qheads * T * D), kv(batch * kvheads * T * D),
        v(batch * kvheads * T * D);
    for (auto& x : q) { x = u(rng); }
    for (auto& x : kv) { x = u(rng); }
    for (auto& x : v) { x = u(rng); }

    // 金标准: 全长因果 attention; draft 窗 = 尾 ts 行
    std::vector<float> ref;
    ninfer::ops::ref::attention_mono(q, kv, v, batch, qheads, kvheads, kgroup, T, D, scale,
                                     ref);

    float *d_q, *d_k, *d_v;
    cudaMalloc(&d_q, q.size() * 4);
    cudaMalloc(&d_k, kv.size() * 4);
    cudaMalloc(&d_v, v.size() * 4);
    cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, kv.data(), kv.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
    (void)d_q; (void)d_k; (void)d_v;  // 全量缓冲 (未直接使用; 分片走 local 拷贝)

    int fails = 0;
    for (int shards : {1, 2, 4}) {
        const int ts = T / shards;      // 每片 kv 段长 (也是 draft 窗长)
        const int q_base = T - ts;      // draft 窗 global 起点
        // q 窗 (每片同一份): [b,hq,i] -> q[.., q_base+i, ..]
        std::vector<float> q_win(batch * qheads * ts * D);
        for (int b = 0; b < batch; ++b)
            for (int h = 0; h < qheads; ++h)
                for (int i = 0; i < ts; ++i)
                    for (int d = 0; d < D; ++d) {
                        q_win[((b * qheads + h) * ts + i) * D + d] =
                            q[((b * qheads + h) * T + q_base + i) * D + d];
                    }
        const int rows = batch * qheads * ts;
        std::vector<float*> pmax(shards), psum(shards), ppv(shards);
        float **d_pmax, **d_psum, **d_ppv;
        cudaMalloc(&d_pmax, shards * 8);
        cudaMalloc(&d_psum, shards * 8);
        cudaMalloc(&d_ppv, shards * 8);
        float* d_qwin = nullptr;
        for (int r = 0; r < shards; ++r) {
            // kv 段拷贝: 位置 [r*ts, (r+1)*ts)
            std::vector<float> kv_seg(batch * kvheads * ts * D);
            std::vector<float> v_seg(batch * kvheads * ts * D);
            for (int b = 0; b < batch; ++b)
                for (int h = 0; h < kvheads; ++h)
                    for (int t = 0; t < ts; ++t)
                        for (int d = 0; d < D; ++d) {
                            kv_seg[((b * kvheads + h) * ts + t) * D + d] =
                                kv[((b * kvheads + h) * T + r * ts + t) * D + d];
                            v_seg[((b * kvheads + h) * ts + t) * D + d] =
                                v[((b * kvheads + h) * T + r * ts + t) * D + d];
                        }
            if (!d_qwin) { cudaMalloc(&d_qwin, q_win.size() * 4);
                cudaMemcpy(d_qwin, q_win.data(), q_win.size() * 4, cudaMemcpyHostToDevice); }
            float *d_ks, *d_vs, *d_lm, *d_ls, *d_lp;
            cudaMalloc(&d_ks, kv_seg.size() * 4);
            cudaMalloc(&d_vs, v_seg.size() * 4);
            cudaMalloc(&d_lm, rows * 4);
            cudaMalloc(&d_ls, rows * 4);
            cudaMalloc(&d_lp, rows * D * 4);
            cudaMemcpy(d_ks, kv_seg.data(), kv_seg.size() * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_vs, v_seg.data(), v_seg.size() * 4, cudaMemcpyHostToDevice);
            // v2: q_rows=ts, q_base, kv_offset=r*ts (全局因果)
            ninfer::ops::split_attention_local(d_qwin, d_ks, d_vs, batch, qheads, kvheads,
                                               kgroup, ts, D, scale, d_lm, d_ls, d_lp, 0,
                                               ts, q_base, r * ts);
            pmax[r] = d_lm;
            psum[r] = d_ls;
            ppv[r] = d_lp;
            cudaFree(d_ks); cudaFree(d_vs);
        }
        cudaMemcpy(d_pmax, pmax.data(), shards * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(d_psum, psum.data(), shards * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ppv, ppv.data(), shards * 8, cudaMemcpyHostToDevice);
        std::vector<float> out(rows * D);
        float* d_out;
        cudaMalloc(&d_out, out.size() * 4);
        ninfer::ops::split_attention_combine(d_pmax, d_psum, d_ppv, shards, rows, D, d_out, 0);
        cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaFree(d_out);
        cudaFree(d_pmax); cudaFree(d_psum); cudaFree(d_ppv);
        cudaFree(d_qwin); d_qwin = nullptr;
        for (int r = 0; r < shards; ++r) {
            cudaFree(pmax[r]); cudaFree(psum[r]); cudaFree(ppv[r]);
        }
        // 对照金标准尾窗行 [q_base, T)
        float maxerr = 0.0f;
        for (int b = 0; b < batch; ++b)
            for (int h = 0; h < qheads; ++h)
                for (int i = 0; i < ts; ++i)
                    for (int d = 0; d < D; ++d) {
                        float a = ref[((b * qheads + h) * T + q_base + i) * D + d];
                        float c = out[((b * qheads + h) * ts + i) * D + d];
                        float e = a > c ? a - c : c - a;
                        if (e > maxerr) { maxerr = e; }
                    }
        bool pass = maxerr < 1e-3f;
        fails += !pass;
        printf("shards=%d maxerr=%.2e %s\n", shards, maxerr, pass ? "PASS" : "FAIL");
    }
    printf("fails=%d\n", fails);
    return fails == 0 ? 0 : 1;
}
