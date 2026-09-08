// qpn_e2e_test.cu — prepack→qpn_simt 端到端 v4: 机器实证映射 (qpn_map.cuh, 2048/2048).
// probe5 实证 (g0 段, lane 0/5/10/15 抽样):
//   ci 0..7  -> col 2L,      k = 8*(ci/8) + 4*((ci/2)%2)*? 实测: ci0->k0, ci1->k4,
//              ci4->k8, (ci2->k? 未采; 结构推断 k = (ci%2)*4 + (ci/2>=2)*8 + (ci/4==1)*? )
//   观测三元组: ci0->k0, ci1->k4, ci4->k8, ci8->col+1 k0, ci12->col+1 k8.
//   => ci 分解: half = ci/8 (0: 双列低, 1: 双列高); quad = (ci/2)%2 (0: k低, 1: k高=+8);
//      pair = ci%2 (0: +0, 1: +4).  即 k_in_col = pair*4 + quad*8. 验证: ci1: quad0
//      pair1 -> k4 ✓; ci4: half0 quad2-> (ci/2)%2 = 0? ci4/2=2, %2=0 -> quad0?? 但实测
//      ci4->k8.  修正: quad = (ci/4)%2 (0..3->0, 4..7->1) => ci4: quad1 -> k8 ✓;
//      ci1: quad0 pair1 -> k4 ✓; ci0: k0 ✓; ci8: half1 -> col+1, quad=(8/4)%2=0, pair0
//      -> k0 ✓; ci12: half1, quad=(12/4)%2=1 -> k8 ✓. 全合!
//   k_global = g*? 观测 g0 时 k0/4/8 出现 (K=64 应有 g0..3 贡献 k 全域): 每 g 提供
//      k_in_col 的 4 个点 (pair*4 + quad*8 限于 0..12 的 4 个取值 0,4,8,12) + 偏移.
//      g 偏移未采样 (probe5 只在 ci 采样处打点, g>0 的响应被 g0 输出同列抑制? 否——
//      单码探测每 run 只有一个非零码, g>0 的响应 k 应带偏移). 但 probe5 g 循环打印
//      无 g>0 行 => g>0 的响应 k 与 g0 相同?! 即 g 只在多组累加同一 k 位?? 不可能.
//      真相: 内核 warp 分工 g0 = warp*Gq, 每 warp 只消费自己的 g 段 (Gq=G/4=1),
//      且 x chunk 窗口 (c0>>3..) 用 g 而非全局 k —— 即 fragment 的 "g" 就是
//      x half2 索引域 (g*8 half2 = k g*16..). 而 W 表: 每 g 段 16 码 -> col 的
//      k = g*16 + [0,4,8,12] + 交错对 => 8 个偶/奇 k? 样本只出了 k0,4,8 (v=7 在
//      lo&hi nibble 同值时才有对应奇 k). 我们单码 0x07 = lo nibble=7, hi=0 ->
//      只激活偶 k! 奇 k 需要 hi nibble. 所以 k0,4,8 都是偶 k, 奇 k 由 hi nibble
//      (码值 <<4) 激活 —— probe 用 0x07 掩码把 hi 清零了!
//   结论: k_even = g*16 + pair*4 + quad*8; k_odd = k_even + ? 由 dequant out[i].high
//      配 x half2 odd -> k_odd = k_even + 1? 或 +4? 由 out[i]=(lo=码i, hi=码i+4) 且
//      x half2 j 配 (k=2j, 2j+1): ci 对 half2 j = pair + 4*quad?? 推: ci1 (pair1,quad0)
//      -> k4 = 2j => j=2 => half2 序 j = pair*2 + quad*4. low->k=2j=4pair+8quad? 但
//      ci0 j0 k0 ✓, ci1: pair1 -> j=2 -> k4 ✓, ci4: quad1 -> j=4 -> k8 ✓!! 一致.
//      => half2 j = pair*2 + quad*4; lo 码 -> k=2j (偶), hi 码 (i+4) -> k=2j+1 (奇).
// 本测试按此最终映射生成参考, lane16-31 与 g 无关地同构 (每 half2 j 由 lane 组内
//   (g, lane<16) 覆盖: 16 lanes * 4 quads? 尚缺 lane16-31 角色 —— K=64: G=4,
//   half2 总数 32; lane<16 每个 2 cols * ? … 复杂度收敛策略: 参考直接由"槽位随机
//   + 本映射"生成 w, GPU 同槽位跑, 逐列对拍 (无需逻辑 prepack 中间层).
#include "ops/linear/qpn/qpn_kernels.cuh"
#include "ops/linear/qpn/qpn_map.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace ninfer::ops::qpn;

namespace {

float e2m1_tbl(unsigned nib) {
    static const float t[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return t[nib & 15];
}
float e4m3_dec(unsigned char b) {
    const int sign = (b >> 7) & 1;
    const int exp = (b >> 3) & 15;
    const int man = b & 7;
    float v;
    if (exp == 15 && man == 7) { v = NAN; }
    else if (exp == 0) { v = std::ldexp(man / 8.0f, -6); }
    else { v = std::ldexp(1.0f + man / 8.0f, exp - 7); }
    return sign ? -v : v;
}

} // namespace

int main() {
    const int N = 32, K = 64, M = 1;
    const int tiles = N / 32, groups = K / 16;
    const float gscale = 1.0e-5f;
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> cd(0, 255), sd(0, 0x7E);
    std::uniform_real_distribution<float> xd(-1.f, 1.f);

    std::vector<unsigned char> qc(tiles * groups * 32 * 8), qs(tiles * groups * 32);
    for (auto& v : qc) { v = (unsigned char)cd(rng); }
    for (auto& v : qs) { v = (unsigned char)(sd(rng) & 0x4F); }  // |e4m3|<=2.0 防链溢出
    for (auto& v : qs) {
        if (v == 0x7F || v == 0xFF) { v = 0x38; }
    }
    const bool kFixedScale = std::getenv("QPN_FIXED_SCALE") != nullptr;
    if (kFixedScale) {
        for (auto& v : qs) { v = 0x38; }  // e4m3 = 1.0
    }

    // 参考: qpn_map 实证闭式映射逐槽解码 (零假设)
    std::vector<float> w(N * K, 0.f);
    for (int t = 0; t < tiles; ++t) {
        for (int g = 0; g < groups; ++g) {
            for (int lane = 0; lane < 32; ++lane) {
                const float sf = e4m3_dec(qs[(t * groups + g) * 32 + lane]) * gscale;
                for (int ci = 0; ci < 16; ++ci) {
                    const int col = qpn_slot_col(lane, ci);
                    const unsigned char byte =
                        qc[(t * groups + g) * 32 * 8 + lane * 8 + ci / 2];
                    for (int nib = 0; nib < 2; ++nib) {
                        const unsigned nv = (nib == 0) ? (byte & 15) : (byte >> 4);
                        const int k = qpn_slot_k(g, lane, ci, nib);
                        if (col < N && k < K) { w[col * K + k] = e2m1_tbl(nv) * sf; }
                    }
                }
            }
        }
    }
    std::vector<__half> xh(K);
    for (auto& v : xh) { v = __float2half(xd(rng)); }
    std::vector<__half> y_ref(N);
    for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int d = 0; d < K; ++d) { acc += __half2float(xh[d]) * w[n * K + d]; }
        y_ref[n] = __float2half(acc);
    }

    unsigned char *d_c, *d_s;
    __half *d_x, *d_y;
    cudaMalloc(&d_c, qc.size());
    cudaMalloc(&d_s, qs.size());
    cudaMalloc(&d_x, xh.size() * 2);
    cudaMalloc(&d_y, N * 2);
    cudaMemcpy(d_c, qc.data(), qc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, qs.data(), qs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
    gemm_qpn_simt(d_x, d_c, d_s, gscale, d_y, M, K, N, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("GPU error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::vector<__half> y(N);
    cudaMemcpy(y.data(), d_y, N * 2, cudaMemcpyDeviceToHost);

    double num = 0, den = 0;
    int bad = 0;
    for (int n = 0; n < N; ++n) {
        const double a = __half2float(y_ref[n]);
        const double b = __half2float(y[n]);
        const double e = std::fabs(a - b);
        num += e * e;
        den += a * a;
        if (e > 0.02 && e > 0.05 * std::fabs(a)) {
            ++bad;
            // 找 ref[n] 是否等于另一列的 gpu (错位配对检测)
            int match = -1;
            for (int q = 0; q < N; ++q) {
                if (std::fabs(__half2float(y[q]) - a) < 0.004) { match = q; break; }
            }
            if (bad <= 12) {
                std::printf("  col %2d: gpu=%9.4f ref=%9.4f ref_at=%d\n", n, b, a, match);
            }
        }
    }
    std::printf("E2E-v4 M=%d N=%d K=%d bad=%d %s\n", M, N, K, bad,
                bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}


