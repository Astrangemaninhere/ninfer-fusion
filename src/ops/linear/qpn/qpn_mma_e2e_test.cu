// qpn_mma_e2e_test.cu — qpn (mma.m8n8k4, M<=16, MT=2) 端到端对拍.
// 假设实证: qpn 与 qpn2 槽位寻址相同 (cb/sb 同式) => 映射沿用 qpn2_slot_*.
// 参考解码用 qpn_map.cuh; 若 PASS 则映射继承实证成立, 若 FAIL 转穷举标定.
#include "ops/linear/qpn/qpn_kernels.cuh"
#include "ops/linear/qpn/qpn_map.cuh"

#include <cmath>
#include <cstdio>
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
    const int N = 32, K = 64, M = 16;
    const int tiles = N / 32, groups = K / 16;
    const float gscale = 1.0e-5f;
    std::mt19937 rng(17);
    std::uniform_int_distribution<int> cd(0, 255), sd(0, 0x4F);
    std::uniform_real_distribution<float> xd(-1.f, 1.f);

    std::vector<unsigned char> qc(tiles * groups * 32 * 8), qs(tiles * groups * 32);
    for (auto& v : qc) { v = (unsigned char)cd(rng); }
    for (auto& v : qs) { v = (unsigned char)sd(rng); }
    for (auto& v : qs) {
        if (v == 0x7F || v == 0xFF) { v = 0x38; }
    }

    // 参考: qpn2 同槽位映射解码
    std::vector<float> w(N * K, 0.f);
    for (int t = 0; t < tiles; ++t) {
        for (int g = 0; g < groups; ++g) {
            for (int lane = 0; lane < 32; ++lane) {
                const float sf = e4m3_dec(qs[(t * groups + g) * 32 + lane]) * gscale;
                for (int ci = 0; ci < 16; ++ci) {
                    const int col = qpn2_slot_col(lane, ci);
                    const unsigned char byte =
                        qc[(t * groups + g) * 32 * 8 + lane * 8 + ci / 2];
                    for (int nib = 0; nib < 2; ++nib) {
                        const unsigned nv = (nib == 0) ? (byte & 15) : (byte >> 4);
                        const int k = qpn2_slot_k(g, lane, ci, nib);
                        if (col < N && k < K) { w[col * K + k] = e2m1_tbl(nv) * sf; }
                    }
                }
            }
        }
    }
    std::vector<__half> xh(M * K);
    for (auto& v : xh) { v = __float2half(xd(rng)); }
    std::vector<__half> y_ref(M * N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int d = 0; d < K; ++d) {
                acc += __half2float(xh[m * K + d]) * w[n * K + d];
            }
            y_ref[m * N + n] = __float2half(acc);
        }
    }

    unsigned char *d_c, *d_s;
    __half *d_x, *d_y;
    cudaMalloc(&d_c, qc.size());
    cudaMalloc(&d_s, qs.size());
    cudaMalloc(&d_x, xh.size() * 2);
    cudaMalloc(&d_y, M * N * 2);
    cudaMemcpy(d_c, qc.data(), qc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, qs.data(), qs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
    // qpn mma MT=2 (M 9..16): grid n/32, 128 threads (4 warps), 无动态 smem
    skinny_nvfp4_qpn<2><<<dim3(N / 32), dim3(128), 0, 0>>>(
        d_c, d_s, d_x, d_y, N, K, M, gscale);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("GPU error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::vector<__half> y(M * N);
    cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const double a = __half2float(y_ref[m * N + n]);
            const double b = __half2float(y[m * N + n]);
            const double e = std::fabs(a - b);
            if (e > 0.02 && e > 0.05 * std::fabs(a)) {
                ++bad;
                if (bad <= 8) {
                    std::printf("  m%d col %2d: gpu=%9.4f ref=%9.4f\n", m, n, b, a);
                }
            }
        }
    }
    std::printf("QPN-MMA-E2E M=%d N=%d K=%d bad=%d %s\n", M, N, K, bad,
                bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
