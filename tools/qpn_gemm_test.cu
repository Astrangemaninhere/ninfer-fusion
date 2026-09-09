// tools/qpn_gemm_test.cu — gemm_qpn M-band sweep acceptance (W4 step 1/4).
//
// The synthetic prepack layout + CPU reference decoder are the same as
// src/ops/linear/qpn/qpn_mma_e2e_test.cu (qpn2 slot map, bit-exact with the
// marlin _qpn_prepack prototype). This test drives the *host dispatch* added
// for W4: gemm_qpn() must route M 4..8 to skinny_nvfp4_qpn<1>, M 9..16 to
// skinny_nvfp4_qpn<2>, and M<=3 to the simt band.
//
// Build (WSL, standalone):
//   nvcc -O2 -std=c++20 -gencode arch=compute_120a,code=[compute_120a,sm_120a] \
//     -I src -I include tools/qpn_gemm_test.cu src/ops/linear/qpn/qpn_host.cu \
//     -o qpn_gemm_test
// Note: this TU deliberately does NOT include qpn_kernels.cuh — that header
// defines non-inline __global__ helpers (skinny_quant_a8), so including it in a
// second TU duplicates symbols at link time. Only the host entry is declared.
#include "ops/linear/qpn/qpn_map.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace ninfer::ops::qpn {
void gemm_qpn(const void* x_half, const void* codes, const void* scales, float gscale,
              void* y_half, int m, int k, int n, cudaStream_t stream);
}

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
    const int N = 32, K = 64, M_MAX = 16;
    const int tiles = N / 32, groups = K / 16;
    const float gscale = 1.0e-5f;
    std::mt19937 rng(17);
    std::uniform_int_distribution<int> cd(0, 255), sd(0, 0x4F);
    std::uniform_real_distribution<float> xd(-1.f, 1.f);

    std::vector<unsigned char> qc(tiles * groups * 32 * 8), qs(tiles * groups * 32);
    for (auto& v : qc) { v = static_cast<unsigned char>(cd(rng)); }
    for (auto& v : qs) { v = static_cast<unsigned char>(sd(rng)); }
    for (auto& v : qs) {
        if (v == 0x7F || v == 0xFF) { v = 0x38; }
    }

    // Reference weights via the qpn2 slot map.
    std::vector<float> w(N * K, 0.f);
    for (int t = 0; t < tiles; ++t) {
        for (int g = 0; g < groups; ++g) {
            for (int lane = 0; lane < 32; ++lane) {
                const float sf = e4m3_dec(qs[(t * groups + g) * 32 + lane]) * gscale;
                for (int ci = 0; ci < 16; ++ci) {
                    const int col = qpn2_slot_col(lane, ci);
                    const unsigned char byte = qc[(t * groups + g) * 32 * 8 + lane * 8 + ci / 2];
                    for (int nib = 0; nib < 2; ++nib) {
                        const unsigned nv = (nib == 0) ? (byte & 15) : (byte >> 4);
                        const int k = qpn2_slot_k(g, lane, ci, nib);
                        if (col < N && k < K) { w[col * K + k] = e2m1_tbl(nv) * sf; }
                    }
                }
            }
        }
    }

    std::vector<__half> xh(M_MAX * K);
    for (auto& v : xh) { v = __float2half(xd(rng)); }

    unsigned char *d_c, *d_s;
    __half *d_x, *d_y;
    cudaMalloc(&d_c, qc.size());
    cudaMalloc(&d_s, qs.size());
    cudaMalloc(&d_x, xh.size() * 2);
    cudaMalloc(&d_y, M_MAX * N * 2);
    cudaMemcpy(d_c, qc.data(), qc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, qs.data(), qs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);

    int failures = 0;
    for (int m = 4; m <= 16; ++m) {
        cudaMemset(d_y, 0, M_MAX * N * 2);
        gemm_qpn(d_x, d_c, d_s, gscale, d_y, m, K, N, nullptr);
        const cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::printf("M=%2d GPU error: %s\n", m, cudaGetErrorString(err));
            ++failures;
            continue;
        }
        std::vector<__half> y(M_MAX * N);
        cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int row = 0; row < m; ++row) {
            for (int n = 0; n < N; ++n) {
                float ref = 0.f;
                for (int d = 0; d < K; ++d) {
                    ref += __half2float(xh[row * K + d]) * w[n * K + d];
                }
                const double a = ref;
                const double b = __half2float(y[row * N + n]);
                const double e = std::fabs(a - b);
                if (e > 0.02 && e > 0.05 * std::fabs(a)) {
                    ++bad;
                    if (bad <= 3) { std::printf("  m%d n%2d gpu=%9.4f ref=%9.4f\n", row, n, b, a); }
                }
            }
        }
        std::printf("M=%2d bad=%d %s\n", m, bad, bad == 0 ? "PASS" : "FAIL");
        if (bad != 0) { ++failures; }
    }
    std::printf("%s\n", failures == 0 ? "QPN_GEMM_TEST PASS" : "QPN_GEMM_TEST FAIL");
    return failures == 0 ? 0 : 1;
}
