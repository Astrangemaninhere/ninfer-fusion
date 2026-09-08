// qpn_probe6.cu — hi nibble 激活 (0x70), 探测其 k 归属.
#include "ops/linear/qpn/qpn_kernels.cuh"

#include <cstdio>
#include <vector>

using namespace ninfer::ops::qpn;

int main() {
    const int N = 32, K = 64;
    const int G = K / 16;
    unsigned char *d_c, *d_s;
    __half *d_x, *d_y;
    cudaMalloc(&d_c, G * 32 * 16);
    cudaMalloc(&d_s, G * 32);
    cudaMalloc(&d_x, K * 2);
    cudaMalloc(&d_y, N * 2);

    for (int ci : {0, 1, 4, 8}) {
        std::vector<unsigned char> codes(G * 32 * 16, 0), scales(G * 32, 0x38);
        codes[ci] = 0x70;  // lane0, hi nibble = 7
        cudaMemcpy(d_c, codes.data(), codes.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_s, scales.data(), scales.size(), cudaMemcpyHostToDevice);
        for (int kp = 0; kp < 16; ++kp) {
            std::vector<__half> x(K);
            for (int i = 0; i < K; ++i) { x[i] = __float2half(i == kp ? 1.f : 0.f); }
            cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
            gemm_qpn_simt(d_x, d_c, d_s, 1.0f, d_y, 1, K, N, 0);
            cudaDeviceSynchronize();
            std::vector<__half> y(N);
            cudaMemcpy(y.data(), d_y, N * 2, cudaMemcpyDeviceToHost);
            for (int j = 0; j < N; ++j) {
                if (std::fabs(__half2float(y[j])) > 1e-3f) {
                    std::printf("lane0 ci%2d HI -> col %2d k %2d (w=%.1f)\n", ci, j, kp,
                                __half2float(y[j]));
                }
            }
        }
    }
    return 0;
}
