// qpn_probe2.cu — 探测 lane 4..31 的列归属 (每 lane 单码 v=7, k=0 脉冲).
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

    for (int lane = 0; lane < 32; ++lane) {
        std::vector<unsigned char> codes(G * 32 * 16, 0), scales(G * 32, 0x38);
        codes[lane * 16] = 0x07;  // 码0 = 6.0
        cudaMemcpy(d_c, codes.data(), codes.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_s, scales.data(), scales.size(), cudaMemcpyHostToDevice);
        std::vector<__half> x(K);
        for (int i = 0; i < K; ++i) { x[i] = __float2half(i == 0 ? 1.f : 0.f); }
        cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        gemm_qpn_simt(d_x, d_c, d_s, 1.0f, d_y, 1, K, N, 0);
        cudaDeviceSynchronize();
        std::vector<__half> y(N);
        cudaMemcpy(y.data(), d_y, N * 2, cudaMemcpyDeviceToHost);
        for (int j = 0; j < N; ++j) {
            if (std::fabs(__half2float(y[j])) > 1e-3f) {
                std::printf("lane %2d -> col %2d (w=%.1f)\n", lane, j, __half2float(y[j]));
            }
        }
    }
    return 0;
}
