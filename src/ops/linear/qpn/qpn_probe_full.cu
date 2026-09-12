// qpn_probe_full.cu — 穷举标定: 每 (slot=lane, g, ci, nib) 单码激活 + x 脉冲扫描
// => 机器可读完整映射表 (col, k, 系数). 输出 CSV 供 e2e 参考直接消费.
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

    std::printf("g,lane,ci,nib,col,k,val\n");
    for (int g = 0; g < G; ++g) {
        for (int lane = 0; lane < 32; ++lane) {
            for (int ci = 0; ci < 16; ++ci) {
                for (int nib = 0; nib < 2; ++nib) {
                    std::vector<unsigned char> codes(G * 32 * 16, 0), scales(G * 32, 0x38);
                    codes[(g * 32 + lane) * 16 + ci] =
                        (unsigned char)(nib == 0 ? 0x07 : 0x70);
                    cudaMemcpy(d_c, codes.data(), codes.size(), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_s, scales.data(), scales.size(), cudaMemcpyHostToDevice);
                    for (int kp = 0; kp < K; ++kp) {
                        std::vector<__half> x(K);
                        for (int i = 0; i < K; ++i) {
                            x[i] = __float2half(i == kp ? 1.f : 0.f);
                        }
                        cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
                        gemm_qpn_simt(d_x, d_c, d_s, 1.0f, d_y, 1, K, N, 0);
                        cudaDeviceSynchronize();
                        std::vector<__half> y(N);
                        cudaMemcpy(y.data(), d_y, N * 2, cudaMemcpyDeviceToHost);
                        for (int j = 0; j < N; ++j) {
                            if (std::fabs(__half2float(y[j])) > 1e-3f) {
                                std::printf("%d,%d,%d,%d,%d,%d,%.1f\n", g, lane, ci, nib, j,
                                            kp, __half2float(y[j]));
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
