// qpn_probe_qpn2.cu — qpn2 穷举标定 (同 qpn_probe_full 法): (g,lane,ci,nib)->(row,col,k).
// qpn2 输出 y[m][tile*32+e&31]; 每 lane 的码服务哪个 (m 行受 M 限制, e 映射) 单码探测.
#include "ops/linear/qpn/qpn_kernels.cuh"

#include <cstdio>
#include <vector>

using namespace ninfer::ops::qpn;

int main() {
    const int N = 32, K = 64, M = 8;
    const int G = K / 16;
    unsigned char *d_c, *d_s;
    __half *d_x, *d_y;
    cudaMalloc(&d_c, G * 32 * 16);
    cudaMalloc(&d_s, G * 32);
    cudaMalloc(&d_x, M * K * 2);
    cudaMalloc(&d_y, M * N * 2);

    std::printf("g,lane,ci,nib,row,col,k,val\n");
    for (int g = 0; g < G; ++g) {
        for (int lane = 0; lane < 32; ++lane) {
            for (int ci = 0; ci < 16; ++ci) {
                for (int nib = 0; nib < 2; ++nib) {
                    std::vector<unsigned char> codes(G * 32 * 16, 0), scales(G * 32, 0x38);
                    codes[(g * 32 + lane) * 16 + ci] =
                        (unsigned char)(nib == 0 ? 0x07 : 0x70);
                    cudaMemcpy(d_c, codes.data(), codes.size(), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_s, scales.data(), scales.size(), cudaMemcpyHostToDevice);
                    // x 全 1 (K 个) 简化: 每 k 贡献同权; 但需要区分 k —— 用脉冲组:
                    // 改为 x[k]=k+1 (斜坡), 单码值 6.0 -> y[m] = 6*(kp+1) 反解 kp.
                    std::vector<__half> x(M * K);
                    for (int m = 0; m < M; ++m) {
                        for (int i = 0; i < K; ++i) {
                            x[m * K + i] = __float2half(float(i + 1));
                        }
                    }
                    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
                    gemm_qpn_simt(d_x, d_c, d_s, 1.0f, d_y, 1, K, N, 0);
                    cudaDeviceSynchronize();
                    std::vector<__half> y(N);
                    cudaMemcpy(y.data(), d_y, N * 2, cudaMemcpyDeviceToHost);
                    for (int j = 0; j < N; ++j) {
                        const float v = __half2float(y[j]);
                        if (std::fabs(v) > 1e-3f) {
                            // 反解 k: 6*(kp+1)=v -> kp = v/6 - 1
                            const int kp = int(v / 6.0f - 1.0f + 0.5f);
                            std::printf("%d,%d,%d,%d,%d,%d,%d,%.1f\n", g, lane, ci, nib, -1,
                                        j, kp, v);
                        }
                    }
                }
            }
        }
    }
    return 0;
}
