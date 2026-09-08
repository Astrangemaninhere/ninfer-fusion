// qpn_probe_map.cu — 探测 qpn_simt 的码位→(列,k) 映射与数值比例.
// 方法: 每 (byte_idx, nibble_idx) 单码非零 (值=码7=6.0), 其余 0; scale=0x38, gscale=1;
// x = 单位脉冲逐 k. 输出 (lane, out_col, k_response) 完全映射表 (首 64 码位).
#include "ops/linear/qpn/qpn_kernels.cuh"

#include <cstdio>
#include <vector>

using namespace ninfer::ops::qpn;

int main() {
    const int N = 32, K = 64, M = 1;
    const int tiles = 1, G = K / 16;
    unsigned char *d_c, *d_s;
    __half *d_x, *d_y;
    cudaMalloc(&d_c, tiles * G * 32 * 16);
    cudaMalloc(&d_s, tiles * G * 32);
    cudaMalloc(&d_x, M * K * 2);
    cudaMalloc(&d_y, M * N * 2);

    auto run1 = [&](int lane, int byte_i, int nib_i, int k_pulse, float* out_col) {
        std::vector<unsigned char> codes(tiles * G * 32 * 16, 0), scales(tiles * G * 32, 0x38);
        // 值 7 (e2m1=6): nibble 位 = byte_i*8 + nib_i*4 (低 nibble=偶码)
        codes[lane * 16 + byte_i] = (nib_i == 0) ? 0x07 : 0x70;
        cudaMemcpy(d_c, codes.data(), codes.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_s, scales.data(), scales.size(), cudaMemcpyHostToDevice);
        std::vector<__half> x(K);
        for (int i = 0; i < K; ++i) { x[i] = __float2half(i == k_pulse ? 1.f : 0.f); }
        cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        gemm_qpn_simt(d_x, d_c, d_s, 1.0f, d_y, M, K, N, 0);
        cudaDeviceSynchronize();
        std::vector<__half> y(N);
        cudaMemcpy(y.data(), d_y, N * 2, cudaMemcpyDeviceToHost);
        for (int j = 0; j < N; ++j) { out_col[j] = __half2float(y[j]); }
    };

    // 探测 1: lane0 byte0 低 nibble (码0), 脉冲 k 扫 0..15 -> 找响应 k 与响应列
    float col[32];
    std::printf("probe: lane0 code0 (byte0 lo) value=6.0\n");
    for (int kp = 0; kp < 16; ++kp) {
        run1(0, 0, 0, kp, col);
        for (int j = 0; j < 32; ++j) {
            if (col[j] != 0.f) { std::printf("  k=%d -> col%d w=%.3f\n", kp, j, col[j]); }
        }
    }
    std::printf("probe: lane0 code1 (byte0 hi)\n");
    for (int kp = 0; kp < 16; ++kp) {
        run1(0, 0, 1, kp, col);
        for (int j = 0; j < 32; ++j) {
            if (col[j] != 0.f) { std::printf("  k=%d -> col%d w=%.3f\n", kp, j, col[j]); }
        }
    }
    std::printf("probe: lane0 code4 (byte2 lo)\n");
    for (int kp = 0; kp < 16; ++kp) {
        run1(0, 2, 0, kp, col);
        for (int j = 0; j < 32; ++j) {
            if (col[j] != 0.f) { std::printf("  k=%d -> col%d w=%.3f\n", kp, j, col[j]); }
        }
    }
    std::printf("probe: lane1 code0 (byte0 lo)\n");
    for (int kp = 0; kp < 16; ++kp) {
        run1(1, 0, 0, kp, col);
        for (int j = 0; j < 32; ++j) {
            if (col[j] != 0.f) { std::printf("  k=%d -> col%d w=%.3f\n", kp, j, col[j]); }
        }
    }
    return 0;
}
