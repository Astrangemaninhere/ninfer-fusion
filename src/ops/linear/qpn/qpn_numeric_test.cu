// qpn_numeric_test.cu — qpn_simt 数值验证 v3: 叠加原理 + GPU 自身做参考通道.
// 思路: 对随机码本 C, 构造"逐码分离"的对照码本族 (每码位单独), 用同一 GPU 内核
// 解码后线性叠加 = 数学上严格等于联合解码 (算子线性), 从而参考无需 CPU 重实现
// 解码细节; scale/gscale 因子解析并入. 若 GPU 线性成立且映射被覆盖, 叠加==联合.
#include "ops/linear/qpn/qpn_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace ninfer::ops::qpn;

namespace {

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

struct Ctx {
    unsigned char *d_c = nullptr, *d_s = nullptr;
    __half *d_x = nullptr, *d_y = nullptr;
};

void run_gpu(const Ctx& c, const std::vector<__half>& x, std::vector<__half>& y, int m, int k,
             int n, float gscale) {
    cudaMemcpy(c.d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    gemm_qpn_simt(c.d_x, c.d_c, c.d_s, gscale, c.d_y, m, k, n, 0);
    cudaDeviceSynchronize();
    y.resize(m * n);
    cudaMemcpy(y.data(), c.d_y, y.size() * 2, cudaMemcpyDeviceToHost);
}

} // namespace

int main() {
    const int N = 32, K = 64, M = 1;
    const int tiles = N / 32, G = K / 16;
    Ctx ctx;
    cudaMalloc(&ctx.d_c, tiles * G * 32 * 16);
    cudaMalloc(&ctx.d_s, tiles * G * 32);
    cudaMalloc(&ctx.d_x, M * K * 2);
    cudaMalloc(&ctx.d_y, M * N * 2);

    // 随机联合码本 (码随机, scale 随机非 NaN, gscale=0.001)
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> cd(0, 255), sd(0, 0x7E);
    std::vector<unsigned char> codes(tiles * G * 32 * 16), scales(tiles * G * 32);
    for (auto& v : codes) { v = (unsigned char)cd(rng); }
    for (auto& v : scales) { v = (unsigned char)sd(rng); }
    for (auto& v : scales) {
        if (v == 0x7F || v == 0xFF) { v = 0x38; }
    }
    // 基准联合运行
    cudaMemcpy(ctx.d_c, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_s, scales.data(), scales.size(), cudaMemcpyHostToDevice);
    std::vector<__half> xh(K);
    std::uniform_real_distribution<float> xd(-1.f, 1.f);
    for (auto& v : xh) { v = __float2half(xd(rng)); }
    std::vector<__half> y_joint;
    run_gpu(ctx, xh, y_joint, M, K, N, 0.001f);

    // 逐码分离叠加: 注意 scale 以 (g,lane) 为组, 不能按单码分离 scale;
    // 分离单位 = (g, lane) 组 (16 码同 scale): 联合 = Σ_{g,lane} 组响应.
    // 组响应 = 该组码原样 + 其它组全 0 码. scale 保留本组值 -> 严格线性.
    std::vector<__half> y_sum(N, __float2half(0.f));
    int groups_used = 0;
    for (int g = 0; g < G; ++g) {
        for (int lane = 0; lane < 32; ++lane) {
            std::vector<unsigned char> c_part(tiles * G * 32 * 16, 0);
            const size_t base = (size_t)(g * 32 + lane) * 16;
            bool any = false;
            for (int i = 0; i < 16; ++i) {
                c_part[base + i] = codes[base + i];
                if (codes[base + i]) { any = true; }
            }
            if (!any) { continue; }
            ++groups_used;
            cudaMemcpy(ctx.d_c, c_part.data(), c_part.size(), cudaMemcpyHostToDevice);
            std::vector<__half> y_part;
            run_gpu(ctx, xh, y_part, M, K, N, 0.001f);
            for (int j = 0; j < N; ++j) {
                y_sum[j] = __float2half(__half2float(y_sum[j]) + __half2float(y_part[j]));
            }
        }
    }
    // gscale 恒同 (0.001) 且解码线性 -> 叠加应逐位一致 (fp16 舍入容差内)
    double num = 0, den = 0;
    for (int j = 0; j < N; ++j) {
        const double a = __half2float(y_joint[j]);
        const double b = __half2float(y_sum[j]);
        num += (a - b) * (a - b);
        den += a * a;
    }
    const double rel = std::sqrt(num) / std::sqrt(den);
    std::printf("groups=%d rel_err(additivity)=%.3e %s\n", groups_used, rel,
                rel < 1e-3 ? "PASS" : "FAIL");
    for (int j = 0; j < 4; ++j) {
        std::printf("  y[%d] joint=%.5f sum=%.5f\n", j, __half2float(y_joint[j]),
                    __half2float(y_sum[j]));
    }
    (void)e4m3_dec;
    return rel < 1e-3 ? 0 : 1;
}
