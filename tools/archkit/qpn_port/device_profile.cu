// device_profile.cu — 设备特性画像 microbench v2 (PlacementPlanner 输入).
// int8 项重写: rows=65536 (充分占卡), 每线程一行, 独立累积.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {

__global__ void matmul_fp16_kernel(const __half* a, const __half* b, float* c, int n) {
    const int row = blockIdx.y * 16 + (threadIdx.x % 16);
    const int col = blockIdx.x * 16 + (threadIdx.x / 16);
    float acc = 0.f;
    for (int i = 0; i < n; ++i) {
        acc += __half2float(a[row * n + i]) * __half2float(b[i * n + col]);
    }
    c[row * n + col] = acc;
}

__global__ void read_bw_kernel(const float4* src, size_t n4, float* sink) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const float4 v = src[i % n4];
    if (v.x == 12345.678f) { *sink = v.x; }
}

__global__ void empty_kernel() {}

__global__ void int8_dot_kernel(const int8_t* a, const int8_t* b, int* out, int n) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int acc = 0;
    for (int d = 0; d < n; ++d) { acc += a[row * n + d] * b[d]; }
    out[row] = acc;
}

} // namespace

int main() {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    std::printf("{\"compute_tflops\":");

    // 1) fp16 compute: 512^3 x20
    {
        const int n = 512;
        __half *a, *b;
        float* c;
        cudaMalloc(&a, n * n * 2);
        cudaMalloc(&b, n * n * 2);
        cudaMalloc(&c, n * n * 4);
        cudaMemset(a, 0x3c, n * n * 2);
        cudaMemset(b, 0x3c, n * n * 2);
        dim3 grid(n / 16, n / 16), block(256);
        matmul_fp16_kernel<<<grid, block>>>(a, b, c, n);
        cudaEventRecord(t0);
        for (int i = 0; i < 20; ++i) { matmul_fp16_kernel<<<grid, block>>>(a, b, c, n); }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        std::printf("%.1f", 2.0 * n * n * n * 20.0 / (ms * 1e-3) / 1e12);
        cudaFree(a); cudaFree(b); cudaFree(c);
    }
    std::printf(",\"bw_gbs\":");
    // 2) bandwidth: 1GB read x10
    {
        const size_t bytes = 1ULL << 30;
        float* src; float* sink;
        cudaMalloc(&src, bytes);
        cudaMalloc(&sink, 4);
        cudaMemset(src, 1, bytes);
        const size_t n4 = bytes / 16;
        read_bw_kernel<<<static_cast<unsigned>(n4 / 256), 256>>>(
            reinterpret_cast<const float4*>(src), n4, sink);
        cudaEventRecord(t0);
        for (int i = 0; i < 10; ++i) {
            read_bw_kernel<<<static_cast<unsigned>(n4 / 256), 256>>>(
                reinterpret_cast<const float4*>(src), n4, sink);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        std::printf("%.0f", bytes * 10.0 / (ms * 1e-3) / 1e9);
        cudaFree(src); cudaFree(sink);
    }
    std::printf(",\"launch_us\":");
    // 3) launch latency
    {
        cudaEventRecord(t0);
        for (int i = 0; i < 512; ++i) { empty_kernel<<<1, 32>>>(); }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        std::printf("%.2f", ms / 512.0 * 1000.0);
    }
    std::printf(",\"int8_gops\":");
    // 4) int8: 65536 行 x 4096 dot x20
    {
        const int64_t rows = 65536;
        const int n = 4096;
        int8_t *a, *b;
        int* out;
        cudaMalloc(&a, rows * n);
        cudaMalloc(&b, n);
        cudaMalloc(&out, rows * 4);
        cudaMemset(a, 1, rows * n);
        cudaMemset(b, 1, n);
        const int block = 256;
        const int grid = static_cast<int>(rows / block);
        int8_dot_kernel<<<grid, block>>>(a, b, out, n);
        cudaEventRecord(t0);
        for (int i = 0; i < 20; ++i) { int8_dot_kernel<<<grid, block>>>(a, b, out, n); }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        std::printf("%.0f", 2.0 * rows * n * 20.0 / (ms * 1e-3) / 1e9);
        cudaFree(a); cudaFree(b); cudaFree(out);
    }
    std::printf("}\n");
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return 0;
}
