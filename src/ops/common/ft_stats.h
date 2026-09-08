#pragma once

// FreeToken step-1: per-layer attention energy observation (read-only).
// observe() copies a small sample of partial_l (device) to host — legal only
// outside CUDA graph capture. Observation runs use --no-cuda-graph +
// NINFER_FT_STATS=1; disabled by default (zero cost).

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>

namespace ninfer::ops::ft {

inline constexpr int kMaxLayers = 64;
inline constexpr int kSampleHeads = 32;

inline bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("NINFER_FT_STATS");
        return e && e[0] == '1';
    }();
    return on;
}

inline int period() {
    static const int p = [] {
        const char* e = std::getenv("NINFER_FT_PERIOD");
        return e ? atoi(e) : 64;
    }();
    return p > 0 ? p : 64;
}

inline void observe(cudaStream_t stream, int layer, const float* partial_l_device,
                    int q_heads, int splits) {
    if (!enabled() || layer < 0 || layer >= kMaxLayers || partial_l_device == nullptr) {
        return;
    }
    // 采样每 head 的第一个 split 值 (log-sum-exp 代理), D2H 小拷贝合法化
    float host[kSampleHeads] = {};
    const int n = q_heads < kSampleHeads ? q_heads : kSampleHeads;
    cudaMemcpyAsync(host, partial_l_device, sizeof(float) * n,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    static double sum[kMaxLayers] = {};
    static std::uint64_t cnt[kMaxLayers] = {};
    double acc = 0.0;
    for (int i = 0; i < n; ++i) { acc += host[i]; }
    sum[layer] += acc / n;
    cnt[layer] += 1;
    if (cnt[layer] % static_cast<std::uint64_t>(period()) == 0) {
        std::fprintf(stderr, "[ft] layer=%d mean_l=%.4g rounds=%llu\n", layer,
                     sum[layer] / static_cast<double>(cnt[layer]),
                     static_cast<unsigned long long>(cnt[layer]));
    }
}

} // namespace ninfer::ops::ft
