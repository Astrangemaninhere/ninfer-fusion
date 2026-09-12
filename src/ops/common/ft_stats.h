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

// Per-layer energy sample (FreeToken step 2 consumers read this in-process
// instead of parsing the stderr lines).
struct EnergySample {
    int layer = 0;
    double mean_l = 0.0;
    std::uint64_t rounds = 0;
};

namespace detail {
inline double g_sum[kMaxLayers] = {};
inline std::uint64_t g_count[kMaxLayers] = {};
} // namespace detail

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
    double acc = 0.0;
    for (int i = 0; i < n; ++i) { acc += host[i]; }
    detail::g_sum[layer] += acc / n;
    detail::g_count[layer] += 1;
    if (detail::g_count[layer] % static_cast<std::uint64_t>(period()) == 0) {
        std::fprintf(stderr, "[ft] layer=%d mean_l=%.4g rounds=%llu\n", layer,
                     detail::g_sum[layer] / static_cast<double>(detail::g_count[layer]),
                     static_cast<unsigned long long>(detail::g_count[layer]));
    }
}

// Snapshot of the observed layers (ascending layer order); returns the number
// of entries written. Layers with no observation are skipped.
inline int snapshot(EnergySample* out, int max_out) {
    if (out == nullptr || max_out <= 0) { return 0; }
    int count = 0;
    for (int layer = 0; layer < kMaxLayers && count < max_out; ++layer) {
        if (detail::g_count[layer] == 0) { continue; }
        out[count].layer  = layer;
        out[count].mean_l = detail::g_sum[layer] / static_cast<double>(detail::g_count[layer]);
        out[count].rounds = detail::g_count[layer];
        ++count;
    }
    return count;
}

} // namespace ninfer::ops::ft
