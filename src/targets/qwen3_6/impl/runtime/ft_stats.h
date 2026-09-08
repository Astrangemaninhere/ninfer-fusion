#pragma once

// FreeToken step-1: per-layer attention energy observation (read-only).
// Sampled from the small-T decode reducer's partial_l (log-sum-exp) outputs.
// NINFER_FT_STATS=1 enables periodic stderr reports; NINFER_FT_PERIOD sets
// the report cadence in decode rounds (default 64).

#include <cstddef>
#include <cstdint>

#include <cstdio>
#include <cstdlib>

namespace ninfer::targets::qwen3_6::detail::ft {

inline constexpr std::size_t kMaxLayers = 64;

struct Stats {
    std::uint64_t rounds[kMaxLayers] = {};
    double energy_sum[kMaxLayers] = {};   // sum of exp(mean l) proxies
    std::uint64_t count[kMaxLayers] = {};
};

inline Stats& stats() {
    static Stats s;
    return s;
}

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

// Called post-reduce for a full decode round on `layer`.
inline void observe(int layer, const float* partial_l, int n_values) {
    if (!enabled() || layer < 0 || layer >= static_cast<int>(kMaxLayers)) { return; }
    Stats& s = stats();
    double sum = 0.0;
    for (int i = 0; i < n_values; ++i) { sum += partial_l[i]; }
    s.energy_sum[layer] += sum;
    s.count[layer] += static_cast<std::uint64_t>(n_values);
    s.rounds[layer] += 1;
    if (s.rounds[layer] % static_cast<std::uint64_t>(period()) == 0) {
        std::fprintf(stderr, "[ft] layer=%d energy=%.4g samples=%llu\n", layer,
                     s.energy_sum[layer] / static_cast<double>(s.count[layer]),
                     static_cast<unsigned long long>(s.count[layer]));
    }
}

} // namespace ninfer::targets::qwen3_6::detail::ft
