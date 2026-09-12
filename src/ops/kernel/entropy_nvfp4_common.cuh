#pragma once

// Shared static rANS constants and device helpers for the NVFP4 E2M1 codecs.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kEntropyNvfp4ScaleBits = 12;
inline constexpr std::uint32_t kEntropyNvfp4Scale = 4096u;
inline constexpr std::uint32_t kEntropyNvfp4Mask = 4095u;
inline constexpr std::uint32_t kEntropyNvfp4RansByteL = 1u << 23;

// Mirrors tools/calib/rans_nvfp4.py: nearest-even rounding of
// count/total*4096 with a minimum of one, then the residual is distributed to
// the current largest count (counts are decremented after each grant so a
// dominant symbol can receive several grants) or taken from the largest
// frequency above one. This keeps the CPU feasibility gate byte-compatible
// with the device codec.
__device__ inline void entropy_nvfp4_normalize_freqs(const int hist[16], int total,
                                                     std::uint16_t fs[16]) {
    if (total <= 0) {
        for (int s = 0; s < 16; ++s) { fs[s] = 256; }
        return;
    }

    int counts[16];
    int sum = 0;
    for (int s = 0; s < 16; ++s) {
        counts[s] = hist[s];
        const double q =
            static_cast<double>(hist[s]) * static_cast<double>(kEntropyNvfp4Scale) /
            static_cast<double>(total);
        std::uint32_t f = static_cast<std::uint32_t>(__double2ll_rn(q));
        if (f < 1) { f = 1; }
        fs[s] = static_cast<std::uint16_t>(f);
        sum += static_cast<int>(f);
    }

    int diff = static_cast<int>(kEntropyNvfp4Scale) - sum;
    while (diff > 0) {
        int best = 0;
        for (int s = 1; s < 16; ++s) {
            if (counts[s] > counts[best]) { best = s; }
        }
        fs[best] = static_cast<std::uint16_t>(fs[best] + 1);
        counts[best] = counts[best] > 0 ? counts[best] - 1 : 0;
        --diff;
    }
    while (diff < 0) {
        int best = 0;
        for (int s = 1; s < 16; ++s) {
            const bool s_ok    = fs[s] > 1;
            const bool best_ok = fs[best] > 1;
            if ((s_ok && !best_ok) || (s_ok && best_ok && fs[s] > fs[best])) { best = s; }
        }
        fs[best] = static_cast<std::uint16_t>(fs[best] - 1);
        ++diff;
    }
}

__device__ inline void entropy_nvfp4_make_start(const std::uint16_t fs[16], std::uint32_t start[16]) {
    std::uint32_t acc = 0;
    for (int s = 0; s < 16; ++s) {
        start[s] = acc;
        acc += fs[s];
    }
}

} // namespace ninfer::ops::detail
