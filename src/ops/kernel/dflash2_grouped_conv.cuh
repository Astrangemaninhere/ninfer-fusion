#pragma once

// Implements: include/ninfer/ops/dflash2_grouped_conv.h
// Match: contiguous BF16 hidden/delta/base with the documented token-major layout.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

template <int Block>
__launch_bounds__(Block) __global__ void dflash2_grouped_conv_kernel(
    const __nv_bfloat16* __restrict__ hidden, const __nv_bfloat16* __restrict__ delta,
    const __nv_bfloat16* __restrict__ base, __nv_bfloat16* __restrict__ out, int hidden_size,
    int tokens, int taps, int groups, int group_size, int block_size, int side) {
    const int h = static_cast<int>(blockIdx.x) * Block + static_cast<int>(threadIdx.x);
    if (h >= hidden_size) { return; }
    const int g = h / group_size;
    for (int t = 0; t < tokens; ++t) {
        const std::int64_t hidden_offset = static_cast<std::int64_t>(h) +
                                           static_cast<std::int64_t>(hidden_size) * t;
        const std::int64_t delta_n =
            static_cast<std::int64_t>(groups) * (side * taps) + g;
        const std::int64_t delta_offset =
            delta_n + static_cast<std::int64_t>(2) * taps * groups * t;
        float value = __bfloat162float(hidden[hidden_offset]) *
                      (__bfloat162float(base[h]) + __bfloat162float(delta[delta_offset]));
        const int position = t & (block_size - 1);
        for (int tap = 1; tap < taps; ++tap) {
            if (position < tap) { continue; }
            const std::int64_t prev = static_cast<std::int64_t>(h) +
                                      static_cast<std::int64_t>(hidden_size) * (t - tap);
            const std::int64_t coefficient =
                delta_offset + static_cast<std::int64_t>(tap) * groups;
            value += __bfloat162float(hidden[prev]) *
                     (__bfloat162float(base[static_cast<std::int64_t>(tap) * hidden_size + h]) +
                      __bfloat162float(delta[coefficient]));
        }
        out[hidden_offset] = __float2bfloat16(value);
    }
}

} // namespace ninfer::ops
