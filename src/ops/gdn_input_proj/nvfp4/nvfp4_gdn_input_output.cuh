#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4GdnInputOutput {
    // Row split of the fused projection, declared once in nvfp4_config.h.
    static constexpr std::int32_t kQkvRows = kNvfp4GdnQkvRows;
    static constexpr std::int32_t kZRows   = kNvfp4GdnZRows;

    __nv_bfloat16* qkv;
    __nv_bfloat16* z;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kQkvRows) {
            return qkv + static_cast<std::int64_t>(token) * kQkvRows + parent_row;
        }
        return z + static_cast<std::int64_t>(token) * kZRows + parent_row - kQkvRows;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *destination(parent_row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

static_assert((Nvfp4GdnInputOutput::kQkvRows % 128) == 0);
static_assert((Nvfp4GdnInputOutput::kZRows % 128) == 0);

} // namespace ninfer::ops::detail
