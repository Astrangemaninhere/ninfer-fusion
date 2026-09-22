#pragma once

#include <cuda_runtime.h>

namespace ninfer::ops {

inline constexpr int kWarpSize = 32;

// The shuffle mask's WIDTH is not a free choice: each backend fixes it, and they fix it
// differently.
//   CUDA: `__shfl_*_sync(unsigned mask, ...)` takes a 32-bit mask.
//   HIP:  `__shfl_*_sync(MaskT mask, ...)` is guarded by
//         static_assert(sizeof(MaskT) == 8, "The mask must be a 64-bit integer. Implicitly
//         promoting a smaller integer is almost always an error.")
//         in hip/amd_detail/amd_warp_sync_functions.h -- enforced in wave32 as well as wave64
//         codegen; a wave32 target narrows only the VALUE (__hip_adjust_mask_for_wave32).
// So one 32-bit constant is refused by the HIP compiler, and one 64-bit constant is the wrong
// type for the CUDA entry points. Take the TYPE from the device target and the VALUE from the
// type: ~WarpMask{0} is 0xffffffffu on CUDA, 0xffffffffffffffff on an AMD device.
#if defined(__AMDGCN__) || defined(__HIP_PLATFORM_AMD__)
using WarpMask = unsigned long long;
#else
using WarpMask = unsigned int;
#endif

inline constexpr WarpMask kFullWarpMask = ~WarpMask{0};

template <int Width = kWarpSize, class T>
__device__ __forceinline__ T warp_sum(T x, WarpMask mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(mask, x, offset, Width);
    }
    return x;
}

template <int Width = kWarpSize, class T>
__device__ __forceinline__ T warp_reduce_sum(T x, WarpMask mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(mask, x, offset, Width);
    }
    return x;
}

template <int Width = kWarpSize>
__device__ __forceinline__ float warp_max(float x, WarpMask mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(mask, x, offset, Width));
    }
    return x;
}

template <int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float x, float* sums) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int Warps = BlockSize / kWarpSize;

    x = warp_reduce_sum(x);
    if constexpr (Warps == 1) { return x; }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) { sums[warp] = x; }
    __syncthreads();

    x = threadIdx.x < Warps ? sums[lane] : 0.0f;
    if (warp == 0) { x = warp_reduce_sum<Warps>(x); }
    return x;
}

} // namespace ninfer::ops
