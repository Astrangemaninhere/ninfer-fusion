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

// ---------------------------------------------------------------------------
// APPENDED FROM THE FORK TIP, VERBATIM.  NEVER A REPLACEMENT -- item 0003.
// ---------------------------------------------------------------------------
// src/ops/linear/gfx906/q_gemv_gfx906.cuh:256 calls gfx906_reduce_sum32, and the
// two q4_q5 tiled TUs include that header.  In the fork the helper lives in THIS
// file at :57 -- but the fork's copy of this file is the OLDER branch: it has no
// WarpMask and none of the 64-bit-mask work this tree's copy carries.  Measured,
// per dl/gfx906port/REPORT.md section 11 item 3:
//     tree b92c327467ce0d1ba30b4a2f7d92449da250a4d070ce458681549d66f37669b1 (2872 B)
//     fork 4c3abe09408b5b94c793ad465c16ccfffaab0bf237bc59cca7e0234b1248e49e (3942 B)
// A whole-file replace reverts the tree's newer branch.  The appended block below
// is lifted from that fork file by content-hash, NOT retyped, and it carries the
// fork's OWN guard (`NINFER_GFX906_COMPAT`), which is the same macro item 0001's
// option defines -- so with the option OFF the other 47 TUs that include this
// header preprocess it to exactly the bytes they preprocess today.
//
#if defined(NINFER_GFX906_COMPAT)
// Wave64 DPP reduction (pass 2, donor llama.cpp-gfx906 warp_reduce_amd_f32
// ladder): sums x across each 32-lane half of a wave64 independently, both
// halves at once, using row-local DPP moves instead of ds_bpermute shuffles.
// Every lane of a half ends with that half's total.
//   xor1 / xor2 : quad_perm(1,0,3,2) = 0xB1, quad_perm(2,3,0,1) = 0x4E
//   +4 / +8     : row_ror:4 = 0x124, row_ror:8 = 0x128 (after the quad steps
//                 every lane of a quad holds the quad sum, so rotating by 4 then
//                 8 within the 16-lane row gives the row total in every lane)
//   xor16       : ds_swizzle bit mode (and 0x1f, or 0, xor 0x10) = 0x401F; the
//                 swizzle never crosses a 32-lane group, which is the point.
// Used by the pass-2 GEMV only; warp_reduce_sum users are unchanged.
__device__ __forceinline__ float gfx906_reduce_sum32(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    // row_mask = bank_mask = 0xf (all rows/banks), bound_ctrl = true.
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0xB1, 0xf, 0xf, true));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x4E, 0xf, 0xf, true));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x124, 0xf, 0xf, true));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x128, 0xf, 0xf, true));
    x += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(x), 0x401F));
#else
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        x += __shfl_xor_sync(kFullWarpMask, x, offset, 32);
    }
#endif
    return x;
}
#endif // NINFER_GFX906_COMPAT

} // namespace ninfer::ops
