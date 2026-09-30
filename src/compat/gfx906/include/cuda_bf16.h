// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : JCraigWasTaken/ninfer-gfx906
// Branch      : gfx906-port
// Commit      : 7a3c18d9fcfc977fc9c35766fd9938c0039e9a78
// Source path : src/compat/gfx906/include/cuda_bf16.h
// sha256(src) : e7c884efda5ca297c196d5d9e36a5e6bb7a007aeae937ff6085fcbc8b4994e3d
// Landed by   : /home/user/ninfer-fusion gfx906port line, 2026-09-21 (mi50 3-step plan, step 2 of 3: additive port layer; step 3 = CMake wiring, NOT done here).
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : CUDA->HIP shim header. It SHADOWS a real CUDA header of the same name the moment src/compat/gfx906/include is put ahead of the CUDA include dir -- which is exactly what the fork does, include_directories(BEFORE ...) at src/CMakeLists.txt:5, tests/CMakeLists.txt:2 and bench/CMakeLists.txt:2. Not adding that line is what keeps it inert here. See docs/gfx906/PORT-AUDIT.md.
#pragma once
#include "core/hip_compat.h"

#if defined(__HIPCC__)

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

// ROCm's hip_bf16.h mirrors the CUDA bf16 API on __hip_bfloat16 types.
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
typedef __hip_bfloat16_raw __nv_bfloat16_raw;
typedef __hip_bfloat162_raw __nv_bfloat162_raw;

// Present in CUDA but absent from ROCm 6.4's header: explicit
// round-to-nearest-even variants. The HIP base conversions round RNE.
__device__ __host__ inline __hip_bfloat16 __float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}

__device__ __host__ inline __hip_bfloat162 __floats2bfloat162_rn(float x, float y) {
    return __float22bfloat162_rn(float2{x, y});
}

__device__ inline __hip_bfloat162 __hsub2_rn(__hip_bfloat162 a, __hip_bfloat162 b) {
    return __hsub2(a, b);
}

#else // host-only translation unit

// hip_bf16.h cannot be parsed outside HIP compilation (it reaches for amdgcn
// builtins), but host code only moves bf16 data around. Provide
// layout-compatible stand-ins; all arithmetic stays device-side.
struct __nv_bfloat16 {
    unsigned short x;
};
struct __nv_bfloat162 {
    __nv_bfloat16 x, y;
};
struct __nv_bfloat16_raw {
    unsigned short x;
};
struct __nv_bfloat162_raw {
    unsigned short x, y;
};

#endif // __HIPCC__
