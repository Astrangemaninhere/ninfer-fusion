#pragma once

// Sinkhorn-constrained row scales for the rotated NVFP4 K domain.
//
// For every full-attention (layer, kv_head), a token-independent per-channel
// scale s_d in [0.5, 2.0] balances rotated K row RMS before E4M3/E2M1
// quantization. K is multiplied by s_d on cache write; Q is multiplied by
// 1/s_d before QK, so QK^T is preserved. This mainly protects low-energy
// channels whose E4M3 group scale would otherwise collapse to denormals.
//
// The table is baked from kvcalib-a and stored as BF16 words in constant
// memory (16 * 4 * 256 * 2 = 32 KiB, together with the SO(4) rotation table).

#include <cuda_bf16.h>

#include <cstdint>

extern __constant__ unsigned short kGqaKvRowScaleDev[16][4][256];

namespace ninfer::ops {

__device__ __forceinline__ float gqa_kv_row_scale(int layer, int kv_head, int d) {
    const unsigned short raw = ::kGqaKvRowScaleDev[layer][kv_head][d];
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&raw));
}

__device__ __forceinline__ float gqa_kv_row_scale_inv(int layer, int kv_head, int d) {
    return 1.0f / gqa_kv_row_scale(layer, kv_head, d);
}

} // namespace ninfer::ops
