#pragma once

// Sinkhorn-constrained row scales for the rotated NVFP4 K domain.
//
// For every full-attention (layer, kv_head), a token-independent per-channel
// scale s_d in [0.5, 2.0] balances rotated K row RMS before E4M3/E2M1
// quantization. K is multiplied by s_d on cache write; Q is multiplied by
// 1/s_d before QK, so QK^T is preserved. This mainly protects low-energy
// channels whose E4M3 group scale would otherwise collapse to denormals.
//
// S3 (row-scale wall removal): the table is a FLAT constant pool of BF16 words
// addressed through the runtime geometry descriptor below, not a [16][4][256]
// array baked for one model. The pool keeps a compile-time CAPACITY (the old
// 27b extent, so the constant budget does not move); the live geometry -- how
// many layers / kv_heads / head_dim the words are strided by -- arrives at plan
// time from the loader (gqa_isoquant_row_scale_loader.cu), which also uploads
// the baked payload. Any model with layers*kv_heads*head_dim <= capacity uses
// the calibrated table with no engine change; a model outside the uploaded
// geometry still takes the identity path, exactly like the old fixed extent.
//
// The baked pool + descriptor cover qwen3.8-27b (16 * 4 * 256 * 2 = 32 KiB,
// together with the 4 KiB SO(4) rotation table).

#include <cuda_bf16.h>

#include <cstdint>

// Capacity of the flat pool in BF16 words (compile-time upper bound, 32 KiB).
// The host-side gate carries the same number as kKvRowScalePoolCapacity
// (gqa_isoquant_row_scale_loader.h); loader.cu static_asserts them equal.
inline constexpr int kKvRowScalePoolWords = 16384;

// The geometry the payload below was calibrated for: qwen3.8-27b's 16
// full-attention layers x 4 KV heads x 256 channels. Single-sourced here so
// the device initializer of kGqaKvRowScaleGeom and the loader's "back to auto"
// restore (kv_rowscale_sidecar_apply_spec) can never drift; a static_assert in
// gqa_isoquant_row_scale.cu binds the product to kKvRowScalePoolWords.
inline constexpr int kKvRowScaleBakedGeom[3] = {16, 4, 256};

// BF16 row scales, row-major [layer][kv_head][d] with the RUNTIME strides in
// kGqaKvRowScaleGeom[0..2]. The baked payload covers 27b; the loader overwrites
// it (with the descriptor) when NINFER_KV_ROWSCALE names a sidecar.
extern __constant__ unsigned short kGqaKvRowScalePool[kKvRowScalePoolWords];

// Runtime geometry descriptor, {layers, kv_heads, head_dim, words}:
//   [0] layers   -- full-attention layers the pool covers (upper guard bound)
//   [1] kv_heads -- KV heads per layer (row stride)
//   [2] head_dim -- channels per kv_head (inner stride)
//   [3] words    -- valid words == layers*kv_heads*head_dim, the invariant
//                   kv_rowscale_sidecar_parse() enforces; it is what makes
//                   "every address the guard admits" provably < words, and is
//                   otherwise carried for self-description only.
// The four words share one 16-byte constant line, so a single warp-uniform LDC
// can fetch them; the accessor reads [0..2].
// A never-written (all-zero) descriptor makes every range check below fail, so
// the pool is never read and the accessor answers identity. That is exactly the
// row-scale OFF state (--kv-row-scale off / NINFER_KV_ROWSCALE=off): the
// identity map is expressed by the guard the on-path already pays for, so the
// switch adds no kernel instruction and needs no all-ones sidecar file.
extern __constant__ int kGqaKvRowScaleGeom[4];

namespace ninfer::ops {

__device__ __forceinline__ float gqa_kv_row_scale(int layer, int kv_head, int d) {
    // Runtime geometry: the GEOMETRY words are warp-uniform (broadcast LDC, hoisted).
    // The per-channel index is NOT: it is grp*16 + lane*4 + j, so a warp's four loads span
    // 16 channels -- up to 4 constant cache lines when the caller runs all 32 lanes. Call
    // sites must therefore keep this multiply inside the lane<4 window of the rotate helper
    // (both the decode and the prefill K/Q sites do); the prefill site used to be the only
    // one that did, and the extra span was pure tax, not numerics.
    const int layers = ::kGqaKvRowScaleGeom[0];
    const int kvh    = ::kGqaKvRowScaleGeom[1];
    const int hdim   = ::kGqaKvRowScaleGeom[2];
    // Out-of-geometry reads return 1.0. An out-of-bounds LDC returned garbage,
    // which made a wider model's K write zero/NaN and its Q scale infinite
    // (NaN attention output, _TODO.md 104). Identity keeps QK^T exact (K*s on
    // write, Q/s on read) and only leaves the rotation-domain quantization
    // uncalibrated. Same guard the fixed [16][4][256] extent had.
    if (layer < 0 || layer >= layers || kv_head < 0 || kv_head >= kvh || d < 0 ||
        d >= hdim) {
        return 1.0F;
    }
    const unsigned short raw =
        ::kGqaKvRowScalePool[(layer * kvh + kv_head) * hdim + d];
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&raw));
}

__device__ __forceinline__ float gqa_kv_row_scale_inv(int layer, int kv_head, int d) {
    return 1.0f / gqa_kv_row_scale(layer, kv_head, d);
}

} // namespace ninfer::ops
