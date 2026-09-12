#pragma once

// Baked IsoQuant per-4-channel SO(4) rotations, [64][4][4] fp32,
// imported from the nvfp4rtx offline calibration (isoquant_rot.npy).
// The table lives in constant memory (gqa_isoquant_rot.cu) so kernels index
// it with LDC; a function-local constexpr copy previously made nvcc expand
// the whole table into the hot Q/K quantization loops and spill it through
// the per-thread stack. Applied to K on cache write and to Q before NVFP4
// quantization so QK^T is preserved in the rotated domain.
//
// SEPARATION (kernel-side gate). The rotation is now a component that can be
// switched off from the host, and the gate lives NEXT TO THE MATH, not in the
// option layer: kGqaIsoquantRotGeom[0] is a device-visible enable word and
// gqa_isoquant_rot_block4() is the ONLY function that reads it. With the word
// at its baked default (1) every instruction below is byte-identical to the
// pre-separation code; with the word at 0 the block helper returns the input
// block untouched, i.e. it applies the identity SO(4) map, which is the exact
// "rotation disabled" semantics (see the zero-cost argument in the report).
//
// Why the gate is at 4-channel-block granularity and not per coefficient:
// the identity SO(4) map is I, and there is no scalar (c, d) with
// R_off = c * R + d * I for all blocks, so a "multiply by a 1/0 coefficient"
// cannot express the off state -- only an early return can. Putting the early
// return around the whole 4x4 apply (instead of around each of the four
// kGqaIsoquantRotDev reads) keeps the on-path identical (16 LDC + 16 FFMA,
// same association order, hence bit-identical results) while making the gate
// one warp-uniform test per 4-channel block instead of four selects plus four
// table reads per block.

extern __constant__ float kGqaIsoquantRotDev[64][4][4];

// Runtime gate, {enabled, reserved, reserved, reserved}. Word 0 is the only
// one read by kernels; the baked initializer is 1, so an engine that never
// touches this descriptor reproduces the pre-separation behaviour exactly.
// Written once at plan time (kv_rotation_apply_spec, gqa_isoquant_rot.cu) and
// read with a single warp-uniform LDC that is loop-invariant at every callsite.
extern __constant__ int kGqaIsoquantRotGeom[4];

namespace ninfer::ops {

__device__ __forceinline__ float gqa_isoquant_rot_value(int block, int row, int col) {
    return ::kGqaIsoquantRotDev[block][row][col];
}

// The gate. Warp-uniform by construction (one __constant__ word), so the
// branch it guards never diverges.
__device__ __forceinline__ bool gqa_isoquant_rot_enabled() {
    return ::kGqaIsoquantRotGeom[0] != 0;
}

// Rotate ONE 4-channel block in place: x <- R(block) * x, where `block` is the
// 4-channel block index (d >> 2) and x points at that block's four contiguous
// channels. THE single gate: every kernel path that used to call the scalar
// accessor four times now calls this instead, so "rotation off" cannot be
// half-applied to one side of the QK^T product by accident -- the same word
// disables the K-write rotate and the Q-read rotate, which is what keeps QK^T
// invariant in the off state (identity on both sides).
__device__ __forceinline__ void gqa_isoquant_rot_block4(float* x, int block) {
    if (!gqa_isoquant_rot_enabled()) { return; }
    const float x0 = x[0];
    const float x1 = x[1];
    const float x2 = x[2];
    const float x3 = x[3];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        x[row] = gqa_isoquant_rot_value(block, row, 0) * x0 +
                 gqa_isoquant_rot_value(block, row, 1) * x1 +
                 gqa_isoquant_rot_value(block, row, 2) * x2 +
                 gqa_isoquant_rot_value(block, row, 3) * x3;
    }
}

} // namespace ninfer::ops
