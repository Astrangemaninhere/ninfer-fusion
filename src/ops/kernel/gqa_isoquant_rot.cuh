#pragma once

// Baked IsoQuant per-4-channel SO(4) rotations, [64][4][4] fp32,
// imported from the nvfp4rtx offline calibration (isoquant_rot.npy).
// The table lives in constant memory (gqa_isoquant_rot.cu) so kernels index
// it with LDC; a function-local constexpr copy previously made nvcc expand
// the whole table into the hot Q/K quantization loops and spill it through
// the per-thread stack. Applied to K on cache write and to Q before NVFP4
// quantization so QK^T is preserved in the rotated domain.

extern __constant__ float kGqaIsoquantRotDev[64][4][4];

namespace ninfer::ops {

__device__ __forceinline__ float gqa_isoquant_rot_value(int block, int row, int col) {
    return ::kGqaIsoquantRotDev[block][row][col];
}

} // namespace ninfer::ops
