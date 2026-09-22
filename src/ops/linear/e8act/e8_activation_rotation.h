#pragma once

// ninfer::ops::e8act - launch prototype for the activation-side Hadamard rotation.
//
// This is the LAUNCH SITE, and it is deliberately the only one. It is NOT an entry in the
// `linear` dispatch: src/ops/linear/linear.cpp admits a Weight by its NumericFormat and
// this operator is not a format, so nothing on any runtime path can reach it by accident
// (see dl/e8model/REPORT.md, gap 4 -- the artifact gate is the missing half, and it is not
// this file's to take).

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::e8act {

// Rotate the contraction axis of a BF16 activation in 64-element blocks: `out` receives
//
//     out[t, g*64 + j] = sum_i H64[j, i] * x[t, g*64 + i]      for every g, j
//
// with H64 = H_64 / 8 the normalized Sylvester matrix, i.e. the SAME orthogonal map
// `e8_lattice_hadamard64_t` applies on the KV side. x and out are [K, T] BF16, contiguous,
// with K = ne[0] innermost. `out` may alias `x` (the block is staged in registers before
// the store).
//
// REFUSES BY NAME, does not pad: throws std::invalid_argument when
//   * x/out are not BF16, or not [K,T] rank-2, or not contiguous,
//   * k is not a positive multiple of 64,
//   * the shapes disagree.
// A pad would change the dot product, and it would do so silently.
void e8act_rotate_activation_launch(const Tensor& x, Tensor& out, cudaStream_t stream);

// The F32 spelling, for a caller that has not yet rounded to BF16. Same contract.
void e8act_rotate_activation_launch_f32(const Tensor& x, Tensor& out, cudaStream_t stream);

}   // namespace ninfer::ops::e8act
