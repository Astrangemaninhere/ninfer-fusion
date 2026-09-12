#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * In-place final logit policy (auto-adaptation hook for softcapped heads):
 *
 *   y[i] = multiplier * x[i];  if cap > 0:  y[i] = cap * tanh(y[i] / cap)
 *
 * Mirrors the HF ordering: `logits *= output_multiplier`, then
 * `logits = T * tanh(logits / T)` when final_logit_softcapping = T. A
 * nonpositive cap disables the tanh stage, so the default architecture
 * policy (multiplier = 1, cap = 0) is a bit-exact identity. `x` is a
 * contiguous BF16 tensor; the op is in-place and uses no workspace.
 */
void logit_policy(Tensor& x, float multiplier, float cap, cudaStream_t stream);

} // namespace ninfer::ops
