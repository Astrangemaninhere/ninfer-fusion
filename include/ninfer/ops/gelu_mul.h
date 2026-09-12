#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise gated GELU activation, exact erf form:
 *
 *   ideal[i] = gelu(gate[i]) * up[i],   gelu(z) = 0.5 * z * (1 + erf(z / sqrt(2))).
 *
 * `gate`, `up`, and `out` are same-shaped BF16 tensors. out is contiguous; gate and up may use
 * arbitrary valid Tensor strides. out must not overlap either input (the two read-only inputs may
 * overlap one another). The oracle evaluates `ideal` in FP64 from the represented inputs: the
 * exact-erf GELU of the represented BF16 gate, multiplied by the represented BF16 up. The BF16
 * output is promoted and compared directly with that result; output storage rounding belongs to the
 * Op's numerical criterion, not the oracle. Private kernel arithmetic is implementation-defined.
 *
 * The exact erf form is the whole domain of this Op on purpose: every registered gated architecture
 * declares `hidden_act = "gelu"`, i.e. the erf activation, so the tanh approximation is a different
 * formula that would need its own criterion rather than a mode of this one. The Op writes all of
 * out and uses no workspace or persistent state.
 */
void gelu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
