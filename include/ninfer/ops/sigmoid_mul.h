#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise sigmoid gate:
 *
 *   ideal[i] = x[i] * (1 / (1 + exp(-gate[i]))).
 *
 * `gate` and `x` are non-overlapping contiguous BF16 tensors in one of two forms:
 *
 *   per-element:     x is `[D,H,T]`, gate is `[D,H,T]` (same shape)
 *   headwise scalar: x is `[D,H,T]`, gate is `[H,T]` (one scalar per head and token)
 *
 * The headwise form is the attention-output gate of a headwise-gated architecture: its
 * gate element `h + H*t` is broadcast over the whole head_dim axis, so the formula becomes
 * `ideal[d,h,t] = x[d,h,t] * (1 / (1 + exp(-gate[h,t])))`. It is selected by shape -
 * `x.ne[3] == 1`, `gate.ne[2] == gate.ne[3] == 1`, `gate.ne[0] == x.ne[1]`,
 * `gate.ne[1] == x.ne[2]` - and no registered caller satisfies it (all three live call
 * sites pass a `[head_dim, n_q, T]` gate, and no geometry has `n_q == head_dim`), so the
 * per-element route is unchanged. The two forms overlap only for a single-element tensor,
 * where both compute the same value.
 *
 * The oracle evaluates `ideal` in FP64 from the represented inputs. The updated BF16 x is
 * promoted and compared directly with that result; output storage rounding belongs to the
 * Op's numerical criterion, not the oracle. Private kernel arithmetic is implementation-
 * defined. The Op uses no workspace or other persistent state.
 */
void sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
