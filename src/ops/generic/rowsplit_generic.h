#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// ---------------------------------------------------------------------------
// Generic (unregistered-shape) linear fallback.
//
// The registered per-format kernels are *shape registries*: `select_q4_launch`,
// `select_q5_launch`, `select_q6_launch`, `bf16_linear_add_select`, the five
// `linear_add` variants and the fused `attn_input_proj` / `gdn_input_proj` /
// `gdn_gating_proj` plans each admit a finite list of exact (N, K) problems, and every entry in
// those lists came from some model's geometry.  A target whose geometry no entry covers cannot
// be executed at all, which is a *kernel-directory* gap, not a target-registry gap.
//
// This module closes the gap the only way that is unconditionally correct: it decodes the same
// k128-v1 row-split payload with the same bit conventions (see
// ops/linear/q{4,5,6}/*_rowsplit_storage.cuh) and evaluates the same ideal dot product
//   ideal[n,t] = sum_k FP32Dequant(w)[n,k] * FP32(x)[k,t]
// with a naive, shape-agnostic kernel.  It is deliberately not fast: it exists so that a shape
// the registries do not carry still produces the right numbers.  Performance work (adding real
// registry entries) is strictly additive on top of it.
// ---------------------------------------------------------------------------

/// The fallback's kill switch. `NINFER_GENERIC_ROWDEC=0` restores the pre-fallback behaviour --
/// every unregistered shape is refused exactly as before -- which is what makes "the fallback is
/// what changed the outcome" a falsifiable claim rather than a story. Default: enabled.
[[nodiscard]] bool generic_fallback_enabled() noexcept;

/// `generic_fallback_enabled()` plus the shape-level feasibility the decoder needs: positive
/// extents, and a reduction dimension that is a whole number of 64-element groups.
[[nodiscard]] bool generic_rowsplit_shape_capable(std::int32_t output_rows,
                                                  std::int32_t input_rows) noexcept;

/// True when `w` is a k128-v1 row-split Q4G64_F16S / Q5G64_F16S / Q6G64_F16S weight (or a row
/// view of one) whose declared geometry this module can decode exactly.
[[nodiscard]] bool generic_rowsplit_weight_ok(const Weight& w) noexcept;

/// True when `generic_rowsplit_linear_dispatch(x, w, out)` is a well-formed call.  `out` need
/// not be contiguous: a dim-0 row view of a wider parent is a first-class destination, and its
/// token stride is taken from its own nb[1].  `x` must be contiguous.
[[nodiscard]] bool generic_rowsplit_problem_ok(const Tensor& x, const Weight& w,
                                               const Tensor& out) noexcept;

/// out[N,T] = W[N,K] * x[K,T], with `w` a generic_rowsplit_weight_ok weight.
void generic_rowsplit_linear_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream);

/// residual[N,T] += W[N,K] * x[K,T], with `w` a generic_rowsplit_weight_ok weight.
void generic_rowsplit_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual,
                                          cudaStream_t stream);

/// The [row_begin, row_begin+rows) window of a row-split weight.  Same conventions as the target
/// loaders' `row_view`: the low plane, the high plane and the scale plane are each advanced by
/// their own per-row stride, and n/shape[0]/padded_shape[0] become `rows`.
[[nodiscard]] Weight generic_rowsplit_row_view(const Weight& w, std::int32_t row_begin,
                                               std::int32_t rows);

/// True when `w` is a contiguous BF16_CTRL weight of the declared [N,K] shape.
[[nodiscard]] bool generic_bf16_weight_ok(const Weight& w) noexcept;

/// out[N,T] = W[N,K] * x[K,T] for a contiguous BF16_CTRL weight.
void generic_bf16_linear_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream);

/// True when the generic GDN gating-projection call is well formed.
[[nodiscard]] bool generic_norm_gating_problem_ok(const Tensor& x, const Tensor& norm_weight,
                                                  const Weight& a_weight, const Weight& b_weight,
                                                  const Tensor& A_log, const Tensor& dt_bias,
                                                  const Tensor& h, const Tensor& g,
                                                  const Tensor& beta) noexcept;

/// The reference spelling of the fused BF16 GDN norm-gating projection:
///
///   h[d,t]      = x[d,t] * rsqrt(mean_d(x^2) + eps) * (1 + norm_weight[d])   (rmsnorm unit offset)
///   a[row,t]    = sum_d A[row,d] * h[d,t]
///   b[row,t]    = sum_d B[row,d] * h[d,t]
///   g[row,t]    = -exp(A_log[row]) * softplus(a[row,t] + dt_bias[row])
///   beta[row,t] = sigmoid(b[row,t])
///
/// This is bit-for-bit the arithmetic of `bf16_gdn_gating_proj_gemm_mma.cuh` with
/// `NormalizeInput = true` (the staged x tile is pre-scaled by `1 + norm_weight` and the post-MMA
/// scale is the same `inv`), just evaluated by a generic kernel instead of the registered
/// geometry's MMA.
void generic_norm_gating_proj_dispatch(const Tensor& x, const Tensor& norm_weight, float eps,
                                       const Weight& a_weight, const Weight& b_weight,
                                       const Tensor& A_log, const Tensor& dt_bias, Tensor& h,
                                       Tensor& g, Tensor& beta, cudaStream_t stream);

/// out[i,t] = silu(gate_up[i,t]) * gate_up[i + out.ne[0], t] -- the materialized half of
/// `linear_swiglu`, matching `linear_swiglu/q4/q4_linear_swiglu_plan.cpp`'s Materialized
/// schedule (which is itself `linear` into a [2I,T] scratch followed by exactly this split).
void generic_swiglu_split_dispatch(const Tensor& gate_up, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
