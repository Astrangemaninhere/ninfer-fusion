#pragma once
// prism_fold_launch.h -- the launch prototype for Route A's activation-side transform.
//
// This is the LAUNCH SITE, and (like `e8_activation_rotation.h` beside it) it is deliberately
// the only one.  It is NOT an entry in the `linear` dispatch: `src/ops/linear/linear.cpp`
// admits a `Weight` by its `NumericFormat`/`QType` and Route A is not a format, so nothing on
// any runtime path can reach this operator by accident.
//
// ⛔ AND IT IS NOT LINKED INTO THE ENGINE.  `prism_fold_launch.cu` is not in
// `src/CMakeLists.txt`'s `ninfer_ops` source list, that list is explicit (no GLOB), and
// CMakeLists is forbidden to this line.  So `kPrismFoldRuntimeSupportsTransform` in
// `prism_fold.h` stays `false` for every engine TU and a folded checkpoint reaching `linear`
// is REFUSED BY NAME.  The declaration below exists so that admitting the file later is a
// build-list change and nothing else -- and so that the probe can state, in one place, exactly
// which symbol the link is waiting for.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::prism_fold {

// Rotate the contraction axis of an F32 activation into the weights' basis, `A a = H (D a)`,
// blockwise over `block`, and OPTIONALLY permute the feature axis of each token first.
//
//   x, out    : [K, T] F32, contiguous, K = ne[0] innermost, one contiguous run of K per token.
//   signs     : [K] F32 (device), the contract's own sign vector for this width.  NOT optional:
//               the signs are half of `A`, and a caller with an `identity` sign_mode passes a
//               vector of ones rather than a null.
//   block     : the contract's `prism.hadamard.block_size` (1024 for the released file).
//   n_v, n_k  : `qwen35.ssm.time_step_rank` and `qwen35.ssm.group_count` when the caller wants
//               the GDN permutation (i.e. for `blk.*.ssm_out.weight`); pass BOTH ZERO for the
//               other four tensors and for every non-GDN tensor.
//   stream    : the caller's stream.
//
// Order, and it is load-bearing: PERMUTE, then SIGN, then ROTATE.  Permuting after the rotation
// would be a different operator -- a permutation and a dense rotation do not commute -- and the
// fold's own refusal message in `gguf_extract.py` has always stated this order.
//
// REFUSES BY NAME (throws `std::invalid_argument`) and NEVER PADS.  A pad would change the dot
// product and would do it silently:
//   * x/out not F32, or not [K,T] rank-2, or not contiguous, or not distinct-or-aliased safely,
//   * `k` not a positive multiple of `block`,
//   * `block` not a power of two, or above `kPrismFoldBlock`,
//   * `signs` null,
//   * `(n_v, n_k)` non-zero but not a legal GDN geometry (n_v % n_k, k % n_v).
//
// ⛔ BF16 IS NOT SERVED HERE, AND THAT IS A DECLARED GAP RATHER THAN AN OVERSIGHT.  `linear`
// requires a BF16 activation (`src/ops/linear/linear.cpp:53-55`), so a real call site needs a
// bf16 instantiation of this kernel, which is NOT written and NOT tested because there is no
// GPU window to test it in.  Passing F32 and casting around the call would round the activation
// to bf16 AFTER rotating it, which is a different operator from rotating a bf16 activation, and
// the difference is not one this file is entitled to paper over.
void prism_fold_activation_launch(const Tensor& x, Tensor& out, const Tensor& signs, int block,
                                  int n_v, int n_k, cudaStream_t stream);

// The host-side shape predicate, so a call site can refuse before it allocates anything.  Same
// rules as the launch.
[[nodiscard]] bool prism_fold_activation_shape_ok(const Tensor& x, const Tensor& out,
                                                  const Tensor& signs, int block, int n_v,
                                                  int n_k) noexcept;

}   // namespace ninfer::ops::prism_fold
