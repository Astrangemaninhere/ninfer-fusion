#pragma once

// ninfer::ops::detail - private launch prototype for logit_policy. Included by
// the wrapper (host) and defined by the launcher (.cu). Not part of the public
// api. See docs/op-development.md \u00a72.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void logit_policy_launch(const Tensor& x, float multiplier, float cap,
                         cudaStream_t stream);

} // namespace ninfer::ops::detail
