#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear/q5/q5_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

Q5Launch select_q5_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
Q5Launch select_q5_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

/// True when the A16 registry carries an exact entry for (n, k, t).
[[nodiscard]] bool q5_a16_shape_registered(std::int32_t n, std::int32_t k,
                                           std::int32_t t) noexcept;

void q5_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

} // namespace ninfer::ops::detail
