#pragma once

// ninfer::ops::detail - private launch prototype for suffix_lookup.

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void suffix_lookup_launch(const std::int32_t* ids, std::int32_t history,
                          const std::int32_t* starts, const std::int32_t* lengths,
                          std::int32_t batch, std::int32_t query, std::int32_t min_len,
                          std::int32_t continuation_tokens, std::int32_t* best_len,
                          std::int32_t* best_offset, std::int32_t* continuation,
                          cudaStream_t stream);

} // namespace ninfer::ops::detail
