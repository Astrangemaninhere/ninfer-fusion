#pragma once

// ninfer::ops::detail - private launch prototype for suffix_lookup.

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// `history` is the length of ids[0, history) -- the caller's accepted-token buffer. It is
// ENFORCED as of this change: the launcher refuses history <= 0 by name, and every row's
// length is clamped to it inside the kernel, so neither the scan nor the continuation read can
// leave the buffer. Previously the parameter was accepted and then dropped (declared, taken,
// never passed on), which made a wrong `history` indistinguishable from a right one.
void suffix_lookup_launch(const std::int32_t* ids, std::int32_t history,
                          const std::int32_t* starts, const std::int32_t* lengths,
                          std::int32_t batch, std::int32_t query, std::int32_t min_len,
                          std::int32_t continuation_tokens, std::int32_t* best_len,
                          std::int32_t* best_offset, std::int32_t* continuation,
                          cudaStream_t stream);

} // namespace ninfer::ops::detail
