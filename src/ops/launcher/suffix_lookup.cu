#include "ops/kernel/suffix_lookup.cuh"
#include "ops/launcher/suffix_lookup.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

void suffix_lookup_launch(const std::int32_t* ids, std::int32_t history,
                          const std::int32_t* starts, const std::int32_t* lengths,
                          std::int32_t batch, std::int32_t query, std::int32_t min_len,
                          std::int32_t continuation_tokens, std::int32_t* best_len,
                          std::int32_t* best_offset, std::int32_t* continuation,
                          cudaStream_t stream) {
    if (batch <= 0 || query <= 0 || continuation_tokens <= 0) { return; }
    constexpr int kBlock = 256;
    const int grid = batch;
    suffix_lookup_kernel<kBlock><<<grid, kBlock, 0, stream>>>(
        ids, starts, lengths, batch, query, min_len, continuation_tokens, best_len,
        best_offset, continuation);
}

} // namespace ninfer::ops::detail
