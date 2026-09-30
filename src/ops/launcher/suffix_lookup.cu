#include "ops/kernel/suffix_lookup.cuh"
#include "ops/launcher/suffix_lookup.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

void suffix_lookup_launch(const std::int32_t* ids, std::int32_t history,
                          const std::int32_t* starts, const std::int32_t* lengths,
                          std::int32_t batch, std::int32_t query, std::int32_t min_len,
                          std::int32_t continuation_tokens, std::int32_t* best_len,
                          std::int32_t* best_offset, std::int32_t* continuation,
                          cudaStream_t stream) {
    if (batch <= 0 || query <= 0 || continuation_tokens <= 0) { return; }
    // ---------------------------------------------------------------------------------------
    // `history` WAS A DEAD PARAMETER and is load-bearing now. It used to be declared, taken,
    // validated around (the wrapper checks history >= query + continuation_tokens) and then
    // passed to nothing: the launch below forwarded neither it nor anything derived from it, so
    // a caller that reached this function directly could not tell a valid history from a wrong
    // one. It is the length of the accepted-token buffer ids[0, history), which is a real
    // bound, and it is enforced in two places: here (a named refusal for a value that admits no
    // window at all) and in the kernel (each row's length is clamped to it, so no scan and no
    // continuation read can leave the buffer). For a conforming caller -- lengths[b] <=
    // history, which is what "the history is ids[0, len[b])" means -- the clamp is the identity.
    // ---------------------------------------------------------------------------------------
    if (history <= 0) {
        throw std::invalid_argument(
            "suffix_lookup: history must be > 0 (it is the length of the accepted-token buffer "
            "ids[0, history), so with history <= 0 there is no window to scan)");
    }
    constexpr int kBlock = 256;
    const int grid = batch;
    suffix_lookup_kernel<kBlock><<<grid, kBlock, 0, stream>>>(
        ids, history, starts, lengths, batch, query, min_len, continuation_tokens, best_len,
        best_offset, continuation);
}

} // namespace ninfer::ops::detail
