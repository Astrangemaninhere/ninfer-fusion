// ninfer::ops - suffix_lookup wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/suffix_lookup.h"

#include "ops/launcher/suffix_lookup.h" // detail::suffix_lookup_launch

#include <stdexcept>
#include <string>

namespace ninfer::ops {

void suffix_lookup(const void* ids, std::int32_t history,
                   const std::int32_t* starts, const std::int32_t* lengths,
                   std::int32_t batch, std::int32_t query, std::int32_t min_len,
                   std::int32_t continuation_tokens, std::int32_t* best_len,
                   std::int32_t* best_offset, std::int32_t* continuation,
                   cudaStream_t stream) {
    const auto fail = [](const char* what) {
        throw std::invalid_argument(std::string("suffix_lookup: ") + what);
    };
    if (ids == nullptr || starts == nullptr || lengths == nullptr || best_len == nullptr ||
        best_offset == nullptr || continuation == nullptr) {
        fail("pointers must be non-null");
    }
    if (batch < 0 || query < 1 || continuation_tokens < 1) {
        fail("batch >= 0, query >= 1, continuation_tokens >= 1 required");
    }
    if (min_len < 1 || min_len > query) {
        fail("min_len must be within [1, query]");
    }
    if (history < query + continuation_tokens) {
        fail("history shorter than query + continuation_tokens");
    }
    detail::suffix_lookup_launch(static_cast<const std::int32_t*>(ids), history, starts,
                                 lengths, batch, query, min_len, continuation_tokens,
                                 best_len, best_offset, continuation, stream);
}

} // namespace ninfer::ops
