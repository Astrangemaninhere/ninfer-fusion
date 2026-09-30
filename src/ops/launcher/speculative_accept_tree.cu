// F904 / line subagentfix -- the launcher of the tree-accept overwrite.
//
// WHAT IT MUST DO, AND WHAT THE DRIVER ENFORCES. It checks the shapes it indexes (the same checks
// the accept Op makes, minus the ones the Op has already made on the way in), selects the grid from
// the batch dimension, launches the one-thread-per-row kernel on the caller's stream, and returns
// the launch error through CUDA_CHECK. It issues NO host synchronization and NO device-to-host
// copy, so it is legal inside a stream capture -- which matters, because a round body IS the graph
// definition (src/core/decode_graph.cpp:65) and `dl/treesigabrt` (F-866) measured what an ordered
// D2H inside one costs: `cudaErrorStreamCaptureInvalidated` and rc=134.
//
// WHY IT TAKES target_tokens BUT NOT logits: the tree rule compares the verifier's own ARGMAX per
// column (already written into target_tokens by the verify batch) against the draft fed to each
// column. It reads no logit, so it cannot disagree with the verifier about what the argmax was.
#include "ops/launcher/speculative_accept_tree.h"

#include "core/device.h"
#include "ops/kernel/speculative_accept_tree.cuh"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

namespace {

void require_vector_lex(const Tensor& t, DType dtype, std::int32_t n, const char* name) {
    if (t.data == nullptr || t.dtype != dtype || !t.is_contiguous() || t.ne[0] != n) {
        throw std::invalid_argument(std::string("speculative_accept_tree_greedy_overwrite: ") +
                                    name + " must be a contiguous bound vector of the stated width");
    }
}

} // namespace

void speculative_accept_tree_greedy_overwrite(const Tensor& target_tokens, const Tensor& drafts,
                                              const Tensor& current_extents,
                                              const Tensor& column_masks, Tensor& lengths,
                                              Tensor& anchors, Tensor& licensed_tokens,
                                              Tensor& licensed_counts, Tensor& accepted,
                                              Tensor& accepted_columns, std::int32_t token_domain,
                                              const SamplingConfig* configs, cudaStream_t stream) {
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1 || batch < 1) {
        throw std::invalid_argument(
            "speculative_accept_tree_greedy_overwrite: K and B must be positive");
    }
    if (k + 1 > kTreeAcceptMaximumWidth) {
        throw std::invalid_argument(
            "speculative_accept_tree_greedy_overwrite: the frame is wider than the tree accept can "
            "index; the caller must gate on kTreeAcceptMaximumWidth");
    }
    if (token_domain <= 0 || configs == nullptr) {
        throw std::invalid_argument(
            "speculative_accept_tree_greedy_overwrite: token_domain must be positive and configs "
            "must be non-null");
    }
    if (target_tokens.data == nullptr || target_tokens.dtype != DType::I32 ||
        !target_tokens.is_contiguous() || target_tokens.ne[0] != k + 1 ||
        target_tokens.ne[1] != batch) {
        throw std::invalid_argument(
            "speculative_accept_tree_greedy_overwrite: target_tokens must be I32 [K+1,B]");
    }
    if (drafts.data == nullptr || drafts.dtype != DType::I32 || !drafts.is_contiguous()) {
        throw std::invalid_argument("speculative_accept_tree_greedy_overwrite: drafts must be I32");
    }
    if (column_masks.data == nullptr || column_masks.dtype != DType::I64 ||
        !column_masks.is_contiguous() || column_masks.ne[0] != k + 1 ||
        column_masks.ne[1] != batch) {
        throw std::invalid_argument(
            "speculative_accept_tree_greedy_overwrite: column_masks must be I64 [K+1,B]");
    }
    require_vector_lex(current_extents, DType::I32, batch, "current_extents");
    require_vector_lex(lengths, DType::I32, batch, "lengths");
    require_vector_lex(anchors, DType::I32, batch, "anchors");
    require_vector_lex(licensed_counts, DType::I32, batch, "licensed_counts");
    require_vector_lex(accepted, DType::I32, batch, "accepted");
    require_vector_lex(accepted_columns, DType::I32, batch, "accepted_columns");
    if (licensed_tokens.data == nullptr || licensed_tokens.dtype != DType::I32 ||
        !licensed_tokens.is_contiguous() || licensed_tokens.ne[0] != k + 1 ||
        licensed_tokens.ne[1] != batch) {
        throw std::invalid_argument(
            "speculative_accept_tree_greedy_overwrite: licensed_tokens must be I32 [K+1,B]");
    }

    speculative_accept_tree_greedy_kernel<<<batch, 1, 0, stream>>>(
        static_cast<const std::int32_t*>(target_tokens.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<const std::uint64_t*>(column_masks.data), static_cast<std::int32_t*>(lengths.data),
        static_cast<std::int32_t*>(anchors.data),
        static_cast<std::int32_t*>(licensed_tokens.data),
        static_cast<std::int32_t*>(licensed_counts.data), static_cast<std::int32_t*>(accepted.data),
        static_cast<std::int32_t*>(accepted_columns.data), configs, k + 1, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
