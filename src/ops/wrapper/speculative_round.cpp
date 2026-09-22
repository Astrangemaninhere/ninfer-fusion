#include "ninfer/ops/speculative_round.h"
#include "ops/launcher/speculative_round.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

void require_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void require_dtype(const Tensor& t, DType dtype, const char* op, const char* name) {
    if (t.dtype != dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid dtype for " + name);
    }
    require_contiguous_nonnull(t, op, name);
}

void require_scalar(const Tensor& t, DType dtype, const char* op, const char* name) {
    require_dtype(t, dtype, op, name);
    if (t.ne[0] != 1 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid scalar shape for " + name);
    }
}

void require_vector(const Tensor& t, DType dtype, std::int32_t n, const char* op,
                    const char* name) {
    require_dtype(t, dtype, op, name);
    if (n <= 0 || t.ne[0] != n || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid vector shape for " + name);
    }
}

void require_matrix(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                    const char* op, const char* name) {
    require_dtype(t, dtype, op, name);
    if (rows <= 0 || cols <= 0 || t.ne[0] != rows || t.ne[1] != cols || t.ne[2] != 1 ||
        t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid matrix shape for " + name);
    }
}

} // namespace

std::size_t speculative_accept_greedy_drafts_workspace_capacity_bytes(std::int32_t token_domain,
                                                                      std::int32_t min_drafts,
                                                                      std::int32_t max_drafts,
                                                                      std::int32_t min_batch,
                                                                      std::int32_t max_batch) {
    if (token_domain <= 0 || min_drafts <= 0 || max_drafts < min_drafts || min_batch <= 0 ||
        max_batch < min_batch || max_drafts == std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("speculative accept workspace: invalid draft interval");
    }
    const std::size_t row_bytes =
        sampling_workspace_capacity_bytes(token_domain, min_drafts + 1, max_drafts + 1);
    if (row_bytes != 0 &&
        static_cast<std::size_t>(max_batch) > std::numeric_limits<std::size_t>::max() / row_bytes) {
        throw std::overflow_error("speculative accept workspace capacity overflows size_t");
    }
    return row_bytes * static_cast<std::size_t>(max_batch);
}

void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions, const Tensor& current_extents,
                                       Tensor& verify_ids, Tensor& positions, cudaStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_inputs";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1) { throw std::invalid_argument("speculative_prepare_verify_inputs: K must be >=1"); }
    if (batch < 1) {
        throw std::invalid_argument("speculative_prepare_verify_inputs: B must be >=1");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_vector(base_positions, DType::I32, batch, op, "base_positions");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, k + 1, batch, op, "verify_ids");
    require_matrix(positions, DType::I32, k + 1, batch, op, "positions");
    detail::speculative_prepare_verify_inputs_launch(
        anchors, drafts, base_positions, current_extents, verify_ids, positions, stream);
}

void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    cudaStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_ids";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1 || batch < 1) {
        throw std::invalid_argument("speculative_prepare_verify_ids: K and B must be positive");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, k + 1, batch, op, "verify_ids");
    detail::speculative_prepare_verify_ids_launch(anchors, drafts, current_extents, verify_ids,
                                                  stream);
}

void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& current_extents,
                                      const Tensor& column_masks, Tensor& lengths, Tensor& anchors,
                                      Tensor& licensed_tokens, Tensor& licensed_counts,
                                      Tensor& accepted, Tensor& accepted_columns,
                                      std::int32_t token_domain, const SamplingConfig* configs,
                                      const Tensor& draft_ids, const Tensor& draft_probs,
                                      WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "speculative_accept_greedy_drafts";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1) { throw std::invalid_argument("speculative_accept_greedy_drafts: K must be >=1"); }
    if (batch < 1) {
        throw std::invalid_argument("speculative_accept_greedy_drafts: B must be >=1");
    }
    require_matrix(target_tokens, DType::I32, k + 1, batch, op, "target_tokens");
    require_dtype(logits, DType::BF16, op, "logits");
    if (logits.ne[0] <= 0 || logits.ne[1] != k + 1 || logits.ne[2] != batch || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: logits must be [physical_rows,K+1,B]");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "speculative_accept_greedy_drafts: token_domain must be in [1, logits.ne[0]]");
    }
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_vector(lengths, DType::I32, batch, op, "lengths");
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(licensed_tokens, DType::I32, k + 1, batch, op, "licensed_tokens");
    require_vector(licensed_counts, DType::I32, batch, op, "licensed_counts");
    require_vector(accepted, DType::I32, batch, op, "accepted");
    // ACCMASK-ONE-SHOT diagnostic (HOST ONLY, read-only; deleting this whole block is the
    // complete revert). WHAT IT MEASURES: the `column_masks` THIS op was actually handed --
    // i.e. the very argument the accept path branches on. launcher/speculative_round.cu:59-61
    // turns `column_masks.data == nullptr` into the kernel's `masks_ptr == nullptr`, and
    // kernel/speculative_round.cuh:142 branches exactly on that. The upstream [treebind] probe
    // (mtp_impl.h:194) prints the SLICE's own `.data`, which is non-null whenever the frame is
    // bound at all, so it is not a binding witness. Two things are printed here:
    //   (a) the pointer and the nullptr-test result -- the world-(A)/(B) discriminator;
    //   (b) the VALUES the device holds at that pointer at this point in the stream, read back
    //       by an ordered D2H on the SAME stream, which is what the kernel's `row_masks[]` reads.
    // No numeric path is touched: nothing here writes device memory, nothing printed is fed back
    // into any tensor, and the only added side effect is an extra stream sync (timing, not values).
    // NOTE: the ordered D2H is incompatible with CUDA graph capture -- run with --no-cuda-graph.
    {
        static int accmask_round = 0;
        const int accmask_bound = column_masks.data != nullptr ? 1 : 0;
        std::fprintf(stderr,
                     "[accmask] round=%d bound=%d masks=%p k=%d batch=%d ne=%d,%d,%d,%d "
                     "nb=%lld,%lld,%lld,%lld\n",
                     accmask_round, accmask_bound, static_cast<const void*>(column_masks.data),
                     static_cast<int>(k), static_cast<int>(batch), column_masks.ne[0],
                     column_masks.ne[1], column_masks.ne[2], column_masks.ne[3],
                     static_cast<long long>(column_masks.nb[0]),
                     static_cast<long long>(column_masks.nb[1]),
                     static_cast<long long>(column_masks.nb[2]),
                     static_cast<long long>(column_masks.nb[3]));
        if (accmask_bound != 0) {
            const std::int32_t accmask_cols = k + 1;
            std::vector<std::uint64_t> accmask_host(
                static_cast<std::size_t>(accmask_cols) * static_cast<std::size_t>(batch),
                std::uint64_t{0});
            const int accmask_sync_rc = static_cast<int>(cudaStreamSynchronize(stream));
            const int accmask_copy_rc = static_cast<int>(cudaMemcpyAsync(
                accmask_host.data(), column_masks.data,
                accmask_host.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
            const int accmask_drain_rc = static_cast<int>(cudaStreamSynchronize(stream));
            std::fprintf(stderr,
                         "[accmask] round=%d sync_rc=%d copy_rc=%d drain_rc=%d\n", accmask_round,
                         accmask_sync_rc, accmask_copy_rc, accmask_drain_rc);
            for (std::int32_t accmask_r = 0; accmask_r < batch; ++accmask_r) {
                std::fprintf(stderr, "[accmask] round=%d row=%d masks:", accmask_round,
                             static_cast<int>(accmask_r));
                for (std::int32_t accmask_c = 0; accmask_c < accmask_cols; ++accmask_c) {
                    std::fprintf(stderr, " %llu",
                                 static_cast<unsigned long long>(
                                     accmask_host[static_cast<std::size_t>(accmask_r) *
                                                      accmask_cols +
                                                  static_cast<std::size_t>(accmask_c)]));
                }
                std::fprintf(stderr, "\n");
            }
        }
        ++accmask_round;
    }
    if (column_masks.data != nullptr) {
        require_matrix(column_masks, DType::I64, k + 1, batch, op, "column_masks");
        // accepted_columns is the round's published COLUMN: every reader of it is tree-gated
        // (speculative_target_impl.h selects the continuation hidden from it under the same
        // `column_masks` test, and ops::mtp_tree_commit_history is reached only from the MTP tree
        // round). A chain round publishes no column, so it may leave the output unbound --
        // dflash/dflash2 build their TargetVerifyFrameView without it and the struct's field
        // defaults to the empty Tensor. Demanding it unconditionally is what rejected every
        // dflash2 arm with "invalid dtype for accepted_columns" before its first decode.
        // The requirement stays INSIDE the mask gate on purpose: a tree round, the only round whose
        // consumer reads the value, still fails loudly if it forgets to bind the output.
        require_vector(accepted_columns, DType::I32, batch, op, "accepted_columns");
    }
    if (configs == nullptr) {
        throw std::invalid_argument("speculative_accept_greedy_drafts: configs must be non-null");
    }
    if (draft_ids.data != nullptr) {
        constexpr std::int32_t kTopK = 16;
        if (draft_ids.dtype != DType::I32 || draft_probs.dtype != DType::FP32 ||
            draft_ids.ne[0] != kTopK || draft_ids.ne[1] != k || draft_ids.ne[2] != batch ||
            draft_probs.ne[0] != kTopK || draft_probs.ne[1] != k || draft_probs.ne[2] != batch ||
            !draft_ids.is_contiguous() || !draft_probs.is_contiguous()) {
            throw std::invalid_argument(
                "speculative_accept_greedy_drafts: draft distribution must be [16,K,B]");
        }
    }
    auto scratch_scope = workspace.scope();
    const std::size_t bytes =
        speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, batch, batch);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::speculative_accept_greedy_drafts_launch(
        target_tokens, logits, drafts, current_extents, column_masks, lengths, anchors,
        licensed_tokens, licensed_counts, accepted, accepted_columns, token_domain, configs,
        draft_ids, draft_probs, scratch, stream);
}

void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        cudaStream_t stream) {
    constexpr const char* op = "speculative_select_accepted_hidden";
    require_dtype(hidden, DType::BF16, op, "hidden");
    if (hidden.ne[0] <= 0 || hidden.ne[1] <= 0 || hidden.ne[2] <= 0 || hidden.ne[3] != 1) {
        throw std::invalid_argument("speculative_select_accepted_hidden: invalid hidden shape");
    }
    require_vector(selectors, DType::I32, hidden.ne[2], op, "selectors");
    require_dtype(out, DType::BF16, op, "out");
    if (out.ne[0] != hidden.ne[0] || out.ne[1] != hidden.ne[2] || out.ne[2] != 1 ||
        out.ne[3] != 1) {
        throw std::invalid_argument("speculative_select_accepted_hidden: invalid output shape");
    }
    detail::speculative_select_accepted_hidden_launch(hidden, selectors, out, stream);
}

void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map, std::int32_t n,
                              cudaStream_t stream) {
    constexpr const char* op = "proposal_remap_token_ids";
    require_dtype(proposal_tokens, DType::I32, op, "proposal_tokens");
    if (proposal_tokens.ne[0] <= 0 || proposal_tokens.ne[1] != 1 || proposal_tokens.ne[2] != 1 ||
        proposal_tokens.ne[3] != 1) {
        throw std::invalid_argument(
            "proposal_remap_token_ids: proposal_tokens must be a non-empty vector");
    }
    if (id_map == nullptr || n <= 0) {
        throw std::invalid_argument("proposal_remap_token_ids: id_map must be non-null and n>0");
    }
    detail::proposal_remap_token_ids_launch(proposal_tokens, id_map, n, stream);
}

} // namespace ninfer::ops
