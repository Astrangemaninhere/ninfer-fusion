// Implements: include/ninfer/ops/mtp_round.h
// Match: validated request-major K=1..5 MTP round transition.
#include "ops/launcher/mtp_round.h"

#include "core/device.h"
#include "ops/kernel/mtp_round.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

namespace {

// The registered GQA geometries whose KV cache a tree verify can reach (the wrapper rejects every
// other pair by name before this point): 24 q-heads/4 KV, 16/2, 16/4 and 32/2 -- see
// src/ops/wrapper/gqa_attention.cpp registered_kv_heads().
template <int HeadDim, int KVHeads>
void mtp_tree_commit_geometry(const MtpTreeCommitInvocation& inv, cudaStream_t stream) {
    constexpr int kBlock = 128;
    mtp_tree_commit_history_kernel<HeadDim, KVHeads><<<inv.batch, kBlock, 0, stream>>>(
        inv.column_masks, inv.column_depths, inv.accepted_columns, inv.base_frontiers,
        inv.table_rows, inv.block_tables, inv.table_stride, inv.chain_sources, inv.commit_flags,
        static_cast<__nv_bfloat16*>(inv.cache_k), static_cast<__nv_bfloat16*>(inv.cache_v),
        inv.width, inv.batch, inv.logical_capacity);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void mtp_tree_commit_history_launch(const MtpTreeCommitInvocation& invocation, cudaStream_t stream) {
    if (invocation.head_dim == 256 && invocation.kv_heads == 4) {
        mtp_tree_commit_geometry<256, 4>(invocation, stream);
        return;
    }
    if (invocation.head_dim == 256 && invocation.kv_heads == 2) {
        mtp_tree_commit_geometry<256, 2>(invocation, stream);
        return;
    }
    if (invocation.head_dim == 128 && invocation.kv_heads == 2) {
        mtp_tree_commit_geometry<128, 2>(invocation, stream);
        return;
    }
    // Unreachable through the wrapper; refuse to guess rather than index a cache this launcher has
    // no page layout for.
    throw std::invalid_argument(
        "mtp_tree_commit_history: unsupported Q/KV head geometry for a tree KV commit");
}

void mtp_draft_align_hidden_launch(const Tensor& hidden, const Tensor& chain_sources,
                                   const Tensor& valid_counts, Tensor& out, cudaStream_t stream) {
    const std::int32_t head_dim = hidden.ne[0];
    const std::int32_t width    = hidden.ne[1];
    const std::int32_t batch    = hidden.ne[2];
    // One block per (batch row, depth) column; the D elements of a column are contiguous.
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned int>(batch), static_cast<unsigned int>(width));
    mtp_draft_align_hidden_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::uint16_t*>(hidden.data),
        static_cast<const std::int32_t*>(chain_sources.data),
        static_cast<const std::int32_t*>(valid_counts.data),
        static_cast<std::uint16_t*>(out.data), width, batch, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_svip_entropy_extents_launch(const Tensor& logits, const Tensor& accepted,
                                       Tensor& cuts, float threshold, cudaStream_t stream) {
    const int vocab = logits.ne[0];
    const int cols  = logits.ne[1];
    const int batch = logits.ne[2];
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned int>(batch));
    mtp_svip_entropy_extents_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(accepted.data),
        static_cast<std::int32_t*>(cuts.data), vocab, cols, batch, threshold);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_adaptive_extents_launch(const Tensor& accepted, const Tensor& current_extents,
                                 Tensor& cuts, std::int32_t k_max, cudaStream_t stream) {
    const int batch = accepted.ne[0];
    constexpr int kBlock = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBlock - 1) / kBlock));
    mtp_adaptive_extents_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(accepted.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(cuts.data), k_max, batch);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_prepare_next_round_launch(const Tensor& verify_ids, const Tensor& next_anchors,
                                   const Tensor& accepted, const Tensor& updated_frontiers,
                                   const Tensor& remaining_budgets, const Tensor& licensed_counts,
                                   const Tensor& rope_deltas, Tensor& alignment_ids,
                                   Tensor& next_extents, Tensor& ar_positions,
                                   Tensor& ar_rope_positions, Tensor& ar_valid_columns,
                                   std::int32_t max_context, cudaStream_t stream,
                                   const Tensor* svip_cuts) {
    constexpr int kBlock = 32;
    const int k          = verify_ids.ne[0] - 1;
    const int batch      = verify_ids.ne[1];
    const int ar_step_stride =
        static_cast<int>(ar_positions.nb[1] / static_cast<std::int64_t>(sizeof(std::int32_t)));
    const dim3 grid(static_cast<unsigned int>((k + kBlock) / kBlock),
                    static_cast<unsigned int>(batch));
    mtp_prepare_next_round_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(verify_ids.data),
        static_cast<const std::int32_t*>(next_anchors.data),
        static_cast<const std::int32_t*>(accepted.data),
        static_cast<const std::int32_t*>(updated_frontiers.data),
        static_cast<const std::int32_t*>(remaining_budgets.data),
        static_cast<const std::int32_t*>(licensed_counts.data),
        static_cast<const std::int32_t*>(rope_deltas.data),
        static_cast<std::int32_t*>(alignment_ids.data),
        static_cast<std::int32_t*>(next_extents.data),
        static_cast<std::int32_t*>(ar_positions.data),
        static_cast<std::int32_t*>(ar_rope_positions.data),
        static_cast<std::int32_t*>(ar_valid_columns.data), k, ar_step_stride, max_context,
        svip_cuts != nullptr ? static_cast<const std::int32_t*>(svip_cuts->data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
