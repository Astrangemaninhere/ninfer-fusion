#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// One layer's BF16 K/V planes plus the tree metadata of the round being committed. The launcher
// dispatches on (head_dim, kv_heads) so the cache offset stays the engine's own
// paged_kv_element_offset<> (ops/kernel/paged_kv_address.cuh:63-69) instead of a second copy of
// the page layout. `cache_k`/`cache_v` are the raw plane pointers of one batch layer view.
struct MtpTreeCommitInvocation {
    // The mask word is addressed as an UNSIGNED 64-bit bit set (the same reinterpretation the bf16
    // attention launcher makes): DType::I64 storage, uint64_t semantics.
    const std::uint64_t* column_masks    = nullptr;
    const std::int32_t* column_depths   = nullptr;
    const std::int32_t* accepted_columns = nullptr;
    const std::int32_t* base_frontiers  = nullptr;
    const std::int32_t* table_rows      = nullptr;
    const std::int32_t* block_tables    = nullptr;
    std::int32_t table_stride           = 0;
    std::int32_t* chain_sources         = nullptr;
    std::int32_t* commit_flags          = nullptr;
    void* cache_k                       = nullptr;
    void* cache_v                       = nullptr;
    std::int32_t head_dim               = 0;
    std::int32_t kv_heads               = 0;
    std::int32_t width                  = 0;
    std::int32_t batch                  = 0;
    std::int32_t logical_capacity       = 0;
};

void mtp_tree_commit_history_launch(const MtpTreeCommitInvocation& invocation, cudaStream_t stream);

void mtp_svip_entropy_extents_launch(const Tensor& logits, const Tensor& accepted,
                                       Tensor& cuts, float threshold, cudaStream_t stream);

void mtp_adaptive_extents_launch(const Tensor& accepted, const Tensor& current_extents,
                                 Tensor& cuts, std::int32_t k_max, cudaStream_t stream);

// One depth-ordered draft-alignment gather. No geometry template: the Op touches no paged cache,
// so it has no page layout to dispatch on and no HeadDim/KVHeads pair to select.
void mtp_draft_align_hidden_launch(const Tensor& hidden, const Tensor& chain_sources,
                                   const Tensor& valid_counts, Tensor& out, cudaStream_t stream);

void mtp_prepare_next_round_launch(const Tensor& verify_ids, const Tensor& next_anchors,
                                   const Tensor& accepted, const Tensor& updated_frontiers,
                                   const Tensor& remaining_budgets, const Tensor& licensed_counts,
                                   const Tensor& rope_deltas, Tensor& alignment_ids,
                                   Tensor& next_extents, Tensor& ar_positions,
                                   Tensor& ar_rope_positions, Tensor& ar_valid_columns,
                                   std::int32_t max_context, cudaStream_t stream,
                                   const Tensor* svip_cuts);

} // namespace ninfer::ops::detail
