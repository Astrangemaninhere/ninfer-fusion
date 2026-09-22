#include "ninfer/ops/mtp_round.h"
#include "ops/launcher/mtp_round.h"

#include "ops/kernel/paged_kv_address.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

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

void require_matrix(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                    const char* op, const char* name) {
    if (t.dtype != dtype || rows <= 0 || cols <= 0 || t.ne[0] != rows || t.ne[1] != cols ||
        t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid matrix shape for " + name);
    }
    require_contiguous_nonnull(t, op, name);
}

void require_vector(const Tensor& t, DType dtype, std::int32_t count, const char* op,
                    const char* name) {
    require_matrix(t, dtype, count, 1, op, name);
}

void require_row_pitched_matrix(const Tensor& t, std::int32_t rows, std::int32_t cols,
                                const char* op, const char* name) {
    constexpr std::int64_t element = sizeof(std::int32_t);
    if (rows <= 0 || cols <= 0 || t.dtype != DType::I32 || t.ne[0] != rows || t.ne[1] != cols ||
        t.ne[2] != 1 || t.ne[3] != 1 || t.nb[0] != element || t.nb[1] < rows * element ||
        t.nb[1] % element != 0 || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": invalid row-pitched matrix for " + name);
    }
}

} // namespace

void mtp_draft_align_hidden(const Tensor& hidden, const Tensor& chain_sources,
                            const Tensor& valid_counts, Tensor& out, cudaStream_t stream) {
    constexpr const char* op    = "mtp_draft_align_hidden";
    const std::int32_t head_dim = hidden.ne[0];
    const std::int32_t width    = hidden.ne[1];
    const std::int32_t batch    = hidden.ne[2];
    if (hidden.dtype != DType::BF16 || head_dim <= 0 || width < 2 ||
        width > kMtpTreeMaximumWidth || batch < 1 || hidden.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid hidden tensor");
    }
    require_contiguous_nonnull(hidden, op, "hidden");
    require_matrix(chain_sources, DType::I32, width, batch, op, "chain_sources");
    require_vector(valid_counts, DType::I32, batch, op, "valid_counts");
    if (out.dtype != DType::BF16 || out.ne[0] != head_dim || out.ne[1] != width ||
        out.ne[2] != batch || out.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid out tensor");
    }
    require_contiguous_nonnull(out, op, "out");
    if (out.data == hidden.data) {
        throw std::invalid_argument(std::string(op) + ": out must not alias hidden");
    }
    detail::mtp_draft_align_hidden_launch(hidden, chain_sources, valid_counts, out, stream);
}

void mtp_svip_entropy_extents(const Tensor& logits, const Tensor& accepted, Tensor& cuts,
                              float threshold, cudaStream_t stream) {
    constexpr const char* op = "mtp_svip_entropy_extents";
    if (!(logits.dtype == DType::BF16 && logits.ne[3] == 1 && logits.is_contiguous() &&
          logits.data != nullptr)) {
        throw std::invalid_argument(std::string(op) + ": invalid logits tensor");
    }
    const int batch = logits.ne[2];
    if (batch < 1) { throw std::invalid_argument(std::string(op) + ": empty batch"); }
    require_vector(accepted, DType::I32, batch, op, "accepted");
    require_vector(cuts, DType::I32, batch, op, "cuts");
    if (!(threshold > 0.0F)) {
        throw std::invalid_argument(std::string(op) + ": threshold must be positive");
    }
    detail::mtp_svip_entropy_extents_launch(logits, accepted, cuts, threshold, stream);
}

void mtp_adaptive_extents(const Tensor& accepted, const Tensor& current_extents, Tensor& cuts,
                          std::int32_t k_max, cudaStream_t stream) {
    constexpr const char* op = "mtp_adaptive_extents";
    const int batch = accepted.ne[0];
    if (batch < 1) { throw std::invalid_argument(std::string(op) + ": empty batch"); }
    require_vector(accepted, DType::I32, batch, op, "accepted");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_vector(cuts, DType::I32, batch, op, "cuts");
    if (k_max < 1 || k_max > 16) {
        throw std::invalid_argument(std::string(op) + ": k_max must be in [1,16]");
    }
    detail::mtp_adaptive_extents_launch(accepted, current_extents, cuts, k_max, stream);
}

void mtp_prepare_next_round(const Tensor& verify_ids, const Tensor& next_anchors,
                            const Tensor& accepted, const Tensor& updated_frontiers,
                            const Tensor& remaining_budgets, const Tensor& licensed_counts,
                            const Tensor& rope_deltas, Tensor& alignment_ids, Tensor& next_extents,
                            Tensor& ar_positions, Tensor& ar_rope_positions,
                            Tensor& ar_valid_columns, std::int32_t max_context,
                            cudaStream_t stream, const Tensor* svip_cuts) {
    constexpr const char* op = "mtp_prepare_next_round";
    const std::int32_t T     = verify_ids.ne[0];
    const std::int32_t batch = verify_ids.ne[1];
    // T domain widened 6 -> 16. The kernel is a plain grid-stride loop with no
    // per-T template specialization and the launcher sizes its grid from T, so the
    // old [2,6] bound only asserted the previously validated domain (see the
    // launcher header). Frame, AR-envelope array and width checks are all sized for
    // 16 now. Widen T is validated by measurement: k=9 (T=10) keeps a healthy
    // per-position profile and AL grows monotonically.
    if (T < 2 || T > 16) {
        throw std::invalid_argument("mtp_prepare_next_round: T must be in [2,16]");
    }
    if (batch < 1) { throw std::invalid_argument("mtp_prepare_next_round: B must be positive"); }
    if (max_context <= 0) {
        throw std::invalid_argument("mtp_prepare_next_round: max_context must be positive");
    }
    require_matrix(verify_ids, DType::I32, T, batch, op, "verify_ids");
    require_vector(next_anchors, DType::I32, batch, op, "next_anchors");
    require_vector(accepted, DType::I32, batch, op, "accepted");
    require_vector(updated_frontiers, DType::I32, batch, op, "updated_frontiers");
    require_vector(remaining_budgets, DType::I32, batch, op, "remaining_budgets");
    require_vector(licensed_counts, DType::I32, batch, op, "licensed_counts");
    require_vector(rope_deltas, DType::I32, batch, op, "rope_deltas");
    require_matrix(alignment_ids, DType::I32, T, batch, op, "alignment_ids");
    require_vector(next_extents, DType::I32, batch, op, "next_extents");
    const std::int32_t steps = std::max(T - 2, 1);
    require_row_pitched_matrix(ar_positions, batch, steps, op, "ar_positions");
    require_row_pitched_matrix(ar_rope_positions, batch, steps, op, "ar_rope_positions");
    require_row_pitched_matrix(ar_valid_columns, batch, steps, op, "ar_valid_columns");
    if (ar_rope_positions.nb[1] != ar_positions.nb[1] ||
        ar_valid_columns.nb[1] != ar_positions.nb[1]) {
        throw std::invalid_argument(
            "mtp_prepare_next_round: AR outputs must share one step stride");
    }
    if (svip_cuts != nullptr) {
        require_vector(*svip_cuts, DType::I32, batch, op, "svip_cuts");
    }
    detail::mtp_prepare_next_round_launch(verify_ids, next_anchors, accepted, updated_frontiers,
                                          remaining_budgets, licensed_counts, rope_deltas,
                                          alignment_ids, next_extents, ar_positions,
                                          ar_rope_positions, ar_valid_columns, max_context, stream,
                                          svip_cuts);
}

void mtp_tree_commit_history(const Tensor& column_masks, const Tensor& column_depths,
                             const Tensor& accepted_columns, const Tensor& base_frontiers,
                             const Tensor& table_rows, PagedKVBatchLayerView cache,
                             Tensor& chain_sources, Tensor& commit_flags, cudaStream_t stream) {
    constexpr const char* op = "mtp_tree_commit_history";
    const std::int32_t width = column_masks.ne[0];
    const std::int32_t batch = column_masks.ne[1];
    if (width < 2 || width > kMtpTreeMaximumWidth || batch < 1 || batch > 8) {
        throw std::invalid_argument(std::string(op) + ": unsupported W/B domain");
    }
    require_matrix(column_masks, DType::I64, width, batch, op, "column_masks");
    require_matrix(column_depths, DType::I32, width, batch, op, "column_depths");
    require_matrix(chain_sources, DType::I32, width, batch, op, "chain_sources");
    require_vector(accepted_columns, DType::I32, batch, op, "accepted_columns");
    require_vector(base_frontiers, DType::I32, batch, op, "base_frontiers");
    require_vector(table_rows, DType::I32, batch, op, "table_rows");
    require_vector(commit_flags, DType::I32, batch, op, "commit_flags");
    // The masks are consumed by the bf16 attention routes only -- the small-T decode and, at
    // B = 1, the prompt body (src/ops/wrapper/gqa_attention.cpp refuses a non-BF16 cache, and
    // gqa_attention_prompt_launch refuses the route shapes it cannot index) -- so a commit has no
    // k_scale/v_scale notion to move and this Op takes the same gate: a quantized tier is refused
    // by NAME here, not committed approximately.
    if (cache.dtype != DType::BF16) {
        throw std::invalid_argument(
            std::string(op) + ": only a BF16 text KV cache can be committed (the tree verify's "
                              "column masks are implemented for the bf16 small-T route)");
    }
    if (cache.k_pages.data == nullptr || cache.v_pages.data == nullptr ||
        cache.block_tables.data == nullptr || cache.k_pages.dtype != DType::BF16 ||
        cache.v_pages.dtype != DType::BF16 || cache.block_tables.dtype != DType::I32 ||
        cache.block_tables.ne[1] < batch || cache.head_dim <= 0 || cache.num_kv_heads <= 0) {
        throw std::invalid_argument(std::string(op) + ": invalid BF16 batch cache view");
    }
    const bool registered_geometry =
        (cache.head_dim == 256 && (cache.num_kv_heads == 4 || cache.num_kv_heads == 2)) ||
        (cache.head_dim == 128 && cache.num_kv_heads == 2);
    if (!registered_geometry) {
        throw std::invalid_argument(std::string(op) + ": unsupported Q/KV head geometry");
    }
    const std::int64_t capacity =
        static_cast<std::int64_t>(cache.block_tables.ne[0]) * kPagedKVPageSize;

    detail::MtpTreeCommitInvocation invocation;
    invocation.column_masks     = static_cast<const std::uint64_t*>(column_masks.data);
    invocation.column_depths    = static_cast<const std::int32_t*>(column_depths.data);
    invocation.accepted_columns = static_cast<const std::int32_t*>(accepted_columns.data);
    invocation.base_frontiers   = static_cast<const std::int32_t*>(base_frontiers.data);
    invocation.table_rows       = static_cast<const std::int32_t*>(table_rows.data);
    invocation.block_tables     = static_cast<const std::int32_t*>(cache.block_tables.data);
    invocation.table_stride     = cache.block_tables.ne[0];
    invocation.chain_sources    = static_cast<std::int32_t*>(chain_sources.data);
    invocation.commit_flags     = static_cast<std::int32_t*>(commit_flags.data);
    invocation.cache_k          = cache.k_pages.data;
    invocation.cache_v          = cache.v_pages.data;
    invocation.head_dim         = cache.head_dim;
    invocation.kv_heads         = cache.num_kv_heads;
    invocation.width            = width;
    invocation.batch            = batch;
    invocation.logical_capacity = static_cast<std::int32_t>(capacity);
    detail::mtp_tree_commit_history_launch(invocation, stream);
}

} // namespace ninfer::ops
