// E8-tier prefill launches: the packed 4-bit (E8-lattice K / i4 V)
// instantiations of the int8 prefill kernels, split out of
// gqa_attention_prefill.cu so each translation unit's ptxas stage fits the
// host memory budget (the E8 template flag doubles the kernel set).
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <cstdio>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename Metadata>
void attention_e8_for(const Tensor& q, const Tensor& positions, float scale,
                      const PagedKVBatchLayerView& cache, const Metadata& metadata, Tensor& out,
                      cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    const Tensor& cache_k_scale = cache.k_scale_pages;
    const Tensor& cache_v_scale = cache.v_scale_pages;
    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillI8Br)),
                              static_cast<unsigned>(Geometry::QHeads), 1u);
    gqa_attention_prefill_i8_kernel<Geometry, Metadata, true>
        <<<attention_grid, kGqaPrefillI8Threads, kGqaPrefillI8SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int8_t*>(cache_k.data),
            static_cast<const std::int8_t*>(cache_v.data),
            static_cast<const __half*>(cache_k_scale.data),
            static_cast<const __half*>(cache_v_scale.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
}

template <typename Geometry, typename Metadata>
void append_e8_for(const Tensor& k, const Tensor& v, const Tensor& positions,
                   const PagedKVBatchLayerView& cache, const Metadata& metadata,
                   cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    const Tensor& cache_k_scale = cache.k_scale_pages;
    const Tensor& cache_v_scale = cache.v_scale_pages;
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    constexpr int kFillBlock = 256;
    if (tokens >= 128 && Geometry::KVHeads == 2) {
        constexpr int kPageBlock     = 256;
        constexpr int kTokensPerTile = 8;
        const int max_tiles          = div_up(tokens + kTokensPerTile - 1, kTokensPerTile);
        const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                             static_cast<unsigned>(Geometry::KVHeads),
                             static_cast<unsigned>(kGqaKvQuantGroups));
        gqa_attention_prefill_fill_i8_page_kernel<Geometry, Metadata, true>
            <<<fill_grid, kPageBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data),
                static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data), tokens);
    } else {
        constexpr int kFillWarps = kFillBlock / 32;
        const std::int64_t fill_units =
            static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaKvQuantGroups;
        const int fill_grid =
            static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillWarps)));
        gqa_attention_prefill_fill_i8_kernel<Geometry, Metadata, true>
            <<<fill_grid, kFillBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data),
                static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gqa_attention_prefill_e8_launch(const Tensor& q, const Tensor& positions,
                                     const std::int32_t* valid_columns,
                                     const std::int32_t* table_rows, std::int32_t table_stride,
                                     float scale, PagedKVBatchLayerView cache, Tensor& out,
                                     cudaStream_t stream) {
    const bool masked = valid_columns != nullptr;
    const auto launch = [&]<bool Masked>() {
        const GqaPrefillBatchMetadata<Masked> metadata{
            .tables        = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns = Masked ? valid_columns : nullptr,
            .table_rows    = table_rows,
            .table_stride  = table_stride,
        };
        if (q.ne[1] == Gqa27Geometry::QHeads) {
            attention_e8_for<Gqa27Geometry>(q, positions, scale, cache, metadata, out, stream);
            return;
        }
        attention_e8_for<Gqa35Geometry>(q, positions, scale, cache, metadata, out, stream);
    };
    if (masked) {
        launch.template operator()<true>();
    } else {
        launch.template operator()<false>();
    }
}

void gqa_kv_append_e8_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                             const std::int32_t* valid_columns, const std::int32_t* table_rows,
                             std::int32_t table_stride, PagedKVBatchLayerView cache,
                             cudaStream_t stream) {
    const bool masked = valid_columns != nullptr;
    const auto launch = [&]<bool Masked>() {
        const GqaPrefillBatchMetadata<Masked> metadata{
            .tables        = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns = Masked ? valid_columns : nullptr,
            .table_rows    = table_rows,
            .table_stride  = table_stride,
        };
        if (k.ne[1] == Gqa27Geometry::KVHeads) {
            append_e8_for<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
            return;
        }
        append_e8_for<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
    };
    if (masked) {
        launch.template operator()<true>();
    } else {
        launch.template operator()<false>();
    }
}

void gqa_kv_append_e8_launch_single(const Tensor& k, const Tensor& v, const Tensor& positions,
                                     const PagedKVLayerView& cache, cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    const PagedKVBatchLayerView batch_view{
        .k_pages       = cache.k_pages,
        .v_pages       = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages,
        .v_scale_pages = cache.v_scale_pages,
        .k_residual_pages = cache.k_residual_pages,
        .k_residual_scale_pages = cache.k_residual_scale_pages,
        .v_residual_pages = cache.v_residual_pages,
        .v_residual_scale_pages = cache.v_residual_scale_pages,
        .block_tables  = cache.block_table.view({cache.block_table.ne[0], 1}),
        .cold_slots    = cache.cold_slots,
        .cold_slot_valid = cache.cold_slot_valid,
        .slot_bytes    = cache.slot_bytes,
        .head_dim      = cache.head_dim,
        .num_kv_heads  = cache.num_kv_heads,
        .layer_index   = cache.layer_index,
        .dtype         = cache.dtype,
        .quant_group   = cache.quant_group,
        .v_dtype       = cache.v_dtype,
        .v_quant_group = cache.v_quant_group,
        .sliding_window_tokens = cache.sliding_window_tokens,
    };
    if (k.ne[1] == Gqa27Geometry::KVHeads) {
        append_e8_for<Gqa27Geometry>(k, v, positions, batch_view, metadata, stream);
        return;
    }
    append_e8_for<Gqa35Geometry>(k, v, positions, batch_view, metadata, stream);
}

void gqa_attention_prefill_e8_launch_single(const Tensor& q, const Tensor& positions,
                                            float scale, const PagedKVLayerView& cache,
                                            Tensor& out, cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    const PagedKVBatchLayerView batch_view{
        .k_pages       = cache.k_pages,
        .v_pages       = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages,
        .v_scale_pages = cache.v_scale_pages,
        .k_residual_pages = cache.k_residual_pages,
        .k_residual_scale_pages = cache.k_residual_scale_pages,
        .v_residual_pages = cache.v_residual_pages,
        .v_residual_scale_pages = cache.v_residual_scale_pages,
        .block_tables  = cache.block_table.view({cache.block_table.ne[0], 1}),
        .cold_slots    = cache.cold_slots,
        .cold_slot_valid = cache.cold_slot_valid,
        .slot_bytes    = cache.slot_bytes,
        .head_dim      = cache.head_dim,
        .num_kv_heads  = cache.num_kv_heads,
        .layer_index   = cache.layer_index,
        .dtype         = cache.dtype,
        .quant_group   = cache.quant_group,
        .v_dtype       = cache.v_dtype,
        .v_quant_group = cache.v_quant_group,
        .sliding_window_tokens = cache.sliding_window_tokens,
    };
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        attention_e8_for<Gqa27Geometry>(q, positions, scale, batch_view, metadata, out, stream);
        return;
    }
    attention_e8_for<Gqa35Geometry>(q, positions, scale, batch_view, metadata, out, stream);
}

} // namespace ninfer::ops::detail
