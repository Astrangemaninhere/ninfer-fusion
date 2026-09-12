// ninfer::ops::detail - small-T GQA launch path (kernel instantiations).
// Relocated verbatim from gqa_attention_decode.cu so the instantiation
// cascade, and its ptxas stage, land in their own translation unit.
#include "ops/launcher/gqa_attention.h"
#include "ops/launcher/gqa_attention_decode_partial.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

namespace {

PagedKVBatchLayerView single_row_batch_view(const PagedKVLayerView& cache) {
    return {
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
}

} // namespace

template <typename Geometry, typename CacheInput>
void gqa_attention_small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                      float scale, PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                      Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                      cudaStream_t stream) {
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    // Split reference: a constant of the produced execution graph (the sequence key
    // capacity). Batch-1 decode and the MTP/DFlash verify of the same request carry the
    // same value, so a verify column reduces its keys in the same order as the decode
    // row it replaces. It also replaces the live window in the I8 tile schedule, which
    // is the second way a launch width used to reach the exact result.
    const auto split_reference       = gqa_small_t_split_reference(envelope);
    const auto implementation_window = split_reference;
    const auto split_units           = gqa_small_t_split_units<Geometry>(split_reference);
    const auto splits =
        gqa_small_t_launch_capacity<Geometry>(envelope, invocation.width, cache.dtype);

    // BF16 and FP8 keep the row-tile warp count; INT8 selects its
    // producer/consumer geometry inside launch_tc_partial_i8.
#define NINFER_GQA_SMALL_T_DISPATCH(TOKENS, WARPS)                                                 \
    do {                                                                                           \
        const auto launch_profile = [&]<bool MultiBatch, bool Masked>() {                          \
            if (cache.dtype == DType::I8) {                                                        \
                launch_tc_partial_i8<Geometry, (TOKENS), MultiBatch, Masked, false>(               \
                    q, input, pos, scale, cache, invocation, logical_capacity,                     \
                    implementation_window, splits, partial_acc, partial_m, partial_l, stream);     \
            } else if (cache.dtype == DType::E8Kv) {                                               \
                gqa_attention_decode_e8_launch(q, input, pos, scale, cache, invocation,           \
                                               logical_capacity, implementation_window, splits,   \
                                               partial_acc, partial_m, partial_l, stream);        \
            } else if (cache.dtype == DType::NVFP4 && cache.v_dtype == DType::ISO3) {              \
                const __nv_bfloat16* nvfp4_k = nullptr;                                            \
                const __nv_bfloat16* nvfp4_v = nullptr;                                            \
                if constexpr (CacheInput::writes_cache) {                                          \
                    nvfp4_k = input.k;                                                             \
                    nvfp4_v = input.v;                                                             \
                }                                                                                  \
                if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) { \
                launch_tc_partial_nvfp4<Geometry, (TOKENS), true>(                                \
                    q, nvfp4_k, nvfp4_v, CacheInput::writes_cache, pos, scale, cache, invocation, \
                    logical_capacity, implementation_window, splits, partial_acc, partial_m,      \
                    partial_l, stream);                                                            \
                } else { \
                    (void)nvfp4_k; (void)nvfp4_v; \
                    require_nvfp4_geometry_dim(Geometry::HeadDim); \
                } \
            } else if (cache.dtype == DType::NVFP4) {                                              \
                const __nv_bfloat16* nvfp4_k = nullptr;                                            \
                const __nv_bfloat16* nvfp4_v = nullptr;                                            \
                if constexpr (CacheInput::writes_cache) {                                          \
                    nvfp4_k = input.k;                                                             \
                    nvfp4_v = input.v;                                                             \
                }                                                                                  \
                if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) { \
                launch_tc_partial_nvfp4<Geometry, (TOKENS)>(                                      \
                    q, nvfp4_k, nvfp4_v, CacheInput::writes_cache, pos, scale, cache, invocation, \
                    logical_capacity, implementation_window, splits, partial_acc, partial_m,      \
                    partial_l, stream);                                                            \
                } else { \
                    (void)nvfp4_k; (void)nvfp4_v; \
                    require_nvfp4_geometry_dim(Geometry::HeadDim); \
                } \
            } else if (cache.dtype == DType::FP8_E4M3FN) {                                        \
                launch_tc_partial_fp8<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(           \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits, split_units,             \
                    partial_acc, partial_m, partial_l, stream);                                    \
            } else if (cache.dtype == DType::ISO3) {                                               \
                launch_tc_partial_iso3<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(          \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits, split_units,             \
                    partial_acc, partial_m, partial_l, stream);                                    \
            } else {                                                                               \
                launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(           \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits, split_units,             \
                    partial_acc, partial_m, partial_l, stream);                                    \
            }                                                                                      \
        };                                                                                         \
        const bool masked = invocation.valid_columns != nullptr;                                   \
        if (invocation.batch_size == 1) {                                                          \
            if (masked) {                                                                          \
                launch_profile.template operator()<false, true>();                                 \
            } else {                                                                               \
                launch_profile.template operator()<false, false>();                                \
            }                                                                                      \
        } else if (masked) {                                                                       \
            launch_profile.template operator()<true, true>();                                      \
        } else {                                                                                   \
            launch_profile.template operator()<true, false>();                                     \
        }                                                                                          \
    } while (0)

    switch (invocation.width) {
    case 1:
        NINFER_GQA_SMALL_T_DISPATCH(1, 2);
        break;
    case 2:
        NINFER_GQA_SMALL_T_DISPATCH(2, 4);
        break;
    case 3:
        NINFER_GQA_SMALL_T_DISPATCH(3, 4);
        break;
    case 4:
        NINFER_GQA_SMALL_T_DISPATCH(4, 4);
        break;
    case 5:
        NINFER_GQA_SMALL_T_DISPATCH(5, 4);
        break;
    case 6:
        NINFER_GQA_SMALL_T_DISPATCH(6, 4);
        break;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }
#undef NINFER_GQA_SMALL_T_DISPATCH

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const dim3 reduce_grid(Geometry::QHeads, div_up(kGqaHeadDim, kDChunk),
                           invocation.width * invocation.batch_size);
    const auto launch_reduce = [&]<bool Int8, bool MultiBatch, bool Masked, bool Offset>() {
        gqa_attention_small_t_reduce_output_kernel<Geometry, kDChunk, Int8, MultiBatch, Masked,
                                                   Offset>
            <<<reduce_grid, kReduceBlock, 0, stream>>>(
                static_cast<const float*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, split_units,
                static_cast<__nv_bfloat16*>(out.data));
    };
    const bool masked         = invocation.valid_columns != nullptr;
    const auto launch_profile = [&]<bool Int8, bool MultiBatch, bool Masked>() {
        if (invocation.column_begin == 0) {
            launch_reduce.template operator()<Int8, MultiBatch, Masked, false>();
        } else {
            launch_reduce.template operator()<Int8, MultiBatch, Masked, true>();
        }
    };
    const auto launch_for_dtype = [&]<bool Int8>() {
        if (invocation.batch_size == 1) {
            if (masked) {
                launch_profile.template operator()<Int8, false, true>();
            } else {
                launch_profile.template operator()<Int8, false, false>();
            }
        } else if (masked) {
            launch_profile.template operator()<Int8, true, true>();
        } else {
            launch_profile.template operator()<Int8, true, false>();
        }
    };
    // FP8_E4M3FN and ISO3 are quantized but deliberately use the BF16
    // (Int8=false) reducer path: gqa_small_t_split_count falls through to the
    // generic BF16 policy for both dtypes, and their partial kernels compute
    // active splits with gqa_small_t_active_splits<Geometry,false>. The
    // Int8=true path would apply the I8 token-5/6 active-split specializations
    // and disagree with the launch.
    if (cache.dtype == DType::I8 || cache.dtype == DType::NVFP4) {
        launch_for_dtype.template operator()<true>();
    } else {
        launch_for_dtype.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& pos, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale,
                                  PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                                  std::int32_t column_begin, std::int32_t width,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream) {
    const GqaAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                               static_cast<const __nv_bfloat16*>(v.data)};
    const GqaSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation,
                                                        envelope, partial_acc, partial_m, partial_l,
                                                        out, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        gqa_attention_small_t_launch_for<GqaMuseGeometry>(q, input, pos, scale, cache, invocation,
                                                          envelope, partial_acc, partial_m,
                                                          partial_l, out, stream);
        return;
    }
    if (q.ne[1] == Gqa35Geometry::QHeads && q.ne[0] == Gqa35Geometry::HeadDim) {
        gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation,
                                                        envelope, partial_acc, partial_m,
                                                        partial_l, out, stream);
        return;
    }
    throw std::invalid_argument(
        "gqa_attention_small_t_launch: unsupported query-head geometry (" +
        std::to_string(q.ne[1]) + " q-heads); registered: " +
        std::to_string(Gqa27Geometry::QHeads) + "/" +
        std::to_string(GqaMuseGeometry::QHeads) + "/" +
        std::to_string(Gqa35Geometry::QHeads));
}

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream) {
    const GqaCachedInput input{};
    const GqaSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    const PagedKVBatchLayerView batch_cache = single_row_batch_view(cache);
    if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, batch_cache,
                                                        invocation, envelope, partial_acc,
                                                        partial_m, partial_l, out, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        gqa_attention_small_t_launch_for<GqaMuseGeometry>(q, input, pos, scale, batch_cache,
                                                          invocation, envelope, partial_acc,
                                                          partial_m, partial_l, out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, batch_cache, invocation,
                                                    envelope, partial_acc, partial_m, partial_l,
                                                    out, stream);
}

} // namespace ninfer::ops::detail
