// ninfer::ops - split-KV GQA small-T launcher and unified route dispatcher.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_decode_bf16.cuh"
#include "ops/kernel/gqa_attention_decode_fp8.cuh"
#include "ops/kernel/gqa_attention_decode_iso3.cuh"
#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "ops/kernel/gqa_attention_decode_nvfp4.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/gqa_attention.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
// Defined in gqa_attention_decode_e8.cu: the E8-tier (packed 4-bit) decode
// instantiations live in their own TU so this file's ptxas stage stays small.
void gqa_attention_decode_e8_launch(const Tensor& q, const GqaAppendInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t implementation_window, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream);
void gqa_attention_decode_e8_launch(const Tensor& q, const GqaCachedInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t implementation_window, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream);
namespace {

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval. The dtype-aware wrapper below adds the measured INT8 specializations.
template <typename Geometry>
std::int32_t gqa_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::DecodeSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::DecodeSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::DecodeSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::DecodeSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::DecodeSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::DecodeSplitScale); }

    return (splits < Geometry::DecodeSplits) ? splits : Geometry::DecodeSplits;
}

template <typename Geometry>
std::int32_t gqa_small_t_split_count(std::int32_t window, std::int32_t tokens, DType kv_dtype) {
    // A 64-key default split just above a 32-key boundary makes the partial
    // kernel execute a nearly empty second tile. These short ranges instead
    // launch one 32-key tile per split; the larger CTAs keep the small grid busy.
    if ((kv_dtype == DType::I8 || kv_dtype == DType::E8Kv) && tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::DecodeSplitScale);
    }
    if ((kv_dtype == DType::I8 || kv_dtype == DType::E8Kv) && tokens == 6 && window > 128 && window <= 160) {
        return div_up(window, 24 / Geometry::DecodeSplitScale);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if ((kv_dtype == DType::I8 || kv_dtype == DType::E8Kv) && tokens == 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::DecodeSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::DecodeSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::DecodeSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    // NVFP4 first revision: coarser splits than INT8 to cut redundant Q
    // quantization and split-reduction overhead. Long contexts still scale.
    if (kv_dtype == DType::NVFP4) {
        const std::int32_t target =
            window > 16390 ? 480 / Geometry::DecodeSplitScale
                           : (window > 4096 ? 256 / Geometry::DecodeSplitScale
                                            : 64 / Geometry::DecodeSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::DecodeSplitScale;
        std::int32_t splits         = div_up(window, target);
        splits                      = splits > kMin ? splits : kMin;
        return splits < Geometry::DecodeSplits ? splits : Geometry::DecodeSplits;
    }
    // BF16, FP8_E4M3FN, and ISO3 share the generic split policy.
    return gqa_small_t_split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t gqa_small_t_launch_capacity(GqaExecutionEnvelope envelope, std::int32_t tokens,
                                         DType dtype) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits =
            gqa_small_t_split_count<Geometry>(static_cast<std::int32_t>(window), tokens, dtype);
        capacity = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k = cache.k_pages;
    Tensor& cache_v = cache.v_pages;
    // bf16 kernel uses only static smem (no dynamic staging).
    gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                 Masked, CacheInput><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), input,
        static_cast<const std::int32_t*>(pos.data), static_cast<__nv_bfloat16*>(cache_k.data),
        static_cast<__nv_bfloat16*>(cache_v.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
        invocation.table_rows == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.table_rows->data),
        cache.block_tables.ne[0], invocation.width, invocation.full_width, invocation.column_begin,
        logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_fp8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                           PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                           std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                           Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    // fp8 kernel uses only static smem (no dynamic staging), same as bf16.
    gqa_attention_small_t_tc_partial_fp8_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                Masked, CacheInput><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), input,
        static_cast<const std::int32_t*>(pos.data), static_cast<std::uint8_t*>(cache_k.data),
        static_cast<std::uint8_t*>(cache_v.data),
        static_cast<std::uint8_t*>(cache_k_scale.data),
        static_cast<std::uint8_t*>(cache_v_scale.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
        invocation.table_rows == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.table_rows->data),
        cache.block_tables.ne[0], invocation.width, invocation.full_width, invocation.column_begin,
        logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput, bool Nvfp4K = false>
void launch_tc_partial_iso3(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    // iso3 kernel uses only static smem (no dynamic staging), same as bf16.
    gqa_attention_small_t_tc_partial_iso3_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                 Masked, CacheInput, Nvfp4K>
        <<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), input,
        static_cast<const std::int32_t*>(pos.data), static_cast<std::uint8_t*>(cache_k.data),
        static_cast<std::uint8_t*>(cache_v.data),
        static_cast<std::uint8_t*>(cache_k_scale.data),
        static_cast<std::uint8_t*>(cache_v_scale.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
        invocation.table_rows == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.table_rows->data),
        cache.block_tables.ne[0], invocation.width, invocation.full_width, invocation.column_begin,
        logical_capacity, static_cast<int>(cache.sliding_window_tokens), scale,
        static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, bool E8,
          typename CacheInput>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t implementation_window,
                          std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                          Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    // Revision 2b: INT8-tier cold slots (raw nibble codec). The kernel takes
    // region-relative K/V slot bases; empty tensors disable the cold branch.
    const std::uint8_t* cold_k_i8 =
        cache.cold_slots.data != nullptr && cache.dtype == DType::I8
            ? static_cast<const std::uint8_t*>(cache.cold_slots.data)
            : nullptr;
    const std::uint8_t* cold_v_i8 =
        cold_k_i8 != nullptr && cache.cold_slots.nb[2] != 0
            ? cold_k_i8 + cache.cold_slots.nb[2]
            : nullptr;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kGqaHeadDim) : 0u;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena,
                                                     MultiBatch, Masked, E8, CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                             MinBlocksPerSm, KeyBlock, DynamicArena, MultiBatch,
                                             Masked, E8, CacheInput>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data), cold_k_i8, cold_v_i8,
                cache.slot_bytes,
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    };
    // Revision 2b: INT8-tier cold slots (raw nibble codec). The kernel takes
    // region-relative K/V slot bases; empty tensors disable the cold branch.
    (void)cold_v_i8;
    if constexpr (TokenTile == 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, bool Iso3V = false>
void launch_tc_partial_nvfp4(const Tensor& q, const __nv_bfloat16* input_k,
                             const __nv_bfloat16* input_v, bool writes_cache, const Tensor& pos,
                             float scale, PagedKVBatchLayerView cache,
                             const GqaSmallTInvocation& invocation,
                             std::int32_t logical_capacity, std::int32_t implementation_window,
                             std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                             Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    const bool masked     = invocation.valid_columns != nullptr;
    const std::uint8_t* cold_k = static_cast<const std::uint8_t*>(cache.cold_slots.data);
    const std::uint8_t* cold_v =
        cold_k == nullptr ? nullptr : cold_k + cache.cold_slots.nb[2];
    const std::int32_t* cold_k_valid =
        static_cast<const std::int32_t*>(cache.cold_slot_valid.data);
    const std::int32_t* cold_v_valid =
        cold_k_valid == nullptr
            ? nullptr
            : reinterpret_cast<const std::int32_t*>(
                  reinterpret_cast<const std::uint8_t*>(cold_k_valid) +
                  cache.cold_slot_valid.nb[1]);
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        // Matches the arena layout in gqa_attention_decode_nvfp4_tiled_kernel:
        // two ping-pong tiles (k_pk/v_pk Bc*128 each + k_sf/v_sf Bc*16 each),
        // psc_s (Br*64 bytes, 64-byte row stride over RowTiles*16 rows),
        // repack_a/repack_b (Wc*16*64 each).
        constexpr int kTileBytes = 4 * KeyBlock * 128 + 4 * KeyBlock * 16;
        constexpr int kRowTiles  = (TokenTile * Geometry::GroupSize + 15) / 16;
        constexpr std::size_t kRBytes =
            static_cast<std::size_t>(2 * kTileBytes + kRowTiles * 16 * 64 +
                                     2 * WarpsPerCta * 16 * 64);
        constexpr std::size_t kVDynamicBytes =
            Iso3V ? static_cast<std::size_t>(KeyBlock) * 256ULL * 2ULL : 0ULL;
        constexpr std::size_t kDynamicBytes =
            (DynamicArena ? kRBytes : 0ULL) + kVDynamicBytes;
        if constexpr (DynamicArena || Iso3V) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                gqa_attention_decode_nvfp4_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                        MinBlocksPerSm, KeyBlock, DynamicArena,
                                                        Iso3V>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        gqa_attention_decode_nvfp4_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                                KeyBlock, DynamicArena, Iso3V>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input_k, input_v,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::uint8_t*>(cache_k.data),
                static_cast<std::uint8_t*>(cache_v.data),
                static_cast<std::uint8_t*>(cache_k_scale.data),
                static_cast<std::uint8_t*>(cache_v_scale.data),
                static_cast<std::uint8_t*>(cache.k_residual_pages.data),
                static_cast<std::uint8_t*>(cache.k_residual_scale_pages.data),
                static_cast<std::uint8_t*>(cache.v_residual_pages.data),
                static_cast<std::uint8_t*>(cache.v_residual_scale_pages.data),
                cold_k, cold_v, cold_k_valid, cold_v_valid, cache.slot_bytes,
                static_cast<int>(cache.sliding_window_tokens),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, cache.layer_index, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                invocation.batch_size, masked, writes_cache);
    };
    // Minimal production schedule set for the first NVFP4 revision.
    if constexpr (Geometry::GroupSize == 6) {
        if constexpr (TokenTile <= 4) {
            launch.template operator()<16, 1, 32, true>();
        } else if constexpr (TokenTile == 5) {
            launch.template operator()<8, 1, 32, true>();
        } else {
            launch.template operator()<12, 1, 32, true>();
        }
    } else {
        if constexpr (TokenTile <= 4) {
            launch.template operator()<16, 1, 32, true>();
        } else {
            launch.template operator()<12, 1, 32, true>();
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

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

bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

std::int32_t gqa_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 6 ||
        (cache_dtype != DType::BF16 && cache_dtype != DType::I8 &&
         cache_dtype != DType::NVFP4 && cache_dtype != DType::FP8_E4M3FN &&
         cache_dtype != DType::ISO3 && cache_dtype != DType::E8Kv) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("gqa_attention split capacity: invalid profile");
    }
    if (q_heads == Gqa27Geometry::QHeads) {
        return gqa_small_t_launch_capacity<Gqa27Geometry>(envelope, tokens, cache_dtype);
    }
    if (q_heads == Gqa35Geometry::QHeads) {
        return gqa_small_t_launch_capacity<Gqa35Geometry>(envelope, tokens, cache_dtype);
    }
    throw std::invalid_argument("gqa_attention split capacity: unsupported head geometry");
}

template <typename Geometry, typename CacheInput>
void gqa_attention_small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                      float scale, PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                      Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                      cudaStream_t stream) {
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
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
                launch_tc_partial_nvfp4<Geometry, (TOKENS), true>(                                \
                    q, nvfp4_k, nvfp4_v, CacheInput::writes_cache, pos, scale, cache, invocation, \
                    logical_capacity, implementation_window, splits, partial_acc, partial_m,      \
                    partial_l, stream);                                                            \
            } else if (cache.dtype == DType::NVFP4) {                                              \
                const __nv_bfloat16* nvfp4_k = nullptr;                                            \
                const __nv_bfloat16* nvfp4_v = nullptr;                                            \
                if constexpr (CacheInput::writes_cache) {                                          \
                    nvfp4_k = input.k;                                                             \
                    nvfp4_v = input.v;                                                             \
                }                                                                                  \
                launch_tc_partial_nvfp4<Geometry, (TOKENS)>(                                      \
                    q, nvfp4_k, nvfp4_v, CacheInput::writes_cache, pos, scale, cache, invocation, \
                    logical_capacity, implementation_window, splits, partial_acc, partial_m,      \
                    partial_l, stream);                                                            \
            } else if (cache.dtype == DType::FP8_E4M3FN) {                                        \
                launch_tc_partial_fp8<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(           \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits,             \
                    partial_acc, partial_m, partial_l, stream);                                    \
            } else if (cache.dtype == DType::ISO3) {                                               \
                launch_tc_partial_iso3<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(          \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits,             \
                    partial_acc, partial_m, partial_l, stream);                                    \
            } else {                                                                               \
                launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(           \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits,             \
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
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, static_cast<__nv_bfloat16*>(out.data));
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
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation,
                                                        envelope, partial_acc, partial_m, partial_l,
                                                        out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation,
                                                    envelope, partial_acc, partial_m, partial_l,
                                                    out, stream);
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
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, batch_cache,
                                                        invocation, envelope, partial_acc,
                                                        partial_m, partial_l, out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, batch_cache, invocation,
                                                    envelope, partial_acc, partial_m, partial_l,
                                                    out, stream);
}

} // namespace ninfer::ops::detail
