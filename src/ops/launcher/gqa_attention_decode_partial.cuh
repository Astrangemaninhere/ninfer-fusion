#pragma once

// Shared small-T GQA partial-launch machinery, relocated verbatim from
// gqa_attention_decode.cu: launch_tc_partial_{bf16,fp8,iso3,i8,nvfp4}, the
// split-policy helpers they share with the retained routing TU, the nvfp4
// geometry guard, and the E8-tier launch declarations. Both
// gqa_attention_decode.cu and gqa_attention_decode_smallt.cu include this
// header so every shared entity has exactly one definition in the program.

#include "ops/launcher/gqa_attention.h"

#include "ops/common/ft_stats.h"
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
#include <string>

namespace ninfer::ops::detail {

// nvfp4 tiled decode is 256-only: its QK/QV tiling assumes four 64-wide tiles
// (static_assert(QKKs == 4) in gqa_attention_decode_nvfp4.cuh). Muse's head_dim
// is 128, where that tiling does not exist — before this guard the kernel was
// instantiated against a hardcoded 256 and wrote/read outside the K row into the
// V/scale planes (_TODO.md 126/127). Refuse loudly for other geometries; bf16 and
// int8 KV decode do work at 128.
[[noreturn]] inline void require_nvfp4_geometry_dim(int head_dim) {
    throw std::invalid_argument(
        "nvfp4 KV decode requires head_dim=256 but this geometry has " +
        std::to_string(head_dim) +
        "; use bf16 or int8 KV (the nvfp4 tiled kernel is not ported to 128)");
}

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
        logical_capacity, scale, static_cast<float*>(partial_acc.data),
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
        logical_capacity, scale, static_cast<float*>(partial_acc.data),
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
        static_cast<float*>(partial_acc.data),
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
                logical_capacity, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    };
    // Revision 2b: INT8-tier cold slots (raw nibble codec). The kernel takes
    // region-relative K/V slot bases; empty tensors disable the cold branch.
    if constexpr (Geometry::GroupSize == 16) {
        // Muse (32q/2kv): RowCount = TokenTile*16 -> RowTiles = TokenTile.
        // Wc keeps Wc % TokenTile == 0 and PVNt = 16/(Wc/TokenTile) in {4,8}:
        // TT1->4, TT2->8, TT3->6, TT4->8, TT5->10, TT6->12.
        constexpr int kWc =
            TokenTile == 1 ? 4 : TokenTile == 2 ? 8 : TokenTile == 3 ? 6
            : TokenTile == 4 ? 8 : TokenTile == 5 ? 10 : 12;
        launch.template operator()<kWc, 1, 32, false>();
    } else if constexpr (TokenTile == 6) {
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
        // repack_a/repack_b (Wc*16*64 each, native-PV path only — absent when
        // Iso3V so TT6 stays inside the sm_120 per-block opt-in limit).
        constexpr int kTileBytes = 4 * KeyBlock * 128 + 4 * KeyBlock * 16;
        constexpr int kRowTiles  = (TokenTile * Geometry::GroupSize + 15) / 16;
        constexpr std::size_t kRepackBytes =
            Iso3V ? 0ULL : static_cast<std::size_t>(2 * WarpsPerCta * 16 * 64);
        constexpr std::size_t kRBytes =
            static_cast<std::size_t>(2 * kTileBytes + kRowTiles * 16 * 64) + kRepackBytes;
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
                logical_capacity, cache.layer_index, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                invocation.batch_size, masked, writes_cache);
    };
    // Minimal production schedule set for the first NVFP4 revision.
    if constexpr (Geometry::GroupSize == 16) {
        // Muse: RowTiles = TokenTile (16 rows/token); Wc table keeps
        // Wc % TokenTile == 0 with PVNt in {4,8}.
        constexpr int kWc =
            TokenTile == 1 ? 4 : TokenTile == 2 ? 8 : TokenTile == 3 ? 6
            : TokenTile == 4 ? 8 : TokenTile == 5 ? 10 : 12;
        launch.template operator()<kWc, 1, 32, true>();
    } else if constexpr (Geometry::GroupSize == 6) {
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
    if (ft::enabled()) {
        // FreeToken step-1: sample this layer's partial_l AFTER the reducer wrote it.
        // The probe used to sit before the launches, so it read a workspace buffer the
        // current round had not filled yet (garbage/NaN every round, _TODO.md 94).
        // Host-side D2H; observation runs use --no-cuda-graph.
        ft::observe(stream, cache.layer_index, static_cast<const float*>(partial_l.data),
                    Geometry::QHeads * invocation.width, splits);
    }
}

} // namespace

} // namespace ninfer::ops::detail
