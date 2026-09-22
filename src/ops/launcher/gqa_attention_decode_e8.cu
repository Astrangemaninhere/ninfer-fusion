// E8-tier decode launch: the packed 4-bit (E8-lattice K / i4 V) instantiations
// of the int8 decode kernel, split out of gqa_attention_decode.cu so each
// translation unit's ptxas stage fits the host memory budget (the E8 template
// flag doubles the kernel instantiation set).
#include "ops/launcher/gqa_attention.h"

#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/gqa_attention.h"
// d256_kv_cache_is_e8_family(): the one place the ops layer names the e8 family. Host-only,
// no device code, so it adds nothing to this TU's ptxas budget.
#include "ops/kv_cache/d256_profile.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// FIX-A instrument. gqa_attention_decode_partial.cuh holds the canonical copy; this TU must not
// include that header (it is the tier header the per-codec TUs include, and e8.cu is one of
// them), so the two env readers and the one log line are repeated here verbatim.
inline bool e8_splitdbg_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_SPLITDBG");
        return env != nullptr && *env != '\0' && !(env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

// Same length gate as the canonical copy in gqa_attention_decode_partial.cuh; see there for
// the rationale (NINFER_SPLITDBG_MAX, default 512 lines, 0 disables the instrument).
inline std::size_t e8_splitdbg_cap() {
    static const std::size_t cap = [] {
        const char* env = std::getenv("NINFER_SPLITDBG_MAX");
        if (env == nullptr || *env == '\0') { return std::size_t{512}; }
        const long value = std::strtol(env, nullptr, 10);
        return value <= 0 ? std::size_t{0} : static_cast<std::size_t>(value);
    }();
    return cap;
}

inline bool e8_splitdbg_allow() {
    if (!e8_splitdbg_enabled()) { return false; }
    static std::atomic<std::size_t> used{0};
    return used.fetch_add(1) < e8_splitdbg_cap();
}

inline bool e8_verify_exact() {
    static const bool strict = [] {
        const char* env = std::getenv("NINFER_VERIFY_EXACT");
        if (env == nullptr || *env == '\0') { return true; }
        return !(env[0] == '0' && env[1] == '\0');
    }();
    return strict;
}

inline const char* e8_splitdbg_route(const GqaSmallTInvocation& invocation) {
    return (invocation.column_begin != 0 || invocation.width < invocation.full_width)
               ? "chunked_small_t"
               : "small_t";
}

template <typename Geometry>
inline void e8_splitdbg_tile(const GqaSmallTInvocation& invocation, DType dtype, int token_tile,
                             int warps_per_cta, int min_blocks_per_sm, int key_block,
                             std::int32_t logical_capacity, std::int32_t splits,
                             std::int32_t split_units) {
    if (!e8_splitdbg_allow()) { return; }
    std::fprintf(stderr,
                 "[splitdbg] tile tier=e8 route=%s dtype=%d width=%d full_width=%d "
                 "column_begin=%d batch=%d masked=%d token_tile=%d warps_per_cta=%d "
                 "min_blocks_per_sm=%d key_block=%d logical_capacity=%d splits=%d "
                 "split_units=%d "
                 "q_heads=%d kv_heads=%d group_size=%d head_dim=%d\n",
                 e8_splitdbg_route(invocation), static_cast<int>(dtype), invocation.width,
                 invocation.full_width, invocation.column_begin, invocation.batch_size,
                 invocation.valid_columns != nullptr ? 1 : 0, token_tile, warps_per_cta,
                 min_blocks_per_sm, key_block, logical_capacity, splits, split_units,
                 Geometry::QHeads,
                 Geometry::KVHeads, Geometry::GroupSize, Geometry::HeadDim);
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_i8_e8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                             PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                             std::int32_t logical_capacity, std::int32_t split_reference,
                             std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                             Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    constexpr bool E8 = true;
    // split_reference carries the launch's PINNED split reference (the sequence key capacity),
    // never the live window: it is the only value the fixed split grid may be derived from, so
    // that split s owns the same keys in a batch-1 decode and in the wide MTP verify of the
    // same row. gqa_attention_decode_smallt.cu:60-62 is the only producer, and
    // NINFER_VERIFY_EXACT=1 refuses a launch whose envelope leaves it unpinned.
    const auto split_reference_units = gqa_small_t_split_units<Geometry>(split_reference);
    const int  split_units           = split_reference_units;
    // BF16-COLD-LAND E4: E8 tiers DO have a cold-slot codec now (decoder_state.cpp
    // cold_slot_codec_of(E8Kv) == Int8Raw; entropy_cold_requant's E8KvG64 arm copies
    // the packed 4-bit codes verbatim into the same raw record the int8 tier uses), so
    // the cold pointers are handed over exactly as the int8 launcher hands them over:
    // plane-relative, K = base, V = base + nb[2].
    // ⚠ AND FOR THE NARROW TIERS THE DROP IS NOW LOUD. `cold_slot_codec_of(E8Kv) == Int8Raw`
    // (see the note above) is what makes the hand-over below correct, and it holds for the
    // SHIPPED 4-bit tier only: the cold record is decoded at the 4-bit stride and the requant
    // arm copies "the packed 4-bit codes verbatim". For E8K3Kv / E8K2Kv the ternary below
    // takes its `: nullptr` branch and the cold-slot plane is DROPPED WITH NO DIAGNOSTIC --
    // a wrong answer (retrieval from the hot plane only), not a refusal. Refusing here is the
    // honest direction: it is loud, and it is reachable only for a tier the rest of the tree
    // already refuses, so no shipped path changes.
    if (cache.cold_slots.data != nullptr && d256_kv_cache_is_e8_family(cache.dtype) &&
        cache.dtype != DType::E8Kv) {
        throw std::invalid_argument(
            "gqa_attention_decode_e8: cold slots are present but this tier has no cold-slot "
            "codec -- cold_slot_codec_of maps E8Kv to Int8Raw at the 4-bit record stride "
            "only; refusing rather than silently decoding the cold plane at the wrong width");
    }
    const std::uint8_t* cold_k_i8 =
        cache.cold_slots.data != nullptr && cache.dtype == DType::E8Kv
            ? static_cast<const std::uint8_t*>(cache.cold_slots.data)
            : nullptr;
    const std::uint8_t* cold_v_i8 =
        cold_k_i8 != nullptr && cache.cold_slots.nb[2] != 0
            ? cold_k_i8 + cache.cold_slots.nb[2]
            : nullptr;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        e8_splitdbg_tile<Geometry>(invocation, cache.dtype, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                   KeyBlock, logical_capacity, splits, split_units);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * Geometry::HeadDim) : 0u;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena,
                                                     MultiBatch, Masked, E8, CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                             KeyBlock, DynamicArena, MultiBatch, Masked, E8,
                                             CacheInput>
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
                logical_capacity, split_units, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    };
    if constexpr (Geometry::GroupSize == 4) {
        require_group4_schedule("e8");
    } else if constexpr (Geometry::GroupSize == 16) {
        // Muse (32q/2kv): RowTiles == TokenTile, so the generic schedule table's
        // Wc values violate Wc % RowTiles == 0. Same rule as the nvfp4 and i8
        // Muse branches: Wc keeps Wc % TokenTile == 0 with PVNt in {4,8}.
        constexpr int kWc = TokenTile == 1 ? 4 : TokenTile == 2 ? 8 : TokenTile == 3 ? 6
                          : TokenTile == 4 ? 8 : TokenTile == 5 ? 10 : 12;
        launch.template operator()<kWc, 1, 32, false>();
    } else if constexpr (TokenTile == 6) {
        if (split_reference > 128 && split_reference <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (split_reference <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (split_reference <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            if (split_reference > 128 && split_reference <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (split_reference <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            if (split_reference > 128 && split_reference <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (split_reference <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (split_reference <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (split_reference <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheInput>
void launch_e8_for(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                   PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                   std::int32_t logical_capacity, std::int32_t split_reference,
                   std::int32_t splits, Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                   cudaStream_t stream) {
    const bool masked = invocation.valid_columns != nullptr;
    const auto dispatch = [&]<bool MultiBatch, bool Masked>() {
        switch (invocation.width) {
        case 1:
            launch_tc_partial_i8_e8<Geometry, 1, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, split_reference,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        case 2:
            launch_tc_partial_i8_e8<Geometry, 2, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, split_reference,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        case 3:
            launch_tc_partial_i8_e8<Geometry, 3, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, split_reference,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        case 4:
            launch_tc_partial_i8_e8<Geometry, 4, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, split_reference,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        case 5:
            launch_tc_partial_i8_e8<Geometry, 5, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, split_reference,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        case 6:
            launch_tc_partial_i8_e8<Geometry, 6, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, split_reference,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        default:
            throw std::invalid_argument("E8 decode launch: unsupported T");
        }
    };
    if (invocation.batch_size == 1) {
        if (masked) {
            dispatch.template operator()<false, true>();
        } else {
            dispatch.template operator()<false, false>();
        }
    } else if (masked) {
        dispatch.template operator()<true, true>();
    } else {
        dispatch.template operator()<true, false>();
    }
}

} // namespace

void gqa_attention_decode_e8_launch(const Tensor& q, const GqaAppendInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t split_reference, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        launch_e8_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                     split_reference, splits, partial_acc, partial_m,
                                     partial_l, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        launch_e8_for<GqaMuseGeometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                       split_reference, splits, partial_acc, partial_m,
                                       partial_l, stream);
        return;
    }
    if (q.ne[1] == Gqa16x4Geometry::QHeads && q.ne[0] == Gqa16x4Geometry::HeadDim &&
        cache.num_kv_heads == Gqa16x4Geometry::KVHeads) {
        require_group4_schedule("e8");
    }
    if (q.ne[1] != Gqa35Geometry::QHeads || q.ne[0] != Gqa35Geometry::HeadDim) {
        throw std::invalid_argument("E8 decode launch: unsupported query-head geometry");
    }
    launch_e8_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                 split_reference, splits, partial_acc, partial_m, partial_l,
                                 stream);
}

void gqa_attention_decode_e8_launch(const Tensor& q, const GqaCachedInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t split_reference, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        launch_e8_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                     split_reference, splits, partial_acc, partial_m,
                                     partial_l, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        launch_e8_for<GqaMuseGeometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                       split_reference, splits, partial_acc, partial_m,
                                       partial_l, stream);
        return;
    }
    if (q.ne[1] == Gqa16x4Geometry::QHeads && q.ne[0] == Gqa16x4Geometry::HeadDim &&
        cache.num_kv_heads == Gqa16x4Geometry::KVHeads) {
        require_group4_schedule("e8");
    }
    if (q.ne[1] != Gqa35Geometry::QHeads || q.ne[0] != Gqa35Geometry::HeadDim) {
        throw std::invalid_argument("E8 decode launch (batch): unsupported query-head geometry");
    }
    launch_e8_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                 split_reference, splits, partial_acc, partial_m, partial_l,
                                 stream);
}

} // namespace ninfer::ops::detail
