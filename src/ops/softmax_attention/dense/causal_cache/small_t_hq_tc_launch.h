// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : cometkim/ninfer
// Branch      : feat/1m-context
// Commit      : 2cc56b5db39f951c1e91b2caec8b39274d795682
// Source path : src/ops/softmax_attention/dense/causal_cache/small_t_hq_tc_launch.h
// sha256(src) : 3bf0065ca1ce8e86588b967527d6d1a47785387a64921e41f1222f79cc3e3e28
// Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits. The 2026-09-18 fork-survey merge landed this file with `e8` renamed to `rk4v4`; INTEGRATE2 REVERTED that rename, on three measured grounds. (1) It is FALSE about this codec's own geometry: the codec is 2-bit K against 4-bit V -- the `-2b` token survived the rewrite while the codec name did not, and its type is still `Rk4v4Packed2BitTile`. (2) It collides with the LIVE tier vocabulary that NAMEFIX3 established in src/kvcfg/kv_formats.h (`Rk4v4` / `Rk3v4` / `Rk2v4` as TIERS, 13 tracked files), so one name came to mean two things -- MERGE flag C2 instantiated in code. (3) The revert is free of build consequence: no build reads either name. The body below is now byte-identical to the pinned `sha256(src)`.
#pragma once

// ninfer::ops::detail - hq-e8-2b small-T launch ownership. Two tensor-core instantiations per
// geometry cover single-token decode and the runtime-width speculative tile.
// The codec-heavy kernels stay in the per-geometry instantiation translation units.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/kv_cache/append/hq_kernel.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_hq.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {


template <typename Geometry>
constexpr int hq_token_tile() {
    return Geometry::QHeads == 24 ? 8 : 6;
}

template <typename Geometry, typename CacheInput, bool MultiBatch, bool Masked, bool Narrow>
void launch_hq_partial(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                       PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                       std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                       Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int Warps  = 4;
    constexpr int Tokens = Narrow ? 1 : hq_token_tile<Geometry>();
    const dim3 grid(static_cast<unsigned>(Geometry::KVHeads), static_cast<unsigned>(splits),
                    MultiBatch ? static_cast<unsigned>(invocation.batch_size) : 1u);
    causal_attention_small_t_tc_partial_hq_kernel<Geometry, Tokens, Warps, MultiBatch, Masked,
                                                  CacheInput, Narrow>
        <<<grid, Warps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), input,
            static_cast<const std::int32_t*>(pos.data),
            static_cast<std::uint8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data),
            static_cast<std::uint8_t*>(cache.k_scale_pages.data),
            static_cast<std::uint8_t*>(cache.v_scale_pages.data),
            static_cast<__nv_bfloat16*>(cache.residual_k.data),
            static_cast<__nv_bfloat16*>(cache.residual_v.data),
            static_cast<std::uint32_t*>(cache.side_words.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            invocation.valid_columns
                ? static_cast<const std::int32_t*>(invocation.valid_columns->data)
                : nullptr,
            invocation.table_rows ? static_cast<const std::int32_t*>(invocation.table_rows->data)
                                  : nullptr,
            cache.block_tables.ne[0], invocation.width, invocation.full_width,
            invocation.column_begin, logical_capacity, scale, static_cast<float*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_hq_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                            float scale, PagedKVBatchLayerView cache,
                                            const CausalSmallTInvocation& invocation,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits           = causal_attention_split_capacity(
        Geometry::QHeads, invocation.width, cache.storage, envelope, invocation.batch_size);
    const bool narrow         = invocation.width == 1 && logical_capacity > 1024;
    const auto launch_partial = [&]<bool MultiBatch, bool Masked>() {
        if (narrow) {
            launch_hq_partial<Geometry, CacheInput, MultiBatch, Masked, true>(
                q, input, pos, scale, cache, invocation, logical_capacity, splits, partial_acc,
                partial_m, partial_l, stream);
        } else {
            launch_hq_partial<Geometry, CacheInput, MultiBatch, Masked, false>(
                q, input, pos, scale, cache, invocation, logical_capacity, splits, partial_acc,
                partial_m, partial_l, stream);
        }
    };
    const bool masked = invocation.valid_columns != nullptr;
    if (invocation.batch_size == 1) {
        if (masked) {
            launch_partial.template operator()<false, true>();
        } else {
            launch_partial.template operator()<false, false>();
        }
    } else if (masked) {
        launch_partial.template operator()<true, true>();
    } else {
        launch_partial.template operator()<true, false>();
    }

    // Partials are un-rotated into the original frame inside the partial kernel, so the shared
    // BF16-profile reducer combines them exactly like the other cache dtypes.
    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = Geometry::QHeads == 24 ? 256 : 64;
    const auto launch_reduce   = [&]<bool MultiBatch, bool Masked, bool Offset, bool Narrow>() {
        const dim3 grid(Geometry::QHeads, div_up(kCausalHeadDim, kDChunk),
                          invocation.width * invocation.batch_size);
        causal_attention_small_t_reduce_output_kernel<Geometry, kDChunk, false, MultiBatch, Masked,
                                                        Offset, Narrow>
            <<<grid, kReduceBlock, 0, stream>>>(
                static_cast<const float*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data),
                invocation.valid_columns
                      ? static_cast<const std::int32_t*>(invocation.valid_columns->data)
                      : nullptr,
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    };
    const auto launch_profile = [&]<bool MultiBatch, bool Masked>() {
        if (narrow) {
            if (invocation.column_begin == 0)
                launch_reduce.template operator()<MultiBatch, Masked, false, true>();
            else
                launch_reduce.template operator()<MultiBatch, Masked, true, true>();
        } else {
            if (invocation.column_begin == 0)
                launch_reduce.template operator()<MultiBatch, Masked, false, false>();
            else
                launch_reduce.template operator()<MultiBatch, Masked, true, false>();
        }
    };
    if (invocation.batch_size == 1) {
        if (masked) {
            launch_profile.template operator()<false, true>();
        } else {
            launch_profile.template operator()<false, false>();
        }
    } else if (masked) {
        launch_profile.template operator()<true, true>();
    } else {
        launch_profile.template operator()<true, false>();
    }
}

extern template void causal_attention_small_t_hq_launch_for<CausalD256H24Kv4, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

extern template void causal_attention_small_t_hq_launch_for<CausalD256H24Kv4, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

extern template void causal_attention_small_t_hq_launch_for<CausalD256H16Kv2, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

extern template void causal_attention_small_t_hq_launch_for<CausalD256H16Kv2, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
