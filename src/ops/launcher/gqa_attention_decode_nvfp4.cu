// Cut B: nvfp4-tier decode launch, split out of gqa_attention_decode_smallt.cu (same
// pattern as gqa_attention_decode_e8.cu: every nvfp4 kernel instantiation for the
// 256-dim geometries lives in this TU, so a single nvcc invocation's ptxas stage stays
// small). The width ladder and the v_dtype==ISO3 (Iso3V) selection are exactly what the
// NINFER_GQA_SMALL_T_DISPATCH macro expanded to before the split, and the
// head_dim==256 guard below is the live guard, relocated with them. The dispatcher used
// to split DType::NVFP4 on cache.v_dtype itself; that re-keying now happens here, which
// is the same accept set with one dispatch site fewer.
#include "ops/launcher/gqa_attention_decode_partial.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheInput, int TokenTile, bool Iso3V>
void launch_nvfp4_tile(const Tensor& q, const __nv_bfloat16* nvfp4_k,
                       const __nv_bfloat16* nvfp4_v, const Tensor& pos, float scale,
                       PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                       std::int32_t logical_capacity, std::int32_t implementation_window,
                       std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                       Tensor& partial_l, cudaStream_t stream) {
    if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) {
        launch_tc_partial_nvfp4<Geometry, TokenTile, Iso3V>(
            q, nvfp4_k, nvfp4_v, CacheInput::writes_cache, pos, scale, cache, invocation,
            logical_capacity, implementation_window, splits, partial_acc, partial_m, partial_l,
            stream);
    } else {
        // Relocated verbatim from the live dispatch: the nvfp4 tiled kernel is
        // 256-only (static_assert(QKKs == 4) in gqa_attention_decode_nvfp4.cuh, where
        // QKKs = Geometry::HeadDim / 64). Muse's head_dim is 128, where that tiling
        // does not exist -- instantiating it there ran the kernel against a hardcoded
        // 256 and wrote/read outside the K row into the V/scale planes (_TODO.md
        // 126/127). Refuse loudly; bf16 and int8 KV decode do work at 128.
        (void)q;
        (void)nvfp4_k;
        (void)nvfp4_v;
        (void)pos;
        (void)scale;
        (void)cache;
        (void)invocation;
        (void)logical_capacity;
        (void)implementation_window;
        (void)splits;
        (void)partial_acc;
        (void)partial_m;
        (void)partial_l;
        (void)stream;
        require_nvfp4_geometry_dim(Geometry::HeadDim);
    }
}

template <typename Geometry, typename CacheInput, bool Iso3V>
void launch_nvfp4_ladder(const Tensor& q, const __nv_bfloat16* nvfp4_k,
                         const __nv_bfloat16* nvfp4_v, const Tensor& pos, float scale,
                         PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                         std::int32_t logical_capacity, std::int32_t implementation_window,
                         std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                         Tensor& partial_l, cudaStream_t stream) {
    switch (invocation.width) {
    case 1:
        launch_nvfp4_tile<Geometry, CacheInput, 1, Iso3V>(
            q, nvfp4_k, nvfp4_v, pos, scale, cache, invocation, logical_capacity,
            implementation_window, splits, partial_acc, partial_m, partial_l, stream);
        return;
    case 2:
        launch_nvfp4_tile<Geometry, CacheInput, 2, Iso3V>(
            q, nvfp4_k, nvfp4_v, pos, scale, cache, invocation, logical_capacity,
            implementation_window, splits, partial_acc, partial_m, partial_l, stream);
        return;
    case 3:
        launch_nvfp4_tile<Geometry, CacheInput, 3, Iso3V>(
            q, nvfp4_k, nvfp4_v, pos, scale, cache, invocation, logical_capacity,
            implementation_window, splits, partial_acc, partial_m, partial_l, stream);
        return;
    case 4:
        launch_nvfp4_tile<Geometry, CacheInput, 4, Iso3V>(
            q, nvfp4_k, nvfp4_v, pos, scale, cache, invocation, logical_capacity,
            implementation_window, splits, partial_acc, partial_m, partial_l, stream);
        return;
    case 5:
        launch_nvfp4_tile<Geometry, CacheInput, 5, Iso3V>(
            q, nvfp4_k, nvfp4_v, pos, scale, cache, invocation, logical_capacity,
            implementation_window, splits, partial_acc, partial_m, partial_l, stream);
        return;
    case 6:
        launch_nvfp4_tile<Geometry, CacheInput, 6, Iso3V>(
            q, nvfp4_k, nvfp4_v, pos, scale, cache, invocation, logical_capacity,
            implementation_window, splits, partial_acc, partial_m, partial_l, stream);
        return;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }
}

template <typename Geometry, typename CacheInput>
void launch_nvfp4_for(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                      PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                      std::int32_t logical_capacity, std::int32_t implementation_window,
                      std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                      Tensor& partial_l, cudaStream_t stream) {
    const __nv_bfloat16* nvfp4_k = nullptr;
    const __nv_bfloat16* nvfp4_v = nullptr;
    if constexpr (CacheInput::writes_cache) {
        nvfp4_k = input.k;
        nvfp4_v = input.v;
    }
    if (cache.v_dtype == DType::ISO3) {
        launch_nvfp4_ladder<Geometry, CacheInput, true>(
            q, nvfp4_k, nvfp4_v, pos, scale, cache, invocation, logical_capacity,
            implementation_window, splits, partial_acc, partial_m, partial_l, stream);
    } else {
        launch_nvfp4_ladder<Geometry, CacheInput, false>(
            q, nvfp4_k, nvfp4_v, pos, scale, cache, invocation, logical_capacity,
            implementation_window, splits, partial_acc, partial_m, partial_l, stream);
    }
}

} // namespace

void gqa_attention_decode_nvfp4_launch(const Tensor& q, const GqaAppendInput& input,
                                       const Tensor& pos, float scale,
                                       PagedKVBatchLayerView cache,
                                       const GqaSmallTInvocation& invocation,
                                       std::int32_t logical_capacity,
                                       std::int32_t implementation_window, std::int32_t splits,
                                       Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                       cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
        launch_nvfp4_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                        implementation_window, splits, partial_acc, partial_m,
                                        partial_l, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        launch_nvfp4_for<GqaMuseGeometry>(q, input, pos, scale, cache, invocation,
                                          logical_capacity, implementation_window, splits,
                                          partial_acc, partial_m, partial_l, stream);
        return;
    }
    if (q.ne[1] != Gqa35Geometry::QHeads || q.ne[0] != Gqa35Geometry::HeadDim) {
        throw std::invalid_argument(
            "nvfp4 decode launch (append): unsupported query-head geometry");
    }
    launch_nvfp4_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                    implementation_window, splits, partial_acc, partial_m,
                                    partial_l, stream);
}

void gqa_attention_decode_nvfp4_launch(const Tensor& q, const GqaCachedInput& input,
                                       const Tensor& pos, float scale,
                                       PagedKVBatchLayerView cache,
                                       const GqaSmallTInvocation& invocation,
                                       std::int32_t logical_capacity,
                                       std::int32_t implementation_window, std::int32_t splits,
                                       Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                       cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
        launch_nvfp4_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                        implementation_window, splits, partial_acc, partial_m,
                                        partial_l, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        launch_nvfp4_for<GqaMuseGeometry>(q, input, pos, scale, cache, invocation,
                                          logical_capacity, implementation_window, splits,
                                          partial_acc, partial_m, partial_l, stream);
        return;
    }
    launch_nvfp4_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                    implementation_window, splits, partial_acc, partial_m,
                                    partial_l, stream);
}

} // namespace ninfer::ops::detail
