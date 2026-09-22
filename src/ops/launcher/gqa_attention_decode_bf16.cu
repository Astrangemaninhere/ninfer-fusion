// Cut B: bf16-tier decode launch, split out of gqa_attention_decode_smallt.cu (same
// pattern as gqa_attention_decode_e8.cu: every bf16 kernel instantiation for the three
// geometries lives in this TU, so a single nvcc invocation's ptxas stage stays small).
// The width ladder and the MultiBatch/Masked selection below are exactly what the
// NINFER_GQA_SMALL_T_DISPATCH macro expanded to before the split; only their location
// changed. This is also the dispatcher's fallback tier (every dtype that is not one of the
// five codecs above).
#include "ops/launcher/gqa_attention_decode_partial.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheInput, bool MultiBatch, bool Masked>
void launch_bf16_ladder(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                        PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                        std::int32_t logical_capacity, std::int32_t implementation_window,
                        std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                        Tensor& partial_l, cudaStream_t stream) {
    // launch_tc_partial_bf16 takes the fixed split grid precomputed while the i8/nvfp4 tiers
    // derive it inside their partial; derive it here from the same input, exactly as the
    // pre-split gqa_attention_small_t_launch_for did one frame above the dispatch.
    const std::int32_t split_units = gqa_small_t_split_units<Geometry>(implementation_window);
    switch (invocation.width) {
    case 1:
        launch_tc_partial_bf16<Geometry, 1, 2, MultiBatch, Masked>(
            q, input, pos, scale, cache, invocation, logical_capacity, splits, split_units,
            partial_acc, partial_m, partial_l, stream);
        return;
    case 2:
        launch_tc_partial_bf16<Geometry, 2, 4, MultiBatch, Masked>(
            q, input, pos, scale, cache, invocation, logical_capacity, splits, split_units,
            partial_acc, partial_m, partial_l, stream);
        return;
    case 3:
        launch_tc_partial_bf16<Geometry, 3, 4, MultiBatch, Masked>(
            q, input, pos, scale, cache, invocation, logical_capacity, splits, split_units,
            partial_acc, partial_m, partial_l, stream);
        return;
    case 4:
        launch_tc_partial_bf16<Geometry, 4, 4, MultiBatch, Masked>(
            q, input, pos, scale, cache, invocation, logical_capacity, splits, split_units,
            partial_acc, partial_m, partial_l, stream);
        return;
    case 5:
        launch_tc_partial_bf16<Geometry, 5, 4, MultiBatch, Masked>(
            q, input, pos, scale, cache, invocation, logical_capacity, splits, split_units,
            partial_acc, partial_m, partial_l, stream);
        return;
    case 6:
        launch_tc_partial_bf16<Geometry, 6, 4, MultiBatch, Masked>(
            q, input, pos, scale, cache, invocation, logical_capacity, splits, split_units,
            partial_acc, partial_m, partial_l, stream);
        return;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }
}

template <typename Geometry, typename CacheInput>
void launch_bf16_for(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                     PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                     std::int32_t logical_capacity, std::int32_t implementation_window,
                     std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                     Tensor& partial_l, cudaStream_t stream) {
    const bool masked = invocation.valid_columns != nullptr;
    if (invocation.batch_size == 1) {
        if (masked) {
            launch_bf16_ladder<Geometry, CacheInput, false, true>(
                q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
                splits, partial_acc, partial_m, partial_l, stream);
        } else {
            launch_bf16_ladder<Geometry, CacheInput, false, false>(
                q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
                splits, partial_acc, partial_m, partial_l, stream);
        }
    } else if (masked) {
        launch_bf16_ladder<Geometry, CacheInput, true, true>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
    } else {
        launch_bf16_ladder<Geometry, CacheInput, true, false>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
    }
}

} // namespace

void gqa_attention_decode_bf16_launch(const Tensor& q, const GqaAppendInput& input,
                                      const Tensor& pos, float scale,
                                      PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      std::int32_t logical_capacity,
                                      std::int32_t implementation_window, std::int32_t splits,
                                      Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                      cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
        launch_bf16_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                       implementation_window, splits, partial_acc, partial_m,
                                       partial_l, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        launch_bf16_for<GqaMuseGeometry>(q, input, pos, scale, cache, invocation,
                                         logical_capacity, implementation_window, splits,
                                         partial_acc, partial_m, partial_l, stream);
        return;
    }
    if (q.ne[1] != Gqa35Geometry::QHeads || q.ne[0] != Gqa35Geometry::HeadDim) {
        throw std::invalid_argument("bf16 decode launch (append): unsupported query-head geometry");
    }
    launch_bf16_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                   implementation_window, splits, partial_acc, partial_m,
                                   partial_l, stream);
}

void gqa_attention_decode_bf16_launch(const Tensor& q, const GqaCachedInput& input,
                                      const Tensor& pos, float scale,
                                      PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      std::int32_t logical_capacity,
                                      std::int32_t implementation_window, std::int32_t splits,
                                      Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                      cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
        launch_bf16_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                       implementation_window, splits, partial_acc, partial_m,
                                       partial_l, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        launch_bf16_for<GqaMuseGeometry>(q, input, pos, scale, cache, invocation,
                                         logical_capacity, implementation_window, splits,
                                         partial_acc, partial_m, partial_l, stream);
        return;
    }
    launch_bf16_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                   implementation_window, splits, partial_acc, partial_m,
                                   partial_l, stream);
}

} // namespace ninfer::ops::detail
