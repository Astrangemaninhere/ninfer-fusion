// Cut B: i8-tier decode launch, split out of gqa_attention_decode_smallt.cu (same
// pattern as gqa_attention_decode_e8.cu: every i8 kernel instantiation for the three
// geometries lives in this TU, so a single nvcc invocation's ptxas stage stays small).
// The width ladder and the MultiBatch/Masked selection below are exactly what the
// NINFER_GQA_SMALL_T_DISPATCH macro expanded to before the split; only their location
// changed.
#include "ops/launcher/gqa_attention_decode_partial.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheInput, bool MultiBatch, bool Masked>
void launch_i8_ladder(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                      PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                      std::int32_t logical_capacity, std::int32_t implementation_window,
                      std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                      Tensor& partial_l, cudaStream_t stream) {
    switch (invocation.width) {
    case 1:
        launch_tc_partial_i8<Geometry, 1, MultiBatch, Masked, false>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
        return;
    case 2:
        launch_tc_partial_i8<Geometry, 2, MultiBatch, Masked, false>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
        return;
    case 3:
        launch_tc_partial_i8<Geometry, 3, MultiBatch, Masked, false>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
        return;
    case 4:
        launch_tc_partial_i8<Geometry, 4, MultiBatch, Masked, false>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
        return;
    case 5:
        launch_tc_partial_i8<Geometry, 5, MultiBatch, Masked, false>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
        return;
    case 6:
        launch_tc_partial_i8<Geometry, 6, MultiBatch, Masked, false>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
        return;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }
}

template <typename Geometry, typename CacheInput>
void launch_i8_for(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                   PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                   std::int32_t logical_capacity, std::int32_t implementation_window,
                   std::int32_t splits, Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                   cudaStream_t stream) {
    const bool masked = invocation.valid_columns != nullptr;
    if (invocation.batch_size == 1) {
        if (masked) {
            launch_i8_ladder<Geometry, CacheInput, false, true>(
                q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
                splits, partial_acc, partial_m, partial_l, stream);
        } else {
            launch_i8_ladder<Geometry, CacheInput, false, false>(
                q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
                splits, partial_acc, partial_m, partial_l, stream);
        }
    } else if (masked) {
        launch_i8_ladder<Geometry, CacheInput, true, true>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
    } else {
        launch_i8_ladder<Geometry, CacheInput, true, false>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
    }
}

} // namespace

void gqa_attention_decode_i8_launch(const Tensor& q, const GqaAppendInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t implementation_window, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
        launch_i8_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                     implementation_window, splits, partial_acc, partial_m,
                                     partial_l, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        launch_i8_for<GqaMuseGeometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                       implementation_window, splits, partial_acc, partial_m,
                                       partial_l, stream);
        return;
    }
    if (q.ne[1] != Gqa35Geometry::QHeads || q.ne[0] != Gqa35Geometry::HeadDim) {
        throw std::invalid_argument("i8 decode launch (append): unsupported query-head geometry");
    }
    launch_i8_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                 implementation_window, splits, partial_acc, partial_m, partial_l,
                                 stream);
}

void gqa_attention_decode_i8_launch(const Tensor& q, const GqaCachedInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t implementation_window, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
        launch_i8_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                     implementation_window, splits, partial_acc, partial_m,
                                     partial_l, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        launch_i8_for<GqaMuseGeometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                       implementation_window, splits, partial_acc, partial_m,
                                       partial_l, stream);
        return;
    }
    launch_i8_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                 implementation_window, splits, partial_acc, partial_m, partial_l,
                                 stream);
}

} // namespace ninfer::ops::detail
