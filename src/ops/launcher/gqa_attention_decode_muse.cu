// ninfer::ops::detail - GqaMuseGeometry small-T GQA kernel instantiation TU.
// Split from gqa_attention_decode.cu (per-geometry ptxas memory bound).
#include "ops/launcher/gqa_attention.h"
#include "ops/launcher/gqa_attention_decode_impl.cuh"

namespace ninfer::ops::detail {

void gqa_attention_small_t_muse(const Tensor& q, const Tensor& k, const Tensor& v,
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
    gqa_attention_small_t_launch_for<GqaMuseGeometry>(q, input, pos, scale, cache, invocation, envelope,
                                         partial_acc, partial_m, partial_l, out, stream);
}

void gqa_attention_cached_small_t_muse(const Tensor& q, const Tensor& pos, float scale,
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
    gqa_attention_small_t_launch_for<GqaMuseGeometry>(q, input, pos, scale, batch_cache, invocation,
                                         envelope, partial_acc, partial_m, partial_l, out,
                                         stream);
}

} // namespace ninfer::ops::detail
