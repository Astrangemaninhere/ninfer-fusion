#include "ops/launcher/dflash2_selector.h"

#include "core/device.h"
#include "ops/kernel/dflash2_selector.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void dflash2_selector_launch(const Tensor& unary_logits, const Tensor& projected_hidden,
                             const Weight& predecessor_codebook,
                             const Weight& successor_codebook, const Tensor& anchors,
                             Tensor& candidates, Tensor& unary, Tensor& scores, Tensor& drafts,
                             std::int32_t steps, std::int32_t top_k, cudaStream_t stream) {
    if (steps != 7 || top_k != kDflash2SelectorTopK) {
        throw std::invalid_argument("dflash2_selector: registered domain is S=7, K=16");
    }
    const std::int32_t batch   = anchors.ne[0];
    const std::int32_t columns = steps * batch;
    constexpr int kTopK        = kDflash2SelectorTopK;

    const dim3 topk_grid(columns);
    dflash2_selector_topk_kernel<256, kTopK><<<topk_grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(unary_logits.data),
        static_cast<std::int32_t*>(candidates.data), static_cast<float*>(unary.data),
        unary_logits.ne[0], batch, steps, columns);
    CUDA_CHECK(cudaGetLastError());

    const dim3 scores_grid(columns);
    dflash2_selector_scores_kernel<256><<<scores_grid, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(candidates.data),
        static_cast<const float*>(unary.data),
        static_cast<const __nv_bfloat16*>(projected_hidden.data),
        static_cast<const __nv_bfloat16*>(predecessor_codebook.qdata),
        static_cast<const __nv_bfloat16*>(successor_codebook.qdata),
        static_cast<const std::int32_t*>(anchors.data), static_cast<float*>(scores.data),
        unary_logits.ne[0], batch, steps, top_k, columns);
    CUDA_CHECK(cudaGetLastError());

    dflash2_selector_walk_kernel<<<batch, 32, 0, stream>>>(
        static_cast<const std::int32_t*>(candidates.data),
        static_cast<const float*>(scores.data), static_cast<std::int32_t*>(drafts.data), batch,
        steps, top_k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
