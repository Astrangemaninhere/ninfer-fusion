#include "ops/launcher/dspark_markov_argmax.h"

#include "core/device.h"
#include "ops/kernel/dspark_markov_argmax.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void dspark_markov_argmax_launch(const Tensor& logits, const Weight& markov_w1,
                                 const Weight& markov_w2, const Tensor& anchors, Tensor& drafts,
                                 Tensor& best_value, Tensor& best_index, Tensor& extents,
                                 std::int32_t k, float entropy_threshold, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int32_t batch = anchors.ne[0];
    const dim3 grid(kDsparkMarkovTiles, batch);
    auto* best_value_ptr = static_cast<std::int32_t*>(best_value.data);
    auto* best_index_ptr = static_cast<std::int32_t*>(best_index.data);
    auto* drafts_ptr     = static_cast<std::int32_t*>(drafts.data);
    auto* extents_ptr    = static_cast<std::int32_t*>(extents.data);
    const float threshold_squared = entropy_threshold * entropy_threshold;
    for (std::int32_t position = 0; position < k; ++position) {
        CUDA_CHECK(cudaMemsetAsync(best_value_ptr, 0, sizeof(std::int32_t) * batch, stream));
        dspark_markov_argmax_kernel<kBlock><<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const __nv_bfloat16*>(markov_w1.qdata),
            static_cast<const __nv_bfloat16*>(markov_w2.qdata),
            static_cast<const std::int32_t*>(anchors.data),
            static_cast<const std::int32_t*>(drafts.data), best_value_ptr, best_index_ptr,
            extents_ptr, logits.ne[0], k, position);
        CUDA_CHECK(cudaGetLastError());
        for (std::int32_t row = 0; row < batch; ++row) {
            CUDA_CHECK(cudaMemcpyAsync(drafts_ptr + static_cast<std::int64_t>(row) * k + position,
                                       best_index_ptr + row, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToDevice, stream));
        }
        dspark_svip_entropy_kernel<kBlock><<<batch, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), extents_ptr, logits.ne[0], k, position,
            threshold_squared);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
