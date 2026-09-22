#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_ladder.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4AttnInputGeometry;

struct Nvfp4W4a4AttentionOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kNvfp4AttnKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kNvfp4AttnQueryRows + parent_row;
        }
        if (parent_row < kNvfp4AttnGateBegin) {
            return key + static_cast<std::int64_t>(token) * kNvfp4AttnKeyRows + parent_row -
                   kNvfp4AttnKeyBegin;
        }
        if (parent_row < kNvfp4AttnValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kNvfp4AttnGateRows + parent_row -
                   kNvfp4AttnGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kNvfp4AttnKeyRows + parent_row -
               kNvfp4AttnValueBegin;
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                 Nvfp4W4a4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4W4a4AttentionOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    const float alpha = nvfp4_w4a4_alpha(weight);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, nvfp4_w4a4_tma_route(x.ne[1]), stream);
    const std::int32_t tokens = x.ne[1];
    if (nvfp4_w4a4_tma_route(tokens)) {
        const float alpha = nvfp4_w4a4_alpha(weight);
        launch_nvfp4_w4a4_tma_attention(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(v.data), tokens, alpha, stream);
        return;
    }
    // The T -> schedule decision is shared, not copied: see nvfp4_w4a4_ladder.cuh.
    nvfp4_w4a4_visit_mma_shape(
        nvfp4_w4a4_mma_shape(tokens, kNvfp4IsResidualGeometry<Geometry>,
                             kNvfp4IsGdnInputGeometry<Geometry>),
        [&](auto schedule) {
            launch_gemm<decltype(schedule)>(weight, q, gate, k, v, workspace, tokens, stream);
        });
}

} // namespace ninfer::ops::detail
