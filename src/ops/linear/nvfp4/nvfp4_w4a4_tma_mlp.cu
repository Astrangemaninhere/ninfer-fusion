#include "ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh"

namespace ninfer::ops::detail {

void launch_nvfp4_w4a4_tma_mlp_linear(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                      std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4Problem::MlpGateUp:
        launch_tma<Nvfp4MlpGateUpGeometry, TmaM256N128S2>(
            activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
            Nvfp4IdentityEpilogue{},
            Nvfp4ContiguousOutput{output, Nvfp4MlpGateUpGeometry::kOutputRows}, stream);
        return;
    default:
        return;
    }
}

} // namespace ninfer::ops::detail
