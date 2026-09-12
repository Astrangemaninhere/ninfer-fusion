#include "ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh"

namespace ninfer::ops::detail {

template <class Geometry>
void launch_linear_add(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                       const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                       __nv_bfloat16* residual, std::int32_t tokens, float alpha,
                       cudaStream_t stream) {
    launch_tma<Geometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4AddResidualEpilogue{residual, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{residual, Geometry::kOutputRows}, stream);
}

void launch_nvfp4_w4a4_tma_residual_linear(Nvfp4Problem problem,
                                           const std::uint8_t* activation_codes,
                                           const std::uint8_t* activation_scales,
                                           const std::uint8_t* weight_codes,
                                           const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                           std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4Problem::Residual6144:
        launch_linear<Nvfp4Residual6144Geometry>(activation_codes, activation_scales, weight_codes,
                                                 weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_linear<Nvfp4Residual17408Geometry>(activation_codes, activation_scales, weight_codes,
                                                  weight_scales, output, tokens, alpha, stream);
        return;
    default:
        return;
    }
}

void launch_nvfp4_w4a4_tma_residual_linear_add(Nvfp4Problem problem,
                                               const std::uint8_t* activation_codes,
                                               const std::uint8_t* activation_scales,
                                               const std::uint8_t* weight_codes,
                                               const std::uint8_t* weight_scales,
                                               __nv_bfloat16* residual, std::int32_t tokens,
                                               float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4Problem::Residual6144:
        launch_linear_add<Nvfp4Residual6144Geometry>(activation_codes, activation_scales,
                                                     weight_codes, weight_scales, residual, tokens,
                                                     alpha, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_linear_add<Nvfp4Residual17408Geometry>(activation_codes, activation_scales,
                                                      weight_codes, weight_scales, residual, tokens,
                                                      alpha, stream);
        return;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
        return;
    default:
        return;
    }
}

} // namespace ninfer::ops::detail
