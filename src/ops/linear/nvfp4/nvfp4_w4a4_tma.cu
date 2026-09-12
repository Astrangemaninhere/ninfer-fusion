#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                  const std::uint8_t* activation_scales,
                                  const std::uint8_t* weight_codes,
                                  const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                  std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4Problem::AttnInput:
        launch_nvfp4_w4a4_tma_attn_linear(problem, activation_codes, activation_scales,
                                          weight_codes, weight_scales, output, tokens, alpha,
                                          stream);
        return;
    case Nvfp4Problem::GdnInput:
        launch_nvfp4_w4a4_tma_gdn_linear(problem, activation_codes, activation_scales, weight_codes,
                                         weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launch_nvfp4_w4a4_tma_mlp_linear(problem, activation_codes, activation_scales, weight_codes,
                                         weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::Residual17408:
        launch_nvfp4_w4a4_tma_residual_linear(problem, activation_codes, activation_scales,
                                              weight_codes, weight_scales, output, tokens, alpha,
                                              stream);
        return;
    default:
        return;
    }
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t* activation_codes,
                                     const std::uint8_t* activation_scales,
                                     const std::uint8_t* weight_codes,
                                     const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                     __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                     std::int32_t tokens, float alpha, cudaStream_t stream) {
    launch_nvfp4_w4a4_tma_attn_attention(activation_codes, activation_scales, weight_codes,
                                         weight_scales, query, gate, key, value, tokens, alpha,
                                         stream);
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t* activation_codes,
                               const std::uint8_t* activation_scales,
                               const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                               __nv_bfloat16* qkv, __nv_bfloat16* z, std::int32_t tokens,
                               float alpha, cudaStream_t stream) {
    launch_nvfp4_w4a4_tma_gdn_qkvz(activation_codes, activation_scales, weight_codes, weight_scales,
                                   qkv, z, tokens, alpha, stream);
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __nv_bfloat16* residual,
                                      std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::Residual17408:
        launch_nvfp4_w4a4_tma_residual_linear_add(problem, activation_codes, activation_scales,
                                                  weight_codes, weight_scales, residual, tokens,
                                                  alpha, stream);
        return;
    default:
        return;
    }
}

} // namespace ninfer::ops::detail
