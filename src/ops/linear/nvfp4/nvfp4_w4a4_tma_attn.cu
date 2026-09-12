#include "ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh"

namespace ninfer::ops::detail {

void launch_nvfp4_w4a4_tma_attn_linear(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                       const std::uint8_t* activation_scales,
                                       const std::uint8_t* weight_codes,
                                       const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                       std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4Problem::AttnInput:
        launch_linear<Nvfp4AttnInputGeometry>(activation_codes, activation_scales, weight_codes,
                                              weight_scales, output, tokens, alpha, stream);
        return;
    default:
        return;
    }
}

void launch_nvfp4_w4a4_tma_attn_attention(const std::uint8_t* activation_codes,
                                          const std::uint8_t* activation_scales,
                                          const std::uint8_t* weight_codes,
                                          const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                          __nv_bfloat16* gate, __nv_bfloat16* key,
                                          __nv_bfloat16* value, std::int32_t tokens, float alpha,
                                          cudaStream_t stream) {
    launch_tma<Nvfp4AttnInputGeometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4IdentityEpilogue{}, AttentionOutput{query, key, gate, value}, stream);
}

} // namespace ninfer::ops::detail
