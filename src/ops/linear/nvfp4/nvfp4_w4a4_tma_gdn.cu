#include "ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh"

namespace ninfer::ops::detail {

void launch_nvfp4_w4a4_tma_gdn_linear(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                      std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4Problem::GdnInput:
        launch_linear<Nvfp4GdnInputGeometry>(activation_codes, activation_scales, weight_codes,
                                             weight_scales, output, tokens, alpha, stream);
        return;
    default:
        return;
    }
}

void launch_nvfp4_w4a4_tma_gdn_qkvz(const std::uint8_t* activation_codes,
                                    const std::uint8_t* activation_scales,
                                    const std::uint8_t* weight_codes,
                                    const std::uint8_t* weight_scales, __nv_bfloat16* qkv,
                                    __nv_bfloat16* z, std::int32_t tokens, float alpha,
                                    cudaStream_t stream) {
    launch_tma<Nvfp4GdnInputGeometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4IdentityEpilogue{}, Nvfp4GdnInputOutput{qkv, z}, stream);
}

} // namespace ninfer::ops::detail
