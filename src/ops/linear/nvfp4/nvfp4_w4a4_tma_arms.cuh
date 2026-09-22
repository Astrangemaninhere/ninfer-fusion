#pragma once

// Shared per-geometry launch machinery, relocated verbatim from
// nvfp4_w4a4_tma.cu: the TmaM256N128/TmaM256N128S2 schedule aliases, the
// 14336/1024/6144 attention row layout with AttentionOutput, launch_tma<>
// and launch_linear<>. Each arm TU includes this header and instantiates
// only the geometries it owns, so one ptxas job per geometry family.
//
// The arm entry points declared below are new; they are defined in
// nvfp4_w4a4_tma_{attn,gdn,mlp,residual}.cu and called from the dispatcher
// left behind in nvfp4_w4a4_tma.cu. Their parameter lists mirror the public
// launch_nvfp4_w4a4_tma_* signatures; the Nvfp4Problem parameter feeds the
// relocated switch, whose case blocks are byte-identical to the originals.

#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include "core/arch_caps.h"
#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void launch_nvfp4_w4a4_tma_attn_linear(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                       const std::uint8_t* activation_scales,
                                       const std::uint8_t* weight_codes,
                                       const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                       std::int32_t tokens, float alpha, cudaStream_t stream);

void launch_nvfp4_w4a4_tma_attn_attention(const std::uint8_t* activation_codes,
                                          const std::uint8_t* activation_scales,
                                          const std::uint8_t* weight_codes,
                                          const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                          __nv_bfloat16* gate, __nv_bfloat16* key,
                                          __nv_bfloat16* value, std::int32_t tokens, float alpha,
                                          cudaStream_t stream);

void launch_nvfp4_w4a4_tma_gdn_linear(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                      std::int32_t tokens, float alpha, cudaStream_t stream);

void launch_nvfp4_w4a4_tma_gdn_qkvz(const std::uint8_t* activation_codes,
                                    const std::uint8_t* activation_scales,
                                    const std::uint8_t* weight_codes,
                                    const std::uint8_t* weight_scales, __nv_bfloat16* qkv,
                                    __nv_bfloat16* z, std::int32_t tokens, float alpha,
                                    cudaStream_t stream);

void launch_nvfp4_w4a4_tma_mlp_linear(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                      std::int32_t tokens, float alpha, cudaStream_t stream);

void launch_nvfp4_w4a4_tma_residual_linear(Nvfp4Problem problem,
                                           const std::uint8_t* activation_codes,
                                           const std::uint8_t* activation_scales,
                                           const std::uint8_t* weight_codes,
                                           const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                           std::int32_t tokens, float alpha, cudaStream_t stream);

void launch_nvfp4_w4a4_tma_residual_linear_add(Nvfp4Problem problem,
                                               const std::uint8_t* activation_codes,
                                               const std::uint8_t* activation_scales,
                                               const std::uint8_t* weight_codes,
                                               const std::uint8_t* weight_scales,
                                               __nv_bfloat16* residual, std::int32_t tokens,
                                               float alpha, cudaStream_t stream);

// launch_tma<>, launch_linear<> and the attention row layout below are
// the original file's anonymous-namespace block, unchanged.

namespace {

using TmaM256N128   = Nvfp4W4a4TmaSchedule<256, 3, 1>;
using TmaM256N128S2 = Nvfp4W4a4TmaSchedule<256, 2, 1>;

// The attention row split is shared with the W4A4 launcher and the planners: it is
// declared once in nvfp4_config.h and referenced here.
constexpr std::int32_t kQueryRows  = kNvfp4AttnQueryRows;
constexpr std::int32_t kKeyRows    = kNvfp4AttnKeyRows;
constexpr std::int32_t kGateRows   = kNvfp4AttnGateRows;
constexpr std::int32_t kKeyBegin   = kNvfp4AttnKeyBegin;
constexpr std::int32_t kGateBegin  = kNvfp4AttnGateBegin;
constexpr std::int32_t kValueBegin = kNvfp4AttnValueBegin;

struct AttentionOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kQueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

static_assert((kQueryRows % TmaM256N128::kBlockN) == 0);
static_assert((kKeyRows % TmaM256N128::kBlockN) == 0);
static_assert((kGateRows % TmaM256N128::kBlockN) == 0);

template <class Geometry, class Schedule, class Epilogue, class Output>
void launch_tma(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                std::int32_t tokens, float alpha, Epilogue epilogue, Output output,
                cudaStream_t stream) {
    // THE HOST HALF OF THE ARCH DECISION, and it is a RUNTIME fact on purpose: the guard
    // that compiles the device body out is keyed on __CUDA_ARCH__, which is not defined on the
    // host pass, so a compile-time test here would be answered about the wrong thing. The
    // predicate is the SAME fact the device guard encodes -- "this card has the
    // kind::mxf4nvf4 channel the pipeline's inner mma needs" -- asked of the card in hand off
    // the measured capability table (src/core/arch_caps.h), not off a number.
    //
    // Cached and single-sourced: see nvfp4_tma_rung_has_the_mxf4_channel() in
    // nvfp4_w4a4_tma.cuh, where the device half of this same decision lives.
    if (!nvfp4_tma_rung_has_the_mxf4_channel()) {
        throw std::runtime_error(nvfp4_tma_below_floor_refusal());
    }

    const Nvfp4W4a4TmaDescriptors descriptors =
        make_nvfp4_w4a4_tma_descriptors<Geometry, Schedule::kBlockM>(
            activation_codes, activation_scales, weight_codes, weight_scales, tokens);
    constexpr std::size_t kSharedBytes = sizeof(Nvfp4W4a4TmaSharedStorage<Schedule>);
    static const bool kConfigured      = [] {
        CUDA_CHECK(cudaFuncSetAttribute(nvfp4_w4a4_tma_kernel<Geometry, Schedule, Epilogue, Output>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(kSharedBytes)));
        return true;
    }();
    (void)kConfigured;

    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN, tokens / Schedule::kBlockM);
    nvfp4_w4a4_tma_kernel<Geometry, Schedule>
        <<<grid, Schedule::kThreads, kSharedBytes, stream>>>(descriptors, alpha, epilogue, output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_linear(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                   const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                   __nv_bfloat16* output, std::int32_t tokens, float alpha, cudaStream_t stream) {
    launch_tma<Geometry, TmaM256N128>(activation_codes, activation_scales, weight_codes,
                                      weight_scales, tokens, alpha, Nvfp4IdentityEpilogue{},
                                      Nvfp4ContiguousOutput{output, Geometry::kOutputRows}, stream);
}

} // namespace

} // namespace ninfer::ops::detail
