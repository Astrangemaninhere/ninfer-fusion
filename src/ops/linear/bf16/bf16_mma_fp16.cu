// bf16_mma_fp16.cu -- the TU that carries the fp16-plane arm's kernels and their only launcher.
//
// THE LAUNCHER IS HOST CODE INSIDE A nvcc TU on purpose: it is the address
// tests/test_bf16_fp16_plane.cpp takes, which is what makes the build fact
// (NINFER_HAVE_BF16_FP16_MMA, published beside this source line in src/CMakeLists.txt)
// non-optional. A route that names a kernel the binary does not contain is the phantom this
// tree has already shipped once.
#include "ops/linear/bf16/bf16_fp16_route.h"

#include "core/arch_caps.h"
#include "ops/linear/bf16/bf16_mma_fp16.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

void launch_channel(const Tensor& x, const Weight& weight, Tensor& out,
                    Bf16Fp16Channel channel, cudaStream_t stream) {
    const auto* w = static_cast<const __nv_bfloat16*>(weight.qdata);
    const auto* a = static_cast<const __nv_bfloat16*>(x.data);
    auto* y       = static_cast<__nv_bfloat16*>(out.data);
    const int n   = weight.n;
    const int k   = weight.k;
    const int t   = x.ne[1];
    if (n <= 0 || k <= 0 || t <= 0) {
        throw std::invalid_argument("bf16 fp16-plane mma: empty problem");
    }

    switch (channel) {
    case Bf16Fp16Channel::Mma8n8k4: {
        // K must tile the superchunk: the kernel's smem staging is KC wide and has no K
        // remainder arm. Refusing by name rather than reading out of bounds is the same
        // fail-closed shape the QPN host entry uses for its own geometry guard.
        if (k % kBf16Fp16Mma8KC != 0) {
            throw std::invalid_argument(
                "bf16 fp16-plane mma (m8n8k4): K must be a multiple of the " +
                std::to_string(kBf16Fp16Mma8KC) + "-wide smem superchunk; this problem has K=" +
                std::to_string(k));
        }
        constexpr int kWarps = kBf16Fp16Mma8Warps;
        const dim3 block(32 * kWarps);
        const dim3 grid((n + 8 * kWarps - 1) / (8 * kWarps), (t + 7) / 8);
        const int smem = 8 * (kBf16Fp16Mma8KC / 2 + 1) * (int)sizeof(__half2);
        bf16_mma8_kernel<kBf16Fp16Mma8KC><<<grid, block, smem, stream>>>(w, a, y, n, k, t);
        break;
    }
    case Bf16Fp16Channel::Mma16n8k8: {
        if (k % 8 != 0) {
            throw std::invalid_argument(
                "bf16 fp16-plane mma (m16n8k8): K must be a multiple of the 8-k mma step; this "
                "problem has K=" +
                std::to_string(k));
        }
        constexpr int kWarps = kBf16Fp16Mma1688Warps;
        const dim3 block(32 * kWarps);
        const dim3 grid((n + 16 * kWarps - 1) / (16 * kWarps), (t + 7) / 8);
        bf16_mma1688_kernel<<<grid, block, 0, stream>>>(w, a, y, n, k, t);
        break;
    }
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("bf16 fp16-plane mma launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

} // namespace

// THE ROW'S CHANNEL, MAPPED IN ONE PLACE. arch_caps.h carries the channel as the LITERAL the
// kernel emits (a string_view) and deliberately does NOT carry this enum: the tables are the
// capability layer and the enum is the kernel layer's, so the join lives here. An unknown
// literal FAILS CLOSED rather than picking a default -- a wrong channel is a route that names
// one instruction and emits another, which is the defect this whole arm exists to remove.
Bf16Fp16Channel channel_for_rung(const caps::Fp16PlaneChannelRung& rung) {
    if (rung.channel.find("m16n8k8") != std::string_view::npos) {
        return Bf16Fp16Channel::Mma16n8k8;
    }
    if (rung.channel.find("m8n8k4") != std::string_view::npos) {
        return Bf16Fp16Channel::Mma8n8k4;
    }
    throw std::invalid_argument(
        "bf16 fp16-plane mma: the rung's channel literal is not one this TU emits (" +
        std::string(rung.channel) + "). Add the kernel before adding the row.");
}

void launch_bf16_fp16_mma(const Tensor& x, const Weight& weight, Tensor& out,
                          const Bf16Fp16Plan& plan, cudaStream_t stream) {
    if (!plan.use_arm) {
        throw std::invalid_argument(
            "launch_bf16_fp16_mma called with a plan that did not select the arm (route=" +
            plan.route + "). The caller must branch on Bf16Fp16Plan::use_arm; taking this route "
            "anyway would name a tensor-core path the tables did not answer for. " + plan.why);
    }
    // WHICH CHANNEL IS THE RUNG'S, not this box's: the plan carries the simulated/real rung the
    // tables answered for, and kFp16PlaneChannelRungs is where that rung's measured channel
    // lives. The kernels are both in every cubin, so this is a ROUTE fact rather than a build
    // fact -- and the caveat travels with it: the cubin that executes is this binary's own.
    const caps::Fp16PlaneChannelRung* rung = caps::fp16_plane_channel_rung(plan.effective_sm);
    if (rung == nullptr) {
        throw std::invalid_argument(
            "bf16 fp16-plane mma: rung sm_" + std::to_string(plan.effective_sm) +
            " has no row in kFp16PlaneChannelRungs, so which channel to emit is UNMEASURED here "
            "and is not guessed. Missing measurement: per-rung `-cubin` + `nvdisasm -c` on the "
            "instantiation probe, counting HMMA against CALLs to the in-cubin FFMA routine.");
    }
    launch_channel(x, weight, out, channel_for_rung(*rung), stream);
}

} // namespace ninfer::ops::detail
