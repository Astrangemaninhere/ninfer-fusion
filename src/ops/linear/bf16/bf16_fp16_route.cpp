// bf16_fp16_route.cpp -- the engine-side implementation of the fp16-plane route decision.
// See the header for what it decides and what it does not claim.
#include "ops/linear/bf16/bf16_fp16_route.h"

#include "core/arch_sim.h"
#include "core/kernel_route.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {
namespace {

// THE BUILD FACT. caps::kFp16PlaneInBuild is a preprocessor constant published by
// src/CMakeLists.txt on the same lines that add the source that makes it true, so this TU
// answers from the macro exactly as every other reader of that fact does. WHAT MAKES IT NOT A
// MOOD is the link-time tie: tests/test_bf16_fp16_plane.cpp links ninfer_ops AND takes the
// address of ops::detail::launch_bf16_fp16_mma, so removing the source line while leaving the
// definition is a LINK ERROR rather than a route naming a kernel the binary does not contain --
// the same tie shape tests/test_qpn_build_fact.cpp uses for ops::qpn::gemm_qpn.
//
// NOTE the difference from qpn_arch_route.cpp:36, where the fact is read from the LINKER rather
// than the macro because that TU's target does not carry the definition. ninfer_ops DOES carry
// this one (it is the target the source line lives in), so the macro is the exact answer here
// and a second answer would be a second thing to keep in step.
[[nodiscard]] bool fp16_plane_kernel_in_this_build() noexcept { return caps::kFp16PlaneInBuild; }

struct PlanCache {
    bool valid          = false;
    std::int32_t tokens = 0;
    std::int32_t n      = 0;
    std::int32_t k      = 0;
    Bf16Fp16Plan plan;
};
thread_local PlanCache t_cache;

} // namespace

int bf16_fp16_physical_sm() {
    static const int cached = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { return 0; }
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) { return 0; }
        return prop.major * 10 + prop.minor;
    }();
    return cached;
}

Bf16Fp16Plan bf16_fp16_plan(std::int32_t tokens, std::int32_t n, std::int32_t k) {
    if (t_cache.valid && t_cache.tokens == tokens && t_cache.n == n && t_cache.k == k) {
        return t_cache.plan;
    }

    Bf16Fp16Plan out;
    out.physical_sm = bf16_fp16_physical_sm();
    out.min_tokens  = caps::kBf16Fp16MmaActivationExtent;

    // The three-key override, read through its one entry point. This call is also what prints
    // sim_banner() once per process when the override is honoured, so an engine run can never be
    // simulated silently (arch_sim.h G2).
    const caps::ArchView view = caps::arch_view_for_device(out.physical_sm);
    out.effective_sm          = view.effective_sm;
    out.simulated             = view.simulated();

    if (!view.usable()) {
        // G3: a request that could not be honoured is NOT a licence to answer for the real
        // device. The caller must not proceed on either route.
        out.refused = true;
        out.route   = "none";
        out.why     = caps::sim_refusal_reason(view);
        t_cache     = {true, tokens, n, k, out};
        return out;
    }

    const caps::ProblemShape shape{static_cast<std::uint32_t>(tokens),
                                   static_cast<std::uint32_t>(n),
                                   static_cast<std::uint32_t>(k)};
    const caps::RouteChoice choice =
        caps::select_route(view, artifact::NumericFormat::BF16, shape, caps::kQpnInBuild,
                           fp16_plane_kernel_in_this_build());

    out.route  = std::string(caps::route_name(choice.route));
    out.kernel = std::string(choice.kernel);
    out.why    = choice.why;

    // THE TOKEN THRESHOLD IS A PROPERTY OF THE ATOM, NOT A PREFERENCE, and it is the one place
    // this decision can DECLINE a route the tables selected. A warp-level fp16 mma atom consumes
    // 8 activation rows, so at T < 8 the same mma work is issued for fewer useful tokens and the
    // arm is worse than the FFMA GEMV it would replace -- a T=1 GEMV is weight-bandwidth bound
    // and a tensor core cannot reduce the bytes it must read. So the arm is taken only from
    // kBf16Fp16MmaActivationExtent up, and the decline is EXPLICIT rather than a silent
    // fall-through.
    const bool route_is_this_arm = (choice.route == caps::KernelRoute::MmaFp16Plane);
    out.use_arm = route_is_this_arm && tokens >= out.min_tokens;
    if (route_is_this_arm && !out.use_arm) {
        out.why += " DECLINED BY THE CALLER: the tables selected this arm, and this op has " +
                   std::to_string(tokens) + " token(s), below the atom's own extent of " +
                   std::to_string(out.min_tokens) +
                   " -- at that width the mma issues the same work for fewer useful tokens and "
                   "the FFMA shape table is the better route. The arm is taken from " +
                   std::to_string(out.min_tokens) + " tokens up.";
    }
    t_cache = {true, tokens, n, k, out};
    return out;
}

} // namespace ninfer::ops::detail
