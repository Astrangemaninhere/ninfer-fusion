// qpn_arch_route.cpp -- the engine-side implementation of the arch route decision.
//
// HOST-ONLY, and deliberately a .cpp: the decision reads environment variables and builds
// std::string reasons, so it has no business being in a device translation unit, and keeping
// kernel_route.h/arch_sim.h out of nvfp4_w4a4.cu's include graph is why the declaration in
// qpn_arch_route.h carries no caps:: types at all.
#include "ops/linear/qpn/qpn_arch_route.h"

#include "core/arch_sim.h"
#include "core/kernel_route.h"
#include "ops/linear/qpn/qpn_host.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

namespace ninfer::ops::qpn {
namespace {

// THE BUILD FACT, TAKEN FROM THE LINKER RATHER THAN FROM A MACRO.
//
// caps::kQpnInBuild is a preprocessor constant published by src/CMakeLists.txt together with
// the source line that makes it true, and it is the right answer INSIDE a translation unit
// that receives it. This TU does not: it lives in ninfer_ops, the fact is published on
// ninfer_artifact (and on ninfer_qpn_build_fact), and pushing it onto ninfer_ops would change
// the command line of ~500 translation units to deliver one -D to one file -- the cost
// FALLBACK explicitly refused to pay on ninfer_core. MEASURED 2026-09-18 with this line absent
// (ARCH2/arms/probe_sim_norequest.txt): kQpnInBuild == false inside the engine's own decision,
// so `select_route` refused a route the binary could serve.
//
// `&gemm_qpn` cannot be null: the symbol is either resolved by the linker or the link fails.
// So this expression is a LINK-TIME fact, it is exact (it cannot disagree with what is in the
// binary, which is the failure mode the macro has), and it is the same tie
// tests/test_qpn_build_fact.cpp already relies on.
[[nodiscard]] bool qpn_kernel_in_this_build() noexcept { return &gemm_qpn != nullptr; }


// The route decision cannot change within a process: the device does not, and
// arch_view_for_device() latches the environment on its first call. The SHAPE can, so the
// cache is keyed on it. thread_local because the engine is free to decode on more than one
// thread and a shared mutable cache on the hot path is a data race, not an optimisation.
struct DecisionCache {
    bool        valid = false;
    std::int32_t m = 0;
    std::int32_t n = 0;
    std::int32_t k = 0;
    QpnArchRoute route;
};
thread_local DecisionCache t_cache;

QpnArchRoute decide(int physical_sm, std::int32_t m, std::int32_t n, std::int32_t k) {
    QpnArchRoute out;
    out.physical_sm     = physical_sm;
    out.kernel_in_build = qpn_kernel_in_this_build();
    out.route           = "none";

    // The three-key override, read through its one entry point. This call is also what prints
    // sim_banner() once per process when the override is honoured, so an engine run can never
    // be simulated silently.
    const caps::ArchView view = caps::arch_view_for_device(physical_sm);
    out.effective_sm          = view.effective_sm;
    out.simulated             = view.simulated();

    const caps::ProblemShape shape{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(n),
                                  static_cast<std::uint32_t>(k)};

    if (!view.usable()) {
        // G3, and it is the whole point of having a three-state view: a request that could
        // not be honoured is NOT a licence to answer for the real device.
        out.refused = true;
        out.why     = caps::sim_refusal_reason(view);
        return out;
    }

    const caps::RouteChoice choice = caps::select_route(
        view, artifact::NumericFormat::NVFP4, shape, qpn_kernel_in_this_build());
    out.route  = std::string(caps::route_name(choice.route));
    out.kernel = std::string(choice.kernel);
    out.wide_m = choice.wide_m;
    out.why    = choice.why;
    out.use_qpn = choice.outcome == caps::RouteOutcome::Selected &&
                  choice.route == caps::KernelRoute::QpnW4a16;
    return out;
}

} // namespace

int current_device_sm() {
    static const int kSm = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { return 0; }
        int major = 0;
        int minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
            cudaSuccess) {
            return 0;
        }
        if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) !=
            cudaSuccess) {
            return 0;
        }
        return major * 10 + minor;
    }();
    return kSm;
}

QpnArchRoute select_qpn_arch_route(int physical_sm, std::int32_t m, std::int32_t n,
                                   std::int32_t k, bool emit_log_line) {
    DecisionCache& cache = t_cache;
    if (!cache.valid || cache.m != m || cache.n != n || cache.k != k) {
        cache.route = decide(physical_sm, m, n, k);
        cache.m     = m;
        cache.n     = n;
        cache.k     = k;
        cache.valid = true;
        // THE `[route][SIMULATED]` LINE, and this is its first caller in the tree. Printed on
        // a CHANGE OF SHAPE rather than per invocation -- see the header for why. Only a
        // simulated or refused answer is printed: the shipping path (ArchView::Disabled)
        // must not gain a stderr line, or every log this project has measured its runs
        // against would shift underneath the measurement.
        const QpnArchRoute& r = cache.route;
        if (emit_log_line && (r.simulated || r.refused)) {
            caps::RouteChoice mirror;
            mirror.simulated = r.simulated;
            mirror.kernel    = r.kernel;
            mirror.why       = r.why;
            if (r.refused) {
                mirror.outcome = caps::RouteOutcome::NoKernelInTree;
                mirror.route   = caps::KernelRoute::None;
            } else if (r.use_qpn) {
                mirror.outcome = caps::RouteOutcome::Selected;
                mirror.route   = caps::KernelRoute::QpnW4a16;
            } else {
                mirror.outcome = caps::RouteOutcome::NoKernelInTree;
                mirror.route   = caps::KernelRoute::None;
            }
            const caps::ProblemShape shape{static_cast<std::uint32_t>(m),
                                           static_cast<std::uint32_t>(n),
                                           static_cast<std::uint32_t>(k)};
            const caps::RouteLogLine line = caps::route_log_line(
                physical_sm, artifact::NumericFormat::NVFP4, shape, mirror);
            std::fprintf(stderr, "%s\n", line.text.c_str());
            if (r.refused) {
                std::fprintf(stderr,
                             "  the run must not continue on the real device either: a refused "
                             "simulation fails closed (src/core/arch_sim.h, G3).\n");
            }
        }
    }
    return cache.route;
}

void dispatch_qpn_fallback(const QpnArchRoute& route, QpnWeightLayout layout, const void* x_half,
                          const void* codes, const void* scales, float gscale, void* y_half,
                          std::int32_t m, std::int32_t k, std::int32_t n, void* stream) {
    if (route.refused) {
        throw std::runtime_error(
            "QPN fallback dispatch was asked to run after a REFUSED simulation. " + route.why);
    }
    if (!route.use_qpn) {
        throw std::logic_error(
            "dispatch_qpn_fallback was called without a selected QPN route (route='" +
            route.route + "', effective sm_" + std::to_string(route.effective_sm) + "). " +
            route.why);
    }
    if (route.wide_m && layout == QpnWeightLayout::NativeBlockScale) {
        // THE WIDE BAND IS NOT A PREPACK BAND, so the layout half of the refusal below does
        // not apply to it: gemm_qpn_wmma_native consumes exactly the blob the engine holds.
        // What still stands between this route and a run is the DTYPE, and only that, so the
        // refusal is NARROWED rather than lifted -- passing __nv_bfloat16 buffers to a `half`
        // kernel would compute numbers with no relationship to the model, which is strictly
        // worse than refusing.
        throw std::runtime_error(
            "QPN W4A16 wide-M band SELECTED for sm_" + std::to_string(route.effective_sm) +
            " (kernel '" + route.kernel +
            "'), and the weights ARE in the layout that kernel consumes ([N][K/2] codes + "
            "[N][K/16] e4m3 scales -- the artifact's own planes; no prepack is needed for "
            "this band). ONE port is still missing, and it is not a flag: an "
            "activation/output dtype conversion, because gemm_qpn_wmma_native takes `half` "
            "and every engine tensor on this path is __nv_bfloat16. bf16 and fp16 are "
            "different bit patterns, so dispatching now would be silent numerical "
            "corruption. The native -> qpn_prepack weight converter that the M 1..16 bands "
            "need is NOT the blocker for this band. Route reason: " + route.why);
    }
    if (layout != QpnWeightLayout::QpnPrepacked) {
        // FAIL CLOSED, and the reason is a MEASURED layout fact rather than a missing flag --
        // see the header. The alternative (feeding the native blob to gemm_qpn) would produce
        // numbers with no relationship to the model, which is strictly worse than refusing.
        throw std::runtime_error(
            "QPN W4A16 route SELECTED for sm_" + std::to_string(route.effective_sm) +
            " (kernel '" + route.kernel +
            "'), but the weights are in the artifact's own block-scale layout ([N][K/2] codes "
            "+ [N][K/16] e4m3 scales) and gemm_qpn consumes the qpn_prepack layout "
            "([tile][group][lane][16B] codes, scales indexed tile*G*32+lane). The two blobs "
            "are not the same bytes in the same order, so dispatching would be silent "
            "numerical corruption. Two pieces are missing, and both are ports rather than "
            "flags: (1) a native->qpn_prepack weight converter (reference: "
            "tools/archkit/qpn_port/qpn_prepack_proto.py; machine-verified slot map: "
            "src/ops/linear/qpn/qpn_map.cuh), and (2) an activation/output dtype conversion, "
            "because gemm_qpn takes `half` and every engine tensor on this path is "
            "__nv_bfloat16. Until both exist this route is SELECTED BY THE TABLES AND NOT "
            "DISPATCHABLE BY THE ENGINE, and it refuses rather than computing the wrong "
            "answer. Route reason: " + route.why);
    }
    gemm_qpn(x_half, codes, scales, gscale, y_half, m, k, n, static_cast<cudaStream_t>(stream));
}

} // namespace ninfer::ops::qpn
