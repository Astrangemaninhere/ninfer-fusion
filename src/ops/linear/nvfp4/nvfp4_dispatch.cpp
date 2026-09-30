#include "ops/linear/nvfp4/nvfp4_dispatch.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear/qpn/qpn_arch_route.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearRoute : std::uint8_t {
    A16,
    W4A4,
};

// THE RUNG, AND WHY IT IS AN ARGUMENT RATHER THAN A SECOND READING (dl/gapclose, F-769).
//
// `w4a4_channel` is "this rung carries kind::mxf4nvf4", asked ONCE per call site through
// qpn::nvfp4_w4a4_channel_available(), which reads the SAME arch view the load-time gate read. It
// is an argument so that the workspace plan and the run cannot ask two different questions -- they
// are two call sites of THIS one function, which is what makes "the plan reserved what the run
// needs" true by construction rather than by a second derivation.
//
// WITH IT FALSE, EVERY SHAPE TAKES A16. Not just the small-token bands: on a rung whose only
// NVFP4 route is the FMA-pipe one, a W4A4 schedule is not a schedule this build can launch at all,
// so the answer is the tensor-core-free arm for every t -- which is the arm the arch_caps.h row's
// own text names (nvfp4_gemv / nvfp4_small_t), and exactly what a real card of that rung runs.
Nvfp4LinearRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                               LinearPolicy policy, std::int32_t tokens,
                               bool w4a4_channel) {
    if (tokens <= 0 || !is_nvfp4_linear_problem(output_rows, input_rows)) {
        throw std::invalid_argument("nvfp4 linear: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) { return Nvfp4LinearRoute::A16; }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    }
    // THE RUNG-AWARE GUARD, AND IT IS FIRST ON PURPOSE. Placed before the shape switch so that no
    // problem shape -- least of all Nvfp4Problem::GdnInput, which returned W4A4 at EVERY token
    // count and is the mechanism of the 76..119 hole -- can reach a kind::mxf4nvf4 kernel on a rung
    // that has no such instruction.
    if (!w4a4_channel) { return Nvfp4LinearRoute::A16; }

    switch (resolve_nvfp4_problem(output_rows, input_rows)) {
    case Nvfp4Problem::AttnInput:
        return tokens >= 4 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    case Nvfp4Problem::GdnInput:
        return Nvfp4LinearRoute::W4A4;
    case Nvfp4Problem::MlpGateUp:
        return tokens >= 5 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::Residual17408:
        return tokens >= 8 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    case Nvfp4Problem::MuseMlpGateUp:
    case Nvfp4Problem::MuseMlpDown:
    case Nvfp4Problem::MuseVocabulary:
        // Muse geometries keep the A16 (w4a16) route for every token count;
        // W4A4 schedules await Muse-specific measurement.
        return Nvfp4LinearRoute::A16;
    }
    throw std::logic_error("unreachable NVFP4 linear problem");
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t kChunk = kNvfp4LastSmallT;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor output_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            launch_nvfp4_decode(input_chunk, weight, output_chunk, stream);
        } else {
            launch_nvfp4_small_t(input_chunk, weight, output_chunk, stream);
        }
    }
}

} // namespace

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    }
    const bool w4a4_channel = qpn::nvfp4_w4a4_channel_available(qpn::current_device_sm());
    (void)resolve_route(output_rows, input_rows, policy, min_tokens, w4a4_channel);
    return resolve_route(output_rows, input_rows, policy, max_tokens, w4a4_channel) ==
                   Nvfp4LinearRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (!is_nvfp4_linear_problem(weight.n, weight.k) || x.ne[1] <= 0) {
        throw std::invalid_argument("nvfp4 linear: unsupported shape");
    }

    if (resolve_route(weight.n, weight.k, policy, x.ne[1],
                      qpn::nvfp4_w4a4_channel_available(qpn::current_device_sm())) ==
        Nvfp4LinearRoute::A16) {
        launch_a16(x, weight, out, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 linear requires caller workspace");
    }
    // ---------------------------------------------------------------------------
    // THE ARCHITECTURE ROUTE DECISION, and this is the engine's first one
    // ---------------------------------------------------------------------------
    // Before this call the route layer had no engine consumer at all (MEASURED 2026-09-18,
    // SIM-RUN: `select_route` had zero callers under src/ and apps/, `route_log_line()` had
    // zero callers anywhere, and six engine arms under NINFER_SIM_ARCH produced byte-identical
    // token ids because the simulated rung never ran). Two properties are load-bearing:
    //
    //  * ON A SHIPPING BUILD THIS IS A NO-OP. With no environment set, arch_view_for_device()
    //    returns Disabled, effective_sm == the card's own, and select_route(sm_120, NVFP4, ..)
    //    answers nvfp4-w4a4-tma -- the QPN arm needs caps::fp16_fallback_executable(), which
    //    is false on every rung whose m8n8k4 lowering is not HardwareMma884. So `use_qpn` is
    //    false, no stderr line is printed, and the native path below runs unchanged.
    //  * ON A SIMULATED RUN IT IS NOT SILENT. A refused request throws here rather than
    //    falling back to the real device, and a SELECTED QPN route throws in
    //    dispatch_qpn_fallback() naming what is missing -- it never quietly computes tokens on
    //    a route the run did not ask for, which is the exact trap the six identical arms fell
    //    into.
    const qpn::QpnArchRoute arch =
        qpn::select_qpn_arch_route(qpn::current_device_sm(), x.ne[1], weight.n, weight.k);
    if (arch.refused) {
        throw std::invalid_argument("nvfp4 W4A4 linear: the architecture request was REFUSED, "
                                    "so the native route is not taken either. " +
                                    arch.why);
    }
    if (arch.use_qpn) {
        // -----------------------------------------------------------------------------------
        // THE QPN ROUTE IS SELECTED AND NOT DISPATCHABLE. TAKE THE TENSOR-CORE-FREE ARM
        // INSTEAD OF THROWING. THIS IS THE EDIT dl/floorfix (F-720) MADE HERE, AND IT CLOSES A
        // MEASURED FALSE ADMISSION.
        // -----------------------------------------------------------------------------------
        // WHAT USED TO HAPPEN, measured rather than argued. The load-time gate
        // (evaluate_artifact_formats) admitted an NVFP4 artifact on sm_70/sm_75 through the
        // fp16 fallback, and this function then called qpn::dispatch_qpn_fallback(), which
        // THREW -- because the weights are in the artifact's own block-scale layout
        // ([N][K/2] codes + [N][K/16] e4m3 scales) and gemm_qpn consumes the qpn_prepack
        // permutation, with two ports missing that are ports and not flags (qpn_arch_route.h).
        // So the engine REFUSED an artifact its own gate called Supported: an admission that
        // never emitted a token.
        //
        // WHY SUBSTITUTING A ROUTE IS NOT THE PHANTOM THIS STACK FORBIDS. The phantom is a route
        // that names a kernel nothing can launch. This arm names kernels that ARE launched, on
        // the SHIPPING card, for the same weight bytes and the same op -- `launch_a16` is this
        // file's own non-W4A4 route, taken natively for Muse geometries at every token count and
        // for AttnInput at t < 4. Its kernels (nvfp4_gemv.cu at t == 1, nvfp4_small_t.cu above
        // it) read the artifact's OWN planes with FP32 FMA and NO tensor-core instruction --
        // per-TU census: dl/floorfix/out/tc_free_census/census.txt. So the alternative to
        // refusing is not a wrong answer; it is the correct answer on a narrower engine.
        // PRECISION: FP32 FMA is precision-UP against any fp16 route (e2m1 codes and e4m3
        // scales are exactly representable in FP32), which is the owner's first rule deciding
        // the tie against the FASTER QPN tensor-core route.
        //
        // THE REFUSAL IS NOT DELETED, IT IS MOVED ONE LAYER DOWN. dispatch_qpn_fallback() still
        // throws for every layout it cannot serve, and stays the guard for the day the two ports
        // land -- at which point this arm must be REMOVED and the QPN route used, because QPN is
        // the faster route and the 1cat-parity one.
        //
        // LOUD, ONCE PER PROCESS AND ON A SHAPE CHANGE, following the routing line's own
        // reasoning (a line per invocation at 20 tok/s would bury the marker it exists to make
        // visible).
        static thread_local std::string announced;
        if (announced != arch.route + "|" + arch.kernel) {
            announced = arch.route + "|" + arch.kernel;
            std::fprintf(stderr,
                         "[nvfp4] QPN W4A16 route SELECTED for sm_%d but NOT DISPATCHABLE with "
                         "the weights this artifact holds (the native -> qpn_prepack converter "
                         "and the bf16 -> fp16 step are both absent), so the TENSOR-CORE-FREE "
                         "A16 arm is taken: nvfp4_gemv / nvfp4_small_t, FP32 FMA over this "
                         "artifact's own planes. NOT the route the tables named; the route that "
                         "runs. See src/ops/linear/qpn/qpn_arch_route.h for the two ports.\n",
                         arch.effective_sm);
        }
        launch_a16(x, weight, out, stream);
        return;
    }
    // NOTE, so the next reader does not think an arm is missing: there is NO
    // `arch.tensor_core_free` arm here, and that is deliberate. dl/floorfix first wrote one --
    // keyed on select_route() answering ConservativeSimt for NVFP4 on every rung without
    // Cap::Mxf4Nvfp4BlockScale -- and DELETED it after measuring the consequence on the RUNTIME
    // rather than on the table: on 86/89/90/100/103 the op would then be admitted at load and
    // fall through to launch_nvfp4_w4a4, whose kind::mxf4nvf4 instruction those ISAs do not have.
    // A simulated run of that cell would emit tokens on THIS box (the cubin is sm_120a) while a
    // real Ada card faulted -- the false-positive class this stack exists to prevent. So the
    // NVFP4 row carries NO simt_kernel_evidence entry, the 76..119 hole stays open and REPORTED,
    // and the only NVFP4 degradation here is the `use_qpn` one above, which fires exactly on the
    // rungs where the tables named a route the engine cannot dispatch.
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    launch_nvfp4_w4a4(x, weight, out, scratch, stream);
}

} // namespace ninfer::ops::detail
