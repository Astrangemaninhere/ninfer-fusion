// test_bf16_fp16_plane.cpp -- THE LINK-TIME TIE for the fp16-plane build fact (dl/fp16route,
// F-736), plus the two properties the new arm has to hold that are properties of the TABLES
// rather than of a kernel.
//
// WHY THIS FILE TAKES AN ADDRESS AND DOES NOTHING WITH IT. caps::kFp16PlaneInBuild is a
// preprocessor constant that src/core/arch_caps.h turns into the first clause of
// caps::fp16_plane_executable(), and a -D is not evidence that the object file exists. So this
// target links ninfer_ops and REFERENCES ops::detail::launch_bf16_fp16_mma -- the launcher
// defined in src/ops/linear/bf16/bf16_mma_fp16.cu, the source src/CMakeLists.txt adds beside
// the definition. Removing the source line while leaving the definition therefore fails to
// LINK rather than producing a route that names a kernel the binary does not contain.
// That is the same tie (and the same reasoning) tests/test_arch_generic_fallback.cpp uses for
// ops::qpn::gemm_qpn -- see its header at line 20.
//
// WHAT IT DOES NOT DO: it does not launch a kernel, does not need a GPU, and does not claim any
// throughput. The arm's timing is a separate instrument (dl/fp16route/landq/bf16_fp16_bench.cu)
// and this file deliberately reports nothing about it.

#include "core/arch_caps.h"
#include "core/kernel_route.h"
#include "ops/linear/bf16/bf16_fp16_route.h"

#include <cstdio>
#include <string>

using ninfer::artifact::NumericFormat;
using ninfer::caps::Cap;
using ninfer::caps::fp16_plane_executable;
using ninfer::caps::KernelRoute;
using ninfer::caps::ProblemShape;
using ninfer::caps::RouteOutcome;
using ninfer::caps::select_route;
using ninfer::caps::simt_floor_executable;

static int g_failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

int main() {
    // ---- 1. THE TIE. Taking the address is the whole assertion; a build without the source
    //         line does not link. `volatile` keeps the compiler from folding it away, and the
    //         value is asserted non-null because a null function pointer to a defined function
    //         is not a thing the linker produces -- it is stated so the test says WHY it cannot
    //         be observed to fail.
    auto* volatile launcher = &ninfer::ops::detail::launch_bf16_fp16_mma;
    check(launcher != nullptr, "the fp16-plane launcher must be linked into this binary");
    check(ninfer::caps::kFp16PlaneInBuild,
          "and the macro that publishes the fact must be defined for this target");

    // ---- 2. THE REACHABLE SET IS {70, 75}, COMPUTED RATHER THAN RESTATED. Every rung of the
    //         ladder is asked, so a ladder change moves this set automatically instead of
    //         leaving a hand-written list behind.
    int reachable[64];
    int n_reachable = 0;
    for (std::size_t i = 0; i < ninfer::caps::kArchLadderSize; ++i) {
        const int sm = ninfer::caps::kArchLadder[i].sm;
        if (fp16_plane_executable(sm, NumericFormat::BF16, true)) {
            reachable[n_reachable++] = sm;
        }
    }
    check(n_reachable == 2 && reachable[0] == 70 && reachable[1] == 75,
          "the arm is reachable on exactly {70, 75} -- the rungs that HAVE fp16 tensor cores "
          "and do NOT cover Cap::Bf16Mma, which is the wall's own definition. Got " +
              std::to_string(n_reachable) + " rung(s)");

    // ---- 3. THE CHANNEL IS PER-RUNG AND MEASURED, and the route must name the one the rung
    //         has. sm_70 has exactly one fp16 mma form; sm_75 has two and the table picks the
    //         one that is still hardware above Turing.
    const auto* c70 = ninfer::caps::fp16_plane_channel_rung(70);
    const auto* c75 = ninfer::caps::fp16_plane_channel_rung(75);
    check(c70 != nullptr && c75 != nullptr, "both wall rungs must have a channel row");
    if (c70 != nullptr && c75 != nullptr) {
        check(c70->channel.find("m8n8k4") != std::string_view::npos,
              "sm_70's row must name mma.m8n8k4: Volta has no other fp16 mma form");
        check(c75->channel.find("m16n8k8") != std::string_view::npos,
              "sm_75's row must name mma.m16n8k8: Turing's native fp16 form, and the one that "
              "stays HMMA above Turing");
        check(c70->lowering == ninfer::caps::Fp16PlaneLowering::HardwareFp16Mma &&
                  c75->lowering == ninfer::caps::Fp16PlaneLowering::HardwareFp16Mma,
              "both wall rungs must be measured HARDWARE, or the arm would be a tensor-core "
              "label on an FMA-pipe execution");
        check(ninfer::caps::fp16_plane_channel_rung(80) == nullptr,
              "sm_80 must have NO channel row: its floor is met, the arm is never asked, and a "
              "row there would be a claim nothing exercises");
    }

    // ---- 4. THE ROUTE ANSWER, and it is the acceptance's own predicate: the CITATION.
    const ProblemShape verify{8, 4096, 4096};
    for (const int sm : {70, 75}) {
        const ninfer::caps::RouteChoice ch = select_route(sm, NumericFormat::BF16, verify);
        check(ch.outcome == RouteOutcome::Selected && ch.route == KernelRoute::MmaFp16Plane,
              "sm_" + std::to_string(sm) + " x bf16 must be Selected on the fp16-plane route");
        check(std::string(ch.kernel).find("bf16_mma_fp16.cuh") != std::string::npos,
              "and the route must CITATE the kernel file, not merely name a capability");
        check(std::string(ch.why).find("SIMULATED") == std::string::npos,
              "a table question is not a simulated answer");
    }

    // ---- 5. THE NEGATIVE CONTROLS. A rung that must not move, a format that must not move, and
    //         the not-blind control: the SAME questions with the build fact OFF.
    for (const int sub : {50, 52, 53, 60, 61, 62}) {
        const ninfer::caps::RouteChoice low = select_route(sub, NumericFormat::BF16, verify);
        check(low.route != KernelRoute::MmaFp16Plane,
              "sm_" + std::to_string(sub) + " has no tensor core and must not get the arm");
        check(low.route == KernelRoute::ConservativeSimt,
              "sm_" + std::to_string(sub) +
                  " keeps FFMA -- FFMA IS its native route and that is not a defect");
        check(!fp16_plane_executable(sub, NumericFormat::BF16, true),
              "and the predicate itself must say so, not only the arm");
        check(simt_floor_executable(sub, NumericFormat::BF16),
              "the sub-70 rungs must keep the FFMA rescue: the new clause must not have taken it");
    }
    // NVFP4, whose 18 sim cells must stay byte-identical to their PRE state (F-722).
    for (const int sm : {70, 75, 86, 89, 90, 100, 103}) {
        const ninfer::caps::RouteChoice nv = select_route(sm, NumericFormat::NVFP4, verify);
        check(nv.route != KernelRoute::MmaFp16Plane,
              "NVFP4 must never take the bf16-plane arm, on any rung");
        check(!fp16_plane_executable(sm, NumericFormat::NVFP4, true),
              "and the predicate must refuse it, because reading an e2m1 code plane is a "
              "different kernel");
    }
    // The four groupwise-int formats and fp8: same shape of control.
    for (const NumericFormat g :
         {NumericFormat::Q4G64_F16S, NumericFormat::Q5G64_F16S, NumericFormat::Q6G64_F16S,
          NumericFormat::W8G32_F16S, NumericFormat::FP8_E4M3FN_ROW_BF16S}) {
        check(!fp16_plane_executable(70, g, true),
              std::string(ninfer::artifact::format_name(g)) +
                  " has no fp16-plane arm: its bytes would have to be decoded to fp16 first");
        check(simt_floor_executable(70, g),
              std::string(ninfer::artifact::format_name(g)) +
                  " keeps the FFMA rescue on sm_70, which is where it was");
        check(select_route(70, g, verify).route == KernelRoute::ConservativeSimt,
              std::string(ninfer::artifact::format_name(g)) +
                  " must still be routed to the FFMA kernel on sm_70");
    }
    // THE NOT-BLIND CONTROL, on the predicate itself: the same (rung, format) with the build
    // fact OFF must answer the opposite. If it did not, the fact would be decorative.
    check(fp16_plane_executable(70, NumericFormat::BF16, true) &&
              !fp16_plane_executable(70, NumericFormat::BF16, false),
          "the build fact must be the difference between the two answers: PRE != POST");

    if (g_failures == 0) { std::printf("BF16_FP16_PLANE_OK\n"); }
    return g_failures == 0 ? 0 : 1;
}
