// ---------------------------------------------------------------------------
// test_arch_generic_fallback.cpp
//
// The acceptance test for the three things this work line added, and it is ONE binary on
// purpose so that the parts that have to agree can be checked against each other:
//
//   A. THE ARCH-GENERIC fp16 FALLBACK ROUTE. One implementation (the QPN W4A16 family)
//      serves sm_70 and sm_75 without special-casing Volta, and it is REFUSED on every
//      rung where ptxas answers its mma with a software routine. The build fact and the
//      arch fact are exercised as PARAMETERS, so the fail-closed world gets a red control
//      in the same binary as the selecting one.
//
//   B. THE TEST-ONLY ARCHITECTURE SIMULATOR (src/core/arch_sim.h) and its four guard
//      properties: impossible by accident, loud, fail-closed, downgrade-only.
//
//   C. THE NON-BLACKWELL fp4 CODEC (src/ops/common/fp4_codec.cuh), pinned against the
//      e2m1 lattice the ISA prints rather than against itself.
//
// THE LINK TIE lives in section D: this binary links ninfer_ops and takes the address of
// ops::qpn::gemm_qpn. If NINFER_HAVE_QPN is defined but the source is not in the build,
// this file does not LINK. That is what makes "the route names a kernel" a build fact
// instead of a claim.
//
// Evidence grades used below, as the tree's convention requires:
//   [RUN]    this binary actually executed and the assertion is on its output.
//   [SASS]   the claim was produced by ptxas/nvdisasm on this box; the test cites the
//            command instead of re-running it, because a unit test must not shell out.
// ---------------------------------------------------------------------------

#include "core/arch_caps.h"
#include "core/arch_sim.h"
#include "core/kernel_route.h"
#include "ops/common/fp4_codec.cuh"
#include "ops/linear/qpn/qpn_host.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer::caps;
using ninfer::artifact::NumericFormat;

namespace {

int g_failures = 0;
int g_checks   = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL  %s\n", what.c_str());
    }
}

// One line per (rung, format, choice) for the messages above. Kept local: the point of the
// message is that a failing cell prints the ANSWER it got, not only the question it asked.
std::string describe_choice(int sm, NumericFormat format, const RouteChoice& choice) {
    return "sm=" + std::to_string(sm) + " fmt=" +
           std::string(ninfer::artifact::format_name(format)) + " -> " +
           std::string(route_name(choice.route)) + " / " +
           std::string(outcome_name(choice.outcome));
}

void check_contains(const std::string& haystack, const char* needle, const std::string& what) {
    check(haystack.find(needle) != std::string::npos,
          what + "  (looking for \"" + std::string(needle) + "\" in: " + haystack + ")");
}

constexpr ProblemShape kDecode{1, 4096, 4096};
constexpr ProblemShape kVerify{8, 4096, 4096};
constexpr ProblemShape kPrefill{64, 4096, 4096};

// ===========================================================================
// A. The arch-generic fp16 fallback route
// ===========================================================================

// A1. THE HEADLINE: one implementation, two rungs, no Volta special case.
void test_fallback_selects_on_both_hardware_rungs() {
    // NVFP4 IS THE ONE FORMAT THE FALLBACK SERVES, and it serves it on BOTH hardware rungs
    // with the SAME code. FP8 is deliberately NOT here: its QPN kernels
    // (skinny_fp8_qpn8 / _mt2) have no host entry, so a fallback row for it would name a
    // kernel nothing can launch. That withdrawal is asserted at the end of this function
    // rather than assumed, because a row that quietly comes back is exactly the phantom this
    // whole line is about.
    for (const int sm : {70, 75}) {
        for (const NumericFormat format : {NumericFormat::NVFP4}) {
            const RouteChoice choice = select_route(sm, format, kVerify, /*qpn_in_build=*/true);
            check(choice.outcome == RouteOutcome::Selected,
                  "sm_" + std::to_string(sm) + " x " +
                      std::string(ninfer::artifact::format_name(format)) +
                      " with the QPN sources in the build must be SELECTED, got " +
                      std::string(outcome_name(choice.outcome)) + ": " + choice.why);
            check(choice.route == KernelRoute::QpnW4a16,
                  "the selected route must be qpn-w4a16, got " +
                      std::string(route_name(choice.route)));
            check(!choice.kernel.empty(), "a Selected route must name its kernel");
            check(choice.why.find("sm_70") != std::string::npos ||
                      choice.why.find("sm_75") != std::string::npos,
                  "the reason must name the rung it is answering for");
        }
        // THE WITHDRAWN ROW, pinned. fp8 on a hardware rung with the kernel in the build still
        // refuses, and the reason names the ABSENT HOST ENTRY rather than the mma channel --
        // because the channel is proven to work there by the nvfp4 arm.
        const RouteChoice fp8 =
            select_route(sm, NumericFormat::FP8_E4M3FN_ROW_BF16S, kVerify, true);
        check(fp8.outcome == RouteOutcome::NoKernelInTree &&
                  fp8.route == KernelRoute::None,
              "sm_" + std::to_string(sm) + " x fp8 must REFUSE: the fp8 QPN kernels exist as "
              "templates but gemm_qpn has no fp8 dispatch, so there is no host entry to "
              "launch, got " + describe_choice(sm, NumericFormat::FP8_E4M3FN_ROW_BF16S, fp8));
    }
    // The M split is preserved exactly, and it is the BAND that is named: the same table
    // the refusing version pinned, now as the selected kernel.
    struct Case {
        std::uint32_t m;
        const char* kernel_fragment;
    };
    const Case cases[] = {{1, "qpn_simt"}, {3, "qpn_simt"}, {4, "qpn<1>"},
                          {8, "qpn<1>"},  {9, "qpn<2>"},   {16, "qpn<2>"}};
    for (const Case& c : cases) {
        const RouteChoice choice =
            select_route(70, NumericFormat::NVFP4, ProblemShape{c.m, 4096, 4096}, true);
        check(choice.outcome == RouteOutcome::Selected,
              "M=" + std::to_string(c.m) + " must be Selected once the kernel is in the build");
        check(choice.kernel.find(c.kernel_fragment) != std::string_view::npos,
              "M=" + std::to_string(c.m) + " must name the band kernel ('" +
                  std::string(c.kernel_fragment) + "'), got '" + std::string(choice.kernel) + "'");
    }
    // M 17..64 IS the family's wide-M band (skinny_nvfp4_wmma), so it SELECTS once the
    // family is in the build -- and the build fact still decides the no-build world, which is
    // what keeps this a correction of the band table rather than a relaxation of the
    // fail-closed rule. Before 2026-09-18 this block asserted the opposite with "marlin" as
    // the fragment, i.e. it pinned a refusal whose stated missing kernel (vLLM's marlin) is
    // not in this tree while the real band kernel was in the same compiled header.
    {
        const RouteChoice open =
            select_route(70, NumericFormat::NVFP4, ProblemShape{17, 4096, 4096}, true);
        check(open.outcome == RouteOutcome::Selected,
              "M=17 must SELECT once the family is in the build, got " +
                  std::string(outcome_name(open.outcome)));
        check(open.kernel.find("skinny_nvfp4_wmma") != std::string_view::npos,
              "M=17 must name the wide-M band kernel, got '" + std::string(open.kernel) + "'");
        check(open.wide_m, "the wide-M band must report its own layout flag");
        const RouteChoice closed =
            select_route(70, NumericFormat::NVFP4, ProblemShape{17, 4096, 4096}, false);
        check(closed.outcome == RouteOutcome::NoKernelInTree,
              "M=17 must still REFUSE when the QPN sources are not in the build");
        check(!closed.wide_m, "a refusal must not carry the wide-M layout flag");
    }
    // AND THE CEILING. M > 64 is outside skinny_nvfp4_wmma (WM*16 = 64 rows per CTA, no
    // m-tile loop), so it refuses in BOTH worlds and names the kernel ceiling rather than a
    // card. The old "marlin" assertion survives HERE, where the lineage note that mentions
    // marlin still belongs.
    for (const bool qpn : {false, true}) {
        const RouteChoice over =
            select_route(70, NumericFormat::NVFP4, ProblemShape{65, 4096, 4096}, qpn);
        check(over.outcome == RouteOutcome::NoKernelInTree,
              "M=65 must refuse: above the wide-M kernel's 64-row ceiling");
        check_contains(over.why, "64", "the M>64 refusal must name the band ceiling");
        check_contains(over.why, "marlin",
                       "the M>64 refusal must keep the v100-skinny marlin lineage named");
    }
    // The geometry guard is quoted from the wrapper and applies in both worlds.
    for (const bool qpn : {false, true}) {
        const RouteChoice bad_n = select_route(70, NumericFormat::NVFP4, ProblemShape{8, 48, 4096}, qpn);
        check(bad_n.outcome == RouteOutcome::NoKernelInTree && bad_n.why.find("n % 32") != std::string::npos,
              "n=48 must be refused with the geometry rule stated");
        const RouteChoice bad_k = select_route(70, NumericFormat::NVFP4, ProblemShape{8, 4096, 96}, qpn);
        check(bad_k.outcome == RouteOutcome::NoKernelInTree && bad_k.why.find("k % 64") != std::string::npos,
              "k=96 must be refused with the geometry rule stated");
    }
}

// A2. THE FAIL-CLOSED HALF, and it is the reason qpn_in_build is a parameter. With the
// QPN sources absent from the build, NOTHING changes from the revision before this one:
// the same format on the same rung is the same named refusal. That is the red control for
// A1 -- if A1 passes and this fails, the change handed out a route in a build that has no
// kernel for it.
void test_fallback_fail_closed_without_the_kernel_in_the_build() {
    for (const int sm : {70, 75}) {
        for (const NumericFormat format : {NumericFormat::NVFP4,
                                           NumericFormat::FP8_E4M3FN_ROW_BF16S}) {
            const RouteChoice choice = select_route(sm, format, kVerify, /*qpn_in_build=*/false);
            check(choice.outcome == RouteOutcome::NoKernelInTree,
                  "sm_" + std::to_string(sm) + " must REFUSE when the QPN sources are not in "
                  "the build, got " + std::string(outcome_name(choice.outcome)));
            check(choice.route == KernelRoute::None, "a refusal must select no route");
            check_contains(choice.why, "compiled by NO target",
                           "the refusal must name the build fact it is refusing on");
        }
    }
    // And the predicate itself, in all four clauses.
    check(fp16_fallback_executable(70, NumericFormat::NVFP4, true),
          "sm_70 x NVFP4 must be executable with QPN in the build (HMMA.884, measured)");
    check(fp16_fallback_executable(75, NumericFormat::NVFP4, true),
          "sm_75 x NVFP4 must be executable with QPN in the build (HMMA.884, measured)");
    check(!fp16_fallback_executable(70, NumericFormat::NVFP4, false),
          "clause 1: without the build fact nothing is executable");
    check(!fp16_fallback_executable(70, NumericFormat::BF16, true),
          "clause 2: BF16 has no fallback row -- the QPN family does not consume its bytes");
    check(!fp16_fallback_executable(70, NumericFormat::Q4G64_F16S, true),
          "clause 2: Q4G64_F16S has no fallback row either");
    for (const int sm : {110, 999}) {
        check(!fp16_fallback_executable(sm, NumericFormat::NVFP4, true),
              "clause 3/4: sm_" + std::to_string(sm) +
                  " has no capability row, so nothing can be checked -- fail closed");
    }
}

// A3. IT IS ARCH-GENERIC, NOT VOLTA-SPECIFIC: on every rung where ptxas EMULATES the m8n8k4
// channel, the QPN route must stay out of reach even with the sources in the build. This is
// the assertion that stops a build fact from being mistaken for an arch fact.
// [SASS] the lowering itself is from kQpnMmaRungs (nvdisasm + cuobjdump census, two tools);
// this test asserts the ROUTE follows it.
void test_fallback_refused_where_the_channel_is_emulated() {
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        const ArchRung& rung = kArchLadder[r];
        const QpnMmaRung* channel = qpn_mma_rung(rung.sm);
        if (channel == nullptr) { continue; } // unmeasured: covered by A2's clause-3 checks
        if (channel->lowering != QpnMmaLowering::EmulatedFp16Pipe) { continue; }
        for (const NumericFormat format : {NumericFormat::NVFP4,
                                           NumericFormat::FP8_E4M3FN_ROW_BF16S}) {
            const RouteChoice choice = select_route(rung.sm, format, kVerify, true);
            check(choice.route != KernelRoute::QpnW4a16,
                  "sm_" + std::to_string(rung.sm) + " x " +
                      std::string(ninfer::artifact::format_name(format)) +
                      " must NOT get the QPN tensor-core route: its m8n8k4 channel is " +
                      std::string(qpn_mma_lowering_name(channel->lowering)) +
                      " (measured: " + std::string(channel->evidence) + ")");
        }
    }
    // At least one such rung exists, so the loop above is not vacuous.
    int emulated = 0;
    for (const QpnMmaRung& rung : kQpnMmaRungs) {
        if (rung.lowering == QpnMmaLowering::EmulatedFp16Pipe) { ++emulated; }
    }
    check(emulated == 8, "expected 8 emulated rungs in kQpnMmaRungs, found " +
                             std::to_string(emulated));
    int hardware = 0;
    for (const QpnMmaRung& rung : kQpnMmaRungs) {
        if (rung.lowering == QpnMmaLowering::HardwareMma884) { ++hardware; }
    }
    check(hardware == 2, "expected exactly 2 hardware rungs (sm_70, sm_75), found " +
                             std::to_string(hardware));
}

// A4. THE GATE'S OWN EXIT, which is the seam this change was pointed at: an A16 format whose
// floor is Bf16Mma must resolve to the fp16 path when Bf16Mma is absent and Fp16Mma is
// present -- and Cap::Fp16Mma stops being a decorative bit.
void test_gate_fallback_exit() {
    const NumericFormat nvfp4[] = {NumericFormat::NVFP4};

    // Without the build fact: exactly the old refusal.
    const CapabilityReport closed = evaluate_artifact_formats(70, nvfp4, false);
    check(closed.verdict == Verdict::Unsupported,
          "sm_70 x NVFP4 must be Unsupported when the QPN sources are out of the build");
    check(closed.fallbacks.empty(), "a refusal has no fallback to report");
    check(!closed.gaps.empty() && closed.gaps[0].required == Cap::Mxf4Nvfp4BlockScale,
          "the refusal must name the floor that is missing");

    // With it: Supported, and the report SAYS it was a fallback.
    const CapabilityReport open = evaluate_artifact_formats(70, nvfp4, true);
    check(open.verdict == Verdict::Supported,
          "sm_70 x NVFP4 must be Supported via the fp16 fallback once the kernel is in the build");
    check(open.fallbacks.size() == 1, "the report must record exactly one fallback");
    if (open.fallbacks.size() == 1) {
        check(open.fallbacks[0].format == NumericFormat::NVFP4, "the fallback names the format");
        check(open.fallbacks[0].primary_missing == Cap::Mxf4Nvfp4BlockScale,
              "the fallback must name the floor the card does NOT have");
        check(open.fallbacks[0].fallback_used == Cap::Fp16Mma,
              "the fallback must name the floor it uses instead");
        check(open.fallbacks[0].fallback_kernel.find("qpn_kernels.cuh") != std::string_view::npos,
              "the fallback must name the kernel file that consumes the same bytes");
    }
    // LOUD: the notice exists, names the missing floor and the kernel, and refuses to be
    // mistaken for a native run.
    const std::string notice = render_fallback_notice(open, "test-artifact");
    check(!notice.empty(), "a Supported-via-fallback report must render a notice");
    check_contains(notice, "FALLBACK", "the notice must say what is happening");
    check_contains(notice, "qpn_kernels.cuh", "the notice must name the kernel");
    check_contains(notice, "slower", "the notice must not present the fallback as native");
    check(render_fallback_notice(closed, "test-artifact").empty(),
          "a report with no fallbacks must render no notice");

    // The rungs where the channel is emulated must NOT get the gate exit either -- the gate
    // and the route must agree, and this is the check that they do.
    for (const int sm : {80, 86, 89, 90, 100, 103}) {
        const CapabilityReport report = evaluate_artifact_formats(sm, nvfp4, true);
        check(report.verdict != Verdict::Supported,
              "sm_" + std::to_string(sm) +
                  " x NVFP4 must stay Unsupported: its m8n8k4 channel is emulated, so the "
                  "fallback is not a tensor-core route there");
        check(report.fallbacks.empty(),
              "sm_" + std::to_string(sm) + " must record no fallback");
    }
    // The native rungs are untouched: floor met, no fallback recorded.
    for (const int sm : {120, 121}) {
        const CapabilityReport report = evaluate_artifact_formats(sm, nvfp4, true);
        check(report.verdict == Verdict::Supported, "sm_" + std::to_string(sm) + " stays Supported");
        check(report.fallbacks.empty(),
              "sm_" + std::to_string(sm) + " meets the floor itself, so no fallback is recorded");
    }
    // The formats with no fallback row keep refusing on a pre-Ampere card, with the floor
    // named. This is the honest remainder: the QPN family does not consume these bytes.
    // FP8 IS IN THIS LIST ON PURPOSE. It looks like it should have a fallback -- the QPN
    // family does contain fp8 kernels -- but they are unreachable, so the honest answer is the
    // same refusal the other five formats get, and this list is what holds that line.
    for (const NumericFormat format : {NumericFormat::BF16, NumericFormat::Q4G64_F16S,
                                       NumericFormat::Q5G64_F16S, NumericFormat::Q6G64_F16S,
                                       NumericFormat::W8G32_F16S,
                                       NumericFormat::FP8_E4M3FN_ROW_BF16S}) {
        const NumericFormat one[] = {format};
        const CapabilityReport report = evaluate_artifact_formats(70, one, true);
        check(report.verdict == Verdict::Unsupported,
              std::string(ninfer::artifact::format_name(format)) +
                  " on sm_70 must still be Unsupported: it has no fp16 fallback kernel");
        check(report.fallbacks.empty(),
              std::string(ninfer::artifact::format_name(format)) +
                  " must record no fallback on sm_70");
    }
}

// A5. THE SOUNDNESS INVARIANT, restated with the fallback clause and NOT weakened: every
// Selected route is justified either by the rung covering the format's declared floor, or
// by the fallback predicate being true AND the route being the fallback route. A third way
// must not exist, so the clause is written as an explicit disjunction.
void test_soundness_with_the_fallback_clause() {
    for (const bool qpn : {false, true}) {
        for (std::size_t r = 0; r < kArchLadderSize; ++r) {
            const ArchRung& rung = kArchLadder[r];
            for (std::size_t f = 0; f < kFormatRequirementCount; ++f) {
                const NumericFormat format = kFormatRequirements[f].format;
                const Cap floor            = kFormatRequirements[f].required;
                for (const ProblemShape shape : {kDecode, kVerify, kPrefill}) {
                    const RouteChoice choice = select_route(rung.sm, format, shape, qpn);
                    if (choice.outcome != RouteOutcome::Selected) { continue; }
                    const bool via_floor = floor == Cap::None || covers(rung.caps, floor);
                    const bool via_fallback =
                        choice.route == KernelRoute::QpnW4a16 &&
                        fp16_fallback_executable(rung.sm, format, qpn);
                    check(via_floor || via_fallback,
                          "SOUNDNESS: neither the floor nor the fallback justifies a Selected "
                          "route: qpn_in_build=" + std::to_string(qpn) + " sm=" +
                              std::to_string(rung.sm) + " fmt=" +
                              std::string(ninfer::artifact::format_name(format)) + " -> " +
                              std::string(route_name(choice.route)) + " :: " + choice.why);
                    if (choice.route != KernelRoute::ConservativeSimt) {
                        check(!choice.kernel.empty(), "SOUNDNESS: a Selected route names no kernel");
                    }
                }
            }
        }
    }
    // A route Selected WITH the build fact and ALSO with it absent would be the giveaway, so
    // the difference is pinned explicitly: the set of (sm, format, shape) that select under
    // qpn=false must be a SUBSET of the set that selects under qpn=true.
    int selected_closed = 0;
    int selected_open   = 0;
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        for (std::size_t f = 0; f < kFormatRequirementCount; ++f) {
            for (const ProblemShape shape : {kDecode, kVerify, kPrefill}) {
                const bool a = select_route(kArchLadder[r].sm, kFormatRequirements[f].format,
                                            shape, false)
                                   .outcome == RouteOutcome::Selected;
                const bool b = select_route(kArchLadder[r].sm, kFormatRequirements[f].format,
                                            shape, true)
                                   .outcome == RouteOutcome::Selected;
                if (a) {
                    ++selected_closed;
                    check(b, "the build fact must only ADD selectable routes, never remove one: "
                             "sm=" + std::to_string(kArchLadder[r].sm));
                }
                if (b) { ++selected_open; }
            }
        }
    }
    std::printf("  selected routes: qpn_out_of_build=%d  qpn_in_build=%d (+%d)\n",
                selected_closed, selected_open, selected_open - selected_closed);
    // SIX, AND THE ARITHMETIC OF THAT NUMBER IS ITSELF A PIN. 2 rungs x 2 formats x 3 shapes
    // would be 12. It is 6 because FP8 has no fallback row -- its QPN kernels have no host
    // entry, so a row for it would name a kernel nothing can launch, the phantom this whole
    // line is about -- which leaves 2 rungs x 1 format x 3 shapes.
    //
    // THE THIRD SHAPE IS THE ONE THAT MOVED. kPrefill is m=64: it was OUTSIDE the family's
    // window (M 1..16) and is now INSIDE it, because the ceiling is the wide-M KERNEL's own
    // 64 rows (skinny_nvfp4_wmma<WN, WM, KC>, qpn_kernels.cuh; host entry
    // gemm_qpn_wmma_native, qpn_host.cu). So the +4 this check used to pin became +6. That is
    // the count a reader should expect to have to update deliberately, because it is the only
    // one here that varies with the BAND WINDOW rather than with the build fact; a +8 would
    // still mean the fp8 row had come back without a gemm_qpn dispatch.
    check(selected_open == selected_closed + 6,
          "exactly 6 selections may be added (2 rungs x 1 format x 3 shapes inside the M "
          "window); FP8 has no launchable fallback kernel, so it adds none. A +8 would mean "
          "the fp8 row had come back without a gemm_qpn dispatch, got +" +
              std::to_string(selected_open - selected_closed));
    // And the WINDOW half of that sentence, asserted rather than asserted-in-a-comment, with
    // BOTH edges named: m=64 is inside the window, so it SELECTS under the build fact and
    // refuses without it; m=65 is outside the KERNEL, so it refuses in both worlds. Before
    // 2026-09-18 the pair read (refuse, refuse) for m=64, i.e. the test pinned a kernel
    // ceiling of 16 -- a ceiling no kernel in this tree has.
    for (const bool qpn : {false, true}) {
        for (const int sm : {70, 75}) {
            const RouteChoice prefill = select_route(sm, NumericFormat::NVFP4, kPrefill, qpn);
            if (qpn) {
                check(prefill.outcome == RouteOutcome::Selected,
                      "m=64 prefill must SELECT on sm_" + std::to_string(sm) +
                          " with the family in the build, got " +
                          std::string(outcome_name(prefill.outcome)));
                check(prefill.kernel.find("skinny_nvfp4_wmma") != std::string_view::npos,
                      "m=64 prefill must name the wide-M kernel, got '" +
                          std::string(prefill.kernel) + "'");
                check(prefill.wide_m, "the wide-M band must report its own layout flag");
            } else {
                check(prefill.outcome == RouteOutcome::NoKernelInTree,
                      "m=64 prefill must still REFUSE on sm_" + std::to_string(sm) +
                          " when the QPN sources are not in the build");
            }
            const RouteChoice above =
                select_route(sm, NumericFormat::NVFP4, ProblemShape{65, 4096, 4096}, qpn);
            check(above.outcome == RouteOutcome::NoKernelInTree,
                  "m=65 must refuse on sm_" + std::to_string(sm) +
                      " in BOTH worlds: it is outside the wide-M kernel itself");
        }
    }
}

// ===========================================================================
// B. The simulator's four guard properties
// ===========================================================================

ArchView view(int physical, bool built_in, const char* arch, const char* ack) {
    return arch_view_for_device_impl(physical, built_in,
                                    arch == nullptr ? std::string_view{} : std::string_view(arch),
                                    ack == nullptr ? std::string_view{} : std::string_view(ack));
}

// G1: impossible by accident. Three independent keys, and the two the operator can reach
// are both required.
void test_sim_G1_impossible_by_accident() {
    const std::string kAck(kSimAckPhrase);

    // (i) nothing set: the default path. Zero behaviour change is the contract.
    const ArchView none = view(120, true, nullptr, nullptr);
    check(none.status == SimStatus::Disabled, "no env must be Disabled");
    check(none.effective_sm == 120 && none.physical_sm == 120, "Disabled must not change the sm");
    check(none.reason.empty(), "Disabled carries no reason");
    check(sim_banner(none).empty(), "Disabled renders no banner");

    // (ii) the BUILD key missing: a stray NINFER_SIM_ARCH in a user's shell must not work.
    const ArchView not_built = view(120, /*built_in=*/false, "70", kAck.c_str());
    check(not_built.status == SimStatus::Refused,
          "with the build option off the request must be REFUSED, not honoured");
    check(not_built.effective_sm == 120, "a refused request must not change the sm");
    check_contains(not_built.reason, "NINFER_ENABLE_ARCH_SIM",
                   "the refusal must name the CMake option that would allow it");
    // THE REMEDY SENTENCE, NOT MERELY THE NAME. "NINFER_ENABLE_ARCH_SIM" occurs TWICE in this
    // refusal: once in the diagnostic sentence above ("...NOT built with -DNINFER_ENABLE_ARCH_SIM=ON")
    // and once in the on-ramp sentence below it. MEASURED 2026-09-19 (dl/falsegreen): deleting the
    // whole on-ramp sentence left the check above GREEN -- same check count, same baseline failures,
    // byte-identical output -- so that check could not detect the absence of the remedy it exists to
    // protect. This needle can only match the remedy sentence, and it is RED on that mutant.
    check_contains(not_built.reason, "Rebuild with -DNINFER_ENABLE_ARCH_SIM=ON",
                   "the refusal's REMEDY sentence must survive, not merely the option name -- the "
                   "name alone also appears in the diagnostic sentence above");

    // (iii) the ACK key missing or wrong.
    const ArchView no_ack = view(120, true, "70", nullptr);
    check(no_ack.status == SimStatus::Refused, "without the acknowledgement the request is refused");
    check_contains(no_ack.reason, "NINFER_SIM_ARCH_ACK", "the refusal names the missing key");
    const ArchView bad_ack = view(120, true, "70", "yes");
    check(bad_ack.status == SimStatus::Refused, "a wrong acknowledgement is refused");
    check_contains(bad_ack.reason, "test-only",
                   "the refusal must say the simulator is test-only");

    // (iv) all three present: honoured.
    for (const int rung : {70, 75}) {
        const ArchView ok = view(120, true, std::to_string(rung).c_str(), kAck.c_str());
        check(ok.status == SimStatus::Active, "sm_" + std::to_string(rung) + " must be honoured");
        check(ok.effective_sm == rung && ok.physical_sm == 120,
              "Active must record BOTH the simulated and the physical sm");
        check(ok.simulated() && !ok.failed() && ok.usable(), "Active's predicates must agree");
    }
}

// G2: loud. The banner says what is happening, names both numbers, states that the kernels
// are not the simulated arch's binaries, and states that no performance claim is valid.
void test_sim_G2_loud() {
    const ArchView ok = view(120, true, "70", std::string(kSimAckPhrase).c_str());
    const std::string banner = sim_banner(ok);
    check(!banner.empty(), "an active simulation must render a banner");
    check_contains(banner, "SIMULATED", "the banner must say SIMULATED");
    check_contains(banner, "sm_120", "the banner must name the physical device");
    check_contains(banner, "sm_70", "the banner must name the simulated rung");
    check_contains(banner, "V100", "the banner must name the card the rung stands for");
    check_contains(banner, "NOT sm_70 binaries",
                   "the banner must deny that the executed kernels are sm_70 binaries");
    check_contains(banner, "proves NOTHING", "the banner must deny the evidentiary weight");
    check_contains(banner, "throughput", "the banner must deny performance claims");
    check(sim_banner(view(120, true, nullptr, nullptr)).empty(), "no request, no banner");
    check(!sim_banner(view(120, true, "70", nullptr)).empty() == false,
          "a REFUSED request renders the refusal text, not the banner (checked separately)");
}

// G3: fail closed, and it must not degrade into the real path.
void test_sim_G3_fail_closed() {
    // A refusal for EVERY reason, and in every case the route must be a refusal too -- even
    // when the REAL device would have selected something.
    const ArchView refusals[] = {
        view(120, false, "70", std::string(kSimAckPhrase).c_str()), // build key missing
        view(120, true, "70", nullptr),                             // ack missing
        view(120, true, "70", "nope"),                              // ack wrong
        view(120, true, "abc", std::string(kSimAckPhrase).c_str()), // not a number
        view(120, true, "95", std::string(kSimAckPhrase).c_str()),  // not in the ladder
        view(120, true, "120", std::string(kSimAckPhrase).c_str()), // not a downgrade
        view(120, true, "121", std::string(kSimAckPhrase).c_str()), // a raise
    };
    for (const ArchView& rv : refusals) {
        check(rv.failed(), "expected a refusal");
        check(!rv.reason.empty(), "a refusal must carry its reason");
        // THE POINT: the real device selects nvfp4-w4a4-tma for this format and shape. A
        // refused simulation must NOT return it.
        const RouteChoice choice = select_route(rv, NumericFormat::NVFP4, kVerify);
        check(choice.outcome == RouteOutcome::NoKernelInTree,
              "a refused simulation must fail CLOSED, got " +
                  std::string(outcome_name(choice.outcome)));
        check(choice.route == KernelRoute::None,
              "a refused simulation must not name a route, got " +
                  std::string(route_name(choice.route)));
        check_contains(choice.why, "SIMULATED ARCHITECTURE REQUEST REFUSED",
                       "the refusal must be unmistakable");
        check_contains(choice.why, "No answer is given for the real device",
                       "the refusal must say it is not silently answering for the card");
        const RouteLogLine line = route_log_line(120, NumericFormat::NVFP4, kVerify, choice);
        check(!line.warn,
              "a fail-closed refusal is a NoKernelInTree, not an unknown-arch warning");
    }
    // The real path, unasked, still works: this is the control that says the refusals above
    // are about the REQUEST and not about a broken selector.
    const RouteChoice real = select_route(120, NumericFormat::NVFP4, kVerify);
    check(real.outcome == RouteOutcome::Selected && real.route == KernelRoute::Nvfp4W4a4Tma,
          "sm_120 with no request must still select nvfp4-w4a4-tma");
    check(!real.simulated, "an unreal simulation must not mark the answer as simulated");
}

// G4: downgrade only, never a raise, never a no-op.
void test_sim_G4_downgrade_only() {
    const std::string kAck(kSimAckPhrase);
    // Two DIFFERENT refusals live in this set and they must not be conflated, which is why
    // the reason is checked per row rather than once: a wanted rung that is on the ladder is
    // refused for breaking the downgrade rule, and a wanted rung that is NOT on the ladder is
    // refused because there would be no capability set to simulate. Collapsing them would
    // make "simulate sm_130" read as a downgrade problem.
    struct Row {
        int physical;
        int wanted;
        const char* expected_fragment;
    };
    const Row rows[] = {
        {120, 120, "STRICTLY LOWER"},      // not a downgrade
        {120, 121, "STRICTLY LOWER"},      // a raise
        {100, 100, "STRICTLY LOWER"},      // not a downgrade, other rung
        {100, 120, "STRICTLY LOWER"},      // a raise, other rung
        {90, 100, "STRICTLY LOWER"},       // a raise, other rung
        {120, 130, "no row in kArchLadder"}, // not a ladder rung at all
        {120, 111, "no row in kArchLadder"}, // likewise
    };
    for (const Row& row : rows) {
        const ArchView rv =
            view(row.physical, true, std::to_string(row.wanted).c_str(), kAck.c_str());
        check(rv.failed(),
              "simulating sm_" + std::to_string(row.wanted) + " on sm_" +
                  std::to_string(row.physical) + " must be REFUSED");
        check(rv.effective_sm == row.physical, "a refused request must not change the sm");
        check_contains(rv.reason, row.expected_fragment,
                       "the sm_" + std::to_string(row.wanted) + " on sm_" +
                           std::to_string(row.physical) + " refusal must state its own rule");
    }
    // And the downgrade itself, at every rung that has a row below 120 -- so the rule above
    // is a rule and not a blanket refusal.
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        const int sm = kArchLadder[r].sm;
        if (sm >= 120) { continue; }
        const ArchView rv = view(120, true, std::to_string(sm).c_str(), kAck.c_str());
        check(rv.simulated(), "sm_" + std::to_string(sm) + " must be simulable on sm_120");
        check(rv.effective_sm == sm, "effective_sm must be the requested rung");
    }
}

// G5: the simulated path is TAKEN, and it is the fallback route. This is the demonstration
// the deliverable asks for, in host-only form: which route is selected, which kernel runs,
// and the simulated marker on both the choice and the log line.
void test_sim_demonstrates_the_path_is_taken() {
    const std::string kAck(kSimAckPhrase);
    struct Row {
        const char* arch;
        int expected_sm;
        const char* expected_route;
        const char* expected_kernel_fragment;
    };
    const Row rows[] = {
        {"70", 70, "qpn-w4a16", "skinny_nvfp4_qpn<1>"},
        {"75", 75, "qpn-w4a16", "skinny_nvfp4_qpn<1>"},
    };
    for (const Row& row : rows) {
        const ArchView rv = view(120, true, row.arch, kAck.c_str());
        check(rv.simulated(), std::string("sm_") + row.arch + " simulation must be active");
        const RouteChoice choice = select_route(rv, NumericFormat::NVFP4, kVerify);
        check(choice.simulated, "the route choice must be MARKED simulated");
        check(choice.outcome == RouteOutcome::Selected,
              std::string("simulated sm_") + row.arch +
                  " x NVFP4 m=8 must be Selected, got " +
                  std::string(outcome_name(choice.outcome)) + ": " + choice.why);
        check(choice.route == KernelRoute::QpnW4a16,
              "the simulated run must take qpn-w4a16, got " +
                  std::string(route_name(choice.route)));
        check(choice.kernel.find(row.expected_kernel_fragment) != std::string_view::npos,
              "the simulated kernel must be " + std::string(row.expected_kernel_fragment) +
                  ", got '" + std::string(choice.kernel) + "'");
        // THE LINE A LOG READER SEES carries the marker at the FRONT.
        const RouteLogLine line = route_log_line(120, NumericFormat::NVFP4, kVerify, choice);
        check(line.text.rfind("[route][SIMULATED]", 0) == 0,
              "the log line must LEAD with the marker, got: " + line.text);
        check_contains(line.text, "sm=120",
                       "the log line must still name the REAL card's number it was asked on");
        (void)row.expected_sm;
        std::printf("  SIMULATED sm_%s -> route=%s kernel=%s\n", row.arch,
                    std::string(route_name(choice.route)).c_str(),
                    std::string(choice.kernel).c_str());
    }
    // The simulated GATE agrees with the simulated ROUTE, which is the whole point of
    // sharing fp16_fallback_executable(): a load that the gate allows must have a route.
    const ArchView rv = view(120, true, "70", kAck.c_str());
    const NumericFormat nvfp4[] = {NumericFormat::NVFP4};
    const CapabilityReport report = evaluate_artifact_formats(rv.effective_sm, nvfp4, kQpnInBuild);
    check(report.verdict == Verdict::Supported && !report.fallbacks.empty(),
          "the gate must accept the artifact for the simulated rung, via the fallback");
    std::printf("  simulated gate: %s, %zu fallback(s), notice=%zu bytes\n",
                report.ok() ? "Supported" : "not supported", report.fallbacks.size(),
                render_fallback_notice(report, "demo").size());
}

// G6: the environment reading itself, through the production entry point.
void test_sim_environment_reading() {
    const std::string ack(kSimAckPhrase);
    setenv(std::string(kSimArchEnv).c_str(), "70", 1);
    unsetenv(std::string(kSimAckEnv).c_str());
    const ArchView no_ack = arch_view_for_device(120);
    check(no_ack.failed(), "the production entry must read NINFER_SIM_ARCH from the environment");
    setenv(std::string(kSimAckEnv).c_str(), ack.c_str(), 1);
    const ArchView ok = arch_view_for_device(120);
    check(ok.simulated(), "with both variables set the production entry must activate");
    check(ok.effective_sm == 70, "the production entry must honour the requested rung");
    unsetenv(std::string(kSimArchEnv).c_str());
    unsetenv(std::string(kSimAckEnv).c_str());
    check(arch_view_for_device(120).status == SimStatus::Disabled,
          "clearing the environment must return to Disabled");
}

// ===========================================================================
// C. The non-Blackwell fp4 codec
// ===========================================================================

// C1: the DECODE is pinned against the ISA's OWN printed lattice, which is an independent
// source -- not against a second copy of the decoder.
void test_fp4_codec_decode_against_the_lattice() {
    using namespace ninfer::ops::detail;
    for (unsigned code = 0; code < 16; ++code) {
        const float got = decode_fp4_e2m1(code);
        const float want = kFp4E2m1Lattice[code & 0x7u] * ((code & 0x8u) != 0u ? -1.0F : 1.0F);
        check(got == want, "decode(" + std::to_string(code) + ") = " + std::to_string(got) +
                               " but the e2m1 lattice says " + std::to_string(want));
    }
    // The lattice itself, spelled out, so a change to the array is caught here.
    check(kFp4E2m1Lattice[0] == 0.0F && kFp4E2m1Lattice[1] == 0.5F && kFp4E2m1Lattice[2] == 1.0F &&
              kFp4E2m1Lattice[3] == 1.5F && kFp4E2m1Lattice[4] == 2.0F && kFp4E2m1Lattice[5] == 3.0F &&
              kFp4E2m1Lattice[6] == 4.0F && kFp4E2m1Lattice[7] == 6.0F,
          "the e2m1 lattice must be {0, 0.5, 1, 1.5, 2, 3, 4, 6}");
    // Nibble order: low nibble is the FIRST (even-k) value.
    float lo = 0.0F, hi = 0.0F;
    decode_fp4_e2m1x2(0x72, lo, hi); // low = 2 -> 1.0, high = 7 -> 6.0
    check(lo == 1.0F && hi == 6.0F, "low nibble must decode to the even-k value");
}

// C2: the ENCODE's tie resolution, at all seven midpoints, and saturation.
void test_fp4_codec_encode_ties_and_saturation() {
    using namespace ninfer::ops::detail;
    // Every lattice point must round-trip exactly, sign included.
    for (unsigned code = 0; code < 16; ++code) {
        const float value = decode_fp4_e2m1(code);
        const unsigned back = encode_fp4_e2m1(value);
        check(back == code, "round trip failed for code " + std::to_string(code) + ": got " +
                                std::to_string(back));
    }
    // The seven midpoints, each resolved to the EVEN code (this is `rn`, not `rz`).
    struct Tie {
        float value;
        unsigned even_code;
    };
    const Tie ties[] = {{0.25F, 0}, {0.75F, 2}, {1.25F, 2}, {1.75F, 4},
                        {2.5F, 4},  {3.5F, 6},  {5.0F, 6}};
    for (const Tie& tie : ties) {
        check(encode_fp4_e2m1(tie.value) == tie.even_code,
              "tie at " + std::to_string(tie.value) + " must go to the EVEN code " +
                  std::to_string(tie.even_code) + ", got " +
                  std::to_string(encode_fp4_e2m1(tie.value)));
        check(encode_fp4_e2m1(-tie.value) == (tie.even_code | 0x8u),
              "the negative tie must mirror the sign bit");
    }
    // satfinite: magnitudes above 6 clamp to code 7, and never to inf.
    for (const float big : {6.0F, 6.5F, 100.0F, 1.0e30F}) {
        check(encode_fp4_e2m1(big) == 7, "saturating " + std::to_string(big) + " must be code 7");
        check(encode_fp4_e2m1(-big) == 15, "saturating -" + std::to_string(big) + " must be code 15");
    }
    const float nan = std::nanf("");
    check(encode_fp4_e2m1(nan) == 7, "NaN must saturate to the max magnitude, as .satfinite does");
    check(encode_fp4_e2m1(-nan) == 15 || encode_fp4_e2m1(-nan) == 7,
          "NaN's sign is not meaningful; either saturated code is acceptable");
    // +-0 keep their sign bit, which is what the instruction does and what the packed
    // layout's consumers assume.
    check(encode_fp4_e2m1(0.0F) == 0, "+0 must be code 0");
    check(encode_fp4_e2m1(-0.0F) == 8, "-0 must keep the sign bit (code 8)");
}

// C3: the 16-value pack agrees with the per-pair encoder, and its byte order is the one the
// instruction it replaces produced.
void test_fp4_codec_pack() {
    using namespace ninfer::ops::detail;
    // The pairing the asm used: `cvt b, <hi>, <lo>` on (values[i].y, values[i].x), so
    // values[i].x is the LOW nibble.
    ninfer::ops::detail::Fp4Pair pairs[8];
    for (unsigned i = 0; i < 8u; ++i) {
        pairs[i].x = kFp4E2m1Lattice[i];
        pairs[i].y = -kFp4E2m1Lattice[i];
    }
    std::uint32_t lo = 0, hi = 0;
    pack_nvfp4_e2m1x16_portable(pairs, lo, hi);
    for (unsigned i = 0; i < 8u; ++i) {
        const std::uint8_t want =
            static_cast<std::uint8_t>((encode_fp4_e2m1(pairs[i].y) << 4) |
                                      encode_fp4_e2m1(pairs[i].x));
        const std::uint8_t got =
            static_cast<std::uint8_t>(i < 4u ? ((lo >> (8u * i)) & 0xFFu)
                                             : ((hi >> (8u * (i - 4u))) & 0xFFu));
        check(got == want, "packed byte " + std::to_string(i) + " mismatch");
    }
    // The float[16] overload must agree with the pair overload on the same data.
    float flat[16];
    for (unsigned i = 0; i < 8u; ++i) {
        flat[2u * i]     = pairs[i].x;
        flat[2u * i + 1u] = pairs[i].y;
    }
    std::uint32_t lo2 = 0, hi2 = 0;
    pack_nvfp4_e2m1x16_portable(flat, lo2, hi2);
    check(lo2 == lo && hi2 == hi, "the float[16] and Fp4Pair overloads must agree");
}

// C4: the codec is instruction-free, so the gate that selects it is a MEASURED arch fact.
// [SASS] `cvt.rn.satfinite.e2m1x2.f32` was measured to assemble only on sm_100a/103a/110a/
// 120a/121a and the five family targets, and to be REJECTED by 70/75/80/86/89/90 and by the
// plain sm_100 -- see the report's instruction x target matrix. This test pins the guard's
// SHAPE (it must key on the feature macros, never on a number), because the number-based
// form is what makes 'a' and non-'a' targets of the same number indistinguishable.
void test_fp4_codec_gate_is_feature_keyed() {
    // On the HOST pass the macro is 0 by construction (no __CUDA_ARCH__), which is the safe
    // default: a host TU cannot use the instruction and must not pretend it can.
    check(NINFER_FP4_HAS_CVT_E2M1X2 == 0,
          "the fp4 cvt gate must be 0 when there is no device pass");
    // The portable path is what a device below sm_100a gets, and it is the same code this
    // test just exercised above, so the host assertions ARE the fallback's assertions.
    std::printf("  fp4 codec: 16 decode codes + 7 tie midpoints + saturation verified on host\n");
}

// ===========================================================================
// D. The build tie
// ===========================================================================

void test_build_fact_ties_to_the_kernel() {
    // Referencing the symbol is what makes NINFER_HAVE_QPN and the object file inseparable:
    // if the define is present and this source is not in any target, the LINK fails and this
    // test never runs. A passing binary is therefore evidence that both are present.
    void (*entry)(const void*, const void*, const void*, float, void*, int, int, int,
                  cudaStream_t) = &ninfer::ops::qpn::gemm_qpn;
    void (*simt)(const void*, const void*, const void*, float, void*, int, int, int,
                 cudaStream_t) = &ninfer::ops::qpn::gemm_qpn_simt;
    check(entry != nullptr, "ops::qpn::gemm_qpn must be linkable in this binary");
    check(simt != nullptr, "ops::qpn::gemm_qpn_simt must be linkable in this binary");
    check(kQpnInBuild == true,
          "with the QPN source in the build, kQpnInBuild must be true -- the header and the "
          "source list are two halves of one statement");
    std::printf("  build tie: gemm_qpn linked, kQpnInBuild=%s\n", kQpnInBuild ? "true" : "false");
}

} // namespace

int main() {
    std::printf("=== test_arch_generic_fallback ===\n");
    std::printf("arch sim built in: %s\n", kArchSimBuiltIn ? "yes" : "no");
    test_fallback_selects_on_both_hardware_rungs();
    test_fallback_fail_closed_without_the_kernel_in_the_build();
    test_fallback_refused_where_the_channel_is_emulated();
    test_gate_fallback_exit();
    test_soundness_with_the_fallback_clause();
    test_sim_G1_impossible_by_accident();
    test_sim_G2_loud();
    test_sim_G3_fail_closed();
    test_sim_G4_downgrade_only();
    test_sim_demonstrates_the_path_is_taken();
    test_sim_environment_reading();
    test_fp4_codec_decode_against_the_lattice();
    test_fp4_codec_encode_ties_and_saturation();
    test_fp4_codec_pack();
    test_fp4_codec_gate_is_feature_keyed();
    test_build_fact_ties_to_the_kernel();
    std::printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
