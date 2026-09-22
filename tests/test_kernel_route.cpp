// test_kernel_route.cpp -- unit predicates for src/core/kernel_route.h.
//
// Host-only: kernel_route.h takes the compute capability as a PARAMETER, so every rung of
// kArchLadder is exercisable here, on a single-GPU host, with no CUDA include and no device
// query. That is the point: "what route does a V100 take for an nvfp4 artifact" is a table
// question.
//
// The checks are grouped so that each group fails for ONE reason:
//   1. SOUNDNESS INVARIANT   -- every Selected route is justified by a capability the rung
//      actually has, or by a route whose floor is lower than the format's. This is the
//      check that catches a table believing a card can do something it cannot.
//   2. NON-GATE PRINCIPLE    -- an unknown sm is answered WITHOUT a throw and WITH a reason
//      that names the missing kernel. It used to be "warns and falls back"; the fallback it
//      promised was the QPN SIMT kernel, which no target compiles, so what is left of the
//      principle is the part that was ever load-bearing: never throw, never guess a
//      neighbouring row's capability set, and always say what has to be written.
//   3. THE V100 ANSWER       -- the concrete verdicts the multi-device/V100 question needs.
//   4. M SPLIT AND GEOMETRY  -- the QPN family's M windows and its own slot guards.

#include "core/kernel_route.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::format_name;
using ninfer::artifact::NumericFormat;
using ninfer::caps::arch_rung;
using ninfer::caps::Cap;
using ninfer::caps::has_cap;
using ninfer::caps::kArchLadder;
using ninfer::caps::kArchLadderSize;
using ninfer::caps::kFormatRequirementCount;
using ninfer::caps::kFormatRequirements;
using ninfer::caps::KernelRoute;
using ninfer::caps::ProblemShape;
using ninfer::caps::RouteChoice;
using ninfer::caps::RouteOutcome;
using ninfer::caps::route_log_line;
using ninfer::caps::route_name;
using ninfer::caps::select_route;

int failures = 0;
int checks   = 0;
int reported = 0;
constexpr int kReportLimit = 25;

void check(bool condition, const std::string& what) {
    ++checks;
    if (!condition) {
        ++failures;
        if (reported < kReportLimit) {
            ++reported;
            std::cout << "FAIL: " << what << "\n";
        } else if (reported == kReportLimit) {
            ++reported;
            std::cout << "FAIL: ... further failures suppressed; the count below is exact\n";
        }
    }
}

const ProblemShape kDecode{1, 4096, 4096};
const ProblemShape kVerify{8, 4096, 4096};
const ProblemShape kPrefill{64, 4096, 4096};

std::string describe(int sm, NumericFormat format, const ProblemShape& shape,
                     const RouteChoice& choice) {
    return "sm=" + std::to_string(sm) + " fmt=" + std::string(format_name(format)) +
           " m=" + std::to_string(shape.m) + " -> " + std::string(route_name(choice.route)) +
           " [" + std::string(ninfer::caps::outcome_name(choice.outcome)) + "] " + choice.why;
}

// ---------------------------------------------------------------------------
// 1. Soundness invariant
// ---------------------------------------------------------------------------
//
// A route is Selected only when the card can really execute it. There are exactly two
// ways that can be true, and the invariant names both so a third cannot be smuggled in
// later:
//   (a) the rung's capability set covers the format's declared floor; or
//   (b) the format has NO tensor-core floor (scale words / index payloads); or
//   (c) the route IS the fp16 fallback route AND caps::fp16_fallback_executable() is true
//       for that (rung, format, build).
// Clause (c) USED TO BE DELETED, and the reason it was deleted is still the reason it is
// written so narrowly now: at the time it said "the route needs only fp16 mma rather than
// the format's floor" and that licensed a Selected route naming a kernel NO CMake target
// compiled -- a phantom. It is back only because the claim is now CHECKED rather than
// asserted: fp16_fallback_executable() requires the kernel to be in this build AND the
// rung's MEASURED m8n8k4 lowering to be HMMA.884. It is deliberately NOT "the rung has
// Cap::Fp16Mma" -- that weaker form is what put the sm-70 text on nine rungs in the first
// place, and no rung reaches (c) through it.
void test_soundness_invariant() {
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        const auto& rung = kArchLadder[r];
        for (std::size_t f = 0; f < kFormatRequirementCount; ++f) {
            const NumericFormat format = kFormatRequirements[f].format;
            const Cap floor            = kFormatRequirements[f].required;
            for (const ProblemShape& shape : {kDecode, kVerify, kPrefill}) {
                const RouteChoice choice = select_route(rung.sm, format, shape);
                if (choice.outcome != RouteOutcome::Selected) { continue; }
                const bool justified =
                    (floor == Cap::None) || ninfer::caps::covers(rung.caps, floor) ||
                    (choice.route == KernelRoute::QpnW4a16 &&
                     ninfer::caps::fp16_fallback_executable(rung.sm, format,
                                                           ninfer::caps::kQpnInBuild));
                check(justified,
                      "SOUNDNESS: a Selected route is not justified by the rung's "
                      "capabilities: " + describe(rung.sm, format, shape, choice));
                if (choice.route != KernelRoute::ConservativeSimt) {
                    check(!choice.kernel.empty(),
                          "SOUNDNESS: a Selected non-simt route names no kernel: " +
                              describe(rung.sm, format, shape, choice));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. The non-gate principle
// ---------------------------------------------------------------------------

void test_unknown_sm_never_throws_and_always_warns() {
    // The whole 0..200 range, including values far outside any real card, and the two
    // values that are KNOWN to be dangerous because a number is not evidence:
    //   95 -- not a shipped compute capability; a neighbouring row must NOT be assumed.
    //   120 vs 120a -- same number, different kernels (measured; tools/archkit/_GPU_MATRIX.md).
    for (int sm = 0; sm <= 200; ++sm) {
        for (std::size_t f = 0; f < kFormatRequirementCount; ++f) {
            for (const ProblemShape& shape : {kDecode, kVerify, kPrefill}) {
                const RouteChoice choice = select_route(sm, kFormatRequirements[f].format, shape);
                check(!choice.why.empty(),
                      "NON-GATE: an outcome with no reason at all: sm=" + std::to_string(sm));
                if (arch_rung(sm) == nullptr) {
                    // Unknown capability: the conservative answer, and NEVER a throw. A
                    // NoKernelInTree is acceptable here too, because the conservative
                    // fallback is itself bounded (the tensor-core-free SIMT path only
                    // reaches M 1..3) -- what is not acceptable is silence or a crash.
                    check(choice.outcome == RouteOutcome::UnknownArchFallback ||
                              choice.outcome == RouteOutcome::NoKernelInTree,
                          "NON-GATE: unknown sm produced a Selected route: " +
                              describe(sm, kFormatRequirements[f].format, shape, choice));
                    check(choice.outcome != RouteOutcome::Selected,
                          "NON-GATE: unknown sm must not report a plain Selected");
                }
            }
        }
    }
    // The specific values, named so a regression reports which one broke.
    // sm_88 USED TO BE IN THIS LIST. It stopped being an unknown capability when its row was
    // added to kArchLadder, and this test reddened -- which is the test doing its job. The
    // list now pins capabilities that genuinely have no row.
    for (const int sm : {95, 110, 111, 130, 0, 999}) {
        const RouteChoice choice = select_route(sm, NumericFormat::NVFP4, kDecode);
        check(arch_rung(sm) == nullptr, "ladder unexpectedly contains sm_" + std::to_string(sm));
        // The conservative answer for an unknown card is now a NAMED REFUSAL, not a warning
        // that promises a fallback: this build's only tensor-core-free GEMM is the QPN SIMT
        // kernel and no target compiles it, so UnknownArchFallback -- whose contract is "the
        // run may proceed on the fallback" -- would be a promise this build cannot keep.
        // What must NOT change is the part the non-gate principle is actually about: no
        // throw, a reason on every outcome, and the card named in the line.
        check(choice.outcome == RouteOutcome::NoKernelInTree,
              "NON-GATE: sm_" + std::to_string(sm) + " must be a named refusal, got " +
                  std::string(ninfer::caps::outcome_name(choice.outcome)));
        check(choice.route == KernelRoute::None,
              "NON-GATE: sm_" + std::to_string(sm) + " must select no route, got " +
                  std::string(route_name(choice.route)));
        check(!choice.warns(),
              "NON-GATE: sm_" + std::to_string(sm) +
                  " must not warn about a fallback this build cannot deliver");
        check(choice.why.find("skinny_nvfp4_qpn_simt") != std::string::npos,
              "NON-GATE: the refusal must name the missing kernel, got: " + choice.why);
        const auto line = route_log_line(sm, NumericFormat::NVFP4, kDecode, choice);
        check(!line.warn, "NON-GATE: the log line must not be marked warn when it refuses");
        check(line.text.find("sm=" + std::to_string(sm)) != std::string::npos,
              "NON-GATE: the log line must name the card");
    }
    // A malformed-but-known arch is still a table question, not a crash: sm_75 is real and
    // its row exists, so it must NOT be reported as an unknown arch.
    check(!select_route(75, NumericFormat::NVFP4, kDecode).warns(),
          "sm_75 has a ladder row and must not warn about an unknown arch");
}

void test_every_rung_has_fp16_mma_so_the_no_tensor_core_branch_is_unreachable() {
    // kernel_route.h has a branch for "the rung has no tensor core at all". With the ladder
    // as it stands that branch is UNREACHABLE, and this pins that fact instead of pretending
    // the branch is covered. If a future rung drops fp16 mma, this check fails first and says
    // which rung, so the branch's coverage becomes a deliberate decision rather than an
    // accident.
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        check(has_cap(kArchLadder[r].caps, Cap::Fp16Mma),
              std::string("the no-tensor-core branch of select_route is currently "
                          "unreachable, and rung sm_") +
                  std::to_string(kArchLadder[r].sm) + " (" +
                  std::string(kArchLadder[r].label) +
                  ") broke that: it has no fp16 mma, so it now exercises the branch");
    }
}

// ---------------------------------------------------------------------------
// 3. The V100 answer
// ---------------------------------------------------------------------------

void test_v100_verdicts() {
    const int kVolta = 70;
    check(arch_rung(kVolta) != nullptr, "sm_70 must have a ladder row");
    check(arch_rung(kVolta)->caps == Cap::Fp16Mma, "sm_70 must be fp16-mma-only");

    // THE HEADLINE ANSWER, WHICH CHANGED ON 2026-09-17 AND FOR A MEASURED REASON. An nvfp4
    // artifact on a V100 is now SELECTED onto the QPN W4A16 route, because
    // ops/linear/qpn/qpn_host.cu is in ninfer_ops (src/CMakeLists.txt) and this rung's
    // mma.sync.m8n8k4 channel is a HARDWARE instruction -- HMMA.884.F32.F32.STEP0..3, with
    // ZERO calls into ptxas's software FFMA routine, in a real `nvcc -cubin -arch=sm_70`
    // produced by this tree's own qpn_host.cu. The route may therefore be claimed, and the
    // claim has to name the kernel that will run rather than the card that asked.
    const RouteChoice nvfp4 = select_route(kVolta, NumericFormat::NVFP4, kVerify);
    check(nvfp4.outcome == RouteOutcome::Selected,
          "V100 x nvfp4 m=8 must be SELECTED onto the fp16 fallback, got " +
              describe(kVolta, NumericFormat::NVFP4, kVerify, nvfp4));
    check(nvfp4.route == KernelRoute::QpnW4a16,
          "V100 x nvfp4 must take the QPN W4A16 route, got " +
              std::string(route_name(nvfp4.route)));
    for (const std::string_view needle : {std::string_view("skinny_nvfp4_qpn"),
                                          std::string_view("qpn_host.cu"),
                                          std::string_view("HMMA.884")}) {
        check(nvfp4.why.find(needle) != std::string::npos,
              "the V100 x nvfp4 selection must name the kernel AND the measurement ('" +
                  std::string(needle) + "'), got: " + nvfp4.why);
    }
    // THE RED CONTROL, IN THE SAME BINARY. Without the kernel in the build the very same
    // question must be the very same refusal it was before this change -- so "selected" is
    // the build fact talking and not a blanket relaxation.
    const RouteChoice nvfp4_no_build =
        select_route(kVolta, NumericFormat::NVFP4, kVerify, /*qpn_in_build=*/false);
    check(nvfp4_no_build.outcome == RouteOutcome::NoKernelInTree &&
              nvfp4_no_build.route == KernelRoute::None,
          "V100 x nvfp4 must go back to a named refusal when the QPN sources are not in the "
          "build, got " + describe(kVolta, NumericFormat::NVFP4, kVerify, nvfp4_no_build));
    for (const std::string_view needle : {std::string_view("skinny_nvfp4_qpn_simt"),
                                          std::string_view("src/ops/linear/qpn/"),
                                          std::string_view("compiled by NO target")}) {
        check(nvfp4_no_build.why.find(needle) != std::string::npos,
              "the no-kernel refusal must name the missing kernel and the build fact ('" +
                  std::string(needle) + "'), got: " + nvfp4_no_build.why);
    }

    // FP8 IS THE ASYMMETRIC CASE, AND IT IS ASSERTED RATHER THAN ASSUMED. The route table
    // once said the published NVFP4/FP8 mixed weights both run through QPN on sm_70. The
    // NVFP4 half is now true. The FP8 half is NOT, and the reason is not the card: the fp8 arm
    // of the QPN family (skinny_fp8_qpn8 / _mt2) exists as a template with NO HOST ENTRY --
    // gemm_qpn dispatches the nvfp4 kernels only -- so kFormatRequirements carries Cap::None
    // for its fallback floor and the format refuses. Selecting it would name a kernel nothing
    // can launch, which is the phantom this whole line exists to remove. Both worlds are
    // asserted, and both must refuse, because "the QPN sources are in the build" must not be
    // able to close a question about a FORMAT.
    for (const bool qpn : {false, true}) {
        const RouteChoice fp8 =
            select_route(kVolta, NumericFormat::FP8_E4M3FN_ROW_BF16S, kVerify, qpn);
        check(fp8.outcome == RouteOutcome::NoKernelInTree && fp8.route == KernelRoute::None,
              "V100 x fp8 must REFUSE in both worlds (no gemm_qpn fp8 dispatch, so no "
              "launchable kernel), got " +
                  describe(kVolta, NumericFormat::FP8_E4M3FN_ROW_BF16S, kVerify, fp8) +
                  " -- qpn_in_build=" + std::to_string(qpn));
        check(fp8.why.find("RIGHT kind of rung") != std::string::npos,
              "the fp8 refusal must say the RUNG is not the problem (the nvfp4 arm proves the "
              "channel is hardware there), got: " + fp8.why);
    }

    // THE PIN THAT REPLACES "no rung may report the QPN route", AND IT IS STRONGER. With the
    // kernel IN the build, the QPN route must still be out of reach on every rung whose
    // measured m8n8k4 lowering is EMULATED -- which is the arch fact that was the real
    // reason all along, and the one a build-fact pin could never express.
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        const auto* row = ninfer::caps::qpn_mma_rung(kArchLadder[r].sm);
        if (row == nullptr ||
            row->lowering != ninfer::caps::QpnMmaLowering::EmulatedFp16Pipe) {
            continue;
        }
        for (std::size_t f = 0; f < kFormatRequirementCount; ++f) {
            for (const ProblemShape& shape : {kDecode, kVerify, kPrefill}) {
                const RouteChoice choice = select_route(kArchLadder[r].sm,
                                                        kFormatRequirements[f].format, shape,
                                                        /*qpn_in_build=*/true);
                check(choice.route != KernelRoute::QpnW4a16,
                      "no EMULATED rung may report the QPN W4A16 route even with the kernel "
                      "in the build: " +
                          describe(kArchLadder[r].sm, kFormatRequirements[f].format, shape,
                                   choice));
            }
        }
    }

    // The honest refusals. groupwise-int on sm_70 has NO fp16 route in this tree, and the
    // reason must name the missing kernel rather than the card.
    const RouteChoice w8 = select_route(kVolta, NumericFormat::W8G32_F16S, kVerify);
    check(w8.outcome == RouteOutcome::NoKernelInTree,
          "V100 x groupwise-int must be a named refusal, got " +
              describe(kVolta, NumericFormat::W8G32_F16S, kVerify, w8));
    check(w8.why.find("mma_s8") != std::string::npos,
          "V100 x groupwise-int must name the missing kernel (mma_s8 port), got: " + w8.why);

    // bf16 on Volta: no bf16 mma exists on sm_70 at all.
    const RouteChoice bf16 = select_route(kVolta, NumericFormat::BF16, kVerify);
    check(bf16.outcome == RouteOutcome::NoKernelInTree,
          "V100 x bf16 must refuse (Volta has no bf16 mma), got " +
              describe(kVolta, NumericFormat::BF16, kVerify, bf16));

    // And the same questions at the other end of the ladder, so a fix for V100 cannot
    // silently break the native target.
    const RouteChoice native = select_route(120, NumericFormat::NVFP4, kVerify);
    check(native.outcome == RouteOutcome::Selected &&
              native.route == KernelRoute::Nvfp4W4a4Tma,
          "sm_120 x nvfp4 must stay on the W4A4 TMA route, got " +
              describe(120, NumericFormat::NVFP4, kVerify, native));

    // sm_89/sm_90 have the PLAIN fp8 form but not kind::f8f6f4, so they must NOT be routed
    // to the A8 kernel. This is the ladder's own note: "today only the A16 fp8 route ... is
    // executable on sm_89/sm_90".
    for (const int sm : {89, 90}) {
        const RouteChoice choice = select_route(sm, NumericFormat::FP8_E4M3FN_ROW_BF16S, kVerify);
        check(choice.route == KernelRoute::Fp8A16Bf16,
              "sm_" + std::to_string(sm) + " x fp8 must take the A16 route, got " +
                  std::string(route_name(choice.route)));
    }
    for (const int sm : {100, 103, 120, 121}) {
        const RouteChoice choice = select_route(sm, NumericFormat::FP8_E4M3FN_ROW_BF16S, kVerify);
        check(choice.route == KernelRoute::Fp8A8KindF8f6f4,
              "sm_" + std::to_string(sm) + " x fp8 must take the kind::f8f6f4 route, got " +
                  std::string(route_name(choice.route)));
    }
}

// ---------------------------------------------------------------------------
// 4. M split and the QPN geometry guards
// ---------------------------------------------------------------------------

void test_m_split_and_geometry() {
    struct Case {
        std::uint32_t m;
        std::string_view kernel_fragment;
    };
    // `in_window` splits the table, and the split is the point: M 1..16 names the BAND it
    // SELECTS (the kernels are in a target now), while M >= 17 names the band the FAMILY does
    // not have -- registering sources could not help, so it refuses in both worlds. The
    // window itself is unchanged from the refusing version, so a widened or narrowed split
    // still has to come here and say so.
    struct Row {
        std::uint32_t m;
        const char* fragment;
        bool in_window;
    };
    // M 17..64 IS A BAND TOO. This row read {17,"marlin",false},{64,"marlin",false} until
    // 2026-09-18, i.e. the test pinned a refusal that named a kernel absent from the tree
    // (marlin) while the family's own wide-M kernel, skinny_nvfp4_wmma, sat in the same
    // compiled header. The window now reaches the KERNEL's ceiling (WM*16 = 64) and the
    // band above it is the fail-closed control.
    const Row rows[] = {
        {1, "qpn_simt", true}, {3, "qpn_simt", true}, {4, "qpn<1>", true},
        {8, "qpn<1>", true},   {9, "qpn<2>", true},   {16, "qpn<2>", true},
        {17, "wmma", true},    {64, "wmma", true},
    };
    for (const Row& c : rows) {
        const ProblemShape shape{c.m, 4096, 4096};
        const RouteChoice choice = select_route(70, NumericFormat::NVFP4, shape);
        if (c.in_window) {
            check(choice.outcome == RouteOutcome::Selected,
                  "M=" + std::to_string(c.m) + " on sm_70 must be SELECTED onto the QPN "
                  "family's band kernel, got " + describe(70, NumericFormat::NVFP4, shape, choice));
            check(choice.kernel.find(c.fragment) != std::string_view::npos,
                  "M=" + std::to_string(c.m) + " must select the band kernel ('" +
                      std::string(c.fragment) + "'), got '" + std::string(choice.kernel) + "'");
            // AND THE RED CONTROL: the band is a band of a kernel that must be in the build,
            // which is what M>=17 is not.
            const RouteChoice closed =
                select_route(70, NumericFormat::NVFP4, shape, /*qpn_in_build=*/false);
            check(closed.outcome == RouteOutcome::NoKernelInTree,
                  "M=" + std::to_string(c.m) + " must refuse when the kernel is not in the build");
            check(closed.why.find(c.fragment) != std::string::npos,
                  "the no-build refusal for M=" + std::to_string(c.m) +
                      " must name the same band kernel ('" + std::string(c.fragment) +
                      "'), got: " + closed.why);
        } else {
            for (const bool qpn : {false, true}) {
                const RouteChoice closed =
                    select_route(70, NumericFormat::NVFP4, shape, qpn);
                check(closed.outcome == RouteOutcome::NoKernelInTree,
                      "M=" + std::to_string(c.m) + " must refuse in BOTH worlds: it is a band "
                      "the QPN family LACKS, not one the build lacks (qpn_in_build=" +
                      std::to_string(qpn) + ")");
                check(closed.why.find(c.fragment) != std::string::npos,
                      "M=" + std::to_string(c.m) + " must name the missing band ('" +
                          std::string(c.fragment) + "'), got: " + closed.why);
            }
        }
    }

    // THE BAND'S OWN CEILING, which is a kernel fact and therefore refuses in BOTH worlds:
    // skinny_nvfp4_wmma stages WM*16 = 64 rows per CTA with no m-tile loop, so 65 is outside
    // the kernel no matter what is compiled in. A widened window has to come here.
    for (const bool qpn : {false, true}) {
        const ProblemShape over{65, 4096, 4096};
        const RouteChoice choice = select_route(70, NumericFormat::NVFP4, over, qpn);
        check(choice.outcome == RouteOutcome::NoKernelInTree,
              "M=65 must refuse: it is above the wide-M band's kernel ceiling "
              "(qpn_in_build=" +
                  std::to_string(qpn) + "), got " +
                  std::string(outcome_name(choice.outcome)));
        check(choice.why.find("64") != std::string::npos,
              "the M=65 refusal must name the ceiling it is refusing on, got: " + choice.why);
    }

    // The QPN slot guard, quoted from the wrapper: K % 64 == 0 and N % 32 == 0.
    const RouteChoice bad_n = select_route(70, NumericFormat::NVFP4, ProblemShape{8, 48, 4096});
    check(bad_n.outcome == RouteOutcome::NoKernelInTree,
          "n=48 (not a multiple of 32) must be refused, got " +
              describe(70, NumericFormat::NVFP4, ProblemShape{8, 48, 4096}, bad_n));
    check(bad_n.why.find("n % 32") != std::string::npos,
          "the n guard must state its rule, got: " + bad_n.why);
    const RouteChoice bad_k = select_route(70, NumericFormat::NVFP4, ProblemShape{8, 4096, 96});
    check(bad_k.outcome == RouteOutcome::NoKernelInTree,
          "k=96 (not a multiple of 64) must be refused, got " +
              describe(70, NumericFormat::NVFP4, ProblemShape{8, 4096, 96}, bad_k));
    check(bad_k.why.find("k % 64") != std::string::npos,
          "the k guard must state its rule, got: " + bad_k.why);
}

void test_non_tensor_core_formats_have_no_route() {
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        for (const NumericFormat format : {NumericFormat::FP32, NumericFormat::I32}) {
            const RouteChoice choice = select_route(kArchLadder[r].sm, format, kDecode);
            check(choice.outcome == RouteOutcome::Selected &&
                      choice.kernel == "(no tensor-core operand)",
                  "sm_" + std::to_string(kArchLadder[r].sm) + " x " +
                      std::string(format_name(format)) +
                      " must select the no-operand route, got " +
                      describe(kArchLadder[r].sm, format, kDecode, choice));
        }
    }
}

void test_route_log_line_is_greppable() {
    // BOTH WORLDS ARE PINNED, and the pair is what makes the line trustworthy: the same
    // (card, format, shape) produces a SELECTED line when the kernel is in the build and a
    // NAMED REFUSAL when it is not. A line that claimed qpn-w4a16 while the kernel was in no
    // target was the phantom this whole change is about, so the refusal's negative half is
    // asserted as ABSENT rather than left unpinned.
    const RouteChoice selected = select_route(70, NumericFormat::NVFP4, kVerify);
    const std::string line =
        ninfer::caps::route_selection_line(70, NumericFormat::NVFP4, kVerify, selected);
    for (const std::string_view needle : {"[route]", "sm=70", "format=NVFP4", "m=8",
                                          "qpn-w4a16", "skinny_nvfp4_qpn", "outcome=selected"}) {
        check(line.find(needle) != std::string::npos,
              "the SELECTED line must contain '" + std::string(needle) + "': " + line);
    }
    check(line.rfind("[route][SIMULATED]", 0) != 0,
          "an unreal simulation must not mark the line SIMULATED: " + line);
    const RouteChoice refused =
        select_route(70, NumericFormat::NVFP4, kVerify, /*qpn_in_build=*/false);
    const std::string refused_line =
        ninfer::caps::route_selection_line(70, NumericFormat::NVFP4, kVerify, refused);
    for (const std::string_view needle : {"[route]", "sm=70", "format=NVFP4", "m=8",
                                          "skinny_nvfp4_qpn_simt",
                                          "outcome=no-kernel-in-tree"}) {
        check(refused_line.find(needle) != std::string::npos,
              "the REFUSAL line must contain '" + std::string(needle) + "': " + refused_line);
    }
    for (const std::string_view needle : {"qpn-w4a16", "outcome=selected"}) {
        check(refused_line.find(needle) == std::string::npos,
              "the REFUSAL line must NOT claim '" + std::string(needle) + "': " + refused_line);
    }
}

// ---------------------------------------------------------------------------
// 5. THE ARCH GRADING OF THE QPN ARM -- DECLARED vs MEASURED
// ---------------------------------------------------------------------------
//
// The QPN arm of select_route() used to be reached by every rung with fp16 mma and it handed
// all of them the SAME answer: the sm-70 text. The SASS says that text is true on two of those
// rungs and false on the rest. In the QPN family's ONLY tensor-core channel --
// `mma.sync.aligned.m8n8k4` -- ptxas emits HMMA.884 on sm_70/sm_75 and, from sm_80 on, emits a
// CALL to one of seven ~88-instruction FFMA routines it places in the same cubin (32 FFMA
// each, HMMA = 0). Measured, both with nvdisasm and with cuobjdump as an independent tool;
// src/core/kernel_route.h kQpnMmaRungs carries the numbers and
// /home/user/scratch/QPN-GRADE/logs/T1g_corrected.txt carries the raw census.
//
// THIS GROUP SEPARATES "DECLARED" FROM "MEASURED", because the failure mode it guards against
// is a table row for an arch nobody measured -- a tier asserted with no instrument behind it.
// Three consequences, all deliberate:
//   * kMeasuredQpnRungs below is a HARDCODED list of the rungs whose SASS was read. The table
//     is checked against it in BOTH directions, so an unmeasured row is a failure and a
//     discarded measurement is a failure.
//   * A rung the arm can be REACHED on but which has no census is NOT skipped -- it is
//     asserted FAIL-CLOSED: the refusal must refuse, and must name the missing measurement.
//     There is no `continue` that turns "unmeasured" into a pass.
//   * A rung the arm can NOT be reached on must have NO row at all. sm_120/sm_121 are that
//     cell: they carry Mxf4Nvfp4BlockScale, so NVFP4 meets its floor and the arm is
//     short-circuited before any table lookup. A tier for them would be a claim that nothing
//     exercises -- and the runtime key cannot even tell sm_120 from sm_120a.
//
// The load-bearing half is (b), which is why the cells there are the ones with the injections
// aimed at them: the refusal must QUOTE the row's own census verbatim. That cell cannot pass
// if the arm stops consulting the table, and it cannot pass if a row is flipped to the wrong
// tier. "outcome == NoKernelInTree" alone would be green both before and after the change --
// a probe true by construction.
//
// THE CRITERION SET IS {rc, checks}, NOT rc. Two measured holes in rc as a criterion:
//   (a) kReportLimit above truncates the FAIL LIST after 25 lines while ++failures keeps
//       counting, so counting FAIL lines under-reports a large failure set as 26;
//   (b) deleting a test_*() call from main() leaves rc = 0 and prints NO failure at all.
// (b) is closed in main() by counting the groups that actually ran against a constant.
const int kMeasuredQpnRungs[] = {70, 75, 80, 86, 87, 88, 89, 90, 100, 103};

void test_qpn_arm_is_graded_by_the_measured_mma_lowering() {
    using ninfer::caps::covers;
    using ninfer::caps::format_requirement;
    using ninfer::caps::kQpnMmaRungCount;
    using ninfer::caps::kQpnMmaRungs;
    using ninfer::caps::qpn_arm_reachable;
    using ninfer::caps::qpn_mma_lowering_name;
    using ninfer::caps::qpn_mma_rung;
    using ninfer::caps::QpnMmaLowering;

    // (a) The table must be exactly the measured set, in both directions, and every row must
    // carry both its measured target and its numbers -- otherwise the key's '-a' folding is
    // unqualified and the row reads as a measurement it is not.
    for (const int sm : kMeasuredQpnRungs) {
        check(qpn_mma_rung(sm) != nullptr,
              "sm_" + std::to_string(sm) + " is listed as MEASURED but kQpnMmaRungs has no row "
              "for it: a measurement that reached the list and not the table");
    }
    for (std::size_t r = 0; r < kQpnMmaRungCount; ++r) {
        const int sm = kQpnMmaRungs[r].sm;
        bool declared = false;
        for (const int m : kMeasuredQpnRungs) { if (m == sm) { declared = true; break; } }
        check(declared,
              "kQpnMmaRungs declares a tier for sm_" + std::to_string(sm) +
                  " which is NOT in kMeasuredQpnRungs: a tier for an arch nobody read the "
                  "SASS of is a claim with no instrument behind it");
        check(!kQpnMmaRungs[r].measured_target.empty(),
              "the sm_" + std::to_string(sm) + " row must name the exact -arch= its census "
              "came from: the ladder key folds sm_90/sm_90a and sm_120/sm_120a, so a row "
              "without that field reads as a measurement of a target it never measured");
        check(!kQpnMmaRungs[r].evidence.empty(),
              "the sm_" + std::to_string(sm) + " row must carry its census, not just a tier");
    }

    // (b) GRADE or FAIL-CLOSED, for every rung the arm can actually be reached on.
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        const auto& rung = kArchLadder[r];
        if (!qpn_arm_reachable(rung)) { continue; }
        const auto* measured = qpn_mma_rung(rung.sm);
        if (measured == nullptr) {
            // FAIL-CLOSED. Not a skip: an unmeasured rung must refuse AND say what is missing.
            const RouteChoice choice = select_route(rung.sm, NumericFormat::NVFP4, kVerify);
            check(choice.outcome == RouteOutcome::NoKernelInTree &&
                      choice.route == KernelRoute::None,
                  "reachable-but-UNMEASURED sm_" + std::to_string(rung.sm) +
                      " must fail closed (no route, named refusal), got " +
                      describe(rung.sm, NumericFormat::NVFP4, kVerify, choice));
            check(choice.why.find("UNMEASURED") != std::string::npos,
                  "sm_" + std::to_string(rung.sm) + " has no census, so its refusal must name "
                  "the missing MEASUREMENT (it must contain the word UNMEASURED), got: " +
                      choice.why);
            continue;
        }
        // MEASURED: assert the refusal quotes THIS row's census, for each format that reaches
        // the arm on this rung.
        for (const NumericFormat format : {NumericFormat::NVFP4,
                                           NumericFormat::FP8_E4M3FN_ROW_BF16S}) {
            const auto* req = format_requirement(format);
            if (req == nullptr || covers(rung.caps, req->required)) { continue; }
            const RouteChoice choice = select_route(rung.sm, format, kVerify);
            // ON A HARDWARE-CHANNEL RUNG THE FALLBACK IS THE ANSWER, and on an EMULATED one
            // it is not -- so the expectation is the PREDICATE, not a constant. Both halves
            // are asserted, and in every case the census must be quoted, which is what makes
            // this cell impossible to pass by not consulting kQpnMmaRungs.
            const bool fallback_applies =
                ninfer::caps::fp16_fallback_executable(rung.sm, format,
                                                      ninfer::caps::kQpnInBuild);
            if (fallback_applies) {
                check(choice.outcome == RouteOutcome::Selected &&
                          choice.route == KernelRoute::QpnW4a16,
                      "sm_" + std::to_string(rung.sm) + " x " + std::string(format_name(format)) +
                          " has a HARDWARE m8n8k4 channel and the kernel is in the build, so "
                          "the route must be SELECTED, got " +
                          describe(rung.sm, format, kVerify, choice));
                // The SELECTED answer must still be attributable: it names the band kernel AND
                // the census that makes the channel a tensor core on this rung.
                check(!choice.kernel.empty(),
                      "a Selected fallback route must name its band kernel");
                check(choice.why.find(std::string(measured->evidence)) != std::string::npos,
                      "the sm_" + std::to_string(rung.sm) + " x " +
                          std::string(format_name(format)) +
                          " selection must QUOTE its own measured lowering (" +
                          std::string(qpn_mma_lowering_name(measured->lowering)) +
                          "), which is what makes kQpnMmaRungs load-bearing; got: " + choice.why);
                continue;
            }
            check(choice.outcome == RouteOutcome::NoKernelInTree &&
                      choice.route == KernelRoute::None,
                  "sm_" + std::to_string(rung.sm) + " x " + std::string(format_name(format)) +
                      " must stay a NAMED REFUSAL and not become a selected route: " +
                      describe(rung.sm, format, kVerify, choice));
            check(choice.why.find(std::string(measured->evidence)) != std::string::npos,
                  "the sm_" + std::to_string(rung.sm) + " x " + std::string(format_name(format)) +
                      " refusal must QUOTE its own measured lowering (" +
                      std::string(qpn_mma_lowering_name(measured->lowering)) +
                      "), which is what makes kQpnMmaRungs load-bearing; got: " + choice.why);
            // Only the REFUSAL has to name a kernel to write: a Selected route already names
            // the one that will run. Asserted per branch rather than after the if, because
            // "the string must appear" is a different rule in the two cases.
            check(choice.why.find("skinny_nvfp4_qpn_simt") != std::string::npos ||
                      choice.why.find("qpn_host.cu") != std::string::npos,
                  "the sm_" + std::to_string(rung.sm) +
                      " refusal must still name the MISSING KERNEL as well, got: " + choice.why);
        }
    }

    // (c) The two tiers must not have collapsed into one answer, and the difference must be in
    // the ACTIONABLE half, not the wording. On the emulated tier the port is NOT the
    // recommended fix (a port would produce a routine call, not a tensor-core op), so the
    // refusal has to point at a route this card can execute today: a groupwise-int format
    // served by mma_bf16. That suggestion must be ABSENT on the hardware tier, where the port
    // IS the fix.
    const std::string hardware = select_route(70, NumericFormat::NVFP4, kVerify).why;
    const std::string emulated = select_route(90, NumericFormat::NVFP4, kVerify).why;
    check(hardware != emulated,
          "the arch grading must change the refusal: sm_70 and sm_90 answering the same text "
          "means the tier table is decoration");
    check(emulated.find("Q4G64_F16S") != std::string::npos,
          "the emulated tier must name the route this card CAN take (groupwise-int -> "
          "mma_bf16), got: " + emulated);
    check(hardware.find("Q4G64_F16S") == std::string::npos,
          "the hardware tier's answer is the port, so it must NOT be told to requantize "
          "instead, got: " + hardware);

    // (c2) And the ADVICE the emulated tier prints has to be TRUE on the card it prints it for:
    // a refusal that sends the operator to another refusal is not an answer. Positive/negative
    // pair -- Selected exactly on the rungs that have bf16 mma, refused on the rung that does
    // not.
    for (const int sm : {80, 86, 87, 88, 89, 90, 100, 103}) {
        for (const NumericFormat format : {NumericFormat::Q4G64_F16S, NumericFormat::Q5G64_F16S,
                                           NumericFormat::Q6G64_F16S,
                                           NumericFormat::W8G32_F16S}) {
            const RouteChoice choice = select_route(sm, format, kVerify);
            check(choice.outcome == RouteOutcome::Selected &&
                      choice.route == KernelRoute::MmaBf16,
                  "the emulated tier tells sm_" + std::to_string(sm) + " to requantize to " +
                      std::string(format_name(format)) +
                      ", so that route must actually be Selected there, got " +
                      describe(sm, format, kVerify, choice));
        }
    }
    for (const NumericFormat format : {NumericFormat::Q4G64_F16S, NumericFormat::W8G32_F16S}) {
        const RouteChoice choice = select_route(70, format, kVerify);
        check(choice.outcome != RouteOutcome::Selected,
              "sm_70 has no bf16 mma, so the groupwise-int advice does not apply there and "
              "the route must not be Selected, got " +
                  describe(70, format, kVerify, choice));
    }

    // (d) The escaped case: an EMULATED rung covering neither the floor NOR bf16 mma has no
    // requantization escape, and the refusal must say so rather than offer one. No rung is in
    // that shape today, so this pins the CONDITION instead of a rung -- the same "name the
    // branch you cannot reach" style the no-tensor-core pin uses. sm_70 is NOT in this loop: it
    // has no bf16 mma, but it does not reach the emulated tier either, so the sentence is not
    // live for it.
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        const auto* row = qpn_mma_rung(kArchLadder[r].sm);
        if (row == nullptr || row->lowering != QpnMmaLowering::EmulatedFp16Pipe) { continue; }
        check(ninfer::caps::has_cap(kArchLadder[r].caps, Cap::Bf16Mma),
              std::string("rung sm_") + std::to_string(kArchLadder[r].sm) +
                  " reaches the EMULATED tier of the QPN arm without bf16 mma, so the arm's "
                  "no-requantization-escape sentence is now LIVE for it and this pin has to be "
                  "updated with a real cell");
    }

    // (e) The complement: a rung the arm can NOT reach must have no row, must not be in the
    // reachable set, and must take the route its floor actually earns. sm_120 is the cell the
    // '-a' folding worry lands on.
    for (const int sm : {120, 121}) {
        const auto* rung = arch_rung(sm);
        check(rung != nullptr, "sm_" + std::to_string(sm) + " must be on the ladder");
        if (rung == nullptr) { continue; }
        check(!qpn_arm_reachable(*rung),
              "sm_" + std::to_string(sm) + " must NOT be in the QPN arm's reachable set: it "
              "carries Mxf4Nvfp4BlockScale, so the floor short-circuits the arm");
        check(qpn_mma_rung(sm) == nullptr,
              "sm_" + std::to_string(sm) + " must have NO lowering row: the arm never looks one "
              "up for it, so a tier here would be a claim nothing exercises -- and the runtime "
              "key cannot distinguish sm_120 from sm_120a anyway");
        const RouteChoice choice = select_route(sm, NumericFormat::NVFP4, kVerify);
        check(choice.outcome == RouteOutcome::Selected &&
                  choice.route == KernelRoute::Nvfp4W4a4Tma,
              "sm_" + std::to_string(sm) + " x nvfp4 must be SELECTED on its own floor route, "
              "got " + describe(sm, NumericFormat::NVFP4, kVerify, choice));
    }

    // (f) COVERAGE of the arm itself: at least one rung must be in the reachable set and at
    // least one measured rung must be in it, or every cell above ran zero times and this whole
    // group is vacuously green. This is the "the probe must be able to be non-zero" control.
    int reachable = 0;
    int reachable_and_measured = 0;
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        if (!qpn_arm_reachable(kArchLadder[r])) { continue; }
        ++reachable;
        if (qpn_mma_rung(kArchLadder[r].sm) != nullptr) { ++reachable_and_measured; }
    }
    check(reachable >= 2,
          "the QPN arm's reachable set collapsed to " + std::to_string(reachable) +
              " rung(s): the grading has nothing to grade");
    check(reachable_and_measured >= 2,
          "only " + std::to_string(reachable_and_measured) +
              " reachable rung(s) have a census: the quote-the-evidence cells above ran "
              "almost never, which would read as green");
}


// ---------------------------------------------------------------------------
// 9. A format this tree DECLARES and has NO KERNEL for
// ---------------------------------------------------------------------------
//
// FP8_E4M3FN_ROW_F32S has no kFormatRequirements row ON PURPOSE (arch_caps.h's
// kUncoveredFormatNotes says why: a row is a claim about a kernel file, and no kernel in this tree
// consumes a 4-byte-scale FP8 row encoding). So the selector must give the DELIBERATE refusal and
// not the TABLE-defect text -- the latter tells the operator to add the missing row, which is the
// one action that would turn a loud refusal into a silent claim of support. Before this group the
// two cases were indistinguishable from the outside, and the honest one read as a table bug.
void test_a_declared_format_with_no_kernel_refuses_by_name() {
    using ninfer::caps::format_requirement;
    using ninfer::caps::uncovered_format_note;

    constexpr NumericFormat kFormat = NumericFormat::FP8_E4M3FN_ROW_F32S;
    check(format_requirement(kFormat) == nullptr,
          "this group tests the NO-ROW case: if this format ever gains a row, this group no "
          "longer tests what it says it tests");
    check(uncovered_format_note(kFormat) != nullptr,
          "the deliberate named refusal must exist for it");
    for (const int sm : {70, 75, 80, 86, 90, 100, 120, 121}) {
        const RouteChoice choice = select_route(sm, kFormat, kDecode);
        check(choice.outcome == RouteOutcome::NoKernelInTree && choice.route == KernelRoute::None,
              "sm_" + std::to_string(sm) + " must refuse with no route, got " +
                  describe(sm, kFormat, kDecode, choice));
        check(choice.why.find("FP8_E4M3FN_ROW_F32S") != std::string::npos,
              "the refusal must name the format, got: " + choice.why);
        check(choice.why.find("NO KERNEL") != std::string::npos,
              "the refusal must say a KERNEL is what is missing, got: " + choice.why);
        check(choice.why.find("Missing entry: a kFormatRequirements row") == std::string::npos,
              "the refusal must NOT be the table-defect text: 'add a row' is the one instruction "
              "that must not be given for this format, got: " + choice.why);
    }
}

} // namespace

int main() {
    // The groups that ran, counted so that deleting a call below cannot hide. rc alone does
    // not carry this: a removed call prints no FAIL and leaves rc = 0 (see group 5's header).
    constexpr int kExpectedTestGroups = 9;
    int groups = 0;
    test_soundness_invariant();                                                    ++groups;
    test_unknown_sm_never_throws_and_always_warns();                               ++groups;
    test_every_rung_has_fp16_mma_so_the_no_tensor_core_branch_is_unreachable();     ++groups;
    test_v100_verdicts();                                                          ++groups;
    test_m_split_and_geometry();                                                   ++groups;
    test_non_tensor_core_formats_have_no_route();                                  ++groups;
    test_route_log_line_is_greppable();                                            ++groups;
    test_qpn_arm_is_graded_by_the_measured_mma_lowering();                         ++groups;
    test_a_declared_format_with_no_kernel_refuses_by_name();                       ++groups;

    check(groups == kExpectedTestGroups,
          "test groups run = " + std::to_string(groups) + ", expected " +
              std::to_string(kExpectedTestGroups) +
              ": a test_*() call was added to or removed from main(), and WITHOUT this check "
              "the criterion rc would not move");

    std::cout << "kernel_route: " << checks << " checks, " << failures << " failures -> "
              << (failures == 0 ? "PASS" : "FAIL") << "\n";
    return failures == 0 ? 0 : 1;
}
