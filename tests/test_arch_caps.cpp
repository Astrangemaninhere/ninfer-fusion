// Unit test for the per-architecture capability ladder and the artifact-format capability
// gate (src/core/arch_caps.h).
//
// Host-only by construction: the header has no CUDA include and takes the compute
// capability as a *parameter*, so every rung of the ladder -- including the cards this
// machine does not have -- is exercisable here, on a single-GPU host. That is the whole
// point of the parameterized form: the "does a V100 reject the nvfp4 artifact" question is a
// table question, not a hardware question.
//
// The checks fall in three groups:
//   1. table integrity (ladder order, monotonicity, exhaustiveness over NumericFormat);
//   2. floors that are pinned against the kernel sources cited in each row (so a kernel that
//      changes its intrinsic breaks this build instead of silently keeping a stale claim);
//   3. end-to-end verdicts and the actionability of the rendered message.

#include "core/arch_caps.h"
// kernel_route.h is included ONLY for one vocabulary check (group 6): arch_caps.h cannot include
// it -- kernel_route.h includes arch_caps.h, so the dependency would be a cycle -- and the two
// spellings of NoKernelInTree must be compared against each other rather than against a literal
// copied into this file. It is host-only like arch_caps.h (same header set, no CUDA include), so
// this test keeps the property its own header comment claims: no GPU, no device query.
#include "core/kernel_route.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::format_name;
using ninfer::artifact::NumericFormat;
using ninfer::caps::Cap;
using ninfer::caps::Verdict;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        ++failures;
        std::cout << "FAIL: " << what << "\n";
    }
}

bool mentions(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

std::vector<NumericFormat> one(NumericFormat format) { return {format}; }

// THE ENUMERATION OF artifact::NumericFormat COMES FROM THE ENUM, NOT FROM THIS FILE.
//
// WHAT THIS REPLACES, and why it is the point of this landing. The list used to be a hand-written
// `std::array<NumericFormat, 9>` under the comment "written out so a new format in artifact/
// reader.h is a compile error in this test rather than a silently untested enum value". That
// comment was FALSE in both directions: a tenth enumerator in reader.h was no compile error here,
// and the assertion below it compared kFormatRequirementCount against THIS ARRAY'S OWN LENGTH --
// 9 == 9 -- i.e. it asserted a fact about its own literal. Meanwhile the table it was meant to
// check had 9 rows for 12 enumerators, so I64, FP8_E4M3FN_ROW_F32S and U4Z8G16_F16S were never
// looked at and this binary still printed "all checks passed". A count checked against the thing
// it is supposed to be counting is not evidence.
//
// The walk is over the enum's own ordinal range, bounded by the enum's own last member
// (NumericFormat::Count, artifact/reader.h), so every member is visited whether or not anyone
// remembered to write it down. arch_caps.h makes the same demand at COMPILE time for every TU
// that includes it (kFormatTableCoversEveryEnumerator, plus a red-controlled static_assert); the
// static_asserts here keep this test from depending on that header's diligence.
constexpr std::size_t kFormatCountFromEnum = ninfer::caps::kFormatOrdinalCount;
static_assert(kFormatCountFromEnum == static_cast<std::size_t>(NumericFormat::Count),
              "the ordinal walk must be bounded by the enum itself, not by a literal in this file");

constexpr bool format_is_accounted_for(NumericFormat format) {
    return ninfer::caps::find_format_requirement(format) != nullptr ||
           ninfer::caps::find_uncovered_format_note(format) != nullptr;
}

static_assert([] {
                  for (std::size_t raw = 0; raw < kFormatCountFromEnum; ++raw) {
                      if (!format_is_accounted_for(static_cast<NumericFormat>(raw))) {
                          return false;
                      }
                  }
                  return true;
              }(),
              "a NumericFormat enumerator has neither a kFormatRequirements row nor a "
              "kUncoveredFormatNotes refusal: the gate would report it as UnknownFormat on every "
              "GPU, which is a table defect rather than a GPU fact");

std::vector<NumericFormat> all_formats_in_enum_order() {
    std::vector<NumericFormat> formats;
    formats.reserve(kFormatCountFromEnum);
    for (std::size_t raw = 0; raw < kFormatCountFromEnum; ++raw) {
        formats.push_back(static_cast<NumericFormat>(raw));
    }
    return formats;
}

void test_table_integrity() {
    // Group 1a: ascending, no duplicates. arch_rung() is a linear scan and the message
    // prints the matched row, so a duplicate would silently shadow.
    for (std::size_t i = 1; i < ninfer::caps::kArchLadderSize; ++i) {
        check(ninfer::caps::kArchLadder[i - 1].sm < ninfer::caps::kArchLadder[i].sm,
              "ladder is strictly ascending by sm");
    }
    // Group 1b: capabilities only accumulate. A card generation that loses a bit would mean
    // "later hardware cannot run something earlier hardware could", which is a table error.
    for (std::size_t i = 1; i < ninfer::caps::kArchLadderSize; ++i) {
        const Cap previous = ninfer::caps::kArchLadder[i - 1].caps;
        const Cap current  = ninfer::caps::kArchLadder[i].caps;
        check(ninfer::caps::covers(current, previous),
              "ladder capabilities are monotone at sm_" +
                  std::to_string(ninfer::caps::kArchLadder[i].sm));
    }
    // Group 1c: every rung is reachable by exact sm and carries a non-empty route.
    for (const ninfer::caps::ArchRung& rung : ninfer::caps::kArchLadder) {
        check(ninfer::caps::arch_rung(rung.sm) == &rung, "arch_rung resolves every rung");
        check(!rung.route.empty(), "every rung says which route works on that card");
        check(!rung.label.empty() && !rung.cards.empty(),
              "every rung names a microarchitecture and representative cards");
    }
    check(ninfer::caps::arch_rung(60) == nullptr, "sm_60 is not in the ladder");
    check(ninfer::caps::arch_rung(999) == nullptr, "an absurd sm is not in the ladder");
    // Group 1d: EVERY NumericFormat enumerator is accounted for exactly once -- by a floor row or
    // by a deliberate named refusal -- and the totals are checked against the ENUM rather than
    // against anything written out in this file.
    const std::vector<NumericFormat> all_formats = all_formats_in_enum_order();
    check(all_formats.size() == kFormatCountFromEnum,
          "the ordinal walk visits one entry per NumericFormat enumerator");
    for (const NumericFormat format : all_formats) {
        const bool has_row  = ninfer::caps::format_requirement(format) != nullptr;
        const bool has_note = ninfer::caps::uncovered_format_note(format) != nullptr;
        check(has_row || has_note,
              std::string("NumericFormat::") + std::string(format_name(format)) +
                  " has neither a capability row nor a named refusal: it would gate as "
                  "UnknownFormat on every GPU");
        check(!(has_row && has_note),
              std::string(format_name(format)) +
                  " is accounted for twice (a row AND a refusal): the row shadows the refusal");
    }
    check(ninfer::caps::kFormatRequirementCount + ninfer::caps::kUncoveredFormatCount ==
              kFormatCountFromEnum,
          "rows + deliberate refusals must equal the NUMBER OF NumericFormat ENUMERATORS "
          "(derived from the enum: " + std::to_string(kFormatCountFromEnum) + "), not a literal");
}

void test_pinned_floors() {
    const ninfer::caps::ArchRung* v100 = ninfer::caps::arch_rung(70);
    const ninfer::caps::ArchRung* turing = ninfer::caps::arch_rung(75);
    const ninfer::caps::ArchRung* ampere = ninfer::caps::arch_rung(80);
    const ninfer::caps::ArchRung* ada = ninfer::caps::arch_rung(89);
    const ninfer::caps::ArchRung* hopper = ninfer::caps::arch_rung(90);
    const ninfer::caps::ArchRung* blackwell = ninfer::caps::arch_rung(120);

    // Volta has fp16 mma only. The QPN port in the tree is the reason sm_70 is on the ladder
    // at all (ops/linear/qpn/qpn_kernels.cuh:696 emits mma.sync.m8n8k4.f16).
    check(v100 != nullptr && ninfer::caps::has_cap(v100->caps, Cap::Fp16Mma),
          "sm_70 has fp16 mma");
    check(v100 != nullptr && !ninfer::caps::has_cap(v100->caps, Cap::Bf16Mma),
          "sm_70 has no bf16 mma");
    check(v100 != nullptr && !ninfer::caps::has_cap(v100->caps, Cap::Int8Mma),
          "sm_70 has no int8 tensor core");
    // Turing: int8 but still no bf16. This is the bit that makes the shipped Q4/Q5/Q6/W8
    // kernels (all mma_bf16) unrunnable on sm_75, which the old blanket gate could not say.
    check(turing != nullptr && ninfer::caps::has_cap(turing->caps, Cap::Int8Mma),
          "sm_75 has int8 tensor cores");
    check(turing != nullptr && !ninfer::caps::has_cap(turing->caps, Cap::Bf16Mma),
          "sm_75 has no bf16 mma (so q4/q5/q6/w8 linear kernels do not run there)");
    check(ampere != nullptr && ninfer::caps::has_cap(ampere->caps, Cap::Bf16Mma),
          "sm_80 is the bf16 mma floor");
    // FP8 splits in two on this engine: the plain e4m3 form exists from Ada, but the form the
    // tree actually emits (kind::f8f6f4) is Blackwell-only.
    check(ada != nullptr && ninfer::caps::has_cap(ada->caps, Cap::Fp8E4m3MmaPlain),
          "sm_89 has plain-form fp8 e4m3 mma");
    check(ada != nullptr && !ninfer::caps::has_cap(ada->caps, Cap::Fp8F8f6f4KindMma),
          "sm_89 cannot execute kind::f8f6f4 (the form this engine emits)");
    check(hopper != nullptr && !ninfer::caps::has_cap(hopper->caps, Cap::Mxf4Nvfp4BlockScale),
          "sm_90 has no kind::mxf4nvf4 block-scaled mma");
    check(blackwell != nullptr &&
              ninfer::caps::has_cap(blackwell->caps, Cap::Mxf4Nvfp4BlockScale),
          "sm_120 executes kind::mxf4nvf4 block-scaled mma");
    // The nvfp4 floor is the load-bearing row for this tree: it is the compiled-in path.
    // MEASURED FLOOR IS sm_120, NOT sm_100. The only nvfp4 kernel this tree has
    // (ops/common/mma.cuh:89, kind::mxf4nvf4.block_scale.m16n8k64) assembles on
    // sm_120a/sm_121a only; `nvcc -cubin -arch=sm_100a` on it exits 255 with
    // "Instruction 'mma with block scale' not supported on .target 'sm_100a'", and
    // sm_103a/sm_110a reject it the same way. This loop used to assert `sm >= 100`, which
    // is what held the false positive in place.
    for (const int sm : {70, 75, 80, 86, 89, 90, 100, 103, 120, 121}) {
        const ninfer::caps::ArchRung* rung = ninfer::caps::arch_rung(sm);
        check(rung != nullptr, "sm_" + std::to_string(sm) + " is on the ladder");
        if (rung == nullptr) { continue; }
        const bool expected_nvfp4 = sm >= 120;
        check(ninfer::caps::has_cap(rung->caps, Cap::Mxf4Nvfp4BlockScale) == expected_nvfp4,
              "nvfp4 block-scaled mma availability at sm_" + std::to_string(sm));
    }
    // sm_110: NO ROW, on purpose. The row was withdrawn on 2026-09-15, and the reason is
    // that a row is not only a capability claim, it is also a ROUTE decision: with Fp16Mma
    // and no Mxf4Nvfp4BlockScale bit, select_route() used to hand NVFP4 /
    // FP8_E4M3FN_ROW_BF16S to detail::qpn_slots() (src/core/kernel_route.h) and get back
    // KernelRoute::QpnW4a16 -- a kernel family no target in this build compiles (`qpn` has 0
    // hits in src/CMakeLists.txt, whose source lists are explicit; 0 of the 11 archives
    // under build/src/ carry a qpn object member). qpn_slots() now REFUSES with
    // NoKernelInTree naming that kernel, and conservative_fallback() refuses the same way
    // for a card with no row at all, so an sm_110 card gets "no row" AND "no route" instead
    // of a warning that promised a kernel this build cannot launch. The assembly-level
    // evidence the row was added on (kind::f8f6f4 rc=0, kind::mxf4nvf4 rc=255) was never
    // the problem. What this pins: no row to look up, a verdict reported as UnknownArch
    // rather than an Unsupported refusal manufactured out of a missing table row, and a
    // route table that no longer claims a route it cannot serve.
    const ninfer::caps::CapabilityReport sm110 =
        ninfer::caps::evaluate_artifact_formats(110, one(NumericFormat::NVFP4));
    check(ninfer::caps::arch_rung(110) == nullptr &&
              sm110.verdict == Verdict::UnknownArch &&
              ninfer::caps::unknown_arch_warning(sm110, "x/y").find("sm_110") !=
                  std::string::npos,
          "sm_110 must have NO row and take the UnknownArch warning path: the only route its "
          "capability set selects (QPN W4A16) is not in this build");
}

void test_verdicts() {
    // The compiled-in reference path is supported exactly where it is compiled.
    check(ninfer::caps::evaluate_artifact_formats(120, one(NumericFormat::NVFP4)).ok(),
          "nvfp4 is supported on sm_120");
    check(ninfer::caps::evaluate_artifact_formats(121, one(NumericFormat::NVFP4)).ok(),
          "nvfp4 is supported on sm_121");
    // ... and NOWHERE ELSE. sm_100 (B100/B200) and sm_103 (B300) are refused: the engine's
    // own nvfp4 wrapper does not assemble on those targets. This assertion is the red control
    // for the false positive that used to live at sm_100. sm_110 is deliberately NOT in this
    // list any more: it has no row (see above), so its nvfp4 answer is UnknownArch -- the
    // warning path, not an evidence-backed Unsupported refusal.
    for (const int sm : {100, 103}) {
        const ninfer::caps::CapabilityReport report =
            ninfer::caps::evaluate_artifact_formats(sm, one(NumericFormat::NVFP4));
        check(!report.ok(), "nvfp4 is refused on sm_" + std::to_string(sm) +
                                " (its only kernel cannot be assembled there)");
        check(report.verdict == Verdict::Unsupported,
              "the sm_" + std::to_string(sm) + " nvfp4 refusal is Unsupported, not UnknownArch");
        check(report.gaps.size() == 1 && report.gaps[0].format == NumericFormat::NVFP4,
              "the sm_" + std::to_string(sm) + " refusal names nvfp4");
    }
    // Formats with no tensor-core operand never gate anything.
    for (const int sm : {70, 75, 80, 86, 89, 90, 100, 120}) {
        check(ninfer::caps::evaluate_artifact_formats(sm, std::span<const NumericFormat>{})
                  .ok(),
              "an empty format set is supported at sm_" + std::to_string(sm));
        check(ninfer::caps::evaluate_artifact_formats(sm, one(NumericFormat::FP32)).ok(),
              "fp32 scale words are supported at sm_" + std::to_string(sm));
        check(ninfer::caps::evaluate_artifact_formats(sm, one(NumericFormat::I32)).ok(),
              "i32 control payloads are supported at sm_" + std::to_string(sm));
    }
    // bf16 weights need sm_80 and above only.
    check(!ninfer::caps::evaluate_artifact_formats(70, one(NumericFormat::BF16)).ok(),
          "bf16 weights are refused on sm_70");
    check(!ninfer::caps::evaluate_artifact_formats(75, one(NumericFormat::BF16)).ok(),
          "bf16 weights are refused on sm_75");
    check(ninfer::caps::evaluate_artifact_formats(80, one(NumericFormat::BF16)).ok(),
          "bf16 weights are accepted on sm_80");
    // groupwise-int (W8G32_F16S) is the format the community sm_86 fork ships. It is refused
    // on sm_75, which is a correction of the "groupwise-int targets Turing+" claim in
    // CMakeLists.txt:6-8 and layouts_impl.h:931.
    for (const NumericFormat format :
         {NumericFormat::Q4G64_F16S, NumericFormat::Q5G64_F16S, NumericFormat::Q6G64_F16S,
          NumericFormat::W8G32_F16S}) {
        check(!ninfer::caps::evaluate_artifact_formats(75, one(format)).ok(),
              std::string(format_name(format)) + " is refused on sm_75");
        check(ninfer::caps::evaluate_artifact_formats(86, one(format)).ok(),
              std::string(format_name(format)) + " is accepted on sm_86");
    }
    // fp8 weights ride the A16 route (bf16 mma) from sm_80 up; only the A8 route is
    // Blackwell-form. So the artifact format itself is NOT fp8-tensor-core-only.
    check(ninfer::caps::evaluate_artifact_formats(86,
                                                  one(NumericFormat::FP8_E4M3FN_ROW_BF16S))
              .ok(),
          "fp8 row-scale weights are accepted on sm_86 through the A16 route");
    // THE fp16 FALLBACK EXIT, AND ITS BOUNDARY. fp8 row-scale weights need bf16 mma on the
    // A16 route, which sm_75 does not have, and the fp16 fallback exit does NOT rescue them:
    // the fp8 arm of the QPN family (skinny_fp8_qpn8 / _mt2) exists as a template with NO HOST
    // ENTRY -- gemm_qpn dispatches the nvfp4 kernels only -- so kFormatRequirements carries
    // Cap::None for its fallback floor and this format refuses here IN BOTH WORLDS. That is
    // asserted on both, because "the QPN sources are in the build" must not be able to change
    // this answer: the missing piece is a dispatch, not an arch.
    const ninfer::caps::CapabilityReport fp8_sm75 =
        ninfer::caps::evaluate_artifact_formats(75, one(NumericFormat::FP8_E4M3FN_ROW_BF16S));
    check(!fp8_sm75.ok(), "fp8 row-scale weights are refused on sm_75");
    check(fp8_sm75.verdict == Verdict::Unsupported && fp8_sm75.fallbacks.empty(),
          "the refusal is evidence-backed and is NOT a fallback in disguise");
    check(fp8_sm75.gaps.size() == 1 && fp8_sm75.gaps[0].required == Cap::Bf16Mma,
          "the refusal names the floor sm_75 does not have (bf16 mma)");
    check(ninfer::caps::evaluate_artifact_formats(75, one(NumericFormat::FP8_E4M3FN_ROW_BF16S),
                                                  /*qpn_in_build=*/true)
              .verdict == Verdict::Unsupported,
          "and having the QPN sources in the build must NOT change it: there is no gemm_qpn "
          "fp8 dispatch, so there is no host entry to launch");
    check(!ninfer::caps::fp16_fallback_executable(75,
                                                  NumericFormat::FP8_E4M3FN_ROW_BF16S, true),
          "the predicate must be false for fp8 even with the build fact true");
    // Unknown compute capability is a refusal, not an optimistic pass.
    const ninfer::caps::CapabilityReport unknown =
        ninfer::caps::evaluate_artifact_formats(60, one(NumericFormat::BF16));
    check(unknown.verdict == Verdict::UnknownArch, "sm_60 yields UnknownArch");
    check(!unknown.ok(), "an unlisted compute capability is not ok()");
    // Mixed artifacts report every gap, not just the first.
    const std::array<NumericFormat, 2> mixed = {NumericFormat::BF16, NumericFormat::NVFP4};
    const ninfer::caps::CapabilityReport both =
        ninfer::caps::evaluate_artifact_formats(80, mixed);
    check(both.verdict == Verdict::Unsupported, "a mixed artifact on sm_80 is unsupported");
    check(both.gaps.size() == 1, "sm_80 meets the bf16 floor, so only nvfp4 is a gap");
    check(both.gaps[0].format == NumericFormat::NVFP4, "the reported gap is nvfp4");
    // sm_70 on a MIXED artifact: bf16 is still a GAP (it has no fp16 fallback kernel at
    // all), while nvfp4 is now served by the QPN fallback, so it is reported as a FALLBACK
    // and not as a gap. The two lists are what keep "the artifact loads" from being confused
    // with "the card meets every floor", and the counts are asserted on both.
    const ninfer::caps::CapabilityReport v100 =
        ninfer::caps::evaluate_artifact_formats(70, mixed);
    check(v100.gaps.size() == 1, "sm_70 reports exactly one GAP on the mixed artifact (bf16)");
    check(v100.gaps[0].format == NumericFormat::BF16,
          "the remaining gap on sm_70 is bf16, which has no fp16 fallback kernel");
    check(v100.fallbacks.size() == 1 && v100.fallbacks[0].format == NumericFormat::NVFP4,
          "nvfp4 on sm_70 is reported as a FALLBACK, not as a gap");
    check(v100.verdict == Verdict::Unsupported,
          "the mixed artifact is still Unsupported on sm_70: one gap is enough");
    const ninfer::caps::CapabilityReport v100_closed =
        ninfer::caps::evaluate_artifact_formats(70, mixed, /*qpn_in_build=*/false);
    check(v100_closed.gaps.size() == 2 && v100_closed.fallbacks.empty(),
          "without the QPN sources both formats are gaps again, as before this change");
}

void test_message_is_actionable() {
    const std::string nvfp4_on_ada = ninfer::caps::render_capability_report(
        ninfer::caps::evaluate_artifact_formats(89, one(NumericFormat::NVFP4)),
        "qwen3.8-27b/qwen38-nvfp4");
    check(!nvfp4_on_ada.empty(), "a rejected artifact renders a message");
    check(mentions(nvfp4_on_ada, "sm_89"), "the message names the GPU");
    check(mentions(nvfp4_on_ada, "Ada"), "the message names the microarchitecture");
    check(mentions(nvfp4_on_ada, "nvfp4"), "the message names the weight format");
    check(mentions(nvfp4_on_ada, "kind::mxf4nvf4"),
          "the message names the missing instruction family");
    check(mentions(nvfp4_on_ada, "nvfp4_w4a4_mma.cuh:308"),
          "the message cites the kernel that sets the floor");
    check(mentions(nvfp4_on_ada, "what to do"), "the message ends with an action list");
    check(mentions(nvfp4_on_ada, "tools/convert/"),
          "the message points at the requantization path");
    check(mentions(nvfp4_on_ada, "qwen3.8-27b/qwen38-nvfp4"),
          "the message identifies the artifact");
    // The build arch list is unnamed when the build does not export it. Reported rather than
    // faked: the runtime cannot read cudaDeviceProp for the '-a' suffix of its own target.
    check(mentions(nvfp4_on_ada, "compiled for "),
          "the message reports the binary's arch list line");
    // ---------------------------------------------------------------------------------------
    // THE ASSERTION THAT COULD NOT FAIL. It used to be
    //     check(!build_arch_list().empty() || mentions(msg, "NINFER_BUILD_CUDA_ARCHES"), ...)
    // and BOTH disjuncts were satisfied by the defect it looks like it guards: with
    // ..._ARCHES never defined by the build, build_arch_list() was always empty (so the first
    // disjunct could never be the reason it passed) and the message always quoted the macro
    // NAME, so the SECOND disjunct was always true and the whole check always passed. A check
    // whose disjunction is satisfied by the bug is not a check.
    //
    // The replacement branches on the FACT and demands the matching behaviour, so each branch
    // can fail on its own:
    //   * the build exported the list (this TU was given the macro) -> the report must QUOTE it,
    //     and must not print the unreported/remedy text;
    //   * the build did not -> the report must NAME the unreported state, and its remedy must
    //     point at something followable (the CMake variable or the TU that owns the macro),
    //     NOT at a macro name this build cannot define.
    // CORRECTION TO THE LINE THIS REPLACED: it named `patches/src_CMakeLists.txt.diff`. There is
    // no `patches/` directory in this repository (measured 2026-09-19), so from inside the tree
    // that comment pointed at nothing. The file it means is a REAL prepared artifact, but it
    // lives in a sibling's holding area, off-tree, and is not landed: dl/routeprobe/patches/
    // src_CMakeLists.txt.diff, which creates `ninfer_build_cuda_arches` as an INTERFACE target
    // and -- measured -- links it to NOTHING, so as prepared it publishes the fact to no target
    // at all. The batch that closes this must therefore say which targets consume it.
    //
    // WHY THE ASSERTION BELOW IS NOT A #ifdef ANY MORE, and what the #ifdef asserted that was
    // false. The previous shape selected the assertion WITH THE SAME MACRO the assertion was
    // about:
    //     #ifdef NINFER_BUILD_CUDA_ARCHS -> build_arch_list() is non-empty, quoted, ...
    //     #else                          -> build_arch_list().empty() && report says <unreported>
    // In the #else branch, `build_arch_list().empty()` was NOT an observation: that branch of the
    // function returns the empty view UNCONDITIONALLY, so the conjunction was guaranteed true by
    // the very preprocessor choice that selected it. It is the same class of defect as the
    // "assertion that could not fail" the comment above removed -- an unfalsifiable conjunction
    // replacing an unfalsifiable disjunction -- and it was GREEN over a tree in which the macro
    // reached no reader of this header at all.
    //
    // The claim is now one code path, drawn from the VALUE rather than from the macro: whichever
    // way the build is configured, the REPORT must agree with the VALUE.
    const std::string_view arches = ninfer::caps::build_arch_list();
    check(mentions(nvfp4_on_ada, "compiled for "),
          "the report has the binary's arch-list line");
    if (arches.empty()) {
        check(mentions(nvfp4_on_ada, "<unreported>"),
              "an EMPTY arch list is NAMED as unreported, not printed as nothing");
        check(mentions(nvfp4_on_ada, "CMAKE_CUDA_ARCHITECTURES"),
              "the remedy names the CMake variable that owns the fact");
        check(mentions(nvfp4_on_ada, "src/CMakeLists.txt"),
              "the remedy names the FILE that owns the macro, so a reader can follow it");
        check(mentions(nvfp4_on_ada, "does NOT change"),
              "the remedy states that a rebuild alone does not change this line");
    } else {
        check(mentions(nvfp4_on_ada, std::string(arches)),
              "a NON-EMPTY arch list is QUOTED in the report");
        check(!mentions(nvfp4_on_ada, "<unreported>"),
              "a non-empty arch list is not also called unreported");
    }
    check(!mentions(nvfp4_on_ada, "NINFER_BUILD_CUDA_ARCHES"),
          "neither branch may prescribe the with-E spelling: measured, 0 of the 573 recorded "
          "compile commands define it, so following that instruction changes nothing");

    // ---------------------------------------------------------------------------------------
    // THE CLAIM THAT NAMES THE BUILD -- and the reason this test can now fail.
    //
    // Everything above is a claim about the MESSAGE. It holds in both configurations, so it
    // cannot see a build in which NO translation unit that includes this header is given the
    // arch list -- which was exactly the state of the pre-image. This check is that state, and
    // it is the one the previous shape could not express from inside a branch selected by the
    // same macro.
    //
    // MEASURED on the pre-image (2026-09-19): NINFER_BUILD_CUDA_ARCHS appears on 1 of the 573
    // recorded compile commands, core/device_probe.cu, and that file's depfile (440
    // prerequisites) names core/arch_caps.h ZERO times. The rename landed and the publisher did
    // not: the function read the name the build writes and was still empty in every reader.
    check(!arches.empty(),
          "the build must GIVE this translation unit the arch list. It is empty, so no TU that "
          "includes core/arch_caps.h receives NINFER_BUILD_CUDA_ARCHS: the macro is scoped to "
          "core/device_probe.cu, which does not include this header. RED until the publisher "
          "lands -- src/CMakeLists.txt must compile core/build_arch_list.cpp with the macro and "
          "this test must link it. EXPECTED RED on the pre-image; green does NOT mean the list "
          "is right, only that the fact arrived.");

    // An unknown compute capability gets its own text and does not borrow a neighbour's row.
    const std::string unknown = ninfer::caps::render_capability_report(
        ninfer::caps::evaluate_artifact_formats(60, one(NumericFormat::BF16)), "x/y");
    check(mentions(unknown, "not in the capability ladder"),
          "an unknown sm says so instead of quoting a route");

    // A supported artifact renders nothing at all.
    check(ninfer::caps::render_capability_report(
              ninfer::caps::evaluate_artifact_formats(120, one(NumericFormat::NVFP4)), "x/y")
              .empty(),
          "a supported artifact renders an empty report");
}

void test_artifact_formats_scan() {
    using ninfer::artifact::ObjectDescriptor;
    using ninfer::artifact::ResourceDescriptor;
    using ninfer::artifact::TensorDescriptor;

    const TensorDescriptor bf16{.name        = "a",
                                .shape       = {2, 2},
                                .format      = NumericFormat::BF16,
                                .layout      = ninfer::artifact::StorageLayout::ContiguousLeV1,
                                .offset      = 0,
                                .bytes       = 8};
    TensorDescriptor nvfp4 = bf16;
    nvfp4.name             = "b";
    nvfp4.format           = NumericFormat::NVFP4;
    TensorDescriptor bf16_again = bf16;
    bf16_again.name             = "c";
    const ResourceDescriptor resource{
        .name = "tok", .encoding = ninfer::artifact::ResourceEncoding::RawBytesV1, .offset = 0,
        .bytes = 4};

    const std::vector<ObjectDescriptor> objects{bf16, resource, nvfp4, bf16_again};
    const std::vector<NumericFormat> found = ninfer::caps::artifact_formats(objects);
    check(found.size() == 2, "resources are skipped and duplicate formats collapse");
    check(found.size() == 2 && found[0] == NumericFormat::BF16 && found[1] == NumericFormat::NVFP4,
          "the surviving formats keep first-seen order");
    check(ninfer::caps::artifact_formats({}).empty(), "an empty artifact yields no formats");

    // require_artifact_formats_supported is silent when supported and throws the rendered
    // report when not -- the two behaviours the load path depends on.
    bool threw = false;
    try {
        ninfer::caps::require_artifact_formats_supported(120, found, "ok/ok");
    } catch (const std::exception&) { threw = true; }
    check(!threw, "require_* is silent for a supported artifact");
    std::string caught;
    try {
        ninfer::caps::require_artifact_formats_supported(86, found, "ok/ok");
    } catch (const std::invalid_argument& error) { caught = error.what(); }
    check(mentions(caught, "cannot run this artifact"),
          "require_* throws std::invalid_argument carrying the report");
    check(mentions(caught, "nvfp4"), "the thrown report names the offending format");
}


// ---------------------------------------------------------------------------
// The three formats the exhaustiveness claim was silently short of
// ---------------------------------------------------------------------------
//
// I64 AND U4Z8G16_F16S ARE ROWS, BECAUSE NEITHER IS EVER A TENSOR-CORE OPERAND (class (a)). Both
// are declared, both have their own encoded geometry in artifact/storage_layouts.cpp, and both are
// read by a compiled-in consumer through bind_mapped/retain_mapped_tensor -- which maps the bytes
// out of the artifact file, never uploads them as such and never builds a Weight. That is exactly
// the shape of the FP32/I32 rows, so their floor is Cap::None and the honest checks are that the
// row resolves (no longer UnknownFormat) and that the verdict is supported on every rung for the
// same reason fp32's is. A row claiming a tensor-core floor for either would gate host metadata on
// a capability it never touches.
void test_host_metadata_formats_are_covered() {
    for (const NumericFormat format : {NumericFormat::I64, NumericFormat::U4Z8G16_F16S}) {
        const ninfer::caps::FormatRequirement* requirement =
            ninfer::caps::format_requirement(format);
        check(requirement != nullptr,
              std::string(format_name(format)) +
                  " must have a capability row: it is declared, its geometry is implemented, and "
                  "a compiled-in consumer reads it");
        check(requirement != nullptr && requirement->required == Cap::None,
              std::string(format_name(format)) +
                  " is host metadata, so its row must declare Cap::None");
        check(requirement != nullptr &&
                  requirement->kernel_evidence.find("tensor-core operand") !=
                      std::string_view::npos,
              std::string(format_name(format)) +
                  " must say in its citation that it is never a tensor-core operand, which is "
                  "the whole justification for Cap::None");
        // THE RUNGS ON THE LADDER, and NOT 110: evaluate_artifact_formats() answers the ARCH
        // question FIRST, so a compute capability with no ladder row returns UnknownArch (the
        // warning path) and no format is consulted at all. sm_110 is that row -- withdrawn
        // 2026-09-15 -- and the case is pinned explicitly below rather than dropped from a list,
        // because a silently shortened list is the defect this landing is about.
        for (const int sm : {70, 75, 80, 86, 89, 90, 100, 120, 121}) {
            const std::vector<NumericFormat> only = one(format);
            const ninfer::caps::CapabilityReport report =
                ninfer::caps::evaluate_artifact_formats(sm, only);
            check(report.verdict != Verdict::UnknownFormat,
                  std::string(format_name(format)) + " on sm_" + std::to_string(sm) +
                      " must no longer be UnknownFormat: the row is what fixes that");
            check(report.ok(),
                  std::string(format_name(format)) + " on sm_" + std::to_string(sm) +
                      " must be supported, because host metadata has no tensor-core floor");
        }
        const std::vector<NumericFormat> off_ladder_in = one(format);
        const ninfer::caps::CapabilityReport off_ladder =
            ninfer::caps::evaluate_artifact_formats(110, off_ladder_in);
        check(off_ladder.verdict == Verdict::UnknownArch && !off_ladder.ok(),
              std::string(format_name(format)) +
                  " on sm_110 must take the UnknownArch warning path (that capability has no "
                  "ladder row), NOT a format verdict: the arch question is asked first");
    }
}

// FP8_E4M3FN_ROW_F32S IS A NAMED REFUSAL AND NOT A ROW (class (b)), and this checks that the
// refusal is an ON-RAMP. It IS a GEMM operand, so Cap::None would be a lie; and no kernel in this
// tree executes its 4-byte-scale encoding, so a floor row would name a kernel that cannot be
// launched. What is asserted: it refuses on EVERY rung (the missing piece is a kernel, not a
// capability), the verdict is Unsupported and NOT UnknownFormat (whose text tells the operator to
// add the very row that must not exist), and the rendered text names the format and the missing
// pieces.
void test_f32_scale_fp8_refuses_diagnosably() {
    constexpr NumericFormat kFormat = NumericFormat::FP8_E4M3FN_ROW_F32S;
    check(ninfer::caps::format_requirement(kFormat) == nullptr,
          "FP8_E4M3FN_ROW_F32S must have NO floor row: a row is a claim about a kernel file, and "
          "this tree has none for the 4-byte-scale encoding");
    check(ninfer::caps::uncovered_format_note(kFormat) != nullptr,
          "it must carry a NAMED refusal instead, or the gate falls through to UnknownFormat");
    // Every rung ON THE LADDER, plus the two fp4-capable numbers this tree cares about: the
    // missing piece is a kernel, so no compute capability may change this answer.
    for (const int sm : {70, 75, 80, 86, 89, 90, 100, 103, 120, 121}) {
        const std::vector<NumericFormat> only = one(kFormat);
        const ninfer::caps::CapabilityReport report =
            ninfer::caps::evaluate_artifact_formats(sm, only);
        check(!report.ok(), "sm_" + std::to_string(sm) + " must refuse FP8_E4M3FN_ROW_F32S");
        check(report.verdict == Verdict::Unsupported,
              "the sm_" + std::to_string(sm) + " answer must be Unsupported (an evidence-backed "
              "refusal), not Supported and not UnknownFormat (a table-defect label)");
        check(report.gaps.size() == 1 && report.gaps[0].format == kFormat,
              "the sm_" + std::to_string(sm) + " refusal must name the format");
        check(report.gaps.size() == 1 && report.gaps[0].required == Cap::Bf16Mma,
              "the refusal must name the floor a real kernel for it would take");
    }
    const std::string rendered = ninfer::caps::render_capability_report(
        ninfer::caps::evaluate_artifact_formats(120, one(kFormat)), "m/w");
    check(mentions(rendered, "FP8_E4M3FN_ROW_F32S"), "the refusal names the format");
    check(mentions(rendered, "linear.cpp"),
          "the refusal points at the linear dispatch that has no arm for this format");
    check(mentions(rendered, "scale_dtype"),
          "the refusal names the validator requirement the F32 encoding fails");
    check(mentions(rendered, "row_scale_weight"),
          "the refusal warns that the row-scaled materializer arm hard-codes a 2-byte scale word "
          "and would compute a half-size scale plane, so the materializer is not where to fix it");
    check(mentions(rendered, "what to do"), "the refusal keeps an action list");
    check(!mentions(rendered, "has no capability row for"),
          "the refusal must NOT render through the UnknownFormat text, which says the TABLE is "
          "what fixes the floor");
    // sm_110 IS THE ONE PLACE THIS FORMAT IS NOT REFUSED-for-a-missing-kernel, and the reason is
    // the gate's order rather than the format: sm_110 has no ladder row, so UnknownArch is
    // returned before the format is looked at. Pinned so it cannot read as a hole in "refused on
    // every GPU" -- and so that adding an sm_110 row later moves THIS case, loudly.
    const std::vector<NumericFormat> off_ladder_in = one(kFormat);
    const ninfer::caps::CapabilityReport off_ladder =
        ninfer::caps::evaluate_artifact_formats(110, off_ladder_in);
    check(off_ladder.verdict == Verdict::UnknownArch && !off_ladder.ok(),
          "sm_110 has no ladder row, so the format gate is not reached on it: the answer is "
          "UnknownArch and this case must not be counted as the format's refusal");

    // Order-independence: the verdict must not depend on the order the artifact's formats are
    // visited in, or the same file would produce two different answers.
    const std::vector<NumericFormat> refusal_first = {kFormat, NumericFormat::NVFP4};
    const std::vector<NumericFormat> refusal_last  = {NumericFormat::NVFP4, kFormat};
    const ninfer::caps::CapabilityReport first =
        ninfer::caps::evaluate_artifact_formats(120, refusal_first);
    const ninfer::caps::CapabilityReport last =
        ninfer::caps::evaluate_artifact_formats(120, refusal_last);
    check(first.verdict == Verdict::Unsupported && last.verdict == Verdict::Unsupported,
          "a mixed artifact must stay Unsupported whichever order the formats are visited in");
    check(first.gaps.size() == 1 && last.gaps.size() == 1,
          "the mixed artifact reports exactly one gap (nvfp4 meets its floor on sm_120)");
}

// ===========================================================================
// The AMD rungs (additive section at the end of src/core/arch_caps.h)
// ===========================================================================
//
// WHAT THESE CHECKS ARE FOR. The section they cover exists to convert "does AMD support format
// X?" from an implied claim into a NAMED REFUSAL. A named refusal is only worth anything if its
// NAME is checkable, so the checks below are deliberately of the variety this file's own history
// says the others were not:
//
//   1. every citation in kPtxSites / kAmdFormatBlockers / kAmdLdsBlockerSites is READ BACK OFF
//      THE TREE and must contain the exact mnemonic it claims -- so a row cannot be copied out
//      of an assessment document, and the count of rows is not evidence by itself;
//   2. the verdicts are DERIVED from kFormatRequirements and compared against it, so the AMD
//      table cannot become a second census that drifts (the failure mode kernel_route.h's own
//      header names: "restating it in a second table that can go stale");
//   3. the unprobed state is asserted, so no row can quietly start reading as a measurement.
//
// The red control in group 1 is the one that keeps the whole thing honest: a checker that
// accepts every citation is not a check. It is given a mnemonic that is real but on the WRONG
// ONE OF TWO ADJACENT LINES (mma.cuh:113 is bf16, :128 is f16) and must reject it.

#ifdef NINFER_SOURCE_DIR

// "src/ops/common/mma.cuh:113" -> path + line. A citation that cannot be parsed is itself a
// defect, so the parse failure is reported rather than skipped.
bool split_citation(std::string_view citation, std::string& path, long& line) {
    const std::size_t colon = citation.rfind(':');
    if (colon == std::string_view::npos || colon + 1 >= citation.size()) { return false; }
    path.assign(citation.substr(0, colon));
    const std::string digits(citation.substr(colon + 1));
    for (const char c : digits) {
        if (c < '0' || c > '9') { return false; }
    }
    line = std::stol(digits);
    return !path.empty() && line > 0;
}

// Reads the file and returns the 1-based line, or false if the file or line is missing. The
// reason string is what makes a failure actionable instead of just red.
bool read_cited_line(const std::string& root, std::string_view citation, std::string& text,
                     std::string& why) {
    std::string path;
    long line = 0;
    if (!split_citation(citation, path, line)) {
        why = "'" + std::string(citation) + "' is not a parseable file:line citation";
        return false;
    }
    std::ifstream file(root + "/" + path, std::ios::binary);
    if (!file.good()) {
        why = "cited file does not exist: " + path;
        return false;
    }
    long current = 0;
    std::string line_text;
    while (std::getline(file, line_text)) {
        ++current;
        if (current == line) {
            text = line_text;
            return true;
        }
    }
    why = "cited line is past the end of " + path + ":" + std::to_string(line);
    return false;
}

bool cited_line_contains(const std::string& root, std::string_view citation,
                         std::string_view needle, std::string& why) {
    std::string text;
    if (!read_cited_line(root, citation, text, why)) { return false; }
    if (text.find(needle) == std::string::npos) {
        why = std::string(citation) + " does not contain '" + std::string(needle) +
              "'; the line reads: " + text;
        return false;
    }
    return true;
}

// THE REASON HAS TO BE COMPUTED BEFORE THE MESSAGE IS BUILT, and this helper exists because the
// first version of these checks got that wrong: it wrote
//     check(cited_line_contains(root, site.file_line, site.mnemonic, why), "...: " + why);
// where both arguments are evaluated before the call, so `why` was still EMPTY when the message
// was assembled. The check still went red on a mutated citation -- which is why the mutation run
// is what found it -- but it printed "FAIL: PtxSite mma.sync: " with no reason, i.e. a red light
// with no diagnosis. Calling through this helper makes the order structural instead of a thing
// each call site has to remember.
void check_citation(const std::string& root, std::string_view citation, std::string_view needle,
                    const std::string& what) {
    std::string why;
    const bool ok = cited_line_contains(root, citation, needle, why);
    check(ok, what + ": " + why);
}

// Group 1: THE MARKS ARE CHECKABLE. Every citation in the AMD section is read back off the tree.
void test_amd_citations_are_real() {
    const std::string root        = NINFER_SOURCE_DIR;
    std::size_t sites_checked     = 0;
    std::size_t blockers_checked  = 0;
    std::size_t lds_checked       = 0;
    std::size_t families_seen     = 0;
    bool family_seen[static_cast<std::size_t>(ninfer::caps::PtxFamily::Count)] = {};

    for (const ninfer::caps::PtxSite& site : ninfer::caps::kPtxSites) {
        check_citation(root, site.file_line, site.mnemonic,
                       "PtxSite " + std::string(ninfer::caps::ptx_family_name(site.family)));
        check(!site.guard.empty(),
              std::string(site.file_line) + " must name the arch guard it sits inside: without "
              "the guard a reader cannot tell whether the instruction is emitted or trapped");
        family_seen[static_cast<std::size_t>(site.family)] = true;
        ++sites_checked;
    }
    for (std::size_t i = 0; i < ninfer::caps::kPtxFamilyCount; ++i) {
        if (family_seen[i]) { ++families_seen; }
    }
    // A count checked against the thing it is supposed to be counting is not evidence -- so this
    // one is checked against the ENUM's own range, and the total is used to prove the loop ran.
    check(families_seen == ninfer::caps::kPtxFamilyCount,
          "every PtxFamily must have at least one cited site: " +
              std::to_string(families_seen) + " of " +
              std::to_string(ninfer::caps::kPtxFamilyCount) + " families are cited");
    check(sites_checked >= ninfer::caps::kPtxFamilyCount,
          "the site loop must have walked every row: checked " + std::to_string(sites_checked));

    // THE BLOCKERS: both sites, in the format's OWN kernel, and the Cap bit must be the one
    // kFormatRequirements records for that format. That equality is what forbids this table from
    // becoming an independent claim about which formats have a floor.
    for (const ninfer::caps::AmdFormatBlocker& blocker : ninfer::caps::kAmdFormatBlockers) {
        check_citation(root, blocker.mma_site, blocker.mma_mnemonic,
                       std::string(format_name(blocker.format)) + " mma site");
        check_citation(root, blocker.fragment_site, blocker.fragment_mnemonic,
                       std::string(format_name(blocker.format)) + " fragment site");
        const ninfer::caps::FormatRequirement* requirement =
            ninfer::caps::format_requirement(blocker.format);
        check(requirement != nullptr,
              std::string(format_name(blocker.format)) +
                  " has an AMD blocker row but no kFormatRequirements row: an AMD-only claim "
                  "about a format the floor table does not recognise");
        check(requirement != nullptr && requirement->required == blocker.cap,
              std::string(format_name(blocker.format)) +
                  "'s AMD blocker names a different Cap than its floor row: the two tables "
                  "disagree about the same format");
        // The ldmatrix half must be a real ldmatrix and the mma half a real mma: a row that
        // swapped them would still pass the substring check above.
        check(blocker.fragment_mnemonic.find("ldmatrix") != std::string_view::npos &&
                  blocker.mma_mnemonic.find("mma_") != std::string_view::npos,
              std::string(format_name(blocker.format)) +
                  "'s blocker must cite the mma channel and the ldmatrix loader separately");
        // A second route is cited only where one exists; where it is cited, the line must really
        // be an mbarrier site, or the refusal would name a route the format does not have.
        if (!blocker.secondary_route.empty()) {
            check_citation(root, blocker.secondary_route, "mbarrier",
                           std::string(format_name(blocker.format)) + " secondary route");
        }
        ++blockers_checked;
    }
    // The secondary route must exist for the format that has one and for no other: kFormatRequirements'
    // fallback comment states the rule -- "this table refuses to make one up".
    check(ninfer::caps::kAmdFormatBlockers[5].format == NumericFormat::NVFP4 &&
              !ninfer::caps::kAmdFormatBlockers[5].secondary_route.empty(),
          "nvfp4 must carry the secondary TMA route: src/CMakeLists.txt:69 builds "
          "nvfp4_w4a4_tma.cu, so this format has two blocked routes and a refusal that named "
          "only one would understate what a port has to replace");
    for (const ninfer::caps::AmdFormatBlocker& blocker : ninfer::caps::kAmdFormatBlockers) {
        if (blocker.format == NumericFormat::NVFP4) { continue; }
        check(blocker.secondary_route.empty(),
              std::string(format_name(blocker.format)) +
                  " must NOT claim a secondary route: no other format in this tree has a TMA "
                  "variant of its route");
    }

    for (const ninfer::caps::AmdLdsBlockerSite& site : ninfer::caps::kAmdLdsBlockerSites) {
        check_citation(root, site.file_line, site.mnemonic,
                       "LDS blocker " + std::string(site.file_line));
        check(!site.requested.empty() && !site.note.empty(),
              std::string(site.file_line) + " must record the request and why it matters");
        ++lds_checked;
    }

    // RED CONTROL. The checker must be able to tell mma.cuh:113 (bf16) from mma.cuh:128 (f16).
    // Without this, "every citation verified" would be a claim about the checker, not the table.
    std::string why;
    check(!cited_line_contains(root, "src/ops/common/mma.cuh:113",
                               "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", why),
          "RED CONTROL FAILED: the citation checker accepted the f16 mnemonic at mma.cuh:113, "
          "which is the bf16 line -- every citation check above would then be vacuous");
    check(!cited_line_contains(root, "src/ops/common/mma.cuh:99999", "mma.sync", why),
          "RED CONTROL FAILED: the citation checker accepted a line past the end of the file");
    check(!cited_line_contains(root, "src/ops/common/no_such_file.cuh:1", "mma.sync", why),
          "RED CONTROL FAILED: the citation checker accepted a file that does not exist");

    std::cout << "AMD citations checked: " << sites_checked << " PtxSite, " << blockers_checked
              << " format blocker (x2 sites), " << lds_checked << " LDS site\n";
}
#else
void test_amd_citations_are_real() {
    std::cout << "SKIP: NINFER_SOURCE_DIR not defined, AMD citations not read back\n";
}
#endif // NINFER_SOURCE_DIR

// Group 2: THE TWO TABLES STAY APART. An AMD target must never resolve through arch_rung(),
// because a ladder row means "a device reports this number" and no AMD device reports one.
void test_amd_targets_are_not_ladder_rungs() {
    check(ninfer::caps::kAmdLadderSize == 6,
          "the ladder must carry the six declared AMD targets");
    for (const std::string_view target :
         {"gfx906", "gfx908", "gfx90a", "gfx942", "gfx1100", "gfx1201"}) {
        const ninfer::caps::AmdRung* rung = ninfer::caps::amd_rung(target);
        check(rung != nullptr, std::string(target) + " must be in kAmdLadder");
        if (rung == nullptr) { continue; }
        check(!rung->isa.empty() && !rung->cards.empty(),
              std::string(target) + " must name its ISA generation and representative cards");
        check(!rung->evidence.empty(),
              std::string(target) + " must say where its name comes from in this tree");
        // THE SUPPORT GATE, AS A VALUE. probe_evidence is empty on every row and this is the
        // assertion that makes that a fact rather than a hope: filling one in is the edit that
        // turns this table into a support claim, and it must break this test when it happens.
        check(rung->probe_evidence.empty(),
              std::string(target) +
                  " must have NO probe evidence: nothing on this machine can probe an AMD "
                  "target, and a non-empty value here would read as a support claim");
        check(!rung->route.empty(), std::string(target) + " must say what the refusal means");
    }
    check(ninfer::caps::amd_rung("gfx9999") == nullptr, "an unknown gfx target is not in the table");
    check(ninfer::caps::amd_rung("sm_120") == nullptr, "an sm number is not an AMD target");
    check(ninfer::caps::amd_rung("") == nullptr, "the empty target is not in the table");
    // The reverse direction: no AMD target may enter the compute-capability ladder. Checked by
    // scanning the ladder's own rows rather than by a list of numbers, so a row added later with
    // a fabricated key is caught whatever key it picks.
    for (const ninfer::caps::ArchRung& rung : ninfer::caps::kArchLadder) {
        check(rung.cards.find("gfx") == std::string_view::npos &&
                  rung.label.find("gfx") == std::string_view::npos,
              "kArchLadder's sm_" + std::to_string(rung.sm) +
                  " row must not carry an AMD target: a ladder row is a compute capability a "
                  "device reports, and no AMD device reports one");
    }
    // The numbers an AMD target's name could be turned into must all miss.
    for (const int key : {906, 908, 910, 942, 1100, 1201}) {
        check(ninfer::caps::arch_rung(key) == nullptr,
              "sm-key " + std::to_string(key) +
                  " must NOT resolve: it is a gfx target's digits, not a compute capability");
    }
    // No seventh Cap bit. The AMD section adds none on purpose (a bit no floor row requires is
    // the decorative-bit defect this table already fixed once), and these two assertions are how
    // "none was added" stays true rather than being a comment.
    check(ninfer::caps::cap_name(static_cast<Cap>(1u << 5)) ==
              "nvfp4 e2m1 block-scaled mma (kind::mxf4nvf4)",
          "the highest Cap bit must still be Mxf4Nvfp4BlockScale");
    check(ninfer::caps::cap_name(static_cast<Cap>(1u << 6)) == "unknown capability",
          "Cap must have exactly six instruction-set bits: a seventh would be a capability "
          "class no row of this tree emits, i.e. the decorative bit this table already removed");
}

// Group 3: THE VERDICTS ARE DERIVED, NOT RESTATED. Walked over the ENUM, so a format appended
// to artifact/reader.h is covered here without anyone remembering to add it.
void test_amd_verdicts_derive_from_the_format_table() {
    const std::vector<NumericFormat> all_formats = all_formats_in_enum_order();
    std::size_t with_floor = 0;
    for (const NumericFormat format : all_formats) {
        const ninfer::caps::FormatRequirement* requirement =
            ninfer::caps::format_requirement(format);
        // THE RULE, WRITTEN OUT AS THE HEADER STATES IT -- and it has THREE cases, not two. The
        // first version of this check had two and the run went red on FP8_E4M3FN_ROW_F32S, which
        // is exactly the case the third exists for: a format this tree declares with NO KERNEL ON
        // ANY RUNG has `format_requirement() == nullptr`, and NoKernelInTree is right for it --
        // but it is right for its OWN reason (kUncoveredFormatNotes), not because of anything
        // about AMD, and rendering it as an AMD finding would be the misattribution this file
        // already warns about elsewhere. The check was corrected to the rule, not the rule to the
        // check.
        const bool has_floor = requirement != nullptr && requirement->required != Cap::None;
        if (has_floor) { ++with_floor; }
        const bool is_table_wide_refusal =
            ninfer::caps::uncovered_format_note(format) != nullptr;
        const auto expected = (has_floor || is_table_wide_refusal)
                                  ? ninfer::caps::AmdFormatVerdict::NoKernelInTree
                                  : ninfer::caps::AmdFormatVerdict::NotATensorCoreOperand;
        for (const std::string_view target :
             {"gfx906", "gfx908", "gfx90a", "gfx942", "gfx1100", "gfx1201"}) {
            const auto verdict = ninfer::caps::amd_format_verdict(target, format);
            check(verdict == expected,
                  std::string(format_name(format)) + " on " + std::string(target) +
                      ": verdict " +
                      std::string(ninfer::caps::amd_format_verdict_name(verdict)) +
                      " but its floor row gives " +
                      std::string(ninfer::caps::amd_format_verdict_name(expected)));
        }
        // A format with no kernel on ANY rung refuses on every target -- but it must refuse as a
        // NoKernelInTree, never as a NotATensorCoreOperand (which would read as "nothing stands in
        // its way") and never as an AMD-specific finding.
        if (requirement == nullptr) {
            check(ninfer::caps::uncovered_format_note(format) != nullptr,
                  std::string(format_name(format)) +
                      " has no floor row and no named refusal: the format gate would report "
                      "UnknownFormat for it on every GPU, AMD included");
            const auto verdict = ninfer::caps::amd_format_verdict("gfx906", format);
            check(verdict == ninfer::caps::AmdFormatVerdict::NoKernelInTree,
                  std::string(format_name(format)) +
                      " has no kernel on any rung, so the AMD answer must be NoKernelInTree and "
                      "not NotATensorCoreOperand");
        }
        // The blocker table must agree with the floor table EXACTLY -- a blocker row for a
        // format with no floor claims an AMD-only dependency that is not there, and a missing
        // blocker row for a format with a floor admits (silently) a format that needs ldmatrix.
        const ninfer::caps::AmdFormatBlocker* blocker = ninfer::caps::amd_format_blocker(format);
        check((blocker != nullptr) == has_floor,
              std::string(format_name(format)) +
                  (has_floor ? " has a tensor-core floor and must have an AMD blocker row"
                             : " has no tensor-core floor, so an AMD blocker row for it would "
                               "claim a dependency it does not have"));
        // A format this tree declares with NO kernel on any rung keeps ITS OWN refusal: the AMD
        // gate must not present kUncoveredFormatNotes' answer as its own finding.
        if (ninfer::caps::uncovered_format_note(format) != nullptr) {
            check(blocker == nullptr,
                  std::string(format_name(format)) +
                      " is a table-wide refusal, so it must not also carry an AMD blocker");
        }
    }
    check(with_floor == ninfer::caps::kAmdFormatBlockerCount,
          "kAmdFormatBlockerCount (" +
              std::to_string(ninfer::caps::kAmdFormatBlockerCount) +
              ") must equal the number of formats whose floor row requires a capability (" +
              std::to_string(with_floor) +
              "), counted over the ENUM rather than over a literal");
    check(with_floor > 0 && ninfer::caps::kAmdFormatBlockerCount > 0,
          "the blocker table must actually have rows, or the equality above is 0 == 0");
    // A target not in the table fails closed rather than borrowing a neighbour's answer.
    for (const NumericFormat format : {NumericFormat::BF16, NumericFormat::FP32}) {
        check(ninfer::caps::amd_format_verdict("gfx9999", format) ==
                  ninfer::caps::AmdFormatVerdict::UnknownTarget,
              "an unknown AMD target must yield UnknownTarget, never a neighbouring answer");
    }
}

// Group 4: THE UNPROBED STATE IS A VALUE, AND IT IS FALSE EVERYWHERE.
void test_amd_is_declared_unprobed() {
    std::size_t hits = 0;
    for (std::size_t raw = 0; raw < ninfer::caps::kPtxFamilyCount; ++raw) {
        const auto family = static_cast<ninfer::caps::PtxFamily>(raw);
        const ninfer::caps::PtxFamilyAmdStatus* status = nullptr;
        for (const ninfer::caps::PtxFamilyAmdStatus& candidate :
             ninfer::caps::kPtxFamilyAmdStatus) {
            if (candidate.family == family) { status = &candidate; }
        }
        check(status != nullptr,
              std::string(ninfer::caps::ptx_family_name(family)) + " must have an AMD status row");
        if (status == nullptr) { continue; }
        ++hits;
        // NotProbed != Supported, one table over: the AMD-side classification is EXTERNAL and
        // the field says so in a value rather than in a sentence a reader may skip.
        check(!status->measured_in_this_tree,
              std::string(ninfer::caps::ptx_family_name(family)) +
                  " must be marked NOT measured in this tree: no ROCm toolchain and no AMD "
                  "device exist on the machine this table was written on");
        check(status->provenance.find("EXTERNAL-UNPROBED") != std::string_view::npos,
              std::string(ninfer::caps::ptx_family_name(family)) +
                  "'s provenance must carry the EXTERNAL-UNPROBED label so it cannot be read as "
                  "a measurement made here");
        check(status->provenance.find("UNPROBED") == std::string_view::npos ||
                  status->provenance.find("EXTERNAL-UNPROBED") != std::string_view::npos,
              std::string(ninfer::caps::ptx_family_name(family)) +
                  " must not carry an unlabelled 'unprobed'");
    }
    check(hits == ninfer::caps::kPtxFamilyCount,
          "every PtxFamily must have exactly one status row: found " + std::to_string(hits) +
              " for " + std::to_string(ninfer::caps::kPtxFamilyCount) + " families");
    check(ninfer::caps::kPtxFamilyAmdStatusCount == ninfer::caps::kPtxFamilyCount,
          "the status table's own count must equal the enum's count");
    // The hard limit that is not a Cap bit is recorded, and only where it exists.
    check(ninfer::caps::kAmdLdsBlockerSiteCount > 0,
          "the >64KB dynamic-shared-memory limit must be recorded: no translator can see it and "
          "no capability bit can express it");
}

// Group 5: THE REFUSAL TEXT. A refusal that reads as a support claim, or that omits the missing
// kernel, is the failure this whole section was written to avoid -- so the text is checked too.
void test_amd_refusal_text_states_what_is_missing() {
    const std::string text = ninfer::caps::render_amd_format_refusal(
        "gfx906", NumericFormat::NVFP4);
    check(!text.empty(), "a NoKernelInTree answer must render a refusal");
    check(mentions(text, "gfx906"), "the refusal names the target");
    check(mentions(text, "nvfp4"), "the refusal names the weight format");
    check(mentions(text, "Vega20"), "the refusal names the ISA generation it is talking about");
    check(mentions(text, "no kernel in this tree"),
          "the refusal says the missing piece is a KERNEL, not a GPU");
    check(mentions(text, "kind::mxf4nvf4") || mentions(text, "block-scaled"),
          "the refusal names the capability the format needs");
    check(mentions(text, "nvfp4_w4a4_mma.cuh:308"),
          "the refusal cites the format's own kernel file:line");
    check(mentions(text, "ldmatrix"),
          "the refusal names ldmatrix, the family with NO AMD equivalent -- a reader told only "
          "about mma.sync would think the port is a rename");
    check(mentions(text, "mbarrier"),
          "the refusal names mbarrier, the second family with no AMD equivalent");
    check(mentions(text, "BUILD CONFIGURATION, NOT A SUPPORT CLAIM"),
          "the refusal must say in as many words that it is not a support claim");
    check(mentions(text, "probe on real hardware"),
          "the refusal must say what would move it, and that only a probe is evidence");
    check(mentions(text, "NONE"), "the refusal must report the absent probe as absent");
    // NEGATIVE CONTROLS. The text must not read as either an endorsement or a verdict on the
    // hardware: "no obstruction", "is supported", "works" are the words that would make it one.
    for (const std::string forbidden : {"is supported", "no obstruction", "will work", "works on",
                                        "supported on gfx"}) {
        check(!mentions(text, forbidden),
              "the refusal must not contain '" + forbidden +
                  "': that would be a hardware claim no probe on this machine can back");
    }
    // A format with no tensor-core floor has no AMD refusal, and asking for one is not an error.
    check(ninfer::caps::render_amd_format_refusal("gfx906", NumericFormat::FP32).empty(),
          "a format with no tensor-core operand must render nothing, like render_fallback_notice");
    check(ninfer::caps::render_amd_format_refusal("gfx906", NumericFormat::I64).empty(),
          "host metadata is not refused by this table: it is not a tensor-core operand");
    // The table-wide refusal keeps its own name, and is not restated as an AMD finding.
    const std::string f32s = ninfer::caps::render_amd_format_refusal(
        "gfx906", NumericFormat::FP8_E4M3FN_ROW_F32S);
    check(mentions(f32s, "kUncoveredFormatNotes"),
          "FP8_E4M3FN_ROW_F32S must be refused through its own table-wide note, not presented "
          "as an AMD-specific finding");
    // An unknown target fails closed by name.
    const std::string unknown =
        ninfer::caps::render_amd_format_refusal("gfx9999", NumericFormat::NVFP4);
    check(mentions(unknown, "gfx9999") && mentions(unknown, "not in kAmdLadder"),
          "an unknown AMD target must be refused BY NAME, not given a neighbouring answer");
}

// Group 6: ONE VOCABULARY. arch_caps.h cannot include kernel_route.h (kernel_route.h includes
// it), so the two NoKernelInTree spellings are tied by comparing the STRING each produces
// against the other rather than against a literal written here.
void test_no_kernel_in_tree_is_one_word() {
    check(std::string(ninfer::caps::amd_format_verdict_name(
              ninfer::caps::AmdFormatVerdict::NoKernelInTree)) ==
              std::string(ninfer::caps::outcome_name(ninfer::caps::RouteOutcome::NoKernelInTree)),
          "the AMD verdict and kernel_route.h's RouteOutcome must spell NoKernelInTree the same "
          "way: two vocabularies for one fact is how a tree starts disagreeing with itself");
    for (const ninfer::caps::AmdFormatVerdict verdict :
         {ninfer::caps::AmdFormatVerdict::NoKernelInTree,
          ninfer::caps::AmdFormatVerdict::NotATensorCoreOperand,
          ninfer::caps::AmdFormatVerdict::UnknownTarget}) {
        check(ninfer::caps::amd_format_verdict_name(verdict) != "unknown-verdict",
              "every AmdFormatVerdict enumerator needs a name");
    }
}

} // namespace

int main() {
    test_table_integrity();
    test_pinned_floors();
    test_verdicts();
    test_message_is_actionable();
    test_artifact_formats_scan();
    test_host_metadata_formats_are_covered();
    test_f32_scale_fp8_refuses_diagnosably();
    // The AMD rungs: a build configuration, not a support claim. Six groups, and the first one
    // is the one that makes the rest mean anything -- the marks are read back off the tree.
    test_amd_citations_are_real();
    test_amd_targets_are_not_ladder_rungs();
    test_amd_verdicts_derive_from_the_format_table();
    test_amd_is_declared_unprobed();
    test_amd_refusal_text_states_what_is_missing();
    test_no_kernel_in_tree_is_one_word();

    if (failures == 0) {
        std::cout << "arch_caps_test: all checks passed (ladder rungs="
                  << ninfer::caps::kArchLadderSize
                  << ", format rows=" << ninfer::caps::kFormatRequirementCount
                  << ", deliberate refusals=" << ninfer::caps::kUncoveredFormatCount
                  << ", enumerators=" << ninfer::caps::kFormatOrdinalCount
                  << ", AMD rungs=" << ninfer::caps::kAmdLadderSize
                  << " [build configuration, NOT a support claim; none probed]"
                  << ", AMD format blockers=" << ninfer::caps::kAmdFormatBlockerCount
                  << ", PTX sites=" << ninfer::caps::kPtxSiteCount
                  << ", LDS blockers=" << ninfer::caps::kAmdLdsBlockerSiteCount
                  << ", build arches='" << ninfer::caps::build_arch_list() << "')\n";
    }
    return failures == 0 ? 0 : 1;
}
