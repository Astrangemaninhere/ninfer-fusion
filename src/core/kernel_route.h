#pragma once

// kernel_route.h -- the CENTRAL per-architecture route selector that
// docs/maintainer/1cat-remaining-workpackages.md section 5.3 asks for:
//
//     route = select_route(device.sm(), profile, problem_shape);
//
// Three things this header is, and two it is not.
//
// 1. It is the ONE place where "which GEMM route does this (card, weight format,
//    problem shape) get?" is answered. Before it the only per-arch statement in the
//    tree was a blanket reject, and the per-format hardware floors lived only in prose
//    (design doc section 5.1/5.2).
// 2. It is table-driven, and its two tables are NOT new: it consumes
//      * src/core/arch_caps.h   kArchLadder         -- per-card capability set
//      * src/core/arch_caps.h   kFormatRequirements  -- per-format hardware floor, each
//        row carrying the file:line of the mma intrinsic the engine actually emits.
//    So a route decision is derived from the same kernel evidence the artifact-format
//    gate uses, instead of restating it in a second table that can go stale.
// 3. It is NON-GATING by construction. An unknown compute capability does NOT throw and
//    does NOT refuse: it selects the LOWEST-floor route and reports
//    RouteOutcome::UnknownArchFallback so the caller can warn. That is the section 5.3
//    principle "未知 sm 不再 throw, 退回保守路线并 warn". What still comes back as
//    NoKernelInTree is a route whose kernel is genuinely absent from this tree -- and
//    then `why` names the MISSING KERNEL, which is actionable, instead of naming the card.
//
// It is NOT a gate, and it does NOT decide whether an artifact may load. That is
// caps::require_artifact_formats_supported() at artifact-load time (src/targets/
// registry.cpp, construct_target). The two answer different questions over the same
// facts: "may this run at all" vs "which route does this run take". A route can be
// Selected on a card whose artifact gate refuses a different format.
//
// It is NOT a placement planner either. It answers for ONE card at a time. The
// multi-card question (which device runs which layers, and how the KV page stays atomic
// across them) is src/core/shard_plan.h.
//
// THE TOMBSTONE, named because the two mechanisms must not be confused:
// the deleted statement was, verbatim,
//     if (device.sm() != 120) { throw std::invalid_argument(
//         "this ninfer build targets compute capability 12.0 only; your GPU is sm_" ...); }
// It is STILL PRESENT in HEAD 3944a53 at src/targets/qwen3_6/impl/runtime/layouts_impl.h
// (the throw whose message begins "this ninfer build targets compute capability 12.0
// only") and is DELETED in the working tree, where the capability-probe gate
// (src/core/device_capabilities.h + src/core/device_probe.cu) replaced it. Both of those
// are GATES. This header is the route TABLE that was missing from both -- and the reason
// the sm() != 120 form was wrong is not that the number was wrong: it is that a number is
// not evidence for an instruction. `device.sm()` is major*10+minor (src/core/device.cu),
// so it cannot even see the '-a' suffix that decides whether the fp4/TMA kernels exist in
// the binary at all (tools/archkit/_GPU_MATRIX.md, "这张表的键是 (sm 号 + 特性后缀)").

#include "core/arch_caps.h"
// arch_sim.h includes arch_caps.h (never the reverse), so this include is acyclic and is
// what makes the simulation-aware overloads below expressible in this header.
#include "core/arch_sim.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::caps {

// ---------------------------------------------------------------------------
// The routes
// ---------------------------------------------------------------------------

// One enumerator per DISTINCT kernel family the tree can emit for a dense GEMM. Named
// for the mechanism, not for a card, so a row can be justified by a kernel file.
enum class KernelRoute : std::uint8_t {
    None = 0,          // no route: nothing in this tree can serve the request
    ConservativeSimt,  // lowest floor, no tensor-core requirement at all. It serves the
                       // floor-free formats (scale words / index payloads); it is NOT the
                       // unknown-card answer any more -- the only tensor-core-free GEMM in
                       // this tree is the QPN SIMT one, which no target compiles, so
                       // conservative_fallback() refuses instead of selecting this.
    QpnW4a16,          // 4-bit codes expanded inline into fp16 mma; the Volta route.
                       // NO CARD IN THIS BUILD GETS IT, and this route now has TWO
                       // independent obstructions that used to be stated as one:
                       //   (1) the kernels it names live in src/ops/linear/qpn/ and no CMake
                       //       target compiles that directory, so detail::qpn_slots() refuses
                       //       them and `why` names them; and
                       //   (2) the route is a TENSOR-CORE route only on a rung where
                       //       mma.sync.m8n8k4 actually lowers to a hardware instruction, and
                       //       that is NOT the same property as "the rung has fp16 mma": it is
                       //       measured per rung in kQpnMmaRungs below -- real on Volta and
                       //       Turing, EMULATED on the FP16x2 FMA pipe from Ampere on. The QPN
                       //       arm of select_route() therefore keys on the measured lowering
                       //       and not on Cap::Fp16Mma, which is what it used to do.
    MmaBf16,           // bf16 tensor-core mma; serves BF16 AND all groupwise-int formats
    Fp8A16Bf16,        // fp8 weights dequantized to bf16, then mma_bf16
    Fp8A8KindF8f6f4,   // kind::f8f6f4 A8 fp8 (a Blackwell form)
    // kind::mxf4nvf4.block_scale W4A4. MEASURED: that asm assembles on sm_120a / sm_121a
    // only (and their family targets); sm_100a/103a/110a reject it with "Instruction 'mma
    // with block scale' not supported". The earlier text here said "sm_100a / sm_120a",
    // which is what made the capability table hand this route to B200.
    Nvfp4W4a4Tma,
    Count,
};

[[nodiscard]] std::string_view route_name(KernelRoute route) noexcept;

// Why the answer is what it is. Deliberately three values and not a bool, because the
// three call for three different reactions:
//   Selected             -> log the line at info; nothing to do.
//   UnknownArchFallback  -> log the line at WARN. The run may proceed on the fallback.
//                           CURRENTLY UNPRODUCED: its only producer in this tree was
//                           conservative_fallback(), whose "fallback" is the QPN SIMT
//                           kernel (skinny_nvfp4_qpn_simt<M>) and no target compiles the
//                           QPN sources. A warning whose contract is "the run may proceed on
//                           the fallback" is worse than a refusal when there is no fallback,
//                           so that function returns NoKernelInTree instead. The value stays
//                           because the contract it names is real, it is what an
//                           almost-unknown card should get, and it becomes reachable again
//                           either when the QPN sources are compiled in -- and THEN ONLY on a
//                           rung whose measured mma lowering is HardwareMma884, because on a
//                           rung where ptxas SIMULATES that mma (kQpnMmaRungs) a "fallback"
//                           whose contract is "the run may proceed" would promise work on the
//                           FP16x2 pipe under a name that says tensor core, which is the same
//                           overclaim in a new place -- or when a tensor-core-free non-QPN SIMT
//                           kernel exists.
//   NoKernelInTree       -> the caller must refuse THAT format and quote `why`, which
//                           names the kernel that has to be written.
enum class RouteOutcome : std::uint8_t {
    Selected = 0,
    UnknownArchFallback,
    NoKernelInTree,
};

[[nodiscard]] std::string_view outcome_name(RouteOutcome outcome) noexcept;

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

// The problem shape, to the extent the route table actually keys on it. Only `m` and the
// (n, k) divisibility matter today, and both are stated as guards with evidence, not as
// guesses: see select_route().
struct ProblemShape {
    std::uint32_t m = 1; // rows. decode 1..3, verify/draft 4..16, prefill >= 17
    std::uint32_t n = 0; // output columns
    std::uint32_t k = 0; // reduction extent

    friend bool operator==(const ProblemShape&, const ProblemShape&) = default;
};

struct RouteChoice {
    KernelRoute route       = KernelRoute::None;
    RouteOutcome outcome    = RouteOutcome::NoKernelInTree;
    std::string_view kernel = {}; // the kernel/template, or {} when there is none
    std::string why         = {}; // always non-empty
    // True when this answer was produced for a SIMULATED compute capability rather than for
    // the card's own (src/core/arch_sim.h). A separate field rather than a note inside `why`
    // because the caller must be able to branch on it without parsing text: a simulated
    // answer is never a licence to claim anything about the real device. `why` also carries
    // the marker, so a log line is unambiguous even to a reader who cannot see the field.
    bool simulated          = false;

    // THE QPN FAMILY'S WIDE-M BAND, as a FIELD and not as a parse of `kernel`. The wide band
    // takes a DIFFERENT WEIGHT LAYOUT from the M 1..16 bands (the checkpoint-native planes vs
    // the qpn_prepack permutation), and the caller (qpn::dispatch_qpn_fallback) has to branch
    // on which one it holds. Deriving that branch from the text of `kernel` would be the
    // "label read as the mechanism" defect this project has paid for repeatedly. Only a
    // QpnW4a16 route can set it, and only the M 17..64 row does.
    bool wide_m             = false;

    [[nodiscard]] bool ok() const noexcept { return outcome != RouteOutcome::NoKernelInTree; }
    [[nodiscard]] bool warns() const noexcept {
        return outcome == RouteOutcome::UnknownArchFallback;
    }
};

// ---------------------------------------------------------------------------
// The selector
// ---------------------------------------------------------------------------

// Pure function of (compute capability, artifact weight format, shape). No CUDA header,
// no device query, no cudaGetDeviceProperties: the caller passes `sm` in, so every rung
// of the ladder -- including the cards this host does not have -- is exercisable on a
// single-GPU machine. That is the point of the parameterized form: "what route does a
// V100 take for an nvfp4 artifact" is a TABLE question, not a hardware question.
//
// `format` is the artifact's numeric format, i.e. the same key kFormatRequirements is
// keyed on, so the hardware floor comes from the kernel-evidence row rather than from a
// second, drift-prone list.
//
// `qpn_in_build` is the BUILD fact -- whether src/ops/linear/qpn/qpn_host.cu is compiled
// into this binary -- and it defaults to the constant CMake defines
// (caps::kQpnInBuild). It is a PARAMETER for one reason: it lets a host-only test exercise
// the with-QPN and without-QPN worlds in ONE binary, so the fail-closed half gets a real red
// control instead of an assertion about a binary nobody can build. Production callers use
// the default and cannot forget it.
[[nodiscard]] RouteChoice select_route(int sm, artifact::NumericFormat format, ProblemShape shape,
                                       bool qpn_in_build = kQpnInBuild) noexcept;

// The same question asked about a card the ladder does not contain, kept as its own
// entry point so a test can pin the fallback WITHOUT also pinning the ladder.
[[nodiscard]] RouteChoice conservative_fallback(int sm, ProblemShape shape,
                                                bool qpn_in_build = kQpnInBuild) noexcept;

// ---------------------------------------------------------------------------
// The selector, asked about a DEVICE rather than about a number
// ---------------------------------------------------------------------------
//
// The simulation-aware entry points. `arch_view_for_device()` (src/core/arch_sim.h) reads
// the test-only override and returns a three-state view; these overloads are what make the
// override reach the tables, and they are the ONLY way a simulated compute capability gets
// in. Three rules, and they are the guard properties restated at the call site:
//
//   * ArchView::Refused  -> NoKernelInTree. Fail closed: a request for a simulation that
//     could not be honoured must NOT produce the real device's answer, because the caller
//     asked a different question.
//   * ArchView::Active   -> the answer is produced for view.effective_sm and marked
//     `simulated = true`; `why` leads with the SIMULATED marker. Nothing about the real
//     device (view.physical_sm) is claimed.
//   * ArchView::Disabled -> byte-for-byte the behaviour of the plain overload.
[[nodiscard]] RouteChoice select_route(const ArchView& view, artifact::NumericFormat format,
                                       ProblemShape shape) noexcept;

// THE SAME QUESTION, WITH THE BUILD FACT SPELT OUT. The three-argument form above uses the
// macro caps::kQpnInBuild, which is correct only for a translation unit that RECEIVES it. A
// caller that can answer "is the QPN kernel in this binary" from something stronger than a
// macro -- the linker, by taking the address of ops::qpn::gemm_qpn -- passes the answer here.
// MEASURED 2026-09-18: without this overload the engine's own route-decision TU, which lives
// in ninfer_ops, saw kQpnInBuild == false while the kernel WAS in the binary it was linked
// into, and therefore refused a route the build could serve.
//
// It exists as an overload rather than as a caller-side copy of the simulated-answer
// decoration (the SIMULATED marker in `why`, `simulated = true`, the Refused passthrough)
// because that decoration is the part that must not be written twice.
[[nodiscard]] RouteChoice select_route(const ArchView& view, artifact::NumericFormat format,
                                       ProblemShape shape, bool qpn_in_build) noexcept;
[[nodiscard]] RouteChoice conservative_fallback(const ArchView& view,
                                                ProblemShape shape) noexcept;

// ---------------------------------------------------------------------------
// The selection log (design doc section 5.4 acceptance: "sm_120 上 route 选择日志")
// ---------------------------------------------------------------------------

// One line, stable enough to grep. Names the card, the format, the shape key the route
// keys on, the route, the kernel, and the outcome. `warn` is true exactly when the
// outcome is UnknownArchFallback, so the caller does not re-derive it from the text.
[[nodiscard]] std::string route_selection_line(int sm, artifact::NumericFormat format,
                                               ProblemShape shape, const RouteChoice& choice);

struct RouteLogLine {
    std::string text;
    bool warn = false;
};

[[nodiscard]] RouteLogLine route_log_line(int sm, artifact::NumericFormat format,
                                          ProblemShape shape, const RouteChoice& choice);

} // namespace ninfer::caps

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

namespace ninfer::caps {

inline std::string_view route_name(KernelRoute route) noexcept {
    switch (route) {
    case KernelRoute::None: return "none";
    case KernelRoute::ConservativeSimt: return "conservative-simt";
    case KernelRoute::QpnW4a16: return "qpn-w4a16";
    case KernelRoute::MmaBf16: return "mma-bf16";
    case KernelRoute::Fp8A16Bf16: return "fp8-a16-bf16";
    case KernelRoute::Fp8A8KindF8f6f4: return "fp8-a8-kind-f8f6f4";
    case KernelRoute::Nvfp4W4a4Tma: return "nvfp4-w4a4-tma";
    case KernelRoute::Count: break;
    }
    return "unknown-route";
}

inline std::string_view outcome_name(RouteOutcome outcome) noexcept {
    switch (outcome) {
    case RouteOutcome::Selected: return "selected";
    case RouteOutcome::UnknownArchFallback: return "unknown-arch-fallback";
    case RouteOutcome::NoKernelInTree: return "no-kernel-in-tree";
    }
    return "unknown-outcome";
}

// ---------------------------------------------------------------------------
// What mma.sync.m8n8k4 actually BECOMES on each rung -- MOVED
// ---------------------------------------------------------------------------
// This block (QpnMmaLowering, QpnMmaRung, kQpnMmaRungs, kQpnMmaRungCount,
// qpn_mma_rung(), qpn_mma_lowering_name(), qpn_arm_reachable()) now lives in
// src/core/arch_caps.h, UNCHANGED apart from added commentary. It moved because
// the artifact-format GATE needs the same fact the ROUTE selector needs --
// "does this rung execute the fp16 fallback's mma channel in hardware" -- and a
// second copy of the SASS census here would be a second answer to one question.
// arch_caps.h is included at the top of this file, so every name below still
// resolves through this header exactly as before.
// ---------------------------------------------------------------------------


namespace detail {

// The M split INSIDE the QPN family, from design doc section 4.3 item 1:
//   simt M<=3 / qpn<1> M4-8 / qpn<2> M9-16 / wmma >=17
// The kernels named here are the ones the QPN verification programs actually launched
// (qpn2_e2e.cu launches skinny_nvfp4_qpn2<1, 1>; qpn_mma.cu launches skinny_nvfp4_qpn<2>).
// The host wrapper's guards, verbatim (src/ops/linear/qpn/qpn_host.cu):
//   :15   if (m < 1 || m > 3) throw std::invalid_argument("qpn_simt supports M 1..3");
//   :45   if (m < 1) throw std::invalid_argument("gemm_qpn: M must be >= 1");
//   :46   if (k % 64 != 0 || n % 32 != 0) throw std::invalid_argument("gemm_qpn: K%64 N%32");
//   :47-63 M<=3 -> gemm_qpn_simt; M<=8 -> skinny_nvfp4_qpn<1>; M<=16 -> skinny_nvfp4_qpn<2>
//   :65   throw std::invalid_argument("gemm_qpn: M > 16 belongs to the wmma band (17..64)");
// so gemm_qpn DOES reach M 1..16, and the M>=17 row below is the one band the family
// genuinely lacks. (The earlier text here said "no gemm_qpn/qpn2 entry exists" and "M > 3
// has NO host entry point today" -- both were stale: that entry was added on 2026-09-10,
// and this function emitted skinny_nvfp4_qpn2 for M 4..8 three arms away from the claim.)
//
// AND IT REFUSES ALL OF THEM ANYWAY, because they are not in THIS BUILD. The kernels live
// in src/ops/linear/qpn/qpn_kernels.cuh (skinny_nvfp4_qpn_simt<M> :985,
// skinny_nvfp4_qpn2<SPLITK, NACC> :1077, skinny_nvfp4_qpn<MT> :903) with the host entry
// gemm_qpn in src/ops/linear/qpn/qpn_host.cu, and NOTHING under src/ops/linear/qpn/ is in
// any CMake target: `qpn` has 0 hits in src/CMakeLists.txt, whose ninfer_ops source list is
// deliberately explicit -- src/CMakeLists.txt:75-76 says "adding a source is a
// build-boundary decision, not an accidental recursive-glob side effect" -- and no GLOB or
// file(GLOB) appears anywhere in it. MEASURED: 0 qpn object members across all 11 archives
// under build/src/ (247 members in libninfer_ops.a alone), and 0 qpn entries among
// build/compile_commands.json's 503 TUs, with gqa_attention at 52 as the positive control.
// A route SELECTED here would name a kernel the caller cannot launch, and
// RouteOutcome::NoKernelInTree exists for exactly that ("`why` names the MISSING KERNEL",
// this file's header). So every arm below refuses, and `why` names the kernel.
//
// TO RESTORE THE SELECTED ARMS: put ops/linear/qpn/qpn_host.cu into a target and flip the
// three bands below back. The condition is a measured build fact, not a mood, which is why
// it is deliberately NOT a preprocessor switch: a switch that can be turned on without
// compiling the kernel is how a phantom route gets in. (The W4 landing exists and was lost
// in transit -- origin/main commit b8ecb05 added exactly that one source line to
// ninfer_ops, and this lineage was imported without it.) Port plan:
// docs/maintainer/1cat-remaining-workpackages.md section 4.3.

// The one sentence every QPN refusal carries, so `why` always says WHAT to write (the
// actionable half) instead of only WHICH card asked (which is not actionable). It is a
// string_view constant so a host-only probe TU can static_assert its content without
// running anything.
// The QPN family, named once so both worlds below name the same files.
inline constexpr std::string_view kQpnFamily =
    "the QPN W4A16 family -- skinny_nvfp4_qpn_simt<M> (M 1..3), skinny_nvfp4_qpn<1> "
    "(M 4..8), skinny_nvfp4_qpn<2> (M 9..16) and the fp8 arm skinny_fp8_qpn8 / "
    "skinny_fp8_qpn8_mt2 in src/ops/linear/qpn/qpn_kernels.cuh, host entry gemm_qpn in "
    "src/ops/linear/qpn/qpn_host.cu";

// THE REASON WHEN THE KERNEL IS NOT IN THIS BUILD. Kept as the fail-closed text: a route
// whose kernel is absent must name the kernel, and it must say WHAT TO DO (add the source)
// rather than only WHICH card asked.
inline constexpr std::string_view kQpnMissingKernelReason =
    "Missing kernel, and it is a build fact rather than a GPU fact: the QPN W4A16 family -- "
    "skinny_nvfp4_qpn_simt<M> (M 1..3), skinny_nvfp4_qpn2<1, 1> (M 4..8), "
    "skinny_nvfp4_qpn<2> (M 9..16) in src/ops/linear/qpn/qpn_kernels.cuh, host entry "
    "gemm_qpn in src/ops/linear/qpn/qpn_host.cu -- exists in this tree and is compiled by NO "
    "target in it: src/ops/linear/qpn/ is absent from every CMakeLists, and this build's "
    "archives and compile_commands.json carry no QPN object at all. Add "
    "ops/linear/qpn/qpn_host.cu to a target to make this route selectable. Port plan: "
    "docs/maintainer/1cat-remaining-workpackages.md section 4.3.";

// THE REASON WHEN THE KERNEL IS NOT THE PROBLEM. Reachable only once the source is in a
// target; it is what the EMULATED tier (sm_80 and up, kQpnMmaRungs) reports, and what an
// unmeasured rung reports. Deliberately NOT the build sentence, because saying "no target
// compiles it" in a build that DOES compile it is exactly the kind of stale refusal text
// that made the previous revision of this file name a kernel three arms away from the claim.
inline constexpr std::string_view kQpnNotTheBuildReason =
    "The QPN sources ARE in this build (ops/linear/qpn/qpn_host.cu is in ninfer_ops, "
    "NINFER_HAVE_QPN), so this refusal is NOT a build fact: it is the measured tensor-core "
    "channel. src/ops/linear/qpn/qpn_kernels.cuh reaches the tensor core through exactly one "
    "channel, mma.sync.aligned.m8n8k4, and whether that lowers to a hardware HMMA.884 is a "
    "per-rung SASS fact recorded in kQpnMmaRungs (src/core/arch_caps.h) -- not a property of "
    "the instruction, which assembles on every rung from sm_70 to sm_120a (measured).";

// Returns std::string because the not-the-build branch APPENDS the family name, and a
// constexpr string_view cannot concatenate a std::string. Every caller already wraps the
// result in std::string(...), so the change is invisible at the call sites.
[[nodiscard]] inline std::string qpn_missing_reason(bool qpn_in_build) {
    if (!qpn_in_build) { return std::string(kQpnMissingKernelReason); }
    // `why` must always say WHAT to write, not only WHICH card asked -- the file's own rule.
    // The sentence above explains why the build is not the obstruction; this clause names the
    // kernel anyway, so a reader who greps a refusal for a kernel symbol still finds one.
    return std::string(kQpnNotTheBuildReason) +
           " THE MISSING KERNEL IS NAMED ANYWAY, because the reason has to say what to write "
           "and not only which card asked: it is " +
           std::string(kQpnFamily) + ".";
}

// TWO INDEPENDENT QUESTIONS, TWO PARAMETERS, and conflating them was a real defect:
//   `fallback_applies` -- the caller's answer to caps::fp16_fallback_executable() for THIS
//     (sm, format, build). It is the ONLY thing that may produce a Selected route here.
//   `qpn_in_build`     -- which REASON TEXT the refusing paths use.
// The geometry guard, the M-split table and the band names are the same in every world, so the
// split stays pinned by tests/test_kernel_route.cpp regardless of either flag. An earlier
// version took only `qpn_in_build` and selected on it, which made a BUILD fact answer a
// FORMAT question: after the fp8 fallback row was withdrawn, fp8 still came back Selected.
[[nodiscard]] inline RouteChoice qpn_slots(std::uint32_t m, std::uint32_t n, std::uint32_t k,
                                           bool fallback_applies, bool qpn_in_build) {
    RouteChoice out;
    out.route   = KernelRoute::None;
    out.outcome = RouteOutcome::NoKernelInTree;
    out.kernel  = {};
    // The geometry guard comes FIRST, ahead of every M band, and is quoted from the wrapper
    // rather than inferred:
    //   `if (k % 64 != 0 || n % 32 != 0) throw std::invalid_argument("qpn_simt: K%64 N%32");`
    // It is the QPN slot geometry (tile N/32, group K/16), so a shape that fails it has no
    // QPN route at any M -- in this build or in one that compiles the kernels. Keeping the
    // check ahead of the bands is what keeps the reason specific: re-tiling the problem and
    // registering the file are different fixes and are not interchangeable. (Dropping the
    // guard would not help either: the slot map is what the intrinsic's fragment order
    // needs.)
    if (k % 64U != 0U || n % 32U != 0U) {
        out.why = "the QPN slot geometry requires k % 64 == 0 and n % 32 == 0 (the wrapper "
                  "guard in the qpn host entry: \"qpn_simt: K%64 N%32\"); this shape has k=" +
                  std::to_string(k) + " n=" + std::to_string(n) +
                  ". Re-tile the problem, or add a kernel that handles the remainder. " +
                  std::string(qpn_missing_reason(qpn_in_build));
        return out;
    }
    if (m > 16U) {
        // -------------------------------------------------------------------------
        // THE WIDE-M BAND (M 17..64), and it is a BAND rather than an absence.
        // -------------------------------------------------------------------------
        // This row used to read "M >= 17 (prefill) has no route in the QPN family at all ...
        // the v100-skinny policy of deferring M>=17 to marlin has no marlin in
        // src/ops/linear/qpn/". The marlin half is TRUE and irrelevant: marlin is a vLLM
        // W4A16 kernel this lineage never imported (docs/reference/1cat-tp/
        // v100-skinny_fork_patches_marlin.py is a reference copy in no build target). The
        // other half was FALSE: the family's wide-M kernel is skinny_nvfp4_wmma<WN, WM, KC>
        // in qpn_kernels.cuh, its frontier row is "simt M<=3, qpn 4..16, wmma 17..64", and
        // it has a host entry since 2026-09-18 (ops::qpn::gemm_qpn_wmma_native). Only the
        // host arm had been missing, and a table that says "the family lacks it" about a
        // kernel in the same directory is the kind of stale refusal this project has
        // already paid for once.
        //
        // THE CHANNEL IS NOT THE SAME CHANNEL, and that is load-bearing rather than a
        // detail: the M 1..16 bands reach the tensor core through mma.sync.m8n8k4 (the
        // channel kQpnMmaRungs measures), the wide band through nvcuda::wmma m16n16k16 -- a
        // DIFFERENT channel of the same family, which arch_caps.h:333-339 states in so many
        // words. So `fallback_applies` (an m8n8k4-tier fact) is the right gate for the
        // M 1..16 bands and an approximation for the wide band; the rows below say which
        // one they are reading rather than silently reusing the tier.
        if (m > 64U) {
            // The ceiling is the KERNEL's, not the build's: skinny_nvfp4_wmma stages
            // WM*16 = 64 activation rows per CTA and reads A at m * K for its whole tile,
            // with no m-tile loop. TRUE IN BOTH WORLDS, which is why it does not consult
            // qpn_in_build.
            out.why = "M " + std::to_string(m) + " is above the QPN family's wide-M band. "
                      "skinny_nvfp4_wmma<WN, WM, KC> (src/ops/linear/qpn/qpn_kernels.cuh) "
                      "stages WM*16 = 64 activation rows per CTA and has no m-tile loop, so "
                      "M > 64 is outside the kernel -- a kernel fact, not a build fact. "
                      "Missing kernel: a tiled wide-M entry (an m-tile loop over "
                      "skinny_nvfp4_wmma) for M > 64. For the record, the band this row "
                      "used to name: the v100-skinny lineage served M >= 17 by deferring to "
                      "vLLM's marlin (docs/reference/1cat-tp/"
                      "v100-skinny_fork_patches_marlin.py, a reference copy in no target), "
                      "and M 17..64 is served here by this family's own wmma kernel instead "
                      "of by marlin.";
            return out;
        }
        if (!fallback_applies) {
            // The wide band's kernel IS in the tree and its channel is a real tensor-core
            // instruction on the tiers kQpnMmaRungs measured; what is false here is the
            // CALLER's condition (a build without the family, or a format with no launchable
            // lower-floor kernel). The caller composes the other half of this reason, so
            // this branch only has to stop claiming the BAND is the problem.
            out.why = "M " + std::to_string(m) + " falls in the QPN family's wide-M band "
                      "(skinny_nvfp4_wmma<WN, WM, KC>, M 17..64), whose kernel is in this "
                      "tree and whose channel -- nvcuda::wmma m16n16k16, a different channel "
                      "from the m8n8k4 one the M 1..16 bands use -- is measured HARDWARE on "
                      "every rung from sm_70 to sm_103 (kQpnMmaRungs: the HMMA.16816 the "
                      "emulated rows leave standing are this channel's). So this refusal is "
                      "not about the band. " +
                      qpn_missing_reason(qpn_in_build);
            return out;
        }
        out.outcome = RouteOutcome::Selected;
        out.route   = KernelRoute::QpnW4a16;
        out.kernel  = "skinny_nvfp4_wmma<WN, WM, KC> (M 17..64)";
        out.wide_m  = true;
        out.why = "M " + std::to_string(m) + " falls in the QPN family's wide-M band "
                  "(skinny_nvfp4_wmma<WN, WM, KC>, M 17..64; frontier qpn_sweep_20260810: "
                  "simt M<=3, qpn 4..16, wmma 17..64), and the family IS in this build "
                  "(host entry ops::qpn::gemm_qpn_wmma_native, "
                  "src/ops/linear/qpn/qpn_host.cu). LAYOUT: this band consumes the "
                  "checkpoint-NATIVE codes[N][K/2] + scales[N][K/16] planes, NOT the "
                  "qpn_prepack permutation the M 1..16 bands take, so the native -> "
                  "qpn_prepack weight converter qpn_arch_route.h lists as missing is not "
                  "needed here. CHANNEL: nvcuda::wmma m16n16k16, a different channel from "
                  "the m8n8k4 one the M 1..16 bands use (arch_caps.h:333-339). Selected: " +
                  std::string(out.kernel) + ".";
        return out;
    }
    // The shape is inside the family's own window. Name the band FIRST -- it is the kernel
    // that has to exist for this route to be real, and it is what the refusal has to say
    // when the build does not contain it.
    std::string band = (m <= 3U) ? "simt band (skinny_nvfp4_qpn_simt<M>, M 1..3)"
                                 : (m <= 8U) ? "qpn2 band (skinny_nvfp4_qpn<1>, M 4..8)"
                                             : "MT=2 band (skinny_nvfp4_qpn<2>, M 9..16)";
    if (!fallback_applies) {
        // THE REFUSING BRANCH, and it is reached for two different situations that share one
        // question -- "does this (rung, format) have a launchable lower-floor kernel HERE?":
        // a build without the QPN sources at all, and a build WITH them but a format whose
        // fallback_required row is Cap::None (fp8, today: its QPN kernels have no host entry).
        // The two get different text, and qpn_missing_reason picks it, because the actionable
        // half differs: add the source line, versus add the dispatch.
        out.why = "M " + std::to_string(m) + " falls in the QPN family's " + band + ". " +
                  std::string(qpn_missing_reason(qpn_in_build));
        return out;
    }
    // THE SELECTING BRANCH. Reached only when the caller has already established, through
    // caps::fp16_fallback_executable(), that (a) the format HAS a fallback row, (b) the QPN
    // sources are in this build and (c) this rung's measured lowering of the m8n8k4 channel
    // is HardwareMma884 (HMMA.884, a real tensor core -- measured on sm_70 and sm_75 by the
    // SASS census in kQpnMmaRungs). On any other rung the caller does not get here, so this
    // branch cannot hand out a tensor-core route that would execute on the FMA pipe.
    out.outcome = RouteOutcome::Selected;
    out.route   = KernelRoute::QpnW4a16;
    out.kernel  = (m <= 3U)   ? "skinny_nvfp4_qpn_simt<M> (M 1..3)"
                  : (m <= 8U) ? "skinny_nvfp4_qpn<1> (M 4..8)"
                              : "skinny_nvfp4_qpn<2> (M 9..16)";
    out.why = "M " + std::to_string(m) + " falls in the QPN family's " + band +
              ", and the family IS in this build (ops/linear/qpn/qpn_host.cu -> ninfer_ops, "
              "host entry gemm_qpn), and this rung's mma.sync.aligned.m8n8k4 channel is a "
              "HARDWARE instruction here (HMMA.884; measured per rung in kQpnMmaRungs, "
              "src/core/arch_caps.h). Selected: " + std::string(out.kernel) + ".";
    return out;
}

} // namespace detail

inline RouteChoice conservative_fallback(int sm, ProblemShape shape, bool qpn_in_build) noexcept {
    RouteChoice out;
    // The lowest-floor kernel this tree EMITS is still the tensor-core-free SIMT path -- and
    // that path is the QPN one (skinny_nvfp4_qpn_simt<M>). Nothing under
    // src/ops/linear/qpn/ is in a CMake target, so UnknownArchFallback would be a WARNING
    // whose contract is "the run may proceed on the fallback" for a fallback that does not
    // exist in this build. A route is a plan, not a permission, and a plan that cannot be
    // carried out is a refusal with a reason -- so this refuses, and the reason names the
    // missing kernel rather than the card. Do not "fix" this by selecting
    // KernelRoute::ConservativeSimt with an empty kernel: `ok()` would then say the run is
    // fine and the caller would have nothing to launch.
    out.route   = KernelRoute::None;
    out.outcome = RouteOutcome::NoKernelInTree;
    out.kernel  = {};
    out.why = "compute capability sm_" + std::to_string(sm) +
              " has no row in kArchLadder (src/core/arch_caps.h), and the requested shape "
              "has m=" + std::to_string(shape.m) +
              ". The conservative answer for a card whose tensor-core set has never been "
              "measured is the tensor-core-free SIMT path, and this tree has exactly one: "
              "the QPN SIMT kernel. " + std::string(detail::qpn_missing_reason(qpn_in_build)) +
              " NOTE that even with the QPN sources in this build this refusal stands: the "
              "fallback's executability is capped by a MEASURED per-rung lowering "
              "(kQpnMmaRungs), and a compute capability with no ladder row has no measured "
              "lowering, so there is nothing to check the channel against. Add the card's "
              "measured tensor-core set to kArchLadder before trusting any route on it, and "
              "do NOT assume a neighbouring row's set -- sm_120 and sm_120a share the number "
              "120 and do not share the fp4/TMA kernels.";
    return out;
}

namespace detail {

// THE ONE BODY, keyed on a RUNG instead of on a NUMBER. Namespaced `detail` so it is not a
// third public entry point: the two overloads below are the only ways in, and one of them
// takes a measurement. The 230 lines inside are otherwise unchanged.
inline RouteChoice select_route_on_rung(const ArchRung* rung, int sm,
                                        artifact::NumericFormat format, ProblemShape shape,
                                        bool qpn_in_build) noexcept {

    const FormatRequirement* requirement = format_requirement(format);
    if (requirement == nullptr) {
        RouteChoice out;
        out.outcome = RouteOutcome::NoKernelInTree;
        // NO ROW HAS TWO DIFFERENT REASONS AND THEY MUST NOT READ ALIKE (arch_caps.h,
        // kUncoveredFormatNotes): a format this tree DECLARES and deliberately has no kernel for,
        // where the missing piece is the KERNEL -- or a format nobody has accounted for, where the
        // missing piece is the row. Answering the first with the second's text sends the operator
        // to add the one row that would turn a loud refusal into a false claim of support.
        const UncoveredFormatNote* uncovered = uncovered_format_note(format);
        if (uncovered != nullptr) {
            out.why = "artifact format " + std::string(artifact::format_name(format)) +
                      " has NO KERNEL in this tree that can execute it on any rung, so it "
                      "deliberately has no kFormatRequirements row and no route exists for it on "
                      "ANY compute capability. " + std::string(uncovered->reason) +
                      " Do NOT treat a kFormatRequirements row as the fix: the missing piece is "
                      "the kernel named above, not the table.";
            return out;
        }
        out.why = "artifact format " + std::string(artifact::format_name(format)) +
                  " has no row in kFormatRequirements (src/core/arch_caps.h): this is a "
                  "TABLE defect, not a GPU fact, and the table is what fixes the floor. "
                  "Missing entry: a kFormatRequirements row naming the mma intrinsic that "
                  "consumes this format.";
        return out;
    }

    // Scale words and index payloads are never tensor-core operands, so they have no floor
    // and no route. Returning a route for them would invent a kernel that does not exist.
    if (requirement->required == Cap::None) {
        RouteChoice out;
        out.route   = KernelRoute::ConservativeSimt;
        out.outcome = RouteOutcome::Selected;
        out.kernel  = "(no tensor-core operand)";
        out.why = std::string(artifact::format_name(format)) +
                  " is scale words / control or index payloads (" +
                  std::string(requirement->kernel_evidence) +
                  "): it is never fed to a tensor-core GEMM, so no route is required and "
                  "none is selected.";
        return out;
    }

    const bool floor_met = covers(rung->caps, requirement->required);

    if (floor_met) {
        RouteChoice out;
        out.outcome = RouteOutcome::Selected;
        switch (format) {
        case artifact::NumericFormat::NVFP4:
            out.route  = KernelRoute::Nvfp4W4a4Tma;
            out.kernel = "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh -> ops/common/mma.cuh "
                         "(kind::mxf4nvf4.block_scale)";
            break;
        case artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S:
            // The A8 form is a Blackwell form; only take it when the card has it.
            if (has_cap(rung->caps, Cap::Fp8F8f6f4KindMma)) {
                out.route  = KernelRoute::Fp8A8KindF8f6f4;
                out.kernel = "ops/linear/fp8/fp8_a8_mma.cuh:247 (mma_fp8_e4m3, kind::f8f6f4)";
            } else {
                out.route  = KernelRoute::Fp8A16Bf16;
                out.kernel = "ops/linear/fp8/fp8_a16_gemm_mma.cuh:206 (mma_bf16 after "
                             "dequant)";
            }
            break;
        default:
            out.route  = KernelRoute::MmaBf16;
            out.kernel = requirement->kernel_evidence;
            break;
        }
        out.why = std::string("sm_") + std::to_string(sm) + " (" + std::string(rung->label) +
                  ") covers the floor " + std::string(cap_name(requirement->required)) +
                  " that " + std::string(artifact::format_name(format)) + " needs; kernel " +
                  std::string(requirement->kernel_evidence);
        return out;
    }

    // Floor not met. Two very different answers, and the difference is the whole point of
    // having a capability set instead of a version number:
    if (has_cap(rung->caps, Cap::Fp16Mma)) {
        switch (format) {
        case artifact::NumericFormat::NVFP4:
        case artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S: {
            // -----------------------------------------------------------------
            // THE FALLBACK EXIT.
            // -----------------------------------------------------------------
            // This is the fp16 fallback route, and it is checked FIRST, before any of the
            // graded refusals below, because when it is true none of them apply: the card
            // has a kernel that consumes this artifact's own bytes and the channel that
            // kernel uses is a hardware instruction here.
            //
            // EVERY clause lives in caps::fp16_fallback_executable() so the gate and this
            // selector cannot disagree: (1) the QPN sources are in this build, (2) the format
            // declares a fallback floor in kFormatRequirements, (3) the rung covers it, and
            // (4) the rung's MEASURED m8n8k4 lowering is HardwareMma884. (4) is what makes
            // this branch unreachable on sm_80 and up -- measured there as a CALL into a
            // software FFMA routine -- so the route this returns can never be a tensor-core
            // label on an FMA-pipe execution.
            //
            // The QPN family is NOT card-specific: it is one implementation that happens to
            // be REAL on Volta and Turing, which is exactly the "must migrate easily to other
            // cards" requirement. Nothing here special-cases sm_70.
            if (fp16_fallback_executable(sm, format, qpn_in_build)) {
                RouteChoice out =
                    detail::qpn_slots(shape.m, shape.n, shape.k,
                                      /*fallback_applies=*/true, qpn_in_build);
                if (out.outcome == RouteOutcome::Selected) {
                    out.why = std::string("sm_") + std::to_string(sm) + " (" +
                              std::string(rung->label) + ") does not meet " +
                              std::string(artifact::format_name(format)) +
                              "'s own floor (" + std::string(cap_name(requirement->required)) +
                              "), and the fp16 fallback route DOES apply here, so this is a "
                              "Selected route rather than a refusal. " + out.why +
                              " No requantization and no fp16 weight copy: the published "
                              "codes this artifact already stores are expanded inline. " +
                              std::string(qpn_mma_lowering_name(
                                  qpn_mma_rung(sm)->lowering)) +
                              " on this rung -- measured: " + std::string(qpn_mma_rung(sm)->evidence);
                }
                return out;
            }
            // The Volta route, quoted from tools/archkit/_GPU_MATRIX.md's sm-70 row:
            //   "QPN2 W4A16: 直接跑已发布 NVFP4/FP8 混合权重, 4bit 码内联展开喂 fp16 mma,
            //    无需重量化/无需 fp16 副本"
            // (i.e. run the published mixed NVFP4/FP8 weights as-is: expand the 4-bit codes
            // inline into fp16 mma, no requantization and no fp16 weight copy.) THAT IS THE
            // ONE CASE where a lower rung would have a better answer than a refusal -- and
            // the other half of the same citation is the STATUS column of the row being
            // quoted: it reads "pending port", i.e. _GPU_MATRIX.md:15 does not mark sm_70's
            // QPN2 W4A16 as shipped. This build is where the port is still pending (nothing
            // under src/ops/linear/qpn/ is in a target), so the better answer is not
            // available and the honest answer is a refusal that NAMES the kernel instead of
            // naming the card.
            //
            // AND THE ARM IS GRADED BY THE MEASURED LOWERING, not by Cap::Fp16Mma. This used
            // to be one arm reached by every rung with fp16 mma (75/80/86/87/88/89/90/100/103)
            // and it handed all of them the sm-70 text. That is wrong for the majority of
            // them: on sm_80 and up ptxas does NOT lower mma.sync.m8n8k4 to a hardware
            // instruction, it CALLs an FP16x2 FMA simulation (kQpnMmaRungs), so "it has fp16
            // mma, therefore the QPN W4A16 route is the better answer" is a claim about a
            // tensor core being made out of an emulation. The two tiers now get two
            // different answers, and the difference is not cosmetic: it is the difference
            // between "port the kernel, the route is real" and "do not port it for this
            // reason, requantize instead".
            //
            // Neither tier is Selected, and that is deliberate, not an oversight. A Selected
            // route is a plan the caller may carry out (`ok()` is what the caller branches on)
            // and the QPN kernel is compiled by no target in this build, so selecting it
            // would make ok() true for a kernel that cannot be launched -- the phantom route
            // this file's header calls out. The emulated tier has a SECOND, independent
            // reason not to select (above). What the grading changes is the actionable half
            // of `why`; the outcome is the same refusal on both tiers for the BUILD reason,
            // and on the emulated tier for the ARCH reason as well.
            const QpnMmaRung* lowering = qpn_mma_rung(sm);
            if (lowering == nullptr) {
                // A rung of the ladder whose lowering nobody measured. Same rule as a rung
                // with no capability set: say which measurement is missing, do not assume the
                // family's. This is the branch the two rungs A2 never reached land in.
                RouteChoice out;
                out.outcome = RouteOutcome::NoKernelInTree;
                out.why = std::string("sm_") + std::to_string(sm) + " (" +
                          std::string(rung->label) + ") has no " +
                          std::string(cap_name(requirement->required)) +
                          " for " + std::string(artifact::format_name(format)) +
                          ", and the one route in this tree that asks for less than the "
                          "format's floor is the QPN W4A16 family -- whose mma has an "
                          "UNMEASURED lowering on this rung. That is a MISSING MEASUREMENT "
                          "and not a card fact: kQpnMmaRungs (src/core/kernel_route.h) has a "
                          "row for every rung whose SASS has been read, and this one is not "
                          "in it. Missing measurement: `nvlink` + `nvdisasm -c` on the "
                          "instantiation probe, then count HMMA.884 against CALL.REL.NOINC to "
                          "__cuda_sm_8x_mma_row_col_f32_f16_f16_f32. Do NOT copy the "
                          "neighbouring row's lowering -- sm_120 and sm_120a share the number "
                          "120 and do not share the kernels, which is the same trap one level "
                          "down.";
                return out;
            }
            if (lowering->lowering == QpnMmaLowering::HardwareMma884) {
                // THE REAL TENSOR-CORE TIER, AND IT NO LONGER SELECTS. Reaching this branch
                // means the fallback PREDICATE was false while the channel is hardware, so the
                // QPN route is the right answer IN PRINCIPLE and something else is missing.
                // Two shapes, and the reason says which: the family is not in this build, or
                // it is and THIS FORMAT has no fallback row (no launchable lower-floor kernel).
                // It must not select on either -- a build fact may not close a format
                // question, and this branch is where that was measured happening.
                RouteChoice out = detail::qpn_slots(shape.m, shape.n, shape.k,
                                                    /*fallback_applies=*/false, qpn_in_build);
                out.why = std::string("sm_") + std::to_string(sm) + " (" +
                          std::string(rung->label) + ") has no " +
                          std::string(cap_name(requirement->required)) +
                          ", and it is the RIGHT kind of rung for the fp16 fallback: " +
                          std::string(artifact::format_name(format)) +
                          " would take the QPN W4A16 route (4-bit codes expanded inline into "
                          "fp16 mma; no requantization and no fp16 weight copy), and " +
                          std::string(qpn_mma_lowering_name(lowering->lowering)) +
                          " on this rung -- measured: " + std::string(lowering->evidence) +
                          ". So this refusal is NOT about the card. " +
                          (qpn_in_build
                               ? "The QPN sources ARE in this build, so what is missing for "
                                 "THIS FORMAT is a launchable kernel: "
                               : "What is missing is the build fact and then the kernel: ") +
                          out.why;
                return out;
            }
            // THE EMULATED TIER: the fp16 mma exists, but ptxas answers the QPN kernels' PTX
            // with an FMA routine, so this is NOT the rung where the quoted route text holds.
            // Not selected, and NOT the QPN route as the recommended answer either -- the
            // recommendation has to be something this card can actually execute.
            RouteChoice out;
            out.outcome = RouteOutcome::NoKernelInTree;
            out.why = std::string("sm_") + std::to_string(sm) + " (" +
                      std::string(rung->label) + ") has no " +
                      std::string(cap_name(requirement->required)) +
                      " for " + std::string(artifact::format_name(format)) +
                      ", and it is NOT given the QPN W4A16 route either: that route's whole "
                      "premise is fp16 TENSOR-CORE mma, and on this rung the m8n8k4 channel "
                      "is not a hardware instruction -- ptxas answers it with a CALL into a "
                      "software FFMA routine that it emits into the same cubin. This is " +
                      std::string(qpn_mma_lowering_name(lowering->lowering)) +
                      " here, measured: " + std::string(lowering->evidence) +
                      ". The refusal is about the CHANNEL, not about the build: " +
                      (qpn_in_build
                           ? std::string("the QPN sources ARE in this build "
                                         "(ops/linear/qpn/qpn_host.cu is in ninfer_ops, "
                                         "NINFER_HAVE_QPN), so NO SOURCE LINE IS MISSING HERE. ")
                           : std::string("and the sources are not in this build either "
                                         "(nothing under src/ops/linear/qpn/ is in a CMake "
                                         "target), so this build is missing the channel AND "
                                         "the source line. ")) +
                      "Even a ported kernel would run on the FMA pipe here, so selecting it "
                      "would name a tensor-core route and deliver a simulation. ";
            if (has_cap(rung->caps, Cap::Bf16Mma)) {
                // The rungs that reach this arm through NVFP4 all have bf16 mma, so there IS
                // a requantization fix on this card and it is Selected, not refused. Named
                // rather than assumed: the sentence changes if a future rung lacks it.
                out.why +=
                    "What to do instead on THIS card: requantize the weight to a format whose "
                    "floor this rung does cover. The groupwise-int forms (Q4G64_F16S / "
                    "Q5G64_F16S / Q6G64_F16S / W8G32_F16S) bottom out in mma_bf16, the rung "
                    "covers Bf16Mma, and select_route() returns Selected + mma-bf16 for them "
                    "here -- so the artifact runs, today, with no new kernel. Running the "
                    "artifact on a rung that has the format's own floor (measured: " +
                    std::string(requirement->kernel_evidence) +
                    ") is the other way.";
            } else {
                out.why +=
                    "What to do instead on THIS card: there is no requantization escape "
                    "either, because this rung covers neither " +
                    std::string(cap_name(requirement->required)) +
                    " nor Bf16Mma, which is the floor every groupwise-int format in this tree "
                    "bottoms out in. Run the artifact on a rung whose capability set covers "
                    "the floor (measured: " + std::string(requirement->kernel_evidence) + ").";
            }
            // The same one sentence every QPN refusal carries, appended so this arm does not
            // lose the file's rule that `why` names the MISSING KERNEL: on this tier the port
            // is not the RECOMMENDED action, but the operator asking "why is there no QPN
            // route" still gets the kernel's name rather than only the card's.
            out.why += " " + std::string(detail::qpn_missing_reason(qpn_in_build));
            return out;
        }
        default:
            break;
        }
        RouteChoice out;
        out.outcome = RouteOutcome::NoKernelInTree;
        out.why = std::string("sm_") + std::to_string(sm) + " (" + std::string(rung->label) +
                  ") has fp16 mma but not " + std::string(cap_name(requirement->required)) +
                  ", and this tree has no fp16 route for " +
                  std::string(artifact::format_name(format)) +
                  ": the shipped kernels for it bottom out in " +
                  std::string(requirement->kernel_evidence) +
                  ". Missing kernel: an mma_s8 port of the groupwise-int GEMMs, or an "
                  "explicit bf16->fp16 weight conversion ahead of an m8n8k4 GEMM -- the "
                  "m8n8k4 helper now exists (ops/common/mma.cuh, mma_f16_m8n8k4; it "
                  "assembles from sm_70 up, measured), so the missing piece for these "
                  "formats is the conversion and the kernel that drives it, not the mma "
                  "channel. NOTE these formats have NO entry in kFormatRequirements' "
                  "fallback_required column, which is why they still refuse here: the QPN "
                  "family does not consume Q4/Q5/Q6/W8 bytes.";
        return out;
    }

    RouteChoice out;
    out.outcome = RouteOutcome::NoKernelInTree;
    out.why = std::string("sm_") + std::to_string(sm) + " (" + std::string(rung->label) +
              ") has no tensor core at all in kArchLadder, and " +
              std::string(artifact::format_name(format)) + " needs " +
              std::string(cap_name(requirement->required)) +
              ". Missing capability, not a missing table row.";
    return out;
}

} // namespace detail

// THE NAME-ONLY ENTRY POINT, MEANING UNCHANGED. Byte-for-byte the same answer this overload
// gave before this section existed, so every test that pins it keeps pinning it. It is NOT
// deleted and NOT deprecated: it is the honest answer to "what does the TABLE say about a
// number", which is a question a table can answer and a measurement cannot.
inline RouteChoice select_route(int sm, artifact::NumericFormat format, ProblemShape shape,
                                bool qpn_in_build) noexcept {
    const ArchRung* rung = arch_rung(sm);
    if (rung == nullptr) { return conservative_fallback(sm, shape, qpn_in_build); }
    return detail::select_route_on_rung(rung, sm, format, shape, qpn_in_build);
}

// THE MEASURED ENTRY POINT -- the one the ENGINE uses. Same body, fed the row the box's own
// probes narrowed, and every answer carries the measurement's provenance in `why` whether it
// succeeded or refused. A never-probed `measured` is NOT an error: it takes the build-default
// fallback and says so (resolve_measured_row's rule), which is why the engine can call this
// unconditionally.
inline RouteChoice select_route(int sm, artifact::NumericFormat format, ProblemShape shape,
                                const MeasuredCaps& measured, bool qpn_in_build) noexcept {
    const MeasuredRow resolved = resolve_measured_row(sm, measured, "select_route");
    const ArchRung* rung       = resolved.effective();
    RouteChoice out = (rung == nullptr) ? conservative_fallback(sm, shape, qpn_in_build)
                                       : detail::select_route_on_rung(rung, sm, format, shape,
                                                                      qpn_in_build);
    // The provenance goes on EVERY answer, refusals included: a refusal whose reason does not
    // say whether the capability set was measured or named is the same defect one layer down.
    out.why += " " + resolved.why;
    return out;
}

inline RouteLogLine route_log_line(int sm, artifact::NumericFormat format,
                                   ProblemShape shape, const RouteChoice& choice) {
    RouteLogLine out;
    out.warn = choice.warns();
    // A simulated answer is NOT a warning about the route (the route is real for the
    // simulated rung) but it must be impossible to misread, so it gets its own marker at the
    // FRONT of the line rather than a note at the end. `warn` stays what it always was, so no
    // caller's severity mapping changes meaning.
    out.text = choice.simulated ? std::string("[route][SIMULATED] ") : std::string("[route] ");
    out.text += std::string("sm=") + std::to_string(sm) +
               " format=" + std::string(artifact::format_name(format)) +
               " m=" + std::to_string(shape.m) + " n=" + std::to_string(shape.n) +
               " k=" + std::to_string(shape.k) + " -> " +
               std::string(route_name(choice.route)) +
               " kernel='" + std::string(choice.kernel) + "'" +
               " outcome=" + std::string(outcome_name(choice.outcome)) + " | " + choice.why;
    return out;
}

inline std::string route_selection_line(int sm, artifact::NumericFormat format,
                                        ProblemShape shape, const RouteChoice& choice) {
    return route_log_line(sm, format, shape, choice).text;
}

// ---------------------------------------------------------------------------
// The simulation-aware entry points (src/core/arch_sim.h)
// ---------------------------------------------------------------------------
//
// These are the ONLY route entry points that consult the test-only override, and they are
// what the engine and the tests call when a device is in hand. The plain overloads above stay
// pure functions of a number, which is what keeps "what route does a V100 take" a table
// question answerable on a single-GPU host.
// THE PRODUCTION FORM, DEFINED. kernel_route.h DECLARED this overload above and never
// defined it, so any translation unit that called it -- tests/test_arch_generic_fallback.cpp
// is one -- failed at LINK with "undefined reference to ninfer::caps::select_route(
// ninfer::caps::ArchView const&, ninfer::caps::ProblemShape)". It is the form whose
// contract is "production callers use the default and cannot forget it", so the definition
// is the delegation and nothing else. (The same declared-without-definition shape is called
// out in src/core/arch_caps.h's note on require_artifact_formats_supported, which cost the
// fleet an hour on 2026-09-17; that one was deleted, this one is owed a body.)
inline RouteChoice select_route(const ArchView& view, artifact::NumericFormat format,
                               ProblemShape shape) noexcept {
    return select_route(view, format, shape, kQpnInBuild);
}

inline RouteChoice select_route(const ArchView& view, artifact::NumericFormat format,
                               ProblemShape shape, bool qpn_in_build) noexcept {
    // FAIL CLOSED. A simulation was asked for and could not be honoured: the caller's
    // question was "what happens on sm_70", and answering with the real card's route would
    // be an answer to a question nobody asked -- and the one answer most likely to be
    // mistaken for a V100 result.
    if (view.failed()) {
        RouteChoice out;
        out.outcome = RouteOutcome::NoKernelInTree;
        out.route   = KernelRoute::None;
        out.why     = sim_refusal_reason(view);
        return out;
    }
    RouteChoice out = select_route(view.effective_sm, format, shape, qpn_in_build);
    if (!view.simulated()) { return out; }
    // Says WHICH number answered, WHICH card it is not, and that nothing about the real
    // device is being claimed. prepend rather than append, so a reader who stops after one
    // line still knows.
    out.simulated = true;
    out.why = "SIMULATED sm_" + std::to_string(view.effective_sm) + " (physical device sm_" +
              std::to_string(view.physical_sm) +
              "): this route was selected for the SIMULATED capability, so it says what the "
              "engine WOULD do on such a device, and says NOTHING about sm_" +
              std::to_string(view.physical_sm) +
              ". The kernels that execute are this binary's own, not sm_" +
              std::to_string(view.effective_sm) +
              " binaries. " + out.why;
    return out;
}

inline RouteChoice conservative_fallback(const ArchView& view, ProblemShape shape) noexcept {
    if (view.failed()) {
        RouteChoice out;
        out.outcome = RouteOutcome::NoKernelInTree;
        out.route   = KernelRoute::None;
        out.why     = sim_refusal_reason(view);
        return out;
    }
    RouteChoice out = conservative_fallback(view.effective_sm, shape, kQpnInBuild);
    if (view.simulated()) {
        out.simulated = true;
        out.why = "SIMULATED sm_" + std::to_string(view.effective_sm) + " (physical device sm_" +
                  std::to_string(view.physical_sm) + "): " + out.why;
    }
    return out;
}

} // namespace ninfer::caps
