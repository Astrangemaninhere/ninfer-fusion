#pragma once

// ---------------------------------------------------------------------------
// format_probe.h -- THE MEASURED SUPPORT LIST, AND THE GATE THAT READS IT
// ---------------------------------------------------------------------------
//
// THE PROPOSAL THIS IMPLEMENTS, verbatim: "再说你这个选择器就不能干脆直接弄一串试跑从
// fp16 一直到 nvfp4, 然后哪个不通过就报错, 最后把支持列表送进内核选路吗" -- run one probe
// series from fp16 down to nvfp4, report an error for whichever fails, and feed the
// resulting support list into kernel routing.
//
// The proposal is right, and the reason is a defect this file exists to remove: today's
// selector is a HAND-MAINTAINED, NAME-KEYED gate. tools/archkit/_GPU_MATRIX.md's row table
// declares NVFP4 W4A4 in the 100a/120a/121a row and labels the rows BELOW it
// "routes-to-port, not supported" -- rungs nobody can test sitting in a table as if they
// were decisions. A MEASURED support list is strictly better than a declared one, because a
// declared row is a name and a measured row is an experiment.
//
// ===========================================================================
// 0. WHY THERE IS NO NAME IN THE DECISION, AND WHY THERE IS NO CAPABILITY NUMBER EITHER
// ===========================================================================
//
// The user's requirement, verbatim: "甚至依我之见必须完全放弃名字探查这种，因为有改 vbios 的
// 卡" -- name probing must be abandoned ENTIRELY, because there are cards with modified
// VBIOS. The reason defeats a capability-number key too, and it is worth stating plainly
// because it is the design constraint everything below is shaped by:
//
//   A flashed/modified VBIOS can make a card PRESENT a name -- and a declared attribute set
//   -- that does not match the silicon. So a card can report a name that IS in the table
//   while the hardware underneath is different, the lookup returns "supported", and the
//   kernel set is wrong. A declaration can be FALSE, and a modded card is exactly the case
//   where it is false in the direction that hurts: it looks known and is not.
//
// THEREFORE: no name and no sm number is ever the verdict. The verdict is a
// KNOWN-ANSWER EXECUTION TEST -- run the format, compare the arithmetic against a reference
// (section 1) -- and nothing declarative is trusted: not the marketing name, not the compute
// capability, not a feature flag. Declared attributes survive only as HINTS that may order or
// skip probes, and as an AUDIT TRAIL (section 3).
//
// AND THIS IS WHY THE NUMERICAL CRITERION IS LOAD-BEARING FOR A SECOND, INDEPENDENT REASON.
// It is not only that "it ran" would launder a silent corruption into a support list that
// routing then trusts -- this project has paid for that repeatedly and in one day: a
// weight-residency mechanism returning all-zero ids at rc=0 with no warning (four arms); a
// recall arm reporting pages=5/5 while the engine said hole=7 INEXACT on 3 of 5 rounds; a
// flag accepted, echoed and silently not applied (--kv-tier-formats cold=int8); four cold
// pages silently dropped; a converter embedding a digest the engine rejects at load; and a
// "support" row whose verdict was `servable` while the pipeline that printed it never ran an
// inference. It is ALSO that a known-answer comparison is the ONLY instrument that catches a
// card which LIES: a card that misreports its identity still EXECUTES whatever it executes,
// and running the arithmetic against a reference is what exposes the disagreement. A probe
// that asked "does this feature report as available?" would be fooled by the lie. A probe
// that runs the arithmetic cannot be.
//
// WHAT THIS BUYS, and it is a simplification rather than a loss: with names abandoned, the
// Engineering-Sample card, the VBIOS-modded card and the listed retail card all take ONE
// path. There is no "unlisted device" branch and no special case -- the probe series runs,
// the list is built from measurement, and an unprobed combination REFUSES with an ON-RAMP
// that names the command which would measure it.
//
// ===========================================================================
// 1. THE CRITERION
// ===========================================================================
//
// The criterion is NOT invented here. It is quoted, with its file:line, from the suite that
// already OWNS these kernels:
//
//   tests/ops/linear/linear_test_common.cpp:38-48  `tolerance_for(ActivationCompute)`
//   tests/ops/linear/linear_test_common.cpp:246    `cpu_linear_gemm_fp64`  (the reference)
//   tests/ops/linear/linear_test_common.cpp:31-33  the three constants
//
//     A16 path (BF16, Q4G64/Q5G64/Q6G64/W8G32, NVFP4-A16, FP8-A16):
//         relative_l2 <= 1/256,  max_abs <= 1/256 + 2*(1/256) * max|reference|
//     A8  path (FP8 A8, kind::f8f6f4):
//         relative_l2 <= 0.04,   max_abs <= 1/256 + 0.06 * max|reference|
//     A4  path (NVFP4 W4A4):
//         relative_l2 <= 0.16,   max_abs <= 1/256 + 0.16 * max|reference|
//
// WHY THOSE NUMBERS, AND WHY THAT IS JUSTIFIED RATHER THAN SLACK: the relative-L2 allowance
// is exactly ONE BF16 UNIT ROUNDOFF (1.0/256.0) for the A16 path -- that is the storage
// precision of the output, not a fudge factor. The gross allowance covers final bf16 storage
// plus accumulation/reduction rounding. The A8 and A4 paths are LOOSER BY CONSTRUCTION and
// the looseness is quantified, not hand-waved: 0.04 and 0.16 are the quantization error the
// format itself introduces when it quantizes the ACTIVATION to 8 and 4 bits. Those are
// input-precision budgets, and calling them "tolerance" without that reason would be the
// tolerating-for-convenience move this file must not make.
//
// CAN A PROBE BE EXACT? For the GEMM legs, NO, and the reason is structural rather than
// inconvenient: the reference is an FP64 CPU GEMM over the dequantized fp32 weight and the
// bf16 activation, while the kernel accumulates in fp32 with a different summation order and
// stores bf16; and on the A8/A4 legs the activation is quantized before the mma, so the
// output differs from the reference by the quantization error itself. Byte-identical ids
// (this project's acceptance norm for a SEMANTIC change) is the wrong instrument here: a
// weight format is an arithmetic question with a quantified precision budget, not an
// identity question.
//
// WHERE EXACTNESS *IS* AVAILABLE, THE PROBE REQUIRES IT: the format's own code words. e2m1
// and e4m3fn decode is exactly specified and the tree already pins it bitwise against an
// independent scalar oracle (tests/artifact/test_nvfp4_numeric.py: all 16 e2m1 words and all
// 256 e4m3fn words, exact). That is ProbeLeg::ExactCodeWords below, and it is what a CODEC
// change must be accepted on.
//
// WHO ACCEPTS THE TOLERANCE: the linear Op suite, which owns these kernels and whose
// criterion this is. The probe does not get to widen it. It records the measured RATIO
// (error / limit) per case, so a marginal entry is visible as marginal instead of being
// flattened into "supported".
//
// ===========================================================================
// 2. THE THREE PROPERTIES THAT MAKE THIS AN IMPROVEMENT RATHER THAN A NEW CLAIM
// ===========================================================================
//
// (a) FAIL-CLOSED FOR THE UNPROBED. The list is an ALLOW-LIST, never a default-allow. An
//     unprobed (measured-key, format, band) REFUSES, and the refusal is an ON-RAMP: it names
//     the command that would produce the row (section 4). "Unsupported" alone would satisfy
//     fail-closed and fail the user's requirement; "run X to probe this card" satisfies both.
//     THERE IS NO DEFAULT-ALLOW PATH IN THIS FILE, not even a reported one: a reported
//     default is still a default, and the counterexample to design against is an engine that
//     used a generic model for an unlisted capability and said so only in a field nobody
//     reads as "this is unmeasured".
//
// (b) PER-SHAPE, NOT PER-ARCH. The record's own kernel-gap story is "NVFP4 半边 M9-16 中段是
//     让步带 ... M>=17 才让 marlin" (tools/archkit/_GPU_MATRIX.md, "增补 2"): the SAME
//     format's coverage has holes by M. A per-arch entry would therefore be a lie, so the key
//     carries the M band, and the band edges are the ROUTER'S OWN edges
//     (src/core/kernel_route.h:298 and :416 -- M<=3 / 4..8 / 9..16 / >=17) rather than a
//     second set invented here.
//
// (c) DERIVED, NOT HAND-MAINTAINED, WITH PROVENANCE THAT SEPARATES DECLARED FROM MEASURED.
//     The list is produced by tools/archkit/probe_formats.sh from a probe RUN. Every entry
//     carries the measured capability KEY it was taken under, the DECLARED identity the card
//     presented at that moment (marked as a declaration), the criterion id and its
//     file:line, the reference, the date, and the binary's sha16. `freshness()` compares the
//     recorded binary sha16 against a caller-supplied one, so a stale entry is DETECTABLE and
//     refused by default -- the whole point is to stop a dead row from being believed.
//
// THE MECHANISM-CHECK LESSON THIS FILE INHERITS, twice measured on this tree:
// tools/archkit/build_arch.sh:48 declares NINFER_ARCH_NEEDING_LEGACY_TOOLKIT=70 and NO CODE
// PATH READS IT (the real selector is the TOOLCHAIN_FILE_OF table at :53, used at :83); and
// src/core/kernel_route.h carried a row claiming a kernel family LACKS the M 17..64 band
// while that kernel was compiled into ninfer_ops the whole time. A declared name is not a
// mechanism and a declared capability is not a capability. So: the gate below is a function
// that is CALLED, the list is a file that is PARSED, the staleness check COMPARES TWO HASHES,
// and the declared-vs-measured check COMPARES TWO KEYS. What is still missing -- the call
// site in the dispatch -- is named in require_artifact_format_probed()'s refusal text and in
// PATCHSET/PROBE2ROUTE/REPORT.md rather than pretended to exist.
//
// WHAT THIS HEADER DOES NOT DO. It does not write the list (the driver does), it does not run
// the probe (the driver does, under the project's GPU lock), and it is host-only: no CUDA
// header, no device query, so every branch is exercisable on a machine with no GPU. The
// feature axis is NOT re-implemented here -- it is the engine's EXISTING known-answer probe
// (src/core/device_capabilities.h's DeviceCapability vector, filled by device_probe.cu, which
// really launches the format's own mma and compares against a stated reference). This header
// consumes that report; it does not carry a second copy of its verdicts.

#include "artifact/reader.h"
#include "core/arch_caps.h"
#include "core/arch_sim.h"
#include "core/device_capabilities.h"
#include "core/kernel_route.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::caps {

// ---------------------------------------------------------------------------
// 1. THE CRITERION
// ---------------------------------------------------------------------------

// Which measurement produced a verdict. The distinction is load-bearing: a verdict from
// ExactCodeWords and a verdict from RouteSimulated are answers to DIFFERENT QUESTIONS, and a
// routing gate that accepted either as "this rung can compute this format" would be the
// laundering failure this file opens by naming.
enum class ProbeLeg : std::uint8_t {
    Unprobed = 0,   // no measurement. The ONLY honest value for a rung nobody probed.
    ReductionL2,    // GEMM output vs the FP64 CPU reference, under the quoted criterion.
    ExactCodeWords, // the format's own code words, bitwise vs an independent oracle.
    RouteSimulated, // ONLY the route decision was exercised (arch-sim). No arithmetic.
};

[[nodiscard]] inline std::string_view probe_leg_name(ProbeLeg leg) noexcept {
    switch (leg) {
    case ProbeLeg::Unprobed: return "unprobed";
    case ProbeLeg::ReductionL2: return "reduction_l2";
    case ProbeLeg::ExactCodeWords: return "exact_code_words";
    case ProbeLeg::RouteSimulated: return "route_simulated";
    }
    return "unknown-leg";
}

// The numbers, each carrying where it was quoted from. `source` is not decoration: it is what
// lets a reader check that this header and the owning suite still agree, and
// tests/test_format_probe.cpp asserts that agreement by reading that file as text.
struct ProbeCriterion {
    ProbeLeg leg = ProbeLeg::Unprobed;
    std::string_view id = {};
    std::string_view source = {};
    std::string_view reference = {};
    double relative_l2 = 0.0;
    double gross_absolute = 0.0;
    double gross_relative_to_max_reference = 0.0;
    std::string_view rationale = {};
};

// 1.0/256.0 is one bf16 unit roundoff -- the OUTPUT storage precision, which is why the A16
// allowance is exactly that and not something rounder.
inline constexpr double kProbeBf16UnitRoundoff = 1.0 / 256.0;
inline constexpr double kProbeA8QuantizationAllowance = 0.04;
inline constexpr double kProbeA4QuantizationAllowance = 0.16;

inline constexpr ProbeCriterion kProbeCriterionA16{
    ProbeLeg::ReductionL2,
    "linear-reduction-v1/A16",
    "tests/ops/linear/linear_test_common.cpp:40-41",
    "cpu_linear_gemm_fp64 (tests/ops/linear/linear_test_common.cpp:246)",
    kProbeBf16UnitRoundoff,
    kProbeBf16UnitRoundoff,
    2.0 * kProbeBf16UnitRoundoff,
    "relative-L2 allowance is one bf16 unit roundoff (the output's own storage precision); "
    "the gross allowance covers final bf16 storage plus accumulation and reduction rounding."};

inline constexpr ProbeCriterion kProbeCriterionA8{
    ProbeLeg::ReductionL2,
    "linear-reduction-v1/A8",
    "tests/ops/linear/linear_test_common.cpp:42-43",
    "cpu_linear_gemm_fp64 (tests/ops/linear/linear_test_common.cpp:246)",
    kProbeA8QuantizationAllowance,
    kProbeBf16UnitRoundoff,
    1.5 * kProbeA8QuantizationAllowance,
    "0.04 is the A8 path's own activation-quantization error budget, not slack: the A8 route "
    "quantizes the activation to 8 bits before the mma, so it cannot match the FP64 "
    "reference more closely."};

inline constexpr ProbeCriterion kProbeCriterionA4{
    ProbeLeg::ReductionL2,
    "linear-reduction-v1/A4",
    "tests/ops/linear/linear_test_common.cpp:44-45",
    "cpu_linear_gemm_fp64 (tests/ops/linear/linear_test_common.cpp:246)",
    kProbeA4QuantizationAllowance,
    kProbeBf16UnitRoundoff,
    kProbeA4QuantizationAllowance,
    "0.16 is the A4 path's own activation-quantization budget for the same reason as A8, and "
    "it is the largest allowance in this file. An A4 entry measured anywhere NEAR 0.16 is a "
    "finding, not a pass, which is why the ratio is recorded per case."};

// "A16" | "A8" | "A4" -> the criterion, or nullptr for anything else. A caller that cannot say
// which activation path a verdict belongs to has not identified the measurement, so it gets
// no criterion rather than a default one.
[[nodiscard]] inline const ProbeCriterion* probe_criterion_for(std::string_view path) noexcept {
    if (path == "A16") { return &kProbeCriterionA16; }
    if (path == "A8") { return &kProbeCriterionA8; }
    if (path == "A4") { return &kProbeCriterionA4; }
    return nullptr;
}

// ---------------------------------------------------------------------------
// 2. THE SHAPE BANDS -- the router's own edges, not a second set
// ---------------------------------------------------------------------------

// Weights never carry a token extent, but KERNELS do: every band below is a discontinuity in
// which kernel can be launched, and each one is a fact recorded elsewhere in the tree rather
// than a bin chosen for tidiness.
inline constexpr std::uint32_t kProbeBandOpenEnd = 0xFFFFFFFFu;

struct ProbeBand {
    std::uint32_t m_lo = 0;
    std::uint32_t m_hi = 0; // inclusive
    std::string_view basis = {};
};

inline constexpr ProbeBand kProbeBands[] = {
    {1, 3, "kernel_route.h:298 simt band (skinny_nvfp4_qpn_simt<M>, M 1..3)"},
    {4, 8, "kernel_route.h:298 qpn2 band (skinny_nvfp4_qpn<1>, M 4..8)"},
    {9, 16,
     "kernel_route.h:298 MT=2 band (skinny_nvfp4_qpn<2>, M 9..16) -- and "
     "tools/archkit/_GPU_MATRIX.md '增补 2' names exactly this band the NVFP4 让步带 "
     "(concession band), the one the port must fill with MT=2"},
    {17, kProbeBandOpenEnd,
     "kernel_route.h:416 -- M >= 17 has NO route in the QPN family at all; prefill defers to "
     "a wide-M entry that src/ops/linear/qpn/ does not contain"},
};
inline constexpr std::size_t kProbeBandCount = sizeof(kProbeBands) / sizeof(kProbeBands[0]);

[[nodiscard]] inline std::string probe_band_name(std::uint32_t m_lo, std::uint32_t m_hi) {
    return std::to_string(m_lo) + ".." +
           (m_hi == kProbeBandOpenEnd ? std::string("open") : std::to_string(m_hi));
}

// The band an M value falls in, or nullptr for m == 0 (no kernel takes an empty problem, so
// there is no band to answer for and a caller must refuse rather than pick the first one).
[[nodiscard]] inline const ProbeBand* probe_band_of(std::uint32_t m) noexcept {
    if (m == 0) { return nullptr; }
    for (const ProbeBand& band : kProbeBands) {
        if (m >= band.m_lo && m <= band.m_hi) { return &band; }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// ===========================================================================
// 2b. THE SIMULATOR IS A ROUTE INSTRUMENT, NEVER A SUPPORT INSTRUMENT
// ===========================================================================
//
// WHY THIS SECTION EXISTS. The test-only architecture simulator (src/core/arch_sim.h) changes
// WHICH TABLES ANSWER -- the rung the capability and route layers see -- and nothing else. The
// cubins that execute are the ones the running BINARY was compiled for. Therefore a simulated run
// answers "does this rung take this path" and can never answer "does this format compute the right
// numbers on that rung". This file keys on the arithmetic proposition, so a verdict that came from
// the simulator is REFUSED here as support -- and it is refused by NAME, with its rung, and with
// the literal command that would measure the real thing (sim_verdict_refusal(), section 4b),
// rather than being dropped on the floor or read as an ordinary negative.
//
// WHAT THE SIMULATOR IS FOR, because the refusal must not read as "this path is worthless":
// exercising a code path this box owns no hardware for. The CUDA 12.8 chain at /mnt/g/cuda12/tk
// reaches compute_50/52/53 (Maxwell, the GTX 960) and compute_60/61/62 (Pascal), and sm_70 (V100)
// and sm_75 (2080Ti) are the two rungs with a named wall, so a simulated run is the ONLY way this
// tree can see the fallback path TAKEN on those rungs at all. The correct output of such a run is
//
//     "this path was exercised on a simulated rung; its support is UNMEASURED and its
//      arithmetic is UNVERIFIED"
//
// and that is exactly what the refusal says. Deleting the simulator would delete the instrument;
// this file marks its verdicts instead.
//
// THE TWO PROHIBITIONS ARE DIFFERENT PROHIBITIONS AND BOTH STAND. src/core/arch_sim.h:59-70
// forbids sourcing any PERFORMANCE number (throughput, latency) from a simulated run. This file
// forbids sourcing a FUNCTIONAL support claim from one. A support claim is functional -- "this
// rung computes this format within the criterion" -- and a speed claim is not, so these are not
// one rule written twice and neither may be traded for the other.
//
// THE THREE PLACES A SIMULATED VERDICT COULD ENTER THIS MECHANISM, AND WHAT CLOSES EACH:
//   (1) `entry ... supported=yes leg=reduction_l2 simulated=yes` -- a row that admits its route
//       was simulated and still claims support. REFUSED AT PARSE TIME (S2 below), so it can never
//       be looked up, exactly like `supported=yes leg=unprobed`.
//   (2) `meta source="simulated_route"` with a row that is NOT marked simulated, or
//       `meta source="real_gpu"` with a row that IS. A self-contradictory provenance. REFUSED AT
//       PARSE TIME (the provenance/verdict consistency rule at the end of parse()).
//   (3) `entry ... leg=route_simulated supported=yes` (the hand-written shape). REFUSED AT GATE
//       TIME by the named refusal below, and normalised at parse time (S1) so that one predicate,
//       FormatSupportEntry::is_simulated(), covers both spellings of "this came from the
//       simulator" -- a predicate that catches one spelling and misses the other is precisely how
//       a simulated verdict gets admitted as measured.
//
// AND ONE TRAP THIS SECTION IS CAREFUL ABOUT, because it is the shape of a measured incident: the
// simulator's `none` verdict and "the run never reached a route decision" are DIFFERENT FACTS THAT
// LOOK ALIKE IN A SUMMARY. Measured 2026-09-18 on the RTX 5090 D: an sm_89 request produced the
// artifact-gate refusal ("cannot run this artifact on this GPU ... needs nvfp4 e2m1 block-scaled
// mma") with `grep -c '[route]'` = 0, because the gate fires before any op runs -- while an sm_89
// request that DID reach the route layer prints `[route][SIMULATED] ... -> none`, because the emit
// condition is `simulated || refused` and NOT `use_qpn` (src/ops/linear/qpn/qpn_arch_route.cpp).
// So a reader can tell them apart only with THREE observations, and this is the full table:
//
//     NINFER_SIM_ARCH set | sim banner on stderr | `[route]` line | what happened
//     --------------------+----------------------+----------------+---------------------------
//     no                  | no                   | no             | no simulation was requested
//     yes                 | NO                   | no             | the process never reached the
//                         |                      |                | arch view at all, so the
//                         |                      |                | simulator was never consulted
//     yes                 | yes                  | no             | the simulator WAS consulted and
//                         |                      |                | no route decision was reached
//                         |                      |                | (artifact/format gate fired first)
//     yes                 | yes                  | yes            | a route decision exists and its
//                         |                      |                | value is the simulator's answer
//
// The banner is printed by arch_view_for_device() on the FIRST call in the process (src/core/
// arch_sim.h), and the engine's first call is the registry capability gate -- so the second row of
// that table ("env set, no banner") means the run died before it asked the arch view anything. A
// summary that reports `[route] route=none` without saying whether the SIMULATED prefix and the
// banner were present is not reporting a fact, and this file's refusals never produce one.
//
// 3. THE KEY: MEASURED CAPABILITIES, WITH THE DECLARATION KEPT BESIDE IT AS AN AUDIT TRAIL
// ---------------------------------------------------------------------------

// WHAT THE CARD *SAYS*. Recorded verbatim, printed in reports, and used ONLY as a hint --
// to select which entry to COMPARE a new measurement against, never to admit anything. A
// modified VBIOS makes every field of this struct potentially false, which is exactly why it
// is not the key.
struct DeclaredIdentity {
    std::string name = {}; // verbatim device name. NEVER a gate.
    int cc_major = 0;      // as reported by the driver. A HINT, not the verdict: two builds of
    int cc_minor = 0;      // the same cc differ in features (sm_120 refuses fp8/nvfp4, sm_120a
                           // does not), so the number cannot decide the feature set.
    std::string build_arch = {}; // CMAKE_CUDA_ARCHITECTURES of the binary: the '-a' fact that
                                 // cudaDeviceProp cannot see. Still a hint: it describes the
                                 // BINARY, not the silicon.

    [[nodiscard]] std::string render() const {
        return "name='" + name + "';cc=" + std::to_string(cc_major) + "." +
               std::to_string(cc_minor) + ";build=" + (build_arch.empty() ? "-" : build_arch);
    }
};

// WHAT THE KNOWN-ANSWER EXECUTION TESTS SAW. THIS is the key, and every field is the outcome
// of running the format's own kernel and comparing against a reference -- not a flag read
// from anywhere. The names are src/core/device_capabilities.h's DeviceCapability enumerators,
// whose CapabilityProbeSpec::probe text states the input, the reference and the criterion for
// each one; that header is where the probes live and this one does not restate them.
struct MeasuredCapabilities {
    // The vector's VERSION. A change to which capabilities participate is a change of key, so
    // every row measured under the old vector stops matching instead of silently continuing
    // to answer a question whose meaning has moved.
    static constexpr int kVectorVersion = 1;

    bool kernel_image = false;         // a kernel launches and its output copies back
    bool bf16_mma = false;             // mma.sync...m16n8k16...bf16, |err| <= 1e-6
    bool fp16_mma = false;             // mma.sync...m16n8k16...f16,   |err| <= 1e-6
    bool int8_mma = false;             // mma.sync...m16n8k32...s8,    integer-exact
    bool fp8_kind_f8f6f4 = false;      // mma...kind::f8f6f4...e4m3
    bool nvfp4_mma_block_scale = false; // mma...kind::mxf4nvf4.block_scale...e2m1
    bool setmaxnreg = false;           // the TMA/ws kernel's register hand-off
};

// The canonical key string. Stable, greppable, diffable, and ORDER-FIXED so two vectors that
// agree on the facts produce the same string: "<v1>;ki=1;bf16=1;fp16=1;i8=1;f8f6f4=1;mxf4=1;smr=1".
[[nodiscard]] inline std::string measured_capability_key(const MeasuredCapabilities& m) {
    const auto bit = [](bool value) { return value ? "1" : "0"; };
    std::string out = "v";
    out += std::to_string(MeasuredCapabilities::kVectorVersion);
    out += ";ki=";
    out += bit(m.kernel_image);
    out += ";bf16=";
    out += bit(m.bf16_mma);
    out += ";fp16=";
    out += bit(m.fp16_mma);
    out += ";i8=";
    out += bit(m.int8_mma);
    out += ";f8f6f4=";
    out += bit(m.fp8_kind_f8f6f4);
    out += ";mxf4=";
    out += bit(m.nvfp4_mma_block_scale);
    out += ";smr=";
    out += bit(m.setmaxnreg);
    return out;
}

// The adapter from the engine's EXISTING known-answer probe report. No second probe and no
// second table: src/core/device_probe.cu really launches mma_nvfp4_e4m3 / mma_bf16 / the
// setmaxnreg kernel and records ProbeStatus, and ProbeStatus::Supported is defined as "the
// probe ran and the values matched the reference" (device_capabilities.h). So a true bit here
// means a known-answer test passed ON THIS SILICON -- which is the only thing this file is
// willing to key on.
// NOTE the fully-qualified type: there are TWO CapabilityReport types in this tree --
// ninfer::CapabilityReport (src/core/device_capabilities.h, the known-answer probe report,
// which has supported()) and ninfer::caps::CapabilityReport (src/core/arch_caps.h, the
// format-gate report, which does not). This header is in ninfer::caps, so an unqualified name
// binds to the WRONG one and fails to compile. Measured, 2026-09-18.
[[nodiscard]] inline MeasuredCapabilities
measured_capabilities_from(const ninfer::CapabilityReport& report) {
    MeasuredCapabilities out;
    out.kernel_image = report.supported(ninfer::DeviceCapability::KernelImage);
    out.bf16_mma = report.supported(ninfer::DeviceCapability::Bf16Mma);
    out.fp16_mma = report.supported(ninfer::DeviceCapability::Fp16Mma);
    out.int8_mma = report.supported(ninfer::DeviceCapability::Int8Mma);
    out.fp8_kind_f8f6f4 = report.supported(ninfer::DeviceCapability::Fp8MmaKindF8f6f4);
    out.nvfp4_mma_block_scale = report.supported(ninfer::DeviceCapability::Nvfp4MmaBlockScale);
    out.setmaxnreg = report.supported(ninfer::DeviceCapability::SetMaxNreg);
    return out;
}

// ---------------------------------------------------------------------------
// 4. ONE ENTRY, AND THE PROVENANCE THE WHOLE FILE EXISTS FOR
// ---------------------------------------------------------------------------

struct FormatSupportEntry {
    // THE KEY. The measured capability vector this row was taken under. Empty means the row
    // is not usable at all -- a row measured under no capability vector has identified
    // nothing about the silicon.
    std::string key = {};
    // The declaration the card presented WHEN THE ROW WAS MEASURED. An audit trail: it lets a
    // mismatch be reported (section 5), and it never admits anything.
    std::string declared_name = {};
    std::string declared_cc = {};
    artifact::NumericFormat format = artifact::NumericFormat::I32;
    std::string path = {}; // A16 | A8 | A4 -- which criterion the verdict was taken under
    std::uint32_t m_lo = 0;
    std::uint32_t m_hi = 0;
    ProbeLeg leg = ProbeLeg::Unprobed;
    bool supported = false; // DERIVED from the criterion; never from "it ran"
    int cases = 0;          // how many measured cases back this band
    int geometries = 0;     // distinct (n, k) geometries among those cases
    int arm_rc = 0;         // the arm's own exit code; != 0 means the arm was not green
    bool any_ratio = false;
    double worst_ratio = 0.0; // max(error / limit) over the band's cases
    std::string case_example = {};
    // The FIRST case in the band that did not pass the criterion. Non-empty implies
    // supported == false: a band is only admitted when every one of its measured cases
    // passed, so a failing case is never partial credit.
    std::string first_failure = {};
    std::string probe = {}; // the arm that produced it
    bool simulated = false;
    // THE RUNG THE SIMULATION WAS ASKED FOR, i.e. which capability's routing this row's arm
    // actually exercised. Recorded because a refusal must be able to say WHICH rung was
    // simulated: "a simulated verdict exists" without the rung leaves the reader unable to
    // re-probe, and an on-ramp that does not say what to measure on the real hardware is not an
    // on-ramp. Empty on a measured row, and empty on a simulated row whose writer did not record
    // it -- which the refusal names as a defect of that row rather than papering over.
    std::string sim_rung = {};

    [[nodiscard]] bool answers_arithmetic() const noexcept {
        return leg == ProbeLeg::ReductionL2 || leg == ProbeLeg::ExactCodeWords;
    }
    // TRUE FOR BOTH SPELLINGS OF "THIS CAME FROM THE SIMULATOR", and it is ONE predicate on
    // purpose: parse() normalises leg=route_simulated into simulated=true (S1), and every gate
    // branch asks this one question. Two spellings with two checks is how one of them gets missed.
    [[nodiscard]] bool is_simulated() const noexcept {
        return simulated || leg == ProbeLeg::RouteSimulated;
    }
    [[nodiscard]] std::string band() const { return probe_band_name(m_lo, m_hi); }
};

struct FormatSupportProvenance {
    std::string probe_tool = {};
    std::string criterion_id = {};
    std::string criterion_src = {};
    std::string reference = {};
    std::string stats_env = {};
    std::string key = {};           // the measured capability vector of the run
    std::string declared_name = {}; // what the probed card SAID. A declaration.
    std::string declared_cc = {};
    std::string declared_build = {};
    std::string date = {};
    std::string binary_path = {};
    std::string binary_sha16 = {};
    std::string binary_mtime = {};
    std::string library_path = {};
    std::string library_sha16 = {};
    std::string library_mtime = {};
    std::string source = {}; // real_gpu | simulated_route
    std::string arms_not_green = {};
    bool complete = false; // every field above that a verdict depends on is present
    std::string missing = {};
};

// The staleness answer, as a value and not a mood. `fresh` is the only value a gate accepts
// without an explicit opt-in.
enum class ProbeFreshness : std::uint8_t {
    Unknown = 0, // the list does not record a binary sha16: cannot be judged
    Fresh,       // recorded binary sha16 == the caller's
    StaleBinary, // differs: the verdict is about a binary that is not this one
};

[[nodiscard]] inline std::string_view probe_freshness_name(ProbeFreshness f) noexcept {
    switch (f) {
    case ProbeFreshness::Unknown: return "unknown";
    case ProbeFreshness::Fresh: return "fresh";
    case ProbeFreshness::StaleBinary: return "stale_binary";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// 5. THE LIST: PARSED FROM TEXT, RENDERED TO TEXT
// ---------------------------------------------------------------------------

// The wire format, line-oriented and diffable on purpose:
//   * blank lines and lines whose first non-whitespace character is '#' are ignored
//   * `meta <key>=<value> ...`  -> provenance
//   * `entry <key>=<value> ...` -> one FormatSupportEntry
//   * a value is either bare (up to the next space) or "double quoted"
// There is no third record type and no positional field, so a new field is additive.
//
// `arch=` IS NOT A FIELD. A list written by the name/cc-keyed generation of this mechanism is
// REFUSED at parse time with an explanation, rather than read under a key whose meaning has
// changed -- see FormatSupportList::parse.
class FormatSupportList {
public:
    FormatSupportList() = default;

    // Never throws on malformed input: a list that cannot be understood must be
    // distinguishable from an absent one, and both must REFUSE rather than default-allow.
    [[nodiscard]] static FormatSupportList parse(std::string_view text);
    [[nodiscard]] static FormatSupportList load_file(const std::string& path);

    [[nodiscard]] const FormatSupportProvenance& provenance() const noexcept {
        return provenance_;
    }
    [[nodiscard]] const std::vector<FormatSupportEntry>& entries() const noexcept {
        return entries_;
    }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] bool parsed() const noexcept { return parsed_; }
    [[nodiscard]] const std::string& parse_error() const noexcept { return parse_error_; }

    // TRUE only when nothing went wrong AND the header is complete AND at least one entry
    // parsed. A list with no entries would be a list that refuses everything, which is a legal
    // state and a different one from "this file is not a list".
    [[nodiscard]] bool valid() const noexcept {
        return parsed_ && parse_error_.empty() && provenance_.complete && !entries_.empty();
    }

    // ALL entries for (measured capability key, format, band). WHY A LIST AND NOT ONE ENTRY:
    // one weight format can be served by more than one activation path, and each path is a
    // separate measurement with its own criterion (NVFP4 has an A16 route and a W4A4 route;
    // FP8 has an A16 route and an A8 route). Which one a run takes is a DISPATCH question
    // this header cannot answer, so the gate must consider all of them and refuse unless every
    // one of them is admitted.
    [[nodiscard]] std::vector<const FormatSupportEntry*> find_all(std::string_view key,
                                                                 artifact::NumericFormat format,
                                                                 std::uint32_t m) const;

    // The first entry for the key, for a caller that has filtered on `path` itself. NOT
    // sufficient for the gate: with two paths measured, "the first one" is an arbitrary
    // choice between two answers to one question.
    [[nodiscard]] const FormatSupportEntry* find(std::string_view key,
                                                 artifact::NumericFormat format, std::uint32_t m,
                                                 std::string_view path) const;

    // The HINT index: entries whose recorded DECLARATION has this name. Used only to report a
    // declaration that disagrees with a measurement, never to admit anything.
    [[nodiscard]] std::vector<const FormatSupportEntry*> find_by_declared_name(
        std::string_view name) const;

    [[nodiscard]] ProbeFreshness freshness(std::string_view current_binary_sha16) const;

    [[nodiscard]] std::string render() const;

private:
    bool parsed_ = false;
    std::string parse_error_ = {};
    FormatSupportProvenance provenance_{};
    std::vector<FormatSupportEntry> entries_{};
};

namespace detail {

[[nodiscard]] inline std::string trim(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    const auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while (begin < end && blank(text[begin])) { ++begin; }
    while (end > begin && blank(text[end - 1])) { --end; }
    return std::string(text.substr(begin, end - begin));
}

// "key=value ..." with quoted values. Returns false only on a token that has no '=' at all,
// which is a malformed record rather than an unknown field.
[[nodiscard]] inline bool parse_pairs(std::string_view text,
                                      std::vector<std::pair<std::string, std::string>>& out) {
    std::size_t at = 0;
    while (at < text.size()) {
        while (at < text.size() && (text[at] == ' ' || text[at] == '\t')) { ++at; }
        if (at >= text.size()) { break; }
        const std::size_t key_begin = at;
        while (at < text.size() && text[at] != '=' && text[at] != ' ' && text[at] != '\t') { ++at; }
        if (at >= text.size() || text[at] != '=') { return false; }
        const std::string key(text.substr(key_begin, at - key_begin));
        ++at; // '='
        std::string value;
        if (at < text.size() && text[at] == '"') {
            ++at;
            while (at < text.size() && text[at] != '"') {
                value.push_back(text[at]);
                ++at;
            }
            if (at >= text.size()) { return false; } // unterminated quote
            ++at;
            while (at < text.size() && text[at] != ' ' && text[at] != '\t') { ++at; }
        } else {
            const std::size_t value_begin = at;
            while (at < text.size() && text[at] != ' ' && text[at] != '\t') { ++at; }
            value.assign(text.substr(value_begin, at - value_begin));
        }
        if (key.empty()) { return false; }
        out.emplace_back(key, value);
    }
    return true;
}

[[nodiscard]] inline std::optional<artifact::NumericFormat>
probe_format_from_name(std::string_view name) {
    // The names are artifact::format_name()'s own spellings, so a list written by one build is
    // readable by another and by a human. NOTE that this is a FORMAT name (a property of the
    // artifact, which the file itself declares) and NOT a DEVICE name (a declaration that a
    // modded card can falsify) -- the one place a string key is still legitimate.
    for (std::uint32_t raw = 0;
         raw <= static_cast<std::uint32_t>(artifact::NumericFormat::U4Z8G16_F16S); ++raw) {
        const auto candidate = static_cast<artifact::NumericFormat>(raw);
        if (artifact::format_name(candidate) == name) { return candidate; }
    }
    return std::nullopt;
}

[[nodiscard]] inline bool probe_parse_bool(std::string_view text, bool& out) {
    if (text == "yes" || text == "true" || text == "1") {
        out = true;
        return true;
    }
    if (text == "no" || text == "false" || text == "0") {
        out = false;
        return true;
    }
    return false;
}

// "1..3" | "17..open" -> a band. Anything else is a parse failure, not a default band.
[[nodiscard]] inline bool probe_parse_band(std::string_view text, std::uint32_t& lo,
                                           std::uint32_t& hi) {
    const std::size_t sep = text.find("..");
    if (sep == std::string_view::npos) { return false; }
    const std::string first(text.substr(0, sep));
    const std::string second(text.substr(sep + 2));
    if (first.empty() || second.empty()) { return false; }
    for (const char c : first) {
        if (c < '0' || c > '9') { return false; }
    }
    lo = static_cast<std::uint32_t>(std::strtoul(first.c_str(), nullptr, 10));
    if (second == "open") {
        hi = kProbeBandOpenEnd;
        return true;
    }
    for (const char c : second) {
        if (c < '0' || c > '9') { return false; }
    }
    hi = static_cast<std::uint32_t>(std::strtoul(second.c_str(), nullptr, 10));
    return hi >= lo && lo > 0;
}

} // namespace detail

inline FormatSupportList FormatSupportList::parse(std::string_view text) {
    FormatSupportList list;
    list.parsed_ = true;

    std::size_t line_begin = 0;
    while (line_begin <= text.size()) {
        std::size_t line_end = text.find('\n', line_begin);
        if (line_end == std::string_view::npos) { line_end = text.size(); }
        const std::string line = detail::trim(text.substr(line_begin, line_end - line_begin));
        line_begin = line_end + 1;
        if (line.empty() || line[0] == '#') { continue; }

        const std::size_t space = line.find(' ');
        const std::string record = line.substr(0, space);
        const std::string_view body = space == std::string::npos
                                          ? std::string_view{}
                                          : std::string_view(line).substr(space + 1);

        std::vector<std::pair<std::string, std::string>> pairs;
        if (!detail::parse_pairs(body, pairs)) {
            list.parse_error_ = "malformed record: '" + line.substr(0, 80) + "'";
            return list;
        }

        if (record == "meta") {
            for (const auto& [key, value] : pairs) {
                if (key == "arch" || key == "arch_suffix") {
                    // THE GENERATION BOUNDARY, refused rather than reinterpreted. The previous
                    // generation of this file keyed rows on a compute-capability number and a
                    // name; reading such a row under the measured-capability key would be
                    // reading a declaration as a measurement, which is the one thing this file
                    // exists to stop. A list from that generation must be RE-PRODUCED by
                    // tools/archkit/probe_formats.sh, not migrated.
                    list.parse_error_ =
                        "this list carries '" + key +
                        "', so it was written by the NAME/cc-keyed generation of this "
                        "mechanism. That key is no longer read: a declaration (a name, a "
                        "compute-capability number) may be false -- a modified VBIOS can make "
                        "a card present a listed name over different silicon -- so the key is "
                        "now the MEASURED capability vector. Re-produce the list with "
                        "tools/archkit/probe_formats.sh; do not migrate it.";
                    return list;
                }
                if (key == "key") {
                    list.provenance_.key = value;
                } else if (key == "probe_tool") {
                    list.provenance_.probe_tool = value;
                } else if (key == "criterion_id") {
                    list.provenance_.criterion_id = value;
                } else if (key == "criterion_src") {
                    list.provenance_.criterion_src = value;
                } else if (key == "reference") {
                    list.provenance_.reference = value;
                } else if (key == "stats_env") {
                    list.provenance_.stats_env = value;
                } else if (key == "declared_name") {
                    list.provenance_.declared_name = value;
                } else if (key == "declared_cc") {
                    list.provenance_.declared_cc = value;
                } else if (key == "declared_build") {
                    list.provenance_.declared_build = value;
                } else if (key == "date") {
                    list.provenance_.date = value;
                } else if (key == "binary_path") {
                    list.provenance_.binary_path = value;
                } else if (key == "binary_sha16") {
                    list.provenance_.binary_sha16 = value;
                } else if (key == "binary_mtime") {
                    list.provenance_.binary_mtime = value;
                } else if (key == "library_path") {
                    list.provenance_.library_path = value;
                } else if (key == "library_sha16") {
                    list.provenance_.library_sha16 = value;
                } else if (key == "library_mtime") {
                    list.provenance_.library_mtime = value;
                } else if (key == "source") {
                    list.provenance_.source = value;
                } else if (key == "arms_not_green") {
                    list.provenance_.arms_not_green = value;
                }
                // An unknown meta field is IGNORED rather than fatal, so a newer writer can add
                // provenance without breaking an older reader.
            }
            continue;
        }

        if (record != "entry") {
            list.parse_error_ = "unknown record type '" + record + "'";
            return list;
        }

        FormatSupportEntry entry;
        bool have_key = false;
        bool have_format = false;
        bool have_band = false;
        bool have_leg = false;
        bool have_supported = false;
        for (const auto& [key, value] : pairs) {
            if (key == "arch") {
                list.parse_error_ =
                    "entry carries arch=: a row keyed on a compute-capability number is a "
                    "row keyed on a declaration. Re-produce the list with "
                    "tools/archkit/probe_formats.sh so its rows carry the MEASURED capability "
                    "key.";
                return list;
            } else if (key == "key") {
                entry.key = value;
                have_key = true;
            } else if (key == "declared_name") {
                entry.declared_name = value;
            } else if (key == "declared_cc") {
                entry.declared_cc = value;
            } else if (key == "format") {
                const std::optional<artifact::NumericFormat> format =
                    detail::probe_format_from_name(value);
                if (!format.has_value()) {
                    list.parse_error_ = "unknown format name '" + value + "'";
                    return list;
                }
                entry.format = *format;
                have_format = true;
            } else if (key == "path") {
                entry.path = value;
            } else if (key == "band") {
                if (!detail::probe_parse_band(value, entry.m_lo, entry.m_hi)) {
                    list.parse_error_ = "unparsable band '" + value + "'";
                    return list;
                }
                have_band = true;
            } else if (key == "supported") {
                if (!detail::probe_parse_bool(value, entry.supported)) {
                    list.parse_error_ = "unparsable supported '" + value + "'";
                    return list;
                }
                have_supported = true;
            } else if (key == "leg") {
                if (value == "unprobed") {
                    entry.leg = ProbeLeg::Unprobed;
                } else if (value == "reduction_l2") {
                    entry.leg = ProbeLeg::ReductionL2;
                } else if (value == "exact_code_words") {
                    entry.leg = ProbeLeg::ExactCodeWords;
                } else if (value == "route_simulated") {
                    entry.leg = ProbeLeg::RouteSimulated;
                } else {
                    list.parse_error_ = "unknown leg '" + value + "'";
                    return list;
                }
                have_leg = true;
            } else if (key == "cases") {
                entry.cases = std::atoi(value.c_str());
            } else if (key == "geometries") {
                entry.geometries = std::atoi(value.c_str());
            } else if (key == "arm_rc") {
                entry.arm_rc = std::atoi(value.c_str());
            } else if (key == "fail") {
                entry.first_failure = value;
            } else if (key == "ratio") {
                entry.worst_ratio = std::strtod(value.c_str(), nullptr);
                entry.any_ratio = true;
            } else if (key == "case") {
                entry.case_example = value;
            } else if (key == "probe") {
                entry.probe = value;
            } else if (key == "simulated") {
                if (!detail::probe_parse_bool(value, entry.simulated)) {
                    list.parse_error_ = "unparsable simulated '" + value + "'";
                    return list;
                }
            } else if (key == "sim_rung") {
                // THE RUNG IS VALIDATED AGAINST THE LADDER, for the same reason arch_sim.h
                // refuses a rung kWArchLadder does not contain: an unlisted number would make the
                // simulator (and here, the list) the source of a capability claim, and a refusal
                // that names a rung nobody can build for is not an on-ramp to anything.
                bool digits = !value.empty();
                for (const char c : value) {
                    if (c < '0' || c > '9') {
                        digits = false;
                        break;
                    }
                }
                if (!digits || arch_rung(std::atoi(value.c_str())) == nullptr) {
                    list.parse_error_ =
                        "unparsable sim_rung '" + value +
                        "': it must be a decimal compute capability with a row in kArchLadder "
                        "(e.g. 70 for a V100, 75 for a 2080Ti, 86 for an RTX 30), because a "
                        "simulated verdict must be re-probeable on real hardware and a rung this "
                        "table does not contain cannot be.";
                    return list;
                }
                entry.sim_rung = value;
            }
        }
        if (!have_key || !have_format || !have_band || !have_leg || !have_supported) {
            list.parse_error_ = "entry is missing key/format/band/leg/supported";
            return list;
        }
        // A verdict claiming support MUST have been produced by a leg that measured something.
        // An entry with supported=yes and leg=unprobed is exactly the laundered claim this file
        // exists to refuse, and it is refused AT PARSE TIME so it can never be looked up.
        if (entry.supported && entry.leg == ProbeLeg::Unprobed) {
            list.parse_error_ =
                "entry claims supported=yes with leg=unprobed: a support verdict must come "
                "from a measurement, and 'unprobed' is the absence of one";
            return list;
        }
        // ... and a verdict must not be self-contradictory in the other direction either: a
        // band with a recorded failing case, or an arm that was not green, cannot be supported.
        if (entry.supported && (!entry.first_failure.empty() || entry.arm_rc != 0)) {
            list.parse_error_ = "entry claims supported=yes while recording a failure (fail=\"" +
                                entry.first_failure + "\", arm_rc=" +
                                std::to_string(entry.arm_rc) +
                                "): a band is admitted only when every measured case passed "
                                "AND the arm that measured it was green";
            return list;
        }
        // ===================================================================
        // SIMULATED ROWS: NORMALISED FIRST, THEN REFUSED
        // ===================================================================
        // (S1) NORMALISE. leg=route_simulated IS a simulated row; writing the same fact two ways
        // must not produce two different verdicts one level down.
        if (entry.leg == ProbeLeg::RouteSimulated) { entry.simulated = true; }
        // (S2) A SIMULATED VERDICT MAY NOT CLAIM SUPPORT. Mirror of the `supported=yes
        // leg=unprobed` rule above, and refused HERE rather than only at gate time so the row can
        // never be looked up and can never be believed by a second consumer of the file: the
        // simulator changes which tables answer, not which kernels execute, so its verdict is a
        // ROUTE result and a route result is not support.
        if (entry.simulated && entry.supported) {
            list.parse_error_ =
                "entry claims supported=yes while marked simulated=yes (leg=" +
                std::string(probe_leg_name(entry.leg)) +
                "): a verdict produced under the test-only architecture simulator is a ROUTE "
                "result, which cannot be support. The honest record for such a run is "
                "supported=no (the path WAS exercised; the arithmetic was NOT measured) with "
                "sim_rung= naming the rung, and a real row has to be measured on the card itself "
                "with tools/archkit/probe_formats.sh.";
            return list;
        }
        list.entries_.push_back(entry);
    }

    // The completeness rule. A verdict depends on WHICH criterion produced it, on WHICH
    // binary, and on the MEASURED capability vector -- so a list missing any of these is not
    // usable and is not "partially usable" either: it refuses, loudly, rather than gating on
    // half a provenance.
    std::string missing;
    const auto need = [&missing](const std::string& value, const char* name) {
        if (value.empty()) {
            if (!missing.empty()) { missing += ", "; }
            missing += name;
        }
    };
    need(list.provenance_.probe_tool, "probe_tool");
    need(list.provenance_.criterion_id, "criterion_id");
    need(list.provenance_.criterion_src, "criterion_src");
    need(list.provenance_.reference, "reference");
    need(list.provenance_.date, "date");
    need(list.provenance_.binary_sha16, "binary_sha16");
    need(list.provenance_.source, "source");
    need(list.provenance_.key, "key");
    list.provenance_.missing = missing;
    list.provenance_.complete = missing.empty();

    // =======================================================================
    // PROVENANCE vs VERDICT: THE SIMULATED-RUN CONSISTENCY RULE
    // =======================================================================
    // `source` says HOW the run that produced this file was made; `simulated` says whether a row
    // came from a simulated route. If they disagree in either direction the file contradicts
    // itself, and a self-contradictory provenance is not usable -- it is the same class as a row
    // claiming support from a leg that measured nothing, one level up. Both directions refuse.
    if (!list.provenance_.source.empty() && list.provenance_.source != "real_gpu" &&
        list.provenance_.source != "simulated_route") {
        list.parse_error_ =
            "meta source=\"" + list.provenance_.source +
            "\" is not a provenance this mechanism can classify. It must be \"real_gpu\" (the "
            "arms ran on this card, and the routing was this card's) or \"simulated_route\" (the "
            "arms ran under the test-only architecture simulator, so the routing was a rung this "
            "card is not). An unclassifiable provenance is refused rather than assumed benign, "
            "because the one thing this file may not do is present a simulated run as a real one.";
        return list;
    }
    for (const FormatSupportEntry& entry : list.entries_) {
        const bool from_sim_run = list.provenance_.source == "simulated_route";
        if (entry.simulated && !list.provenance_.source.empty() && !from_sim_run) {
            list.parse_error_ =
                "meta source=\"real_gpu\" but an entry is marked simulated=yes (leg=" +
                std::string(probe_leg_name(entry.leg)) +
                "): a row produced under the architecture simulator cannot belong to a list whose "
                "provenance says the runs were real. Re-produce the list with "
                "tools/archkit/probe_formats.sh -- under a simulated rung it writes "
                "source=\"simulated_route\" and marks every row.";
            return list;
        }
        if (from_sim_run && !entry.simulated) {
            list.parse_error_ =
                "meta source=\"simulated_route\" but an entry is not marked simulated=yes (leg=" +
                std::string(probe_leg_name(entry.leg)) +
                "): every row of a simulated run took a simulated route, so a row that does not "
                "say so is claiming a measured routing it did not have.";
            return list;
        }
    }
    return list;
}

inline FormatSupportList FormatSupportList::load_file(const std::string& path) {
    FormatSupportList list;
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        list.parse_error_ = "cannot open '" + path + "'";
        return list;
    }
    std::string text;
    char buffer[4096];
    std::size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof(buffer), file)) > 0) { text.append(buffer, got); }
    std::fclose(file);
    return parse(text);
}

inline std::string FormatSupportList::render() const {
    std::string out;
    out += "# ninfer format support list v2 -- PRODUCED BY A PROBE RUN, NEVER HAND-MAINTAINED.\n";
    out += "# Keyed on the MEASURED capability vector, not on a name and not on a cc number:\n";
    out += "# a declaration may be false (a modified VBIOS can present a listed name over\n";
    out += "# different silicon), so only a known-answer execution test is a verdict.\n";
    out += "# declared_* fields are the card's DECLARATION, kept as an audit trail.\n";
    const auto meta = [&out](const char* key, const std::string& value) {
        if (!value.empty()) {
            out += "meta ";
            out += key;
            out += "=\"";
            out += value;
            out += "\"\n";
        }
    };
    meta("probe_tool", provenance_.probe_tool);
    meta("criterion_id", provenance_.criterion_id);
    meta("criterion_src", provenance_.criterion_src);
    meta("reference", provenance_.reference);
    meta("stats_env", provenance_.stats_env);
    meta("key", provenance_.key);
    meta("source", provenance_.source);
    meta("declared_name", provenance_.declared_name);
    meta("declared_cc", provenance_.declared_cc);
    meta("declared_build", provenance_.declared_build);
    meta("date", provenance_.date);
    meta("binary_path", provenance_.binary_path);
    meta("binary_sha16", provenance_.binary_sha16);
    meta("binary_mtime", provenance_.binary_mtime);
    meta("library_path", provenance_.library_path);
    meta("library_sha16", provenance_.library_sha16);
    meta("library_mtime", provenance_.library_mtime);
    meta("arms_not_green", provenance_.arms_not_green);
    for (const FormatSupportEntry& entry : entries_) {
        out += "entry key=\"";
        out += entry.key;
        out += "\"";
        if (!entry.declared_name.empty()) {
            out += " declared_name=\"";
            out += entry.declared_name;
            out += "\"";
        }
        if (!entry.declared_cc.empty()) {
            out += " declared_cc=\"";
            out += entry.declared_cc;
            out += "\"";
        }
        out += " format=";
        out += artifact::format_name(entry.format);
        out += " path=";
        out += entry.path;
        out += " band=";
        out += entry.band();
        out += " supported=";
        out += entry.supported ? "yes" : "no";
        out += " leg=";
        out += probe_leg_name(entry.leg);
        out += " cases=";
        out += std::to_string(entry.cases);
        out += " geometries=";
        out += std::to_string(entry.geometries);
        out += " arm_rc=";
        out += std::to_string(entry.arm_rc);
        if (entry.any_ratio) {
            char ratio[64];
            std::snprintf(ratio, sizeof(ratio), " ratio=%.6g", entry.worst_ratio);
            out += ratio;
        }
        if (!entry.first_failure.empty()) {
            out += " fail=\"";
            out += entry.first_failure;
            out += "\"";
        }
        if (entry.simulated) { out += " simulated=yes"; }
        if (!entry.sim_rung.empty()) {
            out += " sim_rung=";
            out += entry.sim_rung;
        }
        if (!entry.case_example.empty()) {
            out += " case=\"";
            out += entry.case_example;
            out += "\"";
        }
        if (!entry.probe.empty()) {
            out += " probe=\"";
            out += entry.probe;
            out += "\"";
        }
        out += "\n";
    }
    return out;
}

inline std::vector<const FormatSupportEntry*>
FormatSupportList::find_all(std::string_view key, artifact::NumericFormat format,
                            std::uint32_t m) const {
    std::vector<const FormatSupportEntry*> out;
    const ProbeBand* band = probe_band_of(m);
    if (band == nullptr || key.empty()) { return out; }
    for (const FormatSupportEntry& entry : entries_) {
        if (entry.key != key || entry.format != format) { continue; }
        if (entry.m_lo != band->m_lo || entry.m_hi != band->m_hi) { continue; }
        out.push_back(&entry);
    }
    return out;
}

inline const FormatSupportEntry* FormatSupportList::find(std::string_view key,
                                                        artifact::NumericFormat format,
                                                        std::uint32_t m,
                                                        std::string_view path) const {
    for (const FormatSupportEntry* entry : find_all(key, format, m)) {
        if (entry->path == path) { return entry; }
    }
    return nullptr;
}

inline std::vector<const FormatSupportEntry*>
FormatSupportList::find_by_declared_name(std::string_view name) const {
    std::vector<const FormatSupportEntry*> out;
    if (name.empty()) { return out; }
    for (const FormatSupportEntry& entry : entries_) {
        if (entry.declared_name == name) { out.push_back(&entry); }
    }
    return out;
}

inline ProbeFreshness FormatSupportList::freshness(std::string_view current_binary_sha16) const {
    if (provenance_.binary_sha16.empty() || current_binary_sha16.empty()) {
        return ProbeFreshness::Unknown;
    }
    return provenance_.binary_sha16 == current_binary_sha16 ? ProbeFreshness::Fresh
                                                            : ProbeFreshness::StaleBinary;
}

// ---------------------------------------------------------------------------
// 6. THE GATE
// ---------------------------------------------------------------------------
//
// IT IS OPT-IN AND OFF BY DEFAULT. With NINFER_FORMAT_PROBE_LIST unset nothing below runs and
// the engine's behaviour is byte-identical to today's -- the same discipline
// NINFER_RECALL_REACH and NINFER_KV_PAGING_PREALLOC follow, and the reason an unprobed
// mechanism can be landed without moving any shipping verdict.
//
// IT IS ALSO NOT A REPLACEMENT FOR require_artifact_formats_supported(). That gate answers
// "may this binary execute this artifact's formats on this capability" from the capability
// ladder; this one answers "did anybody MEASURE that this (measured-capability-vector, format,
// band) computes the right numbers". They are different questions and both are wanted. Where
// they differ, this one is stricter, because a measured verdict can contradict a declared one.
inline constexpr std::string_view kProbeListEnv = "NINFER_FORMAT_PROBE_LIST";
// Accepting a verdict whose recorded binary sha16 is not the caller's is a decision a human
// makes once per investigation, so it needs its own explicitly-named opt-in. The default is
// refusal, which is the engine's truth rather than the optimistic guess -- the same
// default-false shape as WeightOffloadLimits::fetch_per_layer_entry.
inline constexpr std::string_view kProbeAcceptStaleEnv = "NINFER_FORMAT_PROBE_ACCEPT_STALE";

struct FormatProbeVerdict {
    bool enabled = false;              // a list was named and read
    bool allowed = false;              // a measured entry admits this combination
    bool refused_for_absent = false;   // no entry for this (key, format, band)
    // TRUE when the card's DECLARATION disagrees with the list's measurement for that
    // declaration. This is the detectable signature of a mismatched or reflashed VBIOS: the
    // card says it is X, and the arithmetic says it is not. Never absorbed, always reported.
    bool declaration_disagrees = false;
    std::string why = {}; // never empty
};

namespace detail {

// The list is re-read when the environment names a DIFFERENT path, and not otherwise, so a run
// cannot observe two versions of one list and a caller that rewrites the file in place gets the
// version it latched rather than a half-written one. The latch is keyed on the path (rather
// than a pure once-flag) because two questions must be answerable in one process: "is the gate
// off" and "does this list admit this combination" -- and because a once-flag would make the
// whole gate untestable without a process per case.
struct ProbeListLatch {
    bool attempted = false;
    std::string path = {};
    FormatSupportList list = {};
    std::string announced_for = {};
};
inline ProbeListLatch& probe_list_latch() {
    static ProbeListLatch latch;
    return latch;
}

inline const FormatSupportList* probe_list_for_process() {
    ProbeListLatch& latch = probe_list_latch();
    const char* env = std::getenv(std::string(kProbeListEnv).c_str());
    const std::string path = env == nullptr ? std::string() : std::string(env);
    if (!latch.attempted || latch.path != path) {
        latch.attempted = true;
        latch.path = path;
        latch.list = FormatSupportList{};
        if (!path.empty()) {
            latch.list = FormatSupportList::load_file(path);
            // LOUD ONCE PER BAD PATH. A gate whose list failed to load is a gate that will
            // refuse every shape, and a run that dies for that reason must say so before it
            // dies.
            if (!latch.list.valid() && latch.announced_for != path) {
                latch.announced_for = path;
                const std::string reason =
                    latch.list.parse_error().empty()
                        ? "provenance incomplete: " + latch.list.provenance().missing
                        : latch.list.parse_error();
                std::fprintf(stderr,
                             "ninfer: %s=\"%s\" could not be used as a format support list "
                             "(%s). Every (format, shape) on this device will be REFUSED rather "
                             "than assumed good: an unreadable allow-list is not a licence to "
                             "allow.\n",
                             std::string(kProbeListEnv).c_str(), path.c_str(), reason.c_str());
            }
        }
    }
    return latch.path.empty() ? nullptr : &latch.list;
}

} // namespace detail

// WHAT WOULD HAVE TO BE PROBED -- the actionable half of a refusal, and an ON-RAMP rather than
// a dead end. Named as a command, because "run a probe" is not an instruction a reader can
// carry out, and because the whole point of abandoning names is that a person with an
// Engineering Sample, a reflashed card, or an ordinary retail card can extend the list
// themselves with the same procedure.
[[nodiscard]] inline std::string probe_refusal_requirement(const DeclaredIdentity& declared,
                                                          const MeasuredCapabilities& measured,
                                                          artifact::NumericFormat format,
                                                          std::uint32_t m) {
    const ProbeBand* band = probe_band_of(m);
    const std::string band_name =
        band == nullptr ? std::string("m=") + std::to_string(m)
                        : probe_band_name(band->m_lo, band->m_hi);
    const std::string key = measured_capability_key(measured);
    std::string out = "An entry for key=\"" + key + "\" format=" +
                      std::string(artifact::format_name(format)) + " band=" + band_name +
                      " has to exist in the support list with supported=yes and "
                      "leg=reduction_l2. ON-RAMP -- this refusal is not a dead end, and nothing "
                      "about it needs a name: on the card in question run\n"
                      "    NINFER_PROBE_KEY='" + key + "' "
                      "NINFER_PROBE_DECLARED='" + declared.render() + "' \\\n"
                      "      tools/archkit/probe_formats.sh <out.list>\n"
                      "which runs the known-answer execution tests and writes the row, then "
                      "re-run with NINFER_FORMAT_PROBE_LIST=<out.list>. The per-format arm it "
                      "needs is the linear Op test whose shape cases cover m in " + band_name +
                      ", run with NINFER_OP_REPORT_STATS=1 so the measured relative-L2 ratio is "
                      "emitted per case. ";
    if (band != nullptr) {
        out += "Band basis: ";
        out.append(band->basis);
        out += " ";
    }
    out += "A band whose cases do not reach this M (for example the NVFP4 W4A4 arm as the "
           "suite stands today: its cases are T=1,2,4,5,8,17,1024, so M 9..16 has NO case) "
           "CANNOT be admitted by this probe, and the honest record for it is an absent entry "
           "that refuses -- not an inference from the neighbouring band. ";
    out += "A capability the driver does not report is exactly what the probe answers: the "
           "driver's cc number is a hint for ordering probes and nothing else. For a rung or a "
           "toolkit below the default chain, the CUDA 12.8 tree at /mnt/g/cuda12/tk reaches "
           "compute_50/52/53 (Maxwell) and compute_60/61/62 (Pascal), so for those rungs the "
           "toolchain is not the blocker -- the kernels are.";
    return out;
}

// ---------------------------------------------------------------------------
// 4b. THE SIMULATED-VERDICT REFUSAL: NAMED, DIAGNOSTIC, AND AN ON-RAMP
// ---------------------------------------------------------------------------
//
// This is the ONE text for "a simulated verdict was offered as support", so the wording cannot
// drift between the parse-time normalisation, the gate and the report. Four things it must carry,
// and each is a requirement rather than a style:
//
//   * it says SIMULATED first, so a reader who stops after one line is not misled -- the same rule
//     route_log_line() follows with its `[route][SIMULATED] ` prefix;
//   * it names the RUNG (or says the entry did not record one, which is a defect of that row and
//     is reported as one), because without the rung the reader cannot re-probe;
//   * it states that the ARITHMETIC is UNVERIFIED and the SUPPORT is UNMEASURED, so it cannot be
//     read as a capability verdict -- and it states what the simulator IS for, so it cannot be
//     read as "this path is worthless" either;
//   * it ends with the ON-RAMP (probe_refusal_requirement): the literal command that would measure
//     the real thing on the real card. A refusal that does not tell the reader how to obtain the
//     measurement fails the requirement it exists to serve, and a refusal that could be mistaken
//     for a capability verdict fails the safety requirement. Both, or neither.
[[nodiscard]] inline std::string sim_verdict_refusal(const DeclaredIdentity& declared,
                                                     const MeasuredCapabilities& measured,
                                                     artifact::NumericFormat format, std::uint32_t m,
                                                     std::string_view entry_leg,
                                                     std::string_view entry_sim_rung,
                                                     std::string_view entry_path,
                                                     std::string_view artifact_identity) {
    const ProbeBand* band = probe_band_of(m);
    const std::string band_name =
        band == nullptr ? std::string("m=") + std::to_string(m)
                        : probe_band_name(band->m_lo, band->m_hi);
    const std::string identity =
        artifact_identity.empty() ? std::string("<unnamed artifact>") : std::string(artifact_identity);
    std::string out = "SIMULATED VERDICT -- REFUSED AS SUPPORT (this is a ROUTE result, not a "
                      "capability verdict). artifact '" + identity + "': the entry for key=\"" +
                      measured_capability_key(measured) + "\" format=" +
                      std::string(artifact::format_name(format)) + " " + band_name + " path=" +
                      std::string(entry_path) + " came from leg=" + std::string(entry_leg) +
                      ", i.e. from the test-only architecture simulator (src/core/arch_sim.h). "
                      "A simulated run exercises the ROUTE DECISION for the rung ";
    if (entry_sim_rung.empty()) {
        out += "IT DID NOT RECORD (this entry carries no sim_rung=, so this refusal cannot name "
               "the rung that was simulated -- that is a defect of the row, not a detail)";
    } else {
        out += "sm_" + std::string(entry_sim_rung) + " (sim_rung=" + std::string(entry_sim_rung) + ")";
    }
    out += ", and nothing about the arithmetic: the kernels such a run executes are the ones this "
           "BINARY was compiled for, not the simulated rung's. So this entry is NOT a measurement "
           "of support -- its SUPPORT is UNMEASURED and its ARITHMETIC is UNVERIFIED -- and it is "
           "refused as support rather than deleted, because the simulator is the only instrument "
           "that exercises those rungs on this box at all (the CUDA 12.8 chain at /mnt/g/cuda12/tk "
           "reaches compute_50/52/53, Maxwell, and compute_60/61/62, Pascal; and sm_70/sm_75 are "
           "the two rungs with a named wall). What such a run legitimately reports is \"this path "
           "was exercised on a simulated rung\" -- which is a statement about the path, never "
           "about the numbers. NOTE the two prohibitions are different and both stand: this one "
           "forbids a FUNCTIONAL/support claim, and src/core/arch_sim.h:59-70 forbids a PERFORMANCE "
           "claim, so no throughput or latency figure from a simulated run may be quoted as the "
           "rung's. To obtain the measurement this row is not, measure it on the card in question: " +
           probe_refusal_requirement(declared, measured, format, m);
    return out;
}

// THE GATE. `declared` is what the card SAYS (a hint and an audit trail); `measured` is what
// the known-answer probes SAW (the key). Passing both is what makes a disagreement detectable.
[[nodiscard]] inline FormatProbeVerdict
format_probe_gate(const DeclaredIdentity& declared, const MeasuredCapabilities& measured,
                  artifact::NumericFormat format, std::uint32_t m,
                  std::string_view artifact_identity, std::string_view current_binary_sha16) {
    FormatProbeVerdict out;
    const FormatSupportList* list = detail::probe_list_for_process();
    if (list == nullptr) {
        out.why = std::string(kProbeListEnv) +
                  " is not set, so the measured-support gate is disabled and this call changes "
                  "nothing (src/core/format_probe.h). The device's MEASURED key is \"" +
                  measured_capability_key(measured) + "\" and its DECLARATION is " +
                  declared.render() +
                  "; if this run is relying on the declaration, it is relying on something a "
                  "modified VBIOS can falsify. Set NINFER_FORMAT_PROBE_LIST to a list produced "
                  "by tools/archkit/probe_formats.sh to gate on measurement instead.";
        return out;
    }
    out.enabled = true;

    const std::string identity =
        artifact_identity.empty() ? std::string("<unnamed artifact>") : std::string(artifact_identity);
    const std::string key = measured_capability_key(measured);
    const std::string band_name = [m] {
        const ProbeBand* band = probe_band_of(m);
        return band == nullptr ? std::string("m=") + std::to_string(m)
                               : probe_band_name(band->m_lo, band->m_hi);
    }();

    if (!list->valid()) {
        out.why = "artifact '" + identity + "': " + std::string(kProbeListEnv) +
                  " was set but the list could not be used (" +
                  (list->parse_error().empty() ? "provenance incomplete: " + list->provenance().missing
                                               : list->parse_error()) +
                  "). REFUSING rather than proceeding: an unreadable allow-list is not a licence "
                  "to allow, and a list with no provenance cannot be told from a stale one. Fix "
                  "the list, or unset the variable to run the shipping path.";
        return out;
    }

    // THE DECLARATION-vs-MEASUREMENT CHECK, and it comes BEFORE the lookup on purpose. A row is
    // found by the HINT index (the declaration), and if that row's measured key is not this
    // device's measured key, then the card DECLARES an identity whose measurements do not
    // reproduce here. That is not a cache miss and it must not be absorbed as one: it is the
    // signature of a reflashed or mismatched card, and the honest action is to invalidate the
    // row for this device, refuse, and say so.
    const std::vector<const FormatSupportEntry*> hinted =
        list->find_by_declared_name(declared.name);
    for (const FormatSupportEntry* entry : hinted) {
        if (entry->key == key) { continue; }
        out.declaration_disagrees = true;
        out.why = "artifact '" + identity + "': DECLARATION AND MEASUREMENT DISAGREE. This "
                  "device declares " + declared.render() + ", and the list has a row measured "
                  "under that declaration whose MEASURED capability key is \"" + entry->key +
                  "\" -- but this device's own known-answer probes measured \"" + key +
                  "\". A declaration that does not reproduce the measurements taken under it is "
                  "the signature of a mismatched or reflashed card (a modified VBIOS can present "
                  "a listed name over different silicon). The declaration is NOT the verdict and "
                  "this row is NOT admitted; it is invalidated FOR THIS DEVICE and a re-probe is "
                  "required before any route may be trusted. " +
                  probe_refusal_requirement(declared, measured, format, m);
        return out;
    }

    const std::vector<const FormatSupportEntry*> entries = list->find_all(key, format, m);
    if (entries.empty()) {
        out.refused_for_absent = true;
        out.why = "artifact '" + identity + "': NOT PROBED -- no row for key=\"" + key +
                  "\" format=" + std::string(artifact::format_name(format)) + " at " + band_name +
                  ", so this run refuses it. The support list is an allow-list and an absent row "
                  "is the engine's honest answer for silicon nobody measured; it is deliberately "
                  "not a default-allow, and there is no default path in this file at all. " +
                  probe_refusal_requirement(declared, measured, format, m);
        return out;
    }

    // EVERY MEASURED ACTIVATION PATH MUST BE ADMITTED, because which path a run takes is a
    // DISPATCH decision this header cannot make. NVFP4 has an A16 route and a W4A4 route, and
    // src/ops/linear/nvfp4/nvfp4_dispatch.cpp:resolve_route() picks between them by token count
    // and problem class. If the A16 path is measured good and the W4A4 path is unprobed, "is
    // this format supported at this band" has no single answer, so the gate refuses rather than
    // answering the easier half.
    std::string paths_admitted;
    for (const FormatSupportEntry* entry : entries) {
        // (1) A SIMULATED VERDICT IS REFUSED BY NAME, AND THIS IS THE FIRST THING CHECKED. THE
        // ORDER IS LOAD-BEARING: such a row is recorded the honest way -- supported=no with
        // sim_rung= naming the rung, because the path WAS exercised and the arithmetic was NOT
        // measured -- so a check that looked at `supported` first would answer it with the
        // generic "MEASURED UNSUPPORTED" text. That text reads as a capability verdict, and a
        // simulated verdict refused as if it were a capability verdict is the misreading this
        // whole mechanism exists to prevent. So the simulated case is named first, and it is the
        // only case in this loop that is refused for what it IS rather than for what it lacks.
        // The parse-time S2 rule is the second lock (a supported=yes/simulated=yes row cannot even
        // be parsed), and the arithmetic check below is the third and is unreachable for
        // supported=yes by construction.
        if (entry->is_simulated()) {
            out.why = sim_verdict_refusal(declared, measured, format, m,
                                          probe_leg_name(entry->leg), entry->sim_rung, entry->path,
                                          identity);
            return out;
        }
        if (!entry->supported) {
            out.why = "artifact '" + identity + "': MEASURED UNSUPPORTED -- key=\"" + key +
                      "\" format=" + std::string(artifact::format_name(format)) + " " + band_name +
                      " path=" + entry->path +
                      " was probed and did not pass its criterion (leg=" +
                      std::string(probe_leg_name(entry->leg)) +
                      ", cases=" + std::to_string(entry->cases) +
                      (entry->first_failure.empty()
                           ? std::string()
                           : ", FIRST FAILING CASE \"" + entry->first_failure + "\"") +
                      (entry->arm_rc == 0
                           ? std::string()
                           : ", the arm itself exited rc=" + std::to_string(entry->arm_rc) +
                                 " so its other passes in this band are not counted as support") +
                      (entry->any_ratio ? ", worst measured error/limit ratio=" +
                                              std::to_string(entry->worst_ratio)
                                        : std::string()) +
                      (entry->case_example.empty() ? std::string()
                                                   : ", from case \"" + entry->case_example + "\"") +
                      "). This is a measurement, not a missing row. " +
                      probe_refusal_requirement(declared, measured, format, m);
            return out;
        }
        // (2) THE SECOND LOCK: a leg that measured no arithmetic is not support. For
        // supported=yes this is now UNREACHABLE by construction -- the simulated leg is named and
        // refused above (1), and the parse-time rules refuse supported=yes with leg=unprobed --
        // and it is kept as a lock anyway. It is NOT claimed as covered by a test: a branch no
        // input can reach is not evidence, and this comment exists so its presence cannot be read
        // as coverage.
        if (!entry->answers_arithmetic()) {
            out.why = "artifact '" + identity + "': the support entry for key=\"" + key +
                      "\" format=" + std::string(artifact::format_name(format)) + " " + band_name +
                      " path=" + entry->path + " came from leg=" +
                      std::string(probe_leg_name(entry->leg)) +
                      ", which is a ROUTE verdict and not an arithmetic one. The arch simulator "
                      "is a legitimate instrument for 'does this rung take this path' -- but the "
                      "kernels it runs are the ones the BINARY was compiled for, so no simulated "
                      "arm may be recorded as support for the numbers. " +
                      probe_refusal_requirement(declared, measured, format, m);
            return out;
        }
        if (!paths_admitted.empty()) { paths_admitted += "+"; }
        paths_admitted += entry->path;
    }

    const ProbeFreshness fresh = list->freshness(current_binary_sha16);
    if (fresh == ProbeFreshness::StaleBinary) {
        const char* accept = std::getenv(std::string(kProbeAcceptStaleEnv).c_str());
        const bool accepted = accept != nullptr && accept[0] != '\0' && std::string(accept) != "0";
        if (!accepted) {
            out.why = "artifact '" + identity + "': the support entry for key=\"" + key +
                      "\" format=" + std::string(artifact::format_name(format)) + " " + band_name +
                      " was measured with a DIFFERENT binary (list records binary_sha16=" +
                      list->provenance().binary_sha16 +
                      (list->provenance().binary_mtime.empty()
                           ? std::string()
                           : " built " + list->provenance().binary_mtime) +
                      "), while this caller is " +
                      (current_binary_sha16.empty() ? std::string("unidentified")
                                                    : std::string(current_binary_sha16)) +
                      ". A stale entry is exactly the row this mechanism exists to stop being "
                      "believed, so it is refused by default. Re-run the probe against this "
                      "binary, or set " + std::string(kProbeAcceptStaleEnv) +
                      "=1 to accept a verdict about a binary that is not this one.";
            return out;
        }
        out.allowed = true;
        out.why = "artifact '" + identity + "': support entry admitted from a STALE list at the "
                  "operator's explicit request (" + std::string(kProbeAcceptStaleEnv) +
                  "=1); the verdict is about binary " + list->provenance().binary_sha16 +
                  ", not about " + std::string(current_binary_sha16) + ".";
        return out;
    }

    out.allowed = true;
    out.why = "artifact '" + identity + "': MEASURED SUPPORTED -- key=\"" + key +
              "\" (measured: " + measured_capability_key(measured) + ") format=" +
              std::string(artifact::format_name(format)) + " " + band_name +
              ", activation path(s) " + paths_admitted + ", criterion=" +
              list->provenance().criterion_id + " (" + list->provenance().criterion_src +
              "), reference=" + list->provenance().reference + ", probed " +
              list->provenance().date + " by " + list->provenance().probe_tool +
              ", freshness=" + std::string(probe_freshness_name(fresh)) +
              ". The DECLARATION " + declared.render() +
              " played no part in this verdict: it is recorded for audit only, and the list's "
              "own declared side is " +
              (list->provenance().declared_name.empty()
                   ? std::string("unrecorded")
                   : "'" + list->provenance().declared_name + "' (" +
                         list->provenance().declared_build + ")") +
              ". Every measured activation path is covered, because the dispatcher may take any "
              "of them for a given shape.";
    return out;
}

// The throwing form, so a caller that wants the engine's own loud refusal gets the same shape
// require_artifact_formats_supported() and build_weight_offload_plan() use: a
// std::invalid_argument carrying the full reason. Silent when the gate is disabled or when the
// combination is admitted -- the ONE refusal path stays the evidence-backed one.
inline void require_artifact_format_probed(const DeclaredIdentity& declared,
                                           const MeasuredCapabilities& measured,
                                           artifact::NumericFormat format, std::uint32_t m,
                                           std::string_view artifact_identity,
                                           std::string_view current_binary_sha16 = {}) {
    const FormatProbeVerdict verdict =
        format_probe_gate(declared, measured, format, m, artifact_identity, current_binary_sha16);
    if (!verdict.enabled || verdict.allowed) {
        if (verdict.enabled) {
            std::fprintf(stderr, "ninfer: measured-support gate: %s\n", verdict.why.c_str());
        }
        return;
    }
    throw std::invalid_argument("measured-support gate: " + verdict.why);
}

} // namespace ninfer::caps
