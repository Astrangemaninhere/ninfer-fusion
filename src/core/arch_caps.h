#pragma once

// Per-architecture capability ladder and the artifact-format capability gate.
//
// Why this file exists
// --------------------
// Before this landing the engine's only architecture statement was a blanket reject in
// src/targets/qwen3_6/impl/runtime/layouts_impl.h ("device.sm() != 120"; deleted
// 2026-09-13, tombstone at layouts_impl.h:993-994), which cannot
// name WHICH artifact format is unsupported or WHY. The gate below is taken at artifact load
// time (src/targets/registry.cpp, construct_target), where the artifact's numeric formats,
// the device's compute capability and the artifact identity are all in scope at once, and it
// is taken before any weight is planned, materialized or uploaded.
//
// What is authoritative here
// --------------------------
// Two independent sources, both recorded per row so nothing is asserted without a citation:
//   1. KERNEL evidence -- the file:line of the mma/ldmatrix intrinsic the engine actually
//      emits for that weight format. That fixes the *hardware floor* of a persisted format
//      independently of any profile, plan or document.
//   2. The project's own route table, tools/archkit/_GPU_MATRIX.md, which fixes the per-card
//      *route* text quoted back to the operator.
// A format with no tensor-core kernel (FP32/I32: scale words, indices, control payloads) has
// no floor and is never gated.
//
// Host-only by construction: no CUDA header, no device query, no cudaGetDeviceProperties.
// The caller passes the compute capability in, so every row of this table is exercisable on
// any machine -- including a single-GPU host (see tests/test_arch_caps.cpp).
//
// What this gate is NOT
// ---------------------
// It is not the placement planner and it does not make a device pick work. It answers one
// question -- "can the registered kernels for this artifact's formats execute on this
// compute capability at all?" -- and, when the answer is no, it names the missing capability
// and the route that would work instead.

#include "artifact/reader.h"
// The probe half. A measured capability set IS a capability fact and its vocabulary
// (DeviceCapability, ProbeStatus, CapabilityReport) already lives there. No cycle:
// device_capabilities.h does not include this file -- MEASURED, and re-measured after
// this change, over the build's own -MM dep lists (see the report).
#include "core/device_capabilities.h"

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::caps {

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

// One bit per instruction-set family the engine's registered kernels actually emit. The
// names are capability classes, not vendors' marketing names, so a row can be justified by
// the asm string in the kernel that needs it.
enum class Cap : std::uint32_t {
    None                = 0u,
    Fp16Mma             = 1u << 0, // mma.sync...f16.f16.f32
    Int8Mma             = 1u << 1, // mma.sync...s32.s8.s8.s32
    Bf16Mma             = 1u << 2, // mma.sync...f32.bf16.bf16.f32
    Fp8E4m3MmaPlain     = 1u << 3, // mma.sync...f32.e4m3.e4m3.f32  (no kind:: qualifier)
    Fp8F8f6f4KindMma    = 1u << 4, // mma.sync.aligned.kind::f8f6f4...
    Mxf4Nvfp4BlockScale = 1u << 5, // mma.sync.aligned.kind::mxf4nvf4.block_scale...
};

constexpr Cap operator|(Cap left, Cap right) noexcept {
    return static_cast<Cap>(static_cast<std::uint32_t>(left) |
                            static_cast<std::uint32_t>(right));
}

constexpr Cap operator&(Cap left, Cap right) noexcept {
    return static_cast<Cap>(static_cast<std::uint32_t>(left) &
                            static_cast<std::uint32_t>(right));
}

constexpr Cap operator~(Cap set) noexcept {
    return static_cast<Cap>(~static_cast<std::uint32_t>(set));
}

constexpr bool has_cap(Cap set, Cap wanted) noexcept {
    return (static_cast<std::uint32_t>(set) & static_cast<std::uint32_t>(wanted)) != 0u;
}

// True when `set` contains every bit of `wanted`.
constexpr bool covers(Cap set, Cap wanted) noexcept {
    return (static_cast<std::uint32_t>(set) & static_cast<std::uint32_t>(wanted)) ==
           static_cast<std::uint32_t>(wanted);
}

std::string_view cap_name(Cap single) noexcept;

// ---------------------------------------------------------------------------
// The ladder
// ---------------------------------------------------------------------------

struct ArchRung {
    int sm;                        // major * 10 + minor, exactly DeviceContext::sm()
    std::string_view label;        // microarchitecture
    std::string_view cards;        // representative parts
    Cap caps;
    std::string_view route;        // per-card route text (tools/archkit/_GPU_MATRIX.md)
};

// Ascending sm. Compute capability is the only architecture fact the runtime can observe
// (cudaDeviceProp::major/minor); the '-a' arch-accelerated suffix of a build target is NOT
// observable at runtime -- see build_arch_note().
inline constexpr ArchRung kArchLadder[] = {
    {70, "Volta", "V100-SXM2/PCIe", Cap::Fp16Mma,
     "fp16 mma only (qpn_kernels.cuh:696 uses mma.sync.m8n8k4). No bf16, no int8 tensor "
     "core, no fp8, no fp4. _GPU_MATRIX.md sm-70 row: QPN2 W4A16 runs published NVFP4/FP8 "
     "weights by expanding 4-bit codes into fp16 mma; no requantization and no fp16 weight "
     "copy needed. THE SAME ROW'S STATUS COLUMN SAYS THAT IS NOT DONE YET (it reads "
     "\"pending port\" -- the cell the citation above omits), so this row must not be read "
     "as \"the V100 has a route\": no CMake target compiles src/ops/linear/qpn/, and "
     "src/core/kernel_route.h therefore refuses NVFP4 / FP8_E4M3FN_ROW_BF16S on sm_70 with a "
     "NoKernelInTree that names the missing QPN kernel. The row stays because fp16 mma is "
     "what it is -- it is the route that is pending, not the capability."},
    {75, "Turing", "RTX 20xx / T4", Cap::Fp16Mma | Cap::Int8Mma,
     "fp16 + int8 tensor core; NO bf16. The shipped Q4/Q5/Q6/W8 linear kernels all bottom "
     "out in mma_bf16 (sm_80 floor), so the groupwise-int profile is NOT executable here as "
     "the sources stand -- it needs the QPN fp16 path or an mma_s8 port of "
     "q4/q5/q6/w8_*_gemm_mma.cuh. The QPN fp16 path is the one named in the sm-70 row "
     "above: it exists in src/ops/linear/qpn/ and is compiled by no target in this tree, so "
     "today it is the mma_s8 port that would have to be written first."},
    {80, "Ampere GA100", "A100 / A30", Cap::Fp16Mma | Cap::Int8Mma | Cap::Bf16Mma,
     "bf16 + int8 tensor core; no fp8, no fp4. groupwise-int (W8G32) and Q4/Q5/Q6 run "
     "through mma_bf16. _GPU_MATRIX.md sm-80/86 row: groupwise-int profile already exists."},
    {86, "Ampere GA10x", "RTX 30xx / A10", Cap::Fp16Mma | Cap::Int8Mma | Cap::Bf16Mma,
     "same floor as sm_80. A community sm_86 fork (Npfries) is reported to run the released "
     "groupwise-int artifact; its throughput is PENDING HARDWARE and no number is quoted "
     "here -- this tree has measured no non-120 tier at all. (An earlier revision of this row "
     "printed a community throughput figure to the operator, unmarked as unmeasured, citing "
     "a per-device file that does not exist in this repository; both are gone.)"},
    {87, "Ampere GA10x refresh", "Orin / RTX 30xx super",
     Cap::Fp16Mma | Cap::Int8Mma | Cap::Bf16Mma,
     "rows added because `nvcc --list-gpu-arch` lists compute_87: a buildable target "
     "with no row here was refused by an exact-match table. Same capability set as "
     "sm_86 (same Ampere GA10x family); no fp8, no fp4."},
    {88, "Ampere GA10x refresh", "Orin / RTX 30xx super",
     Cap::Fp16Mma | Cap::Int8Mma | Cap::Bf16Mma,
     "rows added because `nvcc --list-gpu-arch` lists compute_88. Same set as sm_86. "
     "NOT separately measured on hardware; the set is the family's, and the note in "
     "_GPU_MATRIX.md applies -- any run/not-run conclusion must come from the engine's "
     "own probe, not from this row."},
    {89, "Ada", "RTX 40xx / L40S",
     Cap::Fp16Mma | Cap::Int8Mma | Cap::Bf16Mma | Cap::Fp8E4m3MmaPlain,
     "adds plain-form fp8 e4m3 mma. NOTE the engine's A8 fp8 kernel emits "
     "kind::f8f6f4 (fp8_a8_mma.cuh:247 -> ops/common/mma.cuh:59), which is a Blackwell "
     "form, so today only the A16 fp8 route (dequant to bf16 + mma_bf16, "
     "fp8_a16_gemm_mma.cuh:206) is executable on sm_89/sm_90."},
    {90, "Hopper", "H100 / H200",
     Cap::Fp16Mma | Cap::Int8Mma | Cap::Bf16Mma | Cap::Fp8E4m3MmaPlain,
     "same executable set as sm_89 through the engine's current kernels; the kind::f8f6f4 "
     "and kind::mxf4nvf4 forms still do not exist here."},
    {100, "Blackwell DC", "B100 / B200 (sm_100a)",
     Cap::Fp16Mma | Cap::Int8Mma | Cap::Bf16Mma | Cap::Fp8E4m3MmaPlain |
         Cap::Fp8F8f6f4KindMma,
     "sm_100a executes kind::f8f6f4 (measured: ptxas -cubin rc=0) but NOT the nvfp4 form "
     "this engine emits. ops/common/mma.cuh:89 is kind::mxf4nvf4.block_scale.m16n8k64 and "
     "`nvcc -cubin -arch=sm_100a` on that wrapper exits 255, ptxas saying \"Instruction "
     "'mma with block scale' not supported on .target 'sm_100a'\"; only sm_120a/sm_121a "
     "accept it. So Cap::Mxf4Nvfp4BlockScale is NOT set here, and the sentence that used to "
     "sit here (it claimed native fp4 and cited a CMakeLists.txt guard for "
     "nvfp4_w4a4_tma* that does not exist) was a FALSE POSITIVE: it made the engine's own "
     "gate report B200 as nvfp4-capable and select nvfp4-w4a4-tma for a target whose only "
     "nvfp4 kernel cannot be assembled. Missing kernel: a tcgen05 nvfp4 path for "
     "sm_100a/103a/110a -- B200's fp4 is tcgen05.mma, not this mma.sync form."},
    {103, "Blackwell Ultra", "B300 (sm_103a)",
     Cap::Fp16Mma | Cap::Int8Mma | Cap::Bf16Mma | Cap::Fp8E4m3MmaPlain |
         Cap::Fp8F8f6f4KindMma,
     "same capability set as sm_100 for the forms this engine emits, and measured the same "
     "way: kind::f8f6f4 assembles on sm_103a (rc=0) and kind::mxf4nvf4 does not (rc=255, "
     "the same 'mma with block scale' rejection), so no Mxf4Nvfp4BlockScale bit here."},
    {120, "Blackwell RTX", "RTX 50xx / 5090D (sm_120a)",
     Cap::Fp16Mma | Cap::Int8Mma | Cap::Bf16Mma | Cap::Fp8E4m3MmaPlain |
         Cap::Fp8F8f6f4KindMma | Cap::Mxf4Nvfp4BlockScale,
     "the native target of this tree: CMAKE_CUDA_ARCHITECTURES defaults to 120a "
     "(CMakeLists.txt:19). nvfp4 W4A4 + int8 KV + rk4v4 "
     "KV are all in use here."},
    {121, "Blackwell RTX refresh", "RTX 50xx (sm_121)",
     Cap::Fp16Mma | Cap::Int8Mma | Cap::Bf16Mma | Cap::Fp8E4m3MmaPlain |
         Cap::Fp8F8f6f4KindMma | Cap::Mxf4Nvfp4BlockScale,
     "listed for the same reason as sm_103; not separately validated."},
};

inline constexpr std::size_t kArchLadderSize = sizeof(kArchLadder) / sizeof(kArchLadder[0]);

// Exact match only. nullptr means the capability set is not merely unknown to the caller but
// unknown to this table. Callers must NOT treat that as "assume the best", and must NOT
// treat it as a refusal either: an unlisted compute capability has never been measured by
// this project, so the honest answer is a WARNING plus the conservative route
// (src/core/kernel_route.h, design doc section 5.3). The list is also checked against the
// compiler's own `nvcc --list-gpu-arch`, so a target the toolchain can build should always
// have a row here -- NECESSARY, but not SUFFICIENT: a row also selects a ROUTE, and a row
// that hands a card a route whose kernel is not in this build is a claim the table cannot
// back. Measured (nvcc 13.3): the two lists differ by exactly two elements -- sm_70, which
// this table HAS and CUDA 13.x cannot build (a toolkit fact, not a table gap), and sm_110,
// which the toolkit lists and this table deliberately does NOT.
// sm_110 USED TO HAVE A ROW. It was added on assembly-level evidence -- and that evidence is
// not in question: -cubin rc=0 for ldmatrix/cp.async/mma_bf16/mma_f16/mma_s8/mma_tf32/
// kind::f8f6f4/setmaxnreg, rc=255 for kind::mxf4nvf4 -- and it was withdrawn on 2026-09-15,
// because assembly-level evidence is not route-level evidence. A Cap set with no
// Mxf4Nvfp4BlockScale bit but WITH Fp16Mma is what puts the QPN arm of select_route() in play
// (src/core/kernel_route.h), and when the row was withdrawn that answered
// Selected + KernelRoute::QpnW4a16 -- the QPN W4A16 family, whose kernel THIS BUILD does not
// contain. NOTE the combination above is only that arm's ENTRY condition, not its answer:
// since 2026-09-15 the arm also reads the per-rung MEASURED lowering of the family's tensor
// core CHANNEL (kQpnMmaRungs, same file), because -cubin rc=0 for mma.sync.m8n8k4 is not
// evidence that a tensor core executes it. On sm_80 and up ptxas answers that one channel by
// emitting a CALL to a ~88-instruction FFMA routine into the same cubin, so a card in the
// combination above is told to requantize rather than handed the QPN route. Note the scoping:
// this is the m8n8k4 channel, not the whole card -- the same cubins keep the nvcuda::wmma
// m16n16k16 channel (a different route) on real HMMA.16816. The same "assembly is not the
// route" rule, one level down and one channel narrower. That is measured, not assumed: `qpn` has 0 hits in src/CMakeLists.txt, whose
// source lists are explicit (no GLOB), so nothing under src/ops/linear/qpn/ is compiled into
// any target; 0 of the 11 archives under build/src/ carry a qpn object member either; and
// tools/archkit/_GPU_MATRIX.md:15 still marks the sm-70 QPN2 W4A16 route as a pending port.
// qpn_slots() and conservative_fallback() now RETURN NoKernelInTree instead, naming that
// kernel, so the route table no longer claims a route it cannot serve: an unknown card and
// an sm_110 card both get a named refusal rather than a warning that promised a fallback
// this build cannot launch. Until that kernel is ported, sm_110 keeps the WARNING path at
// the CAPABILITY level (arch_rung(110) == nullptr) while the ROUTE table refuses it by
// name. That withdrawal is why the sentence above reads "should", not "does".

// ===========================================================================
// PROJECT routeprobe -- THE MEASUREMENT THAT REPLACES THE NAME
// ===========================================================================
// WHY THIS SECTION EXISTS. Everything above decides by NUMBER: arch_rung(sm) is an exact
// lookup on `major * 10 + minor`, and that number is a NAME, not a measurement. Two things the
// number cannot do, both of which have already cost this project runs:
//
//   * it cannot see through a REFLASHED VBIOS. cudaDeviceProp::major/minor is whatever the
//     card's firmware reports; a card that says 120 and is not one is answered as if it were.
//   * it cannot see the '-a' suffix. cudaDeviceProp carries no 'a', so an sm_120 card and an
//     sm_120a card are the same integer and NOT the same kernel set (the sm_100 row above says
//     the nvfp4 form does not assemble on a non-'a' 100).
//
// The replacement is NOT "throw the ladder away". Exactly one direction is safe, and this
// section implements only that direction:
//
//   MEASUREMENT CAN ONLY NARROW. The measured set is the row's caps MINUS every bit a probe on
//   THIS device proved absent. It never ADDS a bit: a measurement can never hand a card a kernel
//   the ladder would not have handed it, it can only take one away -- and only after running
//   the engine's own PTX on the device and disagreeing with the reference.
//
// THE THREE-VALUED ANSWER, and this is the part that must not be flattened to a bool:
//   Proved       -- a probe ran here and agreed. The bit stays.
//   Refuted      -- a probe ran here and did NOT agree (device had no image / launch failed /
//                   numbers differed). The bit GOES. This is the only verdict a name can never
//                   produce, and it is what implements the owner's ruling: a lying sm number
//                   cannot make a Refuted bit pass.
//   Unmeasurable -- this BUILD carries no probe that can answer the bit. That is NOT evidence
//                   about the device, and reading it as Refuted is the single most expensive
//                   mistake available here: it would delete capabilities on every card for
//                   every bit whose probe body was compiled out of this arch. An Unmeasurable
//                   bit leaves the row's claim standing AND NAMES THE ABSENCE.
enum class BitVerdict : std::uint8_t {
    Unmeasurable = 0,
    Proved,
    Refuted,
};

[[nodiscard]] inline std::string_view bit_verdict_name(BitVerdict verdict) noexcept {
    switch (verdict) {
    case BitVerdict::Unmeasurable: return "unmeasurable in this build";
    case BitVerdict::Proved: return "proved on this device";
    case BitVerdict::Refuted: return "refuted on this device";
    }
    return "unknown-verdict";
}

// FORWARD DECLARATION, deliberately before MeasuredCaps: the `verdict` member below calls this
// free function, and the definition follows the struct (it is the free form that a static_assert
// can reach without a whole MeasuredCaps, which carries a std::string and is therefore not a
// literal type). The first real -c caught the missing forward declaration as
// "bit_verdict was not declared in this scope; did you mean BitVerdict?".
[[nodiscard]] constexpr BitVerdict bit_verdict(Cap proved, Cap refuted, Cap single) noexcept;

// One Cap bit -> the DeviceCapability whose probe answers it. A bit with NO entry here is
// Unmeasurable BY CONSTRUCTION, and that is a VERDICT and not a gap: Cap::Fp8E4m3MmaPlain is
// exactly this case -- the engine's only plain-form fp8 kernel emits kind::f8f6f4 (the sm_89
// row above says so), so no probe was ever written for the plain form and this table refuses
// to invent one by pointing it at a probe that answers a different question.
struct CapProbe {
    Cap bit;
    ::ninfer::DeviceCapability probe;
};

inline constexpr CapProbe kCapProbes[] = {
    {Cap::Fp16Mma, ::ninfer::DeviceCapability::Fp16Mma},
    {Cap::Int8Mma, ::ninfer::DeviceCapability::Int8Mma},
    {Cap::Bf16Mma, ::ninfer::DeviceCapability::Bf16Mma},
    {Cap::Fp8F8f6f4KindMma, ::ninfer::DeviceCapability::Fp8MmaKindF8f6f4},
    {Cap::Mxf4Nvfp4BlockScale, ::ninfer::DeviceCapability::Nvfp4MmaBlockScale},
};
inline constexpr std::size_t kCapProbeCount = sizeof(kCapProbes) / sizeof(kCapProbes[0]);

// Every bit Cap can carry. One value, so "which bits did NO probe cover" is a computation over
// this list rather than a hand-maintained second list that can drift out of step.
inline constexpr Cap kAllCapBits = Cap::Fp16Mma | Cap::Int8Mma | Cap::Bf16Mma |
                                   Cap::Fp8E4m3MmaPlain | Cap::Fp8F8f6f4KindMma |
                                   Cap::Mxf4Nvfp4BlockScale;

// ---------------------------------------------------------------------------
// The key, and why the writer cannot publish without one
// ---------------------------------------------------------------------------
// The key is the (device, build) pair the numbers were taken on. It is NOT ceremony: a
// capability set measured under one build answers a DIFFERENT question than another build's
// route table asks, because the row's whole job is to name a kernel in THIS BINARY. A
// measurement carried across builds is the stale-comment defect in a new place, and this tree
// has already paid for that once (device_probe.cu:14-18, falsified by -MM).
struct MeasurementKey {
    int device_id = -1;
    std::string_view build_arches{};

    friend constexpr bool operator==(const MeasurementKey&, const MeasurementKey&) = default;
};

struct MeasuredCaps {
    Cap proved       = Cap::None; // a probe ran on this device and agreed
    Cap refuted      = Cap::None; // a probe ran on this device and disagreed / could not run
    Cap unmeasurable = Cap::None; // no probe in THIS BUILD can answer the bit
    // FALSE IS THE NEVER-PROBED ANSWER and it is the default, deliberately: a value no probe
    // wrote must not be able to look like a measurement.
    bool probed = false;
    ::ninfer::ProbeStatus baseline = ::ninfer::ProbeStatus::NotProbed; // KernelImage, i.e. "did anything run at all"
    MeasurementKey key{};
    std::string provenance{}; // one line, ALWAYS non-empty once a probe has written it

    [[nodiscard]] BitVerdict verdict(Cap single) const noexcept {
        return bit_verdict(proved, refuted, single);
    }
};

[[nodiscard]] constexpr BitVerdict bit_verdict(Cap proved, Cap refuted, Cap single) noexcept {
    if (has_cap(proved, single)) { return BitVerdict::Proved; }
    if (has_cap(refuted, single)) { return BitVerdict::Refuted; }
    return BitVerdict::Unmeasurable;
}

// The verdict, as a FREE function over the two masks, and not only as a member. A member needs
// a whole MeasuredCaps to call, and MeasuredCaps carries a std::string provenance -- so it is
// not a literal type and a member call cannot appear in a static_assert. The three-valued
// answer is the part most worth pinning at COMPILE time, so it is reachable without one.
[[nodiscard]] constexpr BitVerdict bit_verdict(Cap proved, Cap refuted, Cap single) noexcept;

// The ladder row, narrowed by what the box proved. `row` points into kArchLadder (or is null
// when the number has no row). `narrowed` is a COPY carrying the narrowed caps, and
// `effective()` picks between them -- a pointer to a member of the object you are HOLDING,
// never a pointer to a temporary's member, so the value is safe to copy and to return.
//
// `why` is ALWAYS non-empty in every state, including the happy one. A route answer that cannot
// say why it was chosen is the defect this type exists to remove.
struct MeasuredRow {
    ArchRung narrowed{};
    const ArchRung* row    = nullptr;
    bool narrowed_by_probe = false;
    Cap refuted            = Cap::None;
    Cap unmeasurable       = Cap::None;
    std::string why{};

    [[nodiscard]] const ArchRung* effective() const noexcept {
        return narrowed_by_probe ? &narrowed : row;
    }
    [[nodiscard]] bool answerable() const noexcept { return effective() != nullptr; }
};

// The whole decision, in one place, so no second copy can answer differently.
//
// NEVER-PROBED RULE -- a decision with a reason, not a default. A device that has not been
// measured falls back to the BUILD DEFAULT ROW (the row for the lowest arch this binary
// actually carries an image for) AND NAMES IT:
//   * it does NOT refuse. Refusing would take the arm away from exactly the devices the probe
//     is moot on (no CUDA device bound, a device whose context was never created), and the
//     ladder row is still a real row backed by a real kernel file.
//   * it does NOT re-probe lazily. A probe is a device operation with a cost and a device
//     binding; a route decision is a pure function and must stay one (kernel_route.h's own
//     contract: no CUDA header, no device query).
//   * it does NOT silently substitute the device's own number. That would be the name-based
//     answer wearing the measurement's clothes -- the exact defect being removed.
[[nodiscard]] inline MeasuredRow resolve_measured_row(int sm, const MeasuredCaps& measured,
                                                     std::string_view required_by);

[[nodiscard]] inline std::string cap_list_text(Cap set);

const ArchRung* arch_rung(int sm) noexcept;

// ---------------------------------------------------------------------------
// Weight formats -> required capability
// ---------------------------------------------------------------------------

struct FormatRequirement {
    artifact::NumericFormat format;
    // Capability the registered kernels for this format need. Cap::None for formats that are
    // never fed to a tensor-core GEMM (scale words, indices, control payloads).
    Cap required;
    std::string_view kernel_evidence;
    // -----------------------------------------------------------------------
    // THE SECOND FLOOR: the fp16 fallback.
    // -----------------------------------------------------------------------
    // WHY THIS FIELD EXISTS. Before it, Cap::Fp16Mma and Cap::Int8Mma were carried in
    // kArchLadder and were required by NO row of this table, so on sm_70/sm_75 they
    // decided nothing at all and every A16 format was refused with
    // `missing = Bf16Mma` even when the card had a perfectly good fp16 tensor core and
    // this tree had a kernel that used it. That was the seam. A row now names a SECOND,
    // LOWER floor whose kernel consumes the SAME persisted bytes, so a pre-Ampere card
    // gets a route instead of a refusal -- and the capability bit that was decorative
    // becomes load-bearing.
    //
    // Cap::None means "this format has no lower-floor kernel in this tree", and that is
    // the default: a format only gets a fallback when a kernel that consumes its own
    // bytes exists and has been measured. Writing a fallback here is therefore a claim
    // about a real kernel file, and this table refuses to make one up.
    //
    // A fallback is NOT a license to weaken the primary floor. It is only honoured when
    // fp16_fallback_executable() below is true, which additionally requires (a) the QPN
    // sources to be IN THIS BUILD and (b) the fallback's tensor-core CHANNEL to lower to
    // a hardware instruction on that rung. (b) is the part that is easy to get wrong:
    // mma.m8n8k4 ASSEMBLES on every rung from sm_70 to sm_120a (measured), and from
    // sm_80 up ptxas answers it with a CALL into a software FFMA routine. Honouring the
    // fallback on such a rung would name an fp16 tensor-core route and deliver an
    // FMA-pipe simulation, so the lowering check is mandatory and fail-closed on an
    // unmeasured rung.
    Cap fallback_required = Cap::None;
    std::string_view fallback_kernel_evidence = {};
};

// ONE ROW PER FORMAT THAT HAS A FLOOR, PLUS A COMPILE-TIME GATE THAT THE ENUM IS COVERED.
//
// The sentence here used to read "Exhaustive over artifact::NumericFormat (tests/
// test_arch_caps.cpp pins that)" -- and it was FALSE. This table had 9 rows for the enum's 12
// members, so I64, FP8_E4M3FN_ROW_F32S and U4Z8G16_F16S reached Verdict::UnknownFormat on EVERY
// GPU, and the test that was supposed to catch it iterated a hand-written 9-element array and
// compared kFormatRequirementCount against THAT ARRAY'S OWN LENGTH (9 == 9). A count checked
// against the thing it is standing in for is not evidence.
//
// The claim is now kept by two tables and a static_assert instead of by this sentence: every
// NumericFormat enumerator must have EITHER a row here (a real floor with a kernel citation) OR a
// named refusal in kUncoveredFormatNotes (a format this tree declares and has no kernel for).
// kFormatTableCoversEveryEnumerator walks the enum's own ordinal range at compile time, so
// appending a member to artifact/reader.h without accounting for it fails the build of every TU
// that includes this header rather than silently passing.
inline constexpr FormatRequirement kFormatRequirements[] = {
    {artifact::NumericFormat::BF16, Cap::Bf16Mma,
     "ops/linear/bf16/bf16_gemm_mma.cuh:273 -> ops/common/mma.cuh:33 (mma_bf16, "
     "mma.sync.aligned.m16n8k16...bf16.bf16.f32)"},
    {artifact::NumericFormat::FP32, Cap::None,
     "scale words / control payloads (artifact/reader.h:23); never a tensor-core operand"},
    {artifact::NumericFormat::I32, Cap::None,
     "control and index payloads (artifact/reader.h:24); never a tensor-core operand"},
    {artifact::NumericFormat::Q4G64_F16S, Cap::Bf16Mma,
     "ops/linear/q4/q4_rowsplit_gemm_mma.cuh:341 and q4/q4_small_t_mma.cuh:168 (mma_bf16, "
     "NOT a 4-bit tensor core)"},
    {artifact::NumericFormat::Q5G64_F16S, Cap::Bf16Mma,
     "ops/linear/q5/q5_rowsplit_gemm_mma.cuh:385 (mma_bf16)"},
    {artifact::NumericFormat::Q6G64_F16S, Cap::Bf16Mma,
     "ops/linear/q6/q6_rowsplit_gemm_mma.cuh:394 (mma_bf16)"},
    {artifact::NumericFormat::W8G32_F16S, Cap::Bf16Mma,
     "ops/linear/w8/w8_rowsplit_gemm_mma.cuh:267 and w8/w8_small_t_mma.cuh:230 (mma_bf16, "
     "NOT mma_s8: the groupwise-int profile is not an int8-tensor-core route)"},
    {artifact::NumericFormat::NVFP4, Cap::Mxf4Nvfp4BlockScale,
     "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh:308 -> ops/common/mma.cuh:90 "
     "(kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.e2m1.e2m1). MEASURED: that asm "
     "assembles on sm_120a/sm_121a and the 120/121 family targets only; sm_100a/sm_103a/"
     "sm_110a reject it, so this format's floor is sm_120, not sm_100.",
     // The fallback consumes the SAME packed e2m1 codes + e4m3 scales this format already
     // stores; it does not requantize. mma.cuh:63's m8n8k4 form is the channel.
     Cap::Fp16Mma,
     "QPN W4A16: src/ops/linear/qpn/qpn_kernels.cuh -- skinny_nvfp4_qpn_simt<M> (M 1..3), "
     "skinny_nvfp4_qpn<1> (M 4..8), skinny_nvfp4_qpn<2> (M 9..16), host entry gemm_qpn in "
     "src/ops/linear/qpn/qpn_host.cu -- expands the 4-bit e2m1 codes inline into fp16 mma "
     "and needs Cap::Fp16Mma only. Its tensor-core channel is mma.sync.aligned.m8n8k4 "
     "(qpn_kernels.cuh:695,:893; helper ops/common/mma.cuh mma_f16_m8n8k4)."},
    {artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S, Cap::Bf16Mma,
     "A16 route: ops/linear/fp8/fp8_a16_gemm_mma.cuh:206 (mma_bf16). A8 route: "
     "ops/linear/fp8/fp8_a8_mma.cuh:247 (mma_fp8_e4m3, kind::f8f6f4, Blackwell form). "
     "THERE IS DELIBERATELY NO FALLBACK ROW HERE, and the reason is measured: the fp8 arm of "
     "the QPN family (skinny_fp8_qpn8, skinny_fp8_qpn8_mt2, qpn_kernels.cuh:1227/:1327) "
     "exists as a template but has NO HOST ENTRY -- qpn_host.cu's gemm_qpn dispatches "
     "skinny_nvfp4_qpn_simt / <1> / <2> and nothing else, and a single-file grep finds the two "
     "fp8 kernels mentioned NOWHERE outside the prose of this file and kernel_route.h. "
     "Declaring them as this format's fallback would hand a pre-Ampere card a route naming a "
     "kernel nothing can launch, which is the phantom this table exists to prevent (and it is "
     "the same failure mode as the sm_100 nvfp4 false positive above). So an fp8 artifact "
     "still REFUSES on sm_70/sm_75, and the missing piece is a gemm_qpn fp8 dispatch -- not "
     "the mma channel, which the nvfp4 arm already proves works there."},
    // -----------------------------------------------------------------------
    // Appended 2026-09-18 (FMTCOVER): the two formats whose absence from this table was a TABLE
    // defect and not a support decision. Both are declared, both have their own encoded geometry
    // in artifact/storage_layouts.cpp, both are read by a compiled-in consumer, and neither is
    // ever a tensor-core operand -- which is exactly the shape of the FP32/I32 rows above, so
    // they take Cap::None for that reason and not one bit more. Cap::None here is a claim that no
    // GEMM reads these bytes; it is verifiable and it is verified in the citation. (The third
    // missing format, FP8_E4M3FN_ROW_F32S, IS a GEMM operand and is deliberately NOT here -- see
    // kUncoveredFormatNotes.)
    {artifact::NumericFormat::I64, Cap::None,
     "host-mapped 8-byte control/index payload (artifact/reader.h:34). Its only producer binds it "
     "through bind_mapped -- the bytes stay mapped from the artifact file and are never uploaded "
     "-- at src/targets/qwen3_8_flash_next/impl/load/bindings.cpp:197-202 (layer_multipliers [3], "
     "ngram_head_offsets [16], ngram_head_vocab_sizes [16]), and its only reader is "
     "require_i64_values (same file, :51-69), which bit-casts them on the CPU. contiguous-le-v1 "
     "sizes it as direct 8-byte words (artifact/storage_layouts.cpp:51-63, whose own error text "
     "already names I64). No tensor-core route exists for it: there is no QType enumerator for it "
     "(src/core/tensor.h:30-45) and no arm of src/ops/linear consumes it; never a tensor-core "
     "operand."},
    {artifact::NumericFormat::U4Z8G16_F16S, Cap::None,
     "host-mapped PLE n-gram table in packed-u4-g16-v1 (artifact/reader.h:36; geometry "
     "artifact/storage_layouts.cpp:293-315). Bound with retain_mapped_tensor -- again never "
     "uploaded as such -- at src/targets/qwen3_8_flash_next/impl/load/bindings.cpp:205-207 "
     "(shards [2500012,160]). The bytes are validated and decoded on the HOST by "
     "make_ple_shard_view / dequantize_ple_row "
     "(src/targets/qwen3_8_flash_next/impl/ple_table.cpp:74-106: elementwise "
     "(code - 8) * fp16_scale), and the PLE path gathers rows into BF16 "
     "(src/ops/ple/ple_table.cu, added to the ninfer_ops sources at src/CMakeLists.txt:173). No "
     "QType enumerator exists for U4 (src/core/tensor.h:30-45) and no src/ops/linear arm consumes "
     "it; never a tensor-core operand."},
};

inline constexpr std::size_t kFormatRequirementCount =
    sizeof(kFormatRequirements) / sizeof(kFormatRequirements[0]);

// ---------------------------------------------------------------------------
// Formats this tree DECLARES and can size but has NO KERNEL for, on ANY rung
// ---------------------------------------------------------------------------
//
// A kFormatRequirements row is a claim about a real kernel file -- the fallback note above says
// in as many words that "this table refuses to make one up". FP8_E4M3FN_ROW_F32S is the case
// where that doctrine says the honest answer is NOT a row: the format is a first-class artifact
// citizen (the wire format carries its name, artifact/reader.cpp:98; its encoded geometry is
// implemented with a 4-byte scale word, artifact/storage_layouts.cpp:228-241, which is
// deliberately NOT the 2-byte word its BF16-scale sibling uses; a target binds and materializes
// it) but no op in this tree can execute it, so a row would name a kernel that cannot be
// launched -- the phantom this table exists to prevent.
//
// WHY THIS TABLE EXISTS AT ALL. Without it the gate answered Verdict::UnknownFormat, whose text
// tells the operator that the TABLE is what fixes the floor and sends them here to add a row. For
// this format that on-ramp is backwards: the row would convert a loud refusal into a silent claim
// of support. So the refusal is KEPT and made diagnosable -- it fires on every rung, and it names
// the format, the encoding, and the exact pieces that would have to exist before a row is
// written.
struct UncoveredFormatNote {
    artifact::NumericFormat format;
    // The channel a REAL kernel for this format would have to take. Recorded so the refusal can
    // say what the missing kernel would lower to. It is NOT a claim that meeting this floor is
    // sufficient, and the reason text says so.
    Cap would_need;
    std::string_view reason;
};

inline constexpr UncoveredFormatNote kUncoveredFormatNotes[] = {
    {artifact::NumericFormat::FP8_E4M3FN_ROW_F32S, Cap::Bf16Mma,
     "this tree declares it, sizes it and binds it, and has NO KERNEL that can execute it on ANY "
     "rung, so there is deliberately no kFormatRequirements row for it: a row is a claim about a "
     "kernel file, and adding one would turn this refusal into a false claim of support. It IS a "
     "GEMM operand (unlike the Cap::None host-metadata rows above), so its floor is real -- the "
     "row it will need once a kernel exists is the same Bf16Mma floor, and the same A16/A8 route, "
     "that its BF16-scale sibling FP8_E4M3FN_ROW_BF16S carries. The two encodings must NOT be "
     "aliased onto each other: the scale word is 2 bytes for ..._BF16S and 4 for this one, so "
     "collapsing them halves the scale plane (artifact/storage_layouts.cpp:228-241). WHAT IS "
     "MISSING, measured 2026-09-18: (1) the op layer has no arm for it -- "
     "src/ops/linear/linear.cpp:98-124 dispatches Q4/Q5/Q6/W8/BF16_CTRL/NVFP4/"
     "FP8_E4M3FN_ROW_BF16S and then throws 'linear: unsupported weight qtype' with no case for "
     "QType::FP8_E4M3FN_ROW_F32S (src/core/tensor.h:44), and "
     "linear_workspace_capacity_bytes() reaches the same throw at linear.cpp:189; (2) the only "
     "row-scale fp8 validator requires QType::FP8_E4M3FN_ROW_BF16S, scale_dtype == DType::BF16, "
     "scale_nb[0] == 2 and an n*2 scale plane (src/ops/linear/fp8/fp8_format.cpp:46-52), none of "
     "which this encoding satisfies; (3) the importing target names it and cannot run it -- "
     "src/targets/qwen3_8_flash_next/impl/gdn.cpp:75, qsa_attention.cpp:79 and "
     "text_decode.cpp:199 pass QType::FP8_E4M3FN_ROW_F32S with LinearPolicy::AllowA8, and that "
     "target is not registered at all (src/targets/registry.cpp:556-562: it has no engine-facing "
     "Instance adapter); (4) the MATERIALIZER has no arm for it either, and that is the layer where "
     "a hasty fix would be silently WRONG -- typed_binding.cpp:13-30 (storage_layout_for) and "
     ":32-54 (qtype_for) have no case for this format, and the row-scaled branch a row would send "
     "it into, row_scale_weight() (typed_binding.cpp:126-146), HARD-CODES scale_dtype == "
     "DType::BF16 with scale_nb[0] == 2 and scale_nb[1] == rows * 2. Routing this 4-byte-scale "
     "encoding through that branch would describe a half-size scale plane and read the wrong scale "
     "word, silently. So the materializer must NOT be extended until row_scale_weight() takes the "
     "scale-word width from the format (artifact/storage_layouts.cpp:228-241 is the only place that "
     "knows it is 4 for this encoding and 2 for its sibling). WHAT WOULD HAVE TO EXIST: an op arm "
     "for the 4-byte-scale encoding -- a "
     "QType::FP8_E4M3FN_ROW_F32S case in the linear dispatch plus a validator that accepts "
     "scale_dtype == DType::FP32, scale_nb[0] == 4 and an n*4 scale plane -- and only then the "
     "row above belongs in kFormatRequirements, with tools/archkit/probe_formats.sh (whose format "
     "set IS the set of formats that have a linear arm) reporting that arm green first. Until "
     "then this refusal is the honest answer, and it is an answer about the missing KERNEL, not "
     "about a missing table entry."},
};

inline constexpr std::size_t kUncoveredFormatCount =
    sizeof(kUncoveredFormatNotes) / sizeof(kUncoveredFormatNotes[0]);

// THE ORDINAL BOUND COMES FROM THE ENUM, NOT FROM A NAME COPIED INTO THIS FILE. The first draft
// of this block bounded the walk with `artifact::NumericFormat::U4Z8G16_F16S` written out here --
// which reproduces the very defect being fixed one level down: a member appended after the copied
// name would be invisible to the walk, so the walk would pass while a format went uncovered. The
// enum carries its own count as its last member (artifact/reader.h, NumericFormat::Count) and that
// is what is used. Members are appended in place and never renumbered (artifact/reader.h:31-33),
// which is what makes an ordinal walk a complete enumeration of the enum.
inline constexpr std::size_t kFormatOrdinalCount =
    static_cast<std::size_t>(artifact::NumericFormat::Count);

constexpr const FormatRequirement*
find_format_requirement(artifact::NumericFormat format) noexcept {
    for (const FormatRequirement& requirement : kFormatRequirements) {
        if (requirement.format == format) { return &requirement; }
    }
    return nullptr;
}

constexpr const UncoveredFormatNote*
find_uncovered_format_note(artifact::NumericFormat format) noexcept {
    for (const UncoveredFormatNote& note : kUncoveredFormatNotes) {
        if (note.format == format) { return &note; }
    }
    return nullptr;
}

namespace detail {

// EVERY enumerator accounted for. This is the walk that replaces the old prose claim, and it is
// what makes a forgotten format a BUILD failure instead of a runtime UnknownFormat.
constexpr bool format_table_has_no_unaccounted_enumerator() noexcept {
    for (std::size_t raw = 0; raw < kFormatOrdinalCount; ++raw) {
        const auto format = static_cast<artifact::NumericFormat>(raw);
        if (find_format_requirement(format) == nullptr &&
            find_uncovered_format_note(format) == nullptr) {
            return false;
        }
    }
    return true;
}

// ... exactly ONCE. A duplicate row would silently shadow (format_requirement returns the first
// match) and a format in BOTH tables would have its refusal shadowed by its row -- neither of
// which the coverage walk above can see.
constexpr bool format_tables_account_for_each_enumerator_once() noexcept {
    if (kFormatRequirementCount + kUncoveredFormatCount != kFormatOrdinalCount) { return false; }
    for (std::size_t i = 0; i < kFormatRequirementCount; ++i) {
        for (std::size_t j = i + 1; j < kFormatRequirementCount; ++j) {
            if (kFormatRequirements[i].format == kFormatRequirements[j].format) { return false; }
        }
        if (find_uncovered_format_note(kFormatRequirements[i].format) != nullptr) { return false; }
    }
    for (std::size_t i = 0; i < kUncoveredFormatCount; ++i) {
        for (std::size_t j = i + 1; j < kUncoveredFormatCount; ++j) {
            if (kUncoveredFormatNotes[i].format == kUncoveredFormatNotes[j].format) {
                return false;
            }
        }
    }
    return true;
}

} // namespace detail

inline constexpr bool kFormatTableCoversEveryEnumerator =
    detail::format_table_has_no_unaccounted_enumerator();
inline constexpr bool kFormatTablesAccountForEachEnumeratorOnce =
    detail::format_tables_account_for_each_enumerator_once();

static_assert(kFormatTableCoversEveryEnumerator,
              "artifact::NumericFormat has an enumerator with neither a kFormatRequirements row "
              "nor a kUncoveredFormatNotes refusal (src/core/arch_caps.h). Give it one: a row if a "
              "kernel consumes it, a named refusal if none does -- and never a row added just to "
              "make this gate pass.");
static_assert(kFormatTablesAccountForEachEnumeratorOnce,
              "the format tables must account for artifact::NumericFormat exactly once each: "
              "(rows + named refusals) == the enumerator count, no duplicate, and no format in "
              "both tables.");

const FormatRequirement* format_requirement(artifact::NumericFormat format) noexcept;

// ---------------------------------------------------------------------------
// What mma.sync.m8n8k4 actually BECOMES on each rung
// ---------------------------------------------------------------------------
//
// THIS BLOCK MOVED HERE FROM src/core/kernel_route.h ON 2026-09-17, and the reason is that
// the artifact-format GATE needs it. The gate and the route selector used to read different
// facts about the same question -- "is the fp16 fallback a real route on this card" -- and
// two tables that answer one question drift. It is a CAPABILITY fact ("does this rung
// execute mma.m8n8k4 in hardware"), so the capability table is its home, and
// kernel_route.h keeps using it through the same names: same namespace, same identifiers,
// nothing renamed.
//
// THIS IS A PER-CHANNEL FACT, NOT A PER-ARCH ONE, and the distinction is the first thing to
// get right. The QPN family reaches the tensor core through exactly ONE channel:
// `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`. A different channel in the same cubin --
// the `nvcuda::wmma` m16n16k16 one used by skinny_nvfp4_wmma / _wmma_ks -- is a DIFFERENT
// route and stays hardware on Ampere (0.5 below). So "sm_80 is emulated" is FALSE as stated;
// "the m8n8k4 channel is emulated on sm_80" is what was measured, and it is what this
// table records.
//
// The QPN W4A16 route asks for LESS than the artifact format's floor: fp16 mma rather than
// kind::mxf4nvf4 or bf16. That is a real route on a rung where the m8n8k4 the QPN kernels
// emit is a HARDWARE instruction. It is not a real route on a rung where ptxas answers the
// same PTX with a software routine, because there the plan named "fp16 mma" would execute on
// the FP16x2 FMA pipe while reading like a tensor-core route.
//
// This table is the missing half of that decision, and it exists because a ptxas rc=0 is NOT
// evidence about hardware: rc=0 proves ptxas accepted the mma and produced something for it,
// never that a tensor core executes it. The difference has to be read out of the SASS.
//
// INDEPENDENTLY CONFIRMED THIS LINE (hand-written PTX + ptxas, both toolkits): the m8n8k4
// asm assembles on sm_70 through sm_120a INCLUDING the three rungs the table has no row for,
// so assembly is not what separates the tiers -- the SASS census below is.
//
// METHOD, so every number below can be re-derived rather than believed:
//   1. forced-instantiation probe over src/ops/linear/qpn/qpn_kernels.cuh (an explicit
//      `template __global__ void ...` definition per kernel, so the inline PTX is not
//      dead-stripped), compiled with `nvcc -cubin -arch=<a>`;
//   2. `nvdisasm -c` on the cubin, and `cuobjdump -sass` as an INDEPENDENT second tool;
//   3. count HMMA.884 (the m8n8k4 hardware form) and the CALL instructions whose OPERAND is
//      `$__internal_N_$__cuda_sm_8x_mma_row_col_f32_f16_f16_f32`.
// THREE COUNTING TRAPS, all of which produced wrong numbers in this file before they were
// measured, so they are named here rather than left to the next reader:
//   (a) THE ROUTINE IS IN THIS CUBIN. It is not an external symbol and not something the
//       device library supplies: it appears as 7 local labels
//       (`$__internal_0_..6_$__cuda_sm_8x_mma_row_col_f32_f16_f16_f32:`) at the END OF EACH
//       KERNEL'S .text SECTION -- so a locator that searches for `.text.<routine-name>` finds
//       NOTHING and reports "0 definitions", which is what this file did first. Locate it by
//       the trailing-colon label. Each body is ~88 instructions with EXACTLY 32 FFMA and
//       HMMA = 0.
//   (b) COUNT CALLS AS INSTRUCTIONS WITH THE ROUTINE AS OPERAND. Counting LINES that merely
//       mention the symbol (which includes the `.weak` / `.type` / `.size` directives and the
//       label itself) over-counts: sm_80 has 252 such lines but 224 actual calls.
//   (c) A SECOND, INDEPENDENT TOOL ON THE SUMS IS MANDATORY, because an empty or mis-anchored
//       counter cannot be told from a true zero. cuobjdump -sass agrees with nvdisasm on every
//       row below (sm_80: HMMA.884 = 0, HFMA2.MMA = 324 in both).
// THE TIE-OUT that makes the emulation story quantitative rather than impressionistic:
//   PTX `mma.sync.aligned.m8n8k4` sites = 224
//   sm_80 CALL instructions to the routine = 224
//   7 routine bodies x 32 FFMA = 224
//   sm_80 FFMA total - sm_70 FFMA total = 350 - 126 = 224
// four independent quantities, all 224.
// ONE THING THAT IS *NOT* EVIDENCE: the HFMA2.MMA these kernels emit. On sm_80 it is 324
// instructions of which only 21 are the `HFMA2.MMA Rd, -RZ, RZ, 0, 0` zeroing idiom and 303
// are real FP16x2 FMAs (`Rd, Ra, Rb, -RZ`); on sm_120a the count is 0 and the simulation is
// HMUL2/HADD2/FFMA shaped instead. So a particular FMA mnemonic is NOT a fingerprint of
// emulation and must not be used as the key -- the CALL to the m8n8k4 routine is the key.
// Raw logs, all under /home/user/scratch/QPN-GRADE/logs/:
//   T1i_final.txt      the FINAL per-arch table, every target measured, both toolkits, plus
//                      the key-folding check (the '-a' and plain forms of each key AGREE on
//                      every counted quantity) and the cross-toolkit check
//   T1g_corrected.txt  the corrected census of the four A2 rows, both tools
//   T1f_reverify.txt   how the routine bodies were located (the mistake it corrects)
//   T1h_hfma2.txt      the HFMA2.MMA operand shapes (arithmetic vs the zeroing idiom)
//   T1j_twotool.txt    the two-tool agreement, EVERY row: 26 rows, 0 disagreements, on
//                      HMMA.884, on the total CALL.REL.NOINC count, and on the calls to the
//                      routine -- nvdisasm by SYMBOL vs cuobjdump by the resolved ADDRESS
//                      matching the routine's label address (two independent computations, so
//                      their agreeing is the cross-check rather than a restatement)
//   T1b_probe.txt      the per-cell probe output for the newly measured targets
// ONE CELL WAS UNAVAILABLE AND IS NAMED RATHER THAN FILLED IN: CUDA 12.8 refuses sm_88 with
// `nvcc fatal : Unsupported gpu architecture 'sm_88'` (rc=1), so the sm_88 row is CUDA 13.3's.
// And sm_87 was measured with 12.8 only. Both rows say so in their own text.
//
// WHICH RUNGS THIS TABLE HAS TO COVER: only the ones the QPN fallback can be reached on,
// which is not the whole ladder. The gate/route arm sits behind `!floor_met && has(Fp16Mma)`,
// and its two formats have different floors, so the reachable set is
//     NVFP4  (floor kind::mxf4nvf4): every rung with fp16 mma and no block-scale bit
//     FP8    (floor Bf16Mma):        every rung with fp16 mma and no bf16 mma
// i.e. {70, 75, 80, 86, 87, 88, 89, 90, 100, 103} -- ten rungs.
// sm_120 / sm_121 are NOT in it: they carry Mxf4Nvfp4BlockScale, so NVFP4 meets the floor and
// the arm is short-circuited at the `floor_met` branch before this table is consulted. They
// therefore have NO ROW HERE, and that is deliberate: a tier for a card the arm never asks
// about is a claim nothing exercises. If a future format makes the arm reachable on 120, the
// missing row makes it refuse with the UNMEASURED reason below -- fail-closed, not guessed.
//
// AND THE KEY FOLDS THE '-a' SUFFIX. `arch_rung`'s own comment above says the
// arch-accelerated suffix is NOT observable at runtime, so key 90 answers sm_90 AND sm_90a,
// key 120 answers sm_120 and sm_120a. That is why each row carries `measured_target`: the
// exact -arch= its census came from, so a row is never read as a measurement of a target it
// did not measure. Where the two targets of a key were BOTH measured they must agree, and the
// measured_target field says so.
//
// A rung of kArchLadder with no row here has an UNMEASURED lowering and the arm refuses it by
// naming the missing measurement -- the same rule the ladder follows for a missing capability.

enum class QpnMmaLowering : std::uint8_t {
    HardwareMma884 = 0, // mma.sync.m8n8k4 lowers to HMMA.884: a genuine tensor-core route
    EmulatedFp16Pipe,   // mma.sync.m8n8k4 lowers to a CALL into a software FFMA routine
};

struct QpnMmaRung {
    int sm;                           // the ladder key; the '-a' suffix is NOT in it
    std::string_view measured_target; // the exact -arch= this census was produced for
    QpnMmaLowering lowering;
    std::string_view evidence;        // the measured numbers, with the toolkit
};

inline constexpr QpnMmaRung kQpnMmaRungs[] = {
    {70, "sm_70", QpnMmaLowering::HardwareMma884,
     "CUDA 12.8: the m8n8k4 channel emits 1024 HMMA.884.F32.F32 -- 224 mma sites x 4 STEPs = "
     "896 in the seven inline-asm kernels, plus 8 x 16 = 128 from the two nvcuda::wmma "
     "kernels' m16n16k16 channel -- and 0 CALL instructions targeting the emulation routine, "
     "0 HFMA2.MMA"},
    {75, "sm_75", QpnMmaLowering::HardwareMma884,
     "CUDA 12.8: 896 HMMA.884.F32.F32 (224 sites x 4 STEPs) plus 32 HMMA.1688.F32 (the wmma "
     "channel is m16n8k8 on Turing, and IT IS HARDWARE TOO), 0 CALL instructions targeting "
     "the emulation routine, 0 HFMA2.MMA"},
    {80, "sm_80", QpnMmaLowering::EmulatedFp16Pipe,
     "CUDA 12.8: 0 hardware tensor ops in the m8n8k4 channel. All 224 mma sites become a CALL "
     "to one of 7 in-cubin routines ($__internal_0_..6_$__cuda_sm_8x_mma_row_col_f32_f16_f16"
     "_f32, ~88 instructions and exactly 32 FFMA each, HMMA = 0), and the cubin's FFMA total "
     "rises by exactly 224 over sm_70 (126 -> 350). The 16 HMMA.16816.F32 that remain belong "
     "to the two nvcuda::wmma kernels -- the m16n16k16 channel, a DIFFERENT route, which "
     "Ampere does execute in hardware"},
    {90, "sm_90 and sm_90a (both measured, both toolkits; they AGREE)", QpnMmaLowering::EmulatedFp16Pipe,
     "CUDA 12.8 and 13.3: same shape as sm_80 -- 224 CALL instructions to the same 7 in-cubin "
     "routines (32 FFMA each, HMMA = 0), FFMA total 350 (126 + 224), 0 HMMA.884, and the only "
     "tensor ops in the cubin are the 16 HMMA.16816.F32 of the two nvcuda::wmma kernels. The "
     "plain and the '-a' target agree on every counted quantity, which is what lets ONE row "
     "answer a key that folds both"},
    {86, "sm_86 (both toolkits, AGREE)", QpnMmaLowering::EmulatedFp16Pipe,
     "CUDA 12.8 and 13.3: 0 HMMA.884, 224 CALL instructions to the 7 in-cubin FFMA routines "
     "(32 FFMA each), FFMA total 350 (126 + 224), 16 HMMA.16816.F32 in the two nvcuda::wmma "
     "kernels. NOTE HFMA2.MMA = 0 here while sm_80 shows 324 -- the same operation is spelled "
     "plain HFMA2 (136 of them) on this target, which is exactly why the table is keyed on the "
     "CALL and not on an FMA mnemonic"},
    {87, "sm_87", QpnMmaLowering::EmulatedFp16Pipe,
     "CUDA 12.8: 0 HMMA.884, 224 CALL instructions to the 7 in-cubin FFMA routines (32 FFMA "
     "each), FFMA total 350 (126 + 224), 16 HMMA.16816.F32 in the two nvcuda::wmma kernels, "
     "325 HFMA2.MMA. CUDA 13.3 WAS NOT RUN for sm_87 -- it measures sm_88 instead; the sm_86 "
     "and sm_89 cells bracket it on both toolkits and agree, but this row's numbers are 12.8's"},
    {88, "sm_88 (CUDA 13.3 ONLY)", QpnMmaLowering::EmulatedFp16Pipe,
     "CUDA 13.3 (12.8 CANNOT BUILD IT -- `nvcc fatal : Unsupported gpu architecture 'sm_88'`, "
     "rc=1): 0 HMMA.884, 224 CALL instructions to the 7 in-cubin FFMA routines (32 FFMA each), "
     "FFMA total 350 (126 + 224), 16 HMMA.16816.F32 in the two nvcuda::wmma kernels, "
     "HFMA2.MMA = 0 (plain HFMA2 = 136)"},
    {89, "sm_89 (both toolkits, AGREE)", QpnMmaLowering::EmulatedFp16Pipe,
     "CUDA 12.8 and 13.3: 0 HMMA.884, 224 CALL instructions to the 7 in-cubin FFMA routines "
     "(32 FFMA each), FFMA total 350 (126 + 224), 16 HMMA.16816.F32 in the two nvcuda::wmma "
     "kernels, HFMA2.MMA = 0 (plain HFMA2 = 136); this row matters because fp8 on sm_89 is the "
     "A16 route, so the QPN arm is only reached here through nvfp4"},
    {100, "sm_100 and sm_100a (both measured, they AGREE)", QpnMmaLowering::EmulatedFp16Pipe,
     "CUDA 13.3: 0 HMMA.884, 224 CALL instructions to the 7 in-cubin FFMA routines (32 FFMA "
     "each), FFMA total 366, 16 HMMA.16816.F32 in the two nvcuda::wmma kernels, HFMA2.MMA = 0 "
     "(plain HFMA2 = 383). The toolkit change moves the FFMA total (350 -> 366) and nothing "
     "that decides the tier: HMMA.884 is 0 and the routine calls are 224 on both toolkits"},
    {103, "sm_103 and sm_103a (both measured, they AGREE)", QpnMmaLowering::EmulatedFp16Pipe,
     "CUDA 13.3: identical to sm_100 on every counted quantity -- 0 HMMA.884, 224 CALL "
     "instructions to the 7 in-cubin FFMA routines, FFMA total 366, 16 HMMA.16816.F32 in the "
     "two nvcuda::wmma kernels, HFMA2.MMA = 0"},
};

inline constexpr std::size_t kQpnMmaRungCount = sizeof(kQpnMmaRungs) / sizeof(kQpnMmaRungs[0]);

// Exact match only, like arch_rung(). nullptr means "not measured", never "assume the
// neighbour's" and never "the card is fine".
[[nodiscard]] inline const QpnMmaRung* qpn_mma_rung(int sm) noexcept {
    for (const QpnMmaRung& rung : kQpnMmaRungs) {
        if (rung.sm == sm) { return &rung; }
    }
    return nullptr;
}

[[nodiscard]] inline std::string_view qpn_mma_lowering_name(QpnMmaLowering lowering) noexcept {
    switch (lowering) {
    case QpnMmaLowering::HardwareMma884: return "hardware mma.m8n8k4 (HMMA.884)";
    case QpnMmaLowering::EmulatedFp16Pipe: return "emulated on the FP16x2 FMA pipe";
    }
    return "unknown-lowering";
}

// A rung is reachable when the QPN arm's own guard admits it: fp16 mma present, and at least
// one of the arm's two formats missing its floor. Written as a function so a test can compute
// the reachable set instead of restating it, and so a ladder change moves it automatically.
[[nodiscard]] inline bool qpn_arm_reachable(const ArchRung& rung) noexcept {
    if (!has_cap(rung.caps, Cap::Fp16Mma)) { return false; }
    const bool nvfp4_short = !covers(rung.caps, Cap::Mxf4Nvfp4BlockScale);
    const bool fp8_short   = !covers(rung.caps, Cap::Bf16Mma);
    return nvfp4_short || fp8_short;
}

// ---------------------------------------------------------------------------
// Is the fp16 fallback EXECUTABLE on this rung, in this build?
// ---------------------------------------------------------------------------
//
// The one predicate the gate and the route selector share, so the two cannot disagree about
// whether a pre-Ampere card has a fallback. Four clauses, and every one of them is either a
// BUILD fact or a MEASURED fact -- there is nothing here a caller can assume:
//
//   1. `qpn_in_build`      -- the QPN sources are compiled into this build. This is the fact
//      that used to make the route a phantom, and it is deliberately a build fact rather than
//      a mood, which is why it is not a preprocessor switch a caller may flip: see
//      src/CMakeLists.txt, where the source line and the definition are written together, and
//      tests/test_qpn_build_fact.cpp, which references the kernel symbol so that a definition
//      without the kernel is a LINK error and not a passing test.
//   2. the format HAS a fallback in the table above (Cap::None means it does not).
//   3. the rung covers the fallback's floor (Cap::Fp16Mma).
//   4. the rung's MEASURED lowering of the fallback's tensor-core channel is HardwareMma884.
//      Fail-closed on an unmeasured rung, which is why 110 - whose capability row was
//      withdrawn - refuses here too instead of inheriting a neighbour's tier.
//
// `sm` is the compute capability the caller wants an answer FOR, so a simulator can pass the
// simulated rung in and every clause above is evaluated against that rung, not the card's.
[[nodiscard]] inline bool fp16_fallback_executable(int sm, artifact::NumericFormat format,
                                                   bool qpn_in_build) noexcept {
    if (!qpn_in_build) { return false; }
    const FormatRequirement* requirement = format_requirement(format);
    if (requirement == nullptr || requirement->fallback_required == Cap::None) { return false; }
    const ArchRung* rung = arch_rung(sm);
    if (rung == nullptr) { return false; }
    if (!covers(rung->caps, requirement->fallback_required)) { return false; }
    const QpnMmaRung* channel = qpn_mma_rung(sm);
    if (channel == nullptr) { return false; }
    return channel->lowering == QpnMmaLowering::HardwareMma884;
}

// Whether the QPN W4A16 family is compiled into THIS binary. Defined by src/CMakeLists.txt
// on the same lines that add ops/linear/qpn/qpn_host.cu to a target; a hand-written -D on a
// shipping build is caught by the link-time tie in tests/test_qpn_build_fact.cpp rather than
// believed.
#ifdef NINFER_HAVE_QPN
inline constexpr bool kQpnInBuild = true;
#else
inline constexpr bool kQpnInBuild = false;
#endif

// ---------------------------------------------------------------------------
// Build arch list
// ---------------------------------------------------------------------------

// The arch list the binary was compiled for, when the build exports it
// (NINFER_BUILD_CUDA_ARCHES, set from CMAKE_CUDA_ARCHITECTURES). Empty when the build does
// not export it. A compile-time '-a' suffix (sm_120a) is NOT observable through
// cudaDeviceProp, which reports only 12.0, so this string is the ONLY way the runtime can
// name the arch-accelerated target of the binary it is running in.
constexpr std::string_view build_arch_list() noexcept;

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

enum class Verdict {
    Supported,    // every format's floor is met
    Unsupported,  // at least one format's floor is not met on this compute capability
    UnknownArch,  // the compute capability is not in the ladder
    UnknownFormat // an artifact format has no row here (a table defect, not a GPU fact)
};

struct FormatGap {
    artifact::NumericFormat format;
    Cap required;
    Cap missing;
    std::string_view kernel_evidence;
};

// A format whose declared floor this rung does NOT meet, but for which the table names a
// lower-floor kernel that this build contains and this rung executes in hardware. Recorded
// rather than silently skipped, so that "the artifact loads on a pre-Ampere card" is always
// accompanied by WHICH fallback made that true and WHICH capability was missing. The verdict
// stays Supported (the artifact really can execute here), and this list is what the caller
// prints so the operator is never told a V100 meets an fp4 floor it does not meet.
struct FormatFallback {
    artifact::NumericFormat format;
    Cap primary_missing;                  // the floor this rung does not cover
    Cap fallback_used;                    // the floor it does cover instead
    std::string_view fallback_kernel;     // the kernel file that consumes the same bytes
};

struct CapabilityReport {
    Verdict verdict = Verdict::Supported;
    int sm           = 0;
    std::vector<FormatGap> gaps;
    std::vector<FormatFallback> fallbacks;

    [[nodiscard]] bool ok() const noexcept { return verdict == Verdict::Supported; }
};

// Pure function over (compute capability, artifact formats). An empty format list is
// Supported: an artifact whose objects are all address-only carries no tensor-core floor.
//
// `qpn_in_build` is the build fact from kQpnInBuild by default, and it is a PARAMETER for the
// same reason `sm` is: it lets one test binary exercise the build-with-QPN and
// build-without-QPN worlds without a rebuild, which is how the fail-closed half gets a red
// control instead of an assertion about a binary nobody can produce.
CapabilityReport evaluate_artifact_formats(int sm,
                                           std::span<const artifact::NumericFormat> formats,
                                           bool qpn_in_build = kQpnInBuild);

// Distinct tensor formats in first-seen order. ResourceDescriptor objects carry no
// numeric format and are skipped; duplicates collapse.
std::vector<artifact::NumericFormat>
artifact_formats(const std::vector<artifact::ObjectDescriptor>& objects);

// Operator-facing text: the GPU, the binary's arch list, the missing capability with the
// kernel that needs it, and the routes that would work on this card. Empty when ok().
std::string render_capability_report(const CapabilityReport& report,
                                     std::string_view artifact_identity);

// The same tables, rendered for the ONE question that needs neither a model nor a device: what
// was this binary COMPILED to run? The arch list the build published, the ladder this build
// ships, and the tensor-core floor of every weight format it can bind -- the three facts
// render_capability_report() consults, with each row's citation kept.
//
// It exists because the operator could not ask that question at all: render_capability_report()
// is reached only through artifact load (src/targets/registry.cpp, construct_target ->
// require_artifact_formats_supported), so on a machine with no artifact the arch list this
// binary was compiled for was unreadable from any surface. The arch line is rendered by
// render_build_arch_line(), the SAME code the refusal uses, so neither can drift from the other.
// Nothing here is fabricated and nothing here is a probe; see the definition for what is
// deliberately NOT printed.
[[nodiscard]] std::string render_build_capability_surface();

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

inline const ArchRung* arch_rung(int sm) noexcept {
    for (const ArchRung& rung : kArchLadder) {
        if (rung.sm == sm) { return &rung; }
    }
    return nullptr;
}

inline const FormatRequirement* format_requirement(artifact::NumericFormat format) noexcept {
    return find_format_requirement(format);
}

// The named refusal for a format this tree declares but has no kernel for on any rung. nullptr
// for a format that HAS a row, and nullptr for a format in neither table -- which is the
// remaining Verdict::UnknownFormat case, a genuine table defect (the static_asserts above make it
// unreachable for a real enumerator).
inline const UncoveredFormatNote* uncovered_format_note(artifact::NumericFormat format) noexcept {
    return find_uncovered_format_note(format);
}

inline constexpr std::string_view build_arch_list() noexcept {
// BOTH SPELLINGS, and the one this build actually writes comes first. The reason is
// measured and is in the report: this function read ..._ARCHES while src/CMakeLists.txt:51
// defines ..._ARCHS, so the value was the empty string on every build and the never-probed
// fallback below had nothing to name.
#if defined(NINFER_BUILD_CUDA_ARCHS)
    return std::string_view(NINFER_BUILD_CUDA_ARCHS);
#elif defined(NINFER_BUILD_CUDA_ARCHES)
    return std::string_view(NINFER_BUILD_CUDA_ARCHES);
#else
    return std::string_view{};
#endif
}

// ---------------------------------------------------------------------------
// Implementation of the measured decision
// ---------------------------------------------------------------------------

// The build default rung, READ OUT OF THE BUILD'S OWN RECORD and never invented.
// NINFER_BUILD_CUDA_ARCHES is the string CMake put there (e.g. "120a", or a list "52;70;75").
// Single target -> that target. MULTI target -> there is no single "default target", so the
// LOWEST number in the list is taken and the WHOLE string is quoted in `why`, so the reader
// sees the choice instead of inferring it. Lowest is the conservative side of a conservative
// fallback: fewer claimed bits, and the binary does carry an image for that arch.
[[nodiscard]] inline constexpr int build_default_sm() noexcept {
    const std::string_view list = build_arch_list();
    int best      = 0;
    bool found    = false;
    std::size_t i = 0;
    while (i < list.size()) {
        if (list[i] < '0' || list[i] > '9') { ++i; continue; }
        int value = 0;
        while (i < list.size() && list[i] >= '0' && list[i] <= '9') {
            value = value * 10 + (list[i] - '0');
            ++i;
        }
        // A trailing 'a'/'f' belongs to the target SPELLING, not to the number; cudaDeviceProp
        // has no such suffix either, so both sides compare the same integer.
        if (!found || value < best) { best = value; found = true; }
    }
    return found ? best : 0;
}

// `[]` for none, so an EMPTY class reads as empty and not as a missing field.
[[nodiscard]] inline std::string cap_list_text(Cap set) {
    if (set == Cap::None) { return "none"; }
    std::string out;
    for (std::size_t i = 0; i < 32; ++i) {
        const Cap single = static_cast<Cap>(1u << i);
        if (!has_cap(kAllCapBits, single)) { continue; }
        if (!has_cap(set, single)) { continue; }
        if (!out.empty()) { out += ", "; }
        out.append(cap_name(single));
    }
    return out.empty() ? std::string("none") : out;
}

// The measurement, read off the engine's OWN probe report. Nothing here re-derives a
// predicate: every bit is a probe's answer, and a bit no probe answers is Unmeasurable.
[[nodiscard]] inline MeasuredCaps measure_from_report(const ::ninfer::CapabilityReport& report,
                                                     MeasurementKey key) {
    MeasuredCaps out;
    out.key      = key;
    out.probed   = report.probed_anything();
    out.baseline = report.status(::ninfer::DeviceCapability::KernelImage);

    Cap covered = Cap::None;
    for (const CapProbe& entry : kCapProbes) {
        covered = covered | entry.bit;
        switch (report.status(entry.probe)) {
        case ::ninfer::ProbeStatus::Supported:
            out.proved = out.proved | entry.bit;
            break;
        case ::ninfer::ProbeStatus::DeviceHasNoImage:
        case ::ninfer::ProbeStatus::LaunchFailed:
        case ::ninfer::ProbeStatus::NumericMismatch:
            // A probe that RAN and did not agree. This is the verdict only a device can give,
            // and the only one allowed to remove a bit the ladder claims.
            out.refuted = out.refuted | entry.bit;
            break;
        case ::ninfer::ProbeStatus::NotInBuild:
        case ::ninfer::ProbeStatus::NotProbed:
            // No probe in this build. NOT evidence about the device.
            out.unmeasurable = out.unmeasurable | entry.bit;
            break;
        }
    }
    out.unmeasurable = out.unmeasurable | (kAllCapBits & ~covered);

    out.provenance = "device " + std::to_string(key.device_id) + ", build arches \"" +
                     std::string(key.build_arches) + "\": proved [" + cap_list_text(out.proved) +
                     "], refuted [" + cap_list_text(out.refuted) + "], unmeasurable [" +
                     cap_list_text(out.unmeasurable) + "]";
    return out;
}

// ---------------------------------------------------------------------------
// The publish / read pair. THE WRITER'S KEY IS MANDATORY AT COMPILE TIME.
// ---------------------------------------------------------------------------
// THIS OVERLOAD IS DELETED, so `publish_measured_caps(caps)` is not a runtime mistake a
// reviewer has to catch -- it is a compile error in the writer's own TU, which is where the
// missing key is cheap to fix. The deleted overload is the whole mechanism: there is no
// defaulted-key overload, no key-less entry point, and no way to spell the call short.
void publish_measured_caps(MeasuredCaps caps) = delete;

namespace detail {

struct MeasurementSlot {
    MeasurementKey key{};
    MeasuredCaps caps{};
    bool filled = false;
};

// This build binds at most 2 devices (device.cu's ExecutionContext) and the probe cache is
// keyed per device id; 8 slots is above any device count this build can bind, and the registry
// is a static array, so a route decision never allocates.
inline constexpr std::size_t kMeasurementSlots = 8;

inline MeasurementSlot* measurement_slots() noexcept {
    static MeasurementSlot slots[kMeasurementSlots]{};
    return slots;
}

} // namespace detail

inline void publish_measured_caps(MeasurementKey key, MeasuredCaps caps) {
    caps.key = key;
    detail::MeasurementSlot* free_slot = nullptr;
    for (std::size_t i = 0; i < detail::kMeasurementSlots; ++i) {
        detail::MeasurementSlot& slot = detail::measurement_slots()[i];
        if (slot.filled && slot.key == key) {
            slot.caps = std::move(caps);
            return;
        }
        if (!slot.filled && free_slot == nullptr) { free_slot = &slot; }
    }
    if (free_slot == nullptr) {
        // Deliberately silent-and-harmless: the route layer then sees a never-probed device and
        // falls back to the named build default, which is the honest answer for "I could not
        // file this measurement". Dropping the WRITE is not the same as inventing a read.
        return;
    }
    free_slot->key    = key;
    free_slot->caps   = std::move(caps);
    free_slot->filled = true;
}

// The read side. A miss is NOT an error and NOT a default: it is the never-probed state, and
// it comes back with `probed == false`, so no caller can mistake it for a measurement.
[[nodiscard]] inline MeasuredCaps measured_caps_for(MeasurementKey key) {
    for (std::size_t i = 0; i < detail::kMeasurementSlots; ++i) {
        const detail::MeasurementSlot& slot = detail::measurement_slots()[i];
        if (slot.filled && slot.key == key) { return slot.caps; }
    }
    MeasuredCaps never;
    never.key = key;
    return never;
}

[[nodiscard]] inline MeasuredRow resolve_measured_row(int sm, const MeasuredCaps& measured,
                                                      std::string_view required_by) {
    MeasuredRow out;

    if (!measured.probed) {
        const int fallback = build_default_sm();
        out.row  = (fallback > 0) ? arch_rung(fallback) : nullptr;
        if (out.row != nullptr) { out.narrowed = *out.row; }
        out.why  = std::string("NEVER PROBED for ") + std::string(required_by) + ": device sm_" +
                   std::to_string(sm) +
                   " has not been measured by this process, so this answer is the BUILD DEFAULT "
                   "ROW sm_" + std::to_string(fallback) +
                   " (the lowest arch in this binary's CMAKE_CUDA_ARCHITECTURES=\"" +
                   std::string(build_arch_list()) +
                   "\"), named here AS A FALLBACK and not as a measurement. Nothing is assumed "
                   "about sm_" + std::to_string(sm) +
                   " and nothing is refused on its behalf. The probe was not run because a route "
                   "decision is a pure function and must not run one; to get a measured answer, "
                   "call probe_device_capabilities() on the thread bound to this device first.";
        if (out.row == nullptr) {
            out.why += " THIS BINARY'S OWN DEFAULT ARCH ALSO HAS NO LADDER ROW, so no answer can "
                       "come from the table at all.";
        }
        return out;
    }

    out.row = arch_rung(sm);
    if (out.row == nullptr) {
        out.why = std::string("MEASURED, BUT NO ROW: ") + std::string(required_by) +
                  " was given a measurement for device sm_" + std::to_string(sm) +
                  ", and kArchLadder has no row for that number, so there is no named set to "
                  "narrow. A measurement can only REMOVE bits from a row; it cannot invent one, "
                  "because the row's other job is to name a kernel in THIS binary.";
        if (!measured.provenance.empty()) { out.why += " " + measured.provenance; }
        return out;
    }

    out.refuted       = measured.refuted & out.row->caps;
    out.unmeasurable  = measured.unmeasurable & out.row->caps;
    out.narrowed      = *out.row;
    out.narrowed.caps = out.row->caps & ~out.refuted;
    out.narrowed_by_probe = out.refuted != Cap::None;

    out.why = std::string(required_by) + ": measured on this device -- " + measured.provenance +
              ". Row sm_" + std::to_string(out.row->sm) + " (" + std::string(out.row->label) +
              ") claims [" + cap_list_text(out.row->caps) + "]";
    if (out.narrowed_by_probe) {
        out.why += "; the probe REFUTED [" + cap_list_text(out.refuted) +
                   "], so the effective set is [" + cap_list_text(out.narrowed.caps) + "]";
    } else {
        out.why += "; nothing was refuted, so the row stands as claimed";
    }
    if (out.unmeasurable != Cap::None) {
        out.why += ". NAMED ABSENCE: [" + cap_list_text(out.unmeasurable) +
                   "] is claimed by the row but NO PROBE IN THIS BUILD CAN ANSWER IT, so it is "
                   "left standing on the row's evidence and is explicitly NOT reported as "
                   "measured -- a probe body compiled out of this arch is not evidence that the "
                   "device lacks it.";
    }
    return out;
}


[[nodiscard]] inline MeasurementKey current_measurement_key(int device_id) noexcept {
    return MeasurementKey{device_id, build_arch_list()};
}

inline std::string_view cap_name(Cap single) noexcept {
    switch (single) {
    case Cap::None: return "none";
    case Cap::Fp16Mma: return "fp16 tensor-core mma";
    case Cap::Int8Mma: return "int8 tensor-core mma";
    case Cap::Bf16Mma: return "bf16 tensor-core mma";
    case Cap::Fp8E4m3MmaPlain: return "fp8 e4m3 tensor-core mma (plain form)";
    case Cap::Fp8F8f6f4KindMma: return "fp8/fp6/fp4 tensor-core mma (kind::f8f6f4)";
    case Cap::Mxf4Nvfp4BlockScale: return "nvfp4 e2m1 block-scaled mma (kind::mxf4nvf4)";
    }
    return "unknown capability";
}

inline std::vector<artifact::NumericFormat>
artifact_formats(const std::vector<artifact::ObjectDescriptor>& objects) {
    std::vector<artifact::NumericFormat> formats;
    for (const artifact::ObjectDescriptor& object : objects) {
        const auto* tensor = std::get_if<artifact::TensorDescriptor>(&object);
        if (tensor == nullptr) { continue; }
        bool seen = false;
        for (const artifact::NumericFormat known : formats) {
            if (known == tensor->format) {
                seen = true;
                break;
            }
        }
        if (!seen) { formats.push_back(tensor->format); }
    }
    return formats;
}

inline CapabilityReport evaluate_artifact_formats(
    int sm, std::span<const artifact::NumericFormat> formats, bool qpn_in_build) {
    CapabilityReport report;
    report.sm = sm;

    // -----------------------------------------------------------------------
    // THE NOTE TABLE IS CONSULTED BEFORE THE RUNG. THIS ORDER IS THE FIX.
    // -----------------------------------------------------------------------
    // WHAT THIS ORDER REPLACES, and it was measured rather than argued. The note lookup below
    // used to be reachable ONLY through the loop that follows the `arch_rung(sm) == nullptr`
    // early return. So on a compute capability with no ladder row -- Maxwell sm_52 / sm_53,
    // Pascal sm_61, and the withdrawn sm_110 -- no kUncoveredFormatNotes row was ever
    // consulted, and the only thing the gate put on stderr was the format-agnostic ladder
    // warning from unknown_arch_warning(), which names NO format. The gate was silent exactly
    // where it had a named refusal to give.
    //
    // MEASURED 2026-09-19 (dl/e8loud, this tree's own note row FP8_E4M3FN_ROW_F32S driven
    // through require_artifact_formats_supported, one sm per process because the UnknownArch
    // warning is once-per-process):
    //   before this reorder: sm_52/53/61/110 -> UnknownArch, gaps={}, RETURNED WITHOUT THROW,
    //     stderr 481 B / 483 B and ZERO occurrences of the format's name;
    //     sm_70/75/80/86/120 -> Unsupported, one gap, THREW, 4271..4786 B naming the format.
    //   after this reorder:  sm_52/53/61/110 -> Unsupported, one gap, THREW, naming the format.
    // The mechanism was always alive; the ordering hid it.
    //
    // WHY A REORDER IS THE WHOLE FIX AND NOT A NEW REFUSAL. Each note row already asserts, in
    // its own text, that no compute capability can change its answer -- the missing piece is a
    // KERNEL, not a capability. Asking the note table before the rung is what those rows already
    // meant; asking it after was the defect. Nothing is ADMITTED by this move -- it can only
    // turn an UnknownArch warning into a NAMED refusal, never the other way round -- so the
    // access matrix's widening set stays empty. Re-measured, not asserted: dl/e8loud.
    //
    // ONLY THE NOTE TABLE MOVES. A format that HAS a kFormatRequirements row is deliberately
    // NOT decided here: its floor is only meaningful against a rung's capability set, so on a
    // rung-less compute capability it must keep taking the UnknownArch warning path. That is
    // what keeps every cell of the access matrix that is not about a note row unchanged.
    //
    // The same lookup still stands inside the loop below. It is now unreachable by
    // construction (this pass returns as soon as a note exists) and is left in place on
    // purpose: it makes this edit an INSERTION that a reviewer can read as one hunk, instead of
    // a rewrite of the loop whose equivalence would have to be re-established.
    for (const artifact::NumericFormat format : formats) {
        if (format_requirement(format) != nullptr) { continue; }
        const UncoveredFormatNote* uncovered = uncovered_format_note(format);
        if (uncovered == nullptr) { continue; }
        report.verdict = Verdict::Unsupported;
        report.gaps.push_back(FormatGap{format, uncovered->would_need,
                                        uncovered->would_need, uncovered->reason});
    }
    if (report.verdict == Verdict::Unsupported) { return report; }

    const ArchRung* rung = arch_rung(sm);
    if (rung == nullptr) {
        report.verdict = Verdict::UnknownArch;
        return report;
    }
    for (const artifact::NumericFormat format : formats) {
        const FormatRequirement* requirement = format_requirement(format);
        if (requirement == nullptr) {
            // No floor row. There are TWO different reasons and they must not read alike: a
            // format this tree declares and deliberately has no kernel for (an evidence-backed
            // REFUSAL that names the missing piece, on every rung), or a format nobody has
            // accounted for (a TABLE defect). Reporting the first as the second is what sent the
            // operator to add the one row that must not exist.
            const UncoveredFormatNote* uncovered = uncovered_format_note(format);
            if (uncovered != nullptr) {
                // Refused before the rung is consulted: the missing piece is a KERNEL, not a
                // capability, so no compute capability can change this answer.
                report.verdict = Verdict::Unsupported;
                report.gaps.push_back(FormatGap{format, uncovered->would_need,
                                                uncovered->would_need, uncovered->reason});
                continue;
            }
            // Do not overwrite a refusal this loop has already decided: the verdict must not
            // depend on the ORDER the artifact's formats are visited in.
            if (report.verdict != Verdict::Unsupported) {
                report.verdict = Verdict::UnknownFormat;
            }
            report.gaps.push_back(FormatGap{format, Cap::None, Cap::None, {}});
            continue;
        }
        if (requirement->required == Cap::None || covers(rung->caps, requirement->required)) {
            continue;
        }
        // The floor is not met. BEFORE refusing, ask whether the table names a lower-floor
        // kernel that this build contains and this rung executes in hardware. That question
        // is fp16_fallback_executable(), and it is the same predicate the route selector
        // uses -- the gate and the route read ONE fact, so "the artifact may load" and
        // "there is a kernel to run" cannot come apart on the same card.
        if (fp16_fallback_executable(sm, format, qpn_in_build)) {
            report.fallbacks.push_back(FormatFallback{
                format, requirement->required & ~rung->caps, requirement->fallback_required,
                requirement->fallback_kernel_evidence});
            continue;
        }
        report.verdict = Verdict::Unsupported;
        report.gaps.push_back(FormatGap{format, requirement->required,
                                        requirement->required & ~rung->caps,
                                        requirement->kernel_evidence});
    }
    if (report.verdict != Verdict::Unsupported && !report.gaps.empty()) {
        report.verdict = Verdict::UnknownFormat;
    }
    return report;
}

// THE ONE RENDERING OF THE BUILD'S ARCH LIST. Two surfaces print it -- the refusal below and
// render_build_capability_surface() further down -- so the <unreported> sentence has exactly one
// home and the two cannot disagree about what this binary was compiled for.
[[nodiscard]] inline std::string render_build_arch_line() {
    std::string out = "  binary     : compiled for ";
    const std::string_view arches = build_arch_list();
    if (arches.empty()) {
        out += "<unreported> (this translation unit was not given the build's arch list. "
               "src/CMakeLists.txt:49-51 scopes the arch-list macro to core/device_probe.cu, so "
               "the list reaches the engine through DeviceFacts::build_architectures and not "
               "here. To name it in THIS report, publish the macro to this TU's target from "
               "CMAKE_CUDA_ARCHITECTURES -- and note that a rebuild alone does NOT change this "
               "line, because the macro is not defined for this translation unit)\n";
    } else {
        out.append(arches);
        out += "\n";
    }
    return out;
}

// A capability set as one line, in the table's own order. Cap::None has its own wording HERE
// and callers must not add a second one: this function is the single home of both the order and
// that sentence, so the ladder print and the refusal print cannot drift.
[[nodiscard]] inline std::string render_cap_set(Cap set) {
    if (set == Cap::None) { return "<no tensor core>"; }
    std::string out;
    bool first = true;
    for (const Cap single : {Cap::Fp16Mma, Cap::Int8Mma, Cap::Bf16Mma, Cap::Fp8E4m3MmaPlain,
                             Cap::Fp8F8f6f4KindMma, Cap::Mxf4Nvfp4BlockScale}) {
        if (!has_cap(set, single)) { continue; }
        if (!first) { out += ", "; }
        out += std::string(cap_name(single));
        first = false;
    }
    return out;
}

inline std::string render_capability_report(const CapabilityReport& report,
                                           std::string_view artifact_identity) {
    if (report.ok()) { return std::string{}; }
    const ArchRung* rung = arch_rung(report.sm);
    std::string out;
    out += "ninfer: cannot run this artifact on this GPU.\n";
    out += "  artifact   : ";
    out.append(artifact_identity);
    out += "\n  GPU        : ";
    if (rung != nullptr) {
        out += "sm_";
        out += std::to_string(report.sm);
        out += " (";
        out.append(rung->label);
        out += ", e.g. ";
        out.append(rung->cards);
        out += ")";
    } else {
        out += "compute capability ";
        out += std::to_string(report.sm / 10);
        out += ".";
        out += std::to_string(report.sm % 10);
        out += " -> sm_";
        out += std::to_string(report.sm);
        out += ", which is not in the capability ladder (src/core/arch_caps.h)";
    }
    out += "\n";
    out += render_build_arch_line();

    if (report.verdict == Verdict::UnknownArch) {
        out += "\n  no route is known for this compute capability. Add a row to kArchLadder "
               "with the card's measured tensor-core set before claiming support; do not "
               "assume a nearby row's set.\n";
        return out;
    }
    if (report.verdict == Verdict::UnknownFormat) {
        out += "\n  an artifact object carries a numeric format this build has no capability "
               "row for (src/core/arch_caps.h, kFormatRequirements).\n";
        for (const FormatGap& gap : report.gaps) {
            out += "    - ";
            out += std::string(artifact::format_name(gap.format));
            out += "\n";
        }
        return out;
    }

    out += "\n  unmet requirement(s):\n";
    for (const FormatGap& gap : report.gaps) {
        out += "    - weight format ";
        out += std::string(artifact::format_name(gap.format));
        out += " needs ";
        out += std::string(cap_name(gap.required));
        out += "\n      kernel floor : ";
        out.append(gap.kernel_evidence);
        out += "\n";
    }
    if (rung != nullptr) {
        out += "\n  this card has : ";
        out += render_cap_set(rung->caps);
        out += "\n  route        : ";
        out.append(rung->route);
        out += "\n";
    }
    out += "\n  what to do:\n";
    out += "    - recreate the weight in a format this card executes (tools/convert/, "
           "requantization matrix keyed on this table), or\n";
    out += "    - run this artifact on a card whose capability set covers the requirement "
           "above, or\n";
    out += "    - add the missing kernel route to the tree if the card can execute one "
           "(tools/archkit/_GPU_MATRIX.md is the route table).\n";
    return out;
}

// ---------------------------------------------------------------------------
// The build surface: the same tables, with no model and no device
// ---------------------------------------------------------------------------

// Sibling of render_capability_report() for the one question that needs no artifact: what was
// this binary COMPILED to run? The measured gap it closes is recorded at
// dl/archdiverge/REPORT.md: the refusal report is reached only through artifact load, so on a
// machine with no artifact the arch list was unreadable from every surface, and `ninfer` had 66
// flags and zero CLI lines naming a capability, a device fact or an arch list.
//
// WHAT IS DELIBERATELY NOT PRINTED HERE.
//   * kQpnInBuild. It is defined by the ninfer_qpn_build_fact INTERFACE target for some
//     translation units and not others (#ifdef NINFER_HAVE_QPN above), so a CLI translation unit
//     would print a DIFFERENT value from the one src/targets/registry.cpp acts on. Printing a
//     per-TU build fact from a front end is the exact two-body trap this header's
//     build_arch_list() is being fixed for, so this surface does not add another one; it names
//     the fact and where it decides, and evaluates nothing.
//   * Any statement about the card in this machine. There is no device query on this path.
//   * Any claim that a ladder rung is a ROUTE. A rung is the card's capability set plus a route
//     TEXT; whether this build has an image for that sm is recorded by the arch line alone, and
//     whether an artifact may load on it is decided at load time by the same tables.
[[nodiscard]] inline std::string render_build_capability_surface() {
    std::string out;
    out += "ninfer: build capability surface (no model, no artifact, no device query)\n";
    out += render_build_arch_line();
    out += "  provenance : the arch line above is this build's own record "
           "(NINFER_BUILD_CUDA_ARCHS <- CMAKE_CUDA_ARCHITECTURES). Every row below is a row of "
           "THIS build's tables (src/core/arch_caps.h). Nothing here is a probe of the card in "
           "this machine and no capability is claimed for it: which card is installed, and "
           "whether it can run a given artifact, is decided when an artifact loads and reported "
           "in full there (src/targets/registry.cpp, construct_target).\n";

    out += "\n  ladder     : ";
    out += std::to_string(kArchLadderSize);
    out += " rung(s) -- the sm values this build's table knows, with each row's microarchitecture "
           "and representative cards. A rung is a CAPABILITY SET plus a per-card route text, NOT "
           "a claim that this binary carries an image for that sm (only the arch line above is "
           "such a record) and not a claim that an artifact may load there. Read the row's route "
           "text and tools/archkit/_GPU_MATRIX.md before claiming a route: a capability bit and "
           "an executable route are different claims.\n";
    for (const ArchRung& rung : kArchLadder) {
        out += "    sm_";
        out += std::to_string(rung.sm);
        out += (rung.sm < 100) ? "   " : "  ";
        out += std::string(rung.label);
        out += " | ";
        out += std::string(rung.cards);
        out += "\n";
    }

    out += "\n  floors     : ";
    out += std::to_string(kFormatRequirementCount);
    out += " weight-format row(s) -- the tensor-core floor of each format this build can bind and "
           "the kernel file that sets it, so \"what would this build refuse, and why\" is "
           "answerable without an artifact:\n";
    for (const FormatRequirement& requirement : kFormatRequirements) {
        out += "    ";
        out += std::string(artifact::format_name(requirement.format));
        out += "  needs ";
        out += (requirement.required == Cap::None) ? std::string("<no tensor-core floor>")
                                                  : std::string(cap_name(requirement.required));
        out += "\n";
        if (!requirement.kernel_evidence.empty()) {
            out += "        kernel floor : ";
            out += std::string(requirement.kernel_evidence);
            out += "\n";
        }
    }

    std::string fallback_rows;
    for (const FormatRequirement& requirement : kFormatRequirements) {
        if (requirement.fallback_required == Cap::None) { continue; }
        if (!fallback_rows.empty()) { fallback_rows += ", "; }
        fallback_rows += std::string(artifact::format_name(requirement.format));
    }
    out += "\n  second floor (pre-Ampere on-ramp): row(s) with a lower fp16 floor: ";
    out += fallback_rows.empty() ? std::string("<none>") : fallback_rows;
    out += "\n               Whether such a floor is HONOURED is not decided here: it needs the "
           "build fact kQpnInBuild (per translation unit, see above) AND a proven hardware "
           "lowering on that rung (fp16_fallback_executable), so it is decided at load time and "
           "reported there.\n";

    out += "\n  no kernel on ANY rung -- declared by this tree, refused on every rung, by name (";
    out += std::to_string(kUncoveredFormatCount);
    out += " row(s)):\n";
    if (kUncoveredFormatCount == 0) {
        out += "    <none>\n";
    }
    for (const UncoveredFormatNote& note : kUncoveredFormatNotes) {
        out += "    ";
        out += std::string(artifact::format_name(note.format));
        out += "  would need ";
        out += std::string(cap_name(note.would_need));
        out += "\n";
    }

    out += "\n  when an artifact IS given, these same tables take the verdict before any weight "
           "byte reaches the device, and a refusal names the format, the missing capability, the "
           "kernel that sets its floor and the routes that would work here.\n";
    return out;
}

// Operator-facing text for a report that is Supported VIA A FALLBACK. Empty when no format
// took one, so a caller may print it unconditionally. This is the LOUD half of the fallback:
// an artifact that loads on a pre-Ampere card because of a lower-floor kernel must say so,
// name the floor the card does NOT have, and name the kernel that made it work. A silent
// fallback would be indistinguishable from a card that meets the floor, which is exactly the
// false positive this table has already produced once (the sm_100 nvfp4 row).
[[nodiscard]] inline std::string render_fallback_notice(const CapabilityReport& report,
                                                       std::string_view artifact_identity) {
    if (report.fallbacks.empty()) { return {}; }
    const ArchRung* rung = arch_rung(report.sm);
    std::string out = "ninfer: this artifact runs here VIA AN fp16 FALLBACK route.\n";
    out += "  artifact   : ";
    out.append(artifact_identity);
    out += "\n  GPU        : sm_";
    out += std::to_string(report.sm);
    if (rung != nullptr) {
        out += " (";
        out.append(rung->label);
        out += ", e.g. ";
        out.append(rung->cards);
        out += ")";
    }
    out += "\n";
    for (const FormatFallback& fallback : report.fallbacks) {
        out += "    - weight format ";
        out += std::string(artifact::format_name(fallback.format));
        out += " does NOT meet its own floor here: it needs ";
        out += std::string(cap_name(fallback.primary_missing));
        out += ", which this card does not have. It is being served by the ";
        out += std::string(cap_name(fallback.fallback_used));
        out += " fallback instead, which consumes the SAME persisted bytes (no "
               "requantization, no fp16 weight copy):\n      kernel : ";
        out.append(fallback.fallback_kernel);
        out += "\n";
    }
    out += "  This is a real execution path, not an approximation, and it is slower than the "
           "native floor would be. Compare the numbers against the native route before "
           "quoting any throughput.\n";
    return out;
}

// Operator-facing text for the ONE verdict that is a warning rather than a refusal:
// Verdict::UnknownArch means "this table has no row for that number", which is a fact
// about the TABLE, not about the card. It is returned instead of thrown (design doc
// section 5.3, "unknown sm does not throw: fall back to the conservative route and warn")
// because gating on a number is exactly the defect this table replaced -- the route
// selector (src/core/kernel_route.h) is where an unknown card gets a conservative route
// and, for a shape it cannot serve, a refusal that NAMES THE MISSING KERNEL.
[[nodiscard]] inline std::string unknown_arch_warning(const CapabilityReport& report,
                                                     std::string_view artifact_identity) {
    if (report.verdict != Verdict::UnknownArch) { return {}; }
    std::string out = "ninfer: compute capability ";
    out += std::to_string(report.sm / 10);
    out += ".";
    out += std::to_string(report.sm % 10);
    out += " (sm_";
    out += std::to_string(report.sm);
    out += ") has no row in kArchLadder (src/core/arch_caps.h), so no capability set can be "
           "assumed for it and NO format floor can be checked. Proceeding on the "
           "conservative route; the route selector will refuse any shape it has no kernel "
           "for and will name that kernel. artifact: ";
    out.append(artifact_identity);
    out += ". Do NOT assume a neighbouring row's capability set -- sm_120 and sm_120a share "
           "the number 120 and do not share the fp4/TMA kernels (tools/archkit/_GPU_MATRIX.md).";
    return out;
}

// Throws std::invalid_argument carrying render_capability_report() when the artifact cannot
// execute on this compute capability. Silent when it can -- except that a Supported verdict
// reached THROUGH THE fp16 FALLBACK prints render_fallback_notice(), because "this artifact
// runs here" and "this artifact runs here natively" are different statements and only one of
// them is true on a pre-Ampere card.
//
// AND ONE MORE ARM SPEAKS, added 2026-09-19: an UnknownArch verdict prints
// unknown_arch_warning() to stderr ONCE PER PROCESS, instead of only when NINFER_ARCH_WARN is
// set. The reason is measured, not stylistic -- see the note at the UnknownArch branch below.
//
// ON THE OVERLOAD SET, because it cost the fleet an hour on 2026-09-17: there is exactly ONE
// declaration and ONE definition, and the build fact reaches both through the declaration's
// default argument. A SECOND, 3-ARG DECLARATION used to sit here with no definition; a 3-arg
// call was then ambiguous between it and the 4-arg form via its default, and
// src/targets/registry.cpp:501 -- the engine's only sm() gate call site -- failed to compile
// with "call of overloaded ... is ambiguous". It was deleted, not reworked, and this note is
// why it must stay deleted. If a declared-then-defined function is ever wanted here, give it
// a definition in the same edit.
inline void require_artifact_formats_supported(
    int sm, std::span<const artifact::NumericFormat> formats,
    std::string_view artifact_identity, bool qpn_in_build = kQpnInBuild) {
    const CapabilityReport report = evaluate_artifact_formats(sm, formats, qpn_in_build);
    if (report.ok()) {
        // Supported, but possibly only via a fallback: say so. Silent when no format took
        // one, which is the case for every card that meets its own floors.
        const std::string notice = render_fallback_notice(report, artifact_identity);
        if (!notice.empty()) { std::fprintf(stderr, "%s\n", notice.c_str()); }
        return;
    }
    // UnknownArch is a warning, not a refusal: see unknown_arch_warning above. THE WARNING
    // PRINTS BY DEFAULT, ONCE PER PROCESS, and this branch is where -- this function is the
    // only production call site in the tree. The comment that used to sit here told callers to
    // call unknown_arch_warning() and log it; measured, no caller in the engine ever did, so
    // the message had no way out and an unplaceable card degraded in silence.
    // NINFER_ARCH_WARN keeps every documented use -- any value still prints -- and gains the
    // off switch an operator expects: 0|off|false|no silences. The ONE refusal path (the throw
    // below) is untouched, so the evidence-backed refusal set is unchanged.
    if (report.verdict == Verdict::UnknownArch) {
        const char* arch_warn = std::getenv("NINFER_ARCH_WARN");
        const std::string_view arch_warn_value = arch_warn != nullptr ? arch_warn : "";
        const bool arch_warn_silenced = arch_warn_value == "0" || arch_warn_value == "off" ||
                                        arch_warn_value == "false" || arch_warn_value == "no";
        static std::atomic<bool> arch_warn_announced{false};
        if (!arch_warn_silenced && !arch_warn_announced.exchange(true)) {
            const std::string warning = unknown_arch_warning(report, artifact_identity);
            std::fprintf(stderr, "%s\n", warning.c_str());
        }
        return;
    }
    throw std::invalid_argument(render_capability_report(report, artifact_identity));
}

// ===========================================================================
// AMD rungs (ROCm / HIP offload-arch targets): A BUILD CONFIGURATION, NOT SUPPORT
// ===========================================================================
//
// READ THIS BEFORE QUOTING ONE LINE OF THE SECTION BELOW.
//
// Nothing in this section is a hardware probe, and nothing in it says AMD works. The machine
// this table was written on has one NVIDIA card and no ROCm toolchain at all (no hipcc, no
// hipify, no /opt/rocm), so the question "does this engine run on gfx906 / gfx908 / gfx90a /
// gfx942 / gfx1100 / gfx1201" is UNANSWERED here and cannot be answered from here. What IS
// answered, and answered by file:line inside this tree, is the narrower question the artifact
// gate actually needs: WHICH OF THIS TREE'S KERNELS depend on an instruction family that an
// amdgcn target has no encoding for, and therefore cannot execute the artifact's format there
// however good the port is.
//
// The discipline is the one already stated above in kArchLadder's own comment and in
// src/core/kernel_route.h's header: a build configuration is not a support claim. A rung below
// is a DECLARED BUILD TARGET with a NAMED REFUSAL attached, and the refusal is the payload --
// its whole purpose is that "does AMD support format X?" now has an answer that names a kernel
// and a file:line instead of an implied claim in either direction.
//
// ---------------------------------------------------------------------------
// WHY THESE ARE NOT ROWS OF kArchLadder. A deliberate deviation, not an oversight.
// ---------------------------------------------------------------------------
//   * ArchRung::sm is documented above as "major * 10 + minor, exactly DeviceContext::sm()".
//     An AMD gfx target has no such number. Inventing one -- gfx906 -> 906 -- would make
//     arch_rung(906) RESOLVE, and every consumer of the ladder reads a row as "a device
//     reports this number": src/core/format_probe.h:906 validates an operator-supplied --sm
//     against the ladder, src/core/arch_sim.h:281 refuses to simulate a number with no row,
//     and src/core/kernel_route.h:562 routes a real device's sm through it. A gfx row in that
//     table would make `--sm 906` a legal way to ask for a simulated gfx906 -- an implied
//     claim, and one made in files this landing does not own.
//   * So the key is the thing ROCm itself names -- the offload-arch target string -- and the
//     table is a SIBLING of kArchLadder, in the same layer, with the same per-row citation
//     discipline. arch_rung() still returns nullptr for every AMD target, and
//     tests/test_arch_caps.cpp pins that so the two tables cannot silently merge.
//
// ---------------------------------------------------------------------------
// AND WHY Cap GAINS NO AMD BITS. Also deliberate.
// ---------------------------------------------------------------------------
// Cap is defined above as "one bit per instruction-set family the engine's registered kernels
// actually emit". THIS TREE EMITS NONE OF THE AMD FAMILIES -- there is no v_mfma_*, v_wmma_*,
// v_dot* or v_lds* anywhere in it -- so an `AmdMfma` bit would be carried by a rung and
// required by no row of kFormatRequirements. That is the exact defect this file has already
// documented and fixed once: see the Fp16Mma/Int8Mma history in the "THE SECOND FLOOR" comment
// on FormatRequirement ("the capability bit that was decorative becomes load-bearing"), where
// two bits that no row required "decided nothing at all". A decorative AMD bit would be worse
// than decorative: it would NAME a hardware feature this tree has never emitted and this box
// cannot probe. What the AMD half needs is not a new capability bit but a new REFUSAL, and
// that is what this section adds.

// ---------------------------------------------------------------------------
// The instruction families this tree emits whose semantics live in inline PTX
// ---------------------------------------------------------------------------
//
// These are the families a toolchain boundary cannot carry by renaming an API. The bound comes
// from the enum's own last member and never from a name copied into a loop, for the reason
// kFormatOrdinalCount above gives: a member appended after a copied name would be invisible to
// the walk, so the walk would pass while a family went uncovered.
enum class PtxFamily : std::uint8_t {
    MmaSync = 0,    // mma.sync.*      -- the tensor-core GEMM channel
    Ldmatrix,       // ldmatrix.sync.* -- the fragment loader the mma operands arrive through
    CpAsync,        // cp.async.*      -- the Ampere async copy in the pipelined GEMMs
    CpAsyncBulkTma, // cp.async.bulk.* -- TMA
    Mbarrier,       // mbarrier.*      -- the TMA completion mechanism
    Setmaxnreg,     // setmaxnreg.*    -- warp-specialisation register hand-off
    Count,
};

[[nodiscard]] inline std::string_view ptx_family_name(PtxFamily family) noexcept {
    switch (family) {
    case PtxFamily::MmaSync: return "mma.sync";
    case PtxFamily::Ldmatrix: return "ldmatrix";
    case PtxFamily::CpAsync: return "cp.async";
    case PtxFamily::CpAsyncBulkTma: return "cp.async.bulk (TMA)";
    case PtxFamily::Mbarrier: return "mbarrier";
    case PtxFamily::Setmaxnreg: return "setmaxnreg";
    case PtxFamily::Count: break;
    }
    return "unknown-ptx-family";
}

inline constexpr std::size_t kPtxFamilyCount = static_cast<std::size_t>(PtxFamily::Count);

// ONE ROW PER (family, site). A row is a CITATION, and the test greps the citation: `mnemonic`
// is the exact text that must be found on the line `file_line` names, so a row cannot be
// copied out of an assessment document -- it is checked against the tree it claims to describe.
// That check is the whole reason this table is data rather than prose.
//
// Every site below is in a SHIPPED source file, i.e. one reached from src/CMakeLists.txt. That
// qualifier is not decoration: this tree also carries fork-survey borrows under src/ whose own
// provenance headers say "ADDITIVE, NOT wired into any build target", and citing one of those
// as a PTX dependency of the engine would be a false mark in the "claims absence where a
// kernel exists" direction -- the site would be real and the dependency would not. Two such
// files hold asm and are therefore deliberately absent below: src/ops/softmax_attention/dense/
// causal_cache/prompt_nvfp4.cuh (unwired; :18 was a phantom #include of
// ops/common/mbarrier.cuh, a path that DOES NOT EXIST in this tree, and the INTEGRATE line has
// now replaced it with an explicit `#error` naming the missing facility and its in-tree
// equivalent, so it can neither compile by accident nor read as working) and
// src/ops/kvarn/decode_kernel.cuh (unwired).
struct PtxSite {
    PtxFamily family;
    std::string_view file_line; // "src/...:NN", inside a shipped file
    std::string_view mnemonic;  // exact text that must appear on that line
    std::string_view guard;     // the arch guard the site sits inside, and what its else arm does
};

inline constexpr PtxSite kPtxSites[] = {
    // mma.sync -- one site per tensor-core channel this tree emits. The channel-to-Cap mapping
    // is the one kFormatRequirements' rows already use, so the two tables read the same fact.
    {PtxFamily::MmaSync, "src/ops/common/mma.cuh:128",
     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
     "NINFER_MMA_HAS_M16N8K16_TC (mma.cuh:30-32, `__CUDA_ARCH__ >= 800`); else arm calls "
     "unsupported_instruction_trap() at mma.cuh:135. This is the mma_f16 channel."},
    {PtxFamily::MmaSync, "src/ops/common/mma.cuh:113",
     "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
     "NINFER_MMA_HAS_M16N8K16_TC (mma.cuh:30-32); else arm traps at mma.cuh:120. This is "
     "mma_bf16 -- the channel EVERY groupwise-int and bf16 format in this tree bottoms out in."},
    {PtxFamily::MmaSync, "src/ops/common/mma.cuh:222",
     "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
     "NINFER_MMA_HAS_M16N8K16_TC (mma.cuh:30-32); else arm traps at mma.cuh:229. This is "
     "mma_s8. NOTE no row of kFormatRequirements requires Cap::Int8Mma today: the "
     "groupwise-int formats route through mma_bf16, not through this channel."},
    {PtxFamily::MmaSync, "src/ops/common/mma.cuh:191",
     "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32",
     "NINFER_MMA_HAS_M8N8K4_F16 (mma.cuh:177-179, `__CUDA_ARCH__ >= 700`); else arm traps at "
     "mma.cuh:200. This is the QPN W4A16 fallback channel named in NVFP4's "
     "fallback_kernel_evidence above."},
    {PtxFamily::MmaSync, "src/ops/common/mma.cuh:237",
     "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32",
     "NINFER_MMA_HAS_KIND_F8F6F4 (mma.cuh:44-49, `__CUDA_ARCH_FEAT_SM*_ALL` || "
     "`__CUDA_ARCH_FAMILY_SPECIFIC__`); else arm traps at mma.cuh:244. This is the fp8 A8 "
     "channel, i.e. the form the engine emits for Cap::Fp8F8f6f4KindMma -- and NOT the plain "
     "e4m3 form Cap::Fp8E4m3MmaPlain names, for which this tree emits no kernel at all."},
    {PtxFamily::MmaSync, "src/ops/common/mma.cuh:276",
     "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X",
     "NINFER_MMA_HAS_KIND_MXF4NVF4 (mma.cuh:54-57, `__CUDA_ARCH_FEAT_SM120/121_ALL` || "
     "(`__CUDA_ARCH_FAMILY_SPECIFIC__` && `__CUDA_ARCH__ >= 1200`)); else arm traps at "
     "mma.cuh:293. This is mma_nvfp4_e4m3, the only nvfp4 kernel this tree has."},
    // ldmatrix -- the four fragment loaders. There is no AMD equivalent to translate them TO,
    // which is why they are their own family rather than a sub-case of mma.sync.
    {PtxFamily::Ldmatrix, "src/ops/common/mma.cuh:65", "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
     "NINFER_MMA_HAS_LDMATRIX (mma.cuh:27-29, `__CUDA_ARCH__ >= 750`); else arm traps at "
     "mma.cuh:70."},
    {PtxFamily::Ldmatrix, "src/ops/common/mma.cuh:77", "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
     "NINFER_MMA_HAS_LDMATRIX (mma.cuh:27-29); else arm traps at mma.cuh:82."},
    {PtxFamily::Ldmatrix, "src/ops/common/mma.cuh:88",
     "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
     "NINFER_MMA_HAS_LDMATRIX (mma.cuh:27-29); else arm traps at mma.cuh:93."},
    {PtxFamily::Ldmatrix, "src/ops/common/mma.cuh:100",
     "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
     "NINFER_MMA_HAS_LDMATRIX (mma.cuh:27-29); else arm traps at mma.cuh:105."},
    // cp.async -- Ampere's non-bulk async copy.
    {PtxFamily::CpAsync, "src/ops/common/memory.cuh:119", "cp.async.cg.shared.global",
     "NINFER_MEMORY_HAS_CP_ASYNC (memory.cuh:18-20, `__CUDA_ARCH__ >= 800`); else arm traps at "
     "memory.cuh:129."},
    {PtxFamily::CpAsync, "src/ops/common/memory.cuh:157", "cp.async.commit_group",
     "NINFER_MEMORY_HAS_CP_ASYNC; else arm traps at memory.cuh:168. The comment there states "
     "the rule this whole section follows: an empty body would silently drop the group "
     "boundary, so a downstream cp_wait would look satisfied while nothing was in flight -- "
     "and 'the route selector is what keeps the kernel from being reached'."},
    {PtxFamily::CpAsync, "src/ops/common/memory.cuh:176", "cp.async.wait_group",
     "NINFER_MEMORY_HAS_CP_ASYNC; else arm traps at memory.cuh:185."},
    // TMA + mbarrier + setmaxnreg -- the Hopper/Blackwell warp-specialisation triple, all four
    // sites in one shipped header whose device body is compiled out below its floor.
    {PtxFamily::CpAsyncBulkTma, "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:264",
     "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
     "NINFER_NVFP4_TMA_DEVICE_ARCH (nvfp4_w4a4_tma.cuh:215-221); when it is 0 the WHOLE KERNEL "
     "BODY is replaced by a single __trap() at nvfp4_w4a4_tma.cuh:295, and the host arm refuses "
     "first (nvfp4_w4a4_tma_arms.cuh). The header's own floor table (nvfp4_w4a4_tma.cuh:194-197) "
     "is measured: mbarrier sm_80+, cp.async.bulk.tensor sm_90+, setmaxnreg sm_90a, "
     "kind::mxf4nvf4 sm_100a+."},
    {PtxFamily::Mbarrier, "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:227",
     "mbarrier.init.shared::cta.b64",
     "NINFER_NVFP4_TMA_DEVICE_ARCH (nvfp4_w4a4_tma.cuh:215-221, false below sm_100a); else the "
     "kernel body is __trap() at nvfp4_w4a4_tma.cuh:295."},
    {PtxFamily::Mbarrier, "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:238",
     "mbarrier.try_wait.parity.shared::cta.b64",
     "NINFER_NVFP4_TMA_DEVICE_ARCH; else the kernel body is __trap() at "
     "nvfp4_w4a4_tma.cuh:295."},
    {PtxFamily::Mbarrier, "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:254",
     "mbarrier.arrive.expect_tx.shared::cta.b64",
     "NINFER_NVFP4_TMA_DEVICE_ARCH; else the kernel body is __trap() at "
     "nvfp4_w4a4_tma.cuh:295."},
    {PtxFamily::Setmaxnreg, "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:316",
     "setmaxnreg.dec.sync.aligned.u32 40;",
     "NINFER_NVFP4_TMA_DEVICE_ARCH; else the kernel body is __trap() at "
     "nvfp4_w4a4_tma.cuh:295. The header records the measurement that fixes this guard: PLAIN "
     "sm_90 is red on this instruction ('Instruction setmaxnreg.dec not supported on .target "
     "sm_90'), so the floor is sm_90a."},
    {PtxFamily::Setmaxnreg, "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:353",
     "setmaxnreg.inc.sync.aligned.u32 232;",
     "NINFER_NVFP4_TMA_DEVICE_ARCH; else the kernel body is __trap() at "
     "nvfp4_w4a4_tma.cuh:295."},
    {PtxFamily::Setmaxnreg, "src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh:95",
     "setmaxnreg.dec.sync.aligned.u32 40;",
     "same NINFER_NVFP4_TMA_DEVICE_ARCH decision, single-sourced from "
     "nvfp4_w4a4_tma.cuh:215-221 (its own comment at :69-71 says so); compiled out of the "
     "kernel body below the floor. This TU is in the build (src/CMakeLists.txt:74)."},
    {PtxFamily::Setmaxnreg, "src/core/device_probe.cu:200",
     "setmaxnreg.inc.sync.aligned.u32 232;",
     "NINFER_PROBE_HAS_SETMAXNREG (defined at device_probe.cu:98-100). This is the TREE'S OWN PROBE "
     "for the capability -- and it is the site that shows why the AMD half cannot be settled by "
     "reading: the probe body is #ifdef'd out when its guard is undefined (device_probe.cu:199), "
     "so on a target that does not define it the probe reports 'not in this build' rather than "
     "answering, and nothing in this tree defines it for an amdgcn target."},
};

inline constexpr std::size_t kPtxSiteCount = sizeof(kPtxSites) / sizeof(kPtxSites[0]);

// THE AMD-SIDE STATUS OF EACH FAMILY -- the ONE external fact in this section, labelled as
// such so it cannot be mistaken for a measurement made here. `measured_in_this_tree` is false
// for every row and the test asserts that, which is this section's analogue of the
// "NotProbed != Supported" rule in src/core/device_capabilities.h:74-81: a classification
// that nobody here probed must not be able to read as one that was.
//
// The classification is taken from ROCm's own CUDA-to-HIP porting guidance, not from memory:
//   * mma.sync -> v_mfma_* (CDNA) / v_wmma_* (RDNA), with a DIFFERENT register layout and a
//     64-lane wavefront; no 1:1 mapping. This is the "different API shape -> re-author" class,
//     not a rename.
//   * ldmatrix -> NO EQUIVALENT AT ALL; fragment layouts have to be regenerated.
//   * cp.async, cp.async.bulk (TMA) and mbarrier -> not present; the closest AMD primitive is a
//     direct-to-LDS load gated by s_waitcnt.
//   * setmaxnreg -> no analogue (it is a warp-specialisation register-reallocation instruction).
//   * And the guard question that decides whether any of this degrades silently: `__CUDA_ARCH__`
//     is deliberately NOT defined for amdgcn (LLVM CFE patch D45387), and the
//     `__CUDA_ARCH_FEAT_SM*_ALL` / `__CUDA_ARCH_FAMILY_SPECIFIC__` macros are CUDA-13-only with
//     no AMD counterpart. EVERY guard cited above keys on one of those, so on an amdgcn target
//     every one of them takes its else arm: the helpers trap and the TMA kernel body is a
//     single __trap().
// That last point is why the marks below are NoKernelInTree rather than "compiles but is
// slow": the kernels exist, the instructions they need do not, and the tree's own convention
// for that situation (memory.cuh:124-129) is a trap plus a route refusal -- never a silent
// fallback.
struct PtxFamilyAmdStatus {
    PtxFamily family;
    // ALWAYS false. False means "no probe on an AMD target backs this", the same way Cap::None
    // in FormatRequirement::fallback_required means "no such kernel exists in this tree".
    bool measured_in_this_tree;
    std::string_view provenance;
};

inline constexpr PtxFamilyAmdStatus kPtxFamilyAmdStatus[] = {
    {PtxFamily::MmaSync, false,
     "EXTERNAL-UNPROBED. No amdgcn encoding of mma.sync exists; the AMD side is v_mfma_* / "
     "v_wmma_* with a different register layout and a 64-lane wavefront (ROCm HIP porting "
     "guidance; ROCmKernelWiki CUDA->HIP migration table). Re-authoring, not translation."},
    {PtxFamily::Ldmatrix, false,
     "EXTERNAL-UNPROBED. ldmatrix has NO AMD equivalent; the fragment layouts it produces must "
     "be regenerated (ROCm HIP porting guidance). This is the single most load-bearing "
     "external fact in this table, because every one of the seven format blockers below uses "
     "an ldmatrix loader as well as an mma.sync instruction."},
    {PtxFamily::CpAsync, false,
     "EXTERNAL-UNPROBED. No cp.async on AMD; the portable shape is a direct-to-LDS load with "
     "s_waitcnt. 'Different API shape', so a re-expression rather than a new architecture."},
    {PtxFamily::CpAsyncBulkTma, false,
     "EXTERNAL-UNPROBED. AMD has no TMA and no tensor-map descriptor; the closest primitive is "
     "the same direct-to-LDS load. 'Does not exist'."},
    {PtxFamily::Mbarrier, false,
     "EXTERNAL-UNPROBED. No mbarrier analogue on AMD; AMD's s_barrier is a single workgroup "
     "barrier and does not carry a transaction count. 'Does not exist'."},
    {PtxFamily::Setmaxnreg, false,
     "EXTERNAL-UNPROBED. No AMD analogue: the instruction exists to reallocate registers "
     "between warp-specialised roles inside one CTA. 'Does not exist'."},
};

inline constexpr std::size_t kPtxFamilyAmdStatusCount =
    sizeof(kPtxFamilyAmdStatus) / sizeof(kPtxFamilyAmdStatus[0]);

// Every family has an AMD status row, and exactly one, walked over the enum's own ordinal
// range -- the same shape as the format-coverage walk above, and for the same reason.
namespace detail {
constexpr bool ptx_families_all_have_amd_status() noexcept {
    for (std::size_t raw = 0; raw < kPtxFamilyCount; ++raw) {
        const auto family = static_cast<PtxFamily>(raw);
        std::size_t hits = 0;
        for (const PtxFamilyAmdStatus& status : kPtxFamilyAmdStatus) {
            if (status.family == family) { ++hits; }
        }
        if (hits != 1) { return false; }
    }
    return kPtxFamilyAmdStatusCount == kPtxFamilyCount;
}
constexpr bool ptx_sites_cover_every_family() noexcept {
    for (std::size_t raw = 0; raw < kPtxFamilyCount; ++raw) {
        const auto family = static_cast<PtxFamily>(raw);
        bool found = false;
        for (const PtxSite& site : kPtxSites) {
            if (site.family == family) { found = true; break; }
        }
        if (!found) { return false; }
    }
    return true;
}
} // namespace detail

inline constexpr bool kPtxFamiliesAllHaveAmdStatus = detail::ptx_families_all_have_amd_status();
inline constexpr bool kPtxSitesCoverEveryFamily = detail::ptx_sites_cover_every_family();

static_assert(kPtxFamiliesAllHaveAmdStatus,
              "every PtxFamily needs exactly one kPtxFamilyAmdStatus row (src/core/arch_caps.h). "
              "A family with no AMD status is a dependency nobody classified -- add the row, do "
              "not delete the family.");
static_assert(kPtxSitesCoverEveryFamily,
              "every PtxFamily needs at least one kPtxSites citation, or a family could be "
              "declared with no site in the tree to justify it.");

// ---------------------------------------------------------------------------
// The rungs themselves
// ---------------------------------------------------------------------------
//
// THE KEYS ARE ROCm OFFLOAD-ARCH TARGETS, i.e. the exact strings that would go into
// CMAKE_HIP_ARCHITECTURES or `--offload-arch=`, because that is the only name an AMD target
// has and it is the name an operator would type. `probe_evidence` is empty on EVERY row and
// that is a claim, not an omission: empty means "no probe exists for this target", exactly as
// Cap::None in a FormatRequirement row means "no such kernel exists in this tree". Making it
// non-empty is the edit that turns this table into support, and whoever makes it owes a
// hardware probe and a file:line for it.
struct AmdRung {
    std::string_view target;    // the ROCm offload-arch name; the key
    std::string_view isa;       // the ISA generation NAME (a label, never a capability claim)
    std::string_view cards;     // representative parts
    std::string_view evidence;  // where this target name comes from in THIS tree, file:line
    std::string_view probe_evidence; // EMPTY on every row: no AMD hardware was probed
    std::string_view route;     // what the operator is told, and what is NOT claimed
};

inline constexpr AmdRung kAmdLadder[] = {
    {"gfx906", "Vega20 / GCN5.1", "MI50 / MI60 / Radeon VII",
     "the ONLY one of these six names that already occurs in this tree -- in 21 files, and every "
     "one of them a fork-survey borrow whose own provenance header reads \"ADDITIVE, NOT wired "
     "into any build target\": include/ninfer/ops/allreduce.h:143 and "
     "src/ops/common/allreduce.cu:263,:345 (NINFER_GFX906_TP2_FLAG_SYNC), plus "
     "docs/gfx906/*.md. `allreduce` has 0 hits in src/CMakeLists.txt, whose source lists are "
     "explicit (no GLOB), so this plumbing is inert. The upstream ENGINE has no gfx906 path "
     "that is in the build.",
     "",
     "lowest rung of the ladder and the one the in-tree fork-survey docs report a bring-up on; "
     "that report is third-party, quoted and unverified, and is NOT evidence for this table. "
     "Every tensor-core format below is refused here by name. Two further limits this rung "
     "carries that no other rung in this file does: (1) three kernels in this tree request more "
     "dynamic shared memory than one workgroup is allowed -- see kAmdLdsBlockerSites below; "
     "(2) the 32-lane warp assumption (src/ops/common/warp.cuh:7 kWarpSize = 32, plus the "
     "direct __shfl_*_sync sites) is a SEMANTIC hazard a static table cannot settle. Neither is "
     "modelled as a Cap bit, because neither is a capability."},
    {"gfx908", "CDNA1", "MI100",
     "appears NOWHERE in this tree (0 files). The target name is the ROCm offload-arch string; "
     "no file in this tree names it.", "",
     "no gfx908 kernel, no gfx908 build target, no gfx908 probe. Every tensor-core format below "
     "is refused by name. This rung is on the ladder because it is a target an operator can "
     "legitimately ask `--offload-arch=` for, and the honest answer for it is a refusal that "
     "names the missing kernels rather than silence."},
    {"gfx90a", "CDNA2", "MI200 / MI210 / MI250 / MI250X",
     "appears NOWHERE in this tree (0 files). The target name is the ROCm offload-arch string; "
     "no file in this tree names it.", "",
     "same as gfx908. It is called out separately because it is the first rung that WOULD "
     "exercise v_mfma_* -- i.e. it is the rung that would answer the mma.sync blocker -- and no "
     "kernel in this tree emits v_mfma_*, so the answer is still a refusal, now for a different "
     "reason: the missing piece is a re-authored kernel, not a hardware feature."},
    {"gfx942", "CDNA3", "MI300A / MI300X",
     "appears NOWHERE in this tree (0 files). The target name is the ROCm offload-arch string; "
     "no file in this tree names it.", "",
     "same as gfx90a. Current-generation and therefore the rung most tempting to assume, which "
     "is exactly why it says what it says: this tree contains no gfx942 kernel and no gfx942 "
     "probe, so nothing here establishes anything about it."},
    {"gfx1100", "RDNA3", "RX 7900 XTX / W7900",
     "appears NOWHERE in this tree (0 files). The target name is the ROCm offload-arch string; "
     "no file in this tree names it.", "",
     "same refusal. RDNA's matrix path is v_wmma_*, NOT v_mfma_*, so even a kernel re-authored "
     "for gfx90a would not serve this rung -- and that difference is a fact about the AMD side, "
     "so it is labelled EXTERNAL-UNPROBED in kPtxFamilyAmdStatus rather than asserted here."},
    {"gfx1201", "RDNA4", "RX 9070 / RX 9070 XT",
     "appears NOWHERE in this tree (0 files). The target name is the ROCm offload-arch string; "
     "no file in this tree names it.", "",
     "same refusal. Highest rung listed, and the one whose ROCm support is newest and least "
     "documented; if anything here is more likely to be wrong rather than merely unprobed, it "
     "is this row's ISA label, which is a NAME and not a capability."},
};

inline constexpr std::size_t kAmdLadderSize = sizeof(kAmdLadder) / sizeof(kAmdLadder[0]);

// Exact match only, on the target string. nullptr means "not an AMD target this table knows",
// and the caller must refuse rather than assume a neighbour -- the same rule arch_rung()
// follows for an unlisted compute capability, and for the same reason.
[[nodiscard]] inline const AmdRung* amd_rung(std::string_view target) noexcept {
    for (const AmdRung& rung : kAmdLadder) {
        if (rung.target == target) { return &rung; }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// The hard limit that is not a Cap bit: dynamic shared memory
// ---------------------------------------------------------------------------
//
// A workgroup on the AMD targets below may request 64 KB of LDS. Three kernels in this tree ask
// for more, so they cannot run there AT ALL -- not slowly, not at reduced tile size: they must
// be split along the K tiling before they can be launched. This is a KERNEL and LINE fact, so it
// is cited the same way the PTX sites are, and it is deliberately NOT modelled as a PtxFamily:
// no instruction is missing and no Cap bit is involved.
//
// It is listed here rather than only in the report because it is the one blocker on the AMD side
// that no translator can see and no capability bit can express -- arithmetic on the request, not
// reading of the instruction -- which makes it exactly the kind of fact that goes missing when a
// port is planned from an instruction census alone.
struct AmdLdsBlockerSite {
    std::string_view file_line;
    std::string_view requested; // the request, in bytes and KB, as read off the cited line
    std::string_view mnemonic;  // exact text that must appear on that line (the citation check)
    std::string_view note;
};

inline constexpr AmdLdsBlockerSite kAmdLdsBlockerSites[] = {
    {"src/ops/linear/qpn/qpn_kernels.cuh:893", "98304 B = 96 KB", "96 * 1024",
     "cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024) on the "
     "config-selectable WMMA entry (set_smem_opt). NOTE this file is the QPN family, which this "
     "build does not contain (0 hits for `qpn` in src/CMakeLists.txt), so this site is a limit "
     "on a route that is already refused for a different reason -- recorded rather than dropped, "
     "because a reader who sees only the two MoE sites below would conclude the limit is "
     "confined to the MoE path."},
    {"src/targets/qwen3_8_flash_next/impl/moe_kernels.cu:1989", "69312 B = 67.7 KB", "69312",
     "flash_next_moe_prefill_gate_up_mma_kernel<false>, also launched with 69312 bytes of "
     "dynamic shared memory at moe_kernels.cu:2007. This target IS in the build."},
    {"src/targets/qwen3_8_flash_next/impl/moe_kernels.cu:1991", "92416 B = 90.25 KB", "92416",
     "flash_next_moe_prefill_gate_up_mma_kernel<true>, also launched with 92416 bytes at "
     "moe_kernels.cu:2024. This target IS in the build."},
};

inline constexpr std::size_t kAmdLdsBlockerSiteCount =
    sizeof(kAmdLdsBlockerSites) / sizeof(kAmdLdsBlockerSites[0]);

// ---------------------------------------------------------------------------
// The mark: one verdict per (AMD target, artifact format)
// ---------------------------------------------------------------------------
//
// NoKernelInTree is the SAME WORD AND THE SAME CONCEPT as caps::RouteOutcome::NoKernelInTree
// (src/core/kernel_route.h:122-125, ": a route whose kernel is genuinely absent from this
// tree"), and it is spelled the same on purpose: two vocabularies for one fact is how a tree
// starts disagreeing with itself. It cannot be the same ENUM here -- kernel_route.h includes
// this header, so the dependency would be a cycle -- so the tie is kept by the string
// amd_format_verdict_name() returns and by a test that compares it against
// outcome_name(RouteOutcome::NoKernelInTree) rather than against a literal.
enum class AmdFormatVerdict : std::uint8_t {
    // The kernel this tree ships for this format emits an instruction family that has no
    // amdgcn encoding, so no kernel in this tree can execute the format on this target.
    NoKernelInTree = 0,
    // The format's job in this tree is not a tensor-core GEMM -- scale words, indices, mapped
    // host metadata -- so none of the PTX families above stands in its way. This is NOT a claim
    // that the format works on AMD; it is the absence of the one specific blocker this table
    // knows how to name.
    NotATensorCoreOperand,
    // Not a target in kAmdLadder. Fail closed: refuse by name, never assume a neighbour's
    // answer -- the rule kArchLadder's own comment sets for an unlisted compute capability.
    UnknownTarget,
};

[[nodiscard]] inline std::string_view amd_format_verdict_name(AmdFormatVerdict verdict) noexcept {
    switch (verdict) {
    case AmdFormatVerdict::NoKernelInTree: return "no-kernel-in-tree";
    case AmdFormatVerdict::NotATensorCoreOperand: return "not-a-tensor-core-operand";
    case AmdFormatVerdict::UnknownTarget: return "unknown-target";
    }
    return "unknown-verdict";
}

// ONE ROW PER FORMAT THAT HAS A TENSOR-CORE FLOOR, i.e. exactly the formats whose
// kFormatRequirements row carries a `required` other than Cap::None. Each row cites TWO sites
// in THAT FORMAT'S OWN kernel -- the mma channel AND the ldmatrix fragment loader -- because
// both are needed for the format and both lack an amdgcn encoding. Citing only the mma.cuh
// channel would understate it: the operand fragments arrive through ldmatrix, which has no AMD
// equivalent at all, so a port cannot even feed the channel it cannot issue.
//
// `cap` is the Cap bit this row is the AMD-side counterpart of, and the test asserts it equals
// format_requirement(format)->required. That is what keeps this table from becoming a second
// census that drifts: it does not restate WHICH formats have a floor, it reads that from the
// format table and only adds WHERE the instruction is.
struct AmdFormatBlocker {
    artifact::NumericFormat format;
    Cap cap;
    std::string_view mma_site;      // file:line in the format's own kernel
    std::string_view mma_mnemonic;  // exact text at that line
    std::string_view fragment_site;
    std::string_view fragment_mnemonic;
    // A SECOND route this tree has for the same format, whose own instructions are blocked
    // independently -- empty when the format has only the one route. Recorded per format rather
    // than as a general sentence, because a general sentence would have to claim it for formats
    // that do not have a second route (the trap kFormatRequirements' fallback comment names:
    // "this table refuses to make one up"). The one non-empty row is NVFP4, whose route in this
    // build ALSO has a warp-specialised TMA variant.
    std::string_view secondary_route;
};

inline constexpr AmdFormatBlocker kAmdFormatBlockers[] = {
    {artifact::NumericFormat::BF16, Cap::Bf16Mma, "src/ops/linear/bf16/bf16_gemm_mma.cuh:296",
     "mma_bf16(", "src/ops/linear/bf16/bf16_gemm_mma.cuh:266", "ldmatrix_x4(", ""},
    {artifact::NumericFormat::Q4G64_F16S, Cap::Bf16Mma,
     "src/ops/linear/q4/q4_rowsplit_gemm_mma.cuh:341", "mma_bf16(",
     "src/ops/linear/q4/q4_rowsplit_gemm_mma.cuh:316", "ldmatrix_x4(", ""},
    {artifact::NumericFormat::Q5G64_F16S, Cap::Bf16Mma,
     "src/ops/linear/q5/q5_rowsplit_gemm_mma.cuh:385", "mma_bf16(",
     "src/ops/linear/q5/q5_rowsplit_gemm_mma.cuh:360", "ldmatrix_x4(", ""},
    {artifact::NumericFormat::Q6G64_F16S, Cap::Bf16Mma,
     "src/ops/linear/q6/q6_rowsplit_gemm_mma.cuh:394", "mma_bf16(",
     "src/ops/linear/q6/q6_rowsplit_gemm_mma.cuh:368", "ldmatrix_x4(", ""},
    {artifact::NumericFormat::W8G32_F16S, Cap::Bf16Mma,
     "src/ops/linear/w8/w8_rowsplit_gemm_mma.cuh:267", "mma_bf16(",
     "src/ops/linear/w8/w8_rowsplit_gemm_mma.cuh:245", "ldmatrix_x4(", ""},
    {artifact::NumericFormat::NVFP4, Cap::Mxf4Nvfp4BlockScale,
     "src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh:308", "mma_nvfp4_e4m3(",
     "src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh:267", "ldmatrix_x4(",
     "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:227"},
    {artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S, Cap::Bf16Mma,
     "src/ops/linear/fp8/fp8_a16_gemm_mma.cuh:206", "mma_bf16(",
     "src/ops/linear/fp8/fp8_a16_gemm_mma.cuh:183", "ldmatrix_x4(", ""},
};

inline constexpr std::size_t kAmdFormatBlockerCount =
    sizeof(kAmdFormatBlockers) / sizeof(kAmdFormatBlockers[0]);

[[nodiscard]] inline const AmdFormatBlocker*
amd_format_blocker(artifact::NumericFormat format) noexcept {
    for (const AmdFormatBlocker& blocker : kAmdFormatBlockers) {
        if (blocker.format == format) { return &blocker; }
    }
    return nullptr;
}

// The verdict for one (target, format) pair. Pure function of the tables above and of
// kFormatRequirements -- no hardware, no probe, no build fact. `target` is the offload-arch
// string, e.g. "gfx906".
//
// The rule, in one line: a format is NoKernelInTree on an AMD target exactly when it has a
// tensor-core floor, because every Cap bit in this tree is emitted as one of the families in
// kPtxSites and every one of those is guarded by a macro that is false on amdgcn. A format with
// no floor (Cap::None) has nothing for those families to stand in the way of, and a format this
// tree declares and has no kernel for on ANY rung (kUncoveredFormatNotes) is NoKernelInTree for
// its own reason, which the AMD gate must not restate as its own.
[[nodiscard]] inline AmdFormatVerdict amd_format_verdict(std::string_view target,
                                                        artifact::NumericFormat format) noexcept {
    if (amd_rung(target) == nullptr) { return AmdFormatVerdict::UnknownTarget; }
    const FormatRequirement* requirement = format_requirement(format);
    if (requirement != nullptr && requirement->required == Cap::None) {
        return AmdFormatVerdict::NotATensorCoreOperand;
    }
    return AmdFormatVerdict::NoKernelInTree;
}

// Operator-facing text for a NoKernelInTree answer, naming the format, the instruction families,
// the kernel file:line for THIS format, and what remains unestablished. Empty for every other
// verdict, so a caller may print it unconditionally in the same shape render_fallback_notice()
// uses.
//
// It deliberately says "no kernel in this tree" and not "unsupported GPU": the missing piece is
// the KERNEL, and a reader who takes it as a verdict on the card has made exactly the mistake
// this section exists to prevent.
[[nodiscard]] inline std::string render_amd_format_refusal(std::string_view target,
                                                          artifact::NumericFormat format) {
    const AmdRung* rung = amd_rung(target);
    if (rung == nullptr) {
        return std::string("ninfer: AMD target '") + std::string(target) +
               "' is not in kAmdLadder (src/core/arch_caps.h), so no kernel, no capability set "
               "and no probe result can be named for it. Refusing rather than assuming a "
               "neighbouring rung's answer.\n";
    }
    if (amd_format_verdict(target, format) != AmdFormatVerdict::NoKernelInTree) {
        return std::string{};
    }
    std::string out = "ninfer: no kernel in this tree can execute this weight format on AMD "
                      "target ";
    out.append(rung->target);
    out += " (";
    out.append(rung->isa);
    out += ", e.g. ";
    out.append(rung->cards);
    out += ").\n  weight format : ";
    out += std::string(artifact::format_name(format));

    const FormatRequirement* requirement = format_requirement(format);
    const AmdFormatBlocker* blocker      = amd_format_blocker(format);
    if (requirement == nullptr) {
        // A format this tree declares and has no kernel for on ANY rung: the AMD gate must not
        // present that as its own finding.
        out += "\n  this format has NO KERNEL ON ANY RUNG of this tree -- see the named refusal "
               "in kUncoveredFormatNotes, which is the answer, and it is not specific to AMD.\n";
        return out;
    }
    out += "\n  needs         : ";
    out += std::string(cap_name(requirement->required));
    out += "\n  kernel floor  : ";
    out.append(requirement->kernel_evidence);
    if (blocker != nullptr) {
        out += "\n  amd blockers  : the kernel above reaches its tensor core through two "
               "instruction families an amdgcn target has no encoding for:\n      mma     : ";
        out.append(blocker->mma_site);
        out += " (";
        out.append(blocker->mma_mnemonic);
        out += ")\n      fragment: ";
        out.append(blocker->fragment_site);
        out += " (";
        out.append(blocker->fragment_mnemonic);
        out += " -- ldmatrix has NO AMD equivalent; the layouts must be regenerated)";
        if (!blocker->secondary_route.empty()) {
            out += "\n      and this format has a SECOND route in this build, blocked on its own "
                   "instructions (mbarrier / cp.async.bulk.tensor / setmaxnreg):\n                ";
            out.append(blocker->secondary_route);
        }
    }
    out += "\n  no amdgcn encoding exists for any of: ";
    bool first_family = true;
    for (const PtxFamilyAmdStatus& status : kPtxFamilyAmdStatus) {
        if (!first_family) { out += ", "; }
        out += std::string(ptx_family_name(status.family));
        first_family = false;
    }
    out += "\n      (kPtxFamilyAmdStatus; every row is labelled EXTERNAL-UNPROBED -- these are "
           "AMD-side facts taken from ROCm's porting guidance, not measurements made here)";
    out += "\n  this is a BUILD CONFIGURATION, NOT A SUPPORT CLAIM. Nothing in this tree was "
           "built for, run on or probed against ";
    out.append(rung->target);
    out += ":\n    rung evidence : ";
    out.append(rung->evidence);
    if (rung->probe_evidence.empty()) {
        out += "\n    probe         : NONE. No AMD hardware and no ROCm toolchain exist on the "
               "machine this table was written on, so every answer here is a refusal derived "
               "from this tree's own source, never a result.\n";
    }
    out += "  The missing piece is a KERNEL to be written, not a card to be replaced, and not a "
           "row to be added: adding a row here would convert this refusal into a claim. What "
           "would move it: a re-authored kernel for this format whose mma and fragment loads are "
           "written in amdgcn (v_mfma_*/v_wmma_* plus regenerated LDS layouts), and then a probe "
           "on real hardware -- only a probe is evidence of support.\n";
    return out;
}

} // namespace ninfer::caps
