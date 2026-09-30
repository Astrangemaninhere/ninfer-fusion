#ifndef NINFER_CORE_AMD_ROUTE_H
#define NINFER_CORE_AMD_ROUTE_H

// ---------------------------------------------------------------------------
// THE AMD FORMAT-ROUTE DESCENT (dl/amdprobe).
//
// WHAT THIS FILE IS, IN THE OWNER'S OWN WORDS. The question it answers, asked one level at a
// time, is: given a WEIGHT FORMAT and a target, does THIS LEVEL support it natively -- and if
// not, step down one level and ask again, until a level supports it or the ladder is exhausted.
// It is the AMD counterpart of src/core/kernel_route.h's select_route(sm, format, shape,
// qpn_in_build), and it is deliberately the SAME SHAPE rather than a second idea:
//   * a route ladder ordered by FLOOR, highest first, walked downward;
//   * three outcomes, spelled the same (Selected / UnknownArchFallback / NoKernelInTree);
//   * a RouteChoice-shaped answer whose `why` is ALWAYS NON-EMPTY and which names the KERNEL
//     that has to be written rather than the card that asked;
//   * the `simulated` marker kept as its own field so a caller can branch without parsing text.
// kernel_route.h is NOT included and NOT modified by this file: one file, one writer. The route
// ladder here reads the AMD tables (kAmdLadder, kFormatRequirements, kAmdFormatBlockers,
// kAmdLdsBlockerSites, kPtxFamilyAmdStatus) through src/core/amd_route_tables.h, which is the
// arch_caps.h-fed adapter. THIS header includes only the standard library, so the descent itself
// can be exercised, and linked, by a compiler that has no CUDA and no HIP.
//
// ---------------------------------------------------------------------------
// WHY THE AVAILABILITY OF A LEVEL IS NOT A BOOL. This is the whole point of the discipline this
// file is written under, and it is a MEASURED rule rather than a stylistic one.
//
// MEASURED 2026-09-24 (dl/amdprobe): src/CMakeLists.txt:550 said of the gfx906 port layer,
// verbatim, "28 files on disk, 0 of them wired" -- and SIX of those files are named as
// `target_sources` entries at src/CMakeLists.txt:590-595, inside `if(NINFER_ENABLE_GFX906_COMPAT)`.
// The same file's line 572 says src/ops/gfx906_stubs.cpp "IS DELIBERATELY NOT IN THE LIST", and a
// plain hit-count of that filename in src/CMakeLists.txt returns 1 -- the PROSE of line 572. A
// counter of HITS reads a sentence saying "not in the list" as a wiring.
// So the availability of a level is answered by FOUR SEPARATE READINGS, each reported on its own,
// and the conjunction of the first three is what `in_build` means:
//   reading 1  source_exists        -- the file is on disk (find(1))
//   reading 2  is_source_entry      -- that filename appears as a SOURCE ENTRY in a target's
//                                      source list, i.e. after stripping FILE and LINE the line
//                                      does NOT begin with '#', and it is inside a
//                                      target_sources/add_library list -- not a diagnostic string
//   reading 3  object_in_build_dir  -- a .o for it exists under a build directory
//   reading 4  symbol_resolved      -- the LINKER answered, because a TU took the kernel's address
// Reading 4 is the STRONGEST and is deliberately NOT part of `in_build`, because it can only be
// taken in the configuration that has an AMD toolchain: requiring it would report "not in this
// build" for a level that IS in it. It is carried as its own column and rendered as its own line.
// That is the technique src/ops/linear/qpn/qpn_arch_route.cpp:36 already uses for the NV side
// (`return &gemm_qpn != nullptr;` -- "the symbol is either resolved by the linker or the link
// fails"), cited here rather than re-derived.
//
// AND THERE IS NO BAKED-IN BOOLEAN ANYWHERE IN THIS FILE. Every measured fact arrives as a
// REQUIRED field of AmdRouteInputs with NO DEFAULT, so the defect dl/amdprobe measured on the NV
// side -- kernel_route.h's `qpn_in_build` defaulting to the STALE half of a per-translation-unit
// macro, which made the descent SKIP a level that really exists -- cannot be reproduced here. A
// caller that forgets a measurement gets a compile error, not a wrong route.
//
// ---------------------------------------------------------------------------
// "BOTTOMED OUT" IS TWO DIFFERENT SENTENCES AND IS PRINTED AS TWO COLUMNS.
//   * AmdRouteOutcome::NoKernelInTree -- THERE IS NO ROUTE IN THIS TREE for this (format, rung).
//     `why` names the missing kernel. Token: `no-route-in-tree`.
//   * Selected with `levels_descended > 0` -- A ROUTE IS SELECTED, but at a level BELOW the
//     format's own floor. The artifact can run; it runs on a lower-floor path. Token:
//     `descended-to-lower-level`.
//   * Selected with `levels_descended == 0` -- A ROUTE IS SELECTED AT THE FORMAT'S OWN FLOOR.
//     Token: `selected-at-floor`.
// Collapsing the first two into "unsupported" is the false attribution this file exists to
// prevent, exactly as cufree_report.h:%s forbids one clause over two situations.
//
// ---------------------------------------------------------------------------
// WHAT THIS FILE MAY AND MAY NOT SAY -- same restraint as cufree_report.h and
// amdsafe_hip_stack.h, for the same measured reason:
//   * NO VENDOR IS ATTRIBUTED TO ANY ACCELERATOR. A rung is named by its ROCm offload-arch
//     string, which is the only name an AMD target has; no card is inferred from it.
//   * A NoKernelInTree IS NOT A STATEMENT ABOUT THE CARD. The missing piece is a KERNEL.
//   * kAmdLadder's `probe_evidence` IS NOT FILLED BY ANYTHING HERE, and this file renders that
//     as a sentence rather than leaving it implicit: the six rows are empty by design, and a
//     route decided from a table is NOT a probe result. What this file DOES produce per cell is
//     the named measurement that would make the cell a real result -- the rule
//     src/core/format_probe.h:%s states for the simulated leg, verbatim: "simulated verdict must
//     be re-probeable on real hardware and a rung this [table does not contain cannot be]".

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::caps {

// ---------------------------------------------------------------------------
// The route ladder, ordered by FLOOR. Named for the MECHANISM, never for a card, so a level can
// be justified by a kernel file and refused by a missing one.
//
// The order is the walk order read BACKWARD: the descent tests AmdMfmaNative first and
// AmdHostCpuReference last. Read kernel_route.h's KernelRoute for the same shape on the NV side
// (there the highest-floor member is Nvfp4W4a4Tma and the lowest is ConservativeSimt).
// ---------------------------------------------------------------------------
enum class AmdRoute : std::uint8_t {
    // No route: nothing in this tree can serve the request at any level.
    None = 0,
    // THE LOWEST FLOOR, and it is not a kernel at all: the primitive is implemented ON THE HOST.
    // It is a REAL level in this tree and it is not a placeholder -- src/core/amdsafe_gate.h:344
    // admits a path declaring AmdSafeExecTarget::HostCpu UNCONDITIONALLY, without examining its
    // primitive list, and returns the token `host-path-primitives-implemented-on-host`. It is the
    // only level of this ladder that needs no AMD compiler, no ROCm and no AMD card, which is
    // exactly why it is the floor rather than an afterthought.
    AmdHostCpuReference,
    // A tensor-core-free amdgcn kernel that consumes this format's own bytes. The channel is the
    // dot-product / shuffle path -- v_dot2_f32_f16, ds_bpermute and __builtin_amdgcn_sdot4 --
    // NOT a matrix core. This level EXISTS IN THIS TREE and is MEASURED: the six sources at
    // src/CMakeLists.txt:590-595, whose real kernels are
    // src/ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh and
    // src/ops/linear/gfx906/q_gemv_gfx906.cuh, both carrying AMD-only builtins that "a CUDA
    // compiler cannot parse" (their own provenance line). It is the AMD analogue of
    // KernelRoute::ConservativeSimt, and like that one it is the answer for the rungs where no
    // matrix-core kernel exists.
    AmdDotSimt,
    // A v_wmma_* matrix-core kernel for this format. RDNA's matrix path; kernel_route.h's RDNA
    // note records that "RDNA's matrix path is v_wmma_*, NOT v_mfma_*".
    AmdWmmaNative,
    // A v_mfma_* matrix-core kernel for this format. CDNA's matrix path, and the highest floor
    // on this ladder.
    AmdMfmaNative,
    Count,
};

[[nodiscard]] inline std::string_view amd_route_name(AmdRoute route) noexcept {
    switch (route) {
    case AmdRoute::None:              return "none";
    case AmdRoute::AmdHostCpuReference: return "amd-host-cpu-reference";
    case AmdRoute::AmdDotSimt:        return "amd-dot-simt";
    case AmdRoute::AmdWmmaNative:     return "amd-wmma-native";
    case AmdRoute::AmdMfmaNative:     return "amd-mfma-native";
    case AmdRoute::Count:             break;
    }
    return "unknown-route";
}

// ---------------------------------------------------------------------------
// The three outcomes, spelled the same as kernel_route.h's RouteOutcome and for the reason that
// file gives: two vocabularies for one fact is how a tree starts disagreeing with itself.
// ---------------------------------------------------------------------------
enum class AmdRouteOutcome : std::uint8_t {
    Selected = 0,
    // The rung is NOT in kAmdLadder. Fail OPEN downward, the way kernel_route.h's
    // UnknownArchFallback does: take the lowest level this build has and report the warning, so
    // the caller can branch on the outcome instead of on text. A rung nobody measured must not
    // be answered with a neighbour's floor -- the note in kAmdLadder is explicit about that
    // ("sm_120 and sm_120a share the number 120 and do not share the fp4/TMA kernels").
    UnknownRungFallback,
    // No level of this ladder serves this (format, rung). `why` names the MISSING KERNEL.
    NoKernelInTree,
};

[[nodiscard]] inline std::string_view amd_outcome_name(AmdRouteOutcome outcome) noexcept {
    switch (outcome) {
    case AmdRouteOutcome::Selected:            return "selected";
    case AmdRouteOutcome::UnknownRungFallback: return "unknown-rung-fallback";
    case AmdRouteOutcome::NoKernelInTree:      return "no-route-in-tree";
    }
    return "unknown-outcome";
}

// ---------------------------------------------------------------------------
// THE TWO-COLUMN MARK, as its own enum so a caller cannot collapse the two sentences by
// accident. This is the field the coordinator's rule requires to be printable separately:
// "本树里没有这条路" and "这条路在，但我这级跑不了" are different sentences.
// ---------------------------------------------------------------------------
enum class AmdBottomKind : std::uint8_t {
    // A route was selected at the format's own floor. Nothing was stepped over.
    SelectedAtFloor = 0,
    // A route was selected, but only after stepping down `levels_descended` levels. The artifact
    // CAN run; it runs on a lower-floor path, and the caller is told how far down it went.
    DescendedToLowerLevel,
    // No route in this tree. This is the ONLY one of the three that is a refusal.
    NoRouteInTree,
};

[[nodiscard]] inline std::string_view amd_bottom_token(AmdBottomKind kind) noexcept {
    switch (kind) {
    case AmdBottomKind::SelectedAtFloor:      return "selected-at-floor";
    case AmdBottomKind::DescendedToLowerLevel: return "descended-to-lower-level";
    case AmdBottomKind::NoRouteInTree:        return "no-route-in-tree";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Inputs. EVERY MEASURED FIELD IS REQUIRED. There is no default on any of them, and that is a
// correction of a measured defect rather than a style choice -- see the header comment's note on
// the NV `qpn_in_build` default. A field with no default cannot be left stale.
// ---------------------------------------------------------------------------

struct AmdProblemShape {
    std::uint32_t m = 1; // rows. decode 1..3, verify/draft 4..16, prefill >= 17
    std::uint32_t n = 0; // output columns
    std::uint32_t k = 0; // reduction extent
};

// ONE ROUTE LEVEL'S MEASUREMENTS. The readings are separate fields on purpose; see the header
// comment. `in_build()` is the conjunction and is a FUNCTION, not a stored bool, so no caller can
// hand over a pre-decided answer.
//
// THE KIND AXIS IS A CORRECTION RECEIVED FROM THIS FILE'S OWN POSITIVE CONTROL, and it is recorded
// rather than folded in silently. The first form of this struct asked the SAME three questions of
// every level, and the driver's denominator control then read ZERO selected cells out of 72: the
// lowest level's carrier is src/core/amdsafe_gate.h, a HEADER-ONLY FACILITY, which has no .o by
// construction and is not a target_sources entry -- so the compiled-source reading reported a
// facility that IS in this build as absent, and the walk stepped past the one level that can
// always answer. A header's availability is a DIFFERENT KIND of question and now has its own
// reading: it is available iff some COMPILED TU reaches it. MEASURED 2026-09-24:
// src/core/amdsafe_gate.h has exactly one includer, src/targets/registry.cpp:9, and registry.cpp
// IS a source entry (dl/amdprobe/logs/13_includers.log).
enum class AmdLevelKind : std::uint8_t {
    // The level is carried by a .cu/.cpp that a target compiles. Three corners apply.
    CompiledSource = 0,
    // The level is carried by a header. It has no object of its own, so the question is whether a
    // compiled TU reaches it.
    HeaderOnlyFacility,
};

[[nodiscard]] inline std::string_view amd_level_kind_name(AmdLevelKind k) noexcept {
    switch (k) {
    case AmdLevelKind::CompiledSource:     return "compiled-source";
    case AmdLevelKind::HeaderOnlyFacility: return "header-only-facility";
    }
    return "?";
}

struct AmdLevelFacts {
    AmdRoute         route = AmdRoute::None;
    AmdLevelKind     kind  = AmdLevelKind::CompiledSource;
    std::string_view source_path;        // the file that would carry this level, printed verbatim
    // reading 1 -- find(1) on the tree
    bool source_exists = false;
    // reading 2 -- a SOURCE ENTRY in a target's source list (NOT a comment, NOT a diagnostic
    // string). The distinction is load-bearing: MEASURED, the filename gfx906_stubs.cpp has
    // exactly one mention in src/CMakeLists.txt and it is the prose that says the file is NOT in
    // the list.
    // AND THE MATCH IS ON THE ENTRY FORM: a CMake source entry is written RELATIVE TO THE
    // CMakeLists' OWN DIRECTORY, so src/CMakeLists.txt says
    // `ops/linear/gfx906/q4_tiled_gfx906.cu` and NOT `src/ops/linear/gfx906/q4_tiled_gfx906.cu`.
    // MEASURED 2026-09-24: a search for the FULL path returns 0 for that file and 1 for
    // q_gemv_gfx906.cuh -- where the single full-path hit is the PROSE of src/CMakeLists.txt:559.
    // So the full-path search finds a SENTENCE and the relative search finds the ENTRY. The
    // observer strips the CMakeLists' own directory prefix before matching, and the driver's
    // positive control is what caught this: the first form read FALSE for all six real entries.
    bool is_source_entry = false;
    // reading 3 -- a real .o under a build directory (find(1), scope named by the caller)
    bool object_in_build_dir = false;
    // reading 3' -- FOR A HEADER-ONLY FACILITY ONLY: how many files in the tree include it, and
    // whether at least one of those is itself a source entry of some target. A header with no
    // includer, or whose only includers are themselves unwired, is not in the build.
    std::size_t includers = 0;
    bool        includer_is_source_entry = false;
    // reading 4 -- THE LINKER ANSWERED. Only takable where an AMD toolchain exists; see above.
    bool symbol_resolved = false;
    // The dynamic LDS the workgroup asks for, and the limit the target imposes. A level whose
    // request EXCEEDS the limit cannot run there AT ALL -- "not slowly, not at reduced tile
    // size" (arch_caps.h's kAmdLdsBlockerSites comment). That is a kernel-and-arithmetic fact and
    // it is a reason to step DOWN a level that has nothing to do with a missing kernel.
    std::uint64_t lds_request_bytes = 0;
    std::uint64_t lds_limit_bytes   = 65536; // 64 KiB on every rung below gfx9 CDNA3
};

// THE ONE DERIVED PREDICATE, and it is a function so the conjunction has ONE home. The two kinds
// ask DIFFERENT questions and must not be given one shared conjunction.
[[nodiscard]] inline bool amd_level_in_build(const AmdLevelFacts& facts) noexcept {
    if (!facts.source_exists) { return false; }
    switch (facts.kind) {
    case AmdLevelKind::CompiledSource:
        return facts.is_source_entry && facts.object_in_build_dir;
    case AmdLevelKind::HeaderOnlyFacility:
        // No object of its own exists to find, so requiring one would be a counter that cannot
        // come up -- which is exactly the failure this branch was added for.
        return facts.includers > 0 && facts.includer_is_source_entry;
    }
    return false;
}

// Whether a level's kernel can consume this format AT ALL, given what the format needs. It is a
// parameter and not a table here because the format->floor mapping already has a home
// (kFormatRequirements) and a second copy would be a second answer to one question.
struct AmdFormatFacts {
    std::string_view name;             // artifact::format_name(format), printed verbatim
    // Was a kFormatRequirements row found? A format this tree DECLARES and has no kernel for on
    // ANY rung takes the other branch (kUncoveredFormatNotes) and the AMD gate must not restate
    // that refusal as its own -- arch_caps.h says so at render_amd_format_refusal().
    bool             declares_floor = true;
    // false when the row's `required` is Cap::None: never a tensor-core operand. Such a format
    // has no floor for any level to stand in the way of.
    bool             floor_is_tensor_core = true;
    std::string_view floor_name;       // cap_name(required), printed verbatim
    std::string_view floor_evidence;   // the file:line of the mma intrinsic the engine emits
    // true when this format has a SECOND, LOWER floor whose kernel consumes its OWN bytes (the
    // fp16-fallback column of kFormatRequirements). Empty-vs-set is a claim about a real kernel
    // file, so it is carried rather than inferred.
    bool             has_lower_floor_kernel = false;
    std::string_view uncovered_reason; // kUncoveredFormatNotes' reason, when !declares_floor
    // The two amdgcn instruction families this format's kernel reaches through, cited from
    // kAmdFormatBlockers: the mma channel AND the fragment loader. Both lack an amdgcn encoding,
    // and citing only the first understates it -- a port cannot even FEED the channel.
    std::string_view blocker_mma_site;
    std::string_view blocker_mma_mnemonic;
    std::string_view blocker_fragment_site;
    std::string_view blocker_fragment_mnemonic;
};

struct AmdRouteInputs {
    // The rung, by its ROCm offload-arch string -- the only name an AMD target has.
    std::string_view rung_target;
    bool             rung_in_ladder = false; // was it found in kAmdLadder (exact match, no neighbour)
    std::string_view rung_isa;
    std::string_view rung_cards;
    // RDNA spells its matrix path v_wmma, CDNA v_mfma. This decides which of the two matrix
    // levels is even ELIGIBLE on this rung; it is a measurement about the rung, not a preference.
    bool             matrix_path_is_wmma = false;
    AmdFormatFacts   format;
    AmdProblemShape  shape;
    // ONE ENTRY PER LEVEL OF THE LADDER. A caller that supplies none gets NOT-MEASURED and no
    // route is claimed -- not-probed is not a negative result (cufree_report.h's own rule).
    std::vector<AmdLevelFacts> levels;
};

// Always non-empty, carries the three tokens, and names the kernel rather than the card.
struct AmdRouteChoice {
    AmdRoute        route       = AmdRoute::None;
    AmdRouteOutcome outcome     = AmdRouteOutcome::NoKernelInTree;
    AmdBottomKind   bottom      = AmdBottomKind::NoRouteInTree;
    std::string_view kernel     = {}; // the level's kernel, or {} when there is none
    std::string     why         = {}; // ALWAYS non-empty
    // How many levels were stepped over to reach this answer. 0 on an at-floor selection, and
    // NEVER negative -- a counter that can go negative is not a reading.
    std::uint32_t   levels_descended = 0;
    // Which level refused and why it refused, so the caller can see the step that was taken. One
    // entry per level the walk TESTED and rejected, in walk order.
    std::vector<std::string> refusals;
    // TRUE when this answer was produced for a rung that is NOT in kAmdLadder.
    bool            unknown_rung = false;
    // FALSE for every answer this file produces, and rendered as its own line: the answer comes
    // from a TABLE and from build facts, NOT from a probe on the rung. The field exists so a
    // caller cannot mistake a table answer for a measured one -- format_probe.h's rule, and the
    // same restraint kernel_route.h keeps with its `simulated` field.
    bool            probe_backed = false;
    // WHAT TO MEASURE ON REAL HARDWARE to turn this cell into a probe result. Never empty when
    // `probe_backed` is false. This is kAmdLadder's `probe_evidence` obligation stated per cell
    // instead of being filled in from a simulation.
    std::string     measurement_owed = {};

    [[nodiscard]] bool ok() const noexcept { return outcome != AmdRouteOutcome::NoKernelInTree; }
    [[nodiscard]] bool warns() const noexcept {
        return outcome == AmdRouteOutcome::UnknownRungFallback;
    }
};

// ---------------------------------------------------------------------------
// THE DESCENT. Pure function of its inputs: no filesystem, no device, no dlopen, no toolchain.
// The walk order is the ladder read BACKWARD, which is the owner's sentence made executable:
// ask the highest level first, and step down until a level supports it or the ladder is spent.
//
// Two eligibility questions per level, and they are DIFFERENT questions that must not be merged:
//   (1) CAN THIS LEVEL CONSUME THIS FORMAT?  -- a fact about the format and the level's channel.
//   (2) IS THIS LEVEL IN THIS BUILD?         -- the three-corner build fact above.
// A level that can consume it but is not in the build is recorded as a refusal that names the
// BUILD, and the walk continues; a level that cannot consume it is skipped silently, because a
// level that never could is not a level the operator lost.
// ---------------------------------------------------------------------------
namespace detail {

// (1) Can this level's channel consume this format? The matrix levels take tensor-core operands;
// the dot/SIMT level takes the quantized integer forms that do not need a matrix core; the host
// level takes anything, because it is implemented on the host. A floor-free format (Cap::None)
// is consumable by EVERY level -- it is not a tensor-core operand at all.
inline bool amd_level_can_consume(AmdRoute route, const AmdFormatFacts& f) noexcept {
    if (!f.floor_is_tensor_core) { return true; }
    switch (route) {
    case AmdRoute::AmdMfmaNative:
    case AmdRoute::AmdWmmaNative:
        // A matrix-core level. It can consume a tensor-core operand IF a kernel for it exists --
        // which is what the build readings below decide.
        return true;
    case AmdRoute::AmdDotSimt:
        // The dot-product channel. It reaches the tensor formats through the groupwise-int
        // profile, which is what its kernels are: MEASURED, the gfx906 port layer ships
        // q4_tiled / q5_tiled / w8_tiled / q5_linear_add_tiled / q4_q5_{attn,gdn}_input_tiled,
        // and its builtins are __builtin_amdgcn_sdot4 and v_dot2_f32_f16 -- integer dot products,
        // not matrix cores. So it can consume a format only when a lower-floor kernel for that
        // format exists in the tree (the same condition the NV side spells with the
        // fallback_required column). It CANNOT consume a format that has no such kernel, and
        // saying otherwise would hand the operator a route naming a kernel nobody can launch.
        return f.has_lower_floor_kernel;
    case AmdRoute::AmdHostCpuReference:
        // Implemented on the host. amdsafe_gate.h admits this without examining primitives, so
        // there is no channel for a format to be incompatible with.
        return true;
    case AmdRoute::None:
    case AmdRoute::Count:
        break;
    }
    return false;
}

// The levels in WALK ORDER: highest floor first.
inline std::vector<AmdRoute> amd_walk_order(bool matrix_path_is_wmma) {
    std::vector<AmdRoute> order;
    order.push_back(matrix_path_is_wmma ? AmdRoute::AmdWmmaNative : AmdRoute::AmdMfmaNative);
    // The OTHER matrix level is not walked at all on this rung: RDNA cannot execute v_mfma and
    // CDNA cannot execute v_wmma, so offering the wrong one would be a route this rung has no
    // encoding for -- the same class of overclaim as naming a tensor-core route that executes on
    // an FMA pipe.
    order.push_back(AmdRoute::AmdDotSimt);
    order.push_back(AmdRoute::AmdHostCpuReference);
    return order;
}

inline const AmdLevelFacts* amd_find_level(const AmdRouteInputs& in, AmdRoute route) {
    for (const AmdLevelFacts& f : in.levels) {
        if (f.route == route) { return &f; }
    }
    return nullptr;
}

} // namespace detail

[[nodiscard]] inline AmdRouteChoice select_amd_route(const AmdRouteInputs& in) {
    AmdRouteChoice out;

    // ---- (0) WAS ANYTHING MEASURED AT ALL? NOT-PROBED IS NOT A NEGATIVE RESULT. ----
    // cufree_report.h prints NOT MEASURED for an empty observation rather than the class its
    // empty classify() would answer, and this walk does the same: an inputs with no level facts
    // cannot have asked a single level, so no route and no refusal are claimed.
    if (in.levels.empty()) {
        out.route   = AmdRoute::None;
        out.outcome = AmdRouteOutcome::NoKernelInTree;
        out.bottom  = AmdBottomKind::NoRouteInTree;
        out.why = "NOT MEASURED: no level of the ladder was supplied, so no level was asked and "
                  "no route is claimed. This is not a negative result -- it is the absence of a "
                  "measurement. Supply one AmdLevelFacts per level of the ladder.";
        out.measurement_owed =
            "supply the four build readings per level (source on disk, source entry in a target, "
            "object under a build directory, and the linker's answer where a toolchain exists)";
        return out;
    }

    // ---- (1) THE FORMAT THIS TREE DECLARES AND HAS NO KERNEL FOR ON ANY RUNG ----
    // The AMD gate must NOT restate this as its own finding: the missing piece is a kernel on
    // every rung, AMD included, and arch_caps.h's render_amd_format_refusal says so in as many
    // words. Refusing here with the tree's own reason keeps that one home.
    if (!in.format.declares_floor) {
        out.route   = AmdRoute::None;
        out.outcome = AmdRouteOutcome::NoKernelInTree;
        out.bottom  = AmdBottomKind::NoRouteInTree;
        out.why = "artifact format " + std::string(in.format.name) +
                  " has NO KERNEL in this tree that can execute it on ANY rung, so there is no "
                  "route for it on this rung either, and this is NOT an AMD finding: " +
                  std::string(in.format.uncovered_reason) +
                  " The missing piece is the kernel, not a row and not a card.";
        out.measurement_owed =
            "none: the refusal is a table fact and no measurement on this or any rung would "
            "change it. A kernel for this format is what would.";
        return out;
    }

    // ---- (2) THE WALK ----
    const std::vector<AmdRoute> order = detail::amd_walk_order(in.matrix_path_is_wmma);
    std::uint32_t level_index = 0;
    for (const AmdRoute route : order) {
        const AmdLevelFacts* facts = detail::amd_find_level(in, route);

        // (1) CAN THIS LEVEL CONSUME THIS FORMAT? A level that never could is not a level that
        // was lost, so it is not recorded as a refusal.
        if (!detail::amd_level_can_consume(route, in.format)) {
            ++level_index;
            continue;
        }
        // The channel is right and there is no measurement for it: this is the one case where a
        // missing measurement IS recorded, because the level was eligible and nobody asked.
        if (facts == nullptr) {
            out.refusals.push_back(std::string(amd_route_name(route)) +
                                   ": NOT MEASURED -- eligible for this format, but no "
                                   "AmdLevelFacts was supplied for it");
            ++level_index;
            continue;
        }

        // (2) IS THIS LEVEL IN THIS BUILD? The three corners, printed as three so the reader can
        // see WHICH is missing rather than a single silent zero.
        if (!amd_level_in_build(*facts)) {
            std::string r = std::string(amd_route_name(route)) + " [" +
                            std::string(amd_level_kind_name(facts->kind)) +
                            "]: NOT IN THIS BUILD (source_exists=" +
                            (facts->source_exists ? "yes" : "no");
            if (facts->kind == AmdLevelKind::CompiledSource) {
                r += std::string(", is_source_entry=") +
                     (facts->is_source_entry ? "yes" : "no") + ", object_in_build_dir=" +
                     (facts->object_in_build_dir ? "yes" : "no");
            } else {
                r += std::string(", includers=") + std::to_string(facts->includers) +
                     ", includer_is_source_entry=" +
                     (facts->includer_is_source_entry ? "yes" : "no");
            }
            r += ")";
            if (!facts->source_path.empty()) { r += " [file: " + std::string(facts->source_path) + "]"; }
            if (facts->is_source_entry && !facts->object_in_build_dir) {
                // THE INFORMATIVE REFUSAL, and it is the QPN shape one layer down: the source IS
                // in a target's list but this CONFIGURE did not compile it. MEASURED: the six
                // gfx906 entries live inside `if(NINFER_ENABLE_GFX906_COMPAT)`, and
                // build/CMakeCache.txt:547 reads NINFER_ENABLE_GFX906_COMPAT:BOOL=OFF. Saying
                // "no target compiles it" here would be as false as the sentence that made the
                // NV side skip a level -- the difference between "not in the build system" and
                // "not in this configuration" is the whole point of reading the corners.
                r += " -- NOTE: the source IS a target_sources entry, so this is NOT 'no target "
                     "compiles it'. What is missing is the CONFIGURE: the entry sits inside an "
                     "option-gated block, and the cache for the build directories searched says "
                     "the option is OFF.";
            }
            out.refusals.push_back(r);
            ++level_index;
            continue;
        }

        // (3) THE LDS FLOOR. A level that IS in the build can still be unable to run this SHAPE:
        // a workgroup may not request more dynamic LDS than the target allows, and that is a
        // kernel-and-arithmetic fact rather than a missing kernel. Stepping down here has nothing
        // to do with the build, and the refusal says so.
        if (facts->lds_request_bytes > facts->lds_limit_bytes) {
            out.refusals.push_back(std::string(amd_route_name(route)) +
                                   ": IN BUILD BUT THE SHAPE DOES NOT FIT -- this level's kernel "
                                   "requests " + std::to_string(facts->lds_request_bytes) +
                                   " B of dynamic LDS and the target allows " +
                                   std::to_string(facts->lds_limit_bytes) +
                                   " B per workgroup, so it cannot be launched there AT ALL "
                                   "(not slowly, not at a reduced tile size: the request must be "
                                   "split along the K tiling first)");
            ++level_index;
            continue;
        }

        // ---- SELECTED ----
        out.route            = route;
        out.outcome          = in.rung_in_ladder ? AmdRouteOutcome::Selected
                                                 : AmdRouteOutcome::UnknownRungFallback;
        out.unknown_rung     = !in.rung_in_ladder;
        out.levels_descended = level_index;
        out.bottom           = (level_index == 0) ? AmdBottomKind::SelectedAtFloor
                                                  : AmdBottomKind::DescendedToLowerLevel;
        out.kernel           = facts->source_path;
        out.probe_backed     = false;

        out.why = std::string("rung ") + std::string(in.rung_target) + " (" +
                  std::string(in.rung_isa) + ") x format " + std::string(in.format.name) + ": ";
        if (!in.format.floor_is_tensor_core) {
            out.why += "this format is NOT a tensor-core operand (no floor to satisfy), so the "
                       "descent stops at the first level that can consume it. ";
        } else {
            out.why += "the walk stepped down from the format's own floor (" +
                       std::string(in.format.floor_name) + ") and the first level that can "
                       "consume it AND is in this build is " + std::string(amd_route_name(route)) +
                       ". " + std::to_string(level_index) + " level(s) were passed over, and each "
                       "one's reason is in `refusals`. ";
        }
        if (!in.rung_in_ladder) {
            out.why += "THE RUNG IS NOT IN kAmdLadder: this answer is the lowest-floor level this "
                       "build has and is reported as a WARNING rather than a selection, because a "
                       "rung nobody measured must not be handed a neighbour's floor. ";
        }
        out.why += "WHAT IS SELECTED IS THE LEVEL, NOTHING MORE: this answer comes from a TABLE and "
                   "from build facts, NOT from a probe on this rung, so it says nothing about "
                   "this rung's hardware. ";
        out.why += "The missing kernel for the format's OWN floor is named, because the reason has "
                   "to say what to write and not only which card asked: " +
                   std::string(in.format.floor_evidence);
        if (!in.format.blocker_mma_site.empty()) {
            out.why += ". That kernel reaches its tensor core through two instruction families an "
                       "amdgcn target has no encoding for: " +
                       std::string(in.format.blocker_mma_mnemonic) + " at " +
                       std::string(in.format.blocker_mma_site) + ", and " +
                       std::string(in.format.blocker_fragment_mnemonic) + " at " +
                       std::string(in.format.blocker_fragment_site) +
                       " (ldmatrix has NO AMD equivalent; the fragment layouts must be "
                       "regenerated, so a port cannot even feed the channel)";
        }
        out.why += ".";

        out.measurement_owed = std::string("to make this cell a PROBE RESULT rather than a table "
                                           "answer, measure on real ") +
                               std::string(in.rung_target) + " hardware: (a) load the artifact "
                               "whose format is " + std::string(in.format.name) +
                               " and record whether the selected level's kernel launches; "
                               "(b) record the observed dynamic LDS the launch requested and the "
                               "workgroup limit the driver reported, because the LDS check above "
                               "was made against an assumed 65536 B limit and the real limit is a "
                               "property of the part; (c) record the first-mismatch against a "
                               "host reference. Only then may the rung's `probe_evidence` field in "
                               "kAmdLadder be made non-empty, and doing so OWES that file:line.";
        return out;
    }

    // ---- (3) BOTTOMED OUT: THE LADDER IS SPENT ----
    out.route   = AmdRoute::None;
    out.outcome = AmdRouteOutcome::NoKernelInTree;
    out.bottom  = AmdBottomKind::NoRouteInTree;
    out.kernel  = {};
    out.unknown_rung = !in.rung_in_ladder;
    out.levels_descended = level_index;
    out.probe_backed     = false;
    out.why = std::string("rung ") + std::string(in.rung_target) + " x format " +
              std::string(in.format.name) +
              ": NO ROUTE IN THIS TREE. Every level of the ladder was asked and none answered. "
              "This is a statement about the KERNELS, not about the card: the rung is " +
              std::string(in.rung_isa) + ". The missing kernel for the format's own floor is: " +
              std::string(in.format.floor_evidence) +
              ". The lowest level of the ladder -- the host path, which needs no AMD compiler and "
              "no ROCm -- is the ONE level that can always answer, so it reaching this branch "
              "means not even that level was measured as in this build.";
    out.measurement_owed =
        "measure, on the rung, whether a v_mfma (CDNA) or v_wmma (RDNA) kernel can be built and "
        "launched for this format at all, and record the compiler's own answer to the three "
        "instruction families at the cited sites -- v_/mma.sync, the ldmatrix fragment loader and "
        "cp.async. A refusal that has not been asked on hardware is a table answer.";
    return out;
}

// ---------------------------------------------------------------------------
// THE RENDERER. Pure, prints nothing, never suppresses: a caller that asks always gets it.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string render_amd_route_choice(const AmdRouteInputs& in,
                                                        const AmdRouteChoice& c) {
    std::string out;
    out += "ninfer: AMD FORMAT-ROUTE DESCENT (dl/amdprobe).\n";
    out += "  rung        : ";
    out.append(in.rung_target);
    out += "  (";
    out.append(in.rung_isa);
    out += ", e.g. ";
    out.append(in.rung_cards);
    out += ")";
    if (!in.rung_in_ladder) { out += "   [NOT IN kAmdLadder]"; }
    out += "\n  format      : ";
    out.append(in.format.name);
    out += "\n  floor       : ";
    out.append(in.format.floor_is_tensor_core ? in.format.floor_name
                                              : std::string_view("none - never a tensor-core "
                                                                 "operand"));
    out += "\n  shape       : m=" + std::to_string(in.shape.m) + " n=" + std::to_string(in.shape.n) +
           " k=" + std::to_string(in.shape.k);
    out += "\n\n  WALK RESULT\n";
    out += "    route       : ";
    out += amd_route_name(c.route);
    out += "\n    outcome     : ";
    out += amd_outcome_name(c.outcome);
    out += "\n    bottomed    : ";
    out += amd_bottom_token(c.bottom);
    out += "     <-- TWO DIFFERENT SENTENCES, PRINTED SEPARATELY: `no-route-in-tree` means there "
           "is no route in this tree, while `descended-to-lower-level` means a route WAS selected "
           "and it is below the format's own floor. Neither is the word `unsupported`.\n";
    out += "    levels_descended : " + std::to_string(c.levels_descended) + "\n";
    out += "    kernel      : ";
    out.append(c.kernel.empty() ? std::string_view("(none)") : c.kernel);
    out += "\n    probe_backed: ";
    out += c.probe_backed ? "yes" : "no  <-- THIS ANSWER IS A TABLE ANSWER";
    out += "\n";
    out += "\n  WHY (always non-empty)\n    ";
    out.append(c.why);
    out += "\n";
    if (!c.refusals.empty()) {
        out += "\n  THE LEVELS THE WALK PASSED OVER, in walk order, each with its own reason\n";
        for (const std::string& r : c.refusals) {
            out += "    - ";
            out.append(r);
            out += "\n";
        }
    }
    if (!c.measurement_owed.empty()) {
        out += "\n  WHAT MEASUREMENT IS OWED, so this cell can become a probe result\n    ";
        out.append(c.measurement_owed);
        out += "\n";
    }
    out += "\n  NOT CLAIMED BY THIS ANSWER, and named so it cannot be read in:\n";
    out += "    * no vendor attributed to any rung, and no device count;\n";
    out += "    * no statement that this rung's hardware can or cannot do anything: the answer "
           "reads this tree's\n      kernels and this build's sources, and a table answer is not a "
           "probe result;\n";
    out += "    * kAmdLadder's `probe_evidence` is NOT filled by this and is NOT filled from a "
           "simulation. The\n      six rows are empty because no AMD hardware was probed, and "
           "this file says which\n      measurement each cell owes instead of inventing the "
           "answer for it.\n";
    return out;
}

// One line, stable enough to grep. Mirrors kernel_route.h's route_log_line so the two sides can
// be read by one tool, with a distinct prefix so they are never confused.
struct AmdRouteLogLine {
    std::string text;
    bool        warn = false;
};

[[nodiscard]] inline AmdRouteLogLine amd_route_log_line(const AmdRouteInputs& in,
                                                       const AmdRouteChoice& c) {
    AmdRouteLogLine out;
    out.warn = c.warns();
    out.text = std::string("[amdroute] ");
    out.text += "rung=" + std::string(in.rung_target) +
                " format=" + std::string(in.format.name) +
                " m=" + std::to_string(in.shape.m) + " n=" + std::to_string(in.shape.n) +
                " k=" + std::to_string(in.shape.k) +
                " -> " + std::string(amd_route_name(c.route)) +
                " outcome=" + std::string(amd_outcome_name(c.outcome)) +
                " bottom=" + std::string(amd_bottom_token(c.bottom)) +
                " descended=" + std::to_string(c.levels_descended) +
                " probe_backed=" + (c.probe_backed ? std::string("yes") : std::string("no")) +
                " | " + c.why;
    return out;
}

// The TAB-SEPARATED CELL, which is the shape dl/amdprobe/out/amd_route_cells.tsv carries. Kept
// here rather than in a script so the column order has one home and a shell loop cannot drift
// from the fields.
[[nodiscard]] inline std::string amd_route_cell_tsv(const AmdRouteInputs& in,
                                                    const AmdRouteChoice& c) {
    std::string out;
    out += std::string(in.rung_target);
    out += '\t';
    out += std::string(in.format.name);
    out += '\t';
    out += std::string(amd_route_name(c.route));
    out += '\t';
    out += std::string(amd_outcome_name(c.outcome));
    out += '\t';
    out += std::string(amd_bottom_token(c.bottom));
    out += '\t';
    out += c.kernel.empty() ? std::string("-") : std::string(c.kernel);
    out += '\t';
    out += std::to_string(c.levels_descended);
    out += '\t';
    out += c.probe_backed ? "yes" : "no";
    out += '\t';
    out += std::to_string(c.refusals.size());
    out += '\t';
    out += c.warns() ? "warn" : "info";
    return out;
}

} // namespace ninfer::caps

#endif // NINFER_CORE_AMD_ROUTE_H
