#pragma once

// ===========================================================================
// nvfp4_ldm_free.cuh -- THE LDMATRIX-FREE FRAGMENT LOADER, AND ITS NAMED SELECTION.
// (line: dl/nvfp4emu. Consumer #1: ops/kernel/gqa_attention_decode_nvfp4.cuh.)
//
// WHY THIS FILE EXISTS AT ALL, IN THE TREE'S OWN WORDS. `ldmatrix` is the one instruction
// family this tree has ALREADY measured a blocker on, and it is the blocker that is NOT an
// mma instruction. src/core/arch_caps.h:1893 says it in its own voice:
//
//   "ldmatrix -- the four fragment loaders. There is no AMD equivalent to translate them TO,"
//
// and src/core/arch_caps.h:1974 / :2002 name the consequence and the boundary:
//
//   "ldmatrix -> NO EQUIVALENT AT ALL; fragment layouts have to be regenerated."
//   "EXTERNAL-UNPROBED. ldmatrix has NO AMD equivalent; the fragment layouts it produces must
//    ... an ldmatrix loader as well as an mma.sync instruction."
//
// and src/core/arch_caps.h:2354 renders it to the operator:
//
//   " -- ldmatrix has NO AMD equivalent; the layouts must be regenerated)"
//
// This header IS that regeneration, for the NVFP4 kernels: the SAME fragment bits, produced by
// warp shuffles plus ordinary shared-memory loads, which both targets have.
//
// ---------------------------------------------------------------------------
// WHAT IT PROVES, AND WHAT IT CANNOT. THIS PAIR IS THE POINT OF THE FILE.
//
//   PROVED BY CONSTRUCTION (any box, any target, no measurement needed): the register values are
//   the values `ldmatrix.sync.aligned.m8n8.xN[.trans].shared.b16` produces. This is a
//   DATA-MOVEMENT substitution and not an arithmetic one: the `mma_*` instruction that consumes
//   the fragment is NOT touched and receives bit-identical operands. So "the two arms agree bit
//   for bit" is a property of the construction, and a measurement of it is a CHECK of the
//   construction rather than the thing that establishes it.
//
//   NOT PROVED, AND NOT PROVABLE ON A BOX WITH TENSOR CORES: that an NVFP4 kernel RUNS on a
//   device with no tensor cores. This header removes the `ldmatrix` dependency (sm_75,
//   src/ops/common/mma.cuh:27-29) and NOT the `mma` dependency -- the NVFP4 kernels still call
//   `mma_nvfp4_e4m3` (needs NINFER_MMA_HAS_KIND_MXF4NVF4 == sm_120a/121a) and `mma_bf16` (needs
//   NINFER_MMA_HAS_M16N8K16_TC == sm_80); see src/ops/common/mma.cuh:27-57 and :269-295. On a
//   device with no tensor cores those helpers still take the `unsupported_instruction_trap()`
//   arm. This is NECESSARY AND NOT SUFFICIENT, and it is EXTERNAL-UNPROBED in exactly the sense
//   src/core/vendor_sim.h:26-35 gives that word for its own table: only the decision this file
//   makes is covered.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#include <cuda_runtime.h>

#include "core/announce_once.h"

namespace ninfer::ops {

// ===========================================================================
// THE DEVICE SIDE: THE FOUR LOADERS, REGENERATED
// ===========================================================================
//
// The calling convention is the SAME as the instruction's: every lane passes the address of the
// 16-byte row of the tile that IT owns; lanes 8*M..8*M+7 own the rows of matrix M. A site that
// already computes `smem_addr(smem + row * stride + col)` to hand to `ldmatrix_x4` hands the
// POINTER `smem + row * stride + col` here instead -- no other change, which is why every
// conversion is one line.
//
// A 64-bit GENERIC pointer is shuffled rather than a shared-window address: `cvta.shared` has no
// AMD spelling (src/ops/common/memory.cuh:94-99's PARAMXLATE2 note records that `cvta` is
// undeclared on ROCm), and the generic form is the one both targets already have. Every lane of a
// warp is in one CTA, so the window bits are identical on both sides of the shuffle.
__device__ __forceinline__ const std::uint8_t* nvfp4_ldm_free_row_ptr(const void* rowptr,
                                                                     int src_lane) {
    const unsigned long long bits = reinterpret_cast<unsigned long long>(rowptr);
    const unsigned long long src  = __shfl_sync(0xffffffffu, bits, src_lane);
    return reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(src));
}

// One 8x8 .b16 matrix of an xN fragment, as a 32-bit register.
//
//   not transposed: lane i holds the 32 bits at row (i>>2), byte (i&3)*4.
//   transposed:     lane i holds the two .b16 at rows (i&3)*2 and (i&3)*2+1, byte (i>>2)*2 of
//                   each -- i.e. the tile in memory is transposed before it is handed out.
//
// `M` selects which eight lanes supply the rows (8*M + r) and MUST be a compile-time constant.
template <bool Trans, int M>
__device__ __forceinline__ unsigned nvfp4_ldm_free_reg(const void* rowptr, int lane) {
    const int li = lane & 31;
    if constexpr (Trans) {
        const std::uint8_t* ra = nvfp4_ldm_free_row_ptr(rowptr, 8 * M + (li & 3) * 2);
        const std::uint8_t* rb = nvfp4_ldm_free_row_ptr(rowptr, 8 * M + (li & 3) * 2 + 1);
        const unsigned lo = *reinterpret_cast<const unsigned short*>(ra + (li >> 2) * 2);
        const unsigned hi = *reinterpret_cast<const unsigned short*>(rb + (li >> 2) * 2);
        return lo | (hi << 16);
    } else {
        const std::uint8_t* r = nvfp4_ldm_free_row_ptr(rowptr, 8 * M + (li >> 2));
        return *reinterpret_cast<const unsigned*>(r + (li & 3) * 4);
    }
}

// `ldmatrix.sync.aligned.m8n8.xN[.trans].shared.b16`, assembled without the instruction.
// N is the `.xN` width (2 or 4); `rowptr` is this lane's row pointer. Like the instruction it
// replaces it is `.aligned`: the WHOLE warp must be active, because the shuffles use the full
// mask. Every call site in this tree is at warp-uniform control flow, which is what the
// instruction required too, so this is not a new precondition.
template <bool Trans, int N>
__device__ __forceinline__ void nvfp4_ldm_free(unsigned (&frag)[N], const void* rowptr, int lane) {
    static_assert(N == 2 || N == 4, "the .xN widths this tree uses are x2 and x4");
    frag[0] = nvfp4_ldm_free_reg<Trans, 0>(rowptr, lane);
    frag[1] = nvfp4_ldm_free_reg<Trans, 1>(rowptr, lane);
    if constexpr (N == 4) {
        frag[2] = nvfp4_ldm_free_reg<Trans, 2>(rowptr, lane);
        frag[3] = nvfp4_ldm_free_reg<Trans, 3>(rowptr, lane);
    }
}

// ===========================================================================
// THE HOST SIDE: THE SELECTION, IN THE SHAPE OF SimtFfmaSelection
// (ops/kernel/gqa_attention_simt_ffma.cuh:240-405), FOR THE REASONS THAT FAMILY RECORDS.
//
// Three properties are copied deliberately and are the whole reason this block is not a bool:
//   * an ANSWER and the REASON for it are different questions, so the reason has a named value
//     and text (Nvfp4FragLdSelection / nvfp4_frag_ld_selection_text());
//   * the OPERATOR OVERRIDE is factored OUT of the decision so the two can be reported
//     separately -- a report that says "the probe answered X" while the route is X's opposite is
//     the same lie in a smaller font (nvfp4_frag_ld_env_override());
//   * an UNPROBED run is never presented as a measured one (UnprobedBuildDefault).
// ===========================================================================
enum class Nvfp4FragLdProbeAnswer : std::uint8_t {
    // The probe has not run on the bound device. The selector falls back to
    // NINFER_NVFP4_FRAG_LD_DEFAULT_ON and SAYS SO through nvfp4_frag_ld_selection().
    NotProbed = 0,
    // MEASURED: the ldmatrix-free loader ran on this device and every fragment it produced was
    // bit-identical to the one `ldmatrix` produced from the same shared memory, AND this device
    // has no working `ldmatrix` path. On a box WITH tensor cores this is NOT the truthful answer
    // -- see RanButLdmatrixArmAlsoHere.
    RanOk = 1,
    // MEASURED: the same bit-for-bit agreement, on a device that ALSO has the `ldmatrix` arm.
    // The arm is available and is NOT the default, and that is the CORRECT default on such a
    // device: the tensor-core loader is one instruction where this one is a shuffle plus a load,
    // so this arm buys PORTABILITY and not speed, and the tree's default must not move for it.
    RanButLdmatrixArmAlsoHere = 2,
    // MEASURED: a fragment did not match what `ldmatrix` produced, or the arm did not launch.
    // A measured "do not take this arm on this device".
    Failed = 3,
};

[[nodiscard]] inline std::string_view nvfp4_frag_ld_probe_answer_name(
    Nvfp4FragLdProbeAnswer answer) {
    switch (answer) {
    case Nvfp4FragLdProbeAnswer::NotProbed: return "not probed";
    case Nvfp4FragLdProbeAnswer::RanOk:
        return "the ldmatrix-free loader ran on this device, every fragment matched the ldmatrix "
               "arm bit for bit, and this device has no working ldmatrix path";
    case Nvfp4FragLdProbeAnswer::RanButLdmatrixArmAlsoHere:
        return "the ldmatrix-free loader ran on this device and every fragment matched the "
               "ldmatrix arm bit for bit, but the ldmatrix arm also works here";
    case Nvfp4FragLdProbeAnswer::Failed:
        return "a fragment did not match the ldmatrix arm, or the arm did not launch (measured: "
               "do not take this arm on this device)";
    }
    return "?";
}

// WHERE THE ANSWER LIVES, AND WHY IT COSTS NO LINK EDGE. One object per process, shared by every
// TU, with no new symbol in any library: a function-local static inside an `inline` function is
// the ODR's single instance, so the writer (a launcher or a probe) and every reader address the
// SAME object (ops/kernel/gqa_attention_simt_ffma.cuh:277-291 says the same thing about its own).
[[nodiscard]] inline Nvfp4FragLdProbeAnswer& nvfp4_frag_ld_probe_state() {
    static Nvfp4FragLdProbeAnswer answer = Nvfp4FragLdProbeAnswer::NotProbed;
    return answer;
}

// The only writer. Called once per device, before any launch asks.
inline void nvfp4_frag_ld_publish_probe_answer(Nvfp4FragLdProbeAnswer answer) {
    nvfp4_frag_ld_probe_state() = answer;
}

// The env override, factored OUT of the decision so that the two can be reported separately.
// Returns -1 when unset, else 0/1. Only a single-character value is accepted, so `=01` and `=1x`
// are NOT overrides -- a half-read flag must not become a route.
[[nodiscard]] inline int nvfp4_frag_ld_env_override() {
    const char* env = std::getenv("NINFER_NVFP4_FRAG_LD");
    if (env != nullptr && env[0] != '\0' && env[1] == '\0') {
        if (env[0] == '1') { return 1; }
        if (env[0] == '0') { return 0; }
    }
    return -1;
}

enum class Nvfp4FragLdSelection : std::uint8_t {
    ForcedOn,             // NINFER_NVFP4_FRAG_LD=1
    ForcedOff,            // NINFER_NVFP4_FRAG_LD=0
    ProbeRanOk,           // measured: it ran here, matched bit for bit, no ldmatrix path here
    ProbeNotTheArm,       // measured: it ran here and matched, but ldmatrix also works here
    ProbeFailed,          // measured: do not take this arm on this device
    UnprobedBuildDefault, // NOT measured: the probe has not run (NINFER_NVFP4_FRAG_LD_DEFAULT_ON)
};

[[nodiscard]] inline const char* nvfp4_frag_ld_selection_text(Nvfp4FragLdSelection how) {
    switch (how) {
    case Nvfp4FragLdSelection::ForcedOn:
        return "NINFER_NVFP4_FRAG_LD=1 (operator override): the ldmatrix-free fragment loader";
    case Nvfp4FragLdSelection::ForcedOff:
        return "NINFER_NVFP4_FRAG_LD=0 (operator override): the ldmatrix fragment loader";
    case Nvfp4FragLdSelection::ProbeRanOk:
        return "MEASURED by the capability probe: the ldmatrix-free fragment loader ran on this "
               "device, every fragment matched the ldmatrix arm bit for bit, and this device has "
               "no working ldmatrix path";
    case Nvfp4FragLdSelection::ProbeNotTheArm:
        return "MEASURED by the capability probe: the ldmatrix-free fragment loader matched bit "
               "for bit on this device, but the ldmatrix arm also works here, so the ldmatrix "
               "loader is what runs (this arm buys portability, not speed)";
    case Nvfp4FragLdSelection::ProbeFailed:
        return "MEASURED by the capability probe: a fragment did not match or the arm did not "
               "launch on this device, so the ldmatrix loader is what runs";
    case Nvfp4FragLdSelection::UnprobedBuildDefault:
        return "NOT MEASURED: the capability probe has not run on this device, so this is the "
               "BUILD default (NINFER_NVFP4_FRAG_LD_DEFAULT_ON)";
    }
    return "?";
}

// ORDER: operator override, then the PROBE, then the build default -- the same three steps and
// the same order as the SIMT FFMA family's selector.
[[nodiscard]] inline Nvfp4FragLdSelection nvfp4_frag_ld_selection() {
    const int override_value = nvfp4_frag_ld_env_override();
    if (override_value == 1) { return Nvfp4FragLdSelection::ForcedOn; }
    if (override_value == 0) { return Nvfp4FragLdSelection::ForcedOff; }
    switch (nvfp4_frag_ld_probe_state()) {
    case Nvfp4FragLdProbeAnswer::RanOk: return Nvfp4FragLdSelection::ProbeRanOk;
    case Nvfp4FragLdProbeAnswer::RanButLdmatrixArmAlsoHere:
        return Nvfp4FragLdSelection::ProbeNotTheArm;
    case Nvfp4FragLdProbeAnswer::Failed: return Nvfp4FragLdSelection::ProbeFailed;
    case Nvfp4FragLdProbeAnswer::NotProbed: return Nvfp4FragLdSelection::UnprobedBuildDefault;
    }
    return Nvfp4FragLdSelection::UnprobedBuildDefault;
}

// The host-side switch. It is a NAMED route choice and not a gate: it can only ever move a launch
// onto this loader, never past a refusal. The dtype/geometry guards all run before it.
[[nodiscard]] inline bool nvfp4_frag_ld_selected() {
    switch (nvfp4_frag_ld_selection()) {
    case Nvfp4FragLdSelection::ForcedOn:
    case Nvfp4FragLdSelection::ProbeRanOk: return true;
    case Nvfp4FragLdSelection::ForcedOff:
    case Nvfp4FragLdSelection::ProbeNotTheArm:
    case Nvfp4FragLdSelection::ProbeFailed: return false;
    case Nvfp4FragLdSelection::UnprobedBuildDefault:
#if defined(NINFER_NVFP4_FRAG_LD_DEFAULT_ON)
        return true;
#else
        return false;
#endif
    }
    return false;
}

// THE ANNOUNCEMENT IS KEYED ON THE SELECTION, NOT ON A BOOL: a second, DIFFERENT selection is
// announced too. A bare `static bool` would announce the first value it saw and then go silent for
// every later one -- src/core/announce_once.h:22-41 names that weaker shape, and this reuses that
// header's corrected set rather than adding a second implementation of it. Returns true when this
// selection had not been announced before.
[[nodiscard]] inline bool nvfp4_frag_ld_announce_this_selection() {
    return ninfer::detail::announce_once_keyed(nvfp4_frag_ld_selection());
}

// THE ONE THING A LAUNCHER READS: the flag the kernel takes, plus the announcement. The printed
// line is generated from the SELECTION VALUE and not from the bool, so "which loader" and "on
// whose word" are both in the log.
[[nodiscard]] inline int nvfp4_frag_ld_launch_flag() {
    const Nvfp4FragLdSelection how = nvfp4_frag_ld_selection();
    if (ninfer::detail::announce_once_keyed(how)) {
        std::fprintf(stderr, "[ninfer] nvfp4 fragment loader: %s\n",
                     nvfp4_frag_ld_selection_text(how));
    }
    return nvfp4_frag_ld_selected() ? 1 : 0;
}

} // namespace ninfer::ops
