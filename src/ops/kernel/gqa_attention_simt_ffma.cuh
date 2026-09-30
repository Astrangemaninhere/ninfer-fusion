#pragma once

// ninfer::ops - THE PRE-VOLTA ATTENTION PATH: FFMA + online softmax.
//
// WHY THIS FILE EXISTS. Measured on HEAD 3944a53 by the line that handed this over
// (dl/oldnvidia/REPORT.md section 3, evidence recon/r5_attention_census.txt): EVERY attention
// kernel in this tree -- decode {bf16,fp8,i8,iso3,nvfp4}, prefill {bf16,i8,nvfp4} and the whole
// softmax_attention/dense/causal_cache set -- contains at least one `mma` and one `ldmatrix`
// reference, and a tree-wide search for `simt|scalar` inside src/ops/kernel and
// src/ops/softmax_attention returns nothing. The instruction floors (recon/isaprobe/, a real
// per-rung compile of the tree's own emitters) are: `ldmatrix` needs sm_75, `mma` needs sm_70
// (`mma.m16n8k16`, `cp.async` and `cvt.bf16x2.f32` need sm_80). Maxwell (sm_52) and Pascal
// (sm_60/61) have no tensor cores at all, so none of those kernels can run there: attention,
// not GEMM, is the binding constraint on every pre-Volta card.
//
// WHAT THIS FILE IS. One decode path and one prefill path, both pure ALU: bf16 loads converted
// to fp32, FFMA dot products, `ex2.approx.f32` softmax, fp32 accumulation, and a warp shuffle
// reduction. No mma, no ldmatrix, no cp.async, no bf16 arithmetic, no TMA -- so the family is
// not arch-specific at all and is compiled into EVERY rung (see the floor note below).
//
// WHAT IT IS NOT. It is not a replacement for the tensor-core arms, and it is not
// performance-complete: it is single-buffered and it does not pipeline. On a Maxwell part the
// honest expectation is bandwidth-bound SIMT decode, not near-tensor-core rates; nothing in this
// file claims a speed, and no number for any pre-Volta card is measured anywhere in this line's
// work (this box is an RTX 5090 D / sm_120, which has no such card -- see REPORT.md's boundary).
//
// STRUCTURE: it MIRRORS the tensor-core family rather than replacing it.
//   * decode: same partial layout (gqa_partial_acc_index / gqa_partial_stat_index), the same
//     split partition (gqa_small_t_split_range, the family's one definition of "which keys does
//     split s own"), the same neutral-partial convention (m = -inf, l = 0, acc = 0) and the same
//     guards. The shared reducer gqa_attention_small_t_reduce_output_kernel is therefore reused
//     VERBATIM and unchanged. `write_neutral` is preserved as a function.
//   * prefill: same signature and the same bottom-right causal alignment as
//     gqa_attention_prefill_bf16_kernel.
//   * the GRID differs from the tensor-core arms on purpose: this family tiles the rows as
//     `WarpsPerCta * kGqaSimtFfmaRowsPerWarp` per CTA (16), it does not use the
//     `tokens * group_size <= WarpsPerCta * 16` capacity rule the tensor-core bf16 partial uses,
//     and its decode grid therefore carries a row-group factor in blockIdx.x
//     (KVHeads * ceil(row_count / Br)). Both launcher arms compute their own grid, so the
//     difference cannot leak into a caller. The partial layout does not depend on the grid.
//
// CORRECTNESS NOTE (what the sm_120 harness caught, and why this file looks like this). The first
// version of this family kept the per-row online-softmax state in SHARED memory with a row stride
// of `VecD`, i.e. `state[2 + i]`. That was wrong twice over, and both were measured on this box
// against the tensor-core arm and a double-precision CPU reference (stage/n1_numerics.cu,
// stage/n2_debug.cu, stage/n3_dump.cu, stage/n6_instr.cu, recon/n1_numerics.txt):
//   1. the state row must span the whole head, not one lane's `VecD` dims: all 32 lanes of a warp
//      hold different d of the SAME row, so `state[2 + i]` made every lane overwrite every other
//      lane's accumulator and the value that survived was the last writer's (a race, measured:
//      the state came back holding lane 24's and lane 16's dims);
//   2. the decode kernel initialized that state nowhere at all, so its first tile read
//      uninitialized shared memory (measured: `-inf` and bf16 -inf patterns in partial_acc).
// The fix is structural rather than a patch: the state now lives in REGISTERS, one entry per
// (row, lane-owned dim) -- `acc[R][VecD]`, `m[R]`, `l[R]` -- which removes the shared-memory
// state, the initialization pass and the race in one move, and drops the decode CTA's shared
// memory to the two K/V tiles alone. The row count per CTA (WarpsPerCta * R = 16) is what keeps
// that in registers.
//
// THE FLOOR. Measured with stage/p1_intrinsics.sh (a real per-rung compile of exactly these
// intrinsics, on nvcc 12.8 for sm_50/52/61/70 and nvcc 13.3 for sm_120a): `__bfloat162float`,
// `__float2bfloat16`, `ex2.approx.f32`, `__shfl_xor_sync`, `__frcp_rn`, `__fmaf_rn` and a
// 32 KiB shared-memory tile all assemble from sm_50 up. So the fence here is not "which arch
// can assemble it" -- it is "which arch has no tensor cores", which is what may SELECT it:

// The arch below which this family is the only attention route that exists at all.
//
// CORRECTED 2026-09-19 (dl/simtpack): this said `< 700` and that number is UNDER-WIDE. The
// boundary is set by the *attention* arms, not by the cheapest `mma` spelling in the tree. The
// arms this family competes with reach, per `dl/qpnguard/GUARDS.md` section 2.4 and section 3
// (the per-helper floors of `ops/common/mma.cuh` and `ops/common/memory.cuh`, each established by
// a real per-rung compile):
//   `ldmatrix`              >= 750   (NINFER_MMA_HAS_LDMATRIX)
//   `mma.m16n8k16` family   >= 800   (NINFER_MMA_HAS_M16N8K16_TC)  <- what the attention arms use
//   `cp.async`/`cp_commit`  >= 800   (NINFER_MEMORY_HAS_CP_ASYNC)  <- every tc decode arm stages
//   `mma.m8n8k4`            >= 700   (only the GEMM/QPN bands, not attention)
// A tensor-core *attention* decode or prefill therefore needs 800, not 700: below 800 every one
// of those kernels' helpers is the `unsupported_instruction_trap()` arm. So on sm_70 and sm_75
// the tensor-core arms DO exist as routes, and declaring this family "the only attention route
// that exists at all" there is wrong.
//
// WHAT CHANGES, STATED HONESTLY. Nothing at run time, because this macro is DEAD: a widened
// census over 3159 in-tree files (all of `src/`, `include/`, `tools/`, `tests/`, every
// `.cu/.cuh/.h/.hpp/.cpp/.cc/.cxx/.c/.py/.cmake/.txt/.md/.sh/.json`) finds
// `NINFER_ATTENTION_SIMT_FFMA_IS_THE_ONLY_ATTENTION` at exactly ONE occurrence -- this
// `#define`. Nothing reads it, so its wrong number cannot mis-route anything today. It is
// corrected because it is the floor of a *route* question, and the next consumer to read this
// macro would inherit the wrong constant with no way to notice.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#define NINFER_ATTENTION_SIMT_FFMA_IS_THE_ONLY_ATTENTION 1
#endif

// THE UNPROBED FALLBACK, AND (as of the probe change below) NOTHING ELSE.
//
// This macro used to BE the decision. It is now only the answer to a question the probe cannot
// answer: "the capability probe has not run on the bound device yet -- what should the build say
// in the meantime?". Everything else about the route is measured (see the selector below).
//
// What it still says is a BUILD fact and it is still true: a binary whose ONLY target is a
// pre-Volta rung cannot run the tensor-core attention arm *at all* (measured above: the tc
// attention arms need sm_80, and the prefill TU does not assemble below sm_80 without the
// pack_bf16x2 fix this line also made). `__CUDA_ARCH_LIST__` is nvcc's own record of the target
// list and is visible in the HOST pass (measured: recon/p2_archlist.txt -- 520 for
// `-arch=sm_52`, 1200 for sm_120a).
//
// WHY IT IS NO LONGER THE DECISION. `__CUDA_ARCH_LIST__` is an IDENTITY of the target LIST, and
// a fat multi-arch build -- the exact build the owner's "adaptation must cover pre-Ada .. + AMD
// + Intel + domestic + NPU" requirement produces -- does NOT define this macro, and its selector
// would fall through to the tensor-core arm on EVERY device, including an sm_52 card whose sm_52
// cubin in that same fatbin contains no `mma` at all. That is a wrong ROUTE decided by a build
// string, and it is the mode the owner's ruling "要调整支持检测模式，而不是放弃" asks to change:
// not "give up on the old card", but "stop deciding this from an identity". The identity
// survives here only as the named, logged answer for the not-yet-probed case.
//
// ---- CORRECTED 2026-09-19 (dl/simtpack): the CONCLUSION was right, the REASON was not ----
//
// The block that used to stand here said: "`__CUDA_ARCH_LIST__` is an IDENTITY of the target
// LIST, and nvcc records the list as a MAXIMUM of the rungs: measured, `-gencode
// arch=compute_120a` + `-gencode arch=compute_52` yields 1200, NOT 520 and NOT 52."
//
// MEASURED (dl/simtpack/probes/p10_archval.out -- an `nvcc -E` of a TU that both stringifies
// and evaluates the macro; nvcc 12.8, the toolkit this family is built with):
//
//   argv                                                __CUDA_ARCH_LIST__        old predicate
//   --generate-code=arch=compute_52,code=sm_52          520                       FIRES
//   --generate-code=arch=compute_120a,code=sm_120a      1200                      does not
//   ...120a AND ...52 (either order)                    520,1200                  does not
//   ...52 ...61 ...70 ...75 ...80 ...120a               520,610,700,750,800,1200  does not
//
// It is NOT a maximum. It is a COMMA-SEPARATED LIST of every rung, ascending and
// order-independent (measured in both orders). The old predicate
// `(__CUDA_ARCH_LIST__ > 0) && (__CUDA_ARCH_LIST__ < 700)` therefore read a LIST as a SCALAR, and
// the C comma operator is what made it false in the fat case: `520,1200 > 0` parses as
// `(520, (1200 > 0))` = 1, and `520,1200 < 700` as `(520, (1200 < 700))` = 0. So the conclusion
// held and the mechanism did not -- which matters for exactly one reason: the truth value came
// out of (a) the comma operator, which the C standard forbids in a `#if` constant-expression and
// this preprocessor merely tolerates, and (b) the LAST element of an nvcc-sorted list. An
// unsupported-but-tolerated construct whose answer depends on sort order is not a floor; it is
// an accident that currently points the right way. Two cases where it points the wrong way for
// the wrong reason, and one it cannot answer at all:
//
//   * `--generate-code=52 --generate-code=61` -> list `520,610` -> TRUE. Correct (a build of
//     only pre-Ada rungs), but for a reason the file does not state.
//   * `--generate-code=70 --generate-code=75` -> list `700,750` -> last element 750 -> FALSE.
//     Also correct, also by luck.
//   * MIXED, and this is the real case: a fat build holding BOTH an sm_52 and an sm_120a cubin.
//     On the sm_52 card the arm this family provides IS the only one that runs, and NO
//     build-time string can know that -- the same fatbin must behave differently on the two
//     cards. The build cannot answer the question, which is exactly why the probe above is the
//     decision and this macro is only the not-yet-probed answer.
//
// So the predicate now asks a question it CAN answer: "is EVERY rung in this build below the
// floor?" That is the only reading under which a build-only answer is sound, and it is expressed
// by counting the list and requiring all of it -- no comma operator, no dependence on sort
// order. Single-rung behaviour is preserved exactly (`520` -> fires, which the real sm_52 build
// at `build-sm52` demonstrates: its preprocessed host pass at
// `build-sm52/.../gqa_attention_decode_i8.cu.o.cudafe1.cpp:98432` selected source line 264
// `return true;`, i.e. this very block fired), and a fat build of several pre-Ada rungs now
// fires correctly too.
//
// NAMED LIMIT: at most NINFER_SIMT_FFMA_MAX_RUNGS (9) rungs. A longer list is a preprocessor
// error rather than a silent wrong answer -- the right failure direction -- and the tree's own
// arch gate (CMakeLists.txt:94-107) already bounds the list well below this.
#define NINFER_SIMT_FFMA_MAX_RUNGS 9
#define NINFER_SIMT_FFMA_RUNG_PRE_TC_(a) ((a) > 0 && (a) < 800)
#define NINFER_SIMT_FFMA_TC_RUNGS_1(a) NINFER_SIMT_FFMA_RUNG_PRE_TC_(a)
#define NINFER_SIMT_FFMA_TC_RUNGS_2(a, b)                                                          \
    (NINFER_SIMT_FFMA_TC_RUNGS_1(a) && NINFER_SIMT_FFMA_RUNG_PRE_TC_(b))
#define NINFER_SIMT_FFMA_TC_RUNGS_3(a, b, c)                                                       \
    (NINFER_SIMT_FFMA_TC_RUNGS_2(a, b) && NINFER_SIMT_FFMA_RUNG_PRE_TC_(c))
#define NINFER_SIMT_FFMA_TC_RUNGS_4(a, b, c, d)                                                    \
    (NINFER_SIMT_FFMA_TC_RUNGS_3(a, b, c) && NINFER_SIMT_FFMA_RUNG_PRE_TC_(d))
#define NINFER_SIMT_FFMA_TC_RUNGS_5(a, b, c, d, e)                                                 \
    (NINFER_SIMT_FFMA_TC_RUNGS_4(a, b, c, d) && NINFER_SIMT_FFMA_RUNG_PRE_TC_(e))
#define NINFER_SIMT_FFMA_TC_RUNGS_6(a, b, c, d, e, f)                                              \
    (NINFER_SIMT_FFMA_TC_RUNGS_5(a, b, c, d, e) && NINFER_SIMT_FFMA_RUNG_PRE_TC_(f))
#define NINFER_SIMT_FFMA_TC_RUNGS_7(a, b, c, d, e, f, g)                                           \
    (NINFER_SIMT_FFMA_TC_RUNGS_6(a, b, c, d, e, f) && NINFER_SIMT_FFMA_RUNG_PRE_TC_(g))
#define NINFER_SIMT_FFMA_TC_RUNGS_8(a, b, c, d, e, f, g, h)                                        \
    (NINFER_SIMT_FFMA_TC_RUNGS_7(a, b, c, d, e, f, g) && NINFER_SIMT_FFMA_RUNG_PRE_TC_(h))
#define NINFER_SIMT_FFMA_TC_RUNGS_9(a, b, c, d, e, f, g, h, i)                                     \
    (NINFER_SIMT_FFMA_TC_RUNGS_8(a, b, c, d, e, f, g, h) && NINFER_SIMT_FFMA_RUNG_PRE_TC_(i))
#define NINFER_SIMT_FFMA_RUNG_COUNT(...)                                                           \
    NINFER_SIMT_FFMA_RUNG_COUNT_(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define NINFER_SIMT_FFMA_RUNG_COUNT_(a1, a2, a3, a4, a5, a6, a7, a8, a9, n, ...) n
#define NINFER_SIMT_FFMA_TC_RUNGS_PICK_(n) NINFER_SIMT_FFMA_TC_RUNGS_##n
#define NINFER_SIMT_FFMA_TC_RUNGS_PICK(n) NINFER_SIMT_FFMA_TC_RUNGS_PICK_(n)
// TRUE iff every rung in the list is below the tensor-core attention floor. When
// `__CUDA_ARCH_LIST__` is undefined the identifier survives into the operand, where the
// preprocessor evaluates it as 0 -- so this is 0 and the block below correctly does not fire.
#define NINFER_SIMT_FFMA_EVERY_RUNG_PRE_TC(list)                                                   \
    NINFER_SIMT_FFMA_TC_RUNGS_PICK(NINFER_SIMT_FFMA_RUNG_COUNT(list))(list)
#if !defined(NINFER_ATTENTION_SIMT_FFMA_DEFAULT_ON) && defined(__CUDA_ARCH_LIST__) &&              \
    NINFER_SIMT_FFMA_EVERY_RUNG_PRE_TC(__CUDA_ARCH_LIST__)
#define NINFER_ATTENTION_SIMT_FFMA_DEFAULT_ON 1
#endif

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"
// The packed KV codecs this family can now read (dl/simtpack). Each is the tree's ONE definition
// of its codec, and the point of including them rather than re-deriving a decode is that a second
// spelling of a dequant is a second answer to "what does this byte mean".
#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/gqa_iso3_codec.cuh"
// dl/nvfp4emu2: ONE keyed, add-only announce set. The same header ops/kernel/nvfp4_ldm_free.cuh
// uses, for the same reason that file gives: a one-shot that can only ADD a key announces a
// SECOND, DIFFERENT selected value instead of going silent after the first.
#include "core/announce_once.h"

#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace ninfer::ops {

// ---------------------------------------------------------------------------
// Tiling. Both tiles stay inside a pre-Volta part's 48 KiB static shared-memory ceiling with no
// state array at all: decode is 2 x Bc x D bf16 (D = 256 -> 32 KiB) plus the page-id table;
// prefill is the same two tiles.
// ---------------------------------------------------------------------------
inline constexpr int kGqaSimtFfmaBc               = 32; // decode + prefill key tile
inline constexpr int kGqaSimtFfmaRowsPerWarp      = 4;  // register-resident rows per warp
inline constexpr int kGqaSimtFfmaWarps            = 4;
inline constexpr int kGqaSimtFfmaRowsPerCta       = kGqaSimtFfmaWarps * kGqaSimtFfmaRowsPerWarp;
inline constexpr int kGqaSimtFfmaPrefillThreads   = kGqaSimtFfmaWarps * 32;
inline constexpr int kGqaSimtFfmaThreadsPerWarp   = 32;
inline constexpr int kGqaSimtFfmaWarpReduction    = 32; // lanes that cooperate on one query row

// THE PROBE'S ANSWER, AS A TYPE. The probe (core/device_probe.cu, DeviceCapability::
// SimtFfmaAttention) RUNS this family's own row passes on the bound device, compares the numbers
// against a host reference, and publishes one of these. It is the only writer; see
// gqa_attention_simt_ffma_publish_probe_answer() below.
//
// BOTH ROW PASSES, BECAUSE THIS ANSWER GATES BOTH ARMS (dl/ffmaroute). The family has two: 
// ops::gqa_simt_ffma_row_pass (bf16, D = 128) and ops::gqa_simt_ffma_row_pass_packed (packed
// codecs, D = 256 -- the shape the NVFP4 route uses). This value is what the bf16 arm AND the
// codec arm both ask for, so an answer measured on the bf16 pass alone is a green that is not
// about the pass a codec launch would run -- the same defect as reading a `-c` success as a
// ptxas verdict, or a DENY list off a broken section splitter. The probe now runs both and
// publishes Failed if EITHER disagrees; the detail text names which one it was.
enum class SimtFfmaProbeAnswer : std::uint8_t {
    // The probe has not run on the bound device (no CUDA device bound, or the preflight never
    // reached this capability). The selector falls back to NINFER_ATTENTION_SIMT_FFMA_DEFAULT_ON
    // and SAYS SO through gqa_attention_simt_ffma_selection() -- an unprobed run is never
    // presented as a measured one.
    NotProbed = 0,
    // BOTH row passes (bf16 D=128 and the packed NVFP4 codec at D=256) ran on this device and
    // matched the host reference, AND this device has no working bf16 tensor-core arm, so this
    // family IS the attention arm that will run -- bf16, and NVFP4 through the codec body.
    RanOk = 1,
    // Both row passes ran on this device and matched, but this device also has a working bf16
    // tensor-core arm: this family is available and NOT the default. (The arm that runs is named
    // by the capability report, not by this value.)
    RanButTensorCoreArmAvailable = 2,
    // The probe ran and did not pass: a body did not launch, or it launched and disagreed with
    // the reference -- OR the bf16 body passed and the codec body did not (the detail says
    // which). Either answer is a measured "do not take this arm on this device": a codec launch
    // that inherits it takes the tensor-core arm instead, i.e. on a pre-Ada rung the named trap.
    Failed = 3,
};

[[nodiscard]] inline std::string_view simt_ffma_probe_answer_name(SimtFfmaProbeAnswer answer) {
    switch (answer) {
    case SimtFfmaProbeAnswer::NotProbed: return "not probed";
    case SimtFfmaProbeAnswer::RanOk:
        return "both row passes (bf16 D=128, NVFP4 codec D=256) ran on this device and matched "
               "the host reference";
    case SimtFfmaProbeAnswer::RanButTensorCoreArmAvailable:
        return "both row passes ran on this device and matched, but a bf16 tensor-core arm "
               "also works here";
    case SimtFfmaProbeAnswer::Failed:
        return "a row pass did not run or did not match the host reference (bf16 D=128 and "
               "the NVFP4 codec at D=256 are measured separately)";
    }
    return "?";
}

// WHERE THE ANSWER LIVES, AND WHY IT COSTS NO LINK EDGE. One object per process, shared by every
// TU, with no new symbol in any library: a function-local static inside an `inline` function is
// the ODR's single instance, so src/core/device_probe.cu (the writer) and each launcher TU (the
// readers) address the SAME object. That matters for the build: the launcher objects must not
// grow a link dependency on device_probe.o just to ask a route question.
[[nodiscard]] inline SimtFfmaProbeAnswer& gqa_attention_simt_ffma_probe_state() {
    static SimtFfmaProbeAnswer answer = SimtFfmaProbeAnswer::NotProbed;
    return answer;
}

// The only writer. Called by the probe in core/device_probe.cu, once per device, before any
// launcher asks.
inline void gqa_attention_simt_ffma_publish_probe_answer(SimtFfmaProbeAnswer answer) {
    gqa_attention_simt_ffma_probe_state() = answer;
}

// The env override, factored OUT of the decision so the two can be reported separately: an
// operator who set NINFER_ATTENTION_SIMT_FFMA=1 has overridden the probe, and a report that says
// "the probe answered X" while the route is X's opposite is the same lie in a smaller font.
// Returns -1 when unset, else 0/1.
[[nodiscard]] inline int gqa_attention_simt_ffma_env_override() {
    const char* env = std::getenv("NINFER_ATTENTION_SIMT_FFMA");
    if (env != nullptr && env[0] != '\0' && env[1] == '\0') {
        if (env[0] == '1') { return 1; }
        if (env[0] == '0') { return 0; }
    }
    return -1;
}

// WHY the selector answered what it answered. Anything that PRINTS the route must read this
// instead of guessing from the bool: "selected" on its own cannot distinguish a measured answer
// from the unprobed build default, and that difference is the whole point of this change.
enum class SimtFfmaSelection : std::uint8_t {
    ForcedOn,             // NINFER_ATTENTION_SIMT_FFMA=1
    ForcedOff,            // NINFER_ATTENTION_SIMT_FFMA=0
    ProbeRanOk,           // measured: the row pass ran on this device and no tensor-core arm is here
    ProbeNotTheArm,       // measured: it ran, but this device has a working tensor-core bf16 arm
    ProbeFailed,          // measured: do not take this arm on this device
    UnprobedBuildDefault, // NOT measured: the probe has not run (NINFER_ATTENTION_SIMT_FFMA_DEFAULT_ON)
};

[[nodiscard]] inline const char* gqa_attention_simt_ffma_selection_text(SimtFfmaSelection how) {
    switch (how) {
    case SimtFfmaSelection::ForcedOn: return "NINFER_ATTENTION_SIMT_FFMA=1 (operator override)";
    case SimtFfmaSelection::ForcedOff: return "NINFER_ATTENTION_SIMT_FFMA=0 (operator override)";
    case SimtFfmaSelection::ProbeRanOk:
        return "MEASURED by the capability probe: this device has no working bf16 tensor-core "
               "arm and BOTH SIMT FFMA row passes (bf16 D=128 and NVFP4 codec D=256) ran here "
               "and matched the host reference";
    case SimtFfmaSelection::ProbeNotTheArm:
        return "MEASURED by the capability probe: this device has a working bf16 tensor-core arm, "
               "so the SIMT FFMA family is not the default here";
    case SimtFfmaSelection::ProbeFailed:
        return "MEASURED by the capability probe: a SIMT FFMA row pass did not run or did not "
               "match on this device (the bf16 D=128 and NVFP4 codec D=256 passes are separate "
               "measurements)";
    case SimtFfmaSelection::UnprobedBuildDefault:
        return "NOT MEASURED: the capability probe has not run on this device, so this is the "
               "BUILD default (NINFER_ATTENTION_SIMT_FFMA_DEFAULT_ON, from __CUDA_ARCH_LIST__)";
    }
    return "?";
}

// The same decision, with its REASON. Every caller that can print (the launcher's splitdbg tier
// line, the capability report, a test) should use this one: "which arm" and "on whose word" are
// different questions and only the second one can tell a measured answer from a fallback.
//
// It is DEFINED BEFORE gqa_attention_simt_ffma_selected(), which calls it: both are inline free
// functions in one header, so a call to a function that is not yet declared is a hard error at
// parse time (measured: the real `nvcc -c` rejects this file with "identifier
// \"gqa_attention_simt_ffma_selection\" is undefined" at the switch in the selector below, while
// `-E` on the same translation unit succeeds and shows the new code -- -E is not a compile
// instrument).
[[nodiscard]] inline SimtFfmaSelection gqa_attention_simt_ffma_selection() {
    const int override_value = gqa_attention_simt_ffma_env_override();
    if (override_value == 1) { return SimtFfmaSelection::ForcedOn; }
    if (override_value == 0) { return SimtFfmaSelection::ForcedOff; }
    switch (gqa_attention_simt_ffma_probe_state()) {
    case SimtFfmaProbeAnswer::RanOk: return SimtFfmaSelection::ProbeRanOk;
    case SimtFfmaProbeAnswer::RanButTensorCoreArmAvailable:
        return SimtFfmaSelection::ProbeNotTheArm;
    case SimtFfmaProbeAnswer::Failed: return SimtFfmaSelection::ProbeFailed;
    case SimtFfmaProbeAnswer::NotProbed: return SimtFfmaSelection::UnprobedBuildDefault;
    }
    return SimtFfmaSelection::UnprobedBuildDefault;
}

// The host-side switch. It is a NAMED route choice and not a gate: it can only ever move a
// launch onto this family, never past a refusal (the capability preflight, the dtype guards and
// the geometry guards all run before it).
//
// ORDER: operator override, then the PROBE, then the build default. The middle step is the change
// this replacement exists for -- the decision is a measured fact about THIS device instead of an
// identity read from the target list.
//
// NINFER_ATTENTION_SIMT_FFMA=1 forces the family on; =0 forces it off. Unset means "the probe".
// The answer is NOT cached in a function-local static any more: the probe may publish after the
// first ask (there is no ordering contract to hide behind), and a stale "not probed" cached once
// would silently turn this back into the build default for the rest of the process. The read is a
// load of one byte, and the probe itself is cached per device in core/device_probe.cu, so there is
// no per-token and no per-launch cost either way.
[[nodiscard]] inline bool gqa_attention_simt_ffma_selected() {
    switch (gqa_attention_simt_ffma_selection()) {
    case SimtFfmaSelection::ForcedOn:
    case SimtFfmaSelection::ProbeRanOk: return true;
    case SimtFfmaSelection::ForcedOff:
    case SimtFfmaSelection::ProbeNotTheArm:
    case SimtFfmaSelection::ProbeFailed: return false;
    case SimtFfmaSelection::UnprobedBuildDefault:
#if defined(NINFER_ATTENTION_SIMT_FFMA_DEFAULT_ON)
        return true;
#else
        return false;
#endif
    }
    return false;
}

// The named refusal for a launch this minimum path cannot serve. It is a refusal and not a
// fallback: on the build this family exists for, the tensor-core arm it would fall back to
// cannot run at all (no mma below sm_70), so "quietly use the other kernel" is exactly the
// silent-wrong-answer shape this tree refuses.
[[nodiscard]] inline const char* gqa_attention_simt_ffma_refusal(bool cold_pool_armed) {
    if (cold_pool_armed) {
        return "the SIMT FFMA attention route has no cold-slot codec: a layer with an armed "
               "cold-slot pool must not be routed here (this family reads the cache plane only)";
    }
    return nullptr;
}

// f32 -> bf16 that does NOT need sm_80's `cvt.rn.bf16x2.f32` (measured: sm_80 and up only), and
// does not need `cvt.rn.bf16.f32` (sm_75 and up) either. Round-to-nearest-even on the top 16
// bits, pure integer ops, so it assembles on every rung in the ladder -- the same spelling
// CUTLASS's software `NumericConverter<bf16_t, float, round_to_nearest>` uses. Only finite
// values are ever fed to it by the kernels below (a softmax weight, a normalized output).
__device__ __forceinline__ std::uint16_t gqa_simt_f32_to_bf16_bits(float value) {
    unsigned bits      = __float_as_uint(value);
    const unsigned lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(bits >> 16);
}

__device__ __forceinline__ __nv_bfloat16 gqa_simt_f32_to_bf16(float value) {
    return __ushort_as_bfloat16(gqa_simt_f32_to_bf16_bits(value));
}

// ===========================================================================
// THE KV CODEC SEAM (dl/simtpack)
// ===========================================================================
//
// WHY. Before this block the family read the raw bf16 plane and nothing else: the staging was an
// unconditional 128-bit bf16 copy with no dtype branch, and a whole-file census for
// `e2m1`/`e4m3`/`dequant` found 0 code occurrences. So `DType::I8`, `FP8_E4M3FN`, `ISO3` and
// `NVFP4` KV layers had no pre-Volta decode at all -- their launches fell through to the
// tensor-core arms, whose `mma`/`ldmatrix`/`cp.async` helpers ARE the
// `unsupported_instruction_trap()` arm below sm_80. That is a trap at dispatch, which is worse
// than a refusal: an operator cannot act on it.
//
// THE SECOND PAYOFF, AND WHY BOTH LIVE HERE. The decode CTA's shared memory is 2*Bc*D bf16 plus
// the page-id table -- 33,792 B at D=256/Bc=32, of which the two tiles are 32,768 -- and that,
// not the register count, is what caps the kernel at 2 blocks/SM (12.5%). Double-buffering makes
// it strictly worse (~1 block). The lever is the tile's SIZE, and a *packed* plane read into the
// tile is exactly that: 288 B/key/KV-head for NVFP4 against 1,024 for bf16. The codec gap and
// the occupancy cap are therefore the SAME edit, and this is that edit.
//
// WHY PURE ALU IS POSSIBLE AT ALL. Every codec below decodes with integer select, shifts, an
// int->float convert and an FMA -- no `cvt.rn.bf16x2.f32` (sm_80), no `__dp4a` (sm_61), no
// `mma`/`ldmatrix`/`cp.async` (sm_80), no library intrinsic. The one exception in the tree's own
// readers was `ldexpf` in the E4M3 decode (`gqa_attention_kv_nvfp4.cuh:122,191`), a libdevice
// call; `gqa_simt_e4m3_normal_to_f32` below is the bit-exact integer spelling of that same
// function (derivation in place), so the seam is ALU-only by construction and not by inspection.
//
// THE FLOOR OF EACH DECODE, from `dl/simtattn-audit/PRIMITIVES.md` section 2.2 and section 4
// (its per-helper table, established by a real per-rung compile -- CITED, not re-derived):
//   bf16   (identity)           sm_20
//   ISO3   `(code&7)`, sign bit  sm_20  -- the purest: no scale decode, no convert
//   NVFP4  E2M1 select           sm_20 for the select; the tree's reader also uses `pack_bf16x2`,
//                                whose software arm (math.cuh:56) is the sm_50 path
//   FP8    E4M3FN data           sm_20 with the integer scale decode below
//   I8     int8 -> f32 * scale   sm_20 (I2F); its g64 scale is fp16, read with `__half2float`,
//                                measured OK at sm_52 by PRIMITIVES.md section 4
//   E8Kv   REFUSED BY NAME -- see `gqa_simt_kv_codec_supported`.
//
// NOT IN SCOPE, NAMED SO IT CANNOT READ AS AN OVERSIGHT: the NVFP4 V **residual** plane, the
// cold-slot pools, and the E8Kv lattice. Each is refused by name in
// `gqa_attention_simt_ffma_refusal` rather than silently mis-decoded.
enum class GqaSimtKvCodec : std::uint8_t {
    Bf16  = 0, // the landed path: the plane IS the tile
    Nvfp4 = 1, // E2M1 nibbles, one E4M3 scale per 16 dims
    Fp8   = 2, // E4M3FN data bytes, one E4M3 scale per 16 dims
    Iso3  = 3, // sign-magnitude nibbles, one E4M3 scale per 16 dims
    I8    = 4, // int8 code bytes, one fp16 scale per 64 dims
};

[[nodiscard]] __host__ __device__ constexpr const char* gqa_simt_kv_codec_name(
    GqaSimtKvCodec codec) {
    switch (codec) {
    case GqaSimtKvCodec::Bf16: return "bf16";
    case GqaSimtKvCodec::Nvfp4: return "nvfp4";
    case GqaSimtKvCodec::Fp8: return "fp8";
    case GqaSimtKvCodec::Iso3: return "iso3";
    case GqaSimtKvCodec::I8: return "i8";
    }
    return "?";
}

// The named answer to "can this seam decode codec C in ALU below sm_80?". It is a constant
// function and not a comment so that a refusal can be *reported* by the code that would otherwise
// have to guess.
[[nodiscard]] __host__ __device__ constexpr bool gqa_simt_kv_codec_supported(GqaSimtKvCodec codec) {
    // E8Kv (`rk4v4`, W2/W3) is REFUSED BY NAME, and NOT because its arithmetic is hard: its
    // decoder is pure ALU too (`e8_lattice_decode_8d_t<float>` = POPC + shifts + a table load +
    // I2F + FMUL + FADD/FMA, with the lattice tables passed in as parameters). It is refused
    // because its PLANE IS NOT THIS SEAM'S SHAPE. Its scale group is 64 dims, its codeword is a
    // 2- or 3-byte plate per 8 dims, and -- the load-bearing part -- its reader's output is
    // ROTATED (`e8_lattice_kv_plane.cuh:217 kE8KvLatticeOutputIsRotated = true`) and is only
    // exact when paired with `e8_kv_lattice_reader_scale()`. The K it hands back is not the
    // stored K but a rotation of it that the QK path must undo with its own scale, so a per-8-dim
    // chunk dequant would answer a different question than the caller asked -- which is exactly
    // the silent wrong this tree refuses. E8Kv needs its own arm, not a row in this one.
    return codec == GqaSimtKvCodec::Bf16 || codec == GqaSimtKvCodec::Nvfp4 ||
           codec == GqaSimtKvCodec::Fp8 || codec == GqaSimtKvCodec::Iso3 ||
           codec == GqaSimtKvCodec::I8;
}

// One lane-local 8-dim chunk is the unit this whole seam works in, and every codec above fits it
// EXACTLY: for D=256, VecD = D/32 = 8, so lane l owns dims [8l, 8l+8); every codec's scale group
// is a multiple of 8, so those 8 dims lie wholly inside ONE scale group and each lane needs
// precisely one scale. That is what keeps the dequant register-local -- no shuffle, no cross-lane
// traffic -- and it is a property, not a coincidence:
static_assert(kGqaSimtFfmaWarpReduction == 32, "one lane owns HeadDim/32 contiguous dims");

template <GqaSimtKvCodec C> struct GqaSimtKvCodecTraits;

// ---- E4M3FN decode as INTEGER BITS: the ALU spelling of `ldexpf(1 + m/8, e - 7)` ----
//
// Derivation, so the equivalence is auditable rather than asserted. `(8+m)/8` for m in [0,7] is
// exact in fp32, with exponent field 127 and mantissa field `m << 20`. Scaling by 2^(e-7) adds
// (e-7) to that exponent field, giving `120 + e`. For e in [1,14] that field is 121..134 -- all
// normal -- so the result is EXACT: there is no rounding for the two emitters to disagree about.
// e == 0 is the subnormal case the tree spells `m / 512.0f`; 2^-9 is exactly representable, so
// `(float)m * 0x1p-9f` is exact and identical to the division.
__device__ __forceinline__ float gqa_simt_e4m3_normal_to_f32(std::uint32_t e, std::uint32_t m) {
    return __uint_as_float(((120u + e) << 23) | (m << 20));
}

// The SCALE plane's reader, i.e. `gqa_kv_nvfp4_e4m3_to_f32`. Its two frozen properties are
// preserved exactly: x <= 0 encodes to 0x00, and 0x7F reads back as 480.0f
// (e=15, m=7 -> exponent field 135, mantissa 7 -> 1.875 * 2^8 = 480).
__device__ __forceinline__ float gqa_simt_e4m3_scale_to_f32(std::uint8_t byte) {
    const std::uint32_t e = (byte >> 3) & 0x0Fu;
    const std::uint32_t m = byte & 0x07u;
    if (e == 0u) { return static_cast<float>(m) * 0x1p-9f; }
    return gqa_simt_e4m3_normal_to_f32(e, m);
}

// The DATA plane's reader, i.e. `gqa_kv_nvfp4_data_e4m3_to_f32`. Bit 7 is the SIGN here, and
// 0x7F/0xFF are the format's NaN codes, which the writer never emits and this reader maps to the
// format maximum (448.0f) rather than letting a stored byte inject a NaN into attention.
__device__ __forceinline__ float gqa_simt_e4m3_data_to_f32(std::uint8_t byte) {
    const std::uint32_t mag = byte & 0x7Fu;
    const std::uint32_t e   = (mag >> 3) & 0x0Fu;
    const std::uint32_t m   = mag & 0x07u;
    float value;
    if (e == 0u) {
        value = static_cast<float>(m) * 0x1p-9f;
    } else if (e == 15u && m == 7u) {
        value = 448.0f;
    } else {
        value = gqa_simt_e4m3_normal_to_f32(e, m);
    }
    return (byte & 0x80u) != 0u ? -value : value;
}

template <> struct GqaSimtKvCodecTraits<GqaSimtKvCodec::Bf16> {
    static constexpr bool kPacked          = false;
    static constexpr int  kCodeBitsPerEl    = 16;
    static constexpr int  kGroupDims        = 8; // no scale plane: the group IS the chunk
    static constexpr int  kScaleBytesPerGrp = 0;
};
template <> struct GqaSimtKvCodecTraits<GqaSimtKvCodec::Nvfp4> {
    static constexpr bool kPacked          = true;
    static constexpr int  kCodeBitsPerEl    = 4;
    static constexpr int  kGroupDims        = 16;
    static constexpr int  kScaleBytesPerGrp = 1; // E4M3
};
template <> struct GqaSimtKvCodecTraits<GqaSimtKvCodec::Fp8> {
    static constexpr bool kPacked          = true;
    static constexpr int  kCodeBitsPerEl    = 8;
    static constexpr int  kGroupDims        = 16;
    static constexpr int  kScaleBytesPerGrp = 1; // E4M3
};
template <> struct GqaSimtKvCodecTraits<GqaSimtKvCodec::Iso3> {
    static constexpr bool kPacked          = true;
    static constexpr int  kCodeBitsPerEl    = 4;
    static constexpr int  kGroupDims        = 16;
    static constexpr int  kScaleBytesPerGrp = 1; // E4M3
};
template <> struct GqaSimtKvCodecTraits<GqaSimtKvCodec::I8> {
    static constexpr bool kPacked          = true;
    static constexpr int  kCodeBitsPerEl    = 8;
    static constexpr int  kGroupDims        = 64;
    static constexpr int  kScaleBytesPerGrp = 2; // fp16 (`kv_scale_half`)
};

// The min-blocks-per-SM the compiler is asked to target, PER CODEC. See fix-3's rationale in
// this line's record: the second `__launch_bounds__` parameter IS the register budget, and the
// landed 2 is what makes the bf16 arm's 140 registers a choice rather than a ceiling. A packed
// codec's shared memory no longer binds, so the target has to move with it or the tile win is not
// realized. The arithmetic, with the landed 128 threads/CTA:
//   minBlocks = 2 -> 65536/(128*2) = 256 registers allowed -> 2 blocks if it uses >= 171
//   minBlocks = 3 -> 65536/(128*3) = 170 registers allowed -> 3 blocks (= 18.75%) if it uses <= 170
[[nodiscard]] __host__ __device__ constexpr int gqa_simt_ffma_min_blocks_per_sm(
    GqaSimtKvCodec codec) {
    return codec == GqaSimtKvCodec::Bf16 ? 2 : 3;
}

// The bytes one key's plane row costs, per KV head. This is the number the occupancy arithmetic
// turns on, so it is a function and not a comment.
[[nodiscard]] __host__ __device__ constexpr int gqa_simt_kv_code_row_bytes(GqaSimtKvCodec codec, int head_dim) {
    switch (codec) {
    case GqaSimtKvCodec::Bf16: return head_dim * 2;
    case GqaSimtKvCodec::Nvfp4: return head_dim / 2;
    case GqaSimtKvCodec::Iso3: return head_dim / 2;
    case GqaSimtKvCodec::Fp8: return head_dim;
    case GqaSimtKvCodec::I8: return head_dim;
    }
    return 0;
}
[[nodiscard]] __host__ __device__ constexpr int gqa_simt_kv_scale_row_bytes(GqaSimtKvCodec codec, int head_dim) {
    switch (codec) {
    case GqaSimtKvCodec::Bf16: return 0;
    case GqaSimtKvCodec::Nvfp4:
    case GqaSimtKvCodec::Fp8:
    case GqaSimtKvCodec::Iso3: return (head_dim / 16) * 1;
    case GqaSimtKvCodec::I8: return (head_dim / 64) * 2;
    }
    return 0;
}
// Bytes per (key, KV head) for BOTH planes of K AND V: the figure the tile cap is computed from.
[[nodiscard]] __host__ __device__ constexpr int gqa_simt_kv_bytes_per_key_kv_head(GqaSimtKvCodec codec, int head_dim) {
    return 2 * (gqa_simt_kv_code_row_bytes(codec, head_dim) +
                gqa_simt_kv_scale_row_bytes(codec, head_dim));
}

// Dequantize the ONE group scale a lane needs for its own 8 dims. Lane-local by the static_assert
// above: dims [8l, 8l+8) lie inside a single group, whose index is 8l / group_dims.
template <GqaSimtKvCodec C>
__device__ __forceinline__ float gqa_simt_kv_lane_scale(const std::uint8_t* __restrict__ scale_row,
                                                        int d_lane) {
    using T = GqaSimtKvCodecTraits<C>;
    static_assert(T::kPacked, "bf16 has no scale plane");
    const int group = d_lane / T::kGroupDims;
    if constexpr (C == GqaSimtKvCodec::I8) {
        return __half2float(*reinterpret_cast<const __half*>(scale_row + group * 2));
    } else {
        return gqa_simt_e4m3_scale_to_f32(scale_row[group]);
    }
}

// Dequantize ONE lane-local chunk of `N` dims of one key into fp32[N]. `code` points at the
// chunk's first code byte. Every arm reaches the tree's OWN codec definition through its helper.
//
// `N` IS VecD = HeadDim/32, so it is 8 at D=256 and 4 at D=128 -- MEASURED, not assumed: the first
// version of this function hard-coded 8 and the second compile attempt failed its static_assert on
// GqaMuseGeometry (probes/c2_proof.out:23). The property the seam actually needs is not a width but
// lane-locality: a lane's `N` dims must lie inside ONE scale group, i.e. `kGroupDims % N == 0`,
// which holds for N in {4,8} against every codec here (16 for nvfp4/fp8/iso3, 64 for i8).
// `kCodeVecBytes` is the byte width of one chunk, and it is what the staging zero-fills and the
// row pass loads, so the two cannot disagree about it.
template <GqaSimtKvCodec C, int N>
__device__ __forceinline__ void gqa_simt_kv_dequant(const std::uint8_t* __restrict__ code,
                                                    float scale, float (&out)[N]) {
    using T = GqaSimtKvCodecTraits<C>;
    static_assert(T::kPacked, "bf16 goes through the raw tile, not this path");
    static_assert(N == 4 || N == 8, "the seam's chunk is VecD, which is 4 or 8");
    if constexpr (C == GqaSimtKvCodec::Nvfp4) {
#pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            out[2 * i]     = gqa_kv_nvfp4_e2m1_to_f32(code[i] & 0x0Fu) * scale;
            out[2 * i + 1] = gqa_kv_nvfp4_e2m1_to_f32(code[i] >> 4) * scale;
        }
    } else if constexpr (C == GqaSimtKvCodec::Iso3) {
#pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            out[2 * i]     = gqa_iso3_decode(code[i] & 0x0Fu) * scale;
            out[2 * i + 1] = gqa_iso3_decode(code[i] >> 4) * scale;
        }
    } else if constexpr (C == GqaSimtKvCodec::Fp8) {
#pragma unroll
        for (int i = 0; i < N; ++i) { out[i] = gqa_simt_e4m3_data_to_f32(code[i]) * scale; }
    } else if constexpr (C == GqaSimtKvCodec::I8) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            out[i] = static_cast<float>(reinterpret_cast<const std::int8_t*>(code)[i]) * scale;
        }
    }
}

// The packed tile's byte layout for one K or V plane, as the ONE value the arena sizing, the
// staging and the row pass all read. Making those three agree by hand is the class of bug this
// struct removes.
struct GqaSimtPackedPlane {
    std::uint8_t* code      = nullptr;
    std::uint8_t* scale     = nullptr;
    int           code_row  = 0; // bytes per key, code plane
    int           scale_row = 0; // bytes per key, scale plane
};

// `gqa_simt_round_visible` is defined a little further down this header, next to the bf16 row
// pass that is its first consumer. The packed row pass below is a second consumer and it is
// declared BEFORE it, so the declaration is forwarded here rather than by moving the seam: the
// seam is one block and moving it would make the diff of this change span the whole file.
__device__ __forceinline__ bool gqa_simt_round_visible(std::uint64_t mask, int rel);

// ===========================================================================
// THE PACKED ROW PASS. Same online-softmax ordering as `gqa_simt_ffma_row_pass` -- rescale BEFORE
// accumulate, the `m == -inf` arm, `l = fma(l, alpha, bl)` -- because a second ordering here
// would be a second numerics contract for one family. Only the two loads change: where the bf16
// pass does `__bfloat162float(k_row[i])`, this one dequantizes the lane's 8 dims from the packed
// planes. KEEP `Bc` AND THE TILE BOUNDARY IDENTICAL to the bf16 arm: the fp32 association of the
// online softmax is preserved if and only if the tile boundary does not move, so a codec must
// never be allowed to change Bc.
// ===========================================================================
template <typename Geometry, int Bc, GqaSimtKvCodec Codec>
__device__ __forceinline__ void gqa_simt_ffma_row_pass_packed(
    const float (&qr)[Geometry::HeadDim / kGqaSimtFfmaWarpReduction],
    const GqaSimtPackedPlane& k_plane, const GqaSimtPackedPlane& v_plane, int lane, int tile_key0,
    int qabs, std::uint64_t mask, bool round_masked, int column_begin, int first_pos,
    int split_start, int split_end, float scale, float& m, float& l,
    float (&acc)[Geometry::HeadDim / kGqaSimtFfmaWarpReduction]) {
    using T                      = GqaSimtKvCodecTraits<Codec>;
    constexpr int D              = Geometry::HeadDim;
    constexpr int VecD           = D / kGqaSimtFfmaWarpReduction;
    constexpr float Log2E        = 1.4426950408889634074f;
    constexpr unsigned FullMask  = 0xffffffffu;
    static_assert(T::kPacked, "the packed row pass is for packed codecs only");
    // The property the dequant needs is lane-LOCALITY, not a width: a lane's VecD dims must lie
    // inside ONE scale group. VecD is 8 at D=256 and 4 at D=128 (measured, see the dequant above).
    static_assert(VecD == 4 || VecD == 8, "the seam's chunk is VecD, which is 4 or 8");
    static_assert(T::kGroupDims % VecD == 0, "a lane's dims must lie inside ONE scale group");

    const int d_lane  = lane * VecD;
    const int code_at = d_lane * T::kCodeBitsPerEl / 8;

    float part[Bc];
#pragma unroll
    for (int kk = 0; kk < Bc; ++kk) {
        const std::uint8_t* kc = k_plane.code + kk * k_plane.code_row + code_at;
        const std::uint8_t* ks = k_plane.scale + kk * k_plane.scale_row;
        float kd[VecD];
        gqa_simt_kv_dequant<Codec, VecD>(kc, gqa_simt_kv_lane_scale<Codec>(ks, d_lane), kd);
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < VecD; ++i) { s = __fmaf_rn(qr[i], kd[i], s); }
        part[kk] = s;
    }
#pragma unroll
    for (int kk = 0; kk < Bc; ++kk) {
        part[kk] = warp_sum<kGqaSimtFfmaWarpReduction>(part[kk], FullMask);
    }

    float tile_max = -CUDART_INF_F;
#pragma unroll
    for (int kk = 0; kk < Bc; ++kk) {
        const int key      = tile_key0 + kk;
        const int rel      = column_begin + key - first_pos;
        const bool visible = (key >= split_start && key < split_end && key <= qabs) &&
                             (!round_masked || gqa_simt_round_visible(mask, rel));
        part[kk]           = visible ? part[kk] * scale : -CUDART_INF_F;
        tile_max           = fmaxf(tile_max, part[kk]);
    }

    const float nm    = fmaxf(m, tile_max);
    const float alpha = (m == -CUDART_INF_F) ? 0.0f : exp2_approx((m - nm) * Log2E);
    if (alpha == 0.0f) {
#pragma unroll
        for (int i = 0; i < VecD; ++i) { acc[i] = 0.0f; }
    } else {
#pragma unroll
        for (int i = 0; i < VecD; ++i) { acc[i] *= alpha; }
    }
    float bl = 0.0f;
#pragma unroll
    for (int kk = 0; kk < Bc; ++kk) {
        const float p = (nm > -CUDART_INF_F && part[kk] > -CUDART_INF_F)
                            ? exp2_approx((part[kk] - nm) * Log2E)
                            : 0.0f;
        bl += p;
        const std::uint8_t* vc = v_plane.code + kk * v_plane.code_row + code_at;
        const std::uint8_t* vs = v_plane.scale + kk * v_plane.scale_row;
        float vd[VecD];
        gqa_simt_kv_dequant<Codec, VecD>(vc, gqa_simt_kv_lane_scale<Codec>(vs, d_lane), vd);
#pragma unroll
        for (int i = 0; i < VecD; ++i) { acc[i] = __fmaf_rn(p, vd[i], acc[i]); }
    }
    l = __fmaf_rn(l, alpha, bl);
    m = nm;
}

// Zero a code chunk or a scale group. The widths are the actual widths -- a wider store here
// would clobber a neighbouring chunk's valid data, so the size is enforced by type and not by
// care. In every codec below a zero code or a zero scale is a zero VALUE (the E4M3 decodes read
// byte 0x00 as exponent 0 -> m/512 = 0), so zeroing BOTH planes is safe independently: the pair
// cannot produce a non-zero where the guard wants a zero, and it cannot produce a NaN.
template <int Bytes>
__device__ __forceinline__ void gqa_simt_kv_zero(std::uint8_t* dst) {
    static_assert(Bytes == 1 || Bytes == 2 || Bytes == 4 || Bytes == 8,
                  "1/2/4/8-byte chunk widths only");
    if constexpr (Bytes == 1) {
        dst[0] = 0u;
    } else if constexpr (Bytes == 2) {
        *reinterpret_cast<std::uint16_t*>(dst) = 0u;
    } else if constexpr (Bytes == 4) {
        *reinterpret_cast<std::uint32_t*>(dst) = 0u;
    } else {
        *reinterpret_cast<std::uint64_t*>(dst) = 0ull;
    }
}

// The packed staging. TWO loops and not one, because the two planes have different granularity:
// the code plane is staged per 8 dims and the scale plane per group. Staging the scale from the
// per-chunk loop would make every chunk of a group write the same byte -- benign in value, but a
// second writer is a hazard this family does not need to carry when a second loop is free.
template <typename Geometry, int Bc, GqaSimtKvCodec Codec, typename CodeSource,
          typename ScaleSource>
__device__ __forceinline__ void gqa_simt_ffma_stage_tile_packed(
    const GqaSimtPackedPlane& k_plane, const GqaSimtPackedPlane& v_plane, int tid, int threads,
    int k0, int split_start, int split_end, const CodeSource& code_source,
    const ScaleSource& scale_source) {
    using T            = GqaSimtKvCodecTraits<Codec>;
    constexpr int D    = Geometry::HeadDim;
    constexpr int VecD = D / kGqaSimtFfmaWarpReduction; // 8 at D=256, 4 at D=128 -- VecD, not 8
    static_assert(T::kPacked, "the packed staging is for packed codecs only");
    static_assert(VecD == 4 || VecD == 8, "the seam's chunk is VecD, which is 4 or 8");
    // Bytes covering ONE chunk of VecD dims: 4 at D=256/nibble, 2 at D=128/nibble, 8 at
    // D=256/byte, 4 at D=128/byte. It is what `gqa_simt_kv_zero` zero-fills and what the code
    // load writes, so the width is derived once and not spelled at either site.
    constexpr int kCodeVecBytes = VecD * T::kCodeBitsPerEl / 8;

    for (int chunk = tid; chunk < Bc * (D / VecD); chunk += threads) {
        const int key_l = chunk / (D / VecD);
        const int d     = (chunk - key_l * (D / VecD)) * VecD;
        const int key   = k0 + key_l;
        std::uint8_t* kc = k_plane.code + key_l * k_plane.code_row + d * kCodeVecBytes / VecD;
        std::uint8_t* vc = v_plane.code + key_l * v_plane.code_row + d * kCodeVecBytes / VecD;
        if (key >= split_start && key < split_end) {
            code_source(key, d, kc, vc);
        } else {
            gqa_simt_kv_zero<kCodeVecBytes>(kc);
            gqa_simt_kv_zero<kCodeVecBytes>(vc);
        }
    }
    constexpr int kGroups = D / T::kGroupDims;
    for (int g = tid; g < Bc * kGroups; g += threads) {
        const int key_l  = g / kGroups;
        const int grp    = g - key_l * kGroups;
        const int key    = k0 + key_l;
        std::uint8_t* ks = k_plane.scale + key_l * k_plane.scale_row + grp * T::kScaleBytesPerGrp;
        std::uint8_t* vs = v_plane.scale + key_l * v_plane.scale_row + grp * T::kScaleBytesPerGrp;
        if (key >= split_start && key < split_end) {
            scale_source(key, grp, ks, vs);
        } else {
            gqa_simt_kv_zero<T::kScaleBytesPerGrp>(ks);
            gqa_simt_kv_zero<T::kScaleBytesPerGrp>(vs);
        }
    }
}

// ===========================================================================
// WHICH CODECS THE SEAM CAN *DECODE* AND WHICH OF THOSE HAVE A WRITTEN ADDRESS PATH.
// Two different questions, so two predicates -- the same separation the selector's
// `..._selection()`/`..._selected()` pair keeps, and for the same reason: "the arithmetic works"
// and "this launch can reach the plane" are answered by different evidence.
//
// DECODE (this predicate): the codec's dequant is pure ALU below sm_80, cited per-codec above.
// ROUTABLE (the next predicate): the codec's PLANE INDEX MATH has been written here and checked
// against that tier's own `gqa_kv_*_index` helpers. Only NVFP4 has that today, because only
// NVFP4's two index helpers were read and matched against this seam's `code_row`/`scale_row`
// arithmetic (dl/simtpack/probes/p8_scales.out section A: the nvfp4 tier reads its code plane at
// `gqa_kv_nvfp4_code_index` and its scale plane at `gqa_kv_nvfp4_scale_index`, whose leading
// extents are exactly `HeadDim/2` and `HeadDim/16`).
//
// WHY THE OTHER THREE ARE REFUSED RATHER THAN WIRED. Their dequant is done and compiles -- the
// hard part is not the arithmetic. What is missing is the *address*: ISO3 shares the NVFP4 plane
// pair but is reached through its own tier's reader, and FP8's and I8's scale planes live in
// different tensors with different leading extents (`gqa_kv_quant_scale_index`, leading extent
// HeadDim/64, fp16 cells). Guessing that math from this header would put a second, untested copy
// of three tiers' paging arithmetic next to the tree's own. A route that guesses an address is
// the silent wrong this tree refuses, so the route is REFUSED BY NAME instead and the reason is
// recorded here.
// ===========================================================================
[[nodiscard]] __host__ __device__ constexpr bool gqa_simt_kv_codec_routable(GqaSimtKvCodec codec) {
    return codec == GqaSimtKvCodec::Bf16 || codec == GqaSimtKvCodec::Nvfp4;
}

// The scale plane is a SECOND tensor in every packed tier (the code plane and the scale plane are
// separate `Tensor`s in `PagedKVBatchLayerView`). It reaches the kernel as a single additional
// pair of pointers, null for bf16 -- whose arena has no scale region at all.
struct GqaSimtKvScalePlanes {
    const std::uint8_t* k = nullptr;
    const std::uint8_t* v = nullptr;
};

// ---- The address path a packed codec's staging uses. Only NVFP4's is written (see
// `gqa_simt_kv_codec_routable`); the others are a compile error rather than a guess. ----
template <typename Geometry, GqaSimtKvCodec Codec>
__device__ __forceinline__ void gqa_simt_ffma_load_code(const std::uint8_t* __restrict__ cache_k,
                                                        const std::uint8_t* __restrict__ cache_v,
                                                        int physical_page, int kv_head, int d,
                                                        int page_offset, std::uint8_t* k_dst,
                                                        std::uint8_t* v_dst) {
    static_assert(Codec == GqaSimtKvCodec::Nvfp4,
                  "only the NVFP4 code plane has a written address path in this seam "
                  "(gqa_simt_kv_codec_routable)");
    using T = GqaSimtKvCodecTraits<Codec>;
    const std::int64_t k_off =
        gqa_kv_nvfp4_code_index<Geometry>(physical_page, kv_head, d, page_offset);
    const std::int64_t v_off = k_off; // K and V are separate tensors, same geometry
    if constexpr (T::kCodeBitsPerEl == 4) {
        // 4 bytes = 8 nibbles. The one true 32-bit load, so the chunk lands in one transaction.
        *reinterpret_cast<std::uint32_t*>(k_dst) =
            *reinterpret_cast<const std::uint32_t*>(cache_k + k_off);
        *reinterpret_cast<std::uint32_t*>(v_dst) =
            *reinterpret_cast<const std::uint32_t*>(cache_v + v_off);
    } else {
        *reinterpret_cast<std::uint64_t*>(k_dst) =
            *reinterpret_cast<const std::uint64_t*>(cache_k + k_off);
        *reinterpret_cast<std::uint64_t*>(v_dst) =
            *reinterpret_cast<const std::uint64_t*>(cache_v + v_off);
    }
}

template <typename Geometry, GqaSimtKvCodec Codec>
__device__ __forceinline__ void gqa_simt_ffma_load_scale(const std::uint8_t* __restrict__ cache_k,
                                                         const std::uint8_t* __restrict__ cache_v,
                                                         int physical_page, int kv_head, int group,
                                                         int page_offset, std::uint8_t* k_dst,
                                                         std::uint8_t* v_dst) {
    static_assert(Codec == GqaSimtKvCodec::Nvfp4,
                  "only the NVFP4 scale plane has a written address path in this seam");
    const std::int64_t off =
        gqa_kv_nvfp4_scale_index<Geometry>(physical_page, kv_head, group, page_offset);
    k_dst[0] = cache_k[off];
    v_dst[0] = cache_v[off];
}

// ===========================================================================
// ⚠ THE APPEND ORDERING -- THE ONE FACT THIS ARM DOES NOT HAVE, AND THEREFORE THE ONE THING IT
// REFUSES RATHER THAN GUESSES.
//
// The bf16 arm reads the CURRENT round's row straight out of `input` (the append source, which the
// engine keeps in bf16), and its comment states the invariant it buys: "the current attention
// reads those rows straight out of `input` below, so no split depends on another split's cache
// write". A CODEC arm cannot use `input`, because `input` is bf16 and a packed tile is not --
// it must read the packed row out of the cache plane. So the codec arm depends on the opposite
// invariant: that the append quantization has ALREADY COMMITTED that row's packed bytes before
// this kernel stages them.
//
// WHETHER THAT HOLDS IS NOT ESTABLISHED HERE, AND IT IS NOT A SMALL QUESTION: if the packed row
// is not yet written, the codec arm computes attention against a stale or zero K row and returns
// a plausible number. That is the silent wrong this tree refuses. It is also not something this
// header can decide, because it is an ordering property of the engine's append path and the
// split scheduler, not of this file.
//
// SO THE COMBINATION IS A COMPILE-TIME REFUSAL, not a runtime one: a codec arm with
// `CacheInput::writes_cache == true` does not compile. `gqa_attention_simt_ffma_refusal` carries
// the same refusal for the host, so a launcher reports it by name instead of failing to build.
// ⚠ WHAT THIS MEANS FOR SERVABILITY, STATED SO IT IS NOT OVERREAD: the codec arm serves a build
// that READS a packed cache and does not append into it (`writes_cache == false`). Whether the
// engine's decode ever runs in that shape is a route-table question this line did not settle --
// see REPORT.md's NOT DETERMINED list. It is NOT a claim that packed KV decode works end to end.
// ===========================================================================

// The named refusal, as a constant expression, so the launcher and the kernel cannot disagree.
[[nodiscard]] __host__ __device__ constexpr bool gqa_simt_kv_codec_needs_committed_cache(bool writes_cache) {
    return writes_cache;
}

// ---------------------------------------------------------------------------
// THE WINDOW BOUNDARY -- WHEN A DECLARED SLIDING WINDOW CANNOT MOVE THIS ARM'S KEYS.
// (dl/ffmawindow. This block is the reason the route can serve qwen3_6_27b at all.)
//
// This family has no window term, so the only question it can answer about a declared window is
// whether that window CHANGES the set of keys it attends. It cannot, when the window is at least
// as large as the largest key index this kernel services -- and that bound is the kernel's own
// guard, not a guess: the body writes the neutral partial and returns for
// `last_pos >= logical_capacity` (:1329), so every launch it services has
// `last_pos + 1 <= logical_capacity`. Take the replaced arm's own frame at such a launch
// (`gqa_attention_decode_nvfp4.cuh:381-385`, restated here because those helpers are TU-local to
// that header: `token_begin = (sliding_window > 0) ? (last_pos + 1) - W : 0`,
// `window_begin = max(token_begin, 0)`, `window = (last_pos + 1) - window_begin`):
//   * `window_begin` = max(0, (last_pos + 1) - W) <= max(0, logical_capacity - W) = 0;
//   * so `window` = last_pos + 1, `stage_begin` = 0, `stage_skew` = 0;
//   * so its tile frame `((split_start + skew) / Bc) * Bc` is `(split_start / Bc) * Bc` here,
//     its page span `((window_begin + split_end - 1) >> shift) - first_global_page + 1` is this
//     family's `((split_end - 1) >> shift) - first_page + 1`, and its masks
//     `key >= window_begin + split_start && key < window_begin + split_end` are this family's
//     `key >= split_start && key < split_end`.
// That is the converse table of dl/ffmaroute REPORT.md 3.4b, sentence for sentence: with the
// origin at 0 the two arms enumerate the same absolute keys, apply the same masks and produce
// the same answer. Serving such a launch is therefore not a guess, it is the identity.
// Where `0 < W < logical_capacity` the window CAN clip, this family still has no term for it, and
// the launch is refused by name -- see gqa_simt_ffma_codec_refusal's window clause.
//
// WHY IT IS A NAMED PREDICATE AND A WITNESS TABLE: a comment is not a check. The table below
// asserts the boundary on the same expression the flag is computed from, including the two cases
// that would make the predicate lie (equality at the bound must NOT refuse; one token past it
// must), and it asserts the consequence on the replaced arm's origin for the LAST key this
// kernel can service. Deleting a case removes the evidence, not the boundary.
//
// REACHABILITY OF THE TWO SIDES ON qwen3_6_27b (G2, READ: targets/qwen3_6_27b/impl/config.h:43
// W = 646,720 declared on all 16 paged layers, :33-41 "for every context <= 646,720 this is
// bit-identical to full attention"; :198 kNativeContext = 262,144; the yarn bound
// include/ninfer/ops/softmax_attention.h:18 is 4 * 262144 = 1,048,576 and
// include/ninfer/ops/gqa_attention.h:20 kGqaAttentionMaximumVisibleKeys = 1,048'576, raised from
// 1'010'000 -- so the band (1'010'000, 1'048'576] is admitted by that header and witnessed below):
//   * every launch whose key bound is <= 646,720 -- the model's whole native range and everything
//     up to the window itself -- is SERVED, because the window cannot clip it;
//   * the 1M band (context > 646,720, which is the only place the window changes an answer on
//     this variant, config.h:38-41) is REFUSED by name.
// `logical_capacity` is the live round's key bound (ops/launcher/gqa_attention_decode_smallt.cu:123
// and :135), so this boundary is the per-launch one, not a per-variant guess.
[[nodiscard]] __host__ __device__ constexpr bool gqa_simt_ffma_window_clips(std::int32_t window_tokens,
                                                                           std::int32_t logical_capacity) {
    // The family's sentinel for "no window" is <= 0; the kernel's serviced key bound is
    // `logical_capacity`; at or above it the origin cannot leave 0.
    return window_tokens > 0 && window_tokens < logical_capacity;
}

struct GqaSimtFfmaWindowWitness {
    std::int32_t window_tokens;
    std::int32_t logical_capacity;
    bool clips;
};
// Every row is a launch this route can be handed. `clips` is what the predicate must say, and the
// check also recomputes the replaced arm's origin at the LAST key this kernel services
// (`last_pos = logical_capacity - 1`) and requires the two to agree -- so a predicate that just
// returned `true` (or just `false`) fails the build.
constexpr GqaSimtFfmaWindowWitness kGqaSimtFfmaWindowWitness[] = {
    {0, 262144, false},        // no window declared (the field is 0)
    {-1, 262144, false},       // the "no window" sentinel
    {4096, 262144, true},      // NINFER_KV_WINDOW_TOKENS=4096 at native context: clips
    {262144, 262144, false},   // W == the bound: origin is 0 at every serviced key
    {646720, 262144, false},   // qwen3_6_27b W=646720 at its native context
    {646720, 646720, false},   // W == the round's last visible key + 1
    {646720, 646721, true},    // one key past: the origin is 1 at last_pos = 646720
    {646720, 1048576, true},   // the 1M band (yarn bound): clips
    {1010000, 1010000, false}, // W == the OLD bound; still a launch this route sees
    {1048576, 1048576, false}, // W == kGqaAttentionMaximumVisibleKeys (raised from 1'010'000)
    {1048575, 1048576, true},  // one key inside it: the boundary the raise newly admits
};
constexpr int kGqaSimtFfmaWindowWitnessCount = 11;
static_assert(sizeof(kGqaSimtFfmaWindowWitness) / sizeof(kGqaSimtFfmaWindowWitness[0]) ==
                  kGqaSimtFfmaWindowWitnessCount,
              "dl/ffmawindow: the window-boundary witness table lost a case. These cases ARE the "
              "check that this route refuses exactly where a declared window can move the key "
              "frame; deleting one to make the build pass removes the check, not the boundary.");

constexpr bool gqa_simt_ffma_window_contract_holds() {
    for (int i = 0; i < kGqaSimtFfmaWindowWitnessCount; ++i) {
        const std::int32_t w = kGqaSimtFfmaWindowWitness[i].window_tokens;
        const std::int32_t c = kGqaSimtFfmaWindowWitness[i].logical_capacity;
        const bool want      = kGqaSimtFfmaWindowWitness[i].clips;
        if (c < 1) { return false; }  // no key space: not a launch this kernel services
        const bool got = gqa_simt_ffma_window_clips(w, c);
        if (got != want) { return false; }
        // The replaced arm's origin at the last key this kernel services, on the expression quoted
        // above. The predicate must agree with it, in both directions.
        const std::int32_t last_pos    = c - 1;
        const std::int32_t token_begin = (w > 0) ? (last_pos + 1) - w : 0;
        const std::int32_t origin      = token_begin > 0 ? token_begin : 0;
        if (got != (origin != 0)) { return false; }
        if (!got) {
            // The rest of the replaced frame, at the same launch: window stays the full prefix and
            // the staging frame carries no skew, which is this family's frame.
            if ((last_pos + 1) - origin != last_pos + 1) { return false; }
            if (w > 0 && w < c) { return false; }  // unreachable: got would have been true
        }
    }
    return true;
}
static_assert(gqa_simt_ffma_window_contract_holds(),
              "dl/ffmawindow: the sliding-window boundary is not the kernel's serviced key bound. "
              "Either the predicate refuses a launch whose origin cannot leave 0 (that launch's "
              "answer would be bit-identical to the arm this route replaces), or it serves one "
              "whose origin can (that launch would be attended outside the declared window).");

// THE CODEC-AWARE REFUSAL, for the launcher. It is a SEPARATE function from
// `gqa_attention_simt_ffma_refusal` above on purpose: that one is the family's original refusal
// (the cold pool) and its signature is referenced by the landed arm, while this one is the codec
// question, which needs types that are declared only from here down. Two functions, two
// questions -- the same separation the selector keeps between "which arm" and "on whose word".
// Returns nullptr when the launch is servable by the codec arm, else the refusal text.
// ===========================================================================
// THE CODEC ROUTE'S REFUSALS, IN ONE PLACE. Two of the five are this line's (dl/ffmaroute); the
// first three were the seam's and are unchanged in substance.
//
// WHY THIS IS A FUNCTION AND NOT A COMMENT: a refusal that exists only in prose is a comment that
// the code that would have to report it cannot read. Every caller of this route asks it FIRST, so
// a launch it cannot serve is a NAMED error and never a wrong number.
//
// (1) NOT DECODABLE / NOT ROUTABLE -- unchanged (see gqa_simt_kv_codec_supported and
//     gqa_simt_kv_codec_routable; E8Kv is the named refusal there).
// (2) A DECLARED SLIDING WINDOW. NEW, AND IT IS THE ONE THAT DECIDES BETWEEN A NAMED ERROR AND
//     SILENT CORRUPTION. The NVFP4 arm this route replaces READS `cache.sliding_window_tokens`
//     and shifts its whole key frame by it (gqa_attention_decode_nvfp4.cuh:381-385:
//     `token_begin = window_full - sliding_window`, `window_begin = gqa_wm_nvfp4_window_origin(
//     token_begin)`, `window = window_full - window_begin`). This family does not: it has no
//     sliding-window term at all and its key range is [0, last_pos] --
//     ops/kernel/gqa_attention_simt_ffma.cuh, `sliding_window` appears 0 times, and the bf16
//     partial kernel it was modelled on is the same (`gqa_attention_decode_bf16.cuh:153`
//     `window = last_pos + 1`). The tree already says what that combination costs, twice:
//       * src/product/kv_component_switch.h:14-24 lists `gqa_attention_simt_ffma.cuh` among the
//         tiers that IGNORE the field and calls the consequence, in those words, SILENT
//         CORRUPTION -- because `cold_host_page_is_read_free()` releases a device page to the host
//         on the strength of a window the kernel then does not apply, and the kernel reads the
//         released page anyway.
//       * src/targets/qwen3_6_27b/impl/config.h:95-100 says the same and adds that those tiers are
//         refused BY NAME at layout time.
//     THAT LAYOUT-TIME REFUSAL DOES NOT COVER THIS ROUTE, and that is why this one has to exist:
//     `kv_window_tier_honoured()` (kv_component_switch.h:32-35) keys on the layer's RESOLVED KV
//     DTYPE, and NVFP4 is one of the two dtypes it admits. So a windowed NVFP4 layer passes the
//     plan-time check -- correctly, because the tensor-core nvfp4 kernel honours the window --
//     and then, on a rung where this family is selected, the LAUNCH is what would silently
//     ignore it. The dtype is admitted; the ARM is not. Only the arm can refuse this.
//     REACHABILITY IS NOT HYPOTHETICAL: on qwen3_6_27b every paged layer declares a window
//     (layouts_impl.h:264, "16/16 non-zero") of 646,720 tokens, and the value is a no-op only
//     while the context is at or below it.
//     dl/ffmawindow SHARPENED THIS REFUSAL TO THAT BOUNDARY. `window_declared` was
//     `sliding_window_tokens != 0`, which refused every launch of that model including the ones
//     whose window cannot clip -- i.e. every context at or below 646,720, which is the whole
//     native range (kNativeContext = 262,144, targets/qwen3_6_27b/impl/config.h:198) and more.
//     The flag is now `gqa_simt_ffma_window_clips(sliding_window_tokens, logical_capacity)`: the
//     refused class is exactly the launches where the replaced arm's origin can leave 0, and the
//     served class is the one where the converse table of dl/ffmaroute REPORT.md 3.4b makes this
//     family's key frame the declared one, key for key. The kernel was NOT changed to earn that:
//     it still has no window term, and it still refuses every launch it cannot honour. The
//     boundary rests on the kernel's own serviced-key guard (:1329 `last_pos >= logical_capacity`
//     -> neutral partial), so it cannot be widened by changing a caller's idea of the bound.
// (3) AN ARMED RESIDUAL PLANE PAIR. Also this line's. The NVFP4 arm this route replaces takes
//     four residual pointers and adds the second-stage term to its reconstruction; this family
//     takes none of them, so it would decode K/V from the CODE plane alone and produce a
//     plausible, systematically different answer. The seam's own text already named this
//     ("NOT IN SCOPE ... the NVFP4 V residual plane") but named it in a comment: no refusal
//     existed, and `arm --kv-residual-layers` (serve_options.cpp:471) is an operator input, so
//     the combination is reachable. It is refused here instead.
// (4) AN APPENDING LAUNCH -- unchanged, see the APPEND ORDERING block.
// (5) AN ARMED COLD-SLOT POOL -- unchanged.
// ===========================================================================
[[nodiscard]] inline const char* gqa_simt_ffma_codec_refusal(GqaSimtKvCodec codec, bool cold_armed,
                                                             bool writes_cache, bool window_clips,    
                                                             bool residual_armed) {
    if (!gqa_simt_kv_codec_supported(codec)) {
        return "the SIMT FFMA attention route has no codec arm for this KV dtype: E8Kv/rk4v4 is "
               "REFUSED BY NAME (its reader emits a ROTATED K that is only exact against "
               "e8_kv_lattice_reader_scale, so a per-chunk dequant would answer a different "
               "question) and every dtype outside {bf16, nvfp4, fp8, iso3, i8} has no decode "
               "here at all";
    }
    if (!gqa_simt_kv_codec_routable(codec)) {
        return "the SIMT FFMA attention route can DECODE this KV dtype in pure ALU but has no "
               "written, tested address path for its plane pair, so the route is refused rather "
               "than guessed: only NVFP4's code/scale index helpers have been matched against "
               "this family's row/plane arithmetic (bf16 needs none; fp8 and i8 read a scale "
               "plane with a different leading extent; iso3 is reached through its own tier's "
               "reader)";
    }
    if (window_clips) {
        return "the SIMT FFMA attention route has NO sliding-window term, and this launch declares "
               "one that CAN CLIP the keys the kernel serves (0 < cache.sliding_window_tokens < "
               "the kernel's own key bound; see gqa_simt_ffma_window_clips): the NVFP4 tensor-core "
               "arm this route replaces shifts its key frame by that window "
               "(gqa_attention_decode_nvfp4.cuh:381-385) while this family's key range is "
               "[0, last_pos] unconditionally (src/product/kv_component_switch.h:235-302 lists "
               "gqa_attention_simt_ffma.cuh among the tiers that ignore the field and calls the "
               "consequence SILENT CORRUPTION, because the Cold Host tier releases a page on the "
               "strength of a window the kernel then does not apply and the kernel reads the "
               "released page anyway). A window at or above that bound is NOT refused here: its "
               "origin cannot leave 0, so this family attends exactly the declared keys and its "
               "answer is bit-identical to the replaced arm's. The plan-time refusal in "
               "layouts_impl.h does NOT cover either case: it admits any layer whose resolved "
               "dtype is NVFP4, which is this layer's dtype; it is the ARM, not the dtype, that "
               "cannot honour the window. Refused by name rather than attended outside the "
               "declared window";
    }
    if (residual_armed) {
        return "the SIMT FFMA attention route has no residual-plane arm: the NVFP4 tensor-core arm "
               "this route replaces reads a second-stage residual pair for K and V and adds it to "
               "its reconstruction, while this family takes no residual pointer at all, so it "
               "would decode from the code plane alone -- a plausible, systematically different "
               "answer rather than a fault. An operator-armed residual (--kv-residual-layers) on "
               "an NVFP4 layer is therefore refused by name";
    }
    if (codec != GqaSimtKvCodec::Bf16 && gqa_simt_kv_codec_needs_committed_cache(writes_cache)) {
        return "the SIMT FFMA attention route cannot stage a packed KV plane from a bf16 append "
               "source, and this arm does not establish that the append quantization has already "
               "committed the row: a launch that appends into a packed cache must not be routed "
               "here";
    }
    if (codec != GqaSimtKvCodec::Bf16 && cold_armed) {
        return "the SIMT FFMA attention route has no cold-slot codec: a layer with an armed "
               "cold-slot pool must not be routed here";
    }
    return nullptr;
}

// ===========================================================================
// THE PACKED NVFP4 PREFILL ARM'S OWN SELECTION (dl/nvfp4emu2).
//
// WHY A SECOND SELECTOR AND NOT A ROW IN THE ONE ABOVE. `SimtFfmaSelection` answers "which
// attention family does this launch run?" -- one process-wide answer, and it is already the answer
// the decode codec arm consumes (ops/launcher/gqa_attention_decode_partial.cuh:590). It does NOT
// answer the question THIS arm asks, which is asked only AFTER the family is selected: "may a
// PREFILL launch stage a PACKED NVFP4 tile, or must it be refused by name?". The two answers can
// disagree, and they disagree in the direction that matters: a build can have the family selected
// (forced on, or measured) and still have no prefill codec arm armed here.
//
// WHAT WAS MISSING, NAMED -- this is the gap this block exists to make answerable. The landed FFMA
// prefill body, ops::gqa_attention_simt_ffma_prefill_bf16_kernel, was bf16-only in three places
// that are ONE fact:
//   (1) its template parameter list had no codec, so a packed tile had no encoder to be staged by;
//   (2) BOTH of its plane parameters were `const __nv_bfloat16*`, so an NVFP4 code plane handed to
//       it would have been decoded as bf16 halves -- a plausible wrong number, not a fault;
//   (3) its only staging call was gqa_simt_ffma_stage_tile (one bf16 element per 8 dims) and its
//       only row pass was gqa_simt_ffma_row_pass (`__bfloat162float(k_row[i])`).
// So the route could not FEED the codec this same file already carries:
// gqa_simt_ffma_row_pass_packed (:702) and gqa_simt_ffma_stage_tile_packed (:803) were landed and
// used by the DECODE body only. The prefill body now reaches them through exactly the same
// `if constexpr (Codec == GqaSimtKvCodec::Bf16)` seam the decode body uses (:1523, :1608).
//
// THE SHAPE, COPIED FROM SimtFfmaSelection ON PURPOSE, ONE PROPERTY PER LINE:
//   * A NAMED REASON VALUE, not a bool. gqa_simt_ffma_prefill_codec_selection_text() has a case
//     for every member, and the not-measured one says NOT MEASURED and names the build default.
//   * THE ENV OVERRIDE IS ITS OWN FUNCTION AND IS REPORTED SEPARATELY. An operator who set
//     NINFER_PREFILL_NVFP4_SIMT has overridden the probe, and a report that says "the probe
//     answered X" while the env var is what decided is the same lie in a smaller font.
//   * AN UNPROBED PROCESS IS NEVER PRESENTED AS A MEASURED ONE: NotProbed falls to the BUILD
//     default, and this arm's build default is OFF (NINFER_PREFILL_NVFP4_SIMT_DEFAULT_ON is not
//     defined anywhere), so every existing launch is unchanged.
//   * THE ANNOUNCEMENT IS KEYED ON THE SELECTED VALUE (core/announce_once.h), because the header
//     it replaces, `static bool`, announced the first value it saw and went silent -- a SECOND,
//     different value reached nobody.
//
// ⚠ WHAT THIS SELECTOR CANNOT DO: it cannot move a launch PAST a refusal. The codec refusals run
// FIRST (gqa_simt_ffma_prefill_codec_refusal below), and they include the same named refusals the
// decode codec arm uses. This arm therefore ADDS a route and widens none.
// ===========================================================================
enum class GqaSimtFfmaPrefillCodecProbeAnswer : std::uint8_t {
    // The probe has not run on the bound device. NOT a measurement, and it is not reported as one.
    NotProbed = 0,
    // The probe ran BOTH prefill bodies (the landed bf16 one and this packed NVFP4 one) on this
    // device, on identical deterministic packed input, and they agreed to the tolerance the probe
    // publishes. ⚠ Nothing calls the publisher below yet -- see its comment.
    RanOk = 1,
    // The probe ran and did not pass: a body did not launch, or it launched and disagreed.
    Failed = 2,
};

[[nodiscard]] inline std::string_view gqa_simt_ffma_prefill_codec_probe_answer_name(
    GqaSimtFfmaPrefillCodecProbeAnswer answer) {
    switch (answer) {
    case GqaSimtFfmaPrefillCodecProbeAnswer::NotProbed: return "not probed";
    case GqaSimtFfmaPrefillCodecProbeAnswer::RanOk:
        return "both prefill row passes (the landed bf16 D=128/D=256 body and the packed NVFP4 "
               "codec this arm adds) ran on this device and agreed within the probe's tolerance";
    case GqaSimtFfmaPrefillCodecProbeAnswer::Failed:
        return "a prefill row pass did not run or did not agree with its reference";
    }
    return "?";
}

// WHERE THE ANSWER LIVES, AND WHY IT COSTS NO LINK EDGE. One object per process, shared by every
// TU: a function-local static inside an `inline` function is the ODR's single instance, so a
// writer in core/device_probe.cu and a reader in a launcher TU address the SAME object, and no
// launcher object grows a link dependency just to ask a route question.
[[nodiscard]] inline GqaSimtFfmaPrefillCodecProbeAnswer&
gqa_simt_ffma_prefill_codec_probe_state() {
    static GqaSimtFfmaPrefillCodecProbeAnswer answer =
        GqaSimtFfmaPrefillCodecProbeAnswer::NotProbed;
    return answer;
}

// The only writer. ⚠ NAMED, NOT HIDDEN: nothing in this tree calls it today (the decode codec
// arm's probe is core/device_probe.cu's DeviceCapability::SimtFfmaAttention, which measures the
// two ROW PASSES and not a full prefill launch). Until something does, the state is NotProbed and
// this arm is OFF by the build default -- which is the honest reading and NOT a measurement.
inline void gqa_attention_simt_ffma_prefill_codec_publish_probe_answer(
    GqaSimtFfmaPrefillCodecProbeAnswer answer) {
    gqa_simt_ffma_prefill_codec_probe_state() = answer;
}

// The env override, factored OUT of the decision so the two can be reported separately.
// SINGLE-CHARACTER ONLY, the same discipline as gqa_attention_simt_ffma_env_override: `=1x`,
// `=10` and `=01` are NOT overrides and return -1 (unset), rather than being read as a 1.
// Returns -1 when unset, else 0/1.
[[nodiscard]] inline int gqa_simt_ffma_prefill_codec_env_override() {
    const char* env = std::getenv("NINFER_PREFILL_NVFP4_SIMT");
    if (env != nullptr && env[0] != '\0' && env[1] == '\0') {
        if (env[0] == '1') { return 1; }
        if (env[0] == '0') { return 0; }
    }
    return -1;
}

// WHY the selector answered what it answered. Anything that PRINTS the route reads this instead of
// guessing from the bool: "armed" alone cannot distinguish a measured answer from the unprobed
// build default, and that difference is the whole point.
enum class GqaSimtFfmaPrefillCodecSelection : std::uint8_t {
    ForcedOn,             // NINFER_PREFILL_NVFP4_SIMT=1 (operator override)
    ForcedOff,            // NINFER_PREFILL_NVFP4_SIMT=0 (operator override)
    ProbeRanOk,           // measured: the probe ran both prefill bodies here and they agreed
    ProbeFailed,          // measured: do not take this arm on this device
    UnprobedBuildDefault, // NOT measured: the probe has not run (NINFER_PREFILL_NVFP4_SIMT_DEFAULT_ON)
};

[[nodiscard]] inline const char* gqa_simt_ffma_prefill_codec_selection_text(
    GqaSimtFfmaPrefillCodecSelection how) {
    switch (how) {
    case GqaSimtFfmaPrefillCodecSelection::ForcedOn:
        return "NINFER_PREFILL_NVFP4_SIMT=1 (operator override)";
    case GqaSimtFfmaPrefillCodecSelection::ForcedOff:
        return "NINFER_PREFILL_NVFP4_SIMT=0 (operator override)";
    case GqaSimtFfmaPrefillCodecSelection::ProbeRanOk:
        return "MEASURED by the prefill-codec probe: both prefill row passes ran on this device "
               "and agreed within the probe's tolerance";
    case GqaSimtFfmaPrefillCodecSelection::ProbeFailed:
        return "MEASURED by the prefill-codec probe: a prefill row pass did not run or did not "
               "agree on this device";
    case GqaSimtFfmaPrefillCodecSelection::UnprobedBuildDefault:
        return "NOT MEASURED: the prefill-codec probe has not run on this device, so this is the "
               "BUILD default (NINFER_PREFILL_NVFP4_SIMT_DEFAULT_ON, undefined => OFF)";
    }
    return "?";
}

// The same decision, with its REASON. ORDER: operator override, then the PROBE, then the build
// default -- the same order the family selector above uses, so the two cannot drift.
//
// ⚠ It does NOT consult gqa_attention_simt_ffma_selected(). That is the caller's job and it is
// deliberate: this arm is a REFINEMENT of the family decision, so forcing it on must not bypass
// the family. The launcher asks the family FIRST and this second.
[[nodiscard]] inline GqaSimtFfmaPrefillCodecSelection gqa_simt_ffma_prefill_codec_selection() {
    const int override_value = gqa_simt_ffma_prefill_codec_env_override();
    if (override_value == 1) { return GqaSimtFfmaPrefillCodecSelection::ForcedOn; }
    if (override_value == 0) { return GqaSimtFfmaPrefillCodecSelection::ForcedOff; }
    switch (gqa_simt_ffma_prefill_codec_probe_state()) {
    case GqaSimtFfmaPrefillCodecProbeAnswer::RanOk:
        return GqaSimtFfmaPrefillCodecSelection::ProbeRanOk;
    case GqaSimtFfmaPrefillCodecProbeAnswer::Failed:
        return GqaSimtFfmaPrefillCodecSelection::ProbeFailed;
    case GqaSimtFfmaPrefillCodecProbeAnswer::NotProbed:
        return GqaSimtFfmaPrefillCodecSelection::UnprobedBuildDefault;
    }
    return GqaSimtFfmaPrefillCodecSelection::UnprobedBuildDefault;
}

// The host-side switch. TRUE only on a measured or an operator-forced answer; the unprobed build
// default is OFF, from a macro that is defined nowhere in this tree.
[[nodiscard]] inline bool gqa_simt_ffma_prefill_codec_selected() {
    switch (gqa_simt_ffma_prefill_codec_selection()) {
    case GqaSimtFfmaPrefillCodecSelection::ForcedOn:
    case GqaSimtFfmaPrefillCodecSelection::ProbeRanOk: return true;
    case GqaSimtFfmaPrefillCodecSelection::ForcedOff:
    case GqaSimtFfmaPrefillCodecSelection::ProbeFailed: return false;
    case GqaSimtFfmaPrefillCodecSelection::UnprobedBuildDefault:
#if defined(NINFER_PREFILL_NVFP4_SIMT_DEFAULT_ON)
        return true;
#else
        return false;
#endif
    }
    return false;
}

// The announcement, KEYED ON THE SELECTED VALUE. Returns true when this value had not been
// announced before; the caller prints on true and does nothing else (the contract
// core/announce_once.h states in one line). ⚠ It is an OBSERVATION, not a knob: there is no
// env var that silences it, and it cannot be cleared.
[[nodiscard]] inline bool gqa_simt_ffma_prefill_codec_announce_once() {
    return ninfer::detail::announce_once_keyed(gqa_simt_ffma_prefill_codec_selection());
}

// THE PREFILL CODEC ARM'S REFUSALS, IN ONE PLACE. It DELEGATES to the shared codec refusal for
// everything that is a property of the CODEC (not-decodable / not-routable / cold pool / residual
// pair / append ordering) and then adds THE ONE CLAUSE THAT IS A PREFILL QUESTION.
//
// ⚠ WHY THE WINDOW CLAUSE IS NOT THE DECODE ONE. gqa_simt_ffma_codec_refusal's window clause is a
// BOUNDARY (gqa_simt_ffma_window_clips): it refuses only when `0 < W < logical_capacity`, because
// on a DECODE launch `logical_capacity` is a HOST value and the replaced arm's origin is then
// provably 0. A PREFILL launch has no such bound available here: the largest absolute key index it
// services is `base_pos + tokens - 1`, and `base_pos` lives in the `positions` DEVICE tensor
// (read at gqa_attention_simt_ffma.cuh's prefill body as `positions[0]`), so a host-side bound
// would need a device-to-host read this launcher does not take. `PagedKVLayerView` and
// `PagedKVBatchLayerView` carry NO capacity or context field (read at src/core/paged_kv_cache.h:35
// -62 and :65-88, not assumed). So this arm refuses EVERY declared window by name rather than
// attending outside it. That is a STRICTER gate than the decode arm's, i.e. it widens nothing.
//
// The tensor-core NVFP4 prefill kernel this arm replaces DOES read the window: it is passed
// `static_cast<int>(cache.sliding_window_tokens)` at gqa_attention_prefill.cu:169 and :186.
//
// BOUNDED FOLLOW-UP, NAMED: thread the same bound in from the caller's capacity, or take the one
// positions[0] device-to-host read, and then use gqa_simt_ffma_window_clips exactly as the decode
// arm does. Neither is done here, and neither is assumed.
[[nodiscard]] inline const char* gqa_simt_ffma_prefill_codec_refusal(bool cold_armed,
                                                                   bool window_declared,
                                                                   bool residual_armed) {
    // (1) everything that is a property of the codec. `writes_cache` is false by CONSTRUCTION on
    //     this route and not by assumption: the attention body is a SEPARATE launch from the append
    //     fill (gqa_kv_append_launch_for at gqa_attention_prefill.cu:292, whose packed fill kernels
    //     are launched at :356-382), and the two run on one stream in that order, so the packed row
    //     this body stages is committed before it is read. That is the invariant the decode arm's
    //     APPEND ORDERING block could not establish for a fused append-decode launch.
    const char* const shared = gqa_simt_ffma_codec_refusal(GqaSimtKvCodec::Nvfp4, cold_armed,
                                                           /*writes_cache=*/false,
                                                           /*window_clips=*/false, residual_armed);
    if (shared != nullptr) { return shared; }
    if (window_declared) {
        return "the SIMT FFMA PREFILL codec arm has NO sliding-window term, and this launcher "
               "cannot decide whether the layer's declared window can clip: the largest absolute "
               "key index a prefill launch services is base_pos + tokens - 1 and base_pos lives in "
               "the `positions` DEVICE tensor, so a host-side bound would need a device-to-host "
               "read this launcher does not take (PagedKVLayerView and PagedKVBatchLayerView carry "
               "no capacity field, src/core/paged_kv_cache.h:35-62 and :65-88). The tensor-core "
               "NVFP4 prefill kernel this arm replaces DOES read the window (gqa_attention_prefill"
               ".cu:169), so the two arms would disagree on a window that clips. Refused by name "
               "rather than attended outside the declared window";
    }
    return nullptr;
}

// M1 (MTP tree verify): the round-relative visibility rule, spelled exactly as
// gqa_attention_prefill_bf16.cuh:115 spells it (bounded shift: a uint64_t shift of 64 or more is
// undefined, and the exact answer for such a key is "no ancestor bit is set"). The decode body of
// the tensor-core family states the same rule unbounded, which it can, because a small-T launch
// clips every key it addresses to the round's own window; the bound costs nothing and is defined
// for every input, so this family uses the bounded form in both bodies.
__device__ __forceinline__ bool gqa_simt_round_visible(std::uint64_t mask, int rel) {
    return rel < 0 || (rel < 64 && ((mask >> static_cast<unsigned>(rel)) & std::uint64_t{1}) != 0);
}

// One query row's tile pass: scores for the whole [Bc] key tile, the softmax update of the
// caller's (m, l, acc) registers, and the P V accumulation. It is shared by both kernels: the
// decode and prefill bodies differ only in which rows they carry and in the visibility rule they
// hand in. `acc` is indexed by the lane's own dims, so no cross-lane traffic happens here at all.
template <typename Geometry, int Bc>
__device__ __forceinline__ void gqa_simt_ffma_row_pass(
    const float (&qr)[Geometry::HeadDim / kGqaSimtFfmaWarpReduction],
    const __nv_bfloat16* __restrict__ k_s, const __nv_bfloat16* __restrict__ v_s, int lane,
    int tile_key0, int qabs, std::uint64_t mask, bool round_masked, int column_begin, int first_pos,
    int split_start, int split_end, float scale, float& m, float& l,
    float (&acc)[Geometry::HeadDim / kGqaSimtFfmaWarpReduction]) {
    constexpr int D    = Geometry::HeadDim;
    constexpr int VecD = D / kGqaSimtFfmaWarpReduction;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    const int d_lane = lane * VecD;
    float part[Bc];
#pragma unroll
    for (int kk = 0; kk < Bc; ++kk) {
        float s                    = 0.0f;
        const __nv_bfloat16* k_row = &k_s[kk * D + d_lane];
#pragma unroll
        for (int i = 0; i < VecD; ++i) {
            s = __fmaf_rn(qr[i], __bfloat162float(k_row[i]), s);
        }
        part[kk] = s;
    }
    // One butterfly per key: the row's dot product is split across the 32 lanes.
#pragma unroll
    for (int kk = 0; kk < Bc; ++kk) {
        part[kk] = warp_sum<kGqaSimtFfmaWarpReduction>(part[kk], FullMask);
    }

    float tile_max = -CUDART_INF_F;
#pragma unroll
    for (int kk = 0; kk < Bc; ++kk) {
        const int key     = tile_key0 + kk;
        const int rel     = column_begin + key - first_pos;
        const bool visible = (key >= split_start && key < split_end && key <= qabs) &&
                             (!round_masked || gqa_simt_round_visible(mask, rel));
        part[kk]  = visible ? part[kk] * scale : -CUDART_INF_F;
        tile_max  = fmaxf(tile_max, part[kk]);
    }

    const float nm = fmaxf(m, tile_max);
    // The `m == -inf` arm is load-bearing: -inf - (-inf) is NaN, and exp2_approx(NaN) is NaN, so
    // the first tile of a row whose split has no visible key yet must not take it. The
    // tensor-core bf16 partial kernel guards the identical expression.
    const float alpha = (m == -CUDART_INF_F) ? 0.0f : exp2_approx((m - nm) * Log2E);
    // RESCALE FIRST, THEN ACCUMULATE. The other order (accumulate, then `acc *= alpha`) is what
    // the first version of this file did, and it scales the NEW contributions by alpha as well:
    // alpha*(old + new) instead of alpha*old + new. It is invisible on a single key tile and it
    // is exactly why the measured error grew with the key count -- 1.1e-3 at one tile (3 visible
    // keys), 2.3e-2 at T=64 and 9.8e-2 at a 190-key window, while the tensor-core arm stayed at
    // ~2e-3 at every size (recon/n1_numerics.txt). alpha == 0 means "this row had no visible key
    // yet", and its accumulator is 0 by construction, so zeroing is the same statement.
    if (alpha == 0.0f) {
#pragma unroll
        for (int i = 0; i < VecD; ++i) { acc[i] = 0.0f; }
    } else {
#pragma unroll
        for (int i = 0; i < VecD; ++i) { acc[i] *= alpha; }
    }
    float bl = 0.0f;
#pragma unroll
    for (int kk = 0; kk < Bc; ++kk) {
        const float p = (nm > -CUDART_INF_F && part[kk] > -CUDART_INF_F)
                            ? exp2_approx((part[kk] - nm) * Log2E)
                            : 0.0f;
        bl += p;
        const __nv_bfloat16* v_row = &v_s[kk * D + d_lane];
#pragma unroll
        for (int i = 0; i < VecD; ++i) {
            acc[i] = __fmaf_rn(p, __bfloat162float(v_row[i]), acc[i]);
        }
    }
    l = __fmaf_rn(l, alpha, bl);
    m = nm;
}

// Stage one [Bc, D] K or V key tile from the paged cache. Keys outside [split_start, split_end)
// are zeroed: a zero K row contributes 0 to every dot product and its score is dropped by the
// row pass's predicate, so an uninitialized/padded cache tail can never feed a NaN into the
// softmax. `physical_pages_s` is the page-id table the caller filled for this tile range.
template <typename Geometry, int Bc, typename StageSource>
__device__ __forceinline__ void gqa_simt_ffma_stage_tile(
    __nv_bfloat16* __restrict__ k_s, __nv_bfloat16* __restrict__ v_s, int tid, int threads,
    int k0, int split_start, int split_end, const StageSource& source) {
    constexpr int D = Geometry::HeadDim;
    for (int chunk = tid; chunk < Bc * (D / 8); chunk += threads) {
        const int key_l      = chunk / (D / 8);
        const int d          = (chunk - key_l * (D / 8)) * 8;
        const int key        = k0 + key_l;
        __nv_bfloat16* k_dst = &k_s[key_l * D + d];
        __nv_bfloat16* v_dst = &v_s[key_l * D + d];
        if (key >= split_start && key < split_end) {
            source(key, d, k_dst, v_dst);
        } else {
            store_vec(k_dst, make_int4(0, 0, 0, 0));
            store_vec(v_dst, make_int4(0, 0, 0, 0));
        }
    }
}

// ---------------------------------------------------------------------------
// DECODE: one split of the split-KV small-T bf16 decode, FFMA + online softmax.
//
// Contract (identical to gqa_attention_small_t_tc_partial_bf16_kernel, minus the cold slots):
// one CTA per (kv_head, split, batch) plus a ROW GROUP, block 128 threads, and every row the CTA
// carries writes its split-local partial: partial_m = the split's max SCALED score (the same
// units the tensor-core kernel uses: raw dot times `scale`), partial_l = the sum of
// exp2((s - m) * log2e) over the split's visible keys, and partial_acc = the same weighted sum
// against V. The reducer turns that into the output, so nothing here normalizes.
//
// grid = (KVHeads * row_groups, splits, batch), row_groups = ceil(tokens * group_size / Br).
// It does NOT take the five cold-slot parameters the tensor-core kernel takes: this minimum path
// reads the bf16 cache plane only, and the parameter list is where that omission is visible.
// ---------------------------------------------------------------------------
// One CTA's shared-memory arena, as the ONE expression the sizing, the staging and the row pass
// all read. The bf16 case is byte-identical to the two bf16 tiles this kernel declared before the
// codec seam existed (2 * Bc * D * 2, with a zero-byte scale plane), so the landed instantiation's
// smem number is unchanged by construction and not by a coincidence we have to keep checking --
// while a packed codec's arena is the packed plane pair that lifts the tile cap.
[[nodiscard]] __host__ __device__ constexpr int gqa_simt_kv_code_plane_bytes(GqaSimtKvCodec codec, int head_dim,
                                                         int bc) {
    return bc * gqa_simt_kv_code_row_bytes(codec, head_dim);
}
[[nodiscard]] __host__ __device__ constexpr int gqa_simt_kv_scale_plane_bytes(GqaSimtKvCodec codec, int head_dim,
                                                          int bc) {
    return bc * gqa_simt_kv_scale_row_bytes(codec, head_dim);
}
[[nodiscard]] __host__ __device__ constexpr int gqa_simt_kv_arena_bytes(GqaSimtKvCodec codec, int head_dim, int bc) {
    return 2 * gqa_simt_kv_code_plane_bytes(codec, head_dim, bc) +
           2 * gqa_simt_kv_scale_plane_bytes(codec, head_dim, bc);
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput, GqaSimtKvCodec Codec = GqaSimtKvCodec::Bf16>
__launch_bounds__(kGqaSimtFfmaThreadsPerWarp * WarpsPerCta,
                  gqa_simt_ffma_min_blocks_per_sm(Codec)) __global__ void
    gqa_attention_small_t_simt_ffma_partial_bf16_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, __nv_bfloat16* cache_k,
        __nv_bfloat16* cache_v, const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::uint64_t* column_masks, const std::int32_t* table_rows, std::int32_t table_stride,
        std::int32_t tokens, std::int32_t full_width, std::int32_t column_begin,
        std::int32_t logical_capacity, std::int32_t split_units, float scale, float* partial_acc,
        float* partial_m, float* partial_l,
        // The packed codec's SECOND plane pair. Null for bf16, whose arena has no scale region;
        // a packed instantiation must pass both, and the staging reads them through the codec's
        // own index helper. It is one extra by-value struct of two pointers, so the landed bf16
        // instantiation's register and constant-bank footprint is unaffected.
        GqaSimtKvScalePlanes scale_planes = {}) {
    constexpr int D       = Geometry::HeadDim;
    constexpr int Bc      = kGqaSimtFfmaBc;
    constexpr int Wc      = WarpsPerCta;
    constexpr int Threads = Wc * kGqaSimtFfmaThreadsPerWarp;
    constexpr int R       = kGqaSimtFfmaRowsPerWarp;
    constexpr int VecD    = D / kGqaSimtFfmaWarpReduction;
    constexpr int Br      = Wc * R; // rows one CTA carries
    constexpr int PageIds = paged_kv_page_ids(kCausalAttentionMaximumVisibleKeysYarn);

    static_assert(D == 128 || D == 256, "the SIMT FFMA attention family covers head_dim 128/256");
    static_assert(VecD * kGqaSimtFfmaWarpReduction == D);
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(Wc >= 1 && Wc <= 4);
    static_assert(R >= 1 && R <= 8);
    static_assert(gqa_simt_kv_codec_supported(Codec),
                  "this codec is REFUSED BY NAME by gqa_simt_kv_codec_supported "
                  "(see gqa_attention_simt_ffma_refusal for why)");
    static_assert(gqa_simt_kv_arena_bytes(Codec, D, Bc) +
                          PageIds * static_cast<int>(sizeof(std::int32_t)) <=
                      48 * 1024,
                  "the SIMT FFMA decode tile must fit a 48 KiB static shared-memory ceiling");

    // ONE arena, typed views per codec. The region order is: K code, V code, K scale, V scale.
    // For Bf16 the two "code" regions ARE k_s/v_s and both scale regions are zero bytes, so the
    // offsets below collapse to exactly the two arrays this kernel used to declare.
    constexpr int kCodeBytes  = gqa_simt_kv_code_plane_bytes(Codec, D, Bc);
    constexpr int kScaleBytes = gqa_simt_kv_scale_plane_bytes(Codec, D, Bc);
    constexpr int kCodeRow    = gqa_simt_kv_code_row_bytes(Codec, D);
    constexpr int kScaleRow   = gqa_simt_kv_scale_row_bytes(Codec, D);
    __shared__ __align__(16) std::uint8_t tile_s[gqa_simt_kv_arena_bytes(Codec, D, Bc)];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __nv_bfloat16* const k_s = reinterpret_cast<__nv_bfloat16*>(tile_s);
    __nv_bfloat16* const v_s = reinterpret_cast<__nv_bfloat16*>(tile_s + kCodeBytes);
    const GqaSimtPackedPlane k_plane{tile_s, tile_s + 2 * kCodeBytes, kCodeRow, kScaleRow};
    const GqaSimtPackedPlane v_plane{tile_s + kCodeBytes, tile_s + 2 * kCodeBytes + kScaleBytes,
                                     kCodeRow, kScaleRow};

    const int kv_head     = static_cast<int>(blockIdx.x) % Geometry::KVHeads;
    const int row_group   = static_cast<int>(blockIdx.x) / Geometry::KVHeads;
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;
    const int row_base  = row_group * Br;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * Geometry::HeadDim * Geometry::QHeads *
                       tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    // The neutral partial (m = -inf, l = 0, acc = 0) for every row THIS CTA carries. The
    // tensor-core kernel writes it from the whole CTA at once because it carries every row of the
    // (kv_head, split); this family carries a row group, so the write is bounded by row_base.
    auto write_neutral = [&]() {
        for (int row = row_base + tid / kGqaSimtFfmaThreadsPerWarp; row < row_base + Br;
             row += Threads / kGqaSimtFfmaThreadsPerWarp) {
            if (row >= row_count) { break; }
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (!gqa_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
            if (lane == 0) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
            const int d_lane = lane * VecD;
            for (int i = 0; i < VecD; ++i) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d_lane + i, token, split,
                                                            tokens)] = 0.0f;
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const GqaSmallTSplitRange split_range =
        gqa_small_t_split_range<Geometry, false>(window, split_count, split_units, TokenTile, Bc,
                                                 split, gqa_verify_exact_mode());
    const int active_split_count = split_range.active;
    const int split_start        = split_range.start;
    const int split_limit        = split_range.limit;
    if (split >= active_split_count) { return; }

    const int split_end = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    if (page_count > PageIds) {
        write_neutral();
        return;
    }
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        // The owning split writes each new row, exactly as the tensor-core partial does: the
        // current attention reads those rows straight out of `input` below, so no split depends
        // on another split's cache write. Every row group of this (kv_head, split) runs this
        // loop; the writes are idempotent (same value, same address), so the repetition is a
        // wasted store and not a hazard.
        for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
            const int token = chunk / (D / 8);
            const int d     = (chunk - token * (D / 8)) * 8;
            const int p_tok = pos[token];
            if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                p_tok < logical_capacity) {
                const std::int64_t new_off = gqa_kv_new_index<Geometry>(kv_head, d, token);
                int physical_page = lane == 0 ? paged_kv_physical_page(block_table, p_tok) : 0;
                physical_page     = __shfl_sync(0xffffffffu, physical_page, 0);
                const std::int64_t cache_off =
                    gqa_cache_index<Geometry>(physical_page, kv_head, d, p_tok & kPagedKVPageMask);
                store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                store_vec(&cache_v[cache_off], load_vec<int4>(&input.v[new_off]));
            }
        }
    }
    __syncthreads();

    // Register-resident online-softmax state: one entry per (row this warp owns, lane-owned dim).
    float acc[R][VecD];
    float m[R];
    float l[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        m[r] = -CUDART_INF_F;
        l[r] = 0.0f;
#pragma unroll
        for (int i = 0; i < VecD; ++i) { acc[r][i] = 0.0f; }
    }
    const int warp_row_base = row_base + warp * R;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if constexpr (Codec == GqaSimtKvCodec::Bf16) {
            gqa_simt_ffma_stage_tile<Geometry, Bc>(
                k_s, v_s, tid, Threads, k0, split_start, split_end,
                [&](int key, int d, __nv_bfloat16* k_dst, __nv_bfloat16* v_dst) {
                    const int new_token = key - first_pos;
                    bool staged         = false;
                    if constexpr (CacheInput::writes_cache) {
                        if (new_token >= 0 && new_token < valid_tokens) {
                            const std::int64_t off =
                                gqa_kv_new_index<Geometry>(kv_head, d, new_token);
                            store_vec(k_dst, load_vec<int4>(&input.k[off]));
                            store_vec(v_dst, load_vec<int4>(&input.v[off]));
                            staged = true;
                        }
                    }
                    if (!staged) {
                        const int phys = physical_pages_s[(key >> kPagedKVPageShift) - first_page];
                        const std::int64_t off =
                            gqa_cache_index<Geometry>(phys, kv_head, d, key & kPagedKVPageMask);
                        store_vec(k_dst, load_vec<int4>(&cache_k[off]));
                        store_vec(v_dst, load_vec<int4>(&cache_v[off]));
                    }
                });
        } else {
            // THE CODEC ARM. Three facts about this staging are load-bearing:
            // (1) `input` (the append source) is ALWAYS bf16 -- the engine quantizes on append --
            //     so a packed tile cannot take its rows from it; every key is read from the cache
            //     plane. That is why this arm refuses `writes_cache` outright (see the APPEND
            //     ORDERING block above): with an append in flight the current row's packed bytes
            //     may not be committed yet, and reading a stale K row is a silent wrong.
            // (2) The cache planes are reached through the codec's OWN index helpers, whose
            //     strides are not `gqa_cache_index`'s. `gqa_simt_ffma_load_code` /
            //     `_load_scale` are that path, and they are the only copy of it.
            // (3) The scale plane is a SECOND tensor pair, which is why the kernel takes
            //     `scale_planes` as one more parameter.
            static_assert(!CacheInput::writes_cache,
                          "REFUSED BY NAME: a packed KV codec cannot be staged from a bf16 "
                          "append source, and this arm does not yet establish that the append "
                          "quantization has committed the row. See the APPEND ORDERING block "
                          "above and gqa_attention_simt_ffma_refusal().");
            gqa_simt_ffma_stage_tile_packed<Geometry, Bc, Codec>(
                k_plane, v_plane, tid, Threads, k0, split_start, split_end,
                [&](int key, int d, std::uint8_t* k_dst, std::uint8_t* v_dst) {
                    const int phys     = physical_pages_s[(key >> kPagedKVPageShift) - first_page];
                    const int page_off = key & kPagedKVPageMask;
                    // The kernel's plane parameters are bf16-typed because the bf16 arm is the
                    // landed one; for a packed codec the SAME two parameters carry the code
                    // planes and are reinterpreted here, at the one place that knows Codec is
                    // not bf16. A separate byte-typed parameter pair would have to be passed as
                    // null by every existing bf16 instantiation, i.e. it would change the landed
                    // signature to pay for the new one.
                    gqa_simt_ffma_load_code<Geometry, Codec>(
                        reinterpret_cast<const std::uint8_t*>(cache_k),
                        reinterpret_cast<const std::uint8_t*>(cache_v), phys, kv_head, d, page_off,
                        k_dst, v_dst);
                },
                [&](int key, int grp, std::uint8_t* k_dst, std::uint8_t* v_dst) {
                    const int phys     = physical_pages_s[(key >> kPagedKVPageShift) - first_page];
                    const int page_off = key & kPagedKVPageMask;
                    gqa_simt_ffma_load_scale<Geometry, Codec>(scale_planes.k, scale_planes.v, phys,
                                                             kv_head, grp, page_off, k_dst, v_dst);
                });
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = warp_row_base + r;
            if (row >= row_count) { continue; }
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (!gqa_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
            const int qabs = pos[token];
            const std::uint64_t mask =
                column_masks == nullptr
                    ? ~std::uint64_t{0}
                    : column_masks[static_cast<std::int64_t>(batch) * full_width + column_begin +
                                   token];
            const int d_lane = lane * VecD;
            float qr[VecD];
#pragma unroll
            for (int i = 0; i < VecD; ++i) {
                qr[i] = __bfloat162float(q[gqa_q_index<Geometry>(q_head, d_lane + i, token)]);
            }
            if constexpr (Codec == GqaSimtKvCodec::Bf16) {
                gqa_simt_ffma_row_pass<Geometry, Bc>(qr, k_s, v_s, lane, k0, qabs, mask,
                                                     column_masks != nullptr, column_begin,
                                                     first_pos, split_start, split_end, scale, m[r],
                                                     l[r], acc[r]);
            } else {
                gqa_simt_ffma_row_pass_packed<Geometry, Bc, Codec>(
                    qr, k_plane, v_plane, lane, k0, qabs, mask, column_masks != nullptr,
                    column_begin, first_pos, split_start, split_end, scale, m[r], l[r], acc[r]);
            }
        }
        __syncthreads();
    }

    // Split-local partials for the rows this warp owns. partial_m / partial_l are per
    // (q_head, token, split), so one lane writes them; partial_acc is per (q_head, d, token,
    // split) and every lane owns its own VecD dims of the row.
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = warp_row_base + r;
        if (row >= row_count) { continue; }
        int q_head = 0;
        int token  = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (!gqa_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
        if (lane == 0) {
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m[r];
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l[r];
        }
        const int d_lane = lane * VecD;
#pragma unroll
        for (int i = 0; i < VecD; ++i) {
            partial_acc[gqa_partial_acc_index<Geometry>(q_head, d_lane + i, token, split, tokens)] =
                acc[r][i];
        }
    }
}

// ---------------------------------------------------------------------------
// PREFILL: FlashAttention-2 forward, FFMA + online softmax, one CTA per (query row block of
// kGqaSimtFfmaRowsPerCta = 16 rows, query head). Same signature and the same bottom-right causal
// alignment as gqa_attention_prefill_bf16_kernel: query row i attends keys [0, base_pos + i].
//
// Difference from the tensor-core body, stated because it is a numerical difference and not a
// hidden one: the tensor-core kernel rounds P to bf16 before the PV MMA (pack_bf16x2 into the p_s
// staging buffer, gqa_attention_prefill_bf16.cuh:412-417,:495-498). This body keeps P in fp32, so
// it is *more* accurate on that axis, and a comparison between the two arms on identical inputs
// is therefore expected to differ at roughly bf16 weight precision (2^-9) rather than at fp32
// rounding -- REPORT.md has the measured numbers from sm_120.
// ---------------------------------------------------------------------------
// ⚠ THE CODEC SEAM (dl/nvfp4emu2). The template list and the trailing by-value parameter below
// are the SAME two seam pieces the DECODE body of this family already carries
// (gqa_attention_small_t_simt_ffma_partial_bf16_kernel, `GqaSimtKvCodec Codec = GqaSimtKvCodec::Bf16`
// and `GqaSimtKvScalePlanes scale_planes = {}`), and they are declared the same way for the same
// reason: the defaults make the landed bf16 launch's source line IDENTICAL, and `if constexpr`
// below makes the landed bf16 INSTANTIATION emit the same cells. ⚠ ONE NAMED DIFFERENCE FROM THE
// DECODE SIDE, DELIBERATE: the decode kernel's __launch_bounds__ carries a second argument,
// gqa_simt_ffma_min_blocks_per_sm(Codec). This one does NOT gain it. Adding it would change the
// register budget the LANDED bf16 prefill instantiation is compiled against, i.e. it would change
// the existing arm to pay for the new one -- and it is not needed, because the packed arena this
// arm stages into is SMALLER than the bf16 tile it replaces (NVFP4 at D=256, Bc=32:
// 2*32*128 + 2*32*16 = 9216 B, against the bf16 tile's 2*32*256*2 = 32768 B), so the packed arm
// cannot be the one that limits occupancy.
template <typename Geometry, typename Metadata, GqaSimtKvCodec Codec = GqaSimtKvCodec::Bf16>
__launch_bounds__(kGqaSimtFfmaPrefillThreads) __global__ void
    gqa_attention_simt_ffma_prefill_bf16_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ cache_k,
        const __nv_bfloat16* __restrict__ cache_v, Metadata metadata,
        const std::int32_t* __restrict__ positions, float scale,
        __nv_bfloat16* __restrict__ out, std::int32_t width,
        // The packed codec's SECOND plane pair. Null (the default) for bf16, whose arena has no
        // scale region at all; a packed instantiation must pass both, and the staging reads them
        // through the codec's own index helper and not through a re-derived one.
        //
        // ⚠ WHY THE TWO PLANE PARAMETERS ABOVE STAY `const __nv_bfloat16*` EVEN FOR A PACKED CODEC.
        // Same reason the decode kernel gives: the bf16 arm is the landed one, so a separate
        // byte-typed parameter pair would have to be passed as null by every existing bf16 launch
        // -- i.e. it would change the landed signature to pay for the new one. A packed
        // instantiation reinterprets the SAME two parameters, at the one place that knows Codec is
        // not bf16 (the staging's code_source below).
        GqaSimtKvScalePlanes scale_planes = {}) {
    constexpr int D     = Geometry::HeadDim;
    constexpr int Br    = kGqaSimtFfmaRowsPerCta;
    constexpr int Bc    = kGqaSimtFfmaBc;

    constexpr int R     = kGqaSimtFfmaRowsPerWarp;
    constexpr int VecD  = D / kGqaSimtFfmaWarpReduction;

    static_assert(D == 128 || D == 256, "the SIMT FFMA attention family covers head_dim 128/256");
    static_assert(VecD * kGqaSimtFfmaWarpReduction == D);
    static_assert(Bc * VecD == D || Bc > 0);
    static_assert(gqa_simt_kv_codec_supported(Codec),
                  "this codec is REFUSED BY NAME by gqa_simt_kv_codec_supported "
                  "(see gqa_attention_simt_ffma_codec_refusal for why); E8Kv/rk4v4 is the named "
                  "refusal there, because its reader emits a ROTATED K");
    static_assert(gqa_simt_kv_arena_bytes(Codec, D, Bc) <= 48 * 1024,
                  "the SIMT FFMA prefill tile must fit a 48 KiB static shared-memory ceiling");

    // ONE arena, typed views per codec -- the decode body's layout, copied rather than re-derived,
    // so the sizing, the staging and the row pass cannot disagree about the byte layout. The region
    // order is: K code, V code, K scale, V scale. For Bf16 the two "code" regions ARE k_s/v_s and
    // BOTH scale regions are zero bytes (gqa_simt_kv_scale_row_bytes returns 0), so the byte count
    // collapses to 2 * Bc * D * 2 = the two arrays this kernel declared before the seam existed.
    // ⚠ That equality is the whole reason the landed bf16 instantiation is unchanged, and it is
    // asserted rather than trusted:
    constexpr int kCodeBytes  = gqa_simt_kv_code_plane_bytes(Codec, D, Bc);
    constexpr int kScaleBytes = gqa_simt_kv_scale_plane_bytes(Codec, D, Bc);
    constexpr int kCodeRow    = gqa_simt_kv_code_row_bytes(Codec, D);
    constexpr int kScaleRow   = gqa_simt_kv_scale_row_bytes(Codec, D);
    static_assert(gqa_simt_kv_arena_bytes(Codec, D, Bc) == 2 * kCodeBytes + 2 * kScaleBytes);
    static_assert(Codec != GqaSimtKvCodec::Bf16 ||
                      2 * kCodeBytes == 2 * Bc * D * static_cast<int>(sizeof(__nv_bfloat16)),
                  "bf16: the byte arena must be exactly the two bf16 tiles it replaced");
    static_assert(Codec != GqaSimtKvCodec::Bf16 || kScaleBytes == 0,
                  "bf16: the scale plane must be zero bytes, as gqa_simt_kv_scale_row_bytes says");
    __shared__ __align__(16) std::uint8_t tile_s[gqa_simt_kv_arena_bytes(Codec, D, Bc)];
    __nv_bfloat16* const k_s = reinterpret_cast<__nv_bfloat16*>(tile_s);
    __nv_bfloat16* const v_s = reinterpret_cast<__nv_bfloat16*>(tile_s + kCodeBytes);
    const GqaSimtPackedPlane k_plane{tile_s, tile_s + 2 * kCodeBytes, kCodeRow, kScaleRow};
    const GqaSimtPackedPlane v_plane{tile_s + kCodeBytes, tile_s + 2 * kCodeBytes + kScaleBytes,
                                     kCodeRow, kScaleRow};

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                               kGqaSimtFfmaPrefillThreads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const std::uint64_t* const column_masks = metadata.column_masks;
    const bool round_masked                 = column_masks != nullptr;
    const int column_begin                  = 0;
    const int first_pos                     = base_pos;

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int n_block_max   = (max_query_abs / Bc) + 1;

    float acc[R][VecD];
    float m[R];
    float l[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        m[r] = -CUDART_INF_F;
        l[r] = 0.0f;
#pragma unroll
        for (int i = 0; i < VecD; ++i) { acc[r][i] = 0.0f; }
    }
    const int warp_row_base = warp * R;

    for (int kb = 0; kb < n_block_max; ++kb) {
        const int k0 = kb * Bc;
        // THE CODEC SEAM, PREFILL SIDE. The bf16 arm below is byte-for-byte the code that was
        // here: same staging call, same split window, same lambda body. The packed arm is the
        // decode body's codec arm (:1546-1585) with its one decode-specific piece removed -- the
        // `physical_pages_s` page-id table, which this kernel does not have and does not need
        // because it reads `block_table` per chunk exactly as the bf16 arm above it does.
        if constexpr (Codec == GqaSimtKvCodec::Bf16) {
            gqa_simt_ffma_stage_tile<Geometry, Bc>(
                k_s, v_s, tid, kGqaSimtFfmaPrefillThreads, k0, 0, max_query_abs + 1,
                [&](int key, int d, __nv_bfloat16* k_dst, __nv_bfloat16* v_dst) {
                    const int phys = block_table[key >> kPagedKVPageShift];
                    const std::int64_t off =
                        paged_kv_element_offset<Geometry::HeadDim, Geometry::KVHeads>(
                            phys, kv_head, key & kPagedKVPageMask, d);
                    store_vec(k_dst, load_vec<int4>(&cache_k[off]));
                    store_vec(v_dst, load_vec<int4>(&cache_v[off]));
                });
        } else {
            // Same window (0, max_query_abs + 1) as the bf16 arm above, and it MUST be the same:
            // gqa_simt_ffma_stage_tile_packed zero-fills both planes for a key outside it, so the
            // two arms cover exactly the same key set with exactly the same padding.
            gqa_simt_ffma_stage_tile_packed<Geometry, Bc, Codec>(
                k_plane, v_plane, tid, kGqaSimtFfmaPrefillThreads, k0, 0, max_query_abs + 1,
                [&](int key, int d, std::uint8_t* k_dst, std::uint8_t* v_dst) {
                    const int phys     = block_table[key >> kPagedKVPageShift];
                    const int page_off = key & kPagedKVPageMask;
                    // The kernel's plane parameters are bf16-typed because the bf16 arm is the
                    // landed one; for a packed codec the SAME two parameters are the CODE planes
                    // and are reinterpreted here, at the one place that knows Codec is not bf16.
                    gqa_simt_ffma_load_code<Geometry, Codec>(
                        reinterpret_cast<const std::uint8_t*>(cache_k),
                        reinterpret_cast<const std::uint8_t*>(cache_v), phys, kv_head, d, page_off,
                        k_dst, v_dst);
                },
                [&](int key, int grp, std::uint8_t* k_dst, std::uint8_t* v_dst) {
                    const int phys     = block_table[key >> kPagedKVPageShift];
                    const int page_off = key & kPagedKVPageMask;
                    gqa_simt_ffma_load_scale<Geometry, Codec>(scale_planes.k, scale_planes.v, phys,
                                                             kv_head, grp, page_off, k_dst, v_dst);
                });
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = warp_row_base + r;
            if (row >= tile_rows) { continue; }
            const int qrow = q0 + row;
            const int qabs = base_pos + qrow;
            const std::uint64_t mask =
                round_masked ? column_masks[(qrow < tokens) ? qrow : 0] : ~std::uint64_t{0};
            const int d_lane = lane * VecD;
            float qr[VecD];
#pragma unroll
            for (int i = 0; i < VecD; ++i) {
                qr[i] = __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d_lane + i, qrow)]);
            }
            // The prefill route's visible set is the causal prefix [0, max_query_abs] (the
            // staging zeroes everything above it), so the split bounds below are the whole tile.
            // The SAME split window (0, 1 << 30) on both arms, and the same visibility
            // predicate inside them: gqa_simt_ffma_row_pass_packed's `key >= split_start &&
            // key < split_end && key <= qabs` is gqa_simt_ffma_row_pass's, spelled the same way
            // (:744 against :1233). That identity is what makes this substitution a DECODE
            // substitution: the only thing that changes between the two arms is where the K and V
            // values come from, not which keys are visible and not the fp32 association of the
            // online softmax (the packed pass keeps P in fp32 too; only Bc would move the
            // association, and Bc is a constant of the family).
            if constexpr (Codec == GqaSimtKvCodec::Bf16) {
                gqa_simt_ffma_row_pass<Geometry, Bc>(qr, k_s, v_s, lane, k0, qabs, mask,
                                                     round_masked, column_begin, first_pos, 0,
                                                     1 << 30, scale, m[r], l[r], acc[r]);
            } else {
                gqa_simt_ffma_row_pass_packed<Geometry, Bc, Codec>(
                    qr, k_plane, v_plane, lane, k0, qabs, mask, round_masked, column_begin,
                    first_pos, 0, 1 << 30, scale, m[r], l[r], acc[r]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = warp_row_base + r;
        if (row >= tile_rows) { continue; }
        const int qrow    = q0 + row;
        const float l_r   = l[r];
        const float inv_l = (l_r > 0.0f) ? __frcp_rn(l_r) : 0.0f;
        const int d_lane  = lane * VecD;
#pragma unroll
        for (int i = 0; i < VecD; ++i) {
            out[gqa_prefill_q_index<Geometry>(q_head, d_lane + i, qrow)] =
                gqa_simt_f32_to_bf16(acc[r][i] * inv_l);
        }
    }
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                           kGqaSimtFfmaPrefillThreads);
}

} // namespace ninfer::ops
