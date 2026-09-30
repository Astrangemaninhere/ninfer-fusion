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
#include <mutex>
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
// ===========================================================================
// THE PRE-75 BLOCK: SIX ROWS BELOW sm_70, ADDED 2026-09-24 (F702, dl/archrow)
// ===========================================================================
// WHY THEY EXIST. The ladder's own refusal for an unlisted number says "add the row with
// MEASURED evidence first", and TWO independent lines had already named the two halves of this
// gap: dl/oldcard said the wall for sm_61 is the ABSENT LADDER ROW (not the attention path), and
// dl/archstack said a row below 70 needs MEASURED evidence. This block is where those two meet.
// The consequence it removes is measured: src/core/arch_sim.h resolves a requested rung with
// arch_rung(), an EXACT-MATCH walk, so before this block the simulator could not express "pre-75"
// AT ALL -- every NINFER_SIM_ARCH below 70 was refused with "has no row in kArchLadder, so
// simulating it would mean inventing a capability set", on every physical device, and a
// rank-guarded V100 (whose reported sm is 70) could not simulate any of the rungs below itself.
//
// WHAT EACH ROW CARRIES, AND THE ONE THING IT MUST NOT CARRY.
//   * `caps` is Cap::None, and that is a MEASURED instruction-set fact rather than a hedge:
//     ptxas rejects `mma.sync` below .target sm_70 ("Feature 'mma' requires .target sm_70 or
//     higher"), recorded per rung for sm_50/52 and sm_61/62 in tools/archkit/_GPU_MATRIX.md
//     section 3.3 (its own citation ops/common/mma.cuh:191 still resolves: the mma_f16_m8n8k4
//     asm). Cap has six bits and every one of them is a tensor-core instruction family, so a
//     rung with no tensor core has NO bit -- the vocabulary has none for SIMT/FFMA. Writing
//     Cap::Fp16Mma here to "be generous" would be the sm_100-nvfp4 false positive in a new
//     place, which is the bug class this table has already paid for twice.
//   * `route` states the FORMAT FLOORS this rung misses WITH THEIR KERNEL file:line, the route
//     table's own cells for the family, and the CALIBER of the evidence for that rung. That is
//     the deliverable: a refusal that names WHICH floor and WHERE, not merely that it refuses.
//   * It does NOT carry a kQpnMmaRungs row, and must not: qpn_arm_reachable() requires
//     Cap::Fp16Mma, so the fp16 fallback arm is unreachable on all six and a lowering row would
//     be a measurement nothing consults (kQpnMmaRungs' own comment says the same for sm_120/121).
//
// A BUILD FACT IS NOT SUPPORT, and these rows do not claim it. CUDA 12.8 (/mnt/g/cuda12/tk,
// V12.8.61) lists compute_50 52 53 60 61 62 -- measured 2026-09-24, `nvcc --list-gpu-arch` on
// that chain returns 17 arches whose first six are exactly these -- and CUDA 13.3 lists none
// below compute_75. tools/archkit/build_arch.sh:83's LEGACY_ARCHS=(50 52 53 60 61 62 70) makes
// the whole pre-75 block ONE tier on ONE toolchain file. _GPU_MATRIX.md's route row for this
// family is unaffected and still reads: tensor core "none", route "pending", status "build
// configuration only (unprobed)". Its cells are Chinese; the English above is a translation of
// that row, not a quotation of English text.
//
// CITATION DRIFT, DISCLOSED BECAUSE THE REFUSAL PRINTS IT -- AND NOW CLOSED (dl/oldkernel).
// Several file:line positions recorded in _GPU_MATRIX.md section 3.3 and in
// kFormatRequirements.kernel_evidence did not resolve. THE ROWS BELOW always cited the LIVE
// positions; the RECORDED ones were fixed on 2026-09-24, in ONE deliberate change, because
// closing them moves this file's own refusal text and that move has to be stated rather than
// discovered. Every position, OLD -> NEW, measured with a 1-based read of the live file:
//   bf16_gemm_mma.cuh:273  -> :296     (:273 is `const int row = wn * WN + ni * 8 + ...`)
//   ops/common/mma.cuh:33  -> :109     (:33 is a comment; :109 is mma_bf16, asm :113)
//   ops/common/mma.cuh:90  -> :269     (:90 is inside ldmatrix_x2; :269 is mma_nvfp4_e4m3)
//   ops/common/mma.cuh:63  -> :191     (:63 is ldmatrix_x2; :191 is the m8n8k4 asm)
//   qpn_kernels.cuh:695    -> :729     (:695 is `const uint2 q2 =`; :729 is the m8n8k4 call)
//   qpn_kernels.cuh:893    -> :929     (the MMA_8N8K4 macro's call site)
//   qpn_kernels.cuh:1227   -> :1257    (:1227 is a blank comment; :1257 is skinny_fp8_qpn8)
//   qpn_kernels.cuh:1327   -> :1357    (skinny_fp8_qpn8_mt2)
//   memory.cuh:65          -> :126     (:65 is a brace; :126 is cp_async, guard :30,
//                                       asm :131/:135, below-floor trap arm :141)
//   qpn_kernels.cuh:404    -> :430     (the wmma fragment, moved by this file's own gate block)
// WHAT THE MOVE COSTS, STATED AND NOT HIDDEN: every one of these strings sits inside a rung's
// route text or a kFormatRequirements row, so EVERY RUNG'S REFUSAL TEXT CHANGED. That text is
// what tests/test_kernel_route.cpp:224-229 and tests/test_arch_caps.cpp read, so those tests are
// this move's re-verification surface. THE FLOORS' MEANINGS DID NOT CHANGE: each new position is
// the same instruction in the same file, re-read.
// TWO THINGS IN THIS BLOCK ARE DELIBERATELY NOT FIXED, and the reason is given rather than left
// as an omission. (a) `grep -c __CUDA_ARCH__` on qpn_kernels.cuh returns 2 while section 3.3
// reports 0: that is a claim in a TOOLING DOCUMENT about a HEAD that no longer exists, and the
// file's own gate block (qpn_kernels.cuh:16-34) is its correction -- editing section 3.3's
// prose would be editing a record of a measurement, which is not what this change is for.
// (b) src/core/device_capabilities.h's evidence column and tests/test_device_capabilities.cpp
// carry the SAME class of drift one table over (mma.cuh:37/46/54/63/90, memory.cuh:40,44,
// mma.cuh:9,16 -- all pre-gate positions), and it is NOT fixed here because moving those strings
// moves a TEST ASSERTION with them: see the report's "adjacent drift, measured and left" section,
// which carries the full OLD -> NEW mapping so a later change is mechanical.
//
// WHAT IS *NOT* MISSING HERE, so these rows are not read as "nothing exists". The FFMA +
// online-softmax attention family (src/ops/kernel/gqa_attention_simt_ffma.cuh) has a floor note
// recording that every instruction it emits assembles from sm_50 up, and the QPN family's
// FFMA-only SIMT band (skinny_nvfp4_qpn_simt, qpn_kernels.cuh:1015) now compiles below sm_70
// because that file gained per-target gates (qpn_kernels.cuh:32 and :52). NEITHER IS A ROUTE ON
// THESE RUNGS: the attention arm is selected in the ops launcher, not from a ladder capability,
// and the QPN band is reached only through fp16_fallback_executable(), which is false here. So
// the honest statement is "the pieces exist and no route arm reaches them" -- not "no kernel".
inline constexpr ArchRung kArchLadder[] = {
    // ===========================================================================
    // THE SEVENTH PRE-75 ROW: sm_35 (Kepler GK110). ADDED 2026-09-25 (F850, dl/rung35)
    // ===========================================================================
    // WHY THIS ROW IS HERE, AND IT IS NOT A WIDENING. The block above (F702) records that an
    // exact-match table refuses a buildable target that has no row, and that the ARCH SIMULATOR
    // could not express a rung the ladder does not contain. Both halves were measured again for
    // sm_35: dl/sg35 (F-830) measured `10 of 184` CUDA TUs emitting a real sm_35 cubin with 174
    // reds and a link that cannot be made, and dl/simemit (F-842) measured the simulator refusing
    // rung 35 with "sm_35 has no row in kArchLadder, so simulating it would mean inventing a
    // capability set. Refusing; add the row with MEASURED evidence first." This block is that
    // sentence obeyed, not overruled: the evidence exists, so the row is added.
    //
    // THE COUNT IS NOW SEVEN, NOT SIX, AND THIS COMMENT SAYS SO RATHER THAN EDITING F702's OWN
    // HEADING. The F702 block is a record of a measurement taken on 2026-09-24; its "SIX ROWS"
    // heading describes what that change added and is left standing. This one is its own dated
    // block for the same reason the file's citation-drift note gives: editing a record of a
    // measurement is not what a later change is for.
    //
    // WHAT IT DOES *NOT* BUY. Nothing is admitted by this row. The simulator's emission floor is
    // rung 70 (dl/simemit), no sm_35 card exists on this box, and the row carries NO capability
    // bit: Cap's six bits are all tensor-core instruction families and Maxwell/Kepler has no
    // tensor core, so a bit here would be the sm_100-nvfp4 false positive in a new place. The row
    // buys a NAMED refusal of the sm_52/sm_61/sm_62 shape instead of a structural absence.
    //
    // ITS RE-VERIFICATION SURFACE IS A TEST THAT ASSERTS THE ABSENCE IT REMOVES:
    // tests/test_arch_caps.cpp:132 reads `arch_rung(35) == nullptr` and :332-336 expects
    // `evaluate_artifact_formats(35, BF16)` to be UnknownArch. Both MOVE with this row, and the
    // file's own history is the precedent (that example was sm_60 until the F702 block landed).
    // This change does NOT silently edit them; the report names them as the follow-up.
    //
    {35, "Kepler GK110", "Tesla K20 / K40 / GTX 780 / Titan / Quadro K6000",
     Cap::None,
     "NO TENSOR CORE, MEASURED AT THIS RUNG. ptxas rejects `mma.sync` below .target sm_70 ('Feature "
     "'mma' requires .target sm_70 or higher'); re-measured 2026-09-25 FOR THIS RUNG by dl/rung35 "
     "with /mnt/g/cuda118/bin/ptxas -arch=sm_35 on the asm at ops/common/mma.cuh:191 "
     "(mma_f16_m8n8k4), one instruction per module at PTX ISA .version 6.4: rc 255, cubin ABSENT, "
     "while the SAME module at -arch=sm_70 is rc 0, cubin PRESENT -- a POSITIVE CONTROL, so the "
     "instrument can tell the two states apart. THE FLOORS THIS RUNG MISSES, EACH WITH ITS KERNEL: "
     "NVFP4 -> kind::mxf4nvf4, nvfp4_w4a4_mma.cuh:308 -> ops/common/mma.cuh:269 (asm :276); BF16 -> "
     "bf16_gemm_mma.cuh:296 -> mma.cuh:109 (asm :113); Q4/Q5/Q6/W8 -> the same bf16 mma, "
     "q4_rowsplit_gemm_mma.cuh:341, q5:385, q6:394, w8:267; FP8_E4M3FN_ROW_BF16S -> "
     "fp8_a16_gemm_mma.cuh:206. NO LOWER FLOOR EITHER: the fp16 fallback needs Cap::Fp16Mma and a "
     "measured HMMA.884 lowering, and neither holds here. WHY THIS ROW EXISTS AT ALL, because the "
     "ladder's own refusal asks for it by name. dl/sg35 (F-830) measured, on the LIVE tree under "
     "CUDA 11.8.89 + recipe R3 + the k40 shim: 10 of 184 CUDA TUs emit a REAL sm_35 cubin (10 of 10 "
     "verified with cuobjdump -lelf) and 174 are red, so the LINK cannot be made; 165 of the 174 "
     "carry family 1's verbatim first error ('\"= default\" cannot be specified on a friend "
     "declaration', src/core/shard_plan.h(76)) and 170 of 174 are that one cause once "
     "src/artifact/reader.h(180)'s sibling form is counted -- a defaulted comparison operator, "
     "which is C++20, against a toolchain whose ceiling is C++17. Re-measured here 2026-09-25: "
     "/mnt/g/cuda118/bin/nvcc --std=c++20 answers `nvcc fatal : Value 'c++20' is not defined for "
     "option 'std'`, rc 1, while the construct is LIVE in the tree -- 67 `friend ... = default;` "
     "declarations in src+include (66 operator==, 1 operator<=>), counted by dl/rung35 with a "
     "declaration-aware scan; a single-line grep sees only 24 of them, because 43 wrap. THE WALL IS "
     "A LANGUAGE-VERSION WALL, NOT AN ARCHITECTURE WALL, and this row records that in place of an "
     "ISA claim. WHAT THIS ROW DOES NOT BUY, SAID PLAINLY: it does NOT buy emission. dl/simemit "
     "(F-842) measured the simulator's emission floor at rung 70, and dl/sg35 measured that no "
     "sm_35 card exists on this box (RTX 5090 D, compute_cap 12.0). What it buys is an HONEST NAMED "
     "REFUSAL of the same SHAPE as sm_52/sm_61/sm_62 -- naming the weight format and the kernel "
     "floor -- where before it there was a structural absence ('sm_35 has no row in kArchLadder, so "
     "simulating it would mean inventing a capability set'). CALIBER: dl/sg35's census is 184 of "
     "184 measured at compute_35 with a cold replay (741.3 s) and reproduced in a second, "
     "independent directory with 0 verdict disagreements; there is NO tools/archkit/_GPU_MATRIX.md "
     "section 3.2/3.3 row for sm_35 (that table's pre-75 coverage stops at 50), so this row's FIELD "
     "evidence is dl/sg35 and its INSTRUCTION evidence is dl/rung35's own ptxas probe, not that "
     "table. WHAT WOULD HAVE TO (FOR A FORMAT WHOSE kFormatRequirements.simt_kernel_evidence COLUMN "
     "IS EMPTY -- the four groupwise-int formats name one and are SELECTED by select_route's "
     "ConservativeSimt arm instead): a route arm that reaches a SIMT linear kernel here, a staged "
     "dist, and a real Kepler card. Do NOT set a capability bit here: Cap has six bits and every "
     "one of them is a tensor-core instruction family, and this rung has no tensor core -- a bit "
     "here would be the sm_100-nvfp4 false positive in a new place.",
    },
    {50, "Maxwell GM10x", "GTX 750 / 750 Ti / 950M",
     Cap::None,
     "NO TENSOR CORE, MEASURED. ptxas rejects `mma.sync` below .target sm_70 ('Feature 'mma' "
     "requires .target sm_70 or higher'); the per-rung probe of this tree's own emitters that "
     "recorded it is tools/archkit/_GPU_MATRIX.md section 3.3, and its citation "
     "ops/common/mma.cuh:191 still resolves (mma_f16_m8n8k4). THE FLOORS THIS RUNG MISSES, EACH "
     "WITH ITS KERNEL (re-measured 2026-09-24): NVFP4 -> kind::mxf4nvf4, "
     "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh:308 -> ops/common/mma.cuh:269 (asm :276); BF16 -> "
     "ops/linear/bf16/bf16_gemm_mma.cuh:296 -> ops/common/mma.cuh:109 (asm :113); "
     "Q4/Q5/Q6/W8 -> the same bf16 mma, q4_rowsplit_gemm_mma.cuh:341, q5:385, q6:394, w8:267; "
     "FP8_E4M3FN_ROW_BF16S -> its A16 route, fp8_a16_gemm_mma.cuh:206. NO LOWER FLOOR EITHER: "
     "kFormatRequirements' fallback_required column is honoured only when "
     "fp16_fallback_executable() is true and that needs Cap::Fp16Mma plus a measured HMMA.884 "
     "lowering; neither holds here. _GPU_MATRIX.md's route row for this family: tensor core "
     "'none', route 'pending', status 'build configuration only (unprobed)'. CALIBER: section "
     "3.2 records `build_arch.sh 50` configure rc=0 after the CMake pre-70 floor was made "
     "inapplicable when a toolchain file is given; the per-TU compile sweep was NOT run for this "
     "rung. WHAT WOULD HAVE TO (FOR A FORMAT WHOSE kFormatRequirements.simt_kernel_evidence COLUMN IS EMPTY -- the four groupwise-int formats name one and are SELECTED by select_route's ConservativeSimt arm instead): a linear/GEMM route arm that reaches a SIMT kernel on this "
     "ISA and a staged dist on a real card. Do NOT set a capability bit here to fill the row: "
     "the bits are instruction-set facts."},
    {52, "Maxwell GM20x", "GTX 960 / 970 / 980 / 980 Ti",
     Cap::None,
     "NO TENSOR CORE, MEASURED, AND THIS IS THE OWNER'S GTX 960 RUNG. ptxas rejects `mma.sync` "
     "below .target sm_70 ('Feature 'mma' requires .target sm_70 or higher'), recorded per rung "
     "in tools/archkit/_GPU_MATRIX.md section 3.3 (citation ops/common/mma.cuh:191 still "
     "resolves: mma_f16_m8n8k4). THE FLOORS THIS RUNG MISSES, EACH WITH ITS KERNEL "
     "(re-measured 2026-09-24): NVFP4 -> kind::mxf4nvf4, nvfp4_w4a4_mma.cuh:308 -> "
     "ops/common/mma.cuh:269 (asm :276); BF16 -> bf16_gemm_mma.cuh:296 -> mma.cuh:109 (asm "
     ":113); Q4/Q5/Q6/W8 -> the same bf16 mma, q4_rowsplit_gemm_mma.cuh:341, q5:385, q6:394, "
     "w8:267; FP8_E4M3FN_ROW_BF16S -> fp8_a16_gemm_mma.cuh:206. NO LOWER FLOOR EITHER: the fp16 "
     "fallback needs Cap::Fp16Mma and a measured HMMA.884 lowering, and neither holds here. "
     "CALIBER, and this rung has the most of the six: section 3.2 records configure rc=0 TWICE "
     "(script path and hand-written path) and 384/385 TUs compiled, with 24 failures of which 21 "
     "are arch-independent renames that were in flight and 3 are this block's own -- of those "
     "three, exactly ONE was the pre-Volta GEMM wall (`using namespace nvcuda;` in the QPN TU, "
     "which section 3.3 records and which that file has since gated at qpn_kernels.cuh:52). "
     "WHAT WOULD HAVE TO (FOR A FORMAT WHOSE kFormatRequirements.simt_kernel_evidence COLUMN IS EMPTY -- the four groupwise-int formats name one and are SELECTED by select_route's ConservativeSimt arm instead): a route arm that reaches a SIMT linear kernel here, a staged "
     "dist, and a real Maxwell card to probe. Do NOT set a capability bit here."},
    {53, "Maxwell Tegra GM20B", "Tegra X1 / Jetson Nano",
     Cap::None,
     "NO TENSOR CORE. Same Maxwell ISA as sm_50/52, whose per-rung ptxas result is recorded in "
     "tools/archkit/_GPU_MATRIX.md section 3.3 ('Feature 'mma' requires .target sm_70 or "
     "higher'), so the capability set here is a FAMILY INFERENCE AND IS DISCLOSED AS ONE -- this "
     "is the weakest-caliber row of the six and it says so. THE FLOORS THIS RUNG MISSES, EACH "
     "WITH ITS KERNEL: NVFP4 -> kind::mxf4nvf4, nvfp4_w4a4_mma.cuh:308 -> ops/common/mma.cuh:269 "
     "(asm :276); BF16/Q4/Q5/Q6/W8 -> bf16 mma, bf16_gemm_mma.cuh:296, q4_rowsplit_gemm_mma.cuh:"
     "341, q5:385, q6:394, w8:267; FP8_E4M3FN_ROW_BF16S -> fp8_a16_gemm_mma.cuh:206. NO LOWER "
     "FLOOR EITHER (the fp16 fallback needs Cap::Fp16Mma, absent). CALIBER: _GPU_MATRIX.md "
     "section 3.2 does NOT cover this rung -- sm_53 is the one member of the pre-75 block whose "
     "configure was never run (its row reads 'not tested; not tested'). That is exactly why the "
     "row is a source-shape fact plus a family inference and nothing more. WHAT WOULD HAVE TO (FOR A FORMAT WHOSE kFormatRequirements.simt_kernel_evidence COLUMN IS EMPTY -- the four groupwise-int formats name one and are SELECTED by select_route's ConservativeSimt arm instead): the same three pieces as sm_50/52, plus a build of this rung at all. Do NOT read "
     "this row as a measurement of a Jetson."},
    {60, "Pascal GP100", "Tesla P100 / Quadro GP100",
     Cap::None,
     "NO TENSOR CORE. Pascal's fp16 rate is FP16x2 arithmetic, not an `mma` instruction: ptxas "
     "rejects `mma.sync` below .target sm_70 ('Feature 'mma' requires .target sm_70 or higher'), "
     "measured for the sibling rung sm_61 in tools/archkit/_GPU_MATRIX.md section 3.3 (citation "
     "ops/common/mma.cuh:191 still resolves). GP100 is that same Pascal ISA, so this row is a "
     "FAMILY INFERENCE, disclosed as one. THE FLOORS THIS RUNG MISSES, EACH WITH ITS KERNEL: "
     "NVFP4 -> kind::mxf4nvf4, nvfp4_w4a4_mma.cuh:308 -> ops/common/mma.cuh:269 (asm :276); "
     "BF16/Q4/Q5/Q6/W8 -> bf16 mma, bf16_gemm_mma.cuh:296, q4_rowsplit_gemm_mma.cuh:341, "
     "q5:385, q6:394, w8:267; FP8_E4M3FN_ROW_BF16S -> fp8_a16_gemm_mma.cuh:206. NO LOWER FLOOR "
     "EITHER. CALIBER: _GPU_MATRIX.md section 3.2 records configure rc=0 only (the hand-written "
     "path); the per-TU sweep was not run for this rung, and its set is taken from the measured "
     "sm_61. WHAT WOULD HAVE TO (FOR A FORMAT WHOSE kFormatRequirements.simt_kernel_evidence COLUMN IS EMPTY -- the four groupwise-int formats name one and are SELECTED by select_route's ConservativeSimt arm instead): a route arm that reaches a SIMT linear kernel here, a "
     "staged dist, and a real card. Do NOT set a capability bit here."},
    {61, "Pascal GP10x", "GTX 1050 / 1060 / 1070 / 1080 / Quadro P1000 / P6000",
     Cap::None,
     "NO TENSOR CORE, MEASURED, AND THIS IS THE OWNER'S QUADRO P1000 RUNG. Pascal GP10x has no "
     "`mma` instruction at all: ptxas rejects `mma.sync` below .target sm_70 ('Feature 'mma' "
     "requires .target sm_70 or higher'), measured for THIS rung in tools/archkit/_GPU_MATRIX.md "
     "section 3.3 (its citation ops/common/mma.cuh:191 still resolves: mma_f16_m8n8k4; the same "
     "table records ldmatrix open at sm_75, cp.async at sm_80, and __grid_constant__ refused "
     "here). THE FLOORS THIS RUNG MISSES, EACH WITH ITS KERNEL (re-measured 2026-09-24): NVFP4 "
     "-> kind::mxf4nvf4, ops/linear/nvfp4/nvfp4_w4a4_mma.cuh:308 -> ops/common/mma.cuh:269 "
     "(asm :276); BF16 -> ops/linear/bf16/bf16_gemm_mma.cuh:296 -> ops/common/mma.cuh:109 (asm "
     ":113); Q4G64_F16S/Q5G64_F16S/Q6G64_F16S/W8G32_F16S -> the same bf16 mma, "
     "q4_rowsplit_gemm_mma.cuh:341, q5:385, q6:394, w8:267; FP8_E4M3FN_ROW_BF16S -> its A16 "
     "route, fp8_a16_gemm_mma.cuh:206. NO LOWER FLOOR EITHER: fp16_fallback_executable() needs "
     "Cap::Fp16Mma and a measured HMMA.884 lowering (this rung has neither), so the QPN W4A16 "
     "fallback is not available here even though its FFMA band compiles below sm_70 now "
     "(qpn_kernels.cuh:1015, gated at :32/:52). CALIBER: _GPU_MATRIX.md section 3.2 records "
     "configure rc=0 and 384/385 TUs compiled, line-for-line the same result as sm_52; the "
     "`__dp4a` this rung adds over sm_50/52 is real (qpn_kernels.cuh:611) and the attention "
     "census that motivated the FFMA family is in that table's section 3.3. WHAT WOULD HAVE TO (FOR A FORMAT WHOSE kFormatRequirements.simt_kernel_evidence COLUMN IS EMPTY -- the four groupwise-int formats name one and are SELECTED by select_route's ConservativeSimt arm instead): a linear route arm that reaches a SIMT kernel on this ISA (the FFMA attention family "
     "already covers the attention half and is selected in the ops launcher, not from here), a "
     "staged dist, and the card. Do NOT set a capability bit here: a bit is an instruction-set "
     "claim and the instruction does not exist on this target."},
    {62, "Pascal Tegra GP10B", "Jetson TX2 / Tegra X2",
     Cap::None,
     "NO TENSOR CORE. Same Pascal ISA as the measured sm_61 (tools/archkit/_GPU_MATRIX.md "
     "section 3.3: 'Feature 'mma' requires .target sm_70 or higher'), so the set here is a "
     "FAMILY INFERENCE, disclosed as one. THE FLOORS THIS RUNG MISSES, EACH WITH ITS KERNEL: "
     "NVFP4 -> kind::mxf4nvf4, nvfp4_w4a4_mma.cuh:308 -> ops/common/mma.cuh:269 (asm :276); "
     "BF16/Q4/Q5/Q6/W8 -> bf16 mma, bf16_gemm_mma.cuh:296, q4_rowsplit_gemm_mma.cuh:341, "
     "q5:385, q6:394, w8:267; FP8_E4M3FN_ROW_BF16S -> fp8_a16_gemm_mma.cuh:206. NO LOWER FLOOR "
     "EITHER (the fp16 fallback needs Cap::Fp16Mma, absent here). CALIBER: _GPU_MATRIX.md "
     "section 3.2 records configure rc=0 only; the per-TU sweep was not run for this rung. The "
     "row is here for the same reason sm_87 and sm_88 are here -- a buildable target with no row "
     "is refused by an exact-match table -- and for one more: the arch simulator cannot express a "
     "rung the ladder does not contain. WHAT WOULD HAVE TO (FOR A FORMAT WHOSE kFormatRequirements.simt_kernel_evidence COLUMN IS EMPTY -- the four groupwise-int formats name one and are SELECTED by select_route's ConservativeSimt arm instead): the same three pieces as "
     "sm_50/52, plus a build of this rung. Do NOT read this row as a measurement of a TX2."},
    {70, "Volta", "V100-SXM2/PCIe", Cap::Fp16Mma,
     "fp16 mma only (qpn_kernels.cuh:729 uses mma.sync.m8n8k4). No bf16, no int8 tensor "
     "core, no fp8, no fp4. _GPU_MATRIX.md sm-70 row: QPN2 W4A16 runs published NVFP4/FP8 "
     "weights by expanding 4-bit codes into fp16 mma; no requantization and no fp16 weight "
     "copy needed. THE SAME ROW'S STATUS COLUMN SAYS THAT IS NOT DONE YET (it reads "
     "\"pending port\" -- the cell the citation above omits), so this row must not be read "
     "as \"the V100 has a route\": ops/linear/qpn/ IS COMPILED BY A TARGET IN THIS TREE (src/CMakeLists.txt:366; this clause used to read that no CMake target compiles it, which the build graph no longer says -- re-measured 2026-09-24), and "
     "src/core/kernel_route.h therefore refuses NVFP4 / FP8_E4M3FN_ROW_BF16S on sm_70 with a "
     "NoKernelInTree that names the missing QPN kernel. The row stays because fp16 mma is "
     "what it is -- it is the route that is pending, not the capability."},
    {75, "Turing", "RTX 20xx / T4", Cap::Fp16Mma | Cap::Int8Mma,
     "fp16 + int8 tensor core; NO bf16. The shipped Q4/Q5/Q6/W8 linear kernels all bottom "
     "out in mma_bf16 (sm_80 floor), so the groupwise-int profile is NOT executable here as "
     "the sources stand -- it needs the QPN fp16 path or an mma_s8 port of "
     "q4/q5/q6/w8_*_gemm_mma.cuh. The QPN fp16 path is the one named in the sm-70 row "
     "above: it exists in src/ops/linear/qpn/ and IS COMPILED BY A TARGET IN THIS TREE (src/CMakeLists.txt:366, re-measured 2026-09-24; this clause used to read compiled by no target), so "
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

    // -----------------------------------------------------------------------
    // THE THIRD FLOOR: the tensor-core-free FFMA/SIMT GEMM.
    // -----------------------------------------------------------------------
    // WHY A THIRD COLUMN AND NOT A SEVENTH Cap BIT. `required` and `fallback_required` are both
    // Cap bits, and Cap's six bits are ALL tensor-core instruction families. A rung whose measured
    // set is Cap::None therefore satisfies neither column, and the soundness invariant the gate
    // and two tests enforce -- `floor == Cap::None || covers(rung->caps, floor) ||
    // (QpnW4a16 && fp16_fallback_executable(...))` -- admits NO tensor-core-free floor at all.
    // Until this column existed, a format whose floor an FMA-PIPE kernel satisfies had no way to
    // say so on such a rung, so every GEMM format refused there whether or not this build shipped
    // a kernel that card could run.
    //
    // WHY NOT WRITE Cap::None IN `required` INSTEAD. Because that column means "this format is
    // never fed to a tensor-core GEMM" and is read as such at three surfaces (select_route's
    // Cap::None arm, render_capability_report's floor gap, and the Verdict). A groupwise-int
    // weight IS fed to a tensor-core GEMM from sm_80 up -- mma_bf16 -- so writing Cap::None there
    // would DELETE a real floor in order to add a second one. Two floors, two columns.
    //
    // THE CLAIM THIS COLUMN MAKES is deliberately about a FILE and not about a card: the named
    // kernel consumes THIS format's persisted bytes and emits no tensor-core instruction, so it is
    // executable on any rung that can assemble it. Empty (the default) means "this format has no
    // tensor-core-free kernel in this tree". Measured for the four rows below with this tree's own
    // flags -- nvcc 12.8 V12.8.61, `-cubin -arch=sm_52` and `-arch=sm_61`, then `nvdisasm -c`:
    // rc=0, ZERO HMMA/IMMA, ZERO ldmatrix, ZERO LDGSTS (the staged copy takes the synchronous
    // cuda_pipeline arm below sm_80 and does not trap).
    std::string_view simt_kernel_evidence = {};

    // -----------------------------------------------------------------------
    // THE FOURTH FLOOR: the fp16 TENSOR-CORE arm over THIS format's own plane.
    // -----------------------------------------------------------------------
    // WHY A FOURTH COLUMN, AND WHY IT IS NOT `fallback_required`. `fallback_required` is
    // already taken: NVFP4's Cap::Fp16Mma there means the QPN W4A16 family, which consumes
    // PACKED e2m1/e4m3 codes. A BF16 PLANE is a different kernel family reading different
    // bytes, and writing its facts into the QPN column would make the one column answer two
    // questions -- the drift this table's own header forbids. Two families, two columns.
    //
    // WHAT IS DIFFERENT ABOUT THIS ONE, AND IT IS THE WHOLE REASON IT EXISTS. The two
    // predicates above are GATED ON THE RUNG being unable to do better (the fp16 arm) or on
    // no tensor-core route existing at all (the SIMT arm). This one is the opposite: it is
    // the answer where the rung HAS fp16 tensor cores and the format's declared floor names
    // a tensor-core family the rung does NOT have -- Volta and Turing against a Cap::Bf16Mma
    // floor, i.e. exactly rungs 70 and 75. Today those two rungs answer with the FFMA/SIMT
    // arm; with this column they answer with a real fp16 tensor-core GEMM that consumes the
    // SAME persisted bf16 bytes (no requantization, no fp16 weight copy).
    //
    // WHERE THE KERNEL IS: src/ops/linear/bf16/bf16_mma_fp16.cuh. The channel it uses is
    // per-rung and MEASURED, in kFp16PlaneChannelRungs below -- `mma.sync.aligned.m8n8k4` on
    // sm_70 (Volta's only fp16 tensor-core form) and `mma.sync.aligned.m16n8k8` on sm_75
    // (Turing's native form). Empty (the default) means "this format has no fp16-plane kernel
    // in this tree", which is every row but BF16 today.
    Cap fp16_plane_required = Cap::None;
    std::string_view fp16_plane_kernel_evidence = {};
    // -----------------------------------------------------------------------
    // THE FLOOR OF THE THIRD FLOOR (added by dl/gapclose, F-769).
    // -----------------------------------------------------------------------
    // The rung BELOW which the tensor-core-free kernel named in `simt_kernel_evidence` was not
    // measured. `0` means "no separate floor": the row's evidence was taken on the rungs the row
    // already admits, and its admission must not move by one cell because this field appeared --
    // which is the state of every row but NVFP4, whose evidence (nvfp4_gemv / nvfp4_small_t) was
    // censused at sm_75 and sm_80 only.
    //
    // WHY A FIELD AND NOT A SENTENCE. Clause 1 of simt_floor_executable() asks only "does this
    // format's row name a tensor-core-free kernel"; that question has the same answer on every
    // rung, so a column WITHOUT this floor admits the format on the six sub-70 rungs as well --
    // a widening of 6 x 9 = 54 cells beyond the band, on rungs this toolchain CANNOT census
    // (CUDA 13.3 refuses `compute_52` / `compute_61` outright: "Unsupported gpu architecture").
    // The floor keeps the field's claim a claim about a MEASURED rung; where it is not measured
    // the answer is the refusal it was before. IT IS A BOUND, NOT A MEASUREMENT, and it says so.
    int simt_evidence_floor_sm = 0;
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
     "ops/linear/bf16/bf16_gemm_mma.cuh:296 -> ops/common/mma.cuh:109 (mma_bf16, "
     "mma.sync.aligned.m16n8k16...bf16.bf16.f32)",
     // ---------------------------------------------------------------------------------------
     // THERE IS NO fp16 FALLBACK FOR BF16, AND THAT IS A MEASURED ABSENCE RATHER THAN AN
     // OVERSIGHT. The fp16 avenue this table honours is
     // fp16_fallback_executable() -> the QPN family -> mma.sync.aligned.m8n8k4, and the QPN
     // kernels consume PACKED 4-BIT e2m1 CODES (skinny_nvfp4_qpn*) or PACKED e4m3 CODES
     // (skinny_fp8_qpn8*), expanded inline to fp16. Neither reads a bf16 weight plane, and a
     // grep of every arm under src/ops/linear/ finds exactly two users of the m8n8k4 fp16
     // channel -- qpn_kernels.cuh:729 and :929 -- both inside that family. So writing
     // Cap::Fp16Mma here would name a route to a kernel that cannot consume these bytes: the
     // phantom this table exists to prevent, and the same failure mode as the sm_100 nvfp4 row
     // recorded above. dl/floorfix measured it rather than inheriting F-719's reading.
     // WHAT EXISTS INSTEAD is the TENSOR-CORE-FREE arm at the bottom of this row, and it is
     // better on the axis the owner put first: bf16 -> FP32 FMA loses nothing (bf16 is exactly
     // representable in fp32), whereas bf16 -> fp16 is precision-up and RANGE-down (fp16's
     // exponent is 5 bits against bf16's 8, so |x| > 65504 saturates). See the row's own
     // measurement note.
     Cap::None,
     {},
     "ops/linear/bf16/bf16_gemv.cuh and ops/linear/bf16/bf16_small_t.cuh (bf16_gemv_kernel / "
     "bf16_small_t_inner_kernel; BF16 x BF16 decoded to FP32 and FMA'd on the FP32 pipe into "
     "per-row accumulators, NO tensor-core instruction) -> TUs ops/linear/bf16/bf16_gemv.cu "
     "and ops/linear/bf16/bf16_small_t.cu, src/CMakeLists.txt:315 and :317, both unconditional. "
     "These are not a fallback SOMEONE HAS TO WIRE: bf16_dispatch.cpp:33/:36 already selects "
     "them BY SHAPE and WITHOUT CONSULTING THE ARCH (t == 1 -> bf16_gemv; t <= "
     "kBf16LinearSmallTDispatchEnd -> bf16_small_t), so on this box the BF16 decode path is "
     "this kernel on every card. MEASURED 2026-09-25 (dl/floorfix): one `-cubin` per TU with this "
     "tree's own include/define set, then `nvdisasm -c` -- see "
     "dl/floorfix/out/tc_free_census/census.txt, whose own rows carry the arch AND the reason one "
     "arch is missing. The result for this pair at sm_75: rc=0, ZERO HMMA, ZERO IMMA, ZERO LDSM, "
     "ZERO call into any emulation routine, against 80 and 33328 FMA-pipe instructions -- and the "
     "SAME instrument reports 1088 HMMA on ops/linear/bf16/bf16_gemm_mma.cu at sm_80, so a zero "
     "here is a reading and not a blind probe. sm_70 COULD NOT BE MEASURED and is NOT claimed: "
     "every toolkit on this box (13.0/13.1/13.3; 12.8 is gone, only its ld.so.conf fragment "
     "remains) rejects `-arch=sm_70` AND `-arch=compute_70` with `nvcc fatal : Unsupported gpu "
     "architecture`. sm_75 is therefore a PROXY for sm_70 here, and the reason it is a good one "
     "is a source fact rather than a hope: this kernel's body contains no mma/ldmatrix call at "
     "all, and the only arch guard on its path keys NINFER_MEMORY_HAS_CP_ASYNC on "
     "__CUDA_ARCH__ >= 800 (ops/common/memory.cuh:30), so sm_70 and sm_75 take the SAME arm. "
     "(That census is the arm this row feeds: unlike the four groupwise-int rows, whose census "
     "was taken at sm_52/sm_61, this row's floor is unmet on sm_70/sm_75, so the measurement had "
     "to be taken at those rungs -- which is the measurement F-709.5 recorded as NOT TAKEN.)",
     // ---------------------------------------------------------------------------------------
     // THE FOURTH FLOOR, AND WHY BF16 IS THE FIRST ROW TO CARRY IT. The paragraph above is
     // still TRUE and is NOT deleted: there is no fp16 fallback for BF16 *in the QPN sense* --
     // the QPN kernels consume packed 4-bit e2m1 or packed e4m3, and neither reads a bf16
     // plane. What changed is that the missing piece the route selector named ("an explicit
     // bf16 -> fp16 weight conversion ahead of an m8n8k4 GEMM ... the missing piece for these
     // formats is the conversion and the kernel that drives it") now EXISTS:
     // src/ops/linear/bf16/bf16_mma_fp16.cuh. So on rungs 70 and 75 -- the two rungs that have
     // fp16 tensor cores and NOT Cap::Bf16Mma, on the record as "the 122 refusing cells of
     // dl/ladderacpt's census" -- a bf16 artifact is served by a real fp16 tensor-core GEMM
     // instead of the FFMA/SIMT one.
     //
     // WHAT IT COSTS, NAMED RATHER THAN HIDDEN. bf16 -> fp16 is exact in the mantissa and
     // NARROWER in range: |x| > 65504 saturates. The SIMT arm is exact where this one
     // saturates, and the owner's first rule is precision first -- so BOTH are reported by the
     // gate (whichever fallback this column carries is what render_fallback_notice prints) and
     // the engine-side plan may decline it (bf16_fp16_route.h).
     //
     // WHAT IS NOT DONE: the SIMT column is not deleted, and no rung is moved off it where it
     // is the only answer. On rungs 50/52/53/60/61/62 -- no tensor core at all, where FFMA IS
     // the native route -- this arm cannot fire (kFp16PlaneChannelRungs has no row below sm_70,
     // and nothing assembles `mma` below .target sm_70), so those six keep the FFMA route and
     // that is not a defect.
     Cap::Fp16Mma,
     "ops/linear/bf16/bf16_mma_fp16.cuh:1 (bf16_mma8_kernel / bf16_mma1688_kernel; the artifact's "
     "OWN bf16 plane is converted inline to fp16 and fed to the fp16 tensor core through "
     "ops/common/mma.cuh's mma_f16_m8n8k4 (sm_70, Volta's only fp16 mma form) or mma_f16_m16n8k8 "
     "(sm_75, Turing's native form). NO requantization and NO second weight copy: the persisted "
     "bytes are read as-is. CHANNEL LOWERING IS PER RUNG AND MEASURED -- see "
     "kFp16PlaneChannelRungs below. KERNEL CENSUS, THIS LINE, CUDA 13.3, `-cubin -arch=<a>` + "
     "`nvdisasm -c` on a TU that instantiates BOTH kernels (dl/fp16route/logs/newkernel_*.sass): "
     "sm_75 HMMA.884 = 64 (= 16 PTX mma sites x 4 STEPs, 0 emulation CALL) AND HMMA.1688 present; "
     "sm_80/86/89/90/100/103/120a HMMA.884 = 0 with 16 CALLs into the same in-cubin FFMA routine the "
     "m8n8k4 channel has always used, while HMMA.168x stays PRESENT -- which is the measurement "
     "that makes m16n8k8 the right channel for a rung whose run may be replayed on a modern box. "
     "sm_70 is NOT COMPILABLE here (all three toolkits reject -arch=sm_70) and is therefore NOT "
     "censused; the sm_70 channel row is inherited from kQpnMmaRungs' own CUDA-12.8 census of the "
     "SAME single channel, which is the same inheritance fp16_fallback_executable() performs."},
    {artifact::NumericFormat::FP32, Cap::None,
     "scale words / control payloads (artifact/reader.h:23); never a tensor-core operand"},
    {artifact::NumericFormat::I32, Cap::None,
     "control and index payloads (artifact/reader.h:24); never a tensor-core operand"},
    {artifact::NumericFormat::Q4G64_F16S, Cap::Bf16Mma,
     "ops/linear/q4/q4_rowsplit_gemm_mma.cuh:341 and q4/q4_small_t_mma.cuh:168 (mma_bf16, "
     "NOT a 4-bit tensor core)",
     Cap::Bf16Mma,
     "the groupwise-int BF16 A16 route above, kept unchanged as the SECOND floor",
     "ops/linear/q4/q4_rowsplit_gemm_simt.cuh:238 (q4_rowsplit_gemm_simt_kernel; Q4 G64 x BF16, "
     "FP32 FMA into per-column accumulators, no tensor-core instruction) -> TU "
     "ops/linear/q4/q4_rowsplit_gemm_simt.cu, src/CMakeLists.txt:374"},
    {artifact::NumericFormat::Q5G64_F16S, Cap::Bf16Mma,
     "ops/linear/q5/q5_rowsplit_gemm_mma.cuh:385 (mma_bf16)",
     Cap::Bf16Mma,
     "the groupwise-int BF16 A16 route above, kept unchanged as the SECOND floor",
     "ops/linear/q5/q5_rowsplit_gemm_simt.cuh:284 (q5_rowsplit_gemm_simt_split4_kernel) and :156 "
     "(split2); warp-per-row, 128-bit coalesced weight-plane loads, FP32 FMA, no tensor-core "
     "instruction -> TU ops/linear/q5/q5_rowsplit_gemm_simt.cu, src/CMakeLists.txt:379"},
    {artifact::NumericFormat::Q6G64_F16S, Cap::Bf16Mma,
     "ops/linear/q6/q6_rowsplit_gemm_mma.cuh:394 (mma_bf16)",
     Cap::Bf16Mma,
     "the groupwise-int BF16 A16 route above, kept unchanged as the SECOND floor",
     "ops/linear/q6/q6_rowsplit_gemm_simt.cuh:223 (q6_rowsplit_gemm_simt_kernel; Q6 G64 x BF16, "
     "FP32 FMA, no tensor-core instruction) -> TU ops/linear/q6/q6_rowsplit_gemm_simt.cu, "
     "src/CMakeLists.txt:418"},
    {artifact::NumericFormat::W8G32_F16S, Cap::Bf16Mma,
     "ops/linear/w8/w8_rowsplit_gemm_mma.cuh:267 and w8/w8_small_t_mma.cuh:230 (mma_bf16, "
     "NOT mma_s8: the groupwise-int profile is not an int8-tensor-core route)",
     Cap::Bf16Mma,
     "the groupwise-int BF16 A16 route above, kept unchanged as the SECOND floor",
     "ops/linear/w8/w8_rowsplit_gemm_simt.cuh:149 (w8_rowsplit_gemm_simt_kernel; W8 G32 x BF16, "
     "bytes converted to FP32 and FMA'd, no tensor-core instruction) -> TU "
     "ops/linear/w8/w8_rowsplit_gemm_simt.cu, src/CMakeLists.txt:419. NOTE this one has NO "
     "format-level host shape table: its callers are the per-projection TUs "
     "(ops/attn_input_proj/w8/w8_attn_input_gemm_simt.cu, ops/linear_add/w8/"
     "w8_linear_add_gemm_simt.cu), which is why the arm this column feeds names a KERNEL FILE "
     "and leaves the exact-geometry decision to the caller"},
    {artifact::NumericFormat::NVFP4, Cap::Mxf4Nvfp4BlockScale,
     "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh:308 -> ops/common/mma.cuh:269 "
     "(kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.e2m1.e2m1). MEASURED: that asm "
     "assembles on sm_120a/sm_121a and the 120/121 family targets only; sm_100a/sm_103a/"
     "sm_110a reject it, so this format's floor is sm_120, not sm_100.",
     // The fallback consumes the SAME packed e2m1 codes + e4m3 scales this format already
     // stores; it does not requantize. mma.cuh:191's m8n8k4 form is the channel.
     Cap::Fp16Mma,
     "QPN W4A16: src/ops/linear/qpn/qpn_kernels.cuh -- skinny_nvfp4_qpn_simt<M> (M 1..3), "
     "skinny_nvfp4_qpn<1> (M 4..8), skinny_nvfp4_qpn<2> (M 9..16), host entry gemm_qpn in "
     "src/ops/linear/qpn/qpn_host.cu -- expands the 4-bit e2m1 codes inline into fp16 mma "
     "and needs Cap::Fp16Mma only. Its tensor-core channel is mma.sync.aligned.m8n8k4 "
     "(qpn_kernels.cuh:729,:929; helper ops/common/mma.cuh mma_f16_m8n8k4). NOTE, MEASURED "
     "2026-09-25 (dl/floorfix): this route is SELECTED BY THIS TABLE AND NOT DISPATCHABLE BY "
     "THE ENGINE -- qpn_arch_route.h:93-108 records the two missing ports (a native -> "
     "qpn_prepack weight converter and a bf16 -> fp16 activation/output step), and "
     "dispatch_qpn_fallback() refuses with QpnWeightLayout::NativeBlockScale, which is the only "
     "layout an artifact delivers. THE ENGINE'S ANSWER IS AT THE OP, not in this table, and it "
     "is a THIRD option this catalogued list does not carry because it is not a lower FLOOR: "
     "src/ops/linear/nvfp4/nvfp4_dispatch.cpp now takes launch_a16 -- nvfp4_gemv / "
     "nvfp4_small_t, FP32 FMA over this artifact's own planes -- instead of throwing. Its "
     "census is dl/floorfix/out/tc_free_census/census.txt. ",
     // -----------------------------------------------------------------------
     // NOTE THE COMMA JUST ABOVE, AND IT IS THE WHOLE DIFFERENCE BETWEEN THIS ROW PARSING AND NOT:
     // the fields of this row are ADJACENT STRING LITERALS, and adjacent literals CONCATENATE. The
     // first revision of this patch ended the QPN paragraph without one, so the tensor-core-free
     // evidence below was glued onto the paragraph and the next initialiser landed on a
     // `std::string_view` -- measured: `error: could not convert 'ninfer::caps::Cap::None' from
     // 'ninfer::caps::Cap' to 'std::string_view'` at arch_caps.h:881, first build attempt.
     // -----------------------------------------------------------------------
     // THE THIRD FLOOR, AND THE 76..119 HOLE IS CLOSED (dl/gapclose, F-769).
     // -----------------------------------------------------------------------
     // THIS PARAGRAPH USED TO SAY THE HOLE "STAYS OPEN AND IS REPORTED, NOT PAPERED OVER" and
     // named the repair. The repair is LANDED, and the two halves are ONE landing because either
     // one alone is a false claim:
     //   * the TABLE half (this column) is what lets the gate see the tensor-core-free kernel;
     //   * the OP half (nvfp4_dispatch.cpp's rung-aware W4A4-vs-A16 decision) is what makes the
     //     run take that kernel instead of falling through to launch_nvfp4_w4a4.
     // Adding this column WITHOUT the op guard was the banned false positive -- simulated green
     // on this sm_120a cubin while a real Ada card faulted inside launch_nvfp4_w4a4 -- and
     // adding the op guard without this column would leave the cells refused for a route the
     // runtime now knows how to take. Both are in the same landq item, with the census below.
     // THE EVIDENCE, MEASURED. src/ops/linear/nvfp4/nvfp4_gemv.cu (t == 1) and
     // nvfp4_small_t.cu (t == 2..kNvfp4LastSmallT) are reached through nvfp4_dispatch.cpp's
     // launch_a16 and compute over THIS artifact's own [N][K/2] codes + [N][K/16] scales with
     // FP32 FMA: no kind::mxf4nvf4, no tensor-core instruction of any family. Census, rc=0 on
     // every rung listed, with the counter SHOWN TO MOVE on a neighbour
     // (POSCTL-bf16_gemm_mma sm_80 -> HMMA=1088, LDSM=816):
     //   nvfp4_gemv.cu    sm_75  rc=0  HMMA=0 IMMA=0 LDSM=0 EMCALL=0  FMA-pipe=2768
     //   nvfp4_small_t.cu sm_75  rc=0  HMMA=0 IMMA=0 LDSM=0 EMCALL=0  FMA-pipe=206720
     //   nvfp4_gemv.cu    sm_80  rc=0  HMMA=0 IMMA=0 LDSM=0 EMCALL=0  FMA-pipe=2768
     // THE ISA SIDE OF THIS ROW IS *NOT* RE-MEASURED HERE, AND THE TWO INSTRUMENTS THAT CLAIMED TO
     // WERE BROKEN. `dl/nvfp4hole/out/M2b_census.tsv` and `dl/floorfix/out/tc_free_census/rc.txt`
     // BOTH report their ISA probe as `ops/linear/nvfp4/nvfp4_w4a4_mma.cu ... rc=1 / NO CUBIN`, and
     // the handover read that as ptxas refusing the instruction. MEASURED 2026-09-25 by dl/gapclose:
     // THAT PATH DOES NOT EXIST -- the asm lives in the HEADER `nvfp4_w4a4_mma.cuh`, and its
     // instantiating TU is `nvfp4_w4a4.cu` (`ls ops/linear/nvfp4/*.cu` lists no `_mma.cu`). Both
     // `rc=1`s are "No such file or directory". The ISA fact therefore stands on THIS ROW's own
     // `required` column text -- ptxas rejecting kind::mxf4nvf4 on sm_100a/sm_103a/sm_110a, so the
     // floor is sm_120 -- and on the ladder carrying Cap::Mxf4Nvfp4BlockScale on 120/121 only. It is
     // NOT claimed here from those two census rows, and the corrected census is
     // `dl/gapclose/scripts/b_isa.sh` (on `nvfp4_w4a4.cu`, the TU that exists).
     // THE FLOOR, AND IT IS A BOUND RATHER THAN A MEASUREMENT: 80 is the lowest rung admitted,
     // because 70/75 are already admitted through fp16_fallback_executable()'s QPN arm (their
     // answer must not move) and the six sub-70 rungs CANNOT be re-censused by this toolchain
     // (CUDA 13.3 refuses `compute_52`/`compute_61`: Unsupported gpu architecture), so they are
     // kept OFF this column BY NAME rather than admitted on an unmeasured rung. The population
     // this moves is the eight in-band rungs 80/86/87/88/89/90/100/103 and nothing else.
     "ops/linear/nvfp4/nvfp4_gemv.cu and ops/linear/nvfp4/nvfp4_small_t.cu, entered through "
     "src/ops/linear/nvfp4/nvfp4_dispatch.cpp launch_a16: FP32 FMA over the artifact's own "
     "packed e2m1 codes and e4m3 scales, no tensor-core instruction. Census above.",
     Cap::None,
     {},
     80},
    {artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S, Cap::Bf16Mma,
     "A16 route: ops/linear/fp8/fp8_a16_gemm_mma.cuh:206 (mma_bf16). A8 route: "
     "ops/linear/fp8/fp8_a8_mma.cuh:247 (mma_fp8_e4m3, kind::f8f6f4, Blackwell form). "
     "THERE IS STILL NO fp16 FALLBACK ROW HERE, and F-719's reading of it was INCOMPLETE -- "
     "dl/floorfix took the measurement and it is a SCALE MODEL mismatch, not only a missing "
     "host entry. The fp8 arm of the QPN family (skinny_fp8_qpn8, skinny_fp8_qpn8_mt2, "
     "qpn_kernels.cuh:1257/:1357) reads its epilogue scale as `const float* tscale` indexed "
     "`tscale[blockIdx.x]` (qpn_kernels.cuh:1258/:1273 and :1358/:1372), i.e. ONE fp32 PER "
     "32-OUTPUT-COLUMN TILE, while this format carries ONE BF16 PER OUTPUT ROW "
     "(artifact/storage_layouts.cpp row_scale_geometry -> scale_word_bytes = 2; "
     "fp8_format.cpp:46/:58 require scale_ne[0] == n and scale_nb[0] == 2). Thirty-two "
     "distinct row scales would be read as one reinterpreted fp32 pair -- silent numerical "
     "corruption, the failure class this table refuses on. Adding the gemm_qpn fp8 dispatch "
     "F-719 asked for would therefore have created the phantom in a NEW way (a route that "
     "launches and computes the wrong answer), so it was NOT added. What would have to exist "
     "first: either a kernel that applies a PER-ROW scale (the CUDA-core arm below already "
     "does, which is why the sm_70/sm_75 answer is a route and not a refusal), or a "
     "prepack/requantizer that folds per-row scales into per-tile ones -- and that one owes its "
     "own precision measurement, because it changes the numbers. The mma channel is NOT the "
     "missing piece, as F-719 said; the scale axis is.",
     Cap::None,
     {},
     "ops/linear/fp8/fp8_gemv.cuh and ops/linear/fp8/fp8_small_t.cuh (fp8_gemv_kernel / "
     "fp8_small_t_kernel; 'Reusable row-scaled FP8 T=1 CUDA-core mainloop' -- each warp owns "
     "contiguous output rows, decodes this format's OWN qdata plane to BF16 and FMA's it into "
     "FP32 accumulators, applying __bfloat162float(row_scales[row]) ONCE per output row; NO "
     "tensor-core instruction) -> TUs ops/linear/fp8/fp8_gemv.cu and "
     "ops/linear/fp8/fp8_small_t.cu, src/CMakeLists.txt:320 and :321, both unconditional. "
     "NOTE the scale model: these read the per-output-row bf16 plane directly "
     "(fp8_gemv.cuh:99 `const __nv_bfloat16* row_scales`, :149/:152/:168), which is exactly "
     "what this format stores -- so unlike the QPN fp8 arm above there is nothing to convert. "
     "Already reached by shape -- fp8_dispatch.cpp:81-85 calls launch_fp8_decode for t == 1 and "
     "launch_fp8_small_t above it. MEASURED 2026-09-25 (dl/floorfix): one `-cubin` per TU with "
     "this tree's own flags, then `nvdisasm -c` -- "
     "dl/floorfix/out/tc_free_census/census.txt. At sm_75 the pair reports rc=0, ZERO HMMA, ZERO "
     "IMMA, ZERO LDSM, ZERO emulation CALL, with 320 and 34112 FMA-pipe instructions; the same "
     "instrument reports 192 HMMA on fp8_a16_gemm_mma.cu at sm_80, which is what makes the zero a "
     "reading. sm_70 is NOT MEASURABLE on this box (all three toolkits reject -arch=sm_70 and "
     "-arch=compute_70) and is NOT claimed; sm_75 is a PROXY, justified because these kernels "
     "contain no mma/ldmatrix call at all."},
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
    // ---- appended 2026-09-29 (line flowopen, marker F1173) ------------------------------------
    // The three low-bit carriers above, each with a Cap::None row and a note that NAMES the
    // absent consumer.  This is not a formality: appending the enum members without these
    // rows fails the static_assert at the bottom of this file, whose whole purpose is that a
    // format cannot exist silently unconsumed.
    {artifact::NumericFormat::Q1G64_F16S, Cap::None,
     "1-bit grouped codes in row-split-k128-v1, geometry artifact/storage_layouts.cpp "
     "quant_geometry() {64, 8, 0}.  The source side already decodes this: "
     "tools/convert/gguf_kquant.py carries DEQUANTIZERS[41] = decode_q1_0 (ggml Q1_0, 18 B "
     "per 128 values, d = sum|x|/128 plus a sign bitmap), recorded in dl/type41/REPORT.md. "
     "What does NOT exist is a consumer: no QType enumerator (src/core/tensor.h) and no arm "
     "of src/ops/linear executes 1-bit codes, so the only route to a GPU is a decode to "
     "BF16 first -- which is what a pre_extract converter does, and which means this format "
     "is a CARRIER, never a tensor-core operand."},
    {artifact::NumericFormat::Q2G64_F16S, Cap::None,
     "2-bit grouped codes, geometry quant_geometry() {64, 16, 0}.  The source side is where "
     "the care is needed rather than here: Prism's private PQ2_0 (ggml id 142, 34 B/128) is "
     "decodable in this tree, while UPSTREAM's Q2_0 is id 42 and is deliberately absent from "
     "GGML_TYPES because it carries two different block layouts across releases -- "
     "gguf_kquant's own comment refuses to guess one, since a table that guesses reads the "
     "other silently, and type_name(42) degrades to \"type42\".  No QType enumerator and no "
     "src/ops/linear arm exists for 2-bit codes; decode-to-BF16 only."},
    {artifact::NumericFormat::Q3G64_F16S, Cap::None,
     "3-bit grouped codes, geometry quant_geometry() {64, 16, 8} == 2 + 1 bit.  CARRIES "
     "GROUPED 3-BIT CODES AND NOT GSQ: the GSQ spelling is pack-int32-le-v1 with pack_factor "
     "10, bias 4 and codes in [-4, 3] (dl/gsq3/REPORT.md), which is ten codes per int32 -- a "
     "different packing from a contiguous 3-bit group of 64, and it needs its own layout "
     "name rather than this format.  Saying so here is cheaper than discovering it in a "
     "converter.  No QType enumerator and no src/ops/linear arm; decode-to-BF16 only."},
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
// THE SIX SUB-70 RUNGS ADDED 2026-09-24 (F702) ARE NOT IN IT EITHER, for the same reason one
// step earlier: qpn_arm_reachable() REQUIRES Cap::Fp16Mma, and sm_50/52/53/60/61/62 carry
// Cap::None (measured: `mma` needs .target sm_70 -- see the block above kArchLadder). The arm is
// therefore not reached at all on those rungs, so they get no row here and NONE WAS ADDED. This
// is the same fail-closed shape as sm_120/121 and NOT a gap: a lowering row for a rung the arm
// never asks about would be a claim nothing exercises, and it would also read as "the QPN
// channel was measured on this ISA", which it was not.
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

// ---------------------------------------------------------------------------
// Is the TENSOR-CORE-FREE (FFMA/SIMT) floor EXECUTABLE on this rung?
// ---------------------------------------------------------------------------
//
// The third floor, and it is a different KIND of fact from the two above it: `required` and
// `fallback_required` are both Cap bits -- tensor-core instruction families -- so a rung whose
// measured set is Cap::None satisfies neither, and every GEMM format used to refuse there. That
// is right for the formats whose only kernel is a tensor-core one, and wrong for the four
// groupwise-int formats this build also ships an FMA-pipe GEMM for.
//
// Three clauses, and the middle one is what keeps this from becoming a blanket "SIMT is free":
//
//   1. the TABLE names a tensor-core-free kernel for this format
//      (kFormatRequirements.simt_kernel_evidence; empty means it does not), and
//   2. the rung cannot serve THIS FORMAT'S DECLARED FLOOR -- `!covers(rung->caps,
//      requirement->required)`. Read that as the exact thing it says, twice over:
//        * `caps == Cap::None` (the six pre-75 rows) is the case this clause was FIRST written
//          for and it is still covered, because a rung with no tensor core covers no Cap bit;
//        * a rung that HAS a tensor core but not the one this format's floor names is a
//          RESCUE and not a downgrade: the answer above this point is a refusal, so the
//          tensor-core-free kernel is the only executable route there is, not a cheaper one
//          chosen over a working one. sm_70 (Cap::Fp16Mma) and sm_75 (Cap::Fp16Mma|Int8Mma)
//          against a Cap::Bf16Mma floor are exactly that case -- the 122 refusing cells of
//          dl/ladderacpt's census.
//      MEASURED, 2026-09-25 (dl/floorfix): that widening moves cells on TWO rungs and no
//      others for a tensor-core floor -- {70, 75} for BF16 / Q4G64_F16S / Q5G64_F16S /
//      Q6G64_F16S / W8G32_F16S / FP8_E4M3FN_ROW_BF16S, and {70, 75, 80, 86, 87, 88, 89, 90,
//      100, 103} for NVFP4, whose floor Cap::Mxf4Nvfp4BlockScale exists only on 120/121. On
//      every rung that DOES cover the floor the earlier branch returns first, so this arm can
//      never take a format off its tensor-core route -- see test_arch_caps.cpp, which asserts
//      that direction rather than the widening alone.
//      WHAT THE CLAUSE IS AND IS NOT. It was `caps == Cap::None` EXACTLY, and the objection to
//      widening it was that a rung with a tensor core would be handed a "silent downgrade". The
//      silence is what was wrong with that, not the rescue: the arm below records
//      FormatFallback{tensor_core_free = true}, render_fallback_notice() prints it, and
//      select_route's ConservativeSimt reason names the kernel FILE. Nothing here is silent,
//      and nothing here is reached while a tensor-core route for the same format is still
//      standing. What the equality DID buy is gone with it, so it is stated rather than
//      inherited: this predicate's coverage is now every (rung, format) pair whose table row
//      names a tensor-core-free kernel and whose declared floor that rung does not cover --
//      computed, not restated, by test_arch_caps.cpp.
//   3. the number has a ladder row at all -- an unlisted number gets an answer from no table.
//
// NOT a clause: the arch list the binary was compiled for. A build that names only sm_120a can
// still be ASKED about sm_61; that is the whole point of the parameterized selector, and the
// answer is about the rung, not about the cubins in one binary.
//
// NOT CLAIMED: that any of these rungs has been RUN. This is an assembly-and-build fact -- the
// kernel compiles for that ISA with zero tensor-core instructions and its source line is
// unconditional in src/CMakeLists.txt. The rung rows in kArchLadder carry their own caliber and
// say which of the six were configured, swept, or neither. The rows that carry a
// simt_kernel_evidence entry and the flags its census used are named in each row's own text.
// Whether the fp16-plane GEMM is compiled into THIS binary. Published by src/CMakeLists.txt on
// the same lines that add ops/linear/bf16/bf16_mma_fp16.cu to a target, exactly like
// NINFER_HAVE_QPN and for the same reason: a hand-written -D with the source line removed would
// name a kernel the binary does not contain. The tie is the link-time reference to
// bf16_mma1688_kernel in tests/test_bf16_fp16_plane.cpp.
//
// DECLARED HERE, ABOVE simt_floor_executable(), because that predicate's new clause 4 reads it
// whereas the rest of this arm's own block is defined below it. The value is a compile-time
// constant, so the position costs nothing and a forward declaration of the predicate is not
// enough on its own.
#ifdef NINFER_HAVE_BF16_FP16_MMA
inline constexpr bool kFp16PlaneInBuild = true;
#else
inline constexpr bool kFp16PlaneInBuild = false;
#endif

// Forward declaration: the definition, with its four clauses and its measured channel table,
// is below simt_floor_executable() -- which calls it. A non-template function's body resolves
// names at its point of definition, so this declaration is required rather than cosmetic.
[[nodiscard]] inline bool fp16_plane_executable(int sm, artifact::NumericFormat format,
                                                bool fp16_plane_in_build) noexcept;

[[nodiscard]] inline bool simt_floor_executable(int sm, artifact::NumericFormat format) noexcept {
    const FormatRequirement* requirement = format_requirement(format);
    if (requirement == nullptr || requirement->simt_kernel_evidence.empty()) { return false; }
    // -----------------------------------------------------------------------
    // CLAUSE 5 (added by dl/gapclose, F-769): the evidence's own measured FLOOR.
    // -----------------------------------------------------------------------
    // This arm is the ONLY one whose evidence is a per-rung MEASUREMENT rather than a build
    // fact, and a measurement has a rung below which it was not taken. With the floor at its
    // default 0 the clause is false and nothing moves; with a floor named, a rung below it keeps
    // the refusal it had. See simt_evidence_floor_sm's own note for why the six sub-70 rungs
    // must keep it: they cannot be censused by this toolchain at all.
    if (requirement->simt_evidence_floor_sm != 0 && sm < requirement->simt_evidence_floor_sm) {
        return false;
    }
    const ArchRung* rung = arch_rung(sm);
    if (rung == nullptr) { return false; }
    // -----------------------------------------------------------------------
    // CLAUSE 4 (added by dl/fp16route, F-736): a rung that can serve this format
    // from its OWN fp16 tensor cores is not a rung this arm rescues.
    // -----------------------------------------------------------------------
    // WHY THIS CLAUSE AND NOT A REORDER ANYWHERE ELSE. Clause 2 above was WIDENED by
    // dl/floorfix (F-720) from `caps == Cap::None` to "this rung cannot serve this
    // format's declared floor", and that file argues the widening correctly: "a rung
    // that HAS a tensor core but not the one this format's floor names is a RESCUE and
    // not a downgrade: the answer above this point is a refusal, so the tensor-core-free
    // kernel is the only executable route there is, not a cheaper one chosen over a
    // working one."
    //
    // THE PREMISE OF THAT ARGUMENT -- "the only executable route there is" -- WAS TRUE
    // WHEN IT WAS WRITTEN and is not true any more for BF16 on sm_70/sm_75, because
    // src/ops/linear/bf16/bf16_mma_fp16.cuh now exists. Where a tensor-core route for
    // THIS format IS executable on THIS rung, the FFMA kernel stops being a rescue and
    // becomes the cheaper route chosen over a working one -- which is the thing F-720's
    // own text says this predicate must never be. So the arm is asked only where no
    // tensor-core route for the format exists, and that is one clause rather than a
    // reorder of the two arms: TWO DECIDERS READ THIS PREDICATE (this one and
    // src/core/kernel_route.h's ConservativeSimt arm), so putting the rule HERE is what
    // keeps them from drifting, which is the same reason clause 2's widening lives here.
    //
    // WHAT IT DOES NOT MOVE, computed rather than asserted:
    //   * the six sub-70 rungs: no fp16 tensor core and no kFp16PlaneChannelRungs row, so
    //     fp16_plane_executable() is false and BF16 keeps the FFMA route, which is its
    //     native route. FFMA stays for 50/52/53/60/61/62 ON PURPOSE.
    //   * every format but BF16: `fp16_plane_required` is Cap::None for all of them, so
    //     the new clause is false and nothing moves.
    //   * NVFP4: unchanged on EVERY rung, and by construction -- its fallback column is
    //     the QPN one and its fp16_plane column is empty, so neither this clause nor the
    //     new arm can see it. dl/floorfix measured why that matters (a simt column for
    //     NVFP4 would widen to 86/89/90/100/103 and the op would fall through to a
    //     kind::mxf4nvf4 kernel those ISAs lack).
    //   * a build WITHOUT the fp16-plane kernel: kFp16PlaneInBuild is false, the clause is
    //     false, and this predicate answers exactly what it answered before -- which is
    //     what keeps the shared pin's own tables unambiguous.
    if (fp16_plane_executable(sm, format, kFp16PlaneInBuild)) { return false; }
    return !covers(rung->caps, requirement->required);
}

// ---------------------------------------------------------------------------
// THE fp16-PLANE TENSOR-CORE CHANNEL, PER RUNG, AND ITS MEASURED LOWERING
// ---------------------------------------------------------------------------
// This is a SECOND channel table, not a second copy of the first one. kQpnMmaRungs above
// measures ONE channel -- `mma.sync.aligned.m8n8k4` -- because that is the only channel the
// QPN family emits; a table that answered for a different channel under the same name would
// be the drift this file forbids. The fp16-plane arm emits TWO channels, and the rung picks
// between them for a reason that is an instruction-set fact rather than a preference.
//
// `measured_target` and `evidence` carry the same two fields kQpnMmaRungs carries, and for
// the same reason: a row is never read as a measurement of a target it did not measure.
//
//  * sm_70 -- mma.sync.aligned.m8n8k4, INHERITED rather than re-measured. Volta's fp16
//    tensor core has exactly one mma form, and this tree has already read its SASS: the
//    sm_70 row of kQpnMmaRungs (CUDA 12.8: 1024 HMMA.884, 0 CALLs into the emulation
//    routine). Re-measuring is IMPOSSIBLE on this box, not merely unnecessary -- every
//    toolkit installed here (13.0/13.1/13.3) rejects `-arch=sm_70` with "nvcc fatal :
//    Unsupported gpu architecture", and 12.8 is gone. So the row says INHERITED.
//  * sm_75 -- mma.sync.aligned.m16n8k8, MEASURED THIS LINE with CUDA 13.3. Turing has both
//    forms and this is the faster one; it is also the one that stays HARDWARE above Turing,
//    which is why it is preferred here rather than m8n8k4 (the same choice the m8n8k4
//    emulation makes costly).
//  * NO ROW FOR sm_80 AND UP, and that is fail-closed rather than a gap: those rungs have
//    Cap::Bf16Mma, so the format's own floor is met and the arm is never asked. A row there
//    would be a claim nothing exercises.
enum class Fp16PlaneLowering : std::uint8_t {
    HardwareFp16Mma = 0, // the channel is an HMMA instruction on this rung
    EmulatedFp16Pipe,    // ptxas answers it with a CALL into a software FFMA routine
};

// THE ATOM'S ACTIVATION EXTENT, and it is quoted by TWO surfaces for the same reason: the route
// arm names the channel and the engine-side plan
// (src/ops/linear/bf16/bf16_fp16_route.h) decides whether to take it, and a threshold written
// twice is a threshold that drifts. A warp-level fp16 mma atom consumes 8 activation rows, so
// below 8 tokens the same mma work is issued for fewer useful tokens and the FFMA GEMV -- which
// is weight-bandwidth bound and cannot be helped by a tensor core -- is the better route. The
// kernel carries a static_assert against this value rather than restating the number.
inline constexpr int kBf16Fp16MmaActivationExtent = 8;

struct Fp16PlaneChannelRung {
    int sm;
    std::string_view measured_target;
    std::string_view channel; // the literal the kernel emits, named for a reader
    Fp16PlaneLowering lowering;
    std::string_view evidence;
};

inline constexpr Fp16PlaneChannelRung kFp16PlaneChannelRungs[] = {
    {70, "sm_70 (INHERITED, not re-measured: no toolkit on this box builds sm_70)",
     "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32", Fp16PlaneLowering::HardwareFp16Mma,
     "Volta's fp16 tensor core has exactly ONE mma form, so the channel this arm emits is the "
     "channel kQpnMmaRungs already measured at sm_70 -- CUDA 12.8: 1024 HMMA.884.F32.F32 (224 "
     "mma sites x 4 STEPs of the seven inline-asm QPN kernels, plus 128 from the two "
     "nvcuda::wmma kernels), 0 CALL instructions targeting the emulation routine, 0 "
     "HFMA2.MMA. THE SAME ROW IS THE MEASUREMENT THIS ARM BORROWS, and the borrowing is the "
     "same one fp16_fallback_executable() already does for the QPN family. WHAT IS NOT CLAIMED: "
     "a census of THIS kernel at sm_70 -- not taken, because no toolkit here builds it."},
    {75, "sm_75", "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32",
     Fp16PlaneLowering::HardwareFp16Mma,
     "MEASURED THIS LINE (dl/fp16route, 2026-09-25), CUDA 13.3 V13.3.73, `-cubin -arch=sm_75` on "
     "a TU that instantiates both kernels of ops/linear/bf16/bf16_mma_fp16.cuh, then `nvdisasm "
     "-c` (dl/fp16route/logs/newkernel_75.sass): HMMA.884 = 64 in the m8n8k4 kernel (= 16 PTX "
     "mma sites x 4 STEPs; the PTX site count is read from -ptx, dl/fp16route/logs/"
     "newkernel_75.ptx.log) with 0 CALL instructions into ptxas's software FFMA routine, AND "
     "HMMA.1688 present for the m16n8k8 kernel with 0 such CALLs. TWO COUNTING TRAPS AVOIDED, "
     "both named in kQpnMmaRungs' own method note: the emulation routine lives INSIDE this cubin "
     "as trailing-colon local labels, so it is counted as CALLs WITH THE ROUTINE AS OPERAND (16 "
     "on the emulated rungs, 0 here) and never by symbol lines; and an mma FMA mnemonic is not a "
     "fingerprint of emulation."},
};

inline constexpr std::size_t kFp16PlaneChannelRungCount =
    sizeof(kFp16PlaneChannelRungs) / sizeof(kFp16PlaneChannelRungs[0]);

[[nodiscard]] inline const Fp16PlaneChannelRung* fp16_plane_channel_rung(int sm) noexcept {
    for (const Fp16PlaneChannelRung& r : kFp16PlaneChannelRungs) {
        if (r.sm == sm) { return &r; }
    }
    return nullptr;
}

[[nodiscard]] inline std::string_view fp16_plane_lowering_name(Fp16PlaneLowering l) noexcept {
    switch (l) {
    case Fp16PlaneLowering::HardwareFp16Mma: return "hardware fp16 mma on this rung";
    case Fp16PlaneLowering::EmulatedFp16Pipe: return "emulated on the FP16x2 FMA pipe";
    }
    return "unknown-lowering";
}

// Whether the fp16-plane GEMM is compiled into THIS binary. THE DEFINITION IS ABOVE, beside
// simt_floor_executable(), which reads it; the prose that belongs with the fact lives here so
// the two halves of the arm stay in one place. See the note there for why the constant is
// positioned above that predicate.

// ---------------------------------------------------------------------------
// Is the fp16-PLANE tensor-core arm EXECUTABLE on this rung, in this build?
// ---------------------------------------------------------------------------
// The sibling of fp16_fallback_executable() for the OTHER fp16 family, and it is deliberately
// a separate function rather than another clause inside that one: that predicate's clause 1 is
// "the QPN sources are in this build" and its clause 4 reads the m8n8k4 channel's lowering, so
// folding a second family into it would make one predicate answer for two families with two
// different build facts -- and NVFP4, which is the family that predicate exists for, would
// change meaning. Four clauses here too, and every one is a BUILD fact or a MEASURED fact:
//
//   1. the kernel is in this build (the NINFER_HAVE_BF16_FP16_MMA fact);
//   2. the format's row declares this arm (fp16_plane_required != Cap::None). Today BF16 only.
//   3. the rung covers the arm's floor (Cap::Fp16Mma) -- Volta and Turing do, the six sub-70
//      rungs do not (nothing assembles `mma` below .target sm_70), and every rung above them
//      has Cap::Bf16Mma, so it never reaches here;
//   4. the rung's MEASURED channel lowering is HardwareFp16Mma. Fail-closed on an unmeasured
//      rung, so a rung with no row gets a refusal naming the missing measurement rather than a
//      neighbour's tier.
//
// THE REACHABLE SET IS THEREFORE EXACTLY {70, 75} -- and that is not a coincidence, it is the
// definition of the wall this arm exists for: "the rung has fp16 tensor cores and its route
// still lands on SIMT/FFMA". What this predicate does NOT answer: whether the resulting
// kernel is FASTER. That is a measurement (dl/fp16route benched it) and not a table fact.
[[nodiscard]] inline bool fp16_plane_executable(int sm, artifact::NumericFormat format,
                                                bool fp16_plane_in_build) noexcept {
    if (!fp16_plane_in_build) { return false; }
    const FormatRequirement* requirement = format_requirement(format);
    if (requirement == nullptr || requirement->fp16_plane_required == Cap::None) { return false; }
    const ArchRung* rung = arch_rung(sm);
    if (rung == nullptr) { return false; }
    if (!covers(rung->caps, requirement->fp16_plane_required)) { return false; }
    const Fp16PlaneChannelRung* channel = fp16_plane_channel_rung(sm);
    if (channel == nullptr) { return false; }
    return channel->lowering == Fp16PlaneLowering::HardwareFp16Mma;
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
    // -----------------------------------------------------------------------
    // WHICH OF THE TWO KINDS OF RESCUE THIS IS (v100fix candidate, NOT LANDED).
    // -----------------------------------------------------------------------
    // `fallback_used` is a Cap, and Cap's six bits are ALL tensor-core instruction families, so a
    // rescue carried by an FFMA/SIMT kernel has no bit to write there. Formatting it as
    // "the <Cap::None> fallback" would print `cap_name(Cap::None)` == "none" and read as a
    // capability the card has, which is the opposite of the fact.
    //
    // TRAILING AND DEFAULTED, on purpose: every existing aggregate initialisation of this struct
    // still compiles, so this member cannot change any path that does not set it. It is set in
    // exactly ONE place -- the third-floor arm of evaluate_artifact_formats() -- and read in
    // exactly one place, render_fallback_notice().
    bool tensor_core_free = false;
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
        // ===================================================================
        // THE FLOOR IS NOT MET. TWO LOWER FLOORS ARE ASKED, AND THE ORDER OF THE TWO WAS
        // CHANGED BY dl/floorfix. THE ORDER IS THE FIX, SO IT IS ARGUED RATHER THAN ASSUMED.
        // ===================================================================
        // BOTH questions are the same question -- "is there a kernel this rung can execute that
        // consumes THIS format's bytes?" -- and both are the SAME predicate the route selector
        // calls, so the gate and the route read ONE fact and "the artifact may load" cannot come
        // apart from "there is a kernel to run" on the same card.
        //
        // WHAT THE ORDER WAS: fp16_fallback_executable() first, the tensor-core-free arm second.
        // On sm_70/sm_75 that answered NVFP4 with "the fp16 QPN channel" and the engine then
        // REFUSED at op time, because that channel is selected by these tables and NOT
        // DISPATCHABLE (qpn_arch_route.h:93-108: the native -> qpn_prepack weight converter and
        // the bf16 -> fp16 activation step are both absent, so dispatch_qpn_fallback() throws on
        // QpnWeightLayout::NativeBlockScale, which is the only layout an artifact delivers).
        // MEASURED 2026-09-25 by dl/floorfix: the gate admitted NVFP4 at sm_70 and the op threw.
        // An admission the engine cannot carry out is not a lower floor; it is the phantom this
        // table exists to prevent, wearing the mask of a fallback.
        //
        // WHAT THE ORDER IS AND WHY. The tensor-core-free arm is asked FIRST, because it is the
        // one whose declared evidence is the artifact's own bytes in the artifact's own layout --
        // there is no converter and no dtype step between the selection and a launch -- and the
        // fp16 arm is asked second, where it still ANSWERS for any format the first arm cannot
        // serve. TWO PROPERTIES, both checkable and both checked below in this order's own test:
        //   * nothing moves on a rung that COVERS the format's floor -- the branch above returns
        //     before either arm is reached, so no tensor-core route is ever traded for a
        //     narrower one;
        //   * nothing moves for a format whose row has no tensor-core-free kernel -- the first
        //     arm's clause 1 is empty-evidence fail-closed, so the fp16 arm keeps every cell it
        //     had. That is why FP8_E4M3FN_ROW_F32S (a note row) and every Cap::None metadata row
        //     are untouched.
        // WHAT IT COSTS, NAMED: on sm_70/sm_75 an NVFP4 artifact now takes the FP32-FMA route
        // instead of the QPN fp16 tensor-core one. That is SLOWER and it is the owner's own
        // ordering -- 保精度优先, speed second -- and the faster route is not merely unpicked,
        // it is UNRUNNABLE until two ports land. When they do, the reversal is this order.
        // -------------------------------------------------------------------
        // THE THIRD FLOOR, ASKED BY THE GATE AT LAST (landed by dl/v100fix, F-710; WIDENED and
        // ASKED FIRST by dl/floorfix, F-720).
        // -------------------------------------------------------------------
        // The route selector has asked this question since F-705 (src/core/kernel_route.h:901:
        // `if (simt_floor_executable(sm, format))`). THIS function did not ask it at all until
        // F-710 landed that candidate, so the two deciders disagreed by construction: on sm_61 a
        // groupwise-int-only artifact was REFUSED AT LOAD with `missing = Bf16Mma` while
        // select_route() named the FFMA kernel that would run it. That disagreement is what the
        // shared predicate exists to prevent.
        // MEASURED, both sides, one host probe (dl/v100fix/out/floors_PRE_qpn1.txt):
        //   gate  evaluate_artifact_formats(61, {FP32,I32,Q4,Q5,Q6,W8}) -> Unsupported, 4 gaps
        //   route select_route(61, Q5G64_F16S, m=8)                    -> Selected conservative-simt
        // The predicate is fail-closed, and dl/floorfix's widening of its second clause (from
        // `caps == Cap::None` EXACTLY to "this rung cannot serve this format's declared floor")
        // is the change that brings sm_70 and sm_75 into its coverage. It is stated once, in
        // simt_floor_executable(), with the measured widening set; it is NOT restated here,
        // because two statements of one rule drift.
        // -------------------------------------------------------------------
        // THE FOURTH FLOOR, ASKED FIRST (dl/fp16route, F-736).
        // -------------------------------------------------------------------
        // This arm and the tensor-core-free arm below it are MUTUALLY EXCLUSIVE by
        // construction -- caps::simt_floor_executable() now carries the clause that returns
        // false wherever this question is true -- so the order cannot move a cell either way.
        // It is stated FIRST anyway so that neither predicate has to be read to know the
        // precedence, and so that a future edit to either predicate cannot silently swap them.
        // What it changes on the wall: rungs 70/75 against a Cap::Bf16Mma format used to be
        // answered "TENSOR-CORE-FREE (FFMA/SIMT) kernel"; they are now answered with a real
        // fp16 tensor-core GEMM over the artifact's OWN bytes. The verdict is Supported in
        // both worlds, which is why this is a fallback and not a verdict change.
        if (fp16_plane_executable(sm, format, kFp16PlaneInBuild)) {
            report.fallbacks.push_back(FormatFallback{format, requirement->required & ~rung->caps,
                                                      requirement->fp16_plane_required,
                                                      requirement->fp16_plane_kernel_evidence});
            continue;
        }
        if (simt_floor_executable(sm, format)) {
            report.fallbacks.push_back(FormatFallback{format, requirement->required & ~rung->caps,
                                                      Cap::None,
                                                      requirement->simt_kernel_evidence, true});
            continue;
        }
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
        // v100fix candidate (NOT LANDED): a TENSOR-CORE-FREE rescue is a different KIND of fact
        // from the fp16 fallback and must not borrow its words. This branch is placed BEFORE the
        // existing sentence so the fp16 path below stays byte-identical.
        if (fallback.tensor_core_free) {
            out += " does NOT meet its own floor here: it needs ";
            out += std::string(cap_name(fallback.primary_missing));
            out += ", which this card does not have -- this rung has NO TENSOR CORE AT ALL. It is "
                   "being served by a TENSOR-CORE-FREE (FFMA/SIMT) kernel instead, which consumes "
                   "the SAME persisted bytes (no requantization, no fp16 weight copy):\n"
                   "      kernel : ";
            out.append(fallback.fallback_kernel);
            out += "\n";
            continue;
        }
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
    // PRINTS BY DEFAULT, and this branch is where -- this function is the only production call
    // site in the tree. The comment that used to sit here told callers to call
    // unknown_arch_warning() and log it; measured, no caller in the engine ever did, so the
    // message had no way out and an unplaceable card degraded in silence.
    // NINFER_ARCH_WARN keeps every documented use -- any value still prints -- and gains the
    // off switch an operator expects: 0|off|false|no silences. The ONE refusal path (the throw
    // below) is untouched, so the evidence-backed refusal set is unchanged.
    //
    // ONCE PER DISTINCT CAPABILITY, NOT ONCE PER PROCESS. The first form of this branch
    // announced the FIRST unlisted number and then went silent for the rest of the process, so a
    // second, DIFFERENT unlisted number -- a second target in a serve process, a second rank, any
    // later construct_target -- got no capability floor checked and NOTHING said. That is the
    // same silent hole this branch was added to close, moved one occurrence along. The set below
    // can only ADD announcements: it removes none, it does not change the silencing list above,
    // and it does not touch the throw. The key is report.sm, the capability number the warning
    // already names.
    if (report.verdict == Verdict::UnknownArch) {
        const char* arch_warn = std::getenv("NINFER_ARCH_WARN");
        const std::string_view arch_warn_value = arch_warn != nullptr ? arch_warn : "";
        const bool arch_warn_silenced = arch_warn_value == "0" || arch_warn_value == "off" ||
                                        arch_warn_value == "false" || arch_warn_value == "no";
        static std::mutex arch_warn_mutex;
        static std::vector<int> arch_warn_announced;
        bool arch_warn_first_seen = true;
        {
            const std::lock_guard<std::mutex> arch_warn_lock(arch_warn_mutex);
            for (const int arch_warn_sm : arch_warn_announced) {
                if (arch_warn_sm == report.sm) {
                    arch_warn_first_seen = false;
                    break;
                }
            }
            if (arch_warn_first_seen) { arch_warn_announced.push_back(report.sm); }
        }
        if (!arch_warn_silenced && arch_warn_first_seen) {
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
    {PtxFamily::CpAsync, "src/ops/common/memory.cuh:131", "cp.async.cg.shared.global",
     "NINFER_MEMORY_HAS_CP_ASYNC (memory.cuh:18-20, `__CUDA_ARCH__ >= 800`); else arm traps at "
     "memory.cuh:129."},
    {PtxFamily::CpAsync, "src/ops/common/memory.cuh:169", "cp.async.commit_group",
     "NINFER_MEMORY_HAS_CP_ASYNC; else arm traps at memory.cuh:168. The comment there states "
     "the rule this whole section follows: an empty body would silently drop the group "
     "boundary, so a downstream cp_wait would look satisfied while nothing was in flight -- "
     "and 'the route selector is what keeps the kernel from being reached'."},
    {PtxFamily::CpAsync, "src/ops/common/memory.cuh:188", "cp.async.wait_group",
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
     "the ONLY one of these six names that already occurs in this tree -- in 84 files whole-tree "
     "(45 of them under src/), MEASURED 2026-09-24 by a boundary-safe grep from the tree root, "
     "`grep -rIlE '(^|[^A-Za-z0-9_])gfx906([^A-Za-z0-9_]|$)' .` -- the same class of file as "
     "before, and the count this sentence used to carry (\"in 21 files\") was a reading from an "
     "earlier revision of this tree and is no longer what the tree says. The plumbing it names is "
     "unchanged and still inert: include/ninfer/ops/allreduce.h:143 and "
     "src/ops/common/allreduce.cu:263,:345 (NINFER_GFX906_TP2_FLAG_SYNC), plus "
     "docs/gfx906/*.md (10 files, each carrying a fork-survey PROVENANCE header). `allreduce` has "
     "0 hits in src/CMakeLists.txt, whose source lists are explicit (no GLOB), so this plumbing is "
     "inert. The upstream ENGINE has no gfx906 path that is in the build. "
     "⚠ ONE PART OF THIS SENTENCE WAS ALSO TOO STRONG AND IS CORRECTED: it said every one of the "
     "files is \"a fork-survey borrow whose own provenance header reads ADDITIVE, NOT wired into "
     "any build target\". That is true of docs/gfx906/*.md (10 files) and of the tests/ carriers, "
     "but NOT of src/compat/gfx906/include/** (11 files), which are this tree's own thin "
     "forwarders to core/hip_compat.h and carry no fork-survey header at all.",
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
     "MEASURED 2026-09-24, boundary-safe, from the tree root: 13 files. Under src/ the count is 4 "
     "and EVERY ONE of them is this table, its tests, or a section of THIS SAME FILE -- but the "
     "string \"(0 files)\" and the sentence \"no file in this tree names it\" that used to stand "
     "here were FALSE, and src/ops/common/math.cuh:76 is the counter-example: "
     "\"cvt.rn.bf16x2.f32 has no AMDGCN spelling on gfx906 (nor gfx908/90a/942/1030/1100/1200/"
     "1201)\". A rung's evidence field is what render_amd_format_refusal() prints, so a false "
     "negative here is printed at an operator, not filed. Corrected rather than softened: this "
     "tree does name gfx908, in a comment about an instruction encoding, and it names no gfx908 "
     "kernel, no gfx908 build target and no gfx908 probe.", "",
     "no gfx908 kernel, no gfx908 build target, no gfx908 probe. Every tensor-core format below "
     "is refused by name. This rung is on the ladder because it is a target an operator can "
     "legitimately ask `--offload-arch=` for, and the honest answer for it is a refusal that "
     "names the missing kernels rather than silence."},
    {"gfx90a", "CDNA2", "MI200 / MI210 / MI250 / MI250X",
     "MEASURED 2026-09-24, boundary-safe, from the tree root: 12 files. Under src/ the count is 3 "
     "and every one of them is this table or a section of this same file -- so for THIS rung the "
     "sentence \"no file in this tree names it\" holds for src/ and is FALSE for the tree as a "
     "whole (the 9 outside src/ are this table's tests and the shadowed copies a build dir "
     "carries). Stated with both denominators rather than as \"0 files\", because \"0 files\" was "
     "a reading of one directory presented as a reading of the tree.", "",
     "same as gfx908. It is called out separately because it is the first rung that WOULD "
     "exercise v_mfma_* -- i.e. it is the rung that would answer the mma.sync blocker -- and no "
     "kernel in this tree emits v_mfma_*, so the answer is still a refusal, now for a different "
     "reason: the missing piece is a re-authored kernel, not a hardware feature."},
    {"gfx942", "CDNA3", "MI300A / MI300X",
     "MEASURED 2026-09-24, boundary-safe, from the tree root: 13 files. The sentence \"no file "
     "in this tree names it\" is FALSE, and the counter-example is in THIS TREE'S OWN GATE: "
     "src/core/amdsafe_gate.h:227, the sdot8 row, reads \"CDNA3 (gfx942) only\". It is ALSO named "
     "at src/core/amdsafe_gate.h:405 (an artefact called nvfp4_gfx1201.hsaco and its load paths). "
     "Corrected rather than softened, because a rung's evidence field is printed by "
     "render_amd_format_refusal() and a false negative there reaches an operator.", "",
     "same as gfx90a. Current-generation and therefore the rung most tempting to assume, which "
     "is exactly why it says what it says: this tree contains no gfx942 kernel and no gfx942 "
     "probe, so nothing here establishes anything about it."},
    {"gfx1100", "RDNA3", "RX 7900 XTX / W7900",
     "MEASURED 2026-09-24, boundary-safe, from the tree root: 12 files. Under src/ the count is 3 "
     "and every one of them is this table or a section of this same file -- so for THIS rung the "
     "sentence \"no file in this tree names it\" holds for src/ and is FALSE for the tree as a "
     "whole. Both denominators are printed rather than the single \"0 files\" this row used to "
     "carry, because \"0 files\" was a reading of one directory presented as a reading of the "
     "tree.", "",
     "same refusal. RDNA's matrix path is v_wmma_*, NOT v_mfma_*, so even a kernel re-authored "
     "for gfx90a would not serve this rung -- and that difference is a fact about the AMD side, "
     "so it is labelled EXTERNAL-UNPROBED in kPtxFamilyAmdStatus rather than asserted here."},
    {"gfx1201", "RDNA4", "RX 9070 / RX 9070 XT",
     "MEASURED 2026-09-24, boundary-safe, from the tree root: 13 files (14 by a raw fixed-string "
     "grep, the difference being occurrences inside longer identifiers such as "
     "nvfp4_gfx1201.hsaco). Under src/ the count is 4 and three of them are NOT this table: "
     "src/ops/common/math.cuh:46 (a measured line: an instruction \"encodes on gfx900..gfx1201\"), "
     "src/core/amdsafe_gate.h:405 (the artefact named nvfp4_gfx1201.hsaco), and "
     "src/core/vendor_sim.h:915. So \"appears NOWHERE in this tree (0 files)\" and \"no file in "
     "this tree names it\" were BOTH false, and they are corrected rather than softened: the two "
     "denominators are printed, and the three named src/ sites are the reading.", "",
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
     "config-selectable WMMA entry (set_smem_opt). "
     "⚠⚠ THIS NOTE WAS WRONG AND IS CORRECTED, MEASURED 2026-09-24. It said \"this file is the "
     "QPN family, which this build does not contain (0 hits for `qpn` in src/CMakeLists.txt)\", and "
     "every clause of that is now false: `grep -ci qpn src/CMakeLists.txt` = 35; "
     "src/CMakeLists.txt:366 lists ops/linear/qpn/qpn_host.cu and :371 lists "
     "ops/linear/qpn/qpn_arch_route.cpp in ninfer_ops' EXPLICIT source list (no GLOB); "
     "qpn_host.cu:6 includes THIS FILE, and build/src/CMakeFiles/ninfer_ops.dir/ops/linear/qpn/"
     "qpn_host.cu.o.d carries the dependency; and `nm -C build/apps/ninfer | grep -c qpn` = 54. "
     "src/CMakeLists.txt:339-342 records the fix that made it false (\"The QPN family existed in "
     "the tree and was compiled by NO target: `qpn` had 0 hits in this file\") -- so this note kept "
     "a reading that the build file itself had already overtaken. The site is therefore a limit on "
     "a route that IS in the build and IS linked into the binary, NOT on one \"already refused for "
     "a different reason\"; qpn_host.cu:82 says so in the tree's own words (\"...IT IS IN THIS "
     "BUILD\"). Recorded rather than dropped, and corrected rather than softened."},
    {"src/targets/qwen3_8_flash_next/impl/moe_kernels.cu:1989", "69312 B = 67.7 KB", "69312",
     "flash_next_moe_prefill_gate_up_mma_kernel<false>, also launched with 69312 bytes of "
     "dynamic shared memory at moe_kernels.cu:2007. "
     "⚠ \"This target IS in the build\" WAS TOO BROAD AND IS CORRECTED, MEASURED 2026-09-24. What "
     "is true: the file is listed at src/targets/qwen3_8_flash_next/CMakeLists.txt:16, and that "
     "directory is added at src/CMakeLists.txt:714 as `add_subdirectory(targets/"
     "qwen3_8_flash_next EXCLUDE_FROM_ALL)`. What is ALSO true and was missing: EXCLUDE_FROM_ALL "
     "means it is not part of the default build, ninfer_engine's own link line "
     "(src/CMakeLists.txt:716-722) names ninfer_artifact ninfer_core ninfer_ops ninfer_text "
     "ninfer_media_decode and NOT this target, and the reading that settles it is "
     "`nm -C build/apps/ninfer | grep -c flash_next_moe_prefill_gate_up_mma_kernel` = 0 while the "
     "same command on `qpn` returns 54 -- so the symbol IS in the archive "
     "(libninfer_qwen3_8_flash_next.a) and NOT in the engine binary. The correct sentence is "
     "\"this target is DECLARED in the build system and COMPILED on demand, and it is linked into "
     "no engine binary\", which is a weaker claim than the one that stood here."},
    {"src/targets/qwen3_8_flash_next/impl/moe_kernels.cu:1991", "92416 B = 90.25 KB", "92416",
     "flash_next_moe_prefill_gate_up_mma_kernel<true>, also launched with 92416 bytes at "
     "moe_kernels.cu:2024. Same correction as the row above, for the same measurement: declared "
     "and compilable, EXCLUDE_FROM_ALL, and NOT linked into ninfer."},
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
