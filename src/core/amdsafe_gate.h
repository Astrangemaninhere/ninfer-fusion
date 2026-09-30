#ifndef NINFER_CORE_AMDSAFE_GATE_H
#define NINFER_CORE_AMDSAFE_GATE_H

// ---------------------------------------------------------------------------
// THE LOAD-TIME BY-NAME GATE (dl/amdsafe, deliverable 3).
//
// WHAT IT IS FOR. dl/hipshim/REPORT.md section 3 is a per-primitive table with three verdicts:
// MAPPED-EXACT (a CUDA expression computes the same function), MAPPED-DEVIATING (plausible, it
// runs, the value or the cost differs), REFUSED (no faithful meaning exists). That table is a
// READING. This header is the same table turned into a GATE, so that a load path which would
// have to lean on a deviating or unmappable primitive is stopped at LOAD TIME WITH THE
// PRIMITIVE'S NAME, rather than producing numbers nobody can tell are wrong.
//
// ---------------------------------------------------------------------------
// SCOPE -- ONE AXIS, AND IT IS THE WHOLE POINT. READ THIS BEFORE CHANGING ANYTHING HERE.
//
// THE GATE APPLIES TO **GPU-SIMULATION** LOAD PATHS ONLY. A path that declares
// AmdSafeExecTarget::HostCpu is ADMITTED UNCONDITIONALLY, WITHOUT EXAMINING ITS PRIMITIVE LIST
// AT ALL, and the decision it returns says so in words. Two reasons, and the second is the
// load-bearing one:
//
//   1. On a host path the primitive is IMPLEMENTED ON THE HOST. There is no mapping from AMD
//      semantics to CUDA semantics to be wrong, so there is nothing for this gate to check.
//      A gate that refused a host path would be refusing a situation it has no evidence about.
//   2. The product requirement is that the artifact RUNS ON THE TARGET THE OPERATOR HAS, and a
//      host execution path is what makes that true on a machine with no usable GPU. A gate that
//      can refuse the host path can therefore take that requirement away -- so the gate is not
//      allowed to. This was a CORRECTION received while this header was being written, and it
//      is recorded by name here rather than silently folded in: the FIRST form of this gate
//      keyed on "the critical path needs primitive P" with no target axis, which would have
//      refused the host path for exactly the reason the host path exists.
//
// The reverse direction is where the gate earns its keep: on a GPU-simulation path a deviating
// primitive is a WRONG NUMBER with no diagnostic, which is the failure class this whole line
// exists to make unreachable.
//
// ---------------------------------------------------------------------------
// WHAT MAKES IT A GATE RATHER THAN A TABLE. The primitive verdicts alone are not enough: an
// unlisted primitive, or a path nobody declared, must also refuse, and must refuse BY NAME.
// Otherwise the gate's silence means "I did not look" and is read as "it is fine" -- the defect
// dl/hipshim records for the blocked-codepoint checker that scanned zero files and printed its
// pass line, and the defect dl/cufree names with "a report that says 'the probe answered X'
// while the route is X's opposite is the same lie in a smaller font".
//
// So there are FOUR refusal reasons, each with its own token:
//   gpu-path-needs-deviating-primitive
//   gpu-path-needs-unmappable-primitive
//   gpu-path-uses-unlisted-primitive
//   path-not-declared
// and one admission reason:
//   host-path-primitives-implemented-on-host      (no primitive check performed)
//   gpu-path-mapped-exact-only
//
// Header-only, std-only, every function `inline`, NO out-of-line symbol, so it costs no link
// edge and can be called from a loader with nothing behind it -- the property
// src/core/cufree_report.h:19-32 measures, copied here rather than re-derived.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::caps {

// ---------------------------------------------------------------------------
// The verdict vocabulary. It is dl/hipshim/REPORT.md section 3's own three words and it is
// spelled the same on purpose: two vocabularies for one fact is how a tree starts disagreeing
// with itself.
// ---------------------------------------------------------------------------
enum class AmdSafeVerdict : std::uint8_t {
    // A CUDA/host expression computes the SAME function. Safe on any path.
    MappedExact = 0,
    // Plausible, it runs, and the value or the cost differs. Safe on NOTHING that needs the AMD
    // value; refused by name on a GPU-simulation path.
    MappedDeviating = 1,
    // No faithful meaning exists at all. Refused by name on a GPU-simulation path.
    Unmappable = 2,
};

[[nodiscard]] inline const char* amdsafe_verdict_token(AmdSafeVerdict v) {
    switch (v) {
    case AmdSafeVerdict::MappedExact:     return "mapped-exact";
    case AmdSafeVerdict::MappedDeviating: return "mapped-deviating";
    case AmdSafeVerdict::Unmappable:      return "unmappable";
    }
    return "?";
}

// WHERE the path executes. This axis is NOT a performance knob -- it selects whether the gate
// has anything to check at all. See the header comment.
enum class AmdSafeExecTarget : std::uint8_t {
    // The primitives are implemented directly on the host. THE GATE DOES NOT APPLY.
    HostCpu = 0,
    // The AMD semantics are being reproduced on a GPU. The gate applies in full.
    GpuSim = 1,
};

[[nodiscard]] inline const char* amdsafe_target_token(AmdSafeExecTarget t) {
    switch (t) {
    case AmdSafeExecTarget::HostCpu: return "host-cpu";
    case AmdSafeExecTarget::GpuSim:  return "gpu-sim";
    }
    return "?";
}

// One row per primitive. `layer` is the six-layer number from dl/amdsafe/REPORT.md so a refusal
// can point at the layer it fires in; `why` is the deviation or the missing thing IN WORDS.
struct AmdSafePrimitive {
    std::string_view name;
    AmdSafeVerdict   verdict;
    std::string_view layer;
    std::string_view why;
};

// ---------------------------------------------------------------------------
// THE PRIMITIVE TABLE. Every row is a row of dl/hipshim/REPORT.md section 3; the verdicts are
// copied, not re-decided, and the `why` text states the deviation rather than pointing at it.
//
// THE TABLE IS A FUNCTION-LOCAL STATIC BEHIND AN `inline` FUNCTION, and that is a CORRECTNESS
// requirement rather than a style: `amdsafe_find_primitive()` hands out a POINTER into it, and
// the FIRST form of this header returned the table BY VALUE, so that pointer pointed into a
// temporary destroyed at the end of the same full expression. The symptom was not a crash at the
// point of the bug -- it was `std::bad_alloc` thrown from the renderer, which then appended a
// `string_view` with a garbage length. An ODR-guaranteed single instance across every TU, the
// same technique src/core/cufree_report.h:252-255 uses for `cufree_class_announce_once()`.
// ---------------------------------------------------------------------------
[[nodiscard]] inline const std::vector<AmdSafePrimitive>& amdsafe_primitive_table() {
    static const std::vector<AmdSafePrimitive> kTable = {
        // ---- L4: MAPPED-EXACT. The port rests on these, and hipshim measured them. ----
        {"shfl_xor_width32", AmdSafeVerdict::MappedExact, "L4",
         "a 32-lane-group shuffle with width=32 IS the AMD intra-group exchange; hipshim T12/T12b "
         "measured n_diff=0 on all 64 lanes in both arms"},
        {"shfl_up_width32", AmdSafeVerdict::MappedExact, "L4",
         "width=32 confines the source to the caller's own 32-lane group, which is the AMD "
         "group boundary"},
        {"shfl_down_width32", AmdSafeVerdict::MappedExact, "L4",
         "same as shfl_up_width32"},
        {"amd_dpp_quad_perm_0xB1", AmdSafeVerdict::MappedExact, "L4",
         "quad_perm(1,0,3,2) is lane^1 inside a 4-lane quad, i.e. intra-group"},
        {"amd_dpp_quad_perm_0x4E", AmdSafeVerdict::MappedExact, "L4",
         "quad_perm(2,3,0,1) is lane^2 inside a quad"},
        {"amd_dpp_row_ror_0x124", AmdSafeVerdict::MappedExact, "L4",
         "row_ror:4 is a ROTATE, not an xor: lane reads row_base | ((lane-4) & 15); spelled out "
         "because lane^4 differs for the four wrapping lane pairs"},
        {"amd_dpp_row_ror_0x128", AmdSafeVerdict::MappedExact, "L4",
         "row_ror:8, same shape as 0x124"},
        {"amd_ds_swizzle_0x401F", AmdSafeVerdict::MappedExact, "L4",
         "bit-mode swizzle (lane & 0x1f) ^ 0x10 NEVER crosses a 32-lane group, which is the "
         "property the DPP ladder depends on"},
        {"amd_sdot4", AmdSafeVerdict::MappedExact, "L4",
         "signed 4-way int8 dot into int32 is the same integer function as __dp4a; no rounding "
         "question exists"},
        {"amd_udot4", AmdSafeVerdict::MappedExact, "L4",
         "unsigned 4-way int8 dot; same as amd_sdot4"},
        {"threadfence_block", AmdSafeVerdict::MappedExact, "L4",
         "block-scoped ordering coincides on both sides for the uses in this tree"},
        {"threadfence_device", AmdSafeVerdict::MappedExact, "L4",
         "device-scoped ordering coincides on both sides"},

        // ---- L4: MAPPED-DEVIATING. Refused on a GPU-simulation path, by name. ----
        {"shfl_xor_default_width", AmdSafeVerdict::MappedDeviating, "L4",
         "with the wave64 default width a delta of 32 EXCHANGES THE TWO HALVES on AMD, while a "
         "32-bit warp shuffle treats laneMask >= width as out of range and returns the caller's "
         "own value; hipshim T1 n_diff=64, worst 8.125, and nothing warns"},
        {"shfl_down_default_width", AmdSafeVerdict::MappedDeviating, "L4",
         "hipshim T2: n_diff=1, LANE 31 ONLY. One lane in sixty-four, which is exactly why this "
         "class survives review"},
        {"shfl_up_default_width", AmdSafeVerdict::MappedDeviating, "L4",
         "same row as shfl_down_default_width"},
        {"shfl_srclane_ge32", AmdSafeVerdict::MappedDeviating, "L4",
         "AMD reads wave-lane 40; a 32-bit shuffle reads 40 & 31 = 8 of the same warp -- a real "
         "value from the WRONG lane, hipshim T3 n_diff=32"},
        {"ballot_wave64", AmdSafeVerdict::MappedDeviating, "L4",
         "the upper-half bits do not exist in a 32-bit ballot: hipshim T4 popc 8 vs 0, T5 popc 64 "
         "vs 32. A 64-bit ballot has no CUDA spelling at all"},
        {"any_all_wave64", AmdSafeVerdict::MappedDeviating, "L4",
         "AMD's domain is the wave, CUDA's is the warp; hipshim T6 n_diff=32"},
        {"activemask_wave_boundary", AmdSafeVerdict::MappedDeviating, "L4",
         "hipshim T7b: with the divergence ON the warp boundary the union is 0xFFFFFFFFFFFFFFFF "
         "(popc 64) against AMD 0xFFFFFFFF (popc 32) -- a STRICT SUPERSET, because two NVIDIA "
         "warps are concurrently live where one wave64 cannot be"},
        {"prefix_scan_64", AmdSafeVerdict::MappedDeviating, "L4",
         "a 32-lane scan restarts at wave-lane 32: hipshim T9 n_diff=32, first_lane=32, worst "
         "156.625"},
        {"ds_bpermute_cross_half", AmdSafeVerdict::MappedDeviating, "L4",
         "correct ONLY when the target lane is inside the caller's own 32-lane group; ds_bpermute "
         "addresses the whole wave64 and an NVIDIA shuffle cannot reach the other half"},
        {"wave_barrier", AmdSafeVerdict::MappedDeviating, "L4",
         "wave scope (64) against warp scope (32): identical under full convergence, different "
         "under divergence, and a translator cannot tell the two apart"},
        {"s_memtime", AmdSafeVerdict::MappedDeviating, "L4",
         "clock64() counts SM cycles per SM; the AMD counter is free-running, so a rate converted "
         "with an assumed AMD tick is wrong by a factor nobody can compute from here"},
        {"s_waitcnt", AmdSafeVerdict::MappedDeviating, "L4",
         "the ORDER is the nearest available; the AMD model is explicitly tracked and NVIDIA's is "
         "implicit, so s_waitcnt used as a performance control has no representation"},
        {"v_dot2_f32_f16", AmdSafeVerdict::MappedDeviating, "L4",
         "an fmaf composition is mathematically equal and NOT bit-identical: the hardware folds "
         "two partial products into the fp32 accumulator, the composition rounds each product to "
         "fp32 first"},
        {"bf16_rounding_mode", AmdSafeVerdict::MappedDeviating, "L4",
         "old ROCm __float2bfloat16 TRUNCATED where CUDA rounds to nearest even; a kernel that "
         "depended on truncation produces different bf16 values here"},
        {"threadfence_system", AmdSafeVerdict::MappedDeviating, "L4",
         "ROCm documents weaker guarantees than CUDA for the same spelling"},

        // ---- L3/L5: UNMAPPABLE. Refused on a GPU-simulation path, by name. ----
        {"ldmatrix", AmdSafeVerdict::Unmappable, "L3",
         "NO CUDA equivalent exists. 245 sites in 59 files; the fragment layouts must be "
         "regenerated, which is a re-authoring job"},
        {"mma_sync_family", AmdSafeVerdict::Unmappable, "L3",
         "the AMD side is v_mfma_* / v_wmma_* with a DIFFERENT fragment-to-lane layout and a "
         "64-lane wavefront; a mechanical substitution compiles, produces the right 16x16 SHAPE, "
         "and the numbers belong to different lanes"},
        {"v_wmma_16x16x16_f16", AmdSafeVerdict::Unmappable, "L3",
         "the SHAPE names collide with wmma m16n16k16 and the LAYOUTS do not: an NVIDIA "
         "accumulator is spread over 32 lanes of a warp, an AMD one over 64 lanes of a wave"},
        {"mbarrier", AmdSafeVerdict::Unmappable, "L5",
         "AMD's s_barrier is a single workgroup barrier and carries no transaction count. 4 sites"},
        {"cp_async", AmdSafeVerdict::Unmappable, "L5",
         "no cp.async on AMD; the portable shape is a direct-to-LDS load with s_waitcnt. 12 sites"},
        {"setmaxnreg", AmdSafeVerdict::Unmappable, "L5",
         "the instruction exists to reallocate registers between warp-specialised roles inside "
         "one CTA; no AMD analogue. 7 sites"},
        {"sdot8", AmdSafeVerdict::Unmappable, "L3",
         "CDNA3 (gfx942) only; the CUDA composition is TWO __dp4a, so the obstacle is the "
         "instruction count and register demand, not arithmetic"},
        {"cooperative_groups_grid", AmdSafeVerdict::Unmappable, "L3",
         "grid_group has no AMD counterpart; only thread_block is mapped"},
        {"lds_over_64k", AmdSafeVerdict::Unmappable, "L5",
         "LDS is 64 KiB per CU on gfx906 against 100-228 KiB per SM: ONE-DIRECTIONAL. A shape "
         "needing 72 KiB per block launches here with rc=0 and cannot run there"},
    };
    return kTable;
}

[[nodiscard]] inline const AmdSafePrimitive* amdsafe_find_primitive(std::string_view name) {
    for (const AmdSafePrimitive& p : amdsafe_primitive_table()) {
        if (p.name == name) { return &p; }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// THE DECLARED MANIFEST: which load path requires which primitives, and WHERE IT RUNS.
//
// A path that is not in this manifest is REFUSED (reason `path-not-declared`), never assumed.
// That is the whole difference between a gate and a table: an undeclared path must not be
// silently fine, or the gate's silence stops meaning anything.
//
// Static behind an `inline` function for the same reason as the primitive table above.
// ---------------------------------------------------------------------------
struct AmdSafePathDeclaration {
    std::string_view              path;
    AmdSafeExecTarget             target;
    std::vector<std::string_view> primitives;
    std::string_view              note;
};

[[nodiscard]] inline const std::vector<AmdSafePathDeclaration>& amdsafe_manifest() {
    static const std::vector<AmdSafePathDeclaration> kManifest = {
        // ---- the GREEN control: a GPU-simulation path that needs mapped-exact primitives only.
        // This is the gfx906_reduce_sum32 DPP ladder, the single call site the port rests on.
        {"gfx906_reduce_sum32_ladder", AmdSafeExecTarget::GpuSim,
         {"amd_dpp_quad_perm_0xB1", "amd_dpp_quad_perm_0x4E", "amd_dpp_row_ror_0x124",
          "amd_dpp_row_ror_0x128", "amd_ds_swizzle_0x401F"},
         "the tree's own DPP ladder (src/ops/common/warp.cuh:105-120); hipshim T11 measured it "
         "bit-exact on one input, and the two addition orders are NOT guaranteed bit-equal in "
         "general"},

        // ---- the GREEN control for the second suite: width<=32 shuffles only.
        {"port_32lane_subgroup_shuffles", AmdSafeExecTarget::GpuSim,
         {"shfl_xor_width32", "shfl_up_width32", "shfl_down_width32"},
         "ninfer's 32-lane-subgroup call sites (kFullWarpMask); T12/T12b measured them equal in "
         "both arms"},

        // ---- the RED control: a GPU-simulation path that needs a DEVIATING primitive by name.
        {"wave64_ballot_reduction", AmdSafeExecTarget::GpuSim,
         {"ballot_wave64", "shfl_xor_default_width", "prefix_scan_64"},
         "a 64-lane ballot reduction; every one of its three primitives is mapped-deviating"},

        // ---- the RED control for the unmappable class.
        {"bf16_gemm_mma_channel", AmdSafeExecTarget::GpuSim,
         {"mma_sync_family", "ldmatrix"},
         "the tree's tensor-core BF16 channel: src/ops/linear/bf16/bf16_gemm_mma.cuh:296 mma_bf16( "
         "and :266 ldmatrix_x4("},

        // ---- the RED control for an undeclared-primitive name.
        {"experimental_wave64_reduce_v2", AmdSafeExecTarget::GpuSim,
         {"ballot_wave64", "wave64_reduce_butterfly_v2"},
         "the SECOND name has no row in the primitive table -- the gate must refuse THAT name "
         "rather than skipping it"},

        // ---- GREEN-2 (the correction's own control): THE SAME 64-LANE SEMANTICS ON THE HOST.
        {"wave64_ballot_reduction_host_model", AmdSafeExecTarget::HostCpu,
         {"ballot_wave64", "shfl_xor_default_width", "prefix_scan_64", "ldmatrix", "mbarrier"},
         "the same primitive NAMES as `wave64_ballot_reduction`, on the host path. It must be "
         "ADMITTED, and its primitive list must not even be examined -- a gate that can refuse the "
         "host path can take away the requirement that the artifact runs at all"},

        {"always_host_reference_model", AmdSafeExecTarget::HostCpu, {},
         "a host reference with no primitive list at all: admitted, and the decision says no "
         "primitive check was performed"},
    };
    return kManifest;
}

// ---------------------------------------------------------------------------
// The decision. `refusals` carries the NAMES, in manifest order, for every refused primitive.
// ---------------------------------------------------------------------------
struct AmdSafeGateDecision {
    std::string              path;              // copied: the decision owns its subject
    AmdSafeExecTarget        target = AmdSafeExecTarget::GpuSim;
    bool                     declared = false;  // was this path in the manifest at all?
    bool                     admitted = false;
    bool                     primitive_check_performed = false;
    std::string              reason_token;      // one of the tokens named in the header comment
    std::vector<std::string> refused;           // primitive names, manifest order
    std::vector<std::string> deviating;         // admitted-but-named (never on GpuSim today)
    std::vector<std::string> unlisted;          // names with no primitive-table row
};

// THE GATE. Pure function of the manifest and the primitive table.
[[nodiscard]] inline AmdSafeGateDecision amdsafe_gate(std::string_view path) {
    AmdSafeGateDecision d;
    d.path.assign(path.begin(), path.end());

    const AmdSafePathDeclaration* decl = nullptr;
    for (const AmdSafePathDeclaration& row : amdsafe_manifest()) {
        if (row.path == path) { decl = &row; break; }
    }
    if (decl == nullptr) {
        // NOT-DECLARED IS NOT A PASS. A path nobody declared is exactly the path whose
        // primitives nobody checked, so silence here would be read as "it is fine".
        d.declared = false;
        d.admitted = false;
        d.reason_token = "path-not-declared";
        return d;
    }
    d.declared = true;
    d.target = decl->target;

    if (decl->target == AmdSafeExecTarget::HostCpu) {
        // THE CORRECTION, EXECUTED. The primitive list is NOT walked, so no primitive name can
        // refuse a host path even by accident.
        d.admitted = true;
        d.primitive_check_performed = false;
        d.reason_token = "host-path-primitives-implemented-on-host";
        return d;
    }

    d.primitive_check_performed = true;
    for (const std::string_view name : decl->primitives) {
        const AmdSafePrimitive* p = amdsafe_find_primitive(name);
        if (p == nullptr) {
            d.unlisted.emplace_back(name);
            d.refused.emplace_back(name);
            continue;
        }
        switch (p->verdict) {
        case AmdSafeVerdict::MappedExact:
            break;
        case AmdSafeVerdict::MappedDeviating:
            d.deviating.emplace_back(name);
            d.refused.emplace_back(name);
            break;
        case AmdSafeVerdict::Unmappable:
            d.refused.emplace_back(name);
            break;
        }
    }

    if (!d.unlisted.empty()) {
        d.admitted = false;
        d.reason_token = "gpu-path-uses-unlisted-primitive";
    } else if (!d.refused.empty()) {
        d.admitted = false;
        d.reason_token = d.deviating.empty() ? "gpu-path-needs-unmappable-primitive"
                                             : "gpu-path-needs-deviating-primitive";
    } else {
        d.admitted = true;
        d.reason_token = "gpu-path-mapped-exact-only";
    }
    return d;
}

// Operator-facing text. Empty when admitted -- so a caller may print it unconditionally in the
// same shape render_amd_format_refusal() uses, and a clean load prints nothing.
[[nodiscard]] inline std::string render_amdsafe_gate_refusal(const AmdSafeGateDecision& d) {
    if (d.admitted) { return {}; }
    std::string out;
    out += "ninfer: REFUSED AT LOAD by the amdsafe gate -- path `";
    out.append(d.path);
    out += "`\n  reason : ";
    out.append(d.reason_token);
    out += "\n  target : ";
    out.append(amdsafe_target_token(d.target));
    out += "\n";
    // ---------------------------------------------------------------------------
    // ROUND 6: THE EARLY RETURN THAT USED TO SIT HERE IS GONE, AND THE REASON IS MEASURED.
    // ---------------------------------------------------------------------------
    // It read `... Declare it WITH ITS TARGET and its primitive list.` then `return out;`, so an
    // UNDECLARED path never reached the "WHAT TO DO NEXT" block below. Round 5 measured the real
    // artefact (nvfp4_gfx1201.hsaco): EVERY load path derived from its own .symtab symbols is
    // undeclared, so 8 of 9 refusals reached the operator with their remedy buried in prose and
    // only the single DECLARED one carried it under the heading -- the one class that needed the
    // heading least. The remedy MOVED into that block rather than being copied into this branch:
    // one sentence with two homes is how two renderings start disagreeing, and this header's own
    // comment says the reason it exists is that a refusal which is not loud is not a refusal.
    if (!d.declared) {
        out += "  This path is NOT IN THE MANIFEST (src/core/amdsafe_gate.h, amdsafe_manifest()).\n";
        out += "  A path nobody declared is the path whose primitives nobody checked, so it is\n";
        out += "  refused rather than assumed.\n";
    } else {
        out += "  WHAT IS MISSING IS NOT A DRIVER AND NOT A CARD: this is a GPU-simulation path that\n";
    out += "  would have to lean on an AMD primitive which has no faithful CUDA/host meaning, so\n";
    out += "  it would run and produce numbers nobody can tell are wrong. Each name below is\n";
    out += "  refused BY NAME, with the deviation stated:\n";
    for (const std::string& name : d.refused) {
        const AmdSafePrimitive* p = amdsafe_find_primitive(name);
        out += "    * ";
        out.append(name);
        if (p == nullptr) {
            out += "  [NO ROW IN THE PRIMITIVE TABLE -- unlisted, refused on that ground alone]\n";
            continue;
        }
        out += "  [";
        out.append(amdsafe_verdict_token(p->verdict));
        out += ", ";
        out.append(p->layer);
        out += "]\n      ";
        out.append(p->why);
        out += "\n";
    }
    // Closed here so that BOTH branches above fall through to ONE next-step block. The heading
    // and the three declared-path items are byte-for-byte what they were; only the undeclared
    // branch gains an item, because it had none under a heading before.
    }
    out += "  WHAT TO DO NEXT, and this is a real choice rather than a shrug:\n";
    if (!d.declared) {
        out += "    * declare this path WITH ITS TARGET and its primitive list, in "
               "amdsafe_manifest()\n";
        out += "      (src/core/amdsafe_gate.h). An undeclared path is refused on that ground "
               "ALONE, so\n";
        out += "      nothing above says the path would be clean once declared -- declaring it "
               "buys it a\n";
        out += "      PRIMITIVE CHECK, not an admission, and the check can still refuse it by "
               "name.\n";
    } else {
    out += "    * run this path on the HOST-CPU execution target. A host path is ADMITTED "
           "UNCONDITIONALLY by\n";
    out += "      this gate -- its primitives are implemented on the host, so there is no mapping "
           "to be wrong\n";
    out += "      -- and it is the target that makes the artifact run on a machine with no usable "
           "GPU AT ALL.\n";
    out += "    * or re-author the kernel so the path needs only mapped-exact primitives. That is "
           "what\n";
    out += "      dl/hipshim/REPORT.md section 3 already did for the 32-lane-subgroup call sites.\n";
    out += "    * a knob that turns this refusal off DOES NOT EXIST in this header, deliberately: "
           "and\n";
    out += "      \"nothing to refuse\" would then be indistinguishable from \"the refusal was "
           "switched\n";
    out += "      off\", which is the defect src/core/cufree_report.h:241-250 names.\n";
    }
    return out;
}

// ---------------------------------------------------------------------------
// THE ONE LINE A LOADER CALLS. Shape taken from dl/sparktarget's by-name refusal at the
// runtime-view fill (its item 0006, `bindings.cpp`): ONE call in the load path, taking the path
// name, returning whether the load may proceed -- so no load path can quietly skip the gate and
// no call site can re-implement the policy.
//
// Returns true when the load may proceed. When it returns false, `render_amdsafe_gate_refusal()`
// on the same decision is the text to print; the caller is expected to abort the load, not to
// continue and hope.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool amdsafe_gate_allows_load(std::string_view path) {
    return amdsafe_gate(path).admitted;
}

} // namespace ninfer::caps

#endif // NINFER_CORE_AMDSAFE_GATE_H
