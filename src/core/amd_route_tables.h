#ifndef NINFER_CORE_AMD_ROUTE_TABLES_H
#define NINFER_CORE_AMD_ROUTE_TABLES_H

// ---------------------------------------------------------------------------
// THE AMD FORMAT-ROUTE DESCENT, FED FROM THIS TREE'S OWN TABLES (dl/amdprobe).
//
// WHY THIS IS A SEPARATE HEADER FROM src/core/amd_route.h, and the split is a MEASURED property
// rather than a preference. This header includes src/core/arch_caps.h, which is the home of every
// table the descent consumes. arch_caps.h is NOT link-free: cufree_report.h:19-32 measured that a
// TU which includes arch_caps.h and calls render_build_capability_surface() does not link without
// ninfer_artifact, because ninfer::artifact::format_name(NumericFormat) is declared there and
// defined out of line -- and linking ninfer_artifact pulls ninfer_core, which carries
// target_link_libraries(ninfer_core PUBLIC CUDA::cudart ...).
//
// So the DECISION lives in amd_route.h, which includes only the standard library, and the TABLES
// live here. The same property that lets amdsafe_hip_stack.h be heard on a machine where ld.so
// kills every other front door therefore belongs to the descent itself, and this header is the
// half that cannot have it. A caller that only needs to REFUSE does not need this header; a
// caller that needs to decide needs it, and knows what it links.
//
// MEASURED 2026-09-24 (dl/amdprobe/logs/11_build_probe.log): arch_caps.h DOES compile and link
// under plain g++ 15.2 with -std=c++20 and no CUDA, and the resulting binary records neither
// libcudart nor libamdhip64 in DT_NEEDED (only libc.so.6). So the table half of this descent is
// exercisable on this box, by a compiler with no toolchain for either vendor.

#include "core/amd_route.h"
#include "core/arch_caps.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::caps {

// ---------------------------------------------------------------------------
// THE MATRIX PATH OF A RUNG, as a NAME plus its PROVENANCE.
//
// It is a three-state value and not a bool, because the lowest rung of kAmdLadder has NO matrix
// core at all: gfx906 is GCN5.1, which predates v_mfma_*, and that is not an inference of ours --
// kAmdLadder's own gfx90a row says gfx908 "is the first rung that WOULD exercise v_mfma_*". A
// bool would have to call gfx906 one of the two, and either answer would offer that rung a matrix
// path it does not have.
//
// EVERY ROW IS EXTERNAL-UNPROBED, spelled the way kPtxFamilyAmdStatus spells it, for the same
// reason: this box has no AMD card and no ROCm toolchain, so none of these three names has been
// measured here. `measured_in_this_tree` is therefore false on all of them and there is no
// encoding in the table for "true" -- the field exists so that a later hardware probe has one
// place to make the claim, not so that this file can.
// ---------------------------------------------------------------------------
enum class AmdMatrixPath : std::uint8_t {
    // No matrix core on this rung. The descent must not offer it v_mfma_* or v_wmma_*.
    NoMatrixCore = 0,
    // RDNA's matrix path. kAmdLadder's gfx1100 row: "RDNA's matrix path is v_wmma_*, NOT
    // v_mfma_*, so even a kernel re-authored for gfx90a would not serve this rung".
    Wmma,
    // CDNA's matrix path.
    Mfma,
};

[[nodiscard]] inline std::string_view amd_matrix_path_name(AmdMatrixPath p) noexcept {
    switch (p) {
    case AmdMatrixPath::NoMatrixCore: return "no-matrix-core";
    case AmdMatrixPath::Wmma:         return "v_wmma";
    case AmdMatrixPath::Mfma:         return "v_mfma";
    }
    return "?";
}

struct AmdMatrixPathRow {
    std::string_view target;
    AmdMatrixPath    path;
    bool             measured_in_this_tree; // ALWAYS false here -- see the enum's comment
    std::string_view provenance;
};

// ONE ROW PER RUNG OF kAmdLadder, and the count is checked against kAmdLadderSize by a
// static_assert below: a rung added to the ladder without a matrix path here would otherwise be
// offered a matrix level by the default, which is the phantom this table exists to prevent.
inline constexpr AmdMatrixPathRow kAmdMatrixPaths[] = {
    {"gfx906", AmdMatrixPath::NoMatrixCore, false,
     "EXTERNAL-UNPROBED, and cited from THIS TREE rather than from a vendor doc: kAmdLadder's own "
     "gfx908 row says gfx908 'is the first rung that WOULD exercise v_mfma_*', so GCN5.1 has no "
     "matrix core. gfx906's status as the lowest rung and its LDS hazard are recorded there too."},
    {"gfx908", AmdMatrixPath::Mfma, false,
     "EXTERNAL-UNPROBED. CDNA1. kAmdLadder's gfx908 row names it as the first v_mfma_* rung, and "
     "that row also records that NO kernel in this tree emits v_mfma_*, so the answer there is a "
     "refusal for a different reason: the missing piece is a re-authored kernel."},
    {"gfx90a", AmdMatrixPath::Mfma, false,
     "EXTERNAL-UNPROBED. CDNA2. kAmdLadder's gfx90a row: 'the first rung that WOULD exercise "
     "v_mfma_* -- i.e. it is the rung that would answer the mma.sync blocker'."},
    {"gfx942", AmdMatrixPath::Mfma, false,
     "EXTERNAL-UNPROBED. CDNA3. kAmdLadder's gfx942 row cites src/core/amdsafe_gate.h (the sdot8 "
     "row, 'CDNA3 (gfx942) only') and records that this tree contains no gfx942 kernel and no "
     "gfx942 probe."},
    {"gfx1100", AmdMatrixPath::Wmma, false,
     "EXTERNAL-UNPROBED. RDNA3. kAmdLadder's gfx1100 row states the v_wmma/v_mfma split by name."},
    {"gfx1201", AmdMatrixPath::Wmma, false,
     "EXTERNAL-UNPROBED. RDNA4. Same split as gfx1100; kAmdLadder's gfx1201 row adds that this "
     "row's ISA label is 'a NAME and not a capability'."},
};

inline constexpr std::size_t kAmdMatrixPathCount =
    sizeof(kAmdMatrixPaths) / sizeof(kAmdMatrixPaths[0]);

// The rung -> matrix path lookup. NOTHING IS ASSUMED FOR AN UNLISTED TARGET: it returns
// NoMatrixCore together with a flag, so an unlisted rung is offered no matrix level rather than a
// neighbour's -- the rule kAmdLadder's own comment sets ("Do NOT copy the neighbouring row's
// lowering").
[[nodiscard]] inline std::pair<AmdMatrixPath, bool> amd_matrix_path_for(
    std::string_view target) noexcept {
    for (const AmdMatrixPathRow& row : kAmdMatrixPaths) {
        if (row.target == target) { return {row.path, true}; }
    }
    return {AmdMatrixPath::NoMatrixCore, false};
}

// The count gate: every rung of kAmdLadder must have exactly one matrix-path row, walked over the
// ladder's own range so an appended rung fails a build rather than reaching the default.
namespace detail {
constexpr bool amd_matrix_paths_cover_every_rung() noexcept {
    for (const AmdRung& rung : kAmdLadder) {
        std::size_t hits = 0;
        for (const AmdMatrixPathRow& row : kAmdMatrixPaths) {
            if (row.target == rung.target) { ++hits; }
        }
        if (hits != 1) { return false; }
    }
    return true;
}
} // namespace detail

inline constexpr bool kAmdMatrixPathsCoverEveryRung = detail::amd_matrix_paths_cover_every_rung();

static_assert(kAmdMatrixPathsCoverEveryRung,
              "every rung of kAmdLadder (src/core/arch_caps.h) needs exactly one "
              "kAmdMatrixPaths row (src/core/amd_route_tables.h), or the descent would offer a "
              "rung a matrix path nobody established. Add the row; do not delete the rung.");

// ---------------------------------------------------------------------------
// THE FORMAT'S FACTS, read off the two tables that already own them. Nothing here restates a
// floor: it reads kFormatRequirements / kUncoveredFormatNotes through the public accessors, so
// there is no second census to drift.
// ---------------------------------------------------------------------------
[[nodiscard]] inline AmdFormatFacts amd_format_facts(artifact::NumericFormat format) {
    AmdFormatFacts f;
    f.name = artifact::format_name(format);

    const FormatRequirement* requirement = format_requirement(format);
    if (requirement == nullptr) {
        // A format this tree DECLARES and has no kernel for on ANY rung. The refusal is the
        // table's, and the AMD gate must not restate it as its own.
        const UncoveredFormatNote* note = uncovered_format_note(format);
        f.declares_floor = false;
        f.floor_is_tensor_core = false;
        f.floor_name = "none (no kernel on any rung)";
        f.uncovered_reason = (note != nullptr) ? note->reason : std::string_view(
            "(no kFormatRequirements row and no kUncoveredFormatNotes row: a TABLE defect, not a "
            "GPU fact)");
        return f;
    }
    f.declares_floor = true;
    f.floor_is_tensor_core = (requirement->required != Cap::None);
    f.floor_name = f.floor_is_tensor_core ? cap_name(requirement->required)
                                          : std::string_view("none - never a tensor-core operand");
    f.floor_evidence = requirement->kernel_evidence;
    f.has_lower_floor_kernel = (requirement->fallback_required != Cap::None);

    const AmdFormatBlocker* blocker = amd_format_blocker(format);
    if (blocker != nullptr) {
        f.blocker_mma_site = blocker->mma_site;
        f.blocker_mma_mnemonic = blocker->mma_mnemonic;
        f.blocker_fragment_site = blocker->fragment_site;
        f.blocker_fragment_mnemonic = blocker->fragment_mnemonic;
    }
    return f;
}

// ---------------------------------------------------------------------------
// THE FOUR BUILD READINGS, OBSERVED. This is the part that must not be a bool and must not be
// read out of a file's own self-annotation, so it is a function that goes and looks.
//
// It takes the TREE ROOT and the BUILD DIRECTORY LIST as parameters -- the shape
// cufree_report.h's observe_cufree_witnesses() uses, and for the same reason: a test or a
// namespace run can pass a synthetic tree, and a reading taken at a root other than "/" says so.
// ---------------------------------------------------------------------------

// The source files that WOULD carry each level, named once. Each is a file this tree really has
// (measured) and each was shown to contain real kernels by the amdgcn census in
// dl/amdprobe/logs/09_amd_wiring_fixed.log.
inline constexpr std::string_view kAmdLevelSourceDotSimt = "src/ops/linear/gfx906/q4_tiled_gfx906.cu";
inline constexpr std::string_view kAmdLevelSourceHostCpu = "src/core/amdsafe_gate.h";

// IS THIS FILENAME A SOURCE ENTRY IN A TARGET'S SOURCE LIST? -- reading 2, and the reading that a
// hit-count gets wrong.
//
// MEASURED 2026-09-24: the filename gfx906_stubs.cpp has exactly ONE mention in
// src/CMakeLists.txt, and it is the PROSE of line 572 that says the file is deliberately NOT in
// the list. A counter of hits reads that sentence as a wiring. So this walks each file with a
// source-list state machine instead:
//   * a line inside target_sources/add_library/add_executable ... ')' is IN A SOURCE LIST;
//   * a line whose first non-blank character is '#' is a COMMENT, whatever list it sits in;
//   * a line whose first non-blank character is '"' is inside a quoted STRING -- a diagnostic
//     sentence, which is CODE to CMake and is still not a source entry (MEASURED: the one
//     amdsafe_gate.h mention that classified CODE is src/CMakeLists.txt:801, a string
//     continuation of a diagnostic list, and it names no source);
//   * otherwise, a line in a source list that CONTAINS the entry is a SOURCE ENTRY.
//
// THE ENTRY FORM, which is the bug this function's first version had and its positive control
// caught: a CMake source entry is written RELATIVE TO THAT CMAKELISTS' OWN DIRECTORY. So
// src/CMakeLists.txt writes `ops/linear/gfx906/q4_tiled_gfx906.cu`, and a search for
// `src/ops/linear/gfx906/q4_tiled_gfx906.cu` returns 0 -- while the same full-path search DOES
// hit once for q_gemv_gfx906.cuh, in PROSE, at src/CMakeLists.txt:559. This function therefore
// strips each CMakeLists' own directory prefix from the path before matching, and it tries BOTH
// forms so a file given either way is found.
[[nodiscard]] inline bool amd_is_source_entry_in(const std::string& cmake_file,
                                                 std::string_view source_path,
                                                 std::string* where = nullptr) {
    std::ifstream in(cmake_file);
    if (!in) { return false; }
    // The directory of the CMakeLists, as it appears in source_path, e.g. "src/".
    std::string dir_prefix;
    {
        const std::size_t slash = cmake_file.find_last_of('/');
        if (slash != std::string::npos) {
            // We only need the LAST path component's parent, so take the trailing directory name.
            const std::string parent = cmake_file.substr(0, slash);
            const std::size_t pslash = parent.find_last_of('/');
            dir_prefix = (pslash == std::string::npos) ? std::string{}
                                                       : parent.substr(pslash + 1) + "/";
        }
    }
    std::string entry_form(source_path);
    if (!dir_prefix.empty() && entry_form.rfind(dir_prefix, 0) == 0) {
        entry_form = entry_form.substr(dir_prefix.size());
    }

    std::string line;
    int         list_depth = 0;
    std::size_t line_no    = 0;
    while (std::getline(in, line)) {
        ++line_no;
        std::string trimmed = line;
        const std::size_t first = trimmed.find_first_not_of(" \t");
        trimmed = (first == std::string::npos) ? std::string{} : trimmed.substr(first);

        const bool opens = trimmed.rfind("target_sources(", 0) == 0 ||
                           trimmed.rfind("add_library(", 0) == 0 ||
                           trimmed.rfind("add_executable(", 0) == 0;
        if (opens) { ++list_depth; }
        const bool closes_here = (list_depth > 0 && trimmed == ")");

        if (list_depth > 0 && !trimmed.empty() && trimmed[0] != '#' && trimmed[0] != '"' &&
            !opens &&
            (trimmed.find(entry_form) != std::string::npos ||
             trimmed.find(std::string(source_path)) != std::string::npos)) {
            if (where != nullptr) { *where = cmake_file + ":" + std::to_string(line_no); }
            return true;
        }
        if (closes_here) { --list_depth; }
        if (list_depth < 0) { list_depth = 0; }
    }
    return false;
}

// The convenience form over a LIST of CMakeLists, which is what the observer uses so that a
// source entry in tests/CMakeLists.txt counts too.
[[nodiscard]] inline bool amd_is_source_entry(const std::vector<std::string>& cmake_files,
                                             std::string_view source_path,
                                             std::string* where = nullptr) {
    for (const std::string& f : cmake_files) {
        if (amd_is_source_entry_in(f, source_path, where)) { return true; }
    }
    return false;
}

// Is there an object file for this source under any of the named build directories? Reading 3.
// The build directories are a PARAMETER so the scope of the claim is always printable: "no .o
// under build*" and "no .o under build-sm70" are different statements.
[[nodiscard]] inline bool amd_object_present(const std::string& tree_root,
                                             const std::vector<std::string>& build_dirs,
                                             std::string_view source_path,
                                             std::string* where = nullptr) {
    // The object is named after the source's stem, e.g. src/ops/linear/gfx906/q4_tiled_gfx906.cu
    // -> q4_tiled_gfx906.cu.o. Searching by stem keeps this independent of CMake's own object
    // tree layout, which differs between generators.
    const std::size_t slash = source_path.find_last_of('/');
    const std::string_view stem = (slash == std::string_view::npos) ? source_path
                                                                   : source_path.substr(slash + 1);
    for (const std::string& dir : build_dirs) {
        const std::string base = tree_root + "/" + dir;
        std::error_code   ec;
        if (!std::filesystem::is_directory(base, ec) || ec) { continue; }
        for (std::filesystem::recursive_directory_iterator it(base, ec), end; it != end && !ec;
             it.increment(ec)) {
            const std::string name = it->path().filename().string();
            if (name.rfind(std::string(stem) + ".o", 0) == 0) {
                if (where != nullptr) { *where = it->path().string(); }
                return true;
            }
        }
    }
    return false;
}

// READING 3' FOR A HEADER-ONLY FACILITY: how many files in the tree include this header, and is
// at least one of THEM a source entry of a target? A header has no object of its own, so this is
// the only reading that can answer "is it in the build" for one -- and requiring an object would
// be a counter that cannot come up, which is the failure this function was added for.
//
// It is TRANSITIVE BY ONE STEP ONLY, deliberately: an includer that is ITSELF a header is not
// followed further, so the answer is "reachable from a compiled TU in one hop". A deeper walk
// would need a real include graph and would be a second answer to what the compiler already knows.
// The one-hop result is printed with its counts so a reader can see it is one hop.
struct AmdHeaderReach {
    std::size_t includers = 0;
    bool        includer_is_source_entry = false;
    std::string ledger;
};

[[nodiscard]] inline AmdHeaderReach amd_header_reach(const std::string& tree_root,
                                                     std::string_view header_path,
                                                     const std::vector<std::string>& cmake_files) {
    AmdHeaderReach out;
    const std::size_t slash = header_path.find_last_of('/');
    const std::string_view base = (slash == std::string_view::npos) ? header_path
                                                                   : header_path.substr(slash + 1);
    const std::vector<std::string> roots = {"src", "include", "apps", "tests"};
    for (const std::string& sub : roots) {
        const std::string dir = tree_root + "/" + sub;
        std::error_code   ec;
        if (!std::filesystem::is_directory(dir, ec) || ec) { continue; }
        for (std::filesystem::recursive_directory_iterator it(dir, ec), end; it != end && !ec;
             it.increment(ec)) {
            if (!it->is_regular_file(ec) || ec) { continue; }
            std::ifstream f(it->path());
            if (!f) { continue; }
            std::string line;
            bool        includes = false;
            while (std::getline(f, line)) {
                if (line.rfind("#include", line.find_first_not_of(" \t")) != std::string::npos ||
                    line.find("#include") != std::string::npos) {
                    if (line.find(base) != std::string::npos) { includes = true; break; }
                }
            }
            if (!includes) { continue; }
            ++out.includers;
            const std::string rel = it->path().string().substr(tree_root.size() + 1);
            out.ledger += "includer: " + rel;
            std::string where;
            if (amd_is_source_entry(cmake_files, rel, &where)) {
                out.includer_is_source_entry = true;
                out.ledger += "  [IS a source entry at " + where + "]";
            } else {
                out.ledger += "  [not a source entry -- it is a header, or unwired]";
            }
            out.ledger += "\n";
        }
    }
    return out;
}

// THE OBSERVER. Returns one AmdLevelFacts per level of the ladder, with the three takable
// readings filled and `symbol_resolved` left FALSE ON PURPOSE -- see the note on its own field in
// amd_route.h: the linker test can only be taken in a configuration that has an AMD toolchain, so
// this observer does not pretend to have taken it.
//
// The LDS figures are the ones this tree's own sites carry: 96 KiB on the QPN WMMA entry
// (kAmdLdsBlockerSites, qpn_kernels.cuh) against the 64 KiB a workgroup may request on the AMD
// targets of kAmdLadder -- the "one blocker no translator can see" that table exists for. They are
// attached to the level whose kernel would be staged that way, and the limit is a parameter
// because it is a property of the part and is 64 KiB only for the rungs listed.
struct AmdObservation {
    std::vector<AmdLevelFacts> levels;
    std::string                tree_root;
    std::vector<std::string>   build_dirs;
    std::vector<std::string>   cmake_files;
    std::string                ledger; // one line per reading actually taken, so a zero is loud
};

[[nodiscard]] inline AmdObservation observe_amd_levels(
    const std::string& tree_root, const std::vector<std::string>& build_dirs,
    std::uint64_t lds_limit_bytes = 65536) {
    AmdObservation obs;
    obs.tree_root  = tree_root;
    obs.build_dirs = build_dirs;
    // The CMake lists that are searched for source entries. Named, not globbed, so the scope of
    // reading 2 is printable -- and so a directory that is NOT searched cannot be mistaken for a
    // directory that was searched and came up empty.
    obs.cmake_files = {tree_root + "/src/CMakeLists.txt", tree_root + "/tests/CMakeLists.txt",
                       tree_root + "/apps/CMakeLists.txt"};

    std::error_code ec;
    auto take = [&](AmdRoute route, AmdLevelKind kind, std::string_view path,
                    std::uint64_t lds_request) {
        AmdLevelFacts f;
        f.route             = route;
        f.kind              = kind;
        f.source_path       = path;
        f.lds_limit_bytes   = lds_limit_bytes;
        f.lds_request_bytes = lds_request;
        if (!path.empty()) {
            f.source_exists = std::filesystem::exists(tree_root + "/" + std::string(path), ec) && !ec;
        }
        std::string entry_where;
        if (kind == AmdLevelKind::CompiledSource && !path.empty()) {
            f.is_source_entry = amd_is_source_entry(obs.cmake_files, path, &entry_where);
        }
        std::string obj_where;
        if (kind == AmdLevelKind::CompiledSource && !path.empty()) {
            f.object_in_build_dir = amd_object_present(tree_root, build_dirs, path, &obj_where);
        }
        if (kind == AmdLevelKind::HeaderOnlyFacility && !path.empty()) {
            const AmdHeaderReach reach = amd_header_reach(tree_root, path, obs.cmake_files);
            f.includers                = reach.includers;
            f.includer_is_source_entry = reach.includer_is_source_entry;
            obs.ledger += "    " + reach.ledger;
        }
        // READING 4 IS NOT TAKEN HERE, AND THAT IS A STATEMENT RATHER THAN AN OMISSION.
        f.symbol_resolved = false;

        // The verdict is computed by amd_route.h's own predicate, so this observer cannot
        // disagree with the walk about what "in this build" means.
        const bool in_build = amd_level_in_build(f);
        obs.ledger += "  " + std::string(amd_route_name(route)) + " [" +
                      std::string(amd_level_kind_name(kind)) + "] " +
                      (path.empty() ? std::string("<no carrier named>") : std::string(path)) +
                      ": source_exists=" + (path.empty() ? "n/a" : (f.source_exists ? "yes" : "no")) +
                      " is_source_entry=" + (f.is_source_entry ? "yes" : "no") +
                      (entry_where.empty() ? std::string() : " (" + entry_where + ")") +
                      " object_in_build_dir=" + (f.object_in_build_dir ? "yes" : "no") +
                      (obj_where.empty() ? std::string() : " (" + obj_where + ")") +
                      " includers=" + std::to_string(f.includers) +
                      " IN_BUILD=" + (in_build ? "YES" : "no") +
                      " symbol_resolved=NOT TAKEN (no AMD toolchain in this configure)\n";
        obs.levels.push_back(f);
    };

    // The two matrix levels have NO CARRIER IN THIS TREE, and that is a measurement rather than a
    // placeholder: the amdgcn census (dl/amdprobe/logs/09_amd_wiring_fixed.log) found v_mfma and
    // v_wmma to occur ONLY in prose -- 0 hits inside any asm block or intrinsic call anywhere under
    // src/. So their source_path is empty and reading 1 is false, which is the honest reading.
    take(AmdRoute::AmdMfmaNative, AmdLevelKind::CompiledSource, std::string_view{}, 0);
    take(AmdRoute::AmdWmmaNative, AmdLevelKind::CompiledSource, std::string_view{}, 0);
    // The dot/SIMT level IS carried by real, wired, option-gated sources.
    take(AmdRoute::AmdDotSimt, AmdLevelKind::CompiledSource, kAmdLevelSourceDotSimt, 0);
    // The floor level is a HEADER-ONLY FACILITY, so it gets the other kind of reading.
    take(AmdRoute::AmdHostCpuReference, AmdLevelKind::HeaderOnlyFacility, kAmdLevelSourceHostCpu, 0);
    return obs;
}

// ---------------------------------------------------------------------------
// THE CELL BUILDER. One call per (rung, format) pair, and it is the ONLY place the descent's
// inputs are assembled -- so no shell loop and no second program can drift from the fields.
// ---------------------------------------------------------------------------
[[nodiscard]] inline AmdRouteInputs amd_route_inputs_for(std::string_view target,
                                                        artifact::NumericFormat format,
                                                        AmdProblemShape shape,
                                                        const AmdObservation& obs) {
    AmdRouteInputs in;
    in.rung_target = target;
    const AmdRung* rung = amd_rung(target);
    in.rung_in_ladder = (rung != nullptr);
    if (rung != nullptr) {
        in.rung_isa   = rung->isa;
        in.rung_cards = rung->cards;
    }
    const auto matrix = amd_matrix_path_for(target);
    // An unlisted rung gets NO matrix level rather than a neighbour's: fail closed, upward here
    // because the walk order is high-to-low.
    in.matrix_path_is_wmma = (matrix.second && matrix.first == AmdMatrixPath::Wmma);
    in.format = amd_format_facts(format);
    in.shape  = shape;
    in.levels = obs.levels;
    return in;
}

// A one-line statement of the matrix path, so a reader of the cell table can see WHY the walk
// tested the level it tested. Empty for an unlisted rung, with the reason.
[[nodiscard]] inline std::string amd_matrix_path_line(std::string_view target) {
    const auto m = amd_matrix_path_for(target);
    if (!m.second) {
        return std::string(target) +
               ": NOT IN kAmdMatrixPaths, so NO matrix level is offered -- an unlisted target "
               "must not be handed a neighbouring rung's path.";
    }
    return std::string(target) + ": " + std::string(amd_matrix_path_name(m.first)) +
           " (EXTERNAL-UNPROBED, measured_in_this_tree=false)";
}

} // namespace ninfer::caps

#endif // NINFER_CORE_AMD_ROUTE_TABLES_H
