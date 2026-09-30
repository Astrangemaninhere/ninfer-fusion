// ---------------------------------------------------------------------------
// ninfer-hostpath -- THE CUDA-FREE FRONT DOOR THAT CAN OPEN AN ARTIFACT (F854, dl/hostsim).
//
// WHY THIS FILE EXISTS, IN ONE SENTENCE: the tree has two front doors that each do HALF of
// what a machine with no usable GPU needs, and neither does both halves.
//   * `ninfer` (apps/cli) opens artifacts and adjudicates them -- and CANNOT START on such a
//     machine. MEASURED 2026-09-26: with CUDA_VISIBLE_DEVICES="" it dies at
//     src/core/device.cu:52 (throw cuda_error_message("cudaGetDeviceCount failed", err)),
//     rc=1, stdout 0 bytes, and it does so EVEN FOR AN ARTIFACT PATH THAT DOES NOT EXIST --
//     so no gate of its own is ever reached.
//   * `ninfer-cufree` (apps/cufree) starts there -- and takes NO model, NO prompt and NO
//     artifact, by its own documented decision ("it never opens a device").
// THE MISSING CAPABILITY IS THE CONJUNCTION: a CUDA-FREE binary that opens a REAL artifact
// with THE ENGINE'S OWN READER and asks THE ENGINE'S OWN GATES about it. That is what this
// program is, and it is the first host-CPU entry point in this tree that does not need CUDA.
//
// HOW IT STAYS CUDA-FREE, AND WHY THAT IS A LINK-LINE FACT RATHER THAN A HOPE. Its own
// target in apps/CMakeLists.txt names NO library that carries CUDA (the same discipline
// ninfer-cufree documents), and it calls NOTHING from ninfer_core -- which is the target that
// PUBLIC-links CUDA::cudart (src/CMakeLists.txt:69). The two translation units it needs,
// artifact/reader.cpp and artifact/storage_layouts.cpp, contain NO CUDA INCLUDE AT ALL, which
// is why they can be built here: the tree's own `ninfer_artifact` cannot be, because
// src/CMakeLists.txt:79 makes it PUBLIC-link ninfer_core. That one line is the reason the
// engine's host half is not reachable on a host without a CUDA toolchain, and this target is
// built around it rather than through it. `objdump -p` on this binary is the check.
//
// WHAT IT DOES NOT DO, NAMED SO IT CANNOT BE READ IN:
//   * IT EMITS NO TOKENS. It runs no model arithmetic. `content_tokens_emitted: 0` is printed
//     in its own verdict, because the arithmetic that would emit does not exist on the host
//     half of this tree (measured: 0 host-CPU compute entry points among 1075 files under
//     src/, against 247 __global__ kernel declarations).
//   * It does not query a card, count devices, or attribute a vendor.
//   * Its route answer is a TABLE answer from this build's own tables plus build facts, and
//     the engine's own prose says so; this program does not upgrade it to a probe result.
//
// usage: ninfer-hostpath <artifact.ninfer> [more.ninfer ...]
// exit : 0 = every artifact opened and every gate answered (a READING was taken)
//        3 = at least one artifact could not be opened (refused, and named)
//        2 = usage error
// ---------------------------------------------------------------------------

#include "artifact/reader.h"          // THE ENGINE'S OWN READER (no CUDA include in it)
#include "core/amd_route_tables.h"    // THE ENGINE'S OWN DESCENT + TABLES
#include "core/amdsafe_gate.h"        // THE ENGINE'S OWN GATE, INCLUDED UNCHANGED

#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace art  = ninfer::artifact;
namespace caps = ninfer::caps;

namespace {

const char* yn(bool v) { return v ? "YES" : "NO"; }

// THE DOOR'S OWN ARITHMETIC, written as an EXPRESSION so this program's subject cannot drift
// from the engine's. src/targets/registry.cpp:642 reads:
//     const std::string amdsafe_path = identity.model_id + "/" + identity.weights_id;
std::string door_key(const art::ArtifactIdentity& id) { return id.model_id + "/" + id.weights_id; }

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "ninfer-hostpath: the CUDA-free front door that opens an artifact.\n"
                 "  usage: %s <artifact.ninfer> [more.ninfer ...]\n"
                 "  It opens each artifact with the engine's own reader, computes the load "
                 "door's\n  key (model_id/weights_id), and asks the engine's own amdsafe gate "
                 "and AMD\n  descent about it. It runs no arithmetic and emits no tokens.\n"
                 "  exit: 0 = a reading was taken; 3 = an artifact was refused; 2 = usage.\n",
                 argv0);
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i] != nullptr ? argv[i] : "";
        if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
    }
    if (argc < 2) { print_usage(argv[0]); return 2; }

    std::printf("ninfer-hostpath (F854): the engine's own reader + the engine's own gates, on a\n");
    std::printf("host with NO device. This binary emits NO tokens and claims NONE.\n\n");

    // The AMD ladder observation is taken ONCE, from THIS tree on disk, and its ledger is
    // printed in full so a zero below is loud rather than a shrug.
    const std::string tree_root = std::string(NINFER_HOSTPATH_TREE_ROOT);
    const caps::AmdObservation obs = caps::observe_amd_levels(tree_root, {"build"});

    int refused_artifacts = 0;
    int opened             = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string path = argv[i];
        std::printf("=====================================================================\n");
        std::printf("ARTIFACT: %s\n", path.c_str());
        std::printf("=====================================================================\n");

        std::error_code ec;
        if (!std::filesystem::exists(path, ec) || ec) {
            std::printf("  READER: ABSENT at this exact path (tested, not inferred).\n");
            std::printf("  This is NOT a reading about any artifact.\n\n");
            ++refused_artifacts;
            continue;
        }

        std::string key;
        art::NumericFormat first_format = art::NumericFormat::Count;
        bool have_format = false;
        try {
            art::Reader reader{std::filesystem::path(path)};
            const art::ArtifactIdentity& id = reader.identity();
            key = door_key(id);

            std::printf("  READER (src/artifact/reader.cpp, no CUDA include): OPENED\n");
            std::printf("    file_bytes  : %llu\n", (unsigned long long)reader.file_bytes());
            std::printf("    objects     : %zu\n", reader.objects().size());
            std::printf("    model_id    : \"%s\"\n", id.model_id.c_str());
            std::printf("    weights_id  : \"%s\"\n", id.weights_id.c_str());

            std::size_t n_tensor = 0, n_resource = 0;
            for (const art::ObjectDescriptor& o : reader.objects()) {
                if (const art::TensorDescriptor* t = std::get_if<art::TensorDescriptor>(&o)) {
                    ++n_tensor;
                    if (!have_format) { first_format = t->format; have_format = true; }
                } else {
                    ++n_resource;
                }
            }
            std::printf("    tensors     : %zu\tresources: %zu\n", n_tensor, n_resource);
            std::printf("    DOOR KEY (registry.cpp:642) : \"%s\"\n", key.c_str());
            ++opened;
        } catch (const std::exception& e) {
            std::printf("  READER: REFUSED BY NAME -- %s\n", e.what());
            ++refused_artifacts;
            continue;
        }

        // ---- THE GATE, asked the exact string the engine's load door asks it. ----
        const caps::AmdSafeGateDecision d = caps::amdsafe_gate(key);
        std::printf("\n  GATE (src/core/amdsafe_gate.h, unmodified)\n");
        std::printf("    declared                  : %s\n", yn(d.declared));
        std::printf("    target                    : %s\n", caps::amdsafe_target_token(d.target));
        std::printf("    admitted                  : %s\n", yn(d.admitted));
        std::printf("    primitive_check_performed : %s\n", yn(d.primitive_check_performed));
        std::printf("    reason_token              : %s\n", d.reason_token.c_str());
        std::printf("    allows_load()             : %s\n", yn(caps::amdsafe_gate_allows_load(key)));

        // ---- THE DESCENT. With no usable device the engine observes NO rung, so the rung
        // named here is kAmdLadder's own floor row and that is SAID rather than passed off as
        // an observation of the operator's card. ----
        if (have_format) {
            caps::AmdProblemShape shape;
            shape.m = 1; shape.n = 4096; shape.k = 4096;
            const caps::AmdRouteInputs in =
                caps::amd_route_inputs_for("gfx906", first_format, shape, obs);
            const caps::AmdRouteChoice ch = caps::select_amd_route(in);
            std::printf("\n  AMD DESCENT (src/core/amd_route.h, tables from amd_route_tables.h)\n");
            std::printf("    format of the first tensor : %s\n",
                        std::string(art::format_name(first_format)).c_str());
            std::printf("    route                      : %s\n",
                        std::string(caps::amd_route_name(ch.route)).c_str());
            std::printf("    outcome                    : %s\n",
                        std::string(caps::amd_outcome_name(ch.outcome)).c_str());
            std::printf("    levels_descended           : %u\n", ch.levels_descended);
            std::printf("    carrier                    : %s\n",
                        ch.kernel.empty() ? "<none>" : std::string(ch.kernel).c_str());
            std::printf("    NOTE: the rung named above is kAmdLadder's own floor row, NOT an\n");
            std::printf("          observation of any card. With no usable device this engine\n");
            std::printf("          observes no rung at all.\n");
        }
        std::printf("\n");
    }

    // ---- THE VERDICT. One block, and it states what it does not claim. ----
    std::printf("=====================================================================\n");
    std::printf("HOSTPATH-VERDICT\n");
    std::printf("  engine_reader_on_host   : %s\n", opened > 0 ? "RAN" : "NO ARTIFACT OPENED");
    std::printf("  artifacts_opened        : %d\n", opened);
    std::printf("  artifacts_refused       : %d\n", refused_artifacts);
    std::printf("  content_tokens_emitted  : 0  <-- THIS BINARY EMITS NOTHING. The arithmetic that\n");
    std::printf("                               would emit does not exist on the host half of this\n");
    std::printf("                               tree, and this verdict is not an emission.\n");
    std::printf("=====================================================================\n");
    return refused_artifacts == 0 ? 0 : 3;
}
