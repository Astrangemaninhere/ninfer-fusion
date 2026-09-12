// ple_ngram_window_test.cpp - host-only, GPU-free acceptance for the PLE
// n-gram row path.
//
// Why this exists next to the coverage that is already in the tree: both
// tests/test_ple_layout.cpp and tools/ple_reference.py drive EVERY token with
// prevs == {eos, eos}. The bigram/trigram mixers therefore only ever see ctx0,
// and the predecessor window -- the part the PLE row ids actually depend on --
// is never exercised with real ids. Neither covers an EOS cut in the middle of
// a stream, a negative (missing) predecessor, or a token that IS eos. And the
// row id -> (physical file, byte offset) -> sidecar bytes leg is only checked
// for "in range"; nothing reads the bytes back and compares them to what the
// sidecar actually stores.
//
// It needs no GPU: derive_rows / row_location / pread are all host code, so
// this runs on a box with no CUDA device at all (which is exactly why it can
// run before the engine is unblocked). The device half of the gather stays
// covered by tests/ops/ple_table_e2e_test.cu.
//
// Build (plain host compile; nvcc/cmake are NOT involved):
//   g++ -std=c++20 -O1 -I src -I third_party -DNINFER_SOURCE_DIR='"."'
//       tools/ple_ngram_window_test.cpp src/ops/ple/ple_layout.cpp
//       -o /tmp/ple_ngram_window_test
// Run:
//   /tmp/ple_ngram_window_test            # synthetic fixture, generated once
//   /tmp/ple_ngram_window_test <root>     # an existing sidecar root
//   /tmp/ple_ngram_window_test --real <root>   # a production sidecar: byte
//                                         # read-back drops to length/non-zero
//                                         # (a real table does not encode its
//                                         # own row ids)
//   NINFER_PLE_STATS=1 /tmp/ple_ngram_window_test
//
// Exit code 0 = every check passed.

#include "ops/ple/ple_layout.h"
#include "ops/ple/ple_stage.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#define NINFER_SOURCE_DIR "."
#endif

namespace {

using ninfer::ops::ple::PleLayout;
using ninfer::ops::ple::PlePhase;
using ninfer::ops::ple::ple_phase_window;

constexpr int kEos    = 151643; // Qwen3.8 vocabulary EOS, the value the other PLE tests use
constexpr int kHeads  = 16;     // (ngram_size - 1) * heads_per_ngram = 2 * 8
constexpr int kRowDim = 160;
constexpr int kStride = 320; // BF16 x 160

// Each step is one decode column: ctx0 plus its two predecessors. The set is
// chosen so a single run covers every branch of the window builder in
// PleLayout::derive_rows_one.
struct Step {
    int ctx0;
    int prev1;
    int prev2;
    const char* what;
};

constexpr Step kSteps[] = {
    {11, kEos, kEos, "sequence start (both predecessors missing)"},
    {12, 11, kEos, "bigram only (prev2 missing)"},
    {13, 12, 11, "full trigram"},
    {14, 13, 12, "full trigram"},
    {15, 14, 13, "full trigram"},
    {21, kEos, 15, "EOS cut between the two predecessors"},
    {22, 21, kEos, "window rebuilt after the cut"},
    {23, 22, 21, "trigram again after the cut"},
    {31, -1, 23, "negative (missing) predecessor"},
    {41, 41, kEos, "repeated token in the window"},
    {kEos, 15, 14, "ctx0 is itself eos"},
    {13, 12, 11, "EXACT repeat of step 2 -> guaranteed row reuse"},
};
constexpr int kT    = static_cast<int>(sizeof(kSteps) / sizeof(kSteps[0]));
constexpr int kPrev = kT * (3 - 1);

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++g_failures; }
}

void summary(const std::string& label, const std::string& value) {
    std::printf("%-12s%-30s%s\n", "summary", label.c_str(), value.c_str());
}

// The oracle lives in the repo, but this binary is meant to be runnable from
// anywhere: look next to the compiled-in source dir first, then walk up from the
// working directory. Returns an empty path when neither is found (the failure is
// then reported with both locations).
std::filesystem::path reference_script() {
    const auto probe = [](std::filesystem::path base) -> std::filesystem::path {
        const std::filesystem::path candidate = base / "tools" / "ple_reference.py";
        std::error_code ec;
        return std::filesystem::exists(candidate, ec) ? candidate : std::filesystem::path{};
    };
    if (auto found = probe(NINFER_SOURCE_DIR); !found.empty()) { return found; }
    std::filesystem::path base = std::filesystem::current_path();
    for (int level = 0; level < 6; ++level) {
        if (auto found = probe(base); !found.empty()) { return found; }
        if (!base.has_parent_path()) { break; }
        base = base.parent_path();
    }
    return {};
}

std::string run_python(const std::string& args) {
    const std::filesystem::path script = reference_script();
    if (script.empty()) { return {}; }
    const std::string command = "python3 \"" + script.string() + "\" " + args + " 2>/dev/null";
    std::string out;
    std::array<char, 4096> buffer{};
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) { return {}; }
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        out += buffer.data();
    }
    pclose(pipe);
    return out;
}

// "13 [a, b, c, ...]" per line, the format tools/ple_reference.py prints.
bool parse_rows(const std::string& text, std::vector<std::array<int, kHeads>>& rows) {
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) { continue; }
        std::istringstream stream(line);
        int token = 0;
        char open = 0;
        stream >> token >> open;
        if (open != '[') { return false; }
        std::array<int, kHeads> row{};
        for (int head = 0; head < kHeads; ++head) {
            int value     = 0;
            char separator = 0;
            stream >> value >> separator;
            row[head] = value;
        }
        rows.push_back(row);
    }
    return rows.size() == static_cast<std::size_t>(kT);
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path root;
    bool real_sidecar = false;
    for (int index = 1; index < argc; ++index) {
        if (std::string(argv[index]) == "--real") {
            real_sidecar = true;
        } else if (root.empty()) {
            root = argv[index];
        }
    }
    if (root.empty()) { root = std::filesystem::temp_directory_path() / "ple_ngram_window_fixture"; }
    const std::filesystem::path manifest = root / "ple-manifest.json";

    // 1. Fixture. No manifest -> generate the synthetic one (32k rows; every
    //    row's BF16 payload encodes its own global row id).
    if (!std::filesystem::exists(manifest)) {
        const std::string gen = run_python("gen \"" + root.string() + "\"");
        if (gen.empty()) {
            std::printf("FAIL fixture generation produced no output for %s\n",
                        root.string().c_str());
            std::printf("     tools/ple_reference.py was looked for under NINFER_SOURCE_DIR=%s\n"
                        "     and above the working directory %s\n",
                        NINFER_SOURCE_DIR, std::filesystem::current_path().string().c_str());
            return 1;
        }
    }
    check(std::filesystem::exists(manifest), "sidecar manifest present: " + manifest.string());
    if (g_failures != 0) { return 1; }

    // 2. The declaration side (data-driven; a spec with no "ple" block is a
    //    valid declaration of absence and must stay a no-op).
    {
        const std::string declared =
            R"({"model_id":"synthetic","geometry":{"hidden":2560},)"
            R"("ple":{"ngram_size":3,"heads_per_ngram":8,"ple_embed_dim":2560,"ple_layer_ids":[2]}})";
        const auto stage = ninfer::ops::ple::PleStageDecl::from_spec_text(declared, "inline");
        check(stage.present, "declaration with a ple block is present");
        check(stage.n_heads() == kHeads, "declaration derives n_heads = (ngram-1)*heads_per_ngram");
        check(stage.declares_layer(2) && !stage.declares_layer(1),
              "declaration gates exactly the listed layers");
        stage.validate();

        const auto absent = ninfer::ops::ple::PleStageDecl::from_spec_text(
            R"({"model_id":"synthetic"})", "inline");
        check(!absent.present, "spec without a ple block declares absence (no-op path)");
        absent.validate_against(3, 8, kHeads, kRowDim); // must not throw when absent

        bool threw = false;
        try {
            stage.validate_against(3, 8, kHeads, 128); // wrong row dim -> embed mismatch
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "declaration/sidecar geometry mismatch fails loudly");
    }

    PleLayout layout = PleLayout::from_manifest(manifest.string());
    check(layout.ngram_size == 3 && layout.n_heads == kHeads && layout.row_stride_bytes == kStride,
          "sidecar geometry is ngram=3 heads=16 stride=320");
    if (g_failures != 0) { return 1; }

    // 2b. The declaration surface, driven from the REAL arch-spec the importer
    //     wrote for this model rather than an inline literal -- this is what
    //     keeps the engine free of a per-model branch, so it has to be checked
    //     against the file the importer actually emits.
    {
        const std::filesystem::path script = reference_script();
        const std::filesystem::path spec =
            script.empty()
                ? std::filesystem::path{}
                : script.parent_path().parent_path() / "tools/archkit/specs/qwen4_exp_spec.json";
        if (!spec.empty() && std::filesystem::exists(spec)) {
            const auto declared = ninfer::ops::ple::PleStageDecl::from_spec_file(spec.string());
            declared.validate();
            check(declared.present, "qwen4_exp_spec.json declares a PLE stage");
            check(declared.ngram_size == 3 && declared.heads_per_ngram == 8 &&
                      declared.n_heads() == kHeads,
                  "qwen4_exp_spec.json gives ngram=3, heads_per_ngram=8 -> 16 heads");
            check(declared.embed_dim == 2560, "qwen4_exp_spec.json gives ple_embed_dim=2560");
            check(declared.declares_layer(2) && !declared.declares_layer(1),
                  "qwen4_exp_spec.json declares ple_layer_ids == [2]");
            declared.validate_against(layout.ngram_size, layout.heads_per_ngram, layout.n_heads,
                                      layout.embedding_row_dimension);
            check(true, "declaration geometry agrees with the sidecar manifest");
            summary("ple declaration source", spec.filename().string());
        } else {
            std::printf("skip arch-spec absent: %s\n", spec.string().c_str());
        }
    }

    std::vector<std::int32_t> tokens(kT);
    std::vector<std::int32_t> prevs(kPrev);
    for (int index = 0; index < kT; ++index) {
        tokens[static_cast<std::size_t>(index)]      = kSteps[index].ctx0;
        prevs[static_cast<std::size_t>(index * 2)]   = kSteps[index].prev1;
        prevs[static_cast<std::size_t>(index * 2 + 1)] = kSteps[index].prev2;
    }

    // 3. Cross-check the C++ window builder against the independent Python
    //    reference, driven by the same real (non-EOS) predecessor stream.
    std::vector<std::array<int, kHeads>> expected;
    std::string replay_args = "replay \"" + manifest.string() + "\" " + std::to_string(kEos) + " " +
                              std::to_string(kT);
    for (const auto value : tokens) { replay_args += " " + std::to_string(value); }
    for (const auto value : prevs) { replay_args += " " + std::to_string(value); }
    const std::string reference = run_python(replay_args);
    check(parse_rows(reference, expected), "python reference produced kT rows");
    if (g_failures != 0) {
        std::printf("---- reference output ----\n%s\n", reference.c_str());
        std::printf("(if empty: tools/ple_reference.py needs the 'replay' command)\n");
        return 1;
    }

    std::vector<std::int32_t> rows(static_cast<std::size_t>(kT) * kHeads);
    layout.derive_rows(tokens, prevs, kEos, rows.data());
    bool derivation_ok = true;
    for (int index = 0; index < kT && derivation_ok; ++index) {
        for (int head = 0; head < kHeads; ++head) {
            const int got = rows[static_cast<std::size_t>(index) * kHeads + head];
            const int want =
                expected[static_cast<std::size_t>(index)][static_cast<std::size_t>(head)];
            if (got != want) {
                std::printf("     step %d (%s) head %d: c++ %d != reference %d\n", index,
                            kSteps[index].what, head, got, want);
                derivation_ok = false;
                break;
            }
        }
    }
    check(derivation_ok, "C++ row derivation matches the reference on real windows");
    if (g_failures != 0) { return 1; }

    // The window must actually vary: all-EOS predecessors (the regime the
    // existing tests use) cannot produce the same rows as real ones.
    {
        std::vector<std::int32_t> flat(static_cast<std::size_t>(kT) * kHeads);
        std::vector<std::int32_t> flat_prevs(kPrev, kEos);
        layout.derive_rows(tokens, flat_prevs, kEos, flat.data());
        check(flat != rows, "real predecessors change the rows (not the all-EOS regime)");
    }

    // 4. Row id -> (physical file, byte offset) -> sidecar bytes. The fixture
    //    encodes each row's global id in its BF16 payload, so a read-back
    //    proves the offset arithmetic, not merely that the row is in range.
    std::vector<int> fds;
    for (const auto& file : layout.physical_files) {
        fds.push_back(::open((root / file.path).c_str(), O_RDONLY));
    }
    bool all_open = !fds.empty();
    for (const int fd : fds) { all_open = all_open && fd >= 0; }
    check(all_open, "all sidecar files opened for read-back");
    if (!all_open) { return 1; }

    std::size_t bytes_read  = 0;
    std::size_t nonzero_row = 0;
    int first_mismatch      = -1;
    for (std::size_t index = 0; index < rows.size(); ++index) {
        std::uint32_t file_index = 0;
        std::uint64_t offset     = 0;
        if (!layout.row_location(static_cast<std::uint64_t>(rows[index]), file_index, offset)) {
            first_mismatch = static_cast<int>(index);
            break;
        }
        std::array<unsigned char, kStride> buffer{};
        const ssize_t got =
            ::pread(fds[file_index], buffer.data(), buffer.size(), static_cast<off_t>(offset));
        if (got != kStride) {
            first_mismatch = static_cast<int>(index);
            break;
        }
        bytes_read += buffer.size();
        std::uint16_t encoded = 0;
        std::memcpy(&encoded, buffer.data(), sizeof(encoded));
        if (real_sidecar) {
            // A production table does not encode its own row ids, so the only
            // read-back claim available is "the bytes are really there".
            if (encoded != 0) { ++nonzero_row; }
            continue;
        }
        if (encoded != static_cast<std::uint16_t>(rows[index] & 0xFFFF)) {
            first_mismatch = static_cast<int>(index);
            break;
        }
    }
    for (const int fd : fds) {
        if (fd >= 0) { ::close(fd); }
    }
    if (real_sidecar) {
        check(first_mismatch < 0 && nonzero_row > 0,
              "every derived row id reads back the full 320 B row from the sidecar");
        summary("ple read-back mode", "real sidecar (length/non-zero only)");
    } else {
        check(first_mismatch < 0,
              "every derived row id resolves to sidecar bytes that encode that id");
        if (first_mismatch >= 0) { std::printf("     first bad row index %d\n", first_mismatch); }
    }

    // 5. Phase semantics. Verify owes the last column only; gathering the whole
    //    batch there costs kT times the faults for one column of useful data.
    {
        const auto verify  = ple_phase_window(PlePhase::Verify, kT);
        const auto prefill = ple_phase_window(PlePhase::Prefill, kT);
        const auto decode  = ple_phase_window(PlePhase::Decode, 1);
        check(verify.first_token == static_cast<std::size_t>(kT - 1) && verify.tokens == 1,
              "Verify window is the last column only");
        check(prefill.first_token == 0 && prefill.tokens == static_cast<std::size_t>(kT),
              "Prefill window is every column");
        check(decode.tokens == 1, "Decode window is one column");

        std::array<std::int32_t, kHeads> single{};
        layout.derive_rows_one(tokens[static_cast<std::size_t>(kT - 1)],
                               prevs.data() + 2 * (kT - 1), kEos, single.data());
        bool same = true;
        for (int head = 0; head < kHeads; ++head) {
            if (single[static_cast<std::size_t>(head)] !=
                rows[static_cast<std::size_t>(kT - 1) * kHeads + head]) {
                same = false;
            }
        }
        check(same, "the Verify-window single-column derivation equals the batched column");
    }

    // 6. The observable. rows_reused is the row-level hit count; it must be
    //    non-zero because step 11 repeats step 2 exactly.
    {
        ninfer::ops::ple::PleForensics forensics;
        const auto window = ple_phase_window(PlePhase::Verify, kT);
        forensics.gathers.fetch_add(1, std::memory_order_relaxed);
        forensics.gathers_by_phase[static_cast<int>(PlePhase::Verify)].fetch_add(
            1, std::memory_order_relaxed);
        forensics.tokens_offered.fetch_add(kT, std::memory_order_relaxed);
        forensics.tokens_gathered.fetch_add(window.tokens, std::memory_order_relaxed);
        forensics.rows_derived.fetch_add(rows.size(), std::memory_order_relaxed);
        forensics.rows_resolved.fetch_add(rows.size(), std::memory_order_relaxed);

        std::vector<std::int32_t> sorted(rows);
        std::sort(sorted.begin(), sorted.end());
        const auto unique = static_cast<std::uint64_t>(std::unique(sorted.begin(), sorted.end()) -
                                                       sorted.begin());
        forensics.rows_unique.fetch_add(unique, std::memory_order_relaxed);
        forensics.rows_reused.fetch_add(rows.size() - unique, std::memory_order_relaxed);
        forensics.bytes_read.fetch_add(bytes_read, std::memory_order_relaxed);

        check(forensics.observed(), "forensics observed() is true after a gather");
        check(forensics.rows_reused.load() > 0,
              "row reuse counter is non-zero (repeated window deduped)");
        summary("ple gathers", std::to_string(forensics.gathers.load()));
        summary("ple rows derived", std::to_string(forensics.rows_derived.load()));
        summary("ple rows unique/reused",
                std::to_string(unique) + "/" + std::to_string(rows.size() - unique));
        summary("ple sidecar bytes read", std::to_string(bytes_read));
        summary("ple verify window", "(" + std::to_string(window.first_token) + "," +
                                         std::to_string(window.tokens) + ") of " +
                                         std::to_string(kT));
        summary("ple stats line", forensics.to_line());
        if (ninfer::ops::ple::ple_stats_enabled()) {
            summary("ple stats gate", "NINFER_PLE_STATS=1");
        }
    }

    // 7. The speed constraint, host side. The per-step host costs are measured
    //    here rather than asserted; the SSD/device half is the existing
    //    tests/ops/ple_table_e2e_test.cu's job. A decode step at 50 tok/s has a
    //    20 ms budget, so anything in the tens of nanoseconds is free by three
    //    orders of magnitude.
    {
        constexpr int kIterations = 2'000'000;
        const auto clock_now     = [] { return std::chrono::steady_clock::now(); };
        const auto elapsed_ns    = [](auto start, auto stop) {
            return std::chrono::duration<double, std::nano>(stop - start).count();
        };

        const auto absent = ninfer::ops::ple::PleStageDecl::from_spec_text(
            R"({"model_id":"no-ple"})", "inline");
        std::int64_t off_sink = 0;
        const auto off_start  = clock_now();
        for (int i = 0; i < kIterations; ++i) {
            if (absent.present) { off_sink += static_cast<std::int64_t>(absent.ngram_size); }
        }
        const double off_ns = elapsed_ns(off_start, clock_now()) / kIterations;

        std::int64_t window_sink = 0;
        const auto window_start  = clock_now();
        for (int i = 0; i < kIterations; ++i) {
            window_sink += static_cast<std::int64_t>(ple_phase_window(
                               PlePhase::Verify, 1 + (i & 7)).tokens);
        }
        const double window_ns = elapsed_ns(window_start, clock_now()) / kIterations;

        std::int32_t one_step[2] = {12, 11};
        std::int32_t step_rows[kHeads]{};
        std::int64_t row_sink = 0;
        const auto derive_start = clock_now();
        for (int i = 0; i < kIterations; ++i) {
            layout.derive_rows_one(13 + (i & 1), one_step, kEos, step_rows);
            row_sink += step_rows[0];
        }
        const double derive_ns = elapsed_ns(derive_start, clock_now()) / kIterations;

        check(off_ns < 10.0, "off-path declaration test costs < 10 ns/token");
        check(window_ns < 10.0, "phase-window decision costs < 10 ns/step");
        check(derive_ns < 5000.0, "host row derivation costs < 5 us/token (decode budget 20 ms)");
        summary("ple off-path ns/token", std::to_string(off_ns));
        summary("ple phase-window ns/step", std::to_string(window_ns));
        summary("ple host derive ns/token", std::to_string(derive_ns));
        (void)off_sink;
        (void)window_sink;
        (void)row_sink;
    }

    std::printf("%s: %d check(s) failed\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
