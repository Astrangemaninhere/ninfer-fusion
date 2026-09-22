// tests/ops/ple_seam_gather_e2e_test.cu — NGRAM-BUILD N2, the CUDA half.
//
// The host test (tests/test_ple_seam_window.cpp) proves the WIRING with a fake runtime,
// on purpose and without a GPU. This test is what keeps ple_seam_window.h out of the
// dead-code state the module is in today: it instantiates the SAME composition
// (ple_seam_gather_step) against the REAL ops::ple::PleTable, so the real SSD sidecar is
// gathered through the production seam by window-produced predecessors.
//
// It also carries the GPU-side load-bearing control: the rows gathered from the window's
// real predecessors must DIFFER from the rows an eos fill would gather. If the seam ever
// goes back to hand-filling prevs = {eos, eos}, the seam arm silently becomes the eos arm
// and that check goes red.
//
// Usage: ple_seam_gather_e2e_test <sidecar_root>   (or NINFER_PLE_SIDECAR_ROOT)
//   sidecar_root = dir with ple-manifest.json + ple/*.bin
// Three states, the same split every real-artifact test in this tree uses:
//   unset/empty        -> SKIP (exit 77)
//   no manifest inside -> MISPECONFIGURATION (exit 1, never 77)
//   a real sidecar     -> run (exit 0 or 1)
#include "ops/ple/ple_table.h"
#include "targets/qwen4_exp/impl/ple_seam_window.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <span>
#include <unistd.h>
#include <vector>

namespace {

int expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        return 1;
    }
    std::printf("ok: %s\n", what);
    return 0;
}

// Host reference: read the given rows straight out of the sidecar's physical files.
bool read_rows_host(const ninfer::ops::ple::PleLayout& layout,
                    const std::filesystem::path& root,
                    const std::vector<std::int32_t>& rows, std::vector<__half>& out) {
    static std::vector<int> fds;
    if (fds.empty()) {
        for (const auto& file : layout.physical_files) {
            fds.push_back(open((root / file.path).c_str(), 0));
        }
    }
    const int dim = static_cast<int>(layout.embedding_row_dimension);
    out.assign(rows.size() * static_cast<std::size_t>(dim), __half(0));
    for (std::size_t i = 0; i < rows.size(); ++i) {
        std::uint32_t fi = 0;
        std::uint64_t off = 0;
        if (!layout.row_location(static_cast<std::uint64_t>(rows[i]), fi, off)) { return false; }
        if (fi >= fds.size() || fds[fi] < 0) { return false; }
        if (pread(fds[fi], out.data() + i * dim, dim * 2, static_cast<off_t>(off)) != dim * 2) {
            return false;
        }
    }
    return true;
}

std::size_t count_differing(const std::vector<__half>& a, const std::vector<__half>& b) {
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    std::size_t differ = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::memcmp(&a[i], &b[i], 2) != 0) { ++differ; }
    }
    return differ;
}

// Same question over row IDS (the load-bearing control compares which rows each path would
// gather, not the gathered bytes).
std::size_t count_differing(const std::vector<std::int32_t>& a,
                            const std::vector<std::int32_t>& b) {
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    std::size_t differ = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) { ++differ; }
    }
    return differ;
}

} // namespace

int main(int argc, char** argv) {
    using namespace ninfer::ops::ple;
    using namespace ninfer::targets::qwen4_exp::detail;

    const char* const sidecar = argc >= 2 ? argv[1] : std::getenv("NINFER_PLE_SIDECAR_ROOT");
    if (sidecar == nullptr || *sidecar == '\0') {
        std::printf("skip: pass <sidecar_root> or set NINFER_PLE_SIDECAR_ROOT\n");
        return 77;
    }
    const std::filesystem::path root = sidecar;
    if (!std::filesystem::is_regular_file(root / "ple-manifest.json")) {
        std::printf("FAIL: %s was set to \"%s\" but that directory holds no "
                    "ple-manifest.json; refusing to report a skip for an explicitly "
                    "configured sidecar root\n",
                    argc >= 2 ? "argv[1]" : "NINFER_PLE_SIDECAR_ROOT", root.string().c_str());
        return 1;
    }

    int failures = 0;
    PleTableOptions opts;
    opts.sidecar_root = root;
    opts.cache_bytes = 1ULL << 20; // tiny on purpose: force real SSD faults
    PleTable table(opts);          // the REAL op, reading the REAL sidecar files

    const std::int32_t eos = 151645;
    constexpr std::int32_t kCommitted = 162042;
    const std::vector<std::int32_t> tokens{9707, 4242}; // a 2-column batch

    const auto& layout = table.layout();
    const int n_heads = static_cast<int>(layout.n_heads);
    const int row_dim = static_cast<int>(layout.embedding_row_dimension);

    // Committed history, then a Decode step: the window is the only source of these
    // predecessors -- the caller never hand-fills them.
    PleNgramWindow window(static_cast<std::uint32_t>(layout.ngram_size), eos);
    window.commit(std::span<const std::int32_t>(&kCommitted, 1));

    const PleSeamPlan plan = ple_seam_plan(PlePhase::Decode, layout.ngram_size, tokens.size());
    std::printf("plan: first_column=%zu columns=%zu prevs_per_token=%zu\n", plan.first_column,
                plan.columns, plan.prevs_per_token);

    const std::size_t elems = plan.columns * static_cast<std::size_t>(n_heads) *
                              static_cast<std::size_t>(row_dim);
    __half* d_seam = nullptr;
    cudaMalloc(&d_seam, elems * sizeof(__half));
    std::vector<std::int32_t> scratch;
    ple_seam_gather_step(table, window, std::span<const std::int32_t>(tokens), PlePhase::Decode,
                         eos, d_seam, /*stream=*/nullptr, scratch);
    cudaError_t err = cudaDeviceSynchronize();
    failures += expect(err == cudaSuccess, "seam composition + real-table GPU gather + sync");
    if (err != cudaSuccess) {
        std::printf("cuda: %s\n", cudaGetErrorString(err));
        return failures + 1;
    }
    // The seam handed the op two predecessors per column, oldest-position first, and they
    // are NOT both eos (that is the whole point of the wiring).
    failures += expect(scratch.size() == tokens.size() * 2 && scratch.front() != eos,
                       "the seam produced real predecessors (not an eos fill)");

    // What the seam derived host-side for the owed column, to locate the expected rows.
    const std::size_t owed_column = plan.first_column;
    std::vector<std::int32_t> prevs_scratch;
    std::vector<std::int32_t> owed_rows(static_cast<std::size_t>(n_heads));
    {
        // Re-derive just the owed column to know which rows to compare against, using the
        // same window (read-only: deriving must not commit).
        std::vector<std::int32_t> one(tokens.begin() + static_cast<std::ptrdiff_t>(owed_column),
                                     tokens.begin() + static_cast<std::ptrdiff_t>(owed_column) + 1);
        std::vector<std::int32_t> one_rows(static_cast<std::size_t>(n_heads));
        ple_seam_derive_rows(layout, window, std::span<const std::int32_t>(one), PlePhase::Decode,
                             eos, prevs_scratch, one_rows.data());
        owed_rows = one_rows;
    }

    std::vector<__half> got(elems);
    cudaMemcpy(got.data(), d_seam, elems * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<__half> ref;
    if (!read_rows_host(layout, root, owed_rows, ref)) {
        std::printf("FAIL: host reference read\n");
        return 1;
    }
    const std::size_t mismatches = count_differing(got, ref);
    std::printf("seam gather vs host pread: %zu elems, mismatches=%zu\n", got.size(), mismatches);
    failures += expect(mismatches == 0, "the seam-gathered column is bit-exact vs host pread");

    // LOAD-BEARING CONTROL (GPU side): the same column gathered from an eos fill instead
    // must differ. Remove the window wiring and these two arms become identical.
    std::vector<std::int32_t> eos_prevs(tokens.size() * 2, eos);
    std::vector<std::int32_t> eos_rows(tokens.size() * static_cast<std::size_t>(n_heads));
    table.derive_rows(std::span<const std::int32_t>(tokens),
                      std::span<const std::int32_t>(eos_prevs), eos, eos_rows.data());
    const std::vector<std::int32_t> eos_owed(
        eos_rows.begin() + static_cast<std::ptrdiff_t>(owed_column * n_heads),
        eos_rows.begin() + static_cast<std::ptrdiff_t>(owed_column * n_heads + n_heads));
    const std::size_t row_differ = count_differing(owed_rows, eos_owed);
    std::printf("seam rows vs eos-fill rows: %zu/%d head ids differ\n", row_differ, n_heads);
    failures += expect(row_differ > 0,
                       "control: window rows != eos-fill rows (identical => wiring gone)");

    cudaFree(d_seam);
    std::printf(failures == 0 ? "PLE_SEAM_GATHER_PASS\n" : "PLE_SEAM_GATHER_FAIL\n");
    return failures == 0 ? 0 : 1;
}
