// ple_table_e2e_test.cu — W2②: PleTable paging-fault + device-gather end-to-end
// against the real 95 GiB SSD sidecar. Verifies the one unproven runtime piece:
// derive_rows -> LRU fault-in (pinned UVA) -> device gather -> values match a
// host-side reference read of the same rows. Also exercises dedup (repeated
// n-gram) and a tiny cache budget to force eviction between gathers.
//
// Usage: ple_table_e2e_test <sidecar_root>
//   sidecar_root = dir with ple-manifest.json + ple/*.bin
#include "ops/ple/ple_table.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
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

// Host reference: read the 16 rows straight from the table via pread.
// (PleLayout::row_location is shared with the engine path, so this validates
// the fault/gather machinery, not the row math — that is covered by
// test_ple_layout and ple_gather_test.py.)
bool read_rows_host(const ninfer::ops::ple::PleTable& table,
                    const std::vector<std::int32_t>& rows,
                    const std::filesystem::path& root,
                    std::vector<__half>& out) {
    // open files once
    static std::vector<int> fds;
    if (fds.empty()) {
        for (int i = 0; i < 4; ++i) {
            char name[96];
            std::snprintf(name, sizeof(name), "ple/ple-bf16-0000%d-of-00004.bin", i + 1);
            fds.push_back(open((root / name).c_str(), 0));
        }
    }
    const int dim = table.layout().embedding_row_dimension;
    out.assign(rows.size() * dim, __half(0));
    for (std::size_t i = 0; i < rows.size(); ++i) {
        std::uint32_t fi = 0;
        std::uint64_t off = 0;
        if (!table.layout().row_location(static_cast<std::uint64_t>(rows[i]), fi, off)) {
            return false;
        }
        if (pread(fds[fi], out.data() + i * dim, dim * 2, static_cast<off_t>(off)) !=
            dim * 2) {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    using namespace ninfer::ops::ple;
    if (argc < 2) {
        std::printf("usage: %s <sidecar_root>\n", argv[0]);
        return 2;
    }
    const std::filesystem::path root = argv[1];
    int failures = 0;

    // Tiny budget on purpose: first gather must evict to fit the second.
    PleTableOptions opts;
    opts.sidecar_root = root;
    opts.cache_bytes = 1ULL << 20; // 1 MiB > 16 rows (5 KB) but forces churn
    PleTable table(opts);

    const std::int32_t eos = 151645;
    // prevs is FLAT: ngram_size-1 = 2 predecessors per token (near, far).
    // token1 and token2 share the same (current, prevs) window -> identical
    // row set (in-gather dedup); token0's window differs (EOS-cut).
    const std::vector<std::int32_t> tokens{162042, 9707, 9707};
    const std::vector<std::int32_t> prevs{
        eos,     eos,     // token0: both predecessors EOS -> cut to unigram-ish
        162042,  eos,     // token1: near=162042, far=EOS -> cut
        162042,  eos,     // token2: same window as token1
    };
    const int n = static_cast<int>(tokens.size());
    const int heads = static_cast<int>(table.layout().n_heads);
    const int dim = static_cast<int>(table.layout().embedding_row_dimension);

    std::vector<std::int32_t> rows(n * heads);
    table.derive_rows(tokens, prevs, eos, rows.data());
    failures += expect(rows[0] != rows[heads], "distinct windows -> distinct head0 rows");
    bool same = true;
    for (int h = 0; h < heads; ++h) {
        same = same && rows[heads + h] == rows[2 * heads + h];
    }
    failures += expect(same, "repeated window -> identical row set (dedup candidate)");

    // Device gather.
    __half* d_out = nullptr;
    cudaMalloc(&d_out, n * heads * dim * sizeof(__half));
    table.gather(rows.data(), n, d_out, 0);
    cudaError_t e = cudaDeviceSynchronize();
    failures += expect(e == cudaSuccess, "gather + sync");
    if (e != cudaSuccess) {
        std::printf("cuda: %s\n", cudaGetErrorString(e));
        return failures + 1;
    }
    std::vector<__half> got(n * heads * dim);
    cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost);

    // Host reference from the same files.
    std::vector<__half> ref;
    if (!read_rows_host(table, rows, root, ref)) {
        std::printf("FAIL: host reference read\n");
        return 1;
    }
    int mismatches = 0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a = __half2float(got[i]);
        const double b = __half2float(ref[i]);
        max_abs = std::max(max_abs, std::fabs(a - b));
        if (std::memcmp(&got[i], &ref[i], 2) != 0) { mismatches++; }
    }
    std::printf("gather vs host reference: %zu elems, mismatches=%d, max|diff|=%g\n",
                got.size(), mismatches, max_abs);
    failures += expect(mismatches == 0, "device gather bit-exact vs host pread");

    // Second gather with a different token must still be correct after LRU
    // churn under the tiny budget (eviction of the first epoch's pages).
    const std::vector<std::int32_t> t2{4242, 99, 777001};
    const std::vector<std::int32_t> p2{11, 4242, 99};
    std::vector<std::int32_t> rows2(n * heads);
    table.derive_rows(t2, p2, eos, rows2.data());
    table.gather(rows2.data(), n, d_out, 0);
    e = cudaDeviceSynchronize();
    failures += expect(e == cudaSuccess, "second gather after eviction");
    std::vector<__half> got2(n * heads * dim);
    cudaMemcpy(got2.data(), d_out, got2.size() * 2, cudaMemcpyDeviceToHost);
    std::vector<__half> ref2;
    read_rows_host(table, rows2, root, ref2);
    int mismatches2 = 0;
    for (std::size_t i = 0; i < got2.size(); ++i) {
        if (std::memcmp(&got2[i], &ref2[i], 2) != 0) { mismatches2++; }
    }
    std::printf("second gather mismatches=%d\n", mismatches2);
    failures += expect(mismatches2 == 0, "post-eviction gather bit-exact");

    cudaFree(d_out);
    std::printf(failures == 0 ? "PLE_TABLE_E2E_PASS\n" : "PLE_TABLE_E2E_FAIL\n");
    return failures == 0 ? 0 : 1;
}
