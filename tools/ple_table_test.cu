// tools/ple_table_test.cu — PleTable (SSD sidecar) gather acceptance test.
//
// Exercises the real PleTable (ple_layout + ple_table) against the real sidecar
// and cross-checks the gathered bytes with the Python reference
// (tools/archkit/ple_gather_check.py --emit-spec / --emit-golden): the spec file
// carries the token batch and the expected FNV-1a 64 of the gathered payload.
//
// Build (WSL; links the two ops sources directly, no engine library needed):
//   nvcc -O2 -std=c++20 -gencode arch=compute_120a,code=[compute_120a,sm_120a] \
//        -I <repo>/src -I <repo>/third_party \
//        tools/ple_table_test.cu \
//        <repo>/src/ops/ple/ple_layout.cpp <repo>/src/ops/ple/ple_table.cu \
//        -o ple_table_test
// Run:
//   ./ple_table_test --sidecar <dir> --spec ple_spec.txt [--expect-fnv 0x...]
#include "ops/ple/ple_table.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kFnvOffset = 0xCBF29CE484222325ULL;
constexpr std::uint64_t kFnvPrime  = 0x100000001B3ULL;

std::uint64_t fnv1a64(const std::uint8_t* data, std::size_t bytes, std::uint64_t seed) {
    std::uint64_t value = seed;
    for (std::size_t i = 0; i < bytes; ++i) {
        value ^= data[i];
        value *= kFnvPrime;
    }
    return value;
}

struct Spec {
    int ngram = 3;
    int n_heads = 16;
    int row_dim = 160;
    int row_stride = 320;
    int eos = 0;
    std::vector<std::int32_t> tokens;
};

bool load_spec(const std::string& path, Spec& spec, std::string& expect_fnv) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open spec %s\n", path.c_str()); return false; }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) { continue; }
        std::istringstream is(line);
        std::string key;
        is >> key;
        if (key == "ngram") { is >> spec.ngram; }
        else if (key == "n_heads") { is >> spec.n_heads; }
        else if (key == "row_dim") { is >> spec.row_dim; }
        else if (key == "row_stride") { is >> spec.row_stride; }
        else if (key == "eos") { is >> spec.eos; }
        else if (key == "tokens") {
            std::int32_t value = 0;
            while (is >> value) { spec.tokens.push_back(value); }
        }
    }
    if (expect_fnv.empty()) { return !spec.tokens.empty(); }
    return !spec.tokens.empty();
}

}  // namespace

int main(int argc, char** argv) {
    std::string sidecar;
    std::string spec_path;
    std::string expect_fnv;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--sidecar" && i + 1 < argc) { sidecar = argv[++i]; }
        else if (arg == "--spec" && i + 1 < argc) { spec_path = argv[++i]; }
        else if (arg == "--expect-fnv" && i + 1 < argc) { expect_fnv = argv[++i]; }
        else { std::fprintf(stderr, "unknown arg %s\n", arg.c_str()); return 2; }
    }
    if (sidecar.empty() || spec_path.empty()) {
        std::fprintf(stderr, "usage: %s --sidecar <dir> --spec <ple_spec.txt> [--expect-fnv 0x...]\n",
                     argv[0]);
        return 2;
    }

    Spec spec;
    if (!load_spec(spec_path, spec, expect_fnv)) { return 2; }
    const std::size_t n_tokens = spec.tokens.size();
    std::printf("spec: tokens=%zu n_heads=%d row_dim=%d eos=%d\n", n_tokens, spec.n_heads,
                spec.row_dim, spec.eos);

    ninfer::ops::ple::PleTableOptions options;
    options.sidecar_root = sidecar;
    ninfer::ops::ple::PleTable table(options);
    const auto& layout = table.layout();
    if (static_cast<int>(layout.n_heads) != spec.n_heads ||
        static_cast<int>(layout.embedding_row_dimension) != spec.row_dim ||
        static_cast<int>(layout.ngram_size) != spec.ngram) {
        std::fprintf(stderr, "layout/spec mismatch\n");
        return 2;
    }

    // Host-side row derivation through the engine's own code path.
    std::vector<std::int32_t> prevs(n_tokens * (spec.ngram - 1), spec.eos);
    std::vector<std::int32_t> rows(n_tokens * spec.n_heads);
    table.derive_rows(std::span<const std::int32_t>(spec.tokens),
                      std::span<const std::int32_t>(prevs), spec.eos, rows.data());
    std::printf("rows[0][0..3] = %d %d %d %d\n", rows[0], rows[1], rows[2], rows[3]);

    // Device gather through PleTable (sidecar faults satisfied synchronously).
    const std::size_t payload_bytes =
        n_tokens * static_cast<std::size_t>(spec.n_heads) * spec.row_stride;
    void* d_dst = nullptr;
    if (cudaMalloc(&d_dst, payload_bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(%zu) failed\n", payload_bytes);
        return 2;
    }
    table.gather(rows.data(), n_tokens, d_dst, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "gather kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    std::vector<std::uint8_t> host(payload_bytes);
    cudaMemcpy(host.data(), d_dst, payload_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dst);

    const std::uint64_t fnv = fnv1a64(host.data(), host.size(), kFnvOffset);
    std::printf("payload: %zu bytes  fnv1a64=0x%016llx\n", host.size(),
                static_cast<unsigned long long>(fnv));

    bool ok = true;
    if (!expect_fnv.empty()) {
        const std::uint64_t expected = std::strtoull(expect_fnv.c_str(), nullptr, 0);
        ok = (expected == fnv);
        std::printf("expect-fnv: %s (expected=0x%016llx)\n", ok ? "MATCH" : "MISMATCH",
                    static_cast<unsigned long long>(expected));
    }

    // Second gather must hit the pinned cache and produce identical bytes.
    void* d_dst2 = nullptr;
    cudaMalloc(&d_dst2, payload_bytes);
    table.gather(rows.data(), n_tokens, d_dst2, nullptr);
    cudaDeviceSynchronize();
    std::vector<std::uint8_t> host2(payload_bytes);
    cudaMemcpy(host2.data(), d_dst2, payload_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dst2);
    const bool cache_hit_ok = std::memcmp(host.data(), host2.data(), payload_bytes) == 0;
    std::printf("cache-hit replay: %s\n", cache_hit_ok ? "identical" : "MISMATCH");
    ok = ok && cache_hit_ok;

    std::printf("%s\n", ok ? "PLE_TABLE_TEST PASS" : "PLE_TABLE_TEST FAIL");
    return ok ? 0 : 1;
}
