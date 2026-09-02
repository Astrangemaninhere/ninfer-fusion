// PLE real-sidecar probe: validate PleLayout/PleTable against the actual
// Qwen3.8-Flash-Next PLE sidecar (Baekpica SSD-PLE). Usage:
//   ninfer_ple_probe <sidecar_root> <tokens...>
// Verifies manifest parse, row derivation, row location and a UVA gather
// round-trip with finite, non-degenerate data.
#include "ops/ple/ple_layout.h"
#include "ops/ple/ple_table.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: ninfer_ple_probe <sidecar_root> <token> [token...]\n");
        return 2;
    }
    const std::filesystem::path root = argv[1];
    std::vector<int32_t> tokens;
    for (int i = 2; i < argc; ++i) { tokens.push_back(std::atoi(argv[i])); }

    ninfer::ops::ple::PleLayout layout;
    try {
        layout = ninfer::ops::ple::PleLayout::from_manifest((root / "ple-manifest.json").string());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "manifest parse FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("layout: %u heads, %u rows/part, %u physical files, row_stride %u\n",
                layout.n_heads, layout.logical_parts.empty() ? 0U : layout.logical_parts[0].rows,
                static_cast<unsigned>(layout.physical_files.size()), layout.row_stride_bytes);
    std::printf("multipliers: %llu %llu %llu\n",
                (unsigned long long)layout.layer_multipliers[0],
                (unsigned long long)layout.layer_multipliers[1],
                (unsigned long long)layout.layer_multipliers[2]);

    // derive rows (sequence start: all predecessors = eos = 151643 for Qwen3.8)
    const int32_t eos = 151643;
    std::vector<int32_t> prevs(tokens.size() * 2, eos);
    std::vector<int32_t> rows(tokens.size() * layout.n_heads);
    layout.derive_rows(tokens, prevs, eos, rows.data());
    std::printf("derived rows (first token):");
    for (unsigned h = 0; h < layout.n_heads; ++h) {
        std::printf(" %d", rows[h]);
    }
    std::printf("\n");

    // row_location sanity: every derived row must resolve within a file
    for (size_t i = 0; i < rows.size(); ++i) {
        uint32_t fi = 0;
        uint64_t off = 0;
        if (!layout.row_location((uint64_t)rows[i], fi, off)) {
            std::fprintf(stderr, "row %d out of range at token %zu head %zu\n", rows[i], i / 16,
                         i % 16);
            return 1;
        }
        if (fi >= layout.physical_files.size()) {
            std::fprintf(stderr, "row %d -> bad file %u\n", rows[i], fi);
            return 1;
        }
    }
    std::printf("row location OK (%zu rows)\n", rows.size());

    // gather round-trip on the real 25.6 GiB sidecar
    ninfer::ops::ple::PleTableOptions opts;
    opts.sidecar_root = root;
    opts.cache_bytes = 64ULL << 20;
    ninfer::ops::ple::PleTable table(opts);

    const int row_dim = (int)layout.embedding_row_dimension;
    __half* dst = nullptr;
    cudaMalloc(&dst, tokens.size() * layout.n_heads * row_dim * sizeof(__half));
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    table.gather(rows.data(), tokens.size(), dst, stream);
    cudaStreamSynchronize(stream);

    std::vector<__half> host(tokens.size() * layout.n_heads * row_dim);
    cudaMemcpy(host.data(), dst, host.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(dst);
    cudaStreamDestroy(stream);

    // validate: values must be finite and non-degenerate (not all zero)
    unsigned nonzero = 0;
    float min_abs = 1e30f, max_abs = 0.0f;
    for (float v : host) {
        const float a = v == v ? (v < 0 ? -v : v) : 1e30f; // NaN guard
        if (a != a || a > 1e30f) {
            std::fprintf(stderr, "NaN in gathered PLE data!\n");
            return 1;
        }
        if (a != 0.0f) { ++nonzero; }
        if (a < min_abs) { min_abs = a; }
        if (a > max_abs) { max_abs = a; }
    }
    std::printf("gather OK: %u/%zu nonzero, |v| in [%.4g, %.4g]\n", nonzero, host.size(),
                min_abs, max_abs);
    return nonzero == 0 ? 1 : 0;
}
