// Roundtrip CPU check: build a tiny PLE sidecar with ple_sidecar_build.py,
// then parse it with the engine's own PleLayout::from_manifest.
#include "ops/ple/ple_layout.h"

#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: ple_rt <manifest>\n"); return 2; }
    try {
        ninfer::ops::ple::PleLayout l = ninfer::ops::ple::PleLayout::from_manifest(argv[1]);
        std::printf("format=%u ngram=%u heads_per=%u n_heads=%u row_dim=%u "
                    "stride=%u padded=%llu usable=%llu parts=%zu files=%zu\n",
                    l.format_version, l.ngram_size, l.heads_per_ngram, l.n_heads,
                    l.embedding_row_dimension, l.row_stride_bytes,
                    (unsigned long long)l.padded_vocabulary_rows,
                    (unsigned long long)l.usable_vocabulary_rows,
                    l.logical_parts.size(), l.physical_files.size());
        for (const auto& f : l.physical_files) {
            std::printf("  file[%u] %s bytes=%llu payload=%llu\n", f.index, f.path.c_str(),
                        (unsigned long long)f.file_bytes,
                        (unsigned long long)f.payload_bytes);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "parse failed: %s\n", e.what());
        return 1;
    }
}
