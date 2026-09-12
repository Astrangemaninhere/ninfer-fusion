#pragma once

// PLE n-gram table layout and host-side row derivation for the Qwen4Exp
// (Qwen3.8-Flash-Next) predictive-latent-embedding layer.
//
// The PLE table is a plain row-lookup structure: 320,001,536 rows x 160
// BF16 columns, shipped as a 4-file sidecar (Baekpica SSD-PLE layout, see
// RESEARCH-FLASHNEXT.md). Each token gathers 16 rows (8 bigram + 8 trigram
// heads); the rows are derived host-side, so the table can live entirely on
// SSD and never occupy device memory.
//
// Row formula (verbatim from llama.cpp src/models/qwen4exp.cpp, PR #27742):
//   bigram  heads 0..7  : mixed = ctx0*m0 ^ ctx1*m1
//   trigram heads 8..15 : mixed = ctx0*m0 ^ ctx1*m1 ^ ctx2*m2
//   row[h] = mixed % per_head_vocab_size[h] + per_head_offset[h]
// All multiplies happen in uint64 (wrap-around is intentional). An EOS in
// the n-gram window resets everything at or before it; a missing predecessor
// (sequence start) reads as EOS.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ninfer::ops::ple {

struct PleLogicalPart {
    std::uint32_t logical_part           = 0;
    std::uint32_t physical_file_index    = 0;
    std::uint64_t global_row_start       = 0; // first padded row covered by this part
    std::uint64_t file_offset            = 0; // bytes within the physical file (4096-aligned)
    std::uint64_t rows                   = 0;
    std::uint64_t payload_bytes          = 0;
};

struct PlePhysicalFile {
    std::uint32_t index                  = 0;
    std::string   path;                       // relative to the sidecar root, e.g. "ple/ple-bf16-00001-of-00004.bin"
    std::uint64_t file_bytes             = 0;
    std::uint64_t payload_bytes          = 0;
};

// Parsed ple-manifest.json (format_version 1, Baekpica SSD-PLE sidecar).
struct PleLayout {
    std::uint32_t format_version             = 1;
    std::uint32_t ngram_size                 = 3;
    std::uint32_t heads_per_ngram            = 8;
    std::uint32_t n_heads                    = 16;
    std::uint32_t embedding_row_dimension    = 160;
    std::uint32_t row_stride_bytes           = 320; // BF16 x 160
    std::uint64_t padded_vocabulary_rows     = 320001536ULL;
    std::uint64_t usable_vocabulary_rows     = 320001446ULL;
    std::uint64_t total_parameter_count      = 0;
    std::uint64_t alignment_bytes            = 4096;
    std::uint64_t layer_multipliers[4]       = {23703573157769ULL, 20109073645365ULL,
                                                8052911324071ULL, 0};
    std::vector<std::uint32_t> per_head_offsets;      // size n_heads
    std::vector<std::uint32_t> per_head_vocab_sizes;  // size n_heads
    std::vector<PleLogicalPart> logical_parts;        // size 128
    std::vector<PlePhysicalFile> physical_files;      // size 4

    // Loads and validates ple-manifest.json. Throws std::runtime_error on
    // structural mismatch (wrong field names/types, inconsistent geometry).
    static PleLayout from_manifest(const std::string& manifest_path);

    // Resolves a global padded row index to (physical file index, byte offset).
    // Returns false when row is outside [0, padded_vocabulary_rows).
    bool row_location(std::uint64_t row, std::uint32_t& file_index,
                      std::uint64_t& byte_offset) const noexcept;

    // Derives the 16 row ids for one token.
    //
    //   ctx0        : the token itself
    //   prevs       : n_gram-1 predecessors, oldest first; pass eos for a
    //                 missing predecessor or after an EOS cut
    //   rows_out    : 16 outputs (bigram heads 0..7, trigram heads 8..15)
    void derive_rows_one(std::int32_t ctx0, const std::int32_t* prevs,
                         std::int32_t eos, std::int32_t* rows_out) const noexcept;

    // Derives 16*n_tokens rows for a batch. `prevs` is (n_gram-1)*n_tokens,
    // token-major, oldest-first per token; entries at or after an EOS cut are
    // ignored (the cut replaces them with eos internally). rows_out must hold
    // n_heads*n_tokens entries.
    void derive_rows(std::span<const std::int32_t> tokens,
                     std::span<const std::int32_t> prevs, std::int32_t eos,
                     std::int32_t* rows_out) const noexcept;
};

} // namespace ninfer::ops::ple
