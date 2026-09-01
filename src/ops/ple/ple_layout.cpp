#include "ops/ple/ple_layout.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace ninfer::ops::ple {
namespace {

template <class T>
T require(const nlohmann::json& node, const char* key) {
    auto it = node.find(key);
    if (it == node.end()) {
        throw std::runtime_error(std::string("PLE manifest missing key: ") + key);
    }
    try {
        return it->get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("PLE manifest bad type for key: ") + key);
    }
}

} // namespace

PleLayout PleLayout::from_manifest(const std::string& manifest_path) {
    nlohmann::json doc;
    {
        std::ifstream in(manifest_path, std::ios::binary);
        if (!in) { throw std::runtime_error("PLE manifest open failed: " + manifest_path); }
        try {
            in >> doc;
        } catch (const nlohmann::json::exception& e) {
            throw std::runtime_error("PLE manifest parse failed: " + std::string(e.what()));
        }
    }

    PleLayout out;
    out.format_version          = require<std::uint32_t>(doc, "format_version");
    out.ngram_size              = require<std::uint32_t>(doc, "ngram_size");
    out.heads_per_ngram         = require<std::uint32_t>(doc, "heads_per_ngram");
    out.n_heads                 = require<std::uint32_t>(doc, "number_of_ngram_heads");
    out.embedding_row_dimension = require<std::uint32_t>(doc, "embedding_row_dimension");
    out.row_stride_bytes        = require<std::uint32_t>(doc, "row_stride_bytes");
    out.padded_vocabulary_rows  = require<std::uint64_t>(doc, "padded_vocabulary_rows");
    out.usable_vocabulary_rows  = require<std::uint64_t>(doc, "usable_vocabulary_rows");
    out.total_parameter_count   = require<std::uint64_t>(doc, "total_parameter_count");
    out.alignment_bytes         = require<std::uint64_t>(doc, "alignment_bytes");

    const auto multipliers = require<std::vector<std::uint64_t>>(doc, "layer_multipliers");
    if (multipliers.size() != out.ngram_size) {
        throw std::runtime_error("PLE manifest: layer_multipliers size != ngram_size");
    }
    for (std::size_t i = 0; i < multipliers.size(); ++i) {
        out.layer_multipliers[i] = multipliers[i];
    }

    out.per_head_offsets      = require<std::vector<std::uint32_t>>(doc, "per_head_offsets");
    out.per_head_vocab_sizes  = require<std::vector<std::uint32_t>>(doc, "per_head_vocabulary_sizes");
    if (out.per_head_offsets.size() != out.n_heads ||
        out.per_head_vocab_sizes.size() != out.n_heads) {
        throw std::runtime_error("PLE manifest: per-head tables do not match head count");
    }

    for (const auto& part : require<std::vector<nlohmann::json>>(doc, "logical_parts")) {
        PleLogicalPart p;
        p.logical_part        = require<std::uint32_t>(part, "logical_part");
        p.physical_file_index = require<std::uint32_t>(part, "physical_file_index");
        p.global_row_start    = require<std::uint64_t>(part, "global_row_start");
        p.file_offset         = require<std::uint64_t>(part, "file_offset");
        p.rows                = require<std::uint64_t>(part, "rows");
        p.payload_bytes       = require<std::uint64_t>(part, "payload_bytes");
        out.logical_parts.push_back(p);
    }
    for (const auto& file : require<std::vector<nlohmann::json>>(doc, "physical_files")) {
        PlePhysicalFile f;
        f.index         = require<std::uint32_t>(file, "index");
        f.path          = require<std::string>(file, "path");
        f.file_bytes    = require<std::uint64_t>(file, "file_bytes");
        f.payload_bytes = require<std::uint64_t>(file, "payload_bytes");
        out.physical_files.push_back(f);
    }
    if (out.logical_parts.empty() || out.physical_files.empty()) {
        throw std::runtime_error("PLE manifest: empty sidecar layout");
    }
    // Parts must be ordered by global_row_start (row_location binary-searches).
    std::uint64_t prev_start = 0;
    for (const auto& p : out.logical_parts) {
        if (p.global_row_start < prev_start) {
            throw std::runtime_error("PLE manifest: logical parts not ordered by global_row_start");
        }
        prev_start = p.global_row_start + p.rows;
    }
    if (prev_start != out.padded_vocabulary_rows) {
        throw std::runtime_error("PLE manifest: part coverage != padded_vocabulary_rows");
    }
    return out;
}

bool PleLayout::row_location(std::uint64_t row, std::uint32_t& file_index,
                             std::uint64_t& byte_offset) const noexcept {
    if (row >= padded_vocabulary_rows) { return false; }
    // Parts are ordered by global_row_start; find the last part whose start
    // is <= row (owner), then index within the part. file_offset is relative
    // to the part's physical file.
    const auto it = std::upper_bound(
        logical_parts.begin(), logical_parts.end(), row,
        [](std::uint64_t r, const PleLogicalPart& p) { return r < p.global_row_start; });
    if (it == logical_parts.begin()) { return false; }
    const PleLogicalPart& part = *std::prev(it);
    const std::uint64_t row_in_part = row - part.global_row_start;
    if (row_in_part >= part.rows) { return false; }
    file_index  = part.physical_file_index;
    byte_offset = part.file_offset + row_in_part * row_stride_bytes;
    return true;
}

void PleLayout::derive_rows_one(std::int32_t ctx0, const std::int32_t* prevs,
                                std::int32_t eos, std::int32_t* rows_out) const noexcept {
    // Window: ctx[0] = current token; ctx[s] = predecessor s positions back,
    // an EOS cut replaces everything at/after the cut with eos.
    std::int64_t ctx[4] = {ctx0, eos, eos, eos};
    bool cut = false;
    for (std::uint32_t s = 1; s < ngram_size; ++s) {
        const std::int32_t t = prevs[s - 1];
        cut = cut || t < 0 || t == eos;
        ctx[s] = cut ? eos : t;
    }
    for (std::uint32_t n = 2; n <= ngram_size; ++n) {
        std::uint64_t mixed = static_cast<std::uint64_t>(ctx[0]) * layer_multipliers[0];
        for (std::uint32_t j = 1; j < n; ++j) {
            mixed ^= static_cast<std::uint64_t>(ctx[j]) * layer_multipliers[j];
        }
        const std::uint32_t base = (n - 2) * heads_per_ngram;
        for (std::uint32_t g = 0; g < heads_per_ngram; ++g) {
            const std::uint32_t h = base + g;
            rows_out[h] = static_cast<std::int32_t>(mixed % per_head_vocab_sizes[h] +
                                                    per_head_offsets[h]);
        }
    }
}

void PleLayout::derive_rows(std::span<const std::int32_t> tokens,
                            std::span<const std::int32_t> prevs, std::int32_t eos,
                            std::int32_t* rows_out) const noexcept {
    const std::uint32_t n_prev = ngram_size - 1;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        derive_rows_one(tokens[i], prevs.data() + i * n_prev, eos,
                        rows_out + i * n_heads);
    }
}

} // namespace ninfer::ops::ple
