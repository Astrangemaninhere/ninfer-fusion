#include "ops/ple/ple_layout.h"
#include "ops/ple/ple_stage.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <optional>
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

// ----------------------------------------------------------------
// PleStageDecl (ops/ple/ple_stage.h). Defined here because this TU
// already pays for <nlohmann/json.hpp>; including ple_stage.h therefore
// never adds a JSON parse to a heavy translation unit.
// ----------------------------------------------------------------

void PleStageDecl::validate() const {
    if (!present) { return; }
    if (ngram_size < 2) {
        throw std::invalid_argument("PLE declaration: ngram_size must be >= 2");
    }
    if (heads_per_ngram == 0) {
        throw std::invalid_argument("PLE declaration: heads_per_ngram must be > 0");
    }
    if (embed_dim == 0) {
        throw std::invalid_argument("PLE declaration: ple_embed_dim must be > 0");
    }
}

void PleStageDecl::validate_against(std::uint32_t sidecar_ngram_size,
                                    std::uint32_t sidecar_heads_per_ngram,
                                    std::uint32_t sidecar_n_heads,
                                    std::uint32_t sidecar_row_dim) const {
    if (!present) { return; }
    const auto mismatch = [](const char* field, std::uint64_t declared,
                             std::uint64_t sidecar) {
        return std::string("PLE declaration/sidecar mismatch on ") + field +
               ": declaration=" + std::to_string(declared) +
               " sidecar=" + std::to_string(sidecar);
    };
    if (sidecar_ngram_size != ngram_size) {
        throw std::runtime_error(mismatch("ngram_size", ngram_size, sidecar_ngram_size));
    }
    if (sidecar_heads_per_ngram != heads_per_ngram) {
        throw std::runtime_error(mismatch("heads_per_ngram", heads_per_ngram,
                                          sidecar_heads_per_ngram));
    }
    if (sidecar_n_heads != n_heads()) {
        throw std::runtime_error(mismatch("number_of_ngram_heads", n_heads(), sidecar_n_heads));
    }
    if (embed_dim != static_cast<std::uint64_t>(sidecar_n_heads) * sidecar_row_dim) {
        throw std::runtime_error(
            mismatch("ple_embed_dim", embed_dim,
                     static_cast<std::uint64_t>(sidecar_n_heads) * sidecar_row_dim));
    }
}

PleStageDecl PleStageDecl::from_spec_text(const std::string& spec_text,
                                         const std::string& origin) {
    nlohmann::json doc;
    try {
        doc = nlohmann::json::parse(spec_text);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("PLE declaration parse failed (" + origin +
                                 "): " + e.what());
    }

    PleStageDecl out;
    const auto block = doc.find("ple");
    if (block == doc.end() || block->is_null()) {
        // A spec without a "ple" block is a valid declaration that no layer
        // carries a sidecar: the stage stays off and costs one bool test.
        return out;
    }
    if (!block->is_object()) {
        throw std::runtime_error("PLE declaration: \"ple\" must be an object (" +
                                 origin + ")");
    }

    const auto number = [&](const char* key, std::uint32_t fallback) -> std::uint32_t {
        const auto it = block->find(key);
        if (it == block->end() || it->is_null()) { return fallback; }
        if (!it->is_number_unsigned() && !it->is_number_integer()) {
            throw std::runtime_error(std::string("PLE declaration: ") + key +
                                     " must be an integer (" + origin + ")");
        }
        return it->get<std::uint32_t>();
    };
    out.ngram_size      = number("ngram_size", out.ngram_size);
    out.heads_per_ngram = number("heads_per_ngram", out.heads_per_ngram);
    out.embed_dim       = number("ple_embed_dim", out.embed_dim);

    if (const auto it = block->find("ple_layer_ids"); it != block->end() && !it->is_null()) {
        if (!it->is_array()) {
            throw std::runtime_error(
                "PLE declaration: ple_layer_ids must be an array (" + origin + ")");
        }
        for (const auto& entry : *it) {
            if (!entry.is_number_integer()) {
                throw std::runtime_error(
                    "PLE declaration: ple_layer_ids entries must be integers (" + origin +
                    ")");
            }
            out.layer_ids.push_back(entry.get<std::uint32_t>());
        }
    }

    // eos: the arch-spec "ple" block does not carry it today, so accept it
    // either there or at the document root; 0 means "caller supplies it".
    const auto eos_in = [&](const nlohmann::json& node) -> std::optional<std::uint32_t> {
        const auto it = node.find("eos_token_id");
        if (it == node.end() || it->is_null()) { return std::nullopt; }
        if (!it->is_number_integer()) {
            throw std::runtime_error("PLE declaration: eos_token_id must be an integer (" +
                                     origin + ")");
        }
        return it->get<std::uint32_t>();
    };
    if (const auto from_ple = eos_in(*block)) {
        out.eos_token_id = *from_ple;
    } else if (const auto from_root = eos_in(doc)) {
        out.eos_token_id = *from_root;
    }

    // "Declared" means some layer actually carries the stage. A ple block
    // whose layer list is empty or absent declares absence, not work: keeping
    // `present` false keeps the no-op path exact.
    out.present = !out.layer_ids.empty();
    return out;
}

PleStageDecl PleStageDecl::from_spec_file(const std::string& spec_path) {
    std::ifstream in(spec_path, std::ios::binary);
    if (!in) { throw std::runtime_error("PLE declaration open failed: " + spec_path); }
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    return from_spec_text(text, spec_path);
}

} // namespace ninfer::ops::ple
