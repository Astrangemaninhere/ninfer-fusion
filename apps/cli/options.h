#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t prefill_chunk  = 3072;
    int device                   = 0;

    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    bool kv_cache_explicit  = false;
    // Unspecified --spec means auto: the artifact's own draft backend (DFlash2 or MTP)
    // is used. Measured on code: 270.2 tok/s auto vs 69.1 tok/s with speculation off.
    // `--spec none` opts out.
    SpeculativeOptions speculative{SpeculativeBackend::Auto};
    bool enable_vision  = false;
    bool use_cuda_graph = true;
    std::string kv_layer_storage_spec;
    // KV bit budget: a single ceiling per element, or separable per-range ceilings
    // ("0-7:8,8-63:4.5"); the DP never exceeds them and minimises the penalty inside them.
    double kv_bit_budget_bits = 0.0;
    std::string kv_bit_budget_ranges;
    bool kv_bit_budget_explicit = false;
    // Two-score KV selection: 0 = fastest, 1 = most accurate; negative leaves the shipped
    // single-penalty ladder in place. --kv-tier-scores overrides the score table.
    double kv_quality_weight = -1.0;
    std::string kv_tier_scores;
    bool kv_layer_storage_explicit = false;
    // --kv-tier-formats SPEC + --nvfp4-mode: the KV tier vocabulary (kvcfg/kv_formats.h).
    // Stored raw and resolved in the planner, where the layer count is known; the parse
    // site only checks the vocabulary's own rules (product::kv_tier_formats_parse).
    std::string kv_tier_formats_spec;
    bool kv_tier_formats_explicit = false;
    bool kv_nvfp4_pure            = false;
    // SEPARATION: the three KV component switches (include/ninfer/types.h).
    bool kv_rotation_off          = false;
    bool kv_rotation_explicit     = false;
    std::string kv_row_scale_spec;
    bool kv_row_scale_explicit    = false;
    KvVCodec kv_v_codec           = KvVCodec::Iso3;
    bool kv_v_codec_explicit      = false;
    ColdPolicy cold_policy        = ColdPolicy::None;
    std::uint32_t cold_keep_tokens          = 128;
    bool cold_keep_tokens_explicit          = false;
    std::uint64_t cold_host_bytes  = 4ULL << 30;
    // --max-cold-pages: explicit cold-pool cap in pages (0 = derive from the
    // policy). Without it the CLI could only ever use the derived pool
    // (cold_keep_tokens/kPagedKVPageSize + 16), i.e. 18 pages, which is the whole
    // point of "cap the offload" being unreachable from this front end.
    std::uint32_t max_cold_pages   = 0;
    // ColdPolicy::Disk spill budget and directory (serve-only flags before).
    std::uint64_t cold_disk_bytes  = 32ULL << 30;
    std::string cold_disk_path;
    bool yarn_enabled     = false;
    std::uint32_t graph_capture_ceiling = 16;

    bool raw_output      = false;
    bool print_token_ids = false;
    bool enable_thinking = true;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy = false;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::cli
