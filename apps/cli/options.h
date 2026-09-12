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
    bool kv_layer_storage_explicit = false;
    ColdPolicy cold_policy        = ColdPolicy::None;
    std::uint32_t cold_keep_tokens = 128;
    std::uint64_t cold_host_bytes  = 4ULL << 30;
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
