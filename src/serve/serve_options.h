#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens                    = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact identity.model_id
    std::string request_log_jsonl;                // empty => structured request logging disabled
    std::uint32_t max_context          = 8192;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(8192);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    std::filesystem::path context_cost_presets;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    int device                             = 0;
    KvCacheStorage kv_cache                = KvCacheStorage::BFloat16;
    bool kv_cache_explicit                 = false;
    std::array<KvCacheStorage, 64> kv_layer_storage{};
    bool kv_layer_storage_explicit        = false;
    double kv_bit_budget_bits             = 0.0; // --kv-bit-budget, resolved at planner time
    bool kv_bit_budget_explicit           = false;
    // Separable per-range ceilings ("0-7:8,8-63:4.5"); empty means the scalar form above.
    std::string kv_bit_budget_ranges;
    // --kv-tier-formats SPEC + --nvfp4-mode (kvcfg/kv_formats.h vocabulary). Raw text:
    // the per-layer landing needs the model's layer count, so it happens in the planner
    // (product/kv_tier_formats.h documents which tiers the engine can express).
    std::string kv_tier_formats_spec;
    bool kv_tier_formats_explicit         = false;
    bool kv_nvfp4_pure                    = false;
    // SEPARATION: the three KV component switches (same wire as the engine
    // options). --kv-rotation on|off, --kv-row-scale auto|off|<path>,
    // --kv-v-codec iso3|e2m1. All three default to the pre-separation
    // behaviour and are committed to the device in plan_decoder_state().
    bool kv_rotation_off                  = false;
    bool kv_rotation_explicit             = false;
    std::string kv_row_scale_spec;
    bool kv_row_scale_explicit            = false;
    KvVCodec kv_v_codec                   = KvVCodec::Iso3;
    bool kv_v_codec_explicit              = false;
    // Indexed like EngineOptions::kv_residual_layers / layouts.h's 64-slot
    // per-layer tables; a 16-slot array here would reject --kv-residual-layers
    // 16 as out of range while the planner accepts 64.
    std::array<bool, kKvLayerStorageSlots> kv_residual_layers{};
    bool kv_residual_explicit             = false;
    // Serve keeps speculation OFF unless asked for (tests/test_serve_options.cpp asserts it):
    // a server operator opts in with --spec auto|mtp|dflash|dflash2. The CLI, by contrast,
    // defaults --spec to auto because an interactive run always wants the draft backend.
    SpeculativeOptions speculative;
    ContextCacheOptions context_cache;
    bool enable_vision      = false;
    bool use_cuda_graph     = true;
    bool yarn_enabled       = false;
    bool allow_prefix_reuse = true;
    // Issue #142: publish a shared-prefix candidate at the leading
    // system/developer frontier (default on for agent workloads).
    bool auto_system_shared_prefix = true;
    ColdPolicy cold_policy        = ColdPolicy::None;
    std::uint32_t cold_keep_tokens = 128;
    std::uint32_t max_cold_pages   = 0; // --max-cold-pages: 0 = policy-derived
    std::uint64_t cold_host_bytes  = 4ULL << 30;
    std::string cold_disk_path;
    std::uint64_t cold_disk_bytes = 32ULL << 30;
    std::uint64_t weight_host_offload_bytes = 0; // W13 P0; flag is loud-rejected below
    bool enable_thinking =
        true; // default thinking mode for the generation prompt (--no-thinking opts out)
    bool preserve_thinking = false;
    std::optional<std::uint32_t> default_thinking_budget;
    int default_max_tokens = kDefaultMaxTokens;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy = false; // --greedy: force temperature 0 (exact argmax)

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id);
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
