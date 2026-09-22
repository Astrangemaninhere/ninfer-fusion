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
    // --prefill-chunk-mode dynamic|manual (PrefillChunkMode, ninfer/types.h). Unset is not Manual:
    // it defers to NINFER_FT_BW_GOV and then to Dynamic, resolved once by the engine. Both front
    // ends spell the mode the same way because both read the vocabulary from
    // runtime/engine/bandwidth_governor.h (parse_mode), which also owns the mode -> mechanism
    // mapping -- the server's handle on the prefill unit and the CLI's are the same handle.
    std::optional<PrefillChunkMode> prefill_chunk_mode;
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
    // Speed/quality slider for the bit-budget fit (product/kv_bit_budget.h
    // kv_bit_budget_scored_ladder). It was reachable from apps/cli only, so the
    // server -- the front end the operator actually runs -- had no handle at all,
    // and inside apps/cli it was inert without a ceiling. -1 == not given.
    double kv_quality_weight              = -1.0;
    // Inline score table or a file path (same three cases the planner distinguishes).
    std::string kv_tier_scores;
    // ---- K/V bit widths: the TWO entry points of product/kv_kv_bits.h ----------
    // --kv-bits is the JOINT form (ONE overall ceiling, "合起来整体定");
    // --kv-k-bits/--kv-v-bits the SPLIT form ("分开定，内部分层"). 0 == not named.
    double kv_joint_bits                  = 0.0;
    double kv_k_bits                      = 0.0;
    double kv_v_bits                      = 0.0;
    bool kv_kv_bits_explicit              = false;
    KvBitsMode kv_bits_mode               = KvBitsMode::Split;
    bool kv_bits_mode_explicit            = false;
    std::string kv_k_tier_scores;
    std::string kv_v_tier_scores;
    // --kv-score-table show|emit=<path>: the penalty table's own entry point.
    std::string kv_score_table_spec;
    bool kv_score_table_explicit          = false;
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
    // --kv-v-codec iso4e|e2m1. All three default to the pre-separation
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
    // --kv-unload-watermark-pages: free text-KV pool pages at or below which the
    // Engine proactively unloads the blocks its semantic directory judges
    // unloadable (see EngineOptions::unload_watermark_pages for the full rule).
    // 0 = OFF and off is the pre-watermark behaviour; the sentinel derives the
    // reserve from the plan's own prefill chunk.
    std::uint32_t unload_watermark_pages = kUnloadWatermarkDerive;
    // 7 GiB, not 4: the 1M band needs F - D = 5,634 pages = 6.2004 GiB at
    // 1,181,745 B/page, and `4ULL << 30` leaves cold_host_window_band EMPTY.
    std::uint64_t cold_host_bytes  = 7ULL << 30;
    std::string cold_disk_path;
    std::uint64_t cold_disk_bytes = 32ULL << 30;
    std::uint64_t weight_host_offload_bytes = 0; // --weight-host-bytes: 0 = off
    std::uint64_t weight_device_arena_bytes = 0; // --weight-device-arena-bytes: 0 = derive
    std::uint32_t weight_prefetch_layers    = 2; // --weight-prefetch-layers
    std::uint64_t weight_span_floor_bytes   = 0; // --weight-span-floor-bytes: 0 = derive
    // ---- FreeToken line (src/serve/kv_auto_relayout.h) ----------------------
    // The periodic KV relayout is OPT-IN and unset means "defer", so the
    // precedence is CLI > env > default with the default interval 0 = disabled
    // (exactly the behaviour of every build before these flags existed):
    //   --kv-auto-relayout SECS  >  NINFER_FT_RELOAD_SECS > 0  >  off
    std::optional<int> ft_relayout_secs;
    // Gap 1's free-VRAM axis inside that decision. An unset value defers to
    // NINFER_FT_VRAM_AXIS (a kill switch), then to "on" -- which is inert until
    // the relayout itself is enabled, because nothing calls the probe then.
    std::optional<bool> ft_vram_axis;
    // FreeToken step 1 observation (src/ops/common/ft_stats.h). An unset value
    // keeps NINFER_FT_STATS as the operator set it; `--ft-stats on|off` is
    // committed to that same variable at parse time, because ft::enabled()
    // reads it exactly once (see the note in serve_options.cpp).
    std::optional<bool> ft_stats;
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
    // Which slots the --kv-layer-storage spec actually wrote, for kv_layer_storage
    // above (see EngineOptions::kv_layer_storage_set). Without it GenerationService
    // cannot hand the engine a per-layer bf16 request.
    // LAST MEMBER ON PURPOSE: appending cannot move an existing field's offset.
    std::array<bool, 64> kv_layer_storage_set{};
};

ServeOptions parse_serve_options(int argc, char** argv);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id);
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
