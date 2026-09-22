#include "product/kv_options.h"
#include "product/kv_tier_formats.h"
#include "product/kv_kv_bits.h"
#include "runtime/engine/bandwidth_governor.h"
#include "serve/serve_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (value == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (value == "iso4e") { return KvCacheStorage::Iso3Group16; }
    if (value == "rk4v4") { return KvCacheStorage::E8Group64; }
    // The rk4v4 family's narrower K planes. Accepted HERE so the spelling is symmetric with
    // the vocabulary and with --kv-layer-storage, and refused one step later by
    // product::kv_dtype_for_storage, which is the single decision point that knows why.
    if (value == "rk3v4") { return KvCacheStorage::E8K3Group64; }
    if (value == "rk2v4") { return KvCacheStorage::E8K2Group64; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

bool parse_on_off(const char* text, const char* flag) {
    const std::string_view mode(text);
    if (mode == "on") { return true; }
    if (mode == "off") { return false; }
    throw std::invalid_argument("invalid " + std::string(flag) + ": " + std::string(mode) +
                                " (expected on|off)");
}

// FreeToken observation switch. ops::ft::enabled() caches NINFER_FT_STATS in a
// function-local static the first time a kernel asks
// (src/ops/common/ft_stats.h:32-36), so "CLI beats env" for this switch can only
// mean: write the variable before the engine runs. Parsing happens long before
// the first prefill, so committing it here is correct and is the same trade
// src/product/kv_rowscale_persist.h:609-626 makes for NINFER_KV_CALIB_DIR.
int set_process_env(const char* name, const char* value) {
#if defined(_WIN32)
    return ::_putenv_s(name, value);
#else
    return ::setenv(name, value, 1);
#endif
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> [--host H] [--port N] [--api-key KEY] "
           "[--model-id ID] [--max-context N] [--kv-capacity N|auto] [--max-concurrency N] "
           "[--max-pending-requests N] [--pending-timeout-ms N] "
           "[--prefill-chunk N] [--prefill-chunk-mode dynamic|manual] "
           "[--log-stats-interval-ms N] [--device N] "
           "[--context-cost-presets FILE] "
           "[--max-request-mib N] [--media-cache-mib N] [--media-live-mib N] "
           "[--media-preprocess-threads N] "
           "[--device-state-slots N] [--host-state-slots N] [--host-kv-mib N] "
           "[--max-private-continuations N] [--max-shared-prefixes N] "
           "[--max-long-anchors-per-continuation N] "
           "[--request-log-jsonl FILE] "
           "[--response-store-max-records N] [--response-store-max-mib N] "
           "[--kv-dtype bf16|int8|fp8|nvfp4|iso4e|rk4v4|rk3v4|rk2v4] [--kv-tier-formats SPEC] [--nvfp4-mode fusion|pure] "
           "[--kv-rotation on|off] [--kv-row-scale auto|off|FILE] "
           "[--kv-v-codec iso4e|e2m1] "
           "[--kv-bits B] [--kv-bit-budget SCALAR|RANGES] [--kv-k-bits BK --kv-v-bits BV] "
           "[--kv-bits-mode joint|split|ceiling] [--kv-layer-storage SPEC] "
           "[--kv-residual-layers SPEC] "
           "[--kv-tier-scores FILE] [--kv-k-tier-scores FILE] [--kv-v-tier-scores FILE] "
           "[--kv-quality-weight W] [--kv-score-table show|emit=PATH] "
           "[--cold-policy none|window|host|disk|host-then-disk] [--cold-keep-tokens N] "
           "[--max-cold-pages N] [--cold-host-bytes N] [--cold-disk-path FILE] "
           "[--cold-disk-bytes N] [--kv-unload-watermark-pages N] "
           "[--weight-host-bytes N] [--weight-device-arena-bytes N] "
           "[--weight-prefetch-layers N] [--weight-span-floor-bytes N] [--yarn] "
           "[--spec mtp|dflash|dflash2|dspark|auto --draft-tokens N] "
           "[--kv-auto-relayout SECS] [--ft-vram-axis on|off] [--ft-stats on|off] "
           "[--default-max-tokens N] [--default-thinking-budget N] "
           "[--vision] [--no-cuda-graph] [--no-prefix-reuse] "
           "[--lm-head-draft] [--no-lm-head-draft] [--draft-tree L,d] "
           "[--no-thinking] [--preserve-thinking] [--no-auto-system-shared-prefix] [--cors] "
           "[--temperature F] [--top-p F] [--top-k N] [--min-p F] [--presence-penalty F] "
           "[--frequency-penalty F] [--seed N] [--greedy]\n"
           "       serves OpenAI Responses/Chat Completions and Anthropic Messages endpoints\n"
           "       --default-max-tokens defaults to " +
           std::to_string(kDefaultMaxTokens) +
           " when omitted\n"
           "       --max-request-mib defaults to 384 and is enforced before JSON parsing\n"
           "       --media-cache-mib defaults to 1024; 0 disables retained media reuse\n"
           "       --media-live-mib defaults to 2048 and bounds all live BF16 patch payloads\n"
           "       --media-preprocess-threads defaults to 0 (auto, at most 16 workers)\n"
           "       --request-log-jsonl appends full-precision server/request records\n"
           "       --model-id overrides the artifact identity.model_id reported by the server\n"
           "       Responses state is process-local and bounded to 1024 records / 256 MiB by "
           "default\n"
           "       --log-stats-interval-ms defaults to 5000; 0 disables periodic throughput logs\n"
           "       --vision enables media and loads the fixed Vision GPU allocations\n"
           "       --kv-tier-formats hot=auto|bf16|int8,tail=...,cold=... names the KV tiers\n"
           "       (hot = the resident format of every full-attention layer, bf16/int8 only;\n"
           "       cold = the aged-out tier format, whose codec is derived from the layer\n"
           "       dtype and only reachable as int8 today; tail is accepted only when it\n"
           "       repeats hot). --nvfp4-mode pure forbids nvfp4/iso4e/rk4v4.\n"
           "       --kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom\n"
           "       --no-prefix-reuse disables compatible-prefix caching (enabled by default)\n"
           "       context cache defaults: device-state=max-concurrency, private=2x concurrency, "
           "shared=concurrency, anchors=2; Host state=8 slots, Host KV=8192 MiB\n"
           "       --device-state-slots is extra checkpoint capacity beyond active lanes; "
           "--host-kv-mib uses MiB\n"
           "       --default-thinking-budget caps model-origin thinking for enabled requests; "
           "control tokens count toward the request output limit\n"
           "       --preserve-thinking retains closed-turn assistant reasoning in later prompts\n"
           "       --prefill-chunk N is the prefill unit's ceiling, and --prefill-chunk-mode "
           "dynamic|manual says who picks the value under it: dynamic (default) lets the bandwidth "
           "governor shrink the unit while decode latency sits above its measured noise floor, "
           "manual pins it to N for the whole run (NINFER_FT_BW_GOV=0 is the environment spelling "
           "of manual, =1 of dynamic; the flag wins, and NINFER_FT_BW_TRACE=1 prints the mode and "
           "the unit the engine installed).\n"
           "       --kv-bits is the JOINT K/V ceiling (one number for the whole stack); "
           "--kv-k-bits with --kv-v-bits is the SPLIT form, where each plane's per-layer "
           "layering is solved in its own budget and a layer whose K and V requirements meet "
           "no single tier is refused by index with the missing (K,V) cell named. "
           "--kv-bits-mode split (default) | joint | ceiling picks the reading. Both planes "
           "always cost the same bits per element: one DType drives both.\n"
           "       --kv-quality-weight W is the speed/quality slider for the fit (0 fastest, 1 "
           "most accurate); it needs a ceiling (--kv-bits / --kv-k-bits / --kv-v-bits) and is "
           "refused without one rather than silently ignored. --kv-tier-scores FILE replaces "
           "the score table.\n"
           "       --kv-dtype names ONE global KV tier: bf16, int8, fp8, nvfp4, iso4e, "
           "rk4v4, and the narrower K planes rk3v4/rk2v4. The last two are accepted here and "
           "refused one step later by product::kv_dtype_for_storage, which names the missing "
           "reader. --kv-dtype cannot be combined with --kv-bit-budget/--kv-bits, whose "
           "ceiling would replace the table it fills (refused by name, before any load).\n"
           "       --kv-bit-budget takes a scalar bits-per-element ceiling in [0.01,16] or per "
           "layer ranges (\"0-7:8,8-63:4.5\"): the plane-agnostic spelling of the same ceiling "
           "set as --kv-bits/--kv-k-bits/--kv-v-bits, and mutually exclusive with it and with "
           "--kv-layer-storage. --kv-layer-storage SPEC pins a per-layer table; "
           "--kv-residual-layers SPEC names the layers kept at full precision.\n"
           "       --cold-policy picks the aged-out tier (none/off aliases, window, host, disk, "
           "host-then-disk; \"host+disk\" is an equivalent spelling). --cold-keep-tokens and "
           "--max-cold-pages size it (0 keeps the policy-derived value), --cold-host-bytes "
           "bounds the pinned rung and --cold-disk-path with --cold-disk-bytes the spilled one. "
           "--kv-unload-watermark-pages is the free-pool watermark at or below which the engine "
           "proactively unloads what its directory judges unloadable; 0 is OFF and is the "
           "default. Exceeds u32 max is refused, not clamped.\n"
           "       --weight-host-bytes is W13's pinned host mirror of the offloaded weights, "
           "taken at load time and not a soft cap; --weight-device-arena-bytes is the device "
           "working set kept for those layers and must be smaller than --weight-host-bytes, or "
           "the offload would free no device memory (refused by name). "
           "--weight-prefetch-layers needs >= 2, since 1 would let a layer's own prefetch "
           "overwrite the arena slot it is being computed from. "
           "--weight-span-floor-bytes needs a positive --weight-host-bytes. The byte knobs "
           "take a plain integer: no g/m/k suffix.\n"
           "       --yarn enables YaRN rotary scaling and takes no value.\n"
           "       --kv-score-table show|emit=PATH prints or writes the penalty table and needs "
           "no model; with an artifact path it starts the server afterwards.\n"
           "       --kv-score-table show|emit=PATH is the penalty table's own entry point: it "
           "runs with no model, prints or writes the table the planner would use, and states "
           "the real provenance of each column (the shipped quality column is a PRIOR).\n"
           "       --kv-auto-relayout SECS re-derives the per-layer KV table from the "
           "FreeToken energy table every SECS seconds (default 0 = off; "
           "NINFER_FT_RELOAD_SECS is the env spelling). --ft-stats on enables that "
           "observation (NINFER_FT_STATS); --ft-vram-axis off removes the free-VRAM "
           "term from the decision (NINFER_FT_VRAM_AXIS, default on and inert while "
           "the relayout is off). CLI beats the environment beats the default.\n"
           "       sampler defaults come from the loaded model and resolved thinking mode; "
           "server flags and request fields override individual values.\n"
           "       --greedy forces temperature 0 (exact argmax).\n";
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    bool kv_capacity_explicit        = false;
    bool context_capacity_explicit   = false;
    bool unload_watermark_explicit   = false;
    // The same gate as apps/cli/options.cpp: the MTP adaptive escape hatch must not
    // overwrite a width the flag pinned (that file's `draft_tokens_explicit`).
    bool draft_tokens_explicit       = false;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    // --kv-score-table (the penalty table's own entry point) needs no model: accept it in
    // the artifact position too, so `ninfer-serve --kv-score-table show` works from a
    // bare shell. A model path given as well still wins, and main() then starts the
    // server normally after writing the table.
    const bool score_table_standalone = std::string_view(argv[1]) == "--kv-score-table";
    if (!score_table_standalone) { options.artifact_path = argv[1]; }
    for (int i = score_table_standalone ? 1 : 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--prefill-chunk-mode") {
            // dynamic | manual. Same vocabulary, same validation and the same mode -> mechanism
            // mapping as the CLI's flag (BandwidthGovernor::parse_mode), so the server's handle on
            // the prefill unit is literally the same handle and not a second spelling of it.
            options.prefill_chunk_mode = ninfer::runtime::BandwidthGovernor::parse_mode(
                require_value("--prefill-chunk-mode"));
        } else if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--device-state-slots") {
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
            context_capacity_explicit = true;
        } else if (arg == "--host-state-slots") {
            options.context_cache.host_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--host-state-slots"), "host-state-slots"));
            context_capacity_explicit = true;
        } else if (arg == "--host-kv-mib") {
            const std::uint64_t mib = parse_u64(require_value("--host-kv-mib"), "host-kv-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-kv-mib is out of range");
            }
            options.context_cache.host_kv_capacity_bytes = static_cast<std::size_t>(mib << 20);
            context_capacity_explicit                    = true;
        } else if (arg == "--max-private-continuations") {
            options.context_cache.max_private_continuations =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-private-continuations"), "max-private-continuations"));
            context_capacity_explicit = true;
        } else if (arg == "--max-shared-prefixes") {
            options.context_cache.max_shared_prefixes =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-shared-prefixes"), "max-shared-prefixes"));
            context_capacity_explicit = true;
        } else if (arg == "--max-long-anchors-per-continuation") {
            options.context_cache.max_long_anchors_per_continuation = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-long-anchors-per-continuation"),
                                      "max-long-anchors-per-continuation"));
            context_capacity_explicit = true;
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
            options.kv_cache_explicit = true;
        } else if (arg == "--kv-layer-storage") {
            // Table AND mask: `0-11:bf16` must mean BF16 on those layers, not
            // "inherit --kv-dtype" (BFloat16 is also the unset sentinel, which is
            // why the table alone cannot express it -- product/kv_options.h).
            const auto parsed = product::parse_kv_layer_storage_spec(
                require_value("--kv-layer-storage"));
            options.kv_layer_storage     = parsed.table;
            options.kv_layer_storage_set = parsed.set;
            options.kv_layer_storage_explicit = true;
        } else if (arg == "--kv-tier-formats") {
            // KV tier vocabulary ("hot=bf16,tail=fp16,cold=iso4e"; kvcfg/kv_formats.h).
            // Raw text: vocabulary rules are checked after the loop (the mode may come
            // later in argv) and the per-layer landing needs the model's layer count.
            options.kv_tier_formats_spec     = require_value("--kv-tier-formats");
            options.kv_tier_formats_explicit = true;
        } else if (arg == "--nvfp4-mode") {
            const std::string_view mode = require_value("--nvfp4-mode");
            if (mode == "pure") {
                options.kv_nvfp4_pure = true;
            } else if (mode == "fusion") {
                options.kv_nvfp4_pure = false;
            } else {
                throw std::invalid_argument("invalid nvfp4-mode: " + std::string(mode));
            }
            options.kv_tier_formats_explicit = true;
        } else if (arg == "--kv-rotation") {
            // SEPARATION: SO(4) rotation of K on cache write and Q before
            // quantization. off takes the identity map on BOTH sides.
            const std::string_view mode = require_value("--kv-rotation");
            if (mode == "off") {
                options.kv_rotation_off = true;
            } else if (mode == "on" || mode == "auto" || mode == "default") {
                options.kv_rotation_off = false;
            } else {
                throw std::invalid_argument("invalid kv-rotation: " + std::string(mode) +
                                            " (expected on|off)");
            }
            options.kv_rotation_explicit = true;
        } else if (arg == "--kv-row-scale") {
            // SEPARATION: three-state row scale. The vocabulary is validated by
            // kv_rowscale_mode_from_spec() at plan time (the same parser the
            // environment hook uses), so the CLI cannot drift from NINFER_KV_ROWSCALE.
            options.kv_row_scale_spec     = std::string(require_value("--kv-row-scale"));
            options.kv_row_scale_explicit = true;
        } else if (arg == "--kv-v-codec") {
            // SEPARATION: V-plane codec on the NVFP4 tier. iso4e is the engine
            // default; e2m1 is an ablation and is refused when a mechanism that
            // hardcodes ISO4E V (residual plane, cold pool) is active. The canonical
            // spelling is `iso4e`; `iso3` is the DEPRECATED spelling of the same
            // state, accepted for one release with a warning. This front end used
            // to be the exact inverse of the CLI's (that one took only `iso3`).
            const std::string_view mode = require_value("--kv-v-codec");
            if (mode == "iso4e") {
                options.kv_v_codec = KvVCodec::Iso3;
            } else if (mode == "iso3") {
                std::fprintf(stderr,
                             "[kv-v-codec] --kv-v-codec iso3 is deprecated: it names "
                             "the same state as iso4e (KvVCodecMode::Iso4e). Use "
                             "--kv-v-codec iso4e.\n");
                options.kv_v_codec = KvVCodec::Iso3;
            } else if (mode == "e2m1") {
                options.kv_v_codec = KvVCodec::E2M1;
            } else {
                throw std::invalid_argument("invalid kv-v-codec: " + std::string(mode) +
                                            " (expected iso4e|e2m1; iso3 is the deprecated "
                                            "spelling of iso4e, accepted for one release)");
            }
            options.kv_v_codec_explicit = true;
        } else if (arg == "--kv-bit-budget") {
            // Same two forms as the CLI: a scalar ceiling, or separable per-range ceilings
            // ("0-7:8,8-63:4.5") parsed by the allocator itself.
            const std::string budget_spec = require_value("--kv-bit-budget");
            if (budget_spec.find(':') != std::string::npos ||
                budget_spec.find(',') != std::string::npos) {
                options.kv_bit_budget_ranges              = budget_spec;
                options.kv_bit_budget_bits                = 0.0;
                options.kv_bit_budget_explicit            = true;
                continue;
            }
            // budget_spec is already the consumed value: do NOT call require_value again,
            // it would advance past the next argv token (e.g. swallow --port).
            options.kv_bit_budget_bits = parse_float_in(budget_spec.c_str(), "--kv-bit-budget",
                                                        0.01f, 16.0f);
            // Fractional bits/element for the full-attention KV. Stored raw here: this
            // parse site has no model knowledge (the layer count is unknown); the
            // per-layer table is resolved from the DP in make_sequence_planner_impl
            // (layouts_impl.h), where TextConfig::full_attention_layers() is available.
            options.kv_bit_budget_explicit = true;
        } else if (arg == "--kv-bits") {
            // JOINT form: ONE overall ceiling for the whole KV stack ("合起来整体定").
            options.kv_joint_bits =
                parse_float_in(require_value("--kv-bits"), "--kv-bits", 0.01f, 16.0f);
            options.kv_kv_bits_explicit = true;
        } else if (arg == "--kv-k-bits") {
            // SPLIT form: K gets its own ceiling and its own per-layer layering.
            options.kv_k_bits = parse_float_in(require_value("--kv-k-bits"), "--kv-k-bits",
                                               0.01f, 16.0f);
            options.kv_kv_bits_explicit = true;
        } else if (arg == "--kv-v-bits") {
            // SPLIT form: V gets its own ceiling and its own per-layer layering.
            options.kv_v_bits = parse_float_in(require_value("--kv-v-bits"), "--kv-v-bits",
                                               0.01f, 16.0f);
            options.kv_kv_bits_explicit = true;
        } else if (arg == "--kv-bits-mode") {
            const std::string_view mode = require_value("--kv-bits-mode");
            options.kv_bits_mode = product::kv_bits_mode_from_name(mode);
            options.kv_bits_mode_explicit = true;
        } else if (arg == "--kv-tier-scores") {
            options.kv_tier_scores = require_value("--kv-tier-scores");
        } else if (arg == "--kv-k-tier-scores") {
            options.kv_k_tier_scores = require_value("--kv-k-tier-scores");
        } else if (arg == "--kv-v-tier-scores") {
            options.kv_v_tier_scores = require_value("--kv-v-tier-scores");
        } else if (arg == "--kv-quality-weight") {
            options.kv_quality_weight = parse_float_in(require_value("--kv-quality-weight"),
                                                       "--kv-quality-weight", 0.0f, 1.0f);
        } else if (arg == "--kv-score-table") {
            // The penalty table's own entry point; main() acts on it before any model
            // or device work, so the server can run it too.
            options.kv_score_table_spec     = require_value("--kv-score-table");
            options.kv_score_table_explicit = true;
        } else if (arg == "--kv-residual-layers") {
            // The grammar and the two error strings now live in ONE place
            // (product/kv_options.h), because the same flag is reachable from all
            // three front ends and this project's standing failure is a rule that
            // lands in one front end and misses the others. Acceptance is unchanged
            // except that the token must be pure decimal: see the parser's comment
            // for the two spellings this tightens.
            const product::KvResidualLayersSpec parsed =
                product::parse_kv_residual_layers_spec(require_value("--kv-residual-layers"));
            options.kv_residual_layers   = parsed.table;
            options.kv_residual_explicit = true;
        } else if (arg == "--cold-policy") {
            const std::string_view v = require_value("--cold-policy");
            if (v == "none" || v == "off") { options.cold_policy = ColdPolicy::None; }
            else if (v == "window") { options.cold_policy = ColdPolicy::Window; }
            else if (v == "host") { options.cold_policy = ColdPolicy::Host; }
            else if (v == "disk") { options.cold_policy = ColdPolicy::Disk; }
            // Layered cold tier: --cold-host-bytes bounds the pinned tier and the
            // excess spills to --cold-disk-path under --cold-disk-bytes.
            // "host+disk" is an equivalent spelling of the same policy.
            else if (v == "host-then-disk" || v == "host+disk") {
                options.cold_policy = ColdPolicy::HostThenDisk;
            }
            else { throw std::invalid_argument("invalid cold-policy: " + std::string(v)); }
        } else if (arg == "--cold-disk-path") {
            options.cold_disk_path = require_value("--cold-disk-path");
        } else if (arg == "--cold-disk-bytes") {
            options.cold_disk_bytes =
                parse_u64(require_value("--cold-disk-bytes"), "cold-disk-bytes");
            if (options.cold_disk_bytes == 0) {
                throw std::invalid_argument("--cold-disk-bytes must be positive");
            }
        } else if (arg == "--cold-keep-tokens") {
            options.cold_keep_tokens =
                parse_u64(require_value("--cold-keep-tokens"), "cold-keep-tokens");
            if (options.cold_keep_tokens > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--cold-keep-tokens is out of range");
            }
        } else if (arg == "--max-cold-pages") {
            // Explicit cold-pool capacity in pages; 0 keeps the policy-derived
            // value (Window/Disk/HostThenDisk: cold_keep_tokens/page_size + 16).
            // ColdPolicy::None ignores this (layouts_impl forces 0 -- no cold path
            // at all). Under host-then-disk this is the disk rung's working set and
            // is honoured (the Host + cap contradiction is Host-only).
            const std::uint64_t pages =
                parse_u64(require_value("--max-cold-pages"), "max-cold-pages");
            if (pages > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--max-cold-pages is out of range");
            }
            options.max_cold_pages = static_cast<std::uint32_t>(pages);
        } else if (arg == "--kv-unload-watermark-pages") {
            // THE UNLOAD WATERMARK: free text-KV pool pages at or below which the
            // Engine proactively unloads what its semantic directory judges
            // unloadable. 0 = OFF, and OFF is the pre-watermark behaviour. The
            // sentinel kUnloadWatermarkDerive is reachable on purpose (it is the
            // default), so an explicit value here is never clamped onto it.
            const std::uint64_t pages =
                parse_u64(require_value("--kv-unload-watermark-pages"),
                          "kv-unload-watermark-pages");
            if (pages > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--kv-unload-watermark-pages is out of range");
            }
            options.unload_watermark_pages = static_cast<std::uint32_t>(pages);
            unload_watermark_explicit      = true;
        } else if (arg == "--kv-auto-relayout") {
            // FreeToken step 2 (serve/kv_auto_relayout.h): the periodic KV
            // relayout. 0 disables it and is also the default, so the line stays
            // opt-in and a deployment that never passes this flag behaves
            // exactly as before the flag existed. Passing 0 is meaningful: it
            // overrides an enabling NINFER_FT_RELOAD_SECS in the environment.
            options.ft_relayout_secs = parse_nonnegative_int(require_value("--kv-auto-relayout"),
                                                            "kv-auto-relayout");
        } else if (arg == "--ft-vram-axis") {
            // Gap 1: the free-VRAM term of that decision. on|off for the same
            // reason --kv-rotation uses it (one flag, both directions).
            options.ft_vram_axis = parse_on_off(require_value("--ft-vram-axis"), "ft-vram-axis");
        } else if (arg == "--ft-stats") {
            // FreeToken step 1 observation: per-layer attention energy. `on`
            // also disables nothing by itself -- the observation is what
            // --kv-auto-relayout consumes.
            options.ft_stats = parse_on_off(require_value("--ft-stats"), "ft-stats");
        } else if (arg == "--cold-host-bytes") {
            options.cold_host_bytes =
                parse_u64(require_value("--cold-host-bytes"), "cold-host-bytes");
        } else if (arg == "--weight-host-bytes") {
            // W13: the pinned host mirror of the offloaded weights. Not a soft cap --
            // the pinned allocation is unswappable and is taken at load time.
            options.weight_host_offload_bytes =
                parse_u64(require_value("--weight-host-bytes"), "weight-host-bytes");
        } else if (arg == "--weight-device-arena-bytes") {
            // The device working set kept for the offloaded layers. This is the number
            // that decides how much device memory the offload actually frees.
            options.weight_device_arena_bytes =
                parse_u64(require_value("--weight-device-arena-bytes"),
                          "weight-device-arena-bytes");
        } else if (arg == "--weight-prefetch-layers") {
            const int layers = parse_nonnegative_int(require_value("--weight-prefetch-layers"),
                                                     "weight-prefetch-layers");
            if (layers < 2) {
                throw std::invalid_argument(
                    "--weight-prefetch-layers below 2 would let the arena slot of the layer "
                    "being computed be overwritten by its own prefetch");
            }
            options.weight_prefetch_layers = static_cast<std::uint32_t>(layers);
        } else if (arg == "--weight-span-floor-bytes") {
            options.weight_span_floor_bytes =
                parse_u64(require_value("--weight-span-floor-bytes"), "weight-span-floor-bytes");
        } else if (arg == "--yarn") {
            options.yarn_enabled = true;
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
            draft_tokens_explicit = true;
        } else if (arg == "--draft-tree") {
            // L,d -- the MTP tree verify shape. Same spelling, same parse and same domain bound
            // as apps/cli/options.cpp:390-409; the two front ends must not drift on a flag that
            // picks the round's verify topology. The L*d <= 15 node gate and the
            // head-vs-tree refusal stay in the planner
            // (targets/qwen3_6/impl/runtime/layouts_impl.h:1012-1034), which is where the
            // target's own MTP draft domain constant is in scope, and which the server already
            // reaches (layouts_impl.h:1012-1014 says so by name).
            const std::string tree(require_value("--draft-tree"));
            const std::size_t comma = tree.find(',');
            if (comma == std::string::npos || comma == 0 || comma + 1 == tree.size()) {
                throw std::invalid_argument("invalid draft-tree: " + tree + " (expected L,d)");
            }
            const int tree_paths =
                parse_nonnegative_int(tree.substr(0, comma).c_str(), "draft-tree L");
            const int tree_depth =
                parse_nonnegative_int(tree.substr(comma + 1).c_str(), "draft-tree d");
            // serve3 FIX (dl/serve3, 2026-09-22): L and d are BOTH in [1,...], and the CLI
            // enforces that by parsing them with parse_u32(..., allow_zero=false)
            // (apps/cli/options.cpp:600-603), refusing `--draft-tree 0,d` as
            // "invalid draft-tree L: 0". THIS front end used parse_nonnegative_int and then
            // tested only the UPPER bounds, so 0 was admitted -- and because
            // draft_tree_paths == 0 is this front end's own "no tree" sentinel (the
            // `draft_tree_paths != 0` guard further down), an accepted `--draft-tree 0,d`
            // read as "no tree at all" and the operator's d was read by nothing. That is
            // the silent degradation this project bans, on a flag whose comment four lines
            // up says the two front ends "must not drift". Both refusal strings below are
            // the CLI's own, so the two front ends now accept exactly the same set.
            if (tree_paths == 0) {
                throw std::invalid_argument("invalid draft-tree L: 0");
            }
            if (tree_depth == 0) {
                throw std::invalid_argument("invalid draft-tree d: 0");
            }
            options.speculative.draft_tree_paths = static_cast<std::uint32_t>(tree_paths);
            options.speculative.draft_tree_depth = static_cast<std::uint32_t>(tree_depth);
            if (options.speculative.draft_tree_paths > 16 ||
                options.speculative.draft_tree_depth > 15) {
                throw std::invalid_argument(
                    "invalid draft-tree: L must be in [1,16] and d in [1,15]");
            }
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--no-auto-system-shared-prefix") {
            options.auto_system_shared_prefix = false;
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--no-lm-head-draft") {
            // The opt-out that makes a head experiment reversible on the serve side: without it
            // a serve arm could only ever turn the shortlist head ON (CLI has both spellings,
            // apps/cli/options.cpp:490-491). Pure option state -- the head is resolved by the
            // target package before the planner, so nothing serve-specific is involved.
            options.speculative.proposal_head = ProposalHead::Full;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            const int top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
            if (top_k > 20) { throw std::invalid_argument("top-k must be in [0,20]"); }
            options.sampling_overrides.top_k = top_k;
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    // ===========================================================================
    // TEMPORARY adaptive-MTP escape hatch: NINFER_MTP_ADAPTIVE=1 -- the same env and the same
    // effect as apps/cli/options.cpp. DELETE BOTH BLOCKS TOGETHER.
    //
    // This front end already accepts the literal where the CLI does not: --draft-tokens here
    // goes through parse_nonnegative_int, so `--spec mtp --draft-tokens 0` already reaches the
    // adaptive ladder. The env is wired in only so the two front ends cannot drift and so the
    // --spec-less spelling exists on both sides.
    //
    // Mind the default that differs: this parser leaves SpeculativeOptions::backend at None
    // (speculation off), where the CLI defaults it to auto. So here the env alone does NOT turn
    // speculation on -- pass --spec mtp. Only Auto is promoted, and an explicit
    // none|dflash|dflash2 is never rewritten.
    //
    // 0 is the adaptive spelling of the MTP draft width: the add-form survival/cost criterion
    // in mtp_window_cut.h picks the ladder rung per round, while k > 0 pins one captured width
    // (bit-identical to the recorded fixed-k runs). Both downstream gates already take it --
    // product::validate_speculative_cli_options ([0,15], product/speculative_options.h:61) and
    // validate_target_options (0 or [1,kMaximumMtpDraftTokens], layouts_impl.h:890).
    //
    // THE REAL FIX IS NOT THIS ENV VAR. Same three follow-ups as in apps/cli/options.cpp:
    //   (a) letting the criterion in mtp_window_cut.h decide the width where the width is
    //       actually chosen, instead of 0 travelling as a sentinel each gate re-interprets;
    //   (b) making the upper bound VARIANT-aware instead of the hard-wired 15 (kMaximumMtpDraftTokens
    //       is 15 on 27b but only 5 on 35b_a3b and muse_glimmer_30b);
    //   (c) unifying the two parse points so both spellings of "adaptive" mean the same thing.
    // ===========================================================================
    {
        const char* adaptive_env = std::getenv("NINFER_MTP_ADAPTIVE");
        if (adaptive_env != nullptr && std::atoi(adaptive_env) != 0) {
            // Auto is the only spelling of "let the artifact decide" and is not reachable
            // without an explicit --spec here (the default is None), so promoting it cannot
            // contradict an operator's choice. An explicit non-mtp backend is never rewritten:
            // the env is named for mtp.
            if (options.speculative.backend == SpeculativeBackend::Auto) {
                options.speculative.backend = SpeculativeBackend::Mtp;
            }
            if (options.speculative.backend == SpeculativeBackend::Mtp) {
                if (draft_tokens_explicit) {
                    // Flag beats env, the same way and for the same reason as
                    // apps/cli/options.cpp: docs/cli.md:217 (`CLI > environment > default`)
                    // and docs/cli.md:419 ("An explicit --draft-tokens value wins over it").
                    std::fprintf(stderr,
                                 "[mtp-adaptive] NINFER_MTP_ADAPTIVE=1: an explicit "
                                 "--draft-tokens %u wins (CLI > environment); the adaptive "
                                 "ladder is %s\n",
                                 static_cast<unsigned>(options.speculative.draft_tokens),
                                 options.speculative.draft_tokens == 0 ? "the flag's own choice"
                                                                       : "not selected");
                } else {
                    options.speculative.draft_tokens = 0;
                    std::fprintf(stderr,
                                 "[mtp-adaptive] NINFER_MTP_ADAPTIVE=1: draft window forced to 0 "
                                 "(adaptive ladder) on the mtp backend\n");
                }
            }
        }
    }

    // --draft-tree L,d REPLACES the scalar draft width: the node budget L*d IS the round's
    // extent, so naming both contradict each other (layouts_impl.h:1671-1674 makes the tree win
    // silently, which is exactly what must not happen). Refused rather than ignored, the same
    // way and with the same message as apps/cli/options.cpp:624-634.
    // This check is load-bearing on this front end: product::validate_speculative_cli_options
    // does not look at draft_tree_paths at all, so without it `--spec none --draft-tree 1,7`
    // would parse clean and quietly run without a tree.
    if (options.speculative.draft_tree_paths != 0 &&
        (options.speculative.backend != SpeculativeBackend::Mtp ||
         options.speculative.draft_tokens != 0)) {
        throw std::invalid_argument(
            "--draft-tree L,d needs --spec mtp and no --draft-tokens (the tree's node budget "
            "L*d is the draft width)");
    }

    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    // FreeToken: commit an explicit --ft-stats to the environment the observation
    // reads (see set_process_env above). An unset flag leaves the operator's
    // NINFER_FT_STATS untouched, which is what makes "env only" still work.
    if (options.ft_stats.has_value() &&
        set_process_env("NINFER_FT_STATS", *options.ft_stats ? "1" : "0") != 0) {
        throw std::invalid_argument("cannot set NINFER_FT_STATS for --ft-stats");
    }
    if (!options.allow_prefix_reuse) {
        if (context_capacity_explicit) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with context-cache capacity options");
        }
        options.context_cache.enabled                = false;
        options.context_cache.host_state_slots       = 0;
        options.context_cache.host_kv_capacity_bytes = 0;
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.speculative.backend == SpeculativeBackend::DFlash && options.enable_vision) {
        throw std::invalid_argument("--spec dflash cannot be combined with --vision");
    }
    if (options.kv_bit_budget_explicit && options.kv_layer_storage_explicit) {
        throw std::invalid_argument(
            "--kv-bit-budget and --kv-layer-storage are mutually exclusive: the budget "
            "is resolved into exactly the table --kv-layer-storage provides");
    }
    // The K/V bit-width entries (product/kv_kv_bits.h). Same contradiction rule as the
    // CLI: they are a second spelling of the same ceiling, not an addition to it.
    if (options.kv_kv_bits_explicit && options.kv_bit_budget_explicit) {
        throw std::invalid_argument(
            "--kv-bit-budget and --kv-bits/--kv-k-bits/--kv-v-bits are two spellings of the "
            "same ceiling set: --kv-bit-budget is the plane-agnostic form. Give one of them.");
    }
    // res-kv5 FIX (serve front end): the same contradiction the CLI refuses. A ceiling
    // is resolved into exactly the per-layer table --kv-dtype pins, so accepted
    // together one of the two is read by nothing (measured on the pre-fix arm: dtype
    // AND payload identical to the ceiling alone).
    if (options.kv_cache_explicit &&
        (options.kv_bit_budget_explicit || options.kv_kv_bits_explicit)) {
        throw std::invalid_argument(
            "--kv-dtype and --kv-bit-budget/--kv-bits are mutually exclusive: --kv-dtype "
            "pins the global KV tier, and a ceiling is resolved into exactly the "
            "per-layer table that would replace it. Give one of them.");
    }
    if (options.kv_kv_bits_explicit && options.kv_layer_storage_explicit) {
        throw std::invalid_argument(
            "--kv-bits/--kv-k-bits/--kv-v-bits and --kv-layer-storage are mutually exclusive: "
            "the budget is resolved into exactly the table --kv-layer-storage provides");
    }
    if (options.kv_bits_mode_explicit && !options.kv_kv_bits_explicit) {
        throw std::invalid_argument(
            "--kv-bits-mode needs a K/V bit request to act on (--kv-bits / --kv-k-bits / "
            "--kv-v-bits); without one nothing would read it.");
    }
    if (options.kv_quality_weight >= 0.0 && !options.kv_bit_budget_explicit &&
        !options.kv_kv_bits_explicit) {
        // The slider used to be accepted and then ignored on exactly this path (it is
        // read only inside the bit-budget resolution), which is the project's worst
        // outcome. Name the flags that make it act instead.
        throw std::invalid_argument(
            "--kv-quality-weight needs a ceiling to act on: it selects the per-tier penalty "
            "inside the bit budget, and with no --kv-bit-budget / --kv-bits / "
            "--kv-k-bits --kv-v-bits there is no fit for it to weight. Add one, or drop the "
            "flag.");
    }
    // The score-table half of the same rule (product/kv_kv_bits.h refuses these too; the
    // server reaches the same resolver, this is the earlier, louder place).
    const bool any_score_table = !options.kv_tier_scores.empty() ||
                                 !options.kv_k_tier_scores.empty() ||
                                 !options.kv_v_tier_scores.empty();
    if (any_score_table && options.kv_quality_weight < 0.0) {
        throw std::invalid_argument(
            "--kv-tier-scores / --kv-k-tier-scores / --kv-v-tier-scores need "
            "--kv-quality-weight to act on: a table's two columns are combined only by the "
            "weight (penalty = w*quality + (1-w)*speed), so without one the shipped "
            "single-penalty ladder runs and the table would be read by nothing. Add "
            "--kv-quality-weight W, or drop the table.");
    }
    if (options.kv_joint_bits > 0.0 && (options.kv_k_bits > 0.0 || options.kv_v_bits > 0.0)) {
        throw std::invalid_argument(
            "--kv-bits and --kv-k-bits/--kv-v-bits are two spellings of the same ceiling set: "
            "--kv-bits is the ONE overall ceiling, the two per-plane ceilings cover it. Give "
            "one or the other; --kv-bits-mode picks the reading of the per-plane form.");
    }
    if (options.kv_bits_mode_explicit && options.kv_joint_bits > 0.0 &&
        options.kv_bits_mode != KvBitsMode::Joint) {
        throw std::invalid_argument(
            "--kv-bits-mode was given together with --kv-bits, which names ONE overall "
            "ceiling: with a single ceiling there is no second reading for the mode to "
            "select, so it would be accepted and read by nothing. Give --kv-k-bits/--kv-v-bits "
            "for the split and ceiling readings, or drop --kv-bits-mode.");
    }
    const bool joint_reading =
        !(options.kv_k_bits > 0.0 || options.kv_v_bits > 0.0) ||
        (options.kv_bits_mode_explicit && options.kv_bits_mode == KvBitsMode::Joint);
    if (joint_reading &&
        (!options.kv_k_tier_scores.empty() || !options.kv_v_tier_scores.empty())) {
        throw std::invalid_argument(
            "--kv-k-tier-scores / --kv-v-tier-scores were given but this request resolves to "
            "the JOINT reading (one ceiling, one ladder), where a per-plane table has nothing "
            "to fit and would be read by nothing. Give --kv-k-bits/--kv-v-bits so each plane "
            "is solved in its own budget, or drop the per-plane tables.");
    }
    if (options.kv_tier_formats_explicit) {
        // Vocabulary rules only (bad tier, bad format, tier ordering, pure vs iso/rk4v4).
        // The per-layer landing plus the tier/table consistency checks need the model's
        // full-attention layer count and run in make_sequence_planner_impl.
        (void)product::kv_tier_formats_parse(options.kv_tier_formats_spec,
                                             options.kv_nvfp4_pure);
    }
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    // W13 startup contradictions only; the runtime strides are the consumer's to
    // report, exactly as validate_cold_tier_budget() draws that line. An operator-set
    // device arena that is not smaller than what is offloaded cannot free anything,
    // so it is a configuration that cannot do what it says.
    if (options.weight_host_offload_bytes == 0 &&
        (options.weight_device_arena_bytes != 0 || options.weight_span_floor_bytes != 0)) {
        throw std::invalid_argument("--weight-device-arena-bytes / --weight-span-floor-bytes "
                                    "need a positive --weight-host-bytes");
    }
    if (options.weight_host_offload_bytes != 0 &&
        options.weight_device_arena_bytes >= options.weight_host_offload_bytes) {
        throw std::invalid_argument(
            "--weight-device-arena-bytes is not smaller than --weight-host-bytes, so the "
            "offload would free no device memory");
    }
    // THE WATERMARK'S ENV OVERRIDE, and the precedence is CLI > env > default for
    // the same reason every sibling flag follows it: a flag is a deliberate act of
    // one run, an environment variable is a deployment's standing wish. An
    // unparseable value is REFUSED rather than ignored -- a silently-ignored
    // watermark would be a run that believes it is protected and is not, which is
    // precisely the failure this whole knob exists to name.
    if (!unload_watermark_explicit) {
        if (const char* env = std::getenv("NINFER_KV_UNLOAD_WATERMARK_PAGES"); env != nullptr &&
                                                                             env[0] != '\0') {
            const std::uint64_t pages =
                parse_u64(env, "NINFER_KV_UNLOAD_WATERMARK_PAGES");
            if (pages > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument(
                    "NINFER_KV_UNLOAD_WATERMARK_PAGES is out of range");
            }
            options.unload_watermark_pages = static_cast<std::uint32_t>(pages);
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_id.empty()) {
        throw std::logic_error("loaded artifact model_id must not be empty");
    }
    return std::string(artifact_model_id);
}

} // namespace ninfer::serve
