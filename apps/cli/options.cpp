#include "options.h"
#include "product/speculative_options.h"
#include "product/kv_options.h"
#include "product/kv_tier_formats.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace ninfer::cli {
namespace {

std::uint64_t parse_u64(const char* text, std::string_view label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

std::uint32_t parse_u32(const char* text, std::string_view label, bool allow_zero = false) {
    const std::uint64_t value = parse_u64(text, label);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint32_t>(value);
}

int parse_device(const char* text) {
    const std::uint64_t value = parse_u64(text, "device");
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid device: ") + text);
    }
    return static_cast<int>(value);
}

float parse_float(const char* text, std::string_view label, float minimum, float maximum) {
    errno              = 0;
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(value) ||
        value < static_cast<double>(minimum) || value > static_cast<double>(maximum)) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<float>(value);
}

KvCacheStorage parse_kv_cache(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "iso3") { return KvCacheStorage::Iso3Group16; }
    if (text == "e8") { return KvCacheStorage::E8Group64; }
    throw std::invalid_argument("invalid kv-dtype: " + std::string(text));
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    return KvCapacityPolicy::explicit_capacity(parse_u32(text, "kv-capacity"));
}

ReasoningEffort parse_reasoning_effort(std::string_view text) {
    if (text == "low") { return ReasoningEffort::Low; }
    if (text == "medium") { return ReasoningEffort::Medium; }
    if (text == "xhigh") { return ReasoningEffort::XHigh; }
    throw std::invalid_argument("invalid reasoning-effort: " + std::string(text));
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> (--prompt <text>|--messages <messages.json>)\n"
           "       [--max-context N] [--kv-capacity N|auto] [--prefill-chunk N] [--max-new N]\n"
           "       [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8] [--kv-layer-storage SPEC] [--kv-bit-budget SPEC] [--spec auto|mtp|dflash|dflash2|none --draft-tokens N]\n           (--kv-bit-budget takes a ceiling per KV element, or per layer range:\n            \"0-7:8,8-63:4.5\"; it never exceeds the declared ceilings)\n           (--spec defaults to auto; none turns speculation off)\n"
           "       [--kv-tier-formats hot=auto|bf16|int8,tail=...,cold=...] [--nvfp4-mode fusion|pure]\n"
           "           (hot = the resident format of every full-attention layer; only bf16 and\n"
           "            int8 have a resident codec. cold is the aged-out tier format; the cold\n"
           "            slot codec is derived from the layer dtype and only int8 is reachable\n"
           "            today. tail is accepted only when it repeats hot: the engine has no\n"
           "            recent-window tier yet. --nvfp4-mode pure forbids nvfp4/iso3/e8.)\n"
           "       [--kv-rotation on|off] [--kv-row-scale auto|off|FILE]\n"
           "       [--kv-v-codec iso3|e2m1]\n"
           "           (component switches for the NVFP4/FP8/ISO3 KV tiers.\n"
           "            --kv-rotation off takes the identity SO(4) map on BOTH the K write\n"
           "            and the Q read, so QK^T stays exact and only the quantization domain\n"
           "            changes. --kv-row-scale off takes the identity row scale in the kernel\n"
           "            (no identity file needed); auto keeps the baked table; FILE loads an\n"
           "            NINFERKVRS1 sidecar (NINFER_KV_ROWSCALE is the env equivalent).\n"
           "            --kv-v-codec e2m1 stores NVFP4-tier V as E2M1 instead of ISO3 and is\n"
           "            refused when a V residual plane or the cold pool is active.)\n"
           "       [--lm-head-draft]\n"
           "       [--temperature F] [--top-p F] [--top-k N] [--min-p F]\n"
           "       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...\n"
           "       [--raw-output] [--print-token-ids] [--no-thinking] [--thinking-budget N]\n"
           "       [--reasoning-effort low|medium|xhigh] [--vision]\n"
           "       [--cold-policy none|off|window|host|disk] [--cold-keep-tokens N]\n"
           "       [--max-cold-pages N] [--cold-host-bytes N[g|m|k]]\n"
           "       [--cold-disk-path DIR] [--cold-disk-bytes N]\n"
           "       [--no-cuda-graph] [--graph-capture-ceiling N]\n"
           "\n"
           "Streams answer content to stdout and reasoning plus diagnostics to stderr.\n"
           "Structured message content accepts text, image/image_url, and video/video_url parts;\n"
           "media sources may be local paths, HTTP(S) URLs, or base64 data URIs.\n"
           "--vision enables image/video input and loads the fixed Vision GPU allocations.\n"
           "--thinking-budget caps model-origin thinking tokens; inserted control tokens count "
           "toward --max-new.\n"
           "--kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom.\n"
           "Sampling defaults come from the loaded model and thinking mode; flags override "
           "individual fields.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    if (argc >= 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument(".ninfer model path is required"); }
    options.artifact_path     = argv[1];
    bool kv_capacity_explicit = false;

    for (int i = 2; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto value = [&](std::string_view flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };

        if (arg == "--prompt") {
            options.prompt = value(arg);
        } else if (arg == "--messages") {
            options.messages_path = value(arg);
        } else if (arg == "--max-new") {
            options.max_new = parse_u32(value(arg), "max-new");
        } else if (arg == "--max-context") {
            options.max_context = parse_u32(value(arg), "max-context");
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(value(arg));
            kv_capacity_explicit = true;
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value(arg), "prefill-chunk");
        } else if (arg == "--device") {
            options.device = parse_device(value(arg));
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_cache(value(arg));
            options.kv_cache_explicit = true;
        } else if (arg == "--kv-bit-budget") {
            // "N" (one ceiling for every full-attention layer) or "lo-hi:bits,..."
            // (separable per-range ceilings; the DP minimises each range independently).
            const std::string budget_spec = value(arg);
            if (budget_spec.find(':') != std::string::npos ||
                budget_spec.find(',') != std::string::npos) {
                options.kv_bit_budget_ranges   = budget_spec;
                options.kv_bit_budget_bits     = 0.0;
                options.kv_bit_budget_explicit = true;
            } else {
                options.kv_bit_budget_bits =
                    parse_float(budget_spec.c_str(), "--kv-bit-budget", 0.01F, 16.0F);
                options.kv_bit_budget_ranges.clear();
                options.kv_bit_budget_explicit = true;
            }
        } else if (arg == "--kv-quality-weight") {
            // 0 = fastest KV path, 1 = most accurate; the DP minimises
            // w*quality + (1-w)*speed per tier inside the bit ceiling.
            options.kv_quality_weight =
                parse_float(value(arg), "--kv-quality-weight", 0.0F, 1.0F);
        } else if (arg == "--kv-tier-scores") {
            options.kv_tier_scores = value(arg);
        } else if (arg == "--kv-layer-storage") {
            options.kv_layer_storage_spec = value(arg);
            options.kv_layer_storage_explicit = true;
        } else if (arg == "--kv-tier-formats") {
            // KV tier vocabulary ("hot=bf16,tail=fp16,cold=iso3"; kvcfg/kv_formats.h).
            // Parsed raw: the vocabulary's own rules are checked after the loop (the
            // nvfp4 mode may come later in argv) and the per-layer landing needs the
            // model's layer count, so it happens in the planner.
            options.kv_tier_formats_spec     = value(arg);
            options.kv_tier_formats_explicit = true;
        } else if (arg == "--nvfp4-mode") {
            const std::string_view mode = value(arg);
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
            // quantization; off takes the identity map on BOTH sides.
            const std::string_view mode = value(arg);
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
            // SEPARATION: three-state row scale (auto|off|<path>); validated at
            // plan time by the same parser NINFER_KV_ROWSCALE uses.
            options.kv_row_scale_spec     = std::string(value(arg));
            options.kv_row_scale_explicit = true;
        } else if (arg == "--kv-v-codec") {
            // SEPARATION: V-plane codec on the NVFP4 tier (iso3 default).
            const std::string_view mode = value(arg);
            if (mode == "iso3") {
                options.kv_v_codec = KvVCodec::Iso3;
            } else if (mode == "e2m1") {
                options.kv_v_codec = KvVCodec::E2M1;
            } else {
                throw std::invalid_argument("invalid kv-v-codec: " + std::string(mode) +
                                            " (expected iso3|e2m1)");
            }
            options.kv_v_codec_explicit = true;
        } else if (arg == "--spec") {
            options.speculative.backend = product::parse_speculative_backend(value(arg));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = parse_u32(value(arg), "draft-tokens");
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--raw-output") {
            options.raw_output = true;
        } else if (arg == "--print-token-ids") {
            options.print_token_ids = true;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--thinking-budget") {
            options.thinking_budget = parse_u32(value(arg), "thinking-budget");
        } else if (arg == "--reasoning-effort") {
            options.reasoning_effort = parse_reasoning_effort(value(arg));
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--cold-policy") {
            const std::string v = value(arg);
            if (v == "none" || v == "off") { options.cold_policy = ColdPolicy::None; }
            else if (v == "window") { options.cold_policy = ColdPolicy::Window; }
            else if (v == "host") { options.cold_policy = ColdPolicy::Host; }
            else if (v == "disk") { options.cold_policy = ColdPolicy::Disk; }
            else { throw std::invalid_argument("invalid cold-policy: " + v); }
            // Default the window only when the caller did not size it: this assignment
            // used to clobber --cold-keep-tokens regardless of order (and of the value).
            if (!options.cold_keep_tokens_explicit) { options.cold_keep_tokens = 128; }
        } else if (arg == "--cold-keep-tokens") {
            options.cold_keep_tokens          = parse_u32(value(arg), "cold-keep-tokens");
            options.cold_keep_tokens_explicit = true;
        } else if (arg == "--cold-host-bytes") {
            options.cold_host_bytes = parse_u64(value(arg), "cold-host-bytes");
        } else if (arg == "--max-cold-pages") {
            // 0 is meaningful here: it keeps the policy-derived pool size.
            options.max_cold_pages = parse_u32(value(arg), "max-cold-pages", true);
        } else if (arg == "--cold-disk-path") {
            options.cold_disk_path = value(arg);
        } else if (arg == "--cold-disk-bytes") {
            options.cold_disk_bytes = parse_u64(value(arg), "cold-disk-bytes");
            if (options.cold_disk_bytes == 0) {
                throw std::invalid_argument("--cold-disk-bytes must be positive");
            }
        } else if (arg == "--graph-capture-ceiling") {
            options.graph_capture_ceiling = parse_u32(value(arg), "graph-capture-ceiling");
        } else if (arg == "--no-lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Full;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--yarn") {
            options.yarn_enabled = true;
        } else if (arg == "--stop-token-id") {
            const std::uint32_t token = parse_u32(value(arg), "stop-token-id", true);
            if (token > static_cast<std::uint32_t>(std::numeric_limits<TokenId>::max())) {
                throw std::invalid_argument("--stop-token-id exceeds the token domain");
            }
            options.stop_token_ids.push_back(static_cast<TokenId>(token));
        } else if (arg == "--stop" || arg == "--reasoning-stop") {
            std::string text = value(arg);
            if (text.empty()) {
                throw std::invalid_argument(std::string(arg) + " must not be empty");
            }
            options.stop_strings.push_back(StopString{
                .text    = std::move(text),
                .channel = arg == "--stop" ? OutputChannel::Content : OutputChannel::Reasoning,
            });
        } else if (arg == "--temperature") {
            options.sampling.temperature = parse_float(value(arg), "temperature", 0.0F, 2.0F);
        } else if (arg == "--top-p") {
            options.sampling.top_p = parse_float(value(arg), "top-p", 0.0F, 1.0F);
        } else if (arg == "--top-k") {
            const std::uint32_t top_k = parse_u32(value(arg), "top-k", true);
            if (top_k > 20) { throw std::invalid_argument("--top-k must be in [0,20]"); }
            options.sampling.top_k = static_cast<std::int32_t>(top_k);
        } else if (arg == "--min-p") {
            options.sampling.min_p = parse_float(value(arg), "min-p", 0.0F, 1.0F);
        } else if (arg == "--presence-penalty") {
            options.sampling.presence_penalty =
                parse_float(value(arg), "presence-penalty", -2.0F, 2.0F);
        } else if (arg == "--frequency-penalty") {
            options.sampling.frequency_penalty =
                parse_float(value(arg), "frequency-penalty", -2.0F, 2.0F);
        } else if (arg == "--seed") {
            options.sampling.seed = parse_u64(value(arg), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }

    const bool has_prompt   = !options.prompt.empty();
    const bool has_messages = !options.messages_path.empty();
    if (has_prompt == has_messages) {
        throw std::invalid_argument("pass exactly one of --prompt or --messages");
    }
    if (options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens == 0) {
        throw std::invalid_argument("--kv-capacity must be positive");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.speculative.backend == SpeculativeBackend::DFlash && options.enable_vision) {
        throw std::invalid_argument("--spec dflash cannot be combined with --vision");
    }
    if (!options.enable_thinking && options.reasoning_effort) {
        throw std::invalid_argument("--reasoning-effort cannot be combined with --no-thinking");
    }
    if (!options.enable_thinking && options.thinking_budget) {
        throw std::invalid_argument("--thinking-budget cannot be combined with --no-thinking");
    }
    if (options.greedy) { options.sampling.temperature = 0.0F; }
    if (options.kv_tier_formats_explicit) {
        // Vocabulary rules only (bad tier, bad format, tier ordering, pure vs iso/e8).
        // The per-layer landing and the tier/table consistency checks need the model's
        // full-attention layer count and run in make_sequence_planner_impl.
        (void)product::kv_tier_formats_parse(options.kv_tier_formats_spec,
                                             options.kv_nvfp4_pure);
    }
    return options;
}

} // namespace ninfer::cli
