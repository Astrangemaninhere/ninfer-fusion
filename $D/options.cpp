#include "options.h"
#include "product/kv_rowscale_persist.h"
#include "product/speculative_options.h"
#include "product/kv_options.h"
#include "product/kv_tier_formats.h"
#include "product/kv_kv_bits.h"
#include "runtime/engine/bandwidth_governor.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
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

// `--cold-host-bytes` is documented as `N[g|m|k]` (usage_text below), so it accepts the binary
// suffixes the usage advertises. This is the parser catching up with the documented surface,
// not a new surface: a bare decimal goes through parse_u64 unchanged, so every command line
// that parses today keeps its exact value, and the only inputs that change verdict are the
// suffixed spellings the usage already promised (`32m`, `4g`, `256m` were refused by name).
// The multiplier is applied with an overflow check rather than wrapping.
std::uint64_t parse_bytes(const char* text, std::string_view label) {
    if (text == nullptr || *text == '\0') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                    (text == nullptr ? "" : text));
    }
    std::string_view digits(text);
    std::uint64_t multiplier = 1;
    const char suffix        = digits.back();
    if (suffix == 'k' || suffix == 'K') {
        multiplier = 1ULL << 10;
        digits.remove_suffix(1);
    } else if (suffix == 'm' || suffix == 'M') {
        multiplier = 1ULL << 20;
        digits.remove_suffix(1);
    } else if (suffix == 'g' || suffix == 'G') {
        multiplier = 1ULL << 30;
        digits.remove_suffix(1);
    }
    const std::string literal(digits);
    const std::uint64_t value = parse_u64(literal.c_str(), label);
    if (value > std::numeric_limits<std::uint64_t>::max() / multiplier) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return value * multiplier;
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
    // Canonical spelling first, deprecated spelling still accepted (one release), so an
    // operator who reads the engine's own printed `rk4v4-g64` / `iso4e-g16` line
    // can type it back. The canonical parser is product/kv_options.h:parse_kv_storage;
    // this list is NOT a copy of it -- the two differ on "fp8" (here Fp8E4M3Row256,
    // there Fp8Group16), which is why this is two extra accepted spellings and not a
    // delegation.
    if (text == "iso4e" || text == "iso3") { return KvCacheStorage::Iso3Group16; }
    if (text == "rk4v4" || text == "e8") { return KvCacheStorage::E8Group64; }
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

bool parse_on_off(const char* text, std::string_view label) {
    const std::string_view mode(text);
    if (mode == "on") { return true; }
    if (mode == "off") { return false; }
    throw std::invalid_argument("invalid " + std::string(label) + ": " + std::string(mode) +
                                " (expected on|off)");
}

// FreeToken observation switch. ops::ft::enabled() caches NINFER_FT_STATS in a
// function-local static the first time a kernel asks
// (src/ops/common/ft_stats.h:32-36), so "the flag beats the env" for this
// switch can only mean: write the variable before the engine runs. Parsing
// happens long before the first prefill, so committing it here is correct, and
// it is the same trade src/product/kv_rowscale_persist.h:609-626 makes for
// NINFER_KV_CALIB_DIR (this file already includes that header).
int set_process_env(const char* name, const char* value) {
#if defined(_WIN32)
    return ::_putenv_s(name, value);
#else
    return ::setenv(name, value, 1);
#endif
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> (--prompt <text>|--messages <messages.json>)\n"
           "       [--max-context N] [--kv-capacity N|auto] [--prefill-chunk N] [--max-new N]\n"
           "       [--prefill-chunk-mode dynamic|manual]\n"
           "           (who owns the prefill unit. dynamic (default) lets the bandwidth governor\n"
           "            install a unit inside [128, --prefill-chunk] and shrink it while decode\n"
           "            latency sits above its measured noise floor; manual pins the unit to\n"
           "            --prefill-chunk itself for the whole run. NINFER_FT_BW_GOV=0 is the\n"
           "            environment spelling of manual, =1 of dynamic; the flag wins. Either way\n"
           "            --prefill-chunk is the ceiling, and NINFER_FT_BW_TRACE=1 prints the mode\n"
           "            and the unit the engine installed.)\n"
           "       [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|iso4e|iso3|rk4v4|e8] [--kv-layer-storage SPEC] [--kv-bit-budget SPEC] [--spec auto|off|mtp|dflash|dflash2|dspark|none --draft-tokens N]\n           (dspark is a spelling of dflash, not a fourth backend: the DSpark drafter IS\n            the DFlash (v1) runtime, and the artifact's own weights identity\n            (weights_id=nvfp4-dspark) is what decides whether its Markov head runs --\n            dflash/markov_w1+w2 are bound only for that identity, and without them the\n            same --spec dflash drafts by plain argmax.)\n           (--kv-dtype names ONE global KV tier: bf16, int8, fp8 (row-scaled E4M3 D256),\n            nvfp4, and the pair this engine prints as iso4e-g16 / rk4v4-g64 -- iso3 and e8\n            are their deprecated aliases, accepted for one release. nvfp4 is the WEIGHT tier\n            the shipped qwen3_8_27b_nvfp4_modelopt artifact records for itself (weights_id:\n            nvfp4-modelopt), i.e. the tier this project ships. It cannot be combined\n            with --kv-bit-budget/--kv-bits, whose ceiling would replace the table it fills.\n           (--kv-bit-budget takes a ceiling per KV element, or per layer range:\n            \"0-7:8,8-63:4.5\"; it never exceeds the declared ceilings)\n"
           "       [--kv-residual-layers SPEC]\n"
           "           (the per-layer NVFP4 SECOND-STAGE RESIDUAL planes, SPEC = a bare layer\n"
           "            list over the family-wide 64 slots, \"2-5\" or \"0,3,7\". This is the\n"
           "            only flag that expresses a real plane SUBSET: an NVFP4 layer goes from\n"
           "            4 planes to 8. It is NVFP4-only -- naming any other tier is accepted\n"
           "            and inert -- and a residual-bearing layer is NOT cold-capable, so a\n"
           "            stack that names this cannot also spill to the cold pool.)\n"
           "       [--kv-bits B] [--kv-k-bits BK --kv-v-bits BV] [--kv-bits-mode joint|split|ceiling]\n"
           "       [--kv-codec-preference CODEC[,CODEC...]]\n"
           "           (the K/V bit-width entries. --kv-bits is the JOINT form: ONE overall\n"
           "            ceiling for the whole KV stack, one tier per full-attention layer.\n"
           "            --kv-k-bits/--kv-v-bits are the SPLIT form: K and V each get their own\n"
           "            ceiling and each plane's PER-LAYER layering is solved on its own, then\n"
           "            reconciled per layer. A layer whose K and V requirements meet no single\n"
           "            tier is refused BY INDEX with the missing (K,V) cell named, and the run\n"
           "            also prints the deployable plan at min(k,v) so a working spec is one\n"
           "            flag away. --kv-bits-mode split is the default; joint reads the same two\n"
           "            ceilings as one; ceiling deploys min(k,v) on purpose and reports the\n"
           "            headroom that could not be spent. Both planes always cost the SAME bits\n"
           "            per element: one DType drives both planes in this engine.)\n"
           "           (--kv-codec-preference is the OTHER axis of the same fit: which codec\n"
           "            is chosen among candidates that cost the SAME bits -- \"同 bit 宽度下\n"
           "            换一种量化\", NOT fewer bits. The value is an ORDER over the candidate\n"
           "            grammar " + product::kv_gear_candidate_list() +
           " (most-wanted first; built from the ladder rows the\n"
           "            solver actually accepts, not written out by hand here), e.g.\n"
           "            --kv-codec-preference iso4e or --kv-codec-preference iso4e,nvfp4. It\n"
           "            makes the solver try the named codecs first, so it decides only a tie\n"
           "            the fit's own columns already call equal: it cannot lower or raise a\n"
           "            penalty and cannot change the achieved bits. It needs --kv-bits. An\n"
           "            unknown name, an empty element, a repeat, a missing --kv-bits, a\n"
           "            --kv-layer-storage request, and a preference the fit could NOT honour\n"
           "            are each refused by name with the accepted list attached -- never\n"
           "            accepted and ignored. Worked example at 16 layers:\n"
           "              --kv-bits 4.5 --kv-quality-weight 0 --kv-codec-preference iso4e\n"
           "            returns 0-15:iso4e (dtype iso4e-g16), where the same command without\n"
           "            the preference returns 0-15:nvfp4 (nvfp4-g16): SAME bit count,\n"
           "            different codec. Without --kv-quality-weight the shipped ladder's iso4e\n"
           "            pin (penalty 200 against nvfp4's 30) makes the preference lose, and\n"
           "            that loss is reported as a refusal rather than silently ignored.)\n"
           "       [--kv-quality-weight W] [--kv-tier-scores FILE|INLINE]\n"
           "           (speed/quality slider for the bit-budget fit: W=0 fastest KV path,\n"
           "            W=1 most accurate. Needs a ceiling to act on (--kv-bits /\n"
           "            --kv-bit-budget / --kv-k-bits/--kv-v-bits); without one it is refused\n"
           "            instead of being silently ignored. --kv-tier-scores replaces the score\n"
           "            table; --kv-k-tier-scores / --kv-v-tier-scores give the split entry one\n"
           "            table per plane.)\n"
           "       [--capability-report]\n"
           "           (the BUILD capability surface's own entry point: like --kv-score-table it\n"
           "            runs with NO model and NO prompt. It prints the arch list this binary was\n"
           "            COMPILED for -- or the literal <unreported> plus the reason, when the build\n"
           "            did not publish one -- then the arch ladder and the per-format tensor-core\n"
           "            floors THIS build ships, each with its kernel citation, so \"what would\n"
           "            this build refuse, and why\" is answerable without an artifact. It is NOT\n"
           "            a device probe: the card in this machine is not queried and no capability\n"
           "            is claimed for it. A model path given as well continues the run afterwards.)\n"
           "       [--kv-score-table show|emit=PATH]\n"
           "           (the penalty table's OWN entry point: it runs with NO model and NO prompt.\n"
           "            'show' prints the table the planner would use plus the real provenance of\n"
           "            each column; 'emit=PATH' writes it in the grammar --kv-tier-scores reads,\n"
           "            so the table can be inspected, edited and fed straight back in. The\n"
           "            shipped quality column is a PRIOR, not a measurement, and the output says\n"
           "            so.)\n"
           "           (--spec defaults to auto; none turns speculation off)\n           (--spec mtp --draft-tokens k > 0 pins one MTP draft width; --spec mtp --draft-tokens 0,\n            or with the width left unset, or NINFER_MTP_ADAPTIVE=1, uses the ADAPTIVE ladder,\n            where the mtp_window_cut criterion picks the rung per round)\n           (--spec mtp --draft-tree L,d verifies a TREE instead of one chain: L rank-paths\n            per depth, d draft steps, one verify column per node plus the anchor, so the node\n            budget L*d <= 15. That budget IS the round's draft width, so --draft-tree and\n            --draft-tokens are two spellings of one number and cannot both be given.\n            --draft-tree 1,d is the degenerate chain and must reproduce --draft-tokens d\n            token-for-token, which is the tree path's instrument check.)\n"
           "       [--kv-tier-formats hot=auto|bf16|int8,tail=...,cold=...] [--nvfp4-mode fusion|pure]\n"
           "           (hot = the resident format of every full-attention layer; only bf16 and\n"
           "            int8 have a resident codec. cold is the aged-out tier format; the cold\n"
           "            slot codec is derived from the layer dtype and only int8 is reachable\n"
           "            today. tail is accepted only when it repeats hot: the engine has no\n"
           "            recent-window tier yet. --nvfp4-mode pure forbids nvfp4/iso4e/rk4v4.)\n"
           "       [--kv-rotation on|off] [--kv-row-scale auto|off|FILE] [--recalibrate]\n"
           "       [--kv-v-codec iso4e|e2m1]\n"
           "           (component switches for the NVFP4/FP8/ISO4E KV tiers.\n"
           "            --kv-rotation off takes the identity SO(4) map on BOTH the K write\n"
           "            and the Q read, so QK^T stays exact and only the quantization domain\n"
           "            changes. --kv-row-scale off takes the identity row scale in the kernel\n"
           "            (no identity file needed); auto keeps the baked table; FILE loads an\n"
           "            NINFERKVRS1 sidecar (NINFER_KV_ROWSCALE is the env equivalent).\n"
           "            auto also runs the calibration loop: a table persisted next to the\n"
           "            artifact (MODEL.kvrowscale.bin) is loaded and the capture is skipped,\n"
           "            and when there is none -- or it was baked for another model or another\n"
           "            KV configuration -- THIS run captures once and writes one. The first\n"
           "            calibration run needs --no-cuda-graph. --recalibrate ignores the\n"
           "            persisted table, captures again and overwrites it.\n"
           "            --kv-v-codec e2m1 stores NVFP4-tier V as E2M1 instead of ISO4E and is\n"
           "            refused when a V residual plane or the cold pool is active.)\n"
           "       [--yarn]\n"
           "           (static YaRN factor-4 rope: the rope domain goes to 4x the variant's native\n"
           "            context and the yarn4 attention scaling 1.1386 is folded into the sincos\n"
           "            tables (include/ninfer/ops/rope.h). It moves the rope domain the captured K\n"
           "            is built through, so it is part of the row-scale fingerprint\n"
           "            (product/kv_rowscale_persist.h rope_regime, printed as ;rope=1 in the\n"
           "            NINFERKVRS1 tag): a table baked by a run that did not name --yarn does not\n"
           "            validate for a run that does. Off when the flag is absent.)\n"
           "       [--lm-head-draft]\n"
           "       [--no-lm-head-draft]\n"
           "           (the drafter's proposal head, stated explicitly: --lm-head-draft selects the\n"
           "            optimized head, --no-lm-head-draft pins the full-vocabulary head\n"
           "            (ProposalHead::Full). Full is also what the profile-resolved Auto leaves in\n"
           "            place for an artifact that carries no text/draft_head, so this flag exists to\n"
           "            make a run's head independent of what the artifact resolves\n"
           "            (include/ninfer/types.h, ProposalHead).)\n"
           "       [--temperature F] [--top-p F] [--top-k N] [--min-p F]\n"
           "       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...\n"
           "       [--raw-output] [--print-token-ids] [--no-thinking] [--thinking-budget N]\n"
           "       [--reasoning-effort low|medium|xhigh] [--vision]\n"
           "       [--cold-policy none|off|window|host|disk|host-then-disk|host+disk] "
           "(host+disk is an accepted equivalent spelling of host-then-disk; "
           "docs/cli.md)\n"
           "[--cold-keep-tokens N]\n"
           "       [--max-cold-pages N] [--kv-unload-watermark-pages N]\n"
           "       [--append-context-text <text>]\n"
           "       [--cold-host-bytes N[g|m|k]]\n"
           "       [--cold-disk-path DIR] [--cold-disk-bytes N]\n"
           "       [--weight-host-bytes N] [--weight-device-arena-bytes N]\n"
           "       [--weight-prefetch-layers N] [--weight-span-floor-bytes N]\n"
           "       [--no-cuda-graph] [--graph-capture-ceiling N]\n"
           "       [--ft-stats on|off]\n"
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
           "--ft-stats on enables the FreeToken per-layer attention-energy observation "
           "(NINFER_FT_STATS=1 is the env spelling); it is off by default and the flag "
           "beats the env. Its consumer -- the periodic KV relayout -- is a serve-side "
           "feature (--kv-auto-relayout there), because this front end has no decision "
           "cycle.\n"
           "--kv-unload-watermark-pages N is the free text-KV pool pages at or below which "
           "the Engine proactively unloads the blocks its semantic directory judges "
           "unloadable, instead of waiting for the pool to overflow. 0 = off (the "
           "pre-watermark behaviour); the default derives the reserve from --prefill-chunk. "
           "NINFER_KV_UNLOAD_WATERMARK_PAGES is the env spelling and the flag beats it; an "
           "unparseable value from either is refused by name, never ignored.\n"
           "--append-context-text <text> encodes <text> with the artifact's own tokenizer (raw: "
           "no chat template, no implicit special token) and, mid-run, appends that run of tokens "
           "to the running request and prefills it, so the model can attend to it from the next "
           "round. It is input, not output: the appended tokens are never reported in the "
           "generated ids and never consume --max-new. It is armed before the request's first "
           "decode round, and [context-append] on stderr reports what was serviced. Off when the "
           "flag is absent.\n"
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
    // --kv-score-table is the penalty table's own entry point and needs no model, so it
    // is accepted in the model-path position too (`ninfer --kv-score-table show`):
    // there is no artifact it could need, and requiring one would make the entry
    // unreachable from a bare shell. A model path given as well still wins and the run
    // then continues after the table is written.
    const bool score_table_standalone =
        argc >= 2 && std::string_view(argv[1]) == "--kv-score-table";
    // A model path is a POSITION, not a flag, so a `--`-prefixed argv[1] is not
    // one. Assigning it to artifact_path anyway made `ninfer --some-flag bad`
    // report the flag's VALUE as an unknown argument, and hid the flag's own
    // refusal behind it -- the switch was never reached, so a wired flag read as
    // an unwired one. Flags in that position are parsed as flags; a run that then
    // names no model says so by name (checked after the loop).
    const bool first_arg_is_flag =
        argc >= 2 && std::string_view(argv[1]).size() >= 2 &&
        std::string_view(argv[1]).substr(0, 2) == "--";
    if (!first_arg_is_flag) { options.artifact_path = argv[1]; }
    bool kv_capacity_explicit = false;
    // --draft-tokens' own "the operator named it" gate, in the same shape as
    // unload_watermark_explicit below and for the same reason: the MTP adaptive escape
    // hatch (NINFER_MTP_ADAPTIVE) must not overwrite a width the FLAG pinned. Keeps the
    // three cases distinct -- flag absent (the env selects the ladder), `--draft-tokens 0`
    // (adaptive because the flag says so), `--draft-tokens N>0` (the flag wins), which is
    // the CLI > environment > default order docs/cli.md:217 states and the MTP sentence at
    // docs/cli.md:419 repeats.
    bool draft_tokens_explicit = false;
    // The watermark's "the operator named it" gate. It is the SAME gate
    // src/serve/serve_options.cpp:198/523/836 uses, so the two front ends cannot
    // disagree about whether a flag beats NINFER_KV_UNLOAD_WATERMARK_PAGES.
    bool unload_watermark_explicit = false;

    // first_arg_is_flag subsumes score_table_standalone: "--kv-score-table" itself
    // begins with "--", so both cases read every flag from index 1.
    for (int i = first_arg_is_flag ? 1 : 2; i < argc; ++i) {
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
        } else if (arg == "--prefill-chunk-mode") {
            // dynamic | manual. The two spellings, their validation and the mode -> mechanism
            // mapping live in one place (BandwidthGovernor::parse_mode/adapts, included below), so
            // this flag lands on the same state as NINFER_FT_BW_GOV=0 would.
            options.prefill_chunk_mode = runtime::BandwidthGovernor::parse_mode(value(arg));
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
        } else if (arg == "--kv-bits") {
            // JOINT form: ONE overall ceiling for the whole KV stack ("合起来整体定").
            options.kv_joint_bits =
                parse_float(value(arg), "--kv-bits", 0.01F, 16.0F);
            options.kv_kv_bits_explicit = true;
        } else if (arg == "--kv-k-bits") {
            // SPLIT form: K gets its own ceiling, and its own per-layer layering.
            options.kv_k_bits = parse_float(value(arg), "--kv-k-bits", 0.01F, 16.0F);
            options.kv_kv_bits_explicit = true;
        } else if (arg == "--kv-v-bits") {
            // SPLIT form: V gets its own ceiling, and its own per-layer layering.
            options.kv_v_bits = parse_float(value(arg), "--kv-v-bits", 0.01F, 16.0F);
            options.kv_kv_bits_explicit = true;
        } else if (arg == "--kv-bits-mode") {
            // Which reading of a per-plane request runs: joint | split | ceiling.
            options.kv_bits_mode = product::kv_bits_mode_from_name(value(arg));
            options.kv_bits_mode_explicit = true;
        } else if (arg == "--kv-codec-preference") {
            // SLIDERWIRE: WHICH codec the fit picks among candidates that cost the SAME bits.
            // An ORDERED comma-separated list over the candidate grammar, most-wanted first
            // (product/kv_bit_budget.h kv_gear_candidate_list()). The grammar is FIXED, so
            // unlike the ceiling knobs this is resolved at parse time, where an unknown name
            // can be refused by name with the accepted list attached -- the alternative, a
            // list the planner silently drops, is this project's worst outcome.
            const std::string_view spec = value(arg);
            options.kv_codec_preference.clear();
            options.kv_codec_preference_explicit = true;
            if (spec.empty()) {
                throw std::invalid_argument(
                    "--kv-codec-preference needs at least one codec name; the candidate grammar "
                    "is " + product::kv_gear_candidate_list() +
                    " (e.g. --kv-codec-preference iso4e)");
            }
            std::size_t cursor = 0;
            while (cursor <= spec.size()) {
                const std::size_t comma = spec.find(',', cursor);
                const std::string_view item =
                    comma == std::string_view::npos
                        ? spec.substr(cursor)
                        : spec.substr(cursor, comma - cursor);
                cursor = comma == std::string_view::npos ? spec.size() + 1 : comma + 1;
                if (item.empty()) {
                    throw std::invalid_argument(
                        "--kv-codec-preference has an empty codec name in '" +
                        std::string(spec) +
                        "': the value is a comma-separated list with no empty element. The "
                        "candidate grammar is " + product::kv_gear_candidate_list() +
                        " (e.g. --kv-codec-preference iso4e,nvfp4)");
                }
                const std::int32_t slot = product::kv_gear_candidate_slot(item);
                if (slot < 0) {
                    throw std::invalid_argument(
                        "invalid kv-codec-preference codec: " + std::string(item) +
                        " (the candidate grammar is " + product::kv_gear_candidate_list() +
                        "; e8k3/e8k2 are rows of the ladder but outside it on every path, "
                        "because no decode or append kernel in this tree reads a 3-bit or "
                        "2-bit K code plate)");
                }
                for (const std::int32_t seen : options.kv_codec_preference) {
                    if (seen == slot) {
                        throw std::invalid_argument(
                            "--kv-codec-preference names '" + std::string(item) +
                            "' twice: the value is an ORDER over the candidate grammar, so a "
                            "repeat cannot express anything a single entry does not, and it "
                            "would be accepted and read by nothing. Give each codec at most "
                            "once (want " + product::kv_gear_candidate_list() + ").");
                    }
                }
                options.kv_codec_preference.push_back(slot);
            }
        } else if (arg == "--kv-k-tier-scores") {
            // The K plane's own score columns (quality/speed) for the split entry.
            options.kv_k_tier_scores = value(arg);
        } else if (arg == "--kv-v-tier-scores") {
            // The V plane's own score columns for the split entry.
            options.kv_v_tier_scores = value(arg);
        } else if (arg == "--capability-report") {
            // The BUILD capability surface's OWN entry point. No value: it takes no argument,
            // because everything it prints comes from this build's own tables. Handled in
            // main() BEFORE any device work, so the entry runs with no model at all; see
            // src/core/arch_caps.h render_build_capability_surface().
            options.capability_report_requested = true;
        } else if (arg == "--kv-score-table") {
            // The penalty table's OWN entry point. "show" prints it, "emit=<path>"
            // writes it; both are handled in main() BEFORE any device work, so the
            // entry runs with no model at all (see apps/cli/main.cpp).
            options.kv_score_table_spec     = value(arg);
            options.kv_score_table_explicit = true;
        } else if (arg == "--kv-layer-storage") {
            options.kv_layer_storage_spec = value(arg);
            options.kv_layer_storage_explicit = true;
        } else if (arg == "--kv-residual-layers") {
            // Stored raw and parsed in main() next to the EngineOptions it produces,
            // which is also where --kv-layer-storage is turned into a table. The parse
            // still precedes Engine construction, so a typo is refused before any device
            // work (rule: behavioural differences that happen at parse time are settled
            // on the host, never with the GPU in the loop).
            options.kv_residual_layers_spec     = value(arg);
            options.kv_residual_layers_explicit = true;
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
        } else if (arg == "--recalibrate") {
            // N3 runtime loop: re-run the calibration capture and overwrite the
            // persisted row-scale table.
            options.recalibrate = true;
        } else if (arg == "--kv-v-codec") {
            // SEPARATION: V-plane codec on the NVFP4 tier. The canonical spelling is
            // `iso4e` -- that is the name the product's own enum carries
            // (KvVCodecMode::Iso4e, product/kv_bit_budget.h) and the name its
            // printer emits (:1497). `iso3` is the DEPRECATED spelling of the SAME
            // state and is accepted for one release with a warning, exactly as
            // --kv-dtype treats its aliases (options.cpp:98-104, usage_text :159).
            const std::string_view mode = value(arg);
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
        } else if (arg == "--spec") {
            options.speculative.backend = product::parse_speculative_backend(value(arg));
        } else if (arg == "--draft-tokens") {
            // `0` IS the adaptive spelling of the MTP draft width -- product/speculative_options.h
            // states it ("0 means ADAPTIVE"), the engine's own adaptive note states it
            // (`--spec mtp --draft-tokens 0`), and docs/cli.md:194 documents it. parse_u32's
            // default then refused it, so the documented entry was unreachable from THIS front end
            // while src/serve/serve_options.cpp already accepted it (parse_nonnegative_int). This
            // makes the two parse points agree. It weakens no gate: --spec dflash|dflash2 still
            // refuse 0 downstream in product::validate_speculative_cli_options.
            options.speculative.draft_tokens =
                parse_u32(value(arg), "draft-tokens", /*allow_zero=*/true);
            draft_tokens_explicit = true;
        } else if (arg == "--draft-tree") {
            // L,d -- the MTP tree verify shape. Parsed here rather than in the planner so a
            // malformed shape is a CLI error; the planner repeats the domain check because the
            // server reaches it too. The width gate is L*d <= 15 nodes and lives in the planner
            // (targets/qwen3_6/impl/runtime/layouts_impl.h), where the target's own MTP draft
            // domain constant is in scope.
            const std::string tree(value(arg));
            const std::size_t comma = tree.find(',');
            if (comma == std::string::npos || comma == 0 || comma + 1 == tree.size()) {
                throw std::invalid_argument("invalid draft-tree: " + tree + " (expected L,d)");
            }
            options.speculative.draft_tree_paths =
                parse_u32(tree.substr(0, comma).c_str(), "draft-tree L");
            options.speculative.draft_tree_depth =
                parse_u32(tree.substr(comma + 1).c_str(), "draft-tree d");
            if (options.speculative.draft_tree_paths > 16 ||
                options.speculative.draft_tree_depth > 15) {
                throw std::invalid_argument(
                    "invalid draft-tree: L must be in [1,16] and d in [1,15]");
            }
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
            // Layered cold tier: --cold-host-bytes bounds the pinned tier and the
            // excess spills to --cold-disk-path under --cold-disk-bytes.
            // "host+disk" is an equivalent spelling of the same policy.
            else if (v == "host-then-disk" || v == "host+disk") {
                options.cold_policy = ColdPolicy::HostThenDisk;
            }
            else { throw std::invalid_argument("invalid cold-policy: " + v); }
            // Default the window only when the caller did not size it: this assignment
            // used to clobber --cold-keep-tokens regardless of order (and of the value).
            if (!options.cold_keep_tokens_explicit) { options.cold_keep_tokens = 128; }
        } else if (arg == "--cold-keep-tokens") {
            options.cold_keep_tokens          = parse_u32(value(arg), "cold-keep-tokens");
            options.cold_keep_tokens_explicit = true;
        } else if (arg == "--cold-host-bytes") {
            options.cold_host_bytes = parse_bytes(value(arg), "cold-host-bytes");
        } else if (arg == "--max-cold-pages") {
            // 0 is meaningful here: it keeps the policy-derived pool size.
            options.max_cold_pages = parse_u32(value(arg), "max-cold-pages", true);
        } else if (arg == "--kv-unload-watermark-pages") {
            // The proactive free-pool watermark. 0 is the OFF switch and is not a
            // typo, and kUnloadWatermarkDerive (the default) is reachable on purpose,
            // so an explicit value is never clamped onto the sentinel. Parsed exactly
            // as src/serve/serve_options.cpp:510-523 parses it -- parse_u64 plus an
            // explicit range refusal -- so both front ends name the same failures the
            // same way instead of drifting into two vocabularies for one knob.
            const std::uint64_t pages =
                parse_u64(value(arg), "kv-unload-watermark-pages");
            if (pages > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--kv-unload-watermark-pages is out of range");
            }
            options.unload_watermark_pages = static_cast<std::uint32_t>(pages);
            unload_watermark_explicit      = true;
        } else if (arg == "--cold-disk-path") {
            options.cold_disk_path = value(arg);
        } else if (arg == "--cold-disk-bytes") {
            options.cold_disk_bytes = parse_u64(value(arg), "cold-disk-bytes");
            if (options.cold_disk_bytes == 0) {
                throw std::invalid_argument("--cold-disk-bytes must be positive");
            }
        } else if (arg == "--weight-host-bytes") {
            options.weight_host_offload_bytes = parse_u64(value(arg), "weight-host-bytes");
        } else if (arg == "--weight-device-arena-bytes") {
            options.weight_device_arena_bytes = parse_u64(value(arg), "weight-device-arena-bytes");
        } else if (arg == "--weight-prefetch-layers") {
            options.weight_prefetch_layers = parse_u32(value(arg), "weight-prefetch-layers");
            if (options.weight_prefetch_layers < 2) {
                throw std::invalid_argument(
                    "--weight-prefetch-layers below 2 would let the arena slot of the layer "
                    "being computed be overwritten by its own prefetch");
            }
        } else if (arg == "--weight-span-floor-bytes") {
            options.weight_span_floor_bytes = parse_u64(value(arg), "weight-span-floor-bytes");
        } else if (arg == "--graph-capture-ceiling") {
            options.graph_capture_ceiling = parse_u32(value(arg), "graph-capture-ceiling");
        } else if (arg == "--ft-stats") {
            // FreeToken step 1 observation (src/ops/common/ft_stats.h). Off by
            // default; `on` is what makes the per-layer energy lines (and the
            // serve-side auto-relayout consumer) possible. on|off is the same
            // shape --kv-rotation uses on the serve side.
            options.ft_stats = parse_on_off(value(arg), "ft-stats");
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
        } else if (arg == "--append-context-text") {
            // The value is MANDATORY and non-empty on purpose: an empty segment would arm an append
            // that appends nothing, which is indistinguishable from the leg being off and would make
            // a mistyped acceptance run look green.
            options.append_context_text = value(arg);
            if (options.append_context_text.empty()) {
                throw std::invalid_argument("--append-context-text must not be empty");
            }
            options.append_context_explicit = true;
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    // ...and a run that consumed flags without ever naming a model says so by name.
    // --kv-score-table is its own entry point and needs no artifact, which is why it
    // is the one exception.
    if (options.artifact_path.empty() && !options.kv_score_table_explicit &&
        !options.capability_report_requested) {
        throw std::invalid_argument(".ninfer model path is required");
    }

    // THE WATERMARK'S ENV OVERRIDE. Precedence is CLI > env > default, gated on the
    // same `unload_watermark_explicit` variable src/serve/serve_options.cpp:836-847
    // gates on: a flag is one run's deliberate act, an environment variable is a
    // deployment's standing wish. An unparseable value is REFUSED rather than
    // ignored -- a silently ignored watermark is a run that believes it is protected
    // and is not, which is precisely the failure this knob exists to name.
    if (!unload_watermark_explicit) {
        if (const char* env = std::getenv("NINFER_KV_UNLOAD_WATERMARK_PAGES");
            env != nullptr && env[0] != '\0') {
            const std::uint64_t pages =
                parse_u64(env, "NINFER_KV_UNLOAD_WATERMARK_PAGES");
            if (pages > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument(
                    "NINFER_KV_UNLOAD_WATERMARK_PAGES is out of range");
            }
            options.unload_watermark_pages = static_cast<std::uint32_t>(pages);
        }
    }

    // ===========================================================================
    // TEMPORARY adaptive-MTP escape hatch: NINFER_MTP_ADAPTIVE=1.
    // DELETE THIS BLOCK when the real fix lands (the three follow-ups are listed below).
    //
    // Why it exists: 0 is the adaptive spelling of the MTP draft width -- 0 lets the
    // add-form survival/cost criterion in mtp_window_cut.h pick the ladder rung per round,
    // while k > 0 pins one captured width (bit-identical to the recorded fixed-k runs).
    // Both downstream gates already accept the MTP case: product::validate_speculative_cli_options
    // takes [0,15] for mtp (product/speculative_options.h:61) and validate_target_options
    // takes 0 or [1,kMaximumMtpDraftTokens] (targets/qwen3_6/impl/runtime/layouts_impl.h:890).
    // What refuses it is THIS parser: parse_u32() is called without allow_zero at
    // apps/cli/options.cpp:237, and it throws "invalid draft-tokens: 0" from the template at
    // :33. That single argument is the only reason the adaptive CLI entry is unreachable.
    //
    // The entry it restores:
    //   NINFER_MTP_ADAPTIVE=1 <bin> <artifact> ... --spec mtp
    // (--spec may be omitted: this front end defaults it to auto, which the env pins to mtp.)
    // Read AFTER the argument loop and BEFORE every validation below.
    //
    // Scope, deliberately narrow: only the mtp path is touched. An explicit --spec
    // none|dflash|dflash2 is left exactly as given, and no k > 0 is ever rewritten, so every
    // recorded fixed-k measurement keeps its behaviour.
    //
    // THE REAL FIX IS NOT THIS ENV VAR. It is:
    //   (a) letting the criterion -- the add-form judgement in mtp_window_cut.h -- decide the
    //       width where the width is actually chosen, instead of 0 travelling as a sentinel
    //       that every gate has to re-interpret;
    //   (b) making the upper bound VARIANT-aware instead of the hard-wired 15 in
    //       product::validate_speculative_cli_options: kMaximumMtpDraftTokens is 15 on 27b but
    //       only 5 on 35b_a3b and muse_glimmer_30b (src/targets/*/impl/config.h), and that
    //       per-variant bound is reachable only in validate_target_options;
    //   (c) making the two parse points agree -- src/serve/serve_options.cpp already takes 0
    //       (parse_nonnegative_int), apps/cli/options.cpp does not; the same env is wired into
    //       both so the two front ends cannot drift while the real fix is pending.
    // ===========================================================================
    {
        const char* adaptive_env = std::getenv("NINFER_MTP_ADAPTIVE");
        if (adaptive_env != nullptr && std::atoi(adaptive_env) != 0) {
            // --spec auto is both the default and the only spelling of "let the artifact
            // decide", so Auto here means no explicit opt-out was given. The env is named for
            // mtp: an explicit non-mtp backend is never rewritten.
            if (options.speculative.backend == SpeculativeBackend::Auto) {
                options.speculative.backend = SpeculativeBackend::Mtp;
            }
            if (options.speculative.backend == SpeculativeBackend::Mtp) {
                if (draft_tokens_explicit) {
                    // The flag beats the env -- docs/cli.md:217 (`CLI > environment >
                    // default`, the order every other switch here practises) and
                    // docs/cli.md:419 ("An explicit --draft-tokens value wins over it").
                    // 0 is adaptive because the FLAG says so; N > 0 pins N.
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

    // --draft-tree L,d is the MTP tree verify shape and REPLACES the scalar draft width: the
    // node budget L*d IS the round's extent, so the two spellings contradict each other. Refused
    // rather than ignored -- silently dropping one of them would run a width (and a graph) the
    // command line did not name.
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

    // K/V bit widths (product/kv_kv_bits.h). Two entries of their own, and every
    // combination with the pre-existing budget knobs is a contradiction rather than
    // a merge, so they are refused here with the reason named (the planner repeats
    // the checks because the server can also reach it).
    if (options.kv_kv_bits_explicit && options.kv_bit_budget_explicit) {
        throw std::invalid_argument(
            "--kv-bit-budget and --kv-bits/--kv-k-bits/--kv-v-bits are two spellings of the "
            "same ceiling set: --kv-bit-budget is the plane-agnostic form, --kv-bits (or "
            "--kv-k-bits + --kv-v-bits) the K/V form. Give one of them.");
    }
    // kv4 P3 REFUSE: the same rule for the OLD spelling of the ceiling. ninfer-serve
    // has refused this pair all along (src/serve/serve_options.cpp:703); the CLI did
    // not, and the planner then skipped the whole --kv-bit-budget resolution because
    // --kv-layer-storage had already filled the table (layouts_impl.h:
    // "if (options.kv_bit_budget_explicit && !storage_explicit)"), so the ceiling was
    // accepted and read by nothing -- this project's worst outcome.
    if (options.kv_bit_budget_explicit && options.kv_layer_storage_explicit) {
        throw std::invalid_argument(
            "--kv-bit-budget and --kv-layer-storage are mutually exclusive: the budget "
            "is resolved into exactly the table --kv-layer-storage provides");
    }
    // res-kv5 FIX: --kv-dtype x a ceiling is the SAME contradiction as the two rules
    // above -- --kv-dtype is a table provider (it fills every slot with one global
    // tier) and a ceiling is resolved into exactly the per-layer table that replaces
    // it, so one of the two would be read by nothing. Measured BEFORE this fix:
    // `--kv-bit-budget 4.5 --kv-dtype int8` and `--kv-bit-budget 4.5` produced the
    // identical dtype (per-layer 0-9:rk4v4-g64 10-15:nvfp4-g16) and the identical
    // payload (278.00 MiB) -- the int8 was accepted and dropped -- while
    // `--kv-dtype int8` alone is int8-g64 / 528.00 MiB.
    if (options.kv_cache_explicit &&
        (options.kv_bit_budget_explicit || options.kv_kv_bits_explicit)) {
        throw std::invalid_argument(
            "--kv-dtype and --kv-bit-budget/--kv-bits are mutually exclusive: --kv-dtype "
            "pins the global KV tier, and a ceiling is resolved into exactly the per-layer "
            "table that would replace it -- so one of the two would be accepted and read "
            "by nothing. Give one of them.");
    }
    if (options.kv_kv_bits_explicit && options.kv_layer_storage_explicit) {
        throw std::invalid_argument(
            "--kv-bits/--kv-k-bits/--kv-v-bits and --kv-layer-storage are mutually exclusive: "
            "the budget is resolved into exactly the table --kv-layer-storage provides.");
    }
    if (options.kv_bits_mode_explicit && !options.kv_kv_bits_explicit) {
        throw std::invalid_argument(
            "--kv-bits-mode needs a K/V bit request to act on: it names which reading of "
            "--kv-k-bits/--kv-v-bits runs (joint|split|ceiling). Without one nothing would "
            "read it, and a flag that is accepted and ignored is this project's worst "
            "outcome -- add --kv-bits, --kv-k-bits or --kv-v-bits.");
    }
    if (options.kv_quality_weight >= 0.0 && !options.kv_bit_budget_explicit &&
        !options.kv_kv_bits_explicit) {
        // The slider is read only inside a bit-budget resolution, so without a ceiling it
        // used to parse, be accepted, and change nothing. Name the flags that make it act.
        throw std::invalid_argument(
            "--kv-quality-weight / --kv-tier-scores need a ceiling to act on: both are read "
            "only inside the bit-budget fit. Add --kv-bits, --kv-k-bits/--kv-v-bits or "
            "--kv-bit-budget, or drop the flag.");
    }
    // The score-table half of the same rule (product/kv_kv_bits.h refuses all three too, but
    // doing it here means the operator hears about it before any model work starts).
    const bool any_score_table = !options.kv_tier_scores.empty() ||
                                 !options.kv_k_tier_scores.empty() ||
                                 !options.kv_v_tier_scores.empty();
    if (any_score_table && options.kv_quality_weight < 0.0) {
        throw std::invalid_argument(
            "--kv-tier-scores / --kv-k-tier-scores / --kv-v-tier-scores need "
            "--kv-quality-weight to act on: a table's two columns are combined only by the "
            "weight (penalty = w*quality + (1-w)*speed), so without one the shipped "
            "single-penalty ladder runs and the table would be read by nothing. Add "
            "--kv-quality-weight W (0 = fastest, 1 = most accurate), or drop the table.");
    }
    if (options.kv_joint_bits > 0.0 && (options.kv_k_bits > 0.0 || options.kv_v_bits > 0.0)) {
        throw std::invalid_argument(
            "--kv-bits and --kv-k-bits/--kv-v-bits are two spellings of the same ceiling set: "
            "--kv-bits is the ONE overall ceiling, the two per-plane ceilings cover it. Give "
            "one or the other; --kv-bits-mode picks the reading of the per-plane form.");
    }
    // SLIDERWIRE: --kv-codec-preference. Same two-sided rule as --kv-quality-weight below:
    // the preference is read ONLY inside a bit-budget fit, so naming it without one used to
    // be a silent no-op. Refused here with the flag that makes it act, AND refused against
    // the pre-existing --kv-bit-budget / --kv-layer-storage spellings, whose resolution path
    // (src/targets/.../layouts_impl.h, options.kv_bit_budget_explicit) does not carry a
    // candidate order at all: on those the preference would be accepted and read by nothing.
    // --kv-bits (the joint K/V form) is the one entry that carries it.
    if (options.kv_codec_preference_explicit && !options.kv_kv_bits_explicit) {
        throw std::invalid_argument(
            "--kv-codec-preference needs --kv-bits to act on: it names which codec the solver "
            "picks among candidates that cost the SAME bits, and only the --kv-bits (ONE "
            "overall ceiling) resolution carries a candidate order. --kv-bit-budget and "
            "--kv-layer-storage fix the per-layer table directly and would leave the "
            "preference accepted and read by nothing. Add --kv-bits (e.g. --kv-bits 4.5 "
            "--kv-quality-weight 0 --kv-codec-preference iso4e), or drop the flag.");
    }
    if (options.kv_codec_preference_explicit && options.kv_layer_storage_explicit) {
        throw std::invalid_argument(
            "--kv-codec-preference and --kv-layer-storage are mutually exclusive: "
            "--kv-layer-storage IS the per-layer table, so the codec of every layer is already "
            "named there and the preference would be accepted and read by nothing.");
    }
    if (options.kv_bits_mode_explicit && options.kv_joint_bits > 0.0 &&
        options.kv_bits_mode != KvBitsMode::Joint) {
        throw std::invalid_argument(
            "--kv-bits-mode was given together with --kv-bits, which names ONE overall "
            "ceiling: with a single ceiling there is no second reading for the mode to "
            "select, so it would be accepted and read by nothing. Give --kv-k-bits/--kv-v-bits "
            "for the split and ceiling readings, or drop --kv-bits-mode.");
    }
    // The JOINT reading takes ONE ladder, so a per-plane table has nothing to fit there.
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

    // FreeToken: commit an explicit --ft-stats to the variable the observation
    // reads (see set_process_env above). An unset flag leaves the operator's
    // NINFER_FT_STATS alone, which is what keeps "env only" working.
    if (options.ft_stats.has_value() &&
        set_process_env("NINFER_FT_STATS", *options.ft_stats ? "1" : "0") != 0) {
        throw std::invalid_argument("cannot set NINFER_FT_STATS for --ft-stats");
    }

    const bool has_prompt   = !options.prompt.empty();
    const bool has_messages = !options.messages_path.empty();
    // --kv-score-table is its own entry point and needs no prompt: main() acts on it
    // before any prompt or model is touched, so requiring one here would make the
    // entry unreachable from a bare shell.
    if (!options.kv_score_table_explicit && !options.capability_report_requested &&
        has_prompt == has_messages) {
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
    if (options.recalibrate && options.kv_row_scale_explicit &&
        !product::kv_rowscale_spec_is_auto(options.kv_row_scale_spec)) {
        // The loop only owns the auto state; an explicit off/file has no table to
        // overwrite, so the combination is a contradiction rather than a no-op.
        throw std::invalid_argument(
            "--recalibrate cannot be combined with --kv-row-scale " +
            options.kv_row_scale_spec + " (only 'auto' runs the calibration loop)");
    }
    return options;
}

} // namespace ninfer::cli
