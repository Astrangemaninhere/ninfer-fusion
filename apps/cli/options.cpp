#include "options.h"
#include "product/kv_rowscale_persist.h"
#include "product/speculative_options.h"
#include "product/kv_options.h"
#include "product/kv_tier_formats.h"
#include "product/kv_kv_bits.h"
#include "runtime/engine/bandwidth_governor.h"
// --stage-layers' grammar, cover and run-shape refusals. Linked in by the front door so a
// mis-shaped SPEC is refused before any artifact is opened -- and so the ONE parser in
// core/stage_plan.h is called here and, with the artifact's own layer count, in the runtime.
#include "core/stage_plan.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "spec/inject_channel.h"

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

// ---------------------------------------------------------------------------
// SET-BUT-EMPTY-VALUE, for the KV slider family (every `--kv-*` flag here that
// takes a value).  An empty value is not a spelling of anything in any of these
// grammars, and the reason that is not harmless is that the consumers which do
// meet one meet it by IMPUTING a state:
//
//   parse_kv_layer_storage_spec("")   -> a table whose mask is all-false, i.e. "no
//                                        layer was named" -- so the flag parsed,
//                                        was marked explicit, and was read by
//                                        nothing (its own parser documents the
//                                        mask as the thing that decides).
//   kv_rowscale_mode_from_spec("")    -> Auto, i.e. exactly what
//                                        `--kv-row-scale auto` asks for.
//   --kv-tier-scores ""               -> the built-in table, i.e. as if unset.
//   parse_kv_residual_layers_spec("") -> an all-false table under
//                                        kv_residual_explicit = true, which is the
//                                        very "explicitly no residuals" state
//                                        apps/cli/main.cpp warns an all-false table
//                                        must not be handed over unconditionally.
//
// The sites that did refuse printed `invalid <flag>: ` with NOTHING after the
// colon, so an empty value and a missing one read the same to an operator and only
// one of them is a typo.  This is the same class the sibling front end caught on
// `--kv-score-table` (src/core/vendor_sim.h, the `set-but-empty-value` outcome).
//
// The refusal prints what the other refusals in this file print -- the TERM that
// failed, the VALUE it carried (as '' when there is nothing to print) and the set
// it was compared against -- plus, by name, the state the empty spelling imputed,
// because that half cannot be recovered from the accepted-set list.
std::string kv_value_not_empty(std::string_view flag, const char* text,
                               std::string_view accepted, std::string_view consequence) {
    if (text != nullptr && *text != '\0') { return std::string(text); }
    throw std::invalid_argument(
        "set-but-empty-value: " + std::string(flag) + " '' -- the value is EMPTY, and an "
        "empty value is not a spelling of anything: " + std::string(flag) + " accepts " +
        std::string(accepted) + ". Read as empty, " + std::string(consequence) +
        ". Drop the flag, or give one of the accepted spellings.");
}

// How many layers a `--kv-bit-budget` RANGE spec claims for itself: 1 + the largest
// `last` a readable entry names.  This is a TOKENIZER, not a second grammar -- it
// decides nothing about tiling, ceilings or ordering, it never refuses, and an entry
// it cannot read is left to product::kv_bit_budget_parse_ranges
// (product/kv_bit_budget.h), which is the ONE definition of that grammar and whose
// message is what the operator gets.  All this answers is "which layer count would
// make this spec's own last range land inside the table", so the front door can ask
// the real parser the real question at parse time instead of after the artifact is
// already open.
//
// SOUNDNESS (why this cannot refuse anything the planner accepts): the planner
// accepts a spec only when its ranges tile [0, full_layers), which means the last
// range ends at full_layers - 1, so the claim derived here IS full_layers and
// kv_bit_budget_parse_ranges receives exactly the number the planner would have
// given it.  A spec the planner would refuse may now be refused EARLIER, which is
// the point of the change; a spec the planner accepts cannot be refused here.
std::int32_t kv_budget_ranges_claimed_layers(std::string_view spec) {
    std::int32_t claimed = 0;
    std::size_t cursor = 0;
    while (cursor <= spec.size()) {
        const std::size_t comma = spec.find(',', cursor);
        const std::string_view item =
            comma == std::string_view::npos ? spec.substr(cursor)
                                            : spec.substr(cursor, comma - cursor);
        const std::size_t colon = item.find(':');
        if (colon != std::string_view::npos) {
            const std::string_view layers = item.substr(0, colon);
            const std::size_t dash = layers.find('-');
            const std::string_view last_text =
                dash == std::string_view::npos ? layers : layers.substr(dash + 1);
            const std::string text(last_text);
            if (!text.empty()) {
                errno             = 0;
                char* end         = nullptr;
                const long parsed = std::strtol(text.c_str(), &end, 10);
                if (errno == 0 && end == text.c_str() + text.size() && parsed >= 0 &&
                    parsed < std::numeric_limits<std::int32_t>::max()) {
                    const std::int32_t candidate = static_cast<std::int32_t>(parsed) + 1;
                    if (candidate > claimed) { claimed = candidate; }
                }
            }
        }
        if (comma == std::string_view::npos) { break; }
        cursor = comma + 1;
    }
    return claimed;
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
           "       [--max-context N] [--kv-capacity N|auto] [--max-concurrency N] [--prefill-chunk N] [--max-new N]\n"
           "       [--prefill-chunk-mode dynamic|manual]\n"
           "           (who owns the prefill unit. dynamic (default) lets the bandwidth governor\n"
           "            install a unit inside [128, --prefill-chunk] and shrink it while decode\n"
           "            latency sits above its measured noise floor; manual pins the unit to\n"
           "            --prefill-chunk itself for the whole run. NINFER_FT_BW_GOV=0 is the\n"
           "            environment spelling of manual, =1 of dynamic; the flag wins. Either way\n"
           "            --prefill-chunk is the ceiling, and NINFER_FT_BW_TRACE=1 prints the mode\n"
           "            and the unit the engine installed.)\n"
           "       [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|iso4e|iso3|rk4v4|e8] [--kv-layer-storage SPEC] [--kv-bit-budget SPEC] [--spec auto|off|mtp|dflash|dflash2|dspark|none --draft-tokens N]\n           (dspark is a spelling of dflash, not a fourth backend: the DSpark drafter IS\n            the DFlash (v1) runtime, and the artifact's own weights identity\n            (weights_id=nvfp4-dspark) is what decides whether its Markov head runs --\n            dflash/markov_w1+w2 are bound only for that identity, and without them the\n            same --spec dflash drafts by plain argmax.)\n           (--kv-dtype names ONE global KV tier: bf16, int8, fp8 (row-scaled E4M3 D256),\n            nvfp4, and the pair this engine prints as iso4e-g16 / rk4v4-g64 -- iso3 and e8\n            are their deprecated aliases, accepted for one release. nvfp4 is the WEIGHT tier\n            the shipped qwen3_8_27b_nvfp4_modelopt artifact records for itself (weights_id:\n            nvfp4-modelopt), i.e. the tier this project ships. It cannot be combined\n            with --kv-bit-budget/--kv-bits, whose ceiling would replace the table it fills.\n           (--kv-bit-budget takes a ceiling per KV element, or per layer range:\n            \"0-7:8,8-15:4.5\" -- the ranges must TILE every\n            FULL-ATTENTION layer (16 here, NOT the 64 the old example assumed: this\n            variant has 48 GDN layers, and they carry no paged KV); it never exceeds the\n            declared ceilings)\n"
           "       [--stage-layers SPEC] [--stage-handoff DIR] [--stage-handoff-cut]\n"
           "           (THE PIPELINE STAGE PARTITION of the text-layer axis, and the one\n"
           "            surface that makes a pp world askable from a command line instead of\n"
           "            only from a farm probe. SPEC is lo-hi layer ranges, one per STAGE:\n"
           "            0-17,18-35 is a pp world of 2 over 36 layers, and 0-35 is the\n"
           "            IDENTITY -- one stage, i.e. axis none, byte-for-byte a run with no\n"
           "            flag at all. lo-hi is inclusive hi, the same spelling --kv-layer-storage\n"
           "            documents as 0-7:bf16. The grammar, the cover and the axis check live\n"
           "            in ONE place, core/stage_plan.h, called by this front door AND by the\n"
           "            runtime -- a second spelling of the grammar is what this project keeps\n"
           "            paying for.)\n"
           "           (REFUSED BY NAME, never accepted and ignored: refused-stage-layers for a\n"
           "            mis-shaped SPEC; refused-stage-layers-partition for a spec that is not a\n"
           "            cover of [0, layers); refused-stage-layers-axis for a spec the rank axis\n"
           "            does not derive -- plan_shards() itself decides, so a layer range that\n"
           "            is not a shard of the world core/shard_plan.h hands out is refused\n"
           "            rather than run and reported as something it is not. NOTE what is NOT\n"
           "            touched: --stage-layers makes pp REACHABLE, it does not make\n"
           "            validate_virtual_request(pp).active read anything but 0. The virtual\n"
           "            -device guard still refuses pp, with its own reason, unchanged.)\n"
           "           (A PARTIAL range is additionally refused by name where the range cannot\n"
           "            be honoured: refused-stage-layers-spec with speculation on (the drafters\n"
           "            walk the layer axis themselves -- dflash_impl.h:143/:236,\n"
           "            dflash2_impl.h:190/:248, mtp_impl.h:264 -- and this range does not bound\n"
           "            them); refused-stage-layers-w13 with the W13 weight host-offload budget\n"
           "            set (product/weight_residency.h:374-383 asserts that every pass enters\n"
           "            every offloaded layer through note_layer(), and a partial pass does not);\n"
           "            refused-stage-layers-graph with CUDA-graph capture on (the seam is a\n"
           "            HOST-side write/read, which a captured graph would replay stale -- the\n"
           "            same reason W13's H2D is prefill-only).)\n"
           "           (--stage-handoff DIR names the directory the boundary hidden state\n"
           "            crosses through: stage k writes stage_k.bin, reads stage_{k-1}.bin, as\n"
           "            raw bytes plus a header carrying a magic, the producing layer, the\n"
           "            element count and an FNV-1a, so a payload left over from another prompt\n"
           "            is DETECTABLE rather than silently consumed. --stage-handoff-cut is a\n"
           "            NEGATIVE CONTROL, not a feature: it silences the producer so the ids\n"
           "            MOVE, which is how \"the handoff is load-bearing\" is falsified on the\n"
           "            shipped binary instead of asserted.)\n"
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
           "            is built through, so it enters the row-scale fingerprint as the\n"
           "            product/kv_rowscale_persist.h rope_regime knob: mixed in as ;rope=1 when\n"
           "            not default, the tag itself staying rs1.<12 hex> (NINFERKVRS1 is the\n"
           "            sidecar magic, not the tag). A table baked by a run that did not name\n"
           "            --yarn does not validate for a run that does. Off by default.)\n"
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
           "       [--print-prompt-ids] [--print-token-ids] [--no-thinking] [--thinking-budget N]\n"
           "       [--reasoning-effort low|medium|xhigh] [--vision]\n"
           "       [--cold-policy none|off|window|host|disk|host-then-disk|host+disk] "
           "(host+disk is an accepted equivalent spelling of host-then-disk; "
           "docs/cli.md)\n"
           "[--cold-keep-tokens N]\n"
           "       [--max-cold-pages N] [--kv-unload-watermark-pages N]\n"
           "       [--recall-prefill-tokens N]\n"
           "       [--append-context-text <text>]\n"
           "       [--cold-host-bytes N[g|m|k]]\n"
           "       [--cold-disk-path DIR] [--cold-disk-bytes N]\n"
           "       [--ple-sidecar DIR]\n"
           "       [--weight-host-bytes N] [--weight-device-arena-bytes N]\n"
           "       [--weight-prefetch-layers N] [--weight-span-floor-bytes N]\n"
           "       [--no-cuda-graph] [--graph-capture-ceiling N]\n"
           "       [--ft-stats on|off] [--inject-spec PATH]\n"
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
           "--inject-spec PATH declares a CHOSEN TENSOR and the position range it will "
           "occupy (src/spec/inject_channel.h): direction=ingest|egress, dtype=bf16|f16|f32, "
           "layout=token-major, rows, cols, position0, scale, path, digest. The tensor's "
           "alphabet is the input-embedding space, so it enters the model where a gathered "
           "token's embedding enters, and an ingress of the engine's own bytes is a no-op "
           "to the model (the etiquette requirement of src/spec/sum_dir.h:98, made into a "
           "measurement). The declaration is validated here, with the header's own parser, "
           "so a wrong field or a wrong dtype refuses by name before the artifact is loaded; "
           "the checks that need the model (rows, position range) are made at bind time. "
           "Every admitted ingest is reported on stderr with its shape, dtype, position "
           "range and ingested digest, and every mismatch with its refusal name. "
           "NINFER_INJECT_SPEC is the env spelling and the flag beats it.\n"
           "--print-prompt-ids prints, on stderr, the ids the PROMPT was tokenized to, in order. "
           "It is the input side of --print-token-ids, and it exists because a `sum_dir` row's "
           "identity is a digest over its block's token ids (src/spec/sum_dir.h:210-224), so a row "
           "cannot be NAMED -- and therefore cannot be bound to the inject channel "
           "(src/spec/sum_dir_inject.h) -- without them. Read-only: the engine already holds the "
           "sequence (include/ninfer/engine.h:30) and this flag is the surface it never had. "
           "--ft-stats on enables the FreeToken per-layer attention-energy observation "
           "(NINFER_FT_STATS=1 is the env spelling); it is off by default and the flag "
           "beats the env. Its consumer -- the periodic KV relayout -- is a serve-side "
           "feature (--kv-auto-relayout there), because this front end has no decision "
           "cycle.\n"
           "--recall-prefill-tokens N bounds the RE-PREFILL a recall round may do, in TOKENS "
           "(NINFER_RECALL_PREFILL_TOKENS is the env spelling and the flag beats it; no default, "
           "0 = off). It is a different dimension from the 256 MiB byte budget, which the tree "
           "itself calls a MEMORY-SAFETY limit rather than a speed limit. At the edge the round is "
           "REFUSED, by name: stderr carries `refused-prefill-budget` with the tokens wanted and "
           "the budget, at plan time and before a single token is re-prefilled. It is never "
           "truncated to fit -- a truncated run is a prefix of the answer's context, i.e. a "
           "partial or a confidently wrong answer. `--recall-prefill-tokens 0` is refused by "
           "name: 0 is the default, so it would be accepted and read by nothing.\n"
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
    // The same "the operator named it" gate for --weight-prefetch-layers, and the ONLY one of the
    // four W13 knobs that needs a gate rather than a value test: its default is 2 -- a legal depth
    // -- so `weight_prefetch_layers != 2` would let `--weight-prefetch-layers 2` alone through and
    // would hard-code the default here. src/serve/serve_options.cpp carries the same variable.
    bool weight_prefetch_layers_explicit = false;

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
        } else if (arg == "--max-concurrency") {
            // allow_zero on purpose: the bound in the validation block below is what refuses
            // 0, so the refusal names the BOUND ("--max-concurrency must be in [1,16]") exactly
            // as ninfer-serve's does. A plain parse_u32 would refuse 0 first with a message
            // that names the parse instead, and the two front ends would then answer the same
            // argv differently. Precedent for the spelling: --draft-tokens.
            options.max_concurrency =
                parse_u32(value(arg), "max-concurrency", /*allow_zero=*/true);
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
            options.kv_cache = parse_kv_cache(kv_value_not_empty(
                arg, value(arg), "bf16|int8|fp8|nvfp4|iso4e|iso3|rk4v4|e8",
                "the refusal printed the flag and then nothing where the value belongs, "
                "so an empty value and a missing one read the same"));
            options.kv_cache_explicit = true;
        } else if (arg == "--kv-bit-budget") {
            // "N" (one ceiling for every full-attention layer) or "lo-hi:bits,..."
            // (separable per-range ceilings; the DP minimises each range independently).
            const std::string budget_spec = kv_value_not_empty(
                arg, value(arg), "one ceiling like 4.5, or ranges like 0-7:8,8-15:4.5 (ranges tile every full-attention layer)",
                "the refusal printed the flag and then nothing where the value belongs, "
                "so an empty value was indistinguishable from a missing one");
            if (budget_spec.find(':') != std::string::npos ||
                budget_spec.find(',') != std::string::npos) {
                // RANGES: the tiling is now checked HERE, by the allocator's own parser,
                // with the layer count the spec claims for itself
                // (kv_budget_ranges_claimed_layers above). It used to be checked only in
                // the planner (targets/qwen3_6/impl/runtime/layouts_impl.h), i.e. after
                // the artifact was already open, so every malformed range spec reached
                // the engine first: `0-7:` (ceiling missing), `99:4` (layer out of
                // range), `8,4` (no lo-hi:bits), `0-7:0` and `0-7:-1` (non-positive
                // ceiling), `4.5,` (empty entry), `0-7:8,4-5:4.5` (out of order) all
                // parsed clean at this front door.
                //
                // Nothing is duplicated: this call IS product::kv_bit_budget_parse_ranges,
                // and the planner's own call with the MODEL's layer count is unchanged --
                // so a spec that tiles [0,N) for a claim N that is not this model's
                // full-attention count is still refused there, where the count lives.
                try {
                    (void)product::kv_bit_budget_parse_ranges(
                        budget_spec, kv_budget_ranges_claimed_layers(budget_spec));
                } catch (const std::invalid_argument& error) {
                    // The parser's verdict is kept VERBATIM (it is the one definition of
                    // this grammar); only the context it cannot know is added. Several of
                    // its refusals name the FIELD and not the text that landed in it
                    // ("invalid budget: " with the piece elided), so the value the
                    // operator typed is printed in full here.
                    throw std::invalid_argument(std::string(error.what()) +
                                                " [--kv-bit-budget '" + budget_spec + "']");
                }
                options.kv_bit_budget_ranges   = budget_spec;
                options.kv_bit_budget_bits     = 0.0;
                options.kv_bit_budget_explicit = true;
            } else {
                if (budget_spec.find('-') != std::string::npos) {
                    // A RANGE WITHOUT ITS CEILING. `0-7` is not a ceiling and is not a
                    // range either, and the generic number refusal ("invalid
                    // --kv-bit-budget: 0-7", because strtod stops at the '-') leaves the
                    // operator to guess which of the two grammars was expected.
                    throw std::invalid_argument(
                        "invalid --kv-bit-budget: " + budget_spec +
                        " -- this names a LAYER RANGE, and a range needs its ceiling: the "
                        "grammar is lo-hi:bits, comma separated for several ranges (e.g. "
                        "0-7:8,8-15:4.5). A ceiling for every layer at once is a plain "
                        "number (e.g. 4.5).");
                }
                options.kv_bit_budget_bits =
                    parse_float(budget_spec.c_str(), "--kv-bit-budget", 0.01F, 16.0F);
                options.kv_bit_budget_ranges.clear();
                options.kv_bit_budget_explicit = true;
            }
        } else if (arg == "--kv-quality-weight") {
            // 0 = fastest KV path, 1 = most accurate; the DP minimises
            // w*quality + (1-w)*speed per tier inside the bit ceiling.
            options.kv_quality_weight =
                parse_float(kv_value_not_empty(
                                arg, value(arg), "a weight in [0,1] (0 = fastest)",
                                "the refusal printed the flag and then nothing where the "
                                "value belongs")
                                .c_str(),
                            "--kv-quality-weight", 0.0F, 1.0F);
        } else if (arg == "--kv-tier-scores") {
            options.kv_tier_scores = kv_value_not_empty(
                arg, value(arg), "a file path, or the table's own text",
                "the empty spec selects the BUILT-IN table (kv_bit_budget_default_scores), "
                "so the flag would be accepted and change nothing");
        } else if (arg == "--kv-bits") {
            // JOINT form: ONE overall ceiling for the whole KV stack ("合起来整体定").
            options.kv_joint_bits = parse_float(
                kv_value_not_empty(arg, value(arg), "a bits-per-element ceiling like 4.5",
                                   "the refusal printed the flag and then nothing where the "
                                   "value belongs")
                    .c_str(),
                "--kv-bits", 0.01F, 16.0F);
            options.kv_kv_bits_explicit = true;
        } else if (arg == "--kv-k-bits") {
            // SPLIT form: K gets its own ceiling, and its own per-layer layering.
            options.kv_k_bits = parse_float(
                kv_value_not_empty(arg, value(arg), "a bits-per-element ceiling like 4.5",
                                   "the refusal printed the flag and then nothing where the "
                                   "value belongs")
                    .c_str(),
                "--kv-k-bits", 0.01F, 16.0F);
            options.kv_kv_bits_explicit = true;
        } else if (arg == "--kv-v-bits") {
            // SPLIT form: V gets its own ceiling, and its own per-layer layering.
            options.kv_v_bits = parse_float(
                kv_value_not_empty(arg, value(arg), "a bits-per-element ceiling like 4.5",
                                   "the refusal printed the flag and then nothing where the "
                                   "value belongs")
                    .c_str(),
                "--kv-v-bits", 0.01F, 16.0F);
            options.kv_kv_bits_explicit = true;
        } else if (arg == "--kv-bits-mode") {
            // Which reading of a per-plane request runs: joint | split | ceiling.
            options.kv_bits_mode = product::kv_bits_mode_from_name(kv_value_not_empty(
                arg, value(arg), "joint|split|ceiling",
                "the refusal printed the flag and then nothing where the value belongs"));
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
            options.kv_k_tier_scores = kv_value_not_empty(
                arg, value(arg), "a file path, or the table's own text",
                "the empty spec selects the BUILT-IN table (kv_bit_budget_default_scores), "
                "so the flag would be accepted and change nothing");
        } else if (arg == "--kv-v-tier-scores") {
            // The V plane's own score columns for the split entry.
            options.kv_v_tier_scores = kv_value_not_empty(
                arg, value(arg), "a file path, or the table's own text",
                "the empty spec selects the BUILT-IN table (kv_bit_budget_default_scores), "
                "so the flag would be accepted and change nothing");
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
        } else if (arg == "--stage-layers") {
            // THE FLAG THIS LINE ADDS. The value is kept RAW here and parsed by
            // core/stage_plan.h -- the same parser the runtime calls -- so the grammar has one
            // implementation. The grammar IS this front door's promise, so it is checked HERE,
            // before any artifact is opened, exactly as --kv-layer-storage's entries are
            // (options.cpp:686-693).
            const std::string spec = kv_value_not_empty(
                arg, value(arg), "lo-hi layer ranges, one per stage, like 0-17,18-35",
                "an empty SPEC is refused rather than read as \"no stages\", because a run that "
                "quietly became single-device is indistinguishable from a successful world at "
                "the point where the numbers are read");
            ninfer::multi::StagePlan parsed;
            const std::string refusal = ninfer::multi::parse_stage_layers(spec, parsed);
            if (!refusal.empty()) { throw std::invalid_argument(refusal); }
            options.stage_layers_spec     = spec;
            options.stage_layers_explicit = true;
        } else if (arg == "--stage-handoff") {
            const std::string dir = kv_value_not_empty(
                arg, value(arg), "a directory the stage boundary payload crosses through",
                "an empty directory would put stage_k.bin in the process's cwd, where a "
                "leftover file from an unrelated run is consumed as this run's boundary");
            options.stage_handoff_dir     = dir;
            options.stage_handoff_explicit = true;
        } else if (arg == "--stage-handoff-cut") {
            options.stage_handoff_cut      = true;
            options.stage_handoff_explicit = true;
        } else if (arg == "--kv-layer-storage") {
            // The table AND its mask are parsed in main() (the mask is what makes
            // `0-11:bf16` a real per-layer baseline rather than "inherit --kv-dtype"),
            // but the grammar is this front door's promise, so an empty value and an
            // empty entry are refused here instead of being handed over.
            const std::string storage_spec = kv_value_not_empty(
                arg, value(arg), "lo-hi:dtype entries like 0-7:bf16, or all:dtype",
                "product::parse_kv_layer_storage_spec(\"\") returns a table whose mask is "
                "all-false, so the flag would be marked explicit and read by nothing");
            if (storage_spec.back() == ',') {
                // A TRAILING COMMA. parse_kv_layer_storage_spec's loop runs
                // `while (begin < spec.size())`, so a spec that ends on a comma never
                // reaches the `if (item.empty()) throw` inside it: `0-7:bf16,` was
                // accepted, and the empty entry it names was dropped without a word.
                // Refused here, by name, with the entry printed; the parser's own loop
                // bound is the thing that would make this check unnecessary.
                throw std::invalid_argument(
                    "kv-layer-storage: '" + storage_spec +
                    "' ends on a comma, so its last entry is EMPTY. The grammar is a "
                    "comma-separated list with no empty element (e.g. 0-7:bf16,8-15:int8 "
                    "or all:bf16), and an empty final entry is not a spelling of "
                    "anything -- it was dropped in silence before this check.");
            }
            options.kv_layer_storage_spec     = storage_spec;
            options.kv_layer_storage_explicit = true;
        } else if (arg == "--kv-residual-layers") {
            // Stored raw and parsed in main() next to the EngineOptions it produces,
            // which is also where --kv-layer-storage is turned into a table. The parse
            // still precedes Engine construction, so a typo is refused before any device
            // work (rule: behavioural differences that happen at parse time are settled
            // on the host, never with the GPU in the loop).
            options.kv_residual_layers_spec = kv_value_not_empty(
                arg, value(arg), "a layer list like 2-5 or 0,3,7",
                "product::parse_kv_residual_layers_spec(\"\") returns an all-false table "
                "while kv_residual_explicit is set to true, i.e. exactly the \"explicitly "
                "no residuals\" state apps/cli/main.cpp warns an all-false table must not "
                "be handed over unconditionally");
            options.kv_residual_layers_explicit = true;
        } else if (arg == "--kv-tier-formats") {
            // KV tier vocabulary ("hot=bf16,tail=fp16,cold=iso3"; kvcfg/kv_formats.h).
            // Parsed raw: the vocabulary's own rules are checked after the loop (the
            // nvfp4 mode may come later in argv) and the per-layer landing needs the
            // model's layer count, so it happens in the planner.
            options.kv_tier_formats_spec = kv_value_not_empty(
                arg, value(arg), "tier=format pairs like hot=bf16,cold=int8",
                "the vocabulary's own parser accepts the empty text and lands nothing, so "
                "the flag would be marked explicit and read by nothing");
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
            const std::string_view mode = kv_value_not_empty(
                arg, value(arg), "on|off",
                "the refusal printed the flag, then nothing where the value belongs, then "
                "the accepted set");
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
            options.kv_row_scale_spec = kv_value_not_empty(
                arg, value(arg), "auto|off|FILE",
                "kv_rowscale_mode_from_spec(\"\") returns Auto, so the flag would be "
                "accepted and behave exactly as --kv-row-scale auto, with no line saying "
                "the value never arrived");
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
            const std::string_view mode = kv_value_not_empty(
                arg, value(arg), "iso4e|e2m1",
                "the refusal printed the flag, then nothing where the value belongs, then "
                "the accepted set");
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
        } else if (arg == "--print-prompt-ids") {
            options.print_prompt_ids = true;
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
        } else if (arg == "--recall-prefill-tokens") {
            // [PREFILLBUDGET] the token-denominated prefill budget. Positive only: 0 is the
            // default and means OFF, so a named 0 would be accepted and read by nothing. The
            // budget itself is applied at ONE site in the engine (plan_request.token_budget,
            // program_impl.h) and its edge is a NAMED REFUSAL, never a clamp.
            options.recall_prefill_tokens = parse_u32(value(arg), "recall-prefill-tokens");
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
        } else if (arg == "--ple-sidecar") {
            // The FlashNext PLE n-gram sidecar root. Parsed exactly as
            // src/serve/serve_options.cpp parses it, and for the same reason
            // --kv-unload-watermark-pages gives above: both front ends must name
            // the same failures the same way instead of drifting into two
            // vocabularies for one knob. The value is NOT stat'd here; it is
            // checked at startup by product::validate_ple_sidecar_root(), which is
            // where a refusal can still stop the run before the engine is built.
            options.ple_sidecar_root = value(arg);
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
            weight_prefetch_layers_explicit = true;
            if (options.weight_prefetch_layers < 2) {
                throw std::invalid_argument(
                    "--weight-prefetch-layers below 2 would let the arena slot of the layer "
                    "being computed be overwritten by its own prefetch");
            }
        } else if (arg == "--weight-span-floor-bytes") {
            options.weight_span_floor_bytes = parse_u64(value(arg), "weight-span-floor-bytes");
        } else if (arg == "--graph-capture-ceiling") {
            options.graph_capture_ceiling = parse_u32(value(arg), "graph-capture-ceiling");
        } else if (arg == "--inject-spec") {
            options.inject_spec = value(arg);
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
            // `<flag>=<value>` is not a spelling this front end accepts ANYWHERE: every
            // flag here takes its value as the NEXT argv entry, so the whole token is an
            // unknown argument. The generic refusal printed the token and stopped there,
            // which for the empty-value spelling (`--kv-bit-budget=`) printed the flag
            // and left the VALUE -- the part that is empty, and the whole reason the
            // operator needs to be told -- invisible ("invalid X: " with nothing after
            // the colon, the same half-answer this change removes elsewhere). Both halves
            // are named, and the two-argument spelling is named as the way out.
            const std::size_t equals = arg.find('=');
            if (equals != std::string_view::npos) {
                throw std::invalid_argument(
                    "unknown argument: " + std::string(arg) +
                    " -- this front end takes a flag and its value as TWO arguments, so '" +
                    std::string(arg.substr(0, equals)) + "' carrying the value '" +
                    std::string(arg.substr(equals + 1)) +
                    "' is not a spelling of anything here. An empty value after the '=' is "
                    "not one either: an empty value names no state, and the parse sites "
                    "that do meet one impute a state instead of reporting it.");
            }
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    // ...and a run that consumed flags without ever naming a model says so by name.
    // --kv-score-table is its own entry point and needs no artifact, which is why it
    // is the one exception.
    // ---------------------------------------------------------------------------------
    // --stage-layers: the cross-flag checks this front door owes, refused BY NAME.
    // ---------------------------------------------------------------------------------
    // The SHAPE checks (grammar, cover) are done where the spec is read, above. What is left
    // is what only a whole command line can decide. The AXIS check -- is this a world the rank
    // axis derives -- cannot be done here at all: it needs the artifact's own layer count, so
    // it runs in the runtime, where plan_shards() has a geometry to be asked about. That split
    // is stated in the help text so a reader is not left thinking the front door validated
    // more than it did.
    if (options.stage_handoff_explicit && !options.stage_layers_explicit) {
        throw std::invalid_argument(
            std::string(ninfer::multi::kStageLayersHandoffRefusal) + ": " +
            std::string(ninfer::multi::kStageHandoffFlag) + " was given with no " +
            std::string(ninfer::multi::kStageLayersFlag) +
            ", so there is no stage boundary for it to carry. A directory written and never "
            "read is a flag accepted and ignored.");
    }
    if (options.stage_layers_explicit) {
        ninfer::multi::StagePlan staged;
        const std::string parse_refusal =
            ninfer::multi::parse_stage_layers(options.stage_layers_spec, staged);
        if (!parse_refusal.empty()) { throw std::invalid_argument(parse_refusal); }
        // The cover, with text_layers == 0: this front door has no artifact and must not guess
        // one, so only the shape rules that need no layer count are applied here. The cover
        // against the real layer count is the runtime's.
        if (const std::string refusal =
                ninfer::multi::stage_layers_partition_refusal(staged, 0U);
            !refusal.empty()) {
            throw std::invalid_argument(refusal);
        }
        // The three run shapes a PARTIAL range cannot carry. Each names its own file:line.
        if (const std::string refusal = ninfer::multi::stage_layers_run_shape_refusal(
                staged,
                options.speculative.backend != SpeculativeBackend::None,
                options.weight_host_offload_bytes != 0 || options.weight_device_arena_bytes != 0,
                options.use_cuda_graph);
            !refusal.empty()) {
            throw std::invalid_argument(refusal);
        }
        // And the boundary: a partial range needs one (stated), and the identity must not be
        // given one (a directory written and never read).
        if (const std::string refusal =
                ninfer::multi::stage_handoff_refusal(staged, options.stage_handoff_dir);
            !refusal.empty()) {
            throw std::invalid_argument(refusal);
        }
    }

    // ---------------------------------------------------------------------------------
    // W13 weight offload: the cross-flag contradictions this front door owes.
    // ---------------------------------------------------------------------------------
    // COPIED FROM src/serve/serve_options.cpp:971-981 -- the ONE statement of this rule -- because
    // THIS front door stored both numbers, printed both in its own usage text, passed both into
    // EngineOptions, and read NEITHER. The plan builder's FIRST statement returns an empty plan
    // when the pinned host mirror is zero
    //     if (limits.host_pinned_bytes == 0) { return plan; }
    // (src/product/weight_residency.h:447, twin src/artifact/binder.cpp:188), and the reader of
    // --weight-device-arena-bytes sits BELOW that return (weight_residency.h:554). So the flag was
    // ACCEPTED AND IGNORED, which is this project's own worst outcome, and it was measurable from
    // the outside: passing it alone changed nothing but the usage dump of a 16,936 B stderr.
    // The repair is therefore not a new rule but the rule the sibling front door already makes, in
    // its own words, so the two front ends cannot drift into two vocabularies for one knob.
    // THE RESIDUAL THIS COMMENT USED TO NAME IS CLOSED IN THE SAME BLOCK, and the reasoning that
    // made it hard is KEPT because it is exactly why the instrument is a BIT and not a value test:
    // --weight-prefetch-layers is dropped by the SAME return, its default is 2 -- a legal value --
    // so `prefetch_layers != 2` could not tell "unset" from "explicitly 2" and would hard-code the
    // default in three front ends. Each front end now sets an explicit bit in that flag's OWN parse
    // branch, and the third refusal below covers it in ONE set of words in all three front doors,
    // so the vocabularies cannot diverge.
    if (options.weight_host_offload_bytes == 0 &&
        (options.weight_device_arena_bytes != 0 || options.weight_span_floor_bytes != 0)) {
        throw std::invalid_argument("--weight-device-arena-bytes / --weight-span-floor-bytes "
                                    "need a positive --weight-host-bytes");
    }

    // THE THIRD KNOB OF THE SAME FAMILY, and the reason it is not covered by the check above: the
    // pair's two knobs signal by a NON-ZERO value, but this one's DEFAULT IS A LEGAL VALUE (2), so
    // no value test can tell "unset" from "explicitly 2" and the gate has to be a BIT that flag's
    // own parse branch sets. The plan builder returns an empty plan at its FIRST statement when the
    // pinned host mirror is zero
    //     if (limits.host_pinned_bytes == 0) { return plan; }
    // (src/product/weight_residency.h:447, twin src/artifact/binder.cpp:188), and BOTH readers of
    // the depth sit BELOW that return: the domain check at :448 and the only derive at :563, inside
    // the `else` of :554. So without the partner the flag is ACCEPTED AND IGNORED -- and a flag that
    // is accepted and ignored is this project's own worst outcome, so it is refused here with the
    // reason named rather than left to be discovered.
    if (options.weight_host_offload_bytes == 0 && weight_prefetch_layers_explicit) {
        throw std::invalid_argument("--weight-prefetch-layers needs a positive --weight-host-bytes: "
                                    "without the host mirror the offload plan returns before the "
                                    "depth is read, so the flag would be accepted and ignored");
    }
    if (options.weight_host_offload_bytes != 0 &&
        options.weight_device_arena_bytes >= options.weight_host_offload_bytes) {
        throw std::invalid_argument(
            "--weight-device-arena-bytes is not smaller than --weight-host-bytes, so the "
            "offload would free no device memory");
    }
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
    // kv4 P3 REFUSE -- REMOVED BY F903, because the thing it refused now COMPOSES.
    // WHAT CHANGED IS THE CONSTRUCTION AND NOT THE SPELLING. The old rule was right about
    // the old engine: `--kv-layer-storage` filled the per-layer table and layouts_impl.h
    // then skipped the whole ceiling ("if (options.kv_bit_budget_explicit && !storage_explicit)"),
    // so the ceiling was accepted and read by nothing -- this project's worst outcome, and
    // refusing was the correct thing to do about it. The engine now has a PIN SET: the mask
    // of slots the operator actually WROTE is frozen before any resolution, the ceiling runs
    // over the WHOLE stack, and the pins are applied OVER its plan. Two providers of one
    // table became two ROLES on one table -- a CONSTRAINT and an OBJECTIVE -- which is why
    // the pair no longer needs a door. The one case that still cannot compose (a table naming
    // EVERY layer, where the ceiling really would be read by nothing) is refused BY NAME in
    // layouts_impl.h, where the layer count exists, rather than here, where it does not.
    //
    // A NOTE ON THE TWIN, so it is not mistaken for fixed: src/serve/serve_options.cpp:870
    // still carries this refusal verbatim. The engine side under it composes in this build;
    // the serve front door does not yet say so. Named, not silent.
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
    // F903: the same pair, the other ceiling spelling, removed for the same reason and with
    // the same residual (a table naming EVERY layer is refused by name in layouts_impl.h).
    // The K/V entry composes through the SAME pin application, so the two ceiling spellings
    // cannot disagree about what a pin means.
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
    // SLIDERWIRE: --kv-codec-preference. What SURVIVES here is the side that is still true:
    // the preference is read ONLY inside a bit-budget fit, so naming it with NO ceiling at all
    // is still a silent no-op and is still refused, with the flags that make it act named.
    // WHAT F903 REMOVED IS THE OTHER TWO SIDES, because both had gone STALE against the tree:
    //   * "only the --kv-bits resolution carries a candidate order" -- FALSE. The
    //     --kv-bit-budget entry resolves through product::kv_bit_budget_solve_scored, whose
    //     LAST PARAMETER is `const std::vector<std::int32_t>& candidate_order = {}`
    //     (product/kv_bit_budget.h:1263-1270). layouts_impl.h simply never passed it. It does
    //     now, so the preference acts on the --kv-bit-budget spelling too.
    //   * "and --kv-layer-storage IS the per-layer table" -- FALSE since the pin set. The
    //     table names a SUBSET; the preference acts on the layers the table does NOT name,
    //     and the pinned slots keep the codec the operator wrote. Both are read.
    // TWO RESIDUALS, NAMED RATHER THAN HIDDEN: the RANGE spelling of --kv-bit-budget
    // (kv_bit_budget_scored_ranges takes no candidate_order) and src/serve/serve_options.cpp
    // both still refuse; the first is refused by name in layouts_impl.h, the second is named
    // there in the twin note. A refusal that names the missing plumbing is not a composition,
    // and this comment is the reading, not the verdict.
    if (options.kv_codec_preference_explicit && !options.kv_kv_bits_explicit &&
        !options.kv_bit_budget_explicit) {
        throw std::invalid_argument(
            "--kv-codec-preference needs a ceiling to act on: it names which codec the solver "
            "picks among candidates that cost the SAME bits, and that choice is made inside a "
            "bit-budget fit. Without a ceiling nothing reads it and it would be accepted and "
            "read by nothing. Add --kv-bits (e.g. --kv-bits 4.5 --kv-quality-weight 0 "
            "--kv-codec-preference iso4e) or --kv-bit-budget, or drop the flag.");
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

    // F738 injectchan: the ingress declaration. Validated HERE with the header's own
    // parser, so a mis-shaped or mis-dtyped declaration refuses by name before a
    // multi-gigabyte artifact is loaded; the model-dependent half of the admission (rows
    // against the model's hidden, the position range against the context capacity) is
    // made at bind(), where those numbers first exist. Committed to the variable the
    // engine reads at parse time, which is the --ft-stats precedent and the reason the
    // flag beats the environment.
    if (!options.inject_spec.empty()) {
        const spec::inject::ParseResult declaration =
            spec::inject::parse_spec_file(options.inject_spec);
        if (!declaration.ok()) {
            throw std::invalid_argument(
                std::string("--inject-spec: ") +
                spec::inject::refusal_name(declaration.settlement.refusal) +
                (declaration.settlement.field.empty()
                     ? std::string{}
                     : " (field " + declaration.settlement.field + ")") +
                " in " + options.inject_spec + " : " + declaration.settlement.detail);
        }
        // The PAYLOAD's own checks, at the same parse time and for the same reason: the file, its
        // declared size, the caller's pinned digest and the finiteness of every element all need
        // no model. bake() is the header's own function -- the engine calls it again at bind()
        // where the model-dependent half of the admission is the only thing left to decide, so
        // this is a second call and not a second rule.
        const spec::inject::PayloadSettlement payload = spec::inject::bake(declaration.declaration);
        if (!payload.settled()) {
            throw std::invalid_argument(
                std::string("--inject-spec: ") + spec::inject::refusal_name(payload.refusal) +
                (payload.field.empty() ? std::string{}
                                       : " (field " + payload.field + ")") +
                " in " + options.inject_spec + " : " + payload.detail);
        }
        if (set_process_env("NINFER_INJECT_SPEC", options.inject_spec.c_str()) != 0) {
            throw std::invalid_argument("cannot set NINFER_INJECT_SPEC for --inject-spec");
        }
    }

    // FreeToken: commit an explicit --ft-stats to the variable the observation
    // reads (see set_process_env above). An unset flag leaves the operator's
    // NINFER_FT_STATS alone, which is what keeps "env only" working.
    if (options.ft_stats.has_value() &&
        set_process_env("NINFER_FT_STATS", *options.ft_stats ? "1" : "0") != 0) {
        throw std::invalid_argument("cannot set NINFER_FT_STATS for --ft-stats");
    }

    // [PREFILLBUDGET] The same shape for the prefill budget, and for the same reason:
    // `program_impl.h` reads `NINFER_RECALL_PREFILL_TOKENS` beside `NINFER_TURN_RECALL_BYTES`,
    // both at the one site that nests inside the recall block, so the flag is committed to that
    // variable here and the flag therefore BEATS the environment. An un-named flag leaves the
    // operator's environment value alone, which is what keeps "env only" working.
    if (options.recall_prefill_tokens != 0 &&
        set_process_env("NINFER_RECALL_PREFILL_TOKENS",
                        std::to_string(options.recall_prefill_tokens).c_str()) != 0) {
        throw std::invalid_argument(
            "cannot set NINFER_RECALL_PREFILL_TOKENS for --recall-prefill-tokens");
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
    // The engine's own bound, spelled as ninfer-serve spells it (serve_options.cpp:831-832),
    // so the same argv is refused the same way by both front ends. Checked HERE rather than
    // left to engine.cpp normalize_engine_options, because that function clamps
    // max_concurrency to 1 inside its CausalScoring branch (engine.cpp:52) BEFORE testing the
    // bound -- so an out-of-range request on --score would be silently accepted as 1 there.
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,16]");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens == 0) {
        throw std::invalid_argument("--kv-capacity must be positive");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.speculative.backend == SpeculativeBackend::DFlash && options.enable_vision) {
        // landq/unlock -- THE REASON CATEGORY, NAMED RATHER THAN IMPLIED.
        //
        // The sentence below used to read "cannot be combined with --vision", which is a claim
        // of impossibility. The reading does not support one, and it does not support the
        // opposite either -- it supports a THIRD answer, "not validated":
        //   * NOT an artifact fact (byte caliber, the containers' own objects[]): 9 of the 91
        //     readable .ninfer artifacts on this box declare the four dflash2/* entry objects
        //     AND all five vision/* entry objects, so one artifact carries the draft head and
        //     the vision tower at once. Three more carry vision + the DFlash (v1) head.
        //   * NOT a designed pairing: the upstream family ships its Vision companion as a
        //     separate `vision-mtp-bf16` head, and its acceptance-rate plan spells the backend
        //     as `--spec dflash2 --draft-tokens K` with no --vision.
        //   * MISSING: a validated pair. No run of the two together is on record in this tree,
        //     and the Program refuses the two view sets at program_impl.h:1024 for that same
        //     reason.
        // The refusal STAYS. Only its reason category changes: history, not physics.
        throw std::invalid_argument(
            "--spec dflash with --vision is not co-validated: this front end refuses an "
            "unvalidated pair rather than running it. This is a policy refusal, not an "
            "artifact limit -- one artifact does carry the draft head and the vision tower at "
            "once, and what does not exist is a validation of the two together. Drop one.");
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
