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

    // --capability-report: the BUILD capability surface's own entry point (src/core/arch_caps.h,
    // render_build_capability_surface). Model-free like --kv-score-table and for the same reason
    // -- there is no artifact it could need -- so parse_options() must not demand a model or a
    // prompt for it, and main() acts on it before any prompt or model is touched.
    bool capability_report_requested = false;

    std::filesystem::path artifact_path;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    // --max-concurrency N: how many lanes the engine runs (the "N concurrent prefills"
    // knob). The default 1 and the [1,16] bound are the engine's own
    // (include/ninfer/types.h EngineOptions::max_concurrency; engine.cpp:95-96); the flag
    // spelling, the bound and the refusal wording are ninfer-serve's
    // (src/serve/serve_options.cpp:279-281, :831-832), so the two front ends cannot
    // disagree about the same knob. This flag is the whole of this entry: the field
    // existed in EngineOptions but no CLI path copied it across, so the knob was
    // unreachable from the delivered `ninfer` binary.
    std::uint32_t max_concurrency = 1;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t prefill_chunk  = 3072;
    // --prefill-chunk-mode dynamic|manual: which of the two modes owns the prefill unit
    // (PrefillChunkMode, ninfer/types.h). Unset is NOT the same as Manual -- it defers to
    // NINFER_FT_BW_GOV and then defaults to Dynamic, resolved once in
    // engine.cpp normalize_engine_options, which is the only reader of that switch on this path.
    std::optional<PrefillChunkMode> prefill_chunk_mode;
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
    // --kv-residual-layers SPEC: the per-layer NVFP4 SECOND-STAGE RESIDUAL planes
    // (include/ninfer/types.h EngineOptions::kv_residual_layers). Kept RAW and parsed in
    // main(), next to the table it produces, with the same shared parser the serving front
    // end calls (product/kv_options.h): the parse has to name the flag in its error and has
    // to happen before any device work, and a second implementation of the grammar is the
    // exact shape this project keeps paying for.
    std::string kv_residual_layers_spec;
    // KV bit budget: a single ceiling per element, or separable per-range ceilings
    // ("0-7:8,8-15:4.5", tiling every FULL-ATTENTION layer); the DP never exceeds them and minimises the penalty inside them.
    double kv_bit_budget_bits = 0.0;
    std::string kv_bit_budget_ranges;
    bool kv_bit_budget_explicit = false;
    // Two-score KV selection: 0 = fastest, 1 = most accurate; negative leaves the shipped
    // single-penalty ladder in place. --kv-tier-scores overrides the score table.
    double kv_quality_weight = -1.0;
    std::string kv_tier_scores;
    // K/V bit widths (product/kv_kv_bits.h). TWO entry points:
    //   --kv-bits                      ONE overall ceiling ("合起来整体定")
    //   --kv-k-bits / --kv-v-bits      one ceiling per plane ("分开定，内部分层")
    // --kv-bits-mode picks the reading when the per-plane form is used:
    // joint | split (default) | ceiling. 0 == not named.
    double kv_joint_bits     = 0.0;
    double kv_k_bits         = 0.0;
    double kv_v_bits         = 0.0;
    bool kv_kv_bits_explicit = false;
    KvBitsMode kv_bits_mode  = KvBitsMode::Split;
    bool kv_bits_mode_explicit = false;
    // Per-plane score tables for the split entry (empty == the kv_tier_scores table).
    std::string kv_k_tier_scores;
    std::string kv_v_tier_scores;
    // --kv-codec-preference CODEC[,CODEC...]: WHICH codec the JOINT bit-budget fit picks
    // among candidates that cost the SAME bits ("同 bit 分配不同种类的量化", not fewer bits).
    // An ORDER over the candidate grammar product::kv_gear_candidate_list() --
    // "rk4v4, bf16, int8, fp8, nvfp4, iso4e", the ladder's own candidate slots, which both
    // the help text and the refusal message SPLICE instead of restating. (This comment used
    // to name the ladder's slot spellings, "e8, bf16, int8, fp8, nvfp4, iso3"; the binary's
    // own refusal names the six above -- measured on the shipped binary, which is the side
    // that is right.) Most-wanted first; stored as ladder slots because
    // that is what the selector speaks. Parsed and validated at parse time -- the grammar is
    // fixed, so no model knowledge is needed -- with unknown/empty/duplicate names refused by
    // name rather than accepted and ignored. Empty == the shipped pack order.
    std::vector<std::int32_t> kv_codec_preference;
    bool kv_codec_preference_explicit = false;
    // The penalty table's own entry point (show | emit=<path>).
    std::string kv_score_table_spec;
    bool kv_score_table_explicit = false;
    bool kv_layer_storage_explicit = false;
    bool kv_residual_layers_explicit = false;
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
    // --recalibrate: force the runtime calibration loop to capture again and
    // overwrite the persisted row-scale table next to the artifact. Only
    // meaningful in the row scale's auto state (see product/kv_rowscale_persist.h).
    bool recalibrate              = false;
    KvVCodec kv_v_codec           = KvVCodec::Iso3;
    bool kv_v_codec_explicit      = false;
    ColdPolicy cold_policy        = ColdPolicy::None;
    std::uint32_t cold_keep_tokens          = 128;
    bool cold_keep_tokens_explicit          = false;
    // 7 GiB, not 4: the 1M band needs F - D = 5,634 pages = 6.2004 GiB at
    // 1,181,745 B/page, and `4ULL << 30` leaves cold_host_window_band EMPTY.
    std::uint64_t cold_host_bytes  = 7ULL << 30;
    // --max-cold-pages: explicit cold-pool cap in pages (0 = derive from the
    // policy). Without it the CLI could only ever use the derived pool
    // (cold_keep_tokens/kPagedKVPageSize + 16), i.e. 18 pages, which is the whole
    // point of "cap the offload" being unreachable from this front end.
    std::uint32_t max_cold_pages   = 0;
    // --kv-unload-watermark-pages: the PROACTIVE free-pool watermark, i.e. the free
    // text-KV pages at or below which the Engine unloads the blocks its semantic
    // directory judges unloadable -- instead of waiting for the pool to overflow.
    // 0 = OFF (the pre-watermark behaviour); kUnloadWatermarkDerive = derive the
    // reserve from the plan's own prefill chunk. See
    // EngineOptions::unload_watermark_pages for the full rule.
    std::uint32_t unload_watermark_pages = kUnloadWatermarkDerive;
    // [PREFILLBUDGET] --recall-prefill-tokens N: the BOUNDED-PREFILL budget, in TOKENS of
    // re-prefill per recall round (dl/vectorkey ROW 2; NINFER_RECALL_PREFILL_TOKENS is the
    // environment spelling and the flag beats it). 0 is NOT accepted from the flag: it is the
    // default, so a named 0 would be "accepted and read by nothing", which is the refusal this
    // front end already makes elsewhere by name. It is a DIFFERENT dimension from --cold-*: the
    // byte budget is 256 MiB because a page is 1,181,696 B of device memory, while the cost of a
    // recall under the inject shape is a re-prefill priced per token -- the tree's own comment on
    // that default says it "is a MEMORY-SAFETY limit rather than a speed limit". At the edge the
    // round is REFUSED BY NAME (`refused-prefill-budget`, on stderr, at plan time and before a
    // single token is re-prefilled) rather than truncated, because a truncated run is a PREFIX of
    // the answer's context -- a partial answer, or a confident wrong one.
    std::uint32_t recall_prefill_tokens = 0;
    // ColdPolicy::Disk spill budget and directory (serve-only flags before).
    std::uint64_t cold_disk_bytes  = 32ULL << 30;
    std::string cold_disk_path;
    // qwen4_exp (FlashNext) PLE n-gram sidecar root (--ple-sidecar). Empty =
    // PLE off. Reaches the engine as EngineOptions::ple_sidecar_root and is
    // validated at startup, so a typo is a refusal rather than a quietly
    // missing PLE residual.
    std::string ple_sidecar_root;
    // W13 weight offload. The CLI reaches the same engine knobs as ninfer-serve so a
    // 1M run can be sized from either front end (same reasoning as max_cold_pages).
    std::uint64_t weight_host_offload_bytes = 0;
    std::uint64_t weight_device_arena_bytes = 0;
    std::uint32_t weight_prefetch_layers    = 2;
    std::uint64_t weight_span_floor_bytes   = 0;

    // --stage-layers SPEC: the PIPELINE STAGE PARTITION of the text-layer axis, kept RAW
    // exactly like kv_layer_storage_spec above and parsed by core/stage_plan.h -- the one
    // implementation the runtime calls too. Empty == the flag was absent == one stage ==
    // axis `none` == the pre-existing single-device run.
    std::string stage_layers_spec;
    // --stage-handoff DIR / --stage-handoff-cut: where the boundary hidden state crosses, and
    // the negative control that silences the producer so the ids MOVE. See core/stage_plan.h.
    std::string stage_handoff_dir;
    bool stage_handoff_cut      = false;
    bool stage_layers_explicit  = false;
    bool stage_handoff_explicit = false;
    bool yarn_enabled     = false;
    std::uint32_t graph_capture_ceiling = 16;
    // FreeToken step 1 observation switch (src/ops/common/ft_stats.h):
    // --ft-stats on|off, or NINFER_FT_STATS in the environment. The flag wins and
    // is committed to that variable at parse time, because ft::enabled() reads it
    // exactly once. Off by default (the observation costs a device -> host copy
    // per decode round), and the *consumer* of the observation (the periodic KV
    // relayout, src/serve/kv_auto_relayout.h) is a server feature: this one-shot
    // front end has no decision cycle, so it has no relayout flag.
    std::optional<bool> ft_stats;

    // --inject-spec PATH: THE INGRESS SURFACE (src/spec/inject_channel.h,
    // src/targets/qwen3_6/impl/runtime/inject_ingress.h). PATH is a declaration file, in
    // that header's flat key=value grammar, of a CHOSEN TENSOR and the position range it
    // will occupy -- direction, dtype, layout, rows, cols, position0, scale, path, digest.
    // It is the one input channel that is not a token channel and not `media` (whose
    // alphabet is the range of merger o layers o patch_embed, i.e. a decodable image is
    // required). The declaration is validated with the header's own parser at parse time
    // -- so a mis-shaped or mis-dtyped declaration refuses BY NAME before the artifact is
    // loaded -- and admitted against the loaded model at bind time, where the row count and
    // the context capacity are finally known. It is committed to NINFER_INJECT_SPEC (the
    // --ft-stats precedent) because the consumption point is inside a device schedule
    // that takes no per-request argument. Empty = the ingress is off and the engine's
    // per-chunk cost is one branch.
    std::string inject_spec;

    bool raw_output      = false;
    bool print_token_ids = false;
    // F745 injectbind: the PROMPT side of --print-token-ids. `PreparedPrompt::prompt_token_ids()`
    // exists (include/ninfer/engine.h:30, "so a measurement consumer can record the input side of a
    // run") and had NO CLI surface -- so the ids a `sum_dir` row's content must be named by were not
    // obtainable from a run, and a row could not be bound to the inject channel. This prints them on
    // stderr, in the order the engine tokenized them, as one space-separated list.
    bool print_prompt_ids = false;
    // M21 --append-context-text: the operator entry for appending a run of already-known tokens to
    // the running request and prefilling them mid-run (recall). The text is encoded by the artifact's
    // own tokenizer with NO chat template and NO implicit special token, i.e. exactly the bytes given
    // here. It is INPUT, not output: it never appears in the generated ids and never consumes
    // --max-new. Empty means the leg is off, which is the default.
    std::string append_context_text;
    bool append_context_explicit = false;
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
