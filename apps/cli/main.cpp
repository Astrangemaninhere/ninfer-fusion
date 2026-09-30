#include "options.h"
#include "product/kv_options.h"
#include "product/ple_sidecar_carrier.h"
#include "product/kv_plane_census.h"
#include "product/kv_storage_dtype.h"
#include "product/load_progress/load_progress.h"
#include "product/kv_rowscale_persist.h"
#include "product/prompt_input/prompt_input.h"
#include "product/speculative_options.h"
#include "product/kv_kv_bits.h"

#include "core/arch_caps.h"

#include "ninfer/engine.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {

using Clock = std::chrono::steady_clock;

std::string format_seconds(double seconds) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3) << seconds << " s";
    return output.str();
}

std::string format_rate(double tokens, double seconds) {
    if (tokens <= 0.0 || seconds <= 0.0) { return "n/a"; }
    std::ostringstream output;
    output << std::fixed << std::setprecision(2) << tokens / seconds << " tok/s";
    return output.str();
}

std::string format_percent(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) { return "n/a"; }
    std::ostringstream output;
    output << std::fixed << std::setprecision(2)
           << 100.0 * static_cast<double>(numerator) / static_cast<double>(denominator) << '%';
    return output.str();
}

std::string format_bytes(std::uint64_t bytes) {
    constexpr double kKiB = 1024.0;
    constexpr double kMiB = 1024.0 * kKiB;
    constexpr double kGiB = 1024.0 * kMiB;
    std::ostringstream output;
    output << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        output << static_cast<double>(bytes) / kGiB << " GiB";
    } else if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        output << static_cast<double>(bytes) / kMiB << " MiB";
    } else if (bytes >= static_cast<std::uint64_t>(kKiB)) {
        output << static_cast<double>(bytes) / kKiB << " KiB";
    } else {
        output << bytes << " B";
    }
    return output.str();
}

std::string format_arena_used(const ninfer::ArenaMemorySummary& arena) {
    return format_bytes(arena.used_bytes) + " / " + format_bytes(arena.capacity_bytes);
}

std::string format_arena_peak(const ninfer::ArenaMemorySummary& arena) {
    return format_bytes(arena.peak_used_bytes) + " / " + format_bytes(arena.capacity_bytes);
}

std::string format_sampling(const ninfer::ResolvedSamplingParameters& sampling) {
    if (sampling.temperature <= 0.0F) { return "greedy (temperature 0)"; }
    std::ostringstream output;
    output << std::fixed << std::setprecision(2) << "temp=" << sampling.temperature
           << " top_p=" << sampling.top_p << " top_k=" << sampling.top_k
           << " min_p=" << sampling.min_p << " presence=" << sampling.presence_penalty
           << " freq=" << sampling.frequency_penalty << " seed=" << sampling.seed;
    return output.str();
}

std::string format_finish(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::None:
        return "none";
    case ninfer::FinishReason::OutputLimit:
        return "output-limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context-capacity";
    case ninfer::FinishReason::StopToken:
        return "stop-token";
    case ninfer::FinishReason::StopString:
        return "stop-string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

// The operator's "kv cache dtype" line -- the name an operator reads to confirm WHICH
// TIER ACTUALLY RAN. DERIVED, not copied: it is product::kv_operator_token, the
// canonical token of the one table the engine has for this enum.
//
// This function used to be a second, hand-written switch. It had drifted: SIX of the
// eight rows it keyed off carried a pre-rename spelling, and it named only EIGHT of the
// ten enumerators, so the two narrow rk4v4-family rows printed "unknown". A run
// therefore reported a tier the engine was not running, and every measurement checked
// against that line was ambiguous. The stale-to-canonical table is written out in full
// in product/kv_storage_dtype.h (beside the function this now calls) and in
// tests/test_kv_operator_name.cpp (check 1) -- NOT here, because the census in that
// test reads THIS file as text and must be able to stay strict.
//
// It stays a named function (rather than an inlined call) because
// tests/test_kv_operator_name.cpp asserts the CLI keeps routing this line through it.
// The refusal is inherited: a storage code no enumerator names throws the canonical
// lookup's std::invalid_argument instead of printing something that looks like a tier.
std::string format_kv_cache(ninfer::KvCacheStorage storage) {
    return ninfer::product::kv_operator_token(storage);
}

// The RESOLVED per-layer KV store, compressed into
// "0,1,3,4,6,7:rk4v4-g64 2,5,8-15:nvfp4-g16" (canonical tokens).
// One tier across every full-attention layer keeps the old single-name form, so a
// uniform run reads exactly as it did.
std::string format_kv_layer_store(const ninfer::MemorySummary& memory) {
    const std::uint32_t layers =
        memory.kv_full_attention_layers < memory.kv_layer_storage.size()
            ? memory.kv_full_attention_layers
            : static_cast<std::uint32_t>(memory.kv_layer_storage.size());
    if (layers == 0) { return format_kv_cache(memory.kv_cache); }
    std::ostringstream runs;
    bool uniform = true;
    for (std::uint32_t first = 0; first < layers;) {
        std::uint32_t last = first;
        while (last + 1 < layers &&
               memory.kv_layer_storage[last + 1] == memory.kv_layer_storage[first]) {
            ++last;
        }
        if (first != 0) {
            uniform = false;
            runs << ' ';
        }
        runs << first;
        if (last != first) { runs << '-' << last; }
        runs << ':' << format_kv_cache(memory.kv_layer_storage[first]);
        first = last + 1;
    }
    if (uniform) { return format_kv_cache(memory.kv_layer_storage[0]); }
    return "per-layer " + runs.str() + " (" + std::to_string(layers) +
           " full-attention layers)";
}

// The AGGREGATE of the same table format_kv_layer_store above prints run by run: how
// many distinct codecs the store spans, and how many layers own no plane at all.
//
// It is a SEPARATE line rather than a suffix on the one above because the two answer
// different questions and one of them cannot be read off the other. The run form tells
// an operator WHICH codec sits on WHICH layer; it does not say whether the store is
// MIXED, and reading that off the text means comparing every run by eye -- which is
// exactly the reading that failed before this file carried a derived, total token table
// (see format_kv_cache above). "mixed" is a named state of the store, and named states
// are what a regex and a reader can both key on.
//
// The dropped count is printed even when it is 0, so "no layer was discarded" and "this
// build does not report discard" cannot be confused. The census itself -- including the
// refusal to count a DISCARDED layer as a codec -- lives in product/kv_plane_census.h,
// NOT here: this function only adapts MemorySummary to it and names the caller.
std::string format_kv_plane_census(const ninfer::MemorySummary& memory) {
    return ninfer::product::kv_plane_census_line(
        ninfer::product::kv_plane_census(memory, "kv planes"));
}

std::string format_kv_capacity_mode(ninfer::KvCapacityMode mode) {
    return mode == ninfer::KvCapacityMode::Automatic ? "auto" : "explicit";
}

// `--kv-row-scale FILE`: the file has to exist, and "the file is not there" is the one
// state this switch cannot recover from.  The engine's own loader DOES refuse it by
// throwing "KVRS open: <path>" (src/ops/kernel/gqa_isoquant_row_scale_loader.cu:78), but
// it does so from the row-scale COMMIT POINT -- inside the engine's plan
// (src/targets/qwen3_6/impl/state/decoder_state.cpp), after the artifact has been opened
// and the weights mapped.  A typo in a path therefore cost a full artifact load before it
// was answered, and the `--kv-row-scale <dir>` spelling got as far as fopen and died
// there.  The stat belongs HERE, next to the flag's own landing, because it is a stat on
// a path the operator typed and it needs no model, no device and no plan -- the same
// place and the same reason as validate_ple_sidecar_root below.
//
// Tested against the vocabulary's OWN parser rather than against a list of spellings
// written out again here, so `auto` / `off` stay states (not filenames) exactly as
// long as the loader says they are, and no legal run can become a refusal.
void validate_kv_row_scale_path(const std::string& spec) {
    ninfer::ops::KvRowScaleMode mode = ninfer::ops::KvRowScaleMode::Auto;
    std::string path;
    std::string ignored;
    if (!ninfer::ops::kv_rowscale_mode_from_spec(spec, mode, path, ignored)) { return; }
    if (mode != ninfer::ops::KvRowScaleMode::Path) { return; }
    std::error_code error;
    const bool regular = std::filesystem::is_regular_file(path, error);
    if (regular && !error) { return; }
    throw std::invalid_argument(
        "--kv-row-scale '" + path + "' names a FILE and no readable regular file is "
        "there" + (error ? std::string(" (") + error.message() + ")" : std::string()) +
        ". --kv-row-scale accepts auto (the baked table), off (the identity row scale, "
        "which needs no file) or the path of an NINFERKVRS1 sidecar. The engine refuses a "
        "missing file as well, but only at the row-scale commit point, i.e. after the "
        "artifact is already open, so the typo cost a load before it was answered.");
}

// The two raw-spec parsers that run ahead of Engine construction.  Their verdicts are
// kept VERBATIM -- they are the one definition of those grammars -- and only the value
// the operator typed is added, because several of their refusals name the flag without
// printing the text that failed in it (kv-layer-storage's layer-index bounds say "layer
// index out of range" with neither the index nor the layer text).
template <typename Callable>
auto kv_spec_context(const char* flag, const std::string& spec, Callable&& call)
    -> decltype(call()) {
    try {
        return call();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(std::string(error.what()) + " [" + flag + " '" + spec +
                                    "']");
    }
}

void print_stage(std::string_view group, std::string_view detail, double seconds) {
    std::cerr << std::left << std::setw(12) << group << std::setw(26) << detail << std::right
              << std::setw(12) << format_seconds(seconds) << '\n';
}

void print_metric(std::string_view label, std::string_view value) {
    std::cerr << std::left << std::setw(12) << "summary" << std::setw(26) << label << value << '\n';
}

class StreamingSink final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart) override {}

    void publish(ninfer::OutputDelta delta) override {
        std::ostream& output =
            delta.channel == ninfer::OutputChannel::Reasoning ? std::cerr : std::cout;
        output << delta.text;
        output.flush();
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            reasoning_seen_ = reasoning_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { reasoning_ends_in_newline_ = delta.text.back() == '\n'; }
        } else {
            content_seen_ = content_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { content_ends_in_newline_ = delta.text.back() == '\n'; }
        }
    }

    void finish_streams() const {
        if (!content_seen_ || !content_ends_in_newline_) { std::cout << '\n'; }
        std::cout.flush();
        if (reasoning_seen_ && !reasoning_ends_in_newline_) { std::cerr << '\n'; }
    }

private:
    bool content_seen_              = false;
    bool content_ends_in_newline_   = false;
    bool reasoning_seen_            = false;
    bool reasoning_ends_in_newline_ = false;
};

void print_load_summary(const ninfer::LoadSummary& load, double wall_seconds) {
    print_stage("load", "engine construction", wall_seconds);
    print_stage("load", "artifact/materialize", load.load_seconds);
    print_stage("load", "host to device", load.upload_seconds);
    print_metric("target", load.target);
    print_metric("weights", load.weights_id);
    print_metric("artifact file read", format_bytes(load.artifact_bytes_read));
    print_metric("weight H2D", format_bytes(load.host_to_device_bytes));
    print_metric("pinned staging peak", format_bytes(load.peak_staging_bytes));
    print_metric("tensors/resources",
                 std::to_string(load.tensor_count) + " / " + std::to_string(load.resource_count));
}

void print_generation_summary(const ninfer::GenerationResult& result,
                              const ninfer::ResolvedSamplingParameters& sampling,
                              const ninfer::MemorySummary& memory) {
    print_stage("prepare", "render/preprocess", result.timings.prepare_seconds);
    print_stage("generate", "vision", result.timings.vision_seconds);
    print_stage("generate", "text prefill", result.timings.prefill_seconds);
    print_stage("generate", "decode", result.timings.decode_seconds);
    print_stage("generate", "total", result.timings.total_seconds);

    const std::size_t generated = result.generated_token_ids.size();
    const std::size_t decoded   = generated == 0 ? 0 : generated - 1;
    const double model_seconds  = result.timings.vision_seconds + result.timings.prefill_seconds +
                                 result.timings.decode_seconds;
    print_metric("sampling", format_sampling(sampling));
    print_metric("finish reason", format_finish(result.finish_reason));
    print_metric("prompt tokens", std::to_string(result.prompt.prompt_tokens));
    print_metric("reused prompt tokens", std::to_string(result.reused_prompt_tokens));
    print_metric("generated tokens", std::to_string(generated));
    if (result.thinking.configured_budget) {
        print_metric("thinking budget", std::to_string(*result.thinking.configured_budget));
        print_metric("model thinking tokens",
                     std::to_string(result.thinking.model_thinking_tokens));
        print_metric("thinking control tokens", std::to_string(result.thinking.injected_tokens));
        print_metric("thinking control", result.thinking.applied ? "applied" : "not applied");
    }
    print_metric("model elapsed", format_seconds(model_seconds));
    print_metric("prefill speed", format_rate(static_cast<double>(result.prompt.prompt_tokens),
                                              result.timings.prefill_seconds));
    print_metric("decode speed",
                 format_rate(static_cast<double>(decoded), result.timings.decode_seconds));
    print_metric("throughput (overall)",
                 format_rate(static_cast<double>(generated), model_seconds));

    const std::uint64_t reserved = static_cast<std::uint64_t>(memory.weights.capacity_bytes) +
                                   memory.runtime_reservation_bytes;
    print_metric("device", std::to_string(memory.device));
    print_metric("max context", std::to_string(memory.max_context));
    print_metric("KV capacity policy", format_kv_capacity_mode(memory.kv_capacity_mode));
    print_metric("KV capacity", std::to_string(memory.kv_capacity));
    print_metric("KV page groups", std::to_string(memory.kv_capacity_page_groups) + " / " +
                                       std::to_string(memory.kv_capacity_max_page_groups));
    print_metric("gpu weights used", format_arena_used(memory.weights));
    print_metric("gpu sequence used", format_arena_used(memory.sequence));
    print_metric("kv cache dtype", format_kv_layer_store(memory));
    print_metric("kv planes", format_kv_plane_census(memory));
    print_metric("kv cache payload", format_bytes(memory.kv_payload_bytes));
    print_metric("gpu workspace peak", format_arena_peak(memory.workspace));
    print_metric("runtime reservation", format_bytes(memory.runtime_reservation_bytes));
    print_metric("free after weights", format_bytes(memory.available_after_weights_bytes));
    print_metric("free after startup", format_bytes(memory.available_after_startup_bytes));
    print_metric("KV capacity headroom", format_bytes(memory.kv_capacity_headroom_bytes));
    print_metric("planned slack", format_bytes(memory.planned_slack_bytes));
    print_metric("CUDA Graph allowance", format_bytes(memory.cuda_graph_allowance_bytes));
    print_metric("planned device total", format_bytes(reserved));

    const ninfer::SpeculativeStats& speculative = result.speculative;
    if (speculative.enabled) {
        // Every backend names itself: the previous DFlash-or-MTP ternary labelled DFlash2 runs
        // as "mtp" and made the CLI acceptance summary unattributable.
        const std::string backend =
            ninfer::product::speculative_backend_name(speculative.backend);
        if (speculative.adaptive_window) {
            // Adaptive runs must not look like a fixed k: draft_window is the ladder TOP (the
            // widest captured rung), and the realized mean is what the criterion chose.
            std::ostringstream window;
            window << speculative.draft_window << " (adaptive ladder, top) realized mean "
                   << std::fixed << std::setprecision(2) << speculative.mean_window;
            print_metric(backend + " draft window", window.str());
        } else {
            print_metric(backend + " draft window", std::to_string(speculative.draft_window));
        }
        print_metric(backend + " rounds", std::to_string(speculative.rounds));
        print_metric(backend + " fallback steps", std::to_string(speculative.fallback_steps));
        print_metric(backend + " drafted tokens", std::to_string(speculative.drafted_tokens));
        print_metric(backend + " accepted tokens", std::to_string(speculative.accepted_tokens));
        print_metric(backend + " acceptance rate",
                     format_percent(speculative.accepted_tokens, speculative.drafted_tokens));
        if (speculative.rounds != 0) {
            std::ostringstream length;
            length << std::fixed << std::setprecision(2)
                   << 1.0 + static_cast<double>(speculative.accepted_tokens) /
                                static_cast<double>(speculative.rounds)
                   << " tok/round";
            print_metric(backend + " acceptance length", length.str());
        }
        if (!speculative.accepted_per_position.empty()) {
            std::ostringstream positions;
            for (std::size_t i = 0; i < speculative.accepted_per_position.size(); ++i) {
                if (i != 0) { positions << ','; }
                positions << speculative.accepted_per_position[i];
            }
            print_metric(backend + " accepted by pos", positions.str());
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const ninfer::cli::Options cli = ninfer::cli::parse_options(argc, argv);
        if (cli.help_requested) {
            std::cout << ninfer::cli::usage_text(argv[0]);
            return 0;
        }

        // --capability-report: the BUILD capability surface's own entry point, handled here
        // for the same reason --kv-score-table is (below): it runs with NO model and NO prompt.
        // The flag exists because the arch list this binary was compiled for had NO model-free
        // surface at all -- the refusal report is reached only through artifact load
        // (src/targets/registry.cpp, construct_target -> require_artifact_formats_supported).
        // The text lives in src/core/arch_caps.h beside the refusal it is the sibling of, so the
        // arch line both print has ONE home and the two cannot disagree. A model path given as
        // well continues the run afterwards, exactly as --kv-score-table does.
        if (cli.capability_report_requested) {
            std::cout << ninfer::caps::render_build_capability_surface();
            if (cli.artifact_path.empty()) { return 0; }
        }

        // --kv-score-table: the penalty table's OWN entry point, handled here so it runs
        // with NO model and NO prompt. The action itself lives in
        // product/kv_kv_bits.h (kv_score_table_run), shared with ninfer-serve, so the
        // two front ends cannot disagree about what the entry does. When a model IS
        // given the run continues afterwards, which is how "emit and consume" is proven
        // in one go.
        if (cli.kv_score_table_explicit) {
            std::string error;
            if (!ninfer::product::kv_score_table_run(cli.kv_score_table_spec,
                                                     cli.kv_tier_scores, &error, std::cout,
                                                     std::cerr)) {
                std::cerr << error << "\n";
                return 2;
            }
            if (cli.artifact_path.empty()) { return 0; }
        }

        ninfer::PromptInput input =
            cli.messages_path.empty()
                ? ninfer::product::prompt_from_text(cli.prompt, cli.enable_thinking)
                : ninfer::product::prompt_from_messages(cli.messages_path, cli.enable_thinking,
                                                        cli.enable_vision);
        input.options.reasoning_effort = cli.reasoning_effort;

        ninfer::RequestOptions request;
        request.execution.sampling                = cli.sampling;
        request.execution.requested_output_tokens = cli.max_new;
        request.execution.thinking.budget         = cli.thinking_budget;
        request.stop.token_ids                    = cli.stop_token_ids;
        request.stop.strings                      = cli.stop_strings;
        request.output.raw                        = cli.raw_output;

        std::cerr << "phase       detail                      elapsed/progress\n";
        ninfer::product::LoadProgressRenderer load_progress(
            std::cerr, ninfer::product::stderr_load_progress_options());
        ninfer::EngineOptions engine_options;
        engine_options.artifact_path  = cli.artifact_path;
        engine_options.device         = cli.device;
        engine_options.max_context    = cli.max_context;
        // The knob the engine already had and no CLI path could set. Same flag spelling, same
        // [1,16] bound and same refusal wording as ninfer-serve (src/serve/serve_options.cpp
        // :279-281 the branch, :831-832 the bound). An unset flag leaves the engine default
        // (1, include/ninfer/types.h:380), so every existing invocation is unchanged.
        engine_options.max_concurrency = cli.max_concurrency;
        engine_options.kv_capacity    = cli.kv_capacity;
        engine_options.prefill_chunk  = cli.prefill_chunk;
        // Only when the operator named a mode: an unset mode defers to NINFER_FT_BW_GOV and then to
        // Dynamic, and that resolution belongs to the engine (normalize_engine_options), not here --
        // a front end that guessed would become a second reader of the same switch.
        if (cli.prefill_chunk_mode.has_value()) {
            engine_options.prefill_chunk_mode = cli.prefill_chunk_mode;
        }
        engine_options.kv_cache       = cli.kv_cache;
        engine_options.speculative    = cli.speculative;
        engine_options.enable_vision  = cli.enable_vision;
        engine_options.yarn_enabled  = cli.yarn_enabled;
        engine_options.use_cuda_graph = cli.use_cuda_graph;
        // --stage-layers: the stage partition the decode loop walks. The spec travels RAW --
        // the one parser is core/stage_plan.h and it is called in the runtime, where the
        // artifact's layer count exists and plan_shards() can be asked whether this spec is a
        // world the rank axis derives. A front end that parsed it into a second form here
        // would be the second spelling of the grammar this file refuses to be (see the
        // --kv-layer-storage comment below).
        if (cli.stage_layers_explicit) {
            engine_options.stage_layers_spec = cli.stage_layers_spec;
            engine_options.stage_handoff_dir = cli.stage_handoff_dir;
            engine_options.stage_handoff_cut = cli.stage_handoff_cut;
        }
        engine_options.kv_cache_explicit = cli.kv_cache_explicit;
        if (cli.kv_layer_storage_explicit) {
            // Table AND mask. The mask is what makes `0-11:bf16` a real per-layer
            // BF16 baseline: without it BFloat16 is the "unset" sentinel and every
            // one of those layers silently inherits --kv-dtype instead (the old
            // parse_kv_layer_storage() returns only the table).
            const auto parsed = kv_spec_context(
                "--kv-layer-storage", cli.kv_layer_storage_spec, [&] {
                    return ninfer::product::parse_kv_layer_storage_spec(
                        cli.kv_layer_storage_spec);
                });
            engine_options.kv_layer_storage          = parsed.table;
            engine_options.kv_layer_storage_set      = parsed.set;
            engine_options.kv_layer_storage_explicit = true;
        }
        if (cli.kv_residual_layers_explicit) {
            // The per-layer NVFP4 residual planes: the ONLY handle in the tree that names
            // a real plane SUBSET (4 -> 8 planes on an NVFP4 layer). Two fields, not one:
            // the planner gates on kv_residual_explicit and reads the table only when it
            // is set (layouts_impl.h make_sequence_planner_impl), so an unset flag has to
            // stay "no opinion" -- an all-false table handed over unconditionally would
            // read as "explicitly no residuals" on every other run.
            const auto parsed = kv_spec_context(
                "--kv-residual-layers", cli.kv_residual_layers_spec, [&] {
                    return ninfer::product::parse_kv_residual_layers_spec(
                        cli.kv_residual_layers_spec);
                });
            engine_options.kv_residual_layers   = parsed.table;
            engine_options.kv_residual_explicit = true;
        }
        if (cli.kv_tier_formats_explicit) {
            // Raw passthrough: the vocabulary is validated at parse time and landed on the
            // per-layer dtype table in the planner (product/kv_tier_formats.h explains why).
            engine_options.kv_tier_formats_spec     = cli.kv_tier_formats_spec;
            engine_options.kv_tier_formats_explicit = true;
            engine_options.kv_nvfp4_pure            = cli.kv_nvfp4_pure;
        }
        // SEPARATION: the three KV component switches. Explicit flags only: an
        // unset flag leaves the engine options at their pre-separation defaults
        // (rotation on, row scale auto, V codec iso4e), and none of the three
        // uploads or allocates anything unless it names a non-default state.
        if (cli.kv_rotation_explicit) {
            engine_options.kv_rotation_off      = cli.kv_rotation_off;
            engine_options.kv_rotation_explicit = true;
        }
        if (cli.kv_row_scale_explicit) {
            // A path that is not there is answered HERE, not at the commit point: see
            // validate_kv_row_scale_path above. auto/off are states and are untouched.
            validate_kv_row_scale_path(cli.kv_row_scale_spec);
            engine_options.kv_row_scale_spec     = cli.kv_row_scale_spec;
            engine_options.kv_row_scale_explicit = true;
        }
        if (cli.kv_v_codec_explicit) {
            engine_options.kv_v_codec          = cli.kv_v_codec;
            engine_options.kv_v_codec_explicit = true;
        }
        if (cli.kv_bit_budget_explicit) {
            engine_options.kv_bit_budget_bits     = cli.kv_bit_budget_bits;
            engine_options.kv_bit_budget_ranges   = cli.kv_bit_budget_ranges;
            engine_options.kv_bit_budget_explicit = true;
        }
        // Two-score KV selection: the budget DP reads both of these
        // (layouts_impl.h make_sequence_planner_impl). options.cpp parses them into
        // cli::Options but nothing copied them across, so the flags were accepted
        // and then ignored. The cli defaults (weight -1, empty table) are exactly
        // the engine defaults, so an unset flag stays a no-op.
        engine_options.kv_quality_weight     = cli.kv_quality_weight;
        engine_options.kv_tier_scores        = cli.kv_tier_scores;
        // The K/V bit-width entries (product/kv_kv_bits.h). Copied unconditionally
        // like the two above: the cli defaults (0 / Split) are exactly the engine
        // defaults, so an unset flag stays a no-op.
        engine_options.kv_joint_bits         = cli.kv_joint_bits;
        engine_options.kv_k_bits             = cli.kv_k_bits;
        engine_options.kv_v_bits             = cli.kv_v_bits;
        engine_options.kv_kv_bits_explicit   = cli.kv_kv_bits_explicit;
        engine_options.kv_bits_mode          = cli.kv_bits_mode;
        engine_options.kv_bits_mode_explicit = cli.kv_bits_mode_explicit;
        engine_options.kv_k_tier_scores      = cli.kv_k_tier_scores;
        engine_options.kv_v_tier_scores      = cli.kv_v_tier_scores;
        // SLIDERWIRE: the candidate preference travels with the request (it is read by the
        // joint fit and checked there to have been honoured).
        engine_options.kv_codec_preference   = cli.kv_codec_preference;
        engine_options.cold_policy           = cli.cold_policy;
        engine_options.cold_keep_tokens      = cli.cold_keep_tokens;
        engine_options.cold_host_bytes       = cli.cold_host_bytes;
        // Cold-pool shape and the disk spill target used to be reachable only
        // through ninfer-serve, so the CLI could not size the offload pool at all.
        engine_options.max_cold_pages        = cli.max_cold_pages;
        // The unload watermark reaches the Engine from this front end too, for the
        // same reason as max_cold_pages above: the CLI must be able to arm the leg it
        // is being measured on, or the acceptance run has to go through ninfer-serve
        // purely to set one integer.
        engine_options.unload_watermark_pages = cli.unload_watermark_pages;
        engine_options.cold_disk_bytes       = cli.cold_disk_bytes;
        engine_options.cold_disk_path        = cli.cold_disk_path;
        // The FlashNext PLE n-gram sidecar root. Validated HERE rather than at
        // parse time: the check stats the filesystem, and it has to land before
        // the engine is constructed so an operator typo stops the run instead of
        // leaving the PLE residual out of the arithmetic with no other symptom.
        engine_options.ple_sidecar_root      = cli.ple_sidecar_root;
        ninfer::product::validate_ple_sidecar_root(engine_options.ple_sidecar_root);
        engine_options.weight_host_offload_bytes = cli.weight_host_offload_bytes;
        engine_options.weight_device_arena_bytes = cli.weight_device_arena_bytes;
        engine_options.weight_prefetch_layers    = cli.weight_prefetch_layers;
        engine_options.weight_span_floor_bytes   = cli.weight_span_floor_bytes;
        engine_options.graph_capture_ceiling = cli.graph_capture_ceiling;
        // One CLI invocation owns exactly one request, so retained cross-request context has no
        // consumer and must not reserve an extra Device StateImage or run terminal capture.
        engine_options.context_cache.enabled                = false;
        engine_options.context_cache.host_state_slots       = 0;
        engine_options.context_cache.host_kv_capacity_bytes = 0;
        engine_options.load_progress                        = load_progress.callback();

        // N3 runtime loop. The persisted row-scale table lives next to the
        // artifact, and whether THIS run has to capture is decided here, before
        // the engine exists: the calibration capture synchronizes the producing
        // stream (illegal inside a graph capture), so the graph decision has to
        // be taken before construction. The applicability gate itself runs at the
        // row-scale commit point (targets/qwen3_6/impl/state/decoder_state.cpp),
        // which is where the live KV geometry is known.
        if (!cli.kv_row_scale_explicit ||
            ninfer::product::kv_rowscale_spec_is_auto(cli.kv_row_scale_spec)) {
            ninfer::product::KvRowScaleConfigKnobs knobs;
            knobs.kv_cache_code       = static_cast<int>(cli.kv_cache);
            knobs.kv_cache_explicit   = cli.kv_cache_explicit;
            knobs.layer_storage_spec  = cli.kv_layer_storage_spec;
            knobs.tier_formats_spec   = cli.kv_tier_formats_spec;
            knobs.nvfp4_pure          = cli.kv_nvfp4_pure;
            knobs.rotation_off        = cli.kv_rotation_off;
            knobs.rotation_explicit   = cli.kv_rotation_explicit;
            knobs.v_codec             = static_cast<int>(cli.kv_v_codec);
            knobs.bit_budget_bits     = cli.kv_bit_budget_bits;
            knobs.bit_budget_ranges   = cli.kv_bit_budget_ranges;
            knobs.bit_budget_explicit = cli.kv_bit_budget_explicit;
            knobs.quality_weight      = cli.kv_quality_weight;
            knobs.tier_scores         = cli.kv_tier_scores;
            // A K/V ceiling changes what the rotated K domain is quantized to, which
            // is exactly the class of knob this fingerprint exists to catch
            // (product/kv_rowscale_persist.h): a table baked under one K ceiling must
            // never be reused under another.
            knobs.joint_bits          = cli.kv_joint_bits;
            knobs.k_bits              = cli.kv_k_bits;
            knobs.v_bits              = cli.kv_v_bits;
            knobs.kv_bits_mode        = static_cast<int>(cli.kv_bits_mode);
            knobs.kv_bits_explicit    = cli.kv_kv_bits_explicit;
            knobs.k_tier_scores       = cli.kv_k_tier_scores;
            knobs.v_tier_scores       = cli.kv_v_tier_scores;
            // SLIDERWIRE: the preference changes which per-layer dtype table the fit emits,
            // i.e. what the rotated K domain is quantized to, so a table baked under one
            // preference must not validate against another. Mixed in ONLY when set (see
            // kv_rowscale_config_fingerprint), so no existing hash moves.
            knobs.codec_preference    = [&] {
                std::string text;
                for (const std::int32_t slot : cli.kv_codec_preference) {
                    if (!text.empty()) { text += ","; }
                    text += std::to_string(slot);
                }
                return text;
            }();
            // ROPE REGIME: --yarn moves the frequencies the captured K was built with,
            // i.e. the rotated domain the table is solved from.  It was the one
            // operator-visible knob that reaches the capture and that this fingerprint
            // did not carry, so a table baked without it validated under it.  Mixed in
            // ONLY when it is not the default, so this line cannot invalidate a table
            // baked by a run that used no --yarn.
            knobs.rope_regime         = cli.yarn_enabled ? 1 : 0;
            knobs.cold_policy         = static_cast<int>(cli.cold_policy);
            knobs.max_cold_pages      = cli.max_cold_pages;

            ninfer::product::KvRowScalePersistConfig persist;
            persist.enabled        = true;
            persist.artifact       = cli.artifact_path;
            persist.table          = ninfer::product::kv_rowscale_table_path(cli.artifact_path);
            persist.records        = ninfer::product::kv_rowscale_records_path(persist.table);
            persist.fingerprint    = ninfer::product::kv_rowscale_config_fingerprint(knobs);
            // EXPLICIT INPUT (this line's change). The table's NAME carries the KV
            // configuration it was baked for, so this run reads its own file first and
            // writes ONLY that one: the shared legacy name stays readable (priority 2,
            // and only while its tag still names this configuration) but stops being a
            // write target, so a capture here cannot overwrite the calibration another
            // configuration is still using. NINFER_KV_ROWSCALE_SCOPE=off restores the
            // historical single-file behaviour, and the loop NAMES that choice.
            // The fingerprint is the one computed on the line above -- the same value the
            // tag carries -- so the name and the tag cannot disagree.
            persist.scope_opted_out = ninfer::product::kv_rowscale_scope_opted_out();
            if (!persist.scope_opted_out) {
                persist.scoped_table = ninfer::product::kv_rowscale_scoped_table_path(
                    persist.table, persist.fingerprint);
                persist.records = ninfer::product::kv_rowscale_records_path(persist.scoped_table);
            }
            persist.recalibrate    = cli.recalibrate;
            persist.graphs_enabled = cli.use_cuda_graph;
            // One second of slack: the capture only has to out-date the PREVIOUS
            // run's records, and filesystem timestamps can be coarser than that.
            persist.run_start_unix = static_cast<std::int64_t>(std::time(nullptr)) - 1;
            if (ninfer::product::kv_rowscale_persist_begin(std::move(persist)) ==
                ninfer::product::KvRowScalePlan::Capture) {
                // This is the one place where an offline work mode and CUDA graphs
                // compete, so the trade is made explicit here rather than left to
                // the Engine: NINFER_KVDUMP_DIR / NINFER_FT_STATS resolve the same
                // conflict the same way (they are graphs-off modes too).
                // GRAPH-ON DEFAULT (the defect this arm closes).  A row-scale calibration
                // is one-shot and CANNOT be graph-capturable: it synchronizes the producing
                // stream.  The two are therefore mutually exclusive, and which one this run
                // buys is the operator's choice, not the loop's -- charging EVERY later run's
                // graphs for a capture that only ever has to happen once is what made
                // `--kv-row-scale auto` silently cost most of a decode's throughput.  When
                // graphs are on and no calibration was asked for, the run KEEPS its graphs and
                // decodes from the baked table; the commit point
                // (src/product/kv_rowscale_persist.h, the `config.graphs_enabled` arm) then
                // reports the opt-in instead of arming the capture.  The calibration itself is
                // unchanged and still one command away: both `--no-cuda-graph` and
                // `--recalibrate` take the else arm below.
                if (cli.use_cuda_graph && !cli.recalibrate) {
                    const bool decidable = ninfer::product::kv_rowscale_config_can_calibrate(
                        knobs, cli.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16);
                    std::cerr
                        << "[kvrowscale] CUDA graphs left ON: the calibration capture "
                           "synchronizes the producing stream, so it is deferred rather than "
                           "paid for by every run"
                        << (decidable
                                ? "; re-run with --no-cuda-graph or --recalibrate to calibrate"
                                : ", and this KV configuration cannot be calibrated at all "
                                  "(no full-attention layer resolves to the NVFP4 tier the row "
                                  "scale is read by), so the .skip note is written at the commit "
                                  "point")
                        << '\n';
                } else {
                    engine_options.use_cuda_graph = false;
                    ninfer::product::kv_rowscale_persist_config().graphs_enabled = false;
                    std::cerr << "[kvrowscale] CUDA graphs disabled for this calibration run "
                                 "(the capture synchronizes the producing stream)\n";
                }
            }
        }

        const auto load_started = Clock::now();
        ninfer::Engine engine(std::move(engine_options));
        const double load_wall = std::chrono::duration<double>(Clock::now() - load_started).count();
        print_load_summary(engine.load_summary(), load_wall);
        engine.reset_memory_peaks();

        ninfer::PreparedPrompt prompt = engine.prepare(std::move(input));

        // F1094: the prompt ids must be READ HERE. engine.submit() below moves `prompt` away, and
        // PreparedPrompt::prompt_token_ids() (src/runtime/engine/engine.cpp:288-291) returns an
        // empty vector once impl_ is null -- which is why --print-prompt-ids printed its label
        // with nothing after it. Captured before the move; the print site keeps its wording and
        // its position. When the flag is unset the capture is an empty vector and nothing changes.
        const std::vector<ninfer::TokenId> prompt_token_ids_before_submit =
            cli.print_prompt_ids ? prompt.prompt_token_ids() : std::vector<ninfer::TokenId>{};

        StreamingSink sink;
        ninfer::GenerationHandle generation = engine.submit(std::move(prompt), std::move(request),
                                                            ninfer::OutputConsumerMode::Streaming);
        const ninfer::ResolvedSamplingParameters sampling = generation.resolved_sampling();
        // M21: arm the mid-run context append BEFORE wait(), so the engine services it at a round
        // boundary that precedes this request's first decode round. The engine guarantees that
        // ordering; [context-append] on stderr is the readback of what actually happened.
        if (cli.append_context_explicit) {
            const std::vector<ninfer::TokenId> append_tokens =
                engine.tokenize_text(cli.append_context_text);
            std::cerr << std::left << std::setw(12) << "context-append" << std::setw(26)
                      << "armed (raw tokenizer)" << std::right << std::setw(12)
                      << (std::to_string(append_tokens.size()) + " tok") << '\n';
            engine.append_context_tokens(generation, append_tokens);
        }
        const ninfer::GenerationResult result             = generation.wait(&sink);
        sink.finish_streams();

        // N3 runtime loop: persist what the capture run measured (while the
        // engine is still alive -- the bake reads the SO(4) matrix back from the
        // device). It never throws: a bake that cannot be produced confidently
        // must leave NO table behind, so the next run captures again, rather than
        // leave a wrong one to be loaded.
        ninfer::product::kv_rowscale_persist_finish();

        // F745 injectbind: the INPUT side. Printed before the generated ids so a reader of stderr
        // sees the input first, and to stderr so stdout stays the answer.
        if (cli.print_prompt_ids) {
            const std::vector<ninfer::TokenId>& prompt_ids = prompt_token_ids_before_submit;
            std::cerr << std::left << std::setw(12) << "prompt" << std::setw(26) << "prompt ids";
            for (std::size_t i = 0; i < prompt_ids.size(); ++i) {
                if (i != 0) { std::cerr << ' '; }
                std::cerr << prompt_ids[i];
            }
            std::cerr << '\n';
        }
        if (cli.print_token_ids) {
            std::cerr << std::left << std::setw(12) << "tokens" << std::setw(26) << "generated ids";
            for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
                if (i != 0) { std::cerr << ' '; }
                std::cerr << result.generated_token_ids[i];
            }
            std::cerr << '\n';
        }
        print_generation_summary(result, sampling, engine.memory_summary());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        std::cerr << ninfer::cli::usage_text(argv[0]);
        return 1;
    }
}
