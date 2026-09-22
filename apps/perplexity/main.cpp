#include "corpus.h"
#include "evaluation.h"

#include "ninfer/engine.h"
#include "product/kv_options.h"
#include "product/kv_summary_format.h"
#include "runtime/engine/bandwidth_governor.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using json  = nlohmann::json;
using ninfer::perplexity::CorpusSelection;
using ninfer::perplexity::ScoreAggregate;
using ninfer::perplexity::WindowPlan;

struct Options {
    std::filesystem::path artifact;
    std::optional<std::filesystem::path> corpus;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> output;
    std::uint32_t context     = 4096;
    std::uint32_t stride      = 2048;
    int device                = 0;
    // Unset means "whatever the engine defaults to" (ninfer/types.h:214), so the
    // report below can only be faithful if it reads the value back from the Engine
    // instead of repeating a literal. This knob exists so that the reported number
    // is demonstrably a function of the run and not of the reporting code.
    std::optional<std::uint32_t> prefill_chunk;
    // --prefill-chunk-mode. Only `manual` is accepted here, and that is a FACT about this path, not
    // a preference: a CausalScoreCore owns no Scheduler and no governor (engine.cpp
    // normalize_engine_options), so the score tile cannot adapt. `dynamic` is refused by the engine
    // by name rather than accepted and ignored -- and the field still exists so that the flag's
    // absence here is a decision recorded in the code instead of an omission.
    std::optional<ninfer::PrefillChunkMode> prefill_chunk_mode;
    ninfer::KvCacheStorage kv = ninfer::KvCacheStorage::Fp8E4M3Row256;
    std::array<ninfer::KvCacheStorage, ninfer::kKvLayerStorageSlots> kv_layer_storage{};
    // Which slots --kv-layer-storage actually wrote; see
    // EngineOptions::kv_layer_storage_set and product/kv_options.h KvLayerStorageSpec.
    std::array<bool, ninfer::kKvLayerStorageSlots> kv_layer_storage_set{};
    bool kv_layer_storage_explicit = false;
    // --kv-dtype is an explicit global tier: without this bit, layouts_impl.h
    // always takes either the pinned per-layer table or the target's registered
    // default table, so --kv-dtype is a dead label here (measured: a bf16 run and
    // an fp8 run came out bit-identical). Same field as include/ninfer/types.h.
    bool kv_cache_explicit = false;
    // --kv-residual-layers SPEC: the per-layer NVFP4 second-stage residual planes
    // (include/ninfer/types.h EngineOptions::kv_residual_layers). This is the ONLY handle
    // in the tree that names a real plane SUBSET -- an NVFP4 layer goes from 4 planes to 8
    // -- and until this field existed the flag was reachable from ninfer-serve alone, so
    // the independent-plane layer could not be scanned from a scoring run at all. The
    // table is parsed here (not carried raw) because this front end has no second reader
    // of the spelling; the grammar is the shared product parser's, so all three front
    // ends accept one grammar and emit one set of errors.
    std::array<bool, ninfer::kKvLayerStorageSlots> kv_residual_layers{};
    bool kv_residual_layers_explicit = false;
    bool quick                = false;
};

[[noreturn]] void usage_error(std::string_view message) {
    throw std::invalid_argument(std::string(message) +
                                "\nusage: ninfer-perplexity <model.ninfer> "
                                "(--corpus <manifest.json> [--quick] | --text <utf8-file>) "
                                "[--context N] [--stride N] [--device N] [--prefill-chunk N] "
                                "[--prefill-chunk-mode manual] "
                                "[--kv-dtype bf16|int8|fp8] [--kv-layer-storage SPEC] "
                                "[--kv-residual-layers SPEC] [--output <directory>]");
}

template <class Integer>
Integer parse_integer(std::string_view text, const char* label) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "usage: ninfer-perplexity <model.ninfer> "
                     "(--corpus <manifest.json> [--quick] | --text <utf8-file>)\n"
                     "       [--context N] [--stride N] [--device N] [--prefill-chunk N]\n"
                     "       [--prefill-chunk-mode manual]\n"
                     "       [--kv-dtype bf16|int8|fp8] [--kv-layer-storage SPEC]\n"
                     "       [--kv-residual-layers SPEC]\n"
                     "       [--output <directory>]\n";
        std::exit(0);
    }
    if (argc < 2 || std::string_view(argv[1]).starts_with("--")) {
        usage_error("artifact path is required");
    }
    Options out;
    out.artifact = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&](const char* label) -> std::string_view {
            if (++i >= argc) { usage_error(std::string(label) + " requires a value"); }
            return argv[i];
        };
        if (option == "--corpus") {
            out.corpus = std::filesystem::path(value("--corpus"));
        } else if (option == "--text") {
            out.text = std::filesystem::path(value("--text"));
        } else if (option == "--quick") {
            out.quick = true;
        } else if (option == "--context") {
            out.context = parse_integer<std::uint32_t>(value("--context"), "context");
        } else if (option == "--stride") {
            out.stride = parse_integer<std::uint32_t>(value("--stride"), "stride");
        } else if (option == "--device") {
            out.device = parse_integer<int>(value("--device"), "device");
        } else if (option == "--prefill-chunk") {
            out.prefill_chunk =
                parse_integer<std::uint32_t>(value("--prefill-chunk"), "prefill chunk");
        } else if (option == "--prefill-chunk-mode") {
            // The same two spellings the other two front ends take, from the same place. `dynamic`
            // is accepted HERE and refused by the engine, so the reason it cannot run on this path
            // is stated once, next to the code that knows why, instead of being re-argued by every
            // front end that would otherwise have to re-derive it.
            out.prefill_chunk_mode =
                ninfer::runtime::BandwidthGovernor::parse_mode(value("--prefill-chunk-mode"));
        } else if (option == "--kv-dtype") {
            const std::string_view dtype = value("--kv-dtype");
            out.kv_cache_explicit         = true;
            if (dtype == "bf16") {
                out.kv = ninfer::KvCacheStorage::BFloat16;
            } else if (dtype == "int8") {
                out.kv = ninfer::KvCacheStorage::Int8Group64;
            } else if (dtype == "fp8") {
                out.kv = ninfer::KvCacheStorage::Fp8E4M3Row256;
            } else if (dtype == "nvfp4") {
                out.kv = ninfer::KvCacheStorage::Nvfp4Group16;
            } else {
                usage_error("--kv-dtype must be bf16, int8, fp8, or nvfp4");
            }
        } else if (option == "--kv-layer-storage") {
            // Table AND mask (see apps/cli/main.cpp): a spec that spells `bf16` on a
            // layer means BF16 on that layer, not "inherit --kv-dtype".
            const auto parsed =
                ninfer::product::parse_kv_layer_storage_spec(value("--kv-layer-storage"));
            out.kv_layer_storage          = parsed.table;
            out.kv_layer_storage_set      = parsed.set;
            out.kv_layer_storage_explicit = true;
        } else if (option == "--kv-residual-layers") {
            // The per-layer NVFP4 residual planes (4 -> 8 planes on a layer). Table plus an
            // explicit bit, because the planner reads the table ONLY when the bit is set
            // (layouts_impl.h make_sequence_planner_impl): an all-false table handed over
            // unconditionally would read as "explicitly no residuals" on every other run.
            // Shared grammar and shared error text with the other two front ends.
            const auto parsed = ninfer::product::parse_kv_residual_layers_spec(
                value("--kv-residual-layers"));
            out.kv_residual_layers          = parsed.table;
            out.kv_residual_layers_explicit = true;
        } else if (option == "--output") {
            out.output = std::filesystem::path(value("--output"));
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }
    if (out.corpus.has_value() == out.text.has_value()) {
        usage_error("exactly one of --corpus and --text is required");
    }
    if (out.quick && !out.corpus) { usage_error("--quick requires --corpus"); }
    if (out.context < 2 || out.stride == 0 || out.stride >= out.context) {
        usage_error("context/stride must satisfy context>=2 and 1<=stride<context");
    }
    return out;
}

// Every KvCacheStorage enumerator is named here, and the throw sits AFTER the
// switch. That is deliberate: this function feeds the report's "kv_dtype" field
// AND the output directory name, so a tier that reaches it unnamed does not lose
// a label, it loses the whole run. It used to name 4 of 8 and throw for nvfp4,
// which is one of the four spellings --kv-dtype ITSELF accepts (parse_options,
// below): `--kv-dtype nvfp4` parsed, loaded the artifact, scored the corpus and
// then died at the first kv_name call (prepare_output_directory). A name table
// shorter than its enum is a runtime failure with a compile-time warning next to
// it, and the warning is the only thing that was ever going to catch it, so it
// must be silent. Names follow core/device_capabilities.h kv_storage_name.
std::string kv_name(ninfer::KvCacheStorage value) {
    switch (value) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8-g64";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-r256";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4-g16";
    case ninfer::KvCacheStorage::Fp8Group16:
        return "fp8-g16";
    case ninfer::KvCacheStorage::Iso3Group16:
        return "iso4e-g16";
    case ninfer::KvCacheStorage::E8Group64:
        return "rk4v4-g64";
    case ninfer::KvCacheStorage::Dropped:
        // L26 instrument: a layer with no KV planes (see KvCacheStorage::Dropped).
        return "dropped";
    }
    throw std::logic_error("unknown KV dtype");
}

std::string safe_component(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value) {
        out.push_back(std::isalnum(c) || c == '-' || c == '_' || c == '.' ? static_cast<char>(c)
                                                                          : '-');
    }
    return out.empty() ? "unknown" : out;
}

std::string timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::ostringstream out;
    out << std::put_time(&utc, "%Y%m%d-%H%M%S");
    return out.str();
}

std::filesystem::path prepare_output_directory(const Options& options,
                                               const ninfer::LoadSummary& load,
                                               const CorpusSelection& corpus) {
    std::filesystem::path output = options.output.value_or(
        std::filesystem::path("profiles/perplexity") / safe_component(load.model_id) /
        safe_component(load.weights_id) / kv_name(options.kv) / safe_component(corpus.corpus_id) /
        safe_component(corpus.mode) / timestamp());
    if (std::filesystem::exists(output)) {
        if (!std::filesystem::is_directory(output) ||
            std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator()) {
            throw std::runtime_error("output directory exists and is not empty: " +
                                     output.string());
        }
    } else if (!std::filesystem::create_directories(output)) {
        throw std::runtime_error("cannot create output directory: " + output.string());
    }
    return std::filesystem::absolute(output).lexically_normal();
}

double seconds_since(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

json aggregate_json(const ScoreAggregate& value) {
    return json{{"scored_tokens", value.scored_tokens},
                {"total_nll", value.total_nll},
                {"mean_nll", value.mean_nll()},
                {"perplexity", value.ppl()}};
}

struct EvaluationStream {
    ninfer::perplexity::CorpusStream source;
    std::vector<ninfer::TokenId> tokens;
    std::vector<WindowPlan> windows;
};

int run(const Options& options) {
    const Clock::time_point total_started = Clock::now();
    std::cerr << "[ppl] loading artifact " << options.artifact << '\n';
    std::string load_phase;
    std::uint64_t load_bucket = std::numeric_limits<std::uint64_t>::max();
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path          = options.artifact;
    engine_options.purpose                = ninfer::EnginePurpose::CausalScoring;
    engine_options.device                 = options.device;
    engine_options.max_context            = options.context;
    engine_options.kv_cache               = options.kv;
    engine_options.kv_cache_explicit      = options.kv_cache_explicit;
    engine_options.kv_layer_storage      = options.kv_layer_storage;
    engine_options.kv_layer_storage_set  = options.kv_layer_storage_set;
    engine_options.kv_layer_storage_explicit = options.kv_layer_storage_explicit;
    engine_options.kv_residual_layers        = options.kv_residual_layers;
    engine_options.kv_residual_explicit      = options.kv_residual_layers_explicit;
    if (options.prefill_chunk.has_value()) { engine_options.prefill_chunk = *options.prefill_chunk; }
    // Unset stays unset: the engine resolves the mode (and refuses `dynamic` on this path, which
    // has no governor). Copying a value here would only move the refusal to this file.
    engine_options.prefill_chunk_mode = options.prefill_chunk_mode;
    engine_options.load_progress.callback = [&](std::string_view phase, std::uint64_t done,
                                                std::uint64_t total) {
        const std::uint64_t bucket =
            total == 0 ? 0 : std::min<std::uint64_t>(10, 10 * done / total);
        if (phase != load_phase || bucket != load_bucket) {
            load_phase  = phase;
            load_bucket = bucket;
            std::cerr << "[ppl] load " << phase;
            if (total != 0) { std::cerr << ' ' << (10 * bucket) << '%'; }
            std::cerr << '\n';
        }
    };
    ninfer::Engine engine(std::move(engine_options));
    const ninfer::LoadSummary load = engine.load_summary();
    std::cerr << "[ppl] artifact ready in " << std::fixed << std::setprecision(2)
              << load.load_seconds << "s\n";

    // The `summary` block, printed BEFORE any scoring so that a run which dies mid-corpus
    // still leaves the instrument reading behind. Until this existed the scoring front end
    // emitted NO byte column at all (`grep -c summary` on a ppl stderr was 0 while the
    // generation front end's was 40), so any "perplexity x payload" table had to fall back
    // to the device-total peak -- +/-5 MiB of noise, against a quantity whose interesting
    // steps are 34 MiB. Same vocabulary, same writer and same column widths as
    // apps/cli/main.cpp, through product/kv_summary_format.h, so ONE regex reads either app.
    //
    // Every value here is the ENGINE's own reflection of the plan it built (MemorySummary),
    // not a re-derivation from argv: --kv-dtype / --kv-layer-storage / --kv-bit-budget /
    // --kv-bits all resolve into one per-layer table inside the planner, and only the engine
    // knows which spelling won. `kv drop layers` is read off that table (a discarded layer
    // is reflected as KvCacheStorage::Dropped) and never off NINFER_KV_DROP_LAYERS, whose
    // parser has already been caught reading "0-15" as layer 0 and "abc" as layer 0.
    const ninfer::MemorySummary memory = engine.memory_summary();
    ninfer::product::print_summary_metric(std::cerr, "max context",
                                         std::to_string(memory.max_context));
    ninfer::product::print_summary_metric(std::cerr, "KV capacity",
                                         std::to_string(memory.kv_capacity));
    ninfer::product::print_summary_metric(
        std::cerr, "full attention layers",
        std::to_string(memory.kv_full_attention_layers));
    ninfer::product::print_summary_metric(
        std::cerr, "kv cache dtype", ninfer::product::format_kv_layer_store(memory));
    ninfer::product::print_summary_metric(
        std::cerr, "kv cache payload",
        ninfer::product::format_kv_bytes(memory.kv_payload_bytes));
    ninfer::product::print_summary_metric(
        std::cerr, "kv drop layers", ninfer::product::format_kv_dropped_layers(memory));
    ninfer::product::print_summary_metric(
        std::cerr, "kv residual layers",
        options.kv_residual_layers_explicit
            ? ninfer::product::format_layer_set(options.kv_residual_layers,
                                               memory.kv_full_attention_layers)
            : std::string("-"));

    const Clock::time_point preflight_started = Clock::now();
    std::cerr << "[ppl] corpus preflight started\n";
    CorpusSelection corpus = options.corpus
                                 ? ninfer::perplexity::load_corpus(*options.corpus, options.quick)
                                 : ninfer::perplexity::load_custom_text(*options.text);
    std::vector<EvaluationStream> streams;
    streams.reserve(corpus.streams.size());
    std::uint64_t total_scored_tokens = 0;
    std::uint64_t total_input_tokens  = 0;
    std::uint64_t total_windows       = 0;
    for (auto& source : corpus.streams) {
        std::vector<ninfer::TokenId> tokens = engine.tokenize_text(source.text);
        if (tokens.size() < 2) {
            throw std::runtime_error("stream tokenized to fewer than two tokens: " + source.id);
        }
        std::vector<WindowPlan> windows =
            ninfer::perplexity::plan_windows(tokens.size(), options.context, options.stride);
        total_input_tokens += static_cast<std::uint64_t>(tokens.size());
        total_scored_tokens += static_cast<std::uint64_t>(tokens.size() - 1);
        total_windows += static_cast<std::uint64_t>(windows.size());
        streams.push_back(EvaluationStream{.source  = std::move(source),
                                           .tokens  = std::move(tokens),
                                           .windows = std::move(windows)});
    }
    const double preflight_seconds = seconds_since(preflight_started);
    std::cerr << "[ppl] corpus ready streams=" << streams.size()
              << " input_tokens=" << total_input_tokens << " scored_tokens=" << total_scored_tokens
              << " windows=" << total_windows << " in " << std::setprecision(2) << preflight_seconds
              << "s\n";

    const std::filesystem::path output_directory = prepare_output_directory(options, load, corpus);
    const Clock::time_point scoring_started      = Clock::now();
    Clock::time_point next_progress              = scoring_started + std::chrono::seconds(10);
    ScoreAggregate overall;
    std::map<std::string, ScoreAggregate> domains;
    json stream_reports             = json::array();
    std::uint64_t completed_windows = 0;

    for (std::size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        EvaluationStream& stream = streams[stream_index];
        std::cerr << "[ppl] stream " << (stream_index + 1) << '/' << streams.size() << ' '
                  << stream.source.id << " tokens=" << stream.tokens.size()
                  << " windows=" << stream.windows.size() << '\n';
        const Clock::time_point stream_started = Clock::now();
        ScoreAggregate stream_score;
        json window_reports = json::array();
        for (std::size_t window_index = 0; window_index < stream.windows.size(); ++window_index) {
            const WindowPlan& window = stream.windows[window_index];
            std::vector<ninfer::TokenId> input(
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_begin),
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_end));
            const Clock::time_point window_started = Clock::now();
            std::vector<float> logprobs;
            try {
                logprobs = engine.score_tokens(std::move(input), window.first_target);
            } catch (const std::exception& error) {
                throw std::runtime_error("scoring " + stream.source.id + " window " +
                                         std::to_string(window_index) + " failed: " + error.what());
            }
            const std::size_t expected = window.target_end - window.target_begin;
            if (logprobs.size() != expected) {
                throw std::runtime_error("scoring returned an invalid target count for " +
                                         stream.source.id);
            }
            ScoreAggregate window_score;
            window_score.add(logprobs);
            stream_score.add(window_score);
            overall.add(window_score);
            domains[stream.source.domain].add(window_score);
            ++completed_windows;
            json window_report            = aggregate_json(window_score);
            window_report["index"]        = window_index;
            window_report["input_begin"]  = window.input_begin;
            window_report["input_end"]    = window.input_end;
            window_report["target_begin"] = window.target_begin;
            window_report["target_end"]   = window.target_end;
            window_report["first_target"] = window.first_target;
            window_report["seconds"]      = seconds_since(window_started);
            window_reports.push_back(std::move(window_report));

            if (Clock::now() >= next_progress) {
                const double elapsed = seconds_since(scoring_started);
                const double rate    = static_cast<double>(overall.scored_tokens) / elapsed;
                const std::uint64_t remaining = total_scored_tokens - overall.scored_tokens;
                const double eta = rate > 0 ? static_cast<double>(remaining) / rate : 0.0;
                std::cerr << "[ppl] progress " << overall.scored_tokens << '/'
                          << total_scored_tokens << " tokens windows=" << completed_windows << '/'
                          << total_windows << " mean_nll=" << std::setprecision(5)
                          << overall.mean_nll() << " ppl=" << overall.ppl()
                          << " rate=" << std::setprecision(1) << rate
                          << " tok/s elapsed=" << std::setprecision(1) << elapsed << "s eta=" << eta
                          << "s\n";
                next_progress = Clock::now() + std::chrono::seconds(10);
            }
        }
        const double stream_seconds = seconds_since(stream_started);
        std::cerr << "[ppl] stream complete " << stream.source.id
                  << " scored=" << stream_score.scored_tokens
                  << " mean_nll=" << std::setprecision(5) << stream_score.mean_nll()
                  << " ppl=" << stream_score.ppl() << " in " << std::setprecision(2)
                  << stream_seconds << "s\n";
        json stream_report               = aggregate_json(stream_score);
        stream_report["id"]              = stream.source.id;
        stream_report["domain"]          = stream.source.domain;
        stream_report["path"]            = stream.source.path.string();
        stream_report["bytes"]           = stream.source.text.size();
        stream_report["sha256"]          = stream.source.sha256;
        stream_report["input_tokens"]    = stream.tokens.size();
        stream_report["unscored_tokens"] = 1;
        stream_report["seconds"]         = stream_seconds;
        stream_report["windows"]         = std::move(window_reports);
        stream_reports.push_back(std::move(stream_report));
    }

    const double scoring_seconds = seconds_since(scoring_started);
    json domain_reports          = json::array();
    for (const auto& [domain, aggregate] : domains) {
        json item      = aggregate_json(aggregate);
        item["domain"] = domain;
        domain_reports.push_back(std::move(item));
    }

    // The provenance block used to carry two literals, 1024 for both fields, while the engine
    // scored with 3072 (ninfer/types.h:214 through engine.cpp normalize_engine_options), so every
    // report.json on disk named a prefill unit that no run ever used. Reading the number back from
    // the Engine fixed the literal half; it was still half a fix, because the memory ladder in
    // targets/registry.cpp ("ninfer: reduced prefill chunk to N" on stderr) lowers the chunk that
    // the plan is finalized at, and that decision used to be invisible from here -- so a report
    // written after a reduction still named the requested chunk. The engine now adopts the settled
    // value into its own options (Engine::Impl's constructor) before any caller can read them, so
    // what is read here is the value the plan and the prefill loop were built at.
    // min(prefill_chunk, max_context) is the same quantity layouts_impl.h:1381 derives the plan's
    // chunk from.
    //
    // The MODE this number was reached in is manual by construction on this path
    // (engine.options().prefill_chunk_mode, which normalize_engine_options pins to Manual because a
    // CausalScoreCore has no Scheduler and no governor -- and which refuses `dynamic` by name). So
    // the number above is the score tile itself and not a ceiling some governor may still move: on
    // this path "manual" is the truthful description of the run, not an option the caller picked.
    const std::uint32_t effective_score_tile =
        std::min(engine.options().prefill_chunk, engine.options().max_context);

    json report{
        {"schema_version", 1},
        {"metric",
         {{"name", "fixed-window truncated-context causal perplexity"}, {"log_base", "natural"}}},
        {"artifact",
         {{"path", std::filesystem::absolute(options.artifact).lexically_normal().string()},
          {"target", load.target},
          {"model_id", load.model_id},
          {"weights_id", load.weights_id}}},
        {"corpus",
         {{"id", corpus.corpus_id},
          {"mode", corpus.mode},
          {"source", corpus.source.string()},
          {"stream_count", streams.size()}}},
        {"execution",
         {{"purpose", "causal_scoring"},
          {"device", options.device},
          {"context_tokens", options.context},
          {"stride_tokens", options.stride},
          // The KV store the run really used, from the engine's own plan. `kv_dtype` below
          // records the FLAG; these record the TABLE. Without them a report.json on disk
          // named a flag and left the deployed per-layer table, the discarded layer set, the
          // residual set and the payload reachable only by scraping stderr with a regex --
          // and a discarded layer used to be indistinguishable from "the flag defaulted".
          {"kv_layer_store", ninfer::product::format_kv_layer_store(memory)},
          {"kv_payload_bytes", memory.kv_payload_bytes},
          {"kv_full_attention_layers", memory.kv_full_attention_layers},
          {"kv_capacity_tokens", memory.kv_capacity},
          {"kv_drop_layers", ninfer::product::format_kv_dropped_layers(memory)},
          {"kv_drop_layers_requested_env", std::getenv("NINFER_KV_DROP_LAYERS") == nullptr
                                               ? ""
                                               : std::getenv("NINFER_KV_DROP_LAYERS")},
          {"kv_residual_layers",
           options.kv_residual_layers_explicit
               ? ninfer::product::format_layer_set(options.kv_residual_layers,
                                                   memory.kv_full_attention_layers)
               : std::string("-")},
          {"kv_residual_explicit", options.kv_residual_layers_explicit},
          {"prefill_chunk_tokens", effective_score_tile},
          {"score_tile_tokens", effective_score_tile},
          {"kv_dtype", kv_name(options.kv)}}},
        {"timing",
         {{"load_seconds", load.load_seconds},
          {"read_and_tokenize_seconds", preflight_seconds},
          {"score_seconds", scoring_seconds},
          {"total_seconds", seconds_since(total_started)},
          {"scored_tokens_per_second",
           static_cast<double>(overall.scored_tokens) / scoring_seconds}}},
        {"streams", std::move(stream_reports)},
        {"domains", std::move(domain_reports)},
        {"overall", aggregate_json(overall)},
    };

    const std::filesystem::path temporary = output_directory / "report.json.tmp";
    const std::filesystem::path final     = output_directory / "report.json";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("cannot create report: " + temporary.string()); }
        output << std::setw(2) << report << '\n';
        output.flush();
        if (!output) { throw std::runtime_error("cannot write report: " + temporary.string()); }
    }
    std::filesystem::rename(temporary, final);

    std::cout << "Perplexity result\n"
              << "artifact: " << load.model_id << " / " << load.weights_id << '\n'
              << "kv: " << kv_name(options.kv) << ", corpus: " << corpus.corpus_id << " / "
              << corpus.mode << ", context/stride: " << options.context << '/' << options.stride
              << "\n\n";
    std::cout << std::left << std::setw(24) << "domain" << std::right << std::setw(16) << "tokens"
              << std::setw(16) << "mean_nll" << std::setw(16) << "ppl" << '\n';
    for (const auto& [domain, aggregate] : domains) {
        std::cout << std::left << std::setw(24) << domain << std::right << std::setw(16)
                  << aggregate.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
                  << aggregate.mean_nll() << std::setw(16) << aggregate.ppl() << '\n';
    }
    std::cout << std::left << std::setw(24) << "overall" << std::right << std::setw(16)
              << overall.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
              << overall.mean_nll() << std::setw(16) << overall.ppl() << "\n\n"
              << "score rate: " << std::setprecision(1)
              << static_cast<double>(overall.scored_tokens) / scoring_seconds << " tok/s\n"
              << "report: " << final << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer-perplexity: " << error.what() << '\n';
        return 1;
    }
}
