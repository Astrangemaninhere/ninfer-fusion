#include "product/kv_kv_bits.h"
#include "product/load_progress/load_progress.h"
#include "serve/console_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/kv_auto_relayout.h"
#include "serve/kv_vram_probe.h"
#include "serve/serve_options.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

std::string format_bytes(std::size_t bytes) {
    constexpr double kMiB = 1024.0 * 1024.0;
    constexpr double kGiB = 1024.0 * kMiB;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (static_cast<double>(bytes) >= kGiB) {
        out << static_cast<double>(bytes) / kGiB << " GiB";
    } else {
        out << static_cast<double>(bytes) / kMiB << " MiB";
    }
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error, exception.what());
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error, exception.what());
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }
    // --kv-score-table: the penalty table's OWN entry point (product/kv_kv_bits.h
    // kv_score_table_run, the one implementation the CLI calls too). It needs no model,
    // so it is handled before the server binds anything; with an artifact path given as
    // well the server starts normally after the table is written.
    if (options.kv_score_table_explicit) {
        std::string error;
        if (!ninfer::product::kv_score_table_run(options.kv_score_table_spec,
                                                 options.kv_tier_scores, &error, std::cout,
                                                 std::cerr)) {
            std::cerr << error << "\n";
            return 2;
        }
        if (options.artifact_path.empty()) { return 0; }
    }

    try {
        using Clock = std::chrono::steady_clock;
        ninfer::serve::HttpServer server(options);
        if (!server.bind()) {
            ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error,
                                             "failed to bind " + options.host + ':' +
                                                 std::to_string(options.port));
            return 1;
        }
        // def3: START ACCEPTING BEFORE LOADING. bind() only binds the socket (cpp-httplib does
        // not listen until listen_after_bind()), and the accept loop used to start after the
        // whole launch, so a probe read http=000 for the entire load window and could not tell
        // "still loading" from "crashed". The loop starts here; /health answers a named
        // "loading" state and every other endpoint a 503 until attach() below.
        server.listen_in_background();

        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, "loading model...");
        auto load_progress_options        = ninfer::product::stderr_load_progress_options();
        load_progress_options.line_prefix = [] {
            return ninfer::serve::current_console_log_prefix(ninfer::serve::ConsoleLogLevel::Info);
        };
        ninfer::product::LoadProgressRenderer load_progress(std::cerr,
                                                            std::move(load_progress_options));
        const auto load_start = Clock::now();
        ninfer::serve::GenerationService service(options, load_progress.callback());
        std::ostringstream loaded;
        loaded << "model loaded in "
               << std::chrono::duration<double>(Clock::now() - load_start).count() << " s";
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, loaded.str());

        // FreeToken step 2: periodic KV relayout from the ft energy table.
        // The line is opt-in end to end and the precedence is CLI > env >
        // default everywhere: --kv-auto-relayout SECS beats
        // NINFER_FT_RELOAD_SECS beats the 0 (disabled) default, and
        // --ft-vram-axis on|off beats NINFER_FT_VRAM_AXIS. --ft-stats on is the
        // matching observation switch (NINFER_FT_STATS); without observations
        // the loop has nothing to decide on. Deep layers keep nvfp4; the rest
        // follow the energy tertiles with two-cycle hysteresis.
        auto relayout_config = ninfer::serve::KvAutoRelayout::from_env(
            ninfer::serve::KvAutoRelayout::CliOverrides{
                .interval_secs     = options.ft_relayout_secs,
                .vram_axis_enabled = options.ft_vram_axis,
            });
        relayout_config.current_table          = options.kv_layer_storage;
        relayout_config.current_table_explicit = options.kv_layer_storage_explicit;
        // Gap 1 probe: free_vram_bytes() is the one TU that knows about
        // cudaMemGetInfo (serve/kv_vram_probe.cpp), which is why this TU does
        // not have to -- and why kv_auto_relayout.{h,cpp} stay CUDA-free.
        relayout_config.free_vram_bytes = &ninfer::serve::free_vram_bytes;

        const ninfer::MemorySummary memory            = service.memory_summary();
        const ninfer::ContextCostSummary context_cost = service.load_summary().context_cost;
        const ninfer::EngineOptions& engine           = service.engine_options();
        const ninfer::ContextCacheOptions& cache      = engine.context_cache;
        std::ostringstream capacity;
        capacity << "KV capacity "
                 << (memory.kv_capacity_mode == ninfer::KvCapacityMode::Automatic ? "auto"
                                                                                  : "explicit")
                 << " resolved=" << memory.kv_capacity
                 << " tokens pages=" << memory.kv_capacity_page_groups << '/'
                 << memory.kv_capacity_max_page_groups
                 << " runtime=" << format_bytes(memory.runtime_reservation_bytes)
                 << " free-after-weights=" << format_bytes(memory.available_after_weights_bytes)
                 << " free-after-startup=" << format_bytes(memory.available_after_startup_bytes)
                 << " headroom=" << format_bytes(memory.kv_capacity_headroom_bytes)
                 << " slack=" << format_bytes(memory.planned_slack_bytes)
                 << " graph-allowance=" << format_bytes(memory.cuda_graph_allowance_bytes)
                 << " context-cache=" << (cache.enabled ? "on" : "root-only")
                 << " device-state=" << *cache.device_state_slots << "-cache+"
                 << engine.max_concurrency << "-active" << " host-state=" << cache.host_state_slots
                 << " host-kv=" << format_bytes(cache.host_kv_capacity_bytes)
                 << " private=" << *cache.max_private_continuations
                 << " shared=" << *cache.max_shared_prefixes
                 << " anchors=" << *cache.max_long_anchors_per_continuation;
        capacity << " context-cost-transfer="
                 << ninfer::context_cost_preset_source_name(context_cost.transfer_source)
                 << " context-cost-prefill="
                 << ninfer::context_cost_preset_source_name(context_cost.prefill_source)
                 << " cost-profile=" << context_cost.hardware_class << '/' << context_cost.model_id
                 << '/' << context_cost.weights_id;
        if (options.enable_vision) {
            const ninfer::MediaCacheSummary media = service.media_cache_summary();
            capacity << " media-workers=" << media.preprocess_threads
                     << " media-cache=" << format_bytes(media.capacity_bytes)
                     << " media-live=" << format_bytes(media.live_capacity_bytes);
        }
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, capacity.str());

        // Gap 1 bridge: one full-attention layer's device KV payload at the
        // resolved capacity, which is the quantum the free-VRAM axis moves in.
        // kv_payload_bytes is the layout's own payload (targets/qwen3_6/impl/
        // runtime/layouts.h:31), so its mean over the full-attention layers is
        // that quantum; a zero (no payload, no layer count) leaves the axis
        // inert. The axis needs no geometry of its own because of this line.
        if (relayout_config.full_attn_layers > 0) {
            relayout_config.vram_layer_bytes =
                static_cast<std::uint64_t>(memory.kv_payload_bytes) /
                static_cast<std::uint64_t>(relayout_config.full_attn_layers);
        }

        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, "warming up...");
        service.warmup();

        // Gap 1 baseline: the free VRAM the axis is relative to, sampled AFTER
        // the warmup on purpose. Warmup is what allocates the decode working set
        // and captures the CUDA graphs (MemorySummary::cuda_graph_allowance_bytes),
        // so a pre-warmup sample would read those bytes back as a permanent
        // deficit and pin the rk4v4 band at its cap on every machine. A 0 reading
        // disarms the axis, which leaves the pre-gap-1 policy in place.
        relayout_config.vram_reference_bytes = relayout_config.free_vram_bytes();

        ninfer::serve::KvAutoRelayout auto_relayout(
            std::move(relayout_config), [&service](std::string_view spec) {
                try {
                    service.reload_kv_storage(spec);
                    return true;
                } catch (const std::exception& error) {
                    ninfer::serve::write_console_log(
                        ninfer::serve::ConsoleLogLevel::Warning,
                        std::string("ft auto-relayout rejected: ") + error.what());
                    return false;
                }
            });
        auto_relayout.start();

        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        std::ostringstream listening;
        listening << "listening on http://" << options.host << ':' << options.port
                  << " (model id: " << server.public_model_id()
                  << ", auth: " << (options.api_key.empty() ? "disabled" : "bearer") << ')';
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, listening.str());

        // def3: the accept loop has been running since before the load; this only waits for it
        // (and stops the stats reporter).
        const bool ok = server.wait_for_exit();
        g_server.store(nullptr);
        if (!ok) {
            ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error,
                                             "failed to bind " + options.host + ':' +
                                                 std::to_string(options.port));
            return 1;
        }
        return 0;
    } catch (const std::exception& exception) {
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error, exception.what());
        return 1;
    }
}
