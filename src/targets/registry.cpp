#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/context_cost.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <type_traits>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback = progress.callback};
}

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument("model weights require " + std::to_string(weight_bytes) +
                                    " bytes of device memory, but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {
    const auto& identity                          = reader.identity();
    const auto weights_profile                    = Target::resolve_weights(identity);
    // Resolve --spec auto up front so the planner, the load plan and the
    // program all see the same concrete backend (previously only plan_load
    // resolved it, so the planner built an Auto plan and the frozen startup
    // features mismatch check rejected the loaded weights).
    const EngineOptions resolved_options =
        Target::resolved_auto_speculative(options, weights_profile);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);
    const runtime::ContextCostIdentity context_cost_identity{
        .hardware_class = runtime::context_cost_hardware_class(
            device.props.name, device.props.major, device.props.minor),
        .model_id   = identity.model_id,
        .weights_id = identity.weights_id,
    };
    runtime::ResolvedContextMachineCost context_cost = runtime::resolve_context_machine_cost(
        context_cost_identity, options.context_cost.preset_path);

    artifact::Binder binder(reader);
    auto load_plan = Target::plan_load(binder, resolved_options, weights_profile);
    const std::size_t preflight_runtime_bytes =
        runtime_bytes_after_planned_weights(load_plan.materialization().device_capacity_bytes);

    // Memory-adaptive prefill chunk: the runtime workspace reservation grows with
    // the prefill chunk, so when the device budget cannot fit the requested chunk
    // the layout is retried at successively smaller chunks before failing. The
    // finalized plan carries the reduced chunk into the prefill loop.
    const auto chunk_candidates = [](std::uint32_t requested) {
        std::vector<std::uint32_t> ladder{requested, 2048, 1536, 1024, 768, 512};
        std::sort(ladder.begin(), ladder.end(), std::greater<>());
        ladder.erase(std::unique(ladder.begin(), ladder.end()), ladder.end());
        return ladder;
    };
    const auto build_planner_at = [&](std::uint32_t chunk) {
        EngineOptions chunk_options = resolved_options;
        chunk_options.prefill_chunk = chunk;
        return Target::make_sequence_planner(device, chunk_options, weights_profile);
    };
    const auto resolve_with_fallback = [&](std::size_t budget, std::uint32_t start_chunk,
                                           std::uint32_t& chosen_chunk) {
        std::exception_ptr last_error;
        for (const std::uint32_t chunk : chunk_candidates(resolved_options.prefill_chunk)) {
            if (chunk > start_chunk) { continue; }
            auto planner = build_planner_at(chunk);
            try {
                auto resolution = runtime::resolve_kv_capacity(
                    resolved_options.kv_capacity, planner.capacity_curve(), budget);
                chosen_chunk = chunk;
                return resolution;
            } catch (const std::invalid_argument& error) {
                last_error = std::current_exception();
                (void)error;
            }
        }
        std::rethrow_exception(last_error);
    };

    std::uint32_t active_chunk = resolved_options.prefill_chunk;
    (void)resolve_with_fallback(preflight_runtime_bytes, resolved_options.prefill_chunk,
                                active_chunk);

    auto progress     = artifact_progress(resolved_options.load_progress);
    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                              progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    device.synchronize();
    if (const char* head_dir = std::getenv("NINFER_EXPORT_HEAD_DIR")) {
        Target::export_head_weights(*model, head_dir);
    }
    runtime::KvCapacityResolution capacity_resolution =
        resolve_with_fallback(current_free_device_bytes(), active_chunk, active_chunk);
    if (active_chunk != resolved_options.prefill_chunk) {
        std::fprintf(stderr,
                     "ninfer: reduced prefill chunk to %u (the requested %u does not fit the "
                     "device runtime budget)\n",
                     active_chunk, resolved_options.prefill_chunk);
    }
    auto sequence_planner = build_planner_at(active_chunk);
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    auto loaded   = std::make_unique<Loaded>(std::move(model), resolved_options);
    auto instance = std::make_unique<Instance>(std::move(loaded), capacity_resolution,
                                               std::move(sequence_plan), weights_profile, device);
    device.synchronize();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();

    LoadSummary summary;
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    summary.context_cost         = context_cost.summary;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults,
                             .context_cost      = std::move(context_cost.model)};
}

// Re-run only the sequence plan + program construction for an already-loaded target.
// Mirrors the cold-start flow (chunk ladder + capacity resolution + plan invariant check)
// but skips weight materialization: the instance keeps its loaded model and frontend.
template <class Target, class Instance>
void replan_instance_kv(Instance& instance, const EngineOptions& options, DeviceContext& device) {
    const auto weights_profile = instance.weights_profile;
    const EngineOptions resolved_options =
        Target::resolved_auto_speculative(options, weights_profile);
    const auto chunk_candidates = [](std::uint32_t requested) {
        std::vector<std::uint32_t> ladder{requested, 2048, 1536, 1024, 768, 512};
        std::sort(ladder.begin(), ladder.end(), std::greater<>());
        ladder.erase(std::unique(ladder.begin(), ladder.end()), ladder.end());
        return ladder;
    };
    std::exception_ptr last_error;
    for (const std::uint32_t chunk : chunk_candidates(resolved_options.prefill_chunk)) {
        if (chunk > resolved_options.prefill_chunk) { continue; }
        EngineOptions chunk_options = resolved_options;
        chunk_options.prefill_chunk = chunk;
        try {
            auto planner = Target::make_sequence_planner(device, chunk_options, weights_profile);
            auto resolution = runtime::resolve_kv_capacity(
                chunk_options.kv_capacity, planner.capacity_curve(), current_free_device_bytes());
            auto plan = std::move(planner).finalize(resolution.main_page_groups);
            if (plan.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
                plan.kv_capacity() != resolution.resolved_tokens) {
                throw std::logic_error("resolved KV capacity does not match the finalized target plan");
            }
            instance.kv_capacity_resolution = resolution;
            instance.program =
                Target::create_program(*instance.loaded->model, std::move(plan), device);
            return;
        } catch (const std::exception& error) {
            last_error = std::current_exception();
            (void)error;
        }
    }
    std::rethrow_exception(last_error);
}

} // namespace

void replan_target_kv(ActiveTarget& target, const EngineOptions& options, DeviceContext& device) {
    std::visit(
        [&](auto& instance_ptr) {
            using InstanceType = std::remove_cvref_t<decltype(*instance_ptr)>;
            if (instance_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            // Free the old KV pool first: the new plan's capacity resolution must see the
            // freed device budget. A throw past this point leaves the instance programless
            // (documented at the registry.h declaration).
            instance_ptr->program.reset();
            device.synchronize();
            replan_instance_kv<typename InstanceType::Package>(*instance_ptr, options, device);
        },
        target);
}

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         Qwen3_6_27B::WeightsProfile weights_profile_in,
                                         DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), device)),
      weights_profile(weights_profile_in) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               Qwen3_6_35BA3B::WeightsProfile weights_profile_in,
                                               DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan), device)),
      weights_profile(weights_profile_in) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;


LoadedMuseGlimmer30B::LoadedMuseGlimmer30B(
    std::unique_ptr<MuseGlimmer30B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(MuseGlimmer30B::make_frontend(*model, options)) {}

LoadedMuseGlimmer30B::~LoadedMuseGlimmer30B() = default;

MuseGlimmer30BInstance::MuseGlimmer30BInstance(
    std::unique_ptr<LoadedMuseGlimmer30B> stable_loaded,
    runtime::KvCapacityResolution resolution, MuseGlimmer30B::SequencePlan sequence_plan,
    MuseGlimmer30B::WeightsProfile weights_profile_in, DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(MuseGlimmer30B::create_program(*loaded->model, std::move(sequence_plan), device)),
      weights_profile(weights_profile_in) {}

MuseGlimmer30BInstance::~MuseGlimmer30BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, device, reader, load_start, Qwen3_6_35BA3B::target_key);
    }
    if (identity.model_id == MuseGlimmer30B::model_id) {
        return construct_registered<MuseGlimmer30B, LoadedMuseGlimmer30B, MuseGlimmer30BInstance>(
            options, device, reader, load_start, MuseGlimmer30B::target_key);
    }
        throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
