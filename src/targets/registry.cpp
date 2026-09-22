#include "targets/registry.h"

#include <ninfer/targets/qwen4_exp/package.h>
#include <ninfer/targets/gemma4_31b/package.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arch_caps.h"
#include "core/arch_sim.h"
#include "core/device.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/context_cost.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

    // An artifact whose identity has no prefill row in the compiled defaults is priced with the
    // generic prefill model; say so at load time instead of degrading silently.
    if (const std::string note = runtime::context_prefill_preset_miss_note(context_cost_identity);
        !note.empty()) {
        std::fprintf(stderr, "ninfer: %s\n", note.c_str());
    }

    artifact::Binder binder(reader);
    // W13: the weight offload decision has to be made before the target's load
    // plan fixes device_capacity_bytes, so it rides the binder. With the default
    // 0 budget this is a no-op and the plan carries an empty offload field.
    binder.set_weight_offload_limits(product::WeightOffloadLimits{
        .host_pinned_bytes  = resolved_options.weight_host_offload_bytes,
        .device_arena_bytes = resolved_options.weight_device_arena_bytes,
        .prefetch_layers    = resolved_options.weight_prefetch_layers,
        .min_span_bytes     = resolved_options.weight_span_floor_bytes,
    });
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
                             .context_cost      = std::move(context_cost.model),
                             // The chunk the ladders above actually settled on. Emitting it here (and
                             // adopting it in Engine::Impl) is what makes the decision reachable
                             // outside this function: the reduce message above is the only other
                             // trace of it, and a reader that reports EngineOptions::prefill_chunk
                             // without this writes the *request*.
                             .effective_prefill_chunk = active_chunk};
}

// ---------------------------------------------------------------------------
// Data-driven target registration table: one row per family x detection key.
//
// Columns:
//   family     - the closed registry's operator-facing family label. It is also
//                the compile-time tag budget owner: a row whose family already
//                has a Loaded/Instance pair compiled in needs no new translation
//                unit and no variant change. Only a *new family* additionally
//                needs a src/targets/<dir> entry in src/CMakeLists.txt (the
//                add_subdirectory enumeration) plus one more std::variant
//                alternative in registry.h.
//   model_id   - detection condition: the ArtifactIdentity::model_id this row
//                claims. Matched by exact string equality, first row wins.
//   target_key - reported verbatim as LoadSummary::target (serve request log,
//                CLI "target" metric). Kept per model_id, not per family, so
//                one family can report two distinct keys.
//   entry      - variant entry: the closed construct_registered<Family, Loaded,
//                Instance> instantiation to run. The compile-time enum tag is
//                carried in the template arguments themselves, so a row needs no
//                runtime family switch; taking its address odr-uses exactly the
//                same specializations the previous if chain did.
//
// Adding a model whose family is already in this table = add one row here.
// Print the compiled-in table with NINFER_REGISTRY_DUMP=1 (goes to stderr).
// ---------------------------------------------------------------------------
using TargetEntry = ConstructedTarget (*)(const EngineOptions&, DeviceContext&,
                                          artifact::Reader&, Clock::time_point,
                                          std::string_view);

// A row carries the package's own declared-capability predicate, so the registry asks
// the target what it accepts instead of comparing model ids of its own: "registered"
// then cannot drift away from "accepted".
using DeclaresModel = bool (*)(std::string_view) noexcept;

struct TargetRegistration {
    std::string_view family;
    std::string_view model_id;
    std::string_view target_key;
    DeclaresModel declares_model;
    TargetEntry entry;
};

constexpr TargetRegistration kTargetRegistrations[] = {
    {"qwen3_6_27b", Qwen3_6_27B::model_id, Qwen3_6_27B::target_key,
     &Qwen3_6_27B::declares_model,
     &construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>},
    {"qwen3_6_27b", Qwen3_6_27B::qwen3_8_model_id, Qwen3_6_27B::qwen3_8_target_key,
     &Qwen3_6_27B::declares_model,
     &construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>},
    {"qwen3_6_35b_a3b", Qwen3_6_35BA3B::model_id, Qwen3_6_35BA3B::target_key,
     &Qwen3_6_35BA3B::declares_model,
     &construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>},
    {"muse_glimmer_30b", MuseGlimmer30B::model_id, MuseGlimmer30B::target_key,
     &MuseGlimmer30B::declares_model,
     &construct_registered<MuseGlimmer30B, LoadedMuseGlimmer30B, MuseGlimmer30BInstance>},
    {"qwen3_5_9b", Qwen3_5_9B::model_id, Qwen3_5_9B::target_key,
     &Qwen3_5_9B::declares_model,
     &construct_registered<Qwen3_5_9B, LoadedQwen3_5_9B, Qwen3_5_9BInstance>},
};

constexpr std::size_t kTargetRegistrationCount =
    sizeof(kTargetRegistrations) / sizeof(kTargetRegistrations[0]);

// Every model id the registered packages declare, deduplicated, for the rejection
// message below: a rejected artifact is told both sides of the comparison instead of
// only that it is unsupported.
std::string declared_model_ids() {
    std::string text;
    for (std::size_t i = 0; i < kTargetRegistrationCount; ++i) {
        bool already_listed = false;
        for (std::size_t j = 0; j < i; ++j) {
            if (kTargetRegistrations[j].model_id == kTargetRegistrations[i].model_id) {
                already_listed = true;
            }
        }
        if (already_listed) { continue; }
        if (!text.empty()) { text += ", "; }
        text.append(kTargetRegistrations[i].model_id);
    }
    return text;
}

// Compile-time table hygiene. The previous if chain asserted nothing; a table can
// silently misroute, so the checks that are decidable at compile time are taken
// here instead of being left to a smoke run. A duplicated model_id is the one
// defect that would make first-match-wins order-dependent, and a row referring to
// a family whose package is not in the build fails at link time (its
// construct_registered instantiation is odr-used), so the build stays the gate.
constexpr bool registrations_are_well_formed() {
    for (std::size_t i = 0; i < kTargetRegistrationCount; ++i) {
        const TargetRegistration& row = kTargetRegistrations[i];
        if (row.family.empty() || row.model_id.empty() || row.target_key.empty() ||
            row.declares_model == nullptr || row.entry == nullptr) {
            return false;
        }
        for (std::size_t j = i + 1; j < kTargetRegistrationCount; ++j) {
            if (row.model_id == kTargetRegistrations[j].model_id) { return false; }
        }
        // A route the package does not declare would be dead, and a package model id
        // with no row would be unreachable: both are decidable at compile time.
        if (!row.declares_model(row.model_id)) { return false; }
    }
    return true;
}
static_assert(registrations_are_well_formed(),
              "target registry row is empty or its detection model_id is duplicated; a duplicate "
              "would make first-match-wins routing depend on row order");

// Self-check surface for the table above. Env-gated the same way the existing
// NINFER_EXPORT_HEAD_DIR hook is, so no CLI flag is added and --help is
// unchanged. Diagnostics go to stderr: stdout stays the answer channel.
void dump_registrations() {
    std::fprintf(stderr, "ninfer: target registry: %zu rows\n", kTargetRegistrationCount);
    for (const TargetRegistration& row : kTargetRegistrations) {
        std::fprintf(stderr, "ninfer:   family=%.*s model_id=%.*s target_key=%.*s\n",
                     static_cast<int>(row.family.size()), row.family.data(),
                     static_cast<int>(row.model_id.size()), row.model_id.data(),
                     static_cast<int>(row.target_key.size()), row.target_key.data());
    }
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

LoadedQwen3_5_9B::LoadedQwen3_5_9B(std::unique_ptr<Qwen3_5_9B::LoadedModel> stable_model,
                                   const EngineOptions& options)
    : model(std::move(stable_model)),
      frontend(Qwen3_5_9B::make_frontend(*model, options)) {}

LoadedQwen3_5_9B::~LoadedQwen3_5_9B() = default;

Qwen3_5_9BInstance::Qwen3_5_9BInstance(std::unique_ptr<LoadedQwen3_5_9B> stable_loaded,
                                       runtime::KvCapacityResolution resolution,
                                       Qwen3_5_9B::SequencePlan sequence_plan,
                                       Qwen3_5_9B::WeightsProfile weights_profile_in,
                                       DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(
          Qwen3_5_9B::create_program(*loaded->model, std::move(sequence_plan), device)),
      weights_profile(weights_profile_in) {}

Qwen3_5_9BInstance::~Qwen3_5_9BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    if (std::getenv("NINFER_REGISTRY_DUMP") != nullptr) { dump_registrations(); }
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    // Per-architecture capability gate (src/core/arch_caps.h, and the route table in
    // tools/archkit/_GPU_MATRIX.md). An artifact's numeric formats fix its tensor-core floor
    // on their own, independently of which target package claims the identity, so the verdict
    // is taken here: after the identity is known, before a load plan exists, and before any
    // weight byte reaches the device. The only previous architecture statement was the blanket
    // reject in src/targets/qwen3_6/impl/runtime/layouts_impl.h:933, which cannot name the
    // format that is unsupported, the kernel that sets its floor, or the route that would work
    // on this card.
    // THE ENGINE'S ARCH VIEW, and the ONLY place the test-only override can reach a
    // decision the engine ACTS ON. arch_view_for_device() reads NINFER_SIM_ARCH and
    // NINFER_SIM_ARCH_ACK once per process and prints sim_banner() once when either is set
    // (src/core/arch_sim.h, G2); with neither set the view is Disabled, effective_sm ==
    // device.sm(), and this call is the int overload's behaviour exactly. When the request
    // was REFUSED the overload throws instead of answering off the physical device (G3).
    //
    // MEASURED BEFORE THIS EDIT (SIM-RUN, 2026-09-18): this TU was compiled with -DNDEBUG as
    // its ONLY -D while five sibling ninfer_artifact TUs got -DNINFER_HAVE_QPN=1, so the
    // engine's gate believed the fallback kernel was in no target while it was linked in.
    // See the PUBLIC link in src/CMakeLists.txt for that half.
    const caps::ArchView arch_view = caps::arch_view_for_device(device.sm());
    caps::require_artifact_formats_supported(arch_view, caps::artifact_formats(reader.objects()),
                                            identity.model_id + "/" + identity.weights_id);
    // Table lookup replaces the previous four-branch if chain. Row order is kept
    // identical to the chain, but the routes are disjoint by construction (the
    // static_assert above rejects a duplicated model_id), so first-match-wins is
    // order-independent.
    for (const TargetRegistration& registration : kTargetRegistrations) {
        if (registration.declares_model(identity.model_id)) {
            return registration.entry(options, device, reader, load_start,
                                      registration.target_key);
        }
    }
    // S37 stage (a): the identity is recognized but its runtime does not exist yet.
    // Cross-check the spec-derived config against the artifact's tensor shapes (the
    // manifest carries no semantic geometry, so shapes are its only geometry source),
    // then refuse loudly carrying the verdict. A stub target must never load
    // silently, and a wrong artifact must fail naming both sides of the mismatch
    // instead of failing inside a kernel later.
    if (identity.model_id == qwen4_exp::kModelId) {
        const std::string geometry_verdict = qwen4_exp::validate_stage_a_geometry(reader);
        throw std::runtime_error(
            "artifact identity '" + identity.model_id + "' recognized (" + geometry_verdict +
            "), but its runtime is not implemented yet (W7/P1 staged plan, "
            "_collab/B_s37_flashnext_p1.md); refusing to load a stub target");
    }

    // Gemma-4-31B (model_type gemma4). Same shape as the qwen4_exp branch above, and for the
    // same reason: the identity is recognized so an artifact of this family is never reported as
    // an unknown model, and the refusal carries the measured state of the port instead of a bare
    // "no registered target for this device". There is no kTargetRegistrations row because
    // construct_registered<Target, Loaded, Instance> needs an Instance adapter this target does
    // not have (src/CMakeLists.txt:626-632 records the same conclusion for qwen3_8_flash_next),
    // and not adding a row is what makes this branch provably inert for every other model id:
    // declared_model_ids() and the NINFER_REGISTRY_DUMP output are both derived from that table.
    //
    // The verdict is derived from the artifact's own tensor shapes, and the work it names is
    // pinned as static_asserts in src/targets/gemma4_31b/impl/config.h rather than as prose:
    // the ten full_attention layers are head_dim 512 / 4 KV heads, which
    // src/ops/kernel/gqa_attention_geometry.cuh:15 cannot instantiate (it admits 128 or 256),
    // and rotary_dim is per layer kind while the shared TextConfig has a single slot.
    if (identity.model_id == gemma4_31b::kModelId) {
        const std::string geometry_verdict = gemma4_31b::validate_stage_a_geometry(reader);
        throw std::runtime_error(
            "artifact identity '" + identity.model_id + "/" + identity.weights_id +
            "' recognized (" + geometry_verdict + "), but its runtime is not implemented yet; "
            "refusing to load a stub target. Open work: " + std::string(gemma4_31b::kRuntimeBlockers));
    }

    // The sibling spelling of the same model. igorls/ninfer registers this target as
    // qwen3.8-flash-next / qwen3_8_flash_next, while the abliterated NVFP4 checkpoint on this
    // box declares model_type "qwen4_exp" (and "qwen4_exp_text"). Recognize it so a FlashNext
    // artifact is never reported as an unknown model, and refuse with the measured state of the
    // port instead of a bare "no registered target for this device".
    //
    // MEASURED 2026-09-18, host -fsyntax-only over the tree's own compile_commands.json, no GPU:
    // the imported src/targets/qwen3_8_flash_next is now 24 of 24 host TUs clean. The earlier
    // text here ("10 of 24 host TUs clean ... blocked by six shared-layer symbols") is
    // superseded and was already stale when it was written. The target is still
    // EXCLUDE_FROM_ALL in src/CMakeLists.txt, but for a REGISTRATION reason rather than a
    // compilation one: the copied tree exports Package/plan_load/create_program and has no
    // engine-facing Instance adapter, which is the piece construct_registered<Target, Loaded,
    // Instance> needs, so no row can be written for it yet.
    // See scratch/PATCHSET/FNRUN/REPORT.md.
    constexpr std::string_view kFlashNextSiblingModelId = "qwen3.8-flash-next";
    if (identity.model_id == kFlashNextSiblingModelId) {
        const std::string geometry_verdict = qwen4_exp::validate_stage_a_geometry(reader);
        throw std::runtime_error(
            "artifact identity '" + identity.model_id + "' recognized (" + geometry_verdict +
            "); the qwen3_8_flash_next target compiles in this tree (24/24 host TUs) but has no "
            "engine-facing Instance adapter, so it cannot be registered and is refused rather "
            "than loading a stub target");

    }
        throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device; declared model ids: " +
                             declared_model_ids());
}

} // namespace ninfer::targets
