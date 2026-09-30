#include "targets/registry.h"

#include <ninfer/targets/qwen4_exp/package.h>
#include <ninfer/targets/gemma4_31b/package.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/amdsafe_gate.h"
#include "core/arch_caps.h"
#include "core/arch_sim.h"
#include "core/device.h"
#include "core/vendor_sim.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/context_cost.h"

// The engine-side reader of EngineOptions::ple_sidecar_root. Declared CUDA-free by
// impl/ple_attach.h and defined by impl/ple_session.cpp; before this include no translation
// unit in ninfer_engine read that option at all. See the pre-flight in construct_target().
#include "targets/qwen4_exp/impl/ple_attach.h"

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
        throw std::invalid_argument("Engine max_concurrency must be in [1,16]");
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
        // W13: DERIVED per run, not the field flipped to `true`. The field asserts the
        // residency contract stated in weight_residency.h:374-383 -- every pass over the
        // layer order enters every offloaded layer through note_layer() -- and the engine
        // proves that assertion for THIS purpose alone:
        // normalize_engine_options' CausalScoring case (src/runtime/engine/engine.cpp:79-82)
        // forces speculative={} / enable_vision=false / use_cuda_graph=false /
        // context_cache={enabled=false}, and ProgramImplCore::causal_score re-asserts all
        // four before it scores (program_impl.h:2015-2018). A causal score is teacher
        // forced, not autoregressive, so the only run_layers call site it can reach is the
        // Phase::Prefill one that carries the hook (text_context_impl.h:1506-1511); the two
        // Phase::Verify sites (text_context_impl.h:1015, :1078) have no hook and are
        // unreachable by construction. For Generation the predicate is FALSE and the
        // refusal stands -- that is the arm which measured 64 ids that were all 0
        // (weight_residency.h:583-588), so a generation run cannot reach this plan.
        // (notehook) DERIVED FROM THE GRAPH DECISION, not from the purpose. The residency hook
        // now runs in every phase that is not CUDA-graph captured (text_context_impl.h
        // run_layers) and the H2D is joined to the consuming stream by an event, so the
        // contract holds exactly when the layer walk is never captured -- and `use_cuda_graph`
        // is the ONE flag every capture site reads (decode_impl.h:85, mtp_impl.h:420,
        // dflash_impl.h:615). The old predicate (`purpose == CausalScoring`) was true only
        // because a causal score forces use_cuda_graph=false (src/runtime/engine/engine.cpp:81),
        // so it said nothing about the verify sites, and under it a Generation run could not
        // reach this plan at all. BOTH halves are now stated, so the rotation the arena
        // performs and the capture the run performs cannot disagree in silence.
        // F1059: TRUE UNCONDITIONALLY, INCLUDING UNDER CAPTURE. This field is the caller's
        // assertion that the run enters every offloaded layer through note_layer(); run_layers
        // now does so in every phase, capture included, because the fetch is enrolled into the
        // capture instead of being a host call outside it. The old
        // `!resolved_options.use_cuda_graph` spelled the same thing only because a capture used
        // to make the hook impossible -- and under it a graph-on run was refused for the wrong
        // reason. What still guards the unsafe case is WeightOffloadLimits::decode_graph_captured,
        // which refuses a rotating arena under capture unless the capturable fetch was asked for
        // by name (NINFER_W13_CAPTURABLE_FETCH=1).
        .fetch_per_layer_entry = true,
        .decode_graph_captured = resolved_options.use_cuda_graph,
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
    // Spark-X2.5-4B.  The row the WORK_ITEMS text named twice: once under "C++ target"
    // ("plus registry.h / registry.cpp (:135-141)") and once by the artifact's own
    // absence ("an artifact written now could not be loaded by any engine build").
    // The family label is the artifact directory's name, the model_id is the HF-side
    // spelling the artifact's ArtifactIdentity carries, and target_key is the C++/on-disk
    // spelling -- the three-string separation package.h:10-26 records as S54 section 7
    // clause 1.
    {"spark_x2_5_4b", SparkX2_5_4B::model_id, SparkX2_5_4B::target_key,
     &SparkX2_5_4B::declares_model,
     &construct_registered<SparkX2_5_4B, LoadedSparkX2_5_4B, SparkX2_5_4BInstance>},
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

// Is there a row that would dispatch the qwen4-exp identity? Asked at COMPILE TIME, and over
// the table's model_id column rather than through declares_model(), so the answer stays a
// constant expression. It is the second half of the conditional gate further down: the
// preprocessor arm of that gate asserts this, because kStageBImplemented == true is a claim
// that this table carries the row and an open gate with nothing behind it is the one way the
// fail-safe inversion could go wrong.
constexpr bool qwen4_exp_row_is_registered() {
    for (std::size_t i = 0; i < kTargetRegistrationCount; ++i) {
        if (kTargetRegistrations[i].model_id == qwen4_exp::kModelId) { return true; }
    }
    return false;
}

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

// Spark-X2.5-4B (model_type spark2_5, model_id "spark-x2.5-4b").  Same shape as the
// qwen3_5_9b pair above: one weights profile, no MTP, no draft backend.  What is
// NOT here is the interesting part -- there is no geometry gate in this file for
// Spark because the family needs none: the artifact this target consumes is BF16
// in contiguous-le-v1 and src/core/arch_caps.h admits that on every route.  The
// three family hooks Spark DOES need (qk-norm, per-kind rope, the headwise gate)
// are refused inside the target, by name, at the runtime-view fill
// (src/targets/spark_x2_5_4b/impl/load/bindings.cpp:271-291) -- i.e. AFTER plan_load
// succeeds, so the load PLAN stays structurally testable against a real artifact.
LoadedSparkX2_5_4B::LoadedSparkX2_5_4B(std::unique_ptr<SparkX2_5_4B::LoadedModel> stable_model,
                                       const EngineOptions& options)
    : model(std::move(stable_model)),
      frontend(SparkX2_5_4B::make_frontend(*model, options)) {}

LoadedSparkX2_5_4B::~LoadedSparkX2_5_4B() = default;

SparkX2_5_4BInstance::SparkX2_5_4BInstance(
    std::unique_ptr<LoadedSparkX2_5_4B> stable_loaded,
    runtime::KvCapacityResolution resolution, SparkX2_5_4B::SequencePlan sequence_plan,
    SparkX2_5_4B::WeightsProfile weights_profile_in, DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(SparkX2_5_4B::create_program(*loaded->model, std::move(sequence_plan), device)),
      weights_profile(weights_profile_in) {}

SparkX2_5_4BInstance::~SparkX2_5_4BInstance() = default;

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

    // -----------------------------------------------------------------------------------------
    // THE LOAD-TIME BY-NAME GATE -- src/core/amdsafe_gate.h (landq/amdsafe/0002, amdcpu/0001).
    // -----------------------------------------------------------------------------------------
    // WHY HERE, AND NOWHERE ELSE. This function is the engine's ONLY load door: the artifact is
    // read at :508, the identity is resolved at :509, the arch/format verdict is taken at :529 and
    // the target registration is chosen at :536. It is also the engine's ONLY ARCH/VENDOR
    // DETERMINATION POINT, because the arch_view_for_device() call above is -- in its own words at
    // :518 -- "the ONLY place the test-only override can reach a decision the engine ACTS ON". And
    // it is the whole of ninfer_engine's own source list (src/CMakeLists.txt:633), so the target a
    // compile of ninfer_engine covers is the target THIS GATE COMPILES INTO.
    //
    // MEASURED BEFORE THIS EDIT: `grep -rl 'amdsafe_gate.h' build --include='*.o.d' | wc -l` = 0
    // of 599 dependency files. The header existed, said of itself that "a refusal which is not
    // loud is not a refusal", and NO TU IN THE BUILD INCLUDED IT -- so an AMD-class load path
    // reached the operator with no by-name gate in front of it at all.
    //
    // WHICH AXIS SWITCHES IT ON, AND WHY THE VENDOR AXIS RATHER THAN THE DEVICE. The gate
    // adjudicates an AMD primitive MAPPING, so it must run when AMD semantics are being reproduced
    // and must NOT run when they are not: on the physical device the primitives ARE this binary's
    // own, and the two names the gate calls unmappable (ldmatrix, mma_sync_family) are exactly
    // where those spellings COME FROM on CUDA -- so consulting it unconditionally would refuse a
    // load that is correct. The engine's AMD axis is caps::vendor_view_for_device(), the vendor
    // class's only reachable surface in this tree and the surface root CMakeLists.txt:263-269
    // records as having no engine consumer yet. `physical` is answered VendorClass::Nvidia, and
    // that is a DECISION rather than a device query, which src/core/vendor_sim.h:79-80 requires the
    // wiring line to state: this statement is reached only after device.sm() has SUCCEEDED, i.e.
    // only after a CUDA device answered for this process, and this tree deliberately carries no AMD
    // detector (vendor_sim.h:52-57 measures why one read here would report inverted).
    const caps::VendorView amdsafe_vendor_view =
        caps::vendor_view_for_device(caps::VendorClass::Nvidia);
    if (!amdsafe_vendor_view.usable()) {
        // G3-V, ONE AXIS OVER FROM THE ARCH GATE FOUR LINES UP. A request that was REFUSED is not a
        // licence to answer off the real card, and require_artifact_formats_supported(ArchView, ...)
        // at :530 already throws for exactly that reason. The vendor entry point has printed its own
        // reason; this makes the rule executable instead of advisory.
        throw std::runtime_error(
            std::string("artifact '") + identity.model_id + "/" + identity.weights_id +
            "' was gated against a SIMULATED VENDOR whose request was REFUSED, so no vendor "
            "answer exists for it: " +
            amdsafe_vendor_view.reason);
    }
    if (amdsafe_vendor_view.simulated()) {
        // THE ONE LINE A LOADER CALLS (src/core/amdsafe_gate.h:479). The NAME is the one this
        // function ALREADY hands to require_artifact_formats_supported() one statement above -- the
        // artifact's own identity, computed from the artifact's bytes at :509 -- so the by-name gate
        // and the format gate name the SAME subject and neither can drift from the artifact they
        // describe. A name the manifest does not carry is refused on that ground ALONE, which is the
        // gate's whole difference from a table.
        // ORDERING, AND IT IS LOAD-BEARING: this block must stay ABOVE any wiring of
        // caps::require_vendor_formats_supported() (src/core/vendor_sim.h:811, 0 engine callers
        // today), because that gate throws for EVERY format on an active non-NVIDIA view (G4-V
        // NEVER WIDENS) and wiring it first would make every line below unreachable.
        const std::string amdsafe_path = identity.model_id + "/" + identity.weights_id;
        if (!caps::amdsafe_gate_allows_load(amdsafe_path)) {
            const std::string amdsafe_refusal =
                caps::render_amdsafe_gate_refusal(caps::amdsafe_gate(amdsafe_path));
            std::fputs(amdsafe_refusal.c_str(), stderr);
            throw std::runtime_error(amdsafe_refusal);
        }
    }
    // Table lookup replaces the previous four-branch if chain. Row order is kept
    // identical to the chain, but the routes are disjoint by construction (the
    // static_assert above rejects a duplicated model_id), so first-match-wins is
    // order-independent.
    // ---------------------------------------------------------------------------------------
    // EngineOptions::ple_sidecar_root, READ. It used to be parsed by both front ends
    // (apps/cli/options.cpp:686, src/serve/serve_options.cpp:538), stored
    // (include/ninfer/types.h:571) and STAT'd at startup
    // (product::validate_ple_sidecar_root, from apps/cli/main.cpp:503 and
    // src/serve/generation_service.cpp:313) -- and then read by NOTHING on the engine side.
    // A by-name census over src/targets, src/ops, src/serve and src/product found exactly one
    // consumer of the value anywhere in the engine surface: resolve_ple_sidecar_root()
    // (targets/qwen4_exp/impl/ple_runtime.h:64), which had no caller either. So
    // `--ple-sidecar <valid root>` and no flag produced the same load, which is the silently
    // inert switch F391 names. The cure is not a comment; it is this caller.
    //
    // WHY BEFORE THE TABLE, AND NOT INSIDE A TARGET: this is the engine's only load door and
    // the identity is already known here, so the flag is read on EVERY load attempt, including
    // the ones that a target row answers and the ones no row claims. Putting it after the
    // dispatch would make it reachable only for families that consume it, i.e. exactly the
    // arrangement that made it invisible.
    //
    // THE THREE STATES OF THE FLAG, executed rather than described:
    //   * empty            -> PLE off. No attachment, no filesystem access. This is the default
    //                         and the arm every existing run takes, and it is byte-for-byte the
    //                         old behaviour (the fields it stamps below are empty).
    //   * set, this family -> PleRuntime::attach(): the sidecar (4 fds + a bounded pinned
    //                         cache, ple_table.h:35-38) is opened and its handle rides out on
    //                         the ConstructedTarget, so it is alive for the engine's lifetime.
    //   * set, other family-> REFUSED BY NAME. The PLE residual is ADDITIVE, so a load that
    //                         succeeded with a sidecar attached-and-ignored is indistinguishable
    //                         from PLE off in every downstream number -- the one outcome a gate
    //                         must not produce.
    // The adjacent ple-root/ fallback deliberately stays inside the qwen4_exp branch below: it
    // is that target's documented discovery rule, and applying it to every family would refuse
    // a load over a directory that merely happens to be called ple-root.
    qwen4_exp::PleSidecarHandle ple_sidecar;
    if (!options.ple_sidecar_root.empty()) {
        ple_sidecar = qwen4_exp::attach_ple_sidecar(options);
        if (!qwen4_exp::declares_ple_stage(identity.model_id)) {
            throw std::runtime_error(
                "artifact identity '" + identity.model_id + "/" + identity.weights_id +
                "' was given a PLE n-gram sidecar at '" + ple_sidecar.root +
                "' (--ple-sidecar) but this target declares no PLE stage, so the sidecar "
                "would be attached and then ignored: the PLE residual is ADDITIVE, so nothing "
                "downstream could tell that apart from PLE switched off. Pass an empty value "
                "(--ple-sidecar=) or drop the flag to switch PLE off on purpose.");
        }
    }

    for (const TargetRegistration& registration : kTargetRegistrations) {
        if (registration.declares_model(identity.model_id)) {
            ConstructedTarget constructed = registration.entry(
                options, device, reader, load_start, registration.target_key);
            // The attached sidecar rides out on the constructed target. Without this the
            // handle above would be a local that opens the sidecar and closes it again before
            // this function returns -- the same "the flag changed nothing" shape, only with
            // more file descriptors. It travels the way EngineOptions::prefill_chunk's
            // resolved value already does: a fact about THIS load the caller cannot recompute.
            constructed.ple_sidecar      = std::move(ple_sidecar.owner);
            constructed.ple_sidecar_root = std::move(ple_sidecar.root);
            return constructed;
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
        // THE REFUSAL IS CONDITIONAL NOW. kStageBImplemented is published by the build from an
        // existence test over this target's own stage (b) sources
        // (src/targets/qwen4_exp/stage_b.cmake -> NINFER_QWEN4_EXP_STAGE_B_COMPLETE=0/1), so:
        //   * stage (b) absent (TODAY, definition reads 0) -> refuse BY NAME, and name the
        //     missing sources from the SAME list that produced the verdict;
        //   * stage (b) present -> the refusal is withdrawn without a second edit here.
        // The alternative -- what was here -- took the refusal UNCONDITIONALLY, i.e. whether
        // or not a runtime existed, which made "land stage (b)" a two-file change whose second
        // file nobody would remember.
#if NINFER_QWEN4_EXP_STAGE_B_COMPLETE
        // Stage (b) exists. This arm deliberately contains no dispatch of its own: the row in
        // kTargetRegistrations IS the dispatch, exactly as it is for the four registered
        // families, and the assert below is what makes "both halves or neither" mechanical.
        // Reaching here with no row would otherwise fall through to the unknown-model refusal,
        // which names neither side of the problem.
        static_assert(qwen4_exp_row_is_registered(),
                      "NINFER_QWEN4_EXP_STAGE_B_COMPLETE is 1 but kTargetRegistrations has no "
                      "row whose model_id is the qwen4-exp identity: the stage (b) sources "
                      "landed without the registry row, so this open arm has nothing to "
                      "dispatch to. Add the row in the same patch as the sources (this file's "
                      "own table comment: 'Adding a model whose family is already in this "
                      "table = add one row here').");
        (void)geometry_verdict;
#else
        if (qwen4_exp::kStageBImplemented) {
            // The predicate and the preprocessor are the same fact spelled twice, one runtime
            // and one compile-time. They are computed from one definition, so a disagreement
            // means a build that published it to the header and not to this TU; assert rather
            // than trust, because in this direction trusting would OPEN the gate.
            throw std::logic_error(
                "qwen4-exp stage (b): the runtime predicate reads implemented while this "
                "translation unit's preprocessor reads absent; refusing to guess which is true");
        }
        throw std::runtime_error(
            "artifact identity '" + identity.model_id + "' recognized (" + geometry_verdict +
            "), but its runtime is not implemented yet (W7/P1 staged plan, "
            "_collab/B_s37_flashnext_p1.md): src/targets/qwen4_exp/ is missing " +
            qwen4_exp::stage_b_missing_sources() +
            " [stage_b_predicate=" + std::to_string(qwen4_exp::kStageBImplemented) +
            "]; refusing to load a stub target");
#endif
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
