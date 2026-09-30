#pragma once

// src/targets/spark_x2_5_4b/export/ninfer/targets/spark_x2_5_4b/package.h
//
// The identity header, and the ONLY file of the eight that is a scaffold rather
// than engine glue (tools/archkit/archkit_target.py:121-132 marks it SCAFFOLD).
// Shape copied from src/targets/muse_glimmer_30b/export/ninfer/targets/muse_glimmer_30b/package.h,
// which is the closest sibling: one weights profile, no MTP, no draft backend.
//
// TWO IDENTIFIERS, DELIBERATELY DIFFERENT, AND THAT IS S54 SECTION 7 CLAUSE 1
// -------------------------------------------------------------------------
//   model_id   = "spark-x2.5-4b"   -- the HF-side name.  It is what the artifact's
//                                     ArtifactIdentity carries, it contains '.', and
//                                     it is NEVER a C++ identifier.
//   target_key = "spark_x2_5_4b"  -- the C++ / on-disk name: directory, namespace,
//                                     archkit ns_of(), and the string the front door
//                                     prints.  source: adapt.py's cpp_ident().
//   the weights flavour is "bf16", which is the identity the converter actually
//   emits today: dl/sparkintel's plan prints `identity: spark-x2.5-4b / bf16
//   (target_key spark_x2_5_4b)`.  S54 section 7 clause 2 ruled BF16 for v1 because
//   Spark ships no quantisation products and no new architecture has a registered
//   quantised geometry; an NVFP4 row is a later, separate declaration.
//
// The registry's TargetRegistration row keys on `declares_model(model_id)` and
// reports `target_key`, so the two spellings are the two halves of one routing
// decision and separating them here is what keeps '.' out of the C++ side.

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "targets/declared_capabilities.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::spark_x2_5_4b {

struct Package;

namespace detail {

struct Variant;

// One profile.  v1 is BF16 (S54 section 7 clause 2); a quantised row would be a new
// enumerator and a new accepted_weights row, not a reinterpretation of this one.
enum class WeightsProfile : std::uint8_t {
    SparkBf16,
};

// Sampling defaults.  Spark's own generation config carries the usual 1.0 / 0.95
// pair and no top-k (S54's S54-5 step: "temperature=1.0, top_p=0.95, top_k=-1 ->
// engine top_k=0"), so the engine's "top-k disabled" spelling 0 is used rather than
// -1: the engine's field is a count, and -1 is not a count.
inline constexpr ModelSamplingDefaults kSparkDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 0,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 1.0F,
                     .top_k             = 0,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
};

// The declared-capability vocabulary (src/targets/declared_capabilities.h), imported
// so the declaration in `Package` reads without a path.
using targets::SamplingDeclarationTable;
using targets::WeightsDeclaration;
using targets::WeightsDeclarationTable;

using Frontend        = qwen3_6::Frontend;
using PreparedPrompt  = qwen3_6::PreparedPrompt;
using OutputSession   = qwen3_6::OutputSession;
using PublishedOutput = qwen3_6::PublishedOutput;

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct spark_x2_5_4b::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct spark_x2_5_4b::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "spark-x2.5-4b";
    static constexpr std::string_view target_key = "spark_x2_5_4b";

    using WeightsProfile             = detail::WeightsProfile;
    using LoadPlan                   = detail::LoadPlan;
    using LoadedModel                = detail::LoadedModel;
    using Frontend                   = detail::Frontend;
    using PreparedPrompt             = detail::PreparedPrompt;
    using OutputSession              = detail::OutputSession;
    using PublishedOutput            = detail::PublishedOutput;
    using SequencePlanner            = qwen3_6::SequencePlanner<detail::Variant>;
    using SequencePlan               = qwen3_6::SequencePlan<detail::Variant>;
    using RequestBasePlan            = qwen3_6::RequestBasePlan<detail::Variant>;
    using AdmissionCandidate         = qwen3_6::AdmissionCandidate<detail::Variant>;
    using ResourcePlan               = qwen3_6::ResourcePlan<detail::Variant>;
    using PersistentBackfillProof    = qwen3_6::PersistentBackfillProof<detail::Variant>;
    using SequenceHandle             = qwen3_6::SequenceHandle<detail::Variant>;
    using ContinuationHandle         = qwen3_6::ContinuationHandle<detail::Variant>;
    using SharedPrefixHandle         = qwen3_6::SharedPrefixHandle<detail::Variant>;
    using CaptureOffer               = qwen3_6::CaptureOffer<detail::Variant>;
    using CacheSessionKey            = qwen3_6::PreparedSessionKey;
    using ContinuationSummary        = qwen3_6::ContinuationSummary;
    using SharedPrefixSummary        = qwen3_6::SharedPrefixSummary;
    using PressurePlanningSession    = qwen3_6::PressurePlanningSession<detail::Variant>;
    using PressureTargetHandle       = qwen3_6::PressureTargetHandle;
    using CapturePressurePlan        = qwen3_6::CapturePressurePlan<detail::Variant>;
    using MaterializationResult      = qwen3_6::MaterializationResult<detail::Variant>;
    using ContextTransactionProgress = qwen3_6::ContextTransactionProgress<detail::Variant>;
    using CaptureAssessment          = qwen3_6::CaptureAssessment;
    using ActiveCaptureResult        = qwen3_6::ActiveCaptureResult<detail::Variant>;
    using PendingBatch               = qwen3_6::PendingBatch<detail::Variant>;
    using StartResult                = qwen3_6::StartResult<detail::Variant>;
    using PrefillProgress            = qwen3_6::PrefillProgress<detail::Variant>;
    using CommitResult               = qwen3_6::CommitResult<detail::Variant>;
    using DiscardResult              = qwen3_6::DiscardResult<detail::Variant>;
    using FinishResult               = qwen3_6::FinishResult<detail::Variant>;
    using AbortResult                = qwen3_6::AbortResult<detail::Variant>;
    using ReleaseResult              = qwen3_6::ReleaseResult<detail::Variant>;
    using Program                    = qwen3_6::Program<detail::Variant>;

    // The declared weights capabilities of this target. One row: an artifact that
    // declares any other identity is rejected with this table quoted back. The
    // provenance string is the converter that emits the identity, and it must stay
    // the module that actually writes it -- dl/sparkintel landed that directory, and
    // it is the tree's `tools/convert/spark_x2_5_4b/convert.py`.
    static constexpr std::array<WeightsDeclaration<WeightsProfile>, 1>
        accepted_weights{{
            {.identity   = {.model_id = model_id, .weights_id = "bf16"},
             .profile    = detail::WeightsProfile::SparkBf16,
             .target_key = target_key,
             .provenance = "tools/convert/spark_x2_5_4b/convert.py"},
        }};

    static_assert(declaration_check::identities_are_distinct(accepted_weights),
                  "the declared weights identities must be distinct");
    static_assert(declaration_check::target_keys_are(accepted_weights,
                                                     {target_key}),
                  "every declared weights row must report one of this package's "
                  "target keys");
    static_assert(declaration_check::rows_name_an_identity(accepted_weights),
                  "a declared weights row must name a model id and a weights flavour");

    // The declaration as a table: one lookup, one rejection position.
    [[nodiscard]] static constexpr WeightsDeclarationTable<WeightsProfile, 1>
    weights_declarations() noexcept {
        return WeightsDeclarationTable<WeightsProfile, 1>{accepted_weights, target_key};
    }

    [[nodiscard]] static constexpr bool declares_model(std::string_view model) noexcept {
        return weights_declarations().declares_model(model);
    }

    // The target key this declaration reports for the model id of an artifact identity.
    [[nodiscard]] static constexpr std::string_view target_key_for(std::string_view model) noexcept {
        return weights_declarations().target_key_for(model);
    }

    // The declared sampling presets: one row per model id this target publishes
    // defaults for. Same shape as the weights declaration - one lookup, and a miss
    // names the model ids it serves.
    static constexpr std::array<SamplingDeclarationTable<1>::Declaration, 1>
        accepted_sampling{{
            {.model_id = model_id, .defaults = detail::kSparkDefaults},
        }};

    [[nodiscard]] static constexpr SamplingDeclarationTable<1> sampling_presets() noexcept {
        return SamplingDeclarationTable<1>{accepted_sampling, target_key};
    }

    static_assert(declaration_check::sampling_models_are_declared(accepted_weights,
                                                                 accepted_sampling),
                  "a sampling preset must belong to a model id this target consumes artifacts of");

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static std::string_view declare_identity(std::string_view model,
                                                          std::string_view weights);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static EngineOptions resolved_auto_speculative(const EngineOptions& options,
                                                                 WeightsProfile weights_profile);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model,
                                                const EngineOptions& options);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
    // Writes the loaded token-embedding / output-head payloads as raw
    // codes+scales files for offline teacher-weight export.  Spark's head is the
    // materialised copy of the embedding (tie_word_embeddings), so the two dumps
    // must be byte-identical -- which makes this the cheapest possible check that
    // the tie was materialised the way the converter said it would be.
    static void export_head_weights(const LoadedModel& model, const char* directory);
};

} // namespace targets::spark_x2_5_4b
} // namespace ninfer
