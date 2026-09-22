#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "targets/declared_capabilities.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/runtime.h>

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

namespace targets::qwen3_6_35b_a3b {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t {
    GroupwiseInt,
};

inline constexpr ModelSamplingDefaults kQwen3_6_35BA3BDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

// The declared-capability vocabulary (`src/targets/declared_capabilities.h`), imported so the
// declaration in `Package` reads without a path.
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

    friend struct qwen3_6_35b_a3b::Package;
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

    friend struct qwen3_6_35b_a3b::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen3.6-35b-a3b";
    static constexpr std::string_view target_key = "qwen3_6_35b_a3b";

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

    // The declared weights capabilities of this target. One row: an artifact that declares any
    // other identity is rejected with this table quoted back.
    static constexpr std::array<WeightsDeclaration<WeightsProfile>, 1>
        accepted_weights{{
            {.identity   = {.model_id = model_id, .weights_id = "groupwise-int"},
             .profile    = detail::WeightsProfile::GroupwiseInt,
             .target_key = target_key,
             .provenance = "tools/convert/qwen3_6_35b_a3b/inventory.py"},
        }};

    static_assert(declaration_check::identities_are_distinct(accepted_weights),
                  "the declared weights identities must be distinct");
    static_assert(declaration_check::target_keys_are(accepted_weights,
                                                     {target_key}),
                  "every declared weights row must report one of this package's "
                  "target keys");

    // Every row must name both members of the identity it declares; ported from id1's
    // `identity_table_is_reachable`, which pinned this clause on its own identity table.
    static_assert(declaration_check::rows_name_an_identity(accepted_weights),
                  "a declared weights row must name a model id and a weights flavour");

    // The declaration as a table: one lookup, one rejection position.
    [[nodiscard]] static constexpr WeightsDeclarationTable<WeightsProfile, 1>
    weights_declarations() noexcept {
        return WeightsDeclarationTable<WeightsProfile, 1>{accepted_weights,
                                                                         target_key};
    }

    [[nodiscard]] static constexpr bool declares_model(std::string_view model) noexcept {
        return weights_declarations().declares_model(model);
    }

    // The target key this declaration reports for the model id of an artifact identity.
    [[nodiscard]] static constexpr std::string_view target_key_for(std::string_view model) noexcept {
        return weights_declarations().target_key_for(model);
    }

    // The declared sampling presets: one row per model id this target publishes defaults for. Same
    // shape as the weights declaration - one lookup, and a miss names the model ids it serves.
    static constexpr std::array<SamplingDeclarationTable<1>::Declaration, 1>
        accepted_sampling{{
            {.model_id = model_id, .defaults = detail::kQwen3_6_35BA3BDefaults},
        }};

    [[nodiscard]] static constexpr SamplingDeclarationTable<1> sampling_presets() noexcept {
        return SamplingDeclarationTable<1>{accepted_sampling, target_key};
    }

    static_assert(declaration_check::sampling_models_are_declared(accepted_weights,
                                                                 accepted_sampling),
                  "a sampling preset must belong to a model id this target consumes artifacts of");

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    // The target key this declaration reports for the artifact's identity; throws with the declared
    // accepted set when the artifact declares a weights flavour this package does not consume.
    [[nodiscard]] static std::string_view declare_identity(std::string_view model,
                                                          std::string_view weights);
    // The declared profile of the artifact's identity (see `accepted_weights` above).
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    // Resolves --spec auto (this target has no DFlash2 weights; auto -> MTP).
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
    static void export_head_weights(const LoadedModel&, const char*) {}
};

} // namespace targets::qwen3_6_35b_a3b
} // namespace ninfer
