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

namespace targets::qwen3_6_27b {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t {
    Qwen36GroupwiseInt,
    Qwen38GroupwiseInt,
    Qwen36Nvfp4,
    Qwen38Nvfp4,
    Qwen38Nvfp4Dspark,
    Qwen38Nvfp4DFlash2,
    Qwen38Nvfp4DFlash2Bf16Head,
    // A single-source ModelOpt NVFP4 checkpoint whose object plan is its own
    // declaration: the mixer objects keep the precision the source declared per
    // object instead of the registered layer-index split.
    Qwen38Nvfp4ModelOpt,
};

// General-task presets published with each exact model. Keep the registrations separate even
// while their values agree so an upstream model-specific change has one obvious owner.
inline constexpr ModelSamplingDefaults kQwen3_6Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

inline constexpr ModelSamplingDefaults kQwen3_8Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
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

    friend struct qwen3_6_27b::Package;
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

    friend struct qwen3_6_27b::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id           = "qwen3.6-27b";
    static constexpr std::string_view target_key         = "qwen3_6_27b";
    static constexpr std::string_view qwen3_8_model_id   = "qwen3.8-27b";
    static constexpr std::string_view qwen3_8_target_key = "qwen3_8_27b";

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

    // -------------------------------------------------------------------------------------
    // Declared weights capabilities
    // -------------------------------------------------------------------------------------
    // An artifact that declares one of these identities is loaded as that row's profile; anything
    // else is rejected with this table quoted back (`WeightsDeclarationTable::require`). A new
    // quantization flavour is a row here, and the resolution path has no other way to accept one.
    //
    // `nvfp4` and `nvfp4-modelopt` are the same Qwen3.8-27B NVFP4 export from two sources - the
    // compressed-tensors release and the ModelOpt checkpoint.
    // `docs/maintainer/modelopt-nvfp4-import.md` records that the packed codes and natural scales
    // are byte-identical, so they share one implementation: `Qwen38Nvfp4ModelOpt` is
    // the name this target reports for the ModelOpt source, and every switch that
    // handles `Qwen38Nvfp4` handles it in the same arm.
    static constexpr std::array<WeightsDeclaration<WeightsProfile>, 8>
        accepted_weights{{
            {.identity   = {.model_id = model_id, .weights_id = "groupwise-int"},
             .profile    = detail::WeightsProfile::Qwen36GroupwiseInt,
             .target_key = target_key,
             .provenance = "tools/convert/qwen3_6_27b/inventory.py"},
            {.identity   = {.model_id = qwen3_8_model_id, .weights_id = "groupwise-int"},
             .profile    = detail::WeightsProfile::Qwen38GroupwiseInt,
             .target_key = qwen3_8_target_key,
             .provenance = "tools/convert/qwen3_8_27b/inventory.py"},
            {.identity   = {.model_id = model_id, .weights_id = "nvfp4"},
             .profile    = detail::WeightsProfile::Qwen36Nvfp4,
             .target_key = target_key,
             .provenance = "tools/convert/qwen3_6_27b/inventory_nvfp4.py"},
            {.identity   = {.model_id = qwen3_8_model_id, .weights_id = "nvfp4"},
             .profile    = detail::WeightsProfile::Qwen38Nvfp4,
             .target_key = qwen3_8_target_key,
             .provenance = "tools/convert/qwen3_8_27b/inventory_nvfp4.py"},
            {.identity   = {.model_id = qwen3_8_model_id, .weights_id = "nvfp4-dspark"},
             .profile    = detail::WeightsProfile::Qwen38Nvfp4Dspark,
             .target_key = qwen3_8_target_key,
             .provenance = "qwen3.8-27b NVFP4 export plus the DSpark draft head"},
            {.identity   = {.model_id = qwen3_8_model_id, .weights_id = "nvfp4-dflash2"},
             .profile    = detail::WeightsProfile::Qwen38Nvfp4DFlash2,
             .target_key = qwen3_8_target_key,
             .provenance = "finalize_dflash2.ps1 on the nvfp4 export"},
            {.identity   = {.model_id = qwen3_8_model_id, .weights_id = "nvfp4-dflash2-bf16head"},
             .profile    = detail::WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head,
             .target_key = qwen3_8_target_key,
             .provenance = "as nvfp4-dflash2, with the bf16 head"},
            // The ModelOpt single-source import
            // (`tools/convert/dequant/modelopt.py` -> `tools/convert/qwen3_8_27b/convert_modelopt.py`).
            // Same data contract as `nvfp4` above.
            {.identity   = {.model_id = qwen3_8_model_id, .weights_id = "nvfp4-modelopt"},
             .profile    = detail::WeightsProfile::Qwen38Nvfp4ModelOpt,
             .target_key = qwen3_8_target_key,
             .provenance = "ModelOpt NVFP4 single source"},
        }};

    static_assert(declaration_check::identities_are_distinct(accepted_weights),
                  "the declared weights identities must be distinct");
    static_assert(declaration_check::target_keys_are(accepted_weights,
                                                     {target_key, qwen3_8_target_key}),
                  "every declared weights row must report one of this package's "
                  "target keys");

    // Every row must name both members of the identity it declares; ported from id1's
    // `identity_table_is_reachable`, which pinned this clause on its own identity table.
    static_assert(declaration_check::rows_name_an_identity(accepted_weights),
                  "a declared weights row must name a model id and a weights flavour");

    // `WeightsProfile` above has seven enumerators and this table reaches all seven, through its
    // eight rows (`nvfp4` and `nvfp4-modelopt` share `Qwen38Nvfp4`). Ported from id1's
    // `static_assert(kRealizations.size() == 7, "WeightsProfile changed: ...")`, which pinned the
    // same count on its own realization table. The literal is hand-written: it fires when a row
    // stops reaching a profile, and it is where an enumerator that arrived without a row has to be
    // faced - it cannot detect that case on its own.
    static_assert(declaration_check::distinct_profiles(accepted_weights) == 8,
                  "WeightsProfile changed: declare how the new enumerator is reached");

    // The declaration as a table: one lookup, one rejection position.
    [[nodiscard]] static constexpr WeightsDeclarationTable<WeightsProfile, 8>
    weights_declarations() noexcept {
        return WeightsDeclarationTable<WeightsProfile, 8>{accepted_weights,
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
    static constexpr std::array<SamplingDeclarationTable<2>::Declaration, 2>
        accepted_sampling{{
            {.model_id = model_id, .defaults = detail::kQwen3_6Defaults},
            {.model_id = qwen3_8_model_id, .defaults = detail::kQwen3_8Defaults},
        }};

    [[nodiscard]] static constexpr SamplingDeclarationTable<2> sampling_presets() noexcept {
        return SamplingDeclarationTable<2>{accepted_sampling, target_key};
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
    // Resolves --spec auto to a concrete backend (MTP or DFlash2) using the
    // artifact weights profile and context budget.
    [[nodiscard]] static EngineOptions resolved_auto_speculative(const EngineOptions& options,
                                                                 WeightsProfile weights_profile);
    // The same resolution, additionally handing back WHY the head came out the way it did.
    // `reason` is written by the DECIDING call's own out-parameter (startup_features.h's
    // `resolved_proposal_head`), so a caller that reports the head cannot name a reason that
    // disagrees with it. plan_load needs that: the adaptive spelling decides with the floor 5
    // in place of the raw window 0, so a reason recomputed from the raw window would print
    // `mtp-below-minimum` beside `head=optimized`. The two-argument form above is unchanged and
    // still used by the registry and by tests.
    [[nodiscard]] static EngineOptions resolved_auto_speculative(const EngineOptions& options,
                                                                 WeightsProfile weights_profile,
                                                                 std::string_view* reason);
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
    // Writes the loaded token-embedding / output-head payloads (fp8 row-scale)
    // as raw codes+scales files for offline teacher-weight export.
    static void export_head_weights(const LoadedModel& model, const char* directory);
};

} // namespace targets::qwen3_6_27b
} // namespace ninfer
