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

namespace targets::qwen3_5_9b {

struct Package;

namespace detail {

struct Variant;

// One artifact identity, one storage scheme: the GGUF K-quant export re-quantized
// to the groupwise-64 int family the qwen3_6 runtime already consumes.  The
// enumerator exists so the family's per-profile switches have a name to case on;
// it is deliberately NOT a copy of qwen3_6_27b's eight profiles, because not one
// of that package's `weights_id`s can appear in an artifact this converter writes.
//
// Two flavours, because the draft block is optional AT THE SOURCE: the converter writes
// `gguf-kquant` when the source GGUF declares a nextn block and `gguf-kquant-nomtp` when it
// declares none (tools/convert/qwen3_5_9b/convert.py).  The flavour is a SECOND enumerator
// rather than a flag because `resolve_weights` reads it out of the artifact's identity, i.e.
// out of the artifact itself, before any object is bound -- and that is what lets
// `resolved_auto_speculative` answer `--spec auto` with a backend this artifact can run.
enum class WeightsProfile : std::uint8_t {
    GgufKquant,
    GgufKquantNoMtp,
};

// General-task presets.  Values are the family's registered defaults for a
// thinking/non-thinking Qwen3.5 checkpoint; the GGUF itself publishes only
// eos/bos/pad ids, so nothing here is derived from this checkpoint's own files.
inline constexpr ModelSamplingDefaults kQwen3_5Defaults{
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

    friend struct qwen3_5_9b::Package;
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

    friend struct qwen3_5_9b::Package;
};

} // namespace detail

struct Package {
    // The identity this package claims.  It is the one its converter writes into the
    // container (`tools/convert/qwen3_5_9b/convert.py`, `inventory.MODEL_ID`), and the
    // upstream GGUF's `general.architecture` (`qwen35`) is recorded in the package
    // header as provenance rather than used as an identity: llama.cpp's architecture
    // string names a tensor-name/geometry family, and a name family is not a storage
    // scheme.
    static constexpr std::string_view model_id   = "qwen3.5-9b";
    static constexpr std::string_view target_key = "qwen3_5_9b";

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
    // One row: the identity this package's converter writes.  A second flavour is a new row
    // here, and the resolution path has no other way to accept one.  The `weights_id` names
    // the *storage scheme* of the artifact, not the upstream quantisation type of the GGUF
    // it came from: the converter dequantises Q4_K/Q6_K/F32 to fp32 and re-encodes into the
    // groupwise-64 int family, so an artifact of a different source quantisation is the same
    // artifact identity with different numbers, not a different weights_id.
    static constexpr std::array<WeightsDeclaration<WeightsProfile>, 2> accepted_weights{{
        {.identity   = {.model_id = model_id, .weights_id = "gguf-kquant"},
         .profile    = detail::WeightsProfile::GgufKquant,
         .target_key = target_key,
         .provenance = "tools/convert/qwen3_5_9b/convert.py (Ornith-1.5-9B-Q4_K_M.gguf, "
                       "general.architecture=qwen35, qwen35.nextn_predict_layers=1: the "
                       "artifact carries the 12 mtp/* objects)"},
        {.identity   = {.model_id = model_id, .weights_id = "gguf-kquant-nomtp"},
         .profile    = detail::WeightsProfile::GgufKquantNoMtp,
         .target_key = target_key,
         .provenance = "the same converter on a source that declares NO draft block (e.g. the "
                       "Bonsai family GGUFs, qwen35.block_count=64 with no "
                       "qwen35.nextn_predict_layers key): the artifact carries no mtp/* object "
                       "and `--spec auto` resolves to none for it"},
    }};

    static_assert(declaration_check::identities_are_distinct(accepted_weights),
                  "the declared weights identities must be distinct");
    static_assert(declaration_check::target_keys_are(accepted_weights, {target_key}),
                  "every declared weights row must report one of this package's target keys");
    static_assert(declaration_check::rows_name_an_identity(accepted_weights),
                  "a declared weights row must name a model id and a weights flavour");
    // One enumerator, one row: each flavour above is reached by its own `weights_id`, and
    // this literal is where a second one has to be faced.
    static_assert(declaration_check::distinct_profiles(accepted_weights) == 2,
                  "WeightsProfile changed: the two enumerators are the two rows above -- "
                  "GgufKquant (the source declares a draft block) and GgufKquantNoMtp (it does "
                  "not), reached by the artifact's own weights_id; a third enumerator must "
                  "arrive with the row that reaches it");

    [[nodiscard]] static constexpr WeightsDeclarationTable<WeightsProfile, 2>
    weights_declarations() noexcept {
        return WeightsDeclarationTable<WeightsProfile, 2>{accepted_weights, target_key};
    }

    [[nodiscard]] static constexpr bool declares_model(std::string_view model) noexcept {
        return weights_declarations().declares_model(model);
    }

    [[nodiscard]] static constexpr std::string_view target_key_for(std::string_view model) noexcept {
        return weights_declarations().target_key_for(model);
    }

    static constexpr std::array<SamplingDeclarationTable<1>::Declaration, 1> accepted_sampling{{
        {.model_id = model_id, .defaults = detail::kQwen3_5Defaults},
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
    static void export_head_weights(const LoadedModel& model, const char* directory);
};

} // namespace targets::qwen3_5_9b
} // namespace ninfer
