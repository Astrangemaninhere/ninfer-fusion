// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_8_flash_next/runtime.h>

#include <memory>
#include <string_view>

namespace ninfer {
namespace targets::qwen3_8_flash_next {

struct Package;

namespace detail {

enum class WeightsProfile {
    MixedNvfp4Fp8PleInt4,
};

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

public:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_8_flash_next::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;
    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;

public:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_8_flash_next::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen3.8-flash-next";
    static constexpr std::string_view target_key = "qwen3_8_flash_next";

    using WeightsProfile             = detail::WeightsProfile;
    using LoadPlan                   = detail::LoadPlan;
    using LoadedModel                = detail::LoadedModel;
    using Frontend                   = qwen3_6::Frontend;
    using PreparedPrompt             = qwen3_6::PreparedPrompt;
    using OutputSession              = qwen3_6::OutputSession;
    using PublishedOutput            = qwen3_6::PublishedOutput;
    using SequencePlanner            = qwen3_8_flash_next::SequencePlanner;
    using SequencePlan               = qwen3_8_flash_next::SequencePlan;
    using RequestBasePlan            = qwen3_8_flash_next::RequestBasePlan;
    using AdmissionCandidate         = qwen3_8_flash_next::AdmissionCandidate;
    using ResourcePlan               = qwen3_8_flash_next::ResourcePlan;
    using PersistentBackfillProof    = qwen3_8_flash_next::PersistentBackfillProof;
    using SequenceHandle             = qwen3_8_flash_next::SequenceHandle;
    using ContinuationHandle         = qwen3_8_flash_next::ContinuationHandle;
    using SharedPrefixHandle         = qwen3_8_flash_next::SharedPrefixHandle;
    using CaptureOffer               = qwen3_8_flash_next::CaptureOffer;
    using CacheSessionKey            = qwen3_6::PreparedSessionKey;
    using ContinuationSummary        = qwen3_6::ContinuationSummary;
    using SharedPrefixSummary        = qwen3_6::SharedPrefixSummary;
    using PressurePlanningSession        = qwen3_8_flash_next::PressurePlanningSession;
    using PressureTargetHandle           = qwen3_6::PressureTargetHandle;
    using AssessedPressureTarget         = qwen3_8_flash_next::AssessedPressureTarget;
    using CapturePressurePlan            = qwen3_8_flash_next::CapturePressurePlan;
    using CapturePressurePlanningSession = qwen3_8_flash_next::CapturePressurePlanningSession;
    using MaterializationResult      = qwen3_8_flash_next::MaterializationResult;
    using ContextTransactionProgress = qwen3_8_flash_next::ContextTransactionProgress;
    using CaptureAssessment          = qwen3_6::CaptureAssessment;
    using ActiveCaptureResult        = qwen3_8_flash_next::ActiveCaptureResult;
    using PendingBatch               = qwen3_8_flash_next::PendingBatch;
    using StartResult                = qwen3_8_flash_next::StartResult;
    using PrefillProgress            = qwen3_8_flash_next::PrefillProgress;
    using CommitResult               = qwen3_8_flash_next::CommitResult;
    using DiscardResult              = qwen3_8_flash_next::DiscardResult;
    using FinishResult               = qwen3_8_flash_next::FinishResult;
    using AbortResult                = qwen3_8_flash_next::AbortResult;
    using ReleaseResult              = qwen3_8_flash_next::ReleaseResult;
    using Program                    = qwen3_8_flash_next::Program;

    // -------------------------------------------------------------------------------------
    // Declared capability + engine-registration surface
    // -------------------------------------------------------------------------------------
    // One identity, one storage scheme. The converter writes this weights_id
    // (tools/convert/qwen3_8_flash_next/inventory.py); `declares_model` is the predicate
    // src/targets/registry.cpp asks before it dispatches, and resolve_weights() accepts exactly
    // the same pair. Both read the literals below, so "registered" cannot drift from "accepted".
    static constexpr std::string_view weights_id = "mixed-nvfp4-fp8-ple-int4";

    [[nodiscard]] static constexpr bool declares_model(std::string_view model) noexcept {
        return model == model_id;
    }

    [[nodiscard]] static constexpr std::string_view target_key_for(std::string_view model) noexcept {
        return model == model_id ? target_key : std::string_view{};
    }

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
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
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device,
                   const StartupObserver& startup_observer = {});

    // Resolves both `Auto` spellings before any consumer sees them. Required by
    // src/targets/registry.cpp's construct_registered<>, which hands the result to plan_load, the
    // sequence planner and create_program -- all three must read the same concrete values, or the
    // frozen startup features check rejects the loaded weights. See impl/package.cpp for the
    // resolution and for why `--spec auto` lands on None on this target.
    [[nodiscard]] static EngineOptions resolved_auto_speculative(const EngineOptions& options,
                                                                 WeightsProfile weights_profile);

    // The NINFER_EXPORT_HEAD_DIR hook inside construct_registered<>. Declared because that
    // template odr-uses it for every registered family; see impl/package.cpp for why this target
    // refuses by name instead of writing a dump whose shape would be wrong for it.
    static void export_head_weights(const LoadedModel& model, const char* directory);
};

} // namespace targets::qwen3_8_flash_next
} // namespace ninfer
