#include <ninfer/targets/qwen3_5_9b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cuda_runtime.h>

#include "artifact/reader.h"
#include "targets/qwen3_5_9b/impl/load/bindings.h"
#include "targets/qwen3_5_9b/impl/load/draft_resolution.h"
#include "targets/qwen3_5_9b/impl/variant.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_5_9b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_5_9b::detail

namespace ninfer::targets::qwen3_5_9b {
namespace {

// The declaration and the binder must agree about which profile carries a draft tensor
// group.  This package declares exactly one identity and its `Variant::dspark_weights` /
// `dflash2_weights` return false for it, so a row that ever names `nvfp4-dspark` or
// `nvfp4-dflash2` without the matching predicate (or the reverse) fails the build here
// instead of binding a draft the runtime will not run.
constexpr bool declared_draft_identities_match_payload() {
    for (const targets::WeightsDeclaration<detail::WeightsProfile>& row :
         Package::accepted_weights) {
        const bool names_dspark  = row.identity.weights_id == "nvfp4-dspark";
        const bool names_dflash2 = row.identity.weights_id == "nvfp4-dflash2";
        if (names_dspark != detail::Variant::dspark_weights(row.profile)) { return false; }
        if (names_dflash2 != detail::Variant::dflash2_weights(row.profile)) { return false; }
    }
    return true;
}
static_assert(declared_draft_identities_match_payload(),
              "the declared draft identities must agree with the payload predicates");

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    return sampling_presets().require(model).defaults;
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    return weights_declarations().require(identity.model_id, identity.weights_id).profile;
}

std::string_view Package::declare_identity(std::string_view model, std::string_view weights) {
    return weights_declarations().require(model, weights).target_key;
}

EngineOptions Package::resolved_auto_speculative(const EngineOptions& options,
                                                 WeightsProfile weights_profile) {
    EngineOptions resolved = options;
    // This checkpoint's own nextn/MTP block (`blk.32.nextn.*`, `qwen35.nextn_predict_layers
    // = 1`) is the only speculative backend it can offer, and there is no shortlist proposal
    // head in it.  DFlash / DFlash2 are not selectable: `Variant::supports_dflash` /
    // `supports_dflash2` are false and no dflash object is bound, so asking for one fails in
    // the binder by name rather than silently running the target head.
    //
    // WHETHER THE ARTIFACT HAS A DRAFT BLOCK IS READ FROM THE ARTIFACT, not assumed: the
    // weights flavour comes out of the artifact's identity (`resolve_weights`, called by the
    // registry with `reader.identity()` before anything is bound), and the flavour was written
    // by the converter from the source GGUF's own `nextn_predict_layers`.  So `--spec auto`
    // -- the front end's default, documented as "uses the artifact's own draft backend" --
    // resolves to None for an artifact converted from a source with no MTP layer, instead of
    // resolving to Mtp and dying in the binder with "required artifact object is missing:
    // mtp/input_projection".  That failure was the whole reason a Bonsai-family source (no MTP
    // key at all) could not start the engine.
    //
    // The registry resolves Auto at the same point from the same flavour, so the planner, the
    // load plan and the program see one answer.
    const bool artifact_declares_draft = weights_profile == WeightsProfile::GgufKquant;
    if (options.speculative.backend != SpeculativeBackend::Auto) {
        resolved.speculative.proposal_head = qwen3_6::resolved_proposal_head(
            resolved.speculative.proposal_head, resolved.speculative.backend,
            /*has_shortlist_head=*/false,
            resolved.speculative.draft_tokens == 0 ? qwen3_6::kMtpShortlistMinimumDrafts
                                                  : resolved.speculative.draft_tokens);
        return resolved;
    }
    // The answer itself is the tested pure function (impl/load/draft_resolution.h), so the
    // mapping "artifact declares no draft block + auto -> no speculation" is not a claim about
    // this file: `dl/mtpcopt/cpp/test_draft_resolution.cpp` runs it on every combination.
    const bool draft_window_requested =
        resolved.speculative.draft_tokens > 0 || resolved.speculative.draft_tree_paths > 0;
    switch (detail::resolve_auto_draft(artifact_declares_draft, draft_window_requested)) {
    case detail::DraftResolution::RefuseDraftWindow:
        // A draft WINDOW asked for on top of Auto is named rather than dropped: silently
        // ignoring `--draft-tokens 3` on this artifact would be a run whose reported
        // configuration is not the one it executed.
        throw std::invalid_argument(
            "--spec auto with a draft window (--draft-tokens / --draft-tree) was given for "
            "an artifact whose source GGUF declares no MTP/nextn block (weights flavour "
            "gguf-kquant-nomtp), so there is no draft object to fill the window.  Drop the "
            "draft window, or convert a source that carries a draft block.");
    case detail::DraftResolution::DisableSpeculation:
        resolved.speculative.backend = SpeculativeBackend::None;
        resolved.speculative.proposal_head =
            qwen3_6::resolved_proposal_head(resolved.speculative.proposal_head,
                                            resolved.speculative.backend,
                                            /*has_shortlist_head=*/false,
                                            resolved.speculative.draft_tokens);
        return resolved;
    case detail::DraftResolution::UseMtp:
        break;
    }
    resolved.speculative.backend = SpeculativeBackend::Mtp;
    resolved.speculative.proposal_head =
        qwen3_6::resolved_proposal_head(resolved.speculative.proposal_head,
                                        resolved.speculative.backend,
                                        /*has_shortlist_head=*/false,
                                        resolved.speculative.draft_tokens);
    return resolved;
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    const EngineOptions resolved = Package::resolved_auto_speculative(options, weights_profile);
    if (resolved.enable_vision) {
        throw std::invalid_argument(
            "target 'qwen3_5_9b' declares no vision tower: this artifact carries the text stack "
            "only (the source GGUF has no vision tensors), so --vision has no objects to bind. "
            "Drop --vision; text generation is unaffected.");
    }
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, weights_profile, qwen3_6::startup_features(resolved))));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(model.impl_->data.frontend,
                                  qwen3_6::FrontendOptions{
                                      .vision_enabled = model.impl_->data.runtime.features.vision,
                                      .max_context    = options.max_context,
                                      .media_cache_bytes        = options.media_cache_bytes,
                                      .media_live_bytes         = options.media_live_bytes,
                                      .media_preprocess_threads = options.media_preprocess_threads,
                                  });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen3_6::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

void Package::export_head_weights(const LoadedModel& model, const char* directory) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    const auto& view = model.impl_->data.runtime;
    const auto dump  = [&](const char* name, const Weight& weight) {
        if (weight.qdata == nullptr || weight.n <= 0 || weight.k <= 0) {
            throw std::runtime_error(std::string("head export: ") + name + " is unavailable");
        }
        const std::string base = std::string(directory) + "/" + name;
        const std::size_t code_bytes =
            static_cast<std::size_t>(weight.n) * static_cast<std::size_t>(weight.k);
        const std::size_t scale_bytes = static_cast<std::size_t>(weight.n) * 2;
        std::vector<std::byte> codes(code_bytes);
        std::vector<std::byte> scales(scale_bytes);
        CUDA_CHECK(cudaMemcpy(codes.data(), weight.qdata, code_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scales.data(), weight.scales, scale_bytes, cudaMemcpyDeviceToHost));
        std::FILE* f = std::fopen((base + "_codes.bin").c_str(), "wb");
        if (f == nullptr) { throw std::runtime_error("cannot open " + base + "_codes.bin"); }
        std::fwrite(codes.data(), 1, codes.size(), f);
        std::fclose(f);
        f = std::fopen((base + "_scales.bin").c_str(), "wb");
        if (f == nullptr) { throw std::runtime_error("cannot open " + base + "_scales.bin"); }
        std::fwrite(scales.data(), 1, scales.size(), f);
        std::fclose(f);
    };
    dump("embed", view.token_embedding);
    dump("head", view.output_head);
}

} // namespace ninfer::targets::qwen3_5_9b
