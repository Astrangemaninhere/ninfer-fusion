// ninfer::targets::spark_x2_5_4b - package implementation.  Generated from the
// qwen3_6_27b template in the shape src/targets/muse_glimmer_30b/impl/package.cpp
// established: single weights profile, no speculative backends, nine Package statics.
#include <ninfer/targets/spark_x2_5_4b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cuda_runtime.h>

#include "artifact/reader.h"
#include "targets/spark_x2_5_4b/impl/load/bindings.h"
#include "targets/spark_x2_5_4b/impl/tokenizer_policy.h"
#include "targets/spark_x2_5_4b/impl/variant.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::spark_x2_5_4b::detail {

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

} // namespace ninfer::targets::spark_x2_5_4b::detail

namespace ninfer::targets::spark_x2_5_4b {
namespace {

// The declaration and the binder must agree about which profile carries a draft tensor
// group: `detail::Variant::dspark_weights` / `dflash2_weights` (impl/variant.h) are the
// single source of truth for that.  Both return false unconditionally for Spark -- the
// checkpoint has no MTP head and no draft head, the index's 290 tensors being 36 * 8
// leaves plus embedding plus final norm -- and the single accepted_weights row must
// therefore name no draft flavour either.  Ported from muse's
// `declared_draft_identities_match_payload`, which is where this check was first
// written after the same class of mistake (a repointed identity taking the draft
// dispatch with it) was found elsewhere.
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
                                                 WeightsProfile) {
    EngineOptions resolved = options;
    if (options.speculative.backend == SpeculativeBackend::Auto) {
        // There is nothing to resolve TO.  Auto -> None is not a fallback here, it is
        // the only answer: the checkpoint has no MTP head and no draft head, so every
        // explicit request fails later with a clear weights/feature mismatch and Auto
        // lands on the one backend that exists.  Same shape as muse's resolution and
        // for the same reason.
        resolved.speculative.backend = SpeculativeBackend::None;
    }
    // No shortlist draft head on this target, and a disabled run must land on the full
    // head (layouts_impl.h requires it), so Auto resolves to Full here.
    resolved.speculative.proposal_head = qwen3_6::resolved_proposal_head(
        resolved.speculative.proposal_head, resolved.speculative.backend, false,
        resolved.speculative.draft_tokens);
    return resolved;
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    const EngineOptions resolved = Package::resolved_auto_speculative(options, weights_profile);
    // NOTE, and it is the honest state of this target: plan_load SUCCEEDS.  It produces
    // a ten-objects-per-layer plan whose names and shapes are the measured ones, and a
    // structural test can check it against a real artifact without a device.  What
    // refuses is `LoadedModelData`'s constructor (impl/load/bindings.cpp), i.e. the
    // step that would publish the runtime view, because three of the family's
    // unconditional operations are wrong for this checkpoint.  That split is
    // deliberate: refusing in plan_load would hide the part of this target that IS
    // correct and testable.
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
    // token_domain is passed EXPLICITLY and it is 131072, not the family default.  Two
    // measured reasons (S54 section 3.6):
    //   * `validate_registered_tokenizer` requires exactly this domain
    //     (frontend.cpp's has_exact_token_domain path: valid_token_ids_ all true and the
    //     table exactly that long).  A frontend built at the family default refuses the
    //     tokenizer.
    //   * the family's official special-id table (248053-248076) is a qwen range that
    //     Spark's tokenizer does not have, so `validate_official_special_ids` is false
    //     AND the converter must not insert those ids into the vocabulary.  The
    //     converter half of that is `SUPPLIES_FRONTEND_RESOURCES = False`
    //     (tools/convert/spark_x2_5_4b/convert.py:72) -- see the report's companion land
    //     (3): this target's frontend plan is empty while this call expects bound
    //     resources, and that seam is named, not assumed.
    return qwen3_6::make_frontend(model.impl_->data.frontend,
                                  qwen3_6::FrontendOptions{
                                      .vision_enabled = model.impl_->data.runtime.features.vision,
                                      .token_domain =
                                          static_cast<std::size_t>(detail::Variant::TextConfig::token_domain),
                                      .validate_official_special_ids = false,
                                      // Spark's OWN tokenizer policy, declared in
                                      // impl/tokenizer_policy.h -- the fact dl/sparkx
                                      // measured the engine refusing this model for.
                                      .tokenizer_policy = detail::kTokenizerPolicy,
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
    const auto dump = [&](const char* name, const Weight& weight) {
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
    // Spark's head IS the embedding's bytes (tie_word_embeddings, materialised by the
    // converter).  So the two dumps must come out byte-identical, and this is the
    // cheapest check in the whole target that the tie was materialised the way the
    // converter said it would be -- a `cmp` on the two _codes.bin files is the gate.
    dump("embed", view.token_embedding);
    dump("head", view.output_head);
}

} // namespace ninfer::targets::spark_x2_5_4b
