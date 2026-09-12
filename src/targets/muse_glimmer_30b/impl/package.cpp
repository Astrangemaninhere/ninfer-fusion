// ninfer::targets::muse_glimmer_30b - package implementation (generated from
// the qwen3_6_27b template; single weights profile, no speculative backends).
#include <ninfer/targets/muse_glimmer_30b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cuda_runtime.h>

#include "artifact/reader.h"
#include "targets/muse_glimmer_30b/impl/load/bindings.h"
#include "targets/muse_glimmer_30b/impl/variant.h"

#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <utility>

namespace ninfer::targets::muse_glimmer_30b::detail {

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

} // namespace ninfer::targets::muse_glimmer_30b::detail

namespace ninfer::targets::muse_glimmer_30b {
namespace {

constexpr ModelSamplingDefaults kMuseDefaults{
    .thinking     = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kMuseDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "nvfp4") {
        return WeightsProfile::MuseNvfp4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

EngineOptions Package::resolved_auto_speculative(const EngineOptions& options,
                                                 WeightsProfile) {
    EngineOptions resolved = options;
    if (options.speculative.backend == SpeculativeBackend::Auto) {
        // Muse artifacts carry no draft backend; explicit requests fail later
        // with a clear weights/feature mismatch, Auto resolves to None.
        resolved.speculative.backend = SpeculativeBackend::None;
    }
    return resolved;
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    const EngineOptions resolved = Package::resolved_auto_speculative(options, weights_profile);
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
                                      .token_domain =
                                          static_cast<std::size_t>(detail::Variant::TextConfig::token_domain),
                                      .validate_official_special_ids = false,
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
    dump("embed", view.token_embedding);
    dump("head", view.output_head);
}

} // namespace ninfer::targets::muse_glimmer_30b
