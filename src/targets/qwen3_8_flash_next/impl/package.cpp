// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_8_flash_next/runtime.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include "targets/qwen3_8_flash_next/impl/load/materialized.h"
#include "targets/qwen3_8_flash_next/impl/program_impl.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan,
         bool quantize_output_head_fp8_in, bool quantize_token_embedding_fp8_in)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)),
          quantize_output_head_fp8(quantize_output_head_fp8_in),
          quantize_token_embedding_fp8(quantize_token_embedding_fp8_in) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
    bool quantize_output_head_fp8 = false;
    bool quantize_token_embedding_fp8 = false;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept                 = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept      = default;
LoadPlan::~LoadPlan()                                   = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

class LoadedModel::Impl {
public:
    Impl(BindingPlan plan, artifact::MaterializedArtifact materialized,
         bool quantize_output_head_fp8, bool quantize_token_embedding_fp8)
        : data(std::move(plan), std::move(materialized), quantize_output_head_fp8,
               quantize_token_embedding_fp8) {}

    LoadedModelData data;
};

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel()                                   = default;

} // namespace ninfer::targets::qwen3_8_flash_next::detail

namespace ninfer::targets::qwen3_8_flash_next {

// ---------------------------------------------------------------------------
// Package Static Definitions
// ---------------------------------------------------------------------------

namespace {

constexpr ModelSamplingDefaults kFlashNextDefaults{
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
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kFlashNextDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    detail::validate_identity(identity);
    if (identity.model_id == model_id && identity.weights_id == weights_id) {
        return WeightsProfile::MixedNvfp4Fp8PleInt4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

EngineOptions Package::resolved_auto_speculative(const EngineOptions& options,
                                                 WeightsProfile weights_profile) {
    EngineOptions resolved = options;
    (void)weights_profile;

    // (1) ProposalHead::Auto must never reach the binder or the planner. It is frozen into the
    // artifact binding (LoadFeatures::proposal_head) and read back by the planner at
    // runtime_plan.cpp:174 (`config.proposal_head == ProposalHead::Optimized ? ... : ...`), so an
    // unresolved Auto would take the Full branch there while the load plan recorded "Auto".
    // This checkpoint declares no shortlist head, which is exactly the early return
    // `if (!has_shortlist_head) return ProposalHead::Full;` in
    // qwen3_6::resolved_proposal_head (qwen3_6/export/.../startup_features.h:105) -- the same
    // verdict qwen3_5_9b's package reaches. kMtpShortlistMinimumDrafts is 5 there and this
    // target's planner caps the draft width at 4 (runtime_plan.cpp:73), so the Optimized branch
    // is unreachable on this target by construction, not by choice. Resolved to Full directly,
    // without importing that header, to keep the two facts (no shortlist head; window < 5) here
    // where the target's own bounds are named.
    if (resolved.speculative.proposal_head == ProposalHead::Auto) {
        resolved.speculative.proposal_head = ProposalHead::Full;
    }

    // (2) SpeculativeBackend::Auto. `--spec` DEFAULTS to auto in both front ends
    // (apps/cli/options.cpp:641), and this target's ONLY speculative backend is MTP
    // (Package::plan_load below refuses DFlash/DFlash2 by name), so Auto has exactly one
    // candidate here. It still resolves to None, and that is a decision with two measured
    // supports rather than an oversight:
    //
    //   * Flash-Next's planner has NO adaptive-draft semantics. Every MTP sizing decision in
    //     impl/runtime_plan.cpp is gated on `config.speculative_draft_tokens > 0`, and that field
    //     is `options.speculative.draft_tokens` verbatim (impl/package.cpp, make_sequence_planner)
    //     -- 0 whenever the flag is unset. The runtime agrees: impl/program.cpp:1858 reads
    //     `const bool is_mtp = impl_->has_mtp() && impl_->plan_.config.speculative_draft_tokens > 0;`
    //     So Auto -> Mtp would bind the MTP block (the spliced NVFP4 expert bank, ~3.37 GiB per
    //     tools/convert/qwen3_8_flash_next/splice_mtp.py) and then run ZERO draft rounds.
    //   * The same (Mtp, 0) pair DOES mean "adaptive ladder" for the qwen3_6 family
    //     (apps/cli/options.cpp:607-609, mtp_window_cut.h), so resolving auto -> mtp is right
    //     there and wrong here. The families share the option, not its meaning. This target has
    //     no ladder to fall back on.
    //   * The donor agrees on the default: Package::plan_load takes
    //     `enable_mtp = (backend == SpeculativeBackend::Mtp)`, so upstream Flash-Next never
    //     speculates unless it is asked.
    //
    // An explicit `--spec mtp --draft-tokens k` (k in [1,4], the planner's own bound) is
    // untouched and is the supported way in. The resolution is REPORTED, not silent.
    if (resolved.speculative.backend == SpeculativeBackend::Auto) {
        resolved.speculative.backend = SpeculativeBackend::None;
        std::fprintf(stderr,
                     "ninfer: qwen3_8_flash_next --spec auto resolved to backend=none. This "
                     "target's only speculative backend is MTP, and its MTP path needs an "
                     "explicit draft width (--spec mtp --draft-tokens k, k in [1,4]): with the "
                     "width left at 0 the planner sizes no draft rounds (runtime_plan.cpp) and "
                     "the program's is_mtp is false (program.cpp:1858), so resolving auto to mtp "
                     "would bind the MTP expert bank and never use it.\n");
    }
    return resolved;
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    (void)weights_profile;
    if (options.speculative.backend != SpeculativeBackend::None &&
        options.speculative.backend != SpeculativeBackend::Mtp) {
        throw std::invalid_argument("Flash-Next supports only ordinary decoding and MTP");
    }
    const bool enable_mtp = options.speculative.backend == SpeculativeBackend::Mtp;
    std::uint32_t draft_rows = 32'768;
    if (const char* env = std::getenv("NINFER_FLASH_NEXT_DRAFT_HEAD_ROWS"); env && env[0] != '\0') {
        draft_rows = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
    }
    // The quantize flags have to reach the binder, not just the materializer: they decide whether
    // the BF16 head/embedding is uploaded into the artifact arena or left in the file mapping.
    auto target_plan = detail::bind_artifact(
        binder, detail::LoadFeatures{
                    .vision                       = options.enable_vision,
                    .mtp                          = enable_mtp,
                    .proposal_head                = options.speculative.proposal_head,
                    .draft_head_rows              = draft_rows,
                    .quantize_output_head_fp8     = options.quantize_output_head_fp8,
                    .quantize_token_embedding_fp8 = options.quantize_token_embedding_fp8,
                });
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile, std::move(target_plan), options.quantize_output_head_fp8,
        options.quantize_token_embedding_fp8));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("load plan is empty"); }

    auto impl = std::make_unique<detail::LoadedModel::Impl>(
        std::move(plan.impl_->plan.bindings), std::move(materialized),
        plan.impl_->quantize_output_head_fp8, plan.impl_->quantize_token_embedding_fp8);
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(
        model.impl_->data.frontend,
        qwen3_6::FrontendOptions{
            .vision_enabled           = model.impl_->data.vision.has_value(),
            .max_context              = options.max_context,
            .media_cache_bytes        = options.media_cache_bytes,
            .media_live_bytes         = options.media_live_bytes,
            .media_preprocess_threads = options.media_preprocess_threads,
        });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    (void)device;
    (void)weights_profile;
    const std::uint32_t max_concurrency = std::clamp(options.max_concurrency, 1u, 8u);
    const bool is_mtp = options.speculative.backend == SpeculativeBackend::Mtp;
    const std::uint32_t draft_tokens = is_mtp ? options.speculative.draft_tokens : 0u;
    const std::uint32_t floor_slots =
        detail::flash_next_floor_slots(max_concurrency, draft_tokens);
    const std::uint32_t cont_cap_limit =
        detail::kMaxStateSlots > floor_slots ? (detail::kMaxStateSlots - floor_slots) : 0u;
    const std::uint32_t requested_cont =
        options.context_cache.enabled && options.context_cache.max_private_continuations
            ? *options.context_cache.max_private_continuations
            : 0u;
    const std::uint32_t cont_cap = std::min(requested_cont, cont_cap_limit);
    if (requested_cont > cont_cap_limit) {
        std::fprintf(stderr,
                     "[state-sizer] Continuation capacity clamped from %u to %u (state slot ceiling %u, floor slots %u at concurrency %u).\n",
                     requested_cont, cont_cap, detail::kMaxStateSlots, floor_slots, max_concurrency);
    }
    if (options.context_cache.enabled && cont_cap == 0) {
        std::fprintf(stderr,
                     "[state-sizer] WARNING: continuation capacity is 0 (all %u state slots reserved "
                     "for active decode lanes and rollback). Prefix reuse and context cache are disabled.\n",
                     detail::kMaxStateSlots);
    }
    if (options.kv_cache != KvCacheStorage::BFloat16 &&
        options.kv_cache != KvCacheStorage::Fp8E4M3Row256) {
        throw std::invalid_argument(
            "Flash-Next supports only --kv-dtype bf16 and fp8 (requested unsupported kv-dtype)");
    }
    const std::uint32_t total_state_slots = floor_slots + cont_cap;
    std::uint32_t draft_rows = 32'768;
    if (const char* env = std::getenv("NINFER_FLASH_NEXT_DRAFT_HEAD_ROWS"); env && env[0] != '\0') {
        draft_rows = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
    }
    detail::FlashNextRuntimeConfig config{
        .max_concurrency          = max_concurrency,
        .max_context              = options.max_context,
        .state_slot_capacity      = total_state_slots,
        .continuation_capacity    = cont_cap,
        .prefill_chunk            = options.prefill_chunk,
        .speculative_draft_tokens = draft_tokens,
        .proposal_head            = options.speculative.proposal_head,
        .draft_head_rows          = draft_rows,
        .use_cuda_graph           = options.use_cuda_graph,
        .vision_enabled           = options.enable_vision,
        .max_vision_tokens        = 4096,
        .use_qsa_prefill_mma      = options.use_qsa_prefill_mma, // G18 serve flag; dropped by the upstream merge e650ee62, restored after window 6
        .kv_cache                 = options.kv_cache,
        .gdn_state_storage        = options.gdn_state_storage,
    };
    return SequencePlanner(std::make_unique<detail::SequencePlannerImpl>(config));
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device,
                       const StartupObserver& startup_observer) {
    (void)startup_observer;
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }

    auto program_impl = std::make_unique<detail::ProgramImpl>(
        &model.impl_->data, std::move(plan.impl_->plan), device);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(program_impl)));
}

void Package::export_head_weights(const LoadedModel& model, const char* directory) {
    (void)model;
    (void)directory;
    // The hook's consumer contract is the one qwen3_5_9b implements: `<dir>/<name>_codes.bin`
    // (a groupwise-int payload) plus `<dir>/<name>_scales.bin` (n*2 scale bytes), read off
    // Weight::qdata / Weight::scales. THIS TARGET'S HEAD IS NOT STORED THAT WAY. Its token
    // embedding and output head are BF16 [248'320, 2'560] tensors taken off the materialized
    // backing (impl/load/materialized.cpp:290 and :330, via bf16_weight()), optionally replaced
    // by an FP8-E4M3 row-scaled copy of the same shape (:297, :338). Weight::qdata for those is
    // BF16 / FP8 payload, not int codes, so writing it into `_codes.bin` would produce a file
    // that reads back as garbage under the exact name the consumer expects.
    //
    // Refused by name rather than written wrong. An export that silently writes mis-shaped bytes
    // is the failure this hook exists to rule out, and there is no in-tree consumer of
    // NINFER_EXPORT_HEAD_DIR whose expectation could be consulted instead (checked: no match for
    // the variable in apps/cli, apps/perplexity, apps/serve, src/serve or ninfer-gui.py). Making
    // it real means choosing BF16/FP8 file names and a scale layout -- a new contract, not a
    // port, and not something to invent behind an env var nothing in the tree reads yet.
    throw std::invalid_argument(
        "NINFER_EXPORT_HEAD_DIR is not supported by target 'qwen3_8_flash_next': this target's "
        "token embedding and output head are BF16 (or FP8-E4M3 row-scaled) tensors, not the "
        "groupwise-int qdata/scales pair the hook's <name>_codes.bin + <name>_scales.bin file "
        "shape encodes. Unset the variable; a BF16 dump would need its own file names and scale "
        "layout.");
}

} // namespace ninfer::targets::qwen3_8_flash_next
