#include "ninfer/engine.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/tensor.h"
#include "ninfer/ops/scalar.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/types.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"
#include "ninfer/targets/qwen3_6/prepared_prompt.h"
#include "targets/registry.h"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <thread>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {
namespace {

// CUDA resolves the engine's kernel library lazily, on the first launch of any kernel in
// it: measured one cuLibraryLoadData of 3787.9 ms followed by a 1286.2 ms first
// cudaLaunchKernel, and every engine kernel shares that single module. Today that
// resolution lands in the middle of the weight streaming, where nothing is in flight (the
// nsys timeline shows zero copies overlapping it, because the loader thread is the one
// blocked inside the load). Warming it here, from a helper thread and on the engine's own
// stream, overlaps it with the streaming instead.
void warm_kernel_module(DeviceContext& device) noexcept {
    try {
        device.bind_to_current_thread_noexcept();
        std::int32_t* scratch = nullptr;
        if (cudaMalloc(&scratch, sizeof(std::int32_t)) != cudaSuccess) { return; }
        Tensor scalar(scratch, DType::I32, {1});
        ops::set_i32_scalar(scalar, 0, device.stream);
        (void)cudaStreamSynchronize(device.stream);
        (void)cudaFree(scratch);
    } catch (...) {
        // Best effort: a failed warmup only means the first real launch pays the cost.
    }
}

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        // prefill_chunk here is the SCORE TILE -- the initial prefill unit -- and, through
        // layouts_impl.h:1295 (min(prefill_chunk, max_context)) and program_impl.h:782, the
        // capacity the scoring plan sizes its prefill scratch for. It is not a runtime constant
        // on this path either, it is simply immutable: a CausalScoreCore owns no Scheduler and no
        // governor, so nothing shrinks it while the engine runs. Overwriting every caller's value
        // with 3072 made --prefill-chunk (and any API caller) silently inert here, so keep what
        // the caller asked for and normalize only the two spellings layouts_impl.h:853 rejects:
        // 0 (unset -> today's default tile) and a value off the 128-token prefill alignment.
        if (options.prefill_chunk == 0) { options.prefill_chunk = 3072; }
        options.prefill_chunk -= options.prefill_chunk % 128;
        if (options.prefill_chunk < 128) { options.prefill_chunk = 128; }
        // The same immutability decides the MODE, and it is not an assumption: this path has no
        // Scheduler and no governor (the sentence above), so the tile can only be manual -- and a
        // caller who explicitly asked for `dynamic` is refused HERE, by name, rather than accepted
        // and quietly downgraded to a mode that does nothing the caller asked for. A flag that
        // parses and changes nothing is the failure this whole surface exists to avoid.
        if (options.prefill_chunk_mode == PrefillChunkMode::Dynamic) {
            throw std::invalid_argument(
                "--prefill-chunk-mode dynamic needs a Scheduler with a bandwidth governor to do "
                "the adapting, and the CausalScoring path has neither: its score tile is immutable "
                "(engine.cpp normalize_engine_options). Use --prefill-chunk-mode manual, which is "
                "what this path runs.");
        }
        options.prefill_chunk_mode = PrefillChunkMode::Manual;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    // The prefill unit's mode: "an engaged CLI mode > NINFER_FT_BW_GOV > Dynamic", resolved once,
    // here, for the whole engine path. This is the only place the environment is read for the
    // engine, so the governor's own switch, the worker loop's install, the trace line and
    // Engine::options() are all the same answer -- and the answer is written back into `options`
    // because Engine::options() describes the run, not the request (same contract as the settled
    // chunk the memory ladder produces above). EngineCore reads the same resolved field, so this
    // line is what makes --prefill-chunk-mode reach the mechanism rather than the options struct.
    options.prefill_chunk_mode = runtime::BandwidthGovernor::resolve_mode(options.prefill_chunk_mode);
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,16]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    if (!cache.enabled) {
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (cache.max_private_continuations && *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0)) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        // The disabled branch owns the pledged spare ONLY when the one leg that consumes it
        // is armed. A root request materializes read==write PLUS one pledged spare for the
        // `[context-rebuild]` discard destination (`start_sequence`'s `if (state_slots == 2)`
        // in program_impl.h, materialized by `reserve_state_entitlement`), and
        // `prepare_materialization` reserves exactly
        // `demand.reservation_added.device.state_slots` destinations off this pool. That spare
        // is a whole extra StateImage slot per lane, and a slot is not free: `layouts_impl.h`
        // sizes the device pool as `state_image_slots = max_concurrency + device_state_slots`
        // and every linear-attention conv/recurrent tensor carries `slot_count` as its last
        // dimension (`state_image.cpp`, `linear_attention_state.cpp`), so
        // `device_state_slots = concurrency` doubles those tensors. On qwen3.6-27b that is
        // 146.82 MiB per slot (48 layers x 3.00 MiB FP32 recurrent + 2.81 MiB conv + 0.01 MiB
        // continuation hidden), charged to EVERY run whether the leg runs or not.
        //
        // The leg is gated on BOTH environment keys: `NINFER_RECALL_TEXT` selects the
        // text-cargo path and `NINFER_RECALL_TEXT_REBUILD` inside it selects the
        // redirect-to-the-spare arm. The pool follows that same conjunction -- nothing else --
        // so an unarmed run reserves no spare. `plan_request` plans the matching root demand
        // (request_plan_impl.h: the spare is pledged only when the pool holds one beyond
        // `max_concurrency`), and that pairing is what keeps `demand <= pool` true on both
        // sides: the previous unconditional `= 0` refused every root request, and the
        // unconditional `= concurrency` charged every run an extra slot whether or not
        // anything asked for it.
        // [REPREFILLFAST-E7] THE POOL FOLLOWS THE LEG'S GATE -- BOTH WAYS OF SELECTING THE LEG.
        // The paragraph above asserts "The pool follows that same conjunction -- nothing else",
        // which was true while the conjunction was the ONE legacy key.  E4 (`program_impl.h`, the
        // arm site) then added a SECOND way to select the rebuild leg -- `NINFER_RRFAST_ARM=rebuild`
        // -- and did NOT update this arming.  The two halves then disagreed about what "the leg is
        // armed" means: the arm asked for a discard destination the ingress never pledged.  Round 3
        // measured the consequence on the shipping pin `4c33e7ba`: `NINFER_RRFAST_ARM=rebuild` with
        // the legacy key UNSET printed `[rrfast-arm] env=rebuild spare=0 planned=1` and took
        // `[context-append]` on 5/5 independent evaluations.  A switch that selects a leg the
        // ingress never funds is dead code, and the LEG cannot repair it from where it runs: the
        // pledge is made HERE, at ingress, before any sequence -- and therefore any
        // `reserved_state` -- exists.
        // ⚠ NOTHING MOVES FOR ANY PRE-EXISTING STATE, and that is asserted, not hoped:
        //   neither key                      -> false, as before -> no spare, slots = 0
        //   `NINFER_RECALL_TEXT_REBUILD` set -> true,  as before -> the spare, exactly as before
        //   `NINFER_RRFAST_ARM=append`       -> false, as before (a VALUE test, not a presence
        //                                       test: this is the round-1 divergence, and the
        //                                       leg's reading of it is what criterion 4 checks)
        //   `NINFER_RRFAST_ARM=rebuild`      -> true,  NEW      -> the leg the knob asks for can be
        //                                       COMPLETED instead of declining on
        //                                       `no-spare-state-slot-for-discard`
        // The value test is spelled as the arm site spells it (`std::string(...) == "rebuild"`) so
        // the two cannot drift about which strings select the leg; `<cstdlib>` and `<string>` are
        // already included here.
        const char* const recall_arm_env = std::getenv("NINFER_RRFAST_ARM");
        const bool recall_arm_selects_rebuild =
            recall_arm_env != nullptr && std::string(recall_arm_env) == "rebuild";
        // [MTPADAPT-R3] THE SECOND HALF, AND THE REASON IT IS NOT OPTIONAL.
        // The selector in `program_impl.h` (`NINFER_RRFAST_ARM=policy`) asks whether the POOL
        // pledges the discard destination.  The pool is `max_concurrency + device_state_slots`
        // (`layouts_impl.h`, the assignment below), so unless THIS conjunction arms
        // `device_state_slots`, the pool holds exactly `max_concurrency`, the pledge is false on
        // every sequence, and the selector is dead code -- true whenever it is evaluated is the
        // failure mode, not a safety property.  The pledge is made HERE, at ingress, before any
        // sequence exists, which is why the leg cannot repair this from where it runs: measured
        // twice on this station (`reprefillfast` rounds 3 and 4, `spare=0` with the arm asking for
        // it).  Same value test, same spelling as the arm site, so the two cannot drift.
        // [MTPADAPT-R4] THE SECOND HALF OF THE DEFAULT, AND THE REASON IT IS NOT OPTIONAL.
        // The selector in `program_impl.h` now fires on an UNSET `NINFER_RRFAST_ARM` as well as on
        // `=policy`, and its first conjunct asks whether the POOL pledges the discard destination.
        // The pool is `max_concurrency + device_state_slots` (`layouts_impl.h`, the assignment
        // below), so unless THIS conjunction arms `device_state_slots` on the same values, the
        // pledge is false on every sequence and the default is dead code.  The pledge is made HERE,
        // at ingress, before any sequence -- and therefore any `reserved_state` -- exists, which is
        // why the leg cannot repair this from where it runs: measured twice on this station
        // (`reprefillfast` rounds 3 and 4, `spare=0` with the arm asking for it).  The value test
        // is spelled as the arm site spells it, and the two halves are wired on the same two
        // values: UNSET, or the literal string `policy`.
        //
        // ⚠ WHAT THIS COSTS, SAID OUT LOUD.  For any run that sets `NINFER_RECALL_TEXT`, an unset
        // arm now reserves the pledged spare -- a whole extra StateImage slot per lane, which is
        // `146.82 MiB` on qwen3.6-27b (the note above) -- whether or not a recall plan exists for
        // that run.  Runs that do not set `NINFER_RECALL_TEXT` are unaffected: the outer conjunction
        // below is unchanged, so `discard_spare_armed` stays false and `device_state_slots` stays 0.
        // `NINFER_RRFAST_ARM=none` (or any non-`policy` value) turns the grant back off with it.
        const bool recall_arm_env_unset = recall_arm_env == nullptr;
        const bool recall_arm_names_policy =
            recall_arm_env != nullptr && std::string(recall_arm_env) == "policy";
        const bool recall_arm_selects_policy =
            recall_arm_env_unset || recall_arm_names_policy;
        const bool discard_spare_armed = std::getenv("NINFER_RECALL_TEXT") != nullptr &&
                                         (std::getenv("NINFER_RECALL_TEXT_REBUILD") != nullptr ||
                                          recall_arm_selects_rebuild ||
                                          recall_arm_selects_policy);
        cache.device_state_slots                = discard_spare_armed ? concurrency : 0U;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        return options;
    }

    cache.device_state_slots            = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes               = cache.max_shared_prefixes.value_or(concurrency);
    cache.max_long_anchors_per_continuation = cache.max_long_anchors_per_continuation.value_or(2U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         targets::qwen3_6::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    targets::qwen3_6::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }
std::vector<TokenId> PreparedPrompt::prompt_token_ids() const {
    if (impl_ == nullptr) { return {}; }
    return targets::qwen3_6::PreparedPromptAccess::view(impl_->value).token_ids;
}

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
        virtual void append_context_tokens(std::span<const TokenId> tokens) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

        void append_context_tokens(std::span<const TokenId> tokens) override {
            submission_.append_context_tokens(tokens);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    void append_context_tokens(std::span<const TokenId> tokens) {
        state_->append_context_tokens(tokens);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

void GenerationHandle::append_context_tokens(std::span<const TokenId> tokens) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    impl_->append_context_tokens(tokens);
}

class Engine::Impl {
public:
    using Core27      = runtime::EngineCore<targets::Qwen3_6_27BInstance>;
    using Core35      = runtime::EngineCore<targets::Qwen3_6_35BA3BInstance>;
    using CoreMuse    = runtime::EngineCore<targets::MuseGlimmer30BInstance>;
    using ScoreCore27 = runtime::CausalScoreCore<targets::Qwen3_6_27BInstance>;
    using ScoreCore35 = runtime::CausalScoreCore<targets::Qwen3_6_35BA3BInstance>;
    using ScoreCoreMuse = runtime::CausalScoreCore<targets::MuseGlimmer30BInstance>;
    using Core9B        = runtime::EngineCore<targets::Qwen3_5_9BInstance>;
    using ScoreCore9B   = runtime::CausalScoreCore<targets::Qwen3_5_9BInstance>;
    using CoreSpark      = runtime::EngineCore<targets::SparkX2_5_4BInstance>;
    using ScoreCoreSpark = runtime::CausalScoreCore<targets::SparkX2_5_4BInstance>;
    using Core = std::variant<std::monostate, std::unique_ptr<Core27>, std::unique_ptr<Core35>,
                              std::unique_ptr<CoreMuse>, std::unique_ptr<ScoreCore27>,
                              std::unique_ptr<ScoreCore35>, std::unique_ptr<ScoreCoreMuse>,
                              std::unique_ptr<Core9B>, std::unique_ptr<ScoreCore9B>,
                              std::unique_ptr<CoreSpark>, std::unique_ptr<ScoreCoreSpark>>;

    explicit Impl(EngineOptions engine_options)
        : options(normalize_engine_options(std::move(engine_options))), device(options.device) {
        device.yarn_enabled = options.yarn_enabled;
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        // jthread, not thread: the load path can throw (bad options, a reservation that
        // cannot be met, a rejected KV budget), and a joinable std::thread destroyed during
        // unwinding calls std::terminate - which swallowed those errors as a bare
        // "terminate called without an active exception" abort. jthread joins on scope exit.
        std::jthread module_warmup([&device = device] { warm_kernel_module(device); });
        auto constructed  = targets::construct_target(options, device);
        // The memory ladder inside construct_target() may have lowered the prefill unit to fit the
        // device runtime budget, and everything downstream -- the sequence plan, the Program's
        // prefill_chunk_capacity, the prefill loop -- was built at THAT value. `options` here is
        // the request. Adopting the settled value is what makes Engine::options() describe the run
        // instead of the request: apps/perplexity/main.cpp:376 reads it back to report the run's
        // chunk, and before this adoption the report named a chunk the engine never used whenever
        // the ladder fired. 0 means the target reported nothing, in which case the request stands.
        if (constructed.effective_prefill_chunk != 0) {
            options.prefill_chunk = constructed.effective_prefill_chunk;
        }
        active            = std::move(constructed.active);
        load              = std::move(constructed.load);
        sampling_defaults = constructed.sampling_defaults;
        core              = std::visit(
            [&](auto& target_ptr) -> Core {
                using Instance =
                    typename std::remove_reference_t<decltype(target_ptr)>::element_type;
                if constexpr (std::is_same_v<Instance, targets::Qwen3_6_27BInstance>) {
                    if (options.purpose == EnginePurpose::CausalScoring) {
                        return std::make_unique<ScoreCore27>(*target_ptr, device);
                    }
                    return std::make_unique<Core27>(*target_ptr, device, options,
                                                                 std::move(constructed.context_cost));
                } else if constexpr (std::is_same_v<Instance,
                                                    targets::Qwen3_6_35BA3BInstance>) {
                    if (options.purpose == EnginePurpose::CausalScoring) {
                        return std::make_unique<ScoreCore35>(*target_ptr, device);
                    }
                    return std::make_unique<Core35>(*target_ptr, device, options,
                                                    std::move(constructed.context_cost));
                } else if constexpr (std::is_same_v<Instance, targets::MuseGlimmer30BInstance>) {
                    if (options.purpose == EnginePurpose::CausalScoring) {
                        return std::make_unique<ScoreCoreMuse>(*target_ptr, device);
                    }
                    return std::make_unique<CoreMuse>(*target_ptr, device, options,
                                                      std::move(constructed.context_cost));
                } else if constexpr (std::is_same_v<Instance, targets::Qwen3_5_9BInstance>) {
                    if (options.purpose == EnginePurpose::CausalScoring) {
                        return std::make_unique<ScoreCore9B>(*target_ptr, device);
                    }
                    return std::make_unique<Core9B>(*target_ptr, device, options,
                                                    std::move(constructed.context_cost));
                } else if constexpr (std::is_same_v<Instance, targets::SparkX2_5_4BInstance>) {
                    if (options.purpose == EnginePurpose::CausalScoring) {
                        return std::make_unique<ScoreCoreSpark>(*target_ptr, device);
                    }
                    return std::make_unique<CoreSpark>(*target_ptr, device, options,
                                                       std::move(constructed.context_cost));
                } else {
                    // A new target family must be dispatched HERE, explicitly.  This branch
                    // used to be an unconditional `else` that built the Muse core, so a
                    // fourth `ActiveTarget` alternative was silently handed to
                    // `EngineCore<MuseGlimmer30BInstance>` -- a wrong-core bind that only
                    // surfaced as a wall of conversion errors inside the Muse template.  The
                    // static_assert is what makes the next family fail with one readable
                    // sentence at the line that has to change.
                    static_assert(!std::is_same_v<Instance, Instance>,
                                  "new target instance reached the ActiveTarget variant "
                                  "without an implicit engine-core dispatch branch in "
                                  "src/runtime/engine/engine.cpp");
                    throw std::logic_error("target instance has no engine core");
                }
            },
            active);
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        core.emplace<std::monostate>();
        try {
            device.synchronize();
        } catch (...) {}
    }

    EngineOptions options;
    DeviceContext device;
    targets::ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    Core core;
};

Engine::Engine(EngineOptions options) : impl_(std::make_shared<Impl>(std::move(options))) {}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const SamplingMode sampling_mode =
        input.options.enable_thinking ? SamplingMode::Thinking : SamplingMode::NonThinking;
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare(std::move(input), control);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, sampling_mode, std::move(prepared)));
        },
        impl_->active);
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            if (token_ids.size() > target_ptr->capacity) {
                throw RequestError(RequestErrorKind::ContextLengthExceeded,
                                   context_capacity_error(token_ids.size(), target_ptr->capacity));
            }
            auto prepared      = target_ptr->loaded->frontend.prepare_tokens(std::move(token_ids),
                                                                             allow_prefix_identity);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, SamplingMode::Thinking, std::move(prepared)));
        },
        impl_->active);
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.tokenize_text(text);
        },
        impl_->active);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    PreparedPrompt prompt      = prepare_tokens(std::move(tokens), false);
    const std::size_t expected = prompt.summary().prompt_tokens - first_target;
    std::vector<float> result  = std::visit(
        [&](auto& core) -> std::vector<float> {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore27>> ||
                          std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore35>> ||
                          std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCoreMuse>> ||
                          std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore9B>> ||
                          std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCoreSpark>>) {
                return core->score(std::move(prompt.impl_->value), first_target);
            } else {
                throw std::logic_error("Engine scoring core is unavailable");
            }
        },
        impl_->core);
    if (result.size() != expected) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.count_tokens(std::move(input), control);
        },
        impl_->active);
}

PromptCapabilities Engine::prompt_capabilities() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.prompt_capabilities();
        },
        impl_->active);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }

            void append_context_tokens(std::span<const TokenId>) {
                throw std::logic_error(
                    "a request with no output budget has no context-append target");
            }
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt                     = prompt_summary;
        immediate.result.finish_reason              = FinishReason::OutputLimit;
        immediate.result.thinking.configured_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds    = prepare_seconds;
        immediate.result.timings.total_seconds      = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& core) -> GenerationHandle {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore27>> ||
                                 std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore35>> ||
                                 std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCoreMuse>> ||
                                 std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore9B>> ||
                                 std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCoreSpark>>) {
                throw std::logic_error("Engine generation core is unavailable");
            } else {
                auto submission =
                    core->submit(std::move(prompt.impl_->value), prompt_summary, prepare_seconds,
                                 std::move(resolved_options), consumer_mode, pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->core);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode).wait(sink, cancellation);
}

void Engine::append_context_tokens(GenerationHandle& handle, std::span<const TokenId> tokens) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (!handle) { throw std::invalid_argument("GenerationHandle is empty"); }
    handle.append_context_tokens(tokens);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> MemorySummary {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->memory_summary();
            }
        },
        impl_->core);
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.media_cache_summary();
        },
        impl_->active);
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> RuntimeStats {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->runtime_stats();
            }
        },
        impl_->core);
}

EngineFailureState Engine::failure_state() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> EngineFailureState {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return EngineFailureState{};
            } else if constexpr (requires { core->failure_state(); }) {
                return core->failure_state();
            } else {
                // Causal scoring cores never serve requests, so they cannot be
                // the poisoned engine the caller is asking about.
                return EngineFailureState{};
            }
        },
        impl_->core);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                core->reset_memory_peaks();
            }
        },
        impl_->core);
}

void Engine::reload_kv_storage(
    std::array<KvCacheStorage, kKvLayerStorageSlots> layer_storage,
    std::array<bool, kKvLayerStorageSlots> residual_layers,
    std::array<bool, kKvLayerStorageSlots> layer_storage_set) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("reload_kv_storage requires a Generation Engine");
    }
    // Generation runs on caller threads with no engine-internal queue; the serve layer's
    // drain (no active request admitted) is what makes this program swap safe.
    impl_->device.bind_to_current_thread();
    impl_->options.kv_layer_storage          = layer_storage;
    // Table and mask are set together, from the same caller-side parse: a mask that
    // said "written" for a slot this table does not carry would be a per-layer
    // request for a storage nobody chose, and a table without its mask silently
    // turns every bf16 slot back into "inherit the global dtype".
    impl_->options.kv_layer_storage_set      = layer_storage_set;
    impl_->options.kv_layer_storage_explicit = true;
    impl_->options.kv_residual_layers        = residual_layers;
    impl_->options.kv_residual_explicit      = true;
    targets::replan_target_kv(impl_->active, impl_->options, impl_->device);
    // The replan destroyed the previous Program, so every catalogued
    // continuation handle (context cache / prefix reuse) is stale. Drop the
    // catalog: the next planning pass must not hand a stale owner to the new
    // Program's checkpoint cost model.
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                if constexpr (requires { core->clear_context_catalog_after_replan(); }) {
                    core->clear_context_catalog_after_replan();
                }
            }
        },
        impl_->core);
}

void Engine::recover() {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("recover requires a Generation Engine");
    }
    if (!failure_state().failed) { return; }
    impl_->device.bind_to_current_thread();
    // A failed unit may have left a sticky CUDA error. Clear it and probe the context before
    // touching the Program: an unusable context means recovery must not pretend to succeed.
    (void)cudaGetLastError();
    void* probe                    = nullptr;
    const cudaError_t probe_status = cudaMalloc(&probe, 4096);
    if (probe_status != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::runtime_error(std::string("Engine::recover: CUDA context unusable: ") +
                                 cudaGetErrorString(probe_status));
    }
    (void)cudaFree(probe);
    // The rebuild runs inside recover() after the failed worker has exited: the Program swap
    // (KV pool, graph state, cached prefixes) must not race the worker that poisoned it.
    std::visit(
        [this](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                if constexpr (requires { core->recover([] {}); }) {
                    core->recover([this, &core] {
                        targets::replan_target_kv(impl_->active, impl_->options, impl_->device);
                        if constexpr (requires { core->clear_context_catalog_after_replan(); }) {
                            core->clear_context_catalog_after_replan();
                        }
                    });
                }
            }
        },
        impl_->core);
}

} // namespace ninfer
