#pragma once

#include "ninfer/types.h"

#include <array>
#include <chrono>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    // mtplogx: the ids this prompt was tokenized to. Read-only exposure of the sequence the engine
    // already holds, so a measurement consumer can record the input side of a run; the evaluation
    // path never reads it.
    [[nodiscard]] std::vector<TokenId> prompt_token_ids() const;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

    // M21: append a run of ALREADY-KNOWN tokens to this running request and prefill them, so the
    // model can attend to them from the next round -- the "recall" leg: a segment that was never
    // resident (or was reclaimed) becomes part of the context without being re-decoded one token at
    // a time. The tokens are INPUT: they are not reported in GenerationResult::generated_token_ids,
    // do not consume the output budget, and take no control-token accounting.
    //
    // Ordering: call this BEFORE wait(). The append is serviced at the next engine round boundary
    // and ahead of that request's decode membership, so an append armed immediately after submit()
    // lands before the first sampled token. It blocks until serviced, and throws if the request has
    // already finished. Preconditions and therefore the failure modes are Program's: the sequence
    // must be Active with no prefill in flight and execution_frontier + tokens.size() <= capacity.
    void append_context_tokens(std::span<const TokenId> tokens);

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for repeatable correctness and performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    // Artifact-tokenizer raw-text encoding. No chat template or implicit special token is added.
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;

    // Returns log p(tokens[i] | tokens[0..i)) for i in [first_target,tokens.size()).
    [[nodiscard]] std::vector<float> score_tokens(std::vector<TokenId> tokens,
                                                  std::uint32_t first_target);

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously with a fixed output consumer mode. Destroying an
    // unconsumed handle cancels its request; wait() owns result consumption and may run
    // independently from GPU execution. Streaming mode requires a non-null sink in wait() and
    // publishes one exact GenerationStart before output deltas; Aggregate mode requires a null
    // sink.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputConsumerMode consumer_mode                       = OutputConsumerMode::Aggregate,
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    // M21: the operator-facing entry for "append a segment of tokens and prefill it mid-run".
    // Routes to GenerationHandle::append_context_tokens; see that declaration for the contract.
    void append_context_tokens(GenerationHandle& handle, std::span<const TokenId> tokens);

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    // W16: engine health. After the worker hits an unexpected exception the
    // engine permanently rejects work (serve answers 503); this reports why so
    // the serve layer can surface it instead of a silent unavailable.
    [[nodiscard]] EngineFailureState failure_state() const;
    void reset_memory_peaks() noexcept;

    // Re-runs the KV sequence plan with a new per-layer storage table (FreeToken
    // observe -> decide -> apply loop). Only layer storage/residual tables may change:
    // weights, concurrency and context capacities are reused from startup options.
    // Contract: no request may be in flight or admitted until this returns — the serve
    // layer drains before calling. The old KV pool is destroyed before the new plan is
    // resolved, so a failed replan leaves the Engine without a usable Program.
    void reload_kv_storage(
        std::array<KvCacheStorage, kKvLayerStorageSlots> layer_storage,
        std::array<bool, kKvLayerStorageSlots> residual_layers = {},
        // Which slots `layer_storage` was actually written for (see
        // EngineOptions::kv_layer_storage_set). Defaulted to all-false, which is
        // the pre-mask contract: a BFloat16 entry inherits the global dtype. Pass
        // the mask from the same parse that produced layer_storage when the spec
        // named a slot as bf16, or the replan will silently inherit --kv-dtype on
        // exactly the layers the spec asked to make BF16.
        std::array<bool, kKvLayerStorageSlots> layer_storage_set = {});

    // W16 P1: last-resort recovery from a poisoned engine (failure_state().failed). Rebuilds the
    // Program exactly like reload_kv_storage (the old KV pool and every cached prefix are lost),
    // then respawns the worker. Same contract: no request may be in flight or admitted until this
    // returns — the serve layer drains before calling. Throws if the context itself is unusable.
    void recover();

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
