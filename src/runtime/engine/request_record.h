#pragma once

#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/generation/generation_budget.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::runtime {

enum class RequestEngineHostPhase : std::uint8_t {
    Boundary,
    CommitOutput,
    Maintenance,
};

struct RequestHostTiming {
    std::uint64_t queue_wait_ns                   = 0;
    std::uint64_t engine_boundary_exposed_ns      = 0;
    std::uint64_t program_submit_exposed_ns       = 0;
    std::uint64_t program_post_exposed_ns         = 0;
    std::uint64_t engine_commit_output_exposed_ns = 0;
    std::uint64_t engine_maintenance_exposed_ns   = 0;
    std::uint64_t device_wait_exposed_ns          = 0;
    std::uint64_t decode_host_exposed_ns          = 0;
    std::uint64_t decode_device_wait_exposed_ns   = 0;
    std::uint64_t prefill_units                   = 0;
    std::uint64_t decode_rounds                   = 0;
    std::uint64_t control_units                   = 0;

    void expose_engine(RequestEngineHostPhase phase, std::uint64_t elapsed_ns,
                       bool decode_member) noexcept {
        switch (phase) {
        case RequestEngineHostPhase::Boundary:
            engine_boundary_exposed_ns += elapsed_ns;
            break;
        case RequestEngineHostPhase::CommitOutput:
            engine_commit_output_exposed_ns += elapsed_ns;
            break;
        case RequestEngineHostPhase::Maintenance:
            engine_maintenance_exposed_ns += elapsed_ns;
            break;
        }
        if (decode_member) { decode_host_exposed_ns += elapsed_ns; }
    }

    void expose_program(ExecutionTiming timing, bool decode_member) noexcept {
        program_submit_exposed_ns += timing.submit_host_ns;
        program_post_exposed_ns += timing.post_host_ns;
        device_wait_exposed_ns += timing.device_wait_ns;
        if (decode_member) {
            decode_host_exposed_ns += timing.host_ns();
            decode_device_wait_exposed_ns += timing.device_wait_ns;
        }
    }

    [[nodiscard]] GenerationEngineTiming public_snapshot() const noexcept {
        constexpr double kNanosecondsToSeconds = 1.0e-9;
        return GenerationEngineTiming{
            .queue_wait_seconds = static_cast<double>(queue_wait_ns) * kNanosecondsToSeconds,
            .engine_boundary_exposed_seconds =
                static_cast<double>(engine_boundary_exposed_ns) * kNanosecondsToSeconds,
            .program_submit_exposed_seconds =
                static_cast<double>(program_submit_exposed_ns) * kNanosecondsToSeconds,
            .program_post_exposed_seconds =
                static_cast<double>(program_post_exposed_ns) * kNanosecondsToSeconds,
            .engine_commit_output_exposed_seconds =
                static_cast<double>(engine_commit_output_exposed_ns) * kNanosecondsToSeconds,
            .engine_maintenance_exposed_seconds =
                static_cast<double>(engine_maintenance_exposed_ns) * kNanosecondsToSeconds,
            .device_wait_exposed_seconds =
                static_cast<double>(device_wait_exposed_ns) * kNanosecondsToSeconds,
            .decode_host_exposed_seconds =
                static_cast<double>(decode_host_exposed_ns) * kNanosecondsToSeconds,
            .decode_device_wait_exposed_seconds =
                static_cast<double>(decode_device_wait_exposed_ns) * kNanosecondsToSeconds,
            .prefill_units = prefill_units,
            .decode_rounds = decode_rounds,
            .control_units = control_units,
        };
    }
};

enum class EngineRequestState : std::uint8_t {
    Waiting,
    Materializing,
    Prefill,
    DecodeReady,
    ControlReady,
    ModelFinished,
};

template <class Package>
struct RequestRecord {
    using Clock          = std::chrono::steady_clock;
    using PreparedPrompt = typename Package::PreparedPrompt;
    using OutputSession  = typename Package::OutputSession;
    using BasePlan       = typename Package::RequestBasePlan;
    using SequenceHandle = typename Package::SequenceHandle;

    RequestRecord(std::uint64_t request_identity, std::uint64_t publication_sequence,
                  PreparedPrompt input, OutputSession output_session, PromptSummary summary,
                  double frontend_seconds, ResolvedRequestOptions request_options,
                  OutputConsumerMode output_consumer, Clock::time_point limit,
                  Clock::time_point submit_time)
        : generation_range(nvtx::Name::Generate, nvtx::Category::Runtime, request_identity),
          id(request_identity), publication_order(publication_sequence), prompt(std::move(input)),
          output(std::move(output_session)), prompt_summary(std::move(summary)),
          prepare_seconds(frontend_seconds), options(std::move(request_options)),
          consumer_mode(output_consumer), deadline(limit), submitted(submit_time) {}

    RequestRecord(const RequestRecord&)            = delete;
    RequestRecord& operator=(const RequestRecord&) = delete;

    [[nodiscard]] bool is_waiting() const noexcept {
        return model_state == EngineRequestState::Waiting;
    }

    [[nodiscard]] bool is_prefilling() const noexcept {
        return model_state == EngineRequestState::Prefill;
    }

    [[nodiscard]] bool is_materializing() const noexcept {
        return model_state == EngineRequestState::Materializing;
    }

    [[nodiscard]] bool is_decode_ready() const noexcept {
        return model_state == EngineRequestState::DecodeReady;
    }

    [[nodiscard]] bool is_control_ready() const noexcept {
        return model_state == EngineRequestState::ControlReady;
    }

    [[nodiscard]] bool is_model_finished() const noexcept {
        return model_state == EngineRequestState::ModelFinished;
    }

    // The process range follows Request ownership across submit, worker, and consumer threads.
    nvtx::ScopedAsyncRange generation_range;
    const std::uint64_t id;
    const std::uint64_t publication_order;
    PreparedPrompt prompt;
    OutputSession output;
    PromptSummary prompt_summary;
    double prepare_seconds = 0.0;
    ResolvedRequestOptions options;
    const OutputConsumerMode consumer_mode;
    Clock::time_point deadline;
    Clock::time_point submitted;
    std::optional<Clock::time_point> first_token;
    bool queue_wait_recorded = false;
    std::optional<GenerationBudget> budget;
    std::optional<BeginSummary> admitted_begin;
    std::optional<BeginSummary> begin;
    std::vector<TokenId> generated;
    std::string content;
    std::string reasoning;
    std::optional<LaneId> lane;
    std::optional<SequenceHandle> sequence;
    std::atomic<bool> cancelled{false};
    EngineRequestState model_state        = EngineRequestState::Waiting;
    bool capture_pending                  = false;
    EngineRequestState post_capture_state = EngineRequestState::Prefill;
    std::optional<FinishReason> terminal_reason;
    // M21 context append (EngineCore::append_context_tokens). The caller arms the token run and
    // `context_append_pending`; the worker services it at the first round boundary that observes it,
    // AHEAD of the decode membership for that boundary, then clears `pending` and sets `done`. Since
    // the service point precedes the first decode round, an append armed right after submit() is
    // ordered before this request's first sampled token; `context_append_rounds` records how many
    // decode rounds had already run when it was serviced, so a lost race is visible, not silent.
    // Read and written under `mutex`, which this record already owns.
    std::vector<TokenId> context_append_tokens;
    bool context_append_pending = false;
    bool context_append_done    = false;
    std::uint64_t context_append_rounds = 0;
    std::exception_ptr context_append_error;

    std::optional<BasePlan> base_plan;
    std::uint64_t remaining_service_work = 0;
    // W6 accounting for the service-work reservation above. A prefill unit is charged one quantum,
    // so the number of quanta a prompt needs depends on the prefill unit in force; the plan's
    // projection is made at the plan's unit (Program::prefill_chunk_capacity()) while
    // bandwidth_governor_.prefill_chunk_for() may shrink the unit mid-stream. Keeping both numbers
    // on the request is what lets the engine re-charge the not-yet-charged span at the unit that is
    // actually in force instead of turning a legal event into a whole-engine stop. See
    // Scheduling::recharge_service_work_for_unit_shrink.
    //   prefill_tokens_done -- prompt tokens this request has already charged a prefill unit for.
    //   service_work_unit   -- the prefill unit the reservation was last reconciled against;
    //                          0 means "not reconciled yet", i.e. still the plan's own unit.
    std::uint64_t prefill_tokens_done = 0;
    std::uint32_t service_work_unit   = 0;
    // The plan's own prompt span -- `N` in projected_service_work() at
    // src/targets/qwen3_6/impl/runtime/request_plan_impl.h:101 -- seeded where the
    // reservation is seeded. It CANNOT be read back from base_plan at prefill time:
    // release_planning_state() resets base_plan at the admission transition
    // (engine_core.h:1694) and ensure_base_plan() only runs for pending requests, so a
    // request that is already prefilling has no base_plan at all.
    std::uint32_t planned_prefill_tokens = 0;
    std::uint64_t backfill_epoch         = 0;
    BackfillClass backfill_class         = BackfillClass::None;
    GenerationTimings generation_timings;
    RequestHostTiming host_timing;
    SpeculativeStats speculative_stats;
    MaterializationDiagnostics materialization_diagnostics;

    std::mutex mutex;
    std::condition_variable cv;
    std::optional<GenerationStart> stream_start;
    std::vector<OutputDelta> events;
    GenerationResult result;
    std::exception_ptr error;
    bool response_done     = false;
    bool consumer_released = false;
    bool capacity_released = false;
};

} // namespace ninfer::runtime
