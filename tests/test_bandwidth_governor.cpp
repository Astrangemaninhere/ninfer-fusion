// W6 unit test: the decode-bandwidth governor and its scheduler consumption contract.
// Host-only; no CUDA. The fake request mirrors tests/test_admission_policy.cpp so that
// Scheduler<Request>::choose_execution can be instantiated without an engine.
#include "runtime/engine/bandwidth_governor.h"
#include "runtime/engine/scheduler.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

struct SchedulerRequest {
    using SequenceHandle = std::uint64_t;

    struct Budget {
        std::uint32_t tokens = 0;

        [[nodiscard]] std::uint32_t remaining() const noexcept { return tokens; }
    };

    struct Output {
        std::uint32_t model_tokens = 0;
        std::vector<ninfer::TokenId> control;

        [[nodiscard]] std::uint32_t
        model_token_budget_remaining(std::uint32_t total) const noexcept {
            return std::min(total, model_tokens);
        }

        [[nodiscard]] std::span<const ninfer::TokenId> pending_control_tokens() const noexcept {
            return control;
        }
    };

    enum class State : std::uint8_t { Decode, Control };

    [[nodiscard]] bool is_decode_ready() const noexcept { return state == State::Decode; }

    [[nodiscard]] bool is_control_ready() const noexcept { return state == State::Control; }

    std::uint64_t id                              = 0;
    std::uint64_t remaining_service_work          = 1;
    std::uint64_t backfill_epoch                  = 0;
    ninfer::runtime::BackfillClass backfill_class = ninfer::runtime::BackfillClass::None;
    State state                                   = State::Decode;
    bool capture_pending                          = false;
    std::optional<Budget> budget;
    std::optional<SequenceHandle> sequence;
    Output output;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// The environment IS the engine's reader's front end (bandwidth_detail::env_raw is std::getenv),
// so setting it through the process environment is the same switch the operator sets -- there is
// no test-only override to drift from it. Same shape as tests/test_prefill_chunk_mode.cpp.
void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    ::setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    ::unsetenv(name);
#endif
}

using ninfer::runtime::BandwidthGovernor;

// Synthetic window: a fixed number of decode tokens/rounds plus some prefill work.
struct Window {
    double us_per_token = 1.0;
    std::uint64_t tokens = 100;
    std::uint64_t rounds = 100;
    std::uint64_t prefill_units = 50;
};

class Feed {
public:
    Feed(BandwidthGovernor::Tuning tuning, bool enabled)
        : governor_(tuning, enabled), window_ns_(tuning.window_ns) {}

    void step(const Window& window) {
        counters_.decode_device_ns +=
            static_cast<std::uint64_t>(window.us_per_token * 1000.0 *
                                       static_cast<double>(window.tokens));
        counters_.decode_tokens += window.tokens;
        counters_.decode_rounds += window.rounds;
        counters_.prefill_device_ns += window.prefill_units * 2'000'000ULL;
        counters_.prefill_units += window.prefill_units;
        now_ns_ += window_ns_;
        governor_.observe(now_ns_, counters_);
    }

    [[nodiscard]] BandwidthGovernor& governor() noexcept { return governor_; }

private:
    BandwidthGovernor governor_;
    BandwidthGovernor::Counters counters_{};
    std::uint64_t window_ns_ = 50'000'000ULL;
    std::uint64_t now_ns_    = 1'000'000'000ULL;
};

int test_disabled() {
    int failures = 0;
    BandwidthGovernor::Tuning tuning;
    tuning.window_ns = 1'000'000ULL;
    BandwidthGovernor off(tuning, false);
    BandwidthGovernor::Counters counters{};
    for (int i = 0; i < 200; ++i) {
        counters.decode_device_ns += 100'000'000ULL; // absurdly contended
        counters.decode_tokens += 100;
        counters.decode_rounds += 100;
        off.observe(1'000'000ULL * static_cast<std::uint64_t>(i + 1), counters);
    }
    failures += check(off.snapshot().share == 1.0, "disabled governor changed its share");
    failures += check(off.prefill_allowed(), "disabled governor blocked prefill");
    off.charge_prefill();
    failures += check(off.prefill_allowed(), "disabled governor charged a prefill credit");
    return failures;
}

int test_throttle_and_recover() {
    int failures = 0;
    BandwidthGovernor::Tuning tuning;
    tuning.window_ns = 1'000'000ULL;
    tuning.streak    = 2;
    Feed feed(tuning, true);
    BandwidthGovernor& governor = feed.governor();

    Window calm;
    calm.us_per_token = 1.0;
    for (int i = 0; i < 6; ++i) { feed.step(calm); }
    const BandwidthGovernor::Snapshot settled = governor.snapshot();
    failures += check(settled.share == 1.0, "calm decode throttled prefill");
    failures += check(settled.baseline_us_per_token > 0.9 &&
                          settled.baseline_us_per_token < 1.1,
                      "decode noise floor did not settle on the calm latency");

    Window contended;
    contended.us_per_token = 2.0;
    int throttle_window = -1;
    for (int i = 0; i < 12 && throttle_window < 0; ++i) {
        feed.step(contended);
        if (governor.snapshot().share < 1.0) { throttle_window = i; }
    }
    failures += check(throttle_window >= 0, "contended decode never throttled prefill");
    failures += check(governor.snapshot().share <= 0.5, "throttle step was not halving");

    for (int i = 0; i < 40; ++i) { feed.step(contended); }
    const double floor_share = governor.snapshot().share;
    failures += check(floor_share == tuning.min_share,
                      "sustained contention did not reach the minimum prefill share");
    failures += check(floor_share > 0.0, "minimum prefill share is zero (starvation)");

    // Credit accounting at the floor: one window of decode grants share * rounds credits, capped.
    failures += check(governor.prefill_allowed(),
                      "banked credits did not admit a prefill unit at the share floor");
    governor.charge_prefill();
    governor.charge_prefill();
    failures += check(!governor.prefill_allowed(),
                      "prefill units were admitted without credit");

    Window recovered;
    recovered.us_per_token = 1.0;
    int recover_window = -1;
    for (int i = 0; i < 60 && recover_window < 0; ++i) {
        feed.step(recovered);
        if (governor.snapshot().share >= 1.0) { recover_window = i; }
    }
    failures += check(recover_window >= 0, "calm decode did not restore the full prefill share");

    // The prefill unit shrinks with the share so a decoding request waits for a small unit.
    failures += check(governor.prefill_chunk_for(3072) == 3072,
                      "unthrottled share changed the prefill chunk");
    return failures;
}

int test_prefill_chunk_scaling() {
    int failures = 0;
    BandwidthGovernor::Tuning tuning;
    tuning.window_ns = 1'000'000ULL;
    Feed feed(tuning, true);
    BandwidthGovernor& governor = feed.governor();

    BandwidthGovernor disabled(tuning, false);
    failures += check(disabled.prefill_chunk_for(3072) == 3072,
                      "disabled governor changed the prefill chunk");

    // Unthrottled: the chunk is untouched.
    failures += check(governor.prefill_chunk_for(3072) == 3072,
                      "fresh governor changed the prefill chunk");

    Window calm;
    calm.us_per_token = 1.0;
    for (int i = 0; i < 6; ++i) { feed.step(calm); }

    Window contended;
    contended.us_per_token = 2.0;
    for (int i = 0; i < 12; ++i) { feed.step(contended); }
    const double share = governor.snapshot().share;
    const std::uint32_t chunk = governor.prefill_chunk_for(3072);
    failures += check(chunk < 3072, "throttled governor did not shrink the prefill chunk");
    failures += check(chunk % 128 == 0, "shrunk prefill chunk is not 128-aligned");
    failures += check(chunk >= 128, "shrunk prefill chunk fell below the alignment");
    failures += check(share < 1.0, "contention did not throttle before the chunk check");

    // A chunk already at the alignment floor is left alone.
    failures += check(governor.prefill_chunk_for(128) == 128,
                      "chunk floor was scaled below the alignment");
    return failures;
}

int test_choice_contract() {
    using Scheduler      = ninfer::runtime::Scheduler<SchedulerRequest>;
    using ExecutionAction = typename Scheduler::ExecutionAction;
    Scheduler scheduler;
    int failures = 0;
    failures += check(scheduler.choose_execution(false, true, false, false) ==
                          ExecutionAction::Prefill,
                      "throttled prefill starved when no decode work existed");
    failures += check(scheduler.choose_execution(false, true, true, false) ==
                          ExecutionAction::Prefill,
                      "throttled prefill starved after a decode unit");
    failures += check(scheduler.choose_execution(true, true, false, true) ==
                          ExecutionAction::Decode,
                      "unthrottled alternation stopped favouring decode after prefill");
    failures += check(scheduler.choose_execution(true, true, true, true) ==
                          ExecutionAction::Prefill,
                      "unthrottled alternation stopped alternating to prefill");
    failures += check(scheduler.choose_execution(true, true, false, false) ==
                          ExecutionAction::Decode,
                      "throttled prefill ignored available decode work");
    failures += check(scheduler.choose_execution(true, true, true, false) ==
                          ExecutionAction::Decode,
                      "throttled prefill ran while decode was ready");
    failures += check(scheduler.choose_execution(true, false, true, false) ==
                          ExecutionAction::Decode,
                      "decode was skipped when prefill was not runnable");
    failures += check(scheduler.choose_execution(false, false, true, false) ==
                          ExecutionAction::Wait,
                      "idle worker did not wait");
    return failures;
}

// ---- a NUMERIC environment request is read by the same rule as the tri-state ----------------
// Every case is TWO-SIDED, which is the only shape that can fail: an unreadable value must be
// refused AND the refusal must NAME the variable and QUOTE the offending value, and the
// documented values must still resolve QUIETLY and be READ. A reader that refused everything
// satisfies the first half and fails the second; one that accepted everything fails the first.
// ON THE PRE-IMAGE HEADER every "was accepted instead of refused" line below fails -- the silent
// readers returned the default for all of them -- and main() returns 1.
int test_env_numeric_refused_by_name() {
    const char* kDbl = "NINFER_FT_BW_TOL_HI";
    const char* kInt = "NINFER_FT_BW_STREAK";
    const struct {
        const char* name;
        const char* value;
    } unreadable[] = {
        {kDbl, "abc"}, {kDbl, "0"},   {kDbl, "-1"},  {kDbl, "1.5x"}, {kDbl, " 1.5"}, {kDbl, "1.5 "},
        {kDbl, "inf"}, {kDbl, "nan"}, {kDbl, "1e"},  {kInt, "0"},    {kInt, "-2"},   {kInt, "2.5"},
        {kInt, "2 "},  {kInt, "two"}, {kInt, "0x10"}, {kInt, "99999999999"},
    };
    int failures = 0;
    for (const auto& one : unreadable) {
        unset_env(kDbl);
        unset_env(kInt);
        set_env(one.name, one.value);
        bool threw = false;
        std::string what;
        try {
            const BandwidthGovernor::Tuning tuning = BandwidthGovernor::from_env();
            (void)tuning;
        } catch (const std::exception& error) {
            threw = true;
            what  = error.what();
        }
        {
            const std::string message = std::string(one.name) + "=" + one.value +
                                        " was accepted instead of refused; a value that cannot be "
                                        "read must not be promoted to a meaning";
            failures += check(threw, message.c_str());
        }
        {
            const std::string message = std::string("the refusal for ") + one.name + "=" +
                                        one.value +
                                        " does not name the variable it came from: " + what;
            failures += check(what.find(one.name) != std::string::npos, message.c_str());
        }
        {
            const std::string message = std::string("the refusal for ") + one.name + "=" +
                                        one.value +
                                        " does not quote the offending value: " + what;
            failures += check(what.find(one.value) != std::string::npos, message.c_str());
        }
        unset_env(one.name);
    }

    // The accepted side must still be quiet, and it must be READ rather than replaced by a default.
    {
        unset_env(kDbl);
        unset_env(kInt);
        set_env(kDbl, "2.5");
        set_env(kInt, "3");
        bool threw = false;
        BandwidthGovernor::Tuning tuning;
        try {
            tuning = BandwidthGovernor::from_env();
        } catch (const std::exception&) {
            threw = true;
        }
        failures += check(!threw, "a documented numeric value was refused");
        failures += check(tuning.tol_hi == 2.5, "NINFER_FT_BW_TOL_HI=2.5 was not read");
        failures += check(tuning.streak == 3, "NINFER_FT_BW_STREAK=3 was not read");
        unset_env(kDbl);
        unset_env(kInt);
    }

    // An empty value is "unset", not a refusal: the old reader's default must not have moved.
    set_env(kDbl, "");
    set_env(kInt, "");
    {
        bool threw = false;
        BandwidthGovernor::Tuning tuning;
        try {
            tuning = BandwidthGovernor::from_env();
        } catch (const std::exception&) {
            threw = true;
        }
        failures += check(!threw, "an EMPTY numeric knob is refused instead of defaulting");
        failures += check(tuning.tol_hi == 1.35 && tuning.streak == 2,
                          "an empty value changed the DEFAULTS");
    }
    unset_env(kDbl);
    unset_env(kInt);

    // The trace switch is the same reader family and the same rule.
    unset_env("NINFER_FT_BW_TRACE");
    set_env("NINFER_FT_BW_TRACE", "trace");
    {
        bool threw = false;
        try {
            (void)ninfer::runtime::bandwidth_detail::env_flag("NINFER_FT_BW_TRACE");
        } catch (const std::exception&) {
            threw = true;
        }
        failures += check(threw, "NINFER_FT_BW_TRACE=trace was read as OFF instead of refused");
    }
    unset_env("NINFER_FT_BW_TRACE");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_disabled();
    failures += test_throttle_and_recover();
    failures += test_prefill_chunk_scaling();
    failures += test_choice_contract();
    failures += test_env_numeric_refused_by_name();
    if (failures == 0) { std::cout << "BANDWIDTH_GOVERNOR_TEST PASS\n"; }
    return failures == 0 ? 0 : 1;
}
