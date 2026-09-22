#pragma once

// FreeToken step-3 (W6): decode-bandwidth governor for prefill pacing.
//
// The worker loop alternates decode and prefill units 1:1 (Scheduler::choose_execution). On a long
// prompt that ratio hands a large share of the memory system to prefill and inflates decode latency
// for every co-resident request. This governor watches the only bandwidth evidence the engine owns
// without new device timers — the per-class device-wait counters and committed decode tokens — and
// throttles the prefill share while decode latency sits above its measured noise floor.
//
// Signal. decode_us_per_token = decode_device_wait_ns / committed_decode_tokens, smoothed by an EMA.
// The reference is a noise floor that drops instantly to a new minimum and creeps up slowly, so it
// tracks the uncontended decode latency even when the session starts under contention. The ratio
// ema/floor > tol_hi for `streak` windows halves the prefill share (credit budget), and the ratio
// falling below tol_lo doubles it back, both bounded by [min_share, 1]. share == 1 reproduces the
// historical 1:1 alternation exactly, so the governor is inert until it throttles.
//
// Consumption. The engine asks prefill_allowed() at every decision point and calls charge_prefill()
// when a prefill unit actually runs. Credit accrues per completed decode round, so share == 0.5
// admits one prefill unit per two decode units. With no decode work the scheduler still runs prefill
// (starvation avoidance lives in choose_execution, not here).
//
// ENABLED BY DEFAULT, and now the thing that is enabled has a NAME. The governor is what shrinks
// the prefill unit (prefill_chunk_for below), so while it was gated on NINFER_FT_BW_GOV=1 the
// shrink existed in the code and never ran: the prefill chunk was a fixed value in every default
// configuration. The environment keeps both spellings of the operator surface --
// NINFER_FT_BW_GOV=0 is the kill switch, =1 forces it on -- and NINFER_FT_BW_TRACE=1 prints every
// change of the prefill unit the governor installs (see trace_prefill_chunk for exactly what that
// does and does not promise), so "on" and "off" are distinguishable from outside the process.
//
// THE TWO MODES. The user's ruling is that the prefill unit must be adjustable BOTH ways -- "prefill
// chunk 要能动态调整或者手动调整两种模式" -- so the switch above is exposed as a mode
// (PrefillChunkMode, include/ninfer/types.h) with two names: `dynamic` is this governor owning the
// unit (the default), `manual` is the unit pinned to whatever the caller asked for, with the
// governor constructed disabled so nothing re-derives it. The spellings, the mode -> enabled mapping
// and the "CLI > environment > default" resolution all live in THIS file (mode_name / parse_mode /
// adapts / resolve_mode below), which is what keeps the front ends, the trace and the mechanism from
// drifting: `--prefill-chunk-mode manual` reaches exactly the same state as NINFER_FT_BW_GOV=0, and
// the trace line names which mode produced the number it printed.
//
// All state is worker-thread local. The signal is the worker's own device-wait time, so on a fully
// asynchronous path where the worker never blocks on the device the metric collapses to ~0 and the
// governor stays inert (safe degradation, no false throttling).

#include "ninfer/types.h"

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::runtime {

namespace bandwidth_detail {

// W6: the wire the prefill-unit shrink needs from a target Program. Named ONCE, here, because the
// engine's `if constexpr` guard and the failure branch that goes with it have to test the *same*
// predicate -- a guard and the check behind it disagreeing is how a mechanism ends up with nothing
// behind it. runtime.h records that this guard was false for every target before the two
// forwarders landed, so the shrink was unreachable code and not merely env-gated, and the failure
// was silent: every target still compiled, linked and passed its tests. engine_core.h now fails the
// build when the predicate is false, tests/test_prefill_chunk_wiring.cpp asserts it for each
// target package, and both name this one definition.
template <class Program>
inline constexpr bool exposes_prefill_chunk_wire =
    requires(Program& program, std::uint32_t chunk) {
        program.set_prefill_chunk(chunk);
        { program.prefill_chunk_capacity() } noexcept -> std::same_as<std::uint32_t>;
    };

inline const char* env_raw(const char* name) {
    return std::getenv(name);
}

inline bool env_flag(const char* name) {
    const char* value = env_raw(name);
    return value != nullptr && value[0] == '1';
}

inline double env_double(const char* name, double fallback) {
    const char* value = env_raw(name);
    if (value == nullptr || value[0] == '\0') { return fallback; }
    const double parsed = std::atof(value);
    return parsed > 0.0 ? parsed : fallback;
}

inline int env_int(const char* name, int fallback) {
    const char* value = env_raw(name);
    if (value == nullptr || value[0] == '\0') { return fallback; }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : fallback;
}

// Tri-state boolean: unset -> nullopt (whatever the caller's default is), an explicit off spelling
// -> false, anything else -> true. env_flag() above keeps its strict "== \"1\"" meaning for the
// readers that depend on it; this one exists because the flag that used to mean "on" has to be
// able to say "off" now that the default flipped.
inline std::optional<bool> env_tristate(const char* name) {
    const char* value = env_raw(name);
    if (value == nullptr || value[0] == '\0') { return std::nullopt; }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "off") == 0 || std::strcmp(value, "no") == 0 ||
        std::strcmp(value, "FALSE") == 0 || std::strcmp(value, "OFF") == 0 ||
        std::strcmp(value, "NO") == 0) {
        return false;
    }
    return true;
}

} // namespace bandwidth_detail

class BandwidthGovernor {
public:
    // Monotonic engine counters the governor reads. decode_tokens is the committed-token total, so
    // the latency metric stays comparable across decode batch sizes.
    struct Counters {
        std::uint64_t decode_device_ns  = 0;
        std::uint64_t decode_tokens     = 0;
        std::uint64_t decode_rounds     = 0;
        std::uint64_t prefill_device_ns = 0;
        std::uint64_t prefill_units     = 0;
    };

    struct Tuning {
        // Disturbance band around the decode noise floor. ratio > tol_hi throttles; < tol_lo
        // recovers; in between is a dead band that resets both streaks.
        double tol_hi = 1.35;
        double tol_lo = 1.15;
        // Consecutive windows required before a share change (hysteresis).
        int streak = 2;
        // Lowest prefill share. Never zero: a starved long prompt must still make progress while
        // other requests decode.
        double min_share = 0.125;
        // Noise-floor creep-up per window; the floor itself drops without smoothing.
        double baseline_alpha = 0.02;
        // EMA smoothing for the per-token decode latency.
        double ema_alpha = 0.35;
        std::uint64_t window_ns = 50'000'000ULL;
        // Upper bound on banked prefill units, so a throttle release cannot burst.
        double max_credit = 2.0;
    };

    struct Snapshot {
        bool enabled                 = false;
        double share                 = 1.0;
        double ratio                 = 1.0;
        double credit                = 0.0;
        double decode_us_per_token   = 0.0;
        double baseline_us_per_token = 0.0;
        std::uint64_t windows        = 0;
    };

    BandwidthGovernor() : BandwidthGovernor(from_env(), resolve_enabled()) {}

    explicit BandwidthGovernor(Tuning tuning, bool enabled) noexcept
        : tuning_(tuning), enabled_(enabled), share_(1.0) {}

    // "CLI > environment > default" for the switch itself. A caller that knows the operator's
    // choice passes it in (that is the CLI layer); an unset override defers to NINFER_FT_BW_GOV;
    // and the default is ON, because a prefill unit that never adapts is exactly the fixed chunk
    // this mechanism was added to remove.
    [[nodiscard]] static bool resolve_enabled(std::optional<bool> cli_override = std::nullopt) {
        if (cli_override.has_value()) { return *cli_override; }
        return bandwidth_detail::env_tristate("NINFER_FT_BW_GOV").value_or(true);
    }

    // ---- The two modes, their spellings and their one resolution point ------------------
    // The mode is the operator-facing name of the switch above: Dynamic == the governor may act,
    // Manual == it may not, and the prefill unit is whatever the caller asked for. Both the
    // classes that consume it and the two front ends that spell it read the table from here, so
    // "the flag says manual" and "the mechanism is off" cannot drift apart.
    [[nodiscard]] static const char* mode_name(PrefillChunkMode mode) noexcept {
        return mode == PrefillChunkMode::Manual ? "manual" : "dynamic";
    }

    // The one mapping mode -> the switch. Anything that wants to know whether the engine may
    // move the unit asks here, rather than spelling `== Dynamic` again at a second site.
    [[nodiscard]] static bool adapts(PrefillChunkMode mode) noexcept {
        return mode == PrefillChunkMode::Dynamic;
    }

    // The CLI spelling. Case-sensitive on purpose: the tree's other value flags are
    // (--kv-bits-mode joint|split|ceiling, --spec mtp|dflash), and accepting "Manual" here while
    // refusing "SPLIT" there would be two rules for one kind of argument.
    [[nodiscard]] static PrefillChunkMode parse_mode(std::string_view text) {
        if (text == "dynamic") { return PrefillChunkMode::Dynamic; }
        if (text == "manual") { return PrefillChunkMode::Manual; }
        throw std::invalid_argument(
            "prefill chunk mode must be dynamic|manual, got '" + std::string(text) +
            "': dynamic lets the bandwidth governor shrink the prefill unit while decode is "
            "contended, manual pins it to --prefill-chunk for the whole run");
    }

    // "CLI > environment > default" for the MODE. Called once, by
    // normalize_engine_options(), so that the governor, the worker loop, the trace and
    // Engine::options() cannot disagree about which mode is running. An engaged override wins
    // even against an explicit NINFER_FT_BW_GOV, which is what makes the flag a real second
    // entry point rather than a decoration on the environment variable.
    [[nodiscard]] static PrefillChunkMode resolve_mode(
        std::optional<PrefillChunkMode> cli_override = std::nullopt) {
        if (cli_override.has_value()) { return *cli_override; }
        return resolve_enabled() ? PrefillChunkMode::Dynamic : PrefillChunkMode::Manual;
    }

    static Tuning from_env() {
        Tuning tuning;
        tuning.tol_hi         = bandwidth_detail::env_double("NINFER_FT_BW_TOL_HI", tuning.tol_hi);
        tuning.tol_lo         = bandwidth_detail::env_double("NINFER_FT_BW_TOL_LO", tuning.tol_lo);
        tuning.streak         = bandwidth_detail::env_int("NINFER_FT_BW_STREAK", tuning.streak);
        tuning.min_share      = bandwidth_detail::env_double("NINFER_FT_BW_MIN_SHARE", tuning.min_share);
        tuning.baseline_alpha = bandwidth_detail::env_double("NINFER_FT_BW_BASE_ALPHA", tuning.baseline_alpha);
        tuning.ema_alpha      = bandwidth_detail::env_double("NINFER_FT_BW_EMA_ALPHA", tuning.ema_alpha);
        tuning.window_ns      = static_cast<std::uint64_t>(
            bandwidth_detail::env_double("NINFER_FT_BW_WINDOW_MS", 50.0) * 1.0e6);
        tuning.max_credit     = bandwidth_detail::env_double("NINFER_FT_BW_MAX_CREDIT", tuning.max_credit);
        if (tuning.tol_lo > tuning.tol_hi) { tuning.tol_lo = tuning.tol_hi; }
        if (tuning.min_share > 1.0) { tuning.min_share = 1.0; }
        return tuning;
    }

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    [[nodiscard]] double share() const noexcept { return share_; }

    [[nodiscard]] Snapshot snapshot() const noexcept {
        return Snapshot{.enabled                 = enabled_,
                        .share                   = share_,
                        .ratio                   = ratio_,
                        .credit                  = credit_,
                        .decode_us_per_token     = decode_us_ema_,
                        .baseline_us_per_token   = baseline_us_,
                        .windows                 = windows_};
    }

    // One observation window per call; windows shorter than tuning_.window_ns are ignored, so this
    // is cheap enough to run at every worker-loop boundary.
    void observe(std::uint64_t now_ns, const Counters& counters) noexcept {
        if (!enabled_) { return; }
        if (last_at_ns_ == 0) {
            last_     = counters;
            last_at_ns_ = now_ns;
            return;
        }
        const std::uint64_t dt = now_ns > last_at_ns_ ? now_ns - last_at_ns_ : 0;
        if (dt < tuning_.window_ns) { return; }

        const std::uint64_t decode_tokens  = counters.decode_tokens - last_.decode_tokens;
        const std::uint64_t decode_rounds  = counters.decode_rounds - last_.decode_rounds;
        const std::uint64_t prefill_units  = counters.prefill_units - last_.prefill_units;
        const std::uint64_t decode_ns      = counters.decode_device_ns - last_.decode_device_ns;
        const std::uint64_t prefill_ns     = counters.prefill_device_ns - last_.prefill_device_ns;
        last_      = counters;
        last_at_ns_ = now_ns;
        ++windows_;

        // Credit accrual is per decode round, so share is the prefill:decode admission ratio.
        credit_ = std::min(tuning_.max_credit,
                           credit_ + share_ * static_cast<double>(decode_rounds));

        if (decode_tokens == 0) { return; }
        const double decode_us =
            static_cast<double>(decode_ns) / static_cast<double>(decode_tokens) * 1.0e-3;
        decode_us_ema_ = decode_us_ema_ > 0.0
                             ? decode_us_ema_ + (decode_us - decode_us_ema_) * tuning_.ema_alpha
                             : decode_us;

        if (baseline_us_ <= 0.0) {
            baseline_us_ = decode_us_ema_;
            report("baseline", decode_us, prefill_units, prefill_ns);
            return;
        }
        if (decode_us_ema_ < baseline_us_) {
            baseline_us_ = decode_us_ema_;
        } else {
            baseline_us_ += (decode_us_ema_ - baseline_us_) * tuning_.baseline_alpha;
        }
        ratio_ = baseline_us_ > 0.0 ? decode_us_ema_ / baseline_us_ : 1.0;

        if (ratio_ > tuning_.tol_hi) {
            ++disturb_streak_;
            calm_streak_ = 0;
        } else if (ratio_ < tuning_.tol_lo) {
            ++calm_streak_;
            disturb_streak_ = 0;
        } else {
            disturb_streak_ = 0;
            calm_streak_    = 0;
        }

        if (disturb_streak_ >= tuning_.streak && share_ > tuning_.min_share) {
            share_ = std::max(tuning_.min_share, share_ * 0.5);
            credit_ = 0.0;
            disturb_streak_ = 0;
            report("throttle", decode_us, prefill_units, prefill_ns);
        } else if (calm_streak_ >= tuning_.streak && share_ < 1.0) {
            share_ = std::min(1.0, share_ * 2.0);
            calm_streak_ = 0;
            report("recover", decode_us, prefill_units, prefill_ns);
        }
    }

    // True when a prefill unit may run now. share == 1 keeps the historical alternation.
    [[nodiscard]] bool prefill_allowed() const noexcept {
        if (!enabled_) { return true; }
        if (share_ >= 1.0) { return true; }
        return credit_ >= 1.0;
    }

    // Called only when a prefill unit actually runs.
    void charge_prefill() noexcept {
        if (!enabled_ || share_ >= 1.0) { return; }
        credit_ = credit_ > 1.0 ? credit_ - 1.0 : 0.0;
    }

    // NINFER_FT_BW_TRACE=1: one stderr line per *change* of the prefill unit the governor
    // installs. What the line reports is precise and narrower than "the unit the run used": the
    // engine writes the value into the Program in the same worker iteration, immediately before
    // this call, so `installed=1 chunk=N` means N is what the Program holds at that instant, and it
    // is therefore the unit the *next* prefill unit will use -- the loop recomputes and re-installs
    // on every iteration, so no prefill ever runs under a value this trace has not reported. It is
    // NOT a promise that a prefill unit consumed that value: a change printed after the last
    // prefill unit of a run (one request: prompt prefill is over before the governor can move,
    // because observe() ignores windows with no committed decode tokens) is only ever seen by
    // later prefill work -- the decode-time MTP/DFlash draft prefill, or another request. The
    // counters in the line are what settles that for a given log: prefill_units does not advance
    // again after such a change iff nothing ever ran with it. `installed=0` reports the Program's
    // own capacity value, which the engine did not write this iteration -- that is manual mode, and
    // it is also the only line a manual run prints.
    //
    // `mode=` is what makes the two modes comparable FROM THE LOG rather than from the options the
    // caller passed: it is derived from the governor's own enabled state, which the engine
    // constructs from the resolved mode, so a log that says mode=manual cannot have been produced
    // by a run that throttled. The two modes are then told apart by both fields at once:
    //   dynamic -- installed=1, and chunk drops below capacity the moment the governor throttles;
    //   manual  -- installed=0 and exactly one line, with chunk == capacity == the requested value.
    //
    // One comparison per worker iteration when off. `chunk=` here is what the engine
    // INSTALLED; `consumed=` in the note_prefill_unit() line below is what the loop USED, and
    // both are needed: the first can be right while the second is wrong, which is the failure
    // this file has already paid for once (a value that is set and read by nobody looks exactly
    // like a value that is honoured -- see exposes_prefill_chunk_wire above).
    void trace_prefill_chunk(std::uint32_t chunk, std::uint32_t capacity, bool installed,
                             const Counters& counters) noexcept {
        if (!trace_ || chunk == traced_chunk_) { return; }
        traced_chunk_ = chunk;
        std::fprintf(stderr,
                     "[ft] bw mode=%s chunk=%u capacity=%u installed=%d share=%.3f win=%llu "
                     "prefill_units=%llu decode_tokens=%llu\n",
                     mode_name(enabled_ ? PrefillChunkMode::Dynamic : PrefillChunkMode::Manual),
                     chunk, capacity, installed ? 1 : 0, share_,
                     static_cast<unsigned long long>(windows_),
                     static_cast<unsigned long long>(counters.prefill_units),
                     static_cast<unsigned long long>(counters.decode_tokens));
    }

    // ---- What a prefill unit actually consumed ---------------------------------------------
    // Counters::prefill_units counts units and cumulative_stats_ sums their tokens, but nothing
    // ever paired a unit with the number of prompt tokens THAT unit sliced off. So the one claim
    // the two modes make about the loop -- `manual` slices --prefill-chunk on every unit,
    // `dynamic` slices less once the governor throttles -- was only ever checked against the
    // value the engine INSTALLED (trace_prefill_chunk above), never against the value the loop
    // USED. That gap is the same shape as the dead wire this file already records: a value that
    // is written and read by nobody looks identical to one that is honoured.
    //
    // note_installed_unit() is called from the ONE place that writes the Program, in the worker
    // iteration that is about to run a unit, so the value it records is the one that unit runs
    // under. note_prefill_unit() is called from the ONE place a prefill unit completes
    // (EngineCore::resolve_prefill_progress, next to `++host_work.prefill_units`), with
    // Program::advance_prefill's own processed_prompt_tokens for that unit -- the loop's return
    // value, not a re-derivation of the caller's option. Together they print the pairing:
    //
    //   [ft] bw unit mode=manual chunk=1024 capacity=1024 installed=0 consumed=1024 units=3 \
    //                tokens=3072 min=1024 max=1024 over=0
    //
    // Under `manual` consumed must be exactly the requested value on every full unit; under
    // `dynamic` it must follow the shrunk chunk down. An engine that installed the right number
    // and then sliced a different one prints the same `chunk=` as before and a different
    // `consumed=`, which is the failure this closes.
    //
    // `over=` counts units that consumed MORE tokens than the value installed while they ran.
    // That is not a tolerance question: the loop is handed `min(prefill_chunk, end - cursor)`,
    // so over > 0 means the loop stopped reading the field the engine writes -- the exact
    // regression a per-unit readback exists to catch, and the one driven red by
    // tests/test_prefill_chunk_mode.cpp::test_unit_tokens_pair_with_the_installed_chunk().
    //
    // The accumulator is readable without the trace (unit_tokens()) so a caller can assert on
    // the fingerprint instead of grepping a log.
    struct UnitTokens {
        std::uint64_t units  = 0;
        std::uint64_t tokens = 0;
        std::uint64_t over   = 0;
        std::uint32_t min    = 0;
        std::uint32_t max    = 0;
        std::uint32_t last   = 0;
    };

    // Called once per worker iteration from the engine's install site, immediately before the
    // write (and before trace_prefill_chunk), so this is the unit the NEXT prefill unit runs
    // under -- which is the unit recorded when that unit completes in note_prefill_unit().
    void note_installed_unit(std::uint32_t chunk, std::uint32_t capacity,
                             bool installed) noexcept {
        last_unit_chunk_     = chunk;
        last_unit_capacity_  = capacity;
        last_unit_installed_ = installed;
    }

    // Called once per COMPLETED prefill unit, with the tokens that unit processed.
    void note_prefill_unit(std::uint32_t consumed) noexcept {
        if (units_.units == 0 || consumed < units_.min) { units_.min = consumed; }
        if (consumed > units_.max) { units_.max = consumed; }
        // Valid in both modes: when the engine did not write, the Program holds `capacity`, and
        // last_unit_chunk_ is that capacity, so it is still the unit that ran.
        if (last_unit_chunk_ != 0 && consumed > last_unit_chunk_) { ++units_.over; }
        ++units_.units;
        units_.tokens += consumed;
        units_.last = consumed;
        if (!trace_) { return; }
        std::fprintf(stderr,
                     "[ft] bw unit mode=%s chunk=%u capacity=%u installed=%d consumed=%u "
                     "units=%llu tokens=%llu min=%u max=%u over=%llu\n",
                     mode_name(enabled_ ? PrefillChunkMode::Dynamic : PrefillChunkMode::Manual),
                     last_unit_chunk_, last_unit_capacity_, last_unit_installed_ ? 1 : 0,
                     consumed, static_cast<unsigned long long>(units_.units),
                     static_cast<unsigned long long>(units_.tokens), units_.min, units_.max,
                     static_cast<unsigned long long>(units_.over));
    }

    [[nodiscard]] const UnitTokens& unit_tokens() const noexcept { return units_; }

    // W6: the prefill unit itself shrinks with the admitted share, so a decoding request waits for
    // one small unit instead of a full chunk. Returns `base` unchanged while unthrottled. `base` is
    // ProgramImplCore::prefill_chunk_capacity -- the *ceiling* the workspace and the persistent
    // buffers were sized for at startup -- and the result is always inside [128, base], because
    // shrinking is safe and growing is not (program.h set_prefill_chunk clamps again on top). So
    // prefill_chunk is an initial value with a capacity above it, not one fixed constant.
    //
    // In MANUAL mode `enabled_` is false, so this is the identity for every `base`: that is what
    // makes the manual value verbatim rather than merely the starting point of a range, and it is
    // the half of the contract a test has to pin separately (there is no share that can move it).
    //
    // ON THE `!enabled_` GUARD HERE BEING REDUNDANT WITH observe()'s -- conclusion, not a TODO.
    // Deleting this guard alone changes nothing observable, and that was MEASURED: the injection
    // produced zero failing assertions across the suite. The reason is that `share_` is the only
    // state that can move the result and observe() is its only writer, so while `enabled_` is
    // false the `share_ >= 1.0` term below already short-circuits on a value that cannot move.
    // Deleting observe()'s guard instead is NOT behaviour-preserving (share_ would follow the
    // counters while disabled), so the two guards are not interchangeable: the load-bearing one
    // is observe()'s and this one is defence in depth, kept because a second barrier on the path
    // that must never move is cheaper than the reasoning that it is unnecessary.
    //
    // The invariant behind that conclusion is PINNED, not argued:
    // tests/test_prefill_chunk_mode.cpp::test_disabled_governor_ignores_a_hostile_feed() drives a
    // DISABLED governor with the counter feed that takes an enabled one to min_share and requires
    // share_/windows_/the unit to be untouched. It is red when observe()'s guard is removed, so
    // the redundancy claim has a negative control behind it rather than only a story.
    [[nodiscard]] std::uint32_t prefill_chunk_for(std::uint32_t base) const noexcept {
        constexpr std::uint32_t kAlignment = 128;
        if (!enabled_ || share_ >= 1.0 || base <= kAlignment) { return base; }
        std::uint32_t scaled = static_cast<std::uint32_t>(static_cast<double>(base) * share_);
        scaled -= scaled % kAlignment;
        if (scaled < kAlignment) { scaled = kAlignment; }
        return scaled < base ? scaled : base;
    }

private:
    void report(const char* kind, double decode_us, std::uint64_t prefill_units,
                std::uint64_t prefill_ns) noexcept {
        const double prefill_us = prefill_units == 0
                                      ? 0.0
                                      : static_cast<double>(prefill_ns) /
                                            static_cast<double>(prefill_units) * 1.0e-3;
        std::fprintf(stderr,
                     "[ft] bw %s share=%.3f ratio=%.2f decode_us=%.2f base_us=%.2f "
                     "prefill_us=%.2f win=%llu\n",
                     kind, share_, ratio_, decode_us, baseline_us_, prefill_us,
                     static_cast<unsigned long long>(windows_));
    }

    Tuning tuning_{};
    bool enabled_ = false;
    // NINFER_FT_BW_TRACE=1; see trace_prefill_chunk().
    bool trace_                 = bandwidth_detail::env_flag("NINFER_FT_BW_TRACE");
    std::uint32_t traced_chunk_ = 0;
    double share_               = 1.0;
    double credit_ = 0.0;
    double decode_us_ema_ = 0.0;
    double baseline_us_ = 0.0;
    double ratio_ = 1.0;
    int disturb_streak_ = 0;
    int calm_streak_ = 0;
    std::uint64_t windows_ = 0;
    std::uint64_t last_at_ns_ = 0;
    Counters last_{};
    // The value the engine last made current in the Program (note_installed_unit), so a prefill
    // unit completing later in the same worker iteration can be paired with the chunk it ran
    // under. Overwritten on every iteration, not only when the value changes.
    std::uint32_t last_unit_chunk_    = 0;
    std::uint32_t last_unit_capacity_ = 0;
    bool last_unit_installed_         = false;
    UnitTokens units_{};
};

} // namespace ninfer::runtime
