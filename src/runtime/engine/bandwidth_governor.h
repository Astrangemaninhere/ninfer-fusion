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
// Enabled only by NINFER_FT_BW_GOV=1; all state is worker-thread local.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ninfer::runtime {

namespace bandwidth_detail {

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

    BandwidthGovernor()
        : BandwidthGovernor(from_env(), bandwidth_detail::env_flag("NINFER_FT_BW_GOV")) {}

    explicit BandwidthGovernor(Tuning tuning, bool enabled) noexcept
        : tuning_(tuning), enabled_(enabled), share_(1.0) {}

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
    double share_ = 1.0;
    double credit_ = 0.0;
    double decode_us_ema_ = 0.0;
    double baseline_us_ = 0.0;
    double ratio_ = 1.0;
    int disturb_streak_ = 0;
    int calm_streak_ = 0;
    std::uint64_t windows_ = 0;
    std::uint64_t last_at_ns_ = 0;
    Counters last_{};
};

} // namespace ninfer::runtime
