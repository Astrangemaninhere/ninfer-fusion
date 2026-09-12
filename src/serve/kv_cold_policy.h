#pragma once

// S34 (N4 / user Q4): CLOSED-LOOP hot/warm/cold residency policy — pure
// decision layer over the FreeToken observation tap. Host-only, std-only:
// unit-testable with plain g++ (_collab/C_s34_policy_test.cpp), no CUDA, no
// engine headers.
//
// LOOP (see _collab/C_s34_closed_loop.md for the full design):
//   observe:  ops::ft::snapshot() cumulative sums each cycle -> EXACT
//             per-window means via deltas (no change to ft_stats.h needed);
//   smooth:   EWMA per layer (alpha = ewma_alpha, default 0.5);
//   decide:   cold candidates = coldest eligible layers, |cold| <= cold_cap,
//             eligibility = not deep-protected AND observed enough this
//             window AND dwell/streak gates passed;
//   act:      dry mode logs "[ft][cold:dry] would ..." and touches nothing;
//             live mode hands the hot spec to the existing reload path
//             (cold-planned layers emitted with their nvfp4 hot window, the
//             S30 convention; pool-level residency is the S30 consumer).
//
// INVARIANT (never violated, machine-checked by invariant_violations()):
//   a layer is demoted to cold ONLY IF its EWMA energy was at or below the
//   cold cut in EACH of the last `stable_cycles` windows, it has dwelled in
//   its current residency >= min_dwell_cycles, it is not deep-protected, it
//   was observed >= min_window_rounds times this window, and the resulting
//   cold set still fits cold_cap. Unobserved layers are never demoted.
//
// S46 (defect F3b, closed): an IDLE window is not an observation of energy 0.
//   The live tap reports a layer with no new rounds through
//   window_from_cumulative(sum, n, sum, n) as (mean 0.0, rounds 0); feeding
//   that fabricated zero into the EWMA dragged a layer whose true energy never
//   changed below the cut (and the poisoned value also lowered the cut
//   itself), so the layer was demoted -- confirmed by S39 F3b. Now such a
//   window does not move the EWMA, and any window below the confidence floor
//   does not count as a window at or below the cut: it breaks the streak, per
//   the "in EACH of the last `stable_cycles` windows" clause above. Windows
//   with 0 < rounds < min_window_rounds still move the EWMA: their mean is a
//   real (if noisy) delta, only their demotion vote is withheld. A genuine
//   zero-energy window (rounds > 0, delta sum 0) also still smooths normally.
//   A below-confidence window cannot promote a cold layer out of the pool
//   either (otherwise going idle would eject a cold layer).
//   Still open (not this fix): a layer ABSENT from `observations` leaves its
//   streak frozen (S39 finding F2).

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ninfer::serve {

struct ColdObservation {
    int layer = 0;
    double window_mean_l = 0.0;   // exact mean over this window (delta math)
    std::uint64_t window_rounds = 0;  // observations in this window
};

struct ColdPolicyConfig {
    int full_attn_layers = 16;
    double deep_frac = 0.2;        // last 20% keep nvfp4 (deep protection)
    int cold_cap = 0;              // pool capacity in layers (S30 effective_cold_pages)
    double demote_quantile = 0.25; // coldest quartile is the candidate pool
    int stable_cycles = 2;         // streak below cut required to demote
    int min_dwell_cycles = 3;      // min cycles in current residency
    int min_window_rounds = 32;    // observation confidence floor per window
    double ewma_alpha = 0.5;
    int max_cold = 0;              // 0 -> cold_cap (rate limit per decision)
};

// Mutable decision state carried across cycles (the damping memory).
struct ColdPolicyState {
    std::map<int, double> ewma;         // layer -> smoothed energy
    std::map<int, int> below_streak;    // consecutive CONFIDENT windows at/below cut
    std::map<int, int> dwell;           // cycles in current residency
    std::vector<int> cold;              // current cold set (ascending)
};

struct ColdDecision {
    std::vector<int> cold_layers;      // ascending
    std::string hot_spec_hint;         // dtype spec; cold layers -> "nvfp4"
    std::vector<std::string> reasons;  // forensics, one line per gate
    bool changed = false;
    // Evidence captured AT DECISION TIME (before the dwell bookkeeping resets
    // it) so invariant_violations() can machine-check the demotion gates.
    std::vector<int> prev_cold;
    std::map<int, int> streak_at_decision;
    std::map<int, int> dwell_at_decision;
    std::map<int, std::uint64_t> rounds_this_window;
};

// Cumulative-snapshot delta math: from (sum, count) at two cycle boundaries
// the window mean/rounds are EXACT (ft::snapshot never resets, by design).
inline void window_from_cumulative(double sum0, std::uint64_t count0,
                                   double sum1, std::uint64_t count1,
                                   double& window_mean, std::uint64_t& rounds) {
    if (count1 <= count0) {
        window_mean = 0.0;
        rounds = 0;
        return;
    }
    window_mean = (sum1 - sum0) / static_cast<double>(count1 - count0);
    rounds = count1 - count0;
}

// The cold cut: demote_quantile of the EWMA over non-deep, observed layers.
inline double cold_cut(const std::map<int, double>& ewma,
                       const ColdPolicyConfig& cfg) {
    const int deep_start = static_cast<int>(
        static_cast<double>(cfg.full_attn_layers) * (1.0 - cfg.deep_frac));
    std::vector<double> values;
    for (const auto& [layer, value] : ewma) {
        if (layer < deep_start) { values.push_back(value); }  // deep layers
    }                                                          // never gate the cut
    if (values.empty()) { return 0.0; }
    std::sort(values.begin(), values.end());
    const double q = cfg.demote_quantile;
    const std::size_t idx = static_cast<std::size_t>(
        q * static_cast<double>(values.size() - 1) + 0.5);
    return values[std::min(idx, values.size() - 1)];
}

// One decision cycle. `observations` are THIS window's stats; state carries
// the memory. Returns the decision (cold set + hot spec hint + forensics).
inline ColdDecision decide_cold_residency(
    const std::vector<ColdObservation>& observations,
    const ColdPolicyConfig& cfg, ColdPolicyState& state,
    const std::vector<std::pair<int, double>>& energy_for_spec) {
    ColdDecision decision;
    const int deep_start = static_cast<int>(
        static_cast<double>(cfg.full_attn_layers) * (1.0 - cfg.deep_frac));
    const int cap = cfg.cold_cap > 0 ? cfg.cold_cap : 0;
    const int limit = cfg.max_cold > 0 ? std::min(cfg.max_cold, cap) : cap;

    // 1) EWMA update + streaks (only over observed layers this window).
    //    S46: an IDLE window is not an observation of energy 0. The tap
    //    reports a layer with no new rounds as
    //    window_from_cumulative(sum, n, sum, n) = (mean 0.0, rounds 0), and
    //    folding that fabricated zero into the EWMA dragged a layer whose true
    //    energy never changed below the cut (and the poisoned value lowered
    //    the cut itself). A GENUINE zero-energy window still smooths normally:
    //    it has rounds > 0 (new rounds were observed, the delta sum is 0).
    std::map<int, bool> confident_this_window;
    for (const ColdObservation& obs : observations) {
        if (obs.layer < 0 || obs.layer >= cfg.full_attn_layers) { continue; }
        if (obs.window_rounds == 0) { continue; }  // idle: no EWMA movement
        if (obs.window_rounds >=
            static_cast<std::uint64_t>(cfg.min_window_rounds)) {
            confident_this_window[obs.layer] = true;
        }
        auto it = state.ewma.find(obs.layer);
        const double next = it == state.ewma.end()
                                ? obs.window_mean_l
                                : cfg.ewma_alpha * obs.window_mean_l +
                                      (1.0 - cfg.ewma_alpha) * it->second;
        state.ewma[obs.layer] = next;
    }
    const double cut = cold_cut(state.ewma, cfg);
    decision.reasons.push_back("cut=" + std::to_string(cut));
    for (const ColdObservation& obs : observations) {
        if (obs.layer < 0 || obs.layer >= cfg.full_attn_layers) { continue; }
        int& streak = state.below_streak[obs.layer];
        if (!confident_this_window.count(obs.layer)) {
            // S46: neither an idle window nor a below-confidence window can
            // witness "at or below the cut", so it does not extend the streak
            // (it breaks it) -- the invariant demands a confident window in
            // EACH of the last `stable_cycles` windows.
            streak = 0;
            continue;
        }
        const double value = state.ewma[obs.layer];
        if (value <= cut) {
            ++streak;
        } else {
            streak = 0;
        }
    }

    // 2) Eligible candidates: non-deep, enough observation, dwell satisfied,
    //    streak satisfied. Promotion needs no dwell (quality first).
    // Capture the decision-time evidence FIRST (dwell resets below would
    // otherwise clobber the proof the invariant checker needs).
    decision.prev_cold = state.cold;
    decision.streak_at_decision = state.below_streak;
    decision.dwell_at_decision = state.dwell;
    for (const ColdObservation& obs : observations) {
        decision.rounds_this_window[obs.layer] = obs.window_rounds;
    }
    std::vector<std::pair<double, int>> candidates;  // (ewma, layer)
    for (const ColdObservation& obs : observations) {
        if (obs.layer < 0 || obs.layer >= deep_start) { continue; }
        if (obs.window_rounds < static_cast<std::uint64_t>(cfg.min_window_rounds)) {
            decision.reasons.push_back("skip layer=" + std::to_string(obs.layer) +
                                       " rounds=" + std::to_string(obs.window_rounds) +
                                       " < min_window_rounds");
            continue;
        }
        const bool is_cold = std::find(state.cold.begin(), state.cold.end(),
                                       obs.layer) != state.cold.end();
        if (!is_cold) {
            const int streak = state.below_streak.count(obs.layer)
                                   ? state.below_streak[obs.layer] : 0;
            const int dwell = state.dwell.count(obs.layer)
                                  ? state.dwell[obs.layer] : 0;
            if (streak < cfg.stable_cycles) {
                decision.reasons.push_back(
                    "wait layer=" + std::to_string(obs.layer) + " streak=" +
                    std::to_string(streak) + " < " +
                    std::to_string(cfg.stable_cycles));
                continue;
            }
            if (dwell < cfg.min_dwell_cycles) {
                decision.reasons.push_back(
                    "dwell layer=" + std::to_string(obs.layer) + " " +
                    std::to_string(dwell) + " < " +
                    std::to_string(cfg.min_dwell_cycles));
                continue;
            }
            candidates.emplace_back(state.ewma[obs.layer], obs.layer);
        }
    }
    std::sort(candidates.begin(), candidates.end());

    // 3) Keep promoted layers (cold -> hot) whenever their EWMA leaves the cut
    //    for stable_cycles (streak == 0 for that long, tracked as above_streak
    //    implicitly by below_streak == 0 at decision time is too weak; we
    //    promote when below_streak == 0 for the layer, i.e. last window above).
    //    S46: "streak == 0" now also happens when the decisive window was
    //    below confidence (it breaks the streak), so promotion must require
    //    that window to be a real, confident observation -- otherwise going
    //    IDLE would eject a cold layer from the pool.
    std::vector<int> next_cold;
    for (const int layer : state.cold) {
        const bool confident_now = confident_this_window.count(layer) != 0;
        const int streak = state.below_streak.count(layer)
                               ? state.below_streak[layer] : 0;
        if (confident_now && streak <= 0) {
            decision.reasons.push_back("promote layer=" + std::to_string(layer) +
                                       " (energy above cut)");
            continue;  // leave the cold pool
        }
        next_cold.push_back(layer);
    }

    // 4) Fill the pool up to the limit with the coldest eligible layers.
    for (const auto& [value, layer] : candidates) {
        if (static_cast<int>(next_cold.size()) >= limit) { break; }
        if (std::find(next_cold.begin(), next_cold.end(), layer) != next_cold.end()) {
            continue;
        }
        next_cold.push_back(layer);
        decision.reasons.push_back("demote layer=" + std::to_string(layer) +
                                   " ewma=" + std::to_string(value));
    }
    std::sort(next_cold.begin(), next_cold.end());

    // 5) Dwell bookkeeping.
    for (int layer = 0; layer < cfg.full_attn_layers; ++layer) {
        const bool was_cold = std::find(state.cold.begin(), state.cold.end(),
                                        layer) != state.cold.end();
        const bool now_cold = std::find(next_cold.begin(), next_cold.end(),
                                        layer) != next_cold.end();
        int& dwell = state.dwell[layer];
        if (was_cold == now_cold) {
            ++dwell;
        } else {
            dwell = 0;
        }
    }

    std::sort(state.cold.begin(), state.cold.end());
    decision.changed = next_cold != state.cold;
    state.cold = next_cold;
    decision.cold_layers = next_cold;

    // 6) Hot spec hint: reuse the caller's dtype decision but force cold
    //    layers to their nvfp4 hot window (the S30 convention: cold slots
    //    hold requantized data; the hot window stays nvfp4).
    decision.hot_spec_hint = "use build_ft_spec then override cold->nvfp4: ";
    for (const int layer : next_cold) {
        decision.hot_spec_hint += std::to_string(layer) + " ";
    }
    (void)energy_for_spec;
    return decision;
}

// INVARIANT CHECK (the thing the loop must never violate; used by the
// property test AND runnable live as an assertion after each decision).
// A demotion (hot->cold between decision.prev_cold and cold_layers) is legal
// only if the layer was eligible: non-deep, observed, streak/dwell passed,
// pool fits. Reads the decision's captured evidence, so it is valid at any
// time after decide_cold_residency() returns.
inline std::vector<std::string> invariant_violations(
    const ColdDecision& decision, const ColdPolicyConfig& cfg) {
    std::vector<std::string> bad;
    const int deep_start = static_cast<int>(
        static_cast<double>(cfg.full_attn_layers) * (1.0 - cfg.deep_frac));
    if (static_cast<int>(decision.cold_layers.size()) >
        (cfg.cold_cap > 0 ? cfg.cold_cap : 0)) {
        bad.push_back("cold set " + std::to_string(decision.cold_layers.size()) +
                      " > cold_cap " + std::to_string(cfg.cold_cap));
    }
    for (const int layer : decision.cold_layers) {
        const bool was_cold = std::find(decision.prev_cold.begin(),
                                        decision.prev_cold.end(),
                                        layer) != decision.prev_cold.end();
        if (was_cold) { continue; }  // only NEW demotions are gated here
        if (layer >= deep_start) {
            bad.push_back("demoted deep layer " + std::to_string(layer));
        }
        const auto streak = decision.streak_at_decision.find(layer);
        const int s = streak == decision.streak_at_decision.end()
                          ? -1 : streak->second;
        if (s < cfg.stable_cycles) {
            bad.push_back("demoted layer " + std::to_string(layer) +
                          " streak " + std::to_string(s) + " < " +
                          std::to_string(cfg.stable_cycles));
        }
        const auto d_it = decision.dwell_at_decision.find(layer);
        const int d = d_it == decision.dwell_at_decision.end() ? -1 : d_it->second;
        if (d < cfg.min_dwell_cycles) {
            bad.push_back("demoted layer " + std::to_string(layer) +
                          " dwell " + std::to_string(d) + " < " +
                          std::to_string(cfg.min_dwell_cycles));
        }
        const auto rounds = decision.rounds_this_window.find(layer);
        if (rounds == decision.rounds_this_window.end() ||
            rounds->second < static_cast<std::uint64_t>(cfg.min_window_rounds)) {
            bad.push_back("demoted layer " + std::to_string(layer) +
                          " unobserved/thin window");
        }
    }
    return bad;
}

}  // namespace ninfer::serve
