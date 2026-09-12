#pragma once

#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h"

#include <array>
#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

// STATUS: opt-in, and it stays opt-in. It was verified and lost - do not enable
// NINFER_MTP_WINDOW_CUT as a default, and do not reach for the `auto` front end to turn it
// on (the actuator, not the confidence, has to change first).
//
// A/B (artifact nvfp4-dflash2, shortlist draft head, batch 1, greedy, fixed prompt; raw logs
// dl/_criterion_ab.txt and dl/_criterion_ab2.txt). tok/s / acceptance length:
//   code 160 tok: k=3 199.86/3.79 | k=9 322.67/7.23 | k=15 275.07/7.57 | criterion 236.84/6.36
//   hex  128 tok: k=3 168.69/3.23 | k=9 190.00/4.23 | k=15 159.30/4.38 | criterion 138.76/3.74
//   rep  128 tok: k=3 209.06/3.97 | k=9 437.08/9.77 | k=15 510.09/14.11 | criterion 512.00/14.11
// -27% against the best fixed k on code, -27% on hex, a tie on strong repetition.
//
// WHY: the criterion moves the number of LIVE columns, but the round cost is set by the
// CAPTURED width. Over 54 controlled runs (short context, batch 1, k = 1..15):
// ms/round = 16.71 + 0.737 * W_conf, i.e. b_width = 0.737 ms per column of the requested
// window. Holding W_conf fixed and varying only the live columns gives b_mask = 0.103 ms per
// live column (four independent paired runs: 0.106/0.094/0.107/0.105 - at W_conf=15 the code
// prompt costs 27.02 ms/round drafting 8.24 columns and 27.70 ms/round drafting 14.62, so
// 6.4 columns buy 0.68 ms = 2.4% of the round while giving back 1.21 tokens/round = 16% of
// the acceptance length). The threshold below prices the column it removes at b_width
// instead of b_mask, i.e. 7.2x too high, so its break-even survival is ~7x too high and it
// cuts columns that pay for themselves.
//
// The header's own (a, b) = (14.7 ms, 0.9 ms) IS approximately the width model (measured
// 16.71 / 0.737), so the formula is right about the axis it names; the wiring applies it to
// the other axis. A ratio retune alone does NOT rescue it: with r = b_mask/a the threshold
// never binds inside a legal window (simulated against the measured code-domain hazard:
// drafted/round = 15.00 for r <= 0.0065), so the criterion silently degenerates to "draft
// the configured k" - which is the k=15 point that loses to k=9 on code and on hex. The
// winning action is to change the CAPTURED width (re-capture per width from a ladder) and
// re-decide it at a coarse cadence with this same threshold evaluated on the width model.
// Until that ladder exists the honest configuration is a fixed --draft-tokens picked by
// content class.
//
// SECOND, SMALLER DISCREPANCY (open): on the code prompt the realized window sits about one
// column below what the threshold selects from the run's own accept record - 206 drafted
// columns over 25 rounds where round 1 carries the staged 15, so rounds 2..25 average 7.96,
// against the 9 that S_9 = 0.35 > threshold 0.23 implies. Candidates, none of them checked:
// the per-round budget clamp extent = min(window, remaining-1) in the final rounds, and the
// read-across from a 25-round EMA transient. Instrument the chosen window per round before
// trusting any of these numbers.
//
// Cost model: round(k) = a + b*k with a = 14.7 ms and b = 0.9 ms per draft column, and
// tokens(k) = 1 + sum_{i<=k} S_i where S_i = prod_{j<=i} p_j and the "+1" is the
// correction/bonus token the verifier always emits (speculative_round.cuh). Throughput is
// Phi(k) = N(k) / C(k) with C(k) = a + b*k, so the discrete test for adding column k is
// Phi(k) > Phi(k-1)  <=>  S_k*C(k) > N(k-1)*b, i.e.
//
//     S_k > b * (1 + sum_{i<k} S_i) / (a + b*k).
//
// The denominator is the POST-add cost C(k) = a + b*k. Using C(k-1) instead raises the
// threshold by (a+bk)/(a+b(k-1)) ~ 6% at the operating point and measurably cuts one
// column early (measured on the code prompt: 8 columns for 236.8 tok/s against the
// 322.7 tok/s peak at k=9, while C(k) selects exactly 9 under the survival fit below).
// b/a on its own is the first column's threshold only, so using it directly over-drafts.
//
// p_i is estimated on the host from what the round already reports: an accept run of
// `accepted` drafts out of `extent` live columns observes accept/reject at every depth up
// to min(accepted+1, extent) and is censored deeper, so a depth-conditional decayed count
// is an unbiased estimator and costs no device work, no extra kernel and no sync.
//
// Caveat (measured): inside a *captured* graph the width is baked, so the marginal cost of
// a row-masked column is b_eff <= b - measured b_eff/b = 0.14. The ratio is configurable
// (env NINFER_MTP_WINDOW_RATIO) and the criterion only becomes applicable where b is
// genuinely avoidable, i.e. once the graph-width ladder lands.
inline constexpr float kMtpWindowReachDecay  = 0.9F;    // EMA per round
inline constexpr float kMtpWindowPriorAccept = 1.8F;    // Beta prior, mean 0.9 (optimistic cold start)
inline constexpr float kMtpWindowPriorReach  = 0.2F;
// b/a as calibrated on the *width* sweep (0.9/14.7). Feeding it to the threshold above
// prices a live column at the cost of a captured column, 7.2x high; see STATUS. The value is
// frozen at the original calibration so the recorded A/B stays reproducible - pass
// NINFER_MTP_WINDOW_RATIO=0.0062 to sit at the never-cut end of the range.
inline constexpr float kMtpWindowCostRatio   = 0.9F / 14.7F;  // b/a measured
inline constexpr std::uint32_t kMtpWindowMinimum = 1;

struct MtpWindowState {
    // Depth-conditional decayed counts, indexed by depth 1..kMtpDecodeMaximumDrafts.
    std::array<float, kMtpDecodeMaximumDrafts + 1> reach{};
    std::array<float, kMtpDecodeMaximumDrafts + 1> accept{};
};

// Folds one finished round into the state and returns the window to draft next time.
[[nodiscard]] inline std::uint32_t update_mtp_window_cut(MtpWindowState& state,
                                                        std::uint32_t accepted,
                                                        std::uint32_t extent,
                                                        std::uint32_t window,
                                                        float cost_ratio) noexcept {
    if (window > kMtpDecodeMaximumDrafts) { window = kMtpDecodeMaximumDrafts; }
    if (window < kMtpWindowMinimum) { window = kMtpWindowMinimum; }
    const std::uint32_t observed = accepted + 1 < extent ? accepted + 1 : extent;
    for (std::uint32_t i = 1; i <= kMtpDecodeMaximumDrafts; ++i) {
        state.reach[i]  *= kMtpWindowReachDecay;
        state.accept[i] *= kMtpWindowReachDecay;
        if (i <= observed) {
            state.reach[i] += 1.0F;
            if (i <= accepted) { state.accept[i] += 1.0F; }
        }
    }
    if (!(cost_ratio > 0.0F)) { cost_ratio = kMtpWindowCostRatio; }
    float survival = 1.0F;
    float tokens   = 1.0F;  // the token every round emits regardless of acceptance
    std::uint32_t cut = 0;
    for (std::uint32_t i = 1; i <= window; ++i) {
        const float p = (state.accept[i] + kMtpWindowPriorAccept) /
                        (state.reach[i] + kMtpWindowPriorAccept + kMtpWindowPriorReach);
        survival *= p;
        // Phi(i) > Phi(i-1)  <=>  S_i*C(i) > N(i-1)*b with C(i) = a + b*i, i.e. the
        // threshold below carries the POST-add cost. C(i-1) raises it ~6% at the operating
        // point and measurably cut one column early (8 columns / 236.8 tok/s measured on the
        // code prompt, against 322.7 tok/s at the k=9 peak).
        const float threshold =
            cost_ratio * tokens / (1.0F + cost_ratio * static_cast<float>(i));
        if (survival <= threshold) { break; }
        tokens += survival;
        cut = i;
    }
    return cut < kMtpWindowMinimum ? kMtpWindowMinimum : cut;
}

} // namespace ninfer::targets::qwen3_6::detail
