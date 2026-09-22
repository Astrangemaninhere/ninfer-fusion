#pragma once

#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h"

#include <array>
#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

// STATUS: the criterion itself was verified and it lost -- because the ACTUATOR was on the
// wrong axis, not because the confidence was wrong. It is now wired to the capture-width
// ladder (kMtpWindowLadder, round_state.h): one MTP decode graph per ladder width, selected
// per round, so the round cost really is a + b*chosen_width and a column the criterion drops
// really saves b. The old live-extent-only path is kept, still off by default, behind
// NINFER_MTP_WINDOW_CUT so the recorded A/B below stays reproducible.
//
// A/B (artifact nvfp4-dflash2, shortlist draft head, batch 1, greedy, fixed prompt; raw logs
// dl/_criterion_ab.txt and dl/_criterion_ab2.txt). tok/s / acceptance length:
//   code 160 tok: k=3 199.86/3.79 | k=9 322.67/7.23 | k=15 275.07/7.57 | criterion 236.84/6.36
//   hex  128 tok: k=3 168.69/3.23 | k=9 190.00/4.23 | k=15 159.30/4.38 | criterion 138.76/3.74
//   rep  128 tok: k=3 209.06/3.97 | k=9 437.08/9.77 | k=15 510.09/14.11 | criterion 512.00/14.11
// -27% against the best fixed k on code, -27% on hex, a tie on strong repetition.
//
// WHY IT LOST, AND WHY THE LADDER IS THE FIX. The criterion moved the number of LIVE columns
// while the round cost was set by the CAPTURED width: ms/round = 16.71 + 0.737 * W_conf
// (b_width) against b_mask = 0.103 ms per live column, so the threshold priced the column it
// removed 7.2x too high. A ratio retune does NOT rescue it: with r = b_mask/a the threshold
// never binds inside a legal window (simulated: drafted/round = 15.00 for r <= 0.0065), so the
// criterion degenerates to "draft the configured k". A closed-loop simulation over the three
// measured acceptance curves with the recorded wiring (captured width fixed at 15, live extent
// = the cut) reproduces the recorded loss -- hex -27.1%, rep -0.3%, code -14.3% -- and the same
// simulation with the ladder reaches 98.9/96.8/99.6% of the best fixed k on code/hex/rep
// (scratch/aw1/evidence4.cpp). The winning action is therefore the one taken here: change the
// CAPTURED width (re-capture per width from a ladder) and re-decide that width at a coarse
// cadence with this same threshold.
//
// THE MARGINAL TEST (CORRECTED -- the previous comment had the denominator inverted).
// With round(k) costing C(k) = a + b*k and producing N(k) = 1 + sum_{i<=k} S_i tokens
// (S_i = prod_{j<=i} p_j; the "+1" is the correction/bonus token the verifier always emits),
// throughput is Phi(k) = N(k)/C(k) and the criterion for adding column k is Phi(k) >= Phi(k-1):
//
//     (N(k-1) + S_k) / C(k) >= N(k-1) / C(k-1)
//  <=> S_k * C(k-1) >= N(k-1) * (C(k) - C(k-1))
//  <=> S_k >= b * (1 + sum_{i<k} S_i) / (a + b*(k-1))
//
// so the denominator is the PRE-add cost C(k-1) = a + b*(k-1), NOT the post-add C(k) = a + b*k
// that this header previously prescribed. The C(k) form lowers the threshold by
// (a+bk)/(a+b(k-1)) (4-5% at the operating point) and over-drafts by exactly one column: over a
// 2805-model sweep of the survival family it matches the exhaustive throughput argmax in 94.0%
// of the models (167 misses, every one of them overshooting by exactly 1) against 99.8% for the
// C(k-1) form (5 misses, none by 1), and on the code prompt it selects 10 where the exhaustive
// maximum is 9 (scratch/aw1/evidence2.cpp section (A)). The default below is the exact form;
// the post_add_denominator parameter exists only to reproduce the recorded A/B.
//
// PRIOR SHRINKAGE, ADDED (see the note at the estimate). The open item below is about a window
// one column LOW; measuring this header in the closed loop showed the OPPOSITE error, a window
// 1-3 columns HIGH, because the Beta prior's mean (0.9) was read as a measurement at depths the
// estimator had barely reached. Weighting the prior by reach[i]/reach[1] fixes it: measured mean
// chosen width 9.15/5.34/14.99/2.38 against true optima of 9/5/15/3 on the code-like, hex-like,
// harmless-repetition and no-benefit curves, i.e. 99.0-100.3% of the best fixed k on each.
//
// OPEN, STILL: the header used to report that the realized window sat about one column below
// what the threshold selects (206 drafted columns over 25 rounds where round 1 carries the
// staged 15, so rounds 2..25 average 7.96 against the 9 that S_9 = 0.35 > threshold 0.23
// implies). Two candidates were named -- the per-round budget clamp
// extent = min(window, remaining-1) in the final rounds, and the read-across from a 25-round
// EMA transient -- and a third is visible in the estimator: the Beta prior
// (kMtpWindowPriorAccept / kMtpWindowPriorReach, mean 0.9) is only 20% of the decayed mass, so
// it biases p UP and the closed loop above over-drafts instead (mean chosen width 10.4 on code
// against the true optimum 9, 7.3 on hex against 5) -- the opposite sign. None of the three is
// checked. The instrumentation the note asks for is now in place: NINFER_MTP_WINDOW_TRACE
// prints the chosen width (with the cut, S, p and the threshold behind it) per round, so the
// realized series can be compared with what the threshold selects before any of these numbers
// is trusted again.
//
// p_i is estimated on the host from what the round already reports: an accept run of
// `accepted` drafts out of `extent` live columns observes accept/reject at every depth up
// to min(accepted+1, extent) and is censored deeper, so a depth-conditional decayed count
// is an unbiased estimator and costs no device work, no extra kernel and no sync. Caveat:
// the depths deeper than the window in flight are only extrapolated, so the estimator assumes
// p_i is window-independent.
inline constexpr float kMtpWindowReachDecay  = 0.9F;    // EMA per round
inline constexpr float kMtpWindowPriorAccept = 1.8F;    // Beta prior, mean 0.9 (optimistic cold start)
inline constexpr float kMtpWindowPriorReach  = 0.2F;
// Round cost model round(k) = a + b*k, calibrated on the *width* sweep: (14.7 ms, 0.9 ms) is
// approximately the measured width model (16.71 ms + 0.737 ms per captured column). b/a on its
// own is the first column's threshold only, which is why the threshold below carries the
// running token totals.
inline constexpr float kMtpRoundCostBaseMs      = 14.7F;
inline constexpr float kMtpRoundCostPerColumnMs = 0.9F;
// b/a. NOT b_mask/a: the ladder makes the captured width the real cost axis, so b/a IS the
// right ratio now; with b_mask/a (0.103/14.7 ~ 0.0070) the threshold never binds inside a legal
// window and the criterion silently degenerates to "draft the configured k" (simulated in
// scratch/aw1/evidence2.cpp section (B)). The value is unchanged from the original calibration
// so the recorded A/B stays reproducible; NINFER_MTP_WINDOW_RATIO overrides it.
inline constexpr float kMtpWindowCostRatio = kMtpRoundCostPerColumnMs / kMtpRoundCostBaseMs;
// `ratio`'s PHYSICAL sanity band (see the guard). Absolute units on purpose: the previous band
// was in units of the constant it was supposed to replace, which makes it unfalsifiable.
inline constexpr float kMtpCostInterceptMinMs      = 2.0F;
inline constexpr float kMtpCostInterceptMaxMs      = 400.0F;
inline constexpr float kMtpCostSlopeMinMsPerColumn = 0.05F;
inline constexpr float kMtpCostSlopeMaxMsPerColumn = 8.0F;
inline constexpr std::uint32_t kMtpWindowMinimum = 1;
// THE WIDTH A LADDER-DRIVEN REQUEST STARTS FROM, BEFORE THE CRITERION HAS ANY MEASUREMENT.
//
// It is NOT the ladder top. The top is a decision the criterion has not made yet, and on this
// binary the top is the WORST rung: measured on qwen3_8_27b_nvfp4 at 258 k context, greedy, CUDA
// graphs ON, one round costs 24.78 ms at width 0, 31.31 at 3, 36.82 at 5, 49.16 at 9 and 70.99 at
// 15 (dl/mtp2/REPORT.md rows 1/3/5/7/9), i.e. 72.35 tok/s at width 15 against 99.86 at width 5.
//
// A request that starts at the top pays that round before any evidence exists, and on a request
// shorter than kMtpWidthRedecisionRounds the FIRST decision is also the ONLY one -- so the whole
// request is priced by a decision made from a single round's record. Measured with
// NINFER_MTP_WINDOW_TRACE=2 on binary 90e3f4d6a8468f23: the NIAH arm is 4 rounds, which is below
// the 8-round cadence, so it never re-decides, and its staged first round carried drafted_extent 15
// against a measured optimum of 5.
inline constexpr std::uint32_t kMtpWindowInitialRung = 5;
// Optional DECAYED-REACH FLOOR: below it the scan stops. OFF by default (0), because it was
// measured to cost more than it buys -- see the calibration note in update_mtp_window_cut.
inline constexpr float kMtpWindowMinimumReach = 0.0F;
// AN UNOBSERVED DEPTH IS NOT EVIDENCE, AND IT IS NOT THE PRIOR EITHER. `reach[i] == 0` means
// no folded round has reached depth i. In that case the weighted form above degenerates to 0/0,
// C++ makes that NaN, `!(hit > 0)` catches it and the depth is priced at the prior MEAN 0.9 --
// the most optimistic number in the model, applied to the very depths the estimator knows least
// about. This is not a corner case, it is the FIRST decision of every request (the estimator
// starts empty) and every decision after a round that stopped early. Measured on the recorded
// trace of the 32k adaptive run (dl/mtp2_ctx32k_adaptive.txt) with this header and its defaults:
//
//   empty state, extent 15, accepted  0 of 15 -> cut 12, ladder rung 15 (the ladder TOP)
//   empty state, extent 15, accepted  2 of 15 -> cut 11, ladder rung  9
//   empty state, extent 15, accepted  4 of 15 -> cut 10, ladder rung  9
//   empty state, extent 15, accepted  9 of 15 -> cut 10, ladder rung  9   <- the recorded round 1
//
// i.e. the confidence at depths 2..15 is supplied by the prior, not by the round, and a round
// that accepted NOTHING still asks for the widest ladder rung. The recorded run then carries that
// staged rung (drafted_extent 15 on round 1) and the coarse cadence holds rung 9 for 23 of its 25
// rounds: realized mean width 7.93 and 108.37 tok/s against a measured optimum of 3..5
// (k=3 132.93 tok/s; Phi(5)=131.8 from the same run's own cost fit) -- the -18% this task is about.
//
// The fix is to carry the LAST MEASURED hit rate into the unmeasured depth instead of the prior
// mean: still an extrapolation, but anchored in the deepest real measurement instead of in a
// constant that no round produced. It also keeps the scan alive, which is the property the
// measured-reach floor above does NOT have (the header records that gating the scan on reach
// starves the deep depths and ratchets the window down: -10 points on code-like content). The
// cold start is untouched: with reach[1] == 0 there is no measurement at all and the fallback
// below is the prior mean, exactly as before. Set NINFER_MTP_WINDOW_UNOBSERVED=1 to price
// unobserved depths at the prior mean again (the wiring a recorded run was measured with).
inline constexpr bool kMtpWindowUnobservedPricesAtPrior = false;
// Coarse cadence. The criterion is EVALUATED every round -- the estimator only folds the
// round's own accept record and costs nothing -- but the captured WIDTH is RE-DECIDED every N
// rounds, because changing it is an actuator move (a different graph is installed) rather than
// a measurement. Closed-loop simulation (scratch/aw1/evidence4.cpp): an 8-round cadence reaches
// 98.9/96.8/99.6% of the best fixed k on code/hex/rep against 99.3/97.8/99.6% for a per-round
// decision, so the coarse cadence costs at most ~1 point and removes 7 of 8 installs.
inline constexpr std::uint32_t kMtpWidthRedecisionRounds = 8;

struct MtpWindowState {
    // Depth-conditional decayed counts, indexed by depth 1..kMtpDecodeMaximumDrafts.
    std::array<float, kMtpDecodeMaximumDrafts + 1> reach{};
    std::array<float, kMtpDecodeMaximumDrafts + 1> accept{};
    // Deepest depth this estimator has actually observed, kept for the per-round trace only.
    std::uint32_t max_observed = 0;
};

// RUNTIME COST FIT. The threshold above needs only b/a, and b/a is a property of the
// configuration that is running, not a constant. Measured in this tree (same host, same
// artifacts, greedy, one prompt, 64 new tokens, --spec mtp, this criterion's own defaults):
//
//   FULL proposal head      k=2..15 on qwen3_8_27b_nvfp4       ->  a = 14.57 ms, b = 1.395 ms/col
//                                                                  b/a = 0.0957
//   shortlist head (Optim.) k>=5 on qwen3_8_27b_nvfp4_dflash2  ->  a = 16.57 ms, b = 0.760 ms/col
//                                                                  b/a = 0.0459
// and the STATUS note in this header records a = 16.71, b = 0.737 for that second configuration
// (b/a = 0.0441), i.e. the recorded calibration is a SHORTLIST-head calibration. The two heads
// are 2.1x apart in b/a because the shortlist head makes one draft step much cheaper while the
// per-round part (the target verify pass) is the same. An adaptive run passes draft_tokens == 0,
// which resolved_proposal_head resolves to the FULL head, so pricing it with
// kMtpRoundCostPerColumnMs / kMtpRoundCostBaseMs prices every column it might drop about 1.6x
// too cheap and the scan runs too far.
//
// So measure it instead of asserting it. Every finished round contributes one (captured width,
// wall milliseconds) sample -- both already on the host, no device work and no sync -- and a
// decayed least-squares over those samples gives the (a, b) of round(ms) = a + b * width for the
// configuration that is actually running. The fit follows the prefix length (a grows with it) and
// any graph change for free, because both show up in the samples.
//
// Until the fit is defined (fewer than two samples, or all samples at the same width, which is
// exactly when the denominator below is not positive) the caller's fallback ratio is returned,
// so a run that never moves its rung behaves exactly as it does today. The two bounds in
// `ratio` are VALIDITY GUARDS on a measurement, not calibrations of one: a fit more than 4x away
// from the fallback can only come from a stalled or contended round.
struct MtpCostCalibration {
    std::uint32_t samples   = 0;
    float         weight    = 0.0F;  // sum of decayed sample weights
    float         sum_x     = 0.0F;  // width
    float         sum_y     = 0.0F;  // milliseconds
    float         sum_xx    = 0.0F;
    float         sum_xy    = 0.0F;
    float         slope     = 0.0F;  // measured b, ms per captured column
    float         intercept = 0.0F;  // measured a, ms per round
    float         width_max = 0.0F;
    float         width_min = 0.0F;

    void observe(float width, float milliseconds) noexcept {
        if (!(width > 0.0F) || !(milliseconds > 0.0F)) { return; }
        const float decay = kMtpWindowReachDecay;  // the same per-round EMA as the estimator
        weight   = weight * decay + 1.0F;
        sum_x    = sum_x * decay + width;
        sum_y    = sum_y * decay + milliseconds;
        sum_xx   = sum_xx * decay + width * width;
        sum_xy   = sum_xy * decay + width * milliseconds;
        width_max = width > width_max || width_max == 0.0F ? width : width_max;
        width_min = width < width_min || width_min == 0.0F ? width : width_min;
        ++samples;
        const float denominator = weight * sum_xx - sum_x * sum_x;
        if (!(denominator > 0.0F)) { return; }
        slope     = (weight * sum_xy - sum_x * sum_y) / denominator;
        intercept = (sum_y - slope * sum_x) / weight;
    }

    [[nodiscard]] float ratio(float fallback) const noexcept {
        if (samples < 2 || !(slope > 0.0F) || !(intercept > 0.0F) || !(width_max > width_min)) {
            return fallback;
        }
        const float measured = slope / intercept;
        // AN ABSOLUTE SANITY BAND, NOT ONE EXPRESSED IN UNITS OF `fallback`. The previous form
        // -- reject unless measured is within [fallback/4, fallback*4] -- can never correct a
        // fallback that is off by more than 4x, and the fallback IS off by more than 4x at long
        // prefix, because `a` grows with the prefix while `b` does not, so a legitimate b/a falls.
        // Recorded evidence for the spread a band has to accept: b/a = 0.0957 (full head,
        // a = 14.57 ms), 0.0459 (shortlist, a = 16.57 ms) from this header; 0.138 from the header's
        // own 258 k width table (a = 22.6 ms, b = 3.12 ms/col); ~0.013 from the same 258 k arm's
        // own decode time (1.540 s / 26 rounds = 59.2 ms/round at mean width 8.0 with the shortlist
        // b = 0.760). The old lower edge 0.0153 excludes the last one. Both new bounds are
        // PHYSICAL, not calibration-relative: a decode round cannot cost < 2 ms or > 400 ms, and a
        // captured column cannot cost < 0.05 or > 8 ms.
        if (!(intercept >= kMtpCostInterceptMinMs && intercept <= kMtpCostInterceptMaxMs) ||
            !(slope >= kMtpCostSlopeMinMsPerColumn && slope <= kMtpCostSlopeMaxMsPerColumn)) {
            return fallback;
        }
        return measured;
    }
};

// One criterion evaluation in full, so the actuator can trace the decision (and so the open
// +-1 discrepancy above is diagnosable per round) without re-deriving any of it.
struct MtpWindowDecision {
    std::uint32_t cut       = kMtpWindowMinimum;  // the criterion's cut, in [1, bound]
    float         survival  = 0.0F;  // S at the last examined column
    float         hit_rate  = 0.0F;  // the per-depth estimate p at that column
    float         threshold = 0.0F;  // the threshold that column faced
    std::uint32_t examined  = 0;     // how many columns were examined
};

// Folds one finished round into the state and returns the window decision for the next round.
//
// `bound` is the WIDEST column the decision may reach: the capture ladder's top for an adaptive
// run, or the plan's configured window for a fixed-k run. It is deliberately NOT the previous
// round's window -- passing a shrunk window back in would make the decision shrink-only and it
// could never grow again. The caller at program_impl passes the plan width, so this is a latent
// trap rather than an active defect, and the ladder removes it by construction because the plan
// width IS the ladder top.
//
// `post_add_denominator`, `min_reach` and `prior_shrink` exist so historical measurements stay
// reproducible and so the estimator can be calibrated. post_add_denominator = true with
// prior_shrink = false and min_reach = 0 rebuilds dl/_criterion_ab.txt
// (env NINFER_MTP_WINDOW_DENOM=1, NINFER_MTP_WINDOW_SHRINK=0, NINFER_MTP_WINDOW_MINREACH=0).
// Leave all three at their defaults for anything new.
[[nodiscard]] inline MtpWindowDecision update_mtp_window_cut(MtpWindowState& state,
                                                            std::uint32_t accepted,
                                                            std::uint32_t extent,
                                                            std::uint32_t bound,
                                                            float cost_ratio,
                                                            bool post_add_denominator = false,
                                                            float min_reach = kMtpWindowMinimumReach,
                                                            bool prior_shrink = true,
                                                            bool unobserved_at_prior =
                                                                kMtpWindowUnobservedPricesAtPrior) noexcept {
    if (bound > kMtpDecodeMaximumDrafts) { bound = kMtpDecodeMaximumDrafts; }
    if (bound < kMtpWindowMinimum) { bound = kMtpWindowMinimum; }
    const std::uint32_t observed = accepted + 1 < extent ? accepted + 1 : extent;
    for (std::uint32_t i = 1; i <= kMtpDecodeMaximumDrafts; ++i) {
        state.reach[i]  *= kMtpWindowReachDecay;
        state.accept[i] *= kMtpWindowReachDecay;
        if (i <= observed) {
            state.reach[i] += 1.0F;
            if (i <= accepted) { state.accept[i] += 1.0F; }
        }
    }
    if (observed > state.max_observed) { state.max_observed = observed; }
    if (!(cost_ratio > 0.0F)) { cost_ratio = kMtpWindowCostRatio; }
    MtpWindowDecision decision;
    float survival = 1.0F;
    float tokens   = 1.0F;  // the token every round emits regardless of acceptance
    std::uint32_t cut = 0;
    // The hit rate at the deepest depth that HAS been measured, carried into the depths the
    // estimator has not reached yet (see kMtpWindowUnobservedPricesAtPrior). The prior mean
    // seeds it so that the cold-start fallback below is unchanged.
    float carried_hit = kMtpWindowPriorAccept / (kMtpWindowPriorAccept + kMtpWindowPriorReach);
    for (std::uint32_t i = 1; i <= bound; ++i) {
        if (state.reach[i] < min_reach) { break; }
        // PRIOR SHRINKAGE. The Beta prior is a cold-start device, but its weight was a constant,
        // so where reach[i] is small-but-nonzero the prior dominated and p_i was read as the
        // prior MEAN 0.9 at every depth the window only rarely reached:
        // on a no-benefit curve (p_i = 0.45) that inflates p_4 from 0.45 to
        // (0.4 + 1.8)/(0.9 + 2.0) = 0.76, and the criterion then drafts 6.4 columns where 3 is
        // optimal, which is not nearly free because Phi falls steeply there (88.5% of the best
        // fixed k). Weighting the prior by the share of the round mass that reached the depth --
        // weight = reach[i]/reach[1], i.e. the decayed survival the estimator already maintains --
        // leaves depth 1 untouched and drives deeper depths onto their own counts:
        //   weight    1.00 at i=1 (unchanged),  0.09 at i=4 in the example above
        //   p_4       (0.4 + 1.8*0.09) / (0.9 + 2.0*0.09) = 0.52   (was 0.76)
        // The prior still supplies the cold start: with reach[1] == 0 nothing has been observed
        // at all, weight collapses to 0, and the fallback below is the prior mean again.
        //
        // MEASURED (closed loop, 16 seeds x 512 rounds, cost = a + b*captured width, cells are
        // Phi / Phi(best fixed k) and the mean chosen width; scratch/aw1/evidence7.cpp):
        //                       floor 0            floor 0.02          floor 0.10         floor 1.0
        //   code-like    shrink off 92.1%/15.00   89.0%/7.62         89.0%/7.62         90.0%/6.89
        //                shrink on 100.3%/ 9.15   90.2%/6.40         90.2%/6.40         90.2%/6.32
        //   hex-like     off        74.9%/15.00   92.1%/4.44         92.3%/4.38         92.7%/4.01
        //                on         99.9%/ 5.34   92.9%/3.83         92.9%/3.83         92.9%/3.83
        //   rep-like     off       100.0%/15.00   97.2%/14.38        97.2%/14.38        97.2%/14.38
        //                on        100.0%/14.99   97.2%/14.38        97.2%/14.38        97.2%/14.38
        //   no-benefit   off        64.6%/15.00   99.5%/2.37         99.7%/2.25        100.0%/2.05
        //                on         99.0%/ 2.38  100.0%/2.04        100.0%/2.04        100.1%/2.03
        //
        // Two conclusions. (1) The shrinkage is what matters: at floor 0 it takes the mean chosen
        // width to 9.15/5.34/14.99/2.38 against true optima of 9/5/15/3, i.e. 99.0-100.3% of the
        // best fixed k on all four curves, where the unshrunk prior sat at 15 on three of them.
        // (2) The SCAN FLOOR is NOT worth having: gating the scan on the same reach the window
        // itself controls starves the deep depths, the window shrinks and the shrunk window keeps
        // them starved -- a downward ratchet -- so any nonzero floor costs code-like ~10 points
        // and hex-like ~7 to buy a correction the shrinkage already made. Hence: shrinkage ON,
        // floor 0, and the floor kept only as a calibration knob.
        float hit           = 0.0F;
        const bool measured = state.reach[i] > 0.0F;
        if (prior_shrink && state.reach[1] > 0.0F && measured) {
            const float weight = state.reach[i] / state.reach[1];
            hit = (state.accept[i] + kMtpWindowPriorAccept * weight) /
                  (state.reach[i] + (kMtpWindowPriorAccept + kMtpWindowPriorReach) * weight);
        }
        if (!(hit > 0.0F)) {
            // No measurement at this depth. The historical wiring scored it at the prior mean
            // (0.9, the most optimistic value in the model); the fix carries the deepest measured
            // rate instead. See kMtpWindowUnobservedPricesAtPrior.
            hit = unobserved_at_prior
                      ? kMtpWindowPriorAccept / (kMtpWindowPriorAccept + kMtpWindowPriorReach)
                      : carried_hit;
        }
        if (measured) { carried_hit = hit; }
        survival *= hit;
        // C(i) = a + b*i, C(i-1) = a + b*(i-1). See the derivation above: the exact test is
        // S_i >= b*tokens/C(i-1), so the denominator carries i-1 unless the caller explicitly
        // asked for the historical post-add form.
        const float columns = static_cast<float>(post_add_denominator ? i : (i - 1));
        const float threshold = cost_ratio * tokens / (1.0F + cost_ratio * columns);
        decision.examined  = i;
        decision.survival  = survival;
        decision.hit_rate  = hit;
        decision.threshold = threshold;
        if (survival < threshold) { break; }
        tokens += survival;
        cut = i;
    }
    decision.cut = cut < kMtpWindowMinimum ? kMtpWindowMinimum : cut;
    return decision;
}

} // namespace ninfer::targets::qwen3_6::detail
