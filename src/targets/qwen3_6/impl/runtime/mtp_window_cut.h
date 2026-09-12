#pragma once

#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h"

#include <array>
#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

// Adaptive MTP draft-window control: the survival/cost criterion (replaces the entropy
// heuristic, which measured net negative - entropy is not the accept probability, it
// carries no survival product, and its nats threshold has no relation to the cost ratio).
//
// Per-round objective with the measured cost model (artifact nvfp4-dflash2, shortlist
// draft head, batch 1): round(k) = a + b*k with a = 14.7 ms and b = 0.9 ms per draft
// column, and tokens(k) = 1 + sum_{i<=k} S_i where S_i = prod_{j<=i} p_j and the "+1" is
// the correction/bonus token the verifier always emits (speculative_round.cuh). Column k
// column, and tokens(k) = 1 + sum_{i<=k} S_i where S_i = prod_{j<=i} p_j and the "+1" is
// the correction/bonus token the verifier always emits (speculative_round.cuh).
// Throughput is Phi(k) = N(k) / C(k) with C(k) = a + b*k, so the discrete test for
// adding column k is Phi(k) > Phi(k-1)  <=>  S_k*C(k) > N(k-1)*b, i.e.
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
// a row-masked column is b_eff <= b. The ratio is therefore configurable (env
// NINFER_MTP_WINDOW_RATIO) and the criterion is applied where b is genuinely avoidable once
// the graph-width ladder lands.
inline constexpr float kMtpWindowReachDecay  = 0.9F;    // EMA per round
inline constexpr float kMtpWindowPriorAccept = 1.8F;    // Beta prior, mean 0.9 (optimistic cold start)
inline constexpr float kMtpWindowPriorReach  = 0.2F;
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
