#pragma once

#include "ninfer/types.h"

#include <string_view>

namespace ninfer::targets::qwen3_6 {

struct StartupFeatures {
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool dflash_like() const noexcept { return dflash() || dflash2(); }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }
};

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    return StartupFeatures{
        .vision        = options.enable_vision,
        .speculative   = options.speculative.backend,
        .proposal_head = options.speculative.proposal_head,
    };
}

// ProposalHead::Auto is resolved by the target packages (resolved_auto_speculative)
// before the planner, the load plan and the program see the options, so startup
// features and every downstream ProposalHead switch see a concrete head.
//
// Auto picks the shortlist draft head only for DFlash2, where it is a pure proposal
// mechanism: measured byte-identical greedy output to the full head (df2 k=7: 1/160
// positions, the same single verify-batch near-tie flip both ways) for +2.5% speed.
// MTP must keep the full head: with the shortlist head the drafts leak into the
// emitted tokens (k=3: 89/160 positions differ from plain greedy, 6 edit blocks),
// which contradicts the documented greedy contract "bit-identical to the original
// argmax accept" (speculative_round.cuh). --lm-head-draft still opts in explicitly.
// A disabled run must land on Full: layouts_impl.h rejects a non-full head once
// speculation is off (the target head does the sampling then).
// MTP may take the shortlist head from this draft window up. Measured on
// qwen3.8-27b/nvfp4-dflash2 (same prompt, greedy, 160 tokens, dl/_head_sweep.txt):
// k=3 differs from the full head at 88/160 positions while k=5/7/9 are
// bit-identical, and the shortlist head is 14-28% faster (k=9: 325.63 vs 253.86
// tok/s). Below the threshold the full head is the fidelity-safe choice.
//
// WHAT THIS NUMBER IS, AND WHAT IT IS NOT (provenance, spelled out because the
// equal-looking 5 in two other records is a DIFFERENT measurement):
//   * THIS 5 compares the SHORTLIST head against the FULL head on the same artifact
//     (nvfp4-dflash2), same prompt, greedy, 160 tokens: 88/160 = 55% of positions
//     differ at k=3, and k=5/7/9 are bit-identical. It is a per-depth accept/reject
//     fidelity boundary between two proposal heads.
//   * The "full head 5/6 lossless boundary" recorded elsewhere is a different axis:
//     drafter-vs-target acceptance on which k stops paying, on the FULL head alone.
//   * The 89/160 in the block above is a third axis again: shortlist head vs PLAIN
//     GREEDY (emitted-token leak), not shortlist vs full head (88/160).
//   The three numbers coincide at 5 by numerical accident, NOT by shared provenance,
//   so this constant must not be cited as evidence for either of the other two.
inline constexpr std::uint32_t kMtpShortlistMinimumDrafts = 5;

// WHY a concrete head was chosen. Exists so the `return ProposalHead::Full` below is
// never SILENT: "the run landed on the full head because this artifact has no shortlist
// head" and "because the backend is not eligible" and "because k is below the threshold"
// are three different facts, and before this function they were one indistinguish-
// able `full` in the log. A caller prints the name (qwen3_6_27b/impl/package.cpp,
// plan_load); the resolver keeps returning exactly the same values it did.
[[nodiscard]] inline std::string_view proposal_head_reason(ProposalHead head,
                                                          SpeculativeBackend backend,
                                                          bool has_shortlist_head,
                                                          std::uint32_t draft_tokens) noexcept {
    if (head != ProposalHead::Auto) { return "explicit"; }
    if (!has_shortlist_head) { return "no-shortlist-head"; }
    if (backend == SpeculativeBackend::DFlash2) { return "dflash2"; }
    if (backend == SpeculativeBackend::Mtp) {
        return draft_tokens >= kMtpShortlistMinimumDrafts ? "mtp-at-or-above-minimum"
                                                          : "mtp-below-minimum";
    }
    return "backend-not-eligible";
}

[[nodiscard]] inline ProposalHead resolved_proposal_head(ProposalHead head,
                                                        SpeculativeBackend backend,
                                                        bool has_shortlist_head,
                                                        std::uint32_t draft_tokens,
                                                        std::string_view* reason = nullptr) noexcept {
    if (reason != nullptr) {
        *reason = proposal_head_reason(head, backend, has_shortlist_head, draft_tokens);
    }
    if (head != ProposalHead::Auto) { return head; }
    if (!has_shortlist_head) { return ProposalHead::Full; }
    if (backend == SpeculativeBackend::DFlash2) { return ProposalHead::Optimized; }
    if (backend == SpeculativeBackend::Mtp && draft_tokens >= kMtpShortlistMinimumDrafts) {
        return ProposalHead::Optimized;
    }
    return ProposalHead::Full;
}


} // namespace ninfer::targets::qwen3_6
