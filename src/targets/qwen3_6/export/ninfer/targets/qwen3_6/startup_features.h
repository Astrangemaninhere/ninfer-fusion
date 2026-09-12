#pragma once

#include "ninfer/types.h"

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
[[nodiscard]] inline ProposalHead resolved_proposal_head(ProposalHead head,
                                                        SpeculativeBackend backend,
                                                        bool has_shortlist_head) noexcept {
    if (head != ProposalHead::Auto) { return head; }
    if (backend != SpeculativeBackend::DFlash2 || !has_shortlist_head) {
        return ProposalHead::Full;
    }
    return ProposalHead::Optimized;
}


} // namespace ninfer::targets::qwen3_6
