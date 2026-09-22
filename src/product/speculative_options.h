#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    // "none"/"off" is the explicit opt-out: both front ends default --spec to auto, so
    // there has to be a spelling for turning speculation back off.
    if (value == "none" || value == "off") { return SpeculativeBackend::None; }
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    // "dspark" is an alias for DFlash, not a fourth backend, because the DSpark
    // drafter IS the DFlash (v1) runtime -- measured, by reading the tree:
    //   * the op is reached from the DFlash draft step only, gated on the draft
    //     weights being present: runtime/dflash_impl.h:455-476 calls
    //     ops::dspark_markov_argmax when `markov_w1`/`markov_w2` are bound, and
    //     falls back to ops::argmax (:474) when they are not;
    //   * those two weights are materialized for exactly one weights identity,
    //     `WeightsProfile::Qwen38Nvfp4Dspark` =
    //     {model_id=qwen3.8-27b, weights_id="nvfp4-dspark"}
    //     (targets/qwen3_6_27b/impl/load/bindings.cpp:902-905;
    //     export/ninfer/targets/qwen3_6_27b/package.h:200-203), and the DFlash
    //     config it runs under is the DSpark one (impl/config.h:118-152:
    //     bf16_weights, full_only, svip_entropy_threshold 2.5,
    //     target_feature_layers {4,16,28,40,52});
    //   * the SVIP threshold is runtime/program_impl.h:710-724, consumed at
    //     :15333 and :16730.
    // So which of the two drafters runs is decided by the ARTIFACT, not by this
    // string: this spelling resolves to DFlash and the artifact's own
    // `dspark_weights(profile)` predicate decides whether the Markov head is in
    // play (impl/package.cpp:50-64 makes the declaration and the payload agree).
    // Accepting it here therefore does not promise that the loaded artifact
    // carries the Markov head; a check that refuses `--spec dspark` on an
    // artifact without it needs the artifact, i.e. the target package's
    // resolve_weights/plan_load, not this pure enum translation.
    if (value == "dspark") { return SpeculativeBackend::DFlash; }
    if (value == "auto") { return SpeculativeBackend::Auto; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    case SpeculativeBackend::Auto:
        return "auto";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full &&
                options.proposal_head != ProposalHead::Auto) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        // Upper bound raised 5 -> 15 (the dflash domain bound; the verify width cap of
        // 16 admits k <= 15). Measured: on code k=3/5/9 give 196.6/225.7/327.6 tok/s
        // (AL 3.48/4.32/7.31) while on Chinese k=5 is worse than k=3, so the optimal k
        // is content dependent. That content dependence is now ACTUATED instead of merely
        // stated: 0 means ADAPTIVE -- the engine captures the width ladder (kMtpWindowLadder)
        // and the survival/cost criterion picks the rung per round, so a column it drops
        // really saves b_width. >0 pins a single captured width, i.e. exactly the pre-ladder
        // fixed-k behaviour, for every recorded fixed-k experiment.
        //   adaptive, explicit : --spec mtp --draft-tokens 0
        //   adaptive, default  : --spec auto        (resolves to MTP with draft_tokens left 0)
        //   pinned             : --spec mtp --draft-tokens 9
        // (--spec auto still rejects an explicit --draft-tokens: auto's whole job is to
        // resolve the backend, and pinning a width is spelled with an explicit backend.)
        if (options.draft_tokens > 15) {
            throw std::invalid_argument(
                "--spec mtp requires --draft-tokens in [0,15] (0 = adaptive)");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        // K is the startup-fixed block width (query width K + 1). 0 keeps the
        // historical 7-draft default; 15 is the decode frame domain bound.
        if (options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        // Both proposal heads are supported (propose_batch_impl): the full
        // vocabulary head, and the checkpoint's dedicated shortlist head whose rows
        // are mapped to global ids through text/draft_head_token_ids.
        return;
    case SpeculativeBackend::Auto:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full &&
                options.proposal_head != ProposalHead::Auto) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
