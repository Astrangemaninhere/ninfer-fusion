#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
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
        // is content dependent and is meant to be picked by the adaptive cut.
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,15]");
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
