#pragma once

// Declarative capability registration for target packages.
//
// An artifact states what it is: the container `identity` object, exactly two members
// (`src/artifact/reader.cpp:296`, `tools/artifact/container.py:22`). A target states what it
// consumes: the identities it accepts, the weights profile each identity means, the target key it
// reports, and the sampling presets it publishes per model id. Recognition is then one lookup in a
// table the target owns, and a rejection is one message that carries both sides of the comparison.
//
// This replaces the `if` chain over `identity.model_id` / `identity.weights_id` that used to decide
// whether an artifact was supported. That chain grew one branch per quantization flavour (see
// `research/scripts/_add_bf16head_profile.py` for the last such patch), and a rejected artifact was
// told only that it was "not supported" - never what the target did accept.
//
// The declarations are `constexpr` and header-only on purpose: the loader, the cold registry, the
// context-cost calibrator and the tests then read one declaration, and the CPU path can be
// exercised without a device.

#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets {

// The two members of the artifact identity that select a weights profile. One definition of "which
// artifact is this" for the whole engine: the loader, the cold registry and the context-cost
// presets (`src/runtime/engine/context_cost.cpp`) all speak this key.
struct WeightsIdentityKey {
    std::string_view model_id;
    std::string_view weights_id;

    [[nodiscard]] friend constexpr bool operator==(WeightsIdentityKey left,
                                                   WeightsIdentityKey right) noexcept {
        return left.model_id == right.model_id && left.weights_id == right.weights_id;
    }
};

// One declared capability: an artifact that declares this identity is this profile.
//
// `profile` is the target's own enum - the value the binding family, the workspace sizing and the
// speculative defaults are already keyed by. A second identity that means the same profile reuses
// the same value; it does not get an enum member of its own, because nothing downstream can tell
// the two apart. `target_key` and `provenance` are declarations too, never dispatch: the first is
// what the loader reports in `LoadSummary`, the second is where the flavour comes from.
template <class Profile>
struct WeightsDeclaration {
    WeightsIdentityKey identity;
    Profile profile;
    std::string_view target_key;
    std::string_view provenance;
};

// The accepted set of one target package.
template <class Profile, std::size_t Count>
class WeightsDeclarationTable {
public:
    using Declaration = WeightsDeclaration<Profile>;

    constexpr WeightsDeclarationTable(const std::array<Declaration, Count>& declared,
                                      std::string_view target_key) noexcept
        : declared_(declared), target_key_(target_key) {}

    [[nodiscard]] constexpr std::span<const Declaration> declared() const noexcept {
        return declared_;
    }

    // The lookup. Every accepted identity is a row; nothing is inferred from the model id or from
    // the weights id alone.
    [[nodiscard]] constexpr const Declaration* find(std::string_view model_id,
                                                    std::string_view weights_id) const noexcept {
        for (const Declaration& row : declared_) {
            if (row.identity.model_id == model_id && row.identity.weights_id == weights_id) {
                return &row;
            }
        }
        return nullptr;
    }

    // Does this target consume artifacts of this model id at all? Derived from the same rows as
    // `find`, so "registered" and "accepted" cannot drift apart.
    [[nodiscard]] constexpr bool declares_model(std::string_view model_id) const noexcept {
        for (const Declaration& row : declared_) {
            if (row.identity.model_id == model_id) { return true; }
        }
        return false;
    }

    [[nodiscard]] constexpr std::string_view
    target_key_for(std::string_view model_id) const noexcept {
        for (const Declaration& row : declared_) {
            if (row.identity.model_id == model_id) { return row.target_key; }
        }
        return {};
    }

    // The one rejection position. The message opens with the historical sentence and then adds the
    // two lists a diagnosis needs: what the artifact declared about itself, and what this target
    // declares it accepts.
    [[nodiscard]] const Declaration& require(std::string_view model_id,
                                            std::string_view weights_id) const {
        if (const Declaration* row = find(model_id, weights_id)) { return *row; }
        throw std::runtime_error("artifact identity '" + std::string(model_id) + "/" +
                                 std::string(weights_id) + "' is not supported by target '" +
                                 std::string(target_key_) +
                                 "'; declared accepted weights identities: " + describe() +
                                 "; artifact declares weights identity '" + std::string(model_id) +
                                 "/" + std::string(weights_id) + "'");
    }

    [[nodiscard]] std::string describe() const {
        std::string text;
        for (const Declaration& row : declared_) {
            if (!text.empty()) { text += ", "; }
            text += std::string(row.identity.model_id);
            text += "/";
            text += std::string(row.identity.weights_id);
        }
        return text;
    }

    [[nodiscard]] std::string describe_model_ids() const {
        std::string text;
        for (std::size_t index = 0; index < declared_.size(); ++index) {
            const std::string_view model_id = declared_[index].identity.model_id;
            bool already_listed            = false;
            for (std::size_t prior = 0; prior < index; ++prior) {
                if (declared_[prior].identity.model_id == model_id) { already_listed = true; }
            }
            if (already_listed) { continue; }
            if (!text.empty()) { text += ", "; }
            text += std::string(model_id);
        }
        return text;
    }

private:
    std::span<const Declaration> declared_;
    std::string_view target_key_;
};

// Per model id data that is not a weights flavour: the sampling presets a target publishes. Same
// shape as the weights table - a declared key set, one lookup, one rejection message.
template <std::size_t Count>
class SamplingDeclarationTable {
public:
    struct Declaration {
        std::string_view model_id;
        ModelSamplingDefaults defaults;
    };

    constexpr SamplingDeclarationTable(const std::array<Declaration, Count>& declared,
                                       std::string_view target_key) noexcept
        : declared_(declared), target_key_(target_key) {}

    [[nodiscard]] constexpr const Declaration* find(std::string_view model_id) const noexcept {
        for (const Declaration& row : declared_) {
            if (row.model_id == model_id) { return &row; }
        }
        return nullptr;
    }

    [[nodiscard]] const Declaration& require(std::string_view model_id) const {
        if (const Declaration* row = find(model_id)) { return *row; }
        throw std::runtime_error("model '" + std::string(model_id) +
                                 "' has no sampling defaults in target package '" +
                                 std::string(target_key_) + "'; declared model ids: " +
                                 describe());
    }

    [[nodiscard]] std::string describe() const {
        std::string text;
        for (const Declaration& row : declared_) {
            if (!text.empty()) { text += ", "; }
            text += std::string(row.model_id);
        }
        return text;
    }

private:
    std::span<const Declaration> declared_;
    std::string_view target_key_;
};

// Compile-time guards a package pins its own declaration with.
namespace declaration_check {

// Every row's target key must be one the package declares, so a typo in a row cannot silently
// change what `LoadSummary` reports.
template <class Profile, std::size_t Count>
[[nodiscard]] constexpr bool
target_keys_are(const std::array<WeightsDeclaration<Profile>, Count>& declared,
                std::initializer_list<std::string_view> allowed) noexcept {
    for (const WeightsDeclaration<Profile>& row : declared) {
        bool matched = false;
        for (const std::string_view candidate : allowed) {
            if (candidate == row.target_key) { matched = true; }
        }
        if (!matched) { return false; }
    }
    return true;
}

// Two rows with the same identity would make the lookup ambiguous.
template <class Profile, std::size_t Count>
[[nodiscard]] constexpr bool
identities_are_distinct(const std::array<WeightsDeclaration<Profile>, Count>& declared) noexcept {
    for (std::size_t index = 0; index < Count; ++index) {
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (declared[prior].identity == declared[index].identity) { return false; }
        }
    }
    return true;
}

// Every model id this target publishes sampling defaults for must be a model id it consumes
// artifacts of, so the two declarations cannot disagree about which model ids the target owns.
template <class Profile, std::size_t WeightsCount, class SamplingRow, std::size_t SamplingCount>
[[nodiscard]] constexpr bool sampling_models_are_declared(
    const std::array<WeightsDeclaration<Profile>, WeightsCount>& weights,
    const std::array<SamplingRow, SamplingCount>& sampling) noexcept {
    for (const SamplingRow& row : sampling) {
        bool matched = false;
        for (const WeightsDeclaration<Profile>& weight : weights) {
            if (weight.identity.model_id == row.model_id) { matched = true; }
        }
        if (!matched) { return false; }
    }
    return true;
}

// Every row must name both members of the artifact identity it declares. The load path matches on
// that pair, so a row whose weights id is empty would accept an artifact that declares no weights
// flavour at all, and a row whose model id is empty would make `declares_model("")` true and hand
// such an artifact to this package in `src/targets/registry.cpp`. Ported from id1's
// `identity_table_is_reachable`, which pinned this clause (and only this clause) on its own
// identity table.
template <class Profile, std::size_t Count>
[[nodiscard]] constexpr bool
rows_name_an_identity(const std::array<WeightsDeclaration<Profile>, Count>& declared) noexcept {
    for (const WeightsDeclaration<Profile>& row : declared) {
        if (row.identity.model_id.empty() || row.identity.weights_id.empty()) { return false; }
    }
    return true;
}

// How many distinct profiles the identity table reaches. The literal a package compares this
// against is that package's own `WeightsProfile` enumerator count, written out by hand: ported from
// id1's `static_assert(kRealizations.size() == N, "WeightsProfile changed: ...")`, which pinned the
// same count on its realization table. It fires when a row stops reaching a profile; it cannot see
// an enumerator that arrived without a row, which is what the comment at the literal is for.
template <class Profile, std::size_t Count>
[[nodiscard]] constexpr std::size_t
distinct_profiles(const std::array<WeightsDeclaration<Profile>, Count>& declared) noexcept {
    std::size_t distinct = 0;
    for (std::size_t index = 0; index < Count; ++index) {
        bool already_reached = false;
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (declared[prior].profile == declared[index].profile) { already_reached = true; }
        }
        if (!already_reached) { ++distinct; }
    }
    return distinct;
}

} // namespace declaration_check

} // namespace ninfer::targets
