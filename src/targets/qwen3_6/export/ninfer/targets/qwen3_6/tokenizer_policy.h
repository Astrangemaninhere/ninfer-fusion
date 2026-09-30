#pragma once

// =============================================================================================
// tokenizer_config POLICY -- the SEAM a per-model tokenizer policy arrives through.
// Line `xfer`, marker F879.  See dl/xfer/out/POP.tsv for the population this closes.
//
// WHY THIS FILE EXISTS.  The shared frontend path (src/targets/qwen3_6/impl/frontend/
// frontend.cpp:202-225) tested a model's `tokenizer_config.json` against FOUR literals that are
// facts about ONE tokenizer: `add_bos_token` false, `add_prefix_space` false, `pad_token` the
// STRING "<|endoftext|>", and `chat_template` equal to the artifact's template.  None of the
// four arrived through a parameter.  Measured consequences on two NON-qwen models:
//
//   * `dl/sparkx` (F853): the Spark-X2.5-4B artifact is REFUSED at frontend.cpp:205-208 with
//     0 stdout bytes -- `add_bos_token: false` present, `add_prefix_space` ABSENT, and the
//     engine reads an absent key as TRUE; then refused again at :210-213 because its own
//     `pad_token` is an AddedToken OBJECT, not a string.
//   * `dl/museconv` (F852): muse's artifact is admitted only because BOTH converters REWRITE
//     the model's policy into Qwen's shape -- tools/convert/muse_glimmer_30b/convert.py:280-281
//     and tools/convert/spark_x2_5_4b/convert.py:971-972 each assign `add_prefix_space = False`
//     and `pad_token = "<|endoftext|>"` over the checkpoint's own values.
//
// THE SHAPE OF THE REPAIR IS THE TREE'S OWN.  `FrontendOptions` already carries two per-model
// frontend facts -- `token_domain` (muse 202048, Spark 131072) and `validate_official_special_ids`
// (both false) -- with the comment "other family members override with their own tokenizer
// geometry".  The tokenizer POLICY was the same kind of fact and was left out.  So the seam is
// this struct, default-initialised to Qwen3.6's own values, which is why every existing caller
// is unchanged and why the Qwen path is byte-identical.
//
// ONE IMPLEMENTATION, NOT TWO THAT AGREE BY CARE (F-871's shape).  `tokenizer_config_reasons()`
// below is the single evaluator, and frontend.cpp DELEGATES to it; the four sentences it returns
// under the default policy are byte-identical to the four the old inline block threw.  The
// converter-side Python restatement (tools/convert/qwen3_6/common/frontend_policy.py:223-249)
// is NOT imported by this header and cannot be; it is a third copy, and this line measured it
// against this header instead of assuming it agrees (out/CROSSINSTRUMENT.txt).
// =============================================================================================

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6 {

// How a checkpoint may spell its own pad token.
enum class PadTokenSpelling {
    // exactly the bare JSON string -- Qwen's own shape, and the default.
    StringOnly = 0,
    // a bare string, OR the HuggingFace `AddedToken` OBJECT whose `content` is the token.
    // A checkpoint that spells it the second way is not a different tokenizer; it is the
    // same tokenizer written by a different exporter.
    StringOrAddedTokenObject = 1,
};

// The tokenizer policy a TARGET declares about its OWN tokenizer_config.json.  Every field is a
// fact about THE MODEL'S TOKENIZER; none is a fact about the engine.
struct TokenizerPolicy {
    // The name this policy is declared by.  It appears in the refusal sentence and NOWHERE in
    // the comparison, so a message can name the model whose policy was violated without the
    // engine owning the model's name.  Default = the family this path was written for.
    std::string_view declared_by = "Qwen3.6";

    // What an ABSENT `add_bos_token` / `add_prefix_space` key MEANS for this tokenizer.
    // Qwen3.6's official config carries both keys as `false`, and the engine's old inline
    // `value(k, true)` therefore made an absent key FATAL -- a default that belongs to the
    // tokenizer, not to the engine.  A target whose tokenizer simply omits the key declares
    // the value its own tokenizer means by omission.
    bool add_bos_token_when_absent    = true;
    bool add_prefix_space_when_absent = true;

    // This tokenizer's pad token, and how its config is allowed to spell it.
    // THE ADMISSIBLE SET, not the one value.  One model's revisions state the pad token
    // as DIFFERENT facts (measured on Muse-Glimmer-30B: the NVFP4 revision declares
    // <|end_of_text|>, the BF16 revision declares <|finetune_right_pad|>), and a single
    // value cannot hold both -- which is exactly what muse's own policy header argued for
    // ("only a per-model declaration can hold both") while the mechanism it landed held
    // one.  The set is the declaration of WHICH revisions exist; each member is one
    // revision's fact.  A token outside the set is STILL REFUSED BY NAME, so this does
    // not weaken the gate: <|endoftext|> -- a token the model does not have -- remains
    // inadmissible, which is the whole reason the gate exists.
    // First member is the rung's own default; pad_token is kept readable as "the first".
    std::string_view pad_token       = "<|endoftext|>";
    std::span<const std::string_view> pad_tokens{};   // empty => the set is {pad_token}
    PadTokenSpelling pad_token_spelling = PadTokenSpelling::StringOnly;

    // The policy the engine's shared path has always enforced, spelled once so a reader can
    // see that the defaults above ARE it and not a coincidence.
    [[nodiscard]] static constexpr TokenizerPolicy qwen3_6() noexcept { return TokenizerPolicy{}; }
};

// Every reason a `tokenizer_config.json` does NOT satisfy the declared policy.  An EMPTY vector
// is acceptance.  The four conditions are exactly the four of the old inline block, read against
// the declaration instead of against literals; under `TokenizerPolicy{}` every returned sentence
// is byte-identical to the one the inline block threw.
[[nodiscard]] inline std::vector<std::string> tokenizer_config_reasons(
    const nlohmann::json& tokenizer_config, std::string_view chat_template_jinja,
    const TokenizerPolicy& policy) {
    using Json = nlohmann::json;
    std::vector<std::string> reasons;

    // -- 1/2. prefix semantics, read through the DECLARED absent-key meaning ----------------
    const bool add_bos =
        tokenizer_config.value("add_bos_token", policy.add_bos_token_when_absent);
    const bool add_prefix =
        tokenizer_config.value("add_prefix_space", policy.add_prefix_space_when_absent);
    if (add_bos || add_prefix) {
        reasons.push_back("tokenizer_config.json does not match " +
                          std::string(policy.declared_by) + " tokenizer prefix semantics");
    }

    // -- 3. the pad token, compared for IDENTITY and normalised only for SPELLING -----------
    const auto pad_refusal = [&policy] {
        return "tokenizer_config.json does not use the official " +
               std::string(policy.pad_token) + " pad token";
    };
    if (!tokenizer_config.contains("pad_token")) {
        reasons.push_back(pad_refusal());
    } else {
        const Json& pad = tokenizer_config.at("pad_token");
        std::string spelled;
        bool readable = false;
        if (pad.is_string()) {
            spelled  = pad.get<std::string>();
            readable = true;
        } else if (policy.pad_token_spelling == PadTokenSpelling::StringOrAddedTokenObject &&
                   pad.is_object() && pad.contains("content") && pad.at("content").is_string()) {
            spelled  = pad.at("content").get<std::string>();
            readable = true;
        }
        // Compare against the SET.  An empty set degrades to the single value, so every
        // existing caller is byte-identical; a non-empty set admits exactly the revisions
        // it names and nothing else.  The refusal names the whole admissible set, so a
        // reader is told what to write rather than only that what they wrote is wrong.
        bool admitted = readable && (spelled == policy.pad_token);
        if (!admitted) {
            for (const auto candidate : policy.pad_tokens) {
                if (readable && spelled == candidate) { admitted = true; break; }
            }
        }
        if (!admitted) { reasons.push_back(pad_refusal()); }
    }

    // -- 4. the config's own chat_template must BE the artifact's template object -----------
    if (!tokenizer_config.contains("chat_template") ||
        !tokenizer_config.at("chat_template").is_string()) {
        reasons.push_back("tokenizer_config.json.chat_template must contain the loaded chat "
                          "template");
    } else if (tokenizer_config.at("chat_template").get_ref<const std::string&>() !=
               chat_template_jinja) {
        reasons.push_back(
            "tokenizer_config.json.chat_template does not match frontend/chat_template.jinja");
    }
    return reasons;
}

} // namespace ninfer::targets::qwen3_6
