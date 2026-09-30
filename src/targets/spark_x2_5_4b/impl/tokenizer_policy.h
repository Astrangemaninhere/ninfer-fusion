#pragma once

// Spark-X2.5-4B's own TOKENIZER POLICY, declared once, where the model is declared.
// Line `xfer`, marker F879.  Read off the artifact's own frontend bytes, not chosen:
//
//   dl/sparkx/out/frontend/tokenizer_config.json  (extracted from the artifact)
//     sha256 5597c57dbce318860f1d4af07de12938dbb02ed42c4332730203b1c4823dbf7b
//     add_bos_token      false                 (present)
//     add_prefix_space   ABSENT
//     pad_token          an AddedToken OBJECT, `content` = "<\uFF5C\u2581pad\u2581\uFF5C>"
//
// THIS IS THE MODEL dl/sparkx (F853) MEASURED THE ENGINE REFUSING, twice, at rc=1 with 0
// stdout bytes: first at frontend.cpp:205-208 because an absent `add_prefix_space` was read as
// TRUE, then at :210-213 because the pad token is an object rather than a string.  Both
// refusals were CONVERTER-REPAIRED in tools/convert/spark_x2_5_4b/convert.py:971-972, which is
// what the declaration below makes unnecessary.
//
// The pad token is Gemma-family spelled (`<\uFF5C ... \uFF5C>`), not Qwen's `<|endoftext|>`:
// Spark-X2.5's tokenizer is not Qwen's, and the only reason the engine could not say so was
// that the string was a literal in the shared path.  It is written with universal-character
// escapes so the bytes are the same no matter what encoding this file is transported through.

#include <ninfer/targets/qwen3_6/tokenizer_policy.h>

namespace ninfer::targets::spark_x2_5_4b::detail {

inline constexpr qwen3_6::TokenizerPolicy kTokenizerPolicy{
    .declared_by                      = "Spark-X2.5-4B",
    .add_bos_token_when_absent        = false,
    .add_prefix_space_when_absent     = false,
    .pad_token                        = "<\uFF5C\u2581pad\u2581\uFF5C>",
    .pad_token_spelling               = qwen3_6::PadTokenSpelling::StringOrAddedTokenObject,
};

static_assert(kTokenizerPolicy.pad_token != qwen3_6::TokenizerPolicy{}.pad_token,
              "a declaration that equals the default is not a declaration");

} // namespace ninfer::targets::spark_x2_5_4b::detail
