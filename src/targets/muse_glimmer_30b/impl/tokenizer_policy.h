#pragma once

// Muse-Glimmer-30B's own TOKENIZER POLICY, declared once, where the model is declared.
// Line `xfer`, marker F879.  Read off the checkpoint, not chosen:
//
//   /mnt/g/models/muse/Muse-Glimmer-30B-NVFP4/tokenizer_config.json
//     sha256 ee60d284ed7857ba89155f437cc915951e16056453ca7ae2491c5ae87a395c39
//     add_bos_token      ABSENT
//     add_prefix_space   ABSENT
//     pad_token          "<|end_of_text|>"     (a bare STRING)
//
// The ABSENT keys are the fact the engine had no way to receive: HF's own default for this
// tokenizer is false, while the shared validator's `value(k, true)` read an absent key as TRUE
// and refused.  The revision in Muse-Glimmer-30B (the BF16 repo, sha256 781e6c74...) declares
// `pad_token = "<|finetune_right_pad|>"` instead, which is why the pad token is a DECLARATION
// and not a constant: two revisions of one model state two facts, and only a per-model
// declaration can hold both.
//
// WHAT THIS DOES NOT DO.  It does not admit the muse artifacts that exist TODAY: those carry
// the converter's rewrite (tools/convert/muse_glimmer_30b/convert.py:280-281 wrote
// `add_prefix_space = False` and `pad_token = "<|endoftext|>"`), so they already match
// `TokenizerPolicy{}` and the declaration below neither helps nor hurts them.  What it removes
// is the NEED for that rewrite on the next conversion.

#include <ninfer/targets/qwen3_6/tokenizer_policy.h>

namespace ninfer::targets::muse_glimmer_30b::detail {

// The two revisions this target can load, each with the fact that revision states.
// A third revision is a data edit here, not a new refusal site.
inline constexpr std::string_view kMusePadTokens[] = {
    "<|end_of_text|>",          // Muse-Glimmer-30B-NVFP4  (tokenizer_config sha ee60d284...)
    "<|finetune_right_pad|>",   // Muse-Glimmer-30B       (tokenizer_config sha 781e6c74...)
};

inline constexpr qwen3_6::TokenizerPolicy kTokenizerPolicy{
    .declared_by                      = "Muse-Glimmer-30B",
    .add_bos_token_when_absent        = false,
    .add_prefix_space_when_absent     = false,
    // BOTH revisions, declared.  Measured by dl/flowopen on the real checkpoints
    // (G:/models/muse/): NVFP4 -> <|end_of_text|> (id 200001), BF16 -> <|finetune_right_pad|>
    // (id 200018).  Declaring only the first is what made the BF16 revision admissible at
    // produce time and inadmissible at load -- a per-revision fact used as a per-target gate.
    .pad_token                        = "<|end_of_text|>",
    .pad_tokens                       = kMusePadTokens,
    .pad_token_spelling               = qwen3_6::PadTokenSpelling::StringOnly,
};

static_assert(kTokenizerPolicy.pad_token != qwen3_6::TokenizerPolicy{}.pad_token,
              "a declaration that equals the default is not a declaration; if muse's pad token "
              "ever becomes <|endoftext|> this seam is no longer needed for it and the "
              "assertion is the place that says so");

} // namespace ninfer::targets::muse_glimmer_30b::detail
