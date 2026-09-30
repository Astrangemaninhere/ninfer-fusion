// tests/test_ple_stage_declaration.cpp — the first test that touches
// targets/qwen4_exp/impl/ple_ngram.h's ple_stage_declaration().
//
// WHY THIS FILE EXISTS
//   ple_stage_declaration() was declared at ple_ngram.h:107, defined at
//   ple_ngram.cpp:56, compiled into ninfer_engine (targets/qwen4_exp/CMakeLists.txt:15)
//   -- and referenced by NOTHING, test or engine. A symbol that is compiled, has a
//   non-trivial body, and is called by nobody is exactly the shape that rots silently:
//   the compiler checks the body's TYPES, and nothing checks the body's CLAIMS.
//   This file checks the claims.
//
// WHAT IT CANNOT DO (named, so the green here is not read as more than it is)
//   * It does not load a sidecar, so it cannot read number_of_ngram_heads out of a
//     real ple-manifest.json. The geometry it cross-checks against is
//     PLEConfig's constexpr set (config.h:90-105), which carries its own
//     static_asserts against the sidecar consumer (config.h:152-156).
//   * It does not run the layer-1 PLE residual stack; that runtime does not exist.
//
// THE OBJECT UNDER TEST IS A POLICY, SO THE LOAD-BEARING CHECK IS THE TRAP
//   qwen4_exp_spec.json's ple_layer_ids is [2] and is 1-BASED; the engine walks
//   0-based layers, so the declaration must land on layer 1. Converting once too
//   often (using 2) or never (using 2 as if 0-based) both produce a declaration that
//   looks well-formed and puts the residual on the wrong layer. There is no runtime
//   that would notice, so the check has to be here.
//
// Load-bearing controls, one per check. "red" is what a mutation of the production
// code (or of this file's own guard) has to do to make the row fail:
//   | # | check                                                | red if |
//   |---|------------------------------------------------------|--------|
//   | 1 | declaration carries the config.h geometry            | cfg values change without config.h's static_asserts |
//   | 2 | layer 1 is declared, the spec's 1-based 2 is NOT     | ple_stage_declaration passes the raw spec value through |
//   | 3 | the 1-based value is a DIFFERENT layer (tripwire not vacuous) | the two spellings collide (then #2 proves nothing) |
//   | 4 | eos is passed through verbatim, 0 stays 0            | the body substitutes a default eos for 0 |
//   | 5 | validate() accepts the real declaration              | ngram_size/heads_per_ngram/embed_dim go structural |
//   | 6 | validate() refuses each structural lie               | the guard is deleted (validate becomes vacuous) |
//   | 7 | validate_against() accepts the real sidecar geometry | the cross-check inverts |
//   | 8 | validate_against() refuses each of the four mismatches | any of its four comparisons is dropped |
//   | 9 | an ABSENT declaration is inert                       | the `if (!present) return;` early-outs are removed |

#include "ops/ple/ple_stage.h"
#include "targets/qwen4_exp/impl/ple_ngram.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

namespace {

using ninfer::ops::ple::PleStageDecl;
using ninfer::targets::qwen4_exp::detail::PLEConfig;
using ninfer::targets::qwen4_exp::detail::ple_stage_declaration;

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    } else {
        std::printf("ok: %s\n", what);
    }
}

bool throws_invalid_argument(const std::function<void()>& body) {
    try {
        body();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

bool throws_runtime_error(const std::function<void()>& body) {
    try {
        body();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// The real sidecar's eos, read from flashnext_ple/ple_gather_golden.json ("eos": 248044).
constexpr std::uint32_t kRealSidecarEos = 248044;

// qwen4_exp_spec.json's ple.ple_layer_ids[0]. 1-BASED, and the whole point of check 2.
constexpr std::uint32_t kSpecLayerId1Based = 2;

// The real sidecar geometry, from flashnext_ple/ple-manifest.json
// (ngram_size 3, heads_per_ngram 8, number_of_ngram_heads 16, embedding_row_dimension 160).
constexpr std::uint32_t kSidecarNgram   = 3;
constexpr std::uint32_t kSidecarHeads   = 8;
constexpr std::uint32_t kSidecarNHeads  = 16;
constexpr std::uint32_t kSidecarRowDim  = 160;

} // namespace

int main() {
    const PleStageDecl decl = ple_stage_declaration(kRealSidecarEos);

    // ---- 1. The geometry half: config.h is the single source, and it is pinned there.
    expect(decl.present, "the declaration is present (a qwen4_exp artifact has a PLE stage)");
    expect(decl.ngram_size == kSidecarNgram, "ngram_size is 3 (the sidecar's trigram window)");
    expect(decl.heads_per_ngram == kSidecarHeads, "heads_per_ngram is 8");
    expect(decl.embed_dim == 2560, "ple_embed_dim is 2560 (== TextConfig::hidden)");
    expect(decl.n_heads() == kSidecarNHeads, "n_heads() is 16 = (3-1)*8, the sidecar's head count");
    expect(decl.n_heads() == static_cast<std::uint32_t>(PLEConfig::n_heads),
           "n_heads() agrees with PLEConfig::n_heads (the static_asserted sidecar geometry)");
    expect(decl.embed_dim ==
               static_cast<std::uint32_t>(PLEConfig::embed_dim),
           "embed_dim agrees with PLEConfig::embed_dim");

    // ---- 2 + 3. THE TRAP: 1-based spec value 2 must rebase to 0-based layer 1.
    expect(decl.layer_ids.size() == 1, "the declaration names exactly one layer");
    expect(decl.layer_ids[0] == 1,
           "the declared layer id is 1 (0-based), i.e. the spec's 1-based 2 rebased once");
    expect(decl.declares_layer(1), "declares_layer(1) is true");
    expect(!decl.declares_layer(kSpecLayerId1Based),
           "TRIPWIRE: layer 2 (the raw 1-based spec value) is NOT declared");
    expect(!decl.declares_layer(0), "layer 0 is not declared");

    // The tripwire above is only evidence if the two spellings really are different
    // layers. If a future edit made 1 and 2 the same index space, check 2 would pass
    // for the wrong reason; this row is what stops that.
    {
        PleStageDecl one_based = decl;
        one_based.layer_ids = {kSpecLayerId1Based};
        expect(one_based.layer_ids != decl.layer_ids,
               "the 1-based and 0-based spellings name DIFFERENT layers, so the tripwire is not vacuous");
        expect(!one_based.declares_layer(1) && one_based.declares_layer(kSpecLayerId1Based),
               "the 1-based spelling is detected by declares_layer as a different layer");
    }

    // ---- 4. eos is DATA, passed through; "not known yet" (0) is preserved, never guessed.
    expect(decl.eos_token_id == kRealSidecarEos,
           "eos_token_id is the value the caller passed (248044)");
    expect(ple_stage_declaration(0).eos_token_id == 0,
           "eos_token_id 0 ('not known yet') is preserved verbatim, not replaced by a guess");
    expect(ple_stage_declaration(1).eos_token_id == 1,
           "a different eos reaches the declaration unchanged (the field is not clamped)");

    // ---- 5. The real declaration passes its own structural validate() (called inside).
    expect(!throws_invalid_argument([] { (void)ple_stage_declaration(kRealSidecarEos); }),
           "ple_stage_declaration() runs validate() on its own output and it accepts");

    // ---- 6. ...and validate() is not vacuous: each structural lie is refused.
    expect(throws_invalid_argument([&] {
               PleStageDecl d = decl;
               d.ngram_size = 1;
               d.validate();
           }),
           "validate() refuses ngram_size < 2 (a unigram has no n-gram stage)");
    expect(throws_invalid_argument([&] {
               PleStageDecl d = decl;
               d.heads_per_ngram = 0;
               d.validate();
           }),
           "validate() refuses heads_per_ngram == 0 (a stage with no heads emits nothing)");
    expect(throws_invalid_argument([&] {
               PleStageDecl d = decl;
               d.embed_dim = 0;
               d.validate();
           }),
           "validate() refuses embed_dim == 0 (the residual would be empty)");

    // ---- 7 + 8. The sidecar cross-check: accepts the real geometry, refuses each field.
    {
        bool accepted = true;
        try {
            decl.validate_against(kSidecarNgram, kSidecarHeads, kSidecarNHeads, kSidecarRowDim);
        } catch (...) {
            accepted = false;
        }
        expect(accepted, "validate_against() accepts the real sidecar geometry (3/8/16/160)");
    }
    // Each probe below is chosen to move ONE field and leave the other three
    // comparisons satisfied -- otherwise a dropped comparison is masked by the next
    // one and the row stops being evidence. (MEASURED: probing the head count with
    // 15/160 was masked by the embed_dim comparison, because 15*160 != 2560; the
    // 20/128 probe keeps n_heads*row_dim == 2560 so only the head-count check can
    // be what throws.)
    expect(throws_runtime_error([&] { decl.validate_against(2, 8, 16, 160); }),
           "validate_against() refuses a sidecar ngram_size of 2 (only that field moves)");
    expect(throws_runtime_error([&] { decl.validate_against(3, 4, 16, 160); }),
           "validate_against() refuses a sidecar heads_per_ngram of 4 (only that field moves)");
    expect(throws_runtime_error([&] { decl.validate_against(3, 8, 20, 128); }),
           "validate_against() refuses a sidecar head count of 20 with row_dim 128 "
           "(20*128 == 2560, so only the head-count check can be what throws)");
    expect(throws_runtime_error([&] { decl.validate_against(3, 8, 16, 128); }),
           "validate_against() refuses a row dimension that disagrees with embed_dim");

    // ---- 9. An absent declaration is inert, and that inertness is load-bearing:
    //         every accessor has to be safe on the artifact that has no PLE stage.
    {
        PleStageDecl absent;
        expect(!absent.present, "a default-constructed declaration is absent");
        expect(!absent.declares_layer(1), "an absent declaration declares no layer");
        bool inert = true;
        try {
            absent.validate();
            absent.validate_against(99, 99, 99, 99);
        } catch (...) {
            inert = false;
        }
        expect(inert,
               "validate()/validate_against() early-out on an absent declaration instead of throwing");
    }

    std::printf(failures == 0 ? "PLE_STAGE_DECLARATION_PASS\n" : "PLE_STAGE_DECLARATION_FAIL\n");
    return failures == 0 ? 0 : 1;
}
