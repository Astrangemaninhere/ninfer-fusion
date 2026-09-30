#pragma once

// src/spec/sum_dir_determined.h -- THE `DETERMINED` CONJUNCTION, WRITTEN AS ONE EXECUTION.
// (line `indexgaps`, gap (2))
//
// THE CONJUNCTION, VERBATIM FROM THE DESIGN (dl/vectorkey/REPORT.md sec.1.3, dl/THIRDKIND.md sec.3.2)
//
//     DETERMINED(query_key, store) :=
//           EXISTS! b IN store : K(b) == query_key                                  -- (D1)
//        AND  read_back(b).ids.size() == row.token_count(b)                        -- (D2)
//        AND  recall_identity_tokens(read_back(b).ids) == row.block_identity(b)      -- (D3)
//        AND  read_back(b) is the block the plan NAMES                              -- (D4)
//
//   Every clause already had a carrier in the tree -- the design says so and names each one:
//   (D1) `SumDir::find`'s duplicate run (sum_dir.h:1747-1761) plus `spec/sum_dir_query_key.h`;
//   (D2) the cargo record's ABSENT length field, so the length can only come from the ROW
//        (`sum_dir_row_tokens`, sum_dir.h:515-517); (D3) `sum_dir_block_digest` recomputed on read;
//   (D4) `sum_dir_reach.h [text: substitute the positional first-fit silently; = :110 on 2026-09-25]` + `program_impl.h [text: chooses a page by POSITION; = :16212 on 2026-09-25]`, which forbid choosing a page by
//   POSITION. What did not exist is ONE function that runs all four and can FAIL. This is that
//   function.
//
// THE REFUSAL SIDE IS HALF OF IT, AND IT IS THE HALF THAT MAKES IT SAFE
//   A similarity score, a distance, a cosine, a top-k rank or any threshold may NOT produce
//   `DETERMINED`. It may only contribute to the candidate-set UNION -- program_impl.h [text: exact-or-silent, never approximately right; = :15948 on 2026-09-25],
//   verbatim: "It is exact-or-silent, never approximately right, and that is the property that makes
//   it admissible under this project's refusal discipline. A semantic channel ... belongs in the
//   candidate-set UNION, never in place of this arbiter." This header therefore takes a KEY and an
//   id SPAN and no score of any kind: there is no parameter a score could arrive through.
//
// WHAT MAKES `sum_dir_row_bound()` INSUFFICIENT, SAID HERE SO IT IS NOT USED INSTEAD
//   `sum_dir_row_bound` (sum_dir.h:627-636) is a ROW check, and its own comment says at
//   sum_dir.h:620-626: "It is NOT the authoritative comparison the tree requires before reuse ...
//   True here means 'this row may be offered', NEVER 'reuse these bytes'." A design that reused it as
//   the determination would be one grade too weak; this header is the stronger check, and it is
//   deliberately a different function so the two cannot be confused at a call site.
//
// WHAT THIS HEADER DOES NOT DO, NAMED SO IT IS NOT MISTAKEN FOR DONE
//   * It does not READ the cargo record. It takes the read-back's ids as an input, so that the
//     ids it hashes are the ids that came out of the reader and not a value it computed itself.
//   * It does not decide which row the plan names. `plan_named_row` is an INPUT, because a check
//     that derived it from its own answer could not detect a substitution.
//   * It is not wired into the engine: no engine-side caller, the same property
//     `sum_dir_vector.h:49-52` claims for itself, so landing it cannot move the engine binary.
//
// WHY THE CLAUSES ARE SELECTABLE -- AND WHY THE SELECTOR IS NOT A PRODUCTION KNOB
//   A conjunction whose clauses cannot be individually removed cannot be shown to be LOAD-BEARING:
//   a check that passes because it never reaches a clause looks exactly like a check that passes
//   because the clause holds. `sum_dir_determined_clauses_with_dropped()` exists so the executed
//   check can be RUN with one clause broken, and the whole point of that run is that the AGGREGATE
//   turns red -- the mutant that the full clause set refuses is then ACCEPTED. The tree's own
//   precedent for that shape is `dl/absprod`'s FORCED-RECALL mutant (RED-A: keep the descriptor
//   lossless and the check must go red). It is NOT a production knob: `clauses_all()` is the
//   default, every clause defaults to ON, and no environment variable reaches it.

#include "spec/sum_dir_model_binding.h"
#include "spec/sum_dir_query_key.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::spec::sum_dir::determined {

// The two neighbouring landings, named here once so the conjunction below reads as the design's four
// clauses and not as a set of qualified paths. Each `using` is the WHOLE of what this file needs
// from its neighbours, so a change to either one's contract shows up as a compile error here.
using model_binding::SumDirModelBindingHeader;
using model_binding::SumDirModelBindingVerdict;
using model_binding::SumDirModelFingerprint;
using model_binding::sum_dir_model_binding_check;
using query_key::SumDirQueryKey;
using query_key::SumDirQueryKeyLookup;
using query_key::SumDirQueryKeyVerdict;
using query_key::kSumDirQueryKeyMaxTokens;
using query_key::sum_dir_query_key_lookup;

// The block count after which the (D2)/(D3) comparison is decided. A read-back longer than one block
// is refused by the same producer bound the query key obeys, so the two sides agree about the axis.
inline constexpr std::uint32_t kSumDirDeterminedMaxTokens = kSumDirQueryKeyMaxTokens;

// ---------------------------------------------------------------------------
// the clause set
// ---------------------------------------------------------------------------

struct SumDirDeterminedClauses {
    bool d1_uniqueness        = true;
    bool d2_length_from_row   = true;
    bool d3_rederived_identity = true;
    bool d4_no_substitution   = true;
    // The FOURTH artefact's gate (spec/sum_dir_model_binding.h). ON by default: a determination that
    // survives a weight change is the exact failure the fourth artefact exists to catch, so it is
    // not an optional extra.
    bool model_binding        = true;
    // ⛔ THE MUTANT CARRIER. Setting this true makes the check resolve a |candidates| != 1 BY
    // POSITION -- i.e. the forbidden fallback of sum_dir_reach.h [text: substitute the positional first-fit silently; = :110 on 2026-09-25] -- instead of refusing.
    // It exists so the executed check can be RUN against the forbidden shape and be watched
    // admitting it, which is what makes (D1) load-bearing. It must never be true in a production
    // call, and `sum_dir_determined_clauses_with_dropped("POSITIONAL")` is the only producer of it.
    bool allow_positional_substitution = false;
};

// Everything ON. This is the only policy a production caller passes.
[[nodiscard]] constexpr SumDirDeterminedClauses
sum_dir_determined_clauses_all() noexcept {
    return SumDirDeterminedClauses{};
}

// Drop ONE clause by name, for the mutant arms only. Names: "D1" "D2" "D3" "D4" "MODEL"
// and "POSITIONAL" (which does not drop a clause -- it installs the forbidden substitution the
// other four are supposed to catch). An unknown name drops nothing, so a typo fails toward the
// STRONGER check rather than a weaker one.
[[nodiscard]] inline SumDirDeterminedClauses
sum_dir_determined_clauses_with_dropped(const std::string& clause) noexcept {
    SumDirDeterminedClauses clauses; // all on
    if (clause == "D1") { clauses.d1_uniqueness = false; }
    if (clause == "D2") { clauses.d2_length_from_row = false; }
    if (clause == "D3") { clauses.d3_rederived_identity = false; }
    if (clause == "D4") { clauses.d4_no_substitution = false; }
    if (clause == "MODEL") { clauses.model_binding = false; }
    if (clause == "POSITIONAL") { clauses.allow_positional_substitution = true; }
    return clauses;
}

// ---------------------------------------------------------------------------
// the verdict, one name per way the conjunction can fail
// ---------------------------------------------------------------------------

enum class SumDirDeterminedVerdict : std::uint8_t {
    Determined                 = 0,
    RefusedKeyUnset            = 1,
    RefusedNotSorted           = 2,
    RefusedNoCandidate         = 3, // (D1) -- zero rows
    RefusedAmbiguous           = 4, // (D1) -- more than one row; never crowned by rank
    RefusedReadBackAbsent      = 5,
    RefusedLengthMismatch      = 6, // (D2)
    RefusedIdentityMismatch     = 7, // (D3)
    RefusedSubstituted         = 8, // (D4)
    RefusedModelChanged        = 9, // the fourth artefact
};

[[nodiscard]] constexpr const char* sum_dir_determined_verdict_name(
    SumDirDeterminedVerdict verdict) noexcept {
    switch (verdict) {
    case SumDirDeterminedVerdict::Determined: return "determined";
    case SumDirDeterminedVerdict::RefusedKeyUnset: return "refused-key-unset";
    case SumDirDeterminedVerdict::RefusedNotSorted: return "refused-not-sorted";
    case SumDirDeterminedVerdict::RefusedNoCandidate: return "refused-no-candidate";
    case SumDirDeterminedVerdict::RefusedAmbiguous: return "refused-ambiguous";
    case SumDirDeterminedVerdict::RefusedReadBackAbsent: return "refused-read-back-absent";
    case SumDirDeterminedVerdict::RefusedLengthMismatch: return "refused-length-mismatch";
    case SumDirDeterminedVerdict::RefusedIdentityMismatch: return "refused-identity-mismatch";
    case SumDirDeterminedVerdict::RefusedSubstituted: return "refused-substituted";
    case SumDirDeterminedVerdict::RefusedModelChanged: return "refused-model-changed";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// the inputs
// ---------------------------------------------------------------------------

// The ids the reader yielded. `row` is the row the reader says they came from, which is compared
// against the plan's own naming in (D4) -- the two are different facts and neither is derived from
// the other.
struct SumDirDeterminedReadBack {
    bool ids_present = false;
    std::size_t row = 0;
    std::vector<std::uint32_t> ids;
};

// Every clause's own value, so a failure is a fact and not a boolean. A caller printing this can say
// WHICH clause moved, which is the difference between a diagnosis and a verdict.
struct SumDirDeterminedReport {
    SumDirQueryKeyVerdict key_verdict = SumDirQueryKeyVerdict::RefusedKeyUnset;
    std::uint32_t candidates     = 0;   // (D1) the CARDINALITY, counted
    std::size_t   determined_row = 0;   // (D1) the unique row, when candidates == 1
    std::size_t   plan_named_row = 0;   // (D4) the input, echoed
    std::size_t   read_back_row  = 0;   // (D4) the reader's own claim

    bool clause_d1 = false;
    bool clause_d2 = false;
    bool clause_d3 = false;
    bool clause_d4 = false;

    std::uint32_t read_back_tokens = 0;
    std::uint32_t row_tokens       = 0;   // from sum_dir_row_tokens(row) -- the ROW's length
    SumDirDigest  rederived_identity{};   // sum_dir_block_digest(read_back.ids) -- recomputed here
    SumDirDigest  row_identity{};         // row.block_identity -- the STORED value
    bool          rederived_is_unset = false;

    SumDirModelBindingVerdict model_verdict = SumDirModelBindingVerdict::Bound;
    std::uint32_t directory_payload_digest = 0;
};

// ---------------------------------------------------------------------------
// THE EXECUTION
// ---------------------------------------------------------------------------

// Order is a CONTRACT and it is the order of the conjunction, (D1) first: a determination that
// cannot name a block has nothing for (D2)-(D4) to compare, so reporting a length mismatch on an
// ambiguous key would describe the wrong failure.
//
// The stored model binding and the CURRENT fingerprint are both inputs and they are different
// values on purpose: the stored one travels with the directory, the current one is what the model
// in this process IS. Comparing a value with itself would make the fourth artefact decoration.
[[nodiscard]] inline SumDirDeterminedVerdict sum_dir_determined(
    const SumDir& directory, const SumDirQueryKey& key, std::size_t plan_named_row,
    const SumDirDeterminedReadBack& read_back, const SumDirModelBindingHeader& stored_binding,
    const SumDirModelFingerprint& current_fingerprint,
    const SumDirDeterminedClauses& clauses = SumDirDeterminedClauses{},
    SumDirDeterminedReport* report = nullptr) {
    SumDirDeterminedReport local;
    local.plan_named_row = plan_named_row;
    local.read_back_row  = read_back.row;
    local.read_back_tokens = static_cast<std::uint32_t>(read_back.ids.size());
    local.directory_payload_digest = directory.payload_digest();

    // ---- (D1) UNIQUENESS, exact equality, no score -------------------------
    const SumDirQueryKeyLookup lookup = sum_dir_query_key_lookup(directory, key);
    local.key_verdict  = lookup.verdict;
    local.candidates   = lookup.candidates;
    local.determined_row = lookup.first_row;

    if (clauses.allow_positional_substitution &&
        (lookup.verdict == SumDirQueryKeyVerdict::RefusedAmbiguous ||
         lookup.verdict == SumDirQueryKeyVerdict::RefusedNoCandidate)) {
        // ⛔ THE MUTANT'S ONLY INPUT, and it is the exact shape the tree forbids:
        // sum_dir_reach.h [text: substitute the positional first-fit silently; = :110 on 2026-09-25] -- "that fallback chooses a page by POSITION, which is precisely
        // the 'plausible-looking wrong anchor' the requirement forbids". A |candidates| != 1 is
        // resolved by taking the row the PLAN named instead of refusing. With this input the
        // conjunction ADMITS an ambiguous key, which is what makes (D1) load-bearing rather than
        // decorative.
        local.determined_row = plan_named_row;
        local.candidates = 1U;
        local.clause_d1 = true;
    } else {
        local.clause_d1 = clauses.d1_uniqueness
                              ? (lookup.verdict == SumDirQueryKeyVerdict::Determined)
                              : true;
    }

    SumDirDeterminedVerdict verdict = SumDirDeterminedVerdict::Determined;
    if (!local.clause_d1) {
        switch (lookup.verdict) {
        case SumDirQueryKeyVerdict::RefusedKeyUnset: verdict = SumDirDeterminedVerdict::RefusedKeyUnset; break;
        case SumDirQueryKeyVerdict::RefusedNotSorted: verdict = SumDirDeterminedVerdict::RefusedNotSorted; break;
        case SumDirQueryKeyVerdict::RefusedNoCandidate: verdict = SumDirDeterminedVerdict::RefusedNoCandidate; break;
        case SumDirQueryKeyVerdict::RefusedAmbiguous: verdict = SumDirDeterminedVerdict::RefusedAmbiguous; break;
        case SumDirQueryKeyVerdict::Determined: verdict = SumDirDeterminedVerdict::Determined; break;
        }
    }

    // ---- (D2) the length is the ROW's, not the record's -------------------
    // NOTE, and it is a structural fact rather than a preference: because `sum_dir_block_digest`
    // folds the COUNT into the digest (sum_dir.h:216-219), no input can break (D2) without also
    // breaking (D3). (D2) is therefore not a second GATE -- (D3) is the gate -- it is the clause
    // that NAMES the reason. Dropping it does not let a mutant through; it makes the same refusal
    // arrive under the less specific name `refused-identity-mismatch`. That is observable, which is
    // what the executed check's `drop-D2` arm measures.
    bool row_known = false;
    if (verdict == SumDirDeterminedVerdict::Determined) {
        row_known = local.candidates >= 1U && local.determined_row < directory.rows().size();
    }
    if (row_known) {
        const SumDirRow& row = directory.rows()[local.determined_row];
        local.row_tokens = sum_dir_row_tokens(row);
        local.row_identity = row.block_identity;
    }
    local.clause_d2 = clauses.d2_length_from_row
                          ? (row_known && (local.read_back_tokens == local.row_tokens))
                          : true;
    if (verdict == SumDirDeterminedVerdict::Determined && !local.clause_d2) {
        verdict = SumDirDeterminedVerdict::RefusedLengthMismatch;
    }

    // ---- (D3) the content identity, RE-DERIVED here, never trusted -------
    if (verdict == SumDirDeterminedVerdict::Determined) {
        if (!read_back.ids_present) {
            local.clause_d3 = !clauses.d3_rederived_identity;
            verdict = SumDirDeterminedVerdict::RefusedReadBackAbsent;
        } else {
            local.rederived_identity = sum_dir_block_digest(read_back.ids.data(),
                                                            read_back.ids.size());
            local.rederived_is_unset = local.rederived_identity.is_unset();
            local.clause_d3 = clauses.d3_rederived_identity
                                  ? (row_known && (local.rederived_identity == local.row_identity))
                                  : true;
            if (!local.clause_d3) {
                verdict = SumDirDeterminedVerdict::RefusedIdentityMismatch;
            }
        }
    }

    // ---- (D4) no substitution, positional first-fit forbidden ------------
    if (verdict == SumDirDeterminedVerdict::Determined) {
        if (clauses.allow_positional_substitution) {
            // THE MUTANT. The forbidden shape is admitted on purpose so the check can be watched
            // catching it: any row is taken as if it were the plan's.
            local.clause_d4 = true;
        } else {
            local.clause_d4 = clauses.d4_no_substitution
                                  ? (read_back.row == plan_named_row && row_known &&
                                     local.determined_row == plan_named_row)
                                  : true;
            if (!local.clause_d4) {
                verdict = SumDirDeterminedVerdict::RefusedSubstituted;
            }
        }
    }

    // ---- the fourth artefact: the directory's MODEL binding ---------------
    if (verdict == SumDirDeterminedVerdict::Determined) {
        local.model_verdict = sum_dir_model_binding_check(stored_binding, current_fingerprint,
                                                         local.directory_payload_digest);
        if (clauses.model_binding &&
            local.model_verdict != SumDirModelBindingVerdict::Bound) {
            verdict = SumDirDeterminedVerdict::RefusedModelChanged;
        }
    }

    if (report != nullptr) { *report = local; }
    return verdict;
}

// The same execution with the model gate REMOVED, kept out of the main function on purpose so that
// the difference between "the conjunction" and "the conjunction minus artifact four" is a call site
// and not a boolean a reader has to trace.
[[nodiscard]] inline SumDirDeterminedVerdict sum_dir_determined_without_model_binding(
    const SumDir& directory, const SumDirQueryKey& key, std::size_t plan_named_row,
    const SumDirDeterminedReadBack& read_back,
    const SumDirDeterminedClauses& clauses = SumDirDeterminedClauses{}) {
    // A binding whose every component is UNSET, plus a current fingerprint whose components are also
    // unset -- so the model gate cannot pass, and dropping it is the only way through. That is the
    // point: this helper exists to be the arm that goes green ONLY because the fourth artefact is
    // absent.
    SumDirDeterminedClauses without_model = clauses;
    without_model.model_binding = false;
    SumDirModelBindingHeader unset_binding;      // magic/version set, components unset
    SumDirModelFingerprint unset_fingerprint;    // all components unset
    return sum_dir_determined(directory, key, plan_named_row, read_back, unset_binding,
                              unset_fingerprint, without_model, nullptr);
}

} // namespace ninfer::spec::sum_dir::determined
