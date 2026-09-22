#pragma once

// src/spec/sum_dir_query_key.h -- THE PRODUCER OF THE QUERY KEY. (line `indexgaps`, gap (1))
//
// WHAT WAS MISSING, NAMED EXACTLY
//   `DETERMINED(query_key, store)`'s clause (D1) is
//
//       EXISTS! b IN store : K(b) == query_key
//
//   and `K(b)` is `SumDirRow::block_identity`, a 128-bit CONTENT identity computed by
//   `sum_dir_block_digest()` from the block's tokens and from nothing else (sum_dir.h:212-227,
//   written by `SumDir::append` at sum_dir.h:1609). The CONSUMER of a key already exists and is
//   registered: `QBlockIdentity` -- "which block has content identity H?" -- with the answerer
//   `SumDir::find` (spec/sum_dir_query_registry.h:99-101, sum_dir.h:1747-1761).
//
//   What did NOT exist anywhere in the tree is a PRODUCER of `H`: a function that takes the query's
//   own token ids and yields a value in `block_identity`'s space. That is this file. It invents no
//   key space: it calls the SAME function `SumDir::append` calls, with the SAME domain separation,
//   so a key produced here is by construction the identity of a block built from those ids.
//
// THE ALTERNATIVE SOURCE, REFUSED BY NAME, WITH ITS REASON
//   The obvious candidate is the engine's own rolling digest over the live prefix -- "the model's
//   hidden state", already materialised per prefix. It is REFUSED, and the tree refuses it itself:
//
//       recall_identity.h:588-603 -- "The engine's rolling digest ... mixes the three MRoPE
//       positions at :104-106, so it is a function of (content, POSITION). That makes it unfit to
//       NAME a block -- a recalled copy is regenerated at new positions and would carry a different
//       value for the same text."
//
//   and it is pinned by a constant, `inline constexpr bool kEngineRollingDigestIsPositional = true;`
//   (recall_identity.h:595). A determination must survive a re-prefill; a LOCATOR cannot. The type
//   that carries it, `RecallLocator`, exists for the job it IS good at -- finding WHERE in a
//   resident prefix a candidate begins -- and `sum_dir_query_key_source_admitted()` below returns
//   false for it, at compile time.
//
//   The other two refused sources are refused for their own named reasons, not by association:
//     * an EMBEDDING vector -- a similarity score may narrow (the candidate-set UNION) and may
//       never determine; the tree says so at program_impl.h:14418-14422 ("exact-or-silent, never
//       approximately right ... belongs in the candidate-set UNION, never in place of this
//       arbiter"), and `sum_dir_vector.h`'s own default offer needs an embedder that does not exist
//       in the tree, so it cannot produce a KEY at all;
//     * a CALLER-SUPPLIED identity -- sum_dir.h:1372-1383 refuses it for `append` in the same words
//       this file inherits: "an API taking a caller-supplied identity would be an API through which
//       position could sneak back in".
//
// WHAT THIS HEADER DOES NOT DO, NAMED SO IT IS NOT MISTAKEN FOR DONE
//   * It does not tokenise. It takes token ids, exactly like `SumDir::append`. The tree's inference
//     runtime has no tokenizer in it, and this header does not add one.
//   * It does not decide WHICH ids a query supplies. A caller that wants the block a live span
//     covers hands the span's ids; a caller that wants "does this block exist at all" hands the same
//     ids it would hand `append`. It refuses an empty span and a span longer than one block, and it
//     says which of the two it refused.
//   * It is not wired into the engine: it has no engine-side caller, the same property
//     `sum_dir_vector.h:49-52` claims for itself ("so that landing it CANNOT move the engine binary
//     by one byte"). `QBlockIdentity`'s answerer is unchanged.

#include "spec/recall_identity.h"
#include "spec/sum_dir.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace ninfer::spec::sum_dir::query_key {

// ---------------------------------------------------------------------------
// where a key may come from -- one admitted source, four refused by name
// ---------------------------------------------------------------------------

enum class SumDirQueryKeySource : std::uint8_t {
    // ADMITTED, and it is the only one. The query's own token ids, in order, re-hashed by the very
    // function that hashed the row (`sum_dir_block_digest`), so the key lives in the row's space by
    // construction rather than by agreement.
    QueryIds = 0,
    // REFUSED: position-bearing. See the header comment and recall_identity.h:588-603.
    EngineRollingDigest = 1,
    // REFUSED: same objection as above, plus it is not card-free.
    ModelHiddenState = 2,
    // REFUSED: a score may narrow, never determine (program_impl.h:14418-14422).
    EmbeddingVector = 3,
    // REFUSED: the caller does not get to name the block it is looking for (sum_dir.h:1372-1383).
    CallerSuppliedIdentity = 4,
};

[[nodiscard]] constexpr const char* sum_dir_query_key_source_name(
    SumDirQueryKeySource source) noexcept {
    switch (source) {
    case SumDirQueryKeySource::QueryIds: return "query-ids";
    case SumDirQueryKeySource::EngineRollingDigest: return "engine-rolling-digest";
    case SumDirQueryKeySource::ModelHiddenState: return "model-hidden-state";
    case SumDirQueryKeySource::EmbeddingVector: return "embedding-vector";
    case SumDirQueryKeySource::CallerSuppliedIdentity: return "caller-supplied-identity";
    }
    return "unknown";
}

// The predicate that makes the refusals executable rather than editorial. It is `constexpr`, so the
// static_asserts below fail the BUILD if a future edit admits one of the four.
[[nodiscard]] constexpr bool
sum_dir_query_key_source_admitted(SumDirQueryKeySource source) noexcept {
    return source == SumDirQueryKeySource::QueryIds;
}

// Why, in one sentence, for the four that are not admitted. `QueryIds` returns a sentence too, so a
// caller printing the reason never prints a null.
[[nodiscard]] constexpr const char* sum_dir_query_key_source_reason(
    SumDirQueryKeySource source) noexcept {
    switch (source) {
    case SumDirQueryKeySource::QueryIds:
        return "admitted: the query's own ids, re-hashed by sum_dir_block_digest -- the same "
               "function SumDir::append used for the row, so the key is in the row's space";
    case SumDirQueryKeySource::EngineRollingDigest:
        return "refused: kEngineRollingDigestIsPositional is true (recall_identity.h:595) -- the "
               "rolling digest is a function of (content, position), so it is a LOCATOR and unfit "
               "to NAME a block; a recalled copy carries a different value for the same text";
    case SumDirQueryKeySource::ModelHiddenState:
        return "refused: the model's hidden state is position-bearing and not card-free; the tree "
               "has already rejected it as an identity (recall_identity.h:588-603)";
    case SumDirQueryKeySource::EmbeddingVector:
        return "refused: a similarity score may enter the candidate-set UNION and nothing else "
               "(program_impl.h:14418-14422); it can never produce DETERMINED";
    case SumDirQueryKeySource::CallerSuppliedIdentity:
        return "refused: an identity a caller names is an API through which position sneaks back "
               "in (sum_dir.h:1372-1383)";
    }
    return "unknown";
}

// THE REFUSALS ARE LOAD-BEARING, NOT DECORATIVE: each of these fails the build, so a later edit
// that admits a refused source is a compile error at this line rather than a silent admission.
static_assert(recall_identity::kEngineRollingDigestIsPositional,
              "the refusal of the engine's rolling digest as a query key rests on this constant; "
              "if the engine's digest ever stops being positional, this refusal must be re-derived "
              "rather than inherited");
static_assert(sum_dir_query_key_source_admitted(SumDirQueryKeySource::QueryIds),
              "the query's own ids are the admitted producer of the query key");
static_assert(!sum_dir_query_key_source_admitted(SumDirQueryKeySource::EngineRollingDigest),
              "the engine's rolling digest is position-bearing: LOCATOR, not IDENTITY");
static_assert(!sum_dir_query_key_source_admitted(SumDirQueryKeySource::ModelHiddenState),
              "the model's hidden state is refused as an identity");
static_assert(!sum_dir_query_key_source_admitted(SumDirQueryKeySource::EmbeddingVector),
              "a score may narrow the candidate set; it may never determine");
static_assert(!sum_dir_query_key_source_admitted(SumDirQueryKeySource::CallerSuppliedIdentity),
              "a caller-named identity is how position sneaks back in");
static_assert(recall_identity::kRecallBlockDomain == kSumDirBlockDomain,
              "the two spellings of the block domain must be ONE value, or a key produced through "
              "recall_identity_tokens and a row hashed through sum_dir_block_digest would live in "
              "two domains that only look alike");

// ---------------------------------------------------------------------------
// the key
// ---------------------------------------------------------------------------

// The bound the producer enforces. `kSumDirBlockTokens` (64) is the block's own granularity; a
// SHORT tail block is legitimate (a row's length is `token_end - token_begin`, and the last block of
// a sequence may be shorter), so the producer admits 1..64 and refuses 0 and >64 by name instead of
// silently truncating to the bound.
inline constexpr std::uint32_t kSumDirQueryKeyMaxTokens = kSumDirBlockTokens;

// A query key IS a `SumDirDigest`, plus the count it was derived at. The count is carried because
// the length is folded INTO the digest (`sum_dir_block_digest` mixes it second), so a key derived at
// the wrong length is a different key rather than a wrong-looking match -- reporting the count makes
// that visible instead of mysterious.
struct SumDirQueryKey {
    SumDirDigest  identity{};                 // the 128-bit CONTENT identity, in the row's space
    std::uint32_t token_count = 0;            // how many ids it was derived from; 0 == no key
    SumDirQueryKeySource source = SumDirQueryKeySource::QueryIds;

    // The pristine value, i.e. "this is not a key". Unreachable for real content (the zero-lane pin
    // inside the digest makes it so), which is why every caller may use it as the invalid value.
    [[nodiscard]] constexpr bool is_unset() const noexcept {
        return token_count == 0 || identity.is_unset();
    }
    [[nodiscard]] friend bool operator==(const SumDirQueryKey& a,
                                         const SumDirQueryKey& b) noexcept {
        return a.identity == b.identity && a.token_count == b.token_count;
    }
};

// Everything a caller needs to tell "no key" from "not asked". Every refusal the producer can make
// has a named field, so a zero key is never a silent answer.
struct SumDirQueryKeyReport {
    SumDirQueryKeySource source = SumDirQueryKeySource::QueryIds;
    std::uint32_t requested_tokens = 0;
    bool on_axis              = false; // 1 <= count <= kSumDirQueryKeyMaxTokens
    bool refused_empty_span   = false; // ids == nullptr or count == 0
    bool refused_over_block   = false; // count > kSumDirQueryKeyMaxTokens
    bool identity_unset       = false; // the digest came back pristine -- cannot happen for real ids
};

// ---------------------------------------------------------------------------
// THE PRODUCER
// ---------------------------------------------------------------------------

// The one admitted producer. It re-derives; it never accepts a digest from the caller, which is the
// whole point (a caller-supplied digest is `CallerSuppliedIdentity`, refused above).
[[nodiscard]] inline SumDirQueryKey sum_dir_query_key_from_ids(
    const std::uint32_t* ids, std::uint32_t count,
    SumDirQueryKeyReport* report = nullptr) noexcept {
    SumDirQueryKey key;
    SumDirQueryKeyReport local;
    local.source = SumDirQueryKeySource::QueryIds;
    local.requested_tokens = count;
    key.source = SumDirQueryKeySource::QueryIds;

    if (ids == nullptr || count == 0) {
        local.refused_empty_span = true;
        local.identity_unset = true;
        if (report != nullptr) { *report = local; }
        return key; // token_count stays 0: "not a key"
    }
    if (count > kSumDirQueryKeyMaxTokens) {
        local.refused_over_block = true;
        local.identity_unset = true;
        if (report != nullptr) { *report = local; }
        return key;
    }

    // THE SAME CALL `SumDir::append` MAKES (sum_dir.h:1609). Not a restatement of it.
    key.identity = sum_dir_block_digest(ids, count);
    key.token_count = count;
    local.on_axis = true;
    local.identity_unset = key.identity.is_unset();
    if (report != nullptr) { *report = local; }
    return key;
}

[[nodiscard]] inline SumDirQueryKey
sum_dir_query_key_from_ids(const std::vector<std::uint32_t>& ids,
                           SumDirQueryKeyReport* report = nullptr) noexcept {
    return sum_dir_query_key_from_ids(
        ids.data(), static_cast<std::uint32_t>(ids.size()), report);
}

// The bridge clause, stated as a function so the direction cannot be reversed by accident: a key is
// the identity of a block built from the ids it was derived from, and nothing else is.
// `kSumDirBlockTokens`-length spans only, because the length is folded in.
[[nodiscard]] inline bool
sum_dir_query_key_is_identity_of(const SumDirQueryKey& key,
                                 const std::uint32_t* ids, std::uint32_t count) noexcept {
    if (key.is_unset() || count != key.token_count) { return false; }
    return sum_dir_block_digest(ids, count) == key.identity;
}

// ---------------------------------------------------------------------------
// THE CONSUMER -- `QBlockIdentity` answered by cardinality, never by rank
// ---------------------------------------------------------------------------

// `QBlockIdentity` asks "which block has content identity H?" (sum_dir_query_registry.h:99-101) and
// names `SumDir::find` as its answerer. This is the shape that call has to take for the answer to be
// a DETERMINATION: the candidate set is COUNTED, and `candidates != 1` is a named refusal.
enum class SumDirQueryKeyVerdict : std::uint8_t {
    Determined          = 0, // exactly one row carries this key
    RefusedKeyUnset     = 1, // the key is not a key
    RefusedNotSorted    = 2, // SumDir::find requires sort_rows(); we refuse instead of throwing
    RefusedNoCandidate  = 3, // zero rows -- a SILENT MISS if it were reported as "nothing found"
    RefusedAmbiguous    = 4, // more than one row -- never crowned by rank
};

[[nodiscard]] constexpr const char* sum_dir_query_key_verdict_name(
    SumDirQueryKeyVerdict verdict) noexcept {
    switch (verdict) {
    case SumDirQueryKeyVerdict::Determined: return "determined";
    case SumDirQueryKeyVerdict::RefusedKeyUnset: return "refused-key-unset";
    case SumDirQueryKeyVerdict::RefusedNotSorted: return "refused-not-sorted";
    case SumDirQueryKeyVerdict::RefusedNoCandidate: return "refused-no-candidate";
    case SumDirQueryKeyVerdict::RefusedAmbiguous: return "refused-ambiguous";
    }
    return "unknown";
}

struct SumDirQueryKeyLookup {
    SumDirQueryKeyVerdict verdict = SumDirQueryKeyVerdict::RefusedKeyUnset;
    std::uint32_t candidates = 0;   // the CARDINALITY, counted -- the whole of (D1)
    std::size_t   first_row  = 0;   // valid iff candidates >= 1
    std::size_t   last_row   = 0;   // inclusive, so [first_row, last_row] is the duplicate run
    std::uint32_t rows_total = 0;
};

[[nodiscard]] inline SumDirQueryKeyLookup
sum_dir_query_key_lookup(const SumDir& directory, const SumDirQueryKey& key) noexcept {
    SumDirQueryKeyLookup out;
    out.rows_total = static_cast<std::uint32_t>(directory.rows().size());
    if (key.is_unset()) {
        out.verdict = SumDirQueryKeyVerdict::RefusedKeyUnset;
        return out;
    }
    if (!directory.sorted()) {
        // `SumDir::find` THROWS here (sum_dir.h:1748). A determination is an answer, and an answer
        // that arrives as an exception is not one, so the refusal is named instead.
        out.verdict = SumDirQueryKeyVerdict::RefusedNotSorted;
        return out;
    }
    std::size_t begin = 0;
    std::size_t end   = 0;
    if (!directory.find(key.identity, begin, end)) {
        out.verdict = SumDirQueryKeyVerdict::RefusedNoCandidate;
        return out;
    }
    out.first_row = begin;
    out.last_row  = end > begin ? end - 1U : begin;
    out.candidates = static_cast<std::uint32_t>(end - begin);
    if (out.candidates == 1U) {
        out.verdict = SumDirQueryKeyVerdict::Determined;
    } else {
        // COUNTED, named, and deliberately not resolved: choosing among these by anything but an
        // exact content check would be the "plausible-looking wrong anchor" the tree forbids
        // (sum_dir_reach.h:104-108, program_impl.h:14502-14507).
        out.verdict = SumDirQueryKeyVerdict::RefusedAmbiguous;
    }
    return out;
}

} // namespace ninfer::spec::sum_dir::query_key
