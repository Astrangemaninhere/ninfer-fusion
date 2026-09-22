#pragma once

// src/spec/recall_identity.h -- THE identity of a recall unit, and the arithmetic that lets
// recall units overlap each other and the mechanical Paged-KV pages underneath them.
//
// Independent feature. EXPERIMENTAL. Host-only, std-only: no CUDA, no engine headers, no
// artifact, so it compiles and unit-tests under plain `g++` (tests/test_recall_identity.cpp).
//
// WHY THIS FILE EXISTS (the reconciliation 100-8 forces)
//
//   The tree currently carries THREE spellings of "the digest of a span of tokens":
//
//     * src/targets/qwen3_6/impl/runtime/prefix_identity.cpp:97-117 -- the engine's rolling
//       digest, one value per token frontier, 128 bits, POSITION-BEARING (it mixes the three
//       MRoPE positions at :104-106). Its class comment already says what it is:
//       prefix_identity.h:37-39 "This is only a content shortlist: exact token and
//       ResidentPrefixIdentity comparison remains authoritative for reuse."
//     * src/product/kv_recall_block.h:187-208 -- 64 bits, content-only, the PUBLISHED FNV-1a
//       pair, per logical page.
//     * src/spec/sum_dir.h:155-168 -- 128 bits, content-only, the ENGINE's pair
//       (prefix_identity.cpp:60-61), per catalogue row.
//
//   Three spellings of one concept is the drift this file removes: from here on there is ONE
//   identity function, `recall_identity_tokens`, and the other two are spellings of it. The
//   unification is not asserted in a comment -- tests/test_recall_identity.cpp section 4 proves
//   this function and sum_dir_block_digest return the SAME 128 bits for the same tokens, and
//   section 5 pins the 64-bit legacy digest to the published FNV-1a pair so the two can never
//   be silently swapped for one another.
//
// THE ONE RULE
//
//   IDENTITY IS CONTENT. POSITION IS PROVENANCE.
//
//   A recalled block is re-prefilled as NEW input and therefore regenerated at NEW positions
//   (100-8, 102-1). Anything positional on the identity would make a block fail to match its
//   own record -- the exact failure mode that killed the naive "key the journal by the engine's
//   rolling digest" plan (the B-2 gap: prefix_identity.cpp:104-106 puts the position in, so a
//   recalled copy has a different key than the original by construction). So `RecallIdentity`
//   is fed the token ids and their count and NOTHING ELSE, and every positional quantity lives
//   in `RecallSpan` where it is explicitly labelled provenance.
//
// WHY A BLOCK MAY NOW OVERLAP ANOTHER BLOCK (100-8, the "compression may intrude a little")
//
//   The user's two statements, taken together, forbid the disjoint-interval model the tree
//   assumed:
//     * "分块绝对不能机械分 ... 必须由模型自动发起、整体分块" -- the SEMANTIC block is chosen by
//       the model, variable length, sometimes very large, and it is the ONLY unit of recall.
//     * "正常 64 为单位，但压缩时不完全如此 -- 可以稍微侵入上下文（小幅重叠），只要别漏" --
//       a compression boundary may overlap its neighbour, so the intervals are NOT disjoint.
//   And the mechanical page (64 tokens, across all text layers: cold_host_tier.h:34-41,
//   product/kv_recall_block.h:64-74) is the KV STORAGE and EVICTION unit only. It is not, and
//   may never become, the recall unit -- a 64-token page necessarily cuts a sentence or a
//   function in half (100-8 last line).
//
//   So this header keeps the two axes apart and maps between them:
//       semantic spans  -->  recall_resolve_coverage()   (who owns which token, with overlap)
//       semantic spans  <--> recall_spans_touching_page() / recall_pages_of_span()  (many-to-many)
//   Many-to-many is the normal case here, not an error: one span may touch several pages, one
//   page may be touched by several spans, and a page fully covered by two spans is a page whose
//   bytes are reachable by either.
//
// WHAT THIS FILE DELIBERATELY DOES NOT DO
//
//   It does not re-prefill anything, does not touch KV, does not persist anything, and does not
//   decide when a recall is owed. It is the naming layer the recall path has to agree on first.

// ninfer/types.h declares TokenId (std::int32_t, include/ninfer/types.h:18) and is std-only,
// which is what keeps this header compilable by plain g++ -- the same include
// product/kv_recall_block.h:53 makes for the same reason.
#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::spec::recall_identity {

// ---------------------------------------------------------------------------
// granularity
// ---------------------------------------------------------------------------

// The mechanical unit, restated here so this header stays std-only: it must equal
// kPagedKVPageSize (core/paged_kv_cache.h:17) == kColdHostPageTokens
// (cold_host_tier.h:67-68) == kRecallBlockTokens (product/kv_recall_block.h:74) ==
// kSumDirBlockTokens (src/spec/sum_dir.h:90). Those four are pinned by VALUE in the test,
// because the authoritative headers cannot be included from a plain-g++ test
// (core/paged_kv_cache.h:7 pulls in <cuda_runtime_api.h>).
inline constexpr std::uint32_t kRecallPageTokens = 64U;

// "this segment is not owned by any span" / "this span index does not exist".
inline constexpr std::uint32_t kNoSpan = 0xFFFFFFFFU;

// ---------------------------------------------------------------------------
// the identity -- two-lane FNV-1a over the span's CONTENT ONLY
// ---------------------------------------------------------------------------

// THE ENGINE'S CONSTANTS, copied rather than re-derived (prefix_identity.cpp:60-61, and
// src/spec/sum_dir.h:105-110 copies the same pair for the same reason: a third digest scheme
// would be a third thing to keep in step). The lane-1 rotation is prefix_identity.cpp:66-75's
// (`rotl(value ^ 0x9e3779b97f4a7c15, 29)`), so this function and the engine's rolling digest
// share one mixing primitive and differ only in WHAT is fed to it.
inline constexpr std::uint64_t kRecallDigestOffset0  = 1469598103934665603ULL;
inline constexpr std::uint64_t kRecallDigestOffset1  = 7809847782465536322ULL;
inline constexpr std::uint64_t kRecallDigestPrime0   = 1099511628211ULL;
inline constexpr std::uint64_t kRecallDigestPrime1   = 14029467366897019727ULL;
inline constexpr std::uint64_t kRecallLaneSkew       = 0x9e3779b97f4a7c15ULL;
inline constexpr std::uint32_t kRecallLaneSkewRotate = 29U;

// Domain separation so a *block* digest can never be confused with the engine's *prefix*
// digest: the engine's token domain is 0x6e696e6665722d74 ("ninfer-t", prefix_identity.cpp:62)
// and only the low byte changes here (sum_dir.h:115 makes the identical choice).
inline constexpr std::uint64_t kRecallBlockDomain = 0x6e696e6665722d62ULL; // "ninfer-b"

// A DISCREPANCY REGISTER, not a constant to use. The tree contains two claims about what the
// FNV-1a 64 pair IS:
//
//   * product/kv_recall_block.h:188-189 says "The constant and the algorithm are the published
//     FNV-1a 64 pair (offset basis 0xcbf29ce484222325, prime 0x100000001b3)" and uses it.
//   * prefix_identity.cpp:60 uses 1469598103934665603, and the same value was copied onward
//     into src/spec/sum_dir.h:105.
//
// Those two numbers are NOT equal: the published basis is 14,695,981,039,346,656,037 and the
// engine's constant is 1,469,598,103,934,665,603, i.e. the engine's value has one digit fewer
// (the leading '4' of the published constant is absent). This is not a correctness bug -- FNV
// with any nonzero offset basis is a fine hash -- but it does mean the two files cannot both be
// "the published FNV-1a 64 pair", and a reader who "fixes" either one breaks every landed
// artifact keyed by it. The static_assert below is the tripwire: it makes the difference
// impossible to paper over, and it names both claimants.
inline constexpr std::uint64_t kPublishedFnv1a64OffsetBasis = 0xcbf29ce484222325ULL;
inline constexpr std::uint64_t kPublishedFnv1a64Prime       = 0x100000001b3ULL;
static_assert(kRecallDigestOffset0 != kPublishedFnv1a64OffsetBasis,
              "the engine's digest constants (prefix_identity.cpp:60-61) are NOT the published "
              "FNV-1a 64 pair (product/kv_recall_block.h:188-189 claims the published pair and "
              "uses it). Both are fine hashes; neither may be replaced by the other without a "
              "format version bump.");
static_assert(kRecallDigestPrime0 == kPublishedFnv1a64Prime,
              "the prime IS the published FNV-1a 64 prime in both spellings");

struct RecallIdentity {
    std::uint64_t lo = kRecallDigestOffset0;
    std::uint64_t hi = kRecallDigestOffset1;

    [[nodiscard]] friend bool operator==(const RecallIdentity& a,
                                         const RecallIdentity& b) noexcept {
        return a.lo == b.lo && a.hi == b.hi;
    }
    [[nodiscard]] friend bool operator!=(const RecallIdentity& a,
                                         const RecallIdentity& b) noexcept {
        return !(a == b);
    }
    // Total order so records can be sorted and binary-searched. Only equality is semantic;
    // this is a stable arrangement and nothing more (same choice as sum_dir.h:130-132).
    [[nodiscard]] friend bool operator<(const RecallIdentity& a, const RecallIdentity& b) noexcept {
        return a.lo != b.lo ? a.lo < b.lo : a.hi < b.hi;
    }
    // The pristine value, i.e. "no identity". Unreachable for real content (the zero-lane pin
    // below makes it so), which is exactly why every caller may use it as the invalid value.
    [[nodiscard]] constexpr bool is_unset() const noexcept {
        return lo == kRecallDigestOffset0 && hi == kRecallDigestOffset1;
    }
};

inline void recall_identity_mix(RecallIdentity& identity, std::uint64_t value) noexcept {
    const std::uint64_t skewed = value ^ kRecallLaneSkew;
    const std::uint64_t lane1 =
        skewed << kRecallLaneSkewRotate | skewed >> (64U - kRecallLaneSkewRotate);
    identity.lo ^= value;
    identity.lo *= kRecallDigestPrime0;
    identity.hi ^= lane1;
    identity.hi *= kRecallDigestPrime1;
}

// THE identity of one span: the token ids, in order, and their count, and nothing else.
//
// Deliberately absent: the block index, the absolute token offset, the page, the session, the
// device cold slot, the spill file slot, the catalogue text, the wall clock. Each of those
// changes when the same content is recalled into a new place, so each would make two copies of
// one block compare unequal.
//
// The count is folded IN (as the second thing mixed) rather than appended, so a span and its own
// prefix cannot collide at different lengths. That is the same length binding
// product/kv_recall_block.h:203-206 performs with an explicit fold-and-multiply.
[[nodiscard]] inline RecallIdentity recall_identity_tokens(const TokenId* tokens,
                                                           std::size_t count) noexcept {
    RecallIdentity identity;
    recall_identity_mix(identity, kRecallBlockDomain);
    recall_identity_mix(identity, static_cast<std::uint64_t>(count));
    for (std::size_t i = 0; i < count; ++i) {
        // TokenId is int32 and ids are validated into [0, TextConfig::token_domain) before a
        // plan is built (request_plan_impl.h:235-241), so the widening is a faithful encoding of
        // the id and not of its sign -- the argument product/kv_recall_block.h:192-196 makes.
        recall_identity_mix(identity, static_cast<std::uint64_t>(static_cast<std::uint32_t>(tokens[i])));
    }
    // A zero lane is pinned to 1 so that the pristine value above stays unreachable and no
    // caller has to special-case "0 means nothing" (prefix_identity.cpp:113-115, sum_dir.h:165).
    if (identity.lo == 0) { identity.lo = 1; }
    if (identity.hi == 0) { identity.hi = 1; }
    return identity;
}

[[nodiscard]] inline RecallIdentity recall_identity_tokens(std::span<const TokenId> tokens) noexcept {
    return recall_identity_tokens(tokens.data(), tokens.size());
}

// ---------------------------------------------------------------------------
// the span -- a SEMANTIC recall unit: model-chosen, variable length, may overlap
// ---------------------------------------------------------------------------

// A span is a token range plus the identity of its content. `token_begin`/`token_count` are
// PROVENANCE in the caller's token space (they are how an answer is reported and how a
// candidate is checked against the sequence it came from); they are NOT part of the identity,
// and two spans with equal content at different offsets are the same block.
//
// There is no length limit and no alignment here on purpose: 100-1 says the block is chosen
// whole even when it is large, and 100-8 says the mechanical page is not the recall unit.
struct RecallSpan {
    std::uint32_t token_begin = 0; // provenance
    std::uint32_t token_count = 0; // 0 is never a block; see recall_span_well_formed()
    RecallIdentity identity{};

    [[nodiscard]] constexpr std::uint32_t token_end() const noexcept {
        return token_begin + token_count; // cannot overflow: both are bounded by the ledger
    }
};

[[nodiscard]] inline bool recall_span_well_formed(const RecallSpan& span) noexcept {
    if (span.token_count == 0) { return false; }
    if (span.identity.is_unset()) { return false; }
    if (span.identity.lo == 0 || span.identity.hi == 0) { return false; }
    if (span.token_end() < span.token_begin) { return false; } // wrap
    return true;
}

// Build a span from the ledger the text lives in. The identity is COMPUTED here, from `ledger`
// and from nothing else -- an API taking a caller-supplied identity would be an API through
// which position could sneak back in (sum_dir.h:534-536 makes the identical argument).
// std::nullopt when the range does not lie inside the ledger: a span that claims tokens the
// ledger does not have is not a short span, it is a wrong span.
[[nodiscard]] inline std::optional<RecallSpan>
recall_make_span(std::span<const TokenId> ledger, std::uint32_t token_begin,
                 std::uint32_t token_count) noexcept {
    if (token_count == 0) { return std::nullopt; }
    const std::uint64_t begin = token_begin;
    const std::uint64_t end   = begin + token_count;
    if (end > ledger.size()) { return std::nullopt; }
    RecallSpan span;
    span.token_begin = token_begin;
    span.token_count = token_count;
    span.identity    = recall_identity_tokens(
        ledger.subspan(static_cast<std::size_t>(begin), token_count));
    return span;
}

[[nodiscard]] inline bool recall_spans_overlap(const RecallSpan& a, const RecallSpan& b) noexcept {
    if (a.token_count == 0 || b.token_count == 0) { return false; }
    return a.token_begin < b.token_end() && b.token_begin < a.token_end();
}

[[nodiscard]] inline bool recall_span_covers(const RecallSpan& span, std::uint32_t token_begin,
                                             std::uint32_t token_count) noexcept {
    if (token_count == 0) { return false; }
    return span.token_begin <= token_begin && token_begin + token_count <= span.token_end();
}

// The shared token range of two spans, in ABSOLUTE coordinates, or nullopt when they do not
// overlap. This is the primitive the "small overlap is allowed" rule needs: an overlap is only
// admissible when both sides agree about the tokens in it (see recall_overlap_verdict).
[[nodiscard]] inline std::optional<std::pair<std::uint32_t, std::uint32_t>>
recall_overlap_range(const RecallSpan& a, const RecallSpan& b) noexcept {
    if (!recall_spans_overlap(a, b)) { return std::nullopt; }
    const std::uint32_t begin = a.token_begin > b.token_begin ? a.token_begin : b.token_begin;
    const std::uint32_t end   = a.token_end() < b.token_end() ? a.token_end() : b.token_end();
    return std::make_pair(begin, static_cast<std::uint32_t>(end - begin));
}

// ---------------------------------------------------------------------------
// the mechanical axis: pages <--> spans, many-to-many
// ---------------------------------------------------------------------------

// The pages a span touches, inclusive. A span is NOT page-aligned and need not be: a span that
// starts at token 60 and ends at token 70 touches pages 0 and 1. `page_tokens` must be
// kRecallPageTokens; the argument exists so a caller cannot silently pass a different block
// size without this header noticing.
[[nodiscard]] inline std::pair<std::uint32_t, std::uint32_t>
recall_pages_of_span(const RecallSpan& span, std::uint32_t page_tokens = kRecallPageTokens) noexcept {
    if (span.token_count == 0 || page_tokens == 0) { return {0U, 0U}; }
    const std::uint32_t first = span.token_begin / page_tokens;
    const std::uint32_t last  = (span.token_end() - 1U) / page_tokens;
    return {first, last};
}

// Every span that touches `page` -- the reverse direction of the same mapping. A page covered by
// two spans yields two indices, in ascending order: that IS the many-to-many relation, and a
// caller must not treat the second one as a duplicate.
[[nodiscard]] inline std::vector<std::size_t>
recall_spans_touching_page(std::span<const RecallSpan> spans, std::uint32_t page,
                           std::uint32_t page_tokens = kRecallPageTokens) noexcept {
    std::vector<std::size_t> hits;
    if (page_tokens == 0) { return hits; }
    for (std::size_t index = 0; index < spans.size(); ++index) {
        const auto pages = recall_pages_of_span(spans[index], page_tokens);
        if (spans[index].token_count != 0 && page >= pages.first && page <= pages.second) {
            hits.push_back(index);
        }
    }
    return hits;
}

[[nodiscard]] inline std::vector<std::size_t>
recall_spans_covering(std::span<const RecallSpan> spans, std::uint32_t token_begin,
                      std::uint32_t token_count) noexcept {
    std::vector<std::size_t> hits;
    if (token_count == 0) { return hits; }
    for (std::size_t index = 0; index < spans.size(); ++index) {
        if (recall_span_covers(spans[index], token_begin, token_count)) { hits.push_back(index); }
    }
    return hits;
}

[[nodiscard]] inline std::vector<std::size_t>
recall_spans_overlapping(std::span<const RecallSpan> spans, std::uint32_t token_begin,
                         std::uint32_t token_count) noexcept {
    RecallSpan probe;
    probe.token_begin = token_begin;
    probe.token_count = token_count;
    std::vector<std::size_t> hits;
    if (token_count == 0) { return hits; }
    for (std::size_t index = 0; index < spans.size(); ++index) {
        if (recall_spans_overlap(spans[index], probe)) { hits.push_back(index); }
    }
    return hits;
}

// ---------------------------------------------------------------------------
// overlap resolution -- "the overlapping ranges must all be parseable"
// ---------------------------------------------------------------------------

// A segment is a maximal token range owned by exactly ONE span. `covering` is how many spans
// cover it (>= 1 for a segment in `segments`); > 1 means the segment is inside an overlap, which
// is legal and is reported rather than rejected.
struct RecallCoverageSegment {
    std::uint32_t token_begin = 0;
    std::uint32_t token_count = 0;
    std::uint32_t span_index  = kNoSpan; // kNoSpan == an uncovered segment
    std::uint32_t covering    = 0;
};

struct RecallCoverage {
    std::vector<RecallCoverageSegment> segments; // ascending, contiguous, merged
    std::uint32_t covered_tokens   = 0;
    std::uint32_t uncovered_tokens = 0;
    std::uint32_t overlap_segments = 0;
    std::uint32_t max_covering     = 0;
};

// THE WINNER RULE, stated once because every consumer must agree on it:
//
//   the segment is owned by the SMALLEST covering span (fewest tokens); ties are broken by the
//   smaller token_begin, then by the lower index.
//
// Why smallest: a recall unit is a semantic block, so of two blocks that both cover a token the
// more specific (smaller) one is the one whose description actually claims that token. Why the
// tie-break matters: the partition is then a pure function of the SET of spans, so a caller may
// append spans in any order and get the same answer -- and the same winning span for every
// segment. (`span_index` is a position in the caller's array and therefore permutes with the
// input; the winning SPAN does not. The test checks the stronger property.)
//
// The result is a partition of [0, valid_tokens) into covered segments (one owner each) and
// uncovered gaps. Uncovered tokens are NOT an error: they are the part of the sequence that has
// no recall unit, i.e. the part that cannot be recalled and must stay resident.
[[nodiscard]] inline RecallCoverage recall_resolve_coverage(std::span<const RecallSpan> spans,
                                                            std::uint32_t valid_tokens) {
    RecallCoverage coverage;
    if (valid_tokens == 0 || spans.empty()) {
        coverage.uncovered_tokens = valid_tokens;
        return coverage;
    }

    // Clip each span to the valid range; a span that lies entirely outside contributes nothing.
    std::vector<RecallSpan> live;
    live.reserve(spans.size());
    std::vector<std::uint32_t> live_index;
    live_index.reserve(spans.size());
    for (std::size_t index = 0; index < spans.size(); ++index) {
        const RecallSpan& span = spans[index];
        if (span.token_count == 0 || span.token_begin >= valid_tokens) { continue; }
        const std::uint32_t end = span.token_end() < valid_tokens ? span.token_end() : valid_tokens;
        RecallSpan clipped = span;
        clipped.token_count = end - span.token_begin;
        live.push_back(clipped);
        live_index.push_back(static_cast<std::uint32_t>(index));
    }
    if (live.empty()) {
        coverage.uncovered_tokens = valid_tokens;
        return coverage;
    }

    std::vector<std::uint32_t> points;
    points.reserve(live.size() * 2U + 2U);
    points.push_back(0U);
    points.push_back(valid_tokens);
    for (const RecallSpan& span : live) {
        points.push_back(span.token_begin);
        points.push_back(span.token_end());
    }
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());

    for (std::size_t i = 0; i + 1 < points.size(); ++i) {
        const std::uint32_t begin = points[i];
        const std::uint32_t end   = points[i + 1];
        if (end <= begin) { continue; }

        std::uint32_t best        = kNoSpan;
        std::uint32_t best_count  = 0;
        std::uint32_t best_begin  = 0;
        std::uint32_t covering    = 0;
        for (std::size_t k = 0; k < live.size(); ++k) {
            if (!recall_span_covers(live[k], begin, end - begin)) { continue; }
            ++covering;
            const bool better = best == kNoSpan || live[k].token_count < best_count ||
                                (live[k].token_count == best_count &&
                                 live[k].token_begin < best_begin);
            if (better) {
                best       = live_index[k];
                best_count = live[k].token_count;
                best_begin = live[k].token_begin;
            }
        }
        if (covering > 1 && covering > coverage.max_covering) { coverage.max_covering = covering; }

        RecallCoverageSegment segment;
        segment.token_begin = begin;
        segment.token_count = end - begin;
        segment.span_index  = best;
        segment.covering    = covering;
        if (covering > 1) { ++coverage.overlap_segments; }
        if (best == kNoSpan) {
            coverage.uncovered_tokens += segment.token_count;
        } else {
            coverage.covered_tokens += segment.token_count;
        }
        // Merge with the previous segment when they agree, so `segments` is canonical: one entry
        // per (owner, covering) run rather than one per event interval.
        if (!coverage.segments.empty()) {
            RecallCoverageSegment& last = coverage.segments.back();
            if (last.span_index == segment.span_index && last.covering == segment.covering &&
                last.token_begin + last.token_count == segment.token_begin) {
                last.token_count += segment.token_count;
                continue;
            }
        }
        coverage.segments.push_back(segment);
    }
    return coverage;
}

// ---------------------------------------------------------------------------
// two sources of one block: does the overlap AGREE?
// ---------------------------------------------------------------------------

// The reason an overlap needs checking at all: two spans taken from ONE ledger share their
// tokens by construction (there is only one token stream), but a span taken from a RETRIEVED
// copy of the text -- a recalled block, a re-read spill record, a catalogue entry whose bytes
// were rewritten -- is a second source, and the two sources are only the same block if they
// agree token-for-token on the shared range. This is the check that makes "a small overlap may
// intrude" safe rather than hopeful.
struct RecallOverlapVerdict {
    bool overlap          = false;
    bool agree            = true;
    std::uint32_t shared_begin = 0; // absolute token index of the shared range
    std::uint32_t shared_count = 0;
    std::uint32_t first_mismatch_offset = 0; // offset INSIDE the shared range
};

// `left_tokens` are the tokens of `left`, `right_tokens` those of `right` (each starting at its
// own span's token_begin). Only the shared range is compared; a mismatch is reported at its
// first offset rather than counted, because one offset is enough to say "these are not the same
// text" and a count would invite a tolerance where none is intended.
[[nodiscard]] inline RecallOverlapVerdict
recall_overlap_verdict(const RecallSpan& left, std::span<const TokenId> left_tokens,
                       const RecallSpan& right, std::span<const TokenId> right_tokens) noexcept {
    RecallOverlapVerdict verdict;
    const auto shared = recall_overlap_range(left, right);
    if (!shared.has_value()) { return verdict; }
    verdict.overlap      = true;
    verdict.shared_begin = shared->first;
    verdict.shared_count = shared->second;

    const std::uint64_t left_offset  = shared->first - left.token_begin;
    const std::uint64_t right_offset = shared->first - right.token_begin;
    if (left_offset + shared->second > left_tokens.size() ||
        right_offset + shared->second > right_tokens.size()) {
        // One side does not even carry the tokens it claims: that is a shortage, not a
        // disagreement, and it must not read as "agree".
        verdict.agree                = false;
        verdict.first_mismatch_offset = 0;
        return verdict;
    }
    for (std::uint32_t i = 0; i < shared->second; ++i) {
        if (left_tokens[static_cast<std::size_t>(left_offset) + i] !=
            right_tokens[static_cast<std::size_t>(right_offset) + i]) {
            verdict.agree                 = false;
            verdict.first_mismatch_offset = i;
            return verdict;
        }
    }
    return verdict;
}

// ---------------------------------------------------------------------------
// the probe -- violations as return values, never as throws
// ---------------------------------------------------------------------------

// The same shape as product/kv_recall_block.h:261-316 recall_block_probe and
// serve/kv_cold_policy.h:292-333 invariant_violations: one line per violation, empty when the
// set is sound, so it can be asserted after every write in a test and reported (never thrown) in
// production. The identity is re-derived from the ledger rather than trusted, because a stored
// digest and a recomputed one must be the SAME function over the SAME bytes.
[[nodiscard]] inline std::vector<std::string>
recall_span_probe(std::span<const RecallSpan> spans, std::span<const TokenId> ledger) {
    std::vector<std::string> bad;
    const auto fail = [&bad](std::string message) { bad.push_back(std::move(message)); };

    for (std::size_t index = 0; index < spans.size(); ++index) {
        const RecallSpan& span = spans[index];
        const std::string tag  = "span " + std::to_string(index) + ": ";
        if (span.token_count == 0) {
            fail(tag + "empty span is not a recall unit");
            continue;
        }
        if (span.identity.is_unset()) {
            fail(tag + "identity is unset");
            continue;
        }
        const std::uint64_t end = static_cast<std::uint64_t>(span.token_begin) + span.token_count;
        if (end > ledger.size()) {
            fail(tag + "range [" + std::to_string(span.token_begin) + ", " +
                 std::to_string(end) + ") exceeds ledger size " + std::to_string(ledger.size()));
            continue;
        }
        const RecallIdentity recomputed = recall_identity_tokens(
            ledger.subspan(span.token_begin, span.token_count));
        if (recomputed != span.identity) {
            fail(tag + "identity does not re-derive from the ledger range");
        }
    }
    return bad;
}

// Report every overlapping pair whose shared range does NOT agree. `tokens_of` supplies one
// token stream per span; passing the same ledger for every span is the one-ledger case and
// always agrees. Kept separate from recall_span_probe because it needs the second source, not
// just the spans.
[[nodiscard]] inline std::vector<std::string>
recall_overlap_probe(std::span<const RecallSpan> spans,
                     const std::vector<std::span<const TokenId>>& tokens_of) {
    std::vector<std::string> bad;
    if (tokens_of.size() != spans.size()) {
        bad.push_back("tokens_of must have one entry per span");
        return bad;
    }
    for (std::size_t a = 0; a < spans.size(); ++a) {
        for (std::size_t b = a + 1; b < spans.size(); ++b) {
            const RecallOverlapVerdict verdict =
                recall_overlap_verdict(spans[a], tokens_of[a], spans[b], tokens_of[b]);
            if (!verdict.overlap || verdict.agree) { continue; }
            bad.push_back("spans " + std::to_string(a) + " and " + std::to_string(b) +
                          " overlap at [" + std::to_string(verdict.shared_begin) + ", " +
                          std::to_string(verdict.shared_begin + verdict.shared_count) +
                          ") and disagree at offset " +
                          std::to_string(verdict.first_mismatch_offset));
        }
    }
    return bad;
}

// ---------------------------------------------------------------------------
// the locator -- what a POSITIONAL digest is allowed to be used for
// ---------------------------------------------------------------------------

// The engine's rolling digest (prefix_identity.cpp:97-117) mixes the three MRoPE positions at
// :104-106, so it is a function of (content, position). That makes it unfit to NAME a block --
// a recalled copy is regenerated at new positions and would carry a different value for the same
// text, which is the B-2 gap named in the work log. It remains perfectly good for its own job:
// finding WHERE in a resident prefix a candidate begins. This type exists so that job has a name
// and a place, and so nothing else in the recall path can quietly pass a rolling digest where an
// identity is required.
inline constexpr bool kEngineRollingDigestIsPositional = true;

struct RecallLocator {
    std::array<std::uint64_t, 2> rolling_digest{}; // the engine's, position-bearing
    std::uint32_t frontier = 0;                    // the token frontier it was taken at

    [[nodiscard]] friend bool operator==(const RecallLocator& a,
                                         const RecallLocator& b) noexcept = default;
};

// The mapping rule between the two digests, in one function so the direction can never be
// reversed by accident: an identity is derived from tokens; a locator can only be handed in by
// whoever holds the engine's rolling digests, and it never yields an identity.
[[nodiscard]] inline RecallIdentity
recall_identity_of_ledger_range(std::span<const TokenId> ledger, std::uint32_t token_begin,
                                std::uint32_t token_count) noexcept {
    const std::uint64_t end = static_cast<std::uint64_t>(token_begin) + token_count;
    if (token_count == 0 || end > ledger.size()) { return RecallIdentity{}; }
    return recall_identity_tokens(ledger.subspan(token_begin, token_count));
}

} // namespace ninfer::spec::recall_identity
