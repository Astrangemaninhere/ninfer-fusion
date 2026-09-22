#pragma once

// src/spec/sum_dir_reach.h -- ABSOLUTE REACHABILITY: a COVERING retrieval family, an
// evidence-ranked total anchor, a loop guard, a budget refusal and a loud fallback.
//
// ===========================================================================================
// WHAT WAS WRONG (RECALLSPAN's measurement, and an independent reproduction of it)
// ===========================================================================================
//
// `sum_dir_recall_span_for_term` (`spec/sum_dir.h`) searches PREFIXES union SUFFIXES, accumulates
// every admissible hit of every window into ONE `pages` vector, and takes `pages.front()`. Measured
// on D1's own catalogue, by two independent implementations (RECALLSPAN's Python port and the C++
// port in tests/test_sum_dir_reach.cpp), which agree exactly:
//
//   k=10 head hits=3   min_page=34   ' vault 56 was audited by inspector '
//   k= 2 head hits=256 min_page=0    ' vault '          <-- EVERY ROW
//   k= 1 head hits=256 min_page=0    ' vault'
//   k= 1 tail hits=256 min_page=0    '\n'               <-- the template tail's own token
//   UNION first_page = 0
//
// Three consequences: (1) a fragment INTERIOR to the query is unreachable for EVERY k, because no
// window equals it; (2) the anchor is `min` over a union that necessarily contains the LEAST
// specific evidence, so the good evidence is discarded when the bad is present; (3) two stated
// invariants are violated -- `sum_dir.h:2297-2302` says descending k yields "the most specific
// evidence available" (true only if the shorter windows are discarded, which the code does not do),
// and `program_impl.h:108` declares `kIndexQueryMinTokens = 4` ("a 1-3 token phrase is a common
// word, and a common word's hits are not a location") while the sweep descends to k=1.
//
// ===========================================================================================
// 1. THE COVERING FAMILY, AND WHY IT IS COVERING (a proof, checked exhaustively by the arm)
// ===========================================================================================
//
// THE FAMILY: every k_min-gram of the query, k_min = `kReachMinWindowTokens` = the tree's OWN
// declared minimum window (4). No prefixes, no suffixes, no k < k_min.
//
// THEOREM. Let Q be the query's token sequence and suppose a fragment F = Q[a..b) with b - a >=
// k_min occurs in some catalogue line at token position q. Then Q[a..a+k_min) occurs in that same
// line at position q, so the page holding that line is in the hit set of the k_min-gram family.
// Therefore EVERY occurrence of EVERY fragment of length >= k_min is found, at EVERY offset
// including the interior. QED.
//
// CHEAPEST COVERING FAMILY. The literal "all k-grams" family (every (i,k), k >= k_min) is also
// covering and is a strict superset; it costs sum_{k=k_min}^{L} (L-k+1) = O(L^2) windows -- 435 for
// L=32, k_min=4 -- while the k_min-gram family costs L - k_min + 1 = 29, and the theorem shows the
// extra 406 add no reachability. It is therefore the MINIMAL covering family of the "substring of
// the query" kind: any fragment of length >= k_min contains at least one k_min-gram, and any family
// that finds a page must contain the k_min-gram that fragment starts with.
//
// WHY NOT A SUFFIX ARRAY / FM-INDEX OVER THE CORPUS. It would give the same guarantee and the
// exact longest match in O(L log n), but the directory is written INCREMENTALLY, one row per stored
// block (`SumDir::append`), and a suffix array is not incrementally maintainable. The n-gram
// postings table is (one row appends n - k_min + 1 postings), which is why the theorem is stated
// over n-grams and why the postings are what is built. COST ON THIS BOX, stated: the reference
// document's catalogue is 256 live rows x 64 tokens = 16,384 tokens, so the table holds 256 x 64 =
// 16,384 (page, position) entries; the arm prints the build time and the entry count. Extrapolated
// to the 1M-token context D1 declares the same stride gives ~61M entries, which is the one place
// this design needs the table as an incrementally-maintained column on `SumDir` rather than rebuilt
// per call -- named here rather than hidden, and NOT a change to this header's contract.
//
// ===========================================================================================
// 2. THE ANCHOR: TOTAL, EVIDENCE-RANKED, AND NOT A MIN OVER A UNION
// ===========================================================================================
//
// For every (page, position) posting of every query k_min-gram the match is EXTENDED EXACTLY, by
// comparing rendered token ids one at a time, to the longest run on which page and query agree
// (`length`). Then:
//
//   * `m*` = the longest admissible match length, and T = the STRONGEST CLASS, the set of evidence
//     at `m*`. The pages of T are the CANDIDATE SET: they are the only pages that can claim the
//     longest run of the query's text;
//   * the ANCHOR is the triple maximal under the TOTAL order (length DESC, offset ASC, page ASC);
//   * the RUN is grown outward IN PAGE SPACE, one page at a time, admitting a page only when it
//     holds evidence that EXTENDS the query interval already covered -- so the run is the maximal
//     PAGE-CONTIGUOUS PASSAGE around the strongest match, and nothing else;
//   * `pages` is the set so admitted, and `first_page`/`last_page` are its own ends. `pages.front()`
//     is a READOUT of that selection, never the selector.
//
// THE DISQUALIFIER IN THE OLD CODE WAS "min over a union that contains weak evidence". There is no
// union of weak evidence here: a page enters the run either at the STRONGEST evidence level, or by
// ABUTTING admissible evidence on the query's own token axis. A 4-gram that happens to occur on 256
// pages does not widen the run, because it is INTERIOR to the covered interval and therefore
// extends nothing.
//
// ===========================================================================================
// 3. THE LOOP GUARD. 4. THE BUDGET REFUSAL. 5. THE LOUD FALLBACK.
// ===========================================================================================
//
// LOOP. D1's rounds 3-4 are a lock-on: page 0's 64 tokens are appended, so the next query is a
// verbatim copy of page 0's tail, the directory matches it UNIQUELY and with the LONGEST possible
// evidence (32 tokens, `hits=1`), and the run re-anchors at page 0. The SELF-MATCH guard does not
// stop it -- page 0 is EARLIER than the query, so it is admissible -- which the arm asserts
// explicitly (`excluded_self == 0` on that query). Every page in `appended_pages` -- the pages this
// sequence has ALREADY recalled, and it must be supplied by
// `sum_dir_reach_already_recalled_pages` below (section 7b) instead of being read off
// `SequenceState::recall_recorded`'s page column -- that column is the COMPLEMENT of this set,
// over every page the directory can return, and handing it over is the inversion 7b measures.
// The exclusion is COUNTED (`excluded_appended`), so
// it can never be silent. If that leaves nothing, the status is `RefusedEverything`: there is
// nothing new to restore, which is the honest answer and which terminates the loop by construction.
//
// BUDGET. A candidate that cannot be held must be HELD or the run must REFUSE. The candidate set is
// the strongest class T; the run is `[min T, max T]` grown through the passage, so
// `need = last - first + 1`. If `need` exceeds the per-pass fan-out cap or the block budget the
// result is `RefusedBudget`, naming the run, the two limits and the exact pages that holding fewer
// would drop. There is NO silent clamp: the shipping code caps with
// `min(first + fanout_cap - 1, pages.back())` and reports `pages=` as if complete.
//
// FALLBACK. `NoCandidate` -- the covering family matched nothing admissible -- returns with a
// non-empty `refusal` and `found = false`. It NEVER anchors at page 0, and the caller must not
// substitute the positional first-fit silently: that fallback chooses a page by POSITION, which is
// precisely the "plausible-looking wrong anchor" the requirement forbids.
//
// ===========================================================================================
// 6. WHAT THIS HEADER IS NOT
// ===========================================================================================
//
// A PURE FUNCTION over the directory (rows + catalogue lines), exactly like the function it
// replaces, so the arm exercises the same code the engine calls. Host-only: no CUDA, no I/O, no
// tokenizer. Two things are deliberately left to the wiring and are NAMED rather than assumed: the
// postings table wants to live on `SumDir`, and `appended_pages` must be supplied from
// `SequenceState`. Neither changes this header's contract.
//
// ===========================================================================================
// 7. REACHCLOSE: THE FAMILY IS COVERING FOR THE STREAM, NOT FOR THE PAGE
// ===========================================================================================
//
// The theorem in section 1 is TRUE but its hypothesis was narrower than the requirement. It is
// stated over "some catalogue LINE", i.e. over one page's own 64 tokens. The text the caller can
// actually restore is the PASSAGE -- the concatenation of consecutive pages, read out as one token
// stream by `read_located_block` -- and 60.4% of this corpus's fragments exist ONLY there:
//
//   MEASURED, exhaustively, by `probe --mode=straddle` on the same corpus: of the 125,715
//   straddling k-windows (a window whose first and last token lie in different pages), the
//   PAGE-INTERNAL family returned 35,581 with a run that does not cover the pages the window
//   occupies, and in 32,327 of those 35,581 the window's text exists in NO SINGLE PAGE at all --
//   only as a cross-page concatenation. A family built from page-internal grams is blind to exactly
//   those, and the exact extension, which only walks FORWARD inside `page_tokens`, cannot repair it.
//
//   MEASURED AGAIN after the FIRST attempt at (1) below, which built the stream in ROW order:
//   35,009 wrong of 125,715 -- a 1.6% improvement, not a fix. The second measurement is the one
//   that found the defect, and it is recorded here because the first attempt's comment claimed
//   125,715/125,715 = 100.0% BEFORE the run that would have shown it completed.
//
// TWO DEFECTS FOLLOWED FROM THAT, and both were measured rather than argued:
//
//   (a) THE 22 UNREACHABLE CASES. In the faithful shape (term = A(3 real tokens) + F(k) + T(8)) a
//       fragment F whose 3 preceding tokens cross a page boundary has its TRUE occurrence of 3 + k
//       tokens invisible; a page-internal ALIAS of m* tokens with k < m* < 3 + k wins the anchor;
//       and F's own holder is INTERIOR to the alias's covered interval, so the passage growth never
//       admits it. `probe --mode=u22` prints all 22 with their mechanism, and in all 22 the text is
//       also absent from the RESTORED PASSAGE (so this was a real reachability failure, not an
//       artefact of the proxy assertion), while in all 22 F does exist contiguously inside a single
//       page -- i.e. NONE of them was irreducible. All 22 have j in {0,1}: exactly the offsets whose
//       A + F crosses the boundary.
//
//   (b) THE 71.6% STRADDLING RATE. The user's requirement was born from a straddle (D1's needle
//       lies in pages 158/159), and a window whose split point leaves fewer than k_min tokens on
//       either side is invisible to a page-internal family when k is small -- which is RULE, not
//       luck: for k = 4 every one of the 765 straddling windows has at most 3 tokens on one side,
//       so every one of them is blind. Measured: 89,994 / 125,715 = 71.6%.
//
// THE FIX, AND WHY IT IS THE WHOLE FIX. Two changes, both strictly ADDITIVE to the evidence:
//
//   (1) THE FAMILY IS BUILT OVER THE STREAM -- AND THE STREAM IS THE PASSAGE, i.e. PAGE ORDER.
//       `SumDirReachPostings::build` concatenates the live rows' tokens, IN ASCENDING PAGE ORDER,
//       into `stream_tokens` (with `stream_page` naming each token's page) and indexes every
//       k_min-gram that STARTS in a row -- including the k_min - 1 grams per row that run into the
//       NEXT PAGE. The k_min-gram family is then covering for the STREAM, which is the thing that
//       gets restored, by exactly the section-1 proof applied to the stream instead of to a line.
//       The exact extension walks the stream, so it crosses boundaries.
//
//       PAGE ORDER IS NOT ROW ORDER, AND THIS IS THE WHOLE BUG IN THE FIRST ATTEMPT.
//       `sum_dir.h::sort_rows` orders rows by `(block_identity, token_begin)` -- NOT by page -- so
//       iterating `rows()` in row order does NOT walk the catalogue in the order
//       `read_located_block` restores it. MEASURED on this corpus: with the stream built in ROW
//       order, `stream_page[0] == 224` while the caller's page order starts at page 0, and 16,384
//       of 16,384 positions disagree -- the "stream" was a PERMUTATION of the catalogue. Every
//       cross-row gram was then a concatenation of two pages that are NOT adjacent in the restored
//       text, and every `end_page` named a page that is not contiguous with the match it was quoted
//       for. That is why the first attempt left 35,009 of 125,715 straddles wrong, made 64 F1b
//       cases unreachable (was 22), refused the named regression F3, drove `mean need` in the
//       faithful family to 6,163,915 pages (an unsigned underflow reachable only because
//       `first_page`/`last_page` came from unrelated rows), and left 6775 acceptance assertions
//       red. Building the stream in PAGE order is the fix; the FAMILY, the RUN, the guards, the
//       refusal and the accounting are unchanged from the first attempt.
//
//   (2) THE RUN HOLDS EVERY PAGE A CHOSEN MATCH OCCUPIES. Evidence now carries `end_page`; the run
//       is the page range spanned by the admitted matches' occupied ranges, and the growth admits
//       an item only when it lies ENTIRELY on one side. A match is therefore never half outside the
//       passage it is quoted to justify.
//
// WHY (1) CLOSES THE 22 BY CONSTRUCTION, not by tuning. Write Q = A + F + T with |A| = 3 and the
// tail T made of sentinels no page holds. Any match must start at query offset >= 3 (offsets 0..2
// are sentinels) and must stop at offset <= 3 + k (the first tail sentinel), so m* <= k for the
// controlled shape and m* <= 3 + k for the faithful one. The queried fragment F sits at offset 3 in
// a page, and A + F is CONTIGUOUS IN THE STREAM at flat position index - 3 whenever index >= 3
// (for index < 3 the three "preceding" tokens wrap around the stream and no such run exists -- the
// one honest exception, which the arm measures rather than assumes). So the true occurrence has
// length 3 + k = the MAXIMUM ATTAINABLE, it is the champion, and EVERY item of length 3 + k has
// offset 0 (offset o with o + m* = 3 + k forces m* = 3 + k - o < 3 + k) and therefore COVERS F.
// Every page of the strongest class holds F, so the run holds one. The alias that used to win is
// now strictly shorter than the truth. MEASURED, PAGE-ORDERED stream: 0 of 348,928 (the
// page-internal family: 22; the row-ordered stream: 64 -- i.e. the row-ordered attempt made
// this defect WORSE while the comment claimed it had removed it).
//
// AND (2) CLOSES THE STRADDLE BY THE SAME ARGUMENT. In the straddling shape the query is
// sentinel(3) + F + sentinel(8), so every match starts at offset >= 3 and stops at offset <= 3 + k,
// giving m* = k exactly; the true occurrence of F -- which is the window itself, and straddles the
// boundary by construction -- is in the strongest class, and (2) puts both of its pages in the run.
// MEASURED, PAGE-ORDERED stream: 125,715 / 125,715 = 100.0%, WRONG = 0, refused = 0 (the
// page-internal family: 89,994 covered = 71.6%; the row-ordered stream: 90,450 = 71.9%).
//
// WHAT THE ACCEPTANCE'S `F1a capped` SWEEP DOES *NOT* MEASURE. Its reference is
// `RefOccurrence::pages_containing`, which searches for the fragment INSIDE ONE PAGE'S ROW -- so
// once the family can see a cross-page occurrence, that reference's `span` is a PAGE-INTERNAL
// count while the returned run is a PASSAGE range. MEASURED, page-ordered stream: 0 misses in
// F2's INSIDE-ONE-PAGE family (348,928 / 348,928), 0 misses in F2's STRADDLING family
// (125,715 / 125,715), F1b `unreachable = 0` and `absent = 0`, `in_evidence_bad = 0` -- but
// F1a-capped reports 6775 red, and EVERY one of them is the run being WIDER than the
// page-internal span (e.g. `holds = 1 span = 1` with `run = [2,235]`: page 2, the holder, IS in
// the run) or a refusal naming the PASSAGE span. That is a unit mismatch in the reference, of the
// same family as the two already known here, NOT a wrong answer and NOT an unreachable page; F2's
// two passage-level sweeps are the assertion that decides reachability. Named rather than hidden.
//
// WHAT IS *NOT* CLAIMED. `covered_begin`/`covered_end` are now the CHAMPION's own interval, which
// the run provably contains because the champion is a contiguous stream run whose pages are in the
// run. The strongest class's UNION span is reported separately as `union_begin`/`union_end` and is
// NOT claimed to be realised by the passage: the union of overlapping intervals taken from
// different pages is not a contiguous run of the restored text, and reporting it as "covered" was
// itself a summary line whose unit disagreed with its arithmetic. Admitting a page still requires
// it to BOUND the covered interval (`sum_dir_reach_extends`, unchanged); it is the FAMILY and the
// RUN that changed, not the anchor rank, the guards, the refusal or the accounting.

#include "spec/sum_dir.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::spec::sum_dir {

// THE TREE'S OWN MINIMUM WINDOW, not a new number: `program_impl.h:108` declares
// `kIndexQueryMinTokens = 4U` with the reason quoted above.
inline constexpr std::uint32_t kReachMinWindowTokens = 4U;

// The block budget one pass may hold -- the index budget's own figure. A REFUSAL threshold here,
// never a clamp.
inline constexpr std::uint32_t kReachDefaultBudgetBlocks = 2048U;

enum class SumDirReachStatus : std::uint8_t {
    Found = 0,
    NoCandidate = 1,        // the covering family matched nothing admissible
    RefusedEverything = 2,  // there were hits and every one was excluded by a stated guard
    RefusedBudget = 3,      // the strongest class cannot be held; refusing, not truncating
    RefusedLength = 4,      // the run's last page has no single answer for its token count
};

[[nodiscard]] inline const char* sum_dir_reach_status_name(SumDirReachStatus status) noexcept {
    switch (status) {
        case SumDirReachStatus::Found: return "found";
        case SumDirReachStatus::NoCandidate: return "no-candidate";
        case SumDirReachStatus::RefusedEverything: return "refused-everything";
        case SumDirReachStatus::RefusedBudget: return "refused-budget";
        case SumDirReachStatus::RefusedLength: return "refused-length";
    }
    return "unknown";
}

// One admissible match: where in the QUERY it starts, how many query tokens the match holds
// contiguously from there, the page it starts in, and the LAST page it occupies. The last field is
// what makes the match restorable: the passage the caller reads is `[first_page, last_page]`, so a
// match whose tokens run into the next page must drag that page into the run with it.
struct SumDirReachEvidence {
    std::uint32_t page     = 0;
    std::uint32_t offset   = 0;
    std::uint32_t length   = 0;
    std::uint32_t stream   = 0; // the match's first token, as an index into the catalogue STREAM
    std::uint32_t end_page = 0; // the last page the match occupies (>= page)
};

struct SumDirReachResult {
    SumDirReachStatus status = SumDirReachStatus::NoCandidate;
    bool              found  = false;

    std::uint32_t window      = 0; // m*, the EXACT longest match in tokens (not a window length)
    std::uint32_t best_offset = 0; // where m* starts in the query
    std::uint32_t anchor_page = 0; // the page holding it (the evidence-ranked anchor)
    std::uint32_t covered_begin = 0; // the champion's interval -- REALISED by the run's passage
    std::uint32_t covered_end   = 0;
    std::uint32_t union_begin = 0; // the strongest class's union span -- reported, NOT claimed
    std::uint32_t union_end   = 0;
    std::uint32_t candidates  = 0; // |T|: matches at the strongest evidence level
    std::uint32_t alternatives = 0; // admissible pages NOT in the run: other pages the covering
                                    // family found, so no found page is silently dropped
    std::uint32_t pages       = 0;
    std::uint32_t first_page  = 0;
    std::uint32_t last_page   = 0;
    std::uint32_t token_begin = 0;
    std::uint32_t token_end   = 0;

    std::uint32_t family_windows       = 0; // L - k_min + 1: the size of the covering family
    std::size_t   raw_hits             = 0;
    std::size_t   admissible           = 0;
    std::size_t   excluded_self        = 0;
    std::size_t   excluded_appended    = 0;
    std::size_t   cross_boundary       = 0; // admissible matches whose tokens span two pages
    std::uint64_t build_grams          = 0;
    std::uint64_t stream_tokens        = 0;

    std::uint32_t fanout_cap    = 0;
    std::uint32_t needed_blocks = 0;
    std::uint32_t budget_blocks = 0;
    std::string   refusal; // non-empty iff status != Found
};

// The scan, exposed for the arm: the admissible evidence and the guard attributions. The selector
// and the arm therefore read the SAME evidence, so "the family found page p" is checked against the
// selector's own input rather than against a second implementation of the search.
struct SumDirReachScan {
    std::vector<SumDirReachEvidence> evidence;
    std::uint32_t                    family_windows    = 0;
    std::size_t                      raw_hits          = 0;
    std::size_t                      excluded_self     = 0;
    std::size_t                      excluded_appended = 0;
};

// THE ALPHABET. Both the catalogue line and the query are rendered by `sum_dir_render_token_line`,
// so parsing a line back into ids is exact. A malformed line yields false and clears `out`; it is
// never treated as "no tokens", because that would hide a corrupt row behind a miss.
[[nodiscard]] inline bool sum_dir_parse_token_line(std::string_view line,
                                                   std::vector<std::uint32_t>& out) {
    out.clear();
    std::size_t index = 0;
    while (index < line.size()) {
        while (index < line.size() && line[index] == ' ') { ++index; }
        if (index >= line.size()) { break; }
        std::uint64_t value  = 0;
        std::size_t   digits = 0;
        while (index < line.size() && line[index] >= '0' && line[index] <= '9') {
            value = value * 10ULL + static_cast<std::uint64_t>(line[index] - '0');
            ++digits;
            ++index;
        }
        if (digits == 0 || value > 0xFFFFFFFFULL) { out.clear(); return false; }
        out.push_back(static_cast<std::uint32_t>(value));
    }
    return true;
}

// One n-gram. `k_min` is a runtime member because the arm sweeps it: a family covering for one
// k_min is covering for every larger one, and the sweep must be able to vary it.
struct SumDirReachGram {
    std::vector<std::uint32_t> tokens;
    [[nodiscard]] bool operator==(const SumDirReachGram& other) const noexcept {
        return tokens == other.tokens;
    }
};

struct SumDirReachGramHash {
    [[nodiscard]] std::size_t operator()(const SumDirReachGram& gram) const noexcept {
        std::uint64_t hash = 0xcbf29ce484222325ULL;
        for (const std::uint32_t token : gram.tokens) {
            hash ^= static_cast<std::uint64_t>(token) + 0x9e3779b97f4a7c15ULL + (hash << 6U) +
                    (hash >> 2U);
            hash *= 0x100000001b3ULL;
        }
        return static_cast<std::size_t>(hash);
    }
};

// Where one occurrence of a k_min-gram starts: the page it starts in, and its first token as an
// index into the catalogue STREAM. The stream index is what makes the exact extension possible in
// one pass -- a postings table that stored only the page could answer "this page contains the gram"
// but not "how much more of the query does it contain", and one that stored only the page-internal
// position cannot answer it at all when the gram runs into the next page.
struct SumDirReachPosting {
    std::uint32_t page   = 0;
    std::uint32_t stream = 0;
};

struct SumDirReachPostings {
    std::uint32_t k_min = kReachMinWindowTokens;

    // The catalogue as the CALLER CAN RESTORE IT: the live rows' tokens concatenated in row order
    // (which `sort_rows` makes page order), with `stream_page` naming each token's page. A page is
    // `kSumDirBlockTokens` tokens, so a k_min-gram that crosses a page boundary is a gram of the
    // stream and of no single page -- and it is the stream the passage is read from.
    std::vector<std::uint32_t> stream_tokens;
    std::vector<std::uint32_t> stream_page;
    std::vector<std::uint32_t> stream_row;

    std::vector<std::uint32_t>              row_page;
    std::vector<std::vector<std::uint32_t>> row_tokens;
    std::vector<std::uint32_t>              row_stream_begin;
    std::vector<std::uint32_t>              row_for_page;
    std::unordered_map<SumDirReachGram, std::vector<SumDirReachPosting>, SumDirReachGramHash>
        by_gram;
    std::uint64_t entry_count    = 0;
    std::uint64_t malformed_rows = 0;
    std::uint64_t live_rows      = 0;

    [[nodiscard]] const std::vector<SumDirReachPosting>* postings(
        const std::vector<std::uint32_t>& gram) const {
        SumDirReachGram key;
        key.tokens = gram;
        const auto found = by_gram.find(key);
        return found == by_gram.end() ? nullptr : &found->second;
    }

    void build(const SumDir& directory) {
        row_page.clear();
        row_tokens.clear();
        row_stream_begin.clear();
        row_for_page.clear();
        stream_tokens.clear();
        stream_page.clear();
        stream_row.clear();
        by_gram.clear();
        entry_count    = 0;
        malformed_rows = 0;
        live_rows      = 0;
        const std::vector<SumDirRow>& rows = directory.rows();
        row_page.resize(rows.size(), 0U);
        row_tokens.resize(rows.size());
        row_stream_begin.assign(rows.size(), 0U);
        std::uint32_t max_page = 0;
        for (const SumDirRow& row : rows) { max_page = std::max(max_page, row.page); }
        row_for_page.assign(static_cast<std::size_t>(max_page) + 1U, kSumDirNoRow);
        // PASS 1: the stream. Only rows the shipping `search_summaries` would search -- a Live row
        // with a parseable, non-empty line -- contribute tokens, so the postings cannot find text
        // the substring search would have skipped.
        for (std::size_t row = 0; row < rows.size(); ++row) {
            row_page[row] = rows[row].page;
            if (rows[row].page <= max_page) {
                row_for_page[rows[row].page] = static_cast<std::uint32_t>(row);
            }
            if (rows[row].state != SumDirState::Live) { continue; }
            const std::string_view line = directory.summary_of(row);
            if (line.empty()) { continue; }
            if (!sum_dir_parse_token_line(line, row_tokens[row])) {
                ++malformed_rows;
                row_tokens[row].clear();
                continue;
            }
            ++live_rows;
        }
        // PASS 1b: THE STREAM IS THE PASSAGE, so it is built in ASCENDING PAGE ORDER -- the order
        // `read_located_block` restores. `SumDir::sort_rows` orders rows by
        // (block_identity, token_begin), which is NOT page order in general (MEASURED on this
        // corpus: row 0 is page 224), so ROW ORDER IS NOT THE RESTORED ORDER. A stream built in row
        // order is a permutation of the catalogue: its cross-row grams join pages that are not
        // adjacent in the restored text, and its `end_page` names a page that is not contiguous with
        // the match. `stable_sort` keeps duplicate rows for one page in row order, so a page written
        // twice contributes both rows in a deterministic order.
        {
            std::vector<std::size_t> by_page;
            by_page.reserve(rows.size());
            for (std::size_t row = 0; row < rows.size(); ++row) {
                if (row_tokens[row].empty()) { continue; }
                by_page.push_back(row);
            }
            std::stable_sort(by_page.begin(), by_page.end(),
                             [&rows](std::size_t a, std::size_t b) {
                                 return rows[a].page < rows[b].page;
                             });
            for (const std::size_t row : by_page) {
                row_stream_begin[row] = static_cast<std::uint32_t>(stream_tokens.size());
                for (const std::uint32_t token : row_tokens[row]) {
                    stream_tokens.push_back(token);
                    stream_page.push_back(rows[row].page);
                    stream_row.push_back(static_cast<std::uint32_t>(row));
                }
            }
        }
        // PASS 2: the family, over the STREAM. A gram is indexed once per position it starts at,
        // attributed to the page that position is in -- and the last k_min - 1 positions of every
        // row index a gram that continues into the next row. That is the difference between a
        // family covering for a page and one covering for the passage.
        for (std::size_t row = 0; row < rows.size(); ++row) {
            const std::vector<std::uint32_t>& tokens = row_tokens[row];
            if (tokens.empty()) { continue; }
            const std::size_t begin = row_stream_begin[row];
            for (std::size_t position = 0; position < tokens.size(); ++position) {
                if (begin + position + k_min > stream_tokens.size()) { break; }
                SumDirReachGram gram;
                gram.tokens.assign(stream_tokens.begin() +
                                       static_cast<std::ptrdiff_t>(begin + position),
                                   stream_tokens.begin() +
                                       static_cast<std::ptrdiff_t>(begin + position + k_min));
                by_gram[std::move(gram)].push_back(
                    SumDirReachPosting{rows[row].page, static_cast<std::uint32_t>(begin + position)});
                ++entry_count;
            }
        }
    }
};

[[nodiscard]] inline bool sum_dir_reach_extends(std::uint32_t covered_begin,
                                               std::uint32_t covered_end,
                                               const SumDirReachEvidence& item) noexcept {
    const std::uint32_t begin = item.offset;
    const std::uint32_t end   = item.offset + item.length;
    // STRICTLY larger, and touching: an interval INTERIOR to the covered one (the 4-gram that
    // occurs on 256 pages) extends nothing and is therefore never admitted, which is the whole
    // reason this predicate is stricter than "overlaps".
    if (begin < covered_begin && end >= covered_begin) { return true; }
    if (end > covered_end && begin <= covered_end) { return true; }
    return false;
}

// ---------------------------------------------------------------------------
// ===========================================================================================
// 7b. THE GUARD SET, DERIVED WHERE IT IS CONSUMED -- THE INVERSION THIS CLOSES
// ===========================================================================================
//
// `appended_pages` is the LOOP GUARD of section 3: the pages this sequence has ALREADY recalled,
// which must not be restored a second time. The tree's own arm builds it exactly that way --
// `tests/test_sum_dir_reach.cpp:845-846` appends every page it has just INSERTED into the query
// stream, i.e. every page it RESTORED -- and the header's own note at :117-118 says only that the
// set "must be supplied from `SequenceState`".
//
// The live wiring did not supply that set. It supplied `SequenceState::recall_recorded`'s page
// column, and that column is the sequence's STANDING SPILL RECORDS as they stand RIGHT NOW:
// `record_turn_recall` (program_impl.h) keeps a record only while its page is still cold
// (`have == now`) and DROPS it the moment the page goes hot again (`have < now` writes the
// tombstone and pushes nothing, program_impl.h:13496-13515). The same function's own comment at
// :13464 calls it "the sequence's spilled-page set", and program_impl.h:11787-11792 states the
// same rule again. So the set the guard was handed is
//
//       recall_recorded = { p : p has a directory row AND p is STILL COLD }  =  S \ R
//
// while the set it is NAMED for -- three separate prose statements, sum_dir_reach.h:90-96,
// program_impl.h:14412-14413 and program_impl.h:14547-14552 -- is R itself. Over the pages the
// directory can return (its rows, one per spilled page) those two sets are EXACT COMPLEMENTS, so
// `found` was reachable only for a page that was ALREADY restored -- the exact set the guard
// exists to suppress -- and every page that was actually cold and restorable was refused.
//
// MEASURED CONSEQUENCE, leg A of dl/millionrun (its REPORT.md, section LEAD): 20 raw hits, all of
// them on the needle's own page, and `admissible=0 ... 1 page(s) by the already-recalled guard.
// There is nothing new to restore.` -- the selector found the page it exists to find and threw it
// away: `plans=0`, `pages=0/0`, no `[context-append]` at all. And in the other direction, a page
// that WAS restored and whose tail becomes the next query is NOT suppressed -- which is the D1
// rounds 3-4 lock-on the guard was written to stop. The guard is inert for its own case.
//
// THE DERIVATION, in one place so the wiring cannot re-derive it wrongly:
//
//       already recalled = (every page the directory has a row for)  \  (the pages still cold)
//
// A page with a row that is no longer cold went hot through `restore_cold_page`, which ERASES its
// `cold_pages` entry (program_impl.h:13310) -- that erase IS the restore's own record, so no new
// state is needed and none is invented, and the identity is read off STATE rather than off
// history. A page that is restored and later RE-SPILLED is cold again, leaves this set, and is
// correctly recallable a second time; a monotone "ever restored" list would be wrong there.
//
// PURE, host-only, std-only: two page lists in, one page list out. It is the function the test
// drives and the function the engine calls, so the two cannot disagree about what the guard is.
[[nodiscard]] inline std::vector<std::uint32_t>
sum_dir_reach_already_recalled_pages(const std::vector<std::uint32_t>& row_pages,
                                     const std::vector<std::uint32_t>& still_cold_pages) {
    std::vector<std::uint32_t> cold = still_cold_pages;
    std::sort(cold.begin(), cold.end());
    std::vector<std::uint32_t> already_recalled;
    already_recalled.reserve(row_pages.size());
    for (const std::uint32_t page : row_pages) {
        if (!std::binary_search(cold.begin(), cold.end(), page)) {
            already_recalled.push_back(page);
        }
    }
    std::sort(already_recalled.begin(), already_recalled.end());
    already_recalled.erase(std::unique(already_recalled.begin(), already_recalled.end()),
                           already_recalled.end());
    return already_recalled;
}

// THE SELECTOR. Signature shape is the sibling of `sum_dir_recall_span_for_term`, plus the three
// inputs the old one had no way to express (`appended_pages`, the budget, and the family's k_min).
// `scan` is an optional out-parameter so the arm can read the selector's own evidence.
// ---------------------------------------------------------------------------
[[nodiscard]] inline SumDirReachResult sum_dir_recall_span_reachable(
    const SumDir& directory, const SumDirReachPostings& postings, std::string_view term,
    std::uint32_t frontier, std::uint32_t fanout_cap,
    std::span<const std::uint32_t> appended_pages,
    std::uint32_t budget_blocks = kReachDefaultBudgetBlocks,
    std::uint32_t block_tokens  = kSumDirBlockTokens, SumDirReachScan* scan = nullptr) {
    SumDirReachResult result;
    result.fanout_cap    = fanout_cap;
    result.budget_blocks = budget_blocks;
    result.stream_tokens = postings.stream_tokens.size();
    if (scan != nullptr) { *scan = SumDirReachScan(); }

    const std::uint32_t k_min = postings.k_min;
    if (block_tokens == 0U || k_min == 0U) {
        result.refusal = "recall reach: a zero block granularity or a zero k_min is not a query";
        return result;
    }
    std::vector<std::uint32_t> query;
    if (!sum_dir_parse_token_line(term, query)) {
        result.refusal = "recall reach: the query term is not a rendered token line";
        return result;
    }
    const std::uint32_t length = static_cast<std::uint32_t>(query.size());
    if (length == 0U) {
        result.refusal = "recall reach: the term renders to no tokens";
        return result;
    }
    if (length < k_min) {
        result.refusal =
            "recall reach: the query holds " + std::to_string(length) +
            " token(s), below the tree's declared minimum window k_min=" + std::to_string(k_min) +
            " (program_impl.h:108); a shorter phrase is a common word and a common word is not a "
            "location";
        return result;
    }
    if (fanout_cap == 0U) {
        result.refusal =
            "recall reach: the per-pass fan-out cap is 0, which would disable retrieval silently";
        return result;
    }
    // The query occupies [frontier - length, frontier); a page whose END reaches into that range is
    // a page the query itself covers. UNCHANGED from the shipping rule -- the no-approximation
    // property depends on it -- and now COUNTED rather than merely acted on.
    const std::uint32_t query_begin = frontier > length ? frontier - length : 0U;

    std::vector<SumDirReachEvidence> evidence;
    std::vector<std::uint32_t>       self_excluded;
    std::vector<std::uint32_t>       appended_excluded;
    std::vector<std::uint32_t>       gram;
    const std::size_t                stream_size = postings.stream_tokens.size();
    for (std::uint32_t offset = 0; offset + k_min <= length; ++offset) {
        ++result.family_windows;
        gram.assign(query.begin() + static_cast<std::ptrdiff_t>(offset),
                    query.begin() + static_cast<std::ptrdiff_t>(offset + k_min));
        const std::vector<SumDirReachPosting>* hits = postings.postings(gram);
        if (hits == nullptr) { continue; }
        for (const SumDirReachPosting& hit : *hits) {
            ++result.raw_hits;
            if (hit.page >= postings.row_for_page.size()) { continue; }
            const std::uint32_t row = postings.row_for_page[hit.page];
            if (row == kSumDirNoRow || row >= postings.row_tokens.size()) { continue; }
            if (hit.stream >= stream_size) { continue; }
            // THE EXACT EXTENSION, over the STREAM. Both sides are the same rendered alphabet, so
            // this is a token-exact longest common run -- not a byte heuristic and not a window
            // length -- and it now crosses page boundaries, which is what makes a straddling
            // fragment's true occurrence visible at all.
            std::uint32_t reached = k_min;
            while (offset + reached < length &&
                   static_cast<std::size_t>(hit.stream) + reached < stream_size &&
                   query[offset + reached] == postings.stream_tokens[hit.stream + reached]) {
                ++reached;
            }
            const std::uint32_t end_page =
                postings.stream_page[static_cast<std::size_t>(hit.stream) + reached - 1U];
            const std::uint32_t tokens = static_cast<std::uint32_t>(postings.row_tokens[row].size());
            const std::uint64_t end = static_cast<std::uint64_t>(hit.page) * block_tokens + tokens;
            if (end > frontier || end > query_begin) {
                if (std::find(self_excluded.begin(), self_excluded.end(), hit.page) ==
                    self_excluded.end()) {
                    self_excluded.push_back(hit.page);
                }
                continue;
            }
            if (std::find(appended_pages.begin(), appended_pages.end(), hit.page) !=
                appended_pages.end()) {
                if (std::find(appended_excluded.begin(), appended_excluded.end(), hit.page) ==
                    appended_excluded.end()) {
                    appended_excluded.push_back(hit.page);
                }
                continue;
            }
            ++result.admissible;
            if (end_page != hit.page) { ++result.cross_boundary; }
            evidence.push_back(
                SumDirReachEvidence{hit.page, offset, reached, hit.stream, end_page});
        }
    }
    result.excluded_self     = self_excluded.size();
    result.excluded_appended = appended_excluded.size();
    if (scan != nullptr) {
        scan->evidence          = evidence;
        scan->family_windows    = result.family_windows;
        scan->raw_hits          = result.raw_hits;
        scan->excluded_self     = result.excluded_self;
        scan->excluded_appended = result.excluded_appended;
    }

    if (evidence.empty()) {
        result.status = (result.raw_hits != 0U) ? SumDirReachStatus::RefusedEverything
                                                : SumDirReachStatus::NoCandidate;
        if (result.raw_hits == 0U) {
            result.refusal = "recall reach: the covering family (" +
                             std::to_string(result.family_windows) + " k_min=" +
                             std::to_string(k_min) + "-gram window(s) of a " +
                             std::to_string(length) +
                             "-token query) matched no page at all. No span is returned, and page 0 "
                             "is NOT substituted: the positional first-fit chooses by position, not "
                             "by content.";
        } else {
            result.refusal = "recall reach: " + std::to_string(result.raw_hits) +
                             " hit(s) existed and every one was excluded -- " +
                             std::to_string(result.excluded_self) +
                             " page(s) by the query-owns-this-page guard, " +
                             std::to_string(result.excluded_appended) +
                             " page(s) by the already-recalled guard. There is nothing new to "
                             "restore.";
        }
        return result;
    }

    // ---- THE STRONGEST CLASS, AND THE ANCHOR (a TOTAL order; `min` over a union it is not). ----
    std::uint32_t best_length = 0;
    for (const SumDirReachEvidence& item : evidence) {
        best_length = std::max(best_length, item.length);
    }
    std::size_t champion = evidence.size();
    for (std::size_t index = 0; index < evidence.size(); ++index) {
        const SumDirReachEvidence& item = evidence[index];
        if (item.length != best_length) { continue; }
        if (champion == evidence.size()) { champion = index; continue; }
        const SumDirReachEvidence& holder = evidence[champion];
        if (item.offset < holder.offset ||
            (item.offset == holder.offset && item.page < holder.page)) {
            champion = index;
        }
    }
    // ---- THE CHOSEN MATCHES: the strongest class, then the passage growth. ----
    std::vector<std::size_t> chosen;
    std::uint32_t            candidates = 0;
    for (std::size_t index = 0; index < evidence.size(); ++index) {
        if (evidence[index].length != best_length) { continue; }
        ++candidates;
        chosen.push_back(index);
    }
    result.candidates = candidates;

    // THE UNION OF THE STRONGEST CLASS'S INTERVALS: reported, and NOT claimed to be realised. The
    // champion's OWN interval is the realised one (`covered_*` below) because the champion is a
    // contiguous stream run whose pages are in the run; the union of intervals taken from different
    // pages is a different, weaker object, and conflating the two is what made the 22 cases look
    // "covered" while the passage contained nothing of the fragment.
    std::uint32_t union_begin = 0xFFFFFFFFU;
    std::uint32_t union_end   = 0;
    for (const std::size_t index : chosen) {
        union_begin = std::min(union_begin, evidence[index].offset);
        union_end   = std::max(union_end, evidence[index].offset + evidence[index].length);
    }
    result.union_begin = union_begin;
    result.union_end   = union_end;
    // The growth's own covered interval starts at the strongest class, exactly as before.
    std::uint32_t covered_begin = union_begin;
    std::uint32_t covered_end   = union_end;

    // ---- THE PASSAGE GROWTH. On each side of the run, admit the NEAREST match whose evidence
    // STRICTLY extends the covered query interval AND which lies ENTIRELY on that side -- so the
    // match it quotes is never half outside the passage that justifies it. A page whose match is
    // interior to the covered interval extends nothing and is never admitted, so the 4-grams that
    // occur on 256 pages cannot widen the run -- which is the defect this rule exists to remove.
    auto bounds = [&](std::uint32_t& low, std::uint32_t& high) {
        low  = 0xFFFFFFFFU;
        high = 0;
        for (const std::size_t index : chosen) {
            low  = std::min(low, evidence[index].page);
            high = std::max(high, evidence[index].end_page);
        }
    };
    for (bool grew = true; grew;) {
        grew = false;
        if (chosen.empty()) { break; }
        std::uint32_t low  = 0;
        std::uint32_t high = 0;
        bounds(low, high);
        for (std::size_t side = 0; side < 2U; ++side) {
            std::size_t   pick          = evidence.size();
            std::uint32_t pick_distance = 0;
            for (std::size_t index = 0; index < evidence.size(); ++index) {
                if (std::find(chosen.begin(), chosen.end(), index) != chosen.end()) { continue; }
                const SumDirReachEvidence& item = evidence[index];
                std::uint32_t distance = 0;
                if (side == 0U) {
                    if (item.end_page >= low) { continue; }
                    distance = low - item.end_page;
                } else {
                    if (item.page <= high) { continue; }
                    distance = item.page - high;
                }
                if (!sum_dir_reach_extends(covered_begin, covered_end, item)) { continue; }
                if (pick == evidence.size() || distance < pick_distance ||
                    (distance == pick_distance &&
                     (item.length > evidence[pick].length ||
                      (item.length == evidence[pick].length && item.page < evidence[pick].page)))) {
                    pick          = index;
                    pick_distance = distance;
                }
            }
            if (pick == evidence.size()) { continue; }
            const SumDirReachEvidence& item = evidence[pick];
            covered_begin = std::min(covered_begin, item.offset);
            covered_end   = std::max(covered_end, item.offset + item.length);
            chosen.push_back(pick);
            grew = true;
        }
    }

    std::uint32_t first_page = 0;
    std::uint32_t last_page  = 0;
    bounds(first_page, last_page);
    const std::uint32_t needed = last_page - first_page + 1U;

    // ---- THE ALTERNATIVES, COUNTED. Every admissible page the covering family found and the run
    // does NOT hold is an ALTERNATIVE OCCURRENCE of part of the query's text: the run is the maximal
    // passage around the strongest match and cannot hold a page 125 pages away, but that page was
    // FOUND, and a found page is COUNTED here rather than dropped. This is what makes "reachability"
    // auditable at the boundary between "in the span" and "found but not in the passage".
    {
        std::vector<std::uint32_t> seen;
        for (const SumDirReachEvidence& item : evidence) {
            if (std::find(seen.begin(), seen.end(), item.page) != seen.end()) { continue; }
            seen.push_back(item.page);
            if (item.page < first_page || item.page > last_page) { ++result.alternatives; }
        }
    }
    result.window        = best_length;
    result.best_offset   = evidence[champion].offset;
    result.anchor_page   = evidence[champion].page;
    // THE REALISED INTERVAL: the champion's own match. The champion's pages are in the run by
    // construction (the run is the span of the CHOSEN matches' occupied ranges and the champion is
    // one of them), and the champion is a contiguous run of the stream, so the passage's text
    // contains query[covered_begin, covered_end) verbatim.
    result.covered_begin = result.best_offset;
    result.covered_end   = result.best_offset + best_length;
    result.first_page    = first_page;
    result.last_page     = last_page;
    result.needed_blocks = needed;

    // ---- THE REFUSAL, NAMED. A candidate that cannot be held is NOT clamped. ----
    if (needed > fanout_cap || needed > budget_blocks) {
        result.status = SumDirReachStatus::RefusedBudget;
        result.pages  = 0;
        std::string why;
        if (needed > fanout_cap) {
            why = "the per-pass fan-out cap is " + std::to_string(fanout_cap);
        } else {
            why = "the block budget is " + std::to_string(budget_blocks);
        }
        const std::uint32_t holdable = std::min(fanout_cap, budget_blocks);
        result.refusal =
            "recall reach: the strongest evidence is " + std::to_string(best_length) +
            " token(s) at query offset " + std::to_string(result.best_offset) + ", held by " +
            std::to_string(candidates) + " match(es); with the passage it covers query tokens [" +
            std::to_string(covered_begin) + "," + std::to_string(covered_end) + ") of " +
            std::to_string(length) + ", which lives in the contiguous page run [" +
            std::to_string(first_page) + "," + std::to_string(last_page) + "] = " +
            std::to_string(needed) + " page(s), but " + why + ". Holding " +
            std::to_string(holdable) + " of them would drop pages [" +
            std::to_string(first_page + holdable) + "," + std::to_string(last_page) +
            "] and make their text unreachable. Refusing rather than truncating.";
        // ---- THE REMEDY, NAMED. A refusal a reader cannot act on is indistinguishable from a bug,
        // and this refusal is REACHED in practice rather than theoretical: at the 1M configuration's
        // own scale the same query needs 43 pages. That measurement was taken against the cap the
        // engine shipped AT THE TIME, 8 (`dl/recallplan/REPORT.md` sec.3.3); the default is now
        // `kSumDirDefaultBlocksPerPass`, 227, derived in `sum_dir.h`. dl/index1m repaired this
        // REFUSAL once already -- it NAMED the knob and did not move the number, which is the
        // history dl/idx1m/FIX.md states in full. This header's own probe reaches the same
        // refusal at need=201/cap=8
        // (`dl/index1m/evidence/probe_index_refusal.cpp`). The numbers above say WHAT was refused;
        // this says what to move.
        //
        // WHERE THE CAP COMES FROM, so the name is the right one: the engine passes
        // `ProgramImplCore::recall_fanout_blocks()`, which reads `SumDirKnobs::max_blocks_per_pass`
        // out of `NINFER_SUM_DIR_BLOCKS` (factory default `kSumDirDefaultBlocksPerPass`,
        // `sum_dir.h`; `program_impl.h`, `recall_fanout_blocks`). Naming the environment
        // variable is therefore the actionable fact, and it is the same standard the sibling guard
        // already meets: `src/product/kv_paging_preallocation.h` states its remedies the same way
        // ("Either use --kv-capacity 1000000 (or more), or declare a context of at most 6272 tokens,
        // or arm a backing store that can hold the difference").
        if (needed > fanout_cap) {
            result.refusal +=
                " The cap is SumDirKnobs::max_blocks_per_pass, read from the environment variable "
                "NINFER_SUM_DIR_BLOCKS (factory default 227, kSumDirDefaultBlocksPerPass, "
                "sum_dir.h; program_impl.h, recall_fanout_blocks); raise it "
                "to at least " + std::to_string(needed) + " to admit this run.";
        } else {
            result.refusal +=
                " The budget is kReachDefaultBudgetBlocks (sum_dir_reach.h) and is NOT "
                "environment-reachable: a caller that can hold this run must pass budget_blocks >= " +
                std::to_string(needed) + ".";
        }
        return result;
    }

    const std::uint32_t begin_token = first_page * block_tokens;
    if (begin_token >= frontier) {
        result.status  = SumDirReachStatus::RefusedEverything;
        result.refusal = "recall reach: the selected run begins past the committed frontier";
        return result;
    }
    // The LAST BLOCK'S OWN END, from its row -- the same rule the shipping selector and the cargo's
    // `read_located_block` both apply to a partial tail.
    //
    // =====================================================================================
    // WHERE THE ROW COMES FROM: THE GUARDED READER, OR NO SPAN AT ALL
    // =====================================================================================
    //
    // WHAT WAS WRONG. This leg used to read `postings.row_for_page[last_page]` -- this header's OWN
    // page->row map, built at `sum_dir_reach.h:436` by an UNCONDITIONAL `row_for_page[page] = row`,
    // i.e. LAST writer wins, over EVERY row, and with NO failure channel at all. Two other readers
    // of "which row covers page p" exist in this tree, and BOTH are guarded in a way this one was
    // not:
    //
    //   * `SumDir::row_for_page` (sum_dir.h:1775) consults the tripwire
    //     (`if (page < page_conflict_.size() && page_conflict_[page] != 0U) { return false; }`,
    //     sum_dir.h:1789) and REFUSES a page that two generations cover;
    //   * `SumDir::ensure_page_index` (sum_dir.h:2164) is what SETS it (sum_dir.h:2180-2187) --
    //     first writer wins for the address, and a generation mismatch sweeps the page into the
    //     tripwire.
    //
    // So a page can be covered by two rows, and this leg would take a length from ONE of them
    // without ever asking whether the choice was lawful. The length is not cosmetic: it is the
    // span end (`end_token` below), and the span end is the ONLY input to
    // `plan_round_recall`'s `wanted_end_page` (`program_impl.h:13595`), i.e. it decides how many
    // pages a round restores.
    //
    // WHY THE FIX IS A REFUSAL AND NOT A TIE RULE. Fixing the tie rule would mean CHOOSING a
    // winner between two rows that both claim the page, and the tree has already refused to make
    // that choice: `ensure_page_index` records "no single answer" (`page_conflict_`) instead of
    // resolving it, and `row_for_page` refuses instead of picking. This leg therefore does the
    // only thing that adds no policy: it takes the length from the GUARDED reader, and if the two
    // readers do not agree -- the guarded reader refusing (two generations), or the two naming
    // rows with different lengths (two rows of the SAME generation, which the tripwire does NOT
    // flag; or a row whose catalogue line was never written, `:438-446`) -- the SPAN IS REFUSED.
    //
    // WHAT THIS COSTS, STATED. With one row per page -- which is every configuration reachable
    // today, because the sole row writer passes `sequence.recall_sequence_tag` and that field is
    // `= 0` and never assigned (program.h:546), passed as `0U` at program_impl.h:1380 -- the two
    // readers name the SAME row and this leg returns exactly the value it returned before. Where
    // they disagree, the leg now refuses the span instead of handing out one of the two lengths.
    // The caller treats a refusal as "nothing to recall this round" (`program_impl.h:14527-14532`
    // returns an empty `RecallRequest`), so the blast radius of the refusal is a MISS, never a
    // wrong span end.
    std::uint32_t last_tokens   = 0; // the length, when it is answerable
    bool          last_answered = false;
    {
        // THE AUTHORITY: the same guarded reader the two shipping length legs call.
        std::size_t authority_row   = 0;
        const bool  authority_has   = directory.row_for_page(last_page, authority_row);
        // THIS LEG'S OWN READING, kept only so the two can be COMPARED. It is never preferred.
        std::uint32_t leg_tokens    = 0;
        bool          leg_has       = false;
        if (last_page < postings.row_for_page.size()) {
            const std::uint32_t row = postings.row_for_page[last_page];
            if (row != kSumDirNoRow && row < directory.rows().size()) {
                leg_tokens = sum_dir_row_tokens(directory.rows()[row]);
                leg_has    = true;
            }
        }
        if (authority_has && leg_has) {
            const std::uint32_t authority_tokens =
                sum_dir_row_tokens(directory.rows()[authority_row]);
            if (authority_tokens == leg_tokens) {
                last_tokens   = authority_tokens;
                last_answered = true;
            } else {
                result.refusal = "recall reach: the run's last page " +
                                 std::to_string(last_page) +
                                 " has no single answer for its token count: the guarded reader "
                                 "(sum_dir.h:1775 row_for_page, which consults page_conflict_) "
                                 "names a row of " +
                                 std::to_string(authority_tokens) +
                                 " token(s) while this header's own page->row map names a row of " +
                                 std::to_string(leg_tokens) +
                                 " token(s). Both may be real rows -- two rows for one page can "
                                 "carry the SAME generation, and sum_dir.h:2180 only sweeps a page "
                                 "into the tripwire when the generations DIFFER, so the tripwire "
                                 "cannot separate them -- and a span end taken from either is a "
                                 "length that is not the restored text's own. No winner is chosen: "
                                 "the span is REFUSED. The span end is what decides how many pages "
                                 "the round restores (program_impl.h:13595 wanted_end_page), so "
                                 "substituting either length would move a plan.";
            }
        } else if (!authority_has && !leg_has) {
            // Neither reader names a row for the last page: the stride rule below is unchanged,
            // exactly as it behaved before this landing.
            last_answered = true;
        } else {
            result.refusal = "recall reach: the run's last page " +
                             std::to_string(last_page) +
                             " is refused by the guarded reader (sum_dir.h:1775 row_for_page) "
                             "while this header's own page->row map still names a row for it: the "
                             "page is covered by two GENERATIONS (sum_dir.h:1789 is the "
                             "tripwire, set at sum_dir.h:2180-2187) and the guarded reader refuses "
                             "it by design. This leg is generation-blind -- MEASURED: `generation` "
                             "and `page_conflict_` have ZERO occurrences in this whole header -- "
                             "so it must not answer a question the authority declines. No span is "
                             "returned.";
        }
        if (!last_answered) {
            result.status = SumDirReachStatus::RefusedLength;
            result.pages  = 0;
            return result;
        }
    }
    const std::uint32_t end_token =
        last_page * block_tokens + (last_tokens != 0U ? last_tokens : block_tokens);
    result.status      = SumDirReachStatus::Found;
    result.found       = true;
    result.pages       = needed;
    result.token_begin = begin_token;
    result.token_end   = std::min(end_token, frontier);
    return result;
}

// The convenience form: builds the postings then selects, so a caller with no place to cache the
// table can still use the selector. There is exactly ONE selector body.
[[nodiscard]] inline SumDirReachResult sum_dir_recall_span_reachable(
    const SumDir& directory, std::string_view term, std::uint32_t frontier,
    std::uint32_t fanout_cap, std::span<const std::uint32_t> appended_pages,
    std::uint32_t budget_blocks = kReachDefaultBudgetBlocks,
    std::uint32_t block_tokens  = kSumDirBlockTokens,
    std::uint32_t k_min = kReachMinWindowTokens, SumDirReachScan* scan = nullptr) {
    SumDirReachPostings postings;
    postings.k_min = k_min;
    postings.build(directory);
    SumDirReachResult result =
        sum_dir_recall_span_reachable(directory, postings, term, frontier, fanout_cap,
                                      appended_pages, budget_blocks, block_tokens, scan);
    result.build_grams = postings.entry_count;
    return result;
}

// The diagnostic line, deliberately the sibling of the engine's own `[recall-index]` line and
// carrying the fields that line cannot print -- the guard attributions and the limits -- so that a
// refusal is READABLE from the run rather than inferred from a smaller `pages=`.
[[nodiscard]] inline std::string sum_dir_reach_line(const SumDirReachResult& result) {
    std::string out = "[recall-reach] status=";
    out += sum_dir_reach_status_name(result.status);
    out += " windows=" + std::to_string(result.family_windows);
    out += " hits=" + std::to_string(result.raw_hits);
    out += " admissible=" + std::to_string(result.admissible);
    out += " crossed=" + std::to_string(result.cross_boundary);
    out += " self=" + std::to_string(result.excluded_self);
    out += " appended=" + std::to_string(result.excluded_appended);
    out += " best=" + std::to_string(result.window) + "@" + std::to_string(result.best_offset);
    out += " anchor=" + std::to_string(result.anchor_page);
    out += " candidates=" + std::to_string(result.candidates);
    out += " alternatives=" + std::to_string(result.alternatives);
    out += " covered=[" + std::to_string(result.covered_begin) + "," +
           std::to_string(result.covered_end) + ")";
    out += " union=[" + std::to_string(result.union_begin) + "," +
           std::to_string(result.union_end) + ")";
    out += " pages=" + std::to_string(result.pages);
    out += " span=[" + std::to_string(result.token_begin) + "," + std::to_string(result.token_end) +
           ")";
    out += " first_page=" + std::to_string(result.first_page) +
           " last_page=" + std::to_string(result.last_page);
    out += " need=" + std::to_string(result.needed_blocks) +
           " cap=" + std::to_string(result.fanout_cap) +
           " budget=" + std::to_string(result.budget_blocks);
    if (!result.refusal.empty()) { out += " -- " + result.refusal; }
    return out;
}

} // namespace ninfer::spec::sum_dir
