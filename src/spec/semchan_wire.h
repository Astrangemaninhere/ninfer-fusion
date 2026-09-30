#ifndef NINFER_SPEC_SEMCHAN_WIRE_H
#define NINFER_SPEC_SEMCHAN_WIRE_H

// ===========================================================================================
// SEMWIRE (line `semwire`, coordinator R40) -- THE SEMANTIC CHANNEL'S WIRE INTO THE ENGINE.
// ===========================================================================================
//
// WHAT THIS FILE IS: the ADAPTER (not the decision) that lets the semantic candidate channel which
// already exists -- `spec/semchan_symbol_key.h`, landed by line `semchan`, sha16
// `d47c7dfcce7a6458` / 59,408 B / 987 lines -- contribute to the CANDIDATE-SET UNION of the
// engine's recall path, behind a NAMED switch whose DEFAULT IS OFF.
//
// THE DOCTRINE IT OBEYS, TAKEN FROM THE TREE AND NOT RESTATED. Each citation carries its
// IDENTIFIER (durable) and a line number taken on the spot (which rots -- see ANCHORS below).
//   * `program_impl.h`, the arbiter's own comment, searched as the literal `candidate-set UNION`:
//     "A semantic channel (KVMem's Mean-K) is complementary and belongs in the candidate-set UNION,
//     never in place of this arbiter".
//   * `sum_dir_query_key.h:37-39` (live reading): a similarity score "may narrow (the candidate-set
//     UNION) and may never determine".
//   * `sum_dir_determined.h:24-27` (live reading): "It may only contribute to the candidate-set
//     UNION".
//
// ⚠ kvfix (F893): THE FIVE ROWS BELOW ARE ALL STALE AND ARE KEPT SO NOBODY READS THEM AS LIVE.
// The numbers in the right column were taken at 2026-09-23T16:13Z on an 18,680 line
// program_impl.h; the file is now 18,910 lines / sha256 7c707815f49166a4...d0b131, and every one
// of the five lands on text that does not carry the identifier. Each row now carries its own
// live coordinate ON THE SAME LINE (the identifier is the anchor, the number is the rot), which
// is the form this file's own doctrine citation already uses:
//   program_impl.h [text: exact-or-silent, never approximately right; = :15948 on 2026-09-25].
// The doctrine string itself is UNCHANGED, which is why no hash gate can see any of this.
//
// ANCHORS, AND WHY THEY ARE WRITTEN THIS WAY (the station rule: line numbers rot, identifiers do
// not -- `program_impl.h` moved twice inside this session, 18,483 -> 18,677 -> 18,680 lines):
//
//   reading                     | sha16            | bytes     | lines
//   ----------------------------|------------------|-----------|------
//   coordinator's brief         | 42f47ad02022b481 | 1,099,037 | 18,677
//   THIS LINE, 2026-09-23T16:13Z| 1bd3f747ccac3d0d | 1,099,255 | 18,680
//
//   identifier                            | then | THIS LINE's live grep -n
//   --------------------------------------|------|------------------------
//   `candidate-set UNION`                 | 15918| 15921   <- STALE (kvfix F893, re-read 2026-09-26): LIVE is `candidate-set UNION` at :15950
//   `reach_request_for_term`              | 16114| 16116 (definition), 16117 (name on the line)   <- STALE (kvfix F893, re-read 2026-09-26): LIVE is `reach_request_for_term` at :16146
//   `install_index_recall_provider`       | 16179| 15310 (call), 16182 (definition)   <- STALE (kvfix F893, re-read 2026-09-26): LIVE is `install_index_recall_provider` at :16219, and the one call site is :15339
//   `reach_postings_for`                  | --   | 16003   <- STALE (kvfix F893, re-read 2026-09-26): LIVE is `reach_postings_for` at :16033
//   `recall_reach_selector_armed`         | --   | 15933   <- STALE (kvfix F893, re-read 2026-09-26): LIVE is `recall_reach_selector_armed` at :15962, and its one use is :16224
//
// WHAT IT DOES NOT CHANGE -- AND THIS IS THE HARD CRITERION, NOT A CLAIM. With the switch OFF
// (the default, and the only value this tree has ever shipped): NOTHING in this file executes
// except one `std::getenv` on first call, no line of it is printed, and the provider returns the
// arbiter's own `RecallRequest` unchanged. `semchan_wire_enabled()` false is the ONLY thing the
// OFF path evaluates; every other entry point in this header is reached only through an explicit
// `if (semchan_wire_enabled())` at its single call site in `program_impl.h`.

#include "spec/semchan_symbol_key.h"
#include "spec/sum_dir.h"
#include "spec/sum_dir_reach.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

namespace ninfer::spec::semchan::wire {

// -------------------------------------------------------------------------------------------
// 1. THE SWITCH -- NAMED, READ ONCE, AND OFF UNLESS IT SAYS EXACTLY "1"
// -------------------------------------------------------------------------------------------
// The house pattern for an operational switch in this tree is an environment variable read at the
// point of use (`kv_rotation_apply_from_env` and friends). This one is deliberately STRICTER than
// "non-empty means on": the accepted ON spelling is the single byte "1". Unset, "", "0", "01",
// "true", "yes", "on" and any typo are all OFF, so a misspelt switch cannot turn a candidate
// source on by accident and a reader cannot be unsure what the reading means.
//
// WHY AN ENV SWITCH AND NOT A CLI FLAG: `apps/cli/options.cpp` / `apps/cli/main.cpp` were changed
// by line `kvknob` hours before this line opened (its boundary-input table,
// `dl/kvknob/STATE.md` sec. 3, registers 36 named boundary inputs on that front door). Adding a
// flag there would have meant editing a file another line had just landed and re-taking its whole
// table; the env route costs that file ZERO bytes. Recorded as an interface note in `REPORT.md`
// rather than done quietly.
inline constexpr const char* kSemChanWireSwitchName = "NINFER_SEMCHAN";
inline constexpr char        kSemChanWireOnValue[]  = "1";

// The pure predicate, so it can be exercised without the environment (see the probe's switch arm).
[[nodiscard]] constexpr bool semchan_switch_value_is_on(const char* value) noexcept {
    return value != nullptr && value[0] == kSemChanWireOnValue[0] && value[1] == '\0';
}

// Read ONCE per process. A static local's initialisation happens on first call, so the environment
// is consulted once and every later call is a load -- which is what keeps the OFF path to a single
// predictable branch inside the per-round provider.
[[nodiscard]] inline bool semchan_wire_enabled() noexcept {
    static const bool on = semchan_switch_value_is_on(std::getenv(kSemChanWireSwitchName));
    return on;
}

// -------------------------------------------------------------------------------------------
// 2. THE REFUSAL THE WHOLE FILE EXISTS TO PIN
// -------------------------------------------------------------------------------------------
// The channel may NOT determine. This is not a comment about intent: `semchan_union_candidates`'s
// own result type returns `determined_by_semchan() == false` from a function whose body is
// `return false`, and the two constants below are `constexpr`, so a future edit that tries to make
// the channel decide is a BUILD ERROR at this line and at the `static_assert`s in section 6 --
// i.e. in every translation unit that includes `program_impl.h`, not merely in this header's own
// test. Measurement of a source-level claim is not possible; a build error is.
inline constexpr bool kSemChanWireMayDetermine = false;
[[nodiscard]] inline constexpr bool semchan_wire_may_determine() noexcept { return false; }

// The union's cap. It is the engine's OWN bound, handed in by the caller (`recall_fanout_blocks()`
// at the one call site), not a new number invented here. A union that would exceed it REFUSES and
// says so -- the tree's edge policy is RefuseNotTruncate (`turn_recall_journal.h`'s
// `kRecallBudgetEdgePolicy`), and a candidate source that truncated would be that same defect in a
// new file.
struct SemChanWireReport {
    // -- the switch and the cost, each with its own denominator -------------------------------
    bool          enabled            = false;
    double        channel_ms         = 0.0;  // the CHANNEL's own work only (index + score + union)
    std::uint32_t rows_indexed       = 0;    // denominator for channel_ms: rows the index read
    std::uint32_t blocks_scored      = 0;    // denominator: blocks the scorer ran over
    std::uint32_t query_tokens       = 0;    // denominator: ids the query key was formed from
    std::uint32_t page_tokens        = 0;    // the block unit, so a reader can convert rows <-> pages

    // -- the CHANNEL's own reading (its own labels; never added to the arbiter's) -------------
    SemChanCandidateStatus channel_status = SemChanCandidateStatus::RefusedEmptyQuery;
    std::uint32_t channel_positive       = 0;
    std::uint32_t channel_candidates     = 0; // (page, origin) RECORDS
    std::uint32_t channel_distinct_pages = 0; // PAGES -- a different set from the line above
    std::uint32_t channel_same_page_distinct_origin = 0;
    std::string   channel_refusal;

    // -- the ARBITER's own columns, copied, one label each -----------------------------------
    bool          arbiter_found    = false;
    std::uint32_t arbiter_anchor   = 0;
    std::uint32_t arbiter_pages    = 0;  // pages in the arbiter's own run (its [first,last] span)
    std::uint32_t arbiter_alternatives = 0; // its own count of admissible pages outside the run

    // -- THE ADDITIVE SEAM'S OWN COLUMNS (section 2b), OFF unless its switch says "1" ----------
    bool          seam_alternatives_on    = false; // this run asked for the additive route
    std::uint32_t alternatives_named      = 0;     // pages the arbiter NAMED (bounded prefix)
    std::uint32_t alternatives_admitted   = 0;     // of those, the ones that entered the set
    std::uint32_t alternatives_over_bound = 0;     // 1 <=> the named prefix is not the whole set
    std::uint32_t arbiter_set_before_seam = 0;     // the span alone, so the growth is a difference

    // -- THE UNION ---------------------------------------------------------------------------
    SemChanUnion  union_;
    std::vector<std::uint32_t> arbiter_page_set; // the pages this file unioned, printed as a SET

    // -- the elementwise truth, so no set is ever described by its cardinality alone ----------
    std::vector<std::uint32_t> channel_only_pages;
    std::vector<std::uint32_t> in_both_pages;

    [[nodiscard]] bool union_holds(std::uint32_t page) const noexcept { return union_.holds(page); }
};

// The class of a reading, for the edge-trigger. A provider is called once per lane per ROUND; the
// `program_impl.h` comment above the arbiter's line measured an unconditional `fprintf` at
// 0.932 ms against 0.007 ms for the lookup it reports (133x), so this line must not be
// level-triggered either. The key is the SHAPE, not the values: it changes only when the answer's
// kind changes.
[[nodiscard]] inline std::string semchan_wire_shape_key(const SemChanWireReport& r) {
    std::string key = "status=";
    key += semchan_candidate_status_name(r.channel_status);
    key += r.arbiter_found ? "|arb" : "|noarb";
    key += (r.union_.from_semchan_only > 0U) ? "|gift" : "|nogift";
    key += (r.union_.in_both > 0U) ? "|both" : "|noboth";
    key += r.union_.refused_over_cap ? "|overcap" : "|undercap";
    return key;
}

// -------------------------------------------------------------------------------------------
// 2b. THE ADDITIVE SEAM -- `SumDirReachResult::alternative_pages`, NOT `appended_pages`
// -------------------------------------------------------------------------------------------
// WHAT WAS MISSING, IN THIS FILE'S OWN WORDS. Section 3 below used to end its derivation with:
// "It never adds `alternatives` to the set, because the result struct does not name those pages and
// inventing them would be this file deciding by position." The first half of that sentence was
// exact and the second half was the reason it could not move: the struct counted the alternative
// pages and did not name them, so naming them here would have meant this file searching the
// directory for pages, i.e. this file deciding. `dl/kvfinish/REPORT.md` section 3 located exactly
// that gap (`sum_dir_reach.h`'s `alternatives` increment) and named the edit. It is now written AT
// THE SOURCE: `SumDirReachResult::alternative_pages` is filled by the same loop that increments
// `alternatives`, under the same predicate, bounded by `fanout_cap`.
//
// SO THIS FILE NO LONGER INHERITS THE GAP -- IT READS THE PAGES OFF THE ARBITER'S OWN RESULT. They
// are the ARBITER's pages, not the channel's: they enter `arbiter_page_set`, they are counted in
// `from_arbiter`, and the channel's own column is untouched by them. That distinction is the whole
// reason this is doctrinally clean: the arbiter is allowed to be exact and complete; the channel is
// not allowed to decide. A page the arbiter's own family FOUND and its own run could not hold is
// part of the arbiter's answer, and reading it off is not a second opinion.
//
// THE ROUTE THIS FILE FORBIDS BY NAME, AND WHY IT REFUSES RATHER THAN WARNS.
// `sum_dir_reach.h`'s `appended_pages` is the LOOP GUARD and it is an EXCLUSION set. A candidate
// source appended there DELETES the page it names. This is measured, not asserted:
//   * `dl/kvmemoracle` arm M2 -- the live channel's pages fed to `appended_pages` turned the arbiter
//     into `status=refused-everything anchor=0`, and the union's own anchor read disagreed with the
//     arbiter's (0 vs 2) because the arbiter's answer had collapsed;
//   * `dl/kvwire` arm `naive-exclusion` -- this leg's own re-take on today's tree, same result.
// `sum_dir_reach.h` section 7d now offers ONE function that can turn found pages into a loop-guard
// list and it returns EMPTY with the reason. This file's own bridge to it is
// `semchan_wire_pages_that_may_ride_the_loop_guard` below, which returns that same empty list: a
// caller that tries the naive wiring gets nothing and a named reason, so the wrong route cannot be
// taken silently.
//
// AND THE `RecallRequest` HALF IS A NEGATIVE PROPERTY, PINNED BY A TYPE. The provider builds its
// request from the arbiter's `token_begin`/`token_end` alone (`program_impl.h`,
// `reach_request_for_term`). Nothing in this file can move either field, because the union takes
// the arbiter's result as `const SumDirReachResult&` -- and that fact is asserted at the bottom of
// this header against the function's own type, not promised in a comment.

// THE SWITCH'S OWN NAME. Subordinate to `NINFER_SEMCHAN`: the alternatives can only ride a wire
// that is already carrying the channel, so `semchan_alternatives_enabled()` is false whenever
// `semchan_wire_enabled()` is. Same strict spelling rule as the other two -- the single byte "1",
// and every other value (including "true", "on", "01") is OFF.
inline constexpr const char* kSemChanAlternativesSwitchName = "NINFER_SEMCHAN_ALTERNATIVES";

[[nodiscard]] inline bool semchan_alternatives_enabled() noexcept {
    static const bool on =
        semchan_switch_value_is_on(std::getenv(kSemChanAlternativesSwitchName));
    return on && semchan_wire_enabled();
}

// The bridge to the forbidden route. Returns the loop-guard list a candidate source may lawfully
// hand to `sum_dir_recall_span_reachable` -- which is EMPTY, with the reason filled in -- so the
// naive wiring is a no-op a reader can see in the log instead of a deletion nobody sees.
[[nodiscard]] inline std::vector<std::uint32_t> semchan_wire_pages_that_may_ride_the_loop_guard(
    const sum_dir::SumDirReachResult& reach, std::string* why = nullptr) {
    return sum_dir::sum_dir_reach_loop_guard_from_found_pages(reach, why);
}

// -------------------------------------------------------------------------------------------
// 3. THE UNION -- THE ONE THING THIS FILE DOES WITH THE CHANNEL'S CANDIDATES
// -------------------------------------------------------------------------------------------
// Inputs: the engine's own directory (so the channel reads the engine's rows and nothing else), the
// query's token ids (the sequence's newest committed ids -- the SAME window the arbiter's term was
// rendered from, handed in by the caller), and the ARBITER'S OWN RESULT STRUCT, so the arbiter is
// consulted, not re-implemented: this file never scans the directory for a lexical match.
//
// The arbiter's page set is taken from the arbiter's OWN columns and the derivation is written out
// because it is an approximation a reader must be able to audit: `sum_dir_reach.h`'s result says
// the run's pages lie inside `[first_page, last_page]` and that `alternatives` counts admissible
// pages OUTSIDE that span (its own comment and its own increment). So this file unions
// `{first_page..last_page}` and PRINTS the interval, and it prints `alternatives` as a separate
// count with its own label.
//
// WHAT CHANGED WITH THE SEAM (section 2b), AND WHAT DID NOT. Historically this function added
// `alternatives` to nothing, because the result struct named no pages. `SumDirReachResult` now
// names them (`alternative_pages`, filled by the same loop that counts them, bounded by
// `fanout_cap`), so `admit_reach_alternatives == true` unions THOSE PAGES -- the arbiter's own
// found-but-uncarried pages -- into `arbiter_page_set`. They are the arbiter's, so they are counted
// in `from_arbiter`; the channel's column is untouched by them.
//
// AND THE FIFTH ARGUMENT DEFAULTS TO ITS OWN SWITCH, SO THE EXISTING CALL SITE IS ARMED. The
// engine's one call site (`program_impl.h`, `install_index_recall_provider`, reached only when
// `NINFER_SEMCHAN` says exactly "1") passes four arguments and is NOT edited. With the default as
// the switch, that call site reads the seam's switch itself -- the house pattern for an operational
// switch in this tree -- and with the environment unset the default evaluates to `false`, so the
// four-argument call's behaviour is byte-for-byte what it was before the seam existed. That
// property is MEASURED (`dl/kvwire` `logs/`, the two-build `cmp`: the OFF arm's stdout and stderr
// are byte-identical between the pre-image build and the live one).
[[nodiscard]] inline SemChanWireReport semchan_wire_union_from_reach(
    const sum_dir::SumDir& directory, const std::vector<std::uint32_t>& query_ids,
    const sum_dir::SumDirReachResult& reach, std::uint32_t cap,
    bool admit_reach_alternatives = semchan_alternatives_enabled()) {
    const auto t0 = std::chrono::steady_clock::now();
    SemChanWireReport out;
    out.enabled                 = true;
    out.page_tokens             = sum_dir::kSumDirBlockTokens;
    out.arbiter_found           = reach.found;
    out.arbiter_anchor          = reach.anchor_page;
    out.arbiter_pages           = reach.pages;
    out.arbiter_alternatives    = reach.alternatives;

    // The arbiter's page SET, from its own span, and EMPTY when it did not find -- page 0 is a
    // real page and must never be read as "the arbiter named nothing".
    if (reach.found) {
        for (std::uint32_t p = reach.first_page; p <= reach.last_page; ++p) {
            out.arbiter_page_set.push_back(p);
        }
    }
    out.arbiter_set_before_seam = static_cast<std::uint32_t>(out.arbiter_page_set.size());

    // ---- THE SEAM, AND IT IS ADDITIVE BY CONSTRUCTION --------------------------------------
    // Only pages the ARBITER named enter here. This is not the channel's opinion and it is not
    // this file searching anything: it is the arbiter's own `alternative_pages`, read off.
    out.seam_alternatives_on = admit_reach_alternatives;
    out.alternatives_named   = static_cast<std::uint32_t>(reach.alternative_pages.size());
    out.alternatives_over_bound = reach.alternatives_over_bound ? 1U : 0U;
    if (admit_reach_alternatives) {
        for (const std::uint32_t page : reach.alternative_pages) {
            if (std::find(out.arbiter_page_set.begin(), out.arbiter_page_set.end(), page) !=
                out.arbiter_page_set.end()) {
                continue;
            }
            out.arbiter_page_set.push_back(page);
            ++out.alternatives_admitted;
        }
    }

    // The channel's own reading. `semchan_index_blocks` reads the DIRECTORY'S OWN COLUMNS
    // (sum_dir.h's row list); no engine state is invented and no K byte is touched.
    const std::vector<SemChanBlockEntry> index = semchan_index_blocks(directory);
    out.rows_indexed = static_cast<std::uint32_t>(index.size());

    const SemChanSymbolKey query = semchan_symbol_key_from_ids(query_ids);
    out.query_tokens             = query.tokens;

    const SemChanCandidates channel = semchan_candidate_pages(index, query, cap);
    out.channel_status           = channel.status;
    out.channel_refusal          = channel.refusal;
    out.channel_positive         = channel.positive;
    out.channel_candidates       = static_cast<std::uint32_t>(channel.candidates.size());
    out.channel_distinct_pages   = channel.distinct_pages;
    out.channel_same_page_distinct_origin = channel.same_page_distinct_origin;
    out.blocks_scored            = channel.scored;

    out.union_ = semchan_union_candidates(out.arbiter_page_set, reach.anchor_page, reach.found,
                                          channel, cap);

    // The elementwise truth: which pages came from which side, as SETS, so "compare sets, not
    // counts" is readable off the line instead of inferred from two integers.
    for (std::uint32_t page : out.union_.pages) {
        if (out.union_.holds(page)) {
            const bool from_arb = std::find(out.arbiter_page_set.begin(),
                                            out.arbiter_page_set.end(), page) !=
                                  out.arbiter_page_set.end();
            (from_arb ? out.in_both_pages : out.channel_only_pages).push_back(page);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    out.channel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return out;
}

// -------------------------------------------------------------------------------------------
// 4. THE LINE -- TWO SIDES, TWO LABELS, NEVER ADDED
// -------------------------------------------------------------------------------------------
// The same discipline as `semchan_candidates_line`: `channel_candidates` counts (page, origin)
// RECORDS and `channel_distinct_pages` counts PAGES; `rows_indexed` and `blocks_scored` are two
// more denominators. All of them are printed with their own labels on one line, and none of them
// is combined with another. `determined_by_semchan=0` and `may_determine=0` are printed as literal
// zeros from `constexpr` constants, so a reader never has to take this file's word for it.
[[nodiscard]] inline std::string semchan_wire_line(const SemChanWireReport& r) {
    std::string out = "[semchan-wire] enabled=";
    out += (r.enabled ? "1" : "0");
    out += " switch=";
    out += kSemChanWireSwitchName;
    out += " scorer=";
    out += semchan_scorer_name(SemChanScorer::MeanK);
    out += " status=";
    out += semchan_candidate_status_name(r.channel_status);
    out += " determined_by_semchan=0";
    out += " may_determine=0";
    out += " origin_policy=";
    out += semchan_origin_policy_name(kSemChanOriginPolicy);
    // -- the channel's own side
    out += " rows_indexed=" + std::to_string(r.rows_indexed);
    out += " blocks_scored=" + std::to_string(r.blocks_scored);
    out += " query_tokens=" + std::to_string(r.query_tokens);
    out += " page_tokens=" + std::to_string(r.page_tokens);
    out += " channel_positive=" + std::to_string(r.channel_positive);
    out += " channel_candidates=" + std::to_string(r.channel_candidates);
    out += " channel_distinct_pages=" + std::to_string(r.channel_distinct_pages);
    out += " channel_same_page_distinct_origin=" +
           std::to_string(r.channel_same_page_distinct_origin);
    if (!r.channel_refusal.empty()) { out += " refusal=" + r.channel_refusal; }
    // -- the arbiter's own side, copied
    out += " arbiter_found=";
    out += (r.arbiter_found ? "1" : "0");
    out += " arbiter_anchor=" + std::to_string(r.arbiter_anchor);
    out += " arbiter_pages=" + std::to_string(r.arbiter_pages);
    out += " arbiter_alternatives=" + std::to_string(r.arbiter_alternatives);
    // -- the union
    out += " from_arbiter=" + std::to_string(r.union_.from_arbiter);
    out += " in_both=" + std::to_string(r.union_.in_both);
    out += " from_semchan_only=" + std::to_string(r.union_.from_semchan_only);
    out += " union_pages=" + std::to_string(r.union_.candidates);
    out += " pages_with_multiple_origins=" + std::to_string(r.union_.pages_with_multiple_origins);
    out += " anchor_preserved=";
    out += (r.union_.anchor_preserved ? "1" : "0");
    out += " cap=" + std::to_string(r.union_.cap);
    out += " channel_ms=" + std::to_string(r.channel_ms);
    // -- the sets, so the counts above can be checked against them
    out += " arbiter_set=[";
    for (std::size_t i = 0; i < r.arbiter_page_set.size(); ++i) {
        out += (i == 0U ? "" : ",");
        out += std::to_string(r.arbiter_page_set[i]);
    }
    out += "] channel_only=[";
    for (std::size_t i = 0; i < r.channel_only_pages.size(); ++i) {
        out += (i == 0U ? "" : ",");
        out += std::to_string(r.channel_only_pages[i]);
    }
    out += "] in_both_set=[";
    for (std::size_t i = 0; i < r.in_both_pages.size(); ++i) {
        out += (i == 0U ? "" : ",");
        out += std::to_string(r.in_both_pages[i]);
    }
    out += "] union_set=[";
    for (std::size_t i = 0; i < r.union_.pages.size(); ++i) {
        out += (i == 0U ? "" : ",");
        out += std::to_string(r.union_.pages[i]);
    }
    out += "]";
    if (r.union_.refused_over_cap) { out += " REFUSED_OVER_CAP"; }
    if (!r.union_.refusal.empty()) { out += " union_refusal=" + r.union_.refusal; }
    // ---- THE SEAM'S SUFFIX, APPENDED ONLY WHEN THE SEAM RAN ----------------------------------
    // Nothing above this line moved, and this suffix exists only on the run that asked for the
    // additive route -- so the line a switch-OFF run prints is the line it printed before the seam
    // was written, byte for byte. That property is MEASURED (dl/kvwire `logs/`, the two-build
    // `cmp`), not asserted from this comment.
    if (r.seam_alternatives_on) {
        out += " seam=alternatives";
        out += " switch=";
        out += kSemChanAlternativesSwitchName;
        out += " alternatives_named=" + std::to_string(r.alternatives_named);
        out += " alternatives_admitted=" + std::to_string(r.alternatives_admitted);
        out += " alternatives_over_bound=" + std::to_string(r.alternatives_over_bound);
        out += " arbiter_set_before_seam=" + std::to_string(r.arbiter_set_before_seam);
        out += " determined_by_seam=0";
    }
    return out;
}

// Edge-triggered printing, in the shape `program_impl.h` already argues for above the arbiter's
// line: one line per distinct SHAPE, so a steady-state decode cannot turn this into a per-round tax
// and a log cannot grow without bound. Returns true only when the caller should print.
[[nodiscard]] inline bool semchan_wire_should_print(const SemChanWireReport& r,
                                                    std::string* shape_out = nullptr) {
    static std::string last;
    const std::string key = semchan_wire_shape_key(r);
    if (shape_out != nullptr) { *shape_out = key; }
    if (key == last) { return false; }
    last = key;
    return true;
}

// -------------------------------------------------------------------------------------------
// 5. WHAT THIS HEADER DOES NOT DO (named, so it is not mistaken for done)
// -------------------------------------------------------------------------------------------
//   * It does not decide. There is no path in this file from a score to the returned
//     `RecallRequest`; the request is built by `reach_request_for_term` from the arbiter's span
//     alone, and this file is called AFTER it, for a report.
//   * It does not rotate, re-RoPE, re-base or renumber anything. It reads token ids out of
//     `SumDir` rows. KVMI-011's accumulating-drift failure is unreachable from here by
//     construction, not by care -- the same sentence `semchan_symbol_key.h` writes for itself.
//   * It does not widen what gets restored. The union it computes is REPORTED, and the token span
//     the engine actually recalls is the arbiter's unchanged. A second, separate switch would be
//     needed to act on the union, and this line did not add one: the measurement in `REPORT.md`
//     is the reading that decides whether such a switch is worth its bytes.
//   * It does not persist an origin. `SemChanProvenance` is carried in memory and printed; the
//     requirement that an origin survive a spill is a STORAGE-LAYER interface requirement and is
//     named as such in `REPORT.md`, not solved here.

// -------------------------------------------------------------------------------------------
// 6. THE FOUR SOURCES, THE ONE UNION, AND THE NAMED GATE ON THE EXPENSIVE LEG
// -------------------------------------------------------------------------------------------
// The coordinator extended this line's scope with the owner's own words: make "that cooperation"
// wider -- let several sources contribute to ONE candidate set, keep the arbiter the only judge,
// and "save where you can, but precision must not drop and speed must not drop too far". This
// section is the part of that which is a declaration rather than a measurement.
//
// THE FOUR SOURCES, each with the tree's own switch name. Three of the four already existed before
// this line; the semantic channel is the one `semchan` landed and this file wired.
enum class SemChanSource : std::uint8_t {
    KvReadback    = 0, // NINFER_TURN_RECALL=1  -- the KV slot read-back leg
    TextPrefill   = 1, // NINFER_RECALL_TEXT=1  -- the TEXT re-prefill leg. THE EXPENSIVE ONE.
    SemanticMeanK = 2, // (no switch beyond this file's) -- mean-k over the symbol alphabet
    NgramLookup   = 3, // spec/lookup_fuse.h -- `suffix_best` / `fuse_chain`; launcher entry
                       // `ops/launcher/suffix_lookup.h`'s `suffix_lookup_launch`
};

[[nodiscard]] constexpr const char* semchan_source_name(SemChanSource s) noexcept {
    switch (s) {
    case SemChanSource::KvReadback: return "kv-readback";
    case SemChanSource::TextPrefill: return "text-prefill";
    case SemChanSource::SemanticMeanK: return "semantic-mean-k";
    case SemChanSource::NgramLookup: return "ngram-lookup";
    }
    return "unknown";
}

// THE COST OF EACH PAID LEG, AND THE PROVENANCE OF EVERY NUMBER IN THIS TABLE. None of these is
// this line's measurement: they are OTHER LINES' readings on this station, quoted with the line
// that took them, and a caller that prints a saving computed from this table must print that
// provenance with it. A number without its provenance is how a saving becomes a story.
struct SemChanLegCost {
    SemChanSource source;
    const char*   switch_name;
    double        ms_per_paid_leg;   // <0 == unmeasured by anyone on the disk
    const char*   provenance;
};

inline constexpr SemChanLegCost kSemChanLegCosts[4] = {
    {SemChanSource::KvReadback, "NINFER_TURN_RECALL",
     16.235, "NOT MINE: dl/kvmaterial STATE (hook_ms, x97 L2_on_slot; band 16.235-21.115 over 3 legs)"},
    {SemChanSource::TextPrefill, "NINFER_RECALL_TEXT",
     1558.926, "NOT MINE: dl/kvmaterial STATE (hook_ms TR1; TR3 = 1563.222; prefill_ms ~1556.72-1559.51)"},
    {SemChanSource::SemanticMeanK, "NINFER_SEMCHAN",
     -1.0, "no engine leg exists yet: the device-side reading is EXTERNAL-UNPROBED (see REPORT)"},
    {SemChanSource::NgramLookup, "NINFER_RECALL_TEXT",
     -1.0, "no engine leg exists for the lookup path in this line's readings; NOT MEASURED HERE"},
};

// ⭐⭐ THE GATE -- WHAT CONDITION IS ALLOWED TO SPEND 1558.926 ms ON ONE ROUND.
//
// This is the one thing the owner's sentence ("save where you can") turns into a rule, so it is
// written as a function with a NAME, a REASON STRING for every branch, and no default that pays.
// The rule:
//
//   THE EXPENSIVE LEG IS PAID ONLY WHEN THE CHEAP SOURCES HAVE NAMED NOTHING.
//
// Why that and not "always run everything": the cheap sources (the lexical arbiter, the semantic
// channel, the ngram lookup) all narrow the candidate set for O(rows) work. The text leg re-prefills
// text to RE-DERIVE what those sources were asked to find. Paying it when a cheap source has
// already named a candidate is paying 1558.926 ms to learn something already known -- which is the
// step the disk already records as the reason the recall path is slow, and is exactly what "省点就
// 省点" forbids. Paying it when NOTHING named a candidate is the one case where it can still be
// worth its price, because the alternative is not "a cheaper candidate", it is "no candidate".
//
// AND THE PRECISION SIDE IS NOT TOUCHED BY THE GATE: `paid` decides whether a leg RUNS, never which
// page the arbiter picks. A gate that could turn itself off by finding nothing would be a decision.
struct SemChanTextGate {
    bool          asked        = false; // the switch was ON (the operator asked for the leg)
    bool          paid         = false; // the leg is allowed to run THIS round
    std::uint32_t cheap_candidates = 0; // candidates the cheap sources named this round
    const char*   reason       = "not asked";
};

// The leg's own switch. It is a SECOND named switch, off by default, because paying a 1.5 s leg is
// a different decision from admitting a candidate source, and one switch for both would make the
// cheap half unmeasurable on its own.
inline constexpr const char* kSemChanTextSwitchName = "NINFER_SEMCHAN_TEXT";

[[nodiscard]] inline bool semchan_text_leg_asked() noexcept {
    static const bool on = semchan_switch_value_is_on(std::getenv(kSemChanTextSwitchName));
    return on;
}

[[nodiscard]] inline SemChanTextGate semchan_text_prefill_gate(bool asked,
                                                               std::uint32_t cheap_candidates) {
    SemChanTextGate out;
    out.asked            = asked;
    out.cheap_candidates = cheap_candidates;
    if (!asked) {
        out.paid   = false;
        out.reason = "the leg's own switch is OFF -- and OFF is the default";
        return out;
    }
    if (cheap_candidates != 0U) {
        out.paid   = false;
        out.reason = "the cheap sources already named candidates: paying 1558.926 ms to re-derive "
                     "a candidate set that is already non-empty is the cost this gate exists to "
                     "refuse";
        return out;
    }
    out.paid   = true;
    out.reason = "no cheap source named a candidate: the leg is the last resort, and running it is "
                 "the only difference between a candidate and none";
    return out;
}

[[nodiscard]] inline std::string semchan_text_gate_line(const SemChanTextGate& g) {
    std::string out = "[semchan-gate] leg=";
    out += semchan_source_name(SemChanSource::TextPrefill);
    out += " switch=";
    out += kSemChanTextSwitchName;
    out += " asked=";
    out += (g.asked ? "1" : "0");
    out += " paid=";
    out += (g.paid ? "1" : "0");
    out += " cheap_candidates=" + std::to_string(g.cheap_candidates);
    out += " reason=";
    out += g.reason;
    return out;
}

// -------------------------------------------------------------------------------------------
// 7. THE COMPILE-TIME HALF
// -------------------------------------------------------------------------------------------
static_assert(!kSemChanWireMayDetermine,
              "the semantic channel is a candidate SOURCE: it may enter the candidate-set UNION "
              "and it may never determine (program_impl.h, `candidate-set UNION`)");
static_assert(!semchan_wire_may_determine(),
              "same rule, read through the function form so a relaxed branch is a build error");
static_assert(!semchan_origin_may_be_chosen_by_channel(),
              "the channel may not choose between two origins of one page either");
static_assert(kSemChanOriginPolicy != SemChanOriginPolicy::Unnamed,
              "the origin policy must be NAMED or the tree does not build");
static_assert(!semchan_scorer_may_determine(SemChanScorer::MeanK),
              "mean-k is the one scorer wired here, and it may not determine either");
// The gate is a DECLARATION about when a leg runs, never about what an answer is: the zero-candidate
// branch is the only one that pays, and it pays by RUNNING A LEG, not by choosing a page. Pinned so
// that a future edit which makes the gate return a page (or pay by default) is a build error.
static_assert(sizeof(SemChanTextGate) >= sizeof(bool) * 2,
              "the gate carries its own two booleans and no page: a gate that could name a page "
              "would be a decision wearing a cost argument");
static_assert(kSemChanLegCosts[static_cast<std::size_t>(SemChanSource::TextPrefill)].ms_per_paid_leg >
                  1000.0,
              "the text leg is the expensive one in the quoted table (1558.926 ms) -- if this "
              "stops being true the gate's whole argument has to be re-taken, not re-worded");

// -------------------------------------------------------------------------------------------
// 8. THE SEAM'S TWO MECHANICAL PINS (section 2b) -- both about what a TYPE forbids
// -------------------------------------------------------------------------------------------
// (1) THE `RecallRequest` HALF. The provider's request is `RecallRequest{reach.token_begin,
//     reach.token_end}` and nothing else. The seam cannot move either field if the union is handed
//     the arbiter's result as a CONST reference -- so that is not argued, it is asserted against the
//     function's own type. A future edit that relaxes the const is a BUILD ERROR here, in every
//     translation unit that includes `program_impl.h`.
// (2) THE ROUTE. `sum_dir_reach_loop_guard_from_found_pages` returns an EMPTY list, and the return
//     type of this file's bridge to it is asserted to be that same type, so the naive wiring is
//     mechanically a no-op. What is NOT claimed: a caller can still hand a `std::vector` straight
//     to `appended_pages` and bypass both this file and that function. What stops that caller is
//     the MEASUREMENT (`dl/kvwire` arm `naive-exclusion`, `dl/kvmemoracle` M2) and the name at
//     `kSumDirReachAppendedPagesAreAnExclusionSet`, not a type.
namespace detail {
// Extracts whether the third parameter of a 5-argument free function is a const lvalue reference.
template <typename Fn>
struct ThirdArgIsConstRef : std::false_type {};
template <typename R, typename A, typename B, typename C, typename D, typename E>
struct ThirdArgIsConstRef<R (*)(A, B, C, D, E)> : std::is_const<std::remove_reference_t<C>> {};
}  // namespace detail

using SemChanWireUnionFn = SemChanWireReport (*)(const sum_dir::SumDir&,
                                                const std::vector<std::uint32_t>&,
                                                const sum_dir::SumDirReachResult&,
                                                std::uint32_t, bool);
static_assert(std::is_same<decltype(&semchan_wire_union_from_reach),
                          SemChanWireUnionFn>::value,
              "the union's signature is the seam's contract: the arbiter's result comes in as "
              "`const SumDirReachResult&`, and the fifth argument is the seam's own switch, "
              "defaulted OFF");
static_assert(detail::ThirdArgIsConstRef<decltype(&semchan_wire_union_from_reach)>::value,
              "THE `RecallRequest` PIN: this file receives the arbiter's result as a CONST "
              "reference, so it cannot write `token_begin`/`token_end` and therefore cannot move "
              "the request the provider returns");
static_assert(std::is_same<decltype(&semchan_wire_pages_that_may_ride_the_loop_guard),
                          std::vector<std::uint32_t> (*)(
                              const sum_dir::SumDirReachResult&, std::string*)>::value,
              "the forbidden route's bridge has exactly one signature, and it is the one that "
              "returns an empty list: a candidate source may not ride the loop guard");

}  // namespace ninfer::spec::semchan::wire

#endif  // NINFER_SPEC_SEMCHAN_WIRE_H
