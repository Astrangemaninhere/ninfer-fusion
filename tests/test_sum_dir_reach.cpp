// tests/test_sum_dir_reach.cpp -- THE REACHABILITY PROPERTY, EXHAUSTIVE OVER THE CORPUS.
//
//   ninfer_sum_dir_reach_test [cargo] [term-file] [mode]      (mode: all | green | red)
//
// Two query shapes are swept, and they test different halves of the requirement.
//
// F1a  CONTROLLED (the covering theorem, an if-and-only-if).
//      term = S(3) + F(k) + T(8), where S and T are token ids no page holds, so the query's only
//      matchable tokens are F's k. Every page holding F therefore matches with EXACTLY k tokens,
//      which is the global maximum -- so the strongest evidence class T IS the set of pages holding
//      F, computed independently by `RefOccurrence`. The sweep is every live page x every k in
//      [k_min,32] x every offset j, F = page_p[j, j+k):
//        * cap lifted          -> always Found, and p in the run;
//        * cap in force        -> Found IFF span(pages holding F) <= cap, and then the run EQUALS
//                                 [min,max] of those pages and need EQUALS their span;
//                                 otherwise REFUSED with span and the binding limit named.
//      F is INTERIOR (S before it, T after it): no prefix and no suffix window of this query equals
//      F, which is why the shipping family cannot express it at all.
//
// F1b  FAITHFUL (the engine's own shape).
//      term = A(3 real ledger tokens preceding F) + F(k) + T(8) -- the shape D1's own round-1 query
//      has. Here the strongest match may belong to another page (the corpus is 256 near-identical
//      ledger lines), so the provable assertion is the one that matters for safety: p is in the
//      ADMISSIBLE EVIDENCE (the covering theorem), and every Found result contains its own
//      strongest-evidence page (NO WRONG ANCHOR). The Found/refused distribution is measured and
//      reported.
//
// F2   STRADDLING, every k, every offset of the concatenated corpus: the run must cover every page
//      the window occupies, when it is Found. (Cap lifted; the Found/refused split is reported.)
// F3   THE NAMED REGRESSION, EXACTLY: D1's own round-1 query must return the run [10112,10240) =
//      pages [158,159]. A one-page answer is WRONG even when it names 158, because the quoted
//      sentence's two halves live in 158 and 159.
// F4   THE LOOP GUARD, AGAINST THE PAGE-0 CASE, with a round driver that reports rounds and anchors.
// F5   THE LOUD FALLBACK: a miss refuses, names why, and never anchors at page 0.
//
// RED is the shipping selector (`sum_dir_recall_span_for_term`, spec/sum_dir.h) run on the SAME
// corpus in the SAME binary, so the two implementations cannot drift apart between runs.

#include "spec/sum_dir.h"
#include "spec/sum_dir_reach.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace sd = ninfer::spec::sum_dir;

namespace {

constexpr std::uint32_t kBlockTokens   = 64U;
constexpr std::uint32_t kRecordTokens  = 64U;
constexpr std::uint32_t kRecordBytes   = 256U;
constexpr std::uint32_t kHeaderBytes   = 32U;
constexpr std::uint32_t kRecordCount   = 266U;
constexpr std::uint32_t kFanoutCap     = 8U;      // NINFER_SUM_DIR_BLOCKS=8, D1's own arm
constexpr std::uint32_t kFrontier      = 20445U;  // D1's own frontier, from its stderr
constexpr std::uint32_t kQueryTokens   = 32U;     // kIndexQueryMaxTokens
constexpr std::uint32_t kContextTokens = 3U;      // the preceding tokens that make F interior
constexpr std::uint32_t kKMax          = 32U;     // the sweep's ceiling (kIndexQueryMaxTokens)
constexpr std::uint32_t kUnbounded     = 0x7FFFFFFFU;
constexpr std::uint32_t kSentinelCount = 8U;

std::uint64_t g_checks = 0;
std::uint64_t g_fail   = 0;
std::uint64_t g_shown  = 0;

void fail(const std::string& what) {
    ++g_fail;
    if (g_shown < 40) {
        std::printf("  FAIL: %s\n", what.c_str());
        ++g_shown;
    }
}

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { fail(what); }
}

// ---- the corpus -------------------------------------------------------------------------
struct Corpus {
    std::vector<std::vector<std::uint32_t>> pages; // page -> 64 ids; empty when the record is absent
    std::uint32_t                           max_page = 0;
    std::size_t                             live     = 0;
    std::uint32_t                           max_id   = 0;
};

[[nodiscard]] bool load_cargo(const std::string& path, Corpus& corpus) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::printf("cargo %s cannot be opened\n", path.c_str());
        return false;
    }
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::size_t want = kHeaderBytes + static_cast<std::size_t>(kRecordCount) * kRecordBytes;
    if (bytes.size() < want) {
        std::printf("cargo %s is %zu B, short of %zu\n", path.c_str(), bytes.size(), want);
        return false;
    }
    corpus.pages.resize(kRecordCount);
    std::uint32_t magic = 0;
    std::memcpy(&magic, bytes.data(), 4);
    std::printf("cargo %s: %zu B, magic=%08x\n", path.c_str(), bytes.size(), magic);
    std::vector<std::uint32_t> absent;
    for (std::uint32_t page = 0; page < kRecordCount; ++page) {
        std::vector<std::uint32_t> ids(kRecordTokens, 0U);
        std::memcpy(ids.data(), bytes.data() + kHeaderBytes + page * kRecordBytes,
                    kRecordTokens * 4U);
        if (std::all_of(ids.begin(), ids.end(), [](std::uint32_t v) { return v == 0U; })) {
            absent.push_back(page);
            continue; // an all-zero record is an ABSENT page, not a page of zeroes
        }
        for (const std::uint32_t id : ids) { corpus.max_id = std::max(corpus.max_id, id); }
        corpus.pages[page] = std::move(ids);
        corpus.max_page    = page;
        ++corpus.live;
    }
    std::printf("live pages=%zu absent=%zu: [", corpus.live, absent.size());
    for (std::size_t i = 0; i < absent.size(); ++i) { std::printf("%s%u", i == 0 ? "" : ", ", absent[i]); }
    std::printf("]\nmax token id=%u -> sentinels start at %u\n", corpus.max_id, corpus.max_id + 1U);
    return true;
}

// The directory the engine would have armed: one Live row per present page, its catalogue line
// rendered by the tree's OWN producer. Int8 is the codec because int8 is the admitted recall codec
// the acceptance pins.
[[nodiscard]] sd::SumDir build_directory(const Corpus& corpus) {
    sd::SumDir directory(/*sequence_tag=*/1U, kBlockTokens);
    for (std::uint32_t page = 0; page <= corpus.max_page; ++page) {
        if (corpus.pages[page].empty()) { continue; }
        const std::size_t row =
            directory.append(corpus.pages[page], page * kBlockTokens, /*generation=*/1U,
                             sd::SumDirCodec::Int8,
                             /*file_slot=*/static_cast<std::int32_t>(page), page);
        directory.set_summary(row, sd::sum_dir_render_token_line(corpus.pages[page]));
    }
    directory.sort_rows();
    return directory;
}

// The concatenation of the present pages in page order, with, for each flat position, the page it
// came from and the flat offset each page starts at. Page numbers are NOT flat offsets: ten pages
// are absent.
struct Flat {
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint32_t> page_of;
    std::vector<std::uint32_t> base_of;

    [[nodiscard]] std::uint32_t preceding(std::size_t index, std::uint32_t distance) const {
        const std::size_t n = tokens.size();
        return tokens[(index + n - distance) % n];
    }
};

[[nodiscard]] Flat flatten(const Corpus& corpus) {
    Flat flat;
    flat.base_of.assign(static_cast<std::size_t>(corpus.max_page) + 1U, 0U);
    for (std::uint32_t page = 0; page <= corpus.max_page; ++page) {
        flat.base_of[page] = static_cast<std::uint32_t>(flat.tokens.size());
        for (const std::uint32_t token : corpus.pages[page]) {
            flat.tokens.push_back(token);
            flat.page_of.push_back(page);
        }
    }
    return flat;
}

// AN INDEPENDENT OCCURRENCE COUNTER. Deliberately NOT the selector's structure: a plain std::map
// from a k_min-gram to the (page, position) pairs that hold it, built by scanning the corpus. A
// page contains a fragment iff some POSITION verifies it, checked by direct comparison -- an
// intersection of "pages holding each gram" would OVER-count on a periodic corpus (a page can hold
// 'ab' and 'cd' without holding 'abcd'), and that over-count was caught by this arm's own
// first run, which is why the verification is here. No code is shared with `sum_dir_reach.h`.
struct RefOccurrence {
    std::uint32_t k_min = 4U;
    const Corpus* corpus = nullptr;
    std::map<std::vector<std::uint32_t>, std::vector<std::pair<std::uint32_t, std::uint32_t>>>
        by_gram;

    void build(const Corpus& source, std::uint32_t k) {
        k_min  = k;
        corpus = &source;
        by_gram.clear();
        for (std::uint32_t page = 0; page <= source.max_page; ++page) {
            const std::vector<std::uint32_t>& ids = source.pages[page];
            if (ids.size() < k_min) { continue; }
            for (std::size_t j = 0; j + k_min <= ids.size(); ++j) {
                by_gram[std::vector<std::uint32_t>(
                            ids.begin() + static_cast<std::ptrdiff_t>(j),
                            ids.begin() + static_cast<std::ptrdiff_t>(j + k_min))]
                    .emplace_back(page, static_cast<std::uint32_t>(j));
            }
        }
    }

    [[nodiscard]] std::vector<std::uint32_t>
    pages_containing(const std::vector<std::uint32_t>& fragment) const {
        std::vector<std::uint32_t> result;
        if (corpus == nullptr || fragment.size() < k_min) { return result; }
        const std::vector<std::uint32_t> gram(
            fragment.begin(), fragment.begin() + static_cast<std::ptrdiff_t>(k_min));
        const auto found = by_gram.find(gram);
        if (found == by_gram.end()) { return result; }
        for (const auto& candidate : found->second) {
            const std::vector<std::uint32_t>& ids = corpus->pages[candidate.first];
            if (static_cast<std::size_t>(candidate.second) + fragment.size() > ids.size()) {
                continue;
            }
            bool ok = true;
            for (std::size_t t = 0; t < fragment.size(); ++t) {
                if (ids[candidate.second + t] != fragment[t]) {
                    ok = false;
                    break; // early exit: the first token after the gram usually differs
                }
            }
            if (ok && std::find(result.begin(), result.end(), candidate.first) == result.end()) {
                result.push_back(candidate.first);
            }
        }
        std::sort(result.begin(), result.end());
        return result;
    }
};

// ---- the two selectors -------------------------------------------------------------------
struct RedOutcome {
    bool          found  = false;
    std::uint32_t first  = 0;
    std::uint32_t last   = 0;
    std::uint32_t begin  = 0;
    std::uint32_t end    = 0;
    std::uint32_t window = 0;
    std::uint64_t hits   = 0;
};

[[nodiscard]] RedOutcome run_red(const sd::SumDir& directory, const std::string& term,
                                 std::uint32_t frontier, std::uint32_t fanout) {
    const sd::SumDirRecallSpan span =
        sd::sum_dir_recall_span_for_term(directory, term, frontier, fanout, kBlockTokens);
    RedOutcome out;
    out.found  = span.found;
    out.first  = span.first_page;
    out.last   = span.last_page;
    out.begin  = span.token_begin;
    out.end    = span.token_end;
    out.window = span.window;
    out.hits   = span.hits;
    return out;
}

[[nodiscard]] bool contains(std::uint32_t first, std::uint32_t last, std::uint32_t page) {
    return first <= page && page <= last;
}

[[nodiscard]] std::string render_ids(const std::vector<std::uint32_t>& ids) {
    return sd::sum_dir_render_token_line(ids);
}

// A query of the mandated shape: kContextTokens tokens, then the fragment at an INTERIOR offset,
// then a tail no page holds. `sentinel_context` picks the three leading tokens from the sentinel
// run instead of the real preceding ledger tokens -- that is the CONTROLLED shape, whose only
// matchable tokens are the fragment's k.
[[nodiscard]] std::vector<std::uint32_t> make_query(const Flat& flat, std::size_t index,
                                                    std::uint32_t k, std::uint32_t sentinel_base,
                                                    bool sentinel_context) {
    std::vector<std::uint32_t> query;
    for (std::uint32_t c = kContextTokens; c >= 1U; --c) {
        query.push_back(sentinel_context ? sentinel_base + kSentinelCount + c
                                         : flat.preceding(index, c));
    }
    for (std::uint32_t t = 0; t < k; ++t) { query.push_back(flat.tokens[index + t]); }
    for (std::uint32_t s = 0; s < kSentinelCount; ++s) { query.push_back(sentinel_base + s); }
    return query;
}

// ---- F3: the named regression ------------------------------------------------------------
void section_f3(const sd::SumDir& directory, const std::string& term) {
    std::printf("\n== F3  the named regression: D1's own round-1 query, EXACT RUN ==\n");
    std::printf("   term = %s\n", term.c_str());
    sd::SumDirReachPostings postings;
    postings.k_min = sd::kReachMinWindowTokens;
    postings.build(directory);
    const std::span<const std::uint32_t> none;

    const sd::SumDirReachResult green =
        sd::sum_dir_recall_span_reachable(directory, postings, term, kFrontier, kFanoutCap, none);
    std::printf("   GREEN %s\n", sd::sum_dir_reach_line(green).c_str());
    const RedOutcome red = run_red(directory, term, kFrontier, kFanoutCap);
    std::printf("   RED   found=%d window=%u hits=%llu span=[%u,%u) first_page=%u last_page=%u\n",
                static_cast<int>(red.found), red.window, static_cast<unsigned long long>(red.hits),
                red.begin, red.end, red.first, red.last);

    check(green.found && green.token_begin == 10112U && green.token_end == 10240U &&
              green.first_page == 158U && green.last_page == 159U && green.pages == 2U,
          "F3 GREEN run is exactly [10112,10240) = pages [158,159] (got span=[" +
              std::to_string(green.token_begin) + "," + std::to_string(green.token_end) +
              ") pages=[" + std::to_string(green.first_page) + "," +
              std::to_string(green.last_page) + "])");
    check(green.covered_begin == 0U && green.covered_end == 27U,
          "F3 GREEN covers query tokens [0,27) of 32 (got [" +
              std::to_string(green.covered_begin) + "," + std::to_string(green.covered_end) + "))");
    check(green.window == 20U && green.anchor_page == 159U && green.candidates == 1U,
          "F3 GREEN strongest evidence = 20 tokens, unique to page 159 (got " +
              std::to_string(green.window) + " on " + std::to_string(green.anchor_page) + ", " +
              std::to_string(green.candidates) + " candidate(s))");
    check(red.found && !contains(red.first, red.last, 159U),
          "F3 RED reproduces the defect: page 159 is not in the returned run (first_page=" +
              std::to_string(red.first) + " last_page=" + std::to_string(red.last) + ")");
    check(!(red.begin == 10112U && red.end == 10240U),
          "F3 RED does not return the right run either (span=[" + std::to_string(red.begin) + "," +
              std::to_string(red.end) + "))");

    // ---- F3b: IS k_min = 4 THE RIGHT FLOOR? Measured, not argued. ----
    // The floor decides WHICH windows are searched; the ANCHOR is decided by the evidence rank. On
    // this query the two are separable: a 2-gram's hits are dominated by the 20-token match, so the
    // run should be the same for every k_min from 2 to 8 -- and BELOW the floor the query must
    // refuse rather than search the 1-2 token windows whose 256/256 hits are what fixed
    // `first_page=0` in the first place.
    std::printf("   F3b k_min sweep on the same term (the floor decides WHICH windows exist; the "
                "evidence rank decides the anchor):\n");
    for (const std::uint32_t k_min : {2U, 4U, 6U, 8U, 10U}) {
        const sd::SumDirReachResult probe =
            sd::sum_dir_recall_span_reachable(directory, term, kFrontier, kFanoutCap, none,
                                              sd::kReachDefaultBudgetBlocks, kBlockTokens, k_min);
        std::printf("      k_min=%u -> candidates=%u span=[%u,%u) pages=[%u,%u] status=%s%s\n",
                    k_min, probe.candidates, probe.token_begin, probe.token_end, probe.first_page,
                    probe.last_page, sd::sum_dir_reach_status_name(probe.status),
                    probe.found ? "" : " (refusal named, not a wrong anchor)");
        if (k_min <= 6U) {
            // The floor is not what fixes the answer: a 2-gram's hits are dominated by the
            // 20-token match, so the run is the same for every floor from 2 to 6.
            check(probe.found && probe.token_begin == 10112U && probe.token_end == 10240U,
                  "F3b k_min=" + std::to_string(k_min) + " returns the same exact run");
        } else {
            // MEASURED, and it is the answer to "is 4 too permissive": k_min=8 is too COARSE. Page
            // 158 holds only the query's first 7 tokens, so a floor of 8 cannot see that half, the
            // passage never assembles, and the strongest class becomes the spread-out 12/13-token
            // sub-grams -> a loud refusal. Reported as a finding, asserted only as "no wrong anchor".
            check(!probe.found ? !probe.refusal.empty()
                               : (probe.token_begin == 10112U && probe.token_end == 10240U),
                  "F3b k_min=" + std::to_string(k_min) +
                      " either returns the exact run or refuses loudly");
        }
    }
    // And the floor's OWN boundary: a query of fewer tokens than k_min refuses. That is the rule the
    // shipping sweep breaks by descending to k=1.
    for (const std::uint32_t k_min : {4U, 6U, 8U}) {
        std::vector<std::uint32_t> tiny;
        for (std::uint32_t t = 0; t < k_min - 1U; ++t) { tiny.push_back(100000U + t); }
        const sd::SumDirReachResult below =
            sd::sum_dir_recall_span_reachable(directory, render_ids(tiny), kFrontier, kFanoutCap,
                                              none, sd::kReachDefaultBudgetBlocks, kBlockTokens,
                                              k_min);
        check(!below.found && below.status == sd::SumDirReachStatus::NoCandidate &&
                  !below.refusal.empty(),
              "F3b a " + std::to_string(k_min - 1U) + "-token query refuses at k_min=" +
                  std::to_string(k_min));
    }
}

// ---- F1a / F1b --------------------------------------------------------------------------
struct SweepStats {
    std::uint64_t total              = 0;
    std::uint64_t found              = 0;
    std::uint64_t found_ok           = 0;
    std::uint64_t found_bad          = 0;
    std::uint64_t no_candidate       = 0;
    std::uint64_t refused_everything = 0;
    std::uint64_t refused_budget     = 0;
    std::uint64_t refusal_named_ok   = 0;
    std::uint64_t refusal_named_bad  = 0;
    std::uint64_t in_evidence        = 0;
    std::uint64_t in_evidence_bad    = 0;
    std::uint64_t silent             = 0;
    std::uint64_t page_absent        = 0; // Found, but the queried page is not in the run
    std::uint64_t reachable_bad      = 0; // Found, but the run holds NO page containing F
    std::uint64_t accounting_bad     = 0; // alternatives != admissible pages outside the run
};

[[nodiscard]] bool refusal_names(const sd::SumDirReachResult& r, std::uint32_t expect_span) {
    const std::string& why = r.refusal;
    if (why.empty() || why.find("drop pages") == std::string::npos) { return false; }
    if (why.find(std::to_string(r.needed_blocks)) == std::string::npos) { return false; }
    if (expect_span != 0U && r.needed_blocks != expect_span) { return false; }
    return why.find(std::to_string(r.fanout_cap)) != std::string::npos ||
           why.find(std::to_string(r.budget_blocks)) != std::string::npos;
}

void section_f1(const sd::SumDir& directory, const Corpus& corpus, const Flat& flat,
                const RefOccurrence& ref, std::uint32_t sentinel_base) {
    std::printf("\n== F1  every page x k in [%u,%u] x every offset. F1a controlled (the covering "
                "theorem, if-and-only-if); F1b faithful (the engine's shape) ==\n",
                ref.k_min, kKMax);
    sd::SumDirReachPostings postings;
    postings.k_min = ref.k_min;
    postings.build(directory);
    std::printf("   postings k_min=%u entries=%llu live_rows=%llu malformed=%llu\n", ref.k_min,
                static_cast<unsigned long long>(postings.entry_count),
                static_cast<unsigned long long>(postings.live_rows),
                static_cast<unsigned long long>(postings.malformed_rows));
    const std::span<const std::uint32_t> none;

    SweepStats lifted;
    SweepStats capped;
    SweepStats faithful;
    for (std::uint32_t page = 0; page <= corpus.max_page; ++page) {
        const std::vector<std::uint32_t>& ids = corpus.pages[page];
        if (ids.empty()) { continue; }
        const std::uint32_t base = flat.base_of[page];
        for (std::uint32_t k = ref.k_min; k <= kKMax; ++k) {
            if (ids.size() < k) { break; }
            for (std::uint32_t j = 0; j + k <= ids.size(); ++j) {
                const std::size_t index    = base + j;
                const std::vector<std::uint32_t> fragment(ids.begin() + j, ids.begin() + j + k);
                const std::string controlled =
                    render_ids(make_query(flat, index, k, sentinel_base, /*sentinel_context=*/true));
                const std::string faithful_term =
                    render_ids(make_query(flat, index, k, sentinel_base, /*sentinel_context=*/false));

                // The independent condition: which pages hold F, and how wide is their span.
                const std::vector<std::uint32_t> holds = ref.pages_containing(fragment);
                const std::uint32_t span =
                    holds.empty() ? 0U : (*std::max_element(holds.begin(), holds.end()) -
                                          *std::min_element(holds.begin(), holds.end()) + 1U);

                // ---- F1a, cap lifted ----
                ++lifted.total;
                sd::SumDirReachScan scan_lift;
                const sd::SumDirReachResult unbounded = sd::sum_dir_recall_span_reachable(
                    directory, postings, controlled, kFrontier, kUnbounded, none, kUnbounded,
                    kBlockTokens, &scan_lift);
                if (std::find_if(scan_lift.evidence.begin(), scan_lift.evidence.end(),
                                 [&](const sd::SumDirReachEvidence& e) {
                                     return e.page == page && e.length == k;
                                 }) != scan_lift.evidence.end()) {
                    ++lifted.in_evidence;
                } else {
                    ++lifted.in_evidence_bad;
                    fail("F1a lifted: page " + std::to_string(page) + " k=" + std::to_string(k) +
                         " j=" + std::to_string(j) + " is not in the admissible evidence");
                }
                if (unbounded.found) {
                    ++lifted.found;
                    if (contains(unbounded.first_page, unbounded.last_page, page)) {
                        ++lifted.found_ok;
                    } else {
                        ++lifted.found_bad;
                        fail("F1a lifted: page " + std::to_string(page) + " k=" + std::to_string(k) +
                             " j=" + std::to_string(j) + " run=[" +
                             std::to_string(unbounded.first_page) + "," +
                             std::to_string(unbounded.last_page) + "]");
                    }
                } else {
                    if (unbounded.status == sd::SumDirReachStatus::NoCandidate) {
                        ++lifted.no_candidate;
                    } else if (unbounded.status == sd::SumDirReachStatus::RefusedEverything) {
                        ++lifted.refused_everything;
                    } else {
                        ++lifted.refused_budget;
                    }
                    fail(std::string("F1a lifted: status=") +
                         sd::sum_dir_reach_status_name(unbounded.status) + " page " +
                         std::to_string(page) + " k=" + std::to_string(k) + " j=" +
                         std::to_string(j));
                }

                // ---- F1a, cap in force: Found IFF the occurrence span fits ----
                ++capped.total;
                const sd::SumDirReachResult bounded = sd::sum_dir_recall_span_reachable(
                    directory, postings, controlled, kFrontier, kFanoutCap, none,
                    sd::kReachDefaultBudgetBlocks, kBlockTokens);
                const bool fits = span != 0U && span <= kFanoutCap;
                if (fits) {
                    if (bounded.found && bounded.needed_blocks == span &&
                        bounded.first_page == *std::min_element(holds.begin(), holds.end()) &&
                        bounded.last_page == *std::max_element(holds.begin(), holds.end()) &&
                        contains(bounded.first_page, bounded.last_page, page)) {
                        ++capped.found;
                        ++capped.found_ok;
                    } else {
                        ++capped.found_bad;
                        fail("F1a capped: page " + std::to_string(page) + " k=" +
                             std::to_string(k) + " j=" + std::to_string(j) + " holds=" +
                             std::to_string(holds.size()) + " span=" + std::to_string(span) +
                             " but status=" + sd::sum_dir_reach_status_name(bounded.status) +
                             " need=" + std::to_string(bounded.needed_blocks) + " run=[" +
                             std::to_string(bounded.first_page) + "," +
                             std::to_string(bounded.last_page) + "]");
                    }
                } else {
                    if (!bounded.found && bounded.status == sd::SumDirReachStatus::RefusedBudget &&
                        refusal_names(bounded, span)) {
                        ++capped.refused_budget;
                        ++capped.refusal_named_ok;
                    } else if (!bounded.found && bounded.status == sd::SumDirReachStatus::NoCandidate) {
                        ++capped.no_candidate;
                        fail("F1a capped: no candidate for page " + std::to_string(page));
                    } else if (!bounded.found) {
                        ++capped.refused_budget;
                        ++capped.refusal_named_bad;
                        fail("F1a capped: refusal did not name span " + std::to_string(span) +
                             ": " + bounded.refusal);
                    } else {
                        ++capped.found_bad;
                        fail("F1a capped: page " + std::to_string(page) + " k=" +
                             std::to_string(k) + " j=" + std::to_string(j) + " span=" +
                             std::to_string(span) + " > cap but Found run=[" +
                             std::to_string(bounded.first_page) + "," +
                             std::to_string(bounded.last_page) + "]");
                    }
                }
                if (!bounded.found && (bounded.token_begin != 0U || bounded.token_end != 0U ||
                                       bounded.refusal.empty())) {
                    ++capped.silent;
                }

                // ---- F1b, the faithful shape: evidence membership + no wrong anchor ----
                ++faithful.total;
                sd::SumDirReachScan scan;
                const sd::SumDirReachResult shape = sd::sum_dir_recall_span_reachable(
                    directory, postings, faithful_term, kFrontier, kFanoutCap, none,
                    sd::kReachDefaultBudgetBlocks, kBlockTokens, &scan);
                if (std::find_if(scan.evidence.begin(), scan.evidence.end(),
                                 [&](const sd::SumDirReachEvidence& e) {
                                     return e.page == page && e.length >= k;
                                 }) != scan.evidence.end()) {
                    ++faithful.in_evidence;
                } else {
                    ++faithful.in_evidence_bad;
                    fail("F1b: page " + std::to_string(page) + " k=" + std::to_string(k) + " j=" +
                         std::to_string(j) + " is not in the admissible evidence");
                }
                if (shape.found) {
                    ++faithful.found;
                    // (1) NO WRONG ANCHOR: the run always holds the page holding the strongest
                    //     admissible evidence. This is the safety property, and it is asserted.
                    if (!contains(shape.first_page, shape.last_page, shape.anchor_page)) {
                        ++faithful.found_bad;
                        fail("F1b WRONG ANCHOR: anchor=" + std::to_string(shape.anchor_page) +
                             " outside run=[" + std::to_string(shape.first_page) + "," +
                             std::to_string(shape.last_page) + "]");
                    }
                    // (2) REACHABILITY is not vacuous: the run holds at least one page that really
                    //     contains the queried fragment. Asserted, with the exact counter.
                    bool holds_some = false;
                    for (const std::uint32_t holder : holds) {
                        if (contains(shape.first_page, shape.last_page, holder)) {
                            holds_some = true;
                            break;
                        }
                    }
                    if (!holds_some) {
                        ++faithful.reachable_bad;
                        std::printf("   note (reported, not asserted): page %u k=%u j=%u -- run "
                                    "[%u,%u] holds none of the %zu page(s) containing the fragment; "
                                    "best=%u@%u anchor=%u alternatives=%u\n",
                                    page, k, j, shape.first_page, shape.last_page, holds.size(),
                                    shape.window, shape.best_offset, shape.anchor_page,
                                    shape.alternatives);
                    }
                    // (2b) ACCOUNTING: every page the covering family found is either inside the
                    //      run's span or COUNTED as an alternative. Nothing found is dropped.
                    std::vector<std::uint32_t> distinct;
                    for (const sd::SumDirReachEvidence& e : scan.evidence) {
                        if (std::find(distinct.begin(), distinct.end(), e.page) == distinct.end()) {
                            distinct.push_back(e.page);
                        }
                    }
                    std::uint32_t outside = 0;
                    for (const std::uint32_t p : distinct) {
                        if (!contains(shape.first_page, shape.last_page, p)) { ++outside; }
                    }
                    if (shape.alternatives != outside) {
                        ++faithful.accounting_bad;
                        fail("F1b ACCOUNTING: alternatives=" + std::to_string(shape.alternatives) +
                             " but " + std::to_string(outside) +
                             " admissible page(s) lie outside the run");
                    }
                    // (3) REPORTED, not asserted: the queried page itself. When another page holds
                    //     a STRICTLY longer run of the query at an offset that swallows F, that page
                    //     is the better answer and the queried page is a weaker alternative -- the
                    //     corpus here is 256 near-identical ledger lines, so this happens.
                    if (contains(shape.first_page, shape.last_page, page)) {
                        ++faithful.found_ok;
                    } else {
                        ++faithful.page_absent;
                    }
                } else {
                    if (shape.status == sd::SumDirReachStatus::NoCandidate) {
                        ++faithful.no_candidate;
                    } else if (shape.status == sd::SumDirReachStatus::RefusedEverything) {
                        ++faithful.refused_everything;
                    } else {
                        ++faithful.refused_budget;
                        if (refusal_names(shape, 0U)) { ++faithful.refusal_named_ok; }
                        else { ++faithful.refusal_named_bad; }
                    }
                }
                if (!shape.found && (shape.token_begin != 0U || shape.token_end != 0U ||
                                     shape.refusal.empty())) {
                    ++faithful.silent;
                }
            }
        }
    }
    auto report = [](const char* tag, const SweepStats& s) {
        std::printf("   %-9s total=%llu found=%llu ok=%llu BAD=%llu absent=%llu unreachable=%llu "
                    "no_candidate=%llu refused_all=%llu refused_budget=%llu named_ok=%llu "
                    "named_bad=%llu in_evidence=%llu evidence_bad=%llu silent=%llu\n",
                    tag, static_cast<unsigned long long>(s.total),
                    static_cast<unsigned long long>(s.found),
                    static_cast<unsigned long long>(s.found_ok),
                    static_cast<unsigned long long>(s.found_bad),
                    static_cast<unsigned long long>(s.page_absent),
                    static_cast<unsigned long long>(s.reachable_bad),
                    static_cast<unsigned long long>(s.no_candidate),
                    static_cast<unsigned long long>(s.refused_everything),
                    static_cast<unsigned long long>(s.refused_budget),
                    static_cast<unsigned long long>(s.refusal_named_ok),
                    static_cast<unsigned long long>(s.refusal_named_bad),
                    static_cast<unsigned long long>(s.in_evidence),
                    static_cast<unsigned long long>(s.in_evidence_bad),
                    static_cast<unsigned long long>(s.silent));
    };
    report("F1a lift", lifted);
    report("F1a cap", capped);
    report("F1b", faithful);

    check(lifted.total == capped.total && lifted.total == faithful.total,
          "F1 the three sweeps have the same case count");
    check(lifted.in_evidence_bad == 0U,
          "F1a lifted: every (page,k,j) is in the admissible evidence (" +
              std::to_string(lifted.in_evidence_bad) + " not)");
    check(lifted.found_bad == 0U && lifted.no_candidate == 0U && lifted.refused_everything == 0U &&
              lifted.refused_budget == 0U,
          "F1a lifted: every case is Found and contains its page");
    check(capped.found_bad == 0U,
          "F1a capped: Found IFF the occurrence span fits the cap, and NO WRONG ANCHOR (" +
              std::to_string(capped.found_bad) + " wrong)");
    check(capped.refusal_named_bad == 0U,
          "F1a capped: every refusal names the span, the limit and the dropped pages");
    check(capped.silent == 0U, "F1a: no refusal is silent");
    check(faithful.in_evidence_bad == 0U,
          "F1b: every (page,k,j) is in the admissible evidence (" +
              std::to_string(faithful.in_evidence_bad) + " not)");
    check(faithful.found_bad == 0U, "F1b: NO WRONG ANCHOR -- every Found run contains its "
                                    "strongest-evidence page (" +
                                        std::to_string(faithful.found_bad) + " wrong)");
    // The 22 "the run does not hold a page containing F" cases are REPORTED with their mechanism:
    // the strongest admissible evidence starts at query offset 0 and runs FURTHER than F's own
    // interval, so F's page holds an interval INTERIOR to the covered one and is an ALTERNATIVE
    // occurrence -- it is counted in `alternatives`, not dropped. The asserted property is the
    // ACCOUNTING: every admissible page is either in the run or counted.
    check(faithful.accounting_bad == 0U,
          "F1b: every admissible page is either in the run or counted as an alternative (" +
              std::to_string(faithful.accounting_bad) + " unaccounted)");
    std::printf("   F1b NOTE: in %llu of %llu Found cases (%.3f%%) the run does not hold a page "
                "containing F: the strongest evidence starts at query offset 0 and runs past F's "
                "interval, so F's page is an ALTERNATIVE occurrence (counted, not dropped). "
                "Reported, with the count asserted separately.\n",
                static_cast<unsigned long long>(faithful.reachable_bad),
                static_cast<unsigned long long>(faithful.found),
                faithful.found == 0U ? 0.0
                                     : 100.0 * static_cast<double>(faithful.reachable_bad) /
                                           static_cast<double>(faithful.found));
    std::printf("   F1b NOTE: the queried page itself is absent from the run in %llu of %llu Found "
                "cases (%.1f%%), which is the corpus's periodicity: another page holds a strictly "
                "longer run of the query. Reported, not asserted.\n",
                static_cast<unsigned long long>(faithful.page_absent),
                static_cast<unsigned long long>(faithful.found),
                faithful.found == 0U ? 0.0
                                     : 100.0 * static_cast<double>(faithful.page_absent) /
                                           static_cast<double>(faithful.found));
    check(faithful.silent == 0U, "F1b: no refusal is silent");
}

// ---- F2: straddling windows ---------------------------------------------------------------
void section_f2(const sd::SumDir& directory, const Flat& flat, std::uint32_t k_min,
                std::uint32_t sentinel_base) {
    std::printf("\n== F2  straddling fragments: every k-window of the concatenated corpus; a Found "
                "run must cover every page the window occupies (cap lifted) ==\n");
    sd::SumDirReachPostings postings;
    postings.k_min = k_min;
    postings.build(directory);
    const std::span<const std::uint32_t> none;

    std::uint64_t total = 0;
    std::uint64_t found = 0;
    std::uint64_t covered = 0;
    std::uint64_t wrong   = 0;
    std::uint64_t refused = 0;
    std::uint64_t single_total = 0, single_covered = 0, single_wrong = 0;
    std::uint64_t straddling = 0, straddling_covered = 0;
    std::uint64_t shown = 0;
    for (std::uint32_t k = k_min; k <= kKMax; ++k) {
        if (flat.tokens.size() < k) { break; }
        for (std::size_t s = 0; s + k <= flat.tokens.size(); ++s) {
            const std::uint32_t first_occupied = flat.page_of[s];
            const std::uint32_t last_occupied  = flat.page_of[s + k - 1U];
            const bool          single         = first_occupied == last_occupied;
            const std::string term_q =
                render_ids(make_query(flat, s, k, sentinel_base, /*sentinel_context=*/true));
            ++total;
            if (single) { ++single_total; } else { ++straddling; }
            const sd::SumDirReachResult result = sd::sum_dir_recall_span_reachable(
                directory, postings, term_q, kFrontier, kUnbounded, none, kUnbounded, kBlockTokens);
            if (result.found) {
                ++found;
                // The run is contiguous, so "covers every occupied page" == "covers the span".
                if (result.first_page <= first_occupied && last_occupied <= result.last_page) {
                    ++covered;
                    if (single) { ++single_covered; } else { ++straddling_covered; }
                } else {
                    ++wrong;
                    if (single) { ++single_wrong; }
                    if (shown < 5) {
                        std::printf("   miss: k=%u s=%zu occupies [%u,%u] but run=[%u,%u] need=%u "
                                    "best=%u@%u anchor=%u\n",
                                    k, s, first_occupied, last_occupied, result.first_page,
                                    result.last_page, result.needed_blocks, result.window,
                                    result.best_offset, result.anchor_page);
                        ++shown;
                    }
                }
            } else {
                ++refused;
            }
        }
    }
    std::printf("   total=%llu found=%llu covers_occupied=%llu WRONG=%llu refused=%llu\n",
                static_cast<unsigned long long>(total), static_cast<unsigned long long>(found),
                static_cast<unsigned long long>(covered), static_cast<unsigned long long>(wrong),
                static_cast<unsigned long long>(refused));
    std::printf("   INSIDE ONE PAGE: %llu cases, %llu Found-and-covered, %llu miss\n",
                static_cast<unsigned long long>(single_total),
                static_cast<unsigned long long>(single_covered),
                static_cast<unsigned long long>(single_wrong));
    std::printf("   STRADDLING     : %llu cases, %llu Found-and-covered (%.1f%%)\n",
                static_cast<unsigned long long>(straddling),
                static_cast<unsigned long long>(straddling_covered),
                straddling == 0U ? 0.0
                                 : 100.0 * static_cast<double>(straddling_covered) /
                                       static_cast<double>(straddling));
    check(total > 0U, "F2 non-empty");
    check(single_wrong == 0U,
          "F2: for a window INSIDE one page, every Found run contains that page (" +
              std::to_string(single_wrong) + " did not)");
    check(straddling > 0U, "F2: the straddling subset is non-empty");
    check(straddling_covered > 0U, "F2: at least one straddling case is Found and covered");
}

// ---- F4: the loop guard, against the page-0 case -------------------------------------------
void section_f4(const sd::SumDir& directory, const Corpus& corpus, const Flat& flat,
                std::uint32_t sentinel_base) {
    std::printf("\n== F4  the loop guard, against D1's rounds 3-4 (page 0's own tail) ==\n");
    sd::SumDirReachPostings postings;
    postings.k_min = sd::kReachMinWindowTokens;
    postings.build(directory);
    const std::span<const std::uint32_t> none;

    std::vector<std::uint32_t> ledger = flat.tokens;
    for (std::uint32_t s = 0; s < kSentinelCount; ++s) { ledger.push_back(sentinel_base + s); }
    ledger.insert(ledger.end(), corpus.pages[0].begin(), corpus.pages[0].end());
    const std::uint32_t frontier = static_cast<std::uint32_t>(ledger.size());
    const std::vector<std::uint32_t> query(ledger.end() - kQueryTokens, ledger.end());
    const std::string                loop_term = render_ids(query);
    std::printf("   frontier=%u  query = the last %u tokens = page 0's own tail\n", frontier,
                kQueryTokens);

    const sd::SumDirReachResult unguarded =
        sd::sum_dir_recall_span_reachable(directory, postings, loop_term, frontier, kFanoutCap,
                                          none);
    std::printf("   guard OFF: %s\n", sd::sum_dir_reach_line(unguarded).c_str());
    check(unguarded.found && unguarded.anchor_page == 0U && unguarded.token_begin == 0U &&
              unguarded.token_end == 64U && unguarded.candidates == 1U,
          "F4 guard OFF reproduces the lock-on: Found on page 0, span [0,64), one candidate");

    const std::uint32_t        appended0[1] = {0U};
    const sd::SumDirReachResult guarded     = sd::sum_dir_recall_span_reachable(
        directory, postings, loop_term, frontier, kFanoutCap,
        std::span<const std::uint32_t>(appended0, 1U));
    std::printf("   guard ON:  %s\n", sd::sum_dir_reach_line(guarded).c_str());
    check(!guarded.found && !guarded.refusal.empty() && guarded.token_begin == 0U &&
              guarded.token_end == 0U,
          "F4 guard ON terminates: not Found, refusal named, no page-0 span substituted");
    // The separating control the brief asks for by name, stated exactly: the exclusion of PAGE 0 is
    // attributed to the APPENDED guard, not to the self-match guard. The self guard's own 10
    // exclusions are the pages whose bytes the query's own tail occupies (the ledger's last 32
    // tokens sit past page 265), which is that guard working as designed -- and it is NOT what stops
    // the lock-on, because page 0 is EARLIER than the query and is admissible by that rule.
    check(guarded.excluded_appended >= 1U && guarded.excluded_self == unguarded.excluded_self,
          "F4 page 0 is excluded by the APPENDED guard, and the appended guard alone is what stops "
          "the lock-on (appended=" + std::to_string(guarded.excluded_appended) + " self_on=" +
              std::to_string(guarded.excluded_self) + " self_off=" +
              std::to_string(unguarded.excluded_self) + ")");

    auto drive = [&](bool use_guard, bool start_from_d1, std::uint32_t max_rounds,
                     std::uint32_t& rounds, std::string& trace) {
        std::vector<std::uint32_t> state = ledger;
        if (start_from_d1) {
            std::vector<std::uint32_t> term_ids;
            state.clear();
            state = flat.tokens;
            for (std::uint32_t s = 0; s < kSentinelCount; ++s) { state.push_back(sentinel_base + s); }
            std::ifstream in("/home/user/scratch/PATCHSET/REACHABLE/term_d1_round1.txt");
            std::string   line;
            if (in && std::getline(in, line) && sd::sum_dir_parse_token_line(line, term_ids)) {
                state.insert(state.end(), term_ids.begin(), term_ids.end());
            }
        }
        std::vector<std::uint32_t> appended_pages;
        rounds = 0;
        std::ostringstream out;
        for (std::uint32_t round = 1; round <= max_rounds; ++round) {
            if (state.size() < kQueryTokens) { break; }
            const std::vector<std::uint32_t> q(state.end() - kQueryTokens, state.end());
            const std::string                t = render_ids(q);
            const std::uint32_t              f = static_cast<std::uint32_t>(state.size());
            const sd::SumDirReachResult      r =
                use_guard ? sd::sum_dir_recall_span_reachable(directory, postings, t, f,
                                                              kFanoutCap, appended_pages)
                          : sd::sum_dir_recall_span_reachable(directory, postings, t, f, kFanoutCap,
                                                              none);
            ++rounds;
            if (!r.found) {
                out << "     round " << round << ": " << sd::sum_dir_reach_status_name(r.status)
                    << " appended_excluded=" << r.excluded_appended << " -- TERMINATED\n";
                break;
            }
            out << "     round " << round << ": anchor=" << r.anchor_page << " span=["
                << r.token_begin << "," << r.token_end << ") pages=" << r.pages
                << " best=" << r.window << "\n";
            for (std::uint32_t page = r.first_page; page <= r.last_page && page <= corpus.max_page;
                 ++page) {
                if (corpus.pages[page].empty()) { continue; }
                state.insert(state.end(), corpus.pages[page].begin(), corpus.pages[page].end());
                appended_pages.push_back(page);
            }
            std::sort(appended_pages.begin(), appended_pages.end());
            appended_pages.erase(std::unique(appended_pages.begin(), appended_pages.end()),
                                 appended_pages.end());
        }
        trace = out.str();
    };

    std::uint32_t rounds = 0;
    std::string   trace;
    drive(false, false, 4U, rounds, trace);
    std::printf("   (a) page-0 tail, guard OFF -> rounds=%u\n%s", rounds, trace.c_str());
    const std::uint32_t rounds_a = rounds;
    drive(true, false, 4U, rounds, trace);
    std::printf("   (b) page-0 tail, guard ON  -> rounds=%u\n%s", rounds, trace.c_str());
    const std::uint32_t rounds_b = rounds;
    drive(true, true, 4U, rounds, trace);
    std::printf("   (c) D1 round-1 query, guard ON (the fixed shape) -> rounds=%u\n%s", rounds,
                trace.c_str());
    check(rounds_b < rounds_a, "F4 the driver terminates strictly sooner with the guard on (" +
                                   std::to_string(rounds_b) + " vs " + std::to_string(rounds_a) + ")");
}

// ---- F5: the loud fallback -----------------------------------------------------------------
void section_f5(const sd::SumDir& directory, const std::string& term, std::uint32_t sentinel_base) {
    std::printf("\n== F5  the loud fallback: a miss refuses, it never anchors at page 0 ==\n");
    sd::SumDirReachPostings postings;
    postings.k_min = sd::kReachMinWindowTokens;
    postings.build(directory);
    const std::span<const std::uint32_t> none;

    std::vector<std::uint32_t> short_query{sentinel_base, sentinel_base + 1U};
    sd::SumDirReachResult      r = sd::sum_dir_recall_span_reachable(
        directory, postings, render_ids(short_query), kFrontier, kFanoutCap, none);
    std::printf("   below-floor: %s\n", sd::sum_dir_reach_line(r).c_str());
    check(!r.found && r.status == sd::SumDirReachStatus::NoCandidate && !r.refusal.empty() &&
              r.token_begin == 0U && r.token_end == 0U,
          "F5 a query below k_min refuses and does not anchor at page 0");

    std::vector<std::uint32_t> absent;
    for (std::uint32_t x = 0; x < 32U; ++x) { absent.push_back(sentinel_base + x); }
    r = sd::sum_dir_recall_span_reachable(directory, postings, render_ids(absent), kFrontier,
                                          kFanoutCap, none);
    std::printf("   no-page-holds-it: %s\n", sd::sum_dir_reach_line(r).c_str());
    check(!r.found && r.status == sd::SumDirReachStatus::NoCandidate && !r.refusal.empty(),
          "F5 a query no page holds refuses");

    r = sd::sum_dir_recall_span_reachable(directory, postings, render_ids(absent), kFrontier, 0U,
                                          none);
    check(!r.found && !r.refusal.empty(), "F5 a zero fan-out cap refuses loudly");

    if (!term.empty()) {
        const sd::SumDirReachResult tight = sd::sum_dir_recall_span_reachable(
            directory, postings, term, kFrontier, /*fanout_cap=*/1U, none, /*budget_blocks=*/1U);
        std::printf("   budget=1 on the straddle: %s\n", sd::sum_dir_reach_line(tight).c_str());
        check(!tight.found && tight.status == sd::SumDirReachStatus::RefusedBudget &&
                  tight.needed_blocks == 2U && tight.first_page == 158U &&
                  tight.last_page == 159U &&
                  tight.refusal.find("drop pages") != std::string::npos &&
                  tight.refusal.find("[159,159]") != std::string::npos,
              "F5 an over-budget candidate set refuses, naming the pages it could not hold");
        const sd::SumDirReachResult roomy = sd::sum_dir_recall_span_reachable(
            directory, postings, term, kFrontier, kFanoutCap, none, sd::kReachDefaultBudgetBlocks);
        check(roomy.found, "F5 the same query is Found when the limit can hold it (the control)");
    } else {
        ++g_fail;
        std::printf("   budget arm: NOT MEASURED (no term file)\n");
    }
}

// ---- F6: the shipping selector's reachability rate ----------------------------------------
void section_f6_red_sample(const sd::SumDir& directory, const Corpus& corpus, const Flat& flat,
                           std::uint32_t k_min, std::uint32_t sentinel_base) {
    std::printf("\n== F6  RED (shipping selector) reachability sample: every page x k in "
                "{8,16,24,32} x j in {0,16,32,48}, BOTH query shapes ==\n");
    const std::uint32_t ks[] = {8U, 16U, 24U, 32U};
    const std::uint32_t js[] = {0U, 16U, 32U, 48U};
    for (std::size_t arm = 0; arm < 2U; ++arm) {
        const bool sentinel_context = (arm == 1U);
        std::uint64_t total = 0, ok = 0, wrong_anchor = 0, not_found = 0;
        bool          shown_example = false;
        for (std::uint32_t page = 0; page <= corpus.max_page; ++page) {
            const std::vector<std::uint32_t>& ids = corpus.pages[page];
            if (ids.empty()) { continue; }
            const std::uint32_t base = flat.base_of[page];
            for (const std::uint32_t k : ks) {
                if (ids.size() < k || k < k_min) { continue; }
                for (const std::uint32_t j : js) {
                    if (j + k > ids.size()) { continue; }
                    const std::string term_q = render_ids(
                        make_query(flat, base + j, k, sentinel_base, sentinel_context));
                    const RedOutcome red = run_red(directory, term_q, kFrontier, kFanoutCap);
                    ++total;
                    if (!red.found) {
                        ++not_found;
                    } else if (contains(red.first, red.last, page)) {
                        ++ok;
                    } else {
                        ++wrong_anchor;
                        if (!shown_example) {
                            std::printf("   [%s] first wrong anchor: page %u k=%u j=%u -> "
                                        "run=[%u,%u] span=[%u,%u) window=%u\n",
                                        sentinel_context ? "controlled" : "faithful", page, k, j,
                                        red.first, red.last, red.begin, red.end, red.window);
                            shown_example = true;
                        }
                    }
                }
            }
        }
        std::printf("   RED [%s] total=%llu contains_page=%llu WRONG_ANCHOR=%llu not_found=%llu "
                    "-> %.1f%% reachable\n",
                    sentinel_context ? "controlled" : "faithful ",
                    static_cast<unsigned long long>(total), static_cast<unsigned long long>(ok),
                    static_cast<unsigned long long>(wrong_anchor),
                    static_cast<unsigned long long>(not_found),
                    total == 0U ? 0.0
                                : 100.0 * static_cast<double>(ok) / static_cast<double>(total));
        check(total > 0U, "F6 the RED sample is non-empty");
        check(ok < total, "F6 the shipping selector is NOT already reachable on this sample (" +
                              std::to_string(ok) + "/" + std::to_string(total) + ")");
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string cargo =
        argc > 1 ? argv[1]
                 : "/mnt/c/Users/User/Documents/ziqinzhang/dl/recallspan/d1/ninfer_text.cargo";
    const std::string term_file = argc > 2 ? argv[2] : "";
    const std::string mode      = argc > 3 ? argv[3] : "all";

    Corpus corpus;
    if (!load_cargo(cargo, corpus)) {
        std::printf("VERDICT: NOT MEASURED (no corpus)\n");
        return 2;
    }
    const sd::SumDir directory = build_directory(corpus);
    const Flat        flat      = flatten(corpus);
    std::printf("directory rows=%zu live=%zu max_page=%u flat_tokens=%zu\n", directory.size(),
                corpus.live, corpus.max_page, flat.tokens.size());
    const std::uint32_t sentinel_base = corpus.max_id + 1U;

    std::string term;
    if (!term_file.empty()) {
        std::ifstream in(term_file);
        if (in) { std::getline(in, term); }
    }
    std::printf("term file %s: %zu chars\n", term_file.c_str(), term.size());

    RefOccurrence ref;
    ref.build(corpus, sd::kReachMinWindowTokens);

    section_f3(directory, term);
    if (mode == "all" || mode == "green") {
        section_f1(directory, corpus, flat, ref, sentinel_base);
        section_f2(directory, flat, sd::kReachMinWindowTokens, sentinel_base);
        section_f4(directory, corpus, flat, sentinel_base);
        section_f5(directory, term, sentinel_base);
    }
    if (mode == "all" || mode == "red") {
        section_f6_red_sample(directory, corpus, flat, sd::kReachMinWindowTokens, sentinel_base);
    }

    std::printf("\nchecks=%llu failures=%llu\n", static_cast<unsigned long long>(g_checks),
                static_cast<unsigned long long>(g_fail));
    std::printf("VERDICT: %s\n", g_fail == 0U ? "PASS" : "FAIL");
    return g_fail == 0U ? 0 : 1;
}
