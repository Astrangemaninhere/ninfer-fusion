// tests/test_semchan_symbol_key.cpp -- THE SEMANTIC CANDIDATE CHANNEL, EXERCISED AGAINST THE REAL
// ARBITER, ON A CORPUS IT IS BUILT TO GET WRONG.
//
//   ninfer_semchan_symbol_key_test [mode]        mode: all | green | red   (default: all)
//
// HOST-ONLY, STD-ONLY, NO ninfer LIBRARY. What it includes is `src/spec/sum_dir.h`,
// `src/spec/sum_dir_reach.h` (the ARBITER, the same header the engine compiles into
// program_impl.h) and `src/spec/semchan_symbol_key.h` (the channel under test). Neither header has
// one occurrence of "cuda", so this target needs no CUDA toolchain -- the same property
// `tests/test_sum_dir.cpp` and `tests/test_turn_recall_journal.cpp` require.
//
// WHY THE CORPUS IS ADVERSARIAL AND NOT FRIENDLY
// ----------------------------------------------
// A channel that is green on a friendly corpus proves nothing: `fusionr3`'s own note (DEFECTS.md,
// F480 sec.5) refuses "'没改动也能绿' 不许当判据". So the corpus below is BUILT so that the semantic
// channel's TOP-1 IS WRONG:
//
//   * page 2 is TRUE: the query's 16 ids verbatim, at an interior offset, so the lexical arbiter
//     finds a 16-token exact run there;
//   * page 3 is a DISTRACTOR: it holds 32 copies of the query's most common symbol and 4 of its 8
//     rare ones, arranged so that NO 4-gram of the query occurs anywhere in it. Its mean-key score
//     is 0.25390625 against TRUE's 0.0703125, so the CHANNEL RANKS IT FIRST -- which is precisely
//     the Mean-K failure mode KVMem's own KVMI-007 names ("mean-k 会稀释和碎片化 ... 相关证据").
//
// The arm is GREEN only if BOTH of these hold SIMULTANEOUSLY:
//   (a) the channel put the TRUE page INTO its candidate set anyway (a source narrows; it does not
//       have to be right at rank 1), and
//   (b) the ARBITER's own fields anchored on the TRUE page, and the union carried that anchor
//       through unchanged.
// That is the order's "该通道答错但仲裁者能纠正" case, and it is one arm, not two.
//
// THE PROVENANCE ARMS (the second criterion) are sections P1/P2. In P1 two rows sit on the SAME
// page -- the same position frame -- and carry DIFFERENT source spans. The channel must return TWO
// records and the union must hand the arbiter BOTH origins; if it merges them, P1's named check
// fires. In P2 the same TEXT appears at two origins; the channel must not dedupe by identity, and
// the arbiter -- which is allowed to choose -- anchors one of them, after which the origin of the
// chosen page is UNIQUE and readable.
//
// WHAT THIS FILE DELIBERATELY DOES NOT DO
// ---------------------------------------
// It does not claim that the engine consumes provenance. `install_index_recall_provider`
// (program_impl.h:15985) is not modified by this landing, and REPORT.md sec.5 says why by name.
// What is proved here is the CHANNEL-side property: neither the channel nor the union can drop an
// origin, and neither can produce a determination.

#include "spec/semchan_symbol_key.h"
#include "spec/sum_dir.h"
#include "spec/sum_dir_reach.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace sd = ninfer::spec::sum_dir;
namespace sc = ninfer::spec::semchan;

namespace {

int g_checks   = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    if (!condition) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++g_failures;
    }
}

// The value-printing form is only valid for arithmetic operands, and the test compares STRINGS and
// DIGESTS too. `if constexpr` keeps one call shape while refusing to cast a digest to an integer --
// a "print it anyway" cast is how a comparison of two different things starts to look like a pass.
template <typename A, typename B>
void check_equal(const A& got, const B& want, const char* what) {
    ++g_checks;
    if (!(got == want)) {
        if constexpr (std::is_arithmetic_v<A> && std::is_arithmetic_v<B>) {
            std::fprintf(stderr, "FAIL %s (got=%lld want=%lld)\n", what,
                         static_cast<long long>(got), static_cast<long long>(want));
        } else {
            std::fprintf(stderr, "FAIL %s (values are not arithmetic: printed by the caller)\n",
                         what);
        }
        ++g_failures;
    }
}

void check_near(double got, double want, double tolerance, const char* what) {
    ++g_checks;
    const double delta = got > want ? got - want : want - got;
    if (!(delta <= tolerance)) {
        std::fprintf(stderr, "FAIL %s (got=%.9f want=%.9f delta=%.9f)\n", what, got, want, delta);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// the corpus
// ---------------------------------------------------------------------------

constexpr std::uint32_t kBlockTokens = 64U;

// THE QUERY'S ALPHABET IS DISJOINT FROM EVERY FILLER RANGE, which is what makes the tallies below
// exact rather than statistical: a page that does not hold a query symbol scores EXACTLY zero, so
// `positive` is a counted set and not a threshold.
constexpr std::uint32_t kX = 9001U; // the query's most common symbol
const std::uint32_t kRare = 9101U;  // 9101..9104 appear in both TRUE and the DISTRACTOR
const std::uint32_t kRareOnly = 9005U; // 9005..9008 appear in TRUE only

[[nodiscard]] std::vector<std::uint32_t> the_query() {
    return {kX,          9101U,       kX,     9102U, kX,     9103U,
            kX,          9104U,       kX,     kRareOnly, kX, 9006U,
            kX,          9007U,       kX,     9008U};
}

[[nodiscard]] std::vector<std::uint32_t> filler(std::uint32_t base, std::size_t count) {
    std::vector<std::uint32_t> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) { out.push_back(base + static_cast<std::uint32_t>(i)); }
    return out;
}

// page 2: THE QUERY VERBATIM at offset 24, padded to the block granularity. The padding is what
// makes this a MEAN over 64 tokens and not over 16 -- it is the dilution the channel is built to
// suffer, included on purpose.
[[nodiscard]] std::vector<std::uint32_t> page_true(const std::vector<std::uint32_t>& query) {
    std::vector<std::uint32_t> tokens = filler(2000U, 24U);
    tokens.insert(tokens.end(), query.begin(), query.end());
    const std::vector<std::uint32_t> tail = filler(2100U, 24U);
    tokens.insert(tokens.end(), tail.begin(), tail.end());
    return tokens;
}

// page 3: THE DISTRACTOR. 32 copies of the query's commonest symbol, 4 of its 8 rare symbols, and
// NO 4-gram of the query anywhere -- so the arbiter cannot reach it and the channel cannot resist
// it. The rare symbols are interleaved with symbols that are not in the query, which is what keeps
// every 4-window off the query's gram list.
[[nodiscard]] std::vector<std::uint32_t> page_distractor() {
    std::vector<std::uint32_t> tokens = filler(2300U, 24U);
    for (int i = 0; i < 32; ++i) { tokens.push_back(kX); }
    const std::uint32_t tail[8] = {9102U, 9201U, 9101U, 9202U, 9104U, 9203U, 9103U, 9204U};
    for (const std::uint32_t t : tail) { tokens.push_back(t); }
    return tokens;
}

// One row per page, catalogue line rendered by the tree's OWN producer -- the same call the
// acceptance uses (`tests/test_sum_dir_reach.cpp:46`), so the grams the arbiter searches and the
// grams the channel buckets come from one spelling of the text and cannot drift apart.
struct RowSpec {
    std::vector<std::uint32_t> tokens;
    std::uint32_t              page      = 0;
    std::uint32_t              token_begin = 0;
    std::uint32_t              generation = 1;
};

[[nodiscard]] sd::SumDir build_directory(const std::vector<RowSpec>& specs) {
    sd::SumDir directory(/*sequence_tag=*/1U, kBlockTokens);
    for (const RowSpec& spec : specs) {
        const std::size_t row =
            directory.append(spec.tokens, spec.token_begin, spec.generation, sd::SumDirCodec::Int8,
                             /*file_slot=*/static_cast<std::int32_t>(spec.page), spec.page);
        directory.set_summary(row, sd::sum_dir_render_token_line(spec.tokens));
    }
    directory.sort_rows();
    return directory;
}

[[nodiscard]] std::vector<RowSpec> main_specs(const std::vector<std::uint32_t>& query) {
    return {
        RowSpec{filler(2600U, 64U), 0U, 0U, 1U},
        RowSpec{filler(2700U, 64U), 1U, 64U, 1U},
        RowSpec{page_true(query), 2U, 128U, 1U},
        RowSpec{page_distractor(), 3U, 192U, 1U},
    };
}

// ---------------------------------------------------------------------------
// the arbiter, driven through its own header
// ---------------------------------------------------------------------------

struct Arbiter {
    sd::SumDirReachResult result;
    // `SumDirReachResult` carries `pages` as a COUNT (`std::uint32_t pages`), not as a list: the
    // selector's answer is a CONTIGUOUS token span (`token_begin`, `token_end`) and the pages it
    // covers are `[first_page, last_page]`. The list below is built from THOSE fields and from
    // nothing else, so the union is fed exactly what the engine's own selector reports.
    std::vector<std::uint32_t> pages;
};

// THE REAL SELECTOR, over the REAL directory. `appended_pages` is empty: nothing has been recalled
// yet, so the loop guard excludes nothing and the arm measures the selector and not the guard.
[[nodiscard]] Arbiter run_arbiter(const sd::SumDir& directory,
                                  const std::vector<std::uint32_t>& query, std::uint32_t fanout) {
    const std::string term = sd::sum_dir_render_token_line(query);
    const std::uint32_t frontier = kBlockTokens * 8U;
    const std::vector<std::uint32_t> appended;
    Arbiter out;
    out.result = sd::sum_dir_recall_span_reachable(
        directory, term, frontier, fanout,
        std::span<const std::uint32_t>(appended.data(), appended.size()));
    if (out.result.found) {
        for (std::uint32_t page = out.result.first_page; page <= out.result.last_page; ++page) {
            out.pages.push_back(page);
        }
    }
    return out;
}

[[nodiscard]] std::string field_of(const std::string& line, const std::string& key) {
    const std::string needle = key + "=";
    const std::size_t at = line.find(needle);
    if (at == std::string::npos) { return "<absent>"; }
    const std::size_t begin = at + needle.size();
    std::size_t end = begin;
    while (end < line.size() && line[end] != ' ' && line[end] != '\n') { ++end; }
    return line.substr(begin, end - begin);
}

// HOW MANY RECORDS SCORED POSITIVE, read from the channel's own counter rather than guessed. Used
// by the refusal arms, where the count IS the reason the refusal fired.
[[nodiscard]] std::uint32_t arm_positive(const std::vector<RowSpec>& specs,
                                         const std::vector<std::uint32_t>& query) {
    sd::SumDir directory = build_directory(specs);
    const std::vector<sc::SemChanBlockEntry> index = sc::semchan_index_blocks(directory);
    return sc::semchan_candidate_pages(index, sc::semchan_symbol_key_from_ids(query), 0U).positive;
}

// ---------------------------------------------------------------------------
// S0: the names, the refusals, the constants
// ---------------------------------------------------------------------------

void section0_names_and_refusals() {
    std::printf("---- S0 the named scorers, and the one that is implemented ----\n");
    const sc::SemChanScorer names[5] = {
        sc::SemChanScorer::MeanK, sc::SemChanScorer::PerToken, sc::SemChanScorer::SubBlockMeanK,
        sc::SemChanScorer::KeyDirectionFixed4, sc::SemChanScorer::KeyDirectionAdaptive};
    // THE FIVE NAMES ARE KVMem'S, VERBATIM. Checked by string, so a rename in this header cannot
    // silently stop naming what KVMem names -- and a READER of the report can match the two lists.
    const char* expected[5] = {"mean-k", "per-token", "sub-block-mean-k", "key-direction-fixed4",
                               "key-direction-adaptive"};
    int implemented = 0;
    for (int i = 0; i < 5; ++i) {
        check_equal(std::string(sc::semchan_scorer_name(names[i])), std::string(expected[i]),
                    "the declared scorer names are KVMem's own five, verbatim");
        for (int j = 0; j < 5; ++j) { check(names[i] != names[j] || i == j, "scorer names are distinct"); }
        if (sc::semchan_scorer_is_implemented(names[i])) { ++implemented; }
        check(std::strlen(sc::semchan_scorer_reason(names[i])) > 0,
              "every scorer has a reason, admitted or refused");
    }
    check_equal(implemented, 1, "exactly ONE of the five is implemented");
    check(sc::semchan_scorer_is_implemented(sc::SemChanScorer::MeanK), "and it is mean-k");
    check(!sc::kSemChanMayDetermine, "the channel carries the 'may not determine' constant");
    check(!sc::semchan_origin_may_be_chosen_by_channel(),
          "the channel may not choose between origins either");
    check_equal(static_cast<unsigned>(sc::kSemChanMaxBlockTokens), 64U,
                "the channel's block is the directory's 64-token page");
    check(sc::kSemChanOriginPolicy != sc::SemChanOriginPolicy::Unnamed,
          "the origin policy is NAMED (the house idiom: Unnamed is a disagreement left silent)");
}

// ---------------------------------------------------------------------------
// S1: the producer -- one mean, by counting
// ---------------------------------------------------------------------------

void section1_producer() {
    std::printf("---- S1 the symbol-key producer ----\n");
    sc::SemChanKeyReport report;
    const std::vector<std::uint32_t> ids = {7U, 7U, 7U, 9U};
    const sc::SemChanSymbolKey key = sc::semchan_symbol_key_from_ids(ids, &report);
    check(report.on_axis, "a 4-id span is on the axis");
    check_equal(key.tokens, 4U, "the key reports the count it was derived at");
    check_equal(key.symbols.size(), 2U, "distinct symbols collapse to two buckets");
    check_near(key.weight_of(7U), 0.75, 1e-15, "the mean of three-of-four is 0.75");
    check_near(key.weight_of(9U), 0.25, 1e-15, "and one-of-four is 0.25");
    check_near(key.weight_of(11U), 0.0, 1e-15, "a symbol the block does not hold weighs 0");
    double total = 0.0;
    for (const double w : key.weights) { total += w; }
    check_near(total, 1.0, 1e-12, "the weights are L1-normalized, so lengths are comparable");

    // EMPTY AND OVER-BLOCK ARE REFUSED BY NAME, not clamped: a silently truncated 65-id span would
    // be a block that is not the block, which is the one thing a mean over it must not absorb.
    sc::SemChanKeyReport empty;
    const sc::SemChanSymbolKey none = sc::semchan_symbol_key_from_ids(nullptr, 0U, &empty);
    check(empty.refused_empty_span, "an empty span is refused by name");
    check(none.is_unset(), "and it is NOT a key");
    std::vector<std::uint32_t> over(65U, 1U);
    sc::SemChanKeyReport over_report;
    const sc::SemChanSymbolKey clamped = sc::semchan_symbol_key_from_ids(over, &over_report);
    check(over_report.refused_over_block, "a 65-id span is refused by name");
    check(clamped.is_unset(), "and it is NOT a key either -- refused, never truncated");

    // ORDER IS INVISIBLE TO THE KEY. This is the channel's own admission, measured rather than
    // asserted, and it is the property the whole "candidate only" discipline rests on.
    const std::vector<std::uint32_t> a = {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U};
    std::vector<std::uint32_t> b = a;
    std::reverse(b.begin(), b.end());
    const sc::SemChanSymbolKey ka = sc::semchan_symbol_key_from_ids(a);
    const sc::SemChanSymbolKey kb = sc::semchan_symbol_key_from_ids(b);
    check_near(sc::semchan_mean_k_score(ka, ka), sc::semchan_mean_k_score(ka, kb), 0.0,
               "a permutation of a block scores IDENTICALLY against the same query");
    check(ka.symbols == kb.symbols, "and its symbol vector is the same vector");
}

// ---------------------------------------------------------------------------
// S2: the channel's own fields, on the adversarial corpus
// ---------------------------------------------------------------------------

struct Arm {
    sd::SumDir directory;
    std::vector<sc::SemChanBlockEntry> index;
    sc::SemChanIndexReport index_report;
    sc::SemChanCandidates candidates;
    Arbiter arbiter;
    sc::SemChanUnion union_set;
};

[[nodiscard]] Arm run_arm(const std::vector<RowSpec>& specs,
                          const std::vector<std::uint32_t>& query, std::uint32_t fanout,
                          std::uint32_t cap) {
    Arm arm;
    arm.directory = build_directory(specs);
    arm.index = sc::semchan_index_blocks(arm.directory, &arm.index_report);
    const sc::SemChanSymbolKey query_key = sc::semchan_symbol_key_from_ids(query);
    arm.candidates = sc::semchan_candidate_pages(arm.index, query_key, fanout);
    arm.arbiter = run_arbiter(arm.directory, query, fanout);
    arm.union_set = sc::semchan_union_candidates(arm.arbiter.pages, arm.arbiter.result.anchor_page,
                                                 arm.arbiter.result.found, arm.candidates, cap);
    return arm;
}

// ---------------------------------------------------------------------------
// S3: THE ARM THE ORDER ASKS FOR -- the channel is wrong at rank 1, the arbiter corrects it
// ---------------------------------------------------------------------------

void section3_channel_wrong_arbiter_right() {
    std::printf("---- S3 the channel's rank 1 is WRONG and the arbiter's anchor is RIGHT ----\n");
    const std::vector<std::uint32_t> query = the_query();
    const Arm arm = run_arm(main_specs(query), query, /*fanout=*/2U, /*cap=*/0U);

    std::printf("     index: rows_total=%u indexed=%u not_live=%u empty=%u over_block=%u "
                "no_origin=%u same_frame_distinct_origin=%u\n",
                arm.index_report.rows_total, arm.index_report.rows_indexed,
                arm.index_report.rows_not_live, arm.index_report.rows_empty,
                arm.index_report.rows_over_block, arm.index_report.rows_no_origin,
                arm.index_report.same_frame_distinct_origin);
    std::printf("     %s\n", sc::semchan_candidates_line(arm.candidates).c_str());
    std::printf("     %s\n", sc::semchan_union_line(arm.union_set).c_str());
    std::printf("     arbiter: %s\n", sd::sum_dir_reach_line(arm.arbiter.result).c_str());

    check_equal(arm.index_report.rows_indexed, 4U, "four Live rows indexed");
    check_equal(arm.index_report.rows_not_live, 0U, "no row was skipped on the byte axis");

    // (1) THE CHANNEL IS WRONG AT RANK ONE. Not a hope: page 3's mean-key score is strictly greater
    // than page 2's, because the distractor holds 50% of the query's commonest symbol against
    // TRUE's 12.5%. If this ever stops being true the arm stops being adversarial, so it is
    // CHECKED rather than assumed.
    check_equal(arm.candidates.candidates.size(), 2U, "exactly two records scored positive");
    check_equal(arm.candidates.candidates[0].page, 3U,
                "the channel's RANK 1 is the DISTRACTOR (page 3) -- the channel is wrong here");
    check_equal(arm.candidates.candidates[1].page, 2U, "and the TRUE page is rank 2");
    check(arm.candidates.candidates[0].score > arm.candidates.candidates[1].score,
          "the wrong answer scores strictly HIGHER, not equal");
    check_near(arm.candidates.candidates[0].score, 0.25390625, 1e-12, "distractor score (by hand)");
    check_near(arm.candidates.candidates[1].score, 0.0703125, 1e-12, "true-page score (by hand)");

    // (2) THE CHANNEL STILL PUT THE TRUE PAGE IN. A source narrows; it does not have to be right at
    // rank 1, and this is the whole reason it is admissible. The page list is bound to a LOCAL
    // (see the note at TALLY B): `pages()` returns by value, and two calls in one `std::find` would
    // pair iterators from two different vectors.
    const std::vector<std::uint32_t> channel_pages = arm.candidates.pages();
    check(arm.candidates.found(), "the channel returned a candidate set, not a refusal");
    check(std::find(channel_pages.begin(), channel_pages.end(), 2U) != channel_pages.end(),
          "THE TRUE PAGE IS IN THE CHANNEL'S CANDIDATE SET");
    check_equal(arm.candidates.distinct_pages, 2U, "over two distinct pages");

    // (3) THE ARBITER, READING ITS OWN FIELDS, ANCHORED ON THE TRUE PAGE. `anchor_page` is the
    // engine's own column of `SumDirReachResult`; nothing here is inferred from the plan's shape.
    check(arm.arbiter.result.found, "the arbiter found the query");
    check_equal(arm.arbiter.result.anchor_page, 2U,
                "THE ARBITER'S OWN anchor_page IS THE TRUE PAGE");
    check_equal(arm.arbiter.result.window, 16U, "and its evidence is the full 16-token run");
    check_equal(arm.arbiter.result.alternatives, 0U,
                "the distractor is NOT an alternative: it holds no admissible evidence at all");

    // (4) THE UNION DID NOT MOVE THE ANCHOR, and this is a CALCULATED field, not a claim.
    check_equal(arm.union_set.anchor_page, arm.arbiter.result.anchor_page,
                "the union's anchor IS the arbiter's anchor, copied");
    check(arm.union_set.anchor_preserved, "anchor_preserved is set by the union, and it is true");
    check(!arm.union_set.determined_by_semchan(), "and the channel determined nothing");
    check(arm.union_set.holds(2U), "the union's page set contains the true page");
    check(arm.union_set.holds(3U), "and it also contains the distractors the CHANNEL added");
    check_equal(arm.union_set.from_arbiter, 1U, "the arbiter named one page");
    check_equal(arm.union_set.from_semchan_only, 1U, "the channel added exactly one it had not seen");
    check_equal(arm.union_set.in_both, 1U, "and the two agree on exactly one page");
}

// ---------------------------------------------------------------------------
// S4/P1/P2: THE PROVENANCE CRITERION -- the origin travels, un-merged, and the channel chooses none
// ---------------------------------------------------------------------------

void section4_provenance() {
    std::printf("---- P1 same page (same frame), two origins ----\n");
    const std::vector<std::uint32_t> query = the_query();
    {
        // TWO ROWS ON ONE PAGE. The page number is the engine's addressing unit and here it is the
        // same for both; the ORIGINS are not. A channel that scored by the frame would see one
        // block; this one must see two, and must hand both on.
        std::vector<RowSpec> specs;
        specs.push_back(RowSpec{page_true(query), /*page=*/2U, /*token_begin=*/128U, /*gen=*/1U});
        specs.push_back(RowSpec{page_true(query), /*page=*/2U, /*token_begin=*/4096U + 128U,
                                /*gen=*/2U});
        const Arm arm = run_arm(specs, query, /*fanout=*/2U, /*cap=*/0U);
        std::printf("     %s\n", sc::semchan_candidates_line(arm.candidates).c_str());
        std::printf("     %s\n", sc::semchan_union_line(arm.union_set).c_str());
        std::printf("     %s\n", sd::sum_dir_reach_line(arm.arbiter.result).c_str());

        check_equal(arm.index_report.rows_indexed, 2U, "both rows are indexed");
        check_equal(arm.index_report.same_frame_distinct_origin, 1U,
                    "P1 the index COUNTED one same-frame/different-origin pair");
        check(arm.candidates.found(), "P1 the channel returned candidates");
        check_equal(arm.candidates.candidates.size(), 2U,
                    "P1 THE CHANNEL RETURNED TWO RECORDS, NOT ONE");
        check_equal(arm.candidates.distinct_pages, 1U,
                    "P1 they are ONE distinct page -- the collision is the point");
        check_equal(arm.candidates.same_page_distinct_origin, 1U,
                    "P1 and the channel COUNTED the collision instead of resolving it");
        const std::vector<sc::SemChanProvenance> origins = arm.candidates.origins_of(2U);
        check_equal(origins.size(), 2U, "P1 both origins are readable off the candidates");
        if (origins.size() == 2U) {
            check(origins[0] != origins[1], "P1 and they are DISTINCT origins");
            check_equal(origins[0].content_identity, origins[1].content_identity,
                        "P1 with the SAME position frame (the same content identity)");
            check(origins[0].source_begin != origins[1].source_begin,
                  "P1 and DIFFERENT source spans");
        }
        const std::vector<sc::SemChanProvenance> in_union = arm.union_set.origins_of(2U);
        check_equal(in_union.size(), 2U,
                    "P1 THE UNION HANDED THE ARBITER BOTH ORIGINS, NOT A MERGED ONE");
        check_equal(arm.union_set.pages_with_multiple_origins, 1U,
                    "P1 and the union counted the multiplicity");
        check(!sc::semchan_origin_may_be_chosen_by_channel(),
              "P1 the channel is not allowed to pick one of the two");
    }

    std::printf("---- P2 the same text at two origins, and the arbiter picks one ----\n");
    {
        std::vector<RowSpec> specs;
        specs.push_back(RowSpec{page_true(query), /*page=*/0U, /*token_begin=*/24U, /*gen=*/1U});
        specs.push_back(RowSpec{page_true(query), /*page=*/1U, /*token_begin=*/4096U + 24U,
                                /*gen=*/2U});
        const Arm arm = run_arm(specs, query, /*fanout=*/2U, /*cap=*/0U);
        std::printf("     %s\n", sc::semchan_candidates_line(arm.candidates).c_str());
        std::printf("     %s\n", sc::semchan_union_line(arm.union_set).c_str());
        std::printf("     %s\n", sd::sum_dir_reach_line(arm.arbiter.result).c_str());

        check_equal(arm.index_report.same_frame_distinct_origin, 1U,
                    "P2 the index counted the identity collision across two origins");
        check_equal(arm.candidates.candidates.size(), 2U,
                    "P2 the channel did NOT dedupe by identity -- both records survive");
        check_equal(arm.candidates.distinct_pages, 2U, "P2 on two distinct pages");
        check_equal(arm.candidates.same_page_distinct_origin, 0U,
                    "P2 no same-page collision here, and the counter says so instead of guessing");
        check(arm.arbiter.result.found, "P2 the arbiter found the query");
        check(arm.arbiter.result.anchor_page == 0U || arm.arbiter.result.anchor_page == 1U,
              "P2 the arbiter's anchor is one of the two origins");
        const std::vector<sc::SemChanProvenance> chosen =
            arm.union_set.origins_of(arm.arbiter.result.anchor_page);
        check_equal(chosen.size(), 1U,
                    "P2 the origin of the page the ARBITER chose is UNIQUE and readable");
        if (chosen.size() == 1U) {
            std::printf("     P2 arbiter chose page %u, origin source=[%u,%u) generation=%u\n",
                        arm.arbiter.result.anchor_page, chosen[0].source_begin,
                        chosen[0].source_end, chosen[0].generation);
        }
    }
}

// ---------------------------------------------------------------------------
// S5: the named refusals (the must-fail side of the channel's own contract)
// ---------------------------------------------------------------------------

void section5_named_refusals() {
    std::printf("---- S5 the named refusals, and their controls ----\n");
    const std::vector<std::uint32_t> query = the_query();
    const std::vector<RowSpec> specs = main_specs(query);

    // The control: the same directory, the same query, a fanout that does NOT bind. If this is not
    // green, every red below is meaningless.
    {
        const Arm control = run_arm(specs, query, /*fanout=*/8U, /*cap=*/0U);
        check(control.candidates.found(), "control: fanout 8 does not bind, so it is green");
        check_equal(control.candidates.candidates.size(), 2U, "control: two records");
    }

    // R1  THE BOUND BINDS, SO THE PASS REFUSES. Truncating to the top-1 would be the "drop the
    // anchor" defect the tree already paid for (F480 sec.6), and it would ALSO have hidden the true
    // page here -- the wrong answer would have been the only answer.
    {
        const Arm bound = run_arm(specs, query, /*fanout=*/1U, /*cap=*/0U);
        const bool fired = bound.candidates.status == sc::SemChanCandidateStatus::RefusedBoundBinds;
        check(fired, "R1 a binding fanout REFUSES (refused-bound-binds) instead of truncating");
        check(!bound.candidates.found(), "R1 and it did not answer");
        check(!bound.candidates.refusal.empty(), "R1 and the refusal carries a reason");
        check_equal(arm_positive(specs, query), 2U, "R1 the two positive records are the reason");
    }

    // R2  NO SHARED SYMBOL AT ALL => RefusedNoPositiveScore, which is a SILENT MISS if reported as
    // "nothing found". The control is the same call with one symbol shared.
    {
        std::vector<RowSpec> disjoint;
        disjoint.push_back(RowSpec{filler(3000U, 64U), 0U, 0U, 1U});
        const Arm none = run_arm(disjoint, query, /*fanout=*/8U, /*cap=*/0U);
        check(none.candidates.status == sc::SemChanCandidateStatus::RefusedNoPositiveScore,
              "R2 a zero-overlap query is REFUSED by name, not answered with nothing");
        check(!none.candidates.refusal.empty(), "R2 and the refusal says why");
    }

    // R3  A BLOCK WITH NO ORIGIN MAKES THE PASS REFUSE. This is the provenance criterion's own red
    // arm: emitting the block would hand the arbiter a candidate whose origin is exactly what
    // KVMem's re-based frame loses.
    {
        sd::SumDir directory(/*sequence_tag=*/1U, kBlockTokens);
        const std::vector<std::uint32_t> block = page_true(query);
        const std::size_t row = directory.append(block, 128U, 1U, sd::SumDirCodec::Int8, 2, 2U);
        directory.set_summary(row, sd::sum_dir_render_token_line(block));
        directory.sort_rows();
        // THE MUTATION IS APPLIED TO THE READ, NOT TO THE DIRECTORY: a row whose identity column
        // was zeroed is what a half-written row looks like, and `sum_dir_row_well_formed`
        // (sum_dir.h:591) names the same condition. The index must notice, and the pass must refuse.
        std::vector<sc::SemChanBlockEntry> index = sc::semchan_index_blocks(directory);
        check_equal(index.size(), 1U, "R3 control: one block indexed before the origin is removed");
        for (sc::SemChanBlockEntry& entry : index) {
            entry.provenance.content_identity = sd::SumDirDigest{}; // pristine == "no identity"
        }
        const sc::SemChanCandidates refused = sc::semchan_candidate_pages(
            index, sc::semchan_symbol_key_from_ids(query), 8U);
        check(refused.status == sc::SemChanCandidateStatus::RefusedOriginMissing,
              "R3 an origin-less block makes the PASS refuse (refused-origin-missing)");
        check(!refused.refusal.empty(), "R3 and the refusal names the reason");
        check_equal(refused.candidates.size(), 0U, "R3 no origin-less candidate is emitted");
        // THE DECLARED ALTERNATIVE, so the policy is a policy and not a decoration: under
        // DropAndCountLoudly the same input answers, drops the block, and COUNTS it in the line.
        const sc::SemChanCandidates dropped = sc::semchan_candidate_pages(
            index, sc::semchan_symbol_key_from_ids(query), 8U,
            sc::SemChanOriginPolicy::DropAndCountLoudly);
        check(dropped.status == sc::SemChanCandidateStatus::RefusedNoPositiveScore,
              "R3' under DropAndCountLoudly the block is dropped, and then nothing is positive");
        check_equal(dropped.dropped_no_origin, 1U, "R3' and the drop is COUNTED, not silent");
        check(sc::semchan_candidates_line(dropped).find("dropped_no_origin=1") != std::string::npos,
              "R3' and the count is in the printable line");
    }

    // R4  AN UNFORMED QUERY IS "NOT ASKED", WHICH IS NOT "NOTHING FOUND".
    {
        const Arm unasked = run_arm(specs, {}, /*fanout=*/8U, /*cap=*/0U);
        check(unasked.candidates.status == sc::SemChanCandidateStatus::RefusedEmptyQuery,
              "R4 an empty query is REFUSED as 'not asked'");
    }

    // R5  AN EMPTY INDEX IS ITS OWN REFUSAL, not a zero-overlap answer.
    {
        const sc::SemChanCandidates none = sc::semchan_candidate_pages(
            {}, sc::semchan_symbol_key_from_ids(query), 8U);
        check(none.status == sc::SemChanCandidateStatus::RefusedNoBlocks,
              "R5 an empty index is refused by name");
    }

    // R6  THE UNION REFUSES RATHER THAN TRUNCATING TO THE CAP.
    {
        const Arm arm = run_arm(specs, query, /*fanout=*/8U, /*cap=*/1U);
        check(arm.union_set.refused_over_cap,
              "R6 a cap that the union would exceed makes the union REFUSE");
        check(!arm.union_set.refusal.empty(), "R6 and the union's refusal carries a reason");
        check_equal(arm.union_set.anchor_page, arm.arbiter.result.anchor_page,
                    "R6 and the anchor is STILL the arbiter's, refusal or not");
        // Control: the same union with a cap that fits.
        const Arm fits = run_arm(specs, query, /*fanout=*/8U, /*cap=*/8U);
        check(!fits.union_set.refused_over_cap, "R6 control: a cap that fits does not refuse");
        check_equal(fits.union_set.from_semchan_only, 1U, "R6 control: one page added by the channel");
    }
}

// ---------------------------------------------------------------------------
// S6: THE TWO INDEPENDENT TALLIES, WITH THEIR OWN DENOMINATORS, NEVER SUMMED
// ---------------------------------------------------------------------------

void section6_two_tallies() {
    std::printf("---- S6 two readings of two different sets, denominators printed apart ----\n");
    const std::vector<std::uint32_t> query = the_query();
    const Arm arm = run_arm(main_specs(query), query, /*fanout=*/2U, /*cap=*/0U);

    // TALLY A -- A SET COMPARISON over PAGES. The question is "does the union contain the page the
    // exact evidence names", and the denominator is ONE (the truth is one page here). This is the
    // only tally that says anything about correctness.
    const std::uint32_t true_page = 2U;
    const bool union_holds_truth = arm.union_set.holds(true_page);
    const bool channel_holds_truth =
        std::find(arm.candidates.pages().begin(), arm.candidates.pages().end(), true_page) !=
        arm.candidates.pages().end();
    const bool arbiter_holds_truth = arm.arbiter.result.anchor_page == true_page;
    std::printf("     TALLY A (set comparison, denominator = 1 true page):\n");
    std::printf("       channel candidate set contains truth : %d/1\n", channel_holds_truth ? 1 : 0);
    std::printf("       arbiter anchor is truth              : %d/1\n", arbiter_holds_truth ? 1 : 0);
    std::printf("       union contains truth                 : %d/1\n", union_holds_truth ? 1 : 0);
    check(channel_holds_truth, "TALLY A: the channel's candidate SET contains the truth");
    check(arbiter_holds_truth, "TALLY A: the arbiter decided the truth");
    check(union_holds_truth, "TALLY A: the union contains the truth");

    // TALLY B -- PER-ELEMENT TRUTH over the pages THE DIRECTORY HAS, one row per element. The
    // denominator is the row count (4), NOT 1, and it is a DIFFERENT measurement of a DIFFERENT
    // set: it asks where each directory page ended up, which TALLY A cannot express. The two
    // numbers are printed apart and are NOT summed -- the project rule is 比集合不比个数.
    //
    // `pages()` RETURNS A VECTOR BY VALUE, so it is bound to a LOCAL here and never called twice
    // inside one `std::find`. Calling it twice would make `first` and `last` come from two
    // different temporaries: the `first != last` test then compares pointers into two buffers, the
    // loop runs off its end, and the arm reports matches that do not exist. That is not a
    // hypothetical -- it is this line's own F-01 (dl/semchan/LEDGER.md) and it was caught by
    // exactly this tally disagreeing with the printed page lists above.
    const std::vector<std::uint32_t> channel_pages = arm.candidates.pages();
    std::uint32_t in_arbiter = 0;
    std::uint32_t in_channel = 0;
    std::uint32_t in_both = 0;
    std::uint32_t in_neither = 0;
    for (const sd::SumDirRow& row : arm.directory.rows()) {
        const bool a =
            arm.union_set.holds(row.page) &&
            std::find(arm.arbiter.pages.begin(), arm.arbiter.pages.end(), row.page) !=
                arm.arbiter.pages.end();
        const bool s = std::find(channel_pages.begin(), channel_pages.end(), row.page) !=
                       channel_pages.end();
        if (a && s) { ++in_both; } else if (a) { ++in_arbiter; } else if (s) { ++in_channel; } else { ++in_neither; }
    }
    std::printf("     TALLY B (per-page element truth, denominator = %u directory pages):\n",
                static_cast<unsigned>(arm.directory.rows().size()));
    std::printf("       arbiter only=%u  channel only=%u  both=%u  neither=%u  (sum=%u)\n",
                in_arbiter, in_channel, in_both, in_neither,
                in_arbiter + in_channel + in_both + in_neither);
    std::printf("       arbiter pages=[");
    for (std::size_t i = 0; i < arm.arbiter.pages.size(); ++i) {
        std::printf("%s%u", i == 0 ? "" : ",", arm.arbiter.pages[i]);
    }
    std::printf("]  channel candidate pages=[");
    {
        const std::vector<std::uint32_t> cp = arm.candidates.pages();
        for (std::size_t i = 0; i < cp.size(); ++i) {
            std::printf("%s%u", i == 0 ? "" : ",", cp[i]);
        }
    }
    std::printf("]  union pages=[");
    for (std::size_t i = 0; i < arm.union_set.pages.size(); ++i) {
        std::printf("%s%u", i == 0 ? "" : ",", arm.union_set.pages[i]);
    }
    std::printf("]  directory row pages=[");
    for (std::size_t i = 0; i < arm.directory.rows().size(); ++i) {
        std::printf("%s%u", i == 0 ? "" : ",", arm.directory.rows()[i].page);
    }
    std::printf("]\n");
    check_equal(in_arbiter + in_channel + in_both + in_neither,
                static_cast<std::uint32_t>(arm.directory.rows().size()),
                "TALLY B: every page is accounted for, and the denominator is the row count");
    check_equal(in_arbiter, 0U, "TALLY B: the arbiter named nothing the channel did not");
    check_equal(in_both, 1U, "TALLY B: ONE page named by BOTH -- the true page");
    check_equal(in_channel, 1U,
                "TALLY B: ONE page named ONLY by the channel -- the page the arbiter cannot "
                "reach, and the channel's entire contribution to this arm");
    check_equal(in_neither, 2U, "TALLY B: the two pure-filler pages, named by neither side");

    // THE ANCHOR, PRINTED FROM THE ENGINE'S OWN COLUMNS AND NOT FROM THE UNION'S. This is the
    // "arbiter still decides" evidence in one line, and it is read off `SumDirReachResult`:
    std::printf("     arbiter fields: status=%s found=%d anchor_page=%u window=%u candidates=%u "
                "alternatives=%u raw_hits=%zu admissible=%zu pages=%u\n",
                sd::sum_dir_reach_status_name(arm.arbiter.result.status),
                arm.arbiter.result.found ? 1 : 0, arm.arbiter.result.anchor_page,
                arm.arbiter.result.window, arm.arbiter.result.candidates,
                arm.arbiter.result.alternatives, arm.arbiter.result.raw_hits,
                arm.arbiter.result.admissible, arm.arbiter.result.pages);
    check(std::strcmp(sd::sum_dir_reach_status_name(arm.arbiter.result.status), "found") == 0,
          "the arbiter's own status string is 'found'");
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = "all";
    if (argc > 1) { mode = argv[1]; }
    std::printf("ninfer_semchan_symbol_key_test mode=%s\n", mode.c_str());

    if (mode == "all" || mode == "green") {
        section0_names_and_refusals();
        section1_producer();
        section3_channel_wrong_arbiter_right();
        section4_provenance();
        section5_named_refusals();
        section6_two_tallies();
    }

    // THE RED ARMS. A red arm is a statement about the world OUTSIDE this process (a mutated header
    // image compiled against this same source), so `mode=red` runs the arms that can only be
    // decided there and REQUIRES each named check to have fired. Running them here would be a test
    // that can only pass, which this project refuses by name.
    if (mode == "red") {
        std::printf("     RED mode: this binary's own copy of the channel is UNMUTATED, so the\n"
                    "     mutations are applied by the driver in dl/semchan/sh/40_mutants.sh with\n"
                    "     -I precedence over a mutated header image, and each arm's fired/not-fired\n"
                    "     is read from THAT run's output. Nothing to assert in-process.\n");
    }

    std::printf("checks=%d failures=%d\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("VERDICT=FAIL\n");
        return 1;
    }
    std::printf("VERDICT=PASS\n");
    return 0;
}
