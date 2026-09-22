// tests/test_sum_dir_vector.cpp -- host-only self-test for src/spec/sum_dir_vector.h
//
// Deliberately the same shape as tests/test_sum_dir.cpp beside it: no CUDA, no artifact, no
// fixture, plain `g++`. What this file is FOR, and what each section proves rather than asserts:
//
//   1. knobs + parsers: an ARMED retriever with no width is refused, and a misspelled strategy is
//      reported instead of silently becoming a different strategy.
//   2. POSITION INDEPENDENCE, the header's one load-bearing rule: the same block content at two
//      offsets is ONE retrieval target, and `set()` reads the identity from the directory rather
//      than from the caller, so a vector cannot be filed under a position.
//   3. cosine geometry: the ranking is the known one, and it is invariant to the ENCODER's scale.
//   4. a TOTAL, reproducible order: equal scores resolve to the smaller row.
//   5. top_k: the truncated answer is exactly the prefix of the untruncated one.
//   6. every refusal is NAMED and stores nothing: zero norm, NaN, a width mismatch, a row out of
//      range.
//   7. THE CONTRACT: with a one-hot term axis, the vector retriever's perfect-score set IS
//      `SumDir::search_summaries`'s hit set, over a corpus, including the rows both must skip.
//   8. the `Live` gate and the `RequireCatalogue` gate each refuse exactly what sum_dir.h refuses.
//   9. `sort_rows()`: a slot that is not re-anchored is REFUSED (never answered with the wrong
//      block), `rebind()` re-anchors it, and the block it names does not change.
//  10. the wire format: byte-exact round trip, a torn header byte caught, a torn payload byte
//      caught, and a column taken against a DIFFERENT directory refused.

#include "spec/sum_dir_vector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::spec::sum_dir::SumDir;
using ninfer::spec::sum_dir::SumDirCodec;
using ninfer::spec::sum_dir::SumDirDigest;
using ninfer::spec::sum_dir::SumDirState;
using ninfer::spec::sum_dir::vector_dir::SumDirVectorColumn;
using ninfer::spec::sum_dir::vector_dir::SumDirVectorHit;
using ninfer::spec::sum_dir::vector_dir::SumDirVectorKnobs;
using ninfer::spec::sum_dir::vector_dir::SumDirVectorLoadReport;
using ninfer::spec::sum_dir::vector_dir::SumDirVectorMetric;
using ninfer::spec::sum_dir::vector_dir::SumDirVectorMissing;
using ninfer::spec::sum_dir::vector_dir::SumDirVectorOffer;
using ninfer::spec::sum_dir::vector_dir::SumDirVectorSearchReport;
using ninfer::spec::sum_dir::vector_dir::SumDirVectorSetResult;

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++g_failures;
    }
}

template <typename A, typename B>
void check_equal(const A& got, const B& want, const char* what) {
    if (!(got == want)) {
        std::fprintf(stderr, "FAIL %s (got %lld want %lld)\n", what,
                     static_cast<long long>(got), static_cast<long long>(want));
        ++g_failures;
    }
}

// A block whose tokens are a deterministic function of `seed`, so two calls with the same seed
// produce the SAME content and therefore the same `sum_dir_block_digest` -- which is the only
// thing section 2 needs.
std::vector<std::uint32_t> block_of(std::uint32_t seed, std::size_t count) {
    std::vector<std::uint32_t> tokens;
    tokens.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        tokens.push_back(seed * 7919U + static_cast<std::uint32_t>(i) * 31U + 1U);
    }
    return tokens;
}

// Appends one 64-token block and returns its row. `file_slot >= 0` makes the row Live.
std::size_t append_block(SumDir& directory, std::uint32_t seed, std::uint32_t page,
                         std::int32_t file_slot = 7) {
    const std::vector<std::uint32_t> tokens = block_of(seed, 64);
    return directory.append(tokens, page * 64U, /*generation=*/1U, SumDirCodec::Int8, file_slot,
                            page);
}

std::vector<float> unit(std::uint32_t dim, std::uint32_t axis, float scale = 1.0f) {
    std::vector<float> out(dim, 0.0f);
    out[axis] = scale;
    return out;
}

// ---------------------------------------------------------------------------
// section 1 -- knobs and parsers
// ---------------------------------------------------------------------------
void section_1_knobs() {
    bool recognised = false;
    check_equal(static_cast<unsigned>(ninfer::spec::sum_dir::vector_dir::sum_dir_vector_metric_parse(
                     "dot", recognised)),
                static_cast<unsigned>(SumDirVectorMetric::Dot), "metric dot parses");
    check(recognised, "metric dot recognised");
    ninfer::spec::sum_dir::vector_dir::sum_dir_vector_metric_parse("Dot", recognised);
    check(!recognised, "an unknown metric is reported, never silently accepted");
    check_equal(static_cast<unsigned>(
                    ninfer::spec::sum_dir::vector_dir::sum_dir_vector_metric_parse("Dot",
                                                                                   recognised)),
                static_cast<unsigned>(SumDirVectorMetric::Cosine), "a typo keeps the default");

    ninfer::spec::sum_dir::vector_dir::sum_dir_vector_missing_parse("skip", recognised);
    check(recognised, "missing skip parses");
    ninfer::spec::sum_dir::vector_dir::sum_dir_vector_offer_parse("require-embedding",
                                                                 recognised);
    check(recognised, "offer require-embedding parses");
    ninfer::spec::sum_dir::vector_dir::sum_dir_vector_offer_parse("embedding", recognised);
    check(!recognised, "an unknown offer is reported");

    // The width has no default on purpose: an armed retriever with no width is NOT armed.
    unsetenv("NINFER_SUM_DIR_VEC");
    unsetenv("NINFER_SUM_DIR_VEC_DIM");
    SumDirVectorKnobs unarmed = SumDirVectorKnobs::from_env();
    check(!unarmed.enabled, "the vector retriever is OFF with no environment");
    check(unarmed.dim_missing, "an unset width is REPORTED, not defaulted");

    setenv("NINFER_SUM_DIR_VEC", "1", 1);
    SumDirVectorKnobs armed_no_dim = SumDirVectorKnobs::from_env();
    check(!armed_no_dim.enabled, "an armed retriever with no width refuses to arm");
    setenv("NINFER_SUM_DIR_VEC_DIM", "5120", 1);
    SumDirVectorKnobs armed = SumDirVectorKnobs::from_env();
    check(armed.enabled, "with a width it arms");
    check_equal(armed.dim, 5120U, "the width is read");
    check_equal(armed.top_k, ninfer::spec::sum_dir::vector_dir::kSumDirVectorDefaultTopK,
                "the default top_k is the header's");
    unsetenv("NINFER_SUM_DIR_VEC");
    unsetenv("NINFER_SUM_DIR_VEC_DIM");
}

// ---------------------------------------------------------------------------
// section 2 -- identity is CONTENT, not position
// ---------------------------------------------------------------------------
void section_2_identity_is_content() {
    SumDir directory(/*sequence_tag=*/11);
    // The SAME content at page 0 and page 5. Position differs; content does not.
    const std::size_t early = append_block(directory, /*seed=*/3, /*page=*/0);
    const std::size_t late  = append_block(directory, /*seed=*/3, /*page=*/5);
    check_equal(directory.rows()[early].block_identity.lo,
                directory.rows()[late].block_identity.lo, "one block, two offsets, one identity");
    check(directory.rows()[early].token_begin != directory.rows()[late].token_begin,
          "the two rows really are at different positions");
    check(directory.rows()[early].block_identity == directory.rows()[late].block_identity,
          "and their identities are EQUAL");

    const std::uint32_t dim = 4;
    SumDirVectorColumn column(dim);
    column.set(directory, early, unit(dim, 1));
    column.set(directory, late, unit(dim, 1));
    check_equal(column.size(), std::size_t{2}, "two slots");
    check(column.slots()[early].identity == column.slots()[late].identity,
          "the column files both copies under the SAME key");

    // The identity comes from the DIRECTORY. Handing `set()` a row whose content differs is not a
    // way to file a vector under a position: the caller cannot even name an identity.
    const std::size_t other = append_block(directory, /*seed=*/99, /*page=*/6);
    column.set(directory, other, unit(dim, 3));
    check(column.slots()[other].identity == directory.rows()[other].block_identity,
          "set() reads the identity from the row it is given, so it cannot be spoofed");
    check(column.stale_slots(directory) == 0, "no stale slots immediately after set()");

    SumDirVectorSearchReport report;
    const std::vector<SumDirVectorHit> hits =
        column.search_top_k(directory, unit(dim, 1), 64, &report);
    // Both copies of the same block and nothing else: the query axis is 1, and only those two
    // rows carry it.
    // The retriever RANKS, it does not FILTER: every Live row carrying an embedding is scored and
    // returned (up to top_k), the orthogonal one included, at score 0.0. Sections 5 and 7 encode
    // that same rule (`the untruncated answer has every row`; `eight rows carry an embedding`), and
    // the header exposes `score` on every hit so a caller that wants a filter can threshold. So the
    // assertion is about the TOP of the ranking, not about the size of the set.
    check_equal(hits.size(), std::size_t{3},
                "every eligible row is returned -- top-k ranks, it does not filter");
    check(hits[0].identity == hits[1].identity, "and the two copies of the block are the TOP two");
    check(hits[0].score > 0.99f && hits[1].score > 0.99f, "both with a maximal score");
    check(hits[2].score == 0.0f, "and the orthogonal row comes last, at score 0");
}

// ---------------------------------------------------------------------------
// section 3 -- cosine geometry, and invariance to the encoder's scale
// ---------------------------------------------------------------------------
void section_3_cosine() {
    SumDir directory(12);
    // Three rows with a known geometry in 2-D: the query is (1, 0).
    const std::size_t a = append_block(directory, 1, 0); // angle   0 deg -> cos 1.000
    const std::size_t b = append_block(directory, 2, 1); // angle  60 deg -> cos 0.500
    const std::size_t c = append_block(directory, 3, 2); // angle 120 deg -> cos -0.500
    const float kSqrt3Over2 = 0.8660254f;
    SumDirVectorColumn column(2);
    column.set(directory, a, std::vector<float>{1.0f, 0.0f});
    column.set(directory, b, std::vector<float>{0.5f, kSqrt3Over2});
    column.set(directory, c, std::vector<float>{-0.5f, kSqrt3Over2});

    const std::vector<SumDirVectorHit> hits =
        column.search_top_k(directory, std::vector<float>{1.0f, 0.0f}, 8);
    check_equal(hits.size(), std::size_t{3}, "three hits");
    check_equal(hits[0].row, a, "the identity direction is first");
    check_equal(hits[1].row, b, "the 60-degree row is second");
    check_equal(hits[2].row, c, "the 120-degree row is last");
    check(hits[0].score > 0.99f && hits[0].score <= 1.0f, "cos(self) == 1");
    check(hits[1].score > 0.49f && hits[1].score < 0.51f, "cos(60) == 0.5");
    check(hits[2].score < -0.49f && hits[2].score > -0.51f, "cos(120) == -0.5");

    // SCALE INVARIANCE: an encoder that emits 1000x larger vectors must not change the ranking.
    // This is the property that makes cosine the default rather than dot.
    SumDirVectorColumn scaled(2);
    scaled.set(directory, a, std::vector<float>{1000.0f, 0.0f});
    scaled.set(directory, b, std::vector<float>{500.0f, 866.0254f});
    scaled.set(directory, c, std::vector<float>{-500.0f, 866.0254f});
    const std::vector<SumDirVectorHit> scaled_hits =
        scaled.search_top_k(directory, std::vector<float>{17.0f, 0.0f}, 8);
    check_equal(scaled_hits.size(), hits.size(), "same count under a rescaled encoder");
    for (std::size_t i = 0; i < hits.size(); ++i) {
        check_equal(scaled_hits[i].row, hits[i].row, "same ORDER under a rescaled encoder");
        check(std::fabs(scaled_hits[i].score - hits[i].score) < 1e-4f,
              "same SCORE under a rescaled encoder");
    }

    // Dot keeps the magnitude, which is exactly why it is not the default.
    SumDirVectorColumn dot(2, SumDirVectorMetric::Dot);
    dot.set(directory, a, std::vector<float>{1.0f, 0.0f});
    dot.set(directory, b, std::vector<float>{0.5f, kSqrt3Over2});
    dot.set(directory, c, std::vector<float>{-0.5f, kSqrt3Over2});
    const std::vector<SumDirVectorHit> dot_hits =
        dot.search_top_k(directory, std::vector<float>{10.0f, 0.0f}, 8);
    check(std::fabs(dot_hits[0].score - 10.0f) < 1e-4f, "dot reports the raw inner product");
    check_equal(dot_hits[0].row, a, "and the largest magnitude wins regardless of angle");
}

// ---------------------------------------------------------------------------
// section 4 -- a total, reproducible order
// ---------------------------------------------------------------------------
void section_4_order_is_total() {
    SumDir directory(13);
    // Four rows that all get the SAME vector: the scores are identical, so only the tie-break
    // can order them, and it must be the smaller row first.
    const std::size_t r0 = append_block(directory, 10, 0);
    const std::size_t r1 = append_block(directory, 11, 1);
    const std::size_t r2 = append_block(directory, 12, 2);
    const std::size_t r3 = append_block(directory, 13, 3);
    SumDirVectorColumn column(3);
    for (const std::size_t row : {r0, r1, r2, r3}) {
        column.set(directory, row, unit(3, 2));
    }
    const std::vector<SumDirVectorHit> hits = column.search_top_k(directory, unit(3, 2), 8);
    check_equal(hits.size(), std::size_t{4}, "four tied hits");
    for (std::size_t i = 0; i + 1 < hits.size(); ++i) {
        check(hits[i].score == hits[i + 1].score, "the scores really are tied");
        check(hits[i].row < hits[i + 1].row, "a tie resolves to the SMALLER row");
    }
    // Reproducible: the same call twice cannot disagree.
    const std::vector<SumDirVectorHit> again = column.search_top_k(directory, unit(3, 2), 8);
    check_equal(again.size(), hits.size(), "the same call twice: same count");
    for (std::size_t i = 0; i < hits.size(); ++i) {
        check_equal(again[i].row, hits[i].row, "the same call twice: same order");
    }
}

// ---------------------------------------------------------------------------
// section 5 -- top_k is exactly the prefix
// ---------------------------------------------------------------------------
void section_5_top_k() {
    SumDir directory(14);
    // SIX axes, because the loop below puts row i on axis i for i in 0..5. A five-element vector
    // written on axis 5 is a heap overflow, not a test failure.
    const std::uint32_t dim = 6;
    std::vector<std::size_t> rows;
    SumDirVectorColumn column(dim);
    for (std::uint32_t i = 0; i < 6; ++i) {
        const std::size_t row = append_block(directory, 100 + i, i);
        rows.push_back(row);
        // Row i points at axis i and the query points at axis 2, so the score order is NOT the
        // row order: row 2 scores 1.0 and every other row scores 0.0.
        column.set(directory, row, unit(dim, i));
    }
    const std::vector<SumDirVectorHit> full = column.search_top_k(directory, unit(dim, 2), 64);
    check_equal(full.size(), std::size_t{6}, "the untruncated answer has every row");
    check_equal(full[0].row, rows[2], "the matching axis ranks first");

    for (std::uint32_t k = 1; k <= 4; ++k) {
        SumDirVectorSearchReport report;
        const std::vector<SumDirVectorHit> cut =
            column.search_top_k(directory, unit(dim, 2), k, &report);
        check_equal(cut.size(), static_cast<std::size_t>(k), "the truncated answer has k rows");
        check(report.truncated, "truncation is REPORTED");
        for (std::uint32_t i = 0; i < k; ++i) {
            check_equal(cut[i].row, full[i].row, "the truncated answer is the PREFIX of the full one");
            check(cut[i].score == full[i].score, "with the same scores");
        }
    }
    // top_k == 0 is an empty answer and not a crash, and the rows summary still reports what was
    // looked at.
    SumDirVectorSearchReport zero_report;
    check_equal(column.search_top_k(directory, unit(dim, 2), 0, &zero_report).size(),
                std::size_t{0}, "top_k 0 returns nothing");
    check_equal(zero_report.considered, 6U, "and still reports every row it scored");
}

// ---------------------------------------------------------------------------
// section 6 -- every refusal is named, and nothing is stored
// ---------------------------------------------------------------------------
void section_6_refusals() {
    SumDir directory(15);
    const std::size_t row = append_block(directory, 21, 0);
    SumDirVectorColumn column(4);

    check_equal(static_cast<unsigned>(column.set(directory, row, std::vector<float>{0, 0, 0, 0})),
                static_cast<unsigned>(SumDirVectorSetResult::RefusedZeroNorm),
                "a zero vector is refused: cos against it is undefined, not 0");
    check(!column.has(row), "and nothing is stored");

    std::vector<float> nan_vector{1.0f, 0.0f, 0.0f, 0.0f};
    nan_vector[2] = std::numeric_limits<float>::quiet_NaN();
    check_equal(static_cast<unsigned>(column.set(directory, row, nan_vector)),
                static_cast<unsigned>(SumDirVectorSetResult::RefusedNotFinite),
                "a NaN is refused: it would poison every dot product");
    check(!column.has(row), "and nothing is stored");

    check_equal(static_cast<unsigned>(column.set(directory, row, std::vector<float>{1.0f, 0.0f})),
                static_cast<unsigned>(SumDirVectorSetResult::RefusedDimMismatch),
                "a width mismatch is refused");
    check_equal(static_cast<unsigned>(column.set(directory, 9999, std::vector<float>{1, 0, 0, 0})),
                static_cast<unsigned>(SumDirVectorSetResult::RefusedRowOutOfRange),
                "a row out of range is refused");
    check_equal(static_cast<unsigned>(column.set(directory, row, nullptr, 4)),
                static_cast<unsigned>(SumDirVectorSetResult::RefusedDimMismatch),
                "a null vector is refused");
    check_equal(column.size(), std::size_t{0}, "still no slot after five refusals");

    check_equal(static_cast<unsigned>(SumDirVectorSetResult::Stored), 0U, "Stored is the zero value");
    check(std::string(ninfer::spec::sum_dir::vector_dir::sum_dir_vector_set_result_name(
              SumDirVectorSetResult::RefusedZeroNorm)) == "refused-zero-norm",
          "a refusal has a name");

    // The QUERY is judged ONCE, before any row: a bad query is one refusal, not one per row.
    column.set(directory, row, unit(4, 1));
    SumDirVectorSearchReport report;
    const std::vector<SumDirVectorHit> hits =
        column.search_top_k(directory, std::vector<float>{0, 0, 0, 0}, 8, &report);
    check_equal(hits.size(), std::size_t{0}, "a zero-norm query has no answer");
    check_equal(report.refused_query, 1U, "and it is ONE refusal");
    check_equal(report.refused_missing, 0U, "not one per row");
}

// ---------------------------------------------------------------------------
// section 7 -- THE CONTRACT: same answer as search_summaries
// ---------------------------------------------------------------------------
void section_7_same_as_search_summaries() {
    // Four terms, none of which is a substring of another, so the substring retriever's hit set
    // and a one-hot term axis are the same relation.
    const std::vector<std::string> terms = {"alpha", "beta", "gamma", "delta"};
    const std::uint32_t dim = static_cast<std::uint32_t>(terms.size());

    SumDir directory(16);
    const SumDirVectorColumn defaults(dim);
    // The DEFAULT offering rule is the one the FEATURE is for: a Live row with an embedding is a
    // target, catalogue line or not. Defaulting to RequireCatalogue would re-block vector recall on
    // the summarizer pass the engine does not have.
    check_equal(static_cast<unsigned>(defaults.offer()),
                static_cast<unsigned>(SumDirVectorOffer::RequireEmbedding),
                "the default offering rule does NOT require a catalogue line");

    // The drop-in body: `search_summaries`'s own offering rule, spelled out.
    SumDirVectorColumn column(dim, SumDirVectorMetric::Cosine,
                              SumDirVectorMissing::RefuseWhenMissing,
                              SumDirVectorOffer::RequireCatalogue);
    std::vector<std::size_t> row_of_axis(dim, 0);
    std::uint32_t page = 0;
    for (std::size_t term = 0; term < terms.size(); ++term) {
        for (std::uint32_t copy = 0; copy < 2; ++copy) {
            // Two blocks per term, on DIFFERENT pages, so their positions differ.
            const std::size_t row = append_block(
                directory, static_cast<std::uint32_t>(200 + term * 2 + copy), page);
            directory.set_summary(row, "block about " + terms[term]);
            column.set(directory, row, unit(dim, static_cast<std::uint32_t>(term)));
            row_of_axis[term] = row;
            ++page;
        }
    }
    // A ninth row: an embedding but NO catalogue line. Under the drop-in rule it is a target for
    // neither retriever, and that is what makes the following equality meaningful.
    const std::size_t no_line = append_block(directory, 777, page);
    column.set(directory, no_line, unit(dim, 3)); // HAS an embedding, has NO line
    check(column.has(no_line), "the no-catalogue row does carry an embedding");

    for (std::size_t term = 0; term < terms.size(); ++term) {
        std::vector<std::size_t> expect = directory.search_summaries(terms[term]);
        std::sort(expect.begin(), expect.end());
        check_equal(expect.size(), std::size_t{2}, "the substring retriever finds both copies");
        check_equal(row_of_axis[term], expect[1], "and they are the two rows we appended");

        SumDirVectorSearchReport report;
        const std::vector<SumDirVectorHit> hits =
            column.search_top_k(directory, unit(dim, static_cast<std::uint32_t>(term)), 64,
                                &report);
        std::vector<std::size_t> perfect;
        for (const SumDirVectorHit& hit : hits) {
            if (hit.score > 0.999f) { perfect.push_back(hit.row); }
        }
        std::sort(perfect.begin(), perfect.end());
        check_equal(perfect.size(), expect.size(), "the vector retriever finds the same COUNT");
        for (std::size_t i = 0; i < perfect.size() && i < expect.size(); ++i) {
            check_equal(perfect[i], expect[i], "and names the SAME ROWS");
        }
        // The perfect set must also be the TOP of the ranking, not buried in it.
        check_equal(hits.size(), static_cast<std::size_t>(dim) * 2U, "eight rows carry an embedding");
        for (std::size_t i = 0; i < perfect.size(); ++i) {
            check(hits[i].score > 0.999f, "every perfect hit ranks ABOVE every imperfect one");
        }
    }
    // The row with an embedding and no line is skipped by the DROP-IN body, exactly as
    // search_summaries skips it, and the skip is counted.
    SumDirVectorSearchReport report;
    column.search_top_k(directory, unit(dim, 0), 64, &report);
    check_equal(report.refused_no_catalogue, 1U, "the line-less row is counted, not silently lost");

    // And the default/extension: a Live row with an embedding IS a target even with no line, which
    // is what makes vector recall independent of the summarizer that does not exist yet.
    SumDirVectorColumn extension(dim);
    for (std::size_t term = 0; term < terms.size(); ++term) {
        for (std::uint32_t copy = 0; copy < 2; ++copy) {
            const std::size_t row = static_cast<std::size_t>(term * 2 + copy);
            extension.set(directory, row, unit(dim, static_cast<std::uint32_t>(term)));
        }
    }
    extension.set(directory, no_line, unit(dim, 3));
    SumDirVectorSearchReport extension_report;
    const std::vector<SumDirVectorHit> extension_hits =
        extension.search_top_k(directory, unit(dim, 3), 64, &extension_report);
    bool found_no_line = false;
    for (const SumDirVectorHit& hit : extension_hits) {
        if (hit.row == no_line) { found_no_line = true; }
    }
    check(found_no_line, "under the default rule the line-less row IS a target");
    check_equal(extension_report.refused_no_catalogue, 0U,
                "and nothing is refused for lacking a line");
    check_equal(extension_report.refused_missing, 0U, "and every row has an embedding");
}

// ---------------------------------------------------------------------------
// section 8 -- the Live gate, and the two missing-strategy settings
// ---------------------------------------------------------------------------
void section_8_gates() {
    SumDir directory(17);
    const std::uint32_t dim = 3;
    const std::size_t live_row   = append_block(directory, 31, 0);
    const std::size_t dead_row   = append_block(directory, 32, 1);
    directory.set_summary(live_row, "the live one");
    directory.set_summary(dead_row, "the dead one");
    check(directory.release(dead_row), "the second row is released");
    check_equal(static_cast<unsigned>(directory.rows()[dead_row].state),
                static_cast<unsigned>(SumDirState::Dead), "and it is Dead");

    SumDirVectorColumn column(dim);
    column.set(directory, live_row, unit(dim, 0));
    column.set(directory, dead_row, unit(dim, 0));
    SumDirVectorSearchReport report;
    const std::vector<SumDirVectorHit> hits = column.search_top_k(directory, unit(dim, 0), 8, &report);
    check_equal(hits.size(), std::size_t{1}, "a Dead row is NOT offered");
    check_equal(hits[0].row, live_row, "only the Live one is");
    check_equal(report.refused_not_live, 1U, "the Dead row is counted as such");
    // The same gate, checked against the retriever this one replaces.
    check_equal(directory.search_summaries("the").size(), std::size_t{1},
                "search_summaries also refuses the Dead row");

    // A row with no embedding AT ALL: the two settings disagree, and both are counted.
    const std::size_t gap_row = append_block(directory, 33, 2);
    directory.set_summary(gap_row, "neither has a vector for this");
    SumDirVectorSearchReport refuse_report;
    const std::vector<SumDirVectorHit> refused =
        column.search_top_k(directory, unit(dim, 0), 8, &refuse_report);
    check_equal(refused.size(), std::size_t{0},
                "RefuseWhenMissing is a REFUSAL, not a shorter list");
    check_equal(refuse_report.refused_missing, 1U, "and the refusal is named");

    SumDirVectorColumn skipping(dim, SumDirVectorMetric::Cosine, SumDirVectorMissing::SkipMissing);
    skipping.set(directory, live_row, unit(dim, 0));
    skipping.set(directory, dead_row, unit(dim, 0));
    SumDirVectorSearchReport skip_report;
    const std::vector<SumDirVectorHit> skipped =
        skipping.search_top_k(directory, unit(dim, 0), 8, &skip_report);
    check_equal(skipped.size(), std::size_t{1}, "SkipMissing skips the gap and answers with the rest");
    check_equal(skip_report.refused_missing, 1U, "and still counts it");
}

// ---------------------------------------------------------------------------
// section 9 -- sort_rows() must not be able to lie about which block a vector is for
// ---------------------------------------------------------------------------
void section_9_sort_is_safe() {
    SumDir directory(18);
    const std::uint32_t dim = 3;
    // SIX rows, so that `sort_rows()` (identity ascending, then token_begin) really permutes them
    // with overwhelming probability. The test does not ASSUME a permutation: it reads the outcome
    // out of the directory and asserts `stale_slots >= 1`, so a directory that happened to already
    // be sorted would be a RED check rather than a silent pass.
    const std::size_t last_row = 5;
    SumDirVectorColumn column(dim);
    for (std::uint32_t page = 0; page < 6; ++page) {
        const std::size_t row = append_block(directory, /*seed=*/4 + page, page);
        directory.set_summary(row, "a line");
        column.set(directory, row, unit(dim, page == last_row ? 2 : 0));
    }
    const SumDirDigest identity_of_last = directory.rows()[last_row].block_identity;

    SumDirVectorSearchReport before;
    const std::vector<SumDirVectorHit> before_hits =
        column.search_top_k(directory, unit(dim, 2), 8, &before);
    // SIX Live rows, six fresh slots, top_k 8 > 6: the ranking does NOT filter, so the answer
    // is every eligible row -- the rule sections 2, 5 and 7 encode -- and the QUERIED block is
    // the TOP of it (the two checks below), not the only member of it.
    check_equal(before_hits.size(), std::size_t{6},
                "every eligible row is ranked -- top-k ranks, it does not filter");
    check(before_hits[0].identity == identity_of_last, "and it is the last block's");
    check_equal(before_hits[0].row, last_row, "found at the row it was set on");

    directory.sort_rows();
    check(directory.sorted(), "the directory is sorted");
    const std::uint32_t moved = column.stale_slots(directory);
    check(moved >= 1U, "the sort really permuted at least one slot -- else this section is vacuous");
    // Where the block lives NOW, read out of the directory rather than assumed.
    std::size_t new_row = directory.rows().size();
    for (std::size_t row = 0; row < directory.rows().size(); ++row) {
        if (directory.rows()[row].block_identity == identity_of_last) { new_row = row; break; }
    }
    check(new_row < directory.rows().size(), "the block is still somewhere in the directory");

    // WITHOUT rebind(): the slot still sits at the OLD index. If that index now holds a different
    // block, the column must REFUSE -- never answer with the wrong block, and never claim the
    // block is at the new row.
    SumDirVectorSearchReport stale_report;
    const std::vector<SumDirVectorHit> stale_hits =
        column.search_top_k(directory, unit(dim, 2), 8, &stale_report);
    for (const SumDirVectorHit& hit : stale_hits) {
        check(hit.identity == directory.rows()[hit.row].block_identity,
              "a hit may never name a block it is not for");
    }
    if (moved == 6U) {
        check_equal(stale_hits.size(), std::size_t{0}, "a wholly stale column answers nothing");
        // RefuseWhenMissing is the DEFAULT and it is a REFUSAL, not a shorter list: the scan
        // stops at the FIRST stale slot, so exactly one is counted there. `stale_slots()`
        // (read into `moved` above) is what counts all six. Section 8 asserts this same rule.
        check_equal(stale_report.refused_stale, 1U,
                    "and the refusal is counted at the first stale slot");
    }

    // WITH rebind(): the slot is re-anchored by IDENTITY, so the same block is found again -- and
    // the answer names the block, whose row has moved with it.
    column.rebind(directory);
    check_equal(column.stale_slots(directory), 0U, "no stale slot after rebind()");
    SumDirVectorSearchReport after;
    const std::vector<SumDirVectorHit> after_hits =
        column.search_top_k(directory, unit(dim, 2), 8, &after);
    // The same rule after the sort: rebind() re-anchors all six slots, so all six rows are
    // ranked again and the re-anchored block is the TOP of the answer (checked below, at the
    // row the sort moved it to).
    check_equal(after_hits.size(), std::size_t{6},
                "every eligible row is ranked again after the rebind");
    check(after_hits[0].identity == identity_of_last, "and it is STILL the same block");
    check_equal(after_hits[0].row, new_row, "and it is now named at the row the sort moved it to");
    check_equal(column.rebind_count(), 1U, "rebind() is counted");
}

// ---------------------------------------------------------------------------
// section 10 -- the wire format
// ---------------------------------------------------------------------------
void section_10_wire() {
    SumDir directory(19);
    const std::uint32_t dim = 6;
    std::vector<std::size_t> rows;
    for (std::uint32_t i = 0; i < 5; ++i) {
        const std::size_t row = append_block(directory, 300 + i, i);
        directory.set_summary(row, "line " + std::to_string(i));
        rows.push_back(row);
    }
    SumDirVectorColumn column(dim, SumDirVectorMetric::Cosine, SumDirVectorMissing::SkipMissing,
                             SumDirVectorOffer::RequireEmbedding);
    column.set_top_k_default(3);
    for (std::uint32_t i = 0; i < 5; ++i) {
        std::vector<float> value(dim, 0.0f);
        value[i] = 1.0f;
        value[(i + 1) % dim] = 0.5f;
        column.set(directory, rows[i], value);
    }

    const std::vector<std::uint8_t> bytes = column.serialize(directory);
    check_equal(bytes.size(),
                static_cast<std::size_t>(ninfer::spec::sum_dir::vector_dir::kSumDirVectorHeaderBytes) +
                    rows.size() * ninfer::spec::sum_dir::vector_dir::kSumDirVectorSlotBytes +
                    rows.size() * dim * 4U,
                "the artefact is exactly header + slots + floats");

    SumDirVectorLoadReport report;
    const SumDirVectorColumn back = SumDirVectorColumn::deserialize(directory, bytes, report);
    check(report.header_ok, "the header verifies");
    check(report.payload_ok, "the payload verifies");
    check_equal(report.slots_read, static_cast<std::uint64_t>(rows.size()), "every slot came back");
    check_equal(report.dim, dim, "the width came back");
    check_equal(back.dim(), dim, "and it is the column's width");
    check_equal(static_cast<unsigned>(back.metric()),
                static_cast<unsigned>(SumDirVectorMetric::Cosine), "the metric came back");
    check_equal(static_cast<unsigned>(back.missing()),
                static_cast<unsigned>(SumDirVectorMissing::SkipMissing), "the missing rule came back");
    check_equal(static_cast<unsigned>(back.offer()),
                static_cast<unsigned>(SumDirVectorOffer::RequireEmbedding), "the offer rule came back");
    check_equal(back.top_k_default(), 3U, "the default top_k came back");
    check_equal(back.size(), column.size(), "the slot count came back");
    check_equal(back.stale_slots(directory), 0U, "and no slot is stale against its own directory");

    // Round trip is BYTE-EXACT, which is what makes the artefact checkable at all.
    const std::vector<std::uint8_t> again = back.serialize(directory);
    check_equal(again.size(), bytes.size(), "re-serialising is the same length");
    bool identical = again.size() == bytes.size();
    for (std::size_t i = 0; identical && i < bytes.size(); ++i) {
        if (again[i] != bytes[i]) { identical = false; }
    }
    check(identical, "re-serialising is byte-for-byte identical");

    // The ranking survives the round trip.
    std::vector<float> query(dim, 0.0f);
    query[2] = 1.0f;
    const std::vector<SumDirVectorHit> live_hits = column.search_top_k(directory, query, 5);
    const std::vector<SumDirVectorHit> back_hits = back.search_top_k(directory, query, 5);
    check_equal(back_hits.size(), live_hits.size(), "the round trip keeps the answer size");
    for (std::size_t i = 0; i < live_hits.size() && i < back_hits.size(); ++i) {
        check_equal(back_hits[i].row, live_hits[i].row, "and the same rows in the same order");
        check(back_hits[i].score == live_hits[i].score, "with the same scores");
    }

    // A TORN HEADER byte is caught by the header seal.
    {
        std::vector<std::uint8_t> torn = bytes;
        torn[20] ^= 0x01U; // inside the header, past the magic and the strategy bytes
        SumDirVectorLoadReport torn_report;
        (void)SumDirVectorColumn::deserialize(directory, torn, torn_report);
        check(!torn_report.header_ok, "a torn header byte does NOT verify");
        check(torn_report.error == "header digest mismatch", "and it is named");
    }
    // A TORN PAYLOAD byte is caught by the payload seal, before any slot is read.
    {
        std::vector<std::uint8_t> torn = bytes;
        torn[bytes.size() - 1] ^= 0x80U;
        SumDirVectorLoadReport torn_report;
        (void)SumDirVectorColumn::deserialize(directory, torn, torn_report);
        check(torn_report.header_ok, "the header is still fine");
        check(!torn_report.payload_ok, "the payload does NOT verify");
        check(torn_report.error == "payload digest mismatch", "and it is named");
    }
    // A column taken against a DIFFERENT directory is refused: the binding is in the artefact.
    {
        SumDir other(19);
        const std::size_t other_row = append_block(other, 4242, 0);
        other.set_summary(other_row, "a different world");
        SumDirVectorLoadReport other_report;
        const SumDirVectorColumn other_back =
            SumDirVectorColumn::deserialize(other, bytes, other_report);
        // `header_ok` is the 48-byte header SEAL's verdict and nothing more -- the same use
        // this file makes of it at the torn-header check and at the torn-payload one below.
        // The SEAL is intact here; what refuses is the BINDING, so `header_ok` stays true
        // while `error` names the refusal, and the load is "an empty column plus a report,
        // never a partial load" (sum_dir_vector.h:640-642): no slot comes back.
        check(other_report.header_ok, "the header seal of the artefact is intact");
        check_equal(other_back.size(), std::size_t{0},
                    "but a column from another directory loads no slot at all");
        check(other_report.error == "the vectors were taken against a different directory",
              "and the reason is the binding");
    }
    // Truncation, and a header whose own slot count disagrees with its byte count.
    {
        std::vector<std::uint8_t> cut(bytes.begin(), bytes.begin() + 60);
        SumDirVectorLoadReport cut_report;
        (void)SumDirVectorColumn::deserialize(directory, cut, cut_report);
        check(cut_report.header_ok, "the header of the cut artefact is intact");
        check(cut_report.truncated, "the truncation is REPORTED");
        check(!cut_report.payload_ok, "and nothing half-loads");
    }
}

} // namespace

int main() {
    section_1_knobs();
    section_2_identity_is_content();
    section_3_cosine();
    section_4_order_is_total();
    section_5_top_k();
    section_6_refusals();
    section_7_same_as_search_summaries();
    section_8_gates();
    section_9_sort_is_safe();
    section_10_wire();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_sum_dir_vector: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_sum_dir_vector: OK\n");
    return 0;
}
