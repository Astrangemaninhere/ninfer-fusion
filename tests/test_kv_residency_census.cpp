// test_kv_residency_census.cpp -- the standing properties of the residency census instrument
// (product/kv_residency_census.h).  HOST ONLY, header only; no engine, no device, no model.
//
// WHAT THIS TEST IS FOR. An instrument that cannot say NO is not an instrument. Every property
// below has a red half: the fixture is driven in a way that MUST make the check fail, and the
// check is asserted to have failed. The three properties that matter are:
//   1. `saved` is NEVER 0-as-unknown -- before the baseline exists it must be std::nullopt, and the
//      printed line must name the designed denominator instead of printing a number;
//   2. a mean with a zero denominator is UNDEFINED BY NAME, not 0;
//   3. the counters are monotone: no sequence of observations can decrease one, so a negative
//      reading is unrepresentable rather than merely unexpected.
#include "product/kv_residency_census.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

using ninfer::product::kKvResidencyUnobtained;
using ninfer::product::KvResidencyCensus;
using ninfer::product::kv_residency_census_can_answer_saved;
using ninfer::product::kv_residency_census_denominator;
using ninfer::product::kv_residency_census_line;
using ninfer::product::kv_residency_mean_resident_pages;
using ninfer::product::kv_residency_observe_round;
using ninfer::product::kv_residency_saved_pages;

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) { return; }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// A three-round fixture with distinct shape, so a mean that dropped the denominator cannot pass.
KvResidencyCensus fixture() {
    KvResidencyCensus c;
    kv_residency_observe_round(c, 220, 55, 3, 220, 64);
    kv_residency_observe_round(c, 200, 50, 1, 200, 64);
    kv_residency_observe_round(c,  80, 20, 0,  80, 32);
    return c;
}

// ---------------------------------------------------------------- property 1: saved is not 0
void test_saved_is_unobtained_not_zero() {
    KvResidencyCensus c = fixture();
    check(c.rounds == 3, "three rounds observed (" + std::to_string(c.rounds) + ")");
    check(c.resident_pages_total == 500, "resident total is the SUM of the rounds, got " +
                                             std::to_string(c.resident_pages_total));
    check(c.resident_blocks_total == 125, "blocks are counted in their OWN column, got " +
                                              std::to_string(c.resident_blocks_total));
    check(!kv_residency_census_can_answer_saved(c),
          "with no baseline the census must NOT be able to answer 'saved'");
    const std::string line = kv_residency_census_line(c);
    check(contains(line, kKvResidencyUnobtained),
          "the line must carry the word " + std::string(kKvResidencyUnobtained) + " when saved is "
          "un-obtainable, got: " + line);
    check(contains(line, "baseline_resident_pages_total"),
          "the blank must NAME the reading it needs, got: " + line);
    check(contains(line, "designed denom=rounds_x_pages_per_round=3x80"),
          "the blank must carry its designed denominator, got: " + line);
    check(!contains(line, "saved_pages=0 "),
          "saved must never be printed as a bare 0 while it is unknown, got: " + line);

    // Now supply the baseline: the number becomes obtainable and is the DIFFERENCE, in pages.
    c.baseline_resident_pages_total = 612;
    const auto saved = kv_residency_saved_pages(c);
    check(saved.has_value(), "with a baseline the census must answer");
    check(saved.value_or(0) == 112, "saved = baseline - observed = 612 - 500, got " +
                                        std::to_string(saved.value_or(0)));
    check(kv_residency_census_can_answer_saved(c), "and it must say so");
    check(contains(kv_residency_census_line(c), "saved_pages=112"),
          "the line must print the number it can now defend");

    // THE RED HALF: a baseline BELOW what was observed is not a saving, and must be refused
    // rather than reported as a negative or as a small positive.
    c.baseline_resident_pages_total = 400;
    check(!kv_residency_saved_pages(c).has_value(),
          "a baseline below the observed total must be REFUSED, not reported");
    check(contains(kv_residency_census_line(c), kKvResidencyUnobtained),
          "and the line falls back to " + std::string(kKvResidencyUnobtained));
}

// ---------------------------------------------------------------- property 2: denominators
void test_denominators_are_not_folded_in() {
    const KvResidencyCensus empty;
    check(kv_residency_census_denominator(empty) == 0, "no rounds, no denominator");
    check(!kv_residency_mean_resident_pages(empty).has_value(),
          "a mean with a zero denominator is UNDEFINED BY NAME, never 0");
    const std::string line = kv_residency_census_line(empty);
    check(contains(line, "mean_resident_pages_per_round=" + std::string(kKvResidencyUnobtained)),
          "the empty census must print " + std::string(kKvResidencyUnobtained) + " for the mean");
    check(contains(line, "denom=rounds, rounds=0"),
          "and the zero denominator must be printed beside it, got: " + line);

    KvResidencyCensus c = fixture();
    const auto mean = kv_residency_mean_resident_pages(c);
    check(mean.has_value(), "with rounds the mean exists");
    // 500/3 -- a value only reachable if the divisor is the ROUND count, not the page count.
    check(std::fabs(*mean - 500.0 / 3.0) < 1e-12,
          "the mean's denominator is `rounds` (3), got " + std::to_string(*mean));

    // RED HALF: a mean that folded in the page total instead of the round count would be 1.0 here.
    const double wrong = static_cast<double>(c.resident_pages_total) /
                         static_cast<double>(c.resident_pages_total);
    check(!(std::fabs(*mean - wrong) < 1e-12),
          "the mean must NOT be the page-total identity (1.0): a dropped denominator is catchable");
}

// ---------------------------------------------------------------- property 3: monotone
void test_counters_cannot_go_negative() {
    KvResidencyCensus c;
    // A round that retires MORE than is resident, and a later round with a SMALLER residency:
    // neither may pull a total down.
    kv_residency_observe_round(c, 100, 25, 40, 100, 8);
    const std::uint64_t r1 = c.resident_pages_total, t1 = c.retired_pages_total;
    kv_residency_observe_round(c, 0, 0, 0, 0, 0);
    check(c.resident_pages_total >= r1, "resident total is monotone non-decreasing");
    check(c.retired_pages_total >= t1, "retired total is monotone non-decreasing");
    check(c.rounds_with_no_residency == 1,
          "a zero-resident round is COUNTED as a shape, got " +
              std::to_string(c.rounds_with_no_residency));
    check(c.pages_per_round_last == 0 && c.rounds == 2,
          "the shape column tracks the LAST round while the totals keep the whole history");
    // There is no API that subtracts: the type cannot express a negative counter.
    check(c.resident_pages_total == 100 && c.retired_pages_total == 40,
          "the observations are sums, got resident=" + std::to_string(c.resident_pages_total));
}

// ---------------------------------------------------------------- the three candidate-set readings
// The candidate-set seam cannot change residency on today's wiring. That is a READING (0 pages on a
// stated denominator), and this test pins the distinction so it cannot be re-read as "un-measured".
void test_todays_candidate_seam_saves_no_resident_pages() {
    KvResidencyCensus c;
    // The same run, twice: once with the semantic channel OFF, once ON. Both report the SAME
    // resident pages, because the union never changes the fetch range (semchan_wire.h:167-170).
    for (int r = 0; r < 4; ++r) { kv_residency_observe_round(c, 64, 16, 0, 64, 8); }
    c.baseline_resident_pages_total = 256; // the OFF arm's total, supplied by the engine
    const auto saved = kv_residency_saved_pages(c);
    check(saved.has_value(), "the reading is obtainable once the baseline is supplied");
    check(saved.value_or(1) == 0, "saved == 0 pages is a READING with a denominator, got " +
                                      std::to_string(saved.value_or(1)));
    const std::string line = kv_residency_census_line(c);
    check(contains(line, "saved_pages=0 denom=rounds_x_pages_per_round=4x64"),
          "the 0 must travel WITH its denominator (4 rounds x 64 pages), got: " + line);
    check(!contains(line, kKvResidencyUnobtained),
          "and it must NOT be labelled " + std::string(kKvResidencyUnobtained) +
          ": a measured 0 and an un-measured value are different statements");
}

} // namespace

int main() {
    std::printf("=== test_kv_residency_census: the 'how much KV did we save' instrument ===\n");
    test_saved_is_unobtained_not_zero();
    test_denominators_are_not_folded_in();
    test_counters_cannot_go_negative();
    test_todays_candidate_seam_saves_no_resident_pages();
    if (g_failures == 0) {
        std::printf("PASS: 未取得 is never printed as 0; the mean prints its denominator; the\n"
                    "      counters are monotone; and a measured 0 carries its denominator.\n");
        return 0;
    }
    std::printf("%d FAILURE(S)\n", g_failures);
    return 1;
}
