// Host-only test for the SLIDER'S INSPECTION ENTRY -- `--kv-score-table show|emit=<path>`
// (product/kv_kv_bits.h kv_score_table_run) -- and for the score-table/ladder size
// invariant that entry shares with product/kv_bit_budget.h kv_bit_budget_scored_ladder.
//
// WHY THIS TEST EXISTS. Both of those loops used to be bounded by the LADDER's size (8)
// while indexing the SCORE TABLE (then 6 rows). Two out-of-bounds reads per call. Measured on
// the pinned binary 6e08e846fce9377e:
//   * `ninfer --kv-score-table show`  -> rc=139 (SIGSEGV)
//   * `ninfer --kv-score-table emit=` -> rc=139
//   * `ninfer --kv-score-table bogus` -> rc=2
// i.e. ONLY THE ERROR PATH WORKED, and the documented no-model inspection entry for the
// speed/quality slider crashed precisely when it succeeded. That is why it survived: no
// test ever reached the success path, and the error path is the one everybody exercises.
//   * -D_GLIBCXX_ASSERTIONS named it exactly:
//       std::array<ninfer::product::KvTierScoreRow, 6>::operator[]:
//       Assertion '__n < this->size()' failed.
//   * the compiler had already said so, as a warning, for as long as it existed:
//       warning: iteration 6 invokes undefined behavior [-Waggressive-loop-optimizations]
//
// ⚠ THE TABLE GREW 6 -> 8, AND THE TWO EXTRA ROWS ARE NOW PART OF THE GRAMMAR (line `redkv`,
// dl/redkv/TRIAGE.md #54). The old out-of-bounds read was closed by making the SCORE TABLE
// cover every SELECTABLE ladder row: product/kv_bit_budget.h:1059
// `static_assert(detail::scores_cover_every_selectable_gear(), ...)` loops from
// KvTierScoreTable{}.size() to kKvBitBudgetTiers.size() and REFUSES TO COMPILE if an uncovered
// row is selectable. rk3v4/rk2v4 became selectable in the same item that grew the table
// (dl/e8mixwire/land/patch_flip_batch.py, one item, eight clusters; kv_bit_budget.h:1027-1032
// spells out that the TYPE had to grow in that same edit "or the header does not compile").
// So the OOB regression is now structural, and the two assertions below were written against
// the pre-flip shape of it:
//
// THE ASSERTIONS THAT MATTER HERE ARE STILL THE ONES THAT WOULD FAIL ON THE OLD CODE:
//   * the rendered table names EXACTLY ONE ROW PER LADDER ROW, in ladder order -- the old loop
//     emitted eight lines with the last two read from memory past a six-row table;
//   * the scored ladder covers EVERY score-table row -- the old loop's bound was the ladder's,
//     so a table row the ladder's bound did not reach would keep its declared penalty;
//   * a MUTATED input table is rendered as mutated, row 6 included, so a renderer that ignored
//     its input (or always printed the built-in default) cannot pass this file.
//
// Build note: this test is compiled with -D_GLIBCXX_ASSERTIONS in the standalone invocation
// recorded in the report, which turns any out-of-bounds std::array index into an abort
// rather than a silent read. Both flags are used by the report's invocation.

#include "product/kv_kv_bits.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::product::kKvBitBudgetTiers;
using ninfer::product::KvTierScoreRow;
using ninfer::product::KvTierScoreTable;
using ninfer::product::kv_bit_budget_default_scores;
using ninfer::product::kv_bit_budget_parse_scores;
using ninfer::product::kv_bit_budget_scored_ladder;
using ninfer::product::kv_bits_score_table_text;
using ninfer::product::kv_score_table_run;

int g_failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) { return; }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

// Non-comment lines of the rendered table.
std::vector<std::string> data_lines(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) { line.pop_back(); }
        if (line.empty() || line[0] == '#') { continue; }
        out.push_back(line);
    }
    return out;
}

std::string first_field(const std::string& line) {
    std::istringstream in(line);
    std::string name;
    in >> name;
    return name;
}

// ------------------------------------------------------------- the success path
void test_show_path_renders_one_row_per_ladder_row() {
    const KvTierScoreTable table = kv_bit_budget_default_scores();
    std::ostringstream out;
    std::ostringstream info;
    std::string error;

    // THE ENTRY'S SUCCESS PATH. This call is what used to segfault (rc=139).
    const bool ok = kv_score_table_run("show", "", &error, out, info);
    check(ok, "kv_score_table_run(\"show\") returns true on its success path");
    check(error.empty(), "the success path reports no error");
    check(!out.str().empty(), "the success path wrote a table");

    const std::vector<std::string> rows = data_lines(out.str());
    // THE STRUCTURAL FACT the OOB read used to violate: the score table is not a PREFIX of the
    // ladder any more, it IS the ladder's length (kv_bit_budget.h:1059 makes this a compile-time
    // requirement, "or the header does not compile").
    check(KvTierScoreTable{}.size() == kKvBitBudgetTiers.size(),
          "the score table covers the WHOLE ladder (" +
              std::to_string(KvTierScoreTable{}.size()) + " vs " +
              std::to_string(kKvBitBudgetTiers.size()) + ")");
    check(rows.size() == kKvBitBudgetTiers.size(),
          "the rendered table has exactly one row per ladder row (" +
              std::to_string(rows.size()) + ", want " +
              std::to_string(kKvBitBudgetTiers.size()) + ")");
    // The regression assertion for the OOB read, in the direction the gate flip made true: the
    // old loop ran to the LADDER's size while indexing a six-row table, so the two rows it
    // printed last were built from memory past the end. They are rows of the grammar now, and
    // the property to pin is that they are rendered FROM THE TABLE (the loop below checks their
    // columns) rather than that they are absent.
    for (const char* extra : {"rk3v4", "rk2v4"}) {
        bool named = false;
        for (const std::string& row : rows) {
            if (first_field(row) == extra) { named = true; }
        }
        check(named, std::string("the rendered table names the ladder row '") + extra +
                         "' (it IS a row of the score grammar since the gate opened)");
    }
    // Every ladder row is named, in ladder order, with the TABLE's own numbers -- including the
    // two rows the OOB read used to invent.
    for (std::size_t i = 0; i < rows.size() && i < kKvBitBudgetTiers.size(); ++i) {
        check(first_field(rows[i]) == kKvBitBudgetTiers[i].spec_name,
              "row " + std::to_string(i) + " names the ladder's row '" +
                  kKvBitBudgetTiers[i].spec_name + "', got '" + first_field(rows[i]) + "'");
        check(rows[i].find(std::to_string(table[i].quality_x100)) != std::string::npos &&
                  rows[i].find(std::to_string(table[i].speed_x100)) != std::string::npos,
              "row " + std::to_string(i) + " carries the table's own columns: '" + rows[i] + "'");
    }
    // ... and NONE beyond it: a renderer that still ran to some stale larger bound would emit a
    // row whose first field is not a ladder row's name.
    for (const std::string& row : rows) {
        bool in_ladder = false;
        for (const auto& tier : kKvBitBudgetTiers) {
            if (first_field(row) == tier.spec_name) { in_ladder = true; }
        }
        check(in_ladder, "every rendered row is a ladder row, got '" + row + "'");
    }
}

// A CONTROL THAT CAN FAIL: a renderer that ignored its input, or that always printed the
// built-in default, must not survive this. The mutated values are chosen so they cannot
// appear by accident.
void test_render_is_driven_by_its_input() {
    KvTierScoreTable mutated = kv_bit_budget_default_scores();
    mutated[0] = KvTierScoreRow{111, 222};   // bf16
    mutated[3] = KvTierScoreRow{333, 444};   // nvfp4
    mutated[6] = KvTierScoreRow{555, 666};   // rk3v4 -- the row the OOB read used to invent
    const std::string text = kv_bits_score_table_text(mutated, "unit test");
    const std::vector<std::string> rows = data_lines(text);
    check(rows.size() == kKvBitBudgetTiers.size(),
          "the mutated table still renders one row per ladder row");
    if (rows.size() >= 7) {
        check(rows[0].find("111") != std::string::npos && rows[0].find("222") != std::string::npos,
              "the renderer used the MUTATED bf16 row, got '" + rows[0] + "'");
        check(rows[3].find("333") != std::string::npos && rows[3].find("444") != std::string::npos,
              "the renderer used the MUTATED nvfp4 row, got '" + rows[3] + "'");
        // ⭐ AND THE ROW THAT IS ONLY IN THE TABLE SINCE THE GATE OPENED. A renderer that printed
        // the built-in default for rows 6/7 -- or that read them from a stale bound -- cannot
        // carry these two numbers.
        check(rows[6].find("555") != std::string::npos && rows[6].find("666") != std::string::npos,
              "the renderer used the MUTATED rk3v4 row, got '" + rows[6] + "'");
    }
    // ... and the provenance line it was handed is the one it prints.
    check(text.find("unit test") != std::string::npos,
          "the renderer prints the provenance it was handed");
}

// ------------------------------------------------------------- round-trip
void test_round_trip_through_the_real_loader() {
    const KvTierScoreTable table = kv_bit_budget_default_scores();
    const std::string text = kv_bits_score_table_text(table, "round trip");
    const KvTierScoreTable parsed = kv_bit_budget_parse_scores(text);
    for (std::size_t i = 0; i < table.size(); ++i) {
        check(parsed[i].quality_x100 == table[i].quality_x100 &&
                  parsed[i].speed_x100 == table[i].speed_x100,
              "round trip preserves row " + std::to_string(i));
    }
}

void test_emit_path_writes_a_loadable_file() {
    const std::string path = "/tmp/ninfer_budgetsat_score_table_test.txt";
    std::ostringstream out;
    std::ostringstream info;
    std::string error;
    const bool ok = kv_score_table_run("emit=" + path, "", &error, out, info);
    check(ok, std::string("kv_score_table_run(\"emit=\") returns true: ") + error);
    std::ifstream in(path);
    check(in.good(), "the emitted file exists");
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    check(data_lines(text).size() == kKvBitBudgetTiers.size(),
          "the emitted file has one data row per ladder row");
    // The loader must accept it: this is the contract the entry's doc comment states.
    const KvTierScoreTable parsed = kv_bit_budget_parse_scores(text);
    const KvTierScoreTable table  = kv_bit_budget_default_scores();
    check(parsed[4].quality_x100 == table[4].quality_x100,
          "the emitted file reloads through kv_bit_budget_parse_scores");
    std::remove(path.c_str());
}

// The error path must still work -- it is the half that was always green, kept here so a
// future fix cannot trade the success path for it.
void test_error_path_still_refuses() {
    std::ostringstream out;
    std::ostringstream info;
    std::string error;
    const bool ok = kv_score_table_run("bogus", "", &error, out, info);
    check(!ok, "an unknown mode is refused");
    check(error.find("show") != std::string::npos && error.find("emit=") != std::string::npos,
          "the refusal names both accepted modes, got: " + error);
    check(out.str().empty(), "the refusal writes no table");
}

// ------------------------------------------------- the shared size invariant
// This is the invariant the header now pins with a static_assert; asserted here as a
// RUNNING property too, so the test reports it even where the static_assert is compiled out
// by a stale build artifact.
void test_score_table_covers_every_selectable_gear() {
    // The same property the header pins with static_assert(scores_cover_every_selectable_gear())
    // -- asserted here as a RUNNING property too, so this file reports the invariant even where a
    // stale build artifact compiled the static_assert out. It is NOT vacuous while the sizes are
    // equal: the loop below walks EVERY ladder row, and it fails if any selectable row is not a
    // row of the score grammar (the state that made --kv-score-table show segfault).
    check(KvTierScoreTable{}.size() <= kKvBitBudgetTiers.size(),
          "the score table is a prefix of the ladder");
    for (std::size_t i = 0; i < kKvBitBudgetTiers.size(); ++i) {
        const bool scored = i < KvTierScoreTable{}.size();
        if (kKvBitBudgetTiers[i].selectable) {
            check(scored, std::string("selectable ladder row '") +
                              kKvBitBudgetTiers[i].spec_name +
                              "' must be a row of the score grammar, or the DP could choose a "
                              "gear with no score");
        }
    }
    // ... and the REVERSE non-emptiness, so the assertion above cannot be satisfied by an empty
    // table: at least one selectable row exists and IS scored.
    std::size_t selectable = 0;
    for (const auto& tier : kKvBitBudgetTiers) {
        if (tier.selectable) { ++selectable; }
    }
    check(selectable > 0 && selectable == KvTierScoreTable{}.size(),
          "every ladder row is selectable and every one of them is scored (" +
              std::to_string(selectable) + " / " + std::to_string(KvTierScoreTable{}.size()) + ")");
}

// ------------------------------------------------- the scored ladder's tail
void test_scored_ladder_tail_is_not_read_past_the_end() {
    const KvTierScoreTable table = kv_bit_budget_default_scores();
    for (double w = 0.0; w <= 1.0 + 1e-9; w += 0.125) {
        const auto ladder = kv_bit_budget_scored_ladder(table, w);
        check(ladder.size() == kKvBitBudgetTiers.size(), "the scored ladder is ladder-sized");
        for (std::size_t i = 0; i < table.size(); ++i) {
            const auto want = static_cast<std::int32_t>(
                std::nearbyint(w * table[i].quality_x100 + (1.0 - w) * table[i].speed_x100));
            check(ladder[i].penalty_x100 == (want < 0 ? 0 : want),
                  "row " + std::to_string(i) + " at W=" + std::to_string(w) +
                      " is the weighted combination, got " +
                      std::to_string(ladder[i].penalty_x100) + " want " +
                      std::to_string(want < 0 ? 0 : want));
        }
        // The OOB regression assertion, re-derived for the equal-size shape. The old loop took
        // the LADDER's bound while indexing the SCORE table, so a row the table did not reach
        // kept its declared penalty instead of the table's value. Two rows were in that state;
        // the count is zero now, so "there is no row past the table" must be asserted DIRECTLY
        // -- otherwise the assertion this function is named for would be an empty loop.
        check(table.size() == ladder.size(),
              "no ladder row is past the score table (so no row can keep a stale penalty)");
        for (std::size_t i = table.size(); i < ladder.size(); ++i) {
            check(ladder[i].penalty_x100 == kKvBitBudgetTiers[i].penalty_x100,
                  "row " + std::to_string(i) + " (past the score table) keeps its declared "
                  "penalty " + std::to_string(kKvBitBudgetTiers[i].penalty_x100) + ", got " +
                      std::to_string(ladder[i].penalty_x100));
        }
    }
    // ⭐ AND EVERY ROW IS DRIVEN BY THE TABLE IT WAS HANDED. A BOUND that stopped short of the
    // table's last rows would leave those rows at their declared penalty; a bound that ran past
    // the table would read past the end. Both are caught by perturbing ONE row at a time and
    // requiring the scored ladder to follow it.
    for (std::size_t i = 0; i < table.size(); ++i) {
        KvTierScoreTable one = table;
        one[i] = KvTierScoreRow{0, static_cast<std::int32_t>(4000 + i)};
        const auto ladder = kv_bit_budget_scored_ladder(one, /*quality_weight=*/0.0);
        check(ladder[i].penalty_x100 == static_cast<std::int32_t>(4000 + i),
              "row " + std::to_string(i) + " follows the table it was handed, got " +
                  std::to_string(ladder[i].penalty_x100));
    }
}

}  // namespace

int main() {
    std::printf("=== test_kv_score_table_entry: the slider's inspection entry ===\n");
    std::printf("KvTierScoreTable size = %zu, ladder size = %zu\n", KvTierScoreTable{}.size(),
                kKvBitBudgetTiers.size());
    test_show_path_renders_one_row_per_ladder_row();
    test_render_is_driven_by_its_input();
    test_round_trip_through_the_real_loader();
    test_emit_path_writes_a_loadable_file();
    test_error_path_still_refuses();
    test_score_table_covers_every_selectable_gear();
    test_scored_ladder_tail_is_not_read_past_the_end();
    if (g_failures == 0) {
        std::printf("PASS: the success path renders one row per ladder row from the table it \
was handed, round-trips, and no row is read past the score table\n");
        return 0;
    }
    std::printf("%d FAILURE(S)\n", g_failures);
    return 1;
}
