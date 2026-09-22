// tests/test_recall_restore_units.cpp
//
// [RESTOREUNITS] `turn_recall_counters.pages_restored` HAD ONE NAME AND TWO UNITS, and this file
// is the check that makes the fix falsifiable.
//
// WHAT WAS WRONG, in the engine that shipped:
//   * program_impl.h -- the text-cargo append arm did `++pages_restored;` ONCE PER ROUND, while
//     the KV read-back arm did `+= restored;` ONCE PER PAGE. The context-rebuild arm did `++`.
//     Every recall run on this engine takes the append arm (NINFER_RECALL_TEXT=1 selects the
//     text-cargo path), so a run that appended 8 pages reported 6.
//   * the same number was the NUMERATOR of `pages=R/P` while the denominator, `pages_planned`,
//     accumulates `plan.count()` -- a page count. Two units on one line.
//   * the per-round `[recall-index-path]` line was handed
//     `pages_restored - plan.count()`: a CUMULATIVE counter minus THIS round's page count,
//     evaluated before the first round had advanced the counter, which is where the observed
//     `restored=18446744073709551615` (2^64 - 1 = 0 - 2) came from.
//
// WHAT THIS FILE CHECKS, AND WHY IT IS NOT A TEXT CHECK
//   The comparison "did this round bring back what its plan asked for" now lives in ONE constexpr
//   function in src/spec/turn_recall_journal.h -- `recall_restore_delta` -- and this file
//   EXECUTES it. A text check could only assert that the engine contains the right sentence; this
//   one runs the arithmetic that decides, so the mutation that matters (make the function return
//   the PLANNED count, i.e. force planned == restored) turns it RED without anyone having to edit
//   this file.
//
// THE RED ARMS (each named, each a one-line change to the header, none of them in this file)
//   (J) `d.pages = restored_pages;`  ->  `d.pages = planned_pages;`
//       THE FORBIDDEN FIX. It makes every run read as complete. Reddens checks 2, 4 and 5 below.
//   (K) `if (restored_pages != planned_pages)`  ->  `if (false)`
//       The shortfall stops being counted. Reddens checks 2, 3 and 4.
//   (L) `d.short_pages = planned_pages > restored_pages ? planned_pages - restored_pages : 0;`
//       ->  `d.short_pages = 0;`
//       The mismatch is counted as a round but not measured. Reddens checks 2 and 4.
//
// SHAPE: host-only, std-only, one tree header, no ninfer library, no CUDA -- the same shape as
// tests/test_turn_recall_inexact_gate.cpp and tests/test_cold_slot_release_bytes.cpp, so it keeps
// building and running while the engine's device link does not.
//
//   g++-13 -std=c++20 -O2 -I <repo>/src tests/test_recall_restore_units.cpp -o /tmp/t && /tmp/t
//
// It is NOT registered in tests/CMakeLists.txt: that file is held by another line and is off
// limits for this one. Registering it is a NAMED owed step (README.md section 5).

#include "spec/turn_recall_journal.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>

namespace {

using namespace ninfer::spec::turn_recall;

int failures = 0;
int checks   = 0;

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

void check_eq(std::uint64_t got, std::uint64_t want, const std::string& what) {
    ++checks;
    if (got != want) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s: got %llu want %llu\n", what.c_str(),
                     static_cast<unsigned long long>(got),
                     static_cast<unsigned long long>(want));
    }
}

// 1. A COMPLETE ROUND IS STILL COMPLETE, AND THE UNIT IS PAGES. The plan asked for two pages and
//    two came back: `pages` is 2 (a page count, not the number of rounds), and nothing is flagged.
//    RED-arm (J) reddens this only indirectly; its direct red is check 2.
void test_complete_round() {
    const RecallRestoreDelta d = recall_restore_delta(2, 2);
    check_eq(d.pages, 2, "a complete round reports 2 PAGES (not 1 round)");
    check_eq(d.short_rounds, 0, "and it is not flagged");
    check_eq(d.short_pages, 0, "and its deficit is 0");
}

// 2. ⭐ THE CASE THE COUNTER EXISTS FOR: a plan of 2 pages that restored 1. This is the shape the
//    engine really produces -- the append arm `break`s on a page whose directory row is dead or
//    whose cargo record is missing, and the read-back arm `continue`s on a page that is not
//    compressed -- so the check is not a hypothetical.
//    RED-arm (J): `d.pages = planned_pages` makes this report 2 and the first check fails.
//    RED-arm (K): the `!=` test removed makes `short_rounds` 0 and the second check fails.
//    RED-arm (L): the deficit hard-set to 0 makes the third check fail.
void test_short_round_is_counted() {
    const RecallRestoreDelta d = recall_restore_delta(2, 1);
    check_eq(d.pages, 1, "a round that restored 1 of its plan's 2 pages reports 1 page");
    check_eq(d.short_rounds, 1, "and it is flagged as a round that fell short");
    check_eq(d.short_pages, 1, "and the deficit is measured: 1 page");
    check(d.pages != 2, "the restored count is NOT the planned count (the forbidden fix)");
}

// 3. THE EMPTY APPEND ARM. The arm returns with `REASON=no-cargo-record-for-planned-pages` and
//    appends nothing while the plan was non-empty; the counter must show 0 restored, not 1 round
//    and not the plan's size.
void test_total_loss_is_counted() {
    const RecallRestoreDelta d = recall_restore_delta(2, 0);
    check_eq(d.pages, 0, "a round that brought back nothing reports 0 pages");
    check_eq(d.short_rounds, 1, "and it is flagged");
    check_eq(d.short_pages, 2, "and the deficit is the whole plan");
}

// 4. THE RUN-LEVEL TALLY, driven round by round exactly as the engine drives it: the ratio on the
//    sequence-end line cannot carry this fact, which is why the two counters exist.
//    RED-arm (J) reddens the third round's `pages`; (K)/(L) redden the totals.
void test_a_run_accumulates() {
    RecallRestoreTally t{};
    std::uint64_t pages = 0;

    const RecallRestoreDelta r1 = recall_restore_delta(2, 2);  // complete
    const RecallRestoreDelta r2 = recall_restore_delta(1, 1);  // complete
    const RecallRestoreDelta r3 = recall_restore_delta(2, 1);  // fell short by 1
    for (const RecallRestoreDelta& r : {r1, r2, r3}) {
        pages += r.pages;
        t = recall_restore_tally_add(t, r);
    }

    check_eq(pages, 4, "the run restored 4 PAGES over 3 rounds (4, not 3)");
    check_eq(t.short_rounds, 1, "one of the three rounds fell short");
    check_eq(t.short_pages, 1, "and the run is short by exactly 1 page");
    // The identical run as the OLD counter would have recorded it: one per round. The two numbers
    // differ, which is the whole defect.
    check(pages != 3, "4 pages over 3 rounds is NOT the round count, and the old counter said 3");
}

// 5. A MISMATCH IN THE OTHER DIRECTION IS NOT SMOOTHED AWAY EITHER. The engine cannot over-restore
//    today, but a counter that clamps is a counter that cannot report its own unit drifting again,
//    so the function counts the round and reports the restored number truthfully.
void test_over_report_is_not_smoothed() {
    const RecallRestoreDelta d = recall_restore_delta(1, 2);
    check_eq(d.pages, 2, "an over-reported round reports the number it was handed, unclamped");
    check_eq(d.short_rounds, 1, "and it is flagged as a mismatch");
    check_eq(d.short_pages, 0, "with no deficit (nothing was lost)");
    check(d.pages != 1, "and it is NOT clamped down to the planned count");
}

// 6. THE UNIT IS PAGES FOR EVERY ARM SHAPE THE ENGINE HAS, in one table. The three arms are the
//    text-cargo append arm (`blocks_read`), the context-rebuild arm (`run.positions.size()`) and
//    the KV read-back arm (`restored`) -- all three now hand this function a PAGE count, and the
//    check pins that the function's meaning does not depend on which arm produced the number.
void test_the_unit_does_not_depend_on_the_arm() {
    struct Row {
        std::uint64_t planned;
        std::uint64_t restored;
        std::uint64_t short_rounds;
        std::uint64_t short_pages;
    };
    const Row table[] = {
        {0, 0, 0, 0},  // nothing planned, nothing restored: not a shortfall
        {2, 2, 0, 0},  // append arm, complete
        {2, 1, 1, 1},  // append arm, one dead row cut the loop
        {2, 0, 1, 2},  // append arm, every row dead
        {8, 8, 0, 0},  // read-back arm, complete
        {8, 3, 1, 5},  // read-back arm, five pages not compressed
        {4, 4, 0, 0},  // rebuild arm, complete
        {4, 2, 1, 2},  // rebuild arm, the run stopped early
    };
    for (const Row& r : table) {
        const RecallRestoreDelta d = recall_restore_delta(r.planned, r.restored);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "planned=%llu restored=%llu (any arm) -> pages/short_rounds/short_pages",
                      static_cast<unsigned long long>(r.planned),
                      static_cast<unsigned long long>(r.restored));
        check_eq(d.pages, r.restored, std::string(buf) + " : pages");
        check_eq(d.short_rounds, r.short_rounds, std::string(buf) + " : short_rounds");
        check_eq(d.short_pages, r.short_pages, std::string(buf) + " : short_pages");
    }
}

}  // namespace

int main() {
    test_complete_round();
    test_short_round_is_counted();
    test_total_loss_is_counted();
    test_a_run_accumulates();
    test_over_report_is_not_smoothed();
    test_the_unit_does_not_depend_on_the_arm();
    if (failures == 0) {
        std::printf("ALL CHECKS PASSED (%d checks)\n", checks);
        return 0;
    }
    std::printf("%d FAILURES of %d checks\n", failures, checks);
    return 1;
}
