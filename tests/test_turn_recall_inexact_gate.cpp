// tests/test_turn_recall_inexact_gate.cpp
//
// [INEXACTGATE] THE RECALL PATH'S REFUSALS, MADE AUDITABLE -- five separable sections, one per
// measured finding in dl/holegate/REPORT.md. Each section is RED-able by a NAMED mutation; the
// mutation for every check is written next to it, because a check whose failure mode nobody can
// name is not a check.
//
// WHAT THIS TEST IS AND IS NOT ALLOWED TO BE
//
//   * It is HOST-ONLY, std-only, and links no ninfer library and no CUDA -- deliberately, the
//     same shape (and the same reason) as tests/test_cold_slot_release_bytes.cpp: it must keep
//     building and running while the engine's device link does not. It includes exactly one
//     tree header, src/spec/turn_recall_journal.h, which its own comment declares host-only
//     ("no CUDA, no engine headers, unit-testable with plain g++", :90-91).
//   * The plan/cost/line half is EXECUTED against the real header.
//   * The ENGINE half (program_impl.h:14138-14174) cannot be linked here, so it is asserted
//     against the SOURCE TEXT, in the established shape of tests/test_fnv_convention.cpp and
//     tests/test_cold_slot_release_bytes.cpp ("reads the kernel header AS TEXT"). A text check
//     is only as good as its power to reject, so EVERY text check below is paired with a
//     NEGATIVE CONTROL on a synthetic broken copy of the same snippet.
//   * ⚠ It is NOT a behavioural test of the engine. Nothing here runs a round. The engine hunks
//     it inspects are asserted AS TEXT; what is EXECUTED is the plan/cost/line half. The two
//     policies are NAMED -- `RefuseRound` and `RefuseNotTruncate`, by the owner's ruling that at
//     a budget edge the tree must 「点名拒绝而不是截断」 (a named refusal, not a truncation) -- and
//     the constants that carry them are ASSERTED here, so reverting either to `Unnamed` reddens
//     this test rather than silently restoring the old behaviour. The two alternatives remain
//     exercised, because the §2.5 experiment in dl/holegate's LEAD must be runnable against both
//     arms by changing one token.
//
// THE FIVE SECTIONS, AND THE MUTATION THAT MAKES EACH GO RED
//
//   PA (finding 1: INEXACT is an observer; the engine's only call site computes it AFTER the
//       restore)      -- mutation: move the refusal branch below `recall_cold_pages_for_round`
//       in program_impl.h, or delete the branch's `return;`. ⭐ SECOND RED ARM, for the counter
//       choice: a `dropped_clamped`-only plan must NOT refuse, and a pure-tail hole (hole=1,
//       absent=1, cut_above_hole=0) MUST -- mutations: `return !plan.exact();` (fires on the
//       wrong counter) and gating on `cut_above_hole()` (misses the pure-tail case).
//   PB (finding 2: the signal is INVERTED -- a plan cut EMPTY prints nothing, a plan cut SHORT
//       prints INEXACT)  -- mutation: delete the `refused_empty` block from the empty-plan
//       branch, or gate it behind `if (recall_timing)` again.
//   PC (finding 3: the second gate is VACUOUS -- `net_positive()` compares two constants, so
//       `refused_cost=0` proves no check ran)      -- mutation: drop `++cost_gate_evals;`, or
//       make `verdict()` report `PositiveMeasured` for the shipped defaults.
//   PD (finding 4: `pages=R/P` is ANTI-CORRELATED with exactness) -- mutation: change
//       `cut_pages()` to return `dropped_hole` alone, or remove `admitted=%llu` from the
//       sequence-end format string.
//   PE (finding 5: the 227-page edge is a LOUD TRUNCATION and violates the tree's own
//       "refusing, not truncating" rule)         -- mutation: set `kRecallBudgetEdgePolicy` back
//       to `Unnamed`/`TruncateOldest` (the tree's default path is asserted to REFUSE), make
//       `RefuseNotTruncate` keep the pages, or stop recording the dropped range.

#include "spec/turn_recall_journal.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#if !defined(INEXACTGATE_SOURCE_DIR)
#if defined(NINFER_SOURCE_DIR)
#define INEXACTGATE_SOURCE_DIR NINFER_SOURCE_DIR
#else
#error "define INEXACTGATE_SOURCE_DIR to the tree root the text checks read"
#endif
#endif

namespace tr = ninfer::spec::turn_recall;

namespace {

unsigned failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}
void check_eq(std::uint64_t got, std::uint64_t want, const char* what) {
    if (got != want) {
        std::printf("FAIL: %s (got %llu, want %llu)\n", what,
                    static_cast<unsigned long long>(got), static_cast<unsigned long long>(want));
        ++failures;
    }
}
void check_text(bool ok, const char* what, const char* mutation) {
    if (!ok) {
        std::printf("FAIL: %s\n      the mutation this check rejects: %s\n", what, mutation);
        ++failures;
    }
}

// ---------------------------------------------------------------------------
// text helpers -- the engine half is read as TEXT (see the header comment)
// ---------------------------------------------------------------------------
std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::printf("FAIL: cannot read %s\n", path.c_str());
        ++failures;
        return std::string();
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
bool has(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}
// `a` occurs strictly before `b`. Both must be present: a check that passes because the
// needle it was looking for is GONE is the vacuous-check defect in a test's clothing.
bool has_before(const std::string& text, const std::string& a, const std::string& b) {
    const std::size_t ia = text.find(a);
    const std::size_t ib = text.find(b);
    return ia != std::string::npos && ib != std::string::npos && ia < ib;
}

const char* kProgramImpl =
    INEXACTGATE_SOURCE_DIR "/src/targets/qwen3_6/impl/runtime/program_impl.h";
const char* kProgramH = INEXACTGATE_SOURCE_DIR "/src/targets/qwen3_6/impl/runtime/program.h";
const char* kJournalH  = INEXACTGATE_SOURCE_DIR "/src/spec/turn_recall_journal.h";
const char* kReachH    = INEXACTGATE_SOURCE_DIR "/src/spec/sum_dir_reach.h";

// ===========================================================================
// PA -- finding 1: the refusal is NAMED, and it is BEFORE the restore
// ===========================================================================
void section_PA_the_refusal_is_named_and_ordered_before_the_restore() {
    using Defect = tr::RecallPagePlan::Defect;
    tr::RecallPagePlanRequest request{};
    request.frontier          = 8 * tr::kRecallPageTokens;
    request.page_tokens       = tr::kRecallPageTokens;
    request.page_bytes        = 1220608;  // D1's own bytes per page: the tree's own constant
                                          // (turn_recall_journal_test.cpp:70, kD1PageBytes)
    request.wanted_begin_page = 0;
    request.wanted_end_page   = 8;

    // The three cuts, and the name each one gets. `exact()` alone cannot tell them apart --
    // which is the whole reason the word INEXACT had no referent a reader could act on.
    const tr::RecallPagePlan holed = tr::plan_recall_pages(
        request, [](std::uint32_t page) { return page != 1; });
    const tr::RecallPagePlan clean = tr::plan_recall_pages(
        request, [](std::uint32_t) { return true; });
    tr::RecallPagePlanRequest tight = request;
    tight.byte_budget               = 2 * request.page_bytes;
    const tr::RecallPagePlan budgeted = tr::plan_recall_pages(
        tight, [](std::uint32_t) { return true; });
    tr::RecallPagePlanRequest ahead = request;
    ahead.wanted_end_page           = 12;
    const tr::RecallPagePlan clamped = tr::plan_recall_pages(
        ahead, [](std::uint32_t) { return true; });

    check_eq(static_cast<std::uint64_t>(holed.defect() == Defect::HoleCut), 1,
             "a hole names itself HoleCut");
    check_eq(static_cast<std::uint64_t>(budgeted.defect() == Defect::BudgetTruncated), 1,
             "a budget cut names itself BudgetTruncated");
    check_eq(static_cast<std::uint64_t>(clamped.defect() == Defect::FrontierClamped), 1,
             "a request past the frontier names itself FrontierClamped");
    check_eq(static_cast<std::uint64_t>(clean.defect() == Defect::None), 1,
             "an exact plan has no defect");
    check_eq(static_cast<std::uint64_t>(clean.exact()), 1, "and exact() is unchanged by the split");
    check(holed.inexact() && !clean.inexact(), "inexact() is the complement of exact()");

    // The SPELLING is pinned: the refusal line, the counters and this test must not drift apart.
    check(std::string(tr::recall_plan_defect_name(Defect::HoleCut)) == "hole-cut",
          "the defect's name is the string the line prints");
    check(std::string(tr::recall_plan_defect_name(Defect::BudgetTruncated)) == "budget-truncated",
          "budget-truncated");
    check(std::string(tr::recall_plan_defect_name(Defect::FrontierClamped)) == "frontier-clamped",
          "frontier-clamped");

    // WHICH COUNTERS REFUSE, by the tree's own rules, and WHY the third one does not:
    //   hole   -> N-1 of the dropped pages are LIVE and recallable, thrown away by the
    //             contiguity rule; the executed text is a short prefix of the wanted run.
    //   budget -> `lo += drop` drops the run's LOW END, where the selector's anchor sits, so
    //             sum_dir_reach.h:762-765's covered_* claim is false for what executes.
    //   clamped-> nothing that exists was dropped. THIS LINE'S READING, pinned so a reviewer can
    //             see it and overturn it: a reviewer who wants every inexact plan to refuse
    //             changes recall_plan_refuses_by_tree_rule() and nothing else.
    check(tr::recall_plan_refuses_by_tree_rule(Defect::HoleCut),
          "a holed plan refuses by the tree's own rules");
    check(tr::recall_plan_refuses_by_tree_rule(Defect::BudgetTruncated),
          "a truncated plan refuses by sum_dir_reach.h:257/:773's rule");
    check(!tr::recall_plan_refuses_by_tree_rule(Defect::FrontierClamped),
          "a clamped-only plan does NOT: nothing that exists was dropped");
    check(!tr::recall_plan_refuses_by_tree_rule(Defect::None), "and an exact plan never refuses");

    // ---- ⭐ THE NAMED POLICY, ASSERTED: a silent revert to `Unnamed` reddens this -------------
    check(tr::kInexactAdmissionPolicy == tr::InexactAdmissionPolicy::RefuseRound,
          "the owner's named admission policy is RefuseRound");
    check(tr::kRecallBudgetEdgePolicy == tr::BudgetEdgePolicy::RefuseNotTruncate,
          "and the budget edge's rule is the tree's own: refusing, not truncating");

    // ---- ⭐ THE WRONG-COUNTER RED ARMS ----------------------------------------------------
    // A gate that fires on the WRONG COUNTER is the same defect class these patches exist to fix,
    // so both directions are asserted here, and each has its own mutation named beside it.
    //
    // (i) `dropped_clamped > 0`, the other two zero: NOTHING THAT EXISTS WAS DROPPED -- the plan
    //     holds the whole run the committed frontier can serve -- so the round must EXECUTE.
    //     MUTATION that reddens this: `return !plan.exact();` in recall_plan_refuses_by_tree_rule.
    tr::RecallPagePlanRequest beyond = request;
    beyond.frontier                 = 3 * tr::kRecallPageTokens;  // only three pages committed
    const tr::RecallPagePlan clamped_only =
        tr::plan_recall_pages(beyond, [](std::uint32_t) { return true; });
    check_eq(clamped_only.dropped_clamped, 5, "the ask runs five pages past the frontier");
    check_eq(clamped_only.dropped_hole, 0, "and nothing was cut");
    check_eq(clamped_only.dropped_budget, 0, "and nothing was truncated");
    check_eq(clamped_only.count(), 3, "the whole servable run is held");
    check(!clamped_only.exact(), "the plan is still INEXACT, so the word is still printed");
    check(!tr::recall_plan_refuses_round(clamped_only, tr::InexactAdmissionPolicy::RefuseRound),
          "⛔ but RefuseRound must NOT fire: nothing that exists was dropped (the owner's ruling)");
    // The refusal sentence for it would name `frontier-clamped`; the round line names it and the
    // round still EXECUTES. Both halves are the reading the owner confirmed.
    check(std::string(tr::recall_plan_defect_name(clamped_only.defect())) == "frontier-clamped",
          "and the defect it reports is the clamped one, not a loss");

    // (ii) ⭐ THE PURE-TAIL HOLE: `dropped_hole = 1`, `dropped_absent = 1`, `cut_above_hole() = 0`.
    //      This is exactly the case a gate on `cut_above_hole() > 0` calls CLEAN, and it must
    //      refuse -- the run is one page short, and the missing page is the newest one.
    //      MUTATION that reddens this: gate on `cut_above_hole() != 0` instead of `dropped_hole`.
    const tr::RecallPagePlan pure_tail =
        tr::plan_recall_pages(request, [](std::uint32_t page) { return page != 7; });
    check_eq(pure_tail.dropped_hole, 1, "one page is cut");
    check_eq(pure_tail.dropped_absent, 1, "and it is the one that is GONE");
    check_eq(pure_tail.cut_above_hole(), 0, "nothing recoverable was thrown away");
    check_eq(pure_tail.count(), 7, "seven of the wanted eight are held");
    check(tr::recall_plan_refuses_by_tree_rule(pure_tail.defect()),
          "⛔ and it STILL refuses -- the assertion a `cut_above_hole` gate would miss");
    check(tr::recall_plan_refuses_round(pure_tail, tr::InexactAdmissionPolicy::RefuseRound),
          "so the round does not execute");

    // The policy dispatch is a pure function of the plan and the NAMED policy, so both policies
    // are exercisable here without editing the constant the engine asserts on.
    check(!tr::recall_plan_refuses_round(holed, tr::InexactAdmissionPolicy::ReportOnly),
          "ReportOnly: the round executes");
    check(tr::recall_plan_refuses_round(holed, tr::InexactAdmissionPolicy::RefuseRound),
          "RefuseRound: the round does not");
    check(!tr::recall_plan_refuses_round(clamped, tr::InexactAdmissionPolicy::RefuseRound),
          "and a clamped-only plan executes under BOTH policies (the reading above)");
    check(std::string(tr::inexact_admission_policy_name(
              tr::InexactAdmissionPolicy::Unnamed)) == "unnamed",
          "Unnamed is named as the ABSENCE of a policy, not as one of them");

    // The sentence a refusal prints, from the ONE producer of it.
    const std::string refusal = tr::recall_refusal_line(holed, tr::RecallCost{});
    check(has(refusal, "REFUSED-hole-cut"), "the refusal line names the defect");
    check(has(refusal, "wanted=8"), "and the run that was wanted");
    check(has(refusal, "executed=1"), "and how much of it executed");
    check(has(refusal, "recallable_but_cut=6"),
          "and how many LIVE pages the contiguity rule threw away");
    check(has(refusal, "INEXACT(never approximated)"),
          "and still ends with the word the fleet already greps for");

    // ---- THE ORDERING PROPERTY, WHICH IS THE POINT OF FINDING 1 -------------------------
    // holegate measured that the engine computes the word INSIDE the fprintf that reports a
    // round whose restore ALREADY HAPPENED (`program_impl.h:14166` acts, `:14169` speaks). The
    // checks below assert the refusal branch is BETWEEN the plan and the action, so under
    // RefuseRound the refusal FIRES; they are text checks because a round cannot be run here.
    const std::string prog = read_text_file(kProgramImpl);
    check(!prog.empty(), "program_impl.h was read");
    check_text(has_before(prog, "if (spec::turn_recall::recall_plan_refuses_round(",
                          "recall_cold_pages_for_round(sequence, plan);"),
               "the inexact refusal branch comes BEFORE the restore call",
               "move the branch below `recall_cold_pages_for_round(sequence, plan);`");
    // (Two needles rather than one long literal: the check must not depend on the continuation
    // line's exact indentation, and deleting the assert removes both needles at once.)
    check_text(has(prog, "kInexactAdmissionPolicy !=") &&
                   has(prog, "InexactAdmissionPolicy::Unnamed"),
               "the engine static_asserts the policy is NAMED (no silent default)",
               "delete the static_assert, so a policy can exist without anyone naming it");
    // NEGATIVE CONTROL for the ordering check: the same predicate must REJECT the pre-patch
    // shape (the word computed after the action) and the inverted shape.
    {
        const std::string before = "A_restore(plan);\nB_refuses(plan);\n";
        const std::string after  = "B_refuses(plan);\nA_restore(plan);\n";
        check(!has_before(before, "B_refuses(plan);", "A_restore(plan);"),
              "negative control: the restore-first shape is REJECTED");
        check(has_before(after, "B_refuses(plan);", "A_restore(plan);"),
              "negative control: the refusal-first shape is ACCEPTED");
    }
}

// ===========================================================================
// PB -- finding 2: the SILENT refusal names itself, and silence keeps its meaning
// ===========================================================================
void section_PB_the_silent_refusal_names_itself() {
    // The case that is refused-and-silent today: a hole at the FIRST wanted page. The plan is
    // empty AND carries the whole story (`hole=6 absent=1 cut_above_hole=5`) -- and the engine's
    // `if (plan.empty()) { return; }` prints none of it, while a SHORT plan prints INEXACT.
    tr::RecallPagePlanRequest request{};
    request.frontier          = 6 * tr::kRecallPageTokens;  // 6 committed pages, so the wanted
                                                            // run [0,6) is NOT clamped: this arm
                                                            // tests the HOLE, not the frontier
    request.page_tokens       = tr::kRecallPageTokens;
    request.page_bytes        = 4096;
    request.wanted_begin_page = 0;
    request.wanted_end_page   = 6;
    const tr::RecallPagePlan refused = tr::plan_recall_pages(
        request, [](std::uint32_t page) { return page != 0; });
    check(refused.empty(), "a hole at the first wanted page recalls nothing");
    check_eq(static_cast<std::uint64_t>(refused.defect() != tr::RecallPagePlan::Defect::None), 1,
             "and the empty plan STILL KNOWS WHY (this is the information that was printed as "
             "silence)");
    check_eq(refused.wanted_pages(), 6, "wanted=6");
    check_eq(refused.dropped_absent, 1, "one page is genuinely gone");
    check_eq(refused.cut_above_hole(), 5, "and five live ones were thrown away by contiguity");

    // A plan that is empty for a CLEAN reason (nothing was asked for) has no defect: that is
    // the genuine no-op, and it must stay silent. The two cases are now separable.
    tr::RecallPagePlanRequest nothing{};
    nothing.frontier    = 0;
    nothing.page_tokens = tr::kRecallPageTokens;
    const tr::RecallPagePlan noop = tr::plan_recall_pages(nothing,
                                                          [](std::uint32_t) { return true; });
    check(noop.empty() && noop.defect() == tr::RecallPagePlan::Defect::None,
          "an unasked-for round is empty with NO defect: silence stays reserved for it");

    const std::string refusal = tr::recall_refusal_line(refused, tr::RecallCost{});
    check(has(refusal, "REFUSED-hole-cut"), "the refused-empty line is named");
    check(has(refusal, "executed=0"), "and says nothing executed");

    const std::string prog = read_text_file(kProgramImpl);
    check_text(has(prog, "++turn_recall_counters.refused_empty;"),
               "the empty refusal is COUNTED (a countable form, not only a line)",
               "delete the refused_empty increment from the empty-plan branch");
    check_text(has_before(prog, "++turn_recall_counters.refused_empty;", "if (recall_timing) {"),
               "the refusal print is UNCONDITIONAL, before the timing-gated no-op line",
               "gate the whole refusal block behind `if (recall_timing)`, making it invisible "
               "in a default run again");
    check_text(has_before(prog, "if (plan.empty()) {", "++turn_recall_counters.refused_empty;"),
               "and it lives inside the empty-plan branch",
               "move it out of the empty branch");
    check_text(has(prog, "if (recall_timing) {"),
               "the timing-gated `plan=empty` line is still there for the CLEAN empty case",
               "delete it: the no-op case loses its diagnostic and silence stops meaning no-op");
    // ONE producer of the refusal sentence, so the two refusal sites cannot spell it two ways.
    const std::string head = read_text_file(kJournalH);
    std::size_t count = 0;
    for (std::size_t at = head.find("inline std::string recall_refusal_line(");
         at != std::string::npos;
         at = head.find("inline std::string recall_refusal_line(", at + 1)) {
        ++count;
    }
    check_eq(count, 1, "there is exactly ONE producer of the refusal sentence");
    {
        const std::string without = "";
        check(!has(without, "++turn_recall_counters.refused_empty;"),
              "negative control: the predicate does not find the marker in an empty copy");
    }
}

// ===========================================================================
// PC -- finding 3: the vacuous gate says what it decided
// ===========================================================================
void section_PC_the_cost_gate_says_what_it_decided() {
    using Verdict = tr::RecallCost::Verdict;
    tr::RecallCost cost{};
    check(cost.net_positive(), "net_positive() is unchanged: the shipped rates do beat re-prefill");
    check_eq(static_cast<std::uint64_t>(cost.verdict() == Verdict::PositiveDefaultConstant), 1,
             "and the verdict NAMES that it is the initialiser, not a measurement");
    check(cost.rate_gate_is_default_constant(), "the bool form agrees");

    // A rate that came from a measurement is a different state, and must not be reported as the
    // constant: this is the difference between "the gate ran and passed" and "no gate ran".
    tr::RecallCost measured{};
    measured.read_us_per_token = 2.6;
    check_eq(static_cast<std::uint64_t>(measured.verdict() == Verdict::PositiveMeasured), 1,
             "a measured rate reports PositiveMeasured");
    tr::RecallCost losing{};
    losing.read_us_per_token = 400.0;  // above the 365.0 re-prefill anchor
    check_eq(static_cast<std::uint64_t>(losing.verdict() == Verdict::NotPositive), 1,
             "a losing read reports NotPositive -- the branch the engine refuses on");
    check(!losing.net_positive(), "and net_positive() agrees with it");
    check(std::string(tr::recall_cost_verdict_name(Verdict::PositiveDefaultConstant)) ==
              "positive-default-const",
          "the verdict's printed name is pinned");

    const std::string prog = read_text_file(kProgramImpl);
    check_text(has(prog, "++turn_recall_counters.cost_gate_evals;"),
               "the engine counts CONSULTATIONS of the gate",
               "delete `++turn_recall_counters.cost_gate_evals;` -- then `refused_cost=0` cannot "
               "be told from 'no check ran', which is the finding");
    check_text(has_before(prog, "++turn_recall_counters.cost_gate_evals;", "++turn_recall_counters.refused_cost;"),
               "the count happens BEFORE the refusal branch, not inside it",
               "move it inside `if (... NotPositive)`, so a passing gate is not counted at all");
    const std::string counters = read_text_file(kProgramH);
    check_text(has(counters, "std::uint64_t cost_gate_evals    = 0;"),
               "the counter exists in TurnRecallCounters",
               "delete the counter: the summary line then has nothing to print");
    check_text(has(prog, "cost_gate=%llu"), "and it is PRINTED on the sequence-end line",
               "drop `cost_gate=%llu` from the format string");
    // The header's SECOND stated refusal ("a plan that would stream per token must be refused
    // outright") is not implemented anywhere. It is NAMED as a gap, so nobody reads the prose as
    // a check -- this is the check that the naming itself cannot be deleted quietly.
    const std::string head = read_text_file(kJournalH);
    check_text(has(head, "NAMED GAP -- the SECOND stated refusal above has no predicate"),
               "the unimplemented half of the cost contract is NAMED in the header",
               "delete the NAMED GAP comment, so the prose at :1278-1280 reads as implemented");
    {
        const std::string synthetic = "if (!cost.net_positive()) { return; }";
        check(!has(synthetic, "NAMED GAP"), "negative control: no gap marker in a clean copy");
    }
}

// ===========================================================================
// PD -- finding 4: the truth beside `pages=R/P`, and the identity that guards it
// ===========================================================================
void section_PD_the_truth_beside_the_ratio() {
    // THE INVARIANT, over a sweep, under EVERY edge policy. `count() + dropped_hole +
    // dropped_budget == admitted_pages()` holds whatever the policy does to the truncation --
    // it is the statement "no page is lost unaccounted for", and a future edit that drops a
    // page without a counter fails here instead of quietly lowering a ratio.
    const std::uint32_t frontiers[] = {0, 64, 640, 32064};
    const std::uint32_t begins[]    = {0, 1, 2, 5, 200};
    const std::uint32_t ends[]      = {0, 3, 8, 20, 501};
    const std::uint64_t budgets[]   = {0, 4096, 3 * 4096, 1ULL << 40};
    const tr::BudgetEdgePolicy policies[] = {
        tr::BudgetEdgePolicy::Unnamed, tr::BudgetEdgePolicy::RefuseNotTruncate,
        tr::BudgetEdgePolicy::TruncateOldest, tr::BudgetEdgePolicy::TruncateNewestKeepAnchor};
    unsigned cases = 0;
    unsigned bad_identity = 0, bad_clamp_identity = 0;
    for (std::uint32_t frontier : frontiers) {
        for (std::uint32_t begin : begins) {
            for (std::uint32_t end : ends) {
                for (std::uint64_t budget : budgets) {
                    for (int missing = -2; missing < 12; ++missing) {
                        for (tr::BudgetEdgePolicy policy : policies) {
                            tr::RecallPagePlanRequest request{};
                            request.frontier          = frontier;
                            request.page_tokens       = tr::kRecallPageTokens;
                            request.page_bytes        = 4096;
                            request.wanted_begin_page = begin;
                            request.wanted_end_page   = end;
                            request.byte_budget       = budget;
                            const tr::RecallPagePlan plan = tr::plan_recall_pages_with_policy(
                                request,
                                [missing](std::uint32_t page) {
                                    return static_cast<int>(page) != missing;
                                },
                                policy);
                            ++cases;
                            if (plan.count() + plan.dropped_hole + plan.dropped_budget !=
                                plan.admitted_pages()) {
                                ++bad_identity;
                            }
                            if (plan.admitted_pages() + plan.dropped_clamped !=
                                plan.wanted_pages()) {
                                ++bad_clamp_identity;
                            }
                        }
                    }
                }
            }
        }
    }
    std::printf("PD sweep: %u cases\n", cases);
    check_eq(bad_identity, 0,
             "count() + dropped_hole + dropped_budget == admitted_pages() -- no page is lost "
             "unaccounted for, under every edge policy");
    check_eq(bad_clamp_identity, 0,
             "admitted_pages() + dropped_clamped == wanted_pages() -- the ask and the loss are "
             "separate axes");
    check(cases > 4000, "the sweep actually ran a wide case set");

    // The D1 shape, and what the line now says about it. `pages=5/5` was the summary; the ROUND
    // line carried `hole=7 INEXACT`. After this patch the round line carries the reason, the
    // wanted run and the shortfall, and the summary's denominator is repeated beside the ratio.
    tr::RecallPagePlanRequest d1{};
    d1.frontier          = 8 * tr::kRecallPageTokens;
    d1.page_tokens       = tr::kRecallPageTokens;
    d1.page_bytes        = 1220608;  // kD1PageBytes (turn_recall_journal_test.cpp:70)
    d1.wanted_begin_page = 0;
    d1.wanted_end_page   = 8;
    const tr::RecallPagePlan p = tr::plan_recall_pages(
        d1, [](std::uint32_t page) { return page != 1; });
    check_eq(p.count(), 1, "the run collapses to page 0");
    check_eq(p.wanted_pages(), 8, "the line can now say the run WANTED 8 pages");
    check_eq(p.admitted_pages(), 8, "all 8 were servable by the frontier");
    check_eq(p.cut_pages(), 7, "and 7 were not held");
    tr::RecallCost cost{};
    cost.tokens = static_cast<std::uint64_t>(p.count()) * tr::kRecallPageTokens;
    const std::string line = tr::recall_line(p, cost, 1, 0);
    check(has(line, "defect=hole-cut"), "the round line names the defect");
    check(has(line, "wanted=8"), "the wanted run is on the line");
    check(has(line, "cut=7"), "and the shortfall");
    check(has(line, "INEXACT(never approximated)"),
          "the existing word survives BYTE-IDENTICAL, so every grep in the fleet still hits");
    check(has(line, "hole=7") && has(line, "absent=1") && has(line, "cut_above_hole=6"),
          "and so do the three numbers the reader already had");

    const std::string prog = read_text_file(kProgramImpl);
    // The ratio's own field is NOT redefined: it must still be `pages=%llu/%llu`, and the new
    // fields must come AFTER it. (Changing a printed field's meaning is the defect class this
    // patch exists to fix, so it is checked rather than trusted.)
    check_text(has(prog, "\"pages=%llu/%llu admitted=%llu cut=%llu clamped=%llu inexact=%llu \""),
               "`pages=R/P` keeps its bytes and the truth is ADDED beside it",
               "rewrite `pages=%llu/%llu` itself, e.g. to make the denominator the wanted count: "
               "that silently reinterprets every capture in the record");
    check_text(has(prog, "turn_recall_counters.pages_planned += plan.count();"),
               "pages_planned still accumulates the CUT plan (its meaning is unchanged)",
               "change pages_planned to accumulate the wanted count: the old line's meaning "
               "changes under every reader who already parsed it");
    {
        // negative control
        const std::string broken = "\"pages=%llu/%llu\"";  // nothing added
        check(!has(broken, "admitted=%llu"),
              "negative control: a line without the added fields is REJECTED by the check");
    }
}

// ===========================================================================
// PE -- finding 5: the budget edge, refuse-not-truncate, and the dropped range
// ===========================================================================
void section_PE_the_budget_edge_refuses_rather_than_truncating() {
    // indexfan's measured shape, reproduced from the tree's own default budget: 256 MiB /
    // 1,181,696 B per page = 227 pages; a wanted run of [158, 501) = 343 pages; so 116 are
    // dropped -- FROM THE LOW END, i.e. the anchor at 158 is the first thing thrown away.
    tr::RecallPagePlanRequest request{};
    request.frontier          = 501ULL * tr::kRecallPageTokens;
    request.page_tokens       = tr::kRecallPageTokens;
    request.page_bytes        = 1181696;
    request.wanted_begin_page = 158;
    request.wanted_end_page   = 501;
    request.byte_budget       = 256ULL << 20;  // the engine's own default (program.h:1042)
    const auto live = [](std::uint32_t) { return true; };

    const tr::RecallPagePlan oldest = tr::plan_recall_pages_with_policy(
        request, live, tr::BudgetEdgePolicy::TruncateOldest);
    check_eq(oldest.count(), 227, "227 pages fit the 256 MiB budget (ARITHMETIC, reproduced)");
    check_eq(oldest.dropped_budget, 116, "and 116 are dropped");
    check_eq(oldest.dropped_budget_begin_page, 158, "the dropped range STARTS AT THE ANCHOR");
    check_eq(oldest.dropped_budget_end_page, 274, "and ends where the kept run begins");
    check(oldest.pages.front() == 274, "the kept run begins at 274");
    check(std::find(oldest.pages.begin(), oldest.pages.end(), 158) == oldest.pages.end(),
          "so the anchor page 158 is NOT in the plan -- the measured `anchor in plan = NO`");
    tr::RecallCost cost{};
    cost.tokens = static_cast<std::uint64_t>(oldest.count()) * tr::kRecallPageTokens;
    check(has(tr::recall_line(oldest, cost, 0, 0), "budget_dropped=[158,274)"),
          "and the line names the dropped range, so a reader can compare it with the "
          "selector's own first_page=");

    // The tree's rule, at the same site: sum_dir_reach.h:257/:773 refuses. `pages = 0` there,
    // an empty plan here -- the same statement, so ROW 2's budget can inherit it.
    const tr::RecallPagePlan refused = tr::plan_recall_pages_with_policy(
        request, live, tr::BudgetEdgePolicy::RefuseNotTruncate);
    check(refused.empty(), "RefuseNotTruncate selects NOTHING (never a partial injection)");
    // ⚠ 116 -> 343: THE REFUSAL HOLDS NOTHING, SO IT DROPPED THE WHOLE RUN (line `redkv`).
    // `dropped_budget` is what the byte-budget edge removed from the plan; for TruncateOldest that
    // is the 116 oldest pages, for a WHOLE-RUN refusal it is all 343 (`wanted=343`, `executed=0`
    // on the refusal line two statements down, and `cut_pages() == 343`). The begin page is
    // unchanged, because the oldest page is still where the loss starts.
    check_eq(refused.dropped_budget, 343, "and still says how much could not be held");
    check_eq(refused.dropped_budget_begin_page, 158, "and which range that was");
    check_eq(refused.dropped_budget_end_page, 501, "and that the range is the whole asked run");
    check(refused.defect() == tr::RecallPagePlan::Defect::BudgetTruncated,
          "and names the defect");
    check(tr::recall_plan_refuses_by_tree_rule(refused.defect()),
          "and it is a refusing defect by the tree's rule");
    const std::string refusal = tr::recall_refusal_line(refused, tr::RecallCost{});
    check(has(refusal, "REFUSED-budget-truncated"), "the refusal names the edge");
    check(has(refusal, "wanted=343"), "and the run that could not be held");
    check(has(refusal, "executed=0"), "and that nothing was injected");

    // The third candidate -- keep the anchor -- is expressible too, and is STILL a truncation:
    // it does not falsify sum_dir_reach.h:762-765's claim, and sum_dir_reach.h would still
    // refuse it. The test says both, so nobody reads this option as the tree's rule.
    const tr::RecallPagePlan keep = tr::plan_recall_pages_with_policy(
        request, live, tr::BudgetEdgePolicy::TruncateNewestKeepAnchor);
    check_eq(keep.count(), 227, "keeping the anchor holds the same 227 pages");
    check(keep.pages.front() == 158, "and page 158 -- the anchor -- is now IN the plan");
    check(!keep.exact(), "and it is STILL inexact: this option truncates too");

    // `Unnamed` is not a policy: its interim arithmetic is today's. Called EXPLICITLY now, because
    // the tree's own default is no longer it -- and asserted, so the interim cannot drift unseen.
    const tr::RecallPagePlan unnamed = tr::plan_recall_pages_with_policy(
        request, live, tr::BudgetEdgePolicy::Unnamed);
    check_eq(unnamed.dropped_budget_begin_page, oldest.dropped_budget_begin_page,
             "Unnamed keeps today's arithmetic (documented as an interim, not a choice)");
    check_eq(unnamed.pages.front(), oldest.pages.front(), "same kept run as TruncateOldest");

    // ⭐ AND THE TREE'S DEFAULT NOW REFUSES. This IS policy E at the engine's call site: where
    // `plan_round_recall` used to return a 227-page partial injection, `plan_recall_pages` now
    // returns an EMPTY plan that names the defect, the amount and the dropped range -- so the
    // round becomes a named refusal (and, with patch B, a printed one) instead of an injection
    // whose anchor is missing. MUTATION that reddens this: set `kRecallBudgetEdgePolicy` back to
    // `Unnamed` or `TruncateOldest`.
    const tr::RecallPagePlan tree_default = tr::plan_recall_pages(request, live);
    check(tree_default.empty(), "the tree's default path REFUSES the 227-page partial run");
    check(tree_default.defect() == tr::RecallPagePlan::Defect::BudgetTruncated,
          "and names it budget-truncated");
    check_eq(tree_default.dropped_budget, 343,
             "the amount is still reported, and it is the whole run this time");
    check_eq(tree_default.dropped_budget_begin_page, 158,
             "and the dropped range starts at the anchor, so the refusal is auditable against "
             "the selector's own first_page=");
    check(tr::recall_plan_refuses_round(tree_default, tr::kInexactAdmissionPolicy),
          "and under the named admission policy this round does not execute");
    check(std::string(tr::recall_budget_edge_policy_name(
              tr::BudgetEdgePolicy::RefuseNotTruncate)) == "refuse-not-truncate",
          "the policy's name is pinned");

    // The rule must be NAMED where a reader of the header will find it, with the coordinate of
    // the layer it comes from -- dl/vectorkey ROW 2 takes its edge from THERE.
    // ⚠ line `redkv`: THE NEEDLE USED TO BE ONE OF THE TREE'S TWO SPELLINGS AND THE HEADER USES
    // THE OTHER. "Refusing rather than truncating" is the wording inside the refusal SENTENCE
    // (sum_dir_reach.h:861); the header cites the ENUMERATOR's own comment instead --
    // sum_dir_reach.h:257, which reads "refusing, not truncating" -- and that is the coordinate
    // the header also names. The check is now TWO-SIDED, which is strictly stronger than the
    // needle it replaces: the citing header must name the coordinate AND the rule, AND the file
    // AT that coordinate must still say it. A citation that survives the deletion of its source
    // is the failure this check exists for, and the old shape could not see it.
    const std::string head = read_text_file(kJournalH);
    check_text(has(head, "sum_dir_reach.h:257"), "the header names the rule's source coordinate",
               "delete the citation, so the next budget is modelled on `lo += drop` by default");
    check_text((has(head, "refusing, not truncating") || has(head, "Refusing rather than truncating")),
               "and the header names the rule itself, in one of the tree's own spellings",
               "delete the rule's name, keeping only a bare line number");
    {
        const std::string reach = read_text_file(kReachH);
        check_text(has(reach, "refusing, not truncating") || has(reach, "Refusing rather than truncating"),
                   "and the coordinate it cites still says it: the citation is not a relic",
                   "delete the rule from sum_dir_reach.h itself, so the citing header quotes a "
                   "file that no longer states it");
    }
    const std::string prog = read_text_file(kProgramImpl);
    check_text(has(prog, "spec::turn_recall::kRecallBudgetEdgePolicy !="),
               "the engine static_asserts that the edge policy is NAMED",
               "delete the static_assert: the tree's own rule stays violated in silence");
    {
        const std::string broken = "plan.dropped_budget = drop; lo += drop;";
        check(!has(broken, "dropped_budget_begin_page"),
              "negative control: the pre-patch truncation is REJECTED by the range check");
    }
}

// ---------------------------------------------------------------------------------------------
// PF (finding 6, added by line `recallfix2`): `pages_restored` had TWO UNITS on THREE sites, so
// `pages=R/P` mixed them on one line and the per-round `[recall-index-path]` line subtracted a
// page count from a round count. Each check below is RED-able by a NAMED mutation, and each one
// has a negative control on a synthetic snippet, because a text check that cannot reject is not
// a check.
//
// THE MUTATIONS THAT MAKE THIS SECTION GO RED
//   (h) put `++turn_recall_counters.pages_restored;` back at either round-counting arm
//   (i) put `index_path_join_line(plan, turn_recall_counters.pages_restored - plan.count())`
//       back at the round's census (the subtraction that underflowed to 2^64-1 on round one)
//   (j) print `plan.count()` where the MEASURED delta belongs -- i.e. force planned == restored
//   (k) drop `restored_short=%llu short_pages=%llu` from the sequence-end format string
// ---------------------------------------------------------------------------------------------
void section_PF_the_restore_tally_is_pages_and_a_short_restore_is_named() {
    const std::string prog = read_text_file(kProgramImpl);

    // (a) NO SITE COUNTS ROUNDS ANY MORE. This is the defect itself: two arms incremented once
    //     per round while the third added pages, so the counter's value was a mixture.
    check_text(!has(prog, "++turn_recall_counters.pages_restored;"),
               "no restore arm increments `pages_restored` once per ROUND any more",
               "(h) restore `++turn_recall_counters.pages_restored;` at the append or rebuild "
               "arm: the counter goes back to being a round count on a page-count field");
    check_text(!has(prog, "recall_cold_pages_for_round(sequence, plan);\n    ++turn_recall_counters.pages_restored;"),
               "and the round's census does not advance it per round either",
               "(h') add the per-round `++` back at the census");

    // (b) EACH ARM ADDS THE PAGES IT ACTUALLY BROUGHT BACK, and the number is one the arm has
    //     IN HAND -- the append arm prints it as `blocks=` two lines above, the rebuild arm as
    //     `blocks=`. A fix that added a constant would satisfy (a) and still be wrong.
    check_text(has(prog, "turn_recall_counters.pages_restored += blocks_read;"),
               "the text-cargo append arm adds the blocks it really read (its own `blocks=`)",
               "(b) replace `blocks_read` with `1`, or with `plan.count()`: the first restores "
               "the defect, the second forces planned == restored");
    check_text(has(prog, "turn_recall_counters.pages_restored += run.positions.size();"),
               "the context-rebuild arm adds the blocks it really read back",
               "(b') make the rebuild arm add 1, as it used to");
    check_text(has(prog, "turn_recall_counters.pages_restored += restored;"),
               "and the KV read-back arm's per-PAGE add is UNCHANGED (it was already right)",
               "(b'') change this one to `++`: the arm that was correct is now the broken one");

    // (c) THE ROUND'S OWN LINE IS THIS ROUND'S MEASURED DELTA, and the subtraction is gone.
    //     ⚠ THE NEEDLE IS THE WHOLE CALL AND NOT THE BARE SUBTRACTION. A bare needle for the
    //     subtraction alone is FOOLED BY THIS PATCH'S OWN COMMENT, which quotes the expression it
    //     removes -- measured on the corrected tree: the bare form failed there. (The same
    //     self-inflicted defect is recorded twice in this fleet: dl/prefillland/REPORT.md sec.2.3
    //     and dl/combo1m2/REPORT.md sec.5.4.) The needle below is a full C++ call, which no
    //     comment can quote without the comment becoming that call.
    check_text(!has(prog, "index_path_join_line(plan, turn_recall_counters.pages_restored - plan.count())"),
               "the `cumulative - this_rounds_plan` subtraction is GONE from the join line",
               "(i) restore the subtraction as the join line's third argument -- on round one that "
               "is 0 - N and prints 2^64-1");
    check_text(has(prog, "pages_restored - plan.count()"),
               "and the corrected tree still QUOTES the expression in the comment that explains "
               "it, so the check above had to be the whole call",
               "(this one's mutation: delete the comment that quotes the expression, which is "
               "what makes a bare needle unusable here and therefore worth pinning)");
    check_text(has(prog, "index_path_join_line(plan, restore_delta.pages)"),
               "and it is handed the round's MEASURED restored page count",
               "(i') hand it `plan.count()`: the line then cannot show a short restore");
    check_text(has_before(prog, "const std::uint64_t restored_before = turn_recall_counters.pages_restored;",
                          "recall_cold_pages_for_round(sequence, plan);"),
               "because the measurement is taken ACROSS the restore call, not guessed",
               "(c) move the snapshot below the call: the delta is then always 0");

    // (d) ⭐ THE FORBIDDEN READING -- planned == restored BY CONSTRUCTION -- IS NOT PRESENT.
    check_text(!has(prog, "recall_line(plan, cost, plan.count(), 0)"),
               "the per-round line is NOT handed the plan's own count as `restored=`",
               "(j) pass `plan.count()`: every run would print restored == pages and the field "
               "would be unreadable in the direction that matters");

    // (e) A SHORT RESTORE IS NAMED ON ITS OWN LINE AND COUNTED, so the ratio is not the only
    //     place the fact lives. Both are additive: no existing field moves.
    check_text(has(prog, "recall_restore_delta("),
               "the round's comparison of planned against restored goes through the one "
               "constexpr helper (spec/turn_recall_journal.h), so a test can execute it",
               "(e) inline the comparison in the engine again: the executed check loses its "
               "subject and the `always-equal` mutation becomes invisible");
    check_text(has(prog, "\"[recall] SHORT-RESTORE planned=%llu restored=%llu short=%llu \""),
               "the engine prints a NAMED line when a round brings back fewer than it planned",
               "(e') delete the SHORT-RESTORE fprintf: the event goes back to being a silent "
               "difference between two numbers on a line");
    check_text(has(prog, "restored_short=%llu short_pages=%llu"),
               "and the sequence-end line carries both counters",
               "(k) remove the two appended fields: the run-level fact becomes unreadable");
    check_text(has(prog, "turn_recall_counters.pages_planned += plan.count();"),
               "while `pages_planned` keeps its own meaning byte-for-byte (the CUT plan)",
               "change pages_planned's accumulator: `pages=R/P`'s denominator moves");

    // (f) THE HEADER NAMES THE RULE, so the next reader finds it where the counter lives.
    const std::string head = read_text_file(kJournalH);
    check_text(has(head, "struct RecallRestoreDelta {") && has(head, "short_rounds"),
               "the header defines the delta with the shortfall fields",
               "delete `short_rounds`: a short restore becomes uncountable");
    check_text(has(head, "it does NOT return the planned count as the"),
               "and it states OUT LOUD that it does not force planned == restored",
               "delete the sentence: the next editor has no way to know that `restored = "
               "planned` is the forbidden fix");

    // (g) NEGATIVE CONTROLS on synthetic snippets: the needles must be able to REJECT.
    {
        const std::string broken_round_count = "        ++turn_recall_counters.pages_restored;\n";
        check(!has(broken_round_count, "+= blocks_read"),
              "negative control: the old round-counting line does NOT satisfy the per-page check");
        const std::string broken_sub = "index_path_join_line(plan, turn_recall_counters.pages_restored - plan.count())";
        check(!has(broken_sub, "restore_delta.pages"),
              "negative control: the underflowing subtraction does NOT satisfy the delta check");
        const std::string broken_equal = "recall_line(plan, cost, plan.count(), 0)";
        check(!has(broken_equal, "restore_delta.pages"),
              "negative control: the forced-equal call does NOT satisfy the delta check");
        const std::string short_snip = "\"[recall] SHORT-RESTORE planned=%llu restored=%llu short=%llu \"";
        check(has(short_snip, "SHORT-RESTORE"),
              "positive control: the SHORT-RESTORE needle DOES match its own text");
    }
}

}  // namespace

int main() {
    section_PA_the_refusal_is_named_and_ordered_before_the_restore();
    section_PB_the_silent_refusal_names_itself();
    section_PC_the_cost_gate_says_what_it_decided();
    section_PD_the_truth_beside_the_ratio();
    section_PE_the_budget_edge_refuses_rather_than_truncating();
    section_PF_the_restore_tally_is_pages_and_a_short_restore_is_named();
    if (failures == 0) {
        std::printf("ALL CHECKS PASSED\n");
        return 0;
    }
    std::printf("%u FAILURES\n", failures);
    return 1;
}
