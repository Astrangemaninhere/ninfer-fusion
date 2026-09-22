// ===========================================================================
// tests/test_recall_prefill_budget.cpp
// ===========================================================================
// THE BOUNDED PREFILL BUDGET, ITS EDGE, AND ITS NAME.
//
// WHY THIS FILE EXISTS. The tree had NO speed limit on recall and its own comment says so
// ("a MEMORY-SAFETY limit rather than a speed limit", program_impl.h, NINFER_TURN_RECALL_BYTES);
// it also had NO knob in the dimension the cost is actually in -- measured, before this landing:
// `grep -rn -E 'recall-prefill-tokens|RECALL_PREFILL_TOKENS|recall_prefill_tokens' src apps tests`
// returned ZERO hits. dl/vectorkey/REPORT.md ROW 2 is the specification and its edge has ONE
// admissible form, stated at :44 and again at :282-291:
//
//     a NAMED REFUSAL -- `refused-prefill-budget` -- on the plan, BEFORE any re-prefill,
//     "refusing not truncating", "never a partial injection".
//
// WHAT IS EXECUTED HERE (card-free, host-only, std-only, no GPU):
//   * the POSITIVE arm   -- a run that fits the budget plans EXACTLY the run, element for element;
//   * the CONTROL arm    -- budget 0 (the flag absent) reproduces the un-bounded plan bit for bit,
//                          which is what makes the landing behaviour-preserving by construction;
//   * the RED arm        -- a run that cannot fit is REFUSED WHOLESALE, named, empty, and
//                          `recall_refusal_line` carries `refused-prefill-budget`;
//   * a NEGATIVE CONTROL on a synthetic broken copy, so no check above can pass vacuously.
//
// The reference shape is dl/sweepprep's `PASS_IDS_EXACT` -- one frozen id sequence compared
// ELEMENT FOR ELEMENT, with the fleet's verdict names -- and the card-free floor of that test is
// the INPUT identity, which is what `BLOCKS` below holds (dl/vectorkey/REPORT.md:390-396, :668-671).
//
// ⛔ IT MUST NOT, and does not: assert that a plan SUCCEEDS when it cannot be held; land a default;
//    or claim the model's own output. The 17 ids are a frozen INPUT identity, not a decode.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "spec/turn_recall_journal.h"

namespace tr = ninfer::spec::turn_recall;

namespace {

int g_checks  = 0;
int g_fail    = 0;

void check(bool ok, const std::string& what, const std::string& mutation) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::printf("FAIL  %s\n      mutation: %s\n", what.c_str(), mutation.c_str());
    }
}

template <typename A, typename B>
void check_eq(const A& got, const B& want, const std::string& what, const std::string& mutation) {
    ++g_checks;
    if (!(got == want)) {
        ++g_fail;
        std::printf("FAIL  %s\n      mutation: %s\n", what.c_str(), mutation.c_str());
    }
}

bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

// ===========================================================================
// THE FROZEN REFERENCE (dl/sweepprep's PASS_IDS_EXACT shape, 17 ids, id 28 twice)
// ===========================================================================
// This is the CARD-FREE floor of that check: the block's content identity is a function of these
// ids alone, so they are a frozen INPUT and comparing them element for element is a check that can
// fail. The decoded-output clause of PASS_IDS_EXACT is NOT claimed here -- it needs the card, and
// this line runs without one.
const std::uint32_t kReferenceIds[17] = {
    842u, 2080u, 887u, 28u, 19u, 24u, 18u, 23u, 16u, 22u, 26u, 24614u, 28u, 8023u, 33u, 34643u,
    248046u};

// The plan's own frozen parameters: the page is 64 tokens (kRecallPageTokens), and the proposed
// budget is 512 tokens per recall = 8 pages (dl/vectorkey/REPORT.md:252-253), which is ALSO the
// tree's live `recall_fanout_blocks()` default -- so the budget widens no admission.
constexpr std::uint32_t kPageTokens        = 64u;
constexpr std::uint64_t kProposedBudget    = 512u;   // 8 pages
constexpr std::uint32_t kPagesForBudget    = 8u;
// 256 committed pages, so the 227-page arm below is a run the frontier CAN serve: this test is
// about the BUDGET, and a clamp would put a second cause on the same plan object.
constexpr std::uint32_t kFrontier          = 64u * 256u;

// A page is live everywhere: this test is about the BUDGET, so the store is not the variable.
bool all_live(std::uint32_t) { return true; }

tr::RecallPagePlanRequest make_request(std::uint32_t begin_page, std::uint32_t end_page,
                                       std::uint64_t token_budget) {
    tr::RecallPagePlanRequest request{};
    request.frontier          = kFrontier;
    request.page_tokens       = kPageTokens;
    request.wanted_begin_page = begin_page;
    request.wanted_end_page   = end_page;
    request.page_bytes        = 4096u;   // non-zero, so the byte budget is a live dimension too
    request.byte_budget       = 256ULL << 20;
    request.token_budget      = token_budget;
    return request;
}

std::vector<std::uint32_t> pages_of(const tr::RecallPagePlan& plan) { return plan.pages; }

// ===========================================================================
// A -- THE POSITIVE ARM: the frozen run fits, and is held exactly
// ===========================================================================
void section_A_positive() {
    const tr::RecallPagePlan plan =
        tr::plan_recall_pages(make_request(0u, kPagesForBudget, kProposedBudget), all_live);

    check_eq(plan.count(), kPagesForBudget, "A: an 8-page run inside a 512-token budget is held",
             "clamp or drop pages the budget can afford");
    check(!plan.refused_prefill_budget, "A: nothing was refused",
          "refuse a run that fits the budget");
    check(plan.exact(), "A: and the plan is EXACT", "leave a drop counter set on a clean run");
    check(plan.defect() == tr::RecallPagePlan::Defect::None, "A: no defect is reported",
          "report a defect on a clean run");
    check_eq(plan.prefill_tokens_wanted, static_cast<std::uint64_t>(kProposedBudget),
             "A: 8 pages x 64 tokens is exactly the budget", "count pages instead of tokens");
    check_eq(plan.prefill_token_budget, kProposedBudget, "A: the budget is echoed",
             "drop the budget from the plan");

    // ELEMENT FOR ELEMENT, ascending -- the PASS_IDS_EXACT shape applied to the page list.
    const std::vector<std::uint32_t> want = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
    check_eq(pages_of(plan), want, "A: the pages are the frozen run, element for element",
             "return pages in any order other than ascending, or substitute one by position");

    std::printf("A  plan pages=");
    for (std::uint32_t p : plan.pages) { std::printf("%u ", p); }
    std::printf(" tokens=%llu budget=%llu exact=%d defect=%s\n",
                static_cast<unsigned long long>(plan.prefill_tokens_wanted),
                static_cast<unsigned long long>(plan.prefill_token_budget),
                plan.exact() ? 1 : 0, tr::recall_plan_defect_name(plan.defect()));
}

// ===========================================================================
// B -- THE CONTROL ARM: no budget == today's behaviour, bit for bit
// ===========================================================================
// The flag's default is 0 and 0 means unbounded, so the landing is behaviour-preserving BY
// CONSTRUCTION and this arm is what proves it: the same request with `token_budget = 0` must
// produce the SAME plan object as the positive arm, and a run far past the budget must still be
// held in full.
void section_B_control() {
    const tr::RecallPagePlan unbounded =
        tr::plan_recall_pages(make_request(0u, kPagesForBudget, 0u), all_live);
    const tr::RecallPagePlan bounded =
        tr::plan_recall_pages(make_request(0u, kPagesForBudget, kProposedBudget), all_live);
    check_eq(pages_of(unbounded), pages_of(bounded),
             "B: budget 0 and budget 512 give the SAME plan for a run that fits",
             "make the zero budget refuse or reshape the plan");
    check_eq(unbounded.prefill_tokens_wanted, static_cast<std::uint64_t>(kProposedBudget),
             "B: the run is still measured when no budget is set",
             "skip the measurement when the budget is 0: the line then cannot say what happened");

    // A HARSHER CONTROL, and it is the one that matters: the same 227-page run the tree's 256 MiB
    // byte budget allows (227 x 64 = 14,528 tokens) is UNBOUNDED when no token budget is set --
    // i.e. tonight's baseline behaviour is untouched by this landing.
    const tr::RecallPagePlan today =
        tr::plan_recall_pages(make_request(0u, 227u, 0u), all_live);
    check_eq(today.count(), 227u, "B: without the flag the 227-page run still plans in full",
             "let an unset budget cut the 227-page run");
    check(!today.refused_prefill_budget, "B: and it is not refused",
          "refuse when no budget was named");
    check(today.exact(), "B: and it is exact", "set a drop counter with no budget to drop for");
    std::printf("B  unbounded 227-page run: pages=%u refused=%d defect=%s\n", today.count(),
                today.refused_prefill_budget ? 1 : 0,
                tr::recall_plan_defect_name(today.defect()));
}

// ===========================================================================
// C -- THE RED ARM: the edge is a REFUSAL, and it is BY NAME
// ===========================================================================
// "refusing not truncating -- never a partial injection" (dl/vectorkey/REPORT.md:44). A 9-page run
// against a 512-token budget must be refused WHOLESALE: no page, no token, the whole run named.
void section_C_the_edge_is_refusal() {
    const tr::RecallPagePlan refused =
        tr::plan_recall_pages(make_request(0u, kPagesForBudget + 1u, kProposedBudget), all_live);

    check(refused.refused_prefill_budget, "C: a 9-page run against a 512-token budget is REFUSED",
          "truncate the run to fit and return the remaining pages");
    check_eq(refused.count(), 0u, "C: and NOT truncated -- no page is injected",
             "lo += drop (or hi -= drop): that is a PARTIAL injection of the answer's context");
    check(refused.empty(), "C: the plan is empty", "return a non-empty refused plan");
    check_eq(refused.prefill_tokens_wanted, 9u * kPageTokens,
             "C: the refusal records the run it refused, in TOKENS",
             "record only the excess, or record pages instead of tokens");
    check_eq(refused.prefill_token_budget, kProposedBudget, "C: and the budget that refused it",
             "omit the budget: the refusal stops being arithmetic");
    check(refused.defect() == tr::RecallPagePlan::Defect::PrefillBudget,
          "C: the defect is PrefillBudget", "name the byte budget's defect instead");
    check(!refused.exact(), "C: a refused plan is NOT exact",
             "leave exact() true on a plan that holds nothing");

    // THE NAME THE SPECIFICATION USES, on the line, lowercase, greppable.
    const std::string line = tr::recall_refusal_line(refused, tr::RecallCost{});
    check(has(line, "refused-prefill-budget"),
          "C: the refusal line carries `refused-prefill-budget`",
          "spell the refusal with another word: a harness greps this ONE string");
    check(has(line, "REFUSED-prefill-budget"),
          "C: and the tree's uppercase header shape is preserved beside it",
          "replace the header instead of appending to it: existing harnesses grep REFUSED-");
    check(has(line, "wanted_tokens=576") && has(line, "budget_tokens=512"),
          "C: and the line is arithmetic, not a bare word",
          "print the refusal without the two numbers the decision was made on");
    // The byte budget's own refusal must NOT be spelled the same way -- one string, one refusal.
    const tr::RecallPagePlan byte_refused =
        tr::plan_recall_pages(make_request(0u, 227u, 0u), all_live);
    check(byte_refused.exact(), "C: the control plan needs no refusal to be read against",
          "mislabel the control");
    std::printf("C  REFUSED line: %s\n", line.c_str());
}

// ===========================================================================
// D -- THE NEGATIVE CONTROL: a synthetic broken copy that TRUNCATES, so C cannot pass vacuously
// ===========================================================================
// The tree's own pattern, stated at tests/CMakeLists.txt for ninfer_turn_recall_inexact_gate_test:
// "a NEGATIVE CONTROL on a synthetic broken copy behind every text check, so the test cannot pass
// vacuously". The broken copy here is the mutation section C names: `lo += drop`, the byte budget's
// own arithmetic, applied to the TOKEN budget. If that mutation produced the SAME answer as the
// refusal, section C would be asserting nothing -- so the difference is measured here.
tr::RecallPagePlan mutant_token_budget_truncates(const tr::RecallPagePlanRequest& request) {
    // A faithful, deliberately-wrong copy of the new block: it clamps instead of refusing.
    tr::RecallPagePlan plan;
    std::uint32_t lo = request.wanted_begin_page;
    std::uint32_t hi = request.wanted_end_page;
    if (hi > request.frontier / request.page_tokens) { hi = request.frontier / request.page_tokens; }
    if (lo > hi) { lo = hi; }
    if (request.token_budget != 0) {
        const std::uint64_t want_tokens =
            static_cast<std::uint64_t>(hi - lo) * request.page_tokens;
        plan.prefill_token_budget  = request.token_budget;
        plan.prefill_tokens_wanted = want_tokens;
        const std::uint64_t max_pages = request.token_budget / request.page_tokens;
        if (static_cast<std::uint64_t>(hi - lo) > max_pages) {
            lo += static_cast<std::uint32_t>(static_cast<std::uint64_t>(hi - lo) - max_pages);
        }
    }
    for (std::uint32_t p = lo; p < hi; ++p) { plan.pages.push_back(p); }
    return plan;
}

void section_D_mutant_diverges() {
    const tr::RecallPagePlan refused =
        tr::plan_recall_pages(make_request(0u, kPagesForBudget + 1u, kProposedBudget), all_live);
    const tr::RecallPagePlan mutant =
        mutant_token_budget_truncates(make_request(0u, kPagesForBudget + 1u, kProposedBudget));

    check_eq(refused.count(), 0u, "D: the tree refuses", "truncate in the tree");
    check_eq(mutant.count(), kPagesForBudget, "D: the mutant truncates to 8 pages",
             "make the mutant agree with the tree: then section C asserts nothing");
    check(pages_of(mutant) != pages_of(refused),
          "D: so the mutant's plan DIFFERS from the tree's -- section C has teeth",
          "the mutant and the tree agree, i.e. the refusal is not a refusal");
    // ⭐ AND THE MUTANT SHOWS THE HARM BY NAME: its first page is 1, not 0 -- the ANCHOR the
    // selector rank-1'd sits at the LOW end of the run, so it is the first thing a truncation
    // drops. That is dl/vectorkey/REPORT.md:281's measured failure ("the budget truncates and
    // drops the anchor"), reproduced here as a difference the check can see.
    check(!mutant.pages.empty() && mutant.pages.front() == 1u,
          "D: the mutant's run STARTS AT PAGE 1 -- the anchor at page 0 is dropped",
          "keep the anchor: then the truncation is the benign variant and this control is wrong");
    // And the mutant is NOT refused, so a reader who greps for the refusal gets nothing from it.
    check(!mutant.refused_prefill_budget, "D: the mutant does not report a refusal at all",
          "make the mutant refuse: then the harm is not silent");
    std::printf("D  tree pages=%zu mutant pages=%zu mutant_first=%u\n", refused.pages.size(),
                mutant.pages.size(), mutant.pages.empty() ? 0u : mutant.pages.front());
}

// ===========================================================================
// E -- THE ENGINE HALF, ASSERTED AS TEXT, WITH A NEGATIVE CONTROL BEHIND EVERY CHECK
// ===========================================================================
// It cannot be linked here (the engine's device link is not this test's dependency), so the
// wiring is pinned as TEXT -- the shape tests/CMakeLists.txt already documents for
// ninfer_turn_recall_inexact_gate_test. Every check below is followed by the same predicate run
// against an empty string, so it cannot pass vacuously.
void check_text(bool ok, const std::string& what, const std::string& mutation) {
    check(ok, what, mutation);
    check(!has(std::string(), "NINFER_RECALL_PREFILL_TOKENS"),
          "E negative control: the predicate finds nothing in an empty copy", mutation);
}

void section_E_the_wiring_is_present() {
    const std::string pi = read_text_file(std::string(NINFER_SOURCE_DIR) +
                                          "/src/targets/qwen3_6/impl/runtime/program_impl.h");
    const std::string oc =
        read_text_file(std::string(NINFER_SOURCE_DIR) + "/apps/cli/options.cpp");
    const std::string oh =
        read_text_file(std::string(NINFER_SOURCE_DIR) + "/apps/cli/options.h");
    const std::string ph =
        read_text_file(std::string(NINFER_SOURCE_DIR) +
                       "/src/targets/qwen3_6/impl/runtime/program.h");
    check(!pi.empty() && !oc.empty() && !oh.empty() && !ph.empty(),
          "E: the four engine files were read", "the source tree is not where NINFER_SOURCE_DIR says");

    check_text(has(pi, "NINFER_RECALL_PREFILL_TOKENS"),
               "E: the engine reads NINFER_RECALL_PREFILL_TOKENS",
               "delete the env read: the flag then reaches nothing");
    check_text(has(pi, "plan_request.token_budget    = turn_recall_prefill_token_budget;"),
               "E: and hands it to the plan at the ONE site",
               "stop assigning the budget: the plan is then never bounded in tokens");
    check_text(has(pi, "refused_prefill_tok"),
               "E: the refusal has its own census counter",
               "drop the counter: the refusal stops being countable");
    check_text(has(ph, "turn_recall_prefill_token_budget"),
               "E: program.h carries the member", "read the budget into nothing");
    check_text(has(oh, "recall_prefill_tokens"),
               "E: options.h carries the flag's field", "parse the flag into nothing");
    check_text(has(oc, "NINFER_RECALL_PREFILL_TOKENS"),
               "E: options.cpp commits the flag to the environment (the flag beats the env)",
               "drop the commit: the flag is then accepted and read by nothing");
    check_text(has(oc, "--recall-prefill-tokens"),
               "E: and the flag is named in the usage/help text",
               "ship a flag no help text mentions");
    // The dimension claim, pinned where the tree makes it: the byte default keeps its caliber.
    check_text(has(pi, "MEMORY-SAFETY limit rather than a speed limit"),
               "E: the byte budget keeps its own stated caliber",
               "delete the tree's own sentence about what its budget is for");
    check_text(has(pi, "14,528"), "E: and the ceiling that caliber implies is attached to it",
               "leave the number 14,528 unattached to any caliber in the tree");
}

// ===========================================================================
// F -- THE FROZEN REFERENCE IDS, ELEMENT FOR ELEMENT (the PASS_IDS_EXACT shape)
// ===========================================================================
void section_F_the_reference_is_pinned() {
    const std::uint32_t want[17] = {842u, 2080u, 887u, 28u, 19u, 24u, 18u, 23u, 16u, 22u,
                                    26u,   24614u, 28u, 8023u, 33u, 34643u, 248046u};
    bool same = true;
    for (std::size_t i = 0; i < 17u; ++i) {
        if (kReferenceIds[i] != want[i]) { same = false; }
    }
    check(same, "F: the 17 reference ids are the fleet's, element for element",
          "one id changed: a frozen sequence that drifts is not a reference");
    check_eq(kReferenceIds[12], 28u, "F: id 28 occurs at index 12 as well as index 3",
             "deduplicate the sequence: the fleet's sequence has the repeat");
    std::printf("F  reference ids (17):");
    for (std::size_t i = 0; i < 17u; ++i) { std::printf(" %u", kReferenceIds[i]); }
    std::printf("\n");
}

}  // namespace

int main() {
    section_A_positive();
    section_B_control();
    section_C_the_edge_is_refusal();
    section_D_mutant_diverges();
    section_E_the_wiring_is_present();
    section_F_the_reference_is_pinned();

    std::printf("CHECKS=%d FAILURES=%d\n", g_checks, g_fail);
    if (g_fail != 0) {
        // The fleet's verdict names, from scripts/check_gold.py's table
        // (dl/vectorkey/REPORT.md:313-325): a wrong answer is FAIL_WRONG, exit 3.
        std::printf("VERDICT=FAIL_WRONG\n");
        return 3;
    }
    std::printf("VERDICT=PASS\n");
    return 0;
}
