// Host-only test for the SATURATING contract of the KV bit-budget solver
// (product/kv_bit_budget.h, reached by --kv-bits / --kv-k-bits / --kv-v-bits through
// product/kv_kv_bits.h and by --kv-bit-budget through product/kv_bit_budget.h).
//
// WHY THIS TEST EXISTS. The solver used to select the minimum-penalty final state among
// the states with `bits <= capacity`, i.e. a ceiling was a ONE-SIDED constraint that the
// objective was free to undershoot. Measured at the engine (pinned binary
// 9ace5573cb897532): `--kv-bits 4.5` returned `achieved=4.38 spec=0-7:rk4v4,8-15:nvfp4`
// while all-nvfp4 at exactly 4.50 was feasible, and a 4.60 ceiling returned 4.38 as well
// (shortfall 0.22 bits). A request must instead be a SATURATING constraint: fill the
// ceiling from below, as close as the ladder allows, and never exceed it.
//
// THE ORACLE IS EXHAUSTIVE, NOT A SECOND SOLVER. For a small layer count this file
// ENUMERATES every reachable (total bits, penalty) pair of the same candidate space and
// asserts the solver returns the maximum bits at or below the ceiling. A reference DP
// would have re-encoded the same selection rule and could share a bug with the code under
// test; an enumeration cannot.
//
// ⚠ "THE SAME CANDIDATE SPACE" IS THE LOAD-BEARING HALF, AND IT WAS A HAND-WRITTEN LIST (line
// `redkv`, dl/redkv/TRIAGE.md #53). The list named five ladder slots; the solver may now choose
// eight, because rk3v4/rk2v4 became selectable in dl/e8mixwire/land/patch_flip_batch.py. An
// enumeration that omits a gear the solver can pick reports a SMALLER optimum, so every
// saturation/tie/no-overshoot check keyed on it reddens at once (2960 of them). The candidate set
// is now derived from kKvBitBudgetTiers (selectable + the legacy fp8 clause) and the rk4v4
// exposure limit is applied to the whole FAMILY, which is what product/kv_bit_budget.h:730-738
// does.
//
// THE NEGATIVE CONTROLS ARE THE POINT (same convention as test_kv_kv_bits.cpp):
//   * a ceiling below the ladder's floor must THROW -- never silently return a worse tier;
//   * the ladder must really leave gaps, so that "strictly below the ceiling" is an
//     outcome the sweep exercises -- otherwise the saturation assertions would be
//     indistinguishable from "the ladder always divides the request exactly".

#include "product/kv_bit_budget.h"
#include "product/kv_kv_bits.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::KvVCodec;
using ninfer::product::kKvBitBudgetColdBitsX100;
using ninfer::product::kKvBitBudgetColdPenaltyX100;
using ninfer::product::kKvBitBudgetTiers;
using ninfer::product::KvBitBudgetSolution;
using ninfer::product::kv_bit_budget_factory_spec;
using ninfer::product::kv_bit_budget_solve;
using ninfer::product::kv_gear_evaluate_spec;

int g_failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) { return; }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

void check_throws(const std::function<void()>& fn, const std::string& what) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s -- did not throw\n", what.c_str());
}

// --------------------------------------------------------------- the oracle
// Every reachable total, in 0.01-bit-per-layer units, mapped to the SMALLEST penalty that
// reaches it. Enumerated once over the widest ceiling: the allocations reachable under a
// smaller ceiling are exactly the subset whose total is at or below it, so one table
// answers every request.
//
// `legacy` selects the candidate set of the cold-capable multiset solver, which does NOT
// consult the component mode and therefore includes fp8; the layer-exact solver intersects
// F_l with the mode's admissibility, and the default mode (rotation ON) drops fp8 because
// the fp8 decode kernel rotates K but not Q.
struct Oracle {
    std::map<std::int32_t, double> best_penalty_at;  // bits_x100 total -> min penalty
    bool feasible_any = false;
};

// THE CANDIDATE SET IS THE SOLVER'S, read from the ladder instead of written down.
// ⚠ RE-DERIVED (line `redkv`, dl/redkv/TRIAGE.md #53). This function used to carry a hand-written
// slot list -- `{0, 1, 3, 5}` plus fp8 for the legacy set plus row 4 for the rk4v4 prefix -- and
// that list was correct only while rk3v4 (6) and rk2v4 (7) were unselectable. The gate opened
// (dl/e8mixwire/land/patch_flip_batch.py, one item, eight clusters) and the DP may now choose
// them: measured at 16 layers / 4.50, `kv_bit_budget_solve` returns
// `0-2:rk2v4,3-4:rk4v4,5:rk3v4,6-7:rk2v4,8-9:int8,10-15:nvfp4` = 2725/600 = 4.541667 on a 4.55
// request, while a hand-written oracle that cannot see rows 6/7 still calls 4.50 the optimum.
// An oracle that does not enumerate a gear the solver may pick is not an oracle for that solver.
bool oracle_slot_allowed(std::int32_t tier, bool legacy) {
    if (tier < 0 || static_cast<std::size_t>(tier) >= kKvBitBudgetTiers.size()) { return false; }
    if (!kKvBitBudgetTiers[static_cast<std::size_t>(tier)].selectable) { return false; }
    // fp8 (slot 2) is in the LEGACY candidate set only: the layer-exact solver intersects the
    // ladder with the component mode's admissibility and the default mode drops fp8 because the
    // fp8 decode kernel rotates K but not Q; the cold-capable multiset solver does not consult
    // the mode at all.
    if (tier == 2) { return legacy; }
    return true;
}

// The rk4v4 EXPOSURE LIMIT covers the whole FAMILY, not row 4 alone --
// product/kv_bit_budget.h:730-738: "a 3-bit or 2-bit rk4v4 is strictly weaker than the 4-bit one,
// so counting only row 4 would let a plan spend its rk4v4 budget and then add unbounded rk2v4
// layers on top. COEXISTENCE ITEM: this makes rk4v4_limit cover all three rows." The oracle must
// carry the same running FAMILY count, or it would enumerate plans the DP cannot reach.
bool is_rk4v4_family(std::int32_t tier) { return tier == 4 || tier == 6 || tier == 7; }

void enumerate(std::int32_t layer, std::int32_t layers, std::int32_t rk4v4_limit,
               std::int32_t cold_cap, bool legacy, std::int32_t capacity, std::int32_t bits,
               std::int32_t family_used, std::int32_t cold_used, double penalty, Oracle* out) {
    if (layer == layers) {
        auto it = out->best_penalty_at.find(bits);
        if (it == out->best_penalty_at.end()) {
            out->best_penalty_at.emplace(bits, penalty);
        } else if (penalty < it->second) {
            it->second = penalty;
        }
        out->feasible_any = true;
        return;
    }
    for (std::size_t i = 0; i < kKvBitBudgetTiers.size(); ++i) {
        const std::int32_t t = static_cast<std::int32_t>(i);
        if (!oracle_slot_allowed(t, legacy)) { continue; }
        const bool family = is_rk4v4_family(t);
        if (family && family_used >= rk4v4_limit) { continue; }
        const auto& tier = kKvBitBudgetTiers[i];
        const std::int32_t nb = bits + tier.bits_x100;
        if (nb > capacity) { continue; }
        enumerate(layer + 1, layers, rk4v4_limit, cold_cap, legacy, capacity, nb,
                  family_used + (family ? 1 : 0), cold_used,
                  penalty + tier.penalty_x100 / 100.0, out);
    }
    if (cold_cap > 0 && cold_used < cold_cap) {
        const std::int32_t nb = bits + kKvBitBudgetColdBitsX100;
        if (nb <= capacity) {
            enumerate(layer + 1, layers, rk4v4_limit, cold_cap, legacy, capacity, nb, family_used,
                      cold_used + 1, penalty + kKvBitBudgetColdPenaltyX100 / 100.0, out);
        }
    }
}

Oracle build_oracle(std::int32_t layers, std::int32_t rk4v4_limit, std::int32_t cold_cap,
                    bool legacy) {
    Oracle out;
    enumerate(0, layers, rk4v4_limit, cold_cap, legacy, /*capacity=*/1600 * layers, 0, 0, 0, 0.0,
              &out);
    return out;
}

// The saturating optimum under `capacity` (in 0.01-bit units): the largest enumerated total
// at or below it, with that total's smallest penalty.
bool oracle_best(const Oracle& o, std::int32_t capacity, std::int32_t layers, double* bits,
                 double* penalty) {
    auto it = o.best_penalty_at.upper_bound(capacity);
    if (it == o.best_penalty_at.begin()) { return false; }
    --it;
    *bits = static_cast<double>(it->first) / (layers * 100.0);
    *penalty = it->second;
    return true;
}

// ------------------------------------------------------------------ the sweep
void sweep_case(std::int32_t layers, std::int32_t rk4v4_limit, std::int32_t cold_cap, bool legacy,
                const std::string& label, double first, double last) {
    const Oracle oracle = build_oracle(layers, rk4v4_limit, cold_cap, legacy);
    std::printf("  oracle: %zu distinct reachable totals\n", oracle.best_penalty_at.size());
    double previous = -1.0;
    int solved_count = 0;
    int exact_hits = 0;
    int strict_undershoots = 0;
    int refused = 0;
    for (double r = first; r <= last + 1e-9; r += 0.01) {
        const double request = std::nearbyint(r * 100.0) / 100.0;
        const std::int32_t capacity =
            static_cast<std::int32_t>(std::nearbyint(request * 100.0)) * layers;
        double want_bits = 0.0;
        double want_penalty = 0.0;
        const bool feasible = oracle_best(oracle, capacity, layers, &want_bits, &want_penalty);
        const std::string at = label + " request=" + std::to_string(request);

        KvBitBudgetSolution solved;
        bool threw = false;
        try {
            solved = kv_bit_budget_solve(layers, request, rk4v4_limit, cold_cap);
        } catch (const std::exception&) {
            threw = true;
        }
        if (!feasible) {
            ++refused;
            // NEGATIVE CONTROL: nothing fits, so the solver must REFUSE rather than
            // silently return a worse tier.
            check(threw, "refuses an infeasible ceiling (" + at + ")");
            continue;
        }
        check(!threw, "solves a feasible ceiling (" + at + ")");
        if (threw) { continue; }
        ++solved_count;

        // (1) SATURATION: the answer IS the maximum bits under the ceiling.
        check(std::fabs(solved.achieved_bits - want_bits) < 1e-9,
              "saturates " + at + ": achieved=" + std::to_string(solved.achieved_bits) +
                  " but the enumerated optimum is " + std::to_string(want_bits));
        // ... and the penalty is the best available AT that bit count.
        check(std::fabs(solved.penalty - want_penalty) < 1e-9,
              "breaks the bit tie on penalty " + at + ": got " +
                  std::to_string(solved.penalty) + " want " + std::to_string(want_penalty));
        // (2) NO OVERSHOOT: never above the ceiling.
        check(solved.achieved_bits <= request + 1e-9,
              "never exceeds the ceiling " + at + ": achieved=" +
                  std::to_string(solved.achieved_bits));
        check(solved.achieved_bits <= solved.requested_bits + 1e-9,
              "achieved <= requested on the solution's own accounting (" + at + ")");
        // ... and the shortfall it reports is the one a caller can verify.
        check(std::fabs(solved.shortfall_bits - (solved.requested_bits - solved.achieved_bits)) <
                  1e-9,
              "shortfall is requested - achieved (" + at + ")");
        check(solved.shortfall_bits >= -1e-9, "shortfall is non-negative (" + at + ")");
        // (3) MONOTONE: a larger ceiling never returns fewer bits.
        check(previous < 0.0 || solved.achieved_bits >= previous - 1e-9,
              "request -> achieved is monotone " + at + ": " + std::to_string(previous) + " -> " +
                  std::to_string(solved.achieved_bits));
        previous = solved.achieved_bits;

        if (std::fabs(solved.achieved_bits - request) < 1e-9) { ++exact_hits; }
        if (solved.achieved_bits < request - 1e-9) { ++strict_undershoots; }
    }
    std::printf("  %-44s solved=%4d refused=%3d exact=%4d strict_undershoot=%4d\n",
                label.c_str(), solved_count, refused, exact_hits, strict_undershoots);
    check(solved_count > 0, "the sweep solved something (" + label + ")");
    // NEGATIVE CONTROL on the oracle itself: if NOTHING undershot, the saturation
    // assertions would be indistinguishable from "the ladder always divides the request
    // exactly", i.e. they would prove nothing.
    check(strict_undershoots > 0,
          "the ladder really does leave gaps, so undershooting is an exercised outcome (" +
              label + ")");
}

// The reported defect, pinned exactly.
void test_reported_defect_is_gone() {
    constexpr std::int32_t kLayers = 16;
    // BEFORE: `--kv-bits 4.5` (cold off) returned achieved=4.38, spec 0-7:rk4v4,8-15:nvfp4,
    // i.e. 8*4.25 + 8*4.50 = 4.375 -- 0.125 bits under the ceiling with all-nvfp4 (exactly
    // 4.50, penalty 4.80) available. AFTER: exactly on the ceiling, and the SATURATION is what
    // this check is about.
    //
    // ⚠ THE ANSWER AT 4.50 IS NO LONGER HOMOGENEOUS nvfp4 (line `redkv`, dl/redkv/TRIAGE.md #53).
    // It was, while rk3v4 (375) and rk2v4 (325) were unselectable: nvfp4 x 16 is exactly 450 x 16
    // and nothing else on the ladder reached it below the ceiling. With rows 6/7 selectable the
    // ladder has 120 distinct 16-layer allocations that cost EXACTLY 7200 = 450 x 16 (measured by
    // exhaustively enumerating the post-flip grammar; the pre-flip count was 17), and the shipped
    // ladder rates them at a LOWER penalty than 16 x nvfp4: the chosen mix costs
    // 3*16 + 2*8 + 1*12 + 5*16 + 2*2 + 6*30 = 292 hundredths against nvfp4's 16 x 30 = 480.
    // The saturation rule is "maximum bits at or below the ceiling, then minimum penalty", so the
    // answer moves with the ladder's own prices. This is a CONSEQUENCE of the gate flip, not a
    // change to the rule, and both halves are pinned below.
    const KvBitBudgetSolution at_450 = kv_bit_budget_solve(kLayers, 4.50, 8, 0);
    check(std::fabs(at_450.achieved_bits - 4.50) < 1e-9,
          "--kv-bits 4.50 fills the 4.50 ceiling exactly (got " +
              std::to_string(at_450.achieved_bits) + ")");
    check(at_450.spec == "0-2:rk2v4,3-4:rk4v4,5:rk3v4,6-7:rk2v4,8-9:int8,10-15:nvfp4",
          "--kv-bits 4.50 returns the cheapest exact-4.50 mix the ladder now has, got " +
              at_450.spec);
    check(std::fabs(at_450.penalty - 2.92) < 1e-9,
          "and that mix costs 292 hundredths on the shipped ladder, the cheapest exact-4.50 "
          "allocation (all-nvfp4 would cost 480), got " + std::to_string(at_450.penalty));
    check(at_450.shortfall_bits == 0.0, "--kv-bits 4.50 has zero shortfall");
    // 4.50 is still reachable WITHOUT iso4e: nvfp4 alone is 450, so iso4e's availability cannot
    // be the explanation for the old undershoot.
    check(at_450.spec.find("iso4e") == std::string::npos,
          "the 4.50 answer needs no iso4e (iso4e == nvfp4 == 450, so the pin is not the cause)");
    // ... and the rows the gate opened ARE in it, which is the mechanism claim above.
    check(at_450.spec.find("rk3v4") != std::string::npos ||
              at_450.spec.find("rk2v4") != std::string::npos,
          "and the answer uses a row the gate opened, got " + at_450.spec);

    // A ceiling the ladder cannot fill exactly must return the largest value BELOW it and
    // say so -- not round up, and not fall back to a much cheaper plan. 4.60 is such a ceiling:
    // the reachable totals step over it, and the step is what moved. At 8 rk4v4-family layers the
    // reachable totals used to step 450 -> 475 (the 4-bit row at 425 plus one int8); with rows
    // 6/7 in the family the ladder reaches 459.375 = 2755.5/... i.e. 459.375 exactly measured:
    // 4.59375 = (2*325 + 4*425 + 1*325 + 2*825 + 7*450)/16, one step below 4.60 and 0.09375
    // above the old 4.50.
    const KvBitBudgetSolution at_460 = kv_bit_budget_solve(kLayers, 4.60, 8, 0);
    check(std::fabs(at_460.achieved_bits - 4.59375) < 1e-9,
          "4.60 fills as far as the ladder allows (4.59375), got " +
              std::to_string(at_460.achieved_bits));
    check(std::fabs(at_460.shortfall_bits - 0.00625) < 1e-9,
          "4.60 reports its 0.00625-bit shortfall explicitly, got " +
              std::to_string(at_460.shortfall_bits));

    // The engine's no-request default is the target's registered table and must NOT move:
    // only a request solves, and this fix changes no default.
    const std::string factory = kv_bit_budget_factory_spec(kLayers);
    const auto ev = kv_gear_evaluate_spec(factory, kLayers, 16.00, kKvBitBudgetTiers, 0);
    check(std::fabs(ev.achieved_bits - 4.40625) < 1e-9,
          "the factory table is still 4.40625 bits, got " + std::to_string(ev.achieved_bits));
    check(ev.parsed, "the factory spec still parses");
}

// The FOURTH property: a request never yields less than the no-request default, for any
// request at or above the default. Below it the request itself is lower, so a lower answer
// is what was asked for.
void test_never_below_the_default() {
    constexpr std::int32_t kLayers = 16;
    const auto ev = kv_gear_evaluate_spec(kv_bit_budget_factory_spec(kLayers), kLayers, 16.00,
                                          kKvBitBudgetTiers, 0);
    int checked = 0;
    for (const std::int32_t cold_cap : {0, 6}) {
        for (double r = ev.achieved_bits; r <= 16.0 + 1e-9; r += 0.05) {
            const double request = std::nearbyint(r * 100.0) / 100.0;
            KvBitBudgetSolution solved;
            try {
                solved = kv_bit_budget_solve(kLayers, request, 8, cold_cap);
            } catch (const std::exception&) {
                check(false, "a request at or above the default must be feasible: " +
                                 std::to_string(request) + " cold_cap=" +
                                 std::to_string(cold_cap));
                continue;
            }
            ++checked;
            check(solved.achieved_bits >= ev.achieved_bits - 1e-9,
                  "request " + std::to_string(request) + " (cold_cap=" +
                      std::to_string(cold_cap) + ") returns " +
                      std::to_string(solved.achieved_bits) + " bits, below the no-request default " +
                      std::to_string(ev.achieved_bits));
        }
    }
    check(checked > 400, "the no-below-default sweep actually ran (" + std::to_string(checked) +
                             " requests)");
}

// The two ENTRY SPELLINGS must agree under a request, not merely on one sample:
// `--kv-bits` (the joint entry) and the solver `--kv-bit-budget` calls.
void test_entries_agree() {
    constexpr std::int32_t kLayers = 16;
    int compared = 0;
    for (const std::int32_t cold_cap : {0, 6}) {
        for (double r = 4.30; r <= 9.0 + 1e-9; r += 0.05) {
            const double request = std::nearbyint(r * 100.0) / 100.0;
            KvBitBudgetSolution joint;
            bool joint_threw = false;
            try {
                const auto plan = ninfer::product::kv_kv_bits_entry_joint(
                    kLayers, request, ninfer::product::kv_bit_budget_default_scores(), -1.0, 8,
                    cold_cap, KvVCodec::E2M1);
                joint.spec = plan.spec;
                joint.achieved_bits = plan.achieved_bits;
                joint.penalty = plan.penalty;
            } catch (const std::exception&) {
                joint_threw = true;
            }
            KvBitBudgetSolution solved;
            bool solve_threw = false;
            try {
                solved = kv_bit_budget_solve(kLayers, request, 8, cold_cap);
            } catch (const std::exception&) {
                solve_threw = true;
            }
            if (joint_threw != solve_threw) {
                check(false, "the two entries disagree on feasibility at " +
                                 std::to_string(request) + " cold_cap=" + std::to_string(cold_cap));
                continue;
            }
            if (joint_threw) { continue; }
            ++compared;
            check(joint.spec == solved.spec,
                  "entry spellings agree on the spec at " + std::to_string(request) + ": " +
                      joint.spec + " vs " + solved.spec);
            check(std::fabs(joint.achieved_bits - solved.achieved_bits) < 1e-9,
                  "entry spellings agree on achieved at " + std::to_string(request));
        }
    }
    check(compared > 100, "the entry-agreement sweep actually ran (" + std::to_string(compared) +
                              " comparisons)");
}

}  // namespace

int main() {
    std::printf("=== test_kv_budget_saturation: the ceiling is a SATURATING constraint ===\n");
    // 6 layers keeps the enumeration exhaustive: the oracle now walks the WHOLE selectable ladder
    // (8 rows) instead of five hard-coded slots, so the leaf count grew by the two rows the gate
    // opened rather than by a factor.
    std::printf("sweeps (enumerated oracle over the whole selectable ladder, request 4.00 -> "
                "16.00 step 0.01):\n");
    sweep_case(6, 8, 0, /*legacy=*/false, "6 layers, rk4v4<=8, cold off (layer-exact)", 4.00, 16.00);
    sweep_case(6, 2, 0, /*legacy=*/false, "6 layers, rk4v4<=2 (floor 4.4167), cold off", 4.00, 16.00);
    sweep_case(6, 8, 2, /*legacy=*/true, "6 layers, rk4v4<=8, cold_cap=2 (multiset)", 4.00, 16.00);
    test_reported_defect_is_gone();
    test_never_below_the_default();
    test_entries_agree();
    if (g_failures == 0) {
        std::printf("PASS: saturation, monotonicity, no-overshoot and no-below-default all hold\n");
        return 0;
    }
    std::printf("%d FAILURE(S)\n", g_failures);
    return 1;
}
