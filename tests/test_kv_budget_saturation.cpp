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
// The deploy layer's OWN two calls, so `deploy_layer_accepts` below reads the same decision point
// the build does instead of believing a spec string (landq/kvland pin move, T3).
#include "product/kv_options.h"
#include "product/kv_storage_dtype.h"
// kvland-p1-sat-includes

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

// Does the DEPLOY LAYER accept every layer of this spec?  The deploy layer's OWN two calls --
// product/kv_options.h parse_kv_layer_storage_spec (the plan -> storages grammar) and
// product/kv_storage_dtype.h kv_dtype_for_storage (the ONE deploy decision point) -- which is
// the same walk product/kv_bit_budget.h kv_bit_budget_spec_is_deployable does for the product.
// The test repeats the walk rather than calling that helper ON PURPOSE: the helper is part of the
// entry under test, and a pin that asks the entry whether the entry is right is not a pin.
bool deploy_layer_accepts(const std::string& spec) {
    ninfer::product::KvLayerStorageSpec parsed;
    try {
        parsed = ninfer::product::parse_kv_layer_storage_spec(spec);
    } catch (const std::exception&) {
        return false;
    }
    for (std::size_t i = 0; i < parsed.set.size(); ++i) {
        if (!parsed.set[i]) { continue; }
        try {
            (void)ninfer::product::kv_dtype_for_storage(parsed.table[i], "kvland-pin");
        } catch (const std::exception&) {
            return false;
        }
    }
    return true;
}
// kvland-p2-sat-deploywalk

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
    // ⚠ THE ORACLE HAS TO CARRY THE SPEC, NOT ONLY bits -> penalty (landq/kvreach,
    // dl/_orch/landq/kvreach/PINS.md section 3). `kv_bit_budget_solve` reaches the DP through
    // detail::kv_bit_budget_solve_impl_gated / detail::kv_gear_solve_gated, and those solve TWICE:
    // the shipped candidate set gets the first word, and the rows the deploy layer cannot read
    // (`kv_bit_budget_deployable_rows()`) are made unselectable on the SECOND pass, which runs
    // only when the first pass's answer is a plan product::kv_dtype_for_storage refuses. So "the
    // candidate set" is a function of the REQUEST, and one fixed set cannot be the oracle of this
    // solver. Measured with the single-set oracle this file used to carry
    // (dl/kvland/logs/21_LIVE_kd_kv_test_kv_budget_saturation.txt): 2917 FAILs in three shapes --
    // 1349 "saturates" (different bits), 1480 "breaks the bit tie" (different penalty) and 83
    // "solves a feasible ceiling" (the oracle called it feasible and the solver THREW, because the
    // FIRST-pass answer used a withheld row and the SECOND pass had no plan under the ceiling).
    //
    // Both sets come out of ONE walk: a leaf records whether it used a withheld row, so
    //   * `pen_all`        -- min penalty at this total over the SHIPPED set;
    //   * `pen_deployable` -- the same over the rows the DEPLOY LAYER ACCEPTS, i.e. over the
    //     leaves that used no withheld row (that is exactly the set the second pass solves).
    // Both are read from the product's own census rather than from a list written HERE: a second
    // spelling of "which rows can be built" is the defect this project keeps re-finding, and the
    // note above oracle_slot_allowed already says so about the candidate set itself.
    std::map<std::int32_t, double> pen_all;         // shipped set: bits_x100 total -> min penalty
    std::map<std::int32_t, double> pen_deployable;  // deploy-layer-readable subset, same key
    bool feasible_any = false;
};
// kvland-p3-sat-oracle

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

// THE SECOND SET: which of the rows the DP MAY CHOOSE the deploy layer refuses.  Read from the
// product's own census -- `kv_bit_budget_deployable_rows()`, which is itself derived from
// kv_dtype_for_storage -- so this cannot drift from the pass the solver actually falls back to.
// `selectable` is deliberately NOT consulted here: the census answers "can the engine read it",
// which is a different question, and on this tree exactly two rows answer them differently.
bool oracle_row_withheld(std::int32_t tier) {
    const std::array<bool, 8>& ok = ninfer::product::kv_bit_budget_deployable_rows();
    if (tier < 0 || static_cast<std::size_t>(tier) >= ok.size()) { return true; }
    return !ok[static_cast<std::size_t>(tier)];
}

// The rk4v4 EXPOSURE LIMIT covers the whole FAMILY, not row 4 alone --
// kvland-p4-sat-withheld
// product/kv_bit_budget.h:730-738: "a 3-bit or 2-bit rk4v4 is strictly weaker than the 4-bit one,
// so counting only row 4 would let a plan spend its rk4v4 budget and then add unbounded rk2v4
// layers on top. COEXISTENCE ITEM: this makes rk4v4_limit cover all three rows." The oracle must
// carry the same running FAMILY count, or it would enumerate plans the DP cannot reach.
bool is_rk4v4_family(std::int32_t tier) { return tier == 4 || tier == 6 || tier == 7; }

inline void oracle_record(std::map<std::int32_t, double>* table, std::int32_t bits,
                          double penalty) {
    auto it = table->find(bits);
    if (it == table->end()) {
        table->emplace(bits, penalty);
    } else if (penalty < it->second) {
        it->second = penalty;
    }
}

// `withheld_used` says whether any row on this path is one the deploy layer refuses; a leaf with
// it false is in BOTH candidate sets, a leaf with it true only in the shipped one. The cold
// pseudo-tier is a hot NVFP4 window in the plan the pool will have (product/kv_kv_bits.h
// detail::kv_bits_hot_window_tier), so it never withholds a leaf.
void enumerate(std::int32_t layer, std::int32_t layers, std::int32_t rk4v4_limit,
               std::int32_t cold_cap, bool legacy, std::int32_t capacity, std::int32_t bits,
               std::int32_t family_used, std::int32_t cold_used, bool withheld_used, double penalty,
               Oracle* out) {
    if (layer == layers) {
        oracle_record(&out->pen_all, bits, penalty);
        if (!withheld_used) { oracle_record(&out->pen_deployable, bits, penalty); }
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
                  family_used + (family ? 1 : 0), cold_used, withheld_used || oracle_row_withheld(t),
                  penalty + tier.penalty_x100 / 100.0, out);
    }
    if (cold_cap > 0 && cold_used < cold_cap) {
        const std::int32_t nb = bits + kKvBitBudgetColdBitsX100;
        if (nb <= capacity) {
            enumerate(layer + 1, layers, rk4v4_limit, cold_cap, legacy, capacity, nb, family_used,
                      cold_used + 1, withheld_used,
                      penalty + kKvBitBudgetColdPenaltyX100 / 100.0, out);
        }
    }
}

Oracle build_oracle(std::int32_t layers, std::int32_t rk4v4_limit, std::int32_t cold_cap,
                    bool legacy) {
    Oracle out;
    enumerate(0, layers, rk4v4_limit, cold_cap, legacy, /*capacity=*/1600 * layers, 0, 0, 0, false,
              0.0, &out);
    return out;
}
// kvland-p5-sat-enum

// WHICH CANDIDATE SET ANSWERED, which is a third outcome and not a detail: `Refuse` is the case
// where the shipped pass's answer is undeployable AND the deployable set has nothing under the
// ceiling either, i.e. exactly the throw `kv_bit_budget_solve_impl_gated`'s second pass produces.
enum class OracleAnswer { Ship, Masked, Refuse };

// The saturating optimum under `capacity` (in 0.01-bit units), chosen by THE SOLVER'S OWN
// SEGMENTED RULE rather than by a fixed set, in three steps that mirror
// detail::kv_bit_budget_solve_impl_gated line for line:
//   1. the shipped set's largest total at or below the ceiling is the FIRST pass's total; the
//      first pass returns a deployable plan exactly when a leaf that used no withheld row also
//      achieves that total's smallest penalty (the objective is "maximum bits, then minimum
//      penalty", so the withheld rows matter only when they are what made it optimal);
//   2. if it is deployable, that is the answer -- and the first pass is the untouched code path,
//      which is what makes the zero-move property structural;
//   3. otherwise the DEPLOYABLE set gets the question, and if it has nothing under the ceiling
//      the request is REFUSED.
OracleAnswer oracle_best(const Oracle& o, std::int32_t capacity, std::int32_t layers, double* bits,
                         double* penalty) {
    auto shipped = o.pen_all.upper_bound(capacity);
    if (shipped != o.pen_all.begin()) {
        --shipped;
        const auto deployable = o.pen_deployable.find(shipped->first);
        if (deployable != o.pen_deployable.end() &&
            deployable->second <= shipped->second + 1e-9) {
            *bits = static_cast<double>(shipped->first) / (layers * 100.0);
            *penalty = shipped->second;
            return OracleAnswer::Ship;
        }
    }
    auto masked = o.pen_deployable.upper_bound(capacity);
    if (masked == o.pen_deployable.begin()) { return OracleAnswer::Refuse; }
    --masked;
    *bits = static_cast<double>(masked->first) / (layers * 100.0);
    *penalty = masked->second;
    return OracleAnswer::Masked;
}
// kvland-p6-sat-segmented

// ------------------------------------------------------------------ the sweep
void sweep_case(std::int32_t layers, std::int32_t rk4v4_limit, std::int32_t cold_cap, bool legacy,
                const std::string& label, double first, double last) {
    const Oracle oracle = build_oracle(layers, rk4v4_limit, cold_cap, legacy);
    std::printf("  oracle: %zu shipped-set totals, %zu deployable-set totals\n",
                oracle.pen_all.size(), oracle.pen_deployable.size());
    // The deployable set must be a PROPER subset: the two rows the deploy layer refuses are the
    // ladder's two cheapest, so the shipped set reaches totals the deployable one cannot. If this
    // ever stops holding, the segmentation this oracle reproduces would be a no-op and the sweep
    // would be testing the unsegmented solver again while claiming otherwise.
    check(oracle.pen_deployable.size() > 0 &&
              oracle.pen_deployable.size() < oracle.pen_all.size(),
          "the deployable candidate set is a PROPER subset of the shipped one (" + label + ")");
// kvland-p6b-sat-propersubset
    double previous = -1.0;
    int solved_count = 0;
    int exact_hits = 0;
    int strict_undershoots = 0;
    int refused = 0;
    // HOW OFTEN THE SECOND PASS ANSWERED, and how often the gate refused outright. These are the
    // instrument that keeps the new oracle honest: an oracle that had quietly stopped applying the
    // segmented rule would agree with the solver on the FIRST-pass ceilings and differ on the
    // rest, and these two counters would go back to zero.
    int masked_engaged = 0;
    int gate_refusals = 0;
// kvland-p7-sat-counters
    for (double r = first; r <= last + 1e-9; r += 0.01) {
        const double request = std::nearbyint(r * 100.0) / 100.0;
        const std::int32_t capacity =
            static_cast<std::int32_t>(std::nearbyint(request * 100.0)) * layers;
        double want_bits = 0.0;
        double want_penalty = 0.0;
        const OracleAnswer answer = oracle_best(oracle, capacity, layers, &want_bits, &want_penalty);
        const bool feasible = (answer != OracleAnswer::Refuse);
        if (answer == OracleAnswer::Masked) { ++masked_engaged; }
        if (answer == OracleAnswer::Refuse) { ++gate_refusals; }
        const std::string at = label + " request=" + std::to_string(request);
// kvland-p8-sat-call

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
    std::printf("  %-44s solved=%4d refused=%3d exact=%4d strict_undershoot=%4d "
                "(second-pass=%4d gate-refusals=%4d)\n",
                label.c_str(), solved_count, refused, exact_hits, strict_undershoots,
                masked_engaged, gate_refusals);
    check(solved_count > 0, "the sweep solved something (" + label + ")");
    // NEGATIVE CONTROL ON THE SEGMENTATION ITSELF. Every sweep below reaches requests whose
    // FIRST-pass answer the deploy layer refuses, so the second pass must answer somewhere; if it
    // never does, this oracle has stopped being the oracle of the gated solver and its agreement
    // with the solver on the other requests proves nothing.
    check(masked_engaged + gate_refusals > 0,
          "the deployability segmentation really engaged in this sweep (" + label +
              "): second-pass=" + std::to_string(masked_engaged) +
              " gate-refusals=" + std::to_string(gate_refusals));
// kvland-p9-sat-control
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
    // ⚠ THE ANSWER AT 4.50 MOVED BACK TO ALL-NVFP4, AND THE MOVE IS THE DEPLOY LAYER'S, NOT THE
    // RULE'S (landq/kvreach, dl/_orch/landq/kvreach/PINS.md section 2). The rule is unchanged:
    // "maximum bits at or below the ceiling, then minimum penalty". What changed is the candidate
    // set: the cheaper exact-4.50 mix `0-2:rk2v4,3-4:rk4v4,5:rk3v4,6-7:rk2v4,8-9:int8,10-15:nvfp4`
    // (292 hundredths against all-nvfp4's 480) is a plan
    // product::kv_dtype_for_storage REFUSES, so detail::kv_gear_solve_gated answers with the
    // shipped set only while that answer is runnable and falls back to the rows the deploy layer
    // accepts otherwise. 600 = 2*325 + 2*425 + 375 + 2*325 + 2*825 + 6*450, i.e. 6000/... the
    // mix is still 4.5000 exactly, and so is the answer below.
    check(at_450.spec == "0-15:nvfp4",
          "--kv-bits 4.50 returns the cheapest exact-4.50 mix the DEPLOY LAYER can build, got " +
              at_450.spec);
    check(std::fabs(at_450.penalty - 4.80) < 1e-9,
          "and that mix costs 480 hundredths on the shipped ladder (16 x nvfp4's 30), while the "
          "cheaper 292-hundredths mix is a plan the build refuses, got " +
              std::to_string(at_450.penalty));
    check(at_450.shortfall_bits == 0.0, "--kv-bits 4.50 has zero shortfall");
    // 4.50 is still reachable WITHOUT iso4e: nvfp4 alone is 450, so iso4e's availability cannot
    // be the explanation for the old undershoot.
    check(at_450.spec.find("iso4e") == std::string::npos,
          "the 4.50 answer needs no iso4e (iso4e == nvfp4 == 450, so the pin is not the cause)");
    // ... and now the honest inverse of the pre-gate claim: the answer may NOT contain a row the
    // deploy layer refuses, and the whole plan must be one the deploy layer accepts.
    check(at_450.spec.find("rk3v4") == std::string::npos &&
              at_450.spec.find("rk2v4") == std::string::npos,
          "and the answer uses NO row the deploy layer refuses (rk3v4/rk2v4 are withheld), got " +
              at_450.spec);
    check(deploy_layer_accepts(at_450.spec),
          "and the deploy layer's own two calls accept the answer: " + at_450.spec);
// kvland-p10-sat-450

    // A ceiling the ladder cannot fill exactly must return the largest value BELOW it and
    // say so -- not round up, and not fall back to a much cheaper plan. 4.60 is such a ceiling,
    // and the value it fills to moved when the candidate set narrowed (landq/kvreach): the
    // ladder's own reachable totals still step 4.50 -> 4.59375
    // (4.59375 = (2*325 + 4*425 + 325 + 2*825 + 7*450)/16), but that 4.59375 mix is built from
    // the two withheld rows, so it is not a plan this engine can run. The largest total at or
    // below 4.60 that the DEPLOY LAYER accepts is all-nvfp4's 4.50, and the shortfall must say
    // so: the property under test is "the largest value below the ceiling AND an explicit
    // shortfall", not the number 4.59375.
    const KvBitBudgetSolution at_460 = kv_bit_budget_solve(kLayers, 4.60, 8, 0);
    check(std::fabs(at_460.achieved_bits - 4.50) < 1e-9,
          "4.60 fills as far as the DEPLOYABLE ladder allows (4.50), got " +
              std::to_string(at_460.achieved_bits));
    check(std::fabs(at_460.shortfall_bits - 0.10) < 1e-9,
          "4.60 reports its 0.10000-bit shortfall explicitly, got " +
              std::to_string(at_460.shortfall_bits));
    check(deploy_layer_accepts(at_460.spec),
          "and the 4.60 answer is buildable too: " + at_460.spec);
// kvland-p11-sat-460

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
