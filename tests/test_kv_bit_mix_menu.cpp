// Host-only test for the SLIDER's equal-bit selector in product/kv_bit_budget.h.
//
// THE RULE THIS AXIS EXISTS FOR (user, verbatim): "the slider is there so you can allocate
// DIFFERENT KINDS of quantization at the SAME bit count, not to compress bits." So the bit
// count is an INPUT and the knob chooses among codec MIXES that cost the SAME bits.
//
// WHAT WAS ACTUALLY WRONG, measured before this change (probe sha16 b0445f9801d17b2e,
// log /home/user/scratch/PATCHSET/BITSLIDER/stage/menu.txt):
//   * at a 4.50 request, 17 codec-different mixes all cost EXACTLY 4.5000 bits -- every
//     nvfp4^k + iso4e^(16-k), k = 0..16. iso4e shares nvfp4's plane geometry (450 == 450),
//     which kv_auto_relayout.h:50 states as "re-cutting the iso4e/nvfp4 boundary would trade
//     quality for exactly zero bytes";
//   * the solver returned EXACTLY ONE of them -- `0-15:nvfp4` -- for every weight in
//     {0,0.25,0.5,0.75,1} under BOTH measured score tables AND the default table: 15/15 knob
//     combinations, one spec. The weight axis is therefore NOT a route to the option set;
//   * so the alternative was unreachable, and the reachability gap was the TIE-BREAK, not
//     `selectable`: iso4e is `selectable == true` and its 450 is on the ladder, but among
//     plans the objective rates EQUAL the DP keeps whichever candidate the fixed pack order
//     inserted first (rk4v4, bf16, int8, fp8, nvfp4, iso4e), and nvfp4 always came first.
//
// WHAT THIS TEST PINS, and the negative controls that keep it honest:
//   1. the equal-bit set at 4.50 is enumerated exhaustively (no second solver) and is 17
//      codec-different mixes, ALL of them exactly 4.5000;
//   2. no rk4v4-bearing mix reaches 4.50 at all -- rk4v4 (425) has no bit-equal partner, and the
//      deviation arithmetic that says so is asserted as an INVARIANT over every mix;
//   3. THE DEFAULT IS UNCHANGED: `0-15:nvfp4` at exactly 4.5000. This is the same pin
//      tests/test_kv_budget_saturation.cpp holds, so a selector that moved the default
//      could not land;
//   4. TWO CODEC-DIFFERENT MIXES ARE BOTH SELECTABLE AT ONE BUDGET AND BOTH REPORT THE SAME
//      ACHIEVED BITS -- through the engine's own spec evaluator as well as the solver;
//   5. NEGATIVE CONTROL: the preference is not a penalty override. Under the SHIPPED ladder
//      iso4e costs 200 against nvfp4's 30, so `prefer iso4e` still returns nvfp4. Without
//      this control a "selector" that just always returned the preferred gear would pass;
//   6. NEGATIVE CONTROL: a gear outside the candidate grammar is REFUSED by name, not silently
//      dropped -- a silently dropped preference is the failure mode this knob exists to avoid.
//      ⚠ RE-DERIVED (line `redkv`, dl/redkv/TRIAGE.md #55): rk3v4/rk2v4 USED to be the example of
//      an out-of-grammar gear and are now INSIDE it (dl/e8mixwire/land/patch_flip_batch.py
//      clusters 5-7: kKvGearCandidateSlots grew 6 -> 8, and kv_bit_budget.h's own
//      grammar_covers_every_selectable_gear() makes that mandatory once a row is selectable).
//      The control now uses gears that are genuinely outside it, and the two rows gained a
//      POSITIVE control, so "the preference is accepted" and "the preference is refused" are
//      both pinned.
//   7. ⭐ THE EQUAL-BIT SET IS ENUMERATED OVER THE WHOLE GRAMMAR, not over the five slots the
//      pre-flip grammar had. Excluding rows 6/7 from the enumeration did not make the assertion
//      false, it made it EMPTY: 120 allocations cost exactly 7200 now, 63 of them carry rk4v4
//      (the row that had "no bit-equal partner" while the two narrow rows were illegal).

#include "product/kv_bit_budget.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::product::kKvBitBudgetTiers;
using ninfer::product::KvBitBudgetSolution;
using ninfer::product::KvGearSolution;
using ninfer::product::KvScoreContext;

int g_failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) { return; }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

template <typename Fn>
void check_throws(Fn&& fn, const std::string& what) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s -- did not throw\n", what.c_str());
}

constexpr std::int32_t kLayers = 16;
constexpr std::int32_t kRk4v4Limit = 8;
// The candidate space of a single layer, read from the ladder itself: every SELECTABLE row
// except fp8, which is absent because the default component mode drops it (its decode kernel
// rotates K but not Q). It is EIGHT slots now, not five: rk3v4/rk2v4 left the "not selectable"
// clause when the gate opened. Deriving the list instead of writing it down is what keeps this
// enumeration the solver's candidate space, which is the only thing that makes the counts below
// mean anything.
std::vector<std::int32_t> candidate_slots() {
    std::vector<std::int32_t> slots;
    for (std::size_t i = 0; i < kKvBitBudgetTiers.size(); ++i) {
        const std::string name = kKvBitBudgetTiers[i].spec_name;
        if (!kKvBitBudgetTiers[i].selectable) { continue; }
        if (name == "fp8") { continue; }
        slots.push_back(static_cast<std::int32_t>(i));
    }
    return slots;
}

// The rk4v4 EXPOSURE LIMIT is a FAMILY limit (product/kv_bit_budget.h:730-738: "COEXISTENCE ITEM:
// this makes rk4v4_limit cover all three rows"), so the enumeration must count rows 4, 6 and 7
// against ONE budget. Counting row 4 alone would admit plans the DP cannot reach.
bool is_rk4v4_family(std::int32_t slot) { return slot == 4 || slot == 6 || slot == 7; }

struct Mix {
    std::array<std::int32_t, 8> count{};   // parallel to candidate_slots() (8 slots since the flip)
    std::int32_t total = 0;                // bits x100, summed
    std::int32_t penalty = 0;              // shipped-ladder penalty x100, summed
    [[nodiscard]] std::string signature() const {
        const auto slots = candidate_slots();
        std::string s;
        for (std::size_t i = 0; i < count.size(); ++i) {
            if (count[i] == 0) { continue; }
            if (!s.empty()) { s += "+"; }
            s += std::string(kKvBitBudgetTiers[static_cast<std::size_t>(slots[i])].spec_name) +
                 "x" + std::to_string(count[i]);
        }
        return s;
    }
    [[nodiscard]] bool has_rk4v4() const {
        const auto slots = candidate_slots();
        for (std::size_t i = 0; i < count.size(); ++i) {
            if (count[i] > 0 && std::string(kKvBitBudgetTiers[static_cast<std::size_t>(slots[i])]
                                                .spec_name) == "rk4v4") {
                return true;
            }
        }
        return false;
    }
};

using MixTable = std::map<std::int32_t, std::vector<Mix>>;

void enumerate(std::size_t idx, const std::vector<std::int32_t>& slots,
               std::array<std::int32_t, 8>* count, std::int32_t left, std::int32_t family_used,
               MixTable* out) {
    if (idx == slots.size()) {
        if (left != 0) { return; }
        Mix m;
        m.count = *count;
        for (std::size_t i = 0; i < slots.size(); ++i) {
            const auto& tier = kKvBitBudgetTiers[static_cast<std::size_t>(slots[i])];
            m.total += count->at(i) * tier.bits_x100;
            m.penalty += count->at(i) * tier.penalty_x100;
        }
        (*out)[m.total].push_back(m);
        return;
    }
    const bool family = is_rk4v4_family(slots[idx]);
    const std::int32_t cap = family ? std::min(left, kRk4v4Limit - family_used) : left;
    for (std::int32_t c = 0; c <= cap; ++c) {
        (*count)[idx] = c;
        enumerate(idx + 1, slots, count, left - c, family_used + (family ? c : 0), out);
    }
    (*count)[idx] = 0;
}

MixTable build_table() {
    MixTable out;
    std::array<std::int32_t, 8> count{};
    enumerate(0, candidate_slots(), &count, kLayers, 0, &out);
    return out;
}

double bits_of(std::int32_t total) {
    return static_cast<double>(total) / (kLayers * 100.0);
}

// The third-party statement of the same fact, so this test cannot invent the tie:
// the measured long-context column gives iso4e and nvfp4 IDENTICAL rows.
void test_measured_columns_make_iso4e_and_nvfp4_a_tie() {
    const auto scores = ninfer::product::kv_bit_budget_measured_scores(KvScoreContext::Long);
    const std::int32_t nv = ninfer::product::detail::tier_index("nvfp4");
    const std::int32_t s3 = ninfer::product::detail::tier_index("iso4e");
    check(nv >= 0 && s3 >= 0, "the measured table indexes nvfp4 and iso4e");
    check(scores[static_cast<std::size_t>(nv)].quality_x100 ==
              scores[static_cast<std::size_t>(s3)].quality_x100 &&
              scores[static_cast<std::size_t>(nv)].speed_x100 ==
                  scores[static_cast<std::size_t>(s3)].speed_x100,
          "the measured long-context column scores iso4e exactly as nvfp4 (so the tie the "
          "selector resolves is a MEASURED tie, not one this test constructed)");
    check(kKvBitBudgetTiers[static_cast<std::size_t>(nv)].bits_x100 ==
              kKvBitBudgetTiers[static_cast<std::size_t>(s3)].bits_x100,
          "iso4e and nvfp4 cost the SAME bits (450 == 450): the equal-bit premise");
}

// (1) + (2): the equal-bit option set at 4.50, and the rk4v4 absence, exhaustively.
void test_equal_bit_set_at_450() {
    const MixTable table = build_table();
    const std::int32_t capacity = 450 * kLayers;   // 7200
    const auto it = table.find(capacity);
    check(it != table.end(), "a mix at exactly 450 x 16 layers exists");
    if (it == table.end()) { return; }

    std::set<std::string> signatures;
    int rk4v4_bearing = 0;
    for (const Mix& m : it->second) {
        signatures.insert(m.signature());
        if (m.has_rk4v4()) { ++rk4v4_bearing; }
        // Every one of them is the request filled to the last 0.01 bit.
        check(m.total == capacity, "every equal-bit mix is exactly on the 4.50 capacity");
    }
    std::printf("  equal-bit set at 4.50: %zu mixes, %zu distinct codec signatures, "
                "%d rk4v4-bearing (all at %.4f bits)\n", it->second.size(), signatures.size(),
                rk4v4_bearing, bits_of(capacity));
    // ⚠ 17 -> 120 AND 0 -> 63 (line `redkv`, dl/redkv/TRIAGE.md #55). The pre-flip grammar held
    // five slots and only nvfp4/iso4e shared a cost, so the exact-4.50 set was the 17 mixes
    // nvfp4^k + iso4e^(16-k). rk3v4 (375) and rk2v4 (325) left the "not selectable" clause, and
    // they are the bit-equal partners rk4v4 (425) never had: 2 x rk2v4 = 650 = 425 + 225 ... in
    // per-layer terms a -25 (rk4v4), -75 (rk3v4) and -125 (rk2v4) deviation can now be balanced
    // against the +375 (int8) one inside the 8-layer family cap. The old assertion "NO
    // rk4v4-bearing mix reaches 4.50: rk4v4 (425) has no bit-equal partner" was TRUE and is now
    // FALSE -- 63 of the 120 do -- and the deviation identity below is what still proves the
    // enumeration is the ladder's arithmetic rather than a number this test invented.
    check(it->second.size() == 120,
          "4.50 has 120 exact-bit mixes over the post-flip grammar, got " +
              std::to_string(it->second.size()));
    check(signatures.size() == 120, "all 120 are codec-DIFFERENT, i.e. 120 distinct signatures");
    check(signatures.count("nvfp4x16") == 1, "all-nvfp4 is in the set");
    check(signatures.count("iso4ex16") == 1, "all-iso4e is in the set (the alternative)");
    check(rk4v4_bearing == 63,
          "63 of the 120 carry row 4 (rk4v4) -- the row that had no bit-equal partner while its "
          "two narrower siblings were outside the grammar; got " + std::to_string(rk4v4_bearing));
    // The arithmetic that explains the count, asserted over EVERY mix rather than quoted:
    // relative to an all-450 baseline the per-layer deviations are
    // bf16 +1150, int8 +375, nvfp4/iso4e 0, rk4v4 -25, rk3v4 -75, rk2v4 -125.
    const std::int32_t rk4v4_slot = ninfer::product::detail::tier_index("rk4v4");
    const std::int32_t rk3v4_slot = ninfer::product::detail::tier_index("rk3v4");
    const std::int32_t rk2v4_slot = ninfer::product::detail::tier_index("rk2v4");
    const std::int32_t bf16_slot = ninfer::product::detail::tier_index("bf16");
    const std::int32_t int8_slot = ninfer::product::detail::tier_index("int8");
    const auto slots = candidate_slots();
    int mixes_checked = 0;
    for (const auto& [total, mixes] : table) {
        for (const Mix& m : mixes) {
            std::int32_t deviation = 0;
            for (std::size_t i = 0; i < slots.size(); ++i) {
                if (slots[i] == rk4v4_slot) { deviation += m.count[i] * -25; }
                else if (slots[i] == rk3v4_slot) { deviation += m.count[i] * -75; }
                else if (slots[i] == rk2v4_slot) { deviation += m.count[i] * -125; }
                else if (slots[i] == bf16_slot) { deviation += m.count[i] * 1150; }
                else if (slots[i] == int8_slot) { deviation += m.count[i] * 375; }
            }
            check(450 * kLayers + deviation == total,
                  "the deviation identity holds for " + m.signature());
            ++mixes_checked;
        }
    }
    check(mixes_checked > 1000, "the deviation identity really ran over the table (" +
                                    std::to_string(mixes_checked) + " mixes)");
}

// (3): the DEFAULT answer is the one tests/test_kv_budget_saturation.cpp pins. A selector
// that moved it would be a default change, not a slider.
void test_default_answer_at_450_is_the_cheapest_exact_450_mix() {
    const KvBitBudgetSolution solved =
        ninfer::product::kv_bit_budget_solve(kLayers, 4.50, kRk4v4Limit, 0);
    // ⚠ THE DEFAULT ANSWER MOVED, AND THEN MOVED BACK, FOR TWO DIFFERENT REASONS (line `redkv`,
    // dl/redkv/TRIAGE.md #55 for the first, landq/kvreach for the second).
    //   * It was `0-15:nvfp4` while rows 6/7 were unselectable.
    //   * The gate opened them, the ladder prices rk3v4 at 12 and rk2v4 at 16 against nvfp4's 30,
    //     and the saturating DP -- "maximum bits at or below the ceiling, then minimum penalty" --
    //     found the exact-4.50 mix `0-2:rk2v4,3-4:rk4v4,5:rk3v4,6-7:rk2v4,8-9:int8,10-15:nvfp4`
    //     costing 292 hundredths where all-nvfp4 costs 16 x 30 = 480.
    //   * Those two rows are now WITHHELD, because product::kv_dtype_for_storage refuses that very
    //     plan (product/kv_storage_dtype.h:107-123 and layouts_impl.h:681-684; the set is
    //     product/kv_bit_budget.h kv_bit_budget_deployable_rows(), derived from that decision
    //     point rather than written down). detail::kv_gear_solve_gated answers with the SHIPPED
    //     candidate set only while its answer is runnable and falls back to the deployable rows
    //     otherwise, so the answer at 4.50 is the cheapest exact-4.50 mix THE DEPLOY LAYER CAN
    //     BUILD -- all-nvfp4, at 480 hundredths.
    // THE RULE CHANGED IN NEITHER STEP, so the pin below is still an exact spec, an exact penalty
    // and an exact achieved bits; it also gained the assertion that no withheld row is in it.
    check(solved.spec == "0-15:nvfp4",
          "the default answer at 4.50 is the cheapest exact-4.50 mix the DEPLOY LAYER can build, "
          "got " +
              solved.spec);
    check(std::fabs(solved.achieved_bits - 4.50) < 1e-12,
          "the default answer is still exactly 4.5000, got " +
              std::to_string(solved.achieved_bits));
    check(std::fabs(solved.penalty - 4.80) < 1e-12,
          "and it costs 480 hundredths (all-nvfp4, 16 x 30) against the withheld mix's 292, got " +
              std::to_string(solved.penalty));
    check(solved.spec.find("iso4e") == std::string::npos,
          "the default does not use iso4e");
    check(solved.spec.find("rk3v4") == std::string::npos &&
              solved.spec.find("rk2v4") == std::string::npos,
          "and it uses NO row the deploy layer refuses (rk3v4/rk2v4 are withheld), got " +
              solved.spec);
// kvland-p12-menu-default
    // The pin that DID NOT MOVE: all-nvfp4 is still an exact-4.50 plan, so the answer above is a
    // tie-break on an equal-bit set and not a bit change.
    check(kKvBitBudgetTiers[3].bits_x100 * kLayers == 450 * kLayers &&
              kKvBitBudgetTiers[3].bits_x100 == 450,
          "all-nvfp4 is still exactly 4.5000 bits, so the exact-4.50 set this answer is drawn "
          "from is the same set the old answer came from");
    // The probe's finding, restated as a bounded claim: the solver's OWN knobs do not
    // reach the alternative. If this ever starts failing, the verdict in the report needs
    // revisiting rather than the test being relaxed.
    int distinct = 0;
    std::set<std::string> specs;
    for (const double w : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        for (const KvScoreContext ctx : {KvScoreContext::Long, KvScoreContext::Short}) {
            const auto scores = ninfer::product::kv_bit_budget_measured_scores(ctx);
            const KvBitBudgetSolution s = ninfer::product::kv_bit_budget_solve_scored(
                kLayers, 4.50, scores, w, kRk4v4Limit, 0);
            specs.insert(s.spec);
            ++distinct;
            check(std::fabs(s.achieved_bits - 4.50) < 1e-12,
                  "every weight saturates to 4.5000 (w=" + std::to_string(w) + ")");
        }
    }
    std::printf("  solver knobs swept: %d runs, %zu distinct specs\n", distinct, specs.size());
    check(specs.size() == 1,
          "over every weight and both measured tables the solver returns ONE spec, so the "
          "weight axis is not a route to the equal-bit option set");
}

// (4) THE HEADLINE: at ONE budget, two codec-different mixes are both selectable and both
// report the SAME achieved bits -- through the engine's own evaluator, not just the DP.
void test_two_codec_different_mixes_same_bits() {
    const auto scores = ninfer::product::kv_bit_budget_measured_scores(KvScoreContext::Long);
    int comparisons = 0;
    for (const double w : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const auto ladder = ninfer::product::kv_bit_budget_scored_ladder(scores, w);
        const auto rk4v4 = ninfer::product::kv_bit_budget_rk4v4_prefix_set(kRk4v4Limit);
        const KvGearSolution base =
            ninfer::product::kv_gear_solve_preferring(kLayers, 4.50, ladder, rk4v4, "nvfp4");
        const KvGearSolution alt =
            ninfer::product::kv_gear_solve_preferring(kLayers, 4.50, ladder, rk4v4, "iso4e");
        ++comparisons;
        // BOTH ARE SELECTABLE at this one budget ...
        check(base.feasible && alt.feasible, "both mixes are feasible at 4.50");
        // ... THEY DIFFER IN CODEC ...
        check(base.spec != alt.spec,
              "the two answers are different plans (w=" + std::to_string(w) + "): " +
                  base.spec + " vs " + alt.spec);
        check(base.spec.find("iso4e") == std::string::npos && alt.spec.find("nvfp4") == std::string::npos,
              "the two answers are codec-DIFFERENT (nvfp4-only vs iso4e-only)");
        // ... AND THEY REPORT THE SAME ACHIEVED BITS, exactly.
        check(std::fabs(base.achieved_bits - alt.achieved_bits) < 1e-12,
              "the two mixes report the SAME achieved bits (w=" + std::to_string(w) + "): " +
                  std::to_string(base.achieved_bits) + " vs " + std::to_string(alt.achieved_bits));
        check(std::fabs(alt.achieved_bits - 4.50) < 1e-12,
              "and that shared value is the request, 4.5000");
        // The equality is not the DP's own bookkeeping: the engine's spec evaluator, run
        // on both emitted spec strings, must agree.
        const auto ev_base =
            ninfer::product::kv_gear_evaluate_spec(base.spec, kLayers, 4.50, ladder, 0);
        const auto ev_alt =
            ninfer::product::kv_gear_evaluate_spec(alt.spec, kLayers, 4.50, ladder, 0);
        check(ev_base.parsed && ev_alt.parsed,
              "both emitted specs parse in the engine's own evaluator");
        check(std::fabs(ev_base.achieved_bits - ev_alt.achieved_bits) < 1e-12,
              "the engine's evaluator agrees the two specs cost the same bits");
        check(ev_base.within_budget && ev_alt.within_budget,
              "both are inside the 4.50 ceiling");
        if (w == 0.75) {
            std::printf("  w=0.75: %-12s vs %-12s -- both %.4f bits (engine evaluator: "
                        "%.4f / %.4f)\n",
                        base.spec.c_str(), alt.spec.c_str(), alt.achieved_bits,
                        ev_base.achieved_bits, ev_alt.achieved_bits);
        }
    }
    check(comparisons == 5, "the pair was compared at every weight");
}

// The UNPREFERRED answer of the SAME entry the preference knob belongs to: the request
// kv_gear_solve_preferring builds, with `candidate_order` -- the only field it sets -- left empty.
// See the note inside test_preference_never_overrides_a_strictly_worse_score for why the
// comparison partner may NOT be kv_bit_budget_solve.
KvGearSolution shipped_ladder_unpreferred(double budget) {
    ninfer::product::KvGearSolveRequest request;
    request.layers = kLayers;
    request.budget_bits = budget;
    request.ladder = kKvBitBudgetTiers;
    request.per_layer = ninfer::product::kv_gear_sets_from_rk4v4_set(
        kLayers, ninfer::product::kv_bit_budget_rk4v4_prefix_set(kRk4v4Limit));
    return ninfer::product::kv_gear_solve(request);
}

// (5) NEGATIVE CONTROL: the preference is NOT a penalty override.
void test_preference_never_overrides_a_strictly_worse_score() {
// kvland-p13-menu-helper
    const auto rk4v4 = ninfer::product::kv_bit_budget_rk4v4_prefix_set(kRk4v4Limit);
    // The incumbent is whatever the SHIPPED ladder's own prices return with no preference at all.
    // (There is no "empty preference" call: `kv_gear_solve_preferring` REFUSES a name that is not
    // a candidate gear, which is the control in (6). The unpreferred answer is the plain solver's.)
    //
    // ⚠ IT MUST BE THE SAME ENTRY, AND THAT IS NOT A DETAIL (landq/kvreach,
    // dl/_orch/landq/kvreach/PINS.md section 2, coordinate B). `kv_bit_budget_solve` and
    // `kv_gear_solve_preferring` are no longer one call: the first reaches the DP through
    // detail::kv_gear_solve_gated, which solves the SHIPPED candidate set first and falls back to
    // the rows the deploy layer accepts ONLY when the first answer is a plan it refuses, while the
    // preferring entry calls kv_gear_solve directly. At 4.50 on the shipped ladder that makes
    // `kv_bit_budget_solve` answer `0-15:nvfp4` (the deploy layer refuses the cheaper
    // `0-2:rk2v4,3-4:rk4v4,5:rk3v4,6-7:rk2v4,8-9:int8,10-15:nvfp4` mix) while
    // `kv_gear_solve_preferring` still returns that refused mix. Comparing THOSE two would compare
    // two different candidate sets and would redden for a reason that has nothing to do with the
    // preference, which is the only thing this control is about. So the partner is the same
    // request with no gear tried first -- which is what "no preference at all" means, and what the
    // one-line `kv_bit_budget_solve` call above meant before the gate existed.
    //
    // What it does NOT do: it does not lift the gate off `kv_gear_solve_preferring`. That entry is
    // deliberately left ungated (its only consumer is this test file; gating it would also change
    // the equal-bit pair control in (4), whose whole point is that TWO codec-different mixes are
    // reachable at one budget).
    const KvGearSolution incumbent = shipped_ladder_unpreferred(4.50);
// kvland-p14-menu-incumbent
    // ⚠ The needle moved with the answer (line `redkv`, dl/redkv/TRIAGE.md #55): the incumbent
    // used to be `0-15:nvfp4` because nvfp4 was the only exact-4.50 plan the grammar had. The
    // CONTROL is unchanged and is what is asserted: a preference for a codec whose shipped
    // penalty is STRICTLY WORSE (iso4e, 200 against the incumbent mix's own rows) must not move
    // the answer, and neither must a preference for nvfp4.
    const KvGearSolution prefer_iso4e = ninfer::product::kv_gear_solve_preferring(
        kLayers, 4.50, kKvBitBudgetTiers, rk4v4, "iso4e");
    check(prefer_iso4e.spec == incumbent.spec,
          "under the SHIPPED ladder iso4e's pin (200) still does NOT win: preferring a strictly "
          "worse score must NOT override it, got " + prefer_iso4e.spec + " vs " +
              incumbent.spec);
    check(std::fabs(prefer_iso4e.achieved_bits - 4.50) < 1e-12,
          "and the pinned answer is still exactly 4.5000");
    check(prefer_iso4e.spec.find("iso4e") == std::string::npos,
          "in particular the preference did not smuggle iso4e in, got " + prefer_iso4e.spec);
    // Same statement in the other direction: preferring nvfp4 -- whose shipped penalty is 30
    // against the incumbent mix's own 292/16 = 18.25 -- changes nothing either.
    const KvGearSolution same = ninfer::product::kv_gear_solve_preferring(
        kLayers, 4.50, kKvBitBudgetTiers, rk4v4, "nvfp4");
    check(same.spec == incumbent.spec, "preferring nvfp4 is a no-op too");
}

// (6) NEGATIVE CONTROL: nothing outside the candidate grammar is accepted silently.
void test_preference_refuses_non_candidate_gears() {
    const auto rk4v4 = ninfer::product::kv_bit_budget_rk4v4_prefix_set(kRk4v4Limit);
    // A gear OUTSIDE the candidate grammar. rk3v4/rk2v4 used to be the example; they are inside
    // it now (kKvGearCandidateSlots grew 6 -> 8 in the same item that made them selectable), so
    // the control uses names that are outside it on their own terms: the cold pseudo-tier, the
    // empty name, a name that is not a tier, and a name that differs from a real row by a suffix.
    for (const char* name : {"cold", "", "nvfp5", "rk4v4x", "iso4"}) {
        check_throws(
            [&] {
                (void)ninfer::product::kv_gear_solve_preferring(kLayers, 4.50, kKvBitBudgetTiers,
                                                                rk4v4, name);
            },
            std::string("prefer '") + name + "' is refused by name");
    }
    // ⭐ AND THE POSITIVE HALF, so "refused" and "accepted" are both pinned: the two rows the gate
    // opened are nameable now, they do not throw, and they do not move the achieved bits -- the
    // knob selects among equal-bit mixes and never among bit counts.
    // The unpreferred partner is the SAME entry, for the reason spelled out in (5): a gated
    // comparison partner would compare two candidate sets rather than the preference.
    const KvGearSolution no_pref = shipped_ladder_unpreferred(4.50);
    const KvGearSolution pref_nvfp4 = ninfer::product::kv_gear_solve_preferring(
        kLayers, 4.50, kKvBitBudgetTiers, rk4v4, "nvfp4");
// kvland-p15-menu-nopref
    check(pref_nvfp4.spec == no_pref.spec,
          "the declared ladder's unpreferred answer is the solver's own, got " + pref_nvfp4.spec +
              " vs " + no_pref.spec);
    for (const char* name : {"rk3v4", "rk2v4"}) {
        const KvGearSolution preferred = ninfer::product::kv_gear_solve_preferring(
            kLayers, 4.50, kKvBitBudgetTiers, rk4v4, name);
        check(preferred.feasible, std::string("prefer '") + name + "' is accepted at 4.50");
        check(std::fabs(preferred.achieved_bits - no_pref.achieved_bits) < 1e-12,
              std::string("prefer '") + name +
                  "' keeps the achieved bits, got " + std::to_string(preferred.achieved_bits) +
                  " vs " + std::to_string(no_pref.achieved_bits));
    }
}

// (7) THE CONTRACT OVER A SWEEP: the preference never changes the achieved bits, at any
// budget. "Same bits, different kind" is only a slider if the bits really are the same.
void test_preference_never_changes_the_bits() {
    const auto scores = ninfer::product::kv_bit_budget_measured_scores(KvScoreContext::Long);
    const auto ladder = ninfer::product::kv_bit_budget_scored_ladder(scores, 0.75);
    const auto rk4v4 = ninfer::product::kv_bit_budget_rk4v4_prefix_set(kRk4v4Limit);
    int plannable = 0;
    int changed = 0;
    for (double r = 4.30; r <= 16.0 + 1e-9; r += 0.05) {
        const double request = std::nearbyint(r * 100.0) / 100.0;
        KvGearSolution base;
        try {
            base = ninfer::product::kv_gear_solve_preferring(kLayers, request, ladder, rk4v4, "nvfp4");
        } catch (const std::exception&) {
            continue;
        }
        const KvGearSolution alt =
            ninfer::product::kv_gear_solve_preferring(kLayers, request, ladder, rk4v4, "iso4e");
        ++plannable;
        check(std::fabs(base.achieved_bits - alt.achieved_bits) < 1e-12,
              "preference keeps the achieved bits at request " + std::to_string(request) +
                  ": " + std::to_string(base.achieved_bits) + " vs " +
                  std::to_string(alt.achieved_bits));
        check(base.feasible == alt.feasible, "preference keeps feasibility");
        if (base.spec != alt.spec) { ++changed; }
    }
    std::printf("  sweep: %d plannable requests, %d where the preference changed the plan\n",
                plannable, changed);
    check(plannable > 100, "the same-bits sweep actually ran (" + std::to_string(plannable) +
                               " requests)");
    // NEGATIVE CONTROL for this sweep's own value: if the preference never changed anything
    // the "same bits" assertion would be vacuous.
    check(changed > 0, "the preference really does change the plan somewhere, so the "
                       "same-bits assertion is not vacuous");
}

}  // namespace

int main() {
    std::printf("=== test_kv_bit_mix_menu: the slider selects among EQUAL-BIT mixes ===\n");
    std::printf("rule: the bit count is an INPUT; the knob picks among codecs that cost the "
                "same bits\n");
    test_measured_columns_make_iso4e_and_nvfp4_a_tie();
    test_equal_bit_set_at_450();
    test_default_answer_at_450_is_the_cheapest_exact_450_mix();
    test_two_codec_different_mixes_same_bits();
    test_preference_never_overrides_a_strictly_worse_score();
    test_preference_refuses_non_candidate_gears();
    test_preference_never_changes_the_bits();
    if (g_failures == 0) {
        std::printf("PASS: the equal-bit set is enumerable over the WHOLE grammar, the default "
                    "answer is the cheapest exact-4.50 mix on the shipped ladder, two "
                    "codec-different mixes are selectable at one budget and report the SAME "
                    "achieved bits, and the preference is not a penalty override\n");
        return 0;
    }
    std::printf("%d FAILURE(S)\n", g_failures);
    return 1;
}
