// Host-only test for product/kv_kv_bits.h: the TWO K/V bit-width entries (joint =
// one overall ceiling; split = each plane's per-layer layering solved in its own
// budget) and the penalty table's own entry point.
//
// ⚠ RE-DERIVED (line `redkv`, dl/redkv/TRIAGE.md #52). The ladder's two narrowest K planes,
// rk3v4 (375) and rk2v4 (325), used to sit outside the solver's grammar: `selectable == false`
// AND absent from kKvGearCandidateSlots (product/kv_bit_budget.h:296, :1913). The gate opened in
// dl/e8mixwire/land/patch_flip_batch.py -- "one item, eight clusters", ruled by the owner's
// 2026-09-19 decision quoted at land/patch_bar_pins.py:118-124 ("Firing buys a selectable tier
// that is loudly refused where it cannot yet be served") -- so the DP may now choose them, the
// floor moved, and the equal-ceiling split reconciliation sees the new per-layer tiers. Every
// check below that the flip moved is re-derived from the tree, with the direction it moved named
// in place. No assertion was relaxed: the floor's new value is pinned from BOTH sides, and the
// split entry gained the deploy-or-refuse distinction the flip exposed.
//
// Everything here is host code. No device, no CUDA, no artifact: the header is
// std-only by construction, which is what lets one test pin the behaviours the
// planner, the CLI and the server all share.
//
// THE NEGATIVE CONTROLS ARE THE POINT. A test that only checks "the good case
// works" cannot tell a working oracle from one that always says yes, so:
//   * kv_kv_bits_pair_tier must return -1 for a realizable-looking pair that no
//     tier builds, and must NOT return -1 for the one pair the engine DOES build;
//   * the split entry must REFUSE (not approximate) a request whose per-layer
//     requirements divide, and must DEPLOY the equal-ceiling request -- the same
//     instrument reports both outcomes in the same run;
//   * a constant score table must make the slider inert, which is how we know the
//     slider's effect in the other cases is real.

#include "product/kv_kv_bits.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::KvBitsMode;
using ninfer::product::KvBitsPlan;
using ninfer::product::KvBitsRequest;
using ninfer::product::KvTierScoreTable;

int g_failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) { return; }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

template <typename Fn>
void check_throws(Fn&& fn, const std::string& needle, const std::string& what) {
    try {
        fn();
    } catch (const std::exception& exception) {
        const std::string text(exception.what());
        if (text.find(needle) == std::string::npos) {
            ++g_failures;
            std::fprintf(stderr, "FAIL: %s -- threw but the message lacks \"%s\":\n%s\n",
                         what.c_str(), needle.c_str(), text.c_str());
        }
        return;
    }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s -- did not throw\n", what.c_str());
}

constexpr std::int32_t kLayers = 16;

// ---------------------------------------------------------------- the oracle
void test_realizability_oracle() {
    using ninfer::product::kv_kv_bits_pair_tier;
    using ninfer::KvVCodec;

    // The pairs the engine really builds, at the default V codec.
    check(kv_kv_bits_pair_tier("bf16", "bf16", KvVCodec::Iso3) >= 0, "bf16/bf16 realizable");
    check(kv_kv_bits_pair_tier("i8", "i8", KvVCodec::Iso3) >= 0, "i8/i8 realizable");
    check(kv_kv_bits_pair_tier("fp8", "fp8", KvVCodec::Iso3) >= 0, "fp8/fp8 realizable");
    check(kv_kv_bits_pair_tier("iso4e", "iso4e", KvVCodec::Iso3) >= 0, "iso4e/iso4e realizable");
    // The engine's own two asymmetries.
    check(kv_kv_bits_pair_tier("e2m1", "iso4e", KvVCodec::Iso3) >= 0, "nvfp4 K=E2M1/V=ISO4E");
    check(kv_kv_bits_pair_tier("e8-lattice", "i4", KvVCodec::Iso3) >= 0, "rk4v4 K/V asymmetry");
    // ... and the SAME pair oracle cannot tell rk3v4/rk2v4 from rk4v4, because all three rows
    // carry (K=e8-lattice, V=i4) and differ only in the K plate's bit width
    // (product/kv_e8_width.h). That is why the entry's packing must take the DP's own per-layer
    // tier when the two planes agree -- asserted here so the reason the product fix exists is
    // stated where a reader of this file will meet it.
    check(ninfer::product::detail::kv_bits_hot_window_tier(6) == 6 &&
              ninfer::product::detail::kv_bits_hot_window_tier(7) == 7,
          "the two narrow rows are their OWN tiers, not aliases of rk4v4");
    // NEGATIVE CONTROL: the Rk4v4 tier is ASYMMETRIC (K = E8-lattice, V = i4), so asking
    // for an E8-lattice plane on BOTH sides -- which is what a symmetric-only model of
    // the ladder would say -- is NOT realizable. This is the control that catches an
    // oracle which reads one DType per tier and calls both planes equal.
    check(kv_kv_bits_pair_tier("e8-lattice", "e8-lattice", KvVCodec::Iso3) < 0,
          "NEGATIVE CONTROL: (e8-lattice, e8-lattice) is NOT realizable");
    check(kv_kv_bits_pair_tier("e2m1", "e2m1", KvVCodec::E2M1) >= 0, "nvfp4 K=V=E2M1");
    // NEGATIVE CONTROL 1: the cell the brief names. It must NOT be realizable.
    check(kv_kv_bits_pair_tier("iso4e", "e2m1", KvVCodec::E2M1) < 0,
          "NEGATIVE CONTROL: (K=ISO4E, V=E2M1) must not be realizable");
    // NEGATIVE CONTROL 2: cross-family pairs no tier builds either.
    check(kv_kv_bits_pair_tier("e2m1", "i8", KvVCodec::Iso3) < 0, "NEGATIVE: (E2M1, I8)");
    check(kv_kv_bits_pair_tier("e8-lattice", "i8", KvVCodec::Iso3) < 0,
          "NEGATIVE: (E8-lattice, I8)");
    check(kv_kv_bits_pair_tier("bf16", "iso4e", KvVCodec::Iso3) < 0, "NEGATIVE: (bf16, ISO4E)");
    // The two fills of the missing cell cost the SAME bits per element -- the reason
    // "just add a switch" cannot satisfy the request.
    check(ninfer::product::kv_bits_tier_bits(ninfer::product::kKvBitsIso4eIndex) ==
              ninfer::product::kv_bits_tier_bits(ninfer::product::kKvBitsNvfp4Index),
          "iso4e and nvfp4 cost the same bits per element");
}

// ---------------------------------------------------------------- entry 1: joint
void test_joint_entry() {
    const KvTierScoreTable scores = ninfer::product::kv_bit_budget_default_scores();
    KvBitsPlan plan;
    try {
        plan = ninfer::product::kv_kv_bits_entry_joint(kLayers, 4.50, scores, -1.0,
                                                       ninfer::product::kKvBitBudgetE8LayerLimit,
                                                       0, ninfer::KvVCodec::Iso3);
    } catch (const std::exception& exception) {
        check(false, std::string("joint entry at 4.50 threw: ") + exception.what());
        return;
    }
    check(plan.deployed && !plan.refused, "joint entry deploys");
    check(static_cast<std::int32_t>(plan.rows.size()) == kLayers, "joint rows cover every layer");
    check(plan.spec == plan.joint_spec, "joint spec == joint_spec");
    // ON/OFF EVIDENCE, pinned: the joint entry must agree BIT-FOR-BIT with the
    // pre-existing entry point at the same ceiling. Without this the "two entries"
    // could be two solvers that disagree, which is the whole risk of adding one.
    check(plan.spec == ninfer::product::kv_bit_budget_spec(kLayers, 4.50),
          "joint entry is byte-identical to kv_bit_budget_spec at the same ceiling");
    for (const auto& row : plan.rows) {
        check(row.k_bits == row.v_bits, "joint: every layer has k_bits == v_bits");
    }
    // The joint entry's displayed pair must be the ENGINE's pair, including the two
    // asymmetric tiers -- an rk4v4 layer is (e8-lattice, i4), not (e8-lattice, e8-lattice).
    for (const auto& row : plan.rows) {
        if (row.k_format == "e8-lattice") {
            check(row.v_format == "i4", "an rk4v4 layer's V plane is i4, not a second lattice");
        }
    }
}

// ---------------------------------------------------------------- entry 2: split
void test_split_entry_equal_ceilings_deploys() {
    // Equal ceilings: the two per-plane DPs are the SAME DP, so every layer
    // reconciles trivially and the split entry must DEPLOY. This is the default
    // "分开定" request working -- the old design threw on any k != v, and this is
    // the case that proves the blanket refusal is gone.
    const KvTierScoreTable scores = ninfer::product::kv_bit_budget_default_scores();
    for (const double ceiling : {4.50, 6.00, 8.00, 16.00}) {
        KvBitsPlan plan;
        try {
            plan = ninfer::product::kv_kv_bits_entry_split(
                kLayers, ceiling, ceiling, scores, scores, -1.0,
                ninfer::product::kKvBitBudgetE8LayerLimit, 0, ninfer::KvVCodec::Iso3);
        } catch (const std::exception& exception) {
            check(false, "split entry at equal ceiling " + std::to_string(ceiling) +
                             " threw: " + exception.what());
            continue;
        }
        check(plan.deployed, "split/equal " + std::to_string(ceiling) + " deploys");
        // ⭐ The packing is the one that MADE the rows (kv_kv_bits.h kv_bits_pack_rows): with
        // equal ceilings both per-plane solves are the same solve, so the emitted spec must be
        // the DP's own spec, rk3v4/rk2v4 rows included. A packing that collapsed the rk4v4
        // family onto slot 4 threw a std::logic_error from the "cannot happen" cross-check at
        // 4.50/6.00 before the product fix this line carries.
        check(plan.spec == plan.joint_spec, "split/equal spec == joint plan at min(k,v)");
        check(plan.spec == ninfer::product::kv_bit_budget_spec(kLayers, ceiling),
              "split/equal == the pre-existing entry at the same ceiling");
    }
}

void test_split_entry_divergent_ceilings_refuses_by_index() {
    // NEGATIVE CONTROL 3 (the one that matters): ceilings far enough apart that the
    // K ladder and the V ladder pick different tiers per layer. The entry must
    // REFUSE, must name the layers, and must carry the deployable joint plan.
    const KvTierScoreTable scores = ninfer::product::kv_bit_budget_default_scores();
    KvBitsPlan plan;
    try {
        plan = ninfer::product::kv_kv_bits_entry_split(
            kLayers, 4.50, 8.00, scores, scores, -1.0,
            ninfer::product::kKvBitBudgetE8LayerLimit, 0, ninfer::KvVCodec::Iso3);
    } catch (const std::exception& exception) {
        check(false, std::string("split entry at K=4.50/V=8.00 threw instead of refusing: ") +
                         exception.what());
        return;
    }
    check(plan.refused && !plan.deployed, "split/divergent refuses");
    check(plan.spec.empty(), "split/refused produces no spec");
    check(!plan.refusal.empty(), "split/refused carries a refusal text");
    check(plan.refusal.find("layer ") != std::string::npos,
          "refusal names the layer indices");
    check(plan.refusal.find("missing cell") != std::string::npos,
          "refusal names the missing cell");
    check(plan.refusal.find("--kv-bits-mode joint") != std::string::npos,
          "refusal carries the deployable joint reading");
    check(!plan.joint_spec.empty(), "refusal still computes a deployable joint plan");
    // And the joint plan it reports must be the real one at min(k,v).
    check(plan.joint_spec == ninfer::product::kv_bit_budget_spec(kLayers, 4.50),
          "refusal's joint plan is the real plan at min(k,v)");
    // The divergent ceilings must have produced genuinely different per-plane
    // allocations -- otherwise the refusal above would be about nothing.
    std::size_t divergent = 0;
    for (const auto& row : plan.rows) {
        if (row.k_tier != row.v_tier) { ++divergent; }
    }
    check(divergent > 0, "the two ladders really did lay out differently");
}

void test_split_entry_recorded_pair_refuses() {
    // The brief's cell, reached through the real entry points rather than through the
    // oracle: ask for an ISO4E K stack (K ceiling 4.50 with the iso4e-friendly slider is
    // not enough -- use the per-plane score tables to drive K onto iso4e and V onto
    // E2M1) and check the refusal names (K=iso4e, V=e2m1).
    //
    // K scores: iso4e cheapest quality (it IS the same planes as nvfp4); V scores: e2m1
    // cheapest. That is exactly the request "K iso4e, V e2m1".
    KvTierScoreTable k_scores = ninfer::product::kv_bit_budget_default_scores();
    KvTierScoreTable v_scores = ninfer::product::kv_bit_budget_default_scores();
    k_scores[ninfer::product::kKvBitsIso4eIndex] = {0, 0};   // K: iso4e is the best choice
    k_scores[ninfer::product::kKvBitsNvfp4Index] = {900, 900};
    v_scores[ninfer::product::kKvBitsIso4eIndex] = {900, 900};  // V: never iso4e
    v_scores[ninfer::product::kKvBitsNvfp4Index] = {0, 0};
    KvBitsPlan plan;
    try {
        plan = ninfer::product::kv_kv_bits_entry_split(
            kLayers, 4.50, 4.50, k_scores, v_scores, 0.0,
            ninfer::product::kKvBitBudgetE8LayerLimit, 0, ninfer::KvVCodec::E2M1);
    } catch (const std::exception& exception) {
        check(false, std::string("recorded-pair split threw instead of refusing: ") +
                         exception.what());
        return;
    }
    // K's ladder must land on iso4e for every layer and V's on nvfp4 for every layer,
    // because those are the only two rows the per-plane tables make cheap; with the
    // E2M1 V codec that pair is (K=iso4e, V=e2m1), which no tier builds.
    for (const auto& row : plan.rows) {
        check(row.k_format == "iso4e", "K ladder resolved to iso4e on layer " +
                                          std::to_string(row.layer));
        check(row.v_format == "e2m1", "V ladder resolved to e2m1 on layer " +
                                          std::to_string(row.layer));
    }
    check(plan.refused && !plan.deployed,
          "the (K=iso4e, V=e2m1) request is refused, not approximated");
    check(plan.refusal.find("(K=iso4e, V=e2m1)") != std::string::npos,
          "the refusal names the missing cell (K=iso4e, V=e2m1)");
    check(plan.refusal.find("layer 0") != std::string::npos,
          "the refusal names layer 0");
    check(plan.refusal.find("4.50") != std::string::npos,
          "the refusal records that both fills of the cell cost 4.50 b/element");
}

// ---------------------------------------------------------------- the score table
void test_score_table_round_trip() {
    const KvTierScoreTable table = ninfer::product::kv_bit_budget_default_scores();
    const std::string text = ninfer::product::kv_bits_score_table_text(table, "unit test");
    // The emitted table MUST load back through the real loader, otherwise the entry
    // point produces a file nothing can consume.
    const KvTierScoreTable loaded = ninfer::product::kv_bit_budget_parse_scores(text);
    for (std::size_t i = 0; i < table.size(); ++i) {
        check(table[i].quality_x100 == loaded[i].quality_x100 &&
                  table[i].speed_x100 == loaded[i].speed_x100,
              "score table row " + std::to_string(i) + " survives the round trip");
    }
    // NEGATIVE CONTROL 4: a table missing a tier must be REFUSED by the loader, so
    // "the round trip worked" is not the loader accepting anything.
    std::string truncated = text;
    const std::size_t last = truncated.rfind("iso4e");
    check(last != std::string::npos, "test can find the last tier row");
    truncated.resize(truncated.rfind('\n', last));
    check_throws([&] { (void)ninfer::product::kv_bit_budget_parse_scores(truncated); },
                 "missing tier", "NEGATIVE CONTROL: a table missing a tier is refused");
}

void test_provenance_is_honest() {
    using ninfer::product::kv_scores_provenance;
    const auto builtin = kv_scores_provenance("");
    check(builtin.provisional, "the built-in table is reported as provisional");
    check(!builtin.quality_measured, "the built-in QUALITY column is NOT claimed measured");
    check(builtin.speed_measured, "the built-in SPEED column IS claimed measured");
    check(builtin.line.find("PRIOR") != std::string::npos, "the line says PRIOR");
    check(builtin.line.find("provisional") != std::string::npos,
          "the line cites the tree's own word for the table");
    const auto file = kv_scores_provenance("somewhere.txt");
    check(!file.provisional, "a file is not called provisional");
    check(!file.quality_measured && !file.speed_measured,
          "an operator file makes no measurement claim in either direction");
}

// ---------------------------------------------------------------- request plumbing
void test_request_resolution() {
    using ninfer::product::kv_bits_mode_from_name;
    using ninfer::product::kv_bits_mode_name;
    check(kv_bits_mode_from_name("joint") == KvBitsMode::Joint, "mode name joint");
    check(kv_bits_mode_from_name("split") == KvBitsMode::Split, "mode name split");
    check(kv_bits_mode_from_name("ceiling") == KvBitsMode::Ceiling, "mode name ceiling");
    check_throws([&] { (void)kv_bits_mode_from_name("nonsense"); }, "invalid kv-bits-mode",
                 "NEGATIVE CONTROL: an unknown mode is refused");

    const KvTierScoreTable scores = ninfer::product::kv_bit_budget_default_scores();

    // NEGATIVE CONTROL 5: naming BOTH spellings of the ceiling is a contradiction.
    {
        KvBitsRequest request;
        request.joint_bits = 6.0;
        request.k_bits     = 4.0;
        check_throws(
            [&] {
                (void)ninfer::product::kv_kv_bits_resolve(request, kLayers, scores, scores,
                                                          scores,
                                                          ninfer::product::kKvBitBudgetE8LayerLimit,
                                                          0, ninfer::KvVCodec::Iso3);
            },
            "two spellings of the same ceiling set",
            "NEGATIVE CONTROL: --kv-bits together with --kv-k-bits is refused");
    }
    // A one-sided request under the default (split) mode inherits the named ceiling, so
    // the old "V-only" request still works and still means the same allocation.
    {
        KvBitsRequest request;
        request.v_bits = 8.0;
        const KvBitsPlan plan = ninfer::product::kv_kv_bits_resolve(
            request, kLayers, scores, scores, scores,
            ninfer::product::kKvBitBudgetE8LayerLimit, 0, ninfer::KvVCodec::Iso3);
        check(plan.deployed, "a V-only request deploys");
        check(plan.spec == ninfer::product::kv_bit_budget_spec(kLayers, 8.0),
              "a V-only request is the same plan the pre-existing entry gave");
    }
    // The ceiling reading is opt-in and must be BOTH deployable and explicit.
    {
        KvBitsRequest request;
        request.k_bits = 4.50;
        request.v_bits = 8.00;
        request.mode   = KvBitsMode::Ceiling;
        const KvBitsPlan plan = ninfer::product::kv_kv_bits_resolve(
            request, kLayers, scores, scores, scores,
            ninfer::product::kKvBitBudgetE8LayerLimit, 0, ninfer::KvVCodec::Iso3);
        check(plan.deployed, "mode=ceiling deploys");
        check(plan.mode == KvBitsMode::Ceiling, "mode=ceiling reports itself");
        check(plan.report.find("UNSPENDABLE") != std::string::npos,
              "mode=ceiling reports the headroom it could not spend");
        check(plan.spec == ninfer::product::kv_bit_budget_spec(kLayers, 4.50),
              "mode=ceiling spends min(k,v) on both planes");
    }
}

// ---------------------------------------------------------------- the slider
void test_slider_on_off_and_negative_control() {
    // ON/OFF: the slider must change the allocation at a ceiling where the ladder has a
    // real choice, and it must NOT change it where the ceiling already forces one.
    const std::string fast = ninfer::product::kv_bit_budget_solve_scored(
                                 kLayers, 5.00, ninfer::product::kv_bit_budget_default_scores(),
                                 0.0, ninfer::product::kKvBitBudgetE8LayerLimit, 0)
                                 .spec;
    const std::string accurate =
        ninfer::product::kv_bit_budget_solve_scored(
            kLayers, 5.00, ninfer::product::kv_bit_budget_default_scores(), 1.0,
            ninfer::product::kKvBitBudgetE8LayerLimit, 0)
            .spec;
    check(fast != accurate, "ON/OFF: the slider changes the allocation at 5.00 b/element");

    // NEGATIVE CONTROL 6: with both columns constant the weight cannot matter, so the
    // instrument above is proven able to report "no change" as well as "change".
    KvTierScoreTable flat{};
    for (auto& row : flat) { row = {10, 10}; }
    const std::string flat0 = ninfer::product::kv_bit_budget_solve_scored(
                                  kLayers, 5.00, flat, 0.0,
                                  ninfer::product::kKvBitBudgetE8LayerLimit, 0)
                                  .spec;
    const std::string flat1 = ninfer::product::kv_bit_budget_solve_scored(
                                  kLayers, 5.00, flat, 1.0,
                                  ninfer::product::kKvBitBudgetE8LayerLimit, 0)
                                  .spec;
    check(flat0 == flat1, "NEGATIVE CONTROL: a constant score table makes the slider inert");
}

// ---------------------------------------------------------------- the ladder floor
void test_below_floor_is_named() {
    const KvTierScoreTable scores = ninfer::product::kv_bit_budget_default_scores();
    // ⚠ THE FLOOR MOVED WITH THE GATE (line `redkv`, dl/redkv/TRIAGE.md #52). 4.25 WAS the
    // floor while rows 6/7 were unselectable: the cheapest admissible 16-layer allocation was
    // `rk4v4 x 8 + nvfp4 x 8` = (8*425 + 8*450)/16 = 4.375, and the rk4v4 exposure limit
    // (kKvBitBudgetE8LayerLimit = 8) is what stopped it going lower. With rk3v4 (375) and rk2v4
    // (325) selectable the same cap buys `rk2v4 x 8 + nvfp4 x 8` = (8*325 + 8*450)/16 = 3.875,
    // so 4.25 is a FEASIBLE ceiling now and the refusal moved down with it. Both directions are
    // asserted, so the move is pinned rather than merely observed.
    check_throws(
        [&] {
            (void)ninfer::product::kv_kv_bits_entry_joint(
                kLayers, 3.50, scores, -1.0, ninfer::product::kKvBitBudgetE8LayerLimit, 0,
                ninfer::KvVCodec::Iso3);
        },
        "no feasible allocation", "below-floor joint ceiling is refused");
    check_throws(
        [&] {
            (void)ninfer::product::kv_kv_bits_entry_split(
                kLayers, 3.50, 3.50, scores, scores, -1.0,
                ninfer::product::kKvBitBudgetE8LayerLimit, 0, ninfer::KvVCodec::Iso3);
        },
        "rk4v4 exposure limit", "below-floor split ceiling names the rk4v4 cap");
    // ... and the ceiling that used to be below the floor is now ABOVE it: the two rows the gate
    // opened moved the floor by exactly (8*(425-325))/16 = 0.50 bits/element, and this is the
    // pair that pins WHICH rows did it (a floor that moved for an unrelated reason would take
    // 4.25 with it and this check would still pass -- the split/equal check in
    // test_split_entry_equal_ceilings_deploys is what says the answer is deployable).
    {
        const KvBitsPlan plan = ninfer::product::kv_kv_bits_entry_joint(
            kLayers, 4.25, scores, -1.0, ninfer::product::kKvBitBudgetE8LayerLimit, 0,
            ninfer::KvVCodec::Iso3);
        check(plan.deployed && !plan.refused, "4.25 is a feasible ceiling now");
        check(plan.spec.find("rk2v4") != std::string::npos ||
                  plan.spec.find("rk3v4") != std::string::npos,
              "and the rows that moved the floor are IN the answer, got " + plan.spec);
    }
}

// ------------------------------------------- named tables must never be dropped
// Every case below is a flag combination that used to be ACCEPTED AND READ BY NOTHING.
// The refusal has to name the flag that would make it act; a refusal that does not is not
// actionable, so the needle is the flag name and not just "it threw".
void test_named_tables_are_never_dropped() {
    const KvTierScoreTable scores = ninfer::product::kv_bit_budget_default_scores();
    const auto resolve = [&](const KvBitsRequest& request) {
        return ninfer::product::kv_kv_bits_resolve(
            request, kLayers, scores, scores, scores,
            ninfer::product::kKvBitBudgetE8LayerLimit, 0, ninfer::KvVCodec::Iso3);
    };

    // 1. A named table with no weight: the two columns are combined only by the weight.
    {
        KvBitsRequest request;
        request.joint_bits      = 4.50;
        request.scores_explicit = true;
        check_throws([&] { (void)resolve(request); }, "--kv-quality-weight",
                     "a score table without --kv-quality-weight is refused, naming the flag");
    }
    // 2. --kv-bits-mode with only --kv-bits: one ceiling, nothing for the mode to select.
    for (const KvBitsMode mode : {KvBitsMode::Split, KvBitsMode::Ceiling}) {
        KvBitsRequest request;
        request.joint_bits    = 4.50;
        request.mode          = mode;
        request.mode_explicit = true;
        check_throws([&] { (void)resolve(request); }, "--kv-bits-mode",
                     "an explicit non-joint mode with only --kv-bits is refused");
    }
    // 2b. CONTROL: the consistent spelling is still accepted -- it is honoured, not ignored.
    {
        KvBitsRequest request;
        request.joint_bits    = 4.50;
        request.mode          = KvBitsMode::Joint;
        request.mode_explicit = true;
        const KvBitsPlan plan = resolve(request);
        check(plan.deployed, "CONTROL: --kv-bits --kv-bits-mode joint is still accepted");
        check(plan.spec == ninfer::product::kv_bit_budget_spec(kLayers, 4.50),
              "CONTROL: and it is the same plan the default reading gives");
    }
    // 3. A per-plane table in the JOINT reading: one ladder, nothing to fit.
    {
        KvBitsRequest request;
        request.joint_bits      = 4.50;
        request.k_scores_explicit = true;
        check_throws([&] { (void)resolve(request); }, "--kv-k-tier-scores",
                     "a per-plane table in the JOINT reading is refused, naming the flag");
    }
    // 4. The generic table is the FALLBACK in the split reading, not dead weight: naming it
    //    must change the outcome, and a NAMED per-plane table must win over it.
    {
        // v_only must prefer rk4v4 and extreme must prefer nvfp4, so that "the per-plane table
        // won" cannot be confused with "it was ignored and the generic one ran".
        KvTierScoreTable v_only = scores;
        for (auto& row : v_only) { row.quality_x100 = 900; row.speed_x100 = 900; }
        v_only[ninfer::product::kKvBitsE8Index] = {0, 0};
        KvTierScoreTable extreme = scores;
        for (auto& row : extreme) { row = {900, 900}; }
        extreme[ninfer::product::kKvBitsNvfp4Index] = {0, 0};

        KvBitsRequest base;
        base.k_bits          = 4.50;
        base.v_bits          = 4.50;
        base.quality_weight  = 1.0;
        base.scores_explicit = true;
        const KvTierScoreTable& generic = extreme;
        const KvBitsPlan used_generic = ninfer::product::kv_kv_bits_resolve(
            base, kLayers, generic, scores, scores,
            ninfer::product::kKvBitBudgetE8LayerLimit, 0, ninfer::KvVCodec::Iso3);

        KvBitsRequest plane_wins = base;
        plane_wins.v_scores_explicit = true;
        const KvBitsPlan per_plane = ninfer::product::kv_kv_bits_resolve(
            plane_wins, kLayers, generic, scores, v_only,
            ninfer::product::kKvBitBudgetE8LayerLimit, 0, ninfer::KvVCodec::Iso3);
        check(per_plane.report != used_generic.report,
              "a NAMED per-plane table wins over the generic fallback");
    }
    // 5. And the same table IS inert where it is not read: with no weight at all it is
    //    refused rather than silently dropped (case 1), which is what makes case 4 mean
    //    something.
    {
        KvBitsRequest request;
        request.k_bits            = 4.50;
        request.v_bits            = 4.50;
        request.k_scores_explicit = true;
        check_throws([&] { (void)resolve(request); }, "--kv-quality-weight",
                     "a per-plane table without a weight is refused too");
    }
}

} // namespace

int main() {
    test_realizability_oracle();
    test_joint_entry();
    test_split_entry_equal_ceilings_deploys();
    test_split_entry_divergent_ceilings_refuses_by_index();
    test_split_entry_recorded_pair_refuses();
    test_score_table_round_trip();
    test_provenance_is_honest();
    test_request_resolution();
    test_slider_on_off_and_negative_control();
    test_below_floor_is_named();
    test_named_tables_are_never_dropped();
    if (g_failures != 0) {
        std::fprintf(stderr, "kv_kv_bits_test: %d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::fprintf(stdout, "kv_kv_bits_test: all checks passed\n");
    return 0;
}
