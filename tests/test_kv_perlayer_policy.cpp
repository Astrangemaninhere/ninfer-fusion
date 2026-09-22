// KV-PERLAYER: host test for plan/kv_perlayer_policy.h. No CUDA, no GPU, run-level.
//
// It exercises the delivered predicates and, above all, the red control:
//   * per-layer evidence           -> the policy must give a PER-LAYER verdict;
//   * the SAME evidence collapsed  -> the policy must refuse, and the invariant
//     to ONE GLOBAL SCALAR            checker must go RED if a caller nevertheless
//                                     reports a per-layer table.
// The broken caller at the end is deliberately an implementation of exactly the
// deliverable the user rejected ("跑一个数字分配kv"), so the red is not a tautology:
// a plausible caller really does trip it.
//   * case I (SELWIRE-LAND) closes the three holes SELWIRE2-X named: the free
//     `from_factory_table` exemption, the length-coincidence replacement predicate, and the
//     silent "measured but no tier declared" cell.
#include "product/kv_perlayer_policy.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace ninfer::product;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    // Flush per check: if a LATER check crashes, the already-printed FAIL lines must still reach the
    // log. A buffered FAIL that never lands looks exactly like a pass in the file, and the criterion
    // set here is {rc, checks} -- a crash may cost the run, never the evidence.
    std::fflush(stdout);
    if (!ok) { ++g_fail; }
}

static bool any_contains(const std::vector<std::string>& v, const char* needle) {
    for (const auto& s : v) { if (s.find(needle) != std::string::npos) { return true; } }
    return false;
}

static std::vector<KvPerLayerEvidence> per_layer_fixture() {
    std::vector<KvPerLayerEvidence> ev;
    for (int i = 0; i < 16; ++i) {
        KvPerLayerEvidence e;
        e.layer          = i;
        e.energy_observed = true;
        e.energy_mean_l   = 0.10 + 0.01 * static_cast<double>(i);
        e.energy_rounds   = 512;
        e.k_err_measured  = true;
        e.v_err_measured  = true;
        e.k_err_nmse = (i % 4 == 0) ? 1e-6 : ((i % 4 == 1) ? 5e-4 : ((i % 4 == 2) ? 5e-5 : 2e-3));
        e.v_err_nmse = (i % 3 == 0) ? 1e-6 : 3e-4;
        // The tier these numbers were measured at. Required by the tier-of-evidence gate:
        // an error without a declared tier meets no tier-specific bound.
        e.err_tier    = "nvfp4";
        ev.push_back(e);
    }
    return ev;
}

// "RUN ONE NUMBER": every layer gets the SAME measurement -- the global mean.
static std::vector<KvPerLayerEvidence> global_scalar_fixture() {
    std::vector<KvPerLayerEvidence> ev;
    for (int i = 0; i < 16; ++i) {
        KvPerLayerEvidence e;
        e.layer = i;
        e.energy_observed = true;
        e.energy_mean_l   = 0.185;
        e.energy_rounds   = 512;
        e.k_err_measured  = true;
        e.v_err_measured  = true;
        e.k_err_nmse      = 6.5e-4;
        e.v_err_nmse      = 2.0e-4;
        e.err_tier        = "nvfp4";
        ev.push_back(e);
    }
    return ev;
}

// Case G reads the SAME fixture as case A on purpose: the point is that one error column
// means different things depending on the tier it declares.
static std::vector<KvPerLayerEvidence> tier_fixture() { return per_layer_fixture(); }

int main() {
    std::printf("=== KV-PERLAYER policy test (run-level, host-only) ===\n");

    // ------------------------------------------------------------------ case A
    std::printf("\nA. per-layer evidence (16 layers, distinct measurements)\n");
    KvPerLayerPolicyConfig cfg;
    cfg.drop_enabled = true;
    const auto a = kv_perlayer_decide(per_layer_fixture(), cfg);
    std::printf("  %s\n", a.report.c_str());
    std::printf("  layer:");
    for (const auto& v : a.rows) { std::printf(" %2d", v.layer); }
    std::printf("\n  tier :");
    for (const auto& v : a.rows) { std::printf(" %2s", v.tier.empty() ? "-" : v.tier.substr(0, 2).c_str()); }
    std::printf("\n  drop :");
    for (const auto& v : a.rows) { std::printf(" %2s", v.drop_admissible ? "D" : "."); }
    std::printf("\n");
    check(a.per_layer_information,  "A: per_layer_information == true");
    check(!a.degenerate_input,      "A: degenerate_input == false");
    check(kv_perlayer_is_decisive(a), "A: the verdict vector differs between layers (a real per-layer judgement)");
    check(kv_perlayer_invariant_violations(a).empty(), "A: no invariant violations");
    check(kv_perlayer_declaration_violations(a).empty(),
          "A: no declaration violations (the caller declared its tier)");
    int drops = 0, opined = 0;
    for (const auto& v : a.rows) { if (v.drop_admissible) ++drops; if (!v.tier.empty()) ++opined; }
    std::printf("  tiers issued=%d  drops admitted=%d\n", opined, drops);
    check(drops > 0, "A: at least one layer clears the MEASURED drop predicate");
    // reasons must be attached to every measured verdict (I3)
    bool all_reasoned = true;
    for (const auto& v : a.rows) { if (v.reasons.empty()) { all_reasoned = false; } }
    check(all_reasoned, "A: every verdict carries a reason string (no unexplained decision)");

    // ------------------------------------------------------------------ case B
    std::printf("\nB. THE DEGRADATION: one global number smeared over every layer\n");
    const auto b = kv_perlayer_decide(global_scalar_fixture(), cfg);
    std::printf("  %s\n", b.report.c_str());
    check(b.degenerate_input,        "B: degenerate_input == true  (the policy NOTICED)");
    check(!b.per_layer_information,  "B: per_layer_information == false");
    check(!kv_perlayer_is_decisive(b), "B: the verdict vector is uniform (no per-layer judgement)");
    check(kv_perlayer_invariant_violations(b).empty(), "B: the policy's own output is self-consistent (it abstained)");
    bool all_unmeasured = true;
    for (const auto& v : b.rows) {
        if (v.kind != KvVerdictKind::Unmeasured || !v.tier.empty() || v.drop_admissible) {
            all_unmeasured = false;
        }
    }
    check(all_unmeasured, "B: every row is Unmeasured with no tier and no drop (it abstained, it did not guess)");
    std::printf("  => a global single number yields NO per-layer verdict. This is the rejection\n"
                "     of \"跑一个数字分配 kv\", enforced by the policy itself.\n");

    // ------------------------------------------------------------------ case C
    // THE RED. A plausible caller -- the exact deliverable the user rejected -- takes
    // the collapsed evidence and DOES report a per-layer table from it (here by
    // copying the global tier onto every layer, the only thing a single number can do).
    // The invariant checker must go red.
    std::printf("\nC. RED CONTROL: a caller that reports a per-layer table FROM the global scalar\n");
    KvPerLayerDecision forged = b;              // starts from the honest abstention...
    for (auto& v : forged.rows) {               // ...and fabricates a per-layer opinion
        v.kind            = KvVerdictKind::Measured;
        v.tier            = "nvfp4";            // "the one global number says nvfp4"
        v.drop_admissible = false;
        v.reasons.push_back("assigned from the global mean");   // non-empty: I3 will not catch it
    }
    forged.per_layer_information = true;        // the caller ASSERTS per-layer information
    const auto bad = kv_perlayer_invariant_violations(forged);
    std::printf("  invariant violations reported: %zu\n", bad.size());
    for (const auto& s : bad) { std::printf("    RED: %s\n", s.c_str()); }
    check(!bad.empty(), "C: *** THE CHECKER GOES RED on a global scalar presented as a per-layer table ***");
    bool saw_per_layer_claim = false;
    for (const auto& s : bad) {
        if (s.find("per_layer_information=true") != std::string::npos) { saw_per_layer_claim = true; }
    }
    check(saw_per_layer_claim, "C: and it red-flags the specific claim (degenerate input reported as per-layer)");
    check(!kv_perlayer_is_decisive(forged),
          "C: the forged table is itself UNIFORM -- proof that the global number carried no per-layer content");

    // ------------------------------------------------------------------ case D
    // Honest default: no measurements at all. The policy must abstain, not invent.
    std::printf("\nD. no evidence at all (a fresh box, NINFER_FT_STATS off, no calib corpus)\n");
    std::vector<KvPerLayerEvidence> none;
    for (int i = 0; i < 16; ++i) { KvPerLayerEvidence e; e.layer = i; none.push_back(e); }
    const auto d = kv_perlayer_decide(none, cfg);
    std::printf("  %s\n", d.report.c_str());
    check(!d.per_layer_information, "D: per_layer_information == false");
    check(!d.degenerate_input,      "D: all-unmeasured is NOT called degenerate (it is the honest initial state)");
    check(kv_perlayer_invariant_violations(d).empty(), "D: no invariant violations");
    bool abstained = true;
    for (const auto& v : d.rows) { if (!v.tier.empty() || v.drop_admissible) { abstained = false; } }
    check(abstained, "D: every layer abstained -- the global mechanism decides, today's behaviour preserved");

    // ------------------------------------------------------------------ case E
    // The load-bearing clause: energy is a COST signal. Real per-layer energy, but no
    // error instrument => NO drop is admissible anywhere, whatever the ranking says.
    std::printf("\nE. real per-layer ENERGY but no error instrument\n");
    auto energy_only = per_layer_fixture();
    for (auto& e : energy_only) { e.k_err_measured = false; e.v_err_measured = false; }
    const auto e_dec = kv_perlayer_decide(energy_only, cfg);
    std::printf("  %s\n", e_dec.report.c_str());
    int e_drops = 0;
    for (const auto& v : e_dec.rows) { if (v.drop_admissible) { ++e_drops; } }
    check(e_drops == 0, "E: ZERO drops -- a cost signal never justifies discarding a plane");

    // ------------------------------------------------------------------ case F
    // The drop axis off must be inert, whatever the evidence.
    std::printf("\nF. drop axis NOT opted in (the default config)\n");
    KvPerLayerPolicyConfig off;   // drop_enabled == false
    const auto f = kv_perlayer_decide(per_layer_fixture(), off);
    int f_drops = 0;
    for (const auto& v : f.rows) { if (v.drop_admissible) { ++f_drops; } }
    std::printf("  %s\n", f.report.c_str());
    check(f_drops == 0, "F: drop_enabled=false => zero drops even with perfect evidence");

    // ------------------------------------------------------------------ case G
    // THE TIER-OF-EVIDENCE RED. Two defects, one gate. The SAME error column, fed with no
    // tier declared (which is the ONLY thing the type allowed before this change) used to
    // meet every tier-specific bound silently; and a column measured at a tier that is not
    // the tier the layer would carry used to justify a DROP. Both are now refusals that
    // name the tier involved, and the evidence-side invariant (I6) reports the undeclared
    // and the unbounded case on the INPUT, so no caller can lose the message.
    std::printf("\nG. TIER-OF-EVIDENCE: one error column, three declarations\n");
    struct TierDecl { const char* tier; const char* meaning; };
    const TierDecl tier_decls[] = {
        {"nvfp4", "the tier this fixture's numbers claim to be measured at -> bounds speak"},
        {"rk4v4",    "a rung, but NOT the tier these layers carry -> the rk4v4 bound is withheld"},
        {"fp8",   "a real tier, but NOT a rung of this ladder -> every bound is withheld"},
        {"",      "no tier declared -> every bound is withheld and I6 goes red"},
    };
    for (const TierDecl& decl : tier_decls) {
        auto ev = tier_fixture();
        for (auto& e : ev) { e.err_tier = decl.tier; }
        const auto g = kv_perlayer_decide(ev, cfg);
        int g_drops = 0, g_opined = 0;
        bool saw_named_refusal = false;
        for (const auto& v : g.rows) {
            if (v.drop_admissible) { ++g_drops; }
            if (!v.tier.empty())   { ++g_opined; }
            for (const auto& r : v.reasons) {
                if (r.find("REFUSED: ") != std::string::npos) { saw_named_refusal = true; }
            }
        }
        const auto ev_bad = kv_perlayer_evidence_violations(ev);
        std::printf("  err_tier='%-5s' with_tier=%2d drops=%2d evidence_violations=%2zu named_refusal=%s\n"
                    "      %s\n",
                    decl.tier, g_opined, g_drops, ev_bad.size(),
                    saw_named_refusal ? "yes" : "no ", decl.meaning);
        check(kv_perlayer_invariant_violations(g).empty(), "G: the verdict vector stays self-consistent");
        if (std::string(decl.tier) == "nvfp4") {
            check(g_opined + g_drops > 0, "G: a DECLARED, on-ladder tier still decides (the gate is not a blanket refusal)");
            check(ev_bad.empty(),         "G: I6 clean for a declared, on-ladder tier");
        }
        if (std::string(decl.tier) == "rk4v4") {
            // The consequence this gate exists for: an rk4v4 measurement must not reach the
            // nvfp4 bound, and a DROP may only be admitted for a layer whose rk4v4 error really
            // clears the rk4v4 bound (those are the layers the ladder would legitimately place
            // on rk4v4). Before the gate this same column was 13/16 nvfp4 and 3/16
            // DROP-ADMISSIBLE, one of them on a layer whose rk4v4 error is 5x the rk4v4 bound.
            int g_nvfp4 = 0, drops_above_rk4v4_bound = 0, drops_below_rk4v4_bound = 0;
            for (std::size_t i = 0; i < g.rows.size(); ++i) {
                if (g.rows[i].tier == "nvfp4") { ++g_nvfp4; }
                if (g.rows[i].drop_admissible) {
                    const double err = std::max(ev[i].k_err_nmse, ev[i].v_err_nmse);
                    if (err <= cfg.rk4v4_nmse_max) { ++drops_below_rk4v4_bound; } else { ++drops_above_rk4v4_bound; }
                }
            }
            check(g_nvfp4 == 0, "G: *** the nvfp4 bound is NOT met by an rk4v4 measurement ***");
            check(drops_above_rk4v4_bound == 0,
                  "G: *** no drop is admitted from the rk4v4 column for a layer whose rk4v4 error is above "
                  "the rk4v4 bound ***");
            check(drops_below_rk4v4_bound > 0,
                  "G: control -- layers whose rk4v4 error really clears the rk4v4 bound stay droppable");
        }
        if (std::string(decl.tier) == "fp8") {
            check(g_opined == 0 && g_drops == 0, "G: *** no bound is applied from a tier this ladder cannot bound ***");
            check(ev_bad.size() == ev.size(),    "G: *** I6 red on every row (fp8 is not a rung) ***");
            check(saw_named_refusal,             "G: and the refusal names the tier and the bound");
        }
        if (std::string(decl.tier).empty()) {
            check(g_opined == 0 && g_drops == 0, "G: *** an undeclared tier decides NOTHING (it refuses, it does not guess) ***");
            check(ev_bad.size() == ev.size(),    "G: *** I6 red on every row (no tier declared) ***");
            check(saw_named_refusal,             "G: and the refusal names the bound that was withheld");
        }
    }

    // ------------------------------------------------------------------ case H
    // TABLE ADMISSION over the four non-default writers (--kv-layer-storage,
    // --kv-tier-formats hot=..., the NINFER_KV_*_BITS env layer, POST /reload_kv). A tool
    // that cannot express the factory's rk4v4 layers emits a COMPLETE spec; pasting it
    // replaces the factory table. The gate refuses by name and never merges.
    std::printf("\nH. TABLE ADMISSION (the four writers that are not the factory default)\n");
    {
        std::vector<KvLayerTableEntry> spec;
        std::vector<KvPerLayerEvidence> ev = per_layer_fixture();
        for (auto& e : ev) { e.err_tier = "nvfp4"; }
        for (int i = 0; i < 16; ++i) {
            KvLayerTableEntry e;
            e.layer = i;
            e.tier  = (i == 11) ? "fp8" : "nvfp4";   // the analyzer's shape: complete, no rk4v4
            spec.push_back(e);
        }
        KvPerLayerPolicyConfig cfg_admit;
        const auto v = kv_perlayer_table_violations(spec, ev, cfg_admit);
        std::printf("  a complete 16-entry spec: %zu violation(s)\n", v.size());
        for (const auto& s : v) { std::printf("    REFUSED: %s\n", s.c_str()); }
        check(!v.empty(), "H: *** a complete table from a non-default writer is REFUSED ***");
        bool saw_replace = false, saw_unbounded = false;
        for (const auto& s : v) {
            if (s.find("REPLACES the factory table") != std::string::npos) { saw_replace = true; }
            if (s.find("has no bound for") != std::string::npos) { saw_unbounded = true; }
        }
        check(saw_replace,   "H: and it names the consequence: the factory table would be replaced");
        check(saw_unbounded, "H: and the fp8 entry is refused for having no bound in this ladder");
        KvPerLayerPolicyConfig cfg_ack;
        cfg_ack.allow_full_table_replace = true;
        const auto v2 = kv_perlayer_table_violations(spec, ev, cfg_ack);
        bool only_unbounded = !v2.empty();
        for (const auto& s : v2) {
            if (s.find("REPLACES the factory table") != std::string::npos) { only_unbounded = false; }
        }
        check(only_unbounded,
              "H: acknowledging the replacement does NOT excuse a tier this ladder cannot bound");
        // THE FACTORY SHAPE AGAINST EVIDENCE MEASURED AT ITS OWN TIERS: the rk4v4 rung is
        // expressible, which is the whole point of err_tier.
        std::vector<KvLayerTableEntry> factory;
        std::vector<KvPerLayerEvidence> ev_f;
        for (int i = 0; i < 16; ++i) {
            const bool rk4v4_layer = (i == 0 || i == 1 || i == 3 || i == 4 || i == 6 || i == 7);
            KvLayerTableEntry e;
            e.layer = i;
            e.tier  = rk4v4_layer ? "rk4v4" : "nvfp4";
            factory.push_back(e);
            KvPerLayerEvidence x;
            x.layer           = i;
            x.energy_observed = true;
            x.energy_mean_l   = 0.1;
            x.energy_rounds   = 512;
            x.k_err_measured  = true;
            x.v_err_measured  = true;
            x.err_tier        = e.tier;
            x.k_err_nmse      = rk4v4_layer ? 1e-6 : 5e-4;
            x.v_err_nmse      = rk4v4_layer ? 1e-6 : 5e-4;
            ev_f.push_back(x);
        }
        const auto v3 = kv_perlayer_table_violations(factory, ev_f, cfg_ack);
        std::printf("  the factory-shaped rk4v4 window WITH rk4v4-tier evidence: %zu violation(s)\n", v3.size());
        for (const auto& s : v3) { std::printf("    %s\n", s.c_str()); }
        check(v3.empty(), "H: *** the rk4v4 rung IS expressible: with an rk4v4 measurement the rk4v4 entries pass ***");
        const auto v4 = kv_perlayer_table_violations(factory, ev, cfg_ack);
        check(!v4.empty(), "H: *** and the same rk4v4 entries against nvfp4-only evidence are refused ***");
    }

    // ------------------------------------------------------------------ case I
    // SELWIRE-LAND: THE THREE HOLES SELWIRE2-X NAMED, each with its control.
    //   I-a `from_factory_table` was a free exemption: the gate `continue`d on the bool
    //       alone, so a complete 16-row table of 'bf16' -- a tier this ladder cannot bound,
    //       so every row would otherwise have been refused -- passed with 0 violations.
    //   I-b `entries.size() == evidence.size()` was a length coincidence between two
    //       unrelated vectors: false POSITIVE on a table that reproduces the factory table
    //       (accused of replacing itself) and false NEGATIVE on a genuine complete
    //       replacement whose evidence vector was a different length (waved through).
    //   I-c "measured, but no tier declared" produced with_tier=0 for every layer while
    //       `per_layer_information` stayed true -- a zero indistinguishable from "never
    //       measured", i.e. a silent red for a behaviour-breaking change.
    std::printf("\nI. THE THREE HOLES (from_factory_table / replacement predicate / declaration)\n");
    {
        const bool e8L[16] = {true, true, false, true, true, false, true, true,
                              false, false, false, false, false, false, false, false};

        // ---- I-a: the exemption is checked against the factory table --------------
        std::printf("  I-a. `from_factory_table` is a CLAIM, not a key\n");
        {
            // (1) THE HOLE, verbatim: 16 rows, all flagged factory, all 'bf16'.
            std::vector<KvLayerTableEntry> hole;
            for (int i = 0; i < 16; ++i) {
                KvLayerTableEntry e;
                e.layer = i;
                e.tier  = "bf16";                 // no bound in this ladder at all
                e.from_factory_table = true;      // ...but the bool says "trust me"
                hole.push_back(e);
            }
            const auto vh = kv_perlayer_table_violations(hole, {}, cfg);
            std::printf("    16 rows all 'bf16', all flagged factory: %zu violation(s)\n", vh.size());
            for (const auto& s : vh) { std::printf("      REFUSED: %s\n", s.c_str()); }
            check(vh.size() == hole.size(),
                  "I-a: *** a wholly UNBOUNDED table cannot be admitted by one bool (16/16 refused) ***");
            check(any_contains(vh, "the exemption is not the factory's to grant"),
                  "I-a: and each row says WHY: it is not the factory table's tier");

            // (2) control: the SAME bool, with the factory's OWN tiers, is honest.
            std::vector<KvLayerTableEntry> honest;
            for (int i = 0; i < 16; ++i) {
                KvLayerTableEntry e;
                e.layer = i;
                e.tier  = e8L[i] ? "rk4v4" : "nvfp4";
                e.from_factory_table = true;
                honest.push_back(e);
            }
            const auto vo = kv_perlayer_table_violations(honest, {}, cfg);
            check(vo.empty(), "I-a: control -- the real factory table still passes the gate (the check is not a ban)");

            // (3) the claim on a row the factory table does not own, and a wrong tier.
            std::vector<KvLayerTableEntry> outside;
            { KvLayerTableEntry e; e.layer = 16; e.tier = "rk4v4"; e.from_factory_table = true; outside.push_back(e); }
            { KvLayerTableEntry e; e.layer = 5;  e.tier = "rk4v4"; e.from_factory_table = true; outside.push_back(e); }
            const auto vout = kv_perlayer_table_violations(outside, {}, cfg);
            for (const auto& s : vout) { std::printf("      REFUSED: %s\n", s.c_str()); }
            check(any_contains(vout, "has no layer 16"),
                  "I-a: a factory claim on a row the factory table does not own is refused");
            check(any_contains(vout, "writes 'nvfp4' on this layer"),
                  "I-a: and a factory claim with the WRONG tier names the factory's tier (layer 5 is nvfp4)");
        }

        // ---- I-b: the replacement predicate, both directions ---------------------
        std::printf("  I-b. \"does this table REPLACE the factory table?\"\n");
        std::vector<KvPerLayerEvidence> ev_decl;   // 16 rows, each declaring its table's tier
        for (int i = 0; i < 16; ++i) {
            KvPerLayerEvidence x;
            x.layer           = i;
            x.energy_observed = true;
            x.energy_mean_l   = 0.1;
            x.energy_rounds   = 512;
            x.k_err_measured  = true;
            x.v_err_measured  = true;
            x.err_tier        = e8L[i] ? "rk4v4" : "nvfp4";
            x.k_err_nmse      = e8L[i] ? 1e-6 : 5e-4;
            x.v_err_nmse      = e8L[i] ? 1e-6 : 5e-4;
            ev_decl.push_back(x);
        }
        std::vector<KvLayerTableEntry> factory_plain;   // the factory table, re-derived as plain rows
        for (int i = 0; i < 16; ++i) {
            KvLayerTableEntry e;
            e.layer = i;
            e.tier  = e8L[i] ? "rk4v4" : "nvfp4";
            factory_plain.push_back(e);
        }
        KvPerLayerPolicyConfig cfg_16;
        cfg_16.table_model_layers = 16;                 // the caller says how wide the model is
        const auto vb1 = kv_perlayer_table_violations(factory_plain, ev_decl, cfg_16);
        for (const auto& s : vb1) { std::printf("      REFUSED: %s\n", s.c_str()); }
        check(vb1.empty(),
              "I-b: *** FALSE POSITIVE REMOVED: a table reproducing the factory table is not "
              "'replacing' it (16 rows, 16 evidence rows, no waiver) ***");

        std::vector<KvLayerTableEntry> all_nvfp4;       // a complete replacement, no rk4v4 anywhere
        for (int i = 0; i < 16; ++i) {
            KvLayerTableEntry e; e.layer = i; e.tier = "nvfp4"; all_nvfp4.push_back(e);
        }
        std::vector<KvPerLayerEvidence> ev_nvfp4;       // every row measured AT nvfp4: no per-row red
        for (int i = 0; i < 16; ++i) {
            KvPerLayerEvidence x;
            x.layer = i; x.energy_observed = true; x.energy_mean_l = 0.1; x.energy_rounds = 512;
            x.k_err_measured = true; x.v_err_measured = true; x.err_tier = "nvfp4";
            x.k_err_nmse = 5e-4; x.v_err_nmse = 5e-4;
            ev_nvfp4.push_back(x);
        }
        const auto vb2 = kv_perlayer_table_violations(all_nvfp4, ev_nvfp4, cfg_16);
        for (const auto& s : vb2) { std::printf("      REFUSED: %s\n", s.c_str()); }
        check(vb2.size() == 1 && any_contains(vb2, "REPLACES"),
              "I-b: the gate is NOT weakened: a complete all-nvfp4 replacement is refused, and "
              "only for that reason");

        // THE FALSE NEGATIVE, killed: the same complete replacement, but the caller handed in
        // only the 4 layers it measured. Under `entries.size() == evidence.size()` this table
        // (16 rows vs 4 evidence rows) was accepted SILENTLY -- the gate's whole subject gone
        // because two unrelated numbers differed.
        std::vector<KvPerLayerEvidence> ev_short(ev_nvfp4.begin(), ev_nvfp4.begin() + 4);
        const auto vb3 = kv_perlayer_table_violations(all_nvfp4, ev_short, cfg_16);
        check(any_contains(vb3, "REPLACES the factory table"),
              "I-b: *** FALSE NEGATIVE KILLED: the 16-row replacement is still refused when the "
              "evidence vector is 4 rows long (the old predicate was silent here) ***");

        // Undeclared width: refuse BY NAME rather than guess.
        const auto vb4 = kv_perlayer_table_violations(all_nvfp4, ev_nvfp4, cfg);
        for (const auto& s : vb4) { std::printf("      REFUSED: %s\n", s.c_str()); }
        check(any_contains(vb4, "table_model_layers is 0"),
              "I-b: *** an undeclared model width is a REFUSAL BY NAME, not a silent pass ***");

        // ...and a PARTIAL table stays writer (2)'s ordinary slot-by-slot use: not this gate's subject.
        std::vector<KvLayerTableEntry> partial(all_nvfp4.begin(), all_nvfp4.begin() + 4);
        const auto vb5 = kv_perlayer_table_violations(partial, ev_nvfp4, cfg_16);
        check(!any_contains(vb5, "REPLACES"),
              "I-b: control -- a 4-row table is slot-by-slot replacement and is not called a full replacement");
    }

    // ---- I-c: "measured, but no tier declared" is no longer silent ---------------
    std::printf("  I-c. the caller that does not declare its tier\n");
    {
        auto undeclared = per_layer_fixture();
        for (auto& e : undeclared) { e.err_tier.clear(); }          // the pre-gate world
        KvPerLayerPolicyConfig cfg_c = cfg;
        cfg_c.drop_enabled = true;
        const auto c = kv_perlayer_decide(undeclared, cfg_c);
        int c_opined = 0, c_drops = 0;
        for (const auto& v : c.rows) { if (!v.tier.empty()) ++c_opined; if (v.drop_admissible) ++c_drops; }
        std::printf("    %s\n", c.report.c_str());
        check(c_opined == 0 && c_drops == 0,
              "I-c: the behaviour change is real: with no tier declared NOTHING is decided (16/16 with_tier=0)");
        check(c.rows_refused_no_declared_tier == c.rows.size(),
              "I-c: *** and the decision SAYS SO: rows_refused_no_declared_tier == 16 ***");
        check(c.rows_refused_unbounded_tier == 0,
              "I-c: and it does not blame the ladder for a caller-side omission");
        const auto cv = kv_perlayer_declaration_violations(c);
        std::printf("    declaration violations: %zu\n", cv.size());
        for (const auto& s : cv) { std::printf("      RED: %s\n", s.c_str()); }
        check(!cv.empty(), "I-c: *** THE DECLARATION GATE GOES RED -- the caller can gate its rc on this ***");
        check(c.report.find("REFUSED_NO_DECLARED_TIER=16") != std::string::npos,
              "I-c: and the fact is on the one-line report, so a printf-only caller sees it too");
        check(kv_perlayer_invariant_violations(c).empty(),
              "I-c: NOTE -- the VERDICT-side invariants stay green (this decision abstained "
              "consistently); that is exactly why the declaration gate is a separate red");

        // The contrast that makes the counter meaningful: SAME numbers, tier declared.
        auto declared = per_layer_fixture();
        const auto c2 = kv_perlayer_decide(declared, cfg_c);
        int c2_opined = 0;
        for (const auto& v : c2.rows) { if (!v.tier.empty()) ++c2_opined; }
        check(c2.rows_refused_no_declared_tier == 0 && kv_perlayer_declaration_violations(c2).empty(),
              "I-c: control -- declare err_tier and both the counter and the red go away");
        check(c2_opined > 0, "I-c: control -- ...and verdicts are issued again");

        // A tier that is real but not a rung: the OTHER counter, and it must not be conflated.
        auto unbounded = per_layer_fixture();
        for (auto& e : unbounded) { e.err_tier = "fp8"; }
        const auto c3 = kv_perlayer_decide(unbounded, cfg_c);
        check(c3.rows_refused_unbounded_tier == c3.rows.size() &&
              c3.rows_refused_no_declared_tier == 0,
              "I-c: a real-but-unbounded tier (fp8) lands in its OWN counter, not in the omission one");

        // A fresh box: BOTH counters zero. The counters name a caller-side failure, not
        // "anything that produced no verdict".
        const auto c4 = kv_perlayer_decide(none, cfg_c);
        check(c4.rows_refused_no_declared_tier == 0 && c4.rows_refused_unbounded_tier == 0 &&
              kv_perlayer_declaration_violations(c4).empty(),
              "I-c: control -- a never-measured box trips neither counter");
    }

    // ------------------------------------------------------------------ case J
    // SELWIRE-HARDEN: THE CLAUSES THE DELIVERED SUITE COULD NOT SEE BEING CHANGED. SELWIRE-X2's
    // mutation matrix cut, inverted or loosened eleven load-bearing clauses and FIVE of them came
    // back with 0 FAIL / rc=0: the suite could not tell whether they existed. Each sub-case below
    // names the mutation it isolates and what was measured BEFORE it existed.
    std::printf("\nJ. THE UNPINNED CLAUSES (width / drop bound / int8 rung / factory skip / dedup / degenerate)\n");
    const bool k_rk4v4[16] = {true, true, false, true, true, false, true, true,
                           false, false, false, false, false, false, false, false};
    KvPerLayerPolicyConfig cfg16;
    cfg16.table_model_layers = 16;

    // ---- J-a: the width cross-check (H1) --------------------------------------
    {
        std::printf("  J-a. table_model_layers is CROSS-CHECKED against the factory width\n");
        std::vector<KvLayerTableEntry> repl;
        for (int i = 0; i < 16; ++i) {
            KvLayerTableEntry e; e.layer = i; e.tier = "nvfp4"; repl.push_back(e);
        }
        std::vector<KvPerLayerEvidence> ev20;      // 20 evidence rows: a DIFFERENT LENGTH, on purpose
        for (int i = 0; i < 20; ++i) {
            KvPerLayerEvidence x;
            x.layer = i % 16;
            x.energy_observed = true; x.energy_mean_l = 0.1; x.energy_rounds = 512;
            x.k_err_measured = true; x.v_err_measured = true; x.err_tier = "nvfp4";
            x.k_err_nmse = 5e-4; x.v_err_nmse = 5e-4;
            ev20.push_back(x);
        }
        struct WidthCell { std::int32_t width; const char* needle; const char* meaning; };
        const WidthCell cells[] = {
            {16, "REPLACES the factory table",      "model=16: a complete replacement IS refused, for that one reason"},
            {17, "is WIDER than the factory table", "*** model=17 (X2 L7): 0 violations before H1 -- the gate was OFF ***"},
            {32, "is WIDER than the factory table", "*** model=32 (X2 L8): 0 violations before H1 ***"},
            {0,  "table_model_layers is 0",         "model=0: still a refusal BY NAME, not a silent pass"},
            {8,  "REPLACES the factory table",      "model=8 (under-declared): a width below the write buys no pass"},
        };
        for (const WidthCell& c : cells) {
            KvPerLayerPolicyConfig cfgw; cfgw.table_model_layers = c.width;
            const auto v = kv_perlayer_table_violations(repl, ev20, cfgw);
            std::printf("    model=%-3d violations=%zu  %s\n", c.width, v.size(), c.meaning);
            check(any_contains(v, c.needle), c.meaning);
            if (c.width > 16) {
                check(!any_contains(v, "has no bound for") && !any_contains(v, "unevidenced"),
                      "J-a: the red is the WIDTH alone -- every row of this table is row-compliant");
            }
        }
        check(kv_perlayer_table_violations(repl, ev20, cfg16).size() == 1,
              "J-a: *** model=16 with 20 evidence rows -> exactly ONE violation: the verdict is the "
              "table's CONTENT, not a length coincidence (the removed predicate was silent here) ***");
        // THE CORRECT SHAPES MUST STAY GREEN. BY3's 0 is CORRECT, not a missing check.
        std::vector<KvLayerTableEntry> fac_marked;
        for (int i = 0; i < 16; ++i) {
            KvLayerTableEntry e; e.layer = i; e.tier = k_rk4v4[i] ? "rk4v4" : "nvfp4";
            e.from_factory_table = true; fac_marked.push_back(e);
        }
        check(kv_perlayer_table_violations(fac_marked, {}, cfg).empty(),
              "J-a: *** the factory table itself (undeclared width, NO evidence) is NOT refused: "
              "BY3's 0 is the CORRECT shape and H1 must not turn it red ***");
        check(kv_perlayer_table_violations(fac_marked, {}, cfg16).empty(),
              "J-a: ...and not at model=16 either (it is factory-equivalent by construction)");
        // the single-row carve-out, observable only at width 1
        std::vector<KvLayerTableEntry> one_row;
        { KvLayerTableEntry e; e.layer = 0; e.tier = "int8"; one_row.push_back(e); }
        std::vector<KvPerLayerEvidence> one_ev;
        { KvPerLayerEvidence x; x.layer = 0; x.energy_observed = true; x.energy_mean_l = 0.1;
          x.energy_rounds = 512; x.k_err_measured = true; x.v_err_measured = true;
          x.err_tier = "int8"; x.k_err_nmse = 5e-4; x.v_err_nmse = 5e-4; one_ev.push_back(x); }
        KvPerLayerPolicyConfig cfg1; cfg1.table_model_layers = 1;
        const auto v1r = kv_perlayer_table_violations(one_row, one_ev, cfg1);
        check(!any_contains(v1r, "REPLACES the factory table"),
              "J-a: *** a 1-row table is writer (2)'s slot-by-slot use even at width 1: deleting "
              "`entries.size() > 1` makes it a 'wholesale replacement' ***");

        // ---- M10b: THE `entries.size() > 1` CARVE-OUT, WITH MORE THAN ONE DETECTOR ------------
        // HONEST SCOPE FIRST: this carve-out is observable ONLY where a single row could name every
        // layer, i.e. `table_model_layers <= 1` AND `entries.size() == 1` -- at any wider declaration
        // `names_every_layer` is unreachable for one row and the clause cannot be seen at all. So the
        // added coverage is three INDEPENDENT detectors inside that one domain, not a wider domain:
        // two go red the moment the carve-out is cut, the third pins the boundary from the far side.
        {
            // (b) THE OTHER TIER, WITH ITS OWN EVIDENCE: the claim is exactly "zero violations".
            std::vector<KvLayerTableEntry> one_nv;
            { KvLayerTableEntry e; e.layer = 0; e.tier = "nvfp4"; one_nv.push_back(e); }
            std::vector<KvPerLayerEvidence> one_ev_nv;
            { KvPerLayerEvidence x; x.layer = 0; x.energy_observed = true; x.energy_mean_l = 0.1;
              x.energy_rounds = 512; x.k_err_measured = true; x.v_err_measured = true;
              x.err_tier = "nvfp4"; x.k_err_nmse = 5e-4; x.v_err_nmse = 5e-4;
              one_ev_nv.push_back(x); }
            const auto vb_nv = kv_perlayer_table_violations(one_nv, one_ev_nv, cfg1);
            std::printf("    M10b/b 1 row, tier=nvfp4, evidenced: %zu violation(s)\n", vb_nv.size());
            for (const auto& s : vb_nv) { std::printf("      %s\n", s.c_str()); }
            check(vb_nv.empty(),
                  "M10b/b: *** width=1, ONE row, its own nvfp4 evidence: ZERO violations -- cutting "
                  "`entries.size() > 1` turns this into a 'wholesale replacement' ***");

            // (c) ONE ROW AND NO EVIDENCE AT ALL: exactly one violation, and it belongs to the ROW.
            std::vector<KvLayerTableEntry> one_i8;
            { KvLayerTableEntry e; e.layer = 0; e.tier = "int8"; one_i8.push_back(e); }
            const auto vb_i8 = kv_perlayer_table_violations(one_i8, {}, cfg1);
            std::printf("    M10b/c 1 row, tier=int8, NO evidence: %zu violation(s)\n", vb_i8.size());
            for (const auto& s : vb_i8) { std::printf("      %s\n", s.c_str()); }
            check(vb_i8.size() == 1 && any_contains(vb_i8, "unevidenced") &&
                  !any_contains(vb_i8, "REPLACES the factory table"),
                  "M10b/c: *** the SAME one-row table with no evidence is refused ONCE, for being "
                  "unevidenced -- and never for replacing the factory table ***");

            // (d) THE BOUNDARY FROM THE FAR SIDE. The carve-out is exactly `> 1`, so TWO rows at
            //     width 1 ARE this gate's subject; this cell is the one that goes red if it is widened.
            std::vector<KvLayerTableEntry> two_rows;
            { KvLayerTableEntry e; e.layer = 0; e.tier = "int8"; two_rows.push_back(e); }
            { KvLayerTableEntry e; e.layer = 0; e.tier = "int8"; two_rows.push_back(e); }
            const auto vb_two = kv_perlayer_table_violations(two_rows, one_ev, cfg1);
            std::printf("    M10b/d 2 rows (duplicate layer 0), width=1: %zu violation(s)\n",
                        vb_two.size());
            check(any_contains(vb_two, "REPLACES the factory table"),
                  "M10b/d: *** control -- the carve-out is exactly `size() > 1`: a 2-row table at "
                  "width 1 IS called a replacement (WIDENING the carve-out turns this green) ***");
        }
    }

    // ---- J-b: the drop bound (1e-4), which no delivered test ever wrote --------
    {
        std::printf("  J-b. the DROP BOUND is pinned -- cfg.drop_nmse_max was never touched by the delivered suite\n");
        const double errs[5] = {1e-6, 1e-4, 5e-4, 1e-3, 2e-3};
        std::vector<KvPerLayerEvidence> ev;
        for (int i = 0; i < 5; ++i) {
            KvPerLayerEvidence e;
            e.layer = i; e.energy_observed = true; e.energy_mean_l = 0.1 + 0.01 * i;
            e.energy_rounds = 512; e.k_err_measured = true; e.v_err_measured = true;
            e.err_tier = "nvfp4"; e.k_err_nmse = errs[i]; e.v_err_nmse = errs[i];
            ev.push_back(e);
        }
        const auto d = kv_perlayer_decide(ev, cfg);      // cfg.drop_enabled == true
        std::printf("    %s\n", d.report.c_str());
        const bool want[5] = {true, true, false, false, false};
        for (int i = 0; i < 5; ++i) {
            char what[160];
            std::snprintf(what, sizeof(what),
                          "J-b: max(K,V) err=%-8g -> drop %s (bound 1e-4, INCLUSIVE; 1e-4 -> 1e-1 flips this)",
                          errs[i], want[i] ? "ADMITTED" : "REFUSED");
            check(d.rows[i].drop_admissible == want[i], what);
        }
        check(any_contains(d.rows[2].reasons, "above the drop bound"),
              "J-b: *** err=5e-4 is refused BY THE BOUND and the reason says so (the delivered suite "
              "had no case in (1e-4, 1e-1], so loosening the bound was invisible) ***");
        check(any_contains(d.rows[3].reasons, "above the drop bound"),
              "J-b: *** err=1e-3 (exactly the nvfp4 bound) is refused by the DROP bound, not the tier gate ***");
        check(any_contains(d.rows[4].reasons, "measured at tier 'nvfp4'"),
              "J-b: control -- err=2e-3 is refused by the TIER gate (its rung is int8), not by the bound");
    }

    // ---- J-c: the int8 rung (rank 3) -- the word "int8" never appears in the delivered suite
    {
        std::printf("  J-c. the INT8 rung exists (rank 3)\n");
        check(kv_perlayer_tier_rank("rk4v4") == 1 && kv_perlayer_tier_rank("nvfp4") == 2 &&
              kv_perlayer_tier_rank("int8") == 3,
              "J-c: *** the ladder's three rungs keep their ranks (rk4v4=1, nvfp4=2, int8=3): flipping "
              "int8 to 0 emptied every int8 row and the delivered suite never noticed ***");
        check(kv_perlayer_tier_rank("bf16") == 0 && kv_perlayer_tier_rank("fp8") == 0 &&
              kv_perlayer_tier_rank("iso4e") == 0 && kv_perlayer_tier_rank("") == 0,
              "J-c: and the real tiers this ladder does not bound stay 0");
        std::vector<KvPerLayerEvidence> ev;
        for (int i = 0; i < 4; ++i) {
            KvPerLayerEvidence e;
            e.layer = i; e.energy_observed = true; e.energy_mean_l = 0.1 + 0.01 * i;
            e.energy_rounds = 512; e.k_err_measured = true; e.v_err_measured = true;
            e.err_tier = "int8"; e.k_err_nmse = 2e-3 + 1e-5 * i; e.v_err_nmse = 2e-3 + 1e-5 * i;
            ev.push_back(e);
        }
        KvPerLayerPolicyConfig c_off;      // drop axis off: this is about the TIER verdict
        const auto d = kv_perlayer_decide(ev, c_off);
        int int8_rows = 0;
        for (const auto& v : d.rows) { if (v.tier == "int8") { ++int8_rows; } }
        check(int8_rows == 4,
              "J-c: *** an int8-tier measurement above every nvfp4 bound lands on the int8 rung ***");
        check(kv_perlayer_evidence_violations(ev).empty(),
              "J-c: *** and I6 is CLEAN for int8 (it IS a rung): rank 0 would red all 4 rows ***");
        check(!any_contains(d.rows[0].reasons, "no bound was applied"),
              "J-c: control -- the int8 rung is not reached through the loud refusal");
        std::vector<KvLayerTableEntry> spec;
        for (int i = 0; i < 16; ++i) { KvLayerTableEntry e; e.layer = i; e.tier = "int8"; spec.push_back(e); }
        std::vector<KvPerLayerEvidence> ev16;
        for (int i = 0; i < 16; ++i) {
            KvPerLayerEvidence x;
            x.layer = i; x.energy_observed = true; x.energy_mean_l = 0.1; x.energy_rounds = 512;
            x.k_err_measured = true; x.v_err_measured = true; x.err_tier = "int8";
            x.k_err_nmse = 5e-4; x.v_err_nmse = 5e-4;
            ev16.push_back(x);
        }
        const auto v = kv_perlayer_table_violations(spec, ev16, cfg16);
        check(v.size() == 1 && any_contains(v, "REPLACES the factory table"),
              "J-c: a complete int8 table is refused for REPLACING the factory table, and for nothing else");
        check(!any_contains(v, "has no bound for"),
              "J-c: *** ...and NOT as an unbounded tier (int8 IS a rung): rank 3 -> 0 turns this "
              "into 16 'has no bound for' + 1 replacement ***");
    }

    // ---- J-d: the factory skip (U-a's `continue`), which the suite exercised only by accident
    {
        std::printf("  J-d. the FACTORY SKIP is pinned by name\n");
        std::vector<KvLayerTableEntry> fac;
        for (int i = 0; i < 16; ++i) {
            KvLayerTableEntry e; e.layer = i; e.tier = k_rk4v4[i] ? "rk4v4" : "nvfp4";
            e.from_factory_table = true; fac.push_back(e);
        }
        const auto v1 = kv_perlayer_table_violations(fac, {}, cfg);
        std::printf("    16 factory rows, NO evidence: %zu violation(s)\n", v1.size());
        check(v1.empty(),
              "J-d: *** the factory rows are exempt WITHOUT evidence (deleting the skip turns this "
              "into 16 'unevidenced' + 1 width refusal) ***");
        check(!any_contains(v1, "unevidenced") && !any_contains(v1, "REPLACES"),
              "J-d: *** ...and neither the row check nor the replacement verdict speaks about them ***");
        // the skip must also keep the COUNT honest: a factory row is not `from_other_writers`.
        std::vector<KvLayerTableEntry> mixed = fac;
        { KvLayerTableEntry e; e.layer = 0; e.tier = "nvfp4"; mixed.push_back(e); }
        std::vector<KvPerLayerEvidence> ev0;
        { KvPerLayerEvidence x; x.layer = 0; x.energy_observed = true; x.energy_mean_l = 0.1;
          x.energy_rounds = 512; x.k_err_measured = true; x.v_err_measured = true;
          x.err_tier = "nvfp4"; x.k_err_nmse = 5e-4; x.v_err_nmse = 5e-4; ev0.push_back(x); }
        const auto v2 = kv_perlayer_table_violations(mixed, ev0, cfg16);
        std::printf("    factory rows + 1 honest writer row: %zu violation(s)\n", v2.size());
        check(v2.size() == 1 && any_contains(v2, "REPLACES the factory table"),
              "J-d: *** a factory row is not another writer: the merged table is still judged, and "
              "only the new row's write triggers the verdict ***");
        check(!any_contains(v2, "unevidenced"),
              "J-d: *** ...with no factory row accused: deleting the skip contradicts all 16 ***");
    }

    // ---- J-e: the DEDUP in the replacement count (X2's LA7, the post-C survivor)
    {
        std::printf("  J-e. the DEDUP in 'names every layer' is pinned (X2 LA7: the one post-C survivor)\n");
        std::vector<KvPerLayerEvidence> ev16;
        for (int i = 0; i < 16; ++i) {
            KvPerLayerEvidence x;
            x.layer = i; x.energy_observed = true; x.energy_mean_l = 0.1; x.energy_rounds = 512;
            x.k_err_measured = true; x.v_err_measured = true; x.err_tier = "nvfp4";
            x.k_err_nmse = 5e-4; x.v_err_nmse = 5e-4;
            ev16.push_back(x);
        }
        // (i) a GENUINE replacement PADDED with duplicate rows: the count must not be padded with it.
        std::vector<KvLayerTableEntry> padded;
        for (int i = 0; i < 16; ++i) { KvLayerTableEntry e; e.layer = i; e.tier = "nvfp4"; padded.push_back(e); }
        for (int i = 0; i < 3; ++i) { KvLayerTableEntry e; e.layer = i; e.tier = "nvfp4"; padded.push_back(e); }
        const auto v1 = kv_perlayer_table_violations(padded, ev16, cfg16);
        std::printf("    16 rows + 3 duplicate rows: %zu violation(s)\n", v1.size());
        check(v1.size() == 1 && any_contains(v1, "REPLACES the factory table"),
              "J-e: *** DUPLICATE PADDING does not disarm the verdict: without the dedup the count is "
              "19 != 16 and this complete replacement passes with ZERO violations ***");
        // (ii) the merged table: the factory rows' layer indices reused by a second writer's full write.
        std::vector<KvLayerTableEntry> merged;
        for (int i = 0; i < 16; ++i) {
            KvLayerTableEntry e; e.layer = i; e.tier = k_rk4v4[i] ? "rk4v4" : "nvfp4";
            e.from_factory_table = true; merged.push_back(e);
        }
        for (int i = 0; i < 16; ++i) { KvLayerTableEntry e; e.layer = i; e.tier = "int8"; merged.push_back(e); }
        std::vector<KvPerLayerEvidence> evi;
        for (int i = 0; i < 16; ++i) {
            KvPerLayerEvidence x;
            x.layer = i; x.energy_observed = true; x.energy_mean_l = 0.1; x.energy_rounds = 512;
            x.k_err_measured = true; x.v_err_measured = true; x.err_tier = "int8";
            x.k_err_nmse = 5e-4; x.v_err_nmse = 5e-4;
            evi.push_back(x);
        }
        const auto v2 = kv_perlayer_table_violations(merged, evi, cfg16);
        std::printf("    factory rows + a complete second write: %zu violation(s)\n", v2.size());
        check(v2.size() == 1 && any_contains(v2, "REPLACES the factory table"),
              "J-e: *** a MERGED table is still a replacement: without the dedup the count is 32 != 16 "
              "and the second writer's complete write passes (X2's BY3c) ***");
        // (iii) the control the dedup exists for: 16 ROWS are not 16 LAYERS.
        std::vector<KvLayerTableEntry> fifteen;
        for (int i = 0; i < 15; ++i) { KvLayerTableEntry e; e.layer = i; e.tier = "int8"; fifteen.push_back(e); }
        { KvLayerTableEntry e; e.layer = 0; e.tier = "int8"; fifteen.push_back(e); }
        std::vector<KvPerLayerEvidence> ev15(evi.begin(), evi.begin() + 15);
        const auto v3 = kv_perlayer_table_violations(fifteen, ev15, cfg16);
        std::printf("    15 distinct layers in 16 rows: %zu violation(s)\n", v3.size());
        check(v3.empty(),
              "J-e: *** control -- a duplicated row is not a 16th layer (without the dedup, 16 rows == "
              "16 layers and an innocent 15-layer write is accused of replacing the table) ***");
    }

    // ---- J-f: H3, the MUST gate as one callable -------------------------------
    {
        std::printf("  J-f. THE MUST-RUN-THIS ENTRY POINT (both gates, one call)\n");
        const auto jf_a = kv_perlayer_decide(per_layer_fixture(), cfg);
        check(kv_perlayer_all_violations(jf_a).empty(),
              "J-f: a clean decision is empty through the COMBINED gate (the convention is not a ban)");
        check(kv_perlayer_all_violations(jf_a).size() ==
                kv_perlayer_invariant_violations(jf_a).size() +
                kv_perlayer_declaration_violations(jf_a).size(),
              "J-f: the combined gate is the UNION of the two, not one of them");
        auto undecl = per_layer_fixture();
        for (auto& e : undecl) { e.err_tier.clear(); }
        const auto jf_c = kv_perlayer_decide(undecl, cfg);
        check(kv_perlayer_invariant_violations(jf_c).empty(),
              "J-f: the verdict-side gate stays GREEN for a caller-side omission (case G's reviewed separation)");
        check(kv_perlayer_all_violations(jf_c).size() ==
                kv_perlayer_invariant_violations(jf_c).size() +
                kv_perlayer_declaration_violations(jf_c).size() &&
              !kv_perlayer_all_violations(jf_c).empty(),
              "J-f: *** THE MUST GATE GOES RED where the half-gate alone stays green (dropping the "
              "declaration half makes this empty) ***");
        // A ROBUST FORM ON PURPOSE: `[0]` on an EMPTY vector is UB and it aborted (rc=134) under the
        // mutant that drops the declaration half, which also swallowed the FAIL lines still sitting
        // in the stdio buffer. The criterion set is {rc, checks}; a crash is not a check.
        bool saw_decl_text = false;
        for (const auto& s : kv_perlayer_all_violations(jf_c)) {
            if (s.find("MEASURED a K/V NMSE but did not declare") != std::string::npos) {
                saw_decl_text = true;
            }
        }
        check(saw_decl_text,
              "J-f: and the reason is the original declaration text, preserved verbatim");
    }

    // ---- J-g: H4, the degenerate hand-in ---------------------------------------
    {
        std::printf("  J-g. the DEGENERATE hand-in is distinguishable from 'never measured'\n");
        const auto deg_declared = global_scalar_fixture();
        const auto g1 = kv_perlayer_decide(deg_declared, cfg);
        check(g1.degenerate_input && !g1.per_layer_information,
              "J-g: the policy notices (degenerate_input=true, no per-layer information)");
        check(g1.report.find("DEGENERATE INPUT") != std::string::npos,
              "J-g: ...and says it on the one-line report a printf-only caller reads");
        std::printf("    counters on the degenerate rows: nodesc=%zu unbounded=%zu\n",
                    g1.rows_refused_no_declared_tier, g1.rows_refused_unbounded_tier);
        check(g1.rows_refused_no_declared_tier == 0 && g1.rows_refused_unbounded_tier == 0,
              "J-g: ...while BOTH U-c counters are 0: the per-row counting cannot see this case at all");
        auto deg_undeclared = deg_declared;
        for (auto& e : deg_undeclared) { e.err_tier.clear(); }
        const auto g2 = kv_perlayer_decide(deg_undeclared, cfg);
        const auto g2v = kv_perlayer_declaration_violations(g2);
        check(g2v.size() == 1 && any_contains(g2v, "ONE VALUE SMEARED OVER ALL"),
              "J-g: *** the CALLER-SIDE gate now goes RED for a degenerate hand-in -- before H4 it was "
              "EMPTY, and 'never measured' was the only other state with an empty gate ***");
        check(kv_perlayer_declaration_violations(g1).size() == 1,
              "J-g: *** and it reds the DECLARED degenerate case too: the failure is the collapse, not "
              "the missing tier ***");
        check(kv_perlayer_invariant_violations(g2).empty(),
              "J-g: the verdict-side gate is NOT carrying H4 -- so case G's and case I-c's reviewed "
              "assertions keep their exact meaning");
        std::vector<KvPerLayerEvidence> fresh;
        for (int i = 0; i < 16; ++i) { KvPerLayerEvidence e; e.layer = i; fresh.push_back(e); }
        const auto g3 = kv_perlayer_decide(fresh, cfg);
        check(!g3.degenerate_input && kv_perlayer_declaration_violations(g3).empty(),
              "J-g: *** control -- a never-measured box keeps an EMPTY gate: the two zero-verdict "
              "states are now distinguishable by this gate alone ***");
        check(kv_perlayer_all_violations(g2).size() == 1 + kv_perlayer_invariant_violations(g2).size(),
              "J-g: ...and the degenerate red is reachable through the MUST gate as well");
    }

    // ---- J-h: the load-bearing drop clause, pinned BY ITS OWN REASON ----------
    // HONEST STATEMENT FIRST: this clause's PREDICATE is identical to the second gate's
    // (`quant_error_measured()`), and the second gate refuses with the empty tier when the clause
    // is gone -- so DELETING THE CLAUSE CHANGES NO DROP VERDICT (case E's `e_drops == 0` stays
    // true, which is exactly why it survived X2's matrix). What the clause owns is the REASON: it
    // is the only place that says "a cost signal cannot justify a drop" instead of "no bound was
    // applied". That string is what makes the deletion visible at all; claiming more would be a
    // false claim.
    {
        std::printf("  J-h. the load-bearing drop clause, pinned by its own named reason\n");
        auto energy_only = per_layer_fixture();
        for (auto& e : energy_only) { e.k_err_measured = false; e.v_err_measured = false; }
        const auto d = kv_perlayer_decide(energy_only, cfg);   // drop_enabled, energy present
        int cost_reason = 0, bound_reason = 0;
        for (const auto& v : d.rows) {
            if (any_contains(v.reasons, "a cost signal cannot justify a drop")) { ++cost_reason; }
            if (any_contains(v.reasons, "no bound was applied")) { ++bound_reason; }
        }
        std::printf("    rows naming the cost-signal clause=%d, rows naming the withheld bound=%d\n",
                    cost_reason, bound_reason);
        check(cost_reason == 16,
              "J-h: *** all 16 refusals name THE COST-SIGNAL CLAUSE: cutting it replaces this string "
              "with the tier gate's and nothing else in the suite notices ***");
        check(bound_reason == 0,
              "J-h: control -- the tier gate did NOT have to speak: the load-bearing clause refused "
              "first (removing it pulls the tier gate in, which is what rescued case E)");
    }

    // ---- J-i: U5-a. `allow_full_table_replace` WAIVES Q2 -- and with it the width it needs ----
    // The cell HARDEN named as untested: allow_full_table_replace == true TOGETHER WITH
    // table_model_layers > 16. Reading the gate: `cfg.table_model_layers` has exactly ONE consumer --
    // Q2, the "does this table REPLACE the factory table?" verdict -- and the `<= 0` and `> 16`
    // clauses are themselves Q2 refusals ("cannot tell whether this table REPLACES the factory
    // table: ..."). A caller who has acknowledged the replacement has taken that question off the
    // table, so the width is not consulted at all. That is a DESIGN CHOICE, not a hole, PROVIDED the
    // waiver cannot reach Q1 -- the per-row admission gate -- which is what this case pins.
    {
        std::printf("  J-i. allow_full_table_replace waives Q2 (the width question) and NOT Q1\n");
        std::vector<KvLayerTableEntry> repl16;
        for (int i = 0; i < 16; ++i) {
            KvLayerTableEntry e; e.layer = i; e.tier = "nvfp4"; repl16.push_back(e);
        }
        std::vector<KvPerLayerEvidence> ev16;
        for (int i = 0; i < 16; ++i) {
            KvPerLayerEvidence x;
            x.layer = i; x.energy_observed = true; x.energy_mean_l = 0.1; x.energy_rounds = 512;
            x.k_err_measured = true; x.v_err_measured = true; x.err_tier = "nvfp4";
            x.k_err_nmse = 5e-4; x.v_err_nmse = 5e-4;
            ev16.push_back(x);
        }
        KvPerLayerPolicyConfig waive;
        waive.allow_full_table_replace = true;
        waive.table_model_layers     = 32;   // the over-declaration that used to switch the gate off
        const auto w1 = kv_perlayer_table_violations(repl16, ev16, waive);
        for (const auto& s : w1) { std::printf("    %s\n", s.c_str()); }
        check(w1.empty(),
              "J-i: *** the ACK waives Q2, so the over-declared width is never consulted: a complete, "
              "fully evidenced replacement passes with ZERO violations ***");
        // CONTROL: the waiver must not reach Q1. One unbounded tier is enough to refuse the table.
        std::vector<KvLayerTableEntry> with_bad = repl16;
        { KvLayerTableEntry e; e.layer = 3; e.tier = "fp8"; with_bad.push_back(e); }
        const auto w2 = kv_perlayer_table_violations(with_bad, ev16, waive);
        for (const auto& s : w2) { std::printf("    %s\n", s.c_str()); }
        check(w2.size() == 1 && any_contains(w2, "has no bound for"),
              "J-i: *** control -- the waiver does NOT excuse an unbounded tier: the ROW gate still "
              "fires, exactly once ***");
        check(!any_contains(w2, "REPLACES") && !any_contains(w2, "WIDER"),
              "J-i: ...and no Q2 text appears: the two gates stay separable on this path too");
        // CONTROL: the same table WITHOUT the ACK is refused by the width cross-check -- so the
        // branch J-i pins is a real branch and not a vacuous pass.
        KvPerLayerPolicyConfig strict;
        strict.table_model_layers = 32;
        const auto w3 = kv_perlayer_table_violations(repl16, ev16, strict);
        check(any_contains(w3, "is WIDER than the factory table"),
              "J-i: *** control -- without the ACK the SAME table is refused by the width "
              "cross-check (H1): the waived branch is a real branch ***");
    }

    std::printf("\n=== %s (%d failing check(s)) ===\n",
                g_fail == 0 ? "ALL CHECKS AS EXPECTED" : "UNEXPECTED", g_fail);
    // THE EXIT CODE CARRIES THE RED. Before this change a failed check printed FAIL and the
    // test still returned 0, so a red control could not fail a run: the ctest was green
    // while a check was red. The red controls above (case C and case G) need this to be
    // worth anything.
    return g_fail == 0 ? 0 : 1;
}
