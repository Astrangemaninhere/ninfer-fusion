#pragma once

// ===========================================================================
// Per-layer KV quantization + plane-droppability policy (the KV-PERLAYER line).
// ===========================================================================
// HOST-ONLY, std-only, no CUDA, no engine headers. Same convention as
// product/kv_bit_budget.h, product/kv_kv_bits.h and serve/kv_cold_policy.h, so a
// plain host compile exercises it (see tests/ below) and the CLI, the server and
// an embedder can share ONE resolution path.
//
// WHY THIS HEADER EXISTS.
// The tree can already place a DIFFERENT tier on every layer, but it has only three
// ways to decide WHICH layer gets which, and all three are POSITIONAL:
//   (a) the rk4v4 leading window      -- product/kv_bit_budget.h's rk4v4_limit, applied as a
//                                     COUNT and realised by kKvBitBudgetPackOrder;
//   (b) the cold cap               -- spent by the packer on the next-shallow block;
//   (c) --kv-bit-budget ranges     -- contiguous layer ranges, each its own scalar DP.
// Nothing in that machinery can express "layer 7 is more sensitive than layer 11, so
// layer 7 keeps more bits": the DP state is (bits_used, cold_used), the objective is
// sum_t count_t * penalty_t and the penalty is a per-TIER constant, so the layer index
// never appears in the expression at all. A caller who wants a per-layer answer driven
// by per-layer EVIDENCE has nowhere to put the evidence.
//
// This header is that place. It does NOT replace the DP: it produces a per-layer
// OPINION that a caller may use, with the DP as the fallback for every layer the
// evidence cannot speak about. That is deliberate -- R1's constraint is that a policy
// must be ONE OF the strategies, and the default must not change today's behaviour.
//
// ===========================================================================
// WHAT IS MEASURABLE TODAY, AND WHAT IS NOT. This distinction is the point.
// ===========================================================================
// MEASURABLE TODAY (a real per-layer instrument exists in the tree):
//   * per-layer attention energy, ops::ft::snapshot() (src/ops/common/ft_stats.h):
//     EnergySample{layer, mean_l, rounds}, mean_l = mean over sampled heads of
//     partial_l (a log-sum-exp proxy). Gated by NINFER_FT_STATS=1 and requires
//     --no-cuda-graph. This IS a per-layer, runtime, measured quantity.
//     WARNING: it is a COST / compressibility proxy, NOT a quantization-error signal.
//   * per-layer, per-plane bit cost: exact arithmetic from kKvBitBudgetTiers[].bits_x100
//     and kv_bit_budget_plane_bytes(); not a measurement, but exactly decidable.
//   * per-layer (K format, V format) REALIZABILITY: exact and total today, via
//     kv_kv_bits_pair_tier() (product/kv_kv_bits.h).
//   * per-layer cold eligibility: decidable from the energy tap plus the dwell/streak/
//     confidence gates already in serve/kv_cold_policy.h.
//
// NOT MEASURABLE TODAY (no provider in this tree -- named, not silently defaulted):
//   * per-layer, per-plane QUANTIZATION ERROR at any candidate tier. The only producer
//     is tools/calib/analyze_kv.py, which consumes .kvc frames produced by
//     `ninfer-cli --kv-calib-dir DIR`; THAT FLAG DOES NOT EXIST in this tree's
//     apps/cli/options.cpp, and no .kvc writer (NINFERKVCAL1) exists in program_impl.h,
//     logical_kv_store.h or kv_auto_relayout.cpp. See REPORT.md finding N1.
//     => the whole "quality" column is unmeasurable today, which is why
//        kv_bit_budget_default_scores()'s quality column is BYTE-IDENTICAL to the
//        shipped PRIOR ladder and --kv-quality-weight 1 is a provable no-op.
//   * any evidence that a layer's KV plane may be DISCARDED. Nothing in the tree
//     computes it, and no policy ever EMITS "dropped" (build_ft_spec emits rk4v4/iso4e/
//     nvfp4; kv_kv_bits_pack_rows emits ladder names + cold). See REPORT.md finding N2.
//
// The policy below therefore has exactly ONE honest answer for a layer it has no
// evidence about: NO OPINION (the global mechanism decides), and NEVER a drop.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::product {

// ===========================================================================
// EVIDENCE -- one record per full-attention layer. Every field names its provider.
// ===========================================================================
struct KvPerLayerEvidence {
    std::int32_t layer = 0;

    // --- MEASURABLE TODAY ---------------------------------------------------
    // Provider: ops::ft::snapshot() / EnergySample (src/ops/common/ft_stats.h).
    bool energy_observed  = false;
    double energy_mean_l  = 0.0;   // EnergySample::mean_l
    std::uint64_t energy_rounds = 0;  // EnergySample::rounds (the confidence floor rides on this)

    // --- NOT MEASURABLE TODAY: named, and never fabricated -------------------
    // Provider needed: a per-layer, per-plane post-hoc quantisation error measurement.
    // The tree has none (REPORT.md N1). `measured == false` means "nobody looked",
    // which is NOT the same as "the error is zero".
    bool k_err_measured = false;
    double k_err_nmse   = 0.0;
    bool v_err_measured = false;
    double v_err_nmse   = 0.0;

    // WHICH TIER the two numbers above were measured AT. Empty <=> the caller did not say,
    // and that is a REFUSAL, not a default (see kv_perlayer_bound_applies below): every
    // bound in KvPerLayerPolicyConfig is TIER-SPECIFIC -- rk4v4_nmse_max bounds an error
    // measured at rk4v4, nvfp4_nmse_max one measured at nvfp4, drop_nmse_max one measured at
    // the tier the dropped planes would otherwise carry -- so an error that does not name
    // its tier is evidence for none of them.
    // The producer does not need a new measurement: the analyzer's column NAMES already
    // carry this fact (tools/calib/analyze_kv.py emits nmse_nvfp4_k/v, nmse_int8_k/v,
    // nmse_fp8_k/v, nmse_iso4e_k/v, and each name IS the tier it was measured at). Without
    // this field the compare was done anyway, and one column swap with no tier declared
    // flipped 16/16 layers from "all int8" to "all 16 planes may be discarded".
    std::string err_tier;

    [[nodiscard]] bool quant_error_measured() const noexcept {
        return k_err_measured && v_err_measured;
    }
};

struct KvPerLayerPolicyConfig {
    // Tier selection bounds on the measured per-layer NMSE. Only reached when the
    // evidence actually carries an error measurement.
    double rk4v4_nmse_max    = 1e-5;
    double nvfp4_nmse_max = 1e-3;
    // Drop bound: a plane may be dropped only below this measured per-layer NMSE.
    double drop_nmse_max  = 1e-4;
    // Confidence floor on the energy tap, mirroring kv_cold_policy.h's
    // min_window_rounds: a window below it is a real but noisy delta and must not
    // carry a decision.
    std::uint64_t min_energy_rounds = 32;
    // BOTH OFF BY DEFAULT: an unopted-in policy must be invisible (R1: "default must
    // not change today's behaviour"). With drop_enabled == false no verdict can ever
    // drop a plane, whatever the evidence says.
    bool drop_enabled        = false;
    bool energy_axis_enabled = false;
    // MAY A TABLE FROM ONE OF THE FOUR NON-DEFAULT WRITERS REPLACE THE FACTORY TABLE?
    // False by default. Naming EVERY layer is what a spec written by a tool that cannot
    // express the factory's rk4v4 layers does (tools/calib/analyze_kv.py:420 emits one entry
    // per layer), and a complete table replaces the factory table outright
    // (layouts_impl.h:1234-1240) -- including the layered K-lattice gain that is the ONLY
    // measured win in the tree (src/targets/qwen3_6_27b/impl/variant.cpp:24-30).
    // See kv_perlayer_table_violations below.
    bool allow_full_table_replace = false;
    // HOW WIDE THE MODEL IS -- the only datum that can decide whether a table from writers
    // (2)-(5) NAMES EVERY LAYER, i.e. whether it really replaces the factory table.
    // The previous test for that was `entries.size() == evidence.size()`, which is a
    // coincidence between two unrelated vectors: it refused a table that reproduces the
    // factory table row for row whenever the evidence vector happened to be as long (FALSE
    // POSITIVE -- the factory table accused of replacing itself), and it waved through a
    // genuine complete replacement whose evidence vector was a different length (FALSE
    // 0 == the caller did not say, and that is itself a REFUSAL BY NAME (see
    // kv_perlayer_table_violations), never a silent pass. THE SAME GOES FOR A VALUE ABOVE
    // kKvPerLayerFactoryTableLayers: this field is a CALLER DECLARATION, and a declaration this
    // gate cannot check is a switch that turns the gate OFF rather than a datum it acts on. A
    // width of 17 (or 32) makes "names every layer" unreachable for a 16-row table, so a genuine
    // 16/16 replacement -- every row row-compliant, every row evidenced -- passed with ZERO
    // violations (measured before the cross-check below: model=16 -> 1 refused, model=17 -> 0,
    // model=32 -> 0). The field therefore means THE WIDTH OF THIS POLICY'S DOMAIN, and it is
    // cross-checked against the factory table this gate compares against.
    std::int32_t table_model_layers = 0;
};

enum class KvVerdictKind : std::uint8_t {
    Unmeasured = 0,  // no evidence: NO OPINION, the global mechanism decides
    Measured   = 1,  // decided from a measured per-layer quantity
};

struct KvLayerVerdict {
    std::int32_t layer = 0;
    KvVerdictKind kind = KvVerdictKind::Unmeasured;
    // Empty <=> the policy has NO per-layer opinion for this layer and the caller must
    // fall back to --kv-bit-budget / --kv-bits / --kv-layer-storage. A non-empty tier
    // is a claim the policy is prepared to defend with the reasons below.
    std::string tier;
    // True <=> this layer's KV planes may be DISCARDED (KvCacheStorage::Dropped).
    bool drop_admissible = false;
    std::vector<std::string> reasons;
};

struct KvPerLayerDecision {
    std::vector<KvLayerVerdict> rows;
    // Did the policy have per-layer INFORMATION at all? False when every row carries
    // the same evidence, i.e. when the caller handed in a global number smeared over
    // the layers. Set by the policy, not by the caller.
    bool per_layer_information = false;
    bool degenerate_input      = false;
    std::string report;
    // WHY the policy issued nothing, in a form a caller cannot lose. These two counters
    // exist because `per_layer_information == true` with zero verdicts is otherwise
    // indistinguishable from "this box was never measured" (the honest fresh-box state):
    // in both cases every row reads tier="", drop=false and `per_layer_information`
    // differs only for the degenerate case. A caller that reads those two states as the
    // same thing treats a MISSING DECLARATION as an ABSENT INSTRUMENT and ships the
    // default table -- the quiet degradation this header exists to stop. Counted per row,
    // and only for rows that ended with no verdict at all.
    std::size_t rows_refused_no_declared_tier = 0;  // measured, but err_tier was empty
    std::size_t rows_refused_unbounded_tier   = 0;  // measured at a tier this ladder cannot bound
};

// ===========================================================================
// THE TIER-OF-EVIDENCE CHECK -- the ONLY path to any tier bound below
// ===========================================================================
// A bound may be applied ONLY to an error that DECLARES the tier it was measured at, and
// only to one measured at THAT SAME tier. The three entries that own a bound -- rk4v4, nvfp4
// and drop -- all go through this gate, and none of them has a branch that applies its
// bound anyway. The refusal is BY NAME (which tier was declared, which bound was withheld)
// rather than a default value, because the failure it prevents is silent: the same column,
// fed without a tier or with the wrong one, produced a tier for every layer with no
// message at all.
//
// Forward declaration: the drop predicate below needs the ladder ("the tier it would
// otherwise carry") to know WHICH tier's bound applies to that layer.
[[nodiscard]] inline std::string kv_perlayer_tier_for(const KvPerLayerEvidence& e,
                                                      const KvPerLayerPolicyConfig& cfg,
                                                      std::vector<std::string>* reasons);

// Non-zero <=> this policy's ladder has a rung (and therefore a bound) for that tier name.
// Zero for an undeclared tier ("") AND for the real tiers this ladder does not bound
// (bf16, fp8, iso4e): the caller learns the difference from the reason strings.
[[nodiscard]] inline int kv_perlayer_tier_rank(const std::string& tier) noexcept {
    if (tier == "rk4v4")    { return 1; }
    if (tier == "nvfp4") { return 2; }
    if (tier == "int8")  { return 3; }
    return 0;
}

// True <=> `tier`'s bound may be applied to `e`. False is ALWAYS accompanied by a reason
// string naming both the tier the bound belongs to and the tier the evidence declared, so
// a caller that logs the reasons cannot mistake a refusal for a decision.
[[nodiscard]] inline bool kv_perlayer_bound_applies(const KvPerLayerEvidence& e,
                                                    const std::string& tier,
                                                    std::vector<std::string>* reasons) {
    const auto say = [&](const std::string& s) { if (reasons != nullptr) { reasons->push_back(s); } };
    if (!tier.empty() && e.err_tier == tier) { return true; }
    if (tier.empty()) {
        say("REFUSED: no bound was applied -- this policy has no tier for this layer (the "
            "evidence's tier '" + e.err_tier + "' is not a rung of the ladder), so there is "
            "no tier a drop could replace");
    } else if (e.err_tier.empty()) {
        say("REFUSED: the bound for tier '" + tier + "' was NOT applied -- the evidence does "
            "not declare which tier its NMSE was measured at (err_tier is empty), and this "
            "bound is tier-specific");
    } else {
        say("REFUSED: the bound for tier '" + tier + "' was NOT applied -- the NMSE was "
            "measured at tier '" + e.err_tier + "', and the error at one tier is not evidence "
            "about another tier's bound");
    }
    return false;
}

// ===========================================================================
// PREDICATE 1 -- can this layer's KV planes be DROPPED?
// ===========================================================================
// Executable form of: "a plane may be discarded only if somebody MEASURED that this
// layer's K and V planes, at the tier it would otherwise carry, contribute nothing."
// Every clause is a named datum; there is no default-to-true branch.
[[nodiscard]] inline bool kv_perlayer_drop_admissible(const KvPerLayerEvidence& e,
                                                      const KvPerLayerPolicyConfig& cfg,
                                                      std::vector<std::string>* reasons) {
    const auto say = [&](const std::string& s) { if (reasons != nullptr) { reasons->push_back(s); } };
    if (!cfg.drop_enabled) {
        say("drop_enabled=false: the drop axis is not opted in");
        return false;
    }
    if (!e.quant_error_measured()) {
        // THE LOAD-BEARING CLAUSE. The energy tap is a COST signal; a cheap layer is
        // not an empty layer. Without a per-plane error measurement the answer is NO,
        // and it stays NO however small the energy is.
        say("no per-plane quantisation error measured for this layer (provider absent in "
            "this tree): a cost signal cannot justify a drop");
        return false;
    }
    // ENTRY 3 -- THE TIER OF THE DROP EVIDENCE. "at the tier it would otherwise carry" in
    // the paragraph above is the whole content of this clause: the NMSE must have been
    // measured at the tier this layer would otherwise be given, and it passes the same
    // tier-of-evidence gate as the two bounds above. What this refuses, concretely: a
    // column measured at the rk4v4 tier whose numbers also sit under drop_nmse_max, fed to a
    // layer the ladder would otherwise place on int8 -- the numbers pass, the EVIDENCE does
    // not. Before this gate that evidence answered "drop admissible", i.e. "these planes
    // may be discarded".
    const std::string would_carry = kv_perlayer_tier_for(e, cfg, nullptr);
    if (!kv_perlayer_bound_applies(e, would_carry, reasons)) { return false; }
    if (!(e.k_err_nmse <= cfg.drop_nmse_max && e.v_err_nmse <= cfg.drop_nmse_max)) {
        say("measured per-plane NMSE above the drop bound");
        return false;
    }
    if (!e.energy_observed || e.energy_rounds < cfg.min_energy_rounds) {
        say("energy tap below the confidence floor: the drop would rest on too few rounds");
        return false;
    }
    say("measured per-plane NMSE <= " + std::to_string(cfg.drop_nmse_max) +
        " with a confident energy observation");
    return true;
}

// ===========================================================================
// PREDICATE 2 -- which tier this layer is entitled to
// ===========================================================================
// Returns "" when the policy has no per-layer evidence for this layer. NOTE WHAT THIS
// DOES *NOT* DO: it does not run the global DP. Choosing a tier from a measured
// per-layer error is a different question from fitting a global bit budget, and the
// two are composed by the caller (see the integration note in REPORT.md), not merged
// here -- merging them is exactly the "one number for everything" failure this line
// exists to avoid.
[[nodiscard]] inline std::string kv_perlayer_tier_for(const KvPerLayerEvidence& e,
                                                      const KvPerLayerPolicyConfig& cfg,
                                                      std::vector<std::string>* reasons) {
    const auto say = [&](const std::string& s) { if (reasons != nullptr) { reasons->push_back(s); } };
    if (!e.quant_error_measured()) {
        say("no per-layer quantisation error measured: the policy has NO per-layer opinion "
            "for this layer, the global mechanism decides");
        return {};
    }
    const double err = std::max(e.k_err_nmse, e.v_err_nmse);  // the worse plane binds
    // ENTRY 1 / ENTRY 2. Each bound is reached ONLY through the tier-of-evidence gate, so
    // the rk4v4 bound meets an rk4v4 measurement and the nvfp4 bound meets an nvfp4 one. A bound
    // whose gate refused is not "satisfied": the ladder moves on.
    if (kv_perlayer_bound_applies(e, "rk4v4", reasons) && err <= cfg.rk4v4_nmse_max) {
        say("max(K,V) measured NMSE at the rk4v4 tier <= rk4v4 bound, rk4v4 entitled");
        return "rk4v4";
    }
    if (kv_perlayer_bound_applies(e, "nvfp4", reasons) && err <= cfg.nvfp4_nmse_max) {
        say("max(K,V) measured NMSE at the nvfp4 tier <= nvfp4 bound, nvfp4 entitled");
        return "nvfp4";
    }
    if (kv_perlayer_tier_rank(e.err_tier) == 0) {
        // LOUD REFUSAL, and deliberately NOT the int8 rung: an error measured at an
        // undeclared tier, or at a real tier this ladder has no bound for (bf16/fp8/iso4e),
        // is not evidence for the bottom rung either -- no bound applied, so there is no
        // opinion to defend. The reason names the tier the caller actually handed in.
        say("REFUSED: no bound was applied and the ladder has no rung for the evidence's tier "
            "'" + e.err_tier + "' (ladder: rk4v4/nvfp4/int8) -- NO per-layer opinion for this layer");
        return {};
    }
    say("max(K,V) measured NMSE above every bound that applied to it, int8 entitled");
    return "int8";
}

// ===========================================================================
// THE POLICY: one decision for the whole stack
// ===========================================================================
[[nodiscard]] inline KvPerLayerDecision kv_perlayer_decide(
    const std::vector<KvPerLayerEvidence>& evidence, const KvPerLayerPolicyConfig& cfg) {
    KvPerLayerDecision out;

    // -----------------------------------------------------------------------
    // DEGENERACY DETECTION. If every layer's evidence is identical, the input is a
    // GLOBAL number smeared over the layers: it cannot distinguish layer 3 from layer
    // 11, so a per-layer verdict built from it would be a fiction. Detect it, say so,
    // and refuse to report a per-layer judgement.
    //
    // Unmeasured rows are excluded from the comparison: "nothing measured" for every
    // layer is the honest initial state, not a collapsed global scalar.
    // -----------------------------------------------------------------------
    bool any_measured_ev = false;
    for (const auto& e : evidence) {
        if (e.energy_observed || e.quant_error_measured()) { any_measured_ev = true; break; }
    }
    bool degenerate = false;
    if (any_measured_ev && evidence.size() > 1) {
        degenerate = true;
        for (std::size_t i = 1; i < evidence.size(); ++i) {
            if (evidence[i].energy_observed != evidence[0].energy_observed ||
                evidence[i].energy_mean_l != evidence[0].energy_mean_l ||
                evidence[i].energy_rounds != evidence[0].energy_rounds ||
                evidence[i].k_err_measured != evidence[0].k_err_measured ||
                evidence[i].k_err_nmse != evidence[0].k_err_nmse ||
                evidence[i].v_err_measured != evidence[0].v_err_measured ||
                evidence[i].v_err_nmse != evidence[0].v_err_nmse) {
                degenerate = false;
                break;
            }
        }
    }
    out.degenerate_input = degenerate;

    for (const KvPerLayerEvidence& e : evidence) {
        KvLayerVerdict v;
        v.layer = e.layer;
        std::vector<std::string> reasons;
        if (degenerate) {
            // A collapsed input is reported as UNMEASURED, not as a decision. This is
            // the refusal that makes the red control below non-vacuous.
            v.kind = KvVerdictKind::Unmeasured;
            reasons.push_back("degenerate input: this layer's evidence is identical to every "
                              "other layer's, so it carries no per-layer information");
        } else {
            v.drop_admissible = kv_perlayer_drop_admissible(e, cfg, &reasons);
            v.tier            = v.drop_admissible ? std::string() : kv_perlayer_tier_for(e, cfg, &reasons);
            const bool any = v.drop_admissible || !v.tier.empty();
            v.kind = any ? KvVerdictKind::Measured : KvVerdictKind::Unmeasured;
            if (!any) {
                reasons.push_back("no per-layer opinion: the global mechanism decides this layer");
                // U-c: NAME THE REASON IN THE DECISION, not only in the free-text reasons.
                // A row that measured its K/V error and still got no verdict is one of exactly
                // two things, and neither is "nothing was measured".
                if (e.quant_error_measured()) {
                    if (e.err_tier.empty()) {
                        ++out.rows_refused_no_declared_tier;
                    } else if (kv_perlayer_tier_rank(e.err_tier) == 0) {
                        ++out.rows_refused_unbounded_tier;
                    }
                }
            }
        }
        v.reasons = std::move(reasons);
        out.rows.push_back(std::move(v));
    }

    out.per_layer_information = !degenerate && any_measured_ev;
    {
        std::string report = "[kv-perlayer] layers=" + std::to_string(out.rows.size()) +
                             " per_layer_information=" +
                             (out.per_layer_information ? "yes" : "no");
        if (degenerate) { report += " (DEGENERATE INPUT: one global value smeared over every layer)"; }
        std::size_t dropped = 0, opined = 0;
        for (const auto& v : out.rows) {
            if (v.drop_admissible) { ++dropped; }
            if (!v.tier.empty()) { ++opined; }
        }
        report += " with_tier=" + std::to_string(opined) + " dropped=" + std::to_string(dropped);
        // U-c: the same two facts, on the line a printf-only caller already reads. Without
        // these a log line reading "per_layer_information=yes with_tier=0 dropped=0" is
        // indistinguishable from a box that was never measured.
        if (out.rows_refused_no_declared_tier != 0) {
            report += " REFUSED_NO_DECLARED_TIER=" + std::to_string(out.rows_refused_no_declared_tier);
        }
        if (out.rows_refused_unbounded_tier != 0) {
            report += " REFUSED_UNBOUNDED_TIER=" + std::to_string(out.rows_refused_unbounded_tier);
        }
        if (cfg.energy_axis_enabled) {
            report += " (energy axis armed; it is a COST signal and never decides a drop)";
        }
        out.report = report;
    }
    return out;
}

// ===========================================================================
// MACHINE-CHECKED INVARIANTS
// ===========================================================================
// Returns one string per violation, empty when the decision is sound. A caller that
// prints a per-layer table MUST run this first -- the tree's own convention for a
// policy that can be wrong quietly (serve/kv_cold_policy.h invariant_violations).
//
// ⚠ "THIS FIRST" MEANS THIS *AND* kv_perlayer_declaration_violations. The two are separate
// names ON PURPOSE: this one answers "did my policy contradict itself?" and stays green for a
// decision that abstained consistently; that one answers "was the evidence I was handed
// usable?" and is red for a caller whose measurements cannot meet any bound. Folding the second
// into the first would red a policy that is behaving CORRECTLY, which is exactly why they were
// named apart. THE HARD CONVENTION IS THEREFORE: RUN BOTH -- and what it costs to run only this
// one is observable and expensive, so it is named here instead of left to prose: a caller that
// measured every layer and declared no tier gets GREEN from this function, `with_tier=0` from the
// decision, `per_layer_information=yes` and `REFUSED_NO_DECLARED_TIER=n` on the report line, and
// then ships the factory table -- the quiet degradation this header exists to stop.
// kv_perlayer_all_violations() below is that convention as ONE callable: prefer it, and the
// "I ran the gate" failure cannot happen.
//
// NOTE WHAT IS *NOT* IN HERE. The verdict-side invariants below are about the policy's own
// OUTPUT being self-consistent; they stay green for a caller whose evidence was unusable
// (rows refused because the tier was not declared), because that decision IS self-consistent
// -- it abstained. The unusable-evidence red is a separate gate with a separate name
// (kv_perlayer_declaration_violations), so that "my policy contradicted itself" and "my
// caller handed me evidence I could not use" remain two distinguishable failures.
[[nodiscard]] inline std::vector<std::string> kv_perlayer_invariant_violations(
    const KvPerLayerDecision& d) {
    std::vector<std::string> bad;
    for (const auto& v : d.rows) {
        const std::string at = "layer " + std::to_string(v.layer) + ": ";
        // I1: an unmeasured layer carries no opinion.
        if (v.kind == KvVerdictKind::Unmeasured && (!v.tier.empty() || v.drop_admissible)) {
            bad.push_back(at + "kind=Unmeasured but carries a tier or a drop");
        }
        // I2: a dropped layer owns no planes, so it cannot also name a tier. This is
        // the same fact as KvCacheStorage::Dropped having no DType to resolve.
        if (v.drop_admissible && !v.tier.empty()) {
            bad.push_back(at + "drop_admissible AND tier='" + v.tier +
                          "': a dropped plane has no codec to name");
        }
        // I3: a measured verdict must be justified by at least one reason. An
        // unexplained verdict is the failure mode this header exists to prevent.
        if (v.kind == KvVerdictKind::Measured && v.reasons.empty()) {
            bad.push_back(at + "kind=Measured with no reason recorded");
        }
    }
    // I4: the load-bearing one. A decision built from a COLLAPSED input must not
    // report per-layer information, and must not carry a single tier or drop. This is
    // the invariant that goes RED when the per-layer judgement is degraded into a
    // global single number.
    if (d.degenerate_input) {
        if (d.per_layer_information) {
            bad.push_back("degenerate_input=true but per_layer_information=true: a global "
                          "scalar is being reported as a per-layer judgement");
        }
        for (const auto& v : d.rows) {
            if (!v.tier.empty() || v.drop_admissible) {
                bad.push_back("layer " + std::to_string(v.layer) +
                              ": degenerate input produced a per-layer verdict ('" + v.tier +
                              "', drop=" + (v.drop_admissible ? "true" : "false") + ")");
            }
        }
    }
    // I5: if the policy says it had no per-layer information, then NOTHING in the
    // verdict vector may carry an opinion -- including the all-unmeasured case, where a
    // caller could otherwise inject a tier it did not measure.
    if (!d.per_layer_information) {
        for (const auto& v : d.rows) {
            if (!v.tier.empty() || v.drop_admissible) {
                bad.push_back("layer " + std::to_string(v.layer) +
                              ": per_layer_information=false but the row carries '" + v.tier +
                              "' (drop=" + (v.drop_admissible ? "true" : "false") + ")");
            }
        }
    }
    return bad;
}

// ===========================================================================
// I7 -- THE DECLARATION GATE (the caller-side red)
// ===========================================================================
// Turns the two counters on KvPerLayerDecision into strings a caller can gate a run on, so
// that "the evidence was handed in unusable" stops being a zero that looks like a healthy
// abstention. The three situations a bare `with_tier=0` cannot separate:
//   * this box was NEVER MEASURED          -> rows_refused_* are both 0 (nothing to declare)
//   * this box WAS MEASURED, and the numbers were handed in with no tier named
//                                          -> rows_refused_no_declared_tier == rows.size()
//   * this box WAS MEASURED and the numbers were COLLAPSED into one value smeared over every
//     layer (degenerate_input)             -> rows_refused_* are both 0 AS WELL, because the
//                                             per-row counting runs only on the non-degenerate
//                                             path of kv_perlayer_decide. Without the clause
//                                             below, this state was separated from "never
//                                             measured" by exactly ONE field (`degenerate_input`)
//                                             and by nothing a caller gates on.
// The second is the one that must not be silent: the caller has real numbers, believes it
// has per-layer information (`per_layer_information` is true), gets no verdict for any layer,
// and has no way to tell that the fix is one field. Before this gate the same input produced
// a tier for all 16 layers, so the change is behaviour-breaking and this is the siren.
[[nodiscard]] inline std::vector<std::string> kv_perlayer_declaration_violations(
    const KvPerLayerDecision& d) {
    std::vector<std::string> bad;
    if (d.degenerate_input) {
        // H4. THE COLLAPSED INPUT IS A CALLER-SIDE FAILURE TOO, and the two counters cannot see it
        // (they are per-row and live on the non-degenerate path). Naming it HERE is what makes
        // "one global value smeared over the layers" distinguishable from "this box was never
        // measured" through a gate a caller can act on. The verdict-side gate deliberately does
        // NOT carry it -- the abstention is self-consistent -- so this is not a second copy of I4.
        bad.push_back("the evidence is ONE VALUE SMEARED OVER ALL " +
                      std::to_string(d.rows.size()) + " layers (degenerate input): every row "
                      "carried the same measurement, so no per-layer verdict was issued and none "
                      "could be. This is NOT the same as 'never measured' -- that box trips "
                      "neither this gate nor the two counters and has degenerate_input=false -- "
                      "and it is not a usable per-layer judgement either: measure per layer, or "
                      "accept that this run has no per-layer opinion");
    }
    if (d.rows_refused_no_declared_tier != 0) {
        bad.push_back(std::to_string(d.rows_refused_no_declared_tier) + " of " +
                      std::to_string(d.rows.size()) + " layers MEASURED a K/V NMSE but did not "
                      "declare the tier it was measured at: every bound in this policy is "
                      "tier-specific, so NOTHING was decided -- which is not the same as 'not "
                      "measured'. Fill KvPerLayerEvidence::err_tier (the analyzer's nmse_*_k/v "
                      "key names are exactly this field) or accept that this run has no "
                      "per-layer opinion");
    }
    if (d.rows_refused_unbounded_tier != 0) {
        bad.push_back(std::to_string(d.rows_refused_unbounded_tier) + " of " +
                      std::to_string(d.rows.size()) + " layers measured a K/V NMSE at a tier this "
                      "ladder cannot bound (ladder: rk4v4/nvfp4/int8): no bound applies to that "
                      "evidence, so no verdict was issued for it");
    }
    return bad;
}

// ===========================================================================
// THE "MUST RUN THIS FIRST" ENTRY POINT -- both gates, one call
// ===========================================================================
// The convention stated above kv_perlayer_invariant_violations ("MUST run this first") is only
// worth anything if a caller cannot run HALF of it. This is that convention as ONE callable: the
// verdict-side invariants AND the caller-side declaration gate, so `empty()` answers the question
// a caller actually has -- "may I print this table?" -- and not the narrower one, "is my policy
// self-consistent?". The consequence of using the half-gate is named at that comment: green here,
// `with_tier=0` from the decision, and the factory table shipped.
[[nodiscard]] inline std::vector<std::string> kv_perlayer_all_violations(
    const KvPerLayerDecision& d) {
    std::vector<std::string> bad = kv_perlayer_invariant_violations(d);
    const std::vector<std::string> decl = kv_perlayer_declaration_violations(d);
    bad.insert(bad.end(), decl.begin(), decl.end());
    return bad;
}

// ===========================================================================
// I6 -- THE EVIDENCE-SIDE INVARIANT
// ===========================================================================
// An error measurement that does not name a tier this ladder can bound cannot meet any
// bound (see kv_perlayer_bound_applies above), so a decision built from it is unsound
// however plausible it looks. It is checked on the EVIDENCE, next to the verdict-side
// invariants, for two reasons: a caller that ignores the reasons vector would otherwise
// never hear about it, and a caller that re-implements the compare (the failure this
// header exists to stop) is caught here rather than in a comment.
// Returns one string per violation, empty when every measurement names a usable tier.
[[nodiscard]] inline std::vector<std::string> kv_perlayer_evidence_violations(
    const std::vector<KvPerLayerEvidence>& evidence) {
    std::vector<std::string> bad;
    for (const auto& e : evidence) {
        const std::string at = "layer " + std::to_string(e.layer) + ": ";
        if (!e.quant_error_measured()) { continue; }
        if (e.err_tier.empty()) {
            bad.push_back(at + "a K/V NMSE was measured but err_tier is empty: no tier-specific "
                               "bound can be applied to it (the analyzer's nmse_*_k/v key names "
                               "are exactly this field)");
            continue;
        }
        if (kv_perlayer_tier_rank(e.err_tier) == 0) {
            bad.push_back(at + "err_tier='" + e.err_tier + "' is not a rung of this ladder "
                               "(rk4v4/nvfp4/int8): the bound it would meet does not exist");
        }
    }
    return bad;
}

// ===========================================================================
// WHICH TABLE, AND WHO WROTE IT -- the five writers, named
// ===========================================================================
// "Which layer gets which format" has FIVE writers in this tree, and only the first is a
// default. A policy that only knows about the analyzer answers one fifth of the question.
//   (1) Variant::default_layer_kv_dtypes -- the FACTORY table, and today's default:
//       rk4v4 on {0,1,3,4,6,7}, NVFP4 above (src/targets/qwen3_6_27b/impl/variant.cpp:42-45).
//       It is the only table with an independent MEASURED benefit, and it states it:
//       13.3k zh perplexity at ctx 4096 -> THIS table 1.020 (best of all mixes),
//       all-Rk4v4 1.112, all-NVFP4 1.706, all-I8 1.522 (:24-30).
//   (2) --kv-layer-storage  -- slot-by-slot REPLACEMENT of (1): the table and its write
//       mask come straight from the options (layouts_impl.h:1234-1240).
//   (3) --kv-tier-formats hot=...  -- replaces the WHOLE table and fills the write mask,
//       and is refused by name when combined with (2) (layouts_impl.h:1253-1272).
//   (4) NINFER_KV_{K,V,}_BITS / _MODE / _QUALITY_WEIGHT -- the environment layer,
//       "CLI > environment > default" (src/product/kv_kv_bits.h:341-393).
//   (5) POST /reload_kv {"kv_layer_storage": "..."} -- the grammar of (2), at runtime, on
//       a drained pool (src/serve/http_server.cpp:471-503).
//
// WHY THIS NEEDS ITS OWN GATE. The analyzer emits a COMPLETE spec (one named entry per
// layer), so pasting it is writer (2) naming every slot: the factory table is gone. At
// budget 1.20 the table it emits is 13x nvfp4 + 3x fp8 -- i.e. the all-NVFP4 mix the
// factory table measured at 1.706 while itself measuring 1.020 -- and its rk4v4 window is
// not even expressible there (cost() has no rk4v4 arm and the promotion tuple is
// ("fp8","int8","bf16"), analyze_kv.py:399-418).
//
// The gate below is the admission test for a table from writers (2)-(5). It REFUSES BY
// NAME and never merges, reorders or "fixes up" the table. Entries marked
// from_factory_table are writer (1): the factory table is the SHIPPED default with its
// own measurement behind it, not a claim built out of this evidence channel.
struct KvLayerTableEntry {
    std::int32_t layer = 0;
    std::string  tier;
    bool from_factory_table = false;   // writer (1)
};

// ===========================================================================
// THE FACTORY TABLE AS DATA -- what a `from_factory_table` claim must MATCH
// ===========================================================================
// The prose above NAMES the factory table; a bool cannot be checked against prose, and a
// claim that cannot be checked is not a claim -- it is an exemption for the asking. So the
// table is repeated here in checkable form. Same source of truth as the comment above:
// src/targets/qwen3_6_27b/impl/variant.cpp:42-45 (default_layer_kv_dtypes).
inline constexpr std::int32_t kKvPerLayerFactoryTableLayers = 16;

// rk4v4 exactly where the K-lattice gain was measured to pay: {0,1,3,4,6,7}.
[[nodiscard]] inline bool kv_perlayer_factory_table_is_rk4v4_layer(std::int32_t layer) noexcept {
    return layer == 0 || layer == 1 || layer == 3 || layer == 4 || layer == 6 || layer == 7;
}

// The factory table's tier on `layer`, or "" when the factory table has no such layer --
// i.e. when a `from_factory_table` claim is being made about a row the factory table does
// not own.
[[nodiscard]] inline std::string kv_perlayer_factory_table_tier(std::int32_t layer) {
    if (layer < 0 || layer >= kKvPerLayerFactoryTableLayers) { return {}; }
    return kv_perlayer_factory_table_is_rk4v4_layer(layer) ? std::string("rk4v4") : std::string("nvfp4");
}

// ===========================================================================
// THE TABLE GATE
// ===========================================================================
// Two questions, one per hole this function used to have:
//   * Q1 (per row): is this row what it says it is? A row may be exempted from the
//     evidence checks ONLY by claiming to be writer (1)'s row, and that claim is now
//     verified against the factory table above instead of taken on trust.
//   * Q2 (whole table): does this table REPLACE the factory table? That is decidable from
//     the model's width (KvPerLayerPolicyConfig::table_model_layers) and the table's own
//     content -- and NOT from `entries.size() == evidence.size()`, which was a coincidence
//     between two unrelated vectors (see the field's comment).
[[nodiscard]] inline std::vector<std::string> kv_perlayer_table_violations(
    const std::vector<KvLayerTableEntry>& entries,
    const std::vector<KvPerLayerEvidence>& evidence,
    const KvPerLayerPolicyConfig& cfg) {
    std::vector<std::string> bad;
    std::size_t from_other_writers = 0;
    for (const auto& entry : entries) {
        const std::string at = "layer " + std::to_string(entry.layer) + ": ";
        if (entry.from_factory_table) {
            // Q1. THE CLAIM IS CHECKED, LAYER BY LAYER. `from_factory_table` means "this row
            // IS writer (1)'s row, and writer (1)'s rows are exempt because the factory table
            // is the shipped default with its own measurement behind it". That is decidable
            // against the factory table, so it is decided here. Without this clause the
            // exemption was free: a complete 16-row table whose every row says 'bf16' -- a
            // tier this ladder has no bound for, so the evidence checks below would have
            // refused all 16 rows -- passed this gate with ZERO violations simply by setting
            // the bool on every row, i.e. a wholly unbounded table admitted by one bool.
            const std::string factory_tier = kv_perlayer_factory_table_tier(entry.layer);
            if (factory_tier.empty()) {
                bad.push_back(at + "claims to be the factory table, but the factory table is " +
                              std::to_string(kKvPerLayerFactoryTableLayers) +
                              " layers wide and has no layer " + std::to_string(entry.layer) +
                              ": the exemption is being claimed for a row the factory table does "
                              "not own");
            } else if (entry.tier != factory_tier) {
                bad.push_back(at + "claims to be the factory table but writes '" + entry.tier +
                              "', while the factory table writes '" + factory_tier +
                              "' on this layer (variant.cpp:42-45): the exemption is not the "
                              "factory's to grant");
            }
            continue;   // reached only after the claim above has been verified
        }
        ++from_other_writers;
        if (kv_perlayer_tier_rank(entry.tier) == 0) {
            bad.push_back(at + "the table writes '" + entry.tier + "', which this ladder has no "
                               "bound for (rk4v4/nvfp4/int8): nothing in this evidence channel "
                               "can justify it, so the write is a naming choice and is refused");
            continue;
        }
        const KvPerLayerEvidence* match = nullptr;
        for (const auto& e : evidence) { if (e.layer == entry.layer) { match = &e; break; } }
        if (match == nullptr || !match->quant_error_measured()) {
            bad.push_back(at + "the table writes '" + entry.tier + "' but no per-layer error was "
                               "measured for this layer: the write is unevidenced");
            continue;
        }
        if (match->err_tier != entry.tier) {
            bad.push_back(at + "the table writes '" + entry.tier + "' but this layer's measured "
                               "NMSE was taken at tier '" + match->err_tier + "'");
        }
    }

    // Q2. "DOES THIS TABLE REPLACE THE FACTORY TABLE?" -- asked with the model's width and
    // the table's content, never with a length coincidence. The old predicate compared the
    // table's length against the EVIDENCE vector's length; those two vectors have nothing to
    // do with each other, so it was wrong in BOTH directions:
    //   * FALSE POSITIVE: a table that reproduces the factory table row for row, handed in as
    //     plain entries -- which is exactly what a caller re-deriving writer (1) produces --
    //     was refused as "REPLACES the factory table" whenever the evidence was also 16 long.
    //     The factory table was accused of replacing itself.
    //   * FALSE NEGATIVE: a genuine COMPLETE replacement whose evidence vector was a different
    //     length (only the measured layers handed in, which is the normal case) walked through
    //     SILENTLY: the gate's whole subject bypassed because two unrelated numbers differed.
    // The second clause of the old test -- `entries.size() > 1` -- is kept: a single-row table
    // is writer (2)'s ordinary slot-by-slot use and was never this gate's subject.
    //   * SWITCHED OFF BY OVER-DECLARING THE WIDTH: the width was taken on trust, so a caller
    //     could declare 17 (or 32), make `named_in_range == table_model_layers` unreachable for
    //     its 16-row table, and watch a complete, row-compliant, fully evidenced replacement pass
    //     with ZERO violations. The width is now cross-checked against the factory table and an
    //     over-declaration is refused by name -- the same spirit as the `<= 0` case above.
    if (from_other_writers != 0 && entries.size() > 1 && !cfg.allow_full_table_replace) {
        if (cfg.table_model_layers <= 0) {
            // REFUSAL BY NAME, not a default. An unjudgeable table must not be waved through:
            // a length comparison waved through every table whose evidence happened to be a
            // different length, which is the normal case.
            bad.push_back("cannot tell whether this table REPLACES the factory table: " +
                          std::to_string(from_other_writers) + " of its " +
                          std::to_string(entries.size()) + " rows come from a writer other than "
                          "the factory, and KvPerLayerPolicyConfig::table_model_layers is 0 (the "
                          "model's layer count was not declared), so 'names every layer' is "
                          "undecidable. Declare table_model_layers, or set "
                          "allow_full_table_replace to accept the replacement");
        } else if (cfg.table_model_layers > kKvPerLayerFactoryTableLayers) {
            // THE WIDTH IS CROSS-CHECKED AGAINST THE FACTORY TABLE. `table_model_layers` is a
            // CALLER DECLARATION, and over-declaring it is the switch that turns this gate off:
            // `names_every_layer` is `named_in_range == table_model_layers`, so a 16-row table
            // handed in under `table_model_layers = 17` (or 32) can never name every layer, the
            // replacement verdict is never reached, and a genuine 16/16 replacement -- every row
            // row-compliant, every row evidenced -- passes with ZERO violations (measured before
            // this clause existed: model=16 -> 1 refused, model=17 -> 0, model=32 -> 0). That is
            // the same defect class as the free `from_factory_table` exemption Q1 closes: an
            // unverifiable declaration taking the gate out of the circuit.
            // The comparison is against kKvPerLayerFactoryTableLayers because that constant IS
            // this gate's idea of the model: both "names every layer" and "is equivalent to the
            // factory table" are asked AGAINST THE FACTORY TABLE, and this header's checkable
            // copy of it is that wide (variant.cpp:42-46 is a 64-wide array, but only this policy
            // domain's window is decidable here -- see kv_perlayer_factory_table_tier, which has
            // no answer for layer >= kKvPerLayerFactoryTableLayers). A width the factory table
            // does not reach is not a model width this gate can judge, so it is refused BY NAME.
            bad.push_back("cannot tell whether this table REPLACES the factory table: " +
                          std::to_string(from_other_writers) + " of its " +
                          std::to_string(entries.size()) + " rows come from a writer other than "
                          "the factory, and KvPerLayerPolicyConfig::table_model_layers is " +
                          std::to_string(cfg.table_model_layers) + ", which is WIDER than the "
                          "factory table (" + std::to_string(kKvPerLayerFactoryTableLayers) +
                          " layers, variant.cpp:42-46) that this gate compares against: 'names "
                          "every layer' cannot be judged against a table that does not reach that "
                          "far, and a 16-row replacement under a " +
                          std::to_string(cfg.table_model_layers) +
                          "-layer declaration would otherwise be waved through SILENTLY -- the very "
                          "pass this gate exists to stop. Declare the width of THIS policy's "
                          "domain (kKvPerLayerFactoryTableLayers), or set "
                          "allow_full_table_replace to accept the replacement");
        } else {
            // Does the table NAME EVERY LAYER of the model? That, and only that, is what makes
            // it a wholesale replacement (layouts_impl.h:1234-1240). Counted on the layer
            // INDICES, once each, so a duplicated or out-of-range row cannot pad the count.
            std::vector<char> named(static_cast<std::size_t>(cfg.table_model_layers), 0);
            std::size_t named_in_range = 0;
            for (const auto& entry : entries) {
                if (entry.layer >= 0 && entry.layer < cfg.table_model_layers &&
                    named[static_cast<std::size_t>(entry.layer)] == 0) {
                    named[static_cast<std::size_t>(entry.layer)] = 1;
                    ++named_in_range;
                }
            }
            const bool names_every_layer =
                named_in_range == static_cast<std::size_t>(cfg.table_model_layers);
            // ...and does it differ from the factory table IN EFFECT? A table that reproduces
            // the factory table tier for tier destroys no measured benefit and so needs no
            // waiver. This is the clause that removes the false positive above, and it is
            // CONTENT, not length -- the thing the old predicate could not see.
            bool factory_equivalent = names_every_layer;
            if (factory_equivalent) {
                for (const auto& entry : entries) {
                    if (entry.tier != kv_perlayer_factory_table_tier(entry.layer)) {
                        factory_equivalent = false;
                        break;
                    }
                }
            }
            if (names_every_layer && !factory_equivalent) {
                bad.push_back("this table names every one of the " +
                              std::to_string(cfg.table_model_layers) +
                              " layers and its tiers are not the factory table's, so it REPLACES "
                              "the factory table (layouts_impl.h:1234-1240); the factory table is "
                              "the only one with a measured benefit (1.020 vs all-NVFP4 1.706) and "
                              "the producer cannot express its rk4v4 layers at all. Set "
                              "KvPerLayerPolicyConfig::allow_full_table_replace to accept that");
            }
        }
    }
    return bad;
}

// True <=> the verdict vector actually differs between layers. Exposed because the
// answer is the whole point of this line and a caller should be able to assert it.
[[nodiscard]] inline bool kv_perlayer_is_decisive(const KvPerLayerDecision& d) {
    for (std::size_t i = 1; i < d.rows.size(); ++i) {
        if (d.rows[i].tier != d.rows[0].tier) { return true; }
        if (d.rows[i].drop_admissible != d.rows[0].drop_admissible) { return true; }
    }
    return false;
}

} // namespace ninfer::product
