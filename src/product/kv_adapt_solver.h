#pragma once

// ===========================================================================
// Per-layer KV format selector: exact multiple-choice knapsack over (layer, bytes)
// ===========================================================================
// HOST-ONLY, std-only, no CUDA, no engine headers beyond product/kv_bit_budget.h.
// Same convention as product/kv_bit_budget.h, product/kv_perlayer_policy.h and
// product/kv_kv_bits.h, so a plain host compile exercises it.
//
// WHY THIS HEADER EXISTS (and what it fixes in kv_bit_budget.h).
// kv_bit_budget.h's allocator solves min_T sum_t count_t * penalty_t subject to a
// single bit-budget sum. That objective has NO LAYER INDEX in it, and the three
// positional mechanisms around it (the rk4v4 leading-window COUNT, the cold cap spent
// on the next-shallow block, --kv-bit-budget ranges) are the only ways to say
// WHICH layer gets what. Three consequences, all decidable against the shipped
// header and all fixed here:
//
//   1. THE SHIPPED ALLOCATOR CANNOT EXPRESS ITS OWN FACTORY TABLE.
//      The factory table (src/targets/qwen3_6_27b/impl/variant.cpp,
//      default_layer_kv_dtypes) puts rk4v4 on {0,1,3,4,6,7} -- a set with HOLES at
//      2 and 5. The allocator realises rk4v4 by packing kKvBitBudgetPackOrder from
//      cursor 0, so the rk4v4 layers are always a prefix 0..k-1. Sweeping every
//      budget and every rk4v4_limit therefore produces only the prefix family
//      {}, {0}, {0..6}, {0..7}, ... and {0,1,3,4,6,7} is unreachable at ANY
//      budget. The mechanism is "the packer decides layer identity", not a
//      mistuned parameter.
//   2. THE OBJECTIVE CANNOT DEFEND THE FACTORY TABLE EITHER, only a measurement
//      can. With the shipped prior penalties, rk4v4 (425, penalty 8) is cheaper per
//      element AND lower-penalty than nvfp4 (450, penalty 30), so the prior's own
//      optimum at the factory's byte footprint is "rk4v4 everywhere", never the
//      factory's 6-layer mix. See kv_adapt_prior_option_sets() below: this header
//      makes that comparison runnable rather than asserted.
//   3. AN UPPER BOUND CARRIED IN THE SURVIVING PATH IS NOT A CONSTRAINT OF THE DP
//      STATE. kv_bit_budget.h:428-430/463-472 keeps rk4v4_count as a side value of
//      the surviving path, so at equal (bits, cold) the first-arrived (highest
//      rk4v4-count) path wins and a later lower-count path cannot displace it. That
//      is a heuristic, not an optimal bound. Here the bound is a property of the
//      LAYER's allowed set F_l, so the constraint is a function of the state key
//      and the optimality argument goes through.
//
// WHAT IS DIFFERENT, STATED SO IT CAN BE FALSIFIED.
//   * Decision variable is per-layer: T(l) in F_l, F_l an ARBITRARY subset.
//     Holes are representable; prefixes are only the special case F_l = F for
//     l < k and {} above.
//   * Objective is sum_l W[l][T(l)] -- the layer index appears, which is the
//     necessary condition for per-layer EVIDENCE to have anywhere to live.
//   * Byte grid is 16 B, not 0.01 bit. Every plane cost in the shipped ladder is
//     a multiple of 16 B (gcd(32768,16896,17408,9216,8704)=512; the cold records
//     9232 and 9632 are 16*577 and 16*602) so the byte arithmetic is EXACT and
//     the nearbyint/banker's-rounding class of bug (kv_bit_budget.h:410-419)
//     disappears from the hot path entirely.
//   * Cold is a MODE, not a seventh pseudo-tier: the caller runs the exact DP
//     twice (cold admitted / cold refused) and takes the better, instead of
//     pricing cold with one grid point that cannot tell an int8 layer's 9232 B
//     record from an nvfp4 layer's 9632 B one (kv_bit_budget.h
//     kKvBitBudgetColdSlotBytesInt8Raw names the sign of that difference itself: the
//     grid point OVER-states an int8 layer by 400 B, and it UNDER-stated it by 2544 B
//     only while the rANS record was the retired 6688).
//
// WHAT THIS HEADER DOES NOT DO. It does not choose the measured matrix, it does
// not replace the default table, it does not touch the CLI. W[l][f] is an INPUT,
// and an option whose KvAdaptProvenance::measured is false is an UNMEASURED cell:
// it is selectable (so a P0 run still produces a plan) but it is reported in
// KvAdaptSolution::unmeasured_cells and its loss is never silently treated as a
// measurement. That mirrors product/kv_perlayer_policy.h:85's semantic --
// "measured == false means nobody looked, which is NOT the same as the error is
// zero" -- rather than inventing a second convention.

#include "product/kv_bit_budget.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::product {

// ---------------------------------------------------------------------------
// Byte grid. 16 B is the gcd of every resident plane the ladder can select and
// of both cold records (see the header comment), so costs and budget stay in
// exact integer arithmetic. A cost that is NOT a multiple of 16 is REJECTED
// loudly rather than rounded, because a silent round here is exactly the class
// of defect this header exists to remove.
// ---------------------------------------------------------------------------
inline constexpr std::int64_t kKvAdaptGridBytes = 16;

// ---------------------------------------------------------------------------
// PROVENANCE. One record per measured cell. Everything a reader needs to decide
// whether the cell's loss may be compared to another cell's.
// ---------------------------------------------------------------------------
struct KvAdaptProvenance {
    bool measured = false;          // false => UNMEASURED: nobody looked
    std::string corpus_sha256;
    std::int64_t corpus_unique_lines = -1;   // -1 = not recorded
    std::int64_t corpus_max_line_repeat = -1;
    std::int32_t ctx = 0;
    std::string metric;             // "M1" (needle retrieval) | "M2" (ppl) | ""
    std::string arm_id;
    std::string binary_sha256;
    std::int32_t repeats = 0;
    double noise_floor = -1.0;      // sigma from the repeat arms; < 0 = not measured

    [[nodiscard]] bool usable() const noexcept { return measured; }
};

// ---------------------------------------------------------------------------
// ONE CANDIDATE CELL for one layer: the (K format, V format, codec) triple is
// carried by name, exactly as kv_bit_budget.h carries its tiers by spec_name so
// that every name-keyed path (--kv-layer-storage emission, the planner) keeps
// working unchanged.
// ---------------------------------------------------------------------------
struct KvAdaptOption {
    std::string spec_name;                 // "bf16"|"int8"|"fp8"|"nvfp4"|"iso4e"|"rk4v4"|"cold"
    std::int32_t plane_bytes = 0;          // per (page, kv_head, K|V plane); must be %16==0
    double loss = 0.0;                     // W[ell][f]; meaningful only when prov.measured
    bool is_cold = false;
    KvAdaptProvenance prov;

    [[nodiscard]] std::int64_t units() const noexcept {
        if (plane_bytes <= 0 || plane_bytes % kKvAdaptGridBytes != 0) {
            throw std::invalid_argument(
                "kv-adapt: plane_bytes must be a positive multiple of 16 (got " +
                std::to_string(plane_bytes) + " for '" + spec_name + "')");
        }
        return static_cast<std::int64_t>(plane_bytes) / kKvAdaptGridBytes;
    }
};

// ---------------------------------------------------------------------------
// The per-layer allowed set F_l -- the whole point of the design. Constraint
// kinds C2 (realizability), C4 (kernel exists), C5 (rk4v4 verified per layer),
// C6 (cold per-layer fine print) and the C7 per-range caps are ALL folded into
// F_l by the caller, which is what makes every one of them a function of the
// state key rather than a side condition.
// ---------------------------------------------------------------------------
struct KvAdaptLayerSet {
    std::vector<KvAdaptOption> options;

    [[nodiscard]] const KvAdaptOption* find(std::string_view name) const noexcept {
        for (const KvAdaptOption& option : options) {
            if (option.spec_name == name) { return &option; }
        }
        return nullptr;
    }
};

enum class KvAdaptColdMode : std::int32_t {
    Off = 0,       // cold refused for every layer: F_l excludes every cold cell
    Eligible = 1,  // cold admitted; the caller must have checked the GLOBAL
                   // precondition (kv_tier_formats.h kv_cold_pool_reachable) --
                   // it is an admission predicate evaluated once per plan, not a
                   // per-layer objective term. A stack containing rk4v4 is not
                   // cold-reachable, so admitting cold here while also admitting
                   // rk4v4 is a caller error and is REFUSED, not silently priced.
};

struct KvAdaptRequest {
    std::int32_t layers = 0;
    std::int64_t budget_plane_bytes = 0;     // hard cap, summed over layers
    // Either one set applied to every layer (per_layer.size() == 1) or one per
    // layer (per_layer.size() == layers).
    std::vector<KvAdaptLayerSet> per_layer;
    KvAdaptColdMode cold_mode = KvAdaptColdMode::Off;
};

struct KvAdaptLayerPick {
    std::int32_t layer = 0;
    std::int32_t option_index = -1;
    std::string spec_name;
    std::int32_t plane_bytes = 0;
    double loss = 0.0;
    bool measured = false;
    std::string evidence_arm;   // non-empty only when the cell is measured
};

struct KvAdaptSolution {
    bool feasible = false;
    std::string spec;                        // --kv-layer-storage grammar
    std::int64_t achieved_plane_bytes = 0;
    double predicted_loss = 0.0;
    std::vector<KvAdaptLayerPick> per_layer;
    // Every (layer, format) the plan actually USED that has no measurement.
    // Must be listed, never silently replaced by 0.
    std::vector<std::string> unmeasured_cells;
    // Layers where not one option in F_l is measured: the plan cannot be
    // defended for these layers at all.
    std::vector<std::int32_t> undefensible_layers;
    KvAdaptColdMode cold_mode = KvAdaptColdMode::Off;
    std::int64_t states_expanded = 0;        // (layer, units) pairs visited
    std::int64_t transitions = 0;            // (layer, units, option) triples taken
    std::int64_t capacity_units = 0;         // floor(budget / 16)
    std::int64_t budget_slack_bytes = 0;     // budget - achieved (>= 0)
};

namespace detail {

[[nodiscard]] inline const KvAdaptLayerSet& kv_adapt_set_for(
    const KvAdaptRequest& request, std::int32_t layer) {
    return request.per_layer.size() == 1
               ? request.per_layer.front()
               : request.per_layer[static_cast<std::size_t>(layer)];
}

// Emit the --kv-layer-storage grammar (product/kv_options.h:61 accepts
// first-last:name, single:name, comma-separated, in any order). Runs are emitted
// in ASCENDING layer order, which the parser accepts and which is the only
// ordering a reader can verify by eye against variant.cpp.
[[nodiscard]] inline std::string kv_adapt_emit_spec(
    const std::vector<KvAdaptLayerPick>& picks) {
    std::string spec;
    std::size_t index = 0;
    while (index < picks.size()) {
        const std::string& name = picks[index].spec_name;
        std::size_t end = index;
        while (end + 1 < picks.size() && picks[end + 1].spec_name == name) { ++end; }
        if (!spec.empty()) { spec += ","; }
        spec += std::to_string(picks[index].layer);
        if (end != index) { spec += "-" + std::to_string(picks[end].layer); }
        spec += ":";
        spec += name;
        index = end + 1;
    }
    return spec;
}

} // namespace detail

// ---------------------------------------------------------------------------
// The solver. Exact DP over (layer, bytes-used/16) minimising sum_l W[l][T(l)].
//
//   dp[l+1][u + c(l,f)] = min(dp[l+1][u + c(l,f)], dp[l][u] + W[l][f]), f in F_l
//   answer = min over u <= capacity of dp[L][u], backtracked through choice[].
//
// Complexity O(L * U * max|F|) with U = capacity/16 in units. For the shipped
// ladder the worst case is U = 16*32768/16 = 32768 (all bf16) so the bound is
// ~2.6e6 transitions, and reachability prunes it far below that in practice.
// Ties: a STRICT improvement is required to overwrite, so the first candidate in
// the caller's option order wins an exact tie. That is deterministic and
// documented rather than incidental -- with a LAYER-INDEPENDENT loss (the shipped
// prior) an exact tie is the normal case, and the tie-break is what decides which
// layer receives the indifferent format.
// ---------------------------------------------------------------------------
[[nodiscard]] inline KvAdaptSolution kv_adapt_solve(const KvAdaptRequest& request) {
    KvAdaptSolution solution;
    solution.cold_mode = request.cold_mode;

    if (request.layers <= 0) {
        throw std::invalid_argument("kv-adapt: layer count must be positive");
    }
    if (request.per_layer.empty()) {
        throw std::invalid_argument("kv-adapt: per_layer must not be empty");
    }
    if (request.per_layer.size() != 1 &&
        request.per_layer.size() != static_cast<std::size_t>(request.layers)) {
        throw std::invalid_argument(
            "kv-adapt: per_layer must hold either 1 set or one set per layer");
    }
    if (request.budget_plane_bytes <= 0) {
        throw std::invalid_argument("kv-adapt: budget must be positive");
    }

    for (std::int32_t layer = 0; layer < request.layers; ++layer) {
        const KvAdaptLayerSet& set = detail::kv_adapt_set_for(request, layer);
        if (set.options.empty()) {
            throw std::invalid_argument("kv-adapt: layer " + std::to_string(layer) +
                                        " has an empty allowed set F_l");
        }
        for (const KvAdaptOption& option : set.options) {
            if (option.is_cold && request.cold_mode != KvAdaptColdMode::Eligible) {
                throw std::invalid_argument(
                    "kv-adapt: layer " + std::to_string(layer) + " offers cold option '" +
                    option.spec_name + "' but cold_mode is Off");
            }
            if (request.cold_mode == KvAdaptColdMode::Eligible && !option.is_cold &&
                option.spec_name == "rk4v4") {
                throw std::invalid_argument(
                    "kv-adapt: cold_mode Eligible admits rk4v4 on layer " +
                    std::to_string(layer) +
                    " -- the cold pool's reachability predicate requires EVERY layer to be "
                    "cold-capable (kv_tier_formats.h kv_cold_pool_reachable), which rk4v4 is not; "
                    "run the two modes as separate exact solves instead");
            }
            (void)option.units();  // validates the 16 B grid up front
        }
    }

    const std::int64_t capacity = request.budget_plane_bytes / kKvAdaptGridBytes;
    if (capacity <= 0) {
        throw std::invalid_argument("kv-adapt: budget is smaller than one 16 B grid step");
    }
    solution.capacity_units = capacity;
    if (request.budget_plane_bytes % kKvAdaptGridBytes != 0) {
        // One-sided and named: the capacity is rounded DOWN, so the plan can
        // never exceed the caller's budget; the loss is < one grid step.
        solution.budget_slack_bytes = -(request.budget_plane_bytes % kKvAdaptGridBytes);
    }

    const double kInf = std::numeric_limits<double>::infinity();
    const std::size_t span = static_cast<std::size_t>(capacity) + 1;
    std::vector<double> current(span, kInf);
    std::vector<double> next(span, kInf);
    // choice[layer][units] = index into F_layer; -1 = unreached.
    std::vector<std::vector<std::int32_t>> choice(static_cast<std::size_t>(request.layers));
    for (auto& row : choice) { row.assign(span, -1); }
    std::vector<std::int64_t> order_cur{0};
    std::vector<std::int64_t> order_next;
    order_next.reserve(span);

    current[0] = 0.0;
    for (std::int32_t layer = 0; layer < request.layers; ++layer) {
        std::fill(next.begin(), next.end(), kInf);
        order_next.clear();
        const KvAdaptLayerSet& set = detail::kv_adapt_set_for(request, layer);
        auto& row = choice[static_cast<std::size_t>(layer)];
        for (const std::int64_t from : order_cur) {
            const double base = current[static_cast<std::size_t>(from)];
            if (!(base < kInf)) { continue; }
            ++solution.states_expanded;
            for (std::size_t option_index = 0; option_index < set.options.size();
                 ++option_index) {
                const KvAdaptOption& option = set.options[option_index];
                const std::int64_t to = from + option.units();
                if (to > capacity) { continue; }
                const double value = base + option.loss;
                ++solution.transitions;
                const std::size_t at = static_cast<std::size_t>(to);
                if (next[at] == kInf) {
                    next[at] = value;
                    row[at] = static_cast<std::int32_t>(option_index);
                    order_next.push_back(to);
                } else if (value < next[at]) {
                    next[at] = value;
                    row[at] = static_cast<std::int32_t>(option_index);
                }
            }
        }
        current.swap(next);
        order_cur.swap(order_next);
        if (order_cur.empty()) {
            solution.feasible = false;
            return solution;  // F_l admits nothing inside the budget
        }
    }

    // Minimum-loss reachable final state; ASCENDING units wins an exact tie, so
    // the tie-break is "smallest byte footprint first" and is deterministic.
    std::int64_t best_units = -1;
    double best_loss = kInf;
    for (std::int64_t units = 0; units <= capacity; ++units) {
        const double value = current[static_cast<std::size_t>(units)];
        if (value < best_loss) {
            best_loss = value;
            best_units = units;
        }
    }
    if (best_units < 0) {
        solution.feasible = false;
        return solution;
    }

    // Backtrack.
    std::vector<KvAdaptLayerPick> picks(static_cast<std::size_t>(request.layers));
    std::int64_t units = best_units;
    for (std::int32_t layer = request.layers - 1; layer >= 0; --layer) {
        const auto& row = choice[static_cast<std::size_t>(layer)];
        const std::int32_t option_index = row[static_cast<std::size_t>(units)];
        if (option_index < 0) {
            throw std::logic_error("kv-adapt: broken DP backtrack at layer " +
                                   std::to_string(layer));
        }
        const KvAdaptLayerSet& set = detail::kv_adapt_set_for(request, layer);
        const KvAdaptOption& option = set.options[static_cast<std::size_t>(option_index)];
        KvAdaptLayerPick pick;
        pick.layer = layer;
        pick.option_index = option_index;
        pick.spec_name = option.spec_name;
        pick.plane_bytes = option.plane_bytes;
        pick.loss = option.loss;
        pick.measured = option.prov.usable();
        pick.evidence_arm = option.prov.arm_id;
        picks[static_cast<std::size_t>(layer)] = pick;
        units -= option.units();
    }
    if (units != 0) {
        throw std::logic_error("kv-adapt: backtrack did not consume the whole footprint");
    }

    for (const KvAdaptLayerPick& pick : picks) {
        if (!pick.measured) {
            solution.unmeasured_cells.push_back("layer=" + std::to_string(pick.layer) +
                                                " format=" + pick.spec_name);
        }
    }
    for (std::int32_t layer = 0; layer < request.layers; ++layer) {
        const KvAdaptLayerSet& set = detail::kv_adapt_set_for(request, layer);
        const bool any_measured =
            std::any_of(set.options.begin(), set.options.end(),
                        [](const KvAdaptOption& o) { return o.prov.usable(); });
        if (!any_measured) { solution.undefensible_layers.push_back(layer); }
    }

    std::int64_t achieved = 0;
    for (const KvAdaptLayerPick& pick : picks) { achieved += pick.plane_bytes; }
    solution.feasible = true;
    solution.spec = detail::kv_adapt_emit_spec(picks);
    solution.achieved_plane_bytes = achieved;
    solution.predicted_loss = best_loss;
    solution.per_layer = std::move(picks);
    solution.budget_slack_bytes += request.budget_plane_bytes - achieved;
    return solution;
}

// ---------------------------------------------------------------------------
// Is a GIVEN per-layer assignment inside this request's feasible set?
//
// This is the expressibility test that separates "the solver happened to return
// it" from "the solver COULD return it". A table with holes is feasible iff every
// layer's format is in its F_l and the byte sum fits the budget -- no prefix or
// packing argument is involved anywhere, which is precisely the property
// kv_bit_budget.h's packer lacks. Returns false with `why` filled in.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool kv_adapt_assignment_feasible(
    const KvAdaptRequest& request, const std::vector<std::string>& per_layer_names,
    std::string* why = nullptr) {
    auto fail = [&](std::string message) {
        if (why != nullptr) { *why = std::move(message); }
        return false;
    };
    if (per_layer_names.size() != static_cast<std::size_t>(request.layers)) {
        return fail("assignment names " + std::to_string(per_layer_names.size()) +
                    " layers, request has " + std::to_string(request.layers));
    }
    std::int64_t total = 0;
    for (std::int32_t layer = 0; layer < request.layers; ++layer) {
        const KvAdaptLayerSet& set = detail::kv_adapt_set_for(request, layer);
        const KvAdaptOption* option =
            set.find(per_layer_names[static_cast<std::size_t>(layer)]);
        if (option == nullptr) {
            return fail("layer " + std::to_string(layer) + " format '" +
                        per_layer_names[static_cast<std::size_t>(layer)] +
                        "' is not in that layer's allowed set F_l");
        }
        total += option->plane_bytes;
    }
    if (total > request.budget_plane_bytes) {
        return fail("assignment costs " + std::to_string(total) +
                    " plane bytes, budget is " +
                    std::to_string(request.budget_plane_bytes));
    }
    if (why != nullptr) { why->clear(); }
    return true;
}

// ---------------------------------------------------------------------------
// THE SHIPPED PRIOR, as an F_l family, so "can the allocator express the factory
// table under a given matrix" becomes a runnable question instead of an argument.
//
// Every layer gets the same six options at the shipped plane geometry, with
// loss = the shipped penalty column (kv_bit_budget.h kKvBitBudgetTiers). The
// cold pseudo-tier is appended as a seventh option priced at its REAL record
// -- READ from the header this file already includes, not re-typed:
// kKvBitBudgetColdSlotBytes (== kKvColdPoolStrideBytes, 9632 B) for an nvfp4
// layer and kKvBitBudgetColdSlotBytesInt8Raw (9232 B) for an int8 layer -- which
// is the per-layer charging kv_bit_budget.h:255-271 says its single grid point
// cannot do. This said "6688 B rANS for an nvfp4 layer" until 2026-09-18;
// 6688 is the retired 2.60-bit record and the pool does not allocate it.
//
// The provenance of every cell is left UNMEASURED on purpose: the shipped
// penalty column is a PRIOR (kv_perlayer_policy.h:51-53 proves it is byte-
// identical to the quality column and therefore that --kv-quality-weight 1 is a
// no-op), not a measurement. A plan built from this family is a prior plan and
// says so in unmeasured_cells.
// ---------------------------------------------------------------------------
[[nodiscard]] inline KvAdaptLayerSet kv_adapt_prior_option_set(
    KvAdaptColdMode cold_mode, std::int32_t cold_record_bytes = kKvBitBudgetColdSlotBytes) {
    KvAdaptLayerSet set;
    set.options.reserve(kKvBitBudgetTiers.size() + 1);
    for (const KvBitBudgetTier& tier : kKvBitBudgetTiers) {
        KvAdaptOption option;
        option.spec_name = tier.spec_name;
        option.plane_bytes = kv_bit_budget_plane_bytes(tier.bits_x100);
        option.loss = static_cast<double>(tier.penalty_x100) / 100.0;
        option.is_cold = false;
        option.prov.measured = false;
        option.prov.metric = "prior";
        option.prov.arm_id = "kKvBitBudgetTiers.penalty_x100";
        set.options.push_back(std::move(option));
    }
    if (cold_mode == KvAdaptColdMode::Eligible) {
        KvAdaptOption cold;
        cold.spec_name = "cold";
        cold.plane_bytes = cold_record_bytes;
        cold.loss = static_cast<double>(kKvBitBudgetColdPenaltyX100) / 100.0;
        cold.is_cold = true;
        cold.prov.measured = false;
        cold.prov.metric = "prior";
        cold.prov.arm_id = "kKvBitBudgetColdPenaltyX100";
        set.options.push_back(std::move(cold));
    }
    return set;
}

// Convenience: one uniform prior family applied to every layer.
[[nodiscard]] inline KvAdaptRequest kv_adapt_prior_request(
    std::int32_t layers, std::int64_t budget_plane_bytes,
    KvAdaptColdMode cold_mode = KvAdaptColdMode::Off) {
    KvAdaptRequest request;
    request.layers = layers;
    request.budget_plane_bytes = budget_plane_bytes;
    request.cold_mode = cold_mode;
    request.per_layer.push_back(kv_adapt_prior_option_set(cold_mode));
    return request;
}

// Total plane bytes of a named assignment under the shipped plane geometry.
[[nodiscard]] inline std::int64_t kv_adapt_named_footprint(
    const std::vector<std::string>& per_layer_names,
    std::vector<std::string>* unknown = nullptr) {
    std::int64_t total = 0;
    for (const std::string& name : per_layer_names) {
        if (name == "cold") {
            total += kKvBitBudgetColdSlotBytes;
            continue;
        }
        const std::int32_t tier = detail::tier_index(name);
        if (tier < 0) {
            if (unknown != nullptr) { unknown->push_back(name); }
            continue;
        }
        total += kv_bit_budget_plane_bytes(
            kKvBitBudgetTiers[static_cast<std::size_t>(tier)].bits_x100);
    }
    return total;
}

// ---------------------------------------------------------------------------
// THE BRIDGE FROM M2's SET TO THIS SOLVER'S F_l.
//
// kv_bit_budget.h's rk4v4 constraint is a COUNT, which is a lossy encoding of a SET
// that only round-trips when the set is a prefix. KvBitBudgetRk4v4Set is the
// faithful encoding; this function is where it becomes a per-layer allowed set,
// which is the only form in which the constraint is a function of the DP state
// key rather than a side condition on the path.
//
// `rk4v4.allowed[layer] == false` removes rk4v4 from THAT layer's F_l and from no
// other layer's -- the property a count cannot express, and the property the
// factory table {0,1,3,4,6,7} (holes at 2 and 5) needs.
// ---------------------------------------------------------------------------
[[nodiscard]] inline KvAdaptLayerSet kv_adapt_prior_option_set_from_rk4v4_set(
    const KvBitBudgetRk4v4Set& rk4v4, std::int32_t layer, KvAdaptColdMode cold_mode,
    std::int32_t cold_record_bytes = kKvBitBudgetColdSlotBytes) {
    KvAdaptLayerSet set = kv_adapt_prior_option_set(cold_mode, cold_record_bytes);
    if (layer >= 0 && layer < static_cast<std::int32_t>(rk4v4.allowed.size()) &&
        !rk4v4.allowed[static_cast<std::size_t>(layer)]) {
        set.options.erase(
            std::remove_if(set.options.begin(), set.options.end(),
                           [](const KvAdaptOption& o) { return o.spec_name == "rk4v4"; }),
            set.options.end());
    }
    return set;
}

// Convenience: a whole request built from an rk4v4 set + the shipped prior.
[[nodiscard]] inline KvAdaptRequest kv_adapt_request_from_rk4v4_set(
    std::int32_t layers, std::int64_t budget_plane_bytes, const KvBitBudgetRk4v4Set& rk4v4,
    KvAdaptColdMode cold_mode = KvAdaptColdMode::Off,
    std::int32_t cold_record_bytes = kKvBitBudgetColdSlotBytes) {
    KvAdaptRequest request;
    request.layers = layers;
    request.budget_plane_bytes = budget_plane_bytes;
    request.cold_mode = cold_mode;
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        request.per_layer.push_back(
            kv_adapt_prior_option_set_from_rk4v4_set(rk4v4, layer, cold_mode, cold_record_bytes));
    }
    return request;
}

// ---------------------------------------------------------------------------
// M5: THE DEPLOY ADAPTER -- and why it is a one-liner.
//
// KvAdaptSolution::spec is ALREADY in the --kv-layer-storage grammar
// (product/kv_options.h parse_kv_layer_storage_spec accepts first-last:name,
// single:name and comma-separated entries in any order), which is the same
// grammar the engine's single resolution path already consumes. So the engine
// side needs NO new consumer: a call site feeds solution.spec into whatever
// already takes the packed spec, exactly as it does for
// kv_bit_budget_solve().spec.
//
// This accessor exists so that a call site cannot accidentally reach into the
// struct and re-serialise the table by hand -- re-deriving the string from
// per_layer is how a solver's answer and a run's report drift apart.
// ---------------------------------------------------------------------------
[[nodiscard]] inline const std::string& kv_adapt_deploy_spec(
    const KvAdaptSolution& solution) noexcept {
    return solution.spec;
}

// Rows for a report: one line per layer, "layer<TAB>format<TAB>plane_bytes<TAB>loss
// <TAB>measured". Same information the report needs to stop writing a single
// global kv_dtype for a per-layer run.
[[nodiscard]] inline std::vector<std::string> kv_adapt_report_rows(
    const KvAdaptSolution& solution) {
    std::vector<std::string> rows;
    rows.reserve(solution.per_layer.size());
    for (const KvAdaptLayerPick& pick : solution.per_layer) {
        rows.push_back(std::to_string(pick.layer) + "\t" + pick.spec_name + "\t" +
                       std::to_string(pick.plane_bytes) + "\t" +
                       std::to_string(pick.loss) + "\t" +
                       (pick.measured ? "measured:" + pick.evidence_arm : "UNMEASURED"));
    }
    return rows;
}

} // namespace ninfer::product
