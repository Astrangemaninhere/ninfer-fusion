#pragma once

// Fractional KV bit-budget allocator (C++ port of tools/archkit/kv_bit_budget.py).
//
// Given a target average bit budget per KV element, pick one KV tier per full-attention
// layer so that the total bit cost stays inside the budget while the total quality
// penalty is minimised. Solved exactly with a small DP over (bits used, e8 layers used,
// cold layers used); the resulting per-layer assignment is packed into the existing
// --kv-layer-storage grammar, so the rest of the engine keeps its single resolution path.
//
// Tier costs (bits/element, K+V averaged) and penalties (needle-sweep prior, lower is
// better) mirror the Python tool: bf16 16.00/0.00, int8 8.25/0.02, fp8 8.03/0.03,
// nvfp4 4.50/0.30, e8 4.06/0.08, iso3 3.00/0.50.
//
// e8 is only allowed on the leading layers: the e8 tier is verified for layers 0..7 while
// the shipped high-layer e8 path degrades (_TODO.md 96/116); iso3/nvfp4 take over above.
//
// Opt-in cold tier (mirrors the Python tool's --cold-pages/--cold-bytes; engine knobs:
// --cold-policy none/window/host/disk + max_cold_pages): with a cold capacity in pages,
// a layer may instead live in the cold pool. A cold layer costs 0.00 hot bits/element
// (its GPU bit-budget footprint is freed), occupies ONE cold page of capacity
// (kKvBitBudgetColdSlotBytes), and carries a 0.25 restore-penalty prior (cold slots hold
// int8-grade packed data). At most cold_cap layers may be cold; cold_used is an exact DP
// dimension. Packing keeps e8 FIRST (the e8 limit is a leading-layer constraint, e8 must
// keep slots 0..count-1 inside its verified window), then places cold on the
// next-shallow block; deep layers keep the hot high-precision tail (_TODO.md 46/47).
// "cold" is NOT a --kv-layer-storage tier (grammar: bf16/int8/fp8/nvfp4/iso3/e8):
// cold-planned layers are deployed via --cold-policy host|disk --max-cold-pages N.
// With cold_cap == 0 the allocation is identical to the pre-cold header (grid-diff
// proof: _collab/A_kvbit_cold_cpp.md).
//
// kKvBitBudgetColdSlotBytes mirrors ops::kEntropyNvfp4SlotBytes, defined at
// include/ninfer/ops/entropy_nvfp4_slot.h:14 (used at
// src/targets/qwen3_6/impl/state/decoder_state.cpp:179); the value is spelled out here
// so this header stays host-only (that header includes cuda_runtime.h).

#include "ninfer/types.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::product {

inline constexpr std::int32_t kKvBitBudgetE8LayerLimit = 8;
// 0.01-bit resolution, matching the Python tool.
inline constexpr std::int32_t kKvBitBudgetScale = 100;

struct KvBitBudgetTier {
    const char* spec_name;
    std::int32_t bits_x100;
    std::int32_t penalty_x100;
};

// Cost ladder (cheapest first is not needed; the DP explores all of them).
inline constexpr std::array<KvBitBudgetTier, 6> kKvBitBudgetTiers{{
    {"bf16", 1600, 0},
    {"int8", 825, 2},
    {"fp8", 803, 3},
    {"nvfp4", 450, 30},
    {"e8", 406, 8},
    {"iso3", 300, 50},
}};

// Cold pseudo-tier. It deliberately lives OUTSIDE kKvBitBudgetTiers (index 6) so engine
// consumers of the hot tier table are untouched.
inline constexpr std::int32_t kKvBitBudgetColdTierIndex = 6;
inline constexpr std::int32_t kKvBitBudgetColdBitsX100 = 0;      // frees the hot budget
inline constexpr std::int32_t kKvBitBudgetColdPenaltyX100 = 25;  // 0.25 restore prior
inline constexpr std::int32_t kKvBitBudgetColdSlotBytes = 9536;  // ops::kEntropyNvfp4SlotBytes

// Packing order: e8 first so the verified leading layers receive it (the DP only counts
// tiers, the packing decides which layer gets which).
inline constexpr std::array<const char*, 6> kKvBitBudgetPackOrder{
    {"e8", "bf16", "int8", "fp8", "nvfp4", "iso3"}};

// Cold-mode packing: e8 STILL first (keeps its verified leading window), then cold on
// the next-shallow block; deep layers keep the hot tail.
inline constexpr std::array<const char*, 7> kKvBitBudgetPackOrderCold{
    {"e8", "cold", "bf16", "int8", "fp8", "nvfp4", "iso3"}};

namespace detail {

[[nodiscard]] inline std::int32_t tier_index(std::string_view name) noexcept {
    for (std::size_t i = 0; i < kKvBitBudgetTiers.size(); ++i) {
        if (name == kKvBitBudgetTiers[i].spec_name) { return static_cast<std::int32_t>(i); }
    }
    return -1;
}

[[nodiscard]] inline std::int32_t pack_index(std::string_view name) noexcept {
    for (std::size_t i = 0; i < kKvBitBudgetPackOrder.size(); ++i) {
        if (name == kKvBitBudgetPackOrder[i]) { return static_cast<std::int32_t>(i); }
    }
    return -1;
}

// Index inside kKvBitBudgetTiers for a pack-order name ("cold" -> the cold pseudo-tier).
[[nodiscard]] inline std::int32_t pack_tier_index(std::string_view name) noexcept {
    if (name == "cold") { return kKvBitBudgetColdTierIndex; }
    return tier_index(name);
}

} // namespace detail

struct KvBitBudgetSolution {
    std::string spec;            // --kv-layer-storage grammar ("cold" marks cold-planned slots)
    double achieved_bits = 0.0;  // hot bits/element, cold layers counted at 0
    double penalty = 0.0;        // total quality penalty (exact x100 sum / 100)
    std::map<std::string, std::int32_t> counts;  // tier -> layer count, may hold "cold"
};

// Full solution. Structurally IDENTICAL to the Python DP (tools/archkit/kv_bit_budget.py)
// so the two agree byte-for-byte, including tie-breaks:
//   * state key = (bits, cold_used); the e8 count is carried in the surviving path's
//     value (NOT an extra dimension), exactly like the Python counts dict;
//   * states are expanded in first-insertion order and candidates in the ladder order
//     ORDER + ["cold"]; an equal-penalty candidate never displaces an inserted one;
//   * the final winner is the minimum-penalty state in first-insertion order.
// cold_cap > 0 enables the cold pseudo-tier; cold_cap == 0 keeps the candidate set on
// the pre-cold code path (cold_states == 1 makes the cold index inert).
[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve(
    std::int32_t layers, double budget_bits,
    std::int32_t e8_limit = kKvBitBudgetE8LayerLimit, std::int32_t cold_cap = 0) {
    if (layers <= 0) { throw std::invalid_argument("kv-bit-budget: layer count must be positive"); }
    if (!(budget_bits > 0.0)) {
        throw std::invalid_argument("kv-bit-budget: bit budget must be positive");
    }
    if (e8_limit < 0) { e8_limit = 0; }
    if (e8_limit > layers) { e8_limit = layers; }
    if (cold_cap < 0) { cold_cap = 0; }
    if (cold_cap > layers) { cold_cap = layers; }

    // Rounding must reproduce Python's round() = banker's (round-half-to-EVEN), found
    // the hard way by C's S18 adversarial grid (_collab/C_s14_verify.md): budgets that
    // are k/8 with odd k (4.125, 5.625, 6.625, ...) make budget*100 land on a
    // binary-exact .5 (4.125*100 == 412.5 exactly), where the previous half-up
    // `(int)(x + 0.5)` gave 413 while Python round(412.5) = 412 — a capacity shift of
    // `layers` 0.01-bit units and 8 divergent allocations at cold >= 8, layers >= 48.
    // std::nearbyint honours the default FE_TONEAREST mode = ties-to-even (matches
    // Python on all spot values: 412.5->412, 413.5->414, 562.5->562).
    const std::int32_t budget_x100 = static_cast<std::int32_t>(
        std::nearbyint(budget_bits * kKvBitBudgetScale));
    const std::int32_t capacity = budget_x100 * layers;   // total 0.01-bit units available
    const std::int32_t cold_states = cold_cap + 1;        // 1 when cold is off
    const std::int32_t span = capacity + 1;
    const std::size_t states = static_cast<std::size_t>(span) * static_cast<std::size_t>(cold_states);

    const double kInf = 1.0e18;
    std::vector<double> current(states, kInf);
    std::vector<double> next(states, kInf);
    // Surviving path's e8 usage per state (the Python tool carries it inside `counts`).
    std::vector<std::int8_t> e8_count(states, 0);
    std::vector<std::int8_t> next_e8_count(states, 0);
    // choice[layer][state] = tier chosen for the (layer+1)-th layer when the state was
    // reached (kKvBitBudgetColdTierIndex marks the cold pseudo-tier).
    std::vector<std::vector<std::int8_t>> choice(static_cast<std::size_t>(layers));
    for (auto& layer_choice : choice) { layer_choice.assign(states, -1); }
    // States in first-insertion order (mirrors Python dict iteration order).
    std::vector<std::size_t> order_cur{0};
    std::vector<std::size_t> order_next;
    order_next.reserve(states);

    current[0] = 0.0;
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        std::fill(next.begin(), next.end(), kInf);
        std::fill(next_e8_count.begin(), next_e8_count.end(), 0);
        order_next.clear();
        auto& layer_choice = choice[static_cast<std::size_t>(layer)];
        for (const std::size_t from : order_cur) {
            const double base = current[from];
            if (base >= kInf) { continue; }
            const std::int32_t from_e8 = e8_count[from];
            const std::int32_t from_bits = static_cast<std::int32_t>(from / cold_states);
            const std::int32_t from_cold = static_cast<std::int32_t>(from % cold_states);
            for (std::size_t t = 0; t < kKvBitBudgetTiers.size(); ++t) {
                const KvBitBudgetTier& tier = kKvBitBudgetTiers[t];
                const bool is_e8 = static_cast<std::int32_t>(t) == 4;  // ladder slot of "e8"
                if (is_e8 && from_e8 >= e8_limit) { continue; }
                const std::int32_t new_bits = from_bits + tier.bits_x100;
                if (new_bits > capacity) { continue; }
                const std::size_t to = static_cast<std::size_t>(new_bits) * cold_states + from_cold;
                // Accumulate in the Python float domain (x100/100.0 == the Python literal),
                // in path order: the tool's comparisons and printed penalties are float
                // sums, and near-ties resolve identically only if the arithmetic matches.
                const double value = base + tier.penalty_x100 / 100.0;
                if (next[to] >= kInf) {
                    next[to] = value;
                    next_e8_count[to] = static_cast<std::int8_t>(from_e8 + (is_e8 ? 1 : 0));
                    layer_choice[to] = static_cast<std::int8_t>(t);
                    order_next.push_back(to);
                } else if (value < next[to]) {
                    next[to] = value;
                    next_e8_count[to] = static_cast<std::int8_t>(from_e8 + (is_e8 ? 1 : 0));
                    layer_choice[to] = static_cast<std::int8_t>(t);
                }
            }
            if (cold_cap > 0 && from_cold < cold_cap) {
                // Cold pseudo-tier (candidate LAST, after the ladder, like the Python tool):
                // 0 hot bits, fixed restore penalty.
                const std::size_t to = static_cast<std::size_t>(from_bits) * cold_states +
                                       (from_cold + 1);
                const double value = base + kKvBitBudgetColdPenaltyX100 / 100.0;
                if (next[to] >= kInf) {
                    next[to] = value;
                    next_e8_count[to] = static_cast<std::int8_t>(from_e8);
                    layer_choice[to] = static_cast<std::int8_t>(kKvBitBudgetColdTierIndex);
                    order_next.push_back(to);
                } else if (value < next[to]) {
                    next[to] = value;
                    next_e8_count[to] = static_cast<std::int8_t>(from_e8);
                    layer_choice[to] = static_cast<std::int8_t>(kKvBitBudgetColdTierIndex);
                }
            }
        }
        current.swap(next);
        e8_count.swap(next_e8_count);
        order_cur.swap(order_next);
    }

    // Pick the cheapest reachable final state, first-inserted wins on ties (Python scan).
    std::size_t best = states;
    double best_penalty = kInf;
    for (const std::size_t state : order_cur) {
        if (current[state] < best_penalty) {
            best_penalty = current[state];
            best = state;
        }
    }
    if (best == states) {
        throw std::invalid_argument("kv-bit-budget: no feasible allocation for this budget");
    }
    const std::int32_t best_bits = static_cast<std::int32_t>(best / cold_states);
    const std::int32_t best_cold = static_cast<std::int32_t>(best % cold_states);

    // Backtrack to per-layer tier indices (e8 is not part of the state walk; the guard
    // was applied forward, mirroring the Python counts-carrying value).
    std::vector<std::int32_t> per_layer(static_cast<std::size_t>(layers), -1);
    std::int32_t bits = best_bits;
    std::int32_t cold_used = best_cold;
    for (std::int32_t layer = layers - 1; layer >= 0; --layer) {
        const auto& layer_choice = choice[static_cast<std::size_t>(layer)];
        const std::size_t at = static_cast<std::size_t>(bits) * cold_states + cold_used;
        const std::int32_t tier = layer_choice[at];
        if (tier < 0) { throw std::logic_error("kv-bit-budget: broken DP backtrack"); }
        per_layer[static_cast<std::size_t>(layer)] = tier;
        if (tier == kKvBitBudgetColdTierIndex) {
            bits -= kKvBitBudgetColdBitsX100;
            --cold_used;
        } else {
            bits -= kKvBitBudgetTiers[static_cast<std::size_t>(tier)].bits_x100;
        }
    }

    // Counts per tier name (hot names + "cold").
    std::map<std::string, std::int32_t> counts;
    std::int32_t cold_count = 0;
    for (const std::int32_t tier : per_layer) {
        if (tier == kKvBitBudgetColdTierIndex) {
            ++counts["cold"];
            ++cold_count;
        } else {
            ++counts[kKvBitBudgetTiers[static_cast<std::size_t>(tier)].spec_name];
        }
    }

    // Pack: e8 first (cold mode: e8, then cold, then the rest), then emit layer ranges.
    std::vector<const char*> pack_order;
    if (cold_count > 0) {
        pack_order.assign(kKvBitBudgetPackOrderCold.begin(), kKvBitBudgetPackOrderCold.end());
    } else {
        pack_order.assign(kKvBitBudgetPackOrder.begin(), kKvBitBudgetPackOrder.end());
    }
    std::string spec;
    std::int32_t cursor = 0;
    for (const char* pack_name : pack_order) {
        const std::int32_t tier = detail::pack_tier_index(pack_name);
        if (tier < 0) { continue; }
        std::int32_t count = 0;
        if (tier == kKvBitBudgetColdTierIndex) {
            count = cold_count;
        } else {
            const auto it = counts.find(pack_name);
            count = it == counts.end() ? 0 : it->second;
        }
        if (count == 0) { continue; }
        const std::int32_t begin = cursor;
        const std::int32_t end = cursor + count - 1;
        cursor += count;
        if (!spec.empty()) { spec += ","; }
        spec += std::to_string(begin);
        if (end != begin) { spec += "-" + std::to_string(end); }
        spec += ":";
        spec += pack_name;
    }

    KvBitBudgetSolution solution;
    solution.spec = std::move(spec);
    solution.achieved_bits = static_cast<double>(best_bits) / (layers * kKvBitBudgetScale);
    solution.penalty = best_penalty;  // already the Python float-domain accumulated sum
    solution.counts = std::move(counts);
    return solution;
}

// Returns the --kv-layer-storage spec covering `layers` full-attention layers (thin
// wrapper; signature-compatible with the pre-cold header, cold_cap defaults to off).
[[nodiscard]] inline std::string kv_bit_budget_spec(std::int32_t layers, double budget_bits,
                                                    std::int32_t e8_limit =
                                                        kKvBitBudgetE8LayerLimit,
                                                    std::int32_t cold_cap = 0) {
    return kv_bit_budget_solve(layers, budget_bits, e8_limit, cold_cap).spec;
}

} // namespace ninfer::product
