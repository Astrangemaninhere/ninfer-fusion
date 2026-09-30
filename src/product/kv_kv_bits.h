#pragma once

// K/V bit widths: TWO entry points, ONE penalty table entry.
//
// Host-only, std-only, no CUDA. Same convention as product/kv_bit_budget.h and
// product/kv_tier_formats.h: the CLI, the server and the planner share this one
// resolution path, and a plain host compile (and a plain host test) can exercise it.
//
// ===========================================================================
// WHAT THE USER ASKED FOR (verbatim) AND WHAT THIS FILE DOES ABOUT IT
// ===========================================================================
//   "就是 kv 分别定 bit 和合起来定 bit 的上限的机制都要有，所以这个得很复杂：
//    分开定的时候在内部分层，合起来整体定"
//   "另外惩罚表得有单独跑的入口"
//
// Both mechanisms exist here, as two entries, and they are NOT the same code path:
//
//   ENTRY 1 -- JOINT / "合起来整体定"    kv_kv_bits_entry_joint()
//     ONE overall ceiling for the whole KV stack. The one existing DP
//     (kv_bit_budget_solve / kv_bit_budget_solve_scored) picks one tier per
//     full-attention layer. This entry never refuses for anything but a ceiling no
//     tier can reach.
//
//   ENTRY 2 -- SPLIT / "分开定，内部分层"   kv_kv_bits_entry_split()
//     K gets its own ceiling and V gets its own ceiling, and EACH PLANE'S PER-LAYER
//     LAYERING IS SOLVED ON ITS OWN, in its own budget, on its own ladder (its own
//     score columns when --kv-k-tier-scores / --kv-v-tier-scores are given). Two
//     independent DP runs -> two per-layer tier vectors (k_tier[i], v_tier[i]). The
//     two vectors are then reconciled per layer against the (K format, V format)
//     pairs the engine can actually BUILD (kv_kv_bits_pair_tier). Deployed only when
//     every layer reconciles.
//
// The structural fact that makes entry 2 the hard one (and that this file REFUSES
// to paper over):
//
//   A full-attention layer carries exactly ONE DType and that single dtype drives
//   BOTH planes -- product/kv_tier_formats.h:12-22 "ONE DType per full-attention
//   layer, driving BOTH the K and the V plane" -- so EVERY tier in the budget ladder
//   costs the SAME bits per element on K and on V (product/kv_bit_budget.h:120-127:
//   ONE bits_x100 column, described there as "bits/element, K+V averaged"; the DP's
//   ledger charges two planes of that same width at product/kv_bit_budget.h:353-354).
//
//   A deployed layer whose K requirement and V requirement name DIFFERENT formats
//   can therefore only be built by a tier whose (K format, V format) pair IS the
//   requested pair. The engine has exactly TWO cross-format pairs:
//     * NVFP4 tier: K = E2M1, V = ISO4E by default (V = E2M1 under --kv-v-codec
//       e2m1) -- decoder_state.cpp kv_layer_v_dtype();
//     * Rk4v4 tier:   K = packed 4-bit E8-lattice, V = i4 (include/ninfer/types.h:36,
//       decoder_state.cpp Rk4v4Kv plane geometry, gqa_attention_decode_e8.cu).
//   Everything else a per-plane split can ask for has NO tier; this file refuses
//   that layer BY INDEX and names the missing (K, V) cell.
//
//   The cell the brief names -- (K = ISO4E, V = E2M1) -- is one of them, and building
//   it would not even change the bit width: decoder_state.cpp kv_layer_v_dtype()
//   resolves ISO4E V and E2M1 V onto the SAME plane geometry (two codes per byte, one
//   E4M3FN group scale per 16 channels), so BOTH fills of that cell cost 4.50
//   b/element. That is why "make K and V separately settable" cannot be delivered by
//   adding a switch: it needs per-layer PLANE GEOMETRY, which is the >=10-site
//   change the refusal text lists.
//
// ===========================================================================
// DEFAULT BEHAVIOUR: no blanket refusal
// ===========================================================================
// An earlier attempt at this feature threw for ANY k != v request. That is gone.
// The default path here RUNS and DEPLOYS:
//   * no K/V flag           -> the joint entry, exactly the pre-existing path;
//   * equal ceilings        -> the split entry reconciles (the two DPs are the same
//                              DP) and deploys;
//   * unequal ceilings      -> the split entry deploys every layer that reconciles;
//                              if ANY layer cannot be carried by one tier it refuses
//                              THAT LAYER by index, names the missing cell, and -- in
//                              the same run -- prints the deployable joint plan (min
//                              of the two ceilings) that honours both ceilings, so
//                              the operator has a working spec one flag away;
//   * --kv-bits-mode ceiling -> deploy min(k,v) on both planes with the unspendable
//                              headroom reported, never silent, opt-in only.
// Nothing here silently substitutes one reading for the other.
//
// A NAMED SCORE TABLE IS NEVER DROPPED, and the three ways it could be are refused by name:
//   * a table with no --kv-quality-weight (the two columns are combined ONLY by the weight,
//     kv_bit_budget.h:695, so with no weight the shipped ladder runs and the table would be
//     loaded, its provenance printed, and never read);
//   * --kv-k-tier-scores/--kv-v-tier-scores in the JOINT reading (one ceiling, one ladder,
//     so a per-plane table has nothing to fit);
//   * an explicit non-joint --kv-bits-mode with only --kv-bits (one ceiling, so there is no
//     second reading for the mode to select).
// Conversely --kv-tier-scores is the FALLBACK for a plane whose own table was not named, so
// the split reading consumes it instead of ignoring it (apps/cli/options.h:55 documents
// exactly that: "empty == the kv_tier_scores table").
//
// ===========================================================================
// ENTRY 3 -- THE PENALTY TABLE, RUNNABLE ON ITS OWN
// ===========================================================================
// --kv-score-table show|emit=<path> is an entry point of its own: it runs WITHOUT a
// model (apps/cli/main.cpp returns before any device work), writes a REAL table in
// the exact grammar the planner's loader reads (kv_bit_budget_parse_scores,
// product/kv_bit_budget.h:738-773), and the planner consumes it back through
// --kv-tier-scores. kv_scores_provenance() is the ONE place that decides what the
// numbers in that table really are, so the usage site can no longer restate the
// shipped PRIOR as "measured": the tree used to do exactly that at
// product/kv_bit_budget.h:374 -- now corrected to "a PRIOR quality column" --
// and in layouts_impl.h's [kv-score] line, while the table it prints is
// described as provisional at product/kv_bit_budget.h:718-725.

#include "core/dtype.h"
#include "ninfer/types.h"
#include "product/kv_bit_budget.h"
#include "product/kv_options.h"        // parse_kv_layer_storage_spec: the plan -> storages
#include "product/kv_storage_dtype.h"  // kv_dtype_for_storage: the ONE deploy decision point
// kvreach-e6-includes

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::product {

// ===========================================================================
// The ladder, by index (derived, never restated -- a reordered ladder is a compile
// error rather than a silently different allocation).
// ===========================================================================
inline constexpr std::size_t kKvBitsBf16Index  = 0;
inline constexpr std::size_t kKvBitsInt8Index  = 1;
inline constexpr std::size_t kKvBitsFp8Index   = 2;
inline constexpr std::size_t kKvBitsNvfp4Index = 3;
inline constexpr std::size_t kKvBitsE8Index    = 4;
inline constexpr std::size_t kKvBitsIso4eIndex  = 5;
// The rk4v4 family's narrower K planes, appended after iso4e so rows 0..5 keep the indices
// every existing derivation reads. Row 4 is the 4-bit rk4v4 and is NOT renumbered.
inline constexpr std::size_t kKvBitsE8K3Index  = 6;
inline constexpr std::size_t kKvBitsE8K2Index  = 7;
static_assert(kKvBitBudgetTiers[kKvBitsE8K3Index].bits_x100 == 375,
              "ladder row 6 must be rk3v4/3.75 (product/kv_e8_width.h)");
static_assert(kKvBitBudgetTiers[kKvBitsE8K2Index].bits_x100 == 325,
              "ladder row 7 must be rk2v4/3.25 (product/kv_e8_width.h)");
static_assert(kKvBitBudgetColdTierIndex == static_cast<std::int32_t>(kKvBitBudgetTiers.size()),
              "the cold pseudo-tier must sit immediately past the ladder: the two rk4v4 width "
              "rows took slots 6 and 7, which is where it used to be");
static_assert(kKvBitBudgetTiers[kKvBitsBf16Index].bits_x100 == 1600,
              "ladder row 0 must be bf16/16.00 (product/kv_bit_budget.h:120-127)");
static_assert(kKvBitBudgetTiers[kKvBitsInt8Index].bits_x100 == 825,
              "ladder row 1 must be int8/8.25");
static_assert(kKvBitBudgetTiers[kKvBitsFp8Index].bits_x100 == 850,
              "ladder row 2 must be fp8/8.50");
static_assert(kKvBitBudgetTiers[kKvBitsNvfp4Index].bits_x100 == 450,
              "ladder row 3 must be nvfp4/4.50");
static_assert(kKvBitBudgetTiers[kKvBitsE8Index].bits_x100 == 425,
              "ladder row 4 must be rk4v4/4.25");
static_assert(kKvBitBudgetTiers[kKvBitsIso4eIndex].bits_x100 == 450,
              "ladder row 5 must be iso4e/4.50");

// Same domain as the two CLI parsers (apps/cli/options.cpp --kv-bit-budget,
// src/serve/serve_options.cpp --kv-bit-budget), so a value that parses there cannot
// be refused here for a different reason, and vice versa.
inline constexpr double kKvBitsMinPerPlane = 0.01;
inline constexpr double kKvBitsMaxPerPlane = 16.0;  // bf16 == 16.00 b/element

[[nodiscard]] constexpr double kv_bits_tier_bits(std::size_t tier) noexcept {
    return static_cast<double>(kKvBitBudgetTiers[tier].bits_x100) / kKvBitBudgetScale;
}

// The cheapest plane the ladder can build. Derived, so it follows the ladder: a
// ceiling below this has no feasible allocation at all, and is refused with this
// number named instead of the DP's bare "no feasible allocation".
[[nodiscard]] constexpr double kv_bits_cheapest_tier() noexcept {
    double cheapest = kv_bits_tier_bits(0);
    for (std::size_t i = 1; i < kKvBitBudgetTiers.size(); ++i) {
        // Non-selectable rows are not reachable, so they must not raise the floor: the
        // whole point of this number is to name the lowest ceiling that HAS an
        // allocation, and rk2v4 at 3.25 would otherwise make a 3.5 ceiling look feasible.
        if (!kKvBitBudgetTiers[i].selectable) { continue; }
        cheapest = std::min(cheapest, kv_bits_tier_bits(i));
    }
    return cheapest;
}

// Tier names a ceiling of `bits` can still reach, cheapest first: the actionable half
// of every "no feasible allocation" refusal.
[[nodiscard]] inline std::string kv_bits_reachable_tiers(double bits) {
    std::vector<std::pair<double, std::string>> rows;
    for (const KvBitBudgetTier& tier : kKvBitBudgetTiers) {
        // Same reason as kv_bits_cheapest_tier: this list is the actionable half of a
        // refusal, and naming a tier the DP is forbidden to choose is not actionable.
        if (!tier.selectable) { continue; }
        const double cost = static_cast<double>(tier.bits_x100) / kKvBitBudgetScale;
        if (cost <= bits + 1e-9) { rows.emplace_back(cost, tier.spec_name); }
    }
    std::sort(rows.begin(), rows.end());
    std::string out;
    for (const auto& [cost, name] : rows) {
        if (!out.empty()) { out += ", "; }
        char buffer[48];
        std::snprintf(buffer, sizeof(buffer), "%s %.2f", name.c_str(), cost);
        out += buffer;
    }
    return out.empty() ? std::string("none") : out;
}

// ===========================================================================
// THE PLANE FORMATS -- the (K format, V format) pair each tier really builds
// ===========================================================================
// The K format is the tier's own codec. The V format is the tier's own codec EXCEPT
// on the two tiers the engine builds asymmetrically, and there it is read out of the
// engine's own resolution rule rather than restated:
//   * NVFP4: decoder_state.cpp kv_layer_v_dtype(DType, KvVCodec) -> ISO4E, or NVFP4
//     (= E2M1) under --kv-v-codec e2m1;
//   * Rk4v4:    include/ninfer/types.h:36 "Packed 4-bit E8-lattice K + i4 V".
// Everything else is symmetric: kv_layer_v_dtype returns the layer dtype for every
// non-NVFP4 dtype, so V == K there.
struct KvPlaneFormats {
    std::string_view k;
    std::string_view v;
};

inline constexpr std::string_view kKvFormatBf16      = "bf16";
inline constexpr std::string_view kKvFormatInt8      = "i8";
inline constexpr std::string_view kKvFormatFp8       = "fp8";
inline constexpr std::string_view kKvFormatE2M1      = "e2m1";
inline constexpr std::string_view kKvFormatE8Lattice = "e8-lattice";
inline constexpr std::string_view kKvFormatIso3      = "iso4e";
inline constexpr std::string_view kKvFormatI4        = "i4";

[[nodiscard]] inline std::string_view kv_bits_format_of_dtype(DType dtype) noexcept {
    switch (dtype) {
    case DType::BF16: return kKvFormatBf16;
    case DType::I8: return kKvFormatInt8;
    case DType::FP8_E4M3FN: return kKvFormatFp8;
    case DType::NVFP4: return kKvFormatE2M1;
    case DType::E8Kv: return kKvFormatE8Lattice;
    case DType::E8K3Kv: return kKvFormatE8Lattice;
    case DType::E8K2Kv: return kKvFormatE8Lattice;
    case DType::ISO3: return kKvFormatIso3;
    default: return "unsupported";
    }
}

// The dtype a ladder row resolves to. Mirrors the layouts_impl.h
// KvCacheStorage -> DType chain (the same six spellings), so a row that has no KV
// dtype cannot be silently treated as a tier.
[[nodiscard]] inline DType kv_bits_tier_dtype(std::size_t tier) noexcept {
    switch (tier) {
    case kKvBitsBf16Index: return DType::BF16;
    case kKvBitsInt8Index: return DType::I8;
    case kKvBitsFp8Index: return DType::FP8_E4M3FN;
    case kKvBitsNvfp4Index: return DType::NVFP4;
    case kKvBitsE8Index: return DType::E8Kv;
    case kKvBitsE8K3Index: return DType::E8K3Kv;
    case kKvBitsE8K2Index: return DType::E8K2Kv;
    default: return DType::ISO3;
    }
}

// The engine's own V-side resolution, spelled once here so the oracle and the refusal
// text cannot drift from the engine: decoder_state.cpp kv_layer_v_dtype() for the
// NVFP4 tier, and include/ninfer/types.h:36 for the Rk4v4 tier (which is a SECOND
// K/V asymmetry and NOT a symmetric tier: its V is i4, not another E8-lattice plane).
[[nodiscard]] inline std::string_view kv_bits_tier_v_format(std::size_t tier,
                                                            KvVCodec codec) noexcept {
    const DType k_dtype = kv_bits_tier_dtype(tier);
    if (k_dtype == DType::NVFP4) {
        return codec == KvVCodec::E2M1 ? kKvFormatE2M1 : kKvFormatIso3;
    }
    // "Packed 4-bit E8-lattice K + i4 V, per-64 FP16 scales" (types.h:36), confirmed by
    // the Rk4v4Kv plane geometry in decoder_state.cpp plan_cache() and by
    // gqa_attention_decode_e8.cu / gqa_attention_prefill_e8.cu. There is no DType for
    // it, so the format is named here rather than derived from a dtype.
    if (k_dtype == DType::E8Kv) { return kKvFormatI4; }
    // The two narrow widths keep the SAME V plane (i4): only the K code width moves, so
    // the V format is identical by construction rather than by coincidence.
    if (k_dtype == DType::E8K3Kv || k_dtype == DType::E8K2Kv) { return kKvFormatI4; }
    return kv_bits_format_of_dtype(k_dtype);
}

[[nodiscard]] inline KvPlaneFormats kv_tier_plane_formats(std::size_t tier,
                                                         KvVCodec codec) noexcept {
    return KvPlaneFormats{kv_bits_format_of_dtype(kv_bits_tier_dtype(tier)),
                          kv_bits_tier_v_format(tier, codec)};
}

// The realizable pairs, as one string, for the refusal text.
[[nodiscard]] inline std::string kv_bits_realizable_pairs(KvVCodec codec) {
    std::string out;
    for (std::size_t tier = 0; tier < kKvBitBudgetTiers.size(); ++tier) {
        const KvPlaneFormats pair = kv_tier_plane_formats(tier, codec);
        if (!out.empty()) { out += ", "; }
        out += "(" + std::string(pair.k) + ", " + std::string(pair.v) + ")=" +
               kKvBitBudgetTiers[tier].spec_name;
    }
    return out;
}

// ===========================================================================
// [dl/e8vaxis F1166] THE E8 FAMILY'S OWN PLANE AXIS, AND WHY IT IS NOT A LADDER ROW
// ===========================================================================
// THE DEFECT THIS ANSWERS. Every pair printed above is derived from a LADDER ROW, and the e8
// rows are KV-BOUND: `kv_bits_tier_v_format()` returns i4 for all three of them, because the
// family's ONE input was a K width. So this file could not even EXPRESS the pair the owner asked
// for -- "e8 on both planes at 2 bits" -- and the reason was not the mathematics and not the
// engine: the VOCABULARY had no width on the e8 tokens at all. `kKvFormatE8Lattice` is ONE string
// for the 4-, 3- and 2-bit K planes alike, and the pair (e8-lattice, e8-lattice) is therefore
// genuinely NOT realizable -- there is no 4-bit lattice PLANE, because the codeword has a 16-bit
// form and a 24-bit form and no 32-bit form.
//
// THE TOKENS BELOW CARRY THE WIDTH. That is the smallest edit that makes the question askable.
// They are deliberately NOT added to the ladder: the ladder's numeric floor is its own decision
// (product/kv_bit_budget.h, kKvBitBudgetE8LayerLimit, and the pins that move with it), and a
// planner oracle that invented a tier the solver cannot choose is exactly the
// "deployed=1 refused=0 over a plan the build layer refuses" defect this file's honest counter
// exists to catch.
//
// WHAT THE ORACLE ANSWERS, and it is answered from the GEOMETRY rather than from a table of
// opinions. The codec (src/ops/kv/e8_lattice_plane_codec.cuh), the byte geometry
// (src/product/kv_e8_width.h) and the WRITER (src/ops/kv_cache/append/e8_lattice_narrow_kernel.cuh)
// all carry a V plane at B3/B2 today; what does not exist is a ladder row that PRICES it and a
// READER that decodes it on a runtime path. So this predicate answers for what the family can
// BUILD, and it does not widen what the solver may CHOOSE.
inline constexpr std::string_view kKvFormatE8B3 = "e8-lattice-3";
inline constexpr std::string_view kKvFormatE8B2 = "e8-lattice-2";

// True when (k_format, v_format) is a pair the e8 family can build on the plane axis.
// `kKvFormatE8Lattice` is the 4-bit K plane and its V is i4 -- the shipped rk4v4 row, and the
// only e8 pair the LADDER carries.
[[nodiscard]] inline constexpr bool kv_e8_plane_pair_realizable(
    std::string_view k_format, std::string_view v_format) noexcept {
    // The three K-only rows the ladder already carries, restated here so that ONE predicate
    // answers for the whole family and a caller does not have to ask two questions.
    const bool k_lattice = (k_format == kKvFormatE8B3 || k_format == kKvFormatE8B2);
    if (k_format == kKvFormatE8Lattice || k_lattice) {
        if (v_format == kKvFormatI4) { return true; }   // rk4v4 / rk3v4 / rk2v4
    }
    // THE PLANE AXIS: K and V are chosen INDEPENDENTLY from the same two lattice widths.
    // (B2,B2) is the owner's floor; (B3,B3) is the same statement one width up; the two mixed
    // pairs are what "independent" means as soon as it is not an accident.
    const bool v_lattice = (v_format == kKvFormatE8B3 || v_format == kKvFormatE8B2);
    if (k_lattice && v_lattice) { return true; }
    // EVERYTHING ELSE IS A REFUSAL, and both directions are deliberate. The 4-bit K plane may
    // not take a lattice V (there is no 4-bit lattice), and a lattice K plane may not take a
    // 4-bit V. A default here would be a plan the build layer refuses one layer down, which is
    // the shape the honest counter in this same file was added to make impossible.
    return false;
}

// The realizable plane-axis pairs, as one string, for a report. DELIBERATELY a separate spelling
// from kv_bits_realizable_pairs() below: that one is the LADDER and must keep meaning the ladder,
// or a report would claim a tier is deployable on the strength of a geometry no row prices.
[[nodiscard]] inline std::string kv_e8_plane_pairs() {
    return "(e8-lattice, i4)=rk4v4, (e8-lattice-3, i4)=rk3v4, (e8-lattice-2, i4)=rk2v4, "
           "(e8-lattice-3, e8-lattice-3), (e8-lattice-2, e8-lattice-2)=THE FLOOR, "
           "(e8-lattice-3, e8-lattice-2), (e8-lattice-2, e8-lattice-3)";
}

// THE REALIZABILITY ORACLE. Returns the ladder index of a tier that builds exactly
// (k_format, v_format), or -1. This is the single place that answers "can one tier
// carry this layer's K requirement AND V requirement", so the planner, the CLI and
// the test cannot disagree about it.
//
// ⚠ IT IS THE LADDER'S ORACLE AND NOT THE FAMILY'S (marker F1166): it answers out of
// `kKvBitBudgetTiers`, so it does not know about the e8 plane axis and is right not to -- no
// ladder row prices a V-carrying pair. `kv_e8_plane_pair_realizable()` above is the e8 family's
// own question, and the two are kept apart on purpose: a caller that conflated them would
// report a geometry as a deployed tier.
[[nodiscard]] inline std::int32_t kv_kv_bits_pair_tier(std::string_view k_format,
                                                       std::string_view v_format,
                                                       KvVCodec codec) noexcept {
    for (std::size_t tier = 0; tier < kKvBitBudgetTiers.size(); ++tier) {
        const KvPlaneFormats pair = kv_tier_plane_formats(tier, codec);
        if (k_format == pair.k && v_format == pair.v) {
            return static_cast<std::int32_t>(tier);
        }
    }
    return -1;
}

// ===========================================================================
// THE REQUEST
// ===========================================================================
[[nodiscard]] inline KvBitsMode kv_bits_mode_from_name(std::string_view name) {
    if (name == "joint") { return KvBitsMode::Joint; }
    if (name == "split") { return KvBitsMode::Split; }
    if (name == "ceiling") { return KvBitsMode::Ceiling; }
    throw std::invalid_argument("invalid kv-bits-mode: " + std::string(name) +
                                " (expected joint|split|ceiling)");
}

[[nodiscard]] inline const char* kv_bits_mode_name(KvBitsMode mode) noexcept {
    switch (mode) {
    case KvBitsMode::Joint: return "joint";
    case KvBitsMode::Ceiling: return "ceiling";
    default: return "split";
    }
}

// 0.0 == "this field was not named" (both CLI parsers reject 0, so it is an
// unambiguous sentinel and EngineOptions can carry it without <optional>).
// quality_weight < 0 keeps the shipped single-penalty ladder.
struct KvBitsRequest {
    double k_bits = 0.0;
    double v_bits = 0.0;
    // The joint form's ONE overall ceiling. Exactly one of (joint_bits, k_bits/v_bits)
    // is meant to be set; naming both is refused as a contradiction.
    double joint_bits = 0.0;
    double quality_weight = -1.0;
    KvBitsMode mode = KvBitsMode::Split;
    bool mode_explicit = false;
    bool joint_explicit = false;
    // Was a score table NAMED? A named table cannot be told from the built-in default BY
    // VALUE -- an operator file is allowed to equal it -- so the front ends pass the fact
    // in. Without it the resolver cannot distinguish "the operator replaced the table" from
    // "nothing was given", and both spellings used to be dropped in silence; see the
    // refusals in kv_kv_bits_resolve.
    bool scores_explicit   = false;   // --kv-tier-scores
    bool k_scores_explicit = false;   // --kv-k-tier-scores
    bool v_scores_explicit = false;   // --kv-v-tier-scores
    // SLIDERWIRE: --kv-codec-preference, as the ladder SLOTS to try first. WHICH codec the
    // fit picks among candidates that cost the SAME bits -- "同 bit 分配不同种类的量化", not
    // fewer bits. Empty (the default) is the shipped pack order, so every existing request is
    // unchanged. It is validated against the candidate grammar at the front end and, on the
    // JOINT reading, CHECKED here to have been honoured: a preference the fit could not honour
    // is refused with the cause and the fix named, never accepted and then ignored.
    std::vector<std::int32_t> candidate_order;

    [[nodiscard]] bool k_named() const noexcept { return k_bits > 0.0; }
    [[nodiscard]] bool v_named() const noexcept { return v_bits > 0.0; }
    [[nodiscard]] bool any_plane() const noexcept { return k_named() || v_named(); }
    [[nodiscard]] bool any() const noexcept { return any_plane() || joint_bits > 0.0; }
    [[nodiscard]] bool any_scores_explicit() const noexcept {
        return scores_explicit || k_scores_explicit || v_scores_explicit;
    }
};

inline constexpr const char* kKvBitsEnvK      = "NINFER_KV_K_BITS";
inline constexpr const char* kKvBitsEnvV      = "NINFER_KV_V_BITS";
inline constexpr const char* kKvBitsEnvJoint  = "NINFER_KV_BITS";
inline constexpr const char* kKvBitsEnvMode   = "NINFER_KV_BITS_MODE";
inline constexpr const char* kKvBitsEnvWeight = "NINFER_KV_QUALITY_WEIGHT";

namespace detail {

[[nodiscard]] inline std::optional<double> kv_bits_env_double(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') { return std::nullopt; }
    const std::string text(raw);
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size()) {
        throw std::invalid_argument(std::string(name) + " is not a number: " + text);
    }
    return value;
}

} // namespace detail

// "CLI > environment > default". A field the caller already set wins; an unset field
// is filled from the environment and stays unset when the environment has nothing
// either, so a caller that passes nothing gets exactly today's defaults. Same
// contract as serve::KvAutoRelayout::from_env(const CliOverrides&)
// (kv_auto_relayout.h:110).
[[nodiscard]] inline KvBitsRequest kv_bits_request_from_env(const KvBitsRequest& cli) {
    KvBitsRequest out = cli;
    if (!out.k_named()) {
        if (const auto v = detail::kv_bits_env_double(kKvBitsEnvK)) { out.k_bits = *v; }
    }
    if (!out.v_named()) {
        if (const auto v = detail::kv_bits_env_double(kKvBitsEnvV)) { out.v_bits = *v; }
    }
    if (!(out.joint_bits > 0.0)) {
        if (const auto v = detail::kv_bits_env_double(kKvBitsEnvJoint)) { out.joint_bits = *v; }
    }
    if (!out.mode_explicit) {
        if (const char* raw = std::getenv(kKvBitsEnvMode)) {
            if (*raw != '\0') {
                out.mode          = kv_bits_mode_from_name(raw);
                out.mode_explicit = true;
            }
        }
    }
    if (!(out.quality_weight >= 0.0)) {
        if (const auto v = detail::kv_bits_env_double(kKvBitsEnvWeight)) {
            out.quality_weight = *v;
        }
    }
    return out;
}

// ===========================================================================
// PER-LAYER ROWS
// ===========================================================================
struct KvBitsLayerRow {
    std::int32_t layer = 0;
    std::int32_t k_tier = -1;   // ladder index, or kKvBitBudgetColdTierIndex for a cold layer
    std::int32_t v_tier = -1;
    double k_bits = 0.0;
    double v_bits = 0.0;
    std::string k_format;
    std::string v_format;
    bool realizable  = false;
    std::int32_t tier = -1;     // the tier that carries BOTH requirements (-1 if none)
};

// The result of either entry, plus the per-layer requirement table that makes a
// refusal actionable.
struct KvBitsPlan {
    KvBitsMode mode = KvBitsMode::Joint;
    bool deployed   = false;    // a spec was produced that the planner can use
    bool refused    = false;    // some layer's K and V requirements met no single tier
    std::string spec;           // --kv-layer-storage grammar (empty when refused)
    std::string joint_spec;     // the ONE-ceiling plan, always computed
    double achieved_bits = 0.0; // per plane, as built
    // BUDGETSAT. Filled by the JOINT entry only: the joint form has ONE ceiling, so
    // "requested" is a single number. The split form has two per-plane ceilings, so there
    // is no single requested value to subtract and both stay 0 -- an average would be a
    // number no operator asked for.
    double requested_bits = 0.0;
    double shortfall_bits = 0.0;
    double penalty       = 0.0;
    double k_achieved    = 0.0; // split mode only
    double v_achieved    = 0.0;
    std::vector<KvBitsLayerRow> rows;
    std::string refusal;        // multi-line, names every layer and its missing cell
    std::string report;         // multi-line "[kv-bits] ..." text, always non-empty
};

namespace detail {

// ===========================================================================
// THE HONEST COUNTER -- `refused` MAY NOT SAY 0 WHILE THE DEPLOY LAYER THROWS
// ===========================================================================
// WHY THIS EXISTS. `KvBitsPlan::refused` meant "some layer's K and V requirements met no
// single tier", i.e. the SPLIT reading's reconciliation. It did NOT mean "the plan cannot be
// built", so `kv_kv_bits_entry_joint` set `deployed = true` unconditionally after a
// successful DP run -- and one measured request answered `deployed=1 refused=0` with a spec
// the build layer refuses by name (dl/kdslider/logs/21_probe_v2_baseline.txt section D:
// `--kv-bits 4.50`, spec `0-2:rk2v4,3-4:rk4v4,5:rk3v4,6-7:rk2v4,8-9:int8,10-15:nvfp4`). A
// counter that can be 0 while the next layer throws is not a counter.
//
// WHAT IT IS: the produced plan is WALKED THROUGH THE DEPLOY LAYER'S OWN TWO CALLS
// (`kv_bit_budget_spec_is_deployable`, product/kv_bit_budget.h, which uses
// product/kv_options.h parse_kv_layer_storage_spec + product/kv_storage_dtype.h
// kv_dtype_for_storage) and the first layer the deploy layer refuses makes the plan REFUSED,
// carrying that refusal's own words. Nothing here restates which rows are runnable.
//
// WHERE IT SITS. Item 3 of this line makes the solver FALL BACK to a runnable plan when its
// first answer is one the build refuses (kv_bit_budget.h's gated funnel), so on the shipped
// ladder this counter is a SECOND GUARD rather than the only one. That is deliberate: the
// two failure directions are different. The funnel keeps a runnable plan reachable; the
// counter makes an UNrunnable one impossible to report as deployed -- for the cold path,
// for the split path's reconciliation, for any future row that gains or loses a reader
// asymmetrically, and for any plan that reaches this entry from outside the funnel. Both
// directions are exercised in dl/kvreach: the counter by a two-direction control on the
// predicate and by a mask-off mutation arm (which must make the joint entry report
// refused=1 while the AFTER image reports refused on none of the 932 landings), the funnel
// by the landing table over 3.05..16.05 x {absent, 0.0, 0.5, 1.0}.
[[nodiscard]] inline std::string kv_bits_undeployable_refusal(const char* entry,
                                                              std::string_view plan_spec,
                                                              std::string_view deploy_why,
                                                              std::int32_t bad_layer) {
    std::string out = "kv-kv-bits: ";
    out += entry;
    out += " produced the plan '" + std::string(plan_spec) +
           "', and THE DEPLOY LAYER REFUSES IT: this plan is NOT DEPLOYABLE. The request is "
           "therefore counted as REFUSED rather than deployed -- a plan the next layer throws "
           "on is not a plan.";
    if (bad_layer >= 0) {
        out += " The first refused layer is " + std::to_string(bad_layer) +
               " (the walk is in ascending layer order).";
    }
    out += "\n  The deploy layer's own words: ";
    out += deploy_why;
    out += "\n  The rows this build CAN build (the solver's candidate set): ";
    out += kv_bit_budget_deployable_list();
    out += "\n  The rows it cannot (withheld -- see product/kv_bit_budget.h "
           "kv_bit_budget_deployable_rows): ";
    out += kv_bit_budget_withheld_rows().empty() ? std::string("none")
                                                 : kv_bit_budget_withheld_rows();
    out += "\n  This counter reads the deploy layer's two calls "
           "(product/kv_options.h parse_kv_layer_storage_spec + product/kv_storage_dtype.h "
           "kv_dtype_for_storage) on the plan the solver itself produced, so it cannot "
           "disagree with what the build does.";
    return out;
}

// The cheapest row the solver is allowed to choose. NOT kv_bits_cheapest_tier(): that one
// reads `selectable`, which is still true for the two rows the deploy layer refuses, so it
// would name a floor no plan can reach -- and a refusal naming the wrong floor sends the
// operator to raise a ceiling that was already high enough.
[[nodiscard]] inline double kv_bits_deployable_cheapest_tier() noexcept {
    const std::array<bool, 8>& ok = kv_bit_budget_deployable_rows();
    double cheapest = 0.0;
    bool any = false;
    for (std::size_t i = 0; i < kKvBitBudgetTiers.size() && i < ok.size(); ++i) {
        if (!ok[i] || !kKvBitBudgetTiers[i].selectable) { continue; }
        const double cost = kv_bits_tier_bits(i);
        if (!any || cost < cheapest) { cheapest = cost; any = true; }
    }
    return any ? cheapest : 0.0;
}

// The same set as a "tiers reachable at X" list.
[[nodiscard]] inline std::string kv_bits_deployable_reachable_tiers(double bits) {
    const std::array<bool, 8>& ok = kv_bit_budget_deployable_rows();
    std::vector<std::pair<double, std::string>> rows;
    for (std::size_t i = 0; i < kKvBitBudgetTiers.size() && i < ok.size(); ++i) {
        if (!ok[i] || !kKvBitBudgetTiers[i].selectable) { continue; }
        const double cost = kv_bits_tier_bits(i);
        if (cost <= bits + 1e-9) { rows.emplace_back(cost, kKvBitBudgetTiers[i].spec_name); }
    }
    std::sort(rows.begin(), rows.end());
    std::string out;
    for (const auto& [cost, name] : rows) {
        if (!out.empty()) { out += ", "; }
        char buffer[48];
        std::snprintf(buffer, sizeof(buffer), "%s %.2f", name.c_str(), cost);
        out += buffer;
    }
    return out.empty() ? std::string("none") : out;
}

// Expand a DP spec ("0-7:rk4v4,8-15:nvfp4") into per-layer ladder indices. The DP's own
// kvreach-e7-counter
// grammar, so it cannot drift from what kv_bit_budget_solve_impl emits -- and it is
// the DEPLOYED assignment, because the DP's packer decides which layer gets which
// (product/kv_bit_budget.h:289-297: "the DP only counts tiers, the packing decides
// which layer gets which").
[[nodiscard]] inline std::array<std::int32_t, kKvLayerStorageSlots>
kv_bits_expand_spec(std::string_view spec, std::int32_t layers) {
    std::array<std::int32_t, kKvLayerStorageSlots> out{};
    out.fill(-1);
    std::size_t cursor = 0;
    while (cursor < spec.size()) {
        const std::size_t comma = spec.find(',', cursor);
        const std::string_view item =
            spec.substr(cursor, comma == std::string_view::npos ? spec.size() - cursor
                                                               : comma - cursor);
        cursor = comma == std::string_view::npos ? spec.size() : comma + 1;
        const std::size_t colon = item.rfind(':');
        if (colon == std::string_view::npos) {
            throw std::logic_error("kv-kv-bits: malformed allocator spec: " + std::string(item));
        }
        const std::string_view range = item.substr(0, colon);
        const std::string_view name  = item.substr(colon + 1);
        const std::int32_t tier      = pack_tier_index(name);   // product/kv_bit_budget.h
        if (tier < 0) {
            throw std::logic_error("kv-kv-bits: unknown tier '" + std::string(name) +
                                   "' in allocator spec");
        }
        const std::size_t dash = range.find('-');
        std::int32_t first = 0;
        std::int32_t last  = 0;
        try {
            first = static_cast<std::int32_t>(std::stol(std::string(
                range.substr(0, dash == std::string_view::npos ? range.size() : dash))));
            last = dash == std::string_view::npos
                       ? first
                       : static_cast<std::int32_t>(std::stol(std::string(range.substr(dash + 1))));
        } catch (const std::exception&) {
            throw std::logic_error("kv-kv-bits: malformed layer range in allocator spec: " +
                                   std::string(range));
        }
        for (std::int32_t layer = first; layer <= last; ++layer) {
            if (layer >= 0 && layer < layers &&
                static_cast<std::size_t>(layer) < out.size()) {
                out[static_cast<std::size_t>(layer)] = tier;
            }
        }
    }
    return out;
}

// A cold layer has no ladder row of its own: it is deployed as the NVFP4 hot window
// (layouts_impl.h: "cold-planned layers keep a HOT window at NVFP4"), so its plane
// formats are the NVFP4 tier's. This is where that becomes one fact.
[[nodiscard]] inline std::int32_t kv_bits_hot_window_tier(std::int32_t tier) noexcept {
    return tier == kKvBitBudgetColdTierIndex ? static_cast<std::int32_t>(kKvBitsNvfp4Index)
                                             : tier;
}

[[nodiscard]] inline double kv_bits_tier_bits_of(std::int32_t tier) noexcept {
    const std::int32_t hot = kv_bits_hot_window_tier(tier);
    if (hot < 0 || static_cast<std::size_t>(hot) >= kKvBitBudgetTiers.size()) { return 0.0; }
    return kv_bits_tier_bits(static_cast<std::size_t>(hot));
}

[[nodiscard]] inline const char* kv_bits_tier_name_of(std::int32_t tier) noexcept {
    const std::int32_t hot = kv_bits_hot_window_tier(tier);
    if (hot < 0 || static_cast<std::size_t>(hot) >= kKvBitBudgetTiers.size()) { return "?"; }
    return kKvBitBudgetTiers[static_cast<std::size_t>(hot)].spec_name;
}

// One plane's independent per-layer solve. Exactly ONE DP is called; there is no
// second solver in this file.
struct KvPlaneSolve {
    std::array<std::int32_t, kKvLayerStorageSlots> tier{};
    std::string spec;
    double achieved_bits = 0.0;
    double penalty       = 0.0;
    // BUDGETSAT: the ceiling this plane was solved against and the residual gap to it,
    // straight from the solver's own accounting (product/kv_bit_budget.h).
    double requested_bits = 0.0;
    double shortfall_bits = 0.0;
};

[[nodiscard]] inline KvPlaneSolve kv_bits_solve_plane(std::int32_t layers, double ceiling,
                                                      const KvTierScoreTable& scores,
                                                      double quality_weight,
                                                      std::int32_t rk4v4_limit,
                                                      std::int32_t cold_cap,
                                                      const std::vector<std::int32_t>&
                                                          candidate_order = {}) {
    KvPlaneSolve out;
    KvBitBudgetSolution solved;
    // THE ABSENT WEIGHT IS THE QUALITY END (product/kv_bit_budget.h kKvQualityWeightQualityEnd).
    // It used to select the shipped single-penalty ladder INSTEAD of the scored table, which is
    // the whole reason every refusal in this tree has to say "add --kv-quality-weight 0".
    // Resolving the sentinel here is not a behaviour change: both entries funnel into the same
    // kv_gear_solve(request) and differ ONLY in request.ladder, and the default table's quality
    // column IS the shipped penalty column row for row (static_assert, product/kv_bit_budget.h).
    // Measured over 104 budgets in dl/kdslider/probe/w_sweep.cpp section A: 88 differ BEFORE this
    // alignment, 0 after. ONE edit covers joint, split AND ceiling, because all three reach the
    // solver through this function.
    const double weight = kv_quality_weight_resolved(quality_weight);
    solved = kv_bit_budget_solve_scored(layers, ceiling, scores, weight, rk4v4_limit, cold_cap,
                                        candidate_order);
    out.spec           = solved.spec;
    out.achieved_bits  = solved.achieved_bits;
    out.penalty        = solved.penalty;
    out.requested_bits = solved.requested_bits;
    out.shortfall_bits = solved.shortfall_bits;
    out.tier          = kv_bits_expand_spec(solved.spec, layers);
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        if (out.tier[static_cast<std::size_t>(layer)] < 0) {
            throw std::logic_error(
                "kv-kv-bits: the allocator returned a spec that does not cover layer " +
                std::to_string(layer) + " (spec " + solved.spec + ")");
        }
    }
    return out;
}

// SLIDERWIRE: WAS THE PREFERENCE HONOURED?  A preference is not a penalty override -- it
// decides only among candidates the fit's own columns already rate EQUAL -- so it CAN
// legitimately lose, and left unchecked that loss is SILENT, which this project treats as
// worse than not having the flag. Two distinct causes, both reachable, both named here:
//   * the ladder scores the named codec STRICTLY worse. Concrete and shipped: the iso4e row
//     of kKvBitBudgetTiers carries a deliberate policy pin (penalty 200 against nvfp4's 30,
//     product/kv_bit_budget.h:259-261 "the pin is the preference, the flag is the
//     prohibition"), so `--kv-bits 4.5 --kv-codec-preference iso4e` returns nvfp4 BY DESIGN.
//     What makes the two equal -- and therefore the preference decisive -- is the fit's own
//     columns: --kv-quality-weight 0 leaves the default table's speed column, which rates
//     nvfp4 and iso4e identically at 114.
//   * the named codec is not admissible at this layer set at all (rk4v4 is confined to its
//     leading window by kKvBitBudgetE8LayerLimit), so gear_candidates_ordered dropped it
//     before the DP ran.
[[nodiscard]] inline bool kv_bits_preference_honoured(
    const KvBitsPlan& plan, const std::vector<std::int32_t>& order) {
    for (const KvBitsLayerRow& row : plan.rows) {
        const std::int32_t k = kv_bits_hot_window_tier(row.k_tier);
        const std::int32_t v = kv_bits_hot_window_tier(row.v_tier);
        for (const std::int32_t slot : order) {
            if (slot == k || slot == v) { return true; }
        }
    }
    return false;
}

[[nodiscard]] inline std::invalid_argument kv_bits_preference_unhonoured(
    const KvBitsPlan& plan, const std::vector<std::int32_t>& order, double ceiling,
    std::int32_t layers) {
    std::string named;
    for (const std::int32_t slot : order) {
        if (slot < 0) { continue; }
        if (!named.empty()) { named += ","; }
        named += kv_bits_tier_name_of(slot);
    }
    std::ostringstream message;
    message.precision(4);
    message << std::fixed;
    message << "kv-kv-bits: --kv-codec-preference " << named
            << " was given but the deployed plan at " << ceiling << " b/element over " << layers
            << " full-attention layer(s) carries none of them: it is '" << plan.spec << "'. "
               "A preference is NOT a penalty override -- it decides only among candidates the "
               "fit's own columns already rate EQUAL (product/kv_bit_budget.h "
               "KvGearSolveRequest::candidate_order) -- so it can lose, and this refusal exists "
               "so that losing is never silent. The two causes, with their fixes:\n"
               "  * THE LADDER SCORES THE NAMED CODEC STRICTLY WORSE. Shipped example: the iso4e "
               "row carries a deliberate policy pin (penalty 200 against nvfp4's 30), so "
               "`--kv-bits 4.5 --kv-codec-preference iso4e` returns nvfp4 BY DESIGN. Make the "
               "two candidates equal on the columns the fit minimises and the preference "
               "decides: add `--kv-quality-weight 0` (the default table's speed column rates "
               "nvfp4 and iso4e identically at 114, so at weight 0 they tie and the order is "
               "what breaks the tie), or `--kv-quality-weight W --kv-tier-scores <table>` with "
               "a table whose named codec row equals the incumbent's on both columns.\n"
               "  * THE NAMED CODEC IS NOT ADMISSIBLE AT THIS LAYER SET. rk4v4 is confined to its "
               "leading window (product/kv_bit_budget.h kKvBitBudgetE8LayerLimit), so at some "
               "budgets no layer may take it and the preference is dropped before the DP runs.\n"
            << "  The candidate grammar is " << kv_gear_candidate_list()
            << " (every ladder row the DP may choose, rk3v4/rk2v4 included since the gate opened: "
               "dl/e8mixwire/land/patch_flip_batch.py).";
    return std::invalid_argument(message.str());
}

// The bare "no feasible allocation" the DP throws is not actionable. Re-raise it with
// the floor, the reachable tiers and the two things that usually bind (the rk4v4 exposure
// cap and the ceiling itself) named.
[[nodiscard]] inline std::invalid_argument kv_bits_infeasible(const char* flag, double ceiling,
                                                              std::int32_t layers,
                                                              std::int32_t rk4v4_limit) {
    const double floor_bits =
        (static_cast<double>(rk4v4_limit) * static_cast<double>(kKvBitBudgetTiers[kKvBitsE8Index].bits_x100) +
         static_cast<double>(layers - rk4v4_limit) *
             static_cast<double>(kKvBitBudgetTiers[kKvBitsNvfp4Index].bits_x100)) /
        (100.0 * static_cast<double>(layers));
    std::ostringstream message;
    message.precision(2);
    message << std::fixed;
    message << "kv-kv-bits: " << flag << " " << ceiling << " b/element over " << layers
            << " full-attention layer(s) has no feasible allocation. The cheapest tier the "
               "ladder builds is "
            << kv_bits_deployable_cheapest_tier()
            << " b/element (" << kKvBitBudgetTiers[kKvBitsE8Index].spec_name
// kvreach-e10-floor
            << ", capped at " << rk4v4_limit
            << " layer(s) by the rk4v4 exposure limit product/kv_bit_budget.h:82; "
            << kKvBitBudgetTiers[kKvBitsNvfp4Index].spec_name << " is "
            << kv_bits_tier_bits(kKvBitsNvfp4Index)
            << " b/element and bounds the layers rk4v4 cannot take). The effective floor at this "
               "layer count is "
            << floor_bits << " b/element; tiers reachable at "
            << kKvBitsMaxPerPlane << ": " << kv_bits_deployable_reachable_tiers(kKvBitsMaxPerPlane)
// kvreach-e10b-reach
            << ". Raise the ceiling, or lower --max-cold-pages / free rk4v4 slots.";
    return std::invalid_argument(message.str());
}

} // namespace detail

// ===========================================================================
// THE PENALTY TABLE, AND WHERE IT REALLY COMES FROM (the Gap-D fix)
// ===========================================================================
struct KvScoresProvenance {
    bool provisional      = true;   // true <=> the shipped PRIOR table is in use
    bool quality_measured = false;
    bool speed_measured   = false;
    // [dl/backlog item9-pin] WHICH PIN THE COLUMN WAS MEASURED ON. Empty <=> NOT RECORDED,
    // and that is the honest value for the built-in table: its speed column is a real
    // measurement (86.2 / 80.7 / 40.3 / 27.1 tok/s, kv_bit_budget.h:1284-1288) taken on a
    // pin nobody wrote down, and this pin's measured uniform rates on the same formula are
    // 279.11 / 263.56 / 251.92 / 226.36 tok/s -- 3.2x apart, WITH DIFFERENT RATIOS (prior
    // nvfp4/int8 = 0.467 -> 114 here; measured = 0.903 -> 11). So `speed_measured` is true
    // of `a` run and false of the reading a consumer needs. A CONSUMER MUST READ
    // speed_measured TOGETHER WITH pin: true + empty pin is "a measurement of unknown
    // provenance" and must not be quoted as this build's.
    std::string pin;
    std::string source;
    std::string line;               // one report line, ready to print
};

// `spec` is EngineOptions::kv_tier_scores: empty = the built-in table, a string with a
// newline = an inline table, anything else = a path to a score file (exactly the three
// cases layouts_impl.h distinguishes). This is the ONE place that decides what the
// numbers are, so no caller restates it.
[[nodiscard]] inline KvScoresProvenance kv_scores_provenance(std::string_view spec) {
    KvScoresProvenance out;
    if (spec.empty()) {
        out.provisional      = true;
        // [dl/backlog item9-line] true, AND NOT THIS PIN'S. `pin` stays empty on purpose --
        // see the field's own note. Read the flag with the pin, never without it.
        out.speed_measured   = true;   // the speed column IS the measured uniform-tier rates
        out.pin              = {};     // NOT RECORDED: the rates were taken on an unnamed pin
        out.quality_measured = false;
        out.source = "the built-in table (product/kv_bit_budget.h "
            "kv_bit_budget_default_scores(); = :1277 on 2026-09-26)";
            out.line = "[kv-score] scores: " + out.source +
            " -- PROVENANCE: the quality column is the shipped PRIOR (the needle-sweep "
            "penalty column; = :31-32 on 2026-09-26), NOT a measured per-tier error; the speed "
            "column IS measured (the uniform-tier decode rates recorded in the comments "
            "inside kv_bit_budget_default_scores(); = :1282-1283, :1297-1298 on 2026-09-26, "
            "ranking restated at :277). Its own header calls the table provisional [text: "
            "\"Provisional default score table\"; = :1270 on 2026-09-26]. The old line said "
            "\"measured\" for BOTH columns; corrected there now. Use --kv-score-table emit=<path> to write the table out, or --kv-tier-scores <file> to fit against a different one.";
        return out;
    }
    if (spec.find('\n') != std::string_view::npos) {
        out.provisional = false;
        out.source      = "an inline table from the command line";
        out.line = "[kv-score] scores: " + out.source +
                   " -- PROVENANCE: the loader starts from the built-in prior table and "
                   "overwrites only the rows it sees (product/kv_bit_budget.h:738-773), so "
                   "unwritten rows keep the PRIOR. This tool cannot tell whether the written "
                   "numbers are measured or hand-written and does not claim either.";
        return out;
    }
    out.provisional = false;
    out.source      = "the score file '" + std::string(spec) + "'";
    out.line = "[kv-score] scores: " + out.source +
               " -- PROVENANCE: operator-supplied. This tool does not know whether those "
               "numbers are measured or hand-written and does not claim either; the shipped "
               "quality column is a PRIOR, so a file that copies it inherits that status.";
    return out;
}

// The table text --kv-score-table writes, in the exact grammar
// kv_bit_budget_parse_scores reads (six "<tier> <quality_x100> <speed_x100>" lines),
// with a provenance header that names each column's source. Round-tripping it back
// through the real loader is the test that the entry and the consumer agree.
[[nodiscard]] inline std::string kv_bits_score_table_text(const KvTierScoreTable& scores,
                                                          std::string_view provenance) {
    std::ostringstream out;
    out << "# ninfer KV tier score table -- written by --kv-score-table emit=<path>\n";
    out << "# consumed by --kv-tier-scores <path> (and --kv-k-tier-scores / "
           "--kv-v-tier-scores for the split entry)\n";
    out << "# grammar: '<tier> <quality_x100> <speed_x100>' per line; all eight ladder rows must "
           "be named; both columns are 'lower is better', x100\n";
    out << "# provenance: " << provenance << "\n";
    // BOUNDED BY `scores`, NOT BY `kKvBitBudgetTiers`. The two were different sizes when this
    // loop read two rows past the end of a six-row table -- on the SUCCESS path of the documented
    // no-model entry, which is why `--kv-score-table show` segfaulted (rc=139) while
    // `--kv-score-table bogus` returned rc=2. ⚠ They are equal sizes NOW (8 and 8) because the
    // gate flip grew the table, and the bound is still `scores`: `kKvBitBudgetTiers` is the
    // declared ladder and this function's subject is the table it was HANDED, which a caller may
    // have parsed from a file. Keeping the table's bound is what makes a mutated input render as
    // mutated (tests/test_kv_score_table_entry.cpp).
    for (std::size_t i = 0; i < scores.size(); ++i) {
        char line[96];
        std::snprintf(line, sizeof(line), "%-6s %5d %5d\n", kKvBitBudgetTiers[i].spec_name,
                      scores[i].quality_x100, scores[i].speed_x100);
        out << line;
    }
    return out.str();
}

// ===========================================================================
// ENTRY 3 -- the penalty table, runnable ON ITS OWN (one implementation, both
// front ends)
// ===========================================================================
// "show" writes the table to `out`; "emit=<path>" writes it to that path. Returns
// false and fills `*error` for a malformed spec or an unwritable path. This lives
// here rather than in either front end so the CLI and the server cannot disagree
// about what the entry does, and so the provenance it prints is the same sentence
// everywhere.
[[nodiscard]] inline bool kv_score_table_run(std::string_view spec,
                                             std::string_view tier_scores, std::string* error,
                                             std::ostream& out, std::ostream& info) {
    const bool is_show         = spec == "show";
    constexpr std::string_view kEmit = "emit=";
    const bool is_emit = spec.size() > kEmit.size() && spec.compare(0, kEmit.size(), kEmit) == 0;
    if (!is_show && !is_emit) {
        if (error != nullptr) {
            *error = "--kv-score-table expects 'show' or 'emit=<path>', got '" +
                     std::string(spec) + "'";
        }
        return false;
    }
    const KvTierScoreTable table         = kv_bit_budget_default_scores();
    const KvScoresProvenance provenance  = kv_scores_provenance(tier_scores);
    const std::string text = kv_bits_score_table_text(
        table, provenance.provisional
                   ? std::string("quality = the shipped PRIOR (product/kv_bit_budget.h "
                       "[text: needle-sweep penalty column; = :31-32 on 2026-09-26]), speed = the measured "
                       "uniform-tier decode rates (the comments inside kv_bit_budget_default_scores(); = :1282-1283, :1297-1298 on 2026-09-26)")
                   : provenance.source);
    if (is_show) {
        out << text;
        info << provenance.line << "\n";
        return true;
    }
    const std::string path(spec.substr(kEmit.size()));
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        if (error != nullptr) { *error = "--kv-score-table: cannot write '" + path + "'"; }
        return false;
    }
    file << text;
    file.close();
    info << "[kv-score] wrote " << table.size() << " tier rows to '" << path << "'\n"
         << provenance.line << "\n";
    return true;
}

// ===========================================================================
// ENTRY 1 -- JOINT ("合起来整体定"): ONE overall ceiling
// ===========================================================================
[[nodiscard]] inline KvBitsPlan kv_kv_bits_entry_joint(
    std::int32_t layers, double ceiling, const KvTierScoreTable& scores,
    double quality_weight, std::int32_t rk4v4_limit, std::int32_t cold_cap, KvVCodec codec,
    // SLIDERWIRE: --kv-codec-preference, as ladder slots to try FIRST. Defaulted and
    // appended, so every existing caller is unchanged and an empty list is the shipped order.
    const std::vector<std::int32_t>& candidate_order = {}) {
    if (layers <= 0) {
        throw std::invalid_argument("kv-kv-bits: layer count must be positive");
    }
    if (!(ceiling >= kKvBitsMinPerPlane) || !(ceiling <= kKvBitsMaxPerPlane)) {
        std::ostringstream message;
        message.precision(2);
        message << "kv-kv-bits: the joint ceiling must be in [" << kKvBitsMinPerPlane << ", "
                << kKvBitsMaxPerPlane << "] bits per KV element per plane, got " << ceiling
                << " (bf16 is the widest plane the engine has at "
                << kv_bits_tier_bits(kKvBitsBf16Index) << ")";
        throw std::invalid_argument(message.str());
    }
    KvBitsPlan plan;
    plan.mode = KvBitsMode::Joint;
    detail::KvPlaneSolve solved;
    try {
        solved = detail::kv_bits_solve_plane(layers, ceiling, scores, quality_weight, rk4v4_limit,
                                             cold_cap, candidate_order);
    } catch (const std::invalid_argument&) {
        throw detail::kv_bits_infeasible("--kv-bits", ceiling, layers, rk4v4_limit);
    }
    plan.spec          = solved.spec;
    plan.joint_spec    = solved.spec;
    plan.deployed      = true;
    plan.achieved_bits  = solved.achieved_bits;
    plan.penalty        = solved.penalty;
    plan.requested_bits = solved.requested_bits;
    plan.shortfall_bits = solved.shortfall_bits;
    plan.k_achieved     = solved.achieved_bits;
    plan.v_achieved    = solved.achieved_bits;
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        const std::int32_t tier = solved.tier[static_cast<std::size_t>(layer)];
        const std::int32_t hot  = detail::kv_bits_hot_window_tier(tier);
        KvBitsLayerRow row;
        row.layer      = layer;
        row.k_tier     = tier;
        row.v_tier     = tier;
        row.k_bits     = detail::kv_bits_tier_bits_of(tier);
        row.v_bits     = row.k_bits;
        const KvPlaneFormats pair =
            kv_tier_plane_formats(static_cast<std::size_t>(hot), codec);
        row.k_format   = std::string(pair.k);
        row.v_format   = std::string(pair.v);
        row.realizable = true;
        row.tier       = hot;
        plan.rows.push_back(std::move(row));
    }
    std::ostringstream report;
    report.precision(2);
    report << std::fixed;
    report << "[kv-bits] entry=joint (合起来定: ONE overall ceiling) ceiling=" << ceiling
           << " layers=" << layers << " achieved=" << solved.achieved_bits
           << " penalty=" << solved.penalty << " spec=" << solved.spec << "\n";
    report << "[kv-bits] joint realized per plane: K=" << solved.achieved_bits
           << " V=" << solved.achieved_bits << " b/element (pair "
           << (2.0 * solved.achieved_bits)
           << "); one DType drives both planes, so the two are equal by construction "
              "(product/kv_tier_formats.h:12-22)\n";
    // BUDGETSAT: the ceiling is a SATURATING constraint, so the answer can legitimately
    // land just under it -- but the gap must be stated, never inferred from the spec line.
    // A nonzero shortfall here is the ladder's step at this layer count, not the objective
    // trading bits away (iso4e is not the cause and is not needed: nvfp4 alone is exactly
    // 4.50, and a 4.50 request returns all-nvfp4).
    report << "[kv-bits] saturation: requested=" << ceiling
           << " achieved=" << solved.achieved_bits
           << " shortfall=" << solved.shortfall_bits
           << " b/element (the ceiling is filled from below: maximum bits under it, then "
              "minimum penalty; no plan may exceed it)\n";
    report << "[kv-bits] realizable (K,V) pairs at --kv-v-codec "
           << (codec == KvVCodec::E2M1 ? "e2m1" : "iso4e") << ": "
           << kv_bits_realizable_pairs(codec) << "\n";
    report << "[kv-bits] candidate rows: this build can build " << kv_bit_budget_deployable_list()
           << "; it REFUSES "
           << (kv_bit_budget_withheld_rows().empty() ? std::string("none")
                                                     : kv_bit_budget_withheld_rows())
           << " (the fit runs on the full ladder first and falls back to these rows only when "
              "its answer is a plan the build layer refuses -- product/kv_bit_budget.h "
              "detail::kv_gear_solve_gated)\n";
    plan.report = report.str();

    // ---------------------------------------------------------------------------
    // THE HONEST COUNTER (item 2). Everything above is the FIT; this is the walk through the
    // DEPLOY LAYER on the plan the fit produced. It runs BEFORE `deployed` is allowed to
    // stand, so the pair (deployed, refused) can never be (1, 0) for a plan
    // product::kv_dtype_for_storage refuses. The funnel above makes this unreachable for a
    // bare --kv-bits request on the shipped ladder -- that is the point of having both -- and
    // the counter's OTHER direction is the one the landed tests already pin: a plan that IS
    // deployable must still report refused == false, which dl/kvreach measures over every
    // ceiling in 3.05..16.05 (932 landings, 0 refused).
    // ---------------------------------------------------------------------------
    {
        std::string deploy_why;
        std::int32_t bad_layer = -1;
        if (!kv_bit_budget_spec_is_deployable(plan.spec, &deploy_why, &bad_layer)) {
            plan.refusal  = detail::kv_bits_undeployable_refusal("--kv-bits (joint)", plan.spec,
                                                                 deploy_why, bad_layer);
            plan.deployed = false;
            plan.refused  = true;
            plan.spec.clear();
        }
    }
    return plan;
}

// ===========================================================================
// Packing: reconciled rows -> the --kv-layer-storage grammar, with the DP's own
// kvreach-e8-joint
// pack order and block layout (product/kv_bit_budget.h:527-555), so a deployed plan
// is byte-identical to what the allocator emits for the same per-layer tiers.
// ===========================================================================
[[nodiscard]] inline std::string kv_bits_pack_rows(const std::vector<KvBitsLayerRow>& rows,
                                                   std::int32_t layers) {
    std::array<std::int32_t, kKvLayerStorageSlots> tiers{};
    tiers.fill(-1);
    for (const KvBitsLayerRow& row : rows) {
        if (row.layer < 0 || row.layer >= layers) { continue; }
        if (!row.realizable) { throw std::logic_error("kv-kv-bits: packing a refused plan"); }
        // A cold layer keeps the cold marker so the cold pool still receives it.
        // ⚠ [line `redkv`] WHEN THE TWO PLANES AGREE, THE TIER IS THE ONE THE DP CHOSE -- not
        // `row.tier`, which is the PAIR ORACLE's answer. `kv_tier_plane_formats` gives slot 4
        // (rk4v4), slot 6 (rk3v4) and slot 7 (rk2v4) the SAME (K=e8-lattice, V=i4) formats -- they
        // differ only in the K plate's bit width -- so `kv_kv_bits_pair_tier` collapses all three
        // rows onto slot 4 and the packed spec named rk4v4 for a layer the DP had put on rk2v4.
        // The equal-ceiling cross-check below then compared that collapsed spec against the DP's
        // own and threw a std::logic_error whose own comment says "Cannot happen while the engine
        // has one width per layer" -- measured at 4.50 over 16 layers: packed
        // `0-7:rk4v4,8-9:int8,10-15:nvfp4` against the DP's
        // `0-2:rk2v4,3-4:rk4v4,5:rk3v4,6-7:rk2v4,8-9:int8,10-15:nvfp4`. The pack must be "the one
        // that made the rows" (:913), and the rows came from ONE tier per layer whenever the two
        // per-plane solves agree; `row.tier` remains the right answer only for the genuinely
        // ASYMMETRIC case, which is the refused/unrealizable path.
        const bool planes_agree = row.k_tier == row.v_tier;
        tiers[static_cast<std::size_t>(row.layer)] =
            (row.k_tier == kKvBitBudgetColdTierIndex ||
             row.v_tier == kKvBitBudgetColdTierIndex)
                ? kKvBitBudgetColdTierIndex
                : (planes_agree ? row.k_tier : row.tier);
    }
    std::int32_t cold_count = 0;
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        if (tiers[static_cast<std::size_t>(layer)] == kKvBitBudgetColdTierIndex) {
            ++cold_count;
        }
    }
    // TWO SOLVERS, TWO PACKINGS, and the packing must be the one that made the rows.
    //
    // * A COLD plan is built by the legacy multiset solver (product/kv_bit_budget.h
    //   kv_bit_budget_solve_impl, reached whenever cold_cap > 0), which decides a COUNT per
    //   tier and lets a fixed packing order place them: rk4v4 first, then cold, then the
    //   rest. Its spec is that block order, so it is reproduced below.
    // * A COLD-FREE plan is built by the LAYER-EXACT solver (the same header's "D6" section,
    //   kv_gear_solve -> kv_gear_to_budget_solution), whose decision variable is a per-layer
    //   gear array and whose spec is a run-length emission in ASCENDING LAYER ORDER. Block
    //   order does NOT reproduce it, and re-deriving it that way RELOCATES tiers: measured
    //   with probe_packfaithful.cxx, at 8.00 b/element over 16 layers the solver returns
    //   "0-1:int8,2:rk4v4,3-15:int8" (rk4v4 on layer 2) while the block-order repack returned
    //   "0:rk4v4,1-15:int8" (rk4v4 on layer 0) -- 50 of 248 swept cells disagreed, every one
    //   of them cold-free and every cold cell agreed. Since --kv-layer-storage is a
    //   POSITIONAL grammar, the block-order string does not describe the same allocation.
    //
    // This is the divergence kv_kv_bits_entry_split guards against and the reason
    // tests/test_kv_kv_bits.cpp aborts; the guard is right, the repack was stale.
    if (cold_count == 0) {
        std::string spec;
        std::int32_t begin = 0;
        for (std::int32_t layer = 1; layer <= layers; ++layer) {
            const std::int32_t here = tiers[static_cast<std::size_t>(begin)];
            const bool flush =
                layer == layers || here != tiers[static_cast<std::size_t>(layer)];
            if (!flush) { continue; }
            if (here < 0 || static_cast<std::size_t>(here) >= kKvBitBudgetTiers.size()) {
                throw std::logic_error("kv-kv-bits: packing hit an unset tier at layer " +
                                       std::to_string(begin));
            }
            if (!spec.empty()) { spec += ","; }
            spec += std::to_string(begin);
            if (layer - 1 != begin) { spec += "-" + std::to_string(layer - 1); }
            spec += ":";
            spec += kKvBitBudgetTiers[static_cast<std::size_t>(here)].spec_name;
            begin = layer;
        }
        return spec;
    }
    std::vector<const char*> order;
    if (cold_count > 0) {
        order.assign(kKvBitBudgetPackOrderCold.begin(), kKvBitBudgetPackOrderCold.end());
    } else {
        order.assign(kKvBitBudgetPackOrder.begin(), kKvBitBudgetPackOrder.end());
    }
    std::string spec;
    std::int32_t cursor = 0;
    for (const char* name : order) {
        // The DP's canonical packing is BLOCK order (rk4v4 first, then cold, then the
        // rest), so the count per name is what places the block, not the layer
        // positions: reproduce that exactly.
        std::int32_t count = 0;
        for (std::int32_t layer = 0; layer < layers; ++layer) {
            const std::int32_t tier = tiers[static_cast<std::size_t>(layer)];
            const bool matches =
                std::string_view(name) == "cold"
                    ? tier == kKvBitBudgetColdTierIndex
                    : tier >= 0 &&
                          std::string_view(name) ==
                              kKvBitBudgetTiers[static_cast<std::size_t>(tier)].spec_name;
            if (matches) { ++count; }
        }
        if (count == 0) { continue; }
        const std::int32_t begin = cursor;
        const std::int32_t end   = cursor + count - 1;
        cursor += count;
        if (!spec.empty()) { spec += ","; }
        spec += std::to_string(begin);
        if (end != begin) { spec += "-" + std::to_string(end); }
        spec += ":";
        spec += name;
    }
    if (cursor != layers) {
        throw std::logic_error("kv-kv-bits: packing dropped layers (" + std::to_string(cursor) +
                               " of " + std::to_string(layers) + ")");
    }
    return spec;
}

// ===========================================================================
// ENTRY 2 -- SPLIT ("分开定，内部分层"): K and V each solved on their own
// ===========================================================================
[[nodiscard]] inline KvBitsPlan kv_kv_bits_entry_split(std::int32_t layers, double k_ceiling,
                                                       double v_ceiling,
                                                       const KvTierScoreTable& k_scores,
                                                       const KvTierScoreTable& v_scores,
                                                       double quality_weight,
                                                       std::int32_t rk4v4_limit,
                                                       std::int32_t cold_cap, KvVCodec codec) {
    if (layers <= 0) {
        throw std::invalid_argument("kv-kv-bits: layer count must be positive");
    }
    const auto check = [&](double bits, const char* flag) {
        if (!(bits >= kKvBitsMinPerPlane) || !(bits <= kKvBitsMaxPerPlane)) {
            std::ostringstream message;
            message.precision(2);
            message << flag << " must be in [" << kKvBitsMinPerPlane << ", "
                    << kKvBitsMaxPerPlane << "] bits per KV element, got " << bits
                    << " (bf16 is the widest plane the engine has at "
                    << kv_bits_tier_bits(kKvBitsBf16Index) << ")";
            throw std::invalid_argument(message.str());
        }
    };
    check(k_ceiling, "--kv-k-bits");
    check(v_ceiling, "--kv-v-bits");

    KvBitsPlan plan;
    plan.mode = KvBitsMode::Split;

    // --- the two INDEPENDENT per-plane solves ("内部分层") ---------------------
    // Each plane gets its own ceiling, its own score columns and its own DP run. No
    // cross-plane coupling here on purpose: coupling at this stage is exactly the
    // silent average the two-ceiling request exists to avoid.
    detail::KvPlaneSolve k_solve;
    detail::KvPlaneSolve v_solve;
    try {
        k_solve = detail::kv_bits_solve_plane(layers, k_ceiling, k_scores, quality_weight,
                                             rk4v4_limit, cold_cap);
    } catch (const std::invalid_argument&) {
        throw detail::kv_bits_infeasible("--kv-k-bits", k_ceiling, layers, rk4v4_limit);
    }
    try {
        v_solve = detail::kv_bits_solve_plane(layers, v_ceiling, v_scores, quality_weight,
                                             rk4v4_limit, cold_cap);
    } catch (const std::invalid_argument&) {
        throw detail::kv_bits_infeasible("--kv-v-bits", v_ceiling, layers, rk4v4_limit);
    }
    plan.k_achieved = k_solve.achieved_bits;
    plan.v_achieved = v_solve.achieved_bits;

    // --- per-layer reconciliation against the ENGINE's realizable pairs ---------
    std::int32_t unrealizable = 0;
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        const std::size_t index   = static_cast<std::size_t>(layer);
        const std::int32_t k_tier = k_solve.tier[index];
        const std::int32_t v_tier = v_solve.tier[index];
        const KvPlaneFormats k_pair = kv_tier_plane_formats(
            static_cast<std::size_t>(detail::kv_bits_hot_window_tier(k_tier)), codec);
        const KvPlaneFormats v_pair = kv_tier_plane_formats(
            static_cast<std::size_t>(detail::kv_bits_hot_window_tier(v_tier)), codec);
        KvBitsLayerRow row;
        row.layer  = layer;
        row.k_tier = k_tier;
        row.v_tier = v_tier;
        row.k_bits = detail::kv_bits_tier_bits_of(k_tier);
        row.v_bits = detail::kv_bits_tier_bits_of(v_tier);
        // The K requirement is the format the K plane's own solve chose for K; the V
        // requirement is the format the V plane's own solve chose for V.
        row.k_format = std::string(k_pair.k);
        row.v_format = std::string(v_pair.v);
        row.tier     = kv_kv_bits_pair_tier(row.k_format, row.v_format, codec);
        row.realizable = row.tier >= 0;
        if (!row.realizable) { ++unrealizable; }
        plan.rows.push_back(std::move(row));
    }
    plan.deployed      = unrealizable == 0;
    plan.refused       = unrealizable > 0;
    plan.achieved_bits = 0.5 * (k_solve.achieved_bits + v_solve.achieved_bits);
    plan.penalty       = 0.5 * (k_solve.penalty + v_solve.penalty);

    // The deployable ONE-ceiling plan at the tighter of the two ceilings. Always
    // computed: on success it is the cross-check, on refusal it is the plan the
    // operator can actually run (both ceilings honoured, neither exceeded).
    const double joint_ceiling = std::min(k_ceiling, v_ceiling);
    try {
        plan.joint_spec =
            detail::kv_bits_solve_plane(layers, joint_ceiling, k_scores, quality_weight,
                                        rk4v4_limit, cold_cap)
                .spec;
    } catch (const std::invalid_argument&) {
        plan.joint_spec.clear();
    }

    if (plan.deployed) {
        plan.spec = kv_bits_pack_rows(plan.rows, layers);
        // THE SAME HONEST COUNTER AS THE JOINT ENTRY (item 2, one rule both readings): the
        // packed plan is walked through the deploy layer's two calls, and a plan the build
        // refuses is REPORTED AS REFUSED rather than deployed. The reconciliation above
        // already set `deployed`, so this is the one place that can catch a reconciled plan
        // whose TIER the engine cannot read. `plan.refusal` and the report are the same shape
        // the joint entry produces, so one refusal covers both readings.
        {
            const std::string packed = plan.spec;
            std::string deploy_why;
            std::int32_t bad_layer = -1;
            if (!kv_bit_budget_spec_is_deployable(packed, &deploy_why, &bad_layer)) {
                plan.refusal  = detail::kv_bits_undeployable_refusal(
                    "--kv-k-bits/--kv-v-bits (split)", packed, deploy_why, bad_layer);
                plan.deployed = false;
                plan.refused  = true;
                plan.spec.clear();
            }
        }
        if (plan.deployed && plan.spec != plan.joint_spec) {
// kvreach-e9-split
            // Cannot happen while the engine has one width per layer, and if it ever
            // does the two readings have diverged: say so instead of choosing one.
            throw std::logic_error(
                "kv-kv-bits: the reconciled split plan differs from the joint plan at the same "
                "ceiling (" + plan.spec + " vs " + plan.joint_spec +
                "); the engine's per-layer (K,V) geometry has changed -- re-derive this "
                "entry's contract");
        }
    } else {
        std::ostringstream refusal;
        refusal.precision(2);
        refusal << std::fixed;
        refusal << "kv-kv-bits: --kv-k-bits " << k_ceiling << " --kv-v-bits " << v_ceiling
                << " cannot be carried on " << unrealizable << " of " << layers
                << " full-attention layer(s): no single tier builds the pair each layer's K and "
                   "V requirements name. Per layer (a row appears only where the requirements "
                   "diverge):\n";
        for (const KvBitsLayerRow& row : plan.rows) {
            if (row.realizable) { continue; }
            refusal << "  layer " << row.layer << ": K requires " << row.k_format << " ("
                    << row.k_bits << " b/element, K ladder chose "
                    << detail::kv_bits_tier_name_of(row.k_tier) << "), V requires "
                    << row.v_format << " (" << row.v_bits << " b/element, V ladder chose "
                    << detail::kv_bits_tier_name_of(row.v_tier)
                    << "); the missing cell is (K=" << row.k_format << ", V=" << row.v_format
                    << ") and no tier in the ladder builds it. The engine's realizable pairs "
                       "are: "
                    << kv_bits_realizable_pairs(codec) << "\n";
        }
        refusal << "  Why: a full-attention layer carries exactly ONE DType and that single "
            "dtype drives BOTH planes -- product/kv_tier_formats.h:12-22 \"ONE DType "
            "per full-attention layer, driving BOTH the K and the V plane\" -- so every "
            "tier costs the SAME bits per element on K and on V (product/kv_bit_budget.h "
            "[text: \"Bit costs are the ENGINE's plane geometry (per KV element, K+V "
            "averaged)\"; = :254 on 2026-09-26]: one bits_x100 column). A layer whose K "
            "requirement and V requirement name different formats can only be built by a "
            "tier whose (K format, V format) pair IS that pair, and the engine has exactly "
            "two such pairs; both are listed above.\n";
        refusal << "  (K=ISO4E, V=E2M1) is the cell this build cannot reach, and building it "
                   "would not even change the bit width: decoder_state.cpp kv_layer_v_dtype() "
                   "puts ISO4E V and E2M1 V on the SAME plane geometry (two codes per byte, one "
                   "E4M3FN group scale per 16 channels), so BOTH fills of that cell cost "
                << kv_bits_tier_bits(kKvBitsIso4eIndex)
                << " b/element. A real per-layer (K format, V format) table is the >=10-site "
                   "change described in the header of src/product/kv_kv_bits.h; no flag in "
                   "this build can express it.\n";
        refusal << "  Two readings that DO deploy, both honoured and neither exceeded:\n";
        refusal << "    --kv-bits-mode joint     ONE overall ceiling at " << joint_ceiling
                << " b/element (the tighter of the two; that is the \"合起来整体定\" entry). "
                   "Its plan: "
                << (plan.joint_spec.empty() ? std::string("<infeasible>") : plan.joint_spec)
                << "\n";
        refusal << "    --kv-bits-mode ceiling   both ceilings honoured on the shared width, "
                   "with the unspendable headroom reported (same plan as joint, said out "
                   "loud).\n";
        plan.refusal = refusal.str();
        plan.spec.clear();
    }

    std::ostringstream report;
    report.precision(2);
    report << std::fixed;
    report << "[kv-bits] entry=split (分开定: each plane's per-layer layering solved in its "
              "own budget) K_ceiling="
           << k_ceiling << " V_ceiling=" << v_ceiling << " layers=" << layers << "\n";
    report << "[kv-bits] K ladder (own budget, own score columns): achieved="
           << k_solve.achieved_bits << " penalty=" << k_solve.penalty
           << " plan=" << k_solve.spec << "\n";
    report << "[kv-bits] V ladder (own budget, own score columns): achieved="
           << v_solve.achieved_bits << " penalty=" << v_solve.penalty
           << " plan=" << v_solve.spec << "\n";
    report << "[kv-bits] reconciliation: (" << layers << " - " << unrealizable << ")/" << layers
           << " layer(s) carried by a single tier -> "
           << (unrealizable == 0 ? "DEPLOYED" : "REFUSED (see the refusal below)") << "\n";
    // The per-layer table: the "final per-layer allocation, before/after" artefact.
    // Printed for every run, deployed or refused.
    if (!plan.rows.empty()) {
        std::size_t start = 0;
        for (std::size_t cursor = 1; cursor <= plan.rows.size(); ++cursor) {
            const bool end = cursor == plan.rows.size();
            const bool boundary =
                end || plan.rows[cursor].k_tier != plan.rows[start].k_tier ||
                plan.rows[cursor].v_tier != plan.rows[start].v_tier;
            if (!boundary) { continue; }
            report << "[kv-bits]   layers " << start << "-" << (cursor - 1)
                   << ": K=" << plan.rows[start].k_format << " (" << plan.rows[start].k_bits
                   << " b/el, " << detail::kv_bits_tier_name_of(plan.rows[start].k_tier)
                   << ") V=" << plan.rows[start].v_format << " (" << plan.rows[start].v_bits
                   << " b/el, " << detail::kv_bits_tier_name_of(plan.rows[start].v_tier)
                   << ") tier="
                   << (plan.rows[start].realizable ? detail::kv_bits_tier_name_of(plan.rows[start].tier)
                                                   : std::string("NONE"))
                   << "\n";
            start = cursor;
        }
    }
    if (plan.deployed) {
        report << "[kv-bits] deployed spec=" << plan.spec
               << " (equal to the joint plan at " << joint_ceiling
               << " b/element: " << plan.joint_spec
               << "; with one width per layer the split and joint readings coincide here, and "
                  "that equality is CHECKED, not assumed)\n";
    }
    plan.report = report.str();
    return plan;
}

// ===========================================================================
// ONE DECISION POINT for the whole request
// ===========================================================================
// Turns the raw request into a plan. This is what the planner calls; the CLI and the
// server only validate the flag spellings.
[[nodiscard]] inline KvBitsPlan kv_kv_bits_resolve(const KvBitsRequest& request,
                                                   std::int32_t layers,
                                                   const KvTierScoreTable& scores,
                                                   const KvTierScoreTable& k_scores,
                                                   const KvTierScoreTable& v_scores,
                                                   std::int32_t rk4v4_limit, std::int32_t cold_cap,
                                                   KvVCodec codec) {
    // request.candidate_order (SLIDERWIRE, --kv-codec-preference) is read below, on the JOINT
    // reading only, and checked there to have been honoured. No extra parameter: the
    // preference travels IN the request, which is what the front ends already build.
    if (!request.any()) {
        throw std::invalid_argument(
            "kv-kv-bits: no bit budget given; expected --kv-bits (ONE overall ceiling), "
            "--kv-k-bits/--kv-v-bits (one ceiling per plane) or the pre-existing "
            "--kv-bit-budget");
    }
    // Naming both readings at once is a contradiction, not a merge: the joint form
    // already fixes the overall ceiling the split form derives its two from.
    if (request.joint_bits > 0.0 && request.any_plane()) {
        throw std::invalid_argument(
            "--kv-bits and --kv-k-bits/--kv-v-bits are two spellings of the same ceiling set: "
            "--kv-bits is the ONE overall ceiling (合起来整体定) and the two per-plane ceilings "
            "cover it (分开定). Give one or the other; --kv-bits-mode joint|split|ceiling picks "
            "the reading.");
    }
    // -----------------------------------------------------------------------------------
    // A NAMED SCORE TABLE MUST NOT BE DROPPED ON THE FLOOR. The tables have three ways to
    // become "accepted and read by nothing", which this project treats as worse than not
    // having the flag at all. They are refused HERE -- the one decision point -- rather than
    // at each front end, so the CLI, the server and any library caller get one answer.
    // -----------------------------------------------------------------------------------
    if (request.any_scores_explicit() && !(request.quality_weight >= 0.0)) {
        throw std::invalid_argument(
            "kv-kv-bits: a score table was given (--kv-tier-scores / --kv-k-tier-scores / "
            "--kv-v-tier-scores) but --kv-quality-weight was not. A table's two columns are "
            "combined only by the weight -- penalty = round(w*quality + (1-w)*speed), "
            "product/kv_bit_budget.h:695 -- so with no weight the shipped single-penalty "
            "ladder runs and the table would be loaded, its provenance printed and never "
            "read. Add --kv-quality-weight W (0 = fastest KV path, 1 = most accurate), or "
            "drop the table.");
    }
    if (request.mode_explicit && !request.any_plane() && request.mode != KvBitsMode::Joint) {
        throw std::invalid_argument(
            std::string("kv-kv-bits: --kv-bits-mode ") + kv_bits_mode_name(request.mode) +
            " was given together with --kv-bits, which names ONE overall ceiling. With a single "
            "ceiling there is no second reading for the mode to select, so it would be accepted "
            "and read by nothing. Give --kv-k-bits/--kv-v-bits (one ceiling per plane) for the "
            "split and ceiling readings, or drop --kv-bits-mode.");
    }

    const KvBitsMode mode = request.any_plane() ? request.mode : KvBitsMode::Joint;

    // SLIDERWIRE: the MIRROR IMAGE of the per-plane-table rule below, and refused for the
    // same stated reason -- a knob a reading cannot honour must not be carried into it. The
    // preference names the codec the ONE-ceiling fit should try first; the per-plane readings
    // fit each plane in its OWN budget and then RECONCILE the two per layer, so a layered
    // preference would be decided by a fit that the reconciliation can still refuse.
    if (mode != KvBitsMode::Joint && !request.candidate_order.empty()) {
        throw std::invalid_argument(
            "kv-kv-bits: --kv-codec-preference was given but this request resolves to a "
            "PER-PLANE reading (--kv-k-bits/--kv-v-bits with --kv-bits-mode split|ceiling), "
            "where each plane is fitted in its own budget and the two are then reconciled per "
            "layer, so a preference would be carried into a fit that can still refuse it -- and "
            "a knob a reading cannot honour is refused here rather than accepted and read by "
            "nothing. Give --kv-bits (the ONE overall ceiling), or drop --kv-codec-preference.");
    }

    if (mode == KvBitsMode::Joint) {
        if (request.k_scores_explicit || request.v_scores_explicit) {
            throw std::invalid_argument(
                "kv-kv-bits: --kv-k-tier-scores/--kv-v-tier-scores were given but the request "
                "resolves to the JOINT reading (ONE overall ceiling, ONE tier ladder), where a "
                "per-plane table has nothing to fit: it would be accepted and read by nothing. "
                "Give --kv-k-bits/--kv-v-bits so each plane is solved in its own budget, or "
                "drop the per-plane tables.");
        }
        double ceiling = request.joint_bits;
        if (!(ceiling > 0.0)) {
            // One-sided per-plane request read as the joint form: the plane that WAS
            // named is the overall ceiling.
            ceiling = request.k_named() ? request.k_bits
                                        : (request.v_named() ? request.v_bits : 0.0);
        }
        KvBitsPlan plan = kv_kv_bits_entry_joint(layers, ceiling, scores,
                                                 request.quality_weight, rk4v4_limit, cold_cap, codec,
                                                 request.candidate_order);
        // SLIDERWIRE: THE ONE PLACE A PREFERENCE CAN BE SILENTLY LOST. The DP is free to
        // return a plan that carries none of the named codecs -- that is what "the pin is the
        // preference, the flag is the prohibition" means -- so the miss is turned into a
        // refusal carrying the deployed plan, the cause and the flag pair that fixes it.
        if (plan.deployed && !plan.refused && !request.candidate_order.empty() &&
            !detail::kv_bits_preference_honoured(plan, request.candidate_order)) {
            throw detail::kv_bits_preference_unhonoured(plan, request.candidate_order, ceiling,
                                                        layers);
        }
        return plan;
    }

    // A one-sided request under a per-plane mode: the unnamed plane inherits the named
    // one, which is the only choice that does not invent a ceiling for it.
    double k_ceiling = request.k_named() ? request.k_bits : request.v_bits;
    double v_ceiling = request.v_named() ? request.v_bits : request.k_bits;
    // The generic table is the FALLBACK for a plane whose own table was not named: that is
    // what apps/cli/options.h:55 documents ("empty == the kv_tier_scores table") and what the
    // split reading used to silently ignore. A NAMED per-plane table always wins over it.
    const KvTierScoreTable& k_used = request.k_scores_explicit ? k_scores : scores;
    const KvTierScoreTable& v_used = request.v_scores_explicit ? v_scores : scores;
    KvBitsPlan plan = kv_kv_bits_entry_split(layers, k_ceiling, v_ceiling, k_used, v_used,
                                            request.quality_weight, rk4v4_limit, cold_cap, codec);
    if (mode == KvBitsMode::Ceiling && plan.refused) {
        // Deliberately deployable, opt-in by name. Both declared ceilings are honoured
        // (neither is exceeded) because the built plan spends min(k,v) on BOTH planes;
        // the part that could not be spent is reported instead of discarded. This is
        // NOT the default.
        // The deployed plan is a SINGLE ladder at min(k,v), so it is fitted with the K
        // plane's effective table: that is the one the deployed layers came from, and when
        // only the generic table was named k_used IS that table.
        KvBitsPlan out = kv_kv_bits_entry_joint(
            layers, std::min(k_ceiling, v_ceiling), k_used, request.quality_weight, rk4v4_limit,
            cold_cap, codec);
        std::ostringstream report;
        report << plan.report;
        report.precision(2);
        report << std::fixed;
        report << "[kv-bits] mode=ceiling: the split reading is undeliverable (above), so the "
                  "deployed plan honours BOTH ceilings on the shared width and spends "
               << std::min(k_ceiling, v_ceiling) << " b/element per plane. The "
               << (k_ceiling < v_ceiling ? "V" : "K") << " ceiling's extra "
               << std::abs(k_ceiling - v_ceiling)
               << " b/element is UNSPENDABLE here: it would need K and V on different plane "
                  "geometries, which this build has no mechanism for. Neither ceiling is "
                  "exceeded.\n";
        report << out.report;
        out.report = report.str();
        out.mode   = KvBitsMode::Ceiling;
        return out;
    }
    return plan;
}

} // namespace ninfer::product
