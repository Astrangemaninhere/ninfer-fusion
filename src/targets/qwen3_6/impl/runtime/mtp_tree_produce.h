#pragma once

// ============================================================================
// The PRODUCING half of the MTP draft-tree publication (--draft-tree L,d, L > 1).
//
// WHY THIS FILE EXISTS
//   The receiving half (mtp_tree_publish.h) is the one place a tree layout can enter the
//   runtime; the verify side refuses an L > 1 round whose tree was never published
//   (program_impl.h:13335-13342 @3944a53). This header is the other half: it turns the
//   proposal side's per-depth candidate lattice into that layout.
//
//   It is HOST ONLY, engine free, standard library only -- the same discipline as
//   dflash2_ddtree.h and mtp_tree_publish.h -- so a host test pins the whole tree
//   arithmetic with no device, no artifact and no variant macros.
//
// THE TWO LATTICES, AND WHY THERE ARE TWO
//
//   (1) FULL LATTICE (mtp_tree_columns_from_lattice): the proposal side publishes, per
//       draft depth s, the top_k candidate ids AND the top_k x top_k predecessor x
//       successor score grid of ddtree::build_tree (include/ninfer/ops/dflash2_ddtree.h:
//       31-42, 175-297). This is the shape a beam-L draft head has to produce with L real
//       forwards per depth, because row p is the distribution conditioned on candidate
//       RANK p at depth s-1 and a rank's row cannot be inferred from another rank's line.
//       This entry point runs the REAL reference builder, so the layout is bit-for-bit what
//       ddtree::column_depth() / column_ancestors() would report for a real beam-L tree.
//
//   (2) SIBLING LATTICE (mtp_tree_columns_from_siblings): the proposal side publishes only
//       the top_paths candidate ids per depth -- exactly what the existing per-depth
//       proposal forward already produces, with a top-paths extraction instead of an
//       argmax, and NO extra forward. The layout it defines is:
//
//           level s in [0, depth):   top_paths SIBLING nodes, all at depth s + 1,
//                                    all children of the level s-1 spine node
//           the sibing i == 0 is the spine: it continues to level s + 1
//           the siblings i > 0 are LEAVES
//
//       so the node count is exactly paths * depth == the node budget, the verify columns
//       are exactly the frame width (node_budget + 1), and column 1 + s * paths + i carries
//       node (s, i). Every token in it is a real proposal-head output of a real forward --
//       the only thing given up is that the non-spine siblings are not expanded. That is a
//       COVERAGE reduction, declared here and in the report, NOT a substitution: nothing in
//       this file invents or copies a score, and a chain can never come out of it for
//       L >= 2 because level 0 alone already has `paths` >= 2 siblings.
//
//   Why (2) exists at all: (1) needs L real forwards per depth with per-beam KV and
//   per-beam state slots, which the MTP draft loop does not have (it runs one token per
//   depth through a single per-lane KV table row -- program_impl.h:13346-13366's ingress
//   carries one mtp_kv_table_row and one state slot pair per LANE, not per beam). (2) needs
//   no new forward, no new KV row and no new state slot, so it is landable with the
//   existing proposal loop while (1) is the target the seam is already compatible with.
//
// WHAT THIS FILE DOES NOT DO
//   It does not relax the verify-side guard, and it does not read or write SequenceState
//   directly: it produces a MtpTreeColumns and hands it to publish_mtp_tree() (the
//   receiving half), which validates it against the same contract the verify side reads.
// ============================================================================

#include "targets/qwen3_6/impl/runtime/mtp_tree_publish.h"

#include <ninfer/ops/dflash2_ddtree.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// The candidate count of one depth's proposal row that the MTP proposal head must expose.
// = ops::ddtree::kTopK; named here so the lattice contract and the builder's row width
// cannot drift apart silently.
inline constexpr std::int32_t kMtpLatticeTopK = ops::ddtree::kTopK;

// The dense lattice of entry point (1): `scores` is ddtree::build_tree's own input layout,
// scores[s][p][c] at flat offset s + steps * (p + top_k * c) (ddtree score_offset()).
// `candidates` is [steps * top_k]: the GLOBAL token id of candidate rank c at depth s.
struct MtpProposalLattice {
    std::int32_t steps  = 0;
    std::int32_t top_k  = 0;
    std::vector<std::int32_t> candidates;
    std::vector<double> scores;
};

// The sparse lattice of entry point (2): per depth s, the `paths` best candidate token ids
// of the depth-s proposal row, best first, at ids[s * paths + i]. No scores: the shape is
// fixed by (paths, steps) and the ids alone carry every token, so nothing here depends on a
// score the producer would have had to interpret.
struct MtpSiblingLattice {
    std::int32_t steps = 0; // --draft-tree's d
    std::int32_t paths = 0; // --draft-tree's L
    std::vector<TokenId> ids;
    std::uint32_t ids_per_depth() const noexcept { return static_cast<std::uint32_t>(paths); }
    std::uint32_t nodes() const noexcept {
        return paths <= 0 ? 0 : static_cast<std::uint32_t>(paths * steps);
    }
};

// "" when the dense lattice is well formed, else the defect. `node_budget` is
// L*d == the MTP draft window (layouts_impl.h:1631-1636 @3944a53), which is also the
// frame's live column budget.
inline std::string mtp_lattice_defect(const MtpProposalLattice& lattice,
                                      std::uint32_t node_budget) {
    if (lattice.top_k != kMtpLatticeTopK) {
        return "the lattice row width is not the ddtree top_k";
    }
    if (lattice.steps < 1 || lattice.steps > 15) {
        return "the lattice step count is outside the registered ddtree geometry (1..15)";
    }
    const std::size_t want_c = static_cast<std::size_t>(lattice.steps) *
                               static_cast<std::size_t>(lattice.top_k);
    if (lattice.candidates.size() != want_c) {
        return "the candidate table is not steps * top_k wide";
    }
    const std::size_t want_s = want_c * static_cast<std::size_t>(lattice.top_k);
    if (lattice.scores.size() != want_s) {
        return "the score grid is not steps * top_k * top_k wide";
    }
    if (node_budget < static_cast<std::uint32_t>(lattice.steps) ||
        node_budget > static_cast<std::uint32_t>(ops::ddtree::kMaxNodes)) {
        return "the node budget cannot hold one chain inside the 64-bit ancestor mask";
    }
    return std::string();
}

// "" when the sibling lattice is well formed, else the defect.
inline std::string mtp_sibling_lattice_defect(const MtpSiblingLattice& lattice,
                                              std::uint32_t node_budget) {
    // A single path IS the chain spelling (depths 1..d, prefix masks), so the producer refuses
    // to emit one. That is this producer's own invariant, not a change to the receiving half's
    // rule: the seam keeps its n >= 3 chain check exactly as it shipped.
    if (lattice.paths < 2) {
        return "a single path is the chain spelling, not a tree: this producer needs L >= 2";
    }
    if (lattice.steps < 1) {
        return "the sibling lattice has no steps";
    }
    if (lattice.paths > kMtpLatticeTopK) {
        return "the sibling lattice asks for more paths than the proposal row can rank";
    }
    if (lattice.steps > 15) {
        return "the sibling lattice step count is outside the registered ddtree geometry";
    }
    const std::size_t want = static_cast<std::size_t>(lattice.paths) *
                             static_cast<std::size_t>(lattice.steps);
    if (lattice.ids.size() != want) {
        return "the sibling id table is not paths * steps wide";
    }
    if (lattice.nodes() != node_budget) {
        return "the sibling lattice node count is not the round's node budget";
    }
    if (node_budget > static_cast<std::uint32_t>(ops::ddtree::kMaxNodes)) {
        return "the sibling lattice needs more columns than the 64-bit ancestor mask can index";
    }
    return std::string();
}

// Entry point (1). Runs the REAL reference builder over the published grid and transcribes
// its layout. The node token of node i is the candidate the builder kept at that node's own
// (depth, rank): candidates[node.depth * top_k + node.rank].
inline MtpTreeColumns mtp_tree_columns_from_lattice(const MtpProposalLattice& lattice,
                                                    std::uint32_t node_budget) {
    const std::string defect = mtp_lattice_defect(lattice, node_budget);
    if (!defect.empty()) {
        throw std::invalid_argument("MTP proposal lattice rejected: " + defect);
    }
    const ops::ddtree::Tree tree =
        ops::ddtree::build_tree(lattice.scores.data(), lattice.steps,
                                static_cast<std::int32_t>(node_budget));
    std::vector<TokenId> tokens;
    tokens.reserve(static_cast<std::size_t>(tree.node_count()));
    for (const ops::ddtree::Node& node : tree.nodes) {
        if (node.depth < 0 || node.rank < 0 || node.rank >= lattice.top_k) {
            throw std::logic_error("MTP proposal lattice produced an out-of-table node");
        }
        const std::size_t at = static_cast<std::size_t>(node.depth) *
                                   static_cast<std::size_t>(lattice.top_k) +
                               static_cast<std::size_t>(node.rank);
        tokens.push_back(static_cast<TokenId>(lattice.candidates[at]));
    }
    if (tree.node_count() == 0 ||
        static_cast<std::uint32_t>(tree.columns()) > node_budget + 1U) {
        throw std::logic_error("MTP proposal lattice built a tree outside the round's frame");
    }
    return mtp_tree_columns_from_ddtree(tree, std::span<const TokenId>(tokens));
}

// Entry point (2). The sibling-shape layout, computed by the rule stated at the top of this
// file. It is derived from nothing but (paths, steps) and the published ids, so the host test
// can pin it against the reference contract functions (ddtree::column_depth /
// column_ancestors on an equivalent Tree) instead of against a second copy of the rule.
inline MtpTreeColumns mtp_tree_columns_from_siblings(const MtpSiblingLattice& lattice,
                                                    std::uint32_t node_budget) {
    const std::string defect = mtp_sibling_lattice_defect(lattice, node_budget);
    if (!defect.empty()) {
        throw std::invalid_argument("MTP sibling lattice rejected: " + defect);
    }
    const std::uint32_t paths = static_cast<std::uint32_t>(lattice.paths);
    const std::uint32_t steps = static_cast<std::uint32_t>(lattice.steps);

    MtpTreeColumns columns;
    columns.depths.resize(static_cast<std::size_t>(node_budget) + 1U);
    columns.masks.resize(static_cast<std::size_t>(node_budget) + 1U);
    columns.tokens.resize(node_budget);

    // Column 0 is the anchor column the reference functions already reserve.
    columns.depths[0] = 0;
    columns.masks[0]  = 1ULL;
    for (std::uint32_t s = 0; s < steps; ++s) {
        // The spine node of level s - 1; the anchor column for level 0. Every node of this
        // level hangs off it, which is what makes the siblings siblings and not ancestors.
        const std::uint32_t parent_column = s == 0 ? 0U : 1U + (s - 1U) * paths;
        const std::uint64_t parent_mask   = columns.masks[parent_column];
        for (std::uint32_t i = 0; i < paths; ++i) {
            const std::uint32_t node   = s * paths + i;
            const std::uint32_t column = 1U + node;
            columns.depths[column] = s + 1U;
            // mask[j] == (1 << j) | mask[parent]; the anchor bit and the column's own bit are
            // both covered because parent_mask already carries bit 0 (and every spine column).
            columns.masks[column]  = (1ULL << column) | parent_mask;
            const std::size_t at   = static_cast<std::size_t>(s) * paths + i;
            columns.tokens[node]   = lattice.ids[at];
        }
    }
    return columns;
}

// The one call the host hand-off makes for a tree round: build the layout from the lattice
// the proposal side published, validate it through the receiving seam, write the three
// layout fields and the column-ordered node drafts, and return the extent the round must
// carry (mtp_draft_count == the node count; mtp_tree_required_extent() states why).
inline std::uint32_t publish_mtp_tree_round(const MtpSiblingLattice& lattice,
                                            std::uint32_t node_budget, std::uint32_t width,
                                            const MtpTreePublishSlot& slot) {
    const MtpTreeColumns columns = mtp_tree_columns_from_siblings(lattice, node_budget);
    if (columns.columns() != width) {
        throw std::logic_error("MTP sibling lattice does not fill the round's verify frame");
    }
    publish_mtp_tree(columns, width, slot);
    return mtp_tree_required_extent(columns);
}

// The dense-lattice twin of the call above, for a proposal side that publishes a full grid.
inline std::uint32_t publish_mtp_tree_round(const MtpProposalLattice& lattice,
                                           std::uint32_t node_budget, std::uint32_t width,
                                           const MtpTreePublishSlot& slot) {
    const MtpTreeColumns columns = mtp_tree_columns_from_lattice(lattice, node_budget);
    publish_mtp_tree(columns, width, slot);
    return mtp_tree_required_extent(columns);
}

// A chain round must not inherit the previous round's tree: the verify side reads
// mtp_tree_columns unconditionally (program_impl.h:13354/13360), so a stale non-zero value
// would make an L <= 1 round take the tree branch. Calling this instead of writing the field
// by hand keeps the "a tree is published or the field is zero" invariant in one place.
inline void clear_mtp_tree_round(const MtpTreePublishSlot& slot) { clear_mtp_tree(slot); }

} // namespace ninfer::targets::qwen3_6::detail
