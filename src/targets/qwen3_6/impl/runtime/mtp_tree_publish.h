#pragma once

// ============================================================================
// The RECEIVING half of the MTP draft-tree publication (--draft-tree L,d, L > 1).
//
// WHY THIS FILE EXISTS
//   In a tree round the verify columns are tree NODES, not chain steps: two siblings share
//   a depth and neither is the other's ancestor, so neither the per-column DEPTH nor the
//   per-column ANCESTOR MASK can be derived from the round's extent. The verify side
//   therefore REFUSES an L > 1 round unless the proposal side has published that layout
//   (program_impl.h:13335-13342 reads SequenceState::mtp_tree_columns and throws; the
//   field and its two array neighbours are declared at program.h:465-474, and the only
//   other site in the whole tree is the request-level clear at program_impl.h:6689-6692).
//   This header is the ONE place that layout is written: it takes the layout, checks it
//   against the same contract the verify side reads -- ddtree::column_depth() /
//   ddtree::column_ancestors() (include/ninfer/ops/dflash2_ddtree.h:314-345) -- and copies
//   it into the sequence together with the column-ordered draft token ids that the ingress
//   loop (program_impl.h:13346-13366) consumes.
//
// WHAT IT IS NOT
//   It is NOT a producer. Nothing here decides WHICH tree a round verifies. Producing it
//   (top-L candidates per depth from the draft head, then ddtree::build_tree) belongs to
//   the proposal side, and as of this writing NO code in the tree does it -- which is why
//   an L > 1 round still finds mtp_tree_columns == 0 and is still refused. What this file
//   buys is that the producer's contract is explicit, validated and testable instead of
//   implicit: a producer that gets the layout wrong is rejected loudly here rather than
//   verified as something else downstream.
//
// HOST ONLY / NO CUDA / NO ENGINE TENSOR -- the same discipline as dflash2_ddtree.h, so a
// host test pins the layout arithmetic with no device, no artifact and no variant macros.
// ============================================================================

#include <ninfer/ops/dflash2_ddtree.h>
#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// One round's draft tree in VERIFY-COLUMN ORDER: column 0 is the anchor (the last confirmed
// token, already in the ledger), column j >= 1 carries tree node j - 1.
struct MtpTreeColumns {
    // Per column, the DEPTH of the node it carries relative to the anchor; depths[0] == 0.
    // This is ddtree::column_depth() verbatim (include/ninfer/ops/dflash2_ddtree.h:316-322).
    std::vector<std::uint32_t> depths;
    // Per column, the ancestor bit set: bit i set means "column j may attend column i".
    // masks[0] == 1; masks[j] always carries bit 0 (the anchor) and bit j (itself), and
    // masks[j] == (1 << j) | masks[parent]. ddtree::column_ancestors() verbatim
    // (include/ninfer/ops/dflash2_ddtree.h:333-345).
    std::vector<std::uint64_t> masks;
    // The draft token id carried by column j >= 1: tokens[j - 1] is node j - 1's token.
    std::vector<TokenId> tokens;

    std::uint32_t columns() const noexcept { return static_cast<std::uint32_t>(depths.size()); }
    std::uint32_t nodes() const noexcept {
        return columns() == 0 ? 0 : columns() - 1;
    }
};

// The verify-column layout of a ddtree::Tree whose node i carries `node_tokens[i]`. This is a
// straight transcription of the reference contract, so the published layout and the test's
// expectation cannot drift: depths and masks come from column_depth() / column_ancestors()
// verbatim, and column 0 is the anchor column those two functions already reserve.
inline MtpTreeColumns mtp_tree_columns_from_ddtree(const ops::ddtree::Tree& tree,
                                                   std::span<const TokenId> node_tokens) {
    if (node_tokens.size() != static_cast<std::size_t>(tree.node_count())) {
        throw std::invalid_argument(
            "mtp_tree_columns_from_ddtree: node_tokens must carry one id per ddtree node");
    }
    const std::vector<std::int32_t> depth = ops::ddtree::column_depth(tree);
    const std::vector<std::uint64_t> mask = ops::ddtree::column_ancestors(tree);
    if (depth.size() != mask.size() || depth.empty()) {
        throw std::logic_error("mtp_tree_columns_from_ddtree: ddtree columns are empty");
    }
    MtpTreeColumns columns;
    columns.depths.reserve(depth.size());
    for (std::int32_t d : depth) {
        if (d < 0) {
            throw std::logic_error("mtp_tree_columns_from_ddtree: negative column depth");
        }
        columns.depths.push_back(static_cast<std::uint32_t>(d));
    }
    columns.masks = mask;
    columns.tokens.assign(node_tokens.begin(), node_tokens.end());
    return columns;
}

// "" when `columns` is a well-formed tree for a verify frame of `width` columns, else the
// defect. A defect string rather than a bool, in the style of the rest of the runtime, so a
// rejection says what it found.
inline std::string mtp_tree_layout_defect(const MtpTreeColumns& columns, std::uint32_t width) {
    const std::uint32_t n = columns.columns();
    if (columns.depths.size() != columns.masks.size()) {
        return "depths and masks disagree on the column count";
    }
    if (n == 0) {
        return "no columns (0 columns is the CLEARED state, not a tree)";
    }
    if (columns.tokens.size() + 1 != columns.depths.size()) {
        return "the token column count is not the node count plus the anchor";
    }
    if (width == 0 || n > width) {
        return "columns (" + std::to_string(n) + ") exceed the verify frame width (" +
               std::to_string(width) + ")";
    }
    if (n > 63) {
        return "more columns (" + std::to_string(n) + ") than the 64-bit mask can index";
    }
    if (columns.depths[0] != 0 || columns.masks[0] != 1ULL) {
        return "column 0 is not the anchor (depth 0, mask 1)";
    }
    const std::uint64_t round_bits = (1ULL << n) - 1ULL;
    bool chain_spelling            = true;
    for (std::uint32_t j = 1; j < n; ++j) {
        const std::uint64_t self = 1ULL << j;
        const std::uint64_t mask = columns.masks[j];
        if (columns.depths[j] == 0) {
            return "non-anchor column " + std::to_string(j) + " is at depth 0";
        }
        if ((mask & 1ULL) == 0) {
            return "column " + std::to_string(j) + " does not attend the anchor";
        }
        if ((mask & self) == 0) {
            return "column " + std::to_string(j) + " does not attend itself";
        }
        if ((mask & ~round_bits) != 0) {
            return "column " + std::to_string(j) + " attends a column outside this round";
        }
        const std::uint64_t rest = mask & ~self;
        if (rest == 0) {
            return "column " + std::to_string(j) + " has no parent (it does not attend the anchor)";
        }
        // The parent is the deepest column in `rest`: mask[j] == (1 << j) | mask[parent], so
        // rest must be EXACTLY one existing column's mask, and that column must be exactly one
        // depth shallower. This is what makes "sibling is not an ancestor" checkable.
        std::uint32_t parent = 0;
        for (std::uint32_t i = j; i-- > 0;) {
            if (((rest >> i) & 1ULL) != 0) {
                parent = i;
                break;
            }
        }
        if (rest != columns.masks[parent]) {
            return "column " + std::to_string(j) +
                   "'s ancestors are not its parent's mask plus itself";
        }
        if (columns.depths[parent] + 1 != columns.depths[j]) {
            return "column " + std::to_string(j) + "'s parent is not exactly one depth shallower";
        }
        // The CHAIN spelling: depth == column index and the prefix mask, i.e. the layout the
        // verify side already derives for a chain. Publishing it for an L > 1 round is the one
        // thing this seam must not let through: verifying a chain while reporting a tree is
        // exactly the silent wrong answer the verify-side guard exists to prevent.
        if (columns.depths[j] != j || mask != ((1ULL << (j + 1)) - 1ULL)) {
            chain_spelling = false;
        }
    }
    if (n >= 3 && chain_spelling) {
        return "the layout is the CHAIN spelling, not a tree: an L > 1 round must never verify a "
               "chain as a tree";
    }
    return std::string();
}

inline void validate_mtp_tree_columns(const MtpTreeColumns& columns, std::uint32_t width) {
    const std::string defect = mtp_tree_layout_defect(columns, width);
    if (!defect.empty()) {
        throw std::logic_error("MTP draft-tree layout rejected: " + defect);
    }
}

// The four sequence members the verify side reads, addressed WITHOUT naming SequenceState so
// this header stays engine-free and host-compilable. Field-for-field with program.h:463-474:
//   columns -> SequenceState::mtp_tree_columns      (program.h:472)
//   depths  -> SequenceState::mtp_tree_depths.data() (program.h:473)
//   masks   -> SequenceState::mtp_tree_masks.data()  (program.h:474)
//   drafts  -> SequenceState::mtp_drafts.data()      (program.h:463, the chain spelling)
struct MtpTreePublishSlot {
    std::uint32_t* columns        = nullptr;
    std::uint32_t* depths         = nullptr;
    std::uint64_t* masks          = nullptr;
    TokenId* drafts               = nullptr;
    std::uint32_t array_capacity  = 0; // depth/mask array extent: qwen3_6::kMtpDecodeMaximumWidth
    std::uint32_t draft_capacity  = 0; // mtp_drafts extent: qwen3_6::kMtpDecodeMaximumDrafts
};

// Publish `columns` as the round the ingress fill will build. Validation runs FIRST and throws
// before a single field is written, so a rejected layout can never leave a half-published tree
// behind. On success the three layout fields and the node drafts are the only things written;
// mtp_draft_count is the caller's, because it is the round's own clamp, not the tree's.
inline void publish_mtp_tree(const MtpTreeColumns& columns, std::uint32_t width,
                             const MtpTreePublishSlot& slot) {
    if (slot.columns == nullptr || slot.depths == nullptr || slot.masks == nullptr ||
        slot.drafts == nullptr) {
        throw std::logic_error("MTP draft-tree publish slot is incomplete");
    }
    validate_mtp_tree_columns(columns, width);
    if (columns.columns() > slot.array_capacity) {
        throw std::logic_error("MTP draft-tree publish slot is narrower than the layout");
    }
    if (columns.nodes() > slot.draft_capacity) {
        throw std::logic_error("MTP draft-tree publish slot has no room for the node drafts");
    }
    *slot.columns = columns.columns();
    for (std::uint32_t j = 0; j < columns.columns(); ++j) {
        slot.depths[j] = columns.depths[j];
        slot.masks[j]  = columns.masks[j];
    }
    for (std::uint32_t node = 0; node < columns.nodes(); ++node) {
        slot.drafts[node] = columns.tokens[node];
    }
}

// Clear the publication. A reused lane must not inherit the previous request's tree (the same
// reason program_impl.h:6689-6692 clears it), and calling this instead of writing the field by
// hand keeps the "tree is published or it is zero" invariant in one place.
inline void clear_mtp_tree(const MtpTreePublishSlot& slot) {
    if (slot.columns == nullptr) {
        throw std::logic_error("MTP draft-tree publish slot is incomplete");
    }
    *slot.columns = 0;
}

// The round's live extent that MUST accompany a published tree: SequenceState::mtp_draft_count
// (program.h:464) must equal exactly the node count. The verify loop indexes
// mtp_tree_depths[j] / mtp_tree_masks[j] by VERIFY COLUMN up to the live extent and pulls
// mtp_drafts[j] per column (program_impl.h:13346-13366), and it sets
// target_valid_columns = extent + 1. Publishing a tree with a LARGER extent would keep columns
// live that no node owns: their depth and mask would silently repeat the last node's while their
// draft token is whatever the previous round left in the array. Publishing with a SMALLER extent
// would verify only the shallowest columns of the tree, which is a legal sub-tree (the column
// order is sorted by depth) but is not the tree that was built. This is a named companion rather
// than a fourth written field because mtp_draft_count is also the round's own clamp under the
// adaptive ladder; the caller owns that decision, this function only states the required value.
inline std::uint32_t mtp_tree_required_extent(const MtpTreeColumns& columns) {
    return columns.nodes();
}

} // namespace ninfer::targets::qwen3_6::detail
