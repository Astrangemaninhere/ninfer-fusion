// Host-only test for the MTP draft-tree publication seam (mtp_tree_publish.h).
//
// It deliberately includes NO engine, CUDA, artifact or variant header, so it also compiles and
// runs under a plain
//   g++ -std=gnu++20 -I include -I src -I tests -I third_party
// without a GPU. It uses the REAL ddtree reference builder (include/ninfer/ops/dflash2_ddtree.h)
// rather than a re-implementation, so the layout the seam publishes and the layout the verify
// side reads cannot drift apart without this test failing.

#include "ninfer/ops/dflash2_ddtree.h"
#include "targets/qwen3_6/impl/runtime/mtp_tree_publish.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::ops::ddtree::build_tree;
using ninfer::ops::ddtree::column_ancestors;
using ninfer::ops::ddtree::column_depth;
using ninfer::ops::ddtree::score_offset;
using ninfer::ops::ddtree::Tree;
using ninfer::TokenId;
using ninfer::targets::qwen3_6::detail::clear_mtp_tree;
using ninfer::targets::qwen3_6::detail::MtpTreeColumns;
using ninfer::targets::qwen3_6::detail::MtpTreePublishSlot;
using ninfer::targets::qwen3_6::detail::mtp_tree_columns_from_ddtree;
using ninfer::targets::qwen3_6::detail::mtp_tree_layout_defect;
using ninfer::targets::qwen3_6::detail::mtp_tree_required_extent;
using ninfer::targets::qwen3_6::detail::publish_mtp_tree;

int checks   = 0;
int failures = 0;

// Explicit 64-bit literals: on LP64 `1ULL` is `unsigned long long` while uint64_t is
// `unsigned long`, so a bare ULL literal would not deduce against the array element type.
constexpr std::uint64_t mask64(std::uint64_t v) { return v; }

void report(const std::string& what, bool ok, const std::string& detail = std::string()) {
    ++checks;
    if (ok) { return; }
    ++failures;
    std::cout << "FAIL: " << what << (detail.empty() ? "" : " -- " + detail) << "\n";
}

template <typename T>
void check_eq(const std::string& what, const T& got, const T& want) {
    ++checks;
    if (got == want) { return; }
    ++failures;
    std::cout << "FAIL: " << what << " -- got " << static_cast<long long>(got) << " want "
              << static_cast<long long>(want) << "\n";
}

void check_vec(const std::string& what, const std::vector<std::uint32_t>& got,
               const std::vector<std::uint32_t>& want) {
    ++checks;
    if (got == want) { return; }
    ++failures;
    std::string g = "[", w = "[";
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (i != 0) { g += ","; }
        g += std::to_string(got[i]);
    }
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (i != 0) { w += ","; }
        w += std::to_string(want[i]);
    }
    std::cout << "FAIL: " << what << " -- got " << g << "] want " << w << "]\n";
}

void check_masks(const std::string& what, const std::vector<std::uint64_t>& got,
                 const std::vector<std::uint64_t>& want) {
    ++checks;
    if (got == want) { return; }
    ++failures;
    std::string g = "[", w = "[";
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (i != 0) { g += ","; }
        g += std::to_string(got[i]);
    }
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (i != 0) { w += ","; }
        w += std::to_string(want[i]);
    }
    std::cout << "FAIL: " << what << " -- got " << g << "] want " << w << "]\n";
}

// A deterministic, tie-free ddtree score lattice: candidate rank 0 is always the best successor
// of every predecessor, rank 1 next, and so on, with a small predecessor-dependent term so the
// beam order is unambiguous. Same flat layout as the engine selector
// (dflash2_ddtree.h:107-118): scores[s][p][c] at s + steps * (p + 16 * c).
std::vector<double> lattice(std::int32_t steps) {
    std::vector<double> scores(static_cast<std::size_t>(steps) * 16 * 16, 0.0);
    for (std::int32_t s = 0; s < steps; ++s) {
        for (std::int32_t p = 0; p < 16; ++p) {
            for (std::int32_t c = 0; c < 16; ++c) {
                scores[static_cast<std::size_t>(
                    score_offset(s, steps, p, c))] = 1000.0 - 3.0 * c - 0.25 * p - 0.1 * s;
            }
        }
    }
    return scores;
}

std::vector<TokenId> node_tokens(const Tree& tree, TokenId first) {
    std::vector<TokenId> ids;
    for (int i = 0; i < tree.node_count(); ++i) { ids.push_back(first + i); }
    return ids;
}

// Encode each node's own (rank, depth) into its token id, so the published node ORDER can be
// checked against the ddtree node order instead of merely counted.
std::vector<TokenId> encoded_node_tokens(const Tree& tree) {
    std::vector<TokenId> ids;
    for (int i = 0; i < tree.node_count(); ++i) {
        const auto node = static_cast<std::size_t>(i);
        ids.push_back(static_cast<TokenId>(1000 + tree.nodes[node].rank * 16 +
                                          tree.nodes[node].depth));
    }
    return ids;
}

// ---------------------------------------------------------------------------------------------
// 1. The layout is ddtree::column_depth / column_ancestors, verbatim.
// ---------------------------------------------------------------------------------------------
void test_layout_is_the_ddtree_contract() {
    for (std::int32_t steps = 1; steps <= 4; ++steps) {
        for (std::int32_t budget = steps; budget <= 8; ++budget) {
            const std::vector<double> scores = lattice(steps);
            const Tree tree                  = build_tree(scores.data(), steps, budget);
            const std::vector<TokenId> ids   = node_tokens(tree, 900);
            const MtpTreeColumns columns     = mtp_tree_columns_from_ddtree(tree, ids);

            const std::string tag = "layout/steps=" + std::to_string(steps) + ",budget=" +
                                    std::to_string(budget);
            check_eq(tag + "/columns", columns.columns(),
                     static_cast<std::uint32_t>(tree.columns()));
            std::vector<std::uint32_t> want_depth;
            for (std::int32_t d : column_depth(tree)) {
                want_depth.push_back(static_cast<std::uint32_t>(d));
            }
            check_vec(tag + "/depths", columns.depths, want_depth);
            check_masks(tag + "/masks", columns.masks, column_ancestors(tree));
            check_eq(tag + "/anchor_depth", columns.depths[0], 0U);
            check_eq(tag + "/anchor_mask", columns.masks[0], mask64(1));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 2. Sibling semantics: at L >= 2 two columns share a depth and neither attends the other.
//    This is the property that makes a tree un-derivable from the extent alone.
// ---------------------------------------------------------------------------------------------
void test_siblings_share_a_depth_and_never_attend_each_other() {
    const std::vector<double> scores = lattice(1);
    const Tree tree                  = build_tree(scores.data(), 1, 2); // L = 2, one depth
    check_eq(std::string("sibling/ddtree_nodes"), tree.node_count(), 2);
    const MtpTreeColumns columns =
        mtp_tree_columns_from_ddtree(tree, node_tokens(tree, 700));
    check_vec(std::string("sibling/depths"), columns.depths,
              std::vector<std::uint32_t>{0, 1, 1});
    check_masks(std::string("sibling/masks"), columns.masks,
                std::vector<std::uint64_t>{1, 0b011, 0b101});
    report(std::string("sibling/column_2_does_not_attend_column_1"),
           (columns.masks[2] & (1ULL << 1)) == 0);
    report(std::string("sibling/same_depth"), columns.depths[1] == columns.depths[2]);
    report(std::string("sibling/is_not_the_same_as_the_chain"),
           columns.masks[2] != ((1ULL << 3) - 1ULL));
    report(std::string("valid/real_tree_accepted"),
           mtp_tree_layout_defect(columns, 3).empty(),
           mtp_tree_layout_defect(columns, 3));
}

// ---------------------------------------------------------------------------------------------
// 3. The anti-laundering rule: an L = 1 tree IS the chain spelling, and this seam refuses it.
//    Without this, a producer could "publish" the chain it already had and turn the verify-side
//    guard green while verifying no tree at all.
// ---------------------------------------------------------------------------------------------
void test_chain_spelling_is_refused() {
    const std::vector<double> scores = lattice(3);
    const Tree chain                 = build_tree(scores.data(), 3, 3); // budget/steps -> L = 1
    const MtpTreeColumns columns =
        mtp_tree_columns_from_ddtree(chain, node_tokens(chain, 1));
    check_eq(std::string("chain/columns"), columns.columns(), 4U);
    check_vec(std::string("chain/depths"), columns.depths,
              std::vector<std::uint32_t>{0, 1, 2, 3});
    // Byte-identical to the chain the verify side derives when no tree is published
    // (program_impl.h:13352-13356 / 13359-13361), so it is a well-formed LAYOUT ...
    report(std::string("chain/spelling_is_the_verify_side_prefix" ),
           columns.masks[3] == ((1ULL << 4) - 1ULL));
    // ... and exactly for that reason it must be refused as a tree.
    report(std::string("chain/refused_as_a_tree"), !mtp_tree_layout_defect(columns, 4).empty(),
           mtp_tree_layout_defect(columns, 4));
}

// ---------------------------------------------------------------------------------------------
// 4. Malformed layouts are refused, and each for its own reason.
// ---------------------------------------------------------------------------------------------
MtpTreeColumns good_two_path_tree() {
    const std::vector<double> scores = lattice(1);
    const Tree tree                  = build_tree(scores.data(), 1, 2);
    return mtp_tree_columns_from_ddtree(tree, node_tokens(tree, 700));
}

void test_malformed_layouts_are_refused() {
    const MtpTreeColumns good = good_two_path_tree();

    { // frame narrower than the layout
        report(std::string("bad/frame_too_narrow"), !mtp_tree_layout_defect(good, 2).empty());
    }
    { // width 0 is "no frame", not "a tree of no columns"
        report(std::string("bad/width_zero"), !mtp_tree_layout_defect(good, 0).empty());
    }
    { // widened: a bit outside the round
        MtpTreeColumns c = good;
        c.masks[2] |= (1ULL << 15);
        report(std::string("bad/attends_outside_the_round"),
               !mtp_tree_layout_defect(c, 16).empty());
    }
    { // a sibling turned into an ancestor
        MtpTreeColumns c = good;
        c.masks[2] |= (1ULL << 1);
        report(std::string("bad/sibling_used_as_parent"), !mtp_tree_layout_defect(c, 16).empty(),
               mtp_tree_layout_defect(c, 16));
    }
    { // parent one depth too shallow
        MtpTreeColumns c = good;
        c.depths[2] = 3;
        report(std::string("bad/parent_not_one_shallower"),
               !mtp_tree_layout_defect(c, 16).empty());
    }
    { // token column count disagrees with the node count
        MtpTreeColumns c = good;
        c.tokens.pop_back();
        report(std::string("bad/token_count"), !mtp_tree_layout_defect(c, 16).empty());
    }
    { // anchor demoted
        MtpTreeColumns c = good;
        c.depths[0] = 1;
        report(std::string("bad/anchor_depth"), !mtp_tree_layout_defect(c, 16).empty());
    }
    { // cleared state is not a tree (0 columns means "no tree")
        report(std::string("bad/zero_columns"), !mtp_tree_layout_defect(MtpTreeColumns{}, 16).empty());
    }
}

// ---------------------------------------------------------------------------------------------
// 5. The publish seam, and the RED/GREEN control for the verify-side guard.
//
// SequenceTreeMirror is a field-for-field mirror of the four SequenceState members the verify
// side reads, with the SAME initialisers program.h uses:
//     program.h:463  std::array<TokenId, kMtpDecodeMaximumDrafts>          mtp_drafts{};
//     program.h:472  std::uint32_t                                         mtp_tree_columns = 0;
//     program.h:473  std::array<std::uint32_t, kMtpDecodeMaximumWidth>     mtp_tree_depths{};
//     program.h:474  std::array<std::uint64_t, kMtpDecodeMaximumWidth>     mtp_tree_masks{};
// It is a MIRROR, not the real struct: the real one is variant-instantiated behind
// NINFER_QWEN36_VARIANT in program.h and needs the engine to link. What it reproduces exactly is
// the property under test -- that the field is 0 until a publisher writes it.
// ---------------------------------------------------------------------------------------------
struct SequenceTreeMirror {
    static constexpr std::uint32_t kWidth  = 16; // round_state.h:18 kMtpDecodeMaximumWidth
    static constexpr std::uint32_t kDrafts = 15; // round_state.h:17 kMtpDecodeMaximumDrafts
    std::array<TokenId, kDrafts> mtp_drafts{};
    std::uint32_t mtp_tree_columns = 0;
    std::array<std::uint32_t, kWidth> mtp_tree_depths{};
    std::array<std::uint64_t, kWidth> mtp_tree_masks{};
};

MtpTreePublishSlot slot_of(SequenceTreeMirror& s) {
    MtpTreePublishSlot slot;
    slot.columns        = &s.mtp_tree_columns;
    slot.depths         = s.mtp_tree_depths.data();
    slot.masks          = s.mtp_tree_masks.data();
    slot.drafts         = s.mtp_drafts.data();
    slot.array_capacity = SequenceTreeMirror::kWidth;
    slot.draft_capacity = SequenceTreeMirror::kDrafts;
    return slot;
}

// program_impl.h:13335-13342, verbatim predicate.
bool tree_round_refused(const SequenceTreeMirror& s, std::uint32_t draft_tree_paths) {
    const std::uint32_t tree_columns = s.mtp_tree_columns;
    return draft_tree_paths > 1 && tree_columns == 0;
}

void test_publish_seam_and_guard_control() {
    SequenceTreeMirror s;

    // RED (no publisher): the state a round reaches with nothing having published a tree.
    report(std::string("guard/RED_unpublished_tree_round_is_refused"),
           tree_round_refused(s, 2));
    // The chain round (L <= 1) is never gated by this at all.
    report(std::string("guard/chain_round_never_gated"), !tree_round_refused(s, 1));
    check_eq(std::string("guard/RED_columns_stay_zero"), s.mtp_tree_columns, 0U);

    // GREEN (with the seam): the same round, now published.
    const MtpTreeColumns tree = good_two_path_tree();
    publish_mtp_tree(tree, 3, slot_of(s));
    report(std::string("guard/GREEN_published_tree_round_passes"), !tree_round_refused(s, 2));
    check_eq(std::string("publish/columns"), s.mtp_tree_columns, 3U);
    check_eq(std::string("publish/depth_1"), s.mtp_tree_depths[1], 1U);
    check_eq(std::string("publish/depth_2"), s.mtp_tree_depths[2], 1U);
    check_eq(std::string("publish/mask_0"), s.mtp_tree_masks[0], mask64(1));
    check_eq(std::string("publish/mask_2"), s.mtp_tree_masks[2], mask64(0b101));
    check_eq(std::string("publish/node_draft_0"), s.mtp_drafts[0], tree.tokens[0]);
    check_eq(std::string("publish/node_draft_1"), s.mtp_drafts[1], tree.tokens[1]);

    // Clearing returns the round to RED -- the reused-lane rule of program_impl.h:6689-6692.
    clear_mtp_tree(slot_of(s));
    check_eq(std::string("clear/columns_back_to_zero"), s.mtp_tree_columns, 0U);
    report(std::string("guard/RED_again_after_clear"), tree_round_refused(s, 2));

    // A refused layout must not leave a half-published tree behind: validation runs first.
    SequenceTreeMirror before;
    SequenceTreeMirror after;
    MtpTreePublishSlot a = slot_of(after);
    bool threw           = false;
    try {
        publish_mtp_tree(MtpTreeColumns{}, 16, a);
    } catch (const std::logic_error&) {
        threw = true;
    }
    report(std::string("publish/bad_layout_throws"), threw);
    check_eq(std::string("publish/bad_layout_wrote_nothing"), after.mtp_tree_columns, 0U);
    check_eq(std::string("publish/bad_layout_left_no_masks"), after.mtp_tree_masks[1], mask64(0));
    check_eq(std::string("publish/control_mirror_still_zero"), before.mtp_tree_columns, 0U);

    // A slot with no room is a programming error, not a silent truncation.
    SequenceTreeMirror narrow;
    MtpTreePublishSlot n = slot_of(narrow);
    n.array_capacity     = 2; // the layout has 3 columns
    bool narrow_threw    = false;
    try {
        publish_mtp_tree(tree, 3, n);
    } catch (const std::logic_error&) {
        narrow_threw = true;
    }
    report(std::string("publish/narrow_slot_throws"), narrow_threw);
    check_eq(std::string("publish/narrow_slot_wrote_nothing"), narrow.mtp_tree_columns, 0U);

    // An incomplete slot is refused before anything is dereferenced.
    MtpTreePublishSlot incomplete;
    bool incomplete_threw = false;
    try {
        publish_mtp_tree(tree, 3, incomplete);
    } catch (const std::logic_error&) {
        incomplete_threw = true;
    }
    report(std::string("publish/incomplete_slot_throws"), incomplete_threw);

    // The round's live extent must be exactly the node count: the verify loop keeps columns
    // [0, extent] live and pulls one draft token per column.
    check_eq(std::string("publish/required_extent_is_the_node_count"),
             mtp_tree_required_extent(tree), tree.nodes());
    check_eq(std::string("publish/required_extent_2"), mtp_tree_required_extent(tree), 2U);
}

// ---------------------------------------------------------------------------------------------
// 6. The seam's depth/mask arrays are indexed by VERIFY COLUMN, and the node tokens by node:
//    the ingress loop (program_impl.h:13346-13366) reads mtp_tree_depths[j] / mtp_tree_masks[j]
//    for column j and mtp_drafts[n] for node n. A 5-column tree pins the offset-by-one.
// ---------------------------------------------------------------------------------------------
void test_column_and_node_indexing() {
    const std::vector<double> scores = lattice(2);
    const Tree tree                  = build_tree(scores.data(), 2, 4); // L = 2, d = 2
    const MtpTreeColumns columns     = mtp_tree_columns_from_ddtree(tree, encoded_node_tokens(tree));
    check_eq(std::string("index/columns"), columns.columns(),
             static_cast<std::uint32_t>(tree.columns()));
    check_eq(std::string("index/nodes"), columns.nodes(),
             static_cast<std::uint32_t>(tree.node_count()));
    report(std::string("index/fits_the_frame"), columns.columns() <= 16);
    report(std::string("index/real_tree_accepted"),
           mtp_tree_layout_defect(columns, columns.columns()).empty(),
           mtp_tree_layout_defect(columns, columns.columns()));
    // mtp_drafts[n] carries node n, i.e. verify column n + 1, and the published column order is
    // the ddtree node order -- not merely the same count.
    for (std::uint32_t n = 0; n < columns.nodes(); ++n) {
        const auto node = static_cast<std::size_t>(n);
        check_eq(std::string("index/node_order_preserved"), columns.tokens[n],
                 static_cast<TokenId>(1000 + tree.nodes[node].rank * 16 +
                                      tree.nodes[node].depth));
        check_eq(std::string("index/depth_equals_ddtree_column_depth"), columns.depths[n + 1],
                 static_cast<std::uint32_t>(tree.nodes[node].depth + 1));
        check_eq(std::string("index/mask_equals_ddtree_column_ancestors"), columns.masks[n + 1],
                 column_ancestors(tree)[n + 1]);
    }
}

} // namespace

int main() {
    test_layout_is_the_ddtree_contract();
    test_siblings_share_a_depth_and_never_attend_each_other();
    test_chain_spelling_is_refused();
    test_malformed_layouts_are_refused();
    test_publish_seam_and_guard_control();
    test_column_and_node_indexing();

    std::cout << "mtp_tree_publish_test: " << (checks - failures) << "/" << checks << " checks"
              << (failures == 0 ? " passed OK" : " FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
