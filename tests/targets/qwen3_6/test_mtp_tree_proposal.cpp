// ============================================================================
// Host test for the PROPOSAL-SIDE FILL of the MTP draft-tree publication
// (src/targets/qwen3_6/impl/runtime/mtp_tree_proposal_fill.h).
//
// It compiles with NO engine header, NO CUDA, NO artifact and NO variant macro:
//   c++ -std=gnu++20 -I <tree>/include -I <tree>/src -I <tree>/tests ...
// and it runs on the host. It leaks nothing to disk and touches no device.
//
// WHAT IT PINS, AND WHY EACH CHECK CAN GO RED
//   * the EGRESS INDEX FORMULA the device writer and the host reader must agree on -- pinned
//     against a hand-built array in exactly the device layout, plus a counter-check that reading
//     it with the WRONG stride really returns different data (otherwise the pin is vacuous).
//   * the CROSS-CHECK that turns "the lattice was published" into something checkable: a row
//     whose spine is not the chain argmax must be REFUSED. Both directions are present (the
//     mismatching row is refused AND the matching row is accepted), so a broken check cannot
//     look green.
//   * the PLANNING-TIME head guard: refused for a shortlist head on an L > 1 tree, and NOT
//     fired for the chain spelling (firing there would refuse every shortlist-head
//     --draft-tokens run).
//   * the FILL itself for three legal shapes, against the REAL ddtree contract functions and
//     against the producing half's own mtp_tree_columns_from_siblings -- not against a second
//     copy of the sibling rule.
// ============================================================================

#include "targets/qwen3_6/impl/runtime/mtp_tree_proposal_fill.h"

#include <ninfer/ops/dflash2_ddtree.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

int checks = 0;
int failed = 0;

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failed;
        std::printf("FAIL %s\n", what.c_str());
    }
}

template <typename A, typename B>
void check_eq(const A& got, const B& want, const std::string& what) {
    ++checks;
    if (!(got == want)) {
        ++failed;
        std::printf("FAIL %s (got %lld want %lld)\n", what.c_str(),
                    static_cast<long long>(got), static_cast<long long>(want));
    }
}

using ninfer::TokenId;
namespace ops = ninfer::ops;
using ninfer::targets::qwen3_6::detail::fill_mtp_tree_round;
using ninfer::targets::qwen3_6::detail::MtpProposalRow;
using ninfer::targets::qwen3_6::detail::MtpSiblingLattice;
using ninfer::targets::qwen3_6::detail::MtpTreeColumns;
using ninfer::targets::qwen3_6::detail::MtpTreePublishSlot;
using ninfer::targets::qwen3_6::detail::mtp_proposal_head_defect;
using ninfer::targets::qwen3_6::detail::mtp_proposal_ready_steps;
using ninfer::targets::qwen3_6::detail::mtp_proposal_row_defect;
using ninfer::targets::qwen3_6::detail::mtp_proposal_row_from_egress;
using ninfer::targets::qwen3_6::detail::mtp_proposal_row_from_prefill;
using ninfer::targets::qwen3_6::detail::mtp_sibling_lattice_from_row;
using ninfer::targets::qwen3_6::detail::mtp_tree_columns_from_siblings;
using ninfer::targets::qwen3_6::detail::mtp_tree_layout_defect;
using ninfer::targets::qwen3_6::detail::mtp_tree_proposal_index;
using ninfer::targets::qwen3_6::detail::mtp_tree_publish_slot;

// The declared egress layout constants, as literals:
//   kMtpDecodeMaximumDrafts == 15 (round_state.h:17), kMaximumConcurrency == 8
//   => kMtpTreeProposalDepthStride == 15 * 8 == 120            (round_state.h)
// The companion probe sh/t10_roundstate_probe.cpp asserts the REAL constants are these numbers
// and that sizeof(MtpDecodeEgress) grew by exactly 15 * 15 * 8 * 4, so this test's literals and
// the header cannot drift apart silently.
constexpr std::uint32_t kPathsMax    = 15;
constexpr std::uint32_t kLanes       = 4; // the round's batch, deliberately != the lane capacity
constexpr std::size_t   kDepthStride = static_cast<std::size_t>(kPathsMax) * 8U;
constexpr std::size_t   kChainStride = 8;
static_assert(kDepthStride == 120, "the egress depth stride moved");
// The proposal head's row range, i.e. the `rows` both ops::mtp_proposal_topk call sites pass.
// Literal pin: ninfer::targets::qwen3_6::kTokenDomain == 248077 (frontend.h),
// 27b TextConfig::token_domain = static_cast<int>(qwen3_6::kTokenDomain), TextConfig::output_rows
// = 248320 (the padded matrix). The companion probe sh/t11_probe.sh re-reads the real constants
// and the frame sizes, so these literals and the tree cannot drift apart silently.
constexpr std::uint32_t kTokenDomain = 248077;

// ---------------------------------------------------------------------------------------------
// A hand-built egress image in EXACTLY the device layout:
//     entry (depth s, rank i, lane t) at  s * 120 + i * 4 + t
// Every lane gets a distinct id per (depth, rank), so a mis-indexed read cannot pass by luck.
// ---------------------------------------------------------------------------------------------
struct EgressImage {
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> chain; // next_drafts: [drafts, kMaximumConcurrency]

    static std::int32_t id_of(std::uint32_t s, std::uint32_t i, std::uint32_t t) {
        return static_cast<std::int32_t>(100000 + 1000 * s + 100 * i + t);
    }

    EgressImage(std::uint32_t paths, std::uint32_t steps, std::uint32_t break_spine_at_depth) {
        ids.assign(kDepthStride * steps, -1);
        chain.assign(static_cast<std::size_t>(kChainStride) * steps, -1);
        for (std::uint32_t s = 0; s < steps; ++s) {
            for (std::uint32_t i = 0; i < paths; ++i) {
                for (std::uint32_t t = 0; t < kLanes; ++t) {
                    ids[mtp_tree_proposal_index(s, i, kLanes, t, kDepthStride)] = id_of(s, i, t);
                }
            }
            for (std::uint32_t t = 0; t < kLanes; ++t) {
                // The spine IS the chain draft, except where the corruption asks otherwise.
                chain[static_cast<std::size_t>(s) * kChainStride + t] =
                    (break_spine_at_depth != 0 && s == break_spine_at_depth)
                        ? static_cast<std::int32_t>(id_of(s, 0, t) + 7)
                        : id_of(s, 0, t);
            }
        }
    }
};

// The tree the sibling rule claims to describe, so the REAL column_depth / column_ancestors can
// be run on it (the same construction the producing half's test uses).
ninfer::ops::ddtree::Tree sibling_equivalent_tree(std::int32_t paths, std::int32_t steps) {
    using ninfer::ops::ddtree::kRoot;
    ninfer::ops::ddtree::Tree tree;
    tree.steps       = steps;
    tree.node_budget = paths * steps;
    for (std::int32_t s = 0; s < steps; ++s) {
        for (std::int32_t i = 0; i < paths; ++i) {
            ninfer::ops::ddtree::Node node;
            node.depth  = s;
            node.rank   = i;
            node.parent = s == 0 ? kRoot : (s - 1) * paths + 0;
            tree.nodes.push_back(node);
        }
    }
    tree.path_ends.push_back((steps - 1) * paths);
    return tree;
}

// A field-for-field mirror of the four SequenceState members the fill writes
// (src/targets/qwen3_6/impl/runtime/program.h:463/472-474), with the same names and the same
// array extents, so mtp_tree_publish_slot() is exercised against the real accessor shape.
struct FillMirror {
    std::uint32_t mtp_tree_columns = 0;
    std::array<std::uint32_t, 16> mtp_tree_depths{};
    std::array<std::uint64_t, 16> mtp_tree_masks{};
    std::array<TokenId, 15> mtp_drafts{};
};

// ---------------------------------------------------------------------------------------------
// 0. the reference contract really is the tree's own header.
// ---------------------------------------------------------------------------------------------
void test_contract_fingerprint() {
    check_eq(ops::ddtree::kTopK, 16, "ddtree::kTopK is 16");
    check_eq(ops::ddtree::kMaxNodes, 63, "ddtree::kMaxNodes is 63");
    static_assert(ops::ddtree::kTopK == 16, "the ddtree contract header moved");
}

// ---------------------------------------------------------------------------------------------
// 1. the index formula, and the counter-check that it is load bearing.
// ---------------------------------------------------------------------------------------------
void test_index_formula_is_load_bearing() {
    const EgressImage image(2, 7, 0);
    bool all_planted = true;
    for (std::uint32_t s = 0; s < 7; ++s) {
        for (std::uint32_t i = 0; i < 2; ++i) {
            for (std::uint32_t t = 0; t < kLanes; ++t) {
                if (image.ids[mtp_tree_proposal_index(s, i, kLanes, t, kDepthStride)] !=
                    EgressImage::id_of(s, i, t)) {
                    all_planted = false;
                }
            }
        }
    }
    check(all_planted, "every planted entry is at the index the formula computes");

    // counter-check: the WRONG rank stride (the array's lane capacity instead of the round's
    // batch) must read different data, otherwise the pin above proves nothing.
    int differences = 0;
    for (std::uint32_t s = 0; s < 7; ++s) {
        for (std::uint32_t i = 1; i < 2; ++i) {
            for (std::uint32_t t = 0; t < kLanes; ++t) {
                const std::size_t wrong = static_cast<std::size_t>(s) * kDepthStride + i * 8U + t;
                if (image.ids[wrong] != EgressImage::id_of(s, i, t)) { ++differences; }
            }
        }
    }
    check(differences > 0, "the wrong rank stride really reads different data (pin is not vacuous)");
}

// ---------------------------------------------------------------------------------------------
// 2/3. the read, and the spine cross-check in BOTH directions.
// ---------------------------------------------------------------------------------------------
void test_read_and_ready_steps() {
    const std::uint32_t paths  = 2;
    const std::uint32_t steps  = 7;
    const std::uint32_t budget = paths * steps;

    const EgressImage good(paths, steps, 0);
    const MtpProposalRow row =
        mtp_proposal_row_from_egress(good.ids.data(), good.chain.data(), kLanes, /*lane=*/2, paths,
                                     steps, kDepthStride, kChainStride, kTokenDomain);
    check_eq(row.paths, paths, "read keeps paths");
    check_eq(row.steps, steps, "read keeps steps");
    check_eq(row.ids.size(), static_cast<std::size_t>(budget), "read compacts one id per node");
    bool planted = true;
    for (std::uint32_t s = 0; s < steps; ++s) {
        for (std::uint32_t i = 0; i < paths; ++i) {
            if (row.ids[static_cast<std::size_t>(s) * paths + i] != EgressImage::id_of(s, i, 2)) {
                planted = false;
            }
        }
    }
    check(planted, "the read returns THIS lane's ids, in (depth, rank) order");
    check_eq(mtp_proposal_ready_steps(row), steps, "GREEN: every depth's spine is the chain draft");
    check(mtp_proposal_row_defect(row, budget).empty(), "GREEN: the read row is publishable");
    check_eq(mtp_sibling_lattice_from_row(row, budget).ids.size(),
             static_cast<std::size_t>(budget),
             "GREEN: the row converts into the producer's lattice");

    // RED: one depth's spine disagrees with the chain draft.
    const EgressImage broken(paths, steps, 3);
    const MtpProposalRow broken_row =
        mtp_proposal_row_from_egress(broken.ids.data(), broken.chain.data(), kLanes, 2, paths,
                                     steps, kDepthStride, kChainStride, kTokenDomain);
    check(broken_row.chain != row.chain, "RED: the corrupted image really differs from the good one");
    check(broken_row.ids == row.ids,
          "RED: the corruption is in the chain table alone, so the id read is unchanged");
    check_eq(mtp_proposal_ready_steps(broken_row), 3U,
             "RED: the ready count stops at the first depth whose spine disagrees");
    const std::string defect = mtp_proposal_row_defect(broken_row, budget);
    check(!defect.empty(), "RED: a lattice whose spine is not the chain argmax is refused");
    check(defect.find("depth 3") != std::string::npos,
          "RED: the refusal names the first bad depth (got: " + defect + ")");
    bool threw = false;
    try {
        (void)mtp_sibling_lattice_from_row(broken_row, budget);
    } catch (const std::exception&) { threw = true; }
    check(threw, "RED: the refusal propagates out of the conversion");
}

// ---------------------------------------------------------------------------------------------
// 4. the planning-time head guard.
// ---------------------------------------------------------------------------------------------
void test_head_guard() {
    check(!mtp_proposal_head_defect(/*shortlist_head=*/true, 2, 7).empty(),
          "RED: a shortlist head cannot publish an L > 1 tree");
    check(mtp_proposal_head_defect(true, 2, 7).find("FULL proposal head") != std::string::npos,
          "the refusal names the required head");
    check(mtp_proposal_head_defect(/*shortlist_head=*/false, 2, 7).empty(),
          "GREEN: the full head is allowed");
    // The control that matters: the guard must NOT fire for the chain spelling, or it would
    // refuse every shortlist-head --draft-tokens run.
    check(mtp_proposal_head_defect(true, 1, 7).empty(),
          "control: L == 1 (the chain) is not a tree round and is never refused");
    check(mtp_proposal_head_defect(true, 0, 0).empty(), "control: no tree is never refused");
    check(mtp_proposal_head_defect(true, 7, 0).empty(),
          "control: d == 0 is not a tree round and is never refused");
}

// ---------------------------------------------------------------------------------------------
// 5. the whole fill, for three legal shapes, against the real contract functions.
// ---------------------------------------------------------------------------------------------
void test_fill_for_legal_shapes() {
    struct Shape {
        std::uint32_t paths, steps, budget, width;
    };
    for (const Shape shape : {Shape{2, 1, 2, 3}, Shape{2, 7, 14, 15}, Shape{5, 3, 15, 16}}) {
        const std::string tag = "shape " + std::to_string(shape.paths) + "," +
                                std::to_string(shape.steps) + ": ";
        const EgressImage image(shape.paths, shape.steps, 0);
        const MtpProposalRow row =
            mtp_proposal_row_from_egress(image.ids.data(), image.chain.data(), kLanes, /*lane=*/1,
                                         shape.paths, shape.steps, kDepthStride, kChainStride,
                                         kTokenDomain);

        FillMirror mirror;
        // RED: the gate program_impl.h:13336 evaluates, on the state before the fill.
        const auto gate_refuses = [&] { return shape.paths > 1 && mirror.mtp_tree_columns == 0; };
        check(gate_refuses(), tag + "RED: an unpublished tree round is refused");
        check_eq(mirror.mtp_tree_columns, 0U, tag + "RED: the field really is zero");

        // Build the EXPECTED layout with the producing half's own entry point -- not with a
        // second copy of the sibling rule.
        const MtpSiblingLattice lattice = mtp_sibling_lattice_from_row(row, shape.budget);
        const MtpTreeColumns expected   = mtp_tree_columns_from_siblings(lattice, shape.budget);
        check_eq(expected.columns(), shape.width, tag + "the sibling layout fills the frame");
        check(mtp_tree_layout_defect(expected, shape.width).empty(),
              tag + "the receiving seam accepts the expected layout");

        const std::uint32_t extent =
            fill_mtp_tree_round(row, shape.budget, shape.width, mtp_tree_publish_slot(mirror));

        check(!gate_refuses(), tag + "GREEN: the filled round is not refused");
        check_eq(mirror.mtp_tree_columns, shape.width, tag + "columns == the frame width");
        check_eq(extent, shape.budget, tag + "the round's extent is the node count");
        check_eq(extent, shape.width - 1U, tag + "extent == width - 1 (the frame is exactly full)");
        check_eq(mirror.mtp_tree_columns - 1U, extent, tag + "extent == columns - 1");
        check_eq(mirror.mtp_tree_masks[0], 1ULL, tag + "the anchor column's mask is 1");
        check_eq(mirror.mtp_tree_depths[0], 0U, tag + "the anchor column's depth is 0");

        // field-for-field with the producing half's own layout
        bool columns_match = mirror.mtp_tree_columns == expected.columns();
        for (std::uint32_t j = 0; j < expected.columns(); ++j) {
            if (mirror.mtp_tree_depths[j] != expected.depths[j] ||
                mirror.mtp_tree_masks[j] != expected.masks[j]) {
                columns_match = false;
            }
        }
        check(columns_match, tag + "every published depth and mask equals the producing half's");
        bool drafts_match = true;
        for (std::uint32_t n = 0; n < expected.nodes(); ++n) {
            if (mirror.mtp_drafts[n] != expected.tokens[n]) { drafts_match = false; }
        }
        check(drafts_match, tag + "every published draft is the column-ordered node token");

        // and the layout really is the contracted tree: real column_depth / column_ancestors
        const ops::ddtree::Tree ref =
            sibling_equivalent_tree(static_cast<std::int32_t>(shape.paths),
                                    static_cast<std::int32_t>(shape.steps));
        const std::vector<std::int32_t> ref_depth  = ops::ddtree::column_depth(ref);
        const std::vector<std::uint64_t> ref_mask  = ops::ddtree::column_ancestors(ref);
        check_eq(ref_depth.size(), static_cast<std::size_t>(mirror.mtp_tree_columns),
                 tag + "reference column count agrees");
        bool ref_match = true;
        for (std::uint32_t j = 0; j < mirror.mtp_tree_columns; ++j) {
            if (mirror.mtp_tree_depths[j] != static_cast<std::uint32_t>(ref_depth[j]) ||
                mirror.mtp_tree_masks[j] != ref_mask[j]) {
                ref_match = false;
            }
        }
        check(ref_match, tag + "the published layout IS column_depth / column_ancestors");
        check(mirror.mtp_tree_depths[1] == 1 && mirror.mtp_tree_depths[shape.paths] == 1,
              tag + "level 0 carries all its siblings at depth 1");
        check((mirror.mtp_tree_masks[1] & (1ULL << 2)) == 0 &&
                  (mirror.mtp_tree_masks[2] & (1ULL << 1)) == 0,
              tag + "siblings do not attend each other");
        bool tokens_are_the_row = true;
        for (std::uint32_t n = 0; n < shape.budget; ++n) {
            if (mirror.mtp_drafts[n] != row.ids[n]) { tokens_are_the_row = false; }
        }
        check(tokens_are_the_row, tag + "every node token came from the lattice row, unchanged");

        // the reused-lane control: clearing returns the round to RED
        (void)ninfer::targets::qwen3_6::detail::clear_mtp_tree_round(
            mtp_tree_publish_slot(mirror));
        check_eq(mirror.mtp_tree_columns, 0U, tag + "cleared back to zero");
        check(gate_refuses(), tag + "RED again after the clear");
    }
}

// ---------------------------------------------------------------------------------------------
// 6. the shapes the producer must refuse, each with the control that proves it is not vacuous.
// ---------------------------------------------------------------------------------------------
void test_refusals() {
    const std::uint32_t paths  = 2;
    const std::uint32_t steps  = 7;
    const std::uint32_t budget = paths * steps;
    const EgressImage image(paths, steps, 0);
    const MtpProposalRow good = mtp_proposal_row_from_egress(
        image.ids.data(), image.chain.data(), kLanes, 0, paths, steps, kDepthStride, kChainStride,
        kTokenDomain);

    const auto refuses = [&](const MtpProposalRow& row, std::uint32_t node_budget) {
        try {
            (void)mtp_sibling_lattice_from_row(row, node_budget);
        } catch (const std::exception&) { return true; }
        return false;
    };
    check(!refuses(good, budget), "CONTROL: the good row is not refused");

    // a node budget that is not paths * steps
    check(refuses(good, budget - 1U), "a node budget that is not the row's node count is refused");
    // a single path is the chain spelling
    {
        MtpProposalRow one = good;
        one.paths          = 1;
        check(refuses(one, budget), "L == 1 (the chain spelling) is refused");
    }
    // no depths
    {
        MtpProposalRow none = good;
        none.steps          = 0;
        check(refuses(none, budget), "a row with no depths is refused");
    }
    // a short id table
    {
        MtpProposalRow short_ids = good;
        short_ids.ids.pop_back();
        check(refuses(short_ids, budget), "a short id table is refused");
    }
    // more nodes than the 64-bit mask can index, reached WITHOUT tripping the width check first
    {
        MtpProposalRow too_wide = good;
        too_wide.paths          = 20;
        too_wide.ids.assign(20 * steps, 7);
        const std::string why = mtp_proposal_row_defect(too_wide, 20 * steps);
        check(!why.empty() && why.find("64-bit ancestor mask") != std::string::npos,
              "more nodes than the mask can index is refused for that reason (got: " + why + ")");
    }
    // a sibling that repeats another sibling's id. NOTE the reason quoted here was corrected by
    // t-mtp-tree11: it is NOT "the op was asked for more rows than it was given" (both call sites
    // pass rows == TextConfig::token_domain >> top_l, so the launcher's effective == top_l and its
    // tail-repeat branch is unreachable from this tree). A repeated index today means the op's own
    // contract was broken, which is why it is still worth refusing.
    {
        MtpProposalRow twin = good;
        twin.ids[1]         = twin.ids[0];
        const std::string why = mtp_proposal_row_defect(twin, budget);
        check(!why.empty() && why.find("same id twice") != std::string::npos,
              "two siblings of one level cannot carry the same id (got: " + why + ")");
        // the same defect on a LATER level: level 1's siblings repeat each other. The spine of
        // every level is untouched, so the cross-check stays green and the duplicate check is the
        // one that has to fire.
        MtpProposalRow leaf_twin = good;
        leaf_twin.ids[3]         = leaf_twin.ids[2];
        const std::string later  = mtp_proposal_row_defect(leaf_twin, budget);
        check(!later.empty() && later.find("same id twice") != std::string::npos &&
                  later.find("level 1") != std::string::npos,
              "a repeated id on a later level is refused for that reason (got: " + later + ")");
        // CONTROL: a legal row with distinct siblings is NOT refused for this reason
        check(mtp_proposal_row_defect(good, budget).empty(),
              "CONTROL: a row with distinct siblings is not refused");
    }
    // a negative token id
    {
        MtpProposalRow negative = good;
        negative.ids[0]         = -13;
        negative.chain[0]       = -13; // keep the spine cross-check green so the token check fires
        const std::string why   = mtp_proposal_row_defect(negative, budget);
        check(!why.empty() && why.find("negative token id") != std::string::npos,
              "a negative node token is refused (got: " + why + ")");
    }
    // a narrow slot must not truncate silently -- the receiving half refuses it
    {
        FillMirror mirror;
        MtpTreePublishSlot narrow;
        narrow.columns        = &mirror.mtp_tree_columns;
        narrow.depths         = mirror.mtp_tree_depths.data();
        narrow.masks          = mirror.mtp_tree_masks.data();
        narrow.drafts         = mirror.mtp_drafts.data();
        narrow.array_capacity = 2; // the layout has budget + 1 columns
        narrow.draft_capacity = static_cast<std::uint32_t>(mirror.mtp_drafts.size());
        bool threw            = false;
        try {
            (void)fill_mtp_tree_round(good, budget, budget + 1U, narrow);
        } catch (const std::exception&) { threw = true; }
        check(threw, "a slot narrower than the layout is refused, not truncated");
        check_eq(mirror.mtp_tree_columns, 0U, "a refused fill wrote nothing");
    }
    // a frame narrower than the layout
    {
        FillMirror mirror;
        bool threw = false;
        try {
            (void)fill_mtp_tree_round(good, budget, budget /* one column short */,
                                      mtp_tree_publish_slot(mirror));
        } catch (const std::exception&) { threw = true; }
        check(threw, "a frame narrower than the layout is refused");
        check_eq(mirror.mtp_tree_columns, 0U, "the refused frame stayed unpublished");
    }
    // and the CONTROL for the two width refusals: the right width succeeds
    {
        FillMirror mirror;
        const std::uint32_t extent =
            fill_mtp_tree_round(good, budget, budget + 1U, mtp_tree_publish_slot(mirror));
        check_eq(extent, budget, "CONTROL: the same row fills the right frame");
        check_eq(mirror.mtp_tree_columns, budget + 1U, "CONTROL: and publishes the frame width");
    }
}

// ---------------------------------------------------------------------------------------------
// 7. the bridge (pre-fill) read is the same layout with one lane.
// ---------------------------------------------------------------------------------------------
void test_prefill_read() {
    const std::uint32_t paths = 2;
    const std::uint32_t steps = 2;
    std::vector<std::int32_t> ids(kDepthStride * steps, -1);
    std::vector<TokenId> chain(steps, -1);
    for (std::uint32_t s = 0; s < steps; ++s) {
        for (std::uint32_t i = 0; i < paths; ++i) {
            ids[mtp_tree_proposal_index(s, i, 1U, 0U, kDepthStride)] =
                static_cast<std::int32_t>(500 + 10 * s + i);
        }
        chain[s] = static_cast<TokenId>(500 + 10 * s);
    }
    const MtpProposalRow row =
        mtp_proposal_row_from_prefill(ids.data(), chain.data(), paths, steps, kDepthStride,
                                      kTokenDomain);
    check_eq(row.ids[0], 500, "prefill read: depth 0 rank 0 is the bridge's own draft");
    check_eq(row.ids[1], 501, "prefill read: depth 0 rank 1 follows it");
    check_eq(row.ids[2], 510, "prefill read: depth 1 rank 0 is the next depth's block");
    check_eq(mtp_proposal_ready_steps(row), steps, "prefill read: the spine matches the bridge");

    // RED: the bridge's chain draft disagrees with the extracted spine.
    std::vector<TokenId> bad_chain = chain;
    bad_chain[1]                   = 999;
    const MtpProposalRow bad =
        mtp_proposal_row_from_prefill(ids.data(), bad_chain.data(), paths, steps, kDepthStride,
                                      kTokenDomain);
    check_eq(mtp_proposal_ready_steps(bad), 1U, "RED: the prefill read notices a bad spine");

    // and the control: the bridge's own seeding, published.
    bool threw          = false;
    std::uint32_t extent = 0;
    FillMirror mirror;
    try {
        extent = fill_mtp_tree_round(row, paths * steps, paths * steps + 1U,
                                     mtp_tree_publish_slot(mirror));
    } catch (const std::exception&) { threw = true; }
    check(!threw, "CONTROL: the bridge row publishes");
    check_eq(extent, paths * steps, "CONTROL: the bridge row's extent is its node count");
    check_eq(mirror.mtp_tree_columns, paths * steps + 1U, "CONTROL: the bridge row fills the frame");
}

// ---------------------------------------------------------------------------------------------
// 8. THE ROW RANGE RULE -- the one single-token correctness statement this layer can make, and
//    therefore the only one of the audit's four counterexample classes that is refused here.
//    Every case states the reason it must fire, and every case has a CONTROL that stays green.
// ---------------------------------------------------------------------------------------------
void test_row_range_rule() {
    const std::uint32_t paths  = 2;
    const std::uint32_t steps  = 7;
    const std::uint32_t budget = paths * steps;
    const EgressImage image(paths, steps, 0);
    const MtpProposalRow good = mtp_proposal_row_from_egress(
        image.ids.data(), image.chain.data(), kLanes, 0, paths, steps, kDepthStride, kChainStride,
        kTokenDomain);
    check(mtp_proposal_row_defect(good, budget).empty(), "CONTROL: the in-range row is accepted");

    const auto refuses_naming = [&](const MtpProposalRow& row, const std::string& needle,
                                    const std::string& what) {
        const std::string why = mtp_proposal_row_defect(row, budget);
        check(!why.empty() && why.find(needle) != std::string::npos,
              what + " (got: " + why + ")");
    };

    // (a) exactly one past the last addressable row, on the SPINE (rank 0): the id is a row the
    //     proposal head does not have. chain[0] is patched too, so ONLY the range rule can fire.
    {
        MtpProposalRow out = good;
        out.ids[0]         = static_cast<TokenId>(kTokenDomain);
        out.chain[0]       = out.ids[0];
        refuses_naming(out, "token domain", "an id one past the head's last row is refused");
    }
    // (b) the same on a NON-spine rank: nothing else in this gate looks at rank > 0 at all, so
    //     this is the case the audit's C2/C3 payloads exercise.
    {
        MtpProposalRow out = good;
        out.ids[1]         = static_cast<TokenId>(kTokenDomain + 1);
        refuses_naming(out, "sibling 1", "an out-of-range id on rank 1 is refused and named");
        refuses_naming(out, "token domain", "an out-of-range id on rank 1 is refused for the range");
    }
    // (c) the same on a LATER level, so the rule is not level-0-only.
    {
        MtpProposalRow out = good;
        out.ids[3]         = static_cast<TokenId>(kTokenDomain + 1);
        refuses_naming(out, "level 1", "an out-of-range id on a later level is refused");
    }
    // (d) INT32_MAX -- the audit's C2 payload value.
    {
        MtpProposalRow out = good;
        out.ids[1]         = std::numeric_limits<std::int32_t>::max();
        refuses_naming(out, "token domain", "INT32_MAX is refused as outside the head's rows");
    }
    // (e) CONTROL: the LAST addressable row is inside the range and must be accepted, otherwise the
    //     rule would be "refuse high ids" rather than a range check.
    {
        MtpProposalRow out = good;
        out.ids[1]         = static_cast<TokenId>(kTokenDomain - 1U);
        check(mtp_proposal_row_defect(out, budget).empty(),
              "CONTROL: the head's last addressable row is accepted");
    }
    // (f) FAIL CLOSED: a row that does not state the domain is refused, not silently unchecked.
    {
        MtpProposalRow unstated = good;
        unstated.token_domain   = 0;
        refuses_naming(unstated, "does not state", "a row with no stated token domain is refused");
    }
    // (g) and the readers refuse to build such a row in the first place.
    {
        bool threw = false;
        try {
            (void)mtp_proposal_row_from_egress(image.ids.data(), image.chain.data(), kLanes, 0,
                                               paths, steps, kDepthStride, kChainStride, 0);
        } catch (const std::exception&) { threw = true; }
        check(threw, "the egress read refuses an unstated token domain");
        threw = false;
        try {
            (void)mtp_proposal_row_from_prefill(image.ids.data(), image.chain.data(), paths, steps,
                                                kDepthStride, 0);
        } catch (const std::exception&) { threw = true; }
        check(threw, "the bridge read refuses an unstated token domain");
    }
}

// ---------------------------------------------------------------------------------------------
// 9. THE TWO VERDICTS THIS GATE DOES NOT PRETEND TO CHANGE. Both of these are counterexample
//    classes an independent audit raised against the three checks t-mtp-tree10 added. They are
//    pinned as LEGAL here, deliberately, with the reason written into the test, so that a later
//    "fix" which bans them has to fail THIS test and read the note above mtp_proposal_row_defect()
//    before it can land. The note's argument, in one line each:
//      * a node whose token equals its PARENT's token: the MTP loop feeds depth s-1's prediction
//        back as depth s's input, so a repeated argmax is an ordinary prediction; in a repetition
//        loop -- the highest-acceptance regime for speculative decoding -- the spine repeats at
//        EVERY level. A ban would be a false-positive generator on a legal path.
//      * an in-domain id at rank >= 1 that no extraction produced: undecidable at this layer (the
//        gate holds no score and no row), and the shipped chain path trusts the very same device
//        array with no check at all.
// ---------------------------------------------------------------------------------------------
void test_named_undecidables_are_legal() {
    const auto row_of = [](std::uint32_t paths, std::uint32_t steps, const std::vector<TokenId>& ids) {
        MtpProposalRow row;
        row.paths        = paths;
        row.steps        = steps;
        row.ids          = ids;
        row.token_domain = kTokenDomain;
        row.chain.resize(steps);
        for (std::uint32_t s = 0; s < steps; ++s) {
            row.chain[s] = ids[static_cast<std::size_t>(s) * paths];
        }
        return row;
    };

    // C1: a node whose token equals its PARENT's token (2 paths x 2 depths).
    const MtpProposalRow c1 = row_of(2, 2, {7, 8, 7, 9});
    check(mtp_proposal_row_defect(c1, 4).empty(),
          "PINNED LEGAL: a child token equal to its parent's is an ordinary repeat prediction");

    // C4: seven levels whose spine is the same token -- a repetition lattice.
    std::vector<TokenId> ids(14);
    for (std::uint32_t s = 0; s < 7; ++s) {
        ids[static_cast<std::size_t>(s) * 2 + 0] = 7;
        ids[static_cast<std::size_t>(s) * 2 + 1] = static_cast<TokenId>(100 + s);
    }
    const MtpProposalRow c4 = row_of(2, 7, ids);
    check(mtp_proposal_row_defect(c4, 14).empty(),
          "PINNED LEGAL: a repetition lattice is a legal lattice (highest-acceptance regime)");

    // ... and the claim "that is a chain published as a tree" is FALSE: the 14 draft sequences the
    // lattice publishes are 14 DISTINCT prefixes, which is what makes it a tree. The parent map
    // comes from the producing half's own layout, not from a second copy of the rule.
    const MtpSiblingLattice lattice = mtp_sibling_lattice_from_row(c4, 14);
    const MtpTreeColumns cols       = mtp_tree_columns_from_siblings(lattice, 14);
    check_eq(cols.columns(), 15U, "the repetition lattice still fills a 15-column frame");
    std::vector<std::vector<TokenId>> sequences;
    for (std::uint32_t j = 1; j < cols.columns(); ++j) {
        std::vector<TokenId> seq;
        for (std::uint32_t i = 1; i <= j; ++i) {
            if (((cols.masks[j] >> i) & 1ULL) != 0) { seq.push_back(cols.tokens[i - 1]); }
        }
        sequences.push_back(seq);
    }
    check_eq(sequences.size(), 14U, "one draft sequence per tree node");
    std::size_t distinct = sequences.size();
    for (std::size_t a = 0; a < sequences.size(); ++a) {
        for (std::size_t b = a + 1; b < sequences.size(); ++b) {
            if (sequences[a] == sequences[b]) { --distinct; }
        }
    }
    check_eq(distinct, 14U,
             "all 14 published draft sequences are distinct: the tree is a tree, not a chain");

    // and the reference contract's own layout functions are token-blind: they must accept the
    // repetition lattice's TREE (the equivalence is the shape, and the shape is what they check).
    const ops::ddtree::Tree ref = sibling_equivalent_tree(2, 7);
    check_eq(ops::ddtree::column_depth(ref).size(), 15U, "the reference layout has 15 columns");
    check_eq(mtp_tree_layout_defect(cols, 15).empty() ? 0 : 1, 0,
             "the receiving seam accepts the repetition lattice's layout");
}

} // namespace

int main() {
    test_contract_fingerprint();
    test_index_formula_is_load_bearing();
    test_read_and_ready_steps();
    test_head_guard();
    test_fill_for_legal_shapes();
    test_refusals();
    test_prefill_read();
    test_row_range_rule();
    test_named_undecidables_are_legal();

    std::printf("mtp_tree_proposal_test: %d/%d checks passed %s\n", checks - failed, checks,
                failed == 0 ? "OK" : "FAILED");
    return failed == 0 ? 0 : 1;
}
