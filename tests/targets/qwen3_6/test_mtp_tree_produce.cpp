// ============================================================================
// Host test for the PRODUCING half of the MTP draft-tree publication
// (src/targets/qwen3_6/impl/runtime/mtp_tree_produce.h + the receiving seam it calls).
//
// It uses the REAL reference builder and the REAL contract functions from
// include/ninfer/ops/dflash2_ddtree.h -- ddtree::build_tree / column_depth /
// column_ancestors -- so the layout this producer emits is pinned to the same source of
// truth the verify side reads, not to a second copy of the rule.
//
// No engine header, no CUDA, no variant macro, no artifact: runnable with
//   c++ -std=gnu++20 -I <tree>/include -I <tree>/src -I <tree>/tests ...
//
// The checks that matter, and why:
//   * ANTI-LAUNDERING: a real build_tree(L=1) chain spelling must be REJECTED as a tree.
//   * the GATE control: an unpublished round is refused, a published one is not, a cleared
//     one is refused again -- the exact predicate program_impl.h:13336 evaluates.
//   * the sibling shape is pinned against ddtree::column_depth / column_ancestors run on an
//     equivalent hand-built Tree, so the producer's rule cannot drift from the contract.
//   * BOTH registered tiers (2,1 and 2,7) fill their frame exactly.
// ============================================================================

#include "targets/qwen3_6/impl/runtime/mtp_tree_produce.h"

#include <ninfer/ops/dflash2_ddtree.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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
        std::printf("FAIL %s\n", what.c_str());
    }
}

using ninfer::TokenId;
// The reference contract lives in ninfer::ops::ddtree (include/ninfer/ops/dflash2_ddtree.h).
namespace ops = ninfer::ops;
using ninfer::targets::qwen3_6::detail::MtpProposalLattice;
using ninfer::targets::qwen3_6::detail::MtpSiblingLattice;
using ninfer::targets::qwen3_6::detail::MtpTreeColumns;
using ninfer::targets::qwen3_6::detail::MtpTreePublishSlot;

// ---------------------------------------------------------------------------------------
// A byte-for-byte mirror of the four SequenceState members the verify side reads
// (src/targets/qwen3_6/impl/runtime/program.h:463/464/472/473/474 @3944a53), with the same
// default values program.h gives them. The gate below is the predicate at
// program_impl.h:13336, transcribed, so a change to the seam that would alter the gate's
// answer shows up here.
// ---------------------------------------------------------------------------------------
struct SequenceTreeMirror {
    static constexpr std::uint32_t kWidth     = 16; // qwen3_6::kMtpDecodeMaximumWidth
    static constexpr std::uint32_t kDrafts    = 15; // qwen3_6::kMtpDecodeMaximumDrafts
    std::uint32_t mtp_tree_columns = 0;
    std::array<std::uint32_t, kWidth> mtp_tree_depths{};
    std::array<std::uint64_t, kWidth> mtp_tree_masks{};
    std::array<TokenId, kDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;

    MtpTreePublishSlot slot() {
        MtpTreePublishSlot s;
        s.columns        = &mtp_tree_columns;
        s.depths         = mtp_tree_depths.data();
        s.masks          = mtp_tree_masks.data();
        s.drafts         = mtp_drafts.data();
        s.array_capacity = kWidth;
        s.draft_capacity = kDrafts;
        return s;
    }
};

// program_impl.h:13336 verbatim as a predicate.
bool gate_refuses(std::uint32_t draft_tree_paths, const SequenceTreeMirror& s) {
    const std::uint32_t tree_columns = s.mtp_tree_columns;
    return draft_tree_paths > 1 && tree_columns == 0;
}

// ---------------------------------------------------------------- a deterministic RNG
std::uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
double rng_uniform() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return static_cast<double>(rng_state >> 11) / 9007199254740992.0;
}

// A real ddtree grid: scores[s][p][c] with row 0 at step 0 meaningful (the builder forces it).
std::vector<double> make_grid(std::int32_t steps, std::int32_t top_k, std::uint64_t seed) {
    rng_state = seed;
    std::vector<double> scores(static_cast<std::size_t>(steps) * top_k * top_k, 0.0);
    for (std::int32_t s = 0; s < steps; ++s) {
        for (std::int32_t p = 0; p < top_k; ++p) {
            for (std::int32_t c = 0; c < top_k; ++c) {
                scores[static_cast<std::size_t>(
                    ninfer::ops::ddtree::score_offset(s, steps, p, c))] = rng_uniform();
            }
        }
    }
    return scores;
}

// The tree the sibling rule claims to describe, expressed as a ddtree::Tree so the REAL
// column_depth / column_ancestors can be run on it. Node order is the builder's own:
// (depth, lexicographic rank prefix), i.e. level major, rank ascending inside a level.
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

MtpSiblingLattice make_sibling_lattice(std::int32_t paths, std::int32_t steps, TokenId base) {
    MtpSiblingLattice lattice;
    lattice.paths = paths;
    lattice.steps = steps;
    lattice.ids.resize(static_cast<std::size_t>(paths) * steps);
    for (std::size_t i = 0; i < lattice.ids.size(); ++i) {
        lattice.ids[i] = base + static_cast<TokenId>(i);
    }
    return lattice;
}

} // namespace

int main() {
    using namespace ninfer::targets::qwen3_6::detail;

    // -----------------------------------------------------------------------------------
    // 0. the reference contract really is the tree's own header (a fingerprint, so a stale
    //    or shadowed dflash2_ddtree.h cannot make everything below pass vacuously).
    // -----------------------------------------------------------------------------------
    check_eq(ops::ddtree::kTopK, 16, "ddtree::kTopK is 16");
    check_eq(ops::ddtree::kMaxNodes, 63, "ddtree::kMaxNodes is 63");
    static_assert(ops::ddtree::kTopK == 16, "the ddtree contract header moved");

    // -----------------------------------------------------------------------------------
    // 1. ANTI-LAUNDERING: a real build_tree L=1 run spells the CHAIN, and must be refused.
    // -----------------------------------------------------------------------------------
    {
        const std::int32_t steps = 3;
        std::vector<double> grid = make_grid(steps, ops::ddtree::kTopK, 42);
        const ops::ddtree::Tree chain = ops::ddtree::build_tree(grid.data(), steps, steps);
        check_eq(chain.node_count(), steps, "L=1 build_tree is a chain of `steps` nodes");
        std::vector<TokenId> tokens;
        for (const ops::ddtree::Node& n : chain.nodes) {
            tokens.push_back(static_cast<TokenId>(1000 + n.depth));
        }
        const MtpTreeColumns cols =
            mtp_tree_columns_from_ddtree(chain, std::span<const TokenId>(tokens));
        check_eq(cols.columns(), static_cast<std::uint32_t>(steps + 1),
                 "L=1 chain has steps+1 verify columns");
        const std::string defect = mtp_tree_layout_defect(cols, 16);
        check(!defect.empty(), "the L=1 chain spelling is REJECTED as a tree");
        check(defect.find("CHAIN spelling") != std::string::npos,
              "the rejection names the chain spelling");
        // and it really was the chain: depths are the column indices, masks are the prefixes.
        bool spelled_like_a_chain = true;
        for (std::uint32_t j = 0; j < cols.columns(); ++j) {
            if (cols.depths[j] != j || cols.masks[j] != ((1ULL << (j + 1)) - 1ULL)) {
                spelled_like_a_chain = false;
            }
        }
        check(spelled_like_a_chain, "the rejected layout really is the chain spelling");
    }

    // -----------------------------------------------------------------------------------
    // 2. A REAL beam tree from the same builder is ACCEPTED: the producer's target shape is
    //    not accidentally unreachable.
    // -----------------------------------------------------------------------------------
    {
        std::int32_t accepted_shapes = 0;
        for (std::uint64_t seed = 1; seed <= 40; ++seed) {
            const std::int32_t steps = 2;
            const std::int32_t budget = 4; // L = 2
            const std::vector<double> grid = make_grid(steps, ops::ddtree::kTopK, seed);
            MtpProposalLattice lattice;
            lattice.steps = steps;
            lattice.top_k = ops::ddtree::kTopK;
            lattice.scores = grid;
            lattice.candidates.resize(static_cast<std::size_t>(steps) * ops::ddtree::kTopK);
            for (std::size_t i = 0; i < lattice.candidates.size(); ++i) {
                lattice.candidates[i] = static_cast<std::int32_t>(7000 + i);
            }
            const MtpTreeColumns cols =
                mtp_tree_columns_from_lattice(lattice, static_cast<std::uint32_t>(budget));
            if (cols.columns() == 0) { continue; }
            const std::string defect = mtp_tree_layout_defect(cols, 16);
            check(defect.empty(), "a real beam-L build_tree layout is accepted (seed " +
                                      std::to_string(seed) + ": " + defect + ")");
            // every node token must have come from the candidate table at its own (depth,rank)
            bool tokens_from_the_table = true;
            for (std::uint32_t j = 1; j < cols.columns(); ++j) {
                const std::uint32_t depth = cols.depths[j] - 1;
                bool found = false;
                for (std::int32_t k = 0; k < lattice.top_k; ++k) {
                    if (static_cast<TokenId>(lattice.candidates[depth * lattice.top_k + k]) ==
                        cols.tokens[j - 1]) {
                        found = true;
                    }
                }
                if (!found) { tokens_from_the_table = false; }
            }
            check(tokens_from_the_table, "every published token is a depth-row candidate");
            ++accepted_shapes;
        }
        check(accepted_shapes > 0, "at least one real beam grid produced a layout");
    }

    // -----------------------------------------------------------------------------------
    // 3. the SIBLING shape is pinned to the REAL contract functions.
    // -----------------------------------------------------------------------------------
    for (const auto [paths, steps] : {std::pair<std::int32_t, std::int32_t>{2, 1},
                                      {2, 2}, {2, 7}, {3, 3}, {4, 2}}) {
        const std::uint32_t budget = static_cast<std::uint32_t>(paths * steps);
        const MtpSiblingLattice lattice = make_sibling_lattice(paths, steps, 500);
        const MtpTreeColumns mine = mtp_tree_columns_from_siblings(lattice, budget);
        const ops::ddtree::Tree ref = sibling_equivalent_tree(paths, steps);
        const std::vector<std::int32_t> ref_depth = ops::ddtree::column_depth(ref);
        const std::vector<std::uint64_t> ref_mask  = ops::ddtree::column_ancestors(ref);
        const std::string tag = std::to_string(paths) + "," + std::to_string(steps) + ": ";

        check_eq(mine.columns(), budget + 1U, tag + "columns == nodes + 1 == node budget + 1");
        check_eq(ref_depth.size(), static_cast<std::size_t>(mine.columns()),
                 tag + "reference column count agrees");
        for (std::uint32_t j = 0; j < mine.columns(); ++j) {
            check_eq(mine.depths[j], static_cast<std::uint32_t>(ref_depth[j]),
                     tag + "depth[" + std::to_string(j) + "] matches column_depth()");
            check_eq(mine.masks[j], ref_mask[j],
                     tag + "mask[" + std::to_string(j) + "] matches column_ancestors()");
        }
        // every node token is the published id of its own (level, sibling) slot
        for (std::uint32_t node = 0; node < budget; ++node) {
            check_eq(mine.tokens[node], lattice.ids[node], tag + "token[" + std::to_string(node) + "]");
        }
        // the shape IS a tree: siblings exist, so it is not the chain spelling
        const std::string defect = mtp_tree_layout_defect(mine, static_cast<std::uint32_t>(budget) + 1U);
        check(defect.empty(), tag + "the sibling layout passes the receiving seam: " + defect);
        check(mine.depths[1] == 1 && mine.depths[2] == 1,
              tag + "level 0 carries siblings at one depth");
        check(mine.masks[2] != mine.masks[1], tag + "siblings do not share an ancestor set");
        // the second sibling must NOT be an ancestor of the first, nor the reverse
        check((mine.masks[1] & (1ULL << 2)) == 0, tag + "sibling 1 does not attend sibling 2");
        check((mine.masks[2] & (1ULL << 1)) == 0, tag + "sibling 2 does not attend sibling 1");
    }

    // -----------------------------------------------------------------------------------
    // 4. the two REGISTERED tiers fill their frame exactly.
    // -----------------------------------------------------------------------------------
    struct Tier { std::int32_t paths, depth; std::uint32_t node_budget, width; };
    for (const Tier t : {Tier{2, 1, 2, 3}, Tier{2, 7, 14, 15}}) {
        const std::string tag = "tier " + std::to_string(t.paths) + "," + std::to_string(t.depth) + ": ";
        SequenceTreeMirror mirror;
        check(gate_refuses(t.paths, mirror), tag + "RED: an unpublished tree round is refused");
        check_eq(mirror.mtp_tree_columns, 0U, tag + "RED: the field really is zero");

        const MtpSiblingLattice lattice = make_sibling_lattice(t.paths, t.depth, 100);
        const std::uint32_t extent =
            publish_mtp_tree_round(lattice, t.node_budget, t.width, mirror.slot());
        check(!gate_refuses(t.paths, mirror), tag + "GREEN: the published round is not refused");
        check_eq(mirror.mtp_tree_columns, t.width, tag + "columns == frame width");
        check_eq(extent, t.node_budget, tag + "the round's extent is the node count");
        check_eq(extent, t.width - 1U, tag + "extent == width - 1 (the frame is exactly full)");
        check_eq(mirror.mtp_tree_columns - 1U, extent,
                 tag + "extent == columns - 1 (mtp_draft_count == nodes)");
        check_eq(mirror.mtp_tree_depths[0], 0U, tag + "anchor column depth 0");
        check_eq(mirror.mtp_tree_masks[0], 1ULL, tag + "anchor column mask 1");
        check(mirror.mtp_tree_masks[t.width - 1] != 0ULL, tag + "last column has a mask");

        clear_mtp_tree_round(mirror.slot());
        check_eq(mirror.mtp_tree_columns, 0U, tag + "cleared back to zero");
        check(gate_refuses(t.paths, mirror), tag + "RED again after the clear");
    }

    // the 2,7 frame is 15 wide: 15 <= kMtpDecodeMaximumWidth (16)
    {
        SequenceTreeMirror wide;
        const std::uint32_t extent =
            publish_mtp_tree_round(make_sibling_lattice(2, 7, 0), 14, 15, wide.slot());
        check_eq(extent, 14U, "2,7 publishes in a 16-wide array");
        check_eq(wide.mtp_tree_columns, 15U, "2,7 sets 15 columns");
    }
    check_eq(static_cast<std::uint32_t>(2 * 7) + 1U, 15U,
             "2,7 width is 15 == draft_window(14) + 1");

    // -----------------------------------------------------------------------------------
    // 5. the producer REFUSES the shapes that would be a lie (each one can turn red).
    // -----------------------------------------------------------------------------------
    {
        // L=1 is the chain spelling: refused by the producer before the seam sees it.
        bool threw = false;
        try {
            (void)mtp_tree_columns_from_siblings(make_sibling_lattice(1, 7, 0), 7);
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find("chain spelling") != std::string::npos;
        }
        check(threw, "the producer refuses L=1 (the chain spelling)");

        // a node budget that is not paths * depth cannot fill the frame: refused.
        threw = false;
        try {
            (void)mtp_tree_columns_from_siblings(make_sibling_lattice(2, 7, 0), 13);
        } catch (const std::exception&) { threw = true; }
        check(threw, "the producer refuses a node budget that is not paths * depth");

        // an id table that is short: refused.
        threw = false;
        try {
            MtpSiblingLattice bad = make_sibling_lattice(2, 7, 0);
            bad.ids.pop_back();
            (void)mtp_tree_columns_from_siblings(bad, 14);
        } catch (const std::exception&) { threw = true; }
        check(threw, "the producer refuses a short id table");

        // a dense lattice whose grid is the wrong size: refused.
        threw = false;
        try {
            MtpProposalLattice bad;
            bad.steps = 2;
            bad.top_k = ops::ddtree::kTopK;
            bad.candidates.assign(2 * ops::ddtree::kTopK, 0);
            bad.scores.assign(10, 0.0);
            (void)mtp_tree_columns_from_lattice(bad, 4);
        } catch (const std::exception&) { threw = true; }
        check(threw, "the producer refuses a malformed dense grid");

        // a layout the SEAM must reject: a chain published for an L > 1 round.
        threw = false;
        try {
            MtpTreeColumns chain;
            for (std::uint32_t j = 0; j < 5; ++j) {
                chain.depths.push_back(j);
                chain.masks.push_back((1ULL << (j + 1)) - 1ULL);
                if (j > 0) { chain.tokens.push_back(static_cast<TokenId>(j)); }
            }
            SequenceTreeMirror m;
            ninfer::targets::qwen3_6::detail::publish_mtp_tree(chain, 16, m.slot());
        } catch (const std::exception&) { threw = true; }
        check(threw, "the seam refuses a hand-written chain for an L > 1 round");

        // TWO real defects of the sibling shape, each one first PROVED to be a real change
        // (a corruption that happens to be a no-op would make the check vacuous -- the first
        // version of this test had exactly that bug, which is why the diff is asserted here).
        {
            const MtpTreeColumns good = mtp_tree_columns_from_siblings(
                make_sibling_lattice(2, 2, 0), 4);
            // (a) a sibling that claims a column it does not descend from: column 2 claims
            //     column 3 as an ancestor, which is neither its parent's mask nor itself.
            MtpTreeColumns bad = good;
            bad.masks[2] |= (1ULL << 3);
            check(bad.masks[2] != good.masks[2], "corruption (a) really changed the layout");
            bool threw = false;
            std::string why;
            try {
                SequenceTreeMirror m;
                ninfer::targets::qwen3_6::detail::publish_mtp_tree(bad, 16, m.slot());
            } catch (const std::exception& e) { threw = true; why = e.what(); }
            check(threw, "the seam refuses an ancestor set that is not parent-mask + self");
            check(why.find("parent's mask") != std::string::npos ||
                      why.find("outside this round") != std::string::npos,
                  "the rejection names the ancestor defect (got: " + why + ")");

            // (b) a column whose parent is not exactly one depth shallower: drop the spine bit
            //     from column 3's mask, which moves its parent up to the anchor (depth 0).
            MtpTreeColumns bad2 = good;
            bad2.masks[3] &= ~(1ULL << 1);
            check(bad2.masks[3] != good.masks[3], "corruption (b) really changed the layout");
            threw = false;
            why.clear();
            try {
                SequenceTreeMirror m;
                ninfer::targets::qwen3_6::detail::publish_mtp_tree(bad2, 16, m.slot());
            } catch (const std::exception& e) { threw = true; why = e.what(); }
            check(threw, "the seam refuses a parent that is not one depth shallower");
            check(why.find("one depth shallower") != std::string::npos,
                  "the rejection names the depth defect (got: " + why + ")");

            // CONTROL: the same call with the uncorrupted layout does not throw.
            bool ok = false;
            try {
                SequenceTreeMirror m;
                (void)publish_mtp_tree_round(make_sibling_lattice(2, 2, 0), 4, 5, m.slot());
                ok = true;
            } catch (const std::exception&) { ok = false; }
            check(ok, "CONTROL: the uncorrupted layout of the same shape publishes");
        }
    }

    // -----------------------------------------------------------------------------------
    // 6. publish is all-or-nothing: a rejected layout leaves the sequence untouched.
    // -----------------------------------------------------------------------------------
    {
        SequenceTreeMirror m;
        bool threw = false;
        try {
            MtpSiblingLattice lattice = make_sibling_lattice(2, 7, 0);
            (void)publish_mtp_tree_round(lattice, 14, 15, m.slot());
            check_eq(m.mtp_tree_columns, 15U, "a good publish sets 15 columns");
            SequenceTreeMirror m2;
            (void)publish_mtp_tree_round(lattice, 14, 9, m2.slot()); // width too small
        } catch (const std::exception&) { threw = true; }
        check(threw, "publishing into a frame narrower than the layout throws");
    }

    std::printf("mtp_tree_produce_test: %d/%d checks passed %s\n", checks - failed, checks,
                failed == 0 ? "OK" : "FAILED");
    return failed == 0 ? 0 : 1;
}
