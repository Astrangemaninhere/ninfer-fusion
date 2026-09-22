#include "ninfer/ops/dflash2_ddtree.h"
#include "ops/dflash2_ddtree_golden.h"

// Host-only test: it deliberately includes no engine, CUDA or artifact header, so
// it also compiles and runs under a plain
//   g++ -std=c++20 -fno-fast-math -ffp-contract=off -I include -I tests
// without a GPU. Under CMake it is registered as an op test (tests/CMakeLists.txt)
// with exactly those determinism flags.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using ninfer::ops::ddtree::accepted_length;
using ninfer::ops::ddtree::accepted_node;
using ninfer::ops::ddtree::build_tree;
using ninfer::ops::ddtree::column_ancestors;
using ninfer::ops::ddtree::column_depth;
using ninfer::ops::ddtree::column_rope_offsets;
using ninfer::ops::ddtree::kMaxNodes;
using ninfer::ops::ddtree::kRoot;
using ninfer::ops::ddtree::kTopK;
using ninfer::ops::ddtree::Node;
using ninfer::ops::ddtree::score_offset;
using ninfer::ops::ddtree::single_chain;
using ninfer::ops::ddtree::Tree;
using ninfer::ops::ddtree::tree_paths;

int checks   = 0;
int failures = 0;

void report(const std::string& what, bool ok, const std::string& detail = std::string()) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cout << "FAIL: " << what << (detail.empty() ? "" : " -- " + detail) << "\n";
    }
}

template <typename T>
void check_eq(const std::string& what, const T& got, const T& want) {
    ++checks;
    if (got == want) { return; }
    ++failures;
    std::string detail;
    if constexpr (std::is_arithmetic_v<T>) {
        detail = "got " + std::to_string(got) + " want " + std::to_string(want);
    }
    std::cout << "FAIL: " << what << (detail.empty() ? "" : " -- " + detail) << "\n";
}

std::string to_string(const std::vector<std::int32_t>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i != 0) { out += ","; }
        out += std::to_string(v[i]);
    }
    return out + "]";
}

std::string to_string(const std::vector<std::vector<std::int32_t>>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i != 0) { out += ","; }
        out += to_string(v[i]);
    }
    return out + "]";
}

std::vector<std::int32_t> path_of(const Tree& tree, std::int32_t end) {
    std::vector<std::int32_t> path;
    for (std::int32_t node = end; node != kRoot; node = tree.nodes[node].parent) {
        path.push_back(tree.nodes[node].rank);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// 1) Differential test against the reference builder's golden vectors.
void test_golden() {
    for (const auto& c : ninfer::test::ddtree_golden::make_cases()) {
        const std::string tag = std::string("golden/") + c.name;
        const Tree tree       = build_tree(c.scores.data(), c.steps, c.node_budget);

        check_eq(tag + " chain", single_chain(c.scores.data(), c.steps), c.chain);
        check_eq(tag + " node_count", tree.node_count(),
                 static_cast<int>(c.node_rank.size()));
        check_eq(tag + " path_count", static_cast<int>(tree.path_ends.size()),
                 static_cast<int>(c.paths.size()));

        std::vector<std::vector<std::int32_t>> got_paths = tree_paths(tree);
        std::sort(got_paths.begin(), got_paths.end());
        report(tag + " path_set", got_paths == c.paths, to_string(got_paths));

        std::vector<std::int32_t> got_rank, got_depth, got_parent;
        for (const Node& node : tree.nodes) {
            got_rank.push_back(node.rank);
            got_depth.push_back(node.depth);
            got_parent.push_back(node.parent);
        }
        report(tag + " node_rank", got_rank == c.node_rank, to_string(got_rank));
        report(tag + " node_depth", got_depth == c.node_depth, to_string(got_depth));
        report(tag + " node_parent", got_parent == c.node_parent, to_string(got_parent));

        for (std::size_t t = 0; t < c.truths.size(); ++t) {
            const std::vector<std::int32_t>& truth = c.truths[t];
            const std::int32_t len                 = static_cast<std::int32_t>(truth.size());
            const std::string ttag = tag + " truth" + std::to_string(t);
            check_eq(ttag + " accepted_length", accepted_length(tree, truth.data(), len),
                     c.accepted_length[t]);
            check_eq(ttag + " accepted_node", accepted_node(tree, truth.data(), len),
                     c.accepted_node[t]);
        }
    }
}

// 2) Structural invariants that must hold for every tree.
void check_invariants(const std::string& tag, const Tree& tree) {
    const int n = tree.node_count();
    report(tag + " budget", n <= tree.node_budget,
           std::to_string(n) + " > " + std::to_string(tree.node_budget));
    report(tag + " at_least_one_chain", n >= tree.steps, std::to_string(n));

    // The beam width the reference derives from the budget.
    const int l =
        std::max(1, std::min(kTopK, static_cast<int>(tree.node_budget) / tree.steps));

    int roots = 0;
    for (int i = 0; i < n; ++i) {
        const Node& node = tree.nodes[i];
        if (node.depth == 0) {
            ++roots;
            check_eq(tag + " root_parent", node.parent, kRoot);
        } else {
            report(tag + " parent_before_child", node.parent >= 0 && node.parent < i,
                   std::to_string(i) + " -> " + std::to_string(node.parent));
            if (node.parent >= 0 && node.parent < n) {
                report(tag + " parent_depth", tree.nodes[node.parent].depth == node.depth - 1);
            }
        }
        report(tag + " rank_range", node.rank >= 0 && node.rank < kTopK);
        report(tag + " depth_range", node.depth >= 0 && node.depth < tree.steps);
    }
    report(tag + " roots_within_beam", roots >= 1 && roots <= l,
           std::to_string(roots) + " not in [1," + std::to_string(l) + "]");

    // Prefix closure: every node's parent chain reaches a depth-0 root, and every
    // depth 0 <= d < steps is populated (each kept path spans the whole block).
    std::vector<int> per_depth(static_cast<std::size_t>(tree.steps), 0);
    for (int i = 0; i < n; ++i) {
        int node = i, guard = 0;
        while (tree.nodes[node].parent != kRoot && guard++ <= n) {
            node = tree.nodes[node].parent;
        }
        report(tag + " prefix_closure", tree.nodes[node].depth == 0);
        ++per_depth[static_cast<std::size_t>(tree.nodes[i].depth)];
    }
    for (int d = 0; d < tree.steps; ++d) {
        report(tag + " depth_populated", per_depth[static_cast<std::size_t>(d)] >= 1,
               std::to_string(d));
    }

}

void test_invariants_on_golden() {
    for (const auto& c : ninfer::test::ddtree_golden::make_cases()) {
        const Tree tree = build_tree(c.scores.data(), c.steps, c.node_budget);
        check_invariants(std::string("inv/") + c.name, tree);

        std::vector<std::vector<std::int32_t>> full_depth;
        for (std::int32_t end : tree.path_ends) {
            const std::vector<std::int32_t> path = path_of(tree, end);
            report(std::string("inv/") + c.name + " path_depth",
                   static_cast<int>(path.size()) == tree.steps, std::to_string(path.size()));
            full_depth.push_back(path);
        }
        report(std::string("inv/") + c.name + " chain_is_a_path",
               std::find(full_depth.begin(), full_depth.end(), c.chain) != full_depth.end());

        // Verify-side layout: column 0 is the anchor, node column i + 1 has depth
        // node.depth + 1, rope offsets equal the depth profile, and the ancestor mask
        // of a node is exactly its parent's mask plus its own column (never a
        // sibling).
        const std::vector<std::int32_t> depth   = column_depth(tree);
        const std::vector<std::int32_t> offsets = column_rope_offsets(tree);
        const std::vector<std::uint64_t> mask   = column_ancestors(tree);
        report(std::string("inv/") + c.name + " columns",
               static_cast<int>(depth.size()) == tree.node_count() + 1);
        check_eq(std::string("inv/") + c.name + " anchor_depth", depth[0],
                 static_cast<std::int32_t>(0));
        report(std::string("inv/") + c.name + " rope_offsets", depth == offsets);
        check_eq(std::string("inv/") + c.name + " anchor_mask", mask[0],
                 static_cast<std::uint64_t>(1));
        for (int i = 0; i < tree.node_count(); ++i) {
            const std::uint64_t own = 1ULL << static_cast<unsigned>(i + 1);
            const std::uint64_t parent =
                tree.nodes[i].parent == kRoot
                    ? 1ULL
                    : mask[static_cast<std::size_t>(tree.nodes[i].parent + 1)];
            check_eq(std::string("inv/") + c.name + " node_depth_plus_one",
                     depth[static_cast<std::size_t>(i + 1)],
                     static_cast<std::int32_t>(tree.nodes[i].depth + 1));
            report(std::string("inv/") + c.name + " mask_self",
                   (mask[static_cast<std::size_t>(i + 1)] & own) != 0);
            report(std::string("inv/") + c.name + " mask_parent_subset",
                   (parent & ~mask[static_cast<std::size_t>(i + 1)]) == 0);
            check_eq(std::string("inv/") + c.name + " mask_parent_plus_self",
                     mask[static_cast<std::size_t>(i + 1)], parent | own);
            // The mask never reaches a sibling: exactly one column per depth below.
            for (int d = 1; d < depth[static_cast<std::size_t>(i + 1)]; ++d) {
                int at_depth = 0;
                for (int j = 0; j < tree.node_count(); ++j) {
                    if (depth[static_cast<std::size_t>(j + 1)] != d) { continue; }
                    if ((mask[static_cast<std::size_t>(i + 1)] &
                         (1ULL << static_cast<unsigned>(j + 1))) != 0) {
                        ++at_depth;
                    }
                }
                check_eq(std::string("inv/") + c.name + " mask_one_ancestor_per_depth",
                         at_depth, 1);
            }
        }

        // Determinism: a second build of the same table is identical.
        const Tree again = build_tree(c.scores.data(), c.steps, c.node_budget);
        bool same        = again.nodes.size() == tree.nodes.size();
        for (std::size_t i = 0; same && i < tree.nodes.size(); ++i) {
            same = again.nodes[i].rank == tree.nodes[i].rank &&
                   again.nodes[i].parent == tree.nodes[i].parent &&
                   again.nodes[i].depth == tree.nodes[i].depth;
        }
        report(std::string("inv/") + c.name + " deterministic", same);
    }
}

std::vector<double> random_table(std::mt19937& rng, int steps, int spread) {
    std::vector<double> scores(static_cast<std::size_t>(steps) * kTopK * kTopK);
    std::uniform_int_distribution<int> dist(-spread, spread);
    for (double& value : scores) { value = static_cast<double>(dist(rng)); }
    return scores;
}

// 3) Randomized invariants: the beam can never accept less than the deployed chain
//    (the S19 hardening the reference documents), and accepted_node is always the
//    deepest fully matching node.
void test_randomized() {
    std::mt19937 rng(11);
    for (int trial = 0; trial < 400; ++trial) {
        const int steps  = 1 + trial % 15;
        const int budget = steps + (trial * 7) % (kMaxNodes - steps + 1);
        const std::vector<double> scores = random_table(rng, steps, 1 + trial % 9);
        const Tree tree                  = build_tree(scores.data(), steps, budget);
        const std::string tag             = "rand/" + std::to_string(trial);
        check_invariants(tag, tree);

        const Tree chain_only                 = build_tree(scores.data(), steps, steps);
        const std::vector<std::int32_t> chain = single_chain(scores.data(), steps);
        check_eq(tag + " chain_only_nodes", chain_only.node_count(), steps);

        for (int t = 0; t < 8; ++t) {
            std::vector<std::int32_t> truth(static_cast<std::size_t>(steps));
            std::uniform_int_distribution<int> pick(-1, kTopK - 1);
            for (std::int32_t& v : truth) { v = pick(rng); }
            const std::int32_t len = static_cast<std::int32_t>(truth.size());
            const std::int32_t beam_acc  = accepted_length(tree, truth.data(), len);
            const std::int32_t chain_acc = accepted_length(chain_only, truth.data(), len);
            report(tag + " beam_ge_chain", beam_acc >= chain_acc,
                   std::to_string(beam_acc) + " < " + std::to_string(chain_acc));

            const std::int32_t node = accepted_node(tree, truth.data(), len);
            if (node == kRoot) {
                check_eq(tag + " no_node_means_zero", beam_acc, static_cast<std::int32_t>(0));
                continue;
            }
            const std::vector<std::int32_t> prefix = path_of(tree, node);
            check_eq(tag + " node_depth_matches_length",
                     static_cast<std::int32_t>(prefix.size()), beam_acc);
            check_eq(tag + " node_prefix_fully_accepted",
                     accepted_length(tree, prefix.data(),
                                     static_cast<std::int32_t>(prefix.size())),
                     static_cast<std::int32_t>(prefix.size()));
        }
        // The chain is always a kept path, so replaying it is fully accepted.
        check_eq(tag + " chain_replay",
                 accepted_length(tree, chain.data(), static_cast<std::int32_t>(steps)),
                 static_cast<std::int32_t>(steps));
    }
}

// 4) Engine-layout conformance: the offset formula is the engine's, the builder
//    reads rows through it, and geometry violations throw instead of silently
//    returning a short chain.
void test_layout() {
    const int steps = 7;
    for (int s = 0; s < steps; ++s) {
        for (int p = 0; p < kTopK; ++p) {
            for (int c = 0; c < kTopK; ++c) {
                check_eq(std::string("layout/offset"),
                         score_offset(s, steps, p, c),
                         static_cast<std::int64_t>(s) + steps * (p + kTopK * c));
            }
        }
    }

    // A table whose step-0 rows p > 0 hold a spike must ignore them (the contract
    // reserves step 0 for the anchor row).
    std::vector<double> scores(static_cast<std::size_t>(steps) * kTopK * kTopK, 0.0);
    for (int p = 1; p < kTopK; ++p) {
        scores[static_cast<std::size_t>(score_offset(0, steps, p, 5))] = 1.0e6;
    }
    scores[static_cast<std::size_t>(score_offset(0, steps, 0, 3))] = 1.0;
    check_eq(std::string("layout/step0_ignores_p_gt_0"),
             single_chain(scores.data(), steps)[0], 3);

    // Row following: a spike at (s=1, p=5, c=9) must be taken only if step 0 chose
    // rank 5; with step 0 at rank 3 the walk takes the row-3 argmax instead.
    scores[static_cast<std::size_t>(score_offset(1, steps, 5, 9))] = 1.0e6;
    scores[static_cast<std::size_t>(score_offset(1, steps, 3, 2))] = 1.0;
    check_eq(std::string("layout/row_follows_predecessor"),
             single_chain(scores.data(), steps)[1], 2);

    const auto throws = [&](int s, int budget, const double* table) {
        try {
            build_tree(table, s, budget);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    report("layout/budget_below_steps_throws", throws(steps, steps - 1, scores.data()));
    report("layout/budget_above_mask_throws", throws(steps, kMaxNodes + 1, scores.data()));
    report("layout/zero_steps_throws", throws(0, 1, scores.data()));
    report("layout/too_many_steps_throws", throws(16, 16, scores.data()));
    report("layout/null_scores_throws", throws(steps, steps, nullptr));
}

// 5) The acceptance accounting this header defines is what a tree verify needs: a
//    wrong token truncates everything behind it on that path, and the accepted
//    node's token run is exactly the run the next round's anchors become.
void test_accept_semantics() {
    const int steps = 3;
    std::vector<double> scores(static_cast<std::size_t>(steps) * kTopK * kTopK, 0.0);
    const auto set = [&](int s, int p, int c, double v) {
        scores[static_cast<std::size_t>(score_offset(s, steps, p, c))] = v;
    };
    set(0, 0, 0, 10.0);
    set(0, 0, 1, 9.0);
    set(1, 0, 4, 10.0);
    set(1, 1, 5, 10.0);
    set(2, 4, 6, 10.0);
    set(2, 5, 7, 10.0);
    const Tree tree = build_tree(scores.data(), steps, 2 * steps);
    check_eq(std::string("accept/two_paths"), static_cast<int>(tree.path_ends.size()), 2);

    const std::vector<std::int32_t> truth     = {0, 4, 6};
    const std::vector<std::int32_t> broken    = {0, 4, 7};
    const std::vector<std::int32_t> second    = {1, 5, 7};
    const std::vector<std::int32_t> absent    = {0, -1, 6};
    const std::vector<std::int32_t> nothing   = {-1, -1, -1};
    check_eq(std::string("accept/full_path"), accepted_length(tree, truth.data(), steps),
             static_cast<std::int32_t>(3));
    check_eq(std::string("accept/wrong_deep_token_truncates"),
             accepted_length(tree, broken.data(), steps), static_cast<std::int32_t>(2));
    check_eq(std::string("accept/second_path"), accepted_length(tree, second.data(), steps),
             static_cast<std::int32_t>(3));
    check_eq(std::string("accept/absent_candidate_stops_path"),
             accepted_length(tree, absent.data(), steps), static_cast<std::int32_t>(1));
    check_eq(std::string("accept/nothing"), accepted_length(tree, nothing.data(), steps),
             static_cast<std::int32_t>(0));
    check_eq(std::string("accept/no_node"), accepted_node(tree, nothing.data(), steps), kRoot);

    const std::int32_t node = accepted_node(tree, truth.data(), steps);
    report("accept/node_is_deepest", node != kRoot && tree.nodes[node].depth == 2);
    report("accept/run_equals_truth", node != kRoot && path_of(tree, node) == truth);
}

} // namespace

int main() {
    test_golden();
    test_invariants_on_golden();
    test_randomized();
    test_layout();
    test_accept_semantics();

    std::cout << "dflash2_ddtree_test: " << (checks - failures) << "/" << checks << " checks"
              << (failures == 0 ? " passed OK" : " FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
