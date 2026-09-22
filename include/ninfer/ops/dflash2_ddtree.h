#pragma once

// Host-side DDTree layout builder for the DFlash2 selector lattice.
//
// WHY THIS EXISTS
//   The DFlash2 candidate selector already publishes a full top_k x top_k
//   predecessor x successor score table per (batch element, draft step)
//   (src/ops/kernel/dflash2_selector.cuh:142-189), but the deployed walk
//   (src/ops/kernel/dflash2_selector.cuh:191-319) reads exactly ONE row of that
//   table per step and returns a single chain of `steps` tokens; the runtime side
//   says the same in src/targets/qwen3_6/impl/runtime/dflash2_impl.h:418-420
//   ("selector 已把每步 16x16 前驱-后继分数矩阵算满，而 walk 每个 step 只消费其中 1 行").
//   This header turns ONE batch element's table into a small candidate TREE and
//   emits the layout a tree verifier needs: node order, parent index, per-column
//   depth and the ancestor mask.
//
// SCOPE / NON-GOALS
//   * Host only, header only, standard library only: no engine Tensor, no CUDA,
//     no workspace, no persistent state. Nothing in the decode paths consumes
//     this yet; it is the layout layer a future `--spec dflash2` tree verify
//     needs, landed first so that node ordering and the mask have one testable
//     source of truth (tests/ops/test_dflash2_ddtree.cpp).
//   * It does NOT decide whether a tree is worth its nodes. That is the
//     pre-registered gate in research/scripts/_df2_accept_eval.sh:49-50
//     ("teacher hit@1 >= 0.35 AND (hit@4 - hit@1) >= 0.12 => build the beam-L
//     DDTree; otherwise the tree is not worth its nodes"). The 2026-09-10 run on
//     data/dflash2_ckpts/step_001200.pt judged NO-TREE (teacher path-wise
//     hit@1 = 0.2719, _collab/M_accept_eval.md:29). Do not wire a tree into the
//     decode path unless that gate flips on a newer checkpoint.
//
// SCORE LAYOUT (cited; do not reinvent it)
//   `scores` is ONE batch element's per-block stack in the engine selector
//   layout: scores[s][p][c] lives at flat offset
//       s + steps * (p + top_k * c)
//   which is dflash2_selector_score_offset(b = 0, batch = 1, s, steps, p, c,
//   top_k) from src/ops/kernel/dflash2_selector.cuh:24-27.
//     * row index p = candidate RANK at step s-1 (the walk's `previous`),
//     * column c    = candidate RANK at step s,
//     * larger is better; the column's unary score is already included,
//     * step 0 is the anchor row: only p == 0 is meaningful and the engine writes
//       -inf into p > 0 (src/ops/kernel/dflash2_selector.cuh:168-171), so this
//       builder FORCES row 0 at step 0 and never reads the caller's rows p > 0.
//
// WALK SEMANTICS (matched exactly, and tested against the reference)
//   The single chain is the deployed greedy walk: at each step take
//   argmax_c scores[s][previous][c] with ties broken by the LOWEST candidate
//   rank (src/ops/kernel/dflash2_selector.cuh:242-252). The beam builder keeps
//   the L best cumulative-score prefixes with the same tie-break order
//   (score desc, then candidate rank, then beam index) and always seeds the
//   greedy chain into its result, so a beam tree can never accept less than the
//   deployed chain. This mirrors the S17/S19 reference builder
//   `tools/archkit/dflash2_tree.py::build_tree` of the analysis repo (that file
//   is NOT in this tree; tests/gen_dflash2_ddtree_golden.py records its sha256
//   and regenerates the golden vectors from it).
//
// NODE ORDER (the contract this header defines)
//   A node is a non-empty prefix of candidate ranks = one drafted token at one
//   depth. Nodes are indexed in verify-column order:
//       column 0    -> the anchor / bonus column (the last confirmed token)
//       column 1..N -> nodes sorted by (depth, lexicographic rank prefix)
//   so a parent index is always smaller than its children's, every column's
//   ancestor set is a prefix of the same array, and column_ancestors() returns
//   per column the bit set of columns it attends to: column 0 always, then its
//   own ancestor chain ending at itself, never a sibling.
//   The budget is capped at kMaxNodes because the mask is a 64-bit word. The
//   DEPLOYABLE budget is far lower: the verify width is what limits it. Today the
//   dflash2 verify runs width = k + 1 columns
//   (src/targets/qwen3_6/impl/runtime/dflash2_impl.h:468), and the MTP-side cost
//   evidence is that widening 8 -> 16 columns costs 4.2% decode throughput
//   (research/scripts/_mtp_tree.py: "列数 8 -> 16：tok/s 44.39 -> 42.59（-4.2%）"),
//   i.e. verify slots are close to free but not unlimited.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::ops::ddtree {

// = ops::detail::kDflash2SelectorTopK (src/ops/launcher/dflash2_selector.h:13).
inline constexpr int kTopK = 16;
// Upper bound of the 64-bit ancestor mask; see NODE ORDER for the real cap.
inline constexpr int kMaxNodes = 63;
// Sentinel parent of a depth-0 node, and "no column" for accepted_node().
inline constexpr std::int32_t kRoot = -1;

struct Node {
    std::int32_t rank   = 0;     // candidate rank 0..kTopK-1 at this node's depth
    std::int32_t parent = kRoot; // node index of the parent; kRoot at depth 0
    std::int32_t depth  = 0;     // 0-based draft step
};

struct Tree {
    // Verify-column order: nodes[i] is column i + 1 (see NODE ORDER).
    std::vector<Node> nodes;
    // One entry per kept path: the node index of that path's last node.
    std::vector<std::int32_t> path_ends;
    std::int32_t steps       = 0;
    std::int32_t node_budget = 0;

    int node_count() const { return static_cast<int>(nodes.size()); }
    // Verify columns = the anchor column plus one column per node.
    int columns() const { return node_count() + 1; }
};

// Flat offset of scores[s][p][c] for one batch element.
inline std::int64_t score_offset(std::int32_t s, std::int32_t steps, std::int32_t p,
                                 std::int32_t c) {
    return static_cast<std::int64_t>(s) + static_cast<std::int64_t>(steps) *
                                              (static_cast<std::int64_t>(p) +
                                               static_cast<std::int64_t>(kTopK) * c);
}

inline double score_at(const double* scores, std::int32_t steps, std::int32_t s, std::int32_t p,
                       std::int32_t c) {
    return scores[score_offset(s, steps, p, c)];
}

// Predicate used to sort rank prefixes by (depth, lexicographic ranks), which is
// the node order this header defines.
inline bool prefix_less(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b) {
    if (a.size() != b.size()) { return a.size() < b.size(); }
    return a < b;
}

// Distinct non-empty rank prefixes of `paths` = the tree's node count.
inline int count_nodes(const std::vector<std::vector<std::int32_t>>& paths) {
    std::vector<std::vector<std::int32_t>> prefixes;
    for (const std::vector<std::int32_t>& path : paths) {
        for (std::size_t d = 1; d <= path.size(); ++d) {
            prefixes.emplace_back(path.begin(), path.begin() + static_cast<std::ptrdiff_t>(d));
        }
    }
    std::sort(prefixes.begin(), prefixes.end(), prefix_less);
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());
    return static_cast<int>(prefixes.size());
}

// Cumulative score of one path under the engine row semantics.
inline double path_score(const double* scores, std::int32_t steps,
                         const std::vector<std::int32_t>& path) {
    double total          = 0.0;
    std::int32_t previous = 0;
    for (std::int32_t s = 0; s < steps; ++s) {
        const std::int32_t pred = s == 0 ? 0 : previous;
        total += score_at(scores, steps, s, pred, path[static_cast<std::size_t>(s)]);
        previous = path[static_cast<std::size_t>(s)];
    }
    return total;
}

// The deployed greedy chain: candidate rank per step, ties -> lowest rank.
inline std::vector<std::int32_t> single_chain(const double* scores, std::int32_t steps) {
    std::vector<std::int32_t> chain(static_cast<std::size_t>(steps));
    std::int32_t previous = 0;
    for (std::int32_t s = 0; s < steps; ++s) {
        const std::int32_t pred = s == 0 ? 0 : previous;
        std::int32_t best       = 0;
        for (std::int32_t c = 1; c < kTopK; ++c) {
            // Strict >: ties keep the lowest candidate rank.
            if (score_at(scores, steps, s, pred, c) > score_at(scores, steps, s, pred, best)) {
                best = c;
            }
        }
        chain[static_cast<std::size_t>(s)] = best;
        previous                           = best;
    }
    return chain;
}

// Beam-L candidate tree over the 16x16 next-pair lattice, in the reference
// builder's semantics: L = clamp(node_budget / steps, 1, kTopK), the greedy chain
// is always seeded, and node_count() <= node_budget holds on return.
inline Tree build_tree(const double* scores, std::int32_t steps, std::int32_t node_budget) {
    if (scores == nullptr) { throw std::invalid_argument("ddtree::build_tree: null scores"); }
    if (steps < 1 || steps > 15) {
        throw std::invalid_argument("ddtree::build_tree: steps must be 1..15 (registered dflash2 "
                                    "selector geometry S=1..15)");
    }
    if (node_budget < steps) {
        throw std::invalid_argument("ddtree::build_tree: node_budget < steps cannot hold even one "
                                    "chain");
    }
    if (node_budget > kMaxNodes) {
        throw std::invalid_argument("ddtree::build_tree: node_budget exceeds kMaxNodes (the "
                                    "64-bit ancestor mask limit)");
    }

    const std::int32_t l =
        std::max<std::int32_t>(1, std::min<std::int32_t>(kTopK, node_budget / steps));

    std::vector<std::vector<std::int32_t>> paths;
    if (l == 1) {
        paths.push_back(single_chain(scores, steps));
    } else {
        struct Extension {
            double cum                = 0.0;
            std::int32_t rank         = 0;
            std::int32_t beam         = 0;
            std::vector<std::int32_t> path;
        };
        std::vector<Extension> beams(1);
        for (std::int32_t s = 0; s < steps; ++s) {
            std::vector<Extension> extensions;
            extensions.reserve(beams.size() * static_cast<std::size_t>(kTopK));
            for (std::size_t bi = 0; bi < beams.size(); ++bi) {
                const Extension& beam   = beams[bi];
                const std::int32_t pred = s == 0 ? 0 : beam.rank;
                for (std::int32_t c = 0; c < kTopK; ++c) {
                    Extension extension;
                    extension.cum  = beam.cum + score_at(scores, steps, s, pred, c);
                    extension.rank = c;
                    extension.beam = static_cast<std::int32_t>(bi);
                    extension.path = beam.path;
                    extension.path.push_back(c);
                    extensions.push_back(std::move(extension));
                }
            }
            std::sort(extensions.begin(), extensions.end(),
                      [](const Extension& a, const Extension& b) {
                          if (a.cum != b.cum) { return a.cum > b.cum; }
                          if (a.rank != b.rank) { return a.rank < b.rank; }
                          return a.beam < b.beam;
                      });
            const std::size_t keep = std::min<std::size_t>(static_cast<std::size_t>(l),
                                                          extensions.size());
            beams.assign(extensions.begin(),
                         extensions.begin() + static_cast<std::ptrdiff_t>(keep));
        }
        for (const Extension& beam : beams) {
            if (std::find(paths.begin(), paths.end(), beam.path) == paths.end()) {
                paths.push_back(beam.path); // merged/duplicated beams collapse
            }
        }
        // Seed the greedy chain so beam mode can never lose to the plain chain.
        const std::vector<std::int32_t> chain = single_chain(scores, steps);
        if (std::find(paths.begin(), paths.end(), chain) == paths.end()) {
            paths.push_back(chain);
            std::sort(paths.begin(), paths.end(),
                      [&](const std::vector<std::int32_t>& a,
                          const std::vector<std::int32_t>& b) {
                          const double sa = path_score(scores, steps, a);
                          const double sb = path_score(scores, steps, b);
                          if (sa != sb) { return sa > sb; }
                          return a < b;
                      });
            while (count_nodes(paths) > node_budget && paths.size() > 1) {
                std::size_t drop = paths.size();
                for (std::size_t i = paths.size(); i-- > 0;) {
                    if (paths[i] != chain) {
                        drop = i;
                        break;
                    }
                }
                if (drop == paths.size()) { break; } // only the chain is left
                paths.erase(paths.begin() + static_cast<std::ptrdiff_t>(drop));
            }
        }
    }

    // Materialize the tree: the nodes are the distinct non-empty prefixes, sorted
    // by (depth, lexicographic rank prefix) so parent indices come first.
    std::vector<std::vector<std::int32_t>> prefixes;
    for (const std::vector<std::int32_t>& path : paths) {
        for (std::size_t d = 1; d <= path.size(); ++d) {
            prefixes.emplace_back(path.begin(), path.begin() + static_cast<std::ptrdiff_t>(d));
        }
    }
    std::sort(prefixes.begin(), prefixes.end(), prefix_less);
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());

    Tree tree;
    tree.steps       = steps;
    tree.node_budget = node_budget;
    tree.nodes.reserve(prefixes.size());
    for (const std::vector<std::int32_t>& prefix : prefixes) {
        Node node;
        node.depth = static_cast<std::int32_t>(prefix.size()) - 1;
        node.rank  = prefix.back();
        if (prefix.size() == 1) {
            node.parent = kRoot;
        } else {
            const std::vector<std::int32_t> parent_prefix(
                prefix.begin(), prefix.end() - static_cast<std::ptrdiff_t>(1));
            const auto found = std::lower_bound(prefixes.begin(), prefixes.end(), parent_prefix,
                                                prefix_less);
            node.parent      = static_cast<std::int32_t>(found - prefixes.begin());
        }
        tree.nodes.push_back(node);
    }
    for (const std::vector<std::int32_t>& path : paths) {
        const auto found = std::lower_bound(prefixes.begin(), prefixes.end(), path, prefix_less);
        tree.path_ends.push_back(static_cast<std::int32_t>(found - prefixes.begin()));
    }
    return tree;
}

// The rank path of every kept path, in the builder's order.
inline std::vector<std::vector<std::int32_t>> tree_paths(const Tree& tree) {
    std::vector<std::vector<std::int32_t>> paths;
    paths.reserve(tree.path_ends.size());
    for (std::int32_t end : tree.path_ends) {
        std::vector<std::int32_t> path;
        for (std::int32_t node = end; node != kRoot; node = tree.nodes[node].parent) {
            path.push_back(tree.nodes[node].rank);
        }
        std::reverse(path.begin(), path.end());
        paths.push_back(std::move(path));
    }
    return paths;
}

// Verify-column depth: index 0 is the anchor column (depth 0), column i + 1 is
// nodes[i] one position past the anchor it descends from.
inline std::vector<std::int32_t> column_depth(const Tree& tree) {
    std::vector<std::int32_t> depth(static_cast<std::size_t>(tree.columns()), 0);
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
        depth[i + 1] = tree.nodes[i].depth + 1;
    }
    return depth;
}

// Position of each verify column relative to the anchor, for the verify pass's
// rope: column 0 sits at the anchor, column i + 1 sits depth + 1 positions later.
inline std::vector<std::int32_t> column_rope_offsets(const Tree& tree) {
    return column_depth(tree);
}

// Per-column attention mask: bit j is set iff column j may be attended to. Column
// 0 sees itself; node column i + 1 sees column 0, its own ancestor chain and
// itself, never a sibling.
inline std::vector<std::uint64_t> column_ancestors(const Tree& tree) {
    std::vector<std::uint64_t> mask(static_cast<std::size_t>(tree.columns()), 0);
    mask[0] = 1ULL;
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
        std::uint64_t bits = 1ULL;
        for (std::int32_t node = static_cast<std::int32_t>(i); node != kRoot;
             node             = tree.nodes[node].parent) {
            bits |= 1ULL << static_cast<unsigned>(node + 1);
        }
        mask[i + 1] = bits;
    }
    return mask;
}

// Target accept rule: a node is accepted iff its whole rank prefix equals the
// truth's prefix. `truth_ranks[d]` is the candidate rank of the target's token at
// depth d, or kRoot when that token is not among the step's candidates (the path
// stops there). Returns the accepted length in tokens = accepted depth + 1.
inline std::int32_t accepted_length(const Tree& tree, const std::int32_t* truth_ranks,
                                    std::int32_t truth_len) {
    std::int32_t best = 0;
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
        const std::int32_t depth = tree.nodes[i].depth;
        if (depth + 1 <= best) { continue; }
        std::int32_t node = static_cast<std::int32_t>(i);
        bool matches      = true;
        for (std::int32_t d = depth; d >= 0; --d) {
            if (d >= truth_len || truth_ranks[d] != tree.nodes[node].rank) {
                matches = false;
                break;
            }
            node = tree.nodes[node].parent;
        }
        if (matches) { best = depth + 1; }
    }
    return best;
}

// The column the target accepted at depth accepted_length() - 1, i.e. the last
// token of the verified run; kRoot when nothing was accepted. Walk `parent` back
// to the root to recover the accepted token run, which is what the next round's
// anchors become.
inline std::int32_t accepted_node(const Tree& tree, const std::int32_t* truth_ranks,
                                  std::int32_t truth_len) {
    const std::int32_t length = accepted_length(tree, truth_ranks, truth_len);
    if (length == 0) { return kRoot; }
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
        if (tree.nodes[i].depth != length - 1) { continue; }
        std::int32_t node = static_cast<std::int32_t>(i);
        bool matches      = true;
        for (std::int32_t d = tree.nodes[i].depth; d >= 0; --d) {
            if (d >= truth_len || truth_ranks[d] != tree.nodes[node].rank) {
                matches = false;
                break;
            }
            node = tree.nodes[node].parent;
        }
        if (matches) { return static_cast<std::int32_t>(i); }
    }
    return kRoot;
}

} // namespace ninfer::ops::ddtree
