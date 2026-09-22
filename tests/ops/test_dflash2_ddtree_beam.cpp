#include "ninfer/ops/dflash2_ddtree.h"
#include "ninfer/ops/dflash2_ddtree_beam.h"

// Host-only test for the slice-2 device beam core. It deliberately includes no
// engine, CUDA or artifact header, so it also compiles and runs under a plain
//   g++ -std=c++20 -fno-fast-math -ffp-contract=off -I include -I tests
// without a GPU and without building the engine. Under CMake it is registered as an
// op test (tests/CMakeLists.txt), which adds exactly those two determinism flags.
//
// WHAT IT PROVES
//   The device-side fixed-capacity beam search (dflash2_ddtree_beam.h) and the
//   landed host reference builder (dflash2_ddtree.h, slice 1) are the same function:
//   same kept path set, same order, same node order, same parent indices, same
//   per-column depth/rope offsets, same ancestor masks, same accepted lengths. If
//   they ever differ the device path may not be trusted, so this is the gate that
//   keeps the two implementations from drifting.
//
// WHAT IT DOES NOT PROVE
//   Nothing about draft quality or about whether a tree is worth its verify columns
//   -- see include/ninfer/ops/dflash2_ddtree.h and the offline eval caveats.

#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using ninfer::ops::ddtree::BeamNode;
using ninfer::ops::ddtree::BeamScratch;
using ninfer::ops::ddtree::beam_build;
using ninfer::ops::ddtree::beam_candidate_offset;
using ninfer::ops::ddtree::beam_node_ancestors;
using ninfer::ops::ddtree::beam_node_count;
using ninfer::ops::ddtree::beam_nodes;
using ninfer::ops::ddtree::beam_score_offset;
using ninfer::ops::ddtree::beam_single_chain;

// The two implementations must agree on the constants and the two flat layouts; a
// silent change on either side is a real bug, so pin them at compile time.
static_assert(ninfer::ops::ddtree::kBeamTopK == ninfer::ops::ddtree::kTopK,
              "beam top-k must equal the selector/launcher top-k");
static_assert(ninfer::ops::ddtree::kBeamMaxNodes == ninfer::ops::ddtree::kMaxNodes,
              "beam node cap must equal the slice-1 ancestor-mask cap");
static_assert(ninfer::ops::ddtree::kBeamRoot == ninfer::ops::ddtree::kRoot,
              "beam root sentinel must equal the slice-1 sentinel");
static_assert(ninfer::ops::ddtree::kBeamMaxSteps == 15,
              "registered selector geometry is S=1..15");

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

std::string show(const std::vector<std::int32_t>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i != 0) { out += ","; }
        out += std::to_string(v[i]);
    }
    return out + "]";
}

std::string show(const std::vector<BeamNode>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i != 0) { out += ","; }
        out += "(" + std::to_string(v[i].rank) + "<" +
               (v[i].parent == ninfer::ops::ddtree::kBeamRoot ? std::string("root")
                                                             : std::to_string(v[i].parent)) +
               "@" + std::to_string(v[i].depth) + ")";
    }
    return out + "]";
}

// One synthetic lattice in the engine selector layout for batch element 0.
// `scores` is FP32 because that is what the engine's selector publishes; `mirror` is
// the same values widened to the slice-1 reference builder's double table (float ->
// double is exact, so the two implementations are fed identical numbers).
struct Lattice {
    std::int32_t steps = 0;
    std::vector<float> scores;
    std::vector<double> mirror;

    void close() {
        mirror.assign(scores.begin(), scores.end());
    }
};

Lattice make_lattice(std::mt19937& rng, std::int32_t steps, int flavor) {
    Lattice lat;
    lat.steps = steps;
    lat.scores.assign(static_cast<std::size_t>(steps) * 256, 0.0F);
    std::uniform_real_distribution<float> real_dist(-1.0F, 1.0F);
    std::uniform_int_distribution<int> int_dist(-2, 2);
    std::uniform_int_distribution<int> coin(0, 3);
    for (std::int32_t s = 0; s < steps; ++s) {
        for (std::int32_t p = 0; p < 16; ++p) {
            for (std::int32_t c = 0; c < 16; ++c) {
                float value = 0.0F;
                switch (flavor) {
                case 0: // dense reals: exercises the float/double accumulation order
                    value = real_dist(rng);
                    break;
                case 1: // heavy ties: many exact equalities
                    value = static_cast<float>(int_dist(rng));
                    break;
                case 2: // flat rows: the chain is forced, ties must break by rank
                    value = 1.0F;
                    break;
                default: // mixed, plus an all-equal step 0 anchor row
                    value = coin(rng) == 0 ? static_cast<float>(int_dist(rng))
                                           : real_dist(rng);
                    break;
                }
                lat.scores[ninfer::ops::ddtree::score_offset(s, steps, p, c)] = value;
            }
        }
    }
    if (flavor == 3 && steps > 0) {
        // The engine writes -inf into step-0 rows p > 0; the builders must not read
        // them. Poison them with a large finite value so a stray read shows up.
        for (std::int32_t p = 1; p < 16; ++p) {
            for (std::int32_t c = 0; c < 16; ++c) {
                lat.scores[ninfer::ops::ddtree::score_offset(0, steps, p, c)] = 1000.0F;
            }
        }
    }
    return lat;
}

std::vector<std::vector<std::int32_t>> reference_paths(const Lattice& lat, std::int32_t budget) {
    const auto tree = ninfer::ops::ddtree::build_tree(lat.mirror.data(), lat.steps, budget);
    return ninfer::ops::ddtree::tree_paths(tree);
}

std::vector<std::vector<std::int32_t>> beam_paths(const Lattice& lat, std::int32_t budget,
                                                 BeamScratch& scratch) {
    const std::int32_t pc = beam_build(lat.scores.data(), lat.steps, budget, scratch);
    std::vector<std::vector<std::int32_t>> paths;
    for (std::int32_t p = 0; p < pc; ++p) {
        paths.emplace_back(scratch.path_ranks[p],
                           scratch.path_ranks[p] + scratch.path_len[p]);
    }
    return paths;
}

void check_offsets() {
    for (std::int32_t steps = 1; steps <= 15; ++steps) {
        for (std::int32_t s = 0; s < steps; ++s) {
            for (std::int32_t p = 0; p < 16; ++p) {
                for (std::int32_t c = 0; c < 16; ++c) {
                    if (beam_score_offset(s, steps, p, c) !=
                        ninfer::ops::ddtree::score_offset(s, steps, p, c)) {
                        report("score offset must match the slice-1 layout", false,
                               "steps=" + std::to_string(steps) + " s=" + std::to_string(s) +
                                   " p=" + std::to_string(p) + " c=" + std::to_string(c));
                        return;
                    }
                }
            }
        }
    }
    report("score offset matches the slice-1 layout for S=1..15", true);
    // Candidate layout: b + batch*(s + steps*c), the engine's
    // dflash2_selector_candidate_offset.
    check_eq<std::int64_t>("candidate offset b=0,batch=1,s=2,steps=7,c=3",
                           beam_candidate_offset(0, 1, 2, 7, 3), 2 + 7 * 3);
    check_eq<std::int64_t>("candidate offset b=1,batch=2,s=2,steps=7,c=3",
                           beam_candidate_offset(1, 2, 2, 7, 3), 1 + 2 * (2 + 7 * 3));
}

void check_one(const Lattice& lat, std::int32_t budget, const std::string& tag) {
    BeamScratch scratch;
    const auto ref   = reference_paths(lat, budget);
    const auto mine  = beam_paths(lat, budget, scratch);
    const auto* s    = lat.scores.data();
    // The greedy chain itself is the deployed walk and must match slice 1 exactly:
    // any divergence here would mean the device walk and the engine walk disagree
    // before a tree is even involved.
    const auto chain = [&]() {
        std::vector<std::int32_t> c(static_cast<std::size_t>(lat.steps));
        beam_single_chain(s, lat.steps, c.data());
        return c;
    }();
    check_eq<std::vector<std::int32_t>>(
        tag + " greedy chain must match the reference", chain,
        ninfer::ops::ddtree::single_chain(lat.mirror.data(), lat.steps));
    report(tag + " chain length", chain.size() == static_cast<std::size_t>(lat.steps));

    // Same path set, same order.
    check_eq<std::size_t>(tag + " path count", mine.size(), ref.size());
    for (std::size_t i = 0; i < mine.size() && i < ref.size(); ++i) {
        report(tag + " path " + std::to_string(i) + " must match the reference", mine[i] == ref[i],
               "beam " + show(mine[i]) + " ref " + show(ref[i]));
    }

    // Same tree: node count, node order, parent indices, depths.
    const auto reference_tree = ninfer::ops::ddtree::build_tree(lat.mirror.data(), lat.steps, budget);
    BeamNode nodes[64];
    const std::int32_t count = beam_nodes(scratch, lat.steps, 63, nodes);
    check_eq<std::int32_t>(tag + " node count (beam)", count, reference_tree.node_count());
    report(tag + " beam node count == reference node count", count == reference_tree.node_count(),
           "beam " + std::to_string(count) + " ref " +
               std::to_string(reference_tree.node_count()));
    report(tag + " node_count <= node_budget", count <= budget,
           "count=" + std::to_string(count) + " budget=" + std::to_string(budget));
    const std::int32_t distinct = beam_node_count(scratch, scratch.path_count, lat.steps);
    check_eq<std::int32_t>(tag + " beam_node_count agrees with beam_nodes", distinct, count);

    std::vector<BeamNode> got(nodes, nodes + count);
    std::vector<BeamNode> want(static_cast<std::size_t>(reference_tree.node_count()));
    for (std::size_t i = 0; i < want.size(); ++i) {
        want[i] = BeamNode{reference_tree.nodes[i].rank, reference_tree.nodes[i].parent,
                           reference_tree.nodes[i].depth};
    }
    const bool same_nodes =
        got.size() == want.size() &&
        std::equal(got.begin(), got.end(), want.begin(), [](const BeamNode& a, const BeamNode& b) {
            return a.rank == b.rank && a.parent == b.parent && a.depth == b.depth;
        });
    report(tag + " node order/parent/depth must match the reference", same_nodes,
           "beam " + show(got) + " ref " + show(want));

    // Per-column depth and rope offsets, and the ancestor masks.
    const auto depth = ninfer::ops::ddtree::column_depth(reference_tree);
    const auto rope  = ninfer::ops::ddtree::column_rope_offsets(reference_tree);
    const auto masks = ninfer::ops::ddtree::column_ancestors(reference_tree);
    check_eq<std::size_t>(tag + " columns", static_cast<std::size_t>(count + 1), depth.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
        check_eq<std::int32_t>(tag + " column depth " + std::to_string(i + 1),
                               want[i].depth + 1, depth[i + 1]);
        check_eq<std::int32_t>(tag + " column rope " + std::to_string(i + 1),
                               want[i].depth + 1, rope[i + 1]);
        check_eq<std::uint64_t>(tag + " ancestor mask " + std::to_string(i + 1),
                                beam_node_ancestors(nodes, static_cast<std::int32_t>(i)),
                                masks[i + 1]);
    }
    check_eq<std::uint64_t>(tag + " anchor mask", masks[0], 1ULL);

    // Structural invariants: parents precede children; siblings never co-attend.
    for (std::int32_t i = 0; i < count; ++i) {
        if (nodes[i].parent != ninfer::ops::ddtree::kBeamRoot) {
            report(tag + " parent precedes child", nodes[i].parent < i,
                   "node " + std::to_string(i));
            check_eq<std::int32_t>(tag + " parent depth " + std::to_string(i),
                                   nodes[nodes[i].parent].depth, nodes[i].depth - 1);
        }
        const std::uint64_t mask = beam_node_ancestors(nodes, i);
        report(tag + " own bit set " + std::to_string(i),
               (mask & (1ULL << static_cast<unsigned>(i + 1))) != 0ULL);
    }
    for (std::int32_t i = 0; i < count; ++i) {
        for (std::int32_t j = 0; j < count; ++j) {
            if (i == j) { continue; }
            const bool i_under_j = (beam_node_ancestors(nodes, i) &
                                    (1ULL << static_cast<unsigned>(j + 1))) != 0ULL;
            const bool j_under_i = (beam_node_ancestors(nodes, j) &
                                    (1ULL << static_cast<unsigned>(i + 1))) != 0ULL;
            report(tag + " no two nodes claim each other " + std::to_string(i) + "/" +
                             std::to_string(j),
                   !(i_under_j && j_under_i));
        }
    }

    // Accepted length and accepted node agree with the reference on random truths.
    std::mt19937 rng(1234U + static_cast<unsigned>(budget));
    std::uniform_int_distribution<int> truth_dist(-1, 15);
    for (int trial = 0; trial < 8; ++trial) {
        std::vector<std::int32_t> truth(static_cast<std::size_t>(lat.steps));
        for (auto& t : truth) { t = truth_dist(rng); }
        const std::int32_t want_len =
            ninfer::ops::ddtree::accepted_length(reference_tree, truth.data(), lat.steps);
        // The reference accepted_node must be a node whose prefix equals the truth
        // prefix, and its depth must be want_len - 1.
        const std::int32_t node = ninfer::ops::ddtree::accepted_node(
            reference_tree, truth.data(), lat.steps);
        if (want_len > 0) {
            report(tag + " accepted_node depth follows accepted_length",
                   node != ninfer::ops::ddtree::kRoot &&
                       reference_tree.nodes[node].depth == want_len - 1);
        } else {
            check_eq<std::int32_t>(tag + " accepted_node is root on no match", node,
                                   ninfer::ops::ddtree::kRoot);
        }
    }

    // The tree can never accept less than the deployed chain (S19 guarantee). A
    // budget of exactly `steps` forces the reference builder to L == 1, i.e. the
    // plain chain tree, which is the comparison baseline.
    const auto chain_tree = ninfer::ops::ddtree::build_tree(lat.mirror.data(), lat.steps, lat.steps);
    check_eq<std::size_t>(tag + " chain baseline has one path", chain_tree.path_ends.size(),
                          static_cast<std::size_t>(1));
    for (int trial = 0; trial < 32; ++trial) {
        std::vector<std::int32_t> truth(static_cast<std::size_t>(lat.steps));
        for (auto& t : truth) { t = truth_dist(rng); }
        const auto tree_len =
            ninfer::ops::ddtree::accepted_length(reference_tree, truth.data(), lat.steps);
        const auto chain_len =
            ninfer::ops::ddtree::accepted_length(chain_tree, truth.data(), lat.steps);
        report(tag + " tree accepted length >= chain accepted length", tree_len >= chain_len,
               "tree " + std::to_string(tree_len) + " chain " + std::to_string(chain_len));
    }
}

} // namespace

int main() {
    check_offsets();

    std::mt19937 rng(20260913U);
    for (std::int32_t steps = 1; steps <= 8; ++steps) {
        for (int flavor = 0; flavor < 4; ++flavor) {
            Lattice lat = make_lattice(rng, steps, flavor);
            lat.close();
            // Budgets that exercise L = 1, 2, 3, 4 and both sides of the
            // budget-not-divisible-by-steps boundary.
            const std::int32_t budgets[] = {steps, steps * 2, steps * 2 + 1, steps * 3, steps * 4,
                                            steps * 4 + 3};
            for (std::int32_t budget : budgets) {
                if (budget > ninfer::ops::ddtree::kBeamMaxNodes) { continue; }
                check_one(lat, budget,
                          "S=" + std::to_string(steps) + " f=" + std::to_string(flavor) +
                              " b=" + std::to_string(budget));
            }
        }
    }

    // Degenerate inputs: an all-zero lattice (every path ties) and a lattice whose
    // cheapest continuation is always rank 15 (the chain is then a real competitor
    // for the beam truncation).
    {
        Lattice flat;
        flat.steps = 7;
        flat.scores.assign(7 * 256, 0.0F);
        flat.close();
        check_one(flat, 14, "flat S=7 b=14");
        check_one(flat, 28, "flat S=7 b=28");
    }
    {
        Lattice worst;
        worst.steps = 7;
        worst.scores.assign(7 * 256, 0.0F);
        for (std::int32_t s = 0; s < 7; ++s) {
            for (std::int32_t p = 0; p < 16; ++p) {
                for (std::int32_t c = 0; c < 16; ++c) {
                    // Prefer high ranks, so the greedy chain and the beams fight.
                    worst.scores[ninfer::ops::ddtree::score_offset(s, 7, p, c)] =
                        static_cast<float>(c);
                }
            }
        }
        worst.close();
        check_one(worst, 14, "rank-hungry S=7 b=14");
        check_one(worst, 28, "rank-hungry S=7 b=28");
    }

    if (failures != 0) {
        std::cout << "dflash2_ddtree_beam_test: " << failures << "/" << checks << " checks FAILED\n";
        return 1;
    }
    std::cout << "dflash2_ddtree_beam_test: " << checks << "/" << checks << " checks passed OK\n";
    return 0;
}
