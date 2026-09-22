#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_dflash2_ddtree_golden.py -- regenerate tests/ops/dflash2_ddtree_golden.h.

The golden vectors come from the S17/S19 reference builder
`tools/archkit/dflash2_tree.py::build_tree` (analysis repo), which is NOT part of
this tree, so its path is an explicit argument and its sha256 is recorded in the
generated header. Nothing is invented here: the reference defines the beam
semantics, and this script only

  * lays each case's table in the engine selector layout, i.e. the flat offset
    s + steps * (p + top_k * c)
    (dflash2_selector_score_offset, src/ops/kernel/dflash2_selector.cuh:24-27),
  * derives the canonical node list (rank, depth, parent index) that
    include/ninfer/ops/dflash2_ddtree.h defines: nodes = the distinct non-empty
    prefixes of the kept paths, sorted by (depth, lexicographic rank prefix), so
    a parent index is always smaller than its children's,
  * evaluates the per-truth accepted length with the reference
    `accepted_length(paths, truth)` and asserts it equals the per-node prefix rule
    the header implements (the two agree because the path set is prefix-closed),
  * evaluates the per-truth winning column (accepted_node) by locating the node
    whose rank prefix equals truth[:accepted_length].

Usage: python3 tests/gen_dflash2_ddtree_golden.py --reference <dflash2_tree.py>
                                                   [--out tests/ops/dflash2_ddtree_golden.h]
"""
import argparse
import hashlib
import importlib.util
import os
import random
import sys

TOP_K = 16


def load_reference(path):
    spec = importlib.util.spec_from_file_location("dflash2_tree_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flat_offset(s, steps, p, c):
    """Engine selector layout: dflash2_selector_score_offset for one batch element."""
    return s + steps * (p + TOP_K * c)


def table_to_engine_layout(table, steps):
    """table[s][p][c] -> one flat double array in the engine offset order."""
    flat = [0.0] * (steps * TOP_K * TOP_K)
    for s in range(steps):
        for p in range(TOP_K):
            for c in range(TOP_K):
                flat[flat_offset(s, steps, p, c)] = table[s][p][c]
    return flat


def canonical_nodes(paths):
    """(rank, depth, parent index) per the header's NODE ORDER."""
    prefixes, seen = [], {}
    for path in paths:
        for d in range(1, len(path) + 1):
            prefix = tuple(path[:d])
            if prefix not in seen:
                seen[prefix] = None
                prefixes.append(prefix)
    prefixes.sort(key=lambda p: (len(p), p))
    for index, prefix in enumerate(prefixes):
        seen[prefix] = index
    ranks = [prefix[-1] for prefix in prefixes]
    depths = [len(prefix) - 1 for prefix in prefixes]
    parents = [-1 if len(prefix) == 1 else seen[prefix[:-1]] for prefix in prefixes]
    return ranks, depths, parents


def rank_prefix(ranks, parents, node):
    """The full rank prefix of a node, root first."""
    seq = []
    while node != -1:
        seq.append(ranks[node])
        node = parents[node]
    seq.reverse()
    return seq


def accepted_length_from_nodes(paths, truth):
    """Independent re-derivation of the per-node prefix accept rule."""
    ranks, depths, parents = canonical_nodes(paths)
    best = 0
    for node in range(len(ranks)):
        prefix = rank_prefix(ranks, parents, node)
        if list(truth[:len(prefix)]) == prefix and len(prefix) > best:
            best = len(prefix)
    return best


def accepted_node_from_nodes(paths, truth):
    """Winning column in canonical node order, -1 when nothing matched."""
    ranks, depths, parents = canonical_nodes(paths)
    best, best_node = 0, -1
    for node in range(len(ranks)):
        prefix = rank_prefix(ranks, parents, node)
        if list(truth[:len(prefix)]) == prefix and len(prefix) > best:
            best, best_node = len(prefix), node
    return best_node


def make_table(rng, steps, mode):
    table = [[[0.0] * TOP_K for _ in range(TOP_K)] for _ in range(steps)]
    for s in range(steps):
        for p in range(TOP_K):
            for c in range(TOP_K):
                if mode == "ints":
                    table[s][p][c] = float(rng.randint(-8, 8))
                elif mode == "quarters":
                    table[s][p][c] = 0.25 * rng.randint(-16, 16)
                elif mode == "ties":
                    table[s][p][c] = float(rng.randint(0, 2))
                elif mode == "flat":
                    table[s][p][c] = 0.0
                elif mode == "sparse":
                    table[s][p][c] = float(rng.choice([-1000.0, 0.0, 1.0, 2.0]))
                else:
                    raise ValueError(mode)
    return table


def build_case(reference, name, rng, steps, budget, mode, step0_garbage=False):
    table = make_table(rng, steps, mode)
    if step0_garbage:
        for p in range(1, TOP_K):
            for c in range(TOP_K):
                table[0][p][c] = 999.0
    # The reference forces row 0 at step 0, so a golden table may carry garbage there.
    flat = table_to_engine_layout(table, steps)

    chain = list(reference.single_chain_walk(table, steps))
    # The reference works in tuples (tree_node_count hashes the prefixes), so keep
    # tuples for every reference call and only convert at emission time.
    paths = [tuple(p) for p in reference.build_tree(table, steps, budget)]
    node_ranks, node_depths, node_parents = canonical_nodes(paths)
    assert len(node_ranks) == reference.tree_node_count(paths), name
    assert len(node_ranks) <= budget, (name, len(node_ranks), budget)

    truths = [[-1] * steps, [0] * steps, list(chain)]
    if paths:
        truths.append(list(paths[0]))
    if node_depths:
        # one node's own root path, padded with -1 ("not a candidate") behind it
        mid = next((i for i in range(len(node_depths)) if node_depths[i] == min(1, steps - 1)), 0)
        prefix = rank_prefix(node_ranks, node_parents, mid)
        truths.append(prefix + [-1] * (steps - len(prefix)))
    truths.append([1 if s % 2 == 0 else -1 for s in range(steps)])
    truths.append([15 if s % 3 == 0 else -1 for s in range(steps)])
    truths = [t[:steps] for t in truths]

    ref_acc = [reference.accepted_length(paths, tuple(t)) for t in truths]
    ind_acc = [accepted_length_from_nodes(paths, t) for t in truths]
    assert ref_acc == ind_acc, (name, ref_acc, ind_acc, truths)
    anode = [accepted_node_from_nodes(paths, t) for t in truths]

    return {
        "name": name,
        "steps": steps,
        "node_budget": budget,
        "scores": flat,
        "chain": chain,
        "paths": [list(p) for p in sorted(paths)],
        "node_rank": node_ranks,
        "node_depth": node_depths,
        "node_parent": node_parents,
        "truths": truths,
        "accepted_length": ref_acc,
        "accepted_node": anode,
    }


def fmt_doubles(values):
    out = []
    for v in values:
        if v == int(v) and abs(v) < 1e15:
            out.append("%d.0" % int(v))
        else:
            out.append(repr(v))
    return "{" + ", ".join(out) + "}"


def fmt_ints(values):
    return "{" + ", ".join("%d" % v for v in values) + "}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="path to tools/archkit/dflash2_tree.py")
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops",
                                         "dflash2_ddtree_golden.h"))
    args = ap.parse_args()

    reference = load_reference(args.reference)
    with open(args.reference, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()

    rng = random.Random(7)
    specs = [
        ("d7_budget15_ints", 7, 15, "ints", False),
        ("d7_budget28_ints", 7, 28, "ints", False),
        ("d7_budget14_ints", 7, 14, "ints", False),
        ("d7_budget7_chain_only", 7, 7, "ints", False),
        ("d1_budget1", 1, 1, "ints", False),
        ("d1_budget16", 1, 16, "ints", False),
        ("d7_budget63_flat_ties", 7, 63, "flat", False),
        ("d3_budget9_quarters", 3, 9, "quarters", False),
        ("d7_budget15_step0_garbage", 7, 15, "ints", True),
        ("d7_budget15_ties", 7, 15, "ties", False),
        ("d15_budget30_ints", 15, 30, "ints", False),
        ("d2_budget32_sparse", 2, 32, "sparse", False),
    ]
    cases = [build_case(reference, name, rng, steps, budget, mode, garbage)
             for name, steps, budget, mode, garbage in specs]

    lines = []
    lines.append("// GENERATED by tests/gen_dflash2_ddtree_golden.py -- do not edit by hand.")
    lines.append("// reference implementation: %s" % os.path.basename(args.reference))
    lines.append("// reference sha256: %s" % digest)
    lines.append("// Regenerate with:")
    lines.append("//   python3 tests/gen_dflash2_ddtree_golden.py --reference <dir>/dflash2_tree.py")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("namespace ninfer::test::ddtree_golden {")
    lines.append("")
    lines.append("struct Case {")
    lines.append("    const char* name;")
    lines.append("    std::int32_t steps;")
    lines.append("    std::int32_t node_budget;")
    lines.append("    std::vector<double> scores;                   // engine selector layout")
    lines.append("    std::vector<std::int32_t> chain;              // single_chain()")
    lines.append("    std::vector<std::vector<std::int32_t>> paths; // build_tree() path set")
    lines.append("    std::vector<std::int32_t> node_rank;          // canonical node list")
    lines.append("    std::vector<std::int32_t> node_depth;")
    lines.append("    std::vector<std::int32_t> node_parent;")
    lines.append("    std::vector<std::vector<std::int32_t>> truths;")
    lines.append("    std::vector<std::int32_t> accepted_length;")
    lines.append("    std::vector<std::int32_t> accepted_node;")
    lines.append("};")
    lines.append("")
    lines.append("inline std::vector<Case> make_cases() {")
    lines.append("    std::vector<Case> cases;")
    for case in cases:
        lines.append("    {")
        lines.append("        Case c;")
        lines.append('        c.name        = "%s";' % case["name"])
        lines.append("        c.steps       = %d;" % case["steps"])
        lines.append("        c.node_budget = %d;" % case["node_budget"])
        lines.append("        c.scores      = %s;" % fmt_doubles(case["scores"]))
        lines.append("        c.chain       = %s;" % fmt_ints(case["chain"]))
        lines.append("        c.paths       = {%s};" % ", ".join(fmt_ints(p) for p in case["paths"]))
        lines.append("        c.node_rank   = %s;" % fmt_ints(case["node_rank"]))
        lines.append("        c.node_depth  = %s;" % fmt_ints(case["node_depth"]))
        lines.append("        c.node_parent = %s;" % fmt_ints(case["node_parent"]))
        lines.append("        c.truths      = {%s};" % ", ".join(fmt_ints(t) for t in case["truths"]))
        lines.append("        c.accepted_length = %s;" % fmt_ints(case["accepted_length"]))
        lines.append("        c.accepted_node   = %s;" % fmt_ints(case["accepted_node"]))
        lines.append("        cases.push_back(std::move(c));")
        lines.append("    }")
    lines.append("    return cases;")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace ninfer::test::ddtree_golden")
    lines.append("")

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    print("wrote %s (%d cases) from reference sha256 %s" % (args.out, len(cases), digest[:16]))
    for case in cases:
        print("  %-28s steps=%2d budget=%2d nodes=%2d paths=%d chain=%s" %
              (case["name"], case["steps"], case["node_budget"], len(case["node_rank"]),
               len(case["paths"]), case["chain"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
