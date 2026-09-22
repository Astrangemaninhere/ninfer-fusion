#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: DFlash2 candidate-tree layout for one drafted block (DDTree, slice 2)
 *
 * The selector (`dflash2_selector`) already publishes a full top_k x top_k
 * predecessor x successor score table per (batch element, step) and then walks ONE
 * row per step, which yields a single flat chain. This op runs the beam-L DDTree
 * builder over that SAME, already-computed table and publishes the layout a tree
 * verifier consumes: verify-column order, per-column parent, per-column depth
 * (equivalently the rope offset) and the per-column ancestor mask.
 *
 * It is ADDITIVE and DEFAULT OFF. It reads `candidates`/`scores` produced by
 * `dflash2_selector` and writes only its own scratch, so the deployed single-chain
 * walk and every existing numeric result are untouched. Nothing calls it yet: it is
 * the device-side half of the layout contract, landed next to the host half
 * (`include/ninfer/ops/dflash2_ddtree.h`, slice 1) and checked against it by
 * `tests/ops/test_dflash2_ddtree_beam.cpp`.
 *
 * Inputs
 *   `candidates` I32 [B,S,K], `scores` F32 [B,S,K,K] -- exactly the scratch the
 *   selector leaves behind (`include/ninfer/ops/dflash2_selector.h`). The flat score
 *   layout is b + batch * (s + steps*(p + K*c)); row p is the candidate RANK at step
 *   s-1 (the walk's `previous`), column c is the candidate rank at step s. Step 0 is
 *   the anchor row and only p == 0 is read, so the engine's -inf in its p > 0 rows is
 *   ignored rather than trusted. `anchors` is deliberately NOT an input: the layout
 *   depends on the lattice and the candidate ids only.
 *
 * Outputs (caller-owned scratch, all fully overwritten; one row per batch element)
 *   `node_rank`      I32 [B,max_nodes] candidate rank of each node
 *   `node_parent`    I32 [B,max_nodes] node index of the (depth-1) prefix, -1 at depth 0
 *   `node_depth`     I32 [B,max_nodes] 0-based draft step
 *   `node_token`     I32 [B,max_nodes] global token id of `node_rank` at `node_depth`
 *   `node_ancestors` I64 [B,max_nodes] bit j set iff column j may be attended to
 *   `node_count`     I32 [B]           number of valid nodes, -1 if the invariant broke
 *
 * Column convention (same as slice 1): column 0 is the anchor column, and node i is
 * column i + 1, so the verify width is `node_count + 1`. Nodes are ordered by depth
 * ascending and then by candidate-rank prefix lexicographically, which makes a parent
 * index always smaller than its children's and every column's ancestor set a chain
 * ending at itself -- never a sibling. `node_ancestors` is that chain as bits
 * (bit 0 = the anchor column, always set).
 *
 * Build rule (`node_budget`): L = clamp(node_budget / S, 1, K) paths are kept by
 * cumulative selector score with the engine walk's tie-break (score desc, candidate
 * rank asc, beam index asc), and the deployed greedy chain is always seeded into the
 * result, so a tree can never accept less than the chain the engine runs today. The
 * tree is the union of the kept paths and holds at most `node_budget` nodes; with
 * S = 7 and a budget of 14 that is L = 2, which fits today's width-16 verify
 * (1 anchor + 14 nodes) while a wider budget does not.
 *
 * Registered domain: B = 1..8, S = 1..15, K = 16, node_budget in [S, 63], and the
 * node arrays at least `node_budget` wide (and at most 63, the ancestor-mask width).
 * The op owns no workspace and keeps no persistent state.
 *
 * WHAT THIS OP DOES NOT DO
 *   It does not widen the verify pass, apply the mask, place the columns at their rope
 *   positions or compute the accepted length. Those are the verify-side changes
 *   (slice 3) and they change an existing kernel contract. It also does not decide
 *   whether a tree is worth its columns: the offline EAL numbers that motivated the
 *   shape are builder coverage rather than draft acceptance, and the width cost has
 *   never been measured for the dflash2 geometry.
 */
void dflash2_tree_walk(const Tensor& candidates, const Tensor& scores, Tensor& node_rank,
                       Tensor& node_parent, Tensor& node_depth, Tensor& node_token,
                       Tensor& node_ancestors, Tensor& node_count, std::int32_t steps,
                       std::int32_t top_k, std::int32_t node_budget, cudaStream_t stream);

} // namespace ninfer::ops
