#pragma once

// Host + DEVICE DDTree beam core for the DFlash2 selector lattice (slice 2).
//
// WHY THIS HEADER EXISTS
//   include/ninfer/ops/dflash2_ddtree.h (slice 1) pins the layout contract on the
//   host with std::vector. The device walk a tree verify needs cannot use
//   std::vector, so the same search needs a fixed-capacity form that is
//   comparable to the slice-1 reference. This header IS that form: ONE
//   implementation, compiled both by the kernel
//   (src/ops/kernel/dflash2_tree_walk.cuh) and by a plain host test
//   (tests/ops/test_dflash2_ddtree_beam.cpp), so the device path and the golden
//   path cannot drift apart silently.
//
// NO-ENGINE RULE
//   Only <cstdint>. No CUDA type, no engine Tensor, no allocation, no persistent
//   state, no std container. Every buffer is caller-owned (BeamScratch), which is
//   what lets the kernel place it in shared memory and the host test place it on
//   the stack.
//
// SEMANTICS (matched to slice 1 / the S17-S19 reference builder, not reinvented)
//   score(s,p,c) of ONE batch element lives at flat offset
//       s + steps * (p + kBeamTopK * c)
//   identical to dflash2_selector_score_offset(b = 0, batch = 1, ...) in
//   src/ops/kernel/dflash2_selector.cuh:24-27 and to
//   ninfer::ops::ddtree::score_offset in include/ninfer/ops/dflash2_ddtree.h:108.
//   Row p = candidate RANK at step s-1 (the walk's `previous`), column c =
//   candidate RANK at step s, larger is better, and the column's unary score is
//   already included. Step 0 is the anchor row: only p == 0 is read, matching the
//   engine's "-inf into rows p > 0" contract (dflash2_selector.cuh:168-171) and
//   slice 1's forced row 0 -- callers may leave anything in those rows.
//   Tie-break everywhere: score desc, then candidate rank asc, then beam index asc
//   (dflash2_selector.cuh:242-252, "plain argmax, lowest rank breaking ties").
//   That comparator is a TOTAL order on the extension triples, so the kept top-L
//   set and its order do not depend on the sort algorithm -- which is why a
//   hand-rolled fixed-capacity insertion (this file) and std::sort (slice 1) must
//   agree exactly, and why the host test asserts set equality rather than
//   "close enough".
//   node_budget: L = clamp(node_budget / steps, 1, kBeamTopK), the greedy chain is
//   always seeded, and node_count <= node_budget holds on return (S19 hardening).
//
// COST
//   The beam loop is O(steps * L * kBeamTopK * L) with L <= 16 and steps <= 15, and
//   the two O(n^2) helpers below run over at most kBeamMaxPaths + 1 = 17 paths.
//   That is why one thread per batch row is enough (see the kernel); splitting a
//   row across a warp is an optimisation, not a requirement.
//
// NOT IN SCOPE
//   This produces the tree LAYOUT only: which verify columns exist, who is whose
//   parent, which columns each column attends to. It does not schedule the verify
//   pass, does not own scratch tensors, and does NOT decide whether a tree pays for
//   its nodes -- the offline EAL numbers that motivated the shape are builder
//   coverage rather than draft acceptance (eval_ddtree.py --tree caveat), and the
//   verify-width cost has never been measured for the dflash2 geometry.

#include <cstdint>

#if defined(__CUDACC__)
#define NINFER_DDTREE_HD __host__ __device__
#else
#define NINFER_DDTREE_HD
#endif

namespace ninfer::ops::ddtree {

// = ops::detail::kDflash2SelectorTopK (src/ops/launcher/dflash2_selector.h:13).
// tests/ops/test_dflash2_ddtree_beam.cpp static_asserts these against the slice-1
// constants; do not re-declare them anywhere else.
inline constexpr int kBeamTopK     = 16;
// Registered selector geometry is S = 1..15 (dflash2_selector.h:60, launcher check).
inline constexpr int kBeamMaxSteps = 15;
// One bit per verify column in a 64-bit ancestor mask => at most 63 nodes.
inline constexpr int kBeamMaxNodes = 63;
// One path per beam, plus the seeded greedy chain that may not be one of them.
inline constexpr int kBeamMaxPaths = kBeamTopK + 1;
inline constexpr std::int32_t kBeamRoot = -1;

// One kept path: the candidate rank chosen at each draft step.
struct BeamPath {
    std::int32_t ranks[kBeamMaxSteps] = {};
    std::int32_t length               = 0;
};

// One tree node = one non-empty rank prefix = one drafted token at one depth.
struct BeamNode {
    std::int32_t rank   = 0;         // candidate rank 0..kBeamTopK-1
    std::int32_t parent = kBeamRoot; // index of the (depth-1) prefix node
    std::int32_t depth  = 0;         // 0-based draft step
};

// Caller-owned working set, ~5.5 KB. The kernel puts it in shared memory, the host
// test on the stack. Deliberately a plain aggregate so both work.
struct BeamScratch {
    // Beams as of the step being processed.
    double       cur_cum[kBeamMaxPaths]                 = {};
    std::int32_t cur_own[kBeamMaxPaths][kBeamMaxSteps]  = {};
    std::int32_t cur_len[kBeamMaxPaths]                 = {};
    // Beams of the previous step, the extension sources.
    double       prev_cum[kBeamMaxPaths]                = {};
    std::int32_t prev_own[kBeamMaxPaths][kBeamMaxSteps] = {};
    std::int32_t prev_len[kBeamMaxPaths]                = {};
    // Extension candidates, sorted and truncated to L.
    std::int32_t ext_count                              = 0;
    double       ext_cum[kBeamMaxPaths * kBeamTopK]     = {};
    std::int32_t ext_last[kBeamMaxPaths * kBeamTopK]    = {};
    std::int32_t ext_src[kBeamMaxPaths * kBeamTopK]     = {};
    // The resulting path set.
    std::int32_t path_count                             = 0;
    std::int32_t path_len[kBeamMaxPaths + 1]            = {};
    std::int32_t path_ranks[kBeamMaxPaths + 1][kBeamMaxSteps] = {};
};

// Candidate token id of rank c at step s for batch element b: the engine layout
// dflash2_selector_candidate_offset(b, batch, s, steps, c) = b + batch*(s + steps*c)
// (src/ops/kernel/dflash2_selector.cuh:18-21).
NINFER_DDTREE_HD inline std::int64_t beam_candidate_offset(std::int32_t b, std::int32_t batch,
                                                           std::int32_t s, std::int32_t steps,
                                                           std::int32_t c) {
    return static_cast<std::int64_t>(b) +
           static_cast<std::int64_t>(batch) *
               (static_cast<std::int64_t>(s) + static_cast<std::int64_t>(steps) * c);
}

// Flat offset of scores[s][p][c] for one batch element (see SEMANTICS).
NINFER_DDTREE_HD inline std::int64_t beam_score_offset(std::int32_t s, std::int32_t steps,
                                                       std::int32_t p, std::int32_t c) {
    return static_cast<std::int64_t>(s) + static_cast<std::int64_t>(steps) *
                                              (static_cast<std::int64_t>(p) +
                                               static_cast<std::int64_t>(kBeamTopK) * c);
}

NINFER_DDTREE_HD inline float beam_score_at(const float* scores, std::int32_t steps,
                                            std::int32_t s, std::int32_t p, std::int32_t c) {
    return scores[beam_score_offset(s, steps, p, c)];
}

// Cumulative score of one path under the engine row semantics: step 0 always reads
// row 0 (the anchor), later steps read their predecessor's row. Double accumulator,
// same order as slice 1's ddtree::path_score.
NINFER_DDTREE_HD inline double beam_path_score(const float* scores, std::int32_t steps,
                                               const std::int32_t* ranks, std::int32_t length) {
    double total          = 0.0;
    std::int32_t previous = 0;
    for (std::int32_t s = 0; s < length; ++s) {
        const std::int32_t pred = s == 0 ? 0 : previous;
        total += static_cast<double>(beam_score_at(scores, steps, s, pred, ranks[s]));
        previous = ranks[s];
    }
    return total;
}

// True when path (a,a_len) must be ordered before (b,b_len): cumulative score desc,
// then lexicographic ranks, then shorter first. Same order as slice 1's seeding
// sort comparator.
NINFER_DDTREE_HD inline bool beam_path_less(const float* scores, std::int32_t steps,
                                            const std::int32_t* a, std::int32_t a_len,
                                            const std::int32_t* b, std::int32_t b_len) {
    const double sa = beam_path_score(scores, steps, a, a_len);
    const double sb = beam_path_score(scores, steps, b, b_len);
    if (sa != sb) { return sa > sb; }
    const std::int32_t n = a_len < b_len ? a_len : b_len;
    for (std::int32_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) { return a[i] < b[i]; }
    }
    return a_len < b_len;
}

// The deployed greedy chain: candidate rank per step, ties -> lowest candidate rank.
// Mirrors slice 1's ddtree::single_chain and dflash2_selector_walk_kernel greedy mode.
NINFER_DDTREE_HD inline void beam_single_chain(const float* scores, std::int32_t steps,
                                               std::int32_t* chain_out) {
    std::int32_t previous = 0;
    for (std::int32_t s = 0; s < steps; ++s) {
        const std::int32_t pred = s == 0 ? 0 : previous;
        std::int32_t best       = 0;
        for (std::int32_t c = 1; c < kBeamTopK; ++c) {
            // Strict >: ties keep the lowest candidate rank.
            if (beam_score_at(scores, steps, s, pred, c) >
                beam_score_at(scores, steps, s, pred, best)) {
                best = c;
            }
        }
        chain_out[s] = best;
        previous     = best;
    }
}

// Insert one extension into the top-L array kept sorted by (score desc, candidate
// rank asc, source beam asc), truncating to L. Exact because that comparator is a
// total order and callers feed extensions in generation order (source beam
// ascending, then candidate rank ascending): a full sort of the same multiset into
// a total order has one answer.
// The cumulative score is a DOUBLE, matching the validated slice-1 reference (whose
// table is double) and the python builder. The table itself stays FP32 because that
// is what the engine publishes; float -> double widening is exact, so the two
// implementations see identical values and, with -ffp-contract=off, identical sums.
NINFER_DDTREE_HD inline void beam_insert_extension(BeamScratch& scratch, std::int32_t keep,
                                                   double cum, std::int32_t last,
                                                   std::int32_t src) {
    const std::int32_t at = scratch.ext_count;
    std::int32_t pos      = 0;
    while (pos < at) {
        const double pc       = scratch.ext_cum[pos];
        const std::int32_t pl = scratch.ext_last[pos];
        const std::int32_t ps = scratch.ext_src[pos];
        bool before;
        if (cum != pc) {
            before = cum > pc;
        } else if (last != pl) {
            before = last < pl;
        } else {
            before = src < ps;
        }
        if (before) { break; }
        ++pos;
    }
    if (pos >= keep) { return; } // belongs at or past the truncation point
    const std::int32_t top = at < keep ? at : keep - 1;
    for (std::int32_t i = top; i > pos; --i) {
        scratch.ext_cum[i]  = scratch.ext_cum[i - 1];
        scratch.ext_last[i] = scratch.ext_last[i - 1];
        scratch.ext_src[i]  = scratch.ext_src[i - 1];
    }
    scratch.ext_cum[pos]  = cum;
    scratch.ext_last[pos] = last;
    scratch.ext_src[pos]  = src;
    if (at < keep) { scratch.ext_count = at + 1; }
}

// Distinct non-empty rank prefixes among the first `count` paths = the node count.
// Per depth: gather, insertion sort by prefix, count the first of each run of equal
// prefixes. Equal-length prefixes are the only ones that can be equal, which is why
// one sort per depth is exact.
NINFER_DDTREE_HD inline std::int32_t beam_node_count(const BeamScratch& scratch,
                                                     std::int32_t count,
                                                     std::int32_t steps) {
    std::int32_t total = 0;
    for (std::int32_t d = 1; d <= steps; ++d) {
        std::int32_t idx[kBeamMaxPaths + 1];
        std::int32_t n = 0;
        for (std::int32_t p = 0; p < count; ++p) {
            if (scratch.path_len[p] >= d) { idx[n++] = p; }
        }
        for (std::int32_t i = 1; i < n; ++i) {
            const std::int32_t pi = idx[i];
            std::int32_t j        = i;
            while (j > 0) {
                const std::int32_t pj = idx[j - 1];
                bool before           = false;
                bool diff             = false;
                for (std::int32_t t = 0; t < d; ++t) {
                    if (scratch.path_ranks[pi][t] != scratch.path_ranks[pj][t]) {
                        before = scratch.path_ranks[pi][t] < scratch.path_ranks[pj][t];
                        diff   = true;
                        break;
                    }
                }
                if (!diff || !before) { break; }
                idx[j] = pj;
                --j;
            }
            idx[j] = pi;
        }
        for (std::int32_t i = 0; i < n; ++i) {
            bool duplicate = false;
            if (i > 0) {
                const std::int32_t pa = idx[i - 1];
                const std::int32_t pb = idx[i];
                bool same             = true;
                for (std::int32_t t = 0; same && t < d; ++t) {
                    same = scratch.path_ranks[pa][t] == scratch.path_ranks[pb][t];
                }
                duplicate = same;
            }
            if (!duplicate) { ++total; }
        }
    }
    return total;
}

// Beam-L search. Fills scratch.path_count / path_len / path_ranks with the kept
// paths in the reference builder's order and returns the path count; returns 0 on
// invalid input.
NINFER_DDTREE_HD inline std::int32_t beam_build(const float* scores, std::int32_t steps,
                                                std::int32_t node_budget,
                                                BeamScratch& scratch) {
    if (scores == nullptr || steps < 1 || steps > kBeamMaxSteps) { return 0; }
    if (node_budget < steps || node_budget > kBeamMaxNodes) { return 0; }
    std::int32_t l = node_budget / steps;
    if (l < 1) { l = 1; }
    if (l > kBeamTopK) { l = kBeamTopK; }

    scratch.path_count = 0;
    if (l == 1) {
        scratch.path_len[0] = steps;
        beam_single_chain(scores, steps, scratch.path_ranks[0]);
        scratch.path_count = 1;
        return scratch.path_count;
    }

    std::int32_t ns    = 1;
    scratch.cur_cum[0] = 0.0;
    scratch.cur_len[0] = 0;
    for (std::int32_t s = 0; s < steps; ++s) {
        for (std::int32_t i = 0; i < ns; ++i) {
            scratch.prev_cum[i] = scratch.cur_cum[i];
            scratch.prev_len[i] = scratch.cur_len[i];
            for (std::int32_t d = 0; d < scratch.cur_len[i]; ++d) {
                scratch.prev_own[i][d] = scratch.cur_own[i][d];
            }
        }
        scratch.ext_count = 0;
        for (std::int32_t bi = 0; bi < ns; ++bi) {
            // Step 0 reads the anchor row only (engine contract).
            const std::int32_t pred =
                s == 0 ? 0 : scratch.prev_own[bi][scratch.prev_len[bi] - 1];
            for (std::int32_t c = 0; c < kBeamTopK; ++c) {
                const double cum = scratch.prev_cum[bi] +
                                   static_cast<double>(beam_score_at(scores, steps, s, pred, c));
                beam_insert_extension(scratch, l, cum, c, bi);
            }
        }
        const std::int32_t keep = scratch.ext_count < l ? scratch.ext_count : l;
        if (keep <= 0) { break; }
        for (std::int32_t i = 0; i < keep; ++i) {
            const std::int32_t src = scratch.ext_src[i];
            const std::int32_t len = scratch.prev_len[src] + 1;
            scratch.cur_cum[i]     = scratch.ext_cum[i];
            for (std::int32_t d = 0; d + 1 < len; ++d) {
                scratch.cur_own[i][d] = scratch.prev_own[src][d];
            }
            scratch.cur_own[i][len - 1] = scratch.ext_last[i];
            scratch.cur_len[i]          = len;
        }
        ns = keep;
    }

    // In-progress beams, de-duplicated in beam order (identical prefixes can survive
    // as separate beams), then the S19 chain seeding.
    std::int32_t pc = 0;
    for (std::int32_t i = 0; i < ns; ++i) {
        bool duplicate = false;
        for (std::int32_t j = 0; j < pc && !duplicate; ++j) {
            if (scratch.path_len[j] != scratch.cur_len[i]) { continue; }
            bool same = true;
            for (std::int32_t d = 0; same && d < scratch.cur_len[i]; ++d) {
                same = scratch.path_ranks[j][d] == scratch.cur_own[i][d];
            }
            duplicate = same;
        }
        if (duplicate) { continue; }
        scratch.path_len[pc] = scratch.cur_len[i];
        for (std::int32_t d = 0; d < scratch.cur_len[i]; ++d) {
            scratch.path_ranks[pc][d] = scratch.cur_own[i][d];
        }
        ++pc;
    }

    std::int32_t chain[kBeamMaxSteps] = {};
    beam_single_chain(scores, steps, chain);
    bool chain_is_beam = false;
    for (std::int32_t j = 0; j < pc && !chain_is_beam; ++j) {
        bool same = scratch.path_len[j] == steps;
        for (std::int32_t d = 0; same && d < steps; ++d) {
            same = scratch.path_ranks[j][d] == chain[d];
        }
        chain_is_beam = same;
    }
    if (chain_is_beam) {
        scratch.path_count = pc;
        return pc;
    }

    // The chain is not a beam: seed it, re-sort everything by (score desc,
    // lexicographic ranks) and drop the worst non-chain path until the node budget
    // holds. Identical to slice 1's seeding branch -- including that the re-sort only
    // happens on this branch, so the plain beam order stands when the chain was
    // already present.
    scratch.path_len[pc] = steps;
    for (std::int32_t d = 0; d < steps; ++d) {
        scratch.path_ranks[pc][d] = chain[d];
    }
    ++pc;
    for (std::int32_t i = 1; i < pc; ++i) {
        const std::int32_t li = scratch.path_len[i];
        std::int32_t ri[kBeamMaxSteps];
        for (std::int32_t d = 0; d < li; ++d) {
            ri[d] = scratch.path_ranks[i][d];
        }
        std::int32_t j = i;
        while (j > 0 &&
               beam_path_less(scores, steps, ri, li, scratch.path_ranks[j - 1],
                              scratch.path_len[j - 1])) {
            scratch.path_len[j] = scratch.path_len[j - 1];
            for (std::int32_t d = 0; d < scratch.path_len[j]; ++d) {
                scratch.path_ranks[j][d] = scratch.path_ranks[j - 1][d];
            }
            --j;
        }
        scratch.path_len[j] = li;
        for (std::int32_t d = 0; d < li; ++d) {
            scratch.path_ranks[j][d] = ri[d];
        }
    }
    while (beam_node_count(scratch, pc, steps) > node_budget && pc > 1) {
        std::int32_t drop = pc;
        for (std::int32_t j = pc; j-- > 0;) {
            bool is_chain = scratch.path_len[j] == steps;
            for (std::int32_t d = 0; is_chain && d < steps; ++d) {
                is_chain = scratch.path_ranks[j][d] == chain[d];
            }
            if (!is_chain) {
                drop = j;
                break;
            }
        }
        if (drop == pc) { break; } // only the chain is left; steps <= node_budget holds
        for (std::int32_t j = drop; j + 1 < pc; ++j) {
            scratch.path_len[j] = scratch.path_len[j + 1];
            for (std::int32_t d = 0; d < scratch.path_len[j]; ++d) {
                scratch.path_ranks[j][d] = scratch.path_ranks[j + 1][d];
            }
        }
        --pc;
    }
    scratch.path_count = pc;
    return pc;
}

// Materialize the tree in slice 1's node order: depth ascending, and within a depth
// lexicographic by rank prefix. Parents therefore always precede their children and
// live in the previous depth's contiguous index range [group_start[d-1], ...).
// Returns the node count, or kBeamRoot (-1) when the result does not fit in
// `max_nodes`.
NINFER_DDTREE_HD inline std::int32_t beam_nodes(const BeamScratch& scratch, std::int32_t steps,
                                                std::int32_t max_nodes, BeamNode* nodes) {
    if (nodes == nullptr || max_nodes < 1) { return kBeamRoot; }
    std::int32_t node_src[kBeamMaxNodes];
    std::int32_t group_start[kBeamMaxSteps + 1] = {};
    std::int32_t group_count[kBeamMaxSteps + 1] = {};
    std::int32_t count                          = 0;
    for (std::int32_t d = 1; d <= steps; ++d) {
        std::int32_t idx[kBeamMaxPaths + 1];
        std::int32_t n = 0;
        for (std::int32_t p = 0; p < scratch.path_count; ++p) {
            if (scratch.path_len[p] >= d) { idx[n++] = p; }
        }
        // Stable-by-construction insertion sort of path indices by their depth-d
        // prefix; equal-length prefixes sort into contiguous runs.
        for (std::int32_t i = 1; i < n; ++i) {
            const std::int32_t pi = idx[i];
            std::int32_t j        = i;
            while (j > 0) {
                const std::int32_t pj = idx[j - 1];
                bool before           = false;
                bool diff             = false;
                for (std::int32_t t = 0; t < d; ++t) {
                    if (scratch.path_ranks[pi][t] != scratch.path_ranks[pj][t]) {
                        before = scratch.path_ranks[pi][t] < scratch.path_ranks[pj][t];
                        diff   = true;
                        break;
                    }
                }
                if (!diff || !before) { break; }
                idx[j] = pj;
                --j;
            }
            idx[j] = pi;
        }
        group_start[d]       = count;
        std::int32_t emitted = 0;
        for (std::int32_t i = 0; i < n; ++i) {
            if (i > 0) {
                const std::int32_t pa = idx[i - 1];
                const std::int32_t pb = idx[i];
                bool same             = true;
                for (std::int32_t t = 0; same && t < d; ++t) {
                    same = scratch.path_ranks[pa][t] == scratch.path_ranks[pb][t];
                }
                if (same) { continue; } // adjacent duplicate prefix
            }
            if (count >= max_nodes) { return kBeamRoot; }
            const std::int32_t pi = idx[i];
            BeamNode node;
            node.depth = d - 1;
            node.rank  = scratch.path_ranks[pi][d - 1];
            if (d == 1) {
                node.parent = kBeamRoot;
            } else {
                std::int32_t parent = kBeamRoot;
                for (std::int32_t j = 0; j < group_count[d - 1]; ++j) {
                    const std::int32_t cand = group_start[d - 1] + j;
                    bool same               = true;
                    for (std::int32_t t = 0; same && t < d - 1; ++t) {
                        same = scratch.path_ranks[node_src[cand]][t] ==
                               scratch.path_ranks[pi][t];
                    }
                    if (same) {
                        parent = cand;
                        break;
                    }
                }
                node.parent = parent;
            }
            nodes[count]  = node;
            node_src[count] = pi;
            ++count;
            ++emitted;
        }
        group_count[d] = emitted;
    }
    return count;
}

// Per-column ancestor set of column `node_index + 1` (column 0 is the anchor and
// sees only itself): bit j set iff column j may be attended to. Same value as slice
// 1's ddtree::column_ancestors.
NINFER_DDTREE_HD inline std::uint64_t beam_node_ancestors(const BeamNode* nodes,
                                                          std::int32_t node_index) {
    std::uint64_t bits = 1ULL; // the anchor column, always visible
    for (std::int32_t node = node_index; node != kBeamRoot; node = nodes[node].parent) {
        bits |= 1ULL << static_cast<unsigned>(node + 1);
    }
    return bits;
}

} // namespace ninfer::ops::ddtree
