#pragma once

// ============================================================================
// The PROPOSAL-SIDE FILL of the MTP draft-tree publication (--draft-tree L,d, L > 1).
//
// WHY THIS FILE EXISTS
//   The receiving half (mtp_tree_publish.h) validates and stores a tree layout, and the
//   producing half (mtp_tree_produce.h) builds one out of a candidate lattice. Both are inert
//   until something PUSHES the lattice out of the proposal loop, and nothing did: the verify
//   side refuses an L > 1 round whose tree was never published
//   (src/targets/qwen3_6/impl/runtime/program_impl.h:13336 @3944a53) and, as the producing half
//   states in its own header, `sequence.mtp_tree_columns` still reads 0 in a real run because
//   "the proposal side must publish per-column depths and ancestor masks". This header is that
//   push. It is kept host-only and engine-free so the whole thing stays pinnable by a host test.
//
// THE SHAPE IT PUBLISHES
//   The SIBLING shape of mtp_tree_produce.h (mtp_tree_columns_from_siblings): per draft depth s
//   in [0, d) the proposal head's own top-L candidate ids, sibling 0 continuing as the spine.
//   Its trade-off is declared over there and is restated here so it cannot be lost in the gap
//   between the two files: "(2) 是覆盖率下降（非 spine 兄弟不展开），不是替代 —— 它不发明、不复制
//   任何分数，每个 token 都是真实 proposal 前向的真实输出". This is NOT a complete beam-L tree
//   and must not be described as one.
//
// WHERE THE LATTICE COMES FROM, AND WHAT THE CROSS-CHECK DOES AND DOES NOT PROVE
//   ONE device source, read on the host as two tables of the SAME round and the SAME row:
//     * ids[s * paths + i] -- ops::mtp_proposal_topk over the depth-s proposal row
//       (include/ninfer/ops/mtp_proposal_topk.h), one extraction per depth;
//     * chain[s]           -- the ordinary chain draft for depth s, which is that SAME row's
//       argmax (the ops::argmax inside TextContext::proposal_argmax).
//   They are computed by two different kernels FROM THE SAME LOGITS ROW -- the same tensor, the
//   same row window and the same forward (mtp_impl.h passes TextConfig::token_domain as the
//   extraction's row count precisely because that is the window the argmax path reads). So
//   ids[s * paths] == chain[s] is a SAME-SOURCE CONSISTENCY check and NOT two independent sources.
//   The "two independent device outputs" reading is RETIRED here: an earlier revision of the
//   report kept only the "independent" half of the sentence that used to sit next to it
//   ("computed by different kernels from the same logits row"), and that half is not a property
//   of this design.
//   What it DOES prove, which is not nothing: the extraction RAN for THIS round and landed in
//   THIS round's frame. A depth whose ids block was never written (a bridge that skipped the
//   extraction; a reused frame still holding the previous round's block) carries the previous
//   round's numbers, and those would have to coincide with this round's argmax at every depth to
//   pass; a depth where the two kernels split a tie is likewise not the row the chain draft came
//   from. A mismatch is refused instead of publishing a lattice this round's own device work does
//   not back -- see mtp_proposal_ready_steps().
//   What it does NOT prove: that the numbers ARE the extraction's output. No host-side check can
//   prove that, and the note above mtp_proposal_row_defect() names exactly which classes are
//   undecidable instead of pretending a rule covers them.
//   (The cross-check is usable at all because the two kernels agree on ties: include/ninfer/ops/
//   argmax.h documents "equal maxima select the lowest row index", and
//   src/ops/kernel/mtp_proposal_topk.cuh merges "value descending, then the lowest row index".
//   They were read, not assumed.)
//
// THE PLANNING-TIME LANDMINE THIS HEADER DISARMS
//   mtp_proposal_topk returns ROW INDICES of the table it is handed. That is a global token id
//   ONLY for the FULL proposal head. The OPTIMIZED (shortlist) head never even writes the
//   round's proposal-logits frame -- TextContext::proposal_argmax allocates its own
//   {proposal_head_n_, T} scratch and remaps the argmax through proposal_head_ids_
//   (src/targets/qwen3_6/impl/runtime/text_context_impl.h:697-703) -- so publishing its indices
//   as tree nodes is a silent wrong answer twice over (stale/unwritten rows AND shortlist
//   positions published as token ids). The planner therefore refuses a tree round on a shortlist
//   head: mtp_proposal_head_defect() below, called from layouts_impl.h beside the
//   L*d <= 15 node bound.
//
// WHAT IS NOT IMPLEMENTED HERE (named design, with its cost)
//   A TRUE beam-L lattice (mtp_tree_produce.h's mtp_tree_columns_from_lattice) needs L real
//   forwards PER DEPTH, because a rank p row is the distribution conditioned on candidate rank p
//   at the previous depth and cannot be inferred from another rank's row. The MTP draft loop runs
//   exactly ONE forward per depth, and what blocks the other L-1 is that two ingress arrays are
//   per-LANE rather than per-beam:
//     * src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h:135
//       MtpDecodeIngress::mtp_kv_table_rows -- ONE MTP KV table row per lane;
//     * round_state.h:136-137 MtpDecodeIngress::state_source_slots /
//       state_destination_slots -- ONE state slot pair per lane.
//   A beam-L producer needs L rows and 2L slots per row, i.e. a widened ingress contract plus a
//   per-beam proposal-side KV/state write. Not attempted here; the sibling shape needs none of it
//   (no new forward, no new KV row, no new state slot).
// ============================================================================

#include "targets/qwen3_6/impl/runtime/mtp_tree_produce.h"

#include <ninfer/ops/dflash2_ddtree.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// ---------------------------------------------------------------------------------------
// The PLANNING-TIME refusal (layouts_impl.h, beside the L*d <= 15 node bound).
//
//   shortlist_head : the run selected the Optimized (shortlist) proposal head, i.e.
//                    ProposalHead != ProposalHead::Full.
//   paths, depth   : --draft-tree's L and d. L <= 1 is the chain spelling and is NOT a tree
//                    round, so the guard must not fire for it -- firing there would refuse every
//                    ordinary --draft-tokens run that happens to use the shortlist head.
//
// "" when the configuration may run an L > 1 tree, else the defect. A string rather than a bool
// so the refusal says what it found, in the style of the rest of the runtime.
// ---------------------------------------------------------------------------------------
[[nodiscard]] inline std::string mtp_proposal_head_defect(bool shortlist_head,
                                                               std::uint32_t paths,
                                                               std::uint32_t depth) {
    if (paths <= 1 || depth == 0) { return std::string(); }
    if (shortlist_head) {
        return "--draft-tree L,d with L > 1 needs the FULL proposal head: the shortlist "
               "(Optimized) head never writes the round's proposal-logits frame, so the top-L "
               "row indices are shortlist positions that still have to be remapped, not token "
               "ids. Publishing them as tree nodes would be a silent wrong answer. Use the full "
               "proposal head, or run the round as a chain (L <= 1).";
    }
    return std::string();
}

// ---------------------------------------------------------------------------------------
// One lane's lattice, owned by value so a reader can return it (the device buffers it was
// compacted from are overwritten by the next round).
// ---------------------------------------------------------------------------------------
struct MtpProposalRow {
    std::uint32_t paths = 0; // --draft-tree's L
    std::uint32_t steps = 0; // --draft-tree's d
    // ids[s * paths + i]: the (i + 1)-th best candidate id of depth s's proposal row, best
    // first. ids[s * paths] is the SPINE of level s and must equal chain[s].
    std::vector<TokenId> ids;
    // chain[s]: the id the ordinary chain draft carried at depth s -- the same row's argmax.
    std::vector<TokenId> chain;
    // The proposal head's own ROW RANGE: only ids in [0, token_domain) are tokenizer-addressable
    // rows the head can emit, i.e. the window ops::mtp_proposal_topk is handed as `rows`
    // (TextConfig::token_domain == qwen3_6::kTokenDomain == 248077, frontend.h). Every node token
    // must lie inside it. 0 means "not stated" and is REFUSED by mtp_proposal_row_defect(): a
    // reader that does not know the head's row range cannot check the one property about a single
    // node token that this layer is able to check at all, and a gate that silently skips its
    // strongest rule when a field is left out is worse than no gate.
    std::uint32_t token_domain = 0;

    // How many depths the two tables can both address.
    [[nodiscard]] std::uint32_t available_steps() const noexcept {
        const std::uint32_t by_ids =
            paths == 0 ? 0U : static_cast<std::uint32_t>(ids.size() / paths);
        const std::uint32_t by_chain = static_cast<std::uint32_t>(chain.size());
        return by_ids < by_chain ? by_ids : by_chain;
    }
};

// The longest prefix of depths whose spine IS the chain draft of that depth. This is the
// round's "how many depths did the proposal side really publish" count. It is READ OUT of the
// round's own device frame rather than asserted by the caller -- a count the writer also chooses
// would check nothing -- but it is a SAME-SOURCE check (see the header): it establishes that the
// extraction ran for this round and agrees with the argmax, not that a second, independent
// computation confirms the number.
[[nodiscard]] inline std::uint32_t mtp_proposal_ready_steps(const MtpProposalRow& row) noexcept {
    if (row.paths == 0) { return 0; }
    const std::uint32_t steps = row.available_steps();
    std::uint32_t ready        = 0;
    while (ready < steps) {
        if (row.ids[static_cast<std::size_t>(ready) * row.paths] != row.chain[ready]) { break; }
        ++ready;
    }
    return ready;
}

// "" when `row` is a lattice this producer may publish for a round whose node budget is
// `node_budget` (= draft_window = L * d), else the defect.
//
// THE TRUST BOUNDARY OF THIS GATE -- stated here, where the checks are, not only in a report.
//   The gate is given six things: paths, steps, token_domain, ids[], chain[] and the budget.
//   Every check below is decidable from them. Three classes of wrong lattice are NOT, and naming
//   them is the only honest option left (an audit that found them is right about the LATTICE and
//   wrong about the GATE):
//
//   (a) IN-DOMAIN PROVENANCE AT RANK >= 1. An in-domain id that no extraction produced is not
//       distinguishable from a real one at this layer: a real extraction's rank-i id is an
//       arbitrary in-domain id, and nothing relates rank i of one depth's row to anything else.
//       Only the extraction (running on the device, holding the logits row) knows. The engine's
//       BASELINE trust model is weaker still -- the chain path reads SequenceState::mtp_drafts,
//       the same device array, with no check of any kind -- so this gate already dominates it.
//       (An instrument that would move the boundary: a per-rank SCORE column read back beside the
//       ids, which ops::mtp_proposal_topk does not emit today. It would let the gate require a
//       non-increasing rank order; it still could not prove the ids are the argmax rows, and it
//       costs a second D2H column per depth.)
//   (b) A REPEATED TOKEN ALONG AN ANCESTOR PATH -- a node whose token equals its PARENT's token,
//       and the extreme case where every level's spine repeats. These are NOT defects and MUST NOT
//       be refused. The MTP draft loop feeds depth s-1's prediction back as depth s's input token
//       (mtp_impl.h's draft loop: next_drafts[s - 1] is the batch the next forward consumes), so a
//       repeated argmax is an ordinary prediction -- and in a REPETITION LOOP, where speculative
//       decoding has its highest acceptance, the spine repeats at EVERY level. A rule banning it
//       would refuse the tree exactly where a tree pays off most, i.e. it would be a false-positive
//       generator on a legal path, which is the one thing the sibling shape's own trade-off note
//       forbids. The reference contract agrees that this is not a token question at all:
//       ddtree::build_tree's node identity is the RANK prefix (count_nodes() collapses equal
//       prefixes, never equal tokens) and ddtree::column_depth / column_ancestors are token-blind.
//   (c) AN ID THAT REPEATS ANOTHER DEPTH'S ID (a sibling is already required to differ from its
//       own level's siblings). Same argument as (a).
//
//   The receiving seam cannot see any of them either (mtp_tree_layout_defect validates the layout,
//   not the tokens) and neither can the verify side, which reads column depths and masks.
[[nodiscard]] inline std::string mtp_proposal_row_defect(const MtpProposalRow& row,
                                                        std::uint32_t node_budget) {
    if (row.paths < 2) {
        return "fewer than two paths is the chain spelling, not a tree: a tree producer needs "
               "L >= 2";
    }
    if (row.steps < 1) { return "the proposal row has no depths"; }
    if (row.steps > 15) {
        return "the proposal row has more depths than the registered ddtree geometry (1..15)";
    }
    const std::size_t want =
        static_cast<std::size_t>(row.paths) * static_cast<std::size_t>(row.steps);
    if (row.ids.size() != want) { return "the id table is not paths * steps wide"; }
    if (row.chain.size() < row.steps) {
        return "the chain draft table is shorter than the tree depth";
    }
    if (want > static_cast<std::size_t>(ops::ddtree::kMaxNodes)) {
        return "the proposal row needs more nodes than the 64-bit ancestor mask can index";
    }
    if (static_cast<std::uint32_t>(want) != node_budget) {
        return "the proposal row's node count is not the round's node budget (the frame the "
               "verify side will run is drafted for a different tree)";
    }
    if (row.token_domain == 0) {
        return "the row does not state the proposal head's token domain: without it no node token "
               "can be tied to the rows that head can emit, and the one single-token correctness "
               "statement this layer can make would be skipped silently. Pass "
               "TextConfig::token_domain at the read (mtp_proposal_row_from_egress / "
               "mtp_proposal_row_from_prefill).";
    }
    if (row.token_domain > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        return "the stated token domain does not fit the id type ops::mtp_proposal_topk returns";
    }
    const std::uint32_t ready = mtp_proposal_ready_steps(row);
    if (ready < row.steps) {
        return "the lattice is not published for depth " + std::to_string(ready) +
               ": that depth's top-L spine is not the argmax the chain draft carries. Either the "
               "extraction did not run (the shortlist proposal head never writes this frame) or "
               "the two extractions split a tie. Refusing instead of publishing a lattice no "
               "device output backs.";
    }
    // Per-level token legality. An independent audit of both halves (t-mtp-tree9x) found node
    // tokens are WRITE-ONLY in them: 14 equal ids, a child equal to its parent, -13 and
    // INT32_MAX were all accepted. Of those four, TWO are decidable here and are refused below;
    // the other two are named at the trust-boundary note above rather than pretended away.
    //   (a) THE ROW RANGE. Every node token must lie in [0, token_domain). This is a PROVABLE
    //       property of the real path and therefore a refusal with no false positives:
    //       ops::mtp_proposal_topk only ever writes row indices of the window it is handed
    //       (src/ops/launcher/mtp_proposal_topk.cu: effective = min(top_l, rows), and the kernel
    //       writes `s_idx[best]`, a row index in [0, rows)), and BOTH call sites pass
    //       rows == TextConfig::token_domain. So an id outside that range is not a proposal output
    //       at all -- it is a fabricated value or a mis-indexed read.
    //   (b) DISTINCTNESS WITHIN A LEVEL. Two siblings cannot carry the same id: the merge consumes
    //       each row index it selects. The reason this rule is kept is NOT the one an earlier
    //       revision gave: the tail-repeat branch of ops::mtp_proposal_topk needs rows < top_l, and
    //       both call sites pass rows == TextConfig::token_domain (248077) with top_l <= 15, so
    //       `effective == top_l` and that branch is UNREACHABLE from this tree. The rule stays as
    //       a defence for the day a caller hands the op a narrower window; today a violation means
    //       the op's own contract was broken, which is exactly why it is worth refusing.
    for (std::uint32_t s = 0; s < row.steps; ++s) {
        for (std::uint32_t i = 0; i < row.paths; ++i) {
            const TokenId id = row.ids[static_cast<std::size_t>(s) * row.paths + i];
            if (id < 0) {
                return "level " + std::to_string(s) + " sibling " + std::to_string(i) +
                       " carries a negative token id";
            }
            if (static_cast<std::uint32_t>(id) >= row.token_domain) {
                return "level " + std::to_string(s) + " sibling " + std::to_string(i) +
                       " carries token id " + std::to_string(static_cast<long long>(id)) +
                       ", which is outside the proposal head's token domain (0.." +
                       std::to_string(row.token_domain - 1U) +
                       "): ops::mtp_proposal_topk returns row indices of the window it was handed "
                       "and both call sites pass rows == TextConfig::token_domain, so no "
                       "extraction could have produced it";
            }
            for (std::uint32_t j = 0; j < i; ++j) {
                if (row.ids[static_cast<std::size_t>(s) * row.paths + j] == id) {
                    return "level " + std::to_string(s) + " publishes the same id twice (siblings " +
                           std::to_string(j) + " and " + std::to_string(i) +
                           "): the top-L extraction consumes each row index it selects, so a "
                           "repeated index means the op's own contract was broken -- it cannot come "
                           "from asking for more rows than the window has, because both call sites "
                           "pass rows == TextConfig::token_domain >> top_l";
                }
            }
        }
    }
    return std::string();
}

// The lattice of one lane's row, checked and rewritten into the producer's own contract.
[[nodiscard]] inline MtpSiblingLattice mtp_sibling_lattice_from_row(const MtpProposalRow& row,
                                                                   std::uint32_t node_budget) {
    const std::string defect = mtp_proposal_row_defect(row, node_budget);
    if (!defect.empty()) {
        throw std::invalid_argument("MTP proposal lattice rejected: " + defect);
    }
    MtpSiblingLattice lattice;
    lattice.paths = static_cast<std::int32_t>(row.paths);
    lattice.steps = static_cast<std::int32_t>(row.steps);
    lattice.ids.assign(row.ids.begin(), row.ids.end());
    return lattice;
}

// The publish slot of one SequenceState (src/targets/qwen3_6/impl/runtime/program.h:463/472-474).
// Templated on the sequence type so this header keeps the "no engine type" discipline of
// mtp_tree_publish.h while the field-for-field mapping lives in ONE place instead of being
// re-spelled at every call site. The published capacities are read off the arrays themselves, so
// the slot can never claim more room than the sequence has.
template <typename SequenceLike>
[[nodiscard]] inline MtpTreePublishSlot mtp_tree_publish_slot(SequenceLike& sequence) noexcept {
    MtpTreePublishSlot slot;
    slot.columns        = &sequence.mtp_tree_columns;
    slot.depths         = sequence.mtp_tree_depths.data();
    slot.masks          = sequence.mtp_tree_masks.data();
    slot.drafts         = sequence.mtp_drafts.data();
    slot.array_capacity = static_cast<std::uint32_t>(sequence.mtp_tree_depths.size());
    slot.draft_capacity = static_cast<std::uint32_t>(sequence.mtp_drafts.size());
    return slot;
}

// THE ONE CALL a tree round's hand-off makes: check the lattice, build the verify-column layout
// through the producing half, publish it through the receiving half, and return the live extent
// the round must carry (mtp_tree_required_extent() states why it is the node count).
//
// `node_budget` is L*d, `width` is the verify frame's column budget (draft_window + 1). The two
// invariants the round must leave behind are re-stated here rather than assumed --
// publish_mtp_tree_round() already refuses columns != width, and the extent check below is
// mtp_tree_required_extent() so that a change to either half cannot pass silently.
[[nodiscard]] inline std::uint32_t fill_mtp_tree_round(const MtpProposalRow& row,
                                                      std::uint32_t node_budget,
                                                      std::uint32_t width,
                                                      const MtpTreePublishSlot& slot) {
    const MtpSiblingLattice lattice = mtp_sibling_lattice_from_row(row, node_budget);
    const std::uint32_t extent      = publish_mtp_tree_round(lattice, node_budget, width, slot);
    if (extent != node_budget || extent != static_cast<std::uint32_t>(*slot.columns) - 1U) {
        throw std::logic_error(
            "MTP proposal fill published an extent that is not the node count");
    }
    return extent;
}

// ---------------------------------------------------------------------------------------
// Reading the round's egress
//
// MtpDecodeEgress::next_proposal_ids is laid out by the DEVICE side
// (src/targets/qwen3_6/impl/runtime/mtp_impl.h) as
//     entry (depth s, rank i, lane t) at  s * depth_stride + i * lanes + t
// because ops::mtp_proposal_topk stores its per-token rows with stride `tokens`, and `tokens` is
// the round's batch. `depth_stride` is therefore qwen3_6::kMtpTreeProposalDepthStride (declared
// next to the array), and `lanes` is the round's batch -- NOT the array's lane capacity.
// ---------------------------------------------------------------------------------------
[[nodiscard]] inline constexpr std::size_t mtp_tree_proposal_index(std::uint32_t depth,
                                                                  std::uint32_t rank,
                                                                  std::uint32_t lanes,
                                                                  std::uint32_t lane,
                                                                  std::size_t depth_stride) noexcept {
    return static_cast<std::size_t>(depth) * depth_stride +
           static_cast<std::size_t>(rank) * lanes + lane;
}

// One lane's lattice, compacted out of the round's egress. `ids` points at the whole
// next_proposal_ids array (not at the lane's slice) and `chain` at next_drafts, whose per-depth
// stride is `chain_stride` (== the round's max_concurrency, the same stride the hand-off uses to
// copy the chain drafts into SequenceState::mtp_drafts). `token_domain` is the row count the
// extraction was handed (TextConfig::token_domain) and is what every node token is checked
// against; it is a parameter of the READ because the read is where the head is known.
[[nodiscard]] inline MtpProposalRow mtp_proposal_row_from_egress(const std::int32_t* ids,
                                                                const TokenId* chain,
                                                                std::uint32_t lanes,
                                                                std::uint32_t lane,
                                                                std::uint32_t paths,
                                                                std::uint32_t steps,
                                                                std::size_t depth_stride,
                                                                std::size_t chain_stride,
                                                                std::uint32_t token_domain) {
    if (ids == nullptr || chain == nullptr) {
        throw std::logic_error("MTP proposal egress is missing");
    }
    if (lanes == 0 || lane >= lanes || paths < 2 || steps < 1 || depth_stride == 0) {
        throw std::logic_error("MTP proposal egress read is outside the round's shape");
    }
    if (token_domain == 0) {
        throw std::logic_error(
            "MTP proposal egress read does not state the proposal head's token domain");
    }
    MtpProposalRow row;
    row.paths        = paths;
    row.steps        = steps;
    row.token_domain = token_domain;
    row.ids.resize(static_cast<std::size_t>(paths) * steps);
    row.chain.resize(steps);
    for (std::uint32_t s = 0; s < steps; ++s) {
        for (std::uint32_t i = 0; i < paths; ++i) {
            row.ids[static_cast<std::size_t>(s) * paths + i] = static_cast<TokenId>(
                ids[mtp_tree_proposal_index(s, i, lanes, lane, depth_stride)]);
        }
        row.chain[s] = chain[static_cast<std::size_t>(s) * chain_stride + lane];
    }
    return row;
}

// The PRE-FILL (bridge) variant of the read above. The bridge runs ONE token through the
// proposal head per depth, so `tokens` is 1 and the chain drafts are draft_tokens[0..steps),
// contiguous. Same depth stride, so the two readers decode the same device layout. `token_domain`
// is the same TextConfig::token_domain the bridge's own extraction is handed.
[[nodiscard]] inline MtpProposalRow mtp_proposal_row_from_prefill(const std::int32_t* ids,
                                                                 const TokenId* chain,
                                                                 std::uint32_t paths,
                                                                 std::uint32_t steps,
                                                                 std::size_t depth_stride,
                                                                 std::uint32_t token_domain) {
    if (ids == nullptr || chain == nullptr) {
        throw std::logic_error("MTP bridge proposal lattice is missing");
    }
    if (paths < 2 || steps < 1 || depth_stride == 0) {
        throw std::logic_error("MTP bridge proposal lattice read is outside the round's shape");
    }
    if (token_domain == 0) {
        throw std::logic_error(
            "MTP bridge proposal lattice read does not state the proposal head's token domain");
    }
    MtpProposalRow row;
    row.paths        = paths;
    row.steps        = steps;
    row.token_domain = token_domain;
    row.ids.resize(static_cast<std::size_t>(paths) * steps);
    row.chain.resize(steps);
    for (std::uint32_t s = 0; s < steps; ++s) {
        for (std::uint32_t i = 0; i < paths; ++i) {
            row.ids[static_cast<std::size_t>(s) * paths + i] = static_cast<TokenId>(
                ids[mtp_tree_proposal_index(s, i, 1U, 0U, depth_stride)]);
        }
        row.chain[s] = chain[s];
    }
    return row;
}

} // namespace ninfer::targets::qwen3_6::detail
