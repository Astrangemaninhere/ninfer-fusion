#pragma once

#include "core/layout.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6 {

// Raised 5 -> 15 so the MTP decode frame matches DFlash (width 6 -> 16, the decode
// frame domain bound). Measured on code: k=9 reaches 327.6 tok/s (AL 7.31).
inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 15;
inline constexpr std::uint32_t kMtpDecodeMaximumWidth     = kMtpDecodeMaximumDrafts + 1;
// --draft-tree L,d (L > 1): the most sibling paths one round can publish. The node budget is
// L*d <= 15 with d >= 1 (layouts_impl.h refuses anything else), so L is bounded by the same MTP
// draft domain the window is.
inline constexpr std::uint32_t kMtpTreeMaximumPaths = kMtpDecodeMaximumDrafts;

// The per-DEPTH block stride of MtpDecodeEgress::next_proposal_ids below: every lane's slice of
// one depth lives inside one block, so a depth's writes can never reach another depth's entries.
inline constexpr std::size_t kMtpTreeProposalDepthStride =
    static_cast<std::size_t>(kMtpTreeMaximumPaths) * kMaximumConcurrency;

// The whole lattice one round can carry, in entries.
inline constexpr std::size_t kMtpTreeProposalEntries =
    kMtpTreeProposalDepthStride * kMtpDecodeMaximumDrafts;
inline constexpr std::uint32_t kDFlashDecodeMaximumDrafts = 15;
inline constexpr std::uint32_t kDFlashDecodeMaximumWidth  = kDFlashDecodeMaximumDrafts + 1;

// MTP capture-width LADDER -- the set of widths the adaptive draft window
// (`--spec mtp --draft-tokens 0`, and the backend `auto` resolves to) captures a separate
// decode graph for. One graph per width, selected per round.
//
// WHY A LADDER. The round cost follows the CAPTURED graph width, not the number of live
// columns: over 54 controlled runs (short context, batch 1, k = 1..15)
//     ms/round = 16.71 + 0.737 * W_conf            -> b_width = 0.737 ms per captured column
// while holding W_conf fixed and varying only the live columns gives
//     b_mask = 0.103 ms per live column
// (four paired runs: 0.106/0.094/0.107/0.105). A controller that moves only the LIVE columns
// therefore prices every column it drops at b_width while really saving b_mask, i.e. 7.2x too
// high, and it lost: -27% against the best fixed k on code and on the high-entropy prompt, a
// tie on strong repetition (dl/_criterion_ab.txt, restated in mtp_window_cut.h). Re-capturing
// one graph per width makes the marginal column cost b_width for real, which is the axis the
// survival/cost criterion models.
//
// WHY THIS SET. Brute-forced against a swept survival family S_i = p1^i * r^(i(i-1)/2)
// covering every reachable continuum optimum k* in [1,15] with round(k) = 14.7 + 0.9*k, where
// the regret is 1 - max_{L in W} Phi(L) / Phi(k*) (scratch/aw1/evidence4.cpp):
//     {3,5,9,15}         |W|=4   worst regret 4.25%   mean 0.750%
//     {2,3,5,9,15}       |W|=5   worst 3.04%          mean 0.177%
//     {2,3,5,7,9,15}     |W|=6   worst 2.40%          mean 0.136%   <- this set
//     {2,3,5,7,9,12,15}  |W|=7   worst 2.02%          mean 0.132%
// It contains every MEASURED optimum (code 9, high entropy 5, strong repetition 15, Chinese
// 5), so no measured content class can regress against its own best fixed k, and it ties the
// 7-wide set on the worst case while capturing one fewer executable per batch size. The set is
// fine at the low end because the marginal value of a column is steep there (code: Phi(3) is
// 31.3% below Phi(9) while Phi(7) is only 3.3% below) and coarsens upward where it flattens.
// It floors at 2 and not 1: a one-draft MTP round is strictly worse than an ordinary decode,
// and the engine has no path to drop to ordinary mid-request.
inline constexpr std::uint32_t kMtpWindowLadder[] = {2, 3, 5, 7, 9, 15};
inline constexpr std::size_t   kMtpWindowLadderSize =
    sizeof(kMtpWindowLadder) / sizeof(kMtpWindowLadder[0]);
inline constexpr std::uint32_t kMtpWindowLadderTop = kMtpWindowLadder[kMtpWindowLadderSize - 1];

// Index of the ladder width a criterion cut is snapped to. Ties go UP: at a tie the marginal
// value of the column equals its cost, so either side is a wash in the cost model, and the
// measured failure of a shrink-only actuator on this axis was under-drafting (-27%).
[[nodiscard]] inline constexpr std::size_t mtp_window_ladder_index(std::uint32_t cut) noexcept {
    std::size_t   best     = 0;
    std::uint32_t best_gap = kMtpWindowLadder[0] > cut ? kMtpWindowLadder[0] - cut
                                                       : cut - kMtpWindowLadder[0];
    for (std::size_t i = 1; i < kMtpWindowLadderSize; ++i) {
        const std::uint32_t width = kMtpWindowLadder[i];
        const std::uint32_t gap   = width > cut ? width - cut : cut - width;
        if (gap <= best_gap) { best_gap = gap; best = i; }
    }
    return best;
}

[[nodiscard]] inline constexpr std::uint32_t mtp_window_ladder_width(std::uint32_t cut) noexcept {
    return kMtpWindowLadder[mtp_window_ladder_index(cut)];
}


struct RoundStateSpec {
    std::int32_t hidden          = 0;
    std::int32_t output_rows     = 0;
    std::uint32_t batch_capacity = 1;
    std::uint32_t draft_window   = 0;
    // --draft-tree L,d (MTP): {0,0} = no tree, else the verify frame carries one column per
    // tree node, so draft_window above is the node budget L*d. It changes NO tensor shape --
    // the ingress/egress arrays are already sized at kMtpDecodeMaximumWidth -- it only tells the
    // runtime whether this round's columns are tree nodes (L > 1) or a chain (L <= 1).
    std::uint32_t draft_tree_paths = 0;
    std::uint32_t draft_tree_depth = 0;
    bool enable_mtp              = false;
    bool enable_dflash           = false;
};

// Stable pinned/device transfer format for ordinary decode. The full fixed-size object is copied
// once per round; only its exact-B prefixes are consumed by the model schedule.
struct OrdinaryDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> tokens{};
    std::array<std::int32_t, kMaximumConcurrency> cache_positions{};
    std::array<std::int32_t, kMaximumConcurrency> rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct OrdinaryDecodeEgress {
    std::array<TokenId, kMaximumConcurrency> sampled_tokens{};
};

// Stable pinned/device transfer formats for concurrent MTP decode. The arrays use the maximum
// product domain; RoundState binds only the configured [K,C] and [K+1,C] prefixes.
struct MtpDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> base_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> remaining_budgets{};
    std::array<std::int32_t, kMaximumConcurrency> current_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> current_drafts{};
    std::array<std::int32_t, kMaximumConcurrency * kMtpDecodeMaximumWidth> target_rope_positions{};
    // --draft-tree L,d (L > 1) ONLY: the same positions in the CHAIN spelling, i.e.
    // frontier + min(j, extent) + rope_delta, indexed by CHAIN DEPTH. A tree's verify needs the
    // per-COLUMN depth spelling above (siblings are alternatives for one position); the MTP draft
    // head consumes the ACCEPTED CHAIN and mtp_impl.h drives it with ids/hidden that
    // mtp_draft_align_hidden has already re-indexed to depth order, so at index j it needs
    // frontier + j -- which the column table only happens to carry when the column order IS the
    // depth order (a chain). A chain round never reads this array, which is what keeps
    // --draft-tokens bit-identical.
    std::array<std::int32_t, kMaximumConcurrency * kMtpDecodeMaximumWidth>
        target_chain_rope_positions{};
    // One 64-bit ancestor bit set per verify column: bit i of column j means "column j may attend
    // column i of this round's own block", bit 0 being the anchor column (always set). A chain's
    // column j is the prefix (1 << (min(j, extent) + 1)) - 1, which the per-row prefix
    // (target_valid_columns) plus the position-causal cut already implement, so this array is a
    // no-op for --draft-tokens -- and the runtime does not even pass it unless the round is a tree
    // round (draft_tree_paths > 1). One word per column: the 64-node limit of
    // ddtree::column_ancestors (include/ninfer/ops/dflash2_ddtree.h:333-345) therefore caps the
    // WIDTH, and kMtpDecodeMaximumWidth = 16 is far inside it.
    std::array<std::uint64_t, kMaximumConcurrency * kMtpDecodeMaximumWidth> target_column_masks{};
    // Per verify column, the DEPTH of the tree node it carries (the anchor column is depth 0).
    // The RoPE position of column j is anchor + target_column_depths[j], not its column index: two
    // siblings at the same depth are alternatives for the SAME position, which is what makes a
    // tree's positions non-sequential (ddtree::column_depth, include/ninfer/ops/dflash2_ddtree.h:
    // 314-322). A chain's column j is min(j, extent), so with no published tree this is
    // bit-identical to the `frontier + min(j, extent)` it replaces.
    std::array<std::int32_t, kMaximumConcurrency * kMtpDecodeMaximumWidth> target_column_depths{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> mtp_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<std::int32_t, kMaximumConcurrency> rope_deltas{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct MtpDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
    // Step-major: all B rows for proposal step 0, followed by all B rows for step 1, etc.
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> next_drafts{};
    std::array<std::int32_t, kMaximumConcurrency> next_extents{};
    // --draft-tree L,d (L > 1) ONLY: the per-depth top-L proposal row ids the draft head produced
    // this round, i.e. the candidate lattice of the SIBLING tree shape
    // (src/targets/qwen3_6/impl/runtime/mtp_tree_proposal_fill.h). Entry (depth s, rank i, lane t)
    // is at s * kMtpTreeProposalDepthStride + i * <the round's lane count> + t, because
    // ops::mtp_proposal_topk stores its per-token rows with stride `tokens` == the round's batch.
    // Rank 0 is the SPINE: it must equal next_drafts[s * kMaximumConcurrency + t], which is the
    // same logits row's argmax; a producer that finds them different refuses the round instead of
    // publishing a lattice no device output backs. A round with draft_tree_paths <= 1 never writes
    // this array and never reads it. Only the (s < tree depth, i < paths, t < batch) sub-block is
    // ever defined -- ops::mtp_proposal_topk writes exactly min(top_l, rows) rows per token, so the
    // block's tail and every depth beyond the tree's own depth keep whatever they held before.
    std::array<std::int32_t, kMtpTreeMaximumPaths * kMtpDecodeMaximumDrafts * kMaximumConcurrency>
        next_proposal_ids{};
    // --draft-tree L,d with L > 1 (a TREE round): the verify COLUMN the round accepted at (the
    // deepest node whose ancestor chain matched the verifier's own argmax), 0 for a chain round and
    // for a tree round that accepted nothing. This is the column the continuation hidden is
    // selected from and the column the KV history commit reads.
    std::array<std::int32_t, kMaximumConcurrency> accepted_columns{};
    // The accepted chain -> history map the commit used: chain_sources[i, b] is the verify COLUMN
    // holding the accepted token at depth i + 1, -1 for an unused slot. A chain round is the
    // identity (chain_sources[i, b] == i + 1) and moves no bytes; a tree round is the row's chain.
    std::array<std::int32_t, kMaximumConcurrency * kMtpDecodeMaximumWidth> chain_sources{};
    // Non-zero = the published tree metadata of that row was inconsistent and the round must be
    // refused; see ops::kMtpTreeFlagColumnOutOfRange and friends. The runtime throws on it instead
    // of committing a KV history that came from a guess.
    std::array<std::int32_t, kMaximumConcurrency> tree_commit_flags{};
};

// Stable pinned/device transfer formats for one exact-B DFlash transaction. The proposal is
// produced and verified in the same round, so no draft state crosses the round boundary.
struct DFlashDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> execution_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> context_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> dflash_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> active_lanes{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct DFlashDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kDFlashDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_extents{};
};

struct OrdinaryDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion logits;
    TensorRegion hidden;
};

struct MtpPrefillStateLayout {
    TensorRegion position;
    TensorRegion ar_hidden;
    TensorRegion draft_tokens;
    TensorRegion target_input_ids;
    TensorRegion target_positions;
    // --draft-tree L,d (L > 1): the bridge's own proposal lattice, in the same layout as
    // MtpDecodeEgress::next_proposal_ids. Declared LAST so every offset above it is unchanged.
    TensorRegion lattice_ids;
};

struct DFlashPrefillStateLayout {
    TensorRegion produced_count;
};

struct MtpDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion verify_ids;
    TensorRegion target_positions;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
    TensorRegion proposal_logits;
    TensorRegion alignment_ids;
    TensorRegion alignment_hidden;
    TensorRegion ar_hidden;
    TensorRegion next_hidden;
    TensorRegion ar_positions;
    TensorRegion ar_rope_positions;
    TensorRegion ar_valid_columns;
};

struct DFlashDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion proposal_ids;
    TensorRegion proposal_positions;
    TensorRegion append_positions;
    TensorRegion append_counts;
    TensorRegion draft_tokens;
    TensorRegion verify_ids;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
    // DFlash2 candidate walk outputs: per proposal step the top-K candidate
    // token ids and their softmax scores. Only consumed by the sampling
    // accept path (distribution-correct rejection); greedy never reads them.
    TensorRegion draft_candidate_ids;
    TensorRegion draft_candidate_probs;
};

struct RoundStateLayout {
    RoundStateSpec spec;
    std::optional<OrdinaryDecodeStateLayout> ordinary;
    TensorRegion token;
    TensorRegion pos;
    TensorRegion rope_pos;
    TensorRegion rope_delta;
    TensorRegion logits;
    TensorRegion text_kv_table_row;
    TensorRegion backend_kv_table_row;
    std::optional<MtpPrefillStateLayout> mtp;
    std::optional<DFlashPrefillStateLayout> dflash_prefill;
    std::optional<MtpDecodeStateLayout> mtp_decode;
    std::optional<DFlashDecodeStateLayout> dflash_decode;
    bool complete = false;
};

struct OrdinaryDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor tokens;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor text_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor sampled_tokens;
    Tensor logits;
    Tensor hidden;

    OrdinaryDecodeState() = default;
    OrdinaryDecodeState(DeviceSpan backing, const OrdinaryDecodeStateLayout& layout,
                        std::uint32_t batch_capacity);
};

// The two planning calls expose one deliberate exact-target extension seam after scalar logits.
// This lets a target retain its schedule-sized prefill activation at the established physical
// address without making that activation part of the family round contract.
[[nodiscard]] RoundStateLayout begin_round_state_layout(LayoutBuilder& builder,
                                                        const RoundStateSpec& spec);
void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout);

struct MtpPrefillState {
    Tensor position;
    Tensor ar_hidden;
    Tensor draft_tokens;
    Tensor target_input_ids;
    Tensor target_positions;
    Tensor lattice_ids;

    MtpPrefillState() = default;
    MtpPrefillState(DeviceSpan backing, const MtpPrefillStateLayout& layout);
};

struct DFlashPrefillState {
    Tensor produced_count;

    DFlashPrefillState() = default;
    DFlashPrefillState(DeviceSpan backing, const DFlashPrefillStateLayout& layout);
};

struct MtpDecodeState {
    // --draft-tree L,d (see RoundStateSpec). Set by RoundState from the spec: L <= 1 is the
    // degenerate chain (every tree path is then a no-op) and L > 1 means the round's verify
    // columns are tree nodes, which is what makes the mask tensor reach the attention op.
    std::uint32_t draft_tree_paths = 0;
    std::uint32_t draft_tree_depth = 0;
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor base_frontiers;
    Tensor remaining_budgets;
    Tensor current_extents;
    Tensor target_valid_columns;
    Tensor target_column_masks;
    Tensor target_column_depths;
    Tensor current_drafts;
    Tensor target_rope_positions;
    // The same table in the chain spelling; read only by a tree round's MTP draft head.
    Tensor target_chain_rope_positions;
    Tensor text_kv_table_rows;
    Tensor mtp_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    Tensor rope_deltas;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor next_drafts;
    Tensor next_extents;
    Tensor next_proposal_ids;
    Tensor accepted_columns;
    Tensor chain_sources;
    Tensor tree_commit_flags;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;
    Tensor proposal_logits;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor next_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;

    MtpDecodeState() = default;
    MtpDecodeState(DeviceSpan backing, const MtpDecodeStateLayout& layout,
                   std::uint32_t batch_capacity, std::uint32_t draft_window);
};

struct DFlashDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor execution_frontiers;
    Tensor context_frontiers;
    Tensor proposal_extents;
    Tensor target_valid_columns;
    Tensor text_kv_table_rows;
    Tensor dflash_kv_table_rows;
    Tensor active_lanes;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor egress_proposal_extents;
    Tensor proposal_ids;
    Tensor proposal_positions;
    Tensor append_positions;
    Tensor append_counts;
    Tensor draft_tokens;
    Tensor verify_ids;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;
    Tensor draft_candidate_ids;
    Tensor draft_candidate_probs;

    DFlashDecodeState() = default;
    DFlashDecodeState(DeviceSpan backing, const DFlashDecodeStateLayout& layout,
                      std::uint32_t batch_capacity, std::uint32_t draft_window);
};

struct RoundState {
    std::optional<OrdinaryDecodeState> ordinary;
    Tensor token;
    Tensor pos;
    Tensor rope_pos;
    Tensor rope_delta;
    Tensor logits;
    Tensor text_kv_table_row;
    Tensor backend_kv_table_row;
    std::optional<MtpPrefillState> mtp;
    std::optional<DFlashPrefillState> dflash_prefill;
    std::optional<MtpDecodeState> mtp_decode;
    std::optional<DFlashDecodeState> dflash_decode;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

} // namespace ninfer::targets::qwen3_6
