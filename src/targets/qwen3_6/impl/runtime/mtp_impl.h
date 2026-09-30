#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "core/nvtx.h"
#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/mtp_proposal_topk.h"
#include <ninfer/targets/qwen3_6/round_state.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent,
                        state.mtp_tree_paths, state.mtp_tree_depth);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    // --draft-tree L,d (L > 1): the FIRST verify round's tree is seeded from THIS bridge -- the
    // hand-off copies io.mtp->draft_tokens out and publishes from it -- so the same per-depth top-L
    // extraction the decode loop runs has to run here too, into MtpPrefillState::lattice_ids. The
    // bridge pushes ONE token per depth, so `tokens == 1` and a depth's entries are contiguous:
    // that is the one-lane form of detail::mtp_tree_proposal_index. A chain bridge
    // (draft_tree_paths <= 1) launches no extra work at all. `rows` is TextConfig::token_domain and
    // not the tensor's row count, because that is exactly the window the ordinary argmax path reads
    // (text_context_impl.h proposal_argmax); the head's linear leaves every row above it untouched.
    const std::uint32_t tree_steps = state.mtp_tree_paths > 1 ? state.mtp_tree_depth : 0;
    std::int32_t* const proposal_lattice_ids =
        tree_steps == 0
            ? nullptr
            : static_cast<std::int32_t*>(state.execution.io.mtp->lattice_ids.data);
    const auto extract_proposal_lattice = [&](std::uint32_t depth) {
        if (proposal_lattice_ids == nullptr || depth >= tree_steps || !build_proposal) { return; }
        ops::mtp_proposal_topk(logits.data, TextConfig::token_domain, 1,
                               static_cast<std::int32_t>(state.mtp_tree_paths),
                               proposal_lattice_ids +
                                   static_cast<std::size_t>(depth) * kMtpTreeProposalDepthStride,
                               state.execution.device.stream);
    };
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    // FIX-A: the third field is the pinned split reference, and leaving it out is what made the
    // MTP draft's bridge reduce its keys on a 32-key grid derived from the LIVE window
    // (`[splitdbg] ... pin=0 split_reference=47 ... split_units=32`) while the batch-1 decode
    // and the target verify of the same row used the capacity-pinned grid. The pin is the
    // compile-time maximum visible-key count: a constant of the produced graph, which is what
    // the contract asks for (a partition that cannot follow how far the sequence has advanced).
    // The sequence planner's own capacity is not in scope here; any constant at or above every
    // live window gives the row the same single-split association, which is the property the
    // contract is about, and this one is by definition at or above all of them
    // (validate_envelope refuses a max_visible_keys above it).
    const ops::GqaExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    extract_proposal_lattice(0);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        // FIX-A: same pin as the bridge above, for the same reason: this is the AR step the
        // draft runs once per proposal token, so there are k of them per round and each one
        // used to partition its keys from the live window.
        const ops::GqaExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, next_hidden, logits, next_draft);
        extract_proposal_lattice(static_cast<std::uint32_t>(i));
        CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.execution.io.mtp->ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, state.execution.device.stream));
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpCausalAttentionEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_6::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        // --draft-tree L,d with L > 1: this round's verify columns are tree NODES, so the
        // per-column ancestor masks must reach the attention op. L <= 1 -- which includes every
        // --draft-tokens run -- keeps the pre-tree call topology exactly: no mask tensor, no new
        // kernel input, byte-identical on every KV tier and on every route.
        // 落点 ③: ONE source of truth. Whether this round is a tree round is decided by
        // the SAME member the ingress fill and the gate keyed on
        // (ProgramImplCore::draft_tree_paths), carried here by value at hand-off.
        const bool tree_round = state.tree_paths > 1;
        // TREE-FIX-LAND 打印点 B1（HOST ONLY，只读；整块删除即回退）：帧那份，逐轮。
        std::fprintf(stderr,
                     "[treeframe] batch=%d k=%u frame_dtp=%u frame_dtd=%u tree_round=%d\n",
                     static_cast<int>(batch_size), k, frame.draft_tree_paths,
                     frame.draft_tree_depth, static_cast<int>(tree_round));
        // 落点 ③: same source of truth -- the throw is KEPT; only its operand moves to the
        // copy that decides tree_round, or a tree round whose frame copy is stale would be
        // refused by the fix itself.
        if (tree_round && state.tree_depth == 0) {
            throw std::logic_error("MTP tree verify state is incomplete (depth 0)");
        }
        // --draft-tree L,d (L > 1): the ROW the tree's nodes are drawn from. ops::mtp_proposal_topk
        // returns ROW INDICES of the table it is handed, which are global token ids only under the
        // FULL proposal head -- the shortlist head never writes this frame at all, which is why the
        // planner refuses a tree round on it (detail::mtp_tree_proposal_head_defect, called from
        // layouts_impl.h). This is the ONLY device work the tree adds, and it is behind
        // `tree_round`: an L <= 1 round captures and replays exactly the graph it captured before
        // this change.
        // 落点 ③: the extraction depth and the extraction width both come from the same
        // by-value copy; a stale frame copy here is an empty lattice, i.e. a tree that is
        // never built while the round still claims to verify one.
        const std::uint32_t tree_steps = tree_round ? state.tree_depth : 0;
        std::int32_t* const proposal_lattice_ids =
            tree_round ? static_cast<std::int32_t*>(frame.next_proposal_ids.data) : nullptr;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache);
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor budgets            = frame.remaining_budgets.slice(0, 0, batch_size);
        Tensor current_extents    = frame.current_extents.slice(0, 0, batch_size);
        Tensor target_valid       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor target_masks       = frame.target_column_masks.slice(1, 0, batch_size);
        Tensor current_drafts     = frame.current_drafts.slice(1, 0, batch_size);
        Tensor target_rope        = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor mtp_rows           = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor rope_deltas        = frame.rope_deltas.slice(0, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);
        Tensor next_extents       = frame.next_extents.slice(0, 0, batch_size);
        Tensor alignment_ids      = frame.alignment_ids.slice(1, 0, batch_size);
        Tensor alignment_hidden   = frame.alignment_hidden.slice(2, 0, batch_size);
        Tensor ar_hidden          = frame.ar_hidden.slice(1, 0, batch_size);
        Tensor next_hidden        = frame.next_hidden.slice(1, 0, batch_size);
        Tensor ar_positions       = frame.ar_positions.slice(0, 0, batch_size);
        Tensor ar_rope_positions  = frame.ar_rope_positions.slice(0, 0, batch_size);
        Tensor ar_valid_columns   = frame.ar_valid_columns.slice(0, 0, batch_size);
        Tensor next_drafts        = frame.next_drafts.slice(0, 0, batch_size);

        ops::speculative_prepare_verify_inputs(anchors, current_drafts, frontiers, current_extents,
                                               verify_ids, target_positions,
                                               state.execution.device.stream);
        // TREE-FIX-LAND 打印点 B2（HOST ONLY，只读；整块删除即回退）：真实绑定见证。
        // B1 原来那两列是恒等式（tree_round && frame_dtp > 1 ≡ tree_round，信息量为 0），
        // 所以这里读【真正交给 verify op 的东西】：mtp_impl.h:139 由
        // frame.target_column_masks.slice(1,0,batch_size) 得到的 target_masks 的 data 指针
        // （mtp_impl.h:178 用它绑 .column_masks），以及 depths 张量的 data 指针。
        // 两者皆 nullptr ⇒ 绑的是 Tensor{} ⇒ 内核走链臂。
        {
            const Tensor target_depths_probe =
                frame.target_column_depths.slice(1, 0, batch_size);
            std::fprintf(stderr,
                         "[treebind] batch=%d k=%u tree_round=%d masks_data=%p depths_data=%p\n",
                         static_cast<int>(batch_size), k, static_cast<int>(tree_round),
                         static_cast<const void*>(target_masks.data),
                         static_cast<const void*>(target_depths_probe.data));
        }
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp,
                                           static_cast<std::uint64_t>(width) * batch_size);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 TargetVerifyFrameView{
                                     .ids                     = verify_ids,
                                     .cache_positions         = target_positions,
                                     .rope_positions          = target_rope,
                                     .valid_columns           = target_valid,
                                     .column_masks            = tree_round ? target_masks : Tensor{},
                                     .column_depths =
                                         tree_round ? frame.target_column_depths.slice(
                                                          1, 0, batch_size)
                                                    : Tensor{},
                                     .kv_table_rows           = text_rows,
                                     .state_source_slots      = state_sources,
                                     .state_destination_slots = state_destinations,
                                     .target_hidden           = target_hidden,
                                     .target_logits           = target_logits,
                                     .target_tokens           = target_tokens,
                                     .drafts                  = current_drafts,
                                     .current_extents         = current_extents,
                                     .frontiers               = frontiers,
                                     .anchors                 = anchors,
                                     .licensed_tokens         = licensed_tokens,
                                     .licensed_counts         = licensed_counts,
                                     .accepted_drafts         = accepted,
                                     .accepted_columns        = frame.accepted_columns.slice(
                                         0, 0, batch_size),
                                     .chain_sources           = frame.chain_sources.slice(
                                         1, 0, batch_size),
                                     .tree_commit_flags       = frame.tree_commit_flags.slice(
                                         0, 0, batch_size),
                                     .selected_hidden         = selected_hidden,
                                     .replay_records          = state.execution.replay_records,
                                     .sampling                = frame.sampling,
                                 },
                                 envelopes.target_verify);
        }

        if (tree_round) {
            // The accepted chain occupies arbitrary columns, while the next round reads its history
            // as the position prefix frontier+1..frontier+A and every layer owns its own K/V
            // planes: move each accepted node's rows to the position its depth gives it, in every
            // text layer, before the next round's attention can read them. A chain round
            // (L <= 1) does not enter here at all, which keeps --draft-tokens byte-identical.
            // The slices are named because the Op takes its two outputs by reference.
            Tensor commit_depths  = frame.target_column_depths.slice(1, 0, batch_size);
            Tensor accepted_cols  = frame.accepted_columns.slice(0, 0, batch_size);
            Tensor commit_sources = frame.chain_sources.slice(1, 0, batch_size);
            Tensor commit_flags   = frame.tree_commit_flags.slice(0, 0, batch_size);
            const std::uint32_t text_layers = state.text_cache.layers();
            for (std::uint32_t layer = 0; layer < text_layers; ++layer) {
                ops::mtp_tree_commit_history(target_masks, commit_depths, accepted_cols, frontiers,
                                             text_rows, state.text_cache.batch_layer_view(layer),
                                             commit_sources, commit_flags,
                                             state.execution.device.stream);
            }
        }

        {
            nvtx::ScopedRange draft_range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp,
                                          static_cast<std::uint64_t>(k) * batch_size);
            // SVIP entropy cap: stop drafting before the first verify column whose
            // base-logit softmax entropy exceeds the threshold. Off by default;
            // set NINFER_SVIP_THRESHOLD to enable (value is threshold^2, e.g. 6.25
            // for the DFlashConfig default 2.5).
            static const float kSvipThreshold = [] {
                const char* env = std::getenv("NINFER_SVIP_THRESHOLD");
                return env != nullptr ? static_cast<float>(std::atof(env)) : 0.0F;
            }();
            // Adaptive draft window (auto mode): NINFER_ADAPTIVE_WINDOW=<k_max> enables it.
            // Stateless window control from the round's own accept record; see
            // ops::mtp_adaptive_extents. Takes precedence over the entropy cap.
            static const std::int32_t kAdaptiveWindowMax = [] {
                const char* env = std::getenv("NINFER_ADAPTIVE_WINDOW");
                return env != nullptr ? static_cast<std::int32_t>(std::atoi(env)) : 0;
            }();
            const Tensor* svip_cuts_arg = nullptr;
            Tensor svip_cuts_storage;
            if (kAdaptiveWindowMax > 0) {
                auto scope = state.execution.work.scope();
                svip_cuts_storage = state.execution.work.alloc(DType::I32, {batch_size});
                ops::mtp_adaptive_extents(accepted, current_extents, svip_cuts_storage,
                                          kAdaptiveWindowMax, state.execution.device.stream);
                svip_cuts_arg = &svip_cuts_storage;
            } else if (kSvipThreshold > 0.0F) {
                auto scope = state.execution.work.scope();
                svip_cuts_storage = state.execution.work.alloc(DType::I32, {batch_size});
                ops::mtp_svip_entropy_extents(target_logits, accepted, svip_cuts_storage,
                                              kSvipThreshold, state.execution.device.stream);
                svip_cuts_arg = &svip_cuts_storage;
            }
            ops::mtp_prepare_next_round(verify_ids, anchors, accepted, frontiers, budgets,
                                        licensed_counts, rope_deltas, alignment_ids, next_extents,
                                        ar_positions, ar_rope_positions, ar_valid_columns,
                                        static_cast<std::int32_t>(state.text_cache.max_context()),
                                        state.execution.device.stream, svip_cuts_arg);
            // --draft-tree L,d with L > 1: the MTP head must be driven by the ACCEPTED CHAIN in
            // DEPTH order, not by the verify columns. `alignment_ids[j] = verify_ids[j+1]` pairs
            // index j with `target_hidden[:,j]`, and index j is a chain depth only while the
            // accepted nodes are the column prefix -- a tree's are not. Two substitutions, both
            // inside the tree gate:
            //   ids    : licensed_tokens[j] is the accepted draft at depth j+1 by contract
            //            (speculative_accept_greedy_drafts), so it IS the depth-ordered id list,
            //            and for a chain licensed_tokens[j] == verify_ids[j+1] bit for bit.
            //   hidden : mtp_draft_align_hidden moves column j's hidden to index j (identity for a
            //            chain, chain_sources[j-1] for a tree).
            // The positions need no re-basing: positions[j] = frontier + min(j, extent)
            // (speculative_prepare_verify_inputs) is already `frontier + depth` for every live j,
            // which is exactly where ops::mtp_tree_commit_history put the accepted chain's K/V.
            // The tree's alignment buffer is consumed by the batch forward two statements down, so
            // its arena scope has to OUTLIVE that call: DeviceArena::Scope rolls the arena watermark
            // back on destruction (src/core/arena.cu), and a scope closed before the forward handed
            // the MTP draft head's context hidden back to the allocator -- which then placed the
            // forward's own scratch at exactly that address and overwrote it. The draft head read a
            // buffer its consumer had already clobbered, so every tree round from the second on
            // drafted from a corrupt context and could only ever accept a coincidence. A chain round
            // takes the same two statements WITHOUT opening any scope, so its allocation traffic and
            // every arena offset it sees are unchanged.
            Tensor mtp_context_ids    = alignment_ids;
            Tensor mtp_context_hidden = target_hidden;
            Tensor mtp_context_rope   = target_rope;
            const auto forward_with_mtp_context = [&] {
                card.mtp_forward_decode_batch(mtp_context_ids, mtp_context_hidden, target_positions,
                                              mtp_context_rope, licensed_counts, mtp_rows,
                                              envelopes.batch, alignment_hidden);
                ops::speculative_select_accepted_hidden(alignment_hidden, accepted, ar_hidden,
                                                        state.execution.device.stream);
            };
            if (tree_round) {
                auto context_scope = state.execution.work.scope();
                mtp_context_hidden = state.execution.work.alloc(
                    DType::BF16, {TextConfig::hidden, width, batch_size});
                ops::mtp_draft_align_hidden(target_hidden,
                                            frame.chain_sources.slice(1, 0, batch_size),
                                            licensed_counts, mtp_context_hidden,
                                            state.execution.device.stream);
                mtp_context_ids = licensed_tokens;
                // Depth-indexed RoPE: index j is the accepted node at DEPTH j (the ids above and the
                // hidden just gathered are both in depth order), whose canonical position is
                // frontier + j. target_rope is indexed by verify COLUMN and carries
                // frontier + depth(column j) -- correct for the verify's own rotation, wrong here
                // for every j >= 2 of an L > 1 layout (depth-major: depth(column j) =
                // floor((j-1)/L) + 1). target_positions needs no such substitution: a tree round
                // forces extent == node count, so speculative_prepare_verify_inputs already wrote
                // frontier + j there.
                mtp_context_rope = frame.target_chain_rope_positions.slice(1, 0, batch_size);
                forward_with_mtp_context();
            } else {
                forward_with_mtp_context();
            }

            Tensor proposal_logits = frame.proposal_logits.slice(1, 0, batch_size);
            Tensor draft0          = next_drafts.slice(1, 0, 1).view({batch_size});
            if (tree_round && (proposal_logits.ne[0] < TextConfig::token_domain ||
                               proposal_logits.ne[1] != batch_size)) {
                throw std::logic_error(
                    "MTP tree proposal frame is not the proposal-logits window");
            }
            // One top-L extraction per tree depth, landing directly in the round's egress at that
            // depth's own block. The op writes min(top_l, rows) rows per token with stride
            // `tokens == batch_size`, which is exactly the (depth, rank, lane) layout
            // detail::mtp_tree_proposal_index decodes on the host; the block's tail and every depth
            // past the tree's own depth are never written and never read.
            const auto extract_proposal_lattice = [&](std::uint32_t depth) {
                if (proposal_lattice_ids == nullptr || depth >= tree_steps) { return; }
                ops::mtp_proposal_topk(
                    proposal_logits.data, TextConfig::token_domain, batch_size,
                    static_cast<std::int32_t>(state.tree_paths),
                    proposal_lattice_ids +
                        static_cast<std::size_t>(depth) * kMtpTreeProposalDepthStride,
                    state.execution.device.stream);
            };
            card.mtp_propose_batch(ar_hidden, proposal_logits, draft0);
            extract_proposal_lattice(0);
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                Tensor previous =
                    next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
                Tensor next =
                    next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1).view({batch_size});
                Tensor position =
                    ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1, batch_size});
                Tensor rope = ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                  .view({1, batch_size});
                Tensor valid = ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                                   .view({batch_size});
                Tensor previous_batch    = previous.view({1, batch_size});
                Tensor hidden_batch      = ar_hidden.view({TextConfig::hidden, 1, batch_size});
                Tensor next_hidden_batch = next_hidden.view({TextConfig::hidden, 1, batch_size});
                card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                              mtp_rows, envelopes.ar[step], next_hidden_batch);
                card.mtp_propose_batch(next_hidden, proposal_logits, next);
                extract_proposal_lattice(step + 1);
                CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           state.execution.device.stream));
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpCausalAttentionEnvelopes envelopes,
                              DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
    // GRAPH ROUND: `run_prepared` (graph_impl.h:11-21) ran this body at CAPTURE time and only
    // launched the graph here, so the round's acceptlog block is published from the photograph of
    // this round's width (k + 1 columns) -- host side, after the launch, outside capture. Inert
    // unless NINFER_ACCEPTLOG is set.
    if (executable != nullptr) { acceptlog_replay_dump(static_cast<int>(k) + 1); }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
