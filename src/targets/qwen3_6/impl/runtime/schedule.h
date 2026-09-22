#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/swa.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

using qwen3_6::PreparedPromptData;
using qwen3_6::PromptModality;

struct ExecutionCore {
    DeviceContext& device;
    const LoadedModelData& model;
    WorkspaceArena& work;
    LinearAttentionStatePool& linear_attention;
    const GdnReplayRecords* replay_records;
    qwen3_6::RoundState& io;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk;
    ProposalHead proposal_head;
};

struct PrefillContext {
    ExecutionCore execution;
    qwen3_6::PagedKVCacheView text_kv;
    qwen3_6::PagedKVCacheView mtp_kv;
    const qwen3_6::PagedKVCache& text_cache;
    const qwen3_6::PagedKVCache* mtp_cache;
    DFlashPersistentState* dflash;
    DFlash2PersistentState* dflash2;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    Tensor* rewrite_checkpoint_hidden;
    std::int32_t state_source_slot                          = 0;
    std::int32_t state_destination_slot                     = 0;
    std::uint32_t mtp_proposal_extent                       = 0;
    const qwen3_6::DFlashDecodeIngress* dflash_host_ingress  = nullptr;
    const qwen3_6::DFlashDecodeIngress* dflash2_host_ingress = nullptr;
    // --draft-tree L,d (L > 1): the tree shape THIS prefill's MTP bridge must seed. 0 / 0 for a
    // chain bridge, which then launches no extraction work. The same shape as
    // ProgramImplCore::draft_tree_paths / draft_tree_depth (program.h), carried here because
    // mtp_bridge_and_propose() is the only place the FIRST round's lattice can be produced.
    std::uint32_t mtp_tree_paths                           = 0;
    std::uint32_t mtp_tree_depth                           = 0;
};

struct OrdinaryBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    qwen3_6::OrdinaryDecodeState& frame;
    const qwen3_6::OrdinaryDecodeIngress& host_ingress;
    qwen3_6::OrdinaryDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
};

struct MtpBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    const qwen3_6::PagedKVCache& mtp_cache;
    qwen3_6::MtpDecodeState& frame;
    const qwen3_6::MtpDecodeIngress& host_ingress;
    qwen3_6::MtpDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    // 落点 ③ (TREE-FIX-LAND): the tree shape THIS round's ingress fill and gate used,
    // carried BY VALUE at hand-off. Until now the verify side decided the topology from
    // MtpDecodeState::draft_tree_paths -- a SECOND copy of the same configuration, written
    // on the frame at round_state.cpp:409 -- and the two were observed to disagree (the
    // publisher published a tree while the verifier verified a chain). With one source of
    // truth that combination is structurally unrepresentable.
    std::uint32_t tree_paths = 0;
    std::uint32_t tree_depth = 0;
};

struct DFlashBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    DFlashPersistentState& dflash;
    qwen3_6::DFlashDecodeState& frame;
    const qwen3_6::DFlashDecodeIngress& host_ingress;
    qwen3_6::DFlashDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    // DSpark self-verification length policy (SVIP): drafting stops at the
    // first position whose base-logit entropy sqrt(H) exceeds this threshold.
    // A non-positive value disables the entropy cap (full draft window).
    float svip_entropy_threshold = 2.5F;
};

struct DFlashAppendContext {
    ExecutionCore execution;
    DFlashPersistentState& dflash;
};

struct DFlash2BatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    DFlash2PersistentState& dflash2;
    qwen3_6::DFlashDecodeState& frame;
    const qwen3_6::DFlashDecodeIngress& host_ingress;
    qwen3_6::DFlashDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
};

struct DFlash2AppendContext {
    ExecutionCore execution;
    DFlash2PersistentState& dflash2;
};

struct MtpCausalAttentionEnvelopes {
    ops::GqaExecutionEnvelope target_verify;
    ops::GqaExecutionEnvelope batch;
    std::array<ops::GqaExecutionEnvelope, kMaximumMtpDraftTokens - 1> ar;
};

struct DFlashEnvelopes {
    // DSpark (BF16 masked-block draft): local layers use the swa kernel and
    // full-context layers use the bidirectional GQA kernel.
    ops::SwaContextExecutionEnvelope local;
    ops::GqaContextExecutionEnvelope full;
    ops::KVCacheAppendPrefixExecutionEnvelope append;
};

// DFlash2 is a pure sliding-window draft: its local attention uses the swa
// kernel (symmetric window, no full-context stage), so only the local and
// append envelopes exist.
struct DFlash2Envelopes {
    ops::SwaContextExecutionEnvelope local;
    ops::KVCacheAppendPrefixExecutionEnvelope append;
};

struct TargetVerifyFrameView {
    Tensor ids;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor valid_columns;
    // Per-column ancestor bit masks, I64 [W,B], or an empty Tensor for a chain round. See
    // MtpDecodeIngress::target_column_masks; only the MTP tree verify (L > 1) fills it.
    Tensor column_masks;
    // Per-column DEPTH of the node each verify column carries, I32 [W,B] (see
    // MtpDecodeIngress::target_column_depths). It is the cross-check the KV history commit runs
    // against the mask's own popcount: if the published depth and the mask's chain length disagree,
    // the round's tree is inconsistent and the commit refuses it.
    // Every tree field defaults to the EMPTY Tensor (= "this round is a chain", the struct's
    // existing spelling for replay_records/sampling), which is what lets the dflash construction
    // sites leave them out without a -Wmissing-field-initializers.
    Tensor column_depths{};
    Tensor kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    Tensor target_hidden;
    Tensor target_logits;
    Tensor target_tokens;
    Tensor drafts;
    Tensor current_extents;
    Tensor frontiers;
    Tensor anchors;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    // Tree outputs of the accept (I32 [B] and I32 [W,B]): the accepted COLUMN and the accepted
    // chain -> history map, both consumed by ops::mtp_tree_commit_history and published to the
    // host. For a chain round accepted_columns[b] == accepted_drafts[b] and chain_sources is the
    // identity, which is what keeps --draft-tokens bit-identical. All three default to the empty
    // Tensor, so a construction site that predates the tree keeps compiling warning-free.
    Tensor accepted_columns{};
    Tensor chain_sources{};
    Tensor tree_commit_flags{};
    Tensor selected_hidden;
    Tensor draft_candidate_ids;
    Tensor draft_candidate_probs;
    const GdnReplayRecords* replay_records = nullptr;
    const ops::SamplingConfig* sampling    = nullptr;
    DFlashFeatureSink* feature_sink        = nullptr;
};

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t state_source_slot,
                         std::int32_t state_destination_slot, std::uint32_t mtp_proposal_extent,
                         std::uint32_t mtp_tree_paths, std::uint32_t mtp_tree_depth);
void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope);

// Publishes the acceptlog block of a round that ran as a CAPTURED CUDA Graph. A capture body is a
// graph definition and runs once, so the round that captures it can only record its tensors
// (acceptlog_record, inside the body, no CUDA call); the block itself is emitted from here, after
// the round's cudaGraphLaunch, where a synchronisation and a legacy-stream D2H copy are legal
// again. `width` is the round's column count (k + 1). Inert unless NINFER_ACCEPTLOG is set.
void acceptlog_replay_dump(int width);

[[nodiscard]] PrefillChunkResult prefill_text_chunk(PrefillContext& state,
                                                    std::span<const TokenId> ids,
                                                    std::uint32_t nominal_length,
                                                    std::optional<std::uint32_t> split_frontier,
                                                    bool finalize_at_end);

[[nodiscard]] PrefillChunkResult
prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> split_frontier, bool finalize_at_end);

struct MtpBridgeInput {
    const Tensor* previous_hidden = nullptr;
    std::int32_t position         = 0;
    std::array<std::int32_t, 3> rope_position{};
};

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose);
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding = nullptr);
void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge);

// Executes one exact-B ordinary decode traversal. All request rows enter through the stable
// ordinary ingress, share one model schedule, publish continuation hidden by selector, and leave
// through one compact egress transfer.
void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::GqaExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition);
void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::GqaExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable);

// Executes one exact-B MTP verification/alignment/proposal transaction. Each row may carry a
// different current and next proposal extent while the model traversal remains batched.
void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpCausalAttentionEnvelopes envelopes,
                              DecodeGraphDefinition& definition);
void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable);

[[nodiscard]] DFlashFeatureSink
dflash_feature_sink(PrefillContext& state, DFlashFeatureSink::PrefillConsumer consume_prefill = {});
void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition);
void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes,
                         ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable);

[[nodiscard]] DFlashFeatureSink
dflash2_feature_sink(PrefillContext& state, DFlashFeatureSink::PrefillConsumer consume_prefill = {});
void dflash2_append_context(DFlash2AppendContext& state, const Tensor& features,
                            const Tensor& positions, const Tensor& commit_counts,
                            const Tensor& lanes, const Tensor& table_rows,
                            ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void dflash2_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                            const Tensor& commit_counts, const Tensor& lanes,
                            const Tensor& table_rows,
                            ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void capture_dflash2_decode_batch(DFlash2BatchContext& state, std::int32_t batch_size,
                                  std::uint32_t k, DFlash2Envelopes envelopes,
                                  ops::GqaExecutionEnvelope target_envelope,
                                  DecodeGraphDefinition& definition);
void dflash2_decode_batch(DFlash2BatchContext& state, std::int32_t batch_size, std::uint32_t k,
                          DFlash2Envelopes envelopes,
                          ops::GqaExecutionEnvelope target_envelope,
                          DecodeGraphExecutable* executable);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
