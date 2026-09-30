#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/state_image.h>
#include <ninfer/targets/qwen3_6/startup_features.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;

struct DFlashPersistentLayout {
    qwen3_6::PagedKVCacheLayout full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept { return full.payload_bytes(); }
};

struct DFlash2PersistentLayout {
    CyclicKVCacheLayout local;
    CyclicKVCacheLayout rewrite_checkpoint_local;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return local.payload_bytes() + rewrite_checkpoint_local.payload_bytes();
    }
};

struct PersistentLayout {
    qwen3_6::DecoderStateLayout decoder;
    qwen3_6::StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    std::optional<DFlash2PersistentLayout> dflash2;
    qwen3_6::RoundStateLayout round;
    TensorLayout prefill_hidden;
    std::optional<TensorLayout> score_hidden;
    TensorLayout token_counts;
    TensorLayout sampling_config;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

struct VisionWorkspacePlan {
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill     = 0;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t dflash2_context  = 0;
    std::size_t dflash2_round    = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    std::optional<VisionWorkspacePlan> vision;
    std::size_t capacity = 0;
};

struct SequencePlanningInputs {
    WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    // Explicit --kv-capacity page count (when set below capacity) shrinks the
    // device page-pool floor: max_context then bounds only the rope domain.
    std::optional<std::uint32_t> kv_capacity_tokens;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    // MTP capture-width ladder rungs (kMtpWindowLadder, clamped to the target's MTP draft
    // domain). Empty = fixed-k: one captured width, no width selection. Non-empty = adaptive:
    // one captured graph per rung, selected per round by the survival/cost criterion, and
    // draft_window is the ladder top (the widest rung, which sizes the MTP decode frame).
    std::vector<std::uint32_t> mtp_ladder;
    // --draft-tree L,d (MTP only): the tree shape whose node budget L*d is draft_window above.
    // {0,0} = no tree. Both are carried into the plan so the runtime can tell a tree round from
    // a chain round without re-parsing the command line.
    std::uint32_t draft_tree_paths = 0;
    std::uint32_t draft_tree_depth = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    DType kv_dtype                         = DType::BF16;
    std::int32_t kv_quant_group            = 0;
    std::array<DType, 64> layer_kv_dtypes{};
    // Per-layer two-stage residual planes for the NVFP4 tier.
    std::array<bool, 64> kv_residual_layers{};
    // SEPARATION: V codec of the NVFP4 tier (iso4e default, e2m1 ablation).
    KvVCodec kv_v_codec = KvVCodec::Iso3;
    // SEPARATION: SO(4) rotation of K on write / Q on read; true = identity.
    bool kv_rotation_off = false;
    // SEPARATION: row-scale three-state spec (auto|off|<path>). Empty leaves
    // the decision to NINFER_KV_ROWSCALE, i.e. the pre-separation behaviour.
    std::string kv_row_scale_spec;
    // --stage-layers SPEC, verbatim: the pipeline stage partition this run walks. Carried as
    // the RAW spec so the ONE parser (core/stage_plan.h) is the only reader of the grammar,
    // and so the runtime can refuse against the artifact's own layer count. Empty == absent.
    // Declared here, immediately after kv_row_scale_spec, because the designator lists in
    // layouts_impl.h must follow declaration order and this is where its designator sits.
    std::string stage_layers_spec;
    // --stage-handoff DIR / --stage-handoff-cut: the boundary payload's directory and the
    // negative control. Carried beside the spec they belong to, for the same reason the
    // layer_kv_dtypes_set mask travels with the table it belongs to.
    std::string stage_handoff_dir;
    bool stage_handoff_cut = false;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_cuda_graph = true;
    std::uint32_t graph_capture_ceiling = 16;
    bool causal_scoring = false;
    ColdPolicy cold_policy      = ColdPolicy::None;
    // --max-cold-pages override: 0 = derive from the policy (see layouts_impl).
    std::uint32_t max_cold_pages    = 0;
    std::uint32_t cold_keep_tokens = 128;
    // The unload watermark (EngineOptions::unload_watermark_pages): free text-KV
    // pool pages at or below which the Engine proactively unloads what its
    // semantic directory judges unloadable. 0 = off; kUnloadWatermarkDerive =
    // derive the reserve from the plan's own prefill chunk.
    std::uint32_t unload_watermark_pages = kUnloadWatermarkDerive;
    std::uint64_t cold_host_bytes  = 7ULL << 30;
    // ColdPolicy::Disk: spill directory and budget.
    std::string cold_disk_path;
    std::uint64_t cold_disk_bytes = 32ULL << 30;
    int device          = 0;
    ContextCacheOptions context_cache;
    // WHICH layers the per-layer spec actually WROTE, for layer_kv_dtypes above
    // (see EngineOptions::kv_layer_storage_set). true makes layer_kv_dtypes[L]
    // authoritative even when it is DType::BF16, which is the only spelling of a
    // real per-layer BF16 tier under a quantized kv_dtype. All-false is the
    // pre-mask inheritance rule, bit-for-bit.
    // LAST MEMBER ON PURPOSE: appending cannot move an existing field's offset, so
    // an object compiled before this field existed still reads every field it
    // knows where it expects it.
    std::array<bool, 64> layer_kv_dtypes_set{};
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct SequencePlanImpl<NINFER_QWEN36_VARIANT> {
    typename NINFER_QWEN36_VARIANT::WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t kv_capacity              = 0;
    std::uint32_t main_page_groups         = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    // See SequencePlanningInputs: empty = fixed-k, non-empty = the adaptive capture ladder.
    std::vector<std::uint32_t> mtp_ladder;
    // See SequencePlanningInputs: {0,0} = no tree, else L paths per depth over d steps.
    std::uint32_t draft_tree_paths         = 0;
    std::uint32_t draft_tree_depth         = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    DType kv_dtype                         = DType::BF16;
    std::int32_t kv_quant_group            = 0;
    std::array<DType, 64> layer_kv_dtypes{};
    // Per-layer two-stage residual planes for the NVFP4 tier.
    std::array<bool, 64> kv_residual_layers{};
    // SEPARATION: V codec of the NVFP4 tier (iso4e default, e2m1 ablation).
    KvVCodec kv_v_codec = KvVCodec::Iso3;
    // SEPARATION: SO(4) rotation of K on write / Q on read; true = identity.
    bool kv_rotation_off = false;
    // SEPARATION: row-scale three-state spec (auto|off|<path>). Empty leaves
    // the decision to NINFER_KV_ROWSCALE, i.e. the pre-separation behaviour.
    std::string kv_row_scale_spec;
    // --stage-layers SPEC, verbatim: the pipeline stage partition this run walks. Carried as
    // the RAW spec so the ONE parser (core/stage_plan.h) is the only reader of the grammar,
    // and so the runtime can refuse against the artifact's own layer count. Empty == absent.
    // Declared here, immediately after kv_row_scale_spec, because the designator lists in
    // layouts_impl.h must follow declaration order and this is where its designator sits.
    std::string stage_layers_spec;
    // --stage-handoff DIR / --stage-handoff-cut: the boundary payload's directory and the
    // negative control. Carried beside the spec they belong to, for the same reason the
    // layer_kv_dtypes_set mask travels with the table it belongs to.
    std::string stage_handoff_dir;
    bool stage_handoff_cut = false;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_cuda_graph = true;
    ColdPolicy cold_policy      = ColdPolicy::None;
    // --max-cold-pages override: 0 = derive from the policy (see layouts_impl).
    std::uint32_t max_cold_pages    = 0;
    std::uint32_t cold_keep_tokens = 128;
    // See SequencePlanningInputs::unload_watermark_pages: this is the resolved
    // copy, so ProgramImplCore reads ONE value and never re-derives it.
    std::uint32_t unload_watermark_pages = kUnloadWatermarkDerive;
    std::uint64_t cold_host_bytes  = 7ULL << 30;
    // ColdPolicy::Disk: spill directory and budget.
    std::string cold_disk_path;
    std::uint64_t cold_disk_bytes = 32ULL << 30;
    std::uint32_t graph_capture_ceiling = 16;
    bool causal_scoring = false;
    int device          = 0;
    ContextCacheOptions context_cache;
    NINFER_QWEN36_RUNTIME_NS::PersistentLayout persistent;
    NINFER_QWEN36_RUNTIME_NS::WorkspacePlan workspace;
    std::size_t graph_allowance_bytes    = 0;
    std::size_t device_reservation_bytes = 0;
    // The "was this slot written?" mask for layer_kv_dtypes above; see
    // SequencePlanningInputs and EngineOptions::kv_layer_storage_set.
    // LAST MEMBER ON PURPOSE, same reason as SequencePlanningInputs.
    std::array<bool, 64> layer_kv_dtypes_set{};
};

template <>
struct SequencePlannerImpl<NINFER_QWEN36_VARIANT> {
    NINFER_QWEN36_RUNTIME_NS::SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl<NINFER_QWEN36_VARIANT>> minimum;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using SequencePlanImpl = qwen3_6::detail::SequencePlanImpl<Variant>;

[[nodiscard]] std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
