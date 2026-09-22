#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/host_kv_arena.h"
#include "runtime/engine/context_cost.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/cold_host_tier.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"
#include "targets/qwen3_6/impl/runtime/mtp_window_cut.h"
#include "spec/turn_recall_journal.h"
#include "spec/sum_dir.h"

#include <algorithm>
#include <cstdint>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData    = qwen3_6::PreparedPromptData;
using RewriteCheckpointKind = qwen3_6::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_6::RewriteCheckpointSpec;

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return path == ReusePath::PrivateTurnClosure || path == ReusePath::PrivateResponseReplay;
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? ReusePath::PrivateTurnClosure
                                                      : ReusePath::PrivateResponseReplay;
}

[[nodiscard]] constexpr runtime::CheckpointKind
checkpoint_kind(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? runtime::CheckpointKind::TurnClosure
                                                      : runtime::CheckpointKind::ResponseReplay;
}

enum class RewriteCheckpointDisposition : std::uint8_t {
    RetainExisting,
    ReplaceAtCommittedFrontier,
    DropOptional,
};

struct PreparedCaptureBacking {
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
};

struct PreparedCaptureIdentity {
    std::shared_ptr<const PreparedCaptureBacking> backing;
    qwen3_6::PrefixShortlistKey shortlist_key;
    runtime::PrefillWork rebuild_work;

    [[nodiscard]] std::span<const TokenId> ledger() const noexcept {
        if (!backing || shortlist_key.frontier > backing->ledger.size()) { return {}; }
        return std::span<const TokenId>(backing->ledger).first(shortlist_key.frontier);
    }

    [[nodiscard]] const qwen3_6::detail::ResidentPrefixIdentity* prefix_identity() const noexcept {
        return backing ? &backing->prefix_identity : nullptr;
    }

    [[nodiscard]] bool prefix_equals(const PreparedCaptureIdentity& other) const {
        const std::span<const TokenId> left  = ledger();
        const std::span<const TokenId> right = other.ledger();
        const auto* left_identity            = prefix_identity();
        const auto* right_identity           = other.prefix_identity();
        return left_identity != nullptr && right_identity != nullptr &&
               left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin()) &&
               left_identity->prefix_equals(*right_identity, left.size());
    }
};

struct CaptureGroup {
    std::shared_ptr<const PreparedCaptureIdentity> identity;
    std::optional<RewriteCheckpointKind> rewrite;
    std::uint32_t frontier                  = 0;
    std::uint32_t input_order               = 0;
    bool shared                             = false;
    bool long_anchor                        = false;
    SharedCandidateEvidence shared_evidence = SharedCandidateEvidence::None;
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

// Concrete allocator quantities never cross the Program implementation boundary.
struct PhysicalDeviceResources {
    std::uint32_t active_lanes     = 0;
    std::uint32_t state_slots      = 0;
    std::uint32_t main_kv_pages    = 0;
    std::uint32_t backend_kv_pages = 0;

    [[nodiscard]] friend constexpr bool operator==(PhysicalDeviceResources,
                                                   PhysicalDeviceResources) noexcept = default;
};

struct PhysicalHostResources {
    std::uint32_t state_slots = 0;
    std::size_t kv_bytes      = 0;

    [[nodiscard]] friend constexpr bool operator==(PhysicalHostResources,
                                                   PhysicalHostResources) noexcept = default;
};

struct PhysicalResources {
    PhysicalDeviceResources device;
    PhysicalHostResources host;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalResources&,
                                                   const PhysicalResources&) noexcept = default;
};

struct PhysicalDemand {
    PhysicalResources active_entitlement;
    PhysicalResources reservation_added;
    PhysicalResources reservation_credit;
    PhysicalResources physical_peak_additional;
    PhysicalResources final_removed;
    PhysicalResources final_added;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalDemand&,
                                                   const PhysicalDemand&) noexcept = default;
};

struct PhysicalDelta {
    PhysicalResources removed;
    PhysicalResources added;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalDelta&,
                                                   const PhysicalDelta&) noexcept = default;
};

struct PhysicalPressureEffect {
    PhysicalDelta aggregate_delta;
    PhysicalDelta final_ownership_delta;
    PhysicalDelta active_entitlement_delta;
    PhysicalResources source_optional_resources_added;
    // Pressure can remove the last checkpoint references outside a consumed private source.  The
    // selected StateImage must then be moved into the active lineage instead of forked.  This is a
    // property of the complete post-reference target, not of an individual owner decision.
    std::optional<bool> source_state_fork_required;
    std::optional<bool> source_text_prefix_fork_required;
    std::optional<bool> source_backend_prefix_fork_required;

    [[nodiscard]] friend constexpr bool
    operator==(const PhysicalPressureEffect&, const PhysicalPressureEffect&) noexcept = default;
};

enum class PressureStateDecision : std::uint8_t {
    None,
    DropEndpointDeviceDuplicate,
    DemoteEndpointToHost,
    DropEndpointHostDuplicate,
    DropRewriteDeviceDuplicate,
    DemoteRewriteToHost,
    DropRewriteHostDuplicate,
    DropSharedDeviceDuplicate,
    DemoteSharedToHost,
    DropSharedHostDuplicate,
};

enum class PressureKVDecisionKind : std::uint8_t {
    None,
    DropDeviceDuplicate,
    DemoteToHost,
    DropHostDuplicate,
};

struct PressureKVDecision {
    std::uint32_t begin_page    = 0;
    std::uint32_t page_count    = 0;
    PressureKVDecisionKind kind = PressureKVDecisionKind::None;

    [[nodiscard]] friend constexpr bool operator==(PressureKVDecision,
                                                   PressureKVDecision) noexcept = default;
};

// One Program-private complete outcome for one eligible owner. It may combine State, Main KV,
// Backend KV, and checkpoint changes; common scheduling never observes these physical decisions.
struct PressureDecision {
    std::uint64_t id = 0;
    std::vector<PressureStateDecision> state_changes;
    std::vector<PressureKVDecision> main_kv_changes;
    std::vector<PressureKVDecision> backend_kv_changes;
    std::vector<runtime::CheckpointRef> dropped_checkpoints;
    PhysicalDelta checkpoint_drop_effect;
    PhysicalDelta effect;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::uint32_t checkpoint_drops = 0;
    bool evicts_continuation       = false;
    bool shared_owner              = false;

    [[nodiscard]] friend bool operator==(const PressureDecision&,
                                         const PressureDecision&) noexcept = default;
};

struct PressureBaselineRecovery {
    std::uint32_t owner_ordinal = 0;
    runtime::CheckpointRef checkpoint;
    std::uint64_t recovery_ns = 0;
};

struct CaptureAssessmentImpl {
    PhysicalDemand demand;
    PhysicalDelta active_entitlement_delta;
    PhysicalResources capacity_preparation_removed;
};

template <>
struct RequestBasePlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    detail::PhysicalDemand root_demand;
    runtime::PrefillWork root_rebuild_work;
    std::uint32_t root_rebuild_tail_begin = 0;
    qwen3_6::PreparedContextCache context_cache;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_6::VisionControlPlan> vision_control_plan;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<NINFER_QWEN36_RUNTIME_NS::CaptureGroup> capture_groups;
    std::vector<NINFER_QWEN36_RUNTIME_NS::CaptureGroup> shared_candidates;
    qwen3_6::detail::PrefixShortlistDigests prefix_digests;
    std::uint32_t prefix_identity_tag = 0;
    bool allow_prefix_reuse           = false;
};

template <>
struct AdmissionCandidateImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    runtime::IdentityMaterializationAssessment identity_assessment;
    std::uint64_t planning_revision = 0;
    // Pressure outcomes are canonicalized against the candidate's identity peak.  Composition
    // rewrites demand to the selected post-pressure peak, so regenerating an outcome from that
    // rewritten demand would compare it against a different problem at seal time.
    detail::PhysicalResources identity_pressure_deficit;
    // A structurally valid pressure target can still be blocked by Host extent geometry even when
    // aggregate free bytes are sufficient. Keep the blocked allocation work explicit so a child
    // target can release Host replicas instead of being mistaken for a structurally invalid node.
    std::size_t blocked_host_allocation_bytes = 0;
    detail::PhysicalDemand demand;
    detail::PhysicalResources source_resources;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::Root;
    std::uint32_t reuse_base                  = 0;
    NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointDisposition rewrite_disposition =
        NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointDisposition::DropOptional;
    std::vector<NINFER_QWEN36_RUNTIME_NS::CaptureGroup> capture_groups;
    std::vector<NINFER_QWEN36_RUNTIME_NS::CaptureGroup> shared_candidates;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    runtime::LaneId destination{};
    std::uint64_t destination_epoch = 0;
    bool has_source                 = false;
    bool has_shared_source          = false;
    std::optional<runtime::CheckpointRef> selected_checkpoint;
    std::uint32_t source_index             = 0;
    std::uint64_t source_generation        = 0;
    std::uint32_t shared_source_index      = 0;
    std::uint64_t shared_source_generation = 0;
    runtime::PrefillWork root_rebuild_work;
    std::uint32_t root_rebuild_tail_begin = 0;
    runtime::PrefillWork remaining_prefill_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    runtime::ClaimDisposition source_disposition = runtime::ClaimDisposition::ConsumedToActive;
    detail::PhysicalResources active_optional_resources;
    bool state_fork_required           = false;
    bool text_prefix_fork_required     = false;
    bool backend_prefix_fork_required  = false;
    bool text_retained_tail_release    = false;
    bool backend_retained_tail_release = false;
    bool needs_transfer                = false;
    bool capture_pressure              = false;
    std::vector<qwen3_6::detail::PressureDecision> pressure_options;
    std::vector<std::uint32_t> pressure_indices;
    std::vector<std::uint64_t> pressure_generations;
    std::vector<qwen3_6::detail::PressureDecision> shared_pressure_options;
    std::vector<std::uint32_t> shared_pressure_indices;
    std::vector<std::uint64_t> shared_pressure_generations;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6 {

inline CaptureAssessment::CaptureAssessment()
    : implementation(std::make_shared<detail::CaptureAssessmentImpl>()) {}

template <>
AdmissionCandidate<NINFER_QWEN36_VARIANT>::AdmissionCandidate(
    std::unique_ptr<detail::AdmissionCandidateImpl<NINFER_QWEN36_VARIANT>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
AdmissionCandidate<NINFER_QWEN36_VARIANT>::AdmissionCandidate(AdmissionCandidate&&) noexcept =
    default;

template <>
AdmissionCandidate<NINFER_QWEN36_VARIANT>&
AdmissionCandidate<NINFER_QWEN36_VARIANT>::operator=(AdmissionCandidate&&) noexcept = default;

template <>
AdmissionCandidate<NINFER_QWEN36_VARIANT>::~AdmissionCandidate() = default;

template <>
const runtime::RequestPlanSummary&
AdmissionCandidate<NINFER_QWEN36_VARIANT>::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

template <>
const runtime::IdentityMaterializationAssessment&
AdmissionCandidate<NINFER_QWEN36_VARIANT>::identity_assessment() const noexcept {
    static const runtime::IdentityMaterializationAssessment empty;
    return impl_ != nullptr ? impl_->identity_assessment : empty;
}

} // namespace ninfer::targets::qwen3_6

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using AdmissionCandidateImpl = qwen3_6::detail::AdmissionCandidateImpl<Variant>;
using RequestBasePlanImpl    = qwen3_6::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Finishable,
};

enum class ContinuationSlotRole : std::uint8_t {
    Free,
    ReservedMaterialization,
    Active,
    Catalogued,
};

struct ContinuationSlot {
    ContinuationSlotRole role = ContinuationSlotRole::Free;
    std::uint64_t generation  = 1;
};

struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
    runtime::PrefillWork rebuild_work;
};

struct LongAnchorCheckpoint {
    StateImageHandle state;
    std::uint32_t frontier = 0;
    std::uint32_t ordinal  = 0;
    runtime::PrefillWork rebuild_work;
};

struct SequenceKVBundle {
    KVAddressSpaceHandle text;
    std::optional<KVAddressSpaceHandle> backend;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class = 0;
    // Captured MTP draft width of this profile, i.e. the ladder rung it belongs to. 0 = not a
    // rung (the ordinary/DFlash families, and every profile of a fixed-k MTP run), in which
    // case the replay's width filter is a no-op and behaviour is the pre-ladder one. Selecting
    // by width is what makes the marginal draft column cost b_width: the graph that runs IS the
    // width the criterion chose.
    std::uint32_t draft_width = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    ActiveStateBinding state;
    std::optional<StateImageHandle> rewrite_state;
    std::optional<StateImageHandle> reserved_state;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    qwen3_6::detail::PrefixShortlistDigests prefix_digests;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    // The DRAFT TREE the next verify round is going to check (--draft-tree L,d with L > 1),
    // published by the proposal side in verify-column order: column j (1-based: column 0 is the
    // anchor) carries the node at depth mtp_tree_depths[j] whose ancestor set is mtp_tree_masks[j]
    // (ddtree::column_depth / ddtree::column_ancestors, include/ninfer/ops/dflash2_ddtree.h:314-345).
    // 0 columns = no tree: the round is a chain and every tree path below is a no-op. This is the
    // ONLY place a tree can enter the runtime, which is why an L > 1 tree round without it is
    // refused instead of being verified as a chain.
    std::uint32_t mtp_tree_columns = 0;
    std::array<std::uint32_t, qwen3_6::kMtpDecodeMaximumWidth> mtp_tree_depths{};
    std::array<std::uint64_t, qwen3_6::kMtpDecodeMaximumWidth> mtp_tree_masks{};
    // Adaptive draft-window control (survival/cost criterion, see mtp_window_cut.h):
    // the per-sequence depth-conditional accept statistics and the extent the round in
    // flight actually carried (needed to fold the finished round correctly).
    qwen3_6::detail::MtpWindowState mtp_window{};
    std::uint32_t mtp_drafted_extent = 0;
    // Adaptive capture width: the ladder rung the criterion last selected. 0 means "no decision
    // yet" and the round starts on the ladder TOP (the same staged-wide start the fixed-k path
    // had with its initial extent), so a fresh request never begins at the narrow end.
    std::uint32_t mtp_target_width = 0;
    // Coarse-cadence counter and the realized-width accumulator behind
    // SpeculativeStats::mean_window. Both count only rounds that actually drafted.
    std::uint64_t mtp_width_rounds = 0;
    std::uint64_t mtp_width_sum = 0;
    // Runtime (a, b) fit behind the criterion's cost ratio. The ratio is the ONE calibration the
    // criterion cannot derive from its own estimator (that is why it sat in this header as a
    // constant, calibrated on a different head): it is measured from the round wall times and the
    // widths they were run at, so the threshold is priced for the configuration actually running.
    qwen3_6::detail::MtpCostCalibration mtp_cost{};
    bool tail_hidden_valid        = false;
    bool state_source_retained    = false;
    bool endpoint_valid           = false;
    RewriteCheckpoint rewrite_checkpoint;
    std::vector<LongAnchorCheckpoint> long_anchors;
    std::vector<std::uint32_t> shared_prefix_references;
    runtime::PrefillWork rebuild_work;
    std::uint32_t rebuild_tail_begin = 0;

    // Cold-pool bookkeeping: text pages currently detached into raw cold slots.
    // Released with the sequence or when the rewrite path warms the prefix back
    // into physical pages.
    //
    // The device slot and the spill-file slot are two independent resources with
    // independent lifetimes, so they are tracked apart:
    //   slot      - the device working-set slot that holds this page's compressed
    //               bytes, or -1 when the page holds none. Window policy holds one
    //               for as long as the page is cold, because the block table's
    //               sentinel is the only handle decode has on a cold page.
    //   file_slot - the spill-file slot (ColdPolicy::Disk only, else -1). Its
    //               on-disk offset is file_slot * <layer slot stride>, a location
    //               that is stable for the page's whole cold lifetime. It used to
    //               be the *device* slot index, so recycling a device slot
    //               silently rewrote another page's file region.
    struct ColdPageEntry {
        std::uint32_t page;
        std::int32_t slot;
        std::int32_t file_slot = -1;
    };
    std::vector<ColdPageEntry> cold_pages;
    // First logical page not yet offered to the cold pool; compression scans
    // forward from here so each round only visits the newly retired pages.
    std::uint32_t cold_frontier = 0;

    // Cold Host tier (--cold-policy host): pages whose device replica was
    // released into the pinned Host pool. They carry no device slot -- the page
    // is restored by the ordinary Host -> Device materialization path -- so only
    // the frontier is needed here; the Host replica on the logical page is the
    // record of the eviction itself (and of its restore: the descriptor loses it
    // when a device replica comes back).
    std::uint32_t host_cold_frontier = 0;

    // (6) per-round external recall (P5/P6/P7). `recall_recorded` is this sequence's
    // live L0 records in ascending page order -- the journal's own view of what it has
    // been told about this sequence, so a round can diff it against `cold_pages` (which
    // is ascending by construction: the compressor walks cold_frontier forward and a
    // restore erases from the middle) with no log I/O and no lookup.
    // `recall_sequence_tag` rides every record, so a page number reused by a later
    // session cannot collide with this one's. The journal is a MIRROR of the spill
    // files, never the only replica of an attended page: losing all of this costs a
    // re-read, never an answer.
    std::vector<spec::turn_recall::RecallRecord> recall_recorded;
    std::uint32_t recall_sequence_tag = 0;
};

struct SharedPrefixState {
    std::optional<SequenceKVBundle> kv;
    StateImageHandle state;
    std::shared_ptr<const PreparedCaptureIdentity> identity;
    std::uint32_t frontier         = 0;
    std::uint32_t backend_frontier = 0;
    std::int32_t rope_delta        = 0;
    bool tail_hidden_valid         = false;
    runtime::PrefillWork rebuild_work;
    std::uint32_t active_references = 0;
};

enum class SharedPrefixSlotRole : std::uint8_t {
    Free,
    ReservedCapture,
    ReservedReplacement,
    Catalogued,
};

struct SharedPrefixSlot {
    SharedPrefixSlotRole role = SharedPrefixSlotRole::Free;
    std::uint64_t generation  = 1;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;
    detail::PhysicalResources active_resources;
    detail::PhysicalResources optional_resources;
    bool publish_continuation = true;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        std::vector<CaptureGroup> capture_groups;
        std::size_t next_capture            = 0;
        std::uint64_t pending_capture_offer = 0;
        std::uint32_t base                  = 0;
        std::uint32_t cursor                = 0;
        std::uint32_t prompt_tokens         = 0;
        std::uint32_t initial_mtp_extent    = 0;
        double elapsed_seconds              = 0.0;
        bool prepare_mtp                    = false;
        ReusePath reuse                     = ReusePath::Root;
        MtpBridgeMode mtp_bridge            = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, const SequencePlanImpl& plan,
                    DeviceContext& device);
    ~ProgramImplCore() noexcept;

    // W6: shrink (never grow) the prefill unit so a decoding request is not blocked by a full-size
    // chunk. Workspace and persistent buffers stay sized for prefill_chunk_capacity, so any value
    // in [128, prefill_chunk_capacity] is safe. Aligned down to the 128-token prefill alignment.
    void set_prefill_chunk(std::uint32_t chunk) noexcept {
        constexpr std::uint32_t kAlignment = 128;
        std::uint32_t clamped = chunk < kAlignment ? kAlignment : chunk;
        if (clamped > prefill_chunk_capacity) { clamped = prefill_chunk_capacity; }
        prefill_chunk = clamped - (clamped % kAlignment);
        if (prefill_chunk < kAlignment) { prefill_chunk = kAlignment; }
    }

    [[nodiscard]] RequestBasePlan plan_request(const PreparedPromptData& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPromptData&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::optional<AdmissionCandidate> inspect_admission(
        const PreparedPromptData& prompt, const RequestBasePlan& base, runtime::LaneId destination,
        const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
        std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source,
        const runtime::ContextMachineCostModel& machine_cost);
    [[nodiscard]] std::optional<AdmissionCandidate> seal_materialization(
        const AdmissionCandidate& admission, const PreparedPromptData& prompt,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision* const> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision* const> shared_pressure_options);
    [[nodiscard]] AdmissionCandidate
    make_capture_pressure_candidate(const CaptureAssessment& assessment,
                                    const runtime::ContextMachineCostModel& machine_cost) const;
    void select_shared_captures(AdmissionCandidate& candidate, const PreparedPromptData& prompt,
                                std::span<const std::uint32_t> frontiers);
    [[nodiscard]] std::uint64_t
    shared_capture_split_cost_ns(const AdmissionCandidate& candidate,
                                 const PreparedPromptData& prompt,
                                 std::span<const std::uint32_t> frontiers,
                                 const runtime::ContextMachineCostModel& machine_cost);
    [[nodiscard]] runtime::PreflightStatus
    revalidate_materialization(const AdmissionCandidate& plan,
                               const PreparedPromptData& prompt) const;
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_materialization(AdmissionCandidate&& plan, PreparedPromptData&& prompt,
                            runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool
    persistent_backfill_safe(const RequestBasePlan& blocked_head,
                             const AdmissionCandidate& candidate,
                             std::span<const SequenceHandle> persistent_borrowers) const;
    [[nodiscard]] ContextTransactionProgress<Variant>
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                    const SharedPrefixHandle* replacement,
                    std::optional<runtime::CheckpointRef> private_replacement,
                    bool permit_shared_publication,
                    const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] std::uint64_t
    checkpoint_recovery_ns(const ContinuationHandle& owner, runtime::CheckpointRef checkpoint,
                           const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] std::uint64_t
    checkpoint_recovery_ns(const SharedPrefixHandle& owner, runtime::CheckpointRef checkpoint,
                           const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer& offer,
                                              const SharedPrefixHandle& shared) const;
    void skip_capture(CaptureOffer&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           bool permit_shared_publication,
                           const runtime::ContextMachineCostModel& machine_cost,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
        const SharedPrefixHandle* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        AdmissionCandidate&& pressure, const runtime::ContextMachineCostModel& machine_cost,
        runtime::CancellationFlagView cancellation);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending,
                                      std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    [[nodiscard]] ReleaseResult release_continuation(ContinuationHandle&& continuation) noexcept;
    [[nodiscard]] ReleaseResult release_shared_prefix(SharedPrefixHandle&& shared) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] detail::PhysicalResources admission_capacity() const noexcept;
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return resource_revision_; }

    [[nodiscard]] qwen3_6::PhysicalUsageSnapshot physical_usage() const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    friend struct qwen3_6::detail::PressurePlanningSessionImpl<Variant>;

    const LoadedModelData& model;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const ContextCacheOptions context_cache;
    const std::uint32_t continuation_capacity;
    const std::uint32_t shared_prefix_capacity;
    // Startup chunk (workspace and persistent buffers are sized for it); the mutable copy may be
    // shrunk at runtime by the W6 bandwidth governor so a decoding request is not blocked by a
    // full-size prefill unit. Shrinking is always safe, growing is not.
    const std::uint32_t prefill_chunk_capacity;
    std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    // Capture-width ladder rungs (empty for a fixed-k run). See SequencePlanningInputs: when
    // non-empty the MTP replay selects its graph by the criterion's chosen rung, and
    // draft_window is the ladder top, i.e. the frame/buffer width.
    const std::vector<std::uint32_t> mtp_ladder;
    // --draft-tree L,d (MTP). L*d == draft_window when it is a tree round, because the node
    // budget is the live verify width and the frame has one column per draft. {0,0} = chain.
    // Read by the round's ingress fill (program_impl.h), which refuses a tree round with no
    // published tree instead of verifying a chain and calling it a tree.
    const std::uint32_t draft_tree_paths;
    const std::uint32_t draft_tree_depth;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const bool causal_scoring;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::unique_ptr<HostKVArena> host_kv_arena;
    std::unique_ptr<LogicalKVPageStore> text_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> text_kv_addresses;
    std::unique_ptr<LogicalKVPageStore> backend_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> backend_kv_addresses;
    std::unique_ptr<HostKVExtentStore> host_kv_extents;
    // Cold Host tier (--cold-policy host): the eviction consumer's own pinned
    // pool. Deliberately a SEPARATE arena from host_kv_arena, so
    // --cold-host-bytes is enforced independently of the context-cache Host KV
    // budget (same reasoning as weight_host_offload_bytes in ninfer/types.h:
    // different lifetimes and failure modes, one budget per pool).
    std::unique_ptr<ColdHostTier> cold_host_tier;
    bool cold_host_tier_inert_reported = false;
    std::size_t text_host_kv_page_stride    = 0;
    std::size_t backend_host_kv_page_stride = 0;
    std::unique_ptr<qwen3_6::StateImageDevicePool> state_images;
    std::unique_ptr<qwen3_6::HostStatePool> host_state_images;
    std::unique_ptr<StateImageStore> state_store;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<ops::GdnReplayFoldPlan> replay_fold;
    std::optional<DFlashPersistentState> dflash;
    std::optional<DFlash2PersistentState> dflash2;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    std::optional<Tensor> score_hidden;
    Tensor sampling_config;
    Tensor token_counts;

    std::vector<SequenceState> continuation_states;
    std::vector<ContinuationSlot> continuation_slots;
    std::vector<SharedPrefixState> shared_prefix_states;
    std::vector<SharedPrefixSlot> shared_prefix_slots;
    std::array<std::uint32_t, kMaximumConcurrency> active_continuations{};
    std::array<RequestControl, kMaximumConcurrency> requests;
    std::array<std::uint64_t, kMaximumConcurrency> lane_epochs{};

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;
    DecodeGraphFamily dflash2_graphs;

    PinnedHostBuffer round_host;
    std::optional<PinnedHostBuffer> score_logprobs_host;
    TokenId* host_tokens = nullptr;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_6::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_6::DFlashDecodeEgress* dflash_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash2_host;
    qwen3_6::DFlashDecodeIngress* dflash2_host_ingress = nullptr;
    qwen3_6::DFlashDecodeEgress* dflash2_host_egress   = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;

    // Cold-pool maintenance (rev 2b): staging + per-step compress pass.
    ColdPolicy cold_policy      = ColdPolicy::None;
    std::uint32_t cold_keep_tokens = 128;
    // THE UNLOAD WATERMARK, resolved from the plan (see
    // EngineOptions::unload_watermark_pages). 0 = off, which is byte-for-byte
    // the pre-watermark behaviour: the ONLY other consumer of the free pool,
    // enqueue_cold_compressions(), still runs exactly when --cold-policy says so.
    // A sentinel at construction time means "derive from prefill_chunk_capacity",
    // so the member is never left holding kUnloadWatermarkDerive at run time.
    std::uint32_t unload_watermark_pages = 0;
    std::uint64_t cold_host_bytes  = 7ULL << 30;
    void* cold_requant_codes       = nullptr;
    void* cold_requant_scales      = nullptr;
    std::uint32_t cold_requant_heads = 0;
    // ColdPolicy::Disk: per-layer spill files (one fixed stride per slot) and
    // a pinned staging buffer for the D2H/H2D legs. Files live for the
    // process lifetime; the payload is the same compressed slot bytes.
    // Double-buffered staging lets the warm path prefetch the next slot while
    // the previous one decodes.
    std::string cold_disk_path;
    std::uint64_t cold_disk_bytes = 32ULL << 30;
    std::vector<FILE*> cold_disk_files;
    void* cold_disk_staging[2]    = {nullptr, nullptr};
    std::size_t cold_disk_slot_bytes = 0;
    // File-slot space for ColdPolicy::Disk: every spilled page owns one stable
    // file slot for its whole cold lifetime, and the on-disk offset is
    // file_slot * <that layer's slot stride>. File slots are a resource of their
    // own, sized from --cold-disk-bytes (which used to be parsed and then
    // ignored entirely); the device cold slots are a separate, shared working
    // set, so returning one can never move a page's bytes on disk.
    std::uint64_t cold_disk_file_slots = 0;
    std::vector<std::uint8_t> cold_disk_file_used;
    // [TEXT-CARGO] THE COLD PAYLOAD, AS TEXT. The user's order of 2026-09-15 is that the
    // retired block's DURABLE payload is its 64 token ids and a recall re-prefills them
    // instead of loading KV back ("召回 = 把这段文字重新 prefill"), so this is the cargo the
    // cold path writes beside the per-layer KV spill files. It lives at
    // `<--cold-disk-path>/ninfer_text.cargo`, addressed by BLOCK NUMBER (never by file
    // slot), one fixed 256-byte record per block, format owned by
    // spec/turn_recall_journal.h. `cold_text_kv_bytes` is the KV mirror's own byte count for
    // the SAME retired pages, kept ONLY so the two payloads can be printed side by side: the
    // whole point of the change is that they differ by four orders of magnitude, and a number
    // that is only asserted in a comment is not a measurement.
    std::unique_ptr<spec::turn_recall::TurnRecallTextCargo> cold_text_cargo;
    std::uint64_t cold_text_kv_bytes      = 0;
    std::uint64_t cold_text_cargo_blocks  = 0;
    std::uint64_t cold_text_cargo_refused = 0;
    // [TEXT-CARGO] THE INDEX BESIDE THE CARGO. The cargo holds the TEXT of every retired block; this
    // is the DIRECTORY that makes the text FINDABLE -- one `SumDirRow` per stored block, carrying the
    // block's content identity, its page, its real length and (from `set_summary`) its catalogue
    // line. It is armed and disarmed with the cargo because it is the cargo's index and nothing else:
    // a page the directory names is read back from the cargo file, and a directory with no cargo
    // behind it could only produce misses.
    //
    // WHY IT LIVES HERE AND NOT IN A SERVING LOOP: because the WRITE has to happen where the block
    // becomes cold. `SumDir::set_summary`'s only callers were two test files before this landing --
    // the catalogue column had no producer in any real run, so `search_summaries` and the registered
    // query `QSummarySubstring` could only ever answer "nothing". The producer is the retire path in
    // program_impl.h, and this member is what it writes into.
    std::unique_ptr<spec::sum_dir::SumDir> text_directory;
    std::uint64_t text_directory_rows     = 0; // rows written (1 per stored block)
    std::uint64_t text_directory_refused  = 0; // blocks stored whose codec the directory will not admit
    [[nodiscard]] std::int32_t allocate_cold_disk_file_slot() noexcept;
    void release_cold_disk_file_slot(std::int32_t file_slot) noexcept;
    void prefetch_cold_pages(SequenceState& sequence, std::uint32_t pages,
                             std::span<const std::int32_t> file_slots);
    // The page-admission filter, i.e. the ONE seam through which a caller says
    // WHICH of the retirable pages this pass may actually retire. It exists so the
    // passive path (--cold-policy: "everything older than the newest
    // cold_keep_tokens") and the proactive path (the watermark: "the blocks the
    // semantic judge calls unloadable") can share the spill body -- the device
    // slot, the per-layer codec pack, the validity check, the sentinel, the file
    // slot -- instead of duplicating it. `nullptr` admits every retirable page and
    // is therefore byte-for-byte the pre-watermark pass.
    using ColdPageAdmission = bool (*)(void* context, std::uint32_t page);
    // Returns the number of pages THIS pass actually moved into a cold slot (0 when it
    // moved none). The watermark's own line needs that count and cannot infer it: a
    // device-slot cold transfer is FOOTPRINT-NEUTRAL in the pool (transfer_to_cold ->
    // dematerialize_one credits the page back to the SAME address's reservation), so
    // the free-page delta across this pass is 0 even when it retired nine hundred
    // pages -- which is exactly the reading that used to look like "the judge refused
    // everything" and sent the safety valve down the passive leg for no reason.
    std::uint32_t enqueue_cold_compressions(SequenceState& sequence,
                                           ColdPageAdmission admit = nullptr,
                                           void* admit_context = nullptr);
    // THE PROACTIVE (WATERMARK) TRIGGER, and the user's design order of
    // 2026-09-15: "达到显存剩下多少的时候主动开始总结压缩而不是等到溢出了".
    //
    // It is a SEPARATE entry point from enqueue_cold_compressions() on purpose:
    // that one is the PASSIVE path (--cold-policy says which pages are retired
    // relative to the frontier), and this one is the PROACTIVE path (the free
    // pool has fallen to the watermark, so the Engine asks its semantic
    // directory which blocks may be unloaded at all). The two share the spill
    // legs -- store.transfer_to_cold + cold_pages + file slots -- and share
    // nothing else. Returns the number of pages it retired.
    //
    // It never runs when unload_watermark_pages == 0, so OFF is the pre-watermark
    // behaviour exactly.
    std::uint32_t unload_watermark_trigger(SequenceState& sequence);
    // Free text-KV pool pages, i.e. capacity - allocated - reserved. Saturates at
    // 0 rather than wrapping: a pool that is over-subscribed is a fact, and the
    // watermark must read it as "no headroom" and not as 4 billion pages.
    [[nodiscard]] std::uint32_t text_kv_free_pool_pages() const noexcept;
    // THE ONE SOURCE of "how many text-KV pages a sequence must have RESIDENT",
    // and the reason it is one function and not two statements: the admission
    // reserve (request_plan_impl.h, root_demand) and the watermark's free-pool
    // target are the SAME quantity read from opposite sides -- "never reserve more
    // than must stay resident" and "never let the free pool fall below what must
    // stay resident". Two independent computations of it would drift, and the drift
    // would show up as the worst possible shape: the watermark unloading while the
    // admission still refuses.
    //
    //   newest cold_keep_tokens  -- the engine's own floor: the decode cold-read path
    //                               requires the newest cold_keep_tokens to be hot
    //                               (:11050-11055), so they are resident by rule;
    //   one prefill chunk        -- a chunk's KV cannot be written half-way, so this
    //                               is the largest unit that must be placeable without
    //                               freeing anything first;
    //   the watermark itself     -- when the operator sets a larger reserve than the
    //                               floor, the reserve IS that value (it is the target
    //                               the free pool is kept at, and the admission must
    //                               not reserve what the watermark is trying to keep
    //                               free).
    // Returns 0 when the watermark is OFF, which is how the caller knows to keep the
    // full-span reservation and therefore how the negative control still refuses.
    [[nodiscard]] std::uint32_t resident_text_kv_pages_required() const noexcept;
    // One evidence line per trigger, and NOTHING when nothing triggered -- the
    // --cold-policy path prints through its own ["[cold] compressed N"] line.
    // "没有这行就等于没实现" is the reason it exists: a silent plan.empty() is
    // exactly the failure mode this report is about.
    // [LONGCTX-FIX-D] THE DISTRIBUTION IS AN OPTIONAL ARGUMENT ON PURPOSE. The five
    // fields above keep their exact meaning, their order and their call sites; the two
    // early-exit lines (no-slot-tier / no-requant-codes) deliberately pass no histogram
    // and still compile unchanged. When it IS passed it must be an array of exactly 7
    // counts indexed by `SumDirAdmissibility`'s own enum order (sum_dir.h:838-846):
    // 0 admissible, 1 admissible-text-only, 2 control-token, 3 pinned-by-anchor,
    // 4 resident, 5 system-prefix, 6 empty.
    void report_unload_watermark(std::uint32_t free_before, std::uint32_t free_after,
                                 std::uint32_t blocks_total, std::uint32_t blocks_unloadable,
                                 std::uint32_t pages_retired, const char* reason,
                                 const std::uint32_t* verdict_histogram = nullptr) const;
    void warm_cold_prefix(SequenceState& sequence, std::uint32_t end_page);
    // Restores one cold page into `physical` and releases the cold resources the
    // page no longer needs once it is hot again. Returns false only when no
    // staging slot could be found, in which case nothing was written into
    // `physical` and the caller must put the page back on the cold path (see the
    // long note at the definition).
    [[nodiscard]] bool restore_cold_page(SequenceState& sequence, std::uint32_t page,
                                         std::int32_t device_slot, std::int32_t file_slot,
                                         const DeviceKVPageHandle& physical);
    // Cold Host tier (--cold-policy host): retire read-free pages into the pinned
    // Host pool, and keep that pool's accounting honest when references go away.
    void enqueue_cold_host_evictions(SequenceState& sequence);
    void sweep_host_kv_pools() noexcept;
    void sweep_cold_host_tier() noexcept;
    void report_cold_host_tier(const char* tag);

    // ---- (6) per-round external recall: P5 trigger, P6 selection, P7 journal ----
    //
    // The four legs already exist and are REUSED, never re-implemented:
    //   P1 spill     enqueue_cold_compressions  (:10610, file_slot + cold_pages)
    //   P2 read-back restore_cold_page          (:11023)
    //   P3 prefetch  prefetch_cold_pages        (:11163)
    //   P4 key type  PrefixShortlistDigests::at (prefix_identity.h:42-57)
    // What is missing -- and what these functions add -- is the once-per-round trigger,
    // the page-selection rule, and a log that survives the process.
    //
    // The token-space half ("which span do I want back?") is NOT here: suffix_lookup or
    // an n-gram/BM25 index answers it and the engine injects the answer
    // (set_turn_recall_provider), so no second token index and no second identity
    // comparison is created.
    //
    // Enabled only by NINFER_TURN_RECALL=1 on a ColdPolicy::Disk run whose slot-bearing
    // codec is nvfp4/int8, so the default run is bit-identical to today: with the
    // journal closed, ensure_sequence_kv_mapped_for_round() is the old function.
    spec::turn_recall::RecallCodec turn_recall_codec_of_layers() const;
    [[nodiscard]] std::uint64_t turn_recall_page_bytes() const;
    // [CODECBLIND] The COMPLEMENT of `turn_recall_codec_of_layers()`: how many
    // cold-slot-bearing layers its two `if` arms could not reach, and the name of the first
    // such codec. Two walks rather than one shared walk on purpose -- rewriting the
    // classifier would change the bytes of the function every existing recall capture was
    // produced by, and a report-only change must not move them. Both are read ONCE, at
    // arming time; neither is on the per-round path.
    [[nodiscard]] std::uint32_t turn_recall_codec_blind_layers() const;
    [[nodiscard]] const char* turn_recall_codec_blind_codec() const;
    void record_turn_recall(SequenceState& sequence);
    [[nodiscard]] spec::turn_recall::RecallPagePlan plan_round_recall(const SequenceState& sequence);
    void recall_cold_pages_for_round(SequenceState& sequence,
                                     const spec::turn_recall::RecallPagePlan& plan);
    // The request the engine answers when no retrieval layer was injected: ONE page
    // that is both live in the journal and still cold, or nothing. Bounded by
    // construction -- a restored page leaves cold_pages and gets a tombstone, so the
    // same page is never asked for twice, and this can never become the per-round
    // whole-prefix recall the design forbids.
    [[nodiscard]] spec::turn_recall::RecallRequest recall_round_request(const SequenceState&, std::uint32_t frontier) const;
    // -----------------------------------------------------------------------
    // THE RETRIEVAL INDEX'S OWN ENTRY POINTS
    // -----------------------------------------------------------------------
    // Two entry points, ONE lookup. "What do I want back?" is answered by finding a recently-seen
    // token n-gram in the directory's catalogue column -- a content answer, not a positional one --
    // and both the injected provider and the built-in request funnel into `index_request_for_term`
    // so there is exactly one place the fan-out is decided.
    //
    // `index_query_term` is the QUERY BUILDER: it renders the newest tokens of the sequence's own
    // committed prefix into the byte alphabet `search_summaries` searches (via the ONE renderer,
    // `sum_dir::sum_dir_render_token_line`), so a query is expressed in the same alphabet the
    // producer wrote. It returns false when there is nothing to ask with (no directory, too short a
    // prefix), in which case the caller must fall back rather than ask an empty question.
    [[nodiscard]] bool index_query_term(const SequenceState&, std::uint32_t frontier,
                                       std::string& term) const;
    // The lookup + fan-out. `frontier` is the sequence's committed-token count: a page at or above it
    // is not yet legal to recall, and the blocks the query itself occupies are skipped so the search
    // cannot find the question instead of the answer.
    [[nodiscard]] spec::turn_recall::RecallRequest
    index_request_for_term(const std::string& term, std::uint32_t frontier) const;
    // How many pages ONE pass may return. Read from the engine's own knob
    // (`SumDirKnobs::max_blocks_per_pass`, env `NINFER_SUM_DIR_BLOCKS`, factory default
    // `kSumDirDefaultBlocksPerPass`) -- see the
    // definition in program_impl.h for why that number and not another.
    [[nodiscard]] std::uint32_t recall_fanout_blocks() const;
    // THE INJECTOR. Installs the directory-backed retrieval policy through the ONE seam that already
    // exists (`set_turn_recall_provider`, :993), with a snapshot of the query taken from this lane's
    // own committed prefix. It REFUSES to run when `turn_recall_provider_is_external` is set, so an
    // externally injected provider keeps winning exactly as that seam's contract says.
    void install_index_recall_provider(const SequenceState& sequence);
    void ensure_sequence_kv_mapped_for_round(SequenceState& sequence,
                                            std::uint32_t main_tokens,
                                            std::uint32_t backend_tokens = 0);
    void report_turn_recall(const char* tag);
    void set_turn_recall_provider(spec::turn_recall::RecallRequestProvider provider);

    std::unique_ptr<spec::turn_recall::TurnRecallJournal> turn_recall_journal;
    spec::turn_recall::RecallCodec turn_recall_codec = spec::turn_recall::RecallCodec::Rejected;
    std::uint64_t turn_recall_byte_budget = 256ULL << 20;
    // [PREFILLBUDGET] the BOUNDED PREFILL budget, in TOKENS of re-prefill per recall round.
    // Read once, from `NINFER_RECALL_PREFILL_TOKENS` (the flag `--recall-prefill-tokens` is
    // committed to that variable at parse time -- the same "the flag wins" shape `--ft-stats`
    // uses, options.cpp `set_process_env`), next to the byte budget it complements, and handed to
    // `plan_recall_pages` on EVERY plan. 0 == unset == today's behaviour, bit for bit.
    std::uint64_t turn_recall_prefill_token_budget = 0;
    spec::turn_recall::RecallRequestProvider turn_recall_provider;
    // Set ONLY by `set_turn_recall_provider`, i.e. only by a caller from outside this class. It is
    // what lets the engine install its own default retrieval policy per round (see
    // `install_index_recall_provider`) WITHOUT clobbering a retrieval layer that was injected: the
    // seam's contract is "an injected provider always wins", and a flag is the only way to honour
    // that contract while also having a working default. It is a separate flag rather than a null
    // check because the engine's own installer writes the same member.
    bool turn_recall_provider_is_external = false;
    // Reused across rounds so the steady state (nothing spilled, nothing restored)
    // allocates nothing and only walks the two ascending vectors once.
    std::vector<spec::turn_recall::RecallRecord> recall_scratch_records;
    struct TurnRecallCounters {
        std::uint64_t rounds             = 0;  // hooks run
        std::uint64_t records            = 0;  // Spill records appended
        std::uint64_t tombstones         = 0;  // Release records appended
        std::uint64_t append_failed      = 0;  // the log refused a record (retried next round)
        std::uint64_t digest_unavailable = 0;  // page spilled but its frontier has no digest
        std::uint64_t plans              = 0;  // non-empty plans
        std::uint64_t pages_restored     = 0;
        std::uint64_t pages_planned      = 0;
        // ---- [RESTOREUNITS] THE UNIT OF `pages_restored`, FIXED, AND THE SHORTFALL NAMED ------
        // `pages_restored` is a PAGE count now: it was a ROUND count on two of its three
        // increment sites (see spec/turn_recall_journal.h, `recall_restore_delta`). Fixing that
        // alone leaves the ratio readable in only one direction -- a run that planned 8 pages
        // and brought back 6 would print `pages=6/8` exactly as the OLD, WRONG numerator did --
        // so the fact the ratio cannot carry gets its own two fields, and BOTH are printed on
        // the sequence-end line beside the ratio:
        //   restored_short_rounds  rounds whose restored page count != their own plan's count
        //   restored_short_pages   the deficit over those rounds; 0 on a complete run
        std::uint64_t restored_short_rounds = 0;
        std::uint64_t restored_short_pages  = 0;
        std::uint64_t refused_cost       = 0;  // plans whose read did not beat re-prefill
        // ---- [INEXACTGATE D] THE TRUTH BESIDE `pages=R/P` ---------------------------------
        // `pages_planned` accumulates `plan.count()` -- the ALREADY-CUT plan -- so `pages=R/P`
        // reads HIGHER the more severely the run was cut: D1 (holed, 3x INEXACT) printed
        // `pages=5/5` while C5 (exact, 0 INEXACT) printed `pages=6/20` (dl/holegate sec.3.1).
        // `pages=R/P` is NOT redefined here; these four fields are added beside it so the truth
        // is on the same line, and each one names a different fact:
        //   pages_admitted  Σ of the pages each round COULD have held (the request's run, after
        //                 the frontier clamp) -- the denominator a reader believes `pages=R/P`
        //                 already has. It is NOT `pages_planned` and it is NOT the raw ask: the
        //                 part of the ask that the frontier cannot serve is `pages_clamped`.
        //   pages_cut     Σ of (dropped_hole + dropped_budget): pages that EXISTED and were not
        //                 held. Identity, asserted over a sweep in the sibling test:
        //                 pages_planned + pages_cut == pages_admitted.
        //   pages_clamped Σ dropped_clamped: the part of the ASK the frontier cannot serve.
        //                 Nothing that exists was dropped for it, so it is a separate axis and
        //                 must not be read as a loss. Second identity:
        //                 pages_admitted + pages_clamped == Σ wanted_pages().
        //   inexact_rounds  rounds whose plan was non-exact -- the countable form of the word
        //                 `INEXACT`, which until now existed only as a per-round suffix
        //   refused_empty   rounds REFUSED by an empty-but-non-exact plan, i.e. the refusals
        //                 that used to print NOTHING AT ALL (dl/holegate sec.2.3)
        //   cost_gate_evals how many times the arithmetic gate was CONSULTED. Without it,
        //                 `refused_cost=0` cannot be told from "no check ran" -- and with the
        //                 shipped rate defaults no check CAN fire (dl/holegate sec.2.5).
        std::uint64_t pages_admitted     = 0;
        std::uint64_t pages_cut          = 0;
        std::uint64_t pages_clamped      = 0;
        std::uint64_t inexact_rounds     = 0;
        std::uint64_t refused_empty      = 0;
        std::uint64_t cost_gate_evals    = 0;
        // THE TWO REFUSALS, NAMED. Both are the case "the user set
        // NINFER_TURN_RECALL=1 and NO recall happened", and until they existed that case
        // was reported by ONE stderr line whose text did not say recall was off, with no
        // counter, no summary-line trace and rc=0 -- so "no error" read as "it worked".
        //   refused_codec  -- the stack-level classifier refused the stack's codec
        //                     (program_impl.h:~1071); it refuses EVERY spilled page of
        //                     that stack, not only the two-codec ones.
        //   refused_medium -- --cold-policy cannot spill at all, so the whole recall
        //                     block never ran; this case used to print NOTHING at all.
        std::uint64_t refused_codec      = 0;
        std::uint64_t refused_medium     = 0;
        //   refused_prefill_tok -- [PREFILLBUDGET] rounds refused by the TOKEN budget
        //                     (`refused-prefill-budget`). It is a SUBSET of `refused_empty`
        //                     (both are counted: the empty-but-non-exact refusal is the parent
        //                     fact, this is the cause), and it exists for the same reason
        //                     `cost_gate_evals` does -- so that "the budget never fired" can be
        //                     told apart from "the budget was never consulted". Without it, a run
        //                     with `refused_prefill_tok=0` cannot say which of those two it is.
        std::uint64_t refused_prefill_tok = 0;
        //   codec_blind_layers -- [CODECBLIND] cold-slot-bearing layers whose dtype the
        //                     stack-level classifier cannot name. It is an ASSIGNMENT, not an
        //                     increment: the arming block runs once per run and the fact is a
        //                     property of the RESOLVED per-layer table, not an event. Before
        //                     this field the case left NO trace at all whenever an admitted
        //                     codec was present beside the blind layers -- the run looked
        //                     clean, the chain fired, and the record's stride spanned two
        //                     codecs while the log named one.
        std::uint64_t codec_blind_layers = 0;
    } turn_recall_counters;
    // Set when the refused-admission summary has been emitted, so the line appears once
    // per run instead of once per report_turn_recall() call site.
    bool turn_recall_refusal_reported = false;

    // On-demand graph capture state (see DecodeGraphFamily comment).
    std::uint32_t graph_capture_ceiling = 16;
    void extend_ordinary_graphs(std::uint32_t batch_size, std::uint32_t frontier);
    // On-demand capture for one rung of the MTP ladder: exactly the ordinary machinery, plus the
    // MTP cache's own transient address row, because an MTP round touches both caches. Only ever
    // fills segments of a topology class that the startup capture already instantiated.
    void extend_mtp_graphs(std::uint32_t batch_size, std::uint32_t frontier, std::uint32_t width);
    // mtplogx: the two cumulative quantities behind the routine above. Read-only; a consumer
    // differences two snapshots.
    [[nodiscard]] std::uint64_t mtp_graph_extension_calls() const noexcept {
        return mtp_graph_extension_calls_;
    }
    [[nodiscard]] std::uint64_t mtp_graph_extension_nanoseconds() const noexcept {
        return mtp_graph_extension_nanoseconds_;
    }
    [[nodiscard]] bool mtp_graph_missing(std::uint32_t batch_size, std::uint32_t frontier,
                                         std::uint32_t width) const noexcept;
    schedule::ExecutionCore make_execution_core();
    std::size_t vision_handoff_peak_bytes    = 0;

private:
    void advance_resource_revision() noexcept {
        if (++resource_revision_ == 0) { ++resource_revision_; }
    }

    // mtplogx: the counters exposed by mtp_graph_extension_calls() and
    // mtp_graph_extension_nanoseconds() above.
    std::uint64_t mtp_graph_extension_calls_       = 0;
    std::uint64_t mtp_graph_extension_nanoseconds_ = 0;
    std::uint64_t resource_revision_            = 1;
    std::uint32_t pressure_planning_generation_ = 0;
    bool pressure_planning_active_              = false;

    struct PressurePageScratchSlot {
        std::uint32_t generation     = 0;
        std::uint32_t selected_index = std::numeric_limits<std::uint32_t>::max();
        std::uint64_t host_group     = 0;
        bool projected               = false;
        bool device                  = false;
        bool host                    = false;
        bool pressure_targeted       = false;
    };

    struct PressureSelectedPage {
        LogicalKVPageHandle page;
        std::uint32_t references = 0;
    };

    mutable std::uint32_t pressure_page_scratch_generation_ = 0;
    mutable std::vector<PressurePageScratchSlot> pressure_text_page_scratch_;
    mutable std::vector<PressurePageScratchSlot> pressure_backend_page_scratch_;
    mutable std::vector<PressureSelectedPage> pressure_text_selected_pages_;
    mutable std::vector<PressureSelectedPage> pressure_backend_selected_pages_;
    mutable std::vector<std::uint8_t> pressure_private_owner_scratch_;
    mutable std::vector<std::uint8_t> pressure_shared_owner_scratch_;
    mutable std::vector<std::vector<runtime::CheckpointRef>> pressure_private_drop_scratch_;
    mutable std::vector<StateImageHandle> pressure_state_scratch_;

    void begin_pressure_page_scratch() const noexcept;
    [[nodiscard]] PressurePageScratchSlot& pressure_page_scratch(const LogicalKVPageStore& store,
                                                                 LogicalKVPageHandle page) const;
    [[nodiscard]] const PressurePageScratchSlot*
    find_pressure_page_scratch(const LogicalKVPageStore& store, LogicalKVPageHandle page) const;

    struct MaterializationSourceProtection {
        struct StateOwnershipCandidate {
            StateImageHandle state;
            std::uint32_t source_checkpoint_references = 0;
        };

        std::optional<std::uint32_t> private_source_index;
        bool consumed_private_source = false;
        std::optional<StateImageHandle> state;
        std::uint32_t consumed_state_references = 0;
        bool state_fork_required                = false;
        std::vector<StateOwnershipCandidate> state_ownership_candidates;
        std::optional<KVAddressSpaceHandle> text;
        std::uint32_t text_pages          = 0;
        std::uint32_t text_transfer_pages = 0;
        bool text_prefix_fork_required    = false;
        std::optional<KVAddressSpaceHandle> backend;
        std::uint32_t backend_pages          = 0;
        std::uint32_t backend_transfer_pages = 0;
        bool backend_prefix_fork_required    = false;
    };

    struct PendingTransaction {
        std::uint64_t id = 0;
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<std::uint64_t, kMaximumConcurrency> epochs{};
        std::size_t size = 0;
    };

    std::optional<PendingTransaction> pending_transaction_;
    std::uint64_t next_transaction_id_ = 1;

    struct MaterializationTransaction {
        struct KVRestorePage {
            LogicalKVPageHandle logical;
            HostKVExtentCapability extent;
            std::uint32_t extent_page = 0;
        };

        struct PressureWork {
            struct StateChangeWork {
                std::optional<StateImageTransfer> transfer;
                bool host_released = false;
            };

            struct KVChangeWork {
                std::vector<LogicalKVPageHandle> pages;
                std::vector<DeviceKVPageHandle> sources;
                std::optional<HostKVExtentReservation> backup;
                bool host_released = false;
            };

            qwen3_6::detail::PressureDecision option;
            std::uint32_t continuation_index      = 0;
            std::uint64_t continuation_generation = 0;
            bool shared_owner                     = false;
            std::vector<StateChangeWork> state_changes;
            std::vector<KVChangeWork> main_kv_changes;
            std::vector<KVChangeWork> backend_kv_changes;
            detail::PhysicalDelta committed_delta;
            bool submitted                 = false;
            bool completed                 = false;
            bool checkpoint_drop_published = false;
            bool mutation_published        = false;
            std::uint64_t spill_pages      = 0;
        };

        std::uint64_t id = 0;
        runtime::LaneId destination;
        bool has_source                              = false;
        bool has_shared_source                       = false;
        runtime::ClaimDisposition source_disposition = runtime::ClaimDisposition::ConsumedToActive;
        std::uint32_t source_index                   = 0;
        std::uint64_t source_generation              = 0;
        std::uint32_t shared_source_index            = 0;
        std::uint64_t shared_source_generation       = 0;
        std::optional<MaterializationSourceResult> source_result;
        std::optional<MaterializationSharedSourceResult> shared_source_result;
        std::vector<std::uint32_t> victim_indices;
        std::vector<std::uint64_t> victim_generations;
        std::vector<bool> victim_released;
        std::vector<PressureWork> pressure;
        std::vector<MaterializationVictimResult> pressure_results;
        std::size_t pressure_cursor = 0;
        std::size_t victim_count    = 0;
        std::vector<std::uint32_t> shared_victim_indices;
        std::vector<std::uint64_t> shared_victim_generations;
        std::vector<bool> shared_victim_released;
        std::vector<MaterializationSharedVictimResult> shared_pressure_results;
        std::vector<PressureWork> shared_pressure;
        std::size_t shared_pressure_cursor    = 0;
        std::size_t shared_victim_count       = 0;
        bool pressure_host_releases_published = false;
        bool pressure_copies_prepared         = false;
        bool pressure_copies_submitted        = false;
        bool pressure_copies_published        = false;
        std::array<TransferWork, 3> pressure_transfer_work{};
        std::array<std::uint32_t, 3> pressure_transfer_pages{};
        std::uint64_t pressure_state_images = 0;
        std::uint8_t pressure_timer_mask    = 0;
        std::optional<AdmissionCandidate> plan;
        std::optional<std::uint32_t> root_continuation_index;
        bool root_waiting_for_victim = false;
        std::array<StateImageHandle, 2> reserved_states{};
        std::size_t reserved_state_count = 0;
        std::optional<StateImageHandle> state_fork_destination;
        std::optional<KVAddressSpaceHandle> root_text_address;
        std::optional<KVAddressSpaceHandle> root_backend_address;
        std::optional<KVActivationReservation> text_activation;
        std::optional<KVActivationReservation> backend_activation;
        std::optional<DeviceKVPageReservation> text_source_restore_reservation;
        std::optional<DeviceKVPageReservation> backend_source_restore_reservation;
        std::optional<KVPrefixForkReservation> text_prefix_fork;
        std::optional<KVPrefixForkReservation> backend_prefix_fork;
        std::optional<LogicalKVPageHandle> text_retained_tail;
        std::optional<LogicalKVPageHandle> backend_retained_tail;
        std::optional<HostKVExtentReservation> text_retained_tail_backup;
        std::optional<HostKVExtentReservation> backend_retained_tail_backup;
        std::optional<std::uint32_t> text_activation_frontier;
        std::optional<std::uint32_t> backend_activation_frontier;
        std::optional<StateImageTransfer> state_restore;
        bool split_state_identity = false;
        std::vector<KVRestorePage> text_restores;
        std::vector<DeviceKVPageHandle> text_restore_destinations;
        std::vector<KVRestorePage> backend_restores;
        std::vector<DeviceKVPageHandle> backend_restore_destinations;
        std::vector<runtime::ContextTransferObservation> transfer_observations;
        runtime::ContextOperationCounts operations;
        bool state_restored                 = false;
        bool transfer_submitted             = false;
        std::uint8_t transfer_timer_mask    = 0;
        bool prefix_tail_submitted          = false;
        bool retained_tail_backup_submitted = false;
        bool prefix_forks_ready             = false;
        bool source_prepared                = false;
        bool cancel_pending                 = false;
        bool prepared                       = false;
        bool terminal                       = false;
    };

    std::uint64_t next_materialization_id_ = 1;
    CudaCompletionEvent context_source_ready_;
    CudaCompletionEvent context_completion_;
    std::vector<TokenId> materialization_ledger_;
    qwen3_6::detail::ResidentPrefixIdentity materialization_identity_;
    qwen3_6::detail::PrefixShortlistDigests materialization_prefix_digests_;

    struct ActiveCaptureTransaction {
        std::uint64_t id         = 0;
        std::uint32_t lane       = 0;
        std::uint64_t lane_epoch = 0;
        CaptureGroup group;
        bool publish_private = false;
        bool publish_shared  = false;
        bool replaces_shared = false;
        std::optional<runtime::CheckpointRef> private_replacement;
        std::optional<std::uint32_t> shared_index;
        std::uint64_t replacement_generation = 0;
        StateImageHandle source_state;
        StateImageHandle destination_state;
        qwen3_6::CaptureStatePlacement state_placement = qwen3_6::CaptureStatePlacement::DeviceFork;
        std::optional<StateImageTransfer> state_snapshot;
        std::optional<KVAddressSpaceHandle> active_text_destination;
        std::optional<KVAddressSpaceHandle> active_backend_destination;
        std::optional<KVActiveSnapshotReservation> text_snapshot;
        std::optional<KVActiveSnapshotReservation> backend_snapshot;
        detail::PhysicalDelta resource_delta;
        detail::PhysicalDelta active_entitlement_delta;
        detail::PhysicalResources capacity_preparation_removed;
        ContinuationSummary active_summary;
        std::vector<runtime::ContextTransferRequirement> transfer_requirements;
        std::vector<runtime::ContextTransferObservation> transfer_observations;
        runtime::ContextOperationCounts operations;
        std::vector<std::uint32_t> victim_indices;
        std::vector<std::uint64_t> victim_generations;
        std::vector<MaterializationTransaction::PressureWork> pressure;
        std::vector<MaterializationVictimResult> pressure_results;
        std::vector<std::uint32_t> shared_victim_indices;
        std::vector<std::uint64_t> shared_victim_generations;
        std::vector<MaterializationTransaction::PressureWork> shared_pressure;
        std::vector<MaterializationSharedVictimResult> shared_pressure_results;
        bool pressure_host_releases_published = false;
        bool pressure_copies_prepared         = false;
        bool pressure_copies_submitted        = false;
        bool pressure_copies_published        = false;
        std::array<TransferWork, 3> pressure_transfer_work{};
        std::array<std::uint32_t, 3> pressure_transfer_pages{};
        std::uint64_t pressure_state_images = 0;
        std::uint8_t pressure_timer_mask    = 0;
        bool pressure_committed             = false;
        bool recycles_private_state         = false;
        bool replacement_removed            = false;
        bool prepared                       = false;
        std::uint64_t recycled_state_epoch  = 0;
        bool transfer_enqueue_pending       = false;
        bool transfer_submitted             = false;
        std::uint8_t transfer_timer_mask    = 0;
        bool published                      = false;
    };

    std::uint64_t next_capture_offer_id_ = 1;

    using ContextTransaction =
        std::variant<std::monostate, MaterializationTransaction, ActiveCaptureTransaction>;
    ContextTransaction context_transaction_;

    [[nodiscard]] MaterializationResult
    progress_materialization_transaction(runtime::CancellationFlagView cancellation);
    [[nodiscard]] ActiveCaptureResult
    progress_active_capture_transaction(runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture_impl(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                                const SharedPrefixHandle* replacement,
                                std::optional<runtime::CheckpointRef> private_replacement,
                                bool permit_shared_publication,
                                std::optional<AdmissionCandidate> pressure,
                                const runtime::ContextMachineCostModel& machine_cost,
                                runtime::CancellationFlagView cancellation);

    std::array<CudaEventTimer, 3> context_transfer_timers_;

    [[nodiscard]] std::optional<AdmissionCandidate>
    inspect_lane(std::uint32_t lane, const PreparedPromptData& prompt, const RequestBasePlan& base,
                 const SequenceState* source, const SharedPrefixState* shared_source,
                 std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source);
    [[nodiscard]] StartResult start_request(MaterializationTransaction& transaction);
    void prepare_materialization(MaterializationTransaction& transaction);
    void enqueue_materialization_transfers(MaterializationTransaction& transaction);
    void record_materialization_transfer_observations(MaterializationTransaction& transaction);
    void publish_materialization_transfers(MaterializationTransaction& transaction);
    void prepare_prefix_forks(MaterializationTransaction& transaction);
    void prepare_consumed_source(MaterializationTransaction& transaction);
    void abort_materialization_transfers(MaterializationTransaction& transaction) noexcept;
    void prepare_pressure_bookkeeping(MaterializationTransaction::PressureWork& work);
    void prepare_pressure_work(MaterializationTransaction::PressureWork& work,
                               runtime::ContextResourceClass resource);
    void publish_pressure_host_releases(MaterializationTransaction::PressureWork& work);
    void publish_pressure_work(MaterializationTransaction::PressureWork& work) noexcept;
    void abort_pressure_work(MaterializationTransaction::PressureWork& work) noexcept;
    void start_context_transfer_timer(runtime::ContextResourceClass resource);
    void stop_context_transfer_timer(runtime::ContextResourceClass resource);
    [[nodiscard]] runtime::ContextTransferObservation context_transfer_observation(
        runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
        TransferWork work, std::uint32_t page_count = 0, std::uint64_t state_images = 1) const;

    struct PhysicalReleaseResult {
        runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
        detail::PhysicalDelta delta;
    };

    [[nodiscard]] PhysicalReleaseResult
    release_materialization_victim(MaterializationTransaction& transaction,
                                   std::size_t position) noexcept;
    void start_sequence(std::uint32_t lane, SequenceState& sequence,
                        MaterializationTransaction& transaction);
    void release_materialization_staging(MaterializationTransaction& transaction) noexcept;
    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill_raw(std::uint32_t lane, runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_raw(std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
               runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming
    resolve_prefill_raw(std::uint32_t lane, bool terminal, runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming resolve_pending_raw(
        std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
        std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
        runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] bool valid_sequence(SequenceHandle handle) const noexcept;
    [[nodiscard]] bool valid_continuation(const ContinuationHandle& handle) const noexcept;
    [[nodiscard]] bool valid_shared_prefix(const SharedPrefixHandle& handle) const noexcept;
    [[nodiscard]] bool valid_capture_offer(const CaptureOffer& offer) const noexcept;
    [[nodiscard]] bool materialization_pins(std::uint32_t index,
                                            std::uint64_t generation) const noexcept;
    [[nodiscard]] bool has_unsettled_state_fork() const noexcept;
    [[nodiscard]] bool valid_pending(const PendingBatch& pending) const noexcept;
    [[nodiscard]] detail::PhysicalResources
    resident_resources(const SequenceState& sequence) const noexcept;
    [[nodiscard]] detail::PhysicalResources
    resident_resources(const SharedPrefixState& shared) const noexcept;
    [[nodiscard]] detail::PhysicalResources physical_occupancy() const noexcept;
    [[nodiscard]] bool physical_peak_fits(detail::PhysicalResources peak) const noexcept;
    [[nodiscard]] StateImageHandle
    selected_state(const SequenceState& sequence, ReusePath reuse,
                   std::optional<runtime::CheckpointRef> checkpoint) const;
    [[nodiscard]] std::uint32_t
    selected_state_consumed_references(const SequenceState& sequence, ReusePath reuse,
                                       RewriteCheckpointDisposition rewrite_disposition,
                                       std::optional<runtime::CheckpointRef> checkpoint,
                                       std::uint32_t reuse_base) const;
    [[nodiscard]] bool
    selected_state_requires_fork(const SequenceState& sequence, ReusePath reuse,
                                 RewriteCheckpointDisposition rewrite_disposition,
                                 std::optional<runtime::CheckpointRef> checkpoint,
                                 std::uint32_t reuse_base) const;
    [[nodiscard]] bool can_retain_rewrite_checkpoint(const PreparedPromptData& prompt,
                                                     const RewriteCheckpointSpec& desired,
                                                     const SequenceState& sequence, ReusePath reuse,
                                                     std::uint32_t reuse_base) const;
    [[nodiscard]] std::uint32_t device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                       KVAddressSpaceHandle address,
                                                       std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t shared_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                       KVAddressSpaceHandle address,
                                                       std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                              KVAddressSpaceHandle address,
                                                              std::uint32_t frontier) const;
    [[nodiscard]] bool partial_tail_cow_required(const KVAddressSpaceStore& addresses,
                                                 KVAddressSpaceHandle address,
                                                 std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t
    missing_shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                          KVAddressSpaceHandle address,
                                          std::uint32_t frontier) const;
    [[nodiscard]] std::size_t host_kv_prefix_bytes(const KVAddressSpaceStore& addresses,
                                                   KVAddressSpaceHandle address,
                                                   std::uint32_t frontier) const noexcept;
    [[nodiscard]] qwen3_6::CheckpointSummary
    checkpoint_summary(const SequenceState& sequence, runtime::CheckpointRef checkpoint,
                       StateImageHandle state, runtime::PrefillWork rebuild_work) const;
    [[nodiscard]] qwen3_6::ContinuationSummary
    continuation_summary(const SequenceState& sequence) const;
    void populate_continuation_summary(const SequenceState& sequence,
                                       qwen3_6::ContinuationSummary& summary) const;
    [[nodiscard]] qwen3_6::SharedPrefixSummary
    shared_prefix_summary(const SharedPrefixState& shared) const;
    [[nodiscard]] std::optional<MaterializationSourceProtection>
    materialization_source_protection(const AdmissionCandidateImpl& admission) const;
    [[nodiscard]] detail::PhysicalResources
    materialization_deficit(const AdmissionCandidateImpl& admission) const;
    [[nodiscard]] detail::PhysicalResources
    guided_materialization_deficit(const AdmissionCandidateImpl& admission,
                                   const detail::PhysicalDelta& pressure) const;
    [[nodiscard]] bool
    protected_materialization_page(const MaterializationSourceProtection* protection,
                                   const KVAddressSpaceStore& addresses, std::uint32_t page_offset,
                                   LogicalKVPageHandle page, bool backend) const;
    [[nodiscard]] std::optional<qwen3_6::detail::PressureDecision>
    inspect_pressure_option(const SequenceState& sequence, detail::PhysicalResources deficit,
                            const MaterializationSourceProtection* protection           = nullptr,
                            const qwen3_6::TargetKVRequirement* retained_requirement    = nullptr,
                            std::span<const runtime::CheckpointRef> dropped_checkpoints = {},
                            std::span<const StateImageHandle> released_states           = {},
                            const qwen3_6::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::vector<qwen3_6::detail::PressureDecision>
    inspect_pressure_successors(const SequenceState& sequence, detail::PhysicalResources residual,
                                const MaterializationSourceProtection* protection,
                                const qwen3_6::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::vector<qwen3_6::detail::PressureDecision> inspect_shared_pressure_successors(
        const SharedPrefixState& shared, detail::PhysicalResources residual,
        const MaterializationSourceProtection* protection,
        const qwen3_6::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::optional<qwen3_6::detail::PressureDecision> inspect_shared_pressure_option(
        const SharedPrefixState& shared, detail::PhysicalResources deficit,
        const MaterializationSourceProtection* protection = nullptr,
        const qwen3_6::detail::PressureDecision* current  = nullptr) const;
    [[nodiscard]] std::vector<qwen3_6::detail::PressureDecision> inspect_shared_pressure_options(
        const SharedPrefixState& shared, detail::PhysicalResources deficit,
        const MaterializationSourceProtection* protection = nullptr,
        const qwen3_6::detail::PressureDecision* current  = nullptr) const;
    [[nodiscard]] qwen3_6::detail::PressureDecision
    inspect_eviction_option(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::detail::PressureDecision
    inspect_shared_eviction_option(const SharedPrefixState& shared) const;
    [[nodiscard]] std::optional<qwen3_6::detail::PressureDecision>
    inspect_checkpoint_drop_option(const SequenceState& sequence,
                                   std::span<const runtime::CheckpointRef> checkpoints) const;
    [[nodiscard]] bool
    pressure_decision_valid(const SequenceState& sequence,
                            const qwen3_6::detail::PressureDecision& decision,
                            const MaterializationSourceProtection* protection) const;
    [[nodiscard]] bool
    shared_pressure_decision_valid(const SharedPrefixState& shared,
                                   const qwen3_6::detail::PressureDecision& decision,
                                   const MaterializationSourceProtection* protection) const;
    [[nodiscard]] std::vector<runtime::ContextTransferRequirement>
    checkpoint_restore_requirements(const SequenceKVBundle& kv,
                                    const qwen3_6::TargetKVRequirement& requirement,
                                    StateImageHandle state) const;
    [[nodiscard]] bool pressure_checkpoint_recovery_impacts(
        const AdmissionCandidateImpl& candidate,
        std::span<const ContinuationHandle* const> private_owners,
        std::span<const qwen3_6::detail::PressureDecision* const> private_decisions,
        std::span<const std::uint32_t> private_ordinals,
        std::span<const SharedPrefixHandle* const> shared_owners,
        std::span<const qwen3_6::detail::PressureDecision* const> shared_decisions,
        std::span<const std::uint32_t> shared_ordinals,
        std::span<const qwen3_6::detail::PressureBaselineRecovery> baseline_recovery,
        const runtime::ContextMachineCostModel& machine_cost,
        std::vector<runtime::PressureCheckpointRecoveryImpact>& output,
        std::uint64_t& projection_work) const;
    void publish_checkpoint_drop(SequenceState& sequence, runtime::CheckpointRef checkpoint);
    [[nodiscard]] PrefillProgress wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult step);
    [[nodiscard]] PendingBatch wrap_pending(std::span<const std::uint32_t> lanes,
                                            const runtime::BatchedGeneratedRound& round);
    void invalidate_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] SequenceState& active_sequence(std::uint32_t lane);
    [[nodiscard]] const SequenceState& active_sequence(std::uint32_t lane) const;
    [[nodiscard]] std::optional<std::uint32_t> allocate_continuation_slot() noexcept;
    void release_continuation_slot(std::uint32_t index) noexcept;
    void clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept;
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    [[nodiscard]] StateImageSelectors state_selectors(const SequenceState& sequence) const;
    [[nodiscard]] std::uint32_t state_footprint(const SequenceState& sequence) const noexcept;
    [[nodiscard]] std::uint32_t owned_checkpoint_references(const SequenceState& sequence,
                                                            StateImageHandle state) const noexcept;
    [[nodiscard]] bool state_exclusive_to_sequence(const SequenceState& sequence,
                                                   StateImageHandle state) const noexcept;
    [[nodiscard]] std::optional<AdmissionCandidate> compose_materialization(
        AdmissionCandidate&& admission, std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision* const> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision* const> shared_pressure_options);
    [[nodiscard]] std::optional<detail::PhysicalPressureEffect> combined_pressure_effect(
        const MaterializationSourceProtection* protection,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision> shared_pressure_options,
        std::vector<HostKVPageReplicaRelease>* released_host_pages) const;
    void refresh_state_views(SequenceState& sequence);
    void reserve_state_entitlement(SequenceState& sequence, std::uint32_t slots);
    void settle_state_fork(SequenceState& sequence);
    [[nodiscard]] detail::PhysicalResources
    release_checkpoint_reference(StateImageHandle checkpoint) noexcept;
    [[nodiscard]] detail::PhysicalResources
    release_shared_prefix_state(std::uint32_t index, SharedPrefixSlotRole expected_role);
    [[nodiscard]] detail::PhysicalResources
    install_private_capture(SequenceState& sequence, const CaptureGroup& group,
                            StateImageHandle checkpoint,
                            std::optional<runtime::CheckpointRef> replacement);
    void prepare_active_capture(ActiveCaptureTransaction& transaction);
    void enqueue_active_capture_transfers(ActiveCaptureTransaction& transaction);
    void abort_active_capture(ActiveCaptureTransaction& transaction) noexcept;
    [[nodiscard]] ActiveCaptureResult publish_active_capture(ActiveCaptureTransaction& transaction);
    void release_active_shared_references(SequenceState& sequence) noexcept;
    void release_sequence_state(SequenceState& sequence) noexcept;
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    [[nodiscard]] runtime::ExecutionTiming
    resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                    std::uint32_t accepted_tokens, bool terminal,
                                    runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill(SequenceState& sequence, RequestControl& request,
                    runtime::ExecutionTiming* failed_timing);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets,
                          runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets,
                     runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash2_batch(std::span<const std::uint32_t> lanes,
                         std::span<const runtime::RoundBudget> budgets,
                         runtime::ExecutionTiming* failed_timing);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    // THE MISSING GUARANTEE of the coverage invariant that
    // KVAddressSpaceStore::ensure_mapped_to_tokens() only ever CHECKS
    // (logical_kv_store.h:1572-1586): "the entitlement still covers the frontier this
    // call is about to reach". The check exists and is a real tripwire; nothing used to
    // raise the entitlement as the frontier moved, so the mapping walked into the wall
    // one chunk after install and threw. This raises it, and only ever UP.
    //
    // It is called from ensure_sequence_kv_mapped() -- the ONE funnel every
    // frontier-advancing mapping site goes through (install :10041, the prefill chunk
    // loop :13304, the per-round hook :13550 / :13818 / :14016 / :14279, :13074) -- so
    // the guarantee holds at the same instant at every site instead of being restated
    // per site and drifting between them.
    //
    // [M21-ALLOW] A NO-OP whenever the plan's whole span already covers the target --
    // which is every call on the default (watermark OFF) path EXCEPT one that follows a
    // runtime context append. The old note said the default path "cannot be reached by
    // this function at all", and that was only true while nothing could push the runtime
    // frontier past the plan-time span: the plan's span is
    // pages_for_tokens(prompt_tokens + effective_output_tokens - 1), and
    // append_context_prefill() is the input that leaves it. See the body for the
    // reading and for why the arithmetic below is unchanged.
    void grow_sequence_kv_entitlement_to(SequenceState& sequence, std::uint32_t main_tokens,
                                         std::uint32_t backend_tokens);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void ensure_sequence_kv_mapped(SequenceState& sequence, std::uint32_t main_tokens,
                                   std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    void release_sequence_kv(SequenceState& sequence) noexcept;
    // w-find4 G9: retract this sequence's standing L0 records (one Release per live one,
    // reason = SequenceEnd) and forget them. MUST run before release_sequence_kv() hands the
    // device cold slot and the process-global spill-file slot back to their pools.
    void release_sequence_recall_records(SequenceState& sequence) noexcept;
    void commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                            std::uint32_t backend_tokens = 0);
    [[nodiscard]] qwen3_6::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_6::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_6::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT> {
    using Core               = NINFER_QWEN36_RUNTIME_NS::ProgramImplCore;
    using AdmissionCandidate = qwen3_6::AdmissionCandidate<NINFER_QWEN36_VARIANT>;
    using ContinuationHandle = qwen3_6::ContinuationHandle<NINFER_QWEN36_VARIANT>;
    using SharedPrefixHandle = qwen3_6::SharedPrefixHandle<NINFER_QWEN36_VARIANT>;

    struct Owner {
        const ContinuationHandle* private_handle = nullptr;
        const SharedPrefixHandle* shared_handle  = nullptr;
        std::uint32_t ordinal                    = 0;
        bool shared                              = false;
    };

    struct TargetNode {
        std::uint32_t candidate_index = 0;
        std::vector<std::uint16_t> owner_choices;
        std::optional<detail::PhysicalResources> assessed_residual;
        std::uint32_t stable_ordinal = 0;
        bool root_maximal            = false;
    };

    enum class OwnerParticipation : std::uint8_t {
        MaterializationSource,
        PressureEligible,
    };

    struct CandidateOwnerOptions {
        OwnerParticipation participation = OwnerParticipation::MaterializationSource;
        std::vector<PressureDecision> decisions;
        std::uint16_t eviction_choice = 0;
    };

    struct CandidateOptions {
        std::vector<CandidateOwnerOptions> owners;
        bool populated = false;
    };

    struct PreparedOwnerDecision {
        std::uint32_t candidate_index = 0;
        std::uint32_t owner_index     = 0;
        std::uint16_t choice          = 0;
        PressureDecision decision;
    };

    PressurePlanningSessionImpl(Core& owner, const runtime::ContextMachineCostModel& cost,
                                std::span<const AdmissionCandidate* const> admission_candidates,
                                std::span<const ContinuationHandle* const> private_owners,
                                std::span<const std::uint32_t> private_owner_ordinals,
                                std::span<const SharedPrefixHandle* const> shared_owners,
                                std::span<const std::uint32_t> shared_owner_ordinals);
    ~PressurePlanningSessionImpl() noexcept;

    [[nodiscard]] qwen3_6::PressureTargetHandle
    identity_target(const AdmissionCandidate& candidate) const;
    [[nodiscard]] qwen3_6::PressureTargetHandle
    root_maximal_target(const AdmissionCandidate& root_candidate);
    [[nodiscard]] std::optional<qwen3_6::PressureTargetHandle>
    guided_closure_target(const AdmissionCandidate& candidate,
                          std::span<const std::uint32_t> preferred_owner_ordinals);
    [[nodiscard]] runtime::PressureTargetGuidance guidance(qwen3_6::PressureTargetHandle target);
    [[nodiscard]] runtime::PressureTargetAssessment assess(qwen3_6::PressureTargetHandle target);
    void retain_assessment(qwen3_6::PressureTargetHandle target);
    [[nodiscard]] qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>
    prepare_expansion(qwen3_6::PressureTargetHandle parent);
    [[nodiscard]] qwen3_6::PressureExpansionView
    commit_expansion(qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>&& prepared);
    void discard_expansion(
        qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>&& prepared) noexcept;
    [[nodiscard]] std::optional<AdmissionCandidate>
    seal(qwen3_6::PressureTargetHandle target,
         const NINFER_QWEN36_RUNTIME_NS::PreparedPromptData& prompt);
    [[nodiscard]] std::optional<AdmissionCandidate>
    seal_capture(qwen3_6::PressureTargetHandle target);

    [[nodiscard]] bool valid(qwen3_6::PressureTargetHandle target) const noexcept;
    [[nodiscard]] std::uint32_t candidate_index(const AdmissionCandidate& candidate) const;
    [[nodiscard]] TargetNode* find_target(const TargetNode& target) noexcept;
    [[nodiscard]] const TargetNode* find_target(const TargetNode& target) const noexcept;
    void index_target(std::uint32_t target_index);
    void populate_options(std::uint32_t candidate_index);
    [[nodiscard]] std::vector<PressureDecision>
    pressure_successors(const CandidateOwnerOptions& owner_options, std::size_t owner_index,
                        const detail::PhysicalResources& residual,
                        const typename Core::MaterializationSourceProtection& protection,
                        const PressureDecision* current) const;

    Core* program                                        = nullptr;
    const runtime::ContextMachineCostModel* machine_cost = nullptr;
    std::uint64_t resource_revision                      = 0;
    std::uint32_t generation                             = 1;
    std::uint32_t scratch_generation                     = 1;
    std::vector<const AdmissionCandidate*> candidates;
    std::vector<Owner> owners;
    std::vector<CandidateOptions> candidate_options;
    std::vector<TargetNode> targets;
    std::vector<std::uint32_t> target_hash_table;
    std::vector<TargetNode> expansion_scratch;
    std::vector<PreparedOwnerDecision> prepared_owner_decisions;
    std::vector<qwen3_6::PressureTargetHandle> committed_children;
    std::vector<const ContinuationHandle*> selected_private_owners;
    std::vector<const PressureDecision*> selected_private_decisions;
    std::vector<const SharedPrefixHandle*> selected_shared_owners;
    std::vector<const PressureDecision*> selected_shared_decisions;
    std::vector<const ContinuationHandle*> recovery_private_owners;
    std::vector<const PressureDecision*> recovery_private_decisions;
    std::vector<std::uint32_t> recovery_private_ordinals;
    std::vector<const SharedPrefixHandle*> recovery_shared_owners;
    std::vector<const PressureDecision*> recovery_shared_decisions;
    std::vector<std::uint32_t> recovery_shared_ordinals;
    std::vector<PressureBaselineRecovery> baseline_recovery;
    std::vector<runtime::PressureOwnerOutcome> assessment_outcomes;
    std::vector<runtime::PressureCheckpointRecoveryImpact> assessment_impacts;
    std::vector<runtime::PressureOwnerOutcome> guidance_outcomes;
    std::optional<AdmissionCandidate> latest_projection;
    std::optional<AdmissionCandidate> retained_projection;
    std::uint32_t latest_projection_target   = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t retained_projection_target = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t prepared_new_count         = 0;
    bool scratch_live                        = false;
};

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
