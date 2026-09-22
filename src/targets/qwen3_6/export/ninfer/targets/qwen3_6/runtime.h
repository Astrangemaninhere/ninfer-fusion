#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::runtime {
struct ContextMachineCostModel;
}

namespace ninfer::targets::qwen3_6 {

namespace detail {
struct CaptureAssessmentImpl;

} // namespace detail

// Read-only diagnostics sampled from the real Program stores.  This is not an accounting input.
struct PhysicalUsageSnapshot {
    std::uint64_t resource_revision       = 0;
    std::uint32_t device_state_slots      = 0;
    std::uint32_t host_state_slots        = 0;
    std::uint32_t device_main_kv_pages    = 0;
    std::uint32_t device_backend_kv_pages = 0;
    std::size_t host_kv_bytes             = 0;
    // Dedicated checkpoint pool, when the Program has one; excludes active recurrent slots.
    // APPENDED; donor: igorls/ninfer @ 5e4a66d, same file, :37-39.
    std::optional<std::uint32_t> checkpoint_slots_occupied;
    std::optional<std::uint32_t> checkpoint_slots_capacity;
    std::uint32_t checkpoint_slots_reserved = 0;


    [[nodiscard]] friend constexpr bool operator==(const PhysicalUsageSnapshot&,
                                                   const PhysicalUsageSnapshot&) noexcept = default;
};

enum class TextPhase {
    Prefill,
    Verify,
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
    // Captured MTP draft width this profile belongs to (kMtpDecodeMaximumDrafts domain);
    // 0 for the ordinary/DFlash families and for the MTP profiles of a fixed-k run. A
    // non-zero value means the segment is one rung of the capture-width ladder
    // (kMtpWindowLadder), and the replay selects the profile by width as well as by batch and
    // frontier -- that selection is what makes the marginal draft column cost b_width instead
    // of the 7.2x-smaller b_mask.
    std::uint32_t draft_width    = 0;
};

// Program-minted shortlist metadata. It only narrows catalog inspection; Program still performs
// exact token, position, media and runtime-mode verification before a checkpoint can be selected.
struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier     = 0;
    std::uint32_t identity_tag = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefixShortlistKey,
                                                   PrefixShortlistKey) noexcept = default;
};

struct TargetKVRequirement {
    std::uint32_t main_frontier    = 0;
    std::uint32_t backend_frontier = 0;
    std::uint32_t main_pages       = 0;
    std::uint32_t backend_pages    = 0;

    [[nodiscard]] friend constexpr bool operator==(TargetKVRequirement,
                                                   TargetKVRequirement) noexcept = default;
};

struct CheckpointSummary {
    runtime::CheckpointRef ref;
    runtime::CheckpointScope scope = runtime::CheckpointScope::Private;
    PrefixShortlistKey shortlist_key;
    runtime::ReplicaResidency state_residency = runtime::ReplicaResidency::DeviceOnly;
    TargetKVRequirement required_kv;
    runtime::PrefillWork rebuild_work;

    [[nodiscard]] friend bool operator==(const CheckpointSummary&,
                                         const CheckpointSummary&) noexcept = default;
};

struct ContinuationSummary {
    std::optional<CheckpointSummary> endpoint;
    std::optional<CheckpointSummary> rewrite;
    std::vector<CheckpointSummary> long_anchors;
    std::uint32_t active_references = 0;

    [[nodiscard]] friend bool operator==(const ContinuationSummary&,
                                         const ContinuationSummary&) noexcept = default;
};

struct SharedPrefixSummary {
    CheckpointSummary checkpoint;
    std::uint32_t active_references = 0;

    [[nodiscard]] friend bool operator==(const SharedPrefixSummary&,
                                         const SharedPrefixSummary&) noexcept = default;
};

namespace detail {
template <class Variant>
struct SequencePlanImpl;
template <class Variant>
struct SequencePlannerImpl;
template <class Variant>
struct AdmissionCandidateImpl;
template <class Variant>
struct RequestBasePlanImpl;
template <class Variant>
struct PressurePlanningSessionImpl;
template <class Variant>
class ProgramImpl;
template <class Variant>
struct RuntimeContractAccess;
} // namespace detail

template <class Variant>
class SequencePlanner;
template <class Variant>
class Program;
template <class Variant>
class PressurePlanningSession;
template <class Variant>
class CapturePressurePlan;

// These are the complete family execution types. Exact packages bind them to a private Variant;
// target selection remains outside this layer and happens once in the closed Engine registry.
template <class Variant>
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl_;

    template <class V>
    friend class SequencePlanner;
    template <class V>
    friend class detail::ProgramImpl;
};

template <class Variant>
class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan<Variant> finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl_;

    template <class V>
    friend SequencePlanner<V> make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                    typename V::WeightsProfile);
};

template <class Variant>
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept;
    [[nodiscard]] std::optional<runtime::PrefillWork>
    shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl_;
};

template <class Variant>
class AdmissionCandidate {
public:
    AdmissionCandidate(AdmissionCandidate&&) noexcept;
    AdmissionCandidate& operator=(AdmissionCandidate&&) noexcept;
    ~AdmissionCandidate();

    AdmissionCandidate(const AdmissionCandidate&)            = delete;
    AdmissionCandidate& operator=(const AdmissionCandidate&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const runtime::IdentityMaterializationAssessment&
    identity_assessment() const noexcept;

public:
    // Family-private construction/storage seam. Exact packages expose only the completed alias;
    // Engine code can inspect summary() but not target planning state.
    explicit AdmissionCandidate(
        std::unique_ptr<detail::AdmissionCandidateImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::AdmissionCandidateImpl<Variant>> impl_;

    friend class Program<Variant>;
    friend class PressurePlanningSession<Variant>;
    friend class CapturePressurePlan<Variant>;
    friend struct detail::PressurePlanningSessionImpl<Variant>;
};

// A sealed pressure-only post-state for one active capture. The contained synthetic admission
// candidate is never executable as a request; it only carries Program-private owner outcomes.
template <class Variant>
class CapturePressurePlan {
public:
    CapturePressurePlan(CapturePressurePlan&&) noexcept            = default;
    CapturePressurePlan& operator=(CapturePressurePlan&&) noexcept = default;
    ~CapturePressurePlan()                                         = default;

    CapturePressurePlan(const CapturePressurePlan&)            = delete;
    CapturePressurePlan& operator=(const CapturePressurePlan&) = delete;

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

private:
    CapturePressurePlan(AdmissionCandidate<Variant>&& pressure, std::uint64_t revision) noexcept
        : pressure_(std::move(pressure)), revision_(revision) {}

    AdmissionCandidate<Variant> pressure_;
    std::uint64_t revision_ = 0;

    friend class Program<Variant>;
    friend class PressurePlanningSession<Variant>;
};

// A sealed Program-owned physical decision.  ResourceManager may retain it and inspect the
// request-level summary, but cannot see allocator quantities, references, reservations, or stage
// deltas.  Start validates the bound Program revision before performing any mutation.
template <class Variant>
class ResourcePlan {
public:
    ResourcePlan(ResourcePlan&&) noexcept            = default;
    ResourcePlan& operator=(ResourcePlan&&) noexcept = default;
    ~ResourcePlan()                                  = default;

    ResourcePlan(const ResourcePlan&)            = delete;
    ResourcePlan& operator=(const ResourcePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept {
        return admission_.summary();
    }

    [[nodiscard]] bool needs_transfer() const noexcept { return needs_transfer_; }

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

private:
    ResourcePlan(AdmissionCandidate<Variant>&& admission, std::uint64_t revision,
                 bool needs_transfer) noexcept
        : admission_(std::move(admission)), revision_(revision), needs_transfer_(needs_transfer) {}

    AdmissionCandidate<Variant> admission_;
    std::uint64_t revision_ = 0;
    bool needs_transfer_    = false;

    friend class Program<Variant>;
    friend class PressurePlanningSession<Variant>;
};

// A Program-minted proof that one FIFO borrower cannot consume the maximum physical entitlement
// reserved for the blocked head.  Common scheduling binds the opaque proof to logical identities
// and a revision; it cannot inspect or reproduce the resource arithmetic.
template <class Variant>
class PersistentBackfillProof {
public:
    PersistentBackfillProof(PersistentBackfillProof&&) noexcept            = default;
    PersistentBackfillProof& operator=(PersistentBackfillProof&&) noexcept = default;

    PersistentBackfillProof(const PersistentBackfillProof&)            = delete;
    PersistentBackfillProof& operator=(const PersistentBackfillProof&) = delete;

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

private:
    explicit PersistentBackfillProof(std::uint64_t revision) noexcept : revision_(revision) {}

    std::uint64_t revision_ = 0;

    friend class Program<Variant>;
};

template <class Variant>
class SequenceHandle {
public:
    SequenceHandle() noexcept                                 = default;
    SequenceHandle(const SequenceHandle&) noexcept            = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept = default;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
class ContinuationHandle {
public:
    ContinuationHandle() noexcept = default;
    ~ContinuationHandle()         = default;

    ContinuationHandle(ContinuationHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    ContinuationHandle& operator=(ContinuationHandle&&)      = delete;
    ContinuationHandle(const ContinuationHandle&)            = delete;
    ContinuationHandle& operator=(const ContinuationHandle&) = delete;

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
class SharedPrefixHandle {
public:
    SharedPrefixHandle() noexcept = default;
    ~SharedPrefixHandle()         = default;

    SharedPrefixHandle(SharedPrefixHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    SharedPrefixHandle& operator=(SharedPrefixHandle&&)      = delete;
    SharedPrefixHandle(const SharedPrefixHandle&)            = delete;
    SharedPrefixHandle& operator=(const SharedPrefixHandle&) = delete;

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess<Variant>;
};

class PressureTargetHandle {
public:
    PressureTargetHandle() noexcept = default;

    [[nodiscard]] friend constexpr bool operator==(PressureTargetHandle,
                                                   PressureTargetHandle) noexcept = default;

private:
    const void* session_      = nullptr;
    std::uint32_t generation_ = 0;
    std::uint32_t index_      = 0;

    template <class Variant>
    friend struct detail::PressurePlanningSessionImpl;
    template <class Variant>
    friend class PressurePlanningSession;
};

// RAII lease on one construction slot of a PressurePlanningSession.  A move-only handle whose
// destructor releases the slot; the session accesses session_/slot_/generation_ through the
// detail::PressurePlanningSessionImpl specialization the implementation defines.
// Donor: igorls/ninfer @ 5e4a66d src/targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h,
// verbatim apart from this comment.  Nothing in this tree instantiates it yet: the Flash-Next
// target owns the session that leases it (its own impl/program.cpp).
class PressureConstructionCursor {
public:
    PressureConstructionCursor(PressureConstructionCursor&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)), slot_(other.slot_),
          generation_(other.generation_), release_(other.release_) {}

    PressureConstructionCursor& operator=(PressureConstructionCursor&&)      = delete;
    PressureConstructionCursor(const PressureConstructionCursor&)            = delete;
    PressureConstructionCursor& operator=(const PressureConstructionCursor&) = delete;

    ~PressureConstructionCursor() {
        if (session_) { release_(session_, slot_, generation_); }
    }

private:
    PressureConstructionCursor(const void* session, std::uint32_t slot, std::uint32_t generation,
                               void (*release)(const void*, std::uint32_t, std::uint32_t) noexcept)
        : session_(session), slot_(slot), generation_(generation), release_(release) {}

    const void* session_;
    std::uint32_t slot_;
    std::uint32_t generation_;
    void (*release_)(const void*, std::uint32_t, std::uint32_t) noexcept;
    template <class Variant>
    friend struct detail::PressurePlanningSessionImpl;
};

template <class Variant>
class PreparedPressureExpansion {
public:
    PreparedPressureExpansion(PreparedPressureExpansion&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)),
          session_generation_(std::exchange(other.session_generation_, 0)),
          scratch_generation_(std::exchange(other.scratch_generation_, 0)),
          parent_index_(other.parent_index_), new_canonical_count_(other.new_canonical_count_) {}

    PreparedPressureExpansion& operator=(PreparedPressureExpansion&&)      = delete;
    PreparedPressureExpansion(const PreparedPressureExpansion&)            = delete;
    PreparedPressureExpansion& operator=(const PreparedPressureExpansion&) = delete;

    [[nodiscard]] std::uint32_t new_canonical_count() const noexcept {
        return new_canonical_count_;
    }

private:
    PreparedPressureExpansion(const void* session, std::uint32_t session_generation,
                              std::uint32_t scratch_generation, std::uint32_t parent_index,
                              std::uint32_t new_canonical_count) noexcept
        : session_(session), session_generation_(session_generation),
          scratch_generation_(scratch_generation), parent_index_(parent_index),
          new_canonical_count_(new_canonical_count) {}

    const void* session_               = nullptr;
    std::uint32_t session_generation_  = 0;
    std::uint32_t scratch_generation_  = 0;
    std::uint32_t parent_index_        = 0;
    std::uint32_t new_canonical_count_ = 0;

    friend class PressurePlanningSession<Variant>;
    friend struct detail::PressurePlanningSessionImpl<Variant>;
};

struct PressureExpansionView {
    std::span<const PressureTargetHandle> children;
    std::uint32_t new_canonical_count = 0;
};

template <class Variant>
class PressurePlanningSession {
public:
    PressurePlanningSession(PressurePlanningSession&&) noexcept;
    PressurePlanningSession& operator=(PressurePlanningSession&&) noexcept;
    ~PressurePlanningSession();

    PressurePlanningSession(const PressurePlanningSession&)            = delete;
    PressurePlanningSession& operator=(const PressurePlanningSession&) = delete;

    [[nodiscard]] PressureTargetHandle
    identity_target(const AdmissionCandidate<Variant>& candidate) const;
    [[nodiscard]] PressureTargetHandle
    root_maximal_target(const AdmissionCandidate<Variant>& root_candidate);
    [[nodiscard]] std::optional<PressureTargetHandle>
    guided_closure_target(const AdmissionCandidate<Variant>& candidate,
                          std::span<const std::uint32_t> preferred_owner_ordinals);
    [[nodiscard]] runtime::PressureTargetGuidance guidance(PressureTargetHandle target);
    [[nodiscard]] runtime::PressureTargetAssessment assess(PressureTargetHandle target);
    void retain_assessment(PressureTargetHandle target);
    [[nodiscard]] PreparedPressureExpansion<Variant> prepare_expansion(PressureTargetHandle parent);
    [[nodiscard]] PressureExpansionView
    commit_expansion(PreparedPressureExpansion<Variant>&& prepared);
    void discard_expansion(PreparedPressureExpansion<Variant>&& prepared) noexcept;
    [[nodiscard]] std::optional<ResourcePlan<Variant>> seal(PressureTargetHandle target,
                                                            const PreparedPrompt& prompt);
    [[nodiscard]] std::optional<CapturePressurePlan<Variant>>
    seal_capture(PressureTargetHandle target);

private:
    explicit PressurePlanningSession(
        std::unique_ptr<detail::PressurePlanningSessionImpl<Variant>> impl) noexcept;

    std::unique_ptr<detail::PressurePlanningSessionImpl<Variant>> impl_;

    friend class Program<Variant>;
};

template <class Variant>
class CaptureOffer {
public:
    CaptureOffer() noexcept = default;
    ~CaptureOffer()         = default;

    CaptureOffer(CaptureOffer&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), lane_(other.lane_), epoch_(other.epoch_),
          id_(std::exchange(other.id_, 0)) {}

    CaptureOffer& operator=(CaptureOffer&&)      = delete;
    CaptureOffer(const CaptureOffer&)            = delete;
    CaptureOffer& operator=(const CaptureOffer&) = delete;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;
    std::uint64_t id_    = 0;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(other.row_stride_), timing_(other.timing_) {
        other.tokens_     = {};
        other.row_counts_ = {};
        other.row_stride_ = 0;
        other.timing_     = {};
    }

    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }

    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle<Variant>, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch<Variant>> pending;
    std::optional<CaptureOffer<Variant>> capture;
};

enum class CaptureStatePlacement : std::uint8_t {
    DeviceFork,
    HostSnapshot,
};

struct CaptureAssessment {
    CaptureAssessment();

    // Program retains the physical assessment in this opaque package-private payload.
    std::shared_ptr<detail::CaptureAssessmentImpl> implementation;
    PrefixShortlistKey shortlist_key;
    SharedCandidateEvidence shared_evidence = SharedCandidateEvidence::None;
    runtime::PrefillWork protected_rebuild_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::vector<runtime::CheckpointRef> private_replacement_candidates;
    std::uint32_t frontier                = 0;
    std::uint64_t immediate_ns            = 0;
    std::uint64_t projected_recovery_ns   = 0;
    bool publishes_private                = false;
    bool publishes_shared                 = false;
    bool needs_transfer                   = false;
    bool physically_feasible              = false;
    bool recycles_private_state           = false;
    CaptureStatePlacement state_placement = CaptureStatePlacement::DeviceFork;
};

template <class Variant>
struct SharedPrefixPublication {
    SharedPrefixHandle<Variant> handle;
    SharedPrefixSummary summary;
};

struct MaterializationVictimResult;
struct MaterializationSharedVictimResult;

template <class Variant>
struct ActiveCaptureResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed      = false;
    ContinuationSummary active_summary;
    std::optional<SharedPrefixPublication<Variant>> shared;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

template <class Variant>
struct StartResult {
    SequenceHandle<Variant> sequence;
};

struct MaterializationVictimResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    bool pressure_committed               = false;
    std::optional<ContinuationSummary> final_summary;
    // Which planning owner this victim belonged to.  APPENDED, so positional aggregate
    // initialisation elsewhere is unaffected.  Added for the copied Flash-Next target; the donor
    // carries it FIRST (igorls/ninfer @ 5e4a66d src/targets/qwen3_6/export/.../runtime.h:792).
    runtime::PlanningOwnerId owner;
};

struct MaterializationSharedVictimResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    bool pressure_committed               = false;
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationSourceResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    // Donor spelling of the same fact, ADDED beside `disposition` rather than replacing it so that
    // this tree's sites keep compiling.  It sits BEFORE `final_summary` because a site from the
    // copied target aggregate-initialises this struct with C++20 designated initialisers, which the
    // language requires to appear in declaration order: (.mode, .final_summary).  This tree's own
    // sites use (.disposition, .final_summary), which stays increasing either way.  Donor:
    // igorls/ninfer @ 5e4a66d src/targets/qwen3_6/export/.../runtime.h:806.
    runtime::PrivateSourceMode mode = runtime::PrivateSourceMode::Retain;
    std::optional<ContinuationSummary> final_summary;
};


struct MaterializationSharedSourceResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    std::optional<SharedPrefixSummary> final_summary;
};

template <class Variant>
struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult<Variant>> published;
    std::optional<MaterializationSourceResult> source;
    std::optional<MaterializationSharedSourceResult> shared_source;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

template <class Variant>
using ContextTransactionProgress =
    std::variant<runtime::ContextTransactionInProgress, MaterializationResult<Variant>,
                 ActiveCaptureResult<Variant>>;

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

template <class Variant>
struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    // A prompt-frontier capture becomes valid only after the generated Begin token is committed.
    // Keeping the move-only capability row-aligned avoids exposing provisional prompt state.
    std::array<std::optional<CaptureOffer<Variant>>, kMaximumConcurrency> captures{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

template <class Variant>
struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

template <class Variant>
struct FinishResult {
    runtime::ConsumeStatus status          = runtime::ConsumeStatus::InvariantMismatch;
    runtime::FinishDisposition disposition = runtime::FinishDisposition::Released;
    GenerationTimings timings;
    SpeculativeStats speculative;
    ContinuationSummary summary;
    std::optional<ContinuationHandle<Variant>> continuation;
};

template <class Variant>
struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

template <class Variant>
struct ReleaseResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
};

template <class Variant>
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine owns scheduling and logical residency policy. Program owns physical lanes, opaque
    // capabilities, model state and one immutable pending transaction at a time.
    [[nodiscard]] RequestBasePlan<Variant>
    plan_request(const PreparedPrompt& prompt, const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPrompt&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::optional<AdmissionCandidate<Variant>>
    inspect_admission(const PreparedPrompt& prompt, const RequestBasePlan<Variant>& base,
                      runtime::LaneId destination, const ContinuationHandle<Variant>* source,
                      const SharedPrefixHandle<Variant>* shared_source,
                      std::optional<runtime::CheckpointRef> checkpoint,
                      bool must_retain_private_source,
                      const runtime::ContextMachineCostModel& machine_cost);
    [[nodiscard]] std::optional<ResourcePlan<Variant>>
    seal_identity(const AdmissionCandidate<Variant>& candidate, const PreparedPrompt& prompt);
    [[nodiscard]] PressurePlanningSession<Variant>
    begin_pressure_planning(const runtime::ContextMachineCostModel& machine_cost,
                            std::span<const AdmissionCandidate<Variant>* const> candidates,
                            std::span<const ContinuationHandle<Variant>* const> private_owners,
                            std::span<const std::uint32_t> private_owner_ordinals,
                            std::span<const SharedPrefixHandle<Variant>* const> shared_owners,
                            std::span<const std::uint32_t> shared_owner_ordinals);
    void select_shared_captures(ResourcePlan<Variant>& plan, const PreparedPrompt& prompt,
                                std::span<const std::uint32_t> frontiers);
    [[nodiscard]] std::uint64_t
    shared_capture_split_cost_ns(const ResourcePlan<Variant>& plan, const PreparedPrompt& prompt,
                                 std::span<const std::uint32_t> frontiers,
                                 const runtime::ContextMachineCostModel& machine_cost);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    start_resource_transaction(ResourcePlan<Variant>&& plan, PreparedPrompt&& prompt,
                               runtime::CancellationFlagView cancellation);
    [[nodiscard]] std::optional<PersistentBackfillProof<Variant>>
    prove_persistent_backfill(const RequestBasePlan<Variant>& blocked_head,
                              const ResourcePlan<Variant>& candidate,
                              std::span<const SequenceHandle<Variant>> persistent_borrowers) const;
    [[nodiscard]] ContextTransactionProgress<Variant>
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] PrefillProgress<Variant>
    advance_prefill(SequenceHandle<Variant> sequence,
                    runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CaptureAssessment inspect_capture(
        const CaptureOffer<Variant>& offer, const SharedPrefixHandle<Variant>* exact_shared,
        const SharedPrefixHandle<Variant>* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] std::uint64_t
    checkpoint_recovery_ns(const ContinuationHandle<Variant>& owner,
                           runtime::CheckpointRef checkpoint,
                           const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] std::uint64_t
    checkpoint_recovery_ns(const SharedPrefixHandle<Variant>& owner,
                           runtime::CheckpointRef checkpoint,
                           const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] AdmissionCandidate<Variant>
    make_capture_pressure_candidate(const CaptureAssessment& assessment,
                                    const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer<Variant>& offer,
                                              const SharedPrefixHandle<Variant>& shared) const;
    void skip_capture(CaptureOffer<Variant>&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus reserve_active_capture(
        CaptureOffer<Variant>&& offer, const SharedPrefixHandle<Variant>* exact_shared,
        const SharedPrefixHandle<Variant>* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        const runtime::ContextMachineCostModel& machine_cost,
        runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        CaptureOffer<Variant>&& offer, const SharedPrefixHandle<Variant>* exact_shared,
        const SharedPrefixHandle<Variant>* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        CapturePressurePlan<Variant>&& pressure,
        const runtime::ContextMachineCostModel& machine_cost,
        runtime::CancellationFlagView cancellation);
    [[nodiscard]] PendingBatch<Variant> decode(std::span<const SequenceHandle<Variant>> sequences,
                                               std::span<const runtime::RoundBudget> budgets,
                                               runtime::ExecutionTiming* failed_timing = nullptr);
    // Advance each live sequence with its exact target-owned token row. This does not sample or
    // advance sampler RNG/occurrence state; callers own output publication and budget accounting.
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle<Variant>> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    // M21: append a run of ALREADY-KNOWN tokens to ONE live sequence and prefill it, so the model
    // can attend to them from the very next round. This is the same Program-level operation as
    // append_forced_tokens ("extend the sequence, then prefill the extension") and deliberately the
    // same body: every piece of thinking-control accounting lives in the engine
    // (EngineCore::run_control_batch), never in Program, so the caller that appends context gets
    // that body with NO control-token bookkeeping, and there is no second copy to drift from.
    // Contract (all enforced by the shared body, program_impl.h:8912-8945): the sequence is Active
    // with no prefill in flight (ledger_frontier == execution_frontier + 1,
    // text_kv_valid == execution_frontier), and execution_frontier + tokens.size() <= capacity.
    //
    // What this entry does NOT do, by construction (why is read off text_context_impl.h:1397):
    //  * it does not sample. The appended prefill runs with finalize_at_end=false, so the appended
    //    run's own bonus token comes from the following decode round -- exactly as the forced-token
    //    path behaves. The block that would read PrefillContext::sampling (text_context_impl.h:1479)
    //    is unreachable, so passing a sampling pointer here would be dead code.
    //  * it does not build an MTP proposal ladder (text_context_impl.h:1535 is gated on the same
    //    is_last), so a --draft-tree L,d (L > 1) run cannot get its round-1 lattice from an
    //    appended chunk. Chain MTP (paths <= 1) is unaffected: text_context.h:297-302 states that
    //    the chain spelling launches no extra work at all.
    //  * it does not publish the appended run as prompt material. The identity ledger records it
    //    through append_generated (program_impl.h:9075), not through a materialization identity.
    [[nodiscard]] runtime::ExecutionTiming
    append_context_prefill(SequenceHandle<Variant> sequence, std::span<const TokenId> tokens,
                           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult<Variant>
    commit(PendingBatch<Variant>&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult<Variant> abort_pending(PendingBatch<Variant>&& pending) noexcept;
    [[nodiscard]] FinishResult<Variant> finish(SequenceHandle<Variant> sequence) noexcept;
    [[nodiscard]] AbortResult<Variant> abort(SequenceHandle<Variant> sequence) noexcept;
    [[nodiscard]] ReleaseResult<Variant>
    release_continuation(ContinuationHandle<Variant>&& continuation) noexcept;
    [[nodiscard]] ReleaseResult<Variant>
    release_shared_prefix(SharedPrefixHandle<Variant>&& shared) noexcept;
    void fail_all_cleanup() noexcept;

    [[nodiscard]] bool
    isolated_request_feasible(const RequestBasePlan<Variant>& base) const noexcept;
    [[nodiscard]] std::uint64_t resource_revision() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    // mtplogxface: mtplogx added the counters and their accessors to ProgramImplCore
    // (impl/runtime/program.h, `mtp_graph_extension_calls_` / `..._nanoseconds_`) and made
    // resource_manager.h read them OFF THIS FACADE, but the facade half was never landed
    // (mtplogx's own pre-image set has no runtime.h), so the tree did not compile.
    // Pure insertion; the counters, their increments and every consumer stay mtplogx's.
    [[nodiscard]] std::uint64_t mtp_graph_extension_calls() const noexcept;
    [[nodiscard]] std::uint64_t mtp_graph_extension_nanoseconds() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

    // W6: the mutable prefill unit. The workspace and the persistent buffers are sized for
    // prefill_chunk_capacity() at startup, so installing any value in
    // [prefill_chunk_alignment, prefill_chunk_capacity()] at runtime is safe: shrinking always is,
    // growing past the capacity is not. The engine drives both through
    // `if constexpr (requires { program->set_prefill_chunk(0u); })`, and before these two
    // forwarders existed this wrapper exposed neither member, so that guard was FALSE for every
    // target and the bandwidth governor's prefill-unit shrink could not reach a Program at all --
    // it was unreachable code, not merely env-gated.
    //
    // Both members are pinned: tests/test_prefill_chunk_wiring.cpp asserts the engine's own named
    // predicate (bandwidth_detail::exposes_prefill_chunk_wire) for each target package and odr-uses
    // each member, so a declaration without a definition is a link error, and engine_core.h's call
    // site fails the build outright when the predicate is false. The guard is named in one place
    // because a guard and the check behind it disagreeing is how this came to be dead code.
    void set_prefill_chunk(std::uint32_t chunk) noexcept;
    [[nodiscard]] std::uint32_t prefill_chunk_capacity() const noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl<Variant>> impl_;

    template <class V>
    friend std::unique_ptr<Program<V>> create_program(const typename V::ModelView&,
                                                      typename V::WeightsProfile, SequencePlan<V>&&,
                                                      DeviceContext&);
};

namespace detail {

template <class Variant>
struct RuntimeContractAccess {
    [[nodiscard]] static SequenceHandle<Variant>
    make_sequence(const void* owner, runtime::LaneId lane, std::uint64_t epoch) noexcept {
        SequenceHandle<Variant> out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }

    [[nodiscard]] static ContinuationHandle<Variant>
    make_continuation(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        ContinuationHandle<Variant> out;
        out.owner_      = owner;
        out.index_      = index;
        out.generation_ = generation;
        return out;
    }

    [[nodiscard]] static SharedPrefixHandle<Variant>
    make_shared_prefix(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        SharedPrefixHandle<Variant> out;
        out.owner_      = owner;
        out.index_      = index;
        out.generation_ = generation;
        return out;
    }

    [[nodiscard]] static CaptureOffer<Variant> make_capture_offer(const void* owner,
                                                                  runtime::LaneId lane,
                                                                  std::uint64_t epoch,
                                                                  std::uint64_t id) noexcept {
        CaptureOffer<Variant> out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        out.id_    = id;
        return out;
    }

    [[nodiscard]] static const void* owner(const SequenceHandle<Variant>& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static const void* owner(const CaptureOffer<Variant>& offer) noexcept {
        return offer.owner_;
    }

    [[nodiscard]] static runtime::LaneId lane(const CaptureOffer<Variant>& offer) noexcept {
        return offer.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const CaptureOffer<Variant>& offer) noexcept {
        return offer.epoch_;
    }

    [[nodiscard]] static std::uint64_t id(const CaptureOffer<Variant>& offer) noexcept {
        return offer.id_;
    }

    static void consume(CaptureOffer<Variant>& offer) noexcept {
        offer.owner_ = nullptr;
        offer.id_    = 0;
    }

    [[nodiscard]] static runtime::LaneId lane(const SequenceHandle<Variant>& handle) noexcept {
        return handle.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SequenceHandle<Variant>& handle) noexcept {
        return handle.epoch_;
    }

    [[nodiscard]] static const void* owner(const ContinuationHandle<Variant>& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static std::uint32_t index(const ContinuationHandle<Variant>& handle) noexcept {
        return handle.index_;
    }

    [[nodiscard]] static std::uint64_t epoch(const ContinuationHandle<Variant>& handle) noexcept {
        return handle.generation_;
    }

    static void consume(ContinuationHandle<Variant>& handle) noexcept {
        handle.owner_      = nullptr;
        handle.generation_ = 0;
    }

    [[nodiscard]] static const void* owner(const SharedPrefixHandle<Variant>& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static std::uint32_t index(const SharedPrefixHandle<Variant>& handle) noexcept {
        return handle.index_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SharedPrefixHandle<Variant>& handle) noexcept {
        return handle.generation_;
    }

    static void consume(SharedPrefixHandle<Variant>& handle) noexcept {
        handle.owner_      = nullptr;
        handle.generation_ = 0;
    }

    [[nodiscard]] static PendingBatch<Variant>
    make_pending(const void* owner, std::uint64_t transaction,
                 std::span<const SequenceHandle<Variant>> rows, std::span<const TokenId> tokens,
                 std::span<const std::int32_t> row_counts, std::uint32_t row_stride,
                 runtime::ExecutionTiming timing) {
        PendingBatch<Variant> out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = row_counts;
        out.row_stride_ = row_stride;
        out.timing_     = timing;
        return out;
    }

    [[nodiscard]] static const void* owner(const PendingBatch<Variant>& pending) noexcept {
        return pending.owner_;
    }

    [[nodiscard]] static std::uint64_t transaction(const PendingBatch<Variant>& pending) noexcept {
        return pending.transaction_;
    }

    [[nodiscard]] static std::span<const SequenceHandle<Variant>>
    rows(const PendingBatch<Variant>& pending) noexcept {
        return {pending.rows_.data(), pending.row_count_};
    }

    static void consume(PendingBatch<Variant>& pending) noexcept {
        pending.owner_       = nullptr;
        pending.transaction_ = 0;
        pending.row_count_   = 0;
        pending.tokens_      = {};
        pending.row_counts_  = {};
        pending.row_stride_  = 0;
        pending.timing_      = {};
    }
};

} // namespace detail

template <class Variant>
[[nodiscard]] SequencePlanner<Variant>
make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                      typename Variant::WeightsProfile weights_profile);

template <class Variant>
[[nodiscard]] std::unique_ptr<Program<Variant>>
create_program(const typename Variant::ModelView& model,
               typename Variant::WeightsProfile weights_profile, SequencePlan<Variant>&& plan,
               DeviceContext& device);

} // namespace ninfer::targets::qwen3_6
