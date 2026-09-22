#pragma once

// virtual_device.h -- PRESENT ONE PHYSICAL GPU AS N LOGICAL RANKS, each shaped like a card this
// host does not own.
//
// WHY THIS EXISTS. A tensor-parallel implementation has to be exercised before it can be
// believed, and this box has exactly one GPU (RTX 5090 D, sm_120). The two hardware answers are
// both closed and were checked rather than assumed:
//
//   * MIG: `nvidia-smi mig -lgip` -> "No MIG-supported devices found", `mig.mode.current =
//     [N/A]`. Consumer GeForce has no MIG. Closed by the hardware.
//   * MPS: the only mechanism that gives a REAL soft spatial split
//     (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` = a percentage of the SMs per client,
//     `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` = a per-client memory cap; both documented in
//     /usr/local/cuda-13.3/targets/x86_64-linux/include/cuda.h). The daemon is not installed
//     (`/usr/bin/nvidia-cuda-mps-control` and `nvidia-cuda-mps-server` do not exist, nothing
//     named `nvidia-cuda-mps-*` is in the toolkit bin/, no MPS library in /usr/lib/wsl/lib or
//     /usr/lib/x86_64-linux-gnu), we are under WSL whose CUDA does not support MPS, and
//     `sudo -n` is not passwordless, so nothing can be installed. Closed by the environment.
//
// So the split here is LOGICAL, and the project's own plan already prescribes exactly this
// shape (docs/maintainer/multi-device-and-shard-plan.md: "read this as a contract that exists
// and is exercisable on one GPU", "which is why the whole thing is testable on a single-GPU
// host", "...this host does not have -- is exercisable on a single-GPU machine").
//
// WHAT A LOGICAL SPLIT IS: a per-rank BUDGET and a per-rank IDENTITY, written down where the
// engine already asks the device. It is deliberately NOT a partition of anything:
//
//   * an SM budget is not a partition of the SMs -- the ranks time-slice the same 170 SMs;
//   * a memory budget is not a partition of HBM -- the ranks share one physical allocation.
//
// That is why the banner below says so out loud, in the operator's face, every time it is on.
// A number produced under this guard is a correctness number and never a throughput number,
// and the failure this project keeps catching in its own reports is precisely a synthetic
// figure read as a measurement. The guard is built so that misreading it takes two deliberate
// actions, not one.
//
// THE GUARD, in the shape the task requires (explicit / loud / impossible by accident / no gate
// weakened by default / fails closed):
//
//   * OFF BY DEFAULT. Unset -> `active() == false`, and every consumer returns exactly what it
//     returned before this file existed: rank 0, world 1, the real `sm()`, the real SM count.
//     No behaviour in the engine changes because this header was added.
//   * TWO KEYS. `NINFER_VIRTUAL_DEVICES=<k>` states the intent and `NINFER_VIRTUAL_DEVICES_ACK=yes`
//     acknowledges that the numbers produced are not hardware numbers. One without the other is
//     a REFUSAL, not a silent no-op: an operator who set the first and mistyped the second must
//     not get an ordinary single-device run that looks like a successful simulation.
//   * FAILS CLOSED. Every way the request can be un-honourable is checked BEFORE anything is
//     reported as active, and each produces a named refusal. `active()` is true only when the
//     whole binding validated; a refusal is never reported as a smaller-but-working simulation.
//   * NEVER WEAKENS A GATE. The virtual capability is fed INTO the existing gates as their
//     input, not around them: a virtual V100 asked to run an NVFP4 artifact must be REFUSED by
//     `caps::require_artifact_formats_supported(70, ...)`, because sm_70's ladder row does not
//     cover kind::mxf4nvf4. That is a real V100's answer. The guard changes which QUESTION is
//     asked; it never changes an answer.
//
// Host-only: no CUDA header. The validation and the arithmetic are pure, so the guard itself is
// unit-testable with plain g++ on a machine with no GPU at all -- which is the only way to test
// a guard whose whole job is to be trustworthy.

#include "core/shard_plan.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::multi {

// ---------------------------------------------------------------------------
// 0. THE COMPATIBILITY CONTRACT, TAKEN FROM A SIBLING FORK RATHER THAN INVENTED
// ---------------------------------------------------------------------------
//
// `wamansou/ninfer-tp2-1m` is a fork of this engine's upstream (`Neroued/ninfer`, Apache-2.0,
// branched from upstream `feaf4dd`) that already ships dual-GPU tensor parallelism. The record
// this tree holds for that INTERFACE is `docs/gfx906/TP2-SLICES.md` (its own header carries the
// sha256(src)), and the facts below are adopted from that record because they are constraints
// this tree has to satisfy, not opinions. `tp2-yarn-1m` -- the donor's own spelling of this
// work, cited in that form by core/tp_transport.h -- is the DONOR BRANCH, not a path here:
//
// ⚠ MEASURED 2026-09-20: the donor's design doc is NOT IN THIS TREE, and the donor tree is not
// on this box either (no `ninfer-tp2-1m` checkout under /home/user, and no such file anywhere
// reachable). The name `tp2-yarn-1m` is the DONOR BRANCH (`@ tp2-yarn-1m`, spelled out in
// core/tp_transport.h) turned into a doc path. What this tree DOES hold for this donor is the
// port's own record, `docs/gfx906/TP2-SLICES.md` -- source repo `JCraigWasTaken/ninfer-gfx906 @
// gfx906-port`, source path `docs/gfx906/TP2-SLICES.md`, the sha256(src) recorded in its own
// PROVENANCE block, landed by the fork-survey merge -- and it carries this section's facts as its
// own slice tables (`plan_for`, the `Q|K|Gate|V` and `Q|K|V|Z` fused orders, the row/column
// slice geometry). core/and_reduce.h and core/decode_graph_peer.h already cite exactly that
// path. Read the INTERFACE from there, not from the name above.
//
//   * SHARDING IS A PURE HOST COMPUTATION. `plan_for(object, tp, config)` produces the shard
//     map; no kernel is rewritten for TP. Each shard is a shard-shaped tensor through the
//     existing kernel plus a registry entry and a per-rank wrapper. That is the same shape as
//     core/shard_plan.h's `plan_shards()`, and it is why the arithmetic is checkable on a host.
//   * THE SPLIT BOUNDARY IS THE BINDER'S FUSED ROW ORDER, NOT A NAIVE N CUT. The attention
//     input projection's fused order is "Q | K | Gate | V" and the GDN input projection's is
//     "Q(2048) | K(2048) | V(6144) | Z(6144)". A rank's `weight_columns` Range is therefore a
//     range of COLUMNS IN THE FUSED TENSOR, and the caller must map it through the binder -- the
//     `weight_columns % world_size` rule in shard_plan.h is necessary and not sufficient.
//   * NVFP4 ROW SLICING IS SAFE AT TILE BOUNDARIES. The 128-row tile is the outermost axis of
//     `blockscale-k16-m128x4-v1`, so a shard's row range is contiguous and bindable with no
//     repacking -- which is the answer to "there is no N-dim column slicer in src/artifact/".
//     The row range must land on that boundary; that is a constraint on the shard, not a defect.
//   * GDN SPLITS BY VALUE HEAD WITH NO CROSS-HEAD REDUCTION. It is a pure launch-geometry change
//     (one CTA per value head). So the linear-attention state splits head-locally, and the
//     attention combine in core/shard_rank_axis.h is needed for the 16 full-attention layers.
//   * THE TRANSPORT IS A SEPARATE COMPONENT. On the fork's two cards `cudaDeviceCanAccessPeer`
//     reported 0 in both directions (PHB topology, driver restriction), and the collectives are
//     host-staged async copies over PCIe; `cudaMemcpyPeerAsync` is not stream-capturable, so the
//     D2D/UVA form of `cudaMemcpyAsync` is used. All of that is the shard interface's
//     counter-party. This engine owes it a correct shard interface, a correct split/join algebra
//     and correct per-rank addressing -- and nothing here is a measurement of the transport.
//   * RANK 0 IS PRIMARY. It owns tokenization, sampling, the retained MTP target hidden state and
//     every request-visible state; the other ranks run their half of each sharded op and hold no
//     request-visible state of their own. `VirtualBinding::rank` is therefore a role as well as
//     an index, and a virtual world preserves that.
//   * ONE CROSS-DEVICE CUDA GRAPH. Decode is a single fork/join graph spanning both devices; two
//     live captures fail with `cudaErrorStreamCaptureMerge`, and capture costs about +15% nodes
//     (1888 at tp2 against 640 at tp1). That is a hard constraint on this tree's DecodeGraph
//     (src/core/decode_graph.h), which currently captures one device.
//   * QUALIFICATION IS COMPARATIVE, WITH A NULL CONTROL. The fork gates tp1<->tp2 parity by
//     teacher-forced per-position agreement: 95.73% argmax / 0.99328 cosine / 0.01566 KL against
//     a 0.07142 KL budget, with the device-0-vs-device-1 null control at 100% / 1.0 / 0.0. A
//     single-rank world must be byte-identical to upstream on greedy decode
//     (`scripts/tp1-regression.sh`, narrow scope: greedy text decode, --tp 1, native rope, NVFP4,
//     MTP off, concurrency 1). This tree's equivalent is the world_size == 1 identity that
//     core/shard_plan.h and core/shard_rank_axis.h are built to preserve.
//   * AND THE FRAMING, which decides what is worth building: "TP2 is a CAPACITY feature, not a
//     scale-out feature." The deliverable is compatibility -- the engine builds for and is driven
//     on those cards and on virtual ranks, computes the RIGHT ANSWER, and picks a route that
//     exists. Performance tuning is secondary and comes later.

// The environment surface, named once so nothing can spell it differently.
inline constexpr std::string_view kVirtualDevicesEnv     = "NINFER_VIRTUAL_DEVICES";
inline constexpr std::string_view kVirtualDevicesAckEnv   = "NINFER_VIRTUAL_DEVICES_ACK";
inline constexpr std::string_view kVirtualDeviceProfileEnv = "NINFER_VIRTUAL_DEVICE_PROFILE";
// WHICH rank this DeviceContext object is. The engine has no rank plumbing yet (shard_plan.h
// names it as missing), so the rank arrives from the environment and nowhere else. That is
// enough to instantiate several rank objects IN ONE PROCESS -- which is the whole point of a
// single-process multi-rank simulator, and the only form of it this box can honour.
inline constexpr std::string_view kVirtualRankEnv         = "NINFER_VIRTUAL_RANK";
inline constexpr std::string_view kVirtualAckValue        = "yes";

// The cards this guard can shape a rank like. A closed set on purpose: a profile is a claim
// about a capability set, and a free-form "pretend to be sm_<n>" would let a caller invent a
// capability nobody measured -- the same defect as the deleted `if (device.sm() != 120) throw`,
// which is "a number standing in for evidence". sm_70 is here because V100 is the card the
// record names (tools/archkit/_GPU_MATRIX.md, "sm 70 | V100 (CUDA12 旧链) | 仅 fp16 mma |
// QPN2 W4A16 | 待移植"). sm_75/80/86/89/90 are here because those rungs of the ladder are the
// ones the fallback work is pinned to, and each one must be reachable for a table test.
enum class VirtualProfile : std::uint8_t {
    None,
    V100,  // sm_70,  fp16 mma only,        80 SMs
    T4,    // sm_75,  int8 tensor core,     40 SMs
    A100,  // sm_80,  int8/bf16,            108 SMs
    RTX30, // sm_86,  int8/bf16,            82 SMs
    RTX40, // sm_89,  fp8,                  128 SMs
    H100,  // sm_90,  fp8,                  132 SMs
};

[[nodiscard]] inline std::string_view virtual_profile_name(VirtualProfile profile) noexcept {
    switch (profile) {
    case VirtualProfile::None: return "none";
    case VirtualProfile::V100: return "v100";
    case VirtualProfile::T4: return "t4";
    case VirtualProfile::A100: return "a100";
    case VirtualProfile::RTX30: return "rtx30";
    case VirtualProfile::RTX40: return "rtx40";
    case VirtualProfile::H100: return "h100";
    }
    return "unknown-profile";
}

// sm number as `DeviceContext::sm()` spells it (major * 10 + minor).
[[nodiscard]] inline int virtual_profile_sm(VirtualProfile profile) noexcept {
    switch (profile) {
    case VirtualProfile::None: return 0;
    case VirtualProfile::V100: return 70;
    case VirtualProfile::T4: return 75;
    case VirtualProfile::A100: return 80;
    case VirtualProfile::RTX30: return 86;
    case VirtualProfile::RTX40: return 89;
    case VirtualProfile::H100: return 90;
    }
    return 0;
}

// The SM count of the physical card the profile stands for. A TUNING input only, and its whole
// purpose here is to keep a virtual rank's grid-size derivation inside the soft budget (see
// core/device_sm_count.h, whose doctrine is "TUNING INPUT, NOT A CAPABILITY TEST").
[[nodiscard]] inline int virtual_profile_physical_sm_count(VirtualProfile profile) noexcept {
    switch (profile) {
    case VirtualProfile::None: return 0;
    case VirtualProfile::V100: return 80;
    case VirtualProfile::T4: return 40;
    case VirtualProfile::A100: return 108;
    case VirtualProfile::RTX30: return 82;
    case VirtualProfile::RTX40: return 128;
    case VirtualProfile::H100: return 132;
    }
    return 0;
}

// GB of HBM the profile stands for. Also a budget, never a partition: see the banner.
[[nodiscard]] inline std::uint64_t virtual_profile_memory_bytes(VirtualProfile profile) noexcept {
    constexpr std::uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
    switch (profile) {
    case VirtualProfile::None: return 0;
    case VirtualProfile::V100: return 16ULL * kGiB;
    case VirtualProfile::T4: return 16ULL * kGiB;
    case VirtualProfile::A100: return 40ULL * kGiB;
    case VirtualProfile::RTX30: return 24ULL * kGiB;
    case VirtualProfile::RTX40: return 24ULL * kGiB;
    case VirtualProfile::H100: return 80ULL * kGiB;
    }
    return 0;
}

[[nodiscard]] inline VirtualProfile virtual_profile_from_name(std::string_view name) noexcept {
    if (name == "v100") { return VirtualProfile::V100; }
    if (name == "t4") { return VirtualProfile::T4; }
    if (name == "a100") { return VirtualProfile::A100; }
    if (name == "rtx30") { return VirtualProfile::RTX30; }
    if (name == "rtx40") { return VirtualProfile::RTX40; }
    if (name == "h100") { return VirtualProfile::H100; }
    return VirtualProfile::None;
}

// ---------------------------------------------------------------------------
// The binding: what each rank is told to be, and the refusal that stops a bad one
// ---------------------------------------------------------------------------

// Why a request was refused. A closed enum so a test can assert the SPECIFIC refusal rather
// than "it failed", which is the difference between a check and a smoke test.
enum class VirtualRefusal : std::uint8_t {
    NotRequested,       // neither key is set: the guard is off, and that is not a refusal
    MissingAck,         // the intent key is set and the acknowledgement is not
    BadWorldSize,       // the requested rank count is not a usable world
    BadProfile,         // the profile name is not a card this guard can shape
    BadAxis,            // the requested axis cannot be simulated by ranks on one device
    PhysicalNotSingle,  // more than one physical GPU is present: simulate nothing, use them
    SmsTooFew,          // the soft SM budget cannot give every rank at least one SM
    MemoryTooSmall,     // the profile's memory does not fit the physical card even once
    BadRank,            // the calling rank is outside the world
};

[[nodiscard]] inline std::string_view virtual_refusal_name(VirtualRefusal refusal) noexcept {
    switch (refusal) {
    case VirtualRefusal::NotRequested: return "not-requested";
    case VirtualRefusal::MissingAck: return "missing-acknowledgement";
    case VirtualRefusal::BadWorldSize: return "bad-world-size";
    case VirtualRefusal::BadProfile: return "bad-profile";
    case VirtualRefusal::BadAxis: return "bad-axis";
    case VirtualRefusal::PhysicalNotSingle: return "physical-device-count-not-one";
    case VirtualRefusal::SmsTooFew: return "sm-budget-too-small";
    case VirtualRefusal::MemoryTooSmall: return "memory-budget-too-small";
    case VirtualRefusal::BadRank: return "rank-outside-world";
    }
    return "unknown-refusal";
}

// The request, as read from the environment. Kept separate from the validation so the parse is
// testable without an environment.
struct VirtualRequest {
    bool requested     = false; // NINFER_VIRTUAL_DEVICES was set
    bool acknowledged  = false; // ... and NINFER_VIRTUAL_DEVICES_ACK == "yes"
    std::uint32_t world_size = 0;
    std::uint32_t rank       = 0;
    ParallelAxis axis        = ParallelAxis::Tensor;
    VirtualProfile profile   = VirtualProfile::V100;
    std::string profile_name_raw;
    std::string world_size_raw;
};

// The validated binding. `active` is the ONLY thing a consumer may branch on, and it is true
// only when validation produced no refusal at all.
struct VirtualBinding {
    bool active = false;

    VirtualRefusal refusal = VirtualRefusal::NotRequested;
    std::string refusal_detail; // the operator-facing sentence, empty when active

    // What each rank is told. `rank`/`world_size` are the WorldShape the engine carries.
    WorldShape world{};

    VirtualProfile profile = VirtualProfile::None;
    int capability_sm      = 0; // fed INTO `DeviceContext::sm()`, never around a gate
    std::uint32_t sm_budget = 0; // per-rank soft SM budget (tuning input)
    std::uint64_t memory_budget_bytes = 0; // per-rank soft memory budget
    // THE RANK ROLE, adopted from the two forks that actually built this
    // (`wamansou/ninfer-tp2-1m`, `JCraigWasTaken/ninfer-gfx906`): rank 0 is PRIMARY and owns
    // tokenization, sampling, the retained MTP target hidden state and every request-visible
    // state; the other ranks run their half of each sharded op and hold NO request-visible state
    // of their own. That is not an implementation detail -- it decides which rank may own a
    // request, which rank a sampling result comes from, and which rank a stall is attributable
    // to. Both forks keep the role explicit, so a binding that carried only an index would be
    // less than they already have. `primary` is derived `(rank == 0)` rather than stored, so the
    // two cannot disagree.
    [[nodiscard]] bool primary() const noexcept { return world.rank == 0U; }
    // The PHYSICAL device id this rank is bound to: always 0, because the validator refuses a
    // binding when more than one CUDA device is present. The rank is not the device id -- rank 1
    // of a 2-rank world still runs on physical device 0, which is what makes several rank objects
    // possible in one process.
    std::uint32_t physical_device_index = 0;
};

// The validation, as a pure function of its inputs. Nothing here reads the device: the physical
// facts are passed in, which is what makes every refusal reachable in a host test.
[[nodiscard]] inline VirtualBinding validate_virtual_request(const VirtualRequest& request,
                                                             int physical_device_count,
                                                             int physical_sm_count,
                                                             std::uint64_t physical_memory_bytes) {
    VirtualBinding binding;

    if (!request.requested) {
        binding.refusal        = VirtualRefusal::NotRequested;
        binding.refusal_detail = "no virtual device was requested";
        return binding;
    }
    // FAIL CLOSED on a half-given request. This is the "impossible to enable by accident" half:
    // the intent without the acknowledgement is a mistake, and a mistake must not be served as
    // a working single-device run.
    if (!request.acknowledged) {
        binding.refusal = VirtualRefusal::MissingAck;
        binding.refusal_detail =
            std::string(kVirtualDevicesEnv) + "=" + request.world_size_raw +
            " was set but " + std::string(kVirtualDevicesAckEnv) + " is not \"" +
            std::string(kVirtualAckValue) +
            "\". A virtual device is not a device: the SM and memory numbers it produces are "
            "BUDGETS the ranks share, not partitions, and no throughput or communication "
            "measurement taken under it is a hardware result. Set " +
            std::string(kVirtualDevicesAckEnv) + "=" + std::string(kVirtualAckValue) +
            " to say you know that, or unset " + std::string(kVirtualDevicesEnv) +
            " to run on the real device.";
        return binding;
    }
    if (request.world_size < 2U) {
        binding.refusal = VirtualRefusal::BadWorldSize;
        binding.refusal_detail =
            std::string(kVirtualDevicesEnv) + "=" + request.world_size_raw +
            ": a virtual world of fewer than 2 ranks is not a simulation of anything. A single "
            "rank is the real device, and running it virtually would only add a number nobody "
            "asked for.";
        return binding;
    }
    if (request.world_size > 64U) {
        binding.refusal = VirtualRefusal::BadWorldSize;
        binding.refusal_detail = std::string(kVirtualDevicesEnv) + "=" +
                                 request.world_size_raw +
                                 ": at most 64 ranks; beyond that the soft SM budget cannot "
                                 "give every rank one SM and the simulation would be a fiction "
                                 "about a fiction.";
        return binding;
    }
    if (request.profile == VirtualProfile::None) {
        binding.refusal        = VirtualRefusal::BadProfile;
        binding.refusal_detail = std::string(kVirtualDeviceProfileEnv) + "=\"" +
                                 request.profile_name_raw +
                                 "\" is not a card this guard can shape a rank like. Known: "
                                 "v100 t4 a100 rtx30 rtx40 h100. An unknown profile is refused "
                                 "rather than defaulted, because the capability set it implies "
                                 "is what the engine's gates will be asked about.";
        return binding;
    }
    if (request.rank >= request.world_size) {
        binding.refusal = VirtualRefusal::BadRank;
        binding.refusal_detail =
            std::string(kVirtualRankEnv) + "=" + std::to_string(request.rank) +
            " is outside world_size " + std::to_string(request.world_size) +
            ": a rank must be one of the world's own. Either the rank belongs to a different "
            "world than " + std::string(kVirtualDevicesEnv) +
            " declares, or it was mistyped; in both cases what would be simulated is a world "
            "that does not contain this rank.";
        return binding;
    }
    // A PIPELINE axis cannot be simulated by ranks inside one process, and saying so is the
    // honest answer rather than a caveat. The pipeline axis puts a different SUBSET OF LAYERS on
    // each stage, and the engine's page record is "one logical page ACROSS ALL TEXT LAYERS"
    // (cold_host_tier.h's GRANULARITY, spec/turn_recall_journal.h). Stage-local ranks in one
    // address space would each hold all layers' buffers, so the simulation would not have the
    // property under test -- it would be a tensor world wearing a pipeline's name.
    if (request.axis == ParallelAxis::Pipeline) {
        binding.refusal = VirtualRefusal::BadAxis;
        binding.refusal_detail =
            "axis=pp cannot be simulated by ranks inside one process: a pipeline stage differs "
            "from its neighbours by WHICH LAYERS it owns, and every rank here owns every layer's "
            "buffers, so the property under test (page_content_is_layer_complete(pp) == false) "
            "would be absent by construction. pp needs real processes, hence real devices.";
        return binding;
    }
    if (request.axis == ParallelAxis::None) {
        binding.refusal        = VirtualRefusal::BadAxis;
        binding.refusal_detail = "axis=none with a virtual world: a multi-device world must name "
                                 "the axis it splits on (see shard_plan.h world_shape_refusal)";
        return binding;
    }
    // A second physical card makes the whole exercise pointless AND misleading: the run would
    // look like a simulation while a real second device sat idle.
    if (physical_device_count != 1) {
        binding.refusal = VirtualRefusal::PhysicalNotSingle;
        binding.refusal_detail =
            "this guard presents ONE physical device as several logical ones, but " +
            std::to_string(physical_device_count) +
            " physical CUDA devices are present. Use them: a real world is strictly better "
            "evidence than a simulated one, and running the simulation here would leave real "
            "hardware idle while producing numbers that cannot be compared to it.";
        return binding;
    }
    if (physical_sm_count <= 0) {
        binding.refusal        = VirtualRefusal::SmsTooFew;
        binding.refusal_detail = "the physical device reported no usable SM count, so no per-rank "
                                 "SM budget can be derived";
        return binding;
    }
    const std::uint32_t sm_budget =
        static_cast<std::uint32_t>(physical_sm_count) / request.world_size;
    if (sm_budget == 0) {
        binding.refusal = VirtualRefusal::SmsTooFew;
        binding.refusal_detail =
            "the physical device has " + std::to_string(physical_sm_count) + " SMs and the world "
            "asks for " + std::to_string(request.world_size) +
            " ranks, so a rank would be given 0 SMs. A rank with no SM cannot launch anything, "
            "and reporting one would be a simulation that must be honoured by pretending.";
        return binding;
    }
    const std::uint64_t profile_memory = virtual_profile_memory_bytes(request.profile);
    if (physical_memory_bytes != 0 && profile_memory > physical_memory_bytes) {
        binding.refusal = VirtualRefusal::MemoryTooSmall;
        binding.refusal_detail =
            "the profile '" + std::string(virtual_profile_name(request.profile)) + "' stands for " +
            std::to_string(profile_memory) + " bytes of device memory but the physical device has " +
            std::to_string(physical_memory_bytes) +
            ": a rank would be given a memory budget larger than the card has, so any weight "
            "residency decision taken under it would be unreachable on the real card.";
        return binding;
    }

    binding.active              = true;
    binding.refusal             = VirtualRefusal::NotRequested;
    binding.refusal_detail.clear();
    binding.world               = WorldShape{request.world_size, request.rank, request.axis};
    binding.profile             = request.profile;
    binding.capability_sm       = virtual_profile_sm(request.profile);
    binding.sm_budget           = sm_budget;
    binding.memory_budget_bytes = profile_memory;
    // The physical device id every rank is bound to. See the member's comment: the rank is not
    // the device id, and that is what makes a single-process multi-rank world possible.
    binding.physical_device_index = 0;
    return binding;
}

// ---------------------------------------------------------------------------
// The environment reader and the loud banner
// ---------------------------------------------------------------------------

[[nodiscard]] inline std::string read_env_string(std::string_view name) {
    const std::string key(name);
    const char* value = std::getenv(key.c_str());
    return value == nullptr ? std::string{} : std::string(value);
}

[[nodiscard]] inline VirtualRequest virtual_request_from_environment(
    std::uint32_t rank = 0, ParallelAxis axis = ParallelAxis::Tensor) {
    VirtualRequest request;
    request.rank = rank;
    request.axis = axis;

    request.world_size_raw = read_env_string(kVirtualDevicesEnv);
    request.profile_name_raw = read_env_string(kVirtualDeviceProfileEnv);
    request.requested    = !request.world_size_raw.empty();
    request.acknowledged = read_env_string(kVirtualDevicesAckEnv) == kVirtualAckValue;

    // A rank in the environment overrides the default. Parsed with the same strictness as the
    // world size, so a typo is a refusal rather than rank 0.
    const std::string rank_raw = read_env_string(kVirtualRankEnv);
    if (!rank_raw.empty()) {
        unsigned parsed_rank = 0;
        for (const char c : rank_raw) {
            if (c < '0' || c > '9') { parsed_rank = 0; break; }
            parsed_rank = parsed_rank * 10U + static_cast<unsigned>(c - '0');
            if (parsed_rank > 4096U) { parsed_rank = 0; break; }
        }
        request.rank = parsed_rank;
    }

    unsigned parsed = 0;
    for (const char c : request.world_size_raw) {
        if (c < '0' || c > '9') { parsed = 0; break; }
        parsed = parsed * 10U + static_cast<unsigned>(c - '0');
        if (parsed > 4096U) { parsed = 0; break; }
    }
    request.world_size = parsed;

    // The profile defaults to v100 -- not as a convenience, but because the record names V100 as
    // the card this whole line exists for, and an unset profile must still be a NAMED card
    // rather than a free-form capability.
    request.profile = request.profile_name_raw.empty()
                          ? VirtualProfile::V100
                          : virtual_profile_from_name(request.profile_name_raw);
    return request;
}

// The banner. Loud, on stderr, once per process (the caller owns the once), naming exactly what
// the numbers below are and are not. A run that prints this is a run whose AGGREGATE THROUGHPUT
// may not be quoted as a two-card result, and the text says so in those words.
[[nodiscard]] inline std::string virtual_banner(const VirtualBinding& binding,
                                                std::string_view physical_name,
                                                std::uint32_t physical_sm_count) {
    if (!binding.active) { return {}; }
    std::string out;
    out += "================================ VIRTUAL DEVICES ================================\n";
    out += "NINFER_VIRTUAL_DEVICES is ON: presenting 1 physical GPU as " +
           std::to_string(binding.world.world_size) + " logical ranks shaped like " +
           std::string(virtual_profile_name(binding.profile)) + " (sm_" +
           std::to_string(binding.capability_sm) + ").\n";
    out += "  physical device : " + std::string(physical_name) + ", " +
           std::to_string(physical_sm_count) + " SMs\n";
    out += "  this rank       : " + std::to_string(binding.world.rank) + " of " +
           std::to_string(binding.world.world_size) + " (rank 0 is primary), axis " +
           std::string(axis_name(binding.world.axis)) + "\n";
    out += "  SM budget       : " + std::to_string(binding.sm_budget) +
           " SMs per rank (the ranks TIME-SLICE the same " + std::to_string(physical_sm_count) +
           "; this is a budget, not a partition)\n";
    out += "  memory budget   : " + std::to_string(binding.memory_budget_bytes) +
           " bytes per rank (shared physical HBM; a budget, not a partition)\n";
    out += "WHAT THIS IS       a CAPACITY simulation: N ranks on one card with the shard shape\n";
    out += "                   that --tp N gives on N cards, i.e. per-rank weight and KV\n";
    out += "                   residency divided by the world size.\n";
    out += "TRANSPORT          NOT THIS COMPONENT, and deliberately not simulated. The cross-rank\n";
    out += "                   transport (host-staged PCIe copies, an all-reduce library, or\n";
    out += "                   1Cat's custom all-reduce) is the shard INTERFACE's counter-party.\n";
    out += "                   What this engine owes it is a correct shard interface, a correct\n";
    out += "                   split/join algebra and correct per-rank addressing -- all three\n";
    out += "                   are exercised here. Nothing about the transport needs measuring.\n";
    out += "WHAT IS MEASURABLE per-rank wall clock (each rank timed with its own cudaEvent pair),\n";
    out += "                   the SKEW between ranks (%globaltimer, sampled at a common phase),\n";
    out += "                   and the shard BALANCE. Run each configuration twice -- serialized\n";
    out += "                   and concurrent -- and report both: the ratio is the contention a\n";
    out += "                   shared memory system imposes.\n";
    out += "WHAT IS NOT        AGGREGATE THROUGHPUT SCALING WITH RANK COUNT. The ranks share one\n";
    out += "                   HBM and time-slice one SM pool, so it is absent rather than noisy.\n";
    out += "                   PENDING HARDWARE.\n";
    out += "                   'Dual-GPU TP halves per-card weight and KV residency and is ~40%\n";
    out += "                   faster at long context' (wamansou/ninfer-tp2-1m) was measured on\n";
    out += "                   TWO REAL 5090s. It is neither reproduced nor refuted here and must\n";
    out += "                   not be compared against this run. TP2 is a CAPACITY feature, not a\n";
    out += "                   scale-out feature.\n";
    out += "================================================================================\n";
    return out;
}

// The one-line form for a log that wants the caveat without the box. It names the ONE quantity
// that this guard cannot speak to, rather than disclaiming the run as a whole: per-rank cost,
// skew and balance ARE measurable under a virtual world, and a blanket caveat would throw those
// away along with the aggregate.
[[nodiscard]] inline std::string virtual_caveat(std::string_view quantity) {
    return std::string(quantity) + " is PENDING HARDWARE: this run has N logical ranks on ONE "
                                   "physical GPU, sharing one HBM and one SM pool, so aggregate "
                                   "scaling with rank count is not measurable here. Per-rank cost, "
                                   "inter-rank skew and shard balance ARE.";
}

// ---------------------------------------------------------------------------
// The measurable timing surface -- and the two runs it must be taken in
// ---------------------------------------------------------------------------
//
// WHAT IS MEASURABLE ON ONE PHYSICAL DEVICE, precisely:
//
//   * PER-RANK WALL CLOCK. Each rank's own work timed with that rank's own `cudaEvent` pair
//     yields that rank's elapsed time. It is a real number, and it is not contaminated by the
//     other ranks' results -- only by their PRESENCE, which is what the second run below is for.
//   * INTER-RANK SKEW. `%globaltimer` is a nanosecond-resolution, device-global clock, so a
//     sample taken at a common phase inside each rank gives absolute timestamps whose
//     differences are the skew. This is what a shard plan needs validated: BALANCE. A rank that
//     starts 40 us late or finishes 2x later than its neighbour is a shard-arithmetic defect and
//     is visible here.
//   * BALANCE ITSELF, which needs no device at all: it is a property of the shard plan
//     (core/shard_rank_axis.h computes it) and is therefore checkable before anything runs.
//
// WHAT IS NOT: AGGREGATE THROUGHPUT SCALING WITH RANK COUNT. Two runs, and why both:
//
//   * `Serialized` -- ranks' phases separated by a host barrier, so each rank's cost is measured
//     without the others' kernels in flight. This is the clean per-rank cost.
//   * `Concurrent` -- all ranks' kernels in flight, so the memory system's contention shows up.
//
// The RATIO concurrent/serialized is the contention one memory system imposes. Reporting either
// one alone is misleading in a different direction, so both are part of the contract.
enum class VirtualTimingMode : std::uint8_t {
    Serialized, // ranks' phases separated: the clean per-rank cost
    Concurrent, // ranks' phases overlapping: the same cost plus contention
};

[[nodiscard]] inline std::string_view virtual_timing_mode_name(VirtualTimingMode mode) noexcept {
    switch (mode) {
    case VirtualTimingMode::Serialized: return "serialized";
    case VirtualTimingMode::Concurrent: return "concurrent";
    }
    return "unknown-mode";
}

// One rank's timing, as the two clocks above report it. `globaltimer_ns` is the absolute device
// timestamp at the common phase; `elapsed_ns` is this rank's own cudaEvent pair.
struct RankTiming {
    std::uint32_t rank      = 0;
    std::uint64_t globaltimer_ns = 0; // absolute, device-global, from %globaltimer
    std::uint64_t elapsed_ns     = 0; // this rank's own work, from its own event pair
};

// The spread across ranks: the START-to-start skew (did the ranks enter the phase together?) and
// the elapsed spread (did one rank do more work?). Both are "balance" questions with different
// answers, so they are separate numbers rather than one aggregate.
struct RankTimingSpread {
    bool ok = false;
    std::string reason;
    std::uint64_t start_skew_ns   = 0; // max - min of globaltimer_ns
    std::uint64_t elapsed_max_ns  = 0;
    std::uint64_t elapsed_min_ns  = 0;
    std::uint64_t elapsed_spread_ns = 0;
    double elapsed_imbalance      = 0.0; // max/min, 1.0 == perfectly balanced
};

[[nodiscard]] inline RankTimingSpread rank_timing_spread(const std::vector<RankTiming>& timings) {
    RankTimingSpread out;
    if (timings.empty()) {
        out.reason = "no ranks were timed";
        return out;
    }
    for (const RankTiming& timing : timings) {
        if (timing.elapsed_ns == 0) {
            out.reason = "rank " + std::to_string(timing.rank) +
                         " reported zero elapsed time: an unmeasured rank is not a balanced one";
            return out;
        }
        out.start_skew_ns   = std::max(out.start_skew_ns, timing.globaltimer_ns);
        out.elapsed_max_ns  = std::max(out.elapsed_max_ns, timing.elapsed_ns);
        if (out.elapsed_min_ns == 0 || timing.elapsed_ns < out.elapsed_min_ns) {
            out.elapsed_min_ns = timing.elapsed_ns;
        }
    }
    std::uint64_t earliest = timings.front().globaltimer_ns;
    for (const RankTiming& timing : timings) {
        earliest = std::min(earliest, timing.globaltimer_ns);
    }
    out.start_skew_ns      = out.start_skew_ns - earliest;
    out.elapsed_spread_ns  = out.elapsed_max_ns - out.elapsed_min_ns;
    out.elapsed_imbalance  = static_cast<double>(out.elapsed_max_ns) /
                            static_cast<double>(out.elapsed_min_ns);
    out.ok = true;
    out.reason.clear();
    return out;
}

// WHAT A TIMING NUMBER FROM THIS GUARD MAY BE COMPARED AGAINST, stated once so it is not
// re-derived: the same configuration run `Serialized` gives the per-rank cost; the same
// configuration run `Concurrent` gives that cost under one memory system's contention. Neither
// may be compared against a two-card number, and the aggregate tok/s of a virtual world is not a
// quantity at all -- the ranks are not doing a world's work in parallel, they are doing one
// world's work on one device.
[[nodiscard]] inline std::string rank_timing_spread_line(const RankTimingSpread& spread,
                                                         VirtualTimingMode mode) {
    if (!spread.ok) { return "rank timing unavailable: " + spread.reason; }
    return "rank timing [" + std::string(virtual_timing_mode_name(mode)) +
           "]: start skew " + std::to_string(spread.start_skew_ns) + " ns, elapsed " +
           std::to_string(spread.elapsed_min_ns) + ".." + std::to_string(spread.elapsed_max_ns) +
           " ns, imbalance " + std::to_string(spread.elapsed_imbalance) +
           "x (per-rank cost and balance ARE measurable under a virtual world; aggregate scaling "
           "is PENDING HARDWARE)";
}

} // namespace ninfer::multi
