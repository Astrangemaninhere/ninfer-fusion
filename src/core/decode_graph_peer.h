#pragma once

// decode_graph_peer.h -- THE CROSS-DEVICE FORK/JOIN GRAPH, and the gate that decides whether a
// captured decode step is admissible at all.
//
// THE ORDERING THIS FILE ENFORCES, AND WHY IT IS NOT A STYLE CHOICE
// ------------------------------------------------------------------
// EAGER IS THE IDENTITY REFERENCE; THE GRAPH IS AN OPTIMISATION BEHIND A FLAG. That is not a
// fallback for weak platforms -- it is how both TP2 forks on this box ship, and the reason is a
// MEASUREMENT, not caution:
//
//   * gfx906 (2x MI50, ROCm 6.4.1): "one hipGraph spans both MI50s ... (replay == eager)", and
//     yet capture and instantiate SUCCEED while replay is PATHOLOGICAL -- 0.21 t/s on the graph
//     against 13.44 t/s eager, ~7% of eager's throughput. They shipped eager.
//   * the donor (2x 5090, CUDA 13.1 / driver 580.178.04): one cross-device `cudaGraphExec`,
//     1888 nodes at tp2 against 640 at tp1, "+15% from capture", and the captured output is
//     "exactly identical to the eager dual-device path".
//   * 1Cat (`docs/design/sm70_deepseek_v4_tp8_graph_allreduce.md`) goes the other way for TP8 on
//     V100: "8 Tesla V100-SXM2 GPUs, TP8, CUDA Graph, and NO EAGER EXECUTION", because a
//     collective launched independently on each rank that "waits on device" removes the
//     collective graph-launch delay -- measured, "last kernel start after all inputs ready"
//     38.564 us (NCCL) -> 0.924 us (custom), TPOT -8.46%.
//
// Three implementations, two conclusions, ONE axis, and it is not "one graph vs one per rank":
// it is **does the stack accept a cross-device event wait inside a capture region, AND does it
// REPLAY at a usable rate**. So the rank axis has to be right with `use_cuda_graph == false`
// first, and `peer_graph_admission()` refuses a capture whose eager path has not been validated
// rather than trusting the caller to have done it.
//
// THE TWO FAILURES THIS GATE EXISTS TO CATCH, both recorded:
//   1. "two live captures fail with `cudaErrorStreamCaptureMerge`" -- one capture across the
//      world, not one per rank. So `live_captures` must be exactly 1.
//   2. "`cudaMemcpyPeerAsync` ... is rejected inside a stream capture region
//      (`cudaErrorStreamCaptureUnsupported`), which would make the whole tensor-parallel decode
//      program uncapturable" -- so a path that uses it cannot be captured at all, and the gate
//      refuses a plan that contains it instead of discovering it during `cudaStreamEndCapture`.
//
// Host-only at the top (no CUDA header), so the gate is exercisable on one GPU; the CUDA bridge
// is the guarded section at the bottom.
//
// WHAT THIS FILE DOES NOT DO: it does not edit `src/core/decode_graph.h`. That header captures
// ONE stream (`capture_segments`), which is the single-device path and must keep working. The
// extension it needs to adopt a fork/join graph built here is three lines and is reported rather
// than applied, because that file is shared and other lines are live in this tree.

#include "core/tp_transport.h"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace ninfer::multi {

// ---------------------------------------------------------------------------
// 1. THE SHAPE OF THE CAPTURE -- one capture, forked and joined by events
// ---------------------------------------------------------------------------

// Where this rank's stream joins the peer's and where it rejoins its own. The donor's own words
// for why these are EDGES and not NODES: "with both device streams enrolled in ONE capture -- the
// peer's stream joined to the origin's by an event fork -- the record/wait pairs become graph
// EDGES rather than nodes." That sentence is the strongest argument for the four-event protocol
// over any flag-based one: nothing about the ordering design has to change to be capturable.
struct PeerCapturePlan {
    // The stream the capture is BEGUN and ENDED on. Exactly one.
    int origin_rank = 0;
    // The stream enrolled by the fork. Exactly one, and it must join back before the end.
    int peer_rank = 1;
    // 1: one capture across the whole world. 2 fails with cudaErrorStreamCaptureMerge.
    std::uint32_t live_captures = 1;
    // The per-call ordering, UNCHANGED from the eager path. Equal to `tp_call_plan(op, policy)`
    // by construction and asserted so in the test: capturing must not re-order anything.
    std::vector<TpCallStep> steps;
    // The cross-stream edges the fork/join contributes: one pair per capture, not per call.
    std::uint32_t fork_edges = 1;
    std::uint32_t join_edges = 1;
};

[[nodiscard]] inline PeerCapturePlan peer_capture_plan(ReduceOp op, const SlotPolicy& policy) {
    PeerCapturePlan plan;
    plan.steps    = tp_call_plan(op, policy);
    plan.fork_edges = 1;
    plan.join_edges = 1;
    return plan;
}

// ---------------------------------------------------------------------------
// 2. THE ADMISSION GATE
// ---------------------------------------------------------------------------

// Everything the gate needs, as VALUES, so that each refusal can be produced by a test without a
// device. `eager_validated` is the field that carries the ordering above into the type: there is
// no way to ask for a capture without stating whether the eager path has already been shown to
// be correct, and a false there is a refusal rather than a warning.
struct PeerGraphRequest {
    bool use_cuda_graph = false;   // the flag: OFF by default, and off is a complete path
    bool eager_validated    = false; // the eager path has produced the right answer and was compared
    bool one_capture        = true;  // exactly one capture enrols every rank's stream
    bool uses_memcpy_peer_async = false; // must be false: it is rejected INSIDE a capture
    bool buffers_preallocated   = false; // events, staging and workspace exist BEFORE capture
    bool topology_decided       = false; // decide_topology() was consulted, not skipped
    // A cross-rank SPIN -- a kernel polling a peer's flag -- anywhere inside the captured region.
    // Refused because it is a DOCUMENTED HANG, not because it is impolite: gfx906's card-2 wedge.
    bool cross_rank_spin_in_capture = false;
    std::uint32_t world_size    = 1;
};

enum class PeerGraphVerdict : std::uint8_t {
    // use_cuda_graph == false: run eagerly. NOT a degraded mode -- the identity reference.
    EagerIdentity,
    // use_cuda_graph == true and every precondition holds.
    CaptureAdmissible,
    // Something is missing. Never a silent fallback to either of the above.
    Refused,
};

[[nodiscard]] inline std::string_view peer_graph_verdict_name(PeerGraphVerdict v) noexcept {
    switch (v) {
    case PeerGraphVerdict::EagerIdentity: return "eager-identity";
    case PeerGraphVerdict::CaptureAdmissible: return "capture-admissible";
    case PeerGraphVerdict::Refused: return "refused";
    }
    return "unknown-verdict";
}

struct PeerGraphAdmission {
    PeerGraphVerdict verdict = PeerGraphVerdict::EagerIdentity;
    std::string reason;
    // The path the caller must actually take. For every `Refused` this is the EAGER path, and
    // saying so is the point: a refusal here never means "run nothing".
    bool take_eager_path = true;
};

[[nodiscard]] inline PeerGraphAdmission peer_graph_admission(const PeerGraphRequest& request) {
    PeerGraphAdmission out;
    if (request.world_size <= 1) {
        out.verdict          = PeerGraphVerdict::EagerIdentity;
        out.take_eager_path  = true;
        out.reason =
            "world_size " + std::to_string(request.world_size) +
            ": there is no cross-device graph to build. A single-rank run is the eager path with "
            "nothing to fork to, and no capture is issued.";
        return out;
    }
    if (!request.use_cuda_graph) {
        out.verdict         = PeerGraphVerdict::EagerIdentity;
        out.take_eager_path = true;
        out.reason.clear();
        return out;
    }
    // use_cuda_graph is on. Every one of these is a named refusal, not a degradation.
    out.verdict         = PeerGraphVerdict::Refused;
    out.take_eager_path = true;
    if (!request.eager_validated) {
        out.reason =
            "capture requested before the eager path was validated: the eager path is the identity "
            "REFERENCE, not a fallback. Both TP2 forks on this box ship eager and measure the graph "
            "as an optimisation on top of it (gfx906 measured 0.21 t/s under graph replay against "
            "13.44 t/s eager and shipped eager), so a captured step whose eager twin has not "
            "produced the right answer has nothing to be compared against. Run with "
            "use_cuda_graph=false first.";
        return out;
    }
    if (request.uses_memcpy_peer_async) {
        out.reason =
            "the plan contains cudaMemcpyPeerAsync: it is rejected INSIDE a stream capture region "
            "with cudaErrorStreamCaptureUnsupported (measured on CUDA 13.1 / driver 580.178.04), "
            "which would make the whole tensor-parallel decode program uncapturable. The transport "
            "must be the UVA cudaMemcpyAsync(DeviceToDevice) form, which is one code path covering "
            "both the direct PCIe and the driver-staged transfer.";
        return out;
    }
    if (!request.one_capture) {
        out.reason =
            "more than one capture is live: two live captures across two ranks fail with "
            "cudaErrorStreamCaptureMerge. Enrol every rank's stream in ONE capture, with the "
            "peer's stream joined to the origin's by an event fork and joined back before "
            "cudaStreamEndCapture, so that the record/wait pairs become graph EDGES rather than "
            "nodes.";
        return out;
    }
    if (!request.topology_decided) {
        out.reason =
            "the capture was requested without a topology decision: the transport's shape was "
            "never asked for. Call decide_topology() and carry its result -- a capture built on "
            "an unasked topology encodes whichever path the driver happened to take at capture "
            "time, which is not a property the graph can be replayed against.";
        return out;
    }
    if (!request.buffers_preallocated) {
        out.reason =
            "events, staging or workspace are allocated per call: cudaEventCreate and cudaMalloc "
            "are NOT capturable, so a per-call allocation anywhere on the collective path is a "
            "capture failure rather than a performance problem. Create the events and the staging "
            "once, per (rank, peer) pair, before the capture begins.";
        return out;
    }
    if (request.cross_rank_spin_in_capture) {
        out.reason =
            "the captured region contains a CROSS-RANK SPIN (a kernel polling a peer's flag). This "
            "is refused as a KNOWN HANG rather than as a style preference: gfx906's "
            "docs/gfx906/TP2-SLICES.md documents a card-2 WEDGE caused by a spin-wait inside a "
            "per-device graph, which is why its flag-sync stays opt-in under serve. A captured "
            "graph cannot yield, cannot time out and cannot be interrupted by the host, so a peer "
            "that never satisfies the polled flag wedges the graph AND the card -- and a cold power "
            "cycle was the only recovery in that post-mortem. Use the EVENT order instead: the "
            "record/wait pairs become graph EDGES, which the scheduler resolves, so the donor's "
            "ordering is signal-free by construction and needs no spin.";
        return out;
    }
    out.verdict         = PeerGraphVerdict::CaptureAdmissible;
    out.take_eager_path = false;
    out.reason.clear();
    return out;
}

// The one thing capturing must NOT change, stated as a checkable equality rather than a promise:
// the eager and captured orderings are the same sequence of steps. It is the whole reason the
// donor's ordering design was adopted as written -- nothing had to be redesigned to be capturable.
[[nodiscard]] inline bool capture_preserves_ordering(ReduceOp op, const SlotPolicy& policy) {
    return peer_capture_plan(op, policy).steps == tp_call_plan(op, policy);
}

// ---------------------------------------------------------------------------
// 2b. THE TWO THINGS THAT CANNOT COEXIST WITH A CAPTURED CROSS-RANK STEP
// ---------------------------------------------------------------------------
//
// THE FRAME IS "最后肯定是所有机制都开" -- the end state is every mechanism on at once -- so the
// job here is to name what breaks when they are, rather than to discover it in a hang. Two items,
// both from the record and neither hypothetical:
//
// (C6a) A CROSS-RANK SPIN INSIDE A CAPTURE IS A KNOWN HANG, NOT A RISK. gfx906's own
//   `docs/gfx906/TP2-SLICES.md` documents a **card-2 wedge** caused by a spin-wait inside a
//   per-device graph, which is why its flag-sync stays opt-in under serve and why eager is the
//   identity reference there. A captured graph cannot yield, cannot time out and cannot be
//   interrupted by the host, so a peer that never satisfies the polled flag wedges the graph AND
//   the card. The donor's signal-free design avoids this for free: its ordering is
//   `cudaStreamWaitEvent` graph EDGES, which the scheduler resolves, rather than a flag a kernel
//   polls. So the gate refuses a captured plan that contains a spin, and the reason names the
//   wedge -- this is the single most expensive thing to get wrong on this axis.
//
// (C6b) THE GRAPH NODE ALLOWANCE IS A JOINT BUDGET. The donor measured **1888 nodes at tp2**
//   against **640 at tp1**, and the end state adds MTP and the cold tier on top of the same
//   graph. `use_cuda_graph` therefore cannot be decided by the rank axis alone: the allowance is
//   shared. It is a budgeted quantity with the measured numbers attached, so "every mechanism on"
//   has a place where its cost is summed rather than discovered.
//
// The DEFAULT allowance is the donor's measured tp2 count, and the reason is the honest one: it
// is the largest sharded-plus-captured graph anyone in the record has actually instantiated on
// this class of card. So **tp2 alone exactly spends it and adds nothing else for free** -- which
// is the finding, not an inconvenience. A caller whose graph is smaller may raise `allowance`
// explicitly, and doing so is a decision rather than a default.
inline constexpr std::uint32_t kTp1GraphNodes = 640;  // donor, tp1, 2x 5090
inline constexpr std::uint32_t kTp2GraphNodes = 1888; // donor, tp2, 2x 5090
inline constexpr std::uint32_t kGraphNodeAllowanceDefault = kTp2GraphNodes;

struct GraphNodeBudget {
    std::uint32_t world_size   = 1;
    std::uint32_t mtp_nodes    = 0; // whatever MTP contributes to the same graph
    std::uint32_t cold_nodes   = 0; // the cold tier's own nodes in the same graph
    std::uint32_t allowance    = kGraphNodeAllowanceDefault;

    // The rank axis's own cost, by the donor's measurement rather than by a guess.
    [[nodiscard]] std::uint32_t rank_axis_nodes() const noexcept {
        return world_size >= 2 ? kTp2GraphNodes : kTp1GraphNodes;
    }
    [[nodiscard]] std::uint32_t planned() const noexcept {
        return rank_axis_nodes() + mtp_nodes + cold_nodes;
    }
};

// The refusal when the joint plan does not fit. It names the three contributors separately,
// because "the graph is too big" sends the reader to the wrong file.
[[nodiscard]] inline std::string graph_node_allowance_refusal(const GraphNodeBudget& budget) {
    const std::uint32_t planned = budget.planned();
    if (planned <= budget.allowance) { return {}; }
    return "the joint capture plan needs " + std::to_string(planned) + " nodes against an " +
           std::to_string(budget.allowance) + "-node allowance: " +
           std::to_string(budget.rank_axis_nodes()) + " from the rank axis (the donor measured " +
           std::to_string(kTp1GraphNodes) + " at tp1 and " + std::to_string(kTp2GraphNodes) +
           " at tp2), " + std::to_string(budget.mtp_nodes) + " from MTP and " +
           std::to_string(budget.cold_nodes) +
           " from the cold tier. The allowance is SHARED, so the rank axis cannot decide "
           "use_cuda_graph alone -- and every mechanism being on at once is the stated end state, "
           "not the exception. A rank axis that spends the whole allowance leaves MTP and the cold "
           "tier nowhere to live.";
}


//
// The fork/join itself. It is ~20 lines of CUDA and it needs no library: both streams are
// enrolled in ONE capture because `cudaStreamWaitEvent` PROPAGATES the capture to the stream it
// waits from, and the capture is ended only after the peer's stream has waited back on the
// origin's. The build must be told the join has happened; ending the capture while the peer's
// stream is still enrolled is the `cudaErrorStreamCaptureMerge`/`...Unjoined` failure.

#if defined(__CUDACC__)

struct PeerForkJoin {
    cudaEvent_t fork = nullptr; // origin -> peer
    cudaEvent_t join = nullptr; // peer -> origin

    bool create() noexcept {
        // DisableTiming: these are ordering events, never timed, and a timing event costs a
        // device-side timestamp on every record.
        return cudaEventCreateWithFlags(&fork, cudaEventDisableTiming) == cudaSuccess &&
               cudaEventCreateWithFlags(&join, cudaEventDisableTiming) == cudaSuccess;
    }
    void destroy() noexcept {
        if (fork != nullptr) { cudaEventDestroy(fork); fork = nullptr; }
        if (join != nullptr) { cudaEventDestroy(join); join = nullptr; }
    }
};

// Begin ONE capture on `origin`, fork to `peer`, run both bodies, rejoin, end on `origin`.
// `both_bodies` receives the two streams so the caller can enqueue its own work on each; that is
// the only part that belongs to the caller.
//
// `cudaStreamCaptureModeThreadLocal` matches `src/core/decode_graph.cpp:65` so that a captured
// peer block which touches another thread's stream is a capture ERROR rather than a silent join.
inline cudaError_t capture_fork_join(cudaStream_t origin, cudaStream_t peer,
                                     const PeerForkJoin& events,
                                     const std::function<void(cudaStream_t, cudaStream_t)>& both_bodies,
                                     cudaGraph_t* out_graph) noexcept {
    cudaError_t err = cudaStreamBeginCapture(origin, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) { return err; }
    // The fork: the origin's current position is published, and the peer's stream joins the
    // capture by WAITING on it. Everything the peer enqueues from here is in the same capture.
    err = cudaEventRecord(events.fork, origin);
    if (err != cudaSuccess) { return err; }
    err = cudaStreamWaitEvent(peer, events.fork, 0);
    if (err != cudaSuccess) { return err; }
    both_bodies(origin, peer);
    // The join: the peer's position is published and the origin waits for it, so the capture is
    // complete on the origin's stream when it is ended.
    err = cudaEventRecord(events.join, peer);
    if (err != cudaSuccess) { return err; }
    err = cudaStreamWaitEvent(origin, events.join, 0);
    if (err != cudaSuccess) { return err; }
    return cudaStreamEndCapture(origin, out_graph);
}

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// 4. WHAT IS NOT HERE
// ---------------------------------------------------------------------------
//
//   * THE ADOPTION OF THIS GRAPH BY `src/core/decode_graph.h`. That header captures one stream
//     and is shared; the three lines it needs are `cudaGraph_t` adoption, reported not applied.
//   * ANY TIMING OF THE CAPTURED PATH. The capture's +15% node count and the 1888-vs-640 node
//     figure are the donor's numbers on two 5090s, not ours: this box has one GPU, so a captured
//     cross-device decode step is `pending hardware` and is labelled so once.
//   * ANY KIND OF PERFORMANCE CLAIM. What is provable here is the ordering, the admission gate,
//     and that capturing preserves the ordering -- which is exactly what a captured graph
//     replays, and exactly what the rank axis needs to be right about.

} // namespace ninfer::multi
