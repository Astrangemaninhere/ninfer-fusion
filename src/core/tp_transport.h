#pragma once

// tp_transport.h -- THE COLLECTIVE SEAM: which REDUCTIONS the rank axis needs, which of them a
// sum-only all-reduce cannot express, the topology decision, and the slot protocol that makes
// the sequence capturable.
//
// WHY THIS FILE EXISTS, AND WHY IT IS 1Cat-SHAPED AND NOT ninfer-DONOR-SHAPED
// -------------------------------------------------------------------------
// Two community implementations of TP2 exist for this engine's neighbourhood and they are NOT
// interchangeable. The record, read from source rather than from a description:
//
//   DONOR (`wamansou/ninfer-tp2-1m @ tp2-yarn-1m`, `JCraigWasTaken/ninfer-gfx906 @ gfx906-port`):
//     NOTE 2026-09-20: `tp2-yarn-1m` in that line is the DONOR BRANCH, not a documentation path.
//     The tree cites the record that IS here, `docs/gfx906/TP2-SLICES.md`, and names it from
//     core/virtual_device.h section 0, core/n_dim_slice.h and core/decode_graph_peer.h; no header
//     here turns that branch name back into a doc path.
//     two ops, `allreduce_sum` (a plain SUM, in place) and `allgather_rows` (a pure RELOCATION,
//     "exact, no arithmetic"). Pull-based: every transfer is `cudaMemcpyAsync(..., D2D, ...)`
//     over UVA on the receiving rank's own stream, four events per call, and NO peer-access
//     branch: "when it does not [grant peer access], CUDA transparently stages the same copy
//     through host memory. Both paths are correct ... only bandwidth and latency differ." So the
//     transport is chosen SILENTLY BY THE DRIVER.
//
//   1Cat (`1CatAI/1Cat-vLLM @ fe67339`, read on this box at
//     /mnt/c/Users/User/Documents/ziqinzhang/1Cat-vLLM):
//     `gpu_p2p_access_check(src, tgt)` -- `vllm/distributed/device_communicators/all_reduce_utils.py:312`,
//     called per peer at `custom_all_reduce.py:51` -- DECIDES the path and the decision is carried
//     into the kernel as `CustomAllreduce::fully_connected_`
//     (`csrc/custom_all_reduce.cuh:139`), asserted at every push call site
//     (`csrc/custom_all_reduce.cu:820,858,943,915`). The transport itself is IPC-mapped peer
//     memory with system-scope release/acquire SIGNAL slots
//     (`custom_all_reduce.cuh:440,451`: `st.release.sys.global.u32` / `ld.acquire.sys.global.u32`),
//     a per-block signal grid (`sg.signals[peer]->start[blockIdx.x][rank]`, `:500-502`), and it
//     is CAPTURE-AWARE: `cudaStreamIsCapturing` at `:199` switches the kernel argument to a
//     graph-unregistered slot whose peer pointers are patched AFTER capture from exchanged IPC
//     handles (`get_graph_buffer_ipc_meta`, `:1735`).
//
// Our own plan (`docs/maintainer/multi-device-and-shard-plan.md:185-187`) asks for THREE things
// and the donor supplies one and a half of them:
//
//   1. "an all-reduce over the attention combine triple (row max, row sumexp, partial output)
//      per layer"  -- the donor has a SUM. See section 2 for exactly what that costs.
//   2. "an all-reduce AND of the eviction admit verdict"  -- the donor has no boolean op at all,
//      and NEITHER DOES 1Cat (its `allreduce` dispatches float/half/bfloat16 only,
//      `csrc/custom_all_reduce.cu:24-28`; `tests/distributed/test_custom_all_reduce.py` pins
//      `(torch.int8, False)` and `(torch.float8_e4m3fn, False)`). This one is OURS to specify.
//   3. "`cudaDeviceEnablePeerAccess` / `cudaMemcpyPeer` (or an NCCL equivalent) plus a TOPOLOGY
//      DECISION"  -- the donor deliberately has no branch ("no caller and no test needs a
//      peer-access branch"); 1Cat has the branch, named and gated. The plan asks for 1Cat's
//      shape.
//
// ⚠️ A CLAIM THIS FILE REFUTES WITH THE SOURCE, because it was made to me and the record says
// otherwise: 1Cat does NOT fuse the attention combine into its all-reduce. `CustomAllreduce::allreduce`
// is a plain sum; the combine is a SEPARATE named op one layer up
// (`vllm/v1/attention/ops/common.py:212 cp_lse_ag_out_rs`, `:237 cp_lse_ag_out_ar`, used at
// `vllm/model_executor/layers/attention/mla_attention.py:790` and
// `vllm/v1/attention/backends/flashinfer.py:223`), and it is TWO collectives -- an LSE
// all-gather and an output reduce-scatter/all-reduce -- with a local exp2 weight applied
// between them. What 1Cat's AR IS fused WITH is a NORM, not the combine:
// `minimax_allreduce_rms` / `minimax_allreduce_rms_qk` (`csrc/minimax_reduce_rms_kernel.cu`,
// `csrc/torch_bindings.cpp:959,968`), `sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather`
// (`torch_bindings.cpp:1021`), `sm70_qwen38_hc_{down,output,up_mix}_allgather`
// (`torch_bindings.cpp:1032,1041,1046`). The conclusion below does not change -- 1Cat's SHAPE is
// still the one to adopt -- but the reason is the op SET and the topology decision, not a
// fusion that is not in the code.
//
// THE PROVENANCE OF EACH CLAIM ABOVE IS A FILE AND A LINE, and the tests in
// tests/test_tp_transport.cpp assert the arithmetic and the protocol, not the provenance.
//
// WHAT THIS FILE IS NOT. It is not a transport. Cross-card communication is not this program's
// concern: the transport is somebody else's component and this engine has to be COMPATIBLE with
// it. So this file states (a) the interface, (b) the sharding contract the caller must satisfy,
// (c) the topology DECISION, and (d) a minimal working fallback -- because "somebody else's
// component" has twice been absent or broken: gfx906 measured RCCL 2.30 "cannot complete a
// 2-card collective on this host at all", and both forks hand-rolled a transport. Cost is NOT
// stated here and is NOT apologised for: it is not this component's number to know.
//
// Host-only by construction (like core/shard_plan.h and core/shard_rank_axis.h): no CUDA header,
// no device query, no collective at the top level, so every decision above is exercisable on
// one GPU. The CUDA seam is the guarded section at the bottom.

#include "core/shard_plan.h"
#include "core/shard_rank_axis.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::multi {

// ---------------------------------------------------------------------------
// 0. THE TOPOLOGY DECISION -- asked, not inferred
// ---------------------------------------------------------------------------

// What the probe was allowed to learn, as a value rather than as an environment read, so the
// decision is a pure function and the test can drive every cell of it.
//
// `can_access_peer[a][b]` is `cudaDeviceCanAccessPeer(a, b)`. It is deliberately measured in BOTH
// directions and stored separately: the donor's `enable_peer_access()` probes both and enables
// only when both report support, because "asymmetric support is not a usable transport for a
// symmetric collective". 1Cat's `_can_p2p` asks every peer and returns false on the FIRST
// failure (`custom_all_reduce.py:44-53`). Same conclusion, and this struct is shaped so that the
// asymmetry is visible instead of averaged away.
struct PeerAccessMatrix {
    std::uint32_t device_count = 0;
    // Row-major, device_count x device_count. The diagonal is meaningless and is ignored.
    std::vector<std::uint8_t> can_access_peer;

    [[nodiscard]] bool at(std::uint32_t from, std::uint32_t to) const noexcept {
        if (device_count == 0 || from >= device_count || to >= device_count) { return false; }
        const std::size_t index = static_cast<std::size_t>(from) * device_count + to;
        return index < can_access_peer.size() && can_access_peer[index] != 0;
    }

    // A world is a clique only if EVERY ordered pair of DISTINCT ranks can reach each other.
    // One direction is not enough and 1Cat's `fully_connected_` means exactly this.
    [[nodiscard]] bool fully_connected(std::uint32_t world_size) const noexcept {
        if (world_size < 2 || world_size > device_count) { return false; }
        for (std::uint32_t a = 0; a < world_size; ++a) {
            for (std::uint32_t b = 0; b < world_size; ++b) {
                if (a == b) { continue; }
                if (!at(a, b)) { return false; }
            }
        }
        return true;
    }

    // The first ordered pair that is missing, for a refusal that names the pair instead of
    // saying "not fully connected".
    [[nodiscard]] std::string first_missing_edge(std::uint32_t world_size) const {
        for (std::uint32_t a = 0; a < world_size; ++a) {
            for (std::uint32_t b = 0; b < world_size; ++b) {
                if (a == b) { continue; }
                if (!at(a, b)) {
                    return std::to_string(a) + "->" + std::to_string(b);
                }
            }
        }
        return {};
    }
};

// The three outcomes. `Unavailable` is a THIRD value rather than `false`, because "there is no
// transport" and "the transport is the staged one" are different amounts of code and different
// failure modes -- the same reason `CombineOutcome` and `RouteOutcome` have three values.
enum class TopologyDecision : std::uint8_t {
    // One device. There is nothing to talk to; every collective is the identity and NO
    // COLLECTIVE MUST BE ISSUED.
    Unavailable,
    // Both directions reachable for every ordered pair: the direct peer path is admissible.
    DirectPeer,
    // Reachable in at least one direction but not as a full clique, or not reported at all.
    // The staged path over host memory is a SUPPORTED TRANSPORT, not an error -- this is the
    // donor's own stance and it is correct. What is NOT acceptable is reaching it silently:
    // that is what this decision exists to prevent.
    HostStaged,
};

[[nodiscard]] inline std::string_view topology_decision_name(TopologyDecision d) noexcept {
    switch (d) {
    case TopologyDecision::Unavailable: return "unavailable";
    case TopologyDecision::DirectPeer: return "direct-peer";
    case TopologyDecision::HostStaged: return "host-staged";
    }
    return "unknown-topology";
}

struct TopologyReport {
    TopologyDecision decision = TopologyDecision::Unavailable;
    std::uint32_t world_size  = 1;
    // Non-empty whenever the decision is not a plain DirectPeer, and actionable: it names the
    // missing edge, the device count, or the fact that both directions were not asked.
    std::string reason;
};

// The decision. `probe_completed` is a separate input because a probe that FAILED (NVML down,
// driver error) must not be read as "no peer access": that is the same false-negative shape as
// reading an nvidia-smi timeout as "GPU idle".
[[nodiscard]] inline TopologyReport decide_topology(const PeerAccessMatrix& matrix,
                                                    std::uint32_t world_size,
                                                    bool probe_completed) {
    TopologyReport report;
    report.world_size = world_size;
    if (!probe_completed) {
        report.decision = TopologyDecision::HostStaged;
        report.reason =
            "the peer-access probe did not complete (device enumeration or cudaDeviceCanAccessPeer "
            "failed): a probe that failed is not a peer path that was refused. Falling back to the "
            "staged transport and saying so, rather than reading an absent answer as 'no'.";
        return report;
    }
    if (world_size <= 1 || matrix.device_count <= 1) {
        report.decision = TopologyDecision::Unavailable;
        report.reason =
            "one device and world_size " + std::to_string(world_size) +
            ": there is no peer to reach, so every collective is the identity and none may be "
            "issued. This is not a degraded transport; it is the single-rank contract.";
        return report;
    }
    if (matrix.fully_connected(world_size)) {
        report.decision = TopologyDecision::DirectPeer;
        report.reason.clear();
        return report;
    }
    report.decision = TopologyDecision::HostStaged;
    const std::string edge = matrix.first_missing_edge(world_size);
    report.reason =
        "world_size " + std::to_string(world_size) + " over " +
        std::to_string(matrix.device_count) + " device(s) is not a peer ACCESS clique" +
        (edge.empty() ? std::string{} : " (first unreachable ordered pair " + edge + ")") +
        ": the staged path over host memory is a supported transport and is being used, and it is "
        "named here rather than chosen silently by the driver.";
    return report;
}

// The donor's stance, kept as a named value so the difference is a decision and not an opinion:
// with no branch anywhere, the SAME call runs over direct PCIe when the driver grants peer
// access and over a driver-staged host bounce when it does not ("both paths are correct and
// stream-ordered, so no caller and no test needs a peer-access branch"). It is not wrong. It is
// also not what our plan asks for, and it is unmeasurable from inside the engine: a caller
// cannot tell which one it got.
enum class TopologyPolicy : std::uint8_t {
    SilentDriverChoice, // the donor: one code path, the driver decides, the caller cannot tell
    ExplicitDecision,   // 1Cat and this file: the path is a decision with a name
};

[[nodiscard]] inline std::string_view topology_policy_name(TopologyPolicy policy) noexcept {
    switch (policy) {
    case TopologyPolicy::SilentDriverChoice: return "silent-driver-choice";
    case TopologyPolicy::ExplicitDecision: return "explicit-decision";
    }
    return "unknown-policy";
}

// ---------------------------------------------------------------------------
// 1. THE REDUCTION SET -- the part the plan actually asks for
// ---------------------------------------------------------------------------

// The four reductions the rank axis needs, each named as the FUNCTION it computes rather than as
// the op that might implement it. A boolean AND is not "a sum with a check", and a weighted
// sum-exp merge is not "a sum with extra steps"; the difference is visible in section 2.
enum class ReduceOp : std::uint8_t {
    Sum,             // sum_r x_r                       -- the row-parallel projection output
    SumInPlace,      // same, local buffer               -- what the donor's allreduce_sum is
    AttentionCombine,// merge_r (m_r, l_r, o_r)         -- softmax combine triple (max, sumexp, o)
    AllGatherRows,   // relocate along ne[1]            -- a pure movement, no arithmetic
    AndBool,         // AND_r b_r                       -- the eviction admit verdict
};

[[nodiscard]] inline std::string_view reduce_op_name(ReduceOp op) noexcept {
    switch (op) {
    case ReduceOp::Sum: return "sum";
    case ReduceOp::SumInPlace: return "sum-in-place";
    case ReduceOp::AttentionCombine: return "attention-combine";
    case ReduceOp::AllGatherRows: return "allgather-rows";
    case ReduceOp::AndBool: return "and-bool";
    }
    return "unknown-reduce";
}

// WHO PROVIDES IT. `DonorExact` means the donor's two ops express it directly; `DonorStaged`
// means the caller can build it out of them but has to supply the intermediate steps and know a
// layout precondition; `NeitherReference` means no TP2 reference on this box has the op.
enum class ReduceOpProvenance : std::uint8_t {
    DonorExact,
    DonorStaged,
    NeitherReference,
};

[[nodiscard]] inline std::string_view reduce_op_provenance_name(ReduceOpProvenance p) noexcept {
    switch (p) {
    case ReduceOpProvenance::DonorExact: return "donor-exact";
    case ReduceOpProvenance::DonorStaged: return "donor-staged";
    case ReduceOpProvenance::NeitherReference: return "neither-reference";
    }
    return "unknown-provenance";
}

[[nodiscard]] inline ReduceOpProvenance reduce_op_provenance(ReduceOp op) noexcept {
    switch (op) {
    case ReduceOp::Sum:
    case ReduceOp::SumInPlace:
    case ReduceOp::AllGatherRows:
        return ReduceOpProvenance::DonorExact;
    case ReduceOp::AttentionCombine:
        // Expressible with the donor's ops, but ONLY with the caller doing the weighting
        // between two of them and arranging the LSE layout so the gather axis is ne[1].
        return ReduceOpProvenance::DonorStaged;
    case ReduceOp::AndBool:
        // Not in the donor (no boolean op) and not in 1Cat either: `CustomAllreduce::allreduce`
        // dispatches at::ScalarType::Float/Half/BFloat16 only
        // (csrc/custom_all_reduce.cu:24-28), and the dispatcher test pins int8 and
        // float8_e4m3fn to "not supported" (tests/distributed/test_custom_all_reduce.py).
        return ReduceOpProvenance::NeitherReference;
    }
    return ReduceOpProvenance::NeitherReference;
}

// ---------------------------------------------------------------------------
// 2. WHY A SUM-ONLY ALL-REDUCE IS NOT ENOUGH -- computed, not asserted
// ---------------------------------------------------------------------------

// The cost of expressing each reduction with nothing but `allreduce_sum` + `allgather_rows`.
// This is the number that decides the interface, so it is a function and the test checks every
// row of it against the arithmetic in section 2's two demonstrations.
struct SumOnlyCost {
    // Collectives that must cross the world boundary, in order. "ag" = allgather_rows,
    // "ar" = allreduce_sum.
    std::vector<std::string> collectives;
    // Steps that happen between the collectives and change the DATA the next collective sends.
    // This is the part that a fused op would remove.
    std::uint32_t local_steps = 0;
    // True when the caller must arrange a layout the op cannot fix up (the donor's allgather_rows
    // relocates along ne[1] and "does not transpose").
    bool caller_layout_precondition = false;
    // True when a step needs a host round trip. A sum-only path over DEVICE-side ops does not,
    // and this field is here so that "2-3 round trips with host-side fixup" can be checked
    // rather than repeated: for the combine it is FALSE, because the weight is a device-side
    // exp2/sum over a device-resident LSE.
    bool host_fixup_required = false;
};

[[nodiscard]] inline SumOnlyCost sum_only_cost(ReduceOp op) {
    SumOnlyCost cost;
    switch (op) {
    case ReduceOp::Sum:
    case ReduceOp::SumInPlace:
        cost.collectives = {"ar"};
        return cost;
    case ReduceOp::AllGatherRows:
        cost.collectives = {"ag"};
        return cost;
    case ReduceOp::AttentionCombine:
        // 1Cat's own shape, which is also the minimum for a sum-only toolbox: gather the LSE,
        // weight locally by exp2(lse_r - lse_max), then reduce-sum the weighted outputs. TWO
        // collectives. `cp_lse_ag_out_rs` (common.py:212) is exactly this, with `reduce_scatter`
        // in place of the final `all_reduce` that `cp_lse_ag_out_ar` (:237) uses.
        //
        // The reason a SINGLE sum cannot do it is not the summation: it is that the weight
        // `exp(m_r - m*)` depends on the GLOBAL max, which is not known to any rank before the
        // gather. A caller that sums first and rescales afterwards has already lost the
        // information -- the same defect as MULTIGPU's own first `combine_merge`, which
        // multiplied `o` by the sumexp's weight instead of the max's and produced -0.8314
        // against a reference of -0.5354 (core/shard_rank_axis.h:537).
        cost.collectives           = {"ag", "ar"};
        cost.local_steps           = 2; // (1) lse_max + exp2 weights  (2) final normalise by l
        cost.caller_layout_precondition = true;
        cost.host_fixup_required   = false;
        return cost;
    case ReduceOp::AndBool:
        // A sum CAN encode an AND, and only on the domain {0,1}, which nothing enforces. The
        // demonstration is in `and_encoding_counterexample()` below and the test asserts both
        // directions: it agrees on every well-formed verdict, and it DISAGREES the moment a rank
        // reports anything outside {0,1}. That is the whole argument for a named boolean op: the
        // name makes the out-of-domain input unrepresentable instead of silently wrong.
        cost.collectives = {"ar"};
        return cost;
    }
    return cost;
}

// The counterexample, as a value: `sum_encoding_agrees` is false exactly when the sum encoding
// and the boolean AND disagree, and `witness` says why.
struct AndEncodingWitness {
    bool sum_encoding_agrees = true;
    std::string witness;
};

// `per_rank_admit` is what each rank would send to a sum-encoded admit reduction. The boolean
// function the plan asks for is `global_admit()` (core/shard_plan.h:450): the page leaves the
// device only if EVERY rank can take it.
[[nodiscard]] inline AndEncodingWitness and_encoding_counterexample(
    const std::vector<std::uint32_t>& per_rank_admit) {
    AndEncodingWitness out;
    if (per_rank_admit.empty()) {
        out.sum_encoding_agrees = true;
        return out;
    }
    std::uint32_t sum = 0;
    bool all_nonzero  = true;
    for (std::uint32_t v : per_rank_admit) {
        sum += v;
        if (v == 0) { all_nonzero = false; }
    }
    // The sum encoding under test: "admit iff the ranks' values sum to exactly world_size".
    const bool sum_encoding = (sum == per_rank_admit.size());
    out.sum_encoding_agrees = (sum_encoding == all_nonzero);
    if (!out.sum_encoding_agrees) {
        std::string values;
        for (std::size_t i = 0; i < per_rank_admit.size(); ++i) {
            if (i != 0) { values += ","; }
            values += std::to_string(per_rank_admit[i]);
        }
        out.witness =
            "per-rank admits {" + values + "}: the sum is " + std::to_string(sum) +
            " against world_size " + std::to_string(per_rank_admit.size()) +
            ", so the sum encoding says " + (sum_encoding ? "ADMIT" : "REFUSE") +
            ", while AND over the ranks says " + (all_nonzero ? "ADMIT" : "REFUSE") +
            ". Every rank admitted, but one of them reported a value outside {0,1} -- so the sum "
            "encoding's correctness rests on a precondition the type does not carry.";
    }
    return out;
}

// A named boolean reduction, with the out-of-domain input made unrepresentable by the type.
// `bool` cannot hold 2, which is the entire point: this is the fix the counterexample argues
// for, rather than a runtime check the caller must remember.
[[nodiscard]] inline bool all_reduce_and(const std::vector<bool>& per_rank_admit) noexcept {
    for (bool admit : per_rank_admit) {
        if (!admit) { return false; }
    }
    return true;
}

// The single-rank and no-collective cases, taken by construction rather than by a flag: the
// identity of an AND is TRUE, and of the combine is the OTHER operand. Returns whether a
// collective had to be issued at all, so the caller can prove that world_size == 1 issues none.
[[nodiscard]] inline bool and_reduce_is_identity(std::uint32_t world_size) noexcept {
    return world_size <= 1;
}

// The combine side of the same question, expressed once so both axes agree: with world_size 1
// the LOCAL partial IS the whole row and no arithmetic may touch it.
[[nodiscard]] inline CombineOutcome combine_outcome_for(const ShardPlan& plan) noexcept {
    if (!plan.ok) { return CombineOutcome::MergeRequired; } // a failed plan must not read as free
    if (plan.axis == ParallelAxis::None) { return CombineOutcome::None; }
    return attention_combine_outcome(plan.axis);
}

// ---------------------------------------------------------------------------
// 3. THE SLOT PROTOCOL -- the measured failure, reproduced as a scan
// ---------------------------------------------------------------------------

// A collective call sequence is capturable only if the buffers and the SIGNALS it reuses are
// safe to reuse while a slower rank is still reading the previous occupant. This is the
// write-after-read hazard, and on 1Cat it is not a hypothesis: the record
// (1Cat's `docs/design/sm70_deepseek_v4_tp8_graph_allreduce.md`, "Graph-Skew Stability Fix" and
// "Rejected Paths") measures three configurations and only one of them survives:
//
//   one signal/partial slot        -> "Full model stopped after 22 tokens under larger graph skew"
//   partial double-buffered only   -> "87-call graph microbenchmark entered synchronization spin"
//   BOTH double-buffered + ack     -> "8,700 collectives, 17.099 us/call, complete"
//
// The donor reaches the same property from the other side: its four events per call make the
// second pair a write-after-read barrier, and the post-condition is that "an unbounded sequence
// of calls sharing the same buffers, staging, and PeerEvents needs NO host synchronization
// between calls -- which is exactly what a captured CUDA graph replays."

// The two independent slot spaces. They are separate fields because the measured middle
// configuration -- data double-buffered, signals not -- is a REAL configuration that a caller
// will otherwise build by accident, and it is the one that deadlocks.
struct SlotPolicy {
    std::uint32_t data_slots   = 1u; // buffers a rank writes its own inputs into
    std::uint32_t signal_slots = 1u; // readiness/ack slots peers write and read
    // Whether a rank may reuse slot s for call k + data_slots only after observing the peer's
    // acknowledgement that call k - data_slots's slot was consumed. 1Cat's accepted protocol
    // does this ("acknowledges local-input consumption in parallel with the pair exchange").
    bool acknowledge_before_reuse = false;

    [[nodiscard]] bool valid() const noexcept {
        return data_slots >= 1u && signal_slots >= 1u;
    }
};

struct SlotHazard {
    bool hazard = false;
    std::uint32_t call_index     = 0; // the call whose publication is premature
    std::uint32_t slot           = 0;
    std::string what;                 // which space, and who is reading
};

// The scan. `lead` is how many calls the faster rank is ahead of the slower one; it is an INPUT
// because the hazard is a statement about skew, and the whole point of the protocol is that it
// must hold for a lead the caller cannot bound. `calls` is the length of the captured sequence.
//
// The model is deliberately the crudest one that can express all three measured outcomes, and it
// is one inequality: when the faster rank publishes call k into slot k mod slots, the slower rank
// is at this moment READING call k - lead -- the same slot -- whenever
// `k - lead >= k - slots`, i.e. whenever `lead >= slots`. So the rule is
//
//     a slot space of S can absorb a skew of up to S - 1 calls, and no more,
//     unless the reuse is gated on an acknowledgement of the previous read.
//
// It reproduces the record's table without any tuning: one slot dies at lead 1 (their "stopped
// after 22 tokens"), a signal space narrower than the data space dies at lead 1 as well (their
// "synchronization spin" with the partials double-buffered), two slots survive lead 1 and die at
// lead 2, and the acknowledged protocol survives any lead.
[[nodiscard]] inline SlotHazard scan_slot_hazard(const SlotPolicy& policy,
                                                 std::uint32_t calls,
                                                 std::uint32_t lead) {
    SlotHazard out;
    if (!policy.valid() || calls == 0) { return out; }
    if (policy.acknowledge_before_reuse) { return out; } // reuse is gated: no lead can be too big
    const auto fires = [&](std::uint32_t slots, std::uint32_t k) noexcept {
        return slots == 0 ? false : (lead >= slots && k >= slots);
    };
    for (std::uint32_t k = 0; k < calls; ++k) {
        if (fires(policy.data_slots, k)) {
            out.hazard     = true;
            out.call_index = k;
            out.slot       = k % policy.data_slots;
            out.what =
                "call " + std::to_string(k) + " republishes DATA slot " + std::to_string(out.slot) +
                " (last used by call " + std::to_string(k - policy.data_slots) +
                ") while the peer is reading call " + std::to_string(k >= lead ? k - lead : 0u) +
                " out of the same slot: " + std::to_string(policy.data_slots) +
                " data slot(s) absorb a lead of " +
                std::to_string(policy.data_slots - 1) + " call(s) and the faster rank is " +
                std::to_string(lead) + " ahead.";
            return out;
        }
        // The signal space is separate and can fail on its own -- the measured middle case.
        if (fires(policy.signal_slots, k) && !fires(policy.data_slots, k)) {
            out.hazard     = true;
            out.call_index = k;
            out.slot       = k % policy.signal_slots;
            out.what =
                "call " + std::to_string(k) + " republishes SIGNAL slot " +
                std::to_string(out.slot) + " at the same moment the peer is polling it: the signal "
                "space (" + std::to_string(policy.signal_slots) + " slot(s)) is narrower than the " +
                "data space (" + std::to_string(policy.data_slots) +
                " slot(s)), so the signal overtakes a read of the data it publishes. This is the "
                "configuration the record measured as a synchronization spin even though the "
                "partials were double-buffered, which is why the two spaces are separate fields "
                "here rather than one `slots`.";
            return out;
        }
    }
    return out;
}

// The protocol this file requires, as a value: BOTH spaces at two slots and an acknowledgement
// before reuse. That is 1Cat's accepted configuration, and the test asserts that the three other
// configurations above each produce a hazard while this one produces none at any lead.
[[nodiscard]] inline SlotPolicy required_slot_policy() noexcept {
    SlotPolicy policy;
    policy.data_slots                = 2u;
    policy.signal_slots              = 2u;
    policy.acknowledge_before_reuse  = true;
    return policy;
}

// The donor's four events, named, so that the two protocols can be compared as data rather than
// as prose. The second pair is the write-after-read barrier.
enum class TpCallStep : std::uint8_t {
    RecordInputsReady,  // inputs_ready[r]        <- this rank's inputs are complete
    WaitPeerInputsReady,// wait(inputs_ready[1-r])  the peer's source is complete
    PullPeer,           // pull_peer(peer -> mine)  on MY stream, MY storage
    RecordPullDone,     // pull_done[r]           <- my read of the peer's source is done
    WaitPeerPullDone,   // wait(pull_done[1-r])   the peer has finished reading MY source
    LocalCombine,       // in-place combine (allreduce_sum / the combine merge)
    PublishAck,         // the acknowledgement the SLOT policy needs, if it is used
};

[[nodiscard]] inline std::string_view tp_call_step_name(TpCallStep step) noexcept {
    switch (step) {
    case TpCallStep::RecordInputsReady: return "record(inputs_ready[r])";
    case TpCallStep::WaitPeerInputsReady: return "wait(inputs_ready[1-r])";
    case TpCallStep::PullPeer: return "pull_peer(peer->self)";
    case TpCallStep::RecordPullDone: return "record(pull_done[r])";
    case TpCallStep::WaitPeerPullDone: return "wait(pull_done[1-r])";
    case TpCallStep::LocalCombine: return "local-combine";
    case TpCallStep::PublishAck: return "publish-ack";
    }
    return "unknown-step";
}

// The step list for one call of a given op. `needs_local_combine` is derived from the op, not
// passed in, so a caller cannot forget it for allreduce_sum (which is in place) or add it for
// allgather_rows (which is a pure relocation and must not touch a byte).
[[nodiscard]] inline std::vector<TpCallStep> tp_call_plan(ReduceOp op,
                                                          const SlotPolicy& policy) {
    std::vector<TpCallStep> plan = {
        TpCallStep::RecordInputsReady,
        TpCallStep::WaitPeerInputsReady,
        TpCallStep::PullPeer,
        TpCallStep::RecordPullDone,
        TpCallStep::WaitPeerPullDone,
    };
    switch (op) {
    case ReduceOp::Sum:
    case ReduceOp::SumInPlace:
    case ReduceOp::AttentionCombine:
    case ReduceOp::AndBool:
        plan.push_back(TpCallStep::LocalCombine);
        break;
    case ReduceOp::AllGatherRows:
        // Exact, no arithmetic: verified by exact byte comparison, and a local combine here
        // would be a correctness bug rather than a redundancy.
        break;
    }
    if (policy.acknowledge_before_reuse) { plan.push_back(TpCallStep::PublishAck); }
    return plan;
}

// The count of event objects a call needs, which is the constraint a two-event version violates.
// The donor's record is explicit that the event count is a correctness constraint and not a
// tuning knob: a two-event version was rejected for a write-after-read hazard.
[[nodiscard]] inline std::uint32_t tp_event_objects_per_rank(ReduceOp op) noexcept {
    (void)op;
    return 2u; // inputs_ready[r] and pull_done[r]; the pair shares 4 objects in total
}

// The refusal a two-event version earns, with the reason spelled out. A caller that "optimises"
// the second pair away gets this string, not a silent hazard.
[[nodiscard]] inline std::string tp_event_count_refusal(std::uint32_t events) {
    if (events >= 2) { return {}; }
    return "a collective with " + std::to_string(events) +
           " event object(s) per rank cannot order the peer's READ of this rank's buffer: with "
           "one event the pair records readiness and waits for it, which says the peer's source "
           "is complete but says nothing about whether the peer has finished reading MY source. "
           "The second pair is a write-after-read barrier -- it is what makes an in-place combine "
           "safe and what makes back-to-back calls that reuse the same buffers safe. Two events "
           "is the minimum, four across the pair.";
}

// ---------------------------------------------------------------------------
// 3b. THE SLOTTED PROTOCOL AS A PLAN, AND THE BARRIER THAT DOES NOT DEPEND ON HOST ISSUE ORDER
// ---------------------------------------------------------------------------
//
// WHY THIS SECTION EXISTS. Section 3 scans which slot configurations survive a lead, and section 4
// issues a call. Between them is the part the measured arms did NOT have. The second event pair of a
// call is supposed to be the write-after-read barrier that makes an in-place combine safe, and it is
// NOT ONE whenever the peer's `cudaEventRecord` has not been issued yet at the moment this rank
// calls `cudaStreamWaitEvent`, because a wait SNAPSHOTS the event's state at the time of the call.
// The CUDA documentation says so; `tools/mg4/tp2_sum_stress.cu` section 0 MEASURES it: with 60 ms of
// work behind the record, a wait issued after the record costs 60.003 ms, a wait on an event with no
// pending record costs 0.001 ms, and the SAME record enqueued AFTER the wait costs 0.001 ms too.
//
// The consequence was measured, not argued. With both ranks driven by ONE host thread -- the serial
// emulation of tools/mgpu2/tp2_selfcheck.cu -- the in-place 2-rank sum returns `a + 2b` (this rank
// read its peer after the peer had already folded in its copy of b) in 1051 of 5000 iterations at
// natural timing and 200 of 200 once a deliberate lead is introduced, while the fork/join order is
// clean in 200 of 200 and the NON-in-place AND-reduce is clean on the same serial order. A flaky
// silent corruption is the worst failure mode a TP shard can have, because a short run misses it and
// the wrong number is then attributed to the model.
//
// THE REMEDY IS THE RECORD'S, AND IT IS THE ACKNOWLEDGEMENT. 1Cat's accepted configuration is
// partials double-buffered AND signals double-buffered AND an acknowledgement before reuse; its two
// rejected configurations are one slot ("stopped after 22 tokens") and partials-only ("maintained a
// synchronization spin"). Section 3's scan already encodes that. What an acknowledgement needs in
// order to BE an acknowledgement is a barrier whose effect does not depend on which rank the host
// issued first, and on this stack there are exactly two such primitives, both measured in the stress
// harness's section 4:
//
//   * a stream-ordered MEMORY operation -- `cuStreamWriteValue32` / `cuStreamWaitValue32` -- which
//     waits on a VALUE rather than on a snapshot. Measured here: ACCEPTED inside a capture on CUDA
//     13.3, and the eager pair works (write, wait on the same stream, the word reads back as 7).
//   * a DEVICE-side release/acquire pair inside a kernel -- `st.release.sys.global.u32` /
//     `ld.acquire.sys.global.u32` -- which is 1Cat's own mechanism
//     (`csrc/custom_all_reduce.cuh:440,451`) and the reason its choice looks like the answer.
//
// Both are device-observed: the wait completes when the WORD holds the value, whatever the host did
// first. That is the whole difference, and `scan_call_order()` below turns it into an exhaustive scan
// over the interleavings of two ranks' steps rather than a paragraph of prose.
//
// THE SHAPE IS THE RECORD'S IN BOTH CASES; THE MECHANISM IS THE CHOICE. The step plan is identical
// for both device-observed mechanisms, which is why `tp_slotted_call_plan()` does not take the
// mechanism as a shaping input.

// Which primitive implements a call's barriers. The two device-observed ones are named separately
// rather than collapsed into "the fix", because they differ in cost, in what they need from a
// captured graph, and in what a cross-DEVICE form would require of them.
enum class TpSignalMechanism : std::uint8_t {
    // The donor's four events. Kept named because it is what the pre-fix seam uses and because the
    // scan has to be able to say that it is the failing configuration.
    EventPair,
    // `cuStreamWriteValue32` / `cuStreamWaitValue32`: write and wait are stream-ordered memory
    // operations on a word, so the wait is satisfied by the WORD and not by a snapshot.
    StreamValue,
    // A kernel that publishes with `st.release.sys` and a kernel that polls with `ld.acquire.sys`,
    // i.e. 1Cat's mechanism. Costs a launch per barrier and needs no driver API at all.
    DeviceAcquire,
};

[[nodiscard]] inline std::string_view tp_signal_mechanism_name(TpSignalMechanism m) noexcept {
    switch (m) {
    case TpSignalMechanism::EventPair: return "event-pair (snapshot: host-issue-order dependent)";
    case TpSignalMechanism::StreamValue: return "stream-value (cuStreamWriteValue32/WaitValue32)";
    case TpSignalMechanism::DeviceAcquire: return "device-acquire (st.release.sys/ld.acquire.sys)";
    }
    return "unknown-mechanism";
}

// HOW MANY of a call's barriers are independent of host issue order. This is the number the whole
// section is about, and it is asserted rather than described:
//
//   0  the event pair. Both barriers (the peer's inputs-ready and the peer's read-done) are
//      snapshots, so the call is safe only when the host happens to have issued the peer's records
//      first -- true for a fork/join composition, false for two ranks driven by one thread in
//      sequence. The measured arms show exactly that split.
//   2  both device-observed mechanisms: the readiness barrier and the acknowledgement each hold
//      whatever the issue order was.
[[nodiscard]] inline std::uint32_t tp_issue_order_independent_barriers(TpSignalMechanism m) noexcept {
    return (m == TpSignalMechanism::EventPair) ? 0u : 2u;
}

// The slot and the epoch of call k, DERIVED rather than passed in. Two ranks must agree on both by
// construction; a caller that passes the slot is a caller that can pass two different ones, which is
// the failure mode section 3's scan is written in terms of.
//
// The epoch starts at 1, and that is load-bearing rather than cosmetic: the words are zeroed at
// creation, so an epoch of 0 would make a slot's very first wait satisfied before any rank had done
// anything. `tp_epoch_for_call(k) >= 1` for every k, which the host test asserts.
//
// WHY THE WAITS ARE `>=` AND NOT `==`. A slot's word is written exactly once per call that uses
// that slot, with that call's epoch, and the epochs on one slot are strictly increasing -- so
// `word >= e` is SOUND: it can only hold if epoch e was published, and publishing it means the
// peer finished the corresponding step. `word != e` would be equally sound for a laggard and would
// SPIN FOREVER the moment a rank got a whole slot-cycle ahead of its peer -- which a host that
// issues all of one rank's calls before the other's can do. `>=` removes that hang class without
// weakening the barrier, and it is also the documented default for `cuStreamWaitValue32`
// (`CU_STREAM_WAIT_VALUE_GEQ`), so the driver path spells it out rather than relying on 0.
[[nodiscard]] inline std::uint32_t tp_slot_for_call(std::uint64_t call,
                                                    std::uint32_t slots = 2u) noexcept {
    return (slots == 0u) ? 0u : static_cast<std::uint32_t>(call % slots);
}

[[nodiscard]] inline std::uint32_t tp_epoch_for_call(std::uint64_t call,
                                                     std::uint32_t slots = 2u) noexcept {
    if (slots == 0u) { return 1u; }
    return 1u + static_cast<std::uint32_t>(call / slots);
}

// The steps of ONE slotted call. Same flavour as `TpCallStep`, but a separate enum because the
// barriers are different KINDS of step: a caller must not be able to write a plan in which the
// acknowledgement is an event pair by accident.
enum class TpSlottedStep : std::uint8_t {
    PublishSlotReady,   // signal[slot] <- epoch     MUST be after this rank's input write
    WaitPeerSlotReady,  // wait until the peer's signal[slot] >= epoch (see the GEQ note)
    PullPeer,           // read the peer's slot, write into this rank's staging
    PublishSlotRead,    // the PEER's ack[slot] <- epoch: "I have finished reading your slot"
    WaitPeerReadMine,   // wait until this rank's ack[slot] >= epoch: the peer read MY slot
    LocalCombineInPlace,// the combine that REWRITES this rank's slot
};

[[nodiscard]] inline std::string_view tp_slotted_step_name(TpSlottedStep s) noexcept {
    switch (s) {
    case TpSlottedStep::PublishSlotReady: return "publish(signal[slot]=epoch)";
    case TpSlottedStep::WaitPeerSlotReady: return "wait(signal[peer_slot]>=epoch)";
    case TpSlottedStep::PullPeer: return "pull_peer(peer_slot -> staging)";
    case TpSlottedStep::PublishSlotRead: return "publish(peer_ack[slot]=epoch)";
    case TpSlottedStep::WaitPeerReadMine: return "wait(ack[slot]>=epoch)";
    case TpSlottedStep::LocalCombineInPlace: return "local-combine (IN PLACE)";
    }
    return "unknown-step";
}

// The plan. Identical for both device-observed mechanisms -- they implement the same steps with
// different primitives -- and it is the only place the order is written down.
[[nodiscard]] inline std::vector<TpSlottedStep> tp_slotted_call_plan(TpSignalMechanism /*mechanism*/) {
    return {
        TpSlottedStep::PublishSlotReady,
        TpSlottedStep::WaitPeerSlotReady,
        TpSlottedStep::PullPeer,
        TpSlottedStep::PublishSlotRead,
        TpSlottedStep::WaitPeerReadMine,
        TpSlottedStep::LocalCombineInPlace,
    };
}

// THE ONE INVARIANT THE FIX IS, as a function so that a future edit which reorders the plan fails a
// test instead of a run: the in-place step must come after the wait on the peer's acknowledgement of
// ITS read of this rank's slot, and the pull must come before that wait. `LocalCombineInPlace`
// writes the buffer the peer pulls from, so it is legal only once the peer has said it has read it --
// and "has said" must be a value the DEVICE observes, not a record the host happened to enqueue
// first.
[[nodiscard]] inline bool tp_slotted_plan_is_safe(const std::vector<TpSlottedStep>& plan) noexcept {
    std::size_t wait_at    = plan.size();
    std::size_t combine_at = plan.size();
    std::size_t pull_at    = plan.size();
    for (std::size_t i = 0; i < plan.size(); ++i) {
        if (plan[i] == TpSlottedStep::WaitPeerReadMine && wait_at == plan.size()) { wait_at = i; }
        if (plan[i] == TpSlottedStep::LocalCombineInPlace && combine_at == plan.size()) {
            combine_at = i;
        }
        if (plan[i] == TpSlottedStep::PullPeer && pull_at == plan.size()) { pull_at = i; }
    }
    return wait_at < combine_at && pull_at < wait_at;
}

// The refusal a caller gets for the configuration the record measured as the synchronization spin:
// the partials double-buffered and the SIGNAL space left at one slot. It is a separate field in
// `SlotPolicy` precisely so a caller can build it by accident and be told what it built.
[[nodiscard]] inline std::string tp_slotted_policy_refusal(const SlotPolicy& policy) {
    if (policy.data_slots < 2u) {
        return "data_slots " + std::to_string(policy.data_slots) +
               ": one partial slot cannot absorb a lead of one call, and the record's measured "
               "symptom of it is 'the full model stopped after 22 tokens under larger graph skew'. "
               "Two partial slots is the minimum this protocol admits.";
    }
    if (policy.signal_slots < 2u) {
        return "signal_slots " + std::to_string(policy.signal_slots) + " against data_slots " +
               std::to_string(policy.data_slots) +
               ": this is the record's MIDDLE configuration -- the partials double-buffered and the "
               "signal space not -- and it is the one that maintained a synchronization spin. The "
               "signal space must be at least as wide as the data space, because a signal that "
               "overtakes the read of the data it publishes is worse than no signal.";
    }
    if (!policy.acknowledge_before_reuse) {
        return "acknowledge_before_reuse is false: without the peer's acknowledgement of its read, "
               "the reuse of a slot rests on a barrier that exists only when the host happened to "
               "issue the peer's record first, which is the defect this section fixes.";
    }
    return {};
}

// ---------------------------------------------------------------------------
// 3c. THE SAME QUESTION, AS AN EXHAUSTIVE SCAN OVER INTERLEAVINGS
// ---------------------------------------------------------------------------
//
// The measurements above are on a device. This is the part that can be PROVED on a host, and it is
// what makes the remedy a correctness argument rather than an observation about one machine.
//
// Model one call as nine steps per rank, each touching ONE buffer:
//
//   0 WRITE  slot[r]      the caller's input for this call, on this rank's own stream
//   1 WRITE  signal[r]    publish readiness (the event record, or the value write)
//   2 WAIT   signal[p]    witness: the peer's step 1
//   3 READ   slot[p]      the pull's read of the peer's source
//   4 WRITE  stage[r]     the pull's write into this rank's own staging
//   5 WRITE  ack[p]       publish "I have read your slot" INTO THE PEER's word
//   6 WAIT   ack[r]       witness: the peer's step 5
//   7 READ   stage[r]     the combine's read
//   8 WRITE  slot[r]      the combine's write -- IN PLACE, over a buffer the peer read at step 3
//
// and enumerate EVERY interleaving of the two ranks' 18 steps that keeps each rank's own order. A
// WAIT adds an ordering edge from its witness only when the mechanism permits it:
//
//   * the event pair permits it ONLY IF the witness step precedes the wait IN THE INTERLEAVING --
//     which is exactly what "the record had already been issued when the wait was called" means;
//   * the two device-observed mechanisms permit it ALWAYS, because a wait on a value is satisfied by
//     the value and not by the order in which two host calls were made.
//
// That single difference is encoded once, in `issue_order_only`, and it is the entire content of the
// fix. Two critical pairs are then asked whether they are ORDERED (a path exists in the DAG):
//
//   P1  rank r reads the peer's slot (step 3)  vs  the peer writes it (step 0)  -- the readiness side
//   P2  rank r rewrites its own slot (step 8)  after  the peer reads it (step 3)  -- the combine side
//
// Each pair is asked in the direction that makes it a hazard: a WRITE-AFTER-READ is a hazard when
// the write is NOT ordered after the read, and a READ-AFTER-WRITE when the read is not ordered after
// the write. So P1 asks "is the read at 3 reachable FROM the write at 0?" and P2 asks "is the write
// at 8 reachable FROM the read at 3?" -- both are forward reachability in the DAG, and a pair with
// no path is a pair the ordering does not cover.
//
// The scan reports, for each, in how many interleavings the pair is left UNORDERED. The event pair
// has such interleavings -- the serial merge is one, and it is the one the measurements used, with
// P2 unordered because the combine at step 8 precedes the peer's read at step 3 -- and the two
// device-observed mechanisms have none, which is a statement about all 48620 interleavings rather
// than about the ones anybody happened to run.
//
// The counts, measured on this tree: the event pair leaves the readiness pair unordered in 28028 of
// 48620 merges and the combine pair in 30140, while both device-observed mechanisms leave zero of
// each. Those are deterministic pure functions of the model, so the host test PINS them, and it also
// asks about two merges BY NAME -- the fully serial one and the fork/join composition -- because the
// interesting claim is not the aggregate but the split the device arms measured: serial corrupt,
// fork/join clean.

struct TpOrderScan {
    std::uint64_t interleavings = 0;       // merges of the two ranks' nine steps
    std::uint64_t readiness_unordered = 0; // P1 unordered in this many of them
    std::uint64_t combine_unordered = 0;   // P2 unordered in this many of them
    std::string readiness_witness;         // the first merge that leaves P1 unordered, as rank digits
    std::string combine_witness;           // the first merge that leaves P2 unordered
};

// ONE named merge, so a test can ask about the fork/join composition BY NAME instead of inferring its
// outcome from an aggregate count. `rank_of_step[i]` is the rank of the i-th step in merge order,
// which is exactly the order `TpOrderScan`'s witnesses are printed in.
struct TpMergeVerdict {
    bool readiness_ordered = false; // P1: the peer's write of its slot precedes this rank's read of it
    bool combine_ordered = false;   // P2: the peer's read of this rank's slot precedes its rewrite
};

namespace detail {

// Nine steps per rank, so a merge is an 18-bit mask with exactly nine set bits: C(18,9) = 48620.
inline constexpr int kScanSteps = 9;
inline constexpr int kScanTotal = 2 * kScanSteps;

struct ScanStep {
    int touch = -1;        // buffer id, or -1 for a step that touches none
    bool write = false;
    int witness_rank = -1; // the rank whose step this step waits on, -1 if it waits on nothing
    int witness_step = -1;
};

[[nodiscard]] inline int scan_popcount(std::uint32_t v) noexcept {
    int n = 0;
    while (v != 0u) { v &= (v - 1u); ++n; }
    return n;
}

// The step list of one rank. The mechanism does NOT appear here: every mechanism implements this
// same order, and the only difference between them is the rule by which a wait's edge is admitted.
[[nodiscard]] inline ScanStep scan_step(int r, int i) noexcept {
    const int p = 1 - r;
    switch (i) {
    case 0: return ScanStep{r, true, -1, -1};         // the caller's input into my own slot
    case 1: return ScanStep{4 + r, true, -1, -1};     // publish readiness
    case 2: return ScanStep{-1, false, p, 1};         // wait the peer's readiness
    case 3: return ScanStep{p, false, -1, -1};        // read the peer's slot
    case 4: return ScanStep{2 + r, true, -1, -1};     // write my staging
    case 5: return ScanStep{6 + p, true, -1, -1};     // publish "I read your slot" into the peer's word
    case 6: return ScanStep{-1, false, p, 5};         // wait the peer's acknowledgement of MY slot
    case 7: return ScanStep{2 + r, false, -1, -1};    // the combine reads my staging
    default: return ScanStep{r, true, -1, -1};        // the combine REWRITES my own slot, in place
    }
}

// Is `to` reachable from `from`? A depth-first walk over an 18-node adjacency matrix, because only
// four pairs are ever asked and a full transitive closure would cost 18x more per interleaving.
[[nodiscard]] inline bool reaches(const bool adj[kScanTotal][kScanTotal], int from, int to,
                                  bool seen[kScanTotal]) noexcept {
    if (from == to) { return true; }
    if (seen[from]) { return false; }
    seen[from] = true;
    for (int b = 0; b < kScanTotal; ++b) {
        if (adj[from][b] && reaches(adj, b, to, seen)) { return true; }
    }
    return false;
}

} // namespace detail

// The witness is reported as the rank of each of the 18 steps in merge order, e.g. eighteen zeros
// followed by nine ones is the fully SERIAL merge -- rank 0's whole call, then rank 1's.
[[nodiscard]] inline std::string tp_scan_merge_witness(const int rank_of_step[detail::kScanTotal]) {
    std::string out;
    out.reserve(static_cast<std::size_t>(detail::kScanTotal));
    for (int i = 0; i < detail::kScanTotal; ++i) {
        out.push_back(static_cast<char>('0' + rank_of_step[i]));
    }
    return out;
}

// THE ONE IMPLEMENTATION, for one named merge. `scan_call_order()` below is a loop over this
// function rather than a second copy of it, because two implementations of one ordering rule is how
// they drift.
[[nodiscard]] inline TpMergeVerdict scan_one_merge(TpSignalMechanism mechanism,
                                                  const int rank_of_step[detail::kScanTotal]) {
    using detail::kScanSteps;
    using detail::kScanTotal;
    TpMergeVerdict out;
    int pos[2][kScanSteps];
    int next[2] = {0, 0};
    for (int i = 0; i < kScanTotal; ++i) {
        const int r = rank_of_step[i];
        if (r < 0 || r > 1 || next[r] >= kScanSteps) { return out; }
        pos[r][next[r]] = i;
        ++next[r];
    }
    if (next[0] != kScanSteps || next[1] != kScanSteps) { return out; }

    const bool issue_order_only = (mechanism == TpSignalMechanism::EventPair);
    bool adj[kScanTotal][kScanTotal];
    for (int a = 0; a < kScanTotal; ++a) {
        for (int b = 0; b < kScanTotal; ++b) { adj[a][b] = false; }
    }
    for (int r = 0; r < 2; ++r) {
        for (int a = 0; a + 1 < kScanSteps; ++a) { adj[pos[r][a]][pos[r][a + 1]] = true; }
        for (int s = 0; s < kScanSteps; ++s) {
            const detail::ScanStep st = detail::scan_step(r, s);
            if (st.witness_rank < 0) { continue; }
            const int w  = pos[st.witness_rank][st.witness_step];
            const int me = pos[r][s];
            const bool admitted = !issue_order_only || (w < me);
            if (admitted) { adj[w][me] = true; }
        }
    }

    out.readiness_ordered = true;
    out.combine_ordered   = true;
    for (int r = 0; r < 2; ++r) {
        const int p = 1 - r;
        bool s1[kScanTotal] = {};
        if (!detail::reaches(adj, pos[p][0], pos[r][3], s1)) { out.readiness_ordered = false; }
        bool s2[kScanTotal] = {};
        if (!detail::reaches(adj, pos[p][3], pos[r][8], s2)) { out.combine_ordered = false; }
    }
    return out;
}

[[nodiscard]] inline TpOrderScan scan_call_order(TpSignalMechanism mechanism) {
    using detail::kScanSteps;
    using detail::kScanTotal;
    TpOrderScan out;

    for (std::uint32_t mask = 0; mask < (1u << kScanTotal); ++mask) {
        if (detail::scan_popcount(mask) != kScanSteps) { continue; }
        ++out.interleavings;

        // The merge, with each rank's own order preserved by construction: the n-th set bit of the
        // mask is rank 0's n-th step.
        int rank_of_step[kScanTotal];
        for (int i = 0; i < kScanTotal; ++i) {
            rank_of_step[i] = ((mask >> i) & 1u) != 0u ? 0 : 1;
        }
        const TpMergeVerdict v = scan_one_merge(mechanism, rank_of_step);
        if (!v.readiness_ordered) {
            ++out.readiness_unordered;
            if (out.readiness_witness.empty()) {
                out.readiness_witness = tp_scan_merge_witness(rank_of_step);
            }
        }
        if (!v.combine_ordered) {
            ++out.combine_unordered;
            if (out.combine_witness.empty()) {
                out.combine_witness = tp_scan_merge_witness(rank_of_step);
            }
        }
    }
    return out;
}

// The two-slot claim, as a sequence rather than a diagram: a call never shares a slot with either of
// its neighbours, so the scan above -- which models ONE call's steps -- is the whole question for a
// SEQUENCE of calls and not only for the first one.
[[nodiscard]] inline bool tp_slots_alternate(std::uint64_t calls, std::uint32_t slots = 2u) noexcept {
    if (slots < 2u) { return false; }
    for (std::uint64_t k = 0; k + 1 < calls; ++k) {
        if (tp_slot_for_call(k, slots) == tp_slot_for_call(k + 1, slots)) { return false; }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 4. THE CUDA SEAM (only compiled where a CUDA compiler is present)
// ---------------------------------------------------------------------------
//
// WHAT IS HERE: the minimal working fallback the plan's §1.6 asks for, shaped by the two facts
// that decided the donor's design -- `cudaMemcpyPeerAsync` is rejected INSIDE a capture
// (`cudaErrorStreamCaptureUnsupported`, measured on CUDA 13.1/580.178.04), while
// `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, ...)` over UVA is not, so ONE code path covers
// both the direct and the staged transport.
//
// WHAT IS NOT HERE: the transport itself. No cost is computed, no bandwidth is assumed, and
// nothing here is allowed to report a throughput number. A collective that cannot say which
// transport it got is not reportable either, which is why `TopologyReport` is a return value.

#if defined(__CUDACC__)

// The four events of a 2-rank call, created ONCE per (rank, peer) pair because
// `cudaEventCreate` and `cudaMalloc` are not capturable: a per-call allocation anywhere on the
// collective path is a capture failure, not a performance problem.
struct PeerEvents {
    cudaEvent_t inputs_ready[2]{nullptr, nullptr};
    cudaEvent_t pull_done[2]{nullptr, nullptr};

    bool create() noexcept {
        for (int i = 0; i < 2; ++i) {
            if (cudaEventCreateWithFlags(&inputs_ready[i], cudaEventDisableTiming) != cudaSuccess) {
                return false;
            }
            if (cudaEventCreateWithFlags(&pull_done[i], cudaEventDisableTiming) != cudaSuccess) {
                return false;
            }
        }
        return true;
    }

    void destroy() noexcept {
        for (int i = 0; i < 2; ++i) {
            if (inputs_ready[i] != nullptr) { cudaEventDestroy(inputs_ready[i]); inputs_ready[i] = nullptr; }
            if (pull_done[i] != nullptr) { cudaEventDestroy(pull_done[i]); pull_done[i] = nullptr; }
        }
    }
};

// The probe. It asks BOTH directions and it clears the sticky `cudaErrorPeerAccessAlreadyEnabled`
// so that "the next cudaGetLastError() in an unrelated launcher does not observe it" -- a detail
// the donor records and that costs one line here.
inline PeerAccessMatrix probe_peer_access(std::uint32_t device_count, bool* completed) noexcept {
    PeerAccessMatrix matrix;
    matrix.device_count = device_count;
    matrix.can_access_peer.assign(static_cast<std::size_t>(device_count) * device_count, 0);
    bool ok = true;
    for (std::uint32_t a = 0; a < device_count && ok; ++a) {
        for (std::uint32_t b = 0; b < device_count && ok; ++b) {
            if (a == b) { continue; }
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, static_cast<int>(a), static_cast<int>(b)) != cudaSuccess) {
                ok = false;
                break;
            }
            matrix.can_access_peer[static_cast<std::size_t>(a) * device_count + b] =
                can != 0 ? std::uint8_t{1} : std::uint8_t{0};
        }
    }
    (void)cudaGetLastError(); // clear the sticky ALREADY_ENABLED, as the donor does
    if (completed != nullptr) { *completed = ok; }
    return matrix;
}

// ONE line, and it is the whole transport for the fallback: a device pointer already names its
// device, so this entry point expresses a cross-device transfer as well as a local one. The
// driver uses the direct path when it granted peer access and stages through host memory when it
// did not.
inline cudaError_t pull_peer(void* dst, const void* src, std::size_t bytes,
                             cudaStream_t stream) noexcept {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
}

// The 1Cat task-list shape, for the two reductions our plan names, plus the donor's two. Each
// takes the rank's own stream and the rank's own storage; no op writes a buffer it does not own.
inline cudaError_t allreduce_sum_2rank(void* self, const void* peer_storage, std::size_t bytes,
                                       int rank, cudaStream_t stream,
                                       const PeerEvents& events) noexcept {
    const int peer = 1 - rank;
    cudaError_t err = cudaEventRecord(events.inputs_ready[rank], stream);
    if (err != cudaSuccess) { return err; }
    err = cudaStreamWaitEvent(stream, events.inputs_ready[peer], 0);
    if (err != cudaSuccess) { return err; }
    err = pull_peer(self, peer_storage, bytes, stream);
    if (err != cudaSuccess) { return err; }
    err = cudaEventRecord(events.pull_done[rank], stream);
    if (err != cudaSuccess) { return err; }
    err = cudaStreamWaitEvent(stream, events.pull_done[peer], 0);
    if (err != cudaSuccess) { return err; }
    // The local combine is the caller's kernel; this seam does not assume an addend layout.
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// 4b. THE SLOTTED 2-RANK SUM -- THE FIX, with both device-observed mechanisms
// ---------------------------------------------------------------------------
//
// WHAT WAS WRONG. `allreduce_sum_2rank()` orders a call with the donor's four events, and its second
// pair is supposed to be the barrier that makes an in-place combine safe. It is not one: the caller's
// combine is enqueued after `wait(pull_done[peer])`, and that wait is a SNAPSHOT taken when the host
// called it. When one host thread issues rank 0's whole call and then rank 1's -- the serial
// emulation -- rank 0's wait for rank 1's `pull_done` is issued before rank 1's record exists, so it
// does not block, and rank 0's in-place write races rank 1's read of it. The measured signature is
// `a + 2b`: rank 1 read rank 0's source after rank 0 had already folded in its copy of b.
// Measured in `tools/mg4/tp2_sum_stress.cu`: 1051 of 5000 iterations at natural timing, 200 of 200
// with a deliberate lead, while the fork/join order is clean 200 of 200.
//
// WHAT REPLACES IT. The same order -- section 3b's plan -- with both barriers implemented as
// DEVICE-OBSERVED values instead of host-issued snapshots:
//
//   PublishSlotReady   signal[slot] <- epoch            (the event record, as a publish)
//   WaitPeerSlotReady  wait until the peer's signal[slot] == epoch
//   PullPeer           the pull, unchanged: a UVA cudaMemcpyAsync(D2D) on this rank's own stream
//   PublishSlotRead    the PEER's ack[slot] <- epoch    ("I have finished reading your slot")
//   WaitPeerReadMine   wait until this rank's ack[slot] == epoch
//   LocalCombineInPlace the combine, which rewrites the very buffer the peer read at PullPeer
//
// The acknowledgement is what makes the last step legal, and a value wait is what makes the
// acknowledgement an acknowledgement rather than a race: it is satisfied by the WORD, so it holds
// whether the host issued this rank's wait before or after the peer's publish. `scan_call_order()`
// in section 3c is the same statement exhaustively over all 48620 interleavings of the two ranks'
// nine steps, and it is a host test rather than a device measurement.
//
// BOTH MECHANISMS ARE IMPLEMENTED, for a measured reason rather than for choice: they differ in what
// they need from a captured graph and in what they cost per call, and both are measured in the
// stress harness's section 4 on this stack --
//
//   StreamValue    `cuStreamWriteValue32` / `cuStreamWaitValue32`. Four driver calls per call, no
//                  launch; MEASURED ACCEPTED inside a capture on CUDA 13.3.
//   DeviceAcquire  `st.release.sys` / `ld.acquire.sys` inside kernels, which is 1Cat's own mechanism
//                  (`csrc/custom_all_reduce.cuh:440,451`). Three extra launches per call; needs no
//                  driver API and no link against libcuda; carries the `sys` scope a cross-DEVICE
//                  form needs.
//
// WHY THE DRIVER API IS NAMED HERE AND NOWHERE ELSE: `cuStreamWriteValue32`/`cuStreamWaitValue32`
// are the only stream-ordered MEMORY operations on this stack -- the runtime API does not expose
// them (they are absent from cuda_runtime_api.h under CUDA 13.3) -- and they are the cheapest way to
// get a barrier whose effect does not depend on host issue order.

#include <cuda.h>

// The release/acquire carrier. `sys` scope rather than `gpu` scope because that is what a
// cross-DEVICE form needs and it is 1Cat's recorded choice; on one device it is strictly stronger
// than necessary, and the harness measures both mechanisms on the same one-device shape.
__device__ __forceinline__ void tp_release_u32(unsigned int* word, unsigned int value) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(word), "r"(value) : "memory");
}

__device__ __forceinline__ unsigned int tp_acquire_u32(const unsigned int* word) {
    unsigned int v = 0u;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(word) : "memory");
    return v;
}

__global__ void tp_publish_epoch_kernel(unsigned int* word, unsigned int epoch) {
    tp_release_u32(word, epoch);
}

// ONE thread polls, and no block may pass until it has: the kernel IS the barrier. Launched with one
// thread where the only thing that must be ordered is a following stream operation (the pull), and
// with one block per workgroup where the barrier gates work inside the kernel (the combine). Every
// block polls, so no block can pass early; `__syncthreads()` inside a block suffices because a
// block's threads all poll the same word.
//
// `<` rather than `!=`, and the reason is not style. A slot's word is written exactly once per call
// that uses that slot, with the epoch of that call, and `tp_epoch_for_call()` is strictly increasing
// per slot -- so `word >= e` is sound (it implies epoch e was published) while `word != e` would spin
// forever the moment the peer got a whole slot-cycle ahead of this rank, which a host that issues all
// of one rank's calls before the other's can do. `>=` removes that hang class without weakening the
// barrier, and the corresponding driver flag is CU_STREAM_WAIT_VALUE_GEQ.
__global__ void tp_wait_epoch_kernel(const unsigned int* word, unsigned int epoch) {
    if (threadIdx.x == 0) {
        while (tp_acquire_u32(word) < epoch) { __nanosleep(64); }
    }
    __syncthreads();
}

// The in-place combine, gated INTERNALLY on the peer's acknowledgement. This is the shape a captured
// graph needs: the wait and the write are one node, so no host round trip can be inserted between
// them, and the check cannot be skipped.
__global__ void tp_add_in_place_acquired_kernel(float* self, const float* staging, int n,
                                                const unsigned int* ack_word, unsigned int epoch) {
    if (threadIdx.x == 0) {
        while (tp_acquire_u32(ack_word) < epoch) { __nanosleep(64); }
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { self[i] += staging[i]; }
}

// `ReduceOp::SumInPlace`'s local step, plain: the StreamValue mechanism orders it with a value wait
// on the stream, so the kernel itself needs no barrier. Same arithmetic as the acquired form above,
// which is what makes the two mechanisms comparable rather than two different tests.
__global__ void tp_add_in_place_kernel(float* self, const float* staging, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { self[i] += staging[i]; }
}

// THE CHANNEL: two data slots, two signal slots, two acknowledgement slots, all device words.
//
// LIFETIME, and it is a correctness constraint rather than a convenience: ONE channel per (rank,
// peer) pair for the WHOLE sequence of calls, because the epoch counter must be monotonic. A channel
// recreated per call would restart the epochs, and a wait for epoch 1 would then be satisfied by the
// previous call's leftover 1 before the peer had published anything -- a silent wrong answer, which
// is the failure mode this whole file exists to remove. `create()` and `destroy()` are therefore
// once per sequence (and `cudaMalloc` is not capturable anyway).
struct TpSlotChannel {
    static constexpr std::uint32_t kSlots = 2u;
    int rank = 0;
    std::uint64_t call_index = 0;
    // 2 * kSlots words: [0, kSlots) the readiness signals, [kSlots, 2 * kSlots) the acknowledgements.
    std::uint32_t* words = nullptr;
    std::uint32_t* signal_peer[kSlots] = {nullptr, nullptr};
    std::uint32_t* ack_peer[kSlots]    = {nullptr, nullptr};

    [[nodiscard]] std::uint32_t* signal_local(std::uint32_t slot) const noexcept {
        return words + slot;
    }
    [[nodiscard]] std::uint32_t* ack_local(std::uint32_t slot) const noexcept {
        return words + kSlots + slot;
    }

    bool create() noexcept {
        if (cudaMalloc(reinterpret_cast<void**>(&words),
                       sizeof(std::uint32_t) * 2u * kSlots) != cudaSuccess) {
            return false;
        }
        // ZEROED, because the first epoch is 1 and a zero word must mean "no call has happened".
        return cudaMemset(words, 0, sizeof(std::uint32_t) * 2u * kSlots) == cudaSuccess;
    }

    void destroy() noexcept {
        if (words != nullptr) { cudaFree(words); words = nullptr; }
    }

    // The pointer exchange. On ONE device this is a copy of two addresses. ACROSS devices the same
    // two addresses have to arrive through a `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` exchange,
    // because a signal word the peer cannot read is not a signal -- which is exactly the property
    // `PeerEvents` lacks across processes. That exchange is PENDING HARDWARE on this box and is
    // named here rather than stubbed.
    void link_peer(const TpSlotChannel& peer) noexcept {
        for (std::uint32_t s = 0; s < kSlots; ++s) {
            signal_peer[s] = peer.words + s;
            ack_peer[s]    = peer.words + kSlots + s;
        }
    }

    [[nodiscard]] std::uint32_t slot_for_call() const noexcept {
        return tp_slot_for_call(call_index, kSlots);
    }
    [[nodiscard]] std::uint32_t epoch_for_call() const noexcept {
        return tp_epoch_for_call(call_index, kSlots);
    }
    void advance() noexcept { ++call_index; }
};

// The publish and the wait, one implementation each, so the two mechanisms cannot drift on the
// ORDER they implement -- only on the primitive.
inline cudaError_t tp_publish_epoch(std::uint32_t* word, std::uint32_t epoch, cudaStream_t stream,
                                    TpSignalMechanism mechanism, CUresult* cu_out) noexcept {
    if (mechanism == TpSignalMechanism::DeviceAcquire) {
        tp_publish_epoch_kernel<<<1, 1, 0, stream>>>(word, epoch);
        return cudaGetLastError();
    }
    const CUresult r = cuStreamWriteValue32(reinterpret_cast<CUstream>(stream),
                                            reinterpret_cast<CUdeviceptr>(word), epoch, 0u);
    if (cu_out != nullptr) { *cu_out = r; }
    return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

inline cudaError_t tp_wait_epoch_on_stream(std::uint32_t* word, std::uint32_t epoch,
                                           cudaStream_t stream, TpSignalMechanism mechanism,
                                           CUresult* cu_out) noexcept {
    if (mechanism == TpSignalMechanism::DeviceAcquire) {
        tp_wait_epoch_kernel<<<1, 1, 0, stream>>>(word, epoch);
        return cudaGetLastError();
    }
    const CUresult r = cuStreamWaitValue32(reinterpret_cast<CUstream>(stream),
                                           reinterpret_cast<CUdeviceptr>(word), epoch,
                                           CU_STREAM_WAIT_VALUE_GEQ);
    if (cu_out != nullptr) { *cu_out = r; }
    return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

// ONE rank's whole call of `ReduceOp::SumInPlace`, with section 3b's plan and the in-place combine
// INSIDE it -- because for an in-place local step the barrier's necessity IS the in-placeness, so a
// version that leaves the combine to the caller cannot express the fix.
//
// `self_slot` is this rank's source for this call's slot and it is BOTH the in-place addend and the
// buffer the peer pulls from; `peer_slot` is the peer's source for the same slot; `staging` is this
// rank's own scratch. Nothing here writes a buffer it does not own, and the only buffer written
// after a peer may have read it is `self_slot`, which is precisely what the acknowledgement gates.
//
// A caller with a different addend layout uses `tp_slotted_call_plan()` and its own kernel at the
// `LocalCombineInPlace` position: the barrier is the same, and only the arithmetic differs.
inline cudaError_t allreduce_sum_2rank_slotted(TpSlotChannel& channel, cudaStream_t stream,
                                               void* self_slot, const void* peer_slot,
                                               void* staging, int elements, std::size_t bytes,
                                               TpSignalMechanism mechanism,
                                               CUresult* cu_out = nullptr) noexcept {
    const std::uint32_t slot  = channel.slot_for_call();
    const std::uint32_t epoch = channel.epoch_for_call();
    const int blocks = (elements + 255) / 256;

    cudaError_t err = tp_publish_epoch(channel.signal_local(slot), epoch, stream, mechanism, cu_out);
    if (err != cudaSuccess) { return err; }
    err = tp_wait_epoch_on_stream(channel.signal_peer[slot], epoch, stream, mechanism, cu_out);
    if (err != cudaSuccess) { return err; }
    err = pull_peer(staging, peer_slot, bytes, stream);
    if (err != cudaSuccess) { return err; }
    err = tp_publish_epoch(channel.ack_peer[slot], epoch, stream, mechanism, cu_out);
    if (err != cudaSuccess) { return err; }

    if (mechanism == TpSignalMechanism::DeviceAcquire) {
        // The wait on the peer's acknowledgement and the in-place write are ONE node.
        tp_add_in_place_acquired_kernel<<<blocks, 256, 0, stream>>>(
            static_cast<float*>(self_slot), static_cast<const float*>(staging), elements,
            channel.ack_local(slot), epoch);
        err = cudaGetLastError();
    } else {
        err = tp_wait_epoch_on_stream(channel.ack_local(slot), epoch, stream, mechanism, cu_out);
        if (err != cudaSuccess) { return err; }
        tp_add_in_place_kernel<<<blocks, 256, 0, stream>>>(
            static_cast<float*>(self_slot), static_cast<const float*>(staging), elements);
        err = cudaGetLastError();
    }
    if (err != cudaSuccess) { return err; }
    channel.advance();
    return cudaSuccess;
}

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// 5. WHAT IS NOT HERE, NAMED
// ---------------------------------------------------------------------------
//
//   * THE TRANSPORT. Not a gap: not this component's concern. The seam above is the compatibility
//     surface, and the fallback exists because a transport may be absent (RCCL 2.30) -- not
//     because this engine wants to own one.
//   * THE COST OF ANY COLLECTIVE. Not measured, not estimated, not apologised for. Our plan's
//     §1.4 says the same thing: the rank axis must be right with `use_cuda_graph=false` first.
//   * AGGREGATE SCALING WITH RANK COUNT. One GPU on this box. `pending hardware`, stated once.
//   * THE KERNELS. `allreduce_sum_2rank` orders and moves; the add itself, the LSE merge and the
//     AND are kernels the caller supplies, because their layouts belong to the ops that own them.
//     What is NOT left to the caller is the ORDER, the event count and the slot policy.

} // namespace ninfer::multi
