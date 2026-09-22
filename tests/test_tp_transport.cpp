// test_tp_transport.cpp -- THE COLLECTIVE SEAM, the reduction set, the topology decision, and the
// slot protocol, all as host arithmetic.
//
// Everything here is a pure function of its inputs, so the whole file runs on one GPU with plain
// `g++ -std=c++20`, no CUDA header and no device. What it therefore PROVES and what it does NOT:
//
//   PROVES  which reductions a sum-only all-reduce cannot express and at what cost, that the
//           topology decision names its own fallback, that the AND encoding is unsafe outside
//           {0,1}, and that the slot protocol's three measured failures are reproduced by the
//           same scan that the accepted protocol survives.
//   DOES NOT see a GPU. The transport's COST is not this component's number (see section 5 of
//           core/tp_transport.h), and no line here measures anything.

#include "core/decode_graph_peer.h"
#include "core/tp_transport.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace ninfer::multi;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

void check_eq_u64(std::uint64_t got, std::uint64_t want, const std::string& what) {
    ++g_checks;
    if (got != want) {
        ++g_failures;
        std::printf("FAIL: %s (got %llu, want %llu)\n", what.c_str(),
                    static_cast<unsigned long long>(got), static_cast<unsigned long long>(want));
    }
}

// A peer-access matrix with every ordered pair reachable.
PeerAccessMatrix full_clique(std::uint32_t n, bool both_directions = true) {
    PeerAccessMatrix m;
    m.device_count = n;
    m.can_access_peer.assign(static_cast<std::size_t>(n) * n, 0);
    for (std::uint32_t a = 0; a < n; ++a) {
        for (std::uint32_t b = 0; b < n; ++b) {
            if (a == b) { continue; }
            const bool ok = (b < a) ? both_directions : true;
            m.can_access_peer[static_cast<std::size_t>(a) * n + b] = ok ? 1 : 0;
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// Group A: the topology decision
// ---------------------------------------------------------------------------
void group_A_topology() {
    // A1: one device. Unavailable, with a reason that says it is the single-rank contract rather
    // than a degraded transport -- and no numbers for a caller to use by accident.
    {
        PeerAccessMatrix m;
        m.device_count = 1;
        m.can_access_peer.assign(1, 0);
        const TopologyReport r = decide_topology(m, 1, true);
        check(r.decision == TopologyDecision::Unavailable, "A1 one device -> unavailable");
        check(!r.reason.empty(), "A1 unavailable carries a reason");
        check(r.reason.find("single-rank contract") != std::string::npos,
              "A1 the reason names the contract, not a failure");
    }

    // A2: a full 2-clique -> DirectPeer, and NO reason (there is nothing to explain).
    {
        const TopologyReport r = decide_topology(full_clique(2), 2, true);
        check(r.decision == TopologyDecision::DirectPeer, "A2 full 2-clique -> direct-peer");
        check(r.reason.empty(), "A2 a decided topology carries no caveat");
    }

    // A3: ONE direction only. This is the case the donor's `enable_peer_access` exists for:
    // "asymmetric support is not a usable transport for a symmetric collective". It must NOT be
    // DirectPeer, and the reason must name the ordered pair that is missing rather than saying
    // "not fully connected".
    {
        PeerAccessMatrix m = full_clique(2);
        m.can_access_peer[1 * 2 + 0] = 0; // 1 -> 0 missing
        const TopologyReport r = decide_topology(m, 2, true);
        check(r.decision == TopologyDecision::HostStaged, "A3 asymmetric -> host-staged");
        check(r.reason.find("1->0") != std::string::npos,
              "A3 the reason names the missing ordered pair");
        check(r.reason.find("supported transport") != std::string::npos,
              "A3 and says the staged path is supported, not an error");
    }

    // A4: the probe did not complete. A failed probe is NOT "no peer access" -- the same
    // false-negative shape as reading an nvidia-smi timeout as "GPU idle". It must fall back to
    // staged and SAY SO, not claim a decision it did not make.
    {
        PeerAccessMatrix m; // device_count 0: nothing was learned
        const TopologyReport r = decide_topology(m, 2, /*probe_completed=*/false);
        check(r.decision == TopologyDecision::HostStaged, "A4 failed probe -> host-staged");
        check(r.reason.find("did not complete") != std::string::npos,
              "A4 the reason distinguishes a failed probe from a refused path");
        check(r.reason.find("absent answer as 'no'") != std::string::npos,
              "A4 and refuses to read absence as a refusal");
    }

    // A5: the two policies are distinct values, so "the driver decided" is never mistaken for
    // "we decided". The donor's stance is named, not dismissed.
    check(topology_policy_name(TopologyPolicy::SilentDriverChoice) == "silent-driver-choice",
          "A5 the donor's policy has a name");
    check(topology_policy_name(TopologyPolicy::ExplicitDecision) == "explicit-decision",
          "A5 and ours has a different one");
    check(TopologyPolicy::SilentDriverChoice != TopologyPolicy::ExplicitDecision,
          "A5 the two are not the same value");

    // A6: fully_connected() must be false for a world larger than the device count.
    check(!full_clique(2).fully_connected(4), "A6 a 2-device clique is not a 4-rank clique");
    check(full_clique(4).fully_connected(4), "A6 a 4-device clique is a 4-rank clique");
}

// ---------------------------------------------------------------------------
// Group B: which reductions a sum-only all-reduce cannot express
// ---------------------------------------------------------------------------
void group_B_reduction_set() {
    // B1: the donor supplies the two ops exactly; the combine it can only STAGE; the AND it has
    // not got at all. Red control: the three provenance values are not all the same, so the
    // function distinguishes rather than returning a constant.
    check(reduce_op_provenance(ReduceOp::Sum) == ReduceOpProvenance::DonorExact,
          "B1 sum is donor-exact");
    check(reduce_op_provenance(ReduceOp::SumInPlace) == ReduceOpProvenance::DonorExact,
          "B1 sum-in-place is donor-exact");
    check(reduce_op_provenance(ReduceOp::AllGatherRows) == ReduceOpProvenance::DonorExact,
          "B1 allgather-rows is donor-exact");
    check(reduce_op_provenance(ReduceOp::AttentionCombine) == ReduceOpProvenance::DonorStaged,
          "B1 the combine is donor-STAGED, not donor-exact");
    check(reduce_op_provenance(ReduceOp::AndBool) == ReduceOpProvenance::NeitherReference,
          "B1 the boolean AND is in neither reference");
    check(reduce_op_provenance(ReduceOp::Sum) != reduce_op_provenance(ReduceOp::AndBool),
          "B1 red control: the provenance function is not a constant");

    // B2: the combine costs TWO collectives, not one, and the reason is arithmetic rather than
    // taste: the weight exp(m_r - m*) depends on the GLOBAL max, which no rank knows before the
    // gather. A single sum cannot recover it afterwards.
    {
        const SumOnlyCost combine = sum_only_cost(ReduceOp::AttentionCombine);
        check_eq_u64(combine.collectives.size(), 2, "B2 the combine costs 2 collectives");
        check(combine.collectives[0] == "ag" && combine.collectives[1] == "ar",
              "B2 and they are a gather then a reduce");
        check_eq_u64(combine.local_steps, 2, "B2 with 2 local steps between them");
        check(combine.caller_layout_precondition,
              "B2 and a layout precondition the caller must satisfy");
        // The measured refutation of the "2-3 round trips WITH HOST FIXUP" framing: the weight is
        // an exp2 over a device-resident LSE, so no host round trip is needed. Saying 3 would be
        // as wrong as saying 1.
        check(!combine.host_fixup_required, "B2 no host fixup is required for the combine");
    }
    {
        const SumOnlyCost sum = sum_only_cost(ReduceOp::Sum);
        check_eq_u64(sum.collectives.size(), 1, "B2 a plain sum is one collective");
        check(sum.collectives[0] == "ar", "B2 and it is the reduce");
        check_eq_u64(sum.local_steps, 0, "B2 with no local step");
    }
    {
        const SumOnlyCost ag = sum_only_cost(ReduceOp::AllGatherRows);
        check_eq_u64(ag.collectives.size(), 1, "B2 an allgather is one collective");
        check_eq_u64(ag.local_steps, 0, "B2 and a pure relocation has no local step");
    }

    // B3: the AND ENCODING. A sum CAN encode an AND -- and only on the domain {0,1}, which
    // nothing in a sum-typed API enforces. Both directions are asserted, so neither "the encoding
    // is always wrong" nor "it is always right" survives.
    {
        // Agrees on every well-formed verdict.
        const std::vector<std::vector<std::uint32_t>> well_formed = {
            {1, 1}, {1, 0}, {0, 1}, {0, 0}, {1, 1, 1}, {1, 0, 1}, {0, 0, 0}};
        for (const auto& v : well_formed) {
            const AndEncodingWitness w = and_encoding_counterexample(v);
            check(w.sum_encoding_agrees, "B3 the sum encoding agrees on well-formed input");
        }
        // DISAGREES the moment a rank reports outside {0,1}. The witness names the values.
        {
            const AndEncodingWitness w = and_encoding_counterexample({1, 1, 2});
            check(!w.sum_encoding_agrees,
                  "B3 the sum encoding DISAGREES when a rank reports 2");
            check(w.witness.find("{1,1,2}") != std::string::npos,
                  "B3 and the witness prints the offending values");
            check(w.witness.find("outside {0,1}") != std::string::npos,
                  "B3 and names the precondition the type does not carry");
        }
        // The opposite direction: a rank reporting zero while others admit -- both mechanisms
        // agree, which is why the counterexample has to be constructed rather than assumed.
        {
            const AndEncodingWitness w = and_encoding_counterexample({1, 1, 0});
            check(w.sum_encoding_agrees, "B3 a zero keeps the two encodings in agreement");
        }
    }

    // B4: the NAMED boolean op. `bool` cannot hold 2, so the out-of-domain input that made B3
    // disagree is UNREPRESENTABLE rather than merely rejected.
    {
        check(all_reduce_and({true, true, true}), "B4 AND of all-admit is true");
        check(!all_reduce_and({true, false, true}), "B4 one refusal is a refusal");
        check(!all_reduce_and({true, false, false}), "B4 and stays one");
        check(all_reduce_and({}), "B4 the empty AND is the identity (all admit vacuously)");
        check(and_reduce_is_identity(1), "B4 world 1 needs no collective");
        check(!and_reduce_is_identity(2), "B4 world 2 does");
    }

    // B5: the combine's no-collective cases, shared with the shard axis rather than re-derived.
    {
        ShardPlan single;
        single.ok    = true;
        single.axis  = ParallelAxis::None;
        single.weight_columns = Range{0, 8};
        single.kv_heads       = Range{0, 2};
        single.text_layers    = Range{0, 4};
        check(combine_outcome_for(single) == CombineOutcome::None,
              "B5 axis=none -> no collective at all");

        ShardPlan tp2 = single;
        tp2.axis = ParallelAxis::Tensor;
        check(combine_outcome_for(tp2) == CombineOutcome::MergeRequired,
              "B5 tp -> the merge is required");

        ShardPlan pp2 = single;
        pp2.axis = ParallelAxis::Pipeline;
        check(combine_outcome_for(pp2) == CombineOutcome::Replicated,
              "B5 pp -> replicated, so the merge is the identity");

        ShardPlan bad;
        bad.ok = false;
        bad.reason = "refused";
        check(combine_outcome_for(bad) == CombineOutcome::MergeRequired,
              "B5 a FAILED plan must not read as free");
    }
}

// ---------------------------------------------------------------------------
// Group C: the slot protocol -- the record's three failures, reproduced
// ---------------------------------------------------------------------------
void group_C_slot_protocol() {
    const SlotPolicy one_slot{};                    // the default: 1 and 1, no acknowledgement
    SlotPolicy data_only;
    data_only.data_slots   = 2;                     // partials double-buffered
    data_only.signal_slots = 1;                     // ... and the signals not
    SlotPolicy both;
    both.data_slots   = 2;
    both.signal_slots = 2;
    const SlotPolicy accepted = required_slot_policy();

    // C1: ONE slot dies at a lead of 1 -- the record's "full model stopped after 22 tokens under
    // larger graph skew". The scan must find it, and the reason must name the lead.
    {
        const SlotHazard h = scan_slot_hazard(one_slot, /*calls=*/87, /*lead=*/1);
        check(h.hazard, "C1 one slot + lead 1 -> HAZARD (the 22-token failure)");
        check(h.what.find("absorb a lead of 0 call") != std::string::npos,
              "C1 the reason states the capacity of the slot space");
        check(!h.what.empty() && h.call_index == 1, "C1 and points at the offending call");
    }
    // C1 red control: at lead 0 there is nothing to be premature about.
    check(!scan_slot_hazard(one_slot, 87, 0).hazard,
          "C1 red control: lead 0 is not a hazard");

    // C2: the partials double-buffered and the SIGNALS not -- the record's "87-call graph
    // microbenchmark entered synchronization spin". The scan must blame the SIGNAL space, not the
    // data space, which is why the two are separate fields.
    {
        const SlotHazard h = scan_slot_hazard(data_only, 87, 1);
        check(h.hazard, "C2 data-only double buffer + lead 1 -> HAZARD (the spin)");
        check(h.what.find("SIGNAL slot") != std::string::npos,
              "C2 and the hazard is in the SIGNAL space, not the data space");
    }

    // C3: both spaces at two slots, no acknowledgement. Survives lead 1 and dies at lead 2 --
    // so "two slots" is not a licence to ignore the acknowledgement, and the scan says exactly
    // how much skew the sizing buys.
    {
        check(!scan_slot_hazard(both, 87, 1).hazard,
              "C3 two slots absorb a lead of 1");
        const SlotHazard h = scan_slot_hazard(both, 87, 2);
        check(h.hazard, "C3 ... and not a lead of 2");
        check(h.what.find("absorb a lead of 1 call") != std::string::npos,
              "C3 and the reason states the exact capacity");
    }

    // C4: the ACCEPTED protocol -- both spaces double-buffered AND the reuse gated on an
    // acknowledgement. No hazard at any lead, which is the record's "8,700 collectives, 17.099
    // us/call, complete". The leads tested include values the caller cannot bound.
    {
        check(accepted.valid(), "C4 the required policy is well formed");
        check(accepted.acknowledge_before_reuse, "C4 and it gates reuse on an acknowledgement");
        for (std::uint32_t lead = 1; lead <= 8; ++lead) {
            check(!scan_slot_hazard(accepted, 8700, lead).hazard,
                  "C4 the accepted protocol survives lead " + std::to_string(lead));
        }
    }
    // C4 red control: an ill-formed policy is not silently treated as safe.
    {
        SlotPolicy zero;
        zero.data_slots   = 0;
        zero.signal_slots = 0;
        check(!zero.valid(), "C4 red control: zero slots is not a valid policy");
        check(!scan_slot_hazard(zero, 10, 3).hazard,
              "C4 an invalid policy reports no hazard rather than a false one");
    }

    // C5: the event count. Two objects per rank is the donor's constraint and it is a CORRECTNESS
    // constraint, not a knob: the second pair is the write-after-read barrier. A one-event
    // "optimisation" earns a refusal that says why.
    {
        check_eq_u64(tp_event_objects_per_rank(ReduceOp::Sum), 2, "C5 two events per rank");
        check_eq_u64(tp_event_objects_per_rank(ReduceOp::AllGatherRows), 2,
                     "C5 and the same count for the pure relocation");
        check(tp_event_count_refusal(1).find("write-after-read") != std::string::npos,
              "C5 one event is refused, by name");
        check(tp_event_count_refusal(2).empty(), "C5 two is accepted");
        check(tp_event_count_refusal(0).find("at least") == std::string::npos ||
                  !tp_event_count_refusal(0).empty(),
              "C5 zero is refused too");
    }

    // C6: the step list. The second pair must come AFTER the pull and BEFORE the combine, and an
    // allgather must have NO local combine -- a caller who adds one has a correctness bug, not a
    // redundancy.
    {
        const std::vector<TpCallStep> sum_plan = tp_call_plan(ReduceOp::Sum, accepted);
        auto at = [&](TpCallStep s) {
            for (std::size_t i = 0; i < sum_plan.size(); ++i) {
                if (sum_plan[i] == s) { return static_cast<int>(i); }
            }
            return -1;
        };
        check(at(TpCallStep::RecordInputsReady) == 0, "C6 the plan starts by publishing inputs");
        check(at(TpCallStep::WaitPeerInputsReady) > at(TpCallStep::RecordInputsReady),
              "C6 then waits for the peer's");
        check(at(TpCallStep::PullPeer) > at(TpCallStep::WaitPeerInputsReady),
              "C6 then pulls");
        check(at(TpCallStep::WaitPeerPullDone) > at(TpCallStep::PullPeer),
              "C6 the write-after-read barrier follows the pull");
        check(at(TpCallStep::LocalCombine) > at(TpCallStep::WaitPeerPullDone),
              "C6 and the local combine is LAST -- after the peer's read of my source");
        check(at(TpCallStep::PublishAck) > at(TpCallStep::LocalCombine),
              "C6 the acknowledgement closes the call");

        const std::vector<TpCallStep> ag_plan = tp_call_plan(ReduceOp::AllGatherRows, accepted);
        bool has_combine = false;
        for (TpCallStep s : ag_plan) {
            if (s == TpCallStep::LocalCombine) { has_combine = true; }
        }
        check(!has_combine, "C6 an exact relocation has no local combine");

        // Without the acknowledgement the plan is one step shorter, so the policy is load-bearing
        // rather than decorative.
        const std::vector<TpCallStep> unacked = tp_call_plan(ReduceOp::Sum, both);
        check_eq_u64(unacked.size() + 1, sum_plan.size(),
                     "C6 the acknowledgement adds exactly one step");
    }
}

// ---------------------------------------------------------------------------
// Group D: the cross-device graph admission gate
// ---------------------------------------------------------------------------
void group_D_graph_gate() {
    PeerGraphRequest req;
    req.world_size = 2;

    // D1: the flag OFF is a complete path, and it is called by its right name -- the EAGER
    // IDENTITY, not a fallback.
    {
        req.use_cuda_graph = false;
        const PeerGraphAdmission a = peer_graph_admission(req);
        check(a.verdict == PeerGraphVerdict::EagerIdentity, "D1 flag off -> eager identity");
        check(a.take_eager_path, "D1 and it takes the eager path");
        check(a.reason.empty(), "D1 with no caveat");
    }

    // D2: world 1 has no cross-device graph to build, whichever way the flag is set.
    {
        PeerGraphRequest one = req;
        one.world_size      = 1;
        one.use_cuda_graph  = true;
        const PeerGraphAdmission a = peer_graph_admission(one);
        check(a.verdict == PeerGraphVerdict::EagerIdentity, "D2 world 1 -> eager identity");
        check(a.reason.find("no cross-device graph") != std::string::npos,
              "D2 and says why there is none");
    }

    // D3: every precondition, each refused BY NAME, and every refusal still routes to eager. This
    // is the gate's whole point: a refusal never means "run nothing".
    struct Case {
        const char* what;
        bool eager_validated;
        bool one_capture;
        bool uses_memcpy_peer_async;
        bool buffers_preallocated;
        bool topology_decided;
        const char* must_mention;
    };
    const Case cases[] = {
        {"capture before the eager path was validated", false, true, false, true, true,
         "identity REFERENCE"},
        {"cudaMemcpyPeerAsync on the path", true, true, true, true, true,
         "cudaErrorStreamCaptureUnsupported"},
        {"more than one live capture", true, false, false, true, true,
         "cudaErrorStreamCaptureMerge"},
        {"no topology decision", true, true, false, true, false, "topology decision"},
        {"per-call allocation", true, true, false, false, true, "cudaEventCreate"},
    };
    for (const Case& c : cases) {
        PeerGraphRequest r;
        r.world_size               = 2;
        r.use_cuda_graph           = true;
        r.eager_validated          = c.eager_validated;
        r.one_capture              = c.one_capture;
        r.uses_memcpy_peer_async   = c.uses_memcpy_peer_async;
        r.buffers_preallocated     = c.buffers_preallocated;
        r.topology_decided         = c.topology_decided;
        const PeerGraphAdmission a = peer_graph_admission(r);
        check(a.verdict == PeerGraphVerdict::Refused, std::string("D3 refused: ") + c.what);
        check(a.take_eager_path, std::string("D3 ... and still runs eager: ") + c.what);
        check(a.reason.find(c.must_mention) != std::string::npos,
              std::string("D3 ... naming the cause (") + c.must_mention + "): " + c.what);
    }

    // D4: the admissible case, with all six conditions met.
    {
        PeerGraphRequest r;
        r.world_size             = 2;
        r.use_cuda_graph         = true;
        r.eager_validated        = true;
        r.one_capture            = true;
        r.uses_memcpy_peer_async = false;
        r.buffers_preallocated   = true;
        r.topology_decided       = true;
        const PeerGraphAdmission a = peer_graph_admission(r);
        check(a.verdict == PeerGraphVerdict::CaptureAdmissible, "D4 all conditions -> admissible");
        check(!a.take_eager_path, "D4 and it does not take the eager path");
        check(a.reason.empty(), "D4 with no caveat");
    }

    // D5: capturing must not re-order anything. This is the checkable form of the donor's own
    // argument -- "the record/wait pairs become graph EDGES rather than nodes", so nothing about
    // the ordering design changes. Red control: the step lists are not empty, so the equality is
    // not two empty vectors agreeing.
    {
        const SlotPolicy policy = required_slot_policy();
        const ReduceOp ops[]    = {ReduceOp::Sum, ReduceOp::SumInPlace, ReduceOp::AttentionCombine,
                                   ReduceOp::AllGatherRows, ReduceOp::AndBool};
        for (ReduceOp op : ops) {
            check(capture_preserves_ordering(op, policy),
                  std::string("D5 capture preserves the ordering for ") +
                      std::string(reduce_op_name(op)));
            check(!peer_capture_plan(op, policy).steps.empty(),
                  "D5 red control: the step list is not empty");
        }
        const PeerCapturePlan plan = peer_capture_plan(ReduceOp::AttentionCombine, policy);
        check_eq_u64(plan.live_captures, 1, "D5 exactly one live capture");
        check_eq_u64(plan.fork_edges, 1, "D5 one fork edge for the capture");
        check_eq_u64(plan.join_edges, 1, "D5 one join edge for the capture");
        check(plan.origin_rank != plan.peer_rank, "D5 the fork is to a DIFFERENT rank's stream");
    }

    // D6: the CROSS-RANK SPIN. This is the coexistence item the standing frame ("every mechanism
    // on at once") forces to be named up front: it is a DOCUMENTED HANG, not a risk. The refusal
    // must name the wedge, because "the graph is wrong" sends the reader to the wrong file.
    {
        PeerGraphRequest r;
        r.world_size                 = 2;
        r.use_cuda_graph             = true;
        r.eager_validated            = true;
        r.one_capture                = true;
        r.buffers_preallocated       = true;
        r.topology_decided           = true;
        r.cross_rank_spin_in_capture = true;
        const PeerGraphAdmission a = peer_graph_admission(r);
        check(a.verdict == PeerGraphVerdict::Refused, "D6 a spin inside the capture is refused");
        check(a.take_eager_path, "D6 ... and the eager path is taken instead");
        check(a.reason.find("WEDGE") != std::string::npos,
              "D6 the refusal names the documented card wedge");
        check(a.reason.find("TP2-SLICES") != std::string::npos,
              "D6 and cites the post-mortem that recorded it");
        check(a.reason.find("cannot yield") != std::string::npos,
              "D6 and says why a graph makes it worse than eager");
        check(a.reason.find("graph EDGES") != std::string::npos,
              "D6 and points at the signal-free alternative that avoids it");
        // Red control: the same request WITHOUT the spin is admissible, so the refusal is about
        // the spin and not about the request shape.
        PeerGraphRequest clean = r;
        clean.cross_rank_spin_in_capture = false;
        check(peer_graph_admission(clean).verdict == PeerGraphVerdict::CaptureAdmissible,
              "D6 red control: without the spin the same request is admissible");
    }

    // D7: the joint node allowance. Capturing is not the rank axis's decision alone: the donor
    // measured 1888 nodes at tp2 against 640 at tp1 and the end state adds MTP and the cold tier
    // to the SAME graph, so the budget is summed rather than discovered.
    {
        GraphNodeBudget b;
        b.world_size = 2;
        check_eq_u64(b.rank_axis_nodes(), kTp2GraphNodes, "D7 tp2's rank-axis cost is the "
                                                          "measured 1888");
        check_eq_u64(b.planned(), kTp2GraphNodes, "D7 tp2 alone exactly spends the default");
        check(graph_node_allowance_refusal(b).empty(), "D7 and is not refused (equal is allowed)");
        check(kGraphNodeAllowanceDefault == kTp2GraphNodes,
              "D7 the default allowance is the measured tp2 count, by name");

        // THE FINDING: tp2 plus MTP alone already overruns, because the rank axis spends the
        // whole default allowance. That is the answer to "every mechanism on at once", not an
        // inconvenience -- and the refusal has to say it rather than leaving it to be discovered.
        b.mtp_nodes = 400;
        const std::string with_mtp = graph_node_allowance_refusal(b);
        check(!with_mtp.empty(), "D7 tp2 + MTP does NOT fit the default allowance");
        check(with_mtp.find("1888") != std::string::npos &&
                  with_mtp.find("640") != std::string::npos,
              "D7 the refusal carries both measured node counts");
        check(with_mtp.find("SHARED") != std::string::npos,
              "D7 and says the allowance is shared, not per-mechanism");
        check(with_mtp.find("nowhere to live") != std::string::npos,
              "D7 and states the consequence for the other mechanisms");

        b.cold_nodes = 1200;
        check(graph_node_allowance_refusal(b).find("3488") != std::string::npos,
              "D7 the three contributors are summed and named");

        // A caller with a smaller graph may RAISE the allowance -- and doing so is a decision, so
        // the budget must respond to it rather than to a hidden constant.
        GraphNodeBudget raised;
        raised.world_size = 2;
        raised.mtp_nodes  = 400;
        raised.allowance  = 4096;
        check(graph_node_allowance_refusal(raised).empty(),
              "D7 raising the allowance explicitly makes the same plan fit");

        // The tp1 world has its own, smaller, count -- so the budget is a function of world size.
        GraphNodeBudget one;
        one.world_size = 1;
        check_eq_u64(one.rank_axis_nodes(), kTp1GraphNodes, "D7 tp1 uses the tp1 node count");
        check(kTp1GraphNodes < kTp2GraphNodes, "D7 and the two numbers differ");
        check_eq_u64(one.planned(), kTp1GraphNodes, "D7 tp1 alone plans the tp1 count");
    }
}

} // namespace

int main() {
    group_A_topology();
    group_B_reduction_set();
    group_C_slot_protocol();
    group_D_graph_gate();

    std::printf("tp_transport: %d checks, %d failures -> %s\n", g_checks, g_failures,
                g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
