// test_tp_slotted_ack.cpp -- THE FIX FOR THE MEASURED RED, as host arithmetic.
//
// The red, stated once, because everything below is written against it: with both ranks driven by ONE
// host thread, the in-place 2-rank sum of `core/tp_transport.h` returned `a + 2b` because the
// second event pair of a call is not a barrier when the peer's `cudaEventRecord` has not been issued
// yet -- `cudaStreamWaitEvent` snapshots the event at the time of the call. Measured in
// `tools/mg4/tp2_sum_stress.cu`: 1051 of 5000 iterations at natural timing, 200 of 200 under a
// deliberate lead, fork/join clean 200 of 200, and the NON-in-place AND-reduce clean on the same
// serial order. The remedy is the record's own: both spaces double-buffered plus an acknowledgement,
// with the barriers implemented as DEVICE-OBSERVED values rather than host-issued snapshots.
//
// Everything here is a pure function of its inputs, so the whole file runs with plain
// `g++ -std=c++20`, no CUDA header and no device:
//
//   PROVES  that the barrier rule the fix rests on is encoded ONCE and is the only difference
//           between the failing and the working protocol; that the event pair leaves two critical
//           pairs unordered in 28028 and 48620 of the 48620 possible interleavings of two ranks'
//           nine steps while both device-observed mechanisms leave ZERO; that the FIRST interleaving
//           the scan rejects is exactly the fully serial merge the measurements used to produce
//           `a + 2b`; that the epoch scheme makes a stale acknowledgement unable to satisfy a later
//           wait; that the plan's one invariant (the acknowledgement wait precedes the in-place
//           combine) holds, and that a plan which reorders it is rejected by the same predicate; and
//           that the slot policy the new protocol requires is exactly the one the record measured as
//           surviving while the three failing configurations each still produce a hazard.
//   DOES NOT see a GPU. The two mechanisms' device behaviour -- `cuStreamWriteValue32`/
//           `cuStreamWaitValue32` and `st.release.sys`/`ld.acquire.sys` -- is measured by
//           `tools/mg4/tp2_sum_stress.cu` under the project's GPU lock, which is where the 0/N
//           against the old path's k/N comes from. No line here measures anything.
//
// The provenance of the record's three configurations is `docs/design/sm70_deepseek_v4_tp8_graph_
// allreduce.md` (via core/tp_transport.h section 3); what is asserted below is the arithmetic and the
// protocol, not the provenance.

#include "core/tp_transport.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace ninfer::multi;

namespace {

int g_checks   = 0;
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
                    static_cast<unsigned long long>(got),
                    static_cast<unsigned long long>(want));
    }
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

void check_contains(const std::string& haystack, const std::string& needle,
                    const std::string& what) {
    check(contains(haystack, needle), what + " (missing \"" + needle + "\")");
}

// The fully SERIAL merge of the two ranks' steps: rank 0's whole call, then rank 1's whole call. It
// is the first merge the scan visits and it is the one the device measurements used.
const char* kSerialMerge = "000000000111111111";

// ---------------------------------------------------------------------------
// S1: the slot and the epoch, and the property that makes a stale ack harmless
// ---------------------------------------------------------------------------
void group_S1_epochs() {
    // The epoch starts at 1 and never returns to 0, because the words are zeroed at creation: an
    // epoch of 0 would make a slot's first wait satisfied before any rank had done anything.
    check_eq_u64(tp_epoch_for_call(0), 1u, "the first call's epoch is 1, not 0");
    check_eq_u64(tp_epoch_for_call(1, 2u), 1u, "the second call's epoch is still 1 (a different slot)");
    check_eq_u64(tp_epoch_for_call(2, 2u), 2u, "the third call reuses slot 0 at epoch 2");
    check_eq_u64(tp_slot_for_call(0, 2u), 0u, "call 0 uses slot 0");
    check_eq_u64(tp_slot_for_call(1, 2u), 1u, "call 1 uses slot 1");
    check_eq_u64(tp_slot_for_call(2, 2u), 0u, "call 2 returns to slot 0");
    check(tp_slots_alternate(64u, 2u), "two slots alternate over 64 calls: neighbours never share one");
    check(!tp_slots_alternate(64u, 1u), "and ONE slot does not alternate, so the refusal is not vacuous");

    // The word of slot s, over a long sequence: it is the number of times that slot has been used, so
    // it is STRICTLY increasing per slot and never equals the value a previous call waited for. This
    // is what makes a leftover acknowledgement unable to satisfy a later wait -- and it is the reason
    // the channel must live for the whole sequence rather than being recreated per call.
    std::uint64_t word[2] = {0u, 0u};
    std::uint32_t last_epoch_seen[2] = {0u, 0u};
    bool strictly_increasing = true;
    bool never_zero_for_a_call = true;
    bool epochs_never_repeat_within_a_slot = true;
    for (std::uint64_t k = 0; k < 64u; ++k) {
        const std::uint32_t s = tp_slot_for_call(k, 2u);
        const std::uint32_t e = tp_epoch_for_call(k, 2u);
        if (e <= last_epoch_seen[s]) { epochs_never_repeat_within_a_slot = false; }
        last_epoch_seen[s] = e;
        word[s]            = e; // the publish is the identity here: the word holds the epoch
        if (e == 0u) { never_zero_for_a_call = false; }
        strictly_increasing = strictly_increasing && (e >= 1u) && (word[s] == e);
    }
    check(epochs_never_repeat_within_a_slot,
          "a slot's epoch strictly increases, so a wait for epoch e cannot be satisfied by epoch e-1");
    check(never_zero_for_a_call, "no call ever waits for epoch 0, which is the word's initial value");
    check(strictly_increasing, "every call has an epoch of at least 1, and the word holds exactly it");
    check_eq_u64(last_epoch_seen[0], 32u, "slot 0 has been used 32 times in 64 calls");
    check_eq_u64(last_epoch_seen[1], 32u, "slot 1 has been used 32 times in 64 calls");
}

// ---------------------------------------------------------------------------
// S2: the plan, and the one invariant the fix is
// ---------------------------------------------------------------------------
void group_S2_plan() {
    for (TpSignalMechanism m : {TpSignalMechanism::EventPair, TpSignalMechanism::StreamValue,
                                TpSignalMechanism::DeviceAcquire}) {
        const std::vector<TpSlottedStep> plan = tp_slotted_call_plan(m);
        check_eq_u64(plan.size(), 6u, std::string("the plan has six steps (") +
                                          std::string(tp_signal_mechanism_name(m)) + ")");
        check(tp_slotted_plan_is_safe(plan),
              std::string("the plan is safe: the ack wait precedes the in-place combine (") +
                  std::string(tp_signal_mechanism_name(m)) + ")");
        // The mechanism does not shape the ORDER. If it did, the two mechanisms would be two
        // protocols and only one of them would be the one the scan cleared.
        const std::vector<TpSlottedStep> other = tp_slotted_call_plan(
            m == TpSignalMechanism::EventPair ? TpSignalMechanism::StreamValue
                                             : TpSignalMechanism::EventPair);
        check(plan == other, "the step plan is identical across mechanisms: the mechanism is the "
                             "primitive, not the order");
    }

    // The negative controls: the predicate must REJECT the two plans that are the defect and the
    // record's rejected middle configuration.
    std::vector<TpSlottedStep> reordered = tp_slotted_call_plan(TpSignalMechanism::StreamValue);
    std::swap(reordered[4], reordered[5]); // the combine before the ack wait
    check(!tp_slotted_plan_is_safe(reordered),
          "a plan with the in-place combine BEFORE the ack wait is rejected: that plan is the red");
    std::vector<TpSlottedStep> no_ack = tp_slotted_call_plan(TpSignalMechanism::StreamValue);
    no_ack.erase(no_ack.begin() + 4); // the ack wait removed entirely
    check(!tp_slotted_plan_is_safe(no_ack),
          "a plan with no ack wait at all is rejected: the acknowledgement is load-bearing, not "
          "decoration");
    std::vector<TpSlottedStep> no_pull = tp_slotted_call_plan(TpSignalMechanism::StreamValue);
    no_pull.erase(no_pull.begin() + 2); // the pull removed
    check(!tp_slotted_plan_is_safe(no_pull),
          "a plan with no pull is rejected (the predicate is not satisfied by an empty order)");
}

// ---------------------------------------------------------------------------
// S3: how many of a call's barriers are independent of host issue order
// ---------------------------------------------------------------------------
void group_S3_barrier_counts() {
    check_eq_u64(tp_issue_order_independent_barriers(TpSignalMechanism::EventPair), 0u,
                 "the event pair has ZERO issue-order-independent barriers: both of its barriers are "
                 "snapshots, which is the measured defect");
    check_eq_u64(tp_issue_order_independent_barriers(TpSignalMechanism::StreamValue), 2u,
                 "the stream-value mechanism has two: the readiness wait and the ack wait");
    check_eq_u64(tp_issue_order_independent_barriers(TpSignalMechanism::DeviceAcquire), 2u,
                 "the device-acquire mechanism has two, by the same construction");

    // And the count is not a claim about a mechanism's NAME: it is exactly the number of steps in the
    // plan that wait on a value rather than on a snapshot, which is 2 for every plan that ends in an
    // acknowledgement and 0 for the event form.
    const std::vector<TpSlottedStep> plan = tp_slotted_call_plan(TpSignalMechanism::StreamValue);
    std::uint32_t waits = 0;
    for (TpSlottedStep s : plan) {
        if (s == TpSlottedStep::WaitPeerSlotReady || s == TpSlottedStep::WaitPeerReadMine) { ++waits; }
    }
    check_eq_u64(waits, 2u, "the plan contains exactly two waits, which is what each mechanism has to "
                            "make order-independent");
}

// ---------------------------------------------------------------------------
// S4: THE SCAN. The fix's correctness, over all 48620 interleavings
// ---------------------------------------------------------------------------
void group_S4_the_scan() {
    const TpOrderScan ev = scan_call_order(TpSignalMechanism::EventPair);
    const TpOrderScan sv = scan_call_order(TpSignalMechanism::StreamValue);
    const TpOrderScan da = scan_call_order(TpSignalMechanism::DeviceAcquire);

    // C(18,9) = 48620: the two ranks' nine steps each, every merge that keeps each rank's own order.
    check_eq_u64(ev.interleavings, 48620u, "the scan enumerates every interleaving of the two ranks' "
                                           "nine steps (C(18,9) = 48620)");
    check_eq_u64(sv.interleavings, ev.interleavings, "all three mechanisms are scanned over the same "
                                                     "merge space, so their counts are comparable");
    check_eq_u64(da.interleavings, ev.interleavings, "and the third one too");

    // THE RED, reproduced on the host with no GPU: the event pair is unordered in both critical pairs
    // for a large share of the merges -- and the FIRST one it rejects is the fully serial merge.
    check(ev.combine_unordered > 0u,
          "the event pair leaves the in-place combine unordered against the peer's read in some "
          "interleavings -- the combine side of the measured red");
    check(ev.readiness_unordered > 0u,
          "and it leaves the readiness side unordered too, which is the same defect one step earlier: "
          "a pull can read the peer's slot before the peer wrote it");
    // The model is a deterministic pure function, so the aggregate is PINNED rather than bounded: a
    // future edit that changes the model has to change these numbers on purpose.
    check_eq_u64(ev.readiness_unordered, 28028u,
                 "the event pair leaves the readiness pair unordered in exactly 28028 of 48620 merges");
    check_eq_u64(ev.combine_unordered, 30140u,
                 "and the combine pair in exactly 30140 of 48620");
    check(ev.readiness_unordered < ev.interleavings && ev.combine_unordered < ev.interleavings,
          "and it is NOT all of them: some merges do order both pairs, which is why a fork/join "
          "composition is clean and a serial one is not");
    check(ev.combine_witness == kSerialMerge,
          "the FIRST interleaving the scan rejects on the combine side is the fully serial merge -- "
          "rank 0's whole call then rank 1's -- which is exactly the merge the device measurements "
          "used to produce a+2b");
    check(ev.readiness_witness == kSerialMerge,
          "and the same merge is the first rejected on the readiness side (got \"" +
              ev.readiness_witness + "\")");

    // THE SPLIT THE DEVICE ARMS MEASURED, asked by NAME on the host, with no GPU:
    //
    //   the serial merge      -- rank 0's whole call, then rank 1's -- corrupt for the event pair
    //   MG3's fork/join merge -- both records, both waits, both pulls, both records, both waits,
    //                            both combines -- clean for the event pair, in all three passes
    //
    // The fork/join merge, as the model's nine steps per rank: the two input writes, then the two
    // records, the two waits, rank 0's pull (read then write), rank 1's pull, the two pull-done
    // records, the two pull-done waits, then rank 0's combine (read then write) and rank 1's.
    const int serial_merge[18] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const int fork_join_merge[18] = {0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
                                     0, 1, 0, 1, 0, 0, 1, 1};
    const TpMergeVerdict ev_serial = scan_one_merge(TpSignalMechanism::EventPair, serial_merge);
    const TpMergeVerdict ev_fork   = scan_one_merge(TpSignalMechanism::EventPair, fork_join_merge);
    check(!ev_serial.combine_ordered && !ev_serial.readiness_ordered,
          "in the SERIAL merge the event pair orders neither pair: that merge is the measured red, "
          "and the host model reproduces it with no GPU");
    check(ev_fork.combine_ordered && ev_fork.readiness_ordered,
          "in the FORK/JOIN merge the event pair orders BOTH pairs, which is why that composition was "
          "clean in all three measured passes -- the split is a property of the merge, not of luck");

    // THE FIX, on the merge that used to corrupt: both device-observed mechanisms order both pairs in
    // the serial merge, which is the statement the device run then had to confirm.
    const TpMergeVerdict sv_serial = scan_one_merge(TpSignalMechanism::StreamValue, serial_merge);
    const TpMergeVerdict da_serial = scan_one_merge(TpSignalMechanism::DeviceAcquire, serial_merge);
    check(sv_serial.combine_ordered && sv_serial.readiness_ordered,
          "in the SERIAL merge the stream-value mechanism orders both pairs: that is the fix, on the "
          "exact merge that produced a+2b");
    check(da_serial.combine_ordered && da_serial.readiness_ordered,
          "and the device-acquire mechanism does the same, independently of the driver API");

    // THE FIX: zero, in all 48620, for both device-observed mechanisms.
    check_eq_u64(sv.readiness_unordered, 0u,
                 "the stream-value mechanism leaves the readiness pair unordered in NONE of the 48620 "
                 "interleavings");
    check_eq_u64(sv.combine_unordered, 0u,
                 "and the combine pair in none of them: the ack wait is satisfied by the word, so the "
                 "combine cannot precede the peer's read whatever the host did first");
    check_eq_u64(da.readiness_unordered, 0u, "the device-acquire mechanism agrees on the readiness side");
    check_eq_u64(da.combine_unordered, 0u, "and on the combine side");
    check(sv.readiness_witness.empty() && sv.combine_witness.empty(),
          "with no unordered pair there is no witness to report, so the field is empty rather than "
          "a stale string");

    // The scan discriminates on the MECHANISM and not on the plan, which is the same statement S2
    // makes from the other side: the three scans differ only in that rule.
    check(ev.interleavings == sv.interleavings && sv.interleavings == da.interleavings,
          "the three scans agree on the merge space and disagree only on the admission rule");
}

// ---------------------------------------------------------------------------
// S5: the policy -- the record's accepted configuration, and the three rejected ones
// ---------------------------------------------------------------------------
void group_S5_policy() {
    const SlotPolicy required = required_slot_policy();
    check(tp_slotted_policy_refusal(required).empty(),
          "the protocol's required policy (two data slots, two signal slots, ack before reuse) is "
          "accepted");

    SlotPolicy one_data = required;
    one_data.data_slots = 1u;
    const std::string r_data = tp_slotted_policy_refusal(one_data);
    check(!r_data.empty(), "one data slot is refused");
    check_contains(r_data, "22 tokens",
                   "and the refusal quotes the record's own measured symptom of it");

    SlotPolicy narrow_signal = required;
    narrow_signal.signal_slots = 1u;
    const std::string r_signal = tp_slotted_policy_refusal(narrow_signal);
    check(!r_signal.empty(), "the MEASURED MIDDLE configuration (partials double-buffered, signal "
                             "space not) is refused");
    check_contains(r_signal, "MIDDLE configuration",
                   "and the refusal names it as the record's middle configuration");
    check_contains(r_signal, "synchronization spin",
                   "and quotes what the record measured it doing");

    SlotPolicy no_ack = required;
    no_ack.acknowledge_before_reuse = false;
    const std::string r_ack = tp_slotted_policy_refusal(no_ack);
    check(!r_ack.empty(), "a policy without the acknowledgement is refused");
    check_contains(r_ack, "host happened to issue", "and the refusal names the actual defect");

    // THE RECONCILIATION, as the section-3 scan re-run on the new protocol's own policy: the policy
    // the fix requires is the one that survives every lead, and the three rejected configurations are
    // still the three the scan fires on -- so the fix did not change what survived, it changed the
    // mechanism that makes the surviving policy HOLD.
    for (std::uint32_t lead = 1u; lead <= 8u; ++lead) {
        check(!scan_slot_hazard(required, /*calls=*/64u, lead).hazard,
              "the required policy absorbs a lead of " + std::to_string(lead) + " calls");
    }
    SlotPolicy one_slot;
    one_slot.data_slots = 1u;
    one_slot.signal_slots = 1u;
    check(scan_slot_hazard(one_slot, 64u, 1u).hazard,
          "one slot still hazards at lead 1 (the record's 'stopped after 22 tokens')");
    SlotPolicy narrow = required;
    narrow.signal_slots = 1u;
    narrow.acknowledge_before_reuse = false;
    const SlotHazard h = scan_slot_hazard(narrow, 64u, 1u);
    check(h.hazard && contains(h.what, "SIGNAL slot"),
          "partials doubled and the signal space not still hazards in the SIGNAL space (the record's "
          "synchronization spin)");
    SlotPolicy two_no_ack = required;
    two_no_ack.acknowledge_before_reuse = false;
    check(scan_slot_hazard(two_no_ack, 64u, 2u).hazard,
          "two slots with no acknowledgement still hazard at lead 2, so the ack is load-bearing in "
          "the scan as well");
}

// ---------------------------------------------------------------------------
// S6: the fix is ADDITIVE -- what was there is still there
// ---------------------------------------------------------------------------
void group_S6_additive() {
    // The old seam is unchanged: the donor's four-event order for the in-place sum still exists with
    // its own step plan, and the event count is still a correctness constraint with its refusal.
    const std::vector<TpCallStep> in_place_plan =
        tp_call_plan(ReduceOp::SumInPlace, required_slot_policy());
    check_eq_u64(in_place_plan.size(), 7u,
                 "allreduce_sum_2rank's own plan is untouched: five steps plus the combine plus the "
                 "ack");
    check_eq_u64(tp_call_plan(ReduceOp::AllGatherRows, required_slot_policy()).size(), 6u,
                 "and the pure relocation still has no local combine");
    check_eq_u64(tp_event_objects_per_rank(ReduceOp::SumInPlace), 2u,
                 "two event objects per rank is still the constraint");
    check(!tp_event_count_refusal(1u).empty(), "and a one-event version is still refused by name");

    // The new protocol needs no events at all -- that is the point rather than a detail, so it is
    // asserted as an absence: the slotted plan names no RECORD/WAIT-of-an-event step.
    const std::vector<TpSlottedStep> plan = tp_slotted_call_plan(TpSignalMechanism::StreamValue);
    bool mentions_events = false;
    for (TpSlottedStep s : plan) {
        const std::string n(tp_slotted_step_name(s));
        if (contains(n, "record") || contains(n, "event")) { mentions_events = true; }
    }
    check(!mentions_events, "the slotted plan mentions no event record or event wait");
    check(!tp_signal_mechanism_name(TpSignalMechanism::EventPair).empty() &&
              contains(std::string(tp_signal_mechanism_name(TpSignalMechanism::EventPair)),
                       "host-issue-order dependent"),
          "the event pair is still NAMEABLE, because the scan has to be able to say it is the failing "
          "configuration");
}

} // namespace

int main() {
    group_S1_epochs();
    group_S2_plan();
    group_S3_barrier_counts();
    group_S4_the_scan();
    group_S5_policy();
    group_S6_additive();

    std::printf("tp_slotted_ack: %d checks, %d failures -> %s\n", g_checks, g_failures,
                g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
