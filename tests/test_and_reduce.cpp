// test_and_reduce.cpp -- THE AND-REDUCE, the ONE reduction the plan asks for and NO reference on
// this box supplies, as host arithmetic.
//
// Everything here is a pure function of its inputs, so the whole file runs with plain
// `g++ -std=c++20`, no CUDA header and no device. What it therefore PROVES and what it does NOT:
//
//   PROVES  that the named boolean op agrees with the boolean AND on every canonical input and
//           REFUSES rather than folds the moment a word leaves {0,1}; that `{1,1,2}` at
//           world_size 3 is precisely the input where the two borrowed readings disagree, so
//           neither can be the op; that the protocol's own parts (no collective at world_size 1,
//           a LocalCombine at the position the plan gives it, two event objects) are bound to the
//           seam rather than re-derived here; and that the guard is the DECODE, by showing the
//           permissive route is reachable without it.
//   DOES NOT see a GPU. One physical GPU, no MIG, no MPS: the CUDA half of core/and_reduce.h
//           compiles and its 2-rank form is exercised by tools/mg3/and_reduce_probe.cu under the
//           project's GPU lock. No line here measures anything.
//
// The reduction's PROVENANCE is a file and a line in core/and_reduce.h's header; what is asserted
// below is the arithmetic and the protocol, not the provenance.

#include "core/and_reduce.h"
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
                    static_cast<unsigned long long>(got),
                    static_cast<unsigned long long>(want));
    }
}

// A refusal is only useful if it NAMES the thing. These two helpers keep the assertions about the
// text honest instead of matching on a phrase that a rewrite would silently drop.
bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

void check_contains(const std::string& haystack, const std::string& needle, const std::string& what) {
    check(contains(haystack, needle), what + " (missing \"" + needle + "\")");
}

// ---------------------------------------------------------------------------
// T1: the wire domain. The sender cannot leave {0,1}; the receiver must refuse to pretend it did.
// ---------------------------------------------------------------------------
void group_T1_wire_domain() {
    // The sender-side half: the only constructors are canonical, so a rank cannot produce a 2.
    check_eq_u64(AdmitVerdict::admit().raw(), 1u, "T1 admit encodes as 1");
    check_eq_u64(AdmitVerdict::refuse().raw(), 0u, "T1 refuse encodes as 0");
    check_eq_u64(AdmitVerdict::from_local_predicate(true).raw(), 1u,
                 "T1 the local predicate maps to the canonical word");
    check_eq_u64(AdmitVerdict::from_local_predicate(false).raw(), 0u,
                 "T1 the local predicate maps to the canonical word (false)");
    check(AdmitVerdict::admit().admitted() && !AdmitVerdict::refuse().admitted(),
          "T1 the sender-side type round-trips through its own accessors");

    // The receiver-side half: canonical words decode, and nothing else does.
    const AdmitDecode zero = decode_admit_word(0u);
    const AdmitDecode one  = decode_admit_word(1u);
    check(zero.canonical && !zero.admitted() && zero.violation.empty(),
          "T1 word 0 decodes as a canonical REFUSE with no violation text");
    check(one.canonical && one.admitted() && one.violation.empty(),
          "T1 word 1 decodes as a canonical ADMIT with no violation text");

    // THE LOAD-BEARING NEGATIVE, one word at a time: neither reading is taken. A refusal that
    // said `admitted() == true` would be the permissive route; one that said `canonical == true`
    // would be the lossy route. Both are red here.
    for (const std::uint32_t bad : {2u, 3u, 7u, 255u, 256u, kAdmitWordViolation}) {
        const AdmitDecode d = decode_admit_word(bad);
        check(!d.canonical, "T1 word " + std::to_string(bad) + " is not canonical");
        check(!d.admitted(),
              "T1 word " + std::to_string(bad) + " does not read as an ADMIT (no nonzero=>true)");
        check(!d.violation.empty(),
              "T1 word " + std::to_string(bad) + " carries a refusal instead of a boolean");
        check_contains(d.violation, std::to_string(bad),
                       "T1 the refusal quotes the offending raw value");
        check_eq_u64(d.raw, bad, "T1 the refusal keeps the raw value for a post-mortem");
    }
    check(admit_word_is_canonical(0u) && admit_word_is_canonical(1u) && !admit_word_is_canonical(2u),
          "T1 the canonical-domain predicate is exactly {0,1}");
    // The sentinel is outside the domain by construction, so it cannot collide with a verdict --
    // which is why it is not, say, 2.
    check(!admit_word_is_canonical(kAdmitWordViolation),
          "T1 the device-side violation sentinel is outside the canonical domain");
}

// ---------------------------------------------------------------------------
// T2: the identity taken by construction, and the world sizes that issue no collective
// ---------------------------------------------------------------------------
void group_T2_identity() {
    for (const std::uint32_t world : {0u, 1u}) {
        const AndReduceOutcome id = and_reduce_identity(world);
        check(id.ok, "T2 the identity at world " + std::to_string(world) + " holds");
        check(id.admitted, "T2 the AND identity is TRUE (a vacuous AND)");
        check(!id.collective_issued,
              "T2 world " + std::to_string(world) + " issues NO collective");
        check(!and_reduce_needs_collective(world),
              "T2 and_reduce_needs_collective agrees for world " + std::to_string(world));
    }
    // The empty input reaches the same identity through the collective's own entry point, so the
    // two routes cannot disagree.
    const AndReduceOutcome empty = and_reduce_wire({});
    check(empty.ok && empty.admitted && !empty.collective_issued,
          "T2 the empty word vector is the identity, reached through and_reduce_wire");
    check_eq_u64(empty.world_size, 0u, "T2 the empty input's world size is 0");

    // One rank: that rank's own verdict, and still no collective.
    const AndReduceOutcome one_rank_refuse = and_reduce_wire({0u});
    check(one_rank_refuse.ok && !one_rank_refuse.admitted && !one_rank_refuse.collective_issued,
          "T2 world 1 carries the single rank's own REFUSE and issues nothing");
    const AndReduceOutcome one_rank_admit = and_reduce_verdicts({AdmitVerdict::admit()});
    check(one_rank_admit.ok && one_rank_admit.admitted && !one_rank_admit.collective_issued,
          "T2 world 1 carries the single rank's own ADMIT and issues nothing");

    // And the same single-rank path still refuses a word that is not a verdict: "there is only
    // one rank" is not a licence to accept anything.
    const AndReduceOutcome one_rank_bad = and_reduce_wire({2u});
    check(!one_rank_bad.ok && !one_rank_bad.admitted,
          "T2 world 1 with a non-canonical word is a violation, not an ADMIT");
    check_contains(one_rank_bad.refusal, "world_size 1", "T2 the world-1 refusal says so");

    check(and_reduce_needs_collective(2u) && and_reduce_needs_collective(8u),
          "T2 world n >= 2 does need a collective");
    check_eq_u64(and_reduce_event_objects(), 2u, "T2 the AND takes two event objects per rank");
}

// ---------------------------------------------------------------------------
// T3: on the canonical domain, exhaustive -- the named op IS the boolean AND, for every input
// ---------------------------------------------------------------------------
void group_T3_canonical_domain_is_exhaustive() {
    // n = 1..3 is 2 + 4 + 8 = 14 inputs, which is every verdict vector up to world_size 3 and
    // covers the {1,1,2}-at-3 neighbourhood exhaustively for the values that are IN the domain.
    int cases = 0;
    for (std::uint32_t world = 1; world <= 3; ++world) {
        const std::uint32_t n = 1u << world; // 2^world verdict vectors
        for (std::uint32_t mask = 0; mask < n; ++mask) {
            std::vector<std::uint32_t> wire;
            std::vector<std::uint8_t> bytes;
            std::vector<AdmitVerdict> verdicts;
            std::vector<bool> bools;
            for (std::uint32_t r = 0; r < world; ++r) {
                const bool bit = ((mask >> r) & 1u) != 0u;
                wire.push_back(bit ? 1u : 0u);
                bytes.push_back(bit ? 1u : 0u);
                verdicts.push_back(bit ? AdmitVerdict::admit() : AdmitVerdict::refuse());
                bools.push_back(bit);
            }
            const AndReduceOutcome wire_outcome = and_reduce_wire(wire);
            const AndReduceOutcome typed_outcome = and_reduce_verdicts(verdicts);

            // (a) a canonical input is never a violation.
            check(wire_outcome.ok,
                  "T3 mask " + std::to_string(mask) + " world " + std::to_string(world) +
                      " is canonical and not a violation");
            // (b) the typed entry point and the wire entry point are the SAME function.
            check(typed_outcome.ok && typed_outcome.admitted == wire_outcome.admitted,
                  "T3 the typed and wire entry points agree (mask " + std::to_string(mask) + ")");
            // (c) it equals the plan's own predicate...
            check(wire_outcome.admitted == global_admit(bytes),
                  "T3 the fold equals global_admit on the canonical domain (mask " +
                      std::to_string(mask) + ")");
            // (d) ...and the seam's own boolean fold...
            check(wire_outcome.admitted == all_reduce_and(bools),
                  "T3 the fold equals tp_transport's all_reduce_and (mask " +
                      std::to_string(mask) + ")");
            // (e) ...and the sum encoding, WHICH IS ONLY SOUND HERE. That "only here" is the whole
            // reason the op exists, and T4 is where it goes red.
            std::uint32_t sum = 0;
            for (const std::uint32_t v : wire) { sum += v; }
            check(wire_outcome.admitted == (sum == world),
                  "T3 on the canonical domain the sum encoding agrees (mask " +
                      std::to_string(mask) + ")");
            check_eq_u64(wire_outcome.world_size, world, "T3 the world size is the word count");
            check(wire_outcome.collective_issued == (world > 1),
                  "T3 the collective is issued exactly when world > 1");
            ++cases;
        }
    }
    check_eq_u64(static_cast<std::uint64_t>(cases), 14u,
                 "T3 the exhaustive sweep covered 2+4+8 = 14 canonical inputs");
}

// ---------------------------------------------------------------------------
// T4: {1,1,2} at world_size 3 -- THE case, through all three readings
// ---------------------------------------------------------------------------
void group_T4_the_counterexample() {
    // The record's witness, executed.
    const AndDomainProbe p = and_domain_probe({1u, 1u, 2u});
    check_eq_u64(p.wire.size(), 3u, "T4 the witness is a world of 3");

    // Reading 1: the sum encoding. 1+1+2 = 4 against world_size 3 -> REFUSE.
    check(!p.sum_encoding_says, "T4 the sum encoding says REFUSE on {1,1,2}");

    // Reading 2: the any-nonzero reading of the same word, which is `global_admit()`'s own body
    // (`if (admit == 0) return false;`) reached by skipping the decode. It says ADMIT.
    check(p.boolean_and_says, "T4 the any-nonzero reading says ADMIT on {1,1,2}");
    check(global_admit({1u, 1u, 2u}),
          "T4 global_admit() really does read 2 as true when it is handed the raw value");
    check(p.sum_encoding_says != p.boolean_and_says,
          "T4 THE TWO BORROWED READINGS DISAGREE -- so neither can be the op");

    // Reading 3: the named op. Not ADMIT, not REFUSE -- a protocol violation.
    check(!p.named_op_ok, "T4 the named op does not fold {1,1,2}: it is a protocol violation");
    check(!p.named_op_says, "T4 the named op reports no verdict rather than picking one");
    check_contains(p.named_op_refusal, "rank 2", "T4 the violation names the offending rank");
    check_contains(p.named_op_refusal, "of 3", "T4 and the world size it was an offender in");
    check_contains(p.named_op_refusal, "{0,1}", "T4 and the domain it left");
    check_contains(p.named_op_refusal, "REFUSE", "T4 and the reading it declines to take");
    check_contains(p.named_op_refusal, "ADMIT", "T4 and the other reading it declines to take");
    check(!p.all_three_agree, "T4 the three readings do not agree on this input");

    // The cross-file check: tp_transport.h's own counterexample sees the same disagreement, so
    // the two files cannot drift on the case that motivated the op.
    check(p.tp_transport_agrees,
          "T4 tp_transport's and_encoding_counterexample() reports the same disagreement");

    // AND THE GUARD IS THE DECODE, not the predicate: the predicate is unchanged and still
    // permissive, and the ONLY thing standing between the raw word and a wrongly-admitted page is
    // `decode_admit_word()`. Read this as the red control for "we could have just used a sum".
    const AndReduceOutcome through_the_op = and_reduce_wire({1u, 1u, 2u});
    check(!through_the_op.ok && !through_the_op.admitted && through_the_op.collective_issued,
          "T4 the collective refuses the protocol rather than admitting the page");
    check(global_admit({1u, 1u, 2u}) == true && all_reduce_and({true, true}) == true,
          "T4 and the two readings the op declines to take would BOTH have admitted it "
          "(the sum encoding by refusing, the predicate by allowing)");

    // Neither "always a violation" nor "never a violation" survives: the two well-formed controls
    // around this input are clean, and all three readings agree on them.
    const AndDomainProbe all_admit = and_domain_probe({1u, 1u, 1u});
    check(all_admit.named_op_ok && all_admit.named_op_says && all_admit.all_three_agree,
          "T4 control {1,1,1} at 3: all three readings say ADMIT");
    const AndDomainProbe one_refuse = and_domain_probe({1u, 0u, 1u});
    check(one_refuse.named_op_ok && !one_refuse.named_op_says && one_refuse.all_three_agree,
          "T4 control {1,0,1} at 3: all three readings say REFUSE");
    const AndDomainProbe all_refuse = and_domain_probe({0u, 0u, 0u});
    check(all_refuse.named_op_ok && !all_refuse.named_op_says && all_refuse.all_three_agree,
          "T4 control {0,0,0} at 3: all three readings say REFUSE");

    // The permissive route is UNREACHABLE through this file, for every out-of-domain word, while
    // the predicate on the same raw bytes is still permissive -- which is what makes the decode
    // load-bearing rather than decorative.
    for (const std::uint32_t bad : {2u, 3u, 255u, 256u, kAdmitWordViolation}) {
        const AndReduceOutcome out = and_reduce_wire({1u, 1u, bad});
        check(!out.ok && !out.admitted,
              "T4 word " + std::to_string(bad) + " cannot reach global_admit()'s permissive route");
        check_contains(out.refusal, "of 3",
                       "T4 the refusal for word " + std::to_string(bad) + " names the world size");
    }

    // THE TYPE-LEVEL HALF, stated where it belongs: `all_reduce_and()` from tp_transport.h takes a
    // `std::vector<bool>`, so {1,1,2} is not merely refused there -- it cannot be written down. The
    // wire form is what reintroduces the possibility, and the decoder is what answers it.
    check(all_reduce_and({true, true, true}) && !all_reduce_and({true, false, true}),
          "T4 tp_transport's all_reduce_and is the boolean fold on the domain it can express");
}

// ---------------------------------------------------------------------------
// T5: the 2-rank shape, and the commutativity the pull protocol implies
// ---------------------------------------------------------------------------
void group_T5_two_rank_shape() {
    for (const int rank : {0, 1}) {
        const AndReduceOutcome out =
            and_reduce_peer_words(AdmitVerdict::admit(), AdmitVerdict::refuse(), rank);
        check(out.ok && !out.admitted && out.collective_issued,
              "T5 rank " + std::to_string(rank) + ": admit AND refuse is a REFUSE");
    }
    const AndReduceOutcome by_rank_0 =
        and_reduce_peer_words(AdmitVerdict::admit(), AdmitVerdict::admit(), 0);
    const AndReduceOutcome by_rank_1 =
        and_reduce_peer_words(AdmitVerdict::admit(), AdmitVerdict::admit(), 1);
    check(by_rank_0.ok && by_rank_1.ok && by_rank_0.admitted == by_rank_1.admitted,
          "T5 the AND is commutative across the two ranks' orderings");
    check(by_rank_0.admitted && by_rank_0.admitted == and_reduce_wire({1u, 1u}).admitted,
          "T5 the 2-rank helper is the wire fold, not a second implementation");

    // A rank that is not a rank of a 2-rank collective is refused rather than silently treated as
    // rank 0 -- the same false-negative shape as reading a failed probe as "no peer access".
    const AndReduceOutcome bad_rank =
        and_reduce_peer_words(AdmitVerdict::admit(), AdmitVerdict::admit(), 2);
    check(!bad_rank.ok && !bad_rank.admitted, "T5 self_rank 2 is refused");
    check_contains(bad_rank.refusal, "rank 0 and rank 1",
                   "T5 the refusal names the ranks the protocol is defined for");
}

// ---------------------------------------------------------------------------
// T6: the transport binding -- the plan, the events and the provenance all come from the seam
// ---------------------------------------------------------------------------
void group_T6_transport_binding() {
    const SlotPolicy policy = required_slot_policy();
    const std::vector<TpCallStep> plan = and_reduce_call_plan(policy);

    // It IS the seam's plan for this reduction, not a local re-derivation.
    const std::vector<TpCallStep> seam_plan = tp_call_plan(ReduceOp::AndBool, policy);
    check(plan.size() == seam_plan.size(), "T6 the AND's plan is the seam's plan (length)");
    bool same = true;
    for (std::size_t i = 0; i < plan.size() && i < seam_plan.size(); ++i) {
        if (plan[i] != seam_plan[i]) { same = false; }
    }
    check(same, "T6 every step of the AND's plan is the seam's step for ReduceOp::AndBool");

    // The ORDER the reduction depends on: the peer's read of THIS rank's word is waited for
    // BEFORE the local combine, and the local combine is present exactly once. The AND is not in
    // place -- the peer's word must not be overwritten before it has been read.
    std::size_t at_wait_pull_done = plan.size();
    std::size_t at_combine = plan.size();
    std::size_t combines = 0;
    for (std::size_t i = 0; i < plan.size(); ++i) {
        if (plan[i] == TpCallStep::WaitPeerPullDone) { at_wait_pull_done = i; }
        if (plan[i] == TpCallStep::LocalCombine) { at_combine = i; ++combines; }
    }
    check_eq_u64(static_cast<std::uint64_t>(combines), 1u,
                 "T6 the AND's plan has exactly ONE local combine (it is not in place)");
    check(at_combine != plan.size() && at_wait_pull_done != plan.size() &&
              at_combine > at_wait_pull_done,
          "T6 the local combine is AFTER the peer's-read barrier, as tp_call_plan asserts");
    check_eq_u64(static_cast<std::uint64_t>(plan.size()), 7u,
                 "T6 the acknowledged plan is the seven steps (five + combine + ack)");

    // The acknowledgement: present when the slot policy asks for it, absent when it does not. So
    // the plan responds to a decision rather than to a constant.
    SlotPolicy no_ack = policy;
    no_ack.acknowledge_before_reuse = false;
    const std::vector<TpCallStep> bare = and_reduce_call_plan(no_ack);
    check_eq_u64(static_cast<std::uint64_t>(bare.size()),
                 static_cast<std::uint64_t>(plan.size() - 1u),
                 "T6 dropping acknowledge_before_reuse drops exactly the publish-ack step");
    bool has_ack = false;
    for (const TpCallStep s : bare) {
        if (s == TpCallStep::PublishAck) { has_ack = true; }
    }
    check(!has_ack, "T6 the unacknowledged plan carries no PublishAck");

    // The event count is the seam's, and the one-event version still earns its refusal.
    check_eq_u64(and_reduce_event_objects(), tp_event_objects_per_rank(ReduceOp::AndBool),
                 "T6 the AND's event count is the seam's");
    check(tp_event_count_refusal(1u).size() > 0,
          "T6 one event object per rank is still refused with a reason");
    check(tp_event_count_refusal(2u).empty(), "T6 two event objects per rank is accepted");
    check(and_reduce_ack_is_the_second_event_pair(),
          "T6 the acknowledgement is the second event pair, and that is stated");

    // The provenance, tied to the seam rather than repeated as prose: this reduction is the one
    // NEITHER reference supplies, and the sum-only cost is exactly one collective -- which is the
    // encoding T4 shows to be unsound outside {0,1}.
    check(reduce_op_provenance(ReduceOp::AndBool) == ReduceOpProvenance::NeitherReference,
          "T6 the AND is the reduction neither reference supplies");
    const SumOnlyCost cost = sum_only_cost(ReduceOp::AndBool);
    check_eq_u64(cost.collectives.size(), 1u, "T6 the sum-only cost is one collective");
    check_eq_u64(cost.collectives.empty() ? 0u : (cost.collectives[0] == "ar" ? 1u : 0u), 1u,
                 "T6 and that collective is an all-reduce");
    check_eq_u64(cost.local_steps, 0u, "T6 with no local steps -- which is why it LOOKS free");
    check(!cost.host_fixup_required, "T6 and needs no host fixup, so nothing on the path notices");
}

// ---------------------------------------------------------------------------
// T7: the empty world -- a disagreement in the plan's own code, ASSERTED and not fixed
// ---------------------------------------------------------------------------
// This is an observation, not a change: `global_admit({})` and `residency_state({})` are both in
// core/shard_plan.h, they are both pre-existing, and they answer the empty world differently. It
// is recorded here because the AND-reduce is the function that will be called with the verdict
// vector, so a reader of THIS file is the one who needs to know it.
void group_T7_empty_world_observation() {
    check(global_admit({}), "T7 global_admit({}) is TRUE (a vacuous AND)");
    check(residency_state({}) == ResidencyState::FullyAbsent,
          "T7 residency_state({}) is FullyAbsent -- the same input, the other answer");
    check(global_admit({}) == true && residency_state({}) == ResidencyState::FullyAbsent,
          "T7 the two pre-existing readings of the empty world disagree (observed, not changed)");
    const AndReduceOutcome empty = and_reduce_wire({});
    check(empty.ok && empty.admitted && empty.world_size == 0u,
          "T7 this file takes the AND identity for the empty world and says world_size is 0, so a "
          "caller cannot mistake it for a world of one");
}

} // namespace

int main() {
    group_T1_wire_domain();
    group_T2_identity();
    group_T3_canonical_domain_is_exhaustive();
    group_T4_the_counterexample();
    group_T5_two_rank_shape();
    group_T6_transport_binding();
    group_T7_empty_world_observation();

    std::printf("and_reduce: %d checks, %d failures -> %s\n", g_checks, g_failures,
                g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
