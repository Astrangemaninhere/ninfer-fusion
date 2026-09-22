#pragma once

// and_reduce.h -- THE ONE REDUCTION THE PLAN ASKS FOR AND NO REFERENCE SUPPLIES, built as a
// COLLECTIVE rather than left as a convention.
//
// WHY THIS FILE EXISTS. Our plan (`docs/maintainer/multi-device-and-shard-plan.md`) asks for
// "an all-reduce AND of the eviction admit verdict" and both reference forks are silent on it:
//
//   * the ninfer donor has NO boolean op at all -- two ops, `allreduce_sum` and `allgather_rows`;
//   * 1Cat's `CustomAllreduce::allreduce` (`csrc/custom_all_reduce.cuh:1906`, dispatched at
//     `csrc/custom_all_reduce.cu:24-28`) covers `Float / Half / BFloat16` only, and its own
//     dispatcher test pins `(torch.int8, False)` and `(torch.float8_e4m3fn, False)`;
//   * the record-named `data/v100-skinny/fork_patches/custom_all_reduce.py` (32,973 B) has
//     `logical` 0 and `bitwise` 0 hits -- its one non-sum reduction is the SELECTION reduction
//     `cross_device_top1_argmax`.
//
// So there is nothing to borrow and the shape has to be specified. `src/core/tp_transport.h`
// already carries the two host-side pieces -- `all_reduce_and(std::vector<bool>)` and
// `and_encoding_counterexample()` -- and names `ReduceOp::AndBool` in the reduction set. What was
// still missing is the part a transport actually needs, and it is what this file adds:
//
//   1. THE WIRE DOMAIN. A collective crosses processes as a WORD, and a word is an integer type,
//      so the sender-side guarantee ("a rank cannot report a value outside {0,1}") stops at the
//      process boundary. The receiver therefore needs a DECODER, or the aliasing the record's
//      counterexample exposes comes straight back.
//   2. THE FOLD. `global_admit()` (`src/core/shard_plan.h:450`) is the predicate, and it is left
//      as it is: `admit != 0` is TRUE for 2. This file does not change it; it makes the input
//      canonical BEFORE the predicate sees it, so the predicate's permissive on-any-nonzero
//      reading can no longer be reached by an out-of-domain word.
//   3. THE REFUSAL. A non-canonical word is a PROTOCOL VIOLATION, which is a THIRD outcome --
//      not "admit" and not "refuse the page". Folding it either way is the bug: REFUSE loses the
//      page, and ADMIT admits a page a rank never agreed to hold. The counterexample's whole
//      point is that `{1,1,2}` is exactly the case where those two differ, so a named op that
//      silently picks one of them would be no better than the sum encoding it replaces.
//   4. THE CUDA HALF. The local step of the collective, placed where `tp_call_plan(AndBool, ...)`
//      says it goes (after `WaitPeerPullDone`), with the violation detectable on the DEVICE
//      because a captured graph cannot take a host round trip to find it.
//
// WHAT THE PLAN'S OWN ARITHMETIC SAYS, for the record: a sum CAN encode an AND on the domain
// {0,1} and only there. The record's witness is `{1,1,2}` at world_size 3 -- the sum is 4 against
// a world_size of 3, so the sum encoding says REFUSE while the boolean AND over the ranks says
// ADMIT. `and_domain_probe()` in section 4 executes that case through all three readings and
// returns what each one says, as data. Test T5 asserts the third reading.
//
// Host-only by construction except for the guarded section at the bottom: no CUDA header at the
// top level, no device query, so every decision here is exercisable with plain g++ on a box with
// no GPU -- which is the only form of evidence this machine can produce for tensor parallelism
// (one physical GPU, no MIG, no MPS).

#include "core/shard_plan.h"
#include "core/tp_transport.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::multi {

// ---------------------------------------------------------------------------
// 0. THE WIRE DOMAIN -- two halves of one guarantee, and they are not interchangeable
// ---------------------------------------------------------------------------

// The canonical domain, as a TYPE. Two enumerators and no others, so `AdmitVerdict` below cannot
// express anything outside it: this is the sender-side half, and it is the half that
// `and_encoding_counterexample()` argues for. It is reached only at compile time.
enum class AdmitWord : std::uint32_t {
    Refuse = 0u,
    Admit  = 1u,
};

[[nodiscard]] constexpr std::uint32_t admit_word_raw(AdmitWord word) noexcept {
    return static_cast<std::uint32_t>(word);
}

[[nodiscard]] constexpr bool admit_word_is_admit(AdmitWord word) noexcept {
    return word == AdmitWord::Admit;
}

// The receiver-side half, and the reason this file is not just a `bool` fold: a word that arrives
// over a transport has already left the type system. `kAdmitWordViolation` is outside the
// canonical domain by construction, so it can never collide with a real verdict -- unlike, say,
// using 2 as a sentinel, which is a value a broken rank can actually produce.
inline constexpr std::uint32_t kAdmitWordViolation = 0xFFFFFFFFu;

[[nodiscard]] constexpr bool admit_word_is_canonical(std::uint32_t raw) noexcept {
    return raw <= 1u;
}

// The SENDER-side value. Constructible ONLY from a bool, a plan decision, or an `AdmitWord`, so a
// rank that computes "can I take this page?" cannot put a 2 on the wire by accident, by an
// arithmetic slip, or by passing a byte count where a verdict was wanted.
struct AdmitVerdict {
    AdmitWord word = AdmitWord::Refuse;

    [[nodiscard]] static constexpr AdmitVerdict refuse() noexcept { return AdmitVerdict{AdmitWord::Refuse}; }
    [[nodiscard]] static constexpr AdmitVerdict admit() noexcept { return AdmitVerdict{AdmitWord::Admit}; }

    // The eviction driver's question, in the form it is actually asked: the page may leave only
    // if EVERY rank can take it, and a rank answers with its own local predicate.
    [[nodiscard]] static constexpr AdmitVerdict from_local_predicate(bool can_take) noexcept {
        return can_take ? admit() : refuse();
    }

    [[nodiscard]] constexpr std::uint32_t raw() const noexcept { return admit_word_raw(word); }
    [[nodiscard]] constexpr bool admitted() const noexcept { return admit_word_is_admit(word); }
};

// The RECEIVER-side decode, with the reason carried as a string rather than as a bool, because a
// caller that gets a violation has to be told WHICH rank and WHICH value -- "the AND-reduce
// failed" sends the reader to the wrong file.
struct AdmitDecode {
    bool canonical = false;
    AdmitWord word = AdmitWord::Refuse;
    std::uint32_t raw = 0;
    std::string violation; // empty exactly when canonical

    [[nodiscard]] bool admitted() const noexcept { return canonical && admit_word_is_admit(word); }
};

// Decode ONE word. Non-canonical is refused with the raw value quoted, never coerced: coercing
// any nonzero to `true` is precisely the silently-permissive reading that makes `{1,1,2}` admit
// a page, and coercing it to `false` would throw away a page on the strength of a corrupt word.
[[nodiscard]] inline AdmitDecode decode_admit_word(std::uint32_t raw) {
    AdmitDecode out;
    out.raw = raw;
    if (admit_word_is_canonical(raw)) {
        out.canonical = true;
        out.word      = (raw == 0u) ? AdmitWord::Refuse : AdmitWord::Admit;
        return out;
    }
    out.canonical = false;
    out.violation =
        "an admit word of " + std::to_string(raw) +
        " arrived over the transport, and the canonical domain is {0,1}. This is not a verdict "
        "that can be folded: the sum encoding of an AND reads it as REFUSE (it is not 1) while "
        "every 'nonzero means true' reading of the same word reads it as ADMIT, and those are "
        "different outcomes for the eviction driver. Neither is taken here. A word outside {0,1} "
        "means the sender's type guarantee was bypassed, so the first question is why, not which "
        "boolean to pick.";
    return out;
}

// ---------------------------------------------------------------------------
// 1. THE COLLECTIVE -- the identity taken by construction, the world size checked, then the fold
// ---------------------------------------------------------------------------

// A THIRD value again, for the same reason `TopologyDecision` and `CombineOutcome` have three:
// "the protocol held and the answer is refuse" and "the protocol did not hold and there is no
// answer" are different amounts of code and different failure modes. `ok == false` must never be
// read as `admitted == false`.
struct AndReduceOutcome {
    bool ok = false;                  // the PROTOCOL held: every word was canonical and counted
    bool admitted = false;            // meaningful ONLY when ok
    bool collective_issued = false;   // false for world_size <= 1: the identity is not a call
    std::uint32_t world_size = 1;
    std::string refusal;              // empty exactly when ok

    [[nodiscard]] bool is_violation() const noexcept { return !ok; }
};

// The identity, as a value rather than as a branch a caller can forget: the AND of one rank is
// that rank's own verdict, and the AND of none is TRUE. NO COLLECTIVE IS ISSUED, which is the
// property `and_reduce_is_identity()` in tp_transport.h states and this function now honours.
[[nodiscard]] inline AndReduceOutcome and_reduce_identity(std::uint32_t world_size) {
    AndReduceOutcome out;
    out.ok                 = true;
    out.admitted           = true; // the AND identity
    out.collective_issued  = false;
    out.world_size         = world_size;
    return out;
}

// The collective over the words as they ARRIVED. `wire.size()` IS the world size -- there is one
// word per rank and the world size is not passed separately, so a caller cannot supply a world
// size that disagrees with the number of verdicts it actually has. That is the same choice
// `and_encoding_counterexample()` makes (`per_rank_admit.size()` is the world size), and it is
// what makes the `{1,1,2}`-at-3 case expressible as an input rather than as a comment.
[[nodiscard]] inline AndReduceOutcome and_reduce_wire(const std::vector<std::uint32_t>& wire) {
    if (wire.size() <= 1) {
        // Zero ranks is the AND identity; one rank is that rank's verdict. Both issue nothing.
        AndReduceOutcome out = and_reduce_identity(static_cast<std::uint32_t>(wire.size()));
        if (wire.size() == 1) {
            const AdmitDecode only = decode_admit_word(wire[0]);
            if (!only.canonical) {
                out.ok       = false;
                out.admitted = false; // no answer: `ok == false` must never read as a verdict
                out.refusal  = "world_size 1: " + only.violation;
            } else {
                out.admitted = only.admitted();
            }
        }
        // NOTE the empty world, because the plan's code disagrees with itself here and it is not
        // this file's call to change: `global_admit({})` is TRUE (a vacuous AND) while
        // `residency_state({})` (core/shard_plan.h:440) is `FullyAbsent`, i.e. "nothing is
        // resident". The AND identity is TRUE, which is what is returned, and the test asserts the
        // disagreement rather than papering over it. world_size 0 must not be reached in
        // production: the shard plan's own world sizes start at 1.
        return out;
    }

    std::vector<std::uint8_t> canonical; // the bytes `global_admit()` is allowed to see
    canonical.reserve(wire.size());
    for (std::size_t rank = 0; rank < wire.size(); ++rank) {
        const AdmitDecode decoded = decode_admit_word(wire[rank]);
        if (!decoded.canonical) {
            AndReduceOutcome out;
            out.ok                = false;
            out.collective_issued = true;
            out.world_size        = static_cast<std::uint32_t>(wire.size());
            out.refusal =
                "rank " + std::to_string(rank) + " of " + std::to_string(wire.size()) +
                " published a non-canonical word: " + decoded.violation;
            return out;
        }
        canonical.push_back(static_cast<std::uint8_t>(decoded.raw));
    }

    // THE FOLD IS THE PLAN'S OWN PREDICATE, unchanged: `global_admit()` (core/shard_plan.h:450)
    // is "the page leaves the device only if EVERY rank can take it", and it is handed bytes this
    // function has already proved are in {0,1}. So its `admit != 0` reading can no longer be
    // reached by an out-of-domain word, and this file does not need to touch that file.
    AndReduceOutcome out;
    out.ok                = true;
    out.admitted          = global_admit(canonical);
    out.collective_issued = true;
    out.world_size        = static_cast<std::uint32_t>(wire.size());
    return out;
}

// The type-safe entry point, for a caller that has not serialised anything yet. It exists so the
// in-process path and the wire path are provably the same function: the verdicts are mapped
// through their own canonical words and handed to `and_reduce_wire`, rather than folded by a
// second implementation that could drift from it.
[[nodiscard]] inline AndReduceOutcome and_reduce_verdicts(const std::vector<AdmitVerdict>& verdicts) {
    std::vector<std::uint32_t> wire;
    wire.reserve(verdicts.size());
    for (const AdmitVerdict& v : verdicts) { wire.push_back(v.raw()); }
    return and_reduce_wire(wire);
}

// The 2-rank pull protocol's own shape: a rank holds its own word and the peer's (the peer's was
// pulled into this rank's staging), so the fold is over exactly two words and `self_rank` decides
// only their order -- which the AND does not care about, and which is asserted in the test so
// that a future non-commutative op added next to this one cannot inherit the assumption silently.
[[nodiscard]] inline AndReduceOutcome and_reduce_peer_words(const AdmitVerdict& mine,
                                                            const AdmitVerdict& peer,
                                                            int self_rank) {
    if (self_rank != 0 && self_rank != 1) {
        AndReduceOutcome out;
        out.ok      = false;
        out.refusal = "self_rank " + std::to_string(self_rank) +
                      " is not a rank of a 2-rank collective; the pull protocol is defined for "
                      "rank 0 and rank 1 only.";
        return out;
    }
    return and_reduce_wire({mine.raw(), peer.raw()});
}

// Whether a collective has to be issued at all. Present so that "world_size 1 issues none" can be
// asserted rather than assumed, and so a caller cannot reach the identity by a different route
// than the one the outcome records.
[[nodiscard]] inline bool and_reduce_needs_collective(std::uint32_t world_size) noexcept {
    return world_size > 1;
}

// ---------------------------------------------------------------------------
// 2. THE TRANSPORT BINDING -- this reduction, on the seam, with the ordering NOT re-decided
// ---------------------------------------------------------------------------

// The call plan is `tp_call_plan(ReduceOp::AndBool, policy)`, reached through a named function so
// that the AND cannot drift from the reduction set: a caller who wants "the plan for the AND"
// gets the seam's plan, not a local re-derivation. The three properties that matter are asserted
// in the test: a LOCAL COMBINE is present (the AND is not in place -- the peer's word must not be
// overwritten before it is read), an ALL-GATHER is absent (there is nothing to relocate), and the
// acknowledgement appears exactly when the slot policy asks for it.
[[nodiscard]] inline std::vector<TpCallStep> and_reduce_call_plan(const SlotPolicy& policy) {
    return tp_call_plan(ReduceOp::AndBool, policy);
}

// The event objects, from the seam, for the same reason: the event count is a correctness
// constraint and not a knob (`tp_event_count_refusal()`), and the AND does not get to have a
// different one.
[[nodiscard]] inline std::uint32_t and_reduce_event_objects() noexcept {
    return tp_event_objects_per_rank(ReduceOp::AndBool);
}

// THE ACKNOWLEDGEMENT, RECONCILED. `tp_call_plan(AndBool, required_slot_policy())` ends with
// `PublishAck`, because the slot protocol needs a rank to be able to tell that its previous slot
// was CONSUMED before republishing into it. The donor's four-event form does not add a separate
// signal for that, and this is not a contradiction: its second event pair
// (`RecordPullDone` -> `WaitPeerPullDone`) IS the write-after-read barrier, observed by the peer
// on the peer's own stream, which is exactly what the acknowledgement is for when the two ranks
// are the whole world. The acknowledgement becomes a separate signal only when the slot is reused
// ACROSS calls by a rank that is behind -- which is the `SlotPolicy`'s business and not one
// call's. Stated here because a future reader who sees both `PublishAck` in the plan and no ack
// in the CUDA half below will otherwise add a redundant cross-rank spin -- and a cross-rank spin
// inside a capture is the documented gfx906 card-2 wedge (`docs/gfx906/TP2-SLICES.md`), which is
// a hang with a cold power cycle as its recovery, not a performance regression.
[[nodiscard]] inline bool and_reduce_ack_is_the_second_event_pair() noexcept { return true; }

// ---------------------------------------------------------------------------
// 3. THE {1,1,2} COUNTEREXAMPLE, EXECUTED THROUGH ALL THREE READINGS
// ---------------------------------------------------------------------------

// The record's own witness, as a value. `sum_encoding_says` and `boolean_and_says` are what the
// two borrowed readings produce; `named_op_ok` / `named_op_says` are what this file's collective
// produces. `tp_transport_agrees` is a CROSS-FILE CHECK rather than a restatement: it re-runs
// `and_encoding_counterexample()` from tp_transport.h on the same input, so the two files cannot
// drift into disagreeing about the case that motivated the op.
struct AndDomainProbe {
    std::vector<std::uint32_t> wire;
    bool sum_encoding_says = false; // "admit iff the ranks' values sum to exactly world_size"
    bool boolean_and_says  = false; // `global_admit()` on the raw values: `!= 0` is true for 2
    bool named_op_ok       = false; // this file's collective
    bool named_op_says     = false;
    bool tp_transport_agrees = false;
    bool all_three_agree = false;   // only expected on the canonical domain
    std::string named_op_refusal;
    std::string summary;
};

[[nodiscard]] inline AndDomainProbe and_domain_probe(const std::vector<std::uint32_t>& wire) {
    AndDomainProbe out;
    out.wire = wire;
    if (wire.empty()) { return out; }

    std::uint32_t sum = 0;
    // What a caller who SKIPPED the decode would hand to the predicate. The narrowing preserves
    // nonzero-ness exactly (`v <= 255 ? v : 255`), which is the only thing `global_admit()` reads
    // (`if (admit == 0) return false;`), so this is a faithful rendering of the any-nonzero
    // reading rather than a truncation accident of this probe.
    std::vector<std::uint8_t> raw_bytes;
    raw_bytes.reserve(wire.size());
    for (const std::uint32_t v : wire) {
        sum += v;
        raw_bytes.push_back(v <= 255u ? static_cast<std::uint8_t>(v) : static_cast<std::uint8_t>(255));
    }
    out.sum_encoding_says = (sum == wire.size());
    // THE PERMISSIVE READING, named as such: it is `global_admit()`'s own arithmetic on the raw
    // values, which is what a caller gets if the decode step is skipped.
    out.boolean_and_says = global_admit(raw_bytes);

    const AndReduceOutcome named = and_reduce_wire(wire);
    out.named_op_ok          = named.ok;
    out.named_op_says        = named.admitted;
    out.named_op_refusal     = named.refusal;

    // The cross-file check, on the same input.
    const AndEncodingWitness witness = and_encoding_counterexample(wire);
    out.tp_transport_agrees = (witness.sum_encoding_agrees == (out.sum_encoding_says == out.boolean_and_says));

    out.all_three_agree = out.named_op_ok && (out.named_op_says == out.boolean_and_says) &&
                          (out.named_op_says == out.sum_encoding_says);

    if (!out.named_op_ok) {
        out.summary = "the named op REFUSES THE PROTOCOL: the sum encoding and the boolean AND "
                      "disagree on this input, which is exactly why neither may be folded.";
    } else if (out.all_three_agree) {
        out.summary = "all three readings agree on the canonical domain.";
    } else {
        out.summary = "the input is canonical but the readings still disagree: the sum encoding "
                      "is unsound here, and the named op followed the boolean AND.";
    }
    return out;
}

// ---------------------------------------------------------------------------
// 4. THE CUDA HALF -- the local step, placed where the plan says it goes
// ---------------------------------------------------------------------------
//
// WHAT IS HERE: the decode-and-fold as ONE device step, because that is the only shape a captured
// graph can replay -- a host round trip to check a word would break the capture, and the check
// cannot be skipped. WHAT IS NOT HERE: the transport (it is `core/tp_transport.h`'s) and the
// ordering (it is `tp_call_plan()`'s). This section only fills in the `LocalCombine` slot.
//
// THE VIOLATION IS DETECTABLE ON THE DEVICE, and that is a design decision rather than a detail:
// the kernel writes `kAdmitWordViolation` and STOPS -- it does not pick a boolean -- while the
// offending raw word remains readable in the rank's own staging buffer, because the pull wrote it
// there and nothing has overwritten it. So the eager path can name the rank and the value after
// the fact, and the captured path still has no host round trip.

#if defined(__CUDACC__)

// nvcc auto-includes cuda_runtime.h for a .cu translation unit, so the file this one leans on
// (core/tp_transport.h) compiles without naming it. It is named here anyway: this header is meant
// to be includable from any CUDA context, and depending on an implicit include is how a header
// works in one translation unit and not the next.
#include <cuda_runtime.h>

// One block, one word per rank. `mine` is this rank's own verdict (already canonical by type);
// `peer` is the word pulled into this rank's staging, and is the one that needs the decoder.
__global__ void and_reduce_peer_words_kernel(const unsigned int* mine,
                                            const unsigned int* peer,
                                            unsigned int* out) {
    const unsigned int a = *mine;
    const unsigned int b = *peer;
    if (!(a <= 1u) || !(b <= 1u)) {
        *out = kAdmitWordViolation;
        return;
    }
    *out = (a != 0u && b != 0u) ? 1u : 0u;
}

// The 2-rank call, in the donor's four-event order and with this reduction's local step in it.
// Same signature shape as `allreduce_sum_2rank` so a caller can swap the two, and the same
// guarantee: nothing is written by the peer's stream, every transfer is a PULL on this rank's own
// stream, and the second pair is the write-after-read barrier that makes the in-place step and
// back-to-back calls sharing the same buffers safe with no host synchronisation between them.
//
// `self_word` and `peer_storage` are both DEVICE pointers; `out_word` must be device memory the
// caller owns and reuses (a per-call `cudaMalloc` is a capture failure, not a leak).
inline cudaError_t allreduce_and_2rank(const void* self_word, const void* peer_storage,
                                       void* staging, void* out_word, std::size_t bytes,
                                       int rank, cudaStream_t stream,
                                       const PeerEvents& events) noexcept {
    const int peer = 1 - rank;
    cudaError_t err = cudaEventRecord(events.inputs_ready[rank], stream);
    if (err != cudaSuccess) { return err; }
    err = cudaStreamWaitEvent(stream, events.inputs_ready[peer], 0);
    if (err != cudaSuccess) { return err; }
    err = pull_peer(staging, peer_storage, bytes, stream);
    if (err != cudaSuccess) { return err; }
    err = cudaEventRecord(events.pull_done[rank], stream);
    if (err != cudaSuccess) { return err; }
    err = cudaStreamWaitEvent(stream, events.pull_done[peer], 0);
    if (err != cudaSuccess) { return err; }
    // LocalCombine, at the position `tp_call_plan(ReduceOp::AndBool, ...)` gives it: after the
    // peer has finished reading this rank's source, before the caller may reuse the slot.
    and_reduce_peer_words_kernel<<<1, 1, 0, stream>>>(
        static_cast<const unsigned int*>(self_word),
        static_cast<const unsigned int*>(staging),
        static_cast<unsigned int*>(out_word));
    return cudaGetLastError();
}

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// 5. WHAT IS NOT HERE, NAMED
// ---------------------------------------------------------------------------
//
//   * A HOST READ OF THE VERDICT. The collective ends on the device. A caller that wants the
//     answer on the host synchronises, and that synchronisation is the caller's, not the op's.
//   * THE FORCE-A-REFUSE PATH. If a rank's own local predicate cannot run (its page table is
//     unreadable, its tier is offline), what it should publish is a REFUSE, because the AND is
//     the button that keeps a page that somebody might need. That policy belongs to the eviction
//     driver; what belongs here is that the AND of a refuse is a refuse, which the fold gives.
//   * EXECUTION ON REAL HARDWARE. One physical GPU on this box, no MIG, no MPS: the CUDA half
//     above compiles and its 2-rank form runs two virtual ranks on one device, which exercises
//     the ordering and the decode but CANNOT produce a cross-card number. `pending hardware`.

} // namespace ninfer::multi
