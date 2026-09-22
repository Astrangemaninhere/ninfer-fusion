#pragma once

// src/spec/labd_verify_width.h -- the WIDTH DECISION of LABD (lookup-augmented block diffusion).
//
//   Independent feature. OFF BY DEFAULT. Host-only, std-only: no CUDA, no engine header, no
//   Tensor, no workspace, no persistent state, so it compiles and runs under plain `g++` (the
//   test next to it is src/spec/labd_verify_width_test.cpp).
//
// WHAT THIS FILE IS, AND WHERE IT SITS
//   The LABD lines this tree ALREADY carries, each with its own stated contract:
//     * src/spec/lookup_fuse.h            the v1 CHAIN: which tokens the history fills
//                                         (`suffix_best` / `fuse_chain`), CPU-tested by
//                                         src/spec/lookup_fuse_test.cpp over 6 vectors.
//     * include/ninfer/ops/suffix_lookup.h the kernel contract, and the ONE scan bound
//                                         (`suffix_lookup_scan_limit`) the kernel, the host
//                                         reference and both mirrors all ask.
//     * src/targets/qwen3_6/impl/runtime/dflash2_impl.h:468  `width = k + 1` -- one verify column
//                                         per drafted token plus the anchor.
//   What none of the three decides is WHEN to spend the WIDE round. That decision is the
//   controller the research note names, verbatim (_labd_1cat_research.md section 1):
//     "连续强 copy 信号才升 q16；一次 miss 回 q8；B1 最多 coast 3 步；多请求 miss 不允许 sticky；
//      async 调度默认禁用（无法回传可变提案数）"
//   This header IS that decision -- a pure state machine over integers, so a test can drive it
//   with no GPU, no model and no engine, and so landing it CANNOT move the engine binary by one
//   byte (it has no engine-side caller, exactly like src/spec/sum_dir_vector.h before it).
//
// WHY 16 IS NOT AN ARBITRARY NUMBER HERE (three in-tree statements, quoted)
//   * src/targets/qwen3_6/impl/runtime/speculative_target_impl.h:59
//       "kMtpDecodeMaximumDrafts + 1 = 16 in the widest configuration; 64 is slack, not a
//        reachable limit."
//   * src/ops/linear_add/fp8/fp8_linear_add_plan.cpp:26  `kVerifyWidthCeiling = 16`, and the
//     three siblings it names (fp8_gdn_input_plan.cpp:27, fp8_gdn_conv_plan.cpp:90 and :115) all
//     scope their A8/A16 choice to it, "because kMtpDecodeMaximumDrafts + 1 = 16 is the widest a
//     chain-verify round can present".
//   * src/ops/linear_add/fp8/fp8_linear_add_plan.cpp:27  `kWideVerifyWidthCeiling = 64` is the
//     domain the ngram-draft line introduces. LABD's wide rung is the 16, NOT the 64 that the
//     ngram line widens to: LABD's wide rung is bounded by the NEURAL BLOCK it reuses (anchor +
//     7 trained positions), not by the ngram line's separate widening.
//   The rung's COMPOSITION is stated in the research note: the wide block verifies "7 神经位 + 最多
//   8 个来自上下文的 lookup 位", i.e. 15 drafts, so 15 + 1 anchor = 16 columns == kVerifyWidthCeiling.
//   The two rungs are therefore
//       Q8   = anchor + 7 neural                         ==  8 columns
//       Q16  = anchor + 7 neural + 8 context-filled      == 16 columns
//   and the WIDE rung is reachable only when the history can actually fill those 8 slots.
//
// THE ONE PRECONDITION THIS FILE MAKES NON-OPTIONAL
//   A wide round whose lookup half is NOT filled presents verify positions the neural block has no
//   trained head for: the 8 extra columns exist because the context-filled tokens occupy them, and
//   nothing else supplies them. So the lift is refused unless the continuation the lookup already
//   returns covers EVERY one of `kLabdLookupBitsMaximum` filler slots. A short continuation is a
//   NAMED refusal, never a silent narrower wide round -- there is no such round on this ladder.
//
// WHAT THIS HEADER DOES NOT DO, NAMED SO IT IS NOT MISTAKEN FOR DONE
//   * It does not search the history. That is `suffix_lookup` / `lookup_fuse.h`; the caller feeds
//     this state machine the numbers those already return, as plain integers.
//   * It does not verify anything, does not build a graph, and does not choose a verify width for
//     the RUNTIME: the engine's verify width still comes from its own draft plan. This file only
//     says which rung the copy signal has earned by a policy an operator can print.
//   * It is not wired into the engine, and it deliberately has no engine-side caller, so landing it
//     cannot move a binary. The one-line `ninfer_add_test` registration lives in
//     tests/CMakeLists.txt, which is a no-touch surface for this task; the test is therefore run
//     by hand and its exact command line is recorded in the report.

#include <cstdint>
#include <cstddef>

#if defined(LABD_VERIFY_WIDTH_MUTANT_IGNORE_CONTINUATION) ||                                \
    defined(LABD_VERIFY_WIDTH_MUTANT_STICKY_MULTI) ||                                      \
    defined(LABD_VERIFY_WIDTH_MUTANT_ONESHOT)
// The three NEGATIVE-CONTROL builds. Each one inverts exactly ONE judgement this file makes, so
// that the check which guards it can be shown to go RED against a real build rather than argued
// about. A mutant build is never the landed artifact: the test prints which mutant is in effect
// and the control run asserts that none is.
#define LABD_VERIFY_WIDTH_IS_MUTANT_BUILD 1
#else
#define LABD_VERIFY_WIDTH_IS_MUTANT_BUILD 0
#endif

namespace ninfer::spec::labd {

// ---------------------------------------------------------------------------
// the ladder, as WIDTHS (columns presented to the target in one verify round)
// ---------------------------------------------------------------------------

// The narrow rung: the trained block. anchor + 7 neural drafts = the 8 columns
// src/targets/qwen3_6/impl/runtime/dflash2_impl.h:468 computes as `k + 1` with k = 7.
inline constexpr std::int32_t kLabdVerifyWidthQ8 = 8;

// The wide rung: the narrow block plus the context-filled half. Equals the `kVerifyWidthCeiling`
// the fp8 A8/A16 sites already protect (16), so choosing it enters no routing domain this tree has
// not already measured.
inline constexpr std::int32_t kLabdVerifyWidthQ16 = 16;

// The filler slots that exist ONLY at the wide rung -- the "最多 8 个来自上下文的 lookup 位".
inline constexpr std::int32_t kLabdLookupBitsMaximum = 8;

// The neural drafts of the trained block: "anchor + 7 训练位".
inline constexpr std::int32_t kLabdNeuralDrafts = 7;

static_assert(kLabdVerifyWidthQ8 - 1 == kLabdNeuralDrafts,
              "the narrow rung is the trained block: one anchor plus 7 neural drafts");
static_assert(kLabdVerifyWidthQ16 - kLabdVerifyWidthQ8 == kLabdLookupBitsMaximum,
              "the wide rung's extra columns are exactly the context-filled slots");
static_assert(kLabdVerifyWidthQ16 == 16,
              "the wide rung is the width kVerifyWidthCeiling already scopes (16), not the ngram "
              "line's separate widening to 64");
static_assert(kLabdNeuralDrafts + kLabdLookupBitsMaximum + 1 == kLabdVerifyWidthQ16,
              "anchor + neural drafts + lookup bits == wide width");

enum class LabdVerifyWidth : std::int32_t {
    Q8  = kLabdVerifyWidthQ8,
    Q16 = kLabdVerifyWidthQ16,
};

[[nodiscard]] constexpr const char* labd_verify_width_name(LabdVerifyWidth width) noexcept {
    return width == LabdVerifyWidth::Q16 ? "q16" : "q8";
}

// The drafts a rung proposes: width - 1 (the anchor is not a draft).
[[nodiscard]] constexpr std::int32_t labd_verify_width_drafts(LabdVerifyWidth width) noexcept {
    return static_cast<std::int32_t>(width) - 1;
}

// ---------------------------------------------------------------------------
// what the controller is told about one round (plain integers: no SuffixHit, no Tensor)
// ---------------------------------------------------------------------------

struct LabdCopyObservation {
    // How many requests are being decoded concurrently. The coast allowance is a B1 privilege:
    // "多请求 miss 不允许 sticky".
    std::int32_t requests = 1;
    // Identity of the request this round belongs to. A lift is earned by ONE request over ITS
    // consecutive rounds; a different request may not inherit the latch.
    std::uint64_t request_id = 0;
    // This round's lookup outcome, as `suffix_lookup` reports it: `best_len[b]` and the count of
    // usable entries in `continuation[b]`.
    std::int32_t match_length           = 0;
    std::int32_t query_len              = 0;
    std::int32_t continuation_available = 0;
    // Whether the previous round's verify accepted EVERY column it was given. Default true so a
    // caller that only knows the lookup half is still correct: it can then never lift on a
    // round the copy signal alone did not earn.
    bool previous_round_fully_accepted = true;
};

// A full-window match: the whole query tail was found earlier in this request's own history. A
// partial match may still propose ONE token (`fuse_chain`'s partial branch), but it is not a
// strength signal and it never lifts the ladder.
[[nodiscard]] constexpr bool labd_copy_is_full_window(const LabdCopyObservation& o) noexcept {
    return o.query_len > 0 && o.match_length >= o.query_len;
}

// The context half of the wide rung is FILLABLE: the lookup already returns at least every filler
// slot the wide rung needs.
[[nodiscard]] constexpr bool labd_continuation_fills_wide_rung(
    const LabdCopyObservation& o) noexcept {
    return o.continuation_available >= kLabdLookupBitsMaximum;
}

// Is a lift admissible for this observation? THE single definition of the precondition, so the
// refusal text and the state machine cannot disagree about it.
[[nodiscard]] constexpr bool labd_verify_width_lift_admissible(
    const LabdCopyObservation& o) noexcept {
#if defined(LABD_VERIFY_WIDTH_MUTANT_IGNORE_CONTINUATION)
    // NEGATIVE-CONTROL BUILD: the continuation clause is dropped. A build with this switch must
    // make `lift refused when the continuation is one short` go red. Never land this.
    return labd_copy_is_full_window(o);
#else
    return labd_copy_is_full_window(o) && labd_continuation_fills_wide_rung(o);
#endif
}

// ---------------------------------------------------------------------------
// the policy, and the settings that are refused BY NAME
// ---------------------------------------------------------------------------

struct LabdAdaptivePolicy {
    // Consecutive admissible rounds required before the wide rung is entered. 1 would lift on a
    // single sample; the source asks for consecutive strength, so 1 is refused below.
    std::int32_t strong_run_required = 2;
    // B1 coast: how many consecutive misses the wide rung may survive at ONE request. The source
    // says "最多 coast 3 步".
    std::int32_t b1_coast_steps = 3;
    // Async scheduling cannot carry a variable proposal count back to the scheduler, so it stays
    // off. Kept as a field so a caller that tries to turn it on gets a NAMED refusal instead of a
    // silent behaviour difference.
    bool async_scheduling = false;
};

inline constexpr std::int32_t kLabdStrongRunDefault = 2;
inline constexpr std::int32_t kLabdB1CoastStepsDefault = 3;

// nullptr == accepted. Every string returned here names the clause, never a generic error.
[[nodiscard]] inline const char* labd_adaptive_policy_refusal(
    const LabdAdaptivePolicy& policy) noexcept {
#if defined(LABD_VERIFY_WIDTH_MUTANT_ONESHOT)
    // NEGATIVE-CONTROL BUILD: the consecutive-strength clause is dropped, so a single sample is
    // allowed to lift. This mutant is self-consistent -- it relaxes the controller below by the
    // same amount it relaxes this refusal. A build with this switch must make BOTH `one admissible
    // round is not enough` and `a one-shot policy is refused by name` go red. Never land this.
    constexpr std::int32_t kRequiredFloor = 1;
#else
    constexpr std::int32_t kRequiredFloor = kLabdStrongRunDefault;
#endif
    if (policy.strong_run_required < kRequiredFloor) {
        return "labd adaptive policy refused: strong_run_required < 2 would enter the wide rung on "
               "a single sample, and the rung exists only for CONSECUTIVE strength. There is no "
               "one-shot lift on this ladder.";
    }
    if (policy.b1_coast_steps < 0) {
        return "labd adaptive policy refused: b1_coast_steps < 0 has no meaning; use 0 to forbid "
               "coasting at one request.";
    }
    if (policy.b1_coast_steps > kLabdB1CoastStepsDefault) {
        return "labd adaptive policy refused: b1_coast_steps above the defaulted 3 is outside the "
               "range the research note states ('B1 最多 coast 3 步').";
    }
    if (policy.async_scheduling) {
        return "labd adaptive policy refused: async scheduling is inadmissible for LABD, because "
               "the proposal count is VARIABLE (the wide rung proposes 15 drafts and the narrow "
               "rung 7), and the async path cannot carry a variable proposal count back. The "
               "research note states the same: 'async 调度默认禁用'.";
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// the verdict of one decision, printable, so a test asserts the REASON and not only the rung
// ---------------------------------------------------------------------------

enum class LabdWidthVerdict : std::uint8_t {
    HeldNarrow,                // stayed at q8; nothing earned a lift this round
    HeldNarrowNoSignal,        // stayed at q8; the copy signal was partial or absent
    HeldNarrowRunIncomplete,   // stayed at q8; admissible, but the consecutive run is short
    LiftedWide,                // q8 -> q16, the consecutive run reached the requirement
    StayedWide,                // already q16, this round earned it again
    StayedWideCoasting,        // q16 surviving a miss inside the B1 coast budget
    DroppedMiss,               // q16 -> q8, a miss at more than one request (not sticky)
    DroppedCoastExhausted,     // q16 -> q8, the B1 coast budget ran out
    DroppedRequestChanged,     // q16 -> q8, a different request cannot inherit the latch
};

[[nodiscard]] constexpr const char* labd_width_verdict_name(LabdWidthVerdict v) noexcept {
    switch (v) {
    case LabdWidthVerdict::HeldNarrow:              return "held-narrow";
    case LabdWidthVerdict::HeldNarrowNoSignal:      return "held-narrow-no-signal";
    case LabdWidthVerdict::HeldNarrowRunIncomplete: return "held-narrow-run-incomplete";
    case LabdWidthVerdict::LiftedWide:              return "lifted-wide";
    case LabdWidthVerdict::StayedWide:              return "stayed-wide";
    case LabdWidthVerdict::StayedWideCoasting:      return "stayed-wide-coasting";
    case LabdWidthVerdict::DroppedMiss:             return "dropped-miss";
    case LabdWidthVerdict::DroppedCoastExhausted:   return "dropped-coast-exhausted";
    case LabdWidthVerdict::DroppedRequestChanged:   return "dropped-request-changed";
    }
    return "unknown";
}

// A report, so every clause is MEASURED rather than argued about, and so the counters can be
// printed by whoever runs the test.
struct LabdAdaptiveReport {
    std::int32_t rounds                        = 0;
    std::int32_t lifts                         = 0;
    std::int32_t drops_by_miss                 = 0; // a miss at more than one request
    std::int32_t drops_by_coast_exhausted      = 0;
    std::int32_t drops_by_request_changed      = 0;
    std::int32_t refused_short_continuation    = 0; // admissible match, unfillable wide rung
    std::int32_t refused_partial_match         = 0;
    std::int32_t no_signal_rounds              = 0; // no match at all: neither of the two above
    std::int32_t b1_coast_used                 = 0; // misses absorbed by the B1 budget
    std::int32_t wide_rounds                   = 0;
};

// Every round that did not earn the wide rung is accounted for by EXACTLY ONE of these three
// counters, so `refused_short_continuation + refused_partial_match + no_signal_rounds` equals the
// number of non-earning rounds. The test pins that identity rather than trusting it.
[[nodiscard]] constexpr std::int32_t labd_adaptive_refusal_total(
    const LabdAdaptiveReport& report) noexcept {
    return report.refused_short_continuation + report.refused_partial_match +
           report.no_signal_rounds;
}

// ---------------------------------------------------------------------------
// the controller
// ---------------------------------------------------------------------------

class LabdAdaptiveController {
public:
    explicit LabdAdaptiveController(LabdAdaptivePolicy policy = {}) noexcept
        : policy_(policy), refusal_(labd_adaptive_policy_refusal(policy)) {}

    // The policy this controller was built with, if it was acceptable. Non-null == the controller
    // is INERT: `observe` returns the narrow rung and changes nothing, so an inadmissible policy
    // can never widen a verify round by accident.
    [[nodiscard]] const char* policy_refusal() const noexcept { return refusal_; }

    [[nodiscard]] const LabdAdaptivePolicy& policy() const noexcept { return policy_; }
    [[nodiscard]] LabdVerifyWidth width() const noexcept { return width_; }
    [[nodiscard]] LabdWidthVerdict last_verdict() const noexcept { return last_verdict_; }
    [[nodiscard]] const char* last_verdict_name() const noexcept {
        return labd_width_verdict_name(last_verdict_);
    }
    [[nodiscard]] const LabdAdaptiveReport& report() const noexcept { return report_; }

    // One call per decode round. Returns the rung the NEXT verify round should use.
    LabdVerifyWidth observe(const LabdCopyObservation& o) noexcept {
        ++report_.rounds;
        if (refusal_ != nullptr) { return held(LabdWidthVerdict::HeldNarrowNoSignal); }

        // Non-sticky across requests: a different request starts from the narrow rung and from a
        // clean run count, whatever the previous request had earned.
        if (have_request_ && o.request_id != request_id_) {
            strong_run_ = 0;
            coast_left_ = policy_.b1_coast_steps;
            have_request_ = true;
            request_id_    = o.request_id;
            if (width_ != LabdVerifyWidth::Q8) {
                width_ = LabdVerifyWidth::Q8;
                ++report_.drops_by_request_changed;
                return held(LabdWidthVerdict::DroppedRequestChanged);
            }
            return held(LabdWidthVerdict::HeldNarrowNoSignal);
        }
        have_request_ = true;
        request_id_    = o.request_id;

        const bool earned = labd_verify_width_lift_admissible(o);
        if (!earned) {
            if (labd_copy_is_full_window(o)) {
                ++report_.refused_short_continuation;
            } else if (o.match_length > 0) {
                ++report_.refused_partial_match;
            } else {
                ++report_.no_signal_rounds;
            }
        }
        strong_run_ = earned ? strong_run_ + 1 : 0;

        if (width_ == LabdVerifyWidth::Q8) {
            if (!earned) { return held(LabdWidthVerdict::HeldNarrowNoSignal); }
#if defined(LABD_VERIFY_WIDTH_MUTANT_ONESHOT)
            // NEGATIVE-CONTROL BUILD: one admissible round is treated as a run. See the refusal
            // above; the two clauses are relaxed together on purpose. Never land this.
            constexpr std::int32_t kEffectiveRequired = 1;
#else
            const std::int32_t kEffectiveRequired = policy_.strong_run_required;
#endif
            if (strong_run_ < kEffectiveRequired) {
                return held(LabdWidthVerdict::HeldNarrowRunIncomplete);
            }
            width_     = LabdVerifyWidth::Q16;
            coast_left_ = policy_.b1_coast_steps;
            ++report_.lifts;
            return record(LabdWidthVerdict::LiftedWide);
        }

        // Already wide. A "miss" is a round that did not earn the wide rung, or a round whose
        // verify did not accept every column it was given -- the second one is what the wide rung
        // costs when the copy signal was noise, so it counts the same way.
        ++report_.wide_rounds;
        const bool paid = earned && o.previous_round_fully_accepted;
        if (paid) {
            coast_left_ = policy_.b1_coast_steps;
            return record(LabdWidthVerdict::StayedWide);
        }
        // A miss. The coast allowance is a B1 privilege: at more than one request the source's
        // clause is "多请求 miss 不允许 sticky", so the drop is immediate.
        const std::int32_t budget =
            o.requests > 1
#if defined(LABD_VERIFY_WIDTH_MUTANT_STICKY_MULTI)
                // NEGATIVE-CONTROL BUILD: a multi-request miss keeps the B1 coast budget, i.e. it
                // IS sticky. A build with this switch must make `multi-request miss drops at once`
                // go red. Never land this.
                ? policy_.b1_coast_steps
#else
                ? 0
#endif
                : policy_.b1_coast_steps;
        if (budget > 0 && coast_left_ > 0) {
            --coast_left_;
            ++report_.b1_coast_used;
            return record(LabdWidthVerdict::StayedWideCoasting);
        }
        width_      = LabdVerifyWidth::Q8;
        strong_run_ = 0;
        coast_left_ = policy_.b1_coast_steps;
        if (budget > 0) {
            ++report_.drops_by_coast_exhausted;
            return record(LabdWidthVerdict::DroppedCoastExhausted);
        }
        ++report_.drops_by_miss;
        return record(LabdWidthVerdict::DroppedMiss);
    }

    // Bring the controller back to its constructed state without rebuilding the policy, so a test
    // can run several arms in one process and compare them.
    void reset() noexcept {
        width_         = LabdVerifyWidth::Q8;
        strong_run_    = 0;
        coast_left_    = policy_.b1_coast_steps;
        have_request_  = false;
        request_id_    = 0;
        last_verdict_  = LabdWidthVerdict::HeldNarrow;
        report_        = LabdAdaptiveReport{};
    }

private:
    [[nodiscard]] LabdVerifyWidth held(LabdWidthVerdict verdict) noexcept {
        return record(verdict);
    }
    [[nodiscard]] LabdVerifyWidth record(LabdWidthVerdict verdict) noexcept {
        last_verdict_ = verdict;
        return width_;
    }

    LabdAdaptivePolicy policy_{};
    const char* refusal_ = nullptr;
    LabdVerifyWidth width_ = LabdVerifyWidth::Q8;
    std::int32_t strong_run_ = 0;
    std::int32_t coast_left_ = kLabdB1CoastStepsDefault;
    bool have_request_       = false;
    std::uint64_t request_id_ = 0;
    LabdWidthVerdict last_verdict_ = LabdWidthVerdict::HeldNarrow;
    LabdAdaptiveReport report_{};
};

// The async-scheduling clause as a standalone predicate, so a caller can ask the question without
// constructing a controller.
[[nodiscard]] constexpr bool labd_async_scheduling_admissible() noexcept {
    return false; // the proposal count is variable across the two rungs
}

} // namespace ninfer::spec::labd
