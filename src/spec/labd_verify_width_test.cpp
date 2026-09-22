// labd_verify_width_test.cpp -- the LABD width-decision test.
//
// Runs under plain g++ with NO CUDA, no engine, no artifact, no fixture: the subject is a state
// machine over integers (src/spec/labd_verify_width.h). Sibling of src/spec/lookup_fuse_test.cpp,
// which tests the CHAIN half this file's controller decides the width of.
//
// EXIT CODE: 0 when every check passes, 1 otherwise. Every check is counted and named, and the
// count is printed, so a partial run cannot read as a full one.
//
// NEGATIVE-CONTROL BUILDS: the header can be compiled with one of three switches that each invert
// exactly ONE judgement. This SAME source is then expected to go RED, and it cannot go green by
// accident: the build identity is printed, and the section each mutant targets is named in the
// header. Measured reds are recorded in the report; the numbers here are the live ones.
#include "spec/labd_verify_width.h"

#include <cstdio>
#include <cstdint>
#include <string_view>
#include <vector>

using namespace ninfer::spec::labd;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const char* label) {
    ++g_checks;
    if (!ok) { ++g_failures; }
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", label);
}

// One admissible round: full window, wide rung fillable, previous verify accepted everything.
LabdCopyObservation strong(std::uint64_t request_id = 1, std::int32_t requests = 1,
                           std::int32_t query_len = 64) {
    LabdCopyObservation o;
    o.requests                      = requests;
    o.request_id                    = request_id;
    o.match_length                  = query_len;
    o.query_len                     = query_len;
    o.continuation_available        = kLabdLookupBitsMaximum;
    o.previous_round_fully_accepted = true;
    return o;
}

// A miss at the wide rung with the copy signal still strong: the round was not fully accepted,
// which is what a wide round costs when the copy signal was noise.
LabdCopyObservation wide_round_not_accepted(std::uint64_t request_id = 1,
                                            std::int32_t requests = 1) {
    LabdCopyObservation o = strong(request_id, requests);
    o.previous_round_fully_accepted = false;
    return o;
}

// A partial match: proposes one token via `fuse_chain`'s partial branch, but is not a strength
// signal and never lifts the ladder.
LabdCopyObservation partial(std::uint64_t request_id = 1, std::int32_t requests = 1) {
    LabdCopyObservation o;
    o.requests                      = requests;
    o.request_id                    = request_id;
    o.match_length                  = 3;
    o.query_len                     = 64;
    o.continuation_available        = 1;
    o.previous_round_fully_accepted = true;
    return o;
}

// The six corpora of src/spec/lookup_fuse_test.cpp, verbatim: (ids, query_start, query_len, k).
// They are repeated here rather than included, because lookup_fuse.h pulls in
// ninfer/ops/suffix_lookup.h and therefore <cuda_runtime.h>, which this host-only test does not
// have. The repetition is checked against that file in the report, not asserted here.
struct FuseCorpus {
    std::vector<std::int32_t> ids;
    std::int32_t query_start;
    std::int32_t query_len;
    std::int32_t k;
    std::int32_t expected_filled; // what fuse_chain returns on that corpus
};

std::vector<FuseCorpus> fuse_corpora() {
    return {
        {{1, 2, 3, 9, 9, 1, 2, 3, 7, 8}, 5, 3, 3, 3},  // full-window chain
        {{1, 2, 6, 8, 0, 0, 3, 5, 6, 8}, 6, 4, 3, 1},  // partial probe, one token
        {{1, 5, 2, 6, 3, 7, 4, 8, 9, 0}, 6, 4, 3, 0},  // no hit, fallback
        {{7, 8, 9, 1, 2, 3, 1, 2, 3}, 6, 3, 4, 0},     // self-match rejected by the scan bound
        {{5, 5, 5, 5, 2, 0, 1, 5, 5, 0}, 6, 3, 2, 1},  // tie -> latest, partial
        {{5, 5, 5, 5, 2, 0, 5, 5, 5, 0}, 6, 3, 2, 2},  // tie -> latest, full window
    };
}

} // namespace

int main() {
    std::printf("== labd_verify_width test ==\n");
    std::printf("mutant-build=%d", static_cast<int>(LABD_VERIFY_WIDTH_IS_MUTANT_BUILD));
#if defined(LABD_VERIFY_WIDTH_MUTANT_IGNORE_CONTINUATION)
    std::printf(" (IGNORE_CONTINUATION)");
#endif
#if defined(LABD_VERIFY_WIDTH_MUTANT_STICKY_MULTI)
    std::printf(" (STICKY_MULTI)");
#endif
#if defined(LABD_VERIFY_WIDTH_MUTANT_ONESHOT)
    std::printf(" (ONESHOT)");
#endif
    std::printf("\n");

    // ---- A. geometry -------------------------------------------------------
    std::printf("-- A geometry --\n");
    check(kLabdVerifyWidthQ8 == 8, "A1 the narrow rung is the trained block, 8 columns");
    check(kLabdVerifyWidthQ16 == 16, "A2 the wide rung is 16, the width kVerifyWidthCeiling scopes");
    check(kLabdVerifyWidthQ16 - kLabdVerifyWidthQ8 == kLabdLookupBitsMaximum,
          "A3 the wide rung's extra columns ARE the 8 context-filled slots");
    check(kLabdNeuralDrafts == 7 && labd_verify_width_drafts(LabdVerifyWidth::Q8) == 7,
          "A4 the narrow rung proposes anchor + 7 neural drafts");
    check(labd_verify_width_drafts(LabdVerifyWidth::Q16) == 15,
          "A5 the wide rung proposes 15 drafts (7 neural + 8 lookup)");

    // ---- B. policy refusals by name ----------------------------------------
    std::printf("-- B policy refusals --\n");
    {
        LabdAdaptivePolicy one_shot;
        one_shot.strong_run_required = 1;
        const char* why = labd_adaptive_policy_refusal(one_shot);
        check(why != nullptr, "B1 a one-shot policy is refused by name");
        check(why != nullptr && std::string_view(why).find("one-shot") != std::string_view::npos,
              "B2 the refusal names the clause (one-shot), not a generic error");
    }
    {
        LabdAdaptivePolicy async_on;
        async_on.async_scheduling = true;
        const char* why = labd_adaptive_policy_refusal(async_on);
        check(why != nullptr, "B3 async scheduling is refused by name");
        check(!labd_async_scheduling_admissible(), "B4 the async predicate itself is false");
    }
    {
        LabdAdaptivePolicy too_much_coast;
        too_much_coast.b1_coast_steps = 4;
        check(labd_adaptive_policy_refusal(too_much_coast) != nullptr,
              "B5 a B1 coast budget above the stated 3 is refused by name");
    }
    {
        LabdAdaptivePolicy clean;
        check(labd_adaptive_policy_refusal(clean) == nullptr, "B6 the default policy is accepted");
    }
    {
        LabdAdaptiveController inert(LabdAdaptivePolicy{0, 3, false});
        bool stayed = true;
        for (int i = 0; i < 8; ++i) {
            if (inert.observe(strong()) != LabdVerifyWidth::Q8) { stayed = false; }
        }
        check(inert.policy_refusal() != nullptr && stayed,
              "B7 an inadmissible policy makes the controller INERT, never wide");
    }

    // ---- C. no signal, no lift --------------------------------------------
    std::printf("-- C no-signal arm --\n");
    {
        LabdAdaptiveController c;
        bool narrow = true;
        for (int i = 0; i < 32; ++i) {
            if (c.observe(partial()) != LabdVerifyWidth::Q8) { narrow = false; }
        }
        check(narrow, "C1 32 partial matches never lift the ladder");
        check(c.width() == LabdVerifyWidth::Q8, "C2 the rung is still q8");
        check(c.report().lifts == 0, "C3 the lift counter is 0");
        check(c.report().refused_partial_match == 32,
              "C4 all 32 rounds were counted as partial-match refusals");
    }

    // ---- D. one admissible round is not enough -----------------------------
    std::printf("-- D consecutive run --\n");
    {
        LabdAdaptiveController c;
        check(c.observe(strong()) == LabdVerifyWidth::Q8, "D1 the first admissible round stays q8");
        check(c.last_verdict() == LabdWidthVerdict::HeldNarrowRunIncomplete,
              "D2 and says so: held-narrow-run-incomplete");
    }

    // ---- E. the lift, on the second consecutive admissible round ----------
    std::printf("-- E lift --\n");
    {
        LabdAdaptiveController c;
        c.observe(strong());
        check(c.observe(strong()) == LabdVerifyWidth::Q16,
              "E1 the second consecutive admissible round lifts to q16");
        check(c.last_verdict() == LabdWidthVerdict::LiftedWide, "E2 and says lifted-wide");
        check(c.report().lifts == 1, "E3 the lift counter is 1");
        check(c.observe(strong()) == LabdVerifyWidth::Q16 && c.last_verdict() ==
                  LabdWidthVerdict::StayedWide,
              "E4 the next strong round stays wide (stayed-wide)");
        check(c.report().wide_rounds == 1, "E5 exactly one wide round was counted so far");
    }

    // ---- F. B1 coast ------------------------------------------------------
    std::printf("-- F B1 coast --\n");
    {
        LabdAdaptiveController c; // requests == 1
        c.observe(strong());
        c.observe(strong());
        check(c.width() == LabdVerifyWidth::Q16, "F0 precondition: the controller is wide");
        const std::int32_t before = c.report().b1_coast_used;
        check(c.observe(wide_round_not_accepted()) == LabdVerifyWidth::Q16 &&
                  c.last_verdict() == LabdWidthVerdict::StayedWideCoasting,
              "F1 miss 1 of 3 is absorbed by the coast budget (stayed-wide-coasting)");
        check(c.observe(wide_round_not_accepted()) == LabdVerifyWidth::Q16,
              "F2 miss 2 of 3 is absorbed");
        check(c.observe(wide_round_not_accepted()) == LabdVerifyWidth::Q16,
              "F3 miss 3 of 3 is absorbed");
        check(c.report().b1_coast_used == before + 3, "F4 exactly 3 misses were coasted");
        check(c.observe(wide_round_not_accepted()) == LabdVerifyWidth::Q8,
              "F5 the FOURTH consecutive miss drops to q8");
        check(c.last_verdict() == LabdWidthVerdict::DroppedCoastExhausted,
              "F6 and says dropped-coast-exhausted, not dropped-miss");
        check(c.report().drops_by_coast_exhausted == 1 && c.report().drops_by_miss == 0,
              "F7 the two drop causes are counted apart");
    }
    {
        // A miss resets nothing but the coast: ONE strong round after the drop must not re-lift.
        LabdAdaptiveController c;
        c.observe(strong());
        c.observe(strong());
        for (int i = 0; i < 4; ++i) { c.observe(wide_round_not_accepted()); }
        check(c.width() == LabdVerifyWidth::Q8, "F8 after the drop the rung is q8");
        check(c.observe(strong()) == LabdVerifyWidth::Q8,
              "F9 one strong round after a drop does not re-lift (the run restarts)");
    }

    // ---- G. a multi-request miss is NOT sticky ----------------------------
    std::printf("-- G multi-request miss --\n");
    {
        LabdAdaptiveController c;
        c.observe(strong(7, 2));
        c.observe(strong(7, 2));
        check(c.width() == LabdVerifyWidth::Q16, "G1 precondition: the controller is wide");
        check(c.observe(wide_round_not_accepted(7, 2)) == LabdVerifyWidth::Q8,
              "G2 a miss at 2 requests drops at ONCE (no coast)");
        check(c.last_verdict() == LabdWidthVerdict::DroppedMiss,
              "G3 and says dropped-miss, not dropped-coast-exhausted");
        check(c.report().drops_by_miss == 1 && c.report().drops_by_coast_exhausted == 0,
              "G4 it is counted as a miss drop");
        check(c.report().b1_coast_used == 0, "G5 the B1 coast budget was never touched");
    }

    // ---- H. non-sticky across a request change ----------------------------
    std::printf("-- H request change --\n");
    {
        LabdAdaptiveController c;
        c.observe(strong(1));
        c.observe(strong(1));
        check(c.width() == LabdVerifyWidth::Q16, "H1 precondition: request 1 is wide");
        check(c.observe(strong(2)) == LabdVerifyWidth::Q8,
              "H2 a different request starts narrow, immediately");
        check(c.last_verdict() == LabdWidthVerdict::DroppedRequestChanged,
              "H3 and says dropped-request-changed");
        check(c.report().drops_by_request_changed == 1, "H4 the drop is counted");
        check(c.observe(strong(2)) == LabdVerifyWidth::Q8,
              "H5 request 2's own run starts from scratch, so one round does not lift it");
        check(c.observe(strong(2)) == LabdVerifyWidth::Q16,
              "H6 request 2 earns the wide rung on its OWN second round");
    }

    // ---- I. the continuation precondition, one short ----------------------
    std::printf("-- I continuation precondition --\n");
    {
        LabdAdaptiveController c;
        LabdCopyObservation o = strong();
        o.continuation_available = kLabdLookupBitsMaximum - 1; // 7, one short of the 8 slots
        bool narrow = true;
        for (int i = 0; i < 6; ++i) {
            if (c.observe(o) != LabdVerifyWidth::Q8) { narrow = false; }
        }
        check(narrow, "I1 a full-window match with a ONE-SHORT continuation never lifts");
        check(c.report().lifts == 0, "I2 the lift counter is 0");
        check(c.report().refused_short_continuation == 6,
              "I3 all 6 rounds were counted as short-continuation refusals");
        check(c.last_verdict() == LabdWidthVerdict::HeldNarrowNoSignal,
              "I4 the verdict is held-narrow-no-signal, not a silent narrower wide round");
        check(!labd_verify_width_lift_admissible(o),
              "I5 the admissible predicate itself refuses it");
        LabdCopyObservation enough = strong();
        check(labd_verify_width_lift_admissible(enough),
              "I6 the same round with all 8 slots DOES satisfy the predicate");
    }

    // ---- J. the control arm: the lookup_fuse corpora earn no lift ---------
    std::printf("-- J control arm (existing path unchanged) --\n");
    {
        LabdAdaptiveController c;
        std::int32_t seen = 0;
        std::int32_t full_window = 0;
        for (const FuseCorpus& f : fuse_corpora()) {
            // Feed the controller what the chain half already reports for this corpus: a chain of
            // `expected_filled` tokens, from a window of length `query_len`.
            LabdCopyObservation o;
            o.requests                      = 1;
            o.request_id                    = 1;
            o.match_length                  = f.expected_filled >= f.query_len ? f.query_len
                                                                               : f.expected_filled;
            o.query_len                     = f.query_len;
            o.continuation_available        = f.expected_filled;
            o.previous_round_fully_accepted = true;
            ++seen;
            if (labd_copy_is_full_window(o)) { ++full_window; }
            const LabdVerifyWidth w = c.observe(o);
            check(w == LabdVerifyWidth::Q8 && c.width() == LabdVerifyWidth::Q8,
                  "J: a fuse_chain corpus leaves the width at q8");
        }
        check(seen == 6, "J1 all six corpora were driven through the controller");
        // Corpus 1 of the six ("full-window chain", query_len 3, filled 3) IS a full-window match.
        // That is not a defect: it is the case the ladder is FOR, and it still does not lift,
        // because its chain of 3 cannot fill the wide rung's 8 slots. The count is asserted, not
        // waved past, so a corpus that silently changed shape would move it.
        check(full_window == 1, "J2 exactly one of the six is a full-window match (corpus 1)");
        check(c.report().lifts == 0,
              "J3 the landed default lifts on none of the six existing corpora");
        check(c.report().refused_short_continuation == 1,
              "J4 corpus 1 is a short-continuation refusal, and it is counted");
        check(c.report().refused_partial_match == 3,
              "J5 corpora 2/5/6 are partial-match refusals (chain of 1 or 2 from a 4- or 3-window)");
        check(c.report().no_signal_rounds == 2, "J6 corpora 3/4 have no match at all");
        check(labd_adaptive_refusal_total(c.report()) == 6,
              "J7 every one of the six rounds is accounted for by exactly one refusal cause");
    }

    // ---- K. counters agree with the verdict history ----------------------
    std::printf("-- K report consistency --\n");
    {
        LabdAdaptiveController c;
        c.observe(strong());                                    // run 1
        c.observe(strong());                                    // lift
        c.observe(wide_round_not_accepted());                   // coast 1
        c.observe(wide_round_not_accepted());                   // coast 2
        c.observe(wide_round_not_accepted());                   // coast 3
        c.observe(wide_round_not_accepted());                   // drop
        const LabdAdaptiveReport& r = c.report();
        check(r.rounds == 6, "K1 six rounds were counted");
        check(r.lifts == 1, "K2 one lift");
        check(r.b1_coast_used == 3, "K3 three coasted misses");
        check(r.drops_by_coast_exhausted == 1, "K4 one drop by exhaustion");
        check(r.drops_by_miss == 0 && r.drops_by_request_changed == 0,
              "K5 neither other drop cause fired");
        check(c.width() == LabdVerifyWidth::Q8, "K6 the end state is q8");
        check(labd_adaptive_refusal_total(r) == 0,
              "K7 all six rounds earned the copy signal, so no refusal cause fired either");
    }

    std::printf("== checks=%d failures=%d mutant=%d\n", g_checks, g_failures,
                static_cast<int>(LABD_VERIFY_WIDTH_IS_MUTANT_BUILD));
    std::printf(g_failures == 0 ? "VERDICT: PASS\n" : "VERDICT: FAIL\n");
    return g_failures == 0 ? 0 : 1;
}
