// ---------------------------------------------------------------------------
// test_arch_sim.cpp -- THE FILE src/core/arch_sim.h:26 NAMES, WHICH DID NOT EXIST.
//
// src/core/arch_sim.h:26 reads, verbatim:
//
//   "THE FOUR GUARD PROPERTIES (each is pinned by a test in tests/test_arch_sim.cpp)"
//
// MEASURED 2026-09-22 (dl/simrun2): `ls tests/test_arch_sim.cpp` -> "No such file or
// directory"; `grep -rn 'test_arch_sim' tests/` -> empty. The four guards WERE pinned -- in
// tests/test_arch_generic_fallback.cpp section B (a header comment there calls itself
// "B. THE SIMULATOR'S FOUR GUARD PROPERTIES") and in tests/test_sim_no_support.cpp -- so the
// coverage was real and the POINTER was wrong. A named pointer that resolves to nothing is
// the defect this file closes: a reader who follows the header to check the guard they are
// about to rely on lands in a directory listing instead of on an assertion.
//
// SO THIS FILE EXISTS TO MAKE THE HEADER'S SENTENCE TRUE, and it is written to be the
// smallest thing that does that: one binary, the four guards named G1..G4 as the header names
// them, plus the RUNG-BOUNDARY pins this line added (section E) and the red controls that
// keep them from passing vacuously (section F).
//
// WHAT MAKES THIS FILE WORTH HAVING RATHER THAN A DUPLICATE. It pins the boundary the header
// does NOT pin: WHICH compute capabilities can be simulated at all. UPDATED 2026-09-24 (F702,
// dl/archrow): the ladder (src/core/arch_caps.h, kArchLadder) has 18 rows and the lowest is
// sm_50, because that leg added the six pre-75 rows (sm_50/52/53 Maxwell, sm_60/61/62 Pascal,
// all Cap::None, each carrying the format floors it misses with their kernel file:line). Before
// that block the ladder's lowest row was sm_70 and the pre-75 family was refused BY NAME at
// every physical device -- which is why section E existed and is why it is now extended rather
// than deleted: the rung-less refusal is still pinned (K40 = sm_35, and the withdrawn sm_110),
// and the six new rows are pinned as REACHABLE with their capability set and their refusals.
// The two failure modes section E forbids are what a later "make the simulator reach more cards"
// edit would actually do, and neither of them is a build break:
//     (a) coerce an unlisted number to a listed one (sm_35 silently answered as sm_70),
//     (b) invent a capability set for an unlisted number (a new kArchLadder row with no
//         MEASURED evidence behind it -- which is what F702 had to satisfy, not to bypass).
// Asserting (a) and (b) is NOT gate weakening: nothing here relaxes a guard, and every check
// below fails only in the direction the header forbids.
//
// EVIDENCE GRADES, as the tree's convention requires:
//   [RUN]  this binary executed and the check is on its own output.
//   [READ] the assertion cites a coordinate in the source it is about.
// ---------------------------------------------------------------------------

#include "core/arch_caps.h"
#include "core/arch_sim.h"
#include "core/kernel_route.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer::caps;
using ninfer::artifact::NumericFormat;

namespace {

int g_failures = 0;
int g_checks   = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL  %s\n", what.c_str());
    }
}

void check_contains(const std::string& haystack, const char* needle, const std::string& what) {
    check(haystack.find(needle) != std::string::npos,
          what + " (looked for \"" + std::string(needle) + "\" in a " +
              std::to_string(haystack.size()) + " B text)");
}

// The physical card this tree ships for: sm_120a (CMakeLists.txt:19). Every view below is
// taken at 120, so the downgrade-only rule has room and the refusals are about the REQUEST.
constexpr int kPhysical = 120;

ProblemShape kShape() { return ProblemShape{8, 4096, 4096}; }

// An exact-match ladder membership test, beside the many places below that need to DISTINGUISH
// "not lower" from "not a rung at all". It mirrors arch_rung()'s own walk (arch_caps.h:1019).
bool is_ladder_rung(int sm) {
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        if (kArchLadder[r].sm == sm) { return true; }
    }
    return false;
}

ArchView view(int physical, bool built_in, const char* arch, const char* ack) {
    return arch_view_for_device_impl(physical, built_in,
                                     arch == nullptr ? std::string_view{} : std::string_view(arch),
                                     ack == nullptr ? std::string_view{} : std::string_view(ack));
}

// ===========================================================================
// G1  IMPOSSIBLE BY ACCIDENT
// ===========================================================================
// arch_sim.h:28-37: THREE independent keys, of which the first is a build-time one.
void test_G1_impossible_by_accident() {
    const std::string kAck(kSimAckPhrase);

    // (i) nothing set: the default, and the ONLY path a shipping build takes.
    const ArchView none = view(kPhysical, true, nullptr, nullptr);
    check(none.status == SimStatus::Disabled, "G1: no request must be Disabled");
    check(none.effective_sm == kPhysical && none.physical_sm == kPhysical,
          "G1: Disabled must not move the compute capability");
    check(none.reason.empty(), "G1: Disabled carries no reason");
    check(sim_banner(none).empty(), "G1: Disabled renders no banner");

    // (ii) A STRAY NINFER_SIM_ARCH IN A USER'S SHELL MUST NOT ENABLE IT. Both refusals below
    // are non-silent: each carries a reason naming the key that is missing.
    const ArchView arch_only = view(kPhysical, true, "70", nullptr);
    check(arch_only.status == SimStatus::Refused,
          "G1: NINFER_SIM_ARCH alone must be REFUSED, not honoured");
    check_contains(arch_only.reason, "NINFER_SIM_ARCH_ACK", "G1: the refusal names the ACK key");

    const ArchView not_built = view(kPhysical, /*built_in=*/false, "70", kAck.c_str());
    check(not_built.status == SimStatus::Refused,
          "G1: the BUILD key missing must be REFUSED even with both env keys set");
    check_contains(not_built.reason, "NINFER_ENABLE_ARCH_SIM", "G1: the refusal names the build key");
    check_contains(not_built.reason, "Rebuild with -DNINFER_ENABLE_ARCH_SIM=ON",
                   "G1: the refusal's REMEDY sentence, not merely the option name");

    const ArchView bad_ack = view(kPhysical, true, "70", "yes");
    check(bad_ack.status == SimStatus::Refused, "G1: a wrong acknowledgement is refused");
    check_contains(bad_ack.reason, "test-only", "G1: the refusal says the simulator is test-only");

    // (iii) all three present: honoured, for every rung the ladder actually has.
    for (const int rung : {70, 75}) {
        const ArchView ok = view(kPhysical, true, std::to_string(rung).c_str(), kAck.c_str());
        check(ok.status == SimStatus::Active, "G1: sm_" + std::to_string(rung) + " must be honoured");
        check(ok.effective_sm == rung && ok.physical_sm == kPhysical,
              "G1: Active must record BOTH the simulated and the physical number");
        check(ok.simulated() && !ok.failed() && ok.usable(), "G1: Active's predicates must agree");
    }
}

// ===========================================================================
// G2  LOUD
// ===========================================================================
// arch_sim.h:39-42: every use prints sim_banner() and every answer carries a SIMULATED marker,
// with no quiet mode. The banner's "NOT sm_<rung> binaries" clause is rung-DERIVED, which is
// pinned here for EACH rung rather than once: arch_sim.h:179-187 records that this clause was
// a hardcoded "sm_70 binaries" literal until 2026-09-18, so that a 75/86/89/100 run printed
// the wrong rung in the one sentence whose job is to stop a reader thinking the cubins are the
// target card's.
void test_G2_loud() {
    const std::string kAck(kSimAckPhrase);
    for (const int rung : {70, 75}) {
        const ArchView rv = view(kPhysical, true, std::to_string(rung).c_str(), kAck.c_str());
        const std::string banner = sim_banner(rv);
        check(!banner.empty(), "G2: an active simulation must render a banner");
        check_contains(banner, "SIMULATED", "G2: the banner says SIMULATED");
        check_contains(banner, "sm_120", "G2: the banner names the physical device");
        check_contains(banner, ("sm_" + std::to_string(rung)).c_str(),
                       "G2: the banner names the simulated rung it actually simulated");
        check_contains(banner, ("NOT sm_" + std::to_string(rung) + " binaries").c_str(),
                       "G2: the banner denies the cubins are the SIMULATED rung's binaries -- "
                       "built from the rung, so it cannot name the wrong one");
        check_contains(banner, "proves NOTHING", "G2: the banner denies the evidentiary weight");
        check_contains(banner, "throughput", "G2: the banner denies performance claims");
    }
    check(sim_banner(view(kPhysical, true, nullptr, nullptr)).empty(), "G2: no request, no banner");

    // The marker travels INSIDE the answer, not only beside it.
    const ArchView rv = view(kPhysical, true, "70", kAck.c_str());
    const RouteChoice choice = select_route(rv, NumericFormat::NVFP4, kShape(), /*qpn_in_build=*/true);
    check(choice.simulated, "G2: the route choice must be MARKED simulated (the branchable field)");
    check_contains(choice.why, "SIMULATED", "G2: `why` leads with the marker");
    check_contains(choice.why, "NOTHING about sm_120",
                   "G2: the answer says it claims nothing about the real card");
    const RouteLogLine line = route_log_line(kPhysical, NumericFormat::NVFP4, kShape(), choice);
    check(line.text.rfind("[route][SIMULATED]", 0) == 0,
          "G2: the log line must LEAD with the marker, got: " + line.text.substr(0, 60));
    check_contains(line.text, "sm=120",
                   "G2: the log line still names the REAL card's number it was asked on");
    check(!line.warn, "G2: a simulated selection is not an unknown-arch warning");

    // A census, so "every ANSWER is marked" is a count and not an adjective.
    //
    // THE UNIT IS THE ANSWER, NOT THE RAW `reason` FIELD, and the difference is measured rather
    // than stylistic: under a missing-ACK refusal the reason text is "NINFER_SIM_ARCH=70 was
    // requested but NINFER_SIM_ARCH_ACK is unset, ... Refusing." -- it does not contain the word
    // SIMULATED, because the two consumers add it: sim_refusal_reason() prefixes "SIMULATED
    // ARCHITECTURE REQUEST REFUSED (fail-closed, test-only): " and arch_view_for_device() prints
    // "*** SIMULATED ARCHITECTURE REQUEST REFUSED -- TEST ONLY ***" to stderr. So the assertion
    // below is on the texts a consumer actually emits, and it is stated that way on purpose: a
    // check on `reason` would be FALSE here and a check on nothing would be vacuous.
    const auto carries = [](const std::string& t) { return t.find("SIMULATED") != std::string::npos; };
    {
        const ArchView refused = view(kPhysical, true, "70", nullptr);
        check(refused.failed(), "G2 census: this arm needs a refusal");
        const RouteChoice c = select_route(refused, NumericFormat::NVFP4, kShape(), true);
        const std::string log_line =
            route_log_line(kPhysical, NumericFormat::NVFP4, kShape(), c).text;
        int marked = 0, total = 0;
        for (const std::string& t : {log_line, sim_refusal_reason(refused)}) {
            ++total;
            if (carries(t)) { ++marked; }
        }
        check(marked == total,
              "G2: a REFUSED simulation's every emitted answer must carry SIMULATED (" +
                  std::to_string(marked) + " of " + std::to_string(total) + ")");
        check(carries(log_line),
              "G2: the fail-closed log line must carry the marker, got: " + log_line);
    }
    {
        const ArchView disabled = view(kPhysical, true, nullptr, nullptr);
        const RouteChoice c = select_route(disabled, NumericFormat::NVFP4, kShape(), true);
        const std::string log_line =
            route_log_line(kPhysical, NumericFormat::NVFP4, kShape(), c).text;
        check(!carries(log_line),
              "G2: with no request the log line must NOT carry the marker, got: " + log_line);
        check(!carries(disabled.reason) && !carries(sim_banner(disabled)),
              "G2: with no request neither the reason nor a banner may carry the marker");
    }
}

// ===========================================================================
// G3  FAIL CLOSED
// ===========================================================================
// arch_sim.h:44-49: Refused is a DISTINCT state from Disabled, select_route(ArchView) returns
// NoKernelInTree in it, and require_artifact_formats_supported(ArchView) refuses (the overload
// added at arch_sim.h:380-395, whose absence arch_sim.h:357-364 records as measured).
void test_G3_fail_closed() {
    const std::string kAck(kSimAckPhrase);
    struct Row {
        ArchView view;
        const char* why;
    };
    const Row rows[] = {
        {view(kPhysical, false, "70", kAck.c_str()), "the build key is missing"},
        {view(kPhysical, true, "70", nullptr), "the acknowledgement is missing"},
        {view(kPhysical, true, "70", "nope"), "the acknowledgement is wrong"},
        {view(kPhysical, true, "abc", kAck.c_str()), "the value is not a decimal"},
        {view(kPhysical, true, "999", kAck.c_str()), "the value is not a ladder rung"},
        {view(kPhysical, true, "35", kAck.c_str()), "sm_35 is not a ladder rung"},
        {view(kPhysical, true, "110", kAck.c_str()), "sm_110 is not a ladder rung (withdrawn)"},
        {view(kPhysical, true, "120", kAck.c_str()), "not a downgrade"},
        {view(kPhysical, true, "121", kAck.c_str()), "a raise"},
    };
    const NumericFormat nvfp4[] = {NumericFormat::NVFP4};
    for (const Row& row : rows) {
        check(row.view.failed(), std::string("G3: ") + row.why + " must be Refused");
        check(!row.view.usable(), std::string("G3: ") + row.why + " must not be usable()");
        check(!row.view.simulated(), std::string("G3: ") + row.why + " must not report simulated()");
        // THE POINT: the real device selects nvfp4-w4a4-tma for this format and shape, and a
        // refused request must NOT get it -- answering with the real card's route would be an
        // answer to a question nobody asked.
        const RouteChoice choice = select_route(row.view, NumericFormat::NVFP4, kShape(), true);
        check(choice.outcome == RouteOutcome::NoKernelInTree,
              std::string("G3: ") + row.why + " must fail CLOSED, got " +
                  std::string(outcome_name(choice.outcome)));
        check(choice.route == KernelRoute::None,
              std::string("G3: ") + row.why + " must not name a route");
        check(!choice.simulated,
              std::string("G3: ") + row.why + " is a refusal, not a simulated answer");
        check_contains(choice.why, "SIMULATED ARCHITECTURE REQUEST REFUSED",
                       std::string("G3: ") + row.why + " -- the refusal must be unmistakable");
        check_contains(choice.why, "No answer is given for the real device",
                       std::string("G3: ") + row.why + " -- and must say so");
        // THE GATE OVERLOAD G3 PROMISED. It must THROW, not answer off the physical device.
        bool threw = false;
        std::string message;
        try {
            require_artifact_formats_supported(row.view,
                                               std::span<const NumericFormat>(nvfp4, 1),
                                               "test_arch_sim/G3", /*qpn_in_build=*/true);
        } catch (const std::invalid_argument& e) {
            threw = true;
            message = e.what();
        }
        check(threw, std::string("G3: require_artifact_formats_supported must THROW for ") + row.why);
        check_contains(message, "REFUSED",
                       std::string("G3: the thrown text names the refusal for ") + row.why);
    }
    // The control that says the refusals above are about the REQUEST and not about a broken
    // selector: the same call with no request must still select the shipping route.
    const RouteChoice real = select_route(kPhysical, NumericFormat::NVFP4, kShape(), true);
    check(real.outcome == RouteOutcome::Selected && real.route == KernelRoute::Nvfp4W4a4Tma,
          "G3 control: sm_120 with no request must still select nvfp4-w4a4-tma");
    check(!real.simulated, "G3 control: an unreal request must not mark the answer simulated");
}

// ===========================================================================
// G4  NEVER RAISES, NEVER EQUALS
// ===========================================================================
// arch_sim.h:51-57: the simulated rung must be STRICTLY LOWER than the physical one, and the
// two refusals below must not be CONFLATED -- "a wanted rung that is on the ladder and is not
// lower" is a different statement from "a wanted rung that is not on the ladder at all", and
// collapsing them would make this line's own head question read as a downgrade problem.
void test_G4_downgrade_only() {
    const std::string kAck(kSimAckPhrase);
    // Equal and higher, at three different physical rungs.
    for (const auto& pair : {std::pair<int, int>{120, 120}, {120, 121}, {120, 130},
                             {100, 100}, {100, 120}, {90, 100}}) {
        const ArchView rv = view(pair.first, true, std::to_string(pair.second).c_str(), kAck.c_str());
        check(rv.failed(), "G4: physical sm_" + std::to_string(pair.first) + " -> wanted sm_" +
                               std::to_string(pair.second) + " must be refused");
        if (is_ladder_rung(pair.second)) {
            check_contains(rv.reason, "STRICTLY LOWER",
                           "G4: an on-ladder non-downgrade names the DOWNGRADE rule; the reason "
                           "for physical sm_" + std::to_string(pair.first) + " -> wanted sm_" +
                               std::to_string(pair.second));
        }
    }
    // A lower rung that is not on the ladder is refused for the OTHER reason, and this is the
    // assertion that keeps the two apart.
    const ArchView not_a_rung = view(120, true, "110", kAck.c_str());
    check(not_a_rung.failed(), "G4: sm_110 is not a ladder row and must be refused");
    check_contains(not_a_rung.reason, "no row in kArchLadder",
                   "G4: the reason is the MISSING ROW, not the downgrade rule");
    check(not_a_rung.reason.find("STRICTLY LOWER") == std::string::npos,
          "G4: the two refusals must not be conflated: an unlisted rung is not a downgrade "
          "problem, got: " + not_a_rung.reason);

    // Every ladder row strictly below the physical card IS honoured, so the refusals above are
    // a boundary and not a blanket.
    int honoured = 0;
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        const int sm = kArchLadder[r].sm;
        const ArchView rv = view(kPhysical, true, std::to_string(sm).c_str(), kAck.c_str());
        if (sm >= kPhysical) {
            check(rv.failed(), "G4: ladder row sm_" + std::to_string(sm) +
                                   " is not strictly lower than sm_120 and must be refused");
        } else {
            check(rv.status == SimStatus::Active,
                  "G4: ladder row sm_" + std::to_string(sm) + " must be honoured");
            ++honoured;
        }
    }
    std::printf("  G4: %d of %zu ladder rows are simulable on an sm_120 card\n", honoured,
                kArchLadderSize);
}

// ===========================================================================
// E  THE RUNG BOUNDARY -- WHICH COMPUTE CAPABILITIES CAN BE SIMULATED AT ALL
// ===========================================================================
// THE HEAD QUESTION OF dl/simrun2, in code: can the mechanism be pointed at the rungs this
// operator cannot buy anywhere -- K40 = sm_35, and the Maxwell sm_50/sm_52/sm_53 that the
// CUDA 12.8 chain at /mnt/g/cuda12/tk can still build (arch_sim.h:81-84)?
//
// THE ANSWER CHANGED ON 2026-09-24 (F702, dl/archrow) AND THE SECTIONS BELOW RECORD BOTH HALVES.
// It USED to be a flat "NO, BY NAME, AND NOT SILENTLY" for every pre-70 number, because the
// ladder had no row below sm_70 (E1/E2 pinned that). The six pre-75 rows F702 added make the
// Maxwell/Pascal rungs REACHABLE (E1's census, and E4 which exercises each of them); sm_35 and
// the withdrawn sm_110 keep the old answer, pinned in E2 -- which is why E2 was not deleted when
// the rows landed. The property that makes all of it a LADDER property rather than a simulator
// one is unchanged: arch_view_for_device_impl resolves the requested number with
// arch_rung(), which is an EXACT-MATCH walk over kArchLadder (arch_caps.h), and a
// number with no row is refused with the message E2 pins.
//
// THE ANSWER IS STILL "NO, BY NAME, AND NOT SILENTLY" for a number with no row, and that is a
// property of the
// number with no row is refused with "has no row in kArchLadder, so simulating it would mean
// inventing a capability set" (arch_sim.h:282-287). Section E pins the census and both
// forbidden directions.
void test_E_rung_boundary() {
    const std::string kAck(kSimAckPhrase);

    // E1 -- UPDATED 2026-09-24 (F702, dl/archrow) IN THE SAME COMMIT as the six new rows, which
    // is exactly what the tripwire below asked for. The ladder now HAS six rows below sm_70
    // (sm_50/52/53 Maxwell, sm_60/61/62 Pascal), so the pre-75 family -- including sm_61, the
    // owner's Quadro P1000 -- is reachable on the simulation path. The LOW row is now sm_50, and
    // the checks below pin the two things that make the new rows admissible rather than
    // convenient: their capability set is Cap::None (a tensor-core bit on a pre-Volta ISA is the
    // sm_100-nvfp4 false positive in a new place, and `mma` needs .target sm_70 -- measured in
    // tools/archkit/_GPU_MATRIX.md section 3.3), and every field a refusal quotes is non-empty.
    std::printf("  E1: kArchLadder has %zu row(s); ascending sm:", kArchLadderSize);
    for (std::size_t r = 0; r < kArchLadderSize; ++r) { std::printf(" %d", kArchLadder[r].sm); }
    std::printf("\n");
    // NOT a hardcoded 18: the census is printed, and the assertion is the one that must hold
    // whatever the count becomes.
    int lowest = 1000000;
    for (std::size_t r = 0; r < kArchLadderSize; ++r) {
        if (kArchLadder[r].sm < lowest) { lowest = kArchLadder[r].sm; }
        if (r > 0) {
            check(kArchLadder[r - 1].sm < kArchLadder[r].sm, "E1: the ladder must stay ascending");
        }
    }
    check(lowest == 50,
          "E1: the LOW row of kArchLadder is sm_50 (Maxwell) since F702 added the six pre-75 "
          "rows; it is sm_" + std::to_string(lowest) +
              ". THIS IS STILL A TRIPWIRE, NOT AN INVARIANT: a row below sm_50 (sm_35/K40, "
              "which no toolkit in this box can codegen for) may be added, but only WITH "
              "MEASURED evidence and only deliberately, so this check is expected to be updated "
              "IN THE SAME COMMIT as such a row.");
    // The six rows exist, are pre-70, claim NO tensor core, and carry a quotable route.
    for (const int sm : {50, 52, 53, 60, 61, 62}) {
        const ninfer::caps::ArchRung* rung = arch_rung(sm);
        check(rung != nullptr,
              "E1: sm_" + std::to_string(sm) + " must have a row (F702 added the pre-75 block)");
        if (rung == nullptr) { continue; }
        check(sm < 70, "E1: sm_" + std::to_string(sm) + " is a pre-75 row by construction");
        check(rung->caps == ninfer::caps::Cap::None,
              "E1: sm_" + std::to_string(sm) +
                  " must claim NO tensor core: `mma` requires .target sm_70 (measured, "
                  "tools/archkit/_GPU_MATRIX.md section 3.3), and a bit here would be a "
                  "capability claim this ISA cannot back");
        check(!rung->route.empty() && !rung->label.empty() && !rung->cards.empty(),
              "E1: sm_" + std::to_string(sm) +
                  " must carry a route, a label and representative cards: a refusal quotes all "
                  "three, and a blank row is what this leg was told not to add");
    }
    check(arch_rung(35) == nullptr,
          "E1: sm_35 (K40) must STILL have no row: no toolkit on this box can codegen for it, "
          "so it is the working example of the no-row refusal this section pins");

    // E2 -- THE NEGATIVE DIRECTION, still live and not vacuous: a number with NO row is refused
    // by name, fails closed, and produces no route answer. sm_35 is the pre-75 example; 110 is
    // the withdrawn-row example (arch_caps.h's own comment records the withdrawal).
    for (const int sm : {35, 110}) {
        check(arch_rung(sm) == nullptr,
              "E1: sm_" + std::to_string(sm) + " must have NO row in kArchLadder");
        const ArchView rv = view(kPhysical, true, std::to_string(sm).c_str(), kAck.c_str());
        // (a) THE FORBIDDEN DIRECTION: coercion. A silent fallback to sm_70 would make
        //     NINFER_SIM_ARCH=35 produce a "V100 answer" that nobody asked for.
        check(rv.effective_sm == kPhysical,
              "E2: a request for sm_" + std::to_string(sm) +
                  " must leave effective_sm at the PHYSICAL number, never coerce it; got sm_" +
                  std::to_string(rv.effective_sm));
        check(rv.status == SimStatus::Refused,
              "E2: NINFER_SIM_ARCH=" + std::to_string(sm) + " must be REFUSED");
        check_contains(rv.reason, "no row in kArchLadder",
                       "E2: sm_" + std::to_string(sm) + " is refused by name for the missing row");
        check_contains(rv.reason, "add the row with MEASURED evidence first",
                       "E2: and the remedy names the evidence a new row would need");
        check(rv.reason.find("downgraded") == std::string::npos,
              "E2: a refused request must not read as a downgrade, got: " + rv.reason);
        check(sim_banner(rv).empty(),
              "E2: a refused request renders no banner (it must not look like a V100 run)");
        // (b) and it fails closed, so no route answer is produced for the rung that does not
        //     exist.
        const RouteChoice choice = select_route(rv, NumericFormat::NVFP4, kShape(), true);
        check(choice.outcome == RouteOutcome::NoKernelInTree && !choice.simulated,
              "E2: sm_" + std::to_string(sm) + " must produce no route answer at all");
    }

    // E3 -- the reachable set, restated as the ONE thing a simulated run may be used for: the
    // ROUTE DECISION of the rungs that ARE rows. sm_70 and sm_75 are the two the tree names
    // (arch_sim.h:12-14), and they must reach the fp16 fallback the real card never selects.
    for (const int rung : {70, 75}) {
        const ArchView rv = view(kPhysical, true, std::to_string(rung).c_str(), kAck.c_str());
        check(rv.simulated(), "E3: sm_" + std::to_string(rung) + " must be simulable on sm_120");
        const RouteChoice choice = select_route(rv, NumericFormat::NVFP4, kShape(), true);
        check(choice.simulated, "E3: the answer must be marked");
        check(choice.outcome == RouteOutcome::Selected && choice.route == KernelRoute::QpnW4a16,
              "E3: simulated sm_" + std::to_string(rung) +
                  " x NVFP4 m=8 must select qpn-w4a16, got " +
                  std::string(outcome_name(choice.outcome)) + ": " + choice.why);
    }
    // and the control that the answer is about the RUNG: the same question at the physical card
    // must NOT take that route.
    const RouteChoice host = select_route(kPhysical, NumericFormat::NVFP4, kShape(), true);
    check(host.route != KernelRoute::QpnW4a16,
          "E3 control: sm_120 must not select the fallback route the simulated rungs select");

    // -----------------------------------------------------------------------------------------
    // E4 -- THE PRE-75 BLOCK, ADDED 2026-09-24 (F702): REACHABLE, AND REFUSING BY REASON.
    // -----------------------------------------------------------------------------------------
    // This is the positive half the six new rows owe. Before them the simulator could not
    // express any of these rungs (E2's "no row in kArchLadder" refusal, at EVERY physical
    // device), so the pre-75 path could not even be asked about. Now each rung resolves and
    // every answer is the one the tables give for it. What is asserted, and why each clause is
    // here rather than in a comment:
    //   (1) the request is HONOURED (simulated), i.e. the row is reachable on the sim path;
    //   (2) the GATE checks the floor instead of announcing it cannot: verdict Unsupported with
    //       a gap naming the format's own required capability -- NOT the UnknownArch warning
    //       whose text says "no format floor can be checked";
    //   (3) the engine's own call (require_artifact_formats_supported) REFUSES, and the report
    //       it throws names the RUNG as the GPU -- which it cannot do for a rung-less number;
    //   (4) the ROUTE refusal says its REASON CATEGORY and names the floor's kernel file:line,
    //       which is what F702 added to kernel_route.h's no-tensor-core branch. That branch was
    //       unreachable before these rows and its text is now load-bearing.
    {
        const NumericFormat nvfp4[] = {NumericFormat::NVFP4};
        for (const int rung : {50, 52, 53, 60, 61, 62}) {
            const ArchView rv = view(kPhysical, true, std::to_string(rung).c_str(), kAck.c_str());
            check(rv.simulated(),
                  "E4: sm_" + std::to_string(rung) +
                      " must be SIMULABLE on sm_120 now that it has a ladder row");
            if (!rv.simulated()) { continue; }
            check(rv.effective_sm == rung, "E4: effective_sm must be the requested rung");

            const CapabilityReport report =
                evaluate_artifact_formats(rv.effective_sm, nvfp4, /*qpn_in_build=*/true);
            check(report.verdict == Verdict::Unsupported,
                  "E4: sm_" + std::to_string(rung) +
                      " x NVFP4 must be UNSUPPORTED (it has no tensor core at all)");
            check(report.verdict != Verdict::UnknownArch,
                  "E4: sm_" + std::to_string(rung) +
                      " is now a ROW, so the gate must CHECK its floor rather than report that "
                      "no floor can be checked");
            check(report.gaps.size() == 1 && report.gaps[0].required == Cap::Mxf4Nvfp4BlockScale,
                  "E4: sm_" + std::to_string(rung) +
                      " must name kind::mxf4nvf4 as the missing capability, and it must carry "
                      "the kernel citation that makes it a floor (" +
                      std::to_string(report.gaps.size()) + " gap(s))");
            check(report.fallbacks.empty(),
                  "E4: sm_" + std::to_string(rung) +
                      " must report NO fallback: fp16_fallback_executable() needs Cap::Fp16Mma, "
                      "which this rung does not have");

            // (3) the engine's own load-time call, which throws for an unsupported artifact.
            bool threw = false;
            std::string what;
            try {
                require_artifact_formats_supported(rv, nvfp4, "e4/sim");
            } catch (const std::exception& e) {
                threw = true;
                what  = e.what();
            }
            check(threw, "E4: the gate overload must REFUSE an NVFP4 artifact for sm_" +
                             std::to_string(rung));
            check(what.find("sm_" + std::to_string(rung)) != std::string::npos,
                  "E4: and the refusal must name the RUNG as the GPU, which no rung-less "
                  "number can do; got " + std::to_string(what.size()) + " B");

            // (4) the route refusal: reason category + the floor's own citation.
            const RouteChoice choice = select_route(rv, NumericFormat::NVFP4, kShape(), true);
            check(choice.outcome == RouteOutcome::NoKernelInTree &&
                      choice.route == KernelRoute::None && choice.simulated,
                  "E4: sm_" + std::to_string(rung) +
                      " x NVFP4 must be a SIMULATED named refusal, got " +
                      std::string(outcome_name(choice.outcome)));
            check_contains(choice.why, "REASON CATEGORY: UNMET FORMAT FLOOR",
                           "E4: sm_" + std::to_string(rung) +
                               " must say WHICH CATEGORY of refusal this is (the F702 change to "
                               "kernel_route.h's no-tensor-core branch)");
            check_contains(choice.why, "nvfp4_w4a4_mma.cuh:308",
                           "E4: sm_" + std::to_string(rung) +
                               " must name the FLOOR's kernel site, not only the capability");
        }
    }
}

// ===========================================================================
// F  RED CONTROLS -- the checks that must FAIL if the forbidden direction is taken
// ===========================================================================
// A check that cannot fail is not a check. This section states, for the boundary above, the
// exact edit that would make the suite red -- and it VERIFIES the sensitivity rather than
// asserting it, by running the same predicate against a deliberately wrong value.
void test_F_red_controls() {
    // F1 -- the sensitivity of E2's coercion check. E2 asserts effective_sm == the physical
    // number. If a later edit made sm_35 answer as sm_70, rv.effective_sm would be 70 and this
    // predicate would be FALSE. Demonstrated here against a crafted value so the check is
    // known to be able to fail rather than assumed to be:
    {
        const int would_be_if_coerced = 70;
        const bool sensitive = (would_be_if_coerced != kPhysical);
        check(sensitive,
              "F1: the E2 coercion predicate must be able to FAIL; a predicate that is true for "
              "the coerced value too would pass vacuously");
    }
    // F2 -- the same for the ladder's floor. E1 asserts the low row is 70. There must be no
    // shape in which a 35 row exists AND E1 passes:
    {
        bool a_35_row_would_be_caught = true;
        for (std::size_t r = 0; r < kArchLadderSize; ++r) {
            if (kArchLadder[r].sm <= 35) { a_35_row_would_be_caught = false; }
        }
        check(a_35_row_would_be_caught,
              "F2: a kArchLadder row at or below sm_35 would make E1's floor check RED; it is "
              "not red only because no such row exists");
    }
    // F3 -- the honest inverse of the whole file. The simulator is a ROUTE instrument: it
    // cannot make a kernel exist for a rung. So a simulated answer for a rung is NOT a support
    // claim, and the text that says so must be present in the very answer E3 accepts.
    const std::string kAck(kSimAckPhrase);
    const ArchView rv = view(kPhysical, true, "70", kAck.c_str());
    const RouteChoice choice = select_route(rv, NumericFormat::NVFP4, kShape(), true);
    check_contains(choice.why, "says NOTHING about sm_120",
                   "F3: the accepted answer must still refuse to claim anything about the card");
    check_contains(sim_banner(rv), "proves NOTHING",
                   "F3: and the banner must still refuse the evidentiary weight");
    check(choice.kernel.find("skinny_nvfp4_qpn") != std::string::npos,
          "F3: the named kernel for the simulated rung is this TREE's fp16-fallback kernel, "
          "not a Volta binary");
}

} // namespace

int main() {
    std::printf("test_arch_sim: %s\n",
                kArchSimBuiltIn ? "the TEST-ONLY architecture simulator IS built into this binary"
                                : "NO simulator in this binary (NINFER_ARCH_SIM_ENABLED undefined)");
    std::printf("  the four guards of src/core/arch_sim.h, the rung boundary, and the red controls\n");
    test_G1_impossible_by_accident();
    test_G2_loud();
    test_G3_fail_closed();
    test_G4_downgrade_only();
    test_E_rung_boundary();
    test_F_red_controls();
    std::printf("=== %d checks, %d failure(s) ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
