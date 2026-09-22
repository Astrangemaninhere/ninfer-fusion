// ---------------------------------------------------------------------------
// test_vendor_sim.cpp
//
// The acceptance test for src/core/vendor_sim.h -- THE VENDOR-AXIS CAPABILITY OVERRIDE -- and
// for src/core/announce_once.h, the keyed add-only announce set it is built on.
//
// IT IS ONE BINARY on purpose, so that the parts that have to agree are checked against each
// other: the guard is checked against the AMD table it answers from, and the keyed announce set
// is checked against the ONE-SHOT shape the arch axis uses, in the same process.
//
// ⛔ WHAT A GREEN HERE DOES NOT MEAN, AND IT IS THE FIRST THING A READER MUST BE TOLD:
//   * It is NOT a portability claim. src/core/vendor_sim.h translates NOTHING: not one
//     instruction is rewritten, re-encoded or recompiled for any target, and no toolchain for
//     any target exists on this box.
//   * It is NOT AMD support. No AMD kernel is compiled into this binary and no AMD card was
//     touched. Every vendor-side fact quoted from kPtxFamilyAmdStatus is EXTERNAL-UNPROBED.
//   * ⭐ IT ADDS ZERO ROUTE COVERAGE, ZERO ADMISSION COVERAGE AND ZERO NEW AMD FACT. The AMD
//     branch is ALREADY reached on this box BY PARAMETER -- tests/test_arch_caps.cpp calls
//     amd_rung("gfx906"), walks amd_format_verdict(target, format) over the six literal targets,
//     and calls render_amd_format_refusal("gfx906", NVFP4) with the target as an argument. There
//     is no AMD kernel in this tree to route to, so there is no route decision a different table
//     answer could change. ⇒ What this binary checks is the GUARD MACHINERY, and only that.
//
// ⭐ THE FOUR GUARD GAPS THIS FILE EXISTS TO CLOSE, EACH WITH AN ASSERTION THAT CAN FAIL:
//   (1) G2 was a ONE-SHOT `bool` PER PROCESS, weaker than its own prose, and on src/core/
//       arch_sim.h:324 it is not even synchronized. ==> V2, whose A5c dies on a `bool`.
//   (2) G4 ("strictly lower than the physical rung") HAD NO VENDOR COUNTERPART, because a vendor
//       is not ordered. ==> V4: G4-V NEVER EQUALS + NEVER WIDENS.
//   (3) THE AMD SIDE COULD NOT FAIL CLOSED: render_amd_format_refusal() returns a std::string and
//       a string cannot throw, so a non-NoKernelInTree verdict renders EMPTY and silence reads as
//       "fine". ==> V3, and V6 GAP 3, which MEASURES the silence rather than asserting it by
//       prose: it counts the formats for which the AMD text is empty, prints the count, and
//       asserts the gate refuses every one of them anyway.
//   (4) `sim_vendor` WOULD BE RECORDABLE without `sim_rung`'s parse-time refusal. ==> V6 GAP 4,
//       a source read with a TWO-WAY control: the field is asserted ABSENT and the rule that IS
//       there is asserted PRESENT, so a broken read cannot report a false clean.
//
// THE THREE NAMED MUTATIONS, AND THE ASSERTION EACH MUST DIE ON. ⚠ THEY ARE PREPARED AND
// **NOT RUN**: this line ran no compiler, no build and no ctest (see the report's REFUSALS). The
// landing line owes the run. Each is a text substitution in the file named, applied one at a time:
//
//   M1_oneshot_bool          src/core/vendor_sim.h
//     replace the two `ninfer::detail::announce_once_keyed(...)` calls in vendor_view_for_device() with
//     `static bool announced = false; if (!announced) { announced = true; ... }`
//     ==> MUST FAIL V2 A5c ("a SECOND, DIFFERENT vendor class IS announced"). It may PASS V2
//         A5a/A5b, which is the point: the landed shape is weaker than its own prose, not broken.
//
//   M2_admitted_from_verdict src/core/vendor_sim.h
//     in vendor_format_answer(), set `answer.admitted = (answer.verdict !=
//     AmdFormatVerdict::UnknownTarget);`
//     ==> MUST FAIL V4 NEVER WIDENS, on NotATensorCoreOperand. That is the laundering path
//         src/core/arch_caps.h:2131-2135 names in its own words: the verdict is "NOT a claim that
//         the format works on AMD; it is the absence of the one specific blocker this table knows
//         how to name".
//
//   M3_noop_honoured          src/core/vendor_sim.h
//     delete the `if (view.effective == physical)` block in vendor_view_for_device_impl()
//     ==> MUST FAIL V4 NEVER EQUALS and V5 (the unverifiable `physical` input then no longer fails
//         closed, so a caller that declares the box AMD obtains an ACTIVE AMD view).
//
// Evidence grades used below, as the tree's convention requires:
//   [PARAM] the claim was exercised as a PARAMETER, not through a real device. Every AMD claim in
//           this file is [PARAM], and that is a limitation stated rather than hidden.
//   [READ]  the claim is about text this test read off the tree (it does not shell out).
// ---------------------------------------------------------------------------

#include "core/announce_once.h"
#include "core/arch_caps.h"
#include "core/arch_sim.h"
#include "core/vendor_sim.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
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
          what + "  (looking for \"" + std::string(needle) + "\" in: " + haystack + ")");
}

// The parameterised view: every input is an argument, so this binary exercises the enabled and
// the not-enabled worlds and every refusal value WITHOUT a rebuild and WITHOUT a second process.
// Same idiom as `view()` in tests/test_arch_generic_fallback.cpp.
VendorView vview(VendorClass physical, bool built_in, const char* value, const char* ack) {
    return vendor_view_for_device_impl(
        physical, built_in,
        value == nullptr ? std::string_view{} : std::string_view(value),
        ack == nullptr ? std::string_view{} : std::string_view(ack));
}

const std::string kAck(kVendorSimAckPhrase);
const std::string kArchAck(kSimAckPhrase);

// The six kAmdLadder rows, spelled here so a row REMOVED from the table is a red test rather than
// a silently smaller walk. Spelled from the table the test read (src/core/arch_caps.h:2012-2064).
const char* const kAmdTargets[] = {"gfx906", "gfx908", "gfx90a", "gfx942", "gfx1100", "gfx1201"};
constexpr std::size_t kAmdTargetCount = sizeof(kAmdTargets) / sizeof(kAmdTargets[0]);

// [READ] helpers, the idiom tests/test_sim_no_support.cpp:551-554 uses for the same purpose.
std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}
std::string source_dir() {
#ifdef NINFER_SOURCE_DIR
    return std::string(NINFER_SOURCE_DIR);
#else
    return std::string("/home/user/ninfer-fusion");
#endif
}

// ===========================================================================
// V1 -- G1-V: IMPOSSIBLE BY ACCIDENT. Four independent keys, three of which a
//       stray environment variable cannot satisfy.
// ===========================================================================
void V1_impossible_by_accident() {
    // (a) THE BUILD FACT. With the key absent the request is refused BY THE BUILD FACT, and the
    // refusal must NAME a remedy for EVERY reader who can hit it -- a refusal that does not say
    // what to do is not an on-ramp. THERE ARE TWO READERS AND TWO ROUTES, so the test asserts
    // them SEPARATELY: the CMake option (route 1) and the preprocessor key a hand compile needs
    // (route 2). ONE NEEDLE WOULD ACCEPT A HALF-REMEDY: naming the CMake option alone leaves a
    // hand-compiling reader holding a recipe that cannot reach them, and naming the key alone
    // misdirects the CMake reader. Both are the defect this block was widened for.
    {
        const VendorView v = vview(VendorClass::Nvidia, /*built_in=*/false, "gfx906", kAck.c_str());
        check(v.failed(), "no build key: the request must be REFUSED, not honoured");
        check(v.refusal == VendorRefusal::NotBuiltIn, "no build key: the refusal must be NotBuiltIn");
        check(!v.reason.empty(), "no build key: the refusal must carry a reason");
        // THE REMEDY SENTENCE, NOT MERELY THE NAME. "NINFER_ENABLE_VENDOR_SIM=ON" occurs TWICE
        // in this refusal: once in the diagnostic sentence and once in route (1) below. MEASURED
        // 2026-09-19 (dl/falsegreen): deleting route (1)'s whole content -- option name,
        // "reconfigure", the reason it is the only route for a CMake reader -- while keeping the
        // label left this suite GREEN, 244 checks / 0 failures.
        check_contains(v.reason, "reconfigure with -DNINFER_ENABLE_VENDOR_SIM=ON",
                       "no build key: route 1 must still tell the CMake reader to RECONFIGURE, not "
                       "merely carry the option name somewhere (the name also appears above)");
        check_contains(v.reason, "NINFER_ENABLE_VENDOR_SIM=ON",
                       "no build key: the refusal must name the CMake option to rebuild with "
                       "(route 1: the reader who builds this test through CMake)");
        check_contains(v.reason, "NINFER_VENDOR_SIM_ENABLED",
                       "no build key: the refusal must ALSO name the preprocessor key -- a "
                       "reader who hand-compiles never runs CMake, so the option alone is no "
                       "remedy for them (route 2)");
        // AND THE ROUTES MUST BE ADDRESSED TO A READER, not merely listed. A refusal that
        // prints two commands without saying which one belongs to whom leaves the reader to
        // guess, which is the same dead end as naming only one of them. The two needles
        // below are the route LABELS, one reader each. They are matched case-sensitively
        // because the header spells them in caps, and a reflow that changes a label is
        // meant to re-justify this guard rather than slip past it.
        check_contains(v.reason, "IF YOU BUILD WITH CMAKE",
                       "no build key: the refusal must label route 1 with its reader -- the "
                       "reader who builds this test through CMake");
        check_contains(v.reason, "IF YOU COMPILE THIS TEST BY HAND",
                       "no build key: the refusal must label route 2 with its reader -- the "
                       "reader who hand-compiles, and who therefore never runs CMake");
        check(v.target.empty(), "a refused view must not carry a target");
        check(!v.usable(), "a refused view must not be usable (fail closed)");
    }

    // (c) THE ACKNOWLEDGEMENT. Absent, and WRONG -- the two are different branches of one clause
    // and both must land in the SAME refusal value, because both mean "no acknowledgement".
    {
        const VendorView absent = vview(VendorClass::Nvidia, true, "gfx906", nullptr);
        check(absent.failed() && absent.refusal == VendorRefusal::MissingAck,
              "no acknowledgement: must be refused as MissingAck");
        const VendorView wrong = vview(VendorClass::Nvidia, true, "gfx906", "yes");
        check(wrong.failed() && wrong.refusal == VendorRefusal::MissingAck,
              "a wrong acknowledgement must be refused as MissingAck");
    }

    // ⭐ THE SECOND-PHRASE PROPERTY, which is what stops one environment value selecting two axes.
    {
        const VendorView crossed = vview(VendorClass::Nvidia, true, "gfx906", kArchAck.c_str());
        check(crossed.failed() && crossed.refusal == VendorRefusal::MissingAck,
              "the ARCH-SIM acknowledgement must NOT satisfy the vendor axis -- one environment "
              "value must not be able to select two axes");
        const ArchView arch =
            arch_view_for_device_impl(120, true, "70", std::string(kVendorSimAckPhrase).c_str());
        check(arch.failed(),
              "and the reverse: the VENDOR acknowledgement must not satisfy arch-sim");
    }

    // (b) THE VALUE. Four distinct wrong shapes, four DISTINCT refusal values -- because one
    // "bad vendor" clause over four situations is a false attribution.
    {
        const VendorView unlisted = vview(VendorClass::Nvidia, true, "gfx999", kAck.c_str());
        check(unlisted.failed() && unlisted.refusal == VendorRefusal::NotInVendorLadder,
              "a gfx-shaped value with no kAmdLadder row must be NotInVendorLadder");
        check_contains(unlisted.reason, "gfx906",
                       "the NotInVendorLadder refusal must list the rows this tree HAS -- a "
                       "refusal that does not say what is available is not an on-ramp");
        check_contains(unlisted.reason, "RE-PROBEABLE",
                       "and it must carry sim_rung's own stated ground, which is the rule being "
                       "inherited rather than a new one");

        const VendorView classname = vview(VendorClass::Nvidia, true, "amd", kAck.c_str());
        check(classname.failed() && classname.refusal == VendorRefusal::NotATarget,
              "a class NAME where a TARGET belongs must be NotATarget, not a guess");

        const VendorView intel = vview(VendorClass::Nvidia, true, "intel", kAck.c_str());
        check(intel.failed() && intel.refusal == VendorRefusal::NoTableForVendor,
              "intel is a vendor this tree NAMES and has NO TABLE for: NoTableForVendor, which is "
              "a DIFFERENT sentence from unknown-vendor");
        check_contains(intel.reason, "NO SOURCE OF TRUTH",
                       "the Intel refusal must say WHY there is nothing to simulate");

        const VendorView nothing = vview(VendorClass::Nvidia, true, "banana", kAck.c_str());
        check(nothing.failed() && nothing.refusal == VendorRefusal::UnknownVendor,
              "a value that names nothing must be UnknownVendor");
    }

    // EVERY REFUSAL VALUE IS REACHABLE BY PARAMETER, walked from the enum's own range -- so a
    // refusal value added without a path to it is a red test, not a silent hole.
    {
        std::vector<VendorRefusal> seen;
        const VendorView cases[] = {
            vview(VendorClass::Nvidia, false, "gfx906", kAck.c_str()),
            vview(VendorClass::Nvidia, true, "gfx906", nullptr),
            vview(VendorClass::Nvidia, true, "amd", kAck.c_str()),
            vview(VendorClass::Nvidia, true, "gfx999", kAck.c_str()),
            vview(VendorClass::Nvidia, true, "intel", kAck.c_str()),
            vview(VendorClass::Nvidia, true, "banana", kAck.c_str()),
            vview(VendorClass::Amd, true, "gfx906", kAck.c_str()),
        };
        for (const VendorView& v : cases) {
            check(v.failed() && !v.reason.empty(),
                  "every constructed case must be a real refusal WITH a reason: a refusal that "
                  "does not carry its reason cannot be acted on");
            seen.push_back(v.refusal);
        }
        for (std::size_t raw = 1; raw < kVendorRefusalCount; ++raw) {
            const VendorRefusal wanted = static_cast<VendorRefusal>(raw);
            bool found = false;
            for (const VendorRefusal got : seen) {
                if (got == wanted) { found = true; break; }
            }
            check(found, "refusal value '" + std::string(vendor_refusal_name(wanted)) +
                             "' is not reachable by any parameter in this test; a refusal value "
                             "with no input that produces it is a guard that can never fire");
        }
    }

    // NO REQUEST, NO CHANGE. `Disabled` is not a refusal, and it must be a ZERO-behaviour path.
    {
        const VendorView v = vview(VendorClass::Nvidia, true, nullptr, nullptr);
        check(v.status == SimStatus::Disabled && v.refusal == VendorRefusal::NotRequested,
              "an unset value must be Disabled/NotRequested, not a refusal");
        check(v.usable(), "a Disabled view is usable: the guard is off, not broken");
        check(v.effective == v.physical, "a Disabled view must answer for the physical class");
        check(v.reason.empty(), "a Disabled view carries no reason");
    }
}

// ===========================================================================
// V2 -- G2-V: LOUD, AND KEYED PER CLASS. **THE GAP-1 ASSERTION.**
//       A5a..A5e are dl/cufree's assertion set for cufree_class_announce_once().
//       A5c IS THE ONE A `static bool` DIES ON (mutation M1).
// ===========================================================================
void V2_loud_and_keyed() {
    // ⚠ ORDERING: this function must run before anything else announces a vendor class. It is
    // called first in main(), and it announces only Amd and Intel, which no other group reaches
    // through the production wrapper (every other group uses _impl, which never announces).
    const std::size_t before = ninfer::detail::announce_once_seen_count<VendorClass>();

    // A5a -- the first sighting of a class IS announced.
    check(ninfer::detail::announce_once_keyed(VendorClass::Amd),
          "A5a the first sighting of a class IS announced");
    // A5b -- a repeat sighting of the SAME class is not announced again.
    check(!ninfer::detail::announce_once_keyed(VendorClass::Amd),
          "A5b a repeat sighting of the SAME class is not announced again");
    // A5c -- ⭐ A SECOND, DIFFERENT CLASS IS ANNOUNCED. THIS IS M1'S DEATH.
    check(ninfer::detail::announce_once_keyed(VendorClass::Intel),
          "A5c a SECOND, DIFFERENT class IS announced -- a one-shot `bool` announces the first "
          "value it sees and then goes silent, which is what src/core/arch_sim.h:324 does and "
          "what tests/test_sim_no_support.cpp:568 pins AS THE SHAPE");
    // A5d -- the second class does not announce twice either.
    check(!ninfer::detail::announce_once_keyed(VendorClass::Intel),
          "A5d the second class does not announce twice either");
    // A5e -- the first class STAYS announced and is not re-announced. (The set can only ADD;
    // nothing can clear an entry -- which is why there is no un-announce call to test.)
    check(!ninfer::detail::announce_once_keyed(VendorClass::Amd),
          "A5e the first class stays announced and is not re-announced");
    check(ninfer::detail::announce_once_seen_count<VendorClass>() == before + 2,
          "A5f the set grew by exactly the 2 distinct classes seen, so the membership test is "
          "exact under repetition");

    // THE SECOND KEYSPACE, SAME PRIMITIVE. The refusal value is what keys the refusal line, so a
    // second, DIFFERENT refusal reason in one process is not silent either.
    const std::size_t refusals_before = ninfer::detail::announce_once_seen_count<VendorRefusal>();
    check(ninfer::detail::announce_once_keyed(VendorRefusal::NotInVendorLadder),
          "the refusal keyspace announces its first value");
    check(ninfer::detail::announce_once_keyed(VendorRefusal::NoTableForVendor),
          "and a SECOND, DIFFERENT refusal value is announced too -- the same property, one "
          "keyspace over, with no second mechanism");
    check(!ninfer::detail::announce_once_keyed(VendorRefusal::NoTableForVendor),
          "and it is not announced twice");
    check(ninfer::detail::announce_once_seen_count<VendorRefusal>() == refusals_before + 2,
          "the refusal keyspace grew by exactly 2");

    // ⭐ THE OBSERVABILITY SURFACE IS READ-ONLY, asserted rather than asserted-by-comment: the
    // count does not move when it is read.
    const std::size_t c1 = ninfer::detail::announce_once_seen_count<VendorClass>();
    const std::size_t c2 = ninfer::detail::announce_once_seen_count<VendorClass>();
    check(c1 == c2, "announce_once_seen_count must be read-only: reading it must not change it");
}

// ===========================================================================
// V3 -- G3-V: FAIL CLOSED, WITH A GATE THAT CAN ACTUALLY THROW. **THE GAP-3 ASSERTION.**
// ===========================================================================
void V3_fail_closed_gate() {
    const NumericFormat probe[] = {NumericFormat::NVFP4};
    const char* const kIdentity = "artifact-identity";

    // A REFUSED view must throw, for EVERY refusal reason, and the throw must name the reason.
    const VendorView refused[] = {
        vview(VendorClass::Nvidia, false, "gfx906", kAck.c_str()), // build key missing
        vview(VendorClass::Nvidia, true, "gfx906", nullptr),       // ack missing
        vview(VendorClass::Nvidia, true, "gfx906", "nope"),        // ack wrong
        vview(VendorClass::Nvidia, true, "gfx999", kAck.c_str()),  // not in the ladder
        vview(VendorClass::Nvidia, true, "intel", kAck.c_str()),   // no table for the vendor
        vview(VendorClass::Nvidia, true, "banana", kAck.c_str()),  // unknown vendor
        vview(VendorClass::Amd, true, "gfx906", kAck.c_str()),     // the no-op
    };
    for (const VendorView& v : refused) {
        bool threw = false;
        std::string message;
        try {
            require_vendor_formats_supported(v, probe, kIdentity);
        } catch (const std::invalid_argument& e) {
            threw = true;
            message = e.what();
        }
        check(threw, "a REFUSED vendor view must make the gate THROW, refusal '" +
                         std::string(vendor_refusal_name(v.refusal)) + "'");
        check_contains(message, "fail-closed", "the refusal must say it is fail-closed");
        // THE FAIL-CLOSED SENTENCE ITSELF, NOT THE PHRASE. MEASURED 2026-09-19 (dl/falsegreen):
        // the phrase "question it did not ask" occurs TWICE in this message -- in
        // vendor_refusal_reason's tail, and again in a SECOND refusal text at
        // src/core/vendor_sim.h:786. Deleting the tail sentence, alone and together with its own
        // final clause, left this suite GREEN both times; the only owner I could not remove was
        // the one at :786, which is a neighbouring refusal, not the statement under test.
        check_contains(message, "No answer is given for the real card's own vendor either",
                       "the refusal must carry vendor_refusal_reason's OWN fail-closed sentence -- "
                       "the bare phrase 'question it did not ask' is ALSO carried by a different "
                       "refusal text at src/core/vendor_sim.h:786, so it is not a guard here");
        check_contains(message, "question it did not ask",
                       "the refusal must say the real card's vendor was NOT used either, which is "
                       "the whole content of 'fail closed' here");
        check(message.find(vendor_refusal_name(v.refusal)) != std::string::npos,
              "the throw must NAME its refusal value, so a reader can branch on it");
        check_contains(message, "SIMULATED VENDOR",
                       "the throw must name the mechanism it came from");
    }

    // ⭐ THE RED CONTROL FOR "THE GATE ALWAYS THROWS": a DISABLED view must NOT throw. Without
    // this, "the gate throws" would be indistinguishable from a gate that is simply broken.
    {
        const VendorView disabled = vview(VendorClass::Nvidia, true, nullptr, nullptr);
        bool threw = false;
        try {
            require_vendor_formats_supported(disabled, probe, kIdentity);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(!threw,
              "RED CONTROL: a DISABLED view must NOT throw -- this gate has a non-throwing path, "
              "so the refusals above are about the VENDOR REQUEST and not about a broken gate");
    }

    // AND AN ACTIVE view must throw, with the AMD apparatus's own words in the message rather
    // than a generic "unsupported".
    {
        const VendorView active = vview(VendorClass::Nvidia, true, "gfx906", kAck.c_str());
        check(active.simulated(), "gfx906 on an NVIDIA box with both keys must be ACTIVE");
        check(active.effective == VendorClass::Amd, "and its effective class must be Amd");
        check(active.target == "gfx906", "and it must carry the ladder row's own target string");
        bool threw = false;
        std::string message;
        try {
            require_vendor_formats_supported(active, probe, kIdentity);
        } catch (const std::invalid_argument& e) {
            threw = true;
            message = e.what();
        }
        check(threw, "an ACTIVE vendor view must REFUSE a tensor-core format");
        check(message.find(std::string(ninfer::artifact::format_name(NumericFormat::NVFP4))) !=
                  std::string::npos,
              "the refusal must name the FORMAT it refused");
        check_contains(message, "gfx906", "the refusal must name the target");
        check_contains(message, "UNMEASURED",
                       "the refusal must say the SUPPORT is unmeasured -- the tree's rule that a "
                       "simulated verdict is refused AS SUPPORT, not as arithmetic");
        check_contains(message, "UNVERIFIED", "and that the ARITHMETIC is unverified");
        check_contains(message, "NOT a claim about any AMD card",
                       "and it must refuse the support reading in its own words");
        check_contains(message, "EXTERNAL-UNPROBED",
                       "and the refusal must carry the EXTERNAL-UNPROBED label, because every "
                       "vendor-side fact it quotes came from somewhere other than this box");
    }
}

// ===========================================================================
// V4 -- G4-V: NEVER EQUALS, NEVER WIDENS. **THE GAP-2 ASSERTION.**
// ===========================================================================
void V4_never_equals_never_widens() {
    // NEVER EQUALS. The no-op is refused, and refused for G4's own stated reason. This is the
    // only input that can reach the no-op path: `effective` is set to a real class ONLY by a
    // kAmdLadder target, so the equal case requires `physical == Amd`.
    {
        const VendorView noop = vview(VendorClass::Amd, true, "gfx906", kAck.c_str());
        check(noop.failed() && noop.refusal == VendorRefusal::NotADowngrade,
              "G4-V NEVER EQUALS: asking for the class the device already is must be REFUSED "
              "(deleting the equal-case block is M3)");
        check_contains(noop.reason, "NO-OP", "the no-op refusal must say it is a no-op");
        // THE RATIONALE CLAUSE, NOT THE WORD. "simulated" occurs TWICE in this refusal: in the
        // rationale and in the trailing clause. MEASURED 2026-09-19 (dl/falsegreen): deleting the
        // rationale ("a no-op that wears a \"simulated\" label can hide a genuine defect...")
        // left the check above GREEN.
        check_contains(noop.reason, "wears a \"simulated\" label",
                       "the no-op refusal must keep the LABEL-WEARING rationale, not merely the "
                       "word \"simulated\" (which the trailing clause also carries)");
        check_contains(noop.reason, "simulated",
                       "and must carry G4's own reason: a no-op wearing a simulated label can "
                       "hide a real defect of the real card");
        // and the UNKNOWN class is not the no-op: an honest declaration must still work, or the
        // refusal above would be indistinguishable from a broken entry point.
        const VendorView honest = vview(VendorClass::Unknown, true, "gfx906", kAck.c_str());
        check(honest.simulated(),
              "RED CONTROL: with an honest physical declaration the same target IS honoured, so "
              "the no-op refusal is about the equality and not about the target");
    }

    // ⭐ NEVER WIDENS, WALKED OVER THE FORMAT ENUM'S OWN RANGE -- not over the formats this test
    // happened to think of. `kFormatOrdinalCount` is arch_caps.h:608-609's own count, so an
    // enumerator appended to artifact::NumericFormat is walked automatically.
    const VendorView active = vview(VendorClass::Nvidia, true, "gfx906", kAck.c_str());
    check(active.simulated(), "precondition for the walk: the view is active");

    std::size_t walked = 0;
    std::size_t admitted = 0;
    for (std::size_t raw = 0; raw < kFormatOrdinalCount; ++raw) {
        const NumericFormat format = static_cast<NumericFormat>(raw);
        const NumericFormat one[]  = {format};
        ++walked;
        const VendorFormatAnswer answer = vendor_format_answer(active, format);
        check(answer.answered, "every format must be answerable under an active view");
        check(answer.simulated, "and every answer must be marked simulated");
        check(answer.target == "gfx906", "and every answer must name the target it answered for");
        if (answer.admitted) { ++admitted; }
        bool threw = false;
        try {
            require_vendor_formats_supported(active, one, "artifact-identity");
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "G4-V NEVER WIDENS: the gate must REFUSE format ordinal " +
                         std::to_string(raw) + " (" +
                         std::string(ninfer::artifact::format_name(format)) +
                         ") under an active non-NVIDIA view (setting `admitted` from the verdict "
                         "is M2)");
    }

    // THE COUNT IS CHECKED AGAINST THE THING IT IS SUPPOSED TO BE COUNTING -- the discipline
    // tests/test_arch_caps.cpp:729 states in the file's own words. A walk that visited nothing
    // would otherwise pass every assertion in the loop above.
    check(walked == kFormatOrdinalCount,
          "the format walk must visit every ordinal: walked " + std::to_string(walked) + " of " +
              std::to_string(kFormatOrdinalCount));
    check(walked > 0, "and the walk must not be empty");
    check(admitted == 0,
          "G4-V NEVER WIDENS: admitted(simulated) must be EMPTY -- this tree holds NO AMD kernel "
          "at all, so no format is admissible on any AMD target. Got " + std::to_string(admitted) +
              " admitted of " + std::to_string(walked));

    // ⭐ MONOTONICITY, STATED AS THE PROPERTY ITSELF: admitted(simulated) SUBSET-OF
    // admitted(physical). With the antecedent false it holds vacuously, which is exactly WHY the
    // count above is the real assertion and this is its statement.
    {
        bool subset = true;
        for (std::size_t raw = 0; raw < kFormatOrdinalCount; ++raw) {
            if (vendor_format_answer(active, static_cast<NumericFormat>(raw)).admitted) {
                subset = false;
            }
        }
        check(subset, "admitted(simulated) SUBSET-OF admitted(physical): no simulated admission "
                      "may exist that the physical path does not also make");
    }
}

// ===========================================================================
// V5 -- G5-V: THE UNVERIFIABLE INPUT FAILS CLOSED. Over EVERY kAmdLadder row.
// ===========================================================================
void V5_unverifiable_input_fails_closed() {
    // THE THEOREM: if the caller declares `physical == Amd`, every real kAmdLadder target
    // resolves to effective == Amd == physical, so G4-V NEVER EQUALS refuses it. A caller that
    // lies about the ONE input this header cannot check can therefore obtain NO AMD table answer.
    check(kAmdLadderSize == kAmdTargetCount,
          "kAmdLadder must still have exactly the 6 rows this walk names; a row added or removed "
          "makes this walk's subject a different table");
    for (std::size_t i = 0; i < kAmdTargetCount; ++i) {
        const char* target = kAmdTargets[i];
        // (1) the target really is a row of the table -- otherwise this loop would be asserting
        //     the theorem about a string the ladder does not contain, which proves nothing.
        check(amd_rung(target) != nullptr,
              std::string(target) + " must still be a row of kAmdLadder");
        // (2) the lie produces a REFUSAL, never an active view.
        const VendorView lie = vview(VendorClass::Amd, true, target, kAck.c_str());
        check(lie.failed() && lie.refusal == VendorRefusal::NotADowngrade,
              std::string("G5-V: declaring the box AMD and requesting ") + target +
                  " must be REFUSED, so a false `physical` cannot buy an AMD answer");
        check(!lie.simulated(), "G5-V: and the lie must not produce an ACTIVE view");
        check(lie.target.empty(), "G5-V: and no target may be recorded on the refused view");
        // THE ASSERTION THAT WOULD HAVE CAUGHT IT, added after the walk above FAILED on
        // 2026-09-19 and the failure was measured: a target WAS recorded on the refused view, and
        // because vendor_format_answer() asked for a table query whenever `target` was non-empty,
        // this refused view answered ALL 12 formats with the AMD table's own verdict word -- i.e.
        // the lie DID obtain table answers, the opposite of this header's G5-V sentence. The
        // recording is now unreachable on a refused view, and this walks the CONSEQUENCE, so the
        // hole cannot come back unnoticed.
        {
            std::size_t answered_through_the_lie = 0;
            std::size_t answered_and_silent      = 0;
            for (std::size_t raw = 0; raw < kFormatOrdinalCount; ++raw) {
                const VendorFormatAnswer a =
                    vendor_format_answer(lie, static_cast<NumericFormat>(raw));
                if (a.answered) { ++answered_through_the_lie; }
                if (a.answered && a.text.empty()) { ++answered_and_silent; }
            }
            check(answered_through_the_lie == 0,
                  "G5-V: a refused view must not yield a SINGLE table answer either -- answering "
                  "from the refused view is how a lie about `physical` obtained all " +
                      std::to_string(kFormatOrdinalCount) +
                      " AMD verdicts. Got " + std::to_string(answered_through_the_lie));
            check(answered_and_silent == 0,
                  "G5-V: and no answer may be BOTH made and silent -- an empty text on an answered "
                  "format reads as \"fine\"");
        }
        // (3) the honest declaration DOES work, so (2) is about the lie and not about a broken
        //     path -- the red control for this group.
        const VendorView honest = vview(VendorClass::Unknown, true, target, kAck.c_str());
        check(honest.simulated(),
              std::string("RED CONTROL: the honest declaration must be ACTIVE for ") + target +
                  " -- without this, (2) would be indistinguishable from a broken entry point");
        check(honest.target == target, "and it must carry the ladder row's own target string");
    }

    // THE BANNER, which is the only place the loud text is built.
    {
        const VendorView v = vview(VendorClass::Nvidia, true, "gfx90a", kAck.c_str());
        check(v.simulated() && v.physical != v.effective,
              "the active path must report a physical class DIFFERENT from the effective one, so "
              "no message can present the simulated class as the card's (ArchView::physical_sm's "
              "job, one axis over)");
        const std::string banner = vendor_banner(v);
        check_contains(banner, "SIMULATED VENDOR", "the banner must say SIMULATED");
        check_contains(banner, "nvidia", "the banner must name the REAL class");
        check_contains(banner, "amd", "the banner must name the class the tables answer for");
        check_contains(banner, "gfx90a", "the banner must name the target");
        // THE BANNER'S OWN DENIAL, NOT THE WORD. "NOTHING is translated" occurs twice in the
        // rendered banner: in the first denial and in the honoured request's own reason, which
        // banner() appends. MEASURED 2026-09-19 (dl/falsegreen): deleting the first denial clause
        // left the check above GREEN. This needle can only match the denial.
        check_contains(banner, "NOTHING is translated: no instruction of this tree's",
                       "the banner's FIRST DENIAL must survive -- the phrase also occurs in the "
                       "reason appended below it, so the bare phrase is not a guard");
        check_contains(banner, "NOTHING is translated", "the banner's first denial");
        check_contains(banner, "NOTHING is executed for the target", "the banner's second denial");
        check_contains(banner, "NO SUPPORT CLAIM", "the support prohibition");
        check_contains(banner, "NO PERFORMANCE NUMBER", "the performance prohibition");
        // and the banner is EMPTY when nothing is simulated, so a caller can print it blind.
        check(vendor_banner(vview(VendorClass::Nvidia, true, nullptr, nullptr)).empty(),
              "no request, no banner");
        check(vendor_banner(vview(VendorClass::Amd, true, "gfx906", kAck.c_str())).empty(),
              "a REFUSED request renders the refusal text, not the banner");
    }
}

// ===========================================================================
// V6 -- THE FOUR GAPS, EACH WITH AN ASSERTION THAT CAN FAIL.
// ===========================================================================
void V6_the_four_gaps_have_teeth() {
    // -----------------------------------------------------------------------
    // GAP 1 -- G2 was a ONE-SHOT bool. Asserted behaviourally in V2 A5c. Here the MECHANISM
    // itself: the keyed set is what the vendor axis uses, and the shape it replaces is named.
    // -----------------------------------------------------------------------
    {
        // A `static bool` announced-once would make this FALSE only if this process had already
        // announced Amd through the wrapper -- it has not (V2 announced it through the primitive,
        // and V2 ran first). So `false` here is the keyed set answering, not a bare bool.
        check(!ninfer::detail::announce_once_keyed(VendorClass::Amd),
              "GAP 1: the vendor axis answers from the KEYED ADD-ONLY SET, so Amd is already "
              "announced and stays announced. A one-shot bool would also return true here once -- "
              "which is why the discriminating assertion is V2 A5c and not this line");
        check(ninfer::detail::announce_once_seen_count<VendorClass>() >= 2,
              "GAP 1: the set holds at least the 2 distinct classes V2 announced, so the "
              "membership test is over a set and not over one remembered value");
    }

    // -----------------------------------------------------------------------
    // GAP 2 -- G4 HAD NO VENDOR COUNTERPART. Asserted in V4. Here: the counterpart is REACHABLE
    // from a real input on the axis that has the ordering problem, i.e. it is not a dead branch.
    // -----------------------------------------------------------------------
    {
        const VendorView noop = vview(VendorClass::Amd, true, "gfx90a", kAck.c_str());
        check(noop.failed() && noop.refusal == VendorRefusal::NotADowngrade,
              "GAP 2: G4's vendor counterpart is REACHABLE -- NEVER EQUALS fires on a real input, "
              "so it is a guard and not a comment");
        // and the widening half is reachable-from-nothing by construction, which IS the property:
        const VendorView active = vview(VendorClass::Nvidia, true, "gfx90a", kAck.c_str());
        std::size_t any_admitted = 0;
        for (std::size_t raw = 0; raw < kFormatOrdinalCount; ++raw) {
            if (vendor_format_answer(active, static_cast<NumericFormat>(raw)).admitted) {
                ++any_admitted;
            }
        }
        check(any_admitted == 0,
              "GAP 2: NEVER WIDENS is the reason the widening branch in "
              "require_vendor_formats_supported is unreachable rather than untested");
    }

    // -----------------------------------------------------------------------
    // GAP 3 -- THE AMD SIDE COULD NOT FAIL CLOSED. V3 asserts the gate throws. HERE: the STRING
    // FORM'S HOLE IS MEASURED, so the reason the gate exists is a NUMBER and not a claim.
    // -----------------------------------------------------------------------
    {
        const VendorView active = vview(VendorClass::Nvidia, true, "gfx906", kAck.c_str());
        std::size_t renders_empty = 0;
        std::size_t verdict_not_refusal = 0;
        std::size_t walked = 0;
        for (std::size_t raw = 0; raw < kFormatOrdinalCount; ++raw) {
            const NumericFormat format = static_cast<NumericFormat>(raw);
            ++walked;
            if (render_amd_format_refusal("gfx906", format).empty()) { ++renders_empty; }
            if (amd_format_verdict("gfx906", format) != AmdFormatVerdict::NoKernelInTree) {
                ++verdict_not_refusal;
            }
        }
        std::printf("  GAP 3: render_amd_format_refusal(\"gfx906\", ...) is EMPTY for %zu of %zu "
                    "formats; amd_format_verdict is not NoKernelInTree for %zu. A caller that "
                    "tests only emptiness gets SILENCE for all %zu, and silence reads as "
                    "\"fine\"\n",
                    renders_empty, walked, verdict_not_refusal, renders_empty);
        check(walked == kFormatOrdinalCount, "GAP 3: the walk must be complete");
        check(walked > 0, "GAP 3: and not empty");
        check(renders_empty >= verdict_not_refusal,
              "GAP 3: every format the AMD table declines to refuse is one the string form is "
              "silent about -- that containment IS the hole");
        check(renders_empty > 0,
              "GAP 3: the string form must be EMPTY for at least one format, or there is no hole "
              "for the gate to close and the gate's justification is unmeasured");

        // and the gate closes it for ALL of them, INCLUDING the silent ones -- which is the whole
        // assertion: the formats the AMD text says nothing about are still refused, by name.
        std::size_t gate_refused = 0;
        for (std::size_t raw = 0; raw < kFormatOrdinalCount; ++raw) {
            const NumericFormat format = static_cast<NumericFormat>(raw);
            const NumericFormat one[]  = {format};
            try {
                require_vendor_formats_supported(active, one, "artifact-identity");
            } catch (const std::invalid_argument&) {
                ++gate_refused;
            }
        }
        check(gate_refused == kFormatOrdinalCount,
              "GAP 3: the gate must refuse EVERY format, including the " +
                  std::to_string(renders_empty) +
                  " the string form is silent about. Refused " + std::to_string(gate_refused) +
                  " of " + std::to_string(kFormatOrdinalCount));

        // AND THE SILENCE IS MADE TO FAIL RATHER THAN PRINT, which is the half of GAP 3 the printf
        // above cannot supply. THE DISPOSITION, ESTABLISHED BY READING THE OWNING FILE, AND IT
        // DECIDES WHERE THIS FIX GOES: the empty string is NOT a bug in
        // src/core/arch_caps.h:2240 -- that function documents it as deliberate ("Empty for every
        // other verdict, so a caller may print it unconditionally in the same shape
        // render_fallback_notice() uses"), so the four formats are a DELIBERATE GAP left visible by
        // a file this landing does not own. What is NOT acceptable is that the gap reached a caller
        // of THIS header as an empty string: a hole that is known and does not fail is the shape
        // the owner's rule forbids ("a polite refusal is not acceptable; if it genuinely cannot be
        // done, it must refuse LOUDLY"). So the answer this header hands out is filled where the
        // table is silent -- and it is asserted HERE, so reverting the fill turns this red.
        std::size_t silent_through = 0; // formats where the arch table's own render is EMPTY
        std::size_t still_silent   = 0; // ... and this header's answer is empty anyway
        std::size_t not_named      = 0; // ... and the answer does not name the verdict word
        std::size_t loud           = 0; // answered formats whose text is NOT empty
        std::size_t answered       = 0;
        for (std::size_t raw = 0; raw < kFormatOrdinalCount; ++raw) {
            const NumericFormat format = static_cast<NumericFormat>(raw);
            const bool table_is_silent = render_amd_format_refusal("gfx906", format).empty();
            const VendorFormatAnswer a = vendor_format_answer(active, format);
            if (a.answered) { ++answered; }
            if (!a.text.empty()) { ++loud; }
            if (table_is_silent) {
                ++silent_through;
                if (a.text.empty()) { ++still_silent; }
                if (a.text.find(std::string(amd_format_verdict_name(a.verdict))) ==
                    std::string::npos) {
                    ++not_named;
                }
            }
        }
        check(silent_through == renders_empty && silent_through > 0,
              "GAP 3: the table's own silence must be non-vacuous and identical in both walks -- "
              "otherwise the assertions below prove nothing. Got " +
                  std::to_string(silent_through) + " against " + std::to_string(renders_empty));
        check(answered == kFormatOrdinalCount,
              "GAP 3: every format must be answered under an ACTIVE view, or the walk below is "
              "smaller than it claims");
        check(still_silent == 0,
              "GAP 3: NO format may be answered with an EMPTY text -- silence reads as "
              "\"fine\". " + std::to_string(still_silent) + " of " +
                  std::to_string(kFormatOrdinalCount) + " said nothing");
        check(loud == kFormatOrdinalCount,
              "GAP 3: every format must produce a LOUD answer under an active view. Loud " +
                  std::to_string(loud) + " of " + std::to_string(kFormatOrdinalCount));
        check(not_named == 0,
              "GAP 3: where the table is silent, the answer must NAME the verdict word the tree "
              "gave, so the caller learns the specific blocker rather than a generic sentence. "
              "Unnamed " + std::to_string(not_named) + " of " + std::to_string(silent_through));
    }

    // -----------------------------------------------------------------------
    // GAP 4 -- `sim_vendor` WOULD BE RECORDABLE without `sim_rung`'s parse-time refusal.
    // [READ] TWO-WAY CONTROL: the field is asserted ABSENT from the row schema, and the rule that
    // IS there is asserted PRESENT, so a broken read cannot report a false clean.
    // -----------------------------------------------------------------------
    {
        const std::string probe_h = read_file(source_dir() + "/src/core/format_probe.h");
        if (probe_h.empty()) {
            check(false, "GAP 4: src/core/format_probe.h could not be read from " + source_dir() +
                             " -- the read is the assertion, so a failed read is a FAILURE and "
                             "not a skip");
        } else {
            // (a) THE FIELD IS ABSENT. The day it is added, this line goes red -- and the
            //     parse-time refusal it would then NEED is named here rather than re-derived:
            //     fold it into the ONE predicate FormatSupportEntry::is_simulated()
            //     (format_probe.h:510-512) and refuse it at PARSE for EVERY value, on
            //     sim_rung's own stated ground (:894-915): a simulated verdict must be
            //     RE-PROBEABLE on real hardware, and no card on this machine can re-probe any
            //     AMD target. That is strictly stronger than "must be in a list" -- it needs no
            //     new list, it cannot go stale, and it cannot be satisfied by adding a row.
            check(probe_h.find("sim_vendor") == std::string::npos,
                  "GAP 4: format_probe.h must have NO `sim_vendor` key. It is the recordability "
                  "path, and a field whose every value must be refused has no reason to exist");
            // (b) THE RULE THAT IS THERE, asserted present -- the red control for (a).
            check(probe_h.find("claims supported=yes while marked simulated=yes") !=
                      std::string::npos,
                  "GAP 4 RED CONTROL: the existing parse-time rule must still be present, so "
                  "(a) is a real absence and not a failed read");
            check(probe_h.find("is_simulated") != std::string::npos,
                  "GAP 4: and the ONE predicate must still be there, because a predicate that "
                  "catches one spelling and misses the other is how a simulated verdict gets "
                  "admitted as measured");
        }
    }
}

// ===========================================================================
// V7 -- THE PROHIBITIONS AND THE REFUSALS, ASSERTED WHERE A READER LOOKS.
// ===========================================================================
void V7_prohibitions_and_refusals() {
    // The three-way identity separation on the ACTIVE path: physical, effective, target.
    {
        const VendorView v = vview(VendorClass::Nvidia, true, "gfx942", kAck.c_str());
        check(v.physical == VendorClass::Nvidia, "physical is what the device really is");
        check(v.effective == VendorClass::Amd, "effective is what the tables answer for");
        check(v.target == "gfx942", "target is the ladder key");
        check(v.physical != v.effective, "the two classes must never be conflated in one field");
    }

    // ⭐ THE MEMBER THAT MUST NOT EXIST: there is no numeric capability on this view, because a
    // number is what a caller RAISES (G4-V(ii)). Asserted BY INTERFACE, not by grepping for a
    // field name: the whole vendor axis is a closed 8-bit enum, and no arithmetic exists on it.
    {
        check(!std::is_arithmetic_v<VendorClass>,
              "G4-V(ii): VendorClass must not be arithmetic -- there must be no number to raise");
        check(std::is_same_v<std::underlying_type_t<VendorClass>, std::uint8_t>,
              "the vendor axis is a small closed enum, which is what lets it be an announce key");
        check(std::is_enum_v<VendorRefusal>,
              "and the refusal keyspace is a closed enum too, for the same reason");
        check(!std::is_arithmetic_v<VendorRefusal>, "and it is not a number either");
    }

    // THE TWO DIFFERENT PROHIBITIONS, NAMED AS TWO. Neither may be traded for the other.
    {
        const VendorView v = vview(VendorClass::Nvidia, true, "gfx1100", kAck.c_str());
        const std::string banner = vendor_banner(v);
        check(banner.find("NO SUPPORT CLAIM") != std::string::npos &&
                  banner.find("NO PERFORMANCE NUMBER") != std::string::npos,
              "the banner must carry BOTH prohibitions as separate sentences, so a reader cannot "
              "conclude that satisfying one satisfies the other");
        check_contains(banner, "NOTHING is translated",
                       "the vendor axis adds a THIRD, narrower prohibition: nothing was translated");
    }

    // THE EXTERNAL-UNPROBED LABEL SURVIVES, and this mechanism CANNOT move it.
    {
        std::size_t rows = 0;
        std::size_t probed = 0;
        for (const PtxFamilyAmdStatus& status : kPtxFamilyAmdStatus) {
            ++rows;
            if (status.measured_in_this_tree) { ++probed; }
        }
        check(rows > 0, "the EXTERNAL-UNPROBED table must not be empty");
        check(probed == 0,
              "every kPtxFamilyAmdStatus row must stay measured_in_this_tree == false: this "
              "mechanism cannot move one, because no amdgcn compiler and no AMD silicon exist on "
              "this box -- a fact about the box, not a policy");
        std::printf("  V7: kPtxFamilyAmdStatus %zu rows, %zu probed in this tree\n", rows, probed);
    }

    // BOTH AXES REFUSE AN INVENTED VALUE, and both say the same thing about it. This is the one
    // place the vendor axis is asserted to AGREE with the landed arch axis rather than to differ.
    {
        const VendorView vendor = vview(VendorClass::Nvidia, true, "gfx999", kAck.c_str());
        const ArchView arch = arch_view_for_device_impl(120, true, "95", kArchAck.c_str());
        check(vendor.failed() && arch.failed(),
              "both axes must refuse a value their table does not contain");
        check_contains(vendor.reason, "inventing",
                       "the vendor refusal must say inventing is what is being refused");
        check_contains(arch.reason, "inventing a",
                       "and so must the arch refusal -- same discipline, two tables");
    }
}

} // namespace

int main() {
    std::printf("=== test_vendor_sim ===\n");
    std::printf("vendor sim built in: %s\n", kVendorSimBuiltIn ? "yes" : "no");
    std::printf("arch sim built in  : %s\n", kArchSimBuiltIn ? "yes" : "no");
    std::printf("kAmdLadder rows    : %zu\n", kAmdLadderSize);
    std::printf("kFormatOrdinalCount: %zu\n", kFormatOrdinalCount);

    // HERMETIC: the entry-point tests must not see the ambient environment, because a result that
    // depends on the operator's shell is not a result. Every test above goes through
    // _impl or through parameters, so this makes the binary independent of where it is run.
    ::unsetenv("NINFER_SIM_VENDOR");
    ::unsetenv("NINFER_SIM_VENDOR_ACK");

    V2_loud_and_keyed(); // FIRST: it is the only group that announces anything
    V1_impossible_by_accident();
    V3_fail_closed_gate();
    V4_never_equals_never_widens();
    V5_unverifiable_input_fails_closed();
    V6_the_four_gaps_have_teeth();
    V7_prohibitions_and_refusals();

    std::printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
