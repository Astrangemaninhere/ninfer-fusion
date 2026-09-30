// COLD-SLOT VALIDITY: the host test for spec/cold_slot_validity.h.
//
// Build it out of tree, with neither more than std and -I <repo>/src:
//
//   g++ -std=c++20 -O2 -Wall -Wextra -I <repo>/src spec/cold_slot_validity_test.cpp -o t
//   ./t
//
// It needs no GPU, no model and no engine, because the predicate is a pure function over
// the flag values -- that is the whole reason the predicate is a separate header.
//
// WHAT IT IS FOR, stated as the two MUST-RED arms the landing entry owes:
//
//   ① THE GATE IS LOAD-BEARING. A page whose flags are zero must be REFUSED by name under
//      the shipped policy, and must NOT be refused when the policy is turned off. If the
//      off-arm also refuses, the refusal is coming from somewhere else and the predicate is
//      inert -- this test says `PREDICATE_INERT` by name and fails, rather than reporting a
//      pass it did not earn.
//   ② THE PREDICATE GOES RED ON A KNOWN-BAD PAGE. A flag buffer with exactly one zero must
//      produce `valid == false` with `k_zero == 1`, and the refusal must carry the layer.
//      One zero is the smallest possible failure, so a predicate that only fires on an
//      all-zero plane cannot pass this arm.
//
// And two arms that exist to fail LOUDLY if someone simplifies the header:
//
//   A. the careless edit `return policy == RefusePage;` (dropping the `checked()` guard)
//      must be caught: a layer with NO validity tensor must NOT be refused. The MUTANT arm
//      below recomputes the predicate the careless way and asserts it DISAGREES with the
//      shipped one on that input -- a mutant that agrees would mean this arm has no teeth.
//   B. the "previous occupant" limit, EXECUTED rather than described: flags that are all
//      non-zero but belong to a DIFFERENT page cannot be detected by this predicate,
//      because the flags are not keyed to a page. The arm asserts the predicate returns
//      valid for that input. That is not a defect in the predicate; it is the boundary of
//      what a flag-only gate can prove, and this test is where the boundary is written down
//      instead of being left to prose.
//
// Numbers and the shapes they are printed in are the tree's own: a non-zero 32-bit flag
// word means the codec committed that head-plane's stream inside its fixed budget.

#include "spec/cold_slot_validity.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using ninfer::spec::coldvalid::ColdSlotValidityPolicy;
using ninfer::spec::coldvalid::ColdSlotValidityVerdict;

int failures = 0;
int arms     = 0;

void check(bool condition, const std::string& what) {
    ++arms;
    if (condition) {
        std::printf("  ok    %s\n", what.c_str());
    } else {
        std::printf("  FAIL  %s\n", what.c_str());
        ++failures;
    }
}

[[nodiscard]] std::vector<std::int32_t> flags(std::size_t n, std::int32_t value) {
    return std::vector<std::int32_t>(n, value);
}

} // namespace

int main() {
    using namespace ninfer::spec::coldvalid;

    const std::span<const std::int32_t> no_flags{};
    // A realistic plane width: the flag tensor is [kv_heads, 2, pages], and the shipped
    // stack's head count is a small even number. The predicate must not care which.
    constexpr std::size_t kHeads = 4;

    std::printf("=== 0. the header's own compile-time facts ===\n");
    check(kColdSlotValidityPolicy != ColdSlotValidityPolicy::Unnamed,
          "the shipped policy is named (not Unnamed) -- the header's own static_assert is live");
    check(std::string(kColdSlotInvalidRefusalName) == "refused-cold-slot-invalid",
          "the refusal's name is the tree's own token 'refused-cold-slot-invalid'");
    check(std::string(cold_slot_validity_policy_name(ColdSlotValidityPolicy::RefusePage)) ==
              "refuse-page",
          "the policy's name is spec'd");

    std::printf("=== 1. GREEN: a valid page is not refused ===\n");
    {
        const std::vector<std::int32_t> k = flags(kHeads, 1);
        const std::vector<std::int32_t> v = flags(kHeads, 1);
        const ColdSlotValidityVerdict verdict = cold_slot_validity_of(k, v);
        check(verdict.valid, "all flags non-zero => valid");
        check(verdict.k_zero == 0 && verdict.v_zero == 0, "zero counts are 0/0");
        check(verdict.checked(), "the verdict reports that it did check flags");
        check(!cold_slot_validity_refuses(verdict, ColdSlotValidityPolicy::RefusePage),
              "GREEN arm: RefusePage does NOT refuse a valid page");
    }

    std::printf("=== 2. MUST-RED ②: one zero on the K plane goes red, and names the layer ===\n");
    {
        std::vector<std::int32_t> k = flags(kHeads, 1);
        k[2] = 0;  // exactly ONE head-plane overflowed its stream budget
        const std::vector<std::int32_t> v = flags(kHeads, 1);
        const ColdSlotValidityVerdict verdict =
            cold_slot_validity_of(k, v, /*layer=*/7U, /*layer_index=*/2U);
        check(!verdict.valid, "MUST-RED ②: a single zero flag => valid == false");
        check(verdict.k_zero == 1U, "MUST-RED ②: k_zero == 1 (the smallest possible failure)");
        check(verdict.v_zero == 0U, "MUST-RED ②: v_zero == 0 (the zero is attributable to K)");
        check(verdict.k_flags == kHeads && verdict.v_flags == kHeads,
              "MUST-RED ②: the denominators are the TESTED widths, not an assumed count");
        check(verdict.layer == 7U && verdict.layer_index == 2U,
              "MUST-RED ②: the verdict names the layer and its position");
        check(cold_slot_validity_refuses(verdict, ColdSlotValidityPolicy::RefusePage),
              "MUST-RED ②: RefusePage refuses it");
    }

    std::printf("=== 3. MUST-RED ②': a zero on the V plane goes red too ===\n");
    {
        const std::vector<std::int32_t> k = flags(kHeads, 1);
        std::vector<std::int32_t> v       = flags(kHeads, 1);
        v[0] = 0;
        const ColdSlotValidityVerdict verdict = cold_slot_validity_of(k, v, 11U, 1U);
        check(!verdict.valid && verdict.v_zero == 1U,
              "MUST-RED ②': a V-plane zero is caught as well as a K-plane one");
        check(cold_slot_validity_refuses(verdict, ColdSlotValidityPolicy::RefusePage),
              "MUST-RED ②': RefusePage refuses it");
    }

    std::printf("=== 4. MUST-RED ①: THE GATE IS LOAD-BEARING (off-arm must NOT refuse) ===\n");
    {
        std::vector<std::int32_t> k = flags(kHeads, 1);
        k[1] = 0;
        const std::vector<std::int32_t> v = flags(kHeads, 1);
        const ColdSlotValidityVerdict verdict =
            cold_slot_validity_of(k, v, /*layer=*/6U, /*layer_index=*/0U);

        const bool refused_shipped =
            cold_slot_validity_refuses(verdict, kColdSlotValidityPolicy);
        check(refused_shipped,
              "MUST-RED ①: the SHIPPED policy refuses a page the read path cannot vouch for");

        const bool refused_off =
            cold_slot_validity_refuses(verdict, ColdSlotValidityPolicy::ReportOnly);
        if (refused_off) {
            std::printf("  FAIL  MUST-RED ①: the OFF arm also refuses. The refusal is NOT "
                        "coming from this predicate -- PREDICATE_INERT\n");
            ++failures;
            ++arms;
        } else {
            ++arms;
            std::printf("  ok    MUST-RED ①: with the policy off the SAME page is NOT "
                        "refused => the predicate is what refuses it (not inert)\n");
        }
        // The same statement as an assertion, so the arm cannot pass by printing.
        check(!refused_off && refused_shipped,
              "MUST-RED ①: refused(RefusePage) && !refused(ReportOnly) on one verdict");
    }

    std::printf("=== 5. MUTANT A: the careless edit is caught (drop the checked() guard) ===\n");
    {
        // The edit a simplifier would make:
        //     return policy == ColdSlotValidityPolicy::RefusePage;
        const std::vector<std::int32_t> empty{};
        const ColdSlotValidityVerdict unchecked =
            cold_slot_validity_of(no_flags, no_flags, /*layer=*/3U, /*layer_index=*/1U);
        check(!unchecked.checked(), "MUTANT A: a layer with no flags reports NOT checked");
        check(unchecked.valid, "MUTANT A: all_of over an empty range is TRUE (the trap)");

        const auto careless = [](const ColdSlotValidityVerdict&,
                                 ColdSlotValidityPolicy p) noexcept {
            return p == ColdSlotValidityPolicy::RefusePage;
        };
        const bool careless_says_refuse =
            careless(unchecked, ColdSlotValidityPolicy::RefusePage);
        const bool shipped_says_refuse =
            cold_slot_validity_refuses(unchecked, ColdSlotValidityPolicy::RefusePage);

        check(careless_says_refuse != shipped_says_refuse,
              "MUTANT A: the careless predicate DISAGREES with the shipped one here => this "
              "arm has teeth");
        check(!shipped_says_refuse,
              "MUTANT A: the shipped predicate does NOT refuse a layer with no validity "
              "tensor (refusing it would refuse every page of a stack whose flags are not "
              "written)");
    }

    std::printf("=== 6. THE BOUNDARY, EXECUTED: a foreign-but-nonzero flag set is NOT caught ===\n");
    {
        // After a restore the sentinel names a STAGING slot. Nothing on the read path writes
        // that slot's flags, so they are the previous occupant's -- and if that occupant was
        // valid, every flag is non-zero and this predicate CANNOT tell. Executed here so the
        // limit is a fact of the test and not a sentence in a README.
        const std::vector<std::int32_t> previous_occupants_flags = flags(kHeads, 1);
        const ColdSlotValidityVerdict verdict =
            cold_slot_validity_of(previous_occupants_flags, previous_occupants_flags,
                                  /*layer=*/0U, /*layer_index=*/0U);
        check(verdict.valid,
              "BOUNDARY: non-zero flags from ANY page read as valid -- the predicate is "
              "flag-only and the flags are not keyed to a page");
        check(!cold_slot_validity_refuses(verdict, ColdSlotValidityPolicy::RefusePage),
              "BOUNDARY: so this arm is NOT refused. Closing it needs the flags to travel "
              "WITH the payload (a format change), which is not what this entry does");
    }

    std::printf("=== 7. THE LINE: one spec'd spelling of the refusal ===\n");
    {
        std::vector<std::int32_t> k = flags(kHeads, 1);
        k[0] = 0;
        k[3] = 0;
        const std::vector<std::int32_t> v = flags(kHeads, 1);
        const ColdSlotValidityVerdict verdict = cold_slot_validity_of(k, v, 6U, 2U);
        const std::string line = cold_slot_invalid_refusal_line(156U, verdict, "restore");
        std::printf("  line: %s\n", line.c_str());
        check(line.find("refusal=refused-cold-slot-invalid") == 0,
              "the line opens with the refusal's name");
        check(line.find("page=156") != std::string::npos, "the line names the page");
        check(line.find("layer=6") != std::string::npos, "the line names the layer");
        check(line.find("k_zero=2/4") != std::string::npos,
              "the line carries k_zero over its OWN denominator");
        check(line.find("v_zero=0/4") != std::string::npos,
              "the line carries v_zero over its OWN denominator");
        check(line.find("policy=refuse-page") != std::string::npos,
              "the line names the policy in force");
    }

    std::printf("\n=================================================\n");
    std::printf("arms=%d failures=%d\n", arms, failures);
    if (failures != 0) {
        std::printf("RESULT=FAIL\n");
        return 1;
    }
    std::printf("ALL ASSERTIONS PASSED (and both MUST-RED arms went red as required)\n");
    std::printf("RESULT=PASS\n");
    return 0;
}
