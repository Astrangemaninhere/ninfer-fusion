// test_kv_storage_capability_pin -- SITE A over the WHOLE KvCacheStorage domain, and the five
// INTEGRATE4 tiers that had no ruler on SITE A anywhere in this tree.
//
// WHY THIS FILE EXISTS, AND WHY IT IS NOT test_kv_storage_coverage.cpp AGAIN.
//
// tests/test_kv_storage_coverage.cpp covers SITE A (src/core/device_capabilities.h
// requirements_for_kv_storage) INDIRECTLY. Its check 3 compares
//
//   requirements_for_kv_storage(code).all_of == requirements_for_dtype(kv_dtype_for_storage(code)).all_of
//
// and that comparison NEEDS A DType. src/product/kv_storage_dtype.h refuses five tiers BY NAME --
// Fp8KeyNvfp4Value, RotatedInt8KeyInt4ValueGroup64, RotatedInt4KeyInt4ValueGroup64, RK4V4E8,
// RK2V4E8 (codes 10..14) -- so those five have no DType and check 3 SKIPS them. The file says so
// in its own output: "site A compared 7 of 15 enumerators (the rest are refused by name, so they
// have no DType to compare against)". MEASURED: the 7 are exactly codes 0..6.
//
// The consequence was MEASURED rather than argued. Four candidate rulers exist in this tree, and
// NONE of them can see SITE A for codes 10..14:
//   * tests/test_kv_storage_coverage.cpp   -- check 3 skips them (above);
//   * tests/test_kv_operator_name.cpp      -- names them and pins their SITE C policy as literals,
//                                             but contains 0 occurrences of
//                                             requirements_for_kv_storage;
//   * src/ops/kv/e8_width_contract_test.cpp -- DOES call requirements_for_kv_storage and DOES pin
//                                             a family equality ("rk3v4/rk2v4 demand exactly the
//                                             rk4v4 codec's capabilities"), but only for codes
//                                             6/8/9 -- never 10..14;
//   * tests/test_device_capabilities.cpp   -- calls it for codes 0,1,3,4,5,6,7 only.
// With the five arms set back to the bare kernel-image baseline -- a silent under-report of exactly
// the kind SITE A exists to refuse -- all three of the tree's capability bars stayed GREEN.
//
// WHAT THIS FILE PINS, AND WHY THERE ARE TWO HALVES.
//
//   HALF 1  THE DOMAIN INVARIANT, BOTH SIDES, over the whole uint8 domain:
//             (a) NAMED BUT REFUSED == 0     a code the enumeration names must get an answer;
//             (b) UNNAMED BUT SILENT == 0    a code NO enumerator names must not get one.
//           Neither half is the invariant on its own: (a) alone is satisfied by refusing every
//           code, and (b) alone by answering every code. This tree has been in BOTH wrong states --
//           the pre-image read 0 named-refused / 241 unnamed-silent, and the P3-only image (the
//           trailing return turned into a throw, the five arms not yet added) read 5 / 0. Both are
//           red here; only the state where BOTH numbers are 0 is green.
//
//   HALF 2  THE DEMAND PIN, PER TIER, AS A LITERAL. HALF 1 is about refuse-versus-answer; it is
//           blind to a tier that IS answered with the WRONG demand. MEASURED: with the five arms
//           set to the baseline, HALF 1 reads 0 / 0 -- GREEN -- while HALF 2 goes red on all five
//           rows. Both halves are therefore load-bearing and neither is decoration.
//
// WHAT THIS FILE DELIBERATELY DOES NOT RESTATE.
//   * The SITE C policy for codes 10..14 (they refuse) is pinned as literals by
//     tests/test_kv_operator_name.cpp's own `policy` table. Restating it here would be a second
//     spelling of one question, which is the defect that file's census check exists to refuse.
//   * The enumerator count and the token table are pinned by tests/test_kv_storage_coverage.cpp
//     (checks 0b/4) and product/kv_storage_dtype.h's static_asserts. The count is DERIVED here
//     from the named last enumerator, the same way, and not written as a second literal.
//
// A code NO enumerator names is reachable in production: KvCacheStorage is a std::uint8_t fed from
// option text, so a cast or an untrusted input can produce one. That is why HALF 1 (b) sweeps to
// 0xFF rather than stopping at the count.
//
// HOST-ONLY: no CUDA, no device, no model. Header-only host code, registered exactly like
// ninfer_kv_storage_coverage_test -- the default ninfer_core link is what puts the CUDA include
// root on the line for core/device_capabilities.h. No NEEDS_SOURCE_DIR: this file reads no source.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

#include "core/device_capabilities.h"
#include "product/kv_storage_dtype.h"

using namespace ninfer;
using namespace ninfer::product;

namespace {

int g_ok = 0;
int g_total = 0;

void check(bool pass, const std::string& what) {
    ++g_total;
    if (pass) {
        ++g_ok;
    }
    std::printf("%s  %s\n", pass ? "PASS" : "FAIL", what.c_str());
}

// Derived from the named LAST enumerator, exactly as test_kv_storage_coverage.cpp derives it, and
// for the same reason: a literal count is the thing that goes stale. Sweeping to 0xFF covers the
// codes above it independently.
inline constexpr KvCacheStorage kLastEnumerator = KvCacheStorage::RK2V4E8;
inline constexpr std::size_t kEnumeratorCount = static_cast<std::size_t>(kLastEnumerator) + 1;
inline constexpr std::size_t kDomainSize = 0x100;

[[nodiscard]] std::string code_label(std::size_t code) {
    return "code " + std::to_string(code) + " (\"" +
           std::string(kv_storage_name(static_cast<KvCacheStorage>(code))) + "\")";
}

[[nodiscard]] std::string capability_set_names(CapabilitySet set) {
    std::string out;
    for (std::size_t i = 0; i < kDeviceCapabilityCount; ++i) {
        const auto capability = static_cast<DeviceCapability>(i);
        if (set_contains(set, capability)) {
            if (!out.empty()) {
                out += " | ";
            }
            out += std::string(capability_probe_spec(capability).name);
        }
    }
    return out.empty() ? std::string("(none)") : out;
}

[[nodiscard]] std::string needs_names(CapabilityNeeds needs) {
    std::string out = "all_of = {" + capability_set_names(needs.all_of) + "}";
    out += ", any_of = {" + capability_set_names(needs.any_of) + "}";
    return out;
}

// Do the two sides of the domain split cleanly? A code is NAMED iff kv_storage_name() answers
// something other than "?" -- that is SITE B's contract, and it is the ONLY reading of "?" this
// file accepts (test_kv_storage_coverage.cpp check 1 pins it the same way).
struct DomainReading {
    int named_answered = 0;
    int named_refused = 0;
    int unnamed_refused = 0;
    int unnamed_silent = 0;
};

[[nodiscard]] DomainReading read_domain() {
    DomainReading reading;
    for (std::size_t code = 0; code < kDomainSize; ++code) {
        const auto storage = static_cast<KvCacheStorage>(code);
        const bool named = kv_storage_name(storage) != "?";
        bool threw = false;
        try {
            (void)requirements_for_kv_storage(storage);
        } catch (const std::exception&) {
            threw = true;
        }
        if (named) {
            threw ? ++reading.named_refused : ++reading.named_answered;
        } else {
            threw ? ++reading.unnamed_refused : ++reading.unnamed_silent;
        }
    }
    return reading;
}

std::string offender_list(bool want_named, bool want_refused) {
    std::string out;
    int shown = 0;
    int total = 0;
    for (std::size_t code = 0; code < kDomainSize; ++code) {
        const auto storage = static_cast<KvCacheStorage>(code);
        const bool named = kv_storage_name(storage) != "?";
        bool threw = false;
        try {
            (void)requirements_for_kv_storage(storage);
        } catch (const std::exception&) {
            threw = true;
        }
        if (named != want_named || threw != want_refused) {
            continue;
        }
        ++total;
        if (shown < 8) {
            out += (shown == 0 ? "" : ", ") + std::to_string(code);
            ++shown;
        }
    }
    if (total > shown) {
        out += ", ... (" + std::to_string(total) + " total)";
    }
    return out.empty() ? std::string("(none)") : out;
}

// ---------------------------------------------------------------------------
// HALF 1 -- the domain invariant, BOTH sides.
void check_domain_invariant() {
    std::printf("\n--- HALF 1. DOMAIN INVARIANT: named => answered, unnamed => refused ---\n");
    std::printf("     derived enumerator count = %zu (from KvCacheStorage::RK2V4E8), uint8 domain = %zu\n",
                kEnumeratorCount, kDomainSize);

    const DomainReading reading = read_domain();
    std::printf("     site A domain readings: named_answered=%d named_refused=%d "
                "unnamed_refused=%d unnamed_silent=%d\n",
                reading.named_answered, reading.named_refused, reading.unnamed_refused,
                reading.unnamed_silent);

    // SIDE (a). Every code the enumeration names must get an ANSWER. A throw here is an
    // over-refusal: the tier exists, the table says it does not. The wrong state this catches is
    // the P3-only image, which read named_refused=5 on codes 10..14.
    check(reading.named_refused == 0,
          "SIDE (a) NAMED BUT REFUSED == 0: requirements_for_kv_storage() must answer for EVERY "
          "code the enumeration names (found " + std::to_string(reading.named_refused) +
              "; codes: " + offender_list(true, true) + ")");

    // SIDE (b). Every code NO enumerator names must be REFUSED. Answering one is a silent POSITIVE
    // claim about a tier nobody recognises -- the pre-image state, which read unnamed_silent=241.
    check(reading.unnamed_silent == 0,
          "SIDE (b) UNNAMED BUT SILENT == 0: requirements_for_kv_storage() must REFUSE every code "
          "no enumerator names, rather than claim the kernel-image baseline covers it (found " +
              std::to_string(reading.unnamed_silent) + "; codes: " + offender_list(false, false) +
              ")");

    // ANTI-VACUITY, both sides. Without these, a sweep that touched nothing would satisfy both
    // SIDES above by having counted nothing.
    check(reading.named_answered + reading.named_refused == static_cast<int>(kEnumeratorCount),
          "the NAMED side of the sweep is the full width (" +
              std::to_string(reading.named_answered + reading.named_refused) + " of " +
              std::to_string(kEnumeratorCount) + ")");
    check(reading.unnamed_refused + reading.unnamed_silent ==
              static_cast<int>(kDomainSize - kEnumeratorCount),
          "the UNNAMED side of the sweep is the full width (" +
              std::to_string(reading.unnamed_refused + reading.unnamed_silent) + " of " +
              std::to_string(kDomainSize - kEnumeratorCount) + ")");
}

// ---------------------------------------------------------------------------
// HALF 2 -- the demand pin, per tier, AS A LITERAL.
//
// WHY A LITERAL AND NOT A DERIVATION. A derivation is what HALF 1 already is. The point of a pin is
// that the tree must MOVE IT ON PURPOSE: a change to a demand has to be edited here, in the same
// landing, and the edit is the record. The derivations below are the SECOND opinion -- they check
// the literals against the tree's own neighbouring rows, so a literal that is merely self-consistent
// cannot pass.
//
// A REFUSAL IS READ AS A VALUE HERE, NOT AS AN ABORT. A state that refuses one of these five is a
// state this file must be able to REPORT. The first version of this file called
// requirements_for_kv_storage() unguarded inside the pin loop, so on an image that refuses a NAMED
// tier it died with an uncaught std::invalid_argument (rc=134, zero FAIL lines) instead of printing
// FAIL -- a red for the wrong reason, and this project's own record already names that shape as an
// instrument defect (vendp3 REPORT 7.4, the W3 mutant). Every read below goes through read_demand().
struct DemandRead {
    bool answered = false;
    CapabilityNeeds needs{};
    std::string message;
};

[[nodiscard]] DemandRead read_demand(KvCacheStorage storage) {
    DemandRead read;
    try {
        read.needs = requirements_for_kv_storage(storage);
        read.answered = true;
    } catch (const std::exception& e) {
        read.message = e.what();
    }
    return read;
}

void check_demand_pins() {
    std::printf("\n--- HALF 2. DEMAND PIN: the five tiers' demands, as literals ---\n");

    const DemandRead int8_read = read_demand(KvCacheStorage::Int8Group64);
    const DemandRead fp8_row_read = read_demand(KvCacheStorage::Fp8E4M3Row256);
    const DemandRead nvfp4_read = read_demand(KvCacheStorage::Nvfp4Group16);

    const CapabilitySet kInt8Floor =
        kKernelImageBaseline | capability_bit(DeviceCapability::Int8Mma);

    const CapabilitySet kFp8KeyNvfp4ValueDemand =
        kKernelImageBaseline | capability_bit(DeviceCapability::Fp8MmaKindF8f6f4) |
        capability_bit(DeviceCapability::Int8Mma) | capability_bit(DeviceCapability::Nvfp4MmaBlockScale) |
        capability_bit(DeviceCapability::Iso4eKvCodec);

    struct TierPin {
        KvCacheStorage storage;
        CapabilitySet all_of;
        const char* why;
    };
    static const TierPin kPins[] = {
        // A K/V CODEC PAIR, not one tier: its K plane is Fp8E4M3Row256's and its V plane is
        // Nvfp4Group16's (core/paged_kv_storage.h:92-99), so the demand is the CONJUNCTION.
        {KvCacheStorage::Fp8KeyNvfp4Value, kFp8KeyNvfp4ValueDemand,
         "the fp8 row's demand AND the nvfp4 row's demand (a K/V codec pair)"},
        // The four fork storages: the int8 family's floor, by the tree's OWN classifier
        // (core/paged_kv_storage.h kv_storage_is_int8_family() returns true for exactly these
        // plus Int8Group64).
        {KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, kInt8Floor, "int8 family floor"},
        {KvCacheStorage::RotatedInt4KeyInt4ValueGroup64, kInt8Floor, "int8 family floor"},
        {KvCacheStorage::RK4V4E8, kInt8Floor, "int8 family floor"},
        {KvCacheStorage::RK2V4E8, kInt8Floor, "int8 family floor"},
    };

    int pinned = 0;
    for (const TierPin& pin : kPins) {
        const std::string label = code_label(static_cast<std::size_t>(pin.storage));
        const DemandRead read = read_demand(pin.storage);
        if (!read.answered) {
            check(false,
                  "PIN " + label + " is NAMED yet SITE A REFUSES it -- a named tier must get an "
                  "answer, not a throw (HALF 1 side (a) fires for the same reason); got \"" +
                      read.message + "\"");
            continue;
        }
        ++pinned;
        check(read.needs == CapabilityNeeds{pin.all_of, kNoCapabilities},
              "PIN " + label + " demands exactly {" + capability_set_names(pin.all_of) +
                  "} with no disjunction -- " + std::string(pin.why) + "; got " +
                  needs_names(read.needs));
        // The five were the only tiers with NO disjunction pin anywhere: codes 1/4/3/5/6/7 are
        // walked for this by tests/test_device_capabilities.cpp, and the bf16 slot is the one
        // storage allowed to carry one.
        check(!needs_has_disjunction(read.needs),
              "PIN " + label + " carries no disjunction (only the bf16 slot may)");
    }

    check(std::size(kPins) == 5,
          "the pin table covers all FIVE new tiers (it is the set the SITE A ruler was missing)");
    check(pinned == static_cast<int>(std::size(kPins)),
          "every tier in the pin table was actually READ and COMPARED (" + std::to_string(pinned) +
              " of " + std::to_string(std::size(kPins)) +
              "); a loop that skipped a row on a throw would make the pins above vacuous");

    // DERIVATION 1 -- code 10 is the union of its two halves. This is the second opinion that makes
    // the literal above more than self-consistency: the tree's own neighbouring rows must add up to
    // it. Fp8E4M3Row256 contributes {baseline, fp8, int8}; Nvfp4Group16 contributes
    // {baseline, nvfp4, iso4e}; the union is exactly kFp8KeyNvfp4ValueDemand.
    if (!fp8_row_read.answered || !nvfp4_read.answered) {
        check(false, "DERIVATION code 10: its two halves (codes 2 and 3) could not be read at SITE A");
    } else {
        const DemandRead got = read_demand(KvCacheStorage::Fp8KeyNvfp4Value);
        const CapabilitySet derived = fp8_row_read.needs.all_of | nvfp4_read.needs.all_of |
                                      capability_bit(DeviceCapability::Int8Mma);
        if (!got.answered) {
            check(false, "DERIVATION code 10 could not be read at SITE A (it is a NAMED tier)");
        } else {
            check(got.needs.all_of == derived,
                  "DERIVATION codes 10's demand == all_of(Fp8E4M3Row256) | all_of(Nvfp4Group16) | "
                  "Int8Mma (derived {" + capability_set_names(derived) + "}); got {" +
                      capability_set_names(got.needs.all_of) + "}");
        }
    }

    // DERIVATION 2 -- codes 11..14 sit exactly on Int8Group64's row, which is the same rule
    // src/ops/kv/e8_width_contract_test.cpp already pins one family over.
    for (const TierPin& pin : kPins) {
        if (pin.all_of != kInt8Floor) {
            continue;
        }
        const std::string label = code_label(static_cast<std::size_t>(pin.storage));
        const DemandRead read = read_demand(pin.storage);
        if (!read.answered || !int8_read.answered) {
            check(false, "DERIVATION " + label + " could not be read at SITE A");
            continue;
        }
        check(read.needs == int8_read.needs,
              "DERIVATION " + label + " demands exactly what Int8Group64 demands (" +
                  needs_names(int8_read.needs) + ")");
    }

    // THE LOAD-BEARING NEGATIVE. dl/e8dev measured that mapping RK4V4E8/RK2V4E8 onto the e8 lattice
    // tiers is "a wrong codec under a right name" (core/paged_kv_storage.h sets .e8_lattice = true
    // on RK4V4E8, and src/product/kv_storage_dtype.h refuses that aliasing one table over). Demanding
    // the lattice bit here would BE that aliasing, one table up -- so it is pinned as ABSENT, which
    // is the half a totals-only check can never state.
    for (const KvCacheStorage storage : {KvCacheStorage::RK4V4E8, KvCacheStorage::RK2V4E8}) {
        const std::string label = code_label(static_cast<std::size_t>(storage));
        const DemandRead read = read_demand(storage);
        if (!read.answered) {
            check(false, "NEGATIVE " + label + " could not be read at SITE A");
            continue;
        }
        check(!needs_contains(read.needs, DeviceCapability::E8KvLattice),
              "NEGATIVE " + label +
                  " does NOT demand the E8 lattice rotation: that would alias a root-cylinder "
                  "specimen onto the lattice's name (" + needs_names(read.needs) + ")");
    }
}

} // namespace

int main() {
    std::printf("== kv storage capability pin (SITE A over the whole KvCacheStorage domain) ==\n");
    std::printf("== the domain invariant is TWO-SIDED; the demand pin is PER TIER; each can go red "
                "without the other ==\n");
    check_domain_invariant();
    check_demand_pins();
    std::printf("\n== %d/%d ==\n", g_ok, g_total);
    return g_ok == g_total ? 0 : 1;
}
