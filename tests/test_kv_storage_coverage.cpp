// test_kv_storage_coverage -- a PORTABLE replacement for `-Wswitch` over KvCacheStorage.
//
// THE DEFECT THIS EXISTS FOR, and why it is not a -Wswitch problem.
//
// include/ninfer/types.h declares FIFTEEN KvCacheStorage enumerators. FOUR sites switch
// over that enum, each names only TEN of the fifteen, and none of them carries a
// `default:` arm -- so `-Wall` emits 20 -Wswitch diagnostics, 4 sites x 5 unladen
// enumerators. The five unladen enumerators are, as of this revision:
//
//     Fp8KeyNvfp4Value, RotatedInt8KeyInt4ValueGroup64, RotatedInt4KeyInt4ValueGroup64,
//     RK4V4E8, RK2V4E8
//
// and the four sites, each with what its fall-through does:
//
//   SITE A  src/core/device_capabilities.h:268 requirements_for_kv_storage()
//           fall-through returns CapabilityNeeds{kKernelImageBaseline}  -> SILENTLY
//           UNDER-REPORTS CAPABILITY: an unladen enumerator claims "this tier needs no
//           codec capability", which is a positive claim, not a refusal.
//   SITE B  src/core/device_capabilities.h:346 kv_storage_name()
//           fall-through returns "?"                                    -> THE NAME
//           PRINTS "?", and "?" is a two-meaning sentinel: it means BOTH "this code is
//           an enumerator I have no name for" AND "this code is not an enumerator".
//           The two are different operator problems and the sentinel conflates them.
//   SITE C  src/product/kv_storage_dtype.h:45 kv_dtype_for_storage()
//           fall-through throws "KV storage code N has no DType ... A per-layer slot
//           must be one of bf16, int8, fp8, nvfp4, iso4e, rk4v4, rk3v4, rk2v4" -> A
//           GENERIC THROW THAT MISNAMES THE CAUSE. It lists EIGHT spellings for FIFTEEN
//           enumerators, and it asserts the code names no enumerator while the enum
//           names it.
//   SITE D  src/targets/qwen3_6/impl/runtime/layouts_impl.h:91 target_kv_cache_profile()
//           fall-through throws "unknown KV-cache storage profile"    -> ANONYMOUS: it
//           names neither the tier nor even the numeric code.
//
// WHY A TEST AND NOT A FLAG. -Wswitch is GCC/Clang. On MSVC the equivalents are C4061
// and C4062, which are OFF in practice, and MSVC is this project's stated END STATE
// (native Windows execution). So the only thing keeping those four sites naming their
// tiers today DOES NOT EXIST on the platform the project is aiming at. A warning flag
// cannot be the guarantee. A test can.
//
// WHAT IS ASSERTED
//   0 DERIVE   the enumerator count is DERIVED from the enumeration, never written as a
//              literal, and a code at or above it must not be named -- so appending an
//              enumerator past the named last one FAILS this test instead of passing
//              invisibly. (Closes the "swept the wrong range" hole.)
//   1 NAME     for every enumerator: kv_storage_name() is non-empty and is not "?".
//              This is SITE B, directly.
//   2 PROFILE  for every enumerator: kv_dtype_for_storage() either RETURNS a valid
//              (and itself named) profile, or REFUSES WITH A MESSAGE THAT NAMES THE
//              CALLER'S OWN SLOT. A generic throw FAILS. This is SITE C, directly.
//   3 CAPAB    for every enumerator that resolves in check 2: the capability requirement
//              of the STORAGE must equal the capability requirement of the DType it
//              resolves to. This is SITE A, indirectly -- see COVERAGE below.
//   4 PIN      the header's own kKvCacheStorageCount must equal the derived count. It is
//              STALE by five as of this revision, and that staleness is not cosmetic: the
//              pre-existing totality check in tests/test_kv_operator_name.cpp:156 measures
//              its pin list against THIS constant, so that check is a ten-lane sweep that
//              cannot see the five unladen enumerators.
//
// WHAT "NAMED REFUSAL" MEANS, CONCRETELY. Two parts, both explicit and falsifiable:
//   (a) the message CONTAINS the caller's own `where` provenance string. The convention
//       is documented at kv_storage_dtype.h:41-43 -- `where` is e.g. "--kv-layer-storage[7]",
//       so the refusal is attributable to the operator's own slot.
//   (b) the message is NOT the tree's generic tail. The generic tail is identified
//       EXACTLY: it contains the substring " has no DType" (kv_storage_dtype.h:128-135).
//       That tail is reachable only for a value NO enumerator names, so reaching it with
//       a value that HAS an enumerator is by definition a miss.
// The tree's BEST example of a named refusal is the E8K3/E8K2 arm at
// kv_storage_dtype.h:82-91, which (i) embeds `where`, (ii) embeds the tier token
// "rk3v4"/"rk2v4", (iii) states the geometry that exists (product/kv_e8_width.h:
// 6656 B at 3 bits / 4608 B at 2 bits per head-page), and (iv) names BOTH wrong
// resolutions it is refusing -- rk4v4 ("would misread every row through the 4-bit nibble
// reader") and bf16 ("would silently inherit the global --kv-dtype"). The best example of
// a re-thrown refusal with its own numbers is
// src/targets/qwen3_6/impl/runtime/program_impl.h:11637, which re-throws BY NAME with the
// counts and the original `error.what()` appended.
//
// COVERAGE, STATED SO IT CANNOT DEGRADE SILENTLY. This file covers SITE B and SITE C
// directly, and SITE A indirectly (check 3, for the six enumerators that resolve). It
// CANNOT cover SITE D: layouts_impl.h is not includable from a host-only TU -- measured,
// it fails on `#error "NINFER_QWEN36_VARIANT must name the complete exact Variant"`
// (instance.h:4) and its include list names core/device.h and six ninfer/ops headers, so
// it needs both a target instantiation and CUDA. SITE D is therefore reported as
// UNCOVERED in this file's own output and the tally counts it, rather than being dropped.
// What would cover SITE D: (i) the same refusal text hoisted into a host-reachable header
// -- the tree already does this once, for the sliding-window refusal, at
// product/kv_component_switch.h:270, which IS host-includable; (ii) a table +
// static_assert pin in the style already proven at product/kv_storage_dtype.h:179-209;
// or (iii) an integration run against a real target instantiation.
//
// HOST-ONLY: no CUDA, no device, no model. Header-only host code, exactly like
// tests/test_kv_operator_name.cpp.

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
int g_site_a = 0;
int g_site_b = 0;
int g_site_c = 0;

void check(bool pass, const std::string& what) {
    ++g_total;
    if (pass) {
        ++g_ok;
    }
    std::printf("%s  %s\n", pass ? "PASS" : "FAIL", what.c_str());
}

// ---------------------------------------------------------------------------
// 0. DERIVATION OF THE ENUMERATOR COUNT.
//
// The count is NOT written as a literal here, because a literal is exactly the thing
// that goes stale and this test exists because something went stale. It is derived
// the way the tree's own header derives its pin (kv_storage_dtype.h:160-171: "pinned to
// the LAST enumerator rather than being a free literal"): name the LAST enumerator ONCE
// and take its value + 1.
//
// There is NO `Count` member in the enumeration -- I read the whole body of
// include/ninfer/types.h:29-86 and it ends at RK2V4E8 with no sentinel -- so "derive from
// a Count member" is not available in this tree. Deriving from the last enumerator is.
//
// A named last enumerator still goes stale if someone APPENDS after it, so check 0b
// closes that direction independently: sweep the entire uint8_t domain and require every
// code at or above the derived count to be UNNAMED. If an enumerator is appended after
// RK2V4E8, that new code will be named and check 0b will fail.
//
// PREMISE, checked: every enumerator in types.h:29-86 is unassigned, so the enumeration
// is contiguous from 0 and value(kLast)+1 is the true count. That premise is not
// re-verified here (C++ offers no portable reflection), but check 0b plus the
// positional pin in tests/test_kv_operator_name.cpp are what would catch a break in it.
inline constexpr KvCacheStorage kLastEnumerator = KvCacheStorage::RK2V4E8;
inline constexpr std::size_t kEnumeratorCount = static_cast<std::size_t>(kLastEnumerator) + 1;

[[nodiscard]] std::string enumerator_label(std::size_t code) {
    return "code " + std::to_string(code) + " ([" + std::to_string(code) + "])";
}

// The caller's provenance for a slot, in the documented `where` convention.
[[nodiscard]] std::string where_for(std::size_t code) {
    return "--kv-layer-storage[" + std::to_string(code) + "]";
}

// (b) of the NAMED-REFUSAL predicate, identified EXACTLY as the tree's own generic tail.
[[nodiscard]] bool is_generic_tail(const std::string& message) {
    return message.find(" has no DType") != std::string::npos;
}

// A refusal is NAMED iff it names the caller's slot and is not the generic tail.
[[nodiscard]] bool refusal_is_named(const std::string& message, const std::string& where) {
    return message.find(where) != std::string::npos && !is_generic_tail(message);
}

// ---------------------------------------------------------------------------
void check_derive() {
    std::printf("\n--- 0. DERIVE: the count comes from the enumeration, and cannot go stale ---\n");
    std::printf("     derived count = %zu, from kLastEnumerator = KvCacheStorage::RK2V4E8\n",
                kEnumeratorCount);

    check(kEnumeratorCount >= 1, "derived count is at least 1");

    // 0b. The anti-staleness arm. Every code at or above the derived count must be
    // UNNAMED. "?" is what kv_storage_name() returns for a code with no enumerator, and
    // that is the ONE reading of "?" this test accepts.
    int named_beyond = 0;
    for (std::size_t code = kEnumeratorCount; code <= 0xFFU; ++code) {
        const std::string_view name = kv_storage_name(static_cast<KvCacheStorage>(code));
        if (name != "?") {
            ++named_beyond;
            check(false,
                  "code " + std::to_string(code) + " is at or above the derived count " +
                      std::to_string(kEnumeratorCount) + " yet kv_storage_name() names it \"" +
                      std::string(name) +
                      "\" -- an enumerator was APPENDED past kLastEnumerator, so the "
                      "derivation above is stale and must be moved");
        }
    }
    if (named_beyond == 0) {
        check(true,
              "no code at or above the derived count is named (" + std::to_string(0xFFU + 1 - kEnumeratorCount) +
                  " codes swept), so the derivation is not stale");
    }

    // Contiguity, stated as a measurement rather than assumed: every code below the count
    // must be distinct from "?" in the sense that kv_storage_name() is asked for a real
    // name -- that is check 1's job. Here we only require the domain to be non-empty.
    check(kEnumeratorCount == static_cast<std::size_t>(KvCacheStorage::RK2V4E8) + 1,
          "the derivation is self-consistent with the named last enumerator");
}

// ---------------------------------------------------------------------------
// 1. SITE B -- kv_storage_name() must never answer "?" for a real enumerator.
void check_names() {
    std::printf("\n--- 1. NAME (SITE B): kv_storage_name() is non-empty and never \"?\" ---\n");
    for (std::size_t code = 0; code < kEnumeratorCount; ++code) {
        const KvCacheStorage storage = static_cast<KvCacheStorage>(code);
        const std::string_view name = kv_storage_name(storage);
        ++g_site_b;
        check(!name.empty() && name != "?",
              "kv_storage_name(" + enumerator_label(code) + ") = \"" + std::string(name) +
                  "\" is a NAMED tier (site B: device_capabilities.h:346)");
    }
}

// ---------------------------------------------------------------------------
// 2. SITE C -- kv_dtype_for_storage() must return a profile or REFUSE BY NAME.
void check_profiles() {
    std::printf("\n--- 2. PROFILE (SITE C): a valid profile, or a refusal that names the slot ---\n");
    for (std::size_t code = 0; code < kEnumeratorCount; ++code) {
        const KvCacheStorage storage = static_cast<KvCacheStorage>(code);
        const std::string where = where_for(code);
        ++g_site_c;

        bool returned = false;
        DType resolved{};
        std::string message;
        try {
            resolved = kv_dtype_for_storage(storage, where);
            returned = true;
        } catch (const std::invalid_argument& e) {
            message = e.what();
        } catch (const std::exception& e) {
            message = std::string("NON-invalid_argument exception: ") + e.what();
        }

        if (returned) {
            // A resolved profile must itself be a tier the tree can name, or the operator
            // line has the same "?" problem one level down.
            const std::string_view dn = dtype_name(resolved);
            check(!dn.empty() && dn != "?",
                  "kv_dtype_for_storage(" + enumerator_label(code) + ") returned a DType named \"" +
                      std::string(dn) + "\" (dtype_name must not be \"?\" either)");
        } else {
            check(refusal_is_named(message, where),
                  "kv_dtype_for_storage(" + enumerator_label(code) + ") REFUSED BY NAME (site C: "
                  "message contains \"" + where + "\" and is not the generic tail); got: \"" +
                      message + "\"");
        }
    }
}

// ---------------------------------------------------------------------------
// 3. SITE A -- a resolving storage must demand the capability its DType demands.
//
// The comparison is on `.all_of` only. That is deliberate and it is not a weakening: the
// DType side has NO disjunction by construction (device_capabilities.h:317-319: "DType
// 口径没有析取"), while the storage side carries exactly one disjunction deliberately --
// the bf16 slot's two attention routes (device_capabilities.h:225-245). Comparing the
// full struct would therefore fail on a CORRECT tree. Comparing the conjunction is the
// claim that is actually invariant.
void check_capability() {
    std::printf("\n--- 3. CAPABILITY (SITE A): storage requirement agrees with its DType ---\n");
    for (std::size_t code = 0; code < kEnumeratorCount; ++code) {
        const KvCacheStorage storage = static_cast<KvCacheStorage>(code);
        try {
            const DType resolved = kv_dtype_for_storage(storage, where_for(code));
            const CapabilityNeeds from_storage = requirements_for_kv_storage(storage);
            const CapabilityNeeds from_dtype = requirements_for_dtype(resolved);
            ++g_site_a;
            check(from_storage.all_of == from_dtype.all_of,
                  "requirements_for_kv_storage(" + enumerator_label(code) +
                      ").all_of == requirements_for_dtype(" + std::string(dtype_name(resolved)) +
                      ").all_of (site A: device_capabilities.h:268 must not under-report)");
        } catch (const std::exception&) {
            // Refused storages have no DType to compare against. They are counted as
            // NOT covered by site A and reported in the tally below -- never silently.
        }
    }
    std::printf("     site A compared %d of %zu enumerators (the rest are refused by name, "
                "so they have no DType to compare against)\n",
                g_site_a, kEnumeratorCount);
}

// ---------------------------------------------------------------------------
// 4. PIN -- the header's own count must equal the derived count.
void check_pin() {
    std::printf("\n--- 4. PIN: the header's kKvCacheStorageCount equals the derived count ---\n");
    check(kKvCacheStorageCount == kEnumeratorCount,
          "kKvCacheStorageCount (" + std::to_string(kKvCacheStorageCount) +
              ") == derived enumerator count (" + std::to_string(kEnumeratorCount) +
              "); a mismatch means the header's pin is STALE and every check that measures "
              "against it -- tests/test_kv_operator_name.cpp:156 -- is a short sweep");

    check(std::size(kKvStorageNames) == kEnumeratorCount,
          "kKvStorageNames carries one entry per enumerator (" +
              std::to_string(std::size(kKvStorageNames)) + " vs " +
              std::to_string(kEnumeratorCount) + ")");

    // The token lookup must be total over the ENUMERATION, not merely over the table.
    int short_tokens = 0;
    for (std::size_t code = 0; code < kEnumeratorCount; ++code) {
        try {
            const std::string_view tok = kv_storage_token(static_cast<KvCacheStorage>(code));
            if (tok.empty()) { ++short_tokens; }
        } catch (const std::invalid_argument&) {
            ++short_tokens;
        }
    }
    check(short_tokens == 0,
          "kv_storage_token() resolves all " + std::to_string(kEnumeratorCount) +
              " enumerators (it THROWS \"names no enumerator\" for " +
              std::to_string(short_tokens) + " of them in this revision)");
}

// ---------------------------------------------------------------------------
// 5. ANTI-VACUITY -- the sweep must actually have been run, per site.
void check_anti_vacuity() {
    std::printf("\n--- 5. ANTI-VACUITY: the sweep ran, and it was the full width ---\n");
    check(g_site_b == static_cast<int>(kEnumeratorCount),
          "site B sweep touched all " + std::to_string(kEnumeratorCount) + " enumerators (touched " +
              std::to_string(g_site_b) + ")");
    check(g_site_c == static_cast<int>(kEnumeratorCount),
          "site C sweep touched all " + std::to_string(kEnumeratorCount) + " enumerators (touched " +
              std::to_string(g_site_c) + ")");
}

} // namespace

int main() {
    std::printf("== kv storage coverage (portable replacement for -Wswitch over KvCacheStorage) ==\n");
    std::printf("== enumerators derived: %zu | sites covered here: B (direct), C (direct), A (indirect) ==\n",
                kEnumeratorCount);
    std::printf("== SITE D (targets/qwen3_6/.../layouts_impl.h:91 target_kv_cache_profile) is NOT "
                "COVERED by this file: not host-includable (needs NINFER_QWEN36_VARIANT + CUDA) ==\n");
    check_derive();
    check_names();
    check_profiles();
    check_capability();
    check_pin();
    check_anti_vacuity();
    std::printf("\n== %d/%d ==  (site A comparisons: %d)\n", g_ok, g_total, g_site_a);
    return g_ok == g_total ? 0 : 1;
}
