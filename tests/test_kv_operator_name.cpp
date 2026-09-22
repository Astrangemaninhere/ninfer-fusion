// test_kv_operator_name -- ONE tier must have ONE name, and the name the OPERATOR
// reads must be the name the ENGINE holds.
//
// THE DEFECT THIS EXISTS FOR. product/kv_storage_dtype.h carries the canonical
// KvCacheStorage -> token table (kKvStorageNames, static_assert-pinned, exposed by
// kv_storage_token()). The CLI did NOT print it. apps/cli/main.cpp carried a second,
// hand-written switch that had drifted six spellings and lost two enumerators:
//
//   printed (stale)      canonical (what actually ran)
//   int8-group64         int8-g64
//   fp8-e4m3-row256      fp8-e4m3-r256
//   nvfp4-group16        nvfp4-g16
//   fp8-group16          fp8-g16
//   iso3-group16         iso4e-g16
//   e8-group64           rk4v4-g64
//   "unknown"            rk3v4-g64   (E8K3Group64 was not named at all)
//   "unknown"            rk2v4-g64   (E8K2Group64 was not named at all)
//
// That switch feeds exactly one line --
//     summary      kv cache dtype   <name>
// -- and that line is the line an operator reads to confirm WHICH TIER ACTUALLY RAN.
// A stale name there makes every measurement checked against it ambiguous. Nothing
// forced the two tables to agree and no test failed: a half-landed rename with
// nothing making it atomic.
//
// WHAT IS ASSERTED, and why each check can fail on its own:
//   1 PIN     the operator token equals the LITERAL canonical token, per enumerator.
//             A rename that lands on ONE side only is caught here.
//   2 DERIVE  the operator token equals kv_storage_token() of the same value, for
//             every enumerator AND for a value no enumerator names. A print site that
//             stops deriving and goes back to a switch is caught here.
//   3 TOTAL   the pin list covers every enumerator, and a NEW enumerator makes this
//             test fail rather than silently pass: it cannot be outgrown.
//   4 CENSUS  the operator-visible CLI sources are read AS TEXT: the stale spellings
//             must not appear in them, and the derivation must. Checks 1-3 cannot see
//             the defect's ACTUAL shape -- a re-introduced second copy -- because a
//             second copy that agrees today disagrees tomorrow.
//
// Host-only, no CUDA, no device, no model: header-only host code.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "product/kv_storage_dtype.h"

// A test that loses its census silently is the very shape it exists to refuse, so the
// source directory is required, not optional. CMake passes it (NEEDS_SOURCE_DIR).
#ifndef NINFER_SOURCE_DIR
#error "test_kv_operator_name needs NINFER_SOURCE_DIR: register it with ninfer_add_test(... NEEDS_SOURCE_DIR ...)"
#endif

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

// The canonical vocabulary, written ONCE here as literals. That duplication is the
// point: the literals are the second opinion, so a change to the header's table alone
// (or to the operator token alone) cannot pass unnoticed. This is the only place in
// the tree where a tier token is allowed to be spelled twice on purpose, and check 3
// keeps it total, so the duplication cannot silently go stale.
struct Pin {
    KvCacheStorage storage;
    std::string_view token;
    std::string_view enumerator;
};

const std::vector<Pin>& pins() {
    static const std::vector<Pin> p = {
        {KvCacheStorage::BFloat16, "bf16", "BFloat16"},
        {KvCacheStorage::Int8Group64, "int8-g64", "Int8Group64"},
        {KvCacheStorage::Fp8E4M3Row256, "fp8-e4m3-r256", "Fp8E4M3Row256"},
        {KvCacheStorage::Nvfp4Group16, "nvfp4-g16", "Nvfp4Group16"},
        {KvCacheStorage::Fp8Group16, "fp8-g16", "Fp8Group16"},
        {KvCacheStorage::Iso3Group16, "iso4e-g16", "Iso3Group16"},
        {KvCacheStorage::E8Group64, "rk4v4-g64", "E8Group64"},
        {KvCacheStorage::Dropped, "dropped", "Dropped"},
        {KvCacheStorage::E8K3Group64, "rk3v4-g64", "E8K3Group64"},
        {KvCacheStorage::E8K2Group64, "rk2v4-g64", "E8K2Group64"},
        // INTEGRATE4's five, at the enum's own indices. The token is the enumerator in kebab
        // case; where kebab-casing would collide with a TIER name the collision is broken by
        // the source's own E8 suffix (RK4V4E8 -> "rk4v4e8", one character from the tier
        // "rk4v4"). These five are the names PORTABLEPIN's claim is about: they were the five
        // codes kv_storage_token() threw on while the enum named them.
        {KvCacheStorage::Fp8KeyNvfp4Value, "fp8k-nvfp4v-g64", "Fp8KeyNvfp4Value"},
        {KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, "rot-i8k-i4v-g64",
         "RotatedInt8KeyInt4ValueGroup64"},
        {KvCacheStorage::RotatedInt4KeyInt4ValueGroup64, "rot-i4k-i4v-g64",
         "RotatedInt4KeyInt4ValueGroup64"},
        {KvCacheStorage::RK4V4E8, "rk4v4e8", "RK4V4E8"},
        {KvCacheStorage::RK2V4E8, "rk2v4e8", "RK2V4E8"},
    };
    return p;
}

std::string read_text(const std::string& path, bool& ok) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        ok = false;
        return {};
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    ok = true;
    return buffer.str();
}

void check_pin() {
    std::printf("\n--- 1. PIN: the operator token IS the literal canonical token ---\n");
    for (const Pin& p : pins()) {
        const std::string got = kv_operator_token(p.storage);
        check(got == p.token,
              "kv_operator_token(" + std::string(p.enumerator) + ") == \"" +
                  std::string(p.token) + "\" (got \"" + got + "\")");
    }
}

void check_derive() {
    std::printf("\n--- 2. DERIVE: the operator token is not a second table ---\n");
    for (const Pin& p : pins()) {
        const std::string_view canonical = kv_storage_token(p.storage);
        check(kv_operator_token(p.storage) == canonical,
              "kv_operator_token(" + std::string(p.enumerator) +
                  ") == kv_storage_token(same value)");
    }

    // A value no enumerator names: KvCacheStorage is a uint8_t fed from option text, so
    // a cast or untrusted input can produce one. The canonical lookup refuses it
    // loudly, and the operator line must inherit that refusal rather than print a
    // plausible-looking tier name.
    bool operator_threw = false;
    try {
        (void)kv_operator_token(static_cast<KvCacheStorage>(200));
    } catch (const std::invalid_argument&) {
        operator_threw = true;
    }
    check(operator_threw,
          "kv_operator_token(200) THROWS instead of printing a name for a value no enumerator has");

    bool canonical_threw = false;
    try {
        (void)kv_storage_token(static_cast<KvCacheStorage>(200));
    } catch (const std::invalid_argument&) {
        canonical_threw = true;
    }
    check(canonical_threw, "kv_storage_token(200) throws (the contract the operator line inherits)");
}

void check_total() {
    std::printf("\n--- 3. TOTAL: the pin list covers the enumeration ---\n");
    const std::vector<Pin>& p = pins();
    check(p.size() == kKvCacheStorageCount,
          "pin list has one entry per KvCacheStorage enumerator (" + std::to_string(p.size()) +
              " pins vs kKvCacheStorageCount " + std::to_string(kKvCacheStorageCount) +
              "); a NEW enumerator fails HERE instead of silently going unpinned");

    // Positional: every pin sits at its own enumerator's index. A list that is complete
    // but REORDERED names every value and mislabels all of them, which is the other
    // silent failure mode this file's header already warns about.
    for (std::size_t i = 0; i < p.size(); ++i) {
        check(static_cast<std::size_t>(p[i].storage) == i,
              "pin[" + std::to_string(i) + "] (" + std::string(p[i].enumerator) +
                  ") sits at its own enumerator index");
    }

    // The pins must also be the header's own table, entry for entry. Without this, a
    // rename that lands in the header AND in the operator token but nowhere in this
    // file would be caught by check 1 -- but a rename that lands in this file's pins and
    // in the header, while the operator token still derives, must be VISIBLE here too.
    for (const Pin& pin : p) {
        check(kKvStorageNames[static_cast<std::size_t>(pin.storage)] == pin.token,
              "header table entry for " + std::string(pin.enumerator) +
                  " is still \"" + std::string(pin.token) + "\"");
    }
}

void check_census() {
    std::printf("\n--- 4. CENSUS: the operator sources derive and do not re-spell ---\n");

    // The spellings that were WRONG, i.e. what the drifted copy printed. Two extra
    // phantom forms are named because apps/cli/options.cpp quoted them in prose while
    // describing the printed line, so an operator could type them and get a refusal.
    static const std::vector<std::string> stale = {
        "iso3-group16",  "e8-group64",     "e8k3-g64",       "e8k2-g64",
        "int8-group64",  "fp8-e4m3-row256", "nvfp4-group16", "fp8-group16",
        "rk4v4-group64", "iso4e-group16",
    };

    // These files print, describe or accept the tier names an operator reads. They must
    // derive. bench/ and serve/request_log.cpp are deliberately NOT here: their short
    // vocabulary is documented as separate by design
    // (docs/maintainer/kv-storage-names-and-switch-gates.md:83).
    static const std::vector<std::string> sources = {
        "/apps/cli/main.cpp",
        "/apps/cli/options.cpp",
    };

    for (const std::string& rel : sources) {
        bool ok = false;
        const std::string text = read_text(std::string(NINFER_SOURCE_DIR) + rel, ok);
        if (!ok) {
            check(false, "could not read " + rel + " under NINFER_SOURCE_DIR");
            continue;
        }
        for (const std::string& bad : stale) {
            check(text.find(bad) == std::string::npos,
                  rel + " contains no stale spelling \"" + bad + "\"");
        }
    }

    // The derivation itself. Without this, deleting the two stale literals and calling
    // kv_storage_token() would pass, but replacing them with a switch over CANONICAL
    // literals would also pass -- and that is a second copy again, just a correct one
    // today. The print site must delegate.
    bool ok = false;
    const std::string main_cpp = read_text(std::string(NINFER_SOURCE_DIR) + "/apps/cli/main.cpp", ok);
    if (ok) {
        check(main_cpp.find("kv_operator_token") != std::string::npos,
              "apps/cli/main.cpp CALLS kv_operator_token (the print site derives, not copies)");
        check(main_cpp.find("format_kv_cache") != std::string::npos,
              "apps/cli/main.cpp still routes the operator line through format_kv_cache");
    } else {
        check(false, "could not read /apps/cli/main.cpp for the derivation check");
    }
}

// ---------------------------------------------------------------------------
// 5. SWEEP: EVERY ENUMERATOR IS NAMED, AND EVERY TIER IS RESOLVED OR REFUSED BY NAME
//
// This is the portable replacement for the -Wswitch arms this tree cannot rely on. CMakeLists
// opens no warning flag at all, so a short switch here is not even a warning; and the flag
// that would fire is MSVC's C4061/C4062 on the one platform a warning cannot be checked on.
// A test is a language-level mechanism; a warning is not, and it is not portable.
//
// dl/warnsurface/E8_SWITCH_SITES.md measured FOUR live switch sites (`device_capabilities.h`
// :269 and :347, `kv_storage_dtype.h`:45, `layouts_impl.h`:92) each silently short by the same
// five enumerators -- 20 `-Wswitch` warnings that nothing in the build surfaces. Three of the
// four fall through to a *silent or misleading* answer. This check is the runtime half of
// that measurement: it asserts the property those five arms owed, over every enumerator,
// on the value the operator actually sees.
//
// It can go red three ways, and each was the reason it exists:
//   * an enumerator with no token        -> PROPERTY 1 (this WAS red: 5 codes, PORTABLEPIN)
//   * a tier that neither resolves nor refuses -> PROPERTY 2
//   * a refusal whose text does not name  -> PROPERTY 3 (the class kv_storage_dtype.h's own
//     the tier the operator asked for        `Dropped` comment at :92-108 was written for)
//
// The token minus its scale-group suffix: "rk3v4-g64" -> "rk3v4", "rk4v4e8" stays whole.
// The refusal arms name the tier by this stem, so PROPERTY 3 is stated on the stem rather
// than on the whole token -- a stricter predicate would go red on two arms whose text is
// correct, and a test that is red for being right is the failure this fleet keeps paying for.
std::string tier_stem(std::string_view token) {
    const std::size_t dash = token.rfind('-');
    if (dash != std::string_view::npos && dash + 2 < token.size() && token[dash + 1] == 'g' &&
        token[dash + 2] >= '0' && token[dash + 2] <= '9') {
        return std::string(token.substr(0, dash));
    }
    return std::string(token);
}

void check_sweep() {
    std::printf("\n--- 5. SWEEP: every enumerator is named and resolved-or-refused ---\n");

    // THE LAST ENUMERATOR, named literally. If the enum stops carrying it this file fails to
    // COMPILE, which is the strongest form of the same statement.
    constexpr std::size_t kLast = static_cast<std::size_t>(KvCacheStorage::RK2V4E8);

    for (std::size_t c = 0; c <= kLast; ++c) {
        const auto storage = static_cast<KvCacheStorage>(c);
        const std::string slot = "code " + std::to_string(c);

        // PROPERTY 1 -- NAMED. The enum names it, so the table must name it.
        bool named = true;
        std::string token;
        try {
            token = std::string(kv_storage_token(storage));
        } catch (const std::exception& e) {
            named = false;
            token = e.what();
        }
        check(named, slot + " has a token (enum names it; a throw here means the table is " +
                         "short of the enumeration -- got \"" + token + "\")");
        if (!named) { continue; }

        // PROPERTY 2/3 -- RESOLVED, or REFUSED WITH A MESSAGE THAT NAMES ITS OWN SLOT.
        // Every resolver arm in kv_storage_dtype.h writes the caller's `where` AND the tier.
        // The predicate is therefore: the message carries the caller's provenance, so the
        // operator can tell WHICH slot was refused and not merely that something was.
        const std::string where = "sweep[" + std::to_string(c) + "]";
        std::string outcome;
        try {
            outcome = "resolved to DType " +
                      std::to_string(static_cast<int>(kv_dtype_for_storage(storage, where)));
            check(true, slot + " (" + token + ") " + outcome);
        } catch (const std::exception& e) {
            const std::string msg = e.what();
            check(msg.find(where) != std::string::npos,
                  slot + " (" + token +
                      ") refuses with a message naming ITS OWN SLOT (the caller's provenance)");
            // The TIER predicate is on the token's FAMILY stem, not on the full token, and
            // that is measured rather than convenient: the rk3v4/rk2v4 arms spell the tier
            // "rk3v4" while this table's token is "rk3v4-g64" (the "-g64" is the scale-group
            // suffix, which the table carries and the refusal does not). A stricter predicate
            // would go red on two arms whose text is correct.
            // DROPPED IS EXEMPT FROM THE TIER PREDICATE, for a stated reason and with its
            // own pin. Its message names the CAUSE the operator wrote -- "DISCARDED
            // (NINFER_KV_DROP_LAYERS)" -- rather than the table token "dropped", and for a
            // layer the operator asked to drop that is the right vocabulary: the arm has to
            // tell them their request was honoured, not repeat the word back. Measured, not
            // assumed: without this the landed sweep reads 124/125.
            static const std::vector<std::string> tier_exempt = {"dropped"};
            check(tier_exempt.size() == 1,
                  "exactly ONE tier is exempt from the TIER predicate (a second exemption "
                  "cannot be added without failing here)");
            const bool exempt =
                std::find(tier_exempt.begin(), tier_exempt.end(), token) != tier_exempt.end();
            check(exempt || msg.find(tier_stem(token)) != std::string::npos,
                  slot + " (" + token +
                      ") refuses with a message naming ITS OWN TIER (not a generic one)");
        }
    }

    // ---- 5b. THE POLICY PIN, and it is here because its absence was MEASURED -------------
    // Everything above is a TOTALITY property: it says each tier is named and each is either
    // resolved or refused by name. It cannot tell WHICH, so lifting a refusal passes silently.
    // Measured on the E8K3Group64/E8K2Group64 lift (the flip batch's Item 2): the suite went
    // from == 133/133 == to == 129/129 == with rc=0 -- green, and the only outward sign was
    // the DENOMINATOR. A policy change must not be able to hide behind a denominator.
    //
    // So the current policy is written down once, as literals, and a change fails HERE until
    // someone edits this table on purpose. The table is the second opinion in the same sense
    // as pins(): the header decides, this file records what it decided.
    struct PolicyRow {
        std::size_t code;
        bool resolves;   // true == kv_dtype_for_storage returns a DType; false == it refuses
    };
    static const std::vector<PolicyRow> policy = {
        {0, true},   // bf16
        {1, true},   // int8-g64
        {2, true},   // fp8-e4m3-r256
        {3, true},   // nvfp4-g16
        {4, true},   // fp8-g16
        {5, true},   // iso4e-g16
        {6, true},   // rk4v4-g64
        {7, false},  // dropped -- DISCARDED, owns no planes
        {8, false},  // rk3v4-g64 -- refused until the planner AND a reader exist
        {9, false},  // rk2v4-g64 -- idem
        {10, false}, // fp8k-nvfp4v-g64 -- a K/V codec PAIR, not a tier
        {11, false}, // rot-i8k-i4v-g64 -- fork storage, no page-pool geometry here
        {12, false}, // rot-i4k-i4v-g64 -- idem
        {13, false}, // rk4v4e8 -- root-cylinder specimen, NOT the lattice
        {14, false}, // rk2v4e8 -- idem
    };
    check(policy.size() == kLast + 1,
          "the policy table covers every enumerator (" + std::to_string(policy.size()) +
              " rows vs " + std::to_string(kLast + 1) +
              "); a new tier fails HERE rather than defaulting to whatever the header happens"
              " to do");
    for (const PolicyRow& row : policy) {
        const auto storage = static_cast<KvCacheStorage>(row.code);
        bool resolves = true;
        try {
            (void)kv_dtype_for_storage(storage, "policy");
        } catch (const std::exception&) {
            resolves = false;
        }
        check(resolves == row.resolves,
              "code " + std::to_string(row.code) + " (" + std::string(kv_storage_token(storage)) +
                  ") " + (row.resolves ? "RESOLVES" : "REFUSES") +
                  " -- this row moves only when the policy does, and the move is the point");
    }

    // A code NO enumerator names: the escape hatch for a cast or untrusted option text. It
    // must refuse AND carry its own code, because an operator who typed 200 must be told 200.
    {
        const auto bogus = static_cast<KvCacheStorage>(kLast + 1);
        bool threw = false;
        std::string msg;
        try {
            (void)kv_storage_token(bogus);
        } catch (const std::exception& e) {
            threw = true;
            msg = e.what();
        }
        check(threw, "code " + std::to_string(kLast + 1) +
                         " (past the last enumerator) refuses instead of returning a name");
        check(msg.find(std::to_string(kLast + 1)) != std::string::npos,
              "code " + std::to_string(kLast + 1) + " names ITS OWN CODE in the refusal");
    }
}

} // namespace

int main() {
    std::printf("== kv operator name ==\n");
    check_pin();
    check_derive();
    check_total();
    check_census();
    check_sweep();
    std::printf("\n== %d/%d ==\n", g_ok, g_total);
    return g_ok == g_total ? 0 : 1;
}
