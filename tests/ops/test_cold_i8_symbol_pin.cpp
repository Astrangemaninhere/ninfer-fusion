// test_cold_i8_symbol_pin -- ONE entry point must have ONE spelling, on BOTH sides of the
// declaration/definition pair, and the retired spelling must not come back.
//
// THE DEFECT THIS EXISTS FOR (NAMEFIX5, 2026-09-18). The e8 -> rk4v4 rename half-landed for the
// THIRD time in one night. The first two split enumerators; this one split a SYMBOL PAIR:
//
//   include/ninfer/ops/cold_i8.h:46    DECLARES   void cold_i8_slot_restore_rk4v4_raw(...)
//   program_impl.h:12840/12843         CALLS      ops::cold_i8_slot_restore_rk4v4_raw(...)
//   src/ops/wrapper/cold_i8.cpp:41     DEFINES    void cold_i8_slot_restore_e8_raw(...)
//
// and the engine link died at 100% of the build with
//     undefined reference to `ninfer::ops::cold_i8_slot_restore_rk4v4_raw(...)'
// referenced from four ProgramImplCore::restore_cold_page instantiations inside variant.cpp.
//
// Nothing could see it earlier, and the reason is C++-shaped rather than accidental: the defining
// translation unit DOES include the header, so the declaration was in view the whole time -- but a
// definition whose spelling misses the declaration is not an error, it silently INTRODUCES a new
// namespace-scope function. The compiler is satisfied, the archive is satisfied, and only the last
// link objects. The defining file carries the fossil of the previous half-landing at :27-31 --
// 'undefined reference to ..._restore_{bf16,e8}_raw' -- which is the same defect one level up.
//
// WHAT IS ASSERTED, and why each check can fail on its own:
//
//   1 PIN      the ADDRESS of every public entry point is taken into a `volatile` pointer. That is
//              a real odr-use, so if EITHER side of the pair moves alone the TEST fails to link,
//              naming the symbol -- one build earlier than the engine link, and inside the artifact
//              whose whole job is to prove the surface exists. (This is the only shape available: a
//              declaration/definition pair cannot be pinned by static_assert the way a constant
//              can, because the compiler cannot see the other translation unit.)
//   2 DECLARES the set of names the HEADER declares == the pinned list. A rename that lands on the
//              header alone is caught here, with a message naming the drifted symbol.
//   3 DEFINES  the set of names the WRAPPER defines == the set the header declares. THIS is the
//              check whose absence let the defect land, and the only one of the five that is TRUE
//              OF THE TEXT ALONE: no build, no link, no engine, so it runs in milliseconds in every
//              configuration -- including the ones where no test suite is built at all
//              (build-sim ships BUILD_TESTING=OFF, and that is exactly where the link died).
//   4 RETIRED  the retired spelling must not appear in CODE anywhere in the family's file set.
//              This covers the case check 1 CANNOT see: a re-introduced second definition under the
//              old name, which the linker resolves happily and which therefore reports itself to
//              nobody. Comments are history and are allowed. The same scan covers the deferred
//              gqa_iso3_* family's half-rename marker, `gqa_iso4e_`: the codec is defined once, in
//              gqa_iso3_codec.cuh, and 40+ call sites across three other lines' files all say
//              `gqa_iso3_`. A rename of that family that lands on one side only starts here.
//   5 RATCHET  the number of code sites still spelled `_e8_` in this family (the deferred
//              launcher/kernel helper: ops/launcher/cold_i8.{h,cu}, ops/kernel/cold_i8_kernels.cuh)
//              is capped at its pinned present value and PRINTED. Debt below the cap may be paid
//              silently; it may not GROW silently. The cap was 6 while that helper was deferred and
//              is 0 since NAMEFIX6 (2026-09-18) renamed it to the rk4v4 vocabulary -- so `_e8_`
//              coming back in this family is now a failure on its first occurrence, not its seventh.
//
// Limits, stated rather than implied: the census is a fixed file list, so it cannot see a NEW file;
// it sees every file this rename has actually touched. And it reads text, so it is a census of
// spelling, not a proof of linkage -- check 1 is the linkage.
//
// Host-only: no CUDA call, no device, no model, no lock, no subprocess, no argv.

#include "ninfer/ops/cold_i8.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#error "this test reads the tree as text, so it needs NINFER_SOURCE_DIR"
#endif

namespace {

// ---------------------------------------------------------------------------
// 1. PIN -- the odr-use. `volatile` is load-bearing: without it the compiler
//    folds `&f == nullptr` and drops the very relocation this check exists for.
// ---------------------------------------------------------------------------
auto* volatile g_pin_pack         = &ninfer::ops::cold_i8_slot_pack_raw;
auto* volatile g_pin_restore      = &ninfer::ops::cold_i8_slot_restore_raw;
auto* volatile g_pin_restore_bf16 = &ninfer::ops::cold_i8_slot_restore_bf16_raw;
auto* volatile g_pin_restore_rk4v4 = &ninfer::ops::cold_i8_slot_restore_rk4v4_raw;

// The pinned list. Hand-written ON PURPOSE: it is the contract, not a derivation from either file
// it checks, so "the header agrees with the pin" is a statement that can fail.
const char* const kPinned[] = {
    "cold_i8_slot_pack_raw",
    "cold_i8_slot_restore_raw",
    "cold_i8_slot_restore_bf16_raw",
    "cold_i8_slot_restore_rk4v4_raw",
};

// 4. The retired spellings. A hit in code is a failure; a hit in a comment is history.
const char* const kRetired[] = {
    "cold_i8_slot_restore_e8_raw", // renamed to _rk4v4_raw on 2026-09-18
    "gqa_iso4e_",                  // the deferred gqa_iso3_* family's half-rename marker
};

// The family's file set: everything the e8->rk4v4 rename has touched or was deferred in.
const char* const kCensus[] = {
    "include/ninfer/ops/cold_i8.h",
    "src/ops/wrapper/cold_i8.cpp",
    "src/ops/launcher/cold_i8.h",
    "src/ops/launcher/cold_i8.cu",
    "src/ops/kernel/cold_i8_kernels.cuh",
    "src/ops/kernel/entropy_cold_requant_kernels.cuh",
    "src/ops/kernel/gqa_iso3_codec.cuh",
};

// 5. RATCHET: the `_e8_` spelling in this family today (see the header comment).
// NAMEFIX6 (2026-09-18) LOWERED this pin 6 -> 0 as part of the landing that paid the debt it was
// capping: the deferred detail helper `cold_i8_slot_restore_e8_launch` / `..._e8_kernel` became
// `..._rk4v4_launch` / `..._rk4v4_kernel` at six code sites in four files (ops/launcher/cold_i8.h:24,
// ops/launcher/cold_i8.cu:112/176/186, ops/kernel/cold_i8_kernels.cuh:67, ops/wrapper/cold_i8.cpp:60).
// The measured count after that landing is 0, and this program prints it below -- the number is
// measured here, not asserted from the diff. Updating the pin IS part of that atomic landing.
// A cap is LOWERED only by paying debt; RAISING it is how this guard would be weakened.
constexpr int kPinnedE8CodeSites = 0;

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot open " << path << '\n';
        std::exit(2);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Remove // and /* */ comments. A `//` that appears inside a string literal would be stripped
// wrongly, but this family's files carry no such literal in the lines under test.
std::string strip_comments(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool line_comment = false;
    bool block_comment = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        const char n = (i + 1 < text.size()) ? text[i + 1] : '\0';
        if (line_comment) {
            if (c == '\n') { line_comment = false; out.push_back(c); }
            continue;
        }
        if (block_comment) {
            if (c == '*' && n == '/') { block_comment = false; ++i; }
            else if (c == '\n') { out.push_back(c); }
            continue;
        }
        if (c == '/' && n == '/') { line_comment = true; ++i; continue; }
        if (c == '/' && n == '*') { block_comment = true; ++i; continue; }
        out.push_back(c);
    }
    return out;
}

std::vector<std::string> sorted(std::vector<std::string> v) {
    for (std::size_t i = 0; i + 1 < v.size(); ++i) {
        for (std::size_t j = i + 1; j < v.size(); ++j) {
            if (v[j] < v[i]) { std::swap(v[i], v[j]); }
        }
    }
    return v;
}

std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) { s += (i ? ", " : ""); s += v[i]; }
    return s;
}

// Every `void cold_i8_<name>(` in COMMENT-STRIPPED text -- i.e. a declaration or a definition,
// never a call and never prose.
std::vector<std::string> cold_i8_void_names(const std::string& stripped) {
    std::vector<std::string> names;
    const std::string key = "void cold_i8_";
    for (std::size_t p = stripped.find(key); p != std::string::npos;
         p = stripped.find(key, p + key.size())) {
        const std::size_t b = p + 5; // past "void "
        const std::size_t e = stripped.find('(', b);
        if (e == std::string::npos) { continue; }
        std::string name = stripped.substr(b, e - b);
        bool clean = !name.empty();
        for (char c : name) {
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) { clean = false; }
        }
        if (clean) { names.push_back(name); }
    }
    return sorted(names);
}

} // namespace

int main() {
    const std::string root = NINFER_SOURCE_DIR;
    int failures = 0;

    // ---- 1 PIN: the odr-use. A null here is impossible; the READ is the point. ----------------
    struct Pin { const char* name; bool ok; };
    const Pin pins[] = {
        {"cold_i8_slot_pack_raw",             g_pin_pack != nullptr},
        {"cold_i8_slot_restore_raw",          g_pin_restore != nullptr},
        {"cold_i8_slot_restore_bf16_raw",     g_pin_restore_bf16 != nullptr},
        {"cold_i8_slot_restore_rk4v4_raw",    g_pin_restore_rk4v4 != nullptr},
    };
    for (const Pin& p : pins) {
        if (!p.ok) {
            std::cerr << "PIN: the address of " << p.name << " did not resolve\n";
            ++failures;
        }
    }
    const std::vector<std::string> pinned = sorted(std::vector<std::string>(
        kPinned, kPinned + sizeof(kPinned) / sizeof(kPinned[0])));

    const std::string header = strip_comments(read_file(root + "/include/ninfer/ops/cold_i8.h"));
    const std::string wrapper =
        strip_comments(read_file(root + "/src/ops/wrapper/cold_i8.cpp"));

    // ---- 2 DECLARES: the header's declared set == the pin. ------------------------------------
    const std::vector<std::string> declared = cold_i8_void_names(header);
    if (declared != pinned) {
        std::cerr << "DECLARES: include/ninfer/ops/cold_i8.h and this pin disagree\n"
                  << "  header declares: " << join(declared) << '\n'
                  << "  pin holds      : " << join(pinned) << '\n'
                  << "  (a rename that landed on the header alone is this message)\n";
        ++failures;
    }

    // ---- 3 DEFINES: the wrapper's defined set == the header's declared set. -------------------
    // This is the build-free form of the link, and the check that would have caught NAMEFIX5
    // before any compilation happened.
    const std::vector<std::string> defined = cold_i8_void_names(wrapper);
    if (defined != declared) {
        std::cerr << "DEFINES: the wrapper does not define exactly what the header declares\n"
                  << "  header declares: " << join(declared) << '\n'
                  << "  wrapper defines: " << join(defined) << '\n';
        for (const std::string& d : defined) {
            if (std::find(declared.begin(), declared.end(), d) == declared.end()) {
                std::cerr << "  DEFINED BUT NOT DECLARED: " << d
                          << "  <- the dangerous shape: the linker resolves it, nothing reports it\n";
            }
        }
        for (const std::string& d : declared) {
            if (std::find(defined.begin(), defined.end(), d) == defined.end()) {
                std::cerr << "  DECLARED BUT NOT DEFINED: " << d
                          << "  <- this is the NAMEFIX5 link failure, 100% into the build\n";
            }
        }
        ++failures;
    }

    // ---- 4 RETIRED: a retired spelling must not appear in code. -------------------------------
    for (const char* rel : kCensus) {
        const std::string path = root + "/" + rel;
        const std::ifstream probe(path);
        if (!probe) {
            std::cerr << "RETIRED: census file missing: " << rel << '\n';
            ++failures;
            continue;
        }
        const std::string stripped = strip_comments(read_file(path));
        for (const char* tok : kRetired) {
            if (stripped.find(tok) != std::string::npos) {
                std::cerr << "RETIRED: " << tok << " appears in CODE in " << rel
                          << " -- a retired spelling came back\n";
                ++failures;
            }
        }
    }

    // ---- 5 RATCHET: the remaining `_e8_` spelling in this family, capped and printed. ---------
    int e8_sites = 0;
    std::string e8_where;
    for (const char* rel : kCensus) {
        const std::string path = root + "/" + rel;
        std::ifstream probe(path);
        if (!probe) { continue; }
        const std::string stripped = strip_comments(read_file(path));
        const std::string key = "cold_i8_";
        for (std::size_t p = stripped.find(key); p != std::string::npos;
             p = stripped.find(key, p + key.size())) {
            const std::size_t e = stripped.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_",
                                                             p + key.size());
            const std::string name = stripped.substr(p, e - p);
            if (name.find("_e8_") == std::string::npos) { continue; }
            ++e8_sites;
            e8_where += "    " + std::string(rel) + ": " + name + "\n";
        }
    }
    if (e8_sites > kPinnedE8CodeSites) {
        std::cerr << "RATCHET: the `_e8_` spelling in this family grew from " << kPinnedE8CodeSites
                  << " to " << e8_sites << " code sites:\n" << e8_where
                  << "  Pay the debt, or rename the addition -- but do not add to it.\n";
        ++failures;
    } else {
        std::cout << "RATCHET: " << e8_sites << "/" << kPinnedE8CodeSites
                  << " code sites still spelled `_e8_` (deferred vocabulary debt):\n" << e8_where;
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " cold_i8 symbol pin: one entry point, one spelling, on both sides of the pair\n";
    return failures == 0 ? 0 : 1;
}
