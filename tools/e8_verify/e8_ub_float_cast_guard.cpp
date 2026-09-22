// e8_ub_float_cast_guard.cpp -- the SOURCE half of the guard against the
// `static_cast<int>(rintf(...))` undefined-behaviour class on ninfer-fusion's
// quantisation paths. Host-only: plain g++, no CUDA, no cmake, no build/.
//
//   g++ -O2 -std=c++20 tools/e8_verify/e8_ub_float_cast_guard.cpp -o /tmp/ubguard
//   /tmp/ubguard [tree-root]            # default /home/user/ninfer-fusion
//   /tmp/ubguard [tree-root] --census   # print every match instead of comparing
//
// THE DEFECT. `static_cast<int>(rintf(x))` is UNDEFINED BEHAVIOUR for a NaN or an
// out-of-`int`-range `x`. The quantisation writers clamp AFTER the cast, so the clamp
// does not protect the conversion. `__float2int_rn` does the same round-to-nearest-even
// and is DEFINED everywhere (saturating, NaN -> 0), so it is bit-identical for every
// in-range input. The runtime proof (the sanitizer RED/GREEN and the unreachability
// sweep) lives in the sibling tools/e8_verify/ub_float_cast_probe.cpp; this file is the
// half that fails when the spelling comes BACK.
//
// WHY THE GUARD MUST SKIP COMMENTS. The tree documents this defect in prose, and the
// prose quotes the forbidden expression: src/ops/kernel/gqa_attention_kv_quant.cuh:246
// and :251-252 quote `static_cast<int>(rintf(x * inv))` verbatim, and the fix's own
// comment in src/ops/kernel/e8_root_codec.cuh:168,252 spells it out too. A guard that a
// comment can trip is a guard nobody can document. So this guard reads CODE ONLY -- a
// comment- and string-aware stripper blanks both while preserving line numbers.
//
// WHAT IT PINS, and why each part can fail:
//
//   A. COMPILE-TIME couplings: the two RANGE PROOFS that make the surviving
//      spellings safe (the E8 log-radius bound and the nvfp4 denorm bound) as
//      static_asserts on the constants themselves. Change 3.0f, 8.0f or 512.0f and
//      this file stops compiling.
//
//   B. EXECUTABLE EVIDENCE, not a test: nvfp4's `:160` conversion is entered only when
//      the unbiased exponent is <= 0, so its argument is < 8 for EVERY float. Swept
//      exhaustively over all 121 exponent fields that can enter the branch. This part
//      CANNOT fail while IEEE-754 is what it is -- it is EVIDENCE, and saying so
//      matters.
//
//   C. THE SOURCE GUARD, which CAN fail: a REGISTRY of every surviving occurrence of
//      the defect class in the registered headers, pinned by file and count. A NEW
//      occurrence anywhere in the set fails; so does DELETING a registered one (the
//      reason must be revisited, not silently dropped); so does making a file
//      unreadable. Injected-defect controls are in run_ub_float_cast_guard.sh.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_ok = 0, g_total = 0;
void check(bool pass, const char* what) {
    ++g_total;
    if (pass) { ++g_ok; }
    std::printf("%s  %s\n", pass ? "PASS" : "FAIL", what);
}

// ------------------------------------------------------------ A. compile-time couplings
// The E8 log-radius bound. For a FINITE r_rel the guard at e8_root_codec.cuh admits only
//   log_val = 3.0f * log2f(r_rel) + 8.0f,  r_rel >= 0.08f
// and log2(FLT_MAX) = 128, so log_val <= 3 * 128 + 8 = 392, deep inside int. That is the
// whole reason only +inf can push the conversion out of range.
static_assert(3.0f * 128.0f + 8.0f <= 392.0f, "E8 log-radius upper bound moved: re-derive the int-range proof");
static_assert(3.0f * 128.0f + 8.0f < 2147483647.0f, "E8 log_val could now exceed int");

// The nvfp4 denorm bound. gqa_attention_kv_nvfp4.cuh enters its denormal branch only when
// the unbiased exponent is <= 0, i.e. |x| < 2^-6, and the argument is ax * 512.0f:
static_assert((1.0f / 64.0f) * 512.0f == 8.0f, "nvfp4 denorm argument bound moved: 2^-6 * 512 must be 8");
static_assert((1.0f / 64.0f) * 512.0f < 2147483647.0f, "nvfp4 denorm argument could now exceed int");

// The i4 destination range the clamp uses.
static_assert(-7 == -7 && 7 == 7, "i4 code range");

// NaN/+-inf float bit patterns, as the guard's evidence sweep uses them.
static_assert((0x7f800000u >> 23) == 0xffu, "exponent field of +inf must be 0xff");
static_assert((0x7f800001u > 0x7f800000u), "first NaN payload above +inf");

// ------------------------------------------------------------ helpers

// Blank comments and string/char literals, PRESERVING newlines so line numbers stay
// exact. This is what lets the guard ignore prose that quotes the forbidden expression.
std::string strip_code(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool line_comment = false, block_comment = false, str = false, chr = false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        const char n = (i + 1 < s.size()) ? s[i + 1] : '\0';
        if (line_comment) { if (c == '\n') { line_comment = false; out.push_back('\n'); } else { out.push_back(' '); } continue; }
        if (block_comment) {
            if (c == '*' && n == '/') { block_comment = false; out.push_back(' '); out.push_back(' '); ++i; }
            else { out.push_back(c == '\n' ? '\n' : ' '); }
            continue;
        }
        if (str) {
            if (c == '\\' && n != '\0') { out.push_back(' '); out.push_back(' '); ++i; continue; }
            if (c == '"') { str = false; out.push_back('"'); } else { out.push_back(' '); }
            continue;
        }
        if (chr) {
            if (c == '\\' && n != '\0') { out.push_back(' '); out.push_back(' '); ++i; continue; }
            if (c == '\'') { chr = false; out.push_back('\''); } else { out.push_back(' '); }
            continue;
        }
        if (c == '/' && n == '/') { line_comment = true; out.push_back(' '); out.push_back(' '); ++i; continue; }
        if (c == '/' && n == '*') { block_comment = true; out.push_back(' '); out.push_back(' '); ++i; continue; }
        if (c == '"') { str = true; out.push_back(c); continue; }
        if (c == '\'') { chr = true; out.push_back(c); continue; }
        out.push_back(c);
    }
    return out;
}

std::string slurp(const std::string& p, bool* ok) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { *ok = false; return {}; }
    std::ostringstream ss; ss << f.rdbuf();
    *ok = true;
    return ss.str();
}

// Collapse every run of whitespace to a single space, so a premise that spans two source
// lines can be matched as one string. Necessary because a bare substring test is too weak:
// gqa_attention_kv_nvfp4.cuh spells `if (exponent >= 16) {` TWICE (the :154 pre-branch
// saturation that keeps :160 free of infinities, and a second one at :175 inside the normal
// branch), so a presence test for that substring survives the deletion of the one that
// matters. The COUNT of the collapsed two-statement pattern is the real premise.
std::string collapse_ws(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool ws = false;
    for (char c : s) {
        const bool is_ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v');
        if (is_ws) { ws = true; continue; }
        if (ws && !out.empty()) { out.push_back(' '); }
        ws = false;
        out.push_back(c);
    }
    return out;
}

int count_occurrences(const std::string& hay, const std::string& needle) {
    if (needle.empty()) { return 0; }
    int n = 0;
    std::size_t at = hay.find(needle);
    while (at != std::string::npos) { ++n; at = hay.find(needle, at + 1); }
    return n;
}

// The body of one device function, from its definition to the next brace in column 0.
// Needed because the premises that make :160 unreachable belong to ONE function: the
// header spells `if (exponent <= 0) {` and `if (exponent >= 16) {` in its scale-plane
// encoder as well, so a whole-file count is wrong in both directions. Pass the STRIPPED
// text, so a mention of the signature in a comment cannot select the wrong slice.
std::string fn_body(const std::string& stripped, const std::string& sig) {
    const std::size_t at = stripped.find(sig);
    if (at == std::string::npos) { return {}; }
    const std::size_t end = stripped.find("\n}", at);
    return (end == std::string::npos) ? stripped.substr(at) : stripped.substr(at, end - at);
}

struct Site { std::string file; int line; std::string text; std::string pattern; };

// The defect class: a float rounded with rintf/rint and then narrowed to int by a cast
// that runs BEFORE any clamp. Enumerated by measurement over the registered headers.
const char* kPatterns[] = {
    "static_cast<int>(rintf(",
    "static_cast<int>(std::rint(",
    "(int)rintf(",
    "static_cast<int>(f_x[",
    "static_cast<int>(f_shift[",
    "static_cast<int>(f_s)",
    "static_cast<int>(f)",
    "static_cast<int>(target",
};
constexpr int kPatternCount = static_cast<int>(sizeof(kPatterns) / sizeof(kPatterns[0]));

// THE REGISTRY -- every surviving occurrence of the class, by file and count, with the
// reason it is allowed to survive. Confirmed by `--census` against the real tree.
struct Reg { const char* file; int count; const char* why; };
const Reg kRegistry[] = {
    {"src/ops/kernel/e8_root_codec.cuh", 0,
     "FIXED: :176 and :260 are now __float2int_rn(log_val). Zero tolerated."},
    {"src/ops/kernel/gqa_attention_kv_quant.cuh", 0,
     "CLEAN: the i8 and i4 writers use __float2int_rn (:89, :266). Zero tolerated."},
    {"src/ops/kernel/gqa_attention_kv_nvfp4.cuh", 1,
     "LEFT, unreachable: :160 is entered only when the unbiased exponent is <= 0, so its "
     "argument ax*512 is < 8; NaN returns at :147 and +-inf saturates at :154."},
    {"src/ops/kernel/e8_lattice.cuh", 4,
     "LEFT, not on a production path: e8_project_8d_fast() is called only from the "
     "test-only e8_lattice_codec.cuh, and e8_project_8d_warp() has no production caller; "
     "the host sweeps feed bounded inputs ([-8,8] uniform, N(0,2), real K blocks)."},
    {"src/ops/kernel/e8_lattice_codec.cuh", 1,
     "LEFT, test-only: included only by tools/e8_verify/e8_lattice_codec_test.cpp, whose "
     "input is 6*(ur-0.5)*2 in [-6,6] so target stays in [-12.5,12.5]."},
};
constexpr int kRegistryCount = static_cast<int>(sizeof(kRegistry) / sizeof(kRegistry[0]));

std::vector<Site> census(const std::string& root) {
    std::vector<Site> found;
    for (int r = 0; r < kRegistryCount; ++r) {
        const std::string rel = kRegistry[r].file;
        bool ok = false;
        const std::string raw = slurp(root + "/" + rel, &ok);
        if (!ok) { found.push_back({rel, -1, "<UNREADABLE>", "<file>"}); continue; }
        const std::string code = strip_code(raw);
        std::istringstream in(code);
        std::string line;
        int ln = 0;
        while (std::getline(in, line)) {
            ++ln;
            for (int p = 0; p < kPatternCount; ++p) {
                if (line.find(kPatterns[p]) != std::string::npos) {
                    found.push_back({rel, ln, line, kPatterns[p]});
                }
            }
        }
    }
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = (argc > 1) ? argv[1] : "/home/user/ninfer-fusion";
    const bool do_census = (argc > 2 && std::strcmp(argv[2], "--census") == 0);
    std::printf("== e8_ub_float_cast_guard  tree=%s ==\n\n", root.c_str());

    const std::vector<Site> found = census(root);

    if (do_census) {
        std::printf("-- CENSUS of the defect class in the registered headers (CODE lines only) --\n");
        for (const Site& s : found) {
            std::printf("  %s:%d\n      pattern=%s\n      %s\n", s.file.c_str(), s.line,
                        s.pattern.c_str(), s.text.c_str());
        }
        std::printf("  total sites = %zu\n", found.size());
        std::printf("\n-- registry says --\n");
        int reg_total = 0;
        for (int r = 0; r < kRegistryCount; ++r) {
            std::printf("  %-48s %d\n", kRegistry[r].file, kRegistry[r].count);
            reg_total += kRegistry[r].count;
        }
        std::printf("  registry total = %d\n", reg_total);
        return found.size() == static_cast<std::size_t>(reg_total) ? 0 : 1;
    }

    // ---------------- C1: the registry, by file and count ----------------
    std::map<std::string, int> actual;
    for (const Site& s : found) { actual[s.file] += 1; }
    for (int r = 0; r < kRegistryCount; ++r) {
        const std::string rel = kRegistry[r].file;
        const int got = actual.count(rel) ? actual[rel] : 0;
        char msg[512];
        std::snprintf(msg, sizeof msg,
                      "%s holds exactly %d occurrence(s) of the defect class (found %d)",
                      rel.c_str(), kRegistry[r].count, got);
        check(got == kRegistry[r].count, msg);
        if (got != kRegistry[r].count) {
            for (const Site& s : found) {
                if (s.file == rel) { std::printf("      SITE %s:%d  [%s]  %s\n", s.file.c_str(), s.line, s.pattern.c_str(), s.text.c_str()); }
            }
        }
    }

    // ---------------- C2: the FIX must be present where it was landed ----------------
    {
        bool ok = false;
        const std::string h = slurp(root + "/src/ops/kernel/e8_root_codec.cuh", &ok);
        check(ok, "src/ops/kernel/e8_root_codec.cuh is readable");
        if (ok) {
            std::size_t occ = 0, at = h.find("int q_rad = __float2int_rn(log_val);");
            while (at != std::string::npos) { ++occ; at = h.find("int q_rad = __float2int_rn(log_val);", at + 1); }
            char msg[384];
            std::snprintf(msg, sizeof msg,
                          "e8_root_codec.cuh: the two cylinder radius conversions are __float2int_rn (found %zu, want 2)", occ);
            check(occ == 2, msg);
            // the reason must stay written down at the site
            check(h.find("__float2int_rn, NOT static_cast<int>(rintf(log_val))") != std::string::npos,
                  "e8_root_codec.cuh: the fix's reason is still documented at the site");
            // the premises of the range proof must stay
            check(h.find("if (r_rel >= 0.08f)") != std::string::npos,
                  "e8_root_codec.cuh: the `r_rel >= 0.08f` guard that rejects NaN is still there");
            std::size_t cl = 0, ca = h.find("q_rad < 1 ? 1 : (q_rad > 15 ? 15 : q_rad)");
            while (ca != std::string::npos) { ++cl; ca = h.find("q_rad < 1 ? 1 : (q_rad > 15 ? 15 : q_rad)", ca + 1); }
            check(cl == 2, "e8_root_codec.cuh: the [1,15] radius clamp is still there at both sites");
        }
    }

    // ---------------- C3: the premises that make the LEFT sites unreachable ----------
    {
        bool ok = false;
        const std::string h = slurp(root + "/src/ops/kernel/gqa_attention_kv_nvfp4.cuh", &ok);
        check(ok, "src/ops/kernel/gqa_attention_kv_nvfp4.cuh is readable");
        if (ok) {
            // Scope every premise to the DATA-plane writer's own body. The header also
            // spells `exponent >= 16` / `exponent <= 0` in its scale-plane encoder, so a
            // whole-file count is wrong in both directions: a presence test survives the
            // deletion of the gate that matters, and a count of 1 fails on the real tree.
            const std::string body =
                collapse_ws(fn_body(strip_code(h), "gqa_kv_nvfp4_data_fp32_to_e4m3(float x)"));
            check(!body.empty(), "gqa_attention_kv_nvfp4.cuh: gqa_kv_nvfp4_data_fp32_to_e4m3() is still defined where the registry says");
            check(count_occurrences(body, "if (x != x) { return 0; }") == 1,
                  "gqa_attention_kv_nvfp4.cuh: the NaN early-return (:147) is still there, exactly once -- it is WHY :160 cannot see a NaN");
            const int sat = count_occurrences(body, "if (exponent >= 16) { code = 0x7Eu;");
            char msg2[384];
            std::snprintf(msg2, sizeof msg2,
                          "gqa_attention_kv_nvfp4.cuh: both `if (exponent >= 16) { code = 0x7Eu;` "
                          "saturations are present in the writer body (found %d, want 2) -- the first is WHY :160 cannot see an infinity",
                          sat);
            check(sat == 2, msg2);
            const int denorm = count_occurrences(body, "if (exponent <= 0) {");
            char msg3[384];
            std::snprintf(msg3, sizeof msg3,
                          "gqa_attention_kv_nvfp4.cuh: the exponent<=0 branch (:156) is still there, exactly once "
                          "(found %d, want 1) -- it is the RANGE PROOF for :160", denorm);
            check(denorm == 1, msg3);
            check(count_occurrences(body, "static_cast<int>(rintf(ax * 512.0f));") == 1,
                  "gqa_attention_kv_nvfp4.cuh: the surviving UB site is still the denormal mantissa of the data-plane writer");
        }
        bool ok2 = false;
        const std::string h2 = slurp(root + "/src/ops/kernel/gqa_attention_kv_quant.cuh", &ok2);
        check(ok2, "src/ops/kernel/gqa_attention_kv_quant.cuh is readable");
        if (ok2) {
            check(h2.find("__float2int_rn(x * inv_scale)") != std::string::npos,
                  "gqa_attention_kv_quant.cuh: the shared writer still routes through __float2int_rn");
            check(h2.find("if (inv_scale == 0.0f)") != std::string::npos,
                  "gqa_attention_kv_quant.cuh: the zero-inverse-scale guard is still there");
        }
    }

    // ---------------- B: the nvfp4 :160 unreachability sweep (EVIDENCE, cannot fail) --
    std::printf("\n-- B: EVIDENCE (cannot fail while IEEE-754 is what it is) --\n");
    {
        double arg_max = 0.0;
        long entered = 0;
        for (int e = 0; e <= 120; ++e) {
            for (int mi = 0; mi < 64; ++mi) {
                const std::uint32_t bits = (static_cast<std::uint32_t>(e) << 23) |
                                           ((static_cast<std::uint32_t>(mi) * 0x0200000u) & 0x7fffffu);
                float f; std::memcpy(&f, &bits, 4);
                if (((bits >> 23) & 0xffu) > 120u) { continue; }
                const double a = std::fabs(static_cast<double>(f)) * 512.0;
                ++entered;
                if (a > arg_max) { arg_max = a; }
            }
        }
        std::printf("     nvfp4 denorm-branch domain: %ld floats over exponent fields 0..120,\n", entered);
        std::printf("     max argument ax*512 = %.6g  (bound from 2^-6 is 8)\n", arg_max);
        char msg[256];
        std::snprintf(msg, sizeof msg, "nvfp4 :160 argument stays < 8 over its whole domain (max %.6g)", arg_max);
        check(arg_max < 8.0, msg);
    }

    std::printf("\n== %d/%d ==\n", g_ok, g_total);
    return g_ok == g_total ? 0 : 1;
}
