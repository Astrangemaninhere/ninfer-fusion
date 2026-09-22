// e8_dead_guard_test -- the host-only guard for the `std::array<T,N>::empty()` defect in
// the paged KV layer views.
//
// THE DEFECT. src/targets/qwen3_6/impl/state/decoder_state.cpp decided a per-layer value
// with `layer_dtypes_.empty() ? dtype_ : layer_dtypes_[layer]` (and the same shape for
// layer_plane_base_, layer_residual_ and layer_sliding_windows_, in TWO accessors,
// layer_view() and batch_layer_view()). `layer_dtypes_` is `std::array<DType, 64>`
// (decoder_state.h), and `std::array<T,N>::empty()` is specified as `return false` -- a
// CONSTANT, not a query. So every one of those guards was a constant false and the fallback
// arm was UNREACHABLE. Sixteen call sites (E8VERIFY named 6 of them) plus one comment that
// treated the guard as a semantic signal.
//
// WHAT IT MEANT, AND WHY REMOVAL CHANGES NOTHING. The fallback meant "no per-layer table was
// supplied, so use the pool-wide dtype". That state does not exist: plan_cache() resolves the
// global dtype inheritance itself (kv_resolve_slot_dtype), writes EVERY layer of all four
// tables, and throws on a table shorter than the layer count; plan_cache() is the only
// producer of a PagedKVCacheLayout. So the tables are always fully populated and
// AUTHORITATIVE, and a fallback would have SILENTLY IGNORED the per-layer resolution the
// layout had already performed. Removing the guard is therefore not merely a no-op: the
// guard's own reading of itself was unsound.
//
// WHAT THIS FILE IS. Three parts, and Part C is the one that is a TEST:
//   A. the type-level reason, as static_asserts.
//   B. an executable proof that the guard was a no-op: the removed ternary and the landed
//      unconditional read are evaluated over the whole 64-layer domain, on a table that is
//      populated AND one that is default-constructed, counting how often the fallback arm
//      could be selected. The count is 0. This part CANNOT fail while the standard library
//      is what it is -- it is EVIDENCE, not a test, and saying so matters.
//   C. a SOURCE GUARD over the real decoder_state.cpp / decoder_state.h that FAILS if the
//      pattern returns as CODE, if the table declarations change type, if the reason stops
//      being documented, or if the files cannot be read. It reads CODE lines only: a guard
//      that a comment can trip is a guard nobody can document. This part CAN fail, and its
//      injected-defect controls are in tools/e8_verify/run_dead_guard.sh.
//
// HOST-ONLY: no CUDA, no cmake, no build/ (only core/dtype.h, which the sibling
// e8_width_contract_test.cpp already includes host-side).
//
//   g++ -O2 -std=c++17 -I/home/user/ninfer-fusion/src \
//       tools/e8_verify/e8_dead_guard_test.cpp -o /tmp/e8dg
//   /tmp/e8dg [tree-root]        # default /home/user/ninfer-fusion

#include "core/dtype.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_ok = 0, g_total = 0;
void check(bool pass, const char* what) {
    ++g_total;
    if (pass) { ++g_ok; }
    std::printf("%s  %s\n", pass ? "PASS" : "FAIL", what);
}

// ---------------------------------------------------------------- Part A: type level
// The four members of PagedKVCache, at the types decoder_state.h declares for them.
using LayerDtypes   = std::array<ninfer::DType, 64>;
using LayerResidual = std::array<bool, 64>;
using LayerWindows  = std::array<std::uint32_t, 64>;
using LayerBases    = std::array<std::uint32_t, 64>;

static_assert(LayerDtypes{}.empty() == false,   "std::array<T,64>::empty() is a constant false");
static_assert(LayerResidual{}.empty() == false, "std::array<T,64>::empty() is a constant false");
static_assert(LayerWindows{}.empty() == false,  "std::array<T,64>::empty() is a constant false");
static_assert(LayerBases{}.empty() == false,    "std::array<T,64>::empty() is a constant false");
static_assert(std::array<std::uint32_t, 1>{}.empty() == false,  "even for N == 1");
static_assert(std::array<std::uint32_t, 0>{}.empty() == true,   "N == 0 is the ONLY empty one");

// ---------------------------------------------------------------- Part B: behavioural
struct Equiv {
    std::uint64_t evaluated = 0;
    std::uint64_t guard_true = 0;
    std::uint64_t fell_back = 0;
    std::uint64_t disagreed = 0;
    std::uint64_t intended_fallback = 0;
};

Equiv compare_all_layers(const LayerDtypes& table, ninfer::DType pool_wide, bool table_is_absent) {
    Equiv e;
    for (std::uint32_t layer = 0; layer < 64; ++layer) {
        const bool guard = table.empty();                        // the removed predicate
        const ninfer::DType removed = guard ? pool_wide : table[layer];
        const ninfer::DType landed = table[layer];
        ++e.evaluated;
        if (guard) { ++e.guard_true; ++e.fell_back; }
        if (removed != landed) { ++e.disagreed; }
        if (table_is_absent) { ++e.intended_fallback; }
    }
    return e;
}

// ---------------------------------------------------------------- Part C helpers
bool read_file(const std::string& path, std::string* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { return false; }
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

bool is_comment_line(const std::string& ln) {
    std::size_t i = 0;
    while (i < ln.size() && (ln[i] == ' ' || ln[i] == '\t')) { ++i; }
    if (i + 1 < ln.size() && ln[i] == '/' && ln[i + 1] == '/') { return true; }
    if (i < ln.size() && ln[i] == '*') { return true; }
    if (i + 1 < ln.size() && ln[i] == '/' && ln[i + 1] == '*') { return true; }
    return false;
}

// CODE lines only, with their 1-based line numbers. A guard over comments is a guard that
// documentation trips, which is a guard nobody can document -- so this filter is the point.
std::vector<std::pair<int, std::string>> code_lines(const std::string& text) {
    std::vector<std::pair<int, std::string>> out;
    std::istringstream in(text);
    std::string ln;
    int n = 0;
    while (std::getline(in, ln)) {
        ++n;
        while (!ln.empty() && (ln.back() == '\r' || ln.back() == '\n')) { ln.pop_back(); }
        if (is_comment_line(ln) || ln.empty()) { continue; }
        out.push_back({n, ln});
    }
    return out;
}

std::vector<std::string> offenders_in(const std::string& text, const std::string& member,
                                      const char* where) {
    std::vector<std::string> hits;
    const std::string needle = member + ".empty()";
    for (const auto& [n, ln] : code_lines(text)) {
        if (ln.find(needle) != std::string::npos) {
            hits.push_back(std::string(where) + " line " + std::to_string(n) + ": " + ln);
        }
    }
    return hits;
}

} // namespace

int main(int argc, char** argv) {
    const std::string root = (argc >= 2) ? argv[1] : "/home/user/ninfer-fusion";
    std::printf("=== e8_dead_guard_test (std::array<T,N>::empty() is a constant false) ===\n");
    std::printf("tree root: %s\n\n", root.c_str());

    // ---------------------------------------------------------------- A
    std::printf("--- A. the type-level reason (the static_asserts above must compile) ---\n");
    check(!LayerDtypes{}.empty() && !LayerResidual{}.empty() && !LayerWindows{}.empty() &&
              !LayerBases{}.empty(),
          "all four member types have a constant-false empty()");
    check(std::array<std::uint32_t, 1>{}.empty() == false &&
              std::array<std::uint32_t, 0>{}.empty() == true,
          "empty() is false for N==1 and true for N==0: a size CONSTANT, not a state");

    // ---------------------------------------------------------------- B
    std::printf("\n--- B. the removed ternary vs the landed read, over all 64 layers ---\n");
    {
        LayerDtypes populated{};
        for (std::uint32_t l = 0; l < 64; ++l) {
            populated[l] = (l % 3 == 0) ? ninfer::DType::E8Kv
                           : (l % 3 == 1) ? ninfer::DType::NVFP4
                                          : ninfer::DType::BF16;
        }
        const Equiv a = compare_all_layers(populated, ninfer::DType::I8, /*table_is_absent=*/false);
        std::printf("  populated table, pool-wide dtype deliberately DIFFERENT (I8):\n");
        std::printf("    evaluated %llu  guard-true %llu  fell-back %llu  disagreed %llu\n",
                    (unsigned long long)a.evaluated, (unsigned long long)a.guard_true,
                    (unsigned long long)a.fell_back, (unsigned long long)a.disagreed);
        check(a.guard_true == 0 && a.fell_back == 0,
              "populated table: the guard is never true, so the fallback is never selected");
        check(a.disagreed == 0,
              "populated table: the removed ternary and the landed read agree on all 64 layers");

        LayerDtypes absent{};
        const Equiv b = compare_all_layers(absent, ninfer::DType::E8Kv, /*table_is_absent=*/true);
        std::printf("  default-constructed table, pool-wide dtype E8Kv:\n");
        std::printf("    evaluated %llu  guard-true %llu  fell-back %llu  disagreed %llu\n"
                    "    layers that \"table absent -> use the pool dtype\" WOULD have covered: %llu\n",
                    (unsigned long long)b.evaluated, (unsigned long long)b.guard_true,
                    (unsigned long long)b.fell_back, (unsigned long long)b.disagreed,
                    (unsigned long long)b.intended_fallback);
        check(b.fell_back == 0,
              "default table: the guard is STILL never true -- the fallback was unreachable here too");
        check(b.disagreed == 0,
              "default table: the ternary already returned the table arm (BF16), not the pool dtype");
    }

    // ---------------------------------------------------------------- C
    std::printf("\n--- C. source guard over the real files, CODE lines only (this part CAN fail) ---\n");
    const std::string cpp_path = root + "/src/targets/qwen3_6/impl/state/decoder_state.cpp";
    const std::string h_path =
        root + "/src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h";
    std::string cpp, hdr;
    const bool have_cpp = read_file(cpp_path, &cpp);
    const bool have_hdr = read_file(h_path, &hdr);
    std::printf("  cpp: %s (%s, %zu B)\n", cpp_path.c_str(), have_cpp ? "read" : "MISSING", cpp.size());
    std::printf("  hdr: %s (%s, %zu B)\n", h_path.c_str(), have_hdr ? "read" : "MISSING", hdr.size());
    check(have_cpp, "decoder_state.cpp was readable (a guard that cannot read its subject must fail)");
    check(have_hdr, "decoder_state.h was readable (same reason)");

    if (have_cpp && have_hdr) {
        // C1. The PREMISE: the four declarations must still be fixed-size std::array<..., 64>.
        //     If a table ever becomes a vector or a span, `empty()` stops being a constant and
        //     this guard's conclusion changes -- so it must say so rather than stay green.
        const char* decls[4] = {
            "std::array<DType, 64> layer_dtypes{};",
            "std::array<bool, 64> layer_residual{};",
            "std::array<std::uint32_t, 64> layer_sliding_windows{};",
            "std::array<std::uint32_t, 64> layer_plane_base{};",
        };
        for (const char* d : decls) {
            char msg[320];
            std::snprintf(msg, sizeof(msg), "PREMISE: decoder_state.h still declares \"%s\"", d);
            check(hdr.find(d) != std::string::npos, msg);
        }

        // C2. THE GUARD: no CODE line may test one of the four members with `.empty()`.
        const char* members[4] = {"layer_dtypes_", "layer_residual_", "layer_sliding_windows_",
                                  "layer_plane_base_"};
        std::vector<std::string> offenders;
        for (const char* m : members) {
            for (const std::string& o : offenders_in(cpp, m, "decoder_state.cpp")) {
                offenders.push_back(o);
            }
            for (const std::string& o : offenders_in(hdr, m, "decoder_state.h")) {
                offenders.push_back(o);
            }
        }
        char msg[320];
        std::snprintf(msg, sizeof(msg),
                      "no CODE line calls `.empty()` on the four std::array member tables "
                      "(%zu offender(s))",
                      offenders.size());
        check(offenders.empty(), msg);
        for (const std::string& o : offenders) { std::printf("      OFFENDER: %s\n", o.c_str()); }

        // C2b. The stale comment at decoder_state.cpp:906 treated `.empty()` as a semantic
        //      signal ("they follow the same V codec rule through layer_dtypes_.empty()").
        //      Comments are skipped by C2, so this one is checked on its own terms: the
        //      sentence must no longer ASSERT that the rule goes through `.empty()`.
        check(cpp.find("they follow the same V codec rule through layer_dtypes_.empty()") ==
                  std::string::npos,
              "the MTP-comment no longer asserts the V codec rule goes through `.empty()`");

        // C3. THE REASON MUST STAY DOCUMENTED. If someone deletes the explanation, the next
        //     reader has no way to know the fallback was unsound and unreachable -- and the
        //     whole point of this exercise is that a removed dead branch must say what it
        //     meant. This is a documentation-presence guard, and it can fail.
        check(hdr.find("NO \"table absent\" state") != std::string::npos,
              "decoder_state.h still documents that there is NO \"table absent\" state");
        check(cpp.find("See the\n    // note on the table declarations in decoder_state.h") !=
                      std::string::npos ||
                  cpp.find("note on the table declarations in decoder_state.h") != std::string::npos,
              "decoder_state.cpp still points at that note");

        // C4. The REAL bound must still be there: `layer < layers_` is what replaced the
        //     guard on the two tables that needed a bound at all.
        check(cpp.find("layer < layers_ && layer_residual_[layer]") != std::string::npos &&
                  cpp.find("layer < layers_ ? layer_sliding_windows_[layer] : 0U") !=
                      std::string::npos,
              "layer_view()/batch_layer_view() keep the REAL bound (`layer < layers_`)");

        // C5. THE CONTRAST THAT MAKES THIS A DEFECT RATHER THAN AN IDIOM, and the reason C2
        //     keys on the trailing-underscore MEMBER names and not on the call shape.
        //     plan_cache() takes the same tables as `std::span` PARAMETERS, where `.empty()`
        //     is a REAL query and the fallback IS reachable -- and those guards must survive,
        //     because deleting them would be the opposite mistake. Measured on this file:
        //       decoder_state.cpp:296,299,303,306,311,320,363  <- spans, REAL, must stay
        //       decoder_state.cpp:1017..1114 (16 sites)        <- std::array members, constant
        //     Same idiom, same names, opposite truth; only the TYPE distinguishes them.
        check(cpp.find("if (!layer_dtypes.empty() && layer_dtypes.size() < layers)") !=
                      std::string::npos &&
                  cpp.find("return !layer_residual.empty() && layer_residual[layer];") !=
                      std::string::npos &&
                  cpp.find("const DType override_dtype = layer_dtypes.empty() ? DType::BF16 : "
                           "layer_dtypes[layer];") != std::string::npos &&
                  cpp.find("window_flags[layer]  = layer_windows.empty() ? 0U : layer_windows[layer];") !=
                      std::string::npos,
              "plan_cache()'s SPAN-based guards (where empty() is REAL) are untouched");
    }

    std::printf("\n== %d/%d ==\n", g_ok, g_total);
    return g_ok == g_total ? 0 : 1;
}
