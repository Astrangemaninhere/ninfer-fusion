// ninfer::tests - RUNTIME geometry probe for the split-KV small-T GQA decode (FIX-B).
//
// WHY THIS EXISTS. tests/ops/test_gqa_decode_split_exact.cu pins the contract but only compiles
// alongside the fix, so it cannot show the defect. This program CAN: it is one binary that is run
// twice, once with NINFER_VERIFY_EXACT unset and once with NINFER_VERIFY_EXACT=1, and it prints,
// to stderr, how many (window, token-tile) pairs give a batch-1 decode a DIFFERENT split geometry
// than the verify chunk that replaces it. With the flag off the count is non-zero; with the flag
// on it is 0. Same binary, two env states -- that is the 2x2 cell for the kernel-side flag.
//
// It needs NO GPU and takes NO build lock: `exact` is derived from the environment on the host and
// passed explicitly to gqa_small_t_active_splits_exact / gqa_small_t_split_range, which are
// __host__ __device__ -- and that is exactly the value the kernels see, because the device mirrors
// the same environment read through gqa_verify_exact_mode() (see the toggle block in
// ops/kernel/gqa_attention_decode.cuh for how the value is published to device memory, and
// ops/launcher/gqa_attention_decode_smallt.cu:62 for where the publish is triggered).
//
// Build and run (deliberately outside the project's build system: it must never need the shared
// build lock, and it must stay runnable while the engine's device link does not):
//
//   nvcc -std=c++20 -O1 -arch=sm_120 -I tests -I include -I src -I third_party \
//        -I third_party/utf8proc -o /tmp/gqa_geom_probe \
//        tests/ops/gqa_split_geometry_probe.cu
//   /tmp/gqa_geom_probe ; echo "rc=$?"                     # NINFER_VERIFY_EXACT unset
//   NINFER_VERIFY_EXACT=1 /tmp/gqa_geom_probe ; echo "rc=$?" # exact policy
//
// TWO REGIMES, because they answer different questions:
//   * PINNED  (split_units = gqa_small_t_split_units(capacity), which is what every launcher of
//     this family passes): the partition is [s*split_units, (s+1)*split_units) outright, so the
//     token tile can only reach the count through the active-split clamp. Predicted mismatch
//     count 0 in BOTH flag states -- this is the regime production runs in.
//   * WINDOW  (split_units == 0, the legacy window-driven partition): this is where the
//     `tokens == 5 / tokens == 6` cases live. Mismatch count > 0 with the flag OFF, 0 with it ON.
//
// Exit code: 0 when the observed counts agree with the flag state's prediction, 1 when the flag
// is ON and a mismatch survives (the fix did not hold), 2 when the flag is OFF and the WINDOW
// regime shows no mismatch at all (the probe cannot see the defect it exists to show), and 3 when
// the PINNED regime shows a mismatch with the flag OFF -- which would falsify the analysis in
// PATCHSET/FIX-B/REPORT.md section 3 and must be reported loudly.

#include "ops/kernel/gqa_attention_decode.cuh"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::ops;

struct Geometry {
    int active = 0;
    std::vector<int> start;
    std::vector<int> limit;
    bool operator==(const Geometry& o) const {
        return active == o.active && start == o.start && limit == o.limit;
    }
    std::string str() const {
        std::ostringstream os;
        os << "active=" << active << " [";
        for (std::size_t i = 0; i < start.size(); ++i) {
            os << (i ? " " : "") << start[i] << ".." << limit[i];
        }
        os << "]";
        return os.str();
    }
};

template <typename Geometry_, bool Int8>
Geometry geometry_of(int window, int split_count, int split_units, int tokens, int key_block,
                     int exact) {
    Geometry g;
    const GqaSmallTSplitRange first = gqa_small_t_split_range<Geometry_, Int8>(
        window, split_count, split_units, tokens, key_block, 0, exact);
    g.active = first.active;
    for (int split = 0; split < first.active; ++split) {
        const GqaSmallTSplitRange r = gqa_small_t_split_range<Geometry_, Int8>(
            window, split_count, split_units, tokens, key_block, split, exact);
        g.start.push_back(r.start);
        g.limit.push_back(r.limit);
    }
    return g;
}

const int kWindows[] = {100, 129, 160, 300, 512, 2048, 2055, 3000, 5001, 8198, 8199, 16390, 16391,
                        65536, 114688, 131072, 262144};
// The reference the launcher derives every split constant from: --kv-capacity 114688 (bf16
// domain) or 262144 (nvfp4/mixed domain). `split_reference_keys` is 0 on the eager calls, so
// gqa_small_t_split_reference falls back to max_visible_keys -- pinning the reference is what the
// whole contract rests on.
const int kReference = 262144;

struct RegimeResult {
    int checked   = 0;
    int mismatches = 0;
    std::string first;
};

RegimeResult scan(const char* name, int split_units, int exact, bool verbose) {
    RegimeResult out;
    using G  = Gqa27Geometry;   // 24 q-heads / 4 kv-heads, DecodeSplits 85 -- the 27B GQA shape
    constexpr bool kInt8 = true; // the branch that carried tokens == 5 / tokens == 6
    const int cap = G::DecodeSplits;
    for (const int kb : {32, 64}) {
        for (const int window : kWindows) {
            const Geometry decode = geometry_of<G, kInt8>(window, cap, split_units, 1, kb, exact);
            for (const int tokens : {5, 6}) {
                const Geometry verify =
                    geometry_of<G, kInt8>(window, cap, split_units, tokens, kb, exact);
                ++out.checked;
                if (!(decode == verify)) {
                    ++out.mismatches;
                    if (out.first.empty()) {
                        std::ostringstream os;
                        os << "regime=" << name << " Bc=" << kb << " W=" << window
                           << " split_units=" << split_units << " decode(T=1) " << decode.str()
                           << "  vs  verify(T=" << tokens << ") " << verify.str();
                        out.first = os.str();
                    }
                    if (verbose) {
                        std::fprintf(stderr,
                                     "  MISMATCH regime=%s Bc=%d W=%d T=%d  decode: %s\n"
                                     "                                      verify: %s\n",
                                     name, kb, window, tokens, decode.str().c_str(),
                                     verify.str().c_str());
                    }
                }
            }
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const bool verbose = argc > 1 && std::string(argv[1]) == "--verbose";
    const char* env    = std::getenv("NINFER_VERIFY_EXACT");
    const int exact    = (env != nullptr && env[0] == '1' && env[1] == '\0') ? 1 : 0;

    const int pinned = gqa_small_t_split_units<ninfer::ops::Gqa27Geometry>(kReference);
    std::fprintf(stderr, "=== GQA split-geometry probe ===\n");
    std::fprintf(stderr, "NINFER_VERIFY_EXACT=%s -> exact=%d ; split_reference=%d -> split_units=%d\n",
                 env != nullptr ? env : "(unset)", exact, kReference, pinned);

    const RegimeResult pin = scan("pinned", pinned, exact, verbose);
    const RegimeResult win = scan("window", 0, exact, verbose);

    std::fprintf(stderr, "pinned regime (split_units=%d): pairs=%d mismatches=%d\n", pinned,
                 pin.checked, pin.mismatches);
    std::fprintf(stderr, "window regime (split_units=0)    : pairs=%d mismatches=%d\n",
                 win.checked, win.mismatches);
    if (!pin.first.empty()) { std::fprintf(stderr, "first pinned mismatch: %s\n", pin.first.c_str()); }
    if (!win.first.empty()) { std::fprintf(stderr, "first window mismatch: %s\n", win.first.c_str()); }
    std::fprintf(stderr, "RESULT exact=%d pinned_mismatches=%d window_mismatches=%d\n", exact,
                 pin.mismatches, win.mismatches);

    if (exact != 0) {
        if (pin.mismatches != 0 || win.mismatches != 0) {
            std::fprintf(stderr, "VERDICT: FAIL -- NINFER_VERIFY_EXACT=1 must leave no token-tile "
                                 "dependence in either regime\n");
            return 1;
        }
        std::fprintf(stderr, "VERDICT: PASS -- flag ON, no token-tile dependence in either regime\n");
        return 0;
    }
    if (win.mismatches == 0) {
        std::fprintf(stderr, "VERDICT: PROBE CANNOT SEE THE DEFECT -- the window regime shows no "
                             "token-tile dependence at all\n");
        return 2;
    }
    if (pin.mismatches != 0) {
        std::fprintf(stderr,
                     "VERDICT: PREDICTION FALSIFIED -- the PINNED regime shows a token-tile "
                     "dependence with the flag OFF. REPORT.md section 3 says it cannot; report "
                     "this immediately.\n");
        return 3;
    }
    std::fprintf(stderr,
                 "VERDICT: DEFECT VISIBLE (flag OFF) -- window regime shows %d token-tile "
                 "dependent (window, tile) pairs; the pinned regime shows 0, as predicted.\n",
                 win.mismatches);
    return 0;
}
