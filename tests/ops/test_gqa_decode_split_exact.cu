// ninfer::tests - the split-geometry determinism contract of the split-KV small-T GQA decode
// (ops/kernel/gqa_attention_decode.cuh): the required-but-untested duty recorded as
// "verify must partition like the batch-1 decode".
//
// THE CONTRACT (src/targets/qwen3_6/impl/runtime/program_impl.h:602-605, verbatim):
//   "The split reference is pinned to the sequence capacity, not to the per-round frontier + k:
//    the verify columns must reduce exactly like the batch-1 decode of the same row, so their
//    split grid may not depend on the draft window or on how far the sequence has advanced."
//
// WHAT IS ASSERTED, and why each check can go red:
//
//   [1] GEOMETRY PARITY (exact policy). For every window W in the set below, every geometry,
//       every KV dtype (Int8 == true is the i8/nvfp4 branch that carried the
//       `tokens == 5 / tokens == 6` specializations) and both key blocks (Bc 32/64), the tuple
//       (active_split_count, start, limit) the partial kernel derives for split s must be
//       BIT-IDENTICAL for TokenTile == 1 and TokenTile == 6.
//       NEGATIVE CONTROL: the same tuple under the legacy policy (exact == 0) must DIFFER for
//       at least one (W, dtype) in the set -- otherwise the check could not tell a working fix
//       from a no-op. The legacy policy is where the tokens == 5 / 6 cases live, so the control
//       is expected to fire on the I8 windows in (128, 512] and (5000, 8198].
//
//   [2] REDUCER WINDOW PARITY. The visible-key count the reducer uses for one column must be the
//       column's OWN, never the launch's last column, so a 6-column verify chunk reduces a row
//       exactly as a 1-column decode of that row does. NEGATIVE CONTROL: under the legacy policy
//       the count is the launch's last column, which differs from the column's own for every
//       column but the last -- asserted, so the check cannot pass vacuously.
//
//   [3] THE COMBINE ORDER IS COMPILE-TIME FIXED. The per-column partial combination must not take
//       its shape from blockDim.x (which the launcher's (TOKENS, WARPS) template dispatch picks
//       per verify width) and must not walk splits with a blockDim.x stride. This check reads the
//       kernel header AS TEXT -- deliberately not including it, the same technique
//       ninfer_fnv_convention_test uses -- so a later edit that reintroduces
//       `stride = blockDim.x / 2` or `split += blockDim.x` inside the reducer turns it red.
//
// This test launches NO kernel and needs no GPU: every producer above is __host__ __device__, so
// the geometry the kernels would select is evaluated on the host. It deliberately links no ninfer
// library (only CUDA::cudart) so it keeps building and running while the engine's device link
// does not. __CUDACC_RDC__ is undefined in this TU, so the strict-mode device symbol does not
// exist here and `exact` is passed explicitly: 1 = the NINFER_VERIFY_EXACT=1 policy, 0 = the
// legacy policy the same binary keeps when the flag is unset.

#include "ops/kernel/gqa_attention_decode.cuh"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// The producers under test all live in ninfer::ops (ops/kernel/gqa_attention_decode.cuh).
using namespace ninfer::ops;

int g_checks   = 0;
int g_failures = 0;

void expect(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL  %s\n", what.c_str());
    }
}

struct SplitRanges {
    int active = 0;
    std::vector<int> start;
    std::vector<int> limit;

    bool operator==(const SplitRanges& other) const {
        return active == other.active && start == other.start && limit == other.limit;
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

// Every split the launch can dispatch, in launch order -- the partition the partial kernel
// writes and the reducer reads, as one comparable value.
template <typename Geometry, bool Int8>
SplitRanges split_partition(int window, int split_count, int split_units, int tokens,
                            int key_block, int exact) {
    SplitRanges out;
    const GqaSmallTSplitRange first = gqa_small_t_split_range<Geometry, Int8>(
        window, split_count, split_units, tokens, key_block, 0, exact);
    out.active = first.active;
    for (int split = 0; split < first.active; ++split) {
        const GqaSmallTSplitRange r = gqa_small_t_split_range<Geometry, Int8>(
            window, split_count, split_units, tokens, key_block, split, exact);
        out.start.push_back(r.start);
        out.limit.push_back(r.limit);
    }
    return out;
}

// The windows the contract is stated over: both sides of every boundary the split policy has
// (64/128/160/256/512/4096/5000/8198/16390), the 100-key floor, a 32-key tile boundary, and a
// long window well past the last tier.
const int kWindows[] = {100, 129, 160, 300, 2048, 2055, 8198, 8199, 16390, 16391, 131072};

// The two split references the acceptance matrix runs with: --kv-capacity 114688 (bf16 domain)
// and 262144 (nvfp4/mixed domain).
const int kReferences[] = {114688, 262144};

const std::string kHeaderPath = std::string(NINFER_SOURCE_DIR) +
                                "/src/ops/kernel/gqa_attention_decode.cuh";

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

// Extracts the reducer's body: from the reduce_output kernel's opening line to the end of the
// function. Text work only: the point is to pin the SHAPE of the combine, which no host-side
// call can observe.
std::string reducer_body(const std::string& text) {
    const std::string needle = "gqa_attention_small_t_reduce_output_kernel";
    const std::size_t at     = text.find(needle);
    if (at == std::string::npos) { return std::string(); }
    return text.substr(at);
}

template <typename Geometry, bool Int8>
void geometry_parity(const char* geometry_name, int reference, int split_count, int key_block) {
    // Both partition regimes are checked. `split_units > 0` is the pinned (production) grid, in
    // which the token tile could only reach the count through the active-split clamp;
    // `split_units == 0` is the window-driven grid, and that is the regime the
    // `tokens == 5 / tokens == 6` legacy cases live in -- hence the negative control below runs
    // there, where the dependence is expected to be visible.
    const int regimes[] = {gqa_small_t_split_units<Geometry>(reference), 0};

    for (const int split_units : regimes) {
        bool legacy_differed = false;
        for (const int window : kWindows) {
            // One column: the batch-1 decode selects TokenTile == 1.
            const SplitRanges decode = split_partition<Geometry, Int8>(
                window, split_count, split_units, 1, key_block, 1 /*exact*/);
            for (int tokens = 2; tokens <= 6; ++tokens) {
                const SplitRanges verify = split_partition<Geometry, Int8>(
                    window, split_count, split_units, tokens, key_block, 1 /*exact*/);
                std::ostringstream os;
                os << geometry_name << (Int8 ? " Int8" : " bf16") << " Bc=" << key_block
                   << " ref=" << reference << " split_units=" << split_units << " W=" << window
                   << " T=1 vs T=" << tokens;
                expect(decode == verify, os.str() + "\n        T=1  : " + decode.str() +
                                             "\n        T=" + std::to_string(tokens) + "  : " +
                                             verify.str());
            }

            // The legacy policy is where the token-tile dependence lives; the control below
            // proves this test can see it.
            const SplitRanges legacy_decode = split_partition<Geometry, Int8>(
                window, split_count, split_units, 1, key_block, 0 /*legacy*/);
            for (int tokens = 2; tokens <= 6; ++tokens) {
                const SplitRanges legacy_verify = split_partition<Geometry, Int8>(
                    window, split_count, split_units, tokens, key_block, 0 /*legacy*/);
                if (!(legacy_decode == legacy_verify)) { legacy_differed = true; }
            }
        }

        std::ostringstream os;
        os << "NEGATIVE CONTROL: legacy policy must show the token-tile dependence "
           << geometry_name << (Int8 ? " Int8" : " bf16") << " Bc=" << key_block
           << " ref=" << reference << " split_units=" << split_units;
        // The control is only stated where the dependence can exist at all. The legacy
        // token-tile cases (`tokens == 5 / tokens == 6`) are gated on `if constexpr (Int8)`, and
        // they live in the window-driven regime (`split_units == 0`) -- with a pinned grid,
        // split_units > 0 short-circuits the count for every dtype. Asserting the control
        // elsewhere would be asserting that a correct policy is wrong.
        if (Int8 && split_units == 0) {
            expect(legacy_differed, os.str());
        } else {
            expect(!legacy_differed,
                   "no token-tile dependence is expected here: " + os.str());
        }
    }
}

void reducer_window_parity() {
    const int references[] = {114688, 262144};
    bool legacy_differed   = false;
    for (const int reference : references) {
        const int split_units = gqa_small_t_split_units<ninfer::ops::Gqa27Geometry>(reference);
        for (const int window : kWindows) {
            const int last_pos = window - 1;
            for (int tokens = 1; tokens <= 6; ++tokens) {
                for (int token = 0; token < tokens; ++token) {
                    const int own_pos = last_pos - (tokens - 1) + token;
                    const int exact_window =
                        gqa_small_t_reduce_window(last_pos, own_pos, split_units, 1 /*exact*/,
                                                 0 /*no sliding window*/);
                    // The column's own count, and nothing else: identical for a 1-column decode
                    // and for a 6-column chunk, for every column of the chunk.
                    std::ostringstream os;
                    os << "reducer window (exact) ref=" << reference << " W=" << window
                       << " tokens=" << tokens << " token=" << token;
                    expect(exact_window == own_pos + 1, os.str());

                    const int legacy_window =
                        gqa_small_t_reduce_window(last_pos, own_pos, split_units, 0 /*legacy*/,
                                                  0 /*no sliding window*/);
                    if (legacy_window != exact_window) { legacy_differed = true; }
                }
            }
        }
    }
    expect(legacy_differed,
           "NEGATIVE CONTROL: the legacy reducer window must move with the token tile");
}

void combine_order_is_compile_time_fixed() {
    const std::string text = read_file(kHeaderPath);
    if (text.empty()) { expect(false, "kernel header readable: " + kHeaderPath); return; }
    const std::string body = reducer_body(text);
    expect(!body.empty(), "reducer body found in " + kHeaderPath);
    if (body.empty()) { return; }

    expect(body.find("blockDim.x / 2") == std::string::npos,
           "reducer tree stride must not be blockDim.x / 2 (the (TOKENS, WARPS) dispatch "
           "would change the fp32 association)");
    expect(body.find("split += blockDim.x") == std::string::npos,
           "reducer split walk must not step by blockDim.x");
    expect(body.find("stride = kSlots / 2") != std::string::npos,
           "reducer tree stride must be the compile-time slot count kSlots / 2");
    expect(body.find("kGqaReduceSlots") != std::string::npos,
           "reducer must name the compile-time slot count kGqaReduceSlots");
    expect(static_cast<int>(ninfer::ops::kGqaReduceSlots) == 256,
           "kGqaReduceSlots must equal the reducer's launch block (256)");

    // The five tier kernels must share ONE key-range definition; five inlined spellings are how
    // the legacy tiling could have grown a variant (item 3 of the fix).
    // The 5th entry is the FILE stem, not the product token. The ISO4E tier's kernel
    // file is src/ops/kernel/gqa_attention_decode_iso3.cuh (`iso3` is the deprecated
    // alias of the product token `iso4e`, and the file kept the old stem). Spelling it
    // "_iso4e" read a path that does not exist, so for that tier the readability check
    // failed and the two checks below were SKIPPED -- i.e. the coverage was vacuous.
    const char* tiers[] = {"_bf16", "_i8", "_nvfp4", "_fp8", "_iso3"};
    for (const char* tier : tiers) {
        const std::string path = std::string(NINFER_SOURCE_DIR) + "/src/ops/kernel/"
                              + "gqa_attention_decode" + tier + ".cuh";
        const std::string tier_text = read_file(path);
        expect(!tier_text.empty(), "tier header readable: " + path);
        if (tier_text.empty()) { continue; }
        expect(tier_text.find("gqa_small_t_split_range<Geometry,") != std::string::npos,
               std::string("tier ") + tier + " must use gqa_small_t_split_range");
        expect(tier_text.find("units_per_split") == std::string::npos,
               std::string("tier ") + tier + " must not spell the window-driven tiling itself");
    }
}

} // namespace

int main() {
    using namespace ninfer::ops;

    for (const int reference : kReferences) {
        const int cap27 = Gqa27Geometry::DecodeSplits;
        const int cap35 = Gqa35Geometry::DecodeSplits;
        geometry_parity<Gqa27Geometry, false>("Gqa27", reference, cap27, 32);
        geometry_parity<Gqa27Geometry, true>("Gqa27", reference, cap27, 32);
        geometry_parity<Gqa27Geometry, true>("Gqa27", reference, cap27, 64);
        geometry_parity<Gqa35Geometry, false>("Gqa35", reference, cap35, 32);
        geometry_parity<Gqa35Geometry, true>("Gqa35", reference, cap35, 64);
    }
    reducer_window_parity();
    combine_order_is_compile_time_fixed();

    std::printf("ninfer_gqa_decode_split_exact_test: %d checks, %d failures\n", g_checks,
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
