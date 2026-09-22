// e8_scale_convention.cpp -- HOST-ONLY. Which normalization is right for the e8 K plane:
// the codec's amax/2^(w-1) with the two's-complement range, or the shipped row's
// amax/(2^(w-1)-1) with the symmetric-clamp range?  Decided by measurement on the
// ENGINE'S OWN writer domain (real K, Hadamard-rotated inside one head's 64-group),
// not by argument.
//
//   g++ -O2 -std=c++17 -I/home/user/ninfer-fusion/src \
//       tools/e8_verify/e8_scale_convention.cpp -o /tmp/e8sc
//   /tmp/e8sc <kvsrc_prefix> <token_cap>
//
// Conventions compared, per width w (the codec's three widths):
//   B  (SHIPPED style)  divisor 2^(w-1)-1   range [-(2^(w-1)-1), 2^(w-1)-1]   7 -> 7
//   A  (CODEC)          divisor 2^(w-1)     range [-2^(w-1), 2^(w-1)-1]       8 -> -8..7
//   A' (DIAGNOSTIC)     divisor 2^(w-1)     range [-(2^(w-1)-1), 2^(w-1)-1]   8 -> -7..7
// A' isolates whether the benefit/penalty of A lives in its DIVISOR or in its RANGE.
//
// The scale is stored the way each writer stores it: fp16. This harness uses the codec's
// own e8_kv_fp16_bits/e8_kv_fp16_from_bits pair for BOTH conventions so the comparison
// cannot be an artifact of two different fp16 roundings. It also prints the float-scale
// (no fp16) numbers, which isolate the convention from the storage rounding.

#include "product/kv_e8_width.h"
#include "product/kv_e8_width_codec.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer::product;

namespace {

// The tree's own Hadamard-64 butterfly (src/ops/kernel/e8_lattice.cuh), copied verbatim
// from tools/e8_verify/e8_lattice_math.cpp:321-338 so this harness quantizes the SAME
// writer domain the engine does.
void hadamard64(float v[64]) {
    float x0[32], x1[32], t0[32], t1[32];
    for (int i = 0; i < 32; ++i) { x0[i] = v[i]; x1[i] = v[i + 32]; }
    for (int off = 1; off < 32; off <<= 1) {
        for (int i = 0; i < 32; ++i) {
            const float y0 = x0[i ^ off], y1 = x1[i ^ off];
            const bool hi = (i & off) != 0;
            t0[i] = hi ? y0 - x0[i] : x0[i] + y0;
            t1[i] = hi ? y1 - x1[i] : x1[i] + y1;
        }
        for (int i = 0; i < 32; ++i) { x0[i] = t0[i]; x1[i] = t1[i]; }
    }
    for (int i = 0; i < 32; ++i) {
        const float a = x0[i], b = x1[i];
        v[i] = (a + b) * 0.125f;
        v[i + 32] = (a - b) * 0.125f;
    }
}

struct Conv {
    const char* name;
    int divisor;   // amax / divisor
    int lo;
    int hi;
};

struct Res {
    double se = 0.0;      // sum of squared error over elements
    double sw = 0.0;      // sum of squares of the input
    std::uint64_t n = 0;
    std::uint64_t clamped = 0;   // elements whose rounded quotient hit a bound
    std::uint64_t nonint = 0;    // (unused)
};

// Quantize one 64-group under one convention and accumulate.
void run_group(const Conv& c, const float* g, std::int32_t n, bool fp16, Res* r) {
    float amax = 0.0f;
    for (std::int32_t i = 0; i < n; ++i) { const float a = std::fabs(g[i]); if (a > amax) { amax = a; } }
    if (!(amax > 0.0f)) { return; }
    const float s_f = amax / static_cast<float>(c.divisor);
    const float s = fp16 ? e8_kv_fp16_from_bits(e8_kv_fp16_bits(s_f)) : s_f;
    if (!(s > 0.0f)) { return; }
    for (std::int32_t i = 0; i < n; ++i) {
        const float q = std::nearbyint(g[i] / s);
        const float qc = q < static_cast<float>(c.lo)   ? static_cast<float>(c.lo)
                         : (q > static_cast<float>(c.hi) ? static_cast<float>(c.hi) : q);
        if (qc != q) { ++r->clamped; }
        const double e = static_cast<double>(qc) * static_cast<double>(s) -
                         static_cast<double>(g[i]);
        r->se += e * e;
        r->sw += static_cast<double>(g[i]) * static_cast<double>(g[i]);
        ++r->n;
    }
}

void report(const char* label, const Conv& c, const std::vector<float>& groups, std::int32_t n,
            const Res& base, bool print_clamp) {
    Res r;
    for (std::size_t g = 0; g + static_cast<std::size_t>(n) <= groups.size(); g += static_cast<std::size_t>(n)) {
        run_group(c, groups.data() + g, n, true, &r);
    }
    const double mse = r.n ? r.se / static_cast<double>(r.n) : 0.0;
    const double bmse = base.n ? base.se / static_cast<double>(base.n) : 0.0;
    const double rel = r.sw > 0.0 ? std::sqrt(r.se / r.sw) : 0.0;
    std::printf("  %-26s div=%-2d range [%3d,%3d]  MSE %.6e  relL2 %7.4f%%  MSE/B %.6f  %+7.4f dB",
                label, c.divisor, c.lo, c.hi, mse, 100.0 * rel, bmse > 0.0 ? mse / bmse : 0.0,
                bmse > 0.0 && mse > 0.0 ? 10.0 * std::log10(bmse / mse) : 0.0);
    if (print_clamp) {
        std::printf("  clamped %.4f%%", 100.0 * static_cast<double>(r.clamped) /
                                         static_cast<double>(r.n ? r.n : 1));
    }
    std::printf("\n");
    // The same convention with an EXACT float scale, i.e. without the fp16 storage
    // rounding. If the verdict flips here, the verdict is about fp16, not the convention.
    Res rf;
    for (std::size_t g = 0; g + static_cast<std::size_t>(n) <= groups.size(); g += static_cast<std::size_t>(n)) {
        run_group(c, groups.data() + g, n, false, &rf);
    }
    const double fmse = rf.n ? rf.se / static_cast<double>(rf.n) : 0.0;
    std::printf("  %-26s (float scale)          MSE %.6e  relL2 %7.4f%%  MSE/B %.6f  %+7.4f dB\n",
                label, fmse, 100.0 * (rf.sw > 0.0 ? std::sqrt(rf.se / rf.sw) : 0.0),
                bmse > 0.0 ? fmse / bmse : 0.0,
                bmse > 0.0 && fmse > 0.0 ? 10.0 * std::log10(bmse / fmse) : 0.0);
}

// Pack one 64-group's codes under a convention, 4-bit two's complement, nibble = c & 0xF.
void pack_group4(const Conv& c, const float* g, std::int32_t n, std::uint8_t* out, int* nbytes) {
    float amax = 0.0f;
    for (std::int32_t i = 0; i < n; ++i) { const float a = std::fabs(g[i]); if (a > amax) { amax = a; } }
    const float s = e8_kv_fp16_from_bits(e8_kv_fp16_bits(amax / static_cast<float>(c.divisor)));
    *nbytes = n / 2;
    std::memset(out, 0, static_cast<std::size_t>(*nbytes));
    for (std::int32_t i = 0; i < n; ++i) {
        float q = std::nearbyint(g[i] / s);
        if (q < static_cast<float>(c.lo)) { q = static_cast<float>(c.lo); }
        if (q > static_cast<float>(c.hi)) { q = static_cast<float>(c.hi); }
        const unsigned code = static_cast<unsigned>(static_cast<int>(q)) & 0x0Fu;
        out[i / 2] |= static_cast<std::uint8_t>(code << ((i % 2) * 4));
    }
}

} // namespace

int main(int argc, char** argv) {
    std::vector<float> groups;   // 64 floats each
    const char* src_label = "synthetic";
    if (argc >= 2) {
        const std::string p = std::string(argv[1]) + "_kn.bin";
        if (std::FILE* f = std::fopen(p.c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long nb = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            std::vector<std::uint8_t> k(static_cast<std::size_t>(nb));
            const std::size_t got = std::fread(k.data(), 1, k.size(), f);
            std::fclose(f);
            if (got == k.size()) {
                const int D = 256, H = 4;
                const int T = static_cast<int>(k.size() / (2ULL * D * H));
                const int cap = (argc >= 3) ? std::atoi(argv[2]) : T;
                for (int t = 0; t < (T < cap ? T : cap); ++t) {
                    for (int h = 0; h < H; ++h) {
                        float row[256];
                        for (int d = 0; d < D; ++d) {
                            const std::size_t off = 2 * (static_cast<std::size_t>(d) + D * (h + static_cast<std::size_t>(H) * t));
                            const std::uint16_t bits = static_cast<std::uint16_t>(
                                k[off] | (static_cast<std::uint16_t>(k[off + 1]) << 8));
                            const std::uint32_t f32 = static_cast<std::uint32_t>(bits) << 16;
                            std::memcpy(&row[d], &f32, 4);
                        }
                        for (int g = 0; g < 4; ++g) {
                            float blk[64];
                            for (int i = 0; i < 64; ++i) { blk[i] = row[g * 64 + i]; }
                            hadamard64(blk);
                            for (int i = 0; i < 64; ++i) { groups.push_back(blk[i]); }
                        }
                    }
                }
                src_label = argv[1];
            } else {
                std::printf("could not read %s\n", p.c_str());
            }
        } else {
            std::printf("no such file: %s  -- synthetic only\n", p.c_str());
        }
    }
    // A deterministic synthetic control so the real-K answer can be seen next to a
    // distribution with no engine in it.
    if (groups.empty()) {
        std::uint32_t st = 0x2545F491u;
        auto nx = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
        for (int g = 0; g < 4096; ++g) {
            float acc[64];
            for (int i = 0; i < 64; ++i) {
                float s = 0.0f;
                for (int z = 0; z < 4; ++z) { s += (static_cast<float>(nx() & 0xFFFFFFu) / 8388608.0f) - 1.0f; }
                acc[i] = s * 0.5f;
            }
            for (int k = 0; k < 2; ++k) { acc[nx() % 64u] *= 12.0f; }
            for (int i = 0; i < 64; ++i) { groups.push_back(acc[i]); }
        }
        src_label = "synthetic (4-sum uniform + 2 outliers)";
    }

    const std::size_t n_groups = groups.size() / 64;
    std::printf("=== e8 scale convention: codec (amax/2^(w-1)) vs shipped (amax/7) ===\n");
    std::printf("data   : %s\n", src_label);
    std::printf("groups : %zu  (64 floats each, rotated writer domain)  elements: %zu\n\n",
                n_groups, groups.size());

    std::printf("--- 0. what each width's two conventions actually ARE ---\n");
    std::printf("  w=4: SHIPPED amax/7 range [-7,7]   | CODEC amax/8 range [-8,7]\n");
    std::printf("  w=3: SHIPPED amax/3 range [-3,3]   | CODEC amax/4 range [-4,3]\n");
    std::printf("  w=2: SHIPPED amax/1 range [-1,1]   | CODEC amax/2 range [-2,1]\n\n");

    struct W {
        int w;
        Conv shipped;
        Conv codec;
        Conv diag;
    };
    const W tab[3] = {
        {4, {"B shipped w=4", 7, -7, 7}, {"A codec w=4", 8, -8, 7}, {"A' codec-div, sym clamp w=4", 8, -7, 7}},
        {3, {"B shipped-style w=3", 3, -3, 3}, {"A codec w=3", 4, -4, 3}, {"A' codec-div, sym clamp w=3", 4, -3, 3}},
        {2, {"B shipped-style w=2", 1, -1, 1}, {"A codec w=2", 2, -2, 1}, {"A' codec-div, sym clamp w=2", 2, -1, 1}},
    };

    for (const W& t : tab) {
        std::printf("--- w=%d ---\n", t.w);
        Res base;
        for (std::size_t g = 0; g < groups.size(); g += 64) { run_group(t.shipped, groups.data() + g, 64, true, &base); }
        const double bmse = base.n ? base.se / static_cast<double>(base.n) : 0.0;
        std::printf("  %-26s div=%-2d range [%3d,%3d]  MSE %.6e  relL2 %7.4f%%  MSE/B %.6f  %+7.4f dB  (BASELINE)\n",
                    t.shipped.name, t.shipped.divisor, t.shipped.lo, t.shipped.hi, bmse,
                    100.0 * std::sqrt(base.se / base.sw), 1.0, 0.0);
        report(t.codec.name, t.codec, groups, 64, base, true);
        report(t.diag.name, t.diag, groups, 64, base, true);
        std::printf("\n");
    }

    // --- code-plane divergence at w=4 between the two conventions ---
    std::printf("--- w=4: do the two conventions store the SAME BYTES? ---\n");
    {
        std::uint64_t same_bytes = 0, same_scale = 0, diff_elt = 0;
        std::vector<std::uint8_t> ba(64 / 2), bb(64 / 2);
        int nba = 0, nbb = 0;
        for (std::size_t g = 0; g < groups.size(); g += 64) {
            const float* grp = groups.data() + g;
            pack_group4(tab[0].shipped, grp, 64, bb.data(), &nbb);
            pack_group4(tab[0].codec, grp, 64, ba.data(), &nba);
            if (std::memcmp(ba.data(), bb.data(), static_cast<std::size_t>(nba)) == 0) { ++same_bytes; }
            for (int i = 0; i < nba; ++i) { if (ba[i] != bb[i]) { ++diff_elt; break; } }
            float amax = 0.0f;
            for (int i = 0; i < 64; ++i) { const float a = std::fabs(grp[i]); if (a > amax) { amax = a; } }
            const std::uint16_t sa = e8_kv_fp16_bits(amax / 8.0f);
            const std::uint16_t sb = e8_kv_fp16_bits(amax / 7.0f);
            if (sa == sb) { ++same_scale; }
        }
        std::printf("  groups                              : %zu\n", n_groups);
        std::printf("  identical CODE bytes                : %llu (%.4f%%)\n",
                    static_cast<unsigned long long>(same_bytes),
                    100.0 * static_cast<double>(same_bytes) / static_cast<double>(n_groups ? n_groups : 1));
        std::printf("  differing in >=1 code byte          : %llu (%.4f%%)\n",
                    static_cast<unsigned long long>(diff_elt),
                    100.0 * static_cast<double>(diff_elt) / static_cast<double>(n_groups ? n_groups : 1));
        std::printf("  identical stored fp16 SCALE         : %llu (%.4f%%)\n",
                    static_cast<unsigned long long>(same_scale),
                    100.0 * static_cast<double>(same_scale) / static_cast<double>(n_groups ? n_groups : 1));
    }

    std::printf("\n== DONE ==\n");
    return 0;
}
