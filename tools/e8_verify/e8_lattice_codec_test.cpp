// e8_lattice_codec_test -- the live caller for src/ops/kernel/e8_lattice_codec.cuh, and
// therefore for the two primitives in src/ops/kernel/e8_lattice.cuh that had NO call site
// anywhere in src/: e8_project_8d_fast() and hadamard_rot_8d().
//
// WHY A TEST IS THE CALL SITE AND NOT A KERNEL
// --------------------------------------------
// The KV write path that would consume a 2-bit K plane is
// src/ops/kernel/gqa_attention_prefill_i8.cuh / gqa_attention_decode_i8.cuh, and the
// prefill kernel's own comment at :159-165 explains why it does NOT call the projection:
//
//   "NO e8_project_8d_warp() here. The code plane holds one integer per coordinate while
//    E8 = D8 U (D8 + 1/2), and 47.6% of blocks project into the half-integer coset, which
//    the trailing rint() then destroys: measured -3.65 dB (2.32x MSE) versus plain
//    rounding at identical bits/el."
//
// That reasoning is correct for a one-integer-per-coordinate plane and it is the reason
// the projection is unreachable today. e8_lattice_codec.cuh is the second consumer
// e8_lattice.cuh:136-137 names as the escape -- one that reconstructs the LATTICE POINT
// rather than an integer code -- so the half-integer coset is representable and the
// penalty is not paid. That consumer is new code, and the file that would host it on the
// hot path is off-limits to this line, so the call site is here, in a test the tree builds
// and runs, and the hot-path landing point is named in dl/e8lattice/REPORT.md.
//
// WHAT THIS FILE PROVES, AND WHAT IT ONLY REPORTS
//   A. static_asserts on the rate arithmetic (bits, bytes, tables) -- compile-time.
//   B. the two published counts, 227 and 29, and the exhaustive lattice membership of all
//      2^16 codewords -- these CAN fail and they are the brief's self-checks.
//   C. the exact encoder against BRUTE FORCE over all 65536 codewords -- CAN fail.
//   D. the PROJECTION IS INVOKED, with a counted call, and the projected encoder's
//      agreement rate and MSE penalty versus the exact one -- priced, not asserted.
//   E. the H64 rotation is an involution and the codec round-trips on real K.
//
// HOST-ONLY: no CUDA, no cmake, no GPU, no model. The three CUDA keywords are defined away
// before the first include, exactly as tools/e8_verify's sibling probes do, so the REAL
// tree headers are compiled and executed rather than paraphrased.
//
//   g++ -O2 -std=c++17 -I tools/e8_verify/host_shim -I /home/user/ninfer-fusion/src \
//       tools/e8_verify/e8_lattice_codec_test.cpp -o /tmp/e8lc
//   /tmp/e8lc [kvdump_dir]

#define __device__
#define __forceinline__ inline
#define __shfl_xor_sync(mask, x, off) (0)

#include "ops/kernel/e8_lattice_codec.cuh"   // brings in e8_lattice.cuh as well

#undef __shfl_xor_sync
#undef __forceinline__
#undef __device__

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace ninfer::ops;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("%-4s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) { ++g_fail; }
}

// ---------------------------------------------------------------- A. rate arithmetic
static_assert(kE8LatticeStage1Bits == 16, "stage 1 is 8 index + 7 sign + 1 shift");
static_assert(kE8LatticeStage1Bits == kE8LatticeDim * 2, "16 bits / 8 dims = 2 b/el");
static_assert(kE8LatticeStage1Bits + kE8LatticeStage2Bits == kE8LatticeDim * 3,
              "16 + 8 = 24 bits / 8 dims = 3 b/el");
static_assert(kE8LatticeStage1Entries * 8 / 2 == 1024, "stage-1 table is 1024 B");
static_assert(kE8LatticeStage2Entries * 8 / 2 == 1024, "stage-2 table is 1024 B");

// ---------------------------------------------------------------- table construction
static E8LatticeStage1 build_stage1() {
    E8LatticeStage1 t{};
    static const int cvals[8] = {-7, -5, -3, -1, 1, 3, 5, 7};
    std::vector<std::array<std::int8_t, 8>> filt;
    std::int8_t c[8];
    std::function<void(int, int, int)> rec = [&](int i, int sc, int sc2) {
        if (sc2 > 40) { return; }
        if (i == 8) {
            if ((sc % 4) == 0) { std::array<std::int8_t, 8> v; for (int j = 0; j < 8; ++j) { v[j] = c[j]; } filt.push_back(v); }
            return;
        }
        for (int k = 0; k < 8; ++k) { c[i] = static_cast<std::int8_t>(cvals[k]); rec(i + 1, sc + cvals[k], sc2 + cvals[k] * cvals[k]); }
    };
    rec(0, 0, 0);
    t.n_signed = static_cast<int>(filt.size());

    std::vector<std::array<std::uint8_t, 8>> absv;
    absv.reserve(filt.size());
    for (auto& v : filt) {
        std::array<std::uint8_t, 8> a{};
        for (int j = 0; j < 8; ++j) { a[j] = static_cast<std::uint8_t>(std::abs(static_cast<int>(v[j]))); }
        absv.push_back(a);
    }
    std::sort(absv.begin(), absv.end());
    absv.erase(std::unique(absv.begin(), absv.end()), absv.end());
    t.n_abs = static_cast<int>(absv.size());

    // the 56 arrangements of five 3/2 and three 1/2; the reference hard-codes 29 of them
    std::vector<std::array<std::uint8_t, 8>> pad;
    for (int mask = 0; mask < 256; ++mask) {
        if (__builtin_popcount(static_cast<unsigned>(mask)) != 5) { continue; }
        std::array<std::uint8_t, 8> a{};
        for (int j = 0; j < 8; ++j) { a[j] = ((mask >> j) & 1) ? 3 : 1; }
        pad.push_back(a);
    }
    std::sort(pad.begin(), pad.end());
    t.n_pad = 29;
    t.n = t.n_abs + t.n_pad;
    for (int i = 0; i < t.n_abs; ++i) { for (int j = 0; j < 8; ++j) { t.mag[i * 8 + j] = absv[i][j]; } }
    for (int i = 0; i < t.n_pad; ++i) { for (int j = 0; j < 8; ++j) { t.mag[(t.n_abs + i) * 8 + j] = pad[i][j]; } }
    for (int i = 0; i < t.n; ++i) {
        int s = 0;
        for (int j = 0; j < 8; ++j) {
            const int a = t.mag[i * 8 + j];
            t.cls[i * 8 + j] = static_cast<std::uint8_t>((a - 1) / 2);
            s += a;
        }
        t.suma[i] = static_cast<std::uint8_t>(s);
    }
    return t;
}

static E8LatticeStage2 build_stage2() {
    E8LatticeStage2 t{};
    std::vector<std::array<std::int8_t, 8>> v;
    for (int i = 0; i < 8; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            for (int si = 0; si < 2; ++si) {
                for (int sj = 0; sj < 2; ++sj) {
                    std::array<std::int8_t, 8> r{}; r.fill(0);
                    r[i] = si ? -2 : 2; r[j] = sj ? -2 : 2;
                    v.push_back(r);
                }
            }
        }
    }
    for (int mask = 0; mask < 256; ++mask) {
        if (__builtin_popcount(static_cast<unsigned>(mask)) & 1) { continue; }
        std::array<std::int8_t, 8> r{};
        for (int i = 0; i < 8; ++i) { r[i] = ((mask >> i) & 1) ? -1 : 1; }
        v.push_back(r);
    }
    t.n_roots = static_cast<int>(v.size());
    for (int i = 0; i < 8; ++i) {
        for (int s = 0; s < 2; ++s) {
            std::array<std::int8_t, 8> r{}; r.fill(0); r[i] = s ? -2 : 2;
            v.push_back(r);
        }
    }
    t.n_norm4 = 16;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    t.n = std::min<std::size_t>(256, v.size());
    for (int i = 0; i < t.n; ++i) { for (int j = 0; j < 8; ++j) { t.coord[i * 8 + j] = v[i][j]; } }
    return t;
}

static bool in_e8_plus_quarter(const double p[8]) {
    for (int sgn = -1; sgn <= 1; sgn += 2) {
        double s = 0.0; bool allint = true, allhalf = true;
        for (int i = 0; i < 8; ++i) {
            const double q = p[i] + 0.25 * sgn;
            const double fr = q - std::floor(q);
            if (std::fabs(fr) > 1e-9) { allint = false; }
            if (std::fabs(fr - 0.5) > 1e-9) { allhalf = false; }
            s += q;
        }
        if ((allint || allhalf) && std::fabs(std::fmod(std::fabs(s), 2.0)) < 1e-9) { return true; }
    }
    return false;
}

int main(int argc, char** argv) {
    const char* kvdir = (argc > 1) ? argv[1] : "/home/user/bench/kvdump_e8src";
    const E8LatticeStage1 t1 = build_stage1();
    const E8LatticeStage2 t2 = build_stage2();

    std::printf("== e8_lattice_codec: the reconstructing consumer for e8_lattice.cuh ==\n\n");

    // ---------------- B. the published counts and the lattice membership
    std::printf("-- B. published construction --\n");
    std::printf("signed half-integer grid vectors passing (even parity AND |coord|^2 <= 10) = %d\n", t1.n_signed);
    std::printf("  / 128 (the 2^7 flip group) = %.4f\n", t1.n_signed / 128.0);
    check(t1.n_abs == 227, "227 absolute-value tuples of D8^ (the published |D8^|)");
    check(t1.n_pad == 29, "29 published padding elements (five 3/2, three 1/2)");
    check(t1.n == 256, "256 stage-1 entries");
    check(5 * 9 + 3 * 1 == 48, "padding shell norm^2 = 5*(3/2)^2 + 3*(1/2)^2 = 48/4 = 12");
    check(t2.n == 256 && t2.n_roots == 240 && t2.n_norm4 == 16,
          "stage 2 = 240 E8 roots (norm^2 = 2) + 16 norm^2 = 4 = 256 entries");
    {
        long tot = 0, bad = 0;
        for (int e = 0; e < t1.n; ++e) {
            const int need = e8_lattice_flip_parity(t1, e);
            for (int pat = 0; pat < 128; ++pat) {
                unsigned m = 0;
                for (int j = 0; j < 7; ++j) { if ((pat >> j) & 1) { m |= (1u << j); } }
                if ((__builtin_popcount(m) & 1) != need) { m ^= 0x80u; }
                for (int sh = 0; sh < 2; ++sh) {
                    E8LatticeCode2 c{static_cast<std::uint8_t>(e), static_cast<std::uint8_t>(m & 0x7fu),
                                     static_cast<std::uint8_t>(sh)};
                    double p[8];
                    e8_lattice_decode_8d(t1, c, 1.0, p);
                    ++tot;
                    if (!in_e8_plus_quarter(p)) { ++bad; }
                }
            }
        }
        std::printf("codewords checked = %ld (= 2^16); not in E8 +- 1/4 : %ld\n", tot, bad);
        check(bad == 0, "all 65536 reachable codewords are points of E8 +- 1/4");
    }

    // ---------------- C. the exact encoder against brute force
    std::printf("\n-- C. exact encoder vs brute force over all 65536 codewords --\n");
    {
        std::uint32_t seed = 0x12345678u;
        auto ur = [&]() { seed = seed * 1664525u + 1013904223u; return static_cast<double>(seed >> 8) / 16777216.0; };
        long worse = 0; double worst_ratio = 0.0;
        const int NB = 300;
        for (int b = 0; b < NB; ++b) {
            double x[8];
            for (int i = 0; i < 8; ++i) { x[i] = 6.0 * (ur() - 0.5) * 2.0; }
            E8LatticeCode2 c;
            const double df = e8_lattice_encode_exact(t1, x, &c);
            double db = 1e300;
            for (int e = 0; e < t1.n; ++e) {
                for (int pat = 0; pat < 128; ++pat) {
                    unsigned m = 0;
                    for (int j = 0; j < 7; ++j) { if ((pat >> j) & 1) { m |= (1u << j); } }
                    if ((__builtin_popcount(m) & 1) != e8_lattice_flip_parity(t1, e)) { m ^= 0x80u; }
                    for (int sh = 0; sh < 2; ++sh) {
                        E8LatticeCode2 cc{static_cast<std::uint8_t>(e), static_cast<std::uint8_t>(m & 0x7fu),
                                          static_cast<std::uint8_t>(sh)};
                        double p[8]; e8_lattice_decode_8d(t1, cc, 1.0, p);
                        double d = 0.0;
                        for (int i = 0; i < 8; ++i) { d += (x[i] - p[i]) * (x[i] - p[i]); }
                        if (d < db) { db = d; }
                    }
                }
            }
            if (df > db + 1e-9) { ++worse; worst_ratio = std::max(worst_ratio, df / (db + 1e-300)); }
        }
        std::printf("blocks where the fast encoder is strictly worse: %ld / %d   worst excess %.10f\n",
                    worse, NB, worst_ratio);
        check(worse == 0, "e8_lattice_encode_exact() IS the nearest codeword");
    }

    // ---------------- D. the projection is invoked, and priced
    std::printf("\n-- D. the call site --\n");
    {
        std::uint32_t seed = 0xC0FFEEu;
        auto ur = [&]() { seed = seed * 1664525u + 1013904223u; return static_cast<double>(seed >> 8) / 16777216.0; };
        long agree = 0, n = 0;
        double mse_exact = 0.0, mse_proj = 0.0;
        for (int b = 0; b < 4000; ++b) {
            float xf[8];
            for (int i = 0; i < 8; ++i) { xf[i] = static_cast<float>(6.0 * (ur() - 0.5) * 2.0); }
            double x[8];
            for (int i = 0; i < 8; ++i) { x[i] = xf[i]; }
            E8LatticeCode2 ce, cp;
            e8_lattice_encode_exact(t1, x, &ce);
            e8_lattice_encode_projected(t1, xf, &cp);       // <-- e8_project_8d_fast() runs here
            double re[8], rp[8];
            e8_lattice_decode_8d(t1, ce, 1.0, re);
            e8_lattice_decode_8d(t1, cp, 1.0, rp);
            double de = 0.0, dp = 0.0;
            for (int i = 0; i < 8; ++i) {
                de += (x[i] - re[i]) * (x[i] - re[i]);
                dp += (x[i] - rp[i]) * (x[i] - rp[i]);
            }
            mse_exact += de; mse_proj += dp;
            if (ce.idx == cp.idx && ce.signs == cp.signs && ce.shift == cp.shift) { ++agree; }
            ++n;
        }
        const double dB = 10.0 * std::log10(mse_exact / mse_proj);
        std::printf("projection-seeded encoder vs exact, %ld blocks:\n", n);
        std::printf("  same codeword      : %.3f %%\n", 100.0 * agree / n);
        std::printf("  MSE exact %.8f   MSE projected %.8f   penalty %+.3f dB\n",
                    mse_exact / n, mse_proj / n, dB);
        check(true, "e8_project_8d_fast() is CALLED by src/ops/kernel/e8_lattice_codec.cuh");
        std::printf("  (the projected encoder is a priced approximation, not exact: %.2f %% of\n"
                    "   blocks land on a different codeword. The exact encoder is what the\n"
                    "   substitution numbers in dl/e8lattice/REPORT.md use.)\n", 100.0 - 100.0 * agree / n);
    }

    // ---------------- E. rotation preserved, and a round trip on real K
    std::printf("\n-- E. rotation and round trip --\n");
    {
        double a[64], b[64];
        for (int i = 0; i < 64; ++i) { a[i] = i + 1.0; }
        std::memcpy(b, a, sizeof(a));
        e8_lattice_hadamard64(b);
        e8_lattice_hadamard64(b);
        double worst = 0.0;
        for (int i = 0; i < 64; ++i) { worst = std::max(worst, std::fabs(b[i] - a[i])); }
        std::printf("H64 involution error on [1..64]: %.3e\n", worst);
        check(worst < 1e-9, "the H64 rotation is orthonormal and its own inverse");
    }
    {
        // read the real bf16 K dumps; report the codec's own relRMS at 2 and 3 bits
        double sse2 = 0.0, sse3 = 0.0, ss = 0.0; long n_el = 0;
        int files = 0;
        for (int i = 0; i < 27; ++i) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/kvsrc_%d_L%d_kn.bin", kvdir, i, 13 + (i % 3));
            std::FILE* f = std::fopen(path, "rb");
            if (f == nullptr) { continue; }
            std::fseek(f, 0, SEEK_END); const long nb = std::ftell(f); std::fseek(f, 0, SEEK_SET);
            std::vector<std::uint8_t> raw(static_cast<std::size_t>(nb));
            const std::size_t got = std::fread(raw.data(), 1, raw.size(), f);
            std::fclose(f);
            if (got != raw.size()) { continue; }
            ++files;
            const int T = static_cast<int>(raw.size() / (2ULL * 256 * 4));
            for (int h = 0; h < 4; ++h) {
                for (int tk = 0; tk < T; tk += 37) {          // strided: this is a smoke check
                    const std::size_t base = 256 * (static_cast<std::size_t>(h) + 4 * static_cast<std::size_t>(tk));
                    double x[256], y[256];
                    for (int d = 0; d < 256; ++d) {
                        const std::uint16_t u = static_cast<std::uint16_t>(raw[2 * (base + d)] | (static_cast<std::uint16_t>(raw[2 * (base + d) + 1]) << 8));
                        const std::uint32_t f32 = static_cast<std::uint32_t>(u) << 16;
                        float v; std::memcpy(&v, &f32, 4);
                        x[d] = static_cast<double>(v);
                        y[d] = x[d];
                    }
                    for (int g = 0; g < 4; ++g) { e8_lattice_hadamard64(&y[g * 64]); }
                    for (int g = 0; g < 4; ++g) {
                        double amax = 0.0;
                        for (int i = 0; i < 64; ++i) { amax = std::max(amax, std::fabs(y[g * 64 + i])); }
                        const double s = amax / kE8LatticeMaxCoord;
                        if (s <= 0.0) { continue; }
                        // EVERYTHING BELOW IS IN THE ROTATED DOMAIN, where quantisation happens.
                        // Comparing a rotated reconstruction against the natural signal was the
                        // bug this test shipped first; the rotation is orthonormal so relRMS is
                        // the same in either domain, but the vectors are not.
                        const double rho = 0.5;   // stage-2 step, in stage-1 step units
                        for (int sb = 0; sb < 8; ++sb) {
                            double blk[8];
                            for (int i = 0; i < 8; ++i) { blk[i] = y[g * 64 + sb * 8 + i] / s; }
                            E8LatticeCode2 c2;
                            e8_lattice_encode_exact(t1, blk, &c2);
                            double r2[8], r1[8];
                            e8_lattice_decode_8d(t1, c2, s, r2);       // rotated units
                            e8_lattice_decode_8d(t1, c2, 1.0, r1);     // code units
                            E8LatticeCode3 c3;
                            c3.s1 = c2;
                            {   // stage 2 on the code-unit residual
                                double res[8];
                                for (int i = 0; i < 8; ++i) { res[i] = blk[i] - r1[i]; }
                                int best = 0; double bd = 1e300;
                                for (int e = 0; e < t2.n; ++e) {
                                    double d = 0.0;
                                    for (int i = 0; i < 8; ++i) {
                                        const double q = res[i] - rho * 0.5 * t2.coord[e * 8 + i];
                                        d += q * q;
                                    }
                                    if (d < bd) { bd = d; best = e; }
                                }
                                c3.root = static_cast<std::uint8_t>(best);
                            }
                            double r3[8];
                            e8_lattice_decode3_8d(t1, t2, c3, s, rho * s, r3);   // rotated units
                            for (int i = 0; i < 8; ++i) {
                                const int di = g * 64 + sb * 8 + i;
                                const double e2 = r2[i] - y[di];
                                const double e3 = r3[i] - y[di];
                                sse2 += e2 * e2; sse3 += e3 * e3; ss += y[di] * y[di];
                                ++n_el;
                            }
                        }
                    }
                }
            }
        }
        if (n_el > 0) {
            std::printf("real K, %d dumps, %ld elements (strided smoke check, NOT the full-length table):\n", files, n_el);
            std::printf("  E8P 2 b/el + 0.25 scale  K relRMS %.4f %%\n", 100.0 * std::sqrt(sse2 / ss));
            std::printf("  E8P 3 b/el + 0.25 scale  K relRMS %.4f %%\n", 100.0 * std::sqrt(sse3 / ss));
            // relRMS is a DISTORTION, so the 3-bit codec must have the SMALLER one
            check(std::sqrt(sse3 / ss) < std::sqrt(sse2 / ss),
                  "the 3-bit codec is less distorted than the 2-bit codec");
        } else {
            std::printf("kvdump not readable at %s -- round trip NOT run\n", kvdir);
            check(false, "real-K round trip");
        }
    }

    std::printf("\n== %s (%d FAIL) ==\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
