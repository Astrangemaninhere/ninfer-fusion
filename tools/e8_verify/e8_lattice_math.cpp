// e8_lattice_math.cpp -- E8VERIFY: is the tree's E8 projection actually a
// nearest-point projection onto E8 = D8 U (D8 + 1/2*1)?
//
// Criterion (used as given, NOT inferred):
//   D8      = { x in Z^8 : sum x_i  == 0 (mod 2) }
//   E8      = D8 U (D8 + 1/2*1)
//   membership: (all x_i in Z and sum x_i even) OR (all x_i in Z+1/2 and sum(x_i-1/2) even)
//   nearest D8 point = round(v) then, if sum round(v) is odd, flip the coordinate with the
//                      LARGEST |v_i - round(v_i)| outward
//   nearest (D8+1/2) = that rule applied to (v - 1/2*1), then + 1/2*1
//   nearest E8       = the closer of the two, deterministic tie rule
//
// Independent reference = EXHAUSTIVE windowed search. The true nearest D8 point differs from
// round(v) in at most one coordinate by +-1 (that is the algorithm's own claim), so searching
// all integer vectors in the 3^8 window around round(v) whose sum is even is EXACT, and it does
// not assume the rule it is checking. Same on (v - 1/2*1) for the half coset.
//
// Build: g++ -O2 -std=c++17 -o /tmp/e8math tools/e8_verify/e8_lattice_math.cpp

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const double kEps = 1e-9;

// ---------------------------------------------------------------- membership
struct Membership { bool in_d8, in_half; bool ok() const { return in_d8 || in_half; } };

static Membership e8_membership(const double x[8]) {
    int si = 0, sh = 0;
    bool all_int = true, all_half = true;
    for (int i = 0; i < 8; ++i) {
        const double ri = x[i] - std::nearbyint(x[i]);
        if (std::fabs(ri) > kEps) { all_int = false; }
        const double h = x[i] - 0.5;
        const double rh = h - std::nearbyint(h);
        if (std::fabs(rh) > kEps) { all_half = false; }
        si += static_cast<int>(std::nearbyint(x[i]));
        sh += static_cast<int>(std::nearbyint(h));
    }
    Membership m{false, false};
    m.in_d8 = all_int && (si % 2 == 0);
    m.in_half = all_half && (sh % 2 == 0);
    return m;
}

static double dist2(const double a[8], const double b[8]) {
    double d = 0.0;
    for (int i = 0; i < 8; ++i) { d += (a[i] - b[i]) * (a[i] - b[i]); }
    return d;
}

// ---------------------------------------------------------------- exact reference
// W = 1 is provably sufficient (see header). Chooses D8 on ties, matching the
// implementation's `dist_d8 <= dist_coset1`.
struct Ref { double pt[8]; double d2; bool half; };

static Ref ref_nearest(const double v[8]) {
    double best[8];
    double best_d2 = 1e300;
    bool best_half = false;
    for (int coset = 0; coset < 2; ++coset) {
        double base[8];
        const double shift = (coset == 0) ? 0.0 : 0.5;
        for (int i = 0; i < 8; ++i) { base[i] = std::nearbyint(v[i] - shift); }
        // exhaustive over the 3^8 window, keeping even sums
        int idx[8];
        for (int code = 0; code < 6561; ++code) {
            int c = code, s = 0;
            for (int i = 0; i < 8; ++i) { idx[i] = c % 3 - 1; c /= 3; s += static_cast<int>(base[i]) + idx[i]; }
            if (s % 2 != 0) { continue; }
            double cand[8];
            for (int i = 0; i < 8; ++i) { cand[i] = base[i] + idx[i] + shift; }
            const double d = dist2(v, cand);
            if (d < best_d2 - 1e-15) {
                best_d2 = d;
                std::memcpy(best, cand, sizeof(best));
                best_half = (coset == 1);
            }
        }
    }
    Ref r;
    std::memcpy(r.pt, best, sizeof(best));
    r.d2 = best_d2;
    r.half = best_half;
    return r;
}

// ---------------------------------------------------------------- the tree's implementations
// src/ops/kernel/e8_lattice.cuh:54-123, scalar form.
static void impl_scalar(const double x[8], double out[8], bool* half_coset) {
    double f_x[8];
    int sum_f = 0;
    double max_err = -1.0;
    int worst = 0;
    for (int i = 0; i < 8; ++i) {
        f_x[i] = std::nearbyint(x[i]);
        sum_f += static_cast<int>(f_x[i]);
        const double err = std::fabs(x[i] - f_x[i]);
        if (err > max_err) { max_err = err; worst = i; }
    }
    double d8[8];
    for (int i = 0; i < 8; ++i) { d8[i] = f_x[i]; }
    if ((sum_f & 1) != 0) { d8[worst] += (x[worst] >= f_x[worst]) ? 1.0 : -1.0; }

    double f_s[8];
    int sum_s = 0;
    double max_err_s = -1.0;
    int worst_s = 0;
    for (int i = 0; i < 8; ++i) {
        const double xs = x[i] - 0.5;
        f_s[i] = std::nearbyint(xs);
        sum_s += static_cast<int>(f_s[i]);
        const double err = std::fabs(xs - f_s[i]);
        if (err > max_err_s) { max_err_s = err; worst_s = i; }
    }
    double coset1[8];
    for (int i = 0; i < 8; ++i) { coset1[i] = f_s[i] + 0.5; }
    if ((sum_s & 1) != 0) { coset1[worst_s] += ((x[worst_s] - 0.5) >= f_s[worst_s]) ? 1.0 : -1.0; }

    const double dd = dist2(x, d8);
    const double dc = dist2(x, coset1);
    const bool use_d8 = (dd <= dc);
    for (int i = 0; i < 8; ++i) { out[i] = use_d8 ? d8[i] : coset1[i]; }
    *half_coset = !use_d8;
}

// src/ops/kernel/e8_lattice.cuh:140-206, warp form. Each lane computes its OWN coordinate,
// and all lanes redundantly run the __shfl_xor_sync reductions; emulated exactly, because a
// sequential reduction and a butterfly reduction agree only if the butterfly is double-buffered.
static void impl_warp(const double x[8], double out[8], bool* half_coset) {
    // Every lane l holds x[l]; the reductions are __shfl_xor_sync butterflies over the
    // 8-lane sub-group, so each stage must read PRE-update partner values (double buffer).
    double d8[8], coset1[8], dist0[8], dist1[8];

    // ---- D8 candidate ----
    double f[8];
    double s[8], me[8];
    int wl[8];
    for (int l = 0; l < 8; ++l) {
        f[l] = std::nearbyint(x[l]);
        s[l] = static_cast<double>(static_cast<int>(f[l]));
        me[l] = std::fabs(x[l] - f[l]);
        wl[l] = l;
    }
    for (int off = 1; off < 8; off <<= 1) {
        double t[8];
        for (int l = 0; l < 8; ++l) { t[l] = s[l] + s[l ^ off]; }
        std::memcpy(s, t, sizeof(s));
    }
    const int sum_f = static_cast<int>(s[0]);
    for (int off = 1; off < 8; off <<= 1) {
        double te[8];
        int tw[8];
        for (int l = 0; l < 8; ++l) {
            const double oe = me[l ^ off];
            const int ol = wl[l ^ off];
            if (oe > me[l] || (oe == me[l] && ol < wl[l])) { te[l] = oe; tw[l] = ol; }
            else { te[l] = me[l]; tw[l] = wl[l]; }
        }
        std::memcpy(me, te, sizeof(me));
        std::memcpy(wl, tw, sizeof(wl));
    }
    for (int l = 0; l < 8; ++l) {
        d8[l] = f[l];
        if ((sum_f & 1) != 0 && l == wl[0]) { d8[l] += (x[l] >= f[l]) ? 1.0 : -1.0; }
    }

    // ---- half coset ----
    double fs[8], s2[8], mes[8];
    int wls[8];
    for (int l = 0; l < 8; ++l) {
        const double xs = x[l] - 0.5;
        fs[l] = std::nearbyint(xs);
        s2[l] = static_cast<double>(static_cast<int>(fs[l]));
        mes[l] = std::fabs(xs - fs[l]);
        wls[l] = l;
    }
    for (int off = 1; off < 8; off <<= 1) {
        double t[8];
        for (int l = 0; l < 8; ++l) { t[l] = s2[l] + s2[l ^ off]; }
        std::memcpy(s2, t, sizeof(s2));
    }
    const int sum_fs = static_cast<int>(s2[0]);
    for (int off = 1; off < 8; off <<= 1) {
        double te[8];
        int tw[8];
        for (int l = 0; l < 8; ++l) {
            const double oe = mes[l ^ off];
            const int ol = wls[l ^ off];
            if (oe > mes[l] || (oe == mes[l] && ol < wls[l])) { te[l] = oe; tw[l] = ol; }
            else { te[l] = mes[l]; tw[l] = wls[l]; }
        }
        std::memcpy(mes, te, sizeof(mes));
        std::memcpy(wls, tw, sizeof(wls));
    }
    for (int l = 0; l < 8; ++l) {
        coset1[l] = fs[l] + 0.5;
        if ((sum_fs & 1) != 0 && l == wls[0]) { coset1[l] += ((x[l] - 0.5) >= fs[l]) ? 1.0 : -1.0; }
    }

    // ---- squared distances, reduced the same way ----
    double dv[8], tv[8];
    for (int l = 0; l < 8; ++l) { dv[l] = (x[l] - d8[l]) * (x[l] - d8[l]); }
    for (int off = 1; off < 8; off <<= 1) {
        for (int l = 0; l < 8; ++l) { tv[l] = dv[l] + dv[l ^ off]; }
        std::memcpy(dv, tv, sizeof(dv));
    }
    for (int l = 0; l < 8; ++l) { dist0[l] = dv[l]; }
    for (int l = 0; l < 8; ++l) { dv[l] = (x[l] - coset1[l]) * (x[l] - coset1[l]); }
    for (int off = 1; off < 8; off <<= 1) {
        for (int l = 0; l < 8; ++l) { tv[l] = dv[l] + dv[l ^ off]; }
        std::memcpy(dv, tv, sizeof(dv));
    }
    for (int l = 0; l < 8; ++l) { dist1[l] = dv[l]; }

    bool any_half = false;
    for (int i = 0; i < 8; ++i) {
        const bool use_d8 = (dist0[i] <= dist1[i]);
        out[i] = use_d8 ? d8[i] : coset1[i];
        if (!use_d8) { any_half = true; }
    }
    *half_coset = any_half;
}

// ---------------------------------------------------------------- defective controls
static void ctl_d8_only(const double x[8], double out[8], bool* h) {
    impl_scalar(x, out, h);
    double f[8], d8[8];
    int s = 0; double me = -1.0; int worst = 0;
    for (int i = 0; i < 8; ++i) {
        f[i] = std::nearbyint(x[i]); s += static_cast<int>(f[i]);
        const double e = std::fabs(x[i] - f[i]);
        if (e > me) { me = e; worst = i; }
    }
    for (int i = 0; i < 8; ++i) { d8[i] = f[i]; }
    if ((s & 1) != 0) { d8[worst] += (x[worst] >= f[worst]) ? 1.0 : -1.0; }
    for (int i = 0; i < 8; ++i) { out[i] = d8[i]; }
    *h = false;
}

static void ctl_parity_min(const double x[8], double out[8], bool* h) {
    // parity fix applied to the SMALLEST error coordinate instead of the largest
    double f[8];
    int s = 0; double me = 1e300; int worst = 0;
    for (int i = 0; i < 8; ++i) {
        f[i] = std::nearbyint(x[i]); s += static_cast<int>(f[i]);
        const double e = std::fabs(x[i] - f[i]);
        if (e < me) { me = e; worst = i; }
    }
    double d8[8];
    for (int i = 0; i < 8; ++i) { d8[i] = f[i]; }
    if ((s & 1) != 0) { d8[worst] += (x[worst] >= f[worst]) ? 1.0 : -1.0; }
    for (int i = 0; i < 8; ++i) { out[i] = d8[i]; }
    *h = false;
}

static void ctl_parity_plus(const double x[8], double out[8], bool* h) {
    // parity fix always +1, direction ignored
    impl_scalar(x, out, h);
    double f[8];
    int s = 0; double me = -1.0; int worst = 0;
    for (int i = 0; i < 8; ++i) {
        f[i] = std::nearbyint(x[i]); s += static_cast<int>(f[i]);
        const double e = std::fabs(x[i] - f[i]);
        if (e > me) { me = e; worst = i; }
    }
    double d8[8];
    for (int i = 0; i < 8; ++i) { d8[i] = f[i]; }
    if ((s & 1) != 0) { d8[worst] += 1.0; }
    for (int i = 0; i < 8; ++i) { out[i] = d8[i]; }
    *h = false;
}

// ---------------------------------------------------------------- 240-root codebook control
// E8's 240 minimal vectors: +-e_i +- e_j (i<j) and (+-1/2)^8 with an even number of minus
// signs, scaled by sqrt(2) so the root lattice is integral. Measures what a roots-only
// "E8" quantizer costs versus the true lattice.
static void build_roots(std::vector<std::array<double, 8>>* roots) {
    for (int i = 0; i < 8; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            for (int si = -1; si <= 1; si += 2) {
                for (int sj = -1; sj <= 1; sj += 2) {
                    std::array<double, 8> r{};
                    r[i] = si;
                    r[j] = sj;
                    roots->push_back(r);
                }
            }
        }
    }
    for (int m = 0; m < 256; ++m) {
        int bits = 0;
        std::array<double, 8> r{};
        for (int i = 0; i < 8; ++i) {
            const int b = (m >> i) & 1;
            r[i] = b ? -0.5 : 0.5;
            if (b) { ++bits; }
        }
        if (bits % 2 == 0) { roots->push_back(r); }
    }
}

// ---------------------------------------------------------------- data
static std::uint32_t g_seed = 0x12345678u;
static double urand() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return static_cast<double>(g_seed >> 8) / 16777216.0;
}
static double grand() {
    double u1 = urand() + 1e-12, u2 = urand();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

static void hadamard64(float v[64]) {
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

struct Stats {
    std::size_t n = 0;
    std::size_t mem_viol = 0;
    std::size_t non_nearest = 0;   // strictly worse than the exact reference
    std::size_t ties = 0;
    std::size_t chose_half = 0;
    std::size_t ref_half = 0;
    std::size_t missed_half = 0;   // ref chose the half coset, impl did not
    std::size_t spurious_half = 0; // ref chose D8, impl chose the half coset
    double sum_impl = 0.0, sum_ref = 0.0;
    double worst_excess_ratio = 0.0;
    void add(const double v[8], const double a[8], const Ref& r, bool half_impl) {
        ++n;
        if (!e8_membership(a).ok()) { ++mem_viol; }
        const double da = dist2(v, a);
        const double dr = r.d2;
        sum_impl += da;
        sum_ref += dr;
        if (da > dr + 1e-12) {
            ++non_nearest;
            const double ex = dr > 0 ? da / dr : (da > 0 ? 1e9 : 1.0);
            if (ex > worst_excess_ratio) { worst_excess_ratio = ex; }
        } else if (da > dr - 1e-12) {
            ++ties;
        }
        if (half_impl) { ++chose_half; }
        if (r.half) { ++ref_half; }
        if (r.half && !half_impl) { ++missed_half; }
        if (!r.half && half_impl) { ++spurious_half; }
    }
    void print(const char* label) const {
        std::printf("  %-34s n=%zu  non-lattice=%zu  strictly-non-nearest=%zu (%.4f%%)  ties=%zu\n",
                    label, n, mem_viol, non_nearest,
                    n ? 100.0 * static_cast<double>(non_nearest) / static_cast<double>(n) : 0.0, ties);
        std::printf("  %-34s MSE ratio impl/ref = %.9f   worst single excess = %.6f\n", "", 
                    sum_ref > 0 ? sum_impl / sum_ref : 0.0, worst_excess_ratio);
        std::printf("  %-34s chose 1/2-coset=%zu (%.3f%%)  ref chose 1/2=%zu (%.3f%%)  missed=%zu  spurious=%zu\n",
                    "", chose_half,
                    n ? 100.0 * static_cast<double>(chose_half) / static_cast<double>(n) : 0.0, ref_half,
                    n ? 100.0 * static_cast<double>(ref_half) / static_cast<double>(n) : 0.0, missed_half,
                    spurious_half);
    }
};

int main(int argc, char** argv) {
    std::vector<std::array<double, 8>> real8;
    // optional: real K 8-blocks taken from the engine's own writer domain
    //   argv[1] = kvsrc prefix (bf16 K, ne=256,4,T), argv[2] = token cap
    if (argc >= 2) {
        std::vector<std::uint8_t> k;
        const std::string p = std::string(argv[1]) + "_kn.bin";
        std::FILE* f = std::fopen(p.c_str(), "rb");
        if (f != nullptr) {
            std::fseek(f, 0, SEEK_END);
            const long n = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            k.resize(static_cast<std::size_t>(n));
            const std::size_t got = std::fread(k.data(), 1, k.size(), f);
            (void)got;
            std::fclose(f);
            const int D = 256, H = 4;
            const int T = static_cast<int>(k.size() / (2ULL * D * H));
            const int cap = (argc >= 3) ? std::atoi(argv[2]) : T;
            for (int t = 0; t < std::min(T, cap); ++t) {
                for (int h = 0; h < H; ++h) {
                    float grp[256];
                    for (int d = 0; d < D; ++d) {
                        const std::size_t off = 2 * (static_cast<std::size_t>(d) + D * (h + static_cast<std::size_t>(H) * t));
                        const std::uint16_t bits = static_cast<std::uint16_t>(k[off] | (static_cast<std::uint16_t>(k[off + 1]) << 8));
                        const std::uint32_t f32 = static_cast<std::uint32_t>(bits) << 16;
                        float v;
                        std::memcpy(&v, &f32, 4);
                        grp[d] = v;
                    }
                    for (int g = 0; g < 4; ++g) {
                        float blk[64];
                        for (int i = 0; i < 64; ++i) { blk[i] = grp[g * 64 + i]; }
                        hadamard64(blk);
                        float amax = 0.0f;
                        for (int i = 0; i < 64; ++i) { const float a = std::fabs(blk[i]); if (a > amax) { amax = a; } }
                        const double kinv = amax > 0.0f ? 7.0 / static_cast<double>(amax) : 0.0;
                        for (int sub = 0; sub < 8; ++sub) {
                            std::array<double, 8> v{};
                            for (int i = 0; i < 8; ++i) { v[i] = static_cast<double>(blk[sub * 8 + i]) * kinv; }
                            real8.push_back(v);
                        }
                    }
                }
            }
        }
        std::printf("real K 8-blocks (rotated + scaled, within one head's 64-group): %zu\n", real8.size());
    }

    std::vector<std::array<double, 8>> rnd_uniform, rnd_gauss;
    for (int i = 0; i < 20000; ++i) {
        std::array<double, 8> a{}, b{};
        for (int j = 0; j < 8; ++j) { a[j] = urand() * 16.0 - 8.0; b[j] = grand() * 2.0; }
        rnd_uniform.push_back(a);
        rnd_gauss.push_back(b);
    }

    std::vector<std::array<double, 8>> roots;
    build_roots(&roots);
    std::printf("E8 minimal-vector codebook size = %zu (expected 240)\n", roots.size());

    auto sweep = [&](const char* label, const std::vector<std::array<double, 8>>& data,
                     void (*fn)(const double[8], double[8], bool*), bool do_roots) {
        if (data.empty()) { return; }
        Stats st;
        double root_sum = 0.0, ref_sum = 0.0;
        for (const auto& vv : data) {
            double v[8];
            for (int i = 0; i < 8; ++i) { v[i] = vv[i]; }
            double a[8];
            bool half = false;
            fn(v, a, &half);
            const Ref r = ref_nearest(v);
            st.add(v, a, r, half);
            if (do_roots) {
                double best = 1e300;
                for (const auto& rt : roots) {
                    double d = 0.0;
                    for (int i = 0; i < 8; ++i) { d += (v[i] - rt[i]) * (v[i] - rt[i]); }
                    if (d < best) { best = d; }
                }
                root_sum += best;
                ref_sum += r.d2;
            }
        }
        st.print(label);
        if (do_roots) {
            std::printf("  %-34s MSE ratio roots240/true-lattice = %.6f  (roots are a different quantizer)\n",
                        "", ref_sum > 0 ? root_sum / ref_sum : 0.0);
        }
    };

    std::printf("\n=== 1. the tree's implementations vs an EXHAUSTIVE exact reference ===\n");
    std::printf("-- uniform random v in [-8,8]^8\n");
    sweep("impl_scalar (e8_project_8d_fast)", rnd_uniform, impl_scalar, true);
    sweep("impl_warp (e8_project_8d_warp_single)", rnd_uniform, impl_warp, false);
    std::printf("-- N(0,2) random v\n");
    sweep("impl_scalar", rnd_gauss, impl_scalar, false);
    sweep("impl_warp", rnd_gauss, impl_warp, false);
    if (!real8.empty()) {
        std::printf("-- REAL K 8-blocks, rotated + scaled into the writer's own domain\n");
        sweep("impl_scalar", real8, impl_scalar, true);
        sweep("impl_warp", real8, impl_warp, false);
    }

    std::printf("\n=== 2. NEGATIVE CONTROLS (the harness must catch every one) ===\n");
    sweep("CTL D8-only (never half)", rnd_uniform, ctl_d8_only, false);
    sweep("CTL parity on SMALLEST error", rnd_uniform, ctl_parity_min, false);
    sweep("CTL parity always +1", rnd_uniform, ctl_parity_plus, false);
    if (!real8.empty()) {
        sweep("CTL D8-only on REAL blocks", real8, ctl_d8_only, false);
    }

    std::printf("\n=== 3. what the integer-plane consumer costs (attribution of '+3.65 dB') ===\n");
    {
        double s_plain = 0.0, s_projrint = 0.0, s_latt = 0.0, s_ref = 0.0;
        std::size_t n = 0, half = 0;
        const std::vector<std::array<double, 8>>& data = real8.empty() ? rnd_uniform : real8;
        for (const auto& vv : data) {
            double v[8];
            for (int i = 0; i < 8; ++i) { v[i] = vv[i]; }
            double pr[8];
            bool h = false;
            impl_scalar(v, pr, &h);
            const Ref r = ref_nearest(v);
            for (int i = 0; i < 8; ++i) {
                const double pl = std::nearbyint(v[i]);
                const double pp = std::nearbyint(pr[i]);
                s_plain += (v[i] - pl) * (v[i] - pl);
                s_projrint += (v[i] - pp) * (v[i] - pp);
                s_latt += (v[i] - pr[i]) * (v[i] - pr[i]);
                s_ref += (v[i] - r.pt[i]) * (v[i] - r.pt[i]);
            }
            if (h) { ++half; }
            ++n;
        }
        std::printf("  n=%zu  half-coset chosen on %.3f%%\n", n,
                    n ? 100.0 * static_cast<double>(half) / static_cast<double>(n) : 0.0);
        std::printf("  MSE plain rint                  = %.9f\n", s_plain / (8.0 * n));
        std::printf("  MSE project-then-rint           = %.9f   ratio/plain = %.4f\n", s_projrint / (8.0 * n),
                    s_plain > 0 ? s_projrint / s_plain : 0.0);
        std::printf("  MSE lattice point (project)     = %.9f   ratio/plain = %.4f  -> %.3f dB\n",
                    s_latt / (8.0 * n), s_plain > 0 ? s_latt / s_plain : 0.0,
                    s_plain > 0 ? 10.0 * std::log10(s_latt / s_plain) : 0.0);
        std::printf("  MSE EXACT reference lattice pt  = %.9f   ratio/plain = %.4f  -> %.3f dB\n",
                    s_ref / (8.0 * n), s_plain > 0 ? s_ref / s_plain : 0.0,
                    s_plain > 0 ? 10.0 * std::log10(s_ref / s_plain) : 0.0);
        std::printf("  implementation vs exact reference MSE = %.9f  (1.0 == the impl IS the nearest point)\n",
                    s_ref > 0 ? s_latt / s_ref : 0.0);
    }
    std::printf("\n=== 4. THE SHIPPED CODEC'S OWN EMITTED CODEWORDS vs the membership predicate ===\n");
    // The shipped 'e8' tier never calls a projection, so the codeword its plane stores for an
    // 8-tuple of the rotated domain is just the vector of its 4-bit codes. Test THAT against
    // the criterion: it must be in Z^8 with even sum, or in (Z+1/2)^8 with even coordinate-parity.
    if (argc >= 4) {
        std::vector<std::uint8_t> kc, bt;
        auto rd = [](const std::string& p, std::vector<std::uint8_t>* o) {
            std::FILE* f = std::fopen(p.c_str(), "rb");
            if (f == nullptr) { return false; }
            std::fseek(f, 0, SEEK_END);
            const long n = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            o->assign(static_cast<std::size_t>(n), 0);
            std::fread(o->data(), 1, o->size(), f);
            std::fclose(f);
            return true;
        };
        const std::string kp = std::string(argv[3]);
        const bool ok_k = rd(kp + "_k.bin", &kc);
        rd(kp + "_bt.bin", &bt);
        if (ok_k) {
            const std::int32_t* tbl = reinterpret_cast<const std::int32_t*>(bt.data());
            const int H = 4, ROWB = 128, GPR = 4;
            const int cap = (argc >= 5) ? std::atoi(argv[4]) : 64;
            std::size_t n8 = 0, not_lattice = 0, odd_int = 0, all_half = 0, neither = 0;
            for (int t = 0; t < cap; ++t) {
                const int page = (bt.size() >= 4u * ((t >> 6) + 1)) ? tbl[t >> 6] : (t >> 6);
                const int off = t & 63;
                for (int h = 0; h < H; ++h) {
                    for (int g = 0; g < GPR; ++g) {
                        int code[64];
                        for (int b = 0; b < 32; ++b) {
                            const std::size_t idx =
                                static_cast<std::size_t>(ROWB) * 64 * (h + static_cast<std::size_t>(H) * page) +
                                static_cast<std::size_t>(ROWB) * off + g * 32 + b;
                            if (idx >= kc.size()) { continue; }
                            const std::uint8_t pk = kc[idx];
                            code[2 * b] = static_cast<int>(pk & 0x0Fu ^ 8u) - 8;
                            code[2 * b + 1] = static_cast<int>((pk >> 4) ^ 8u) - 8;
                        }
                        for (int sub = 0; sub < 8; ++sub) {
                            double x[8];
                            for (int i = 0; i < 8; ++i) { x[i] = static_cast<double>(code[sub * 8 + i]); }
                            const Membership m = e8_membership(x);
                            ++n8;
                            if (!m.ok()) {
                                ++not_lattice;
                                int s = 0;
                                for (int i = 0; i < 8; ++i) { s += code[sub * 8 + i]; }
                                if (s % 2 != 0) { ++odd_int; } else { ++neither; }
                            }
                            if (m.in_half) { ++all_half; }
                        }
                    }
                }
            }
            std::printf("  8-tuples of emitted shipped codewords = %zu\n", n8);
            std::printf("  NOT an E8 point                = %zu (%.4f%%)\n", not_lattice,
                        n8 ? 100.0 * static_cast<double>(not_lattice) / static_cast<double>(n8) : 0.0);
            std::printf("    of those, integer with ODD sum (neither D8 nor D8+1/2) = %zu (%.4f%%)\n", odd_int,
                        n8 ? 100.0 * static_cast<double>(odd_int) / static_cast<double>(n8) : 0.0);
            std::printf("  in the D8+1/2 coset            = %zu (%.4f%%)   [shipped codes are integers]\n",
                        all_half, n8 ? 100.0 * static_cast<double>(all_half) / static_cast<double>(n8) : 0.0);
            // WHY the odd-sum rate is what it is: measure the mechanism, do not assert it.
            std::size_t nz = 0, sat = 0, ncod = 0;
            std::size_t oddcount_hist[9] = {0};
            for (int t = 0; t < cap; ++t) {
                const int page = (bt.size() >= 4u * ((t >> 6) + 1)) ? tbl[t >> 6] : (t >> 6);
                const int off = t & 63;
                for (int h = 0; h < H; ++h) {
                    for (int g = 0; g < GPR; ++g) {
                        int code[64];
                        for (int b = 0; b < 32; ++b) {
                            const std::size_t idx =
                                static_cast<std::size_t>(ROWB) * 64 * (h + static_cast<std::size_t>(H) * page) +
                                static_cast<std::size_t>(ROWB) * off + g * 32 + b;
                            if (idx >= kc.size()) { continue; }
                            const std::uint8_t pk = kc[idx];
                            code[2 * b] = static_cast<int>(pk & 0x0Fu ^ 8u) - 8;
                            code[2 * b + 1] = static_cast<int>((pk >> 4) ^ 8u) - 8;
                        }
                        for (int sub = 0; sub < 8; ++sub) {
                            int nodd = 0;
                            for (int i = 0; i < 8; ++i) {
                                const int c = code[sub * 8 + i];
                                ++ncod;
                                if (c == 0) { ++nz; }
                                if (c == 7 || c == -7) { ++sat; }
                                if ((c & 1) != 0) { ++nodd; }
                            }
                            oddcount_hist[nodd]++;
                        }
                    }
                }
            }
            std::printf("  codeword mass: zeros = %.2f%%   saturated at +-7 = %.2f%%\n",
                        ncod ? 100.0 * static_cast<double>(nz) / ncod : 0.0,
                        ncod ? 100.0 * static_cast<double>(sat) / ncod : 0.0);
            std::printf("  #odd codes per 8-tuple:");
            for (int i = 0; i <= 8; ++i) { std::printf(" %d:%.2f%%", i, n8 ? 100.0 * static_cast<double>(oddcount_hist[i]) / n8 : 0.0); }
            std::printf("\n  (an unconstrained integer code vector would put ~50%% of tuples at odd #odd)\n");
            // control: the same statistic on uniform random integers in [-7,7]
            {
                std::size_t ctl_odd = 0, ctl_n = 0;
                for (int i = 0; i < 200000; ++i) {
                    int s = 0;
                    for (int j = 0; j < 8; ++j) { s += static_cast<int>(urand() * 15.0) - 7; }
                    ++ctl_n;
                    if (s % 2 != 0) { ++ctl_odd; }
                }
                std::printf("  CONTROL uniform random codes in [-7,7]: odd-sum = %.3f%%\n",
                            ctl_n ? 100.0 * static_cast<double>(ctl_odd) / ctl_n : 0.0);
            }
        } else {
            std::printf("  (kvc plane not readable at %s)\n", kp.c_str());
        }
    } else {
        std::printf("  (pass the kvc prefix as argv[3] to run this)\n");
    }
    return 0;
}
