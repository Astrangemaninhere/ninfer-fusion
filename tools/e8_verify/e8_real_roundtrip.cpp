// e8_real_roundtrip.cpp -- E8VERIFY host harness, real K/V path, no CUDA.
//
// Purpose: settle, on REAL engine data (NINFER_KVDUMP_DIR planes) and not on
// synthetic patterns:
//   (1) does a host replication of the shipped e8 K WRITER reproduce the engine's
//       own stored code plane BYTE FOR BYTE?      (writer model == shipped writer)
//   (2) what is the K fidelity of the SHIPPED reader path, measured as relL2 of
//       (un-rotated decoded plane) vs (the bf16 source the engine ingested)?
//   (3) what fraction of REAL 8-blocks take the E8 half-integer coset D8+1/2, and
//       what does the projection actually cost / buy on those blocks?
//
// Everything here replicates device arithmetic that already exists in the tree:
//   hadamard  : src/ops/kernel/gqa_attention_kv_quant.cuh:22-36
//   writer    : src/ops/kernel/gqa_attention_prefill_i8.cuh:152-226 (E8 arm)
//   reader    : src/ops/kernel/gqa_attention_kv_quant.cuh:195-211
//   layout    : src/ops/kernel/paged_kv_address.cuh:55-69
//   projection: src/ops/kernel/e8_lattice.cuh:54-123
//
// Build: g++ -O2 -std=c++17 -o /tmp/e8rt tools/e8_verify/e8_real_roundtrip.cpp

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------- fp16 / bf16
static float bf16_to_f32(std::uint16_t h) {
    std::uint32_t f = static_cast<std::uint32_t>(h) << 16;
    float o;
    std::memcpy(&o, &f, sizeof(o));
    return o;
}

// __float2half_rn: round to nearest, ties to EVEN. (kv_scale_half -> __float2half_rn)
static std::uint16_t f32_to_f16_rne(float v) {
    std::uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    const std::uint32_t sign = (f >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((f >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t man = f & 0x7FFFFFu;
    if (((f >> 23) & 0xFFu) == 0xFFu) {  // inf / nan
        return static_cast<std::uint16_t>(sign | 0x7C00u | (man != 0 ? 0x200u : 0u));
    }
    if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7C00u); }
    if (exp <= 0) {  // subnormal or zero
        if (exp < -10) { return static_cast<std::uint16_t>(sign); }
        man |= 0x800000u;
        const int shift = 14 - exp;
        std::uint32_t hm = man >> shift;
        const std::uint32_t rem = man & ((1u << shift) - 1u);
        const std::uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1u))) { ++hm; }
        return static_cast<std::uint16_t>(sign | hm);
    }
    std::uint32_t hm = man >> 13;
    const std::uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
        ++hm;
        if (hm == 0x400u) { hm = 0; ++exp; if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7C00u); } }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | hm);
}

static float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1Fu;
    std::uint32_t man = h & 0x3FFu;
    std::uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sign; }
        else {
            exp = 1;
            while ((man & 0x400u) == 0u) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (man << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float o;
    std::memcpy(&o, &f, sizeof(o));
    return o;
}

// ---------------------------------------------------------------- the rotation
// src/ops/kernel/gqa_attention_kv_quant.cuh:22-36, replicated over a 64-vector.
// lane i owns dim i (x0) and dim i+32 (x1).
static void hadamard64(float v[64]) {
    float x0[32], x1[32], t0[32], t1[32];
    for (int i = 0; i < 32; ++i) { x0[i] = v[i]; x1[i] = v[i + 32]; }
    for (int off = 1; off < 32; off <<= 1) {
        // __shfl_xor_sync hands every lane the PRE-update value, so the stage must be
        // double-buffered here: an in-place loop would read a value already rewritten
        // at the partner index and is NOT the same transform.
        for (int i = 0; i < 32; ++i) {
            const float y0 = x0[i ^ off];
            const float y1 = x1[i ^ off];
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

// The transform must be an involution and norm-preserving; assert it, do not assume it.
static void hadamard64_selftest(double* worst_norm_rel, double* worst_invol_rel) {
    std::uint32_t s = 12345u;
    auto rnd = [&]() {
        s = s * 1664525u + 1013904223u;
        return (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;
    };
    double wn = 0.0, wi = 0.0;
    for (int trial = 0; trial < 200; ++trial) {
        float a[64], b[64], c[64];
        for (int i = 0; i < 64; ++i) { a[i] = rnd(); }
        double n0 = 0.0;
        for (int i = 0; i < 64; ++i) { n0 += static_cast<double>(a[i]) * a[i]; }
        for (int i = 0; i < 64; ++i) { b[i] = a[i]; }
        hadamard64(b);
        double n1 = 0.0;
        for (int i = 0; i < 64; ++i) { n1 += static_cast<double>(b[i]) * b[i]; }
        const double nr = std::fabs(n1 - n0) / (n0 > 0 ? n0 : 1.0);
        if (nr > wn) { wn = nr; }
        for (int i = 0; i < 64; ++i) { c[i] = b[i]; }
        hadamard64(c);
        double worst = 0.0;
        for (int i = 0; i < 64; ++i) {
            const double e = std::fabs(static_cast<double>(c[i]) - a[i]) / (std::fabs(a[i]) + 1e-3);
            if (e > worst) { worst = e; }
        }
        if (worst > wi) { wi = worst; }
    }
    *worst_norm_rel = wn;
    *worst_invol_rel = wi;
}

// ------------------------------------------------------------- E8 projection
// src/ops/kernel/e8_lattice.cuh:54-123 (scalar form; same algorithm as the warp one).
struct ProjectionResult {
    bool half_integer_coset;  // the chosen nearest point lies in D8 + 1/2
    float pts[8];
    float cost_ship;   // squared error of the SHIPPED encoding of this block
    float cost_proj;   // squared error of "project, then rint" (the configuration the note warns about)
    float cost_latt;   // squared error of the lattice point itself (a reconstructing consumer)
};

static void e8_project_8d_fast(const float x[8], float out[8], bool* half_coset) {
    float f_x[8];
    int sum_f = 0;
    float max_err = -1.0f;
    int worst = 0;
    for (int i = 0; i < 8; ++i) {
        f_x[i] = std::nearbyint(x[i]);
        sum_f += static_cast<int>(f_x[i]);
        const float err = std::fabs(x[i] - f_x[i]);
        if (err > max_err) { max_err = err; worst = i; }
    }
    float d8[8];
    for (int i = 0; i < 8; ++i) { d8[i] = f_x[i]; }
    if ((sum_f & 1) != 0) { d8[worst] += (x[worst] >= f_x[worst]) ? 1.0f : -1.0f; }

    float f_s[8];
    int sum_s = 0;
    float max_err_s = -1.0f;
    int worst_s = 0;
    for (int i = 0; i < 8; ++i) {
        const float xs = x[i] - 0.5f;
        f_s[i] = std::nearbyint(xs);
        sum_s += static_cast<int>(f_s[i]);
        const float err = std::fabs(xs - f_s[i]);
        if (err > max_err_s) { max_err_s = err; worst_s = i; }
    }
    float coset1[8];
    for (int i = 0; i < 8; ++i) { coset1[i] = f_s[i] + 0.5f; }
    if ((sum_s & 1) != 0) { coset1[worst_s] += ((x[worst_s] - 0.5f) >= f_s[worst_s]) ? 1.0f : -1.0f; }

    float dd = 0.0f, dc = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const float d0 = x[i] - d8[i];
        const float d1 = x[i] - coset1[i];
        dd += d0 * d0;
        dc += d1 * d1;
    }
    const bool use_d8 = (dd <= dc);
    for (int i = 0; i < 8; ++i) { out[i] = use_d8 ? d8[i] : coset1[i]; }
    *half_coset = !use_d8;
}

// ---------------------------------------------------------------- file helpers
static bool read_file(const std::string& path, std::vector<std::uint8_t>* out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { return false; }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out->assign(static_cast<std::size_t>(n), 0);
    const std::size_t got = std::fread(out->data(), 1, static_cast<std::size_t>(n), f);
    std::fclose(f);
    return got == static_cast<std::size_t>(n);
}

struct Meta {
    int layer = -1, head_dim = 0, kv_heads = 0, tokens = 0, quant_group = 0, dtype = -1;
};

static bool parse_meta(const std::string& path, Meta* m) {
    std::vector<std::uint8_t> b;
    if (!read_file(path, &b)) { return false; }
    const std::string s(reinterpret_cast<const char*>(b.data()), b.size());
    auto geti = [&](const char* key) -> int {
        const std::size_t p = s.find(key);
        if (p == std::string::npos) { return -1; }
        return std::atoi(s.c_str() + p + std::strlen(key));
    };
    m->layer = geti("layer=");
    m->dtype = geti("dtype=");
    m->quant_group = geti("quant_group=");
    m->head_dim = geti("head_dim=");
    m->kv_heads = geti("num_kv_heads=");
    if (m->tokens < 0) { m->tokens = 0; }
    const std::size_t tp = s.find("tokens=");
    if (tp != std::string::npos) { m->tokens = std::atoi(s.c_str() + tp + 7); }
    // kvsrc meta uses "T=" instead
    if (m->tokens == 0) {
        const std::size_t t2 = s.find(" T=");
        if (t2 != std::string::npos) { m->tokens = std::atoi(s.c_str() + t2 + 3); }
    }
    if (m->head_dim == 0) { m->head_dim = geti("head_dim="); }
    return true;
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <kvc_prefix> <kvsrc_prefix>\n"
                     "  e.g. %s /home/user/bench/kvdump_e8src/kvc_2_t3072_L14 "
                     "/home/user/bench/kvdump_e8src/kvsrc_4_L14\n",
                     argv[0], argv[0]);
        return 2;
    }
    const std::string kvc = argv[1];
    const std::string src = argv[2];

    Meta km, sm;
    if (!parse_meta(kvc + "_meta.txt", &km)) { std::fprintf(stderr, "no kvc meta\n"); return 2; }
    if (!parse_meta(src + "_meta.txt", &sm)) { std::fprintf(stderr, "no src meta\n"); return 2; }

    std::vector<std::uint8_t> kbytes, ksbytes, srcbytes, btbytes;
    if (!read_file(kvc + "_k.bin", &kbytes)) { std::fprintf(stderr, "no k.bin\n"); return 2; }
    if (!read_file(kvc + "_ks.bin", &ksbytes)) { std::fprintf(stderr, "no ks.bin\n"); return 2; }
    std::string srcfile = src + "_kn.bin";
    if (!read_file(srcfile, &srcbytes)) {
        srcfile = src + "_k.bin";
        if (!read_file(srcfile, &srcbytes)) { std::fprintf(stderr, "no src kn/k bin\n"); return 2; }
    }
    read_file(src + "_bt.bin", &btbytes);

    const int D = km.head_dim;         // 256
    const int H = km.kv_heads;         // 4
    const int T = sm.tokens;           // source token count
    const int GPR = D / 64;            // 4 groups per row
    const int ROWB = D / 2;            // 128 code bytes per row
    if (D != 256 || H <= 0 || T <= 0) { std::fprintf(stderr, "unexpected geometry\n"); return 2; }

    std::printf("== E8VERIFY real round trip ==\n");
    std::printf("kvc   : %s  (meta layer=%d dtype=%d qg=%d D=%d H=%d)\n", kvc.c_str(), km.layer,
                km.dtype, km.quant_group, km.head_dim, km.kv_heads);
    std::printf("src   : %s  (meta layer=%d dtype=%d T=%d)\n", src.c_str(), sm.layer, sm.dtype, T);
    std::printf("sizes : k=%zu ks=%zu src=%zu bt=%zu  file=%.6f of full plane\n", kbytes.size(),
                ksbytes.size(), srcbytes.size(), btbytes.size(),
                static_cast<double>(kbytes.size()) / (static_cast<double>(ROWB) * 64 * H * 512));
    {
        double wn = 0.0, wi = 0.0;
        hadamard64_selftest(&wn, &wi);
        std::printf("rotation self-test: worst |norm-ratio-1| = %.3e ; worst involution rel err = %.3e\n",
                    wn, wi);
    }

    const std::int32_t* bt = reinterpret_cast<const std::int32_t*>(btbytes.data());
    auto phys_page = [&](int position) -> int {
        const int entry = position >> 6;
        if (btbytes.size() >= static_cast<std::size_t>(4 * (entry + 1))) { return bt[entry]; }
        return entry;
    };

    // ---- (1) the writer, replicated, plus the fidelity of the shipped reader ----
    std::vector<float> srcf(static_cast<std::size_t>(D) * H * T);
    for (std::size_t i = 0; i < srcf.size(); ++i) {
        const std::size_t off = 2 * i;
        srcf[i] = bf16_to_f32(static_cast<std::uint16_t>(srcbytes[off] |
                                                         (static_cast<std::uint16_t>(srcbytes[off + 1]) << 8)));
    }
    auto sval = [&](int d, int h, int t) -> float {
        return srcf[static_cast<std::size_t>(d) + static_cast<std::size_t>(D) * (h + static_cast<std::size_t>(H) * t)];
    };

    double se_ship = 0.0, ss = 0.0, se_v_ship = 0.0, ss_v = 0.0;
    std::size_t bytes_cmp = 0, bytes_eq = 0;
    std::size_t nblk = 0, nblk_half = 0;
    double se_plain = 0.0, se_proj = 0.0, se_latt = 0.0, se_scale = 0.0;
    std::size_t rows_done = 0, rows_present = 0, nelem = 0;
    std::size_t scalecmp = 0, scaleeq = 0, inv_groups = 0, inv_groups_ok = 0;
    std::size_t scalecmp_raw = 0, scaleeq_raw = 0;
    std::size_t bytes_eq_b = 0, bytes_eq_c = 0;
    double se_cur = 0.0, ss_cur = 0.0;
    std::vector<double> ratios_rot, ratios_raw;
    auto rstat = [](std::vector<double> v, double* mn, double* md, double* mx) {
        if (v.empty()) { *mn = *md = *mx = 0; return; }
        std::sort(v.begin(), v.end());
        *mn = v.front();
        *md = v[v.size() / 2];
        *mx = v.back();
    };
    double worst_row_rel = 0.0;

    for (int t = 0; t < T; ++t) {
        const int page = phys_page(t);
        const int off = t & 63;
        for (int h = 0; h < H; ++h) {
            // decode the engine's stored plane for this (page, head, token)
            float dec[256];
            bool row_ok = true;
            for (int b = 0; b < ROWB; ++b) {
                const std::size_t idx =
                    static_cast<std::size_t>(ROWB) * 64 *
                        (static_cast<std::size_t>(h) + H * static_cast<std::size_t>(page)) +
                    static_cast<std::size_t>(ROWB) * off + b;
                if (idx >= kbytes.size()) { row_ok = false; break; }
                const std::uint8_t packed = kbytes[idx];
                const int lo = static_cast<int>(packed & 0x0Fu);
                const int hi = static_cast<int>(packed >> 4);
                dec[2 * b] = static_cast<float>(static_cast<int>(lo ^ 8u) - 8);
                dec[2 * b + 1] = static_cast<float>(static_cast<int>(hi ^ 8u) - 8);
            }
            if (!row_ok) { break; }
            float scl[4];
            std::uint16_t sclraw[4];
            bool sc_ok = true;
            for (int g = 0; g < GPR; ++g) {
                const std::size_t idx =
                    static_cast<std::size_t>(GPR) * 64 *
                        (static_cast<std::size_t>(h) + H * static_cast<std::size_t>(page)) +
                    static_cast<std::size_t>(GPR) * off + g;
                if (2 * idx + 1 >= ksbytes.size()) { sc_ok = false; break; }
                const std::uint16_t raw = static_cast<std::uint16_t>(
                    ksbytes[2 * idx] | (static_cast<std::uint16_t>(ksbytes[2 * idx + 1]) << 8));
                scl[g] = f16_to_f32(raw);
                sclraw[g] = raw;
            }
            if (!sc_ok) { break; }
            ++rows_present;
            // dequant in the ROTATED domain
            float rot[256];
            for (int g = 0; g < GPR; ++g) {
                for (int i = 0; i < 64; ++i) { rot[g * 64 + i] = dec[g * 64 + i] * scl[g]; }
            }
            // un-rotate with the same (self-inverse) butterfly
            float un[256];
            for (int g = 0; g < GPR; ++g) {
                float blk[64];
                for (int i = 0; i < 64; ++i) { blk[i] = rot[g * 64 + i]; }
                hadamard64(blk);
                for (int i = 0; i < 64; ++i) { un[g * 64 + i] = blk[i]; }
            }
            double row_se = 0.0, row_ss = 0.0;
            for (int d = 0; d < D; ++d) {
                const double e = static_cast<double>(un[d]) - static_cast<double>(sval(d, h, t));
                row_se += e * e;
                row_ss += static_cast<double>(sval(d, h, t)) * static_cast<double>(sval(d, h, t));
                se_ship += e * e;
                ss += static_cast<double>(sval(d, h, t)) * static_cast<double>(sval(d, h, t));
                ++nelem;
            }
            if (row_ss > 0) {
                const double r = std::sqrt(row_se / row_ss);
                if (r > worst_row_rel) { worst_row_rel = r; }
            }

            // ---- writer replication + projection study, per 64-group ----
            for (int g = 0; g < GPR; ++g) {
                float x[64];
                for (int i = 0; i < 64; ++i) { x[i] = sval(g * 64 + i, h, t); }
                hadamard64(x);                       // post-rotation values, exactly as the writer sees
                float amax = 0.0f;
                for (int i = 0; i < 64; ++i) { const float a = std::fabs(x[i]); if (a > amax) { amax = a; } }
                // pre-rotation amax of the SAME 64-group, to test which domain the scale came from
                float xraw[64];
                for (int i = 0; i < 64; ++i) { xraw[i] = sval(g * 64 + i, h, t); }
                float amax_raw = 0.0f;
                for (int i = 0; i < 64; ++i) {
                    const float a = std::fabs(xraw[i]);
                    if (a > amax_raw) { amax_raw = a; }
                }
                const std::uint16_t ksh_raw = f32_to_f16_rne(amax_raw > 0.0f ? amax_raw / 7.0f : 0.0f);
                ++scalecmp_raw;
                if (ksh_raw == sclraw[g]) { ++scaleeq_raw; }
                if (ksh_raw != 0 && sclraw[g] != 0) {
                    ratios_raw.push_back(static_cast<double>(f16_to_f32(sclraw[g])) /
                                         static_cast<double>(f16_to_f32(ksh_raw)));
                }
                const std::uint16_t ksh = f32_to_f16_rne(amax > 0.0f ? amax / 7.0f : 0.0f);
                const float ks = f16_to_f32(ksh);
                const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
                ++scalecmp;
                if (ksh == sclraw[g]) { ++scaleeq; }
                if (ksh != 0 && sclraw[g] != 0) {
                    ratios_rot.push_back(static_cast<double>(f16_to_f32(sclraw[g])) /
                                         static_cast<double>(f16_to_f32(ksh)));
                }

                std::int8_t mine[64];
                int mx = 0;
                for (int i = 0; i < 64; ++i) {
                    int q = static_cast<int>(std::nearbyint(x[i] * kinv));
                    q = q < -7 ? -7 : (q > 7 ? 7 : q);
                    mine[i] = static_cast<std::int8_t>(q);
                    if (std::abs(q) > mx) { mx = std::abs(q); }
                }
                ++inv_groups;
                if (mx == 7) { ++inv_groups_ok; }
                // CURRENT shipped arithmetic, real-data round trip:
                //   encode (rotate -> scale amax_post/7 -> rint -> clamp) -> decode -> un-rotate
                {
                    float back[64];
                    for (int i = 0; i < 64; ++i) { back[i] = static_cast<float>(mine[i]) * ks; }
                    hadamard64(back);  // self-inverse
                    for (int i = 0; i < 64; ++i) {
                        const double e = static_cast<double>(back[i]) - static_cast<double>(xraw[i]);
                        se_cur += e * e;
                        ss_cur += static_cast<double>(xraw[i]) * static_cast<double>(xraw[i]);
                    }
                }
                // byte-for-byte against the engine's stored nibbles, under THREE writer models:
                //   (a) shipped model : rotate, scale = amax_post/7
                //   (b) pre-rotation  : rotate, scale = amax_pre/7  (_TODO.md 116/116b -- the
                //                       configuration the current comment says was FIXED)
                //   (c) no rotation   : raw codes, scale = amax_pre/7
                std::int8_t mine_b[64], mine_c[64];
                {
                    const float sp = f16_to_f32(sclraw[g]);
                    const float kp = sp > 0.0f ? 1.0f / sp : 0.0f;
                    for (int i = 0; i < 64; ++i) {
                        int q = static_cast<int>(std::nearbyint(x[i] * kp));
                        mine_b[i] = static_cast<std::int8_t>(q < -7 ? -7 : (q > 7 ? 7 : q));
                        q = static_cast<int>(std::nearbyint(xraw[i] * kp));
                        mine_c[i] = static_cast<std::int8_t>(q < -7 ? -7 : (q > 7 ? 7 : q));
                    }
                }
                for (int b = 0; b < 32; ++b) {
                    const std::size_t idx =
                        static_cast<std::size_t>(ROWB) * 64 *
                            (static_cast<std::size_t>(h) + H * static_cast<std::size_t>(page)) +
                        static_cast<std::size_t>(ROWB) * off + g * 32 + b;
                    if (idx >= kbytes.size()) { break; }
                    const std::uint8_t want = static_cast<std::uint8_t>(
                        (static_cast<unsigned>(mine[2 * b]) & 0x0Fu) |
                        ((static_cast<unsigned>(mine[2 * b + 1]) & 0x0Fu) << 4));
                    const std::uint8_t want_b = static_cast<std::uint8_t>(
                        (static_cast<unsigned>(mine_b[2 * b]) & 0x0Fu) |
                        ((static_cast<unsigned>(mine_b[2 * b + 1]) & 0x0Fu) << 4));
                    const std::uint8_t want_c = static_cast<std::uint8_t>(
                        (static_cast<unsigned>(mine_c[2 * b]) & 0x0Fu) |
                        ((static_cast<unsigned>(mine_c[2 * b + 1]) & 0x0Fu) << 4));
                    ++bytes_cmp;
                    if (want == kbytes[idx]) { ++bytes_eq; }
                    if (want_b == kbytes[idx]) { ++bytes_eq_b; }
                    if (want_c == kbytes[idx]) { ++bytes_eq_c; }
                }
                // projection study on the SCALED rotated block (what e8_project_8d_warp would see)
                for (int sub = 0; sub < 8; ++sub) {
                    float blk[8], proj[8];
                    bool half = false;
                    for (int i = 0; i < 8; ++i) { blk[i] = x[sub * 8 + i] * kinv; }
                    e8_project_8d_fast(blk, proj, &half);
                    ++nblk;
                    if (half) { ++nblk_half; }
                    for (int i = 0; i < 8; ++i) {
                        const double xs = blk[i];
                        const double plain = std::nearbyint(xs);
                        const double pl = plain < -7 ? -7 : (plain > 7 ? 7 : plain);
                        const double pp = std::nearbyint(proj[i]);
                        const double ppc = pp < -7 ? -7 : (pp > 7 ? 7 : pp);
                        se_plain += (xs - pl) * (xs - pl);
                        se_proj += (xs - ppc) * (xs - ppc);
                        se_latt += (xs - proj[i]) * (xs - proj[i]);
                        se_scale += xs * xs;
                    }
                }
            }
            ++rows_done;
        }
    }

    const double relL2 = std::sqrt(se_ship / ss);
    std::printf("\n-- (0b) V PLANE: same codec keys, NO rotation involved --\n");
    {
        std::vector<std::uint8_t> vcode, vsc, vsrc;
        const bool have_vc = read_file(kvc + "_v.bin", &vcode);
        const bool have_vs = read_file(kvc + "_vs.bin", &vsc);
        const bool have_vs_src = read_file(src + "_v.bin", &vsrc);
        if (have_vc && have_vs && have_vs_src) {
            std::size_t vcmp = 0, veq = 0, vse = 0;
            double vse_sum = 0.0, vss_sum = 0.0;
            for (int t = 0; t < T; ++t) {
                const int page = phys_page(t);
                const int off = t & 63;
                for (int h = 0; h < H; ++h) {
                    for (int g = 0; g < GPR; ++g) {
                        const std::size_t sidx =
                            static_cast<std::size_t>(GPR) * 64 *
                                (static_cast<std::size_t>(h) + H * static_cast<std::size_t>(page)) +
                            static_cast<std::size_t>(GPR) * off + g;
                        if (2 * sidx + 1 >= vsc.size()) { continue; }
                        const std::uint16_t raw = static_cast<std::uint16_t>(
                            vsc[2 * sidx] | (static_cast<std::uint16_t>(vsc[2 * sidx + 1]) << 8));
                        float amax = 0.0f;
                        for (int i = 0; i < 64; ++i) {
                            const int d = g * 64 + i;
                            const std::size_t so = 2 * (static_cast<std::size_t>(d) +
                                                        static_cast<std::size_t>(D) *
                                                            (h + static_cast<std::size_t>(H) * t));
                            const float v0 = bf16_to_f32(static_cast<std::uint16_t>(
                                vsrc[so] | (static_cast<std::uint16_t>(vsrc[so + 1]) << 8)));
                            const float a = std::fabs(v0);
                            if (a > amax) { amax = a; }
                        }
                        const std::uint16_t exp = f32_to_f16_rne(amax > 0.0f ? amax / 7.0f : 0.0f);
                        ++vcmp;
                        if (exp == raw) { ++veq; }
                        // value-level round trip, no rotation
                        const float s = f16_to_f32(raw);
                        for (int i = 0; i < 64; ++i) {
                            const int d = g * 64 + i;
                            const std::size_t b =
                                static_cast<std::size_t>(ROWB) * 64 *
                                    (static_cast<std::size_t>(h) + H * static_cast<std::size_t>(page)) +
                                static_cast<std::size_t>(ROWB) * off + d / 2;
                            if (b >= vcode.size()) { continue; }
                            const unsigned nib =
                                (d & 1) ? (vcode[b] >> 4) : (vcode[b] & 0x0Fu);
                            const float dec = static_cast<float>(static_cast<int>(nib ^ 8u) - 8) * s;
                            const std::size_t so = 2 * (static_cast<std::size_t>(d) +
                                                        static_cast<std::size_t>(D) *
                                                            (h + static_cast<std::size_t>(H) * t));
                            const float v0 = bf16_to_f32(static_cast<std::uint16_t>(
                                vsrc[so] | (static_cast<std::uint16_t>(vsrc[so + 1]) << 8)));
                            vse_sum += (static_cast<double>(dec) - v0) * (static_cast<double>(dec) - v0);
                            vss_sum += static_cast<double>(v0) * static_cast<double>(v0);
                            ++vse;
                        }
                    }
                }
            }
            std::printf("  V: stored scale == fp16(max|v_src|/7) : %zu / %zu (%.4f%%)   [JOIN KEY, no rotation]\n",
                        veq, vcmp, vcmp ? 100.0 * static_cast<double>(veq) / vcmp : 0.0);
            std::printf("  V: relL2(decoded, source) = %.5f  (%.2f%%) over %zu elements\n",
                        vss_sum > 0 ? std::sqrt(vse_sum / vss_sum) : 0.0,
                        100.0 * (vss_sum > 0 ? std::sqrt(vse_sum / vss_sum) : 0.0), vse);
        } else {
            std::printf("  V plane files missing (code=%d scale=%d src=%d)\n", (int)have_vc,
                        (int)have_vs, (int)have_vs_src);
        }
    }

    std::printf("\n-- (0) source-free invariants + the scale-plane JOIN KEY --\n");    std::printf("  groups with max|stored code| == 7 : %zu / %zu (%.4f%%)   [source-free]\n",
                inv_groups_ok, inv_groups,
                inv_groups ? 100.0 * static_cast<double>(inv_groups_ok) / static_cast<double>(inv_groups) : 0.0);
    std::printf("  stored scale == fp16(max|H k_src|/7) : %zu / %zu (%.4f%%)   [JOIN KEY, POST-rotation]\n",
                scaleeq, scalecmp,
                scalecmp ? 100.0 * static_cast<double>(scaleeq) / static_cast<double>(scalecmp) : 0.0);
    std::printf("  stored scale == fp16(max|k_src|/7)   : %zu / %zu (%.4f%%)   [JOIN KEY, PRE-rotation]\n",
                scaleeq_raw, scalecmp_raw,
                scalecmp_raw ? 100.0 * static_cast<double>(scaleeq_raw) / static_cast<double>(scalecmp_raw) : 0.0);
    {
        double mn, md, mx;
        rstat(ratios_rot, &mn, &md, &mx);
        std::printf("  ratio stored/fp16(max|Hk|/7): min=%.6f med=%.6f max=%.6f (n=%zu)\n", mn, md, mx,
                    ratios_rot.size());
        rstat(ratios_raw, &mn, &md, &mx);
        std::printf("  ratio stored/fp16(max|k|/7) : min=%.6f med=%.6f max=%.6f (n=%zu)\n", mn, md, mx,
                    ratios_raw.size());
    }

    std::printf("\n-- (1) writer replication vs the engine's stored bytes --\n");
    std::printf("  (a) rotate + scale amax_post/7 (SHIPPED model) : %.6f%% match\n",
                bytes_cmp ? 100.0 * static_cast<double>(bytes_eq) / static_cast<double>(bytes_cmp) : 0.0);
    std::printf("  (b) rotate + scale amax_pre/7  (116/116b cfg)  : %.6f%% match\n",
                bytes_cmp ? 100.0 * static_cast<double>(bytes_eq_b) / static_cast<double>(bytes_cmp) : 0.0);
    std::printf("  (c) NO rotate + scale amax_pre/7               : %.6f%% match\n",
                bytes_cmp ? 100.0 * static_cast<double>(bytes_eq_c) / static_cast<double>(bytes_cmp) : 0.0);
    std::printf("  code bytes compared = %zu\n", bytes_cmp);
    std::printf("  rows decoded = %zu (of %d tokens x %d heads = %d expected)\n", rows_done, T, H, T * H);

    std::printf("\n-- (2) shipped reader fidelity, REAL K plane vs the bf16 source --\n");
    std::printf("  relL2(decoded engine plane, source) over %zu elements = %.5f  (%.2f%%)\n", nelem, relL2,
                100.0 * relL2);
    std::printf("  worst single-row relL2 = %.5f\n", worst_row_rel);
    std::printf("  relL2 of the CURRENT shipped arithmetic on the SAME real K = %.5f  (%.2f%%)\n",
                ss_cur > 0 ? std::sqrt(se_cur / ss_cur) : 0.0,
                100.0 * (ss_cur > 0 ? std::sqrt(se_cur / ss_cur) : 0.0));
    std::printf("  V relL2 (i4, unrotated, same source)                        = see (0b) above\n");

    std::printf("\n-- (3) half-integer coset + projection cost, REAL 8-blocks --\n");
    std::printf("  8-blocks examined = %zu ; in the D8+1/2 coset = %zu (%.3f%%)\n", nblk, nblk_half,
                nblk ? 100.0 * static_cast<double>(nblk_half) / static_cast<double>(nblk) : 0.0);
    const double m_plain = se_plain / static_cast<double>(nblk * 8);
    const double m_proj = se_proj / static_cast<double>(nblk * 8);
    const double m_latt = se_latt / static_cast<double>(nblk * 8);
    const double m_scale = se_scale / static_cast<double>(nblk * 8);
    std::printf("  MSE plain rint (SHIPPED)      = %.6f   rel = %.6f (%.3f%%)\n", m_plain,
                std::sqrt(m_plain / m_scale), 100.0 * std::sqrt(m_plain / m_scale));
    std::printf("  MSE project-then-rint (warned) = %.6f   rel = %.6f (%.3f%%)\n", m_proj,
                std::sqrt(m_proj / m_scale), 100.0 * std::sqrt(m_proj / m_scale));
    std::printf("  MSE lattice point (reconstruct) = %.6f   rel = %.6f (%.3f%%)\n", m_latt,
                std::sqrt(m_latt / m_scale), 100.0 * std::sqrt(m_latt / m_scale));
    std::printf("  ratio MSE(proj)/MSE(plain)  = %.4f   (note claims 2.32x)\n", m_proj / m_plain);
    std::printf("  penalty in dB               = +%.3f dB   (note claims +3.65 dB)\n",
                10.0 * std::log10(m_proj / m_plain));
    std::printf("  gain  MSE(latt)/MSE(plain)  = %.4f\n", m_latt / m_plain);
    return 0;
}
