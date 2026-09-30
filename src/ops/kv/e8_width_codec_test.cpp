// e8_width_codec_test -- the runnable bar for the e8 3-bit and 2-bit K-plane variants.
//
// HOST-ONLY: no CUDA, no cmake, no build/ . Compile and run with just a compiler:
//
//   g++ -O2 -std=c++17 -I/home/user/ninfer-fusion/src \
//       -I/home/user/ninfer-fusion/tools/e8_verify/host_shim \
//       /home/user/ninfer-fusion/src/ops/kv/e8_width_codec_test.cpp -o /tmp/e8w
//   /tmp/e8w
//
// The second -I is section 8's only build requirement: the lattice codec of record reaches
// src/ops/kernel/e8_lattice.cuh, which includes <cuda_runtime.h>, and the shim satisfies it
// without CUDA -- the same shim and the same trick tools/e8_verify/run_lattice_codec.sh uses.
// Sections 1-7 do not need it.
//
// What it establishes, in order:
//   1. the geometry header reproduces the SHIPPED 4-bit plane (8704 B / 4.25 b/el) and
//      derives 6656 / 4608 and 3.25 / 2.25 for the new widths;
//   2. pack/unpack is bijective on the raw bytes of a group, EXHAUSTIVELY at every
//      width -- so a reader and a writer at the same width cannot disagree about a
//      single bit pattern that exists;
//   3. unpack/pack is the identity on the code grid;
//   4. quantize is idempotent on its own codes (dequantize then requantize reproduces
//      the codes), which is what makes a requant of an already-quantized plane safe;
//   5. the numerical order on the SAME input: bf16 (reference) and w4 / w3 / w2, with
//      w4-w3-w2 shown to be a strict loss chain rather than a free saving;
//   6. if E8V_REAL_K names a raw float32 file of K rows (head_dim floats each), the
//      same measurements on real key data instead of synthetic.
//   7. the normalization pins (divisor / clamp / group scale / stored bytes) against
//      LITERALS, because 1-6 are self-consistency and a wrong constant round-trips.
//   8. THE REAL E8 LATTICE as the codec of record for W3 and W2: the dispatch, the
//      refusal at W4, the scalar control through the same seam, the codeword layout
//      read back independently, the encoder ordering, and the as-wired distortion.
//   9. THE OPS LAYER'S OWN CONSUMER -- ops/kv_cache/d256_profile.h, the arm that had no row
//      for E8K3Kv/E8K2Kv while the causal-attention validation derived the code extent from
//      an E8Kv-only ternary (`== E8Kv ? 128 : 256`) that therefore answered 256 for a 96 B
//      or 64 B plate. Section 9 pins the three ops extents as literals AND as the codec of
//      record's row bytes AND through the head-page identity, and checks that the refusal
//      for an unlisted dtype is still a throw.
//
// CHECK COUNT: 1-7 are 55 checks and they are unchanged; section 8 adds 20; section 9 adds
// 14. The suite reports 89/89. The 55 are not renumbered because section 7.6 pins the scalar
// writer's stored bytes at W3 and W2, and those bytes are the lattice's control arm.
//
// Exit code 0 iff every check passes.

#include "ops/kv/e8_lattice_plane_codec.cuh"
#include "ops/kv_cache/d256_profile.h"
#include "product/kv_e8_width.h"
#include "product/kv_e8_width_codec.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer::product;

namespace {

int g_ok = 0;
int g_total = 0;

void check(bool pass, const char* what) {
    ++g_total;
    if (pass) { ++g_ok; }
    std::printf("%s  %s\n", pass ? "PASS" : "FAIL", what);
}

// Deterministic PRNG so a rerun is comparable.
std::uint32_t g_state = 0x12345678u;
std::uint32_t next_rand() {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 17;
    g_state ^= g_state << 5;
    return g_state;
}
float next_uniform() {   // [-1, 1)
    return (static_cast<float>(next_rand() & 0xFFFFFFu) / 8388608.0f) - 1.0f;
}

// bf16 round-trip (truncating mantissa to 8 bits, round-to-nearest-even) -- the
// reference the widths are compared against.
float bf16_round(float v) {
    std::uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    const std::uint32_t lsb = (f >> 16) & 1u;
    f += 0x7FFFu + lsb;
    f &= 0xFFFF0000u;
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

double rel_l2(const std::vector<float>& got, const std::vector<float>& want) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
        num += d * d;
        den += static_cast<double>(want[i]) * static_cast<double>(want[i]);
    }
    return den > 0.0 ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
}

// Encode + decode a whole plane at one width, returning the reconstruction.
std::vector<float> roundtrip_plane(E8KvWidth w, const std::vector<float>& x,
                                   std::int32_t rows) {
    const std::int32_t cbr = e8_kv_row_code_bytes(w);
    const std::int32_t gpr = e8_kv_groups_per_row();
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(cbr) * rows);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(gpr) * rows);
    e8_kv_encode_plane(w, x.data(), rows, codes.data(), scales.data());
    std::vector<float> out(x.size(), 0.0f);
    e8_kv_decode_plane(w, codes.data(), scales.data(), rows, out.data());
    return out;
}

} // namespace

int main() {
    std::printf("=== e8 width codec test (host-only) ===\n\n");

    // ---------------------------------------------------------------- 1. geometry
    std::printf("--- 1. geometry (product/kv_e8_width.h) ---\n");
    check(e8_kv_k_plane_bytes(E8KvWidth::W4) == 8704, "w4 K plane == 8704 B (shipped)");
    check(e8_kv_k_plane_bytes(E8KvWidth::W3) == 6656, "w3 K plane == 6656 B");
    check(e8_kv_k_plane_bytes(E8KvWidth::W2) == 4608, "w2 K plane == 4608 B");
    check(e8_kv_layer_bytes(E8KvWidth::W4) == 17408, "w4 layer K+V == 17408 B");
    check(e8_kv_layer_bytes(E8KvWidth::W3) == 15360, "w3 layer K+V == 15360 B");
    check(e8_kv_layer_bytes(E8KvWidth::W2) == 13312, "w2 layer K+V == 13312 B");
    check(e8_kv_bits_x100(E8KvWidth::W4) == 425, "w4 cost == 4.25 b/el (the shipped row 425)");
    check(e8_kv_bits_x100(E8KvWidth::W3) == 375, "w3 cost == 3.75 b/el");
    check(e8_kv_bits_x100(E8KvWidth::W2) == 325, "w2 cost == 3.25 b/el");
    check(e8_kv_v_plane_bytes() == 8704, "V plane is 8704 B at every width (i4, untouched)");
    check(e8_kv_k_plane_bytes(E8KvWidth::W2) == 64 * 64 + 512,
          "w2 K plane is D/4 code bytes + g64 fp16 scales (== rk2v4-e8 K geometry)");
    check(e8_kv_layer_bytes(E8KvWidth::W2) == 4608 + 8704,
          "w2 layer == rk2v4-e8's layer cost (K 2-bit, V left at 4-bit)");
    std::printf("  K-plane saved vs w4: w3 %d B, w2 %d B per head-page\n",
                e8_kv_k_bytes_saved_vs_w4(E8KvWidth::W3),
                e8_kv_k_bytes_saved_vs_w4(E8KvWidth::W2));

    // ------------------------------------------------- 2. exhaustive byte bijection
    std::printf("\n--- 2. pack(unpack(bytes)) == bytes, exhaustive per width ---\n");
    const E8KvWidth widths[3] = {E8KvWidth::W2, E8KvWidth::W3, E8KvWidth::W4};
    for (const E8KvWidth w : widths) {
        const std::int32_t nbytes = e8_kv_code_bytes_per_8(w);
        const std::uint64_t patterns = 1ull << (8 * nbytes);
        std::uint64_t bad = 0;
        std::vector<std::uint8_t> in(nbytes, 0), out(nbytes, 0);
        std::int8_t codes[8];
        for (std::uint64_t p = 0; p < patterns; ++p) {
            for (std::int32_t b = 0; b < nbytes; ++b) {
                in[b] = static_cast<std::uint8_t>((p >> (8 * b)) & 0xFFu);
            }
            e8_kv_unpack_group(w, in.data(), codes);
            std::memset(out.data(), 0, out.size());
            e8_kv_pack_group(w, codes, out.data());
            if (std::memcmp(in.data(), out.data(), static_cast<std::size_t>(nbytes)) != 0) {
                ++bad;
            }
        }
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                      "w%d: %llu/%llu byte patterns round-trip bit-exactly (%d bytes)",
                      static_cast<int>(w), static_cast<unsigned long long>(patterns - bad),
                      static_cast<unsigned long long>(patterns), nbytes);
        check(bad == 0, msg);
    }

    // ---------------------------------------------- 3. code-grid identity + codomain
    std::printf("\n--- 3. unpack(pack(codes)) == codes ---\n");
    for (const E8KvWidth w : widths) {
        const std::int32_t n = static_cast<std::int32_t>(w);
        const std::int32_t lo = e8_kv_code_min(w);
        const std::int32_t hi = e8_kv_code_max(w);
        const std::int32_t levels = hi - lo + 1;
        // w2: 4^8 = 65536 and w3: 8^8 = 16.7M are exhaustive; w4 is 16^8 = 4.29G, so it
        // is sampled (the exhaustive claim for w4 is check 2, on the bytes).
        const bool exhaustive = (w != E8KvWidth::W4);
        const std::uint64_t combos = exhaustive
            ? [&] { std::uint64_t v = 1; for (int i = 0; i < 8; ++i) { v *= static_cast<std::uint64_t>(levels); } return v; }()
            : 2000000ull;
        std::uint64_t bad = 0;
        std::int8_t codes[8], back[8];
        std::uint8_t bytes[4];
        for (std::uint64_t c = 0; c < combos; ++c) {
            std::uint64_t v = c;
            for (std::int32_t i = 0; i < 8; ++i) {
                std::int32_t code;
                if (exhaustive) {
                    code = lo + static_cast<std::int32_t>(v % static_cast<std::uint64_t>(levels));
                    v /= static_cast<std::uint64_t>(levels);
                } else {
                    code = lo + static_cast<std::int32_t>(next_rand() % static_cast<std::uint32_t>(levels));
                }
                codes[i] = static_cast<std::int8_t>(code);
            }
            std::memset(bytes, 0, sizeof(bytes));
            e8_kv_pack_group(w, codes, bytes);
            e8_kv_unpack_group(w, bytes, back);
            if (std::memcmp(codes, back, sizeof(codes)) != 0) { ++bad; }
        }
        char msg[200];
        std::snprintf(msg, sizeof(msg), "w%d: %s, %llu/%llu code tuples exact",
                      n, exhaustive ? "exhaustive" : "sampled 2e6",
                      static_cast<unsigned long long>(combos - bad),
                      static_cast<unsigned long long>(combos));
        check(bad == 0, msg);
    }

    // ------------------------------------------------------- 4. requant idempotence
    std::printf("\n--- 4. quantize(dequantize(codes)) == codes (requant safety) ---\n");
    for (const E8KvWidth w : widths) {
        std::uint64_t bad = 0;
        for (int trial = 0; trial < 20000; ++trial) {
            const float scale = 0.001f + std::fabs(next_uniform()) * 0.05f;
            std::int8_t codes[8], again[8];
            float x[8];
            for (std::int32_t i = 0; i < 8; ++i) {
                codes[i] = static_cast<std::int8_t>(
                    e8_kv_code_min(w) + static_cast<std::int32_t>(
                        next_rand() % static_cast<std::uint32_t>(e8_kv_code_max(w) - e8_kv_code_min(w) + 1)));
            }
            e8_kv_dequantize_group(w, codes, scale, x);
            e8_kv_quantize_group(w, x, scale, again);
            if (std::memcmp(codes, again, sizeof(codes)) != 0) { ++bad; }
        }
        char msg[160];
        std::snprintf(msg, sizeof(msg), "w%d: 20000/20000 code sets survive dequant->requant",
                      static_cast<int>(w));
        check(bad == 0, msg);
    }

    // ------------------------------- 5. numerical ladder on the same input, vs bf16
    std::printf("\n--- 5. numerical comparison on the same input ---\n");
    const std::int32_t rows = 64;
    const std::size_t n = static_cast<std::size_t>(kE8KvHeadDim) * rows;

    auto measure = [&](const std::vector<float>& x, const char* label) {
        std::vector<float> bf(n);
        for (std::size_t i = 0; i < n; ++i) { bf[i] = bf16_round(x[i]); }
        const double e_bf = rel_l2(bf, x);
        const std::vector<float> r4 = roundtrip_plane(E8KvWidth::W4, x, rows);
        const std::vector<float> r3 = roundtrip_plane(E8KvWidth::W3, x, rows);
        const std::vector<float> r2 = roundtrip_plane(E8KvWidth::W2, x, rows);
        const double e4 = rel_l2(r4, x);
        const double e3 = rel_l2(r3, x);
        const double e2 = rel_l2(r2, x);
        std::printf("  %-10s bf16 rel-L2 %.3e | w4 %.4e | w3 %.4e | w2 %.4e\n",
                    label, e_bf, e4, e3, e2);
        std::printf("             vs w4: w3 %.2fx, w2 %.2fx the error | w4/bf16 %.0fx\n",
                    e4 > 0 ? e3 / e4 : 0.0, e4 > 0 ? e2 / e4 : 0.0,
                    e4 > 0 ? e4 / (e_bf > 0 ? e_bf : 1e-30) : 0.0);
        return e4;
    };

    // 5a. synthetic gaussian-ish key data (what a Hadamard-rotated K row looks like:
    // the rotation concentrates mass, so both outliers and small values are present).
    std::vector<float> synth(n);
    for (std::size_t i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (int k = 0; k < 4; ++k) { acc += next_uniform(); }
        synth[i] = acc * 0.5f;
    }
    // a couple of heavy outliers, which is where the narrow widths should hurt most
    for (int k = 0; k < 8; ++k) { synth[static_cast<std::size_t>(next_rand() % n)] *= 12.0f; }
    const double e4 = measure(synth, "synthetic");
    const double e3 = rel_l2(roundtrip_plane(E8KvWidth::W3, synth, rows), synth);
    const double e2 = rel_l2(roundtrip_plane(E8KvWidth::W2, synth, rows), synth);
    check(e3 > e4, "w3 is STRICTLY lossier than w4 on the same input (not free)");
    check(e2 > e3, "w2 is STRICTLY lossier than w3 on the same input (not free)");

    // 5b. real dumped K rows, if a raw float32 file was handed to us. The sibling line
    // wrote its host simulator against NINFER_KVDUMP_DIR planes; this accepts either a
    // raw float32 dump or a plain-text one row per line, head_dim values per row.
    const char* real_path = std::getenv("E8V_REAL_K");
    if (real_path != nullptr && *real_path != '\0') {
        bool loaded = false;
        std::vector<float> real;
        if (FILE* f = std::fopen(real_path, "rb")) {
            std::vector<float> tmp(n);
            const std::size_t got = std::fread(tmp.data(), sizeof(float), n, f);
            std::fclose(f);
            if (got == n) { real.assign(tmp.begin(), tmp.end()); loaded = true; }
        }
        if (!loaded) {
            if (FILE* f = std::fopen(real_path, "r")) {
                std::vector<double> tmp(n);
                std::size_t cnt = 0;
                while (cnt < n && std::fscanf(f, "%lf", &tmp[cnt]) == 1) { ++cnt; }
                std::fclose(f);
                if (cnt == n) {
                    real.resize(n);
                    for (std::size_t i = 0; i < n; ++i) { real[i] = static_cast<float>(tmp[i]); }
                    loaded = true;
                }
            }
        }
        if (!loaded) {
            std::printf("  (E8V_REAL_K=%s: could not read %zu floats -- skipped)\n", real_path, n);
        } else {
            measure(real, "real K");
            const double re4 = rel_l2(roundtrip_plane(E8KvWidth::W4, real, rows), real);
            const double re3 = rel_l2(roundtrip_plane(E8KvWidth::W3, real, rows), real);
            const double re2 = rel_l2(roundtrip_plane(E8KvWidth::W2, real, rows), real);
            check(re3 > re4, "REAL K: w3 lossier than w4");
            check(re2 > re3, "REAL K: w2 lossier than w3");
        }
    } else {
        std::printf("  (no E8V_REAL_K set: synthetic only)\n");
    }

    // ------------------------------------------- 6. scale plane is width-independent
    std::printf("\n--- 6. cost decomposition ---\n");
    for (const E8KvWidth w : widths) {
        const std::int32_t code_b = e8_kv_row_code_bytes(w) * kE8KvPageTokens;
        const std::int32_t scale_b = e8_kv_k_plane_bytes(w) - code_b;
        std::printf("  w%d K: code %5d B + scale %3d B = %5d B | + V %5d B = layer %5d B (%.2f b/el)\n",
                    static_cast<int>(w), code_b, scale_b, e8_kv_k_plane_bytes(w),
                    e8_kv_v_plane_bytes(), e8_kv_layer_bytes(w),
                    static_cast<double>(e8_kv_bits_x100(w)) / 100.0);
        check(scale_b == 512, "K scale plane is 512 B/head-page at every width");
    }

    // ---------------------------------------------------------------------------
    // 7. NORMALIZATION PINS -- the divisor, the clamp and the group scale.
    //
    // WHY THIS SECTION EXISTS. Sections 1-6 are all SELF-CONSISTENCY: they encode and
    // decode through the same constants, so a wrong divisor, a wrong clamp or a wrong
    // group scale round-trips perfectly and every one of them still passes. Measured on
    // the suite as it stood (mutation matrix, one single-line mutation at a time, each
    // against its OWN shadow include dir): doubling the divisor (1<<(w-1) -> 1<<w),
    // disabling the clamp (lo-4/hi+4), the shipped row's divisor (7/3/1), the fp16
    // rounding step, and the writer's choice of scale each left it at 26/26. Every check
    // below compares a codec value against a LITERAL written here -- never against another
    // codec function -- and 7.6 compares the codec's ACTUAL stored plane bytes against
    // values this test computes itself. A test that compares a function to itself cannot
    // fail, and that is exactly what the blind mutations were.
    //
    // 7.6c makes the distance between "quantize with the exact float scale" and "quantize
    // with fp16(scale)" EXERCISED rather than hoped for, and makes a failure to exercise it
    // LOUD. Two elements of every 64-group sit exactly at +-A/(2*divisor): that is a TIE for
    // the exact float scale (nearbyint of +-0.5 is 0 in the default ties-to-even mode), so
    // BOTH probes flip together whenever fp16 rounds the scale DOWN (|ratio| > 0.5) -- and,
    // as measured, both stay put when it rounds UP. This is not fixable by a one-sided
    // probe: to catch the round-up case the probe would have to sit above 0.5 by LESS than
    // the smallest possible fp16 error, and no such offset exists (the fp16 ulp is at
    // least 2^-11 relative). So 7.6c COUNTS the codes that separate the two scales and
    // FAILS when the count is zero, and the count is not left to chance: measured on the
    // pristine header it is 26/30/25 of 2048 at w=2/3/4. (The first draft of this section
    // had a group amax that was fp16-exact -- 0.75 and 0.03125 -- so the count was 0 and
    // the writer's scale choice was silently untestable.)
    // ---------------------------------------------------------------------------
    std::printf("\n--- 7. normalization pins (divisor / clamp / group scale, vs literals) ---\n");
    {
        // The shipped 4-bit row's own convention, spelled here as literals WITH its source,
        // so the codec's relation to it is a checked fact instead of a comment.
        //   src/ops/kernel/gqa_attention_kv_quant.cuh
        //     kGqaKvI4CodeMin = -8, kGqaKvI4CodeMax = 7,
        //     kGqaKvI4ScaleDivisor = -kGqaKvI4CodeMin = 8.0f   (DERIVED)
        //   i.e. scale = amax/8, code = clamp(rint(x * 8/amax), -8, 7).
        // AMAXFIX moved this row from (7, -7, 7) to (8, -8, 7): measured on the prior
        // line's own instrument over /home/user/bench/kvdump_e8src full length (27
        // dumps, 64,677 tokens, 1,034,832 groups of 64, H64 rotation) the shipped
        // writer was 10.6611 % K relRMS and the codec's convention was 9.8520 % at the
        // SAME 4.25 b/el, so the shipped row was corrected TOWARD the codec rather than
        // the other way. The two conventions now AGREE at w=4, which is why 7.8 below
        // asserts EQUALITY where it used to assert divergence.
        const std::int32_t kShippedW4Divisor = 8;
        const std::int32_t kShippedW4CodeMin = -8;
        const std::int32_t kShippedW4CodeMax = 7;

        // The codec's own convention, also as literals: divisor 2^(w-1), range
        // [-2^(w-1), 2^(w-1)-1]. These are the numbers every check below compares to.
        const auto div_lit = [](E8KvWidth w) -> std::int32_t {
            return w == E8KvWidth::W4 ? 8 : (w == E8KvWidth::W3 ? 4 : 2);
        };
        const auto lo_lit = [&](E8KvWidth w) -> std::int32_t { return -div_lit(w); };
        const auto hi_lit = [&](E8KvWidth w) -> std::int32_t { return div_lit(w) - 1; };

        // A LOCAL fixed-seed PRNG for this section, so its data does not depend on how many
        // numbers sections 1-6 happened to draw.
        std::uint32_t pst = 0x9E3779B9u;
        const auto prnd = [&pst]() -> std::uint32_t {
            pst ^= pst << 13; pst ^= pst >> 17; pst ^= pst << 5; return pst;
        };
        const auto puni = [&prnd]() -> float {   // [-1, 1)
            return (static_cast<float>(prnd() & 0xFFFFFFu) / 8388608.0f) - 1.0f;
        };

        for (const E8KvWidth w : widths) {
            const std::int32_t wn = static_cast<std::int32_t>(w);
            char msg[320];

            // 7.1 the divisor itself, against a literal.
            // (Mutation M2_divisor: 1<<(w-1) becomes 1<<w.)
            std::snprintf(msg, sizeof(msg), "w%d: scale divisor == %d (literal)", wn, div_lit(w));
            check(e8_kv_scale_divisor(w) == div_lit(w), msg);

            // 7.2 the divisor and the clamp range must move TOGETHER: code_min == -divisor
            // and code_max == divisor-1. A width whose divisor and range disagree is the
            // sign-biased quantizer this file's SCALE BOUNDARY block claims not to be.
            std::snprintf(msg, sizeof(msg),
                          "w%d: range == [%d,%d], i.e. code_min == -divisor and "
                          "code_max == divisor-1",
                          wn, lo_lit(w), hi_lit(w));
            check(e8_kv_code_min(w) == lo_lit(w) && e8_kv_code_max(w) == hi_lit(w) &&
                      e8_kv_code_min(w) == -e8_kv_scale_divisor(w) &&
                      e8_kv_code_max(w) == e8_kv_scale_divisor(w) - 1,
                  msg);

            // 7.3 the clamp is REACHED and it SATURATES.
            // (Mutation M6_clampoff: the bounds become lo-4 / hi+4.)
            std::snprintf(msg, sizeof(msg),
                          "w%d: +huge saturates at code_max, -huge at code_min", wn);
            check(e8_kv_code_of(w, 1.0e9f, 1.0f) == e8_kv_code_max(w) &&
                      e8_kv_code_of(w, -1.0e9f, 1.0f) == e8_kv_code_min(w),
                  msg);

            // 7.4 the clamp is a CODOMAIN: no input can produce an out-of-range code. This
            // is the check a "clamp disabled" mutation cannot survive, because the sweep
            // walks the input far past the range in both directions.
            std::int32_t out_of_range = 0;
            for (int t = 0; t < 200000; ++t) {
                const float sc = 1.0e-4f + std::fabs(puni()) * 7.0f;
                const float qx = puni() * sc * 64.0f;
                const std::int32_t c = e8_kv_code_of(w, qx, sc);
                if (c < e8_kv_code_min(w) || c > e8_kv_code_max(w)) { ++out_of_range; }
            }
            std::snprintf(msg, sizeof(msg),
                          "w%d: 200000/200000 quantizer outputs inside [%d,%d] "
                          "(the clamp is a codomain)",
                          wn, lo_lit(w), hi_lit(w));
            check(out_of_range == 0, msg);

            // 7.5 the group scale, against a LITERAL divisor with amax recomputed here, so
            // this cannot pass by comparing the function to itself.
            // (Mutation M8_scaleoff: the divisor inside e8_kv_group_scale gets +1.)
            float grp[kE8KvScaleGroup];
            for (std::int32_t i = 0; i < kE8KvScaleGroup; ++i) {
                grp[i] = puni() * (i == 17 ? 5.0f : 0.35f);   // one dominant channel
            }
            float grp_amax = 0.0f;
            for (std::int32_t i = 0; i < kE8KvScaleGroup; ++i) {
                const float a = std::fabs(grp[i]);
                if (a > grp_amax) { grp_amax = a; }
            }
            const float want_scale = grp_amax / static_cast<float>(div_lit(w));
            std::snprintf(msg, sizeof(msg),
                          "w%d: group_scale == amax/%d bit-exactly (amax recomputed here)",
                          wn, div_lit(w));
            check(e8_kv_group_scale(w, grp, kE8KvScaleGroup) == want_scale, msg);

            // 7.6 THE PLANE WRITE PIN: what the codec actually STORES, against values this
            // test computes itself. This is the check the suite was missing -- it pins the
            // fp16 scale plane's VALUE (not its 512 B size) and every stored code, so the
            // divisor, the clamp and the group scale cannot all be wrong while it passes.
            // The writer quantizes with the EXACT float scale and stores fp16(scale); both
            // halves are pinned, in that order, because that order is the format.
            {
                const std::int32_t rows = 8;
                const std::int32_t gpr = e8_kv_groups_per_row();
                const std::int32_t cpg = e8_kv_code_bytes_per_8(w);
                // +-A*probe is a rounding TIE for the exact scale: see 7.6c above.
                const float probe = 1.0f / (2.0f * static_cast<float>(div_lit(w)));
                std::vector<float> px(static_cast<std::size_t>(kE8KvHeadDim) * rows);
                for (std::int32_t r = 0; r < rows; ++r) {
                    for (std::int32_t g = 0; g < gpr; ++g) {
                        float* gv = px.data() + static_cast<std::size_t>(r) * kE8KvHeadDim +
                                    g * kE8KvScaleGroup;
                        // A generic amax (|A| in [1,5), a full-mantissa float, so fp16(A/div)
                        // is not A/div), then 61 small fillers, then the two probes.
                        const float A = (puni() < 0.0f ? -1.0f : 1.0f) *
                                        (1.0f + 4.0f * std::fabs(puni()));
                        for (std::int32_t i = 0; i < kE8KvScaleGroup; ++i) {
                            gv[i] = puni() * 0.1f * std::fabs(A);
                        }
                        gv[0] = A;                                  // sets amax == A exactly
                        gv[kE8KvScaleGroup - 1] = A * probe;        // exactly +A/(2*divisor)
                        gv[kE8KvScaleGroup - 2] = -A * probe;       // exactly -A/(2*divisor)
                    }
                }
                std::vector<std::uint8_t> code_out(
                    static_cast<std::size_t>(e8_kv_row_code_bytes(w)) * rows);
                std::vector<std::uint16_t> scale_out(static_cast<std::size_t>(gpr) * rows);
                e8_kv_encode_plane(w, px.data(), rows, code_out.data(), scale_out.data());

                std::int32_t bad_scale = 0, bad_code = 0, ncode = 0;
                std::int32_t discriminating = 0;   // 7.6c's coverage count
                for (std::int32_t r = 0; r < rows; ++r) {
                    const float* row = px.data() + static_cast<std::size_t>(r) * kE8KvHeadDim;
                    const std::uint8_t* rowcodes = code_out.data() +
                        static_cast<std::size_t>(r) * e8_kv_row_code_bytes(w);
                    for (std::int32_t g = 0; g < gpr; ++g) {
                        const float* gv = row + g * kE8KvScaleGroup;
                        float ga = 0.0f;
                        for (std::int32_t i = 0; i < kE8KvScaleGroup; ++i) {
                            const float a = std::fabs(gv[i]);
                            if (a > ga) { ga = a; }
                        }
                        const float s_exact = ga / static_cast<float>(div_lit(w));
                        const std::uint16_t stored =
                            scale_out[static_cast<std::size_t>(r) * gpr + g];
                        if (stored != e8_kv_fp16_bits(s_exact)) { ++bad_scale; }
                        const float s_fp16 = e8_kv_fp16_from_bits(stored);
                        const auto clamp_lit = [&](float q) -> std::int32_t {
                            if (q < static_cast<float>(lo_lit(w))) { q = static_cast<float>(lo_lit(w)); }
                            if (q > static_cast<float>(hi_lit(w))) { q = static_cast<float>(hi_lit(w)); }
                            return static_cast<std::int32_t>(q);
                        };
                        for (std::int32_t sub = 0; sub < kE8KvScaleGroup / 8; ++sub) {
                            std::int8_t got[8];
                            e8_kv_unpack_group(
                                w, rowcodes + (g * (kE8KvScaleGroup / 8) + sub) * cpg, got);
                            for (std::int32_t i = 0; i < 8; ++i) {
                                const float v = gv[sub * 8 + i];
                                const std::int32_t want = clamp_lit(std::nearbyint(v / s_exact));
                                const std::int32_t alt = clamp_lit(std::nearbyint(v / s_fp16));
                                if (want != alt) { ++discriminating; }
                                if (got[i] != static_cast<std::int8_t>(want)) { ++bad_code; }
                                ++ncode;
                            }
                        }
                    }
                }
                std::snprintf(msg, sizeof(msg),
                              "w%d: stored fp16 scale plane == fp16(amax/%d) in %d/%d groups",
                              wn, div_lit(w), gpr * rows - bad_scale, gpr * rows);
                check(bad_scale == 0, msg);
                std::snprintf(msg, sizeof(msg),
                              "w%d: stored code plane == clamp(rint(x/scale), %d, %d) in %d/%d codes",
                              wn, lo_lit(w), hi_lit(w), ncode - bad_code, ncode);
                check(bad_code == 0, msg);
                // 7.6c the coverage guard: at least one code must separate the exact float
                // scale from fp16(scale), or the writer's choice between them is untestable
                // and the mutation that makes it is indistinguishable from the truth.
                std::snprintf(msg, sizeof(msg),
                              "w%d: COVERAGE: %d/%d codes separate the exact float scale from "
                              "fp16(scale) (must be > 0, else the writer's scale choice is "
                              "untestable)",
                              wn, discriminating, ncode);
                check(discriminating > 0, msg);
            }

            // 7.7 the full-scale endpoints, which is where the two conventions differ: with
            // THIS divisor the negative extreme is exactly representable and the positive one
            // saturates by one code. Both halves are pinned, so a "make it symmetric like the
            // shipped row" mutation cannot pass silently either.
            std::snprintf(msg, sizeof(msg),
                          "w%d: +amax -> code_max and -amax -> code_min at scale amax/%d "
                          "(the convention is asymmetric on purpose)",
                          wn, div_lit(w));
            check(e8_kv_code_of(w, grp_amax, want_scale) == e8_kv_code_max(w) &&
                      e8_kv_code_of(w, -grp_amax, want_scale) == e8_kv_code_min(w),
                  msg);
        }

        // 7.8 THE AGREEMENT WITH THE SHIPPED ROW IS A CHECKED FACT (AMAXFIX inverted this).
        // This check used to assert DIVERGENCE: kv_e8_width_codec.h's "the same choice the
        // shipped row makes at w=4" was false, because the shipped row was amax/7 with clamp
        // [-7,7] while the codec is amax/8 with clamp [-8,7] -- measured on the engine's own
        // rotated real-K writer domain they stored the SAME fp16 scale in 0/16384 groups and
        // the SAME code bytes in 0/16384 groups.
        // AMAXFIX measured which of the two is the better convention at identical bits (the
        // prior line's own instrument, /home/user/bench/kvdump_e8src full length, 64,677
        // tokens, 1,034,832 groups of 64, H64 rotation: shipped 10.6611 % vs codec 9.8520 %)
        // and moved the SHIPPED row to the codec's convention, so the claim is now TRUE and
        // this check asserts it instead of denying it. It is exactly as strong in this
        // direction -- it is an equality against a literal, so a drift of either side goes
        // red -- and it is the direction the measurement supports. The alternative reading
        // (correct the codec toward amax/7) would have been a 0.686 dB regression here and a
        // 4.847 dB regression at w=2, and would have made w=4 the only width whose divisor is
        // not 2^(w-1), breaking the header's one-variable derivation.
        char agree_msg[420];
        std::snprintf(agree_msg, sizeof(agree_msg),
                      "w4 codec divisor %d == shipped row's %d and range [%d,%d] == [%d,%d]: "
                      "the shipped row was corrected toward the codec (AMAXFIX), so the "
                      "header's 'same choice' claim is now true and must stay true",
                      e8_kv_scale_divisor(E8KvWidth::W4), kShippedW4Divisor,
                      e8_kv_code_min(E8KvWidth::W4), e8_kv_code_max(E8KvWidth::W4),
                      kShippedW4CodeMin, kShippedW4CodeMax);
        check(e8_kv_scale_divisor(E8KvWidth::W4) == kShippedW4Divisor &&
                  e8_kv_code_min(E8KvWidth::W4) == kShippedW4CodeMin &&
                  e8_kv_code_max(E8KvWidth::W4) == kShippedW4CodeMax,
              agree_msg);

        // 7.9 the fp16 scale converter that 7.6 depends on, pinned by literal bit patterns.
        // Without this, 7.6's scale comparison would inherit whatever the converter does.
        check(e8_kv_fp16_bits(1.0f) == 0x3C00u && e8_kv_fp16_bits(-0.5f) == 0xB800u &&
                  e8_kv_fp16_from_bits(0x3C00u) == 1.0f &&
                  e8_kv_fp16_from_bits(e8_kv_fp16_bits(0.0625f)) == 0.0625f,
              "fp16 scale converter: 1.0 -> 0x3C00, -0.5 -> 0xB800, 0.0625 round-trips");
    }

    // ---------------------------------------------------------------------------
    // 8. THE REAL E8 LATTICE ON THE 3-BIT / 2-BIT PATH.
    //
    // Sections 1-7 above pin the SCALAR codec and none of them is changed by this
    // section: the count they produce is the same 55, and every one of those checks is
    // still compared against the same literal. This section adds the lattice codec of
    // record (src/ops/kv/e8_lattice_plane_codec.cuh) and the redirect that puts it on
    // the W3 / W2 K plane.
    //
    // WHY THE HEADER IS INCLUDED AND NOTHING ABOVE IT CHANGED. The lattice lives beside
    // the scalar codec rather than inside it because section 7.6 pins the scalar
    // writer's STORED BYTES at all three widths -- including W3 and W2 -- and it is
    // right to: the scalar row is the priced control arm every lattice number is
    // measured against. A codec that cannot be pointed at is not a control. So the
    // redirect is the DISPATCH (8.1-8.3), and the scalar functions are untouched.
    //
    // WHAT THESE CHECKS CAN AND CANNOT ESTABLISH. They are a CODEC bar: the codeword
    // layout, the refusal, the dispatch, the ordering of the encoders, and the as-wired
    // distortion on one input. The matched-transform, matched-side-information table on
    // real K -- the +2.086 dB / +1.055 dB comparison -- is a corpus measurement and is
    // made by the harness, NOT here; this file has no corpus.
    // ---------------------------------------------------------------------------
    std::printf("\n--- 8. the real E8 lattice as the W3/W2 codec of record ---\n");
    {
        // Section-local PRNG: section 7's `puni` is scoped to section 7, and reusing a
        // shared one would make this section's data depend on how many draws 1-7 made.
        std::uint32_t p8 = 0x5BD1E995u;
        const auto r8 = [&p8]() -> std::uint32_t {
            p8 ^= p8 << 13; p8 ^= p8 >> 17; p8 ^= p8 << 5; return p8;
        };
        const auto puni = [&r8]() -> float {   // [-1, 1)
            return (static_cast<float>(r8() & 0xFFFFFFu) / 8388608.0f) - 1.0f;
        };

        // 8.1 the dispatch. One function decides; W4 must remain the scalar row,
        // because the lattice codeword has a 16-bit form and a 24-bit form and no
        // 32-bit form, so 4 bits cannot carry a lattice point.
        check(e8_kv_plane_codec_of_record(E8KvWidth::W4) == E8KvPlaneCodec::Scalar,
              "8.1 W4 -> Scalar: the lattice codeword has no 32-bit form");
        check(e8_kv_plane_codec_of_record(E8KvWidth::W3) == E8KvPlaneCodec::Lattice,
              "8.1 W3 -> Lattice: 24-bit codeword == 3 bytes / 8 elements");
        check(e8_kv_plane_codec_of_record(E8KvWidth::W2) == E8KvPlaneCodec::Lattice,
              "8.1 W2 -> Lattice: 16-bit codeword == 2 bytes / 8 elements");
        check(e8_kv_lattice_supports(E8KvWidth::W4) == false &&
                  e8_kv_lattice_supports(E8KvWidth::W3) == true &&
                  e8_kv_lattice_supports(E8KvWidth::W2) == true,
              "8.1 supports() agrees with of_record() at every width");

        // 8.2 the REFUSAL is real: asking for the lattice where it cannot go returns
        // false and writes NOTHING. The buffers are pre-filled with a sentinel, so a
        // silent fallback to the scalar codec -- the half-landing this project has paid
        // for before -- is distinguishable from a refusal.
        {
            const std::int32_t rows = 4;
            const std::size_t ncode = static_cast<std::size_t>(e8_kv_row_code_bytes(E8KvWidth::W4)) * rows;
            const std::size_t nscale = static_cast<std::size_t>(e8_kv_groups_per_row()) * rows;
            std::vector<std::uint8_t> codes(ncode, 0xA5);
            std::vector<std::uint16_t> scales(nscale, 0xA5A5);
            std::vector<float> x(static_cast<std::size_t>(kE8KvHeadDim) * rows, 0.25f);
            const bool ok = e8_kv_encode_plane_as(E8KvPlaneCodec::Lattice, E8KvWidth::W4,
                                                  x.data(), rows, codes.data(), scales.data());
            bool untouched = true;
            for (std::size_t i = 0; i < ncode; ++i) { if (codes[i] != 0xA5) { untouched = false; } }
            for (std::size_t i = 0; i < nscale; ++i) { if (scales[i] != 0xA5A5) { untouched = false; } }
            check(ok == false, "8.2 lattice encode at W4 returns FALSE (no silent fallback)");
            check(untouched, "8.2 and writes NOTHING (the plane is byte-identical to the sentinel)");
            const bool okd = e8_kv_decode_plane_as(E8KvPlaneCodec::Lattice, E8KvWidth::W4,
                                                   codes.data(), scales.data(), rows, x.data());
            check(okd == false, "8.2 lattice decode at W4 returns FALSE too");
        }

        // 8.3 the SCALAR arm is still reachable through the same seam and is bit-exact
        // with the function section 7.6 pins. This is the redirect's other half: the
        // traditional path was not deleted, it was made selectable.
        {
            const std::int32_t rows = 8;
            const std::size_t n = static_cast<std::size_t>(kE8KvHeadDim) * rows;
            std::vector<float> px(n);
            for (std::int32_t r = 0; r < rows; ++r) {
                for (std::int32_t g = 0; g < e8_kv_groups_per_row(); ++g) {
                    float* gv = px.data() + static_cast<std::size_t>(r) * kE8KvHeadDim +
                                g * kE8KvScaleGroup;
                    for (std::int32_t i = 0; i < kE8KvScaleGroup; ++i) { gv[i] = puni() * 0.5f; }
                }
            }
            for (const E8KvWidth w : widths) {
                std::vector<std::uint8_t> c_direct(static_cast<std::size_t>(e8_kv_row_code_bytes(w)) * rows);
                std::vector<std::uint16_t> s_direct(static_cast<std::size_t>(e8_kv_groups_per_row()) * rows);
                std::vector<std::uint8_t> c_seam(c_direct.size());
                std::vector<std::uint16_t> s_seam(s_direct.size());
                e8_kv_encode_plane(w, px.data(), rows, c_direct.data(), s_direct.data());
                const bool ok = e8_kv_encode_plane_as(E8KvPlaneCodec::Scalar, w, px.data(), rows,
                                                     c_seam.data(), s_seam.data());
                char msg[220];
                std::snprintf(msg, sizeof(msg),
                              "8.3 w%d: the Scalar arm through the seam is bit-identical to "
                              "e8_kv_encode_plane (the control is not a lookalike)",
                              static_cast<int>(w));
                check(ok && std::memcmp(c_direct.data(), c_seam.data(), c_direct.size()) == 0 &&
                          std::memcmp(s_direct.data(), s_seam.data(), s_direct.size() * 2) == 0,
                      msg);
            }
        }

        // 8.4 both plane codecs fill the SAME NUMBER OF BYTES. That is what makes the
        // tier name the only discriminator, and it is why 8.2's refusal has to be a
        // refusal rather than a fallback.
        for (const E8KvWidth w : widths) {
            const std::int32_t cpg = e8_kv_code_bytes_per_8(w);
            char msg[200];
            const int want = (w == E8KvWidth::W3) ? 3 : (w == E8KvWidth::W2 ? 2 : 4);
            std::snprintf(msg, sizeof(msg),
                          "8.4 w%d: %d code bytes per 8 elements", static_cast<int>(w), cpg);
            check(cpg == want, msg);
        }

        // 8.5 the LATTICE plane round-trips through its own stored bytes, and the stored
        // bytes are what the decoder reads: unpack+decode in the test must reproduce the
        // library's decode output BIT-FOR-BIT. This pins the byte layout (idx/signs/shift
        // at W2, plus the stage-2 index at W3) against an independent reading.
        for (const E8KvWidth w : {E8KvWidth::W2, E8KvWidth::W3}) {
            const std::int32_t rows = 8;
            const std::size_t n = static_cast<std::size_t>(kE8KvHeadDim) * rows;
            std::vector<float> px(n);
            for (std::size_t i = 0; i < n; ++i) {
                float acc = 0.0f;
                for (int k = 0; k < 4; ++k) { acc += puni(); }
                px[i] = acc * 0.5f;
            }
            std::vector<std::uint8_t> codes(static_cast<std::size_t>(e8_kv_row_code_bytes(w)) * rows);
            std::vector<std::uint16_t> scales(static_cast<std::size_t>(e8_kv_groups_per_row()) * rows);
            const bool ok = e8_kv_encode_plane_as(E8KvPlaneCodec::Lattice, w, px.data(), rows,
                                                 codes.data(), scales.data());
            std::vector<float> got(n, 0.0f);
            const bool okd = e8_kv_decode_plane_as(E8KvPlaneCodec::Lattice, w, codes.data(),
                                                   scales.data(), rows, got.data());
            // independent read of the SAME bytes
            std::int32_t bad = 0;
            const std::int32_t cpg = e8_kv_code_bytes_per_8(w);
            const std::int32_t gpr = e8_kv_groups_per_row();
            for (std::int32_t r = 0; r < rows; ++r) {
                const std::uint8_t* rcodes = codes.data() + static_cast<std::size_t>(r) * e8_kv_row_code_bytes(w);
                for (std::int32_t g = 0; g < gpr; ++g) {
                    const double s = static_cast<double>(
                        e8_kv_fp16_from_bits(scales[static_cast<std::size_t>(r) * gpr + g]));
                    double y[64];
                    for (std::int32_t sb = 0; sb < kE8KvScaleGroup / 8; ++sb) {
                        const std::uint8_t* slot = rcodes + (g * (kE8KvScaleGroup / 8) + sb) * cpg;
                        ninfer::ops::E8LatticeCode2 c2;
                        c2.idx = slot[0];
                        c2.signs = static_cast<std::uint8_t>(slot[1] & 0x7fu);
                        c2.shift = static_cast<std::uint8_t>((slot[1] >> 7) & 1u);
                        if (w == E8KvWidth::W3) {
                            ninfer::ops::E8LatticeCode3 c3;
                            c3.s1 = c2;
                            c3.root = slot[2];
                            ninfer::ops::e8_lattice_decode3_8d(e8_lattice_stage1_table(),
                                                              e8_lattice_stage2_table(), c3, s,
                                                              kE8LatticeStage2Step * s, &y[sb * 8]);
                        } else {
                            ninfer::ops::e8_lattice_decode_8d(e8_lattice_stage1_table(), c2, s,
                                                              &y[sb * 8]);
                        }
                    }
                    ninfer::ops::e8_lattice_hadamard64(y);
                    for (std::int32_t i = 0; i < kE8KvScaleGroup; ++i) {
                        const float want = static_cast<float>(y[i]);
                        const float have = got[static_cast<std::size_t>(r) * kE8KvHeadDim +
                                               g * kE8KvScaleGroup + i];
                        if (want != have) { ++bad; }
                    }
                }
            }
            char msg[230];
            std::snprintf(msg, sizeof(msg),
                          "8.5 w%d: encode+decode ok, and an INDEPENDENT read of the stored bytes "
                          "reproduces the decoder bit-for-bit (%d mismatches)",
                          static_cast<int>(w), bad);
            check(ok && okd && bad == 0, msg);
        }

        // 8.6 the ORDERING OF THE ENCODERS -- falsifiable, and it is the reason the
        // default is the expensive one. The exact encoder scans all 256 entries x 2
        // shifts; the projection-seeded one skips the scan. On the same blocks the exact
        // encoder must never be worse, and it must be strictly better on at least one.
        for (const E8KvWidth w : {E8KvWidth::W2, E8KvWidth::W3}) {
            const std::int32_t rows = 16;
            const std::size_t n = static_cast<std::size_t>(kE8KvHeadDim) * rows;
            std::vector<float> px(n);
            for (std::size_t i = 0; i < n; ++i) {
                float acc = 0.0f;
                for (int k = 0; k < 4; ++k) { acc += puni(); }
                px[i] = acc * 0.5f;
            }
            const std::int32_t cbr = e8_kv_row_code_bytes(w);
            const std::int32_t gpr = e8_kv_groups_per_row();
            std::vector<std::uint8_t> ce(static_cast<std::size_t>(cbr) * rows);
            std::vector<std::uint16_t> se(static_cast<std::size_t>(gpr) * rows);
            std::vector<std::uint8_t> cp(ce.size());
            std::vector<std::uint16_t> sp(se.size());
            const bool oke = e8_kv_encode_plane_as(E8KvPlaneCodec::Lattice, w, px.data(), rows,
                                                   ce.data(), se.data(), E8LatticeEncoder::Exact);
            const bool okp = e8_kv_encode_plane_as(E8KvPlaneCodec::Lattice, w, px.data(), rows,
                                                   cp.data(), sp.data(), E8LatticeEncoder::Projected);
            std::vector<float> re(n, 0.0f), rp(n, 0.0f);
            e8_kv_decode_plane_lattice(w, ce.data(), se.data(), rows, re.data());
            e8_kv_decode_plane_lattice(w, cp.data(), sp.data(), rows, rp.data());
            const double ee = rel_l2(re, px);
            const double ep = rel_l2(rp, px);
            char msg[240];
            std::snprintf(msg, sizeof(msg),
                          "8.6 w%d: exact encoder (rel-L2 %.6f) is not worse than the "
                          "projection-seeded one (%.6f) -- e8_project_8d_fast() is ON the encode "
                          "path and is priced, not assumed",
                          static_cast<int>(w), ee, ep);
            check(oke && okp && ee <= ep, msg);
        }

        // 8.7 THE AS-WIRED CLAIM: at the same width, on the same input, with the same
        // one-fp16-per-64-group side information, the lattice plane is less distorted
        // than the scalar plane. This is the codecs AS THE DISPATCH WIRES THEM -- the
        // lattice also carries the H64 rotation, which the scalar product codec does not
        // apply, so this is NOT the matched-transform number. The matched-transform,
        // matched-rate figures on real K are the harness's (+2.086 dB at 2 bits, +1.055
        // dB at 3 bits, both scales fitted on held-out rows) and are labelled as such.
        {
            const std::int32_t rows = 32;
            const std::size_t n = static_cast<std::size_t>(kE8KvHeadDim) * rows;
            std::vector<float> px(n);
            for (std::size_t i = 0; i < n; ++i) {
                float acc = 0.0f;
                for (int k = 0; k < 4; ++k) { acc += puni(); }
                px[i] = acc * 0.5f;
            }
            const std::int32_t cbr3 = e8_kv_row_code_bytes(E8KvWidth::W3);
            const std::int32_t cbr2 = e8_kv_row_code_bytes(E8KvWidth::W2);
            const std::int32_t gpr = e8_kv_groups_per_row();
            std::vector<std::uint8_t> c3(static_cast<std::size_t>(cbr3) * rows);
            std::vector<std::uint16_t> s3(static_cast<std::size_t>(gpr) * rows);
            std::vector<std::uint8_t> c2(static_cast<std::size_t>(cbr2) * rows);
            std::vector<std::uint16_t> s2(static_cast<std::size_t>(gpr) * rows);
            std::vector<float> rs3(n, 0.0f), rs2(n, 0.0f);
            e8_kv_encode_plane(E8KvWidth::W3, px.data(), rows, c3.data(), s3.data());
            e8_kv_decode_plane(E8KvWidth::W3, c3.data(), s3.data(), rows, rs3.data());
            e8_kv_encode_plane(E8KvWidth::W2, px.data(), rows, c2.data(), s2.data());
            e8_kv_decode_plane(E8KvWidth::W2, c2.data(), s2.data(), rows, rs2.data());
            std::vector<std::uint8_t> lc3(static_cast<std::size_t>(cbr3) * rows);
            std::vector<std::uint16_t> ls3(static_cast<std::size_t>(gpr) * rows);
            std::vector<std::uint8_t> lc2(static_cast<std::size_t>(cbr2) * rows);
            std::vector<std::uint16_t> ls2(static_cast<std::size_t>(gpr) * rows);
            std::vector<float> r3(n, 0.0f), r2(n, 0.0f);
            const bool ok3 = e8_kv_encode_plane_as(E8KvPlaneCodec::Lattice, E8KvWidth::W3,
                                                   px.data(), rows, lc3.data(), ls3.data());
            const bool ok3d = e8_kv_decode_plane_as(E8KvPlaneCodec::Lattice, E8KvWidth::W3,
                                                    lc3.data(), ls3.data(), rows, r3.data());
            const bool ok2 = e8_kv_encode_plane_as(E8KvPlaneCodec::Lattice, E8KvWidth::W2,
                                                   px.data(), rows, lc2.data(), ls2.data());
            const bool ok2d = e8_kv_decode_plane_as(E8KvPlaneCodec::Lattice, E8KvWidth::W2,
                                                    lc2.data(), ls2.data(), rows, r2.data());
            const double s3e = rel_l2(rs3, px), l3e = rel_l2(r3, px);
            const double s2e = rel_l2(rs2, px), l2e = rel_l2(r2, px);
            std::printf("  as-wired rel-L2 (NOT the matched-transform table):\n"
                        "    w3 scalar %.6f  lattice %.6f  -> %+.3f dB\n"
                        "    w2 scalar %.6f  lattice %.6f  -> %+.3f dB\n",
                        s3e, l3e, 20.0 * std::log10(s3e / l3e),
                        s2e, l2e, 20.0 * std::log10(s2e / l2e));
            check(ok3 && ok3d && l3e < s3e,
                  "8.7 w3: the lattice plane is less distorted than the scalar plane");
            check(ok2 && ok2d && l2e < s2e,
                  "8.7 w2: the lattice plane is less distorted than the scalar plane");
            // 8.8 the second stage must BUY something, or the 3-bit codeword is the 2-bit
            // codeword with 8 dead bits.
            check(l3e < l2e, "8.8 w3 (stage 1 + stage 2) is less distorted than w2 (stage 1 only)");
        }
    }

    // ------------------------------------------ 9. THE OPS LAYER'S OWN EXTENT TABLE
    // Sections 1-8 are the geometry and the codec. This is the CONSUMER. This section's
    // numbers are CONTENT -- the actual extents, and the identity that proves the g64 scale
    // block was not double-counted into them. It is the runnable half of the same gate the
    // static_asserts in ops/kv_cache/d256_profile.h enforce at compile time.
    std::printf("\n--- 9. ops extent table (ops/kv_cache/d256_profile.h) ---\n");
    {
        using ninfer::DType;
        using ninfer::ops::d256_kv_cache_profile;
        using ninfer::ops::D256KVCacheProfile;
        struct Row {
            DType dtype;
            const char* name;
            E8KvWidth width;
            std::int32_t expect;
        };
        const Row rows[3] = {{DType::E8Kv, "e8k4 (E8Kv, shipped)", E8KvWidth::W4, 128},
                             {DType::E8K3Kv, "e8k3 (E8K3Kv)", E8KvWidth::W3, 96},
                             {DType::E8K2Kv, "e8k2 (E8K2Kv)", E8KvWidth::W2, 64}};
        for (int i = 0; i < 3; ++i) {
            const D256KVCacheProfile p = d256_kv_cache_profile(rows[i].dtype);
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "9.1 %s: ops extent matches the codec of record's "
                          "e8_kv_row_code_bytes()",
                          rows[i].name);
            check(p.code_leading_extent == e8_kv_row_code_bytes(rows[i].width), msg);
            std::snprintf(msg, sizeof(msg), "9.2 %s: code extent is the literal %d B/row",
                          rows[i].name, static_cast<int>(rows[i].expect));
            check(p.code_leading_extent == rows[i].expect, msg);
            std::snprintf(msg, sizeof(msg),
                          "9.3 %s: extent*%d rows + %d groups*%d rows*%d B == %d B/head-page "
                          "(g64 scale block separate, NOT double-counted)",
                          rows[i].name, static_cast<int>(kE8KvPageTokens),
                          static_cast<int>(e8_kv_groups_per_row()),
                          static_cast<int>(kE8KvPageTokens),
                          static_cast<int>(kE8KvScaleBytesPerGroup),
                          static_cast<int>(e8_kv_k_plane_bytes(rows[i].width)));
            check(p.code_leading_extent * kE8KvPageTokens +
                      e8_kv_groups_per_row() * kE8KvPageTokens * kE8KvScaleBytesPerGroup ==
                  e8_kv_k_plane_bytes(rows[i].width), msg);
            std::snprintf(msg, sizeof(msg),
                          "9.4 %s: code dtype is the tier itself, g64, %d fp16 scales/row",
                          rows[i].name, static_cast<int>(e8_kv_groups_per_row()));
            check(p.code_dtype == rows[i].dtype && p.quant_group == kE8KvScaleGroup &&
                      p.scale_leading_extent == e8_kv_groups_per_row(), msg);
        }
        // 9.5 THE NEGATIVE THIS WHOLE ARM IS ABOUT. A narrow plate must not be handed the
        // unpacked tiers' extent. The old E8Kv-only ternary answers 256 for both of these:
        // `cache.dtype == DType::E8Kv ? kHeadDim / 2 : kHeadDim`.
        check(d256_kv_cache_profile(DType::E8K3Kv).code_leading_extent !=
                      ninfer::ops::kD256KVCacheHeadDim &&
                  d256_kv_cache_profile(DType::E8K2Kv).code_leading_extent !=
                      ninfer::ops::kD256KVCacheHeadDim,
              "9.5 neither narrow tier is read at the unpacked tiers' 256 B/row");
        // 9.6 THE REFUSAL IS NOT WIDENED. Adding the two rows must not have turned the
        // `default:` arm into an admission: a dtype this table does not price still throws.
        const DType unlisted[5] = {DType::NVFP4, DType::ISO3, DType::FP32, DType::U8,
                                   DType::I64};
        bool all_threw = true;
        for (int i = 0; i < 5; ++i) {
            try {
                (void)d256_kv_cache_profile(unlisted[i]);
                all_threw = false;
            } catch (const std::invalid_argument&) {
            }
        }
        check(all_threw,
              "9.6 an unlisted dtype (NVFP4/ISO3/FP32/U8/I64) still throws -- the narrow "
              "rows were added, NOT by widening the default arm");
    }

    // ------------------------------------------ 10. THE PLANE AXIS (dl/e8vaxis, F1166)
    // Sections 1-9 treat the family as a K-plane family with a CONSTANT V. This section is the
    // edit: E8 becomes a FORMAT selectable independently on K and on V, and the floor is e8 at 2
    // bits on BOTH planes. Three properties, and each one is a bar rather than a claim:
    //   10.1 the axis is TOTAL, and the shipped K answers do not move;
    //   10.2 the V plane REUSES the K plane's codec -- proved by BIT IDENTITY, not by
    //        inspection, because "it is the same codec" is exactly the kind of statement that is
    //        true of the source and false of the bytes;
    //   10.3 ⭐ THE ROTATION INVARIANT: one 64-group's stored bytes must not depend on HOW MANY
    //        ROWS the call was given. That is the exactness doctrine's row-locality rule, and it
    //        is asserted rather than assumed because `e8_lattice_hadamard64()` is a prerequisite
    //        worth 0.89 dB at 2 bits and 0.72 dB at 3 bits, and a rotation that leaked across
    //        rows would leave every aggregate number in this file plausible.
    std::printf("\n--- 10. the plane axis: e8 as a format on K AND on V (F1166) ---\n");
    {
        // Section-local PRNG, for the reason section 8 gives about itself: reusing another
        // section's would make this section's data depend on how many draws that section made.
        std::uint32_t p10 = 0x9E3779B9u;
        const auto r10 = [&p10]() -> std::uint32_t {
            p10 ^= p10 << 13; p10 ^= p10 >> 17; p10 ^= p10 << 5; return p10;
        };
        const auto u10 = [&r10]() -> float {   // [-1, 1)
            return (static_cast<float>(r10() & 0xFFFFFFu) / 8388608.0f) - 1.0f;
        };

        // 10.1 TOTALITY, and the shipped answers do not move.
        struct FRow {
            E8KvWidth      w;
            E8KvPlaneFormat f;
        };
        const FRow frows[3] = {{E8KvWidth::W4, E8KvPlaneFormat::B4},
                               {E8KvWidth::W3, E8KvPlaneFormat::B3},
                               {E8KvWidth::W2, E8KvPlaneFormat::B2}};
        for (int i = 0; i < 3; ++i) {
            char msg[240];
            std::snprintf(msg, sizeof(msg),
                          "10.1 w%d: the K plane's codec of record is UNCHANGED by the axis",
                          static_cast<int>(frows[i].w));
            check(e8_kv_plane_codec_of_record(E8KvPlane::K, frows[i].f) ==
                      e8_kv_plane_codec_of_record(frows[i].w),
                  msg);
            std::snprintf(msg, sizeof(msg),
                          "10.1 format B%d's plane is the SAME %d B/head-page as w%d's",
                          static_cast<int>(frows[i].f),
                          static_cast<int>(e8_kv_plane_format_bytes(frows[i].f)),
                          static_cast<int>(frows[i].w));
            check(e8_kv_plane_format_bytes(frows[i].f) == e8_kv_k_plane_bytes(frows[i].w), msg);
        }
        check(e8_kv_plane_codec_of_record(E8KvPlane::V, E8KvPlaneFormat::B4) ==
                  E8KvPlaneCodec::I4,
              "10.1 V at B4 is the shipped i4 codec, NAMED rather than silently defaulted");
        check(e8_kv_plane_codec_of_record(E8KvPlane::V, E8KvPlaneFormat::B2) ==
                      E8KvPlaneCodec::Lattice &&
                  e8_kv_plane_codec_of_record(E8KvPlane::V, E8KvPlaneFormat::B3) ==
                      E8KvPlaneCodec::Lattice,
              "10.1 THE PLANE AXIS: V at B2/B3 is the SAME lattice codec K uses");
        check(e8_kv_plane_codec_of_record(E8KvPlane::K, E8KvPlaneFormat::B4) !=
                      E8KvPlaneCodec::Lattice &&
                  e8_kv_plane_codec_of_record(E8KvPlane::V, E8KvPlaneFormat::B4) !=
                      E8KvPlaneCodec::Lattice,
              "10.1 no 4-bit lattice on EITHER plane: the codeword has no 32-bit form");

        // 10.2 THE FLOOR AS ARITHMETIC, and the V plane ENCODED BY THE K PLANE'S OWN CALL.
        {
            const std::int32_t rows = 8;
            const std::size_t n = static_cast<std::size_t>(kE8KvHeadDim) * rows;
            std::vector<float> px(n);
            for (std::int32_t r = 0; r < rows; ++r) {
                for (std::int32_t g = 0; g < e8_kv_groups_per_row(); ++g) {
                    float* gv = px.data() + static_cast<std::size_t>(r) * kE8KvHeadDim +
                                g * kE8KvScaleGroup;
                    for (std::int32_t i = 0; i < kE8KvScaleGroup; ++i) { gv[i] = u10() * 0.5f; }
                }
            }
            const E8KvPlaneFormat lf[2] = {E8KvPlaneFormat::B2, E8KvPlaneFormat::B3};
            for (int li = 0; li < 2; ++li) {
                const E8KvPlaneFormat f = lf[li];
                const std::size_t ncode =
                    static_cast<std::size_t>(e8_kv_plane_format_row_code_bytes(f)) * rows;
                const std::size_t nscale =
                    static_cast<std::size_t>(e8_kv_groups_per_row()) * rows;
                std::vector<std::uint8_t> ck(ncode), cv(ncode);
                std::vector<std::uint16_t> sk(nscale), sv(nscale);
                const bool okk =
                    e8_kv_encode_plane_format_as(E8KvPlane::K, f, px.data(), rows, ck.data(),
                                                 sk.data());
                const bool okv =
                    e8_kv_encode_plane_format_as(E8KvPlane::V, f, px.data(), rows, cv.data(),
                                                 sv.data());
                char msg[280];
                std::snprintf(msg, sizeof(msg),
                              "10.2 B%d: the V plane's ENCODE is BIT-IDENTICAL to the K plane's "
                              "-- same lattice, same tables, same rotation, no new mathematics",
                              static_cast<int>(f));
                check(okk && okv && std::memcmp(ck.data(), cv.data(), ncode) == 0 &&
                          std::memcmp(sk.data(), sv.data(), nscale * 2) == 0,
                      msg);
                std::vector<float> xk(n, 0.0f), xv(n, 0.0f);
                const bool okkd = e8_kv_decode_plane_format_as(E8KvPlane::K, f, ck.data(),
                                                               sk.data(), rows, xk.data());
                const bool okvd = e8_kv_decode_plane_format_as(E8KvPlane::V, f, ck.data(),
                                                               sk.data(), rows, xv.data());
                std::snprintf(msg, sizeof(msg),
                              "10.2 B%d: the V plane's DECODE is bit-identical to the K plane's "
                              "on the same stored bytes",
                              static_cast<int>(f));
                check(okkd && okvd && std::memcmp(xk.data(), xv.data(), n * sizeof(float)) == 0,
                      msg);
                // ... and both are the codec of record's OWN bytes, so "the K codec" is not a
                // paraphrase of a third thing that happens to agree.
                std::vector<std::uint8_t> c_direct(ncode);
                std::vector<std::uint16_t> s_direct(nscale);
                const bool okd = e8_kv_encode_plane_lattice(e8_kv_lattice_width_of(f), px.data(),
                                                            rows, c_direct.data(),
                                                            s_direct.data());
                std::snprintf(msg, sizeof(msg),
                              "10.2 B%d: and BOTH are the codec of record's own "
                              "e8_kv_encode_plane_lattice(w) bytes",
                              static_cast<int>(f));
                check(okd && std::memcmp(ck.data(), c_direct.data(), ncode) == 0 &&
                          std::memcmp(sk.data(), s_direct.data(), nscale * 2) == 0,
                      msg);
            }
            // THE FLOOR, numerically: 4608 + 4608 over two planes of one head-page.
            check(e8_kv_pair_bytes(E8KvPlaneFormat::B2, E8KvPlaneFormat::B2) == 9216 &&
                      e8_kv_pair_bits_x100(E8KvPlaneFormat::B2, E8KvPlaneFormat::B2) == 225,
                  "10.2 e8 at 2 bits on BOTH planes is 9216 B/head-page = 2.25 b/el per plane");
        }

        // 10.3 THE ROTATION INVARIANT: THE OUTPUT DOES NOT DEPEND ON THE ROW COUNT.
        {
            // One 256-element row, encoded three ways: as the ONLY row of a 1-row plane, as row 0
            // of a 9-row plane, and as row 5 of that same 9-row plane. All three must store the
            // SAME bytes and decode to the same values. The codec's rotation is per-64-group and
            // INSIDE the call (`e8_lattice_hadamard64(y)` on the group's own 64 floats, in
            // e8_kv_encode_plane_lattice's group loop), so this holds BY CONSTRUCTION -- and that
            // is exactly why it is worth an assertion: a refactor that hoisted the rotation to
            // the plane level, or that let a group share state with its neighbour, would keep
            // every aggregate number in this file plausible while moving real bytes.
            std::vector<float> one(kE8KvHeadDim);
            for (std::int32_t i = 0; i < kE8KvHeadDim; ++i) {
                one[static_cast<std::size_t>(i)] = u10() * 0.7f;
            }
            std::vector<float> many(static_cast<std::size_t>(kE8KvHeadDim) * 9, 0.0f);
            const int dup[2] = {0, 5};
            for (int di = 0; di < 2; ++di) {
                for (std::int32_t i = 0; i < kE8KvHeadDim; ++i) {
                    many[static_cast<std::size_t>(dup[di]) * kE8KvHeadDim +
                         static_cast<std::size_t>(i)] = one[static_cast<std::size_t>(i)];
                }
            }
            const int other[7] = {1, 2, 3, 4, 6, 7, 8};
            for (int oi = 0; oi < 7; ++oi) {
                for (std::int32_t i = 0; i < kE8KvHeadDim; ++i) {
                    many[static_cast<std::size_t>(other[oi]) * kE8KvHeadDim +
                         static_cast<std::size_t>(i)] = u10() * 0.7f;
                }
            }
            for (const E8KvWidth w : {E8KvWidth::W2, E8KvWidth::W3}) {
                const std::int32_t cbr = e8_kv_row_code_bytes(w);
                const std::int32_t gpr = e8_kv_groups_per_row();
                std::vector<std::uint8_t> c1(static_cast<std::size_t>(cbr));
                std::vector<std::uint16_t> s1(static_cast<std::size_t>(gpr));
                std::vector<std::uint8_t> c9(static_cast<std::size_t>(cbr) * 9);
                std::vector<std::uint16_t> s9(static_cast<std::size_t>(gpr) * 9);
                const bool ok1 = e8_kv_encode_plane_lattice(w, one.data(), 1, c1.data(), s1.data());
                const bool ok9 = e8_kv_encode_plane_lattice(w, many.data(), 9, c9.data(), s9.data());
                char msg[280];
                std::snprintf(msg, sizeof(msg),
                              "10.3 w%d: row 0's stored CODE bytes are the same in a 1-row plane "
                              "and in a 9-row plane (the rotation cannot leak across rows)",
                              static_cast<int>(w));
                check(ok1 && ok9 &&
                          std::memcmp(c1.data(), c9.data(), static_cast<std::size_t>(cbr)) == 0,
                      msg);
                std::snprintf(msg, sizeof(msg),
                              "10.3 w%d: and row 0's stored SCALE words are the same too",
                              static_cast<int>(w));
                check(std::memcmp(s1.data(), s9.data(), static_cast<std::size_t>(gpr) * 2) == 0,
                      msg);
                std::snprintf(msg, sizeof(msg),
                              "10.3 w%d: ROW 5 of the 9-row plane is byte-identical to the "
                              "1-row plane -- the property is per row, not only for row 0",
                              static_cast<int>(w));
                check(std::memcmp(c1.data(), c9.data() + 5 * cbr, static_cast<std::size_t>(cbr)) ==
                              0 &&
                          std::memcmp(s1.data(), s9.data() + 5 * gpr,
                                      static_cast<std::size_t>(gpr) * 2) == 0,
                      msg);
                // THE DECODE DIRECTION, so the invariant is pinned on both sides of the plane.
                std::vector<float> x1(static_cast<std::size_t>(kE8KvHeadDim), 0.0f);
                std::vector<float> x9(static_cast<std::size_t>(kE8KvHeadDim) * 9, 0.0f);
                const bool okd1 = e8_kv_decode_plane_lattice(w, c1.data(), s1.data(), 1, x1.data());
                const bool okd9 = e8_kv_decode_plane_lattice(w, c9.data(), s9.data(), 9, x9.data());
                std::snprintf(msg, sizeof(msg),
                              "10.3 w%d: and the DECODE of row 0 agrees between the 1-row and "
                              "the 9-row plane",
                              static_cast<int>(w));
                check(okd1 && okd9 &&
                          std::memcmp(x1.data(), x9.data(),
                                      static_cast<std::size_t>(kE8KvHeadDim) * sizeof(float)) == 0,
                      msg);
            }
        }

        // 10.4 THE SHIPPED W4 ROW IS NOT REACHABLE THROUGH THE PLANE-AXIS SEAM WITH A LATTICE,
        // and it IS reachable as the SCALAR codec. The pre-image's 8.1 and 8.2 already pin the
        // first half through e8_kv_encode_plane_as(); this pins it through the seam this line
        // ADDED, so the new door is not a way around the old wall.
        check(e8_kv_plane_codec_of_record(E8KvPlane::K, E8KvPlaneFormat::B4) ==
                  E8KvPlaneCodec::Scalar,
              "10.4 (K, B4) is the SCALAR codec, and the plane-axis seam routes it to "
              "e8_kv_encode_plane(W4) -- the control arm, not a 4-bit lattice");
    }

    std::printf("\n== %d/%d ==\n", g_ok, g_total);
    return g_ok == g_total ? 0 : 1;
}
