#pragma once
// e8_lattice_plane_codec.cuh -- the REAL E8 lattice, as a KV K-PLANE codec at the
// tree's own 3-bit and 2-bit K-plane widths.
//
// WHY THIS FILE EXISTS
// --------------------
// src/ops/kernel/e8_lattice.cuh:125-137 says, in its own words, that the Conway-Sloane
// projection's output "CANNOT BE STORED in the KV code plane the reader uses", because
// that plane holds one signed integer per coordinate while the projection returns either
// D8 (integer) or D8+1/2 (all-eight-coordinates half-integer) points, and 47.6% of real
// K blocks take the half-integer coset. The rintf that must follow therefore costs
// +3.65 dB (2.32x MSE). It then names two escapes: a plane holding 2*coordinate
// (+1 bit/el), OR "where the consumer reconstructs the lattice point rather than an
// integer code".
//
// src/ops/kernel/e8_lattice_codec.cuh built the SECOND escape: an 8-dimensional
// CODEWORD that is a lattice point (8-bit table index, 7 sign bits, 1 shift bit, and at
// 3 bits a second 8-bit stage), reconstructed at decode time. No coordinate is ever
// stored as an integer, so the half-integer coset is representable and the -3.65 dB is
// not paid. That file calls e8_project_8d_fast() for real (its :259).
//
// BUT IT HAD NO PRODUCTION CALLER. It was reachable only from
// tools/e8_verify/e8_lattice_codec_test.cpp:45, because the two files that would host it
// on the hot path -- src/ops/kernel/gqa_attention_{prefill,decode}_i8.cuh -- belonged to
// another line. Its own test records that, and dl/e8lattice/REPORT.md:410-413 names the
// landing point: "(b) route the K writer through this codec's decoder, which needs no
// change to e8_lattice.cuh at all and is what 3 prices."
//
// THIS FILE IS THAT LANDING, in the layer that owns the K-plane geometry rather than in
// gqa_attention*. It is the plane-level codec the product layer
// (src/product/kv_e8_width_codec.h) names as the codec of record for the 3-bit and
// 2-bit e8 K planes.
//
// THE RATES ARE THE TREE'S OWN, AND THEY ARE PINNED, NOT ASSERTED
// --------------------------------------------------------------
//   stage 1 only        16 b / 8 dims = 2.00 b/el -> 2.25 b/el on the K plane = W2
//   stage 1 + stage 2   24 b / 8 dims = 3.00 b/el -> 3.25 b/el on the K plane = W3
// and the product layer's own byte counts are e8_kv_code_bytes_per_8() = 2 at W2 and
// 3 at W3. Those are the same numbers arrived at from opposite directions, and the
// static_asserts at the bottom of this file are the check that they cannot drift.
//
// THE ROTATION IS KEPT, AND IT IS A PREREQUISITE
// ----------------------------------------------
// e8_lattice_hadamard64() is applied to each 64-element group BEFORE the 8-dimensional
// lattice, exactly as src/ops/kernel/gqa_attention_kv_quant.cuh's E8 branch does and as
// the codec's own header requires. Measured on real K, removing it costs the lattice
// 0.89 dB at 2 bits and 0.72 dB at 3 bits (dl/e8lattice/REPORT.md 4b), so it is not an
// optional extra. H64 is orthonormal and symmetric, hence its own inverse, so the same
// call both rotates and un-rotates.
//
// THE TWO ENCODERS, AND WHY THE DEFAULT IS THE EXPENSIVE ONE
// ----------------------------------------------------------
//   Exact     -- e8_lattice_encode_exact(): the 256-entry x 2-shift table scan, proven
//                against brute force over all 65536 codewords (0/300 strictly worse,
//                worst excess 0.0000000000). THE DEFAULT, because it is the
//                configuration dl/e8lattice/REPORT.md section 3c measured.
//   Projected -- e8_lattice_encode_projected(): e8_project_8d_fast() seeds the candidate
//                and the scan is skipped. CHEAPER, and it is the only mode in which the
//                Conway-Sloane projection is on the encode path -- but it agrees with the
//                exact encoder on only 24.4% of blocks and is 0.772 dB worse in MSE
//                (measured, dl/e8lattice/REPORT.md section 5), and the projection does NOT
//                prune the table scan. Offered, priced, NOT the default: making it the
//                default would spend 0.77 dB of the 2.09 dB the lattice wins at 2 bits.
//
// WHAT IS NOT HERE
// ----------------
//   * W4. The codeword has a 16-bit form and a 24-bit form; there is no 32-bit form, so
//     the real lattice has no 4-bit stage. e8_kv_lattice_supports(W4) returns false and
//     the dispatch REFUSES rather than falling back silently.
//   * The stage-2 table is the one tools/e8_verify/e8_lattice_codec_test.cpp builds
//     (240 E8 roots of norm^2 = 2 plus 16 points of norm^2 = 4 = 256 entries = 8 bits).
//     It is this tree's construction; dl/e8lattice/REPORT.md item 6 records that the
//     reference's 1-bit codebook could not be reconciled with an 8-bit index.
//   * No kernel. This is the host-side codec of record; a device reader must supply the
//     tables (they are 2 KiB total, shared by every layer) rather than build them.
//
// HOST AND DEVICE -- WHERE THE LINE IS, AND WHY
//   CUDA-qualified like its neighbours, and host-callable the same way
//   src/ops/common/fp4_codec.cuh is: when not compiling under nvcc the CUDA keywords
//   degrade here, and <cuda_runtime.h> is satisfied by the shim at
//   tools/e8_verify/host_shim/ (the include path tools/e8_verify/run_lattice_codec.sh and
//   the ninfer_e8_width_codec_test target both carry).
//
//   THE LINE: under nvcc this header provides the THREE DEVICE-CALLABLE PRIMITIVES
//   (e8_lattice_encode3_8d, e8_lattice_pack_word, e8_lattice_unpack_word) and nothing
//   else -- the table builders, the static tables, the plane encode/decode and the
//   dispatch are HOST-SIDE and are compiled out. A device reader supplies its own tables.
//   The single definition of the arithmetic is preserved; what is not duplicated is the
//   table, which is where duplication would be able to drift. The reason is written at
//   the guard: this was measured, not assumed.

#include <cstdint>

#if !defined(__CUDACC__) && !defined(__CUDA_ARCH__)
// A host-only compiler has no __device__ / __forceinline__, and no __shfl_xor_sync
// intrinsic. Same degradation as src/ops/common/fp4_codec.cuh:56-63, and like that file
// the definitions are NEUTRAL and left in place: this header's own functions are
// declared __device__ __forceinline__ below, so undefining them here would break the
// very file that defined them.
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
// e8_lattice.cuh's warp-cooperative variants are DEFINED (not called) on this path, so
// the intrinsic has to exist as a form the host compiler accepts.
#ifndef __shfl_xor_sync
#define __shfl_xor_sync(mask, x, off) (0)
#endif
#endif

#include "ops/kernel/e8_lattice_codec.cuh"   // brings in ops/kernel/e8_lattice.cuh
#include "product/kv_e8_width.h"
#include "product/kv_e8_width_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace ninfer::product {

// ---------------------------------------------------------------- encoder choice
enum class E8LatticeEncoder : std::uint8_t {
    Exact = 0,       // the 256-entry table scan; the measured default
    Projected,       // e8_project_8d_fast() seeds it; cheaper, 0.772 dB worse
};

// The stage-2 step, in stage-1 step units. Decode multiplies it by the group scale and
// by 0.5 (the stored coordinate is doubled), which is exactly the midpoint refinement
// that halves the gaps the stage-1 grid leaves at 1.0 and 2.0 -- see the arithmetic in
// e8_lattice_encode3_8d below. dl/e8lattice's harness fits 0.500 on real K.
inline constexpr double kE8LatticeStage2Step = 0.5;

// ---------------------------------------------------------------- table builders
// Host-only: they use std::vector. A device reader supplies prebuilt tables instead.
#if !defined(__CUDA_ARCH__)

// 227 order-preserving absolute-value tuples of D8^ (the sorted-then-deduped form is 7,
// not 227 -- dedup MUST be on the order-preserving tuple) plus the 29 lexicographic
// padding arrangements of five 3/2 and three 1/2. Both counts are asserted by the test.
inline ninfer::ops::E8LatticeStage1 e8_lattice_build_stage1() {
    ninfer::ops::E8LatticeStage1 t{};
    static const int cvals[8] = {-7, -5, -3, -1, 1, 3, 5, 7};
    std::vector<std::array<std::int8_t, 8>> filt;
    std::int8_t c[8];
    // sc = sum of coords, sc2 = sum of squares; |coord|^2 <= 10 prunes at 40 in the
    // doubled-magnitude units this recursion carries.
    for (int a = 0; a < 8; ++a) {
        c[0] = static_cast<std::int8_t>(cvals[a]);
        for (int b = 0; b < 8; ++b) {
            c[1] = static_cast<std::int8_t>(cvals[b]);
            for (int d = 0; d < 8; ++d) {
                c[2] = static_cast<std::int8_t>(cvals[d]);
                for (int e = 0; e < 8; ++e) {
                    c[3] = static_cast<std::int8_t>(cvals[e]);
                    for (int f = 0; f < 8; ++f) {
                        c[4] = static_cast<std::int8_t>(cvals[f]);
                        for (int g = 0; g < 8; ++g) {
                            c[5] = static_cast<std::int8_t>(cvals[g]);
                            for (int h = 0; h < 8; ++h) {
                                c[6] = static_cast<std::int8_t>(cvals[h]);
                                for (int k = 0; k < 8; ++k) {
                                    c[7] = static_cast<std::int8_t>(cvals[k]);
                                    int sc = 0, sc2 = 0;
                                    for (int j = 0; j < 8; ++j) {
                                        sc += c[j];
                                        sc2 += c[j] * c[j];
                                    }
                                    if (sc2 > 40 || (sc % 4) != 0) { continue; }
                                    std::array<std::int8_t, 8> v;
                                    for (int j = 0; j < 8; ++j) { v[j] = c[j]; }
                                    filt.push_back(v);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    t.n_signed = static_cast<int>(filt.size());

    std::vector<std::array<std::uint8_t, 8>> absv;
    absv.reserve(filt.size());
    for (const auto& v : filt) {
        std::array<std::uint8_t, 8> a{};
        for (int j = 0; j < 8; ++j) {
            a[j] = static_cast<std::uint8_t>(v[j] < 0 ? -v[j] : v[j]);
        }
        absv.push_back(a);
    }
    // order-preserving sort-then-unique (NOT abs-then-sort: sorting first destroys the
    // permutations and yields 7)
    std::sort(absv.begin(), absv.end());
    absv.erase(std::unique(absv.begin(), absv.end()), absv.end());
    t.n_abs = static_cast<int>(absv.size());

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
    for (int i = 0; i < t.n_abs; ++i) {
        for (int j = 0; j < 8; ++j) { t.mag[i * 8 + j] = absv[i][j]; }
    }
    for (int i = 0; i < t.n_pad; ++i) {
        for (int j = 0; j < 8; ++j) { t.mag[(t.n_abs + i) * 8 + j] = pad[i][j]; }
    }
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

inline ninfer::ops::E8LatticeStage2 e8_lattice_build_stage2() {
    ninfer::ops::E8LatticeStage2 t{};
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
    for (int i = 0; i < t.n; ++i) {
        for (int j = 0; j < 8; ++j) { t.coord[i * 8 + j] = v[i][j]; }
    }
    return t;
}

// The process-wide tables, built once. 2 KiB total, shared by every layer, which is the
// L1-residency argument the codec's header makes (AQLM's codebook is ~1 MiB per layer).
inline const ninfer::ops::E8LatticeStage1& e8_lattice_stage1_table() {
    static const ninfer::ops::E8LatticeStage1 t = e8_lattice_build_stage1();
    return t;
}
inline const ninfer::ops::E8LatticeStage2& e8_lattice_stage2_table() {
    static const ninfer::ops::E8LatticeStage2 t = e8_lattice_build_stage2();
    return t;
}

#endif   // !__CUDA_ARCH__

}   // namespace ninfer::product

// ===========================================================================
// THE DEVICE-CALLABLE CORE (compiled under nvcc too). These three are the pieces a
// device reader instantiates: it supplies its own tables (2 KiB, shared by every
// layer) and calls them from its own 64-group loop. They are in ninfer::ops, beside
// src/ops/kernel/e8_lattice_codec.cuh whose missing stage-2 encoder
// e8_lattice_encode3_8d is, and they carry no host-only dependency.
// ===========================================================================
namespace ninfer::ops {

// MOVED (dl/e8dev STEP 0): the W3 stage-2 encoder,
// e8_lattice_pack_word() and e8_lattice_unpack_word() now live in
// ops/kernel/e8_lattice_codec.cuh, which this header already includes at
// the top. They are NOT redefined here -- one definition, one home.

}   // namespace ninfer::ops

// ===========================================================================
// THE HOST-SIDE PLANE CODEC OF RECORD, AND WHY IT IS BOUNDED HERE
//
// This block is compiled only when it CAN be: under nvcc it is absent. That is not
// convenience. The plane codec's two dependencies are host-only by construction --
// the table BUILDERS use std::vector, and the static tables they build are
// function-local statics, which a __device__ function cannot have -- so nvcc's
// device pass would parse these functions and find the tables undefined. Two ways to
// make it compile were available: give the device pass its own copy of the builders
// (drift: two tables that can disagree, exactly the defect e8_lattice_codec.cuh:66-69
// says it includes rather than forward-declares in order to avoid), or state the
// boundary. This states the boundary: A DEVICE READER USES THE THREE PRIMITIVES
// ABOVE AND SUPPLIES ITS OWN TABLES. Nothing is missing on the device side; the
// 64-group loop and the fp16 scale plane belong to the reader.
//
// This was not a guess. The first build of this header under nvcc (a -cubin probe on
// sm_120) failed with exactly this: "identifier e8_lattice_stage1_table is
// undefined" at the two plane entry points, from the device pass.
// ===========================================================================
#if !defined(__CUDACC__)

namespace ninfer::product {

// ---------------------------------------------------------------- the plane codec
// One FP16 scale per 64-element group, exactly the geometry kv_e8_width.h prices
// (kE8KvScaleGroup = 64, kE8KvScaleBytesPerGroup = 2 -> 0.25 b/el of side information),
// and the closed-form scale rule is the codec's own natural one, amax / kE8LatticeMaxCoord
// -- the analogue of the scalar row's amax / 2^(w-1). Both conventions are one
// derivation from one input, which is what makes the two arms comparable at matched bits.
//
// THE ENCODER AND THE DECODER USE THE SAME SCALE. The scalar row quantizes with the exact
// float scale and stores fp16(scale), so a reader reconstructs with a slightly different
// step than the writer used; the lattice row quantizes with fp16(scale) itself, which is
// what makes the round-trip invariant exact rather than nearly exact. The scale plane is
// one fp16 either way, so the side-information rate is identical.
inline constexpr bool e8_kv_lattice_supports(E8KvWidth w) noexcept {
    return w == E8KvWidth::W2 || w == E8KvWidth::W3;
}

// Encode `rows` rows of kE8KvHeadDim elements into the plane geometry of width w.
// Returns false (and writes nothing) for a width the real lattice cannot carry -- W4 --
// rather than falling back to the scalar codec silently.
inline bool e8_kv_encode_plane_lattice(E8KvWidth w, const float* x, std::int32_t rows,
                                       std::uint8_t* code_out, std::uint16_t* scale_out,
                                       E8LatticeEncoder enc = E8LatticeEncoder::Exact) {
    if (!e8_kv_lattice_supports(w)) { return false; }
    const ninfer::ops::E8LatticeStage1& t1 = e8_lattice_stage1_table();
    const ninfer::ops::E8LatticeStage2& t2 = e8_lattice_stage2_table();
    const std::int32_t gpr = e8_kv_groups_per_row();
    const std::int32_t cbr = e8_kv_row_code_bytes(w);
    const std::int32_t cpg = e8_kv_code_bytes_per_8(w);
    const bool three_bit = (w == E8KvWidth::W3);

    for (std::int32_t r = 0; r < rows; ++r) {
        const float* row = x + static_cast<std::size_t>(r) * kE8KvHeadDim;
        std::uint8_t* codes = code_out + static_cast<std::size_t>(r) * cbr;
        std::uint16_t* scales = scale_out + static_cast<std::size_t>(r) * gpr;
        for (std::int32_t g = 0; g < gpr; ++g) {
            const float* grp = row + g * kE8KvScaleGroup;
            // THE ROTATION FIRST, on the 64-group, before the 8-dimensional lattice.
            // (floatfix) THE REAL TYPE IS `float`, AND THAT IS THE CODEC'S ARITHMETIC
            // RATHER THAN A NARROWING.  Measured (dl/floatfix, 512 rows = 16 384 blocks):
            // the two instantiations of ONE search are handed the same float values in all
            // 5 blocks that ever disagree, and each instantiation called with the OTHER
            // side's block returns the OTHER side's bytes -- so the search is one function
            // and only the TARGET differs, `fl32(y/s)` here against `fl64(y/s)` there.  The
            // two codewords it then chooses between tie EXACTLY in cost (relative gap
            // <= 5.2e-16, twice exactly 0.0), and the exhaustive codebook argmin under this
            // target is this path's own choice, so nothing is given up: the plane's bytes
            // change on 5 blocks in 16 384 and every one of those bytes costs the same.
            // The alternative -- the arm on these double overloads -- is measured at
            // 16 722 `.f64` lines in the arm's PTX against 0 on the float ones, and the
            // target hardware may have no fp64 at all (e8_lattice_codec.cuh:100-110).
            float y[64];
            for (std::int32_t i = 0; i < kE8KvScaleGroup; ++i) {
                y[i] = grp[i];
            }
            ninfer::ops::e8_lattice_hadamard64(y);

            // AMAXDOMAIN: the group scale is taken on the ROTATED group, not on the raw
            // one. The rotation is orthonormal, so it is the same quantity in a different
            // frame -- but the codec quantises the rotated coordinates, and H64 concentrates
            // them (E[amax_rot^2]/E[amax^2] = 0.68, gqa_attention_decode_i8.cuh:291-292), so
            // the raw amax OVERSTATES the span the lattice actually has to cover and spends
            // codebook on coordinates that are not there. This is the SAME rule the shipped
            // scalar E8 branch already uses: its amax is taken AFTER gqa_kv_hadamard64
            // (gqa_attention_decode_i8.cuh:297-299, gqa_attention_prefill_i8.cuh:165-167),
            // and _TODO.md 116/116b records why -- "a pre-rotation scale clipped the top of
            // the range". The lattice codec was the one place still taking it pre-rotation.
            //
            // MEASURED by dl/e8dev (evidence/amax_domain.cpp, ONE line differing between the
            // two arms, decoder = this file's own e8_kv_decode_plane_lattice, never
            // reimplemented, /home/user/bench/kvdump_e8src 1509 rows = 386 304 elements):
            //   W2  34.6103 % -> 30.2949 % relRMS   MSE 2.871969e-01 -> 2.200436e-01   (+1.157 dB)
            //   W3  19.6730 % -> 17.5358 % relRMS   MSE 9.279222e-02 -> 7.372604e-02   (+1.001 dB)
            // ⚠️ The effect was relayed as +0.4 dB; on this instrument it is +1.0 to +1.2 dB.
            // The absolute relRMS levels are corpus-slice dependent (this slice is not the
            // full 27 dumps), so the DELTA is the robust quantity, and the delta agrees in
            // sign and in "both widths" with the relayed figure.
            float amax = 0.0f;
            for (std::int32_t i = 0; i < kE8KvScaleGroup; ++i) {
                const float a = std::fabs(static_cast<float>(y[i]));
                if (a > amax) { amax = a; }
            }
            scales[g] = e8_kv_fp16_bits(amax /
                                        static_cast<float>(ninfer::ops::kE8LatticeMaxCoord));
            const float s = e8_kv_fp16_from_bits(scales[g]);
            if (!(s > 0.0f)) {
                // A zero (or denormal-flushed) group scale: an all-zero group, and the
                // codeword for it is the one the decoder maps back to zero. (Before this
                // change an all-zero group was the only way to reach here; now a group whose
                // ROTATED amax still flushes is too, which is the same set.)
                for (std::int32_t b = 0; b < (kE8KvScaleGroup / 8) * cpg; ++b) {
                    codes[g * (kE8KvScaleGroup / 8) * cpg + b] = 0;
                }
                continue;
            }

            for (std::int32_t sb = 0; sb < kE8KvScaleGroup / 8; ++sb) {
                // (floatfix) ONE input, in the type the primitives are instantiated at.
                // Before this the block existed twice -- `blk_f` for the projection-seeded
                // mode and `blk` (double) for the exact one -- which is how the codec's
                // exact mode came to aim at `fl64(y/s)` while its own device arm aims at
                // `fl32(y/s)`.
                float blk[8];
                for (std::int32_t i = 0; i < 8; ++i) {
                    blk[i] = y[sb * 8 + i] / s;
                }
                std::uint8_t* slot = codes + (g * (kE8KvScaleGroup / 8) + sb) * cpg;
                if (three_bit) {
                    ninfer::ops::E8LatticeCode3 c3;
                    if (enc == E8LatticeEncoder::Projected) {
                        // stage 1 from the projection, stage 2 still exact on the residual
                        ninfer::ops::E8LatticeCode2 p;
                        (void)ninfer::ops::e8_lattice_encode_projected(t1, blk, &p);
                        double r1[8];
                        ninfer::ops::e8_lattice_decode_8d(t1, p, 1.0, r1);
                        double res[8];
                        for (int i = 0; i < 8; ++i) { res[i] = blk[i] - r1[i]; }
                        int best = 0; double bd = 1e300;
                        for (int e = 0; e < t2.n; ++e) {
                            double d = 0.0;
                            for (int i = 0; i < 8; ++i) {
                                const double q = res[i] -
                                    kE8LatticeStage2Step * 0.5 *
                                    static_cast<double>(t2.coord[e * 8 + i]);
                                d += q * q;
                            }
                            if (d < bd) { bd = d; best = e; }
                        }
                        c3.s1 = p;
                        c3.root = static_cast<std::uint8_t>(best);
                    } else {
                        (void)ninfer::ops::e8_lattice_encode3_8d(
                            t1, t2, blk, static_cast<float>(kE8LatticeStage2Step), &c3);
                    }
                    ninfer::ops::e8_lattice_pack_word(c3.s1, slot);
                    slot[2] = c3.root;
                } else {
                    ninfer::ops::E8LatticeCode2 c2;
                    if (enc == E8LatticeEncoder::Projected) {
                        (void)ninfer::ops::e8_lattice_encode_projected(t1, blk, &c2);
                    } else {
                        (void)ninfer::ops::e8_lattice_encode_exact(t1, blk, &c2);
                    }
                    ninfer::ops::e8_lattice_pack_word(c2, slot);
                }
            }
        }
    }
    return true;
}

// Decode the plane back to kE8KvHeadDim-wide natural-domain rows.
inline bool e8_kv_decode_plane_lattice(E8KvWidth w, const std::uint8_t* code_in,
                                       const std::uint16_t* scale_in, std::int32_t rows,
                                       float* x) {
    if (!e8_kv_lattice_supports(w)) { return false; }
    const ninfer::ops::E8LatticeStage1& t1 = e8_lattice_stage1_table();
    const ninfer::ops::E8LatticeStage2& t2 = e8_lattice_stage2_table();
    const std::int32_t gpr = e8_kv_groups_per_row();
    const std::int32_t cbr = e8_kv_row_code_bytes(w);
    const std::int32_t cpg = e8_kv_code_bytes_per_8(w);
    const bool three_bit = (w == E8KvWidth::W3);

    for (std::int32_t r = 0; r < rows; ++r) {
        float* row = x + static_cast<std::size_t>(r) * kE8KvHeadDim;
        const std::uint8_t* codes = code_in + static_cast<std::size_t>(r) * cbr;
        const std::uint16_t* scales = scale_in + static_cast<std::size_t>(r) * gpr;
        for (std::int32_t g = 0; g < gpr; ++g) {
            float* grp = row + g * kE8KvScaleGroup;
            const double s = static_cast<double>(e8_kv_fp16_from_bits(scales[g]));
            double y[64];
            for (std::int32_t sb = 0; sb < kE8KvScaleGroup / 8; ++sb) {
                const std::uint8_t* slot = codes + (g * (kE8KvScaleGroup / 8) + sb) * cpg;
                if (three_bit) {
                    ninfer::ops::E8LatticeCode3 c3;
                    c3.s1 = ninfer::ops::e8_lattice_unpack_word(slot);
                    c3.root = slot[2];
                    ninfer::ops::e8_lattice_decode3_8d(t1, t2, c3, s,
                                                       kE8LatticeStage2Step * s, &y[sb * 8]);
                } else {
                    const ninfer::ops::E8LatticeCode2 c2 =
                        ninfer::ops::e8_lattice_unpack_word(slot);
                    ninfer::ops::e8_lattice_decode_8d(t1, c2, s, &y[sb * 8]);
                }
            }
            // H64 is orthonormal and symmetric, so this is the inverse of the encode-side
            // rotation by the same function.
            ninfer::ops::e8_lattice_hadamard64(y);
            for (std::int32_t i = 0; i < kE8KvScaleGroup; ++i) {
                grp[i] = static_cast<float>(y[i]);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------- the dispatch
// ONE function decides which codec a width's K plane carries. This is the redirect: the
// scalar codec in product/kv_e8_width_codec.h is NOT deleted and is NOT changed -- it
// stays the codec of record for W4 (the shipped row, whose stored bytes the 55-check
// suite pins at every width) and stays available as the priced control arm. What moves
// is the DEFAULT for the two narrow widths, which now carry the real lattice.
enum class E8KvPlaneCodec : std::uint8_t {
    Scalar = 0,   // the one-integer-per-coordinate codec; the shipped W4 row and the control
    Lattice,      // the reconstructing E8 lattice point codec; W3 and W2 only
};

[[nodiscard]] constexpr E8KvPlaneCodec e8_kv_plane_codec_of_record(E8KvWidth w) noexcept {
    return e8_kv_lattice_supports(w) ? E8KvPlaneCodec::Lattice : E8KvPlaneCodec::Scalar;
}

// The seam. `codec` selects explicitly; there is no silent fallback in either direction:
// asking for Lattice at a width that cannot carry it returns false and writes nothing.
[[nodiscard]] inline bool e8_kv_encode_plane_as(E8KvPlaneCodec codec, E8KvWidth w,
                                                const float* x, std::int32_t rows,
                                                std::uint8_t* code_out,
                                                std::uint16_t* scale_out,
                                                E8LatticeEncoder enc =
                                                    E8LatticeEncoder::Exact) {
    if (codec == E8KvPlaneCodec::Lattice) {
        return e8_kv_encode_plane_lattice(w, x, rows, code_out, scale_out, enc);
    }
    e8_kv_encode_plane(w, x, rows, code_out, scale_out);
    return true;
}

[[nodiscard]] inline bool e8_kv_decode_plane_as(E8KvPlaneCodec codec, E8KvWidth w,
                                                const std::uint8_t* code_in,
                                                const std::uint16_t* scale_in,
                                                std::int32_t rows, float* x) {
    if (codec == E8KvPlaneCodec::Lattice) {
        return e8_kv_decode_plane_lattice(w, code_in, scale_in, rows, x);
    }
    e8_kv_decode_plane(w, code_in, scale_in, rows, x);
    return true;
}

}   // namespace ninfer::product

#endif   // !__CUDACC__

namespace ninfer::product {

// The two sides of the rate arithmetic, pinned against each other. The left operand is
// the codec's own geometry (8 dims at 2 or 3 bits), the right operand is the product
// layer's byte count. If either moves alone, this file stops compiling.
static_assert(ninfer::ops::kE8LatticeStage1Bits == 8 * 2,
              "stage 1 is 16 bits / 8 dims = 2 b/el -- the W2 rate");
static_assert(ninfer::ops::kE8LatticeStage1Bits + ninfer::ops::kE8LatticeStage2Bits == 8 * 3,
              "16 + 8 = 24 bits / 8 dims = 3 b/el -- the W3 rate");
static_assert(e8_kv_code_bytes_per_8(E8KvWidth::W2) == 2,
              "W2's plane is 2 bytes / 8 elements: the 16-bit lattice codeword, exactly");
static_assert(e8_kv_code_bytes_per_8(E8KvWidth::W3) == 3,
              "W3's plane is 3 bytes / 8 elements: the 24-bit lattice codeword, exactly");
static_assert(e8_kv_code_bytes_per_8(E8KvWidth::W4) == 4,
              "W4's plane is 4 bytes / 8 elements and the codeword has no 32-bit form");
static_assert(kE8KvScaleGroup == 64, "the rotation and the scale group are the same 64");
static_assert(ninfer::ops::kE8LatticeGroup == 64, "the codec's own group == the product's");

}   // namespace ninfer::product
