#pragma once

// E8 LATTICE CODEC -- the reconstructing consumer that src/ops/kernel/e8_lattice.cuh
// names as the only place its projections may be used.
//
// WHY THIS FILE EXISTS
// --------------------
// e8_lattice.cuh:125-137 says, in its own words, that e8_project_8d_fast()'s output
// "CANNOT BE STORED in the KV code plane the reader uses", because that plane holds one
// signed integer per coordinate while the projection returns either D8 (integer) or
// D8 + 1/2 (all-eight-coordinates half-integer) points, and 47.6% of real K blocks take
// the half-integer coset. It then names the two escapes: a plane holding 2*coordinate
// (5 bits at +-15 = +1 bit/el), or "where the consumer reconstructs the lattice point
// rather than an integer code".
//
// This header is the second escape, built. The codeword is NOT an integer code: it is
//   p = s (*) (a/2) +- 1/4,  a in {1,3,5}^8 doubled magnitudes,
// an 8-dimensional POINT of E8 +- 1/4, reconstructed at decode time from an 8-bit index
// into a 256-entry absolute-value table, 7 sign bits and one shift bit. No coordinate is
// ever stored as an integer, so the projection's half-integer coset is representable and
// the +3.65 dB (2.32x MSE) penalty e8_lattice.cuh:129-134 measured is not paid. That
// penalty is a property of the old INTEGER-CODE consumer, not of the projection.
//
// At 2 bits/element the integer-code consumer does not exist at all: a 2-bit code plane
// can address at most 4 levels per coordinate. The only way to spend 2 bits on K with E8
// in it is a reconstructed lattice point, which is what this is.
//
// THE HADAMARD ROTATION IS PRESERVED, AND IT IS A PREREQUISITE, NOT A TARGET
// -------------------------------------------------------------------------
// Every entry point below takes the 64-group in the tree's own rotated layout and
// e8_lattice_hadamard64() is applied BEFORE the lattice, exactly as
// src/ops/kernel/gqa_attention_kv_quant.cuh's E8 branch does. The rotation is what makes
// the 8-dimensional block Gaussian-like, which is what a lattice quantiser's shape gain
// is defined against. Removing it does not make the lattice cheaper; it removes the
// assumption the lattice is packed for.
//
// RATES (stated as arithmetic, because the scale word is part of the rate)
//   one FP16 scale per 64-group  = 16/64 = 0.25 b/el, shared by both planes below
//   stage 1 only, 16 b / 8 dims  = 2.00 b/el  ->  2.25 b/el on the K plane
//   stage 1 + stage 2, 24 b / 8  = 3.00 b/el  ->  3.25 b/el on the K plane
//   the shipped 4-bit plane      = 4.00 b/el  ->  4.25 b/el on the K plane
//
// CODEBOOK SIZE
//   stage 1: 256 entries x 8 nibbles = 1024 B (1.00 KiB), one table shared by all layers
//   stage 2: 256 entries x 8 nibbles = 1024 B (1.00 KiB)
// so the 2-bit codec costs 1 KiB of L1-resident table and the 3-bit codec 2 KiB. For
// scale: AQLM's codebook is ~1 MiB per layer, which is why the published comparison has
// AQLM slower than FP16; 1 KiB is the whole reason the L1-residency argument holds.
//
// DECODE COST -- analytic, counted below, NOT measured on hardware (no GPU taken)
//   per block, shared: 1 POPC + 1 AND + 1 CMP + 1 predicated XOR for the flip parity,
//                      1 index scale, 2 scale-derived constants (s/2, s/4)
//   per weight:        1 nibble/magnitude LUT load, 1 sign-bit XOR, 1 FMA, plus the
//                      packed nibble extract amortised over the pair
//   => ~4 arithmetic instructions per weight plus one shared L1 table load.
//
// HOST AND DEVICE
//   The header is CUDA-qualified like its neighbour. It is host-callable by defining the
//   three CUDA keywords away before including it -- the same shim
//   tools/e8_verify/e8_lattice_codec_test.cpp uses, and the same trick the E8LATTICE
//   line's harness used to reach e8_project_8d_fast() from a CPU-only process.

#include <cstdint>

// The dependency is real and it is the point of the file: the encoder below calls
// e8_project_8d_fast(). Included rather than forward-declared so the CUDA build links
// against the tree's own definition in src/ops/kernel/e8_lattice.cuh and cannot drift
// from it. A host TU supplies cuda_runtime.h through a shim and defines the three CUDA
// keywords away before including this header.
#include "ops/kernel/e8_lattice.cuh"

namespace ninfer::ops {

// ---------------------------------------------------------------- geometry
inline constexpr int kE8LatticeDim          = 8;
inline constexpr int kE8LatticeStage1Entries = 256;   // 8-bit index
inline constexpr int kE8LatticeStage1Bits    = 16;    // 8 index + 7 sign + 1 shift
inline constexpr int kE8LatticeStage2Entries = 256;   // 8-bit index = 1 bit/element
inline constexpr int kE8LatticeStage2Bits    = 8;
inline constexpr int kE8LatticeGroup         = 64;    // the scale group, = the tree's

// The absolute-value grid, doubled so every magnitude is an integer:
//   doubled magnitude in {1,3,5}; coordinate magnitude = doubled/2 +- 1/4
inline constexpr int kE8LatticeMaxDoubled = 5;

// The codec's own maximum |coordinate|, read off the table at build time. The closed-form
// group scale is amax / this, i.e. the analogue of the tree's amax/2^(w-1) rule.
inline constexpr double kE8LatticeMaxCoord = 2.75;   // 5/2 + 1/4

// ---------------------------------------------------------------- the rotation
// The tree's 64-point Sylvester butterfly, host form of gqa_kv_hadamard64
// (src/ops/kernel/gqa_attention_kv_quant.cuh:22-35) and byte-for-byte the function at
// tools/e8_verify/e8_lattice_math.cpp:321-338. Orthonormal and symmetric, hence its own
// inverse, so the same call both rotates and un-rotates. Amplitude 0.125 = 1/8 = 1/sqrt(64).
// ===========================================================================
// THE PRECISION REPAIR (e8kernel). The arithmetic is written ONCE, templated on the real
// type R, and every literal is cast to R so nothing silently promotes to double.
//
// WHY IT IS NEEDED: measured at SASS, 14-17 % of every lattice arm's instructions were
// fp64-class (DADD / DMUL / DFMA / F2F.F64) on a consumer RTX 5090 D, against 0 % on every
// scalar arm -- and on the NPU / domestic-card targets this engine must reach, fp64 may not
// exist at all. Nothing here needs it: every quantity is a multiple of 1/4 with
// |coordinate| <= 2.75, and every cost is a sum of squares of such values (<= 8 * 2.75^2),
// all EXACTLY representable in fp32. The repair is therefore DECISION-IDENTICAL on the real
// corpus, not merely close -- which is what makes it safe to land under a green test.
//
// WHY THIS SHAPE: one spelling of the math. `e8_lattice_hadamard64_t<R>` is the only copy;
// the `double` overload below is what every existing caller already gets, unchanged; the
// `float` overload is what a device reader calls to get zero fp64. Two hand-written copies
// would be free to drift, which is the exact defect class this file warns about elsewhere.
// ===========================================================================
template <typename R>
__device__ __forceinline__ void e8_lattice_hadamard64_t(R v[64]) {
    R x0[32], x1[32], t0[32], t1[32];
    for (int i = 0; i < 32; ++i) { x0[i] = v[i]; x1[i] = v[i + 32]; }
    for (int off = 1; off < 32; off <<= 1) {
        for (int i = 0; i < 32; ++i) {
            const R y0 = x0[i ^ off], y1 = x1[i ^ off];
            const bool hi = (i & off) != 0;
            t0[i] = hi ? y0 - x0[i] : x0[i] + y0;
            t1[i] = hi ? y1 - x1[i] : x1[i] + y1;
        }
        for (int i = 0; i < 32; ++i) { x0[i] = t0[i]; x1[i] = t1[i]; }
    }
    for (int i = 0; i < 32; ++i) {
        const R a = x0[i], b = x1[i];
        v[i] = (a + b) * static_cast<R>(0.125);
        v[i + 32] = (a - b) * static_cast<R>(0.125);
    }
}
__device__ __forceinline__ void e8_lattice_hadamard64(double v[64]) {
    e8_lattice_hadamard64_t<double>(v);
}
__device__ __forceinline__ void e8_lattice_hadamard64(float v[64]) {
    e8_lattice_hadamard64_t<float>(v);
}

// ---------------------------------------------------------------- table
// 256 entries. 227 are the published D8^ absolute-value tuples, 29 are the published
// sqrt(12) padding elements (five 3/2 and three 1/2); both counts are asserted by the
// host test, and so is the membership of all 65536 reachable codewords in E8 +- 1/4.
struct E8LatticeStage1 {
    std::uint8_t mag[256 * 8];    // doubled magnitudes, odd values in {1,3,5}
    std::uint8_t cls[256 * 8];    // (mag-1)/2, the 0..2 magnitude class
    std::uint8_t suma[256];       // sum(mag); its parity fixes the required flip parity
    int n = 0, n_abs = 0, n_pad = 0, n_signed = 0;
};

// 240 E8 roots (norm^2 = 2) + 16 points of norm^2 = 4 = 256 entries = 8 bits = 1 bit/el.
// Stored doubled in {-2,-1,0,1,2}.
struct E8LatticeStage2 {
    std::int8_t coord[256 * 8];
    int n = 0, n_roots = 0, n_norm4 = 0;
};

// ---------------------------------------------------------------- popcount
// __popc is a CUDA intrinsic. The host shim that makes this header CPU-callable does not
// define it, so the parity test goes through one helper that resolves to the device
// intrinsic on a device pass and to the compiler builtin on a host pass. Both give the
// same 32-bit population count, so the codec's arithmetic is identical either way.
__device__ __forceinline__ int e8_lattice_popc(unsigned v) {
#ifdef __CUDA_ARCH__
    return __popc(v);
#else
    return __builtin_popcount(v);
#endif
}

// The 7 stored sign bits select a flip mask whose PARITY IS FORCED BY THE ENTRY. That is
// not a cost: the parity is a function of the entry (sum(mag) mod 4) and the 8-bit index
// already carries it, so the same 7 bits address 2^7 reachable patterns either way and
// 256 * 2^7 = 2^15 holds. Derivation: flipping k coordinates changes sum(mag) by -2k mod 4
// and every doubled magnitude is odd, so the class {sum % 4 == 0} is preserved exactly by
// EVEN k and {== 2} exactly by ODD k.
__device__ __forceinline__ int e8_lattice_flip_parity(const E8LatticeStage1& t, int entry) {
    return ((t.suma[entry] >> 1) & 1);
}

// ---------------------------------------------------------------- decode
// Decode is the point of this file: it RECONSTRUCTS the lattice point. `scale` is the
// group scale; `s2` scales stage 2 and is 0 for the 2-bit codec.
struct E8LatticeCode2 { std::uint8_t idx, signs, shift; };
struct E8LatticeCode3 { E8LatticeCode2 s1; std::uint8_t root; };

template <typename R>
__device__ __forceinline__ void e8_lattice_decode_8d_t(const E8LatticeStage1& t,
                                                       E8LatticeCode2 c, R scale, R out[8]) {
    const int need = e8_lattice_flip_parity(t, c.idx);
    unsigned m = c.signs;
    if ((e8_lattice_popc(m) & 1) != need) { m ^= 0x80u; }          // 1 POPC + AND + CMP + XOR, shared
    const R d = c.shift ? static_cast<R>(0.25) : static_cast<R>(-0.25);
    const R half = static_cast<R>(0.5) * scale;
    const R off = d * scale;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const R mag = half * static_cast<R>(t.mag[c.idx * 8 + i]);             // 1 LUT load
        out[i] = (((m >> i) & 1u) != 0u) ? (off - mag) : (off + mag);          // 1 sign + 1 FMA
    }
}
__device__ __forceinline__ void e8_lattice_decode_8d(const E8LatticeStage1& t,
                                                     E8LatticeCode2 c, double scale,
                                                     double out[8]) {
    e8_lattice_decode_8d_t<double>(t, c, scale, out);
}
__device__ __forceinline__ void e8_lattice_decode_8d(const E8LatticeStage1& t,
                                                     E8LatticeCode2 c, float scale,
                                                     float out[8]) {
    e8_lattice_decode_8d_t<float>(t, c, scale, out);
}

template <typename R>
__device__ __forceinline__ void e8_lattice_decode3_8d_t(const E8LatticeStage1& t,
                                                        const E8LatticeStage2& t2,
                                                        E8LatticeCode3 c, R scale, R s2, R out[8]) {
    e8_lattice_decode_8d_t<R>(t, c.s1, scale, out);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] += s2 * static_cast<R>(0.5) * static_cast<R>(t2.coord[c.root * 8 + i]);
    }
}
__device__ __forceinline__ void e8_lattice_decode3_8d(const E8LatticeStage1& t,
                                                      const E8LatticeStage2& t2,
                                                      E8LatticeCode3 c, double scale, double s2,
                                                      double out[8]) {
    e8_lattice_decode3_8d_t<double>(t, t2, c, scale, s2, out);
}
__device__ __forceinline__ void e8_lattice_decode3_8d(const E8LatticeStage1& t,
                                                      const E8LatticeStage2& t2,
                                                      E8LatticeCode3 c, float scale, float s2,
                                                      float out[8]) {
    e8_lattice_decode3_8d_t<float>(t, t2, c, scale, s2, out);
}

// ---------------------------------------------------------------- encode (exact)
// Exact nearest codeword over the 65536-entry codebook. For a fixed entry and shift the
// best sign pattern is s_i = sign(u_i); if its flip parity is wrong the cheapest repair is
// the single coordinate minimising 4*|u_i|*A_i, because flipping sign i changes
// (|u| - A)^2 into (|u| + A)^2, a cost of exactly 4|u|A. Any odd-k repair costs at least as
// much, so one flip suffices. The whole search is therefore 256 entries x 2 shifts x 8
// adds, and tools/e8_verify/e8_lattice_codec_test.cpp checks it against brute force over
// all 65536 codewords.
template <typename R>
__device__ __forceinline__ R e8_lattice_encode_exact_t(const E8LatticeStage1& t, const R x[8],
                                                       E8LatticeCode2* out) {
    R best = static_cast<R>(1e300); E8LatticeCode2 bc{0, 0, 0};
    for (int sh = 0; sh < 2; ++sh) {
        const R d4 = sh ? static_cast<R>(0.25) : static_cast<R>(-0.25);
        R au[8], d[8][3];
        unsigned neg = 0;
        for (int i = 0; i < 8; ++i) {
            const R u = x[i] - d4;
            au[i] = u < static_cast<R>(0.0) ? -u : u;
            for (int m = 0; m < 3; ++m) {
                const R A = static_cast<R>(0.5) * static_cast<R>(2 * m + 1);
                const R q = au[i] - A;
                d[i][m] = q * q;
            }
            if (u < static_cast<R>(0.0)) { neg |= (1u << i); }
        }
        for (int e = 0; e < t.n; ++e) {
            const std::uint8_t* cc = &t.cls[e * 8];
            R cost = d[0][cc[0]] + d[1][cc[1]] + d[2][cc[2]] + d[3][cc[3]]
                   + d[4][cc[4]] + d[5][cc[5]] + d[6][cc[6]] + d[7][cc[7]];
            unsigned m = neg;
            if ((e8_lattice_popc(m) & 1) != e8_lattice_flip_parity(t, e)) {
                int arg = 0; R worst = static_cast<R>(1e300);
                for (int i = 0; i < 8; ++i) {
                    const R c = static_cast<R>(4.0) * au[i] *
                                (static_cast<R>(0.5) * static_cast<R>(2 * cc[i] + 1));
                    if (c < worst) { worst = c; arg = i; }
                }
                m ^= (1u << arg);
                cost += worst;
            }
            if (cost < best) {
                best = cost;
                bc.idx = static_cast<std::uint8_t>(e);
                bc.signs = static_cast<std::uint8_t>(m & 0x7fu);
                bc.shift = static_cast<std::uint8_t>(sh);
            }
        }
    }
    if (out != nullptr) { *out = bc; }
    return best;
}
// The historical entry point, byte-for-byte the same decisions as the pre-image.
__device__ __forceinline__ double e8_lattice_encode_exact(const E8LatticeStage1& t,
                                                          const double x[8], E8LatticeCode2* out) {
    return e8_lattice_encode_exact_t<double>(t, x, out);
}
// The device entry point: identical decisions, zero fp64-class instructions (measured).
__device__ __forceinline__ float e8_lattice_encode_exact(const E8LatticeStage1& t,
                                                         const float x[8], E8LatticeCode2* out) {
    return e8_lattice_encode_exact_t<float>(t, x, out);
}

// ---------------------------------------------------------------- encode (projection-seeded)
// THE CALL SITE FOR e8_project_8d_fast().
//
// The projection is the exact Conway-Sloane nearest point of E8, so
//   (nearest E8 point of (x - 1/4)) + 1/4
// is the exact nearest point of the coset E8 + 1/4, which is the lattice the codebook is
// drawn from. That point is not itself a codeword -- the codebook is a 2^16 subset -- so a
// code-from-point step is still required, and it is the part the projection's interface
// does NOT supply. This function supplies it by indexing the 256-entry absolute-value grid
// from the projected point's magnitudes, then correcting the sign parity.
//
// It is a FAST CANDIDATE, not an exact encoder: the nearest codeword is not in general the
// codeword nearest the nearest lattice point. The host test measures both the agreement
// rate with e8_lattice_encode_exact() and the MSE penalty, so the approximation is priced
// rather than asserted. The exact encoder remains available and is what the substitution
// numbers in dl/e8lattice/REPORT.md were produced with.
__device__ __forceinline__ double e8_lattice_encode_projected(const E8LatticeStage1& t,
                                                              const float xf[8],
                                                              E8LatticeCode2* out) {
    float p[8];
    e8_project_8d_fast(xf, p);          // the previously dead primitive, called for real
    double best = 1e300; E8LatticeCode2 bc{0, 0, 0};
    for (int sh = 0; sh < 2; ++sh) {
        const double d4 = sh ? 0.25 : -0.25;
        std::uint8_t snap[8];
        for (int i = 0; i < 8; ++i) {
            // the projected point is on E8; the codeword lives on E8 +- 1/4, so index the
            // magnitude grid by |p - d4| doubled, forced odd (the grid holds only odd values)
            double target = 2.0 * (static_cast<double>(p[i]) - d4);
            if (target < 0.0) { target = -target; }
            int a = static_cast<int>(target + 0.5);
            if ((a & 1) == 0) { a += (target > static_cast<double>(a)) ? 1 : -1; }
            if (a < 1) { a = 1; }
            if (a > kE8LatticeMaxDoubled) { a = kE8LatticeMaxDoubled; }
            snap[i] = static_cast<std::uint8_t>(a);
        }
        int be = 0; double bd = 1e300;
        for (int e = 0; e < t.n; ++e) {
            double dd = 0.0;
            for (int i = 0; i < 8; ++i) {
                const double dm = static_cast<double>(t.mag[e * 8 + i]) - static_cast<double>(snap[i]);
                dd += dm * dm;
            }
            if (dd < bd) { bd = dd; be = e; }
        }
        unsigned m = 0;
        for (int i = 0; i < 8; ++i) { if (static_cast<double>(p[i]) - d4 < 0.0) { m |= (1u << i); } }
        if ((e8_lattice_popc(m) & 1) != e8_lattice_flip_parity(t, be)) { m ^= 0x80u; }
        E8LatticeCode2 cand{static_cast<std::uint8_t>(be), static_cast<std::uint8_t>(m & 0x7fu),
                            static_cast<std::uint8_t>(sh)};
        double rec[8];
        e8_lattice_decode_8d(t, cand, 1.0, rec);
        double cost = 0.0;
        for (int i = 0; i < 8; ++i) { const double dd = static_cast<double>(xf[i]) - rec[i]; cost += dd * dd; }
        if (cost < best) { best = cost; bc = cand; }
    }
    if (out != nullptr) { *out = bc; }
    return best;
}

// ---- MOVED HERE (dl/e8dev STEP 0) -----------------------------------------------
// These three were defined in src/ops/kv/e8_lattice_plane_codec.cuh:288-351, which a
// device TU cannot include without dragging two product/ headers in. They are the
// stage-2 encoder and the codeword packer of THIS codec, and the codec is the only
// thing a kernel-side reader is allowed to depend on, so they belong beside it.
// The text is byte-identical; only its home changed.

// ---------------------------------------------------------------- W3 stage-2 encoder
// THE PIECE THAT WAS MISSING. src/ops/kernel/e8_lattice_codec.cuh has the stage-2
// DECODER (e8_lattice_decode3_8d, its :177) and the stage-1 encoder, but no stage-2
// encoder: tools/e8_verify/e8_lattice_codec_test.cpp:352-365 had to build
// E8LatticeCode3::root with an inline brute-force scan of its own. This is that scan,
// extracted with the SAME arithmetic (rho * 0.5 on the doubled coordinate, residual
// measured in code units) so the numbers cannot differ between the test and the codec.
//
// The residual is taken in code units, i.e. x is expected already divided by the group
// scale -- the same domain e8_lattice_encode_exact() works in.
// The real type is a template parameter for the SAME reason as in e8_lattice_codec.cuh:
// one spelling, two instantiations, and the float one is the device path with no fp64.
// NOTE ON THE RETURN VALUE (unchanged from the pre-image, and it is NOT the codeword error):
// `return bd + d1` is stage-1 cost plus stage-2 cost, so it is NOT the distance from x to the
// reconstructed point. Measured on the zero block: the reconstruction is EXACTLY zero (so the
// 3-bit codebook does represent it) while the return is 0.5. Every caller in the tree ignores
// this value -- the plane encoder casts it to void -- so the defect is latent, not load-bearing.
template <typename R>
__device__ __forceinline__ R e8_lattice_encode3_8d_t(
        const ninfer::ops::E8LatticeStage1& t,
        const ninfer::ops::E8LatticeStage2& t2,
        const R x[8], R rho, ninfer::ops::E8LatticeCode3* out) {
    ninfer::ops::E8LatticeCode2 c1;
    const R d1 = ninfer::ops::e8_lattice_encode_exact(t, x, &c1);
    R r1[8];
    ninfer::ops::e8_lattice_decode_8d(t, c1, static_cast<R>(1.0), r1);      // code units
    R res[8];
    for (int i = 0; i < 8; ++i) { res[i] = x[i] - r1[i]; }
    int best = 0; R bd = static_cast<R>(1e300);
    for (int e = 0; e < t2.n; ++e) {
        R d = static_cast<R>(0.0);
        for (int i = 0; i < 8; ++i) {
            const R q = res[i] - rho * static_cast<R>(0.5) *
                        static_cast<R>(t2.coord[e * 8 + i]);
            d += q * q;
        }
        if (d < bd) { bd = d; best = e; }
    }
    if (out != nullptr) {
        out->s1 = c1;
        out->root = static_cast<std::uint8_t>(best);
    }
    return bd + d1;   // stage-1 cost + stage-2 cost, both in code units
}
__device__ __forceinline__ double e8_lattice_encode3_8d(
        const ninfer::ops::E8LatticeStage1& t,
        const ninfer::ops::E8LatticeStage2& t2,
        const double x[8], double rho, ninfer::ops::E8LatticeCode3* out) {
    return e8_lattice_encode3_8d_t<double>(t, t2, x, rho, out);
}
// THE DEVICE ENTRY POINT: same decisions, zero fp64-class instructions (measured at SASS on
// sm_120 and sm_75). A device reader supplies its own tables and calls THIS overload.
__device__ __forceinline__ float e8_lattice_encode3_8d(
        const ninfer::ops::E8LatticeStage1& t,
        const ninfer::ops::E8LatticeStage2& t2,
        const float x[8], float rho, ninfer::ops::E8LatticeCode3* out) {
    return e8_lattice_encode3_8d_t<float>(t, t2, x, rho, out);
}

// ---------------------------------------------------------------- codeword packing
// The lattice plane's byte layout, defined once:
//   W2, 2 bytes / 8 dims:  bits 0..7   = stage-1 table index
//                          bits 8..14  = 7 sign bits
//                          bit  15     = the +-1/4 shift bit
//   W3, 3 bytes / 8 dims:  bits 0..15  = the W2 word above
//                          bits 16..23 = the stage-2 table index
// little-endian, byte b holds bits 8b..8b+7 -- the same byte order the scalar codec
// uses, and the SAME BYTE COUNTS (2 and 3), so the two codecs fill planes of identical
// size and only the TIER NAME distinguishes them. That is a real property, not a
// convenience: it is why e8_kv_plane_codec_of_record() below is a single function.
__device__ __forceinline__ void e8_lattice_pack_word(
        const ninfer::ops::E8LatticeCode2& c, std::uint8_t* out) {
    out[0] = c.idx;
    out[1] = static_cast<std::uint8_t>((c.signs & 0x7fu) | ((c.shift & 1u) << 7));
}
__device__ __forceinline__ ninfer::ops::E8LatticeCode2 e8_lattice_unpack_word(
        const std::uint8_t* in) {
    ninfer::ops::E8LatticeCode2 c;
    c.idx = in[0];
    c.signs = static_cast<std::uint8_t>(in[1] & 0x7fu);
    c.shift = static_cast<std::uint8_t>((in[1] >> 7) & 1u);
    return c;
}


} // namespace ninfer::ops
