#pragma once
// e8_lattice_kv_plane.cuh -- STEP 0: the DEVICE K-PLANE ARM for the real E8 lattice
// at the tree's own 3-bit and 2-bit K-plane widths.
//
// WHY THIS FILE EXISTS
// --------------------
// src/ops/kv/e8_lattice_plane_codec.cuh is the codec of record for W3/W2, and
// src/product/kv_e8_width.h:73-77 already points at it by name. Both are HOST-side: the
// plane API sits inside `#if !defined(__CUDACC__)` (`:74-75`, `:372`...`:575`) and the
// file says so itself -- "No kernel. This is the host-side codec of record; a device
// reader must supply the tables (they are 2 KiB total, shared by every layer) rather
// than build them." The three arithmetic primitives (`e8_lattice_encode3_8d`,
// `e8_lattice_pack_word`, `e8_lattice_unpack_word`) are device-callable and SASS-verified;
// the 64-group loop and the fp16 g64 scale plane "belong to the reader" (`:364-366`).
// THIS FILE IS THE READER.
//
// It is a separate header and NOT a change to the host codec on purpose: the host codec's
// arithmetic, byte layout and dispatch are pinned by a 75/75 suite and a full-corpus
// round-trip invariant (0 / 66 229 248 at both widths). Nothing here re-spells any of it,
// so every quantity below is CALLED from the tree's own primitives or DERIVED from one
// constant, and the derivations are asserted.
//
// ONE DELIBERATE NARROWING: the arm exposes the EXACT encoder only. The host codec also
// offers a projection-seeded mode (`E8LatticeEncoder::Projected`), which is priced at
// 0.772 dB worse in MSE and agrees with the exact encoder on only 24.4% of blocks
// (`e8_lattice_plane_codec.cuh:57-63`) -- it stays host-side, where its measurement lives.
//
// THE ROTATION DOMAIN -- READ THIS BEFORE WIRING THE ARM IN (it is worth 1.202 dB)
// ------------------------------------------------------------------------------
// MEASURED (dl/e8div, full length, g=64, the tree's own codecs, raw relRMS at w3):
//
//   LATTICE fed RAW          19.0370 %  ->  -0.383 dB   (its designed domain)
//   LATTICE fed PRE-ROTATED  21.8620 %  ->  +0.819 dB
//   SCALAR  on PRE-ROTATED   19.8954 %      the SHIPPED reference
//
// `gqa_attention_kv_quant.cuh`'s E8 branch rotates the K row itself
// (`gqa_attention_decode_i8.cuh:297`, `gqa_attention_prefill_i8.cuh:165, :302`) and then
// calls a codec that does NOT rotate. The lattice codec DOES rotate, internally, on each
// 64-group (`e8_lattice_plane_codec.cuh:428-433`): it takes the group in the NATURAL
// domain, applies `e8_lattice_hadamard64()`, divides by the group scale, then encodes.
//
//   ==> THE CODEC'S INPUT MUST BE *RAW* K. Feeding it pre-rotated K rotates twice (H64 is
//       its own inverse), the lattice then quantises the NATURAL coordinates, and the
//       whole benefit of the rotation is lost: +1.202 dB at w3, +0.209 dB at w2.
//
// THE OTHER SIDE OF THE SAME COIN -- AND WHY Q DOES NOT MOVE
// ---------------------------------------------------------
// The codeword is a point of the lattice in the ROTATED domain, because the encoder
// divides `H64(K)` by the scale before searching. The host decoder UNS-rotates
// (`:521-526`) because its interface is raw -> raw. A DEVICE reader serving the attention
// kernel must NOT: the shipped Q path rotates Q (`decode_i8.cuh:390`, `prefill_i8.cuh:447`)
// and the shipped K plane holds the ROTATED K. So:
//
//   * writer: hand the codec RAW K (drop the shipped `gqa_kv_hadamard64` on K -- the codec
//     applies it internally). The K plane stays in the ROTATED domain: exactly the domain
//     it is in today. Nothing about the plane's domain changes.
//   * reader: reconstruct in the codeword's native (ROTATED) domain and STOP. The trailing
//     `e8_lattice_hadamard64` of the host decoder MUST NOT be applied here.
//   * Q: UNCHANGED. Still rotated. `<H64(Q), H64(K)>` is the correct pairing.
//
// THE MIRROR QUESTION, ANSWERED: if the arm instead fed PRE-ROTATED K (i.e. left the
// shipped `gqa_kv_hadamard64` in place) then the codeword's native domain would be the
// NATURAL domain, and the reader would have to apply the trailing H64 -- at which point a
// rotated Q dotted against a natural K is `<H64(Q), K>`, a WRONG dot product, not a
// 1.202 dB loss. The two conventions are a matched pair; a one-sided fix on either side is
// silent corruption. `kE8KvLatticeInputIsNatural` / `kE8KvLatticeOutputIsRotated` below
// name the convention in one place so a reader cannot pick the other one by accident.
//
// THE TABLES, AND WHAT THEY ACTUALLY COST
// ---------------------------------------
// The reader supplies them; they are process-wide constants, built once by the host-side
// builders that already exist (`e8_lattice_plane_codec.cuh:144-275`). Their real size,
// measured off the two structs rather than quoted from the prose (which says "2 KiB"):
//   E8LatticeStage1 : mag[2048] + cls[2048] + suma[256] + 4*int32 = 4368 B
//   E8LatticeStage2 : coord[2048]            + 3*int32            = 2060 B
//                                                         TOTAL   = 6428 B
// `cls` is a pure function of `mag` (`(mag-1)/2`) and is read only by the exact encoder, so
// it could be dropped at four ALU ops per coordinate -- it is NOT dropped here, because the
// arithmetic must stay the codec's. `E8KvLatticeTables` is a plain pair of device pointers,
// which is what makes the supply the caller's choice: a `cudaMalloc` + `cudaMemcpy` from the
// host builders, owned by the launcher and filled once, needs no new TU, no CMake entry and
// no `-rdc=true`. That is the recommended supply, and it is why this header contains no
// `__constant__`, no `extern` and nothing that could need a second definition to drift from.
//
// THE READER'S INTEGER CODEC -- what the tensor-core QK path needs
// --------------------------------------------------------------
// The int8 kernel does its QK product with an s8 MMA and applies the two group scales
// AFTERWARDS (`decode_i8.cuh:684-687`: `score += q_scale * k_scale * mma`). That is exact
// only if the reconstructed K coordinate is an exact multiple of the group scale. For the
// lattice it is:
//
//   coordinate_i / scale = +/- mag_i/2 +/- 1/4                 (W2, mag in {1,3,5})
//                        = +/- mag_i/2 +/- 1/4 + c2_i/4        (W3, c2 in {-2,-1,0,1,2})
//                        = m_i / 4 ,  m_i an INTEGER
//
// so `decode_group_codes` emits `m_i` as an int8 and the caller pairs it with
// `e8_kv_lattice_reader_scale(scale)` = `scale/4`. |m_i| <= 11 at W2 and <= 13 at W3: both
// fit an int8 with room, and `q_i * m_i` summed over 32 dims is `127 * 13 * 32 = 52832`,
// far inside int32. So the MMA path is EXACT for the lattice; nothing about the tensor-core
// code changes except the one factor of 4 and the K plane's row stride.

#include <cmath>
#include <cstdint>

#include "ops/kernel/e8_lattice_codec.cuh"   // brings in ops/kernel/e8_lattice.cuh

namespace ninfer::ops {

// ---------------------------------------------------------------- the fp16 scale word
// THE SCALE WORD IS THE PRODUCT LAYER'S, BIT FOR BIT, AND THAT IS NOT `__float2half_rn`.
// `ninfer::product::e8_kv_fp16_bits()` (src/product/kv_e8_width_codec.h:78-92) is a
// hand-rolled conversion: truncate the fp32 mantissa to 13 bits, add 0x1000, carry, and
// flush every underflow to a signed zero -- ROUND-HALF-UP, no denormals.
// `__float2half_rn` is round-half-to-EVEN, with denormals; the two disagree whenever the
// 13 discarded bits are exactly 0x1000 and across the whole denormal range. Using the
// intrinsic here would make this arm's stored scale plane differ from the host codec's on
// those inputs -- a silent writer/reader split of exactly the kind
// `gqa_attention_kv_quant.cuh`'s AMAXFIX note exists to make impossible. (A first draft of
// this header DID use the intrinsic; the harness's adversarial sweep is what caught it.)
//
// The product helper is a HOST-ONLY function in a HOST-ONLY product header, which a kernel
// TU must not include -- no `.cuh` under `src/ops/` includes a `product/` header today --
// so the algorithm is restated here in INTEGER OPS ONLY and held to the original by
// MEASUREMENT, not by hope: `dl/e8dev/evidence/check_e8_kv_plane.cpp` compares the two
// bit-for-bit over the corpus's own group amax values and over a randomized and an
// adversarial set, and prints the disagreement count.
[[nodiscard]] __device__ __forceinline__ std::uint16_t
e8_kv_lattice_fp16_bits(float v) noexcept {
    std::uint32_t f;
    __builtin_memcpy(&f, &v, sizeof(f));
    const std::uint32_t sign = (f >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((f >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t man = f & 0x7FFFFFu;
    if (exp <= 0) { return static_cast<std::uint16_t>(sign); }             // underflow -> +-0
    if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7C00u); }  // overflow -> inf
    man = (man + 0x1000u) >> 13;                                           // round-half-up
    if (man == 0x400u) {
        man = 0;
        ++exp;
        if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7C00u); }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | man);
}

[[nodiscard]] __device__ __forceinline__ float e8_kv_lattice_fp16_from_bits(
        std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1Fu;
    std::uint32_t man = h & 0x3FFu;
    std::uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign;
        } else {
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
    float out;
    __builtin_memcpy(&out, &f, sizeof(out));
    return out;
}

// ---------------------------------------------------------------- the width axis
// The arm is parameterised by the K-plane code width in BITS (2 or 3). `product/` headers
// are a leaf in this tree -- no `.cuh` under `src/ops/` includes one -- so the byte
// arithmetic is re-derived here from the width alone. The binding to the product layer's
// own `e8_kv_code_bytes_per_8()` is asserted in the host check harness
// (dl/e8dev/evidence/check_e8_kv_plane.cpp), which includes both.
template <int WBits>
inline constexpr int kE8KvLatticeCodeBytesPer8 = (8 * WBits + 7) / 8;   // 2 at W2, 3 at W3

// A VARIABLE template, not a constexpr function: nvcc refuses a call to a constexpr
// __host__ function from inside a __device__ function ("calling a constexpr __host__
// function from a __device__ function is not allowed", measured -- it is what the first
// green -c of this arm was blocked on), and the fix must not be a per-TU
// --expt-relaxed-constexpr flag. A variable template is a constant expression, not a call.
template <int WBits>
inline constexpr bool e8_kv_lattice_width_supported = (WBits == 2 || WBits == 3);

static_assert(kE8KvLatticeCodeBytesPer8<2> == 2, "W2's plate is 2 bytes / 8 dims");
static_assert(kE8KvLatticeCodeBytesPer8<3> == 3, "W3's plate is 3 bytes / 8 dims");
// The 8-dim block and the 64-group are the codec's, not this file's.
static_assert(kE8LatticeDim == 8, "the lattice block is 8-dimensional");
static_assert(kE8LatticeGroup == 64, "the scale group is the codec's own 64");
static_assert(kE8LatticeGroup / kE8LatticeDim == 8, "8 blocks per 64-group");

// The stage-2 step, in stage-1 step units. The codec's own value is
// `ninfer::product::kE8LatticeStage2Step` (`e8_lattice_plane_codec.cuh:135`), which lives
// in a host-only product header this kernel-side file must not include. The value is
// therefore restated here AND the property the reader depends on is asserted, so a change
// to the codec's constant cannot pass silently: the stage-2 term must be exactly ONE code
// unit, i.e. `4 * step * 0.5 == 1` -- that is the `m_i = +/-2*mag +/- 1 + c2_i` identity
// above, and it is what makes `kE8KvLatticeCodeUnitDivisor == 4` the right divisor.
inline constexpr double kE8KvLatticeStage2Step = 0.5;
static_assert(4.0 * kE8KvLatticeStage2Step * 0.5 == 1.0,
              "the stage-2 refinement must be exactly one code unit, or the integer "
              "reader's m_i is not the decoder's output and the QK scale is wrong");

// ---------------------------------------------------------------- the table supply
// A handle, not an owner: where the 6 428 B live is the caller's decision.
struct E8KvLatticeTables {
    const E8LatticeStage1* s1 = nullptr;
    const E8LatticeStage2* s2 = nullptr;

    __device__ __forceinline__ bool valid() const { return s1 != nullptr && s2 != nullptr; }
};

// ---------------------------------------------------------------- the domain contract
inline constexpr bool kE8KvLatticeInputIsNatural = true;    // the writer takes RAW K
inline constexpr bool kE8KvLatticeOutputIsRotated = true;   // the reader emits ROTATED K

// The reader's integer code is in units of scale/4: `coordinate = scale * m / 4`.
inline constexpr int kE8KvLatticeCodeUnitDivisor = 4;
inline constexpr int kE8KvLatticeCodeMaxW2 = 4 * 2 + 3;           // 4*(mag/2 + 1/4), mag = 5
inline constexpr int kE8KvLatticeCodeMaxW3 = kE8KvLatticeCodeMaxW2 + 2;

static_assert(kE8KvLatticeCodeMaxW2 == 11, "W2: |m| <= 4*(5/2) + 1 = 11");
static_assert(kE8KvLatticeCodeMaxW3 == 13, "W3 adds the stage-2 term, |c2| <= 2");

[[nodiscard]] __device__ __forceinline__ constexpr int
e8_kv_lattice_code_max(int wbits) noexcept {
    return wbits == 3 ? kE8KvLatticeCodeMaxW3 : kE8KvLatticeCodeMaxW2;
}
static_assert(e8_kv_lattice_code_max(2) == 11 && e8_kv_lattice_code_max(3) == 13,
              "the code bound must follow the width");
static_assert(e8_kv_lattice_code_max(3) <= 127,
              "the emitted code is an int8 and the s8 MMA multiplies it exactly");

// The scale the CALLER must pair the integer code with, so that `(scale/4) * m` is the
// codec's own reconstruction. One definition, used by the reader and by the check harness,
// so the /4 cannot be applied on one side only.
[[nodiscard]] __device__ __forceinline__ float e8_kv_lattice_reader_scale(float scale) noexcept {
    return scale / static_cast<float>(kE8KvLatticeCodeUnitDivisor);
}

// Round-to-nearest-EVEN, the same tie rule `__float2int_rn` uses, written with `rintf` so
// that this header is callable from a HOST pass as well. That matters: the exactness of
// the integer code (`(scale/4)*m == the decoder's output`) is a claim that has to be
// MEASURED, and the only instrument that can measure it without a card is a host build of
// this same header. Using a CUDA-only intrinsic here would put the one property the MMA
// path depends on behind a GPU that this round does not have. The values are bounded by
// |m| <= 13, far inside int, so the missing saturation cannot matter.
[[nodiscard]] __device__ __forceinline__ int e8_kv_lattice_int_rn(float x) noexcept {
    return static_cast<int>(rintf(x));
}

// ===========================================================================
// THE WRITER -- 64 raw K elements in, one packed codeword group + one fp16 scale out.
//
// Byte-identical to `e8_kv_encode_plane_lattice`'s per-group body
// (`src/ops/kv/e8_lattice_plane_codec.cuh:410-482`): the same raw-group amax, the same
// `amax / kE8LatticeMaxCoord` scale word, the same internal H64, the same
// `e8_lattice_encode3_8d` at W3 / `e8_lattice_encode_exact` at W2, the same
// `e8_lattice_pack_word` plus the trailing root byte at W3. What is NOT duplicated is a
// second spelling of any of those: every one is called.
//
// `kraw` MUST be the NATURAL-domain group (see the domain block above). The caller must
// NOT pre-rotate it.
//
// `scale_bits_out` receives the fp16 scale WORD (the plane's stored bytes). Returns true
// when a non-zero codeword was written; false means the group was all-zero (or its fp16
// scale flushed to zero) and `code_out` holds `blocks * bytes_per_8` zero bytes with scale
// bits 0 -- exactly what the host encoder writes for that group.
// ===========================================================================
template <int WBits>
__device__ __forceinline__ bool e8_kv_lattice_encode_group(
        const E8KvLatticeTables& t, const float kraw[kE8LatticeGroup],
        std::uint8_t* code_out, std::uint16_t* scale_bits_out) {
    static_assert(e8_kv_lattice_width_supported<WBits>,
                  "the E8 lattice K plane exists at 2 and 3 bits only: the codeword has a "
                  "16-bit and a 24-bit form and no 32-bit form, so W4 must refuse rather "
                  "than fall back to a scalar code silently");
    static_assert(kE8KvLatticeInputIsNatural, "the writer's contract, restated");
    constexpr int cpg    = kE8KvLatticeCodeBytesPer8<WBits>;
    constexpr int blocks = kE8LatticeGroup / kE8LatticeDim;

    // 1. THE ROTATION, on the 64-group, before the 8-dimensional lattice. The input is
    //    natural-domain K, so this is the codec's own single rotation, not a second one:
    //    this call is the ONLY place H64 is applied on the K write path. The FLOAT overload
    //    is the device path -- the double one is the historical entry point.
    float y[kE8LatticeGroup];
#pragma unroll
    for (int i = 0; i < kE8LatticeGroup; ++i) { y[i] = kraw[i]; }
    e8_lattice_hadamard64(y);

    // 2. AMAXDOMAIN: the group scale is taken on the ROTATED group, which is what the
    //    codec quantises. H64 concentrates the coordinates (E[amax_rot^2]/E[amax^2] = 0.68,
    //    gqa_attention_decode_i8.cuh:291-292), so the raw amax overstates the span and
    //    spends codebook on coordinates that are not there. This is the SAME rule the
    //    shipped scalar E8 branch already uses (`:297-299`: "The rotation runs BEFORE the
    //    max: a pre-rotation scale clipped the top of the range", _TODO.md 116/116b).
    //    MEASURED, full length, g=64, the tree's own decoder (dl/e8dev evidence/s07):
    //      2-bit  32.4475 % -> 30.4918 % relRMS
    //      3-bit  18.9532 % -> 17.2937 % relRMS
    //    i.e. +0.4 dB at BOTH widths.
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kE8LatticeGroup; ++i) {
        const float a = fabsf(y[i]);
        if (a > amax) { amax = a; }
    }
    if (amax <= 0.0f) {
#pragma unroll
        for (int b = 0; b < blocks * cpg; ++b) { code_out[b] = 0; }
        *scale_bits_out = 0;
        return false;
    }
    // The scale word is the PRODUCT's conversion, so the stored plane byte is the one the
    // host codec would store for the same group (see the fp16 block above), and the value
    // the encoder divides by is the decoder's own read-back -- the codec's "the encoder and
    // the decoder use the same scale" property, which is what makes the round-trip exact.
    *scale_bits_out = e8_kv_lattice_fp16_bits(amax / static_cast<float>(kE8LatticeMaxCoord));
    const float sf = e8_kv_lattice_fp16_from_bits(*scale_bits_out);
    if (!(sf > 0.0f)) {
#pragma unroll
        for (int b = 0; b < blocks * cpg; ++b) { code_out[b] = 0; }
        return false;
    }

    // 3. the 8 blocks, in code units (divided by the group scale).
#pragma unroll
    for (int b = 0; b < blocks; ++b) {
        float blk[kE8LatticeDim];
#pragma unroll
        for (int i = 0; i < kE8LatticeDim; ++i) { blk[i] = y[b * kE8LatticeDim + i] / sf; }
        std::uint8_t* slot = code_out + b * cpg;
        if constexpr (WBits == 3) {
            E8LatticeCode3 c3{};
            (void)e8_lattice_encode3_8d(t.s1[0], t.s2[0], blk,
                                        static_cast<float>(kE8KvLatticeStage2Step), &c3);
            e8_lattice_pack_word(c3.s1, slot);
            slot[2] = c3.root;
        } else {
            E8LatticeCode2 c2{};
            (void)e8_lattice_encode_exact(t.s1[0], blk, &c2);
            e8_lattice_pack_word(c2, slot);
        }
    }
    return true;
}

// ===========================================================================
// THE READER -- the codeword group back to the ROTATED-domain reconstruction.
//
// `out` is what the HOST plane decoder produces BEFORE its trailing H64
// (`e8_lattice_plane_codec.cuh:507-520`) -- which is exactly the domain an attention
// kernel's rotated Q needs. Applying `e8_lattice_hadamard64` to `out` would be the host
// codec's raw->raw interface and the WRONG domain here; see the domain block above.
//
// `scale` is the fp16 group scale as the decoder reads it (fp16 -> float), i.e. the value
// the host codec passes as `s`.
// ===========================================================================
template <int WBits>
__device__ __forceinline__ void e8_kv_lattice_decode_group(
        const E8KvLatticeTables& t, const std::uint8_t* code_in, float scale,
        float out[kE8LatticeGroup]) {
    static_assert(e8_kv_lattice_width_supported<WBits>, "2 and 3 bits only");
    static_assert(kE8KvLatticeOutputIsRotated, "the reader's contract, restated");
    constexpr int cpg    = kE8KvLatticeCodeBytesPer8<WBits>;
    constexpr int blocks = kE8LatticeGroup / kE8LatticeDim;
#pragma unroll
    for (int b = 0; b < blocks; ++b) {
        const std::uint8_t* slot = code_in + b * cpg;
        if constexpr (WBits == 3) {
            E8LatticeCode3 c3{};
            c3.s1   = e8_lattice_unpack_word(slot);
            c3.root = slot[2];
            // the FLOAT overload: the device path, zero fp64-class instructions (measured)
            e8_lattice_decode3_8d(t.s1[0], t.s2[0], c3, scale,
                                  scale * static_cast<float>(kE8KvLatticeStage2Step),
                                  &out[b * kE8LatticeDim]);
        } else {
            const E8LatticeCode2 c2 = e8_lattice_unpack_word(slot);
            e8_lattice_decode_8d(t.s1[0], c2, scale, &out[b * kE8LatticeDim]);
        }
    }
}

// ===========================================================================
// THE READER THE QK PATH NEEDS -- the same reconstruction, emitted as the integer
// `m = 4 * coordinate / scale`, so the existing s8 MMA keeps computing `sum q_i * m_i`
// exactly and the caller pairs it with `e8_kv_lattice_reader_scale(s)`.
//
// This is not an approximation of the reconstruction: it IS the reconstruction, exactly,
// because every lattice coordinate is `scale * m / 4` with `m` an integer. `m` is formed
// from the DECODER's own output, not from the codeword's fields, so a future change to the
// decoder's geometry goes through this function rather than around it. Exactness is not
// asserted here -- asserting it costs a compare per element on the hot path -- it is
// MEASURED by the check harness (`dl/e8dev/evidence/check_e8_kv_plane.cpp`, which requires
// a zero residual over every codeword at both widths and every table entry).
// ===========================================================================
template <int WBits>
__device__ __forceinline__ bool e8_kv_lattice_decode_group_codes(
        const E8KvLatticeTables& t, const std::uint8_t* code_in, float scale,
        std::int8_t out[kE8LatticeGroup]) {
    static_assert(e8_kv_lattice_width_supported<WBits>, "2 and 3 bits only");
    float y[kE8LatticeGroup];
    e8_kv_lattice_decode_group<WBits>(t, code_in, scale, y);
    const float unit = e8_kv_lattice_reader_scale(scale);
    if (!(unit > 0.0f)) {
#pragma unroll
        for (int i = 0; i < kE8LatticeGroup; ++i) { out[i] = 0; }
        return false;
    }
#pragma unroll
    for (int i = 0; i < kE8LatticeGroup; ++i) {
        out[i] = static_cast<std::int8_t>(e8_kv_lattice_int_rn(y[i] / unit));
    }
    return true;
}

}   // namespace ninfer::ops
