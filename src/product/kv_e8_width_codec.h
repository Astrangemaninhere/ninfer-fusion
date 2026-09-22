#pragma once
// kv_e8_width_codec.h -- pack / unpack / quantize / dequantize for the e8 KV family's
// three K-plane code widths. HOST-ONLY and header-only on purpose:
//
//   * it derives every byte count from product/kv_e8_width.h, so the codec and the
//     plane geometry cannot disagree, and
//   * it compiles with no CUDA, which is what lets tests/ (or a plain g++ driver)
//     round-trip it bit-exactly without the nvcc build.
//
// THE CODE CONTRACT (one signed integer per coordinate)
// -----------------------------------------------------
// The shipped e8 plane's contract, from src/ops/kernel/e8_lattice.cuh:129-130, is
// "one signed integer per coordinate". This codec keeps that contract at every width:
// each element is a w-bit TWO'S COMPLEMENT integer, range -(2^(w-1)) .. 2^(w-1)-1.
//
//   w=4  -8..7    2 codes/byte
//   w=3  -4..3    8 codes/24 bits (3 bytes) -- NOT byte-aligned per element
//   w=2  -2..1    4 codes/byte
//
// The ISO3 reference layout our own kv_formats.h:27-29 describes ("sign + 2-bit
// magnitude, 8 elements / 3 bytes") is the SAME 8-elements-in-3-bytes shape but a
// DIFFERENT mapping (sign-magnitude over a /3 scale boundary). Two's complement is
// used at all three widths here so that the three widths differ in exactly one
// variable -- the width -- and because a sign-magnitude mapping would import iso3's
// declared scale contract into a family that never declared it.
//
// SCALE BOUNDARY: one FP16 scale per 64-channel group, scale = amax / 2^(w-1). The
// divisor tracks the width (8 / 4 / 2), which is the same choice the shipped row makes
// at w=4 (amax/7 vs amax/8 is a half-step, not a geometry change: 16 bits per 64
// elements is the term that matters and it is width-independent).
//
// WHAT THIS FILE IS, AND WHAT NOW SITS BESIDE IT (rebuilt 2026-09-18)
// ------------------------------------------------------------------
// This header is the SCALAR codec: one signed integer per coordinate, at all three
// widths. Every function below is unchanged and is still the shipped W4 row's codec.
//
// It used to carry a "WHAT IS *NOT* HERE -- the E8 LATTICE PROJECTION" block saying the
// lattice was absent by deliberate choice. HALF OF THAT BLOCK IS STILL TRUE AND HALF OF
// IT IS NOW FALSE, so it is rebuilt here rather than left as a false claim:
//
//   STILL TRUE. src/ops/kernel/e8_lattice.cuh:125-137 measured that the projection's
//   half-integer coset needs one more bit per element than a one-integer-per-coordinate
//   plane has, and 47.6% of real blocks take that coset, so at w=3 (floor 4) and w=2
//   (floor 3) the projection is NOT representable in the plane THIS scalar codec fills.
//   Applying it here anyway is the +3.65 dB / 2.32x-MSE configuration that note warns
//   about. That is a property of the INTEGER-CODE consumer and it has not changed.
//
//   NOW FALSE. The note's second escape -- "where the consumer reconstructs the lattice
//   point rather than an integer code" -- has been built, and it clears the floor
//   without spending the extra bit. That codec is
//       src/ops/kv/e8_lattice_plane_codec.cuh
//   and it is the CODEC OF RECORD for the two narrow widths:
//       e8_kv_plane_codec_of_record(W3) == E8KvPlaneCodec::Lattice
//       e8_kv_plane_codec_of_record(W2) == E8KvPlaneCodec::Lattice
//       e8_kv_plane_codec_of_record(W4) == E8KvPlaneCodec::Scalar   (this file)
//   so a W3 or W2 K plane no longer means "a narrower scalar code in the same plane
//   shape" -- it means an E8 lattice point. The widths are unchanged and so are the byte
//   counts: the lattice codeword is 16 bits at W2 and 24 at W3, i.e. exactly the 2 and 3
//   bytes per 8 elements e8_kv_code_bytes_per_8() reports, and that identity is pinned by
//   static_asserts in the lattice header.
//
// WHY BOTH STILL LIVE HERE. The scalar path is NOT deleted. It is (a) the shipped W4 row,
// whose stored bytes the 55-check suite pins at every width -- including W3 and W2 -- and
// (b) the priced control arm every lattice number is measured against. A plane of the two
// codecs has the SAME SIZE but a different MEANING, so the tier name is the only
// discriminator; that is why the decision is one function and not a scattered flag.

#include "product/kv_e8_width.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ninfer::product {

// FP16 bit-pattern helpers (host, no CUDA), declared first because the plane
// encode/decode below stores the scale plane as raw uint16 bit patterns.
[[nodiscard]] inline std::uint16_t e8_kv_fp16_bits(float v) noexcept {
    std::uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    const std::uint32_t sign = (f >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((f >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t man = f & 0x7FFFFFu;
    if (exp <= 0) { return static_cast<std::uint16_t>(sign); }        // underflow -> +-0
    if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7C00u); }  // overflow -> inf
    man = (man + 0x1000u) >> 13;   // round to nearest, 10-bit mantissa
    if (man == 0x400u) {
        man = 0;
        ++exp;
        if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7C00u); }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | man);
}

[[nodiscard]] inline float e8_kv_fp16_from_bits(std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1Fu;
    std::uint32_t man = h & 0x3FFu;
    std::uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sign; }
        else {
            // subnormal: normalize
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
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// Two's complement code range for a width.
[[nodiscard]] constexpr std::int32_t e8_kv_code_min(E8KvWidth w) noexcept {
    return -(1 << (static_cast<std::int32_t>(w) - 1));
}
[[nodiscard]] constexpr std::int32_t e8_kv_code_max(E8KvWidth w) noexcept {
    return (1 << (static_cast<std::int32_t>(w) - 1)) - 1;
}
static_assert(e8_kv_code_min(E8KvWidth::W4) == -8 && e8_kv_code_max(E8KvWidth::W4) == 7, "w4 range");
static_assert(e8_kv_code_min(E8KvWidth::W3) == -4 && e8_kv_code_max(E8KvWidth::W3) == 3, "w3 range");
static_assert(e8_kv_code_min(E8KvWidth::W2) == -2 && e8_kv_code_max(E8KvWidth::W2) == 1, "w2 range");

// The scale divisor the boundary above uses.
[[nodiscard]] constexpr std::int32_t e8_kv_scale_divisor(E8KvWidth w) noexcept {
    return 1 << (static_cast<std::int32_t>(w) - 1);   // 8 / 4 / 2
}

// ---------------------------------------------------------------------------
// Bit layout. Element i occupies bit (i * w) of a little-endian integer of 8*w bits,
// written out as e8_kv_code_bytes_per_8(w) bytes. One layout, three widths, so a
// mismatch between a writer and a reader at different widths cannot silently produce
// plausible-looking codes: the widths differ in stride, not in endianness.
// ---------------------------------------------------------------------------

// Pack 8 codes into the group's bytes. Codes must already be in range.
inline void e8_kv_pack_group(E8KvWidth w, const std::int8_t codes[8],
                             std::uint8_t* out) noexcept {
    const std::int32_t width = static_cast<std::int32_t>(w);
    const std::int32_t nbytes = e8_kv_code_bytes_per_8(w);
    std::uint32_t acc = 0;
    for (std::int32_t i = 0; i < 8; ++i) {
        const std::uint32_t bits =
            static_cast<std::uint32_t>(codes[i]) & ((1u << width) - 1u);
        acc |= bits << (i * width);
    }
    for (std::int32_t b = 0; b < nbytes; ++b) {
        out[b] = static_cast<std::uint8_t>((acc >> (8 * b)) & 0xFFu);
    }
}

// Unpack the group's bytes back to 8 sign-extended codes.
inline void e8_kv_unpack_group(E8KvWidth w, const std::uint8_t* in,
                               std::int8_t codes[8]) noexcept {
    const std::int32_t width = static_cast<std::int32_t>(w);
    const std::int32_t nbytes = e8_kv_code_bytes_per_8(w);
    std::uint32_t acc = 0;
    for (std::int32_t b = 0; b < nbytes; ++b) {
        acc |= static_cast<std::uint32_t>(in[b]) << (8 * b);
    }
    const std::uint32_t mask = (1u << width) - 1u;
    const std::uint32_t sign = 1u << (width - 1);
    for (std::int32_t i = 0; i < 8; ++i) {
        std::uint32_t v = (acc >> (i * width)) & mask;
        if ((v & sign) != 0u) { v |= ~mask; }   // sign-extend
        codes[i] = static_cast<std::int8_t>(static_cast<std::int32_t>(v));
    }
}

// Clamp / round a real value onto the width's code grid.
[[nodiscard]] inline std::int8_t e8_kv_code_of(E8KvWidth w, float x, float scale) noexcept {
    if (!(scale > 0.0f)) { return 0; }   // a zero (or NaN) group scale: all-zero group
    float q = std::nearbyint(x / scale);
    const float lo = static_cast<float>(e8_kv_code_min(w));
    const float hi = static_cast<float>(e8_kv_code_max(w));
    if (q < lo) { q = lo; }
    if (q > hi) { q = hi; }
    return static_cast<std::int8_t>(static_cast<std::int32_t>(q));
}

// Quantize one 8-element slice. `scale` is the group's scale, supplied by the caller so
// the 64-channel boundary stays outside the codec.
inline void e8_kv_quantize_group(E8KvWidth w, const float x[8], float scale,
                                 std::int8_t codes[8]) noexcept {
    for (std::int32_t i = 0; i < 8; ++i) { codes[i] = e8_kv_code_of(w, x[i], scale); }
}

inline void e8_kv_dequantize_group(E8KvWidth w, const std::int8_t codes[8], float scale,
                                   float x[8]) noexcept {
    (void)w;
    for (std::int32_t i = 0; i < 8; ++i) {
        x[i] = static_cast<float>(codes[i]) * scale;
    }
}

// The group scale the boundary defines: amax / 2^(w-1) over the group's elements.
[[nodiscard]] inline float e8_kv_group_scale(E8KvWidth w, const float* x,
                                             std::int32_t n) noexcept {
    float amax = 0.0f;
    for (std::int32_t i = 0; i < n; ++i) {
        const float a = std::fabs(x[i]);
        if (a > amax) { amax = a; }
    }
    return amax / static_cast<float>(e8_kv_scale_divisor(w));
}

// ---------------------------------------------------------------------------
// Whole-plane encode / decode, for a test or a host-side requant. The plane is
// `rows` rows of kE8KvHeadDim elements, one group of 64 per row-quarter; the scale
// plane is one FP16 (stored as the raw uint16 bit pattern) per group, which is the
// geometry kv_e8_width.h prices.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::int32_t e8_kv_groups_per_row() noexcept {
    return kE8KvHeadDim / kE8KvScaleGroup;   // 4
}

// Encode `rows` rows: codes into `code_out` (e8_kv_row_code_bytes(w) * rows bytes) and
// fp16 scales into `scale_out` (groups_per_row * rows uint16s).
inline void e8_kv_encode_plane(E8KvWidth w, const float* x, std::int32_t rows,
                               std::uint8_t* code_out, std::uint16_t* scale_out) noexcept {
    const std::int32_t gpr  = e8_kv_groups_per_row();
    const std::int32_t cbr  = e8_kv_row_code_bytes(w);
    const std::int32_t cpg  = e8_kv_code_bytes_per_8(w);
    for (std::int32_t r = 0; r < rows; ++r) {
        const float* row = x + static_cast<std::size_t>(r) * kE8KvHeadDim;
        std::uint8_t* codes = code_out + static_cast<std::size_t>(r) * cbr;
        std::uint16_t* scales = scale_out + static_cast<std::size_t>(r) * gpr;
        for (std::int32_t g = 0; g < gpr; ++g) {
            const float* grp = row + g * kE8KvScaleGroup;
            const float scale = e8_kv_group_scale(w, grp, kE8KvScaleGroup);
            scales[g] = e8_kv_fp16_bits(scale);
            for (std::int32_t s = 0; s < kE8KvScaleGroup; s += 8) {
                std::int8_t c[8];
                e8_kv_quantize_group(w, grp + s, scale, c);
                // group index inside the row, then the 8-element slot
                const std::int32_t byte_off =
                    (g * (kE8KvScaleGroup / 8) + (s / 8)) * cpg;
                e8_kv_pack_group(w, c, codes + byte_off);
            }
        }
    }
}

inline void e8_kv_decode_plane(E8KvWidth w, const std::uint8_t* code_in,
                               const std::uint16_t* scale_in, std::int32_t rows,
                               float* x) noexcept {
    const std::int32_t gpr = e8_kv_groups_per_row();
    const std::int32_t cbr = e8_kv_row_code_bytes(w);
    const std::int32_t cpg = e8_kv_code_bytes_per_8(w);
    for (std::int32_t r = 0; r < rows; ++r) {
        float* row = x + static_cast<std::size_t>(r) * kE8KvHeadDim;
        const std::uint8_t* codes = code_in + static_cast<std::size_t>(r) * cbr;
        const std::uint16_t* scales = scale_in + static_cast<std::size_t>(r) * gpr;
        for (std::int32_t g = 0; g < gpr; ++g) {
            float* grp = row + g * kE8KvScaleGroup;
            const float scale = e8_kv_fp16_from_bits(scales[g]);
            for (std::int32_t s = 0; s < kE8KvScaleGroup; s += 8) {
                std::int8_t c[8];
                const std::int32_t byte_off =
                    (g * (kE8KvScaleGroup / 8) + (s / 8)) * cpg;
                e8_kv_unpack_group(w, codes + byte_off, c);
                e8_kv_dequantize_group(w, c, scale, grp + s);
            }
        }
    }
}

} // namespace ninfer::product
