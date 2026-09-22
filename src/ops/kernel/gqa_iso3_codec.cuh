#pragma once

// ninfer::ops - the "ISO3" KV nibble codec. ONE definition, shared by every
// path that writes or reads an ISO3 code (decode kernel, prefill fill kernels,
// prefill attention, cold requant).
//
// ---------------------------------------------------------------------------
// WHAT THIS CODEC ACTUALLY IS (verified 2026-09-13, scratch/iso3/report.md)
// ---------------------------------------------------------------------------
//   * 4 bits per element: bit3 = sign, bits0-2 = MAGNITUDE 0..7. All four bits
//     carry information; 15 distinct levels {0, +-1..+-7} * scale.
//   * one scale per 16-channel group, amax/7, stored as E4M3FN with a 2^-9
//     floor: `fmaxf(amax / 7.0f, 0.001953125f)`.
//   * two codes per byte (low nibble = even channel), 16 bytes per 16-group,
//     i.e. the NVFP4 code plane geometry.
//
//   Therefore the shipped "ISO3" is NOT the 3-bit ISO3 of
//   src/ops/kv/iso_codec.h (== tools/convert/kv_iso_ref.py), which is: 8
//   elements per group, 3 bits each (bit2 sign + 2-bit magnitude 0..3), scale
//   amax/3 in fp16, packed 8 elements into 3 bytes.  There are two differences
//   and they are independent:
//
//     (1) RATE. The reference ISO3 spends 3.00 b/el on codes and 2.00 b/el on
//         its group-8 fp16 scale = 5.00 b/el; this codec spends 4.00 + 0.50
//         (group-16 E4M3 scale) = 4.50 b/el. The engine reserves an nvfp4-sized
//         plane pair for ISO3 (product/kv_tier_formats.h "iso3 shares the nvfp4
//         plane pair", product/kv_bit_budget.h {"iso3", 450, 200}), so the
//         plane is 4-bit wide and a true 3-bit code would waste 1 bit per
//         element rather than save memory.
//
//     (2) QUANTIZER. amax/7 with a +-7 level set and round-half-away-from-zero
//         is EXACTLY the ISO4 quantizer of src/ops/kv/iso_codec.h:72-104; only
//         the code WORD assignment differs (sign-magnitude here, two's
//         complement there), which does not change any reconstructed value.
//         Recomputed on 3.2e6 i.i.d. Gaussian elements: max |xhat(here) -
//         xhat(iso4 quantizer @ E4M3 scale)| = 0.000e+00.
//
//   So: this is an ISO4-rate sign-magnitude codec wearing the ISO3 name. It is
//   numerically BETTER than the reference contract at the same footprint
//   (3.2e6 i.i.d. Gaussian elements: NMSE 7.97e-3 here vs 2.80e-2 for the
//   reference 3-bit codec), which is why nobody noticed. What is wrong is the
//   NAME and the declared 3-bit rate, not the arithmetic.
//
//   If a true 3-bit ISO3 is ever wanted it needs its own plane geometry
//   (3 bytes / 8 elements), its own reader/writer (the MMA staging path loads
//   one 4-byte word per 8 channels and cannot be reused), and its own tier
//   entry; see scratch/iso3/report.md section B.2 for the measured cost
//   (5.00 b/el, NMSE 2.80e-2) before doing that work.
//
// The two helpers below were byte-identical copies in
// gqa_attention_prefill_nvfp4.cuh (anonymous namespace) and
// entropy_cold_requant_kernels.cuh (global namespace) until 2026-09-13; a
// change to one and not the other would have desynchronised the prefill writer
// from the cold-requant reader with no compiler complaint. Do not reintroduce
// local copies.
//
// Negative zero encodes as zero (the sign bit is only set for code != 0), and
// the decode is the exact inverse of the encode for every 0..15 input.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Max representable magnitude; the scale is chosen as amax / kGqaNibbleMaxMag,
// so the group's largest element sits exactly on the last level.
inline constexpr int kGqaNibbleMaxMag = 7;

__device__ __forceinline__ std::uint8_t gqa_iso3_nibble(float value, float scale) {
    float mag = roundf(fabsf(value) / scale);
    if (mag > static_cast<float>(kGqaNibbleMaxMag)) { mag = static_cast<float>(kGqaNibbleMaxMag); }
    if (mag < 0.0f) { mag = 0.0f; }
    std::uint8_t code = static_cast<std::uint8_t>(mag);
    if (value < 0.0f && code != 0) { code |= 0x08u; }
    return code;
}

__device__ __forceinline__ float gqa_iso3_decode(std::uint8_t code) {
    const float mag = static_cast<float>(code & 0x07u);
    return (code & 0x08u) != 0 ? -mag : mag;
}

} // namespace ninfer::ops
