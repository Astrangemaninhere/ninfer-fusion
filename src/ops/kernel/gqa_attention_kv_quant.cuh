#pragma once

// ninfer::ops - signed int8, per-token group-wise KV cache codec (shared device
// helpers). Quantization (append) and dequantization (stage) are FUSED into the
// GQA attention kernels themselves (decode partial kernel, prefill fill/attention);
// this header only provides the index math, the vectorized dequant, and the scalar
// quantize helper they share. There is deliberately no standalone quant/dequant
// kernel: that would defeat the halved-bandwidth goal.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {
// 64-dim Sylvester rotation (PR #35 lineage). x0/x1 carry dims (d, d+32) of one
// 64-group so the butterfly matches the g64 scale domain.
__device__ __forceinline__ void gqa_kv_hadamard64(float& x0, float& x1,
                                                  unsigned mask = 0xffffffffu) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float y0 = __shfl_xor_sync(mask, x0, offset);
        const float y1 = __shfl_xor_sync(mask, x1, offset);
        const bool hi  = (static_cast<int>(threadIdx.x) & offset) != 0;
        x0             = hi ? y0 - x0 : x0 + y0;
        x1             = hi ? y1 - x1 : x1 + y1;
    }
    const float a = x0;
    const float b = x1;
    x0            = (a + b) * 0.125f;
    x1            = (a - b) * 0.125f;
}



// S36/S45: 256-reference constant for the Qwen3.6-27B/35B KV tiers. It is NOT the
// plane geometry: the pool's KVPlaneGeometry carries leading_extent = head_dim for the
// int8 codes and head_dim/kGqaKvQuantGroup for the fp16 group scales
// (decoder_state.cpp:106-116), and the wrapper validates the paged tensors against
// exactly those extents (gqa_attention.cpp:90-91 and :126). Every index below is
// therefore derived from Geometry::HeadDim (S45, _collab/E3_s45_i8_plane_stride.md).
inline constexpr int kGqaKvQuantHeadDim = 256;
inline constexpr int kGqaKvQuantGroup   = 64;
inline constexpr int kGqaKvQuantGroups  = kGqaKvQuantHeadDim / kGqaKvQuantGroup;

// LeadingExtent is the per-(page_offset) row length in elements: one code per channel
// for the int8 plane, one fp16 scale per 64-channel group for the scale plane.
template <typename Geometry>
inline constexpr int kGqaKvQuantCodeLeading = Geometry::HeadDim;
template <typename Geometry>
inline constexpr int kGqaKvQuantScaleLeading = Geometry::HeadDim / kGqaKvQuantGroup;

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_code_index(int physical_page, int kv_head,
                                                                int d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantCodeLeading<Geometry>, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_scale_index(int physical_page, int kv_head,
                                                                 int group, int page_offset) {
    static_assert(Geometry::HeadDim % kGqaKvQuantGroup == 0,
                  "int8/E8 KV scale planes require head_dim % 64 == 0");
    return paged_kv_element_offset<kGqaKvQuantScaleLeading<Geometry>, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_src_index(int kv_head, int d, int token) {
    // Dense K/V source tensor: ne[0] is validated == cache.head_dim
    // (gqa_attention.cpp:486-487 require_shape), so the token stride is the
    // geometry's head_dim -- the 256 reference would read a 2x-strided source.
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(Geometry::HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

// Quantize one bf16 value with a precomputed 1/scale (scale is the FP16-rounded
// per-group absmax/127). Round-to-nearest-even + symmetric clamp to keep codes
// bit-identical to the CPU oracle and to bf16 parity.
__device__ __forceinline__ std::int8_t gqa_kv_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

// Saturating fp32 -> fp16 conversion for the fp16 KV scale plane. The quantize
// paths used bare `__float2half_rn(amax / divisor)`: once amax/divisor exceeds the
// largest finite fp16 (65504) that returns +inf, the reader then forms
// code * scale = 0 * inf = NaN for every element of the group (each 4-bit tier emits
// at least one zero code, so the group is never all-nonzero), and nothing in the
// format records that the group is bad -- the neighbouring groups stay finite.
// Saturating is free: the divisor is >= 7, so the clamp can only fire for a group
// amax above 65504 * 7 = 458752, ~3e4x the largest group amax (15.0) in the
// L13-L15 KV forensics dumps under /home/user/bench/kvdump_e8src.
inline constexpr float kKvScaleHalfMax = 65504.0f;

__device__ __forceinline__ __half kv_scale_half(float s) {
    return __float2half_rn(s < kKvScaleHalfMax ? s : kKvScaleHalfMax);
}

// Dequantize 8 consecutive int8 codes (dims [d, d+8), aligned to a multiple of 8
// so they lie inside one 64-group) into 8 bf16 packed as an int4, given a pointer
// to the 8 codes and the group's dequant scale. The codes are read with ONE 64-bit
// (int2) load; the pointer may be in global or shared memory. This keeps the dequant
// ALU identical whether the codes were streamed via cp.async into smem (decode) or
// read directly from the cache (prefill).
__device__ __forceinline__ int4 gqa_kv_dequant_i8x8_from(const std::int8_t* codes8, float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// ---- packed 4-bit codec (rk4v4/E8 tier) ----
// Two 4-bit codes per byte; codes are signed [-8,7] stored as offset-binary
// nibbles. group64 scales (same plane as int8). K and V both use this codec.
//
// THE E8 KV PLANE'S FORMAT IS *THIS PAIR*, AND THE FORMAT CARRIES NO MARKER.
// gqa_kv_quant_i4_code() (writer) and gqa_kv_unpack_i4() (reader) are the two
// halves of one format: a code c in [-8,7] with scale = group_amax / divisor.
// The plane is written by the decode append path and by the prefill fill path,
// and read back by both (plus the small-T TU). The divisor used to be a bare
// `7.0f` repeated at SIX call sites in two headers
//   gqa_attention_decode_i8.cuh : ksh, vsh
//   gqa_attention_prefill_i8.cuh: two fill arms, ksh_e + vsh_e each
// so a codec change could land in the header but not at all six sites, or in a
// producer but not its consumer, and the planes would then be decoded with a
// mixed convention -- a scale error times an offset on every K and V element,
// which is already visible in the FIRST decoded token's logits, with nothing in
// the format to record it. Hence: one divisor constant, one clamp range, and a
// compile-time proof that the pair is inverse over the whole alphabet.
//
// AMAXFIX: the second half of that claim was NOT true -- the *divisor* was one
// constant at all six scale sites, but the *clamp range* was still spelled as a
// bare `max(-7, min(7, ...))` at TWELVE K-writer sites in the same two headers
// (decode_i8: 4, prefill_i8: 8; the V path was the only one routed through
// gqa_kv_quant_i4_code()). That mattered as soon as the divisor moved: a
// `amax/8` scale with a `[-7,7]` clamp cannot reach the code whose job is to
// carry full scale, and the plane silently loses the whole point of the change.
// AMAXFIX routed all twelve through kGqaKvI4CodeMin/kGqaKvI4CodeMax.
//
// The convention is now DEFINED BY THE CLAMP RANGE ALONE, and the divisor is
// DERIVED from it, so the two cannot drift apart again:
//   - the code space is the tree codec's `-2^(w-1) .. 2^(w-1)-1` (w = 4);
//   - the divisor is the full-scale magnitude, i.e. -CodeMin == 2^(w-1);
//   - the negative extreme lands exactly on CodeMin, the positive extreme
//     saturates by one code -- the SAME single choice kv_e8_width_codec.h makes
//     at w=4 (divisor 2^(w-1), range [-2^(w-1), 2^(w-1)-1]), and the choice the
//     reader's offset-binary nibble map has ALWAYS decoded: `(nibble ^ 8) - 8`
//     is a bijection onto [-8,7], so the shipped writer was using 15 of the 16
//     codes the format can carry.
// MEASURED (AMAXFIX, the prior line's own instrument, /home/user/bench/kvdump_e8src
// full length, 27 dumps, 64,677 tokens, 1,034,832 groups of 64, H64 rotation,
// one FP16 scale per 64, same 4.25 b/el and the same pack layout):
//   amax/7 + [-7,7]  ->  K relRMS 10.6611 %   (the recorded anchor)
//   amax/8 + [-8,7]  ->  K relRMS  9.8520 %   (-0.686 dB, MSE 2.73425e-02 -> 2.33499e-02)
//   amax/8 + [-7,7]  ->  K relRMS 10.3173 %   (-0.285 dB only: this is what a
//                        partial landing that moved the divisor but missed the
//                        twelve clamps would have produced, leaving 0.401 dB of the
//                        0.686 dB on the table and making code_min unreachable)
// The counter-case is NOT universal: on an outlier-heavy synthetic 64-group
// whose amax is a lone spike, amax/7 wins by ~0.02 dB because its span is
// 2*amax against this convention's 1.875*amax. That case is not reachable in
// this engine's data -- see AMAXFIX REPORT.md section 5.
inline constexpr int   kGqaKvI4CodeMin      = -8;      // nibble payload range
inline constexpr int   kGqaKvI4CodeMax      = 7;
// DERIVED, not spelled again: the full-scale magnitude is the negative extreme.
inline constexpr float kGqaKvI4ScaleDivisor = static_cast<float>(-kGqaKvI4CodeMin);
static_assert(kGqaKvI4CodeMax == -kGqaKvI4CodeMin - 1,
              "kGqaKvI4ScaleDivisor is defined as -kGqaKvI4CodeMin, so the code space "
              "MUST stay the full-scale range [-2^(w-1), 2^(w-1)-1]; a clamp range whose "
              "extremes are not -divisor/divisor-1 makes the derived divisor a lie");
static_assert(kGqaKvI4ScaleDivisor == 8.0f,
              "AMAXFIX: the measured-best 4-bit scale divisor on this engine's real K is "
              "8 (amax/8 + [-8,7] = 9.8520 % vs amax/7 + [-7,7] = 10.6611 % at identical "
              "bits). Changing this number changes the meaning of EVERY stored 4-bit K "
              "and V plane byte and makes all pre-AMAXFIX planes and dumps unreadable.");
static_assert(kGqaKvI4CodeMax - kGqaKvI4CodeMin == 15,
              "the packed-4-bit writer's clamp width and the reader's nibble map "
              "must be revised together; AMAXFIX moved this from 14 to 15 IN THE SAME "
              "EDIT as CodeMin -7 -> -8 and ScaleDivisor 7.0f -> 8.0f. 14 was the "
              "15-of-16 promise; the reader has always decoded all 16.");

// CLAMSHARE: gqa_kv_quant_i4_code() short-circuits a zero inverse scale to code 0,
// and that value now reaches the K plane too (the twelve K clamps call it).  So 0 has
// to be a code the reader maps back -- the single spelling of the guard would return a
// constant offset for every all-zero group otherwise.  This is the one thing about the
// guard that a future edit to the range could quietly break.
static_assert(kGqaKvI4CodeMin <= 0 && 0 <= kGqaKvI4CodeMax,
              "gqa_kv_quant_i4_code() returns code 0 for inv_scale == 0, so 0 must lie in "
              "[kGqaKvI4CodeMin, kGqaKvI4CodeMax]; otherwise the zero-guard emits a code "
              "the reader's nibble map can never produce and every zero group decodes as an "
              "offset");

namespace detail {
// (c & 0x0f) -> ((nibble ^ 8) - 8) must give c back for every code the writer
// can emit. If this stops holding, the E8/packed-4-bit KV plane would be
// decoded with a mixed writer/reader convention.
constexpr bool gqa_kv_i4_codec_is_inverse() noexcept {
    for (int c = kGqaKvI4CodeMin; c <= kGqaKvI4CodeMax; ++c) {
        const unsigned nibble = static_cast<unsigned>(c) & 0x0fu;
        if (static_cast<int>(nibble ^ 8u) - 8 != c) { return false; }
    }
    return true;
}
static_assert(gqa_kv_i4_codec_is_inverse(),
              "gqa_kv_quant_i4_code() and gqa_kv_unpack_i4() are no longer inverse: "
              "the E8/packed-4-bit KV plane would be decoded with a mixed convention");
} // namespace detail

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_i4_code_index(int physical_page, int kv_head,
                                                             int packed_d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantCodeLeading<Geometry> / 2, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, packed_d);
}

// The clamp: ONE definition, and the ONLY place the code range is applied to a value
// that has already been scaled.  gqa_kv_quant_i4_code() below routes through it.
__device__ __forceinline__ std::int8_t gqa_kv_i4_clamp_code(int q) {
    return static_cast<std::int8_t>(max(kGqaKvI4CodeMin, min(kGqaKvI4CodeMax, q)));
}

// Writer half of the format above: the code is round-nearest and clamped to
// [kGqaKvI4CodeMin, kGqaKvI4CodeMax], and `inv_scale` must be
// 1 / (group_amax / kGqaKvI4ScaleDivisor) -- same constant as every call site,
// or the plane is unreadable.
//
// CLAMSHARE: this is now the ONLY writer half in the tree, for BOTH sides of the
// format.  The K side used to spell the same mapping as a bare
// `max(kGqaKvI4CodeMin, min(kGqaKvI4CodeMax, static_cast<int>(rintf(x * inv))))` at
// TWELVE sites (4 in gqa_attention_decode_i8.cuh, 8 in gqa_attention_prefill_i8.cuh)
// and so owned its own copy of the rounding and of the zero guard, while the V side
// called this function.  AMAXFIX had already routed the twelve through the same two
// *constants*, which made the RANGE agree but left three things spelled twice:
//   * the rounding -- the twelve used `static_cast<int>(rintf(...))`, whose result for
//     a NaN or an out-of-int-range float is UNDEFINED; __float2int_rn is defined (it
//     saturates to 0x80000000, which the clamp then turns into CodeMin);
//   * the zero guard -- the twelve had none, so a group whose fp16 scale underflowed
//     to zero and also contained a non-finite clamped to CodeMin, the most negative
//     code, standing for a group that is not near full scale at all;
//   * the scale application -- the twelve multiplied into a named temporary first,
//     and each of those six temporaries was used exactly twice.
// The twelve now call this function, so divisor, range, rounding and guard have
// exactly one spelling and one owner.  On real data this is not a behaviour change:
// with inv_scale == 0 the old expression clamped `x * 0` to 0 for every finite x, which
// is what the guard returns, so the measured writer/reader invertibility and the
// K-vs-V agreement are unchanged (CLAMSHARE REPORT.md, SASS variants D and C).
__device__ __forceinline__ std::int8_t gqa_kv_quant_i4_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    return gqa_kv_i4_clamp_code(__float2int_rn(x * inv_scale));
}

__device__ __forceinline__ std::uint8_t gqa_kv_pack_i4(std::int8_t lo, std::int8_t hi) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(lo) & 0x0fu) |
                                     ((static_cast<unsigned>(hi) & 0x0fu) << 4));
}

// Reader half of the format above; kept inverse to gqa_kv_quant_i4_code() by
// the static_assert next to the contract note.
__device__ __forceinline__ std::int8_t gqa_kv_unpack_i4(std::uint8_t packed, int high) {
    const unsigned nibble = high ? (packed >> 4) : (packed & 0x0fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

// One 8-byte packed chunk holds 16 four-bit codes. Named so a caller's staging array cannot
// disagree with gqa_kv_unpack_i4x16 about how many values come out.
inline constexpr int kGqaKvUnpackWidth = 16;

__device__ __forceinline__ void gqa_kv_unpack_i4x16(const std::uint8_t* src8,
                                                    std::int8_t* dst16) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst16[2 * i]     = gqa_kv_unpack_i4(src8[i], 0);
        dst16[2 * i + 1] = gqa_kv_unpack_i4(src8[i], 1);
    }
}

// Repack four i8 values into one little-endian 32-bit word: this is the whole pack order, and
// every caller used to spell these four lines out by hand.
__device__ __forceinline__ int gqa_kv_pack_i8x4(const std::int8_t (&values)[4]) {
    return static_cast<int>(values[0]) | (static_cast<int>(values[1]) << 8) |
           (static_cast<int>(values[2]) << 16) | (static_cast<int>(values[3]) << 24);
}

// Inverse of gqa_kv_unpack_i4x16: one int4 worth of repacked i8 values.
__device__ __forceinline__ int4
gqa_kv_pack_i8x16_to_int4(const std::int8_t (&values)[kGqaKvUnpackWidth]) {
    return make_int4(gqa_kv_pack_i8x4(*reinterpret_cast<const std::int8_t(*)[4]>(values + 0)),
                     gqa_kv_pack_i8x4(*reinterpret_cast<const std::int8_t(*)[4]>(values + 4)),
                     gqa_kv_pack_i8x4(*reinterpret_cast<const std::int8_t(*)[4]>(values + 8)),
                     gqa_kv_pack_i8x4(*reinterpret_cast<const std::int8_t(*)[4]>(values + 12)));
}

} // namespace ninfer::ops
