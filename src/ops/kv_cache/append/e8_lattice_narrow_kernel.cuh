#pragma once
// e8_lattice_narrow_kernel.cuh -- THE APPEND WRITER FOR THE NARROW e8 K PLANE (W3 / W2).
//
// WHY THIS FILE EXISTS
// --------------------
// The e8 family has three widths (`src/product/kv_e8_width.h`: W4 shipped, W3 and W2 new)
// and exactly ONE append arm: `gqa_attention_prefill_e8.cu`'s `gqa_kv_append_e8_launch`,
// whose own first line says it is "the packed 4-bit (E8-lattice K / i4 V)" instantiation
// set. There is no 3-bit or 2-bit writer on the append route anywhere in the tree:
//
//   * `src/ops/kv_cache/append/launch.cu`'s `launch_full` is {FP8, I8, else BF16} and has
//     no e8 arm at all (its BF16 fallback casts `cache.k_pages.data` to
//     `__nv_bfloat16*`, which over a 128/96/64-byte row stride is the corruption the
//     256 B/row check in `kv_cache_append.cpp` exists to refuse);
//   * the five other arms in this directory (hq/k8v4/nvfp4/rk2v4e8 + this one's base) are
//     all "ADDITIVE, NOT wired into any build target" fork-survey imports, and the three
//     with `e8` in their names are a DIFFERENT CODEC at the same geometry -- an E8
//     ROOT-CYLINDER code (2 bytes per 8 dims, a 240-root codebook index + a radius byte),
//     not this family's per-element codes over a shared scale. See the AXIS block in
//     `src/product/kv_e8_width.h:13-46`: "They are the same GEOMETRY. They are NOT the
//     same CODEC";
//   * the W3/W2 codec of record's DEVICE arm (`e8_lattice_kv_plane.cuh`'s
//     `e8_kv_lattice_encode_group<2>,<3>`) exists and is compiled, but its only
//     instantiator is `e8_lattice_kv_plane_inst.cu`, whose own charter is that it "adds no
//     host entry point, no launch wrapper, no API and no caller".
//
// So the codec existed and the plane did not. This file is the plane: the row writer that
// hands the codec RAW K and puts its plate bytes and its scale word where the geometry of
// record says they go.
//
// WHAT IT DOES NOT DO
// -------------------
// It does NOT touch `kv_cache_append.cpp`'s e8 refusal, does NOT relax the kFullHeadDim
// (256) shape check, and does NOT widen any admission. A `kv_cache_append` call still
// refuses the e8 family by name exactly as before. This arm is reachable through its own
// declared entry point (`detail::kv_cache_append_e8_lattice_launch`), which is the same
// standing the 4-bit tier's `gqa_kv_append_e8_launch` has -- and lifting the op-level
// refusal is a separate decision that needs the extent-aware shape check and the owner's
// call, not this file.
//
// THE V PLANE IS NOT NARROWED, AND THAT IS THE FAMILY'S OWN RULE
// --------------------------------------------------------------
// `kv_e8_width.h`: "the V plane is NOT a function of the width: this family narrows K only
// ... and the shipped V is i4 at 4 bits + the same g64 FP16 scale plane, i.e. the W4
// geometry" (`e8_kv_v_plane_bytes() == 8704`). So V here is byte-for-byte the shipped W4
// V plane, written through the ONE spelling of that format that the tree already has --
// `gqa_kv_quant_i4_code()` / `gqa_kv_pack_i4()` / `kv_scale_half()` -- rather than a second
// copy of it. The CLAMSHARE note in `gqa_attention_kv_quant.cuh:244-267` is why that
// matters: the K side of that format used to be spelled twelve times and the four things
// the spellings could disagree about (rounding, zero guard, clamp range, divisor) are
// exactly the things a fresh copy would get wrong.
//
// ⚠ THE SCALE CONVENTION IS THE SHIPPED ONE, INCLUDING ITS ASYMMETRY
// -----------------------------------------------------------------
// The shipped V writer stores `kv_scale_half(amax / kGqaKvI4ScaleDivisor)` and then takes
// the code from `1.0f / <that stored fp16 value>` -- NOT from `1.0f / (amax / divisor)`
// (`gqa_attention_prefill_i8.cuh:306-316`). The reciprocal of the ROUNDED scale is what
// makes `code * stored_scale` land back inside the group's range; deriving it from the
// unrounded amax would make this plane's codes differ from the shipped writer's on the
// same input. Copied deliberately, then, and not "fixed".
//
// ⚠ RAW K IN, ROTATED K OUT -- THE CODEC'S OWN DOMAIN, NOT THIS FILE'S CHOICE
// -------------------------------------------------------------------------
// `e8_lattice_kv_plane.cuh:28-67` measured the constraint (worth 1.202 dB at w3): the
// lattice codec applies `e8_lattice_hadamard64()` to each 64-group INTERNALLY, so the
// writer must hand it NATURAL-domain K. The shipped scalar K path rotates before its codec
// (`gqa_attention_decode_i8.cuh:297`), and copying that here would rotate twice -- H64 is
// its own inverse -- and silently quantise the natural coordinates instead. This file
// therefore applies NO rotation to K. The plane's stored domain is unchanged (rotated);
// only the input convention is the codec's.

#include "ninfer/ops/kv_cache_append.h"              // Tensor, the paged cache views

#include "ops/kernel/e8_lattice_kv_plane.cuh"        // the codec of record's device arm
#include "ops/kernel/gqa_attention_kv_quant.cuh"     // the i4 V codec: ONE spelling
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"       // kv_cache_fp8_src_index<Geometry>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// ---------------------------------------------------------------- the narrow K row
// A row is one head over one token: kKVCacheAppendFullHeadDim (256) elements, which is
// kE8AppendKGroups (4) of the codec's own 64-element group. Each group becomes
// `kE8LatticeGroup / kE8LatticeDim` (8) codewords of `kE8KvLatticeCodeBytesPer8<WBits>`
// bytes -- 24 B at W3, 16 B at W2 -- so one row is 96 or 64 code bytes.
inline constexpr int kE8AppendKGroups = kKVCacheAppendFullHeadDim / kE8LatticeGroup;

template <int WBits>
inline constexpr int kE8AppendKCodeBytesPerGroup =
    (kE8LatticeGroup / kE8LatticeDim) * kE8KvLatticeCodeBytesPer8<WBits>;

template <int WBits>
inline constexpr int kE8AppendKCodeBytesPerRow =
    kE8AppendKGroups * kE8AppendKCodeBytesPerGroup<WBits>;

// The one FP16 scale word per 64-group, so a row carries 4 of them -- on BOTH planes.
inline constexpr int kE8AppendScaleWordsPerRow = kE8AppendKGroups;

// V: packed i4, two codes per byte, at the W4 geometry this family never narrows.
inline constexpr int kE8AppendVCodeBytesPerRow = kKVCacheAppendFullHeadDim / 2;

static_assert(kE8AppendKGroups == 4, "a 256-wide row is four 64-groups");
static_assert(kE8AppendKCodeBytesPerRow<3> == 96, "W3 K code extent is 96 B/row");
static_assert(kE8AppendKCodeBytesPerRow<2> == 64, "W2 K code extent is 64 B/row");
// The extents this kernel writes must be the ones `d256_profile.h` already publishes and
// derives its plane identity from. That header cannot be included from a kernel TU (no
// `.cuh` under `src/ops/` includes a `product/` header), so the binding is asserted here
// against the same literals its `static_assert`s pin.
static_assert(kE8AppendKCodeBytesPerRow<3> * 64 +
                      4 * 64 * 2 == 6656,
              "W3: 96*64 + 512 must be the geometry of record's 6656 B/head-page");
static_assert(kE8AppendKCodeBytesPerRow<2> * 64 +
                      4 * 64 * 2 == 4608,
              "W2: 64*64 + 512 must be the geometry of record's 4608 B/head-page");
static_assert(kE8AppendVCodeBytesPerRow * 64 +
                      4 * 64 * 2 == 8704,
              "V stays at the shipped W4 plane: 128*64 + 512 = 8704 B/head-page");
static_assert(kE8AppendScaleWordsPerRow * 64 * 2 == 512,
              "the g64 fp16 scale block is 512 B/head-page on both planes");

// ---------------------------------------------------------------- addressing
// The narrow K plane's leading extent is the row's CODE bytes -- 96 or 64 -- and never the
// 256 the unpacked tiers use, which is the whole reason `d256_profile.h` derives the extent
// from the dtype instead of writing `kHeadDim / 2` at the call site.
template <typename Geometry, int WBits>
__device__ __forceinline__ std::int64_t e8_append_k_code_index(int physical_page, int kv_head,
                                                               int byte_in_row, int page_off) {
    return paged_kv_element_offset<kE8AppendKCodeBytesPerRow<WBits>, Geometry::KVHeads>(
        physical_page, kv_head, page_off, byte_in_row);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t e8_append_v_code_index(int physical_page, int kv_head,
                                                               int packed_d, int page_off) {
    return paged_kv_element_offset<kE8AppendVCodeBytesPerRow, Geometry::KVHeads>(
        physical_page, kv_head, page_off, packed_d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t e8_append_scale_index(int physical_page, int kv_head,
                                                              int group, int page_off) {
    return paged_kv_element_offset<kE8AppendScaleWordsPerRow, Geometry::KVHeads>(
        physical_page, kv_head, page_off, group);
}

// ===========================================================================
// ONE 64-GROUP OF ONE (token, kv_head) ROW.
//
// One THREAD owns the whole group, not a warp: `e8_kv_lattice_encode_group` takes the 64
// values as a per-thread array and rotates them itself, so the warp-cooperative shape the
// other arms in this directory use (8 elements per lane) cannot express the codec. That
// costs registers -- the group plus the codec's own working copy -- and that cost is
// accepted rather than paid for by re-spelling the codec.
// ===========================================================================
template <typename Geometry, int WBits>
__device__ __forceinline__ void kv_cache_append_full_e8_lattice_group(
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const E8KvLatticeTables& tables, std::uint8_t* __restrict__ cache_k,
        std::uint8_t* __restrict__ cache_v, __half* __restrict__ scale_k,
        __half* __restrict__ scale_v, int token, int kv_head, int group, int physical_page,
        int page_off) {
    static_assert(e8_kv_lattice_width_supported<WBits>,
                  "the narrow e8 K plane exists at 3 and 2 bits only: W4 is the shipped "
                  "packed tier and must not fall into this arm");
    constexpr int blocks = kE8LatticeGroup / kE8LatticeDim;   // 8 codewords per group

    const int d0 = group * kE8LatticeGroup;

    // ---- K: RAW K in. NO rotation here -- the codec applies its own (see the header).
    float kraw[kE8LatticeGroup];
#pragma unroll
    for (int i = 0; i < kE8LatticeGroup; ++i) {
        kraw[i] = __bfloat162float(k[kv_cache_fp8_src_index<Geometry>(kv_head, d0 + i, token)]);
    }
    std::uint16_t k_scale_bits = 0;
    std::uint8_t* k_slot =
        cache_k + e8_append_k_code_index<Geometry, WBits>(physical_page, kv_head,
                                                          group * blocks * kE8KvLatticeCodeBytesPer8<WBits>,
                                                          page_off);
    (void)e8_kv_lattice_encode_group<WBits>(tables, kraw, k_slot, &k_scale_bits);
    // The stored word is the codec's own, unrounded by anything here: a re-conversion would
    // be a second spelling of the product layer's round-half-up conversion.
    scale_k[e8_append_scale_index<Geometry>(physical_page, kv_head, group, page_off)] =
        __ushort_as_half(k_scale_bits);

    // ---- V: the shipped W4 plane, i4, narrowed by nothing.
    float v_absmax = 0.0f;
#pragma unroll
    for (int i = 0; i < kE8LatticeGroup; ++i) {
        const float a =
            fabsf(__bfloat162float(v[kv_cache_fp8_src_index<Geometry>(kv_head, d0 + i, token)]));
        if (a > v_absmax) { v_absmax = a; }
    }
    const __half v_scale =
        kv_scale_half(v_absmax > 0.0f ? v_absmax / kGqaKvI4ScaleDivisor : 0.0f);
    scale_v[e8_append_scale_index<Geometry>(physical_page, kv_head, group, page_off)] = v_scale;
    // ⚠ the reciprocal of the STORED (fp16) scale, exactly as the shipped writer takes it
    // (`gqa_attention_prefill_i8.cuh:174-175`: `const float ks_e = __half2float(ksh_e);
    // const float kinv_e = ks_e > 0.0f ? 1.0f / ks_e : 0.0f;`). The intermediate float is
    // not decoration: CUDA 13 has no unambiguous `__half > float`, so the shipped spelling
    // converts first and this one does too.
    const float v_scale_f = __half2float(v_scale);
    const float v_inv     = v_scale_f > 0.0f ? 1.0f / v_scale_f : 0.0f;
#pragma unroll
    for (int i = 0; i < kE8LatticeGroup; i += 2) {
        const int d = d0 + i;
        const float lo =
            __bfloat162float(v[kv_cache_fp8_src_index<Geometry>(kv_head, d, token)]);
        const float hi =
            __bfloat162float(v[kv_cache_fp8_src_index<Geometry>(kv_head, d + 1, token)]);
        cache_v[e8_append_v_code_index<Geometry>(physical_page, kv_head, d >> 1, page_off)] =
            gqa_kv_pack_i4(gqa_kv_quant_i4_code(lo, v_inv), gqa_kv_quant_i4_code(hi, v_inv));
    }
}

// ===========================================================================
// THE APPEND KERNEL. One thread per (token, kv_head, 64-group); the grid strides so the
// caller can size it freely, and `metadata.valid_tokens(width)` is honoured the same way
// every other arm in this directory honours it.
// ===========================================================================
template <typename Geometry, typename Metadata, int WBits>
__launch_bounds__(256) __global__ void kv_cache_append_full_e8_lattice_kernel(
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const std::int32_t* __restrict__ positions, Metadata metadata,
        std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
        __half* __restrict__ scale_k, __half* __restrict__ scale_v, E8KvLatticeTables tables,
        std::int32_t width) {
    const int tokens = metadata.valid_tokens(width);
    const int units_per_token = Geometry::KVHeads * kE8AppendKGroups;
    const std::int32_t total = tokens * units_per_token;
    const std::int32_t stride = static_cast<std::int32_t>(gridDim.x * blockDim.x);
    const std::int32_t base   = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    for (std::int32_t unit = base; unit < total; unit += stride) {
        const int group = unit % kE8AppendKGroups;
        const int head  = (unit / kE8AppendKGroups) % Geometry::KVHeads;
        const int token = unit / units_per_token;
        const int position     = positions[0] + token;
        const int physical_page = paged_kv_physical_page(metadata.block_table(), position);
        kv_cache_append_full_e8_lattice_group<Geometry, WBits>(
            k, v, tables, cache_k, cache_v, scale_k, scale_v, token, head, group, physical_page,
            position & kPagedKVPageMask);
    }
}

}   // namespace ninfer::ops

namespace ninfer::ops::detail {

// ===========================================================================
// THE ARM'S ENTRY POINTS. Declared HERE and not in `ops/kv_cache/append/launch.h`
// because they take an `E8KvLatticeTables`, which is only visible to a CUDA TU -- while
// `launch.h` is also included by `kv_cache_append.cpp`, which the build compiles with the
// host compiler. Defined in `ops/kv_cache/append/launch.cu` (already in ninfer_ops).
//
// ⚠ THEY DO NOT LIFT ANY ADMISSION. `kv_cache_append()` still refuses the whole e8 family
// by name in `kv_cache_append.cpp`, and the kFullHeadDim (256) shape check there is
// untouched. These are direct entry points -- the same standing `gqa_kv_append_e8_launch`
// has -- for a caller that already holds a pool laid out at ITS OWN tier extent.
//
// `cache.dtype` must be E8K3Kv or E8K2Kv. E8Kv (4-bit) is REFUSED here rather than served
// by a narrow writer, because the 4-bit K plane is the packed i4 codec at 128 B/row and
// this arm writes a lattice plate at 96/64 -- accepting it would be the corruption the
// kFullHeadDim check exists to prevent, one level down.
// ===========================================================================
void kv_cache_append_e8_lattice_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                       PagedKVLayerView cache, const E8KvLatticeTables& tables,
                                       cudaStream_t stream);

void kv_cache_append_e8_lattice_batch_launch(const Tensor& k, const Tensor& v,
                                             const Tensor& positions, const Tensor& valid_columns,
                                             const Tensor& table_rows,
                                             PagedKVBatchLayerView cache,
                                             const E8KvLatticeTables& tables,
                                             cudaStream_t stream);

}   // namespace ninfer::ops::detail
