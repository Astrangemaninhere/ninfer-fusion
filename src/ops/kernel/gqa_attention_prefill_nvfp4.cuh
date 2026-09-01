#pragma once

// ninfer::ops - NVFP4 GQA prompt path.
//
//   * Fill: K is rotated per 4-channel block with the baked IsoQuant matrix and
//     quantized to packed E2M1 with E4M3 per-16-group scales. V is gain-only
//     quantized without rotation.
//   * Attention: one CTA runs a warp-specialized producer/consumer pair.
//     Four producer warps stage K/V while four consumer warps run the
//     FlashAttention body (QK + online softmax + PV). For NVFP4 K, QK runs
//     on native m16n8k64.kind::mxf4nvf4 tensor cores with Q quantized
//     on-chip to E2M1 and K staged straight from the packed cache; V keeps
//     the exact BF16 PV path over the dequantized tile. FP8/ISO3 K retain
//     the exact BF16 QK path.
//
// The 32-key tile keeps two ping-pong K/V buffers inside the sm_120 opt-in
// shared-memory ceiling (98.3 KiB + flags of 101.4 KiB).

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/gqa_isoquant_rot.cuh"
#include "ops/kernel/gqa_isoquant_row_scale.cuh"
#include "ops/kernel/entropy_nvfp4_slot.cuh"

#include "core/dtype.h"

#include <cstdint>

namespace ninfer::ops {
namespace {

using namespace ninfer::ops::detail;

__device__ __forceinline__ float gqa_prefill_nvfp4_rot(float x0, float x1, float x2, float x3,
                                                       int block, int row) {
    return gqa_isoquant_rot_value(block, row, 0) * x0 +
           gqa_isoquant_rot_value(block, row, 1) * x1 +
           gqa_isoquant_rot_value(block, row, 2) * x2 +
           gqa_isoquant_rot_value(block, row, 3) * x3;
}

// Rotate eight contiguous dims (two 4-blocks) in registers.
__device__ __forceinline__ void gqa_prefill_nvfp4_rotate_8(float (&x)[8], int d) {
    const int block0 = d >> 2;
    float y0[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        y0[row] = gqa_prefill_nvfp4_rot(x[0], x[1], x[2], x[3], block0, row);
    }
#pragma unroll
    for (int row = 0; row < 4; ++row) { x[row] = y0[row]; }
    const int block1 = block0 + 1;
    float y1[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        y1[row] = gqa_prefill_nvfp4_rot(x[4], x[5], x[6], x[7], block1, row);
    }
#pragma unroll
    for (int row = 0; row < 4; ++row) { x[4 + row] = y1[row]; }
}

__device__ __forceinline__ void gqa_prefill_bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count));
}

__device__ __forceinline__ unsigned gqa_prefill_nvfp4_nibble_bits(std::uint8_t code) {
    const unsigned mag = code & 0x07u;
    const unsigned small =
        (mag >= 1 && mag <= 3) ? (0x3F00u + (mag - 1) * 0x80u) : 0u;
    const unsigned large = (mag >= 4) ? (0x4000u + (mag - 4) * 0x40u) : 0u;
    unsigned bits = small | large;
    if ((code & 0x08u) != 0) { bits |= 0x8000u; }
    return bits;
}

// ISO3 = sign-magnitude INT3: low 3 bits encode magnitude 0..7, bit3 is the
// sign (1 = negative). Negative zero encodes as zero.
__device__ __forceinline__ std::uint8_t gqa_iso3_nibble(float value, float scale) {
    float mag = roundf(fabsf(value) / scale);
    if (mag > 7.0f) { mag = 7.0f; }
    if (mag < 0.0f) { mag = 0.0f; }
    std::uint8_t code = static_cast<std::uint8_t>(mag);
    if (value < 0.0f && code != 0) { code |= 0x08u; }
    return code;
}

__device__ __forceinline__ float gqa_iso3_decode(std::uint8_t code) {
    const float mag = static_cast<float>(code & 0x07u);
    return (code & 0x08u) != 0 ? -mag : mag;
}

// ---- native mxf4nvf4 QK staging (NVFP4 K only) ----
//
// Q is quantized on-chip to packed E2M1 with per-(row,16-group) E4M3 scales
// and K stays packed in the cache; the block-scale mma instruction applies
// both scale vectors, so scores land in the scaled domain exactly like the
// decode kernel. The packed K tile keeps the decode kernel's 128-byte row
// layout consumed by gqa_prefill_mxf4_load_b_frag.

constexpr float kGqaPrefillMxf4MinScale = 0.001953125f; // 2^-9, E4M3 smallest normal
constexpr std::uint8_t kGqaPrefillMxf4E4M3One = 0x38u;  // E4M3FN encoding of 1.0

__device__ __forceinline__ void gqa_prefill_mxf4_load_a_frag(unsigned (&frag)[4],
                                                             const std::uint8_t* smem, int lane,
                                                             int k_step) {
    const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int col = (lane >> 4) * 16 + k_step * 32;
    ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], smem_addr(smem + row * 128 + col));
}

__device__ __forceinline__ void gqa_prefill_mxf4_load_b_frag(unsigned (&frag)[2],
                                                             const std::uint8_t* smem, int lane,
                                                             int n_tile, int k_step) {
    const int row = (lane & 7) + n_tile * 8;
    const int col = ((lane >> 3) & 1) * 16 + k_step * 32;
    ldmatrix_x2(frag[0], frag[1], smem_addr(smem + row * 128 + col));
}

// Lane l < 4 loads its 4-channel block, applies the baked SO(4) rotation, and
// returns the rotated block in x[]. src points at the 16-d group start.
__device__ __forceinline__ void gqa_prefill_mxf4_rotate_4(float (&x)[4],
                                                         const __nv_bfloat16* src, int group,
                                                         int lane) {
    if (lane < 4) {
        const int block = group * 4 + lane;
        const int base  = lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) { x[j] = __bfloat162float(src[base + j]); }
        const float y0 = gqa_prefill_nvfp4_rot(x[0], x[1], x[2], x[3], block, 0);
        const float y1 = gqa_prefill_nvfp4_rot(x[0], x[1], x[2], x[3], block, 1);
        const float y2 = gqa_prefill_nvfp4_rot(x[0], x[1], x[2], x[3], block, 2);
        const float y3 = gqa_prefill_nvfp4_rot(x[0], x[1], x[2], x[3], block, 3);
        x[0] = y0;
        x[1] = y1;
        x[2] = y2;
        x[3] = y3;
    } else {
        x[0] = x[1] = x[2] = x[3] = 0.0f;
    }
}

__device__ __forceinline__ float gqa_prefill_mxf4_group_max4(float local_max,
                                                             unsigned full_mask) {
    local_max = fmaxf(local_max, __shfl_xor_sync(full_mask, local_max, 1));
    local_max = fmaxf(local_max, __shfl_xor_sync(full_mask, local_max, 2));
    return local_max;
}

// Warm producer: copy the packed 128-byte K row and its 16 E4M3 group scales
// straight into the mxf4 staging tile (one 16-byte vector per 32 dims).
template <typename Geometry, int Threads>
__device__ __forceinline__ void gqa_prefill_mxf4_stage_k_packed(
    std::uint8_t* k_pk, std::uint8_t* k_sf, const std::uint8_t* cache_codes,
    const std::uint8_t* cache_scales, int kv_head, int k0, int valid_start,
    int max_query_abs, int physical_page, int tid) {
    constexpr int Bc = kNvfp4PrefillBc;
    for (int row = tid; row < Bc; row += Threads) {
        const int key = k0 + row;
        if (key <= max_query_abs && key >= valid_start) {
            const std::int64_t scale_off =
                gqa_kv_nvfp4_scale_index<Geometry>(physical_page, kv_head, 0,
                                                   key & kPagedKVPageMask);
            store_vec(&k_sf[row * 16], load_vec<uint4>(&cache_scales[scale_off]));
        } else {
            store_vec(&k_sf[row * 16], make_int4(0, 0, 0, 0));
        }
    }
    for (int chunk = tid; chunk < Bc * 8; chunk += Threads) {
        const int key_l = chunk >> 3;
        const int j     = chunk & 7;
        const int d     = j * 32;
        const int key   = k0 + key_l;
        std::uint8_t* dst = &k_pk[key_l * 128 + j * 16];
        if (key <= max_query_abs && key >= valid_start) {
            const std::int64_t code_off =
                gqa_kv_nvfp4_code_index<Geometry>(physical_page, kv_head, d,
                                                  key & kPagedKVPageMask);
            store_vec(dst, load_vec<uint4>(&cache_codes[code_off]));
        } else {
            store_vec(dst, make_int4(0, 0, 0, 0));
        }
    }
}

// Cold producer: rANS stream `tid` decodes rows (2*tid, 2*tid+1) of the packed
// 128-byte-row tile directly; all producer threads copy the slot scale tail.
template <typename Geometry, int Threads>
__device__ __forceinline__ void gqa_prefill_mxf4_stage_k_cold(
    std::uint8_t* k_pk, std::uint8_t* k_sf, const std::uint8_t* slot, int slot_bytes,
    int half, int k0, int valid_start, int max_query_abs, int tid) {
    constexpr int Bc = kNvfp4PrefillBc;
    if (tid < kEntropyNvfp4SlotStreamsPerHalf) {
        std::uint8_t* dst = k_pk + tid * kEntropyNvfp4SlotStreamBytes;
        if (!entropy_nvfp4_slot_decode_stream(slot, half, tid, dst)) {
            for (int i = 0; i < kEntropyNvfp4SlotStreamBytes; ++i) { dst[i] = 0; }
        }
    }
    const std::uint8_t* scale_tail = entropy_nvfp4_slot_scales(slot, slot_bytes);
    for (int row = tid; row < Bc; row += Threads) {
        const int key = k0 + row;
        if (key <= max_query_abs && key >= valid_start) {
            store_vec(&k_sf[row * 16], load_vec<uint4>(&scale_tail[(half * 32 + row) * 16]));
        } else {
            store_vec(&k_sf[row * 16], make_int4(0, 0, 0, 0));
        }
    }
}

// Producer dequant: one [Bc, D] K or V tile from the packed paged cache into a
// swizzled BF16 smem buffer. Producer threads are indexed 0..127. Sixteen dims
// are decoded per iteration: four bytes of E2M1 codes + one E4M3 scale become
// four BF16x2 pairs per 8-d swizzle block, multiplied by the group scale.
template <typename Geometry, int Threads>
__device__ __forceinline__ void gqa_prefill_nvfp4_stage_kv(__nv_bfloat16* dst,
                                                           const std::uint8_t* cache_codes,
                                                           const std::uint8_t* cache_scales,
                                                           int kv_head, int k0, int valid_start,
                                                           int max_query_abs,
                                                           int physical_page, int tid) {
    constexpr int D         = kGqaPrefillHeadDim;
    constexpr int Bc        = kNvfp4PrefillBc;
    constexpr int VecPerRow = D / 16; // 16 chunks of 16 dims
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key_l = chunk / VecPerRow;
        const int d     = (chunk - key_l * VecPerRow) << 4;
        const int key   = k0 + key_l;
        __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(
            &dst[key_l * D + gqa_prefill_swz(key_l, d)]);
        __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(
            &dst[key_l * D + gqa_prefill_swz(key_l, d + 8)]);
        if (key <= max_query_abs && key >= valid_start) {
            const int group   = d >> 4;
            const float scale = gqa_kv_nvfp4_e4m3_to_f32(cache_scales[
                gqa_kv_nvfp4_scale_index<Geometry>(physical_page, kv_head, group,
                                                   key & kPagedKVPageMask)]);
            const __nv_bfloat162 scale2 = __floats2bfloat162_rn(scale, scale);
            const std::uint8_t* codes =
                &cache_codes[gqa_kv_nvfp4_code_index<Geometry>(physical_page, kv_head, d,
                                                               key & kPagedKVPageMask)];
            const uint2 raw = load_vec<uint2>(codes);
            const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
            __nv_bfloat162 pair[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const unsigned lo = gqa_prefill_nvfp4_nibble_bits(bytes[i] & 0x0Fu);
                const unsigned hi = gqa_prefill_nvfp4_nibble_bits(bytes[i] >> 4);
                const unsigned bits = lo | (hi << 16);
                pair[i] = *reinterpret_cast<const __nv_bfloat162*>(&bits) * scale2;
            }
            store_vec(p0 + 0, make_int4(*reinterpret_cast<const int*>(&pair[0]),
                                        *reinterpret_cast<const int*>(&pair[1]),
                                        *reinterpret_cast<const int*>(&pair[2]),
                                        *reinterpret_cast<const int*>(&pair[3])));
            store_vec(p1 + 0, make_int4(*reinterpret_cast<const int*>(&pair[4]),
                                        *reinterpret_cast<const int*>(&pair[5]),
                                        *reinterpret_cast<const int*>(&pair[6]),
                                        *reinterpret_cast<const int*>(&pair[7])));
        } else {
            store_vec(p0 + 0, make_int4(0, 0, 0, 0));
            store_vec(p1 + 0, make_int4(0, 0, 0, 0));
        }
    }
}

// Cold half-page producer: thread `stream` (0..15) decodes its 512-nibble
// rANS stream directly into the swizzled BF16 tile, applying the slot's
// uncompressed E4M3FN scales on the fly. Out-of-range rows still advance the
// rANS state but store zero. scale_tail points at the slot's 1024-byte scale
// tail (both halves).
template <typename Geometry, bool Iso3>
__device__ __forceinline__ void gqa_prefill_nvfp4_cold_decode_kv(
    __nv_bfloat16* dst, const std::uint8_t* slot, const std::uint8_t* scale_tail, int half,
    int k0, int valid_start, int max_query_abs, int stream) {
    std::uint8_t packed[kEntropyNvfp4SlotStreamBytes];
    if (!entropy_nvfp4_slot_decode_stream(slot, half, stream, packed)) {
        for (int i = 0; i < kEntropyNvfp4SlotStreamBytes; ++i) { packed[i] = 0; }
    }
    for (int byte_index = 0; byte_index < kEntropyNvfp4SlotStreamBytes; ++byte_index) {
        const int row_in_stream = byte_index >> 7;
        const int row           = 2 * stream + row_in_stream;
        const int byte_in_row   = byte_index & 127;
        const int key           = k0 + row;
        const std::uint8_t byte = packed[byte_index];
#pragma unroll
        for (int nibble = 0; nibble < 2; ++nibble) {
            const int dim = byte_in_row * 2 + nibble;
            const std::uint8_t code = nibble == 0 ? (byte & 0x0f) : (byte >> 4);
            float value = 0.0f;
            if (key <= max_query_abs && key >= valid_start) {
                const int group = dim >> 4;
                const float scale =
                    gqa_kv_nvfp4_e4m3_to_f32(scale_tail[(half * 32 + row) * 16 + group]);
                if constexpr (Iso3) {
                    value = gqa_iso3_decode(code) * scale;
                } else {
                    // Match the warm prefill producer exactly: it dequantizes the
                    // packed code through gqa_prefill_nvfp4_nibble_bits and
                    // multiplies the BF16 value by the BF16 scale.
                    const unsigned bits = gqa_prefill_nvfp4_nibble_bits(code);
                    const float decoded =
                        __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
                    value = decoded * scale;
                }
            }
            dst[row * 256 + gqa_prefill_swz(row, dim)] = __float2bfloat16(value);
        }
    }
}

// Producer dequant for ISO3 codes: two nibbles per byte, one E4M3FN scale per
// 16-channel group. Same 16-dim iteration, code layout, and swizzled BF16
// output as the NVFP4 producer; only the nibble decode differs.
template <typename Geometry, int Threads>
__device__ __forceinline__ void gqa_prefill_iso3_stage_kv(__nv_bfloat16* dst,
                                                          const std::uint8_t* cache_codes,
                                                          const std::uint8_t* cache_scales,
                                                          int kv_head, int k0, int max_query_abs,
                                                          int physical_page, int tid) {
    constexpr int D         = kGqaPrefillHeadDim;
    constexpr int Bc        = kNvfp4PrefillBc;
    constexpr int VecPerRow = D / 16; // 16 chunks of 16 dims
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key_l = chunk / VecPerRow;
        const int d     = (chunk - key_l * VecPerRow) << 4;
        const int key   = k0 + key_l;
        __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(
            &dst[key_l * D + gqa_prefill_swz(key_l, d)]);
        __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(
            &dst[key_l * D + gqa_prefill_swz(key_l, d + 8)]);
        if (key <= max_query_abs) {
            const int group   = d >> 4;
            const float scale = gqa_kv_nvfp4_e4m3_to_f32(cache_scales[
                gqa_kv_nvfp4_scale_index<Geometry>(physical_page, kv_head, group,
                                                   key & kPagedKVPageMask)]);
            const std::uint8_t* codes =
                &cache_codes[gqa_kv_nvfp4_code_index<Geometry>(physical_page, kv_head, d,
                                                               key & kPagedKVPageMask)];
            const uint2 raw = load_vec<uint2>(codes);
            const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
            __nv_bfloat162 pair[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float lo = gqa_iso3_decode(bytes[i] & 0x0Fu) * scale;
                const float hi = gqa_iso3_decode(bytes[i] >> 4) * scale;
                pair[i]        = __floats2bfloat162_rn(lo, hi);
            }
            store_vec(p0 + 0, make_int4(*reinterpret_cast<const int*>(&pair[0]),
                                        *reinterpret_cast<const int*>(&pair[1]),
                                        *reinterpret_cast<const int*>(&pair[2]),
                                        *reinterpret_cast<const int*>(&pair[3])));
            store_vec(p1 + 0, make_int4(*reinterpret_cast<const int*>(&pair[4]),
                                        *reinterpret_cast<const int*>(&pair[5]),
                                        *reinterpret_cast<const int*>(&pair[6]),
                                        *reinterpret_cast<const int*>(&pair[7])));
        } else {
            store_vec(p0 + 0, make_int4(0, 0, 0, 0));
            store_vec(p1 + 0, make_int4(0, 0, 0, 0));
        }
    }
}

// Adds the second ISO3 V residual stage on top of an already-staged BF16 V
// tile. The main stage must have run first so dst holds the first-stage values.
template <typename Geometry, int Threads>
__device__ __forceinline__ void gqa_prefill_iso3_stage_v_residual(
    __nv_bfloat16* dst, const std::uint8_t* cache_codes, const std::uint8_t* cache_scales,
    int kv_head, int k0, int max_query_abs, int physical_page, int tid) {
    constexpr int D         = kGqaPrefillHeadDim;
    constexpr int Bc        = kNvfp4PrefillBc;
    constexpr int VecPerRow = D / 16;
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key_l = chunk / VecPerRow;
        const int d     = (chunk - key_l * VecPerRow) << 4;
        const int key   = k0 + key_l;
        __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(
            &dst[key_l * D + gqa_prefill_swz(key_l, d)]);
        __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(
            &dst[key_l * D + gqa_prefill_swz(key_l, d + 8)]);
        if (key <= max_query_abs) {
            const int group   = d >> 4;
            const float scale = gqa_kv_nvfp4_e4m3_to_f32(cache_scales[
                gqa_kv_nvfp4_scale_index<Geometry>(physical_page, kv_head, group,
                                                   key & kPagedKVPageMask)]);
            const std::uint8_t* codes =
                &cache_codes[gqa_kv_nvfp4_code_index<Geometry>(physical_page, kv_head, d,
                                                               key & kPagedKVPageMask)];
            const uint2 raw = load_vec<uint2>(codes);
            const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
            __nv_bfloat162 pair[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float lo = gqa_iso3_decode(bytes[i] & 0x0Fu) * scale;
                const float hi = gqa_iso3_decode(bytes[i] >> 4) * scale;
                pair[i]        = __floats2bfloat162_rn(lo, hi);
            }
            __nv_bfloat162 cur[8];
            cur[0] = load_vec<__nv_bfloat162>(p0 + 0);
            cur[1] = load_vec<__nv_bfloat162>(p0 + 1);
            cur[2] = load_vec<__nv_bfloat162>(p0 + 2);
            cur[3] = load_vec<__nv_bfloat162>(p0 + 3);
            cur[4] = load_vec<__nv_bfloat162>(p1 + 0);
            cur[5] = load_vec<__nv_bfloat162>(p1 + 1);
            cur[6] = load_vec<__nv_bfloat162>(p1 + 2);
            cur[7] = load_vec<__nv_bfloat162>(p1 + 3);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float lo = __bfloat162float(cur[i].x) + __bfloat162float(pair[i].x);
                const float hi = __bfloat162float(cur[i].y) + __bfloat162float(pair[i].y);
                pair[i]        = __floats2bfloat162_rn(lo, hi);
            }
            store_vec(p0 + 0, make_int4(*reinterpret_cast<const int*>(&pair[0]),
                                        *reinterpret_cast<const int*>(&pair[1]),
                                        *reinterpret_cast<const int*>(&pair[2]),
                                        *reinterpret_cast<const int*>(&pair[3])));
            store_vec(p1 + 0, make_int4(*reinterpret_cast<const int*>(&pair[4]),
                                        *reinterpret_cast<const int*>(&pair[5]),
                                        *reinterpret_cast<const int*>(&pair[6]),
                                        *reinterpret_cast<const int*>(&pair[7])));
        }
    }
}


template <typename Geometry, int Threads>
__device__ __forceinline__ void gqa_prefill_fp8_stage_kv(__nv_bfloat16* dst,
                                                         const std::uint8_t* cache_codes,
                                                         const std::uint8_t* cache_scales,
                                                         int kv_head, int k0, int max_query_abs,
                                                         int physical_page, int tid) {
    constexpr int D         = kGqaPrefillHeadDim;
    constexpr int Bc        = kNvfp4PrefillBc;
    constexpr int VecPerRow = D / 16;
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key_l = chunk / VecPerRow;
        const int d     = (chunk - key_l * VecPerRow) << 4;
        const int key   = k0 + key_l;
        __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(
            &dst[key_l * D + gqa_prefill_swz(key_l, d)]);
        __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(
            &dst[key_l * D + gqa_prefill_swz(key_l, d + 8)]);
        if (key <= max_query_abs) {
            const int group = d >> 4;
            const float scale = gqa_kv_nvfp4_e4m3_to_f32(cache_scales[
                gqa_kv_nvfp4_scale_index<Geometry>(physical_page, kv_head, group,
                                                   key & kPagedKVPageMask)]);
            const __nv_bfloat162 scale2 = __floats2bfloat162_rn(scale, scale);
            const std::uint8_t* codes = &cache_codes[
                paged_kv_element_offset<kGqaPrefillHeadDim, Geometry::KVHeads>(
                    physical_page, kv_head, key & kPagedKVPageMask, d)];
            const uint4 raw = load_vec<uint4>(codes);
            const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
            __nv_bfloat162 pair[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float lo = gqa_kv_nvfp4_e4m3_to_f32(bytes[2 * i]) * scale;
                const float hi = gqa_kv_nvfp4_e4m3_to_f32(bytes[2 * i + 1]) * scale;
                pair[i]        = __floats2bfloat162_rn(lo, hi);
            }
            store_vec(p0, make_int4(*reinterpret_cast<const int*>(&pair[0]),
                                    *reinterpret_cast<const int*>(&pair[1]),
                                    *reinterpret_cast<const int*>(&pair[2]),
                                    *reinterpret_cast<const int*>(&pair[3])));
            store_vec(p1, make_int4(*reinterpret_cast<const int*>(&pair[4]),
                                    *reinterpret_cast<const int*>(&pair[5]),
                                    *reinterpret_cast<const int*>(&pair[6]),
                                    *reinterpret_cast<const int*>(&pair[7])));
        } else {
            store_vec(p0, make_int4(0, 0, 0, 0));
            store_vec(p1, make_int4(0, 0, 0, 0));
        }
    }
}

} // namespace

// One warp owns one (token, kv_head, 16-d group) unit. K rotation runs lanes
// 0..3 over the four 4-channel sub-blocks; V uses all 16 lanes.
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void gqa_attention_prefill_fill_nvfp4_kernel(const __nv_bfloat16* __restrict__ k,
                                                 const __nv_bfloat16* __restrict__ v,
                                                 const std::int32_t* __restrict__ positions,
                                                 int layer, Metadata metadata,
                                                 std::uint8_t* __restrict__ cache_k,
                                                 std::uint8_t* __restrict__ cache_v,
                                                 std::uint8_t* __restrict__ scale_k,
                                                 std::uint8_t* __restrict__ scale_v,
                                                 std::uint8_t* __restrict__ cache_k_residual,
                                                 std::uint8_t* __restrict__ scale_k_residual,
                                                 std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads * kGqaKvNvfp4Groups;
    if (unit >= units) { return; }

    const int group                 = unit % kGqaKvNvfp4Groups;
    const int tmp                   = unit / kGqaKvNvfp4Groups;
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    page     = __shfl_sync(FullMask, page, 0);
    const int page_off = position & kPagedKVPageMask;

    // ---- K: rotate + pack ----
    float kx[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < 4) {
        const int block = group * 4 + lane;
        const std::int64_t src =
            gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16, token) + lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) { kx[j] = __bfloat162float(k[src + j]); }
        const float y0 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 0);
        const float y1 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 1);
        const float y2 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 2);
        const float y3 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 3);
        kx[0] = y0;
        kx[1] = y1;
        kx[2] = y2;
        kx[3] = y3;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            kx[j] *= gqa_kv_row_scale(layer, kv_head, group * 16 + lane * 4 + j);
        }
    }
    float kmax = fmaxf(fmaxf(fabsf(kx[0]), fabsf(kx[1])), fmaxf(fabsf(kx[2]), fabsf(kx[3])));
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        kmax = fmaxf(kmax, __shfl_xor_sync(FullMask, kmax, off));
    }
    const float kscale = fmaxf(kmax / 6.0f, 0.001953125f);
    if (lane < 4) {
        const std::int64_t code =
            gqa_kv_nvfp4_code_index<Geometry>(page, kv_head, group * 16, page_off);
        cache_k[code + 2 * lane] =
            static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(kx[0] / kscale) |
                                      (gqa_kv_nvfp4_e2m1_nibble(kx[1] / kscale) << 4));
        cache_k[code + 2 * lane + 1] =
            static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(kx[2] / kscale) |
                                      (gqa_kv_nvfp4_e2m1_nibble(kx[3] / kscale) << 4));
    }
    if (lane == 0) {
        scale_k[gqa_kv_nvfp4_scale_index<Geometry>(page, kv_head, group, page_off)] =
            gqa_kv_nvfp4_fp32_to_e4m3(kscale);
    }

    // ---- K residual: second E2M1 stage over the first-stage error ----
    if (cache_k_residual != nullptr) {
        float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (lane < 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint8_t code_j = gqa_kv_nvfp4_e2m1_nibble(kx[j] / kscale);
                res[j] = kx[j] - gqa_kv_nvfp4_e2m1_to_f32(code_j) * kscale;
            }
        }
        float rmax = fmaxf(fmaxf(fabsf(res[0]), fabsf(res[1])),
                           fmaxf(fabsf(res[2]), fabsf(res[3])));
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            rmax = fmaxf(rmax, __shfl_xor_sync(FullMask, rmax, off));
        }
        const float rscale = fmaxf(rmax / 6.0f, 0.001953125f);
        if (lane < 4) {
            const std::int64_t rcode =
                gqa_kv_nvfp4_code_index<Geometry>(page, kv_head, group * 16, page_off);
            cache_k_residual[rcode + 2 * lane] =
                static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(res[0] / rscale) |
                                          (gqa_kv_nvfp4_e2m1_nibble(res[1] / rscale) << 4));
            cache_k_residual[rcode + 2 * lane + 1] =
                static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(res[2] / rscale) |
                                          (gqa_kv_nvfp4_e2m1_nibble(res[3] / rscale) << 4));
        }
        if (lane == 0) {
            scale_k_residual[gqa_kv_nvfp4_scale_index<Geometry>(page, kv_head, group, page_off)] =
                gqa_kv_nvfp4_fp32_to_e4m3(rscale);
        }
    }

    // ---- V: gain-only pack ----
    const float v0 = lane < 16 ? __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(
                                     kv_head, group * 16 + lane, token)])
                               : 0.0f;
    float vmax = fabsf(v0);
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(FullMask, vmax, off));
    }
    const float vscale = fmaxf(vmax / 6.0f, 0.001953125f);
    if (lane < 8) {
        const float ve =
            __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16 + lane * 2,
                                                                token)]);
        const float vo =
            __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16 + lane * 2 + 1,
                                                                token)]);
        const std::int64_t code =
            gqa_kv_nvfp4_code_index<Geometry>(page, kv_head, group * 16, page_off);
        cache_v[code + lane] =
            static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(ve / vscale) |
                                      (gqa_kv_nvfp4_e2m1_nibble(vo / vscale) << 4));
    }
    if (lane == 0) {
        scale_v[gqa_kv_nvfp4_scale_index<Geometry>(page, kv_head, group, page_off)] =
            gqa_kv_nvfp4_fp32_to_e4m3(vscale);
    }
}

// ISO3 cache append: K is rotated per 4-channel block (same IsoQuant matrix as
// NVFP4), then both K and V quantize to packed sign-magnitude INT3 nibbles with
// one E4M3FN scale per 16-channel group.
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void gqa_attention_prefill_fill_iso3_kernel(const __nv_bfloat16* __restrict__ k,
                                                const __nv_bfloat16* __restrict__ v,
                                                const std::int32_t* __restrict__ positions,
                                                Metadata metadata,
                                                std::uint8_t* __restrict__ cache_k,
                                                std::uint8_t* __restrict__ cache_v,
                                                std::uint8_t* __restrict__ scale_k,
                                                std::uint8_t* __restrict__ scale_v,
                                                std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads * kGqaKvNvfp4Groups;
    if (unit >= units) { return; }

    const int group                 = unit % kGqaKvNvfp4Groups;
    const int tmp                   = unit / kGqaKvNvfp4Groups;
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    page     = __shfl_sync(FullMask, page, 0);
    const int page_off = position & kPagedKVPageMask;

    // ---- K: rotate + pack ----
    float kx[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < 4) {
        const int block = group * 4 + lane;
        const std::int64_t src =
            gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16, token) + lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) { kx[j] = __bfloat162float(k[src + j]); }
        const float y0 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 0);
        const float y1 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 1);
        const float y2 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 2);
        const float y3 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 3);
        kx[0] = y0;
        kx[1] = y1;
        kx[2] = y2;
        kx[3] = y3;
    }
    float kmax = fmaxf(fmaxf(fabsf(kx[0]), fabsf(kx[1])), fmaxf(fabsf(kx[2]), fabsf(kx[3])));
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        kmax = fmaxf(kmax, __shfl_xor_sync(FullMask, kmax, off));
    }
    const float kscale = fmaxf(kmax / 7.0f, 0.001953125f);
    if (lane < 4) {
        const std::int64_t code =
            gqa_kv_nvfp4_code_index<Geometry>(page, kv_head, group * 16, page_off);
        cache_k[code + 2 * lane] =
            static_cast<std::uint8_t>(gqa_iso3_nibble(kx[0], kscale) |
                                      (gqa_iso3_nibble(kx[1], kscale) << 4));
        cache_k[code + 2 * lane + 1] =
            static_cast<std::uint8_t>(gqa_iso3_nibble(kx[2], kscale) |
                                      (gqa_iso3_nibble(kx[3], kscale) << 4));
    }
    if (lane == 0) {
        scale_k[gqa_kv_nvfp4_scale_index<Geometry>(page, kv_head, group, page_off)] =
            gqa_kv_nvfp4_fp32_to_e4m3(kscale);
    }

    // ---- V: gain-only pack ----
    const float v0 = lane < 16 ? __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(
                                     kv_head, group * 16 + lane, token)])
                               : 0.0f;
    float vmax = fabsf(v0);
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(FullMask, vmax, off));
    }
    const float vscale = fmaxf(vmax / 7.0f, 0.001953125f);
    if (lane < 8) {
        const float ve =
            __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16 + lane * 2,
                                                                token)]);
        const float vo =
            __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16 + lane * 2 + 1,
                                                                token)]);
        const std::int64_t code =
            gqa_kv_nvfp4_code_index<Geometry>(page, kv_head, group * 16, page_off);
        cache_v[code + lane] =
            static_cast<std::uint8_t>(gqa_iso3_nibble(ve, vscale) |
                                      (gqa_iso3_nibble(vo, vscale) << 4));
    }
    if (lane == 0) {
        scale_v[gqa_kv_nvfp4_scale_index<Geometry>(page, kv_head, group, page_off)] =
            gqa_kv_nvfp4_fp32_to_e4m3(vscale);
    }
}

// Mixed cache append for the K=NVFP4 / V=ISO3 global tier: K keeps the NVFP4
// E2M1 codec after IsoQuant rotation, V stores ISO3 sign-magnitude nibbles.
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void gqa_attention_prefill_fill_nvfp4k_iso3v_kernel(
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const std::int32_t* __restrict__ positions, int layer, Metadata metadata,
        std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
        std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v,
        std::uint8_t* __restrict__ cache_k_residual, std::uint8_t* __restrict__ scale_k_residual,
        std::uint8_t* __restrict__ cache_v_residual, std::uint8_t* __restrict__ scale_v_residual,
        std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads * kGqaKvNvfp4Groups;
    if (unit >= units) { return; }

    const int group                 = unit % kGqaKvNvfp4Groups;
    const int tmp                   = unit / kGqaKvNvfp4Groups;
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    page     = __shfl_sync(FullMask, page, 0);
    const int page_off = position & kPagedKVPageMask;

    // ---- K: rotate + NVFP4 E2M1 pack ----
    float kx[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < 4) {
        const int block = group * 4 + lane;
        const std::int64_t src =
            gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16, token) + lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) { kx[j] = __bfloat162float(k[src + j]); }
        const float y0 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 0);
        const float y1 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 1);
        const float y2 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 2);
        const float y3 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 3);
        kx[0] = y0;
        kx[1] = y1;
        kx[2] = y2;
        kx[3] = y3;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            kx[j] *= gqa_kv_row_scale(layer, kv_head, group * 16 + lane * 4 + j);
        }
    }
    float kmax = fmaxf(fmaxf(fabsf(kx[0]), fabsf(kx[1])), fmaxf(fabsf(kx[2]), fabsf(kx[3])));
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        kmax = fmaxf(kmax, __shfl_xor_sync(FullMask, kmax, off));
    }
    const float kscale = fmaxf(kmax / 6.0f, 0.001953125f);
    if (lane < 4) {
        const std::int64_t code =
            gqa_kv_nvfp4_code_index<Geometry>(page, kv_head, group * 16, page_off);
        cache_k[code + 2 * lane] =
            static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(kx[0] / kscale) |
                                      (gqa_kv_nvfp4_e2m1_nibble(kx[1] / kscale) << 4));
        cache_k[code + 2 * lane + 1] =
            static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(kx[2] / kscale) |
                                      (gqa_kv_nvfp4_e2m1_nibble(kx[3] / kscale) << 4));
    }
    if (lane == 0) {
        scale_k[gqa_kv_nvfp4_scale_index<Geometry>(page, kv_head, group, page_off)] =
            gqa_kv_nvfp4_fp32_to_e4m3(kscale);
    }

    // ---- K residual: second E2M1 stage over the first-stage error ----
    if (cache_k_residual != nullptr) {
        float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (lane < 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint8_t code_j = gqa_kv_nvfp4_e2m1_nibble(kx[j] / kscale);
                res[j] = kx[j] - gqa_kv_nvfp4_e2m1_to_f32(code_j) * kscale;
            }
        }
        float rmax = fmaxf(fmaxf(fabsf(res[0]), fabsf(res[1])),
                           fmaxf(fabsf(res[2]), fabsf(res[3])));
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            rmax = fmaxf(rmax, __shfl_xor_sync(FullMask, rmax, off));
        }
        const float rscale = fmaxf(rmax / 6.0f, 0.001953125f);
        if (lane < 4) {
            const std::int64_t rcode =
                gqa_kv_nvfp4_code_index<Geometry>(page, kv_head, group * 16, page_off);
            cache_k_residual[rcode + 2 * lane] =
                static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(res[0] / rscale) |
                                          (gqa_kv_nvfp4_e2m1_nibble(res[1] / rscale) << 4));
            cache_k_residual[rcode + 2 * lane + 1] =
                static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(res[2] / rscale) |
                                          (gqa_kv_nvfp4_e2m1_nibble(res[3] / rscale) << 4));
        }
        if (lane == 0) {
            scale_k_residual[gqa_kv_nvfp4_scale_index<Geometry>(page, kv_head, group, page_off)] =
                gqa_kv_nvfp4_fp32_to_e4m3(rscale);
        }
    }

    // ---- V: gain-only ISO3 pack ----
    const float v0 = lane < 16 ? __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(
                                     kv_head, group * 16 + lane, token)])
                               : 0.0f;
    float vmax = fabsf(v0);
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(FullMask, vmax, off));
    }
    const float vscale = fmaxf(vmax / 7.0f, 0.001953125f);
    if (lane < 8) {
        const float ve =
            __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16 + lane * 2,
                                                                token)]);
        const float vo =
            __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16 + lane * 2 + 1,
                                                                token)]);
        const std::int64_t code =
            gqa_kv_nvfp4_code_index<Geometry>(page, kv_head, group * 16, page_off);
        cache_v[code + lane] =
            static_cast<std::uint8_t>(gqa_iso3_nibble(ve, vscale) |
                                      (gqa_iso3_nibble(vo, vscale) << 4));
    }
    if (lane == 0) {
        scale_v[gqa_kv_nvfp4_scale_index<Geometry>(page, kv_head, group, page_off)] =
            gqa_kv_nvfp4_fp32_to_e4m3(vscale);
    }

    // ---- V residual: second ISO3 stage over the first-stage error ----
    if (cache_v_residual != nullptr) {
        float res[2] = {0.0f, 0.0f};
        float rmax = 0.0f;
        if (lane < 8) {
            const float ve =
                __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16 + lane * 2,
                                                                    token)]);
            const float vo = __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(
                kv_head, group * 16 + lane * 2 + 1, token)]);
            const std::uint8_t ce = gqa_iso3_nibble(ve, vscale);
            const std::uint8_t co = gqa_iso3_nibble(vo, vscale);
            res[0] = ve - gqa_iso3_decode(ce) * vscale;
            res[1] = vo - gqa_iso3_decode(co) * vscale;
            rmax = fmaxf(fabsf(res[0]), fabsf(res[1]));
        } else if (lane < 16) {
            const float vd =
                __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16 + lane,
                                                                    token)]);
            const std::uint8_t code_d = gqa_iso3_nibble(vd, vscale);
            res[0] = vd - gqa_iso3_decode(code_d) * vscale;
            rmax = fabsf(res[0]);
        }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            rmax = fmaxf(rmax, __shfl_xor_sync(FullMask, rmax, off));
        }
        const float rvscale = fmaxf(rmax / 7.0f, 0.001953125f);
        if (lane < 8) {
            const std::int64_t rcode =
                gqa_kv_nvfp4_code_index<Geometry>(page, kv_head, group * 16, page_off);
            cache_v_residual[rcode + lane] =
                static_cast<std::uint8_t>(gqa_iso3_nibble(res[0], rvscale) |
                                          (gqa_iso3_nibble(res[1], rvscale) << 4));
        }
        if (lane == 0) {
            scale_v_residual[gqa_kv_nvfp4_scale_index<Geometry>(page, kv_head, group, page_off)] =
                gqa_kv_nvfp4_fp32_to_e4m3(rvscale);
        }
    }
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void gqa_attention_prefill_fill_fp8_kernel(const __nv_bfloat16* __restrict__ k,
                                               const __nv_bfloat16* __restrict__ v,
                                               const std::int32_t* __restrict__ positions,
                                               Metadata metadata,
                                               std::uint8_t* __restrict__ cache_k,
                                               std::uint8_t* __restrict__ cache_v,
                                               std::uint8_t* __restrict__ scale_k,
                                               std::uint8_t* __restrict__ scale_v,
                                               std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads * kGqaKvNvfp4Groups;
    if (unit >= units) { return; }

    const int group                 = unit % kGqaKvNvfp4Groups;
    const int tmp                   = unit / kGqaKvNvfp4Groups;
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    page     = __shfl_sync(FullMask, page, 0);
    const int page_off = position & kPagedKVPageMask;

    // ---- K: rotate + FP8 pack ----
    float kx[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < 4) {
        const int block = group * 4 + lane;
        const std::int64_t src =
            gqa_kv_nvfp4_src_index<Geometry>(kv_head, group * 16, token) + lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) { kx[j] = __bfloat162float(k[src + j]); }
        const float y0 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 0);
        const float y1 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 1);
        const float y2 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 2);
        const float y3 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 3);
        kx[0] = y0;
        kx[1] = y1;
        kx[2] = y2;
        kx[3] = y3;
    }
    float kmax = fmaxf(fmaxf(fabsf(kx[0]), fabsf(kx[1])), fmaxf(fabsf(kx[2]), fabsf(kx[3])));
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        kmax = fmaxf(kmax, __shfl_xor_sync(FullMask, kmax, off));
    }
    const float kscale = fmaxf(kmax / 448.0f, 0.001953125f);
    if (lane < 4) {
        const std::int64_t base = paged_kv_element_offset<kGqaKvNvfp4HeadDim, Geometry::KVHeads>(
            page, kv_head, page_off, group * 16 + lane * 4);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            cache_k[base + j] = gqa_kv_nvfp4_fp32_to_e4m3(kx[j] / kscale);
        }
    }
    if (lane == 0) {
        scale_k[gqa_kv_nvfp4_scale_index<Geometry>(page, kv_head, group, page_off)] =
            gqa_kv_nvfp4_fp32_to_e4m3(kscale);
    }

    // ---- V: gain-only FP8 pack ----
    const float v0 = lane < 16 ? __bfloat162float(v[gqa_kv_nvfp4_src_index<Geometry>(
                                     kv_head, group * 16 + lane, token)])
                               : 0.0f;
    float vmax = fabsf(v0);
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(FullMask, vmax, off));
    }
    const float vscale = fmaxf(vmax / 448.0f, 0.001953125f);
    if (lane < 16) {
        const std::int64_t base = paged_kv_element_offset<kGqaKvNvfp4HeadDim, Geometry::KVHeads>(
            page, kv_head, page_off, group * 16 + lane);
        cache_v[base] = gqa_kv_nvfp4_fp32_to_e4m3(v0 / vscale);
    }
    if (lane == 0) {
        scale_v[gqa_kv_nvfp4_scale_index<Geometry>(page, kv_head, group, page_off)] =
            gqa_kv_nvfp4_fp32_to_e4m3(vscale);
    }
}

// Warp-specialized FlashAttention-2 forward over the packed cache. Producer
// warps dequantize; consumer warps run the exact BF16 tensor-core attention
// body with Bc = 32.
template <typename Geometry, typename Metadata, DType KVDType, DType VVDType = KVDType>
__launch_bounds__(kNvfp4PrefillThreads, 1) __global__
    void gqa_attention_prefill_nvfp4_kernel(const __nv_bfloat16* __restrict__ q,
                                            const std::uint8_t* __restrict__ cache_k,
                                            const std::uint8_t* __restrict__ cache_v,
                                            const std::uint8_t* __restrict__ cache_k_scale,
                                            const std::uint8_t* __restrict__ cache_v_scale,
                                            const std::uint8_t* __restrict__ cache_k_residual,
                                            const std::uint8_t* __restrict__ cache_k_residual_scale,
                                            const std::uint8_t* __restrict__ cache_v_residual,
                                            const std::uint8_t* __restrict__ cache_v_residual_scale,
                                            const std::uint8_t* __restrict__ cold_k_slots,
                                            const std::uint8_t* __restrict__ cold_v_slots,
                                            const std::int32_t* __restrict__ cold_k_valid,
                                            const std::int32_t* __restrict__ cold_v_valid,
                                            int cold_slot_bytes, int sliding_window, int layer,
                                            Metadata metadata,
                                            const std::int32_t* __restrict__ positions, float scale,
                                            __nv_bfloat16* __restrict__ out, std::int32_t width) {
    constexpr int D             = kGqaPrefillHeadDim;
    constexpr int Br            = kGqaPrefillBr; // 64
    constexpr int Bc            = kNvfp4PrefillBc; // 32
    constexpr int Threads       = kNvfp4PrefillThreads; // 256
    constexpr int ProducerThreads = 128;
    constexpr int QKNt          = Bc / 8;  // 4
    constexpr int QKKs          = D / 16;  // 16
    constexpr int PVNt          = D / 8;   // 32
    constexpr int PVKs          = Bc / 16; // 2
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Threads == 256);
    static_assert(ProducerThreads == 128);
    static_assert(QKNt == 4);
    static_assert(PVKs == 2);
    static_assert(KVDType == DType::NVFP4 || KVDType == DType::FP8_E4M3FN ||
                  KVDType == DType::ISO3);
    static_assert(VVDType == DType::NVFP4 || VVDType == DType::FP8_E4M3FN ||
                  VVDType == DType::ISO3);

    extern __shared__ __align__(16) std::uint8_t nvfp4_smem[];
    constexpr bool Mxf4QK    = KVDType == DType::NVFP4;
    constexpr int Mxf4QKKs   = D / 64;
    static_assert(!Mxf4QK || Mxf4QKKs == 4);

    __nv_bfloat16* q_s = nullptr;
    std::uint8_t* q_a  = nullptr;
    std::uint8_t* q_sf = nullptr;
    std::uint8_t* k_pk0 = nullptr;
    std::uint8_t* k_sf0 = nullptr;
    std::uint8_t* k_rpk0 = nullptr;
    std::uint8_t* k_rsf0 = nullptr;
    std::uint8_t* k_pk1 = nullptr;
    std::uint8_t* k_sf1 = nullptr;
    std::uint8_t* k_rpk1 = nullptr;
    std::uint8_t* k_rsf1 = nullptr;
    __nv_bfloat16* k_s0 = nullptr;
    __nv_bfloat16* k_s1 = nullptr;
    __nv_bfloat16* v_s0 = nullptr;
    __nv_bfloat16* v_s1 = nullptr;
    volatile std::uint32_t* flags = nullptr;
    if constexpr (Mxf4QK) {
        // Q packed E2M1 + scales, two packed 32-key K main/residual tiles,
        // then the BF16 V tiles consumed by the BF16 PV body.
        std::uint8_t* smem8 = nvfp4_smem;
        q_a                 = smem8; // [Br, 128]
        q_sf                = q_a + Br * 128; // [Br, 16]
        k_pk0               = q_sf + Br * 16; // [Bc, 128]
        k_rpk0              = k_pk0 + Bc * 128;
        k_sf0               = k_rpk0 + Bc * 128; // [Bc, 16]
        k_rsf0              = k_sf0 + Bc * 16;
        k_pk1               = k_rsf0 + Bc * 16;
        k_rpk1              = k_pk1 + Bc * 128;
        k_sf1               = k_rpk1 + Bc * 128;
        k_rsf1              = k_sf1 + Bc * 16;
        v_s0 = reinterpret_cast<__nv_bfloat16*>(k_rsf1 + Bc * 16);
        v_s1 = v_s0 + Bc * D;
        flags = reinterpret_cast<volatile std::uint32_t*>(v_s1 + Bc * D);
    } else {
        q_s   = reinterpret_cast<__nv_bfloat16*>(nvfp4_smem); // [Br, D]
        k_s0 = q_s + Br * D;
        k_s1 = k_s0 + Bc * D;
        v_s0 = k_s1 + Bc * D;
        v_s1 = v_s0 + Bc * D;
        flags = reinterpret_cast<volatile std::uint32_t*>(v_s1 + Bc * D);
    }

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid, Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    // ---- stage Q into smem once (all threads) ----
    if constexpr (Mxf4QK) {
        // On-chip Q quantization: rotate each 4-channel block with the baked
        // IsoQuant matrix, then pack per-16-group E2M1 with E4M3 scales.
        for (int i = tid; i < Br * 128; i += Threads) { q_a[i] = 0; }
        for (int i = tid; i < Br * 16; i += Threads) { q_sf[i] = kGqaPrefillMxf4E4M3One; }
        __syncthreads();
        constexpr int Groups = kGqaKvNvfp4Groups;
        const int q_rows     = min(Br, tokens - q0);
        for (int unit = warp; unit < q_rows * Groups; unit += 8) {
            const int row = unit / Groups;
            const int grp = unit - row * Groups;
            const __nv_bfloat16* src =
                q + gqa_prefill_q_index<Geometry>(q_head, grp * 16, q0 + row);
            float qx[4];
            gqa_prefill_mxf4_rotate_4(qx, src, grp, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                qx[j] *= gqa_kv_row_scale_inv(layer, kv_head, grp * 16 + lane * 4 + j);
            }
            float qmax = fmaxf(fmaxf(fabsf(qx[0]), fabsf(qx[1])),
                               fmaxf(fabsf(qx[2]), fabsf(qx[3])));
            qmax       = gqa_prefill_mxf4_group_max4(qmax, FullMask);
            const float qscale = fmaxf(qmax / 6.0f, kGqaPrefillMxf4MinScale);
            if (lane < 4) {
                q_a[row * 128 + grp * 8 + 2 * lane] =
                    static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(qx[0] / qscale) |
                                              (gqa_kv_nvfp4_e2m1_nibble(qx[1] / qscale) << 4));
                q_a[row * 128 + grp * 8 + 2 * lane + 1] =
                    static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(qx[2] / qscale) |
                                              (gqa_kv_nvfp4_e2m1_nibble(qx[3] / qscale) << 4));
            }
            if (lane == 0) {
                q_sf[row * 16 + grp] = gqa_kv_nvfp4_fp32_to_e4m3(qscale);
            }
        }
    } else {
        constexpr int VecPerRow      = D / 8;
        constexpr int QRowStride     = D * Geometry::QHeads;
        const __nv_bfloat16* q_block = q + gqa_prefill_q_index<Geometry>(q_head, 0, q0);
        for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
            const int row = chunk / VecPerRow;
            const int d   = (chunk - row * VecPerRow) << 3;
            __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
            if (q0 + row < tokens) {
                float x[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    x[j] = __bfloat162float(q_block[row * QRowStride + d + j]);
                }
                gqa_prefill_nvfp4_rotate_8(x, d);
                unsigned packed[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    packed[i] = pack_bf16x2(x[2 * i], x[2 * i + 1]);
                }
                store_vec(p, make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                                        static_cast<int>(packed[2]), static_cast<int>(packed[3])));
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
    }

    for (int i = tid; i < 8; i += Threads) { flags[i] = 0; }
    if (tid == 1) { flags[1] = 1; } // K slot 0 free
    if (tid == 3) { flags[3] = 1; } // K slot 1 free
    if (tid == 5) { flags[5] = 1; } // V slot 0 free
    if (tid == 7) { flags[7] = 1; } // V slot 1 free
    __syncthreads();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int window        = (sliding_window > 0 && KVDType == DType::NVFP4) ? sliding_window : 0;
    const int visible_start = window > 0 ? max(0, base_pos + q0 - window + 1) : 0;
    const int kb_start      = visible_start / (2 * Bc);
    const int n_block64     = (max_query_abs / (2 * Bc)) + 1 - kb_start;
    const float scale_l2    = scale * Log2E;

    if (warp >= 4) {
        // ---- producer: stage packed K and dequantized V sub-tiles into
        // ping-pong smem buffers. Named barrier 0 is the full-CTA handshake;
        // producer threads first decode any cold slot half-page, synchronized
        // by producer-only named barrier 1. ----
        const int ptid = tid - ProducerThreads;
        const auto stage_v = [&](__nv_bfloat16* v_s, int k0i, int page) {
            if constexpr (VVDType == DType::FP8_E4M3FN) {
                gqa_prefill_fp8_stage_kv<Geometry, ProducerThreads>(
                    v_s, cache_v, cache_v_scale, kv_head, k0i, max_query_abs, page, ptid);
            } else if constexpr (VVDType == DType::ISO3) {
                gqa_prefill_iso3_stage_kv<Geometry, ProducerThreads>(
                    v_s, cache_v, cache_v_scale, kv_head, k0i, max_query_abs, page, ptid);
                if (cache_v_residual != nullptr) {
                    gqa_prefill_iso3_stage_v_residual<Geometry, ProducerThreads>(
                        v_s, cache_v_residual, cache_v_residual_scale, kv_head, k0i,
                        max_query_abs, page, ptid);
                }
            } else {
                gqa_prefill_nvfp4_stage_kv<Geometry, ProducerThreads>(
                    v_s, cache_v, cache_v_scale, kv_head, k0i, visible_start, max_query_abs, page,
                    ptid);
            }
        };
        const auto stage_k_bf16 = [&](__nv_bfloat16* k_s, int k0i, int page) {
            if constexpr (KVDType == DType::FP8_E4M3FN) {
                gqa_prefill_fp8_stage_kv<Geometry, ProducerThreads>(
                    k_s, cache_k, cache_k_scale, kv_head, k0i, max_query_abs, page, ptid);
            } else if constexpr (KVDType == DType::ISO3) {
                gqa_prefill_iso3_stage_kv<Geometry, ProducerThreads>(
                    k_s, cache_k, cache_k_scale, kv_head, k0i, max_query_abs, page, ptid);
            } else {
                gqa_prefill_nvfp4_stage_kv<Geometry, ProducerThreads>(
                    k_s, cache_k, cache_k_scale, kv_head, k0i, visible_start, max_query_abs, page,
                    ptid);
            }
        };
        const auto stage_k_cold_bf16 = [&](__nv_bfloat16* k_s, const std::uint8_t* k_slot,
                                           int half, int k0i) {
            if (ptid < kEntropyNvfp4SlotStreamsPerHalf) {
                gqa_prefill_nvfp4_cold_decode_kv<Geometry, false>(
                    k_s, k_slot, entropy_nvfp4_slot_scales(k_slot, cold_slot_bytes), half, k0i,
                    visible_start, max_query_abs, ptid);
            }
        };
        for (int kb = 0; kb < n_block64; ++kb) {
            const int kb64         = kb_start + kb;
            const int k0           = kb64 * 2 * Bc;
            const int table_entry  = block_table[kb64];
            const bool cold_available = table_entry <= -2 && cold_k_slots != nullptr &&
                                         cold_v_slots != nullptr && cold_k_valid != nullptr &&
                                         cold_v_valid != nullptr && cold_slot_bytes >= 1024 + 320;
            const int slot_base    = cold_available ? -table_entry - 2 : 0;
            // Region-relative flat slot index: slot * 2*KVHeads + head; the V
            // plane's valid entries sit one KVHeads block later.
            const int cold_slot_id = slot_base * (2 * Geometry::KVHeads) + kv_head;
            const bool cold        = cold_available && cold_k_valid[cold_slot_id] != 0 &&
                              cold_v_valid[cold_slot_id + Geometry::KVHeads] != 0;
            const int physical_page = cold ? 0 : table_entry;
            const std::uint8_t* k_slot =
                cold ? cold_k_slots + static_cast<std::int64_t>(cold_slot_id) * cold_slot_bytes
                     : nullptr;
            const std::uint8_t* v_slot =
                cold ? cold_v_slots + static_cast<std::int64_t>(cold_slot_id) * cold_slot_bytes
                     : nullptr;

            // ---- half 0 (slot 0) ----
            if constexpr (Mxf4QK) {
                if (cold) {
                    gqa_prefill_mxf4_stage_k_cold<Geometry, ProducerThreads>(
                        k_pk0, k_sf0, k_slot, cold_slot_bytes, 0, k0, visible_start,
                        max_query_abs, ptid);
                    for (int chunk = ptid; chunk < Bc * 8; chunk += ProducerThreads) {
                        const int key_l = chunk >> 3;
                        const int j     = chunk & 7;
                        store_vec(&k_rpk0[key_l * 128 + j * 16], make_int4(0, 0, 0, 0));
                    }
                    for (int row = ptid; row < Bc; row += ProducerThreads) {
                        store_vec(&k_rsf0[row * 16], make_int4(0, 0, 0, 0));
                    }
                } else {
                    gqa_prefill_mxf4_stage_k_packed<Geometry, ProducerThreads>(
                        k_pk0, k_sf0, cache_k, cache_k_scale, kv_head, k0, visible_start,
                        max_query_abs, physical_page, ptid);
                    if (cache_k_residual != nullptr) {
                        gqa_prefill_mxf4_stage_k_packed<Geometry, ProducerThreads>(
                            k_rpk0, k_rsf0, cache_k_residual, cache_k_residual_scale, kv_head, k0,
                            visible_start, max_query_abs, physical_page, ptid);
                    } else {
                        for (int chunk = ptid; chunk < Bc * 8; chunk += ProducerThreads) {
                            const int key_l = chunk >> 3;
                            const int j     = chunk & 7;
                            store_vec(&k_rpk0[key_l * 128 + j * 16], make_int4(0, 0, 0, 0));
                        }
                        for (int row = ptid; row < Bc; row += ProducerThreads) {
                            store_vec(&k_rsf0[row * 16], make_int4(0, 0, 0, 0));
                        }
                    }
                }
            } else {
                if (cold) {
                    stage_k_cold_bf16(k_s0, k_slot, 0, k0);
                } else {
                    stage_k_bf16(k_s0, k0, physical_page);
                }
            }
            if (cold) {
                if (ptid >= kEntropyNvfp4SlotStreamsPerHalf &&
                    ptid < 2 * kEntropyNvfp4SlotStreamsPerHalf) {
                    gqa_prefill_nvfp4_cold_decode_kv<Geometry, VVDType == DType::ISO3>(
                        v_s0, v_slot, entropy_nvfp4_slot_scales(v_slot, cold_slot_bytes), 0, k0,
                        visible_start, max_query_abs, ptid - kEntropyNvfp4SlotStreamsPerHalf);
                }
                gqa_prefill_bar_sync(1, ProducerThreads);
            } else {
                stage_v(v_s0, k0, physical_page);
            }
            gqa_prefill_bar_sync(0, Threads);

            // ---- half 1 (slot 1) ----
            if constexpr (Mxf4QK) {
                if (cold) {
                    gqa_prefill_mxf4_stage_k_cold<Geometry, ProducerThreads>(
                        k_pk1, k_sf1, k_slot, cold_slot_bytes, 1, k0 + Bc, visible_start,
                        max_query_abs, ptid);
                    for (int chunk = ptid; chunk < Bc * 8; chunk += ProducerThreads) {
                        const int key_l = chunk >> 3;
                        const int j     = chunk & 7;
                        store_vec(&k_rpk1[key_l * 128 + j * 16], make_int4(0, 0, 0, 0));
                    }
                    for (int row = ptid; row < Bc; row += ProducerThreads) {
                        store_vec(&k_rsf1[row * 16], make_int4(0, 0, 0, 0));
                    }
                } else {
                    gqa_prefill_mxf4_stage_k_packed<Geometry, ProducerThreads>(
                        k_pk1, k_sf1, cache_k, cache_k_scale, kv_head, k0 + Bc, visible_start,
                        max_query_abs, physical_page, ptid);
                    if (cache_k_residual != nullptr) {
                        gqa_prefill_mxf4_stage_k_packed<Geometry, ProducerThreads>(
                            k_rpk1, k_rsf1, cache_k_residual, cache_k_residual_scale, kv_head,
                            k0 + Bc, visible_start, max_query_abs, physical_page, ptid);
                    } else {
                        for (int chunk = ptid; chunk < Bc * 8; chunk += ProducerThreads) {
                            const int key_l = chunk >> 3;
                            const int j     = chunk & 7;
                            store_vec(&k_rpk1[key_l * 128 + j * 16], make_int4(0, 0, 0, 0));
                        }
                        for (int row = ptid; row < Bc; row += ProducerThreads) {
                            store_vec(&k_rsf1[row * 16], make_int4(0, 0, 0, 0));
                        }
                    }
                }
            } else {
                if (cold) {
                    stage_k_cold_bf16(k_s1, k_slot, 1, k0 + Bc);
                } else {
                    stage_k_bf16(k_s1, k0 + Bc, physical_page);
                }
            }
            if (cold) {
                if (ptid >= kEntropyNvfp4SlotStreamsPerHalf &&
                    ptid < 2 * kEntropyNvfp4SlotStreamsPerHalf) {
                    gqa_prefill_nvfp4_cold_decode_kv<Geometry, VVDType == DType::ISO3>(
                        v_s1, v_slot, entropy_nvfp4_slot_scales(v_slot, cold_slot_bytes), 1,
                        k0 + Bc, visible_start, max_query_abs,
                        ptid - kEntropyNvfp4SlotStreamsPerHalf);
                }
                gqa_prefill_bar_sync(1, ProducerThreads);
            } else {
                stage_v(v_s1, k0 + Bc, physical_page);
            }
            gqa_prefill_bar_sync(0, Threads);

            gqa_prefill_bar_sync(0, Threads);
        }
        return;
    }

    // ---- consumer: exact BF16 FlashAttention body over the dequantized tiles ----
    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int b_rin     = lane & 7;
    const int warp_row0 = warp * 16;

    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    constexpr int QKNt64 = 8;   // 64-key score n-tiles
    constexpr int PVKs64 = 4;   // 64-key PV contraction groups

    const auto qk_half_mxf4 = [&](const std::uint8_t* k_pk, const std::uint8_t* k_sf,
                                  const std::uint8_t* k_rpk, const std::uint8_t* k_rsf,
                                  float (&score)[QKNt][4]) {
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
#pragma unroll
        for (int k = 0; k < Mxf4QKKs; ++k) {
            unsigned af[4];
            gqa_prefill_mxf4_load_a_frag(af, q_a + warp_row0 * 128, lane, k);
            const unsigned sfa = load_vec<unsigned>(
                q_sf + warp_row0 * 16 + (gid + (lid & 1) * 8) * 16 + k * 4);
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                unsigned bf[2];
                gqa_prefill_mxf4_load_b_frag(bf, k_pk, lane, nt, k);
                const unsigned sfb = load_vec<unsigned>(k_sf + (gid + nt * 8) * 16 + k * 4);
                mma_nvfp4_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3],
                               af[0], af[1], af[2], af[3], bf[0], bf[1], sfa, sfb);
            }
        }
        // Second pass accumulates the E2M1 residual K plane.
#pragma unroll
        for (int k = 0; k < Mxf4QKKs; ++k) {
            unsigned af[4];
            gqa_prefill_mxf4_load_a_frag(af, q_a + warp_row0 * 128, lane, k);
            const unsigned sfa = load_vec<unsigned>(
                q_sf + warp_row0 * 16 + (gid + (lid & 1) * 8) * 16 + k * 4);
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                unsigned bf[2];
                gqa_prefill_mxf4_load_b_frag(bf, k_rpk, lane, nt, k);
                const unsigned sfb = load_vec<unsigned>(k_rsf + (gid + nt * 8) * 16 + k * 4);
                mma_nvfp4_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3],
                               af[0], af[1], af[2], af[3], bf[0], bf[1], sfa, sfb);
            }
        }
    };

    const auto qk_half_bf16 = [&](const __nv_bfloat16* k_s, float (&score)[QKNt][4]) {
        const int a_mat    = lane >> 3;
        const int a_rin    = lane & 7;
        const int a_rowoff = a_rin + ((a_mat & 1) << 3);
        const int b_koff   = ((lane >> 3) & 1) << 3;
        const unsigned q_sbase    = smem_addr(q_s);
        const unsigned q_lane_base =
            q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
        const unsigned q_as = static_cast<unsigned>((a_mat >> 1) << 4);
        const unsigned q_r  = static_cast<unsigned>(a_rin << 4);
        const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
        const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
        const unsigned k_sbase = smem_addr(k_s);
        const unsigned k_lane_base =
            k_sbase + static_cast<unsigned>(b_rin * 512) +
            (static_cast<unsigned>(lane >> 4) << 12);
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        {
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        gqa_prefill_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                            gqa_prefill_swz_addr(
                                k_lane_base + static_cast<unsigned>(nt2 * 4096), 0u, k_as, k_r));
            }
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            gqa_prefill_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0],
                        bf[nxt][nt2 + 1][1],
                        gqa_prefill_swz_addr(
                            k_lane_base + static_cast<unsigned>(nt2 * 4096), ck, k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }
    };

    for (int kb = 0; kb < n_block64; ++kb) {
        const int k0 = (kb_start + kb) * 2 * Bc;

        // ---- QK^T over the two 32-key halves, then one 64-key softmax ----
        gqa_prefill_bar_sync(0, Threads); // slot 0 staged by producers
        float score_a[QKNt][4];
        if constexpr (Mxf4QK) {
            qk_half_mxf4(k_pk0, k_sf0, k_rpk0, k_rsf0, score_a);
        } else {
            qk_half_bf16(k_s0, score_a);
        }

        gqa_prefill_bar_sync(0, Threads); // slot 1 staged; slot 0 read done
        float score_b[QKNt][4];
        if constexpr (Mxf4QK) {
            qk_half_mxf4(k_pk1, k_sf1, k_rpk1, k_rsf1, score_b);
        } else {
            qk_half_bf16(k_s1, score_b);
        }

        float score[QKNt64][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0]     = score_a[nt][0];
            score[nt][1]     = score_a[nt][1];
            score[nt][2]     = score_a[nt][2];
            score[nt][3]     = score_a[nt][3];
            score[QKNt + nt][0] = score_b[nt][0];
            score[QKNt + nt][1] = score_b[nt][1];
            score[QKNt + nt][2] = score_b[nt][2];
            score[QKNt + nt][3] = score_b[nt][3];
        }

        const int row0  = warp_row0 + gid;
        const int row1  = warp_row0 + gid + 8;
        const int qrow0 = q0 + row0;
        const int qrow1 = q0 + row1;
        const int qabs0 = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1 = (qrow1 < tokens) ? base_pos + qrow1 : -1;
        const bool full_score_tile =
            (q0 + Br <= tokens) && ((k0 + 2 * Bc - 1) <= (base_pos + q0)) &&
            (window == 0 || k0 >= max(0, max_query_abs - window + 1));

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt64; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt64; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                const int row0_start = (window > 0 && qabs0 >= 0) ? max(0, qabs0 - window + 1) : 0;
                const int row1_start = (window > 0 && qabs1 >= 0) ? max(0, qabs1 - window + 1) : 0;
                score[nt][0] = (qrow0 < tokens && key0 <= qabs0 && key0 >= row0_start)
                                   ? score[nt][0]
                                   : -CUDART_INF_F;
                score[nt][1] = (qrow0 < tokens && key1 <= qabs0 && key1 >= row0_start)
                                   ? score[nt][1]
                                   : -CUDART_INF_F;
                score[nt][2] = (qrow1 < tokens && key0 <= qabs1 && key0 >= row1_start)
                                   ? score[nt][2]
                                   : -CUDART_INF_F;
                score[nt][3] = (qrow1 < tokens && key1 <= qabs1 && key1 >= row1_start)
                                   ? score[nt][3]
                                   : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(m0, bm0);
        const float nm1        = fmaxf(m1, bm1);
        const float nm0_scaled = nm0 * scale_l2;
        const float nm1_scaled = nm1 * scale_l2;
        const float alpha0     = exp2_approx(__fmaf_rn(m0, scale_l2, -nm0_scaled));
        const float alpha1     = exp2_approx(__fmaf_rn(m1, scale_l2, -nm1_scaled));

        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs64][4];
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt64; ++nt) {
                const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
                const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
                const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
                const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt64; ++nt) {
                const float p00 = (score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = (score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = (score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = (score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        }

        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        // ---- O += P V over the two 32-key V halves ----
        constexpr int PVHalf  = PVNt / 2;
        constexpr int PVLoads = PVKs * PVHalf;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const __nv_bfloat16* v_s = half == 0 ? v_s0 : v_s1;
            const unsigned v_sbase = smem_addr(v_s);
            const unsigned v_lane_base =
                v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                static_cast<unsigned>(b_rin * 512);
            unsigned vf[2][4];
            {
                ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                              gqa_prefill_swz_addr(v_lane_base, 0u, v_as, v_r));
            }
#pragma unroll
            for (int li = 0; li < PVLoads; ++li) {
                const int k   = li / PVHalf;
                const int n2  = (li % PVHalf) * 2;
                const int cur = li & 1;
                const int nxt = cur ^ 1;
                if (li + 1 < PVLoads) {
                    const int k2       = (li + 1) / PVHalf;
                    const int n2b      = ((li + 1) % PVHalf) * 2;
                    const unsigned ckv = static_cast<unsigned>(n2b << 4);
                    ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                                  gqa_prefill_swz_addr(
                                      v_lane_base + static_cast<unsigned>(k2 * 8192), ckv, v_as,
                                      v_r));
                }
                const int pk = half * PVKs + k;
                mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[pk][0],
                         p_frag[pk][1], p_frag[pk][2], p_frag[pk][3], vf[cur][0], vf[cur][1]);
                mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3],
                         p_frag[pk][0], p_frag[pk][1], p_frag[pk][2], p_frag[pk][3], vf[cur][2],
                         vf[cur][3]);
            }
        }
        gqa_prefill_bar_sync(0, Threads); // both halves consumed; buffers reusable
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);

    const float inv_l0 = (l0 > 0.0f) ? __frcp_rn(l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? __frcp_rn(l1) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0    = n * 8 + 2 * lid;
        const int qrow0 = q0 + warp_row0 + gid;
        const int qrow1 = q0 + warp_row0 + gid + 8;
        if (qrow0 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (qrow1 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                           ProducerThreads);
}

} // namespace ninfer::ops
