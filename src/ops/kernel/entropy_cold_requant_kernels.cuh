#pragma once

// ninfer::ops::detail - cold-page requantization kernel for the entropy slot
// codec (revision 2).
//
// The revision-1 slot codec rANS-encoded the stored group-16 NVFP4 code
// nibbles directly; on real Qwen3.8 pages those codes are near-uniform
// (~4.0 bits/nibble) so the fixed-slot encoder fell back to the uncompressed
// plane and the cold pool saved nothing. Requantizing the same values with
// one E4M3 scale per 64 channels skews the code distribution enough for the
// unchanged order-0 rANS to compress it: on real engine-dumped .kvc frames the
// requantized codes measure 3.31-3.86 bits/code (K 3.8455 mean, worst half
// 3.8613; V 3.3133; the g16 storage codes they replace measure 3.9533 / 3.6415).
// INSTRUMENT: dl/ransceil/rans_probe.cu (see its REPORT.md), which includes THIS
// header and runs the real entropy_nvfp4_slot_encode_kernel on those planes. The
// '2.0-2.6 bits/code' this comment carried until 2026-09-18 is not in any
// measurement; 3-bit signed requant was measured worse: 0.04-0.13
// NMSE vs 0.011-0.022 for g64 E2M1).
//
// The kernel reads one page plane in its stored format and writes fresh
// NVFP4 planes: packed E2M1 codes plus one E4M3FN scale per 64-channel group
// replicated into the four group-16 scale slots it covers. Downstream slot
// encode, decode, scale scatter, and attention producers stay byte-identical
// to revision 1; only the codes fed into the rANS change.

#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include <cuda_bf16.h> // BF16-COLD-LAND A3: the bf16 source arm reads a bf16 plane
// The ISO4E nibble codec is shared with the attention writers/readers so the
// cold requant and the warm path can never drift apart; it used to be pasted
// in here as a second, byte-identical copy (2026-09-13).
// NOTE (NAMEFIX3 2026-09-18): this file is the ONLY consumer that reached for a
// `gqa_iso4e_*` spelling. The codec definition (ops/kernel/gqa_iso3_codec.cuh) and its
// 40+ other call sites (ops/kernel/gqa_attention_prefill_nvfp4.cuh,
// gqa_attention_decode_iso3.cuh, gqa_attention_decode_nvfp4.cuh) all keep `gqa_iso3_*`,
// so the symbol family is DEFERRED WHOLE rather than half-renamed: renaming it needs
// those three files, which are another line's.
//
// CORRECTION (NAMEFIX4 2026-09-18). The sentence that stood here deferred the
// ColdRequantSource ENUMERATORS along with the gqa_iso3_* SYMBOL family, on the ground
// that the enum declaration and ops/wrapper/entropy_cold_requant.cpp "both say so".
// That reasoning was wrong, and it is worth naming why: a deferral is safe only while
// EVERY user still holds the old name. program_impl.h:11758/11800/11808 had already
// moved to EntropyColdRequantMode::Iso4eVG16 / ::Rk4v4KvG64, so the deferred name was no
// longer debt -- it was a name that did not exist, referenced from a TU ninfer_engine
// must compile. The enumerators are therefore renamed (Iso4eVG16 / Rk4v4KvG64) while
// the gqa_iso3_* SYMBOL family above is still deferred and still another line's.
#include "ops/kernel/gqa_iso3_codec.cuh"
#include "ops/launcher/entropy_cold_requant.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One block per (kv_head, page); 256 threads = 64 token rows x 4 groups of
// 64 channels. dst planes use the nvfp4 page-major layout the slot encoder
// expects: codes [128, 64, kv_heads, pages], scales [16, 64, kv_heads, pages].
__global__ void entropy_cold_requant_kernel(const std::uint8_t* __restrict__ src_codes,
                                            const std::uint8_t* __restrict__ src_scales,
                                            ColdRequantSource mode, int kv_heads,
                                            std::uint8_t* __restrict__ dst_codes,
                                            std::uint8_t* __restrict__ dst_scales) {
    const int head  = static_cast<int>(blockIdx.x);
    const int page  = static_cast<int>(blockIdx.y);
    const int token = static_cast<int>(threadIdx.x) >> 2;
    const int group = static_cast<int>(threadIdx.x) & 3;
    const int lane0 = group * 64;

    // Row strides in bytes: nvfp4 codes 256/2, nvfp4 scales 256/16,
    // int8 codes 256, int8 scales 4 fp16 = 8.
    const std::int64_t page_rows = static_cast<std::int64_t>(kPagedKVPageSize);
    const std::int64_t head_off =
        page_rows * (static_cast<std::int64_t>(head) +
                     static_cast<std::int64_t>(kv_heads) * static_cast<std::int64_t>(page));

    if (mode == ColdRequantSource::Rk4v4KvG64) {
        // BF16-COLD-LAND E3: the Rk4v4 tier's packed 4-bit codes are copied VERBATIM into
        // the raw slot's code plane and only the g64 fp16 scale is requantized, so this
        // arm returns before the shared amax/E2M1 path below. Both planes have the same
        // geometry as every other arm's source: 128 B of codes and 4 fp16 scales per
        // 64-token row, plane-major over (kv_head, page) -- the rk4v4 write path addresses
        // them with gqa_kv_i4_code_index / gqa_kv_quant_scale_index, which are exactly
        // paged_kv_element_offset<128, KVHeads> / <4, KVHeads>.
        std::uint8_t* dst_c = dst_codes + 128 * head_off + 128 * token + group * 32;
        const std::uint8_t* src_c =
            src_codes + 128 * head_off + 128 * token + group * 32;
#pragma unroll
        for (int i = 0; i < 32; ++i) { dst_c[i] = src_c[i]; }
        const __half* src_s =
            reinterpret_cast<const __half*>(src_scales + 8 * head_off + 8 * token);
        const std::uint8_t scale_byte =
            gqa_kv_nvfp4_fp32_to_e4m3(fmaxf(__half2float(src_s[group]), 0x1p-9f));
        std::uint8_t* dst_s = dst_scales + 16 * head_off + 16 * token;
        dst_s[group * 4 + 0] = scale_byte;
        dst_s[group * 4 + 1] = scale_byte;
        dst_s[group * 4 + 2] = scale_byte;
        dst_s[group * 4 + 3] = scale_byte;
        return;
    }

    float vals[64];
    if (mode == ColdRequantSource::Nvfp4G16) {
        const std::uint8_t* codes   = src_codes + 128 * head_off + 128 * token;
        const std::uint8_t* scales = src_scales + 16 * head_off + 16 * token;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            const int          d    = lane0 + i;
            const std::uint8_t byte = codes[d >> 1];
            const std::uint8_t nib  = (d & 1) != 0 ? static_cast<std::uint8_t>(byte >> 4)
                                                   : static_cast<std::uint8_t>(byte & 0x0F);
            vals[i] = gqa_kv_nvfp4_e2m1_to_f32(nib) * gqa_kv_nvfp4_e4m3_to_f32(scales[d >> 4]);
        }
    } else if (mode == ColdRequantSource::Iso4eVG16) {
        // The global NVFP4 tier stores V as ISO4E sign-magnitude INT3 nibbles in
        // the same two-per-byte plane geometry. Requant keeps the native ISO4E
        // nibble semantics so the warm producers' dequant path is unchanged;
        // only the scales are re-derived per 64 channels.
        const std::uint8_t* codes   = src_codes + 128 * head_off + 128 * token;
        const std::uint8_t* scales = src_scales + 16 * head_off + 16 * token;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            const int          d    = lane0 + i;
            const std::uint8_t byte = codes[d >> 1];
            const std::uint8_t nib  = (d & 1) != 0 ? static_cast<std::uint8_t>(byte >> 4)
                                                   : static_cast<std::uint8_t>(byte & 0x0F);
            vals[i] = gqa_iso3_decode(nib) * gqa_kv_nvfp4_e4m3_to_f32(scales[d >> 4]);
        }
    } else if (mode == ColdRequantSource::Bf16G64) {
        // BF16-COLD-LAND A3: a bf16 layer stores its plane raw -- no code plane, no
        // scale plane -- so src_scales is unused and the source is read as 256 bf16 per
        // key row, over the same `head_off`/`token` row geometry the other two arms use
        // (64 rows x kv_heads x pages, 512 B per row). Every `vals[i]` below is the exact
        // resident value; the E2M1/g64 pair produced downstream has the same shape and
        // the same meaning as the int8 arm's.
        const __nv_bfloat16* plane = reinterpret_cast<const __nv_bfloat16*>(src_codes) +
                                     256 * head_off + 256 * token;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            vals[i] = __bfloat162float(plane[lane0 + i]);
        }
    } else {
        const std::int8_t* codes = reinterpret_cast<const std::int8_t*>(src_codes) + 256 * head_off +
                                   256 * token;
        const __half* scales = reinterpret_cast<const __half*>(src_scales + 8 * head_off +
                                                              8 * token);
        const float s = __half2float(scales[group]);
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            vals[i] = static_cast<float>(codes[lane0 + i]) * s;
        }
    }

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 64; ++i) { amax = fmaxf(amax, fabsf(vals[i])); }
    const bool iso4e_out = mode == ColdRequantSource::Iso4eVG16;
    const std::uint8_t scale_byte =
        gqa_kv_nvfp4_fp32_to_e4m3(fmaxf(amax / (iso4e_out ? 7.0f : 6.0f), 0x1p-9f));
    const float s = gqa_kv_nvfp4_e4m3_to_f32(scale_byte);

    std::uint8_t* dst_c = dst_codes + 128 * head_off + 128 * token;
#pragma unroll
    for (int i = 0; i < 64; i += 2) {
        std::uint8_t lo;
        std::uint8_t hi;
        if (iso4e_out) {
            lo = gqa_iso3_nibble(vals[i], s);
            hi = gqa_iso3_nibble(vals[i + 1], s);
        } else {
            lo = gqa_kv_nvfp4_e2m1_nibble(vals[i] / s);
            hi = gqa_kv_nvfp4_e2m1_nibble(vals[i + 1] / s);
        }
        dst_c[(lane0 + i) >> 1] = static_cast<std::uint8_t>(lo | (hi << 4));
    }
    std::uint8_t* dst_s = dst_scales + 16 * head_off + 16 * token;
    dst_s[group * 4 + 0] = scale_byte;
    dst_s[group * 4 + 1] = scale_byte;
    dst_s[group * 4 + 2] = scale_byte;
    dst_s[group * 4 + 3] = scale_byte;
}

} // namespace ninfer::ops::detail
