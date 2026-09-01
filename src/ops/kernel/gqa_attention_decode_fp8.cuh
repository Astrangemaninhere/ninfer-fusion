#pragma once

// ninfer::ops - split-KV GQA small-T attention, FP8 E4M3FN KV-cache partial
// kernel. Kept as a separate file from the BF16 kernel so the FP8 cache path
// can be tuned independently. The QK/softmax/PV tensor-core body is copied
// byte-for-byte from gqa_attention_decode_bf16.cuh; only the cache append
// (BF16 -> rotated/quantized E4M3FN planes) and the K/V tile staging
// (E4M3FN -> BF16 qkv_s) differ.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_isoquant_rot.cuh"
#include "ops/kernel/gqa_attention_prefill_nvfp4.cuh" // gqa_prefill_nvfp4_rot

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(128, 2) __global__ void gqa_attention_small_t_tc_partial_fp8_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos,
    std::uint8_t* cache_k, std::uint8_t* cache_v,
    std::uint8_t* cache_k_scale, std::uint8_t* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

    constexpr int Wc      = WarpsPerCta;
    constexpr int Br      = Wc * 16;
    constexpr int Bc      = 32;
    constexpr int D       = kGqaHeadDim;
    constexpr int Threads = Wc * 32;
    constexpr int QKNt    = Bc / 8;
    constexpr int QKKs    = D / 16;
    constexpr int PVNt    = D / 8;
    constexpr int PVKs    = Bc / 16;
    // The YaRN-extended 1,010,000-key maximum envelope spans at most 186 pages in one 27B split.
    constexpr int PageIds       = 256;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int QkvRows       = 2 * Bc;

    static_assert(QkvRows >= Br);

    __shared__ __align__(16) __nv_bfloat16 qkv_s[QkvRows * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Wc * 16 * Bc];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Br || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        // The owning split writes each new row into the quantized E4M3FN
        // planes. K is rotated per 4-channel block before quantization; V is
        // gain-only. The subsequent tile staging reads every key from the
        // cache, so no split depends on another split's cache write.
        constexpr int kFp8Groups = D / 16;
        const int append_units   = valid_tokens * kFp8Groups;
        for (int unit = warp; unit < append_units; unit += WarpsPerCta) {
            const int group = unit % kFp8Groups;
            const int token = unit / kFp8Groups;
            const int p_tok = pos[token];
            if (p_tok < split_start || p_tok >= split_end || p_tok < 0 ||
                p_tok >= logical_capacity) {
                continue;
            }
            int physical_page = lane == 0 ? paged_kv_physical_page(block_table, p_tok) : 0;
            physical_page     = __shfl_sync(FullMask, physical_page, 0);
            const int page_off = p_tok & kPagedKVPageMask;

            // K: rotate the four 4-channel blocks of this 16-group, then
            // quantize with one shared E4M3FN scale for the group.
            float kx[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (lane < 4) {
                const int block = group * 4 + lane;
                const std::int64_t src =
                    gqa_kv_new_index<Geometry>(kv_head, group * 16, token) + lane * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) { kx[j] = __bfloat162float(input.k[src + j]); }
                const float y0 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 0);
                const float y1 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 1);
                const float y2 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 2);
                const float y3 = gqa_prefill_nvfp4_rot(kx[0], kx[1], kx[2], kx[3], block, 3);
                kx[0]          = y0;
                kx[1]          = y1;
                kx[2]          = y2;
                kx[3]          = y3;
            }
            float kmax = fmaxf(fmaxf(fabsf(kx[0]), fabsf(kx[1])),
                               fmaxf(fabsf(kx[2]), fabsf(kx[3])));
#pragma unroll
            for (int off = 1; off <= 2; off <<= 1) {
                kmax = fmaxf(kmax, __shfl_xor_sync(FullMask, kmax, off));
            }
            const float kscale = fmaxf(kmax / 448.0f, 0.001953125f);
            if (lane < 4) {
                const std::int64_t base = paged_kv_element_offset<kGqaHeadDim, Geometry::KVHeads>(
                    physical_page, kv_head, page_off, group * 16 + lane * 4);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    cache_k[base + j] = gqa_kv_nvfp4_fp32_to_e4m3(kx[j] / kscale);
                }
            }
            if (lane == 0) {
                cache_k_scale[gqa_kv_nvfp4_scale_index<Geometry>(physical_page, kv_head, group,
                                                                 page_off)] =
                    gqa_kv_nvfp4_fp32_to_e4m3(kscale);
            }

            // V: gain-only FP8 E4M3FN quantization, no rotation.
            const float v0 = lane < 16
                                 ? __bfloat162float(input.v[gqa_kv_new_index<Geometry>(
                                       kv_head, group * 16 + lane, token)])
                                 : 0.0f;
            float vmax = fabsf(v0);
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                vmax = fmaxf(vmax, __shfl_xor_sync(FullMask, vmax, off));
            }
            const float vscale = fmaxf(vmax / 448.0f, 0.001953125f);
            if (lane < 16) {
                const std::int64_t base = paged_kv_element_offset<kGqaHeadDim, Geometry::KVHeads>(
                    physical_page, kv_head, page_off, group * 16 + lane);
                cache_v[base] = gqa_kv_nvfp4_fp32_to_e4m3(v0 / vscale);
            }
            if (lane == 0) {
                cache_v_scale[gqa_kv_nvfp4_scale_index<Geometry>(physical_page, kv_head, group,
                                                                 page_off)] =
                    gqa_kv_nvfp4_fp32_to_e4m3(vscale);
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const int warp_row0 = warp * 16;
    __nv_bfloat16* p_sw = &p_s[warp * 16 * Bc];

    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                    smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();
    int physical_page = physical_pages_s[0];
    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        }
        // Stage the FP8 E4M3FN K/V key tile synchronously into the swizzled
        // BF16 qkv_s tile (K at offset 0, V at offset Bc*D). Each 8-element
        // chunk shares one per-16-group scale; out-of-range rows store zero.
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l      = chunk / (D / 8);
            const int d          = (chunk - key_l * (D / 8)) * 8;
            const int key        = k0 + key_l;
            __nv_bfloat16* k_dst = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            __nv_bfloat16* v_dst = &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            if (key >= split_start && key < split_end) {
                const int group    = d >> 4;
                const int page_off = key & kPagedKVPageMask;
                const float k_scale = gqa_kv_nvfp4_e4m3_to_f32(cache_k_scale[
                    gqa_kv_nvfp4_scale_index<Geometry>(physical_page, kv_head, group, page_off)]);
                const uint2 k_raw = load_vec<uint2>(&cache_k[
                    paged_kv_element_offset<kGqaHeadDim, Geometry::KVHeads>(
                        physical_page, kv_head, page_off, d)]);
                const std::uint8_t* k_code = reinterpret_cast<const std::uint8_t*>(&k_raw);
                unsigned k_packed[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float x0 = gqa_kv_nvfp4_e4m3_to_f32(k_code[2 * i]) * k_scale;
                    const float x1 = gqa_kv_nvfp4_e4m3_to_f32(k_code[2 * i + 1]) * k_scale;
                    k_packed[i]    = pack_bf16x2(x0, x1);
                }
                store_vec(k_dst, make_int4(static_cast<int>(k_packed[0]),
                                           static_cast<int>(k_packed[1]),
                                           static_cast<int>(k_packed[2]),
                                           static_cast<int>(k_packed[3])));

                const float v_scale = gqa_kv_nvfp4_e4m3_to_f32(cache_v_scale[
                    gqa_kv_nvfp4_scale_index<Geometry>(physical_page, kv_head, group, page_off)]);
                const uint2 v_raw = load_vec<uint2>(&cache_v[
                    paged_kv_element_offset<kGqaHeadDim, Geometry::KVHeads>(
                        physical_page, kv_head, page_off, d)]);
                const std::uint8_t* v_code = reinterpret_cast<const std::uint8_t*>(&v_raw);
                unsigned v_packed[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float x0 = gqa_kv_nvfp4_e4m3_to_f32(v_code[2 * i]) * v_scale;
                    const float x1 = gqa_kv_nvfp4_e4m3_to_f32(v_code[2 * i + 1]) * v_scale;
                    v_packed[i]    = pack_bf16x2(x0, x1);
                }
                store_vec(v_dst, make_int4(static_cast<int>(v_packed[0]),
                                           static_cast<int>(v_packed[1]),
                                           static_cast<int>(v_packed[2]),
                                           static_cast<int>(v_packed[3])));
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
                store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        __syncthreads();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                         af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head0, token0);
        gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head1, token1);
        const int qabs0 = (row0 < row_count) ? pos[token0] : -1;
        const int qabs1 = (row1 < row_count) ? pos[token1] : -1;

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = k0 + col0;
            const int key1 = col1 + k0;
            score[nt][0] =
                (row0 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                    ? score[nt][0] * scale
                    : -CUDART_INF_F;
            score[nt][1] =
                (row0 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                    ? score[nt][1] * scale
                    : -CUDART_INF_F;
            score[nt][2] =
                (row1 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                    ? score[nt][2] * scale
                    : -CUDART_INF_F;
            score[nt][3] =
                (row1 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs1)
                    ? score[nt][3] * scale
                    : -CUDART_INF_F;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0    = fmaxf(m0, bm0);
        const float nm1    = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0  = nt * 8 + 2 * lid;
            const int col1  = col0 + 1;
            const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                  : 0.0f;
            const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                  : 0.0f;
            const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                  : 0.0f;
            const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);

        l0 = l0 * alpha0 + bl0;
        l1 = l1 * alpha1 + bl1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
        __syncwarp();

#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_sw[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }

    // MMA fragments hold each row in four-lane groups. Stage the final split-local
    // accumulator through shared memory so partial_acc is written as contiguous d-vector stores.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int d1   = d0 + 1;
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            qkv_s[row0 * D + d0] = __float2bfloat16(acc[n][0]);
            qkv_s[row0 * D + d1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < row_count) {
            qkv_s[row1 * D + d0] = __float2bfloat16(acc[n][2]);
            qkv_s[row1 * D + d1] = __float2bfloat16(acc[n][3]);
        }
    }
    __syncthreads();

    for (int chunk = tid; chunk < row_count * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
        }
    }
}

} // namespace ninfer::ops
