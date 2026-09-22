#pragma once

// ninfer::ops - split-KV GQA small-T attention, BF16 KV-cache partial kernel.
// Standalone from the int8 kernel (gqa_attention_decode_i8.cuh): shared scaffolding
// lives in gqa_attention_decode.cuh, but the body/append/load are not shared so the
// bf16 path can be tuned independently. Processes one KV head, one query-head
// subgroup, and one token tile; a reducer combines the split-local partials.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
// BF16-COLD-LAND A4: the raw int8-raw cold slot and its device accessors. The bf16
// tier's OWN cold record is that slot (decoder_state.cpp cold_slot_codec_of(BF16) ==
// ColdSlotCodec::Int8Raw, A1), so the decode body reuses cold_i8_slot_codes/_scales
// instead of growing a second slot format.
#include "ops/kernel/cold_i8_kernels.cuh"

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(128, 2) __global__ void gqa_attention_small_t_tc_partial_bf16_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, __nv_bfloat16* cache_k,
    __nv_bfloat16* cache_v,
    // BF16-COLD-LAND A4: the cold slot pool, PLANE-RELATIVE -- the launcher hands over
    // K = base, V = base + nb[2] for the slots and V = base + nb[1] for the validity
    // planes, exactly as the i8/nvfp4 decode launchers do, so K and V take the same flat
    // id `slot * 2*KVHeads + head` here. Null on every tier with no cold codec and null
    // whenever the pool was not armed, which makes the cold branch below dead code on
    // those stacks and the tile load byte-identical to what it was before.
    //
    // UNLOAD-NORMALQUANT FIX (argument #6): these five parameters sit HERE, straight
    // after cache_v, because that is where BOTH callers put them --
    // gqa_attention_decode_partial.cuh:112 and gqa_attention_decode_impl.cuh:163 read
    // `... cache_v(data), cold_k_slots, cold_v_slots, cold_k_valid, cold_v_valid,
    // cache.slot_bytes, block_tables, ...`. A4 originally declared them AFTER
    // table_stride, so the sixth argument (const uint8_t*) met the sixth parameter
    // (const int32_t* block_tables) and nvcc refused the instantiation with
    // "argument #6 does not match parameter" at decode_partial.cuh(108) -- i.e. the
    // tree did not compile at all, on any target, not just on this stack. The position
    // is also the one the tier this shares its record with uses:
    // gqa_attention_decode_i8.cuh:75-79 puts cold_k_slots/cold_v_slots/slot_bytes after
    // the cache and scale planes and before block_tables.
    const std::uint8_t* cold_k_slots, const std::uint8_t* cold_v_slots,
    const std::int32_t* cold_k_valid, const std::int32_t* cold_v_valid, std::int32_t slot_bytes,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::uint64_t* column_masks, const std::int32_t* table_rows, std::int32_t table_stride,
    std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
    std::int32_t split_units, float scale,
    float* partial_acc, float* partial_m, float* partial_l,
    // bf16win: the declared sliding window, in TOKENS, as the nvfp4/i8 siblings take it
    // (gqa_attention_decode_nvfp4.cuh:234, gqa_attention_decode_i8.cuh). DEFAULTED so that
    // every call site compiled before this patch keeps full attention: 0 means "the field is
    // not read", which is byte-identical to `window = last_pos + 1` below.
    std::int32_t sliding_window = 0) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

    constexpr int Wc      = WarpsPerCta;
    constexpr int Br      = Wc * 16;
    constexpr int Bc      = 32;
    constexpr int D       = Geometry::HeadDim;
    constexpr int Threads = Wc * 32;
    constexpr int QKNt    = Bc / 8;
    constexpr int QKKs    = D / 16;
    constexpr int PVNt    = D / 8;
    constexpr int PVKs    = Bc / 16;
    // The YaRN-extended 1,010,000-key maximum envelope spans at most 186 pages in one 27B split.
    constexpr int PageIds       = paged_kv_page_ids(kCausalAttentionMaximumVisibleKeysYarn);
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
    q += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * Geometry::HeadDim * Geometry::QHeads * tokens *
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
                    0.0f;
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

    // bf16win: THE WINDOW IS READ NOW. Before this line the bf16 decode path answered
    // `last_pos + 1` unconditionally, i.e. full attention, which is why the field's readers
    // were the census {NVFP4, ISO3} and why installing a windowed artifact on bf16 was
    // refused by name (product/kv_component_switch.h:kv_sliding_window_domain_error). The
    // form below is the nvfp4 kernel's (gqa_attention_decode_nvfp4.cuh:381), including its
    // max(0, ...) clamp: token_begin = max(0, window_full - sliding_window).
    const int window_full = last_pos + 1;
    const int token_begin =
        (sliding_window > 0 && window_full > sliding_window) ? window_full - sliding_window : 0;
    const int window = window_full - token_begin;
    // Fixed split grid (split_units > 0): split s owns the keys
    // [s*split_units, min((s+1)*split_units, window)). Its interior boundaries are
    // launch constants, so the partial a split contributes for a key range -- and the fp32
    // addition order it used to build it -- no longer move when the launch covers a
    // different number of tokens. The live window still clips every range (no split
    // addresses a key past the last valid one) and split_units == 0 keeps the legacy
    // window-driven partition.
    //
    // The count and the tiling come from gqa_small_t_split_range, the family's one
    // definition of "which keys does split s own" (ops/kernel/gqa_attention_decode.cuh).
    // With NINFER_VERIFY_EXACT=1 the token tile cannot reach either of them: with a pinned
    // split_units the range is [s*split_units, ...) outright, and the legacy branch derives
    // its tiling from `window` alone because the active count does. At one and the same
    // window, and for one and the same dtype, the tiling and the count are therefore
    // identical for TokenTile == 1 and TokenTile == 6 (item 3 of the fix), and the KV dtype
    // no longer moves the lossless-region bound (item 1 / H39).
    const GqaSmallTSplitRange split_range =
        gqa_small_t_split_range<Geometry, false>(window, split_count, split_units, TokenTile,
                                                  Bc, split, gqa_verify_exact_mode());
    const int active_split_count = split_range.active;
    const int split_start        = split_range.start;
    const int split_limit        = split_range.limit;
    if (split >= active_split_count) { return; }

    const int split_end = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    // Contract: page_count is bounded by PageIds =
    // paged_kv_page_ids(envelope) for every split a launcher can dispatch.
    // Without this check a larger envelope would run off the end of shared
    // memory silently instead of declining the split.
    if (page_count > PageIds) {
        write_neutral();
        return;
    }
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        // The owning split writes each new row. Current attention reads those rows directly from
        // input below, so no split depends on another split's cache write.
        for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
            const int token = chunk / (D / 8);
            const int d     = (chunk - token * (D / 8)) * 8;
            const int p_tok = pos[token];
            if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                p_tok < logical_capacity) {
                const std::int64_t new_off = gqa_kv_new_index<Geometry>(kv_head, d, token);
                const int lane             = tid & 31;
                int physical_page = lane == 0 ? paged_kv_physical_page(block_table, p_tok) : 0;
                physical_page     = __shfl_sync(FullMask, physical_page, 0);
                const std::int64_t cache_off =
                    gqa_cache_index<Geometry>(physical_page, kv_head, d, p_tok & kPagedKVPageMask);
                store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                store_vec(&cache_v[cache_off], load_vec<int4>(&input.v[new_off]));
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
    // BF16-COLD-LAND A4: the block-table ENTRY for the tile, tracked alongside the page
    // it resolves to. A cold page is written into the table as a sentinel (entry <= -2)
    // and `physical_page` IS that sentinel, so the cache-plane loads below must not run
    // for it.
    int page_entry = physical_pages_s[0];
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
            page_entry    = physical_page;
        }
        // BF16-COLD-LAND A4. A Bc=32 tile never crosses a 64-token page boundary, so the
        // entry cached for this tile decides the whole tile's load path. The sentinel
        // semantics, the guard set and the flat head-slot id `slot * 2*KVHeads + head`
        // are the i8 tile body's (gqa_attention_decode_i8.cuh:443-455) and the bf16
        // causal-cache body's (small_t_bf16.cuh:231-238); the second validity plane
        // arrives plane-relative, so K and V use the SAME flat id here.
        const int slot_flat = (-page_entry - 2) * (2 * Geometry::KVHeads) + kv_head;
        const bool cold     = page_entry <= -2 && cold_k_slots != nullptr &&
                          cold_v_slots != nullptr && cold_k_valid != nullptr &&
                          cold_v_valid != nullptr && slot_bytes >= 1024 + 320 &&
                          cold_k_valid[slot_flat] != 0 && cold_v_valid[slot_flat] != 0;
        const std::uint8_t* k_slot =
            cold ? cold_k_slots + static_cast<std::int64_t>(slot_flat) * slot_bytes : nullptr;
        const std::uint8_t* v_slot =
            cold ? cold_v_slots + static_cast<std::int64_t>(slot_flat) * slot_bytes : nullptr;
        // Stage the bf16 K/V key tile with one cp.async wave (16B/thread, high MLP).
        // Current-step tokens come from k_new/v_new; tail slots are zeroed.
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l      = chunk / (D / 8);
            const int d          = (chunk - key_l * (D / 8)) * 8;
            const int key        = k0 + key_l;
            __nv_bfloat16* k_dst = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            __nv_bfloat16* v_dst = &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            if (key >= split_start && key < split_end) {
                if constexpr (CacheInput::writes_cache) {
                    const int new_token = key - first_pos;
                    const bool from_new =
                        new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                    if (from_new) {
                        const std::int64_t off = gqa_kv_new_index<Geometry>(kv_head, d, new_token);
                        ninfer::ops::cp_async<16>(k_dst, &input.k[off]);
                        ninfer::ops::cp_async<16>(v_dst, &input.v[off]);
                    } else if (cold) {
                        // A cold page: decode the key row out of the raw slot instead of
                        // the (freed) cache plane. 128 B of packed E2M1 nibbles and 16 B
                        // of E4M3 group-16 scales per 64-token row, the slot's fixed
                        // geometry, are the same numbers the warm restore writes back.
                        const int row = key & kPagedKVPageMask;
                        const std::uint8_t* k_row = detail::cold_i8_slot_codes(k_slot) + row * 128;
                        const std::uint8_t* k_row_s =
                            detail::cold_i8_slot_scales(k_slot) + row * 16;
                        const std::uint8_t* v_row = detail::cold_i8_slot_codes(v_slot) + row * 128;
                        const std::uint8_t* v_row_s =
                            detail::cold_i8_slot_scales(v_slot) + row * 16;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int chan        = d + i;
                            const std::uint8_t kb = k_row[chan >> 1];
                            const std::uint8_t vb = v_row[chan >> 1];
                            const float k_code =
                                gqa_kv_nvfp4_e2m1_to_f32((chan & 1) ? (kb >> 4) : (kb & 0x0F));
                            const float v_code =
                                gqa_kv_nvfp4_e2m1_to_f32((chan & 1) ? (vb >> 4) : (vb & 0x0F));
                            const float k_scale = gqa_kv_nvfp4_e4m3_to_f32(k_row_s[chan >> 4]);
                            const float v_scale = gqa_kv_nvfp4_e4m3_to_f32(v_row_s[chan >> 4]);
                            k_dst[i]            = __float2bfloat16(k_code * k_scale);
                            v_dst[i]            = __float2bfloat16(v_code * v_scale);
                        }
                    } else {
                        const std::int64_t off = gqa_cache_index<Geometry>(
                            physical_page, kv_head, d, key & kPagedKVPageMask);
                        ninfer::ops::cp_async<16>(k_dst, &cache_k[off]);
                        ninfer::ops::cp_async<16>(v_dst, &cache_v[off]);
                    }
                } else if (cold) {
                    const int row = key & kPagedKVPageMask;
                    const std::uint8_t* k_row   = detail::cold_i8_slot_codes(k_slot) + row * 128;
                    const std::uint8_t* k_row_s = detail::cold_i8_slot_scales(k_slot) + row * 16;
                    const std::uint8_t* v_row   = detail::cold_i8_slot_codes(v_slot) + row * 128;
                    const std::uint8_t* v_row_s = detail::cold_i8_slot_scales(v_slot) + row * 16;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int chan        = d + i;
                        const std::uint8_t kb = k_row[chan >> 1];
                        const std::uint8_t vb = v_row[chan >> 1];
                        const float k_code =
                            gqa_kv_nvfp4_e2m1_to_f32((chan & 1) ? (kb >> 4) : (kb & 0x0F));
                        const float v_code =
                            gqa_kv_nvfp4_e2m1_to_f32((chan & 1) ? (vb >> 4) : (vb & 0x0F));
                        const float k_scale = gqa_kv_nvfp4_e4m3_to_f32(k_row_s[chan >> 4]);
                        const float v_scale = gqa_kv_nvfp4_e4m3_to_f32(v_row_s[chan >> 4]);
                        k_dst[i]            = __float2bfloat16(k_code * k_scale);
                        v_dst[i]            = __float2bfloat16(v_code * v_scale);
                    }
                } else {
                    const std::int64_t off = gqa_cache_index<Geometry>(physical_page, kv_head, d,
                                                                       key & kPagedKVPageMask);
                    ninfer::ops::cp_async<16>(k_dst, &cache_k[off]);
                    ninfer::ops::cp_async<16>(v_dst, &cache_v[off]);
                }
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
                store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
        ninfer::ops::cp_wait<0>();
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
        // M1: the two query rows' ancestor masks. Rows past row_count take column 0's word so the
        // load stays in bounds; their scores are discarded by the `row < row_count` guards below.
        const std::uint64_t mask0 =
            column_masks == nullptr
                ? ~std::uint64_t{0}
                : column_masks[static_cast<std::int64_t>(batch) * full_width + column_begin +
                               (row0 < row_count ? token0 : 0)];
        const std::uint64_t mask1 =
            column_masks == nullptr
                ? ~std::uint64_t{0}
                : column_masks[static_cast<std::int64_t>(batch) * full_width + column_begin +
                               (row1 < row_count ? token1 : 0)];

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = k0 + col0;
            const int key1 = col1 + k0;
            // M1: the same cut, plus each query's ancestor bit set for the columns of this round
            // block. `rel` is the key's column index within the ROUND's whole row (the round's
            // column 0 sits at cache position first_pos - column_begin), so rel < 0 is a key below
            // the round -- the shared, already committed history -- and is always visible, while
            // every key of this round is visible only where the mask says so.
            //
            // The round base (and not this launch's chunk base) is what makes the rule true for a
            // CHUNKED launch: a width above kSmallTChunkTokens is dispatched as several launches
            // with column_begin > 0, and the keys below a chunk are not all history -- the earlier
            // chunks' columns belong to this same round block and a tree's sibling is not an
            // ancestor. With the chunk-relative spelling every earlier chunk's column was admitted
            // unconditionally (rel < 0), which silently over-visible a tree round's branches. For
            // a chain the mask is the prefix (1 << (j+1)) - 1, so every column that rel < 0 used to
            // admit is admitted by the mask too: bit-for-bit unchanged, and for column_begin == 0
            // the two spellings are the same expression.
            const int rel0 = column_begin + key0 - first_pos;
            const int rel1 = column_begin + key1 - first_pos;
            const bool vis0k0 =
                rel0 < 0 || ((mask0 >> static_cast<unsigned>(rel0)) & std::uint64_t{1}) != 0;
            const bool vis0k1 =
                rel1 < 0 || ((mask0 >> static_cast<unsigned>(rel1)) & std::uint64_t{1}) != 0;
            const bool vis1k0 =
                rel0 < 0 || ((mask1 >> static_cast<unsigned>(rel0)) & std::uint64_t{1}) != 0;
            const bool vis1k1 =
                rel1 < 0 || ((mask1 >> static_cast<unsigned>(rel1)) & std::uint64_t{1}) != 0;
            score[nt][0] =
                (row0 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs0 &&
                 vis0k0)
                    ? score[nt][0] * scale
                    : -CUDART_INF_F;
            score[nt][1] =
                (row0 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs0 &&
                 vis0k1)
                    ? score[nt][1] * scale
                    : -CUDART_INF_F;
            score[nt][2] =
                (row1 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs1 &&
                 vis1k0)
                    ? score[nt][2] * scale
                    : -CUDART_INF_F;
            score[nt][3] =
                (row1 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs1 &&
                 vis1k1)
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

    // FP32 partials: each lane already owns two adjacent d values of one row, so the
    // split-local accumulator is written straight from the MMA fragments. Nothing rounds it
    // through bf16 staging any more, and the block barrier that staging needed is gone.
    const int prow0 = warp_row0 + gid;
    const int prow1 = prow0 + 8;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0 = n * 8 + 2 * lid;
        if (prow0 < row_count) {
            int aq_head = 0;
            int atoken  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(prow0, tokens, kv_head, aq_head, atoken);
            if (gqa_valid_q_head<Geometry>(kv_head, aq_head)) {
                const std::int64_t dst =
                    gqa_partial_acc_index<Geometry>(aq_head, d0, atoken, split, tokens);
                store_vec(&partial_acc[dst], make_float2(acc[n][0], acc[n][1]));
            }
        }
        if (prow1 < row_count) {
            int aq_head = 0;
            int atoken  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(prow1, tokens, kv_head, aq_head, atoken);
            if (gqa_valid_q_head<Geometry>(kv_head, aq_head)) {
                const std::int64_t dst =
                    gqa_partial_acc_index<Geometry>(aq_head, d0, atoken, split, tokens);
                store_vec(&partial_acc[dst], make_float2(acc[n][2], acc[n][3]));
            }
        }
    }
}

} // namespace ninfer::ops
