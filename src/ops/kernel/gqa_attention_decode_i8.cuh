#pragma once

// ninfer::ops - split-KV GQA small-T attention, int8 KV-cache partial kernel.
// Historical design: docs/archive/optimization-era/2026-07-08-gqa-decode-int8-kernel-redesign.md
// -- NOT in this tree and no copy reachable on this box; what follows is what survives of it.
//
//   * QK runs on native m16n8k32.s8 tensor cores. Q is quantized on-chip to int8
//     per (row, 64-group); K stays int8 in the cache and is read straight into
//     smem (no dequant). The int32 MMA output is rescaled per 64-group by
//     qs[row,g]*ks[key,g]. This halves the QK MMA count vs bf16 and removes the
//     entire K dequant.
//   * PV stays bf16 (V is quantized per key, so its scale cannot be factored out
//     of a key-contracted int8 accumulation): V int8 is staged, dequanted once to
//     a bf16 tile, then the existing bf16 PV MMA runs. V is still read from DRAM
//     as int8, so the bandwidth win is kept.
//   * All keys (history AND the current/diagonal tokens) are read from the
//     quantized cache; the fused append writes the new tokens first and a
//     __syncthreads orders the in-block readback. No from_new special-casing.
//
// Standalone from the bf16 kernel; shared scaffolding (layout constants, ldmatrix
// helpers, the s8/bf16 MMA helpers, the reducer) lives in gqa_attention_decode.cuh.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/cold_i8_kernels.cuh"
#include "ops/kernel/e8_lattice.cuh"

#include <cstdint>

namespace ninfer::ops {

// Store one int8 code into a d-contiguous-as-b16 swizzled tile so the same
// gqa_small_t_tc_swz / ldmatrix path that serves bf16 tiles serves the int8 tile.
// A b16 lane holds two packed int8 (d even = low byte, d odd = high byte); this
// matches the byte layout a 16 B cp.async of d-contiguous cache bytes produces
// (see the design doc / kernel comments), so Q (byte stores) and K (cp.async)
// agree.

// Stage exactly `Groups` fp16 group scales for one key row: 4 halves (8 B) at
// head_dim 256, 2 halves (4 B) at Muse's 128. S36 made `Groups` geometry-derived;
// the unconditional 8-byte store/copy that used to be equivalent (Groups==4) then
// overlapped the next key's slots and ran 4 B past k_scale_s[Bc*Groups] on the
// last key. Derive the size, never write it out again.
template <int Groups>
__device__ __forceinline__ void gqa_i8_scale_row_clear(__half* row) {
#pragma unroll
    for (int g = 0; g < Groups; ++g) { row[g] = __float2half(0.0f); }
}

__device__ __forceinline__ void gqa_small_t_i8_store_swz(std::int8_t* tile, int row, int d,
                                                         int d_b16_stride, std::int8_t code) {
    const int c   = d >> 1;
    const int lo  = d & 1;
    const int off = (row * d_b16_stride + gqa_small_t_tc_swz(row, c)) * 2 + lo;
    tile[off]     = code;
}

// Decode-specialized producer/consumer kernel for T=1..6. One producer warp per
// m16 row tile computes QK + online softmax, while all CTA warps partition the
// tile's 256-wide PV output. This keeps each thread's PV accumulator at 16, 32,
// or 64 floats instead of 128 and uses otherwise-idle warps for useful output
// work.
//
// Q has a dedicated shared tile so producers can reload one 64-dimension group
// at a time. K/V codes and scales are staged asynchronously; non-producer warps
// dequantize V while producers execute QK. After both consume the code tile, the
// next K/V tile is prefetched into the same arena while the current PV runs.
template <typename Geometry, int TokenTile, int WarpsPerCta, int MinBlocksPerSm, int KeyBlock,
          bool DynamicArena, bool MultiBatch, bool Masked, bool E8, typename CacheInput>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
    void gqa_attention_decode_i8_tiled_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
        std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
        const std::uint8_t* cold_k_slots, const std::uint8_t* cold_v_slots, int slot_bytes,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, std::int32_t split_units,
        float scale, float* partial_acc, float* partial_m, float* partial_l,
        // i8win: the declared sliding window, in TOKENS, exactly as the nvfp4/iso3 siblings
        // take it. DEFAULTED so that every call site compiled before this patch keeps full
        // attention: 0 means "the field is not read", byte-identical to `window = last_pos + 1`.
        std::int32_t sliding_window = 0) {
    constexpr int Wc                   = WarpsPerCta;
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Br                   = RowTiles * 16;
    constexpr int Bc                   = KeyBlock;
    constexpr int D                    = Geometry::HeadDim;
    constexpr int DB16                 = D / 2;
    constexpr int Threads              = Wc * 32;
    constexpr int Groups               = D / kGqaKvQuantGroup;  // geometry-derived: 4 for head_dim 256, 2 for Muse's 128
    constexpr int GroupKc              = kGqaKvQuantGroup / 32;
    constexpr int QKKs                 = D / 32;
    constexpr int QKNt                 = Bc / 8;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs                 = Bc / 16;
    // The YaRN-extended 1,010,000-key maximum envelope spans at most 186 pages in one 27B split.
    constexpr int PageIds         = paged_kv_page_ids(kCausalAttentionMaximumVisibleKeysYarn);
    constexpr int ProducerThreads = RowTiles * 32;
    constexpr int VLoaderThreads  = Threads - ProducerThreads;
    constexpr float Log2E         = 1.4426950408889634074f;
    constexpr unsigned FullMask   = 0xffffffffu;

    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(Bc == 32 || Bc == 64);
    // RowTiles == TokenTile for GroupSize-16 geometries (Muse 32q/2kv); the
    // dispatch kWc table keeps every derived quantity in range up to 6.
    static_assert(RowTiles >= 1 && RowTiles <= 6);
    static_assert(Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 2 || PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);
    static_assert(QKKs == Groups * GroupKc);

    // Keep Q in a compact dedicated tile so the producer can reload one
    // 64-dimension group at a time instead of carrying all eight fragments in
    // registers across the whole kernel. The main arena holds K i8, V i8, and
    // V bf16 during the key loop.
    __shared__ __align__(16) std::int8_t q_s[Br * D];
    __shared__ __align__(16) std::int8_t static_r_s[DynamicArena ? 16 : 4 * Bc * D];
    extern __shared__ __align__(16) std::int8_t dynamic_r_s[];
    std::int8_t* r_s      = DynamicArena ? dynamic_r_s : static_r_s;
    std::int8_t* q_i8     = q_s;
    float* q_scale_tmp    = reinterpret_cast<float*>(r_s);
    std::int8_t* k_i8     = r_s;
    __nv_bfloat16* q_b16  = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16  = reinterpret_cast<__nv_bfloat16*>(k_i8);
    std::int8_t* v_i8     = r_s + Bc * D;
    __nv_bfloat16* v_bf16 = reinterpret_cast<__nv_bfloat16*>(r_s + 2 * Bc * D);
    __shared__ __align__(16) __nv_bfloat16 p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) __half k_scale_s[Bc * Groups];
    __shared__ __align__(16) __half v_scale_s[Bc * Groups];
    __shared__ std::int32_t physical_pages_s[PageIds];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }
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
        partial_acc += static_cast<std::int64_t>(batch) * Geometry::HeadDim * Geometry::QHeads *
                       TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = 0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, TokenTile)] =
                    0.0f;
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    // i8win: THE WINDOW IS READ NOW. Before this line the INT8 decode path answered
    // `last_pos + 1` unconditionally, i.e. full attention, which is why the field's readers
    // were the census {NVFP4, ISO3} and why `--kv-layer-storage 0-15:int8` was refused by
    // name on a windowed artifact (product/kv_component_switch.h, the domain refusal).
    // The form below is the nvfp4 kernel's (gqa_attention_decode_nvfp4.cuh:381), including
    // its max(0, ...) clamp: token_begin = max(0, window_full - sliding_window).
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
        gqa_small_t_split_range<Geometry, true>(window, split_count, split_units, TokenTile,
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
        // The owning split quantizes each current row before its cache tile is consumed.
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            int physical_page       = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kGqaKvQuantGroup + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
            float kv0               = __bfloat162float(input.k[src0]);
            float kv1               = __bfloat162float(input.k[src1]);
            float vv0               = __bfloat162float(input.v[src0]);
            float vv1               = __bfloat162float(input.v[src1]);
            float kamax             = fmaxf(fabsf(kv0), fabsf(kv1));
            float vamax             = fmaxf(fabsf(vv0), fabsf(vv1));
            kamax                   = warp_max(kamax, FullMask);
            vamax                   = warp_max(vamax, FullMask);
            if constexpr (E8) {
                // E8 tier: K codes are 8-dim Hadamard-rotated, packed 4-bit with a
                // g64 scale = amax/8 over codes [-8,7] (AMAXFIX; was amax/7 over
                // [-7,7], measured 0.686 dB worse at identical bits); V is scalar i4
                // with the same budget. Both the divisor and the K clamp range come
                // from gqa_attention_kv_quant.cuh -- do not respell either here.
                // The rotation runs BEFORE the max: a pre-rotation scale clipped the
                // top of the range (_TODO.md 116/116b).
                //
                // The E8 LATTICE PROJECTION is deliberately not applied here.
                // The code plane stores ONE integer per coordinate, but
                // E8 = D8 U (D8 + 1/2): the nearest lattice point of 47.6% of the
                // 8-blocks lies in the coset D8 + 1/2, i.e. all eight of its
                // coordinates are half-integers, and the rint() that has to follow
                // snaps each of them 0.5 step away. Measured over the 27 L13/L14/L15
                // KV forensics dumps (66.2M K elements, e8verify.cpp in
                // scratch/fixE1): projecting first costs 2.32x MSE (+3.65 dB) against
                // plain rounding at identical bits/el -- K relRMS 10.66% without the
                // projection vs 16.24% with it. Buying the 0.58 dB lattice gain
                // instead would need a code plane able to hold 2*coordinate, i.e. 5
                // bits at +-15, for +1 bit/el: strictly worse than spending the same
                // bit on a finer uniform grid. Keep the rotation (it is exactly
                // orthonormal and lowers the group amax: E[amax_rot^2]/E[amax^2] =
                // 0.68), drop the projection.
                // AMAXFIX: the "10.66%" quoted above is the plain-rounding arm AT THE
                // OLD DIVISOR (amax/7, [-7,7]). The shipped rounded arm is now 9.8520%
                // (amax/8, [-8,7]), so the 16.24% projection arm is 3.65 dB -> 4.34 dB
                // WORSE than plain rounding. The conclusion is unchanged and stronger.
                gqa_kv_hadamard64(kv0, kv1, FullMask);
                float kamax_e = fmaxf(fabsf(kv0), fabsf(kv1));
                kamax_e       = warp_max(kamax_e, FullMask);
                const __half ksh =
                    kv_scale_half(kamax_e > 0.0f ? kamax_e / kGqaKvI4ScaleDivisor : 0.0f);
                const __half vsh =
                    kv_scale_half(vamax > 0.0f ? vamax / kGqaKvI4ScaleDivisor : 0.0f);
                const float ks   = __half2float(ksh);
                const float vs   = __half2float(vsh);
                const float k_inv = ks > 0.0f ? 1.0f / ks : 0.0f;
                const float v_inv = vs > 0.0f ? 1.0f / vs : 0.0f;
                const int c0 = static_cast<int>(gqa_kv_quant_i4_code(kv0, k_inv));
                const int c1 = static_cast<int>(gqa_kv_quant_i4_code(kv1, k_inv));
                const int vc0 = static_cast<int>(gqa_kv_quant_i4_code(vv0, v_inv));
                const int vc1 = static_cast<int>(gqa_kv_quant_i4_code(vv1, v_inv));
                physical_page = __shfl_sync(FullMask, physical_page, 0);
                // Adjacent dims share a byte; even lanes pack the pair.
                if ((lane & 1) == 0) {
                    const float k0n_nb = __shfl_xor_sync(FullMask, kv0, 1);
                    const float k1n_nb = __shfl_xor_sync(FullMask, kv1, 1);
                    const int c0n = static_cast<int>(gqa_kv_quant_i4_code(k0n_nb, k_inv));
                    const int c1n = static_cast<int>(gqa_kv_quant_i4_code(k1n_nb, k_inv));
                    const int vc0n = static_cast<int>(__shfl_xor_sync(FullMask, vc0, 1));
                    const int vc1n = static_cast<int>(__shfl_xor_sync(FullMask, vc1, 1));
                    const std::int64_t kb0 = gqa_kv_i4_code_index<Geometry>(
                        physical_page, kv_head, d0 >> 1, page_offset);
                    cache_k_i8[kb0] = gqa_kv_pack_i4(static_cast<std::int8_t>(c0),
                                                     static_cast<std::int8_t>(c0n));
                    const std::int64_t kb1 = gqa_kv_i4_code_index<Geometry>(
                        physical_page, kv_head, d1 >> 1, page_offset);
                    cache_k_i8[kb1] = gqa_kv_pack_i4(static_cast<std::int8_t>(c1),
                                                     static_cast<std::int8_t>(c1n));
                    const std::int64_t vb0 = gqa_kv_i4_code_index<Geometry>(
                        physical_page, kv_head, d0 >> 1, page_offset);
                    cache_v_i8[vb0] = gqa_kv_pack_i4(static_cast<std::int8_t>(vc0),
                                                     static_cast<std::int8_t>(vc0n));
                    const std::int64_t vb1 = gqa_kv_i4_code_index<Geometry>(
                        physical_page, kv_head, d1 >> 1, page_offset);
                    cache_v_i8[vb1] = gqa_kv_pack_i4(static_cast<std::int8_t>(vc1),
                                                     static_cast<std::int8_t>(vc1n));
                }
                if (lane == 0) {
                    const std::int64_t so =
                        gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, grp, page_offset);
                    cache_k_scale[so] = ksh;
                    cache_v_scale[so] = vsh;
                }
            } else {
                const __half ksh = __float2half_rn(kamax > 0.0f ? kamax / 127.0f : 0.0f);
                const __half vsh = __float2half_rn(vamax > 0.0f ? vamax / 127.0f : 0.0f);
                const float ks   = __half2float(ksh);
                const float vs   = __half2float(vsh);
                const float k_inv = ks > 0.0f ? 1.0f / ks : 0.0f;
                const float v_inv = vs > 0.0f ? 1.0f / vs : 0.0f;
                physical_page    = __shfl_sync(FullMask, physical_page, 0);
                cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                              page_offset)] =
                    gqa_kv_quant_code(kv0, k_inv);
                cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                              page_offset)] =
                    gqa_kv_quant_code(kv1, k_inv);
                cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                              page_offset)] =
                    gqa_kv_quant_code(vv0, v_inv);
                cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                              page_offset)] =
                    gqa_kv_quant_code(vv1, v_inv);
                if (lane == 0) {
                    const std::int64_t so =
                        gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, grp,
                                                           page_offset);
                    cache_k_scale[so] = ksh;
                    cache_v_scale[so] = vsh;
                }
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < Br * D; i += Threads) { q_i8[i] = 0; }
    for (int i = tid; i < RowCount * Groups; i += Threads) { q_scale_tmp[i] = 0.0f; }
    __syncthreads();

    for (int unit = warp; unit < RowCount * Groups; unit += Wc) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        const int d0  = grp * kGqaKvQuantGroup + lane;
        const int d1  = d0 + 32;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        float x0        = __bfloat162float(q[gqa_q_index<Geometry>(q_head, d0, token)]);
        float x1        = __bfloat162float(q[gqa_q_index<Geometry>(q_head, d1, token)]);
        if constexpr (E8) { gqa_kv_hadamard64(x0, x1, FullMask); }
        float amax      = fmaxf(fabsf(x0), fabsf(x1));
        amax            = warp_max(amax, FullMask);
        const float qs  = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
        gqa_small_t_i8_store_swz(q_i8, row, d0, DB16, gqa_kv_quant_code(x0, inv));
        gqa_small_t_i8_store_swz(q_i8, row, d1, DB16, gqa_kv_quant_code(x1, inv));
        if (lane == 0) { q_scale_tmp[row * Groups + grp] = qs; }
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

    float q_scale_r0[Groups];
    float q_scale_r1[Groups];
    if (warp < RowTiles) {
        const int producer_row0 = warp * 16 + gid;
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            float qs0     = (lid == 0 && producer_row0 < RowCount)
                                ? q_scale_tmp[producer_row0 * Groups + g]
                                : 0.0f;
            float qs1     = (lid == 0 && producer_row0 + 8 < RowCount)
                                ? q_scale_tmp[(producer_row0 + 8) * Groups + g]
                                : 0.0f;
            q_scale_r0[g] = __shfl_sync(FullMask, qs0, gid * 4);
            q_scale_r1[g] = __shfl_sync(FullMask, qs1, gid * 4);
        }
    }
    __syncthreads();

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    auto issue_kv_tile = [&](int tile_k0, int physical_page) {
        // Revision 2b cold staging: raw nibble slots hold g64-requantized
        // E2M1 codes + E4M3 g16 scales; decode each key row straight into the
        // kernel's native int8 codes + fp16 group scales so the QK tensor-core
        // path runs unchanged. Slot addressing is region-relative: flat index
        // slot * 2*KVHeads + head for both the K and V slot regions.
        if (physical_page <= -2 && cold_k_slots != nullptr &&
            slot_bytes >= ninfer::ops::kColdI8SlotBytes) {
            const std::int64_t slot_flat =
                static_cast<std::int64_t>(-physical_page - 2) * (2 * Geometry::KVHeads) +
                kv_head;
            const std::uint8_t* k_slot =
                cold_k_slots + slot_flat * slot_bytes;
            const std::uint8_t* v_slot =
                cold_v_slots == nullptr ? nullptr
                                         : cold_v_slots + slot_flat * slot_bytes;
            if constexpr (E8) {
                // BF16-COLD-LAND E4. The e8 tier's record is the SAME raw record, but its
                // code plane holds the tier's packed 4-bit codes VERBATIM (the requant's
                // E8KvG64 arm) rather than a requantized E2M1 code, so this stages exactly
                // what the hot path stages -- 8 bytes of codes per 16 dims into the packed
                // smem position, four g64 scales per row -- and lets the existing E8
                // unpack pass expand it. Nothing about the QK/PV math changes, so a cold
                // key row is decoded by the same instructions as a hot one.
                //
                // The g64 scale is the E4M3 byte the requant replicated into its four g16
                // slots, so g reads tail byte 4*g.
                for (int key_l = tid; key_l < Bc; key_l += Threads) {
                    const int key = tile_k0 + key_l;
                    if (key >= split_start && key < split_end && v_slot != nullptr) {
                        const int row = key & kPagedKVPageMask;
                        const std::uint8_t* k_row_s =
                            ninfer::ops::detail::cold_i8_slot_scales(k_slot) + row * 16;
                        const std::uint8_t* v_row_s =
                            ninfer::ops::detail::cold_i8_slot_scales(v_slot) + row * 16;
#pragma unroll
                        for (int g = 0; g < Groups; ++g) {
                            k_scale_s[key_l * Groups + g] = __float2half_rn(
                                gqa_kv_nvfp4_e4m3_to_f32(k_row_s[g * 4]));
                            v_scale_s[key_l * Groups + g] = __float2half_rn(
                                gqa_kv_nvfp4_e4m3_to_f32(v_row_s[g * 4]));
                        }
                    } else {
                        gqa_i8_scale_row_clear<Groups>(&k_scale_s[key_l * Groups]);
                        gqa_i8_scale_row_clear<Groups>(&v_scale_s[key_l * Groups]);
                    }
                }
#pragma unroll 1
                for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
                    const int key_l = chunk / (D / 16);
                    const int dc    = chunk - key_l * (D / 16);
                    const int key   = tile_k0 + key_l;
                    std::int8_t* kdst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                    std::int8_t* vdst = &v_i8[key_l * D + dc * 16];
                    if (key >= split_start && key < split_end && v_slot != nullptr) {
                        const int row = key & kPagedKVPageMask;
                        ninfer::ops::cp_async<8>(
                            kdst,
                            ninfer::ops::detail::cold_i8_slot_codes(k_slot) + row * 128 + dc * 8);
                        ninfer::ops::cp_async<8>(
                            vdst,
                            ninfer::ops::detail::cold_i8_slot_codes(v_slot) + row * 128 + dc * 8);
                    } else {
                        store_vec(kdst, make_int4(0, 0, 0, 0));
                        store_vec(vdst, make_int4(0, 0, 0, 0));
                    }
                }
                ninfer::ops::cp_commit();
                return;
            }
            for (int key_l = tid; key_l < Bc; key_l += Threads) {
                const int key = tile_k0 + key_l;
                if (key >= split_start && key < split_end && v_slot != nullptr) {
                    const int row = key & kPagedKVPageMask;
                    std::int8_t row_codes[Geometry::HeadDim];
                    __half row_scales[Geometry::HeadDim / kGqaKvQuantGroup];
                    ninfer::ops::detail::cold_i8_decode_row<Groups>(k_slot, row, row_codes,
                                                                           row_scales);
#pragma unroll 8
                    for (int d = 0; d < Geometry::HeadDim; ++d) {
                        ninfer::ops::gqa_small_t_i8_store_swz(k_i8, key_l, d,
                                                              Geometry::HeadDim / 2, row_codes[d]);
                    }
#pragma unroll
                    for (int g = 0; g < Groups; ++g) {
                        k_scale_s[key_l * Groups + g] = row_scales[g];
                    }
                    ninfer::ops::detail::cold_i8_decode_row<Groups>(v_slot, row, row_codes,
                                                                           row_scales);
#pragma unroll 8
                    for (int d = 0; d < Geometry::HeadDim; ++d) {
                        v_i8[key_l * Geometry::HeadDim + d] = row_codes[d];
                    }
#pragma unroll
                    for (int g = 0; g < Groups; ++g) {
                        v_scale_s[key_l * Groups + g] = row_scales[g];
                    }
                } else {
#pragma unroll 1
                    for (int dc = 0; dc < Geometry::HeadDim / 16; ++dc) {
                        std::int8_t* dst = &k_i8[key_l * Geometry::HeadDim +
                                                gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                        ninfer::ops::store_vec(dst, make_int4(0, 0, 0, 0));
                        ninfer::ops::store_vec(&v_i8[key_l * Geometry::HeadDim + dc * 16],
                                               make_int4(0, 0, 0, 0));
                    }
                    gqa_i8_scale_row_clear<Groups>(&k_scale_s[key_l * Groups]);
                    gqa_i8_scale_row_clear<Groups>(&v_scale_s[key_l * Groups]);
                }
            }
            ninfer::ops::cp_commit();
            return;
        }
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                const std::int64_t off = gqa_kv_quant_scale_index<Geometry>(
                    physical_page, kv_head, 0, key & kPagedKVPageMask);
                ninfer::ops::cp_async<Groups * static_cast<int>(sizeof(__half))>(
                    &k_scale_s[key_l * Groups], &cache_k_scale[off]);
                ninfer::ops::cp_async<Groups * static_cast<int>(sizeof(__half))>(
                    &v_scale_s[key_l * Groups], &cache_v_scale[off]);
            } else {
                gqa_i8_scale_row_clear<Groups>(&k_scale_s[key_l * Groups]);
                gqa_i8_scale_row_clear<Groups>(&v_scale_s[key_l * Groups]);
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                if constexpr (E8) {
                    // Packed 4-bit tier: 8 bytes of codes per 16 dims; the
                    // unpack pass after cp_wait expands them into i8 in place.
                    const std::int64_t off = gqa_kv_i4_code_index<Geometry>(
                        physical_page, kv_head, d / 2, key & kPagedKVPageMask);
                    std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                    ninfer::ops::cp_async<8>(dst, &cache_k_i8[off]);
                    ninfer::ops::cp_async<8>(&v_i8[key_l * D + d], &cache_v_i8[off]);
                } else {
                    const std::int64_t off = gqa_kv_quant_code_index<Geometry>(
                        physical_page, kv_head, d, key & kPagedKVPageMask);
                    std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                    ninfer::ops::cp_async<16>(dst, &cache_k_i8[off]);
                    ninfer::ops::cp_async<16>(&v_i8[key_l * D + d], &cache_v_i8[off]);
                }
            } else {
                std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                // Full slot clear: E8's unpack pass skips out-of-range rows,
                // so the whole 16-byte tile must already read as zeros.
                store_vec(dst, make_int4(0, 0, 0, 0));
                store_vec(&v_i8[key_l * D + d], make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    int physical_page = physical_pages_s[0];
    auto unpack_tile = [&](int tile_k0) {
        // Expand packed 4-bit K/V codes to i8 in the smem tiles (16 dims per
        // 8-byte chunk; the packed bytes are read into registers first, so the
        // in-place writes never clobber an unread source byte).
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                std::int8_t* kdst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                std::int8_t* vdst = &v_i8[key_l * D + d];
                const int2 kpacked = load_vec<int2>(kdst);
                const int2 vpacked = load_vec<int2>(vdst);
                std::int8_t kout[kGqaKvUnpackWidth];
                std::int8_t vout[kGqaKvUnpackWidth];
                gqa_kv_unpack_i4x16(reinterpret_cast<const std::uint8_t*>(&kpacked), kout);
                gqa_kv_unpack_i4x16(reinterpret_cast<const std::uint8_t*>(&vpacked), vout);
                store_vec(kdst, gqa_kv_pack_i8x16_to_int4(kout));
                store_vec(vdst, gqa_kv_pack_i8x16_to_int4(vout));
            }
        }
        __syncthreads();
    };
    issue_kv_tile(first_tile, physical_page);
    ninfer::ops::cp_wait<0>();
    __syncthreads();
    if constexpr (E8) { unpack_tile(first_tile); }

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

        // One warp per row tile produces P and alpha while the remaining warps
        // stream/dequant V.
        if (warp < RowTiles) {
            const int producer_row_base = warp * 16;
            __nv_bfloat16* p_sw         = &p_s[producer_row_base * Bc];
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = 0.0f;
                score[nt][1] = 0.0f;
                score[nt][2] = 0.0f;
                score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = g * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(
                        af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                        smem_addr(&q_b16[(producer_row_base + a_rowoff) * DB16 +
                                         gqa_small_t_tc_swz(producer_row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = g * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(
                            bf[0], bf[1],
                            smem_addr(&k_b16[brow * DB16 + gqa_small_t_tc_swz(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya = nt * 8 + 2 * lid;
                    const int keyb = keya + 1;
                    float ka       = 0.0f;
                    float kb2      = 0.0f;
                    if (gid == 0) {
                        ka  = __half2float(k_scale_s[keya * Groups + g]);
                        kb2 = __half2float(k_scale_s[keyb * Groups + g]);
                    }
                    ka  = __shfl_sync(FullMask, ka, lid);
                    kb2 = __shfl_sync(FullMask, kb2, lid);
                    score[nt][0] += q_scale_r0[g] * ka * static_cast<float>(c0);
                    score[nt][1] += q_scale_r0[g] * kb2 * static_cast<float>(c1);
                    score[nt][2] += q_scale_r1[g] * ka * static_cast<float>(c2);
                    score[nt][3] += q_scale_r1[g] * kb2 * static_cast<float>(c3);
                }
            }

            const int row0 = producer_row_base + gid;
            const int row1 = row0 + 8;
            int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head0, token0);
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head1, token1);
            const int qabs0 = (row0 < RowCount) ? pos[token0] : -1;
            const int qabs1 = (row1 < RowCount) ? pos[token1] : -1;
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = k0 + col0;
                const int key1 = k0 + col1;
                score[nt][0] =
                    (row0 < RowCount && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                        ? score[nt][0] * scale
                        : -CUDART_INF_F;
                score[nt][1] =
                    (row0 < RowCount && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                        ? score[nt][1] * scale
                        : -CUDART_INF_F;
                score[nt][2] =
                    (row1 < RowCount && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                        ? score[nt][2] * scale
                        : -CUDART_INF_F;
                score[nt][3] =
                    (row1 < RowCount && key1 >= split_start && key1 < split_end && key1 <= qabs1)
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
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else {
            const int loader_tid = tid - ProducerThreads;
#pragma unroll 1
            for (int chunk = loader_tid; chunk < Bc * (D / 8); chunk += VLoaderThreads) {
                const int key_l    = chunk / (D / 8);
                const int dc       = chunk - key_l * (D / 8);
                const int d        = dc * 8;
                const int key      = k0 + key_l;
                __nv_bfloat16* dst = &v_bf16[key_l * D + gqa_small_t_tc_swz(key_l, d)];
                if (key >= split_start && key < split_end) {
                    const int grp = d >> 6;
                    float vs      = 0.0f;
                    if ((lane & 7) == 0) { vs = __half2float(v_scale_s[key_l * Groups + grp]); }
                    vs = __shfl_sync(FullMask, vs, grp * 8);
                    store_vec(dst, gqa_kv_dequant_i8x8_from(&v_i8[key_l * D + d], vs));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            const int next_k0 = k0 + Bc;
            if ((next_k0 & kPagedKVPageMask) == 0) {
                physical_page = physical_pages_s[(next_k0 >> kPagedKVPageShift) - first_page];
            }
            issue_kv_tile(next_k0, physical_page);
        }

        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        __nv_bfloat16* p_consumer   = &p_s[consumer_row_base * Bc];
        const float alpha0          = alpha_s[consumer_row_base + gid];
        const float alpha1          = alpha_s[consumer_row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = consumer_slice * PVNtPerWarp + n;
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    smem_addr(&p_consumer[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_bf16[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
        if (has_next) {
            ninfer::ops::cp_wait<0>();
            if constexpr (E8) { unpack_tile(k0 + Bc); }
        }
        __syncthreads();
    }

    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l1;
        }
    }

#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        const int d0                = (consumer_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        const int row0              = consumer_row_base + gid;
        const int row1              = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(acc[n][0], acc[n][1]);
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(acc[n][2], acc[n][3]);
        }
    }
}

} // namespace ninfer::ops
