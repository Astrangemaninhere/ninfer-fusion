#pragma once

// ninfer::ops - split-KV GQA small-T attention, NVFP4 KV-cache partial kernel.
//
//   * QK runs on native m16n8k64.kind::mxf4nvf4 tensor cores. Q is quantized
//     on-chip to packed E2M1 with per-(row,16-group) E4M3 scales; K stays
//     packed in the cache and is staged straight into shared memory. The
//     hardware block-scale instruction applies both scale vectors, so the
//     QK result is already in the scaled domain.
//   * K cache writes apply the baked IsoQuant per-4-channel rotation, and Q
//     is rotated with the same matrix before quantization, preserving QK^T.
//   * PV runs on native m16n8k64.kind::mxf4nvf4 tensor cores too: P is
//     folded with the V E4M3 scale, re-quantized per (row,16-d-group) to
//     E2M1, and the packed V codes are re-tiled in shared memory. The MMA
//     applies the folded P scale vector against a constant 1.0 scale
//     vector, so the accumulated result is in the scaled domain.
//   * All keys (history AND current diagonal tokens) are read from the
//     quantized cache; the fused append writes new tokens first and a
//     __syncthreads orders the in-block readback.
//
// This kernel is the SM120 native successor to the vLLM-side nvfp4rtx
// nvfp4-mma-v15 kernel, retargeted to NInfer's paged KV layout.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_isoquant_rot.cuh"
#include "ops/kernel/gqa_isoquant_row_scale.cuh"
#include "ops/kernel/entropy_nvfp4_slot.cuh"
#include "ops/kernel/gqa_attention_prefill_nvfp4.cuh" // gqa_iso3_nibble / gqa_iso3_decode

#include <cstdint>

namespace ninfer::ops {
namespace {

using namespace ninfer::ops::detail;

constexpr float kNvfp4MinScale = 0.001953125f;                    // 2^-9, E4M3 smallest normal
constexpr std::uint8_t kNvfp4E4M3One = 0x38u;                     // E4M3FN encoding of 1.0

__device__ __forceinline__ void gqa_nvfp4_load_a_frag(unsigned (&frag)[4], const std::uint8_t* smem,
                                                      int lane, int k_step) {
    const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int col = (lane >> 4) * 16 + k_step * 32;
    ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], smem_addr(smem + row * 128 + col));
}

__device__ __forceinline__ void gqa_nvfp4_load_b_frag(unsigned (&frag)[2], const std::uint8_t* smem,
                                                      int lane, int n_tile, int k_step) {
    const int row = (lane & 7) + n_tile * 8;
    const int col = ((lane >> 3) & 1) * 16 + k_step * 32;
    ldmatrix_x2(frag[0], frag[1], smem_addr(smem + row * 128 + col));
}

// PV repack tiles use a 64-byte row stride (16-byte k-fragment per mxf4nvf4
// m16n8k64 operand row).
__device__ __forceinline__ void gqa_nvfp4_load_a_frag_64(unsigned (&frag)[4],
                                                         const std::uint8_t* smem, int lane) {
    const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int col = (lane >> 4) * 16;
    ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], smem_addr(smem + row * 64 + col));
}

__device__ __forceinline__ void gqa_nvfp4_load_b_frag_64(unsigned (&frag)[2],
                                                         const std::uint8_t* smem, int lane,
                                                         int n_tile) {
    const int row = (lane & 7) + n_tile * 8;
    const int col = ((lane >> 3) & 1) * 16;
    ldmatrix_x2(frag[0], frag[1], smem_addr(smem + row * 64 + col));
}

__device__ __forceinline__ float gqa_nvfp4_rotated(float x0, float x1, float x2, float x3,
                                                   int block, int row) {
    return gqa_isoquant_rot_value(block, row, 0) * x0 +
           gqa_isoquant_rot_value(block, row, 1) * x1 +
           gqa_isoquant_rot_value(block, row, 2) * x2 +
           gqa_isoquant_rot_value(block, row, 3) * x3;
}

// Lane l < 4 loads the four values of its 4-channel block, applies the baked
// SO(4) rotation, and returns the rotated block in x[]. The caller's src
// pointer ALREADY points at the 16-dimension group start.
__device__ __forceinline__ void gqa_nvfp4_load_rotate_4(float (&x)[4], const __nv_bfloat16* src,
                                                       int group, int lane) {
    if (lane < 4) {
        const int block = group * 4 + lane;
        const int base  = lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) { x[j] = __bfloat162float(src[base + j]); }
        const float y0 = gqa_nvfp4_rotated(x[0], x[1], x[2], x[3], block, 0);
        const float y1 = gqa_nvfp4_rotated(x[0], x[1], x[2], x[3], block, 1);
        const float y2 = gqa_nvfp4_rotated(x[0], x[1], x[2], x[3], block, 2);
        const float y3 = gqa_nvfp4_rotated(x[0], x[1], x[2], x[3], block, 3);
        x[0] = y0;
        x[1] = y1;
        x[2] = y2;
        x[3] = y3;
    } else {
        x[0] = x[1] = x[2] = x[3] = 0.0f;
    }
}

// Reduction over the active lanes of one 4-channel rotated block (lanes 0..3).
__device__ __forceinline__ float gqa_nvfp4_group_max4(float local_max, unsigned full_mask) {
    local_max = fmaxf(local_max, __shfl_xor_sync(full_mask, local_max, 1));
    local_max = fmaxf(local_max, __shfl_xor_sync(full_mask, local_max, 2));
    return local_max;
}

__device__ __forceinline__ float gqa_nvfp4_group_max16(float local_max, unsigned full_mask) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(full_mask, local_max, off));
    }
    return local_max;
}

} // namespace

template <typename Geometry, int TokenTile, int WarpsPerCta, int MinBlocksPerSm, int KeyBlock,
          bool DynamicArena, bool Iso3V = false>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
    void gqa_attention_decode_nvfp4_tiled_kernel(
        const __nv_bfloat16* q, const __nv_bfloat16* input_k, const __nv_bfloat16* input_v,
        const std::int32_t* pos, std::uint8_t* cache_k,
        std::uint8_t* cache_v, std::uint8_t* cache_k_scale, std::uint8_t* cache_v_scale,
        std::uint8_t* cache_k_residual, std::uint8_t* cache_k_residual_scale,
        std::uint8_t* cache_v_residual, std::uint8_t* cache_v_residual_scale,
        const std::uint8_t* cold_k_slots, const std::uint8_t* cold_v_slots,
        const std::int32_t* cold_k_valid, const std::int32_t* cold_v_valid,
        int slot_bytes, int sliding_window,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, int layer, float scale,
        __nv_bfloat16* partial_acc, float* partial_m, float* partial_l,
        std::int32_t batch_size, bool masked, bool writes_cache) {
    constexpr int Wc      = WarpsPerCta;
    constexpr int RowCount = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles = (RowCount + 15) / 16;
    constexpr int Br       = RowTiles * 16;
    constexpr int Bc       = KeyBlock;
    constexpr int D        = kGqaHeadDim;
    constexpr int Threads  = Wc * 32;
    constexpr int Groups   = kGqaKvNvfp4Groups;
    constexpr int QKKs     = D / 64;
    constexpr int QKNt     = Bc / 8;
    constexpr int PVKs     = Bc / 16;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp = D / (ConsumerWarpsPerTile * 8);
    constexpr int DgCount     = PVNtPerWarp / 2;
    constexpr int PageIds     = 256;
    constexpr float Log2E         = 1.4426950408889634074f;
    constexpr unsigned FullMask   = 0xffffffffu;
    constexpr unsigned kOnesScale = 0x38383838u;

    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(Bc == 32);
    static_assert(RowTiles >= 1 && RowTiles <= 3);
    static_assert(Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 2 || PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);
    static_assert(PVNtPerWarp >= 2 && (PVNtPerWarp % 2) == 0);
    static_assert(QKKs == 4);

    // Shared arena:
    //   k_pk/v_pk + k_sf/v_sf: two ping-pong Bc-token tiles, so the next tile
    //     is prefetched while the current tile still runs native PV
    //   psc_s      Br*64  P*V fold scales (4 bytes per row/dg)
    //   repack_a/b Wc*1024 per-warp P/V operand repack tiles
    constexpr int kTileBytes = 4 * Bc * 128 + 4 * Bc * 16;
    __shared__ __align__(16) std::uint8_t q_a[Br * 128];
    __shared__ __align__(16) std::uint8_t q_sf[Br * 16];
    __shared__ __align__(16) std::uint8_t static_r_s[DynamicArena
                                                        ? 16
                                                        : 2 * kTileBytes + Br * 16 * 4 +
                                                              2 * Wc * 16 * 64];
    extern __shared__ __align__(16) std::uint8_t nvfp4_dynamic_r_s[];
    std::uint8_t* r_s      = DynamicArena ? nvfp4_dynamic_r_s : static_r_s;
    std::uint8_t* psc_s    = r_s + 2 * kTileBytes;
    std::uint8_t* repack_a = psc_s + Br * 16 * 4;
    std::uint8_t* repack_b = repack_a + Wc * 16 * 64;
    __shared__ __align__(16) __nv_bfloat16 p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ std::int32_t physical_pages_s[PageIds];
    // Hybrid V path decodes the packed ISO3 V tile into BF16 with the exact
    // full-D tc swizzle the ldmatrix PV path expects.
    constexpr int kRBytes = 2 * kTileBytes + Br * 16 * 4 + 2 * Wc * 16 * 64;
    __nv_bfloat16* v_bf16 =
        reinterpret_cast<__nv_bfloat16*>(DynamicArena ? nvfp4_dynamic_r_s + kRBytes
                                                      : nvfp4_dynamic_r_s);

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = static_cast<int>(blockIdx.z);
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if (masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }
    std::int64_t column_base = column_begin + static_cast<std::int64_t>(batch) * full_width;
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if (writes_cache) {
        input_k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input_v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads *
                   TokenTile * split_count;
    partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;

#define NINFER_NVFP4_WRITE_NEUTRAL()                                                          \
    do {                                                                                       \
        for (int row = tid; row < RowCount; row += Threads) {                                  \
            int q_head = 0;                                                                    \
            int token  = 0;                                                                    \
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);        \
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {                                 \
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = \
                    -CUDART_INF_F;                                                             \
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = \
                    0.0f;                                                                      \
            }                                                                                  \
        }                                                                                      \
        for (int idx = tid; idx < RowCount * D; idx += Threads) {                              \
            const int row = idx / D;                                                           \
            const int d   = idx - row * D;                                                     \
            int q_head    = 0;                                                                 \
            int token     = 0;                                                                 \
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);        \
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {                                 \
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split,           \
                                                            TokenTile)] =                     \
                    __float2bfloat16(0.0f);                                                    \
            }                                                                                  \
        }                                                                                      \
    } while (0)

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        NINFER_NVFP4_WRITE_NEUTRAL();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        NINFER_NVFP4_WRITE_NEUTRAL();
        return;
    }

    const int window_full   = last_pos + 1;
    // Slide in 32-key tiles: a staging tile must not straddle a 64-key page
    // boundary, so the ring window start is rounded up to the Bc grid.
    const int token_begin   = (sliding_window > 0) ? window_full - sliding_window : 0;
    const int window_begin =
        (sliding_window > 0) ? ((max(0, token_begin) + Bc - 1) / Bc) * Bc : 0;
    const int window       = window_full - window_begin;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, true>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        NINFER_NVFP4_WRITE_NEUTRAL();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_global_page = (window_begin + first_tile) >> kPagedKVPageShift;
    const int page_count =
        ((window_begin + split_end - 1) >> kPagedKVPageShift) - first_global_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_global_page + page];
    }

    // ---- fused cache append: quantize current K/V rows into the NVFP4 planes ----
    if (writes_cache) {
_Pragma("unroll 1")
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position - window_begin < split_start ||
                position - window_begin >= split_end) {
                continue;
            }
            int physical_page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
            physical_page     = __shfl_sync(FullMask, physical_page, 0);
            const int page_offset = position & kPagedKVPageMask;
            const int src0        = gqa_kv_nvfp4_src_index<Geometry>(kv_head, grp * 16, token);

            // K: rotate per 4-channel block with the baked IsoQuant matrix and
            // apply the Sinkhorn-constrained row scale before E4M3/E2M1 packing.
            float kx[4];
            gqa_nvfp4_load_rotate_4(kx, input_k + src0, grp, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                kx[j] *= gqa_kv_row_scale(layer, kv_head, grp * 16 + lane * 4 + j);
            }
            float kmax = fmaxf(fmaxf(fabsf(kx[0]), fabsf(kx[1])),
                               fmaxf(fabsf(kx[2]), fabsf(kx[3])));
            kmax       = gqa_nvfp4_group_max4(kmax, FullMask);
            const float kscale = fmaxf(kmax / 6.0f, kNvfp4MinScale);
            const std::int64_t kcode =
                gqa_kv_nvfp4_code_index<Geometry>(physical_page, kv_head, grp * 16, page_offset);
            if (lane < 4) {
                cache_k[kcode + 2 * lane] =
                    static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(kx[0] / kscale) |
                                              (gqa_kv_nvfp4_e2m1_nibble(kx[1] / kscale) << 4));
                cache_k[kcode + 2 * lane + 1] =
                    static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(kx[2] / kscale) |
                                              (gqa_kv_nvfp4_e2m1_nibble(kx[3] / kscale) << 4));
            }
            if (lane == 0) {
                cache_k_scale[gqa_kv_nvfp4_scale_index<Geometry>(physical_page, kv_head, grp,
                                                                 page_offset)] =
                    gqa_kv_nvfp4_fp32_to_e4m3(kscale);
            }
            // K residual: second E2M1 stage over the first-stage error.
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
                rmax       = gqa_nvfp4_group_max4(rmax, FullMask);
                const float rscale = fmaxf(rmax / 6.0f, kNvfp4MinScale);
                if (lane < 4) {
                    const std::int64_t rcode =
                        gqa_kv_nvfp4_code_index<Geometry>(physical_page, kv_head, grp * 16,
                                                          page_offset);
                    cache_k_residual[rcode + 2 * lane] =
                        static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(res[0] / rscale) |
                                                  (gqa_kv_nvfp4_e2m1_nibble(res[1] / rscale) << 4));
                    cache_k_residual[rcode + 2 * lane + 1] =
                        static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(res[2] / rscale) |
                                                  (gqa_kv_nvfp4_e2m1_nibble(res[3] / rscale) << 4));
                }
                if (lane == 0) {
                    cache_k_residual_scale[gqa_kv_nvfp4_scale_index<Geometry>(
                        physical_page, kv_head, grp, page_offset)] =
                        gqa_kv_nvfp4_fp32_to_e4m3(rscale);
                }
            }
            // V: no rotation, only per-16 gain quantization.
            const float v0 = lane < 16 ? __bfloat162float(input_v[src0 + lane]) : 0.0f;
            float vmax     = fabsf(v0);
            vmax           = gqa_nvfp4_group_max16(vmax, FullMask);
            const std::int64_t vcode =
                gqa_kv_nvfp4_code_index<Geometry>(physical_page, kv_head, grp * 16, page_offset);
            if constexpr (Iso3V) {
                const float vscale = fmaxf(vmax / 7.0f, 0.001953125f);
                if (lane < 8) {
                    cache_v[vcode + lane] =
                        static_cast<std::uint8_t>(gqa_iso3_nibble(v0, vscale) |
                                                  (gqa_iso3_nibble(
                                                       __bfloat162float(
                                                           input_v[src0 + lane * 2 + 1]),
                                                       vscale)
                                                   << 4));
                }
                if (lane == 0) {
                    cache_v_scale[gqa_kv_nvfp4_scale_index<Geometry>(
                        physical_page, kv_head, grp, page_offset)] =
                        gqa_kv_nvfp4_fp32_to_e4m3(vscale);
                }
                // V residual: second ISO3 stage over the first-stage error.
                if (cache_v_residual != nullptr) {
                    float res[2] = {0.0f, 0.0f};
                    float rmax   = 0.0f;
                    if (lane < 8) {
                        const float ve = v0;
                        const float vo =
                            __bfloat162float(input_v[src0 + lane * 2 + 1]);
                        const std::uint8_t ce = gqa_iso3_nibble(ve, vscale);
                        const std::uint8_t co = gqa_iso3_nibble(vo, vscale);
                        res[0] = ve - gqa_iso3_decode(ce) * vscale;
                        res[1] = vo - gqa_iso3_decode(co) * vscale;
                        rmax   = fmaxf(fabsf(res[0]), fabsf(res[1]));
                    } else if (lane < 16) {
                        const std::uint8_t cd = gqa_iso3_nibble(v0, vscale);
                        res[0] = v0 - gqa_iso3_decode(cd) * vscale;
                        rmax   = fabsf(res[0]);
                    }
#pragma unroll
                    for (int off = 8; off > 0; off >>= 1) {
                        rmax = fmaxf(rmax, __shfl_xor_sync(FullMask, rmax, off));
                    }
                    const float rvscale = fmaxf(rmax / 7.0f, 0.001953125f);
                    if (lane < 8) {
                        cache_v_residual[vcode + lane] =
                            static_cast<std::uint8_t>(gqa_iso3_nibble(res[0], rvscale) |
                                                      (gqa_iso3_nibble(res[1], rvscale) << 4));
                    }
                    if (lane == 0) {
                        cache_v_residual_scale[gqa_kv_nvfp4_scale_index<Geometry>(
                            physical_page, kv_head, grp, page_offset)] =
                            gqa_kv_nvfp4_fp32_to_e4m3(rvscale);
                    }
                }
            } else {
                const float vscale = fmaxf(vmax / 6.0f, kNvfp4MinScale);
                if (lane < 8) {
                    cache_v[vcode + lane] =
                        static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(v0 / vscale) |
                                                  (gqa_kv_nvfp4_e2m1_nibble(
                                                       __bfloat162float(
                                                           input_v[src0 + lane * 2 + 1]) /
                                                       vscale)
                                                   << 4));
                }
                if (lane == 0) {
                    cache_v_scale[gqa_kv_nvfp4_scale_index<Geometry>(
                        physical_page, kv_head, grp, page_offset)] =
                        gqa_kv_nvfp4_fp32_to_e4m3(vscale);
                }
            }
        }
        __syncthreads();
    }

    // ---- on-chip Q quantization (same rotation as K) ----
    for (int i = tid; i < Br * 128; i += Threads) { q_a[i] = 0; }
    for (int i = tid; i < Br * 16; i += Threads) { q_sf[i] = kNvfp4E4M3One; }
    __syncthreads();

_Pragma("unroll 1")
    for (int unit = warp; unit < RowCount * Groups; unit += Wc) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        const int src = gqa_q_index<Geometry>(q_head, grp * 16, token);
        float qx[4];
        gqa_nvfp4_load_rotate_4(qx, q + src, grp, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            qx[j] *= gqa_kv_row_scale_inv(layer, kv_head, grp * 16 + lane * 4 + j);
        }
        float qmax = fmaxf(fmaxf(fabsf(qx[0]), fabsf(qx[1])), fmaxf(fabsf(qx[2]), fabsf(qx[3])));
        qmax       = gqa_nvfp4_group_max4(qmax, FullMask);
        const float qscale = fmaxf(qmax / 6.0f, kNvfp4MinScale);
        if (lane < 4) {
            q_a[row * 128 + grp * 8 + 2 * lane] =
                static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(qx[0] / qscale) |
                                          (gqa_kv_nvfp4_e2m1_nibble(qx[1] / qscale) << 4));
            q_a[row * 128 + grp * 8 + 2 * lane + 1] =
                static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(qx[2] / qscale) |
                                          (gqa_kv_nvfp4_e2m1_nibble(qx[3] / qscale) << 4));
        }
        if (lane == 0) { q_sf[row * 16 + grp] = gqa_kv_nvfp4_fp32_to_e4m3(qscale); }
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

#define NINFER_NVFP4_STAGE_TILE(TILE_K0, PHYSICAL_PAGE, SLOT)                               \
    do {                                                                                     \
        const int stage_tile_k0       = (TILE_K0);                                           \
        const int stage_global_k0    = window_begin + stage_tile_k0;                         \
        const int stage_physical_page = (PHYSICAL_PAGE);                                     \
        const bool stage_cold         = stage_physical_page <= -2 &&                         \
                                 cold_k_slots != nullptr && cold_v_slots != nullptr &&       \
                                 slot_bytes >= 1024 + 320;                              \
        const int stage_slot_base = stage_cold ? -stage_physical_page - 2 : 0;               \
        const int stage_slot_id   = stage_slot_base + kv_head;                               \
        const int stage_half      = (stage_global_k0 & kPagedKVPageMask) >> 5;               \
        const std::uint8_t* stage_k_slot =                                                    \
            stage_cold ? cold_k_slots + static_cast<std::int64_t>(stage_slot_id) *            \
                             slot_bytes                                                  \
                       : nullptr;                                                            \
        const std::uint8_t* stage_v_slot =                                                    \
            stage_cold ? cold_v_slots + static_cast<std::int64_t>(stage_slot_id) *            \
                             slot_bytes                                                  \
                       : nullptr;                                                            \
        std::uint8_t* stage_k_pk      = r_s + (SLOT) * kTileBytes;                           \
        std::uint8_t* stage_k_rpk     = stage_k_pk + Bc * 128;                               \
        std::uint8_t* stage_v_pk      = stage_k_rpk + Bc * 128;                              \
        std::uint8_t* stage_v_rpk     = stage_v_pk + Bc * 128;                               \
        std::uint8_t* stage_k_sf      = stage_v_rpk + Bc * 128;                              \
        std::uint8_t* stage_k_rsf     = stage_k_sf + Bc * 16;                                \
        std::uint8_t* stage_v_sf      = stage_k_rsf + Bc * 16;                               \
        std::uint8_t* stage_v_rsf     = stage_v_sf + Bc * 16;                               \
        for (int key_l = tid; key_l < Bc; key_l += Threads) {                                \
            const int key = stage_global_k0 + key_l;                                         \
            if (key >= window_begin + split_start && key < window_begin + split_end) {       \
                if (stage_cold) {                                                            \
                    const std::uint8_t* k_scales =                                             \
                        entropy_nvfp4_slot_scales(stage_k_slot, slot_bytes);             \
                    const std::uint8_t* v_scales =                                             \
                        entropy_nvfp4_slot_scales(stage_v_slot, slot_bytes);             \
                    ninfer::ops::cp_async<16>(&stage_k_sf[key_l * 16],                        \
                                              &k_scales[(stage_half * 32 + key_l) * 16]);      \
                    ninfer::ops::cp_async<16>(&stage_v_sf[key_l * 16],                        \
                                              &v_scales[(stage_half * 32 + key_l) * 16]);      \
                    store_vec(&stage_k_rsf[key_l * 16], make_int4(0, 0, 0, 0));               \
                    store_vec(&stage_v_rsf[key_l * 16], make_int4(0, 0, 0, 0));               \
                } else {                                                                     \
                    const std::int64_t scale_off = gqa_kv_nvfp4_scale_index<Geometry>(       \
                        stage_physical_page, kv_head, 0, key & kPagedKVPageMask);            \
                    ninfer::ops::cp_async<16>(&stage_k_sf[key_l * 16], &cache_k_scale[scale_off]); \
                    ninfer::ops::cp_async<16>(&stage_v_sf[key_l * 16], &cache_v_scale[scale_off]); \
                    if (cache_k_residual_scale != nullptr) {                                   \
                        ninfer::ops::cp_async<16>(&stage_k_rsf[key_l * 16],                    \
                                                  &cache_k_residual_scale[scale_off]);         \
                    } else {                                                                   \
                        store_vec(&stage_k_rsf[key_l * 16], make_int4(0, 0, 0, 0));            \
                    }                                                                           \
                    if (cache_v_residual_scale != nullptr) {                                   \
                        ninfer::ops::cp_async<16>(&stage_v_rsf[key_l * 16],                    \
                                                  &cache_v_residual_scale[scale_off]);         \
                    } else {                                                                   \
                        store_vec(&stage_v_rsf[key_l * 16], make_int4(0, 0, 0, 0));            \
                    }                                                                           \
                }                                                                            \
            } else {                                                                         \
                store_vec(&stage_k_sf[key_l * 16], make_int4(0, 0, 0, 0));                   \
                store_vec(&stage_k_rsf[key_l * 16], make_int4(0, 0, 0, 0));                  \
                store_vec(&stage_v_sf[key_l * 16], make_int4(0, 0, 0, 0));                   \
                store_vec(&stage_v_rsf[key_l * 16], make_int4(0, 0, 0, 0));                  \
            }                                                                                \
        }                                                                                    \
        if (stage_cold) {                                                                    \
            for (int chunk = tid; chunk < Bc * 8; chunk += Threads) {                        \
                const int key_l  = chunk >> 3;                                               \
                const int j      = chunk & 7;                                                \
                store_vec(&stage_k_rpk[key_l * 128 + j * 16], make_int4(0, 0, 0, 0));        \
                store_vec(&stage_v_rpk[key_l * 128 + j * 16], make_int4(0, 0, 0, 0));        \
            }                                                                                \
            if (tid < kEntropyNvfp4SlotStreamsPerHalf) {                                     \
                std::uint8_t* dst = stage_k_pk + tid * kEntropyNvfp4SlotStreamBytes;          \
                if (!entropy_nvfp4_slot_decode_stream(stage_k_slot, stage_half, tid, dst)) { \
                    for (int i = 0; i < kEntropyNvfp4SlotStreamBytes; ++i) { dst[i] = 0; }    \
                }                                                                            \
            } else if (tid < 2 * kEntropyNvfp4SlotStreamsPerHalf) {                          \
                const int stream = tid - kEntropyNvfp4SlotStreamsPerHalf;                    \
                std::uint8_t* dst = stage_v_pk + stream * kEntropyNvfp4SlotStreamBytes;       \
                if (!entropy_nvfp4_slot_decode_stream(stage_v_slot, stage_half, stream,      \
                                                      dst)) {                                \
                    for (int i = 0; i < kEntropyNvfp4SlotStreamBytes; ++i) { dst[i] = 0; }    \
                }                                                                            \
            }                                                                                \
            __syncthreads();                                                                 \
        } else {                                                                             \
_Pragma("unroll 1")                                                                         \
            for (int chunk = tid; chunk < Bc * 8; chunk += Threads) {                        \
                const int key_l  = chunk >> 3;                                               \
                const int j      = chunk & 7;                                                \
                const int d      = j * 32;                                                   \
                const int key    = stage_global_k0 + key_l;                                  \
                std::uint8_t* dst_k = &stage_k_pk[key_l * 128 + j * 16];                     \
                std::uint8_t* dst_r = &stage_k_rpk[key_l * 128 + j * 16];                    \
                std::uint8_t* dst_v = &stage_v_pk[key_l * 128 + j * 16];                     \
                std::uint8_t* dst_vr = &stage_v_rpk[key_l * 128 + j * 16];                   \
                if (key >= window_begin + split_start && key < window_begin + split_end) {   \
                    const std::int64_t code_off = gqa_kv_nvfp4_code_index<Geometry>(         \
                        stage_physical_page, kv_head, d, key & kPagedKVPageMask);            \
                    ninfer::ops::cp_async<16>(dst_k, &cache_k[code_off]);                    \
                    if (cache_k_residual != nullptr) {                                       \
                        ninfer::ops::cp_async<16>(dst_r, &cache_k_residual[code_off]);       \
                    } else {                                                                 \
                        store_vec(dst_r, make_int4(0, 0, 0, 0));                             \
                    }                                                                        \
                    ninfer::ops::cp_async<16>(dst_v, &cache_v[code_off]);                    \
                    if (cache_v_residual != nullptr) {                                       \
                        ninfer::ops::cp_async<16>(dst_vr, &cache_v_residual[code_off]);      \
                    } else {                                                                 \
                        store_vec(dst_vr, make_int4(0, 0, 0, 0));                            \
                    }                                                                        \
                } else {                                                                     \
                    store_vec(dst_k, make_int4(0, 0, 0, 0));                                 \
                    store_vec(dst_r, make_int4(0, 0, 0, 0));                                 \
                    store_vec(dst_v, make_int4(0, 0, 0, 0));                                 \
                    store_vec(dst_vr, make_int4(0, 0, 0, 0));                                \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
        ninfer::ops::cp_commit();                                                            \
    } while (0)

    int physical_page = physical_pages_s[0];
    NINFER_NVFP4_STAGE_TILE(first_tile, physical_page, 0);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0        = first_tile + kb * Bc;
        const int global_k0 = window_begin + k0;
        const int slot      = kb & 1;
        std::uint8_t* k_pk = r_s + slot * kTileBytes;
        std::uint8_t* k_rpk = k_pk + Bc * 128;
        std::uint8_t* v_pk = k_rpk + Bc * 128;
        std::uint8_t* v_rpk = v_pk + Bc * 128;
        std::uint8_t* k_sf = v_rpk + Bc * 128;
        std::uint8_t* k_rsf = k_sf + Bc * 16;
        std::uint8_t* v_sf = k_rsf + Bc * 16;
        std::uint8_t* v_rsf = v_sf + Bc * 16;

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
            for (int k = 0; k < QKKs; ++k) {
                unsigned af[4];
                gqa_nvfp4_load_a_frag(af, q_a + producer_row_base * 128, lane, k);
                const unsigned sfa = load_vec<unsigned>(
                    q_sf + producer_row_base * 16 + (gid + (lid & 1) * 8) * 16 + k * 4);
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    unsigned bf[2];
                    gqa_nvfp4_load_b_frag(bf, k_pk, lane, nt, k);
                    const unsigned sfb = load_vec<unsigned>(k_sf + (gid + nt * 8) * 16 + k * 4);
                    mma_nvfp4_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3],
                                   af[0], af[1], af[2], af[3], bf[0], bf[1], sfa, sfb);
                }
            }
            // Second QK pass accumulates the E2M1 residual K plane.
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned af[4];
                gqa_nvfp4_load_a_frag(af, q_a + producer_row_base * 128, lane, k);
                const unsigned sfa = load_vec<unsigned>(
                    q_sf + producer_row_base * 16 + (gid + (lid & 1) * 8) * 16 + k * 4);
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    unsigned bf[2];
                    gqa_nvfp4_load_b_frag(bf, k_rpk, lane, nt, k);
                    const unsigned sfb =
                        load_vec<unsigned>(k_rsf + (gid + nt * 8) * 16 + k * 4);
                    mma_nvfp4_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3],
                                   af[0], af[1], af[2], af[3], bf[0], bf[1], sfa, sfb);
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
                const int key0 = global_k0 + col0;
                const int key1 = global_k0 + col1;
                const int split_begin = window_begin + split_start;
                const int split_limit = window_begin + split_end;
                score[nt][0] =
                    (row0 < RowCount && key0 >= split_begin && key0 < split_limit && key0 <= qabs0)
                        ? score[nt][0] * scale
                        : -CUDART_INF_F;
                score[nt][1] =
                    (row0 < RowCount && key1 >= split_begin && key1 < split_limit && key1 <= qabs0)
                        ? score[nt][1] * scale
                        : -CUDART_INF_F;
                score[nt][2] =
                    (row1 < RowCount && key0 >= split_begin && key0 < split_limit && key0 <= qabs1)
                        ? score[nt][2] * scale
                        : -CUDART_INF_F;
                score[nt][3] =
                    (row1 < RowCount && key1 >= split_begin && key1 < split_limit && key1 <= qabs1)
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
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            const int next_k0        = k0 + Bc;
            const int next_global_k0 = window_begin + next_k0;
            if ((next_global_k0 & kPagedKVPageMask) == 0) {
                physical_page =
                    physical_pages_s[(next_global_k0 >> kPagedKVPageShift) - first_global_page];
            }
            NINFER_NVFP4_STAGE_TILE(next_k0, physical_page, (kb + 1) & 1);
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

        if constexpr (Iso3V) {
            // Hybrid PV: K keeps native mxf4nvf4 QK, V is decoded from ISO3
            // nibbles into a full-D swizzled BF16 tile and P x V runs on BF16 mma.
            for (int idx = tid; idx < Bc * D; idx += Threads) {
                const int pos = idx / D;
                const int d   = idx - pos * D;
                const std::uint8_t byte = v_pk[pos * 128 + (d >> 1)];
                const std::uint8_t code = (d & 1) ? (byte >> 4) : (byte & 0x0F);
                const float vscale = gqa_kv_nvfp4_e4m3_to_f32(v_sf[pos * 16 + (d >> 4)]);
                float value = gqa_iso3_decode(code) * vscale;
                const std::uint8_t rbyte = v_rpk[pos * 128 + (d >> 1)];
                const std::uint8_t rcode = (d & 1) ? (rbyte >> 4) : (rbyte & 0x0F);
                const float vrscale = gqa_kv_nvfp4_e4m3_to_f32(v_rsf[pos * 16 + (d >> 4)]);
                value += gqa_iso3_decode(rcode) * vrscale;
                v_bf16[pos * D + gqa_small_t_tc_swz(pos, d)] = __float2bfloat16(value);
            }
            __syncthreads();

            const int a_mat    = lane >> 3;
            const int a_rin    = lane & 7;
            const int a_rowoff = a_rin + ((a_mat & 1) << 3);
            const int a_coloff = (a_mat >> 1) << 3;
            const int b_rin    = lane & 7;
            const int b_koff   = ((lane >> 3) & 1) << 3;
            for (int ddg = 0; ddg < DgCount; ++ddg) {
                const int dg = consumer_slice * DgCount + ddg;
                const int n0 = 2 * ddg;
                for (int k = 0; k < PVKs; ++k) {
                    unsigned pf[4];
                    const int pcol = k * 16 + a_coloff;
                    ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                                smem_addr(&p_consumer[a_rowoff * Bc +
                                                     gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                    for (int nt = 0; nt < 2; ++nt) {
                        unsigned vf[2];
                        const int vrow = k * 16 + b_koff + b_rin;
                        const int vcol = dg * 16 + nt * 8;
                        ldmatrix_x2_t(vf[0], vf[1],
                                      smem_addr(&v_bf16[vrow * D +
                                                       gqa_small_t_tc_swz(vrow, vcol)]));
                        float dd[4] = {acc[n0 + nt][0], acc[n0 + nt][1],
                                       acc[n0 + nt][2], acc[n0 + nt][3]};
                        mma_bf16(dd[0], dd[1], dd[2], dd[3], pf[0], pf[1], pf[2], pf[3],
                                 vf[0], vf[1]);
                        acc[n0 + nt][0] = dd[0];
                        acc[n0 + nt][1] = dd[1];
                        acc[n0 + nt][2] = dd[2];
                        acc[n0 + nt][3] = dd[3];
                    }
                }
            }
        } else {
        // ---- native PV: quantize P*Vscale to E2M1 and run mxf4nvf4 mma ----
        std::uint8_t* ra = repack_a + warp * 16 * 64;
        std::uint8_t* rb = repack_b + warp * 16 * 64;
        const int row0   = consumer_row_base + gid;
        const int row1   = row0 + 8;
#pragma unroll
        for (int ddg = 0; ddg < DgCount; ++ddg) {
            const int dg = consumer_slice * DgCount + ddg;

            // Fold-max of P * Vscale over this warp's 32-position row window.
            float vmax0 = 0.0f;
            float vmax1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int pos0 = nt * 8 + lid * 2;
                const int pos1 = pos0 + 1;
                const float sv0 = gqa_kv_nvfp4_e4m3_to_f32(v_sf[pos0 * 16 + dg]);
                const float sv1 = gqa_kv_nvfp4_e4m3_to_f32(v_sf[pos1 * 16 + dg]);
                if (row0 < RowCount) {
                    const float p0 =
                        __bfloat162float(p_consumer[gid * Bc + gqa_small_t_tc_swz32(gid, pos0)]) *
                        sv0;
                    const float p1 =
                        __bfloat162float(p_consumer[gid * Bc + gqa_small_t_tc_swz32(gid, pos1)]) *
                        sv1;
                    vmax0 = fmaxf(vmax0, fmaxf(fabsf(p0), fabsf(p1)));
                }
                if (row1 < RowCount) {
                    const float p0 =
                        __bfloat162float(
                            p_consumer[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, pos0)]) *
                        sv0;
                    const float p1 =
                        __bfloat162float(
                            p_consumer[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, pos1)]) *
                        sv1;
                    vmax1 = fmaxf(vmax1, fmaxf(fabsf(p0), fabsf(p1)));
                }
            }
            vmax0 = fmaxf(vmax0, __shfl_xor_sync(FullMask, vmax0, 1));
            vmax0 = fmaxf(vmax0, __shfl_xor_sync(FullMask, vmax0, 2));
            vmax1 = fmaxf(vmax1, __shfl_xor_sync(FullMask, vmax1, 1));
            vmax1 = fmaxf(vmax1, __shfl_xor_sync(FullMask, vmax1, 2));
            const float sc0 = fmaxf(vmax0 / 6.0f, kNvfp4MinScale);
            const float sc1 = fmaxf(vmax1 / 6.0f, kNvfp4MinScale);
            psc_s[row0 * 64 + dg * 4 + lid] =
                row0 < RowCount ? gqa_kv_nvfp4_fp32_to_e4m3(sc0) : kNvfp4E4M3One;
            psc_s[row1 * 64 + dg * 4 + lid] =
                row1 < RowCount ? gqa_kv_nvfp4_fp32_to_e4m3(sc1) : kNvfp4E4M3One;
            __syncwarp();

#pragma unroll
            for (int i = lane; i < (16 * 64) / 16; i += 32) {
                reinterpret_cast<uint4*>(ra)[i] = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(rb)[i] = make_uint4(0, 0, 0, 0);
            }
            __syncwarp();

            // A = packed P*Vscale values (rows 0..15, 16 bytes of k per row).
            for (int i = lane; i < 16 * 16; i += 32) {
                const int r    = i >> 4;
                const int byte = i & 15;
                const int pos  = byte * 2;
                const int abs_row = consumer_row_base + r;
                if (abs_row < RowCount) {
                    const float sv0 = gqa_kv_nvfp4_e4m3_to_f32(v_sf[pos * 16 + dg]);
                    const float sv1 = gqa_kv_nvfp4_e4m3_to_f32(v_sf[(pos + 1) * 16 + dg]);
                    const float f0 =
                        __bfloat162float(
                            p_consumer[r * Bc + gqa_small_t_tc_swz32(r, pos)]) *
                        sv0;
                    const float f1 =
                        __bfloat162float(
                            p_consumer[r * Bc + gqa_small_t_tc_swz32(r, pos + 1)]) *
                        sv1;
                    float sc = gqa_kv_nvfp4_e4m3_to_f32(
                        psc_s[abs_row * 64 + dg * 4 + (pos >> 4)]);
                    // The minimum fold scale (2^-9) rounds to the E4M3 zero
                    // code; keep the v15 guard so an all-tiny block still
                    // quantizes instead of dividing by zero.
                    if (sc < kNvfp4MinScale) { sc = kNvfp4MinScale; }
                    ra[r * 64 + byte] =
                        static_cast<std::uint8_t>(gqa_kv_nvfp4_e2m1_nibble(f0 / sc) |
                                                  (gqa_kv_nvfp4_e2m1_nibble(f1 / sc) << 4));
                } else {
                    ra[r * 64 + byte] = 0;
                }
            }
            // B = packed V codes for dg's 16 output dims.
            for (int i = lane; i < 16 * 16; i += 32) {
                const int dd   = i >> 4;
                const int byte = i & 15;
                const int pos  = byte * 2;
                const std::uint8_t b0 = v_pk[pos * 128 + dg * 8 + (dd >> 1)];
                const std::uint8_t b1 = v_pk[(pos + 1) * 128 + dg * 8 + (dd >> 1)];
                const std::uint8_t n0 = (dd & 1) ? (b0 >> 4) : (b0 & 0x0Fu);
                const std::uint8_t n1 = (dd & 1) ? (b1 >> 4) : (b1 & 0x0Fu);
                rb[dd * 64 + byte]    = static_cast<std::uint8_t>(n0 | (n1 << 4));
            }
            __syncwarp();

            unsigned af[4];
            gqa_nvfp4_load_a_frag_64(af, ra, lane);
            const unsigned sfa = load_vec<unsigned>(
                psc_s + (consumer_row_base + (gid + (lid & 1) * 8)) * 64 + dg * 4);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                unsigned bf[2];
                gqa_nvfp4_load_b_frag_64(bf, rb, lane, nt);
                float dd[4] = {acc[2 * ddg + nt][0], acc[2 * ddg + nt][1],
                               acc[2 * ddg + nt][2], acc[2 * ddg + nt][3]};
                mma_nvfp4_e4m3(dd[0], dd[1], dd[2], dd[3], af[0], af[1], af[2], af[3], bf[0],
                               bf[1], sfa, kOnesScale);
                acc[2 * ddg + nt][0] = dd[0];
                acc[2 * ddg + nt][1] = dd[1];
                acc[2 * ddg + nt][2] = dd[2];
                acc[2 * ddg + nt][3] = dd[3];
            }
            __syncwarp();
        }
        } // Iso3V PV branch
        if (has_next) { ninfer::ops::cp_wait<0>(); }
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
            *reinterpret_cast<unsigned*>(&partial_acc[dst]) = pack_bf16x2(acc[n][0], acc[n][1]);
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<unsigned*>(&partial_acc[dst]) = pack_bf16x2(acc[n][2], acc[n][3]);
        }
    }
}

} // namespace ninfer::ops
