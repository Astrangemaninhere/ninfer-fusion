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
#include "ops/kernel/nvfp4_ldm_free.cuh"             // the ldmatrix-free fragment loader arm

#include <cstdint>

namespace ninfer::ops {
namespace {

using namespace ninfer::ops::detail;

constexpr float kNvfp4MinScale = 0.001953125f;                    // 2^-9, E4M3 smallest normal
constexpr std::uint8_t kNvfp4E4M3One = 0x38u;                     // E4M3FN encoding of 1.0

// THE ARM'S FOUR LOADERS (dl/nvfp4emu; the loader itself is ops/kernel/nvfp4_ldm_free.cuh).
// `simt` is the arm's decision, handed down from the kernel's own `nvfp4_frag_ld` parameter so
// that ONE compiled kernel serves both arms and the A/B is a launch-time choice -- the same shape
// the SIMT FFMA family uses for its own two bodies. It is uniform across the launch, so each
// branch below is a predicate and not a divergence.
//
// THE TWO ARMS ARE BIT-IDENTICAL BY CONSTRUCTION, and that is the only reason this arm may claim
// it: the pointer handed to the ldmatrix-free loader is the very pointer whose `smem_addr()` the
// ldmatrix arm is given, so the same bytes land in the same registers, and the `mma_*` that
// consumes them cannot tell which arm produced them.
__device__ __forceinline__ void gqa_nvfp4_load_a_frag(unsigned (&frag)[4], const std::uint8_t* smem,
                                                      int lane, int k_step, bool simt) {
    const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int col = (lane >> 4) * 16 + k_step * 32;
    if (simt) {
        nvfp4_ldm_free<false, 4>(frag, smem + row * 128 + col, lane);
        return;
    }
    ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], smem_addr(smem + row * 128 + col));
}

__device__ __forceinline__ void gqa_nvfp4_load_b_frag(unsigned (&frag)[2], const std::uint8_t* smem,
                                                      int lane, int n_tile, int k_step,
                                                      bool simt) {
    const int row = (lane & 7) + n_tile * 8;
    const int col = ((lane >> 3) & 1) * 16 + k_step * 32;
    if (simt) {
        nvfp4_ldm_free<false, 2>(frag, smem + row * 128 + col, lane);
        return;
    }
    ldmatrix_x2(frag[0], frag[1], smem_addr(smem + row * 128 + col));
}

// PV repack tiles use a 64-byte row stride (16-byte k-fragment per mxf4nvf4
// m16n8k64 operand row).
__device__ __forceinline__ void gqa_nvfp4_load_a_frag_64(unsigned (&frag)[4],
                                                         const std::uint8_t* smem, int lane,
                                                         bool simt) {
    const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int col = (lane >> 4) * 16;
    if (simt) {
        nvfp4_ldm_free<false, 4>(frag, smem + row * 64 + col, lane);
        return;
    }
    ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], smem_addr(smem + row * 64 + col));
}

__device__ __forceinline__ void gqa_nvfp4_load_b_frag_64(unsigned (&frag)[2],
                                                         const std::uint8_t* smem, int lane,
                                                         int n_tile, bool simt) {
    const int row = (lane & 7) + n_tile * 8;
    const int col = ((lane >> 3) & 1) * 16;
    if (simt) {
        nvfp4_ldm_free<false, 2>(frag, smem + row * 64 + col, lane);
        return;
    }
    ldmatrix_x2(frag[0], frag[1], smem_addr(smem + row * 64 + col));
}

// Lane l < 4 loads the four values of its 4-channel block and applies the
// baked SO(4) rotation to it; the runtime on/off gate lives inside
// gqa_isoquant_rot_block4() (gqa_isoquant_rot.cuh) and nowhere else. The
// caller's src pointer ALREADY points at the 16-dimension group start.
__device__ __forceinline__ void gqa_nvfp4_load_rotate_4(float (&x)[4], const __nv_bfloat16* src,
                                                       int group, int lane) {
    if (lane < 4) {
        const int block = group * 4 + lane;
        const int base  = lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) { x[j] = __bfloat162float(src[base + j]); }
        gqa_isoquant_rot_block4(x, block);
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

// ===========================================================================
// winmech LOUD CHECK -- Site A.  Two contracts, both of which FAIL THE BUILD.
//
// This translation unit opens no warning flags, so a silent wrong answer would
// otherwise be invisible.  -DNDEBUG also makes `assert` dead, so the guard here
// is deliberately NOT a runtime assert: it is a constant-expression check on the
// SAME functions the kernel calls, so if either contract is broken again the
// build stops and says which one.
//
// CONTRACT 1 (origin): the mask origin must be the DECLARED oldest visible key,
//   max(0, context - sliding_window).  Rounding it UP to the Bc grid -- which is
//   what the kernel used to do, because the STAGING wanted the alignment -- makes
//   the mask exclude up to Bc-1 keys the declaration says are visible.  Measured
//   on one vehicle/binary/flag set: W=7552 DEGENERATE, W=7553 clean, W=7569
//   DEGENERATE (non-monotone), and the defect is decode-side from step 2
//   (dl/winmech/REPORT.md).
//
// CONTRACT 2 (tile frame): staging and compute must enumerate the SAME absolute
//   keys.  A run-1 draft of this fix re-based the staging on `stage_begin` and
//   left the compute frame on `window_begin`; the two frames then differ by
//   `stage_skew`, which silently attributes one key's data to another key and
//   never stages the top `stage_skew` keys of the range.  The check below states
//   the two properties that make the shifted frame sound: the first tile starts
//   at or below the range start, and the last tile covers the range top.
// ===========================================================================
constexpr int kWmNvfp4Bc = 32;  // == KeyBlock; the kernel static_asserts Bc == 32.

__host__ __device__ constexpr int gqa_wm_nvfp4_window_origin(int token_begin) {
    // The DECLARED oldest visible key.  Never rounded.
    return token_begin > 0 ? token_begin : 0;
}

__host__ __device__ constexpr int gqa_wm_nvfp4_stage_begin(int window_begin, int bc) {
    // Floor, never ceil: keys below window_begin are staged and then rejected by the
    // mask, which is the safe direction.  A ceil puts keys IN the window out of reach.
    return (window_begin / bc) * bc;
}

__host__ __device__ constexpr int gqa_wm_nvfp4_first_tile(int split_start, int stage_skew,
                                                         int bc) {
    // The split's offset expressed in the staging frame, floored onto the tile grid.
    return ((split_start + stage_skew) / bc) * bc;
}

__host__ __device__ constexpr int gqa_wm_nvfp4_key_blocks(int first_tile, int split_end,
                                                          int stage_skew, int bc) {
    return ((split_end + stage_skew - first_tile) + bc - 1) / bc;
}

// The witness table.  Its LENGTH is asserted below, so a later edit cannot quietly
// delete the cases that make the contracts load-bearing -- deleting a case removes
// the evidence, not the defect.  The origins are the measured arms's origins
// (context - window at P=7680 and P=4028); the split offsets are the family's fixed
// split grid (split_units = 512).
struct GqaWmNvfp4Witness {
    int window_begin;
    int split_start;
    int split_end;
};
constexpr GqaWmNvfp4Witness kGqaWmNvfp4Witness[] = {
    {127, 0, 127},    // wm_a    origin 7680-7553  -> skew 31 (the worst case)
    {127, 96, 127},   // wm_a    last split
    {128, 0, 128},    // t_7552  origin 7680-7552  -> skew 0 (already aligned)
    {111, 0, 111},    // wm_b    origin 7680-7569  -> skew 15
    {64, 0, 64},      // t_7616  origin 64         -> skew 0
    {64, 32, 64},     // t_7616  split 1
    {1, 0, 1},        // t_7679  origin 1          -> skew 1
    {128, 0, 512},    // s_m128  a full split_units row
    {31, 0, 512},     // skew 31 with a partial split grid
    {0, 0, 0},        // b_262144: the window is a no-op; the kernel early-outs
};
constexpr int kGqaWmNvfp4WitnessCount = 10;
static_assert(sizeof(kGqaWmNvfp4Witness) / sizeof(kGqaWmNvfp4Witness[0]) ==
                  kGqaWmNvfp4WitnessCount,
              "winmech SITE A: the window-origin witness table lost a case. These cases ARE "
              "the check that the mask origin never excludes more keys than the sliding "
              "window declares; deleting one to make the build pass removes the check, not "
              "the defect (dl/winmech/REPORT.md).");

constexpr bool gqa_wm_nvfp4_contract_holds() {
    for (int i = 0; i < kGqaWmNvfp4WitnessCount; ++i) {
        const int wb = kGqaWmNvfp4Witness[i].window_begin;
        const int ss = kGqaWmNvfp4Witness[i].split_start;
        const int se = kGqaWmNvfp4Witness[i].split_end;
        if (wb < 0) { return false; }
        // CONTRACT 1, stated as an identity on the audited function: the origin of a
        // context/window pair is its declared oldest key.  A ceil-round-up anywhere in
        // this function makes these two disagree.
        if (gqa_wm_nvfp4_window_origin(wb) != wb) { return false; }
        if (gqa_wm_nvfp4_window_origin(-1) != 0) { return false; }
        const int stage = gqa_wm_nvfp4_stage_begin(wb, kWmNvfp4Bc);
        if (stage > wb) { return false; }                     // floor, never above
        if ((wb - stage) >= kWmNvfp4Bc) { return false; }      // ... and never a whole tile
        if ((stage % kWmNvfp4Bc) != 0) { return false; }       // the staging base is aligned
        const int skew = wb - stage;
        if (ss >= se) { continue; }  // the kernel early-outs before computing any tile
        // CONTRACT 2: the tile grid, in the staging frame.
        const int ft = gqa_wm_nvfp4_first_tile(ss, skew, kWmNvfp4Bc);
        if ((ft % kWmNvfp4Bc) != 0) { return false; }  // every staged tile is on the grid
        if (ft > ss + skew) { return false; }          // ... and never above the range start
        const int kb = gqa_wm_nvfp4_key_blocks(ft, se, skew, kWmNvfp4Bc);
        if (kb <= 0) { return false; }
        if (ft + kb * kWmNvfp4Bc < se + skew) { return false; }  // the range TOP is covered
    }
    return true;
}
static_assert(gqa_wm_nvfp4_contract_holds(),
              "winmech SITE A: the decode window is not the declared sliding window. EITHER "
              "the mask origin is not max(0, context - sliding_window) -- i.e. the Bc "
              "round-up is back, and up to Bc-1 declared-visible keys are silently excluded "
              "-- OR the staging/compute tile frame is not the 32-aligned `stage_begin` "
              "frame widened by `stage_skew`, so staged data and attended keys disagree "
              "(dl/winmech/REPORT.md).");

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
        std::int32_t column_begin, std::int32_t logical_capacity, std::int32_t split_units,
        int layer, float scale, float* partial_acc, float* partial_m, float* partial_l,
        std::int32_t batch_size, bool masked, bool writes_cache, int nvfp4_frag_ld) {
    // THE ARM'S SWITCH (dl/nvfp4emu): see nvfp4_frag_ld_selected() in
    // ops/kernel/nvfp4_ldm_free.cuh. Uniform across the launch.
    const bool frag_ld    = nvfp4_frag_ld != 0;
    constexpr int Wc      = WarpsPerCta;
    constexpr int RowCount = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles = (RowCount + 15) / 16;
    constexpr int Br       = RowTiles * 16;
    constexpr int Bc       = KeyBlock;
    constexpr int D        = Geometry::HeadDim;
    constexpr int Threads  = Wc * 32;
    constexpr int Groups   = (Geometry::HeadDim / kGqaKvNvfp4Group);
    constexpr int QKKs     = D / 64;
    constexpr int QKNt     = Bc / 8;
    constexpr int PVKs     = Bc / 16;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp = D / (ConsumerWarpsPerTile * 8);
    constexpr int DgCount     = PVNtPerWarp / 2;
    constexpr int PageIds     = paged_kv_page_ids(kCausalAttentionMaximumVisibleKeysYarn);
    constexpr float Log2E         = 1.4426950408889634074f;
    constexpr unsigned FullMask   = 0xffffffffu;
    constexpr unsigned kOnesScale = 0x38383838u;

    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(Bc == 32);
    // RowTiles == TokenTile for GroupSize-16 geometries (Muse 32q/2kv); the
    // dispatch kWc table keeps every derived quantity in range up to 6.
    static_assert(RowTiles >= 1 && RowTiles <= 6);
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
    // The per-warp P/V repack tiles exist only for the native mxf4nvf4 PV path.
    // The Iso3V branch decodes V into v_bf16 and runs BF16 mma instead, so the
    // repack bytes are dead weight there; dropping them keeps TT6 (Wc=12) inside
    // the sm_120 per-block opt-in limit (99 KiB) instead of 102.9 KiB.
    constexpr int kRepackBytes = Iso3V ? 0 : 2 * Wc * 16 * 64;
    __shared__ __align__(16) std::uint8_t q_a[Br * 128];
    __shared__ __align__(16) std::uint8_t q_sf[Br * 16];
    __shared__ __align__(16) std::uint8_t static_r_s[DynamicArena
                                                        ? 16
                                                        : 2 * kTileBytes + Br * 16 * 4 +
                                                              kRepackBytes];
    extern __shared__ __align__(16) std::uint8_t nvfp4_dynamic_r_s[];
    std::uint8_t* r_s      = DynamicArena ? nvfp4_dynamic_r_s : static_r_s;
    std::uint8_t* psc_s    = r_s + 2 * kTileBytes;
    // Native-PV only: never dereferenced on the Iso3V path.
    std::uint8_t* repack_a = Iso3V ? r_s : psc_s + Br * 16 * 4;
    std::uint8_t* repack_b = Iso3V ? r_s : repack_a + Wc * 16 * 64;
    __shared__ __align__(16) __nv_bfloat16 p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ std::int32_t physical_pages_s[PageIds];
    // Hybrid V path decodes the packed ISO3 V tile into BF16 with the exact
    // full-D tc swizzle the ldmatrix PV path expects.
    constexpr int kRBytes = 2 * kTileBytes + Br * 16 * 4 + kRepackBytes;
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
    q += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if (writes_cache) {
        input_k += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::KVHeads * column_base;
        input_v += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    partial_acc += static_cast<std::int64_t>(batch) * Geometry::HeadDim * Geometry::QHeads *
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
                    0.0f;                                                    \
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
    // The window ORIGIN is the DECLARED origin. It is what the MASK compares keys
    // against, so it must not be rounded. It used to be rounded UP to the Bc grid here,
    // which served the STAGING -- a 32-key staging tile must not straddle a 64-key page
    // -- and was then charged to the window: up to Bc-1 keys the declaration said were
    // visible silently fell out of the mask. The two roles are separate now. `stage_begin`
    // is the floor-aligned base the staging and its tile grid need, and every absolute key
    // in this kernel is derived from it; the mask keeps comparing against `window_begin`.
    // See the winmech block above: the contract is static_asserted, not assumed.
    // `stage_skew` shifts the split offsets into the staging frame, so the first tile
    // starts at or below the range start and the last tile covers the range top.
    const int token_begin   = (sliding_window > 0) ? window_full - sliding_window : 0;
    const int window_begin  = gqa_wm_nvfp4_window_origin(token_begin);
    const int window        = window_full - window_begin;
    const int stage_begin   = gqa_wm_nvfp4_stage_begin(window_begin, Bc);
    const int stage_skew    = window_begin - stage_begin;
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
        NINFER_NVFP4_WRITE_NEUTRAL();
        return;
    }
    // The tile grid lives in the STAGING frame: `stage_skew` moves the split's offsets
    // onto it, so the first tile starts at or below the range start and the last tile
    // covers the range top. `first_global_page` uses the staging base; `page_count` keeps
    // the window's own top, so the load span is
    // [stage_begin + first_tile, window_begin + split_end) and no visible key is unloaded.
    const int first_tile = gqa_wm_nvfp4_first_tile(split_start, stage_skew, Bc);
    const int key_blocks = gqa_wm_nvfp4_key_blocks(first_tile, split_end, stage_skew, Bc);
    const int first_global_page = (stage_begin + first_tile) >> kPagedKVPageShift;
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
            // Inside the lane<4 window on purpose: lanes >= 4 hold x = 0 from the rotate
            // helper and their group max is reduced within their own 4-lane group and never
            // consumed, so scaling them is dead work that widens the constant-memory address
            // span from 16 to 128 words (4 cache lines) per warp. Matches the prefill site.
            if (lane < 4) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    kx[j] *= gqa_kv_row_scale(layer, kv_head, grp * 16 + lane * 4 + j);
                }
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
                    // Byte lane holds the (2*lane, 2*lane+1) channel pair,
                    // matching gqa_prefill_fill_nvfp4k_iso3v_kernel. v0 is
                    // channel `lane` (used by the 16-lane max above), so the
                    // even member must be reloaded here, not v0.
                    const float ve = __bfloat162float(input_v[src0 + lane * 2]);
                    const float vo =
                        __bfloat162float(input_v[src0 + lane * 2 + 1]);
                    cache_v[vcode + lane] =
                        static_cast<std::uint8_t>(gqa_iso3_nibble(ve, vscale) |
                                                  (gqa_iso3_nibble(vo, vscale) << 4));
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
                        const float ve = __bfloat162float(input_v[src0 + lane * 2]);
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
                    // Even channel 2*lane, paired with odd 2*lane+1 below
                    // (same byte layout as the ISO3 branch and the prefill
                    // fill kernels).
                    cache_v[vcode + lane] =
                        static_cast<std::uint8_t>(
                            gqa_kv_nvfp4_e2m1_nibble(__bfloat162float(
                                                         input_v[src0 + lane * 2]) /
                                                     vscale) |
                            (gqa_kv_nvfp4_e2m1_nibble(
                                 __bfloat162float(input_v[src0 + lane * 2 + 1]) /
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
        // Same window as the K side: only lanes 0..3 carry the rotated block, and the
        // reciprocal is a full IEEE division, so keeping it off the dead lanes removes
        // 28 of 32 divisions per warp per group with no numeric change.
        if (lane < 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                qx[j] *= gqa_kv_row_scale_inv(layer, kv_head, grp * 16 + lane * 4 + j);
            }
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
        const int stage_global_k0    = stage_begin + stage_tile_k0;                         \
        const int stage_physical_page = (PHYSICAL_PAGE);                                     \
        const int stage_slot_base = stage_physical_page <= -2 ? -stage_physical_page - 2 : 0; \
        /* Pool layout is [slot_bytes, kv_heads, 2, pages]: the flat head-slot  */            \
        /* index is slot * 2*KVHeads + head, as in the i8 decode, the i8 prefill */           \
        /* and the nvfp4 prefill. Both slot regions and BOTH validity planes    */            \
        /* arrive plane-relative (launcher: V = K + nb[1] / + nb[2]), so K and V  */          \
        /* take the same flat id here -- the single-base readers (small_t_*) are  */          \
        /* the ones that add KVHeads for the plane offset.                      */            \
        const int stage_slot_id = stage_slot_base * (2 * Geometry::KVHeads) + kv_head;        \
        const bool stage_cold         = stage_physical_page <= -2 &&                         \
                                 cold_k_slots != nullptr && cold_v_slots != nullptr &&       \
                                 cold_k_valid != nullptr && cold_v_valid != nullptr &&        \
                                 slot_bytes >= 1024 + 320 &&                                  \
                                 cold_k_valid[stage_slot_id] != 0 &&                          \
                                 cold_v_valid[stage_slot_id] != 0;                            \
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
                if (!entropy_nvfp4_slot_decode_stream(stage_k_slot, slot_bytes, stage_half, tid, dst)) { \
                    for (int i = 0; i < kEntropyNvfp4SlotStreamBytes; ++i) { dst[i] = 0; }    \
                }                                                                            \
            } else if (tid < 2 * kEntropyNvfp4SlotStreamsPerHalf) {                          \
                const int stream = tid - kEntropyNvfp4SlotStreamsPerHalf;                    \
                std::uint8_t* dst = stage_v_pk + stream * kEntropyNvfp4SlotStreamBytes;       \
                if (!entropy_nvfp4_slot_decode_stream(stage_v_slot, slot_bytes, stage_half,  \
                                                      stream, dst)) {                        \
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
        const int global_k0 = stage_begin + k0;
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
                gqa_nvfp4_load_a_frag(af, q_a + producer_row_base * 128, lane, k, frag_ld);
                const unsigned sfa = load_vec<unsigned>(
                    q_sf + producer_row_base * 16 + (gid + (lid & 1) * 8) * 16 + k * 4);
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    unsigned bf[2];
                    gqa_nvfp4_load_b_frag(bf, k_pk, lane, nt, k, frag_ld);
                    const unsigned sfb = load_vec<unsigned>(k_sf + (gid + nt * 8) * 16 + k * 4);
                    mma_nvfp4_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3],
                                   af[0], af[1], af[2], af[3], bf[0], bf[1], sfa, sfb);
                }
            }
            // Second QK pass accumulates the E2M1 residual K plane. The plane is only
            // allocated for layers that opted into the residual (kv_residual_layers), and
            // the staging zero-fills it otherwise - so when it is absent this pass
            // multiplies zeros and is pure cost (it consumed the whole fp4-vs-s8 QK
            // advantage: one mxf4 instruction carries 2x the MACs of mma_s8, so the extra
            // pass left nvfp4 issuing exactly as many QK MMAs per key block as int8).
            // Guarding it on the pointer is uniform across the launch and changes nothing
            // numerically: the skipped MMAs only ever added 0.
            const bool has_k_residual = cache_k_residual != nullptr;
            if (has_k_residual) {
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned af[4];
                gqa_nvfp4_load_a_frag(af, q_a + producer_row_base * 128, lane, k, frag_ld);
                const unsigned sfa = load_vec<unsigned>(
                    q_sf + producer_row_base * 16 + (gid + (lid & 1) * 8) * 16 + k * 4);
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    unsigned bf[2];
                    gqa_nvfp4_load_b_frag(bf, k_rpk, lane, nt, k, frag_ld);
                    const unsigned sfb =
                        load_vec<unsigned>(k_rsf + (gid + nt * 8) * 16 + k * 4);
                    mma_nvfp4_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3],
                                   af[0], af[1], af[2], af[3], bf[0], bf[1], sfa, sfb);
                }
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
            const int next_global_k0 = stage_begin + next_k0;
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
            // One thread per 16-element group (Bc*D/16 == the thread count here), so the
            // E4M3 group scale is decoded once instead of once per element, and the residual
            // plane is only touched when it exists. Same element count per thread as before.
            const bool has_v_residual = cache_v_residual != nullptr;
            for (int g = tid; g < Bc * (D / 16); g += Threads) {
                const int pos = g / (D / 16);
                const int grp = g - pos * (D / 16);
                const float vscale  = gqa_kv_nvfp4_e4m3_to_f32(v_sf[pos * 16 + grp]);
                const float vrscale =
                    has_v_residual ? gqa_kv_nvfp4_e4m3_to_f32(v_rsf[pos * 16 + grp]) : 0.0F;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int d = grp * 16 + j;
                    const std::uint8_t byte = v_pk[pos * 128 + (d >> 1)];
                    const std::uint8_t code = (d & 1) ? (byte >> 4) : (byte & 0x0F);
                    float value = gqa_iso3_decode(code) * vscale;
                    if (has_v_residual) {
                        const std::uint8_t rbyte = v_rpk[pos * 128 + (d >> 1)];
                        const std::uint8_t rcode = (d & 1) ? (rbyte >> 4) : (rbyte & 0x0F);
                        value += gqa_iso3_decode(rcode) * vrscale;
                    }
                    v_bf16[pos * D + gqa_small_t_tc_swz(pos, d)] = __float2bfloat16(value);
                }
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
                    const __nv_bfloat16* const p_row =
                        &p_consumer[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)];
                    if (frag_ld) {
                        nvfp4_ldm_free<false, 4>(pf, p_row, lane);
                    } else {
                        ldmatrix_x4(pf[0], pf[1], pf[2], pf[3], smem_addr(p_row));
                    }
                    for (int nt = 0; nt < 2; ++nt) {
                        unsigned vf[2];
                        const int vrow = k * 16 + b_koff + b_rin;
                        const int vcol = dg * 16 + nt * 8;
                        const __nv_bfloat16* const v_row =
                            &v_bf16[vrow * D + gqa_small_t_tc_swz(vrow, vcol)];
                        if (frag_ld) {
                            nvfp4_ldm_free<true, 2>(vf, v_row, lane);
                        } else {
                            ldmatrix_x2_t(vf[0], vf[1], smem_addr(v_row));
                        }
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
            gqa_nvfp4_load_a_frag_64(af, ra, lane, frag_ld);
            const unsigned sfa = load_vec<unsigned>(
                psc_s + (consumer_row_base + (gid + (lid & 1) * 8)) * 64 + dg * 4);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                unsigned bf[2];
                gqa_nvfp4_load_b_frag_64(bf, rb, lane, nt, frag_ld);
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
