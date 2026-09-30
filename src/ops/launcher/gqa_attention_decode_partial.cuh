#pragma once

// Shared small-T GQA partial-launch machinery, relocated verbatim from
// gqa_attention_decode.cu: launch_tc_partial_{bf16,fp8,iso3,i8,nvfp4}, the nvfp4
// geometry guard, and the E8-tier launch declarations. The small-T instantiation TU
// (gqa_attention_decode_smallt.cu) is the only consumer left: the routing TU
// (gqa_attention_decode.cu) includes gqa_attention_decode_split.h instead, so it does
// not pay for this cascade at all. The split-policy helpers the two used to share moved
// into that header, which this file includes at FILE SCOPE below -- never from inside a
// namespace (its body reopens ninfer::ops::detail, so nesting it would declare those
// entities in a nested namespace and every caller would see them as undefined).

#include "ops/launcher/gqa_attention.h"
#include "ops/launcher/gqa_attention_decode_split.h"

#include "ops/common/ft_stats.h"
#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_decode_bf16.cuh"
#include "ops/kernel/gqa_attention_decode_fp8.cuh"
#include "ops/kernel/gqa_attention_decode_iso3.cuh"
#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "ops/kernel/gqa_attention_decode_nvfp4.cuh"
// PREVOLTA-ATTN: the FFMA + online-softmax attention family (no tensor cores, no ldmatrix, no
// cp.async). Selected by gqa_attention_simt_ffma_selected(); see the arm in
// launch_tc_partial_bf16 below and ops/kernel/gqa_attention_simt_ffma.cuh for the floor.
#include "ops/kernel/gqa_attention_simt_ffma.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/gqa_attention.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// ---------------------------------------------------------------------------
// NINFER_SPLITDBG / NINFER_VERIFY_EXACT observation and contract instrument (FIX-A).
//
// NINFER_SPLITDBG=1 prints ONE line per split-KV small-T partial-kernel launch carrying every
// host-side input that decides how that launch partitions and reduces its keys: the route, the
// invocation geometry (width/full_width/column_begin), the selected tile schedule
// (TokenTile, WarpsPerCta, MinBlocksPerSm, KeyBlock), the window, the launched split count, the
// fixed split grid (split_units) and the head geometry. This is the only point in the tree that
// records the geometry of a decode launch and of a verify launch side by side; before it, the
// claim "the verify partitions a row's keys like the batch-1 decode of that row" could not be
// inspected at all (no such log ever existed).
//
// NINFER_VERIFY_EXACT (default ON) is the strict contract of
// GqaExecutionEnvelope::split_reference_keys (include/ninfer/ops/gqa_attention.h:18-25): the
// split grid must be a constant of the produced graph, not a function of how far the sequence
// has advanced. Set NINFER_VERIFY_EXACT=0 for the LEGACY arm.
inline bool gqa_splitdbg_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_SPLITDBG");
        return env != nullptr && *env != '\0' && !(env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

// The instrument is gated on length as well as on the flag: a per-launch line at every layer
// of every round of an 8,375-round run is a firehose (that run's stderr was already 10 MB
// with no per-launch logging at all). NINFER_SPLITDBG_MAX sets the number of lines this
// process will ever print (default 512, 0 disables the instrument outright); each of the two
// hosts of this helper (the dispatcher line and the tier line) counts against its own copy on
// purpose, so a small cap still yields both halves of a pair.
inline std::size_t gqa_splitdbg_cap() {
    static const std::size_t cap = [] {
        const char* env = std::getenv("NINFER_SPLITDBG_MAX");
        if (env == nullptr || *env == '\0') { return std::size_t{512}; }
        const long value = std::strtol(env, nullptr, 10);
        return value <= 0 ? std::size_t{0} : static_cast<std::size_t>(value);
    }();
    return cap;
}

inline bool gqa_splitdbg_allow() {
    if (!gqa_splitdbg_enabled()) { return false; }
    static std::atomic<std::size_t> used{0};
    return used.fetch_add(1) < gqa_splitdbg_cap();
}

inline bool gqa_verify_exact() {
    static const bool strict = [] {
        const char* env = std::getenv("NINFER_VERIFY_EXACT");
        if (env == nullptr || *env == '\0') { return true; }
        return !(env[0] == '0' && env[1] == '\0');
    }();
    return strict;
}

// The route that reached this launch. The prompt route never reaches a partial kernel (it has its
// own body), so a partial launch is small_t (one launch covering the whole query block) or
// chunked_small_t (one launch per <= kGqaSmallTChunkTokens column run, column_begin > 0 for every
// launch but the first).
inline const char* gqa_splitdbg_route(const GqaSmallTInvocation& invocation) {
    return (invocation.column_begin != 0 || invocation.width < invocation.full_width)
               ? "chunked_small_t"
               : "small_t";
}

template <typename Geometry>
inline void gqa_splitdbg_tile(const char* tier, const GqaSmallTInvocation& invocation, DType dtype,
                              int token_tile, int warps_per_cta, int min_blocks_per_sm,
                              int key_block, std::int32_t logical_capacity,
                              std::int32_t splits, std::int32_t split_units) {
    if (!gqa_splitdbg_allow()) { return; }
    std::fprintf(stderr,
                 "[splitdbg] tile tier=%s route=%s dtype=%d width=%d full_width=%d "
                 "column_begin=%d batch=%d masked=%d token_tile=%d warps_per_cta=%d "
                 "min_blocks_per_sm=%d key_block=%d logical_capacity=%d splits=%d "
                 "split_units=%d q_heads=%d kv_heads=%d group_size=%d head_dim=%d\n",
                 tier, gqa_splitdbg_route(invocation), static_cast<int>(dtype), invocation.width,
                 invocation.full_width, invocation.column_begin, invocation.batch_size,
                 invocation.valid_columns != nullptr ? 1 : 0, token_tile, warps_per_cta,
                 min_blocks_per_sm, key_block, logical_capacity, splits, split_units,
                 Geometry::QHeads,
                 Geometry::KVHeads, Geometry::GroupSize, Geometry::HeadDim);
}

// nvfp4 tiled decode is 256-only: its QK/QV tiling assumes four 64-wide tiles
// (static_assert(QKKs == 4) in gqa_attention_decode_nvfp4.cuh). Muse's head_dim
// is 128, where that tiling does not exist — before this guard the kernel was
// instantiated against a hardcoded 256 and wrote/read outside the K row into the
// V/scale planes (_TODO.md 126/127). Refuse loudly for other geometries; bf16 and
// int8 KV decode do work at 128.
[[noreturn]] inline void require_nvfp4_geometry_dim(int head_dim) {
    throw std::invalid_argument(
        "nvfp4 KV decode requires head_dim=256 but this geometry has " +
        std::to_string(head_dim) +
        "; use bf16 or int8 KV (the nvfp4 tiled kernel is not ported to 128)");
}

// Defined in gqa_attention_decode_e8.cu: the E8-tier (packed 4-bit) decode
// instantiations live in their own TU so this file's ptxas stage stays small.
void gqa_attention_decode_e8_launch(const Tensor& q, const GqaAppendInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t split_reference, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream);
void gqa_attention_decode_e8_launch(const Tensor& q, const GqaCachedInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t split_reference, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream);

namespace {

// The split-policy arithmetic (gqa_small_t_split_upper_bound / _reference / _count /
// _launch_capacity) moved to gqa_attention_decode_split.h, which this header includes at
// file scope at the top. Do NOT re-declare any of it here: the include and the definition
// would collide in the same TU, and the routing TU's whole point is that it can take the
// split policy WITHOUT this cascade.

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits,
                            std::int32_t split_units, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    // FIX-A instrument: bf16 has no KeyBlock/MinBlocksPerSm template parameter -- the bf16
    // partial kernel hardcodes Bc = 32 and __launch_bounds__(128, 2) -- so those two are
    // reported as the kernel's own constants.
    gqa_splitdbg_tile<Geometry>("bf16", invocation, cache.dtype, TokenTile, WarpsPerCta, 2, 32,
                               logical_capacity, splits, split_units);
    Tensor& cache_k = cache.k_pages;
    Tensor& cache_v = cache.v_pages;
    // BF16-COLD-LAND A4: a bf16 layer's cold record IS the int8 raw slot
    // (decoder_state.cpp cold_slot_codec_of, A1), so this body reads the same slot the
    // i8 body reads, through the same device helpers. Plane-relative bases, as in the
    // i8/nvfp4 partials. The dtype is part of the guard because a cache whose dtype
    // disagrees with the codec has no correct answer here, and reading a slot through
    // the wrong codec is worse than not reading it at all.
    const bool cold_bf16_ok = cache.dtype == DType::I8 || cache.dtype == DType::BF16;
    const std::uint8_t* cold_k_slots =
        cache.cold_slots.data != nullptr && cold_bf16_ok
            ? static_cast<const std::uint8_t*>(cache.cold_slots.data)
            : nullptr;
    const std::uint8_t* cold_v_slots =
        cold_k_slots != nullptr && cache.cold_slots.nb[2] != 0
            ? cold_k_slots + cache.cold_slots.nb[2]
            : nullptr;
    const std::int32_t* cold_k_valid =
        cold_k_slots == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(cache.cold_slot_valid.data);
    const std::int32_t* cold_v_valid =
        cold_k_valid == nullptr
            ? nullptr
            : reinterpret_cast<const std::int32_t*>(
                  reinterpret_cast<const std::uint8_t*>(cold_k_valid) +
                  cache.cold_slot_valid.nb[1]);
    // bf16 kernel uses only static smem (no dynamic staging).
    //
    // PREVOLTA-ATTN: the FFMA + online-softmax arm. It writes the SAME partial layout through
    // the same index helpers, so the reducer above is untouched and the two arms are
    // interchangeable at this call site. It has no cold-slot codec, so a launch with an armed
    // cold pool is REFUSED by name here rather than routed to the tensor-core kernel (which
    // would silently answer a different question on a rung where it can run, and trap on the
    // rung this arm exists for). The tier name in the splitdbg line is what keeps a debug log
    // from claiming the tensor-core kernel ran.
    if (gqa_attention_simt_ffma_selected()) {
        const char* const refusal = gqa_attention_simt_ffma_refusal(cold_k_slots != nullptr);
        if (refusal != nullptr) { throw std::invalid_argument(refusal); }
        // This family carries kGqaSimtFfmaRowsPerWarp rows per warp, so one CTA covers
        // WarpsPerCta * kGqaSimtFfmaRowsPerWarp rows and the ROW GROUPS become part of blockIdx.x
        // (the tensor-core arm carries every row of the (kv_head, split) in one CTA instead, which
        // is why its grid has no such factor). The partial layout is unaffected: every CTA writes
        // only the rows it carries, and the neutral partial for a row this kernel cannot compute
        // is written by whichever row group owns it.
        const int row_count = invocation.width * Geometry::GroupSize;
        constexpr int kRowsPerCta = WarpsPerCta * kGqaSimtFfmaRowsPerWarp;
        const int row_groups      = (row_count + kRowsPerCta - 1) / kRowsPerCta;
        const dim3 simt_grid(static_cast<unsigned>(Geometry::KVHeads) *
                                 static_cast<unsigned>(row_groups > 0 ? row_groups : 1),
                             static_cast<unsigned>(splits),
                             static_cast<unsigned>(invocation.batch_size));
        gqa_splitdbg_tile<Geometry>("bf16simt-ffma", invocation, cache.dtype, TokenTile,
                                    WarpsPerCta, 2, 32, logical_capacity, splits, split_units);
        gqa_attention_small_t_simt_ffma_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta,
                                                            MultiBatch, Masked,
                                                            CacheInput><<<simt_grid, kBlock, 0,
                                                                          stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), input,
            static_cast<const std::int32_t*>(pos.data), static_cast<__nv_bfloat16*>(cache_k.data),
            static_cast<__nv_bfloat16*>(cache_v.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data),
            invocation.column_masks == nullptr
                ? nullptr
                : static_cast<const std::uint64_t*>(invocation.column_masks->data),
            invocation.table_rows == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.table_rows->data),
            cache.block_tables.ne[0], invocation.width, invocation.full_width,
            invocation.column_begin, logical_capacity, split_units, scale,
            static_cast<float*>(partial_acc.data), static_cast<float*>(partial_m.data),
            static_cast<float*>(partial_l.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                 Masked, CacheInput><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), input,
        static_cast<const std::int32_t*>(pos.data), static_cast<__nv_bfloat16*>(cache_k.data),
        static_cast<__nv_bfloat16*>(cache_v.data), cold_k_slots, cold_v_slots, cold_k_valid,
        cold_v_valid, cache.slot_bytes,
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
        invocation.column_masks == nullptr
            ? nullptr
            : static_cast<const std::uint64_t*>(invocation.column_masks->data),
        invocation.table_rows == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.table_rows->data),
        cache.block_tables.ne[0], invocation.width, invocation.full_width, invocation.column_begin,
        logical_capacity, split_units, scale, static_cast<float*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
        // bf16win: the live bf16 call site for this family (gqa_attention_decode_bf16.cu
        // includes THIS header; impl.cuh's g35 sibling is left on the default 0).
        static_cast<std::int32_t>(cache.sliding_window_tokens));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_fp8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                           PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                           std::int32_t logical_capacity, std::int32_t splits,
                           std::int32_t split_units, Tensor& partial_acc,
                           Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    gqa_splitdbg_tile<Geometry>("fp8", invocation, cache.dtype, TokenTile, WarpsPerCta, 2, 32,
                               logical_capacity, splits, split_units);
    // fp8 kernel uses only static smem (no dynamic staging), same as bf16.
    gqa_attention_small_t_tc_partial_fp8_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                Masked, CacheInput><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), input,
        static_cast<const std::int32_t*>(pos.data), static_cast<std::uint8_t*>(cache_k.data),
        static_cast<std::uint8_t*>(cache_v.data),
        static_cast<std::uint8_t*>(cache_k_scale.data),
        static_cast<std::uint8_t*>(cache_v_scale.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
        invocation.table_rows == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.table_rows->data),
        cache.block_tables.ne[0], invocation.width, invocation.full_width, invocation.column_begin,
        logical_capacity, split_units, scale, static_cast<float*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput, bool Nvfp4K = false>
void launch_tc_partial_iso3(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits,
                            std::int32_t split_units, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    gqa_splitdbg_tile<Geometry>("iso4e", invocation, cache.dtype, TokenTile, WarpsPerCta, 2, 32,
                               logical_capacity, splits, split_units);
    // iso4e kernel uses only static smem (no dynamic staging), same as bf16.
    gqa_attention_small_t_tc_partial_iso3_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                 Masked, CacheInput, Nvfp4K>
        <<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), input,
        static_cast<const std::int32_t*>(pos.data), static_cast<std::uint8_t*>(cache_k.data),
        static_cast<std::uint8_t*>(cache_v.data),
        static_cast<std::uint8_t*>(cache_k_scale.data),
        static_cast<std::uint8_t*>(cache_v_scale.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
        invocation.table_rows == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.table_rows->data),
        cache.block_tables.ne[0], invocation.width, invocation.full_width, invocation.column_begin,
        logical_capacity, split_units, static_cast<int>(cache.sliding_window_tokens), scale,
        static_cast<float*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, bool E8,
          typename CacheInput>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t split_reference,
                          std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                          Tensor& partial_l, cudaStream_t stream) {
    // split_reference carries the launch's PINNED split reference (the sequence key
    // capacity), never the live window: it is the only value the fixed split grid may be
    // derived from, so that split s owns the same keys in a batch-1 decode and in the wide
    // MTP verify of the same row (gqa_attention_decode_smallt.cu:60-62 hands it over, and
    // NINFER_VERIFY_EXACT=1 refuses a launch whose envelope leaves it unpinned).
    const int split_units = gqa_small_t_split_units<Geometry>(split_reference);
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    // Revision 2b: INT8-tier cold slots (raw nibble codec). The kernel takes
    // region-relative K/V slot bases; empty tensors disable the cold branch.
    const std::uint8_t* cold_k_i8 =
        cache.cold_slots.data != nullptr && cache.dtype == DType::I8
            ? static_cast<const std::uint8_t*>(cache.cold_slots.data)
            : nullptr;
    const std::uint8_t* cold_v_i8 =
        cold_k_i8 != nullptr && cache.cold_slots.nb[2] != 0
            ? cold_k_i8 + cache.cold_slots.nb[2]
            : nullptr;
    // [F1259 kvfill] The narrow class' four planes. Empty tensors (the knob unset, a dropped
    // layer, a non-i8 tier) disable the kernel's narrow branch exactly the way the null cold
    // pointers disable the cold one -- so this is a no-op on every pre-image launch.
    const std::uint8_t* narrow_k_codes =
        cache.k_narrow_pages.data != nullptr
            ? static_cast<const std::uint8_t*>(cache.k_narrow_pages.data)
            : nullptr;
    const std::uint8_t* narrow_v_codes =
        cache.v_narrow_pages.data != nullptr
            ? static_cast<const std::uint8_t*>(cache.v_narrow_pages.data)
            : nullptr;
    const std::uint8_t* narrow_k_scales =
        cache.k_narrow_scale_pages.data != nullptr
            ? static_cast<const std::uint8_t*>(cache.k_narrow_scale_pages.data)
            : nullptr;
    const std::uint8_t* narrow_v_scales =
        cache.v_narrow_scale_pages.data != nullptr
            ? static_cast<const std::uint8_t*>(cache.v_narrow_scale_pages.data)
            : nullptr;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        gqa_splitdbg_tile<Geometry>("i8", invocation, cache.dtype, TokenTile, WarpsPerCta,
                                   MinBlocksPerSm, KeyBlock, logical_capacity, splits,
                                   split_units);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kGqaHeadDim) : 0u;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena,
                                                     MultiBatch, Masked, E8, CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                             MinBlocksPerSm, KeyBlock, DynamicArena, MultiBatch,
                                             Masked, E8, CacheInput>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data), cold_k_i8, cold_v_i8,
                cache.slot_bytes,
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, split_units, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                // i8win: the live INT8 call site (gqa_attention_decode_i8.cu instantiates
                // launch_tc_partial_i8 for this family).
                static_cast<std::int32_t>(cache.sliding_window_tokens),
                narrow_k_codes, narrow_v_codes, narrow_k_scales, narrow_v_scales,
                static_cast<std::int32_t>(cache.narrow_page_capacity));
    };
    // Revision 2b: INT8-tier cold slots (raw nibble codec). The kernel takes
    // region-relative K/V slot bases; empty tensors disable the cold branch.
    if constexpr (Geometry::GroupSize == 16) {
        // Muse (32q/2kv): RowCount = TokenTile*16 -> RowTiles = TokenTile.
        // Wc keeps Wc % TokenTile == 0 and PVNt = 16/(Wc/TokenTile) in {4,8}:
        // TT1->4, TT2->8, TT3->6, TT4->8, TT5->10, TT6->12.
        constexpr int kWc =
            TokenTile == 1 ? 4 : TokenTile == 2 ? 8 : TokenTile == 3 ? 6
            : TokenTile == 4 ? 8 : TokenTile == 5 ? 10 : 12;
        launch.template operator()<kWc, 1, 32, false>();
    } else if constexpr (TokenTile == 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (split_reference > 128 && split_reference <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (split_reference <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (split_reference <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (split_reference > 128 && split_reference <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (split_reference <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (split_reference > 128 && split_reference <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (split_reference <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (split_reference <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (split_reference <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, bool Iso3V = false>
void launch_tc_partial_nvfp4(const Tensor& q, const __nv_bfloat16* input_k,
                             const __nv_bfloat16* input_v, bool writes_cache, const Tensor& pos,
                             float scale, PagedKVBatchLayerView cache,
                             const GqaSmallTInvocation& invocation,
                             std::int32_t logical_capacity, std::int32_t split_reference,
                             std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                             Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    const bool masked     = invocation.valid_columns != nullptr;
    // Pinned split grid, exactly as launch_tc_partial_i8 above: split_reference is the
    // envelope's constant, never the live window.
    const int split_units = gqa_small_t_split_units<Geometry>(split_reference);
    const std::uint8_t* cold_k = static_cast<const std::uint8_t*>(cache.cold_slots.data);
    const std::uint8_t* cold_v =
        cold_k == nullptr ? nullptr : cold_k + cache.cold_slots.nb[2];
    const std::int32_t* cold_k_valid =
        static_cast<const std::int32_t*>(cache.cold_slot_valid.data);
    const std::int32_t* cold_v_valid =
        cold_k_valid == nullptr
            ? nullptr
            : reinterpret_cast<const std::int32_t*>(
                  reinterpret_cast<const std::uint8_t*>(cold_k_valid) +
                  cache.cold_slot_valid.nb[1]);
    // PREVOLTA-ATTN-CODEC (seam: dl/simtpack; ROUTE: dl/ffmaroute): the FFMA + online-softmax arm
    // over the PACKED NVFP4 plane pair. This is the route the pre-Ada rung was missing -- before
    // it, an NVFP4 layer fell through to the tensor-core kernel below, whose mma/ldmatrix/cp.async
    // helpers ARE the `unsupported_instruction_trap()` arm below sm_80, i.e. a TRAP AT DISPATCH
    // rather than a refusal an operator can act on. It writes the SAME partial layout through the
    // same index helpers, so the reducer above is untouched and the two arms stay interchangeable
    // here.
    //
    // IT REFUSES MORE THAN IT SERVES, AND THAT IS THE HONEST STATE. Because EVERY launch this
    // route is reachable from was, before it, a trap on the rung it exists for, a refusal here is
    // never a regression: it turns a trap into a named error. `gqa_simt_ffma_codec_refusal` is
    // asked first and its five reasons are enumerated at its definition; two of them are this
    // line's and both are cases where the arm it replaces does something this family cannot:
    //   * a declared sliding window that CAN CLIP this launch -- the replaced kernel shifts its key
    //     frame by it, this family has no such term, and the tree calls the combination SILENT
    //     CORRUPTION in those words (src/product/kv_component_switch.h:235-302). The PLAN-time
    //     refusal does not cover it, because that one keys on the layer's dtype and NVFP4 is a
    //     dtype it admits. dl/ffmawindow narrowed this one: a window at or above the kernel's
    //     serviced key bound is SERVED, because it provably cannot clip and the two arms then
    //     attend the same keys, key for key (gqa_simt_ffma_window_clips).
    //   * an armed residual plane pair -- the replaced kernel adds a second-stage term this family
    //     never reads.
    // plus the three the seam already had: the append source (writes_cache), an armed cold pool,
    // and any dtype other than NVFP4.
    //
    // ---------------------------------------------------------------------------------------
    // THE THREE DECISIONS THIS ARM HAS TO MAKE, AND WHY EACH IS THE ONE IT IS.
    // The kernel takes them as TEMPLATE parameters, so they are decided here, in the caller, and
    // the compiler can no longer see the values the runtime dispatch in `launch_tc_partial_bf16`
    // above sees. Each is named with the failure it prevents.
    //
    // (1) `MultiBatch = true`. THIS IS THE CORRECTNESS DECISION, NOT A SCHEDULING ONE.
    //     The kernel reads the batch index in exactly three places and every one of them is inside
    //     `if constexpr (MultiBatch)` (ops/kernel/gqa_attention_simt_ffma.cuh:1178 `blockIdx.z`,
    //     :1192 the q/pos column base, :1202-:1207 the three partial pointers). With MultiBatch
    //     false the kernel evaluates `batch = 0` and adds zero for all of them, so it reads batch
    //     0's columns and writes batch 0's partial slots -- and it does so from EVERY CTA whose
    //     blockIdx.z is non-zero, because the partial index helpers
    //     (`gqa_partial_stat_index` / `gqa_partial_acc_index`, :1221/:1227/:1425/:1431) carry no
    //     batch term of their own. The result is not a crash and not a NaN: it is a plausible
    //     batch-0 answer for every batch, overwritten batch_size times. That is the silent wrong
    //     this arm exists to not commit.
    //     The grid below has z extent `invocation.batch_size` UNCONDITIONALLY (the tensor-core arm
    //     below makes the same choice at :498), so `true` is the only value that is correct for
    //     every batch_size. It is also correct at batch_size == 1, where blockIdx.z is 0 and all
    //     three batch terms evaluate to zero, i.e. `true` and `false` are the SAME program there.
    //     `false` never becomes correct by being cheaper.
    //     The tie is left explicit rather than implicit: `kBatchDimIsGridZ` is read by the grid
    //     AND by the kernel's template argument, so an edit that changes one and not the other is
    //     a visible edit rather than a silent one. dl/ffmaroute measured the difference on the
    //     SASS of the two instantiations: `SR_CTAID.Z` is read in the `true` one and absent in the
    //     `false` one.
    //
    // (2) `WarpsPerCta = 4` (kBlock = 128). The family's row count per CTA is
    //     WarpsPerCta * kGqaSimtFfmaRowsPerWarp = 16 and the ROW GROUPS become part of blockIdx.x;
    //     the partial layout is unaffected. 4 is (a) the one block shape this codec has an sm_52
    //     ptxas reading for -- dl/ffmacodec's probe instantiated
    //     `<Gqa27Geometry, 1, 4, false, false, GqaCachedInput, Nvfp4>` and got 168 registers, 0
    //     spill stores/loads, against the 170-register budget `min_blocks_per_sm(Nvfp4) = 3`
    //     implies at 128 threads; (b) the bf16 ladder's own choice for every width but 1
    //     (ops/launcher/gqa_attention_decode_bf16.cu:33-55); and (c) the value that keeps the grid
    //     identical to the bf16 arm's at width 1 (row_count 6 <= 16, so row_groups is 1 either
    //     way). kGqaSimtFfmaThreadsPerWarp * WarpsPerCta is the launch's block size AND the
    //     kernel's first __launch_bounds__ argument, from one constant.
    //
    // (3) `Masked` -- DISPATCHED AT RUNTIME, because it cannot be a constant here.
    //     `masked` is `invocation.valid_columns != nullptr` (:482), a runtime value, and the two
    //     values are NOT interchangeable in either direction:
    //       * Masked = true with a null `valid_columns` dereferences it (:1185
    //         `valid_columns[batch]`) -- a fault;
    //       * Masked = false with a non-null `valid_columns` drops the only consumer of the
    //         caller's own valid-column count, i.e. the `valid_tokens == 0 -> write_neutral()`
    //         arm at :1237. On a launch whose column window for this batch is already exhausted
    //         that turns the neutral partial (m = -inf, l = 0, acc = 0) into a computed one, which
    //         the reducer then averages into the output -- again a plausible number rather than a
    //         fault.
    //     So both instantiations are built and the branch picks; this is exactly the shape
    //     `launch_bf16_for` uses (gqa_attention_decode_bf16.cu:68-87) for the same two values.
    //     The kernel's other two mask inputs, `column_masks` and `column_begin`, are already
    //     runtime parameters and are passed unchanged. NAMED ASYMMETRY, NOT REFUSED: of the six
    //     partial kernels, only the bf16 one and this family read `column_masks` at all (measured
    //     counts: bf16 5, simt_ffma 8, nvfp4 0, i8 0, fp8 0, iso3 0), so on a launch that declares
    //     one this arm applies a mask the NVFP4 arm it replaces ignores. Applying a declared
    //     visibility constraint is the more faithful reading and this family reads it by design,
    //     so the launch is served rather than refused -- but the two arms are known to differ
    //     there and this line did not determine which reading the engine intends.
    // ---------------------------------------------------------------------------------------
    if (gqa_attention_simt_ffma_selected()) {
        // The two terms the arm this route replaces reads and this family does not. Both are taken
        // from the SAME view the tensor-core arm below is handed, so the two arms cannot disagree
        // about what the launch declares.
        // THE WINDOW IS PASSED AS A BOUNDARY, NOT AS A FLAG (dl/ffmawindow). It used to be
        // `sliding_window_tokens != 0`, which refused EVERY windowed layer -- including all 16
        // paged layers of qwen3_6_27b, whose window is 646,720 (targets/qwen3_6_27b/impl/config.h
        // :43) and is a no-op for every context at or below it (:33-41, in the tree's own words).
        // A declared window the kernel provably cannot clip does not need to be refused: with the
        // origin at 0 this family's key frame IS the declared frame, so it is served; the rest are
        // still refused by name. `logical_capacity` here is the same value the kernel guards
        // `last_pos` against (gqa_attention_simt_ffma.cuh:1329), so the predicate and the kernel
        // cannot disagree about the bound. The argument and a failing-the-build witness table are
        // at gqa_simt_ffma_window_clips.
        const bool window_clips = gqa_simt_ffma_window_clips(
            static_cast<std::int32_t>(cache.sliding_window_tokens), logical_capacity);
        const bool residual_armed  = cache.k_residual_pages.data != nullptr ||
                                    cache.k_residual_scale_pages.data != nullptr ||
                                    cache.v_residual_pages.data != nullptr ||
                                    cache.v_residual_scale_pages.data != nullptr;
        const char* const refusal =
            gqa_simt_ffma_codec_refusal(GqaSimtKvCodec::Nvfp4, cold_k != nullptr, writes_cache,
                                        window_clips, residual_armed);
        if (refusal != nullptr) { throw std::invalid_argument(refusal); }
        constexpr int kWarpsPerCta = 4;
        // Read by the grid AND by the template argument below: one constant, two readers, so they
        // cannot disagree. See (1) above for why `false` here is a silent batch-0 answer.
        constexpr bool kBatchDimIsGridZ = true;
        const auto launch_codec          = [&]<bool Masked>() {
            const int row_count       = invocation.width * Geometry::GroupSize;
            constexpr int kRowsPerCta = kWarpsPerCta * kGqaSimtFfmaRowsPerWarp;
            const int row_groups      = (row_count + kRowsPerCta - 1) / kRowsPerCta;
            const dim3 simt_grid(
                static_cast<unsigned>(Geometry::KVHeads) *
                    static_cast<unsigned>(row_groups > 0 ? row_groups : 1),
                static_cast<unsigned>(splits),
                kBatchDimIsGridZ ? static_cast<unsigned>(invocation.batch_size) : 1u);
            // The tier name is what keeps a debug log from claiming the tensor-core kernel ran;
            // the two constants reported here are the FFMA kernel's own (Bc = kGqaSimtFfmaBc and
            // the codec's __launch_bounds__ target), not the tensor-core schedule's.
            gqa_splitdbg_tile<Geometry>("nvfp4simt-ffma", invocation, cache.dtype, TokenTile,
                                        kWarpsPerCta,
                                        gqa_simt_ffma_min_blocks_per_sm(GqaSimtKvCodec::Nvfp4),
                                        kGqaSimtFfmaBc, logical_capacity, splits, split_units);
            // The plane pair: `cache_k`/`cache_v` are the CODE planes and `cache_k_scale`/
            // `cache_v_scale` the SCALE planes. The kernel's two plane parameters are bf16-typed
            // because the bf16 arm is the landed one; the codec arm reinterprets them, which is
            // exactly why the codec is a template parameter of that kernel and not a runtime flag.
            // The CacheInput is spelled `GqaCachedInput` and not the function's `CacheInput`
            // because the codec arm's staging reads every key out of the cache plane and never
            // touches the append source; the landed kernel refuses `writes_cache` at compile time
            // (static_assert in the CODEC ARM block), and the runtime refusal above is what turns
            // that into a named error for an appending launch instead of a build break.
            gqa_attention_small_t_simt_ffma_partial_bf16_kernel<
                Geometry, TokenTile, kWarpsPerCta, kBatchDimIsGridZ, Masked, GqaCachedInput,
                GqaSimtKvCodec::Nvfp4><<<simt_grid, kWarpsPerCta * kGqaSimtFfmaThreadsPerWarp, 0,
                                         stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), GqaCachedInput{},
                static_cast<const std::int32_t*>(pos.data),
                reinterpret_cast<__nv_bfloat16*>(cache_k.data),
                reinterpret_cast<__nv_bfloat16*>(cache_v.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.column_masks == nullptr
                    ? nullptr
                    : static_cast<const std::uint64_t*>(invocation.column_masks->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.width, invocation.full_width,
                invocation.column_begin, logical_capacity, split_units, scale,
                static_cast<float*>(partial_acc.data), static_cast<float*>(partial_m.data),
                static_cast<float*>(partial_l.data),
                GqaSimtKvScalePlanes{static_cast<const std::uint8_t*>(cache_k_scale.data),
                                     static_cast<const std::uint8_t*>(cache_v_scale.data)});
            CUDA_CHECK(cudaGetLastError());
        };
        if (masked) {
            launch_codec.template operator()<true>();
        } else {
            launch_codec.template operator()<false>();
        }
        return;
    }

    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        gqa_splitdbg_tile<Geometry>("nvfp4", invocation, cache.dtype, TokenTile, WarpsPerCta,
                                   MinBlocksPerSm, KeyBlock, logical_capacity, splits,
                                   split_units);
        // Matches the arena layout in gqa_attention_decode_nvfp4_tiled_kernel:
        // two ping-pong tiles (k_pk/v_pk Bc*128 each + k_sf/v_sf Bc*16 each),
        // psc_s (Br*64 bytes, 64-byte row stride over RowTiles*16 rows),
        // repack_a/repack_b (Wc*16*64 each, native-PV path only — absent when
        // Iso3V so TT6 stays inside the sm_120 per-block opt-in limit).
        constexpr int kTileBytes = 4 * KeyBlock * 128 + 4 * KeyBlock * 16;
        constexpr int kRowTiles  = (TokenTile * Geometry::GroupSize + 15) / 16;
        constexpr std::size_t kRepackBytes =
            Iso3V ? 0ULL : static_cast<std::size_t>(2 * WarpsPerCta * 16 * 64);
        constexpr std::size_t kRBytes =
            static_cast<std::size_t>(2 * kTileBytes + kRowTiles * 16 * 64) + kRepackBytes;
        constexpr std::size_t kVDynamicBytes =
            Iso3V ? static_cast<std::size_t>(KeyBlock) * 256ULL * 2ULL : 0ULL;
        constexpr std::size_t kDynamicBytes =
            (DynamicArena ? kRBytes : 0ULL) + kVDynamicBytes;
        if constexpr (DynamicArena || Iso3V) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                gqa_attention_decode_nvfp4_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                        MinBlocksPerSm, KeyBlock, DynamicArena,
                                                        Iso3V>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        gqa_attention_decode_nvfp4_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                                KeyBlock, DynamicArena, Iso3V>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input_k, input_v,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::uint8_t*>(cache_k.data),
                static_cast<std::uint8_t*>(cache_v.data),
                static_cast<std::uint8_t*>(cache_k_scale.data),
                static_cast<std::uint8_t*>(cache_v_scale.data),
                static_cast<std::uint8_t*>(cache.k_residual_pages.data),
                static_cast<std::uint8_t*>(cache.k_residual_scale_pages.data),
                static_cast<std::uint8_t*>(cache.v_residual_pages.data),
                static_cast<std::uint8_t*>(cache.v_residual_scale_pages.data),
                cold_k, cold_v, cold_k_valid, cold_v_valid, cache.slot_bytes,
                static_cast<int>(cache.sliding_window_tokens),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, split_units, cache.layer_index, scale,
                static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                invocation.batch_size, masked, writes_cache,
                nvfp4_frag_ld_launch_flag());
    };
    // Minimal production schedule set for the first NVFP4 revision.
    if constexpr (Geometry::GroupSize == 16) {
        // Muse: RowTiles = TokenTile (16 rows/token); Wc table keeps
        // Wc % TokenTile == 0 with PVNt in {4,8}.
        constexpr int kWc =
            TokenTile == 1 ? 4 : TokenTile == 2 ? 8 : TokenTile == 3 ? 6
            : TokenTile == 4 ? 8 : TokenTile == 5 ? 10 : 12;
        launch.template operator()<kWc, 1, 32, true>();
    } else if constexpr (Geometry::GroupSize == 6) {
        if constexpr (TokenTile <= 4) {
            launch.template operator()<16, 1, 32, true>();
        } else if constexpr (TokenTile == 5) {
            launch.template operator()<8, 1, 32, true>();
        } else {
            launch.template operator()<12, 1, 32, true>();
        }
    } else {
        if constexpr (TokenTile <= 4) {
            launch.template operator()<16, 1, 32, true>();
        } else {
            launch.template operator()<12, 1, 32, true>();
        }
    }
    CUDA_CHECK(cudaGetLastError());
    if (ft::enabled()) {
        // FreeToken step-1: sample this layer's partial_l AFTER the reducer wrote it.
        // The probe used to sit before the launches, so it read a workspace buffer the
        // current round had not filled yet (garbage/NaN every round, _TODO.md 94).
        // Host-side D2H; observation runs use --no-cuda-graph.
        ft::observe(stream, cache.layer_index, static_cast<const float*>(partial_l.data),
                    Geometry::QHeads * invocation.width, splits);
    }
}

} // namespace

} // namespace ninfer::ops::detail
