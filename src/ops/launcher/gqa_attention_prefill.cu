// ninfer::ops - gqa_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include "ops/kernel/gqa_attention_prefill_nvfp4.cuh"
// PREVOLTA-ATTN: the FFMA + online-softmax prompt body (no tensor cores, no ldmatrix, no
// cp.async, no bf16 arithmetic). Same grid, same signature, same bottom-right causal alignment
// as gqa_attention_prefill_bf16_kernel; selected by gqa_attention_simt_ffma_selected().
#include "ops/kernel/gqa_attention_simt_ffma.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
// Defined in gqa_attention_prefill_e8.cu (E8-tier packed 4-bit launches).
void gqa_attention_prefill_e8_launch(const Tensor& q, const Tensor& positions,
                                     const std::int32_t* valid_columns,
                                     const std::int32_t* table_rows, std::int32_t table_stride,
                                     float scale, PagedKVBatchLayerView cache, Tensor& out,
                                     cudaStream_t stream);
void gqa_attention_prefill_e8_launch_single(const Tensor& q, const Tensor& positions, float scale,
                                            const PagedKVLayerView& cache, Tensor& out,
                                            cudaStream_t stream);
void gqa_kv_append_e8_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                             const std::int32_t* valid_columns, const std::int32_t* table_rows,
                             std::int32_t table_stride, PagedKVBatchLayerView cache,
                             cudaStream_t stream);
void gqa_kv_append_e8_launch_single(const Tensor& k, const Tensor& v, const Tensor& positions,
                                    const PagedKVLayerView& cache, cudaStream_t stream);
namespace {

// S45 gate. The int8/E8 prompt path (both the append fill and the attention body) is
// structurally 256-only: kGqaPrefillI8Groups==4 / static_assert(kGqaPrefillI8SmemBytes
// == 92672) / DB16==128 / static_assert(PVNtPerWarp==8) all derive from the 256
// reference (gqa_attention_prefill_common.cuh:21, gqa_attention_prefill_i8.cuh:44-51,
// :382-383). Porting it to head_dim 128 is a tile/smem re-derivation, not a constant
// swap, so head_dim 128 refuses loudly -- same shape as require_nvfp4_geometry_dim
// (gqa_attention_decode.cu:27-32). The gate sits in BOTH entry points so that the
// append fill cannot write a half-fixed plane before the attention throws.
[[noreturn]] inline void require_i8_prefill_geometry_dim(std::int32_t head_dim) {
    throw std::invalid_argument(
        "int8/E8 prefill requires head_dim=256 but this geometry has " +
        std::to_string(head_dim) +
        "; the i8/E8 prompt kernels are not ported to 128; use bf16 KV "
        "(see _collab/E3_s45_i8_plane_stride.md)");
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                               float scale, const CacheView& cache,
                                               Metadata metadata, Tensor& out,
                                               cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    // Both dtype-specialized kernels exceed the default 48 KiB dynamic-smem ceiling.
    static const cudaError_t attr_bf16 =
        cudaFuncSetAttribute(gqa_attention_prefill_bf16_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillSmemBytes);
    CUDA_CHECK(attr_bf16);
    static const cudaError_t attr_i8 =
        cudaFuncSetAttribute(gqa_attention_prefill_i8_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillI8SmemBytes);
    CUDA_CHECK(attr_i8);
    static const cudaError_t attr_i8_e8 =
        cudaFuncSetAttribute(gqa_attention_prefill_i8_kernel<Geometry, Metadata, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillI8SmemBytes);
    CUDA_CHECK(attr_i8_e8);
    // S45d: the four packed-tier kernels are 256-only by construction
    // (gqa_attention_prefill_nvfp4.cuh:1024-1026: Mxf4QKKs = D/64 must be 4,
    // Threads/ProducerThreads are 256/128) and binding their address here already
    // instantiates them, so one group guard discards all four at head_dim 128.
    if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) {
    static const cudaError_t attr_nvfp4 =
        cudaFuncSetAttribute(gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, DType::NVFP4>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kNvfp4PrefillSmemBytes);
    CUDA_CHECK(attr_nvfp4);
    static const cudaError_t attr_nvfp4k_iso3v = cudaFuncSetAttribute(
        gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, DType::NVFP4, DType::ISO3>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kNvfp4PrefillSmemBytes);
    CUDA_CHECK(attr_nvfp4k_iso3v);
    static const cudaError_t attr_fp8 =
        cudaFuncSetAttribute(gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, DType::FP8_E4M3FN>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kNvfp4PrefillSmemBytes);
    CUDA_CHECK(attr_fp8);
    static const cudaError_t attr_iso3 =
        cudaFuncSetAttribute(gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, DType::ISO3>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kNvfp4PrefillSmemBytes);
    CUDA_CHECK(attr_iso3);
    }
    // S45d: deliberately NO else-throw here. These statements are unconditional in
    // the function body, so an else-throw would also fire for bf16 and i8/E8 at
    // head_dim 128 -- i.e. for Muse's only working prefill tier. The per-arm
    // guards below refuse where the dtype is known.

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if (cache.dtype == DType::I8 || cache.dtype == DType::E8Kv) {
        if constexpr (Geometry::HeadDim != kGqaKvQuantHeadDim) {
            require_i8_prefill_geometry_dim(Geometry::HeadDim);
        }
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillI8Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        if (cache.dtype == DType::E8Kv) {
            if constexpr (std::is_same_v<Metadata, GqaPrefillDirectMetadata>) {
                gqa_attention_prefill_e8_launch_single(q, positions, scale, cache, out, stream);
            } else {
                gqa_attention_prefill_e8_launch(
                    q, positions, metadata.valid_columns, metadata.table_rows,
                    metadata.table_stride, scale, cache, out, stream);
            }
            return;
        } else {
            gqa_attention_prefill_i8_kernel<Geometry, Metadata>
                <<<attention_grid, kGqaPrefillI8Threads, kGqaPrefillI8SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const std::int8_t*>(cache_k.data),
                    static_cast<const std::int8_t*>(cache_v.data),
                    static_cast<const __half*>(cache_k_scale.data),
                    static_cast<const __half*>(cache_v_scale.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
        }
    } else if (cache.dtype == DType::NVFP4) {
        // S45d: this tier's prefill kernel is 256-only by construction
        // (gqa_attention_prefill_nvfp4.cuh:1024-1026: Mxf4QKKs = D/64 must be 4,
        // Threads/ProducerThreads are 256/128), so refuse loudly at head_dim 128.
        if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kNvfp4PrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        const std::uint8_t* cold_k =
            static_cast<const std::uint8_t*>(cache.cold_slots.data);
        const std::uint8_t* cold_v = cold_k == nullptr
                                         ? nullptr
                                         : cold_k + cache.cold_slots.nb[2];
        const std::int32_t* cold_k_valid =
            static_cast<const std::int32_t*>(cache.cold_slot_valid.data);
        const std::int32_t* cold_v_valid =
            cold_k_valid == nullptr
                ? nullptr
                : reinterpret_cast<const std::int32_t*>(
                      reinterpret_cast<const std::uint8_t*>(cold_k_valid) +
                      cache.cold_slot_valid.nb[1]);
        if (cache.v_dtype == DType::ISO3) {
            gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, DType::NVFP4, DType::ISO3>
                <<<attention_grid, kNvfp4PrefillThreads, kNvfp4PrefillSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const std::uint8_t*>(cache_k.data),
                    static_cast<const std::uint8_t*>(cache_v.data),
                    static_cast<const std::uint8_t*>(cache_k_scale.data),
                    static_cast<const std::uint8_t*>(cache_v_scale.data),
                    static_cast<const std::uint8_t*>(cache.k_residual_pages.data),
                    static_cast<const std::uint8_t*>(cache.k_residual_scale_pages.data),
                    static_cast<const std::uint8_t*>(cache.v_residual_pages.data),
                    static_cast<const std::uint8_t*>(cache.v_residual_scale_pages.data),
                    cold_k, cold_v,
                    cold_k_valid, cold_v_valid, cache.slot_bytes,
                    static_cast<int>(cache.sliding_window_tokens), cache.layer_index, metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
        } else {
            gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, DType::NVFP4>
                <<<attention_grid, kNvfp4PrefillThreads, kNvfp4PrefillSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const std::uint8_t*>(cache_k.data),
                    static_cast<const std::uint8_t*>(cache_v.data),
                    static_cast<const std::uint8_t*>(cache_k_scale.data),
                    static_cast<const std::uint8_t*>(cache_v_scale.data),
                    static_cast<const std::uint8_t*>(cache.k_residual_pages.data),
                    static_cast<const std::uint8_t*>(cache.k_residual_scale_pages.data),
                    static_cast<const std::uint8_t*>(cache.v_residual_pages.data),
                    static_cast<const std::uint8_t*>(cache.v_residual_scale_pages.data),
                    cold_k, cold_v,
                    cold_k_valid, cold_v_valid, cache.slot_bytes,
                    static_cast<int>(cache.sliding_window_tokens), cache.layer_index, metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
        }
        } else {
            throw std::invalid_argument(
                "nvfp4 prefill requires head_dim=256; this geometry is "
                "128 — that kernel is not ported; use bf16 or int8 KV");
        }
    } else if (cache.dtype == DType::ISO3) {
        if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kNvfp4PrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, DType::ISO3>
            <<<attention_grid, kNvfp4PrefillThreads, kNvfp4PrefillSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::uint8_t*>(cache_k.data),
                static_cast<const std::uint8_t*>(cache_v.data),
                static_cast<const std::uint8_t*>(cache_k_scale.data),
                static_cast<const std::uint8_t*>(cache_v_scale.data),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::int32_t*>(nullptr),
                static_cast<const std::int32_t*>(nullptr), 0, 0, cache.layer_index, metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
        } else {
            throw std::invalid_argument(
                "nvfp4 prefill requires head_dim=256; this geometry is "
                "128 — that kernel is not ported; use bf16 or int8 KV");
        }
    } else if (cache.dtype == DType::FP8_E4M3FN) {
        if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kNvfp4PrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, DType::FP8_E4M3FN>
            <<<attention_grid, kNvfp4PrefillThreads, kNvfp4PrefillSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::uint8_t*>(cache_k.data),
                static_cast<const std::uint8_t*>(cache_v.data),
                static_cast<const std::uint8_t*>(cache_k_scale.data),
                static_cast<const std::uint8_t*>(cache_v_scale.data),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::uint8_t*>(nullptr),
                static_cast<const std::int32_t*>(nullptr),
                static_cast<const std::int32_t*>(nullptr), 0, 0, cache.layer_index, metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
        } else {
            throw std::invalid_argument(
                "nvfp4 prefill requires head_dim=256; this geometry is "
                "128 — that kernel is not ported; use bf16 or int8 KV");
        }
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        // PREVOLTA-ATTN: the FFMA + online-softmax arm for this dtype. It is the same launch
        // geometry (the Br = 64 / Bc = 32 tiling is internal) and the same signature, so the only
        // thing that changes is which body runs; on a rung without tensor cores this arm is the
        // only one of the two that can run at all. A cold pool is refused by name rather than
        // routed back to the tensor-core body, for the same reason the decode arm refuses it.
        if (gqa_attention_simt_ffma_selected()) {
            if (cache.cold_slots.data != nullptr) {
                throw std::invalid_argument(
                    "the SIMT FFMA prefill route has no cold-slot codec: this layer has an armed "
                    "cold-slot pool, so it must not be routed here");
            }
            // The FFMA body carries kGqaSimtFfmaRowsPerCta rows per CTA (a different row tiling
            // from the tensor-core body's kGqaPrefillBr), so it derives its own grid; the key tiling
            // and the causal alignment are internal to the kernel.
            const dim3 simt_grid(
                static_cast<unsigned>(div_up(tokens, kGqaSimtFfmaRowsPerCta)),
                static_cast<unsigned>(Geometry::QHeads), 1u);
            gqa_attention_simt_ffma_prefill_bf16_kernel<Geometry, Metadata>
                <<<simt_grid, kGqaSimtFfmaPrefillThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(cache_k.data),
                    static_cast<const __nv_bfloat16*>(cache_v.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        gqa_attention_prefill_bf16_kernel<Geometry, Metadata>
            <<<attention_grid, kGqaPrefillThreads, kGqaPrefillSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __nv_bfloat16*>(cache_v.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_kv_append_launch_for(const Tensor& k, const Tensor& v, const Tensor& positions,
                              CacheView cache, Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    Tensor& cache_k   = cache.k_pages;
    Tensor& cache_v   = cache.v_pages;
    if (cache.dtype == DType::I8 || cache.dtype == DType::E8Kv) {
        if constexpr (Geometry::HeadDim != kGqaKvQuantHeadDim) {
            require_i8_prefill_geometry_dim(Geometry::HeadDim);
        }
        if (cache.dtype == DType::E8Kv) {
            if constexpr (std::is_same_v<Metadata, GqaPrefillDirectMetadata>) {
                gqa_kv_append_e8_launch_single(k, v, positions, cache, stream);
            } else {
                gqa_kv_append_e8_launch(k, v, positions, metadata.valid_columns,
                                        metadata.table_rows, metadata.table_stride, cache, stream);
            }
            return;
        }
        Tensor& cache_k_scale    = cache.k_scale_pages;
        Tensor& cache_v_scale    = cache.v_scale_pages;
        constexpr int kFillBlock = 256;
        if (tokens >= 128 && Geometry::KVHeads == 2) {
            constexpr int kPageBlock     = 256;
            constexpr int kTokensPerTile = 8;
            const int max_tiles          = div_up(tokens + kTokensPerTile - 1, kTokensPerTile);
            const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                                 static_cast<unsigned>(Geometry::KVHeads),
                                 static_cast<unsigned>(kGqaKvQuantGroups));
            gqa_attention_prefill_fill_i8_page_kernel<Geometry, Metadata>
                <<<fill_grid, kPageBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::int8_t*>(cache_k.data),
                    static_cast<std::int8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        } else {
            constexpr int kFillWarps = kFillBlock / 32;
            const std::int64_t fill_units =
                static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaKvQuantGroups;
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillWarps)));
            gqa_attention_prefill_fill_i8_kernel<Geometry, Metadata>
                <<<fill_grid, kFillBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::int8_t*>(cache_k.data),
                    static_cast<std::int8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        }
        CUDA_CHECK(cudaGetLastError());
    } else if (cache.dtype == DType::NVFP4) {
        Tensor& cache_k_scale = cache.k_scale_pages;
        Tensor& cache_v_scale = cache.v_scale_pages;
        constexpr int kFillBlock = 256;
        constexpr int kFillWarps = kFillBlock / 32;
        const std::int64_t fill_units =
            static_cast<std::int64_t>(tokens) * Geometry::KVHeads * (Geometry::HeadDim / kGqaKvNvfp4Group);
        const int fill_grid =
            static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillWarps)));
        if (cache.v_dtype == DType::ISO3) {
            gqa_attention_prefill_fill_nvfp4k_iso3v_kernel<Geometry, Metadata>
                <<<fill_grid, kFillBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), cache.layer_index, metadata,
                    static_cast<std::uint8_t*>(cache_k.data),
                    static_cast<std::uint8_t*>(cache_v.data),
                    static_cast<std::uint8_t*>(cache_k_scale.data),
                    static_cast<std::uint8_t*>(cache_v_scale.data),
                    static_cast<std::uint8_t*>(cache.k_residual_pages.data),
                    static_cast<std::uint8_t*>(cache.k_residual_scale_pages.data),
                    static_cast<std::uint8_t*>(cache.v_residual_pages.data),
                    static_cast<std::uint8_t*>(cache.v_residual_scale_pages.data), tokens);
        } else {
            gqa_attention_prefill_fill_nvfp4_kernel<Geometry, Metadata>
                <<<fill_grid, kFillBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), cache.layer_index, metadata,
                    static_cast<std::uint8_t*>(cache_k.data),
                    static_cast<std::uint8_t*>(cache_v.data),
                    static_cast<std::uint8_t*>(cache_k_scale.data),
                    static_cast<std::uint8_t*>(cache_v_scale.data),
                    static_cast<std::uint8_t*>(cache.k_residual_pages.data),
                    static_cast<std::uint8_t*>(cache.k_residual_scale_pages.data), tokens);
        }
        CUDA_CHECK(cudaGetLastError());
        if (cache.layer_index == 15 || cache.layer_index == 2 || cache.layer_index == 5) {
            std::int32_t dbg_bt0 = -1;
            const std::int32_t* dbg_bt_ptr = nullptr;
            if constexpr (std::is_same_v<CacheView, PagedKVBatchLayerView>) {
                dbg_bt_ptr = static_cast<const std::int32_t*>(cache.block_tables.data);
            } else {
                dbg_bt_ptr = static_cast<const std::int32_t*>(cache.block_table.data);
            }
            CUDA_CHECK(cudaMemcpy(&dbg_bt0, dbg_bt_ptr, sizeof(dbg_bt0), cudaMemcpyDeviceToHost));
        }
    } else if (cache.dtype == DType::ISO3) {
        Tensor& cache_k_scale = cache.k_scale_pages;
        Tensor& cache_v_scale = cache.v_scale_pages;
        constexpr int kFillBlock = 256;
        constexpr int kFillWarps = kFillBlock / 32;
        const std::int64_t fill_units =
            static_cast<std::int64_t>(tokens) * Geometry::KVHeads * (Geometry::HeadDim / kGqaKvNvfp4Group);
        const int fill_grid =
            static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillWarps)));
        gqa_attention_prefill_fill_iso3_kernel<Geometry, Metadata>
            <<<fill_grid, kFillBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::uint8_t*>(cache_k.data),
                static_cast<std::uint8_t*>(cache_v.data),
                static_cast<std::uint8_t*>(cache_k_scale.data),
                static_cast<std::uint8_t*>(cache_v_scale.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    } else if (cache.dtype == DType::FP8_E4M3FN) {
        Tensor& cache_k_scale = cache.k_scale_pages;
        Tensor& cache_v_scale = cache.v_scale_pages;
        constexpr int kFillBlock = 256;
        constexpr int kFillWarps = kFillBlock / 32;
        const std::int64_t fill_units =
            static_cast<std::int64_t>(tokens) * Geometry::KVHeads * (Geometry::HeadDim / kGqaKvNvfp4Group);
        const int fill_grid =
            static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillWarps)));
        gqa_attention_prefill_fill_fp8_kernel<Geometry, Metadata>
            <<<fill_grid, kFillBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::uint8_t*>(cache_k.data),
                static_cast<std::uint8_t*>(cache_v.data),
                static_cast<std::uint8_t*>(cache_k_scale.data),
                static_cast<std::uint8_t*>(cache_v_scale.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    } else {
        constexpr int kBlock           = Geometry::KVHeads == 4 ? 128 : 96;
        constexpr int kFillVecElems    = 8;
        const std::int64_t kv_elements = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                                         (Geometry::HeadDim / kFillVecElems);
        const int fill_grid =
            static_cast<int>(div_up(kv_elements, static_cast<std::int64_t>(kBlock)));
        gqa_attention_prefill_fill_bf16_kernel<Geometry, Metadata>
            <<<fill_grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(k.data),
                                               static_cast<const __nv_bfloat16*>(v.data),
                                               static_cast<const std::int32_t*>(positions.data),
                                               metadata, static_cast<__nv_bfloat16*>(cache_k.data),
                                               static_cast<__nv_bfloat16*>(cache_v.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, Tensor& out,
                                           cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
        gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(q, positions, scale, cache,
                                                                 metadata, out, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        gqa_attention_prompt_attention_launch_for<GqaMuseGeometry>(q, positions, scale, cache,
                                                                   metadata, out, stream);
        return;
    }
    if (q.ne[1] == Gqa16x4Geometry::QHeads && q.ne[0] == Gqa16x4Geometry::HeadDim) {
        gqa_attention_prompt_attention_launch_for<Gqa16x4Geometry>(q, positions, scale, cache,
                                                                   metadata, out, stream);
        return;
    }
    if (q.ne[1] == Gqa35Geometry::QHeads && q.ne[0] == Gqa35Geometry::HeadDim) {
        gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(q, positions, scale, cache,
                                                                 metadata, out, stream);
        return;
    }
    throw std::invalid_argument(
        "gqa_attention_prompt_attention_launch: unsupported q-head geometry (" +
        std::to_string(q.ne[1]) + ")");
}

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    // The append path sees only k/v (kv-heads x head_dim). Gqa27Geometry (24q/4kv) and
    // Gqa16x4Geometry (16q/4kv, Spark-X2.5) differ only in QHeads, which no fill kernel
    // reads, so the 4/256 plane instance is exact for both: keep one instantiation and
    // record the equivalence instead of duplicating it.
    if (k.ne[1] == Gqa27Geometry::KVHeads && cache.head_dim == Gqa27Geometry::HeadDim) {
        gqa_kv_append_launch_for<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
        return;
    }
    if (cache.head_dim == GqaMuseGeometry::HeadDim) {
        gqa_kv_append_launch_for<GqaMuseGeometry>(k, v, positions, cache, metadata, stream);
        return;
    }
    if (k.ne[1] == Gqa35Geometry::KVHeads && cache.head_dim == Gqa35Geometry::HeadDim) {
        gqa_kv_append_launch_for<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
        return;
    }
    throw std::invalid_argument(
        "gqa_kv_append_launch: unsupported KV geometry (" + std::to_string(k.ne[1]) +
        " kv-heads, head-dim " + std::to_string(cache.head_dim) + ")");
}

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& column_masks, const Tensor& table_rows, float scale,
                                 PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream) {
    // M1: the per-column ancestor masks of an MTP tree verify round. This route applies them in
    // the BF16 prompt body, where the position-causal cut is applied, and nowhere else. Every
    // property that makes that well defined is REFUSED BY NAME here instead of assumed, because
    // dropping a mask silently verifies a chain and calls it a tree:
    //   * the mask row is selected per query row inside the CTA, while this route's metadata
    //     reduces to ONE table row (GqaPrefillBatchMetadata::block_table() reads table_rows[0]),
    //     so a batch above 1 has no row to read a mask from;
    //   * the mask drives the per-key cut of ONE query block (Br = 64 columns), so a round wider
    //     than that would leave the columns above the block unmasked;
    //   * only the BF16 body reads the mask -- the i8/E8/packed arms of this route implement the
    //     `valid_columns` prefix cut only;
    //   * the mask cannot replace the valid-columns cut: it bounds which KEYS a live column sees,
    //     not how many columns are live (the runtime states that separately as
    //     target_valid_columns = extent + 1, program_impl.h:13428).
    if (column_masks.data != nullptr) {
        if (valid_columns.data == nullptr) {
            throw std::invalid_argument(
                "gqa_attention_prompt_launch: per-column masks (MTP tree verify) need the valid "
                "column count of the round; this call has no valid-columns tensor");
        }
        if (cache.dtype != DType::BF16) {
            throw std::invalid_argument(
                "gqa_attention_prompt_launch: per-column masks (MTP tree verify) are implemented "
                "for the BF16 prompt body only; this call has a quantized or E8 KV tier");
        }
        if (q.ne[3] != 1) {
            throw std::invalid_argument(
                "gqa_attention_prompt_launch: per-column masks (MTP tree verify) are implemented "
                "for a single sequence; this call has B = " + std::to_string(q.ne[3]));
        }
        if (q.ne[2] > kGqaPrefillBr) {
            throw std::invalid_argument(
                "gqa_attention_prompt_launch: per-column masks (MTP tree verify) index one query "
                "block of at most " + std::to_string(kGqaPrefillBr) +
                " columns; this call has W = " + std::to_string(q.ne[2]));
        }
    }
    const auto launch = [&]<bool Masked>() {
        const GqaPrefillBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
            .column_masks = column_masks.data == nullptr
                                ? nullptr
                                : static_cast<const std::uint64_t*>(column_masks.data),
        };
        if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
            gqa_kv_append_launch_for<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
            gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(q, positions, scale, cache,
                                                                     metadata, out, stream);
            return;
        }
        if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
            gqa_kv_append_launch_for<GqaMuseGeometry>(k, v, positions, cache, metadata, stream);
            gqa_attention_prompt_attention_launch_for<GqaMuseGeometry>(q, positions, scale, cache,
                                                                       metadata, out, stream);
            return;
        }
        if (q.ne[1] == Gqa16x4Geometry::QHeads && q.ne[0] == Gqa16x4Geometry::HeadDim &&
            cache.num_kv_heads == Gqa16x4Geometry::KVHeads) {
            // The fill instance is interchangeable at this plane (see gqa_kv_append_launch).
            gqa_kv_append_launch_for<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
            gqa_attention_prompt_attention_launch_for<Gqa16x4Geometry>(
                q, positions, scale, cache, metadata, out, stream);
            return;
        }
        if (q.ne[1] == Gqa35Geometry::QHeads && q.ne[0] == Gqa35Geometry::HeadDim &&
            cache.num_kv_heads == Gqa35Geometry::KVHeads) {
            gqa_kv_append_launch_for<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
            gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(q, positions, scale, cache,
                                                                     metadata, out, stream);
            return;
        }
        throw std::invalid_argument("gqa_attention_prompt_launch: unsupported q-head geometry (" +
                                    std::to_string(q.ne[1]) + ")");
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
