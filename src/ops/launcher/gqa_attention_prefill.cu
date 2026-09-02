// ninfer::ops - gqa_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include "ops/kernel/gqa_attention_prefill_nvfp4.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
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

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if (cache.dtype == DType::I8 || cache.dtype == DType::E8Kv) {
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
    } else if (cache.dtype == DType::ISO3) {
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
    } else if (cache.dtype == DType::FP8_E4M3FN) {
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
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
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
            static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaKvNvfp4Groups;
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
            static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaKvNvfp4Groups;
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
            static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaKvNvfp4Groups;
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
                                         (kGqaPrefillHeadDim / kFillVecElems);
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
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(q, positions, scale, cache,
                                                                 metadata, out, stream);
        return;
    }
    gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(q, positions, scale, cache, metadata,
                                                             out, stream);
}

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (k.ne[1] == Gqa27Geometry::KVHeads) {
        gqa_kv_append_launch_for<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
        return;
    }
    gqa_kv_append_launch_for<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
}

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const GqaPrefillBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == Gqa27Geometry::QHeads) {
            gqa_kv_append_launch_for<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
            gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(q, positions, scale, cache,
                                                                     metadata, out, stream);
            return;
        }
        gqa_kv_append_launch_for<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
        gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(q, positions, scale, cache,
                                                                 metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
