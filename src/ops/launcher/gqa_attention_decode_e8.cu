// E8-tier decode launch: the packed 4-bit (E8-lattice K / i4 V) instantiations
// of the int8 decode kernel, split out of gqa_attention_decode.cu so each
// translation unit's ptxas stage fits the host memory budget (the E8 template
// flag doubles the kernel instantiation set).
#include "ops/launcher/gqa_attention.h"

#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/gqa_attention.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_i8_e8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                             PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                             std::int32_t logical_capacity, std::int32_t implementation_window,
                             std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                             Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    constexpr bool E8 = true;
    // E8 tiers have no cold-slot codec; the cold branch stays disabled.
    const std::uint8_t* cold_k_i8 = nullptr;
    const std::uint8_t* cold_v_i8 = nullptr;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
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
        gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                             KeyBlock, DynamicArena, MultiBatch, Masked, E8,
                                             CacheInput>
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
                logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    };
    if constexpr (TokenTile == 6) {
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheInput>
void launch_e8_for(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                   PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                   std::int32_t logical_capacity, std::int32_t implementation_window,
                   std::int32_t splits, Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                   cudaStream_t stream) {
    const bool masked = invocation.valid_columns != nullptr;
    const auto dispatch = [&]<bool MultiBatch, bool Masked>() {
        switch (invocation.width) {
        case 1:
            launch_tc_partial_i8_e8<Geometry, 1, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        case 2:
            launch_tc_partial_i8_e8<Geometry, 2, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        case 3:
            launch_tc_partial_i8_e8<Geometry, 3, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        case 4:
            launch_tc_partial_i8_e8<Geometry, 4, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        case 5:
            launch_tc_partial_i8_e8<Geometry, 5, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        case 6:
            launch_tc_partial_i8_e8<Geometry, 6, MultiBatch, Masked>(
                q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
                splits, partial_acc, partial_m, partial_l, stream);
            break;
        default:
            throw std::invalid_argument("E8 decode launch: unsupported T");
        }
    };
    if (invocation.batch_size == 1) {
        if (masked) {
            dispatch.template operator()<false, true>();
        } else {
            dispatch.template operator()<false, false>();
        }
    } else if (masked) {
        dispatch.template operator()<true, true>();
    } else {
        dispatch.template operator()<true, false>();
    }
}

} // namespace

void gqa_attention_decode_e8_launch(const Tensor& q, const GqaAppendInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t implementation_window, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        launch_e8_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                     implementation_window, splits, partial_acc, partial_m,
                                     partial_l, stream);
        return;
    }
    launch_e8_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                 implementation_window, splits, partial_acc, partial_m, partial_l,
                                 stream);
}

void gqa_attention_decode_e8_launch(const Tensor& q, const GqaCachedInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t implementation_window, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        launch_e8_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                     implementation_window, splits, partial_acc, partial_m,
                                     partial_l, stream);
        return;
    }
    launch_e8_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation, logical_capacity,
                                 implementation_window, splits, partial_acc, partial_m, partial_l,
                                 stream);
}

} // namespace ninfer::ops::detail
