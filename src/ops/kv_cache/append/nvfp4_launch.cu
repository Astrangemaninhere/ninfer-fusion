// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : sergiuszm/ninfer-4090
// Branch      : rtx4090-port
// Commit      : 1bd56c9a1bdf457c6188391a9385d44d86e953aa
// Source path : src/ops/kv_cache/append/nvfp4_launch.cu
// sha256(src) : f8a3babec7bf56ea19753acecd54de4e62e990db5431dd083905fcef2b2ebe72
// Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits (rk4v4-e8 = rotated K 4b, V 4b, rk4v4 in the mix), which is a DIFFERENT axis from src/kvcfg/kv_formats.h's tier vocabulary (hot/tail/cold, Rk4v4).
#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/nvfp4_kernel.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <typename Geometry, typename CacheView, typename Metadata>
void launch_nvfp4_for(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                      Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    auto* cache_k     = static_cast<std::uint8_t*>(cache.k_pages.data);
    auto* cache_v     = static_cast<std::uint8_t*>(cache.v_pages.data);
    auto* scale_k     = static_cast<std::uint8_t*>(cache.k_scale_pages.data);
    auto* scale_v     = static_cast<std::uint8_t*>(cache.v_scale_pages.data);
    if (tokens >= 128 && Geometry::KVHeads == 2) {
        constexpr int TokensPerTile = 8;
        const int max_tiles         = div_up(tokens + TokensPerTile - 1, TokensPerTile);
        const dim3 grid(static_cast<unsigned>(max_tiles), static_cast<unsigned>(Geometry::KVHeads));
        kv_cache_append_full_nvfp4_page_kernel<Geometry, Metadata><<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata, cache_k, cache_v, scale_k,
            scale_v, tokens);
    } else {
        constexpr int FillWarps       = kBlock / 32;
        const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads;
        const int grid = static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(FillWarps)));
        kv_cache_append_full_nvfp4_kernel<Geometry, Metadata><<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata, cache_k, cache_v, scale_k,
            scale_v, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename CacheView, typename Metadata>
void dispatch_nvfp4(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                    Metadata metadata, cudaStream_t stream) {
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch_nvfp4_for<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, stream);
        return;
    }
    launch_nvfp4_for<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, stream);
}

} // namespace

void kv_cache_append_nvfp4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  PagedKVLayerView cache, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    dispatch_nvfp4(k, v, positions, cache, metadata, stream);
}

void kv_cache_append_nvfp4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        PagedKVBatchLayerView cache, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        dispatch_nvfp4(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
