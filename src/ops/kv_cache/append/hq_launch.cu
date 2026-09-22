// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : cometkim/ninfer
// Branch      : feat/1m-context
// Commit      : 2cc56b5db39f951c1e91b2caec8b39274d795682
// Source path : src/ops/kv_cache/append/hq_launch.cu
// sha256(src) : 92c438910e14b1d4435b97023d8736935016240be5bd1294d1394fb78dbc773b
// Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits. The 2026-09-18 fork-survey merge landed this file with `e8` renamed to `rk4v4`; INTEGRATE2 REVERTED that rename, on three measured grounds. (1) It is FALSE about this codec's own geometry: the codec is 2-bit K against 4-bit V -- the `-2b` token survived the rewrite while the codec name did not, and its type is still `Rk4v4Packed2BitTile`. (2) It collides with the LIVE tier vocabulary that NAMEFIX3 established in src/kvcfg/kv_formats.h (`Rk4v4` / `Rk3v4` / `Rk2v4` as TIERS, 13 tracked files), so one name came to mean two things -- MERGE flag C2 instantiated in code. (3) The revert is free of build consequence: no build reads either name. The body below is now byte-identical to the pinned `sha256(src)`.
// ninfer::ops::detail - hq-e8-2b append launch ownership.
#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/hq_kernel.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = kKVCacheHqFillWarps * 32;

template <typename Geometry, typename CacheView, typename Metadata>
void launch_hq_for(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                   Metadata metadata, cudaStream_t stream) {
    const auto tokens             = static_cast<std::int32_t>(k.ne[2]);
    const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads * 2;
    const int grid =
        static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kKVCacheHqFillWarps)));
    kv_cache_append_full_hq_kernel<Geometry, Metadata>
        <<<grid, kBlock, kKVCacheHqFillSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata,
            static_cast<std::uint8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data),
            static_cast<std::uint8_t*>(cache.k_scale_pages.data),
            static_cast<std::uint8_t*>(cache.v_scale_pages.data),
            static_cast<__nv_bfloat16*>(cache.residual_k.data),
            static_cast<__nv_bfloat16*>(cache.residual_v.data),
            static_cast<std::uint32_t*>(cache.side_words.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CacheView, typename Metadata>
void dispatch_hq(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                 Metadata metadata, cudaStream_t stream) {
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch_hq_for<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, stream);
        return;
    }
    launch_hq_for<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, stream);
}

} // namespace

void kv_cache_append_hq_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                               PagedKVLayerView cache, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data), cache.slot};
    dispatch_hq(k, v, positions, cache, metadata, stream);
}

void kv_cache_append_hq_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
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
        dispatch_hq(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
