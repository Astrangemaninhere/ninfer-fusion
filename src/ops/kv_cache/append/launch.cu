#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/e8_lattice_narrow_kernel.cuh"
#include "ops/kv_cache/append/kernel.cuh"
#include "ops/kv_cache/d256_profile.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <typename Geometry, typename CacheView, typename Metadata>
void launch_full(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                 Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    Tensor& cache_k   = cache.k_pages;
    Tensor& cache_v   = cache.v_pages;
    if (cache.dtype == DType::FP8_E4M3FN) {
        Tensor& cache_k_scale = cache.k_scale_pages;
        Tensor& cache_v_scale = cache.v_scale_pages;
        if (tokens >= 128 && Geometry::KVHeads == 2) {
            constexpr int TokensPerTile = 8;
            const int max_tiles         = div_up(tokens + TokensPerTile - 1, TokensPerTile);
            const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                                 static_cast<unsigned>(Geometry::KVHeads));
            kv_cache_append_full_fp8_page_kernel<Geometry, Metadata>
                <<<fill_grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::uint8_t*>(cache_k.data),
                    static_cast<std::uint8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        } else {
            constexpr int FillWarps       = kBlock / 32;
            const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads;
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(FillWarps)));
            kv_cache_append_full_fp8_kernel<Geometry, Metadata><<<fill_grid, kBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::uint8_t*>(cache_k.data), static_cast<std::uint8_t*>(cache_v.data),
                static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
                tokens);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (cache.dtype == DType::I8) {
        Tensor& cache_k_scale = cache.k_scale_pages;
        Tensor& cache_v_scale = cache.v_scale_pages;
        if (tokens >= 128 && Geometry::KVHeads == 2) {
            constexpr int TokensPerTile = 8;
            const int max_tiles         = div_up(tokens + TokensPerTile - 1, TokensPerTile);
            const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                                 static_cast<unsigned>(Geometry::KVHeads));
            kv_cache_append_full_i8_page_kernel<Geometry, Metadata>
                <<<fill_grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::int8_t*>(cache_k.data),
                    static_cast<std::int8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        } else {
            constexpr int FillWarps       = kBlock / 32;
            const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads;
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(FillWarps)));
            kv_cache_append_full_i8_kernel<Geometry, Metadata><<<fill_grid, kBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::int8_t*>(cache_k.data), static_cast<std::int8_t*>(cache_v.data),
                static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
                tokens);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    constexpr int Block         = Geometry::KVHeads == 4 ? 128 : 96;
    constexpr int VecElems      = 8;
    const std::int64_t elements = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                                  (kKVCacheAppendFullHeadDim / VecElems);
    const int fill_grid = static_cast<int>(div_up(elements, static_cast<std::int64_t>(Block)));
    kv_cache_append_full_bf16_kernel<Geometry, Metadata><<<fill_grid, Block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data), metadata,
        static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
        tokens);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// THE NARROW e8 K-PLANE ARM (W3 / W2).
//
// `launch_full` above is {FP8, I8, else BF16} and has NO e8 arm: its BF16 fallback casts
// `cache.k_pages.data` to `__nv_bfloat16*`, which over a 128/96/64-byte e8 row stride
// would write 256 bf16 elements per row. That is why `kv_cache_append.cpp` refuses the
// whole e8 family by name and why the kFullHeadDim (256) check there was left alone.
//
// This is the missing writer, and it is deliberately NOT folded into `launch_full`: the
// e8 narrow arm needs the codec of record's 6 428 B table handle, which a `kv_cache_append`
// caller does not have and which the op's admission does not carry. It is a direct entry
// point instead (the standing `gqa_kv_append_e8_launch` has), and it changes nothing about
// what `kv_cache_append()` admits -- NO ADMISSION IS WIDENED HERE.
//
// The width comes from `cache.dtype`, through the same family predicate and the same
// profile table the ops layer already derives its extents from, so this arm cannot serve a
// width the geometry of record does not publish.
// ---------------------------------------------------------------------------
template <typename Geometry, typename CacheView, typename Metadata>
void launch_e8_narrow(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                      Metadata metadata, const E8KvLatticeTables& tables, cudaStream_t stream) {
    // ⚠ NOT `tables.valid()`: that member is `__device__ __forceinline__`, and calling it
    // from this HOST launcher is refused by nvcc ("calling a __device__ function from a
    // __host__ function is not allowed", measured on this very line). The two pointers are
    // plain host-visible members, so the guard is spelled on them.
    if (tables.s1 == nullptr || tables.s2 == nullptr) {
        throw std::invalid_argument(
            "kv_cache_append_e8_lattice_launch: the e8 lattice tables are not supplied; the "
            "codec of record needs its 6 428 B of stage-1/stage-2 tables, and a null handle "
            "would fault on device rather than refuse here");
    }
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    if (tokens <= 0) { return; }
    auto* cache_k = static_cast<std::uint8_t*>(cache.k_pages.data);
    auto* cache_v = static_cast<std::uint8_t*>(cache.v_pages.data);
    auto* scale_k = static_cast<__half*>(cache.k_scale_pages.data);
    auto* scale_v = static_cast<__half*>(cache.v_scale_pages.data);
    constexpr int kBlock = 256;
    // [dl/e8vaxis F1166] THE PLANE AXIS REACHES THE LAUNCHER AS A SECOND TEMPLATE PARAMETER.
    // Both arms below name `kE8AppendVBitsShippedI4` for V, so the bytes this launcher writes
    // are UNCHANGED by the axis; a caller that wants the family's lattice on V names its width
    // (3 or 2) instead, and then only the V plane's code extent moves.
    const auto launch = [&]<int WBits, int VBits>() {
        const std::int64_t units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                                   kE8AppendKGroups;
        const int grid = static_cast<int>(div_up(units, static_cast<std::int64_t>(kBlock)));
        kv_cache_append_full_e8_lattice_kernel<Geometry, Metadata, WBits, VBits>
            <<<grid, kBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata, cache_k, cache_v,
                scale_k, scale_v, tables, tokens);
        CUDA_CHECK(cudaGetLastError());
    };
    if (cache.dtype == DType::E8K3Kv) {
        launch.template operator()<3, kE8AppendVBitsShippedI4>();
        return;
    }
    if (cache.dtype == DType::E8K2Kv) {
        launch.template operator()<2, kE8AppendVBitsShippedI4>();
        return;
    }
    // E8Kv (4-bit) is NOT served here, and neither is any unpacked tier. The 4-bit K plane
    // is the packed i4 codec at 128 B/row and the shipped i4 V plane; writing a 96/64-byte
    // lattice plate into it is the same class of corruption the op-level check refuses.
    throw std::invalid_argument(
        "kv_cache_append_e8_lattice_launch: the narrow arm serves E8K3Kv and E8K2Kv only; "
        "E8Kv is the shipped 4-bit packed tier and appends through gqa_kv_append_e8_launch");
}

template <typename CacheView, typename Metadata>
void dispatch_e8_narrow(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                        Metadata metadata, const E8KvLatticeTables& tables, cudaStream_t stream) {
    // The same KV-head ladder every other arm in this file uses: 4 heads is
    // KVCacheAppendD256Kv4, everything else that got this far is the 2-head geometry.
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch_e8_narrow<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, tables, stream);
        return;
    }
    launch_e8_narrow<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, tables, stream);
}

void validate_plan(const Tensor& k, const KVCacheAppendPrefixPlan& plan) {    if (plan.tokens != k.ne[2] || plan.min_count < 0 || plan.max_count < plan.min_count ||
        plan.max_count > plan.tokens) {
        throw std::invalid_argument("kv_cache_append_prefix: inconsistent plan");
    }
}

void launch_paged(const Tensor& k, const Tensor& v, const Tensor& positions, const Tensor& counts,
                  const Tensor& table_rows, PagedKVBatchLayerView cache,
                  const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    validate_plan(k, plan);
    if (plan.max_count == 0) return;
    auto* cache_k       = static_cast<__nv_bfloat16*>(cache.k_pages.data);
    auto* cache_v       = static_cast<__nv_bfloat16*>(cache.v_pages.data);
    const auto* input_k = static_cast<const __nv_bfloat16*>(k.data);
    const auto* input_v = static_cast<const __nv_bfloat16*>(v.data);
    const auto* pos     = static_cast<const std::int32_t*>(positions.data);
    const auto* count   = static_cast<const std::int32_t*>(counts.data);
    const auto* rows    = static_cast<const std::int32_t*>(table_rows.data);
    const auto* tables  = static_cast<const std::int32_t*>(cache.block_tables.data);

    const dim3 grid(1 + (plan.max_count - 1) / 4, k.ne[3], 1);
    kv_cache_append_prefix_paged_kernel<<<grid, kBlock, 0, stream>>>(
        input_k, input_v, pos, count, rows, cache_k, cache_v, tables, cache.k_pages.ne[2],
        cache.block_tables.ne[0], plan.min_count, plan.max_count, plan.tokens);
    CUDA_CHECK(cudaGetLastError());
}

void launch_cyclic(const Tensor& k, const Tensor& v, const Tensor& positions, const Tensor& counts,
                   const Tensor& lanes, CyclicKVCacheLayerView cache,
                   const KVCacheAppendPrefixPlan& plan, std::uint32_t window,
                   cudaStream_t stream) {
    validate_plan(k, plan);
    if (plan.max_count == 0) return;
    auto* cache_k       = static_cast<__nv_bfloat16*>(cache.k.data);
    auto* cache_v       = static_cast<__nv_bfloat16*>(cache.v.data);
    const auto* input_k = static_cast<const __nv_bfloat16*>(k.data);
    const auto* input_v = static_cast<const __nv_bfloat16*>(v.data);
    const auto* pos     = static_cast<const std::int32_t*>(positions.data);
    const auto* count   = static_cast<const std::int32_t*>(counts.data);
    const auto* lane    = static_cast<const std::int32_t*>(lanes.data);
    const int padded    = static_cast<int>(cache.padded_capacity);

    const dim3 grid(1 + (plan.max_count - 1) / 4, k.ne[3], 1);
    const auto launch_window = [&]<int Window>() {
        kv_cache_append_prefix_cyclic_kernel<Window><<<grid, kBlock, 0, stream>>>(
            input_k, input_v, pos, count, lane, cache_k, cache_v, plan.min_count, plan.max_count,
            plan.tokens, padded);
        CUDA_CHECK(cudaGetLastError());
    };
    switch (window) {
    case 2048:
        launch_window.template operator()<2048>();
        break;
    case 4096:
        launch_window.template operator()<4096>();
        break;
    default:
        throw std::invalid_argument("kv_cache_append_prefix: unsupported cyclic window");
    }
}

} // namespace

void kv_cache_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                            PagedKVLayerView cache, cudaStream_t stream) {
    const KVCacheAppendDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch_full<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, stream);
        return;
    }
    launch_full<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, stream);
}

void kv_cache_append_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  PagedKVBatchLayerView cache, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const KVCacheAppendBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
            launch_full<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, stream);
            return;
        }
        launch_full<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

KVCacheAppendPrefixPlan
kv_cache_append_prefix_resolve_plan(std::int32_t tokens,
                                    KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (tokens < 1) {
        throw std::invalid_argument("kv_cache_append_prefix plan: T must be positive");
    }
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument("kv_cache_append_prefix plan: invalid execution envelope");
    }
    return {
        .tokens    = tokens,
        .min_count = static_cast<std::int32_t>(envelope.min_count),
        .max_count = static_cast<std::int32_t>(envelope.max_count),
    };
}

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& table_rows,
                                   PagedKVBatchLayerView cache, const KVCacheAppendPrefixPlan& plan,
                                   cudaStream_t stream) {
    launch_paged(k, v, positions, counts, table_rows, cache, plan, stream);
}

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& lanes,
                                   CyclicKVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, std::uint32_t window,
                                   cudaStream_t stream) {
    launch_cyclic(k, v, positions, counts, lanes, cache, plan, window, stream);
}

void kv_cache_append_e8_lattice_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                       PagedKVLayerView cache, const E8KvLatticeTables& tables,
                                       cudaStream_t stream) {
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("kv_cache_append_e8_lattice_launch: k/v must be BF16");
    }
    const KVCacheAppendDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    dispatch_e8_narrow(k, v, positions, cache, metadata, tables, stream);
}

// THE HOST-VISIBLE BRIDGE (declared in `launch.h`). This is the ONE place the raw device
// addresses the public admission carries are cast to the codec's own types, so no host TU
// ever names `E8KvLatticeTables`. It admits nothing and checks no geometry: the op layer
// owns every admission decision, and this function is transport plus a null guard, because
// it is also reachable directly by an in-tree caller.
void kv_cache_append_e8_lattice_tables_launch(const Tensor& k, const Tensor& v,
                                              const Tensor& positions, PagedKVLayerView cache,
                                              E8KvAppendTables tables, cudaStream_t stream) {
    if (tables.stage1 == nullptr || tables.stage2 == nullptr) {
        // The arm below raises its own refusal naming itself, and the op-level overload
        // raises a third naming `kv_cache_append`. This guard is repeated here so that a
        // DIRECT caller of this bridge meets the arm's own wall rather than reaching a
        // launch with a null handle; the three messages differ because they name three
        // different entry points, and each is pinned separately by the admission test.
        throw std::invalid_argument(
            "kv_cache_append_e8_lattice_launch: the e8 lattice tables are not supplied; the "
            "codec of record needs its 6 428 B of stage-1/stage-2 tables, and a null handle "
            "would fault on device rather than refuse here");
    }
    const E8KvLatticeTables codec_tables{
        static_cast<const E8LatticeStage1*>(tables.stage1),
        static_cast<const E8LatticeStage2*>(tables.stage2)};
    kv_cache_append_e8_lattice_launch(k, v, positions, cache, codec_tables, stream);
}

void kv_cache_append_e8_lattice_batch_launch(const Tensor& k, const Tensor& v,
                                             const Tensor& positions, const Tensor& valid_columns,
                                             const Tensor& table_rows,
                                             PagedKVBatchLayerView cache,
                                             const E8KvLatticeTables& tables,
                                             cudaStream_t stream) {
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument(
            "kv_cache_append_e8_lattice_batch_launch: k/v must be BF16");
    }
    const auto launch = [&]<bool Masked>() {
        const KVCacheAppendBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        dispatch_e8_narrow(k, v, positions, cache, metadata, tables, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
