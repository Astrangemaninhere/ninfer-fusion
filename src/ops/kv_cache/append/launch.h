#pragma once

#include "ninfer/ops/kv_cache_append.h"

namespace ninfer::ops::detail {

void kv_cache_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                            PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  PagedKVBatchLayerView cache, cudaStream_t stream);

// THE HOST-VISIBLE BRIDGE TO THE NARROW e8 K-PLANE ARM.
//
// The arm itself takes an `E8KvLatticeTables` (the codec of record's 6 428 B device table
// handle) and is declared in `ops/kv_cache/append/e8_lattice_narrow_kernel.cuh`. That type
// lives in a CUDA-only header, so it cannot appear in THIS header -- which `kv_cache_append.cpp`
// includes and the build compiles with the host compiler. This bridge takes the same two device
// addresses as raw pointers (the `E8KvAppendTables` the public admission already carries) and is
// the single place the cast to the codec's types happens.
//
// It is DEFINED in `launch.cu` (a CUDA TU, already in `ninfer_ops`) by forward-declaring nothing
// new: `launch.cu` includes `e8_lattice_narrow_kernel.cuh` and calls the arm directly.
//
// What it does NOT do: it does not admit anything, does not check the pool geometry, and does not
// fall back for a non-e8 dtype. Every admission decision stays in the op layer; this is transport.
// A null handle is refused here as well as at the arm, because this function is also reachable
// directly by an in-tree caller that is not going through the op-level overload.
void kv_cache_append_e8_lattice_tables_launch(const Tensor& k, const Tensor& v,
                                              const Tensor& positions, PagedKVLayerView cache,
                                              E8KvAppendTables tables, cudaStream_t stream);

struct KVCacheAppendPrefixPlan {
    std::int32_t tokens;
    std::int32_t min_count;
    std::int32_t max_count;
};

[[nodiscard]] KVCacheAppendPrefixPlan
kv_cache_append_prefix_resolve_plan(std::int32_t tokens,
                                    KVCacheAppendPrefixExecutionEnvelope envelope);

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& table_rows,
                                   PagedKVBatchLayerView cache, const KVCacheAppendPrefixPlan& plan,
                                   cudaStream_t stream);
void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& lanes,
                                   CyclicKVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, std::uint32_t window,
                                   cudaStream_t stream);

// THE NARROW e8 K-PLANE APPEND ARM (W3 / W2) IS DECLARED IN ITS OWN HEADER, NOT HERE.
// `ops/kv_cache/append/e8_lattice_narrow_kernel.cuh` declares and
// `ops/kv_cache/append/launch.cu` defines `kv_cache_append_e8_lattice_launch` and
// `..._batch_launch`. They take an `E8KvLatticeTables` (the codec of record's 6 428 B
// table handle, supplied by the caller), and that type is only visible to a CUDA TU --
// while THIS header is included by `kv_cache_append.cpp`, which the build compiles with
// the host compiler. Declaring them here would make the host TU depend on
// `ops/kernel/e8_lattice_kv_plane.cuh`'s `__device__` arm. Hence the split, and hence the
// raw-handle bridge above, which is the only thing this header may name.

} // namespace ninfer::ops::detail
