// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : MirkoCovizzi/ninfer-rtx5090-mobile
// Branch      : feat/kvarn-production
// Commit      : 750aac554d52b758c3ee33eff81f08bf1dceb201
// Source path : include/ninfer/ops/kvarn_attention.h
// sha256(src) : 3810a6116ddcf8fef57c8dafb1eedfeaa64e249a775eee32e8d2c1e93965ef42
// Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits (rk4v4-e8 = rotated K 4b, V 4b, rk4v4 in the mix), which is a DIFFERENT axis from src/kvcfg/kv_formats.h's tier vocabulary (hot/tail/cold, Rk4v4).
#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/kvarn.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

[[nodiscard]] std::size_t kvarn_attention_workspace_capacity_bytes(
    std::int32_t query_heads, CausalAttentionExecutionEnvelope envelope, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width);

// Appends and attends in the orthonormal KVarN frame. Current-chunk K/V stay unquantized throughout
// attention. Completed non-sink groups are encoded only after their tokens are committed;
// provisional groups remain in the tail until acceptance. Rejected suffixes are overwritable.
// Q/K/V are disposable. For single-row final-query prefill, Q may contain only the last query
// while K/V and positions contain the entire appended chunk.
void kvarn_attention(Tensor query, Tensor key, Tensor value, const Tensor& positions,
                     const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                     KvarnPagedBatchLayerView cache, bool provisional,
                     CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                     Tensor& output, cudaStream_t stream);

void kvarn_attention_cached(Tensor query, const Tensor& positions, const Tensor& kv_table_rows,
                            float scale, const KvarnPagedBatchLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& output, cudaStream_t stream);

void kvarn_kv_append(Tensor key, Tensor value, const Tensor& positions, const Tensor& valid_columns,
                     const Tensor& kv_table_rows, KvarnPagedBatchLayerView cache, bool provisional,
                     cudaStream_t stream);

// accepted_columns is an I32 prefix count per batch row. Completed non-sink groups in that prefix
// are encoded from their unquantized tails before their markers are retired. Sinks remain lossless.
void kvarn_commit_pages(const Tensor& positions, const Tensor& accepted_columns,
                        const Tensor& kv_table_rows, KvarnPagedBatchLayerView cache,
                        cudaStream_t stream);

// Settles completed live groups through the final committed frontier and re-establishes its
// writable tail. A historical partial group is decoded; an already-live partial tail is preserved.
// Runtime calls this after output publication has selected the actual prefix, not at licensing.
// Batches one to sixteen disjoint layer views at the same frontier without device scratch.
void kvarn_restore_tail(std::int32_t frontier, std::span<const KvarnPagedLayerView> layers,
                        cudaStream_t stream);

} // namespace ninfer::ops
