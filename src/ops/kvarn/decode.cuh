// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : MirkoCovizzi/ninfer-rtx5090-mobile
// Branch      : feat/kvarn-production
// Commit      : 750aac554d52b758c3ee33eff81f08bf1dceb201
// Source path : src/ops/kvarn/decode.cuh
// sha256(src) : c778d02268425dfe0f65bb0cf26b19d7e5aae6170222344a3121c9c23b02809a
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
#include <cuda_bf16.h>

namespace ninfer::ops::kvarn {

// Rotated current-chunk values remain unquantized until its attention has completed.
struct CurrentKV {
    const __nv_bfloat16* key      = nullptr;
    const __nv_bfloat16* value    = nullptr;
    const std::int32_t* positions = nullptr;
    std::int32_t width            = 0;
};

void decode_attention(const Tensor& query, const Tensor& positions, const Tensor& valid_columns,
                      const Tensor& table_rows, float scale, KvarnPagedBatchLayerView cache,
                      CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                      Tensor& output, cudaStream_t stream, CurrentKV current = {});

} // namespace ninfer::ops::kvarn
