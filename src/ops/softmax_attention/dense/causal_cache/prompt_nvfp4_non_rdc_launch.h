// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : sergiuszm/ninfer-4090
// Branch      : rtx4090-port
// Commit      : 1bd56c9a1bdf457c6188391a9385d44d86e953aa
// Source path : src/ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h
// sha256(src) : d93e69856499d5abee65b8aae3eb11ce5766a4d032fbe36422a77128f5d63ca1
// Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits (rk4v4-e8 = rotated K 4b, V 4b, rk4v4 in the mix), which is a DIFFERENT axis from src/kvcfg/kv_formats.h's tier vocabulary (hot/tail/cold, Rk4v4).
#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void causal_attention_prompt_nvfp4_kernel_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 Tensor& out, cudaStream_t stream);

void causal_attention_prompt_nvfp4_batch_kernel_launch(const Tensor& q, const Tensor& positions,
                                                       const Tensor& valid_columns,
                                                       const Tensor& table_rows, float scale,
                                                       const PagedKVBatchLayerView& cache,
                                                       Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
