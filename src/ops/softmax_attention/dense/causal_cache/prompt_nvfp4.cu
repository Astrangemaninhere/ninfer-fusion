// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : sergiuszm/ninfer-4090
// Branch      : rtx4090-port
// Commit      : 1bd56c9a1bdf457c6188391a9385d44d86e953aa
// Source path : src/ops/softmax_attention/dense/causal_cache/prompt_nvfp4.cu
// sha256(src) : 660209c70494e67f8410da51e6044a69f2a8f7523b9df5e433704a09d930054b
// Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits (rk4v4-e8 = rotated K 4b, V 4b, rk4v4 in the mix), which is a DIFFERENT axis from src/kvcfg/kv_formats.h's tier vocabulary (hot/tail/cold, Rk4v4).
// ninfer::ops::detail - public-op composition for group-16 NVFP4 causal prompt attention.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"

namespace ninfer::ops::detail {

void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    Tensor& out, cudaStream_t stream) {
    causal_attention_prompt_nvfp4_kernel_launch(q, positions, scale, cache, out, stream);
}

void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache, Tensor& out,
                                          cudaStream_t stream) {
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    causal_attention_prompt_nvfp4_batch_kernel_launch(q, positions, valid_columns, table_rows,
                                                      scale, cache, out, stream);
}

} // namespace ninfer::ops::detail
