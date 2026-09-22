// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : cometkim/ninfer
// Branch      : feat/1m-context
// Commit      : 2cc56b5db39f951c1e91b2caec8b39274d795682
// Source path : src/ops/softmax_attention/dense/causal_cache/small_t_hq_h24.cu
// sha256(src) : 52e3054b0e6531b510c3e3076a1cec080db79ff121face52de781370ed73138d
// Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits (rk4v4-e8 = rotated K 4b, V 4b, rk4v4 in the mix), which is a DIFFERENT axis from src/kvcfg/kv_formats.h's tier vocabulary (hot/tail/cold, Rk4v4).
#include "ops/softmax_attention/dense/causal_cache/small_t_hq_tc_launch.h"

namespace ninfer::ops::detail {

template void causal_attention_small_t_hq_launch_for<CausalD256H24Kv4, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

template void causal_attention_small_t_hq_launch_for<CausalD256H24Kv4, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
