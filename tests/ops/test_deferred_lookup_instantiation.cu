// tests/ops/test_deferred_lookup_instantiation.cu  (SPLIT7, landed 2026-09-18)
//
// WHY THIS FILE EXISTS
//   The seventh split: a name CALLED inside a kernel-template body that is DEFINED NOWHERE in the
//   tree (causal_small_t_quantized_active_splits), hidden behind deferred two-phase lookup because
//   the enclosing kernel template was never instantiated. A TU that merely #includes such a header
//   never looks the name up, so it compiles CLEAN -- the header TU is the instrument that produces
//   the false green. The fix for that class is to FORCE the instantiation, which is what this TU
//   does: it takes the address of the kernel template-ids inside host function templates and then
//   explicitly instantiates those host templates. Compiling this TU is the test; main() only has to
//   exist for the target shape used by the sibling contract tests in this directory.
//
// COVERAGE -- stated exactly, because a check that claims more than it has is the same defect one
// level up:
//   COVERS: every dependent name in the BODY of
//     - causal_attention_small_t_k8v4_tiled_kernel   and ..._k8v4_reduce_output_kernel
//     - causal_attention_small_t_nvfp4_tiled_kernel  and ..._nvfp4_reduce_output_kernel
//   for two real specializations per kernel (TokenTile == 1 and TokenTile == 6, the two branch
//   shapes the launchers dispatch: RowTiles/Bc/Warps/MinBlocks differ between them) for both real
//   geometries CausalD256H24Kv4 and CausalD256H16Kv2. A name used unconditionally anywhere in the
//   body is looked up by any one of these instantiations; that is why one TU per family is enough.
//   DOES NOT COVER:
//     - names reachable only in an `if constexpr` branch that no instantiated specialization takes
//       (a dependent call in a DISCARDED branch is still never looked up);
//     - the other landed borrow families (kvarn, hq, rk2v4e8, prompt_*, kv_cache/append, ...) --
//       each needs its own forcing TU of this shape;
//     - anything that only a full build or a link could see (this check is compile-only, no link,
//       no device code is launched);
//     - the launcher TU itself (small_t_k8v4.cu / small_t_nvfp4.cu): its own two incompatibilities
//       with our PagedKVBatchLayerView (row 5 of the landing order) are NOT addressed here.
//
// NOT REGISTERED in tests/CMakeLists.txt. To register it, the shape that works while the engine's
// device link is broken is the one the sibling contract tests use:
//     add_executable(ninfer_deferred_lookup_instantiation_test
//                    ops/test_deferred_lookup_instantiation.cu)
//     target_include_directories(ninfer_deferred_lookup_instantiation_test PRIVATE
//       ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
//       ${PROJECT_SOURCE_DIR}/third_party ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
//     target_link_libraries(ninfer_deferred_lookup_instantiation_test PRIVATE CUDA::cudart)
//     add_test(NAME ninfer_deferred_lookup_instantiation_test
//              COMMAND ninfer_deferred_lookup_instantiation_test)
// It is not registered by this line because the fleet's engine link was in flight and adding a
// target changes the build graph for every other line.

#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_k8v4.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_nvfp4.cuh"

namespace ninfer::ops {

// ---- k8v4 -----------------------------------------------------------------------------------
// Template arguments mirror the launcher's own arithmetic (small_t_k8v4.cu: RowTiles =
// (TokenTile*GroupSize + 15)/16, Warps = RowTiles == 3 ? 12 : 8, KeyBlock = TokenTile == 1 ? 32 :
// 64, MinBlocks = TokenTile == 1 ? 2 : 1, DynamicArena = true).
template <typename Geometry, typename CacheInput>
__host__ void split7_force_k8v4_instantiation() {
    auto tiled_t1 = &causal_attention_small_t_k8v4_tiled_kernel<Geometry, 1, 8, 2, 32, true, false,
                                                                false, CacheInput>;
    auto tiled_t6 = &causal_attention_small_t_k8v4_tiled_kernel<Geometry, 6, 12, 1, 64, true, false,
                                                                 false, CacheInput>;
    auto reduce_plain = &causal_attention_small_t_k8v4_reduce_output_kernel<Geometry, false, false,
                                                                            false>;
    auto reduce_masked = &causal_attention_small_t_k8v4_reduce_output_kernel<Geometry, true, true,
                                                                             true>;
    (void)tiled_t1;
    (void)tiled_t6;
    (void)reduce_plain;
    (void)reduce_masked;
}
template __host__ void split7_force_k8v4_instantiation<CausalD256H24Kv4, CausalAppendInput>();
template __host__ void split7_force_k8v4_instantiation<CausalD256H16Kv2, CausalAppendInput>();

// ---- nvfp4 ----------------------------------------------------------------------------------
// small_t_nvfp4.cu: Warps = RowTiles == 3 ? 12 : 8, KeyBlock = 32, MinBlocks = RowTiles <= 2 ? 2 : 1.
template <typename Geometry, typename CacheInput>
__host__ void split7_force_nvfp4_instantiation() {
    auto tiled_t1 = &causal_attention_small_t_nvfp4_tiled_kernel<Geometry, 1, 8, 2, 32, true, false,
                                                                 false, CacheInput>;
    auto tiled_t6 = &causal_attention_small_t_nvfp4_tiled_kernel<Geometry, 6, 12, 1, 32, true, false,
                                                                 false, CacheInput>;
    auto reduce_plain = &causal_attention_small_t_nvfp4_reduce_output_kernel<Geometry, false, false,
                                                                             false>;
    auto reduce_masked = &causal_attention_small_t_nvfp4_reduce_output_kernel<Geometry, true, true,
                                                                             true>;
    (void)tiled_t1;
    (void)tiled_t6;
    (void)reduce_plain;
    (void)reduce_masked;
}
template __host__ void split7_force_nvfp4_instantiation<CausalD256H24Kv4, CausalAppendInput>();
template __host__ void split7_force_nvfp4_instantiation<CausalD256H16Kv2, CausalAppendInput>();

} // namespace ninfer::ops

int main() {
    // Compiling this TU IS the check. Nothing is launched, no device is touched, no ninfer library
    // is linked -- the same shape as tests/ops/test_gqa_split_contract.cu.
    return 0;
}
