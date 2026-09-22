#pragma once

// ---------------------------------------------------------------------------
// qpn_host.h -- the host entry of the QPN W4A16 fp16 fallback family.
//
// This declaration existed nowhere before 2026-09-17: src/ops/linear/qpn/qpn_host.cu
// defined gemm_qpn/gemm_qpn_simt and no header declared them, because nothing called them
// -- the directory was in no CMake target. It has one consumer with a purpose beyond
// calling it: tests/test_arch_generic_fallback.cpp takes the address of gemm_qpn, so the
// build fact NINFER_HAVE_QPN (src/CMakeLists.txt, published on ninfer_core) and the OBJECT
// FILE that makes it true have to arrive together. A definition without the source is a
// LINK error instead of a route that names a kernel the binary cannot launch.
//
// The signature is the one qpn_host.cu already had; only the declaration moved out.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

namespace ninfer::ops::qpn {

// M-dispatch entry. `codes`/`scales` must already be in the qpn_prepack layout (bit-exact
// with the marlin _qpn_prepack reference -- see the layout header comment in
// qpn_kernels.cuh). Bands: M 1..3 SIMT, M 4..8 skinny_nvfp4_qpn<1>, M 9..16
// skinny_nvfp4_qpn<2>. Throws std::invalid_argument on a shape or band it cannot serve, so
// a caller can never get a silent wrong answer from a geometry the prepack does not
// describe.
//
// M >= 17 IS NOT SERVED HERE, AND THAT IS A LAYOUT FACT RATHER THAN A MISSING KERNEL. The
// wide-M band's kernel is skinny_nvfp4_wmma<WN, WM, KC> (qpn_kernels.cuh, the "wmma (M>=9)"
// kernel whose frontier row reads "simt M<=3, qpn 4..16, wmma 17..64") and it consumes the
// CHECKPOINT-NATIVE planes -- codes u8 [N][K/2], scales u8 [N][K/16] -- not the qpn_prepack
// permutation this entry's precondition names. The two are the same published values in a
// different byte order, so an M-keyed layout switch inside this function is exactly the
// silent-corruption shape the family refuses. The wide band gets its own entry below.
void gemm_qpn(const void* x_half, const void* codes, const void* scales, float gscale,
              void* y_half, int m, int k, int n, cudaStream_t stream);

// THE WIDE-M BAND (M 17..64), AND THE SECOND LAYOUT.
//
// The kernel was never missing: skinny_nvfp4_wmma<WN, WM, KC> has been in qpn_kernels.cuh
// since the port, its frontier row is "simt M<=3, qpn 4..16, wmma 17..64"
// (qpn_sweep_20260810), and it is compiled into ninfer_ops because this file is. What was
// missing was a host arm, so gemm_qpn refused M >= 17 by naming a band it could not launch
// -- and src/core/kernel_route.h turned that into a route-table row saying the FAMILY lacks
// the band (and that vLLM's marlin would have covered it). Neither statement was true of
// this tree.
//
// `codes`/`scales` ARE THE CHECKPOINT-NATIVE PLANES: codes u8 [N][K/2] with two e2m1 codes
// per byte (low nibble = even k), scales u8 [N][K/16] fp8-e4m3, one per 16-k group. That is
// the SAME layout an NVFP4 artifact already stores (ops/linear/nvfp4/nvfp4_format.cpp,
// validate_nvfp4_weight), which is the layout qpn_prepack_proto.py permutes -- so for THIS
// band the "native -> qpn_prepack weight converter" that qpn_arch_route.h lists as one of
// the two missing pieces is NOT needed. The activation/output dtype is still `half`, so an
// engine whose tensors are __nv_bfloat16 still needs the bf16 -> fp16 step.
//
// WHAT DIFFERS FROM THE M 1..16 BANDS: the mma shape and therefore the reduction order
// (nvcuda::wmma m16n16k16 fragments vs m8n8k4 quadpairs, which src/core/arch_caps.h
// records as two DIFFERENT channels of the same family). Same W4A16 arithmetic on the same
// published codes and scales -- a different kernel, not a different arithmetic. The route
// table decides which rungs that channel is hardware on; this function only enforces the
// kernel's own limits (M 1..64, K % 128 == 0, N % 32 == 0).
void gemm_qpn_wmma_native(const void* x_half, const void* codes, const void* scales,
                          float gscale, void* y_half, int m, int k, int n,
                          cudaStream_t stream);

// The tensor-core-free SIMT arm, M 1..3 only.
void gemm_qpn_simt(const void* x_half, const void* codes, const void* scales, float gscale,
                   void* y_half, int m, int k, int n, cudaStream_t stream);

} // namespace ninfer::ops::qpn
