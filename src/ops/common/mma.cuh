#pragma once

#include "ops/common/memory.cuh"

// ---------------------------------------------------------------------------
// Per-target feature gates (MULTIARCH: the cross-arch build wall).
//
// Every asm in this file is emitted UNCONDITIONALLY in HEAD 3944a53, so a build
// for a target that cannot assemble one of them fails at ptxas and takes the
// whole arch down with it. This file is reached by ninfer_core through
// core/device_probe.cu, so nothing in the tree could be built for such a target.
// Measured on HEAD: src/ops/common/mma.cuh@846ed94c160608ba has ZERO occurrences
// of `CUDA_ARCH` (single-file grep rc=1), while emitting kind::f8f6f4 and
// kind::mxf4nvf4. Measured consequence: `nvcc -cubin -arch=sm_75` on a TU that
// calls these helpers exits 255 with
//   "Feature '.kind::f8f6f4' not supported on .target 'sm_75'".
//
// The macro names repeat the `guard` column of capability_probe_spec()
// (src/core/device_capabilities.h) on purpose, so the probe's "compiled out of
// this build" answer and this file's absence cannot drift apart.
//
// When a family is absent the helper TRAPS. It does not compute a wrong answer,
// and it is NOT deleted: the kernel symbol stays, the route selector
// (src/core/kernel_route.h) is what keeps it from being reached, and it names
// the missing route rather than the missing card.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
#define NINFER_MMA_HAS_LDMATRIX 1
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define NINFER_MMA_HAS_M16N8K16_TC 1
#endif
// The 'f' (family) targets need their own clause, and it is NOT the same clause
// as the 'a' targets. MEASURED on nvcc 13.3 with a preprocessor probe: sm_120f
// and sm_121f define __CUDA_ARCH_FAMILY_SPECIFIC__ and NEITHER
// __CUDA_ARCH_FEAT_SM120_ALL NOR __CUDA_ARCH_FEAT_SM121_ALL; sm_100f/110f also
// define __CUDA_ARCH_FAMILY_SPECIFIC__ only. So keying a guard on the _ALL macros
// alone compiles the Blackwell kernels OUT of a family build (measured: the first
// version of this patch did exactly that -- f8f6f4=0, trap=1 for sm_120f, while
// HEAD emits both forms there). So the family clause is unconditional on
// __CUDA_ARCH_FAMILY_SPECIFIC__ for THIS form: measured, the f8f6f4 kind IS
// assemblable on all five family targets (sm_100f/103f/110f/120f/121f). Only the
// mxf4nvf4 form below needs the extra __CUDA_ARCH__ >= 1200 bound.
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) ||                 \
    defined(__CUDA_ARCH_FEAT_SM110_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) ||                 \
    defined(__CUDA_ARCH_FEAT_SM121_ALL) ||                                                        \
    defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
#define NINFER_MMA_HAS_KIND_F8F6F4 1
#endif
// MEASURED: kind::mxf4nvf4.block_scale assembles only on sm_120a/121a AND on the
// 120/121 family targets. sm_100a/103a/110a all reject it (ptxas: "Instruction
// 'mma with block scale' not supported on .target 'sm_100a'"), and so do
// sm_100f/103f/110f. The family clause therefore keeps the __CUDA_ARCH__ >= 1200 bound.
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||                \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH__ >= 1200))
#define NINFER_MMA_HAS_KIND_MXF4NVF4 1
#endif

namespace ninfer::ops {

// unsupported_instruction_trap() lives in ops/common/memory.cuh, the leaf include.

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
#ifdef NINFER_MMA_HAS_LDMATRIX
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
#else
    (void)addr; (void)r0; (void)r1;
    unsupported_instruction_trap();
#endif
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
#ifdef NINFER_MMA_HAS_LDMATRIX
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#else
    (void)addr; (void)r0; (void)r1; (void)r2; (void)r3;
    unsupported_instruction_trap();
#endif
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
#ifdef NINFER_MMA_HAS_LDMATRIX
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
#else
    (void)addr; (void)r0; (void)r1;
    unsupported_instruction_trap();
#endif
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
#ifdef NINFER_MMA_HAS_LDMATRIX
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#else
    (void)addr; (void)r0; (void)r1; (void)r2; (void)r3;
    unsupported_instruction_trap();
#endif
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#ifdef NINFER_MMA_HAS_M16N8K16_TC
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    (void)c0; (void)c1; (void)c2; (void)c3;
    unsupported_instruction_trap();
#endif
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#ifdef NINFER_MMA_HAS_M16N8K16_TC
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    (void)c0; (void)c1; (void)c2; (void)c3;
    unsupported_instruction_trap();
#endif
}

// ---------------------------------------------------------------------------
// THE NON-BLACKWELL fp16 FALLBACK CHANNEL (the QPN W4A16 GEMM's tensor core)
// ---------------------------------------------------------------------------
//
// This file's f16 form above is m16n8k16, which MEASURED requires sm_80. That single
// fact is why the tree had no fp16 GEMM it could assemble for sm_70/sm_75 at all, and
// why the QPN kernels -- which reach the tensor core through mma.sync.m8n8k4 precisely
// because it is the one form Volta has -- had no helper here to be written against.
//
// MEASURED, this line, hand-written PTX + ptxas (12.8 for sm_70, 13.3 otherwise), so
// the verdict is the assembler's and not a compiler's:
//
//   mma.m8n8k4.f16     70 OK  75 OK  80 OK  86 OK  89 OK  90 OK 100 OK 100a OK 120a OK
//   mma.m16n8k8.f16    70 X   75 OK  80 OK  86 OK  89 OK  90 OK 100 OK 100a OK 120a OK
//   mma.m16n8k16.f16   70 X   75 X   80 OK  ...
//
// TWO CONSEQUENCES, and the second is the one that matters for the fallback route:
//   * m8n8k4 is assemblable on EVERY rung of kArchLadder, sm_70 through sm_120a. It is
//     therefore the arch-generic channel: one implementation, no Volta special case.
//   * "assembles" is NOT "the tensor core executes it". On sm_70/sm_75 the SASS is
//     HMMA.884; from sm_80 up ptxas answers the same PTX with a CALL into a software
//     FFMA routine. That per-rung fact is MEASURED and recorded in kQpnMmaRungs
//     (src/core/arch_caps.h), and the route selector -- not this header -- is what
//     keeps an emulated rung off this channel. A helper cannot make that decision,
//     because it is a property of the rung, not of the instruction.
//
// THE TWO FORMS ARE **NOT** INTERCHANGEABLE. m8n8k4 and m16n8k8 take different
// accumulator and operand fragment geometries (8 f32 accumulator registers and two
// 32-bit A/B pairs versus 4 f32 and a four-plus-two A/B pair), so there is deliberately
// no auto-dispatching wrapper: picking between them is a KERNEL decision about its
// fragment layout, not a build decision. m16n8k8 is offered because it is Turing's
// native faster channel and this file had no way to name it; it is NOT yet consumed by
// any kernel in this tree, and the report says so rather than implying a route.
//
// The m8n8k4 form is written exactly as src/ops/linear/qpn/qpn_kernels.cuh:695 and :895
// emit it (the eight-register vector is the form ptxas accepts; the four-register form
// that the PTX manual's signature suggests is REJECTED with "Argument vector size
// mismatch", measured on all eight targets above).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#define NINFER_MMA_HAS_M8N8K4_F16 1
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
#define NINFER_MMA_HAS_M16N8K8_F16 1
#endif

// 8x8x4 fp16 -> f32, row.col. Volta's form, and the only fp16 tensor-core form that
// assembles on sm_70.
__device__ __forceinline__ void mma_f16_m8n8k4(float& c0, float& c1, float& c2, float& c3,
                                               float& c4, float& c5, float& c6, float& c7,
                                               unsigned a0, unsigned a1, unsigned b0,
                                               unsigned b1) {
#ifdef NINFER_MMA_HAS_M8N8K4_F16
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3), "+f"(c4), "+f"(c5), "+f"(c6),
                   "+f"(c7)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    (void)c0; (void)c1; (void)c2; (void)c3; (void)c4; (void)c5; (void)c6; (void)c7;
    (void)a0; (void)a1; (void)b0; (void)b1;
    unsupported_instruction_trap();
#endif
}

// 16x8x8 fp16 -> f32, row.col. Turing's native channel; sm_75 and up (measured).
__device__ __forceinline__ void mma_f16_m16n8k8(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned b0) {
#ifdef NINFER_MMA_HAS_M16N8K8_F16
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
#else
    (void)c0; (void)c1; (void)c2; (void)c3;
    (void)a0; (void)a1; (void)b0;
    unsupported_instruction_trap();
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#ifdef NINFER_MMA_HAS_M16N8K16_TC
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    (void)c0; (void)c1; (void)c2; (void)c3;
    unsupported_instruction_trap();
#endif
}

__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
#ifdef NINFER_MMA_HAS_KIND_F8F6F4
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    (void)c0; (void)c1; (void)c2; (void)c3;
    unsupported_instruction_trap();
#endif
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#ifdef NINFER_MMA_HAS_M16N8K16_TC
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    (void)c0; (void)c1; (void)c2; (void)c3;
    unsupported_instruction_trap();
#endif
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
#ifdef NINFER_MMA_HAS_KIND_MXF4NVF4
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1; (void)sfa; (void)sfb;
    (void)c0; (void)c1; (void)c2; (void)c3;
    unsupported_instruction_trap();
#endif
}

} // namespace ninfer::ops
