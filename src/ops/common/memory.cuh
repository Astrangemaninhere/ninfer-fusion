#pragma once

#if defined(__AMDGCN__) || defined(__HIP_PLATFORM_AMD__)
// NINFER_GFX906_COMPAT -- the AMD arm of this include, and why it is here at all.
// MEASURED: ROCm 7.2.4 ships NO cuda_pipeline.h (0 occurrences anywhere in its include tree), so
// the include cannot be spelled on this target.  The pipeline primitives below carry their own AMD
// arm instead, which is what this file's own sibling,
// src/compat/gfx906/include/cuda_pipeline.h, already states verbatim:
//   "ops/common/memory.cuh replaces every pipeline primitive with synchronous loads under
//    NINFER_GFX906_COMPAT; nothing is provided here."
// (That comment described a contract this file did not discharge: NINFER_GFX906_COMPAT occurred 0
// times in it.  This arm is that contract, in this file, where the port layer said it would be.)
#else
#include <cuda_pipeline.h>
#endif
#include <cuda_runtime.h>


// ---------------------------------------------------------------------------
// Per-target feature gate + the one named trap (MULTIARCH).
//
// cp.async is sm_80+; HEAD emits it unconditionally, so a sm_70/sm_75 build
// fails here. The macro name repeats the `guard` column of
// capability_probe_spec() (src/core/device_capabilities.h).
//
// The trap helper lives in THIS file because memory.cuh is the leaf include:
// mma.cuh does `#include "ops/common/memory.cuh"`, so a helper declared in
// mma.cuh is not visible to the bodies below.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define NINFER_MEMORY_HAS_CP_ASYNC 1
#endif

namespace ninfer::ops {

// A trap is the honest behaviour for an instruction family this target cannot
// assemble: the alternative (leaving the destination untouched) would report
// success while copying nothing.
[[noreturn]] __device__ __forceinline__ void unsupported_instruction_trap() {
#if defined(__AMDGCN__) || defined(__HIP_PLATFORM_AMD__)
    // PARAMXLATE: the AMDGCN spelling. MEASURED on gfx906 through the real device
    // pipeline: __trap() itself is UNDECLARED on this target while __builtin_trap()
    // compiles and lowers to `s_trap 2` (+ `s_endpgm`), which is AMDGCN's trap.
    __builtin_trap();
#else
    __trap();
#endif
    __builtin_unreachable();
}

enum class Cache { ca, cg };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
#if defined(__AMDGCN__) || defined(__HIP_PLATFORM_AMD__)
    // PARAMXLATE2 -- family F: the SHARED-SPACE ADDRESS parameter, translated.
    //
    //   PTX/NV   __cvta_generic_to_shared(p) == `cvta.to.shared`.  A REAL conversion: the
    //            generic address of a shared object is not its shared-window address.
    //   AMDGCN   the operation has no AMD spelling AND NEEDS NONE, because on this target the
    //            local<->generic conversion is the IDENTITY.  The evidence:
    //              * MEASURED: ROCm 7.2.4 has ZERO occurrences of `cvta` in its whole include/
    //                tree, and clang 21's and 22's HIP wrappers add none.  The identifier is
    //                simply undeclared, so this line was the FIRST wall on the AMD path -- a
    //                parse failure, before any instruction is ever considered.
    //              * READ, vendor source, hip/amd_detail/amd_device_functions.h:666:
    //                  __device__ inline void* __local_to_generic(void* p) { return p; }
    //                the vendor's own local->generic conversion returns its argument UNCHANGED,
    //                so the generic address of a shared object IS its LDS byte offset -- which
    //                is precisely the operand the LDS instructions take.
    //              * READ, vendor source, hip/amd_detail/device_library_decls.h:127:
    //                  __device__ inline static __local void* __to_local(unsigned x)
    //                  { return (__local void*)x; }
    //                an LDS pointer is constructible from a BARE u32: the same statement again.
    //
    // MAPPED PARAMETERS (in / out)
    //   operand width   : input a 64-bit generic pointer, output u32, on BOTH targets.  The
    //                     truncation is the same operation on both, and on AMD it is lossless by
    //                     construction, because an LDS offset is a 32-bit quantity.
    //   dynamic shared  : the one parameter that looks like it needs care.  A dynamic
    //                     `extern __shared__` segment begins at the group segment base
    //                     (vendor: __get_dynamicgroupbaseptr() == __to_local(__builtin_amdgcn_
    //                     groupstaticsize())), NOT at 0.  It still needs NO `+ base` term here,
    //                     for the same reason: that base is itself an LDS offset and the generic
    //                     pointer already carries it.  This is not a hypothetical caller --
    //                     MEASURED: 40 files in this tree declare dynamic shared, and 33 of them
    //                     also call smem_addr.
    //   address space   : addrspace(3) on both targets.  It is not written here because the
    //                     result is consumed as an INTEGER operand of inline asm.  Typing the
    //                     intermediate as __local is legal and MEASURED to yield the same value
    //                     but 64 more bytes of object, and no consumer reads the type.
    //   qualifiers      : none, on either target.  No cache policy, no scope, no ordering --
    //                     `cvta.to.shared` is a pure address conversion, and this helper is a
    //                     pure function of its argument, as it already was on the CUDA side.
    //
    // NOT CLAIMED: that any downstream construct consumes this offset.  On this target the
    // LDS consumers are exactly the families that REFUSE (see the report's price list); this
    // arm makes the file PARSE, which turns a build wall into a named refusal.
    return static_cast<unsigned>(reinterpret_cast<unsigned long long>(ptr));
#else
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
#endif
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
#ifdef NINFER_MEMORY_HAS_CP_ASYNC
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
#else
    (void)smem_dst; (void)gmem_src;
    unsupported_instruction_trap();
#endif
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
#ifdef NINFER_MEMORY_HAS_CP_ASYNC
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
#else
    (void)smem_dst; (void)gmem_src; (void)src_bytes;
    unsupported_instruction_trap();
#endif
}

__device__ __forceinline__ void cp_commit() {
#ifdef NINFER_MEMORY_HAS_CP_ASYNC
    asm volatile("cp.async.commit_group;\n");
#else
    // Traps, like every other arm of this gate, and for the same reason cp_wait below
    // states: an empty body here would SILENTLY DROP the group boundary on a target that
    // has no async copy at all, so a downstream cp_wait would look satisfied while
    // nothing was ever in flight. That is exactly the "compute a wrong answer" this
    // patch's header says it refuses. (An earlier revision of this patch left cp_commit
    // empty and only cp_wait trapping; the asymmetry was found by 对照线 MULTIARCH-X and
    // is closed here. Measured consequence of closing it: no TU that built before stops
    // building -- every caller of cp_commit() is inside the same `>= 800` gate as its
    // cp_async/cp_wait partner.)
    unsupported_instruction_trap();
#endif
}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
#ifdef NINFER_MEMORY_HAS_CP_ASYNC
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
#else
    // A trap, not a static_assert: a static_assert here only MOVES the failure.
    // Measured: with the static_assert, ops/linear/bf16/bf16_gemm_mma.cu still
    // failed for sm_75, because a real pipelined kernel calls cp_wait<1..7>.
    // Trapping lets the TU BUILD; the route selector is what keeps the kernel
    // from being reached, and a correct pre-sm_80 pipelined GEMM is a new kernel
    // (see the QPN route), not a guard.
    (void)0;
    unsupported_instruction_trap();
#endif
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
#if defined(__AMDGCN__) || defined(__HIP_PLATFORM_AMD__)
    // NINFER_GFX906_COMPAT -- the AMD arm.  ROCm declares NONE of __pipeline_* (MEASURED: 0
    // occurrences in the whole ROCm 7.2.4 include tree), so the honest behaviour is the trap this
    // file already defines at the top, for exactly this situation:
    //   "A trap is the honest behaviour for an instruction family this target cannot assemble: the
    //    alternative (leaving the destination untouched) would report success while copying
    //    nothing."
    // NOT a silent substitute: a caller that reaches this arm stops here rather than computing a
    // wrong number.
    (void)smem_dst; (void)gmem_src;
    unsupported_instruction_trap();
#else
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
#endif
}

__device__ __forceinline__ void pipe_commit() {
#if defined(__AMDGCN__) || defined(__HIP_PLATFORM_AMD__)
    // NINFER_GFX906_COMPAT -- see pipe_copy: trapping, not empty.  An empty body would SILENTLY DROP
    // the group boundary, so a downstream pipe_wait would look satisfied while nothing was ever in
    // flight -- the same asymmetry the cp_async family in this file already closed by name.
    unsupported_instruction_trap();
#else
    __pipeline_commit();
#endif
}

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
#if defined(__AMDGCN__) || defined(__HIP_PLATFORM_AMD__)
    // NINFER_GFX906_COMPAT -- the AMD arm.  A static_assert here would only MOVE the failure (the
    // same reasoning the cp_wait arm above records); trapping lets the TU BUILD and keeps the
    // channel from being reached silently.
    (void)0;
    unsupported_instruction_trap();
#else
    __pipeline_wait_prior(Groups);
#endif
}

} // namespace ninfer::ops
