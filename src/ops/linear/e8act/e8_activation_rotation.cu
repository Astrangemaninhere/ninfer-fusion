// ninfer::ops - the activation-side Hadamard rotation, launch site.
//
// WHY THIS TU EXISTS AT ALL, AND WHY IT IS NOT A PUBLIC OP
// -------------------------------------------------------
// src/ops/linear/e8act/e8_activation_rotation.cuh is entirely `__device__ __forceinline__`,
// so a TU that merely INCLUDES it emits no device code. That is not a guess: the tree
// measured exactly this defect for its own E8 reader and wrote it down at
// src/CMakeLists.txt:170-174 ("a bare `nvcc -c` of an include-only TU exits 0, writes a
// 6,656-byte object, and nm shows no reader symbol"). This file therefore has to CALL the
// operator from a __global__, and that is its first job.
//
// Its second job is the LAUNCH SITE itself: it is the only place that knows how the
// [K, T] plane is partitioned into 64-blocks. Keeping the partition here, next to the
// operator rather than in a caller, is deliberate -- a caller that partitioned the plane
// differently (say, 64-blocks taken across T instead of along K) would produce a rotation
// that is still orthogonal and still compiles, and every downstream number would be wrong
// with nothing failing. The partition is a contract, so it lives in one file.
//
// WHAT IT IS NOT
//   * Not a `linear` dispatch arm. `linear` admits a Weight by NumericFormat
//     (src/artifact/reader.h:21-48); this operator is not a format and takes no Weight.
//   * Not wired to a runtime path. It has a launch site and a host prototype, and that is
//     all it has -- the same honesty the tree demands of its own reader
//     (src/ops/kernel/e8_lattice_kv_plane_inst.cu:28-30, "no caller").
//   * Not a numerical-exactness claim on BF16. The butterfly runs in FP32 and the RESULT is
//     rounded to BF16 on the way out, so the BF16 entry point is NOT an involution:
//     applying it twice returns each element to within one BF16 rounding of where it
//     started, not exactly. That is a property of the storage type, not of the operator,
//     and it is the same rounding the tree already accepts on the KV side. The F32 entry
//     point IS exact in the involution sense (up to one FP32 rounding per element).

#include "ops/linear/e8act/e8_activation_rotation.h"

#include "core/device.h"
#include "ops/linear/e8act/e8_activation_rotation.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::e8act {
namespace {

constexpr int kThreads = 256;

// One thread owns ONE 64-element block. Consecutive threads therefore take consecutive
// 64-blocks, so a full warp reads 32 * 64 * sizeof(T) = 8 KiB of contiguous memory.
__global__ __launch_bounds__(kThreads) void e8act_rotate_bf16_kernel(
        const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
        std::int32_t k, std::int64_t total_blocks) {
    const std::int64_t b = static_cast<std::int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (b >= total_blocks) { return; }
    const std::int64_t groups = k / kE8ActBlock;
    const std::int64_t g      = b % groups;
    const std::int64_t t      = b / groups;
    const std::int64_t off    = t * k + g * kE8ActBlock;

    float v[kE8ActBlock];
#pragma unroll
    for (int i = 0; i < kE8ActBlock; ++i) {
        v[i] = __bfloat162float(src[off + i]);
    }
    e8act_rotate_block_t(v);
#pragma unroll
    for (int i = 0; i < kE8ActBlock; ++i) {
        dst[off + i] = __float2bfloat16(v[i]);
    }
}

__global__ __launch_bounds__(kThreads) void e8act_rotate_f32_kernel(
        const float* __restrict__ src, float* __restrict__ dst, std::int32_t k,
        std::int64_t total_blocks) {
    const std::int64_t b = static_cast<std::int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (b >= total_blocks) { return; }
    const std::int64_t groups = k / kE8ActBlock;
    const std::int64_t g      = b % groups;
    const std::int64_t t      = b / groups;
    const std::int64_t off    = t * k + g * kE8ActBlock;

    float v[kE8ActBlock];
#pragma unroll
    for (int i = 0; i < kE8ActBlock; ++i) {
        v[i] = src[off + i];
    }
    e8act_rotate_block_t(v);
#pragma unroll
    for (int i = 0; i < kE8ActBlock; ++i) {
        dst[off + i] = v[i];
    }
}

// The refusal, written once. Both entry points take it, so neither can be the lenient one.
std::int32_t e8act_shape_of(const Tensor& x, const Tensor& out) {
    if (x.dtype != DType::BF16 && x.dtype != DType::FP32) {
        throw std::invalid_argument("e8act rotate: x must be BF16 or F32");
    }
    if (x.dtype != out.dtype) {
        throw std::invalid_argument("e8act rotate: x/out dtypes must match");
    }
    if (x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("e8act rotate: x/out must have shape [K,T]");
    }
    if (x.ne[0] != out.ne[0] || x.ne[1] != out.ne[1]) {
        throw std::invalid_argument("e8act rotate: x/out shapes must agree");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("e8act rotate: x/out must be contiguous");
    }
    const std::int32_t k = x.ne[0];
    if (!e8act_shape_supported(k)) {
        throw std::invalid_argument("e8act rotate: k must be a positive multiple of 64");
    }
    if (x.ne[1] <= 0) {
        throw std::invalid_argument("e8act rotate: t must be positive");
    }
    return k;
}

unsigned int e8act_grid(std::int32_t k, std::int32_t t) {
    const std::int64_t blocks = e8act_blocks(k, t);
    return static_cast<unsigned int>((blocks + kThreads - 1) / kThreads);
}

}   // namespace

void e8act_rotate_activation_launch(const Tensor& x, Tensor& out, cudaStream_t stream) {
    const std::int32_t k = e8act_shape_of(x, out);
    if (x.dtype != DType::BF16) {
        throw std::invalid_argument("e8act rotate: use e8act_rotate_activation_launch_f32 for F32");
    }
    e8act_rotate_bf16_kernel<<<e8act_grid(k, x.ne[1]), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data), k,
        e8act_blocks(k, x.ne[1]));
    CUDA_CHECK(cudaGetLastError());
}

void e8act_rotate_activation_launch_f32(const Tensor& x, Tensor& out, cudaStream_t stream) {
    const std::int32_t k = e8act_shape_of(x, out);
    if (x.dtype != DType::FP32) {
        throw std::invalid_argument("e8act rotate: use e8act_rotate_activation_launch for BF16");
    }
    e8act_rotate_f32_kernel<<<e8act_grid(k, x.ne[1]), kThreads, 0, stream>>>(
        static_cast<const float*>(x.data), static_cast<float*>(out.data), k,
        e8act_blocks(k, x.ne[1]));
    CUDA_CHECK(cudaGetLastError());
}

}   // namespace ninfer::ops::e8act
