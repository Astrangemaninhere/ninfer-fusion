// Implements: include/ninfer/ops/sigmoid_mul.h
// Finite dispatch: aligned BF16x8 production route, BF16x2 fallback, then
// scalar fallback for two-byte-aligned sliced storage.
#include "ops/launcher/sigmoid_gate_mul.h"

#include "ops/common/math.h"
#include "ops/kernel/sigmoid_gate_mul.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Private sweep entry (bench/ops/sigmoid_mul_bench.cu calls it directly), so it states its own
// kernel assumptions instead of inheriting the wrapper predicate: the bf16x8 route reads gate and x
// as contiguous 16-byte packs, and a truncated pack count would leave the tail unwritten.
void sigmoid_gate_mul_bf16x8_launch(const Tensor& gate, Tensor& x, int block, cudaStream_t stream) {
    const std::int64_t n = x.numel();
    const auto gate_addr = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto x_addr    = reinterpret_cast<std::uintptr_t>(x.data);
    if (!gate.is_contiguous() || !x.is_contiguous() ||
        ((gate_addr | x_addr) & (alignof(Bf16x8Pack) - 1)) != 0 || (n % 8) != 0) {
        throw std::invalid_argument("sigmoid_mul: bf16x8 route requires contiguous, 16-byte aligned "
                                    "operands and an element count that is a multiple of eight");
    }
    const std::int64_t packs = n / 8;
    constexpr int kMaxGrid   = 4096;
    const int grid           = static_cast<int>(std::min<std::int64_t>(
        kMaxGrid, std::max<std::int64_t>(1, div_up(packs, static_cast<std::int64_t>(block)))));
    sigmoid_gate_mul_bf16x8_kernel<<<grid, block, 0, stream>>>(
        static_cast<const Bf16x8Pack*>(gate.data), static_cast<Bf16x8Pack*>(x.data), packs);
    CUDA_CHECK(cudaGetLastError());
}

// Headwise scalar form: one sigmoid per (head, token) broadcast over the head_dim axis.
// x is [head_dim, H, T] contiguous and gate is [H, T] contiguous.
void headwise_sigmoid_gate_mul_launch(const Tensor& gate, Tensor& x, std::int32_t head_dim,
                                      cudaStream_t stream) {
    const std::int64_t n = x.numel();
    constexpr int kBlock   = 256;
    constexpr int kMaxGrid = 16384;
    const auto gate_addr   = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto x_addr      = reinterpret_cast<std::uintptr_t>(x.data);
    if (head_dim > 0 && (head_dim % 8) == 0 && (n % 8) == 0 &&
        ((gate_addr | x_addr) & (alignof(Bf16x8Pack) - 1)) == 0) {
        const std::int64_t packs  = n / 8;
        const std::int32_t per_head = head_dim / 8;
        const int grid = static_cast<int>(std::min<std::int64_t>(
            kMaxGrid, std::max<std::int64_t>(1, div_up(packs, static_cast<std::int64_t>(kBlock)))));
        headwise_sigmoid_gate_mul_bf16x8_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data), static_cast<Bf16x8Pack*>(x.data),
            packs, per_head);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int grid = static_cast<int>(std::min<std::int64_t>(
        kMaxGrid, std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock)))));
    headwise_sigmoid_gate_mul_scalar_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(x.data), n,
        head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid_gate_mul_launch(const Tensor& gate, Tensor& x, cudaStream_t stream) {
    const std::int64_t n   = x.numel();
    constexpr int kBlock   = 256;
    constexpr int kMaxGrid = 4096;
    const auto gate_addr   = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto x_addr      = reinterpret_cast<std::uintptr_t>(x.data);
    if (((gate_addr | x_addr) & (alignof(Bf16x8Pack) - 1)) == 0 && (n % 8) == 0) {
        sigmoid_gate_mul_bf16x8_launch(gate, x, kBlock, stream);
        return;
    }
    if (((gate_addr | x_addr) & (alignof(__nv_bfloat162) - 1)) != 0) {
        const int scalar_grid = static_cast<int>(
            std::min<std::int64_t>(kMaxGrid, div_up(n, static_cast<std::int64_t>(kBlock))));
        sigmoid_gate_mul_scalar_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(x.data), n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n2                 = n / 2;
    constexpr std::int64_t kPairsPerBlock = kBlock * kSigmoidGateMulPairsPerThread;
    const int grid                        = static_cast<int>(
        std::min<std::int64_t>(kMaxGrid, std::max<std::int64_t>(1, div_up(n2, kPairsPerBlock))));

    sigmoid_gate_mul_bf16x2_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
