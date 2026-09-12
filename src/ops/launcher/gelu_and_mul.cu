// ninfer::ops — gelu_mul launcher: grid/block/stream configuration + kernel launch.
// The only translation unit that includes this op's kernel header.
// See docs/maintainer/op-development.md §2.
#include "ops/launcher/gelu_and_mul.h"

#include "ops/common/bf16_vector.cuh"
#include "ops/common/math.h"
#include "ops/kernel/gelu_and_mul.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

void gelu_and_mul_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = out.numel();
    constexpr int kBlock   = 256;
    constexpr int kMaxGrid = 16384;
    const auto gate_addr   = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto up_addr     = reinterpret_cast<std::uintptr_t>(up.data);
    const auto out_addr    = reinterpret_cast<std::uintptr_t>(out.data);
    if (((gate_addr | up_addr | out_addr) & (alignof(Bf16x8Pack) - 1)) == 0 && (n % 8) == 0) {
        const std::int64_t packs = n / 8;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            kMaxGrid, std::max<std::int64_t>(1, div_up(packs, static_cast<std::int64_t>(kBlock)))));
        gelu_and_mul_bf16x8_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const Bf16x8Pack*>(gate.data), static_cast<const Bf16x8Pack*>(up.data),
            static_cast<Bf16x8Pack*>(out.data), packs);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (!gate.is_contiguous() || !up.is_contiguous()) {
        const int grid = static_cast<int>(std::min<std::int64_t>(
            kMaxGrid, std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock)))));
        gelu_and_mul_strided_input_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data),
            static_cast<const __nv_bfloat16*>(up.data), static_cast<__nv_bfloat16*>(out.data), n,
            gate.ne[0], gate.ne[1], gate.ne[2], gate.nb[0], gate.nb[1], gate.nb[2], gate.nb[3],
            up.nb[0], up.nb[1], up.nb[2], up.nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int grid = static_cast<int>(std::min<std::int64_t>(
        kMaxGrid, std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock)))));
    gelu_and_mul_scalar_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data), static_cast<const __nv_bfloat16*>(up.data),
        static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
