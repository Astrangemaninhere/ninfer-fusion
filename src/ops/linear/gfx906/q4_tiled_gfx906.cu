// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : JCraigWasTaken/ninfer-gfx906
// Branch      : gfx906-port
// Commit      : 7a3c18d9fcfc977fc9c35766fd9938c0039e9a78
// Source path : src/ops/linear/gfx906/q4_tiled_gfx906.cu
// sha256(src) : 64a5107922e04d006bbe3207fb98f2cb4e52ac1b9ae6f83351d9e33486c0f8f3
// Landed by   : /home/user/ninfer-fusion gfx906port line, 2026-09-21 (mi50 3-step plan, step 2 of 3: additive port layer; step 3 = CMake wiring, NOT done here).
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : gfx906 (wave64) kernel variant, selected only by the fork's own stage8_route.h / gqa_geometry_dispatch.cuh gates; neither is wired here. Contains no AMD-only builtin, so its source parses as CUDA as well as HIP.
#include "ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh"

#include "core/device.h"
#include "ops/linear/q4/q4_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Cols>
void launch_q4_tiled(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.ne[0] % 4 != 0) {
        throw std::invalid_argument("q4 tiled gfx906: K must be a multiple of 4");
    }
    const std::int32_t rows   = out.ne[0];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    launch_rowsplit_tiled_gemm_gfx906<Q4TileAtomGfx906, Cols>(
        x, w, rows, Gfx906TiledStoreEpilogue{static_cast<__nv_bfloat16*>(out.data), out_ld},
        stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q4_tiled_c16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_tiled<16>(x, w, out, stream);
}

void launch_q4_tiled_c32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_tiled<32>(x, w, out, stream);
}

void launch_q4_tiled_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_tiled<64>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
