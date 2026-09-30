// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : JCraigWasTaken/ninfer-gfx906
// Branch      : gfx906-port
// Commit      : 7a3c18d9fcfc977fc9c35766fd9938c0039e9a78
// Source path : src/ops/linear_add/q5/q5_linear_add_tiled_gfx906.cu
// sha256(src) : 94f247cc7808d7cfe74a39add5a5cea39e279b5a5224b6ccbab66d384def3510
// Landed by   : /home/user/ninfer-fusion gfx906port line, 2026-09-21 (mi50 3-step plan, step 2 of 3: additive port layer; step 3 = CMake wiring, NOT done here).
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : gfx906 (wave64) kernel variant, selected only by the fork's own stage8_route.h / gqa_geometry_dispatch.cuh gates; neither is wired here. Contains no AMD-only builtin, so its source parses as CUDA as well as HIP.
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/gfx906/rowsplit_tiled_gemm_gfx906.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void q5_linear_add_tiled_gfx906_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    if (x.ne[0] % 4 != 0) {
        throw std::invalid_argument("q5 linear_add tiled gfx906: K must be a multiple of 4");
    }
    const std::int32_t rows   = residual_out.ne[0];
    const std::int32_t cols   = x.ne[1];
    const std::int32_t out_ld =
        static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16));
    const Gfx906TiledResidualAddEpilogue epilogue{
        static_cast<__nv_bfloat16*>(residual_out.data), out_ld};
    if (cols <= 16) {
        launch_rowsplit_tiled_gemm_gfx906<Q5TileAtomGfx906, 16>(x, w, rows, epilogue, stream);
    } else if (cols <= 32) {
        launch_rowsplit_tiled_gemm_gfx906<Q5TileAtomGfx906, 32>(x, w, rows, epilogue, stream);
    } else {
        launch_rowsplit_tiled_gemm_gfx906<Q5TileAtomGfx906, 64>(x, w, rows, epilogue, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
