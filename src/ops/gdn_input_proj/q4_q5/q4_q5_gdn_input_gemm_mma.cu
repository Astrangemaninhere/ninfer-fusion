#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/token_slices.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

RowSplitGroupedMmaJob make_job(const Weight& weight, std::int32_t weight_row_offset,
                               std::int32_t rows, Tensor& out, std::int32_t output_row_offset) {
    const std::int64_t groups = weight.padded_shape[1] / weight.group;
    const auto* codes         = static_cast<const std::uint8_t*>(weight.qdata) +
                        static_cast<std::int64_t>(weight_row_offset) * groups * 32;
    const auto* high   = weight.qtype == QType::Q5G64_F16S
                             ? static_cast<const std::uint8_t*>(weight.qhigh) +
                                 static_cast<std::int64_t>(weight_row_offset) * groups * 8
                             : nullptr;
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales) +
                         static_cast<std::int64_t>(weight_row_offset) * groups * 2;
    return RowSplitGroupedMmaJob{
        codes,
        high,
        scales,
        static_cast<__nv_bfloat16*>(out.data),
        rows,
        out.ne[0],
        output_row_offset,
        weight.qtype == QType::Q5G64_F16S,
    };
}

void launch_slice(bool full, const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                  Tensor& qkv, Tensor& z, cudaStream_t stream) {
    using Schedule = GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>;
    // The value/z split is the operands' own geometry: the value_z weight's first half lands in
    // qkv after the qk rows, its second half lands in z. Reading it from the tensors is what lets
    // one kernel body serve a second (value_rows, z_rows) geometry with no new instantiation --
    // and a mismatch is refused here rather than silently mis-tiled.
    const std::int32_t value_rows = z.ne[0];
    if (value_z_weight.n != 2 * value_rows || qkv.ne[0] != qk_weight.n + value_rows) {
        throw std::invalid_argument("GDN Q4/Q5 grouped MMA: value/z row split does not match");
    }
    const RowSplitGroupedMmaJob qk    = make_job(qk_weight, 0, qk_weight.n, qkv, 0);
    const RowSplitGroupedMmaJob value =
        make_job(value_z_weight, 0, value_rows, qkv, qk_weight.n);
    const RowSplitGroupedMmaJob output_gate =
        make_job(value_z_weight, value_rows, value_rows, z, 0);
    RowSplitGroupedMmaJob empty{};
    const int tiles = div_up(qk.n, Schedule::BM) + div_up(value.n, Schedule::BM) +
                      div_up(output_gate.n, Schedule::BM);
    const int cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)));

    if (full) {
        rowsplit_grouped_mma_kernel<Schedule, true, RowSplitGroupedMmaCodec::Mixed, 4>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), qk,
                                                     value, output_gate, empty, x.ne[0], cols,
                                                     x.ne[0]);
    } else {
        rowsplit_grouped_mma_kernel<Schedule, false, RowSplitGroupedMmaCodec::Mixed, 4>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), qk,
                                                     value, output_gate, empty, x.ne[0], cols,
                                                     x.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_q5_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                        cudaStream_t stream) {
    constexpr std::int32_t kTileCols = 128;
    const bool full                  = (x.ne[1] % kTileCols) == 0;
    for_each_token_slice(x.ne[1], kTileCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor qkv_slice     = qkv.slice(1, offset, count);
        Tensor z_slice       = z.slice(1, offset, count);
        launch_slice(full, x_slice, qk_weight, value_z_weight, qkv_slice, z_slice, stream);
    });
}

} // namespace ninfer::ops::detail
