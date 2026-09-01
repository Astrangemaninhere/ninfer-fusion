#include "ninfer/ops/dflash2_grouped_conv.h"

#include "ops/launcher/dflash2_grouped_conv.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

void dflash2_grouped_conv(const Tensor& hidden, const Tensor& delta, const Weight& base,
                          std::int32_t block_size, std::int32_t group_size, std::int32_t taps,
                          std::int32_t side, Tensor& out, cudaStream_t stream) {
    if (hidden.dtype != DType::BF16 || delta.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("dflash2_grouped_conv: hidden/delta/out must be BF16");
    }
    if (!hidden.is_contiguous() || hidden.data == nullptr || !delta.is_contiguous() ||
        delta.data == nullptr || !out.is_contiguous() || out.data == nullptr) {
        throw std::invalid_argument("dflash2_grouped_conv: tensors must be contiguous and non-null");
    }
    const std::int32_t hidden_size = hidden.ne[0];
    const std::int32_t tokens      = hidden.ne[1];
    if (hidden_size != 5120 || tokens < 1 || tokens > 64 || block_size != 8 ||
        group_size != 16 || taps != 2 || side < 0 || side > 1 ||
        (block_size & (block_size - 1)) != 0) {
        throw std::invalid_argument(
            "dflash2_grouped_conv: registered domain is H=5120, T=1..64, taps=2, G=16, "
            "block=8, side in {0,1}");
    }
    if (hidden_size % group_size != 0 || out.ne[0] != hidden_size || out.ne[1] != tokens) {
        throw std::invalid_argument("dflash2_grouped_conv: invalid hidden/out shapes");
    }
    const std::int32_t groups = hidden_size / group_size;
    if (delta.ne[0] != 2 * taps * groups || delta.ne[1] != tokens) {
        throw std::invalid_argument("dflash2_grouped_conv: invalid delta shape");
    }
    if (base.qtype != QType::BF16_CTRL || base.n != taps || base.k != hidden_size ||
        base.qdata == nullptr) {
        throw std::invalid_argument("dflash2_grouped_conv: base must be BF16 [taps,H]");
    }

    detail::dflash2_grouped_conv_launch(hidden, delta, base, block_size, group_size, taps, side,
                                        out, stream);
}

} // namespace ninfer::ops
