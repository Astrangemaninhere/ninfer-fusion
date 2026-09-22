#include "ninfer/ops/mtp_proposal_topk.h"
#include "ops/kernel/mtp_proposal_topk.cuh"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

void mtp_proposal_topk(const void* logits_bf16, std::int32_t rows, std::int32_t tokens,
                       std::int32_t top_l, std::int32_t* ids_out, cudaStream_t stream) {
    if (logits_bf16 == nullptr || ids_out == nullptr || rows <= 0 || tokens <= 0 || top_l <= 0) {
        return;
    }
    constexpr int kBlock = 128;
    const int effective  = top_l < rows ? top_l : rows;
    const std::size_t shared =
        static_cast<std::size_t>(kBlock) * static_cast<std::size_t>(effective) *
        (sizeof(float) + sizeof(int));
    mtp_proposal_topk_kernel<kBlock><<<tokens, kBlock, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(logits_bf16), rows, tokens, effective, ids_out);
}

} // namespace ninfer::ops
