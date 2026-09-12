#include "ops/kernel/vocab_topk16.cuh"
#include "ninfer/ops/vocab_topk16.h"

#include <cstdint>

namespace ninfer::ops {

void vocab_topk16(const void* logits_bf16, std::int32_t vocab, std::int32_t tokens,
                  std::int32_t* ids_out, std::uint16_t* vals_bf16_out,
                  cudaStream_t stream) {
    if (vocab <= 0 || tokens <= 0 || logits_bf16 == nullptr || ids_out == nullptr ||
        vals_bf16_out == nullptr) {
        return;
    }
    constexpr int kBlock = 256;
    vocab_topk16_kernel<kBlock><<<tokens, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits_bf16), vocab, tokens, ids_out,
        reinterpret_cast<__nv_bfloat16*>(vals_bf16_out));
}

} // namespace ninfer::ops
