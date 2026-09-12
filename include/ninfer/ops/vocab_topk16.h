#pragma once
// vocab_topk16: per-token top-16 over vocab-major logits [vocab, tokens].
// Raw-pointer API (self-contained; adapter at call sites).
#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops {

void vocab_topk16(const void* logits_bf16, std::int32_t vocab, std::int32_t tokens,
                  std::int32_t* ids_out, std::uint16_t* vals_bf16_out,
                  cudaStream_t stream);

} // namespace ninfer::ops
