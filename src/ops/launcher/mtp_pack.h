#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void mtp_pack_fc_input_launch(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                              cudaStream_t stream);

bool mtp_norm_pack_fc_input_admits(std::int32_t d, const Tensor& embedding,
                                   const Tensor& embedding_weight, const Tensor& hidden,
                                   const Tensor& hidden_weight, const Tensor& out);

void mtp_norm_pack_fc_input_launch(const Tensor& embedding, const Tensor& embedding_weight,
                                   const Tensor& hidden, const Tensor& hidden_weight, Tensor& out,
                                   float eps, cudaStream_t stream);

bool mtp_residual_norm_admits(std::int32_t d, const Tensor& delta, const Tensor& residual,
                              const Tensor& weight, const Tensor& out);

void mtp_residual_norm_launch(const Tensor& delta, Tensor& residual, const Tensor& weight,
                              Tensor& out, float eps, cudaStream_t stream);

// The packed input's row count is the two destination head-blocks doubled, q_rows then kv_rows --
// the split the kernel performs, taken from the destinations rather than named as a constant
// (src/ops/kernel/mtp_pack.cuh:141-149). The wrapper's contract and this launch's guard both read
// it from here, so a stack whose heads differ from the one this Op was first written for moves
// both at once instead of leaving one of them holding a number the other cannot meet.
constexpr std::int32_t mtp_split_attn_rows(std::int32_t q_rows, std::int32_t kv_rows) {
    return 2 * (q_rows + kv_rows);
}

void mtp_split_attn_in_launch(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
