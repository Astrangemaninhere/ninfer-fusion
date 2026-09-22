#pragma once

// Cut B: extern declarations of the per-codec decode launchers. Every KV codec's kernel
// instantiation cascade lives in its own translation unit
// (gqa_attention_decode_<i8|nvfp4|fp8|iso3|bf16>.cu), so one nvcc invocation's ptxas stage
// stays small -- the move gqa_attention_decode_e8.cu already makes for the E8 tier. The
// dispatcher (gqa_attention_decode_smallt.cu) calls these instead of expanding
// NINFER_GQA_SMALL_T_DISPATCH (the 6-width X batch/masked X dtype macro the small-T TU used
// to carry) over all five tiers.
//
// The E8 pair is declared here as well: it is a tier like the other five, and the dispatcher
// is its only caller. gqa_attention_decode_partial.cuh repeats the same declarations verbatim
// in its tier comment; identical redeclarations in two headers are legal, and that file is
// included by the per-tier TUs (which never call E8) rather than by the dispatcher.

#include "ops/launcher/gqa_attention.h"

#include "ops/kernel/gqa_attention_decode.cuh"  // GqaAppendInput / GqaCachedInput

#include <cstdint>

namespace ninfer::ops::detail {

// One pair of overloads per tier. The argument list is the launcher's own minus the per-tier
// kernel schedule: `implementation_window` is the launch's split reference (the value
// gqa_small_t_split_reference returns, which gqa_attention_decode_smallt.cu already computes),
// and each tier derives its own tile schedule and its own split_units from it -- exactly what
// the pre-split dispatch macro did inside its own expansion.

void gqa_attention_decode_i8_launch(const Tensor& q, const GqaAppendInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t implementation_window, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream);
void gqa_attention_decode_i8_launch(const Tensor& q, const GqaCachedInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t implementation_window, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream);

void gqa_attention_decode_nvfp4_launch(const Tensor& q, const GqaAppendInput& input,
                                       const Tensor& pos, float scale,
                                       PagedKVBatchLayerView cache,
                                       const GqaSmallTInvocation& invocation,
                                       std::int32_t logical_capacity,
                                       std::int32_t implementation_window, std::int32_t splits,
                                       Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                       cudaStream_t stream);
void gqa_attention_decode_nvfp4_launch(const Tensor& q, const GqaCachedInput& input,
                                       const Tensor& pos, float scale,
                                       PagedKVBatchLayerView cache,
                                       const GqaSmallTInvocation& invocation,
                                       std::int32_t logical_capacity,
                                       std::int32_t implementation_window, std::int32_t splits,
                                       Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                       cudaStream_t stream);

void gqa_attention_decode_fp8_launch(const Tensor& q, const GqaAppendInput& input,
                                     const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                     const GqaSmallTInvocation& invocation,
                                     std::int32_t logical_capacity,
                                     std::int32_t implementation_window, std::int32_t splits,
                                     Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                     cudaStream_t stream);
void gqa_attention_decode_fp8_launch(const Tensor& q, const GqaCachedInput& input,
                                     const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                     const GqaSmallTInvocation& invocation,
                                     std::int32_t logical_capacity,
                                     std::int32_t implementation_window, std::int32_t splits,
                                     Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                     cudaStream_t stream);

void gqa_attention_decode_iso3_launch(const Tensor& q, const GqaAppendInput& input,
                                      const Tensor& pos, float scale,
                                      PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      std::int32_t logical_capacity,
                                      std::int32_t implementation_window, std::int32_t splits,
                                      Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                      cudaStream_t stream);
void gqa_attention_decode_iso3_launch(const Tensor& q, const GqaCachedInput& input,
                                      const Tensor& pos, float scale,
                                      PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      std::int32_t logical_capacity,
                                      std::int32_t implementation_window, std::int32_t splits,
                                      Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                      cudaStream_t stream);

void gqa_attention_decode_bf16_launch(const Tensor& q, const GqaAppendInput& input,
                                      const Tensor& pos, float scale,
                                      PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      std::int32_t logical_capacity,
                                      std::int32_t implementation_window, std::int32_t splits,
                                      Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                      cudaStream_t stream);
void gqa_attention_decode_bf16_launch(const Tensor& q, const GqaCachedInput& input,
                                      const Tensor& pos, float scale,
                                      PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      std::int32_t logical_capacity,
                                      std::int32_t implementation_window, std::int32_t splits,
                                      Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                      cudaStream_t stream);

// Defined in gqa_attention_decode_e8.cu (UNCHANGED by this cut; same text as the declarations
// next to that TU's comment inside gqa_attention_decode_partial.cuh).
void gqa_attention_decode_e8_launch(const Tensor& q, const GqaAppendInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t implementation_window, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream);
void gqa_attention_decode_e8_launch(const Tensor& q, const GqaCachedInput& input,
                                    const Tensor& pos, float scale, PagedKVBatchLayerView cache,
                                    const GqaSmallTInvocation& invocation,
                                    std::int32_t logical_capacity,
                                    std::int32_t implementation_window, std::int32_t splits,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
