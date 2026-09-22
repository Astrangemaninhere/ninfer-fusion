#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void cold_i8_slot_pack_launch(const std::uint8_t* src_codes, const std::uint8_t* src_scales,
                              int kv_heads, int page_count, std::uint8_t* slots,
                              std::int32_t* slot_valid, int slot_bytes, cudaStream_t stream);

void cold_i8_slot_restore_launch(const std::uint8_t* slots, int kv_heads, int page_count,
                                 std::int8_t* dst_codes, void* dst_scales_fp16,
                                 int slot_bytes, cudaStream_t stream);

// BF16-COLD-LAND A5: the bf16-plane inverse (see ninfer/ops/cold_i8.h).
void cold_i8_slot_restore_bf16_launch(const std::uint8_t* slots, int kv_heads, int page_count,
                                      __nv_bfloat16* dst_bf16, int slot_bytes,
                                      cudaStream_t stream);

// BF16-COLD-LAND E5: the rk4v4-plane inverse (see ninfer/ops/cold_i8.h).
void cold_i8_slot_restore_rk4v4_launch(const std::uint8_t* slots, int kv_heads, int page_count,
                                    std::int8_t* dst_codes, void* dst_scales_fp16,
                                    int slot_bytes, cudaStream_t stream);

} // namespace ninfer::ops::detail
