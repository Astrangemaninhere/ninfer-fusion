#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void cold_i8_slot_pack_launch(const std::uint8_t* src_codes, const std::uint8_t* src_scales,
                              int kv_heads, int page_count, std::uint8_t* slots,
                              std::int32_t* slot_valid, int slot_bytes, cudaStream_t stream);

void cold_i8_slot_restore_launch(const std::uint8_t* slots, int kv_heads, int page_count,
                                 std::int8_t* dst_codes, void* dst_scales_fp16,
                                 int slot_bytes, cudaStream_t stream);

} // namespace ninfer::ops::detail
