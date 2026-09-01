#include "ninfer/ops/cold_i8.h"

#include "ops/launcher/cold_i8.h"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

void cold_i8_slot_pack_raw(const std::uint8_t* src_codes, const std::uint8_t* src_scales,
                           int kv_heads, int page_count, std::uint8_t* slots,
                           std::int32_t* slot_valid, int slot_bytes, cudaStream_t stream) {
    detail::cold_i8_slot_pack_launch(src_codes, src_scales, kv_heads, page_count, slots,
                                     slot_valid, slot_bytes, stream);
}

void cold_i8_slot_restore_raw(const std::uint8_t* slots, int kv_heads, int page_count,
                              std::int8_t* dst_codes, void* dst_scales_fp16,
                              int slot_bytes, cudaStream_t stream) {
    detail::cold_i8_slot_restore_launch(slots, kv_heads, page_count, dst_codes,
                                        static_cast<__half*>(dst_scales_fp16), slot_bytes,
                                        stream);
}

} // namespace ninfer::ops
