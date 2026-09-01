#include "ninfer/ops/entropy_nvfp4_slot.h"

#include "ops/launcher/entropy_nvfp4_slot.h"

#include <cstdint>

namespace ninfer::ops {

void entropy_nvfp4_slot_encode_raw(const std::uint8_t* codes, const std::uint8_t* scales,
                                   int kv_heads, int page_count, std::uint8_t* slots,
                                   int slot_bytes, std::int32_t* slot_valid,
                                   const std::int32_t* page_ids, int valid_page_stride,
                                   cudaStream_t stream) {
    detail::entropy_nvfp4_slot_encode_launch(codes, scales, kv_heads, page_count, slots,
                                             slot_bytes, slot_valid, page_ids,
                                             valid_page_stride, stream);
}

void entropy_nvfp4_slot_decode_grid_raw(const std::uint8_t* slots, int slot_bytes,
                                        std::int32_t slot_base, int kv_heads,
                                        std::uint8_t* dst, cudaStream_t stream) {
    detail::entropy_nvfp4_slot_decode_grid_launch(slots, slot_bytes, slot_base, kv_heads, dst,
                                                  stream);
}

void entropy_nvfp4_slot_scales_scatter_raw(const std::uint8_t* slots, int slot_bytes,
                                           int slot_page_stride, int kv_heads,
                                           int page_count, const std::int32_t* page_ids,
                                           int scale_page_stride, std::uint8_t* scales,
                                           cudaStream_t stream) {
    detail::entropy_nvfp4_slot_scales_scatter_launch(slots, slot_bytes, slot_page_stride,
                                                     kv_heads, page_count, page_ids,
                                                     scale_page_stride, scales, stream);
}

void entropy_nvfp4_slot_restore_plane_raw(const std::uint8_t* slots, int slot_bytes,
                                          std::int32_t slot_base, int kv_heads,
                                          std::uint8_t* dec_scratch,
                                          std::uint8_t* dst_codes, cudaStream_t stream) {
    detail::entropy_nvfp4_slot_restore_plane_launch(slots, slot_bytes, slot_base, kv_heads,
                                                    dec_scratch, dst_codes, stream);
}

} // namespace ninfer::ops
