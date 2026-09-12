#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Raw one-page convenience used by runtime owners that hold non-contiguous
// paged-cache slices. Pointers must address one or more (page, kv_head) planes
// with the same PageMajor strides as the Tensor form: page stride =
// kv_heads * 8192 (codes) / kv_heads * 1024 (scales) / slot_bytes (slots) /
// kv_heads (valid). page_count is the grid.y extent.
void entropy_nvfp4_slot_encode_launch(const std::uint8_t* codes, const std::uint8_t* scales,
                                      int kv_heads, int page_count, std::uint8_t* slots,
                                      int slot_bytes, std::int32_t* slot_valid,
                                      const std::int32_t* page_ids, int valid_page_stride,
                                      cudaStream_t stream);

// Decodes all (kv_head, half) streams of one cold-page slot base into a
// contiguous buffer: item (head * 2 + half) holds half_bytes packed bytes.
void entropy_nvfp4_slot_decode_grid_launch(const std::uint8_t* slots, int slot_bytes,
                                           std::int32_t slot_base, int kv_heads,
                                           std::uint8_t* dst, cudaStream_t stream);

// Decodes one slot base and scatters the packed rows into the native
// NVFP4 page-major code plane (row-major [64 x 128 B] per kv_head).
// dec_scratch must hold kv_heads * 8192 bytes.
void entropy_nvfp4_slot_restore_plane_launch(const std::uint8_t* slots, int slot_bytes,
                                             std::int32_t slot_base, int kv_heads,
                                             std::uint8_t* dec_scratch,
                                             std::uint8_t* dst_codes, cudaStream_t stream);

// Scatters the uncompressed 1024-byte scale tail of every (page, kv_head)
// slot into the matching paged scale plane. slots uses the host-cold layout:
// page stride slot_page_stride, head stride slot_bytes; scale page stride is
// scale_page_stride and page_ids supplies physical pages.
void entropy_nvfp4_slot_scales_scatter_launch(const std::uint8_t* slots, int slot_bytes,
                                              int slot_page_stride, int kv_heads,
                                              int page_count, const std::int32_t* page_ids,
                                              int scale_page_stride, std::uint8_t* scales,
                                              cudaStream_t stream);

} // namespace ninfer::ops::detail
