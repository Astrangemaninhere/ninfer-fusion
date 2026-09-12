#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Fixed slot budget covering the rANS max: 320 B header + 32 streams x 256 B
// + 1024 B scale tail. The INT8 tier packs its raw 9232 B layout into the
// same buffer, so one pool size serves both codecs.
inline constexpr std::int32_t kEntropyNvfp4SlotBytes = 9536;

/**
 * Page-slot NVFP4 E2M1 rANS codec. One slot stores the 8192 packed code bytes
 * of one (physical page, kv_head, K|V plane). Each slot contains two 32-token
 * half-pages; each half-page is 16 independent rANS streams of 512 code
 * nibbles. Frequencies are shared by the 16 streams of one half and stored in
 * the 320-byte slot header. Stream offsets in the header are relative to the
 * slot start and include the 4-byte little-endian final state at each stream
 * end. See ops/kernel/entropy_nvfp4_slot.cuh for the exact layout.
 *
 * encode: codes is the packed page plane shaped [128, 64, kv_heads, pages] and
 * scales is the matching E4M3FN page plane [16, 64, kv_heads, pages] in
 * paged-cache PageMajor layout. slots is U8 [slot_bytes, kv_heads, pages];
 * slot_valid is I32 [kv_heads, pages] (1 = valid compressed slot, 0 = keep
 * using the uncompressed code plane). The 1024 scale bytes of a head-page are
 * stored uncompressed at the end of the slot, so the whole physical page group
 * can be returned to the pool while the page is cold.
 *
 * decode: decodes selected half-pages into packed codes. slot_ids is I32
 * [items] with flattened slot indices page * kv_heads + head; halves is I32
 * [items] (0 or 1); codes_out is U8 [4096, items] (16 contiguous 256-byte
 * streams per item).
 */
void entropy_nvfp4_slot_encode(const Tensor& codes, const Tensor& scales, Tensor& slots,
                               Tensor& slot_valid, cudaStream_t stream);

// Raw one-page convenience used by runtime owners that hold non-contiguous
// paged-cache slices. Pointers must address one or more (page, kv_head) planes
// with the same PageMajor strides as the Tensor form: page stride =
// kv_heads * 8192 (codes) / kv_heads * 1024 (scales) / slot_bytes (slots) /
// kv_heads (valid). page_count is the grid.y extent.
void entropy_nvfp4_slot_encode_raw(const std::uint8_t* codes, const std::uint8_t* scales,
                                   int kv_heads, int page_count, std::uint8_t* slots,
                                   int slot_bytes, std::int32_t* slot_valid,
                                   cudaStream_t stream);

// Batched raw encode for paged owners with a logical-to-physical page map.
// codes/scales address the page-0 plane base; blockIdx.y logical page p reads
// physical page page_ids[p]. slots strides logically (page * kv_heads *
// slot_bytes) and slot_valid strides valid_page_stride elements per page.
// Pass page_ids == nullptr with valid_page_stride == kv_heads for the
// contiguous Tensor-form layout.
void entropy_nvfp4_slot_encode_raw(const std::uint8_t* codes, const std::uint8_t* scales,
                                   int kv_heads, int page_count, std::uint8_t* slots,
                                   int slot_bytes, std::int32_t* slot_valid,
                                   const std::int32_t* page_ids, int valid_page_stride,
                                   cudaStream_t stream);

void entropy_nvfp4_slot_decode_half(const Tensor& slots, const Tensor& slot_ids,
                                    const Tensor& halves, Tensor& codes_out,
                                    cudaStream_t stream);

// Raw batched half-page decode for runtime owners: slot_ids/halves are host
// pointers valid for the launch, dst receives half_bytes contiguous packed
// bytes per item.
void entropy_nvfp4_slot_decode_half_raw(const std::uint8_t* slots, int slot_bytes,
                                        const std::int32_t* slot_ids,
                                        const std::int32_t* halves, std::uint8_t* dst,
                                        int half_bytes, int items, cudaStream_t stream);

// Decodes all (kv_head, half) streams of one cold-page slot base into a
// contiguous buffer: item (head * 2 + half) holds half_bytes packed bytes.
void entropy_nvfp4_slot_decode_grid_raw(const std::uint8_t* slots, int slot_bytes,
                                        std::int32_t slot_base, int kv_heads,
                                        std::uint8_t* dst, cudaStream_t stream);

// Decodes one slot base and scatters the packed rows into the native
// NVFP4 page-major code plane (row-major [64 x 128 B] per kv_head).
// dec_scratch must hold kv_heads * 8192 bytes.
void entropy_nvfp4_slot_restore_plane_raw(const std::uint8_t* slots, int slot_bytes,
                                          std::int32_t slot_base, int kv_heads,
                                          std::uint8_t* dec_scratch,
                                          std::uint8_t* dst_codes, cudaStream_t stream);

// Scatters the uncompressed 1024-byte scale tail of every (page, kv_head)
// slot into the matching paged scale plane. slots uses the host-cold layout:
// page stride slot_page_stride, head stride slot_bytes; scale page stride is
// scale_page_stride and page_ids supplies physical pages.
void entropy_nvfp4_slot_scales_scatter_raw(const std::uint8_t* slots, int slot_bytes,
                                           int slot_page_stride, int kv_heads,
                                           int page_count, const std::int32_t* page_ids,
                                           int scale_page_stride, std::uint8_t* scales,
                                           cudaStream_t stream);

} // namespace ninfer::ops
