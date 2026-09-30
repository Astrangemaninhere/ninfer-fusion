#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Fixed slot budget covering the rANS MAX -- and "max" is the load-bearing word in
// that sentence, because it is the one this file used to be wrong about.
//
// This is the record the encoder is handed, and the encoder reads its per-stream
// budget out of it at RUNTIME: budget = (slot_bytes - 320 - 1024) / 32, unclamped,
// and a stream that does not fit clears the slot's valid flag so the page stays hot
// (ops/kernel/entropy_nvfp4_slot_kernels.cuh pass A and pass B). So the record's own
// width IS the budget, and the budget must cover the largest stream the codec really
// produces. Measured on the real encoder over the 2048 K streams of 16 engine-dumped
// Qwen3.8 27B pages (dl/ransceil/rans_probe.cu includes THIS header's kernel and runs
// it; see dl/ransceil/REPORT.md): K min 243 / p50 250 / p90 252 / p99 254 / max
// 259 B per 512-symbol stream. 259 B over 512 symbols is 4.0469 bits/symbol, ABOVE
// the 4-bit nibble width, because the requantized K codes are near-uniform.
//
// Hence: 320 B header + 32 streams x 259 B (ceil(512 codes x 4.04 / 8)) + 1024 B
// scale tail = 9632 B. The literal was 6688 (32 x 167 B, the 2.60 bits/code ceiling)
// until 2026-09-18, when the encoder was run for the first time on real planes and
// that record was measured to validate 0 of 64 K slots and 0 of 64 V slots: the codec
// produced nothing at all while the cost model advertised 2528 B/head-page saved.
// The ceiling that justified 2.60 came from a "2.0-2.6 bits/code" band that
// entropy_cold_requant.h now retracts; the same instrument measures 3.31-3.86
// bits/code of marginal entropy, and the per-stream worst case is 4.0469.
//
// THE PRICE, so no reader has to derive it: 9632 B is ABOVE the 9216 B nvfp4 resident
// plane pair this record gives back, so on the 4-bit classes the cold rANS tier is a
// net device-memory COST (+416 B/head-page on nvfp4/iso4e, +928 on rk4v4). It is not a
// saving and must not be reported as one. 9536 B (the 4-bit no-expansion record) is
// narrower but validates only 61 of 64 K slots; 6688 B is narrower still and validates
// none. There is no rANS record that both validates and is narrower than the plane.
//
// This literal is a mirror, not a second decision: decoder_state.cpp derives the
// stride from kColdSlotRansBitsPerCodeX100 (404) and static_asserts that the two agree;
// product/kv_tier_formats.h states the same integer ceil host-side and
// product/kv_bit_budget.h carries the same number as the ladder's cold grid point.
//
// The INT8 tier packs its raw 9232 B layout into the same buffer, so one pool
// size serves both codecs.
inline constexpr std::int32_t kEntropyNvfp4SlotBytes = 9632;

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
// scale_page_stride. `physical_page` is the destination page and arrives BY VALUE -- it used to be
// a device-dereferenced `const std::int32_t* page_ids` that every caller filled from a HOST stack
// array (dl/coldcrash/REPORT.md). page_count is the grid.y extent and must be 1.
void entropy_nvfp4_slot_scales_scatter_raw(const std::uint8_t* slots, int slot_bytes,
                                           int slot_page_stride, int kv_heads,
                                           int page_count, std::int32_t physical_page,
                                           int scale_page_stride, std::uint8_t* scales,
                                           cudaStream_t stream);

} // namespace ninfer::ops
