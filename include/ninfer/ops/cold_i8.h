#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Fixed raw-slot size: 16 B header + 8192 B E2M1 nibbles + 1024 B E4M3 g16
// scales (see ops/kernel/cold_i8_kernels.cuh for the composition).
inline constexpr std::int32_t kColdI8SlotBytes = 9232;

// Raw cold-slot codec for INT8-tier KV pages (entropy-cold revision 2b).
//
// Slot layout (9232 B, fixed, no overflow): 16 B header | 8192 B packed
// E2M1 nibbles | 1024 B E4M3 group-16 scales, both in the nvfp4 page-major
// geometry produced by entropy_cold_requant_raw(Int8G64). Measured on real
// Qwen3.8 INT8 planes: NMSE 0.012-0.014 (inside the accepted NVFP4-layer
// envelope) at 1.83x per head-page vs the raw int8 plane (16896 B).
void cold_i8_slot_pack_raw(const std::uint8_t* src_codes, const std::uint8_t* src_scales,
                           int kv_heads, int page_count, std::uint8_t* slots,
                           std::int32_t* slot_valid, int slot_bytes, cudaStream_t stream);

// Inverse: unpack slots into the INT8 tier's native planes (int8 codes +
// fp16 group-64 scales). Used by the warm-restore path.
void cold_i8_slot_restore_raw(const std::uint8_t* slots, int kv_heads, int page_count,
                              std::int8_t* dst_codes, void* dst_scales_fp16,
                              int slot_bytes, cudaStream_t stream);

} // namespace ninfer::ops
