#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Cold-page requantization for the entropy slot codec (revision 2).
// Requantizes one or more stored KV page planes (E2M1 g16 NVFP4 or int8 g64)
// into fresh NVFP4 planes whose codes carry one E4M3FN scale per 64 channels
// (replicated into the four g16 scale slots). The output layout matches the
// page-major nvfp4 planes entropy_nvfp4_slot_encode_raw consumes, so the slot
// codec, decode path, scale scatter, and attention producers stay unchanged;
// the g64 requant is what makes the rANS streams compressible (measured
// 2.0-2.6 bits/code vs ~4.0 for g16 storage codes on Qwen3.8 frames).
enum class EntropyColdRequantMode : int {
    Nvfp4G16 = 0,
    Int8G64  = 1,
    Iso3VG16 = 2,
};

// src planes use page-major strides: codes kv_heads*8192 (nvfp4) or
// kv_heads*16384 (int8) bytes per page, scales kv_heads*1024 (nvfp4) or
// kv_heads*512 (int8). dst planes are nvfp4 page-major: codes
// [128, 64, kv_heads, page_count], scales [16, 64, kv_heads, page_count].
void entropy_cold_requant_raw(const std::uint8_t* src_codes, const std::uint8_t* src_scales,
                              EntropyColdRequantMode mode, int kv_heads, int page_count,
                              std::uint8_t* dst_codes, std::uint8_t* dst_scales,
                              cudaStream_t stream);

} // namespace ninfer::ops
