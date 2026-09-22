#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class ColdRequantSource : int {
    Nvfp4G16   = 0, // E2M1 nibbles + E4M3 g16 scales (K planes)
    Int8G64    = 1, // int8 codes + fp16 g64 scales
    Iso4eVG16  = 2, // ISO4E sign-magnitude nibble code + E4M3 g16 scales (V planes)
    Bf16G64    = 3, // raw bf16 plane, 256 bf16 per key row, no scale plane (A3)
    Rk4v4KvG64 = 4, // packed 4-bit rk4v4 codes copied verbatim + fp16 g64 scales (E3)
};

void entropy_cold_requant_raw_launch(const std::uint8_t* src_codes,
                                     const std::uint8_t* src_scales, ColdRequantSource mode,
                                     int kv_heads, int page_count, std::uint8_t* dst_codes,
                                     std::uint8_t* dst_scales, cudaStream_t stream);

} // namespace ninfer::ops::detail
