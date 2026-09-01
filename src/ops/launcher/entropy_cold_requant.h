#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class ColdRequantSource : int {
    Nvfp4G16 = 0, // E2M1 nibbles + E4M3 g16 scales (K planes)
    Int8G64  = 1, // int8 codes + fp16 g64 scales
    Iso3VG16 = 2, // ISO3 sign-magnitude INT3 nibbles + E4M3 g16 scales (V planes)
};

void entropy_cold_requant_raw_launch(const std::uint8_t* src_codes,
                                     const std::uint8_t* src_scales, ColdRequantSource mode,
                                     int kv_heads, int page_count, std::uint8_t* dst_codes,
                                     std::uint8_t* dst_scales, cudaStream_t stream);

} // namespace ninfer::ops::detail
