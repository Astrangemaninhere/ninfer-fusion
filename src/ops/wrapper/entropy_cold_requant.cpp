#include "ninfer/ops/entropy_cold_requant.h"

#include "ops/launcher/entropy_cold_requant.h"

#include <cstdint>

namespace ninfer::ops {

void entropy_cold_requant_raw(const std::uint8_t* src_codes, const std::uint8_t* src_scales,
                              EntropyColdRequantMode mode, int kv_heads, int page_count,
                              std::uint8_t* dst_codes, std::uint8_t* dst_scales,
                              cudaStream_t stream) {
    detail::ColdRequantSource source = detail::ColdRequantSource::Nvfp4G16;
    if (mode == EntropyColdRequantMode::Int8G64) {
        source = detail::ColdRequantSource::Int8G64;
    } else if (mode == EntropyColdRequantMode::Iso3VG16) {
        source = detail::ColdRequantSource::Iso3VG16;
    }
    detail::entropy_cold_requant_raw_launch(src_codes, src_scales, source, kv_heads, page_count,
                                            dst_codes, dst_scales, stream);
}

} // namespace ninfer::ops
