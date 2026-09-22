#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer {

enum class DType : std::uint8_t {
    BF16       = 0,
    FP32       = 1,
    I32        = 2,
    U8         = 3,
    I64        = 4,
    I8         = 5,
    FP16       = 6,
    FP8_E4M3FN = 7,
    // Packed E2M1 nibble plane (two codes per byte) with per-16-channel
    // E4M3FN scales; see the per-layer KV storage table.
    NVFP4      = 8,
    ISO3       = 9,
    // Packed 4-bit E8-lattice K codes + i4 V codes (two codes per byte) with
    // per-64-channel FP16 scales; consumed by the int8 attention kernels
    // (stage unpacks nibbles to i8). See the per-layer KV storage table.
    E8Kv       = 10,
    // e8 family, narrower K-plane code widths (e8k3 / e8k2): the SAME K+V plane pair
    // and the same per-64 FP16 scale plane as E8Kv, with the K code plane packed at 3
    // or 2 bits per element. Geometry and cost are derived in product/kv_e8_width.h
    // (K plane 8704 -> 6656 -> 4608 B/head-page; layer 4.25 -> 3.75 -> 3.25 b/element,
    // K+V averaged, which is what the bit-budget ladder prices).
    // NOT a different codebook: product/kv_e8_width.h records that the ecosystem's
    // rk2v4-e8 reproduces this exact GEOMETRY while its 240-root cylinder CODEC does
    // not, and that the measured lattice floor puts both widths BELOW the projection
    // they are named for.
    E8K3Kv     = 11,
    E8K2Kv     = 12,
};


std::size_t dtype_size(DType dtype);

} // namespace ninfer
