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
};


std::size_t dtype_size(DType dtype);

} // namespace ninfer
