#pragma once

// ninfer::ops - page-slot NVFP4 E2M1 rANS codec.
//
// One slot stores one (physical page, kv_head, K|V plane): the 64-token page
// is split into two 32-token half-pages, and each half-page is split into 16
// independent rANS streams of 512 code nibbles (two contiguous 128-byte rows
// per stream). Streams are independent so a 16-thread group can decode a
// half-page in parallel; this is the granularity the attention producers use.
//
// Slot layout (all offsets relative to the slot start):
//   header 320 bytes:
//     magic/version/flags (8) + 2 halves x 128 bytes
//   each half header:
//     uint16 freqs[16] (shared by all 16 streams, normalized from their
//     combined histogram), uint32 offsets[17] (start of each stream and the
//     end sentinel), 28 bytes reserved.
//   stream bytes: canonical rANS streams (renorm bytes in emission order,
//     4-byte final state last) packed back-to-back in stream order.
//
// A page whose codes are incompressible at the fixed stream budget keeps the
// normal code plane and clears the slot's valid flag; callers fall back to the
// uncompressed plane.

#include "ops/kernel/entropy_nvfp4_common.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::uint32_t kEntropyNvfp4SlotMagic    = 0x3156524Eu; // "NRV1"
inline constexpr std::uint16_t kEntropyNvfp4SlotVersion  = 1;
inline constexpr std::uint16_t kEntropyNvfp4SlotFlagValid = 0x0001u;
inline constexpr int kEntropyNvfp4SlotHalfBytes          = 4096; // 32 x 128
inline constexpr int kEntropyNvfp4SlotStreamsPerHalf     = 16;
inline constexpr int kEntropyNvfp4SlotStreamBytes        = 256; // 512 nibbles
inline constexpr int kEntropyNvfp4SlotStreamSymbols      = 512;
inline constexpr int kEntropyNvfp4SlotStreams            = 32;
inline constexpr int kEntropyNvfp4SlotHeaderBytes        = 320;
// One uncompressed E4M3FN scale byte per 16-channel group and token row:
// 16 x 64 = 1024 bytes. Scales live at the end of the slot so the whole
// physical page group (code planes and scale planes) can be freed while cold.
inline constexpr int kEntropyNvfp4SlotScaleBytes = 1024;

struct alignas(8) EntropyNvfp4SlotHalf {
    std::uint16_t freqs[16];
    std::uint32_t offsets[17];
    std::uint8_t reserved[28];
};
static_assert(sizeof(EntropyNvfp4SlotHalf) == 128);

struct alignas(8) EntropyNvfp4SlotHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t flags;
    EntropyNvfp4SlotHalf halves[2];
    std::uint8_t reserved[56];
};
static_assert(sizeof(EntropyNvfp4SlotHeader) == 320);

// Read half->stream 0..15: rows half*32 + [2*stream, 2*stream+2), each row
// 128 packed code bytes. dst must hold 256 bytes.
__device__ __forceinline__ void entropy_nvfp4_slot_stream_source(
    const std::uint8_t* page_codes, int half, int stream, std::uint8_t (&dst)[256]) {
#pragma unroll
    for (int row = 0; row < 2; ++row) {
        const int src_row = half * 32 + 2 * stream + row;
        const std::uint8_t* src = page_codes + src_row * 128;
#pragma unroll
        for (int i = 0; i < 128; ++i) { dst[row * 128 + i] = src[i]; }
    }
}

__device__ __forceinline__ std::uint32_t
entropy_nvfp4_slot_rans_encode_count(const std::uint8_t (&codes)[256], int count,
                                     const std::uint16_t (&fs)[16]) {
    std::uint32_t start[16];
    entropy_nvfp4_make_start(fs, start);

    std::uint32_t x       = kEntropyNvfp4RansByteL;
    std::uint32_t emitted = 0;
    for (int i = count - 1; i >= 0; --i) {
        const std::uint8_t byte = codes[i >> 1];
        const int symbol        = (i & 1) ? (byte >> 4) : (byte & 0x0f);
        const std::uint32_t freq = fs[symbol];
        const std::uint32_t x_max =
            ((kEntropyNvfp4RansByteL >> kEntropyNvfp4ScaleBits) << 8) * freq;
        while (x >= x_max) {
            ++emitted;
            x >>= 8;
        }
        x = ((x / freq) << kEntropyNvfp4ScaleBits) + (x % freq) + start[symbol];
    }
    return emitted + 4; // renorm bytes + little-endian final state
}

// Returns the stream size including the final state, or -1 on capacity
// overflow. The stream uses the canonical layout: renorm bytes in emission
// order followed by the 4-byte final state.
__device__ __forceinline__ int
entropy_nvfp4_slot_rans_encode_to(std::uint8_t* dst, int capacity,
                                  const std::uint8_t (&codes)[256], int count,
                                  const std::uint16_t (&fs)[16]) {
    std::uint32_t start[16];
    entropy_nvfp4_make_start(fs, start);

    std::uint32_t x     = kEntropyNvfp4RansByteL;
    std::uint8_t* ptr   = dst;
    const int state_room = capacity - 4;
    if (state_room < 0) { return -1; }
    for (int i = count - 1; i >= 0; --i) {
        const std::uint8_t byte = codes[i >> 1];
        const int symbol        = (i & 1) ? (byte >> 4) : (byte & 0x0f);
        const std::uint32_t freq = fs[symbol];
        const std::uint32_t x_max =
            ((kEntropyNvfp4RansByteL >> kEntropyNvfp4ScaleBits) << 8) * freq;
        while (x >= x_max) {
            if (ptr - dst >= state_room) { return -1; }
            *ptr++ = static_cast<std::uint8_t>(x & 0xffu);
            x >>= 8;
        }
        x = ((x / freq) << kEntropyNvfp4ScaleBits) + (x % freq) + start[symbol];
    }
    if (ptr - dst + 4 > capacity) { return -1; }
    ptr[0] = static_cast<std::uint8_t>(x & 0xffu);
    ptr[1] = static_cast<std::uint8_t>((x >> 8) & 0xffu);
    ptr[2] = static_cast<std::uint8_t>((x >> 16) & 0xffu);
    ptr[3] = static_cast<std::uint8_t>((x >> 24) & 0xffu);
    return static_cast<int>(ptr - dst) + 4;
}

// Decode stream `stream` of `half` from a valid slot, invoking fn(i, symbol)
// for every one of the 512 code nibbles in symbol order (i 0..511). Returns
// false when the header/stream is malformed. This is the low-level hook used
// by attention producers that want to dequantize on the fly into BF16 smem.
template <typename Fn>
__device__ __forceinline__ bool entropy_nvfp4_slot_decode_stream_apply(const std::uint8_t* slot,
                                                                       int half, int stream,
                                                                       Fn&& fn) {
    const EntropyNvfp4SlotHeader* header =
        reinterpret_cast<const EntropyNvfp4SlotHeader*>(slot);
    if (header->magic != kEntropyNvfp4SlotMagic ||
        header->version != kEntropyNvfp4SlotVersion ||
        (header->flags & kEntropyNvfp4SlotFlagValid) == 0) {
        return false;
    }
    const EntropyNvfp4SlotHalf& half_header = header->halves[half];
    const std::uint32_t begin               = half_header.offsets[stream];
    const std::uint32_t end                 = half_header.offsets[stream + 1];
    if (end < begin + 4) { return false; }
    const int size = static_cast<int>(end - begin);

    std::uint16_t fs[16];
    std::uint32_t start[16];
    for (int s = 0; s < 16; ++s) { fs[s] = half_header.freqs[s]; }
    entropy_nvfp4_make_start(fs, start);

    const std::uint8_t* stream_bytes = slot + begin;
    std::uint32_t x                  = static_cast<std::uint32_t>(stream_bytes[size - 4]) |
                      (static_cast<std::uint32_t>(stream_bytes[size - 3]) << 8) |
                      (static_cast<std::uint32_t>(stream_bytes[size - 2]) << 16) |
                      (static_cast<std::uint32_t>(stream_bytes[size - 1]) << 24);
    int pos = size - 4;

    for (int i = 0; i < kEntropyNvfp4SlotStreamSymbols; ++i) {
        const std::uint32_t xmask = x & kEntropyNvfp4Mask;
        int symbol                = 0;
        while (symbol < 15 &&
               !(start[symbol] <= xmask && xmask < start[symbol] + fs[symbol])) {
            ++symbol;
        }
        fn(i, symbol);
        x = fs[symbol] * (x >> kEntropyNvfp4ScaleBits) + xmask - start[symbol];
        while (x < kEntropyNvfp4RansByteL) {
            if (pos == 0) { return false; }
            --pos;
            x = (x << 8) | stream_bytes[pos];
        }
    }
    return true;
}

// Decode stream `stream` of `half` from a valid slot into 256 contiguous
// packed code bytes. Returns false when the header/stream is malformed.
__device__ __forceinline__ bool entropy_nvfp4_slot_decode_stream(
    const std::uint8_t* slot, int half, int stream, std::uint8_t* dst) {
    std::uint8_t packed = 0;
    return entropy_nvfp4_slot_decode_stream_apply(
        slot, half, stream, [&](int i, int symbol) {
            const int byte_index = i >> 1;
            if ((i & 1) == 0) {
                packed         = static_cast<std::uint8_t>(symbol);
                dst[byte_index] = packed;
            } else {
                dst[byte_index] = static_cast<std::uint8_t>((symbol << 4) | packed);
            }
        });
}

// Fixed tail region holding the uncompressed E4M3FN scale page.
__device__ __forceinline__ const std::uint8_t* entropy_nvfp4_slot_scales(const std::uint8_t* slot,
                                                                         int slot_bytes) {
    return slot + slot_bytes - kEntropyNvfp4SlotScaleBytes;
}

// Cooperative half-page decode: threads 0..15 each decode one 256-byte stream
// into the contiguous 4096-byte half buffer. Other threads do nothing.
__device__ __forceinline__ void entropy_nvfp4_slot_decode_half_parallel(const std::uint8_t* slot,
                                                                        int half, int lane,
                                                                        std::uint8_t* half_dst) {
    if (lane >= kEntropyNvfp4SlotStreamsPerHalf) { return; }
    entropy_nvfp4_slot_decode_stream(slot, half, lane,
                                     half_dst + lane * kEntropyNvfp4SlotStreamBytes);
}

} // namespace ninfer::ops::detail
