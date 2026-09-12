#pragma once

// Launcher-only __global__ kernels for the page-slot NVFP4 rANS codec. Kept
// separate from entropy_nvfp4_slot.cuh so device helper headers can be included
// by attention kernels without duplicate device-link definitions.

#include "ops/kernel/entropy_nvfp4_slot.cuh"
#include "ops/common/memory.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One block per (page, kv_head) slot. codes addresses a full page plane
// [128 bytes, 64 rows, kv_heads, pages] and scales addresses the matching
// [16 bytes, 64 rows, kv_heads, pages] plane in paged-cache PageMajor layout:
// page stride = kv_heads * 8192 (codes) / kv_heads * 1024 (scales), head stride
// = 8192 / 1024 bytes. Thread t owns stream (t & 15) of half (t >> 4).
__global__ void entropy_nvfp4_slot_encode_kernel(const std::uint8_t* __restrict__ codes,
                                                 const std::uint8_t* __restrict__ scales,
                                                 int kv_heads,
                                                 std::uint8_t* __restrict__ slots,
                                                 int slot_bytes,
                                                 std::int32_t* __restrict__ slot_valid,
                                                 const std::int32_t* __restrict__ page_ids,
                                                 int valid_page_stride) {
    constexpr int StreamsPerHalf = kEntropyNvfp4SlotStreamsPerHalf;
    constexpr int Streams        = kEntropyNvfp4SlotStreams;
    constexpr int HeaderBytes    = kEntropyNvfp4SlotHeaderBytes;

    __shared__ std::uint32_t s_hist[2][16];
    __shared__ std::uint16_t s_freqs[2][16];
    __shared__ std::uint32_t s_sizes[2][16];
    __shared__ std::uint32_t s_prefix[2][16];
    __shared__ std::uint32_t s_data_base[2];
    __shared__ std::uint32_t s_half_total[2];
    __shared__ bool s_overflow;

    const int head  = static_cast<int>(blockIdx.x);
    const int page  = static_cast<int>(blockIdx.y);
    const int physical = page_ids == nullptr ? page : static_cast<int>(page_ids[page]);
    const int tid   = static_cast<int>(threadIdx.x);
    const int half  = tid >> 4;
    const int lane  = tid & 15;

    const std::int64_t head_stride = static_cast<std::int64_t>(kEntropyNvfp4SlotHalfBytes) * 2;
    const std::uint8_t* page_codes = codes + (static_cast<std::int64_t>(physical) * kv_heads + head) *
                                                head_stride;
    const std::uint8_t* page_scales =
        scales + (static_cast<std::int64_t>(physical) * kv_heads + head) *
                     kEntropyNvfp4SlotScaleBytes;
    std::uint8_t* slot = slots + (static_cast<std::int64_t>(page) * kv_heads + head) * slot_bytes;

    for (int i = tid; i < slot_bytes; i += Streams) { slot[i] = 0; }
    __syncthreads();

    if (tid == 0) {
        for (int h = 0; h < 2; ++h) {
            for (int s = 0; s < 16; ++s) {
                s_hist[h][s]   = 0;
                s_freqs[h][s]  = 0;
                s_sizes[h][s]  = 0;
                s_prefix[h][s] = 0;
            }
        }
        s_overflow = false;
    }
    __syncthreads();

    // --- pass A: combined histogram -> shared frequencies -> stream sizes ----
    std::uint8_t codes_stream[256];
    entropy_nvfp4_slot_stream_source(page_codes, half, lane, codes_stream);
    std::uint32_t local_hist[16] = {0};
    for (int i = 0; i < kEntropyNvfp4SlotStreamSymbols; ++i) {
        const std::uint8_t byte = codes_stream[i >> 1];
        ++local_hist[(i & 1) ? (byte >> 4) : (byte & 0x0f)];
    }
    for (int s = 0; s < 16; ++s) {
        atomicAdd(&s_hist[half][s], local_hist[s]);
    }
    __syncthreads();

    if (tid == 0) { s_overflow = false; }
    __syncthreads();
    if (lane == 0) {
        int combined[16];
        for (int s = 0; s < 16; ++s) {
            combined[s] = static_cast<int>(s_hist[half][s]);
        }
        entropy_nvfp4_normalize_freqs(combined, kEntropyNvfp4SlotStreamSymbols * StreamsPerHalf,
                                      s_freqs[half]);
    }
    __syncthreads();

    {
        std::uint16_t fs[16];
        for (int s = 0; s < 16; ++s) { fs[s] = s_freqs[half][s]; }
        const std::uint32_t size =
            entropy_nvfp4_slot_rans_encode_count(codes_stream, kEntropyNvfp4SlotStreamSymbols, fs);
        s_sizes[half][lane] = size;
    }
    __syncthreads();

    const int data_bytes = slot_bytes - HeaderBytes - kEntropyNvfp4SlotScaleBytes;
    if (lane == 0) {
        const int budget   = data_bytes / Streams;
        std::uint32_t prefix = 0;
        bool overflow        = false;
        for (int s = 0; s < StreamsPerHalf; ++s) {
            const std::uint32_t size = s_sizes[half][s];
            if (size > static_cast<std::uint32_t>(budget)) { overflow = true; }
            s_prefix[half][s] = prefix;
            prefix += size;
        }
        s_half_total[half] = prefix;
        if (prefix > static_cast<std::uint32_t>(data_bytes / 2)) { overflow = true; }
        if (overflow) { s_overflow = true; }
    }
    __syncthreads();
    if (s_overflow) {
        slot_valid[static_cast<std::int64_t>(page) * valid_page_stride + head] = 0;
        if (tid == 0) {
            EntropyNvfp4SlotHeader* header = reinterpret_cast<EntropyNvfp4SlotHeader*>(slot);
            header->magic                  = kEntropyNvfp4SlotMagic;
            header->version                = kEntropyNvfp4SlotVersion;
            header->flags                  = 0;
        }
        return;
    }

    if (tid == 0) {
        const std::uint32_t half0_total = s_half_total[0];
        s_data_base[0]                  = HeaderBytes;
        s_data_base[1]                  = HeaderBytes + half0_total;
    }
    __syncthreads();

    const std::uint32_t base = s_data_base[half];
    if (lane == 0) {
        EntropyNvfp4SlotHeader* header = reinterpret_cast<EntropyNvfp4SlotHeader*>(slot);
        header->magic                  = kEntropyNvfp4SlotMagic;
        header->version                = kEntropyNvfp4SlotVersion;
        header->flags                  = kEntropyNvfp4SlotFlagValid;
        EntropyNvfp4SlotHalf& half_header = header->halves[half];
        for (int s = 0; s < 16; ++s) { half_header.freqs[s] = s_freqs[half][s]; }
        for (int s = 0; s < 16; ++s) {
            half_header.offsets[s] = base + s_prefix[half][s];
        }
        half_header.offsets[16] = base + s_half_total[half];
    }
    __syncthreads();

    // --- pass B: encode to the final compacted positions ----
    {
        std::uint16_t fs[16];
        for (int s = 0; s < 16; ++s) { fs[s] = s_freqs[half][s]; }
        std::uint8_t* stream_dst = slot + base + s_prefix[half][lane];
        const int budget = data_bytes / Streams;
        const int size =
            entropy_nvfp4_slot_rans_encode_to(stream_dst, budget, codes_stream,
                                              kEntropyNvfp4SlotStreamSymbols, fs);
        if (size < 0) {
            s_overflow = true;
        }
    }
    __syncthreads();
    if (s_overflow) {
        slot_valid[static_cast<std::int64_t>(page) * valid_page_stride + head] = 0;
        return;
    }

    // --- copy the uncompressed E4M3FN scale page to the fixed tail region ----
    {
        const std::uint8_t* scale_src = page_scales;
        std::uint8_t* scale_dst = slot + slot_bytes - kEntropyNvfp4SlotScaleBytes;
        for (int i = tid; i < kEntropyNvfp4SlotScaleBytes; i += Streams) {
            scale_dst[i] = scale_src[i];
        }
    }
    __syncthreads();
    slot_valid[static_cast<std::int64_t>(page) * valid_page_stride + head] = 1;
}

// One block per output half. Thread t (0..15) decodes stream t of the
// selected half into dst + t * 256 contiguous packed bytes.
__global__ void entropy_nvfp4_slot_decode_half_kernel(const std::uint8_t* __restrict__ slots,
                                                      int slot_bytes,
                                                      const std::int32_t* __restrict__ slot_ids,
                                                      const std::int32_t* __restrict__ halves,
                                                      std::uint8_t* __restrict__ dst,
                                                      int half_bytes) {
    const int item = static_cast<int>(blockIdx.x);
    const int stream = static_cast<int>(threadIdx.x);
    if (stream >= kEntropyNvfp4SlotStreamsPerHalf) { return; }
    const std::int32_t slot_id = slot_ids[item];
    const std::int32_t half    = halves[item];
    const std::uint8_t* slot   = slots + static_cast<std::int64_t>(slot_id) * slot_bytes;
    std::uint8_t* stream_dst =
        dst + static_cast<std::int64_t>(item) * half_bytes + stream * kEntropyNvfp4SlotStreamBytes;
    if (!entropy_nvfp4_slot_decode_stream(slot, half, stream, stream_dst)) {
        for (int i = 0; i < kEntropyNvfp4SlotStreamBytes; ++i) { stream_dst[i] = 0; }
    }
}

// Direct grid decode: blockIdx.x selects the half (0/1), blockIdx.y selects
// kv_head. Thread t decodes stream t of that half into item
// (kv_head * 2 + half) of the contiguous output buffer.
__global__ void entropy_nvfp4_slot_decode_half_grid_kernel(const std::uint8_t* __restrict__ slots,
                                                           int slot_bytes,
                                                           std::int32_t slot_base,
                                                           std::uint8_t* __restrict__ dst,
                                                           int half_bytes) {
    const int half   = static_cast<int>(blockIdx.x);
    const int head   = static_cast<int>(blockIdx.y);
    const int stream = static_cast<int>(threadIdx.x);
    if (stream >= kEntropyNvfp4SlotStreamsPerHalf) { return; }
    const std::uint8_t* slot =
        slots + static_cast<std::int64_t>(slot_base + head) * slot_bytes;
    std::uint8_t* stream_dst =
        dst + static_cast<std::int64_t>(head * 2 + half) * half_bytes +
        stream * kEntropyNvfp4SlotStreamBytes;
    if (!entropy_nvfp4_slot_decode_stream(slot, half, stream, stream_dst)) {
        for (int i = 0; i < kEntropyNvfp4SlotStreamBytes; ++i) { stream_dst[i] = 0; }
    }
}

// Scatters the uncompressed 1024-byte scale tail of every (page, kv_head)
// slot into the matching paged scale plane. Grid = (kv_heads, page_count);
// each 64-thread block copies 16-byte vectors. slots uses the host-cold
// layout: page stride = slot_page_stride, head stride = slot_bytes.
__global__ void entropy_nvfp4_slot_scales_scatter_kernel(
    const std::uint8_t* __restrict__ slots, int slot_bytes, int slot_page_stride,
    const std::int32_t* __restrict__ page_ids, int scale_page_stride,
    std::uint8_t* __restrict__ scales) {
    constexpr int ScaleBytes = 1024;
    const int head            = static_cast<int>(blockIdx.x);
    const int page            = static_cast<int>(blockIdx.y);
    const int vec             = static_cast<int>(threadIdx.x);
    if (vec >= ScaleBytes / 16) { return; }
    const std::uint8_t* src =
        slots + static_cast<std::int64_t>(page) * slot_page_stride +
        static_cast<std::int64_t>(head) * slot_bytes + (slot_bytes - ScaleBytes) + vec * 16;
    std::uint8_t* dst =
        scales + static_cast<std::int64_t>(page_ids[page]) * scale_page_stride +
        static_cast<std::int64_t>(head) * ScaleBytes + vec * 16;
    const uint4 value = load_vec<uint4>(src);
    store_vec(dst, value);
}

} // namespace ninfer::ops::detail
