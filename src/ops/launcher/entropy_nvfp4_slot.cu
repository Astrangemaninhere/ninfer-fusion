#include "ops/launcher/entropy_nvfp4_slot.h"

#include "core/device.h"
#include "ops/kernel/entropy_nvfp4_slot_kernels.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void entropy_nvfp4_slot_encode_launch(const std::uint8_t* codes, const std::uint8_t* scales,
                                      int kv_heads, int page_count, std::uint8_t* slots,
                                      int slot_bytes, std::int32_t* slot_valid,
                                      const std::int32_t* page_ids, int valid_page_stride,
                                      cudaStream_t stream) {
    const dim3 grid(kv_heads, page_count);
    entropy_nvfp4_slot_encode_kernel<<<grid, kEntropyNvfp4SlotStreams, 0, stream>>>(
        codes, scales, kv_heads, slots, slot_bytes, slot_valid, page_ids, valid_page_stride);
    CUDA_CHECK(cudaGetLastError());
}

void entropy_nvfp4_slot_decode_grid_launch(const std::uint8_t* slots, int slot_bytes,
                                           std::int32_t slot_base, int kv_heads,
                                           std::uint8_t* dst, cudaStream_t stream) {
    const dim3 grid(2, kv_heads);
    entropy_nvfp4_slot_decode_half_grid_kernel<<<grid, kEntropyNvfp4SlotStreamsPerHalf, 0,
                                                 stream>>>(slots, slot_bytes, slot_base, dst,
                                                           kEntropyNvfp4SlotHalfBytes);
    CUDA_CHECK(cudaGetLastError());
}

void entropy_nvfp4_slot_scales_scatter_launch(const std::uint8_t* slots, int slot_bytes,
                                              int slot_page_stride, int kv_heads,
                                              int page_count, const std::int32_t* page_ids,
                                              int scale_page_stride, std::uint8_t* scales,
                                              cudaStream_t stream) {
    const dim3 grid(kv_heads, page_count);
    entropy_nvfp4_slot_scales_scatter_kernel<<<grid, 64, 0, stream>>>(
        slots, slot_bytes, slot_page_stride, page_ids, scale_page_stride, scales);
    CUDA_CHECK(cudaGetLastError());
}

// Restore the full (page, kv_head) K|V plane from its entropy slot into the
// native NVFP4 page-major geometry. dec items (head*2+half) hold the 16
// decoded streams; stream t covers rows (2t, 2t+1) of that half, so the
// grid copies each stream's two 128 B rows to the row-major code plane.
__global__ void entropy_nvfp4_slot_restore_plane_kernel(
    const std::uint8_t* __restrict__ dec, int kv_heads,
    std::uint8_t* __restrict__ dst_codes) {
    const int head = static_cast<int>(blockIdx.y);
    const int half = static_cast<int>(blockIdx.x);
    const int stream = static_cast<int>(threadIdx.x);
    if (stream >= kEntropyNvfp4SlotStreamsPerHalf) { return; }
    const std::uint8_t* item =
        dec + static_cast<std::int64_t>(head * 2 + half) * kEntropyNvfp4SlotHalfBytes +
        stream * kEntropyNvfp4SlotStreamBytes;
    const int row0 = (half * kEntropyNvfp4SlotStreamsPerHalf + stream) * 2;
    std::uint8_t* dst =
        dst_codes + static_cast<std::int64_t>(head) * (64 * 128) + row0 * 128;
#pragma unroll
    for (int i = 0; i < 128; ++i) { dst[i] = item[i]; }
#pragma unroll
    for (int i = 0; i < 128; ++i) { dst[128 + i] = item[128 + i]; }
}

void entropy_nvfp4_slot_restore_plane_launch(const std::uint8_t* slots, int slot_bytes,
                                             std::int32_t slot_base, int kv_heads,
                                             std::uint8_t* dec_scratch,
                                             std::uint8_t* dst_codes, cudaStream_t stream) {
    entropy_nvfp4_slot_decode_grid_launch(slots, slot_bytes, slot_base, kv_heads, dec_scratch,
                                          stream);
    const dim3 grid(2, kv_heads);
    entropy_nvfp4_slot_restore_plane_kernel<<<grid, kEntropyNvfp4SlotStreamsPerHalf, 0, stream>>>(
        dec_scratch, kv_heads, dst_codes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
