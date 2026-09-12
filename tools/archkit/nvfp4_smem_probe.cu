// _nvfp4_smem_probe.cu — standalone smem probe for the nvfp4 Muse decode kernel.
//
// Answers the open question in _TODO.md 65/66: which TT instantiation actually
// exceeds the sm_120 per-block opt-in limit, and what the real static shared
// size is (compiler-reported, not hand-counted). No engine build required:
// just this TU + the kernel headers, compiled with -I <repo>/src.
//
// Build (Windows, MSVC env via vcvars64):
//   nvcc -O2 -std=c++17 -arch=sm_120a -I <repo>/src _nvfp4_smem_probe.cu -o probe.exe
#include "ops/kernel/gqa_attention_decode_nvfp4.cuh"

#include <cstdio>
#include <cuda_runtime.h>

namespace {

using Geo = ninfer::ops::GqaMuseGeometry;  // 32 q-heads / 2 kv-heads / head_dim 128

template <int TT, int Wc>
void probe_one() {
    constexpr int KeyBlock = 32;                       // launcher: gqa_attention_decode.cu Muse branch
    constexpr int kTileBytes = 4 * KeyBlock * 128 + 4 * KeyBlock * 16;
    constexpr int kRowTiles = (TT * Geo::GroupSize + 15) / 16;
    constexpr std::size_t kRBytes =
        2ull * kTileBytes + static_cast<std::size_t>(kRowTiles) * 16 * 64 + 2ull * Wc * 16 * 64;
    constexpr std::size_t kVDynamic = static_cast<std::size_t>(KeyBlock) * 256 * 2;  // Iso3V V tile
    constexpr std::size_t kDynamicBytes = kRBytes + kVDynamic;

    auto kernel = ninfer::ops::gqa_attention_decode_nvfp4_tiled_kernel<
        Geo, TT, Wc, 1, KeyBlock, /*DynamicArena=*/true, /*Iso3V=*/true>;
    cudaFuncAttributes attr{};
    const cudaError_t e_attr = cudaFuncGetAttributes(&attr, kernel);
    const cudaError_t e_set = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kDynamicBytes));
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    const std::size_t total = static_cast<std::size_t>(attr.sharedSizeBytes) + kDynamicBytes;
    std::printf(
        "TT=%d Wc=%2d static=%7u dyn=%7zu total=%7zu (%.1f KiB) optin=%d "
        "getattr=%s setattr=%s\n",
        TT, Wc, attr.sharedSizeBytes, kDynamicBytes, total,
        static_cast<double>(total) / 1024.0, optin,
        cudaGetErrorName(e_attr), cudaGetErrorName(e_set));
}

}  // namespace

int main() {
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    std::printf("device=%s sm_%d%d optin=%zu\n", prop.name, prop.major, prop.minor,
                static_cast<std::size_t>(prop.sharedMemPerBlockOptin));
    // launcher kWc table for Muse (GroupSize == 16): TT1->4 TT2->8 TT3->6 TT4->8 TT5->10 TT6->12
    probe_one<1, 4>();
    probe_one<2, 8>();
    probe_one<3, 6>();
    probe_one<4, 8>();
    probe_one<5, 10>();
    probe_one<6, 12>();
    return 0;
}
