// temp: standalone verdict test for the Muse vocabulary NVFP4 decode GEMV.
//
// All codes are e2m1(1.0) = 2, all scales are e4m3(1.0) = 0x38, x is all ones and the weight
// divisor is 1, so every output row must equal K = 6656. Rows that come back 0 (or otherwise)
// localise a tile/row indexing bug in the M=1 GEMV path (_TODO.md 102).
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops::detail;

int main() {
    using Geo      = Nvfp4MuseVocabularyGeometry;
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geo>::Type;
    constexpr int N = Geo::kOutputRows;   // 202112
    constexpr int K = Geo::kInputRows;    // 6656
    constexpr int kGroups = K / 16;
    constexpr int kScaleTiles = kGroups / 4;
    constexpr int kMTiles = N / 128;

    std::printf("geometry N=%d K=%d groups=%d scale_tiles=%d m_tiles=%d rowsPerCta=%d threads=%d\n",
                N, K, kGroups, kScaleTiles, kMTiles, Schedule::kRowsPerCta, Schedule::kThreads);

    const std::size_t code_bytes  = static_cast<std::size_t>(N) * (K / 2);
    const std::size_t scale_bytes = static_cast<std::size_t>(kMTiles) * kScaleTiles * 512;
    std::printf("code plane %.1f MiB, scale plane %.1f MiB\n",
                static_cast<double>(code_bytes) / 1048576.0,
                static_cast<double>(scale_bytes) / 1048576.0);

    std::vector<std::uint8_t> codes(code_bytes, 0x22);      // e2m1 1.0 in both nibbles
    std::vector<std::uint8_t> scales(scale_bytes, 0x38);    // e4m3 1.0
    std::vector<__nv_bfloat16> x(K, __float2bfloat16(1.0f));
    std::vector<__nv_bfloat16> out(N, __float2bfloat16(-1.0f));

    std::uint8_t *d_codes, *d_scales;
    __nv_bfloat16 *d_x, *d_out;
    if (cudaMalloc(&d_codes, code_bytes) != cudaSuccess ||
        cudaMalloc(&d_scales, scale_bytes) != cudaSuccess ||
        cudaMalloc(&d_x, K * 2) != cudaSuccess || cudaMalloc(&d_out, N * 2) != cudaSuccess) {
        std::printf("cudaMalloc failed\n");
        return 1;
    }
    cudaMemcpy(d_codes, codes.data(), code_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales.data(), scale_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out.data(), N * 2, cudaMemcpyHostToDevice);

    const Nvfp4ContiguousOutput output{d_out, N};
    constexpr int kBlocks = N / Schedule::kRowsPerCta;
    nvfp4_gemv_kernel<Geo, Schedule><<<kBlocks, Schedule::kThreads, 0, nullptr>>>(
        d_x, d_codes, d_scales, 1.0f, Nvfp4IdentityEpilogue{}, output);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(out.data(), d_out, N * 2, cudaMemcpyDeviceToHost);

    int exact = 0, zero = 0, other = 0;
    int first_bad = -1, last_bad = -1;
    for (int i = 0; i < N; ++i) {
        const float v = __bfloat162float(out[i]);
        if (v == 6656.0f) {
            ++exact;
        } else if (v == 0.0f) {
            ++zero;
            if (first_bad < 0) { first_bad = i; }
            last_bad = i;
        } else {
            ++other;
            if (first_bad < 0) { first_bad = i; }
            last_bad = i;
        }
    }
    std::printf("exact=%d zero=%d other=%d first_bad=%d last_bad=%d\n", exact, zero, other,
                first_bad, last_bad);
    if (first_bad >= 0) {
        std::printf("sample rows: ");
        for (int i = first_bad; i < first_bad + 4 && i < N; ++i) {
            std::printf("[%d]=%.1f ", i, __bfloat162float(out[i]));
        }
        std::printf("\n");
    }
    std::printf("%s\n", exact == N ? "MUSE_VOCAB_GEMV_PASS" : "MUSE_VOCAB_GEMV_FAIL");
    return exact == N ? 0 : 1;
}
