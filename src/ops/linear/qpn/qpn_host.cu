// qpn_host.cu — ninfer 原生 wrapper (移植自 v100-skinny torch host 段).
// 消费 qpn_prepack_proto.py 位精确 prepack 布局:
//   codes: [tile][group][lane][16B] -> 连续 u8; scales: [N/32][16] u8
// M 档位: qpn_simt M<=3 (decode), qpn2/qpn M<=16 (verify/宽批).
#include "ops/linear/qpn/qpn_kernels.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::ops::qpn {

void gemm_qpn_simt(const void* x_half, const void* codes, const void* scales, float gscale,
                   void* y_half, int m, int k, int n, cudaStream_t stream) {
    if (m < 1 || m > 3) { throw std::invalid_argument("qpn_simt supports M 1..3"); }
    if (k % 64 != 0 || n % 32 != 0) { throw std::invalid_argument("qpn_simt: K%64 N%32"); }
    const int smem = static_cast<int>(m * 4096 * sizeof(half2)) + 4 * 32 * m * sizeof(float);

#define NINFER_LAUNCH_QPN_SIMT(MM)                                                        \
    do {                                                                                   \
        auto* fn = skinny_nvfp4_qpn_simt<MM>;                                              \
        if (smem > 48 * 1024) {                                                            \
            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);   \
        }                                                                                  \
        fn<<<dim3(n / 32), dim3(128), smem, stream>>>(                                     \
            static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales),       \
            static_cast<const half*>(x_half), static_cast<half*>(y_half), n, k, gscale);   \
    } while (0)

    switch (m) {
    case 1: NINFER_LAUNCH_QPN_SIMT(1); break;
    case 2: NINFER_LAUNCH_QPN_SIMT(2); break;
    default: NINFER_LAUNCH_QPN_SIMT(3); break;
    }
#undef NINFER_LAUNCH_QPN_SIMT
}

// M-dispatch entry (W4): the band table mirrors qpn_kernels.cuh's frontier
// comment — simt M<=3, mma.m8n8k4 MT=1 for M 4..8, MT=2 for M 9..16 (MT is the
// number of 8-row A tiles; MT=2 decodes the weight stream once for M 9..16).
// Weights must already be in the qpn_prepack_proto layout (bit-exact with the
// marlin _qpn_prepack reference).
void gemm_qpn(const void* x_half, const void* codes, const void* scales, float gscale,
              void* y_half, int m, int k, int n, cudaStream_t stream) {
    if (m < 1) { throw std::invalid_argument("gemm_qpn: M must be >= 1"); }
    if (k % 64 != 0 || n % 32 != 0) { throw std::invalid_argument("gemm_qpn: K%64 N%32"); }
    if (m <= 3) {
        gemm_qpn_simt(x_half, codes, scales, gscale, y_half, m, k, n, stream);
        return;
    }
    const dim3 grid(static_cast<unsigned>(n / 32));
    const dim3 block(128);
    const auto* qcodes  = static_cast<const uint8_t*>(codes);
    const auto* qscales = static_cast<const uint8_t*>(scales);
    const auto* x       = static_cast<const half*>(x_half);
    auto* y             = static_cast<half*>(y_half);
    if (m <= 8) {
        skinny_nvfp4_qpn<1><<<grid, block, 0, stream>>>(qcodes, qscales, x, y, n, k, m, gscale);
        return;
    }
    if (m <= 16) {
        skinny_nvfp4_qpn<2><<<grid, block, 0, stream>>>(qcodes, qscales, x, y, n, k, m, gscale);
        return;
    }
    throw std::invalid_argument("gemm_qpn: M > 16 belongs to the wmma band (17..64)");
}

} // namespace ninfer::ops::qpn
