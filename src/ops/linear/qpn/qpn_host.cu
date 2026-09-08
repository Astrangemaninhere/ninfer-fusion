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

} // namespace ninfer::ops::qpn
