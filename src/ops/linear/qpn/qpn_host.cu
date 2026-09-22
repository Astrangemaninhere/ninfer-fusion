// qpn_host.cu — ninfer 原生 wrapper (移植自 v100-skinny torch host 段).
// 消费 qpn_prepack_proto.py 位精确 prepack 布局:
//   codes: [tile][group][lane][16B] -> 连续 u8; scales: [N/32][16] u8
// M 档位: qpn_simt M<=3 (decode), qpn2/qpn M<=16 (verify/宽批).
#include "ops/linear/qpn/qpn_host.h"
#include "ops/linear/qpn/qpn_kernels.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

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
    // M >= 17 IS THE WIDE-M BAND, AND THIS ENTRY CANNOT SERVE IT FOR A LAYOUT REASON --
    // not because the band has no kernel. skinny_nvfp4_wmma<WN, WM, KC> (this header, the
    // "wmma (M>=9)" / frontier-17..64 kernel) is the band's kernel and it consumes the
    // CHECKPOINT-NATIVE planes ([N][K/2] codes + [N][K/16] e4m3 scales), while `codes`/
    // `scales` here are the qpn_prepack permutation. Feeding either blob to the other
    // kernel is silent numerical corruption, so the wide band gets its own entry
    // (gemm_qpn_wmma_native below) instead of an M-keyed layout guess. What WAS wrong here
    // is the text: it said M > 16 "belongs to the wmma band (17..64)" and named no entry,
    // which src/core/kernel_route.h turned into a route-table row claiming the FAMILY lacks
    // the band and that marlin would have covered it. The kernel was in this header the
    // whole time; only the host arm was missing.
    throw std::invalid_argument(
        "gemm_qpn: M > 16 is the wide-M band. This entry's precondition is the qpn_prepack "
        "layout and it serves M 1..16; call gemm_qpn_wmma_native() for M 17..64, which "
        "consumes the checkpoint-native codes[N][K/2] + scales[N][K/16] planes. The band's "
        "kernel is skinny_nvfp4_wmma<WN, WM, KC> in qpn_kernels.cuh and IT IS IN THIS BUILD.");
}

// ---------------------------------------------------------------------------
// THE WIDE-M ARM: skinny_nvfp4_wmma, in the CHECKPOINT-NATIVE layout.
// ---------------------------------------------------------------------------
// WHY A SECOND ENTRY INSTEAD OF AN EXTRA BAND IN gemm_qpn: the layout axis and the M axis
// would then be entangled, and the failure mode of getting that wrong is silent numbers --
// the class of failure the prepack precondition exists to prevent. Two entry points, two
// preconditions, no inference.
//
// EVERY NUMBER IN THE LAUNCH IS A CONSTRAINT OF THE KERNEL, not a tuning choice made here:
//
//   * WM = 4  ->  MT = WM * 16 = 64 rows per CTA, which IS the band's ceiling. The kernel
//     masks its stores with `gm < m_real` but reads A at `x + (size_t)m * K` for the whole
//     tile, so M > MT would read past the activation. M > 64 therefore REFUSES here, and
//     that is a kernel fact rather than a build fact.
//   * KC must satisfy CSEG = NT*(KC/16)/NTHREADS and XSEG = MT*(KC/8)/NTHREADS, both exact
//     and both >= 1 -- the kernel static_asserts them, so a wrong KC is a compile error
//     rather than a silent one. WN = 2 / KC = 128 gives NT = 32, NTHREADS = 256,
//     CSEG = 1, XSEG = 4: the smallest cell that tiles.
//   * KC also makes `K % KC == 0` a PRECONDITION: the k loop steps by KC with no remainder
//     handling, so this entry refuses K % 128 != 0 instead of running a tail wrong.
//   * N must be a multiple of NT = 32: gridDim.x = N / NT and the kernel writes its whole
//     tile column unconditionally.
//   * smem = 2 * 16 * (WN + WM) * (KC + 16) = 27648 B, under the 48 KB default ceiling, so
//     no cudaFuncSetAttribute is needed. (set_smem_opt's 96 KB opt-in is for the larger
//     (WN, WM, KC) cells of the sweep table, not for this one.)
//
// WHAT IS *NOT* CLAIMED HERE: the wide band's channel is nvcuda::wmma m16n16k16, a
// DIFFERENT channel from the m8n8k4 quadpairs the M 4..16 bands use (src/core/arch_caps.h
// says so explicitly). This entry does not decide which rungs that channel is hardware on;
// that is a route-table question (src/core/kernel_route.h). This function refuses a shape
// it cannot serve and nothing else.
void gemm_qpn_wmma_native(const void* x_half, const void* codes, const void* scales,
                          float gscale, void* y_half, int m, int k, int n,
                          cudaStream_t stream) {
    constexpr int WN         = 2;
    constexpr int WM         = 4;
    constexpr int KC         = 128;
    constexpr int NT         = WN * 16;   // output columns per CTA
    constexpr int MT         = WM * 16;   // output rows per CTA == the band ceiling
    constexpr int NTHREADS   = WN * WM * 32;
    constexpr int SMEM_BYTES = 2 * 16 * (WN + WM) * (KC + 16);

    if (m < 1 || m > MT) {
        throw std::invalid_argument(
            "gemm_qpn_wmma_native: M must be 1..64 -- skinny_nvfp4_wmma stages WM*16 = 64 "
            "activation rows per CTA and has no m-tile loop");
    }
    if (k % KC != 0 || n % NT != 0) {
        throw std::invalid_argument("gemm_qpn_wmma_native: K%128 N%32");
    }

    const dim3 grid(static_cast<unsigned>(n / NT));
    const dim3 block(NTHREADS);
    skinny_nvfp4_wmma<WN, WM, KC><<<grid, block, SMEM_BYTES, stream>>>(
        static_cast<const std::uint8_t*>(codes), static_cast<const std::uint8_t*>(scales),
        static_cast<const half*>(x_half), static_cast<half*>(y_half), n, k, m, gscale);
    // gemm_qpn above does not check its launch; this arm does, because an unlaunched wide
    // band would otherwise leave the caller's output buffer untouched and be read as a
    // result.
    const cudaError_t launched = cudaGetLastError();
    if (launched != cudaSuccess) {
        throw std::runtime_error(std::string("gemm_qpn_wmma_native: launch failed: ") +
                                 cudaGetErrorString(launched));
    }
}

} // namespace ninfer::ops::qpn
