#pragma once

// ---------------------------------------------------------------------------
// bf16_mma_fp16.cuh -- THE fp16 TENSOR-CORE ARM FOR A BF16 WEIGHT PLANE.
// ---------------------------------------------------------------------------
// WHY THIS FILE EXISTS, AND WHAT WAS MISSING BEFORE IT.
//
// On sm_70 (Volta) and sm_75 (Turing) the rung HAS fp16 tensor cores and the
// tree had no GEMM that could use them for a BF16 artifact. The table's own
// BF16 row (src/core/arch_caps.h, kFormatRequirements) records the absence in
// these words: "There is no fp16 fallback for BF16, and that is a measured
// absence rather than an oversight. The fp16 avenue this table honours is
// fp16_fallback_executable() -> the QPN family -> mma.sync.aligned.m8n8k4, and
// the QPN kernels consume PACKED 4-BIT e2m1 CODES ... Neither reads a bf16
// weight plane."
//
// select_route()'s own no-fp16-route arm names the missing piece exactly:
//   "Missing kernel: an mma_s8 port of the groupwise-int GEMMs, or an explicit
//    bf16->fp16 weight conversion ahead of an m8n8k4 GEMM -- the m8n8k4 helper
//    now exists (ops/common/mma.cuh, mma_f16_m8n8k4; it assembles from sm_70
//    up, measured), so the missing piece for these formats is the conversion
//    and the kernel that drives it, not the mma channel."
//
// This file is that conversion and that kernel. It adds NO new instruction
// family and NO new claim about a card: it uses the two fp16 mma helpers
// ops/common/mma.cuh already owns, and it reads the bf16 plane the artifact
// ALREADY stores (no requantization, no second weight copy).
//
// ---------------------------------------------------------------------------
// THE TWO CHANNELS, AND WHY THERE ARE TWO RATHER THAN ONE
// ---------------------------------------------------------------------------
// mma.sync.aligned.m8n8k4 is the ONLY fp16 tensor-core form Volta has, and it
// is the channel kQpnMmaRungs (src/core/arch_caps.h) measures per rung:
// HardwareMma884 (HMMA.884) on sm_70 and sm_75, EmulatedFp16Pipe from sm_80 up.
// mma.sync.aligned.m16n8k8 needs sm_75, and it is the form that stays HARDWARE
// on every rung above Turing as well -- which is the whole reason this arm has
// a second channel at all.
//
// MEASURED THIS LINE (CUDA 13.3, `-cubin -arch=<a>` + `nvdisasm -c`, on a probe
// TU that forces both helpers -- dl/fp16route/probe/chan_probe.cu, logs in
// dl/fp16route/logs/census_<arch>.sass):
//
//   arch     mma_f16_m8n8k4                 mma_f16_m16n8k8
//   sm_75    HMMA.884 = 4, emul CALL = 0    HMMA.1688, emul CALL = 0
//   sm_80    0 HMMA.884, 1 emul CALL        HMMA, 0 emul CALL
//   sm_90    0 HMMA.884, 1 emul CALL        HMMA, 0 emul CALL
//   sm_120a  0 HMMA.884, 1 emul CALL        HMMA, 0 emul CALL
//
// So: on Volta and Turing BOTH channels are real tensor-core instructions; from
// Ampere up the m8n8k4 channel is a CALL into ptxas's FFMA routine while the
// m16n8k8 channel is still HMMA. The rung decides the channel, which is why the
// route (not this file) picks it -- see bf16_fp16_route.h.
//
// sm_70 IS NOT COMPILABLE ON THIS BOX and is not claimed to be: every toolkit
// here (13.0/13.1/13.3) rejects `-arch=sm_70` with "nvcc fatal : Unsupported gpu
// architecture" (12.8, which could, is gone). The m8n8k4 CHANNEL's sm_70
// lowering is therefore inherited from kQpnMmaRungs' own sm_70 row -- a CUDA
// 12.8 measurement of the same single channel, which is the same inheritance
// fp16_fallback_executable() already performs -- and the whole-kernel sm_70
// census is NOT taken. Say that rather than implying it.
//
// ---------------------------------------------------------------------------
// PRECISION, STATED RATHER THAN HIDDEN
// ---------------------------------------------------------------------------
// bf16 -> fp16 is exact in the mantissa (fp16 carries 11 against bf16's 8) and
// NARROWER in range (fp16's exponent is 5 bits against bf16's 8), so
// |x| > 65504 overflows. The conversion here is the IEEE one,
// __floats2half2_rn(__bfloat162float(b)): exact inside fp16's range, +-inf
// outside it. That is the same range caveat the BF16 row of kFormatRequirements
// already carries ("bf16 -> fp16 is precision-up and RANGE-down"), and the
// accumulator is f32 throughout, so no accumulation error is added. The SIMT arm
// the table selects today is EXACT where this one saturates, and that trade is
// why the route arm reports both and the caller may decline (bf16_fp16_plan()).
//
// ---------------------------------------------------------------------------
// WHERE THIS IS *NOT* THE RIGHT ANSWER, MEASURED RATHER THAN ASSUMED
// ---------------------------------------------------------------------------
// A warp-level fp16 mma atom consumes 8 activation rows (m8n8k4) or 8 activation
// columns (m16n8k8). Below that token count the kernel issues the same mma work
// for fewer useful tokens, so the useful-work fraction is T/8 and the arm is
// *worse* than the FFMA GEMV it would replace: a T=1 GEMV is weight-bandwidth
// bound and the tensor core cannot reduce the bytes it must read. So the plan
// requires T >= kBf16Fp16MinTokens (8) and leaves T < 8 to the existing shape
// table. That threshold is a property of the atom, not a preference.

#include "core/arch_caps.h"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// ---------------------------------------------------------------------------
// The channel. NOT keyed on __CUDA_ARCH__: the channel is a ROUTE fact (the rung
// the tables answered for), so both kernels are compiled into every cubin and
// the selector chooses. Below sm_70 the helpers trap (NINFER_MMA_HAS_M8N8K4_F16 /
// NINFER_MMA_HAS_M16N8K8_F16), which is the tree's existing fail-closed shape.
// ---------------------------------------------------------------------------
enum class Bf16Fp16Channel : std::uint8_t {
    Mma8n8k4,  // Volta's only fp16 tensor-core form; the channel kQpnMmaRungs measures
    Mma16n8k8, // Turing's native form, and hardware all the way up (measured above)
};

[[nodiscard]] inline const char* bf16_fp16_channel_name(Bf16Fp16Channel c) noexcept {
    switch (c) {
    case Bf16Fp16Channel::Mma8n8k4: return "bf16 fp16 mma.m8n8k4 (HMMA.884 on Volta/Turing)";
    case Bf16Fp16Channel::Mma16n8k8:
        return "bf16 fp16 mma.m16n8k8 (HMMA.1688 on Turing; HMMA.16816 from Ampere up)";
    }
    return "unknown-channel";
}

inline constexpr int kBf16Fp16MinTokens = caps::kBf16Fp16MmaActivationExtent;
// ONE NUMBER, TWO SURFACES: the route arm names the channel and the engine-side plan decides
// whether to take it, so the atom's activation extent is defined in the capability header
// (caps::kBf16Fp16MmaActivationExtent) and this is an alias rather than a second value. The
// static_assert is what makes the alias a fact instead of a comment.
static_assert(kBf16Fp16MinTokens == 8, "the fp16 mma atom consumes 8 activation rows");

// bf16 -> fp16, 2 at a time. The IEEE conversion; see the header note above.
__device__ __forceinline__ unsigned bf16x2_to_f16x2(unsigned b) {
    const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(&b);
    const __half2 h        = __floats2half2_rn(__bfloat162float(p.x), __bfloat162float(p.y));
    return *reinterpret_cast<const unsigned*>(&h);
}

// ---------------------------------------------------------------------------
// CHANNEL m8n8k4 -- the Volta route (rung 70), and the only fp16 tensor-core
// form sm_70 has.
//
// FRAGMENT MAP: taken verbatim from this tree's own proven m8n8k4 GEMM,
// skinny_nvfp4_mma8 (src/ops/linear/qpn/qpn_kernels.cuh:645-652), which derived
// it empirically on V100 with mma8_probe.cu:
//
//   A row-major / B col-major: QP lanes {0-3,16-19} hold row/col
//   (lane&3)+4*(lane>=16), 4 contiguous k each.
//   C fp32: reg i of lane L -> row (i&2)|(L>=16?4:0)|(L&1),
//                            col (i&1)|(((L>>1)&1)<<1)|((i>>2)<<2).
//   Warp = 8 weight rows; the 4 QPs split K in 64-element superchunks
//   (32B-sector-coalesced reads), butterfly-reduced once at the end.
//
// WHERE IT CAME FROM IN THE SOURCE TREE: exactly two users of the m8n8k4 fp16
// channel existed (qpn_kernels.cuh:729 and :929), both in the QPN family, both
// consuming PACKED e2m1/e4m3 codes. This kernel is the third user, and the first
// one that consumes a BF16 plane.
//
// A is the WEIGHT plane (8 output rows x K) and B is the ACTIVATION plane
// (8 tokens x K) -- the QPN site's own roles. Both operands are read as 16
// CONTIGUOUS values at k = kb, converted once, and fed as the register pairs the
// helper takes. blockIdx.y walks the token axis in steps of 8.
// ---------------------------------------------------------------------------
template <int KC>
__global__ void bf16_mma8_kernel(const __nv_bfloat16* __restrict__ w, // [N][K] row-major
                                 const __nv_bfloat16* __restrict__ x, // [T][K] row-major
                                 __nv_bfloat16* __restrict__ y,       // [T][N], leading dim N
                                 int N, int K, int T) {
    extern __shared__ char smem_raw[];
    half2* xs = reinterpret_cast<half2*>(smem_raw); // [8][PITCH] plain k order, fp16
    constexpr int PITCH = KC / 2 + 1;               // odd pitch: 8 rows spread the banks

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qp  = (lane >> 2) & 3;
    const int idx = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int n0  = (blockIdx.x * (int)(blockDim.x >> 5) + warp) * 8;
    const int t0  = (int)blockIdx.y * 8;
    const __nv_bfloat16* wrow = w + (std::size_t)(n0 + idx) * K;

    float c[8];
#pragma unroll
    for (int i = 0; i < 8; i++) c[i] = 0.f;

    for (int k0 = 0; k0 < K; k0 += KC) {
        __syncthreads();
        // Stage the 8 activation rows of this K chunk as fp16 -- one conversion per element.
        for (int t = (int)threadIdx.x; t < 8 * (KC / 2); t += (int)blockDim.x) {
            const int m = t / (KC / 2), j = t % (KC / 2);
            half2 v     = __float2half2_rn(0.f);
            const int tok = t0 + m;
            if (tok < T) {
                const unsigned raw =
                    *reinterpret_cast<const unsigned*>(x + (std::size_t)tok * K + k0 + 2 * j);
                const unsigned h = bf16x2_to_f16x2(raw);
                v                = *reinterpret_cast<const half2*>(&h);
            }
            xs[m * PITCH + j] = v;
        }
        __syncthreads();

#pragma unroll
        for (int s = 0; s < KC / 64; s++) {
            const int kb = k0 + s * 64 + qp * 16; // this QP's 16-k window
            // A: 16 contiguous bf16 of THIS thread's weight row -> 8 f16x2 -> 4 mma quads.
            half2 af[4][2];
            {
                const unsigned* wq = reinterpret_cast<const unsigned*>(wrow + kb); // 8 words
                unsigned h[8];
#pragma unroll
                for (int i = 0; i < 8; i++) h[i] = bf16x2_to_f16x2(wq[i]);
#pragma unroll
                for (int j = 0; j < 4; j++) { // af[j] = the 4 CONTIGUOUS k of mma j, in k order
                    af[j][0] = *reinterpret_cast<const half2*>(&h[2 * j]);
                    af[j][1] = *reinterpret_cast<const half2*>(&h[2 * j + 1]);
                }
            }
            const half2* xrow = xs + idx * PITCH + ((kb - k0) >> 1);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const half2 b0 = xrow[2 * j], b1 = xrow[2 * j + 1];
                const unsigned* A = reinterpret_cast<const unsigned*>(af[j]);
                ::ninfer::ops::mma_f16_m8n8k4(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], A[0],
                                              A[1], *reinterpret_cast<const unsigned*>(&b0),
                                              *reinterpret_cast<const unsigned*>(&b1));
            }
        }
    }

    // The 4 QP k-partials butterfly down to one, then the C map above writes it out.
#pragma unroll
    for (int i = 0; i < 8; i++) {
        c[i] += __shfl_xor_sync(0xffffffffu, c[i], 4);
        c[i] += __shfl_xor_sync(0xffffffffu, c[i], 8);
    }
    if ((lane & 12) == 0) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r   = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
            const int cc  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            const int tok = t0 + cc;
            if (n0 + r < N && tok < T) {
                y[(std::size_t)tok * N + n0 + r] = __float2bfloat16_rn(c[i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CHANNEL m16n8k8 -- the Turing route (rung 75), and the channel that stays
// HARDWARE from sm_75 up, INCLUDING the sm_120a cubin this box executes.
//
// FRAGMENT MAP: this tree's own m16n8k16 map (ops/linear/bf16/bf16_gemm_mma.cuh,
// bf16_gemm_mma_kernel's epilogue) with the K=16 pair dropped -- m16n8k8 is the
// same atom with a0,a1 and b0 only, which is exactly the signature
// mma_f16_m16n8k8 already has in ops/common/mma.cuh:
//
//   A (weights, 16 rows x 8 k, row-major):
//     a0 = {A[gid][2*lid], A[gid][2*lid+1]}, a1 = {A[gid+8][2*lid], A[gid+8][2*lid+1]}
//   B (activations, 8 k x 8 tokens, k-contiguous per token = col-major):
//     b0 = {B[2*lid][gid], B[2*lid+1][gid]}
//   C (16 x 8 f32), gid = lane>>2, lid = lane&3:
//     c0 = C[gid][2*lid], c1 = C[gid][2*lid+1], c2 = C[gid+8][2*lid], c3 = C[gid+8][2*lid+1]
//
// NO ldmatrix and no shared-memory staging: every fragment element is read
// directly from global. That costs bandwidth and buys independence from the
// big tiled GEMM's smem swizzle. This kernel is therefore NOT the tiled
// bf16_gemm_mma_kernel with a cheaper instruction -- it is a second, simpler
// implementation written against the same fragment map, and the report says so.
//
// WORK IS EXACT AT T = 8: one warp computes 16 weight rows x 8 tokens with
// K/8 mmas of 16x8x8 = 1024 MACs, i.e. exactly 16*8*K. At T < 8 the kernel still
// issues K/8 mmas, so the useful fraction is T/8.
// ---------------------------------------------------------------------------
__global__ void bf16_mma1688_kernel(const __nv_bfloat16* __restrict__ w, // [N][K]
                                    const __nv_bfloat16* __restrict__ x, // [T][K]
                                    __nv_bfloat16* __restrict__ y,       // [T][N], leading dim N
                                    int N, int K, int T) {
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    // blockDim.x = 32*WARPS: warp w covers weight rows [m0 + w*16, +16).
    const int m0  = (blockIdx.x * (int)(blockDim.x >> 5) + warp) * 16;
    const int t0  = (int)blockIdx.y * 8;
    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int r0 = m0 + gid;
    const int r1 = r0 + 8;
    const int tok = t0 + gid;

    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < K; k += 8) {
        const int ka = k + 2 * lid;
        unsigned a0 = 0, a1 = 0, b0 = 0;
        if (r0 < N) { a0 = *reinterpret_cast<const unsigned*>(w + (std::size_t)r0 * K + ka); }
        if (r1 < N) { a1 = *reinterpret_cast<const unsigned*>(w + (std::size_t)r1 * K + ka); }
        if (tok < T) { b0 = *reinterpret_cast<const unsigned*>(x + (std::size_t)tok * K + ka); }
        ::ninfer::ops::mma_f16_m16n8k8(c[0], c[1], c[2], c[3], bf16x2_to_f16x2(a0),
                                       bf16x2_to_f16x2(a1), bf16x2_to_f16x2(b0));
    }

    const int tok0 = t0 + 2 * lid;
    const int tok1 = tok0 + 1;
    if (r0 < N) {
        if (tok0 < T) { y[(std::size_t)tok0 * N + r0] = __float2bfloat16_rn(c[0]); }
        if (tok1 < T) { y[(std::size_t)tok1 * N + r0] = __float2bfloat16_rn(c[1]); }
    }
    if (r1 < N) {
        if (tok0 < T) { y[(std::size_t)tok0 * N + r1] = __float2bfloat16_rn(c[2]); }
        if (tok1 < T) { y[(std::size_t)tok1 * N + r1] = __float2bfloat16_rn(c[3]); }
    }
}

inline constexpr int kBf16Fp16Mma8KC       = 256; // K superchunk; requires K % KC == 0
inline constexpr int kBf16Fp16Mma8Warps    = 4;
inline constexpr int kBf16Fp16Mma1688Warps = 4;

} // namespace ninfer::ops::detail
