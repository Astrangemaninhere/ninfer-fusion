// device_probe.cu — 能力探针的 device 半侧: 真起 kernel, 真比数值。
//
// 契约在 src/core/device_capabilities.h。这里只做三件事:
//   1) 每个能力跑一次极小的探针 (输入/参考/判据见 capability_probe_spec());
//   2) 把 launch / 同步 / 拷贝的错误如实分类 (没有 image / 启动失败 / 数值不符);
//   3) 每设备缓存一次, 缓存值就是门禁读的那份证据。
//
// 关键设计: 探针缓冲在每次 launch 前填满 0xAD。如果拷回来还是 0xAD..., 说明
// 探针体根本没执行 —— 因为本次构建的 arch guard 把它编掉了。这一条把
// "这个能力在当前 arch 下根本编不出来" 从"最容易假装解决的地方"变成一个有明确
// 状态 (ProbeStatus::NotInBuild) 的**答案**: 编不出来 -> 探针体为空 -> 运行期报
// "本 build 未编译该能力" + guard 文本, 既不会崩, 也不是瞎猜。
//
// 但要看清边界: 这套机制成立的前提是**探针体有 guard**。引擎自己今天没有 ——
// src/ops/common/mma.cuh / memory.cuh 里 arch 相关的 PTX 是无条件发射的
// (grep __CUDA_ARCH__ src/ops/common 零命中), 所以那些 arch 的构建是全有或全无:
// 编不过就是整个 build 失败 (实测见 report.md 的指令底线矩阵), 根本走不到运行期。
// 给它们加 guard + 逐能力回退是 slice 2 的事, 本片只把探针和门禁落地。

#include "core/device_capabilities.h"

#include <cuda_runtime.h>

#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

// 用引擎**自己的**指令实现做探针: 探针跑的 PTX 与线上 kernel 逐字相同。
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

// The SIMT FFMA bf16 attention family, for its SELECTOR only. The capability below is a route
// question and the answer has to be the launcher's own, so this TU asks
// ops::gqa_attention_simt_ffma_selected() BY NAME instead of re-deriving the predicate -- a second
// copy of a route test is free to answer differently from the arm that actually runs.
//
// It is a __host__ function (MEASURED: nvcc rejects calling it from a __global__ --
// "calling a __host__ function ... from a __global__ function is not allowed"; dl/capor/r2d_incl.sh),
// which is why the record is made on the host side, next to the arithmetic-only records.
//
// COST OF THIS INCLUDE, measured because it is paid by a TU that must compile for EVERY rung the
// tree targets: 3.2 s wall and 148 MB maxRSS at sm_52, same at sm_120a (nvcc 12.8, one -cubin).
// It is paid once per arch. The alternative -- a hand-written copy of the two-line predicate --
// would be free and would be a second answer to "which arm runs", which is the one thing this
// capability is about.
#include "ops/kernel/gqa_attention_simt_ffma.cuh"

#ifndef NINFER_BUILD_CUDA_ARCHS
#define NINFER_BUILD_CUDA_ARCHS "(NINFER_BUILD_CUDA_ARCHS not defined by the build)"
#endif

namespace ninfer {
namespace {

constexpr std::size_t kScratchBytes = 1024;
constexpr unsigned kProbeThreads    = 32; // MMA 只需要一个 warp
constexpr unsigned kLaneValues      = 4;  // m16n8k* 每 lane 4 个累加器

constexpr int kSentinelByte      = 0xAD;
constexpr unsigned kSentinelWord = 0xADADADADu;

// ---------------------------------------------------------------------------
// Which capability the probe kernel may assume THIS build was compiled for.
//
// A CUDA target has THREE spellings and they do not share a macro:
//   120   (no suffix)       -> neither macro
//   120a  (arch-specific)   -> __CUDA_ARCH_FEAT_SM120_ALL
//   120f  (family, 12.9+)   -> __CUDA_ARCH_FAMILY_SPECIFIC__ ONLY
// Measured on nvcc 13.3 with a per-target macro probe: sm_120f/sm_121f define
// __CUDA_ARCH_FAMILY_SPECIFIC__ and NEITHER _ALL macro; sm_100f/103f/110f likewise.
//
// So keying only on the _ALL macros answered "this build does not have that kernel"
// (ProbeStatus::NotInBuild) for a family build in which ptxas DOES assemble the form --
// a wrong answer of exactly the kind this probe exists to prevent. The family clause is
// therefore added below, ADDITIVELY: every _ALL clause keeps its exact meaning and no
// clause is widened. A plain numeric target (120) still defines NONE of these macros and
// is still reported as not built.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) ||                 \
    defined(__CUDA_ARCH_FEAT_SM110_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) ||                 \
    defined(__CUDA_ARCH_FEAT_SM121_ALL) ||                                                        \
    defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
// kind::f8f6f4 只在 'a' 与 'f' 目标上可汇编 (实测 sm_89 / sm_90a / sm_100(无a) 都编不过;
// 实测 sm_100f/103f/110f/120f/121f 全部 -cubin rc=0)。
#define NINFER_PROBE_HAS_KIND_F8F6F4 1
#endif
// The family clause here KEEPS the __CUDA_ARCH__ >= 1200 bound, because that bound carries
// the fact: measured, sm_100f/103f/110f REJECT kind::mxf4nvf4 while sm_120f/121f accept it.
// Dropping the bound would report nvfp4 as in-build for a 100f/110f build that cannot
// assemble it.
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||                \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH__ >= 1200))
// kind::mxf4nvf4.block_scale 实测只在 sm_120a / sm_121a 可汇编 (sm_100a/103a/110a 都拒)。
#define NINFER_PROBE_HAS_MXF4NVF4 1
#endif
#if defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(NINFER_PROBE_HAS_KIND_F8F6F4) ||                \
    defined(NINFER_PROBE_HAS_MXF4NVF4)
#define NINFER_PROBE_HAS_SETMAXNREG 1
#endif

// ---------------------------------------------------------------------------
// 探针 kernel (每个都对应 capability_probe_spec() 里写明的指令)
// ---------------------------------------------------------------------------
__global__ void probe_kernel_image(unsigned* out) { out[0] = 0x5A5A0001u; }

__global__ void probe_bf16_mma(float* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned ones = 0x3F803F80u; // bf16(1.0) x2
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    ops::mma_bf16(c0, c1, c2, c3, ones, ones, ones, ones, ones, ones);
    const unsigned lane          = threadIdx.x & 31u;
    out[lane * kLaneValues + 0]  = c0;
    out[lane * kLaneValues + 1]  = c1;
    out[lane * kLaneValues + 2]  = c2;
    out[lane * kLaneValues + 3]  = c3;
#endif
}

__global__ void probe_f16_mma(float* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned ones = 0x3C003C00u; // fp16(1.0) x2
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    ops::mma_f16(c0, c1, c2, c3, ones, ones, ones, ones, ones, ones);
    const unsigned lane          = threadIdx.x & 31u;
    out[lane * kLaneValues + 0]  = c0;
    out[lane * kLaneValues + 1]  = c1;
    out[lane * kLaneValues + 2]  = c2;
    out[lane * kLaneValues + 3]  = c3;
#endif
}

__global__ void probe_s8_mma(int* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned ones = 0x01010101u; // s8(1) x4
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    ops::mma_s8(c0, c1, c2, c3, ones, ones, ones, ones, ones, ones);
    const unsigned lane         = threadIdx.x & 31u;
    out[lane * kLaneValues + 0] = c0;
    out[lane * kLaneValues + 1] = c1;
    out[lane * kLaneValues + 2] = c2;
    out[lane * kLaneValues + 3] = c3;
#endif
}

__global__ void probe_fp8_mma(float* out) {
#ifdef NINFER_PROBE_HAS_KIND_F8F6F4
    const unsigned ones = 0x38383838u; // e4m3(1.0) x4
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    ops::mma_fp8_e4m3(c0, c1, c2, c3, ones, ones, ones, ones, ones, ones);
    const unsigned lane          = threadIdx.x & 31u;
    out[lane * kLaneValues + 0]  = c0;
    out[lane * kLaneValues + 1]  = c1;
    out[lane * kLaneValues + 2]  = c2;
    out[lane * kLaneValues + 3]  = c3;
#endif
}

__global__ void probe_nvfp4_mma(float* out) {
#ifdef NINFER_PROBE_HAS_MXF4NVF4
    // e2m1 码 0b0010 = 1.0, 两码一字节 -> 0x22; ue4m3(1.0) = 0x38。
    // 全 1.0 操作数 + 全 1.0 scale => 每个累加器 = k = 64 (与 nibble/scale 布局无关)。
    const unsigned codes  = 0x22222222u;
    const unsigned scales = 0x38383838u;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    ops::mma_nvfp4_e4m3(c0, c1, c2, c3, codes, codes, codes, codes, codes, codes, scales, scales);
    const unsigned lane          = threadIdx.x & 31u;
    out[lane * kLaneValues + 0]  = c0;
    out[lane * kLaneValues + 1]  = c1;
    out[lane * kLaneValues + 2]  = c2;
    out[lane * kLaneValues + 3]  = c3;
#endif
}

__global__ void probe_cp_async(float* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) float tile[4];
    ops::cp_async<16, ops::Cache::cg>(&tile[0], out + 8);
    ops::cp_commit();
    ops::cp_wait<0>();
    __syncwarp();
    out[0] = tile[0];
#endif
}

__global__ void probe_ldmatrix(unsigned* out) {
    __shared__ __align__(16) unsigned short tile[64];
    const unsigned lane = threadIdx.x & 31u;
    tile[lane]          = static_cast<unsigned short>(0x1234u + lane);
    __syncwarp();
    unsigned r0 = 0u;
    unsigned r1 = 0u;
    ops::ldmatrix_x2(r0, r1, static_cast<unsigned>(__cvta_generic_to_shared(&tile[0])));
    out[lane * 2 + 0] = r0;
    out[lane * 2 + 1] = r1;
}

__global__ void probe_setmaxnreg(unsigned* out) {
#ifdef NINFER_PROBE_HAS_SETMAXNREG
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" : : : "memory");
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" : : : "memory");
    out[0] = 0x5A5A0002u;
#endif
}

// ---------------------------------------------------------------------------
// THE SIMT FFMA ATTENTION PROBE BODY. It is a ROUTE probe and it is now a REAL LAUNCH: it calls
// the family's own row pass (ops::gqa_simt_ffma_row_pass, gqa_attention_simt_ffma.cuh:172) -- the
// single function the family's decode AND prefill kernels both funnel through -- with a geometry
// the decode kernel uses (D = 128, VecD = 4, a full key tile of kGqaSimtFfmaBc keys, all 32 lanes
// active on the warp butterfly). Nothing here is a re-implementation of the arithmetic: the FFMA
// chain, `ex2.approx.f32`, the `__fmaf_rn` accumulation, the -inf guard and the butterfly across
// lanes are the SAME instructions the online kernel executes.
//
// It is compiled UNCONDITIONALLY -- no `#if __CUDA_ARCH__` around it -- and that is the point.
// The family's floor note MEASURED that every intrinsic it uses assembles from sm_50 up, so the
// thing that can be absent is not the code, it is the DEVICE. If a future guard does leave the
// body out, the 0xAD sentinel ProbeRunner already fills turns that into ProbeStatus::NotInBuild
// rather than a silent pass.
//
// The judge. q = k = v = bf16(1.0), D = 128, all 32 keys visible, scale = 1:
//   * each lane's 4-term FFMA partial is exactly 4.0 and the 32-lane butterfly is exactly 128.0
//     (both integers, so exact in fp32) => every score is 128.0;
//   * one tile, so the online softmax gives m = 128.0, l = sum_k exp2(0) = 32.0, and
//     acc[dim] = sum_k p_k * v_k = 32.0.
// Four words are checked: {acc(dim0), acc(dim1), m / 4, l}. `m` is pre-divided by four because
// ProbeRunner compares EVERY checked word against ONE expected value (32.0) and 0.25 is a power
// of two, so the division is exact and loses nothing.
// ---------------------------------------------------------------------------
struct SimtFfmaProbeGeometry {
    static constexpr int HeadDim = 128;
};

__global__ void probe_simt_ffma_attention(float* out) {
    constexpr int D    = SimtFfmaProbeGeometry::HeadDim;
    constexpr int Bc   = ops::kGqaSimtFfmaBc;
    constexpr int VecD = D / ops::kGqaSimtFfmaWarpReduction;
    __shared__ __nv_bfloat16 k_s[Bc * D];
    __shared__ __nv_bfloat16 v_s[Bc * D];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    for (int i = static_cast<int>(threadIdx.x); i < Bc * D; i += static_cast<int>(blockDim.x)) {
        k_s[i] = __float2bfloat16(1.0f);
        v_s[i] = __float2bfloat16(1.0f);
    }
    __syncthreads();
    float qr[VecD];
#pragma unroll
    for (int i = 0; i < VecD; ++i) { qr[i] = __bfloat162float(k_s[i]); }
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[VecD];
#pragma unroll
    for (int i = 0; i < VecD; ++i) { acc[i] = 0.0f; }
    // tile_key0 = 0, qabs = Bc - 1, mask = 0 with round_masked = false, [0, Bc) visible, scale = 1.
    ops::gqa_simt_ffma_row_pass<SimtFfmaProbeGeometry, Bc>(qr, k_s, v_s, lane, 0, Bc - 1,
                                                           std::uint64_t{0}, false, 0, 0, 0, Bc,
                                                           1.0f, m, l, acc);
    const int d_lane = lane * VecD;
#pragma unroll
    for (int i = 0; i < VecD; ++i) { out[d_lane + i] = acc[i]; }
    // Lane 0's own four slots are then overwritten on purpose: that is the four-word contract the
    // runner checks. D is 128 and VecD is 4, so dims 4..127 stay covered by lanes 1..31.
    if (lane == 0) {
        out[2] = m * 0.25f;
        out[3] = l;
    }
}

// ---------------------------------------------------------------------------
// THE CODEC ROW PASS, ON THE SAME DEVICE, AGAINST THE SAME JUDGE.
//
// WHY THIS EXISTS. `probe_simt_ffma_attention` above runs the family's BF16 row pass, and its
// answer -- `SimtFfmaProbeAnswer` -- is what gates the WHOLE family, codec arm included. Until this
// body existed, a launch routed to the NVFP4 codec arm inherited a green that was never about the
// codec: the same defect family as a `-c` read as a ptxas verdict, or a DENY list read off a broken
// section splitter. It is not a hypothetical here: `gqa_simt_ffma_row_pass_packed` is a SECOND row
// pass with its OWN two loads, its own lane-chunk arithmetic (`code_at = d_lane * bits / 8`) and
// its own lane-locality requirement (`kGroupDims % VecD == 0`), so "the bf16 one matched" is not
// evidence about it.
//
// WHAT IT RUNS, AND WHY IT IS THE SAME FUNCTION THE KERNEL RUNS. It calls
// `ops::gqa_simt_ffma_row_pass_packed` -- the function the decode kernel's codec arm calls -- on
// planes it fills itself. No arithmetic is re-implemented here.
//
// THE GEOMETRY IS D = 256, NOT THE BF16 PROBE'S D = 128, AND THAT IS THE POINT. The codec route
// only exists where `Geometry::HeadDim == kGqaKvQuantHeadDim` (the nvfp4 tiled kernel is 256-only,
// see require_nvfp4_geometry_dim), so the shape that inherits this answer is D = 256, VecD = 8.
// Probing at 128 would exercise `gqa_simt_kv_dequant<Nvfp4, 4>` and leave `<Nvfp4, 8>` -- the one
// the route uses -- unmeasured, which is the same defect one level down. This body is therefore
// about D = 256/VecD = 8 and says so; the bf16 body above stays about D = 128/VecD = 4.
//
// THE INPUTS, CHOSEN SO THE ANSWER IS EXACT AND THE JUDGE IS THE BF16 ONE.
//   * E2M1 code 2 decodes to exactly 1.0f (`gqa_kv_nvfp4_e2m1_to_f32`, gqa_attention_kv_nvfp4.cuh:62)
//     and the E4M3 scale byte 0x38 (exponent 7, mantissa 0) decodes to exactly 1.0f
//     (`gqa_kv_nvfp4_e4m3_to_f32`, :122), so every codec reader reconstructs exactly 1.0 for every
//     dim of every key, by the codec's own arithmetic.
//   * q is 1.0, one full key tile of kGqaSimtFfmaBc keys is visible, scale = 1.
//     => each lane's 8-term FFMA partial is exactly 8.0, the 32-lane butterfly is exactly 256.0,
//        so m = 256.0 and l = sum_k exp2(0) = 32.0, and acc[dim] = sum_k p_k * v_k = 32.0.
//   * The four checked words are the same four the bf16 body reports: {acc(dim0), acc(dim1), m/K,
//     l}, where K = VecD = 8. `m` is divided by 8 (not by 4 as above) for one reason: the runner
//     compares every checked word against ONE expected value (32.0), m is VecD * 32, and 0.125 is a
//     power of two, so the division is exact and loses nothing. Both bodies therefore land on the
//     SAME judge -- 32.0 -- which is what makes "the codec row pass matched" mean the same thing as
//     "the bf16 row pass matched".
//
// SCOPE, NAMED SO IT IS NOT READ TOO WIDE: the codec measured here is NVFP4 and no other. NVFP4 is
// the only packed codec `gqa_simt_kv_codec_routable` admits; FP8/ISO3/I8 are decodable by the seam
// but have no written address path, so no route inherits an answer about their row passes and this
// body does not pretend to speak for them.
// ---------------------------------------------------------------------------
struct SimtFfmaCodecProbeGeometry {
    static constexpr int HeadDim = 256;
};

__global__ void probe_simt_ffma_attention_codec(float* out) {
    constexpr int D          = SimtFfmaCodecProbeGeometry::HeadDim;          // 256
    constexpr int Bc         = ops::kGqaSimtFfmaBc;                         // 32
    constexpr int VecD       = D / ops::kGqaSimtFfmaWarpReduction;          // 8
    constexpr int kCodeRow   = D / 2;                                       // NVFP4: 4 bits/element
    constexpr int kScaleRow  = D / ops::kGqaKvNvfp4Group;                    // one E4M3 byte / 16 dims
    // The plane geometry is the codec's, read from the seam's own functions rather than restated:
    // if either row width moves, this body stops compiling instead of probing the wrong layout.
    static_assert(ops::gqa_simt_kv_code_row_bytes(ops::GqaSimtKvCodec::Nvfp4, D) == kCodeRow, "");
    static_assert(ops::gqa_simt_kv_scale_row_bytes(ops::GqaSimtKvCodec::Nvfp4, D) == kScaleRow, "");
    static_assert(VecD == 8 && D / VecD == ops::kGqaSimtFfmaWarpReduction, "");

    __shared__ __align__(16) std::uint8_t k_code[Bc * kCodeRow];
    __shared__ __align__(16) std::uint8_t v_code[Bc * kCodeRow];
    __shared__ std::uint8_t k_scale[Bc * kScaleRow];
    __shared__ std::uint8_t v_scale[Bc * kScaleRow];
    // One whole byte of E2M1 code 2, and one byte of E4M3 0x38. Byte-wise, so every nibble of every
    // chunk is code 2 and every scale group is 1.0 -- there is no position a lane can land in that
    // is not covered.
    for (int i = static_cast<int>(threadIdx.x); i < Bc * kCodeRow;
         i += static_cast<int>(blockDim.x)) {
        k_code[i]  = 0x22u;
        v_code[i]  = 0x22u;
    }
    for (int i = static_cast<int>(threadIdx.x); i < Bc * kScaleRow;
         i += static_cast<int>(blockDim.x)) {
        k_scale[i] = 0x38u;
        v_scale[i] = 0x38u;
    }
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x) & 31;
    float qr[VecD];
#pragma unroll
    for (int i = 0; i < VecD; ++i) { qr[i] = 1.0f; }
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[VecD];
#pragma unroll
    for (int i = 0; i < VecD; ++i) { acc[i] = 0.0f; }

    const ops::GqaSimtPackedPlane k_plane{k_code, k_scale, kCodeRow, kScaleRow};
    const ops::GqaSimtPackedPlane v_plane{v_code, v_scale, kCodeRow, kScaleRow};
    // tile_key0 = 0, qabs = Bc - 1, mask = 0 with round_masked = false, [0, Bc) visible, scale = 1 --
    // byte for byte the arguments the bf16 body passes, so the two differ only in the row pass.
    ops::gqa_simt_ffma_row_pass_packed<SimtFfmaCodecProbeGeometry, Bc,
                                       ops::GqaSimtKvCodec::Nvfp4>(
        qr, k_plane, v_plane, lane, 0, Bc - 1, std::uint64_t{0}, false, 0, 0, 0, Bc, 1.0f, m, l,
        acc);
    const int d_lane = lane * VecD;
#pragma unroll
    for (int i = 0; i < VecD; ++i) { out[d_lane + i] = acc[i]; }
    // Lane 0's own slots carry the m and l words, exactly as above. m / 8 == 32.0 because m is
    // exactly VecD * 32 == 256.0, and 0.125 is a power of two.
    if (lane == 0) {
        out[2] = m * 0.125f;
        out[3] = l;
    }
}


// ---------------------------------------------------------------------------
// 探针执行
// ---------------------------------------------------------------------------
enum class RawKind { Unsigned, Float, Int };

[[nodiscard]] double decode(const void* raw, std::size_t index, RawKind kind) {
    switch (kind) {
    case RawKind::Unsigned: return static_cast<double>(static_cast<const unsigned*>(raw)[index]);
    case RawKind::Float: return static_cast<double>(static_cast<const float*>(raw)[index]);
    case RawKind::Int: return static_cast<double>(static_cast<const int*>(raw)[index]);
    }
    return 0.0;
}

class ProbeRunner {
public:
    explicit ProbeRunner(unsigned* device_scratch) : scratch_(device_scratch) {}

    // expected 为 NaN 时只做功能探针 (kernel 跑过 = 通过), 不做数值比较。
    template <class LaunchFn>
    ProbeRecord run(double expected, ProbeTolerance tolerance, std::size_t words, RawKind kind,
                    const char* guard, LaunchFn&& launch_fn) {
        ProbeRecord record;
        record.expected = expected;

        const cudaError_t fill = cudaMemsetAsync(scratch_, kSentinelByte, kScratchBytes, nullptr);
        if (fill != cudaSuccess) {
            record.status = ProbeStatus::LaunchFailed;
            record.detail = std::string("cudaMemsetAsync: ") + cudaGetErrorName(fill);
            return record;
        }

        const cudaError_t err = launch_fn();
        if (err != cudaSuccess) {
            // 内核在二进制里, 但设备上没有它的 image: 等于这台设备不认这个 arch 的 cubin。
            if (err == cudaErrorNoKernelImageForDevice || err == cudaErrorInvalidDeviceFunction) {
                record.status = ProbeStatus::DeviceHasNoImage;
                record.detail = std::string(cudaGetErrorName(err)) + " (" +
                                std::to_string(static_cast<int>(err)) + ")";
            } else {
                record.status = ProbeStatus::LaunchFailed;
                record.detail = std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err);
            }
            return record;
        }

        const cudaError_t sync = cudaDeviceSynchronize();
        if (sync != cudaSuccess) {
            record.status = ProbeStatus::LaunchFailed;
            record.detail = std::string("device sync: ") + cudaGetErrorName(sync) + ": " +
                            cudaGetErrorString(sync);
            return record;
        }

        unsigned raw[256];
        const cudaError_t copied = cudaMemcpy(raw, scratch_, kScratchBytes, cudaMemcpyDeviceToHost);
        if (copied != cudaSuccess) {
            record.status = ProbeStatus::LaunchFailed;
            record.detail = std::string("cudaMemcpy: ") + cudaGetErrorName(copied);
            return record;
        }
        if (raw[0] == kSentinelWord) {
            // 探针体没跑: 本次构建的 arch guard 把它编掉了。
            record.status = ProbeStatus::NotInBuild;
            record.detail = std::string("probe body compiled out (guard: ") + guard + ")";
            return record;
        }

        const bool numeric = expected == expected; // NaN => 只做功能探针
        record.observed    = numeric ? 0.0 : decode(raw, 0, kind);
        for (std::size_t i = 0; i < (numeric ? words : 1u); ++i) {
            const double value = decode(raw, i, kind);
            if (i == 0) { record.observed = value; }
            if (numeric && !probe_within(tolerance, expected, value)) {
                record.status = ProbeStatus::NumericMismatch;
                record.observed = value;
                record.detail   = "first mismatch " + std::to_string(value) + " (expected " +
                                std::to_string(expected) +
                                ", tolerance +-" +
                                std::to_string(tolerance.absolute +
                                                tolerance.relative * expected) + ")";
                return record;
            }
        }
        record.status = ProbeStatus::Supported;
        return record;
    }

private:
    unsigned* scratch_ = nullptr;
};

[[nodiscard]] ProbeRecord failed(ProbeStatus status, std::string detail) {
    ProbeRecord record;
    record.status = status;
    record.detail = std::move(detail);
    return record;
}

[[nodiscard]] CapabilityReport run_all_probes() {
    CapabilityReport report;

    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        for (std::size_t i = 0; i < kDeviceCapabilityCount; ++i) {
            report.record(static_cast<DeviceCapability>(i),
                          failed(ProbeStatus::LaunchFailed,
                                 "no CUDA device bound to this thread"));
        }
        return report;
    }

    // 探针开销: 一块 1 KiB 的 device 暂存 (用完即还, 不占常驻显存、不进 arena 账本)
    // + 每个能力一次 32 线程的 launch。整轮在毫秒以下。
    unsigned* scratch = nullptr;
    const cudaError_t allocated = cudaMalloc(&scratch, kScratchBytes);
    if (allocated != cudaSuccess) {
        for (std::size_t i = 0; i < kDeviceCapabilityCount; ++i) {
            report.record(static_cast<DeviceCapability>(i),
                          failed(ProbeStatus::LaunchFailed,
                                 std::string("probe scratch cudaMalloc failed: ") +
                                     cudaGetErrorName(allocated)));
        }
        return report;
    }

    ProbeRunner runner(scratch);

    report.record(DeviceCapability::KernelImage,
                  runner.run(std::numeric_limits<double>::quiet_NaN(), kIntegerProbe, 1,
                             RawKind::Unsigned, "always", [&] {
                                 probe_kernel_image<<<1, kProbeThreads>>>(scratch);
                                 return cudaPeekAtLastError();
                             }));

    // bf16/f16: m16n8k16 全 1.0 -> 16; s8: m16n8k32 -> 32;
    // fp8 kind::f8f6f4: m16n8k32 -> 32; nvfp4 kind::mxf4nvf4: m16n8k64 -> 64。
    report.record(DeviceCapability::Bf16Mma,
                  runner.run(16.0, kFp32ExactProbe, kProbeThreads * kLaneValues, RawKind::Float,
                             "__CUDA_ARCH__ >= 800", [&] {
                                 probe_bf16_mma<<<1, kProbeThreads>>>(
                                     reinterpret_cast<float*>(scratch));
                                 return cudaPeekAtLastError();
                             }));
    report.record(DeviceCapability::Fp16Mma,
                  runner.run(16.0, kFp32ExactProbe, kProbeThreads * kLaneValues, RawKind::Float,
                             "__CUDA_ARCH__ >= 800", [&] {
                                 probe_f16_mma<<<1, kProbeThreads>>>(
                                     reinterpret_cast<float*>(scratch));
                                 return cudaPeekAtLastError();
                             }));
    report.record(DeviceCapability::Int8Mma,
                  runner.run(32.0, kIntegerProbe, kProbeThreads * kLaneValues, RawKind::Int,
                             "__CUDA_ARCH__ >= 800", [&] {
                                 probe_s8_mma<<<1, kProbeThreads>>>(
                                     reinterpret_cast<int*>(scratch));
                                 return cudaPeekAtLastError();
                             }));
    report.record(DeviceCapability::Fp8MmaKindF8f6f4,
                  runner.run(32.0, kFp32ExactProbe, kProbeThreads * kLaneValues, RawKind::Float,
                             "__CUDA_ARCH_FEAT_SM100/103/110/120/121_ALL", [&] {
                                 probe_fp8_mma<<<1, kProbeThreads>>>(
                                     reinterpret_cast<float*>(scratch));
                                 return cudaPeekAtLastError();
                             }));
    report.record(DeviceCapability::Nvfp4MmaBlockScale,
                  runner.run(64.0, kFp32ExactProbe, kProbeThreads * kLaneValues, RawKind::Float,
                             "__CUDA_ARCH_FEAT_SM120_ALL || __CUDA_ARCH_FEAT_SM121_ALL", [&] {
                                 probe_nvfp4_mma<<<1, kProbeThreads>>>(
                                     reinterpret_cast<float*>(scratch));
                                 return cudaPeekAtLastError();
                             }));

    report.record(DeviceCapability::AsyncCopy,
                  runner.run(1.5, kFp32ExactProbe, 1, RawKind::Float, "__CUDA_ARCH__ >= 800", [&] {
                      // 图案必须写在 memset 之后 (memset 会把它抹掉), 所以放在 launch 里。
                      const float pattern[4] = {1.5f, -2.25f, 3.5f, 4.75f};
                      cudaMemcpy(reinterpret_cast<float*>(scratch) + 8, pattern, sizeof(pattern),
                                 cudaMemcpyHostToDevice);
                      probe_cp_async<<<1, kProbeThreads>>>(reinterpret_cast<float*>(scratch));
                      return cudaPeekAtLastError();
                  }));

    // 功能探针: ldmatrix 的输出是 smem 里的图案, 我们只要求这个 warp 真的执行了。
    report.record(DeviceCapability::LdMatrix,
                  runner.run(std::numeric_limits<double>::quiet_NaN(), kIntegerProbe, 1,
                             RawKind::Unsigned, "always", [&] {
                                 probe_ldmatrix<<<1, kProbeThreads>>>(scratch);
                                 return cudaPeekAtLastError();
                             }));
    report.record(DeviceCapability::SetMaxNreg,
                  runner.run(std::numeric_limits<double>::quiet_NaN(), kIntegerProbe, 1,
                             RawKind::Unsigned, "any __CUDA_ARCH_FEAT_SM*_ALL", [&] {
                                 probe_setmaxnreg<<<1, kProbeThreads>>>(scratch);
                                 return cudaPeekAtLastError();
                             }));

    // 纯运算的两个 KV codec: iso4e (src/ops/kv/iso_codec.h) 与 Rk4v4 (e8_lattice.cuh)
    // 里没有设备相关指令 (无 asm), 算术本身跑不出"不支持"。它们的**数值一致性**由各自
    // 的消费方能力覆盖 (int8 mma / nvfp4 mma); 这里如实说明理由, 不假装测过。
    // 但设备连基线 image 都没有时, 这条"通过"没有意义, 如实标成 moot。
    const bool baseline_runs = report.supported(DeviceCapability::KernelImage);
    const auto arithmetic_only = [&](std::string_view why) {
        if (baseline_runs) {
            return ProbeRecord{ProbeStatus::Supported, 0.0, 0.0,
                               std::string(why) +
                                   "; numerics not separately probed - slice 2"};
        }
        return ProbeRecord{ProbeStatus::NotProbed, 0.0, 0.0,
                           "moot: this build has no kernel image on this device"};
    };
    report.record(DeviceCapability::Iso4eKvCodec,
                  arithmetic_only("arithmetic only (no asm in src/ops/kv/iso_codec.h)"));
    report.record(DeviceCapability::E8KvLattice,
                  arithmetic_only("arithmetic only (no asm in src/ops/kernel/e8_lattice.cuh)"));

    // SIMT FFMA bf16 attention (src/ops/kernel/gqa_attention_simt_ffma.cuh).
    //
    // THE PROBE IS A ROUTE QUESTION AND NOT AN INSTRUCTION ONE. MEASURED (the family's own FLOOR
    // note): __bfloat162float, __float2bfloat16, ex2.approx.f32, __shfl_xor_sync, __frcp_rn,
    // __fmaf_rn and a 32 KiB shared-memory tile all assemble from sm_50 up, so there is no arch
    // floor here and an INSTRUCTION probe would answer "Supported" on EVERY rung.
    //
    // WHAT CHANGED. This record used to ASK the selector and report whatever it said, and the
    // selector read __CUDA_ARCH_LIST__ -- so the probe was a WITNESS of the identity decision, not
    // a replacement for it, and it could not disagree with it. The dependency now runs the other
    // way: probe_simt_ffma_attention RUNS the family's row pass on the bound device and this block
    // publishes the answer through ops::gqa_attention_simt_ffma_publish_probe_answer(), which is
    // the selector's only writer. The identity survives only as the NOT-PROBED fallback, and the
    // selector names it when it is used.
    //
    // The statuses it can produce, and why each is the honest one:
    //   * Supported   -- the row pass RAN ON THIS DEVICE and matched the host reference. That is a
    //                    measured fact about the device. It does NOT by itself say this family is
    //                    the default arm: that is the published answer's job, and a device that
    //                    also has a working bf16 tensor-core arm resolves the bf16 KV requirement
    //                    to Cap::Bf16Mma first (resolve_needs walks the bits in enum order).
    //   * NotInBuild  -- the probe body is not in this build (sentinel intact).
    //   * NumericMismatch / DeviceHasNoImage / LaunchFailed -- the runner's own classifications,
    //                    unchanged and unsoftened: a body that ran and disagreed is a MISMATCH, not
    //                    a "not in build".
    //   * NotProbed   -- moot: there is no kernel image on this device at all.
    report.record(DeviceCapability::SimtFfmaAttention, [&]() -> ProbeRecord {
        if (!baseline_runs) {
            ops::gqa_attention_simt_ffma_publish_probe_answer(ops::SimtFfmaProbeAnswer::NotProbed);
            return ProbeRecord{ProbeStatus::NotProbed, 0.0, 0.0,
                               "moot: this build has no kernel image on this device"};
        }
        ProbeRecord record = runner.run(32.0, kFp32ExactProbe, 4, RawKind::Float, "always", [&] {
            probe_simt_ffma_attention<<<1, ops::kGqaSimtFfmaWarpReduction>>>(
                reinterpret_cast<float*>(scratch));
            return cudaPeekAtLastError();
        });
        if (record.status != ProbeStatus::Supported) {
            // The runner's detail already says which of NotInBuild / NumericMismatch /
            // DeviceHasNoImage / LaunchFailed this is. Publish the measured "do not take this arm".
            ops::gqa_attention_simt_ffma_publish_probe_answer(ops::SimtFfmaProbeAnswer::Failed);
            return record;
        }
        // ---- THE CODEC ROW PASS, MEASURED HERE OR NOT AT ALL ----
        // The bf16 body above is the family's first row pass; the codec arm's is the second. The
        // answer published below gates BOTH (`gqa_attention_simt_ffma_selected()` is what the codec
        // route asks), so an answer measured on one of them is a green that is not about the other.
        // This launch closes that: it runs `gqa_simt_ffma_row_pass_packed` at the codec route's own
        // geometry (D = 256, VecD = 8) with the SAME judge, and either half failing makes the whole
        // answer Failed -- because a codec launch that inherits a Failed family answer takes the
        // tensor-core arm, which on a pre-Ada rung is the named trap rather than a wrong number.
        ProbeRecord codec = runner.run(32.0, kFp32ExactProbe, 4, RawKind::Float, "always", [&] {
            probe_simt_ffma_attention_codec<<<1, ops::kGqaSimtFfmaWarpReduction>>>(
                reinterpret_cast<float*>(scratch));
            return cudaPeekAtLastError();
        });
        if (codec.status != ProbeStatus::Supported) {
            ops::gqa_attention_simt_ffma_publish_probe_answer(ops::SimtFfmaProbeAnswer::Failed);
            codec.detail =
                std::string("the bf16 row pass (D=128) matched the host reference on this device, "
                            "but the NVFP4 packed-codec row pass (D=256) did NOT (") +
                codec.detail + "); the family serves both arms, so the answer is Failed";
            return codec;
        }
        const bool tensor_core_bf16_here = report.supported(DeviceCapability::Bf16Mma);
        if (tensor_core_bf16_here) {
            ops::gqa_attention_simt_ffma_publish_probe_answer(
                ops::SimtFfmaProbeAnswer::RanButTensorCoreArmAvailable);
            record.detail =
                "the SIMT FFMA row pass ran on this device and matched the host reference "
                "(m=128, l=32, acc=32 for q=k=v=bf16(1.0)), but this device ALSO has a working "
                "bf16 tensor-core arm, so this family is a named route here and not the default "
                "(NINFER_ATTENTION_SIMT_FFMA=1 forces it on)";
            return record;
        }
        ops::gqa_attention_simt_ffma_publish_probe_answer(ops::SimtFfmaProbeAnswer::RanOk);
        record.detail =
            "the SIMT FFMA row pass ran on this device and matched the host reference (m=128, "
            "l=32, acc=32 for q=k=v=bf16(1.0)) and this device has no working bf16 tensor-core "
            "arm, so this family IS the bf16 attention arm that will run -- the launcher names it "
            "\"bf16simt-ffma\" (ops/launcher/gqa_attention_decode_partial.cuh:228)";
        return record;
    }());

    cudaFree(scratch);
    return report;
}

} // namespace

const CapabilityReport& probe_device_capabilities() {
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) { device_id = -1; }
    static std::mutex mutex;
    static std::unordered_map<int, CapabilityReport> cache;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto found = cache.find(device_id);
    if (found != cache.end()) { return found->second; }
    return cache.emplace(device_id, run_all_probes()).first->second;
}

DeviceFacts current_device_facts() {
    DeviceFacts facts;
    facts.build_architectures = NINFER_BUILD_CUDA_ARCHS;
    int device_id            = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        facts.name = "<no CUDA device bound to this thread>";
        return facts;
    }
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device_id) == cudaSuccess) {
        facts.name  = props.name;
        facts.major = props.major;
        facts.minor = props.minor;
    }
    int driver = 0;
    if (cudaDriverGetVersion(&driver) == cudaSuccess) {
        // cudaDriverGetVersion 给的是驱动**支持的 CUDA API 版本** (13030 = 13.3),
        // 不是驱动自身的版本号 (那个在 nvidia-smi 里)。别把它印成 "driver 13.3"。
        facts.cuda_driver_api = std::to_string(driver / 1000) + "." +
                                std::to_string((driver % 1000) / 10) + " (cudaDriverGetVersion=" +
                                std::to_string(driver) + ")";
    }
    return facts;
}

} // namespace ninfer
