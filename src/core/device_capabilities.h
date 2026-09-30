#pragma once
// device_capabilities.h — "实测能力"探针的 host 半侧 (纯 C++, 不含 CUDA 头)。
//
// 这份头文件取代"按核心编号判定"。被取代的那处判定是
//   src/targets/qwen3_6/impl/runtime/layouts_impl.h:933  `if (device.sm() != 120) throw`
// 它有两个已实证的缺陷:
//   1) device.sm() (src/core/device.cu:122) = props.major * 10 + props.minor,
//      而 cudaDeviceProp 里没有 'a' 后缀, 所以运行期根本无法区分 sm_120 与 sm_120a;
//   2) 它按编号**推断**能力, 不是**实测**能力。编号相同不代表 cubin / 驱动 / 设备
//      的组合真能跑那条指令 —— 实测才是唯一可信的答案。
//
// 模型:
//   * DeviceCapability: 一次运行真正会用到的能力, 每项钉在引擎真实 kernel 上
//     (CapabilityProbeSpec::evidence 全是 file:line);
//   * CapabilityReport: 每项能力的探针结论 + 证据。探针在 device_probe.cu 里
//     真起 kernel, 每设备跑一次并缓存;
//   * requirements_for_*(): "格式 -> 能力需求"表 (KvCacheStorage / DType 两种口径)。
//     这是本文件里唯一保留"格式"概念的地方 —— 格式是 artifact 的客观属性,
//     核心编号不是;
//   * capability_failure_message(): 失败时给人看的话 —— 跑了什么探针、期望什么、
//     实测到什么、这条能力是从哪个 kernel 来的、以及怎么办。

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "core/dtype.h"
#include "ninfer/types.h"

namespace ninfer {

// ---------------------------------------------------------------------------
// 能力清单。每项都是"引擎真的会执行的指令 / 调用", 不是"某个编号应该有"。
// ---------------------------------------------------------------------------
enum class DeviceCapability : std::uint8_t {
    // 基线: 本 build 的 cubin 在这个设备上到底能不能起 kernel。
    // 引擎里任何一次 launch 都依赖它, 所以它永远在需求集里。
    KernelImage,
    // src/ops/common/mma.cuh:37  mma.sync.aligned.m16n8k16...bf16
    Bf16Mma,
    // src/ops/common/mma.cuh:46  mma.sync.aligned.m16n8k16...f16
    Fp16Mma,
    // src/ops/common/mma.cuh:54  mma.sync.aligned.m16n8k32...s8
    Int8Mma,
    // src/ops/common/mma.cuh:63  mma.sync.aligned.kind::f8f6f4.m16n8k32...e4m3
    Fp8MmaKindF8f6f4,
    // src/ops/common/mma.cuh:90  mma.sync.aligned.kind::mxf4nvf4.block_scale...
    Nvfp4MmaBlockScale,
    // src/ops/common/memory.cuh:40,44  cp.async.cg / cp.async.ca
    AsyncCopy,
    // src/ops/common/mma.cuh:9,16  ldmatrix
    LdMatrix,
    // src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:213,250  setmaxnreg
    SetMaxNreg,
    // src/ops/kv/iso_codec.h:35  ISO3 KV 编解码 (纯运算, 无特殊指令)
    Iso4eKvCodec,
    // src/ops/kernel/e8_lattice.cuh:106  E8 格 Hadamard 旋转 (纯 FMA)
    E8KvLattice,
    // src/ops/kernel/gqa_attention_simt_ffma.cuh:290 (decode) / :547 (prefill)
    //   SIMT FFMA bf16 attention: online-softmax attention on the FMA pipe only. It emits no
    //   mma, no ldmatrix, no cp.async and does no bf16 arithmetic.
    //
    //   It is a ROUTE and not an instruction, and that decides what its probe can say: MEASURED,
    //   every intrinsic this family uses assembles from sm_50 up, so there is NO __CUDA_ARCH__
    //   floor to test here -- the thing that can be absent is the SELECTION of this family as the
    //   bf16 attention arm (ops/kernel/gqa_attention_simt_ffma.cuh's own FLOOR note). A
    //   capability whose answer is "the probe body was compiled out" is therefore answered
    //   correctly by the same mechanism, with the guard being the SELECTION condition rather
    //   than an arch number -- see capability_probe_spec()'s guard column and device_probe.cu.
    SimtFfmaAttention,
    Count,
};

inline constexpr std::size_t kDeviceCapabilityCount = static_cast<std::size_t>(DeviceCapability::Count);

[[nodiscard]] constexpr std::size_t capability_index(DeviceCapability capability) noexcept {
    return static_cast<std::size_t>(capability);
}

// ---------------------------------------------------------------------------
// 探针状态。四类"不通过"刻意分开, 因为处理方式完全不同 (见失败消息)。
// ---------------------------------------------------------------------------
enum class ProbeStatus : std::uint8_t {
    NotProbed,       // 还没探 (调用方不该把这个当通过)
    Supported,       // 探针跑了, 数值与参考一致
    NotInBuild,      // 探针体被本次构建的 arch guard 编掉了 (本 build 不含该能力)
    DeviceHasNoImage, // 内核在 build 里, 但这个设备上没有它的 image -> 起不来
    LaunchFailed,    // 其它 CUDA 错误 (驱动/上下文)
    NumericMismatch, // 起来了, 但算出来的和参考不一样
};

[[nodiscard]] inline std::string probe_status_name(ProbeStatus status) {
    switch (status) {
    case ProbeStatus::NotProbed: return "未探";
    case ProbeStatus::Supported: return "通过";
    case ProbeStatus::NotInBuild: return "本 build 未编译该能力";
    case ProbeStatus::DeviceHasNoImage: return "设备上没有该内核的 image";
    case ProbeStatus::LaunchFailed: return "内核启动失败";
    case ProbeStatus::NumericMismatch: return "数值与参考不一致";
    }
    return "未知";
}

struct CapabilityProbeSpec {
    std::string_view name;        // 人话名字
    std::string_view probe;       // 探针到底跑什么 (输入规模 + 参考 + 判据)
    std::string_view evidence;    // 引擎里因此依赖它的 kernel (file:line)
    std::string_view guard;       // 探针体被编进来的条件 (就是边界所在)
};

[[nodiscard]] inline const CapabilityProbeSpec& capability_probe_spec(DeviceCapability capability) {
    static constexpr CapabilityProbeSpec kSpecs[kDeviceCapabilityCount] = {
        {"kernel image (基线)",
         "空 kernel, grid=1 block=32, 只写一个魔数再拷回; 判据: 启动与拷贝都成功",
         "src/core/device.cu:57 (DeviceContext 之后每一次 launch 都依赖它)",
         "总是编译进来"},
        {"bf16 tensor core MMA",
         "1 warp, m16n8k16, 所有操作数 = bf16(1.0); 参考: 每个累加器 = 16.0, |误差| <= 1e-6",
         "src/ops/common/mma.cuh:37",
         "__CUDA_ARCH__ >= 800"},
        {"fp16 tensor core MMA",
         "1 warp, m16n8k16, 所有操作数 = fp16(1.0); 参考: 每个累加器 = 16.0, |误差| <= 1e-6",
         "src/ops/common/mma.cuh:46",
         "__CUDA_ARCH__ >= 800"},
        {"int8 tensor core MMA",
         "1 warp, m16n8k32, 所有操作数 = s8(1); 参考: 每个累加器 = 32 (整数精确)",
         "src/ops/common/mma.cuh:54 (消费方 src/ops/kernel/gqa_attention_decode_i8.cuh)",
         "__CUDA_ARCH__ >= 800"},
        {"fp8 (e4m3) MMA, kind::f8f6f4",
         "1 warp, m16n8k32, 所有操作数 = e4m3(0x38=1.0); 参考: 每个累加器 = 32.0, |误差| <= 1e-6",
         "src/ops/common/mma.cuh:63",
         "__CUDA_ARCH_FEAT_SM100/103/110/120/121_ALL 之一 (实测: 非 'a' 目标, 含 sm_89, 都编不过)"},
        {"nvfp4 (e2m1) W4A4 MMA, kind::mxf4nvf4",
         "1 warp, m16n8k64 block_scale, 操作数 nibble 全 e2m1(1.0), scale 字节全 ue4m3(0x38=1.0); "
         "参考: 每个累加器 = 64.0 * 1.0, |误差| <= 1e-6 (与 nibble 布局无关)",
         "src/ops/common/mma.cuh:90 (消费方 src/ops/kernel/gqa_attention_decode_nvfp4.cuh)",
         "__CUDA_ARCH_FEAT_SM120_ALL || __CUDA_ARCH_FEAT_SM121_ALL (实测: sm_100a/103a/110a 都编不过)"},
        {"cp.async 16B",
         "1 block, cp.async.cg 16B 从 gmem 到 smem, commit/wait 后拷回; 判据: 内容相等",
         "src/ops/common/memory.cuh:40,44",
         "__CUDA_ARCH__ >= 800"},
        {"ldmatrix",
         "1 warp, ldmatrix.x2 从 smem 取; 判据: 启动成功且内容非魔数",
         "src/ops/common/mma.cuh:9,16",
         "总是编译进来"},
        {"setmaxnreg",
         "1 block, setmaxnreg.inc 232 + dec 40; 判据: 启动成功",
         "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:213,250",
         "任一 __CUDA_ARCH_FEAT_SM*_ALL (实测: 所有非 'a' 目标都编不过)"},
        {"ISO3 KV 编解码",
         "iso3_encode/decode 16 元素回环, 与 host 参考实现比 (src/ops/kv/iso_codec.h 同一份代码)",
         "src/ops/kv/iso_codec.h:35",
         "总是编译进来 (纯运算)"},
        {"E8 格 Hadamard 旋转",
         "hadamard_rot_8d 一组 8 维向量, 与 host 参考实现比 (正交变换, 范数守恒)",
         "src/ops/kernel/e8_lattice.cuh:106",
         "总是编译进来 (纯 FMA)"},
        {"SIMT FFMA bf16 attention (no tensor core)",
         "ROUTE probe, and it is a REAL LAUNCH: probe_simt_ffma_attention (device_probe.cu) runs "
         "this family's own row pass (gqa_attention_simt_ffma.cuh:172) on the bound device with "
         "q=k=v=bf16(1.0), D=128, a full kGqaSimtFfmaBc key tile and scale=1; judgement: the four "
         "checked words {acc0, acc1, m/4, l} all equal 32.0 exactly (m=128, l=32, acc=32), and "
         "then the answer is published to the selector through "
         "ops::gqa_attention_simt_ffma_publish_probe_answer() -- so the SELECTION is measured and "
         "not read off __CUDA_ARCH_LIST__ ('bf16simt-ffma' is still the arm token the launcher "
         "prints at ops/launcher/gqa_attention_decode_partial.cuh:228)",
         "src/ops/kernel/gqa_attention_simt_ffma.cuh:172 (the row pass both bodies call) / :290 "
         "(decode partial kernel) / :547 (prefill kernel); consumed by "
         "ops::gqa_attention_simt_ffma_selected() at src/ops/launcher/"
         "gqa_attention_decode_partial.cuh:212 and src/ops/launcher/gqa_attention_prefill.cu:259",
         "总是编译进来 (the body has no arch guard on purpose: it must be able to say "
         "'this DEVICE cannot run it', not 'this BUILD does not contain it'); the selector's "
         "UNPROBED fallback is NINFER_ATTENTION_SIMT_FFMA_DEFAULT_ON (__CUDA_ARCH_LIST__ < 700), "
         "and an explicit NINFER_ATTENTION_SIMT_FFMA=1/=0 overrides the probe either way"},
    };
    return kSpecs[capability_index(capability)];
}

// ---------------------------------------------------------------------------
// 探针结论
// ---------------------------------------------------------------------------
struct ProbeTolerance {
    double absolute = 0.0;
    double relative = 0.0;
};

// 全 1.0 操作数累加到 fp32 是精确的 (整数 <= 2^24), 所以 fp 判据收紧到 1e-6。
inline constexpr ProbeTolerance kFp32ExactProbe{1e-6, 0.0};
inline constexpr ProbeTolerance kIntegerProbe{0.0, 0.0};

[[nodiscard]] constexpr bool probe_within(ProbeTolerance tolerance, double expected,
                                          double observed) {
    const double delta = expected > observed ? expected - observed : observed - expected;
    const double slack = tolerance.absolute + tolerance.relative * (expected < 0.0 ? -expected : expected);
    return delta <= slack;
}

struct ProbeRecord {
    ProbeStatus status = ProbeStatus::NotProbed;
    double expected    = 0.0;
    double observed    = 0.0;
    std::string detail;   // 现场字符串: CUDA 错误名/guard 文本/差值
};

// ---------------------------------------------------------------------------
// 需求集 (按能力位)
// ---------------------------------------------------------------------------
using CapabilitySet = std::uint32_t;

[[nodiscard]] constexpr CapabilitySet capability_bit(DeviceCapability capability) noexcept {
    return CapabilitySet{1} << capability_index(capability);
}

[[nodiscard]] constexpr bool set_contains(CapabilitySet set, DeviceCapability capability) noexcept {
    return (set & capability_bit(capability)) != 0;
}

inline constexpr CapabilitySet kNoCapabilities      = 0;
inline constexpr CapabilitySet kKernelImageBaseline = capability_bit(DeviceCapability::KernelImage);

// ---------------------------------------------------------------------------
// A requirement: a conjunction, plus an OPTIONAL disjunction.
// ---------------------------------------------------------------------------
// WHY THIS TYPE EXISTS. CapabilitySet is an AND-mask -- CapabilityReport::all_supported() asks
// every set bit to be Supported, which is right for a stack of parts that must ALL be there. It
// was WRONG for one question the table below has to answer. A bf16 KV plane is not consumed by
// one kernel: it is consumed by the bf16 tensor-core attention arm, which needs Cap::Bf16Mma, OR
// by the SIMT FFMA bf16 attention family, which needs no tensor core at all
// (ops/kernel/gqa_attention_simt_ffma.cuh). Spelling that as
// `bit(Bf16Mma) | bit(SimtFfmaAttention)` inside an AND-mask would have demanded BOTH -- the
// exact opposite of the truth, and the reason an sm_52 build refused a bf16 KV tier before any
// kernel was chosen. So a requirement is now:
//
//   * `all_of`  -- the conjunction. Semantics UNCHANGED, and the membership is unchanged for
//                  every storage that has no alternate route (their `any_of` is 0).
//   * `any_of`  -- the disjunction. Satisfied when AT LEAST ONE of its bits is Supported. `0`
//                  means "there is no alternate route" and the field has no meaning at all.
//
// `any_of` is NOT a licence to weaken `all_of`: BOTH halves are checked, and a requirement with
// `any_of == 0` is exactly as strict as it was before this struct existed. A caller cannot
// silently read only half of a requirement, because CapabilitySet and CapabilityNeeds do not
// convert into one another.
struct CapabilityNeeds {
    CapabilitySet all_of = kNoCapabilities;
    CapabilitySet any_of = kNoCapabilities; // 0 == no alternate route

    friend constexpr bool operator==(CapabilityNeeds, CapabilityNeeds) = default;
};

// 某一位是否出现在这条需求里 (合取或析取任一侧), 供 "这条需求是谁提的" 一类报告使用。
[[nodiscard]] constexpr bool needs_contains(CapabilityNeeds needs,
                                            DeviceCapability capability) noexcept {
    return set_contains(needs.all_of, capability) || set_contains(needs.any_of, capability);
}

[[nodiscard]] constexpr bool needs_has_disjunction(CapabilityNeeds needs) noexcept {
    return needs.any_of != kNoCapabilities;
}

// 格式 -> 能力需求。KV 存储的"含义"照 include/ninfer/types.h:29-38 与
// src/targets/qwen3_6/impl/state/decoder_state.cpp:63 的落地读法。
// bf16 槽位是这张表里唯一一条带析取的需求, 理由写在 CapabilityNeeds 上面: tensor-core 那条
// 路要 Cap::Bf16Mma, SIMT FFMA 那条路一条 tensor core 指令都不要。两条都是真的路线, 而一张
// 只能写合取的表没法说出"或"。
[[nodiscard]] inline CapabilityNeeds requirements_for_kv_storage(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        // THE DISJUNCTION. Baseline is required in all cases; then ONE of the two bf16 attention
        // arms must be supported. On a build whose only target is pre-Volta, the tensor-core arm
        // is not in the build at all (its probe body is behind __CUDA_ARCH__ >= 800) while the
        // SIMT FFMA family IS what that build selects, so the bit that carries this tier is
        // SimtFfmaAttention. On every other build Bf16Mma carries it and this arm is unchanged.
        return CapabilityNeeds{kKernelImageBaseline,
                               capability_bit(DeviceCapability::Bf16Mma) |
                                   capability_bit(DeviceCapability::SimtFfmaAttention)};
    case KvCacheStorage::Int8Group64:
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::Int8Mma)};
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Fp8Group16:
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::Fp8MmaKindF8f6f4) |
               capability_bit(DeviceCapability::Int8Mma)};
    case KvCacheStorage::Nvfp4Group16:
        // K 面是 E2M1 的 mxf4nvf4, V 面默认 ISO3 (types.h:46-49)。
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::Nvfp4MmaBlockScale) |
               capability_bit(DeviceCapability::Iso4eKvCodec)};
    case KvCacheStorage::Iso3Group16:
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::Iso4eKvCodec)};
    case KvCacheStorage::E8Group64:
        // types.h:36-37: E8 码由 int8 attention kernel 消费。
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::E8KvLattice) |
               capability_bit(DeviceCapability::Int8Mma)};
    case KvCacheStorage::E8K3Group64:
    case KvCacheStorage::E8K2Group64:
        // Same codec family as E8Group64 and the same consumption route (the int8
        // attention kernels read the decoded plate), so the capability demand is the
        // same. This arm is unreachable today -- kv_dtype_for_storage refuses both
        // storages before a plan can carry one -- and is named so that -Wswitch keeps
        // pointing here if that ever changes.
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::E8KvLattice) |
               capability_bit(DeviceCapability::Int8Mma)};
    case KvCacheStorage::Dropped:
        // NAMED, and the answer is the one the trailing `return` used to give: a
        // discarded layer (NINFER_KV_DROP_LAYERS) owns no KV planes, so it adds no
        // codec capability to the preflight.  That is a DECISION, not a fallthrough
        // that happens to work.  Naming it is what keeps -Wswitch pointed at this
        // switch when the enumeration grows. The sentence that used to end here ("the trailing
        // return stays for a storage code that no enumerator names") is now FALSE and is
        // rewritten rather than left standing: the trailing return became a refusal, and the five
        // tiers INTEGRATE4 appended are named further down this same switch, so EVERY code this
        // enum names has an arm of its own and the refusal is reachable only for a code no
        // enumerator names (KvCacheStorage is a uint8_t, so a cast or an untrusted input can
        // produce one). That is what makes this arm's answer unique to this layer: it is the only
        // storage that owns no KV planes at all.
        return CapabilityNeeds{kKernelImageBaseline};
    // ---------------------------------------------------------------------------------------
    // INTEGRATE4's five, NAMED here as well. They are named in kv_storage_name() below with
    // tokens pinned by static_assert (product/kv_storage_dtype.h:276-302) and they are refused
    // BY NAME at product/kv_storage_dtype.h (SITE 3) and at target_kv_cache_profile() (SITE 4),
    // so naming them here does not admit them anywhere -- it removes a POSITIVE CLAIM. Falling
    // past this switch is not "no answer": the trailing `return` makes the claim "the
    // kernel-image baseline covers this tier" about a tier the enum names and whose planes are
    // declared in core/paged_kv_storage.h:92-123. That silent under-report is named in
    // dl/warnsurface/E8_SWITCH_SITES.md:31 ("a silent under-report: no codec capability is
    // demanded for the tier") and in dl/e8mixwire/REPORT.md S1, which left it OPEN.
    //
    // The demand per tier is DERIVED from the tree, not chosen; the two derivations are stated
    // in the arms. None of these five is reachable from --kv-dtype or --kv-layer-storage
    // (product/kv_options.h:45 parse_kv_storage returns nullopt for all five), so no operator
    // spelling becomes newly admitted by this landing.
    case KvCacheStorage::Fp8KeyNvfp4Value:
        // A K/V CODEC PAIR: its resolved planes (core/paged_kv_storage.h:92-99) are
        // {FP8_E4M3FN, 256, FP16, 1} for K -- byte-for-byte Fp8E4M3Row256's arm above -- and
        // {U8, 128, U8, 16} for V -- byte-for-byte Nvfp4Group16's. include/ninfer/types.h:55-60
        // says the same in the enumerator's own words. A layer carrying the pair must decode BOTH
        // planes, so the demand is the CONJUNCTION of the two rows above, not either one alone:
        // the fp8 row alone would miss the fp4 MMA the V plane needs, and the nvfp4 row alone
        // would miss the fp8 MMA the K plane needs. The conjunction is strictly MORE demanding
        // than either neighbour, so this arm can only narrow.
        //
        // The V-side codec choice (KvVCodec Iso3|E2M1, include/ninfer/types.h:106-124) is
        // deliberately not a capability: its own note says the choice is "a pure decoder switch:
        // no extra plane, no extra allocation, and no new kernel instantiation".
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::Fp8MmaKindF8f6f4) |
                               capability_bit(DeviceCapability::Int8Mma) |
                               capability_bit(DeviceCapability::Nvfp4MmaBlockScale) |
                               capability_bit(DeviceCapability::Iso4eKvCodec)};
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
    case KvCacheStorage::RK4V4E8:
    case KvCacheStorage::RK2V4E8:
        // THE FOUR INT8-FAMILY STORAGES. The rule that decides their demand is the tree's own
        // classifier, not this line's judgement: core/paged_kv_storage.h:139-150
        // kv_storage_is_int8_family() returns true for exactly {Int8Group64,
        // RotatedInt8KeyInt4ValueGroup64, RotatedInt4KeyInt4ValueGroup64, RK4V4E8, RK2V4E8} and
        // false for every other enumerator, and their declared planes
        // (core/paged_kv_storage.h:104-123) are I8/U8 data with per-64 FP16 group scales -- the
        // int8 family's shape. So the demand is exactly Int8Group64's row above: this is the same
        // rule src/ops/kv/e8_width_contract_test.cpp:219-220 already pins one family over
        // ("rk3v4/rk2v4 demand exactly the rk4v4 codec's capabilities").
        //
        // WHAT IS DELIBERATELY *NOT* DEMANDED, and this is the load-bearing sentence:
        // `E8KvLattice` is NOT demanded for RK4V4E8/RK2V4E8, even though
        // core/paged_kv_storage.h:158-165 sets `.e8_lattice = true` on RK4V4E8. dl/e8dev measured
        // that these two names are "two characters from this tree's rk4v4/rk3v4/rk2v4 TIERS and
        // describe a different codec on the same geometry (the 240-root root-cylinder specimen)",
        // and that mapping them onto the lattice is "a WRONG CODEC UNDER A RIGHT NAME"
        // (dl/e8dev/patch/06-SWITCH-ARMS.md section 3, cited by product/kv_storage_dtype.h:153-156,
        // which refuses that aliasing one table over). Demanding the lattice bit here would BE that
        // aliasing, one table up. And the rotation codec has no capability bit in this tree --
        // DeviceCapability::E8KvLattice is core/e8_lattice.cuh's E8 Hadamard rotation, a different
        // codec from the fork's -- so nothing above the family floor can be truthfully demanded.
        // The open question this leaves is named in this landing's REPORT.md; it is not silently
        // settled here.
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::Int8Mma)};
    }
    // A STORAGE CODE NO ENUMERATOR NAMES, and it is reachable -- the comment above records it,
    // and KvCacheStorage is a uint8_t (include/ninfer/types.h:29) so a cast or an untrusted input
    // can produce one. This used to `return CapabilityNeeds{kKernelImageBaseline};`, which makes
    // a silent POSITIVE claim ("the kernel-image baseline covers this tier") about a tier nobody
    // recognises. That is the defect shape the owner rates worse than a refusal, so it refuses
    // instead. The code is printed so the caller can find where it came from.
    //
    // THE CALLERS CAN PROPAGATE IT, MEASURED (a caller census, which DELTA.md section 4 row 4
    // recorded as not determined): src/targets/qwen3_6/impl/runtime/layouts_impl.h:1109 (inside
    // required_capabilities(), which has no noexcept -- that file contains 0 occurrences of the
    // word), src/core/device_capabilities.h:553, and four test TUs. Nothing in the chain is
    // noexcept, so this is an std::invalid_argument and not a terminate.
    throw std::invalid_argument(
        "kv storage code " + std::to_string(static_cast<unsigned>(storage)) +
        " is not a KvCacheStorage this build knows, so this table cannot say which capabilities it "
        "needs; refusing to assume the kernel-image baseline covers it. The tiers this build knows "
        "are enumerated in include/ninfer/types.h and their canonical tokens are in "
        "product/kv_storage_dtype.h.");
}

// DType 口径 (layouts_impl.h 的 layer_overrides 表用的就是 DType)。
// DType 口径没有析取: 这一侧问的是"这种权重元素的算子在不在这份 build 里", 每一条
// 都只有一个答案, 所以 any_of 一律为 0 (语义与这张表引入 CapabilityNeeds 之前完全一致)。
[[nodiscard]] inline CapabilityNeeds requirements_for_dtype(DType dtype) {
    switch (dtype) {
    case DType::BF16:
    case DType::FP16:
    case DType::FP32:
        return CapabilityNeeds{kKernelImageBaseline};
    case DType::I8:
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::Int8Mma)};
    case DType::FP8_E4M3FN:
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::Fp8MmaKindF8f6f4) |
               capability_bit(DeviceCapability::Int8Mma)};
    case DType::NVFP4:
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::Nvfp4MmaBlockScale) |
               capability_bit(DeviceCapability::Iso4eKvCodec)};
    case DType::ISO3:
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::Iso4eKvCodec)};
    case DType::E8Kv:
    case DType::E8K3Kv:
    case DType::E8K2Kv:
        return CapabilityNeeds{kKernelImageBaseline | capability_bit(DeviceCapability::E8KvLattice) |
               capability_bit(DeviceCapability::Int8Mma)};
    default:
        return CapabilityNeeds{kKernelImageBaseline};
    }
}

[[nodiscard]] inline std::string_view kv_storage_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8-g64";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8-e4m3-row256";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4-g16";
    case KvCacheStorage::Fp8Group16: return "fp8-g16";
    case KvCacheStorage::Iso3Group16: return "iso4e-g16";
    case KvCacheStorage::E8Group64: return "rk4v4-g64";
    // RENAMED to match the canonical table. These two read "e8k3-g64"/"e8k2-g64" while the
    // arm three lines above already returned "rk4v4-g64", so this function contradicted itself
    // three lines apart. product/kv_storage_dtype.h pins "rk4v4-g64"/"rk3v4-g64"/"rk2v4-g64"
    // under static_assert (:206-209), tests/test_kv_operator_name.cpp:94-95 pins the new
    // spellings, and kv_bit_budget.h:567-569 normalises "e8k3" -> "rk3v4": the direction is
    // determined by the tree, not chosen here.
    case KvCacheStorage::E8K3Group64: return "rk3v4-g64";
    case KvCacheStorage::E8K2Group64: return "rk2v4-g64";
    // INTEGRATE4's five. NAMED because the alternative is the trailing `return "?"` below:
    // this switch has no `default:`, so each of these five fell past every arm and a diagnostic
    // that asked for the tier's name printed "?" -- the tier an operator cannot identify in a
    // report. Tokens are the ones product/kv_storage_dtype.h now pins.
    case KvCacheStorage::Fp8KeyNvfp4Value: return "fp8k-nvfp4v-g64";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64: return "rot-i8k-i4v-g64";
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64: return "rot-i4k-i4v-g64";
    case KvCacheStorage::RK4V4E8: return "rk4v4e8";
    case KvCacheStorage::RK2V4E8: return "rk2v4e8";
    case KvCacheStorage::Dropped: return "dropped";
    }
    return "?";
}

[[nodiscard]] inline std::string_view dtype_name(DType dtype) {
    switch (dtype) {
    case DType::BF16: return "bf16";
    case DType::FP32: return "fp32";
    case DType::I32: return "i32";
    case DType::U8: return "u8";
    case DType::I64: return "i64";
    case DType::I8: return "i8";
    case DType::FP16: return "fp16";
    case DType::FP8_E4M3FN: return "fp8-e4m3";
    case DType::NVFP4: return "nvfp4";
    case DType::ISO3: return "iso4e";
    case DType::E8Kv: return "rk4v4";   // LEGACY (dl/e8names F1194): pair(B4,B4); its
                                         // plane-level spelling is `e8-b4/e8-b4`
    // The e8 family's narrower K planes. Unreachable today (product::kv_dtype_for_storage
    // refuses both storages), named so this switch stays complete over DType.
    // LEGACY (dl/e8names F1194). `rk3v4` / `rk2v4` are the DEPLOYED tokens and are KEPT,
    // byte for byte: this function's spellings are asserted equal by TEXT to the ones the
    // stage's own refusal quotes, and a refusal that re-spelled its tier would name a tier
    // the operator did not ask for. The plane-level primary spellings of these same two
    // pairs are `E8KvB3B4` / `E8KvB2B4` (core/dtype.h) and `e8-b3/e8-b4` / `e8-b2/e8-b4`
    // (product/kv_e8_width.h).
    case DType::E8K3Kv: return "rk3v4";
    case DType::E8K2Kv: return "rk2v4";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// 报告
// ---------------------------------------------------------------------------
class CapabilityReport {
public:
    void record(DeviceCapability capability, ProbeRecord record) {
        records_[capability_index(capability)] = std::move(record);
    }

    [[nodiscard]] const ProbeRecord& record(DeviceCapability capability) const {
        return records_[capability_index(capability)];
    }

    [[nodiscard]] ProbeStatus status(DeviceCapability capability) const {
        return record(capability).status;
    }

    [[nodiscard]] bool supported(DeviceCapability capability) const {
        return status(capability) == ProbeStatus::Supported;
    }

    [[nodiscard]] bool all_supported(CapabilitySet needed) const {
        return !first_unsupported(needed).has_value();
    }

    [[nodiscard]] std::optional<DeviceCapability> first_unsupported(CapabilitySet needed) const {
        for (std::size_t i = 0; i < kDeviceCapabilityCount; ++i) {
            const auto capability = static_cast<DeviceCapability>(i);
            if (set_contains(needed, capability) && !supported(capability)) { return capability; }
        }
        return std::nullopt;
    }

    // 每行: 能力名 | 状态 | 现场 —— 直接进日志, 也是报错消息的来源。
    [[nodiscard]] std::string table() const {
        std::string out;
        for (std::size_t i = 0; i < kDeviceCapabilityCount; ++i) {
            const auto capability = static_cast<DeviceCapability>(i);
            const ProbeRecord& entry = records_[i];
            out += "  ";
            out += capability_probe_spec(capability).name;
            out += ": ";
            out += probe_status_name(entry.status);
            if (!entry.detail.empty()) {
                out += " (";
                out += entry.detail;
                out += ")";
            }
            out += "\n";
        }
        return out;
    }

    [[nodiscard]] bool probed_anything() const {
        for (const ProbeRecord& entry : records_) {
            if (entry.status != ProbeStatus::NotProbed) { return true; }
        }
        return false;
    }

private:
    std::array<ProbeRecord, kDeviceCapabilityCount> records_{};
};


struct DeviceFacts {
    std::string name;                  // cudaDeviceProp::name
    int major = 0;
    int minor = 0;
    std::string build_architectures;   // 本次构建的 CMAKE_CUDA_ARCHITECTURES
    std::string cuda_driver_api;       // 驱动支持的 CUDA API 版本 (不是驱动自身版本)
};

[[nodiscard]] inline std::string device_facts_line(const DeviceFacts& facts) {
    std::string out = facts.name + " (compute capability " + std::to_string(facts.major) + "." +
                      std::to_string(facts.minor) + ")";
    if (!facts.cuda_driver_api.empty()) { out += ", CUDA driver API " + facts.cuda_driver_api; }
    return out;
}

// ---------------------------------------------------------------------------
// A requirement -> the answer: who satisfies it, or that nobody does
// ---------------------------------------------------------------------------
// WHY NOT all_supported(). With a disjunction in play a requirement has TWO questions -- "is it
// met" and "WHAT meets it" -- and the second one is the one that names the kernel that will run.
// `served_by` is read from the REPORT (report.supported), never derived from this build's macros:
// a name derived from a macro is free to point at an arm that will not execute, which is the
// silent-wrong-answer shape this whole area exists to prevent.
struct NeedsResolution {
    bool satisfied = false;
    std::optional<DeviceCapability> missing_required; // all_of 里第一个不满足的位
    std::optional<DeviceCapability> served_by;        // any_of 里第一个 Supported 的位
};

[[nodiscard]] inline NeedsResolution resolve_needs(CapabilityNeeds needs,
                                                  const CapabilityReport& report) {
    NeedsResolution out;
    out.missing_required = report.first_unsupported(needs.all_of);
    out.satisfied        = !out.missing_required.has_value();
    if (needs.any_of != kNoCapabilities) {
        for (std::size_t i = 0; i < kDeviceCapabilityCount; ++i) {
            const auto capability = static_cast<DeviceCapability>(i);
            if (set_contains(needs.any_of, capability) && report.supported(capability)) {
                out.served_by = capability;
                break;
            }
        }
        // 一条可选路线都没过 -> 这条需求不满足, 即使合取部分全过。
        if (!out.served_by.has_value()) { out.satisfied = false; }
    }
    return out;
}

// 这条能力兑现成的 KERNEL. The token is the launcher's OWN route name -- "bf16simt-ffma" is the
// string src/ops/launcher/gqa_attention_decode_partial.cuh:228 hands to gqa_splitdbg_tile -- so a
// log line that prints it names the same thing a debug log names, and tests/test_device_capabilities
// .cpp checks the token occurs VERBATIM at that call site rather than trusting this comment.
//
// Capacities with no such token return an empty view on purpose: the caller falls back to the
// probe spec's file:line evidence. Inventing a kernel name for a capability that has none would be
// the same lie in a smaller font.
[[nodiscard]] inline std::string_view capability_kernel_arm(DeviceCapability capability) noexcept {
    switch (capability) {
    case DeviceCapability::SimtFfmaAttention: return "bf16simt-ffma";
    case DeviceCapability::Bf16Mma:
        return "bf16 tensor-core attention (mma_bf16, src/ops/common/mma.cuh:37)";
    default: return {};
    }
}

// 某一条可选路线的人话描述: 能力名 + (有的话) 它兑现成的 kernel 名。
[[nodiscard]] inline std::string capability_arm_text(DeviceCapability capability) {
    std::string text;
    const std::string_view name = capability_probe_spec(capability).name;
    text.append(name.data(), name.size());
    const std::string_view arm = capability_kernel_arm(capability);
    if (!arm.empty()) {
        text += " -> kernel \"";
        text.append(arm.data(), arm.size());
        text += "\"";
    } else {
        text += " -> ";
        const std::string_view evidence = capability_probe_spec(capability).evidence;
        text.append(evidence.data(), evidence.size());
    }
    return text;
}

// 一条需求现在的状态, 一行, 可以直接进日志。
//
// 满足时它点名**这次运行真的会被使用的那个 kernel**; 不满足时它是一条 NAMED REFUSAL: 逐个点名
// 这条需求的每一条可选路线, 每条都带自己的 ProbeStatus 和现场 detail ("为什么不是它"), 而不是
// 一句"不支持这个格式"。一条没有可选路线的需求也走这里, 于是它和被替换掉的写法一样严。
[[nodiscard]] inline std::string kv_storage_route_line(KvCacheStorage storage,
                                                       const CapabilityReport& report) {
    const CapabilityNeeds needs = requirements_for_kv_storage(storage);
    const NeedsResolution res   = resolve_needs(needs, report);

    std::string out = "kv storage ";
    out += kv_storage_name(storage);
    if (res.satisfied) {
        out += ": accepted";
        if (res.served_by.has_value()) {
            out += " -- served by kernel \"";
            const std::string_view arm = capability_kernel_arm(*res.served_by);
            if (!arm.empty()) {
                out.append(arm.data(), arm.size());
            } else {
                out += "(unnamed capability)";
            }
            out += "\" [";
            const std::string_view name = capability_probe_spec(*res.served_by).name;
            out.append(name.data(), name.size());
            out += "]";
            if (needs_has_disjunction(needs)) {
                out += "; the other arm of this requirement was not the one that resolved, and "
                       "the choice is read from the probe report, not from this build's macros";
            }
        } else {
            out += " -- every required capability is supported (this requirement has no alternate "
                   "route, so there is no arm to choose)";
        }
        return out;
    }

    out += ": REFUSED -- no arm of this requirement is available:";
    for (std::size_t i = 0; i < kDeviceCapabilityCount; ++i) {
        const auto capability = static_cast<DeviceCapability>(i);
        if (!needs_contains(needs, capability)) { continue; }
        const ProbeRecord& entry = report.record(capability);
        out += "\n    - ";
        out += capability_arm_text(capability);
        out += " : ";
        out += probe_status_name(entry.status);
        if (!entry.detail.empty()) {
            out += " (";
            out += entry.detail;
            out += ")";
        }
    }
    return out;
}

// 析取需求不满足时抛给操作者的话: 结论 -> 是谁要的 -> 每条可选路线各为什么不在 -> 设备/构建 ->
// 出路。它和 capability_failure_message() 并列而不是替代它: 单能力的需求仍然用那一条, 因为它们
// 的回答形状不同 (那条说"跑了什么探针/期望什么/实测到什么"), 这一条要说的是"两条路都没了, 分别
// 因为什么"。
[[nodiscard]] inline std::string capability_needs_failure_message(CapabilityNeeds needs,
                                                                 const CapabilityReport& report,
                                                                 const DeviceFacts& facts,
                                                                 std::string_view required_by) {
    std::string out = "capability probe failed: none of the available arms of this requirement is "
                      "supported";
    out += "\n  required by : ";
    out.append(required_by.data(), required_by.size());
    out += "\n  arms        :";
    for (std::size_t i = 0; i < kDeviceCapabilityCount; ++i) {
        const auto capability = static_cast<DeviceCapability>(i);
        if (!needs_contains(needs, capability)) { continue; }
        const ProbeRecord& entry = report.record(capability);
        out += "\n    * ";
        out += capability_arm_text(capability);
        out += "\n        status   : ";
        out += probe_status_name(entry.status);
        if (!entry.detail.empty()) {
            out += "\n        observed : ";
            out += entry.detail;
        }
        out += "\n        probe    : ";
        const std::string_view probe = capability_probe_spec(capability).probe;
        out.append(probe.data(), probe.size());
        out += "\n        guard    : ";
        const std::string_view guard = capability_probe_spec(capability).guard;
        out.append(guard.data(), guard.size());
        out += "\n        kernel   : ";
        const std::string_view evidence = capability_probe_spec(capability).evidence;
        out.append(evidence.data(), evidence.size());
    }
    out += "\n  device      : ";
    out += device_facts_line(facts);
    out += "\n  this build  : CMAKE_CUDA_ARCHITECTURES=";
    out += facts.build_architectures;
    out += "\n  fix         : the two arms are not interchangeable -- one of them has to exist on "
           "this device in this build before a bf16 KV tier can be served at all. Build the arch "
           "that carries one of them, or choose a KV format whose own requirement is met.";
    return out;
}

// 失败怎么写: 先给结论, 再给"跑了什么探针 / 期望 / 实测", 最后给两条出路。
[[nodiscard]] inline std::string capability_failure_message(DeviceCapability missing,
                                                            const CapabilityReport& report,
                                                            const DeviceFacts& facts,
                                                            std::string_view required_by) {
    const CapabilityProbeSpec& spec = capability_probe_spec(missing);
    const ProbeRecord& entry        = report.record(missing);

    std::string out = "capability probe failed: ";
    out += spec.name;
    out += "\n  required by : ";
    out.append(required_by.data(), required_by.size());
    out += "\n  probe       : ";
    out.append(spec.probe.data(), spec.probe.size());
    out += "\n  expected    : ";
    if (entry.expected == entry.expected) {
        out += std::to_string(entry.expected);
    } else {
        out += "n/a (功能探针: 只要求这个内核真的执行一次)";
    }
    if (!entry.detail.empty()) {
        out += "\n  observed    : ";
        out += probe_status_name(entry.status);
        out += " - ";
        out += entry.detail;
    } else {
        out += "\n  observed    : ";
        out += probe_status_name(entry.status);
    }
    out += "\n  kernel      : ";
    out.append(spec.evidence.data(), spec.evidence.size());
    out += "\n  device      : ";
    out += device_facts_line(facts);
    out += "\n  this build  : CMAKE_CUDA_ARCHITECTURES=" + facts.build_architectures;
    switch (entry.status) {
    case ProbeStatus::DeviceHasNoImage:
        out += "\n  fix         : this binary carries no image for this device - rebuild for "
               "its arch (-DCMAKE_CUDA_ARCHITECTURES=";
        out += std::to_string(facts.major);
        out += std::to_string(facts.minor);
        out += "a), or choose a format whose probe passes";
        break;
    case ProbeStatus::NotInBuild:
        out += "\n  fix         : the probe body for this capability was compiled out of this "
               "build (guard: ";
        out.append(spec.guard.data(), spec.guard.size());
        out += ") - build the arch that has it, or choose a format whose probe passes";
        break;
    case ProbeStatus::NumericMismatch:
        out += "\n  fix         : the kernel ran but disagreed with the reference - do not "
               "trust this format on this device; use a format whose probe passes";
        break;
    case ProbeStatus::LaunchFailed:
        out += "\n  fix         : the kernel did not launch - see the CUDA error above";
        break;
    case ProbeStatus::NotProbed:
        out += "\n  fix         : the probe never ran (internal error: probe_device_capabilities "
               "was not called or failed early)";
        break;
    case ProbeStatus::Supported:
        break;
    }
    return out;
}

// ---------------------------------------------------------------------------
// device 半侧 (实现在 core/device_probe.cu)
// ---------------------------------------------------------------------------
// 每个设备跑一次全部探针并缓存; 返回的引用在进程生命周期内有效。
// 必须在已经绑定 CUDA 设备的线程上调用 (DeviceContext 的线程绑定即可)。
[[nodiscard]] const CapabilityReport& probe_device_capabilities();

// 报错消息里用的现场 (cudaGetDeviceProperties + cudaDriverGetVersion + 构建 arch 列表)。
[[nodiscard]] DeviceFacts current_device_facts();

} // namespace ninfer
