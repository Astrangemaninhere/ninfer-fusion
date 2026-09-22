// test_device_capabilities — host 侧单测: 格式->能力需求表, 探针结论聚合, 失败消息。
//
// 纯 host: 只用 core/device_capabilities.h (它不含 CUDA 头), 因此可以用普通
// g++ 直接编译运行, 不需要 GPU, 也不需要跑 cmake:
//   g++ -std=c++20 -Iinclude -Isrc -DNINFER_SOURCE_DIR='"<repo>"' test_device_capabilities.cpp
//
// 这里验的三件事, 对应"按编号判定 -> 按实测判定"这段改动的三个承诺:
//   1) 需求表按**格式**给的 (nvfp4 要 fp4 MMA, rk4v4 要 int8 MMA ...), 不是按编号;
//   2) "没探到" 不算通过 (NotProbed != Supported), 所以门禁不会因为表没填就放行;
//   3) 失败消息真的可行动 (探针/期望/实测/kernel file:line/设备/构建/出路),
//      且每条能力的 evidence 都必须指向一个**真实存在**的 file:line。
#include "core/device_capabilities.h"

#include <cstdio>
#include <fstream>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (condition) { return; }
    std::printf("FAIL: %s\n", what);
    ++failures;
}

[[nodiscard]] bool has(const std::string& haystack, std::string_view needle) {
    return haystack.find(needle) != std::string::npos;
}

using ninfer::CapabilityReport;
using ninfer::DeviceCapability;
using ninfer::DeviceFacts;
using ninfer::ProbeRecord;
using ninfer::ProbeStatus;

[[nodiscard]] CapabilityReport all_pristine() { return CapabilityReport{}; }

void test_requirements_are_format_driven() {
    using ninfer::KvCacheStorage;
    using ninfer::requirements_for_kv_storage;
    using ninfer::set_contains;

    const auto nvfp4 = requirements_for_kv_storage(KvCacheStorage::Nvfp4Group16);
    expect(needs_contains(nvfp4, DeviceCapability::Nvfp4MmaBlockScale),
           "nvfp4 KV must require the nvfp4 (e2m1) MMA");
    expect(needs_contains(nvfp4, DeviceCapability::Iso4eKvCodec),
           "nvfp4 KV keeps ISO4E on the V plane by default (types.h:46-49)");
    expect(needs_contains(nvfp4, DeviceCapability::KernelImage),
           "every format needs the kernel-image baseline");
    expect(!needs_contains(nvfp4, DeviceCapability::Fp8MmaKindF8f6f4),
           "nvfp4 KV must not drag in the fp8 MMA");
    expect(!needs_contains(nvfp4, DeviceCapability::SetMaxNreg),
           "nvfp4 KV alone must not require the TMA/register path");

    const auto bf16 = requirements_for_kv_storage(KvCacheStorage::BFloat16);
    // bf16 KV is the ONE requirement with an alternate route, and the model has to say so in a
    // shape that cannot be read as a conjunction: NOTHING in all_of may carry Bf16Mma, and the
    // disjunction must carry BOTH arms. Asserting the disjunction alone would pass even if the bit
    // were ALSO a conjunct -- i.e. even if both arms were demanded, which is the pre-patch bug.
    expect(!set_contains(bf16.all_of, DeviceCapability::Bf16Mma) &&
               !set_contains(bf16.all_of, DeviceCapability::SimtFfmaAttention),
           "neither bf16 attention arm may be a CONJUNCT -- that is the pre-patch bug");
    expect(set_contains(bf16.any_of, DeviceCapability::Bf16Mma),
           "the bf16 requirement must still offer the tensor-core arm");
    expect(set_contains(bf16.any_of, DeviceCapability::SimtFfmaAttention),
           "the bf16 requirement must offer the SIMT FFMA arm");
    expect(set_contains(bf16.all_of, DeviceCapability::KernelImage),
           "the disjunction must not have swallowed the baseline");
    expect(!needs_contains(bf16, DeviceCapability::Nvfp4MmaBlockScale),
           "bf16 KV must not require the fp4 MMA");
    // Every OTHER storage is unchanged: a pure conjunction, exactly as strict as before.
    for (const auto storage : {KvCacheStorage::Int8Group64, KvCacheStorage::Fp8Group16,
                               KvCacheStorage::Nvfp4Group16, KvCacheStorage::Iso3Group16,
                               KvCacheStorage::E8Group64, KvCacheStorage::Dropped}) {
        expect(!needs_has_disjunction(requirements_for_kv_storage(storage)),
               "only bf16 may have an alternate route");
    }

    const auto rk4v4 = requirements_for_kv_storage(KvCacheStorage::E8Group64);
    expect(needs_contains(rk4v4, DeviceCapability::E8KvLattice), "rk4v4 KV must require the lattice");
    expect(needs_contains(rk4v4, DeviceCapability::Int8Mma),
           "rk4v4 KV codes are consumed by the int8 attention kernels (types.h:36-37)");

    const auto iso4e = requirements_for_kv_storage(KvCacheStorage::Iso3Group16);
    expect(needs_contains(iso4e, DeviceCapability::Iso4eKvCodec), "iso4e KV must require the codec");
    expect(!needs_contains(iso4e, DeviceCapability::Nvfp4MmaBlockScale),
           "iso4e KV alone must not require the fp4 MMA");

    const auto fp8 = requirements_for_kv_storage(KvCacheStorage::Fp8Group16);
    expect(needs_contains(fp8, DeviceCapability::Fp8MmaKindF8f6f4),
           "fp8 KV must require the fp8 MMA");

    // DType 口径与 KvCacheStorage 口径必须一致 (两层表说的是同一件事)。
    using ninfer::requirements_for_dtype;
    expect(requirements_for_dtype(ninfer::DType::NVFP4) == nvfp4,
           "DType::NVFP4 and KvCacheStorage::Nvfp4Group16 must agree");
    expect(requirements_for_dtype(ninfer::DType::ISO3) == iso4e,
           "DType::ISO3 and KvCacheStorage::Iso3Group16 must agree");
    expect(requirements_for_dtype(ninfer::DType::E8Kv) == rk4v4,
           "DType::E8Kv and KvCacheStorage::E8Group64 must agree");
}

// THE TASK. A build whose only CUDA target is sm_52 has no bf16 tensor core: Bf16Mma's probe body
// is behind __CUDA_ARCH__ >= 800, so it reads NotInBuild. Before this landing that single fact
// refused a bf16 KV tier before any kernel was chosen. It must now be ACCEPTED, and the acceptance
// must name the kernel that will actually run.
void test_sm52_accepts_bf16_and_names_the_kernel() {
    using ninfer::KvCacheStorage;
    using ninfer::ProbeRecord;
    using ninfer::ProbeStatus;
    using ninfer::requirements_for_kv_storage;

    CapabilityReport report{};
    report.record(DeviceCapability::KernelImage,
                  ProbeRecord{ProbeStatus::Supported, 0.0, 0.0, ""});
    report.record(DeviceCapability::Bf16Mma,
                  ProbeRecord{ProbeStatus::NotInBuild, 0.0, 0.0,
                              "probe body compiled out (guard: __CUDA_ARCH__ >= 800)"});
    // This record is what an sm_52 build produces, verified by RUNNING the launcher's own selector
    // in a binary built for sm_52 (dl/capor/s1_selector.cu: selected=1, __CUDA_ARCH_LIST__=520).
    report.record(DeviceCapability::SimtFfmaAttention,
                  ProbeRecord{ProbeStatus::Supported, 0.0, 0.0,
                              "route selected; bf16simt-ffma"});

    const auto needs = requirements_for_kv_storage(KvCacheStorage::BFloat16);
    const auto res   = ninfer::resolve_needs(needs, report);
    expect(res.satisfied, "an sm_52 build must ACCEPT a bf16 KV tier via the SIMT arm");
    expect(res.served_by.has_value() && *res.served_by == DeviceCapability::SimtFfmaAttention,
           "the SIMT arm is the one that resolved and must be the one named");

    const std::string line = ninfer::kv_storage_route_line(KvCacheStorage::BFloat16, report);
    expect(has(line, "accepted"), "the route line must say it is accepted");
    expect(has(line, "bf16simt-ffma"),
           "the route line must name the kernel as the launcher itself names it");

    // NAMED REFUSAL. With neither arm the answer is still a refusal, and it names BOTH arms and
    // the reason each is absent -- never a silent fallback to a kernel that cannot run.
    CapabilityReport neither = report;
    neither.record(DeviceCapability::SimtFfmaAttention,
                   ProbeRecord{ProbeStatus::NotInBuild, 0.0, 0.0,
                               "the SIMT FFMA bf16 attention route is NOT selected"});
    expect(!ninfer::resolve_needs(needs, neither).satisfied,
           "with no arm available the requirement must not be satisfied");
    const std::string refused = ninfer::kv_storage_route_line(KvCacheStorage::BFloat16, neither);
    expect(has(refused, "REFUSED"), "the route line must refuse");
    expect(has(refused, "bf16simt-ffma"), "the refusal must name the SIMT arm by name");
    expect(has(refused, "tensor core"), "the refusal must name the tensor-core arm");
    expect(has(refused, ninfer::probe_status_name(ProbeStatus::NotInBuild)),
           "the refusal must give each arm's status");
    expect(!has(refused, "falling back"), "a refusal must never advertise a fallback");
}

// The arm sm_120a takes must still be the tensor-core one: a disjunction that always preferred the
// new family would be a regression wearing a fix's clothes.
void test_tensor_core_arm_still_wins_where_it_exists() {
    using ninfer::KvCacheStorage;
    using ninfer::ProbeRecord;
    using ninfer::ProbeStatus;
    CapabilityReport report{};
    report.record(DeviceCapability::KernelImage, ProbeRecord{ProbeStatus::Supported, 0, 0, ""});
    report.record(DeviceCapability::Bf16Mma, ProbeRecord{ProbeStatus::Supported, 16.0, 16.0, ""});
    report.record(DeviceCapability::SimtFfmaAttention,
                  ProbeRecord{ProbeStatus::NotInBuild, 0, 0, "not selected on this build"});
    const auto res = ninfer::resolve_needs(
        ninfer::requirements_for_kv_storage(KvCacheStorage::BFloat16), report);
    expect(res.satisfied, "the tensor-core arm alone must satisfy the requirement");
    expect(res.served_by.has_value() && *res.served_by == DeviceCapability::Bf16Mma,
           "where the tensor-core arm exists it must be the one that resolves");
}

void test_tolerance() {
    expect(ninfer::probe_within(ninfer::kFp32ExactProbe, 16.0, 16.0), "exact fp hit rejected");
    expect(ninfer::probe_within(ninfer::kFp32ExactProbe, 64.0, 64.0000005), "fp slack too tight");
    expect(!ninfer::probe_within(ninfer::kFp32ExactProbe, 64.0, 63.5), "fp miss accepted");
    expect(ninfer::probe_within(ninfer::kIntegerProbe, 32.0, 32.0), "exact int hit rejected");
    expect(!ninfer::probe_within(ninfer::kIntegerProbe, 32.0, 31.0), "int miss accepted");
}

void test_not_probed_is_not_supported() {
    CapabilityReport report = all_pristine();
    const auto nvfp4        = ninfer::requirements_for_kv_storage(ninfer::KvCacheStorage::Nvfp4Group16);
    expect(!report.supported(DeviceCapability::Nvfp4MmaBlockScale), "pristine probe reads as支持");
    expect(!ninfer::resolve_needs(nvfp4, report).satisfied,
           "a never-probed requirement must not count as supported");
    const auto missing = report.first_unsupported(nvfp4.all_of);
    expect(missing.has_value(), "missing capability not reported");

    // 全通过之后才放行。
    for (std::size_t i = 0; i < ninfer::kDeviceCapabilityCount; ++i) {
        report.record(static_cast<DeviceCapability>(i), ProbeRecord{ProbeStatus::Supported, 0.0, 0.0, ""});
    }
    expect(ninfer::resolve_needs(nvfp4, report).satisfied,
           "all-supported report still refuses");

    // 一个 NumericMismatch 就必须把这条需求判死。
    report.record(DeviceCapability::Nvfp4MmaBlockScale,
                  ProbeRecord{ProbeStatus::NumericMismatch, 64.0, 61.0, "first mismatch 61"});
    expect(!ninfer::resolve_needs(nvfp4, report).satisfied, "numeric mismatch must fail the capability");
    expect(report.first_unsupported(nvfp4.all_of).value() == DeviceCapability::Nvfp4MmaBlockScale,
           "the mismatching capability must be the one reported");
}

void test_failure_message_is_actionable() {
    const DeviceFacts facts{"NVIDIA GeForce RTX 4090", 8, 9, "120a", "610.88"};

    CapabilityReport report{};
    report.record(DeviceCapability::Nvfp4MmaBlockScale,
                  ProbeRecord{ProbeStatus::DeviceHasNoImage, 64.0, 0.0,
                              "cudaErrorNoKernelImageForDevice (209)"});
    const std::string message = ninfer::capability_failure_message(
        DeviceCapability::Nvfp4MmaBlockScale, report, facts, "--kv-dtype nvfp4");
    expect(has(message, "--kv-dtype nvfp4"), "message must name who required the capability");
    expect(has(message, "src/ops/common/mma.cuh:90"),
           "message must name the kernel the capability comes from");
    expect(has(message, "cudaErrorNoKernelImageForDevice"),
           "message must carry the observed evidence");
    expect(has(message, "64"), "message must carry the expected value");
    expect(has(message, "RTX 4090"), "message must carry the device");
    expect(has(message, "CMAKE_CUDA_ARCHITECTURES=120a"), "message must carry the build arch");
    expect(has(message, "fix"), "message must offer a way out");
    expect(has(message, "89a"), "the rebuild hint must name this device's arch");

    // 每类不通过都要有自己的说法 (不能把 NotInBuild 说成"你的卡太老")。
    const ProbeStatus statuses[] = {ProbeStatus::NotInBuild, ProbeStatus::LaunchFailed,
                                    ProbeStatus::NumericMismatch, ProbeStatus::NotProbed};
    for (const ProbeStatus status : statuses) {
        CapabilityReport one{};
        one.record(DeviceCapability::Bf16Mma, ProbeRecord{status, 16.0, 0.0, "detail"});
        const std::string text = ninfer::capability_failure_message(
            DeviceCapability::Bf16Mma, one, facts, "--kv-dtype bf16");
        expect(has(text, ninfer::probe_status_name(status)), "status text missing from the message");
        expect(has(text, "fix"), "every failure mode needs a fix line");
    }

    CapabilityReport built_out{};
    built_out.record(DeviceCapability::SetMaxNreg,
                     ProbeRecord{ProbeStatus::NotInBuild, 0.0, 0.0,
                                 "probe body compiled out (guard: any __CUDA_ARCH_FEAT_SM*_ALL)"});
    const std::string not_in_build = ninfer::capability_failure_message(
        DeviceCapability::SetMaxNreg, built_out, facts, "--weights-profile nvfp4");
    expect(has(not_in_build, "__CUDA_ARCH_FEAT_SM*_ALL"),
           "NotInBuild must quote the guard that compiled the probe out");
    expect(!has(not_in_build, "your GPU is"), "the old numbered phrasing must be gone");
}

// 每条 capability 的 evidence 必须是**真实存在**的 file:line: 这是"不许臆造"的机器检查。
void test_evidence_points_at_real_lines() {
#ifdef NINFER_SOURCE_DIR
    const std::string root = NINFER_SOURCE_DIR;
    const std::regex reference(R"(([A-Za-z0-9_/.-]+\.(?:h|hpp|cuh|cu|cpp)):(\d+))");
    std::size_t checked = 0;
    for (std::size_t i = 0; i < ninfer::kDeviceCapabilityCount; ++i) {
        const auto& spec = ninfer::capability_probe_spec(static_cast<DeviceCapability>(i));
        const std::string evidence(spec.evidence);
        for (auto it = std::sregex_iterator(evidence.begin(), evidence.end(), reference);
             it != std::sregex_iterator(); ++it) {
            const std::string relative = (*it)[1].str();
            const long line            = std::stol((*it)[2].str());
            std::ifstream file(root + "/" + relative, std::ios::binary);
            expect(file.good(), ("evidence file does not exist: " + relative).c_str());
            if (!file.good()) { continue; }
            long lines = 0;
            std::string text;
            while (std::getline(file, text)) { ++lines; }
            expect(line >= 1 && line <= lines,
                   ("evidence line out of range: " + relative + ":" + std::to_string(line)).c_str());
            ++checked;
        }
    }
    expect(checked >= 9, "expected at least 9 file:line evidence references");
    std::printf("evidence references checked: %zu\n", checked);
#else
    std::printf("SKIP: NINFER_SOURCE_DIR not defined, evidence paths not checked\n");
#endif
}

void test_report_table_lists_every_capability() {
    CapabilityReport report{};
    report.record(DeviceCapability::Bf16Mma, ProbeRecord{ProbeStatus::Supported, 0.0, 0.0, ""});
    const std::string table = report.table();
    for (std::size_t i = 0; i < ninfer::kDeviceCapabilityCount; ++i) {
        const auto& spec = ninfer::capability_probe_spec(static_cast<DeviceCapability>(i));
        expect(has(table, std::string(spec.name)), "capability missing from the report table");
    }
}

} // namespace

int main() {
    test_requirements_are_format_driven();
    test_sm52_accepts_bf16_and_names_the_kernel();
    test_tensor_core_arm_still_wins_where_it_exists();
    test_tolerance();
    test_not_probed_is_not_supported();
    test_failure_message_is_actionable();
    test_evidence_points_at_real_lines();
    test_report_table_lists_every_capability();

    if (failures != 0) {
        std::printf("device capabilities test FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("device capabilities test OK\n");
    return 0;
}
