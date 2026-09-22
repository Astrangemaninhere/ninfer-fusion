// coldagree/proposed/test_cold_page_units.cpp
//
// ⚠️ THIS FILE IS A PROPOSAL. It lives in dl/coldagree/proposed/ and is NOT registered.
// The registration line it needs is in dl/coldagree/proposed/tests_CMakeLists.patch, and
// the SEQUENCING rule that makes the registration safe is in dl/coldagree/REPORT.md TASK 2.
//
// THE PIN THE TREE DOES NOT HAVE: the host page unit and the cold page record, read from
// BOTH authorities in ONE translation unit. No TU in the tree includes both headers today
// (core/host_kv_arena.h and product/kv_tier_formats.h have disjoint includer sets), which
// is why tests/test_cold_host_window.cpp:79 could name the cold record `kPageBytes` and
// nothing objected.
//
// CARD-FREE BY CONSTRUCTION, so it is registerable in the same shape as its sibling
// ninfer_cold_host_window_test (tests/CMakeLists.txt:926-929: LIBRARIES ninfer_core
// ninfer_artifact, documented there as host-only "on a box with no free device"):
//   * the HOST side calls the real plan_host_kv_page_layout() on a DECLARED geometry --
//     no device, no decoder, no engine;
//   * the COLD side reads the cuda-free product/kv_tier_formats.h constants and mirrors
//     the two functions that pick a record (decoder_state.cpp cold_slot_codec_of and
//     cold_slot_stride_for) rather than calling them, because those two live in a .cpp
//     anonymous namespace and in a target TU;
//   * what CANNOT be card-free is the LIVE pair: turn_recall_page_bytes() is a member of
//     ProgramImplCore and needs a planned decoder and a CUDA device. That half is named
//     in the report as unregisterable today, by name, and it is the half the shadowed
//     program_impl.h check covers at runtime instead.
//
// ⚠️ THE VALUES IN test_a_* ARE THE CLAIM OF dl/coldagree/REPORT.md TASK A.2 AND THEY ARE
// THIS FILE'S FALSIFIER. If any of them is wrong, that report section is wrong and these
// numbers must be replaced by this test's own output -- which is the whole point of
// pinning them here instead of in prose.

#include "core/host_kv_arena.h"                  // HostKVPageLayout, plan_host_kv_page_layout
#include "product/kv_tier_formats.h"             // the cold record, cuda-free
#include "targets/qwen3_6/impl/runtime/cold_page_unit_check.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks   = 0;

void check(bool condition, const std::string& what) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cout << "FAIL: " << what << "\n";
    }
}

// ---- THE SHAPE, read from the tree -------------------------------------------------
// src/targets/qwen3_6_27b/impl/config.h:59-60, :96  kv_heads 4, head_dim 256, 16 layers
// src/core/paged_kv_cache.h:17                       kPagedKVPageSize 64
constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kKvHeads = 4;
constexpr std::uint32_t kLayers = 16;
// DECLARED, not read: kNvfp4KvQuantGroup lives in the target's EXPORT header
// (src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h:18), which is a
// PRIVATE include of ninfer_engine -- reaching it would make this test need the engine
// and therefore the card, which is the one thing it is built not to need. The value is
// pinned by this test's geometry instead: if the group moves, test_a's H rows move.
constexpr std::int32_t kGroup16 = 16;

// ---- THE TWO FUNCTIONS THAT PICK A COLD RECORD, mirrored ---------------------------
// decoder_state.cpp cold_slot_codec_of(): I8/NVFP4/BF16/E8Kv named, everything else None.
// decoder_state.cpp cold_slot_stride_for(): Int8Raw -> the raw record, else the rANS
// record -- and "else" includes None, because a layer with no codec still owns a cold
// tensor at the WIDEST record ("their footprint is exactly what it is today").
[[nodiscard]] constexpr bool cold_record_is_raw(ninfer::DType dtype) noexcept {
    switch (dtype) {
    case ninfer::DType::I8:
    case ninfer::DType::BF16:
    case ninfer::DType::E8Kv: return true;
    default: return false; // NVFP4 rANS, and the widest record for fp8/iso4e/rk3v4/rk2v4
    }
}

[[nodiscard]] constexpr std::int32_t cold_record_bytes(ninfer::DType dtype) noexcept {
    return cold_record_is_raw(dtype) ? ninfer::product::kKvColdInt8PayloadBytes
                                     : ninfer::product::kKvColdPoolStrideBytes;
}

// turn_recall_page_bytes(): sum over slot-bearing layers of cold_slots.nb[3], and
// nb[3] = stride x kv_heads x 2 (decoder_state.cpp:1190-1192).
[[nodiscard]] constexpr std::uint64_t cold_page_record(const std::vector<ninfer::DType>& stack) {
    std::uint64_t bytes = 0;
    for (const ninfer::DType d : stack) {
        bytes += static_cast<std::uint64_t>(cold_record_bytes(d)) * kKvHeads * 2U;
    }
    return bytes;
}

// The plane list decoder_state.cpp:384-434 pushes for ONE layer of `dtype`.
[[nodiscard]] ninfer::KVPageGeometry geometry_for(const std::vector<ninfer::DType>& stack) {
    ninfer::KVPageGeometry geometry;
    geometry.page_tokens = 64;
    geometry.device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor;
    for (const ninfer::DType d : stack) {
        const std::int32_t group = (d == ninfer::DType::I8 || d == ninfer::DType::E8Kv)
                                       ? 64
                                       : kGroup16;
        switch (d) {
        case ninfer::DType::BF16:
            geometry.planes.push_back({ninfer::DType::BF16, kHeadDim, kKvHeads, 256});
            geometry.planes.push_back({ninfer::DType::BF16, kHeadDim, kKvHeads, 256});
            break;
        case ninfer::DType::I8:
            geometry.planes.push_back({ninfer::DType::I8, kHeadDim, kKvHeads, 256});
            geometry.planes.push_back({ninfer::DType::I8, kHeadDim, kKvHeads, 256});
            geometry.planes.push_back({ninfer::DType::FP16, kHeadDim / group, kKvHeads, 256});
            geometry.planes.push_back({ninfer::DType::FP16, kHeadDim / group, kKvHeads, 256});
            break;
        case ninfer::DType::E8Kv:
            geometry.planes.push_back({ninfer::DType::U8, kHeadDim / 2, kKvHeads, 256});
            geometry.planes.push_back({ninfer::DType::U8, kHeadDim / 2, kKvHeads, 256});
            geometry.planes.push_back({ninfer::DType::FP16, kHeadDim / group, kKvHeads, 256});
            geometry.planes.push_back({ninfer::DType::FP16, kHeadDim / group, kKvHeads, 256});
            break;
        case ninfer::DType::FP8_E4M3FN:
            geometry.planes.push_back({ninfer::DType::FP8_E4M3FN, kHeadDim, kKvHeads, 256});
            geometry.planes.push_back({ninfer::DType::FP8_E4M3FN, kHeadDim, kKvHeads, 256});
            geometry.planes.push_back(
                {ninfer::DType::FP8_E4M3FN, kHeadDim / group, kKvHeads, 256});
            geometry.planes.push_back(
                {ninfer::DType::FP8_E4M3FN, kHeadDim / group, kKvHeads, 256});
            break;
        default: // NVFP4 / ISO4E, no residual
            geometry.planes.push_back({ninfer::DType::U8, kHeadDim / 2, kKvHeads, 256});
            geometry.planes.push_back({ninfer::DType::U8, kHeadDim / 2, kKvHeads, 256});
            geometry.planes.push_back({ninfer::DType::FP8_E4M3FN, kHeadDim / group, kKvHeads, 256});
            geometry.planes.push_back({ninfer::DType::FP8_E4M3FN, kHeadDim / group, kKvHeads, 256});
            break;
        }
    }
    return geometry;
}

struct Stack {
    const char* name;
    std::vector<ninfer::DType> dtypes;
};

[[nodiscard]] std::vector<Stack> stacks() {
    using ninfer::DType;
    return {
        {"all NVFP4/ISO4E", std::vector<DType>(kLayers, DType::NVFP4)},
        {"factory 6xrk4v4+10xnvfp4", [] {
             std::vector<DType> v(6, DType::E8Kv);
             v.insert(v.end(), 10, DType::NVFP4);
             return v;
         }()},
        {"all E8Kv (rk4v4)", std::vector<DType>(kLayers, DType::E8Kv)},
        {"all I8", std::vector<DType>(kLayers, DType::I8)},
        {"all FP8_E4M3FN", std::vector<DType>(kLayers, DType::FP8_E4M3FN)},
        {"all BF16", std::vector<DType>(kLayers, DType::BF16)},
    };
}

// ---- the 1M shape (REPLAN1M / HOOKMS) and the two --cold-host-bytes milestones -----
constexpr std::uint32_t k1MFrontierPages = 15782U;
constexpr std::uint32_t k1MDevicePages   = 10148U;
constexpr std::uint64_t kHost4GiB        = 4ULL << 30;
constexpr std::uint64_t kHost7GiB        = 7ULL << 30;
constexpr std::uint64_t kRetired4BitRecord = 9536ULL; // kv_tier_formats.h:523
constexpr std::uint32_t kRequiredHostPages = k1MFrontierPages - k1MDevicePages; // 5,634

// =====================================================================================
// A. THE TWO PAGE UNITS, READ FROM THEIR OWN AUTHORITIES. THIS IS THE PIN.
// =====================================================================================
void test_a_the_two_units_are_different_numbers() {
    struct Expected {
        const char* name;
        std::uint64_t stride;
        std::uint64_t record;
    };
    // ⚠️ THE CLAIM. If a row is wrong, REPORT.md TASK A.2 is wrong: replace it with this
    // test's own printout, do not adjust the test.
    const Expected expected[] = {
        {"all NVFP4/ISO4E", 1179648ULL, 1232896ULL},
        {"factory 6xrk4v4+10xnvfp4", 1155072ULL, 1213696ULL},
        {"all E8Kv (rk4v4)", 1114112ULL, 1181696ULL},
        {"all I8", 2162688ULL, 1181696ULL},
        {"all FP8_E4M3FN", 2228224ULL, 1232896ULL},
        {"all BF16", 4194304ULL, 1181696ULL},
    };
    std::size_t i = 0;
    for (const Stack& s : stacks()) {
        const std::uint64_t stride =
            static_cast<std::uint64_t>(ninfer::plan_host_kv_page_layout(geometry_for(s.dtypes))
                                           .page_stride);
        const std::uint64_t record = cold_page_record(s.dtypes);
        const Expected& e = expected[i];
        check(std::string(s.name) == e.name, "stack order changed");
        check(stride == e.stride, std::string("H (HostKVPageLayout::page_stride) for ") + s.name +
                                      " = " + std::to_string(stride) + ", expected " +
                                      std::to_string(e.stride));
        check(record == e.record, std::string("D (the cold page record) for ") + s.name + " = " +
                                      std::to_string(record) + ", expected " +
                                      std::to_string(e.record));
        // ⭐ THE CORRECTION OF dl/diskblast/REPORT.md section 3.6, AS A CHECK: the two
        // authorities are NOT the same number on ANY stack, so "one value, two
        // authorities" was never true -- 1,181,696 is D only.
        check(stride != record, std::string("H and D must differ on ") + s.name +
                                    " -- if they are equal here and the numbers above were "
                                    "right, REPORT.md TASK A.2 needs re-deriving");
        // And the retired record must not be either of them anywhere.
        check(stride != kLayers * kKvHeads * 2 * kRetired4BitRecord,
              std::string("H must not be the retired 128 x 9,536 on ") + s.name);
        check(record != kLayers * kKvHeads * 2 * kRetired4BitRecord,
              std::string("D must not be the retired 128 x 9,536 on ") + s.name);
        ++i;
    }
}

// =====================================================================================
// B. THE CONSEQUENCE THE FLEET'S ARITHMETIC CANNOT SEE: on a byte cap, a bigger
//    resident page is FEWER pages, and 1M's host bill is a function of the dtype table.
// =====================================================================================
void test_b_the_band_is_dtype_conditional() {
    const std::uint64_t fleet_record = 1181696ULL; // what every quoted H has divided by
    for (const Stack& s : stacks()) {
        const std::uint64_t stride =
            static_cast<std::uint64_t>(static_cast<std::uint64_t>(ninfer::plan_host_kv_page_layout(geometry_for(s.dtypes)).page_stride));
        const std::uint32_t at7 =
            static_cast<std::uint32_t>(kHost7GiB / stride); // the tier's own capacity
        const std::uint32_t quoted =
            static_cast<std::uint32_t>(kHost7GiB / fleet_record); // the fleet's H
        const bool tier_closes = at7 >= kRequiredHostPages;
        const bool quoted_closes = quoted >= kRequiredHostPages;
        // The two denominators must AGREE on the stacks where 1M is reachable under
        // both, and DISAGREE exactly where the conflation changes the answer.
        const auto verdict =
            ninfer::targets::qwen3_6::detail::cold_page_unit_verdict(
                kHost7GiB, stride, fleet_record, kRequiredHostPages);
        check(verdict.diverges == (tier_closes != quoted_closes),
              std::string("the divergence flag must equal the disagreement, on ") + s.name);
        if (std::string(s.name) == "all NVFP4/ISO4E" ||
            std::string(s.name) == "factory 6xrk4v4+10xnvfp4" ||
            std::string(s.name) == "all E8Kv (rk4v4)") {
            check(!verdict.diverges, std::string("the shipped/factory stacks must stay SILENT: ") + s.name);
        } else {
            check(verdict.diverges && !tier_closes,
                  std::string("int8/bf16/fp8 must FIRE and must NOT close 1M at 7 GiB: ") + s.name);
        }
        // The pre-move 4 GiB world failed under BOTH denominators on every stack, so the
        // band being empty there is a different finding and this check says nothing.
        check(!ninfer::targets::qwen3_6::detail::cold_page_unit_verdict(
                   kHost4GiB, stride, fleet_record, kRequiredHostPages)
                   .diverges,
              std::string("4 GiB fails under both denominators on ") + s.name);
    }
}

// =====================================================================================
// C. NEGATIVE CONTROL: the check is sensitive to WHICH unit it is handed. Feeding it
//    one LAYER's record instead of the page record must be visible, otherwise "silent"
//    above would only mean "the predicate ignores its arguments".
// =====================================================================================
void test_c_the_predicate_reads_its_arguments() {
    using ninfer::targets::qwen3_6::detail::cold_page_unit_verdict;
    const auto same        = cold_page_unit_verdict(kHost7GiB, 1179648ULL, 1181696ULL, kRequiredHostPages);
    const auto one_layer   = cold_page_unit_verdict(kHost7GiB, 1179648ULL, 73856ULL, kRequiredHostPages);
    const auto tiny        = cold_page_unit_verdict(1ULL << 20, 1179648ULL, 1181696ULL, kRequiredHostPages);
    check(!same.diverges, "the shipped pair must be silent");
    check(one_layer.pages_by_record == (kHost7GiB / 73856ULL),
          "the record argument must move pages_by_record");
    check(tiny.pages_by_stride == 0U && tiny.pages_by_record == 0U,
          "a cap below one page admits nothing, the rule kv_cold_tier_budget.h:108-113 states");
    check(!tiny.diverges, "a cap that admits nothing under both is not a divergence");
    const auto off = cold_page_unit_verdict(kHost7GiB, 1179648ULL, 0ULL, kRequiredHostPages);
    check(!off.applicable && !off.diverges,
          "no cold record (ColdPolicy::Host) means not applicable, never divergent");
}

void report_table() {
    std::cout << "stack                        H=page_stride   D=record      "
                 "7GiB/stride  7GiB/record  verdict\n";
    for (const Stack& s : stacks()) {
        const std::uint64_t stride = static_cast<std::uint64_t>(ninfer::plan_host_kv_page_layout(geometry_for(s.dtypes)).page_stride);
        const std::uint64_t record = cold_page_record(s.dtypes);
        const auto v = ninfer::targets::qwen3_6::detail::cold_page_unit_verdict(
            kHost7GiB, stride, record, kRequiredHostPages);
        std::cout << "  " << s.name << "  " << stride << "  " << record << "  "
                  << v.pages_by_stride << "  " << v.pages_by_record << "  "
                  << (v.diverges ? "FIRE" : "silent") << "\n";
    }
}

} // namespace

int main() {
    test_a_the_two_units_are_different_numbers();
    test_b_the_band_is_dtype_conditional();
    test_c_the_predicate_reads_its_arguments();
    report_table();
    std::cout << "cold_page_units: " << checks << " checks, " << failures << " failures -> "
              << (failures == 0 ? "PASS" : "FAIL") << "\n";
    return failures == 0 ? 0 : 1;
}
