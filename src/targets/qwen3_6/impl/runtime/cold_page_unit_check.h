#pragma once

// ===========================================================================
// THE HOST PAGE UNIT AND THE COLD PAGE RECORD, SIDE BY SIDE.
//
// This header exists because the tree has THREE ways to say "one page" and no
// site that compares any two of them (dl/coldagree/REPORT.md, TASK A):
//
//   H  HostKVPageLayout::page_stride   src/core/host_kv_arena.cpp:76
//        align_up(cursor, 256) over the page pool's RESIDENT plane geometry
//        (decoder_state.cpp:367-436). This is the unit the Cold Host tier is
//        built from: program_impl.h text_host_kv_page_stride -> cold_budget
//        .host_page_bytes -> product/kv_cold_tier_budget.h:130, and the unit
//        HostKVArena divides --cold-host-bytes by (cold_host_tier.h:471-473).
//   D1 turn_recall_page_bytes()        program_impl.h:13335-13345
//        sum over slot-bearing layers of cold_slots.nb[3]. This is the unit
//        the DISK FILE-SLOT BITMAP divides --cold-disk-bytes by
//        (program_impl.h:1134-1143 -> :11786-11794).
//   D2 spill_page_bytes                program_impl.h:1403-1408
//        an independently written loop with the identical body, assigned to
//        cold_budget.disk_page_bytes at :1409: the unit the LADDER's disk rung
//        is priced in (product/kv_cold_tier_budget.h:134-135).
//
// H and D1 are NOT the same object and are NOT equal on any shipped stack: H
// is a function of the RESIDENT plane geometry and D is a function of the cold
// slot RECORD, so they move with different dtype axes. On the 27B text stack
// (head_dim 256, kv_heads 4, 16 layers: config.h:59-60, :96) H is
// 1,179,648 B for an all-NVFP4/ISO4E table and 2,162,688 B for an all-int8
// table, while D is 1,181,696 B (all int8-raw 9232 B records) or 1,232,896 B
// (all nvfp4 rANS 9632 B records).
//
// ⚠️ THAT MEANS THE ONE COINCIDENCE THE FLEET HAS BEEN QUOTING -- "1,181,696
// B/page, the same number out of two different functions" -- IS NOT A
// COINCIDENCE AND NOT AN EQUALITY. 1,181,696 is D only (128 x ops::kColdI8SlotBytes).
// It became the window band's `H` because tests/test_cold_host_window.cpp:79
// names it `kPageBytes`, and its own comment at :67 says in words that it is
// the cold-slot record. A number that moves independently of the tier it is
// used to size is the reading hazard this header closes.
//
// WHAT IS CHECKED, AND WHY IT IS A RUNTIME CHECK AND NOT A static_assert:
//   * H is produced by plan_host_kv_page_layout(), a THROWING, vector-
//     allocating runtime function (src/core/host_kv_arena.cpp:47-78); it is
//     not constexpr and cannot be. D1 is a non-static member function of
//     ProgramImplCore and needs a live decoder and live tensor views; it can
//     never appear in a static_assert. No TU even includes both headers
//     (core/host_kv_arena.h and product/kv_tier_formats.h have disjoint
//     includer sets), and layouts_impl.h:2066-2067 states in words that BOTH
//     quantities are invisible at the paging-plan site. So the house's
//     two-sided static_assert idiom (decoder_state.cpp:838-848, :863) has NO
//     LEGAL SITE for this pair. See dl/coldagree/REPORT.md TASK A.4.
//   * The predicate below is the part that CAN be made a compile-time
//     statement, and is: it is constexpr, CUDA-free, allocation-free, and
//     probe/pin_probe.cpp static_asserts it over every stack's numbers.
//
// WHAT FIRES: `diverges` is true only when the two denominators DISAGREE about
// whether --cold-host-bytes covers the committed prefix (F <= D + H). It is
// silent when the question has the same answer under both, which is the case
// on the shipped NVFP4/E8Kv stacks, on the factory 6xE8Kv+10xNVFP4 table, and
// on every run whose context fits the device pool. It fires on the int8 /
// bf16 / fp8 stacks, where the resident page is 1.83x-3.55x the cold record
// and --cold-host-bytes, being a BYTE cap, silently holds far fewer pages than
// the band arithmetic assumes. A check that fired on every stack would be
// decoration; this one fires exactly when the answer changes.
// ===========================================================================

#include <cstdint>
#include <string>

namespace ninfer::targets::qwen3_6::detail {

struct ColdPageUnitVerdict {
    bool applicable      = false; // both units known (a tier with no cold slots is off)
    bool diverges        = false; // the two denominators disagree about coverage
    std::uint32_t pages_by_stride = 0; // --cold-host-bytes / HostKVPageLayout::page_stride
    std::uint32_t pages_by_record = 0; // --cold-host-bytes / turn_recall_page_bytes()
    std::uint32_t required        = 0; // F - D, pages; 0 when the device pool already covers F
};

// The SAME rule product/kv_cold_tier_budget.h:108-113 states, restated here
// rather than re-derived: a cap smaller than one page admits nothing, and the
// result saturates at uint32. One definition, two callers.
[[nodiscard]] constexpr std::uint32_t cold_page_unit_pages(std::uint64_t bytes,
                                                           std::uint64_t unit) noexcept {
    if (unit == 0) { return 0; }
    const std::uint64_t pages = bytes / unit;
    return pages > 0xFFFFFFFFULL ? 0xFFFFFFFFU : static_cast<std::uint32_t>(pages);
}

// `required_host_pages` is `frontier_pages - device_pages`, the *condition*
// cold_host_tier.h:316 states and cold_host_window_refusal() reports. The
// band's own W_min subtracts the plan cap as well, and that is deliberately
// NOT this predicate's job: the question here is narrower and older -- can the
// host cap close the prefix at all -- because that is the question whose
// answer flips.
[[nodiscard]] constexpr ColdPageUnitVerdict
cold_page_unit_verdict(std::uint64_t host_bytes, std::uint64_t stride_bytes,
                       std::uint64_t record_bytes,
                       std::uint32_t required_host_pages) noexcept {
    ColdPageUnitVerdict verdict{};
    verdict.required = required_host_pages;
    // A policy whose cold pool is off (ColdPolicy::Host: effective_cold_pages
    // returns 0, so no layer owns a cold record and turn_recall_page_bytes()
    // returns 0) has no second denominator to disagree with. Not applicable,
    // not divergent -- a check that fired here would refuse a shipped default.
    if (stride_bytes == 0 || record_bytes == 0) { return verdict; }
    verdict.applicable      = true;
    verdict.pages_by_stride = cold_page_unit_pages(host_bytes, stride_bytes);
    verdict.pages_by_record = cold_page_unit_pages(host_bytes, record_bytes);
    verdict.diverges        = (verdict.pages_by_stride >= required_host_pages) !=
                              (verdict.pages_by_record >= required_host_pages);
    return verdict;
}

// The one-shot line. Every number is named and the two verdicts are printed as
// verdicts, because the defect this reports is precisely that a reader cannot
// tell which denominator produced a page count they were handed.
[[nodiscard]] inline std::string
cold_page_unit_message(std::uint64_t host_bytes, std::uint64_t stride_bytes,
                       std::uint64_t record_bytes, std::uint32_t frontier_pages,
                       std::uint32_t device_pages, const ColdPageUnitVerdict& verdict) {
    const std::string prefix = "[cold] HOST PAGE UNIT VS COLD RECORD: ";
    if (!verdict.diverges) { return {}; }
    return prefix +
           "--cold-host-bytes=" + std::to_string(host_bytes) + " B holds " +
           std::to_string(verdict.pages_by_stride) + " pages at " +
           "HostKVPageLayout::page_stride=" + std::to_string(stride_bytes) +
           " B (the unit the Cold Host tier and its HostKVArena are built from: " +
           "program_impl.h text_host_kv_page_stride -> cold_budget.host_page_bytes) but " +
           std::to_string(verdict.pages_by_record) + " pages at turn_recall_page_bytes()=" +
           std::to_string(record_bytes) +
           " B (the unit every --cold-disk-bytes figure and the window band's H have been " +
           "quoted in). The committed prefix needs " + std::to_string(verdict.required) +
           " host pages (frontier=" + std::to_string(frontier_pages) +
           " device=" + std::to_string(device_pages) +
           "), so \"does --cold-host-bytes cover the prefix\" is YES under one denominator " +
           "and NO under the other. The tier enforces the page_stride one: raise " +
           "--cold-host-bytes to at least " +
           std::to_string(static_cast<std::uint64_t>(verdict.required) * stride_bytes) +
           " B (" + std::to_string(verdict.required) +
           " pages), or lower --kv-capacity. Reconcile the two units before quoting either. " +
           "(dl/coldagree/REPORT.md TASK A)\n";
}

} // namespace ninfer::targets::qwen3_6::detail
