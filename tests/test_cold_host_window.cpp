// test_cold_host_window.cpp -- the WINDOW arithmetic of
// src/targets/qwen3_6/impl/runtime/cold_host_tier.h, and its red controls.
//
// WHAT THIS ANSWERS, in the brief's own terms: "what window admits enough host pages, what does
// it cost, and can the window be sized from the plan rather than a constant?"
//
//   1. the band max(0, F-K-H) <= W <= D-K, and the feasibility condition F <= D+H
//   2. the DERIVED --cold-host-bytes value, and why that constant may grow where the plan's
//      byte budget may not
//   3. the per-round cost, and the property that makes it different from the plan-budget arm:
//      the eviction term does NOT grow with the window
//   4. the answer to the plan-sizing question, as an ASSERTED number rather than an opinion:
//      sizing the window from the plan's own span is NOT a fix, and the check below is the
//      arithmetic that says so on the plan's own measured output
//
// Host-only: every input is a PARAMETER, no CUDA, no device, no model, no engine process.
// F, D and K are the tree's own measured/quoted ones and are written as literals here so
// this test does not silently re-derive them. H is NOT one of them: it is READ from the
// shipped default's own authority, because a pinned H is exactly how this file came to
// assert the PRE-CHANGE world. See kShippedHostBytes and kControl4GiBPages below.
//
//   F = 15,782 pages   at 1,010,048 tokens (INDEX1M2 sec.5.5a)
//   D = 10,148 pages   11.17 GiB of device KV headroom / 1.127 MiB per page (INDEX1M2 sec.5.5)
//   H = SHIPPED       --cold-host-bytes READ from EngineOptions::cold_host_bytes / page_bytes
//   K =     43 pages   the P6 selector's measured plan on the 1M D1 catalogue
//                      (sum_dir_recall_span_reachable: pages=43 span=[10112,12864))
//   page_bytes = 1.127 MiB measured, 1.13-1.21 MiB band

#include "ninfer/types.h" // EngineOptions::cold_host_bytes: the shipped default, READ not typed
#include "targets/qwen3_6/impl/runtime/cold_host_tier.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

using ninfer::targets::qwen3_6::detail::cold_host_bytes_for_window;
using ninfer::targets::qwen3_6::detail::cold_host_page_is_read_free;
using ninfer::targets::qwen3_6::detail::cold_host_page_is_window_free_declared;
using ninfer::targets::qwen3_6::detail::cold_host_window_band;
using ninfer::targets::qwen3_6::detail::cold_host_window_refusal;
using ninfer::targets::qwen3_6::detail::cold_host_window_round_cost;
using ninfer::targets::qwen3_6::detail::kColdHostPageTokens;

int failures = 0;
int checks   = 0;
int reported = 0;
constexpr int kReportLimit = 25;

void check(bool condition, const std::string& what) {
    ++checks;
    if (!condition) {
        ++failures;
        if (reported < kReportLimit) {
            ++reported;
            std::cout << "FAIL: " << what << "\n";
        }
    }
}

// The 1M configuration, on the tree's own numbers.
constexpr std::uint32_t k1MFrontierPages = 15782U;
constexpr std::uint32_t k1MDevicePages   = 10148U;
constexpr std::uint32_t kPlanPages       = 43U;
// THE PAGE SIZE, with its scope named so the next reader cannot silently swap objects.
// 1,181,696 B is the engine's MEASURED cold-slot page record: 128 x ops::kColdI8SlotBytes
// (include/ninfer/ops/cold_i8.h:11 = 16 B header + 8192 B E2M1 nibbles + 1024 B E4M3 g16
// scales) = 16 layers x 4 kv_heads x 2 planes, i.e. `turn_recall_page_bytes()`, the number
// the engine prints as `bytes_per_record=` on its [textcargo] line.
// tests/test_kv_paging_preallocation.cpp pins 128 x 9232 == 1,181,696 against the engine's
// own 16-int8-layer reading. It is 256-ALIGNED; 1,181,745 is not (1,181,745 mod 256 == 49)
// and therefore cannot be a HostKVPageLayout::page_stride at all -- plan_host_kv_page_layout
// always returns align_up(cursor, 256) (src/core/host_kv_arena.cpp:15 and :76).
// This file used to carry 1,181,745, which is floor(1.127 * 2^20): the PROSE rounding
// "1.127 MiB" expanded back into bytes, 49 B above the measured record. The band's verdicts
// are identical at both values AND at the two other sizes quoted tonight -- that is asserted
// in test_w6 -- so this is a correctness fix to a constant, not a way to make a check pass.
constexpr std::uint64_t kPageBytes       = 1181696ULL; // 1.126953 MiB = 9,232 B x 128, measured
static_assert(kPageBytes % 256ULL == 0ULL,
              "a HostKVPageLayout::page_stride is always align_up(cursor, 256) "
              "(src/core/host_kv_arena.cpp:15, :76): a page size that is not 256-aligned is "
              "not a page stride and must not be divided with as one");
// THE SHIPPED DEFAULT, READ FROM ITS OWN AUTHORITY -- never re-typed here.
// `EngineOptions::cold_host_bytes` (include/ninfer/types.h) is the member initialiser the
// engine ships; the CLI and serve front ends only OVERWRITE it when the flag is passed
// (apps/cli/options.cpp, src/serve/serve_options.cpp), so this struct is the value that
// reaches the tier (program_impl.h:1284 text_host_kv_page_stride -> :1301
// cold_budget.host_page_bytes -> product/kv_cold_tier_budget.h:130).
// NOT constexpr, and that is MEASURED rather than assumed: EngineOptions is not a literal
// type (it holds a std::string), so `constexpr ... = ninfer::EngineOptions{}.cold_host_bytes;`
// is rejected -- "temporary of non-literal type ... does not have 'constexpr' destructor".
// That is one of two independent reasons no static_assert can pin this default; the other is
// that the divisor is the per-plan runtime stride, not any fixed page size.
const std::uint64_t kShippedHostBytes = ninfer::EngineOptions{}.cold_host_bytes;
const std::uint32_t kShippedHostPages =
    static_cast<std::uint32_t>(kShippedHostBytes / kPageBytes);
// THE MUTATION CONTROL: the PRE-CHANGE world, kept so the empty-band arithmetic stays
// covered. Its page COUNT is the DIVISION of the budget under test, so the only literal is
// the budget itself and the control cannot go stale the way the shipped default just did.
constexpr std::uint64_t kControlHostBudget4GiB = 4ULL << 30;
constexpr std::uint32_t kControl4GiBPages      =
    static_cast<std::uint32_t>(kControlHostBudget4GiB / kPageBytes);
// Host capacity that IS available here: MemTotal 21.5 GiB, and the resident ladder's own limit at
// 786,432 tokens reported ~11.2 GiB disposable (INDEX1M2). 7 GiB is inside it with margin.
constexpr std::uint64_t kHostBudget7GiB  = 7ULL << 30;
constexpr std::uint32_t kHostPages7GiB   = static_cast<std::uint32_t>(kHostBudget7GiB / kPageBytes);

// ---------------------------------------------------------------------------
// 1. The band at the SHIPPED default -- READ from the authority, not re-typed as a
//    literal -- plus the 4 GiB mutation control that keeps the empty-band arithmetic
//    covered. The two worlds are separate on purpose: this section used to assert the
//    4 GiB world while claiming to be "the SHIPPED host budget", which is how it came
//    to be green against a default it contradicted.
// ---------------------------------------------------------------------------
void test_w1_the_band_at_1m_with_the_shipped_host_budget() {
    // THE SHIPPED DEFAULT, derived. If --cold-host-bytes moves again these move with it;
    // if it moves back below F - D they go RED, which is what a check is for.
    const auto shipped = cold_host_window_band(k1MFrontierPages, k1MDevicePages,
                                               kShippedHostPages, kPlanPages);
    check(kShippedHostPages >= shipped.required_host_pages(),
          "the SHIPPED --cold-host-bytes must hold the pages the device cannot: H = " +
              std::to_string(kShippedHostPages) + " >= F - D = " +
              std::to_string(shipped.required_host_pages()));
    check(shipped.shortfall_pages() == 0U,
          "so the shipped default must leave NO shortfall, got " +
              std::to_string(shipped.shortfall_pages()));
    check(shipped.feasible(),
          "and the band at the shipped default must be FEASIBLE (F <= D + H): " +
              std::to_string(k1MFrontierPages) + " <= " + std::to_string(k1MDevicePages) +
              " + " + std::to_string(kShippedHostPages));
    check(shipped.min_window_pages <= shipped.max_window_pages,
          "and it must be NON-EMPTY: min " + std::to_string(shipped.min_window_pages) +
              " <= max " + std::to_string(shipped.max_window_pages));
    check(cold_host_window_refusal(shipped).empty(),
          "a feasible band must refuse NOTHING (silence is the contract), got: " +
              cold_host_window_refusal(shipped));
    check(shipped.admits(shipped.min_window_pages) && shipped.admits(shipped.max_window_pages),
          "both endpoints of a non-empty band must be admitted");

    // (1b) THE MUTATION CONTROL: the same band at 4 GiB, which is what the flag used to
    // default to. These are the ORIGINAL section 1's assertions, kept -- inverting an
    // expectation instead of re-pointing it at the right world would be a deletion.
    const auto control = cold_host_window_band(k1MFrontierPages, k1MDevicePages,
                                               kControl4GiBPages, kPlanPages);
    check(control.max_window_pages == k1MDevicePages - kPlanPages,
          "W_max must be D - K = 10148 - 43 = 10105, got " +
              std::to_string(control.max_window_pages));
    check(control.min_window_pages == k1MFrontierPages - kPlanPages - kControl4GiBPages,
          "W_min must be F - K - H with H = 4 GiB / " + std::to_string(kPageBytes) + " = " +
              std::to_string(kControl4GiBPages) + ", got " +
              std::to_string(control.min_window_pages));
    // THE CONTROL'S WHOLE POINT: at 4 GiB the band IS empty. That is a fact about 4 GiB,
    // not a claim about the shipped default -- the distinction this file used to blur.
    check(control.min_window_pages > control.max_window_pages,
          "at a 4 GiB host budget the band must be EMPTY: min " +
              std::to_string(control.min_window_pages) + " > max " +
              std::to_string(control.max_window_pages));
    check(!control.feasible(), "and that band must report itself infeasible");
    check(!control.admits(k1MDevicePages),
          "no window at all fits while the host is 4 GiB -- including the largest one");
    // The shortfall is HOST capacity, and it is named in pages.
    check(control.required_host_pages() == k1MFrontierPages - k1MDevicePages,
          "the host requirement must be F - D = 5634 pages, got " +
              std::to_string(control.required_host_pages()));
    check(control.shortfall_pages() == 5634U - kControl4GiBPages,
          "the shortfall must be 5634 - " + std::to_string(kControl4GiBPages) + " = " +
              std::to_string(5634U - kControl4GiBPages) + " pages, got " +
              std::to_string(control.shortfall_pages()));
    // And the refusal must exist, be loud, and NAME the flag that fixes it.
    const std::string why = cold_host_window_refusal(control);
    check(!why.empty(), "an infeasible band must produce a refusal, not silence");
    // DERIVED, not frozen. This check used to grep the refusal for the literal "1997",
    // which was the shortfall at the stale 3,637-page count and is NOT the shortfall at
    // the true one. It is now the band's own shortfall, so it cannot drift again.
    check(why.find(std::to_string(control.shortfall_pages())) != std::string::npos,
          "the refusal must carry the shortfall in pages, got: " + why);
    check(why.find("--cold-host-bytes") != std::string::npos,
          "the refusal must name the knob that fixes it, got: " + why);
    check(why.find("plan's own byte budget") != std::string::npos,
          "and it must say that the PLAN's budget is a different budget, got: " + why);
}

// ---------------------------------------------------------------------------
// 2. The derived host budget: 7 GiB makes the band non-empty, and the DERIVATION is the point.
// ---------------------------------------------------------------------------
void test_w2_the_derived_host_budget_opens_the_band() {
    check(kHostPages7GiB >= 5634U,
          "7 GiB must hold the 5634 pages the device cannot, got " +
              std::to_string(kHostPages7GiB));
    const auto band = cold_host_window_band(k1MFrontierPages, k1MDevicePages, kHostPages7GiB,
                                            kPlanPages);
    check(band.feasible(), "the band must be feasible at 7 GiB");
    check(cold_host_window_refusal(band).empty(),
          "and a feasible band must refuse NOTHING (silence is the contract)");
    // THE PROPERTY THAT LICENSES THE BIGGER CONSTANT: it admits no window that overruns the
    // device, so it cannot admit the whole prefix.
    check(!band.admits(k1MDevicePages),
          "the window may never exceed D - K, so the bigger host budget cannot admit the "
          "whole prefix as resident");
    check(band.admits(band.min_window_pages) && band.admits(band.max_window_pages),
          "both endpoints of a non-empty band must be admitted");
    check(!band.admits(band.min_window_pages - 1U),
          "and one page below the minimum must NOT be (the band has hard edges)");
    // The DERIVED value, which is what --cold-host-bytes should be set to.
    const std::uint64_t derived =
        cold_host_bytes_for_window(k1MFrontierPages, k1MDevicePages, kPageBytes);
    check(derived == 5634ULL * kPageBytes,
          "cold_host_bytes_for_window must be (F - D) * page_bytes");
    check(derived <= kHostBudget7GiB,
          "and it must fit inside the 7 GiB recommendation, got " + std::to_string(derived) +
              " bytes needed vs " + std::to_string(kHostBudget7GiB) + " recommended");
}

// ---------------------------------------------------------------------------
// 3. The cost: the eviction term does NOT grow with the window, which is the whole difference
//    from the plan-budget arm.
// ---------------------------------------------------------------------------
void test_w3_the_per_round_cost_is_independent_of_the_window() {
    const auto cost = cold_host_window_round_cost(kPlanPages, kPageBytes, kColdHostPageTokens);
    // One token per round against 64-token pages: the round that crosses a page boundary pays
    // for the whole page, and there are 64 rounds per crossing.
    check(cost.evict_d2h_bytes == kPageBytes,
          "a crossing round must pay exactly ONE page of D2H, got " +
              std::to_string(cost.evict_d2h_bytes));
    check(cost.recall_h2d_bytes == kPlanPages * kPageBytes,
          "a recall round must pay K pages of H2D");
    // THE SHAPE THAT MATTERS, and it is the reason a window is affordable at 1M where the arm is
    // not: this function has NO window parameter. There is no argument a caller could pass that
    // would make the steady-state traffic grow with the window, because the traffic is set by how
    // fast the frontier ADVANCES, not by how much is resident.
    const auto tiny  = cold_host_window_round_cost(0U, kPageBytes, kColdHostPageTokens);
    check(tiny.evict_d2h_bytes == cost.evict_d2h_bytes,
          "the eviction term must be identical for a 1-page plan and a 43-page one");
    check(tiny.total_bytes() == kPageBytes,
          "a steady-state round (empty plan) must cost exactly one page, not one window");
    // Against the measured round: 18 KiB of D2H on a 1.127 MiB page is ~0.9 us at 20 GB/s, i.e.
    // 26.9 ms -> 27.0 ms, versus the arm's 256 MiB = ~13 ms on the SAME round.
    check(cost.evict_d2h_bytes * 64ULL == 64ULL * kPageBytes,
          "64 rounds of D2H must add up to ONE page per round-crossing, i.e. amortised 1/64");
}

// ---------------------------------------------------------------------------
// 4. CAN THE WINDOW BE SIZED FROM THE PLAN? The brief asks; this is the answer, asserted.
// ---------------------------------------------------------------------------
void test_w4_the_plan_cannot_size_the_window() {
    // The plan's own measured output is first_page=158, span=[10112,12864) on a 15,782-page
    // catalogue. Sizing the window as "everything from the earliest planned page to the
    // frontier" is the only plan-derived size that COVERS the plan without a second policy, and
    // it is:
    constexpr std::uint32_t kPlanFirstPage = 158U;
    constexpr std::uint32_t kPlanSpanPages = 43U;
    constexpr std::uint32_t kWindowFromPlan = k1MFrontierPages - kPlanFirstPage;
    check(kWindowFromPlan == 15624U,
          "the plan-derived window is F - first_page = 15624 pages, got " +
              std::to_string(kWindowFromPlan));
    // It is 1.54x the device, i.e. it does not fit -- and it is within 10% of the WHOLE prefix,
    // which is the shape the journal forbids by name.
    check(kWindowFromPlan > k1MDevicePages,
          "the plan-derived window must NOT fit the device (it is the forbidden shape)");
    check(kWindowFromPlan * 10U >= k1MFrontierPages * 9U,
          "and it is within 10% of the whole prefix: 15624 of 15782 pages");
    // So it is refused, by the band's OWN condition, without a new rule.
    const auto band = cold_host_window_band(k1MFrontierPages, k1MDevicePages, kHostPages7GiB,
                                            kPlanPages);
    check(!band.admits(kWindowFromPlan),
          "the band must REFUSE the plan-derived window -- no new predicate was needed to say "
          "so, because W <= D - K already excludes it");
    // WHAT THE PLAN *DOES* BUY, which is the honest form of the answer: it lowers W_min by K,
    // i.e. it lets the window be K pages SMALLER. It shifts pages between the two media; it does
    // not make the gap closable. That is the `F <= D + H` half of the condition, which does not
    // mention K at all.
    const auto without_plan = cold_host_window_band(k1MFrontierPages, k1MDevicePages,
                                                    kHostPages7GiB, 0U);
    check(band.min_window_pages + kPlanPages == without_plan.min_window_pages,
          "the plan must lower W_min by exactly K -- its whole contribution to feasibility");
    check(band.feasible() == without_plan.feasible(),
          "and it must NOT change feasibility at all: F <= D + H does not mention K");
    // The span the plan DOES supply is the recall working set, and it is bounded and small:
    check(kPlanSpanPages * kColdHostPageTokens == 2752U,
          "the plan's span is 2752 tokens = 43 pages, which is the FETCH size, not the "
          "resident size");
}

// ---------------------------------------------------------------------------
// 5. The declared window, and its red control.
// ---------------------------------------------------------------------------
void test_w5_the_declared_window_and_its_control() {
    // A window of 0 is full attention and admits NOTHING -- the same answer the model-declared
    // predicate gives, reached from the policy side.
    check(!cold_host_page_is_window_free_declared(0U, kColdHostPageTokens, 1000000U, 0U),
          "a declared window of 0 is full attention: no page is window-free");
    // ...and it AGREES with the model-declared predicate on a stack whose every window is 0,
    // which is what qwen3_6 declares. Two independent spellings of "full attention admits
    // nothing" that must not drift apart.
    const std::uint32_t full_attention_windows[2] = {0U, 0U};
    for (std::uint32_t page = 0U; page < 8U; ++page) {
        check(cold_host_page_is_window_free_declared(page, kColdHostPageTokens, 1000000U, 0U) ==
                  cold_host_page_is_read_free(page, kColdHostPageTokens, 1000000U,
                                              full_attention_windows),
              "the declared-window form must agree with the model-declared form on a "
              "full-attention stack, page " + std::to_string(page));
    }
    // A real window: page p is free once (p+1)*P + W*P <= frontier.
    constexpr std::uint32_t kWindow = 10240U; // 640 pages
    const std::uint32_t frontier = (kWindow + 200U) * kColdHostPageTokens;
    check(cold_host_page_is_window_free_declared(100U, kColdHostPageTokens, frontier, kWindow),
          "page 100 with a 200-page cushion must be window-free");
    check(!cold_host_page_is_window_free_declared(kWindow + 199U, kColdHostPageTokens, frontier,
                                                  kWindow),
          "and the page just inside the cushion must NOT be");
    // RED CONTROL: the predicate must be able to say NO, and it must say it for the reason
    // claimed. A zero-token page is refused rather than divided by.
    check(!cold_host_page_is_window_free_declared(0U, 0U, 1000000U, kWindow),
          "a zero page_tokens must be refused, not divided by");
    // The boundary is EXACT and one-sided: page_end + window_begin == frontier still admits,
    // frontier - 1 does not. That is the `+ W` slack kept deliberately absurd (eviction must be
    // impossible, not unlikely).
    // page_end = (page+1)*P and window_begin = window_pages*P, so the exact boundary for
    // page 200 is (200 + 1 + kWindow) * P -- one page MORE than (page + window_pages) * P,
    // because `page` is zero-based and `page_end` is exclusive.
    const std::uint32_t exact = (200U + 1U + kWindow) * kColdHostPageTokens;
    check(cold_host_page_is_window_free_declared(200U, kColdHostPageTokens, exact, kWindow),
          "page_end + window_begin == frontier must be ADMITTED (inclusive)");
    check(!cold_host_page_is_window_free_declared(200U, kColdHostPageTokens, exact - 1U, kWindow),
          "and frontier - 1 must NOT be (the boundary is one-sided)");
}

// ---------------------------------------------------------------------------
// 6. THE PAGE SIZE THE BAND IS COMPUTED WITH -- what each disputed size IS, and why the
//    dispute changes no endpoint. Four sizes were quoted tonight; only two of them are
//    256-aligned, and a HostKVPageLayout::page_stride is always align_up(cursor, 256), so
//    the other two cannot be page strides at all. Comparing them as if they were is the
//    mistake; the 49 B is its symptom, not its cause.
// ---------------------------------------------------------------------------
void test_w6_the_page_size_dispute_changes_no_endpoint() {
    constexpr std::uint64_t kWrittenNowhere  = 1180909ULL; // 4 GiB / 3,637 -- in NO file
    constexpr std::uint64_t kInt8Record      = 1181696ULL; // 128 x 9,232, the int8 stack
    constexpr std::uint64_t kProse1p127MiB   = 1181745ULL; // floor(1.127 * 2^20), 49 mod 256
    constexpr std::uint64_t kNvfp4Record9536 = 1220608ULL; // 128 x 9,536, a retired slot size
    // The discriminator, as arithmetic. A page stride is 256-aligned by construction, so
    // the alignment test says which of the four can be one and which are prose.
    check(kInt8Record % 256ULL == 0ULL, "the int8 cold-slot record must be 256-aligned");
    check(kNvfp4Record9536 % 256ULL == 0ULL, "and so must the nvfp4 cold-slot record");
    check(kProse1p127MiB % 256ULL == 49ULL,
          "1,181,745 must NOT be 256-aligned: it is floor(1.127 MiB), a prose rounding");
    check(kWrittenNowhere % 256ULL == 237ULL,
          "1,180,909 must NOT be 256-aligned either: it is 4 GiB / 3,637, an inverse");
    check(kInt8Record == 128ULL * 9232ULL, "1,181,696 = 128 slots x 9,232 B");
    check(kNvfp4Record9536 == 128ULL * 9536ULL, "1,220,608 = 128 slots x 9,536 B");
    check(kProse1p127MiB == kInt8Record + 49ULL,
          "and the prose literal is exactly 49 B above the record it restates");
    // THE INVARIANT THE DISPUTE IS ACTUALLY ABOUT: the band's verdict at EVERY candidate
    // size is the same. If a future page size crosses either line this fires, instead of a
    // claim in a comment going quietly stale. This is also the answer to "why not a
    // static_assert": the default is not a constant expression (see kShippedHostBytes) and
    // the divisor is a runtime stride -- so the guard lives here, at the candidate values.
    const std::uint64_t candidates[4] = {kWrittenNowhere, kInt8Record, kProse1p127MiB,
                                         kNvfp4Record9536};
    for (std::uint64_t page_bytes : candidates) {
        const std::uint32_t h4 =
            static_cast<std::uint32_t>(kControlHostBudget4GiB / page_bytes);
        const auto at4 = cold_host_window_band(k1MFrontierPages, k1MDevicePages, h4, kPlanPages);
        check(at4.min_window_pages > at4.max_window_pages,
              "at 4 GiB the band must be EMPTY at page_bytes = " + std::to_string(page_bytes));
        const std::uint32_t h7 = static_cast<std::uint32_t>(kHostBudget7GiB / page_bytes);
        const auto at7 = cold_host_window_band(k1MFrontierPages, k1MDevicePages, h7, kPlanPages);
        check(at7.feasible() && at7.min_window_pages <= at7.max_window_pages,
              "at 7 GiB the band must be NON-EMPTY and feasible at page_bytes = " +
                  std::to_string(page_bytes));
    }
}
} // namespace

int main() {
    test_w1_the_band_at_1m_with_the_shipped_host_budget();
    test_w2_the_derived_host_budget_opens_the_band();
    test_w3_the_per_round_cost_is_independent_of_the_window();
    test_w4_the_plan_cannot_size_the_window();
    test_w5_the_declared_window_and_its_control();
    test_w6_the_page_size_dispute_changes_no_endpoint();
    std::cout << "cold_host_window: " << checks << " checks, " << failures << " failures -> "
              << (failures == 0 ? "PASS" : "FAIL") << "\n";
    return failures == 0 ? 0 : 1;
}
