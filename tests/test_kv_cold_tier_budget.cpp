// Unit test for the cold-KV tier budgets (product/kv_cold_tier_budget.h).
// Host-only: the ladder, the accounting and the page-cap derivation are pure
// functions over integers, so they are testable without CUDA, a GPU or an
// artifact -- the same shape as tests/test_kv_tier_formats.cpp.
#include "product/kv_cold_tier_budget.h"

#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

namespace p = ninfer::product;

using ninfer::ColdPolicy;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

template <typename Fn>
bool rejects(const std::string& needle, Fn&& operation) {
    try {
        operation();
    } catch (const std::invalid_argument& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    } catch (...) { return false; }
    return false;
}

constexpr std::uint64_t kKiB = 1024ULL;
constexpr std::uint64_t kMiB = 1024ULL * kKiB;
constexpr std::uint64_t kGiB = 1024ULL * kMiB;

// The shipped defaults: 4 GiB host, 32 GiB disk, 100 KiB host page.
p::ColdTierBudget host_default(std::uint64_t page_bytes = 100ULL * kKiB) {
    p::ColdTierBudget budget = p::cold_tier_budget_from(ColdPolicy::Host, 4ULL * kGiB, 32ULL * kGiB);
    budget.host_page_bytes   = page_bytes;
    budget.disk_page_bytes   = 10ULL * kKiB;
    return budget;
}

void test_budget_derivation() {
    const p::ColdTierBudget host =
        p::cold_tier_budget_from(ColdPolicy::Host, 4ULL * kGiB, 32ULL * kGiB);
    check(host.host_tier_enabled && !host.disk_tier_enabled,
          "host policy enables exactly the host tier");
    check(host.host_bytes == 4ULL * kGiB && host.disk_bytes == 32ULL * kGiB,
          "both caps are carried even when only one tier is enabled");

    const p::ColdTierBudget disk =
        p::cold_tier_budget_from(ColdPolicy::Disk, 4ULL * kGiB, 32ULL * kGiB);
    check(!disk.host_tier_enabled && disk.disk_tier_enabled,
          "disk policy enables exactly the disk tier");

    for (const ColdPolicy policy : {ColdPolicy::None, ColdPolicy::Window}) {
        const p::ColdTierBudget none = p::cold_tier_budget_from(policy, 4ULL * kGiB, 32ULL * kGiB);
        check(!none.host_tier_enabled && !none.disk_tier_enabled,
              "policy without a byte budget enables no tier");
        const p::ColdTierDecision decision =
            p::admit_cold_page(none, p::ColdTierUsage{});
        check(decision.tier == p::ColdTier::StayHot,
              "a policy with no byte-budgeted tier never admits a page");
    }
}

void test_page_capacity() {
    check(p::cold_tier_page_capacity(4ULL * kGiB, 100ULL * kKiB) == 41943U,
          "4 GiB / 100 KiB floors to 41943 whole pages");
    check(p::cold_tier_page_capacity(100ULL * kKiB, 100ULL * kKiB) == 1U,
          "one page exactly fills its cap");
    check(p::cold_tier_page_capacity(100ULL * kKiB - 1, 100ULL * kKiB) == 0U,
          "a cap one byte short of a page admits nothing");
    check(p::cold_tier_page_capacity(0, 100ULL * kKiB) == 0U, "a zero cap admits nothing");
    check(p::cold_tier_page_capacity(4ULL * kGiB, 0) == 0U,
          "an unknown page stride admits nothing rather than dividing by zero");
    check(p::cold_tier_page_capacity(~0ULL, 1) == 0xFFFFFFFFU,
          "an astronomically large cap saturates instead of overflowing");
}

void test_ladder_host_first() {
    const p::ColdTierBudget budget = host_default();
    p::ColdTierBudget both         = budget;
    both.disk_tier_enabled         = true;
    both.device_cold_pages         = 4096;

    p::ColdTierUsage usage;
    const p::ColdTierDecision fresh = p::admit_cold_page(both, usage);
    check(fresh.tier == p::ColdTier::Host, "an empty ladder admits into the host tier first");
    p::reserve_cold_page(usage, fresh.tier);
    check(usage.host_pages == 1 && usage.disk_pages == 0 && usage.device_cold_slots == 0,
          "a host admission consumes host pages only, never a device cold slot");

    // Fill the host tier to exactly its cap.
    const std::uint32_t host_capacity =
        p::cold_tier_page_capacity(both.host_bytes, both.host_page_bytes);
    while (usage.host_pages < host_capacity) {
        const p::ColdTierDecision decision = p::admit_cold_page(both, usage);
        check(decision.tier == p::ColdTier::Host, "the host tier keeps admitting below its cap");
        p::reserve_cold_page(usage, decision.tier);
    }
    check(usage.host_pages == host_capacity, "the host tier fills to exactly its cap");

    const p::ColdTierDecision overflow = p::admit_cold_page(both, usage);
    check(overflow.tier == p::ColdTier::Disk, "one page past the host cap goes to the disk tier");
    p::reserve_cold_page(usage, overflow.tier);
    check(usage.host_pages == host_capacity, "the disk overflow does not push past the host cap");
    check(usage.disk_pages == 1 && usage.device_cold_slots == 1,
          "a disk admission consumes a disk page and a device cold slot");
}

void test_ladder_disk_exhaustion() {
    p::ColdTierBudget both = host_default();
    both.disk_tier_enabled = true;
    both.host_bytes        = 1ULL * kMiB; // 10 host pages: reachable in the test
    both.disk_bytes        = 100ULL * kKiB;
    both.device_cold_pages = 3;

    p::ColdTierUsage usage;
    std::uint32_t admitted_host = 0;
    std::uint32_t admitted_disk = 0;
    std::uint32_t stayed_hot    = 0;
    for (int i = 0; i < 64; ++i) {
        const p::ColdTierDecision decision = p::admit_cold_page(both, usage);
        switch (decision.tier) {
        case p::ColdTier::Host: ++admitted_host; break;
        case p::ColdTier::Disk: ++admitted_disk; break;
        case p::ColdTier::StayHot: ++stayed_hot; break;
        }
        p::reserve_cold_page(usage, decision.tier);
    }
    check(admitted_host == 10, "the host tier admitted exactly cold_host_bytes / page_stride pages");
    check(admitted_disk == 3,
          "the disk tier stopped at the smaller of its byte cap and the device slot pool");
    check(stayed_hot == 64 - 13, "everything past both caps stays hot");
    check(usage.host_pages == 10 && usage.disk_pages == 3 && usage.device_cold_slots == 3,
          "the saturating ladder never exceeds either cap or the device pool");
    check(p::admit_cold_page(both, usage).tier == p::ColdTier::StayHot,
          "a saturated ladder stays hot and reports it");

    // Freeing one disk slot admits exactly one more page, and frees nothing else.
    p::release_cold_page(usage, p::ColdTier::Disk);
    const p::ColdTierDecision refill = p::admit_cold_page(both, usage);
    check(refill.tier == p::ColdTier::Disk, "a released disk slot is reusable");
    p::reserve_cold_page(usage, refill.tier);

    // Freeing a host page changes the answer back to the host tier.
    p::release_cold_page(usage, p::ColdTier::Host);
    check(p::admit_cold_page(both, usage).tier == p::ColdTier::Host,
          "releasing a host page re-opens the host tier");
}

void test_host_only_has_no_overflow_tier() {
    p::ColdTierBudget host = host_default();
    host.host_bytes        = 100ULL * kKiB; // exactly one page
    p::ColdTierUsage usage;
    check(p::admit_cold_page(host, usage).tier == p::ColdTier::Host, "the single host page fits");
    p::reserve_cold_page(usage, p::ColdTier::Host);
    const p::ColdTierDecision decision = p::admit_cold_page(host, usage);
    check(decision.tier == p::ColdTier::StayHot && decision.reason != nullptr,
          "a full host-only ladder stays hot and names the reason");
    check(std::string(decision.reason).find("no disk tier") != std::string::npos,
          "the hot decision explains that no overflow tier is configured");
    check(usage.host_pages == 1, "the refused page consumed no host bytes");
}

void test_accounting_round_trip() {
    p::ColdTierUsage usage;
    p::reserve_cold_page(usage, p::ColdTier::Host);
    p::reserve_cold_page(usage, p::ColdTier::Disk);
    p::release_cold_page(usage, p::ColdTier::Host);
    p::release_cold_page(usage, p::ColdTier::Disk);
    check(usage.host_pages == 0 && usage.disk_pages == 0 && usage.device_cold_slots == 0,
          "reserve/release are symmetric");
    // Underflow must not wrap a running server's accounting.
    p::release_cold_page(usage, p::ColdTier::Host);
    p::release_cold_page(usage, p::ColdTier::Disk);
    p::release_cold_page(usage, p::ColdTier::StayHot);
    check(usage.host_pages == 0 && usage.disk_pages == 0 && usage.device_cold_slots == 0,
          "releasing an empty tier clamps instead of wrapping");
}

void test_demotion_victim_is_the_oldest_page() {
    const std::vector<std::uint32_t> order{7, 9, 11};
    const std::optional<std::uint32_t> victim = p::select_host_demotion_victim(order);
    check(victim.has_value() && *victim == 7, "the oldest host-resident cold page is demoted");
    check(!p::select_host_demotion_victim(std::span<const std::uint32_t>()).has_value(),
          "an empty host tier has no demotion victim");
}

void test_counters_and_report() {
    p::ColdTierCounters counters;
    check(!counters.any(), "a fresh counter reports nothing");
    counters.host_admissions = 3;
    counters.disk_admissions = 2;
    counters.stays_hot       = 1;
    counters.releases        = 4;
    check(counters.any(), "any admission makes the counters reportable");
    const std::string line = counters.describe();
    for (const std::string needle : {"host=3", "disk=2", "stay_hot=1", "released=4"}) {
        check(line.find(needle) != std::string::npos,
              "the counter line carries " + needle);
    }
}

void test_validation() {
    const p::ColdTierBudget shipped =
        p::cold_tier_budget_from(ColdPolicy::Host, 4ULL * kGiB, 32ULL * kGiB);
    bool threw = false;
    try {
        p::validate_cold_tier_budget(shipped);
    } catch (...) { threw = true; }
    check(!threw, "the shipped defaults validate");

    threw = false;
    try {
        p::validate_cold_tier_budget(p::cold_tier_budget_from(ColdPolicy::Window, 4ULL * kGiB,
                                                             32ULL * kGiB));
    } catch (...) { threw = true; }
    check(!threw, "the window policy has no byte budget to validate");

    check(rejects("can never admit", [] {
              p::validate_cold_tier_budget(
                  p::cold_tier_budget_from(ColdPolicy::Host, 0, 32ULL * kGiB));
          }),
          "host policy with a zero host cap is a loud contradiction");

    check(rejects("can never admit", [] {
              p::validate_cold_tier_budget(
                  p::cold_tier_budget_from(ColdPolicy::Disk, 4ULL * kGiB, 0));
          }),
          "disk policy with a zero spill cap is a loud contradiction");

    // A zero cap under a policy that does not use it stays legal (no regression
    // for `--cold-host-bytes 0 --cold-policy window`).
    threw = false;
    try {
        p::validate_cold_tier_budget(
            p::cold_tier_budget_from(ColdPolicy::Disk, 0, 32ULL * kGiB));
    } catch (...) { threw = true; }
    check(!threw, "a zero host cap under the disk policy is not a contradiction");

    const std::string summary = p::describe_cold_tier_budget(shipped, 41943, 3355443);
    check(summary.find("host tier on") != std::string::npos &&
              summary.find("disk tier off") != std::string::npos,
          "the startup summary names which tiers are live");
}

} // namespace

int main() {
    test_budget_derivation();
    test_page_capacity();
    test_ladder_host_first();
    test_ladder_disk_exhaustion();
    test_host_only_has_no_overflow_tier();
    test_accounting_round_trip();
    test_demotion_victim_is_the_oldest_page();
    test_counters_and_report();
    test_validation();

    if (failures == 0) { std::cout << "kv_cold_tier_budget_test: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
