#pragma once

// Cold-KV tier budgets and admission (the CONSUMER-side semantics of
// --cold-host-bytes / --cold-disk-bytes).
//
// Both byte budgets are carried all the way into the engine and never read
// again (program_impl.h:772-774 copies them into the cold-pool state and
// ProgramImplCore never consults them: `cold_host_bytes` has no other
// reference in src/, and `cold_disk_bytes` neither). So today neither cap
// restricts anything. This header owns the missing meaning:
//
//   * what a cap counts (whole pages of a fixed stride -- so page accounting
//     and byte accounting are the same accounting and the cap can never be
//     exceeded by rounding);
//   * the admission ladder (host tier first, disk tier as its overflow, stay
//     hot as the last resort);
//   * the accounting moves (reserve/release/demote) and the counters that make
//     an unmet budget visible instead of silent.
//
// It is deliberately pure: the consumer that actually moves a page calls these
// functions and therefore cannot invent its own notion of "full", and the
// invariants are unit-testable with plain g++ (tests/test_kv_cold_tier_budget.cpp),
// no CUDA, no engine headers -- the same shape as serve/kv_cold_policy.h.
//
// Ladder for one cold page:
//
//   1. Host (tier 1, --cold-host-bytes): pinned host memory. A host-resident
//      cold page occupies NO device cold slot and is restored by one pinned H2D
//      copy (no rANS decode, no staging slot), so it is the cheapest tier to
//      leave, hence first.
//   2. Disk (tier 2, --cold-disk-bytes): the SSD spill. Tier 2 additionally
//      needs one DEVICE cold slot per spilled page (its working set, sized by
//      --max-cold-pages), so admission checks the disk cap AND that pool.
//   3. Stay hot: the page keeps its device replica. Counted, reported, never
//      thrown -- an unmet budget is a capacity event, not a program error;
//      throwing would turn memory pressure into a dead server. The only hard
//      errors are startup contradictions, see validate_cold_tier_budget().
//
// Host-first matches the operator's phrasing: host offload is the bounded fast
// tier, and the excess goes to SSD.

#include "ninfer/types.h"

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

enum class ColdTier : std::uint8_t {
    Host,
    Disk,
    StayHot,
};

[[nodiscard]] inline std::string_view cold_tier_name(ColdTier tier) noexcept {
    switch (tier) {
    case ColdTier::Host: return "host";
    case ColdTier::Disk: return "disk";
    case ColdTier::StayHot: return "hot";
    }
    return "?";
}

// Fixed per-page cost of each tier. Tier 1 uses HostKVPageLayout::page_stride
// for the page store in question; tier 2 uses the per-page spill stride (the
// sum of the layer slot strides); the device cold slot pool is counted in
// pages because that is how the layout reserves it.
struct ColdTierBudget {
    std::uint64_t host_bytes        = 0; // --cold-host-bytes (tier 1 cap)
    std::uint64_t disk_bytes        = 0; // --cold-disk-bytes (tier 2 cap)
    std::uint64_t host_page_bytes   = 0;
    std::uint64_t disk_page_bytes   = 0;
    std::uint32_t device_cold_pages = 0; // --max-cold-pages (tier 2 working set)
    // Resolved policy flags. ColdPolicy::Window has no byte budget at all: its
    // cold pool is device-resident and capped in pages, so both flags stay
    // false and every page that asks for a tier stays hot (tier 3).
    bool host_tier_enabled = false;
    bool disk_tier_enabled = false;
};

// Derives the tier flags from the resolved cold policy. Kept separate from the
// budget struct so the pure ladder never has to know about ColdPolicy.
[[nodiscard]] inline ColdTierBudget
cold_tier_budget_from(ColdPolicy policy, std::uint64_t cold_host_bytes,
                      std::uint64_t cold_disk_bytes) noexcept {
    ColdTierBudget budget;
    budget.host_bytes        = cold_host_bytes;
    budget.disk_bytes        = cold_disk_bytes;
    budget.host_tier_enabled = policy == ColdPolicy::Host;
    budget.disk_tier_enabled = policy == ColdPolicy::Disk;
    return budget;
}

// A cap expressed in whole pages. A cap smaller than one page admits nothing
// rather than rounding up: a limit must never be exceeded, and "--cold-host-bytes
// 4096" on a 100 KiB page really does mean "no page fits".
[[nodiscard]] inline std::uint32_t cold_tier_page_capacity(std::uint64_t bytes,
                                                           std::uint64_t page_bytes) noexcept {
    if (page_bytes == 0) { return 0; }
    const std::uint64_t pages = bytes / page_bytes;
    return pages > 0xFFFFFFFFULL ? 0xFFFFFFFFU : static_cast<std::uint32_t>(pages);
}

struct ColdTierUsage {
    std::uint32_t host_pages        = 0;
    std::uint32_t disk_pages        = 0;
    std::uint32_t device_cold_slots = 0;
};

struct ColdTierDecision {
    ColdTier tier      = ColdTier::StayHot;
    const char* reason = "";
};

// The ladder. Order and per-tier resources ARE the semantics of the two caps.
[[nodiscard]] inline ColdTierDecision admit_cold_page(const ColdTierBudget& budget,
                                                      const ColdTierUsage& usage) noexcept {
    const std::uint32_t host_capacity =
        cold_tier_page_capacity(budget.host_bytes, budget.host_page_bytes);
    if (budget.host_tier_enabled && usage.host_pages < host_capacity) {
        return ColdTierDecision{ColdTier::Host, "host cap has room"};
    }
    const std::uint32_t disk_capacity =
        cold_tier_page_capacity(budget.disk_bytes, budget.disk_page_bytes);
    if (budget.disk_tier_enabled && usage.disk_pages < disk_capacity &&
        usage.device_cold_slots < budget.device_cold_pages) {
        return ColdTierDecision{ColdTier::Disk, "disk cap and cold-slot pool have room"};
    }
    if (budget.host_tier_enabled && budget.disk_tier_enabled) {
        return ColdTierDecision{ColdTier::StayHot, "host cap full, disk cap or slots full"};
    }
    if (budget.host_tier_enabled) {
        return ColdTierDecision{ColdTier::StayHot, "host cap full, no disk tier configured"};
    }
    if (budget.disk_tier_enabled) {
        return ColdTierDecision{ColdTier::StayHot, "disk cap or cold-slot pool full"};
    }
    return ColdTierDecision{ColdTier::StayHot, "no byte-budgeted cold tier for this policy"};
}

// Accounting moves. reserve_cold_page() must be called exactly once for the page
// that was actually placed; release_cold_page() exactly once when it leaves the
// tier (restore or demotion). Both are idempotent-safe against underflow, since
// an accounting slip must not corrupt a running server.
inline void reserve_cold_page(ColdTierUsage& usage, ColdTier tier) noexcept {
    switch (tier) {
    case ColdTier::Host: ++usage.host_pages; break;
    case ColdTier::Disk:
        ++usage.disk_pages;
        ++usage.device_cold_slots;
        break;
    case ColdTier::StayHot: break;
    }
}

inline void release_cold_page(ColdTierUsage& usage, ColdTier tier) noexcept {
    switch (tier) {
    case ColdTier::Host:
        if (usage.host_pages != 0) { --usage.host_pages; }
        break;
    case ColdTier::Disk:
        if (usage.disk_pages != 0) { --usage.disk_pages; }
        if (usage.device_cold_slots != 0) { --usage.device_cold_slots; }
        break;
    case ColdTier::StayHot: break;
    }
}

// When the host cap is saturated the newcomer can still take a host slot by
// demoting the OLDEST host-resident cold page to tier 2. Cold-page age is
// monotone in distance behind the decode frontier, and the pages a rewrite /
// fork / resume touches are the newest ones, so the oldest page is the one
// least likely to be restored. `host_page_order` is the host tier's insertion
// order (ascending cold page index); its front is the victim.
[[nodiscard]] inline std::optional<std::uint32_t>
select_host_demotion_victim(std::span<const std::uint32_t> host_page_order) noexcept {
    if (host_page_order.empty()) { return std::nullopt; }
    return host_page_order.front();
}

struct ColdTierCounters {
    // Exactly the events a running consumer can observe. Restores are NOT here on
    // purpose: a Host-tier page comes back through the engine's materialization
    // path, which the tier never sees, so counting them here would be a lie.
    std::uint64_t host_admissions = 0;
    std::uint64_t disk_admissions = 0;
    std::uint64_t stays_hot       = 0;
    std::uint64_t releases        = 0;

    [[nodiscard]] bool any() const noexcept {
        return host_admissions != 0 || disk_admissions != 0 || stays_hot != 0 || releases != 0;
    }

    // One line, printed at most once per round by the caller. A cap that is
    // being hit has to be visible: "the excess goes to SSD" is only a working
    // contract if the excess and the saturations are reported.
    [[nodiscard]] std::string describe() const {
        return "[cold-tier] host=" + std::to_string(host_admissions) + " disk=" +
               std::to_string(disk_admissions) + " stay_hot=" + std::to_string(stays_hot) +
               " released=" + std::to_string(releases);
    }
};

// Human-readable budget summary for the startup banner: the effective tier
// capacities in pages, derived from the byte caps and the caller's strides.
[[nodiscard]] inline std::string describe_cold_tier_budget(const ColdTierBudget& budget,
                                                           std::uint32_t host_page_capacity,
                                                           std::uint32_t disk_page_capacity) {
    std::string out = "[cold-tier] host tier ";
    out += budget.host_tier_enabled ? "on" : "off";
    out += " (" + std::to_string(budget.host_bytes) + " B = " +
           std::to_string(host_page_capacity) + " pages), disk tier ";
    out += budget.disk_tier_enabled ? "on" : "off";
    out += " (" + std::to_string(budget.disk_bytes) + " B = " +
           std::to_string(disk_page_capacity) + " pages, " +
           std::to_string(budget.device_cold_pages) + " device slots)";
    return out;
}

// Startup contradictions only. A tier the operator explicitly asked for whose
// cap cannot admit a single page is not a capacity event, it is a configuration
// that cannot do what it says -- the same rule layouts_impl.h already applies to
// --max-cold-pages + host ("make the contradiction loud instead", its comment
// at effective_cold_pages), and the same rule --weight-host-bytes follows.
// Everything that depends on the runtime strides (a page larger than the whole
// cap) is left to the consumer, which reports it instead of throwing.
inline void validate_cold_tier_budget(const ColdTierBudget& budget) {
    if (budget.host_tier_enabled && budget.host_bytes == 0) {
        throw std::invalid_argument(
            "--cold-policy host with --cold-host-bytes 0 can never admit a cold page; "
            "set a positive host budget or drop the policy");
    }
    if (budget.disk_tier_enabled && budget.disk_bytes == 0) {
        throw std::invalid_argument(
            "--cold-policy disk with --cold-disk-bytes 0 can never admit a cold page; "
            "set a positive spill budget or drop the policy");
    }
}

} // namespace ninfer::product
