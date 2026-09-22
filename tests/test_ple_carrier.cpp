// tests/test_ple_carrier.cpp — host-only acceptance for the n-gram carrier layer
// (src/product/ple_sidecar_carrier.h). No GPU, no CUDA, no engine: the header is pure
// budget/policy arithmetic, like tests/test_kv_cold_tier_budget.cpp.
//
// The point of these checks is the user's rule and its two failure modes:
//   rule : "检测到 flashnext 就默认 ngram 丢 ssd，但到时候 gui 允许不丢 ssd 保留在内存里"
//   fail 1: an unservable DEFAULT must be recorded and reported, never silent
//   fail 2: an unservable EXPLICIT override must throw at startup, never be honoured
//           silently (an operator flag that lies is worse than a refusal)
//
// Load-bearing controls, one per check:
//   | # | check                                              | control that makes it RED |
//   |---|----------------------------------------------------|---------------------------|
//   | 1 | flashnext => intended SSD, recorded downgrade      | drop the servability test -> downgraded stays false while resolver=memory (silent) |
//   | 2 | explicit memory override wins over the SSD default  | ignore explicit_override -> intended becomes SSD |
//   | 3 | no flashnext => memory, no downgrade               | make the default SSD -> downgraded flips |
//   | 4 | explicit SSD override that cannot be served THROWS | drop the override branch in validate_startup -> no throw |
//   | 5 | unservable DEFAULT does NOT throw (recorded, not contradictory) | make validate_startup throw on any downgrade -> check 5 red |
//   | 6 | memory-resident table larger than the pin budget throws | drop the budget verdict -> no throw |
//   | 7 | the SSD hook is explicitly absent and NAMED       | flip ple_ssd_backend_linked() -> 1 stops downgrading (this check records today's state) |

#include "product/ple_sidecar_carrier.h"

#include <cstdio>
#include <functional>
#include <string>

namespace {

using ninfer::product::PleCarrierBudget;
using ninfer::product::PleCarrierClass;
using ninfer::product::PleCarrierRequest;
using ninfer::product::PleCarrierResolution;

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    } else {
        std::printf("ok: %s\n", what);
    }
}

bool throws_invalid_argument(const std::function<void()>& body) {
    try {
        body();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// The real 95 GiB PLE table: 320,001,536 rows x 320 B.
constexpr std::uint64_t kTableBytes = 320001536ULL * 320ULL;

} // namespace

int main() {
    using namespace ninfer::product;

    // ---- 7. The hook's current state is explicit, and it is what check 1 rests on.
    {
        expect(!ple_ssd_backend_linked(), "the SSD carrier backend is NOT linked in this tree");
        expect(!ple_carrier_class_servable(PleCarrierClass::SsdPaged),
               "SsdPaged is therefore not servable (the 判据 follows the hook)");
        expect(ple_carrier_class_servable(PleCarrierClass::MemoryResident),
               "MemoryResident is servable");
        const std::string requirement = ple_ssd_backend_requirement();
        std::printf("ssd requirement: %s\n", requirement.c_str());
        expect(requirement.find("ssd_bytes") != std::string::npos,
               "the SSD requirement NAMES the missing budget, not just 'unsupported'");
    }

    // ---- 1. THE RULE: FlashNext detected => the default intends SSD.
    {
        PleCarrierRequest req;
        req.flashnext_detected = true;
        const PleCarrierResolution res = ple_carrier_resolve(req);
        std::printf("%s\n", res.to_line().c_str());
        expect(res.intended == PleCarrierClass::SsdPaged,
               "FlashNext detection selects SSD as the DEFAULT (the user's rule, part b)");
        // ...and because the backend is absent, that intent is recorded, not faked.
        expect(res.resolved == PleCarrierClass::MemoryResident && res.downgraded,
               "an unservable default resolves to memory WITH a recorded downgrade");
        expect(!res.reason.empty(), "the downgrade carries a reason (never silent)");
        expect(!res.served(), "served() reports the downgrade to its caller");
    }

    // ---- 2. THE OVERRIDE: "gui 允许不丢 ssd 保留在内存里" — memory wins, no downgrade.
    {
        PleCarrierRequest req;
        req.flashnext_detected = true;
        req.explicit_override  = PleCarrierClass::MemoryResident;
        const PleCarrierResolution res = ple_carrier_resolve(req);
        expect(res.intended == PleCarrierClass::MemoryResident && res.resolved == res.intended,
               "an explicit memory override beats the FlashNext SSD default");
        expect(!res.downgraded,
               "forcing memory is not a downgrade (nothing was demoted; that is the point of the override)");
    }

    // ---- 3. No FlashNext => memory, untouched.
    {
        PleCarrierRequest req;
        const PleCarrierResolution res = ple_carrier_resolve(req);
        expect(res.intended == PleCarrierClass::MemoryResident && !res.downgraded,
               "a non-FlashNext artifact stays memory-resident with no downgrade");
    }

    // ---- 4 + 5. Startup validation: explicit-but-unservable throws; default does not.
    //
    // The budget here is the DEFAULT pinned page-cache budget (512 MiB). Note what that
    // means for a "memory-resident" resolution: PleCarrierClass::MemoryResident means the
    // WHOLE table is pinned, so resolving to memory for the 95 GiB table also requires a
    // pin budget that can hold it. The two contradictions are checked separately below,
    // because an operator who forces memory must supply that budget explicitly -- the
    // page-cache default is a different thing and cannot silently become a whole-table pin.
    {
        PleCarrierRequest explicit_ssd;
        explicit_ssd.flashnext_detected = true;
        explicit_ssd.explicit_override  = PleCarrierClass::SsdPaged;
        const PleCarrierBudget budget; // default page-cache budget
        expect(throws_invalid_argument([&] {
                   (void)ple_carrier_validate_startup(explicit_ssd, budget, kTableBytes);
               }),
               "an EXPLICIT SSD override this tree cannot serve is a startup contradiction (throws)");
    }
    {
        // A default (nobody asked by name) that cannot be served must be RECORDED, not
        // thrown -- the carrier half of the cold-tier doctrine. The budget must be able to
        // hold the table here, otherwise the budget check (checked separately) fires first
        // and this would be testing the wrong contradiction.
        PleCarrierBudget big;
        big.pinned_bytes = kTableBytes;
        PleCarrierRequest default_ssd;
        default_ssd.flashnext_detected = true;
        const PleCarrierResolution res =
            ple_carrier_validate_startup(default_ssd, big, kTableBytes);
        std::printf("unservable default -> %s\n", res.to_line().c_str());
        expect(res.downgraded && res.resolved == PleCarrierClass::MemoryResident,
               "an unservable DEFAULT is a recorded downgrade, NOT a startup contradiction");
    }
    {
        // And the budget half, on the same unservable default: with the page-cache default
        // budget the memory resolution cannot hold the table, which IS a startup
        // contradiction (it would otherwise silently pin nothing / fault forever).
        PleCarrierBudget page_cache; // 512 MiB
        PleCarrierRequest default_ssd;
        default_ssd.flashnext_detected = true;
        expect(throws_invalid_argument([&] {
                   (void)ple_carrier_validate_startup(default_ssd, page_cache, kTableBytes);
               }),
               "the same default WITH the page-cache budget is a contradiction: memory-resident means pinning the whole table");
    }

    // ---- 6. Budget arithmetic on the memory side (the implemented half).
    {
        PleCarrierBudget budget;
        budget.pinned_bytes = 512ULL << 20;
        PleCarrierRequest req;
        const PleCarrierResolution res = ple_carrier_resolve(req);
        const auto verdict = ple_carrier_budget_verdict(res, budget, kTableBytes);
        expect(verdict.has_value(),
               "the whole 95 GiB table does NOT fit the 512 MiB pinned page-cache budget");
        if (verdict) { std::printf("budget verdict: %s\n", verdict->c_str()); }
        expect(throws_invalid_argument([&] {
                   (void)ple_carrier_validate_startup(req, budget, kTableBytes);
               }),
               "pinning a table larger than the pin budget throws at startup");

        // A budget that does cover the table is accepted (control for the check above).
        PleCarrierBudget big;
        big.pinned_bytes = kTableBytes;
        expect(!ple_carrier_budget_verdict(res, big, kTableBytes).has_value(),
               "a budget that covers the table yields no verdict (the check is not vacuous)");
    }

    std::printf(failures == 0 ? "PLE_CARRIER_PASS\n" : "PLE_CARRIER_FAIL\n");
    return failures == 0 ? 0 : 1;
}
