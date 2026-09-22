// RECALLSPAN: the admission test for the paging pre-allocation wiring.
//
// WHY THIS FILE EXISTS, AND WHAT IT WOULD HAVE CAUGHT. product/kv_paging_preallocation.h landed a
// correct contract (`paging_explicit_capacity_refusal`) with ZERO non-test consumers, so it could
// not fire on anything -- the defect PAGEPREALLOC's contract exists to prevent (a page with neither
// a device replica nor a cold tier, silently counted as stored) ran anyway, on D1's argv:
// `--max-context 1000000 --kv-capacity 22016 --cold-policy disk --max-cold-pages 256`.
//
// The wiring lives at layouts_impl.h:1808+ (the pool-sizing site). It cannot supply
// `cold_page_record_bytes` -- the BYTES PER COLD PAGE ACROSS ALL SLOT-BEARING LAYERS -- because the
// cold-slot tensors whose nb[3] carries it are declared by the engine, later
// (decoder_state.cpp:931-933). It therefore passes 0 = "unknown" and consults the guard ONLY when
// the verdict provably cannot depend on that unknown, i.e. when the device cold-slot pool -- the
// backing store's hard ceiling under DeviceWindow/Disk/HostThenDisk -- is already below the
// requirement. Otherwise it SKIPS and prints `[paging-prealloc] NOT EVALUATED`, rather than let a
// stand-in force a refusal.
//
// SO THE ARMS BELOW PIN THE TWO PROPERTIES THE WIRING'S SOUNDNESS RESTS ON, and the third arm is
// the one that would have gone RED if this rule were dropped:
//
//   A. On D1's own argv the guard refuses, with the engine's own arithmetic -- and the numbers
//      `resident` / the "at most N tokens" hint in the message are the ones the REBUILT ENGINE
//      PRINTED at 12:29:14 on 2026-09-18 (runs/R1_gate_ON/stderr.txt), not numbers chosen here.
//   B. The verdict on D1's argv is INVARIANT over the byte-per-cold-page, so passing 0 cannot
//      change it: under Disk the backing capacity is min(disk pages, device cold slots) and the
//      slot pool caps it at 256 pages against a requirement of 15,527.
//   C. When the ceiling is NOT below the requirement the unknown DOES flip the verdict -- there a
//      pass-through would produce a FALSE REFUSAL of a run whose disk would in fact have sufficed.
//      This arm asserts the flip, which is the licence for the skip branch.
//
// Host-only: the header is std-only (no CUDA, no ninfer/types.h), so plain g++ builds and runs it.

#include "product/kv_paging_preallocation.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

namespace p = ninfer::product;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

void check_eq(std::uint64_t got, std::uint64_t want, const std::string& what) {
    if (got == want) { return; }
    std::cerr << "FAIL: " << what << ": got " << got << ", want " << want << '\n';
    ++failures;
}

void check_mentions(const std::string& needle, const std::string& haystack, const std::string& what) {
    if (haystack.find(needle) != std::string::npos) { return; }
    std::cerr << "FAIL: " << what << ": '" << needle << "' not in '" << haystack << "'\n";
    ++failures;
}

// D1's argv, as the wiring supplies it: the fields the site can fill, and 0 for the one it cannot.
p::PagingPreallocationInputs d1_inputs(std::uint64_t page_record_bytes) {
    p::PagingPreallocationInputs in;
    in.max_context_tokens       = 1000000ULL;   // --max-context 1000000
    in.page_tokens              = 64;           // kPagedKVPageSize
    in.max_concurrency          = 1;
    in.cold_keep_tokens         = 128;          // --cold-keep-tokens 128
    in.prefill_chunk            = 3072;         // the value the ENGINE resolved (see arm A)
    in.unload_watermark_pages   = 48;           // --kv-unload-watermark-pages 48 (explicit, resolved)
    in.medium                   = p::PagingColdMedium::Disk;  // --cold-policy disk
    in.max_cold_pages           = 256;          // --max-cold-pages 256
    in.cold_disk_bytes          = 268435456ULL; // --cold-disk-bytes 268435456
    in.cold_host_bytes          = 0;
    in.cold_page_record_bytes   = page_record_bytes;
    in.host_page_bytes          = 0;
    return in;
}

constexpr std::uint32_t kD1RequestedPages = 344;  // --kv-capacity 22016 -> ceil(22016 / 64)

// ARM A -- the refusal D1's argv gets, with the numbers the ENGINE printed.
void arm_A_the_refusal_fires_on_d1_argv_with_the_engines_own_numbers() {
    const p::PagingPreallocationInputs in = d1_inputs(0);
    check_eq(p::declared_context_floor_pages(in.max_context_tokens, in.page_tokens, in.max_concurrency),
             15625, "declared context pages");
    // 98 = keep(128/64 = 2) + round(3072/64 = 48) + watermark(48). The engine printed
    // "declare a context of at most 6272 tokens" = 98 x 64, which is where this is from.
    check_eq(p::paging_resident_pages(in), 98, "resident pages (engine's own resolution)");
    check_eq(p::paging_required_backing_pages(in), 15625 - 98, "required backing pages");
    check_eq(p::paging_device_cold_slot_pages(in.cold_keep_tokens, in.page_tokens, in.medium,
                                             in.max_cold_pages),
             256, "device cold-slot pool = --max-cold-pages");
    const std::string refusal = p::paging_explicit_capacity_refusal(in, kD1RequestedPages);
    check(!refusal.empty(), "D1's argv must be refused: 344 pages cannot cover 15625");
    check_mentions("--kv-capacity 22016", refusal, "the refusal names the request in tokens");
    check_mentions("344 pages of 64 tokens", refusal, "and in pages");
    check_mentions("15625", refusal, "and the declared context");
    check_mentions("short by 15281 pages", refusal, "and the page shortfall it computed");
    check_mentions("6272", refusal, "and the context the resident set could actually hold");
    // The default path must be untouched: an unarmed run cannot be refused by this header at all.
    p::PagingPreallocationInputs unarmed = in;
    unarmed.medium = p::PagingColdMedium::None;
    check(p::paging_preallocation_refusal(unarmed).empty(),
          "medium == None must never refuse -- an unarmed run is bit-identical to before");
    std::cout << "  armA " << refusal << '\n';
}

// ARM B -- on D1's argv the verdict does NOT depend on the byte-per-cold-page.
void arm_B_the_verdict_is_invariant_over_the_unknown_on_d1_argv() {
    const std::uint64_t required = p::paging_required_backing_pages(d1_inputs(0));
    const std::uint64_t kOneLayerStride = 9536ULL;   // NOT a page -- the 128x unit defect
    const std::uint64_t kTruePage = 1220608ULL;      // 16 layers x 4 kv_heads x 2 x 9536 B
    const std::uint64_t page_choices[5] = {0U, 9232U, kOneLayerStride, kTruePage, (1ULL << 40)};
    for (const std::uint64_t page : page_choices) {
        const p::PagingPreallocationInputs in = d1_inputs(page);
        const std::uint64_t capacity = p::paging_backing_capacity_pages(in);
        check(capacity <= in.max_cold_pages, "under Disk the slot pool caps the backing capacity");
        check(capacity < required, "capacity stays below the requirement for every byte-per-page");
        check(!p::paging_explicit_capacity_refusal(in, kD1RequestedPages).empty(),
              "so the refusal holds for every byte-per-page -- including the unknown (0)");
    }
    std::cout << "  armB invariant over {0, 9232, 9536, 1220608, 2^40}, required=" << required << '\n';
}

// ARM C -- and here the unknown DOES decide, which is exactly where the wiring must skip.
void arm_C_the_unknown_decides_when_the_ceiling_is_not_below_the_requirement() {
    p::PagingPreallocationInputs in = d1_inputs(0);
    in.max_cold_pages    = 20000;              // a pool big enough to be the ceiling
    in.cold_disk_bytes   = 20ULL << 30;        // 21.47 GB of spill
    const std::uint32_t required = p::paging_required_backing_pages(in);
    const std::uint32_t ceiling  = p::paging_device_cold_slot_pages(
        in.cold_keep_tokens, in.page_tokens, in.medium, in.max_cold_pages);
    check(!(ceiling < required), "this is the NOT-decidable shape: the ceiling is not below it");
    // With the byte-per-page unknown the wiring would refuse...
    check(!p::paging_explicit_capacity_refusal(d1_inputs(0)  , kD1RequestedPages).empty(),
          "unknown byte-per-page -> refuses (capacity reads 0 pages)");
    // ...but the truth is that this disk DOES hold the requirement, so refusing would be FALSE.
    p::PagingPreallocationInputs known = in;
    known.cold_page_record_bytes = 1220608ULL;
    const std::uint32_t known_capacity = p::paging_backing_capacity_pages(known);
    check(known_capacity >= required, "the real disk capacity does cover the requirement");
    check(p::paging_explicit_capacity_refusal(known, kD1RequestedPages).empty(),
          "with the true byte-per-page this run is ADMITTED -- so a pass-through would be a FALSE "
          "refusal, and the wiring's skip branch is load-bearing");
    std::cout << "  armC ceiling=" << ceiling << " required=" << required
              << " known_capacity=" << known_capacity << '\n';
}

} // namespace

int main() {
    arm_A_the_refusal_fires_on_d1_argv_with_the_engines_own_numbers();
    arm_B_the_verdict_is_invariant_over_the_unknown_on_d1_argv();
    arm_C_the_unknown_decides_when_the_ceiling_is_not_below_the_requirement();

    if (failures == 0) {
        std::cout << "ok\n";
        return 0;
    }
    std::cerr << failures << " failure(s)\n";
    return 1;
}
