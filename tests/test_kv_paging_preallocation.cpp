// Unit test for product/kv_paging_preallocation.h -- the change that sizes the device Main-KV
// page pool from the PAGING WORKING SET (resident set + spill backing) instead of from the
// declared context.
//
// Host-only by construction: the header is std-only (no CUDA, no ninfer/types.h), so this runs with
// plain g++ like tests/test_kv_cold_tier_budget.cpp and tests/test_weight_residency.cpp.
//
// THE UNIT, STATED FIRST BECAUSE GETTING IT WRONG IS THE DEFECT THIS FILE NOW GUARDS AGAINST.
// A cold page's byte footprint is NOT one layer's stride. The engine's unit is the PAGE, summed
// over every slot-bearing layer, times kv_heads x 2:
//
//   decoder_state.cpp:931-933   cold_slots[layer] = add_tensor(U8, {stride, kv_heads, 2, cold_pages})
//   program_impl.h:12606        k_slot_base + slot * cold_slots.nb[3]     -- nb[3] is a BYTE stride
//   program_impl.h:12839-12855  turn_recall_page_bytes() = SUM over slot-bearing layers of nb[3]
//   program_impl.h:12071-12085  printed as `bytes_per_record` on the `[textcargo]` line
//   program_impl.h:12031        the engine's own prose: "1.18 MB/page vs 192-256 B/page"
//
// so `nb[3]` = stride_layer * kv_heads * 2 for that ONE layer, and a page costs the SUM. The first
// version of this test plugged in 9,536 B -- one layer's one head-plane -- and under-counted by the
// number of slot-bearing layers times kv_heads times 2 (128x on a 16-layer, 4-KV-head stack).
// Three engine PRINted readings are reproduced exactly below, which is how the unit was settled.
//
// WHAT IT PINS, and why each arm is here:
//
//  1. THE OLD EXPRESSION, EXACTLY. `medium == None` must return
//     max(ceil(max_context/page_tokens), max_concurrency) -- the floor at layouts_impl.h:1808-1815
//     and :985-992. If this arm ever needs editing, the change is no longer allocation-only.
//  2. THE WORKING-SET SIZE, with the arithmetic written out so a reader can check it.
//  3. THE REFUSAL -- the load-bearing part. Shrinking the pool removes the "stay hot" fallback
//     that makes product/kv_cold_tier_budget.h's ladder sound today (:35-38), so a page with
//     neither a replica nor a tier must THROW, naming the page range and the shortfall. Several
//     shapes, including the live LEVER1M control arm's own knobs
//     (--cold-policy disk --cold-disk-bytes 268435456 --max-cold-pages 256).
//  4. THE SPILL-CAPACITY CONTRACT at the largest context the mainline can declare -- IN THE
//     ENGINE'S UNIT -- including the honest conclusion that backing every declared page is not
//     honourable on this box, and what the honourable envelope actually is.
//  5. THE FLASH-NEXT NUMBERS on the same shared arithmetic: 1024 groups at max_context 262144,
//     and the missing-escape-hatch refusal for that target, which has no cold path at all
//     (text_decode.cpp:114-127, :294-297) and therefore may not shrink its pool at all.

#include "product/kv_paging_preallocation.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
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

void check_mentions(const std::string& needle, const std::string& text, const std::string& what) {
    if (text.find(needle) != std::string::npos) { return; }
    std::cerr << "FAIL: " << what << ": '" << text << "' does not mention '" << needle << "'\n";
    ++failures;
}

template <typename Fn>
void rejects(const std::string& needle, Fn&& operation) {
    try {
        operation();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(needle) == std::string::npos) {
            std::cerr << "FAIL: error '" << error.what() << "' does not mention '" << needle
                      << "'\n";
            ++failures;
        }
        return;
    }
    std::cerr << "FAIL: expected a throw mentioning '" << needle << "'\n";
    ++failures;
}

// The live LEVER1M control chain's own cold knobs, so arm 3 tests the configuration that is
// actually running on this box rather than a convenient one:
//   --cold-policy disk --cold-disk-bytes 268435456 --max-cold-pages 256
constexpr std::uint64_t kReferenceColdDiskBytes = 268435456ULL;  // 256 MiB
constexpr std::uint32_t kReferenceMaxColdPages = 256U;

// ONE LAYER's ONE HEAD-PLANE stride -- the building block of a cold record, NOT a page.
// 9,536 B is the 4-BIT NO-EXPANSION record (research/notes/TODO.md:6976-6980: 320 header +
// 32 x 256 streams + 1024). It is the record the nvfp4 layers were priced at when the three
// `bytes_per_record` readings below were taken, and it is NOT today's nvfp4 pool stride: that
// one is p::kKvColdPoolStrideBytes (9,632 B, 320 + 32 x 259 + 1024, the measured ceiling), and
// the two are named apart in product/kv_tier_formats.h (kKvColdRansRecordAt4BitBytes against
// kKvColdPoolStrideBytes). 9,232 B is the int8 slot's, as the engine printed it:
//   [kv-cold-tier] ... codec=int8 raw slot [pool stride 9232 B, fills 9232 B,
//                   resident plane 16896 B/head-page] -> SAVES 7664 B/head-page (45.4 %)
constexpr std::uint64_t kNvfp4RecordAt4BitBytes = 9536ULL;
constexpr std::uint64_t kInt8SlotBytes         = 9232ULL;
constexpr std::uint64_t kKvHeads               = 4ULL;  // qwen3_6_27b/impl/config.h:38
constexpr std::uint64_t kKvPlanes              = 2ULL;  // K and V

// BYTES PER COLD PAGE ACROSS ALL SLOT-BEARING LAYERS -- the engine's unit, via
// turn_recall_page_bytes() (program_impl.h:12839-12855) and the tensor declared
// {stride, kv_heads, 2, cold_pages} (decoder_state.cpp:931-933).
[[nodiscard]] constexpr std::uint64_t page_record_bytes(std::uint64_t i8_layers,
                                                        std::uint64_t nvfp4_layers) {
    return (i8_layers * kInt8SlotBytes + nvfp4_layers * kNvfp4RecordAt4BitBytes) * kKvHeads *
           kKvPlanes;
}
// The D1 arm's own `[textcargo] ... bytes_per_record=1181696` line, reproduced: 16 int8 layers.
constexpr std::uint64_t kPageRecordAllInt8 = page_record_bytes(16, 0);
// 8 int8 + 6 nvfp4 + 2 int8 -> 1,196,288; the engine's own reading for that stack.
constexpr std::uint64_t kPageRecordMixed = page_record_bytes(10, 6);
// 6 int8 + 10 nvfp4 -> 1,206,016.
constexpr std::uint64_t kPageRecordMostlyNvfp4 = page_record_bytes(6, 10);

// The box's available runtime capacity at the 1M vector, as the ENGINE reported it to another
// line's arm: "requested Engine runtime reservation requires 19633161216 bytes, but only
// 11244332032 bytes are available for runtime capacity". Quoted, not derived here.
constexpr std::uint64_t kAvailableRuntimeBytes = 11244332032ULL;
constexpr std::uint64_t kRefusedReservationBytes = 19633161216ULL;

p::PagingPreallocationInputs mainline(std::uint64_t max_context, std::uint32_t prefill_chunk,
                                      std::uint32_t watermark) {
    p::PagingPreallocationInputs in;
    in.max_context_tokens = max_context;
    in.page_tokens = 64;  // kPagedKVPageSize, src/core/paged_kv_cache.h:17
    in.max_concurrency = 1;
    in.cold_keep_tokens = 128;  // --cold-keep-tokens default
    in.prefill_chunk = prefill_chunk;
    in.unload_watermark_pages = watermark;
    return in;
}

// ---------------------------------------------------------------- the unit itself, pinned first.

void arm_0_the_unit_is_the_page_not_the_layer() {
    // The three readings the engine has PRINTED, each reproduced by the sum-over-layers form.
    check_eq(kPageRecordAllInt8, 1181696ULL, "16 int8 layers = the D1 arm's bytes_per_record");
    check_eq(kPageRecordMixed, 1196288ULL, "8i8+6nvfp4+2i8 = 1,196,288");
    check_eq(kPageRecordMostlyNvfp4, 1206016ULL, "6i8+10nvfp4 = 1,206,016");
    // And the under-count, stated as arithmetic rather than as prose: one layer's one head-plane
    // is 9,536 B against 1,196,288 B for the page -- a factor of layers x kv_heads x 2.
    check_eq(kPageRecordMixed / kNvfp4RecordAt4BitBytes, 125, "under-count factor, truncated");
    check_eq(kPageRecordAllInt8 / kInt8SlotBytes, 128, "all-int8 under-count factor");
    check_eq(16ULL * kKvHeads * kKvPlanes, 128ULL, "layer count x kv_heads x 2");
    // The reference arm's own numbers, so arm 4's conclusions are anchored here.
    check_eq(256ULL * kPageRecordMixed, 306249728ULL, "256 pages in the engine's unit");
    check_eq(256ULL * kPageRecordMixed / (1024ULL * 1024ULL), 292ULL, "256 pages, MiB (truncated)");
    check_eq(18ULL * kPageRecordMixed, 21533184ULL, "the derived 18 cold pages");
    check_eq(kReferenceColdDiskBytes / kPageRecordMixed, 224ULL,
             "--cold-disk-bytes 256 MiB holds this many pages, NOT 256");

    // ---------------------------------------------------------- the two strides, pinned both ways.
    // The two cold strides product/kv_paging_preallocation.h names in its refusal text, asserted
    // against the authority's own pinned values. This file cannot include kv_tier_formats.h: that
    // would pull ninfer/types.h in and end the `-I src`-alone buildability the paging header's :56
    // rule buys. So the coupling is the one kv_tier_formats.h itself uses over the ops authority
    // (:505-507, literal 9232 with the authority named in the message): restate the value and NAME
    // the authority -- 9,632 is kKvColdPoolStrideBytes (kv_tier_formats.h:289, pinned == 9632 at
    // :329) and 9,232 is kKvColdInt8PayloadBytes (:301, pinned == 9232 at :348). The asserts in
    // the paging header name these same two constants back, so a stride cannot move in either file
    // without one of the two failing.
    check_eq(p::kPagingColdStrideRansBytes, 9632, "== kv_tier_formats.h kKvColdPoolStrideBytes");
    check_eq(p::kPagingColdStrideInt8RawBytes, 9232,
             "== kv_tier_formats.h kKvColdInt8PayloadBytes");
    check_eq(p::kPagingColdStrideInt8RawBytes, kInt8SlotBytes,
             "and it is the same 9,232 B the int8 reading above uses");
    // The other half of the pair, so neither number can be read as the other: 9,536 is the 4-BIT
    // no-expansion record this arm's readings were taken at, and it is NOT the nvfp4 pool stride
    // the refusal text must quote. A future edit that "fixes" the readings to the current stride,
    // or the refusal string back to 9,536, fails here first.
    check(p::kPagingColdStrideRansBytes != kNvfp4RecordAt4BitBytes,
          "today's nvfp4 pool stride is NOT the 4-bit record the readings were taken at");
    std::cout << "  arm0 recorded at " << kNvfp4RecordAt4BitBytes << " B / pinned now "
              << p::kPagingColdStrideRansBytes << " B (nvfp4 rANS), "
              << p::kPagingColdStrideInt8RawBytes << " B (int8 raw)\n";
    std::cout << "  arm0 page record (mixed) = " << kPageRecordMixed << " B = "
              << (kPageRecordMixed / 1024) << " KiB; one slot stride = " << kNvfp4RecordAt4BitBytes
              << " B\n";
}

// ---------------------------------------------------------------- 1. the old expression, exactly.

void arm_1_default_path_is_the_declared_context_floor() {
    check_eq(p::paging_page_count(262144, 64), 4096, "page_count(262144, 64)");
    check_eq(p::declared_context_floor_pages(262144, 64, 1), 4096,
             "declared floor at 262144 tokens");
    check_eq(p::declared_context_floor_pages(262144, 64, 4), 4096,
             "declared floor is page-bound, not concurrency-bound");
    check_eq(p::declared_context_floor_pages(64, 64, 8), 8,
             "declared floor is concurrency-bound for a tiny context");
    check_eq(p::paging_page_count(65, 64), 2, "page_count rounds up");

    // NO --cold-policy: the plan must be the declared floor, byte-for-byte, and must not claim to
    // be paging at all. This is the bit-identical arm of the change.
    auto in = mainline(32768, 3072, 48);
    check_eq(p::paging_backing_capacity_pages(in), 0, "no medium => no backing capacity");
    const auto plan = p::plan_paging_preallocation(in);
    check(!plan.paging_active, "medium None must not report paging_active");
    check_eq(plan.declared_pages, 512, "declared pages at 32768 tokens");
    check_eq(plan.device_pool_pages, 512, "medium None => pool == declared context (unchanged)");
    check_eq(plan.total_device_pages, 512, "medium None => no cold slots");
    check_eq(plan.required_backing_pages, 0, "medium None => nothing needs backing");
    check(p::paging_explicit_capacity_refusal(in, 512).empty(),
          "explicit capacity at the declared floor must be admitted");
    check(p::paging_explicit_capacity_refusal(in, 4096).empty(),
          "explicit capacity above the declared floor must be admitted");
}

// ------------------------------------------------------------- 2. the working-set size, spelled out.

void arm_2_working_set_sizing() {
    // 32,768 tokens / 64 = 512 declared pages.
    //   keep      = ceil(128/64)  =   2 pages
    //   round     = ceil(3072/64) =  48 pages
    //   watermark =                  48 pages
    //   resident  = 2 + 48 + 48   =  98 pages ;  required backing = 512 - 98 = 414 pages
    auto in = mainline(32768, 3072, 48);
    in.medium = p::PagingColdMedium::Disk;
    in.cold_page_record_bytes = kPageRecordMixed;
    in.max_cold_pages = 414;
    in.cold_disk_bytes = 414ULL * kPageRecordMixed;

    const auto plan = p::plan_paging_preallocation(in);
    check(plan.paging_active, "a disk medium must report paging_active");
    check_eq(plan.keep_pages, 2, "keep pages");
    check_eq(plan.round_pages, 48, "round pages");
    check_eq(plan.watermark_pages, 48, "watermark pages");
    check_eq(plan.resident_pages, 98, "resident pages");
    check_eq(plan.declared_pages, 512, "declared pages");
    check_eq(plan.required_backing_pages, 414, "required backing pages");
    check_eq(plan.backing_capacity_pages, 414, "backing capacity pages");
    check_eq(plan.device_pool_pages, 98, "device pool = resident set");
    check_eq(plan.device_cold_slot_pages, 414, "cold slots = --max-cold-pages");
    // THE INVARIANT, unchanged by the unit correction.
    check_eq(plan.device_pool_pages + plan.required_backing_pages, plan.declared_pages,
             "pool + required backing == declared (no page uncovered)");
    std::cout << "  arm2 " << plan.describe() << '\n';

    // Backing capacity is bounded by BOTH the disk byte cap and the device cold-slot pool
    // (product/kv_cold_tier_budget.h:136-137): one device cold slot per spilled page.
    auto slot_bound = in;
    slot_bound.cold_disk_bytes = 100000ULL * kPageRecordMixed;
    slot_bound.max_cold_pages = 100;
    check_eq(p::paging_backing_capacity_pages(slot_bound), 100,
             "disk capacity is bounded by the device cold-slot pool");
    rejects("would have neither a device replica nor a cold tier",
            [&] { (void)p::plan_paging_preallocation(slot_bound); });

    auto byte_bound = in;
    byte_bound.cold_disk_bytes = 100ULL * kPageRecordMixed;
    byte_bound.max_cold_pages = 100000;
    check_eq(p::paging_backing_capacity_pages(byte_bound), 100,
             "disk capacity is bounded by --cold-disk-bytes too");

    auto small = mainline(8192, 512, 48);
    small.medium = p::PagingColdMedium::Disk;
    small.cold_page_record_bytes = kPageRecordMixed;
    small.max_cold_pages = 70;
    small.cold_disk_bytes = 70ULL * kPageRecordMixed;
    const auto small_plan = p::plan_paging_preallocation(small);
    check_eq(small_plan.resident_pages, 58, "short-context resident pages");
    check_eq(small_plan.required_backing_pages, 70, "short-context required backing");
    check_eq(small_plan.device_pool_pages, 58, "short-context pool = resident");

    auto tiny = mainline(2048, 3072, 48);
    tiny.medium = p::PagingColdMedium::Disk;
    tiny.cold_page_record_bytes = kPageRecordMixed;
    tiny.max_cold_pages = 256;
    tiny.cold_disk_bytes = kReferenceColdDiskBytes;
    const auto tiny_plan = p::plan_paging_preallocation(tiny);
    check_eq(tiny_plan.declared_pages, 32, "tiny declared pages");
    check_eq(tiny_plan.resident_pages, 32, "resident clamps to declared");
    check_eq(tiny_plan.device_pool_pages, 32,
             "resident >= declared => pool is the declared context");
    check_eq(tiny_plan.required_backing_pages, 0, "resident >= declared => nothing to back");
}

// ------------------------------------------------------------------ 3. THE REFUSAL (load-bearing).

void arm_3_the_refusal() {
    // 3a. THE LIVE REFERENCE ARM'S OWN KNOBS at 262,144 tokens: 4096 declared, 98 resident,
    // 3998 to back. 256 MiB of spill holds 224 pages of the engine's unit, and the arm's
    // --max-cold-pages 256 holds 256 -- so BOTH halves are short, and the byte half binds first.
    auto in = mainline(262144, 3072, 48);
    in.medium = p::PagingColdMedium::Disk;
    in.cold_page_record_bytes = kPageRecordMixed;
    in.max_cold_pages = kReferenceMaxColdPages;
    in.cold_disk_bytes = kReferenceColdDiskBytes;
    check_eq(in.max_context_tokens / 64, 4096, "262144 tokens is 4096 pages of 64");
    check_eq(p::paging_required_backing_pages(in), 3998, "required backing at 262144 tokens");
    check_eq(p::paging_backing_capacity_pages(in), 224,
             "the reference arm's 256 MiB binds first, at 224 pages");
    rejects("would have neither a device replica nor a cold tier",
            [&] { (void)p::plan_paging_preallocation(in); });
    rejects("pages [322, 4096)", [&] { (void)p::plan_paging_preallocation(in); });
    rejects("short by 3774 pages", [&] { (void)p::plan_paging_preallocation(in); });
    rejects("raise --max-cold-pages to 3998", [&] { (void)p::plan_paging_preallocation(in); });
    // 3774 pages x 1,196,288 B = 4,514,790,912 B of spill that does not exist.
    rejects("4514790912 B", [&] { (void)p::plan_paging_preallocation(in); });

    // Raising only the DISK budget must NOT be accepted as a fix: then the slot pool binds, and
    // the shortfall becomes 3998 - 256 = 3742 pages. Both refusals, same guard.
    auto disk_only = in;
    disk_only.cold_disk_bytes = 1ULL << 40;  // 1 TiB of spill bytes
    check_eq(p::paging_backing_capacity_pages(disk_only), 256,
             "with bytes plentiful, --max-cold-pages is the bound again");
    rejects("short by 3742 pages", [&] { (void)p::plan_paging_preallocation(disk_only); });

    // ... and the same knobs DO admit once BOTH halves cover the difference.
    auto fixed = in;
    fixed.max_cold_pages = 3998;
    fixed.cold_disk_bytes = 3998ULL * kPageRecordMixed;
    const auto plan = p::plan_paging_preallocation(fixed);
    check_eq(plan.device_pool_pages, 98, "fixed: pool = resident set");
    check_eq(plan.device_cold_slot_pages, 3998, "fixed: cold slots");
    std::cout << "  arm3 " << plan.describe() << '\n';

    // 3b. NO medium: the derived path must NOT throw -- that would change every run that arms no
    // cold tier. It returns the declared floor unchanged instead.
    auto none = mainline(262144, 3072, 48);
    check(p::paging_preallocation_refusal(none).empty(),
          "no --cold-policy is not itself a refusal: it is the unchanged default path");
    const auto none_plan = p::plan_paging_preallocation(none);
    check(!none_plan.paging_active, "no --cold-policy => paging_active false");
    check_eq(none_plan.device_pool_pages, 4096, "no --cold-policy => the declared floor, unchanged");

    // 3c. An unknown per-page record may not be guessed at.
    auto unknown_unit = mainline(262144, 3072, 48);
    unknown_unit.medium = p::PagingColdMedium::Disk;
    unknown_unit.max_cold_pages = 3998;
    unknown_unit.cold_disk_bytes = kReferenceColdDiskBytes;
    unknown_unit.cold_page_record_bytes = 0;
    rejects("silent approximation", [&] { (void)p::plan_paging_preallocation(unknown_unit); });
    rejects("page record", [&] { (void)p::plan_paging_preallocation(unknown_unit); });

    // 3d. The watermark derive sentinel must be resolved, not guessed.
    auto unresolved = mainline(262144, 3072, 48);
    unresolved.unload_watermark_pages = p::kPagingWatermarkDerive;
    unresolved.medium = p::PagingColdMedium::Disk;
    unresolved.cold_page_record_bytes = kPageRecordMixed;
    unresolved.max_cold_pages = 3998;
    unresolved.cold_disk_bytes = 3998ULL * kPageRecordMixed;
    rejects("derive sentinel", [&] { (void)p::plan_paging_preallocation(unresolved); });

    // 3e. A pool whose whole usable range lies inside the unload watermark's slack can never
    // trigger an unload. Fires exactly when keep and round are both 0 and the watermark is not.
    auto never_binds = mainline(32768, 0, 48);
    never_binds.cold_keep_tokens = 0;
    never_binds.medium = p::PagingColdMedium::Disk;
    never_binds.cold_page_record_bytes = kPageRecordMixed;
    never_binds.max_cold_pages = 464;
    never_binds.cold_disk_bytes = 464ULL * kPageRecordMixed;
    check_eq(p::paging_resident_pages(never_binds), 48, "watermark-only resident set");
    check_eq(p::paging_required_backing_pages(never_binds), 464, "watermark-only required backing");
    rejects("could never fire", [&] { (void)p::plan_paging_preallocation(never_binds); });

    // 3f. The explicit-capacity guard, no backing store (flash-next's exact situation).
    auto fn = mainline(262144, 3072, 48);
    check(p::paging_explicit_capacity_refusal(fn, 4096).empty(),
          "flash-next: explicit capacity at the floor is admitted");
    const std::string refusal = p::paging_explicit_capacity_refusal(fn, 1024);
    check(!refusal.empty(), "flash-next: explicit capacity below the floor must be refused");
    check_mentions("NO backing store is configured", refusal,
                   "flash-next refusal must say no backing store is configured");
    check_mentions("1024 pages", refusal, "flash-next refusal must name the requested count");
    check_mentions("4096", refusal, "flash-next refusal must name the floor it is below");
    check_mentions("262144", refusal, "flash-next refusal must name the max_context");
    check_mentions("max(ceil(max_context/page_tokens), max_concurrency)", refusal,
                   "flash-next refusal must name the expression that produced the floor");
    check_mentions("--kv-capacity 262144", refusal,
                   "flash-next refusal must name the value that would be accepted");
    std::cout << "  arm3f " << refusal << '\n';

    // The explicit guard returns its message rather than throwing (it is a predicate over a flag).
    auto ex = mainline(262144, 3072, 48);
    ex.medium = p::PagingColdMedium::Disk;
    ex.cold_page_record_bytes = kPageRecordMixed;
    ex.max_cold_pages = 3998;
    ex.cold_disk_bytes = 3998ULL * kPageRecordMixed;
    const std::string below_resident = p::paging_explicit_capacity_refusal(ex, 97);
    check(!below_resident.empty(), "explicit pool below the resident set must be refused");
    check_mentions("below the resident set", below_resident, "explicit-below-resident refusal");
    check_mentions("keep=", below_resident, "explicit-below-resident refusal breaks down the set");
    check(p::paging_explicit_capacity_refusal(ex, 98).empty(),
          "explicit pool == resident set with full backing must be admitted");

    auto ex_small = ex;
    ex_small.max_cold_pages = 1000;
    ex_small.cold_disk_bytes = 1000ULL * kPageRecordMixed;
    const std::string gap = p::paging_explicit_capacity_refusal(ex_small, 2000);
    check(!gap.empty(), "explicit pool with an uncoverable gap must be refused");
    check_mentions("short by 1096 pages", gap, "explicit-gap refusal names the shortfall");
    check_mentions("--kv-capacity 262144", gap, "explicit-gap refusal names the accepted value");
}

// ------------------------------------------------- 4. the spill-capacity contract, IN ENGINE UNITS.

void arm_4_spill_capacity_contract_at_the_largest_declarable_context() {
    // 1,010,000 tokens is the largest context the record measured as DONE:
    // research/notes/TODO.md:5762-5764 -- `--max-context 1010000 --yarn` passes while
    // `--max-context 1048576 --yarn` fails on the gqa_attention workspace profile. So 1,048,576 is
    // unreachable for a reason that has nothing to do with memory, and 1,010,000 is the honest
    // largest case.
    check_eq(p::paging_page_count(1010000, 64), 15782, "page_count(1010000, 64)");
    auto in = mainline(1010000, 3072, 48);
    in.medium = p::PagingColdMedium::Disk;
    in.cold_page_record_bytes = kPageRecordMixed;
    in.max_cold_pages = kReferenceMaxColdPages;
    in.cold_disk_bytes = kReferenceColdDiskBytes;
    check_eq(p::paging_resident_pages(in), 98, "resident pages at 1010000 tokens");
    check_eq(p::paging_required_backing_pages(in), 15684, "required backing at 1010000 tokens");

    // THE REQUIREMENT IN THE ENGINE'S UNIT. 15,684 pages x 1,196,288 B/page.
    const std::uint64_t required_bytes =
        static_cast<std::uint64_t>(p::paging_required_backing_pages(in)) * kPageRecordMixed;
    check_eq(required_bytes, 18762580992ULL, "spill required at 1010000 tokens, engine's unit");
    check_eq(required_bytes / (1024ULL * 1024ULL), 17893ULL, "required, MiB (truncated)");
    check_eq(required_bytes / (1024ULL * 1024ULL * 1024ULL), 17ULL, "required, GiB (truncated)");
    std::cout << "  arm4 contract requires " << required_bytes << " B = " << (required_bytes >> 30)
              << " GiB across " << p::paging_required_backing_pages(in) << " pages\n";

    // THE WITHDRAWN CLAIM, asserted as FALSE rather than merely dropped: an earlier version of
    // this change's report said --cold-disk-bytes 268435456 "covers the byte half". It does not.
    check(required_bytes > kReferenceColdDiskBytes,
          "withdrawn: 256 MiB does NOT cover the byte half at 1010000 tokens");
    check_eq(kReferenceColdDiskBytes / kPageRecordMixed, 224ULL, "256 MiB holds 224 pages");
    check_eq(p::paging_required_backing_pages(in) - kReferenceColdDiskBytes / kPageRecordMixed, 15460ULL,
             "byte-half shortfall at 1010000 tokens, pages");
    rejects("short by 15460 pages", [&] { (void)p::plan_paging_preallocation(in); });
    rejects("--max-cold-pages to 15684", [&] { (void)p::plan_paging_preallocation(in); });

    // THE HONEST CONSEQUENCE. The plan IS satisfiable in coverage terms -- raise both knobs and
    // the guard admits -- but the DEVICE COLD-SLOT POOL that requires (one slot per spilled page,
    // kv_cold_tier_budget.h:136-137) is 17.47 GiB, against the 10.47 GiB the engine reported
    // available to another line's arm. So backing every declared page is NOT honourable on this
    // box: the engine's OWN reservation guard is what must refuse, and it did
    // ("requested Engine runtime reservation requires 19633161216 bytes, but only 11244332032
    // bytes are available for runtime capacity").
    auto ok = in;
    ok.max_cold_pages = 15684;
    ok.cold_disk_bytes = required_bytes;
    const auto plan = p::plan_paging_preallocation(ok);
    check_eq(plan.device_pool_pages, 98, "coverage satisfied: pool = 98 pages");
    check_eq(plan.device_cold_slot_pages, 15684, "coverage satisfied: 15684 cold slots");
    const std::uint64_t slot_bytes =
        static_cast<std::uint64_t>(plan.device_cold_slot_pages) * kPageRecordMixed;
    check_eq(slot_bytes, required_bytes, "the device cold-slot pool costs the same as the spill");
    check_eq(slot_bytes >> 30, 17ULL, "device cold-slot pool, GiB (truncated)");
    check(slot_bytes > kAvailableRuntimeBytes,
          "NOT HONOURABLE here: the cold-slot pool alone exceeds the available reservation");
    check_eq(slot_bytes - kAvailableRuntimeBytes, 7518248960ULL, "overrun, bytes");
    check_eq((slot_bytes - kAvailableRuntimeBytes) >> 30, 7ULL, "overrun, GiB (truncated)");
    check_eq(kRefusedReservationBytes, 19633161216ULL,
             "the engine's own refusal figure, quoted from the arm that ran");
    std::cout << "  arm4 device cold-slot pool " << slot_bytes << " B = " << (slot_bytes >> 30)
              << " GiB vs " << kAvailableRuntimeBytes << " B available\n";

    // THE HONOURABLE ENVELOPE, in the same unit. 256 pages cost 292.05 MiB -- the reference
    // document's own need -- and the derived 18 cold pages cost 20.54 MiB.
    check_eq(256ULL * kPageRecordMixed, 306249728ULL, "256 pages in the engine's unit");
    check_eq(18ULL * kPageRecordMixed, 21533184ULL, "the derived 18 cold pages");
    check(256ULL * kPageRecordMixed < kAvailableRuntimeBytes,
          "256 pages IS honourable on this box");
    // The largest declared context the box's available capacity can back for the cold-slot half.
    const std::uint64_t honourable_cold_pages = kAvailableRuntimeBytes / kPageRecordMixed;
    check_eq(honourable_cold_pages, 9399ULL, "cold pages that fit in the available reservation");
    const std::uint64_t honourable_declared_pages = honourable_cold_pages + 98ULL;
    check_eq(honourable_declared_pages * 64ULL, 607808ULL,
             "largest honourable declared context, tokens (arithmetic, not measured)");
    check(honourable_declared_pages * 64ULL < 1010000ULL,
          "the honourable ceiling is BELOW the measured working ceiling");

    auto host = mainline(1010000, 3072, 48);
    host.medium = p::PagingColdMedium::PinnedHost;
    host.host_page_bytes = kPageRecordMixed;
    host.cold_host_bytes = 15684ULL * kPageRecordMixed;
    check_eq(p::paging_device_cold_slot_pages(128, 64, p::PagingColdMedium::PinnedHost, 0), 0,
             "a host-tier page occupies no device cold slot");
    const auto host_plan = p::plan_paging_preallocation(host);
    check_eq(host_plan.device_pool_pages, 98, "host tier: pool = resident set");
    check_eq(host_plan.device_cold_slot_pages, 0, "host tier: zero device cold slots");
    check_eq(host_plan.total_device_pages, 98, "host tier: device total is just the resident set");

    auto both = mainline(1010000, 3072, 48);
    both.medium = p::PagingColdMedium::HostThenDisk;
    both.host_page_bytes = kPageRecordMixed;
    both.cold_page_record_bytes = kPageRecordMixed;
    both.max_cold_pages = 10000;
    both.cold_host_bytes = 6000ULL * kPageRecordMixed;
    both.cold_disk_bytes = 10000ULL * kPageRecordMixed;
    check_eq(p::paging_backing_capacity_pages(both), 16000,
             "host-then-disk capacity is host + min(disk bytes, slots)");
    const auto both_plan = p::plan_paging_preallocation(both);
    check_eq(both_plan.device_pool_pages, 98, "host-then-disk: pool = resident set");
}

// ------------------------------------------------- 5. the flash-next numbers on the shared arithmetic.

void arm_5_flash_next_numbers() {
    // runtime_plan.cpp:213-217  min_groups = max(ceil(max_context / kMainPageGroupTokens), concurrency)
    // runtime_plan.h:25         kMainPageGroupTokens = 256
    // runtime_plan.cpp:249      curve.minimum_main_page_groups = min_groups
    // runtime_plan.cpp:252-254  minimum_device_reservation_bytes = fixed_base + min_groups * stride
    // At max_context 262,144 -- the most that target accepts at all (runtime_plan.cpp:48-49) --
    // that is 1,024 groups, and 1,024 x 6,488,064 B = 6,643,777,536 B = 6,336 MiB = 6.19 GiB
    // (runtime_plan.h:57-58, kPhysicalStrideBytesPerGroupBf16 = 6,488,064).
    check_eq(p::paging_page_count(262144, 256), 1024, "flash-next groups at max_context 262144");
    check_eq(p::declared_context_floor_pages(262144, 256, 1), 1024,
             "flash-next declared floor == 1024 groups");
    check_eq(1024ULL * 6488064ULL, 6643777536ULL, "flash-next KV at its own ceiling, BF16");
    check_eq(6643777536ULL >> 30, 6, "flash-next KV at its own ceiling, GiB (truncated)");
    check_eq(6643777536ULL / (1024ULL * 1024ULL), 6336, "flash-next KV at its own ceiling, MiB");

    // The target has NO cold path, so it may not shrink: every non-resident page would be neither
    // recallable nor refused. The guard is the explicit-capacity refusal.
    p::PagingPreallocationInputs fn;
    fn.max_context_tokens = 262144;
    fn.page_tokens = 256;  // kMainPageGroupTokens, NOT kPagedKVPageSize -- this target's unit
    fn.max_concurrency = 1;
    fn.cold_keep_tokens = 128;
    fn.prefill_chunk = 3072;
    fn.unload_watermark_pages = 48;
    fn.medium = p::PagingColdMedium::None;
    // ceil(3072/256) = 12 round groups, ceil(128/256) = 1 keep group, + 48 watermark = 61.
    check_eq(p::paging_resident_pages(fn), 61, "flash-next resident groups");
    check_eq(p::paging_backing_capacity_pages(fn), 0, "flash-next has no backing store");
    check(p::paging_explicit_capacity_refusal(fn, 1024).empty(),
          "flash-next: naming its own floor stays legal");
    const std::string r = p::paging_explicit_capacity_refusal(fn, 256);
    check(!r.empty(), "flash-next: a pool of 256 groups with no cold path must be refused");
    check_mentions("NO backing store is configured", r, "flash-next refusal names the gap");
    check_mentions("1024", r, "flash-next refusal names the 1024-group floor");
    check_mentions("262144", r, "flash-next refusal names the max_context that produced it");
    std::cout << "  arm5 " << r << '\n';
}

} // namespace

int main() {
    arm_0_the_unit_is_the_page_not_the_layer();
    arm_1_default_path_is_the_declared_context_floor();
    arm_2_working_set_sizing();
    arm_3_the_refusal();
    arm_4_spill_capacity_contract_at_the_largest_declarable_context();
    arm_5_flash_next_numbers();

    if (failures == 0) {
        std::cout << "ok\n";
        return 0;
    }
    std::cerr << failures << " failure(s)\n";
    return 1;
}
