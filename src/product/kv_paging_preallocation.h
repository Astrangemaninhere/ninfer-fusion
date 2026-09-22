#pragma once

// Pre-allocation keyed to the PAGING WORKING SET, not to the declared context.
//
// THE PROBLEM THIS EXISTS FOR. Today the device Main-KV page pool is floored at the
// declared context:
//
//   layouts_impl.h:1808-1815 (the curve builder)
//       const std::uint32_t logical_pages = page_count(inputs.capacity);   // capacity == max_context
//       std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
//       if (inputs.kv_capacity_tokens) { minimum_pages = std::max(page_count(*inputs.kv_capacity_tokens), ...); }
//   layouts_impl.h:985-992 (the validate path, same expression)
//   src/runtime/engine/kv_capacity.cpp:99-115 (Automatic then GROWS the pool from that floor to
//       "all VRAM left after weights", so the floor is also the only thing holding it down)
//
// and `--kv-capacity auto` inherits that floor. The engine's own refusal at 1M says so verbatim
// (research/notes/TODO.md:5768-5773):
//
//   error: minimum Engine runtime reservation requires 18912736512 bytes in addition to
//          1073741824 bytes of automatic headroom, but only 11600323584 bytes are available after weights
//
// 18,912,736,512 B is the floor. It is the declared context, not a need: the mechanism that reads
// pages is paged and cold-backed, and the read set of ONE round is bounded by the pool the
// watermark keeps free, not by max_context. So the pool should be sized for what paging needs --
// the resident set plus the backing store's capacity -- and everything else should live in cold
// storage.
//
// WHY THIS IS NOT A FREE LUNCH, AND WHY THE REFUSAL IS THE POINT. Shrinking the pool removes the
// fallback that makes the cold tier sound today. product/kv_cold_tier_budget.h:35-38 says the last
// rung of the admission ladder is "stay hot: the page keeps its device replica. Counted, reported,
// never thrown". That rung is only available because the pool is big enough to hold every page, so
// a page no cold tier can take still has a device replica. Shrink the pool and a page can end up
// with NEITHER a replica NOR a tier -- and the ladder would answer that by counting a `stays_hot`
// it did not actually store. That is a silent approximation of exactly the kind this project
// rejects, and a wrong answer attributed to the model is the worst possible outcome.
//
// So this header makes the trade explicit and TOTAL:
//
//   device_pool_pages + required_backing_pages == declared_context_pages   (no page uncovered)
//
// and when the backing store cannot hold what the pool no longer holds, it THROWS. It never
// returns a pool that quietly covers less than the declared context. The refusal names the page
// range that would have no home, the page and byte shortfall, and the two knobs that fix it.
//
// IT IS OPT-IN. `medium == None` (no --cold-policy) returns the declared-context floor unchanged,
// which is what makes the default path bit-identical rather than merely equivalent. Adopting this
// cannot change a run that does not arm a cold tier.
//
// IT DOES NOT ITSELF ALLOCATE ANYTHING, and it does not touch the three things the coordinator
// fenced off: the `maximum_blocks` clamp (runtime_plan.cpp:224-225, :300-301) is a separate
// arithmetic ceiling and is NOT changed here; the withdrawal/watermark POLICY is unchanged (this
// header reads `--kv-unload-watermark-pages`, it does not redefine it); and the recall path is
// untouched. This header only answers "how big must the device pool be".
//
// Sibling of product/kv_cold_tier_budget.h (the admission ladder) and product/weight_residency.h
// (W13's residency plan): pure, no CUDA, no ninfer/types.h, so it is unit-testable with plain g++
// (tests/test_kv_paging_preallocation.cpp).

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::product {

// The sentinel EngineOptions::unload_watermark_pages uses for "derive it from the plan's prefill
// chunk" (include/ninfer/types.h:273, kUnloadWatermarkDerive). Pre-allocation runs before that
// derivation, so the resolved value is an input here and the sentinel is refused by name rather
// than guessed at.
inline constexpr std::uint32_t kPagingWatermarkDerive = 0xFFFFFFFFU;

// THE TWO COLD STRIDES THE UNIT WARNINGS BELOW NAME, named ONCE so that neither the comment on
// `cold_page_record_bytes` nor the refusal text can carry a private copy of a number the geometry
// owns. They are the same two records product/kv_tier_formats.h defines and pins:
//
//   kPagingColdStrideRansBytes    == kKvColdPoolStrideBytes  == 9632 B (kv_tier_formats.h:329)
//   kPagingColdStrideInt8RawBytes == kKvColdInt8PayloadBytes == 9232 B (kv_tier_formats.h:348)
//
// and each of those asserts names its authority (ops::kEntropyNvfp4SlotBytes /
// ops::kColdI8SlotBytes) in its message rather than including it, because the real ops header
// carries <cuda_runtime.h> and cannot be included host-side at all. The include is not taken from
// here either: this is the one product header with a "no ninfer/types.h" rule (:56), and that rule
// is what keeps tests/test_kv_paging_preallocation.cpp buildable with plain g++ and `-I src`
// alone. That test cannot take the mirror include either without giving the property up, so the
// coupling is done the way kv_tier_formats.h does it over the ops authority (:325-329): the value
// is restated and the AUTHORITY IS NAMED. Both spellings are then pinned from both sides -- the
// asserts below name kKvColdInt8PayloadBytes / kKvColdPoolStrideBytes, and that test asserts these
// same two names back against 9232 / 9632 -- so neither stride can move without one of the two
// failing, which is what "one definition, spelled two ways, asserted equal" means here.
inline constexpr std::int32_t kPagingColdStrideInt8RawBytes = 9232;
static_assert(kPagingColdStrideInt8RawBytes == 9232,
              "the raw cold stride moved: product/kv_tier_formats.h kKvColdInt8PayloadBytes and "
              "the refusal text in paging_preallocation_refusal() must be re-derived deliberately");
inline constexpr std::int32_t kPagingColdStrideRansBytes = 9632;
static_assert(kPagingColdStrideRansBytes == 9632,
              "the rANS cold stride moved: product/kv_tier_formats.h kKvColdPoolStrideBytes and "
              "the refusal text in paging_preallocation_refusal() must be re-derived deliberately");

// Which medium actually holds the pages the device pool does not. Mirrors
// include/ninfer/types.h:165-187 ColdPolicy, minus the values that carry no backing store.
//   DeviceWindow  -- ColdPolicy::Window:  a device-resident cold-slot pool, capped in pages.
//   PinnedHost    -- ColdPolicy::Host:    pinned host memory, and NO device cold slot per page
//                                         (kv_cold_tier_budget.h:28-31: "occupies NO device cold
//                                         slot ... restored by one pinned H2D copy").
//   Disk          -- ColdPolicy::Disk:    the SSD spill, whose admission needs the disk byte cap
//                                         AND one device cold slot per spilled page.
//   HostThenDisk  -- ColdPolicy::HostThenDisk: host first, disk as the overflow (both live).
enum class PagingColdMedium : std::uint8_t {
    None,
    DeviceWindow,
    PinnedHost,
    Disk,
    HostThenDisk,
};

[[nodiscard]] inline std::string_view paging_cold_medium_name(PagingColdMedium medium) noexcept {
    switch (medium) {
    case PagingColdMedium::None: return "none";
    case PagingColdMedium::DeviceWindow: return "window";
    case PagingColdMedium::PinnedHost: return "host";
    case PagingColdMedium::Disk: return "disk";
    case PagingColdMedium::HostThenDisk: return "host-then-disk";
    }
    return "?";
}

// Whole pages over a token count, rounding up: the same arithmetic as
// layouts_impl.h:78-81 page_count() / kv_capacity.cpp:40-41 explicit_page_groups(), written once.
[[nodiscard]] inline std::uint32_t paging_page_count(std::uint64_t tokens,
                                                     std::uint32_t page_tokens) noexcept {
    if (page_tokens == 0 || tokens == 0) { return 0; }
    const std::uint64_t pages = 1ULL + (tokens - 1ULL) / page_tokens;
    return pages > 0xFFFFFFFFULL ? 0xFFFFFFFFU : static_cast<std::uint32_t>(pages);
}

// THE OLD EXPRESSION, verbatim in behaviour: max(page_count(max_context), max_concurrency).
// layouts_impl.h:1808-1812 and :985-988 both compute exactly this. It is a function here so the
// equivalence arm of the test compares against the old expression instead of re-deriving it, and
// so `medium == None` can return it unchanged.
[[nodiscard]] inline std::uint32_t
declared_context_floor_pages(std::uint64_t max_context_tokens, std::uint32_t page_tokens,
                             std::uint32_t max_concurrency) noexcept {
    const std::uint32_t pages = paging_page_count(max_context_tokens, page_tokens);
    const std::uint32_t floor_concurrency = max_concurrency == 0 ? 1U : max_concurrency;
    return pages > floor_concurrency ? pages : floor_concurrency;
}

// The device cold-slot pool. This is `effective_cold_pages` at layouts_impl.h:176-206: Window,
// Disk and HostThenDisk all take `--max-cold-pages` when set, else cold_keep_tokens/page + 16;
// Host takes 0, because its pages occupy no device slot.
[[nodiscard]] inline std::uint32_t
paging_device_cold_slot_pages(std::uint32_t cold_keep_tokens, std::uint32_t page_tokens,
                              PagingColdMedium medium, std::uint32_t max_cold_pages) noexcept {
    if (medium == PagingColdMedium::None || medium == PagingColdMedium::PinnedHost) { return 0; }
    if (max_cold_pages != 0) { return max_cold_pages; }
    const std::uint32_t keep_pages = paging_page_count(cold_keep_tokens, page_tokens);
    return keep_pages + 16U;
}

struct PagingPreallocationInputs {
    // --max-context: the rope domain / logical ceiling, NOT a device requirement.
    std::uint64_t max_context_tokens = 0;
    // kPagedKVPageSize, src/core/paged_kv_cache.h:17 = 64. A parameter, never a constant here.
    std::uint32_t page_tokens = 64;
    std::uint32_t max_concurrency = 1;

    // The resident set's three terms.
    //   keep      -- --cold-keep-tokens: the hot prefix nothing evicts.
    //   round     -- --prefill-chunk: the most tokens a single round can have in flight, which is
    //                the largest amount that can arrive between two unload points.
    //   watermark -- --kv-unload-watermark-pages, RESOLVED (not kPagingWatermarkDerive): the free
    //                slack the unload trigger maintains, hence headroom the pool must be able to
    //                be below.
    std::uint32_t cold_keep_tokens = 128;
    std::uint32_t prefill_chunk = 0;
    std::uint32_t unload_watermark_pages = 0;

    // The backing store, exactly as the engine sizes it.
    PagingColdMedium medium = PagingColdMedium::None;
    std::uint32_t max_cold_pages = 0;   // --max-cold-pages (0 = derive, see the function above)
    std::uint64_t cold_disk_bytes = 0;  // --cold-disk-bytes
    std::uint64_t cold_host_bytes = 0;  // --cold-host-bytes
    // BYTES PER COLD PAGE ACROSS ALL SLOT-BEARING LAYERS -- the engine's own unit, and NOT one
    // layer's stride. This is `turn_recall_page_bytes()` (program_impl.h:12839-12855), which sums
    // `view.cold_slots.nb[3]` over every slot-bearing layer. `nb[3]` is a BYTE stride, not a page
    // count -- program_impl.h:12606 indexes slots with it -- because the tensor is declared
    // `{stride, kv_heads, 2, cold_pages}` (decoder_state.cpp:931-933), so
    //
    //     nb[3]                  = stride_layer * kv_heads * 2        for ONE layer, and
    //     cold_page_record_bytes = SUM over slot-bearing layers of (stride_layer * kv_heads * 2)
    //
    // It is also what the engine prints as `bytes_per_record` on the `[textcargo]` line
    // (program_impl.h:12071-12085), ~1.18 MB/page (program_impl.h:12031). Passing ONE layer's
    // stride here (kPagingColdStrideInt8RawBytes raw / kPagingColdStrideRansBytes rANS, named
    // above) under-counts a page by the number of slot-bearing layers
    // times kv_heads times 2 -- 128x on a 16-layer, 4-KV-head stack -- and that is exactly the
    // under-count the first version of this change shipped. 0 means the caller does not know it,
    // which is refused rather than approximated.
    //
    // WARNING FOR THE NEXT READER -- FIXED 2026-09-19, and this is what it was. The same unit
    // confusion lived one file over: program_impl.h's SPILL FILE-SLOT BITMAP was sized as
    // `--cold-disk-bytes / max_l nb[3]_l`, i.e. it divided by ONE layer's per-page stride, while
    // a page's records occupy the SUM across the per-layer files, so a declared 4 GiB admitted
    // 63.9997 GiB of pages (16x on the shipped 16-layer stack -- dl/pagemath/REPORT.md section
    // 1). The bitmap block of ProgramImplCore's constructor now divides by
    // `turn_recall_page_bytes()`, the very sum this header calls `cold_page_record_bytes`, so
    // the bitmap and the ladder's disk capacity agree by construction. If you meet ANOTHER site
    // that uses one layer's stride as a page size, check the unit before quoting it.
    std::uint64_t cold_page_record_bytes = 0;
    // Per-page pinned-host stride (HostKVPageLayout::page_stride) for the Host medium.
    std::uint64_t host_page_bytes = 0;
};

// The resident set: what the pool must hold at once for paging to work at all.
[[nodiscard]] inline std::uint32_t
paging_resident_pages(const PagingPreallocationInputs& in) noexcept {
    const std::uint32_t keep_pages = paging_page_count(in.cold_keep_tokens, in.page_tokens);
    const std::uint32_t round_pages = paging_page_count(in.prefill_chunk, in.page_tokens);
    const std::uint64_t sum = static_cast<std::uint64_t>(keep_pages) + round_pages +
                              in.unload_watermark_pages;
    const std::uint64_t capped = sum > 0xFFFFFFFFULL ? 0xFFFFFFFFULL : sum;
    const std::uint32_t floor_concurrency = in.max_concurrency == 0 ? 1U : in.max_concurrency;
    return static_cast<std::uint32_t>(capped) > floor_concurrency
               ? static_cast<std::uint32_t>(capped)
               : floor_concurrency;
}

// Whole pages the backing store can hold. For Disk and HostThenDisk this is bounded by the device
// cold-slot pool as well as the byte cap: kv_cold_tier_budget.h:136-137 admits a disk page only
// when `usage.disk_pages < disk_capacity && usage.device_cold_slots < budget.device_cold_pages`,
// i.e. the number of pages that can be cold-on-disk is min(disk pages, --max-cold-pages). This is
// the single most important number in the whole contract, and it is the one the record's 1M
// arithmetic never counted.
[[nodiscard]] inline std::uint32_t
paging_backing_capacity_pages(const PagingPreallocationInputs& in) noexcept {
    const auto byte_pages = [](std::uint64_t bytes, std::uint64_t page_bytes) -> std::uint64_t {
        if (page_bytes == 0) { return 0; }
        return bytes / page_bytes;
    };
    const std::uint64_t slots = paging_device_cold_slot_pages(
        in.cold_keep_tokens, in.page_tokens, in.medium, in.max_cold_pages);
    const auto cap = [](std::uint64_t value) -> std::uint32_t {
        return value > 0xFFFFFFFFULL ? 0xFFFFFFFFU : static_cast<std::uint32_t>(value);
    };
    switch (in.medium) {
    case PagingColdMedium::None:
        return 0;
    case PagingColdMedium::DeviceWindow:
        // The cold pool IS device slots; there is no separate medium.
        return cap(slots);
    case PagingColdMedium::PinnedHost:
        // Read-free pages: no device slot per page, so the host byte cap is the whole story.
        return cap(byte_pages(in.cold_host_bytes, in.host_page_bytes));
    case PagingColdMedium::Disk: {
        const std::uint64_t disk = byte_pages(in.cold_disk_bytes, in.cold_page_record_bytes);
        return cap(disk < slots ? disk : slots);
    }
    case PagingColdMedium::HostThenDisk: {
        const std::uint64_t host = byte_pages(in.cold_host_bytes, in.host_page_bytes);
        const std::uint64_t disk = byte_pages(in.cold_disk_bytes, in.cold_page_record_bytes);
        const std::uint64_t disk_bounded = disk < slots ? disk : slots;
        return cap(host + disk_bounded);
    }
    }
    return 0;
}

// What the backing store must hold: every declared page the device pool does not. The plan is only
// sound when `paging_backing_capacity_pages() >= paging_required_backing_pages()`, and that is
// exactly the comparison the refusal below makes.
[[nodiscard]] inline std::uint32_t
paging_required_backing_pages(const PagingPreallocationInputs& in) noexcept {
    const std::uint32_t declared = declared_context_floor_pages(
        in.max_context_tokens, in.page_tokens, in.max_concurrency);
    const std::uint32_t resident = paging_resident_pages(in);
    return resident >= declared ? 0U : declared - resident;
}

struct PagingPreallocation {
    // false => "this is the declared-context path, unchanged". True only when a cold medium is
    // armed, which is why an unarmed run cannot be affected by this header at all.
    bool paging_active = false;
    std::uint32_t declared_pages = 0;
    std::uint32_t resident_pages = 0;
    std::uint32_t keep_pages = 0;
    std::uint32_t round_pages = 0;
    std::uint32_t watermark_pages = 0;
    // required_backing + device_pool_pages == declared_pages, always.
    std::uint32_t required_backing_pages = 0;
    std::uint32_t backing_capacity_pages = 0;
    std::uint32_t device_pool_pages = 0;
    std::uint32_t device_cold_slot_pages = 0;
    // The device KV total this plan implies: the main page pool plus the cold-slot pool. Compare
    // against the declared floor to see what was actually saved.
    std::uint32_t total_device_pages = 0;

    [[nodiscard]] std::string describe() const {
        std::string out = "paging-prealloc active=";
        out += paging_active ? "yes" : "no";
        out += " declared=" + std::to_string(declared_pages);
        out += " resident=" + std::to_string(resident_pages);
        out += " (keep=" + std::to_string(keep_pages) + " round=" + std::to_string(round_pages) +
               " watermark=" + std::to_string(watermark_pages) + ")";
        out += " backing=" + std::to_string(backing_capacity_pages) + "/" +
               std::to_string(required_backing_pages);
        out += " device_pool=" + std::to_string(device_pool_pages);
        out += "+slots=" + std::to_string(device_cold_slot_pages);
        out += " total_device=" + std::to_string(total_device_pages);
        return out;
    }
};

// The INPUT invariants, separate from the capacity question so both entry points can check them
// before any arithmetic runs. This is not cosmetic: an unresolved watermark sentinel makes
// paging_resident_pages() saturate at UINT32_MAX, and a `resident >= declared` early-return would
// then silently swallow it as "the old path already fits". The sentinel has to be refused before
// anything is computed from it.
[[nodiscard]] inline std::string
paging_preallocation_input_refusal(const PagingPreallocationInputs& in) {
    if (in.page_tokens == 0) {
        return "paging pre-allocation: page_tokens is 0, so no page count can be formed";
    }
    if (in.max_context_tokens == 0) {
        return "paging pre-allocation: max_context is 0, so there is no declared context to cover";
    }
    if (in.unload_watermark_pages == kPagingWatermarkDerive) {
        return "paging pre-allocation: --kv-unload-watermark-pages is still the derive sentinel "
               "(kUnloadWatermarkDerive); resolve it against the plan's prefill chunk before "
               "sizing a pool that has to be able to fall below it";
    }
    return std::string();
}

// The refusal, as text: "" when admitted. Every branch names what is missing AND the knob that
// provides it, because a capacity refusal without its arithmetic is what sends the reader guessing
// (the same reason layouts_impl.h:267-269 prints its bounds).
[[nodiscard]] inline std::string
paging_preallocation_refusal(const PagingPreallocationInputs& in) {
    const std::string input_refusal = paging_preallocation_input_refusal(in);
    if (!input_refusal.empty()) { return input_refusal; }
    // With NO cold medium the declared-context floor IS the plan: nothing leaves the pool, so there
    // is nothing to refuse. This is the default path and it must not throw -- a header that refused
    // every run without --cold-policy would be a behaviour change, not a capacity check. The loud
    // refusal for a caller that names a SMALLER pool anyway than it can back lives in
    // paging_explicit_capacity_refusal() below, and that is the guard the flash-next target needs.
    if (in.medium == PagingColdMedium::None) { return std::string(); }
    const std::uint32_t declared =
        declared_context_floor_pages(in.max_context_tokens, in.page_tokens, in.max_concurrency);
    const std::uint32_t resident = paging_resident_pages(in);
    if (resident >= declared) { return std::string(); }
    const std::uint32_t required = declared - resident;
    switch (in.medium) {
    case PagingColdMedium::None:
        // Unreachable: returned above. Kept so the switch stays exhaustive over the enum.
        return std::string();
    case PagingColdMedium::Disk:
    case PagingColdMedium::HostThenDisk:
        if (in.cold_page_record_bytes == 0) {
            return "paging pre-allocation: --cold-policy " +
                   std::string(paging_cold_medium_name(in.medium)) +
                   " needs the BYTES PER COLD PAGE -- the sum over slot-bearing layers of "
                   "(stride x kv_heads x 2), i.e. turn_recall_page_bytes() "
                   "(program_impl.h:12839-12855), the engine's own `bytes_per_record` -- to size "
                   "--cold-disk-bytes into pages, and it was not supplied; guessing a per-page "
                   "record would be a silent approximation. NOTE THE UNIT: one layer's stride "
                   "(" + std::to_string(kPagingColdStrideInt8RawBytes) + " B raw / " +
                   std::to_string(kPagingColdStrideRansBytes) + " B rANS) is NOT a page";
        }
        break;
    case PagingColdMedium::PinnedHost:
        if (in.host_page_bytes == 0) {
            return "paging pre-allocation: --cold-policy host needs HostKVPageLayout::page_stride "
                   "to size --cold-host-bytes into pages, and it was not supplied";
        }
        break;
    case PagingColdMedium::DeviceWindow:
        break;
    }
    const std::uint32_t capacity = paging_backing_capacity_pages(in);
    if (capacity >= required) { return std::string(); }
    const std::uint64_t shortfall = static_cast<std::uint64_t>(required) - capacity;
    // The page range that has no home, spelled out, because "not enough memory" is not an answer.
    const std::uint32_t first_uncovered = resident + capacity;
    std::string out = "paging pre-allocation: the backing store cannot hold what the pool no "
                      "longer holds -- pages [" + std::to_string(first_uncovered) + ", " +
                      std::to_string(declared) + ") (of " + std::to_string(declared) +
                      " at " + std::to_string(in.max_context_tokens) + " tokens / " +
                      std::to_string(in.page_tokens) + " tokens per page) would have neither a "
                      "device replica nor a cold tier";
    out += "; resident=" + std::to_string(resident) + " pages, required backing=" +
           std::to_string(required) + ", backing capacity=" + std::to_string(capacity) +
           " (" + std::string(paging_cold_medium_name(in.medium)) + "), short by " +
           std::to_string(shortfall) + " pages";
    if (in.cold_page_record_bytes != 0) {
        out += " = " + std::to_string(shortfall * in.cold_page_record_bytes) + " B of spill";
    }
    switch (in.medium) {
    case PagingColdMedium::Disk:
    case PagingColdMedium::HostThenDisk:
        out += "; a disk page is admitted only with BOTH --cold-disk-bytes room AND a device cold "
               "slot (product/kv_cold_tier_budget.h:136-137), so raise --max-cold-pages to ";
        out += std::to_string(required);
        out += " (or above " + std::to_string(first_uncovered) + ") and --cold-disk-bytes to at "
               "least " +
               std::to_string(static_cast<std::uint64_t>(required) * in.cold_page_record_bytes) +
               " B (" + std::to_string(required) + " pages at " +
               std::to_string(in.cold_page_record_bytes) + " B/page)";
        break;
    case PagingColdMedium::PinnedHost:
        out += "; raise --cold-host-bytes to at least " + std::to_string(required) +
               " host pages";
        break;
    case PagingColdMedium::DeviceWindow:
        out += "; raise --max-cold-pages to " + std::to_string(required);
        break;
    case PagingColdMedium::None:
        break;
    }
    out += ". Refusing rather than reserving a pool that covers less than the declared context.";
    return out;
}

// An EXPLICIT pool request (`--kv-capacity N`, or any caller naming a page count directly) is the
// other way to size below the declared context, and it needs the same guard. The rule is the same
// one the two floors use:
//
//   a pool of R pages is sound  <=>  R covers the resident set
//                               AND  R + (what the backing store can hold) >= declared
//
// With no backing store armed this reduces to "R must be the declared context", which is the
// flash-next target's situation today: its capacity curve never reads `options.kv_capacity` at all
// (runtime_plan.cpp:213-225 takes the floor from max_context and :249-254 returns it in the
// curve), so adding the knob there without this check would let a caller name a pool that silently
// cannot cover the run. Returns "" when the request is admitted.
[[nodiscard]] inline std::string
paging_explicit_capacity_refusal(const PagingPreallocationInputs& in,
                                 std::uint32_t requested_pages) {
    const std::string input_refusal = paging_preallocation_input_refusal(in);
    if (!input_refusal.empty()) { return input_refusal; }
    const std::uint32_t declared =
        declared_context_floor_pages(in.max_context_tokens, in.page_tokens, in.max_concurrency);
    if (requested_pages == 0) {
        return "paging pre-allocation: an explicit pool of 0 pages is not a capacity";
    }
    if (declared == 0) {
        return "paging pre-allocation: max_context is 0, so there is no declared context to cover";
    }
    if (requested_pages >= declared) { return std::string(); }
    const std::uint32_t resident = paging_resident_pages(in);
    if (requested_pages < resident) {
        return "paging pre-allocation: an explicit pool of " + std::to_string(requested_pages) +
               " pages is below the resident set of " + std::to_string(resident) +
               " pages (keep=" + std::to_string(paging_page_count(in.cold_keep_tokens, in.page_tokens)) +
               " + round=" + std::to_string(paging_page_count(in.prefill_chunk, in.page_tokens)) +
               " + watermark=" + std::to_string(in.unload_watermark_pages) +
               "), so one round of work could not be held; raise the pool or lower "
               "--kv-unload-watermark-pages / --prefill-chunk";
    }
    const std::uint32_t required = declared - requested_pages;
    const std::uint32_t capacity = paging_backing_capacity_pages(in);
    if (capacity >= required) {
        if (requested_pages <= in.unload_watermark_pages) {
            return "paging pre-allocation: an explicit pool of " + std::to_string(requested_pages) +
                   " pages cannot fall below the unload watermark of " +
                   std::to_string(in.unload_watermark_pages) +
                   " pages, so the cold tier would retire nothing";
        }
        return std::string();
    }
    // Diagnosable, not merely loud: WHAT was asked (pages, and the tokens/context that produced
    // them), WHAT could not be provided (the page range and the byte shortfall), WHY (which medium
    // is or is not configured, and which expression sized it), and WHICH knob fixes it. A reader
    // who cannot tell from the message whether to raise the capacity, drop the context or configure
    // cold storage has not been told enough.
    std::string out = "paging pre-allocation: --kv-capacity ";
    out += std::to_string(static_cast<std::uint64_t>(requested_pages) * in.page_tokens);
    out += " (" + std::to_string(requested_pages) + " pages of " + std::to_string(in.page_tokens) +
           " tokens) was asked for a declared context of " + std::to_string(in.max_context_tokens) +
           " tokens, which is " + std::to_string(declared) +
           " pages (max(ceil(max_context/page_tokens), max_concurrency))";
    if (in.medium == PagingColdMedium::None) {
        out += "; NO backing store is configured (no --cold-policy), so pages [" +
               std::to_string(requested_pages) + ", " + std::to_string(declared) +
               ") would have neither a device replica nor a cold tier";
    } else {
        out += "; pages [" + std::to_string(requested_pages + capacity) + ", " +
               std::to_string(declared) + ") would have neither a device replica nor a cold tier "
               "(" + std::string(paging_cold_medium_name(in.medium)) + " holds " +
               std::to_string(capacity) + ")";
    }
    out += "; needs " + std::to_string(required) + " pages of backing (declared " +
           std::to_string(declared) + " - pool " + std::to_string(requested_pages) +
           "), backing can hold " + std::to_string(capacity) + ", short by " +
           std::to_string(required - capacity) + " pages";
    if (in.cold_page_record_bytes != 0) {
        out += " = " + std::to_string(static_cast<std::uint64_t>(required - capacity) *
                                      in.cold_page_record_bytes) + " B";
    }
    out += ". Either use --kv-capacity " +
           std::to_string(static_cast<std::uint64_t>(declared) * in.page_tokens) +
           " (or more), or declare a context of at most " +
           std::to_string(static_cast<std::uint64_t>(resident) * in.page_tokens) + " tokens, or arm "
           "a backing store that can hold the difference.";
    return out;
}

// The plan. Throws on every refusal above -- it cannot return a pool that quietly covers less than
// the declared context, which is the whole reason it exists as a function and not as an expression
// at the call site.
[[nodiscard]] inline PagingPreallocation
plan_paging_preallocation(const PagingPreallocationInputs& in) {
    PagingPreallocation plan;
    // Input invariants FIRST: see paging_preallocation_input_refusal for why the order matters.
    const std::string input_refusal = paging_preallocation_input_refusal(in);
    if (!input_refusal.empty()) { throw std::invalid_argument(input_refusal); }
    plan.declared_pages =
        declared_context_floor_pages(in.max_context_tokens, in.page_tokens, in.max_concurrency);
    if (plan.declared_pages == 0) {
        throw std::invalid_argument(
            "paging pre-allocation: declared context is 0 pages, so there is nothing to size");
    }

    plan.keep_pages = paging_page_count(in.cold_keep_tokens, in.page_tokens);
    plan.round_pages = paging_page_count(in.prefill_chunk, in.page_tokens);
    plan.watermark_pages = in.unload_watermark_pages;

    // No cold medium: the declared-context floor, byte-for-byte as today. Nothing may leave the
    // pool, so nothing may be removed from it.
    if (in.medium == PagingColdMedium::None) {
        plan.paging_active = false;
        plan.resident_pages = plan.declared_pages;
        plan.required_backing_pages = 0;
        plan.backing_capacity_pages = 0;
        plan.device_pool_pages = plan.declared_pages;
        plan.device_cold_slot_pages = 0;
        plan.total_device_pages = plan.declared_pages;
        return plan;
    }

    plan.paging_active = true;
    plan.resident_pages = paging_resident_pages(in);
    plan.device_cold_slot_pages = paging_device_cold_slot_pages(
        in.cold_keep_tokens, in.page_tokens, in.medium, in.max_cold_pages);

    // The old path already fits inside the resident set: shrink nothing. This is the equivalence
    // case, and it is why a short context keeps today's pool exactly.
    if (plan.resident_pages >= plan.declared_pages) {
        plan.resident_pages = plan.declared_pages;
        plan.required_backing_pages = 0;
        plan.backing_capacity_pages = paging_backing_capacity_pages(in);
        plan.device_pool_pages = plan.declared_pages;
        plan.total_device_pages = plan.declared_pages + plan.device_cold_slot_pages;
        return plan;
    }

    plan.required_backing_pages = plan.declared_pages - plan.resident_pages;
    plan.backing_capacity_pages = paging_backing_capacity_pages(in);
    const std::string refusal = paging_preallocation_refusal(in);
    if (!refusal.empty()) { throw std::invalid_argument(refusal); }

    plan.device_pool_pages = plan.resident_pages;
    plan.total_device_pages = plan.device_pool_pages + plan.device_cold_slot_pages;

    // A pool whose whole usable size is the watermark can never fall below it without first
    // emptying, so the unload trigger would never fire and nothing would ever be retired. The
    // record already paid for this once: a pool that never binds retires nothing and the criterion
    // reads zero (research/notes/TODO.md:6096-6099, INDEXMTP REPORT.md section 3.2). Refuse the
    // shape instead of shipping a run whose cold tier is pure overhead.
    if (plan.device_pool_pages <= plan.watermark_pages) {
        throw std::invalid_argument(
            "paging pre-allocation: a device pool of " + std::to_string(plan.device_pool_pages) +
            " pages cannot fall below the unload watermark of " +
            std::to_string(plan.watermark_pages) +
            " pages, so --kv-unload-watermark-pages could never fire and the cold tier would be "
            "overhead that retires nothing; lower the watermark, raise --cold-keep-tokens, or "
            "declare a larger context");
    }
    return plan;
}

} // namespace ninfer::product
