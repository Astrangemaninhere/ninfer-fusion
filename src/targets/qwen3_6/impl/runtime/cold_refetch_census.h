#pragma once

// [COLD-HOST REFETCH CENSUS] -- the number the recall window's whole economy rests on, and
// which the tree could count NOWHERE before this header existed.
//
// THE CLAIM BEING MEASURED, quoted from the file that makes it, verbatim:
//
//   src/targets/qwen3_6/impl/runtime/cold_host_tier.h:239
//       "//   re-fetch of an evicted page: ZERO, and that is the entire point of the window."
//
// That line is load-bearing. A window that admits pages attention will read again pays a fresh
// H2D of every page it admitted EVERY round -- the arm's measured shape, "227 pages x 1.127 MiB =
// 256 MiB/round ... against an unpaged round of 26.9 ms" (cold_host_tier.h:227-231). The window
// is economical ONLY because it evicts pages no kernel reads again, so the same page is never
// fetched twice. Until this header, "ZERO" was, exactly as `hook_ms` was before it, an
// unmeasured number standing in for a measurement: `turn_recall_counters` has no refetch field
// (program.h:1062-1075), `ColdTierCounters` says in as many words that restores cannot live in
// it ("Restores are NOT here on purpose: a Host-tier page comes back through the engine's
// materialization path, which the tier never sees, so counting them here would be a lie",
// product/kv_cold_tier_budget.h:69-72), and nothing anywhere counted the event.
//
// WHAT A REFETCH IS, IN THIS CODEBASE. Precise, and there is exactly ONE site:
//
//   A refetch is a page that the Cold Host tier evicted -- its device replica released by
//   `LogicalKVPageStore::transfer_to_cold_host` (logical_kv_store.h:835), the surviving replica
//   living in the tier's own pinned arena -- being given a device replica AGAIN by the engine's
//   materialization path: `ProgramImplCore::prepare_materialization` (program_impl.h:5219)
//   -> its `prepare_kv_restores` lambda (program_impl.h:5411) -> the host branch
//   -> `pages.reserve_device_replica(logical, reservation)` (program_impl.h:5464), which is the
//   ONLY call site of `reserve_device_replica` in the whole tree, and the page is then handed to
//   `enqueue_materialization_transfers` (program_impl.h:5634) whose `copy_from_host` (:5660) is
//   the H2D byte mover.
//
//   The decision to refetch is therefore not a decision at all: it is the loop
//   `for (page = 0; page < mapped; ++page)` in `prepare_kv_restores` restoring EVERY page of the
//   mapped range that has no device replica. A tier-evicted page inside that range IS refetched,
//   by construction, with no predicate consulting the window, the plan or the read-free gate.
//
// COST OF ONE REFETCH: `host_page_bytes` (1.127 MiB measured at the paged 1M shape, i.e. one
// full text page across all layers) of H2D per page, plus the same bytes read from pinned host
// memory. At the arm's own measured bandwidth (~20 GB/s, cold_host_tier.h:228-230) that is
// ~56 us per page; 227 pages is the 256 MiB / ~13 ms per round the arm pays.
//
// WHY A COUNTER AND NOT A PRINT. The site is a per-page loop inside a materialization, and this
// tree has already measured what an unconditional fprintf costs there (0.932 ms/call, 1001
// lines/1000 calls, `cold_fallback_census.h`'s own header note). So:
//
//   * increment cost is one integer add per refetched page, on a path that is already doing a
//     device-page reservation and an asynchronous H2D of 1.127 MiB -- orders of magnitude below
//     the work it counts;
//   * the report is emitted ONCE per process, from `ProgramImplCore`'s destructor, BESIDE the
//     cold-fallback census that already rides there (program_impl.h:1565-1580) -- not per round,
//     not per page, not per pass;
//   * a run in which nothing is ever refetched prints NOTHING AT ALL -- `format()` returns 0 and
//     writes not one byte when `any()` is false, and the call site is additionally guarded, so a
//     clean run stays quiet and silence means "the window's claim held for this run".
//
// THE PAIR IS THE POINT. `[cold-tier] host=... disk=... stay_hot=... released=...` counts pages
// going OUT (the eviction side, product/kv_cold_tier_budget.h). This census counts pages coming
// BACK. Divide one by the other and "zero refetch" stops being a sentence in a comment and
// becomes a ratio a run reports, or a run stays silent -- and it is the SECOND of those that a
// reader can check against the per-round traffic the window promises.
//
// HONESTY NOTE ON EXERCISE (the shape of the remaining gap, said out loud). This header is pure
// C++ -- no CUDA, no engine headers -- so its arithmetic and its reporting contract can be
// exercised with plain g++ at -O0 and -O3 -DNDEBUG (dl/refcnt/probe_refetch_*.cpp). The RATE
// needs a windowed stack (`cold_host_layers_are_windowed` is false on qwen3_6, which declares
// window 0 on every layer, so the tier is correctly inert and NOTHING is evicted to refetch), a
// `--cold-policy host` arm, and a GPU. Landing this header without that run leaves the rate
// UNMEASURED, and the counter then stands as landed-but-unexercised -- which is strictly better
// than absent, because the next windowed run either prints a refetch count or prints nothing
// while a reader knows what that silence would have had to hide.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

// A feature marker for a host test that must compile on BOTH SIDES of this header's landing:
// the PRE image has no such symbol, so a probe asks for the census through this marker instead
// of through a name that exists on only one side.
#define NINFER_COLD_HOST_REFETCH_CENSUS 1

namespace ninfer::targets::qwen3_6::cold_host_refetch {

struct Census {
    std::uint64_t events = 0;  // materializations that refetched at least one page
    std::uint64_t pages  = 0;  // pages refetched out of the Cold Host tier
    std::uint64_t bytes  = 0;  // pages x host_page_bytes, the H2D traffic a window predicted as 0

    [[nodiscard]] bool any() const noexcept { return pages != 0; }

    // One line, and NOTHING AT ALL when the count is zero: `out` is left empty and 0 is
    // returned, so a caller that prints unconditionally still prints nothing meaningful -- the
    // same contract the cold-fallback census states, enforced here as well as at the call site.
    [[nodiscard]] std::size_t format(char* out, std::size_t cap) const noexcept {
        if (out == nullptr || cap == 0) { return 0; }
        out[0] = '\0';
        if (!any()) { return 0; }
        const int used = std::snprintf(
            out, cap,
            "[cold-refetch] refetched_pages=%llu in_events=%llu bytes=%llu -- the window's "
            "\"re-fetch of an evicted page: ZERO\" claim was FALSE for this run "
            "(cold_host_tier.h:239)",
            static_cast<unsigned long long>(pages), static_cast<unsigned long long>(events),
            static_cast<unsigned long long>(bytes));
        if (used < 0) {
            out[0] = '\0';
            return 0;
        }
        const std::size_t len = static_cast<std::size_t>(used);
        if (len >= cap) { return cap - 1; }
        return len;
    }

    // std::string convenience for host tests. NOT for the noexcept call site.
    [[nodiscard]] std::string describe() const {
        char buf[512];
        const std::size_t len = format(buf, sizeof(buf));
        return std::string(buf, len);
    }
};

// Process-cumulative. Namespace scope, not function-local, because the report is emitted by
// ProgramImplCore's destructor -- a different member function from the pass that increments it.
// `inline` so every TU that includes it shares one instance under C++17+ semantics.
inline Census g_census{};

// THE ONE INCREMENT ENTRY POINT. Called from the single site that re-materializes a page whose
// only replica is the Cold Host tier's (program_impl.h, `prepare_kv_restores`). `pages` is 1 per
// refetched page; `host_page_bytes` is the tier's own per-page stride
// (`ColdHostTier::budget().host_page_bytes`), so the byte total is the tier's number and not a
// second, drifting one.
inline void note_refetch_pages(std::uint32_t pages, std::uint64_t host_page_bytes) noexcept {
    if (pages == 0) { return; }
    g_census.pages += pages;
    g_census.events += 1;
    g_census.bytes += static_cast<std::uint64_t>(pages) * host_page_bytes;
}

} // namespace ninfer::targets::qwen3_6::cold_host_refetch
