#pragma once

// Cold Host tier: the eviction consumer behind `--cold-policy host`.
//
// WHAT IT CONSUMES  device KV pages that no attention kernel can read any more
//                   (`cold_host_page_is_read_free`), one pinned Host extent per
//                   page, bounded by `--cold-host-bytes`.
// WHAT IT PRODUCES  the release of those pages' device replicas (real device
//                   memory back into the pool) and, consequently, a page whose
//                   only replica is on the host.
// HOW IT RESTORES   through the engine's existing Host->Device materialization
//                   path. A page with no device replica and a current Host
//                   replica is exactly the "HostOnly replica" state that
//                   materialization already plans (program_impl.h:
//                   prepare_kv_restores counts `pages.host_resident`) and
//                   executes, so the tier needs no read-back path, no sentinel
//                   block-table entry and no device cold slot.
//
// WHY READ-FREE IS THE GATE, not page age (which is what the slot tiers use):
// the Window/Disk cold pools stay READABLE -- attention decodes their pages
// inline from device cold slots (ops/softmax_attention/dense/causal_cache/
// small_t.cu reads cache.cold_slots + cache.cold_slot_valid), which is why they
// reserve device slots at all. A Host page has no device slot, so it may only be
// evicted when no kernel will ever look at it again. On a full-attention layer
// every committed token is re-read on every decode round, so no page is ever
// read-free; with a sliding window W on every layer, page p (tokens
// [p*P, (p+1)*P)) is unread once (p+1)*P + W <= frontier. A stack with any
// full-attention layer therefore has no read-free page, and the tier must say so
// loudly instead of reserving host memory it can never use -- which is the real
// reason `effective_cold_pages(Host)` returns 0 (layouts_impl.h:124-141): the
// device cold-slot pool is the *slot* tiers' resource, and 0 is the correct
// device-side consequence of a Host tier, not a placeholder.
//
// GRANULARITY: one logical page ACROSS ALL TEXT LAYERS, never a slice of one. A
// page is the block table's addressing unit and the Host replica's unit
// (HostKVPageLayout packs every text layer's planes for one page end to end), and
// the tiers below follow the same rule: a cold-slot sentinel encodes ONE slot base
// and the attention kernels index that slot number in every layer's cold_slots at
// once, so a device page can only be demoted whole-slot x all layers (H1). The
// tier therefore has no per-layer and no per-head eviction notion at all: one page
// is one residency bit, which is exactly the state materialization restores.
//
// The byte budget itself (the cap, the ladder, the page arithmetic) lives in
// product/kv_cold_tier_budget.h; this class only owns the tier's resources and
// its counters, so there is exactly one definition of "full".

#include "core/host_kv_arena.h"
#include "core/shard_rank_axis.h"
#include "product/kv_cold_tier_budget.h"
#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// The tier's unit is one Paged-KV page, the same unit the slot tiers count (the
// keep-tokens arithmetic in program_impl.h and the DP's cold capacity both use
// it). Named here so the read-free predicate's `page_tokens` argument cannot be
// silently fed a different granularity.
inline constexpr std::uint32_t kColdHostPageTokens = 64;
static_assert(kColdHostPageTokens == static_cast<std::uint32_t>(kPagedKVPageSize),
              "the Cold Host tier must use the Paged KV page as its unit");

// A page is read-free when every layer's sliding window has already moved past
// it: layer L attends [frontier+1-W_L, frontier], so page p is unread by L iff
// (p+1)*P + W_L <= frontier. `layer_windows` holds the first `layers` entries of
// the decoder's per-layer table (the remaining entries of that fixed-size array
// are padding and must not be passed here). A layer with window 0 is full
// attention and reads every committed token, so it makes the predicate false for
// every page. The `+ W` form keeps one token of slack: eviction has to be
// impossible, not merely unlikely.
[[nodiscard]] inline bool
cold_host_page_is_read_free(std::uint32_t page, std::uint32_t page_tokens,
                            std::uint32_t frontier,
                            std::span<const std::uint32_t> layer_windows) noexcept {
    if (page_tokens == 0 || layer_windows.empty()) { return false; }
    const std::uint64_t page_end = (static_cast<std::uint64_t>(page) + 1U) * page_tokens;
    for (const std::uint32_t window : layer_windows) {
        if (window == 0) { return false; }
        if (page_end + window > frontier) { return false; }
    }
    return true;
}

// True when the tier can ever admit a page on this model. A false answer means
// --cold-policy host (and any --cold-host-bytes) is structurally inert, which the
// program reports once at construction.
[[nodiscard]] inline bool
cold_host_layers_are_windowed(std::span<const std::uint32_t> layer_windows) noexcept {
    if (layer_windows.empty()) { return false; }
    return std::all_of(layer_windows.begin(), layer_windows.end(),
                       [](std::uint32_t window) { return window != 0; });
}

// ===========================================================================
// THE SECOND ADMISSION ARM: A PAGE THAT IS READ, BUT FETCHED ON DEMAND
// ===========================================================================
//
// WHY READ-FREE IS NOT THE ONLY SOUND GATE, stated as the capability it depends on.
// Read-free is sound because the tier owns no read-back path: a Host page has no device
// slot, so no attention kernel can decode it inline. But the ENGINE owns a materialization
// path for a page whose only replica is on the host -- `prepare_kv_restores` counts
// `pages.host_resident` and runs at the round boundary, BEFORE the kernels -- so a page may
// ALSO leave the device when something is guaranteed to fetch it back before any kernel
// reads it. That is a ROUND-SCOPED read-free property, and it is exactly what the P5/P6
// work creates: `plan_round_recall` names the pages a round will restore and
// `recall_cold_pages_for_round` restores them at the same round boundary
// `ensure_sequence_kv_mapped` runs at.
//
// The plan is passed as a SORTED span, because that is the order `plan_recall_pages`
// returns and a binary search is the only lookup this predicate may do on a per-page path.
[[nodiscard]] inline bool
cold_host_page_is_fetchable(std::uint32_t page, std::uint32_t page_tokens,
                            std::uint32_t frontier,
                            std::span<const std::uint32_t> layer_windows,
                            std::span<const std::uint32_t> round_plan_pages) noexcept {
    if (cold_host_page_is_read_free(page, page_tokens, frontier, layer_windows)) { return true; }
    return std::binary_search(round_plan_pages.begin(), round_plan_pages.end(), page);
}

// WHAT THIS ARM COSTS, AND ITS CEILING ON A FULL-ATTENTION STACK -- the honest arithmetic,
// because the arm is real but SMALL and must not be mistaken for the 6 GiB the 1M study is
// short of.
//
// Every text layer of qwen3_6 is full attention (window 0), so every committed page is read
// EVERY round. A round plan that names a page therefore needs that page's bytes back in the
// SAME round, so a host page admitted under this arm is materialized again on the next round:
// the saving is real only for pages that LEAVE the plan. The plan is bounded by
// `recall_fanout_blocks()` and `turn_recall_byte_budget` (256 MiB by default, i.e. ~218 pages
// at 1.17 MiB), so on this model
//
//     saved_device_bytes <= turn_recall_byte_budget          = 256 MiB,
//
// against the ~6 GiB INDEX1M2 sec.5.5 measures missing at 1,010,000 tokens. IT DOES NOT CLOSE
// THAT GAP. Nor would an unbounded plan: on full attention an unbounded plan IS the whole
// prefix, which turn_recall_journal.h measures at 18.00 GiB/step = 0.36 tok/s and FORBIDS by
// name. A page evicted under this arm and re-materialized next round also pays a fresh H2D of
// its own bytes every round, which is a NEW fixed per-round cost -- so a caller arming this
// arm must measure it rather than assume it is free.
//
// WHERE IT IS ACTUALLY LARGE: on a stack whose attended span is bounded. There, read-free
// already admits the pages outside the window without any of this, and this arm adds only the
// pages inside the window that a bounded plan can cover. So the arm is a real capability for a
// windowed model and a <=256 MiB capability on this one; the gap below is closed by a WINDOW,
// not by this predicate.
[[nodiscard]] inline bool
cold_host_tier_can_admit(std::span<const std::uint32_t> layer_windows,
                         bool bounded_round_plan_available) noexcept {
    return cold_host_layers_are_windowed(layer_windows) || bounded_round_plan_available;
}

// ===========================================================================
// THE WINDOW: WHAT ADMITS ENOUGH PAGES TO REACH 1M, AND WHY IT IS NOT THE ARM ABOVE
// ===========================================================================
//
// The arm above admits a page the round WILL read, by naming it in the plan. That works, and it
// is small: the plan is bounded by `turn_recall_byte_budget`, so the arm's ceiling on a
// full-attention stack is 256 MiB = 227 pages = 1.4% of the 15,782 pages 1M needs. The rest of
// the gap is closed by a WINDOW -- a page the window has passed is read by NOBODY, so it is
// read-free with no plan, no fetch and no new predicate. The predecessor established that and
// stopped there; what follows is the arithmetic that says HOW BIG the window has to be and what
// it costs, so "a window closes it" stops being a claim and becomes a number.
//
// Everything below is in PAGES of kColdHostPageTokens, because that is the tier's unit and the
// plan's unit, so page accounting and byte accounting stay the same accounting.
//
// ---------------------------------------------------------------------------
// 1. THE BAND. Four quantities, one inequality, and the answer is a SPAN not a size.
// ---------------------------------------------------------------------------
//
//   F  frontier_pages   pages the sequence has committed        15,782 at 1,010,048 tokens
//   D  device_pages     device KV capacity, in these pages      11.17 GiB  -> 10,148
//   H  host_pages       --cold-host-bytes, in these pages       4 GiB      ->  3,637
//   K  plan_pages       the round's plan (P6's output)          43, measured
//   W  window_pages     pages kept resident because they are READ
//
// A page may leave the device only when NOTHING WILL READ IT, and there are exactly two
// disjoint ways to qualify:
//
//   (i)  the window has passed it -- read-free, `cold_host_page_is_read_free` admits it today;
//   (ii) the round's plan names it -- fetched back at the round boundary,
//        `cold_host_page_is_fetchable` admits it.
//
// So the resident set is W+K, and the host holds F-(W+K). Both must fit:
//
//     W + K <= D            the resident set fits the DEVICE
//     F - W - K <= H        everything else fits the HOST
//   =>
//     max(0, F - K - H) <= W <= D - K
//
// which is feasible IFF
//
//     K <= D   and   F <= D + H
//
// ⭐ Note the SHAPE of that condition. `F <= D + H` is a SPAN condition, exactly as the
// predecessor concluded, and it does not mention K at all: the plan does not make the gap
// closable, it only shifts pages between the two media. The plan's role is to let W be SMALLER
// than F-D (a page the plan names does not need the window to cover it), and that is the whole
// of its contribution to feasibility.
//
// Solving the band for 1M at the SHIPPED defaults, on the numbers above:
//
//     K = 43 <= D = 10,148                           OK
//     F = 15,782 > D + H = 10,148 + 3,637 = 13,785   NOT FEASIBLE
//     short by 1,997 pages = 2.2 GiB
//     W_min = F - K - H = 12,102   >   W_max = D - K = 10,105     (the band is EMPTY)
//
// and the shortfall is HOST capacity, not device: the device is already carrying the most it
// can. The host requirement is
//
//     H_needed = F - D = 15,782 - 10,148 = 5,634 pages
//
// ---------------------------------------------------------------------------
// 2. WHAT THE WINDOW COSTS, IN BYTES, PER ROUND -- and why it is not the arm's cost
// ---------------------------------------------------------------------------
//
// The arm above pays a fresh H2D of EVERY page it admitted, EVERY round, because on full
// attention those pages are read again immediately: 227 pages x 1.127 MiB = 256 MiB/round, which
// is ~13 ms at this box's ~20 GB/s, against an unpaged round of 26.9 ms. That is the cost of
// admitting pages that ARE read.
//
// A WINDOW admits pages that are NOT read, so the same page is never fetched twice:
//
//   evictions   a decode round advances the frontier by ONE token, so it pushes at most
//               ceil(1/64) = one page past the window every 64 rounds and pays
//               1,181,745 / 64 = 18,465 B ~ 18 KiB of D2H per round;
//   recalls     K pages, and ONLY on a round whose plan is non-empty. A steady-state round's
//               plan is EMPTY by construction -- `plan_round_recall`'s own note: "recall the
//               prefix every round" is the 18.00 GiB/step shape the journal forbids by name --
//               so this is a per-EVENT cost, not a per-round one. 43 pages = 48.5 MiB ~ 2.4 ms.
//   re-fetch of an evicted page: ZERO, and that is the entire point of the window.
//
// So the window's steady-state per-round traffic is ~18 KiB against the arm's 256 MiB -- a factor
// of ~14,000 -- and the window reaches 5,634 host pages where the arm reaches 227.
//
// ---------------------------------------------------------------------------
// 3. WHAT THIS CANNOT DO, STATED SO NOBODY ASSUMES OTHERWISE
// ---------------------------------------------------------------------------
//
// ⚠️ A WINDOW IS AN ATTENTION PATTERN, AND DECLARING ONE HERE DOES NOT CREATE ONE IN THE
// KERNELS. `cold_host_page_is_read_free` is sound because the model's own config says every
// layer's window is smaller than the frontier (`layer_windows`, from
// `TextConfig::is_swa_attention(i) ? sliding_window : 0`). qwen3_6 declares window 0 on every
// layer, i.e. full attention, so no page is read-free and the tier is correctly inert. A
// caller that wants the band above to be NON-EMPTY on this stack has to make the kernels stop
// reading at `window_pages * kColdHostPageTokens` behind the frontier -- which is a change to
// attention, is not in this header, and is not something a predicate can assert its way to.
//
// That is why the window-free test below is named `_declared` and takes the window as an
// ARGUMENT rather than reading a flag: a caller cannot get a true answer out of it without
// writing down the number the kernels must honour, in the same place, at the same time.

// The band, resolved. `min_window_pages > max_window_pages` is how an empty band reads, and
// `feasible()` is the whole condition rather than a restatement of one inequality.
struct ColdHostWindowBand {
    std::uint32_t frontier_pages   = 0;
    std::uint32_t device_pages     = 0;
    std::uint32_t host_pages       = 0;
    std::uint32_t plan_pages       = 0;
    std::uint32_t min_window_pages = 0; // max(0, F - K - H): the SMALLEST window that fits the host
    std::uint32_t max_window_pages = 0; // D - K: the LARGEST window the device can hold

    [[nodiscard]] bool feasible() const noexcept {
        return plan_pages <= device_pages && frontier_pages <= device_pages + host_pages;
    }
    [[nodiscard]] bool admits(std::uint32_t window_pages) const noexcept {
        return feasible() && window_pages >= min_window_pages && window_pages <= max_window_pages;
    }
    // The host pages this frontier needs, i.e. what the device CANNOT hold. Independent of the
    // plan and of the window, because it is the `F <= D + H` half of the condition.
    [[nodiscard]] std::uint32_t required_host_pages() const noexcept {
        return frontier_pages > device_pages ? frontier_pages - device_pages : 0U;
    }
    [[nodiscard]] std::uint32_t shortfall_pages() const noexcept {
        const std::uint32_t need = required_host_pages();
        return need > host_pages ? need - host_pages : 0U;
    }
};

[[nodiscard]] inline ColdHostWindowBand
cold_host_window_band(std::uint32_t frontier_pages, std::uint32_t device_pages,
                      std::uint32_t host_pages, std::uint32_t plan_pages) noexcept {
    ColdHostWindowBand band;
    band.frontier_pages = frontier_pages;
    band.device_pages   = device_pages;
    band.host_pages     = host_pages;
    band.plan_pages     = plan_pages;
    const std::uint64_t planned = static_cast<std::uint64_t>(plan_pages) + host_pages;
    band.min_window_pages = frontier_pages > planned
                                ? static_cast<std::uint32_t>(frontier_pages - planned)
                                : 0U;
    band.max_window_pages = device_pages > plan_pages ? device_pages - plan_pages : 0U;
    return band;
}

// WHY AN EMPTY BAND IS A REFUSAL AND NOT A CLAMP, in the tree's own discipline: the two ways to
// "make it fit" are both silent corruption. Clamping the window up OVERRUNS THE DEVICE (the
// resident set no longer fits, so an admission succeeds against a pool that cannot hold it);
// clamping it down OVERRUNS THE HOST (a page is evicted into an arena that has no room, or stays
// hot and the device overruns anyway). There is no third answer, so there is no third behaviour.
[[nodiscard]] inline std::string
cold_host_window_refusal(const ColdHostWindowBand& band) noexcept {
    if (band.plan_pages > band.device_pages) {
        return "the round plan names " + std::to_string(band.plan_pages) +
               " pages but the device holds " + std::to_string(band.device_pages) +
               ": the plan cannot be resident at all, so no window makes it fit";
    }
    if (band.frontier_pages <= band.device_pages + band.host_pages) { return {}; }
    return "the committed prefix is " + std::to_string(band.frontier_pages) +
           " pages against device " + std::to_string(band.device_pages) + " + host " +
           std::to_string(band.host_pages) + " = " +
           std::to_string(band.device_pages + band.host_pages) +
           " pages of cold capacity: a window cannot close this, because the band is empty "
           "(min " + std::to_string(band.min_window_pages) + " > max " +
           std::to_string(band.max_window_pages) + "). Raise --cold-host-bytes by at least " +
           std::to_string(band.shortfall_pages()) +
           " pages; the plan's own byte budget is a different budget and does not help";
}

// ---------------------------------------------------------------------------
// bandfail: THE COLD-HOST WINDOW BAND, NAMED AT THE POINTS THAT DECIDE.
// ---------------------------------------------------------------------------
// `cold_host_window_band` and `cold_host_window_refusal` above ARE the tree's own statement
// of when a window cannot close the gap between the committed prefix and the cold capacity:
// the band is `max(0, F-K-H) <= W <= D-K`, and it is EMPTY exactly when `F > D + H`. The `K`
// term deforms both ends and cancels out of the verdict, which is what the note above says
// in words ("it does not mention K at all: the plan does not make the gap closable, it only
// shifts pages between the two media"). Both functions had ZERO production callers, so an
// operator could reach 1M with a `--cold-host-bytes` that provably cannot hold the `F - D`
// pages the device cannot, and be told nothing until the DEVICE POOL refused -- at a frontier
// shorter than the one declared, under the pool's name instead of the band's, and with no
// mention of the knob that fixes it.
//
// This returns the tree's own refusal text VERBATIM -- including the "Raise
// --cold-host-bytes by at least N pages" quantity -- or an empty string when the band is
// non-empty. It deliberately invents no arithmetic: F from the caller's frontier, D from the
// page pool, H from the tier's own capacity.
//
// ⚠️ THE TWO GATES, AND WHY THEY ARE NOT OPTIONAL (each is a control in
// dl/bandfail/guard/out/controls.txt, and each was MEASURED to change the verdict):
//   1. the policy must have a live HOST rung. Under `none|window|disk` the budget's host
//      tier is disabled, so H is 0 -- and calling the band with H = 0 refuses EVERY 1M
//      serve, including the ones those policies serve today (control C9). That is not a
//      guard, it is a defect.
//   2. the window table must be ALL non-zero. `cold_host_layers_are_windowed` is `all_of`
//      (:42-47) and `cold_host_page_is_read_free` returns false on the first zero (:26-37),
//      so on muse_glimmer_30b (2048 on 39 of 52) or an all-zero table the tier is
//      structurally inert and H is not a capacity that can hold a single page.
// ⚠️ AND IT NEVER READS `W`. The band's *emptiness* is a property of F, D and H alone; a plan
// whose chosen W merely sits outside `[min, max]` while a different W fits is a PLAN-QUALITY
// defect, not infeasibility, and must not be refused here (control C10).
[[nodiscard]] inline std::string
cold_host_band_refusal_at(ColdPolicy policy, std::span<const std::uint32_t> layer_windows,
                          std::uint32_t frontier_pages, std::uint32_t device_pages,
                          std::uint64_t host_bytes, std::uint64_t host_page_stride) {
    if (policy != ColdPolicy::Host && policy != ColdPolicy::HostThenDisk) { return {}; }
    if (!cold_host_layers_are_windowed(layer_windows)) { return {}; }
    const std::uint32_t host_pages =
        host_page_stride == 0
            ? 0U
            : static_cast<std::uint32_t>(
                  product::cold_tier_page_capacity(host_bytes, host_page_stride));
    const ColdHostWindowBand band =
        cold_host_window_band(frontier_pages, device_pages, host_pages, 0U);
    return cold_host_window_refusal(band);
}

// The host bytes the band needs, DERIVED from the device rather than chosen. This is the number
// `--cold-host-bytes` must reach, and it is why that constant is allowed to grow where the
// plan's byte budget is not:
//
//   * `--cold-host-bytes` bounds a pool that can only ever hold pages the WINDOW HAS PASSED, so
//     its ceiling is F-W pages by construction. Raising it cannot admit the whole prefix.
//   * `turn_recall_byte_budget` bounds what a round FETCHES, and an unbounded fetch on full
//     attention IS the whole prefix -- the 18.00 GiB/step shape. It must not move.
[[nodiscard]] inline std::uint64_t
cold_host_bytes_for_window(std::uint32_t frontier_pages, std::uint32_t device_pages,
                           std::uint64_t page_bytes) noexcept {
    const std::uint64_t need = frontier_pages > device_pages ? frontier_pages - device_pages : 0U;
    return need * page_bytes;
}

// A window a FULL-ATTENTION stack can actually have. `cold_host_page_is_read_free` returns false
// for every page when any layer's window is 0, which is every layer of qwen3_6 -- correctly, for
// THAT predicate, which reads the model's declaration. This one takes the window as a parameter
// the caller DECLARES, so a stack whose config has no window can still bound its reads by policy.
//
// ⚠️ SOUND ONLY IF THE KERNELS HONOUR IT. See section 3 above: this is a statement about what
// attention will read, and the caller is the one who knows whether that is true. The name carries
// the precondition on purpose.
[[nodiscard]] inline bool
cold_host_page_is_window_free_declared(std::uint32_t page, std::uint32_t page_tokens,
                                       std::uint32_t frontier,
                                       std::uint32_t window_pages) noexcept {
    if (page_tokens == 0) { return false; }
    // A declared window of 0 is full attention, and the answer is false for every page -- the
    // same answer `cold_host_page_is_read_free` gives, reached from the policy side.
    if (window_pages == 0) { return false; }
    const std::uint64_t window_begin =
        static_cast<std::uint64_t>(window_pages) * page_tokens;
    // The same `+ page_tokens` slack the model-declared form keeps: eviction must be IMPOSSIBLE,
    // not merely unlikely, so the window_end used here is the end of the LAST page the window
    // still covers.
    const std::uint64_t page_end = (static_cast<std::uint64_t>(page) + 1U) * page_tokens;
    return page_end + window_begin <= frontier;
}

// The per-round traffic of a window, in bytes, so "the window is cheap" is a number. Kept as
// arithmetic rather than a measurement because the measurement needs a paged GPU run; what this
// DOES prove is the SHAPE -- the eviction term is one page per `page_tokens` rounds, i.e. it
// does not grow with the window, which is the property the arm above does not have.
struct ColdHostWindowRoundCost {
    std::uint64_t evict_d2h_bytes  = 0; // pages pushed past the window this round
    std::uint64_t recall_h2d_bytes = 0; // the round's plan, zero on a steady-state round
    [[nodiscard]] std::uint64_t total_bytes() const noexcept {
        return evict_d2h_bytes + recall_h2d_bytes;
    }
};

[[nodiscard]] inline ColdHostWindowRoundCost
cold_host_window_round_cost(std::uint32_t plan_pages, std::uint64_t page_bytes,
                            std::uint32_t page_tokens,
                            std::uint32_t tokens_per_round = 1U) noexcept {
    ColdHostWindowRoundCost cost;
    if (page_tokens != 0) {
        // Round UP: the round that crosses a page boundary pays for the whole page, and a
        // fractional-pages-per-round answer would understate the traffic on exactly that round.
        const std::uint64_t crossed =
            (static_cast<std::uint64_t>(tokens_per_round) + page_tokens - 1U) / page_tokens;
        cost.evict_d2h_bytes = crossed * page_bytes;
    }
    cost.recall_h2d_bytes = static_cast<std::uint64_t>(plan_pages) * page_bytes;
    return cost;
}

class ColdHostTier {
public:
    // THE RANK CONTEXT. Defaults are the identity world with a world-total cap, which is what
    // every construction before this parameter existed was -- so the parameter's existence
    // changes no existing behaviour, and `world_shape_refusal()` refuses a malformed one by name.
    //
    // `scope` says whether `budget.host_bytes` bounds the WORLD or this rank alone. It has to be
    // declared because `cold_tier_page_capacity(bytes, page_bytes)` is a division: read a world
    // cap against a per-rank stride and the capacity comes back world_size times too large, read
    // a per-rank cap against the whole stride and it comes back too small. Neither error is
    // visible in a single-device run, and both are silent.
    struct RankContext {
        multi::WorldShape world{};
        multi::BudgetScope scope = multi::BudgetScope::WorldTotal;
    };

    // `budget.host_bytes` is the tier's memory: the arena is created with exactly the
    // per-rank capacity, so the cap holds at the allocator as well as in the arithmetic.
    // `layouts` are the page-store layouts this tier serves -- under a split world they are
    // THIS RANK's layouts, which is why the stride below is already the per-rank stride.
    ColdHostTier(product::ColdTierBudget budget, std::span<const HostKVPageLayout> layouts)
        : ColdHostTier(budget, layouts, RankContext{}) {}

    ColdHostTier(product::ColdTierBudget budget, std::span<const HostKVPageLayout> layouts,
                 RankContext rank)
        : budget_(budget), rank_(rank) {
        const std::string world_refusal = multi::world_shape_refusal(rank_.world);
        if (!world_refusal.empty()) {
            throw std::invalid_argument("ColdHostTier: " + world_refusal);
        }
        if (!budget_.host_tier_enabled || budget_.host_bytes == 0 || layouts.empty()) { return; }
        std::size_t minimum_stride = layouts.front().page_stride;
        for (const HostKVPageLayout& layout : layouts) {
            minimum_stride = std::min(minimum_stride, layout.page_stride);
        }
        if (minimum_stride == 0) { return; }
        per_rank_stride_ = minimum_stride;

        // THE TWO NUMBERS THAT MUST AGREE. The arena divides `host_bytes` by `layouts[].page_stride`
        // while `capacity_pages()` divides `host_bytes` by `budget.host_page_bytes`. When those two
        // strides differ, the tier reports a capacity its own arena does not have -- and under a
        // rank split the difference is exactly the factor of world_size, because one of the pair
        // is the whole page and the other is the rank's share of it.
        //
        // Under a SPLIT world this is a hard refusal: it means the caller mixed the two scopes, and
        // a cold tier that believes it has world_size times its memory will admit pages it cannot
        // hold. Under the identity world it is recorded and reported instead of thrown, because
        // today's single-device behaviour must not change shape because a guard was added.
        if (budget_.host_page_bytes != 0 && budget_.host_page_bytes != minimum_stride) {
            const std::string mixed =
                "the tier's page stride and the budget's page stride disagree: layouts say " +
                std::to_string(minimum_stride) + ", the budget says " +
                std::to_string(budget_.host_page_bytes) +
                (rank_.world.axis == multi::ParallelAxis::None
                     ? " (single-device: reported, not refused)"
                     : " (world_size " + std::to_string(rank_.world.world_size) +
                           ": the pair is a world/per-rank scope mix, so the capacity this tier "
                           "would report is off by a factor of world_size)");
            if (rank_.world.axis != multi::ParallelAxis::None) {
                throw std::invalid_argument("ColdHostTier: " + mixed);
            }
            stride_mismatch_note_ = mixed;
        }

        // THE ONE DIVISION. A world-total cap becomes this rank's cap here and nowhere else.
        const std::uint64_t per_rank_bytes =
            rank_.scope == multi::BudgetScope::WorldTotal
                ? budget_.host_bytes /
                      static_cast<std::uint64_t>(rank_.world.world_size)
                : budget_.host_bytes;
        if (per_rank_bytes == 0) { return; }

        // The arena gets the PER-RANK capacity, so the cap holds at the allocator too. Without
        // this the arena would be built with the whole world's bytes on every rank, i.e. the
        // world would reserve world_size times the cap in pinned host memory.
        arena_ = std::make_unique<HostKVArena>(static_cast<std::size_t>(per_rank_bytes), layouts);
        const std::size_t extent_capacity =
            static_cast<std::size_t>(per_rank_bytes) / minimum_stride;
        if (extent_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Cold Host tier extent capacity exceeds uint32");
        }
        if (extent_capacity == 0) { return; }
        extents_ = std::make_unique<HostKVExtentStore>(
            *arena_, static_cast<std::uint32_t>(extent_capacity));
    }

    // A tier is usable only when it owns both the arena and a non-empty descriptor
    // pool: an enabled policy with a budget that cannot hold one page stays inert
    // (and reports it) rather than failing at the first eviction.
    [[nodiscard]] bool enabled() const noexcept {
        return budget_.host_tier_enabled && extents_ != nullptr;
    }

    [[nodiscard]] HostKVExtentStore* extents() noexcept { return extents_.get(); }

    [[nodiscard]] const product::ColdTierBudget& budget() const noexcept { return budget_; }

    // The page capacity, with the scope applied exactly once. For the identity world with a
    // world-total cap and agreeing strides this is `cold_tier_page_capacity(host_bytes,
    // host_page_bytes)` -- the same number the tier reported before the rank axis existed.
    [[nodiscard]] std::uint32_t capacity_pages() const noexcept {
        const std::uint64_t cap_bytes =
            rank_.scope == multi::BudgetScope::WorldTotal
                ? budget_.host_bytes / static_cast<std::uint64_t>(rank_.world.world_size)
                : budget_.host_bytes;
        const std::uint64_t stride =
            budget_.host_page_bytes != 0 ? budget_.host_page_bytes : per_rank_stride_;
        return multi::page_capacity_from_bytes(cap_bytes, stride);
    }

    // The capacity a caller BRANCHING ON THE RAW BUDGET would compute, i.e. the world's cap read
    // against this rank's stride. Exposed -- rather than merely avoided -- so the factor of
    // world_size is a nameable number in a test and in the operator's log, instead of a silent
    // inflation.
    [[nodiscard]] std::uint32_t inflated_capacity_pages() const noexcept {
        if (per_rank_stride_ == 0) { return 0; }
        return multi::page_capacity_from_bytes(budget_.host_bytes, per_rank_stride_);
    }

    [[nodiscard]] const multi::WorldShape& world() const noexcept { return rank_.world; }
    [[nodiscard]] multi::BudgetScope budget_scope() const noexcept { return rank_.scope; }
    [[nodiscard]] std::size_t per_rank_page_stride() const noexcept { return per_rank_stride_; }
    [[nodiscard]] const std::string& stride_mismatch_note() const noexcept {
        return stride_mismatch_note_;
    }

    // One line per rank context, so a sharded run's log says which rank's tier this is and how
    // its budget was scoped rather than leaving a reader to infer it from the byte counts.
    [[nodiscard]] std::string rank_context_line() const {
        if (rank_.world.axis == multi::ParallelAxis::None) { return {}; }
        std::string out = "cold host tier: rank " + std::to_string(rank_.world.rank) + " of " +
                          std::to_string(rank_.world.world_size) + ", axis " +
                          std::string(multi::axis_name(rank_.world.axis)) + ", cap scope " +
                          std::string(multi::budget_scope_name(rank_.scope)) +
                          ", per-rank page stride " + std::to_string(per_rank_stride_) +
                          ", capacity " + std::to_string(capacity_pages()) + " pages";
        if (!stride_mismatch_note_.empty()) { out += " [" + stride_mismatch_note_ + "]"; }
        return out;
    }

    [[nodiscard]] std::uint32_t used_pages() const noexcept { return usage_.host_pages; }

    [[nodiscard]] product::ColdTierDecision admit() const noexcept {
        return product::admit_cold_page(budget_, usage_);
    }

    void note_admission(product::ColdTier tier) noexcept {
        product::reserve_cold_page(usage_, tier);
        switch (tier) {
        case product::ColdTier::Host: ++counters_.host_admissions; break;
        case product::ColdTier::Disk: ++counters_.disk_admissions; break;
        case product::ColdTier::StayHot: break;
        }
    }

    // The page could not leave the device: counted, never thrown.
    void note_refused() noexcept { ++counters_.stays_hot; }

    // A page's Host replica was dropped (teardown sweep): the bytes are back.
    void note_page_released(std::uint32_t pages) noexcept {
        for (std::uint32_t i = 0; i < pages; ++i) {
            product::release_cold_page(usage_, product::ColdTier::Host);
        }
        counters_.releases += pages;
    }

    [[nodiscard]] const product::ColdTierCounters& counters() const noexcept { return counters_; }

    [[nodiscard]] std::string description() const {
        return product::describe_cold_tier_budget(budget_, capacity_pages(),
                                                  product::cold_tier_page_capacity(
                                                      budget_.disk_bytes, budget_.disk_page_bytes));
    }

    [[nodiscard]] std::string counters_line() const { return counters_.describe(); }

private:
    product::ColdTierBudget budget_{};
    product::ColdTierUsage usage_{};
    product::ColdTierCounters counters_{};
    RankContext rank_{};
    std::size_t per_rank_stride_ = 0;
    std::string stride_mismatch_note_;
    std::unique_ptr<HostKVArena> arena_;
    std::unique_ptr<HostKVExtentStore> extents_;
};

} // namespace ninfer::targets::qwen3_6::detail
