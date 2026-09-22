#pragma once

// [COLD-FALLBACK CENSUS] -- the fallback rate of the cold rANS slot codec, which the
// tree could count nowhere before this header existed.
//
// THE QUESTION THIS ANSWERS. src/targets/qwen3_6/impl/state/decoder_state.cpp's own
// rule is that the rANS ceiling "must be a measured decision, not a default": the
// ceiling (kColdSlotRansBitsPerCodeX100) is a FIXED-STRIDE record, so tightening it
// costs HIT RATE, never correctness -- an rANS stream that would need more than its
// 167 B budget clears the slot's valid flag (entropy_nvfp4_slot_kernels.cuh pass A/B)
// and the page stays hot. The cost of the ceiling is therefore exactly the fraction
// of pages that fail to fit. Until this header, that fraction was counted NOWHERE:
// the only signal in the tree was the `[cold] page N INVALID at layer L` line in
// program_impl.h, which the `validity_logged` guard collapses to one line per layer
// per pass -- by construction untallyable from a log.
//
// WHY A COUNTER AND NOT A PRINT (the shape, and why it is this one).
// The `validity_logged` guard is NOT load-bearing for correctness: it latches a
// vector<bool> and nothing reads that vector. It is load-bearing as a COST brake, and
// this tree has been bitten by exactly that defect class before: an unconditional
// fprintf in a neighbouring area measured 0.932 ms/call and 1001 lines/1000 calls,
// and had to be made edge-triggered. So the guard stays EXACTLY as it is, and this
// header adds a tally that runs BESIDE it:
//
//   * increment cost is one integer add on the failure path, plus one per evaluated
//     page for the denominator -- orders of magnitude below the instrument that feeds
//     it, because the page loop already performs 2 x layers SYNCHRONOUS cudaMemcpy
//     D2H calls per page just to evaluate the predicate;
//   * the report is emitted ONCE per process, from ProgramImplCore's destructor --
//     not per pass, not per page;
//   * a run in which no page ever falls back prints NOTHING AT ALL.
//
// The counter is process-cumulative and lives at namespace scope so the destructor (a
// different member function from the pass that increments it) can read it. It is
// deliberately NOT reported on the per-pass `[spillgate]` line: that line already
// prints once per pass, and appending to it would make the rate converge only if a
// reader diffs consecutive lines. One converged number at teardown is the measurement.
//
// HONESTY NOTE ON EXERCISE. This header is pure C++ (no CUDA, no engine headers) so
// its arithmetic can be exercised with plain g++, in the style of
// src/product/kv_cold_tier_budget.h. The counters themselves need a real encode pass
// on a real stack (a GPU arm). Landing the header without running that arm leaves the
// rate UNMEASURED, and the counters then stand as landed-but-unexercised -- which is
// strictly better than absent, and the command that exercises them is named in
// REPORT.md of the line that landed this.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

// FEATURE MARKERS, for a host test that must compile on BOTH SIDES of the 2026-09-18 meaning
// change (dl/censusrate/cold_fallback_inspected_rate_selftest.cpp). The RED half of that test is
// compiled against this header's PRE image, which defines neither symbol, so it asks for the
// published overflow rate through these markers instead of through a symbol that exists on only
// one side. They are declared here, by the revision that has the symbols, and nowhere else.
#define NINFER_COLD_FALLBACK_CENSUS_INSPECTED_RATE 1
#define NINFER_COLD_FALLBACK_CENSUS_TRUNCATION 1

namespace ninfer::targets::qwen3_6::cold_fallback {

// --- Codec geometry ---------------------------------------------------------
// The three literals the existing `[cold] page N INVALID` print already names, and
// whose authoritative statement is include/ninfer/ops/entropy_nvfp4_slot.h plus the
// slot header layout:
//   * 32 streams per slot, 16 per half-page (kColdSlotRansStreamsPerHalf);
//   * 512 E2M1 symbols per stream (one 16-thread group, 512 code nibbles);
//   * a 320 B slot header holding the per-stream offsets and frequency tables, plus a
//     1024 B uncompressed scale tail.
// None of these is the CEILING. The ceiling is chosen in decoder_state.cpp's
// kColdSlotRansBitsPerCodeX100, and it reaches this header only as `record_bytes`.
inline constexpr std::uint32_t kStreams          = 32;
inline constexpr std::uint32_t kSymbolsPerStream = 512;
inline constexpr std::uint32_t kHeaderBytes      = 320;
inline constexpr std::uint32_t kScaleTailBytes   = 1024;

// Layers are indexed by the pass's own layer ordinal. A stack wider than this is not
// rejected; the per-layer histogram simply stops growing, and the page totals -- which
// are what the rate is computed from -- stay exact.
inline constexpr std::uint32_t kMaxLayers = 128;

// The per-stream byte budget the record implies: (record - header - tail) / streams.
// At the shipped 6688 B record this is 167 B, i.e. ceil(512 * 2.60 / 8) = ceil(166.4).
[[nodiscard]] constexpr std::uint32_t stream_budget_bytes(std::uint32_t record_bytes) noexcept {
    return record_bytes <= kHeaderBytes + kScaleTailBytes
               ? 0
               : (record_bytes - kHeaderBytes - kScaleTailBytes) / kStreams;
}

// The ENFORCED ceiling in millionths of a bit per code, derived from the record:
// budget * 8 / symbols. The record is a byte ceil, so this sits up to 7/8 bit per
// stream ABOVE the nominal ceiling it was derived from -- at 6688 B it is 2.609375
// against a nominal 2.60, i.e. 0.009375 b/c WIDER. The ceil errs safe: a stream is
// refused only when it exceeds the byte budget, so no page is ever refused for a rate
// the nominal ceiling admits.
[[nodiscard]] constexpr std::uint64_t ceiling_x1e6(std::uint32_t record_bytes) noexcept {
    return static_cast<std::uint64_t>(stream_budget_bytes(record_bytes)) * 8ULL * 1000000ULL /
           kSymbolsPerStream;
}

// The same ceiling as a fixed-point hundredth of a bit per code (the unit
// kColdSlotRansBitsPerCodeX100 and tests/test_kv_tier_formats.cpp use), rounded up:
// 2609375e-6 -> 261.
[[nodiscard]] constexpr std::int32_t ceiling_x100(std::uint32_t record_bytes) noexcept {
    return static_cast<std::int32_t>((ceiling_x1e6(record_bytes) + 9999ULL) / 10000ULL);
}

// Break-even budget in bytes per stream for a resident plane of `resident_bytes`, from
// the COLD CODEC RULE in src/product/kv_tier_formats.h: the rANS record is below a
// plane of R bytes while ceil(512*b/8) <= (R - 1344)/32, i.e. while the per-stream
// budget is at most (R - 1344)/32. Kept here (rather than re-derived at each call
// site) so the reconciliation against the break-evens is one expression.
[[nodiscard]] constexpr std::uint32_t break_even_budget_bytes(std::uint32_t resident_bytes) noexcept {
    return resident_bytes <= kHeaderBytes + kScaleTailBytes
               ? 0
               : (resident_bytes - kHeaderBytes - kScaleTailBytes) / kStreams;
}

// WHY A PAGE LOOP ENDED EARLY. One enumerator per `break` in program_impl.h's page loop; the
// three have three different fixes (size the pool, prepare the spill files, raise
// --cold-disk-bytes), which is why they are not pooled into one number. Namespace scope, not a
// nested type: the free functions at the bottom of this header name it.
enum class TruncationReason : std::uint32_t {
    ColdPoolExhausted = 0, // program_impl.h: `if (slot < 0) { break; }`
    NoDiskStaging     = 1, // disk policy: cold_disk_staging[0] == nullptr
    NoDiskFileSlot    = 2, // disk policy: --cold-disk-bytes exhausted
    kReasons          = 3,
};

struct Census {
    // Denominator: pages that reached the codec's validity check, i.e. pages whose
    // layers all resolved to a cold codec and were encoded. A page that dies on the
    // dtype dispatch never gets here and is counted in pages_refused instead.
    std::uint64_t pages_evaluated = 0;
    // Numerator: of those, the pages where at least one layer's slot_valid came back 0
    // -- the rANS stream budget overflowed, the slot was released and the page stayed
    // hot. THIS IS THE FALLBACK COUNT.
    std::uint64_t pages_fell_back = 0;
    // Pages the dtype dispatch refused (no cold codec for some layer). A SEPARATE event
    // from the fallback: no rANS ran on them.
    std::uint64_t pages_refused = 0;

    // Head-plane granularity. The validity tensor is one int32 per (page, kv_head, plane)
    // PER LAYER, and the pass treats ANY zero as "the page stays hot", so the page count alone
    // cannot say how far over the budget the data sits. Two plane counts can.
    //
    // WHO WRITES WHAT -- one writer per field, and it is load-bearing:
    //   * head_planes_evaluated is written by note_page_evaluated() AND NOWHERE ELSE. One
    //     page charges its 2*kv_heads planes exactly once, at the moment it reaches the
    //     codec, whether or not it then falls back.
    //   * head_planes_inspected is written by note_layer_inspected() AND NOWHERE ELSE: one
    //     call per (page, layer) pair whose validity tensor the scan actually READ.
    //   * head_planes_overflowed is written by note_page_fell_back() AND NOWHERE ELSE.
    //
    // [MEANING FIX 2026-09-18, later the same day: head_plane_overflow_ppm WAS NOT A RATE.]
    // That function -- kept, unchanged, because dl/censusfix/cold_fallback_census_selftest.cpp pins its exact
    // arithmetic -- divides head_planes_overflowed by head_planes_evaluated, and those are TWO
    // DIFFERENT POPULATIONS:
    //   * the NUMERATOR counts the zero flags of ONE layer per falling page: the layer the scan
    //     stopped at. program_impl.h's scan is `layer < layers && valid`, an early exit, so no
    //     deeper layer is ever read and no deeper zero can be in this term.
    //   * the DENOMINATOR counts 2*kv_heads per EVALUATED PAGE -- one layer's worth of planes
    //     per page -- while the codec writes one whole flag set per layer, i.e. `layers` such
    //     lots per page, and the scan reads between 1 and `layers` of them.
    // So the old ratio was a page-share index of a per-layer quantity: it named neither the
    // flags the codec WROTE nor the flags the scan READ, and on a 16-layer stack it overstated
    // the density of overflow among the written flags by about 16x. It is SUPERSEDED and is no
    // longer published as a rate -- format() does not print it -- but it is not deleted: a
    // second instrument that quietly disappears is a second instrument nobody can audit.
    // head_plane_inspected_overflow_ppm() is the replacement. It IS a rate, over exactly ONE
    // population: the head-plane validity flags the validity scan READ. That is the only
    // population whose zero count is exactly known at zero extra GPU cost -- every flag read
    // before the stopping layer is provably non-zero (the scan stops at the first zero), the
    // stopping layer's zeros are exactly the k_zero + v_zero that note_page_fell_back() is
    // handed, and the layers after it were never read and so are in NEITHER term. It remains a
    // LOWER BOUND on the overflow over all flags the codec wrote, and it says so by naming its
    // own denominator. The reproduction is
    // dl/censusrate/cold_fallback_inspected_rate_selftest.cpp.
    // A falling page is ALSO an evaluated page (see the ordering invariant below), so an
    // increment on the falling branch as well would charge it twice and divide the reported
    // rate by (1 + f). That is exactly what this header did until 2026-09-18; the
    // reproduction is dl/censusfix/cold_fallback_census_selftest.cpp's check_one_falling_page_is_one_
    // page_of_denominator() and check_peer_scenario_four_pages_two_falling().
    std::uint64_t head_planes_evaluated  = 0; // SUPERSEDED page-share denominator; not published
    std::uint64_t head_planes_overflowed = 0; // the numerator, shared by both ratios
    std::uint64_t head_planes_inspected  = 0; // denominator of the PUBLISHED rate: flags the scan read

    // Per-layer first-offender histogram: the layer ordinal at which the validity scan
    // stopped. The scan early-exits (`layer < layers && valid`), so this is the FIRST
    // offending layer of each falling page -- the layer the guarded print names.
    std::array<std::uint64_t, kMaxLayers> layer_fell_back{};

    // --- TRUNCATION OF THE POPULATION (added 2026-09-18) -------------------
    // THE DEFECT THIS NAMES. The page loop in program_impl.h leaves by `break` on three
    // conditions, and a `break` is not a per-page skip: from that page onward the loop is
    // over, so EVERY subsequent page of that pass never reaches the codec and never reaches
    // any counter in this struct. The three are
    //   * the cold slot pool is exhausted (`if (slot < 0) { break; }`),
    //   * the disk policy has no staging buffer (the spill files could not be prepared),
    //   * --cold-disk-bytes is exhausted (no free file slot).
    // The first of the three is ROUTINE, not exceptional: the pool is a capacity, so once it is
    // full every later pass truncates at its first page. Before this block the only trace such
    // a pass left was that it retired fewer pages than the engine expected, and the counters
    // above silently described a population that had been cut short -- the rate was a LOWER
    // BOUND whose truncation was unreported. It is reported now:
    //   * truncated_passes counts the page loops that left early (one per pass, per break);
    //   * pages_unattempted counts the pages that stopped contributing, i.e. the page the
    //     break happened on PLUS every page after it in that pass, summed over passes. Those
    //     pages are NOT in pages_evaluated: they were never asked of the codec, and folding
    //     them in would corrupt the very rate this header exists to make honest.
    // WHAT THIS BUYS A READER: pages_unattempted != 0 says the page rate above is a rate over
    // the pages the codec was ASKED about -- which is the only population it ever claimed -- and
    // not over the pages the pass intended to ask about.
    std::uint64_t truncated_passes  = 0;
    std::uint64_t pages_unattempted = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(TruncationReason::kReasons)>
        truncated_by_reason{};

    [[nodiscard]] std::uint64_t truncated_pages(TruncationReason reason) const noexcept {
        const auto index = static_cast<std::size_t>(reason);
        return index < truncated_by_reason.size() ? truncated_by_reason[index] : 0;
    }

    // A truncated pass with nothing else to say still prints: pages_unattempted is the one
    // event whose whole purpose is to be visible in the case where the pass got nowhere.
    [[nodiscard]] bool any() const noexcept {
        return pages_evaluated != 0 || pages_fell_back != 0 || pages_refused != 0 ||
               truncated_passes != 0;
    }

    [[nodiscard]] std::uint64_t fallback_ppm() const noexcept {
        return pages_evaluated == 0 ? 0 : pages_fell_back * 1000000ULL / pages_evaluated;
    }

    // SUPERSEDED, and deliberately NOT renamed: the peer's 39-check selftest asserts this
    // function's exact arithmetic, and that arithmetic is correct -- it is the two POPULATIONS
    // it divides that do not match (see the meaning note above). Not published by format().
    [[nodiscard]] std::uint64_t head_plane_overflow_ppm() const noexcept {
        return head_planes_evaluated == 0
                   ? 0
                   : head_planes_overflowed * 1000000ULL / head_planes_evaluated;
    }

    // THE PUBLISHED RATE. Numerator and denominator are the same population: the head-plane
    // validity flags the scan READ. Exact (see the meaning note above), and a lower bound on
    // the overflow over all flags the codec wrote, because the scan stops at the first zero.
    [[nodiscard]] std::uint64_t head_plane_inspected_overflow_ppm() const noexcept {
        return head_planes_inspected == 0
                   ? 0
                   : head_planes_overflowed * 1000000ULL / head_planes_inspected;
    }

    // ORDERING INVARIANT: note_page_evaluated() is called for a page BEFORE
    // note_page_fell_back() can be called for it -- the pass evaluates a page's whole
    // layer stack and only then learns the slot is invalid. Every counter here is a
    // tally, not a state machine, so the only consequence of breaking the order is that
    // fallback_ppm() divides by a denominator that is too small: it guards against zero,
    // so it would report an over-stated rate rather than crash.
    //
    // This order is ALSO the reason note_page_fell_back() must not touch
    // head_planes_evaluated: the denominator was already charged for this very page.
    void note_page_evaluated(std::uint32_t kv_heads) noexcept {
        ++pages_evaluated;
        head_planes_evaluated += 2ULL * kv_heads; // K and V plane, one flag per head
    }

    // ONE CALL PER (PAGE, LAYER) PAIR THE SCAN READ -- not per page, and only for a layer whose
    // validity tensor was actually present (program_impl.h's scan `continue`s on a null tensor
    // without reading any flag, and such a layer is in neither term of the rate). This is the
    // denominator the published rate divides by, which is why it is charged where the flags are
    // read and nowhere else.
    void note_layer_inspected(std::uint32_t kv_heads) noexcept {
        head_planes_inspected += 2ULL * kv_heads; // K and V plane, one flag per head
    }

    // One page loop left early: `pages` counts the page the break happened on plus every page
    // after it in that pass, none of which reached the codec.
    void note_page_loop_truncated(TruncationReason reason, std::uint64_t pages) noexcept {
        ++truncated_passes;
        pages_unattempted += pages;
        const auto index = static_cast<std::size_t>(reason);
        if (index < truncated_by_reason.size()) { truncated_by_reason[index] += pages; }
    }

    void note_page_fell_back(std::uint32_t layer, std::uint64_t k_zero, std::uint64_t v_zero,
                             std::uint32_t kv_heads) noexcept {
        (void)kv_heads; // NOT a denominator term: this page's 2*kv_heads planes were already
                        // charged by note_page_evaluated(). The parameter is kept because the
                        // call site passes it and the signature is cited by
                        // tests/test_cold_slot_release_bytes.cpp. Do not add it back.
        ++pages_fell_back;
        head_planes_overflowed += k_zero + v_zero;
        if (layer < kMaxLayers) { ++layer_fell_back[layer]; }
    }

    void note_page_refused() noexcept { ++pages_refused; }

    // --- reporting ---------------------------------------------------------
    // `format()` is the primitive, and it performs NO heap allocation: the call site is
    // ProgramImplCore's destructor, which is `noexcept`, so a std::string built there
    // would turn an allocation failure at teardown into std::terminate -- and teardown is
    // exactly where this VM's memory pressure is highest. The line's load-bearing fields
    // (both tallies, the rate, the enforced ceiling) are written FIRST, so a truncated
    // line still carries the measurement; only the per-layer histogram, which is
    // redundant with the totals, can be cut off.
    //
    // Decimal formatting is integer-only (`%llu.%06llu` / `%llu.%04llu`) rather than a
    // std::to_string helper, for the same noexcept reason and because a rounded
    // 2.609 or 2.60 would lose the header's whole honesty note about the byte ceil.
    std::size_t format(char* out, std::size_t cap, std::uint32_t record_bytes) const noexcept {
        if (cap == 0) { return 0; }
        const std::uint64_t ceil_micros = ceiling_x1e6(record_bytes);
        const std::uint64_t ppm         = fallback_ppm();
        // The PUBLISHED overflow rate. head_plane_overflow_ppm() is deliberately not used here:
        // its denominator is a different population from its numerator (see the meaning note),
        // and a teardown line is the one place a number must not need a footnote.
        const std::uint64_t hp_ppm      = head_plane_inspected_overflow_ppm();
        int used = std::snprintf(
            out, cap,
            "[cold-fallback] pages evaluated=%llu fell_back=%llu (%llu.%04llu%%) refused=%llu | "
            "head-planes inspected=%llu overflowed=%llu (%llu.%04llu%% of inspected) | "
            "truncated passes=%llu pages_unattempted=%llu (pool=%llu staging=%llu spill=%llu) | "
            "record=%u B, budget=%u B per %u-symbol stream, enforced ceiling=%llu.%06llu b/c",
            static_cast<unsigned long long>(pages_evaluated),
            static_cast<unsigned long long>(pages_fell_back),
            static_cast<unsigned long long>(ppm / 10000ULL),
            static_cast<unsigned long long>(ppm % 10000ULL),
            static_cast<unsigned long long>(pages_refused),
            static_cast<unsigned long long>(head_planes_inspected),
            static_cast<unsigned long long>(head_planes_overflowed),
            static_cast<unsigned long long>(hp_ppm / 10000ULL),
            static_cast<unsigned long long>(hp_ppm % 10000ULL),
            static_cast<unsigned long long>(truncated_passes),
            static_cast<unsigned long long>(pages_unattempted),
            static_cast<unsigned long long>(truncated_pages(TruncationReason::ColdPoolExhausted)),
            static_cast<unsigned long long>(truncated_pages(TruncationReason::NoDiskStaging)),
            static_cast<unsigned long long>(truncated_pages(TruncationReason::NoDiskFileSlot)),
            record_bytes, stream_budget_bytes(record_bytes), kSymbolsPerStream,
            static_cast<unsigned long long>(ceil_micros / 1000000ULL),
            static_cast<unsigned long long>(ceil_micros % 1000000ULL));
        if (used < 0) {
            out[0] = '\0';
            return 0;
        }
        std::size_t len = static_cast<std::size_t>(used);
        if (len >= cap) { return cap - 1; }
        for (std::uint32_t layer = 0; layer < kMaxLayers; ++layer) {
            if (layer_fell_back[layer] == 0) { continue; }
            const int wrote =
                std::snprintf(out + len, cap - len,
                              len == static_cast<std::size_t>(used) ? " | fell_back by layer: L%u:%llu"
                                                                    : " L%u:%llu",
                              layer,
                              static_cast<unsigned long long>(layer_fell_back[layer]));
            if (wrote < 0 || static_cast<std::size_t>(wrote) >= cap - len) {
                len = cap - 1;
                break;
            }
            len += static_cast<std::size_t>(wrote);
        }
        return len;
    }

    // std::string convenience for host tests. NOT for the noexcept call site.
    [[nodiscard]] std::string describe(std::uint32_t record_bytes) const {
        std::array<char, 2048> buf{};
        const std::size_t len = format(buf.data(), buf.size(), record_bytes);
        return std::string(buf.data(), len);
    }
};

// Process-cumulative. Namespace scope, not function-local, because the report is emitted
// by ProgramImplCore's destructor -- a different member function from the pass that
// increments it. `inline` so the TUs that include runtime/program_impl.h share one
// instance, which is what makes the number a rate over the process rather than per TU.
inline Census g_census{};

inline void note_page_evaluated(std::uint32_t kv_heads) noexcept {
    g_census.note_page_evaluated(kv_heads);
}
inline void note_page_fell_back(std::uint32_t layer, std::uint64_t k_zero, std::uint64_t v_zero,
                                std::uint32_t kv_heads) noexcept {
    g_census.note_page_fell_back(layer, k_zero, v_zero, kv_heads);
}
inline void note_page_refused() noexcept { g_census.note_page_refused(); }
inline void note_layer_inspected(std::uint32_t kv_heads) noexcept {
    g_census.note_layer_inspected(kv_heads);
}
inline void note_page_loop_truncated(TruncationReason reason, std::uint64_t pages) noexcept {
    g_census.note_page_loop_truncated(reason, pages);
}

} // namespace ninfer::targets::qwen3_6::cold_fallback
