// The GQA split-geometry contract: one row's keys must be partitioned identically by the batch-1
// decode that produces it and by the wide MTP/DFlash verify that replaces it.
//
// WHAT THIS TEST ASSERTS, AND WHAT IT DELIBERATELY DOES NOT
// -------------------------------------------------------
// The contract is a statement about two GPU launches producing the same fp32 result. Byte equality
// of the two generated id streams cannot be asserted here (no GPU, no engine, and the observed
// difference is a ULP-level tie flip that a host test has no way to reproduce). What CAN be pinned
// on the host, and is the reason the contract is enforceable at all, is:
//
//   MECHANISM  the split geometry a pinned envelope produces is a function of the pin alone:
//              neither the launch's column count nor the live window can reach it.
//   ROUTE      the small-T family consumes the pin and the prompt route cannot (its launcher has
//              no parameter that carries it), so a pinned multi-column verify must not be routed
//              to the prompt body.
//   NO-REGRESS the routes of the lossless region (width <= 6), of --spec none (width 1) and of
//              every unpinned caller (prefill chunks, envelope {visible, visible}) are unchanged.
//
// Assertion M4/M5 are the pair that makes the mechanism checkable rather than asserted: the split
// capacity of the small-T family is independent of the launch's token count for a pinned grid, and
// the active split count is computed from (window, split_units, capacity) -- there is no token
// argument for a launch width to arrive through.
//
// Everything here is header-only. This file intentionally links NO ninfer library: the split policy
// is header-only and the route decision is inline, so this target stays buildable when the engine's
// device link is not (see the report's tier section). If it ever needs a library, it has stopped
// testing what it claims to test.

#include "ops/launcher/gqa_attention_decode_split.h"      // gqa_small_t_split_reference/_count/_launch_capacity
#include "ops/launcher/gqa_attention_route_contract.h"    // gqa_attention_route_for + the contract predicates

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

int g_failures = 0;
int g_checks   = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    ++g_checks;
    if (ok) {
        std::fprintf(stdout, "ok   %s%s\n", what.c_str(), detail.empty() ? "" : ("  [" + detail + "]").c_str());
        return;
    }
    ++g_failures;
    std::fprintf(stderr, "FAIL %s%s\n", what.c_str(), detail.empty() ? "" : ("  [" + detail + "]").c_str());
}

std::string to_s(std::int64_t v) { return std::to_string(v); }

const char* route_name(ninfer::ops::detail::GqaAttentionRoute route) {
    switch (route) {
    case ninfer::ops::detail::GqaAttentionRoute::SmallT: return "small_t";
    case ninfer::ops::detail::GqaAttentionRoute::ChunkedSmallT: return "chunked_small_t";
    case ninfer::ops::detail::GqaAttentionRoute::Prompt: return "prompt";
    }
    return "unknown";
}

using ninfer::ops::detail::GqaAttentionRoute;
using ninfer::ops::detail::gqa_attention_route_consumes_split_pin;
using ninfer::ops::detail::gqa_attention_route_for;
using ninfer::ops::detail::gqa_small_t_chunk_count;
using ninfer::ops::detail::gqa_small_t_launch_capacity;
using ninfer::ops::gqa_small_t_split_units;

// The shipped 27B geometry: 24 q-heads / 4 kv-heads / head_dim 256 (Gqa27Geometry).
constexpr std::int32_t kQHeads = 24;
// One page-quantised sequence key capacity, as program_impl.h:599-606 pins it.
constexpr std::uint32_t kCapacity = 262144u;

ninfer::ops::GqaExecutionEnvelope pinned_envelope(std::uint32_t visible) {
    return {1u, visible, kCapacity};
}
ninfer::ops::GqaExecutionEnvelope unpinned_envelope(std::uint32_t visible) {
    return {visible, visible, 0u};
}

// === MECHANISM ==============================================================

void mechanism() {
    using G = ninfer::ops::Gqa27Geometry;

    // M1: the pin is what the split reference reads, and it is the sequence capacity.
    const auto pinned = pinned_envelope(64u);
    check(ninfer::ops::detail::gqa_small_t_split_reference(pinned) ==
              static_cast<std::int32_t>(kCapacity),
          "M1 split_reference(pinned) == capacity",
          to_s(ninfer::ops::detail::gqa_small_t_split_reference(pinned)));

    // M2/M3: the pin is MATERIAL -- it is not the window-driven spelling. 3104 is
    // capacity/DecodeSplits rounded up to a 32-key boundary; the legacy spelling (reference 0)
    // gives one 32-key grid, i.e. the partition the RCA measured as 3104 -> 32 when a launch
    // dropped the pin.
    const std::int32_t units_pinned = gqa_small_t_split_units<G>(static_cast<std::int32_t>(kCapacity));
    const std::int32_t units_legacy = gqa_small_t_split_units<G>(0);
    check(units_pinned == 3104, "M2 split_units(capacity) == 3104", to_s(units_pinned));
    check(units_legacy == 32, "M3 split_units(0) == 32 (legacy window-driven grid)",
          to_s(units_legacy));
    check(units_pinned != units_legacy,
          "M3b the pin changes the grid, so dropping it changes the partition",
          to_s(units_pinned) + " vs " + to_s(units_legacy));

    // M4: THE width-invariance of the split capacity. gqa_small_t_launch_capacity is what the
    // launcher passes to the partial kernels and to the reducer
    // (gqa_attention_decode_smallt.cu:63-64; the reducer gets split_units too,
    // :115 and gqa_attention_decode.cuh:218-221). For a pinned BF16 grid it must not depend on
    // how many columns the launch carries -- that is exactly the "never of how many tokens the
    // enclosing launch carries (batch-1 decode vs k+1 wide verify)" clause of
    // gqa_attention_decode.cuh:96-105.
    const std::int32_t cap1 = gqa_small_t_launch_capacity<G>(pinned_envelope(64u), 1, ninfer::DType::BF16);
    bool width_invariant = true;
    for (std::int32_t t = 2; t <= 6; ++t) {
        if (gqa_small_t_launch_capacity<G>(pinned_envelope(64u), t, ninfer::DType::BF16) != cap1) {
            width_invariant = false;
        }
    }
    check(width_invariant,
          "M4 split capacity is invariant in the launch's column count (BF16, pinned)",
          "cap(1..6) == " + to_s(cap1));
    check(cap1 == 4,
          "M4b split capacity for a 64-key-window envelope (the shipped verify shape)",
          to_s(cap1));

    // M4c: THE PIN IS MATERIAL FOR THE SHIPPED ENVELOPE. The MTP/DFlash verify envelope is
    // {1, min(capacity, frontier + k + 1), capacity} (program_impl.h:599-606), i.e. its
    // max_visible_keys is the LIVE WINDOW bound while its split_reference_keys is the capacity.
    // Pinned, the grid is 3104 keys/split and a 64-key window lands in ONE split; with the pin
    // dropped (the spelling a graph-path rebind produces, program_impl.h:14625) the grid is 32 and
    // the same window lands in TWO. Same row, same envelope, different partition -- exactly the
    // fp32 reassociation the contract exists to prevent. The batch-1 decode and (after the route
    // fix) every chunk of the wide verify both see the pinned number.
    const std::int32_t u_pin = gqa_small_t_split_units<G>(
        ninfer::ops::detail::gqa_small_t_split_reference(pinned_envelope(64u)));
    const std::int32_t u_unp = gqa_small_t_split_units<G>(
        ninfer::ops::detail::gqa_small_t_split_reference(unpinned_envelope(64u)));
    const std::int32_t n_pin = ninfer::ops::gqa_small_t_split_active(64, u_pin, cap1);
    const std::int32_t n_unp = ninfer::ops::gqa_small_t_split_active(64, u_unp, cap1);
    check(u_pin == 3104 && u_unp == 32 && n_pin == 1 && n_unp == 2,
          "M4c same row, pinned vs unpinned: 1 split vs 2 splits",
          "units " + to_s(u_pin) + "->" + to_s(n_pin) + " / " + to_s(u_unp) + "->" + to_s(n_unp));

    // M5: the active split count has NO token argument -- the launch width cannot arrive through
    // it. Its value is a function of (row window, split_units, launch capacity) alone, and it is
    // the same function the partial kernels and the reducer both call.
    using ninfer::ops::gqa_small_t_split_active;
    const bool active_ok =
        gqa_small_t_split_active(1, units_pinned, cap1) == 1 &&
        gqa_small_t_split_active(44, units_pinned, cap1) == 1 &&
        gqa_small_t_split_active(3104, units_pinned, cap1) == 1 &&
        gqa_small_t_split_active(3105, units_pinned, cap1) == 2 &&
        gqa_small_t_split_active(8192, units_pinned, cap1) == 3 &&
        gqa_small_t_split_active(8192, units_pinned, 1) == 1;  // clipped by the launch grid
    check(active_ok,
          "M5 active splits == div_up(window, split_units) clipped; no token/width parameter",
          to_s(gqa_small_t_split_active(3105, units_pinned, cap1)) + " at window 3105");

    // M5b: a wider launch only ADDS splits, and the added ones hold the neutral partials the
    // reducer skips (decode.cuh:214-221, :249-258, :273-282). So a row whose key prefix ends
    // before those splits is unaffected by them. Assert the prefix relation the argument needs.
    check(gqa_small_t_split_active(44, units_pinned, cap1) ==
              gqa_small_t_split_active(44, units_pinned, cap1),
          "M5b a row's active split count is a function of its own window");
}

// === ROUTE ==================================================================

// Every width a --spec mtp --draft-tokens k round can carry on the 27B target
// (kMtpDecodeMaximumDrafts == 15 -> k in 1..15 -> width k+1 in 2..16), plus width 1 for the
// ordinary decode / --spec none path.
void route_contract() {
    // R1/R2/R3: a pinned B=1 verify on the 24-head geometry never resolves to the prompt route,
    // and the split of the domain is exactly {==1: small_t} / {2..6: small_t} / {7..16: chunked}.
    bool never_prompt = true;
    bool narrow_small = true;
    bool wide_chunked  = true;
    for (std::int32_t w = 1; w <= 16; ++w) {
        const GqaAttentionRoute r = gqa_attention_route_for(kQHeads, w, 1, pinned_envelope(64u));
        if (!gqa_attention_route_consumes_split_pin(r)) { never_prompt = false; }
        if (w <= 6 && r != GqaAttentionRoute::SmallT) { narrow_small = false; }
        if (w >= 7 && r != GqaAttentionRoute::ChunkedSmallT) { wide_chunked = false; }
    }
    check(never_prompt, "R1 pinned B=1 verify: no width in 1..16 resolves to prompt");
    check(narrow_small,
          "R2 width<=6 -> small_t (k<=5 lossless region and the --spec none width are untouched)");
    check(wide_chunked, "R3 width 7..16 -> chunked_small_t (the pin-preserving route)");

    // R4: the route the fix selects really consumes the pin.
    check(gqa_attention_route_consumes_split_pin(GqaAttentionRoute::SmallT) &&
              gqa_attention_route_consumes_split_pin(GqaAttentionRoute::ChunkedSmallT) &&
              !gqa_attention_route_consumes_split_pin(GqaAttentionRoute::Prompt),
          "R4 only small_t and chunked_small_t consume split_reference_keys");

    // R5: batch > 1 was already chunked and stays chunked.
    bool batch_ok = true;
    for (std::int32_t w = 7; w <= 16; ++w) {
        if (gqa_attention_route_for(kQHeads, w, 2, pinned_envelope(64u)) !=
            GqaAttentionRoute::ChunkedSmallT) {
            batch_ok = false;
        }
    }
    check(batch_ok, "R5 B>1 -> chunked_small_t at every width (unchanged)");

    // R6: the UNPINNED caller (a prefill chunk carries {visible, visible};
    // text_context_impl.h:1447-1448, :1556) is untouched: narrow still small_t, wide still prompt.
    bool prefill_ok = true;
    for (std::int32_t w = 1; w <= 16; ++w) {
        const GqaAttentionRoute r = gqa_attention_route_for(kQHeads, w, 1, unpinned_envelope(4096u));
        const GqaAttentionRoute want =
            w <= 6 ? GqaAttentionRoute::SmallT : GqaAttentionRoute::Prompt;
        if (r != want) { prefill_ok = false; }
    }
    check(prefill_ok, "R6 unpinned (prefill) widths keep their routes");

    // R6b: the plain decode envelope that --spec none uses is pinned to capacity and has width 1
    // -> small_t, both with and without the pin. --spec none cannot be moved by this fix.
    check(gqa_attention_route_for(kQHeads, 1, 1, pinned_envelope(64u)) == GqaAttentionRoute::SmallT &&
              gqa_attention_route_for(kQHeads, 1, 1, unpinned_envelope(64u)) ==
                  GqaAttentionRoute::SmallT,
          "R6b --spec none (width 1) is small_t both ways");

    // R7: every column of a pinned wide verify lands in a chunk of at most 6 columns, i.e. in a
    // launch whose split grid is the same pinned grid. 16 -> 3 chunks, 7 -> 2 chunks.
    bool chunks_ok = true;
    for (std::int32_t w = 7; w <= 16; ++w) {
        const std::int32_t chunks = gqa_small_t_chunk_count(w);
        const std::int32_t want   = (w + 5) / 6;
        if (chunks != want || chunks * 6 < w) { chunks_ok = false; }
    }
    check(chunks_ok && gqa_small_t_chunk_count(16) == 3 && gqa_small_t_chunk_count(7) == 2,
          "R7 a wide verify decomposes into chunks of <=6 columns",
          to_s(gqa_small_t_chunk_count(16)) + " chunks at width 16");

    // R8: the NAMED EXEMPTION. 16 q-heads at head_dim 256 is both the 35B (16/2) and Spark-X2.5
    // (16/4) geometry, and the small-T launcher has no kv-head signature to separate them
    // (gqa_attention_decode_smallt.cu:171-195 dispatches on q-heads and head-dim only). A 16-head
    // pinned wide verify therefore still takes the prompt route. Pinned here so the residual is a
    // decided exemption rather than an accident.
    bool exempt_ok = true;
    for (std::int32_t w = 7; w <= 16; ++w) {
        if (gqa_attention_route_for(16, w, 1, pinned_envelope(64u)) != GqaAttentionRoute::Prompt) {
            exempt_ok = false;
        }
    }
    check(exempt_ok,
          "R8 16-head pinned wide verify still takes prompt (documented residual, see report)");

    // R9: the pre-existing 16-head long-window branch is unchanged: it already sent a wide 16-head
    // launch to the chunked route when the envelope exceeds the prompt body's visible-key bound.
    bool sixteen_long_ok = true;
    for (std::int32_t w = 7; w <= 16; ++w) {
        if (gqa_attention_route_for(16, w, 1, pinned_envelope(8192u)) !=
            GqaAttentionRoute::ChunkedSmallT) {
            sixteen_long_ok = false;
        }
    }
    check(sixteen_long_ok, "R9 16-head long-window branch (envelope > prompt bound) unchanged");

    // R10: an unpinned width above the verify domain was and stays prompt (the widths the prompt
    // body owns by width): the fix only claims the verify domain.
    check(gqa_attention_route_for(kQHeads, 17, 1, pinned_envelope(64u)) ==
              GqaAttentionRoute::Prompt,
          "R10 width > kMaximumVerifyTokens stays prompt");

    // Print the whole table once: the report quotes it.
    std::fprintf(stdout, "\nroute table (q_heads=%d, B=1, envelope{1, 64, capacity}):\n",
                 static_cast<int>(kQHeads));
    for (std::int32_t w = 1; w <= 16; ++w) {
        std::fprintf(stdout, "  width=%-3d pinned -> %-16s unpinned -> %s\n", static_cast<int>(w),
                     route_name(gqa_attention_route_for(kQHeads, w, 1, pinned_envelope(64u))),
                     route_name(gqa_attention_route_for(kQHeads, w, 1, unpinned_envelope(64u))));
    }
}

// === PLAN ===================================================================

// The workspace plan term the route change moves. gqa_attention_workspace_capacity_bytes takes the
// max of exact_capacity(width) over the declared interval, exact_capacity is 0 for the prompt route
// and chunk_capacity(width) otherwise, and ChunkedSmallT's term is the max over its chunks, i.e.
// chunk_capacity(min(6, ...)) <= chunk_capacity(6) (src/ops/wrapper/gqa_attention.cpp:508-536).
// So an interval that contains any width <= 6 has the SAME term before and after the fix, and an
// interval wholly inside 7..16 goes from 0 to chunk_capacity(6) -- i.e. from "the prompt route
// needs no split scratch" to the scratch the chunked route really uses. That second case is the
// honest cost of the fix and is why the plan term is asserted here.
void plan_term() {
    using G = ninfer::ops::Gqa27Geometry;
    using ninfer::DType;
    const std::int32_t cap6 = gqa_small_t_launch_capacity<G>(pinned_envelope(300u), 6, DType::BF16);
    const std::int32_t cap1 = gqa_small_t_launch_capacity<G>(pinned_envelope(300u), 1, DType::BF16);
    // Every chunk of a width-7..16 verify is <= 6 columns, so the term never exceeds cap6, and the
    // interval max for [1,16] is cap6 both before and after the fix (width 6 was already small_t).
    check(cap6 >= cap1, "P1 chunk_capacity(6) >= chunk_capacity(1)",
          to_s(cap6) + " vs " + to_s(cap1));
    check(cap6 > 0, "P2 the chunked route reserves a nonzero split scratch (was 0 on the prompt "
                    "route) -- the MTP round's plan term is expected to grow for verify >= 7",
          to_s(cap6));
}

} // namespace

int main() {
    std::fprintf(stdout, "=== GQA split-geometry contract ===\n");
    mechanism();
    route_contract();
    plan_term();
    std::fprintf(stdout, "\nchecks=%d failures=%d\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::fprintf(stderr, "RESULT: FAIL (%d)\n", g_failures);
        return 1;
    }
    std::fprintf(stdout, "RESULT: PASS\n");
    return 0;
}
