#pragma once

// COLD-SLOT VALIDITY: the ONE definition of "this cold slot's bytes are known to be a
// complete record", owned at the SPEC layer so that the WRITE path and the READ path ask
// the same question instead of each carrying its own copy of it.
//
// WHY THIS HEADER EXISTS (the defect it closes, stated as measured facts):
//
//   * `src/targets/qwen3_6/impl/runtime/program_impl.h`'s spill pass clears a layer's
//     valid flag when the rANS stream budget overflows, and the tree prints
//     `[cold] page N INVALID at layer L` for it. An invalid page is dropped and
//     "stays hot" (its `cold_pages` entry is never created), which is why such a page is
//     a MISS at plan time and is never spilled.
//   * BUT THE READ PATH NEVER ASKS. `restore_cold_page()` reads the spill region into a
//     STAGING slot, decodes it into the K/V planes and publishes the sentinel -- and it
//     reads `cold_slot_valid` NOWHERE. The tree states the consequence itself, in its own
//     words: "the decode kernels trust the published sentinel instead of re-reading the
//     flags". So after a restore the sentinel names a staging slot whose valid flags
//     belong to whatever page last used that slot.
//   * The protection that exists today is therefore NOT a check on the read path: it is
//     the WRITE path's discipline ("an invalid page never becomes cold"), i.e. an
//     invariant held by construction and never re-verified where the bytes are consumed.
//     `NINFER_RRFAST_BYTE_GUARD` is the only gate that consults the flags on the byte leg,
//     and it is off by default, it is RUN-SCOPED (it returns before the whole restore
//     loop, so it cannot name a page), and the tree's own record says its condition is
//     necessary-at-best and not sufficient.
//
// WHAT THIS HEADER IS, AND WHAT IT IS NOT:
//
//   * It is a PURE function over the flag values, so it is testable with plain g++ and no
//     GPU (`spec/turn_recall_journal.h`'s own convention: "buildable out of tree, since
//     neither this file nor its header needs more than std and -I src"). The DEVICE read
//     stays at the call site -- the caller copies the flags D2H with the same two
//     `cudaMemcpy` calls the write-side scan already pays, and hands this header the
//     values. Nothing here knows about CUDA, tensors, pages or the engine.
//   * It is NOT a run-level gate. The granularity is deliberately ONE (page, layer): the
//     verdict carries the offending layer's index and both planes' zero counts, so a
//     refusal can name the page and the layer it refused instead of announcing that the
//     run is bad. A run-scoped refusal is what the tree already has and what this
//     replaces.
//   * It does NOT decide the policy by itself. The policy lives in one constant that the
//     consuming translation unit asserts against, so the choice cannot be left unmade
//     silently -- the same shape `spec/turn_recall_journal.h` already uses three times
//     (`kCodecBlindPolicy`, `kInexactAdmissionPolicy`, `kRecallBudgetEdgePolicy`).
//
// THE FLAG GEOMETRY, as the tree spells it. Per slot the validity tensor is
// `[kv_heads, 2, pages]` col-major -- head innermost, page outermost, so one slot's
// planes sit at `slot * 2 * kv_heads`, K first and V at `+ kv_heads`. Each plane is
// `kv_heads` 32-bit words and a word is a boolean: non-zero means the codec committed that
// head-plane's stream inside its fixed budget, zero means it did not. The widths and the
// two planes are the caller's; this header takes them as spans and asserts nothing about
// how many there are, because the flag count is `kv_heads` and `kv_heads` is a property of
// the model, not of the format.
//
// std-only, host-only, no CUDA, no engine headers.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace ninfer::spec::coldvalid {

// ---------------------------------------------------------------------------
// THE REFUSAL'S NAME, OWNED HERE
// ---------------------------------------------------------------------------
// A plain constant, not a `[[nodiscard]]` one (the attribute is meaningless on a variable
// and gcc says so). The spelling is NOT new: the tree already prints this exact token at
// `program_impl.h`'s `NINFER_RRFAST_BYTE_GUARD` site, so naming it here does not mint a
// second refusal -- it gives the FIRST one an owner, and gives the read path something
// spec'd to print. A run, a test and a report therefore cannot spell it three ways.
inline constexpr const char* kColdSlotInvalidRefusalName = "refused-cold-slot-invalid";

// ---------------------------------------------------------------------------
// THE VERDICT: what the flags said, for ONE (page, layer)
// ---------------------------------------------------------------------------
// Carries the layer and the counts rather than a bare bool, because a bare bool cannot
// name what it refused -- and "which layer overflowed" is the whole content of the
// diagnosis. `k_zero` / `v_zero` are counts over the SAME spans that were tested, so the
// printed numbers and the decision cannot disagree.
struct ColdSlotValidityVerdict {
    bool          valid       = true;  // every flag on both planes non-zero
    std::uint32_t layer       = 0;     // the layer this verdict is about (caller-supplied)
    std::size_t   k_zero      = 0;     // zero flags on the K plane
    std::size_t   v_zero      = 0;     // zero flags on the V plane
    std::size_t   k_flags     = 0;     // K plane width as TESTED (the denominator)
    std::size_t   v_flags     = 0;     // V plane width as TESTED
    std::size_t   layer_index = 0;     // 0-based position among the layers examined so far

    // The read path's own name for "there was nothing to check". A layer with no validity
    // tensor is not evidence of validity and is not evidence of corruption: the tree's own
    // scan SKIPS such a layer (`cold_valid.data == nullptr -> continue`), and so must this
    // verdict, or the read path would refuse pages whose flags the engine never wrote.
    [[nodiscard]] bool checked() const noexcept { return k_flags != 0 || v_flags != 0; }
};

// ---------------------------------------------------------------------------
// THE ONE PREDICATE
// ---------------------------------------------------------------------------
// `std::all_of` over an empty range is TRUE, which is the correct answer here and also the
// trap: a caller that hands in two empty spans (a layer with no validity tensor, or a
// mis-sized buffer) gets `valid == true` and might read that as "checked and good". The
// verdict above keeps `checked()` separate for exactly that reason, and the policy below
// refuses on `!valid` only -- never on `!checked()`.
[[nodiscard]] inline ColdSlotValidityVerdict
cold_slot_validity_of(std::span<const std::int32_t> k_flags,
                      std::span<const std::int32_t> v_flags,
                      std::uint32_t layer             = 0,
                      std::size_t   layer_index       = 0) noexcept {
    ColdSlotValidityVerdict verdict;
    verdict.layer       = layer;
    verdict.layer_index = layer_index;
    verdict.k_flags     = k_flags.size();
    verdict.v_flags     = v_flags.size();
    verdict.k_zero      = static_cast<std::size_t>(
        std::count(k_flags.begin(), k_flags.end(), std::int32_t{0}));
    verdict.v_zero      = static_cast<std::size_t>(
        std::count(v_flags.begin(), v_flags.end(), std::int32_t{0}));
    // Both planes, and `all_of` is used rather than `none_of(zero)` because it short-
    // circuits on the first zero: a page that fell back pays for the failing head-plane
    // and not for the rest of the plane, exactly as the write-side scan does.
    const bool k_all = std::all_of(k_flags.begin(), k_flags.end(),
                                   [](std::int32_t value) { return value != 0; });
    const bool v_all = std::all_of(v_flags.begin(), v_flags.end(),
                                   [](std::int32_t value) { return value != 0; });
    verdict.valid = k_all && v_all;
    return verdict;
}

// Convenience for a caller that already has the spans in one array of two.
[[nodiscard]] inline ColdSlotValidityVerdict
cold_slot_validity_of(std::span<const std::span<const std::int32_t>> planes,
                      std::uint32_t layer       = 0,
                      std::size_t   layer_index = 0) noexcept {
    const std::span<const std::int32_t> empty{};
    const std::span<const std::int32_t> k = planes.size() > 0 ? planes[0] : empty;
    const std::span<const std::int32_t> v = planes.size() > 1 ? planes[1] : empty;
    return cold_slot_validity_of(k, v, layer, layer_index);
}

// ---------------------------------------------------------------------------
// THE POLICY: what a caller does with a verdict
// ---------------------------------------------------------------------------
// `Unnamed` is NOT a policy -- it is the ABSENCE of one, and the consuming translation
// unit carries the `static_assert` that refuses to build while it is set. That is the
// idiom `turn_recall_journal.h` uses three times, and it is why a future editor cannot
// silence this header by forgetting to choose.
enum class ColdSlotValidityPolicy : std::uint8_t {
    Unnamed    = 0,  // not a policy. Engine-side static_assert, never shipped.
    ReportOnly = 1,  // the run executes and the flags are counted/printed only
    RefusePage = 2,  // the page is REFUSED by name before any decode kernel reads it
};

// THE SHIPPED VALUE. `RefusePage`, because anything weaker ships no fix: the defect is
// that the read path never asks, and `ReportOnly` is exactly "never asks, but says so".
// The one-token way back to the old behaviour is `ReportOnly`, and the must-red arms in
// the landing entry's driver use it to prove this predicate is load-bearing rather than
// decorative.
inline constexpr ColdSlotValidityPolicy kColdSlotValidityPolicy =
    ColdSlotValidityPolicy::RefusePage;

[[nodiscard]] inline constexpr const char*
cold_slot_validity_policy_name(ColdSlotValidityPolicy policy) noexcept {
    switch (policy) {
        case ColdSlotValidityPolicy::Unnamed:    return "unnamed";
        case ColdSlotValidityPolicy::ReportOnly: return "report-only";
        case ColdSlotValidityPolicy::RefusePage: return "refuse-page";
    }
    return "unknown";
}

// THE DECISION, as a pure function of the verdict and the policy, so the engine's branch
// is one call and a host test can exercise every policy against every verdict with no GPU.
//
// ⚠ THE ARM A CARELESS EDIT LOSES: `!verdict.checked()` must NEVER refuse. A layer with no
// validity tensor has no flags to test, and refusing it would refuse every page of a stack
// whose validity tensors are absent -- i.e. it would turn "the engine did not write flags
// here" into "the bytes are corrupt". The test that owns this header has a RED arm for
// exactly that, because the mistake is one conjunction away and it is silent:
// `return policy == RefusePage;` would do it.
[[nodiscard]] inline bool cold_slot_validity_refuses(const ColdSlotValidityVerdict& verdict,
                                                     ColdSlotValidityPolicy policy) noexcept {
    if (policy != ColdSlotValidityPolicy::RefusePage) { return false; }
    if (!verdict.checked()) { return false; }
    return !verdict.valid;
}

// ---------------------------------------------------------------------------
// THE LINE: one spec'd spelling of a per-page refusal
// ---------------------------------------------------------------------------
// Built here so the read path's message and the predicate cannot drift, and so the refusal
// carries the four things a reader needs and no fifth: the name, the page, the layer, and
// the two zero counts over their own denominators.
[[nodiscard]] inline std::string
cold_slot_invalid_refusal_line(std::uint32_t page,
                               const ColdSlotValidityVerdict& verdict,
                               std::string_view where) {
    std::string out = std::string("refusal=") + kColdSlotInvalidRefusalName;
    out += " page=" + std::to_string(page);
    out += " layer=" + std::to_string(verdict.layer);
    out += " k_zero=" + std::to_string(verdict.k_zero) + "/" + std::to_string(verdict.k_flags);
    out += " v_zero=" + std::to_string(verdict.v_zero) + "/" + std::to_string(verdict.v_flags);
    out += " policy=" + std::string(cold_slot_validity_policy_name(kColdSlotValidityPolicy));
    if (!where.empty()) { out += " at=" + std::string(where); }
    return out;
}

// ---------------------------------------------------------------------------
// COMPILE-TIME FACTS THIS HEADER OWNS
// ---------------------------------------------------------------------------
// The name must not be empty and must not be a prefix of another refusal the tree prints:
// a report that greps for `refused-cold-slot-invalid` must find this refusal and only it.
static_assert(kColdSlotInvalidRefusalName != nullptr &&
                  std::string_view(kColdSlotInvalidRefusalName).size() > 0U,
              "the refusal must have a name; an unnamed refusal is the defect this closes");

// `Unnamed` must not be the shipped value. This is the compile-time half of the idiom; the
// engine-side `static_assert` is the other half, and both exist because this header can be
// included by a TU that never branches on the policy at all.
static_assert(kColdSlotValidityPolicy != ColdSlotValidityPolicy::Unnamed,
              "cold_slot_validity.h's kColdSlotValidityPolicy is Unnamed: name it "
              "ReportOnly (the run executes, the flags are counted and printed, and the "
              "read path still reads bytes it cannot vouch for) or RefusePage (the page is "
              "refused by name before the first decode kernel). The owner's choice is one "
              "token at that constant; leaving it Unnamed is not a default, it is a "
              "decision nobody made");

} // namespace ninfer::spec::coldvalid
