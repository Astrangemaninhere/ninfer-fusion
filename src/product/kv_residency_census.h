// kv_residency_census.h -- the instrument that answers "how much KV did the recall path SAVE".
//
// WHY THIS FILE EXISTS. `dl/semwire/REPORT.md` section 7.1 records the answer to that question as
// 未取得 and names the missing piece: the engine's OWN resident-page / resident-block census, the
// family that prints `pages evaluated=` / `retired=` (src/targets/qwen3_6/impl/runtime/
// cold_fallback_census.h:321 and the `[unload] ... retired=` line). The numbers that exist in the
// tree today are per-STAGE (a fallback census, an unload census); what does not exist is one census
// that puts a RESIDENT count, a RETIRED count and the candidate-set NARROWING on the same
// denominator, so that "saved" is a ratio and not a bare integer.
//
// WHAT THIS FILE IS. Purely additive and header-only, like every other instrument in
// src/product/. It takes the counts the engine already produces, per round, and answers the
// question with:
//   * each count in its OWN column, with its OWN denominator (never summed across units);
//   * `saved pages` as a std::optional -- std::nullopt IS "未取得", and it is NEVER 0-as-unknown;
//   * a NAMED designed denominator printed beside every un-obtainable field, so the blank is
//     actionable rather than mysterious;
//   * a monotone-only API: no operation can decrease a counter, so a negative reading is
//     unrepresentable rather than merely unexpected.
//
// WHAT IT DELIBERATELY DOES NOT DO. It does not compute "saved" from a model. `saved` is only ever
// `baseline_resident - observed_resident`, both of which must have been SUPPLIED BY THE ENGINE for
// the same run. If the engine has not supplied a baseline, the number is UNOBTAINED with its
// denominator, and a caller that wants a different definition has to say so by name.
//
// RELATION TO THE CANDIDATE-SET SEAM. Feeding candidates into `SumDirReachResult::appended_pages`
// is an EXCLUSION set (src/spec/sum_dir_reach.h:517-559) -- measured by dl/kvmemoracle: it deletes
// the true answer rather than adding recall. A candidate set that never changes the FETCH RANGE
// cannot change the resident set either, so on today's wiring the honest reading is
// `saved == 0 pages, denominator = rounds x pages_per_round` -- and that is a READING, not an
// absence. This census is what makes the difference visible: it is the same instrument that would
// report a non-zero saving once a candidate set is allowed to narrow the fetch range.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace ninfer {
namespace product {

// The un-obtainable field is named here, once, so no caller invents its own word for it.
inline constexpr const char* kKvResidencyUnobtained = "UNOBTAINED";

// One run's residency census. Every field is a count of PAGES unless the name says otherwise;
// DRAM and SSD/COLD pages are counted in their own fields and are never added together here (they
// are different units, and the whole point of the split is that the SSD bytes are the claim).
struct KvResidencyCensus {
    // Denominators, printed separately from every numerator.
    std::uint64_t rounds              = 0; // rounds OBSERVED; the denominator of every mean below
    std::uint64_t pages_per_round_last = 0; // the LAST round's page count, reported as a shape

    // Numerators.
    std::uint64_t resident_pages_total  = 0; // sum over rounds of pages resident at round end
    std::uint64_t resident_blocks_total = 0; // sum over rounds of blocks resident (blocks != pages)
    std::uint64_t retired_pages_total   = 0; // sum over rounds of pages retired
    std::uint64_t evaluated_pages_total = 0; // sum over rounds of pages the cold path evaluated
    std::uint64_t requested_pages_total = 0; // sum over rounds of pages the recall ASKED for

    // The baseline half. `baseline_resident_pages_total` is what the SAME run would have held
    // resident with the narrowing OFF. It can only come from the engine; until it does, `saved`
    // is un-obtainable and says so.
    std::optional<std::uint64_t> baseline_resident_pages_total = std::nullopt;

    std::uint64_t rounds_with_no_residency = 0; // rounds that reported 0 resident: a shape, not a 0
};

// The designed denominator of every mean in this census. Printed WITH the means, never folded in.
[[nodiscard]] inline std::uint64_t kv_residency_census_denominator(
    const KvResidencyCensus& c) noexcept {
    return c.rounds;
}

// Observe one round. Monotone-only: there is no way to subtract, so a counter that could go
// negative is not expressible through this API.
inline void kv_residency_observe_round(KvResidencyCensus& c, std::uint64_t resident_pages,
                                       std::uint64_t resident_blocks, std::uint64_t retired_pages,
                                       std::uint64_t evaluated_pages,
                                       std::uint64_t requested_pages) noexcept {
    c.rounds += 1;
    c.pages_per_round_last = resident_pages;
    c.resident_pages_total  += resident_pages;
    c.resident_blocks_total += resident_blocks;
    c.retired_pages_total   += retired_pages;
    c.evaluated_pages_total += evaluated_pages;
    c.requested_pages_total += requested_pages;
    if (resident_pages == 0) { c.rounds_with_no_residency += 1; }
}

// "How much KV did the narrowing save", as an optional. std::nullopt IS 未取得; it is never 0.
// Defined ONLY as baseline - observed, on totals of the SAME unit (pages), from the SAME run.
[[nodiscard]] inline std::optional<std::uint64_t> kv_residency_saved_pages(
    const KvResidencyCensus& c) noexcept {
    if (!c.baseline_resident_pages_total.has_value()) { return std::nullopt; }
    const std::uint64_t base = *c.baseline_resident_pages_total;
    if (base < c.resident_pages_total) {
        // The narrowing cannot have INCREASED residency and still be a saving. Refusing here is
        // what keeps "the counter went negative" from being reportable as a small saving.
        return std::nullopt;
    }
    return base - c.resident_pages_total;
}

// The mean resident pages per round, or nullopt when there is no denominator. A mean with a zero
// denominator is UNDEFINED BY NAME, not 0.
[[nodiscard]] inline std::optional<double> kv_residency_mean_resident_pages(
    const KvResidencyCensus& c) noexcept {
    const std::uint64_t d = kv_residency_census_denominator(c);
    if (d == 0) { return std::nullopt; }
    return static_cast<double>(c.resident_pages_total) / static_cast<double>(d);
}

// The one line. Every numerator carries its unit; the denominator and the un-obtainable
// denominator are printed on the same line so a reader cannot silently drop them.
[[nodiscard]] inline std::string kv_residency_census_line(const KvResidencyCensus& c) {
    const std::optional<double> mean = kv_residency_mean_resident_pages(c);
    std::string out = "[kv-residency] rounds=" + std::to_string(c.rounds) +
                      " denom=rounds pages_per_round_last=" + std::to_string(c.pages_per_round_last) +
                      " resident_pages_total=" + std::to_string(c.resident_pages_total) +
                      " resident_blocks_total=" + std::to_string(c.resident_blocks_total) +
                      " retired_pages_total=" + std::to_string(c.retired_pages_total) +
                      " evaluated_pages_total=" + std::to_string(c.evaluated_pages_total) +
                      " requested_pages_total=" + std::to_string(c.requested_pages_total) +
                      " rounds_with_no_residency=" + std::to_string(c.rounds_with_no_residency) +
                      " mean_resident_pages_per_round=";
    if (mean.has_value()) {
        out += std::to_string(*mean);
    } else {
        out += std::string(kKvResidencyUnobtained) + "(denom=rounds, rounds=0)";
    }
    out += " saved_pages=";
    const std::optional<std::uint64_t> saved = kv_residency_saved_pages(c);
    if (saved.has_value()) {
        out += std::to_string(*saved) + " denom=rounds_x_pages_per_round=" +
               std::to_string(kv_residency_census_denominator(c)) + "x" +
               std::to_string(c.pages_per_round_last);
    } else {
        // The blank carries its own recipe. "Needs the engine" is printed as a field name, so the
        // person who owns the engine side can see exactly which reading they have to supply.
        out += std::string(kKvResidencyUnobtained) +
               "(needs baseline_resident_pages_total for the same run; designed denom="
               "rounds_x_pages_per_round=" +
               std::to_string(kv_residency_census_denominator(c)) + "x" +
               std::to_string(c.pages_per_round_last) + ")";
    }
    return out;
}

// The census is READY to answer "saved" only when both halves are present. Callers that must print
// a number are told by name that they cannot.
[[nodiscard]] inline bool kv_residency_census_can_answer_saved(
    const KvResidencyCensus& c) noexcept {
    return kv_residency_saved_pages(c).has_value();
}

} // namespace product
} // namespace ninfer
