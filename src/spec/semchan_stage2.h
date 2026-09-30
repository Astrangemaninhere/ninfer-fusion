#ifndef NINFER_SPEC_SEMCHAN_STAGE2_H
#define NINFER_SPEC_SEMCHAN_STAGE2_H

// ===========================================================================================
// STAGE2 (line `lookwire`) -- THE SECOND STAGE OF THE CANDIDATE-SOURCE SEAM.
// ===========================================================================================
//
// WHAT STAGE ONE LEFT OPEN, IN ITS OWN WORDS. `spec/semchan_wire.h` section 5 says, of the union it
// computes: "It does not widen what gets restored. The union it computes is REPORTED, and the token
// span the engine actually recalls is the arbiter's unchanged. **A second, separate switch would be
// needed to act on the union, and this line did not add one**: the measurement in `REPORT.md` is
// the reading that decides whether such a switch is worth its bytes."
//
// This header is that switch, and the first thing it does is decide whether it is worth its bytes --
// because the honest answer on today's tree is NO, and the refusal is the delivery.
//
// ===========================================================================================
// 1. THE STRUCTURAL FACT THAT DECIDES EVERYTHING BELOW (read off the tree, not assumed)
// ===========================================================================================
//
// The brief for this line named the missing piece as "feed the nominated candidates into the
// RETURNED `RecallRequest`'s candidate set". THAT PHRASE DESCRIBES SOMETHING THAT DOES NOT EXIST,
// and the whole shape of the second stage follows from it. Three readings, each a coordinate:
//
//   (a) `RecallRequest` IS NOT A SET. `spec/turn_recall_journal.h:1235-1238`:
//           struct RecallRequest {
//               std::uint32_t token_begin = 0; // absolute, in the sequence's own token space
//               std::uint32_t token_end = 0;   // exclusive
//           };
//       Two integers. ONE CONTIGUOUS SPAN. No container, no page list, no cap -- so there is no
//       field a nominated page could enter and no place a union could be written.
//
//   (b) ITS CONSUMER TURNS THE SPAN INTO A CONTIGUOUS PAGE RANGE. `program_impl.h:14307-14311`:
//           plan_request.wanted_begin_page = request.token_begin / page_tokens;
//           plan_request.wanted_end_page   = (request.token_end + page_tokens - 1U) / page_tokens;
//       One `[begin, end)` page interval. `plan_recall_pages_with_policy`
//       (`turn_recall_journal.h:1540-1670`) then builds `plan.pages` as the gap-free sub-run of it,
//       cutting at the first hole (`:1569-1582`, "a partial run is still a valid (shorter)
//       contiguous run") and truncating only from the low end (`:1622`, "keep the NEWEST pages
//       (drop the oldest), so the result stays gap-free").
//
//   (c) THE TREE SAYS THE CONTIGUITY IS LOAD-BEARING. `turn_recall_journal.h:1259-1264`: the
//       selected set is "a *contiguous, gap-free* run, which is the only shape `warm_cold_prefix`
//       can restore today".
//
// THEREFORE: a nominated page OUTSIDE the arbiter's run cannot be added to the recall. It can only
// be ENCLOSED -- by growing the run upward until it contains the nomination, which drags in EVERY
// page in between. **The union route through `RecallRequest` is a route that COSTS KV, not one that
// saves it.** That is why `dl/kvwire`'s landed seam measured `saved_pages = 0`, and why
// `dl/kvfinish` REPORT section 4.2 can say the same from the other side: the candidate set "never
// changes the fetch range, so it cannot change the resident set".
//
// ===========================================================================================
// 2. WHAT THIS HEADER THEREFORE DOES -- A COSTED, NAMED, REFUSING SECOND STAGE
// ===========================================================================================
//
// It does not pretend the union can be unioned into a set. It computes THE ONE ACTION THE TYPE
// ADMITS -- an enclosing span extension -- and it applies the tree's own edge policy,
// `RefuseNotTruncate`, to the DECISION: an extension whose gap pages cost at least as much as the
// nominations it buys is REFUSED BY NAME, and the refusal carries the geometry that justifies it.
// The switch defaults OFF, so the default path is the pre-stage2 expression verbatim.
//
// ===========================================================================================
// 3. THE MULTI-LEVEL CRITERION (the owner's own design order) -- NOMINATION vs DELIVERY
// ===========================================================================================
// The owner's design (`sum_dir.h:3-14`, verbatim): the DIRECTORY's rows "ARE THE RECALL BLOCKS",
// one catalogue line per block, and "that line -- not a similarity score over the block's bytes --
// becomes the key a later recall is targeted with ... you look a thing up BY ITS DESCRIPTION".
// The sizing that makes it cheap is in the same header: a row is `kSumDirRowWireBytes` = 80 B
// (`:511`) for a block of `kSumDirBlockTokens` = 64 tokens, while that block's KV at the codec
// constant the tree ASSERTS (`turn_recall_journal.h:130`, `kRecallNvfp4BytesPerToken = 18432`) is
// 64 x 18,432 = 1,179,648 B -- 14,745 : 1.
//
//   => THE ROWS STAY RESIDENT, so NOMINATION NEVER PAYS A DISK READ; ONLY DELIVERY DOES.
//
// The delivery route is chosen against the tree's own exchange rates (`turn_recall_journal.h:50-60`,
// quoted and not re-derived): batched SSD read 2.6-6.1 us/token against a re-prefill at
// 365-690 us/token, i.e. **batched read : re-prefill = 60-260 : 1**. So the SHORT part can be
// FETCHED and the LONG part should be RE-PREFILLED -- and a re-prefill regenerates at natural new
// positions, so the KVMI-011 re-RoPE drift does not arise on that route at all.
//
// AND ONE COST THE CRITERION MUST CARRY, named in the tree and easy to omit: `program_impl.h`
// :1766-1776 -- the host tier does "an EAGER pin of the FULL `--cold-host-bytes` cap, allocated up
// front", and its own text says "The pre-run MemAvailable gate cannot see this allocation." So a
// level that is "in memory" is not free and its pages are not marginal.

#include "spec/semchan_wire.h"
#include "spec/sum_dir.h"
#include "spec/sum_dir_reach.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace ninfer::spec::semchan::stage2 {

// -------------------------------------------------------------------------------------------
// 1. THE THIRD SWITCH -- SUBORDINATE TO THE OTHER TWO, OFF UNLESS IT SAYS EXACTLY "1"
// -------------------------------------------------------------------------------------------
// The same strict spelling as its two siblings (`semchan_switch_value_is_on`, `semchan_wire.h:80`):
// the accepted ON spelling is the single byte "1"; unset, "", "0", "01", "true", "on" and any typo
// are OFF. It is SUBORDINATE: an extension can only ride a wire already carrying the additive
// route, so this switch is false whenever `NINFER_SEMCHAN_ALTERNATIVES` is not on.
inline constexpr const char* kSemChanSpanSwitchName = "NINFER_SEMCHAN_SPAN";

[[nodiscard]] inline bool semchan_span_enabled() noexcept {
    static const bool on = wire::semchan_switch_value_is_on(std::getenv(kSemChanSpanSwitchName));
    return on && wire::semchan_alternatives_enabled();
}

// -------------------------------------------------------------------------------------------
// 2. THE EXTENSION -- THE ONLY ACTION `RecallRequest` ADMITS, COSTED AND REFUSED BY NAME
// -------------------------------------------------------------------------------------------
// THE VERDICT VOCABULARY IS CLOSED AND EVERY MEMBER CARRIES ITS REASON. There is no branch that
// returns an admissible extension without also printing what it costs.
enum class SemChanSpanVerdict : std::uint8_t {
    NotAsked = 0,         // the switch is OFF -- the default, and the pre-stage2 expression
    NoArbiterRun = 1,     // the arbiter did not find: there is no run to extend
    NothingToBuy = 2,     // every nomination is already inside the arbiter's own run: cost 0, gain 0
    OverBudget = 3,       // the widened run exceeds the per-pass fan-out cap
    FreeEnclosure = 4,    // every page between is itself nominated: the enclosure has no gap
    BoughtAtAPrice = 5,   // gap > 0 but gap < nominations bought: admissible, price printed
    RefusedNetLoss = 6,   // gap >= nominations bought: paying at least as many pages as it buys
};

[[nodiscard]] constexpr const char* semchan_span_verdict_name(SemChanSpanVerdict v) noexcept {
    switch (v) {
    case SemChanSpanVerdict::NotAsked: return "not-asked";
    case SemChanSpanVerdict::NoArbiterRun: return "no-arbiter-run";
    case SemChanSpanVerdict::NothingToBuy: return "nothing-to-buy";
    case SemChanSpanVerdict::OverBudget: return "over-budget";
    case SemChanSpanVerdict::FreeEnclosure: return "free-enclosure";
    case SemChanSpanVerdict::BoughtAtAPrice: return "bought-at-a-price";
    case SemChanSpanVerdict::RefusedNetLoss: return "refused-net-loss";
    }
    return "unknown";
}

// THE RESULT. Every count is printed with a denominator beside it and every byte figure with the
// codec it was priced under, because a saving without its denominator is a story.
struct SemChanSpanExtension {
    bool               asked   = false;
    SemChanSpanVerdict verdict = SemChanSpanVerdict::NotAsked;
    std::string        reason  = "the switch is OFF -- and OFF is the default";

    // ---- the geometry, in pages, with `span_pages` as the denominator for all of it ----
    std::uint32_t span_first_page    = 0;
    std::uint32_t span_last_page     = 0;
    std::uint32_t span_pages         = 0; // the arbiter's own run: DENOMINATOR
    std::uint32_t nominated          = 0; // records the union handed in
    std::uint32_t nominated_distinct = 0; // PAGES -- a different set from the count above
    std::uint32_t nominated_inside   = 0; // already inside the run: cost 0, gain 0
    std::uint32_t nominated_outside  = 0; // the only pages an enclosure can BUY
    std::uint32_t nominal_below_run  = 0; // nominations below the run: NOT enclosed, counted anyway
    std::uint32_t widened_pages      = 0; // the enclosing run's size, before the cap
    std::uint32_t gap_pages          = 0; // widened - span - bought: the price, in pages

    // ---- the extension itself as a `RecallRequest`, and `token_begin` NEVER MOVES UP ----
    // The extension can only ADD pages ABOVE `last_page`, so `token_begin` is the arbiter's own
    // value in every branch. That is the property that keeps the anchor's end of the run intact;
    // it is asserted below and MEASURED in the harness, not promised here.
    std::uint32_t arbiter_token_begin = 0; // what the arbiter said: the fallback, always
    std::uint32_t arbiter_token_end   = 0;
    std::uint32_t widened_token_begin = 0; // the COUNTERFACTUAL enclosure -- what it WOULD be
    std::uint32_t widened_token_end   = 0;
    // ⭐ THE EFFECTIVE REQUEST, AND IT IS THE ONLY THING A CALLER MAY RETURN. On every refusal the
    // arbiter's own span, exactly as if this header did not exist -- so a refusal is not merely
    // "reported", it is the value. Reporting the counterfactual span where the effective one is
    // meant would make a refusal read as an action, which is the too-loose pass this station
    // refuses.
    std::uint32_t token_begin       = 0;
    std::uint32_t token_end         = 0;
    bool          begin_is_arbiters = true;
    bool          admissible        = false; // the criterion's yes/no, one flag, never inferred
    bool          request_is_arbiters = true; // true <=> the effective request is the arbiter's own

    // ---- the price in the TIER's own units, so no two units are ever added ----
    std::uint32_t block_tokens       = sum_dir::kSumDirBlockTokens;
    std::uint32_t bytes_per_token    = 0; // the tier's own codec constant
    std::uint64_t kv_bytes_added     = 0; // (widened - span) * block_tokens * bytes_per_token
    std::uint64_t row_bytes_resident = 0; // nominated_distinct * kSumDirRowWireBytes
};

// THE DECISION, AS A PURE FUNCTION OF ITS INPUTS. It reads the ARBITER'S OWN result (`reach`, so it
// can never disagree with the run it is extending) and the pages the union handed in (which
// `semchan_wire.h` builds out of the arbiter's `alternative_pages` and nowhere else).
//
// ⭐ THERE IS NO SCORE PARAMETER. The same property `spec/sum_dir_determined.h` states for itself --
// "a KEY and an id SPAN and no score of any kind: there is no parameter a score could arrive
// through" -- holds here by signature: a result struct, a page set, and three integers, every one
// of them a fact about GEOMETRY or about STORAGE SIZE.
[[nodiscard]] inline SemChanSpanExtension semchan_span_extension_from_union(
    const sum_dir::SumDirReachResult& reach, const std::vector<std::uint32_t>& nominated,
    std::uint32_t block_tokens, std::uint32_t bytes_per_token, std::uint32_t fanout_cap,
    bool asked = semchan_span_enabled()) {
    SemChanSpanExtension out;
    out.asked           = asked;
    out.block_tokens    = block_tokens == 0U ? sum_dir::kSumDirBlockTokens : block_tokens;
    out.bytes_per_token = bytes_per_token;
    out.nominated       = static_cast<std::uint32_t>(nominated.size());
    out.arbiter_token_begin = reach.token_begin;
    out.arbiter_token_end   = reach.token_end;

    // THE ONE SETTLE POINT. Every exit runs through it, so "the effective request" cannot drift
    // from "the verdict": on any refusal it is the arbiter's own span, byte for byte.
    const auto settle = [&out]() -> SemChanSpanExtension {
        out.token_begin         = out.admissible ? out.widened_token_begin : out.arbiter_token_begin;
        out.token_end           = out.admissible ? out.widened_token_end : out.arbiter_token_end;
        out.begin_is_arbiters   = (out.token_begin == out.arbiter_token_begin);
        out.request_is_arbiters = (out.token_begin == out.arbiter_token_begin &&
                                   out.token_end == out.arbiter_token_end);
        return out;
    };

    if (!asked) {
        out.verdict  = SemChanSpanVerdict::NotAsked;
        out.reason   = "the switch is OFF -- and OFF is the default";
        return settle();
    }
    if (!reach.found) {
        out.verdict  = SemChanSpanVerdict::NoArbiterRun;
        out.reason   = "the arbiter did not find a run, so there is no span to extend: a candidate "
                       "source cannot create a run the arbiter refused to";
        return settle();
    }

    out.span_first_page = reach.first_page;
    out.span_last_page  = reach.last_page;
    out.span_pages      = (reach.last_page >= reach.first_page)
                              ? (reach.last_page - reach.first_page + 1U)
                              : 0U;

    // The DISTINCT pages, then the split. A page inside the run is not a purchase: buying it would
    // be paying for something the run already holds.
    std::vector<std::uint32_t> distinct;
    for (const std::uint32_t page : nominated) {
        if (std::find(distinct.begin(), distinct.end(), page) == distinct.end()) {
            distinct.push_back(page);
        }
    }
    out.nominated_distinct = static_cast<std::uint32_t>(distinct.size());

    std::uint32_t hi = out.span_last_page;
    for (const std::uint32_t page : distinct) {
        if (page >= out.span_first_page && page <= out.span_last_page) {
            ++out.nominated_inside;
        } else if (page > out.span_last_page) {
            ++out.nominated_outside;
            hi = std::max(hi, page);
        } else {
            // ⚠ A NOMINATION BELOW THE RUN IS COUNTED AND NOT ENCLOSED, BY NAME. Enclosing it
            // would move `token_begin` DOWN -- i.e. it would need a different run, not a longer
            // one, and this function buys extensions, never relocations. The page is still a real
            // nomination, so it is counted rather than quietly dropped.
            ++out.nominal_below_run;
        }
    }
    out.row_bytes_resident =
        static_cast<std::uint64_t>(out.nominated_distinct) * sum_dir::kSumDirRowWireBytes;

    if (out.nominated_outside == 0U) {
        out.verdict       = SemChanSpanVerdict::NothingToBuy;
        out.reason        = "no nominated page lies above the arbiter's run: the enclosure buys "
                            "nothing and costs nothing, so it is not taken";
        out.widened_pages = out.span_pages;
        out.widened_token_begin = out.span_first_page * out.block_tokens;
        out.widened_token_end   = (out.span_last_page + 1U) * out.block_tokens;
        out.admissible    = false;
        return settle();
    }

    out.widened_pages = hi - out.span_first_page + 1U;
    // THE PRICE, IN PAGES: what the widened run would hold that is neither the arbiter's run nor a
    // nomination this enclosure buys.
    const std::uint32_t bought_pages = out.nominated_outside;
    out.gap_pages = (out.widened_pages > out.span_pages + bought_pages)
                        ? (out.widened_pages - out.span_pages - bought_pages)
                        : 0U;

    out.widened_token_begin = out.span_first_page * out.block_tokens;
    out.widened_token_end   = (hi + 1U) * out.block_tokens;
    out.kv_bytes_added      = static_cast<std::uint64_t>(out.widened_pages - out.span_pages) *
                              static_cast<std::uint64_t>(out.block_tokens) *
                              static_cast<std::uint64_t>(out.bytes_per_token);

    if (out.widened_pages > fanout_cap) {
        out.verdict    = SemChanSpanVerdict::OverBudget;
        out.reason     = "the widened run is " + std::to_string(out.widened_pages) +
                         " page(s) against the per-pass fan-out cap " + std::to_string(fanout_cap) +
                         " -- refused rather than truncated";
        out.admissible = false;
        return settle();
    }
    if (out.gap_pages == 0U) {
        out.verdict    = SemChanSpanVerdict::FreeEnclosure;
        out.admissible = true;
        out.reason     = "every page between the run's edge and the farthest nomination is itself "
                         "nominated: the enclosure has no gap, so it buys the nominations at the "
                         "price of exactly the pages it names";
        return settle();
    }
    // ⭐ THE EDGE, AND IT IS `RefuseNotTruncate` APPLIED TO A PURCHASE. Paying at least as many
    // gap pages as nominated pages is not a saving and not a wash: it is more KV held than the
    // nomination is worth. The tree's rule at the edge is refusal, not approximation, so this is a
    // refusal -- and it names the two numbers it compared.
    if (out.gap_pages >= bought_pages) {
        out.verdict    = SemChanSpanVerdict::RefusedNetLoss;
        out.admissible = false;
        out.reason     = "the enclosure would add " +
                         std::to_string(out.widened_pages - out.span_pages) +
                         " page(s) to the run to buy " + std::to_string(bought_pages) +
                         " nominated page(s), i.e. " + std::to_string(out.gap_pages) +
                         " gap page(s) that neither the arbiter found nor the union named: the "
                         "price is at least the purchase, so it is refused rather than paid";
        return settle();
    }
    out.verdict    = SemChanSpanVerdict::BoughtAtAPrice;
    out.admissible = true;
    out.reason     = "the enclosure adds " + std::to_string(out.widened_pages - out.span_pages) +
                     " page(s) to buy " + std::to_string(bought_pages) + " nominated page(s), with " +
                     std::to_string(out.gap_pages) + " gap page(s) -- admissible, at the price named";
    return settle();
}

// ⭐ THE VALUE A CALLER RETURNS. This is the whole interface: the provider's `RecallRequest` is
// this function's output and nothing else's, so a refusal is the arbiter's own request rather than
// a widened one that a caller forgot not to use.
[[nodiscard]] inline turn_recall::RecallRequest
semchan_span_effective_request(const SemChanSpanExtension& e) noexcept {
    turn_recall::RecallRequest r;
    r.token_begin = e.token_begin;
    r.token_end   = e.token_end;
    return r;
}

// -------------------------------------------------------------------------------------------
// 2b. THE ONE ENTRY POINT THE ENGINE MAY CALL -- AND WHY IT TAKES NO PAGE SET
// -------------------------------------------------------------------------------------------
// `semchan_span_extension_from_union` above is general: a caller supplies the page list. THE ENGINE
// MUST NOT CALL IT WITH ANY LIST BUT THE ARBITER'S OWN, and rather than write that as a rule this
// header writes it as a SIGNATURE.
//
// WHY IT MATTERS, AND IT IS THE WHOLE DOCTRINE. The union `semchan_wire.h` computes has TWO
// columns: the ARBITER's pages (its run plus the `alternative_pages` it found and could not carry)
// and the CHANNEL's pages (built from `semchan_mean_k_score`). Both may be REPORTED. Only the
// first may be ACTED ON:
//   * the arbiter's pages are its own exact findings -- enclosing them adds recall the arbiter
//     itself established, and no score is involved anywhere in the path;
//   * the channel's pages are a similarity ranking. Letting one of them WIDEN THE RECALLED SPAN
//     would mean a score changed what the engine restores -- which is the one thing
//     `semchan_wire.h`'s section 2 pins as refused ("the channel may NOT determine"), and
//     `sum_dir_determined.h`'s "no parameter a score could arrive through" forbids structurally.
//     Detecting the difference matters: today the channel is REPORT-ONLY and that is a property of
//     the code, not of a promise -- `channel_only_pages` is printed and never handed to anything.
//
// SO THE ENGINE'S ENTRY POINT TAKES NO PAGE SET AT ALL: the nomination is read off the arbiter's
// own result, inside this function, where a caller cannot substitute one. This is the same move
// `sum_dir_reach.h` section 7d made for the naive route -- one choke point, not a convention.
[[nodiscard]] inline SemChanSpanExtension semchan_span_extension_from_reach(
    const sum_dir::SumDirReachResult& reach, std::uint32_t block_tokens,
    std::uint32_t bytes_per_token, std::uint32_t fanout_cap,
    bool asked = semchan_span_enabled()) {
    return semchan_span_extension_from_union(reach, reach.alternative_pages, block_tokens,
                                             bytes_per_token, fanout_cap, asked);
}

// A NAME for the property, so a reader can find it without reading this header's body. It is not a
// mechanism -- the mechanism is the signature above and the harness arm that measures what happens
// when a channel-derived page set is substituted by hand.
inline constexpr const char* kSemChanSpanNominationIsTheArbitersOwn =
    "the enclosure's nomination is read off `SumDirReachResult::alternative_pages`, in the "
    "function, from the arbiter's own result: no page set from anywhere else can arrive";

// -------------------------------------------------------------------------------------------
// 3. THE MULTI-LEVEL CRITERION -- NOMINATION FROM RESIDENT ROWS, DELIVERY BY THE ROUTE IT PICKS
// -------------------------------------------------------------------------------------------
// THREE LEVELS, and they are the owner's own three: a part of the SSD range comes back to memory
// EARLY, and the longer part STAYS ON SSD to be re-prefilled on demand. The rows are what stays
// resident; the KV is what gets decided about.
enum class SemChanLevel : std::uint8_t {
    ResidentRows = 0, // the 80 B catalogue lines: resident, so nomination never pays a disk read
    PrefetchKv   = 1, // the KV itself, brought into (pinned) host memory: the batched-read route
    SsdReprefill = 2, // the KV stays on SSD and the TEXT is re-prefilled when nominated
};

[[nodiscard]] constexpr const char* semchan_level_name(SemChanLevel l) noexcept {
    switch (l) {
    case SemChanLevel::ResidentRows: return "resident-rows";
    case SemChanLevel::PrefetchKv: return "prefetch-kv";
    case SemChanLevel::SsdReprefill: return "ssd-reprefill";
    }
    return "unknown";
}

// THE EXCHANGE RATES. ⚠⚠ THEY ARE A COST MODEL, NOT A MEASUREMENT, AND THE TREE SAYS SO ITSELF.
// This is the correction that decides how these numbers may be used, and it is quoted rather than
// discovered here: `turn_recall_journal.h:1779` reads, verbatim --
//   "!! THE TWO NUMBERS ON THE `[recall]` LINE ARE THE COST MODEL, NOT A MEASUREMENT. `read_ms()`
//    and `reprefill_ms()` are `tokens * <rate constant>` (the RecallCost defaults above, 6.1 and
//    365.0 us/token). The engine's own measured cost of the SAME 64 tokens is `prefill_ms=2218.87`
//    = 34,670 us/token, i.e. 95x the model's 365.0."
// And `net_positive()` (`:1726-1728`) compares two RATES, so `tokens` cancels and the predicate is
// the COMPILE-TIME CONSTANT `6.1 < 365.0` (`:1728-1731`: with the shipped defaults the caller's
// `return;` "CANNOT be reached", so `refused_cost=0` "is evidence that NO CHECK RAN" -- the state
// the Verdict enum names `PositiveDefaultConstant`, `:1742`).
// ⇒ SO: the 60-260:1 band is a MODEL TO BE VERIFIED. It is printed here ONLY as the model, labelled,
// and a caller that quotes a two-route comparison from it is quoting a model. The ONE measured pair
// on this tree is `kSemChanMeasuredLeg` below, and it is 95x apart from the model on the re-prefill
// side -- which is why the model must never be read as a ratio a real leg costs.
struct SemChanRouteCost {
    double      us_per_token_low  = 0.0;
    double      us_per_token_high = 0.0;
    const char* provenance        = "";
};
inline constexpr SemChanRouteCost kSemChanBatchedRead{
    2.6, 6.1, "COST MODEL: turn_recall_journal.h:52/1723 (TODO.md:5830-5831, :5884, :6182) -- SSD "
              "read of packed KV 2.6 us/token (7 GB/s) .. 6.1 us/token (3 GB/s). `read_ms()` is "
              "`tokens * 6.1/1000`, i.e. the model, NOT a measurement"};
inline constexpr SemChanRouteCost kSemChanReprefill{
    365.0, 690.0, "COST MODEL: turn_recall_journal.h:53/1726 -- re-prefill the same token 365 "
                  "us/token (2740 tok/s, in-tree load) .. 690 us/token. `reprefill_ms()` is "
                  "`tokens * 365.0/1000`: the tree's own measured cost of the SAME 64 tokens is "
                  "34,670 us/token = 95x this constant (turn_recall_journal.h:1781-1784)"};
inline constexpr const char* kSemChanRouteRatioProvenance =
    "COST MODEL, NOT A MEASUREMENT: turn_recall_journal.h:57 states 'batched read : re-prefill = "
    "60-260 : 1 (buy compute with bandwidth, cheaply)', and the same file at :1779 states that the "
    "two numbers it is derived from ARE THE COST MODEL. Do not quote this band as a measured ratio";
// ⭐ THE ONE MEASURED PAIR ON THIS TREE, for the SAME 64 tokens, quoted with its run and its line.
// Both numbers are read off the engine's own `[recall]` line, and the file that carries them says
// only `prefill_ms` (on the `[context-append]` line) is the measurement.
struct SemChanMeasuredLeg {
    std::uint32_t tokens          = 64;
    double        read_ms         = 0.390400;  // MODEL: 64 * 6.1/1000
    double        reprefill_ms    = 2218.87;   // MEASURED: `[context-append] ... prefill_ms=2218.87`
    double        reprefill_us_per_token = 34670.0;
    const char*   provenance =
        "MEASURED leg: turn_recall_journal.h:1773-1777, quoting dl/fusionrefresh/runs/"
        "A2_armed_fixed at the 1M geometry, `--cold-policy disk`: "
        "'[context-append] lane=0 tokens=64 source=text-cargo blocks=1 cargo_bytes=256 "
        "prefill_ms=2218.87' and '[recall] pages=1 bytes=1232896 restored=1 read_ms=0.390400 "
        "reprefill_ms=23.360000 ratio=59.836066:1 hook_ms=2225.490'. `read_ms` is the model and "
        "`reprefill_ms` on that line is the model; `prefill_ms` is the measurement";
};
inline constexpr SemChanMeasuredLeg kSemChanMeasuredLeg{};
// The measured comparison, computed from the two numbers above rather than from the model's band.
[[nodiscard]] constexpr double semchan_measured_ratio(const SemChanMeasuredLeg& m) noexcept {
    return m.read_ms > 0.0 ? m.reprefill_ms / m.read_ms : 0.0;
}
// By how much the model's re-prefill rate is optimistic against the measured one.
[[nodiscard]] constexpr double semchan_model_optimism(const SemChanMeasuredLeg& m) noexcept {
    return kSemChanReprefill.us_per_token_low > 0.0
               ? m.reprefill_us_per_token / kSemChanReprefill.us_per_token_low
               : 0.0;
}
inline constexpr const char* kSemChanHostPinProvenance =
    "NOT MINE: program_impl.h:1766-1776 -- 'an EAGER pin of the FULL --cold-host-bytes cap, "
    "allocated up front ... The pre-run MemAvailable gate cannot see this allocation.'";

struct SemChanTierCriterion {
    // ---- inputs, all named ----
    std::uint32_t blocks                 = 0; // DENOMINATOR for every count below
    std::uint32_t block_tokens           = sum_dir::kSumDirBlockTokens;
    std::uint32_t row_wire_bytes         = sum_dir::kSumDirRowWireBytes;
    std::uint32_t bytes_per_token        = 0;
    std::uint32_t nominated              = 0; // blocks the directory's rows nominated
    std::uint32_t prefetch_budget_blocks = 0; // how many blocks may be held in host memory
    std::uint64_t host_pin_bytes         = 0; // the EAGER pin, if the host tier is on: not marginal
    // ---- the decision, per level, with the denominator riding along ----
    bool          asked                = false;
    std::uint32_t resident_row_blocks  = 0;
    std::uint32_t prefetch_kv_blocks   = 0;
    std::uint32_t ssd_reprefill_blocks = 0;
    std::uint64_t resident_row_bytes   = 0;
    std::uint64_t kv_bytes_in_memory   = 0;
    std::uint64_t kv_bytes_left_on_ssd = 0;
    // ---- `saved_pages`: the numerator, its denominator, and both routes priced on the SAME set ----
    std::uint64_t saved_pages              = 0;
    std::uint32_t saved_pages_denom_blocks = 0;
    double        fetch_ms_for_saved       = 0.0;
    double        reprefill_ms_for_saved   = 0.0; // the MODEL's re-prefill price
    double        reprefill_ms_measured_equivalent = 0.0; // the MEASURED rate, same token set
    std::string   route_picked             = "none";
    std::string   accuracy_reading         = "UNOBTAINED";
    std::string   accuracy_provenance      = "caller-supplied; this header fabricates no reading";
};

// THE CRITERION. Rows are always resident -- that is the whole point of a directory, and it is what
// keeps nomination off the disk. The KV is then split: the newest `prefetch_budget_blocks` of the
// NOMINATED set are brought into memory (the batched-read route), and the rest stays on SSD to be
// re-prefilled -- because the tree's own rate says a re-prefill costs 60-260x a batched read, so
// the LONG part is the part that should be re-prefilled rather than fetched; and a re-prefill
// regenerates at natural new positions, so KVMI-011's re-RoPE drift does not arise on that route.
[[nodiscard]] inline SemChanTierCriterion semchan_tier_criterion(
    std::uint32_t blocks, std::uint32_t bytes_per_token, std::uint32_t nominated,
    std::uint32_t prefetch_budget_blocks, std::uint64_t host_pin_bytes, bool asked,
    const char* accuracy_reading = "UNOBTAINED",
    const char* accuracy_provenance = "caller-supplied; this header fabricates no reading") {
    SemChanTierCriterion out;
    out.asked                    = asked;
    out.blocks                   = blocks;
    out.bytes_per_token          = bytes_per_token;
    out.nominated                = nominated;
    out.prefetch_budget_blocks   = prefetch_budget_blocks;
    out.host_pin_bytes           = host_pin_bytes;
    out.accuracy_reading         = accuracy_reading;
    out.accuracy_provenance      = accuracy_provenance;
    out.saved_pages_denom_blocks = blocks;

    // Level A: the rows. Always resident, and their size is the reason the design exists at all.
    out.resident_row_blocks = blocks;
    out.resident_row_bytes  = static_cast<std::uint64_t>(blocks) *
                             static_cast<std::uint64_t>(out.row_wire_bytes);

    // Levels B and C: the KV, split by the budget. Both are COUNTED and never added -- they are
    // two different levels, not two readings of one.
    const std::uint32_t budget = prefetch_budget_blocks;
    out.prefetch_kv_blocks     = nominated < budget ? nominated : budget;
    out.ssd_reprefill_blocks   = nominated - out.prefetch_kv_blocks;
    const std::uint64_t bytes_per_block =
        static_cast<std::uint64_t>(out.block_tokens) * static_cast<std::uint64_t>(bytes_per_token);
    out.kv_bytes_in_memory = static_cast<std::uint64_t>(out.prefetch_kv_blocks) * bytes_per_block;
    out.kv_bytes_left_on_ssd =
        static_cast<std::uint64_t>(out.ssd_reprefill_blocks) * bytes_per_block;

    // ⭐ `saved_pages`: the pages NOT resident because their ROW is resident instead. Its
    // denominator is the block count this decision was taken over, and it is a monotone count over
    // one run -- never a ratio of two different units.
    out.saved_pages = out.ssd_reprefill_blocks;
    // The two routes priced on the SAME set (`ssd_reprefill_blocks x block_tokens` tokens), so the
    // two numbers are comparable. ⚠ BOTH ARE THE COST MODEL -- see `kSemChanBatchedRead` /
    // `kSemChanReprefill` above and the tree's own statement at `turn_recall_journal.h:1779`. The
    // measured pair for the same 64 tokens is carried beside them and never substituted for them.
    const double tokens = static_cast<double>(out.ssd_reprefill_blocks) *
                          static_cast<double>(out.block_tokens);
    out.fetch_ms_for_saved     = tokens * kSemChanBatchedRead.us_per_token_low / 1000.0;
    out.reprefill_ms_for_saved = tokens * kSemChanReprefill.us_per_token_low / 1000.0;
    out.reprefill_ms_measured_equivalent =
        tokens * kSemChanMeasuredLeg.reprefill_us_per_token / 1000.0;
    out.route_picked = (out.ssd_reprefill_blocks == 0U)
                           ? std::string("none-required (nothing was left on SSD)")
                           : std::string("ssd-reprefill (the LONG part) -- at the MODEL's rate; the "
                                         "MEASURED re-prefill is 95x that rate, so this choice is "
                                         "priced by a model the tree says is not a measurement");
    return out;
}

[[nodiscard]] inline std::string semchan_span_line(const SemChanSpanExtension& e) {
    std::string out = "[semchan-span] switch=";
    out += kSemChanSpanSwitchName;
    out += " enabled=";
    out += (e.asked ? "1" : "0");
    out += " verdict=";
    out += semchan_span_verdict_name(e.verdict);
    out += " span_pages=" + std::to_string(e.span_pages);
    out += " span=[" + std::to_string(e.span_first_page) + "," + std::to_string(e.span_last_page) +
           "]";
    out += " nominated=" + std::to_string(e.nominated);
    out += " nominated_distinct=" + std::to_string(e.nominated_distinct);
    out += " nominated_inside=" + std::to_string(e.nominated_inside);
    out += " nominated_outside=" + std::to_string(e.nominated_outside);
    out += " nominal_below_run=" + std::to_string(e.nominal_below_run);
    out += " widened_pages=" + std::to_string(e.widened_pages);
    out += " gap_pages=" + std::to_string(e.gap_pages);
    out += " admissible=";
    out += (e.admissible ? "1" : "0");
    out += " effective_request=[" + std::to_string(e.token_begin) + "," +
           std::to_string(e.token_end) + ")";
    out += " request_is_arbiters=";
    out += (e.request_is_arbiters ? "1" : "0");
    out += " counterfactual_request=[" + std::to_string(e.widened_token_begin) + "," +
           std::to_string(e.widened_token_end) + ")";
    out += " begin_is_arbiters=";
    out += (e.begin_is_arbiters ? "1" : "0");
    out += " kv_bytes_added=" + std::to_string(e.kv_bytes_added);
    out += " kv_bytes_per_token=" + std::to_string(e.bytes_per_token);
    out += " row_bytes_resident=" + std::to_string(e.row_bytes_resident);
    out += " determined_by_span=0";
    out += " reason=";
    out += e.reason;
    return out;
}

[[nodiscard]] inline std::string semchan_tier_line(const SemChanTierCriterion& c) {
    std::string out = "[semchan-tier] blocks=" + std::to_string(c.blocks);
    out += " block_tokens=" + std::to_string(c.block_tokens);
    out += " row_wire_bytes=" + std::to_string(c.row_wire_bytes);
    out += " bytes_per_token=" + std::to_string(c.bytes_per_token);
    out += " resident_row_blocks=" + std::to_string(c.resident_row_blocks);
    out += " resident_row_bytes=" + std::to_string(c.resident_row_bytes);
    out += " nominated=" + std::to_string(c.nominated);
    out += " prefetch_kv_blocks=" + std::to_string(c.prefetch_kv_blocks);
    out += " ssd_reprefill_blocks=" + std::to_string(c.ssd_reprefill_blocks);
    out += " kv_bytes_in_memory=" + std::to_string(c.kv_bytes_in_memory);
    out += " kv_bytes_left_on_ssd=" + std::to_string(c.kv_bytes_left_on_ssd);
    out += " host_pin_bytes=" + std::to_string(c.host_pin_bytes);
    out += " saved_pages=" + std::to_string(c.saved_pages);
    out += " saved_pages_denom_blocks=" + std::to_string(c.saved_pages_denom_blocks);
    out += " route_picked=" + c.route_picked;
    out += " fetch_ms_MODEL_for_same_set=" + std::to_string(c.fetch_ms_for_saved);
    out += " reprefill_ms_MODEL_for_same_set=" + std::to_string(c.reprefill_ms_for_saved);
    out += " reprefill_ms_MEASURED_equivalent=" +
           std::to_string(c.reprefill_ms_measured_equivalent);
    out += " model_optimism_x=" + std::to_string(semchan_model_optimism(kSemChanMeasuredLeg));
    out += " measured_read_over_reprefill_ratio=" +
           std::to_string(semchan_measured_ratio(kSemChanMeasuredLeg));
    out += " accuracy_reading=" + c.accuracy_reading;
    out += " accuracy_provenance=" + c.accuracy_provenance;
    return out;
}

// -------------------------------------------------------------------------------------------
// 4. THE POSITIONAL IN-USE SIGNAL -- THE 指向, AND WHERE IT CANNOT REACH TODAY
// -------------------------------------------------------------------------------------------
// THE OWNER'S DESIGN, IN HIS OWN WORDS (relayed verbatim by the coordinator, 2026-09-24):
// 「那个lookup的复用，就是那个给mtp加速的东西；这个直接存kv不合适，但是我想借用」 -- BORROW the lookup
// that already accelerates MTP; do not build a new per-block index, and do not store KV for it.
//
// THE LOOKUP HE MEANS IS REAL AND IS IN THIS TREE, READ IN FULL BEFORE THIS SECTION WAS WRITTEN:
// `spec/lookup_fuse.h` (80 lines). `SuffixHit{length, offset}` is a SUFFIX MATCH of the query
// window against the history (`suffix_best`, :29-51), and on a FULL match `fuse_chain` (:56-78)
// fills K drafts STRAIGHT FROM THE HISTORY -- its own comment says so: "上下文整段来自历史, 草稿器不
// 参与" (the whole span comes from the history; the drafter does not take part). Its ONE
// engine-side caller is `program_impl.h:15760` inside `index_draft_fill()`
// (`program_impl.h:15719-15789`), reached only when `NINFER_INDEX_DRAFTS` says exactly "1"
// (`:15707-15713`), once per lane per round, on the host.
//
// ⭐ AND `SumDirReachResult` ALREADY CARRIES THE SAME FACT AS A NAMED SPAN. `sum_dir_reach.h`
// :307-315 (inside "ABSOLUTE REACHABILITY: a COVERING retrieval family") sets
//     result.window = best_length;  result.best_offset = evidence[champion].offset;
//     result.covered_begin = result.best_offset;
//     result.covered_end   = result.best_offset + best_length;
// so `covered_begin`/`covered_end` IS the lookup's own hit expressed as a token span, in the
// reach result the arbiter already returns. ONE lookup, TWO USES: it fills drafts for MTP, and it
// points at the KV that must stay resident.
//
// ⭐⭐ THE CRITERION THIS BUYS IS POSITIONAL AND PRECISE -- "this block lies inside the covered
// span" -- NOT statistical ("this page is old", or "this page is similar"). That distinction is
// the whole reason it belongs in this tree: a positional fact needs no score, and it is falsifiable
// against two integers.
//
// ⚠⚠ WHERE IT CANNOT REACH TODAY, PROVEN BY A SIGNATURE RATHER THAN BY A FAILED GREP.
// The unload pass (`program_impl.h:13078-13330`, the owner's watermark order of 2026-09-15) decides
// with `spec::sum_dir::sum_dir_block_admissibility` (`sum_dir.h:1068-1087`), a rank-argmax over
// four keep reasons driven by `sum_dir_avl_rank` (`:996-1007`), one of which is
// `SumDirAdmissibility::PinnedByAnchor` (rank 3, `:1001`) -- filled from
// `SumDirBlockFacts::pinned_by_anchor` ("an anchor / checkpoint frontier sits inside it", `:1031`).
// And the facts are built by
//     `sum_dir.h:1250`  sum_dir_facts_from_sequence(std::uint32_t token_count) noexcept
// called at
//     `program_impl.h:13276`  spec::sum_dir::sum_dir_facts_from_sequence(sequence.text_kv_valid);
// ⇒ THE FUNCTION TAKES ONE ARGUMENT AND IT IS A TOKEN COUNT. No span can arrive through it, at any
// call site, because there is no parameter for one. So `pinned_by_anchor` is false on every block
// and the pass's own line prints `dist.pinned-by-anchor=0` -- which is what the engine's counter
// says, not what a comment says.
//
// ⇒ CLOSING IT IS A SMALL CHANGE AND THIS SECTION IS ITS CRITERION: a facts builder that ALSO
// receives `[covered_begin, covered_end)`, and a block whose token range INTERSECTS that span is
// in use. Nothing else moves: the rank table, the argmax and `sum_dir_is_unloadable` are all the
// tree's own and are CALLED here, not re-written.
struct SemChanInUseSpan {
    bool          present       = false; // the arbiter found, so there is a covered span at all
    std::uint32_t window        = 0;     // the lookup fit length in tokens: the span's DENOMINATOR
    std::uint32_t covered_begin = 0;
    std::uint32_t covered_end   = 0;     // exclusive
    std::uint32_t block_tokens  = sum_dir::kSumDirBlockTokens;
    // ---- the decision, over a set of blocks, with the block count as the denominator ----
    std::uint32_t blocks_evaluated   = 0;
    std::uint32_t blocks_in_use      = 0; // intersect the covered span
    std::uint32_t blocks_not_in_use  = 0;
    std::uint32_t unloadable_before  = 0; // under the facts as `sum_dir_facts_from_sequence` builds them
    std::uint32_t unloadable_after   = 0; // with the in-use fact added
    // `unloadable_before - unloadable_after` == blocks_in_use, asserted in the self-test: a block
    // the signal protects must stop being unloadable, and a block it does not touch must not.
};

// THE 指向 ITSELF: the lookup's hit, read off the arbiter's own result.
[[nodiscard]] inline SemChanInUseSpan
semchan_in_use_span_from_reach(const sum_dir::SumDirReachResult& reach) noexcept {
    SemChanInUseSpan out;
    out.present       = reach.found;
    out.window        = reach.window;
    out.covered_begin = reach.found ? reach.covered_begin : 0U;
    out.covered_end   = reach.found ? reach.covered_end : 0U;
    return out;
}

// THE POSITIONAL TEST, and it is an interval overlap -- two comparisons, no score, no tolerance.
[[nodiscard]] inline bool semchan_block_is_in_use(const SemChanInUseSpan& span,
                                                  std::uint32_t block_token_begin,
                                                  std::uint32_t block_token_end) noexcept {
    if (!span.present) { return false; }
    if (block_token_end <= block_token_begin) { return false; }
    return block_token_begin < span.covered_end && span.covered_begin < block_token_end;
}

// THE FACTS BUILDER THAT DOES NOT EXIST YET -- the small change this section names. It is
// additive: it starts from the tree's own `sum_dir_facts_from_sequence` and writes exactly ONE
// field, and only when the block really is inside the covered span.
[[nodiscard]] inline sum_dir::SumDirBlockFacts
semchan_facts_with_in_use(sum_dir::SumDirBlockFacts facts, bool in_use) noexcept {
    if (in_use) { facts.pinned_by_anchor = true; }
    return facts;
}

// THE CRITERION, run against THE TREE'S OWN JUDGE. `sum_dir_block_admissibility` and
// `sum_dir_is_unloadable` are called, never restated, so this cannot drift from what the pass does.
[[nodiscard]] inline SemChanInUseSpan semchan_iv_use_judge(
    const SemChanInUseSpan& span_in, const std::vector<std::uint32_t>& block_token_begins,
    const std::vector<std::uint32_t>& block_token_ends,
    const std::vector<sum_dir::SumDirBlockFacts>& base_facts) {
    SemChanInUseSpan out = span_in;
    const std::size_t n = std::min(block_token_begins.size(), block_token_ends.size());
    for (std::size_t i = 0; i < n && i < base_facts.size(); ++i) {
        ++out.blocks_evaluated;
        const bool in_use =
            semchan_block_is_in_use(span_in, block_token_begins[i], block_token_ends[i]);
        if (in_use) {
            ++out.blocks_in_use;
        } else {
            ++out.blocks_not_in_use;
        }
        if (sum_dir::sum_dir_is_unloadable(sum_dir::sum_dir_block_admissibility(base_facts[i]))) {
            ++out.unloadable_before;
        }
        const sum_dir::SumDirBlockFacts with_use =
            semchan_facts_with_in_use(base_facts[i], in_use);
        if (sum_dir::sum_dir_is_unloadable(sum_dir::sum_dir_block_admissibility(with_use))) {
            ++out.unloadable_after;
        }
    }
    return out;
}

[[nodiscard]] inline std::string semchan_iv_use_line(const SemChanInUseSpan& s) {
    std::string out = "[semchan-iv-use] present=";
    out += (s.present ? "1" : "0");
    out += " covered=[" + std::to_string(s.covered_begin) + "," + std::to_string(s.covered_end) + ")";
    out += " window=" + std::to_string(s.window);
    out += " block_tokens=" + std::to_string(s.block_tokens);
    out += " blocks_evaluated=" + std::to_string(s.blocks_evaluated);
    out += " blocks_in_use=" + std::to_string(s.blocks_in_use);
    out += " blocks_not_in_use=" + std::to_string(s.blocks_not_in_use);
    out += " unloadable_before=" + std::to_string(s.unloadable_before);
    out += " unloadable_after=" + std::to_string(s.unloadable_after);
    out += " protected_pages=" + std::to_string(s.unloadable_before - s.unloadable_after);
    out += " judge=sum_dir_block_admissibility+sum_dir_is_unloadable (the tree's own)";
    out += " criterion=positional-overlap";
    out += " signal=SumDirAdmissibility::PinnedByAnchor";
    out += " seam=sum_dir::sum_dir_facts_from_sequence(std::uint32_t token_count) has no span "
           "parameter; program_impl.h:13276 is its engine call site";
    return out;
}

// -------------------------------------------------------------------------------------------
// 5. THE COMPILE-TIME HALF -- the doctrine, pinned where a relaxation is a build error
// -------------------------------------------------------------------------------------------
// (1) THE UNION STILL MAY NOT DETERMINE. Re-asserted so a future edit that routes a span extension
//     into a decision is a build error in every TU that includes this header.
static_assert(!wire::kSemChanWireMayDetermine,
              "stage 2 does not relax stage 1's refusal: the channel may enter the candidate-set "
              "UNION and it may never determine");
// (2) THE EXTENSION CANNOT WRITE THE ARBITER'S RESULT. The function takes it as a CONST reference,
//     so `first_page`/`token_begin` cannot be moved in place; the returned `RecallRequest` is a
//     value and the harness measures `begin_is_arbiters` on it rather than taking a comment's word.
using SemChanSpanExtensionFn = SemChanSpanExtension (*)(const sum_dir::SumDirReachResult&,
                                                        const std::vector<std::uint32_t>&,
                                                        std::uint32_t, std::uint32_t, std::uint32_t,
                                                        bool);
static_assert(std::is_same<decltype(&semchan_span_extension_from_union),
                          SemChanSpanExtensionFn>::value,
              "the second stage's signature IS its contract: the arbiter's result comes in as "
              "`const SumDirReachResult&`, the nomination as a page list, and there is NO parameter "
              "a score could arrive through -- the property sum_dir_determined.h states for itself");
// (3) THE CRITERION NAMES NO PAGE. A criterion that could name a page would be a decision wearing a
//     cost argument, which is the shape `semchan_wire.h`'s text gate already refuses.
static_assert(std::is_same<decltype(&semchan_tier_criterion),
                           SemChanTierCriterion (*)(std::uint32_t, std::uint32_t, std::uint32_t,
                                                    std::uint32_t, std::uint64_t, bool,
                                                    const char*, const char*)>::value,
              "the criterion takes counts and bytes and returns counts and bytes: no page crosses "
              "its signature, so it decides about STORAGE and never about an answer");
// (4) THE DIRECTORY'S PREMISE. One 80 B row addresses one block whose KV is at least
//     64 x 18,432 B, so a resident row is a strictly cheaper way to keep a block nominable.
static_assert(sum_dir::kSumDirRowWireBytes * 100U <
                  sum_dir::kSumDirBlockTokens *
                      sum_dir::sum_dir_codec_bytes_per_token(sum_dir::SumDirCodec::Nvfp4),
              "the directory has no premise if a row is not far cheaper than the block it "
              "addresses -- and a stage-2 criterion priced on that difference has to stop being "
              "written if the difference stops existing");
// (5) ⭐⭐ THE VISIBILITY GAP, PINNED AS A TYPE RATHER THAN AS A FAILED GREP. The facts builder the
//         unload pass uses takes ONE `std::uint32_t` and returns a type whose NAME says what it
//         cannot carry (`SumDirBlockFactsFromSequence`, `sum_dir.h:1233-1241`); its body writes only
//         `content_present` and `overlap_frontier` (`:1245-1256`). So `pinned_by_anchor` is false on
//         every block and the pass's own line prints `dist.pinned-by-anchor=0`. If this assert ever
//         fails, the seam section 4 names has been closed and section 4 must be re-taken, not
//         re-worded.
using SumDirFactsFromSequenceFn =
    std::vector<sum_dir::SumDirBlockFactsFromSequence> (*)(std::uint32_t) noexcept;
static_assert(std::is_same<decltype(&sum_dir::sum_dir_facts_from_sequence),
                           SumDirFactsFromSequenceFn>::value,
              "the unload judge's facts builder takes a token count and NO SPAN, and returns the "
              "sequence-only half: the 指向 cannot reach the judge until a span arrives where the "
              "two halves are joined");
// (6) THE POSE IS NOT A SCORE EITHER. The in-use test is two comparisons over four integers.
using SemChanInUseTestFn = bool (*)(const SemChanInUseSpan&, std::uint32_t, std::uint32_t) noexcept;
static_assert(std::is_same<decltype(&semchan_block_is_in_use), SemChanInUseTestFn>::value,
              "the 指向 is POSITIONAL: four token indices in, one bool out, and no parameter a "
              "score could arrive through -- the same property sum_dir_determined.h states");

}  // namespace ninfer::spec::semchan::stage2

#endif  // NINFER_SPEC_SEMCHAN_STAGE2_H
