#pragma once

// F1160 -- THE BLOCK AXIS'S SECOND STAGE: a PER-BLOCK allocation rule under the
// fit's residual ceiling, evaluated as a LOOKUP once per block at formation.
//
// WHY THIS FILE EXISTS, AND WHY IT IS A SECOND STAGE AND NOT A FIT DIMENSION.
// product/kv_adapt_solver.h states its own decision variable, verbatim:
//     "Decision variable is per-layer: T(l) in F_l, F_l an ARBITRARY subset."
// and its DP is indexed by (layer, units): `dp[l+1][u + c(l,f)]`. The horizon of
// that DP is the LAYER COUNT, which is a compile-time-shaped fact about the
// artifact. A BLOCK is not: it is created at RUNTIME from arriving text, its
// number is unbounded and unknown when the fit runs, and the container rule is
// that THE ARTIFACT CARRIES THE RULE AND NOT THE BLOCKS. So the fit cannot have a
// block as a decision variable -- a DP whose outer index is "block" has no
// horizon to iterate to. What it CAN have is a RESIDUAL: the ceiling minus the
// bytes its layer plan spent. That residual is this stage's input, and the stage
// is where a block becomes the unit of allocation.
//
// THE BRIDGE IS THE ENGINE'S OWN IDENTITY, NOT AN INVENTED ONE.
// program_impl.h, verbatim: "A page's block index IS the page number under the
// engine's own identity: the directory's block is the 64-token Paged-KV page
// (sum_dir.h:136-138) and a concatenated unload text names its FIRST block
// (:1824-1833)." And product/kv_recall_block.h pins the same equality from the
// other side: kRecallBlockTokens is 64 and `page_tokens` must equal it, so a
// block IS a 64-token page ACROSS ALL TEXT LAYERS. The residency mechanism is
// page-granular for the same reason (core/shard_plan.h, quoting cold_host_tier.h:
// "GRANULARITY: one logical page ACROSS ALL TEXT LAYERS, never a slice of one.
// ... one page is one residency bit"), and the cold record is per
// (page, head, plane) at a stride that is the layer's own codec width. So the
// block-level rule reaches the page-level residency mechanism BY BEING THE SAME
// GRANULE: the rule is evaluated exactly where the page is offered for
// retirement, and its answer is the public admission verdict of that pass.
//
// THE RULE IS A LOOKUP, NEVER A SOLVER. decide() is O(1): it compares the
// block's byte charge against the bytes the ceiling has left. There is no search,
// no per-token work, and no re-solve per block -- the order's own constraint:
// "the allocation MUST REMAIN A LOOKUP evaluated per block at formation, NEVER a
// per-token solver. A per-token solver fails the floor regardless of how good the
// allocation is."
//
// THE CEILING IS THE FIT'S RESIDUAL, AND IT IS THE ONLY THING THE FIT OWES.
// kv_adapt_solver.h carries it already: KvAdaptSolution::budget_slack_bytes is
// "budget - achieved (>= 0)". A caller that has run the fit passes that number
// (in bytes, converted to cold-record bytes by the charge below) as the ceiling;
// a caller with no fit passes 0, which is the OFF switch and leaves the pass
// byte-for-byte what it was before this file existed.
//
// WHAT THIS FILE DOES NOT DO. It does not choose the per-layer plan (the fit
// does), it does not pick a codec per block (the cold record's codec is resolved
// per layer by that layer's dtype -- program_impl.h, "ONE CODEC PER LAYER
// DTYPE"), and it does not touch a kernel. Its whole surface is: a byte charge,
// an O(1) verdict, and an accounting line.

#include "product/kv_cell_modes.h"   // [F1172] the cell's mode set, its byte table, BlockChargeTable
#include "product/kv_tier_formats.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>   // [F1239 kvrate] the rate arm's DERIVED overflow gate
#include <string>

namespace ninfer::product {

// The stage's verdict for ONE block. Three values, and the third exists so a
// reader can tell "the semantic directory did not offer it" from "the ceiling
// refused it" -- the same distinction kv_tier_formats.h draws between FITS and
// PAYS, and for the same reason: the two failures have different fixes.
enum class KvBlockStageVerdict : std::uint8_t {
    NotOffered = 0,      // the caller's own gate declined the block (untouched)
    Admitted = 1,        // the block's charge fits the ceiling's remainder
    RefusedByCeiling = 2 // offered, but the remainder cannot pay for it
};

[[nodiscard]] inline const char* kv_block_stage_verdict_name(
    KvBlockStageVerdict verdict) noexcept {
    switch (verdict) {
    case KvBlockStageVerdict::Admitted: return "admitted";
    case KvBlockStageVerdict::RefusedByCeiling: return "refused-by-ceiling";
    case KvBlockStageVerdict::NotOffered: break;
    }
    return "not-offered";
}

// The per-block BYTE CHARGE. A block is one 64-token page across ALL text layers,
// and the cold pool reserves one record per (page, head, plane) whose stride is
// the layer's codec width (decoder_state.cpp cold_slot_stride_for). So the charge
// is a pure function of the stack shape:
//     kv_heads * 2 planes * (rans_layers * rans_stride + raw_layers * raw_stride)
// with both strides taken FROM THE TREE (kKvColdPoolStrideBytes 9632 for the
// rANS slot, kKvColdInt8PayloadBytes 9232 for the raw slot) rather than re-typed,
// so the stage and the pool cannot drift apart.
[[nodiscard]] constexpr std::int64_t kv_block_stage_bytes_per_block(
    std::uint32_t layers, std::int32_t kv_heads, std::uint32_t rans_layers) noexcept {
    if (layers == 0 || kv_heads <= 0 || rans_layers > layers) { return 0; }
    // [planes 2026-09-29] ⭐ THE ORDER'S RULER AND THE CHARGE'S RULER DISAGREED; THIS MAKES THEM
    // ONE -- AND THIS IS THE SECOND HALF OF THAT CUT (the first is `cell_vector_charge`, which
    // sums `plane_bytes` now). The scalar must move with the plan, or an UNPLANNED block would
    // charge 1,181,696 while a planned one charges 2,162,688 and the ceiling would be spent in two
    // currencies at once.
    // WHAT MOVED: the two prices, from the COLD SLOT strides to the RESIDENT PLANES --
    // `cell_rung(CellMode::Nvfp4).plane_bytes` (9,216) where the rANS stride 9,632 was, and
    // `cell_rung(CellMode::Int8).plane_bytes` (16,896) where the raw stride 9,232 was. Read
    // through `cell_rung` so the two cannot drift from the rung table.
    // WHAT DID NOT: the signature (`rans_layers` still selects the two rungs a COUNT can name) and
    // the pool's own strides -- `cold_slot_stride_for` still sizes every layer's cold slot at
    // 9,232 / 9,632. The stride is the SLOT; this is the CHARGE. They are different questions and
    // they were only conflated because one number served both.
    // ⚠ THE RETIRED VALUE IS KEPT AS A READING, NOT DELETED: see
    // `kLegacyChargePerBlockRawInt8_16x4 == 1,181,696` in `kv_cell_modes.h`, which is what the
    // ledger (`blob_F1160.md` section 8.1), `ceiling.txt` and every arm already run tonight are
    // priced in. `kv_block_stage_bytes_per_block(16, 4, 0)` is now 2,162,688 -- the number this
    // call site will print as `bytes_per_block`, and the unit a supplied budget is spent in.
    const std::uint32_t raw_layers = layers - rans_layers;
    const std::int64_t per_layer_pair =
        static_cast<std::int64_t>(rans_layers) * cell_rung(CellMode::Nvfp4).plane_bytes +
        static_cast<std::int64_t>(raw_layers) * cell_rung(CellMode::Int8).plane_bytes;
    return 2 * static_cast<std::int64_t>(kv_heads) * per_layer_pair;
}

// [F1172] THE SAME SUM, FROM THE PER-LAYER STRIDES THE POOL ACTUALLY BOOKED.
//
// WHERE THE REAL INPUT ALREADY IS, AND NOBODY SUMS IT. `PagedKVBatchLayerView::slot_bytes` is
// the stride the layer's cold slot was RESERVED with, resolved per layer by that layer's dtype
// (`cold_slot_stride_for`), and the pass already prints it once per layer, verbatim:
//     "[cold]   L%-3u dropped=%d dtype=%u %-46s cold_record=%-3s stride=%d B residual=%d"
// So `sum_l stride(V[l])` is computable TODAY from the engine's own object; what is missing is
// only that no one sums it and hands it to the charge. That is this function, and it is the
// "change the input, not the function" landing the design judged sufficient.
//
// WHY THE `rans_layers` COUNT IS **NOT** A SUFFICIENT STATISTIC, WHICH IS WHY THE COUNT ABOVE IS
// DEMOTED TO A READING. `kv_block_stage_bytes_per_block(layers, kv_heads, rans_layers)` computes
// the same sum ONLY BECAUSE THE RUNG TABLE HAS EXACTLY TWO DISTINCT PRICED RECORDS TODAY (9232
// and 9632): with two prices a multiset is determined by its count. The order's ladder needs at
// least THREE -- it adds the floor -- and with three prices a count is no longer a multiset.
// THE COUNT IS THEREFORE SUFFICIENT FOR TODAY'S TABLE AND INSUFFICIENT FOR THE ORDER'S.
[[nodiscard]] inline std::int64_t kv_block_stage_bytes_per_block_from_strides(
    const std::int32_t* const slot_bytes, std::uint32_t layers, std::int32_t kv_heads) noexcept {
    if (slot_bytes == nullptr || layers == 0 || kv_heads <= 0) { return 0; }
    std::int64_t per_layer_pair = 0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        // A layer with no record (a dropped layer, or one whose view is empty) has NO stride and
        // the sum is NOT computable. Returning 0 here is the caller's OFF signal, the same
        // convention `kv_block_stage_bytes_per_block` uses for an uncomputable stack.
        if (slot_bytes[layer] <= 0) { return 0; }
        per_layer_pair += slot_bytes[layer];
    }
    return 2 * static_cast<std::int64_t>(kv_heads) * per_layer_pair;
}

// [F1172] THE `rans_layers` READING, TAKEN FROM THE SAME ARRAY SO THE TWO SPELLINGS CANNOT
// DISAGREE. THIS IS A CODEC QUESTION ASKED OF A RECORD NUMBER, AND THE TREE'S OWN E3 IS WHY THAT
// IS ONLY ANSWERABLE WHILE THE TWO RECORDS ARE DISTINCT: `Int8Raw`, `Bf16Raw` and `Rk4v4Raw`
// share the 9232 record, so if the raw and rANS records ever became equal this count would
// silently become "all layers" or "no layers" and nothing else would notice. The static_assert
// is the only thing that keeps that from being a silent defect, so it is here and not in a
// comment.
static_assert(kKvColdPoolStrideBytes != kKvColdInt8PayloadBytes,
              "F1172: the rANS-vs-raw count is taken by comparing STRIDES, which is a RECORD "
              "question standing in for a CODEC question. It is only answerable while the two "
              "records differ. If this fires, count codecs via LayerColdCodec, not bytes.");
[[nodiscard]] inline std::uint32_t kv_block_stage_rans_layers_from_strides(
    const std::int32_t* const slot_bytes, std::uint32_t layers) noexcept {
    if (slot_bytes == nullptr) { return 0; }
    std::uint32_t count = 0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        if (slot_bytes[layer] == kKvColdPoolStrideBytes) { count += 1; }
    }
    return count;
}

// The stage's state. `ceiling_bytes <= 0` means OFF, and OFF is the pre-image
// behaviour of every caller that does not set it.
// The environment spelling, and the ONLY place the ceiling enters. Unset or
// unparseable-nonpositive means OFF (0), which is the negative control.
inline constexpr const char* kKvBlockStageBudgetEnv = "NINFER_KV_BLOCK_BUDGET_BYTES";
// [F1231-budget-ruler] THE SAME KNOB IN THE CHARGE'S OWN UNIT. The byte spelling above is a number
// whose RULER the engine cannot see -- an arm that authors its budget in the RETIRED cold-record
// stride (x1,181,696) passes a value that reads as a budget and behaves as a budget 1.8302x too
// small: MEASURED, `b016`'s 16 x 1,181,696 = 18,907,136 B against a floor that costs 28,311,552 B
// in the charge ruler, so every pass walked to the floor and still printed `fits=no`. This spelling
// removes the ambiguity: the number is a COUNT of blocks priced at the charge `decide()` spends, and
// the engine multiplies by the unit it already computes. IF BOTH ARE SET THE BYTE SPELLING WINS --
// a new spelling must never silently override an operator's explicit byte budget.
inline constexpr const char* kKvBlockStageBudgetBlocksEnv = "NINFER_KV_BLOCK_BUDGET_BLOCKS";
// [F1239 kvrate] ⭐ THE RATE SPELLING -- THE BUDGET AS A FREQUENCY OVER THE POPULATION, not a total.
//
// WHY IT EXISTS, AND IT IS A MEASUREMENT RATHER THAN A PREFERENCE. Against ONE fixed budget the
// engine answered `sat_yes=1 / sat_NO=23` on the new pair: the candidate population walked
// 736 -> 1248 -> 1760 -> ... -> 10,208 cells over 24 passes, so `min_possible_bytes` walked
// 27,131,904 -> 46,006,272 -> 64,880,640 while the budget did not move. The verdict recorded with
// that reading was "A fixed byte budget cannot be satisfiable against a population that grows 10x
// during one prefill. Any future budget arm must state WHICH pass it bound on." A rate states it by
// construction: it is resolved against THIS pass's own population, so every pass is the same pass.
//
// THE DENOMINATION, and it is the engine's own unit rather than a new one:
//
//     ceiling_bytes(W, R) = W * charge_per_block * R / 10000          [rounded, not truncated]
//
// `R` is in 10000ths of the ALL-INT8 plane, so `R / 10000 == T / 8.25` for a target `T` b/el:
// `R = 10000` prices every cell at int8 (the ceiling IS this pass's all-int8 charge); the ladder's
// own floor is `R = 2727` (2.25 b/el). ⚠ THE RATE IS NOT A CONSTANT AND THE CODE ADDS NONE: `R` is
// the operator's number, `charge_per_block` is computed from the running stack, and `W` is this
// pass's own block count. 少写死多适配, as the testable property: ADDING A MODEL CHANGES `W` AND
// `charge_per_block` AND TOUCHES NO LINE OF THIS ARM AND NO KV CONSTANT.
//
// WHY SATISFIABILITY STOPS DEPENDING ON THE PASS. The floor is `W * per_block_floor`, so
// `ceiling >= floor` reduces to `charge_per_block * R / 10000 >= per_block_floor` -- **W CANCELS**.
// Under a rate arm the answer is a property of the RATE and the STACK, never of the pass, which is
// exactly the property an absolute total cannot have.
//
// ⚠ AND AN UNRESOLVABLE RATE REFUSES BY NAME -- it never becomes a silent OFF. A rate with no
// population is a number with no denominator, so `population_blocks == 0` sets `refusal` and
// `describe()` prints it; `live()` is false, so the caller cannot mistake it for "no budget set".
//
// PRECEDENCE: `..._BYTES` > `..._BLOCKS` > `..._RATE_X10000`. An explicit total keeps winning, for
// the reason the block spelling already gives: a new spelling must never silently override an
// operator's explicit byte budget.
//
// ⭐ REFERENCES (searched for this landing; read depth is stated in the report and NOT overclaimed):
//   * RFC 2697 srTCM / RFC 2698 trTCM (Heinanen & Guerin, 1999) -- a traffic profile is a RATE plus
//     burst SIZES, never an absolute total, and the meter is agnostic to what the traffic is. That
//     is the property borrowed here: denominate against a rate, keep the unit the payer's own.
//     [FULL TEXT FETCHED]
//   * US20150100630A1 "Throttling service requests having non-uniform workloads" -- tokens are
//     deducted in units of the WORK a request performs, with the generation rate tied to a target
//     committed work throughput, rather than one token per request. Same move as charging a ceiling
//     in the population's own charge blocks. [SEARCH SUMMARY ONLY -- the fetch timed out]
//   * RDKV (arXiv 2605.08317) is explicitly NOT a reference for this layer: it allocates ONCE after
//     prefill and records no streaming re-budgeting and NO duty bound of any kind. [ABSTRACT PAGE]
//   * RFC 3290 -- fetched for a hysteresis/duty-cycle precedent and it has NONE; recorded so the
//     absence is a reading rather than an assumption. [FULL TEXT FETCHED, negative result]
inline constexpr const char* kKvBlockStageBudgetRateEnv = "NINFER_KV_BLOCK_BUDGET_RATE_X10000";
// The scale of the rate arm's unit (10000ths of the int8 plane). NAMED because the rounding rule
// below reads it, not because it is a knob.
inline constexpr std::int64_t kKvBlockStageBudgetRateScale = 10000;
// THE FEATURE MACRO THE TWO-SIDED RULER ARM READS, so ONE arm source compiles against the
// pre-fix headers (where the charge-block spelling is UNEXPRESSIBLE and the byte reader returns
// 0 = OFF = `steps=0`) and against these (where the same env binds and `steps>0`).
#define NINFER_KV_BUDGET_RULER_F1231 1

// ⚠ THE UNIT IS NOT "BLOCKS" -- IT IS `2 * kv_heads * layers * plane_bytes(int8)`, i.e.
// 262,144 x 8.25 = 2,162,688 on the shipped 4-head/16-layer stack, so THE COUNT IS WINDOW-RELATIVE:
// a target of T bits per element over a W-block window is `N = W * T / 8.25`, and the achievable T
// quantises at `8.25 / W` per block (0.171875 b/el at W = 48). ANY block-count guidance must state
// the window width it assumes, because a reader on another window gets a different N. (Correction
// supplied by the `kvpost` line's own arithmetic, 2026-09-29.)
//
// WHICH RULER THE BUDGET ARRIVED IN, AS A VALUE, so the log can say it and no reading has to guess.
enum class KvBudgetRuler : std::uint8_t {
    Unset        = 0,  // neither spelling set (or unparseable/<= 0): OFF, the pre-image behaviour
    ChargeBytes  = 1,  // NINFER_KV_BLOCK_BUDGET_BYTES
    ChargeBlocks = 2,  // NINFER_KV_BLOCK_BUDGET_BLOCKS x the charge unit
    // [F1239 kvrate] ⭐ THE THIRD ARM, AND IT IS THE ONLY ONE THAT IS NOT A TOTAL. The two above
    // are ABSOLUTE byte totals: they are statements about ONE population size, so the same arm can
    // be satisfiable on one pass and infeasible on the next (MEASURED: `sat_yes=1 / sat_NO=23`, and
    // the population grew 736 -> 10,208 cells inside ONE prefill). This arm is a RATE over the
    // population, so it is a statement about every pass at once.
    RatePerElement = 3,  // NINFER_KV_BLOCK_BUDGET_RATE_X10000 x (this pass's own population)
};

[[nodiscard]] constexpr const char* kv_budget_ruler_name(KvBudgetRuler ruler) noexcept {
    switch (ruler) {
    case KvBudgetRuler::ChargeBytes: return "charge-bytes(env:..._BYTES)";
    case KvBudgetRuler::ChargeBlocks: return "charge-blocks(env:..._BLOCKS)";
    case KvBudgetRuler::RatePerElement: return "rate-per-element(env:..._RATE_X10000)";
    case KvBudgetRuler::Unset: break;
    }
    return "unset(OFF)";
}

// THE ONE RESOLUTION. `charge_per_block` is the CHARGE ruler's unit -- the same number
// `KvBlockStageState::bytes_per_block` holds and `decide()` spends -- so the caller must pass it,
// and both readers (the descent's stop total and the stage's residual ceiling) then receive the
// SAME bytes from the SAME call. `blocks_x100` is a reading: the budget expressed in charge-priced
// blocks, x100, so a log line can say what the number actually is.
struct KvBlockStageBudget {
    std::int64_t ceiling_bytes = 0;
    std::int64_t blocks_x100 = 0;
    KvBudgetRuler ruler = KvBudgetRuler::Unset;

    [[nodiscard]] bool live() const noexcept {
        // [F1239 kvrate] A REFUSED ARM IS NOT A LIVE ONE. If the rate could not be resolved the
        // ceiling stays 0, and `live()` says so -- so a caller cannot read an unevaluable budget as
        // "no budget", which is the silently-rest-on-the-floor failure this arm must not have.
        return ceiling_bytes > 0 && refusal == nullptr;
    }
    [[nodiscard]] std::string describe() const {
        std::string out = std::string("ruler=") + kv_budget_ruler_name(ruler) +
               " ceiling_bytes=" + std::to_string(ceiling_bytes) +
               " blocks_x100=" + std::to_string(blocks_x100) +
               " charge_per_block=" + std::to_string(charge_per_block_seen);
        if (ruler == KvBudgetRuler::RatePerElement) {
            // ⭐ A RATE THAT DOES NOT NAME THE POPULATION IT WAS READ OVER IS NOT A READING -- that
            // is the whole complaint against the absolute spelling. So the rate and its denominator
            // are printed TOGETHER, and the b/el equivalence is printed beside them so a reader
            // never has to multiply by 8.25 in their head.
            out += " rate_x10000=" + std::to_string(rate_x10000);
            out += " population_blocks=" + std::to_string(population_blocks);
            out += " rate_bits_per_element_x100=" + std::to_string(rate_bits_per_element_x100);
            out += " this_populations_floor_bits_per_element_x100=" +
                   std::to_string(floor_bits_per_element_x100);
            out += " rate_is_pass_invariant=";
            out += (refusal == nullptr) ? "yes(W cancels out of ceiling>=floor)"
                                        : "no(this arm was not resolved)";
        }
        if (refusal != nullptr) { out += std::string(" refusal=") + refusal; }
        return out;
    }
    std::int64_t charge_per_block_seen = 0;

    // =============================================================== [F1239 kvrate] THE RATE ARM
    // `rate_x10000` IS WHAT WAS ASKED (kept verbatim). `population_blocks` (W) IS THE PASS IT WAS
    // RESOLVED AGAINST -- so the log itself answers "which pass did this bind on".
    // `rate_bits_per_element_x100` and `floor_bits_per_element_x100` are the SAME statement in the
    // operator's currency (b/el), the second read off the ladder and NOT re-typed, so a refusal can
    // name the gap ("the rate is at 1.27 b/el and this population's floor is at 2.25 b/el").
    // `refusal` NON-NULL MEANS THE ARM WAS SET AND COULD NOT BE EVALUATED: the reason is a string
    // literal with static storage, so it stays valid while the returned-by-value struct is copied.
    std::int64_t rate_x10000 = 0;
    std::int64_t population_blocks = 0;
    std::int64_t rate_bits_per_element_x100 = 0;
    std::int64_t floor_bits_per_element_x100 = 0;
    const char* refusal = nullptr;
};

// ⚠ `population_blocks` IS DEFAULTED SO NO EXISTING CALLER CHANGES MEANING: an absolute arm is
// readable without a population and behaves IDENTICALLY whether W is passed or not. Only the rate
// arm needs it, and a rate arm with W == 0 REFUSES BY NAME rather than guessing.
[[nodiscard]] inline KvBlockStageBudget kv_block_stage_budget_from_env(
    std::int64_t charge_per_block, std::int64_t population_blocks = 0) noexcept {
    KvBlockStageBudget out{};
    out.charge_per_block_seen = charge_per_block;
    // The floor is a READING OF THE LADDER, not a constant: `kv_cell_cheapest_bits_x100()` walks
    // `kCellRungs` (kv_cell_modes.h). If a future rung becomes the floor this moves by itself.
    out.floor_bits_per_element_x100 = kv_cell_cheapest_bits_x100();
    const char* const bytes_text = std::getenv(kKvBlockStageBudgetEnv);
    if (bytes_text != nullptr && *bytes_text != '\0') {
        char* end = nullptr;
        const long long parsed = std::strtoll(bytes_text, &end, 10);
        if (end != bytes_text && parsed > 0) {
            out.ceiling_bytes = static_cast<std::int64_t>(parsed);
            out.ruler = KvBudgetRuler::ChargeBytes;
            if (charge_per_block > 0) {
                out.blocks_x100 = out.ceiling_bytes * 100 / charge_per_block;
            }
            return out;
        }
    }
    const char* const blocks_text = std::getenv(kKvBlockStageBudgetBlocksEnv);
    if (blocks_text != nullptr && *blocks_text != '\0') {
        char* end = nullptr;
        const long long blocks = std::strtoll(blocks_text, &end, 10);
        if (end != blocks_text && blocks > 0 && charge_per_block > 0) {
            out.ceiling_bytes = static_cast<std::int64_t>(blocks) * charge_per_block;
            out.blocks_x100 = static_cast<std::int64_t>(blocks) * 100;
            out.ruler = KvBudgetRuler::ChargeBlocks;
            return out;
        }
    }
    // [F1239 kvrate] ⭐ THE RATE ARM, AND IT IS LAST ON PURPOSE: an explicit total wins.
    const char* const rate_text = std::getenv(kKvBlockStageBudgetRateEnv);
    if (rate_text != nullptr && *rate_text != '\0') {
        char* end = nullptr;
        const long long rate = std::strtoll(rate_text, &end, 10);
        out.ruler = KvBudgetRuler::RatePerElement;
        out.population_blocks = population_blocks;
        if (end == rate_text || rate <= 0) {
            out.refusal =
                "rate-arm-set-but-not-a-positive-count(..._RATE_X10000 must be an integer > 0; "
                "the arm is REFUSED rather than read as 0, because 0 would be indistinguishable "
                "from OFF)";
            return out;
        }
        out.rate_x10000 = static_cast<std::int64_t>(rate);
        // The operator's rate, in b/el, USING THE LADDER'S OWN int8 PRICE AS THE RULER -- so the
        // equivalence is derived and 8.25 is not typed anywhere in this file.
        const std::int32_t int8_plane_bytes = cell_rung(CellMode::Int8).plane_bytes;
        const std::int32_t int8_bits_x100 = kv_cell_bits_x100_of_plane_bytes(int8_plane_bytes);
        if (int8_bits_x100 > 0) {
            out.rate_bits_per_element_x100 = out.rate_x10000 * int8_bits_x100 / 10000;
        }
        if (charge_per_block <= 0) {
            out.refusal =
                "rate-arm-set-but-this-stack-has-no-computable-charge(charge_per_block<=0: the "
                "rate has no denominator, so it is REFUSED, not read as OFF)";
            return out;
        }
        if (population_blocks <= 0) {
            out.refusal =
                "rate-arm-set-but-THE-POPULATION-WAS-NOT-SUPPLIED(blocks<=0: a rate over no "
                "population is a number with no denominator -- the caller must pass this pass's "
                "own block count W, and the ABSOLUTE spellings are the only ones that can be read "
                "without one)";
            return out;
        }
        // ⚠ THE OVERFLOW GATE IS DERIVED, NOT A MAGIC BOUND: the largest rate this population and
        // this charge can carry without leaving int64. `W * charge * R` is computed in int64, so a
        // rate big enough to wrap would otherwise produce a SMALL ceiling -- the worst possible
        // failure for a budget, because it would look like a binding budget.
        const std::int64_t rate_ceiling_max =
            (std::numeric_limits<std::int64_t>::max() / kKvBlockStageBudgetRateScale) /
            population_blocks / charge_per_block;
        if (out.rate_x10000 > rate_ceiling_max) {
            out.refusal =
                "rate-arm-set-but-WOULD-OVERFLOW(W * charge_per_block * R leaves int64 on this "
                "population; REFUSED rather than wrapped, because a wrapped ceiling reads as a "
                "BINDING budget)";
            return out;
        }
        // ROUNDED, NOT TRUNCATED, and the reason is a measured property of the exact case: at
        // R == 10000 the ceiling must be EXACTLY `W * charge_per_block` -- this pass's own all-int8
        // charge -- and a truncating divide can land 1 B below it once W and the unit stop being
        // round, which would make the engine call an exactly-fitting budget infeasible.
        const std::int64_t numerator =
            population_blocks * charge_per_block * out.rate_x10000;
        out.ceiling_bytes =
            (numerator + kKvBlockStageBudgetRateScale / 2) / kKvBlockStageBudgetRateScale;
        out.blocks_x100 = out.ceiling_bytes * 100 / charge_per_block;
        return out;
    }
    return out;   // Unset: OFF, byte for byte the pre-image path
}

// =================================================================================================
// [F1245 kvcarry] A SET SPELLING THAT YIELDS A DEAD CEILING MUST SAY SO -- AND SAY WHY.
// =================================================================================================
//
// THE DEFECT THIS CLOSES, MEASURED TWICE. F1244's whole-battery worst arm was
// `NINFER_KV_BLOCK_BUDGET_RATE_X10000=4500` (`dl/cmpfire/out/speed/B_rate/`): rc=0, wall=424.6 s
// against the 24.1 s of the same arm with the charge-block spelling, and ZERO lines of any kind --
// `grep -c '[kv-block-descent]' stderr.txt` = 0 and `grep -c '[kv-budget]'` = 0. The knob WAS read
// (`kv_block_stage_budget_from_env`, below), it REFUSED BY NAME
// (`rate-arm-set-but-THE-POPULATION-WAS-NOT-SUPPLIED`), the refusal left `ceiling_bytes` at 0, the
// stage's own gate (`ceiling <= 0 -> return false`) declared it not live, and the ONLY print sites
// are guarded by `block_stage_live` -- so **the reason the arm did nothing was itself unprintable**.
// The engine's own house rule is the opposite one and it is written three files over:
// *"a pass that decided not to unload has to say so, because 'the plan was empty and nothing was
// printed' is exactly the silence that hid this whole area"* (program_impl.h, the unload report's
// contract). A knob that was set and produced nothing is that silence.
//
// SO THE TWO READERS BELOW ARE THE INSTRUMENT, IN ONE PLACE, SO THE CALLER CANNOT FORGET HALF:
//   * `kv_block_stage_budget_env_present()` -- was ANY spelling present at all? This is the OFF
//     guard: when NOTHING is set the caller prints NOTHING and the pre-image path is untouched,
//     byte for byte. It reads the raw environment, so a spelling that is set to garbage (which the
//     resolver above cannot distinguish from unset in the `ruler` field) still counts as SET.
//   * `kv_block_stage_dead_budget_reason(...)` -- the NAMED reason the ceiling is 0 on a stack
//     where the operator asked for a ceiling. `budget.refusal` (the resolver's own string) is used
//     verbatim when it is set, because the resolver knows WHY it refused; the remaining cases are
//     the ones the resolver cannot see (a ceiling that parsed to a non-positive number, a stack
//     with no computable charge, a stack with no layers). It NEVER returns nullptr, so a caller
//     cannot print an empty reason.
[[nodiscard]] inline bool kv_block_stage_budget_env_present() noexcept {
    const char* const names[3] = {kKvBlockStageBudgetEnv, kKvBlockStageBudgetBlocksEnv,
                                  kKvBlockStageBudgetRateEnv};
    for (const char* const name : names) {
        const char* const text = std::getenv(name);
        if (text != nullptr && *text != '\0') { return true; }
    }
    return false;
}

[[nodiscard]] inline const char* kv_block_stage_dead_budget_reason(
    const KvBlockStageBudget& budget, std::int64_t charge_per_block, std::uint32_t layers,
    std::int32_t kv_heads) noexcept {
    // (1) THE RESOLVER'S OWN REFUSAL, VERBATIM. It is a string literal with static storage, so it
    // outlives this call (the same property the resolver's own comment relies on).
    if (budget.refusal != nullptr) { return budget.refusal; }
    // (2) IT RESOLVED AND STILL PRODUCED NO CEILING: the only way is a spelling that was present and
    // unparseable/non-positive, which the resolver deliberately does not distinguish from OFF (both
    // leave `ruler == Unset`), so the reason is named here where the presence IS known.
    if (budget.ruler == KvBudgetRuler::Unset) {
        return "spelling-set-but-NOT-READ-AS-A-POSITIVE-COUNT(the environment names a budget, and "
               "it did not parse to a positive integer -- so the ceiling is 0 and the block stage "
               "is OFF. 0 remains the legitimate OFF value when NOTHING is set; a SET spelling that "
               "reads as 0 is this refusal, not a decision to run unbudgeted)";
    }
    if (budget.ceiling_bytes <= 0) {
        return "ceiling-is-not-positive(the spelling resolved and produced ceiling_bytes<=0: the "
               "block stage is OFF for this pass)";
    }
    // (3) THE CEILING IS FINE AND THE STAGE STILL DID NOT COME UP: the charge or the stack shape.
    if (charge_per_block <= 0) {
        return "charge-not-computable(ceiling>0 but charge_per_block<=0: this stack's per-block "
               "charge has no value, so a ceiling denominated in it cannot be spent)";
    }
    if (layers == 0 || kv_heads <= 0) {
        return "stack-shape-is-empty(ceiling>0 and charge>0 but the stack declares no layers or no "
               "KV heads: there is no block axis to charge)";
    }
    return "stage-not-live(ceiling>0, charge>0, stack shaped, and the stage still refused -- this "
           "reason is not enumerated in kv_block_budget_stage.h and the code must be read)";
}

// AND A NOTE WHERE THE STATE IS DECLARED, so a reader of the struct knows where the ruler is.
struct KvBlockStageState {
    std::int64_t ceiling_bytes      = 0;
    std::int64_t bytes_per_block    = 0;
    // ⚠⚠ [F1209 2026-09-29] THE RETIRED SPELLING, KEPT AS A READING RATHER THAN DELETED, AND THE
    // FIELD THAT PROVES THE UNIFICATION. `bytes_per_block` above is the CHARGE -- the same ruler the
    // plan's per-block bytes are in -- and this is the sum of the strides the POOL ACTUALLY BOOKED
    // (9,232 / 9,632 per layer per plane-pair, `kv_block_stage_bytes_per_block_from_strides`). The
    // two are DIFFERENT QUANTITIES and were only conflated because one number served both: the
    // stride is the SLOT the pool reserves, the charge is what a block costs against the ceiling.
    // Until F1209 the stride sum was ASSIGNED INTO `bytes_per_block` at the end of
    // `kv_block_stage_from_strides`, which is the line that put the printed scalar (1,181,696) and
    // the spend (`spent_bytes` in units of 2,162,688) in two rulers at once -- the same disease as
    // the order/delta split. The value is not lost: it is here, and printed beside the charge.
    std::int64_t booked_stride_bytes_per_block = 0;
    // -----------------------------------------------------------------------------------------
    // [F1172] THE PER-BLOCK CHARGE INPUT, AND IT IS THE WHOLE OF THE WIDENING.
    //
    // `bytes_per_block` above is ONE scalar for every block because the engine's only production
    // call site passes the LITERAL `0` for `rans_layers` (impl/runtime/program_impl.h, byte
    // ~765,805): the charge is `2 * kv_heads * layers * 9232`, blind to both the layer axis and
    // the block axis. The order's unit is the CELL `(block, layer)`, so the charge has to be
    // readable per block.
    //
    // WHY A TABLE AND NOT A `BlockVectorSource`: a table is THREE FIELDS and no virtual call, so
    // this header keeps its deliberate plainness (its own comment: the header is "plain-g++").
    // The vectors themselves live one layer up, in `kv_block_descent.h`, because they are the
    // decision's object while this is only its PRICE.
    //
    // OFF IS BYTE-FOR-BYTE, AND IT IS BY CONSTRUCTION RATHER THAN BY A SWITCH: with
    // `block_charge.bytes == nullptr`, `bytes_per_block_for(block)` returns `bytes_per_block`,
    // so `decide()`'s arithmetic is the SAME EXPRESSION it always was. The default vector of the
    // descent is all-`int8`, which IS the literal `rans_layers = 0` charge, so a live plan over
    // an undriven walk also reproduces it (pinned in `kv_block_pin_tu.cpp`).
    product::BlockChargeTable block_charge{};
    // [F1172] A READING, NOT AN INPUT: the real count of layers whose booked stride is the rANS
    // record. `kv_block_stage_bytes_per_block` still takes it (its signature is untouched), and
    // this field only lets `describe()` print the truth instead of the literal 0.
    std::uint32_t rans_layers = 0;
    bool charge_from_plan = false;
    std::int64_t spent_bytes        = 0;
    std::int32_t layers             = 0;
    std::int32_t kv_heads           = 0;
    std::uint32_t offered           = 0;
    std::uint32_t admitted          = 0;
    std::uint32_t refused_by_ceiling = 0;
    std::uint32_t blocks_seen       = 0;
    // [F1231] F-4's OWN COUNT, AND IT COUNTS THE OTHER DIRECTION TOO. `blocks_seen` is how many
    // blocks the pass consulted; `block_charge.count` is how many the plan covers. The plan
    // OVER-covering (`plan.blocks - blocks_seen`, F1230's measurement) and the pass consulting a
    // block the plan does NOT cover are different defects with different fixes, and one number
    // cannot carry both -- this is the second one. A consultation outside `[origin, count)` is
    // charged the SCALAR (`bytes_per_block_for`'s fallback), i.e. a different ruler, so a non-zero
    // reading here is "the ceiling was spent in two currencies on this pass".
    std::uint32_t blocks_outside_plan = 0;
    // [F1231-budget-ruler] THE BUDGET'S OWN RULER, CARRIED ON THE STATE so `describe()` can print it
    // and a log line can never be silent about which spelling produced the ceiling.
    KvBudgetRuler budget_ruler = KvBudgetRuler::Unset;
    std::int64_t budget_blocks_x100 = 0;
    // [F1239 kvrate] THE RATE THE CEILING CAME FROM, AND THE PASS IT WAS RESOLVED AGAINST. Carried
    // on the STATE for the same reason `budget_ruler` is: a reader of the accounting line must be
    // able to see that this pass's ceiling was a RATE over THIS pass's population, not a total that
    // happened to fit. 0 for both when an absolute spelling (or nothing) is set.
    std::int64_t budget_rate_x10000 = 0;
    std::int64_t budget_population_blocks = 0;
    // [F1239 kvrate] THE REFUSAL, CARRIED. Non-null when the budget arm was set and could NOT be
    // evaluated; `live()` is false in that state, and the caller must report it rather than treat
    // the pass as unbudgeted. See the never-silently-collapse rule above.
    const char* budget_refusal = nullptr;

    [[nodiscard]] bool live() const noexcept {
        return ceiling_bytes > 0 && bytes_per_block > 0;
    }

    [[nodiscard]] std::int64_t remaining_bytes() const noexcept {
        return ceiling_bytes - spent_bytes;
    }

    // THE LOOKUP. O(1), one call per block, at the moment the block is formed.
    // `offered` is the caller's own verdict (the semantic directory's, in this
    // engine) and this function never overrides an offered block upward: the
    // ceiling can only REFUSE. That is what keeps the floor rule honest -- the
    // budget can never admit something the directory kept.
    // [F1172] THE CHARGE OF ONE BLOCK. Two readings, one expression: this function is what
    // `decide()` spends and what the old scalar was. `(void)block;` used to sit in `decide()`
    // because the block's identity "is carried for the record" and "does not participate in any
    // arithmetic"; it participates now, and only when a plan covers it.
    [[nodiscard]] std::int64_t bytes_per_block_for(std::uint32_t block) const noexcept {
        if (block_charge.covers(block)) {
            return block_charge.bytes[block - block_charge.origin];
        }
        return bytes_per_block;
    }

    [[nodiscard]] KvBlockStageVerdict decide(std::uint32_t block, bool offered) noexcept {
        ++blocks_seen;
        // [F1231] COUNTED BEFORE THE OFFER TEST ON PURPOSE: this is a POPULATION question, not a
        // spend question. A page the pass consulted and the caller's gate then declined is still a
        // page the plan should have covered, so it belongs in the count.
        if (!block_charge.covers(block)) { ++blocks_outside_plan; }
        if (!offered) { return KvBlockStageVerdict::NotOffered; }
        ++this->offered;
        const std::int64_t charge = bytes_per_block_for(block);
        if (remaining_bytes() < charge) {
            ++refused_by_ceiling;
            return KvBlockStageVerdict::RefusedByCeiling;
        }
        spent_bytes += charge;
        ++admitted;
        return KvBlockStageVerdict::Admitted;
    }

    // [F1231] THE PLAN'S OWN TOTAL, IN THE STAGE'S RULER -- RECOMPUTED BY THE CONSUMER. This is
    // the plan's `total_bytes` computed from the charge table alone, i.e. the number the ceiling
    // was actually spent against, obtained WITHOUT asking the producer. A producer and its
    // consumer agreeing on one sum is the property F-4 is about; making the consumer do the sum
    // is what turns the agreement into a reading rather than a hope. O(count) once per pass.
    [[nodiscard]] std::int64_t planned_charge_bytes() const noexcept {
        if (block_charge.bytes == nullptr || block_charge.count == 0) { return 0; }
        std::int64_t sum = 0;
        for (std::uint32_t i = 0; i < block_charge.count; ++i) { sum += block_charge.bytes[i]; }
        return sum;
    }

    // The accounting line, ONE per pass, printed by the caller so this header
    // stays free of I/O policy.
    [[nodiscard]] std::string describe() const {
        std::string out = "ceiling_bytes=" + std::to_string(ceiling_bytes) +
                          " bytes_per_block=" + std::to_string(bytes_per_block) +
                          // [F1209 2026-09-29] THE TWO RULERS, BOTH PRINTED, NEITHER AMBIGUOUS.
                          // `bytes_per_block` is the CHARGE (the unit `spent_bytes`, `left_bytes`
                          // and every plan charge are in); this one is what the pool BOOKED and it
                          // is a reading, like `rans_layers` below. A reader dividing a budget by
                          // the wrong one is exactly the 16,896/9,232 = 1.83x error this pair now
                          // makes impossible to make silently. On the shipped all-raw stack they
                          // read 2,162,688 and 1,181,696.
                          " booked_stride_bytes_per_block=" +
                          std::to_string(booked_stride_bytes_per_block) +
                          " blocks_seen=" + std::to_string(blocks_seen) +
                          " offered=" + std::to_string(offered) +
                          " admitted=" + std::to_string(admitted) +
                          " refused_by_ceiling=" + std::to_string(refused_by_ceiling) +
                          " spent_bytes=" + std::to_string(spent_bytes) +
                          " left_bytes=" + std::to_string(remaining_bytes()) +
                          " layers=" + std::to_string(layers) +
                          " kv_heads=" + std::to_string(kv_heads) +
                          " rans_layers=" + std::to_string(rans_layers) +
                          " charge_from_plan=" + std::string(charge_from_plan ? "yes" : "no") +
                          " planned_blocks=" + std::to_string(block_charge.count) +
                          " plan_origin=" + std::to_string(block_charge.origin) +
                          // ---------------------------------------------------------------------
                          // [F1231] F-4, AS A PRINTED COLUMN. `P3_B_64k_65536_r1.log` printed
                          // `planned_blocks=50` on this line and `blocks=50` on the descent's, while
                          // this line's own `blocks_seen` said 48 -- a disagreement of two blocks
                          // (4,325,376 B at the shipped charge) that NOTHING reconciled and that a
                          // reader had to subtract by hand from two different lines. The three
                          // columns below are that subtraction, done once, on the line that owns the
                          // spend:
                          //   `population_delta`        = blocks_seen - planned_blocks, SIGNED, so
                          //                               "the plan over-covers" and "the pass
                          //                               consulted something unplanned" are
                          //                               distinguishable without arithmetic;
                          //   `planned_charge_bytes`    = the plan's total, recomputed HERE from
                          //                               the charge table (the consumer's own sum);
                          //   `unspent_planned_bytes`   = what the plan allocated and the pass never
                          //                               consulted, i.e. F-4's own quantity.
                          // A zero delta is the plan and the stage agreeing on the population; a
                          // non-zero one is now a reading instead of a defect nobody could see.
                          " population_delta=" +
                          std::to_string(static_cast<long long>(blocks_seen) -
                                         static_cast<long long>(block_charge.count)) +
                          " planned_charge_bytes=" + std::to_string(planned_charge_bytes()) +
                          " unspent_planned_bytes=" +
                          std::to_string(planned_charge_bytes() - spent_bytes) +
                          " blocks_outside_plan=" + std::to_string(blocks_outside_plan) +
                          // [F1231-budget-ruler] WHICH RULER THE CEILING CAME FROM, in charge
                          // blocks. A reader dividing a budget by the wrong unit is the F-3 defect;
                          // this column makes that impossible to do silently from a log.
                          " budget_ruler=" + std::string(kv_budget_ruler_name(budget_ruler)) +
                          " budget_charge_blocks_x100=" + std::to_string(budget_blocks_x100) +
                          // [F1239 kvrate] THE RATE COLUMNS, so the stage's own line says the one
                          // thing an absolute total cannot: WHICH POPULATION the ceiling was a rate
                          // over. Both are 0 for the absolute spellings, which is the reading that
                          // says "this ceiling was a total". `budget_refusal` is printed whenever it
                          // is set, so a set-but-unevaluable arm is never silent.
                          " budget_rate_x10000=" + std::to_string(budget_rate_x10000) +
                          " budget_rate_population_blocks=" +
                          std::to_string(budget_population_blocks) +
                          " budget_refusal=" +
                          std::string(budget_refusal == nullptr ? "none"
                                                               : budget_refusal);
        return out;
    }
};

// The caller's one-liner: run the lookup iff the stage is live, otherwise return
// the caller's verdict untouched (so an unset ceiling is byte-for-byte the
// pre-image path).
[[nodiscard]] inline bool kv_block_stage_admits(KvBlockStageState* state, std::uint32_t block,
                                               bool offered) noexcept {
    if (state == nullptr || !state->live()) { return offered; }
    return state->decide(block, offered) == KvBlockStageVerdict::Admitted;
}


[[nodiscard]] inline std::int64_t kv_block_stage_ceiling_from_env() noexcept {
    const char* const text = std::getenv(kKvBlockStageBudgetEnv);
    if (text == nullptr || *text == '\0') { return 0; }
    char* end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (end == text || parsed <= 0) { return 0; }
    return static_cast<std::int64_t>(parsed);
}

// Build the state from the environment + the stack shape. Returns true when the
// stage is LIVE, i.e. when the ceiling is set and the charge is computable. A
// ceiling set on an uncomputable stack is refused as OFF and must be reported by
// the caller, not swallowed -- the same "never silently collapse" rule every gate
// in product/kv_tier_formats.h follows.
[[nodiscard]] inline bool kv_block_stage_from_env(KvBlockStageState& state, std::uint32_t layers,
                                                  std::int32_t kv_heads,
                                                  std::uint32_t rans_layers,
                                                  std::int64_t population_blocks = 0) noexcept {
    state = KvBlockStageState{};
    // [F1231-budget-ruler] THE UNIT FIRST, THEN THE BUDGET -- because the block spelling needs the
    // unit to become bytes, and reading the env before the unit is computed is what forced the two
    // readers to be two reads. The BYTE spelling's behaviour is byte-for-byte what it was: a set and
    // positive `..._BYTES` is adopted verbatim, exactly as `kv_block_stage_ceiling_from_env` did.
    const std::int64_t per_block = kv_block_stage_bytes_per_block(layers, kv_heads, rans_layers);
    const KvBlockStageBudget budget =
        kv_block_stage_budget_from_env(per_block, population_blocks);
    state.budget_ruler = budget.ruler;
    state.budget_blocks_x100 = budget.blocks_x100;
    state.budget_rate_x10000 = budget.rate_x10000;
    state.budget_population_blocks = budget.population_blocks;
    state.budget_refusal = budget.refusal;
    const std::int64_t ceiling = budget.ceiling_bytes;
    if (ceiling <= 0 || layers == 0 || kv_heads <= 0) { return false; }
    if (per_block <= 0) { return false; }
    state.ceiling_bytes   = ceiling;
    state.bytes_per_block = per_block;
    state.layers          = static_cast<std::int32_t>(layers);
    state.kv_heads        = kv_heads;
    state.rans_layers     = rans_layers;
    return state.live();
}

// [F1172] THE STRIDE-SPELLING ENTRY POINT.  It does NOT replace the one above: it fills the same
// state from the array the pool owns, derives the `rans_layers` reading from that SAME array, and
// leaves `block_charge` for the caller to point at its plan. A caller with no strides keeps the
// scalar spelling and gets byte-for-byte the old state -- the OFF path is unchanged because it is
// a different function.
[[nodiscard]] inline bool kv_block_stage_from_strides(KvBlockStageState& state,
                                                      const std::int32_t* const slot_bytes,
                                                      std::uint32_t layers,
                                                      std::int32_t kv_heads,
                                                      std::int64_t population_blocks = 0) noexcept {
    const std::int64_t per_block =
        kv_block_stage_bytes_per_block_from_strides(slot_bytes, layers, kv_heads);
    const std::uint32_t rans_layers =
        kv_block_stage_rans_layers_from_strides(slot_bytes, layers);
    if (!kv_block_stage_from_env(state, layers, kv_heads, rans_layers, population_blocks)) {
        return false;
    }
    // The two spellings must agree, or the accounting line would describe a stack the charge does
    // not implement. This is the runtime half of the `static_assert` above.
    // [G6 / GAP-THIRDRECORD] THE SUM IS THE CHARGE; THE COUNT IS A READING. The gate that used
    // to sit here (`state.bytes_per_block != per_block` -> OFF) switched the stage OFF the day a
    // third price appeared: a count is a sufficient statistic only while the table has TWO prices
    // (see the static_assert above). The sum is the real input, so it is ADOPTED; the count stays
    // a reading. The OFF convention for an UNCOMPUTABLE sum is unchanged (a layer with no booked
    // stride -> per_block == 0 -> OFF), so nothing becomes silently uncapped.
    if (per_block <= 0) {
        state = KvBlockStageState{};
        return false;
    }
    // ⚠⚠ [F1209 2026-09-29] THE STRIDE SUM IS NOT WRITTEN INTO `bytes_per_block` ANY MORE, AND THAT
    // ONE ASSIGNMENT WAS THE OTHER HALF OF THE SPLIT RULER. WHAT IT USED TO DO: this line read
    // `state.bytes_per_block = per_block;` -- the sum of the pool's BOOKED STRIDES -- while the call
    // three lines above (`kv_block_stage_from_env`, which has ALREADY set `bytes_per_block` from
    // `kv_block_stage_bytes_per_block(layers, kv_heads, rans_layers)`) put the PLANE-ruler charge
    // there. The deeper assignment therefore UNDID the one above it, on the LIVE path only.
    //
    // ⭐ THE TWO SPELLINGS ARE THE SAME HEAD-PAGES IN TWO RULERS, WHICH IS WHY THIS IS A UNIT BUG
    // AND NOT A DISAGREEMENT ABOUT THE OBJECT:
    //     1,181,696 / 9,232  = 128.0 EXACTLY      (128 head-pages, priced as a COLD RECORD)
    //     2,162,688 / 16,896 = 128.0 EXACTLY      (the same 128 head-pages, priced as a RESIDENT
    //                                              PLANE)
    // One block is 16 layers x 4 kv heads x 2 planes = 128 head-pages either way. So the stage was
    // pricing THE IDENTICAL 128 HEAD-PAGES per block as a cold record (9,232/page) on the path that
    // reads this scalar and as a resident plane (16,896/page) on the planned path -- inside ONE
    // ceiling. That is a 1.83x (16,896/9,232) slack on every block the plan does not cover, which is
    // what "one ceiling, two currencies" means concretely.
    //
    // WHY THAT IS A DEFECT AND NOT A PREFERENCE: the plan's `block_bytes[b]` is
    // `cell_vector_charge` = 2 * kv_heads * sum_l plane_bytes, so a PLANNED block charges 2,162,688
    // at the shipped all-int8 shape while an UNPLANNED one -- `bytes_per_block_for`, the fallback
    // this scalar IS -- charged 1,181,696. This file's own note at the scalar's definition says
    // exactly why the scalar had to move ("the scalar must move with the plan, or an UNPLANNED block
    // would charge 1,181,696 while a planned one charges 2,162,688 and the ceiling would be spent in
    // two currencies at once") -- and this assignment was defeating it. The planes cut had moved the
    // FUNCTION; this line moved the value back.
    //
    // ⚠ THE ALTERNATIVE, NAMED AND REJECTED HERE RATHER THAN LEFT TO A READER: one could argue the
    // stride is the pool's BOOKED residency, so charging the ceiling at the plane price makes it
    // CONSERVATIVE -- safe, but admitting ~45 % fewer unplanned blocks than the booked bytes would
    // hold. I REJECT THAT, FOR THE CUT'S OWN REASON: the charge sums `plane_bytes` because that is
    // the memory a DEMOTION actually gives back, so the plane ruler is the authoritative one and the
    // stride spelling is the stale one -- and a ceiling whose unit depends on whether a block
    // happens to be planned is the split this line closes. A ceiling that is cautious by one ruler
    // is a smaller error than a ceiling that is two ceilings at once. (If a run ever shows the
    // booked stride is ALSO the resident truth, the fix is to correct `plane_bytes`, not to keep two
    // prices here; that reading does not exist yet and is NOT claimed.)
    //
    // THE OFF GATE ABOVE IS UNTOUCHED: `per_block <= 0` is still "a layer has no booked stride, so
    // the sum is not computable", which is this function's OFF convention and not a ruler question.
    // AND THE READING IS KEPT: `per_block` is stored in `booked_stride_bytes_per_block`, printed
    // beside the charge, so the pool's own booking is still quotable and the before/after arms can
    // still be compared against the retired ruler.
    state.booked_stride_bytes_per_block = per_block;
    return true;
}

// [F1172] ATTACH THE PLAN.  One call, and it is the ONLY place the per-block charge enters. A
// null table is the OFF switch for the widening alone: the scalar charge still governs.
// [F1231-budget-ruler] THE TWO ENV SPELLINGS AND THE ONE RESOLUTION LIVE ABOVE THIS FILE's
// `KvBlockStageState` -- moved there because the state carries `budget_ruler` as a field and its
// `describe()` prints it. `kv_block_stage_ceiling_from_env()` below is kept for the byte spelling
// alone and is no longer the live reader: `kv_block_stage_budget_from_env(charge_per_block)` is.
inline void kv_block_stage_attach_plan(KvBlockStageState& state,
                                       const product::BlockChargeTable& table) noexcept {
    state.block_charge = table;
    state.charge_from_plan = table.bytes != nullptr && table.count != 0;
}

} // namespace ninfer::product
