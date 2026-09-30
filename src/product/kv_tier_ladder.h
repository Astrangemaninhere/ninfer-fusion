#pragma once

// F1202 -- THE TIER LADDER: **4bit / 3bit / 2bit, EACH ONE REACHABLE, ONE RUNG AT A TIME, PER
// CELL, AND EACH STEP'S CURRENCY ON THE SAME RULER THE CHARGE SUMS.**
//
// WHY THIS FILE EXISTS -- THE OWNER'S TWO SENTENCES, VERBATIM, ARE THE SPEC:
//   * 「注意不是整个格int8 是格里的某些层int8 格里应该有各种各样的量化模式而不是整格一致」
//   * 「4bit 3bit 2bit的三层降级都得是通的」
//
// ⚠ AND THE TREE FAILS THE SECOND ONE TODAY, IN CODE, IN TWO SEPARATE PLACES. Both are read in
// source at `src/product/kv_cell_modes.h` sha256 `b2a77dd4…` (the charge-cut revision, the one the
// running relink is compiling), and both are measured by the proofs in `../tests/`:
//
//   (1) THE FIRST STEP OF THE RUNG ORDER CROSSES FAMILIES -- AND THIS PARAGRAPH OVER-CLAIMED.
//       [⚠ CORRECTED BY F1209 2026-09-29; THE OVER-CLAIM IS QUOTED AND MARKED, NOT DELETED.]
//       `cell_next_cheaper`'s site ② picks `argmax{ plane_bytes' < plane_bytes }` over EVERY rung in
//       the set. From `int8` (plane 16,896) the strictly cheaper rungs are
//       `nvfp4 9,216 / rk4v4 8,704 / rk3v4 6,656 / e8-2bit 4,608`, and the argmax of those is
//       **`nvfp4`** -- a rung in a DIFFERENT family whose cold record is WIDER (9,632 > 9,232), so
//       the walk's first step EXPANDS the cell's cold slot.
//       WHAT THIS PARAGRAPH USED TO SAY, IN ITS OWN WORDS: "So the walk's first step is always
//       int8 -> nvfp4, `rk4v4` is **never** the answer to any step, and the 4-bit and 3-bit tiers are
//       dead values in the enum. **A rung that exists in an enum but cannot be selected is exactly
//       the failure mode the owner named.**" THE "DEAD VALUES" HALF OF THAT IS FALSE, read out of
//       the source at F1209: site ① DISCARDS every rung that is not strictly cheaper in plane
//       (`candidate.plane_bytes >= here.plane_bytes -> continue`), and site ② then keeps the LARGEST
//       plane among the survivors -- the rung NEAREST below `here`, NOT the cheapest. From `int8`
//       the survivors are `nvfp4 9,216 / rk4v4 8,704 / rk3v4 6,656 / e8-2bit 4,608` and site ②
//       answers `nvfp4`; from `nvfp4` the survivors are `rk4v4 / rk3v4 / e8-2bit` and site ② answers
//       **`rk4v4`**, then `rk3v4`, then `e8-2bit`. THE CHAIN IS TRAVERSABLE, one rung at a time:
//       int8 -> nvfp4 -> rk4v4 -> rk3v4 -> e8-2bit, and NO rung of the table is a dead enum value.
//       SITE ② MUST NOT BE "FIXED": "nearest below" is precisely what makes a one-rung-at-a-time
//       walk walk one rung at a time, and pruning `nvfp4` out of the order would be a LADDER CHOICE,
//       not a repair -- which is why this file's lattice is offered as a SELECTABLE alternative
//       (the cold-slot-expansion-free chain) and never as a replacement for the cheaper-below chain.
//       WHAT WAS ACTUALLY DEAD WAS THE FIRST STEP'S CURRENCY -- defect (2) below, and nothing else.
//       The reason nothing moved before F1209 is that one site, and this paragraph's own remedy is
//       intact: the 4/3/2 lattice IS the owner's ladder and IS what `cell_next_tier` answers with.
//
//   (2) THE STEP'S CURRENCY IS ON THE RETIRED RULER. `cell_next_cheaper`'s site ③ returns
//       `here.record_bytes - cell_rung(best).record_bytes`. On the live revision the CHARGE is
//       `cell_vector_charge` summing `rung.plane_bytes` (`kv_cell_modes.h:851`, the charge cut)
//       while ③ still returns a RECORD delta, so the driver's `plan.total_bytes -= step.bytes_saved`
//       subtracts one ruler from another. And the number it produces on the step (1) forces is
//       `9,232 - 9,632 = -400` -- negative, so the driver refuses it at
//       `kv_block_descent.h:480` (`if (step.bytes_saved <= 0)`) and **the walk moves ZERO cells on
//       every budget**. MEASURED in `blob_F1192.md` 33.4 on the arm that binds: `refused=768` =
//       `cells`, `steps=0`.
//
// THE FIX, AND IT IS ONE RULE WITH TWO HALVES -- BOTH HALVES ARE FORCED BY THE TWO DEFECTS:
//
//   next(S, f)  :=  argmax_{ f' in S : priced(f')
//                                AND plane_bytes(f') < plane_bytes(f)        [ a demotion ]
//                                AND record_bytes(f') <= record_bytes(f) }   [ NOT a regression ]
//                   plane_bytes(f')
//
//   * HALF ONE (the `record_bytes` exclusion) is what makes the ladder the owner's ladder: a rung
//     whose cold record is WIDER than the cell's current record is not a DEGRADATION of that cell,
//     it is an EXPANSION of its cold slot -- `nvfp4`'s record is 9,632 against the raw slot's
//     9,232, so a step INTO `nvfp4` costs the pool 400 B/plane while it saves the resident plane.
//     Excluding it is not a judgement call: `nvfp4` is the only priced rung in the table whose
//     record is not the minimum, and the tree's own three static_asserts
//     (`decoder_state.cpp:870-872`) already prove the other five SHARE one record. With it
//     excluded, the argmax from `int8` is **`rk4v4`**, and the lattice ladder
//     `int8 -> rk4v4 -> rk3v4 -> e8-2bit` is walked one rung at a time, in that order.
//   * HALF TWO (the currency) is what makes the step BUY anything: `plane_bytes(here) -
//     plane_bytes(to)`, the same quantity `cell_vector_charge` now sums (`:851`). The three steps
//     are `8,192 / 2,048 / 2,048` B per plane -- all strictly positive, so the driver's
//     strictly-decreasing termination argument holds AND the quantity it decreases is the quantity
//     the stop condition reads. The two rulers are one ruler again.
//
// ⚠ WHAT THIS FILE DOES *NOT* DO, NAMED SO NO READER HAS TO DISCOVER IT:
//   * It does not delete `nvfp4` and it does not touch `cell_next_cheaper`. `nvfp4` remains a
//     priced rung, remains in `kCellRungs`, and remains reachable by a caller that asks for it
//     (`cell_next_cheaper` is unchanged). It is excluded from THE LADDER'S NEXT-RUNG STEP because
//     that step is a demotion and `nvfp4` is not one. The two functions are different questions
//     and both stay answerable: `cell_next_cheaper` = "the priciest rung below me", this file's
//     `cell_next_tier` = "the next rung of the 4/3/2 lattice below me". The second is the one the
//     owner's sentence names, because it is the one that keeps the three tiers connected.
//   * It does not change the charge. `cell_vector_charge` is untouched and still sums
//     `plane_bytes`; this file ADOPTS that ruler for the delta rather than moving the charge again.
//   * [F1209 2026-09-29] IT LANDS ITSELF NOW, SO THIS BULLET IS SUPERSEDED AND KEPT AS HISTORY. Its
//     own words were: "It does not land itself. Nothing in `src/` includes this header, so it is
//     INERT in the engine until a follow-up window adds the include. That is the compile-time gate:
//     the file is additive, header-only, reads no environment, and cannot change a `qwen3_6` target
//     by existing." TWO OF THESE CLAUSES ARE NOW FALSE AND ONE WAS ALREADY FALSE WHEN WRITTEN: F1209
//     added the include (the driver's step selection reaches this file), and the
//     `NINFER_KV_DESCENT_MAX_TIER` reader further down IS an environment read that predates this
//     note. What survives, and is the part that mattered: the file is ADDITIVE -- it moves no charge
//     and no dtype by itself -- so what it changes in `qwen3_6` is exactly what the driver's call
//     site is told to change, and nothing else.
//   * It does not invent a codec or a pack arm. `e8-2bit`'s row still has `LayerColdCodec::Rk4v4Raw`
//     and the tree still has no cold slot for `DType::E8K2Kv`; a cell that reaches the floor is
//     still not echoable (`cell_mode_codec`'s note). This file prices the WALK, not the packer.

// ⚠ [F1209 2026-09-29] THIS HEADER USED TO INCLUDE `product/kv_block_descent.h`, AND MUST NOT.
// The driver now includes THIS header -- that is the wiring that makes the ladder non-inert, and it
// is the whole point of F1209 -- so a back-include would close a cycle, and a cycle here is NOT
// benign: a TU that includes THIS header first would skip the driver's include of it (this file is
// already mid-inclusion) and the driver's body would then be compiled with `cell_next_tier` still
// undeclared. No include ORDER fixes that; breaking the cycle does. The include was DECORATIVE --
// nothing in this file uses a name from `kv_block_descent.h` (every walk here is built on
// `kv_cell_modes.h` alone) -- so removing it costs this file nothing and makes the wiring
// order-independent.
#include "product/kv_cell_modes.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::product {

// ---------------------------------------------------------------------------------------------
// 1. THE TIER AXIS, AS THE OWNER'S THREE NAMES PLUS THE TWO ENDS OF THE LATTICE.
// ---------------------------------------------------------------------------------------------
//
// `TierBits` is the number in the owner's own sentence -- "4bit 3bit 2bit". It is DERIVED, not
// declared: the only function below that fills it reads `kv_e8_width.h`'s width axis through the
// rung's dtype, so a future rung cannot be given a tier by hand. `None` is not a failure value:
// it is the truthful answer for `int8` (a raw slot, not a lattice tier) and for `bf16`.
enum class TierBits : std::uint8_t {
    None = 0,  // not an e8 lattice tier (int8 / bf16 / nvfp4)
    B2   = 2,  // the floor, the owner's "2bit"
    B3   = 3,  // the middle rung, the owner's "3bit"
    B4   = 4,  // the top lattice rung, the owner's "4bit"
};

[[nodiscard]] constexpr const char* tier_bits_name(TierBits bits) noexcept {
    switch (bits) {
    case TierBits::B4: return "4bit";
    case TierBits::B3: return "3bit";
    case TierBits::B2: return "2bit";
    case TierBits::None: break;
    }
    return "-";
}

// ---------------------------------------------------------------------------------------------
// 2. THE LADDER ITSELF: FOUR RUNGS, ORDERED, WITH THE DEPTH WRITTEN DOWN RATHER THAN COMPUTED.
// ---------------------------------------------------------------------------------------------
//
// WHY THE DEPTH IS A LITERAL AND NOT DERIVED BY COUNTING: `depth` is the number of
// `cell_next_tier` steps from the start (`int8`). Writing it as a literal makes the reachability
// asserts below STATEMENTS ABOUT NUMBERS a reader can check by eye, and makes an accidental
// reordering of this array a compile error rather than a silent behaviour change.
//
// THE ORDER IS THE OWNER'S: 4bit -> 3bit -> 2bit, and the START is above all three of them. Every
// `plane_bytes` in this table is read through `cell_rung()` from `kCellRungs`, so not one number is
// re-typed here -- the column is the one `kv_cell_modes.h` §3b `static_assert`s as strictly
// decreasing (`bf16 > int8 > nvfp4 > rk4v4 > rk3v4 > e8-2bit`, `:453-464`).
struct LadderRung {
    CellMode mode = CellMode::Int8;
    TierBits bits = TierBits::None;
    std::uint32_t depth = 0;   // steps from `Int8`; 0 is the start
};

inline constexpr std::array<LadderRung, 4> kLatticeLadder{{
    {CellMode::Int8,    TierBits::None, 0},  // the START -- the owner's (i), the engine's default
    {CellMode::Rk4v4,   TierBits::B4,   1},  // 4bit -- e8 lattice, plane 8,704
    {CellMode::Rk3v4,   TierBits::B3,   2},  // 3bit -- e8 lattice, plane 6,656
    {CellMode::E8_2bit, TierBits::B2,   3},  // 2bit -- THE FLOOR, plane 4,608
}};

inline constexpr std::uint32_t kLatticeRungCount = 4;
// THE FLOOR'S DEPTH, as a named constant so "how much reached the floor" has one spelling.
inline constexpr std::uint32_t kLatticeFloorDepth = 3;

[[nodiscard]] constexpr const LadderRung* ladder_rung_of(CellMode mode) noexcept {
    for (const LadderRung& rung : kLatticeLadder) {
        if (rung.mode == mode) { return &rung; }
    }
    return nullptr;   // NOT on the lattice: `bf16` and `nvfp4` are above / beside it
}

[[nodiscard]] constexpr TierBits tier_bits_of(CellMode mode) noexcept {
    const LadderRung* const rung = ladder_rung_of(mode);
    return rung == nullptr ? TierBits::None : rung->bits;
}

// THE DEPTH, WITH `-1` FOR "NOT ON THE LATTICE". `-1` is not an error and not a zero: `bf16` and
// `nvfp4` genuinely have no depth on this ladder, and a caller that read them as depth 0 would be
// reading them as the start.
[[nodiscard]] constexpr std::int32_t ladder_depth(CellMode mode) noexcept {
    const LadderRung* const rung = ladder_rung_of(mode);
    return rung == nullptr ? -1 : static_cast<std::int32_t>(rung->depth);
}

[[nodiscard]] constexpr CellMode ladder_rung_at_depth(std::int32_t depth) noexcept {
    for (const LadderRung& rung : kLatticeLadder) {
        if (static_cast<std::int32_t>(rung.depth) == depth) { return rung.mode; }
    }
    return CellMode::Int8;   // out of range: the identity, the START
}

[[nodiscard]] constexpr TierBits tier_bits_at_depth(std::int32_t depth) noexcept {
    for (const LadderRung& rung : kLatticeLadder) {
        if (static_cast<std::int32_t>(rung.depth) == depth) { return rung.bits; }
    }
    return TierBits::None;
}

// THE PLANE PRICE OF A CELL, READ THROUGH THE RUNG TABLE AND NEVER RE-TYPED. This is the ruler the
// charge moved onto (`kv_cell_modes.h:851`), and it is the only ruler this file uses for a delta.
[[nodiscard]] constexpr std::int32_t ladder_plane_bytes(CellMode mode) noexcept {
    return cell_rung(mode).plane_bytes;
}

// ---------------------------------------------------------------------------------------------
// 3. THE STEP: THE CORRECTED RUNG-BELOW, ONE RUNG AT A TIME, ON THE CHARGE'S RULER.
// ---------------------------------------------------------------------------------------------
//
// THE FOUR REFUSALS, AND THEY ARE THE SAME FOUR `StepRefusal` VALUES THE TREE ALREADY HAS -- this
// function returns the tree's own `CellStep` so a caller can swap one for the other without
// touching a field name. What changes is only WHICH rung is reached and WHAT the delta is:
//
//   AlreadyAtFloor      the lattice has nothing below the cell (the 2bit rung included)
//   RungBytesUnknown    G6: a cheaper-by-reputation rung has no byte cost. NOTHING in the current
//                       table is `Unpriced` (six `static_assert`s pin all six prices), so this is
//                       the master's ruling held in reserve: if the next rung ever loses its price,
//                       STOP AT THIS RUNG rather than skip it. The test sits ABOVE the `found`
//                       return for exactly the reason `kv_cell_modes.h:663-674` gives.
//   NoByteCurrency      kept for API symmetry with `cell_next_cheaper`. UNREACHABLE HERE BY
//                       CONSTRUCTION: because half one excludes every rung whose record is wider,
//                       and every rung this function can return is strictly cheaper on the plane
//                       ruler, the delta is strictly positive. `NoByteCurrency` returns only if a
//                       future rung table gains a rung that is plane-cheaper AND record-equal --
//                       which is a legal no-op step the walk must NOT take, so the arm is kept.
//   CurrentModeNotInSet / CurrentModeUnpriced   the caller's two errors, unchanged.
[[nodiscard]] constexpr CellStep cell_next_tier(CellAllowedSet set, CellMode from) noexcept {
    const CellRung& here = cell_rung(from);
    if (here.price_state != PriceState::Priced) {
        return CellStep{from, StepRefusal::CurrentModeUnpriced, 0, false, false};
    }
    if (!set.has(from)) {
        return CellStep{from, StepRefusal::CurrentModeNotInSet, 0, false, false};
    }

    CellMode best = from;
    bool found = false;
    bool blocked_by_unpriced = false;
    // `blocked_by_same_record` KEEPS THE LIVE FIELD'S MEANING -- "this step is BLOCKED because the
    // only cheaper rungs write the same cold record" -- and it is therefore FALSE on every legal
    // step of this function, INCLUDING the record-equal ones. On the record ruler a 9232 -> 9232
    // step is a G7 no-op; on the plane ruler the charge now sums (`kv_cell_modes.h:851`) it is
    // 8,192 B of resident memory, so it is a real step and not a block. Reporting it as "blocked"
    // would be the same ruler confusion in the other direction. It is returned as the SECOND FACT
    // only where the tree's own refusal semantics want it: on a refusal, where a caller can see
    // that the cheaper rungs it declined all wrote one record.
    bool has_record_equal_cheaper = false;

    for (const CellRung& candidate : kCellRungs) {
        if (candidate.mode == from || !set.has(candidate.mode)) { continue; }
        // G6: THE MASTER'S RULING, HELD IN RESERVE. An unpriced rung makes the byte ORDER itself
        // unknown, so the walk stops at `from` instead of spending the rung below the hole.
        if (candidate.price_state != PriceState::Priced) {
            blocked_by_unpriced = true;
            continue;
        }
        // HALF ONE, THE LINE THAT MAKES THE LADDER CONNECTED. A rung whose COLD RECORD is wider
        // than the cell's is an expansion of the cell's cold slot, not a demotion of it. It is
        // excluded from the STEP -- and it is still a priced row of `kCellRungs`, still reachable
        // through `cell_next_cheaper`, still reported by the census. This single comparison is the
        // difference between "int8 -> nvfp4 and stop" (the live bug) and "int8 -> rk4v4 -> rk3v4 ->
        // e8-2bit" (the owner's sentence).
        if (candidate.record_bytes > here.record_bytes) { continue; }
        if (candidate.plane_bytes >= here.plane_bytes) { continue; }             // the demotion test
        if (!found || candidate.plane_bytes > cell_rung(best).plane_bytes) {     // argmax, as ②
            best = candidate.mode;
            found = true;
        }
        // A plane-cheaper rung that leaves the cold slot alone. Recorded as a FACT for the
        // refusal paths below and for no other purpose -- see the note at the declaration.
        if (candidate.record_bytes == here.record_bytes) { has_record_equal_cheaper = true; }
    }

    if (blocked_by_unpriced) {
        return CellStep{from, StepRefusal::RungBytesUnknown, 0, true, has_record_equal_cheaper};
    }
    if (!found) {
        // The lattice is exhausted. `NoByteCurrency` is the tree's name for "the only rungs left
        // are codec changes on one record" and it is the honest answer here too.
        if (has_record_equal_cheaper) {
            return CellStep{from, StepRefusal::NoByteCurrency, 0, false, true};
        }
        return CellStep{from, StepRefusal::AlreadyAtFloor, 0, false, false};
    }
    // HALF TWO: THE DELTA IS `plane_bytes`, THE QUANTITY THE CHARGE SUMS. Strictly positive
    // because `found` was set only for candidates strictly below `here.plane_bytes` -- so the
    // driver's termination argument ("every accepted step strictly decreases the total") holds on
    // the very quantity `plan.total_bytes` now measures.
    //
    // ⚠⚠ THE UNIT OF THIS RETURNED NUMBER, NAMED BECAUSE THE LIVE DRIVER GETS IT WRONG: it is ONE
    // PLANE OF ONE LAYER. The total it must be subtracted from is
    // `cell_vector_charge(...) = 2 * kv_heads * sum_l plane_bytes(V[l])`, so a caller stepping one
    // cell must subtract `2 * kv_heads * bytes_saved`, NOT `bytes_saved`. `DEVELOPMENT-3` below
    // records the live defect and `ladder_step_total_delta` is the one spelling of the factor.
    // ✔ [F1209 2026-09-29] CLOSED: the driver's subtraction now reads
    // `plan.total_bytes -= ladder_step_total_delta(step, kv_heads);`, so the ONE spelling is the
    // only spelling left. This note is kept above as the statement of what was wrong, because the
    // assert that pins the fixed behaviour lives twenty lines further down.
    return CellStep{best, StepRefusal::None, here.plane_bytes - cell_rung(best).plane_bytes, false,
                    false};
}

// THE STEP'S EFFECT ON `plan.total_bytes`, IN THE TOTAL'S OWN UNIT. One place, so the
// `2 * kv_heads` factor cannot be forgotten at a call site.
//
// ⚠⚠ DEVELOPMENT-3, NAMED AND MEASURED -- A THIRD DEFECT OF THE SAME FAMILY AS DEVELOPMENT-1 AND
// -2, FOUND BY THE SWEEP IN `../tests/` AND NOT REPORTED ANYWHERE IN THIS LINE'S RECORDS: the live
// driver does `plan.total_bytes -= step.bytes_saved;` (`kv_block_descent.h:485`) where
// `total_bytes` is a `2 * kv_heads *`-scaled SUM over layers and `bytes_saved` is a ONE-LAYER,
// ONE-PLANE delta. The subtraction is therefore SHORT BY EXACTLY `2 * kv_heads` (= 8 on the
// shipped 4-head stack). MEASURED: with the fix applied to the walk in `../tests/`, a 48-block
// plan whose cells all reach the floor reports `total_bytes = 28,311,552` -- and the unscaled
// subtraction reports `94,371,840` for the very same vector set (`= 103,809,024 - 9,437,184`).
// THE CONSEQUENCE IS NOT COSMETIC: `saved_bytes` is understated 8x, so `total_bytes <= budget`
// stops being reachable at a budget that is in fact reachable, and every "the budget does not bind"
// reading taken from a moving walk is off by that factor. It is the same disease as DEVELOPMENT-2
// (a delta in one ruler subtracted from a total in another) with a different missing factor, which
// is why it is named here rather than folded into -2.
[[nodiscard]] constexpr std::int64_t ladder_step_total_delta(const CellStep& step,
                                                            std::int32_t kv_heads) noexcept {
    if (!step.legal() || kv_heads <= 0) { return 0; }
    return 2 * static_cast<std::int64_t>(kv_heads) * static_cast<std::int64_t>(step.bytes_saved);
}

// ---------------------------------------------------------------------------------------------
// 4. THE TIER CENSUS, BY BLOCK AND BY LAYER. THIS IS THE PER-BLOCK HALF.
// ---------------------------------------------------------------------------------------------
//
// THE OWNER RULED TWICE THAT BLOCK GRANULARITY IS CORE ("块粒度是核心，必须深入到块来分配"; 「梯度要
// 记得按块，不能全层下降」). `blob_F1192.md` 33.5 named what was missing: the four counters the
// driver prints (`demoted_layers` / `demoted_blocks` / `demoted_max_per_layer` /
// `demoted_max_per_block`) tell per-cell from whole-layer/whole-block, but NOTHING reports where the
// demoted cells LANDED. A per-cell descent that moved everything to one tier and a per-cell descent
// that spread cells across all three would print the same four numbers.
//
// SO THE CENSUS IS PER BLOCK, AND IT CARRIES THE THREE TIERS SEPARATELY. `per_block[b]` is one row
// of `kCellModeCount` counts, and the three numbers `cells_at_bits4 / bits3 / bits2` are what make
// "the 4bit, 3bit and 2bit demotions all happened" a READING rather than a claim.
struct TierCensus {
    std::array<std::uint32_t, kCellModeCount> cells{};   // how many cells sit on each rung
    std::uint32_t cells_at_bits4 = 0;   // the owner's "4bit"
    std::uint32_t cells_at_bits3 = 0;   // the owner's "3bit"
    std::uint32_t cells_at_bits2 = 0;   // the owner's "2bit" -- the floor
    std::uint32_t cells_at_int8 = 0;    // the start; still the majority under a loose budget

    constexpr void count(CellMode mode) noexcept {
        cells[static_cast<std::uint32_t>(mode)] += 1;
        switch (tier_bits_of(mode)) {
        case TierBits::B4: cells_at_bits4 += 1; break;
        case TierBits::B3: cells_at_bits3 += 1; break;
        case TierBits::B2: cells_at_bits2 += 1; break;
        case TierBits::None: break;
        }
        if (mode == CellMode::Int8) { cells_at_int8 += 1; }
    }

    // THE THREE RUNGS EACH HAD A CELL: the owner's acceptance item, as one predicate. It is FALSE
    // on the pre-image tree for every budget, because the walk cannot leave `int8` at all.
    [[nodiscard]] constexpr bool all_three_tiers_used() const noexcept {
        return cells_at_bits4 != 0 && cells_at_bits3 != 0 && cells_at_bits2 != 0;
    }
};

// ---------------------------------------------------------------------------------------------
// 4b. THE TRANSIT CENSUS -- AND WHY THE ONE ABOVE CANNOT ANSWER THE OWNER'S QUESTION ALONE.
// ---------------------------------------------------------------------------------------------
//
// ⚠⚠ MEASURED, AND IT INVALIDATED THIS LINE'S FIRST ACCEPTANCE PREDICATE. The walk is LEVEL-WISE
// (`block_descent_plan`'s own note: "every cell moves one rung before any cell moves two"). So when
// the walk is in level 2, EVERY cell is already at the 4-bit rung and none is still at int8; a
// strictly level-wise walk therefore has at most TWO tiers occupied AT ANY INSTANT, and
// `all_three_tiers_used()` on a resting state is **UNSATISFIABLE BY CONSTRUCTION, not by budget**.
// The sweep in `../tests/` proved it: no arm in the grid ever printed 4bit, 3bit and 2bit all
// non-zero -- each arm printed either `768 0 0`, `0 768 0`, `0 0 768`, or a split between exactly
// two adjacent tiers.
//
// THE OWNER'S SENTENCE IS ABOUT DEMOTIONS, NOT ABOUT RESTING STATES. 「4bit 3bit 2bit的三层降级都得
// 是通的」 says the THREE DEMOTIONS must be live -- and a demotion is an EVENT with a time, not a
// state a cell can rest in. So the instrument that answers it counts the STEPS: every accepted step
// whose destination is a lattice rung is one fired demotion, recorded by the tier it entered. That
// counter IS satisfiable by a level-wise walk (and is satisfied: three levels -> 768 events each),
// and it is the honest reading of "the rung is exercised".
struct TierTransit {
    std::uint32_t into_bits4 = 0;   // the "4bit" demotion fired this many times
    std::uint32_t into_bits3 = 0;   // the "3bit" demotion
    std::uint32_t into_bits2 = 0;   // the "2bit" demotion, i.e. arrivals at the floor
    std::uint32_t into_int8 = 0;    // bf16 -> int8; only fires if a cell ever STARTS above int8

    [[nodiscard]] constexpr bool all_three_demotions_fired() const noexcept {
        return into_bits4 != 0 && into_bits3 != 0 && into_bits2 != 0;
    }
    [[nodiscard]] constexpr std::uint32_t total() const noexcept {
        return into_bits4 + into_bits3 + into_bits2 + into_int8;
    }

    // WHERE A STEP LANDED, IN THE TIER'S OWN NAME. `TierBits::None` (a step to int8 or bf16) is
    // counted in `into_int8` only for the int8 case, because that is the only `None`-tier step the
    // ladder can produce: `bf16` is never a destination (`ladder_chain_steps` walks away from it).
    constexpr void record_step_into(CellMode to) noexcept {
        switch (tier_bits_of(to)) {
        case TierBits::B4: into_bits4 += 1; break;
        case TierBits::B3: into_bits3 += 1; break;
        case TierBits::B2: into_bits2 += 1; break;
        case TierBits::None: if (to == CellMode::Int8) { into_int8 += 1; } break;
        }
    }
};

struct TierTransitByBlock {
    std::vector<TierTransit> per_block;   // size == blocks
    TierTransit total{};

    [[nodiscard]] std::uint32_t blocks_with_all_three_demotions() const noexcept {
        std::uint32_t n = 0;
        for (const TierTransit& row : per_block) {
            if (row.all_three_demotions_fired()) { n += 1; }
        }
        return n;
    }

    [[nodiscard]] std::string describe() const {
        std::string out = "demotions[into4bit=" + std::to_string(total.into_bits4) +
                          " into3bit=" + std::to_string(total.into_bits3) +
                          " into2bit=" + std::to_string(total.into_bits2) +
                          " intoint8=" + std::to_string(total.into_int8) + "]";
        out += " blocks_that_fired_all_three=" +
               std::to_string(blocks_with_all_three_demotions()) + "/" +
               std::to_string(per_block.size());
        return out;
    }
};

struct TierCensusByBlock {
    std::vector<TierCensus> per_block;   // size == blocks
    TierCensus total{};

    [[nodiscard]] std::uint32_t blocks_using_all_three_tiers() const noexcept {
        std::uint32_t n = 0;
        for (const TierCensus& row : per_block) {
            if (row.all_three_tiers_used()) { n += 1; }
        }
        return n;
    }
    // THE PER-BLOCK UNIFORMITY CHECK, the mirror of `block_vector_is_uniform`: TRUE exactly when
    // every block's row is the same row, i.e. the descent was effectively block-blind. This is the
    // number a whole-window descent prints 1 for and a per-block descent does not.
    [[nodiscard]] bool rows_all_equal() const noexcept {
        for (std::size_t i = 1; i < per_block.size(); ++i) {
            if (per_block[i].cells != per_block[0].cells) { return false; }
        }
        return !per_block.empty();
    }

    [[nodiscard]] std::string describe() const {
        std::string out = "tiers[4bit=" + std::to_string(total.cells_at_bits4) +
                          " 3bit=" + std::to_string(total.cells_at_bits3) +
                          " 2bit=" + std::to_string(total.cells_at_bits2) +
                          " int8=" + std::to_string(total.cells_at_int8) + "]";
        out += " blocks=" + std::to_string(per_block.size());
        out += " blocks_using_all_three=" + std::to_string(blocks_using_all_three_tiers());
        out += " rows_all_equal=" + std::string(rows_all_equal() ? "yes" : "no");
        return out;
    }
};

[[nodiscard]] inline TierCensusByBlock ladder_census(const std::vector<BlockVector>& vectors) {
    TierCensusByBlock census{};
    census.per_block.assign(vectors.size(), TierCensus{});
    for (std::size_t b = 0; b < vectors.size(); ++b) {
        const BlockVector& vector = vectors[b];
        for (std::uint32_t layer = 0; layer < vector.layers; ++layer) {
            const CellMode mode = vector.mode[layer];
            census.per_block[b].count(mode);
            census.total.count(mode);
        }
    }
    return census;
}

// ---------------------------------------------------------------------------------------------
// 5. THE LEVEL WALK, ON THE LADDER. THIS IS THE "EACH RUNG EXERCISED BY THE DESIGN" PROOF.
// ---------------------------------------------------------------------------------------------
//
// WHY A SECOND WALK EXISTS WHEN `block_descent_plan` ALREADY HAS ONE. Because the driver's walk is
// the thing that must be CHANGED, and a change to a `product/` header cannot be its own proof. This
// walk is the same shape (level-wise, one rung per cell per sweep, stop when the total fits) over
// `cell_next_tier`, so the proof program in `../tests/` can show the three rungs being reached
// WITHOUT the follow-up window's edit having landed yet. When the edit lands, this walk and the
// driver's walk must agree cell for cell, and that equality is a test a later round can write.
//
// ✔ [F1209 2026-09-29] THE EDIT LANDED. The driver selects this file's step function when
// `NINFER_KV_DESCENT_CHAIN=lattice` and its ceiling when `NINFER_KV_DESCENT_MAX_TIER` is set, so
// the equality above is now a ONE-SETTING statement instead of a hypothetical: on the `lattice`
// chain the two walks are the same shape (level-wise, one rung per cell per sweep, stop when the
// total fits), the same ceiling rule (`depth >= 0 && depth >= ceiling` -> refuse the cell BY NAME,
// never skip the rung), and the same delta scaling (`ladder_step_total_delta`). THE EQUALITY IS
// STILL OWED AS A RUN -- `--run` needs the card, and this window had none -- and on the DEFAULT
// chain the two walks deliberately differ by the `nvfp4` step, which is the whole reason the chain
// is a setting and not a decision taken in this file.
//
// The `ladder_ceiling_depth` input is the RUNG CEILING: 0 = int8 only (nothing may move), 1 = stop
// at 4bit, 2 = stop at 3bit, 3 = the floor is reachable. It exists because it is the second axis of
// the table in `../tools/`, and because "how deep did the walk go" is the owner's "按预算" story
// made into a knob rather than a guess.
struct LadderWalkSpec {
    std::uint32_t blocks = 0;
    std::uint32_t layers = kBlockVectorLayerAxis;
    std::int32_t kv_heads = 0;
    std::int64_t budget_bytes = 0;          // <= 0 means OFF (the default vector, unwalked)
    std::int32_t ladder_ceiling_depth = static_cast<std::int32_t>(kLatticeFloorDepth);
    std::vector<CellAllowedSet> allowed;    // size == layers; empty means `cell_all_modes()`
};

struct LadderWalkResult {
    std::vector<BlockVector> vectors;
    std::vector<std::int64_t> block_bytes;
    TierCensusByBlock census{};
    TierTransitByBlock transit{};           // ⭐ the owner's item: the three DEMOTIONS, counted
    std::int64_t planned_bytes = 0;
    std::int64_t total_bytes = 0;
    std::uint32_t steps = 0;
    std::uint32_t sweeps = 0;
    std::uint32_t refused = 0;
    std::uint32_t demoted = 0;
    std::uint32_t cells_total = 0;
    std::uint32_t max_depth_reached = 0;
    bool priced = true;
    bool budget_binds = false;              // planned > budget at entry

    [[nodiscard]] std::int64_t saved_bytes() const noexcept { return planned_bytes - total_bytes; }
    [[nodiscard]] std::int64_t headroom_bytes() const noexcept { return budget_bytes - total_bytes; }
    std::int64_t budget_bytes = 0;

    [[nodiscard]] std::string describe() const {
        std::string out = "ladder-walk";
        out += " budget_bytes=" + std::to_string(budget_bytes);
        out += " planned_bytes=" + std::to_string(planned_bytes);
        out += " total_bytes=" + std::to_string(total_bytes);
        out += " saved_bytes=" + std::to_string(saved_bytes());
        out += " headroom_bytes=" + std::to_string(headroom_bytes());
        out += " binds=" + std::string(budget_binds ? "yes" : "no");
        out += " cells=" + std::to_string(cells_total);
        out += " sweeps=" + std::to_string(sweeps);
        out += " steps=" + std::to_string(steps);
        out += " demoted=" + std::to_string(demoted);
        out += " refused=" + std::to_string(refused);
        out += " max_depth=" + std::to_string(max_depth_reached);
        out += " price=" + std::string(priced ? "priced" : "UNPRICED");
        out += " " + census.describe();
        out += " " + transit.describe();
        return out;
    }
};

// ONE RUN OF THE LADDER WALK. Pure, no I/O, no environment, no allocation beyond the vectors.
[[nodiscard]] inline LadderWalkResult ladder_level_walk(const LadderWalkSpec& spec) {
    LadderWalkResult result{};
    result.budget_bytes = spec.budget_bytes;

    const std::uint32_t layers =
        spec.layers == 0 ? 0U
                         : (spec.layers < kBlockVectorLayerAxis ? spec.layers
                                                                : kBlockVectorLayerAxis);
    const std::uint32_t blocks = spec.blocks;
    result.cells_total = blocks * layers;

    auto allowed_for = [&spec, layers](std::uint32_t layer) -> CellAllowedSet {
        if (spec.allowed.empty() || layer >= spec.allowed.size() || layer >= layers) {
            return cell_all_modes();
        }
        return spec.allowed[layer];
    };

    // THE START: every cell at its default rung, which is the owner's (i) "start = all-layer int8"
    // and which is also the engine's pre-image charge.
    BlockVector base = block_vector_default(layers);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        base.set(layer, cell_default_tier(allowed_for(layer)).mode);
    }
    result.vectors.assign(blocks, base);
    result.block_bytes.assign(blocks, 0);
    result.transit.per_block.assign(blocks, TierTransit{});

    const CellCharge base_charge = cell_vector_charge(base, spec.kv_heads);
    result.planned_bytes = static_cast<std::int64_t>(blocks) * base_charge.bytes;
    result.total_bytes = result.planned_bytes;
    result.budget_binds = spec.budget_bytes > 0 && result.planned_bytes > spec.budget_bytes;

    if (!base_charge.priced || spec.budget_bytes <= 0) {
        result.priced = base_charge.priced;
        for (std::uint32_t b = 0; b < blocks; ++b) { result.block_bytes[b] = base_charge.bytes; }
        result.census = ladder_census(result.vectors);
        return result;
    }

    // THE CEILING, CLAMPED INTO THE LADDER'S OWN RANGE.
    std::int32_t ceiling = spec.ladder_ceiling_depth;
    if (ceiling < 0) { ceiling = 0; }
    if (ceiling > static_cast<std::int32_t>(kLatticeFloorDepth)) {
        ceiling = static_cast<std::int32_t>(kLatticeFloorDepth);
    }

    const std::uint64_t max_sweeps = kLatticeRungCount;
    std::vector<bool> refused_here(result.cells_total, false);
    for (std::uint64_t sweep = 0; sweep < max_sweeps; ++sweep) {
        bool progressed = false;
        for (std::uint32_t block = 0; block < blocks; ++block) {
            for (std::uint32_t layer = 0; layer < layers; ++layer) {
                if (result.total_bytes <= spec.budget_bytes) { break; }
                BlockVector& vector = result.vectors[block];
                const CellMode current = vector.at(layer);
                const std::int32_t depth = ladder_depth(current);
                const std::size_t cell_index = static_cast<std::size_t>(block) * layers + layer;
                // THE RUNG CEILING, AND IT IS A REFUSAL RATHER THAN A SMALLER SET: a cell whose
                // next rung would be deeper than the ceiling is refused BY NAME, so the census can
                // tell "the ceiling stopped it" from "the ladder ran out".
                if (depth >= 0 && depth >= ceiling) {
                    if (!refused_here[cell_index]) {
                        refused_here[cell_index] = true;
                        result.refused += 1;
                    }
                    continue;
                }
                const CellStep step = cell_next_tier(allowed_for(layer), current);
                if (!step.legal() || step.bytes_saved <= 0) {
                    if (!refused_here[cell_index]) {
                        refused_here[cell_index] = true;
                        result.refused += 1;
                    }
                    continue;
                }
                vector.set(layer, step.to);
                result.total_bytes -= ladder_step_total_delta(step, spec.kv_heads);
                result.steps += 1;
                // ⭐ THE THREE DEMOTIONS, COUNTED WHERE AND WHEN THEY FIRE -- per cell, and
                // attributed to the block the cell belongs to. This is the owner's
                // 「三层降级都得是通的」 as a reading rather than a claim, and it is per-block by
                // construction: there is one row per block and no aggregate is written first.
                result.transit.per_block[block].record_step_into(step.to);
                result.transit.total.record_step_into(step.to);
                progressed = true;
                if (ladder_depth(step.to) > static_cast<std::int32_t>(result.max_depth_reached)) {
                    result.max_depth_reached = static_cast<std::uint32_t>(ladder_depth(step.to));
                }
            }
            if (result.total_bytes <= spec.budget_bytes) { break; }
        }
        result.sweeps += 1;
        if (result.total_bytes <= spec.budget_bytes) { break; }
        if (!progressed) { break; }
    }

    for (std::uint32_t b = 0; b < blocks; ++b) {
        result.block_bytes[b] = cell_vector_charge(result.vectors[b], spec.kv_heads).bytes;
    }
    result.census = ladder_census(result.vectors);
    for (std::uint32_t b = 0; b < blocks; ++b) {
        const BlockVector& vector = result.vectors[b];
        for (std::uint32_t layer = 0; layer < vector.layers; ++layer) {
            if (vector.mode[layer] != base.mode[layer]) { result.demoted += 1; }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------------------------
// 6b. THE RUNG CEILING, AS AN ENVIRONMENT KNOB -- BECAUSE THE TABLE'S SECOND AXIS NEEDS ONE.
// ---------------------------------------------------------------------------------------------
//
// THE SWEEP IN `../tools/` VARIES TWO THINGS: the budget and the DEEPEST RUNG THE WALK MAY REACH.
// The budget already has an env spelling the stage owns (`NINFER_KV_BLOCK_BUDGET_BYTES`); the rung
// ceiling had none, and a sweep axis that cannot be set per arm is not an axis. This is that knob.
//
// ⚠ IT IS READ HERE AND NOT IN ANY LIVE HEADER, and that is deliberate: the live tree carries no
// reader for it, so an arm run against a binary built BEFORE this file was included sees the env
// IGNORED. `kv_tier_sweep.py` therefore CHECKS OBSERVABILITY rather than trusting the setting -- a
// requested ceiling that the `rungs[...]` census contradicts REFUSES the row. That check is what
// keeps a table honest on a binary where the knob is inert.
//
// UNSET IS NOT ZERO. Unset means "the ladder's full depth", i.e. the floor is reachable -- the
// permissive default, so an arm that forgets the variable measures the same thing the engine does
// today. A value that is not an integer, or is outside `[0, kLatticeFloorDepth]`, is CLAMPED and
// never silently treated as unset: a caller who typed `9` gets the floor, not a surprise OFF.
inline constexpr const char* kKvTierCeilingEnv = "NINFER_KV_DESCENT_MAX_TIER";

[[nodiscard]] inline std::int32_t ladder_ceiling_from_env() noexcept {
    const char* const text = std::getenv(kKvTierCeilingEnv);
    if (text == nullptr || *text == '\0') {
        return static_cast<std::int32_t>(kLatticeFloorDepth);   // unset: the whole ladder
    }
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text) { return static_cast<std::int32_t>(kLatticeFloorDepth); }  // not a number
    if (parsed < 0) { return 0; }
    if (parsed > static_cast<long>(kLatticeFloorDepth)) {
        return static_cast<std::int32_t>(kLatticeFloorDepth);
    }
    return static_cast<std::int32_t>(parsed);
}

// The ceiling's NAME, so an arm's log and the table's column cannot disagree about what "2" meant.
[[nodiscard]] inline std::string ladder_ceiling_name(std::int32_t ceiling) noexcept {
    std::string out = "ceiling=" + std::to_string(ceiling);
    switch (ceiling) {
    case 0: out += "(int8-only)"; break;
    case 1: out += "(stop@4bit)"; break;
    case 2: out += "(stop@3bit)"; break;
    case 3: out += "(floor:2bit)"; break;
    default: out += "(clamped)"; break;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// 7. THE REACHABILITY PROOFS. compile-time, and each one is the negation of a named defect.
// ---------------------------------------------------------------------------------------------
//
// The three that matter are (1) the ladder is CONNECTED, (2) THE 4/3/2 TIERS ARE EACH REACHABLE
// FROM THE START, and (3) the step from `int8` is NOT `nvfp4`. (3) is the live bug written as an
// assertion; if a future table edit reintroduces it, the build fails here.

static_assert(cell_next_tier(cell_all_modes(), CellMode::Int8).to == CellMode::Rk4v4,
              "F1202: from the START the next rung must be the 4-BIT tier `rk4v4`. The live "
              "`cell_next_cheaper` returns `nvfp4` here because its argmax crosses families and "
              "`nvfp4`'s plane (9,216) beats `rk4v4`'s (8,704) -- which is the defect this file "
              "exists to fix (the 4bit tier is otherwise unreachable and the enum value is dead).");
static_assert(cell_next_tier(cell_all_modes(), CellMode::Rk4v4).to == CellMode::Rk3v4,
              "F1202: 4bit -> 3bit must be one step, or the middle rung is unreachable.");
static_assert(cell_next_tier(cell_all_modes(), CellMode::Rk3v4).to == CellMode::E8_2bit,
              "F1202: 3bit -> 2bit must be one step, or the owner's floor is unreachable.");
static_assert(cell_next_tier(cell_all_modes(), CellMode::E8_2bit).refusal ==
                  StepRefusal::AlreadyAtFloor,
              "F1202: the 2bit rung IS the floor; below it there is nothing, and the walk must "
              "say so by NAME rather than return a step.");
static_assert(cell_next_tier(cell_all_modes(), CellMode::Int8).to != CellMode::Nvfp4,
              "F1202: THE LIVE BUG, AS AN ASSERTION. `nvfp4`'s cold record (9,632) is WIDER than "
              "the raw slot's (9,232), so a step into it expands the cell's cold slot. It is a "
              "priced rung and it is reachable through `cell_next_cheaper`; it is NOT the lattice's "
              "next rung.");

// EVERY STEP'S CURRENCY IS STRICTLY POSITIVE, ON THE CHARGE'S OWN RULER.
static_assert(cell_next_tier(cell_all_modes(), CellMode::Int8).bytes_saved ==
                  cell_rung(CellMode::Int8).plane_bytes - cell_rung(CellMode::Rk4v4).plane_bytes,
              "F1202: the int8 -> 4bit step must be paid in PLANE bytes (16,896 - 8,704 = 8,192), "
              "the ruler `cell_vector_charge` sums. A RECORD delta here is -400 and the driver "
              "refuses it, which is the measured `steps=0` of blob_F1192 33.4.");
static_assert(cell_next_tier(cell_all_modes(), CellMode::Rk4v4).bytes_saved == 2048 &&
                  cell_next_tier(cell_all_modes(), CellMode::Rk3v4).bytes_saved == 2048,
              "F1202: 4bit -> 3bit and 3bit -> 2bit are 2,048 B/plane each (8,704-6,656 and "
              "6,656-4,608). Both strictly positive, so the driver's termination argument holds "
              "for a three-rung descent.");
static_assert(cell_next_tier(cell_all_modes(), CellMode::Int8).bytes_saved > 0 &&
                  cell_next_tier(cell_all_modes(), CellMode::Rk4v4).bytes_saved > 0 &&
                  cell_next_tier(cell_all_modes(), CellMode::Rk3v4).bytes_saved > 0,
              "F1202: THE THREE RUNGS, EACH WITH A POSITIVE CURRENCY. This is the conjunct that "
              "was false in the live tree: nvfp4's step is +7,680 on the plane ruler but its "
              "RECORD delta is -400, so the live walk's only reachable step buys nothing.");

// THE LADDER IS THE OWNER'S THREE NAMES, IN THE OWNER'S ORDER, WITH THE OWNER'S DEPTHS.
static_assert(kLatticeLadder[0].mode == CellMode::Int8 && kLatticeLadder[0].depth == 0,
              "F1202: the START is `int8`, the owner's (i) and the engine's default.");
static_assert(kLatticeLadder[1].bits == TierBits::B4 && kLatticeLadder[2].bits == TierBits::B3 &&
                  kLatticeLadder[3].bits == TierBits::B2,
              "F1202: THE ORDER IS 4bit, 3bit, 2bit -- the owner's 「4bit 3bit 2bit」.");
static_assert(ladder_depth(CellMode::Rk4v4) == 1 && ladder_depth(CellMode::Rk3v4) == 2 &&
                  ladder_depth(CellMode::E8_2bit) == 3,
              "F1202: the three tiers are at depths 1, 2 and 3 from the start, so EACH IS "
              "REACHABLE and each needs exactly that many one-rung steps. This is the code-level "
              "reachability statement: no tier is a dead enum value.");
static_assert(ladder_depth(CellMode::Bf16) == -1 && ladder_depth(CellMode::Nvfp4) == -1,
              "F1202: `bf16` and `nvfp4` are NOT on the 4/3/2 lattice -- -1, not 0. They are "
              "reachable rungs of the eight-rung table and neither is a lattice tier.");

// THE LADDER'S PRICES ARE THE RUNG TABLE'S, READ THROUGH IT, AND STRICTLY DECREASING.
static_assert(ladder_plane_bytes(CellMode::Int8) > ladder_plane_bytes(CellMode::Rk4v4) &&
                  ladder_plane_bytes(CellMode::Rk4v4) > ladder_plane_bytes(CellMode::Rk3v4) &&
                  ladder_plane_bytes(CellMode::Rk3v4) > ladder_plane_bytes(CellMode::E8_2bit),
              "F1202: the ladder's plane column must be strictly decreasing, or a rung has no "
              "edge and the walk stops above the floor. This restates the rung table's own chain "
              "assert restricted to the four lattice cells.");
static_assert(ladder_plane_bytes(CellMode::Int8) == 16896 &&
                  ladder_plane_bytes(CellMode::E8_2bit) == 4608,
              "F1202: the runway from the start to the floor is 16,896 - 4,608 = 12,288 B/plane, "
              "the number blob_F1192 34.5 derives. Not re-typed here -- these two asserts pin that "
              "`ladder_plane_bytes` really is reading `cell_rung().plane_bytes`.");

// THE REACHABILITY ITSELF, EXECUTED BY THE COMPILER. This is the strongest form the claim can
// take without the follow-up window's edit: the step function is `constexpr`, so a `constexpr`
// chain of it IS the walk, evaluated at compile time, allocation-free. A `static_assert` on the
// result is a statement that the compiler walked the ladder -- not a comment saying it could.
//
// It is deliberately NOT written with `std::vector` / `ladder_level_walk`: `std::vector<bool>` is
// not a constexpr-friendly container, and a proof that only holds in a hand-written dialect of the
// walk would prove nothing about the walk. The step chain IS the walk's inner statement.
[[nodiscard]] constexpr std::uint32_t ladder_chain_steps(CellMode from) noexcept {
    CellMode mode = from;
    std::uint32_t steps = 0;
    for (std::uint32_t i = 0; i < 8; ++i) {
        const CellStep step = cell_next_tier(cell_all_modes(), mode);
        if (!step.legal()) { break; }
        mode = step.to;
        steps += 1;
    }
    return steps;
}

// THE TOTAL RUNWAY THE DESCENT CAN SPEND FROM THE START, IN PLANE BYTES PER LAYER.
[[nodiscard]] constexpr std::int32_t ladder_runway_bytes(CellMode from) noexcept {
    CellMode mode = from;
    std::int32_t total = 0;
    for (std::uint32_t i = 0; i < 8; ++i) {
        const CellStep step = cell_next_tier(cell_all_modes(), mode);
        if (!step.legal()) { break; }
        mode = step.to;
        total += step.bytes_saved;
    }
    return total;
}

// ⭐⭐ THE OWNER'S ACCEPTANCE ITEM, AS A COMPILE-TIME FACT: three one-rung steps from the start
// reach the floor, i.e. THE 4BIT, 3BIT AND 2BIT TIERS ARE EACH REACHABLE AND THE LADDER IS
// CONNECTED. `3` is not a count of rungs -- it is the number of steps the walk must be ABLE to
// take, and on the pre-image tree the same chain returns 0 (the first step is refused).
static_assert(ladder_chain_steps(CellMode::Int8) == 3,
              "F1202: THE THREE RUNGS. From the START, `cell_next_tier` must take exactly THREE "
              "legal one-rung steps -- int8 -> rk4v4 -> rk3v4 -> e8-2bit -- and then stop at the "
              "floor. Anything other than 3 means a tier is unreachable (a dead enum value, the "
              "failure mode the owner named) or the ladder has a rung the owner did not ask for.");
static_assert(ladder_chain_steps(CellMode::Bf16) == 4,
              "F1202: a cell that starts at `bf16` takes FOUR steps -- bf16 -> int8 (15,872) and "
              "then the whole lattice below it (int8 -> 4bit -> 3bit -> 2bit). CORRECTED BY THE "
              "GATE: the first draft of this assert said 1, on the reasoning that `int8` is the "
              "ladder's TOP RUNG. It is the top rung and it is still a rung: the chain continues "
              "below it, which is the whole point of the ladder. The compiler refused the wrong "
              "number (g++: 'the comparison reduces to (4 == 1)'), which is what a reachability "
              "claim asserted in code buys over one asserted in prose. `bf16` is above the ladder, "
              "not on it -- depth -1 -- and this is the only place the 15,872 B bf16 -> int8 step "
              "can appear.");
static_assert(ladder_chain_steps(CellMode::Rk4v4) == 2 && ladder_chain_steps(CellMode::Rk3v4) == 1,
              "F1202: from the 4bit rung there are exactly TWO steps left and from the 3bit rung "
              "exactly ONE -- so no tier is a dead end and none has a rung the owner did not ask "
              "for.");
static_assert(ladder_chain_steps(CellMode::E8_2bit) == 0,
              "F1202: the floor takes no steps; `AlreadyAtFloor` is the named answer.");
static_assert(ladder_runway_bytes(CellMode::Int8) == 12288,
              "F1202: int8 -> floor is 16,896 - 4,608 = 12,288 B/plane, the runway blob_F1192 "
              "34.5 derives ('those 12,288 B are PLANE bytes'). This is the total currency the "
              "whole three-rung descent can spend, and it is what the owner's 「多少得看预算约束」 "
              "is measured against.");
static_assert(ladder_runway_bytes(CellMode::Bf16) == 15872 + 12288,
              "F1202: a bf16 cell's runway is the largest step (bf16 -> int8 = 15,872) plus the "
              "lattice's 12,288 = 28,160 B/plane. Since the shipped START is all-int8 no cell "
              "starts here, which is why the 15,872 step is off the path -- blob_F1192 34.5.");

static_assert(ladder_step_total_delta(cell_next_tier(cell_all_modes(), CellMode::Int8), 4) ==
                  8192 * 8,
              "F1202 DEVELOPMENT-3: one int8 -> 4bit step gives the total back 8,192 B/plane x "
              "2 planes x 4 kv_heads = 65,536 B, NOT 8,192. The live driver subtracts 8,192.");
static_assert(ladder_step_total_delta(cell_next_tier(cell_all_modes(), CellMode::Int8), 1) == 16384,
              "F1202: the factor is exactly `2 * kv_heads`; at 1 head the step is 2 x 8,192.");
// THE SAME CHAIN ON THE CHEAPER-BELOW STEP FUNCTION, AS A COMPILE-TIME READING OF WHAT F1209
// CHANGED -- AND THE PRE-IMAGE NUMBERS ARE KEPT HERE AS HISTORY RATHER THAN AS AN ASSERT THAT WOULD
// NOW FIRE. On `kv_cell_modes.h` `b2a77dd4` (the charge-cut revision) this assert read, verbatim:
//     static_assert(cell_next_cheaper(cell_all_modes(), CellMode::Int8).to == CellMode::Nvfp4 &&
//                       cell_next_cheaper(cell_all_modes(), CellMode::Int8).bytes_saved == -400,
//                   "F1202: THE PRE-IMAGE STEP, PINNED. ... a currency of 9,232 - 9,632 = -400 ...")
// -- `nvfp4` with a RECORD currency of -400, refused by the driver's `bytes_saved <= 0` guard: the
// walk that moved nothing. F1209 closed defect (2) by moving that delta onto the plane ruler, so the
// SECOND conjunct is now +7,680 (= 16,896 - 9,216) and the assert had to change with it -- a pin is
// only worth anything if it is re-derived when the pinned thing moves, which is what the assert's
// own message told a later round to do. It is replaced by the two readings below, NOT deleted.
static_assert(cell_next_cheaper(cell_all_modes(), CellMode::Int8).to == CellMode::Nvfp4,
              "F1202/F1209: `nvfp4` is still the NEAREST rung below `int8` on the plane ruler, so "
              "the cheaper-below chain still leaves `int8` through a cold-slot EXPANSION (9,632 > "
              "9,232). If this fires, site 2 -- the order -- changed, and the paragraph at the top "
              "of this file must be re-derived rather than deleted.");
static_assert(cell_next_cheaper(cell_all_modes(), CellMode::Int8).bytes_saved ==
                  cell_rung(CellMode::Int8).plane_bytes - cell_rung(CellMode::Nvfp4).plane_bytes,
              "F1209: THE CURRENCY FIX, PINNED ON THE OTHER CHAIN TOO. `cell_next_cheaper`'s delta "
              "is now a PLANE delta (16,896 - 9,216 = 7,680), the ruler the charge sums, so the "
              "step the walk takes out of the start is ACCEPTED instead of refused. The pre-image "
              "number was -400. If this fires, site 3 moved again and the derivation at "
              "`kv_block_descent.h`'s subtraction is stale -- re-measure it, do not delete it.");

} // namespace ninfer::product
