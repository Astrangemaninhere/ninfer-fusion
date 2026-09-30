#pragma once

// F1172 -- THE DESCENT: START EVERY CELL AT ITS DEFAULT RUNG, WALK ONE RUNG AT A TIME IN ORDER
// OF "THE LATER FREQUENCY", AND STOP WHEN THE TOTAL FITS THE BUDGET.
//
// THE ORDER, WORD FOR WORD (its own English paraphrase, kept as the order gave it):
//   * "the different formats inside a cell -- a cell has many layers, right? -- DEMOTE ALONG THE
//      LADDER ACCORDING TO THE BUDGET. Layers used a lot LATER inside the cell keep int8; the
//      less-used ones are slowly tapered down, and in the END they go to e8 2bit. It is DEMOTED
//      ONE RUNG AT A TIME. How much of it actually reaches e8 2bit DEPENDS ON THE BUDGET
//      CONSTRAINT."
//
// THE FIVE REQUIREMENTS, AND WHERE EACH ONE LANDS IN THIS FILE. There are five and there is no
// sixth, because the design fixed them (blob_F1168.md section A.1) and because each one is a
// separately falsifiable sentence about the code below:
//
//   1. START      -- every cell at its DEFAULT rung (`cell_default_tier`, kv_cell_modes.h):
//                    "at the START they all default to int8 or bf16".   -> step A below.
//   2. SORT KEY   -- "the later frequency", at `(block, layer)` granularity: the order's words
//                    are "the LAYERS used a lot later inside the cell", and the tree's one
//                    block-granular quantity (`sum_dir_verdict_recalls_bytes`, one bit) is NOT a
//                    substitute -- using it would collapse every layer of a block onto one
//                    value and degenerate the policy to the block-uniform one the order's first
//                    sentence forbids.                        -> step B below, and GAP-FREQ.
//   3. STEP COST  -- one cell, one rung: the gain is the difference of that cell's two PLANE bytes
//                    (`cell_next_cheaper`, kv_cell_modes.h), SCALED BY `2 * kv_heads` to reach the
//                    total's unit (`ladder_step_total_delta`, kv_tier_ladder.h) -> step C below.
//                    [F1209 2026-09-29] THIS REQUIREMENT'S ORIGINAL WORDS ARE SUPERSEDED AND KEPT:
//                    "the gain is the difference of that cell's two record bytes". TWO things were
//                    wrong with that sentence and both are fixed below, not deleted: the gain is a
//                    PLANE difference (site 3 of `cell_next_cheaper` was still on the record ruler
//                    while the charge summed planes -- the SPLIT RULER, MEASURED as
//                    `steps=0 refused=cells` on every budget), and the `2 * kv_heads` factor was
//                    missing from the subtraction (DEVELOPMENT-3, derived at the subtraction site).
//   4. STOP       -- walk until `sum_b bytes(V(b)) <= budget`; how MUCH reached the floor is an
//                    OUTPUT, not an input.                      -> step D below.
//   5. TERMINATION-- every accepted step strictly DECREASES the total by >= 1 byte and every
//                    cell has finitely many rungs, so steps <= #cells * (#rungs - 1); PLUS a
//                    hard `max_steps` so a future rung with a non-positive delta cannot hang
//                    this loop. A step with a zero or negative delta is REFUSED BY NAME
//                    (`StepRefusal::NoByteCurrency`) and is NOT taken.  -> step E below.
//
// G14 -- THE TENSION, AT A NAMED PLACE, BOTH SIDES QUOTED, NEITHER SIDE HIDDEN.
// `kv_block_budget_stage.h` states its own rule twice and verbatim: "THE RULE IS A LOOKUP, NEVER
// A SOLVER." and "There is no search, no per-token work, and no re-solve per block". THIS FILE
// IS A SORT AND A WALK: `O(#cells log #cells)` for the order plus `O(#cells * #rungs)` for the
// sweeps, with `#cells = blocks * layers` and THE BLOCK COUNT UNBOUNDED AT RUNTIME (that header
// says so itself, verbatim: "A BLOCK is not: it is created at RUNTIME from arriving text, its
// number is unbounded and unknown when the fit runs"). Both statements are true and they are not
// compatible. The resolution CANNOT BE TAKEN HERE -- it is a decision for the owner/main -- and
// the two candidate shapes are named in `block_descent_plan`'s POPULATION note rather than
// silently picked. What this file does instead is make the cost EXPLICIT AND MEASURABLE: the
// plan reports `cells_total`, `steps` and `sweeps`, so a run can be read against any window
// bound a future decision imposes.
//
// GAP-FREQ -- THE SORT KEY HAS NO PRODUCER, AND THAT IS A STATE OF THE INPUT, NOT A TODO.
// The frequency axis is a TYPE here (`CellFrequencySource`) with a NAMED absent state
// (`CellFrequencyKind::Absent`), not a zero. When the source is absent the walk still runs, the
// order degenerates to the tie-break, and THE PLAN SAYS SO in `drive` and in `describe()`.
// The prohibition the tree states is respected BY THE SHAPE OF THE TYPE: there is no field on
// `CellFrequencySource` that can hold an age, a timestamp, an LRU rank, a distance, a
// similarity, or a threshold, so none can arrive through it. (The seam's own comment, verbatim:
// "not an age, not an LRU rank".)

// THE STANDARD HEADERS COME FIRST, AND TWO OF THEM ARE LOAD-BEARING: `<cstdlib>` declares
// `std::getenv` (read by this file's chain reader below, and by the ladder's ceiling reader) and
// `<cstring>` declares `std::strcmp`. They are listed EXPLICITLY because `kv_tier_ladder.h` is now
// included from here: a header included by its own user must not depend on the user's transitive
// includes happening to declare the names it uses.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "product/kv_cell_modes.h"
// [F1209 2026-09-29] THE LADDER, AND WHY THIS INCLUDE IS HERE RATHER THAN LATER. The driver now
// SELECTS its step function from this header, so the include cannot be deferred past the call site
// (a function body in this file calls `cell_next_tier` and `ladder_step_total_delta` below). It sits
// directly after `kv_cell_modes.h` because `kv_tier_ladder.h` is built on that header ALONE -- F1209
// removed its back-include of THIS file to break the cycle -- so the two are a DAG and either
// include order now compiles. Verified in both orders by the syntax gate, not assumed.
#include "product/kv_tier_ladder.h"
// [F1231] THE LAW'S CARRIER, INCLUDED HERE AND NOT THE OTHER WAY ROUND. `kv_descent_control.h`
// includes `kv_tier_ladder.h` and nothing else, so this edge is a DAG and compiles in either
// order -- but the DIRECTION is load-bearing and is stated at that file's top: the engine's charge
// table (`kv_block_budget_stage.h`) consumes a PLAN, so the plan must be the walker, and a control
// header that could call the plan would need this file back and close a cycle.
#include "product/kv_descent_control.h"

// THE FEATURE MACRO THE TWO-SIDED ARM READS. It exists so ONE arm source compiles against the
// pre-fix headers AND against these: an arm that had to guess a field name would be testing its own
// guess, and an arm that could not tell the two trees apart could not show a rule FAILING first.
// Defined AFTER the include so plain preprocessing is enough (no `__has_include`).
#define NINFER_KV_DESCENT_LAW_F1231 1

// [F1260 kvplanar] THE SAME DISCIPLINE FOR THIS LANDING, AND FOR THE SAME REASON: the arm that has
// to show an ASYMMETRIC CELL must compile against the pre-fix headers too (to produce the "before"
// half of the same-arm comparison, where the cell CANNOT be asymmetric), and it can only name
// `DescentStepOrder::LatticeK_FamilyV` where the enumerator exists. A source guarded by this macro
// is one source with two readings.
#define NINFER_KV_CELL_PAIR_F1260 1

namespace ninfer::product {

// ---------------------------------------------------------------------------------------------
// A. THE FREQUENCY AXIS AS A TYPE -- AND THE HOLE, AS A NAMED STATE
// ---------------------------------------------------------------------------------------------
//
// GAP-FREQ, READ IN SOURCE (blob_F1168.md section 7, and re-read by this line in the same
// mirrors): the `spec/` headers have ZERO hits for `frequency` / `lookback` / `re-reference` /
// `recently referenced` / `reuse count`; `SumDirAdmissibility` is a seven-value enum of
// position/structure facts with no frequency rung; `sum_dir_facts_from_sequence` takes ONE
// scalar (`std::uint32_t token_count`). THERE IS NO PRODUCER.
//
// SO THE HONEST TYPE IS ONE WHOSE "WE HAVE NOTHING" BRANCH IS A VALUE. `Absent` is not zero and
// not an error: it is the statement "no producer exists in this build", and it changes what the
// plan's OUTPUT MEANS. A caller cannot get a driven plan by forgetting to pass anything, and a
// reader cannot mistake an undriven plan for a driven one, because `drive` is printed.
enum class CellFrequencyKind : std::uint8_t {
    Absent  = 0,  // NO PRODUCER EXISTS (the state of this tree today)
    Measured = 1, // a producer exists and counts revisits
};

// THE ONLY QUANTITY THAT MAY ENTER. A COUNT of how many times this cell was READ BACK after it
// went cold -- a non-negative integer, monotone in "used later".
//
// WHY IT MAY DRIVE AN ALLOCATION AND MAY NOT PRODUCE A VERDICT (this is the tree's own law, not
// an interpretation): `sum_dir_determined.h` says, verbatim, "A similarity score, a distance, a
// cosine, a top-k rank or any threshold may NOT produce `DETERMINED` ... there is no parameter a
// score could arrive through." A revisit COUNT is not a score, and this type reaches only the
// charge side of the stage -- which sits AFTER `offered` and can only REFUSE. So the count can
// move a byte total and cannot move a judgement. The same defence is written here as a shape:
// there is nothing in this struct a distance or a threshold could be stored in.
struct CellFrequencySource {
    CellFrequencyKind kind = CellFrequencyKind::Absent;
    // revisit_count(self, block, layer). MAY NOT be null when kind == Measured.
    std::uint32_t (*revisit_count)(const void* self, std::uint32_t block,
                                   std::uint32_t layer) = nullptr;
    const void* self = nullptr;

    [[nodiscard]] bool usable() const noexcept {
        return kind == CellFrequencyKind::Measured && revisit_count != nullptr;
    }
};

[[nodiscard]] inline CellFrequencySource cell_frequency_absent() noexcept {
    return CellFrequencySource{};
}

// ---------------------------------------------------------------------------------------------
// B. WHY A PLAN AT ALL, AND THE POPULATION QUESTION (GAP-POPULATION), NAMED
// ---------------------------------------------------------------------------------------------
//
// The stage consults the charge ONCE PER BLOCK, during the page loop, and it must know each
// block's charge AT THAT MOMENT. So something has to answer "what is V(b), and what does it
// cost" before the loop reaches b. That something is this plan. It is built ONCE, it is
// read-only during the loop, and it is addressed by block.
//
// GAP-POPULATION -- THE ONE INPUT THE PASS CANNOT SUPPLY. The descent needs its POPULATION (the
// cells it is allocating over) BEFORE the walk, but the population is decided DURING the pass:
// a page is only admitted if `store.can_cold_transfer(text, page)` holds AND the semantic
// directory offers it, and neither is known at plan time. Two shapes are representable:
//
//   * CandidateWindow (implemented): plan over every page the pass MAY retire. CONSERVATIVE --
//     it demotes cells whose block will never be offered, so it gives up int8 coverage it did
//     not have to give up. That is a real loss and it is EXACTLY the loss the order's mechanism
//     exists to avoid, so it is REPORTED (`planned_cells` vs the eventual `offered` count in the
//     stage's own line) and it is a decision for the owner/main, not a silent default.
//   * OfferedPopulation (NOT implemented): plan over the pages the directory will actually
//     offer. Requires the plan to be built after the offering decisions, i.e. a second pass over
//     the page range, which the seam's one-pass shape does not have today.
//
// The third possibility -- making the descent per-block and O(1) so it can live INSIDE the loop
// -- is not the order's algorithm: "the less-used ones are slowly tapered down" compares cells
// ACROSS blocks, and a per-block rule cannot see another block's cells. It is therefore named
// here and not implemented, rather than implemented and mislabelled.
enum class DescentPopulation : std::uint8_t {
    CandidateWindow  = 0,  // every page this pass MAY retire (conservative; the implementable one)
    OfferedPopulation = 1, // not implementable before the pass; kept as a NAME, not a stub
};

// WHY THE WALK STOPPED, AS A VALUE. A plan that stopped because the budget was met is a
// different object from one that stopped because no cell could move, and the difference is the
// whole finding of this line (see `blob_F1172.md` section 4).
enum class DescentDrive : std::uint8_t {
    BudgetOff         = 0,  // `budget_bytes <= 0`: OFF, and OFF is the default vector + pre-image
    StoppedByBudget   = 1,  // the total fits: the order's stop condition, reached
    NoMovableCell     = 2,  // no cell could take a legal step: see the refusal histogram
    StepBoundReached  = 3,  // the hard `max_steps` guard fired (only reachable with a future rung)
    UnpricedCell      = 4,  // at least one cell's DEFAULT rung has no byte cost: no byte total
    // [F1231] ⚠ THE TWO VALUES A ONE-SHOT WALK NEEDS, AND WHY FOLDING THEM INTO `NoMovableCell`
    // WOULD BE A LIE. Under `OneShot` the walk stops after ONE step whether or not the budget is
    // met, so `total > budget` is the NORMAL state of a trigger and says nothing about whether any
    // cell could still move. A reader (the F1217 table, or any census) that saw `no-movable-cell`
    // there would read "the ladder is exhausted" while the ladder still had every rung left.
    OneShotStepTaken  = 5,  // one demotion taken, the trigger is spent, the budget is NOT met yet
    OneShotNothingMovable = 6,  // the trigger found nothing admissible to move (floor, or the gate)
    // [F1240 kvsolve] ⚠ THE REFUSAL THAT MUST HAVE A NAME OF ITS OWN. A stateless SOLVE has one
    // constraint in the quality axis (a measured precision floor) and one requirement in the speed
    // axis (the byte budget), and the two can CONTRADICT: the budget may demand more demotion than
    // the floor permits. Neither existing value may be used for that state -- `StoppedByBudget`
    // would claim the budget was met, and `NoMovableCell` would claim the ladder ran out, and BOTH
    // would be false while the truth is "the caller's two constraints are incompatible". Folding it
    // into either is exactly the silent compromise 「不能放任，否则精度可能无法确定」 forbids, so it
    // is its own value and the solver prints the DEFICIT beside it.
    RefusedByFloor = 7,  // the measured precision floor refuses what the byte budget requires
};

[[nodiscard]] constexpr const char* descent_drive_name(DescentDrive drive) noexcept {
    switch (drive) {
    case DescentDrive::BudgetOff: return "budget-off";
    case DescentDrive::StoppedByBudget: return "stopped-by-budget";
    case DescentDrive::NoMovableCell: return "no-movable-cell";
    case DescentDrive::StepBoundReached: return "step-bound-reached";
    case DescentDrive::UnpricedCell: return "unpriced-cell";
    case DescentDrive::OneShotStepTaken: return "one-shot-step-taken(budget-not-yet-met)";
    // [F1240 kvsolve] The infeasible case, named. See the enumerator's own note: this value is the
    // ONLY one that means "the two constraints contradict", and the deficit is in the plan's
    // `alloc_note` rather than folded into a byte column.
    case DescentDrive::RefusedByFloor: return "REFUSED-by-precision-floor(budget-unreachable)";
    case DescentDrive::OneShotNothingMovable: break;
    }
    return "one-shot-nothing-movable";
}

// ---------------------------------------------------------------------------------------------
// B2. WHICH RUNG ORDER, AND HOW DEEP -- THE TWO KNOBS, AND WHY BOTH ARE KNOBS. [F1209 2026-09-29]
// ---------------------------------------------------------------------------------------------
//
// ⚠ THE TWO CHAINS ARE NOT INTERCHANGEABLE AND THIS FILE DOES NOT PICK A WINNER FOR THE OWNER.
// There are two code-level answers to "the next rung below me" in this tree, and they are different
// LADDERS:
//
//   * `PriciestBelow` -- `cell_next_cheaper` (kv_cell_modes.h), the PRE-IMAGE order and the default
//     here. From `int8` it answers `nvfp4` (plane 9,216, the priciest rung strictly below), whose
//     COLD RECORD is 9,632 > 9,232 -- a step that EXPANDS the cell's cold slot -- and the chain it
//     walks is int8 -> nvfp4 -> rk4v4 -> rk3v4 -> e8-2bit, FOUR steps, all three tiers reached.
//   * `Lattice432` -- `cell_next_tier` (kv_tier_ladder.h), the owner's 4/3/2 ladder. It excludes
//     every rung whose record is WIDER than the cell's (half one) and is therefore
//     int8 -> rk4v4 -> rk3v4 -> e8-2bit, THREE steps, and it never enters `nvfp4`.
//
// BOTH reach the floor; both are one rung at a time; they differ in the number of steps and in
// whether the walk passes through a cold-slot expansion. `blob_F1202.md`'s claim that the cheaper
// order leaves the 4-bit and 3-bit rungs DEAD is FALSE (site 1 discards every rung that is not
// strictly plane-cheaper and site 2 keeps the largest of the survivors -- the rung NEAREST below, so
// the chain is traversable; the correction is recorded in `kv_tier_ladder.h`). So what was missing
// was never a rung: it was the CURRENCY, and it is fixed at the subtraction below.
//
// WHY A KNOB AND NOT A REPLACEMENT -- THE OWNER'S OWN STANDING RULE, QUOTED BY `blob_F1202.md`:
// "the table must compare in several dimensions and be produced automatically for a decision rather
// than hard-coded, otherwise it may not apply to other cards". A hard-coded single chain IS the
// failure mode that sentence names, in either direction. So the ORDER of the pre-image is the
// default (`PriciestBelow`, so an unset environment measures the same chain tonight's runs were
// built on), the owner's 4/3/2 ladder is one environment variable away, and the sweep's table is
// what compares them. THE DEFAULT IS NAMED AS A CHOICE, NOT HIDDEN AS A FACT: flipping it is the
// one-token change of this enum's default below.
enum class DescentStepOrder : std::uint8_t {
    PriciestBelow = 0,  // `cell_next_cheaper`: the pre-image order, nvfp4 first (DEFAULT)
    Lattice432    = 1,  // `cell_next_tier`: the owner's 4bit/3bit/2bit lattice, no expansion step
    // =============================================================================================
    // [F1260 kvplanar] ⭐ THE TWO-SIDED SELECTIONS, APPENDED AT VALUES ABOVE EVERY PRE-EXISTING ONE.
    //
    // WHY THE SELECTION IS A VALUE OF *THIS* ENUM AND NOT A NEW FIELD. A plan is a function of its
    // spec, and this enum IS the spec's carrier for "which rung is below me"; the caller already
    // resolves it from the environment through its own reader (`descent_step_order_from_env`, this
    // file). So a TWO-PLANE selection needs no new field, no new environment name and no caller
    // edit -- it needs this enum to be able to SAY it, which is what these four values are.
    //
    // WHY THESE FOUR AND NOT THIRTY-SIX. A pair is admissible only if the ENGINE can build it
    // (`cell_pair_realizable`, kv_cell_modes.h -- the typed mirror of
    // `kv_e8_plane_pair_realizable`). That gate is strict on purpose: the engine builds SYMMETRIC
    // planes for `int8`/`bf16`/`nvfp4` and the seven listed pairs for the e8 family, and nothing
    // else. So most K/V order combinations are refused by construction, and the selection that is
    // both ASYMMETRIC and BUILDABLE is the deployed family's own: **K on the 4/3/2 ladder, V on the
    // family's V plane** -- the `rk4v4 -> rk3v4 -> rk2v4` chain verbatim ("only the K code width
    // moves", `kv_kv_bits.h:279`). The other three are kept because they are the mirror and the mixed
    // cases, and because a selection the gate refuses is a READING (`refused=cells`,
    // `saved_bytes=0`) rather than a silent substitution.
    LatticeK_FamilyV   = 2,  // K: the 4/3/2 lattice. V: the family's own V plane, then held.
    PriciestK_FamilyV  = 3,  // K: priciest-below. V: the family's own V plane, then held.
    LatticeK_PriciestV = 4,  // K: the 4/3/2 lattice. V: priciest-below.
    PriciestK_LatticeV = 5,  // K: priciest-below. V: the 4/3/2 lattice.
};

// ⭐ THE PER-PLANE ORDER, WHICH IS WHAT THE VALUES ABOVE RESOLVE TO. `FamilyV` is not a ladder: it
// is the DEPLOYED FAMILY'S OWN RULE stated as an order -- "the family's V plane, and it does not
// narrow" (`kv_kv_bits.h:279`, `kv_bit_budget.h:301-312`). Once a cell's V plane IS the family's
// plane this order HOLDs it, and that hold is exactly what makes the family chain's second and third
// steps produce the ASYMMETRIC pairs `(B3,B4)` and `(B2,B4)`.
enum class PlaneOrder : std::uint8_t {
    PriciestBelow = 0,
    Lattice432    = 1,
    FamilyV       = 2,
};

struct PlaneOrders {
    PlaneOrder k = PlaneOrder::PriciestBelow;
    PlaneOrder v = PlaneOrder::PriciestBelow;
};

[[nodiscard]] constexpr const char* plane_order_name(PlaneOrder order) noexcept {
    switch (order) {
    case PlaneOrder::Lattice432: return "lattice-4/3/2";
    case PlaneOrder::FamilyV: return "family-v-plane-then-hold";
    case PlaneOrder::PriciestBelow: break;
    }
    return "priciest-below";
}

// THE ONE PLACE A SELECTION BECOMES TWO ORDERS. `PriciestBelow` and `Lattice432` map to BOTH SIDES
// EQUAL -- which is what makes every pre-existing arm's step a DIAGONAL step and therefore, by
// `cell_pair_diagonal_is_the_preimage_charge`, the same step it took before this landing.
[[nodiscard]] constexpr PlaneOrders plane_orders_of(DescentStepOrder order) noexcept {
    switch (order) {
    case DescentStepOrder::Lattice432:
        return PlaneOrders{PlaneOrder::Lattice432, PlaneOrder::Lattice432};
    case DescentStepOrder::LatticeK_FamilyV:
        return PlaneOrders{PlaneOrder::Lattice432, PlaneOrder::FamilyV};
    case DescentStepOrder::PriciestK_FamilyV:
        return PlaneOrders{PlaneOrder::PriciestBelow, PlaneOrder::FamilyV};
    case DescentStepOrder::LatticeK_PriciestV:
        return PlaneOrders{PlaneOrder::Lattice432, PlaneOrder::PriciestBelow};
    case DescentStepOrder::PriciestK_LatticeV:
        return PlaneOrders{PlaneOrder::PriciestBelow, PlaneOrder::Lattice432};
    case DescentStepOrder::PriciestBelow: break;
    }
    return PlaneOrders{PlaneOrder::PriciestBelow, PlaneOrder::PriciestBelow};
}

[[nodiscard]] constexpr const char* descent_step_order_name(DescentStepOrder order) noexcept {
    switch (order) {
    case DescentStepOrder::PriciestBelow: return "priciest-below(4-step,nvfp4-first)";
    case DescentStepOrder::Lattice432: return "lattice-4/3/2(3-step,no-nvfp4)";
    case DescentStepOrder::LatticeK_FamilyV:
        return "k:lattice-4/3/2,v:family-plane(ASYMMETRIC: the deployed rk4v4/rk3v4/rk2v4 chain)";
    case DescentStepOrder::PriciestK_FamilyV:
        return "k:priciest-below,v:family-plane(ASYMMETRIC)";
    case DescentStepOrder::LatticeK_PriciestV:
        return "k:lattice-4/3/2,v:priciest-below(ASYMMETRIC)";
    case DescentStepOrder::PriciestK_LatticeV:
        return "k:priciest-below,v:lattice-4/3/2(ASYMMETRIC)";
    }
    return "priciest-below(4-step,nvfp4-first)";
}

// THE ENV SPELLING, AND WHY A READER LIVES IN THIS HEADER. The sweep's second axis has to be able to
// select the chain per arm, and the ladder's own `NINFER_KV_DESCENT_MAX_TIER` reader set the
// precedent for a knob owned by a header whose caller cannot be edited per arm. The READ is made by
// the LIVE CALLER (`program_impl.h`), not by `block_descent_plan`: this header keeps its rule that a
// plan cannot be driven by an environment it reads behind the caller's back -- the same reason the
// budget arrives as a field. A value that is not `lattice` (or is unset, or is empty) is
// `PriciestBelow`, and the default is spelled out rather than left to chance.
// =============================================================================================
// [F1260 kvplanar] ⚠ THE ONE GRAMMAR THIS LINE WIDENS, AND WHAT IS NOT NEW ABOUT IT.
//
// NOT NEW: the name, the reader's position, the caller, the field it fills, and the meaning of unset
// and of `lattice`. NEW: the variable can now name the two planes SEPARATELY, because the cell has
// two -- `NINFER_KV_DESCENT_CHAIN=k:lattice,v:familyv` is the deployed family's own chain (K
// narrows, V keeps the family's plane) and is the one spelling that produces ASYMMETRIC cells.
// A side token is one of `priciest` / `lattice` / `familyv`; ANYTHING ELSE -- an unset variable, an
// empty value, a misspelled side, a missing `,v:` -- is `PriciestBelow` on BOTH planes, i.e. the
// pre-image, which is the same rule this reader has always applied to a value it does not recognise.
// =============================================================================================
inline constexpr const char* kKvDescentChainEnv = "NINFER_KV_DESCENT_CHAIN";

[[nodiscard]] inline bool descent_chain_token_is(const char* text, std::size_t begin,
                                                 std::size_t end, const char* word) noexcept {
    const std::size_t n = std::strlen(word);
    if (end < begin || (end - begin) != n) { return false; }
    for (std::size_t i = 0; i < n; ++i) {
        if (text[begin + i] != word[i]) { return false; }
    }
    return true;
}

// A SIDE TOKEN AS A VALUE, WITH A `bool` FOR "IT WAS A TOKEN AT ALL" -- so an unrecognised side
// cannot be mistaken for a recognition whose value happens to be the default.
[[nodiscard]] inline bool descent_plane_order_from_token(const char* text, std::size_t begin,
                                                         std::size_t end,
                                                         PlaneOrder& out) noexcept {
    if (descent_chain_token_is(text, begin, end, "priciest")) {
        out = PlaneOrder::PriciestBelow; return true;
    }
    if (descent_chain_token_is(text, begin, end, "lattice")) {
        out = PlaneOrder::Lattice432; return true;
    }
    if (descent_chain_token_is(text, begin, end, "familyv")) {
        out = PlaneOrder::FamilyV; return true;
    }
    return false;
}

[[nodiscard]] inline DescentStepOrder descent_step_order_from_env() noexcept {
    const char* const text = std::getenv(kKvDescentChainEnv);
    if (text == nullptr || *text == '\0') { return DescentStepOrder::PriciestBelow; }
    if (std::strcmp(text, "lattice") == 0) { return DescentStepOrder::Lattice432; }
    if (std::strcmp(text, "priciest") == 0) { return DescentStepOrder::PriciestBelow; }
    if (std::strncmp(text, "k:", 2) != 0) { return DescentStepOrder::PriciestBelow; }
    // k:<s>,v:<t> -- the comma and the `v:` are REQUIRED, so a half-spelled two-sided value is a
    // refusal and not a guess about which plane the author meant.
    const std::size_t len = std::strlen(text);
    std::size_t split = len;
    for (std::size_t i = 2; i + 2 < len; ++i) {
        if (text[i] == ',' && text[i + 1] == 'v' && text[i + 2] == ':') { split = i; break; }
    }
    if (split == len) { return DescentStepOrder::PriciestBelow; }
    PlaneOrder k_side = PlaneOrder::PriciestBelow;
    PlaneOrder v_side = PlaneOrder::PriciestBelow;
    if (!descent_plane_order_from_token(text, 2, split, k_side)) {
        return DescentStepOrder::PriciestBelow;
    }
    if (!descent_plane_order_from_token(text, split + 3, len, v_side)) {
        return DescentStepOrder::PriciestBelow;
    }
    const PlaneOrders orders{k_side, v_side};
    if (orders.k == PlaneOrder::Lattice432 && orders.v == PlaneOrder::FamilyV) {
        return DescentStepOrder::LatticeK_FamilyV;
    }
    if (orders.k == PlaneOrder::PriciestBelow && orders.v == PlaneOrder::FamilyV) {
        return DescentStepOrder::PriciestK_FamilyV;
    }
    if (orders.k == PlaneOrder::Lattice432 && orders.v == PlaneOrder::PriciestBelow) {
        return DescentStepOrder::LatticeK_PriciestV;
    }
    if (orders.k == PlaneOrder::PriciestBelow && orders.v == PlaneOrder::Lattice432) {
        return DescentStepOrder::PriciestK_LatticeV;
    }
    // The remaining combinations ARE one of the two symmetric orders (both sides equal), and they
    // are returned as exactly that rather than as a fifth spelling of the same behaviour.
    if (orders.k == PlaneOrder::Lattice432) { return DescentStepOrder::Lattice432; }
    return DescentStepOrder::PriciestBelow;
}
static_assert(plane_orders_of(DescentStepOrder::PriciestBelow).k == PlaneOrder::PriciestBelow &&
                  plane_orders_of(DescentStepOrder::PriciestBelow).v == PlaneOrder::PriciestBelow &&
                  plane_orders_of(DescentStepOrder::Lattice432).k == PlaneOrder::Lattice432 &&
                  plane_orders_of(DescentStepOrder::Lattice432).v == PlaneOrder::Lattice432,
              "F1260: the TWO PRE-IMAGE SELECTIONS MUST RESOLVE TO EQUAL SIDES -- that is the whole "
              "of what makes an existing arm's step diagonal and therefore unchanged");

// THE RUNG CEILING, AS A PLAN INPUT RATHER THAN AS A HIDDEN READ. `ladder_ceiling_from_env()` (the
// ladder's own reader) is what the live caller uses to fill this; the walk applies it as a REFUSAL
// with a NAME, never as a silently smaller allowed set, so a census can tell "the ceiling stopped
// it" from "the ladder ran out" -- the same rule `ladder_level_walk` uses, deliberately, because the
// two walks must agree cell for cell on the lattice chain.
// `kDescentCeilingUnset` is not 0 and not the floor: it means "the caller supplied nothing", and it
// is resolved to the floor (the permissive default, identical to the ladder's own unset behaviour).
inline constexpr std::int32_t kDescentCeilingUnset = -1;

// ---------------------------------------------------------------------------------------------
// B3. [F1231] WHICH WALK -- AND WHY THE PRE-IMAGE ONE IS NO LONGER THE LIVE PATH.
// ---------------------------------------------------------------------------------------------
//
// WHAT THE LIVE PATH DID BEFORE THIS LINE, MEASURED BY THE ARM THAT SHIPS WITH IT (case
// `64k-mid`, the frontiers of `P3_B_64k_65536_r1.log`, budget 18,907,136 B):
//
//     sweeps=5 steps=3200 demoted=800 of cells=800 ... newest demoted page=191
//
// ONE call to this function moved NINETY-NINE POINT SIX PERCENT of the population (800 cells of
// 800, four rungs each, five sweeps) in a single pass, and its newest demoted page is 191 -- a page
// ABOVE the engine's own retirement bound of 190, i.e. a page the engine forbids to leave. THAT IS
// 「一直进行」 AND IT IS THE INVERSION IN ONE LINE OF OUTPUT: the whole population, newest included.
// (The engine calls this function ONCE PER PASS, so "one call" and "one trigger" are the same event
// -- the plan is built once at `program_impl.h`'s descent site and consumed by that pass.)
//
// SO THE LIVE SHAPE IS THE ONE-SHOT WALK, AND IT IS THE DEFAULT:
//   * `OneShot` -- the outer loop runs ONCE and takes AT MOST ONE accepted step. The cell that
//     moved is remembered (`next_offset`), the MODES are carried back to the caller's cursor, and
//     the next trigger resumes at the cell after it. When the scan reaches the end of the
//     population without a step, the offset wraps to 0: that is one LEVEL completed, and the next
//     level begins -- which is how "every cell moves one rung before any cell moves two" stays
//     true over many triggers instead of inside one.
//   * `Sweep`  -- the pre-image shape, kept REACHABLE AND NAMED, because the two-sided arm has to
//     be able to measure against it and because deleting a behaviour is how a regression becomes
//     unmeasurable. IT IS NOT THE LIVE PATH: the live caller (`program_impl.h`) passes `OneShot`.
//
// ⚠ THE ONE THING THE DEFAULT CHANGE DOES *NOT* TOUCH: the OFF path. `budget_bytes <= 0` still
// returns the default vector and the pre-image charge before any walk is entered, so "OFF is
// byte-for-byte the pre-image" is a theorem about a branch that no walk mode can reach.
enum class DescentWalk : std::uint8_t {
    OneShot = 0,  // ONE TRIGGER => ONE DEMOTION (the live path, and the zero value)
    Sweep   = 1,  // the pre-image whole-population sweep: every cell one rung, up to 6 sweeps
    // [F1240 kvsolve] THE STATELESS PER-CELL ALLOCATION SOLVE, WHICH IS NOT A WALK AT ALL.
    // Its carrier is `product/kv_cell_alloc_solve.h`, included at the BOTTOM of this file (it
    // consumes this header's types, so that include direction is the only one that is a DAG), and
    // the dispatch is `block_alloc_plan` in that header. WHY IT IS AN ENUMERATOR OF *THIS* ENUM
    // RATHER THAN A NEW SPEC FIELD: the choice must be a FUNCTION OF THE SPEC -- this file's own
    // rule, stated for `budget_bytes`, `step_order` and `ladder_ceiling_depth` -- and the caller
    // already fills `walk`. It is APPENDED with the value 2 so `OneShot` keeps the zero value,
    // i.e. a default-constructed spec still means the live path, and so no existing call site can
    // change behaviour by omission.
    // ⚠ IT IS NOT A `DescentWalk` IN THE SENSE THE OTHER TWO ARE -- it neither iterates nor
    // carries -- so the three readings that need a cursor (`triggers`, `steps_taken`, `law_holds`)
    // print as 0/0/yes and `carry=` prints `absent`. That is finding (B): a solve has no position,
    // so there is nothing to carry and nothing to bound.
    Solve   = 2,  // a fresh per-cell selection computed from a measurement table; NO cursor
};

[[nodiscard]] constexpr const char* descent_walk_name(DescentWalk walk) noexcept {
    switch (walk) {
    case DescentWalk::OneShot: return "one-shot(one-trigger-one-demotion)";
    // [F1240 kvsolve] The name says which SHAPE ran; `carry=absent` beside it says the shape has
    // no position to carry.
    case DescentWalk::Solve: return "solve(stateless-per-cell-allocation,F1240)";
    case DescentWalk::Sweep: break;
    }
    return "sweep(pre-image:whole-population)";
}

// ---------------------------------------------------------------------------------------------
// C. THE SPEC AND THE PLAN
// ---------------------------------------------------------------------------------------------
struct DescentSpec {
    std::int64_t budget_bytes = 0;      // THE STOP CONDITION. <= 0 means OFF (the order's default)
    std::uint32_t blocks = 0;           // the population's block count (the candidate window)
    std::uint32_t layers = kBlockVectorLayerAxis;
    std::int32_t kv_heads = 0;
    // [F1255 kvaxisA] THE STORAGE'S DECLARED NARROW CLASS, in pages per layer. 0 (the default) IS
    // THE PRE-IMAGE POOL: no layer carries a second plane set, and the axis3 columns are absent from
    // `describe()`. It arrives through the SPEC rather than being read inside `block_descent_plan`
    // for the same reason `budget_bytes` does -- a plan is a function of its spec, and the ONE
    // reader of the knob is `kv_axis3_narrow_pages_from_env`, called by the caller.
    std::uint32_t narrow_pages_per_layer = 0;
    // `F_(l,b)`. One set per LAYER (the tree's `F_l`), applied to every block of this plan. A
    // caller with a per-cell set supplies a finer array later; today nothing in the tree can
    // vary it by block (the layer's dtype is a property of the artifact, not of the page).
    //
    // ⚠ GAP-CELLSET -- [planes 2026-09-29] NAMED, BECAUSE A COMMENT IS NOT AN INVARIANT. The owner
    // ruled twice that block granularity is core ("块粒度是核心，必须深入到块来分配"; "梯度要记得
    // 按块，不能全层下降"), so the shape of this field is an acceptance item and must be stated as a
    // GAP rather than trusted to prose:
    //   * WHAT IS ALREADY PER-CELL, AND MEASURABLY SO: the ALLOCATION. The walk steps one cell at a
    //     time (`vector.set(cell.layer, step.to)` on the `(key, layer, block)`-ordered list below),
    //     so the mode of `(b, l)` is its own decision and moves on its own budget turn. The four
    //     counters `demoted_layers` / `demoted_blocks` / `demoted_max_per_layer` /
    //     `demoted_max_per_block` exist to make that a READING -- and the RULE IS THE ONE THE LOOP
    //     IMPLEMENTS, not the one intuition reaches for. `demoted_per_layer[layer]` is incremented
    //     once PER DEMOTED CELL IN THAT LAYER (:782), so its axis is `blocks`, while
    //     `demoted_per_block[b]` counts cells and its axis is `layers`:
    //       whole-LAYER unit => `demoted_max_per_layer == blocks` (and `demoted_layers == 1`)
    //       whole-BLOCK unit => `demoted_max_per_block == layers` (and `demoted_blocks == 1`)
    //       per-cell         => both maxima small AND `demoted` spread over `demoted_layers` rows
    //                           and `demoted_blocks` columns
    //     ⚠ AND WHEN `demoted == cells` ALL FOUR COUNTERS ARE FORCED AND CARRY NO INFORMATION:
    //     every layer then holds all `blocks` of its demoted cells and every block all `layers`, so
    //     a whole-layer unit, a whole-block unit and a per-cell walk print the SAME quadruple.
    //     That is the regime of every moving pass measured so far (F1209 pair: 768/768, 784/784,
    //     800/800), so these four numbers may NOT be cited as evidence there. The printout says
    //     which regime it is in (`granularity_informative=` below) and the DEEPEST RUNG'S
    //     PER-LAYER SHAPE is what discriminates instead: a whole-layer unit can only emit 0 or
    //     `blocks` in a layer, so a partial column rules it out. [F1215, from F1218's finding]
    //   * WHAT IS NOT, AND IS OWED: the ADMISSIBLE SET `F_(l,b)` itself. This field is indexed by
    //     layer alone, so every block of a layer shares one `F_l` and no caller can narrow it per
    //     page. WHY IT CANNOT BE A ONE-LINE FIX: `F_(l,b)` is also the input to the START (the base
    //     vector's `cell_default_tier` and the `planned_bytes` baseline, both computed once per
    //     layer above and then copied to every block), so a genuinely per-cell set means per-block
    //     base vectors and a per-block baseline -- a change to `block_descent_plan`'s prologue, not
    //     to this field. The producer in the tree (`program_impl.h`) supplies no set at all today,
    //     which is why the gap is invisible in a reading: `allowed_for()` returns `cell_all_modes()`
    //     for every cell and the uniformity is total.
    //   * WHAT IT DOES *NOT* BLOCK: nothing about this round's measurement. The demotion is already
    //     per-cell; this GAP is about how finely a cell's ALLOWED modes can be narrowed, and it is
    //     listed here so the next reader does not have to discover it, or worse, read this field's
    //     size as evidence that the descent is per-layer.
    std::vector<CellAllowedSet> allowed;   // size == layers; empty means `cell_all_modes()`
    CellFrequencySource frequency{};
    DescentPopulation population = DescentPopulation::CandidateWindow;
    std::uint64_t max_steps = 0;           // 0 means "derive from the rung count"
    // [F1209 2026-09-29] THE TWO KNOBS OF SECTION B2, AS FIELDS. A plan is a FUNCTION OF ITS SPEC,
    // never of the ambient environment -- the same rule that makes the budget a field -- so the
    // live caller reads the environment once per pass and fills these. BOTH DEFAULTS ARE THE
    // PRE-IMAGE SHAPE, which is what an unset environment must mean: the cheaper-below order, and
    // the full ladder reachable (`kDescentCeilingUnset` -> the floor, exactly like the ladder's own
    // unset behaviour).
    DescentStepOrder step_order = DescentStepOrder::PriciestBelow;
    std::int32_t ladder_ceiling_depth = kDescentCeilingUnset;
    // ---------------------------------------------------------------------------------------
    // [F1231] THE THREE FIELDS THE OWNER'S LAW NEEDS, AND WHERE EACH ONE COMES FROM.
    //
    // `age` IS F1202's `AgeGate` -- the file `kv_descent_control.h` that nothing included until
    // this line. `keep_recent_blocks` is the owner's K (「新内容应当仍然保持int8」) and its ZERO
    // value is the pre-image eligibility, so an unset field cannot change a run.
    //
    // ⚠ WHAT THE LIVE VALUE OF K IS, AND WHY IT IS NOT A BIGGER NUMBER: the population is now the
    // RETIREMENT span (`descent_population_pages`, kv_descent_control.h), so every block in it is
    // already older than the engine's own keep floor (`cold_keep_tokens`) -- K's work is already
    // done by the population. It is armed rather than left out because a FUTURE widening of the
    // population must not be able to re-invert R3 silently: with K set, the gate stops the newest
    // K blocks whatever the window is. `descent_keep_recent_pages_from_env()` is the caller's
    // reader, and the caller fills this field -- a plan is a function of its spec.
    AgeGate age{};
    // `walk` AND `cursor`: THE LAW'S SHAPE AND ITS POSITION. `OneShot` is R4's shape and the
    // default. `cursor == nullptr` means "no carry": a plan that is handed no cursor starts from
    // the default vector EVERY time, which is exactly the pre-image one-shot behaviour and is
    // named in the printout (`carry=absent`) rather than assumed.
    DescentWalk walk = DescentWalk::OneShot;
    DescentCursor* cursor = nullptr;
    // [F1231 window] THE POPULATION'S FIRST PAGE INDEX. A plan is a function of its spec, and the
    // spec must say WHICH PAGES the population covers, or a carried mode set cannot be checked for
    // identity -- only for shape, which is what let a slid window alias. The live caller passes the
    // window's own left edge (`cold_frontier`).
    std::uint32_t population_origin = 0;
};

struct BlockDescentPlan {
    // THE PAYLOAD INTO THE STAGE: `block_bytes[b] == bytes(V(b))`, and it is the ONLY thing the
    // charge needs. The vectors are carried beside it because the SIDECAR owes the per-layer
    // echo and because a reader has to be able to see WHY a block costs what it costs.
    std::vector<BlockVector> vectors;
    std::vector<std::int64_t> block_bytes;

    // THE OUTPUT THE ORDER ASKED FOR: "how much of it actually reaches e8 2bit DEPENDS ON THE
    // BUDGET CONSTRAINT" -- so the depth reached, and the count of cells that reached a floor,
    // are RESULTS, and they are recorded rather than assumed.
    std::uint32_t cells_total = 0;
    std::uint32_t cells_demoted = 0;          // cells that moved at least one rung
    // [planes 2026-09-29] ⭐ THE GRANULARITY PROOF, AS FOUR NUMBERS. The owner ruled twice that
    // "块粒度是核心 / 梯度要记得按块，不能全层下降", so "the walk is per-cell" must be MEASURED,
    // not claimed -- and `cells_demoted` alone cannot do it: 4 demotions are 4 cells in ONE layer
    // (a whole-layer move) or 4 cells over 4 different blocks (per-cell), and the count is 4 both
    // ways. These four separate the two shapes -- READ WITH THE AXES THE LOOP USES, which are
    // NOT the axes the names suggest (a whole-layer unit moves `blocks` cells, not `layers`):
    //   * WHOLE-LAYER descent  => demoted_max_per_layer == blocks (and demoted_layers == 1);
    //   * WHOLE-BLOCK descent  => demoted_max_per_block == layers (and demoted_blocks == 1);
    //   * PER-CELL descent     => BOTH maxima stay well below their axes, and `demoted_layers` /
    //                             `demoted_blocks` show the demotions spread over many
    //                             (block, layer) pairs rather than sitting on one row or column.
    // ⚠ [F1215, from F1218's finding] `demoted == cells` MAKES ALL FOUR VACUOUS: every cell has
    // moved, so every layer holds `blocks` demoted cells and every block `layers`, and the whole
    // quadruple is FORCED -- identical under a whole-layer unit, a whole-block unit and a per-cell
    // walk. Every moving pass measured so far is in that regime (F1209 pair: 768/768, 784/784,
    // 800/800), which is exactly where these four were cited as the proof. They are an instrument
    // with a REGIME, not an invariant: `granularity_informative=` (printed by `describe()`) names
    // the regime, and the deepest rung's per-layer shape is what discriminates inside the
    // degenerate one.
    // `demoted_layers` and `demoted_blocks` are the two DISTRIBUTIONS, reduced to the two numbers
    // that discriminate when they discriminate at all; the full per-layer and per-block vectors are
    // computed below, and `describe()` now prints the DEEPEST RUNG's per-layer shape from
    // `vectors` directly, which needs no extra state.
    std::uint32_t demoted_layers = 0;          // distinct layers with >= 1 demoted cell
    std::uint32_t demoted_blocks = 0;          // distinct blocks with >= 1 demoted cell
    std::uint32_t demoted_max_per_layer = 0;   // the largest demoted count inside ONE layer
    std::uint32_t demoted_max_per_block = 0;   // the largest demoted count inside ONE block
    std::uint32_t cells_at_floor = 0;         // the set has nothing cheaper: THE FLOOR
    std::uint32_t cells_blocked_by_unknown = 0;// the order's floor exists but is Unpriced (G6)
    std::uint32_t cells_blocked_by_currency = 0;// next rung is a codec change at the same record (G7)
    // The same two facts counted WITHOUT precedence, so a cell that is blocked by both is
    // visible in both columns. See `CellStep`'s note.
    std::uint32_t cells_with_unpriced_rung = 0;
    std::uint32_t cells_with_same_record_rung = 0;
    // [F1231, F-5 site 2] THE REFUSAL'S OWN FACT, BESIDE THE RECORD READING. G7's refusal is made
    // on `blocked_by_same_plane`; this counter is what reconciles `blocked_no_currency(G7)` against
    // a census, while the record counter above stays the codec reading it always was.
    std::uint32_t cells_with_same_plane_rung = 0;
    std::array<std::uint32_t, kCellModeCount> final_rung_histogram{};
    std::uint32_t sweeps = 0;                 // how many one-rung passes the walk completed
    std::uint32_t steps = 0;                  // accepted steps
    std::uint32_t refused_steps = 0;          // refusals seen at a cell (counted once per cell/sweep)
    // [F1209 2026-09-29] THE CEILING'S OWN COUNTER. A ceiling refusal is NOT one of the four
    // `StepRefusal` values -- the step function is never reached -- so folding it into the four
    // reason counters would corrupt the histogram that is this line's whole product. It is counted
    // here and printed by name, and it stays 0 whenever the ceiling is not the thing that stopped a
    // cell (including on every default run, where the ceiling is the floor and the check is off).
    std::uint32_t cells_stopped_by_ceiling = 0;
    // =========================================================================================
    // [F1231] THE LAW'S OWN READINGS. These are the columns the two-sided arm reads, and each one
    // is a number a rule can be judged against rather than a restatement of the rule.
    // =========================================================================================
    DescentWalk walk = DescentWalk::OneShot;   // the walk that actually ran (never the request)
    // R4: ONE TRIGGER => ONE DEMOTION. `triggers_fired` counts calls the CURSOR has been handed
    // (0 when the caller passed no cursor -- the honest reading for an uncarried plan), and
    // `steps_taken` is the cumulative accepted steps. `law_holds` is `steps_taken <= triggers_fired`
    // and it is evaluated HERE, on the real object, not only inside the file that defines it.
    std::uint32_t triggers_fired = 0;
    std::uint32_t steps_taken = 0;
    bool law_holds = true;
    // R2/R3: WHERE THE DEMOTION LANDED, IN THE ENGINE'S OWN AGE AXIS (the block index). Under R2
    // `demoted_newest_block` must sit below the population's newest block; under R3 the newest
    // block must not appear here at all. `age_gated_cells` is the count the GATE spared, so a
    // reader can tell "spared by the gate" from "never a candidate (at the floor)".
    std::uint32_t demoted_oldest_block = 0;
    std::uint32_t demoted_newest_block = 0;
    bool any_demoted = false;
    std::uint32_t age_gated_cells = 0;
    std::uint32_t age_keep_recent_blocks = 0;
    std::uint32_t age_frontier = 0;
    std::uint32_t newest_eligible_block = 0;
    // THE CARRY, AS READINGS: where this trigger resumed, where the next one resumes, and whether
    // a carry was OFFERED BUT REFUSED because its shape did not match this population.
    std::uint32_t resumed_offset = 0;
    std::uint32_t next_offset = 0;
    bool carry_refused = false;
    // [F1231 window] WHY the carry was refused, in words, and the origin it was built for -- so a
    // refusal is never mistaken for "there was nothing to carry".
    std::string carry_refusal_reason;
    std::uint32_t carry_origin = 0;
    std::uint32_t population_origin_seen = 0;
    // =========================================================================================
    // [F1245 kvcarry] THE RE-ANCHOR'S OWN ACCOUNTING. A carried mode set is re-anchored by ABSOLUTE
    // PAGE IDENTITY (`descent_reanchor_local`, kv_descent_control.h §5a) instead of being refused
    // whole when the window's block count or origin moved, and these four numbers are what makes
    // that a READING rather than a claim:
    //   * `carry_blocks_kept`    -- carried modes that landed on the page they were built for;
    //   * `carry_blocks_dropped` -- carried modes whose page had LEFT the window (RETIRED: the left
    //                               edge is a moving frontier, so this is the churn, counted);
    //   * `carry_blocks_new`     -- pages in this window that had no carried mode (they entered at
    //                               the default, which is the same start a cursor-less walk gets);
    //   * `carry_origin_delta`   -- `new_origin - cursor.origin`, SIGNED, so a window that moved
    //                               back (a restored sequence resets `cold_frontier` to 0) is
    //                               readable as such rather than as a normal forward slide;
    //   * `carry_position_kept`  -- whether the carried POSITION named a page this window still
    //                               covers. `false` means the lap restarts at this window's first
    //                               cell -- the position's page is gone, which is not the same
    //                               statement as "the modes were dropped" and is why it is its own
    //                               column.
    // The block and origin columns are 0 and the two flags false on a plan handed NO cursor, i.e.
    // on the pre-image path and on a fresh process's first trigger.
    bool carry_reanchor_attempted = false;
    std::uint32_t carry_blocks_kept = 0;
    std::uint32_t carry_blocks_dropped = 0;
    std::uint32_t carry_blocks_new = 0;
    std::int64_t carry_origin_delta = 0;
    bool carry_position_kept = false;

    std::int64_t budget_bytes = 0;
    std::int64_t planned_bytes = 0;           // the default vector's total, before any step
    // [F1231-budget-ruler] ⭐ THE PLAN'S OWN FLOOR, AND WHETHER THE BUDGET CAN EVER MEET IT.
    // `min_possible_bytes` is the cheapest charge this population can reach -- one cell at its own
    // cheapest priced rung, summed over the population -- so `budget >= min_possible_bytes` is the
    // SATISFIABILITY question, and it is the reading that separates the two ways a run ends with
    // `fits=no`: (i) the walk stopped early because the ladder ran out (drive=no-movable-cell with
    // `total == min_possible_bytes`), or (ii) the budget was authored in ANOTHER RULER and is below
    // the floor, in which case NO mix can fit it and the walk rests on the floor at maximum
    // degradation. MEASURED (b016_union_c3): the floor costs cells x 36,864 to the byte while the
    // budget was 16 x 1,181,696 -- 19.8x the floor's own... no: BELOW it, by 9,404,416 B. That is
    // case (ii), and it printed as if the mixer had failed.
    std::int64_t min_possible_bytes = 0;
    std::int64_t budget_blocks_x100 = 0;      // the budget in charge-priced blocks, x100 (a reading)
    bool budget_satisfiable = true;           // false ONLY when a live budget is below the floor
    // ============================================== [F1239 kvrate] THE RATE, AND THE PASS IT BOUND ON
    // ⭐ THE FIX FOR LAYER 1, AS A READING RATHER THAN A NEW KNOB. An absolute byte budget has NO
    // single rate: the same total is a different rate on every pass, because the pass's population
    // is its denominator. So the DESCRIPTIVE fix -- and it is available on the reading side of the
    // engine without touching how a caller authors a budget -- is to PRINT THE RATE THE TOTAL TURNS
    // OUT TO BE, on every plan, beside the floor's own rate.
    //
    // THEN A REFUSAL NAMES ITS PASS, which is the whole of the recorded verdict: *"A fixed byte
    // budget cannot be satisfiable against a population that grows 10x during one prefill. Any
    // future budget arm must state WHICH pass it bound on."* The pass IS `blocks` on this line, and
    // these two columns are the reason it bound: the budget's rate and the floor's rate, in the SAME
    // currency, so the gap is a subtraction a reader can see instead of a sentence they must trust.
    //
    // ⚠ THE CURRENCY IS DIMENSIONLESS AND ADDS NO CONSTANT: 10000ths of THIS population's own
    // all-int8 charge. `budget_rate_x10000 == 10000` means the budget prices every cell at int8;
    // the floor's rate is `min_possible_bytes`' own. MEASURED ROLE: on the fixed 36,700,160 B arm
    // the rate reads ~3044 at 46 blocks and ~1272 at 110 blocks -- ONE total, TWO rates -- while a
    // RATE arm prints the same number on all 24 passes, which is the reading that says the pass has
    // stopped being an input.
    std::int64_t budget_rate_x10000 = 0;
    std::int64_t floor_rate_x10000 = 0;
    // [F1239 kvrate] THE TRIGGER'S RATE AS IT ACTUALLY RAN, beside the walk it belongs to, so the
    // two-sided arm is a column on the line and not a diff: 1 before, `layers` after.
    std::uint32_t steps_per_call = 1;
    // =========================================================================================
    // [F1248 kvrate2] THE DERIVATION BEHIND `steps_per_call`, AS FOUR COLUMNS. `steps_per_call`
    // alone cannot say WHY it has the value it has, and the whole of this line is the claim that the
    // value is DERIVED from the pass's own rate gap rather than chosen -- so the derivation's own
    // components travel with it and are printed by `describe()` when (and only when) the derivation
    // actually ran:
    //   * `rate_derived`      -- did the derived path execute? FALSE on every plan whose budget is
    //                            OFF (the pre-image path returns before the derivation) and on
    //                            every non-`OneShot` walk, so no existing line grows a column.
    //   * `rate_gap_bytes`    -- `total_bytes - budget_bytes`: the distance still to travel, in the
    //                            charge's own bytes. `> 0` while the rate still binds.
    //   * `rate_travel_bytes` -- `planned_bytes - min_possible_bytes`: the travel this population
    //                            can buy at all, i.e. the denominator that turns the gap into a
    //                            COUNT OF CELLS. Both ends are this pass's own numbers.
    //   * `rate_blocks_per_call` -- the derived count, IN BLOCKS, which is the unit the owner's
    //                            「梯度要记得按块」 fixes. `steps_per_call` is this x `layers` (after
    //                            the one clamp at the derivation), so the two read as one fact.
    // `false`/`0`/`0`/`0` on a plan that took the fallback, which is what makes the fallback
    // VISIBLE rather than merely not-printed.
    bool rate_derived = false;
    std::int64_t rate_gap_bytes = 0;
    std::int64_t rate_travel_bytes = 0;
    std::uint64_t rate_blocks_per_call = 0;
    std::int64_t total_bytes = 0;             // after the walk: `sum_b bytes(V(b))`
    std::int64_t saved_bytes = 0;             // planned - total: THE CURRENCY, MEASURED
    DescentDrive drive = DescentDrive::BudgetOff;
    CellFrequencyKind frequency = CellFrequencyKind::Absent;
    DescentPopulation population = DescentPopulation::CandidateWindow;
    bool priced = true;                       // false: some cell's rung has no byte cost
    // [F1209 2026-09-29] THE TWO SELECTORS AS THEY WERE ACTUALLY USED, filled by the walk after
    // clamping, so a reading shows what ran rather than what was requested -- the same discipline
    // the sweep's `rung_ceiling_req` / `rung_ceiling_seen` pair uses. `ladder_ceiling_depth` here is
    // the CLAMPED value; `DescentSpec`'s is what the caller asked for.
    DescentStepOrder step_order = DescentStepOrder::PriciestBelow;
    std::int32_t ladder_ceiling_depth = static_cast<std::int32_t>(kLatticeFloorDepth);
    // [F1209 2026-09-29] THE RECONCILIATION READING, AND IT IS THE INSTRUMENT FOR THE FIX ABOVE.
    // `total_bytes` is carried INCREMENTALLY through the walk (it is the stop condition, so it must
    // be O(1) per step) while `block_bytes` is recomputed from the FINAL VECTORS at the end. Those
    // are two spellings of one sum, and the DEVELOPMENT-3 defect was exactly a drift between an
    // incremental total and the object it is supposed to measure -- so the two are compared here and
    // the answer is a READING. `false` prints a loud note; `true` is the silence that says the
    // walk's arithmetic and the charge agree. The check costs one addition per block, inside a loop
    // that was already running.
    bool totals_agree = true;

    // =========================================================================================
    // [F1240 kvsolve] THE SOLVE'S OWN LINE, CARRIED ON THE PLAN SO A LOG IS THE WHOLE READING.
    // This is the ONE field the stateless per-cell allocation solve adds to this object, and it is
    // EMPTY BY DEFAULT AND APPENDED ONLY WHEN NON-EMPTY, so a plan produced by the walk -- i.e.
    // every plan every existing run has ever printed -- is byte-identical to the pre-image string.
    // It is a `std::string` and not a set of typed columns on purpose: the solve's readings are
    // about a DIFFERENT object (a measurement table, a relaxation bound, a refusal) and giving them
    // typed homes here would put a second plan's vocabulary inside the walk's plan.
    // The producer is `product/kv_cell_alloc_solve.h`'s `CellAllocSolveReport::describe()`, and the
    // field is filled by BOTH of the solve's exits: `Solved` and every refusal. So a refused solve
    // is legible -- name, deficit and bound -- instead of printing a bare status quo.
    std::string alloc_note;

    // [fk 2026-09-29] THE SIGN WAS REVERSED, AND IT WAS THE GATE. MEASURED on BLK_cap256:
    // budget_bytes=302514176, total_bytes=59084800, and the line printed headroom_bytes=-243429376
    // -- with 302514176 - 59084800 == 243429376 EXACTLY. So this reported `total - budget`, i.e. it
    // said "no room left" while the plan sat comfortably INSIDE the budget, and
    // `drive=stopped-by-budget` broke out BEFORE any cell reached `cell_next_cheaper` ==> steps=0,
    // saved_bytes=0, at_floor=0, refused=0 (the whole row of zeros followed from this sign, not from
    // the supplied load). The stage prints the SAME quantity POSITIVE as `left_bytes`, so this is
    // "one quantity, two signs". Positive now means "slack remaining", the convention `left_bytes`
    // already uses; the pre-fix reading of this same plan must become +243429376.
    [[nodiscard]] std::int64_t headroom_bytes() const noexcept { return budget_bytes - total_bytes; }

    // =========================================================================================
    // [F1251 kvspend] ⭐⭐ THE DEVICE FACE: HOW MUCH OF `saved_bytes` CAN THE RESIDENT KV POOL GIVE
    // BACK? F-1250 measured the gap this closes: a plan that demoted 96.94 % of its cells and read
    // `saved_bytes = 1,022,623,744` on the charge's own ruler changed the engine's device readings
    // by ZERO BYTES (`kv cache payload 4.38 GiB`, `gpu sequence used 5.71 GiB`, both identical to
    // the original engine's), so the number was real on the plan's ruler and unreal on the device's.
    // These fields are that difference as a READING, filled from the same final vectors the charge
    // was recomputed from, so no run has to discover it again.
    //
    // THE STORAGE FACT THAT DECIDES IT, READ IN SOURCE (core/paged_kv_cache.cpp,
    // `plan_device_kv_page_pool`): the resident pool is laid out as ONE storage region per plane,
    // `{leading_extent, kPagedKVPageSize, head_extent, physical_pages}`, and a plane's only free
    // variable is its `dtype` (`KVPlaneGeometry`). A page's stride is therefore a CONSTANT OF THE
    // PLANE: one plane serves every page of its layer. So a per-(block, layer) rung -- which is what
    // this plan produces -- is expressible on the device if and only if EVERY cell of that layer in
    // this population already sits on ONE rung. A layer whose cells disagree keeps the baseline
    // plane, because its stride cannot carry two prices.
    //
    // ⚠ WHAT THESE FIELDS ARE NOT: they are not a promise, and a zero is not "the plan is wrong".
    // `device_realizable_saved_bytes == 0` with `saved_bytes > 0` means the plan's currency is the
    // resident plane and the plan's UNIT is the cell, while the pool's granularity axis is the
    // layer: the two are orthogonal and the storage has no third axis today. `device_blocker` names
    // that state. ⚠ AND THE OTHER DIRECTION MUST NOT BE MISREAD EITHER: a NON-zero value says the
    // rung a layer's whole population reached is the one its plane may take -- it does not say a
    // relayout happens, because the pool is laid out BEFORE the first token is generated (the page
    // count from `--kv-capacity`/`--max-context` and the per-plane dtype from the storage spec,
    // `resolve_kv_capacity` + `plan_device_kv_page_pool`), while this plan is a RUNTIME decision.
    std::uint32_t device_realizable_layers = 0;   // layers whose whole population may move a plane
    std::uint32_t device_uniform_layers = 0;      // layers whose whole population is on ONE rung
    std::int64_t device_realizable_bytes = 0;     // the plan's total, at the pool's own granularity
    std::int64_t device_realizable_saved_bytes = 0;  // the part of `saved_bytes` the pool can hold
    std::string device_blocker;                   // NAMED, and non-empty only when it is zero

    // ---------------------------------------------------------------------------------------------
    // [F1255 kvaxisA] THE THIRD AXIS' OWN COLUMNS. Filled at the same exit as the device face above,
    // from the same final vectors, and NON-ZERO ONLY WHEN THE POOL ACTUALLY CARRIES THE NARROW CLASS
    // (`NINFER_KV_AXIS3_NARROW_PAGES`, resolved by `kv_axis3_narrow_pages_from_env` and carried in
    // through `DescentSpec`). ⚠ WITH THE KNOB UNSET EVERY ONE OF THESE IS 0 AND `describe()` ADDS
    // NOTHING, so a run that sets nothing prints the pre-image string byte for byte.
    std::uint32_t axis3_narrow_pages_per_layer = 0;   // the pool's declared narrow class, in pages
    std::uint32_t axis3_demoted_cells = 0;            // cells the walk moved off their baseline rung
    std::int64_t axis3_saving_per_cell = 0;           // 2 * kv_heads * (base_plane - narrow_plane)
    std::int64_t axis3_pool_capacity_bytes = 0;       // what the narrow class can hold, in bytes
    std::int64_t axis3_realizable_saved_bytes = 0;    // min(saved_bytes, the capacity above)

    // =============================================================================================
    // [F1260 kvplanar] ⭐ THE PLANE PAIR, AS READINGS. A cell's state is (K format, V format) and
    // these are the columns that make the two of them SEPARATELY readable -- the thing the owner's
    // 「k 和 v 从开始就有不同的量化偏好基准、不同的量化次序，且不一定对称」 asks for.
    //
    // ⚠ THE WHOLE BLOCK IS SILENT ON A PLAN WHOSE CELLS ARE ALL DIAGONAL (`plane_axis_live` is
    // false), and every field before it is byte-identical to the pre-image string. That is a
    // STRONGER property than "unset means the pre-image": a plan with a live budget, a lattice
    // chain and a ceiling also prints character for character what it printed before, as long as its
    // two planes never parted -- which is every arm that existed before this landing.
    // =============================================================================================
    PlaneOrder plane_order_k = PlaneOrder::PriciestBelow;
    PlaneOrder plane_order_v = PlaneOrder::PriciestBelow;
    bool plane_axis_live = false;              // some cell's final pair is NOT diagonal
    std::uint32_t cells_asymmetric = 0;        // cells whose (K, V) rows differ
    std::uint32_t cells_pair_reserved = 0;     // of those, ones the DEVICE cannot read yet
    // The V plane's own census, beside the K plane's (`final_rung_histogram`) -- the two columns the
    // acceptance reads. For a diagonal plan they are the same vector, which is the point.
    std::array<std::uint32_t, kCellModeCount> v_rung_histogram{};
    // THE PAIR CENSUS: all `kCellModeCount x kCellModeCount` combinations, indexed `k * 6 + v`. A
    // fixed array and not a map, so the print has no allocation order to depend on.
    std::array<std::uint32_t, kCellModeCount * kCellModeCount> pair_histogram{};
    CellPair deepest_pair{};                   // the cheapest pair any cell reached (by pair bytes)
    std::int32_t deepest_pair_bytes = 0;
    CellPair floor_pair{};                     // the cheapest pair THIS SELECTION can reach
    std::int32_t floor_pair_bytes = 0;
    // THE TWO PREFERENCE BASELINES, AS THE START THE WALK BEGAN FROM. On today's producer both
    // planes start on the same row (the order's own "at the START they all default to int8 or
    // bf16"), and they are carried separately because a producer that gives K and V different
    // admissible rows would move ONE of them -- see the print, where the pair is named by row.
    CellPair base_pair_seen{};

    // THE ACCOUNTING LINE, one per pass, printed by the caller so this header stays free of I/O
    // policy -- the same discipline `KvBlockStageState::describe()` follows.
    [[nodiscard]] std::string describe() const {
        std::string out = "block-descent rule=sort-and-walk(G14)";
        out += " drive="; out += descent_drive_name(drive);
        // [F1231] THE LAW'S COLUMNS, BESIDE THE DRIVE, SO A LOG LINE IS THE WHOLE READING. Each of
        // these is a number the four rules can be judged against: `walk` says which shape ran,
        // `triggers`/`steps_taken`/`law_holds` are R4 with its own predicate attached, the age gate
        // is R2/R3's cut AS IT WAS APPLIED, and `demoted_block_range` is where the demotion LANDED
        // in the engine's own age axis. `carry=absent` is printed rather than implied: a plan with
        // no cursor starts from the default every time, which is the pre-image one-shot behaviour.
        out += " walk="; out += descent_walk_name(walk);
        out += " triggers=" + std::to_string(triggers_fired);
        out += " steps_taken=" + std::to_string(steps_taken);
        out += " law_holds=" + std::string(law_holds ? "yes" : "NO");
        out += " carry=";
        out += carry_refused ? ("REFUSED(" + carry_refusal_reason + ")")
                             : (triggers_fired == 0 ? "absent" : "carried");
        out += " carry_origin=" + std::to_string(carry_origin);
        out += " population_origin=" + std::to_string(population_origin_seen);
        // [F1245 kvcarry] THE RE-ANCHOR'S COLUMNS, APPENDED AND NOT SUBSTITUTED: every field before
        // this point is byte-identical to the line the pre-image printed, so a reader of the old
        // columns reads the same line. They appear ONLY when a carry was actually re-anchored (i.e.
        // not on a plan handed no cursor), and they answer the one question the census alone cannot:
        // HOW MUCH OF THE CARRIED MODE SET SURVIVED THE WINDOW'S OWN CHURN.
        if (carry_reanchor_attempted) {
            out += " carry_reanchor[kept=" + std::to_string(carry_blocks_kept) +
                   " dropped=" + std::to_string(carry_blocks_dropped) +
                   " new=" + std::to_string(carry_blocks_new) +
                   " origin_delta=" + std::to_string(carry_origin_delta) +
                   " position_kept=" + std::string(carry_position_kept ? "yes" : "NO") + "]";
        }
        out += " accumulation=";
        if (carry_reanchor_attempted) {
            // ⭐ THE PRE-IMAGE STRING SAID "a refused carry means NO accumulation" AND THAT SENTENCE
            // IS NO LONGER THE WHOLE TRUTH: a re-anchored carry DOES accumulate across passes, for
            // as long as the windows overlap. So the reading is now DERIVED from the two counters:
            // `kept` is how many pages' modes crossed the boundary, `dropped` is how much of the
            // walk's own history the churn took back. n/a only when no carry was attempted.
            out += "re-anchored-by-page-identity(carried=" + std::to_string(carry_blocks_kept) +
                   " pages of " + std::to_string(carry_blocks_kept + carry_blocks_dropped) +
                   " crossed this boundary; a demoted cell survives only while ITS PAGE is in the"
                   " window, so read this beside `demoted=`)";
        } else {
            out += "by-name-off(a refused carry means NO accumulation across passes:"
                   " this engine's windows churn, they do not slide by a little)";
        }
        out += " resumed_offset=" + std::to_string(resumed_offset);
        out += " next_offset=" + std::to_string(next_offset);
        out += " age_gate[K=" + std::to_string(age_keep_recent_blocks) +
               " frontier=" + std::to_string(age_frontier) +
               " newest_eligible=" + std::to_string(newest_eligible_block) + "]";
        out += " age_gated=" + std::to_string(age_gated_cells);
        out += " demoted_block_range=[";
        if (any_demoted) {
            out += std::to_string(demoted_oldest_block) + "," +
                   std::to_string(demoted_newest_block);
        } else {
            out += "none";
        }
        out += "] newest_block=" + std::to_string(vectors.empty() ? 0U : vectors.size() - 1U);
        // [F1209 2026-09-29] WHICH CHAIN AND WHICH CEILING ACTUALLY RAN, not which were asked for.
        // `step_order` and `ladder_ceiling_depth` are the CLAMPED, walk-time values, so a log cannot
        // be silent about the ladder that produced it -- and the sweep's table needs exactly this
        // pair to tell the two chains apart instead of guessing from the census.
        out += " chain="; out += descent_step_order_name(step_order);
        out += " "; out += ladder_ceiling_name(ladder_ceiling_depth);
        out += " frequency=";
        out += (frequency == CellFrequencyKind::Measured ? "measured" : "ABSENT(no-producer)");
        out += " population=";
        out += (population == DescentPopulation::CandidateWindow ? "candidate-window"
                                                                : "offered-population");
        out += " budget_bytes=" + std::to_string(budget_bytes);
        // [F1231-budget-ruler] THE FLOOR, THE SATISFIABILITY AND THE BUDGET'S RULER, on every line.
        out += " min_possible_bytes=" + std::to_string(min_possible_bytes);
        out += " budget_satisfiable=";
        // [F1239 kvrate] THE REFUSAL NOW NAMES THE PASS IT BOUND ON -- and it can, because the
        // budget's own RATE and the floor's own rate are on the line two columns away. The verdict
        // this answers was recorded verbatim with the measurement: *"Any future budget arm must
        // state WHICH pass it bound on."* The pass is `blocks=` (printed below) and the gap is
        // `budget_rate_x10000` against `floor_rate_x10000`; the sentence says so rather than asking
        // a reader to subtract two columns by hand.
        out += budget_bytes <= 0 ? "n/a(budget-off)"
                                 : (budget_satisfiable ? "yes(the budget is at or above this"
                                                        " population's own floor)"
                                                       : ("NO(the budget is BELOW the floor: no mix"
                                                          " can fit it and the walk will rest on the"
                                                          " floor -- check the budget's RULER."
                                                          " THIS PASS IS THE ONE THAT BOUND:"
                                                          " blocks=" + std::to_string(vectors.size()) +
                                                          " is the population this total was read"
                                                          " over, its rate is budget_rate_x10000=" +
                                                          std::to_string(budget_rate_x10000) +
                                                          " against the floor's floor_rate_x10000=" +
                                                          std::to_string(floor_rate_x10000) +
                                                          ", and THE SAME TOTAL IS A DIFFERENT RATE"
                                                          " ON EVERY PASS -- a RATE arm"
                                                          " (NINFER_KV_BLOCK_BUDGET_RATE_X10000)"
                                                          " prints one number on all of them)"));
        out += " budget_charge_blocks_x100=" + std::to_string(budget_blocks_x100);
        // [F1239 kvrate] THE TWO RATES, on every line, and every pass. They are 0 only when there is
        // no live budget, which is itself the reading "no rate exists to be read".
        out += " budget_rate_x10000=" + std::to_string(budget_rate_x10000);
        out += " floor_rate_x10000=" + std::to_string(floor_rate_x10000);
        out += " budget_rate_basis=10000ths-of-THIS-populations-all-int8-charge"
               "(a total prints a different rate per pass; a rate arm prints one)";
        // [F1239 kvrate] THE TRIGGER'S RATE, so (b) is a column and not a claim.
        out += " steps_per_call=" + std::to_string(steps_per_call);
        // [F1248 kvrate2] AND THE DERIVATION BEHIND IT, so (b)'s replacement is a READING too. The
        // bracket is APPENDED AND ONLY WHEN THE DERIVATION RAN, which is the same discipline
        // `carry_reanchor[...]` and `alloc_note` use above: every field printed before this point is
        // character-for-character the pre-image string on every plan that does not derive a rate, so
        // the OFF path and the `Sweep` path are unchanged as TEXT and not merely as behaviour. All
        // four numbers come from the walk that took the steps, and the two byte columns are the same
        // statement as the two rate columns above (`budget_rate_x10000` / `floor_rate_x10000`), in
        // the unit the count is actually built out of.
        if (rate_derived) {
            out += " steps_per_call_basis[derived-from-this-pass's-own-rate-gap:"
                   " gap_bytes=" + std::to_string(rate_gap_bytes) +
                   " travel_bytes=" + std::to_string(rate_travel_bytes) +
                   " cells=" + std::to_string(cells_total) +
                   " blocks_per_call=" + std::to_string(rate_blocks_per_call) +
                   " rule=ceil(gap*cells/travel) rounded up to whole blocks of "
                   + std::to_string(vectors.empty() ? 0U : vectors.front().layers) + " cells]";
        }
        out += " planned_bytes=" + std::to_string(planned_bytes);
        out += " total_bytes=" + std::to_string(total_bytes);
        out += " saved_bytes=" + std::to_string(saved_bytes);
        out += " headroom_bytes=" + std::to_string(headroom_bytes());
        out += " blocks=" + std::to_string(vectors.size());
        out += " layers=" + std::to_string(vectors.empty() ? 0U : vectors.front().layers);
        out += " cells=" + std::to_string(cells_total);
        out += " sweeps=" + std::to_string(sweeps);
        out += " steps=" + std::to_string(steps);
        out += " demoted=" + std::to_string(cells_demoted);
        // [planes 2026-09-29] THE PER-CELL / PER-LAYER DISCRIMINATOR -- AND ITS OWN REGIME.
        // ⚠ [F1215, from F1218's finding] THE RULE THIS COMMENT USED TO STATE WAS BACKWARDS, and
        // the axis swap is the trap: `demoted_per_layer[layer] += 1` fires once PER DEMOTED CELL,
        // so a WHOLE-LAYER unit prints `demoted_max_per_layer == blocks` (not `== layers`), while a
        // whole-block unit does print `demoted_max_per_block == layers`. AND THE WHOLE QUADRUPLE IS
        // VACUOUS WHEN `demoted == cells` -- which is every moving pass measured so far -- because
        // all four then sit at their forced maxima. So the printout says so, by name, on the line:
        // `granularity_informative=no(demoted==cells:...)`, and `deepest_per_layer=[...]` below
        // carries the discriminating column (a whole-layer unit can only emit 0 or `blocks` per
        // layer, so a PARTIAL column rules it out). The four counters are printed unchanged.
        out += " demoted_layers=" + std::to_string(demoted_layers);
        out += " demoted_blocks=" + std::to_string(demoted_blocks);
        out += " demoted_max_per_layer=" + std::to_string(demoted_max_per_layer);
        out += " demoted_max_per_block=" + std::to_string(demoted_max_per_block);
        out += " at_floor=" + std::to_string(cells_at_floor);
        out += " blocked_rung_unpriced(G6)=" + std::to_string(cells_blocked_by_unknown);
        out += " blocked_no_currency(G7)=" + std::to_string(cells_blocked_by_currency);
        out += " cells_with_unpriced_rung(G6)=" + std::to_string(cells_with_unpriced_rung);
        out += " cells_with_same_record_rung(G7)=" + std::to_string(cells_with_same_record_rung);
        out += " cells_with_same_plane_rung(F1231:the-refusal's-own-fact)=" +
               std::to_string(cells_with_same_plane_rung);
        out += " refused=" + std::to_string(refused_steps);
        // [F1209 2026-09-29] THE CEILING'S OWN COLUMN, beside the four `StepRefusal` counters above
        // and never folded into them: a ceiling refusal never reaches the step function, so mixing
        // it with `at_floor` / G6 / G7 would make the histogram unreadable. 0 on every run whose
        // ceiling is the floor, i.e. on every run that does not set the knob.
        out += " stopped_by_ceiling=" + std::to_string(cells_stopped_by_ceiling);
        out += " priced=" + std::string(priced ? "yes" : "NO");
        out += " rungs[";
        // [F1215 2026-09-29] ⚠ THE TWO HALVES OF THIS PAIRING WERE DIFFERENT LABELS, SO FROM
        // i=3 ON EVERY PRINTED NAME WAS OFF ITS BUCKET. MEASURED, and it is the defect OBS-1:
        // the F1209 arm logs print `rk4v4=749 rk3v4=0 nvfp4=0 e8-2bit=19` while the SAME LINE
        // prints `at_floor=0` -- and those two cannot both be true, because a cell on the 2-bit
        // floor IS `at_floor`. The census on those six lines is `rk4v4=749 rk3v4=19`, and 749+19
        // is exactly `cells=768` (697+87 = 784, 645+155 = 800), while the step count closes as
        // `steps = 768 x 2 + 19` and the bytes as `768 x 61440 + 768 x 4096 + 19 x 16384`.
        //
        // WHY. The NAME reads the TABLE's row order -- index 0..5 = Int8, Bf16, Rk4v4, Rk3v4,
        // Nvfp4, E8_2bit (`kv_cell_modes.h:273-389`) -- while the BUCKET is `final_rung_histogram`,
        // which is BINNED BY ENUM VALUE (`kv_cell_modes.h:190-208`: Int8=0 Bf16=1 Rk4v4=2 Nvfp4=3
        // E8_2bit=4 Rk3v4=5), because all three writers of this array index it that way
        // (`:532`, `:543`, `:768`). `Rk3v4` was APPENDED at enum value 5 and its ROW sits at
        // index 3, so the two orders diverge exactly there and the last three columns SHIFT.
        //
        // THE FIX: the row's OWN mode indexes the bucket, so a printed name matches its bucket by
        // construction. The row ORDER of the print is UNCHANGED on purpose -- the table's order is
        // "the order's own set" order and a reading should keep reading left to right as the table
        // does. This is a LABELLING fix: no cell moves, and nothing in the walk, the order, the
        // step function or the charge reads this printer.
        for (std::uint32_t i = 0; i < kCellModeCount; ++i) {
            if (i != 0) { out += " "; }
            out += std::string(cell_mode_name(kCellRungs[i].mode)) + "=" +
                   std::to_string(
                       final_rung_histogram[static_cast<std::uint32_t>(kCellRungs[i].mode)]);
        }
        out += "]";
        // [F1215, from F1218's finding] THE DEEPEST RUNG'S PER-LAYER SHAPE, AND WHETHER THE FOUR
        // COUNTERS ABOVE ARE INFORMATIVE ON THIS PASS.
        //
        // WHY THIS EXISTS. When `demoted == cells` the four counters are all at their forced
        // maxima, so they cannot tell a whole-layer allocation unit from a whole-block one from a
        // per-cell walk -- and that is the regime of every moving pass measured so far. The
        // DEEPEST RUNG is not degenerate: a whole-layer unit moves a whole layer, so its count per
        // layer is only ever 0 or `blocks` (the cell count of one layer), while a per-cell walk
        // emits a PARTIAL COLUMN as soon as the extra-rung count is not a multiple of `blocks`.
        // MEASURED on the F1209 pair's PLN_cap45 pass: blocks=48, and the extra-rung count is 19
        // (749 rk4v4 + 19 rk3v4 = 768 cells), so a partial column is what per-cell predicts and
        // what a whole-layer unit cannot produce. This needs NO new state: `vectors` already
        // carries the FINAL mode of every (block, layer) cell, so this reads the decision where it
        // was made rather than a summary of it.
        //
        // `granularity_informative` is the guard for the four counters above: it is `yes` only
        // when some cell was NOT demoted, i.e. only when the maxima are not forced.
        out += " granularity_informative=";
        out += (cells_demoted == cells_total)
                    ? "no(demoted==cells: all four counters sit at their forced maxima and cannot"
                      " tell a whole-layer unit from a whole-block one from a per-cell walk)"
                    : "yes(some cell did not move, so the four maxima are not forced)";
        if (!vectors.empty() && vectors.front().layers > 0) {
            const std::uint32_t layers_here = vectors.front().layers;
            // [F1260 kvplanar] THE DEEPEST CELL IS THE ONE WITH THE CHEAPEST PAIR, and on a diagonal
            // plan that is the same cell the pre-image expression picked (the argmin of
            // `2 x plane_bytes` and of `plane_bytes` is the same row). The pair is read from the plan
            // so this printer and the `planes[...]` block below cannot disagree about which cell is
            // "deepest" -- one quantity, one spelling.
            const CellMode deepest_mode = deepest_pair.k;
            std::vector<std::uint32_t> deepest_per_layer(layers_here, 0);
            std::uint32_t deepest_total = 0, deepest_max_per_block = 0, deepest_max_per_layer = 0;
            for (const BlockVector& v : vectors) {
                std::uint32_t in_block = 0;
                for (std::uint32_t l = 0; l < v.layers; ++l) {
                    if (v.pair_at(l) == deepest_pair) {
                        deepest_per_layer[l] += 1;
                        deepest_total += 1;
                        in_block += 1;
                    }
                }
                if (in_block > deepest_max_per_block) { deepest_max_per_block = in_block; }
            }
            for (std::uint32_t l = 0; l < layers_here; ++l) {
                if (deepest_per_layer[l] > deepest_max_per_layer) {
                    deepest_max_per_layer = deepest_per_layer[l];
                }
            }
            const std::uint32_t blocks_here = static_cast<std::uint32_t>(vectors.size());
            // A whole-layer allocation unit can only ever emit 0 or `blocks_here` in a layer.
            const bool partial =
                (deepest_max_per_layer > 0) && (deepest_max_per_layer < blocks_here);
            out += " deepest_rung=" + std::string(cell_mode_name(deepest_mode));
            out += " deepest_total=" + std::to_string(deepest_total);
            out += " deepest_max_per_layer=" + std::to_string(deepest_max_per_layer);
            out += " deepest_max_per_block=" + std::to_string(deepest_max_per_block);
            out += " deepest_layer_is_partial=";
            out += partial ? "yes(a whole-layer unit cannot produce 1..blocks-1 in a layer -- this"
                             " is the per-cell signature)"
                           : "no(every layer holds 0 or blocks, which a whole-layer unit CAN"
                             " produce -- this column does not discriminate on THIS pass)";
            out += " deepest_per_layer=[";
            for (std::uint32_t l = 0; l < layers_here; ++l) {
                if (l != 0) { out += " "; }
                out += std::to_string(deepest_per_layer[l]);
            }
            out += "]";
        }
        if (saved_bytes == 0 && drive == DescentDrive::NoMovableCell) {
            // THE LOUD ONE -- a walk that moved NOTHING is still a reading worth shouting about.
            // ⚠ [F1209 2026-09-29] ITS ORIGINAL REASON IS SUPERSEDED AND IS QUOTED HERE AS HISTORY,
            // because it described the tree before the site-3 currency fix and is now FALSE twice
            // over. It read: "On today's rung table the walk buys ZERO bytes, and this is why: the
            // default rung (int8, 9232 B) and the rung below it (rk4v4, 9232 B) write THE SAME
            // RECORD, while the rung the order actually wants (e8-2bit) has NO BYTE COST at all.
            // `cell_next_cheaper` refuses on both and no cell can move." THAT IS NOT THIS TREE ANY
            // MORE: all six rungs of `kCellRungs` are `PriceState::Priced` (read at the table), and
            // a step between two rungs that share a record is now a LEGAL, plane-positive step --
            // site 3 returns a plane delta, so "same record" no longer means "no currency". So the
            // note no longer names rungs. What it names now is the state that can actually produce
            // this line on this tree, and it points at the two places that answer it: the refusal
            // histogram printed immediately above (which carries the reason BY NAME) and the
            // ceiling, whose `0` value stops every cell on purpose.
            out += " note=no-rung-saves-bytes(superseded-F1202-reason-kept-in-source;";
            out += "read-the-refusal-histogram-and-stopped_by_ceiling)";
        }
        if (!totals_agree) {
            // THE RECONCILIATION'S LOUD HALF. It cannot fire on a correct walk -- the incremental
            // total and the recomputed charge are the same sum -- so if it ever prints, the
            // subtraction's unit and the charge's unit have diverged again, which is the exact
            // class of defect DEVELOPMENT-3 was. Silent when they agree.
            out += " note=TOTALS-DISAGREE(incremental=" + std::to_string(total_bytes) +
                   ";recomputed-charge-and-the-walk-have-drifted-apart)";
        }
        // [F1240 kvsolve] THE SOLVE'S LINE, AND IT IS APPENDED ONLY WHEN THERE IS ONE. The string is
        // empty on every plan the walk produces, so this branch is not a formatting change to any
        // existing reading -- the bytes of the pre-image line are unchanged, character for
        // character, which is the same discipline the walk's own `off is byte-for-byte` theorem
        // uses. When it is non-empty it is the WHOLE of the solve's accounting (the floor as it was
        // applied, the spend, the optimality bound with its verified gap, the refusal if there was
        // one, and the reason the carry columns are absent), produced by the object that made the
        // decision rather than reconstructed here.
        if (!alloc_note.empty()) {
            out += " ";
            out += alloc_note;
        }
        // [F1251 kvspend] ⭐⭐ THE DEVICE FACE, APPENDED AT THE END SO EVERY COLUMN BEFORE THIS POINT
        // IS THE PRE-IMAGE STRING CHARACTER FOR CHARACTER -- the same discipline `alloc_note` and
        // `carry_reanchor[...]` use. It is the ONE pair on this line that a device reading can be
        // reconciled against: `saved_bytes` is the plan's ruler, and `realizable_saved_bytes` is what
        // the RESIDENT POOL's own layout can hold (its only free variable is a plane's dtype, and one
        // plane serves all of a layer's pages). A reader who compares `saved_bytes` against
        // `kv cache payload` without this column is reading F-1250's gap; with it, the gap is a
        // number on the line and `device_blocker` names which granularity closed it.
        out += " device_face[realizable_layers=" + std::to_string(device_realizable_layers) + "/" +
               std::to_string(vectors.empty() ? 0U : vectors.front().layers) +
               " uniform_layers=" + std::to_string(device_uniform_layers) +
               " realizable_bytes=" + std::to_string(device_realizable_bytes) +
               " realizable_saved_bytes=" + std::to_string(device_realizable_saved_bytes) +
               " of_saved_bytes=" + std::to_string(saved_bytes) +
               " basis=one-storage-region-per-plane(core/paged_kv_cache.cpp:"
               "plan_device_kv_page_pool, a page's stride is a plane constant)]";
        if (!device_blocker.empty()) {
            out += " device_blocker=" + device_blocker;
        }
        // [F1255 kvaxisA] ⭐ THE THIRD AXIS' READING, APPENDED AFTER THE DEVICE FACE SO EVERY COLUMN
        // BEFORE IT -- INCLUDING THE WHOLE OF `device_face[...]` -- IS THE PRE-IMAGE STRING
        // CHARACTER FOR CHARACTER. ⚠ IT IS ABSENT WHEN THE KNOB IS UNSET: the pre-image pool has no
        // narrow class, and a column that printed a row of zeros for a dimension the storage does
        // not have would be a second reading of "the plan is wrong" that is really "the pool has no
        // third axis".
        if (axis3_narrow_pages_per_layer != 0) {
            out += " axis3[narrow_pages_per_layer=" + std::to_string(axis3_narrow_pages_per_layer) +
                   " demoted_cells=" + std::to_string(axis3_demoted_cells) +
                   " saving_per_cell=" + std::to_string(axis3_saving_per_cell) +
                   " pool_capacity_bytes=" + std::to_string(axis3_pool_capacity_bytes) +
                   " realizable_saved_bytes=" + std::to_string(axis3_realizable_saved_bytes) +
                   " basis=per-plane-page-extent(core/paged_kv_cache.h:KVPlaneGeometry::"
                   "page_group_count -- a layer carries BOTH plane sets, and the class of one "
                   "(block,layer) CELL is that layer's own block-table entry)]";
        }
        // =========================================================================================
        // [F1260 kvplanar] ⭐ THE TWO COLUMNS, APPENDED AFTER EVERY OTHER BLOCK -- SO EVERY COLUMN
        // BEFORE THIS POINT IS THE PRE-IMAGE STRING CHARACTER FOR CHARACTER -- AND PRINTED ONLY
        // WHEN A CELL'S TWO PLANES ACTUALLY PARTED.
        //
        // ⚠ THE GUARD IS THE STRONGEST FORM OF "UNSET = PRE-IMAGE" THIS LINE CAN OFFER, and it is
        // stronger than the one the axis3 landing used: that one is silent when a KNOB is unset, so
        // a plan with the knob set and a diagonal population still grows a block. This one is silent
        // whenever the POPULATION is diagonal -- which is every arm that existed before this landing,
        // with or without a budget, a chain or a ceiling. A reader who diffs tonight's logs against
        // the frozen pair therefore sees NOTHING here unless a two-sided selection was named.
        //
        // WHAT IT CARRIES, AND WHY EACH COLUMN IS NEEDED TO READ THE ONE ABOVE IT:
        //   * `order_k` / `order_v` -- THE TWO ORDERS, by name, as they RAN (not as they were asked
        //     for), because the claim of this landing is that they are independent and a log has to
        //     be able to show it;
        //   * `start` and `floor` -- THE TWO PREFERENCE BASELINES as a reader can tell them apart:
        //     where each plane began and where this selection's walk bottoms out. `start` is
        //     `k:int8/v:int8` on today's producer; `floor` is `k:e8-2bit/v:e8-2bit` (9,216 B) on any
        //     selection whose two planes descend together and `k:e8-2bit/v:rk4v4` (13,312 B) on the
        //     deployed family's -- WHICH IS THE READING THAT SAYS THE TWO PLANES ARE NOT THE SAME
        //     OBJECT ANY MORE;
        //   * `k_rungs[...]` / `v_rungs[...]` -- the two histograms SIDE BY SIDE, same six names in
        //     the same order, so a reader can see the cell-level asymmetry as a DIFF of two columns;
        //   * `pairs[...]` -- the census of the pairs themselves, `k:<row>/v:<row>=count`, which is
        //     where an ASYMMETRIC cell is visible by name rather than by inference;
        //   * `deepest_pair` / `read_side` -- the cheapest pair the population reached, and whether
        //     the DEVICE can read it today (a Reserved pair is PRICED and not deployable, and the
        //     two must never be reported as one).
        if (plane_axis_live) {
            out += " planes[order_k="; out += plane_order_name(plane_order_k);
            out += " order_v="; out += plane_order_name(plane_order_v);
            out += " start=" + cell_pair_name(base_pair_seen);
            out += " asymmetric=" + std::to_string(cells_asymmetric) + "/" +
                   std::to_string(cells_total);
            out += " reserved_pairs=" + std::to_string(cells_pair_reserved);
            out += " k_rungs[";
            for (std::uint32_t i = 0; i < kCellModeCount; ++i) {
                if (i != 0) { out += " "; }
                out += std::string(cell_mode_name(kCellRungs[i].mode)) + "=" +
                       std::to_string(final_rung_histogram[static_cast<std::uint32_t>(kCellRungs[i].mode)]);
            }
            out += "] v_rungs[";
            for (std::uint32_t i = 0; i < kCellModeCount; ++i) {
                if (i != 0) { out += " "; }
                out += std::string(cell_mode_name(kCellRungs[i].mode)) + "=" +
                       std::to_string(v_rung_histogram[static_cast<std::uint32_t>(kCellRungs[i].mode)]);
            }
            out += "] pairs[";
            bool first_pair = true;
            for (std::uint32_t k = 0; k < kCellModeCount; ++k) {
                for (std::uint32_t v = 0; v < kCellModeCount; ++v) {
                    // ⚠⚠ THE NAME COMES FROM THE ROW AND THE BUCKET FROM THE ROW'S OWN ENUM VALUE,
                    // AND THE TWO ORDERS ARE NOT THE SAME (row 3 is `Rk3v4`, whose value is 5). This
                    // is the OBS-1 trap `rungs[...]` below documents at length, and the first draft of
                    // THIS line walked straight into it: it indexed the census with the ROW, so every
                    // pair printed from row 3 on was off its bucket -- measured as a census that said
                    // `k:e8-2bit/v:rk4v4=37` while `deepest_pair` said `k:rk3v4/v:rk4v4` IN THE SAME
                    // LINE. `pair_histogram` is filled by enum value (`pair_index` at the census
                    // pass), so it must be read by enum value.
                    const std::uint32_t n =
                        pair_histogram[static_cast<std::uint32_t>(kCellRungs[k].mode) *
                                           kCellModeCount +
                                       static_cast<std::uint32_t>(kCellRungs[v].mode)];
                    if (n == 0) { continue; }
                    if (!first_pair) { out += " "; }
                    first_pair = false;
                    out += std::string("k:") + cell_mode_name(kCellRungs[k].mode) + "/v:" +
                           cell_mode_name(kCellRungs[v].mode) + "=" + std::to_string(n);
                }
            }
            out += "] deepest_pair=" + cell_pair_name(deepest_pair) + "=" +
                   std::to_string(deepest_pair_bytes);
            out += " floor_pair=" + cell_pair_name(floor_pair) + "=" +
                   std::to_string(floor_pair_bytes);
            out += " read_side=";
            out += (cells_pair_reserved != 0)
                       ? std::string("RESERVED(some cells sit on a pair whose narrow plate has no "
                                     "read-side consumer -- PRICED on the plan's ruler, NOT "
                                     "deployable: kv_storage_dtype.h:113-117)")
                       : std::string("shipped(every cell's pair has a consumer today)");
            out += " basis=per-plane-pair(kv_cell_modes.h:cell_pair_bytes = plane(k)+plane(v), a "
                   "DIAGONAL pair == 2 x the row's own one-plane price by construction; the budget, "
                   "the device face and axis3 all price the SAME pair)]";
        }
        return out;
    }
};

// ---------------------------------------------------------------------------------------------
// C2. [F1251 kvspend] THE DEVICE FACE, FILLED FROM THE FINAL VECTORS AT THE PLAN'S ONE EXIT.
// ---------------------------------------------------------------------------------------------
//
// WHY THIS IS A FUNCTION AND NOT A SENTENCE. `saved_bytes` is denominated in the RESIDENT PLANE's
// bytes on purpose -- `kv_cell_modes.h`'s `[planes 2026-09-29]` note says the charge sums
// `plane_bytes` for the reason that it "is the memory a demotion actually gives back". That sentence
// is TRUE OF A PLANE. Whether it is true of the DEVICE depends on a fact about the storage rather
// than about the plan, and F-1250 measured the two apart: 1,022,623,744 plan bytes, 0 device bytes.
// So the realizability is computed here, from the same FINAL vectors the charge below was recomputed
// from, and carried on the plan as a reading rather than left to be discovered per run.
//
// THE CONDITION, AND WHY IT IS EXACTLY THIS ONE. The pool's layout is one region per plane with
// `physical_pages` as the region's LAST axis (`core/paged_kv_cache.cpp`, `plan_device_kv_page_pool`),
// and a plane is parameterised by `KVPlaneGeometry{dtype, leading_extent, head_extent}`. A page's
// stride is therefore a function of the plane's DTYPE and of nothing per-page: one plane carries all
// of a layer's pages at one price. So a layer's plane may take a rung iff every cell of that layer in
// THIS population is on that rung; if two blocks disagree, neither rung is expressible, and the layer
// keeps its baseline plane. That is the whole of the derivation -- it needs no constant, and it moves
// by itself if `kCellRungs` gains a row or the pool gains an axis.
//
// WHAT IT DELIBERATELY DOES NOT DO: it does not round, it does not pick a "best" rung for a
// disagreeing layer, and it does not fall back to a per-layer average. A layer that cannot express
// its cells gives back zero, which is the honest number, and `device_blocker` says so by name.
inline void descent_fill_device_face(BlockDescentPlan& plan, const BlockVector& base,
                                     std::int32_t kv_heads) noexcept {
    if (plan.vectors.empty() || kv_heads <= 0) { return; }
    const std::uint32_t blocks = static_cast<std::uint32_t>(plan.vectors.size());
    const std::uint32_t layers = plan.vectors.front().layers;
    if (layers == 0 || blocks == 0) { return; }
    std::int64_t realizable_saved = 0;
    bool any_reserved = false;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        // [F1260 kvplanar] THE OBJECT OF THE COMPARISON IS THE PAIR, and "uniform" now means "every
        // block of this layer is on the SAME (K, V) PAIR". On a diagonal plan that is the pre-image
        // test on `mode`; on a two-sided plan it is the only test that can be honest about a layer
        // whose blocks agree on K and disagree on V.
        const CellPair first = plan.vectors.front().pair_at(layer);
        bool uniform = true;
        for (std::uint32_t block = 1; block < blocks; ++block) {
            const CellPair here = plan.vectors[block].pair_at(layer);
            if (here.k != first.k || here.v != first.v) { uniform = false; break; }
        }
        if (!uniform) { continue; }
        plan.device_uniform_layers += 1;
        // THE BASELINE IS THE PLAN'S OWN START, not a literal: a stack whose default rung for this
        // layer is not `int8` must be priced against ITS default, or the saving would be measured
        // from a plane the layer never had.
        const CellPair base_pair = base.pair_at(layer);
        const std::int32_t base_plane = cell_pair_bytes(base_pair);
        const std::int32_t rung_plane = cell_pair_bytes(first);
        if ((first.k == base_pair.k && first.v == base_pair.v) || rung_plane >= base_plane) {
            continue;
        }
        // ⭐ AND PRICED IS NOT THE SAME AS READABLE. A layer whose whole population agrees on a pair
        // the device has no consumer for is NOT realizable: the pool could lay the plane out and no
        // kernel could read it back (`cell_pair_read_side`, kv_cell_modes.h). Reporting it as a
        // saving would be the axis3 landing's own defect in the other direction -- a number on the
        // plan's ruler that the device cannot honour.
        if (cell_pair_read_side(first) == E8KvPlaneReadSide::Reserved) {
            any_reserved = true;
            continue;
        }
        const std::int64_t per_block =
            static_cast<std::int64_t>(kv_heads) *
            static_cast<std::int64_t>(base_plane - rung_plane);
        realizable_saved += per_block * static_cast<std::int64_t>(blocks);
        plan.device_realizable_layers += 1;
    }
    plan.device_realizable_saved_bytes = realizable_saved;
    plan.device_realizable_bytes = plan.planned_bytes - realizable_saved;
    if (plan.saved_bytes > 0 && realizable_saved == 0) {
        // THE NAMED BLOCKER, AND IT IS THE STATE F-1250 MEASURED RATHER THAN A HYPOTHESIS: the walk
        // moves one rung over EVERY block of a layer before any block of it takes a second -- that
        // is the order's own "DEMOTED ONE RUNG AT A TIME" read level-wise -- so the modes DISAGREE
        // across blocks on every layer that moved at all, and not one plane qualifies.
        plan.device_blocker =
            "NO-LAYER-IS-UNIFORM(this plan's rungs differ across blocks in every layer, and the"
            " resident pool's granularity axis is the LAYER, not the page: one storage region per"
            " plane with the page count as its last axis means a page's stride is a plane constant,"
            " so a per-(block,layer) rung is expressible only where a whole layer's population"
            " already agrees. 0 B of saved_bytes is realizable on the resident pool -- the plan's"
            " unit is the cell and the pool's is the layer, and the storage has no third axis"
            " today; a per-PAGE rung needs one that does not exist in KVPageGeometry)";
    }
    // [F1260 kvplanar] ⭐ A SECOND NAMED BLOCKER, AND IT IS ONLY APPENDED WHEN THE PAIR AXIS IS LIVE.
    // A uniform layer that reached a pair the device has no READER for gives back zero for a
    // DIFFERENT reason than the one above, and the two must not be reported as one: the first says
    // "the storage has no such axis", this one says "the storage has the axis and no consumer".
    if (any_reserved) {
        plan.device_blocker +=
            " + PAIR-IS-RESERVED(a layer's whole population agreed on a pair whose KNARROW VBROAD"
            " plate has no read-side consumer: `e8_lattice_kv_plane.cuh`'s"
            " `e8_kv_lattice_decode_group<3>/<2>` is written and never called"
            " (kv_storage_dtype.h:113-117), so the pair is PRICED on the plan's ruler and NOT"
            " realizable on the device's -- the price is real, the read-back is not)";
    }
}

// ---------------------------------------------------------------------------------------------
// C3. [F1255 kvaxisA] ⭐⭐ THE THIRD AXIS: A PER-CELL RUNG IS EXPRESSIBLE, SO `saved_bytes` IS
// REALIZABLE PER CELL RATHER THAN PER LAYER.
// ---------------------------------------------------------------------------------------------
//
// THE RULING THIS IMPLEMENTS, VERBATIM: 「你可以层下降，但我如果需要层里一部分高你该如何实现呢？
// 所以只能逐格」 -- the storage must be able to hold ONE LAYER'S CELLS ON DIFFERENT RUNGS, which is
// exactly what `descent_fill_device_face` above cannot price (`realizable_saved_bytes=0` at 64k, and
// `NO-LAYER-IS-UNIFORM` as its own named blocker).
//
// WHAT CHANGED IN THE STORAGE, READ IN SOURCE. `KVPlaneGeometry` (core/paged_kv_cache.h) now carries
// a per-plane `page_group_count`, so a layer may hold a SECOND plane set sized for the pages that
// class carries while the first set is sized for the rest. The class of a given (block, layer) cell
// is carried by THAT LAYER's block-table entry (`KVExecutionTables` is per-layer, i.e. the unit of
// the decision is the cell and the storage can hold it). The narrow class' page geometry is the
// nvfp4 plane pair, whose `plane_bytes` this header already prices (`kv_cell_modes.h`: nvfp4 9216
// against int8 16896), and whose reader and writer the tree already owns.
//
// WHY THE TWO COLUMNS ARE ONE NUMBER RATHER THAN TWO. This is the budget-ruler discipline
// (`kv_block_stage_budget_from_env`): the layout's saving per slot and this pricing are the SAME
// expression -- `2 * kv_heads * (base_plane - narrow_plane)` -- over the same two rows of the same
// table, so the engine's `kv cache payload` and this plan's currency cannot drift without a build
// failing on `kv_cell_modes.h`'s own `static_assert`s.
//
// ⚠ WHAT THIS IS NOT. It is not a claim that a demotion HAPPENS: it is what the storage can HOLD.
// The number a device reading must be reconciled against is `kv cache payload`, and the column here
// is the plan's own statement of the same bytes.
inline constexpr const char* kKvAxis3NarrowPagesEnv = "NINFER_KV_AXIS3_NARROW_PAGES";

// THE ONE READER of that knob. 0 (unset, empty or unparseable) IS THE PRE-IMAGE SHAPE: the pool
// allocates every plane at the pool's own page count, no narrow plane set exists, and every string
// below is byte-identical to the line the pre-image printed.
[[nodiscard]] inline std::uint32_t kv_axis3_narrow_pages_from_env() noexcept {
    const char* const raw = std::getenv(kKvAxis3NarrowPagesEnv);
    if (raw == nullptr || *raw == '\0') { return 0; }
    std::uint64_t value = 0;
    for (const char* p = raw; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') { return 0; }
        value = value * 10U + static_cast<std::uint64_t>(*p - '0');
        if (value > 1000000U) { return 0; }
    }
    return static_cast<std::uint32_t>(value);
}

struct Axis3DeviceFace {
    std::uint32_t narrow_pages_per_layer = 0;
    std::int32_t narrow_plane_bytes = 0;
    std::int32_t base_plane_bytes = 0;
    std::int64_t saving_per_cell = 0;      // 2 * kv_heads * (base - narrow)
    std::uint32_t rung_cells = 0;          // cells the walk actually moved, from the FINAL vectors
    std::uint32_t demoted_cells = 0;
    std::int64_t realizable_saved_bytes = 0;  // min(walk's saving, what the pool can hold)
    std::int64_t pool_capacity_bytes = 0;     // narrow_pages * layers * saving_per_cell
    [[nodiscard]] bool live() const noexcept { return narrow_pages_per_layer != 0; }
};

// WRITES THE AXIS3 COLUMNS ONTO THE PLAN, at the plan's own exits beside `descent_fill_device_face`.
// The count of cells the walk moved is the PLAN's own; the capacity is the STORAGE's; the realizable
// number is the smaller of the two, which is the only honest way to put a plan count beside a pool
// capacity. A zero `narrow_pages_per_layer` (the knob unset) writes nothing at all.
inline void descent_fill_axis3_face(BlockDescentPlan& plan, const BlockVector& base,
                                    std::int32_t kv_heads, std::uint32_t layers,
                                    std::uint32_t narrow_pages_per_layer) noexcept {
    if (narrow_pages_per_layer == 0 || kv_heads <= 0 || layers == 0) { return; }
    plan.axis3_narrow_pages_per_layer = narrow_pages_per_layer;
    // The narrow class is the nvfp4 plane pair -- the same row this header prices the ladder's
    // 4.50 b/el rung from (`kv_cell_modes.h`). It is read from the TABLE, not written as a literal,
    // and the LAYOUT's own plane geometry for that class is asserted against it by the two
    // `static_assert`s on `cell_rung` that already pin every row's byte cost.
    // ⚠ [F1260 kvplanar] AND IT IS A PAIR, because the layout reserves a PLANE SET for both K and V
    // (`k_narrow_pages` / `v_narrow_pages`, program_impl.h). `cell_pair_bytes(Nvfp4, Nvfp4)` is
    // `2 x cell_rung(Nvfp4).plane_bytes` by the diagonal pin, so this number is the SAME number the
    // pre-image expression produced -- the change is that the expression now says WHY it is that
    // number (two planes of the same row) instead of carrying an unexplained `2 *`.
    const CellPair narrow_pair{CellMode::Nvfp4, CellMode::Nvfp4};
    const std::int32_t narrow_plane = cell_pair_bytes(narrow_pair);
    std::int64_t capacity = 0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        // THE BASELINE IS THE PLAN'S OWN START, per layer: a stack whose default rung for this layer
        // is not int8 must be priced against ITS default.
        const std::int32_t base_plane = cell_pair_bytes(base.pair_at(layer));
        if (narrow_plane >= base_plane) { continue; }
        const std::int64_t saving_per_cell =
            static_cast<std::int64_t>(kv_heads) * static_cast<std::int64_t>(base_plane - narrow_plane);
        plan.axis3_saving_per_cell = saving_per_cell;
        capacity += saving_per_cell * static_cast<std::int64_t>(narrow_pages_per_layer);
    }
    // THE PLAN'S OWN COUNT, from the FINAL vectors: how many cells the walk actually moved off the
    // layer's baseline rung.
    for (const BlockVector& vector : plan.vectors) {
        if (vector.layers != layers) { continue; }
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            if (vector.mode[layer] != base.mode[layer]) { ++plan.axis3_demoted_cells; }
        }
    }
    plan.axis3_pool_capacity_bytes = capacity;
    // THE HONEST PAIRING: what the plan asks for, capped by what the storage can hold.
    plan.axis3_realizable_saved_bytes = std::min<std::int64_t>(plan.saved_bytes, capacity);
}

// ---------------------------------------------------------------------------------------------
// D. THE WALK. FIVE STEPS, IN THE ORDER OF THE REQUIREMENTS ABOVE.
// ---------------------------------------------------------------------------------------------
//
// THE SHAPE IS A ONE-RUNG SWEEP, NOT A DEEP-PER-CELL GREEDY, AND THAT IS A DECISION. The order
// says "IT IS DEMOTED ONE RUNG AT A TIME" and "the less-used ones are slowly tapered down": both
// describe a LEVEL, not a race -- every cell moves one rung before any cell moves two. A
// deepest-first greedy would reach the same byte total with FEWER deep cells and MORE int8
// cells, i.e. it would score better on the objective in section B.1 of the design, and it is
// NOT implemented here because the order's own words are level-wise and because a level-wise
// depth is what makes "how much reached the floor" a single legible number. Flagged for
// owner/main: `blob_F1172.md` section 6, dispatch sentence 4.
// =============================================================================================
// [F1260 kvplanar] D2. THE PAIR STEP: EACH PLANE TAKES ONE RUNG OF *ITS OWN* ORDER.
// =============================================================================================
//
// WHAT CHANGED AND WHAT DID NOT. The unit of a move is still ONE CELL, the granule is still a BLOCK
// (the sort's block-major order is untouched), and the amount is still the rate-derived
// `steps_per_call`. What changed is that a cell has TWO planes, so a step is a pair of per-plane
// steps and the two planes need not agree. THREE consequences, in the order of the requirements:
//
//   * THE PRE-IMAGE IS AN IDENTITY, NOT A TOLERANCE. When both planes start on the same row and
//     take the same order, the two side-steps are the SAME call on the SAME input, so the pair step
//     returns the same target, the same refusal and the same delta as `cell_next_cheaper` /
//     `cell_next_tier` -- asserted for all six rows below. Every existing arm therefore takes the
//     same steps, in the same order, for the same bytes.
//   * THE TWO ORDERS ARE INDEPENDENT (「不同的量化次序」). `plane_orders_of` is the only place a
//     selection becomes two orders, and nothing in this function assumes they agree.
//   * A PAIR MUST BE BUILDABLE (「V 不许被钉在 i4」, and its converse). `cell_pair_realizable` is
//     consulted HERE, on the candidate, so the walk cannot spend a pair the engine would refuse one
//     layer down. That gate is what makes the deployed family's chain -- `(B4,B4) -> (B3,B4) ->
//     (B2,B4)`, K narrowing while V keeps the family's plate -- the shape an asymmetric selection
//     actually takes, instead of an arbitrary mix nobody can build.
struct CellPairStep {
    CellPair to{};
    StepRefusal refusal = StepRefusal::None;
    // ⚠ THE ONE-PLANE CALIBER, KEPT ON PURPOSE. `ladder_step_total_delta(step, kv_heads)` -- the
    // tree's ONE spelling of the `2 * kv_heads` factor, in `kv_tier_ladder.h` -- multiplies this
    // field by `2 * kv_heads`. A pair's own delta is `pair_bytes(before) - pair_bytes(after)`, which
    // is TWICE that quantity for a diagonal step, so this field carries the pair delta HALVED, and
    // the pin below asserts that every plane price (and therefore every pair delta) is even so the
    // halving is exact. For a diagonal step this number IS the pre-image `bytes_saved`, to the byte.
    std::int32_t bytes_saved = 0;
    CellStep k_step{};   // the K side's own step: its refusal, its delta, its G6/G7 facts
    CellStep v_step{};   // the V side's own step
    bool k_held = false; // this plane did not move in THIS step
    bool v_held = false;

    [[nodiscard]] constexpr bool legal() const noexcept { return refusal == StepRefusal::None; }
    [[nodiscard]] constexpr bool asymmetric() const noexcept { return to.k != to.v; }
    // THE STEP AS THE CALIBER THE DRIVER ALREADY SCALES. The flags are the OR of the two sides'
    // own facts, which for a diagonal cell is the same value twice.
    [[nodiscard]] constexpr CellStep as_cell_step() const noexcept {
        return CellStep{to.k, refusal, bytes_saved,
                        k_step.blocked_by_unpriced_rung || v_step.blocked_by_unpriced_rung,
                        k_step.blocked_by_same_record || v_step.blocked_by_same_record,
                        k_step.blocked_by_same_plane || v_step.blocked_by_same_plane};
    }
};
// THE HALVING IS EXACT, and this is why: every row's one-plane price is even, so any difference of
// two pair sums is even. If a future rung has an odd price this fails to compile instead of silently
// truncating a byte out of the budget.
[[nodiscard]] constexpr bool cell_plane_prices_are_even() noexcept {
    for (const CellRung& rung : kCellRungs) {
        if ((rung.plane_bytes % 2) != 0) { return false; }
    }
    return true;
}
static_assert(cell_plane_prices_are_even(),
              "F1260: `CellPairStep::bytes_saved` is the PAIR delta HALVED (the one-plane caliber "
              "`ladder_step_total_delta` scales). An odd plane price would make that halving lossy");

// ONE PLANE, ONE RUNG, OF ITS OWN ORDER. The two ladder orders are the tree's own functions --
// `cell_next_cheaper` (the pre-image) and `cell_next_tier` (the owner's 4/3/2) -- called with the
// plane's own coordinate, so a plane's order is the SAME derived order the 1-D walk used, not a
// lookalike. `FamilyV` is the third order and it is the deployed family's rule as a value.
[[nodiscard]] constexpr CellStep cell_next_plane(CellAllowedSet set, CellMode from,
                                                 PlaneOrder side) noexcept {
    switch (side) {
    case PlaneOrder::Lattice432: return cell_next_tier(set, from);
    case PlaneOrder::FamilyV: {
        // "ONLY THE K CODE WIDTH MOVES" (`kv_kv_bits.h:279`): this plane's only rung is the family's
        // own plate, and once it is there it HOLDS. The hold is reported as `AlreadyAtFloor` because
        // that is exactly what it is -- this plane has nothing further below it IN THIS ORDER.
        E8KvPlaneFormat plane{};
        if (cell_row_e8_plane(from, plane)) {
            return CellStep{from, StepRefusal::AlreadyAtFloor};
        }
        const CellRung& here = cell_rung(from);
        const CellMode target = CellMode::Rk4v4;   // B4: the family's V plate, 8,704 B (kv_e8_width.h)
        if (!set.has(target)) { return CellStep{from, StepRefusal::CurrentModeNotInSet}; }
        const CellRung& to = cell_rung(target);
        if (to.price_state != PriceState::Priced) {
            return CellStep{from, StepRefusal::RungBytesUnknown, 0, true, false, false};
        }
        if (to.plane_bytes >= here.plane_bytes) {
            return CellStep{from, StepRefusal::AlreadyAtFloor};
        }
        return CellStep{target, StepRefusal::None, here.plane_bytes - to.plane_bytes};
    }
    case PlaneOrder::PriciestBelow: break;
    }
    return cell_next_cheaper(set, from);
}

// THE PAIR STEP. THE CANDIDATE ORDER IS THE WHOLE OF THE PRE-IMAGE ARGUMENT: the TWO-SIDED pair is
// tried FIRST, so when the two orders agree the pair that a diagonal cell steps to is the diagonal
// one -- and the single-sided candidates below it can never be reached. They exist for the case the
// two orders DISAGREE, and then the gate decides: a two-sided move whose combination the engine
// cannot build is refused, and the one-sided move (which is exactly "K 降、V 不降") is taken instead.
// That is why the deployed chain comes out of this function rather than being written into it.
[[nodiscard]] constexpr CellPairStep cell_next_pair(CellAllowedSet set, CellPair here,
                                                    PlaneOrder k_side,
                                                    PlaneOrder v_side) noexcept {
    CellPairStep step{};
    step.k_step = cell_next_plane(set, here.k, k_side);
    step.v_step = cell_next_plane(set, here.v, v_side);
    const bool k_moves = step.k_step.legal() && step.k_step.bytes_saved > 0;
    const bool v_moves = step.v_step.legal() && step.v_step.bytes_saved > 0;
    const CellPair both{k_moves ? step.k_step.to : here.k, v_moves ? step.v_step.to : here.v};
    const CellPair k_only{k_moves ? step.k_step.to : here.k, here.v};
    const CellPair v_only{here.k, v_moves ? step.v_step.to : here.v};
    if (k_moves || v_moves) {
        const CellPair candidates[3] = {both, k_only, v_only};
        for (const CellPair& candidate : candidates) {
            // BOTH CONDITIONS ARE REQUIRED: strictly cheaper (the walk's termination argument) and
            // buildable (`cell_pair_realizable`). A pair that is cheaper and unbuildable is NOT a
            // step -- it is a pair the engine would refuse one layer down.
            if (cell_pair_bytes(candidate) >= cell_pair_bytes(here)) { continue; }
            if (!cell_pair_realizable(candidate)) { continue; }
            step.to = candidate;
            step.refusal = StepRefusal::None;
            step.bytes_saved = (cell_pair_bytes(here) - cell_pair_bytes(candidate)) / 2;
            step.k_held = (candidate.k == here.k);
            step.v_held = (candidate.v == here.v);
            return step;
        }
    }
    // THE PAIR IS AT ITS REALIZABLE FLOOR. On a diagonal cell the two side-steps ran the SAME
    // function on the SAME input, so `k_step.refusal` IS the answer the pre-image step returned --
    // the G6-before-G7-before-floor precedence included. On a two-sided selection the K side's own
    // answer is reported, which is the coordinate the walk's census is about.
    step.to = here;
    step.refusal = (here.k == here.v) ? step.k_step.refusal : StepRefusal::AlreadyAtFloor;
    return step;
}

// =============================================================================================
// ⭐⭐ THE PRE-IMAGE PROOF, AS SIX `static_assert`s AND NOT AS A SENTENCE. For every row of the
// table, a diagonal cell under either pre-image selection must produce the SAME target, the SAME
// delta, and -- on a refusal -- the SAME named refusal as the 1-D step this walk used before the
// pair existed. Nothing about an existing arm's plan can move if these hold, because the walk's only
// input is this function's answer.
// =============================================================================================
[[nodiscard]] constexpr bool pair_step_reproduces_cheaper(CellMode from) noexcept {
    const CellAllowedSet all = cell_all_modes();
    const CellPairStep pair =
        cell_next_pair(all, CellPair{from, from}, PlaneOrder::PriciestBelow,
                       PlaneOrder::PriciestBelow);
    const CellStep cell = cell_next_cheaper(all, from);
    return pair.legal() == cell.legal() && pair.refusal == cell.refusal &&
           pair.bytes_saved == cell.bytes_saved && pair.to.k == cell.to && pair.to.v == cell.to &&
           pair.as_cell_step().blocked_by_unpriced_rung == cell.blocked_by_unpriced_rung &&
           pair.as_cell_step().blocked_by_same_record == cell.blocked_by_same_record &&
           pair.as_cell_step().blocked_by_same_plane == cell.blocked_by_same_plane;
}
[[nodiscard]] constexpr bool pair_step_reproduces_tier(CellMode from) noexcept {
    const CellAllowedSet all = cell_all_modes();
    const CellPairStep pair =
        cell_next_pair(all, CellPair{from, from}, PlaneOrder::Lattice432, PlaneOrder::Lattice432);
    const CellStep cell = cell_next_tier(all, from);
    return pair.legal() == cell.legal() && pair.refusal == cell.refusal &&
           pair.bytes_saved == cell.bytes_saved && pair.to.k == cell.to && pair.to.v == cell.to;
}
static_assert(pair_step_reproduces_cheaper(CellMode::Int8) &&
                  pair_step_reproduces_cheaper(CellMode::Bf16) &&
                  pair_step_reproduces_cheaper(CellMode::Rk4v4) &&
                  pair_step_reproduces_cheaper(CellMode::Rk3v4) &&
                  pair_step_reproduces_cheaper(CellMode::Nvfp4) &&
                  pair_step_reproduces_cheaper(CellMode::E8_2bit),
              "F1260: on EVERY row, a diagonal cell under `priciest-below` on both planes must "
              "produce the pre-image `cell_next_cheaper` step EXACTLY -- target, delta, refusal and "
              "both G6/G7 facts. This is what makes `NINFER_KV_DESCENT_CHAIN` unset mean the "
              "pre-image by construction and not by tolerance");
static_assert(pair_step_reproduces_tier(CellMode::Int8) &&
                  pair_step_reproduces_tier(CellMode::Bf16) &&
                  pair_step_reproduces_tier(CellMode::Rk4v4) &&
                  pair_step_reproduces_tier(CellMode::Rk3v4) &&
                  pair_step_reproduces_tier(CellMode::Nvfp4) &&
                  pair_step_reproduces_tier(CellMode::E8_2bit),
              "F1260: the same identity for `NINFER_KV_DESCENT_CHAIN=lattice`, so tonight's lattice "
              "arms measure the same chain they measured before");
// AND THE ASYMMETRIC ONE, WHICH IS THE POINT OF THE WHOLE LANDING: the deployed family's chain,
// taken from the ladder's own first rung, must produce the three deployed pairs in order.
[[nodiscard]] constexpr bool family_chain_is_the_deployed_chain() noexcept {
    const CellAllowedSet all = cell_all_modes();
    CellPair here{CellMode::Int8, CellMode::Int8};
    const CellPair want[3] = {{CellMode::Rk4v4, CellMode::Rk4v4},      // rk4v4 = (B4,B4)
                              {CellMode::Rk3v4, CellMode::Rk4v4},      // rk3v4 = (B3,B4)  <- ASYMMETRIC
                              {CellMode::E8_2bit, CellMode::Rk4v4}};   // rk2v4 = (B2,B4)  <- ASYMMETRIC
    for (const CellPair& expected : want) {
        const CellPairStep step =
            cell_next_pair(all, here, PlaneOrder::Lattice432, PlaneOrder::FamilyV);
        if (!step.legal() || step.to.k != expected.k || step.to.v != expected.v) { return false; }
        if (!cell_pair_realizable(step.to)) { return false; }
        here = step.to;
    }
    // AND THEN IT STOPS: K is at its floor and V is held, so the pair's floor under this selection is
    // the deployed `rk2v4`, 13,312 B -- NOT the diagonal floor (B2,B2) = 9,216, which this selection
    // cannot reach. That difference is a reading on the plan, not a claim in a comment.
    const CellPairStep stuck =
        cell_next_pair(all, here, PlaneOrder::Lattice432, PlaneOrder::FamilyV);
    return !stuck.legal() && cell_pair_bytes(here) == 13312;
}
static_assert(family_chain_is_the_deployed_chain(),
              "F1260: (K: 4/3/2 ladder, V: the family's plane) must walk int8 -> rk4v4 -> rk3v4 -> "
              "rk2v4 -- the three DEPLOYED pairs of `kv_e8_width.h`'s `e8_kv_pair_name_legacy` -- "
              "and then rest at 13,312 B/cell instead of the diagonal floor's 9,216");

[[nodiscard]] inline BlockDescentPlan block_descent_plan(const DescentSpec& spec) {
    BlockDescentPlan plan{};
    plan.budget_bytes = spec.budget_bytes;
    plan.frequency = spec.frequency.kind;
    plan.population = spec.population;

    const std::uint32_t layers =
        spec.layers == 0 ? 0U
                         : (spec.layers < kBlockVectorLayerAxis ? spec.layers
                                                                : kBlockVectorLayerAxis);
    const std::uint32_t blocks = spec.blocks;
    const std::uint32_t cells = blocks * layers;
    plan.cells_total = cells;

    // Resolution of F_(l,b). An empty `allowed` means "everything the layer can hold", which is
    // what a stack with one codec per layer already is.
    // [F1260 kvplanar] THE SELECTION, AS TWO PER-PLANE ORDERS, RESOLVED BEFORE ANYTHING READS IT.
    const PlaneOrders plane_orders = plane_orders_of(spec.step_order);
    auto allowed_for = [&spec, layers](std::uint32_t layer) -> CellAllowedSet {
        if (spec.allowed.empty() || layer >= spec.allowed.size() || layer >= layers) {
            return cell_all_modes();
        }
        return spec.allowed[layer];
    };

    // START: every cell at its default, one vector per block, all identical -- because the
    // population carries no per-block information yet. This is the order's start, and it is ALSO
    // the engine's pre-image charge (the literal `rans_layers = 0` prices every layer as int8).
    BlockVector base = block_vector_default(layers);
    bool any_no_default = false;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const CellDefault def = cell_default_tier(allowed_for(layer));
        base.set(layer, def.mode);
        if (def.state == DefaultTier::NoDefault) { any_no_default = true; }
    }
    plan.base_pair_seen = CellPair{base.mode[0], base.v_at(0)};
    // [F1231] R1 AND THE CARRY, AT THE ONE PLACE A START IS BUILT. The default vector above is
    // the start of the FIRST trigger and of every trigger when no cursor is handed in; with a
    // cursor the CARRY is the start -- read back, never re-derived (`kv_descent_control.h`'s own
    // half-one rule, which is why that file's `descent_cursor_init` is the only place a default
    // vector is built for a carried walk).
    //
    // A CARRY WHOSE SHAPE DOES NOT MATCH IS REFUSED AND SAID SO. A cursor built for a population
    // of 50 blocks must not silently supply the modes for a population of 48: the modes are
    // positionally meaningful, so a mismatch is a different object, and it is dropped with
    // `carry_refused=yes` on the line rather than accepted by luck.
    // [F1245 kvcarry] IDENTITY, NOT SHAPE -- AND THE ONE REFUSAL THAT REMAINS. The pre-image check
    // demanded THREE equalities (block count, layer count, first page) and REFUSED THE WHOLE CARRY
    // on any of them, which is what zeroed the per-cell accumulation across passes (F1244: 24-25 of
    // 27 passes, rung census pinned at `rk4v4=16`). The check is not weakened: the modes are still
    // never applied by INDEX to a population they were not built for. What changed is that the
    // BLOCK axis and the ORIGIN are now resolved in the coordinate that is comparable across two
    // windows -- the ABSOLUTE PAGE INDEX -- so a mode follows its own page (`descent_reanchor_local`)
    // and a page that left is dropped rather than aliased.
    //
    // ⚠ THE LAYER AXIS IS STILL A REFUSAL, BY NAME, AND THAT IS NOT TIMIDITY: `layers` is a
    // property of the STACK, not of the window. A population whose layer count changed is a
    // different OBJECT -- the modes are not a window-local fact that can be re-anchored onto it --
    // and this is the only way the pre-image refusal was about correctness rather than about churn.
    const bool carry_readable = spec.cursor != nullptr && spec.cursor->initialised();
    const bool carry_layers_ok = carry_readable && spec.cursor->layers() == layers;
    plan.carry_refused = carry_readable && !carry_layers_ok;
    if (plan.carry_refused) {
        plan.carry_refusal_reason =
            "layers-changed(" + std::to_string(spec.cursor->layers()) + "->" +
            std::to_string(layers) +
            ": the layer axis is a property of the STACK, not of the window, so a carried mode set"
            " is a different object and is NOT re-anchored onto it)";
    }
    plan.age_keep_recent_blocks = spec.age.keep_recent_blocks;
    plan.age_frontier = spec.age.frontier;
    plan.newest_eligible_block = spec.age.newest_eligible();
    plan.vectors.assign(blocks, base);   // every page enters at the DEFAULT until a carry says otherwise
    plan.block_bytes.assign(blocks, 0);
    if (carry_readable) {
        // THE ORIGIN IT WAS BUILT FOR, READ WHETHER OR NOT THE CARRY SURVIVES: the field's own
        // contract (defined beside it) is "the origin the cursor was built for", and a refusal that
        // printed 0 there could not be told from a cursor built for page 0.
        plan.carry_origin = spec.cursor->origin;
    }
    if (carry_layers_ok) {
        // ⭐ THE RE-ANCHOR. Every carried mode is placed on ITS OWN PAGE: local i of the old window
        // is page `cursor.origin + i`, and its home in this window is that page minus
        // `spec.population_origin`. A page this window no longer covers (retired, or -- in the other
        // direction -- not yet in it) is DROPPED and COUNTED rather than clamped; every local index
        // that receives nothing keeps the default the `assign` above gave it.
        plan.carry_reanchor_attempted = true;
        plan.carry_origin_delta = static_cast<std::int64_t>(spec.population_origin) -
                                  static_cast<std::int64_t>(spec.cursor->origin);
        std::uint32_t kept = 0;
        for (std::uint32_t i = 0; i < spec.cursor->blocks(); ++i) {
            std::uint32_t moved = 0;
            if (!descent_reanchor_local(spec.cursor->origin, spec.population_origin, blocks, i,
                                        moved)) {
                plan.carry_blocks_dropped += 1;
                continue;
            }
            plan.vectors[moved] = spec.cursor->vectors[i];
            kept += 1;
        }
        plan.carry_blocks_kept = kept;
        plan.carry_blocks_new = blocks - kept;
        // THE POSITION, PROJECTED BY THE SAME IDENTITY. A position whose page is gone is not a
        // position: `resumed_offset` then stays at 0, i.e. this window's first cell -- the honest
        // restart, and the column above says it happened.
        std::uint32_t projected = 0;
        if (descent_reanchor_position(spec.cursor->origin, spec.population_origin, blocks, layers,
                                      spec.cursor->next_offset, projected)) {
            plan.carry_position_kept = true;
            plan.resumed_offset = projected < cells ? projected : 0U;
        }
    }

    const CellCharge base_charge = cell_vector_charge(base, spec.kv_heads);
    plan.planned_bytes = static_cast<std::int64_t>(blocks) * base_charge.bytes;
    // ⚠ [F1248 kvrate2] THIS ASSIGNMENT IS THE PRE-IMAGE ONE, AND IT STAYS THE PRE-IMAGE ONE: on a
    // plan that never walks, `total_bytes` is `blocks x bytes(base)` exactly as it always was, so an
    // OFF plan's printed `total_bytes=` / `saved_bytes=` / `headroom_bytes=` are character-for-
    // character the pre-image values. WHERE THE WALK CORRECTS IT, AND WHY, IS STATED AT THE
    // CORRECTION: see the block between the `BudgetOff` return and the sort below.
    plan.total_bytes = plan.planned_bytes;
    if (!base_charge.priced || any_no_default) {
        // A DEFAULT RUNG WITH NO BYTE COST -- either the rung itself is Unpriced, or F_(l,b) has
        // no priced rung at all and `cell_default_tier` said `NoDefault`. There is no byte total,
        // so there is no stop condition, so there is no walk. Named, never silently zeroed.
        plan.priced = false;
        plan.drive = DescentDrive::UnpricedCell;
        for (std::uint32_t b = 0; b < blocks; ++b) { plan.block_bytes[b] = base_charge.bytes; }
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            plan.final_rung_histogram[static_cast<std::uint32_t>(base.mode[layer])] += blocks;
        }
        // [F1251 kvspend] THE DEVICE FACE IS FILLED ON THIS EXIT TOO, SO IT IS ALWAYS SELF-CONSISTENT:
        // no cell moved here, so there is nothing to give back and `realizable_saved_bytes` must read
        // 0 rather than being left uninitialised -- an absent number would be indistinguishable from
        // "the storage can hold none of it", which is the reading this field exists to separate.
        descent_fill_device_face(plan, base, spec.kv_heads);
        descent_fill_axis3_face(plan, base, spec.kv_heads, layers, spec.narrow_pages_per_layer);
        return plan;
    }

    // THE BUDGET IS THE STOP CONDITION. `budget <= 0` is OFF and OFF is the default vector,
    // which is the pre-image charge -- reported as `BudgetOff`, never walked.
    if (spec.budget_bytes <= 0) {
        plan.drive = DescentDrive::BudgetOff;
        for (std::uint32_t b = 0; b < blocks; ++b) { plan.block_bytes[b] = base_charge.bytes; }
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            plan.final_rung_histogram[static_cast<std::uint32_t>(base.mode[layer])] += blocks;
        }
        descent_fill_device_face(plan, base, spec.kv_heads);   // [F1251] same reason as above
        descent_fill_axis3_face(plan, base, spec.kv_heads, layers, spec.narrow_pages_per_layer);
        return plan;
    }

    // =========================================================================================
    // [F1248 kvrate2] ⭐ FROM HERE ON THERE IS A LIVE BUDGET, AND THE INCREMENTAL TOTAL IS CORRECTED
    // TO START WHERE ITS OWN DEFINITION SAYS IT STARTS: AT THE CHARGE OF THE VECTORS THE WALK IS
    // ABOUT TO MOVE. IT IS CORRECTED HERE -- AFTER THE TWO EARLY RETURNS AND BEFORE THE SORT --
    // BECAUSE A PLAN THAT NEVER WALKS MUST KEEP THE PRE-IMAGE READING (the note at the assignment
    // above), AND BECAUSE ONLY A WALK USES THIS NUMBER AS ITS STOP CONDITION.
    //
    // THE DEFECT THIS CLOSES, MEASURED BEFORE IT WAS TOUCHED. The field's own contract (declared
    // beside it) is "`total_bytes`: after the walk, `sum_b bytes(V(b))`" -- the charge of the object
    // the stage is handed. The pre-image spelling on the walk path was still the ALL-DEFAULT product
    // `blocks x bytes(base)`, which equals that sum ONLY when every vector IS the default. After
    // F1245's re-anchor that is false: a carried mode is placed on its own page
    // (`plan.vectors[moved] = spec.cursor->vectors[i]` above), so the walk starts from a population
    // that is ALREADY partly demoted while the total starts from all-int8, and the stages between two
    // triggers do not add up. MEASURED on the F1245 arm `B_hold_rate`
    // (`dl/kvcarry/out/arms2/B_hold_rate/stderr.txt`): `grep -c TOTALS-DISAGREE` = **26 of 27
    // passes**, and its last line reads `demoted=432 ... saved_bytes=1048576 steps=16` -- 432 cells
    // have moved while the plan reports this trigger's 16 steps as the whole saving, because
    // 432 x 65,536 = 28,311,552 bytes of REAL carried saving are invisible to the incremental total.
    // The file's own reconciliation column was printing the drift on every one of those passes, and
    // the drift was not in the arithmetic: it was in the STARTING POINT.
    //
    // WHY THIS IS THE TREE'S OWN RULE AND NOT AN INVENTION. The sibling entry point for the same
    // quantity states it in capitals (`kv_descent_control.h`, `block_descent_trigger`: "THE BUDGET,
    // IN THE CHARGE'S OWN RULER, READ FROM THE CARRIED MODES. Recomputed from `cursor.vectors` --
    // NOT incremented -- so a cursor that was handed modes from elsewhere cannot carry a total that
    // disagrees with them."). One spelling of one quantity: the total IS the charge of the vectors,
    // and the vectors are `plan.vectors`. This loop is the same O(blocks x layers) sum this function
    // already performs after the walk, on the same object -- it adds no new arithmetic, only the
    // correction of where the sum starts. `totals_agree` at the tail is therefore TRUE on a carried
    // plan instead of printing the drift.
    //
    // ⚠ AND IT IS ALSO WHAT MAKES THIS LINE'S DERIVATION MEANINGFUL. The trigger's size is derived
    // from `total_bytes - budget_bytes` (see the derivation note at `steps_per_call`), so a total
    // that cannot see the carried demotions would make every trigger believe it still had the whole
    // job to do, and the walk would descend to the FLOOR on every budget rather than to the rate the
    // operator asked for. The two changes are one change.
    {
        std::int64_t walked_start_bytes = 0;
        for (std::uint32_t b = 0; b < blocks; ++b) {
            walked_start_bytes += cell_vector_charge(plan.vectors[b], spec.kv_heads).bytes;
        }
        plan.total_bytes = walked_start_bytes;
    }

    // (2) THE SORT KEY. Ascending: the LEAST used cell is demoted FIRST, because the order's
    // sentence is "layers used a lot later ... keep int8; the less-used ones are slowly tapered
    // down". The tie-break is `(layer, block)`, so the order is a TOTAL order and the plan is
    // reproducible; with `CellFrequencyKind::Absent` every key is equal and THE TIE-BREAK IS THE
    // ORDER -- named, printed, and the reason `drive` never hides it.
    struct Key {
        std::uint32_t key;
        std::uint32_t layer;
        std::uint32_t block;
    };
    std::vector<Key> order;
    order.reserve(cells);
    for (std::uint32_t block = 0; block < blocks; ++block) {
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            const std::uint32_t key = spec.frequency.usable()
                                          ? spec.frequency.revisit_count(spec.frequency.self,
                                                                          block, layer)
                                          : 0U;
            order.push_back(Key{key, layer, block});
        }
    }
    // [F1231] THE TIE-BREAK MOVED FROM `(layer, block)` TO `(block, layer)`, AND THAT IS F-2's
    // FIX. With every key equal (frequency ABSENT) the pre-image order was LAYER-major -- layer 0
    // of every block, newest included, before layer 1 of any block -- so the traversal order was a
    // reproducible fact and NOT a criterion. The key axis keeps its place (a future frequency
    // producer still dominates, which is what the sort is FOR); the SECOND axis is now the age
    // proxy this tree already has, the block index, so the walk descends the oldest block first and
    // a block's layers stay contiguous. `descent_order_index_age_major` is the same order's closed
    // form, which is what makes a carried offset mean the same thing in two calls.
    std::sort(order.begin(), order.end(), [](const Key& a, const Key& b) {
        if (a.key != b.key) { return a.key < b.key; }
        if (a.block != b.block) { return a.block < b.block; }
        return a.layer < b.layer;
    });

    // =========================================================================================
    // [F1248 kvrate2] THE FLOOR AND THE SATISFIABILITY AND THE TWO RATES, MOVED UP HERE FROM BESIDE
    // THE TOTALS. The words below are the pre-image words, unedited, except for this paragraph: the
    // block was lifted so that the walk's TRIGGER SIZE can be DERIVED from it, which is the whole of
    // this line, and it is lifted rather than duplicated because one quantity must have one spelling.
    // All six of its inputs are settled before the sort above, so it computes the same numbers it
    // computed in its old place -- and `min_possible_bytes` / `budget_rate_x10000` /
    // `floor_rate_x10000` are all printed, so a move that changed them would be a visible DIFF.
    //   [F1231-budget-ruler] THE FLOOR AND THE SATISFIABILITY, from the ALLOWED SETS (so a narrowed
    //   F_(l,b) is priced as narrowed): the cheapest priced rung each layer can reach, x blocks x the
    //   2*kv_heads factor the charge carries.
    {
        // [F1260 kvplanar] ⭐ THE FLOOR IS A PAIR AND A FUNCTION OF THE SELECTION, NOT A PLANE.
        // The pre-image spelling took the cheapest ROW a layer may reach and multiplied it by two,
        // which is the diagonal pair's price -- correct, and only for a selection whose two planes
        // descend in lockstep. Under a two-sided selection the cheapest pair a cell can REACH is not
        // the cheapest pair that EXISTS: `(K: 4/3/2, V: family plane)` bottoms out at the deployed
        // `rk2v4` = 13,312 B/cell while `(B2,B2)` = 9,216 B sits unreachable beside it. So the floor
        // is computed by walking the pair along the plan's OWN orders until the step refuses, which
        // is the definition of "the cheapest charge this population can reach" and not an
        // approximation of it. On either pre-image selection this walk lands on `(e8-2bit, e8-2bit)`
        // and the number is the pre-image `blocks * 2 * kv_heads * 4608`, term for term.
        std::int64_t per_block_floor = 0;
        bool floor_seen = false;
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            const CellAllowedSet set = allowed_for(layer);
            CellPair floor_pair_here{base.mode[layer], base.mode[layer]};
            for (std::uint32_t hop = 0; hop < kCellModeCount * kCellModeCount; ++hop) {
                const CellPairStep walk_step =
                    cell_next_pair(set, floor_pair_here, plane_orders.k, plane_orders.v);
                if (!walk_step.legal() || walk_step.bytes_saved <= 0) { break; }
                floor_pair_here = walk_step.to;
            }
            const bool priced = cell_rung(floor_pair_here.k).price_state == PriceState::Priced &&
                                cell_rung(floor_pair_here.v).price_state == PriceState::Priced;
            if (!priced) { continue; }
            per_block_floor += cell_pair_bytes(floor_pair_here);
            if (!floor_seen || cell_pair_bytes(floor_pair_here) < plan.floor_pair_bytes) {
                plan.floor_pair = floor_pair_here;
                plan.floor_pair_bytes = cell_pair_bytes(floor_pair_here);
                floor_seen = true;
            }
        }
        plan.min_possible_bytes = static_cast<std::int64_t>(blocks) *
                                  static_cast<std::int64_t>(spec.kv_heads) * per_block_floor;
        // THE BUDGET IN BLOCKS OF *THIS PLAN'S OWN DEFAULT CHARGE* -- the same quantity the sweep's
        // table calls `budget_blocks_charge` (budget / 2,162,688 on the shipped stack), computed from
        // the plan's own unit so it cannot drift from the number `decide()` spends.
        if (spec.budget_bytes > 0 && base_charge.bytes > 0) {
            plan.budget_blocks_x100 = spec.budget_bytes * 100 / base_charge.bytes;
            plan.budget_satisfiable = spec.budget_bytes >= plan.min_possible_bytes;
            // [F1239 kvrate] THE TWO RATES, from the SAME three numbers as the three columns above
            // (the budget, the floor, and THIS pass's population x its own unit), for the same
            // reason: one pass's readings must come from one pass.
            const std::int64_t population_charge =
                static_cast<std::int64_t>(blocks) * base_charge.bytes;
            if (population_charge > 0) {
                plan.budget_rate_x10000 = spec.budget_bytes * 10000 / population_charge;
                plan.floor_rate_x10000 = plan.min_possible_bytes * 10000 / population_charge;
            }
        }
    }

    // (5) THE BOUND. One sweep moves every cell at most one rung; a cell has at most
    // `kCellModeCount - 1` rungs below it, so `kCellModeCount` sweeps cannot leave a legal step
    // untaken. The hard step bound is a second belt for a future rung whose delta might not be
    // positive.
    // [F1231] R4, AS A BOUND: `OneShot` runs the outer loop ONCE and takes AT MOST ONE step in it.
    // The pre-image `Sweep` value keeps the old bound so the two-sided arm can measure against it.
    // =========================================================================================
    // ⭐⭐ [F1239 kvrate] THE TRIGGER'S RATE -- AND THE CHOICE IS NOW TAKEN. It was the owner's open
    // question in F1231; THE OWNER PRE-APPROVED THIS CHANGE (「回一条拍一条」, and 「把这几层修了」),
    // and the line `kvrate` landed it. WHAT WAS LANDED: **(b) PER BLOCK**.
    //
    // ⭐ THE JUSTIFICATION IS HIS OWN WORDS, VERBATIM: 「梯度要记得按块，不能全层下降」 -- the gradient
    // must be remembered BY BLOCK; it must not descend whole layers at a time. (b) is that sentence
    // as an expression: one trigger spends `layers` steps, and because the walk's order is
    // BLOCK-MAJOR (`(block, layer)` -- see the tie-break note above), `layers` consecutive steps are
    // ONE BLOCK'S LAYERS. A trigger therefore demotes a BLOCK, one rung per cell in it.
    //
    // ⚠ AND THE RATE WAS 46x SHORT, WHICH IS WHY A CHOICE HAD TO BE MADE: MEASURED on the new pair,
    // the run yielded ONE armed pass per ~2,700 prompt tokens (24 passes at 64,512 tokens) while the
    // 2.8 target needs 1,116 rung steps -- 24 delivered, 1,116 required, a **46x** shortfall. With
    // one step per pass the arithmetic was worse than useless: a 64k prefill could move about 24 of
    // 768 cells, i.e. `bits_per_element` 8.2500 -> ~8.2490, which is INVISIBLE IN ANY TABLE.
    //
    // WHAT (b) CLOSES, AND WHAT IT DOES NOT -- stated so the remainder is not mistaken for a fix:
    //   * CLOSED: the per-trigger UNIT, from one CELL to one BLOCK. 24 passes now deliver up to
    //     24 x `layers` = 384 cells on the shipped 16-layer stack -- THE 16x the (b) note below
    //     always claimed, and it is now a reading rather than an argument.
    //   * ⚠ NOT CLOSED: the remaining ~2.9x is the PASS COUNT -- how often a trigger fires at all --
    //     and that is (a)'s requirement (a trigger source the engine does not have), NOT this line's
    //     gift. NO CLAIM IS MADE THAT (b) REACHES 2.8.
    //
    // ⚠ (a) IS LEFT NAMEABLE AND DELIBERATELY NOT REMOVED. It keeps the whole of its description
    // below, because it still needs a trigger source the engine lacks and a later window may want
    // it: it is finer IN TIME than (b) and it is the only shape that can act on a pressure reading
    // taken BETWEEN passes. Leaving it written down is the point -- a later reader must be able to
    // choose it on evidence rather than re-derive it.
    //
    //   (a) PER PRESSURE EVENT -- keep 1, and let the trigger fire more often than once per pass
    //       (i.e. the caller re-plans while the observed total is still over budget). Finer in
    //       time; needs a trigger source the engine does not have today. ⭐ NOT LANDED -- still
    //       reachable and still described, by the owner's own instrument of a per-pressure-event
    //       trigger. THE ONE THING (a) NEEDS AND THIS LINE CANNOT SUPPLY: a caller that re-enters
    //       the plan while the observed total is over budget. This walk is entered ONCE PER PASS.
    //   (b) PER BLOCK -- `steps_per_call = layers` (one trigger demotes one BLOCK, all its layers).
    //       16x the rate on the shipped 16-layer stack. ⭐⭐ **THIS IS THE ONE THAT WAS LANDED**,
    //       on the owner's 「梯度要记得按块，不能全层下降」, as (b) was always the arm that sentence
    //       argues for.
    //       ⚠⚠ [F1248 kvrate2] **SUPERSEDED, AND THE SENTENCE ABOVE IS KEPT AS THE HISTORY.** The
    //       ruling fixes the UNIT of a move and says nothing about the AMOUNT; `layers` was a
    //       per-model CONSTANT doing the amount's job, and the amount is the gate (MEASURED: 2.68%
    //       of the cells moved, 1.301% of the bytes, both capped by this constant). The unit is
    //       still the block -- what is no longer a constant is HOW MANY blocks one trigger may
    //       move. The derivation, its inputs and its fallback are at the `steps_per_call`
    //       definition below; the (a)/(c) notes that follow are untouched.
    //   (c) the SOLVE -- the selection stops being a walk, so the rate question disappears. OUT OF
    //       SCOPE, AND IT DID NOT LAND HERE: its gate is a measured precision floor and the floor is
    //       still a proxy. Another line owns it; nothing in this file creates it.
    //
    // ⚠ AND ONE MECHANICAL FACT A LATER READER MUST NOT REDISCOVER THE HARD WAY: `steps_per_call`
    // ALONE WOULD HAVE CHANGED NOTHING. The per-step exit used to be `if (one_shot) break;`, i.e.
    // the ONE-SHOT WALK left the cell loop after its first step whatever `steps_per_call` said, and
    // `max_sweeps` is 1 for a one-shot walk -- so setting the rate without moving that exit would
    // have produced a DIFFERENT COMMENT AND IDENTICAL BYTES. Both halves are moved together below,
    // and the two-sided arm reads `steps_per_pass` 1 -> `layers`.
    // =========================================================================================
    const bool one_shot = spec.walk == DescentWalk::OneShot;
    const std::uint64_t max_sweeps = one_shot ? 1ULL : static_cast<std::uint64_t>(kCellModeCount);
    // =========================================================================================
    // ⭐⭐ [F1248 kvrate2] THE TRIGGER'S SIZE IS DERIVED FROM THIS PASS'S OWN RATE GAP.
    // =========================================================================================
    //
    // THE SUPERSEDED LINE IS KEPT, VERBATIM, BECAUSE THE MEASUREMENT THAT KILLED IT IS THE EVIDENCE:
    //     //   [F1239 kvrate] ⭐ (b), AS THE RATE: one trigger, one BLOCK. `layers` steps of the
    //     //   block-major order ARE one block's layers ... ⚠ NO PER-MODEL CONSTANT: a model with 32
    //     //   layers gets a 32-cell block out of this same expression.
    //     const std::uint32_t steps_per_call = one_shot ? layers : 0xFFFFFFFFU;
    // TWO THINGS WERE WRONG WITH IT AND ONLY ONE OF THEM IS THE `layers`. (i) IT IS A CONSTANT
    // DOING A DECISION'S JOB: `layers` is a property of the STACK, and it was being used to answer
    // "how much may move this trigger", which is a property of the BUDGET. (ii) IT MADE THE YIELD A
    // CONSTANT TOO, AND THAT IS THE WHOLE FINDING: MEASURED on F1245's deepest mixed arm
    // (`dl/kvcarry/out/CARRY.tsv`, arm `B_hold_rate`, rate budget R=4500, chain lattice) the run
    // moved **432 of 16,096 cells (2.68%)** over 27 triggers -- 16 cells a trigger, exactly
    // `layers` -- and the byte column closes ARITHMETICALLY as `2.68% x 48.48% = 1.301%` to the
    // digit against `saved_bytes`. So `yield = (fraction of cells moved) x (per-cell saving)` and
    // the fraction was capped by THIS CONSTANT, not by the budget: the same pass printed
    // `budget_satisfiable=yes` with a ceiling at ~45% of its own all-int8 charge while the walk had
    // moved one sixtieth of what that ceiling asked for. A per-model constant can be made bigger,
    // which is why it is the wrong shape; the owner's rule is 「不应该写死任何数字来干预引擎决策」
    // 「不是我们写死，而是引擎自己判断」.
    //
    // ⭐ THE RULE, GIVEN -- NOT A NUMBER. The engine already holds every quantity the question needs,
    // in ITS OWN UNITS, on the pass that is asking:
    //   * `total_bytes - budget_bytes` -- HOW FAR THIS PASS STILL IS from the operator's rate, in
    //     bytes, RIGHT NOW. Signed; `> 0` means the budget still binds. (`total_bytes` is the charge
    //     of `plan.vectors`, i.e. the CARRIED population -- see the note at its initialisation, which
    //     is what makes this subtraction a statement about the object the walk is about to move.)
    //   * `planned_bytes` and `min_possible_bytes` -- the TWO ENDS of this population's OWN travel:
    //     its all-default charge and the cheapest charge it can reach. Their difference is the total
    //     demotion the population can buy, and `cells_total` is the population it is spread over, so
    //     `(planned - min) / cells` is THIS population's own average per-cell travel -- DERIVED, with
    //     no per-model constant anywhere in it.
    //   * `budget_rate_x10000` / `floor_rate_x10000` -- the same two ends in the OPERATOR'S OWN
    //     currency (10000ths of this population's all-int8 charge; R=10000 is int8, the ladder's own
    //     floor is R=2727 = 2.25 b/el). They are the same statement as the two byte columns, in the
    //     unit the RATE arm is spelled in, and they are what makes "derived from the rate" literal
    //     rather than a figure of speech.
    //
    // THE DERIVATION, IN BOTH CURRENCIES, SO IT CAN BE CHECKED IN EITHER:
    //     cells_this_call = ceil( (total_bytes - budget_bytes) * cells_total
    //                             / (planned_bytes - min_possible_bytes) )              [bytes]
    //                     = ceil( (10000 - budget_rate) / (10000 - floor_rate) x cells_total ) [rate]
    // and then ROUNDED UP TO A WHOLE BLOCK, because THE UNIT OF A MOVE IS A BLOCK:
    //     steps_per_call = ceil(cells_this_call / layers) * layers
    // `layers` consecutive steps of this walk's block-major order ARE one block's layers -- the same
    // axis the superseded (b) used -- so the GRANULARITY the owner ruled on is UNTOUCHED: what moved
    // is HOW MANY BLOCKS ONE TRIGGER MAY MOVE, never the granule. (The block unit is exact here for
    // the reason the superseded note gave: with the frequency source ABSENT the sort's key is equal
    // for every cell and the order degenerates to `(block, layer)` lexicographic. `plan.frequency` is
    // printed on the same line, so a run under a future producer can be told apart instead of
    // assumed. GAP-FREQ is unchanged by this line: there is no producer.)
    //
    // WHY IT CONVERGES, AND WHY IT CANNOT RUN AWAY. The count is recomputed against the total the
    // previous triggers produced, so it SHRINKS with the gap. On the shipped shape a first step saves
    // 65,536 total bytes where the average full travel is 98,304, so the estimate OVERSHOOTS and the
    // walk lands at or below the budget; the residual a trigger leaves is bounded by one block,
    // which is the pre-image granularity, and `steps_per_call=0` once the gap is closed (the stop
    // test is the first statement of the cell loop, so a closed gap takes no step at all). Two hard
    // stops that were always here still bound it from above: the budget test, and the end of the
    // order -- a trigger can never move more cells than the window contains, and it never moves a
    // cell twice inside one trigger.
    //
    // WHAT IT DOES NOT DO, NAMED RATHER THAN LEFT TO BE ASSUMED: it does not enlarge the POPULATION
    // (a trigger still moves only cells the caller's window contains), and it does not add TRIGGERS.
    // The number of armed passes is the engine's and (a)'s note above still stands -- "a trigger
    // source the engine does not have". What this closes is the AMOUNT per trigger; the pass count
    // remains the other factor and is reported, not claimed.
    //
    // ⚠ THE FALLBACK IS THE PRE-IMAGE, DELIBERATELY, AND IT IS NOT A NEW CONSTANT. When the
    // subtraction has no meaning -- budget OFF, `blocks == 0`, `layers == 0`, or
    // `planned <= min_possible` so this population can buy no travel at all -- the trigger is ONE
    // BLOCK, i.e. exactly the value that stood here, and `steps_per_call_basis[...]` is not printed,
    // so an UNAVAILABLE measurement cannot masquerade as a rate. In that last case the fallback is
    // also immaterial: when the population can buy no travel, no step has positive currency and the
    // walk moves nothing whatever the count says. The one degenerate-but-reachable case -- a budget
    // already satisfied at the top of the pass -- asks for ZERO blocks, which the stop test turns
    // into zero steps.
    const bool rate_derivable = one_shot && spec.budget_bytes > 0 && layers > 0 && cells > 0 &&
                                plan.planned_bytes > plan.min_possible_bytes;
    std::uint64_t rate_blocks_per_call = 1;   // the pre-image unit: ONE BLOCK (see the fallback note)
    if (rate_derivable) {
        plan.rate_travel_bytes = plan.planned_bytes - plan.min_possible_bytes;
        plan.rate_gap_bytes = plan.total_bytes - spec.budget_bytes;
        if (plan.rate_gap_bytes > 0) {
            // ceil(THE REMAINING GAP x THE POPULATION / THE POPULATION'S OWN TRAVEL), in 128 bits so
            // that no intermediate can leave the range whatever the caller's block count is.
            const unsigned __int128 numerator =
                static_cast<unsigned __int128>(plan.rate_gap_bytes) *
                static_cast<unsigned __int128>(cells);
            const unsigned __int128 denominator =
                static_cast<unsigned __int128>(plan.rate_travel_bytes);
            std::uint64_t cells_this_call =
                static_cast<std::uint64_t>(numerator / denominator) +
                ((numerator % denominator) != 0 ? 1U : 0U);
            if (cells_this_call > cells) { cells_this_call = cells; }
            rate_blocks_per_call = cells_this_call / layers +
                                   ((cells_this_call % layers) != 0 ? 1U : 0U);
        } else {
            rate_blocks_per_call = 0;   // the gap is closed: no block may move, and none needs to
        }
        plan.rate_derived = true;
    }
    plan.rate_blocks_per_call = rate_blocks_per_call;
    // [`uint32_t` BECAUSE THE COLUMN HAS ALWAYS BEEN ONE, AND THE CLAMP IS SPELLED OUT RATHER THAN
    // TRUSTED: `rate_blocks_per_call x layers` is zero only when the gap is closed, and is otherwise
    // at most `cells + layers - 1` after the rounding above, which the order-length stop already
    // bounds. The clamp exists for a caller whose `blocks` is large enough to overflow the CELL
    // product above (`cells` is `std::uint32_t`), and 0xFFFFFFFFU is the walk's own "no per-trigger
    // bound" spelling, so a saturated count degrades to "the trigger may scan the whole order".]
    const std::uint64_t steps_per_call_wide =
        one_shot ? rate_blocks_per_call * static_cast<std::uint64_t>(layers)
                 : 0xFFFFFFFFULL;
    const std::uint32_t steps_per_call =
        steps_per_call_wide > 0xFFFFFFFFULL ? 0xFFFFFFFFU
                                            : static_cast<std::uint32_t>(steps_per_call_wide);
    plan.steps_per_call = steps_per_call;   // [F1239 kvrate] THE RATE, as a reading on the line
    const std::uint64_t hard_max_steps =
        spec.max_steps != 0 ? spec.max_steps
                            : static_cast<std::uint64_t>(cells) * (kCellModeCount - 1) + 1;

    // [F1209 2026-09-29] THE TWO SELECTORS, RESOLVED ONCE PER PLAN AND THEN HELD FIXED, so the walk
    // cannot change chain or ceiling half way down a sweep.
    //   * `ceiling` is CLAMPED into the ladder's own range, exactly as `ladder_level_walk` clamps it:
    //     a caller who typed 9 gets the floor, never a surprise OFF; an unset value gets the floor.
    //   * `ceiling_active` is FALSE when the ceiling is the floor -- and that gate is not an
    //     optimisation, it is ACCOUNTING: at ceiling == floor a cell already at the floor would be
    //     counted `stopped-by-ceiling` instead of `at-floor`, and the refusal histogram is the one
    //     thing this line exists to print. With the gate, the ceiling counters describe ONLY the
    //     ceiling, and a default run's histogram is the pre-image one.
    std::int32_t ceiling = spec.ladder_ceiling_depth;
    if (ceiling < 0) { ceiling = static_cast<std::int32_t>(kLatticeFloorDepth); }
    if (ceiling > static_cast<std::int32_t>(kLatticeFloorDepth)) {
        ceiling = static_cast<std::int32_t>(kLatticeFloorDepth);
    }
    const bool ceiling_active = ceiling < static_cast<std::int32_t>(kLatticeFloorDepth);
    // [F1260 kvplanar] THE SELECTION BECOMES TWO ORDERS ONCE, ABOVE THE FLOOR WALK AND ABOVE THE
    // WALK ITSELF, AND NEITHER CAN CHANGE THEM HALF WAY DOWN -- the same discipline the chain and the
    // ceiling already follow. `order_lattice` is GONE because "which chain" is no longer one
    // boolean: it is two per-plane orders, and the pre-image selections map to two equal orders
    // (asserted at `plane_orders_of`). ⚠ IT IS DECLARED EARLIER THAN IT READS BECAUSE THE FLOOR
    // BELOW IS A FUNCTION OF IT -- a quantity a decision reads must exist before the decision, which
    // is the same rule that moved the floor block up in F1248.
    plan.step_order = spec.step_order;
    plan.plane_order_k = plane_orders.k;
    plan.plane_order_v = plane_orders.v;
    plan.ladder_ceiling_depth = ceiling;

    std::vector<bool> refused_here(cells, false);
    std::vector<bool> age_gated_here(cells, false);   // [F1231] the gate's counter, once per cell
    bool step_bound_hit = false;
    std::uint32_t steps_this_call = 0;
    std::uint32_t next_offset = 0;
    // THE RESUME POINT, AS AN INDEX INTO THE ORDER. A fresh walk starts at 0; a carried one starts
    // where the last trigger stopped (`resumed_offset`), and only the FIRST sweep resumes -- a
    // second sweep is by definition a new level and must re-visit the population from its start.
    const std::uint32_t order_from = plan.resumed_offset;
    for (std::uint64_t sweep = 0; sweep < max_sweeps; ++sweep) {
        bool progressed = false;
        for (std::size_t off = (sweep == 0 ? order_from : 0U); off < order.size(); ++off) {
            const Key& cell = order[off];
            // (4) THE STOP CONDITION, CHECKED BEFORE EVERY STEP: the walk stops the moment the
            // total fits. Cells later in the order therefore never move, which is precisely
            // "how much reached the floor depends on the budget".
            if (plan.total_bytes <= spec.budget_bytes) { break; }
            if (plan.steps >= hard_max_steps) { step_bound_hit = true; break; }
            // (2b) [F1231] R2/R3, AS THE ONE PREDICATE THE OWNER GAVE: `age_admits(block)` is
            // `block + K <= frontier`, i.e. "this block is at least K behind the newest one the
            // caller knows about". A cell the gate refuses is COUNTED AND SKIPPED -- never demoted
            // -- and the counter is per cell, so it reads as a size and not as a visit count.
            // K == 0 IS THE OFF SWITCH and is byte-for-byte the pre-image eligibility (the gate's
            // own contract), which is why an unset K cannot move a reading.
            if (spec.age.keep_recent_blocks != 0 && !spec.age.admits(cell.block)) {
                const std::size_t gated = static_cast<std::size_t>(cell.block) * layers + cell.layer;
                if (!age_gated_here[gated]) {
                    age_gated_here[gated] = true;
                    plan.age_gated_cells += 1;
                }
                continue;
            }

            BlockVector& vector = plan.vectors[cell.block];
            // [F1260 kvplanar] THE CELL'S STATE IS THE PAIR. `here_pair.k` IS `vector.mode[layer]`
            // and `here_pair.v` is the second coordinate; on a vector no pair step has touched they
            // are the same row, which is the pre-image state.
            const CellPair here_pair = vector.pair_at(cell.layer);
            const CellAllowedSet set = allowed_for(cell.layer);
            // (3a) THE RUNG CEILING, CHECKED BEFORE THE STEP. A cell whose CURRENT rung sits at or
            // below the ceiling depth cannot move further, and the refusal is COUNTED under its own
            // name so the line can tell "the ceiling stopped it" from "the ladder ran out". Rungs
            // off the lattice (`bf16`, `nvfp4`) have depth -1 and are NOT subject to the ceiling:
            // the ceiling bounds the LADDER, and this is `ladder_level_walk`'s own rule spelled the
            // same way, because the two walks must agree cell for cell on the lattice chain.
            // Ceiling 0 therefore means "nothing may move", which is what the sweep's second axis
            // needs; with the ceiling at the floor the check is skipped entirely (`ceiling_active`,
            // resolved above) so the default histogram is unchanged.
            if (ceiling_active) {
                // [F1260 kvplanar] THE CEILING BOUNDS THE LADDER, SO IT BOUNDS EACH PLANE'S OWN
                // DEPTH. On a diagonal cell the two depths are the same number and this is the
                // pre-image test verbatim; on a two-sided selection a plane may be stopped by the
                // ceiling while the other continues, which is one more way the two coordinates part
                // -- and it is the ONLY one that can stop a plane ABOVE its floor on purpose.
                const std::int32_t depth_k = ladder_depth(here_pair.k);
                const std::int32_t depth_v = ladder_depth(here_pair.v);
                const bool k_blocked = depth_k >= 0 && depth_k >= ceiling;
                const bool v_blocked = depth_v >= 0 && depth_v >= ceiling;
                if (k_blocked && v_blocked) {
                    const std::size_t flat =
                        static_cast<std::size_t>(cell.block) * layers + cell.layer;
                    if (!refused_here[flat]) {
                        refused_here[flat] = true;
                        plan.refused_steps += 1;
                        plan.cells_stopped_by_ceiling += 1;
                    }
                    continue;
                }
            }
            // (3b) THE STEP: one cell, ONE rung PER PLANE, each plane on its OWN order (section B2).
            // Both chains are one rung at a time; the difference is WHICH rung is "below" (the
            // priciest one, or the next one of the owner's 4/3/2 lattice), and both return a delta
            // priced on the PLANE ruler -- `cell_next_tier` always did, and `cell_next_cheaper` does
            // as of F1209's site-3 fix. A step priced on any other ruler cannot reach a budget
            // denominated in this one, which is what `steps=0` measured.
            //
            // ⚠ AND A STEP IS NOW A PAIR OF PER-PLANE STEPS, with the two orders taken from the
            // plan's selection and the RESULT GATED by `cell_pair_realizable`. On a diagonal cell
            // under either pre-image selection this call IS the 1-D call (six static_asserts above);
            // on a two-sided selection it is the deployed family's chain (the third static_assert).
            const CellPairStep pstep =
                cell_next_pair(set, here_pair, plane_orders.k, plane_orders.v);
            const CellStep step = pstep.as_cell_step();
            if (!step.legal()) {
                if (!refused_here[static_cast<std::size_t>(cell.block) * layers + cell.layer]) {
                    refused_here[static_cast<std::size_t>(cell.block) * layers + cell.layer] = true;
                    plan.refused_steps += 1;
                    switch (step.refusal) {
                    case StepRefusal::AlreadyAtFloor: plan.cells_at_floor += 1; break;
                    case StepRefusal::RungBytesUnknown: plan.cells_blocked_by_unknown += 1; break;
                    case StepRefusal::NoByteCurrency: plan.cells_blocked_by_currency += 1; break;
                    default: break;
                    }
                    // THE SECOND FACTS. A cell can be blocked by G6 and G7 at once, and the
                    // switch above records only the precedence answer. These two counters are
                    // what stop the accounting line from under-reporting either gap.
                    if (step.blocked_by_unpriced_rung) { plan.cells_with_unpriced_rung += 1; }
                    if (step.blocked_by_same_record) { plan.cells_with_same_record_rung += 1; }
                    // [F1231, F-5 site 2] THE PLANE FACT, AS ITS OWN COLUMN, because the refusal
                    // is made on it: printing only the record count would let a reader reconcile
                    // the refusal histogram against the wrong number.
                    if (step.blocked_by_same_plane) { plan.cells_with_same_plane_rung += 1; }
                    // THE SWITCH ABOVE IS DELIBERATELY NON-EXHAUSTIVE AND THE `default` IS
                    // SPELLED OUT, because `StepRefusal::None` is unreachable here (`legal()`
                    // gated it) and naming it would make the arms look like four cases when
                    // there are three. THIS IS A `-Wswitch-enum` HAZARD: this tree escalates
                    // `-Wswitch`/`-Wswitch-enum` in places, so a landing must confirm the
                    // escalation does not reach this file, or add the missing arms.
                }
                continue;
            }
            // (5) STRICTLY DECREASING, ENFORCED HERE AND NOT ASSUMED. A step whose delta is not
            // positive would make the termination argument false, so it is refused rather than
            // taken -- this is the one line that makes "<= #cells * #rungs" true.
            if (step.bytes_saved <= 0) {
                plan.refused_steps += 1;
                continue;
            }
            // [F1260 kvplanar] THE TWO-SIDED WRITE -- the only place the two coordinates can part.
            // `set()` (the pre-image write, both planes to one row) is not used here any more, and
            // the step's own `to` is the pair. On a diagonal step this is exactly `set(layer, m)`.
            vector.set_pair(cell.layer, pstep.to);
            // [F1231] R4, FIRST HALF: THE RESUME POSITION. It is the cell AFTER this one, so the
            // next trigger continues instead of repeating. The one-shot EXIT is the second half
            // and sits at the END of this step's body -- AFTER the subtraction, because a break
            // here would leave `total_bytes` un-decremented and `steps` at zero, which the file's
            // own `totals_agree` check caught the first time this was written the other way round.
            next_offset = static_cast<std::uint32_t>(off) + 1U;
            steps_this_call += 1;
            // (5b) THE SUBTRACTION, IN THE TOTAL'S OWN UNIT. DEVELOPMENT-3, AND HERE IS THE
            // DERIVATION -- READ IT BEFORE TOUCHING THIS LINE.
            //
            // WHAT THE TOTAL IS. `plan.planned_bytes` is `blocks * base_charge.bytes`, and
            // `cell_vector_charge` defines that charge as (its own `charge.bytes` line)
            //     bytes(V) = 2 * kv_heads * sum_over_layers( plane_bytes(V[l]) )
            // so `plan.total_bytes` is a `2 * kv_heads`-SCALED SUM over layers and blocks.
            //
            // WHAT ONE STEP IS. `step.bytes_saved` is ONE PLANE OF ONE LAYER -- the delta a
            // demotion gives back for a single (block, layer) cell. Demoting one cell therefore
            // changes `sum_over_layers( plane_bytes )` by exactly that delta, and changes the TOTAL
            // by
            //     delta_total(step) = 2 * kv_heads * step.bytes_saved
            // (`ladder_step_total_delta`, kv_tier_ladder.h -- the one spelling of the factor, so a
            // call site cannot forget it).
            //
            // SO THE LIVE SPELLING `total_bytes -= step.bytes_saved` WAS SHORT BY THE FACTOR
            // `2 * kv_heads` -- 8 on the shipped 4-head stack -- AND THE ARITHMETIC IS EXACT AT THE
            // SHIPPED SHAPE (48 blocks x 16 layers x 4 kv_heads = 768 cells):
            //     planned, all int8   = 48 * 2*4 * 16 * 16,896 = 48 * 2,162,688 = 103,809,024
            //     all cells at floor  = 48 * 2*4 * 16 *  4,608 = 48 *   589,824 =  28,311,552
            //     TRUE saving  = 103,809,024 - 28,311,552 = 75,497,472 = 768 * 98,304
            //       (98,304 = 2*4 * 12,288, and 12,288 = 16,896 - 4,608 is the runway)
            //     the UNSCALED saving = 768 * (8,192 + 2,048 + 2,048) = 768 * 12,288 = 9,437,184
            //     75,497,472 / 9,437,184 = 8 EXACTLY  ==> the factor is `2 * kv_heads`, nothing else
            //     unscaled total = 103,809,024 - 9,437,184 = 94,371,840 (NOT the all-floor total)
            //     scaled total   = 103,809,024 - 75,497,472 = 28,311,552 (EXACTLY the all-floor
            //                       total computed ABOVE from the rung table -- two independent
            //                       routes to one number, which is the check that makes this a
            //                       derivation rather than an assertion)
            //
            // ⚠ AND THE `3x` FIGURE IN `kv_tier_ladder.h`'s OWN RECORDS IS NOT A FACTOR OF 3. Its
            // proof program modelled "the live spelling" as `planned - cells * 3 * 8192`, i.e. it
            // charged ALL THREE steps the FIRST step's 8,192 (24,576 per cell) instead of
            // 8,192 + 2,048 + 2,048 = 12,288 -- a double-count inside the MODEL, not a scaling of
            // the live line. That model lands on 103,809,024 - 18,874,368 = 84,934,656, which is
            // 3 x 28,311,552 by numerical coincidence and NOT by any factor of the engine. The same
            // header's prose quotes the CORRECT unscaled figure (94,371,840, "= 103,809,024 -
            // 9,437,184"), so that file's two records disagree with each other; the derivation
            // above is the one read out of the SOURCE, and it is the one this line implements. The
            // factor is `2 * kv_heads`; there is no other factor anywhere in this family.
            //
            // WHY IT MATTERS AND NOT JUST FOR TIDINESS: while the subtraction was short, the
            // stop condition `total_bytes <= budget` could not be reached at a budget that WAS
            // reachable -- `saved_bytes` was 8x low -- so every "the budget does not bind" reading
            // taken from a MOVING walk would have been wrong by the same factor. Now the stop reads
            // the quantity the steps actually move, which is the same property site 3's fix gives
            // the step itself: ONE RULER, END TO END.
            plan.total_bytes -= ladder_step_total_delta(step, spec.kv_heads);
            plan.steps += 1;
            progressed = true;
            // [F1231] R4, SECOND HALF: ONE TRIGGER => ONE DEMOTION, AND THE STEP IS NOW COMPLETE
            // (the rung is set, the total is decremented, the counters are up). A `OneShot` call
            // leaves the cell loop HERE. When the scan instead reaches the end of the population
            // with no step taken, `next_offset` stays 0: that is one LEVEL finished, and the next
            // trigger starts the next level from the default-carried modes.
            // [F1239 kvrate] ⭐ THE TRIGGER IS SPENT WHEN `steps_per_call` STEPS ARE TAKEN, not
            // after the first one. The RULE is unchanged and it is the owner's -- one trigger, one
            // demotion (R4) -- what moved is the DEMOTION'S UNIT, cell -> block, which is his
            // 「梯度要记得按块」. Under the pre-image one-shot rate this is exactly the old
            // `if (one_shot) break;`; under (b) it lets ONE trigger spend one block's layers. The
            // position is deliberate and unchanged: AFTER the subtraction, because a break placed
            // before it leaves `total_bytes` un-decremented and `steps` at zero -- the file's own
            // `totals_agree` check caught that the first time this was written the other way round.
            if (steps_this_call >= steps_per_call) { break; }
        }
        plan.sweeps += 1;
        if (steps_this_call >= steps_per_call) { break; }   // [F1231] R4: the trigger is spent
        if (plan.total_bytes <= spec.budget_bytes) { break; }
        if (!progressed) { break; }
        if (step_bound_hit) { break; }
    }
    if (one_shot) { plan.sweeps = 0; }   // ONE CALL IS NOT A SWEEP, and the column says so

    // THE CHARGE INPUT, BLOCK BY BLOCK. Computed from the final vectors and NOT accumulated from
    // the running total, so the number the stage charges and the number this plan reports are the
    // same arithmetic on the same object and cannot drift by a bookkeeping slip.
    // [F1209 2026-09-29] AND THE SAME LOOP NOW RECONCILES THE TWO TOTALS. `total_bytes` above is
    // INCREMENTAL (the stop condition must be readable without re-summing 768 charges per step) and
    // this sum is AUTHORITATIVE (it is the charge of the object that will be handed to the stage).
    // DEVELOPMENT-3 WAS a drift between exactly these two spellings, so the comparison is the
    // instrument for the fix and it is a READING rather than an assert: a disagreement is printed
    // by `describe()` instead of aborting a pass that is otherwise well defined.
    std::int64_t recomputed_total = 0;
    for (std::uint32_t block = 0; block < blocks; ++block) {
        const CellCharge charge = cell_vector_charge(plan.vectors[block], spec.kv_heads);
        plan.block_bytes[block] = charge.bytes;
        recomputed_total += charge.bytes;
    }
    plan.totals_agree = recomputed_total == plan.total_bytes;
    plan.saved_bytes = plan.planned_bytes - plan.total_bytes;
    // =========================================================================================
    // [F1248 kvrate2] ⚠ THE FLOOR / SATISFIABILITY / RATE BLOCK USED TO STAND HERE, BESIDE THE
    // TOTALS, AND IT HAS MOVED ABOVE THE WALK. It moved because the walk's own TRIGGER SIZE is now
    // derived from it (see the derivation note at the `steps_per_call` definition), and a quantity a
    // decision reads must exist BEFORE the decision -- the same rule that makes the budget a field
    // of the spec rather than a read taken behind the caller's back.
    //
    // THE MOVE IS VALUE-PRESERVING AND IT IS CHECKED, NOT ARGUED: not one input that block reads is
    // a walk output (`allowed_for`, `kCellRungs`, `blocks`, `spec.kv_heads`, `spec.budget_bytes`,
    // `base_charge.bytes` -- all of them settled before the first step), so the three numbers it
    // fills are the same numbers on the same pass. And the claim is FALSIFIABLE rather than
    // rhetorical: all three are printed by `describe()` (`min_possible_bytes=`,
    // `budget_rate_x10000=`, `floor_rate_x10000=`, and `budget_satisfiable=` derived from the first),
    // so the plan-level pre-image arm diffs them pre-change vs post-change and a drift introduced by
    // the move would appear as a DIFF on every spec that carries a live budget.
    // [planes 2026-09-29] THE TWO DISTRIBUTIONS BEHIND THE GRANULARITY PROOF. Computed from the
    // final vectors, beside `cells_demoted`, so the four numbers printed by `describe()` and the
    // count they are about come from the same pass and cannot drift.
    std::vector<std::uint32_t> demoted_per_layer(layers, 0);
    std::vector<std::uint32_t> demoted_per_block(blocks, 0);
    plan.deepest_pair = CellPair{base.mode[0], base.mode[0]};
    plan.deepest_pair_bytes = cell_pair_bytes(plan.deepest_pair);
    for (std::uint32_t b = 0; b < blocks; ++b) {
        const BlockVector& vector = plan.vectors[b];
        for (std::uint32_t layer = 0; layer < vector.layers; ++layer) {
            const std::uint32_t index = static_cast<std::uint32_t>(vector.mode[layer]);
            // [F1260 kvplanar] ⭐ THE CELL IS DEMOTED WHEN **EITHER** PLANE MOVED. The pre-image
            // test asked the K coordinate alone, which was the same question while the two planes
            // could not differ. On a diagonal plan this line and the pre-image line are the same
            // comparison; on a two-sided plan it is the only one that counts a V-only move.
            const CellPair pair_here = vector.pair_at(layer);
            const std::uint32_t pair_index = index * kCellModeCount +
                                             static_cast<std::uint32_t>(pair_here.v);
            if (pair_here.k != base.mode[layer] || pair_here.v != base.mode[layer]) {
                plan.cells_demoted += 1;
                demoted_per_layer[layer] += 1;
                demoted_per_block[b] += 1;
                // [F1231] R2/R3's INSTRUMENT. The block index IS this tree's age proxy, so the
                // RANGE of demoted blocks is the age reading: under R2 its upper end must sit below
                // the newest block of the population, and under R3 the newest block must not be in
                // it at all. `b` ascends, so the first hit is the oldest and each hit resets the
                // newest. The compare is against the CARRY as much as against the default: a cell
                // that arrived at int8 and stayed there is not demoted, and a cell carried down
                // from an earlier trigger still counts -- the range is a CENSUS, not a delta.
                plan.demoted_newest_block = b;
                if (!plan.any_demoted) { plan.demoted_oldest_block = b; }
                plan.any_demoted = true;
            }
            plan.final_rung_histogram[index] += 1;
            // ⭐ [F1260 kvplanar] THE SECOND COLUMN, BESIDE THE FIRST, FROM THE SAME PASS AND THE
            // SAME OBJECT. `v_rung_histogram` is what `final_rung_histogram` would have been if the
            // V plane were the cell -- which on a diagonal plan it IS, value for value.
            plan.v_rung_histogram[static_cast<std::uint32_t>(pair_here.v)] += 1;
            plan.pair_histogram[pair_index] += 1;
            if (pair_here.k != pair_here.v) {
                plan.cells_asymmetric += 1;
                plan.plane_axis_live = true;
                if (cell_pair_read_side(pair_here) == E8KvPlaneReadSide::Reserved) {
                    plan.cells_pair_reserved += 1;
                }
            }
            const std::int32_t pair_bytes_here = cell_pair_bytes(pair_here);
            if (pair_bytes_here < plan.deepest_pair_bytes) {
                plan.deepest_pair = pair_here;
                plan.deepest_pair_bytes = pair_bytes_here;
            }
        }
    }
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        if (demoted_per_layer[layer] != 0) { plan.demoted_layers += 1; }
        if (demoted_per_layer[layer] > plan.demoted_max_per_layer) {
            plan.demoted_max_per_layer = demoted_per_layer[layer];
        }
    }
    for (std::uint32_t b = 0; b < blocks; ++b) {
        if (demoted_per_block[b] != 0) { plan.demoted_blocks += 1; }
        if (demoted_per_block[b] > plan.demoted_max_per_block) {
            plan.demoted_max_per_block = demoted_per_block[b];
        }
    }
    // [F1231] THE DRIVE, IN THE ONE-SHOT WORLD, AND THEN THE WRITE-BACK THAT MAKES THE LAW A
    // READING RATHER THAN A CLAIM. The order of the tests is the order of the questions: did the
    // caller's budget get met (the stop condition, unchanged), did the hard bound fire, and -- new
    // -- did THIS TRIGGER take its one step or find nothing to move. The pre-image
    // `NoMovableCell` value is now unreachable from a one-shot call, which is the point: it used to
    // be printed by a walk that had just moved the whole population.
    if (step_bound_hit) {
        plan.drive = DescentDrive::StepBoundReached;
    } else if (plan.total_bytes <= spec.budget_bytes) {
        plan.drive = DescentDrive::StoppedByBudget;
    } else if (one_shot) {
        plan.drive = steps_this_call == 0 ? DescentDrive::OneShotNothingMovable
                                         : DescentDrive::OneShotStepTaken;
    } else {
        plan.drive = DescentDrive::NoMovableCell;
    }
    plan.walk = spec.walk;
    plan.next_offset = next_offset;
    plan.population_origin_seen = spec.population_origin;
    if (spec.cursor != nullptr) {
        DescentCursor& cursor = *spec.cursor;
        // [F1245 kvcarry] THE RESET IS NOW ABOUT THE OBJECT, NOT ABOUT THE WINDOW. Pre-image this
        // read `blocks() != blocks || layers() != layers` and threw the WHOLE cursor away -- the
        // modes AND the counters -- which, together with the read-side refusal, is what made the
        // plan print `triggers=1 steps_taken=16` on all 27 passes of a single run (F1244,
        // `dl/cmpfire/out/speed/B_def/stderr.txt`): every trigger was the FIRST trigger.
        // A BLOCK-COUNT CHANGE IS NOT THAT EVENT. The modes have just been re-anchored by page
        // identity at the top of this call (the block above), so `plan.vectors` IS the correct
        // carrier for THIS window -- writing it into a cursor reset to nothing would throw away the
        // very thing the re-anchor just reconstructed, and zeroing `triggers_fired`/`steps_taken`
        // would hide it. A LAYER-COUNT CHANGE still resets, because that is a different OBJECT (the
        // stack's own axis) and the position's order index is not comparable across it.
        if (spec.cursor->initialised() && spec.cursor->layers() != layers) {
            cursor = DescentCursor{};   // a different stack: the modes and the position are not ours
        }
        cursor.vectors = plan.vectors;          // HALF ONE: the modes, carried
        cursor.next_offset = next_offset;       // HALF TWO: the position, carried
        cursor.order_blocks = blocks;
        cursor.order_layers = layers;
        cursor.origin = spec.population_origin;   // [F1231 window] the identity, carried too
        cursor.triggers_fired += 1;
        cursor.steps_taken += steps_this_call;
        if (steps_this_call == 0) { cursor.refused_triggers += 1; }
        cursor.cells_demoted = plan.cells_demoted;
        cursor.blocks_absorbed = blocks;         // every block of a live plan entered at the default
        cursor.last_step_block = plan.any_demoted ? plan.demoted_newest_block : 0U;
        cursor.last_trigger = steps_this_call == 0 ? DescentTrigger::None
                                                  : DescentTrigger::BudgetCrossed;
        cursor.any_demoted = plan.any_demoted;
        cursor.demoted_oldest_block = plan.demoted_oldest_block;
        cursor.demoted_newest_block = plan.demoted_newest_block;
        cursor.total_bytes = plan.total_bytes;
        // ONE TRIGGER => ONE DEMOTION, EVALUATED ON THE REAL OBJECT. `law_holds()` is
        // `steps_taken <= triggers_fired`, so it can only fail by a trigger taking more than one
        // step -- which is exactly the pre-image sweep, and why the arm can show it failing.
        plan.triggers_fired = cursor.triggers_fired;
        plan.steps_taken = cursor.steps_taken;
        plan.law_holds = cursor.law_holds();
    }
    // [F1251 kvspend] THE DEVICE FACE, AT THE ONE EXIT THAT HAS A WALK'S VECTORS. The OFF path
    // returns above with the default vector, where `saved_bytes` is 0 by construction and the
    // realizable part of it is 0 for the same reason -- so there is nothing there to fill and the
    // pre-image path is untouched. Here the vectors are final, the charge was just recomputed from
    // them, and this reads the SAME object: it is a reading of the decision, not a second decision.
    descent_fill_device_face(plan, base, spec.kv_heads);
    // [F1255 kvaxisA] THE THIRD AXIS IS PRICED ON THE SAME FINAL VECTORS, at the same exit, for the
    // same reason: this is a reading of the decision, not a second decision.
    descent_fill_axis3_face(plan, base, spec.kv_heads, layers, spec.narrow_pages_per_layer);
    return plan;
}

// The stage's charge input, as the stage wants it (`BlockChargeTable`, defined in
// kv_cell_modes.h beside the carrier it is derived from) is what this plan hands over:
//     table.bytes  = plan.block_bytes.data()
//     table.origin = the first block index the plan covers
//     table.count  = plan.block_bytes.size()
// A block outside `[origin, origin + count)` has no plan entry and therefore charges the SCALAR
// charge -- the pre-image value -- which is what keeps a partially planned pass legal.

} // namespace ninfer::product

// =================================================================================================
// [F1240 kvsolve] THE INCLUDE GOES HERE, AND THE POSITION IS THE CONTRACT -- NOT AN ACCIDENT OF
// LAYOUT. `product/kv_cell_alloc_solve.h` consumes this header's types (`DescentSpec`,
// `BlockVector`, `BlockDescentPlan`, `block_descent_plan`), so it MUST be textually after every one
// of their definitions, i.e. after the closing brace above; and it DECLARES NO BACK INCLUDE of its
// own, so the edge is a DAG in exactly one direction. The same file then defines
// `block_alloc_plan(spec, table)` -- THE ONE CALL SITE THAT SELECTS THE SOLVE -- plus
// `descent_walk_from_env()` and `cell_alloc_costs_from_env()`.
//
// MEASURED CONSEQUENCE OF THIS POSITION, AND IT IS THE REASON THE EDIT IS SAFE: because the include
// sits after `block_descent_plan` and after the namespace's brace, nothing ABOVE it can mention the
// solve, so the only new name any earlier line of this file uses is the two enumerators appended to
// `DescentWalk` and `DescentDrive` -- both with a value above every pre-existing one, so no existing
// switch, comparison or default-constructed value changes meaning. A caller that never names
// `DescentWalk::Solve` gets this file's behaviour character for character: `walk` keeps its
// `OneShot` default, `block_alloc_plan` forwards to `block_descent_plan` UNCHANGED, and
// `alloc_note` is empty so `describe()` appends nothing.
#include "product/kv_cell_alloc_solve.h"
