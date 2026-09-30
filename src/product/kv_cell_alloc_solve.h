#pragma once

// =============================================================================================
// F1240 kvsolve -- THE STATELESS PER-CELL ALLOCATION SOLVE.
//
// ⚠ INCLUDE DIRECTION, AND IT IS LOAD-BEARING. This header is included AT THE BOTTOM of
// `product/kv_block_descent.h`, because it consumes `DescentSpec`, `BlockVector`,
// `BlockDescentPlan` and `block_descent_plan`, which that header defines. It therefore declares no
// `#include "product/..."` of its own -- a back include would close a cycle. The precedent is in
// the tree: F1209 removed `kv_tier_ladder.h`'s back include of the descent header for exactly this
// reason, and the two files are a DAG. A reader who wants to know why this file has no product
// include is holding the answer.
//
// =============================================================================================
// 1. WHY THIS EXISTS -- TWO INDEPENDENT FINDINGS, ONE ARCHITECTURE
// =============================================================================================
//
// (A) THE ONE-RUNG-AT-A-TIME RULE WAS A TERMINATION DEVICE FOR AN INCREMENTAL WALK, AND NOTHING
//     ELSE. The walk (`block_descent_plan`) must bound itself because it MODIFIES a carrier: it
//     carries modes and a position through `DescentCursor`, so a step taken in pass N is visible to
//     pass N+1 and the loop needs a reason to stop. Its own source says so: "every accepted step
//     strictly DECREASES the total by >= 1 byte and every cell has finitely many rungs, so
//     steps <= #cells * (#rungs - 1)". A SOLVED selection never iterates -- it computes the whole
//     allocation from its inputs in one pass -- so there is nothing to bound. The rule's own
//     justification does not survive the change of shape, which is why a solve may take any number
//     of rungs in one call without reopening the argument.
//
// (B) A SOLVE NEEDS NO CARRY AT ALL. MEASURED on this engine: the carry is REFUSED on 24 of 24
//     passes (`carry=carried` then `REFUSED(shape-mismatch)` x23), `deepest_per_layer=[1 0 0 ...]`
//     on EVERY pass, `rungs[... nvfp4=1]` on every pass, never 2 -- i.e. 24 INDEPENDENT all-int8
//     restarts, each demoting ONE cell, nothing accumulating. The reason is structural, not a bug:
//     this engine's windows CHURN -- a ~50-page population advancing ~48 pages per pass, i.e. ~2
//     pages of overlap -- so a per-population carry can never make a per-cell allocation persist,
//     and the durable knobs in this engine are per-STACK (the storage dtype table). A solve has no
//     "position", so a churning window cannot invalidate it. THE SHAPE THAT FITS THIS ENGINE IS
//     THEREFORE: SOLVE AFRESH, EVERY PASS, FROM THE MEASUREMENT.
//
// ⇒ A STATELESS ALLOCATION, SOLVED FRESH EACH PASS, IS THE ONLY SHAPE THAT FITS THIS ENGINE.
//
// =============================================================================================
// 2. WHAT IT MUST SATISFY, AND WHERE EACH CONSTRAINT LANDS IN THIS FILE
// =============================================================================================
//
//   * ONE MEASURED PRECISION FLOOR IS THE ONLY CONSTRAINT. `CellAllocCostTable::floor_nats` is a
//     scalar with a `floor_measured` bit. There is no second quality knob in this file, and no
//     per-rung quality constant anywhere in it (see (4)).
//   * MAXIMISE AGGREGATE SPEED SUBJECT TO IT. The objective is the aggregate charge given back,
//     `sum over cells of the rung deltas` in `2 * kv_heads * plane_bytes` -- the engine's own unit,
//     read from `ladder_step_total_delta`. It is NOT a sum of per-tier speeds: see (4).
//   * ⚠ 「不能放任，否则精度可能无法确定」 -- AN INFEASIBLE CASE REFUSES BY NAME. Six named
//     outcomes, all in `AllocSolveOutcome`. THE ONE THAT MATTERS:
//     `RefusedBudgetUnreachableWithinFloor` -- the byte budget demands MORE demotion than the
//     measured floor permits. The pre-image shape would simply demote past the floor and print a
//     byte total; this file returns the STATUS QUO vector instead, names the refusal, and prints
//     the deficit, because a silent compromise here is exactly "the precision may be
//     undeterminable".
//   * THE ALLOCATION UNIT IS THE CELL. The loop addresses `(block, ordinal)`; `BlockVector::set`
//     is called once per cell; the sort key is a cell.
//   * ONE-WAY ONLY. A cell's candidate set is the PREFIX FAMILY of its own chain from its default
//     rung, the emitted plan is compared against the default vector to count `cells_demoted`, and
//     the greedy carries an explicit PREFIX GUARD -- so "never promotes" is a property of the
//     emitted plan and of the loop, readable in both.
//   * OLDEST-FIRST AS A TIE-BREAK AMONG CELLS INDIFFERENT TO THE FLOOR. "Indifferent to the floor"
//     is spelled exactly: an equal marginal ratio. The sort is `(ratio, block, ordinal)` ASCENDING
//     in `block`, and the block index IS this engine's age axis (`kv_block_descent.h`'s own words:
//     "the block index IS this tree's age proxy"). ⚠ 「没有必要拘泥于局部速度」 -- the objective is
//     aggregate: the sort orders WHO IS CONSIDERED FIRST, never how much any one cell gives back.
//   * ⭐ A RULE THAT IS POSITION-BLIND. See section 7. The decision contains no rule that reads a
//     cell's position within its block, and the property is CHECKABLE by text, not asserted.
//   * ⭐⭐ 少写死多适配 -- "ADDING A MODEL MUST NOT REQUIRE TOUCHING KV CODE OR KV CONSTANTS."
//     Everything model-specific arrives through `CellAllocCostTable` (a VALUE) and nothing in this
//     file reads a rung constant to make a decision. THE STRONG FORM IS IMPLEMENTED AS A
//     CALL-THROUGH, not as a special case: an empty table -- or an OFF budget -- makes
//     `cell_alloc_solve` return `block_descent_plan(spec)` and touch nothing (section 5). The
//     identity is BY CONSTRUCTION -- the same function, called with the same spec, which that
//     header's own doctrine says is a pure function of its spec -- and it is proved host-side bit
//     for bit.
//
// =============================================================================================
// 3. THE REFERENCE IMPLEMENTATIONS THIS SHAPE COMES FROM, AND THE ONES REJECTED
// =============================================================================================
//
// READ DEPTH IS MARKED PER ENTRY. This section is a LICENCE TO READ, not a claim of novelty: the
// algorithmic content below is the classical marginal-return family, and the citations say which
// member and why.
//
//   * Fox 1966 (marginal-return analysis), reached through Shoham & Gersho 1988, "Efficient Bit
//     Allocation for an Arbitrary Set of Quantizers", IEEE TASSP 36(9):1445-1453, DOI
//     10.1109/29.90373. [READ DEPTH: search summary + the precondition as quoted by two secondary
//     reads; NOT read in full text.] THIS IS THE SHAPE USED HERE: allocate by MARGINAL RETURN --
//     rank candidate steps by (quality given up) / (bytes given back) and spend the quality budget
//     from the cheapest end. Its own stated precondition is the one that decides this file's
//     honesty: Fox's greedy assumes every rate-quality function is CONVEX, and Shoham-Gersho's
//     contribution is that with an arbitrary (non-convex) set the greedy needs the CONVEX HULL and
//     a Lagrangian. ⭐ THIS FILE IMPLEMENTS BOTH HALVES: the hull (so a non-convex measured table
//     cannot make the greedy wrong without the line saying so) and the VERIFIED precondition (so
//     convexity is MEASURED on the table, never assumed).
//   * Huang & Schultheiss 1963 (the original Lagrangian/truncation result for discrete allocation),
//     and BFOS (Breiman-Friedman-Olshen-Stone's tree-structured pruning by marginal return).
//     [READ DEPTH: cited through Shoham-Gersho's own framing; NOT read in full.] They supply the
//     argument for WHY a Lagrangian is the right relaxation of a single-constraint allocation: the
//     optimum of the relaxation is attained at a breakpoint of the measured staircase, so the dual
//     can be swept in ONE pass over the sorted edges -- which is what makes the bound in (d)
//     affordable rather than a second DP.
//   * He & Mitra 2002, "Optimum bit allocation and accurate rate control for video coding via
//     rho-domain source modeling", IEEE TCSVT 12(10):840-849, DOI 10.1109/TCSVT.2002.804883.
//     [READ DEPTH: abstract + the closed-form result, read in the search summary.] ⭐ REJECTED AS A
//     TEMPLATE, CITED FOR ONE THING. It is the cleanest statement of the family's promise -- "let
//     the measurement decide, hardcode nothing" -- and its closed form
//     `B_i = w_i N_i ln(sigma_i^2 / w_i) + [w_i N_i / sum w_i N_i] * (B - sum w_i N_i ln(...))` is
//     exactly the Lagrangian solution of section 6 with the multiplier determined by the budget.
//     IT IS REJECTED as a template because that closed form is a function of a FITTED MODEL
//     (`theta` from a linear rate model, `alpha` from an exponential distortion model) -- and the
//     same class of move is FORBIDDEN here by measurement: this project refuted the tabled quality
//     column four independent ways, including that a SUM of the table's own units is NOT sufficient
//     (two arms at the same SUM=240 differ by 0.21765 nats). A fitted transfer function is a
//     STRONGER assumption than the sum it would replace. SO THE MULTIPLIER IS KEPT AND THE MODEL IS
//     DROPPED: it is found by SWEEPING THE MEASURED STAIRCASE instead of by inverting a closed
//     form -- same mathematics, no fitted parameters, and the error is the table's own.
//   * RDKV, arXiv 2605.08317 [READ DEPTH: abstract, read by THIS line; a sibling line read it in
//     full and recorded that it allocates per (position-group, head), runs ONCE AFTER PREFILL, and
//     leaves streaming re-budgeting as future work with no duty bound]. ⭐ REJECTED AS A REFERENCE
//     FOR THIS CASE, AND ITS KEY ASSUMPTION IS THE ONE THIS PROJECT FALSIFIED. Three reasons, each
//     checkable: (i) it runs once after prefill, so it says nothing about the incremental case,
//     which is the case here; (ii) its bits range "from full precision down to zero bits guided by
//     reverse water-filling", i.e. its decision variable includes ZERO BITS -- not reachable from
//     this engine's ladder, and its unit is not this engine's charge; (iii) ⭐ ITS SEPARABILITY
//     ASSUMPTION IS EXACTLY WHAT THIS PROJECT'S MEASUREMENT REFUTED (the SUM is not sufficient:
//     two arms at equal SUM=240 differ by 0.21765 nats), and this file REPLACES it with
//     `placement_margin_nats` (section 4) rather than inheriting it.
//   * HAWQ-V2 arXiv 1911.03852 [READ DEPTH: abstract only; the allocation PROCEDURE (greedy vs
//     knapsack vs Lagrangian) is NOT in the abstract]. ⚠ REJECTED AS AN ALGORITHM SOURCE for the
//     ordinary reason: the abstract states a Pareto-frontier method over Hessian-trace sensitivity
//     without stating the search, and this file does not cite a procedure it has not read. Its
//     Hessian-trace metric IS the right family for a measured sensitivity -- it is a MEASUREMENT,
//     not a per-tier constant -- so the table here can be fed from it.
//   * KVTuner arXiv 2502.04420 [READ DEPTH: abstract + method summary]. ⚠ REJECTED AS A TEMPLATE:
//     its granularity is per-position K/V PRECISION PAIRS found by an OFFLINE multi-objective
//     search and then frozen for online inference -- the opposite of a stateless per-pass solve --
//     and its "online decision-making has high overhead" finding is a statement about ITS search
//     cost, not about this one's (this one is one sort of at most ~4,000 edges). ⚠ ITS
//     K-MORE-SENSITIVE-THAN-V FINDING IS NOT RELIED ON HERE AT ALL: this engine's charge is
//     `2 * kv_heads * plane`, i.e. K and V are ONE cell (`kv_cell_modes.h`'s own note: "the 2 IS
//     the K and V planes"), so a K/V split is not representable in the decision variable.
//   * SqueezeAttention arXiv 2404.04793 [READ DEPTH: abstract + the budget sentence]. ⚠ REJECTED,
//     AND ITS REJECTION IS THE MOST INSTRUCTIVE OF THE FIVE: it gives each position-group a
//     DIFFERENT BUDGET on the fly, from a SIMILARITY-based measurement of prompt differences.
//     ⭐ THIS TREE FORBIDS THAT, AND AS A PROHIBITION OF SHAPE RATHER THAN OF TASTE:
//     `sum_dir_determined.h` -- "A similarity score, a distance, a cosine, a top-k rank or any
//     threshold may NOT produce `DETERMINED` ... there is no parameter a score could arrive
//     through", and `CellFrequencySource` is built so "there is nothing in this struct a distance
//     or a threshold could be stored in". A TABLE OF MEASURED COSTS is not a score and does not
//     produce a verdict -- it only spends a budget the caller declared -- which is why THIS file
//     may read one and SqueezeAttention's mechanism may not enter through it.
//   * BaKlaVa arXiv 2502.13176 [READ DEPTH: NONE -- the fetch FAILED TWICE (ETIMEDOUT) on the
//     sibling line that ranked it. ⚠ ITS QWEN2.5 RESULT IS UNVERIFIED AND NOTHING HERE RELIES ON
//     IT; it is named only so that a reader does not think it was silently dropped.]
//
// ⭐ AND ONE MEASUREMENT THAT CONSTRAINS ANY POSITION-BASED SCHEME, WHICH IS WHY THIS FILE HAS
// NONE. A sibling MEASURED that the position signal is NOT a property of the position: the sign
// INVERTS between adjacent rungs (nvfp4 prefers one end of the block, the ladder's own next rung
// int8 prefers the other), it is ADDITIVE over two 8-wide blocks but NOT over sixteen single
// positions (residual 1.3e-3, which is 25 % of the whole tier step), and the per-position slope
// ORDER DOES NOT REPRODUCE ACROSS CONTEXTS (4 of 16 agree). ⇒ NO STABLE PER-POSITION SENSITIVITY
// EXISTS, so a scheme that encoded one would be encoding noise. This file reads a position only as
// an IDENTITY (which cell) and, through `block`, as AGE -- never as a sensitivity.
//
// =============================================================================================
// 4. THE OBJECTIVE IS GEOMETRY AND THE CONSTRAINT IS MEASUREMENT -- AND WHY NEITHER IS A TABLE
// =============================================================================================
//
// ⚠ DO NOT BUILD THE OBJECTIVE FROM PER-TIER CONSTANTS. TWO MEASUREMENTS FORBID IT, AND BOTH ARE
// POLICY HERE RATHER THAN COMMENT:
//
//   (i) AGGREGATE SPEED IS NOT ADDITIVE. The shipped mix's 57.1 tok/s is +41.7 % ABOVE
//       `max(constituents)` = 40.3, i.e. OUTSIDE THE HULL of its own parts; a speed column that
//       could be summed would have predicted at most 40.3. ⇒ THIS FILE NEVER SUMS SPEEDS. What it
//       sums is BYTES -- `sum over cells of the rung deltas in the charge's own unit` -- and a byte
//       total is an AGGREGATE by construction, not a sum of per-tier rates. The one place a speed
//       could have entered as a per-tier constant (`cost per byte` per rung) is empty: the ranking
//       currency is `nats per byte`, where the numerator is a MEASUREMENT and the denominator is
//       geometry.
//   (ii) QUALITY IS NOT TABLED EITHER. Four independent refutations of the shipped penalty column,
//       including ⭐ **SUM NOT SUFFICIENT: two arms at the same SUM=240 differ by 0.21765 nats.**
//       ⇒ THE CONSTRAINT IS NOT TREATED AS EXACT. A sum-of-per-cell-costs constraint is a
//       SEPARABILITY ASSUMPTION, and this project MEASURED it false. IT IS REPLACED BY:
//
//         spendable = floor_nats - placement_margin_nats
//
//       where `placement_margin_nats` is a MEASURED allowance for exactly the effect the refutation
//       names: at equal sum, WHERE the demotion landed moved the answer by 0.21765 nats. The margin
//       has a NAMED ABSENT STATE (`< 0`), and when nobody measured it the line prints
//       `placement_margin_nats=UNMEASURED(0-used: ...)` LOUDLY rather than silently treating the
//       error as zero -- the convention `kv_perlayer_policy.h:85` states verbatim ("measured ==
//       false means nobody looked, which is NOT the same as the error is zero") and which
//       `kv_adapt_solver.h` reuses.
//
// ⭐ AND THE HONEST LIMIT, STATED SO IT CANNOT BE MISREAD: a sum is still a sum. THIS FILE DOES NOT
// CLAIM ITS SUM EQUALS THE MEASURED QUALITY of the mix it emits. It claims (a) the sum it spends is
// a sum of MEASURED per-step costs, (b) an allowance for the measured non-additivity was subtracted
// BEFORE spending, and (c) the number the solve optimises is BYTES, whose aggregate it CAN compute
// exactly because the engine's charge is a sum over cells by construction. The quality of the
// emitted mix is a MEASUREMENT for an arm to take, and `describe()` names that boundary.
//
// =============================================================================================
// 5. THE EMPTY TABLE -- 少写死多适配 IN ITS STRONG FORM
// =============================================================================================
//
// THE PROPERTY THE OWNER ASKED FOR: "adding a model must not require touching KV code or KV
// constants", so its checkable strong form is -- AN EMPTY MEASUREMENT TABLE MUST COLLAPSE THE
// ALLOCATION ONTO THE PRE-EXISTING BEHAVIOUR BYTE FOR BYTE, so the worst case is "no better than
// today", NEVER WORSE.
//
// ⭐ IT IS IMPLEMENTED AS A CALL-THROUGH AND NOT AS A SPECIAL CASE, AND THE DIFFERENCE IS THE WHOLE
// POINT. The tempting implementation is "with no measurements, treat every cost as zero" -- AND IT
// IS CATASTROPHICALLY WRONG, in the exact direction the property exists to prevent: a cost of zero
// means every step is FREE, so the solver would demote EVERY cell to the FLOOR, and the "no
// measurement" case would be the MOST AGGRESSIVE allocation in the file. An empty table means
// 'nobody looked', and the ONLY allocation that is defensible under 'nobody looked' is the one the
// engine already ships. So:
//
//     if (!table.usable()) { return block_descent_plan(spec); }        // <- the whole collapse
//
// and the identity is BY CONSTRUCTION: `block_descent_plan` is a pure function of its spec (its own
// doctrine, quoted in `kv_descent_control.h`: "a plan is a function of its spec, never of the
// ambient environment"), called ONCE here with the caller's spec, so the object it returns is the
// object the caller would have received had the solve never been selected -- cursor write-back
// included, because it is the same call. The SAME branch covers an OFF budget for the same reason:
// `budget <= 0` is the tree's `BudgetOff`, and OFF is defined as the pre-image charge. A host
// harness proves both collapses bit for bit by comparing the two plans' vectors, block charges and
// totals across a battery of specs (including a carried cursor and an armed age gate), with no GPU.
//
// =============================================================================================
// 6. THE ALGORITHM, ITS TERMINATION, AND ITS EXACTNESS -- ONE PARAGRAPH EACH, THEN THE CODE
// =============================================================================================
//
// (a) BUILD THE CELLS' STAIRCASES. For every cell, walk its own chain DOWN from the default rung
//     with the SAME step function the walk uses (`cell_next_cheaper`, or `cell_next_tier` when the
//     spec selects the owner's 4/3/2 order), and record the prefixes: prefix k has
//     `cost_k = sum of the measured step costs` (nats) and `delta_k = sum of the step deltas`
//     (bytes in the charge's unit). A step whose cost is UNMEASURED STOPS THE CHAIN -- the deeper
//     prefixes cannot be priced, so they are not selectable. ⭐ THIS IS THE TREE'S OWN G6 RULE
//     READ ON THE MEASUREMENT AXIS, verbatim in spirit: "if the next rung has no byte cost today,
//     STOP AT THIS RUNG, do not skip it". An unknown cost is not an infinite cost and it is not
//     zero; it is an unknown, and the chain stops.
// (b) THE HULL. Keep only the prefixes on the CONVEX HULL of `(cost, delta)` in the
//     maximise-delta direction -- Shoham-Gersho -- and count both the prefixes the hull ABSORBED
//     (`hull_merged_prefixes`) and the cells whose raw marginal ratios are not non-decreasing
//     (`nonconvex_cells`). The hull is EXACT, not a heuristic: a prefix strictly inside the hull
//     cannot maximise `delta - lambda * cost` for ANY lambda >= 0, so absorbing it loses no
//     candidate, and the greedy below then ranks only hull edges.
// (c) THE GREEDY, WITH A FEASIBILITY GUARD. Rank every hull edge by `ratio = dcost / ddelta`
//     ASCENDING, tie-break `(block, ordinal)` -- the oldest block first. Take an edge when
//     `spent + dcost <= spendable`; otherwise skip it and keep scanning (a later, smaller edge may
//     still fit). An edge is taken only from the prefix the cell currently sits at, so the emitted
//     vector is always a LEGAL PREFIX -- ONE-WAY, never promoted, and that guard is in the code
//     rather than in the argument.
// (d) THE BOUND. `lp_bound_delta_bytes` is the LAGRANGIAN bound: the minimum, over every breakpoint
//     of the measured staircase, of `sum_cells max_prefix(delta - lambda * cost) + lambda *
//     spendable`. This is an UPPER bound on ANY allocation that meets the floor, because it is the
//     optimum of a relaxation whose feasible set CONTAINS every legal allocation; and it is
//     computed in ONE more sweep of the SAME sorted list the greedy uses, because the staircase's
//     breakpoints ARE the sorted edge ratios. The emitted allocation is FEASIBLE by construction
//     (the guard). ⇒ THE SOLVE NEVER CLAIMS OPTIMALITY WITHOUT A BOUND: `exact` is TRUE ONLY when
//     the emitted delta equals the bound; when it does not, the gap is a number and it is printed
//     on EVERY line, every run.
// (e) TERMINATION, AS A COUNT AND NOT AS AN ARGUMENT. Cells = `blocks * layers` (<= 800 * 16 on
//     every population this engine has today). Prefixes per cell <= `kCellModeCount - 1` = 5, so
//     edges <= 4,000. The cost is ONE sort of <= 4,000 items, one O(edges) prefix scan, one
//     O(edges) hull pass, and two O(edges) sweeps (the greedy's spend and the dual bound).
//     ⚠ THE COMPARISON THAT MATTERS: a sibling's EXACT DP over the byte axis was ~21M transitions
//     in one pass (4,614 states at the measured budget x 768 cells x 6 rungs) against the walk's
//     3,072 accepted steps + 5 sweeps + a 768-element sort. THIS SOLVE IS THE WALK'S OWN ORDER OF
//     COST with NO ITERATION AT ALL -- it is cheaper than the walk it replaces, because it never
//     re-scans the population -- and the only state it carries across the sweep is a running
//     scalar. THERE IS NO `while` AND NO CONVERGENCE CRITERION IN `cell_alloc_decide`: every loop
//     in it is a `for` over a list built once.
//
// =============================================================================================
// 7. THE DECISION IS BLIND TO WHERE A CELL SITS, AND THAT IS A CHECKABLE PROPERTY
// =============================================================================================
//
// THE RULE, IN THE OWNER'S OWN SHAPE: his expectation that the later part of a block keeps its
// precision (「高层不降档本就应该是能够预期的」) MUST EMERGE FROM MEASUREMENT AND MUST NOT BE
// WRITTEN AS A RULE. So the decision never reads a cell's position as a REASON. It reads a position
// twice and for two things only: as an IDENTITY (which cell is being edited -- unavoidable, it IS
// the decision variable) and, through `block`, as AGE (the tie-break the owner asked for). Both are
// inputs; neither is a criterion.
//
// ⭐ HOW TO CHECK IT RATHER THAN BELIEVE IT -- THREE MECHANICAL GATES, ALL RUNNABLE, ALL IN
// `dl/kvsolve/`:
//   (1) TEXT: `cell_alloc_edge_less` and `cell_alloc_decide` -- the comparator and the decision --
//       contain no rule from a position. `grep -nEi 'layer|high|upper'` over those two functions
//       returns NOTHING, and the harness runs exactly that grep and exits non-zero on a hit. (The
//       words cannot appear by accident there: the loop variable is `ordinal`, the geometry read is
//       `plane_bytes`, and the running scalar is `spent`.)
//   (2) BEHAVIOUR, TWO-SIDED: feed the SAME table twice, once with the cost column PERMUTED across
//       positions, and show the emitted per-position column MOVES. A hidden positional rule would
//       make the answer invariant under that permutation. The harness runs it both ways.
//   (3) RULE-FREE EMERGENCE, DEMONSTRATED: with a table in which the later positions cost MORE to
//       demote, the emitted allocation leaves exactly those positions at their default -- produced
//       by the numbers, with no rule anywhere in this file naming a position. The harness prints
//       the two columns side by side. ⇒ A run in which the late positions keep their precision is a
//       READING of the measurements supplied, and it reverses the moment the measurements do.
//
// =============================================================================================
// 8. WHAT THIS FILE DOES NOT DO
// =============================================================================================
//
//   * It does not produce the table. Nothing here measures, fits, or default-fills a cost. A model
//     that ships is a model whose table arrives from outside -- which is the point of section 5.
//     ⇒ TONIGHT THERE IS NO PRODUCER, exactly as GAP-FREQ records for the frequency axis, and
//     `cell_alloc_costs_from_env()` is the NAMED ABSENT state. A solve asked for tonight therefore
//     COLLAPSES, loudly, and that is the intended reading: the architecture is in and the worst case
//     is provably the status quo.
//   * It does not decide the byte budget's ruler. `spec.budget_bytes` arrives in the caller's unit,
//     exactly as the walk takes it, and the satisfiability arithmetic is the walk's own
//     (`min_possible_bytes`, `budget_satisfiable`), recomputed here from the allowed sets so the two
//     plan shapes report ONE set of columns.
//   * It does not claim its sum is the measured quality of the mix. See (4).
//   * It does not touch the cursor, and it does not ask the caller for one. That is finding (B).
//   * It does not decide the TRIGGER. Finding (B) retires the rate question FOR A SOLVE -- there is
//     no rate to bound -- but WHO calls it and HOW OFTEN is the caller's, and this file has no
//     opinion about it.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ninfer::product {

// =============================================================================================
// THE OUTCOMES. `Solved` is not the only one, and the difference between the others IS the point:
// three are REFUSALS BY NAME, two are the 少写死多适配 call-through.
// =============================================================================================
enum class AllocSolveOutcome : std::uint8_t {
    Solved                              = 0,  // a selection was computed from the table
    CollapsedEmptyTable                 = 1,  // no measurement: the pre-image plan, called through
    CollapsedBudgetOff                  = 2,  // budget <= 0: OFF, which this tree defines as the
                                              // pre-image charge -- so OFF is a call-through too
    RefusedFloorUnmeasured              = 3,  // the constraint does not exist: nothing may be spent
    RefusedBudgetUnreachableWithinFloor = 4,  // ⚠ INFEASIBLE: the budget asks past the floor
    RefusedNoPricedRung                 = 5,  // no cell has a byte-costed rung: no total, no solve
    RefusedKvHeadsUnset                 = 6,  // the charge's factor is unknown: bytes uncomputable
};

[[nodiscard]] constexpr const char* alloc_solve_outcome_name(AllocSolveOutcome outcome) noexcept {
    switch (outcome) {
    case AllocSolveOutcome::Solved: return "solved";
    case AllocSolveOutcome::CollapsedEmptyTable:
        return "COLLAPSED-empty-table(pre-image-plan-returned-by-call-through: an empty table means"
               " NOBODY LOOKED, and treating an absent cost as zero would demote EVERY cell to the"
               " floor -- the most aggressive allocation in this file)";
    case AllocSolveOutcome::CollapsedBudgetOff:
        return "COLLAPSED-budget-off(budget<=0: this tree defines OFF as the pre-image charge, so an"
               " OFF budget is a call-through no matter what the table says)";
    case AllocSolveOutcome::RefusedFloorUnmeasured:
        return "REFUSED(floor-unmeasured: the one constraint does not exist, so nothing may be"
               " spent -- a solve with no constraint is a demote-everything rule)";
    case AllocSolveOutcome::RefusedBudgetUnreachableWithinFloor:
        return "REFUSED(budget-unreachable-within-floor: the byte budget demands more demotion than"
               " the measured precision floor permits -- NOT compromised silently, and the plan"
               " returned is the STATUS QUO)";
    case AllocSolveOutcome::RefusedNoPricedRung:
        return "REFUSED(no-priced-rung: no cell has a byte cost, so there is no byte total to"
               " optimise)";
    case AllocSolveOutcome::RefusedKvHeadsUnset:
        return "REFUSED(kv-heads-unset: the charge's 2*kv_heads factor is unknown)";
    }
    return "?";
}

// =============================================================================================
// THE TABLE. THE ONLY MODEL-DEPENDENT INPUT THIS FILE HAS, AND ALL OF IT IS A VALUE.
// =============================================================================================
//
// ⚠ THERE IS DELIBERATELY NO FIELD HERE THAT CAN HOLD A POSITION POLICY, A THRESHOLD, A RANK, A
// DISTANCE OR A SIMILARITY -- the same defence `CellFrequencySource` states in its own comment
// ("there is nothing in this struct a distance or a threshold could be stored in"). A row is
// addressed by (block, ordinal, from, to), i.e. by a CELL AND A STEP, so the most a producer can say
// is "THIS step on THIS cell costs THIS much". It cannot say "these cells are alike", and it cannot
// say "this cell is important".
struct CellAllocCostTable {
    // THE MARGINAL RATIO'S NUMERATOR. `step_cost_nats(self, block, ordinal, from, to)` answers the
    // measured cost of demoting THAT ONE CELL from `from` to `to` -- adjacent rungs of the chosen
    // chain, which is what makes the prefix sum in (a) a sum of measured steps rather than a
    // difference of two measurements taken at different shapes.
    // RETURN CONTRACT: a finite value >= 0 is a measurement; NOTHING ELSE is a measurement. A
    // negative value, a NaN, an infinity or a value at or above 1e300 means THIS STEP IS UNMEASURED,
    // and it STOPS the cell's chain.
    double (*step_cost_nats)(const void* self, std::uint32_t block, std::uint32_t ordinal,
                             CellMode from, CellMode to) = nullptr;
    const void* self = nullptr;

    // THE ONE CONSTRAINT. `floor_measured == false` is a NAMED REFUSAL, never a zero: a floor of
    // zero that nobody measured would mean "spend nothing", and an enormous one that nobody measured
    // would mean "demote everything" -- two opposite behaviours out of one unset field. The bit
    // exists so that neither can happen by omission.
    double floor_nats = 0.0;
    bool floor_measured = false;

    // THE NON-ADDITIVITY ALLOWANCE (section 4). `>= 0` is a measurement and is SUBTRACTED from the
    // floor before anything is spent; `< 0` is the NAMED ABSENT STATE and is printed as UNMEASURED.
    double placement_margin_nats = -1.0;

    // THE ROW COUNT, AND IT IS WHAT MAKES THE COLLAPSE A DECISION RATHER THAN AN ACCIDENT. Zero rows
    // means NOBODY LOOKED, which is precisely the state the call-through is for. A producer that has
    // a `step_cost_nats` but zero rows is not claiming a table -- it is claiming nothing.
    std::uint64_t rows = 0;

    // FREE TEXT. Provenance is a reading too.
    const char* provenance = "unstated";

    [[nodiscard]] bool usable() const noexcept {
        return rows != 0 && step_cost_nats != nullptr;
    }
    [[nodiscard]] bool margin_measured() const noexcept { return placement_margin_nats >= 0.0; }
    // ⭐ `spendable_nats` IS THE FLOOR AFTER THE MEASURED NON-ADDITIVITY ALLOWANCE, and it is
    // CLAMPED AT ZERO: a floor smaller than its own allowance is a table that contradicts itself,
    // and it is NOT a licence to spend negative quality.
    [[nodiscard]] double spendable_nats() const noexcept {
        const double spendable = floor_nats - (margin_measured() ? placement_margin_nats : 0.0);
        return spendable > 0.0 ? spendable : 0.0;
    }
};

// THE ABSENT TABLE, SPELLED AS A VALUE. `CellFrequencySource`'s precedent: the "we have nothing"
// branch is a VALUE, so a caller cannot reach a table by forgetting to pass one.
[[nodiscard]] inline CellAllocCostTable cell_alloc_costs_absent() noexcept {
    return CellAllocCostTable{};
}

// =============================================================================================
// THE READINGS. Every number a reader needs to judge the run WITHOUT re-deriving it, and every one
// of them is about THIS solve -- so nothing here can be mistaken for one of the walk's columns.
// =============================================================================================
struct CellAllocSolveReport {
    AllocSolveOutcome outcome = AllocSolveOutcome::CollapsedEmptyTable;

    // THE CONSTRAINT, AS IT WAS APPLIED.
    double floor_nats = 0.0;
    double placement_margin_nats = -1.0;
    double spendable_nats = 0.0;
    std::uint64_t table_rows = 0;

    // THE POPULATION AND THE STAIRCASE.
    std::uint32_t cells = 0;
    std::uint32_t prefixes = 0;             // priced prefixes, all cells, before the hull
    std::uint32_t hull_merged_prefixes = 0; // prefixes the hull ABSORBED (dominated for every lambda)
    std::uint32_t edges = 0;                // hull edges ranked, all cells
    std::uint32_t nonconvex_cells = 0;      // cells whose raw ratios are not non-decreasing
    std::uint32_t unpriced_chain_cells = 0; // chains STOPPED by an UNMEASURED step (the G6 analogue)
    std::uint32_t gated_cells = 0;          // cells the caller's age gate refused
    std::uint32_t stayed_cells = 0;         // cells the solve emitted at their default rung
    std::uint32_t skipped_for_floor = 0;    // edges NOT taken because they would pass the floor

    // ⭐ THE OPTIMALITY ACCOUNTING. `exact` IS TRUE ONLY WHEN THE EMITTED TOTAL EQUALS THE BOUND, so
    // the line can never print a claim of optimality that was not checked.
    bool exact = false;
    std::int64_t emitted_delta_bytes = 0;  // what was given back, in the charge's unit
    std::int64_t lp_bound_delta_bytes = 0; // an UPPER bound on any allocation that meets the floor
    std::int64_t bound_gap_bytes = 0;      // emitted - bound <= 0; printed on EVERY line

    // THE SPEND, IN THE CONSTRAINT'S UNIT.
    double spent_nats = 0.0;
    double unspent_nats = 0.0;

    // THE FEASIBILITY ARITHMETIC, FOR THE ONE REFUSAL THAT MATTERS.
    std::int64_t budget_bytes = 0;
    std::int64_t reachable_bytes = 0;       // planned - emitted: the smallest total this solve reached
    std::int64_t budget_deficit_bytes = 0;  // reachable - budget (> 0 => the budget was NOT reached)

    [[nodiscard]] std::string describe() const {
        std::string out = "solve=stateless-per-cell(marginal-return) outcome=";
        out += alloc_solve_outcome_name(outcome);
        out += " table_rows=" + std::to_string(table_rows);
        out += " floor_nats=" + std::to_string(floor_nats);
        out += " placement_margin_nats=";
        if (placement_margin_nats < 0.0) {
            out += "UNMEASURED(0-used: nobody measured the equal-sum placement spread, so the"
                   " separability allowance is ZERO and the solve SAYS SO rather than assuming the"
                   " error away)";
        } else {
            out += std::to_string(placement_margin_nats);
        }
        out += " spendable_nats=" + std::to_string(spendable_nats);
        out += " spent_nats=" + std::to_string(spent_nats);
        out += " unspent_nats=" + std::to_string(unspent_nats);
        out += " cells=" + std::to_string(cells);
        out += " prefixes=" + std::to_string(prefixes);
        out += " hull_merged=" + std::to_string(hull_merged_prefixes);
        out += " edges=" + std::to_string(edges);
        out += " diminishing_returns=";
        out += nonconvex_cells == 0
                   ? std::string("ok(ratios non-decreasing on every cell, so Fox 1966's convexity"
                                 " precondition HOLDS on this measured table)")
                   : (std::to_string(nonconvex_cells) +
                      " cells NOT convex (the hull absorbs them; the greedy's precondition is"
                      " MEASURED here, not assumed)");
        out += " unpriced_chain_cells=" + std::to_string(unpriced_chain_cells);
        out += " skipped_for_floor=" + std::to_string(skipped_for_floor);
        out += " age_gated_cells=" + std::to_string(gated_cells);
        out += " stayed_at_default=" + std::to_string(stayed_cells);
        out += " exact=";
        out += exact ? std::string("yes(emitted==the-relaxation-bound)")
                     : std::string("NO(an allocation was emitted that MEETS the floor, and the"
                                   " distance to the bound is printed beside it: the solve does not"
                                   " claim optimality it did not verify)");
        out += " emitted_delta_bytes=" + std::to_string(emitted_delta_bytes);
        out += " lp_bound_delta_bytes=" + std::to_string(lp_bound_delta_bytes);
        out += " bound_gap_bytes=" + std::to_string(bound_gap_bytes);
        out += " budget_bytes=" + std::to_string(budget_bytes);
        out += " reachable_bytes=" + std::to_string(reachable_bytes);
        out += " budget_deficit_bytes=" + std::to_string(budget_deficit_bytes);
        out += " quality_boundary=the-sum-is-spent,NOT-the-mix-measured(this-file-optimises-BYTES-"
               "because-that-aggregate-is-a-sum-by-construction; the-mix's-quality-is-for-an-arm)";
        out += " carry=absent(a-solve-has-no-position: this-engine's-windows-CHURN-at-~48-pages-per-"
               "pass-against-~50-pages-of-population, so no per-population carry can persist a"
               " per-cell allocation -- finding-B)";
        out += " accumulation=by-design-none(each-pass-solves-afresh-from-the-measurement)";
        out += " persistence=per-stack-dtype-table(durable-in-this-engine; a-per-cell-mode-has-no-"
               "durable-home-today)";
        return out;
    }
};

// THE RESULT: THE PLAN THE ENGINE CONSUMES, PLUS THE SOLVE'S OWN LINE. Both halves are returned
// because a caller that wants to REFUSE on a refusal needs the outcome, while the engine's plan
// path needs only the first half -- and `describe()` puts the whole of the second half on the
// plan's own accounting line, so a log is the entire reading either way.
struct CellAllocSolveResult {
    BlockDescentPlan plan{};
    CellAllocSolveReport report{};
};

// =============================================================================================
// 9. THE DECISION -- THE ONLY PLACE A RULE CAN HIDE, AND IT IS TWO FUNCTIONS.
// =============================================================================================
//
// ⚠ EVERY LINE BELOW IS EITHER GEOMETRY READ FROM THE LADDER, A MEASUREMENT READ FROM THE TABLE, OR
// A COMPARISON OF THE TWO. There is deliberately NO comparison between two ordinals, no test on a
// cell's position, and no branch whose condition is "where am I inside the block". The words a
// positional rule would need are not in either function, and the harness greps for them and fails on
// a hit. `block` enters exactly twice -- as the AGE tie-break the owner asked for, and as the
// identity half of a cell's address -- and `ordinal` enters only as the other half of that address.
//
// THE RULE, IN ONE SENTENCE: among the candidate steps that are INDIFFERENT TO THE FLOOR -- which is
// spelled exactly as an equal marginal ratio -- take the one on the OLDEST BLOCK first; and never
// take a step whose accumulated cost would carry the population past the floor.
struct CellAllocEdge {
    double ratio = 0.0;        // nats given up per byte given back. THE ONLY CURRENCY IN THIS FILE.
    double cost = 0.0;         // the edge's measured cost, nats
    std::int64_t delta = 0;    // the edge's byte delta, in the charge's own unit
    std::uint32_t block = 0;   // THE AGE AXIS (the tie-break), and half of the cell's address
    std::uint32_t ordinal = 0; // the other half of the cell's address. NOT a criterion.
    std::uint32_t from = 0;    // the prefix index the edge STARTS at -- the prefix guard, in data
    std::uint32_t to = 0;      // the prefix index the edge LANDS on
};

// TWO RATIOS ARE THE SAME RATIO WHEN THEY ARE THE SAME NUMBER TO WITHIN A KILOBYTE OF FLOAT NOISE.
// The tolerance matters here and only here: the dual sweep groups equal ratios to evaluate the
// Lagrangian at ONE breakpoint, and an exact `==` on two double divisions of equal rationals is not
// guaranteed. The tolerance is RELATIVE and it is 1e-12, i.e. ~4,500 ULPs at 1.0 -- far below any
// ratio a table could distinguish, and far above the error of one division.
[[nodiscard]] inline bool cell_alloc_same_ratio(double a, double b) noexcept {
    const double scale = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
    return std::fabs(a - b) <= 1e-12 * (scale > 1.0 ? scale : 1.0);
}

[[nodiscard]] inline bool cell_alloc_edge_less(const CellAllocEdge& a,
                                               const CellAllocEdge& b) noexcept {
    if (a.ratio != b.ratio) { return a.ratio < b.ratio; }
    if (a.block != b.block) { return a.block < b.block; }
    return a.ordinal < b.ordinal;
}

struct CellAllocDecision {
    std::vector<std::uint32_t> prefix;   // the prefix each cell ends at; 0 == its default rung
    double spent = 0.0;                  // nats spent, in the constraint's unit
    std::uint32_t hops = 0;              // rungs moved, summed over cells (the walk's `steps` unit)
    std::uint32_t edited = 0;            // cells that moved at least one rung
    std::uint32_t skipped_for_floor = 0; // edges not taken because they would pass the floor
};

// THE GREEDY. `sorted` is ascending by `cell_alloc_edge_less`; `cells` is the population size and
// `stride` is the per-block cell count, so a cell's address is `block * stride + ordinal`.
// NO `while`, NO CONVERGENCE CRITERION: the first loop is one `for` over a list built once and the
// second is a `for` over the population.
[[nodiscard]] inline CellAllocDecision cell_alloc_decide(const std::vector<CellAllocEdge>& sorted,
                                                        std::uint32_t cells,
                                                        std::uint32_t stride,
                                                        double spendable) {
    CellAllocDecision decision{};
    decision.prefix.assign(cells, 0U);
    if (!(spendable > 0.0)) { return decision; }
    for (const CellAllocEdge& edge : sorted) {
        const std::uint32_t cell = edge.block * stride + edge.ordinal;
        if (cell >= cells) { continue; }
        // THE PREFIX GUARD, IN THE LOOP AND NOT IN THE ARGUMENT: an edge may be taken only when the
        // cell currently sits exactly where the edge begins. This is what makes ONE-WAY a property
        // of the emitted object rather than a promise about the search.
        if (decision.prefix[cell] != edge.from) { continue; }
        if (!(decision.spent + edge.cost <= spendable)) {
            decision.skipped_for_floor += 1;
            continue;
        }
        decision.spent += edge.cost;
        decision.prefix[cell] = edge.to;
    }
    for (std::uint32_t cell = 0; cell < cells; ++cell) {
        if (decision.prefix[cell] != 0U) {
            decision.edited += 1;
            decision.hops += decision.prefix[cell];   // the prefix index IS the rung count moved
        }
    }
    return decision;
}

// =============================================================================================
// THE SOLVE.
// =============================================================================================
namespace cell_alloc_detail {

// ONE CELL'S STAIRCASE: prefix k is reachable by k hops down its own chain, priced by the table.
struct CellChain {
    std::vector<double> cost{0.0};         // prefix costs, nats; [0] == 0 (the default rung)
    std::vector<std::int64_t> delta{0};    // prefix byte deltas, charge units; [0] == 0
    std::vector<CellMode> mode{};          // the rung each prefix lands on; [0] == the default
    bool gated = false;                    // the caller's age gate refused this cell
    bool unpriced_stop = false;            // an UNMEASURED step stopped the chain (the G6 analogue)
    std::uint32_t merged = 0;              // prefixes the hull absorbed on THIS chain
};

}  // namespace cell_alloc_detail

[[nodiscard]] inline CellAllocSolveResult cell_alloc_solve(const DescentSpec& spec,
                                                           const CellAllocCostTable& table) {
    CellAllocSolveResult result{};

    // -----------------------------------------------------------------------------------------
    // (5) THE TWO COLLAPSES -- BY CALL-THROUGH, SO THE IDENTITY IS CONSTRUCTION, NOT ARGUMENT.
    // An empty table means nobody looked; an OFF budget is defined by this tree as the pre-image
    // charge. Both would be changed by any "solve with defaults" and neither may be.
    //
    // ⚠⚠ THE CALL-THROUGH MUST REWRITE `walk`, AND THIS IS NOT A DETAIL -- IT IS THE ONE DEFECT IN
    // THIS FILE THAT A HOST ARM CAUGHT, SO IT IS WRITTEN OUT. `block_descent_plan` reads its shape
    // as `const bool one_shot = spec.walk == DescentWalk::OneShot;` -- an EQUALITY TEST, not a
    // switch -- so handing it a spec whose `walk` is `Solve` makes it run the PRE-IMAGE
    // WHOLE-POPULATION SWEEP (five sweeps, every cell) instead of the live one-shot walk. MEASURED
    // by this file's arm before this line existed: the "collapse" on the shipped shape emitted
    // `demoted=124 .. sweeps=1 .. drive=stopped-by-budget` where the walk it was supposed to
    // reproduce emitted `demoted=1 .. sweeps=0 .. drive=one-shot-step-taken`. THAT IS EXACTLY THE
    // FAILURE THE 少写死多适配 PROPERTY EXISTS TO PREVENT -- a solve that collapses and is then
    // WORSE than the behaviour it collapsed onto -- and it was invisible to every reading except a
    // byte-for-byte comparison of the two plans. SO: the spec that goes through is the LIVE PATH's
    // (`OneShot`), and the shape's NAME is restored onto the returned plan afterwards, because what
    // SELECTED this plan was the solve and the line has to be able to say so.
    // -----------------------------------------------------------------------------------------
    if (!table.usable() || spec.budget_bytes <= 0) {
        DescentSpec collapse = spec;
        collapse.walk = DescentWalk::OneShot;   // ⚠ see the note above: the live path's own value
        result.plan = block_descent_plan(collapse);
        result.report.outcome = table.usable() ? AllocSolveOutcome::CollapsedBudgetOff
                                               : AllocSolveOutcome::CollapsedEmptyTable;
        result.report.table_rows = table.rows;
        result.report.floor_nats = table.floor_nats;
        result.report.placement_margin_nats = table.placement_margin_nats;
        result.report.spendable_nats = table.spendable_nats();
        result.report.budget_bytes = spec.budget_bytes;
        result.report.cells = result.plan.cells_total;
        result.report.emitted_delta_bytes = result.plan.saved_bytes;
        result.report.lp_bound_delta_bytes = result.plan.saved_bytes;
        result.report.reachable_bytes = result.plan.total_bytes;
        result.plan.walk = spec.walk;
        result.plan.alloc_note = result.report.describe();
        return result;
    }

    CellAllocSolveReport& report = result.report;
    BlockDescentPlan& plan = result.plan;
    report.table_rows = table.rows;
    report.floor_nats = table.floor_nats;
    report.placement_margin_nats = table.placement_margin_nats;
    report.spendable_nats = table.spendable_nats();
    report.budget_bytes = spec.budget_bytes;

    const std::uint32_t layers =
        spec.layers == 0 ? 0U
                         : (spec.layers < kBlockVectorLayerAxis ? spec.layers
                                                                : kBlockVectorLayerAxis);
    const std::uint32_t blocks = spec.blocks;
    const std::uint32_t cells = blocks * layers;
    report.cells = cells;

    plan.budget_bytes = spec.budget_bytes;
    plan.frequency = spec.frequency.kind;
    plan.population = spec.population;
    plan.cells_total = cells;
    plan.walk = spec.walk;
    plan.population_origin_seen = spec.population_origin;
    plan.age_keep_recent_blocks = spec.age.keep_recent_blocks;
    plan.age_frontier = spec.age.frontier;
    plan.newest_eligible_block = spec.age.newest_eligible();

    // THE START, ON THE WALK'S OWN ARITHMETIC: the same default vector through the same helper, so
    // "the solve and the walk start from one object" is not a coincidence. `allowed` is the
    // caller's `F_(l,b)`; an empty vector means everything, exactly as the walk reads it.
    auto allowed_for = [&spec, layers](std::uint32_t ordinal) -> CellAllowedSet {
        if (spec.allowed.empty() || ordinal >= spec.allowed.size() || ordinal >= layers) {
            return cell_all_modes();
        }
        return spec.allowed[ordinal];
    };
    BlockVector base = block_vector_default(layers);
    bool any_no_default = false;
    for (std::uint32_t ordinal = 0; ordinal < layers; ++ordinal) {
        const CellDefault def = cell_default_tier(allowed_for(ordinal));
        base.set(ordinal, def.mode);
        if (def.state == DefaultTier::NoDefault) { any_no_default = true; }
    }
    plan.vectors.assign(blocks, base);
    plan.block_bytes.assign(blocks, 0);

    // THE TWO PRE-CONDITIONS THAT ARE NOT THE FLOOR, AND BOTH ARE REFUSALS WITH A NAME. They are
    // the same two states the walk names (`UnpricedCell`), restated here rather than inherited,
    // because this function must be readable without the walk's body.
    const CellCharge base_charge = cell_vector_charge(base, spec.kv_heads);
    plan.planned_bytes = static_cast<std::int64_t>(blocks) * base_charge.bytes;
    plan.total_bytes = plan.planned_bytes;
    const auto status_quo = [&](AllocSolveOutcome outcome, std::int64_t total,
                                DescentDrive drive) {
        for (std::uint32_t b = 0; b < blocks; ++b) { plan.block_bytes[b] = base_charge.bytes; }
        for (std::uint32_t ordinal = 0; ordinal < layers; ++ordinal) {
            plan.final_rung_histogram[static_cast<std::uint32_t>(base.mode[ordinal])] += blocks;
        }
        report.outcome = outcome;
        report.reachable_bytes = total;
        report.budget_deficit_bytes = total - spec.budget_bytes;
        plan.priced = base_charge.priced && !any_no_default;
        plan.drive = drive;
        plan.alloc_note = report.describe();
        return result;
    };
    if (spec.kv_heads <= 0) {
        return status_quo(AllocSolveOutcome::RefusedKvHeadsUnset, plan.planned_bytes,
                          DescentDrive::UnpricedCell);
    }
    if (!base_charge.priced || any_no_default) {
        return status_quo(AllocSolveOutcome::RefusedNoPricedRung, plan.planned_bytes,
                          DescentDrive::UnpricedCell);
    }
    // ⚠ THE FLOOR IS THE ONE CONSTRAINT, SO AN UNMEASURED FLOOR IS A REFUSAL AND NOT A DEFAULT.
    // Spending against an unmeasured floor is the single most damaging silent move available here:
    // it would demote the whole population against a number nobody took, and call it a budget.
    if (!table.floor_measured) {
        return status_quo(AllocSolveOutcome::RefusedFloorUnmeasured, plan.planned_bytes,
                          DescentDrive::RefusedByFloor);
    }

    // -----------------------------------------------------------------------------------------
    // (a) THE STAIRCASES. One per cell, in the chain the spec selected, STOPPED by an unmeasured
    // step -- the tree's G6 rule read on the measurement axis.
    // -----------------------------------------------------------------------------------------
    const bool order_lattice = spec.step_order == DescentStepOrder::Lattice432;
    const std::uint32_t hop_cap = kCellModeCount;   // a chain is at most kCellModeCount-1 long
    std::vector<cell_alloc_detail::CellChain> chains;
    chains.reserve(cells);
    for (std::uint32_t block = 0; block < blocks; ++block) {
        for (std::uint32_t ordinal = 0; ordinal < layers; ++ordinal) {
            cell_alloc_detail::CellChain chain{};
            const CellAllowedSet set = allowed_for(ordinal);
            CellMode here = base.at(ordinal);
            chain.mode.push_back(here);
            // THE CALLER'S AGE GATE, APPLIED THROUGH THE SAME PREDICATE THE WALK APPLIES. A cell
            // the gate refuses is not a candidate AT ALL: the lawful floor is not a licence to
            // override the caller's own eligibility rule.
            chain.gated = spec.age.keep_recent_blocks != 0 && !spec.age.admits(block);
            if (chain.gated) {
                report.gated_cells += 1;
            } else {
                for (std::uint32_t hop = 0; hop < hop_cap; ++hop) {
                    const CellStep step = order_lattice ? cell_next_tier(set, here)
                                                        : cell_next_cheaper(set, here);
                    if (!step.legal() || step.bytes_saved <= 0) { break; }
                    const double hop_cost =
                        table.step_cost_nats(table.self, block, ordinal, here, step.to);
                    // NOT A MEASUREMENT: the chain STOPS. An unknown cost is not zero and it is not
                    // infinite, and this tree's own ruling is "stop at this rung, do not skip it".
                    if (!(hop_cost >= 0.0) || !(hop_cost < 1e300)) {
                        chain.unpriced_stop = true;
                        break;
                    }
                    const std::int64_t hop_delta = ladder_step_total_delta(step, spec.kv_heads);
                    if (hop_delta <= 0) { break; }
                    chain.cost.push_back(chain.cost.back() + hop_cost);
                    chain.delta.push_back(chain.delta.back() + hop_delta);
                    chain.mode.push_back(step.to);
                    here = step.to;
                }
                if (chain.unpriced_stop) { report.unpriced_chain_cells += 1; }
            }
            report.prefixes += static_cast<std::uint32_t>(chain.cost.size() - 1);
            chains.push_back(std::move(chain));
        }
    }

    // -----------------------------------------------------------------------------------------
    // (b) THE HULL, AND THE CONVEXITY CHECK -- Shoham-Gersho's machinery and Fox's precondition,
    // both ON THIS TABLE. A tentative edge whose ratio is BELOW the last kept edge's has a
    // MORE EFFICIENT step behind it, so the prefix that edge lands on is strictly inside the hull
    // and is absorbed; the loop repeats until the ratios are non-decreasing, which is what makes
    // the kept list a hull. `merged` is what the line reports as `hull_merged`, and a chain with
    // at least one merge is exactly a chain whose raw ratios were not non-decreasing -- which is
    // why one counter answers both questions and there is no second, disagreeing one.
    // -----------------------------------------------------------------------------------------
    std::vector<CellAllocEdge> edges;
    edges.reserve(static_cast<std::size_t>(cells) * hop_cap);
    for (std::size_t cell_index = 0; cell_index < chains.size(); ++cell_index) {
        cell_alloc_detail::CellChain& chain = chains[cell_index];
        if (chain.mode.size() < 2) { continue; }
        std::vector<CellAllocEdge> pending;   // kept small: <= hop_cap
        std::size_t start = 0;
        for (std::size_t i = 1; i < chain.cost.size(); ++i) {
            bool usable = false;
            for (;;) {
                const std::int64_t ddelta = chain.delta[i] - chain.delta[start];
                if (ddelta <= 0) { break; }
                const double dcost = chain.cost[i] - chain.cost[start];
                const double ratio = dcost / static_cast<double>(ddelta);
                if (!pending.empty() && ratio < pending.back().ratio) {
                    // ABSORB: the edge below this one is dominated, so the two become one edge and
                    // the prefix between them is dropped from the candidate set.
                    start = pending.back().from;
                    pending.pop_back();
                    chain.merged += 1;
                    continue;
                }
                CellAllocEdge edge{};
                edge.ratio = ratio;
                edge.cost = dcost;
                edge.delta = ddelta;
                edge.from = static_cast<std::uint32_t>(start);
                edge.to = static_cast<std::uint32_t>(i);
                pending.push_back(edge);
                usable = true;
                break;
            }
            // ⭐ THE EDGE JUST PUSHED IS CONSUMED, SO THE NEXT ONE BEGINS WHERE IT LANDED. WITHOUT
            // THIS LINE THE EDGES ARE CUMULATIVE FROM PREFIX 0 AND THE PREFIX GUARD -- which is the
            // one-way property -- REJECTS EVERY EDGE ABOVE THE FIRST: measured on this file's own
            // arm before the line existed, `emitted_delta_bytes=737280` against a brute-force
            // optimum of `872448`, i.e. the solve spent ONE hop per cell and stopped, with
            // `skipped_for_floor=0` because the loss was not the floor's doing at all. A marginal
            // staircase is what a marginal-return rule ranks; a cumulative one is a different
            // object and the `from`/`to` pair below is what makes it the first.
            if (usable) { start = i; }
        }
        if (chain.merged != 0) { report.nonconvex_cells += 1; }
        report.hull_merged_prefixes += chain.merged;
        // THE EDGES CARRY THEIR CELL'S ADDRESS, and it is assigned HERE so the chain itself stays
        // address-free while it is being priced. The address is `(block, ordinal)` and it is an
        // IDENTITY -- the sort's age tie-break reads `block`, and nothing reads `ordinal` as a
        // criterion.
        const std::uint32_t cell_block =
            layers == 0 ? 0U : static_cast<std::uint32_t>(cell_index / layers);
        const std::uint32_t cell_ordinal =
            layers == 0 ? 0U : static_cast<std::uint32_t>(cell_index % layers);
        for (CellAllocEdge& edge : pending) {
            edge.block = cell_block;
            edge.ordinal = cell_ordinal;
            edges.push_back(edge);
        }
    }
    report.edges = static_cast<std::uint32_t>(edges.size());

    // -----------------------------------------------------------------------------------------
    // (c) + (d) THE SAME SORTED LIST ANSWERS BOTH QUESTIONS: the greedy's feasibility guard and the
    // dual's bound. ONE sort, TWO `for` loops, no iteration between them.
    // -----------------------------------------------------------------------------------------
    std::sort(edges.begin(), edges.end(), cell_alloc_edge_less);
    const double spendable = report.spendable_nats;
    const CellAllocDecision decision = cell_alloc_decide(edges, cells, layers, spendable);
    report.spent_nats = decision.spent;
    report.unspent_nats = spendable - decision.spent;

    // THE BOUND. `dual_prefix[cell]` tracks how far the RELAXATION's per-cell choice has walked.
    //
    // ⚠ THE MULTIPLIER'S DIRECTION IS THE EASY THING TO GET BACKWARDS, AND IT WAS, SO IT IS DERIVED
    // HERE RATHER THAN ASSERTED. The Lagrangian is `L(eta) = max over prefixes of (delta - eta *
    // cost) + eta * spendable`, and adding an edge j to a cell's prefix raises `delta - eta * cost`
    // by `d_j - eta * c_j`. SO THE EDGE IS WORTH TAKING IFF `d_j > eta * c_j`, i.e. iff `eta <
    // d_j / c_j = 1 / r_j` -- which means a SMALLER ratio is taken at a LARGER multiplier. So the
    // sweep adds the sorted edges in ASCENDING ratio order (my accumulation does) while the
    // multiplier DECREASES, and the breakpoints it must be evaluated AT are `eta_j = 1 / r_j`, NOT
    // `r_j`. Evaluating at `r_j` instead inverts the sweep and yields a value that is NOT an upper
    // bound: MEASURED on this file's own arm before the fix, `lp_bound_delta_bytes=368640` against
    // `emitted_delta_bytes=737280`, i.e. the "bound" sat BELOW the allocation it was supposed to
    // bound -- which is exactly the failure mode `exact=` exists to make impossible.
    std::vector<std::uint32_t> dual_prefix(cells, 0U);
    double sum_c = 0.0;
    double sum_d = 0.0;
    double best_bound = 0.0;
    bool bound_set = false;
    std::size_t i = 0;
    while (i < edges.size()) {
        const double ratio = edges[i].ratio;
        while (i < edges.size() && cell_alloc_same_ratio(edges[i].ratio, ratio)) {
            const CellAllocEdge& edge = edges[i];
            const std::uint32_t cell = edge.block * layers + edge.ordinal;
            if (cell < cells && dual_prefix[cell] == edge.from) {
                sum_c += edge.cost;
                sum_d += static_cast<double>(edge.delta);
                dual_prefix[cell] = edge.to;
            }
            ++i;
        }
        // A FREE EDGE (ratio 0) IS TAKEN AT EVERY FINITE MULTIPLIER, so its own breakpoint is at
        // infinity and there is no value to evaluate there.
        if (!(ratio > 0.0)) { continue; }
        const double eta = 1.0 / ratio;
        const double value = sum_d - eta * sum_c + eta * spendable;
        if (!bound_set || value < best_bound) {
            best_bound = value;
            bound_set = true;
        }
    }
    // THE BOUNDARY AT lambda == 0: the state with EVERY edge applied, whose value is `sum_d` alone.
    // A convex piecewise-linear function attains its minimum at a breakpoint OR at an endpoint, and
    // this is the endpoint that a sweep over positive ratios alone would miss.
    {
        double sum_d_all = 0.0;
        for (const cell_alloc_detail::CellChain& chain : chains) {
            sum_d_all += static_cast<double>(chain.delta.back());
        }
        if (!bound_set || sum_d_all < best_bound) {
            best_bound = sum_d_all;
            bound_set = true;
        }
    }
    report.lp_bound_delta_bytes = static_cast<std::int64_t>(best_bound < 0.0 ? 0.0 : best_bound);

    // -----------------------------------------------------------------------------------------
    // THE EMITTED PLAN. The vectors are written ONCE per cell from the chain's own mode list, so
    // ONE-WAY is visible in the object: a cell can only ever be set to a rung BELOW its default,
    // because the chain was built downward from that default and the prefix guard kept the choice
    // on the chain.
    // -----------------------------------------------------------------------------------------
    std::int64_t emitted = 0;
    for (std::uint32_t cell = 0; cell < cells; ++cell) {
        const std::uint32_t block = cell / (layers == 0 ? 1U : layers);
        const std::uint32_t ordinal = cell % (layers == 0 ? 1U : layers);
        const cell_alloc_detail::CellChain& chain = chains[cell];
        const std::uint32_t chosen = decision.prefix[cell] < chain.mode.size()
                                         ? decision.prefix[cell]
                                         : 0U;
        plan.vectors[block].set(ordinal, chain.mode[chosen]);
        emitted += chain.delta[chosen];
        if (chosen == 0U) { report.stayed_cells += 1; }
    }
    report.emitted_delta_bytes = emitted;
    report.skipped_for_floor = decision.skipped_for_floor;
    report.bound_gap_bytes = emitted - report.lp_bound_delta_bytes;
    report.exact = report.bound_gap_bytes == 0;

    // ⚠ THE ONE REFUSAL THAT MATTERS, AND IT IS CHECKED BEFORE ANY DEGRADED PLAN LEAVES THIS
    // FUNCTION. If the emitted allocation does NOT reach the caller's byte budget, then the budget
    // demanded more demotion than the floor permitted -- and the answer is the STATUS QUO plus a
    // NAME, never a plan that passed the floor "a little". See section 2.
    const std::int64_t reachable = plan.planned_bytes - emitted;
    if (reachable > spec.budget_bytes) {
        plan.vectors.assign(blocks, base);
        for (std::uint32_t b = 0; b < blocks; ++b) { plan.block_bytes[b] = base_charge.bytes; }
        for (std::uint32_t ordinal = 0; ordinal < layers; ++ordinal) {
            plan.final_rung_histogram[static_cast<std::uint32_t>(base.mode[ordinal])] += blocks;
        }
        report.outcome = AllocSolveOutcome::RefusedBudgetUnreachableWithinFloor;
        report.reachable_bytes = reachable;
        report.budget_deficit_bytes = reachable - spec.budget_bytes;
        plan.total_bytes = plan.planned_bytes;
        plan.saved_bytes = 0;
        plan.drive = DescentDrive::RefusedByFloor;
        plan.alloc_note = report.describe();
        return result;
    }

    // THE TAIL IS THE WALK'S OWN TAIL, field for field, so the two plan shapes report ONE set of
    // columns and a census can compare them without knowing which produced the object.
    std::int64_t recomputed_total = 0;
    for (std::uint32_t b = 0; b < blocks; ++b) {
        const CellCharge charge = cell_vector_charge(plan.vectors[b], spec.kv_heads);
        plan.block_bytes[b] = charge.bytes;
        recomputed_total += charge.bytes;
    }
    plan.total_bytes = recomputed_total;
    plan.totals_agree = true;               // the total IS the recomputation: one spelling, no drift
    plan.saved_bytes = plan.planned_bytes - plan.total_bytes;
    plan.priced = true;
    plan.steps = decision.hops;
    plan.sweeps = 0;                        // a solve is not a sweep, and the column says so
    plan.triggers_fired = 0;
    plan.steps_taken = 0;
    plan.law_holds = true;                  // 0 <= 0: the law's predicate on a cursor-free object
    plan.carry_refused = false;             // no carry was offered, and none is used (finding B)

    {
        std::int64_t per_block_floor = 0;
        for (std::uint32_t ordinal = 0; ordinal < layers; ++ordinal) {
            std::int32_t cheapest = 0;
            bool any = false;
            for (const CellRung& rung : kCellRungs) {
                if (!allowed_for(ordinal).has(rung.mode)) { continue; }
                if (rung.price_state != PriceState::Priced) { continue; }
                if (!any || rung.plane_bytes < cheapest) { cheapest = rung.plane_bytes; any = true; }
            }
            per_block_floor += any ? cheapest : 0;
        }
        plan.min_possible_bytes = static_cast<std::int64_t>(blocks) * 2 *
                                  static_cast<std::int64_t>(spec.kv_heads) * per_block_floor;
        if (base_charge.bytes > 0) {
            plan.budget_blocks_x100 = spec.budget_bytes * 100 / base_charge.bytes;
            plan.budget_satisfiable = spec.budget_bytes >= plan.min_possible_bytes;
        }
    }

    // THE TWO DISTRIBUTIONS AND THE COUNTERS, from the emitted vectors and not from the decision,
    // so the census is a reading of the OBJECT the stage will charge.
    std::vector<std::uint32_t> demoted_per_ordinal(layers, 0);
    std::vector<std::uint32_t> demoted_per_block(blocks, 0);
    for (std::uint32_t b = 0; b < blocks; ++b) {
        const BlockVector& vector = plan.vectors[b];
        for (std::uint32_t ordinal = 0; ordinal < vector.layers; ++ordinal) {
            const std::uint32_t index = static_cast<std::uint32_t>(vector.mode[ordinal]);
            if (vector.mode[ordinal] != base.mode[ordinal]) {
                plan.cells_demoted += 1;
                demoted_per_ordinal[ordinal] += 1;
                demoted_per_block[b] += 1;
                plan.demoted_newest_block = b;
                if (!plan.any_demoted) { plan.demoted_oldest_block = b; }
                plan.any_demoted = true;
            }
            plan.final_rung_histogram[index] += 1;
            // `cells_at_floor` IS READ WITH THE STEP FUNCTION'S OWN PREDICATE -- "no strictly
            // cheaper priced rung is left in this cell's set" -- so a solved plan and a walked plan
            // mean the same thing by it even though they count it at different moments.
            const CellStep further = order_lattice ? cell_next_tier(allowed_for(ordinal),
                                                                    vector.mode[ordinal])
                                                   : cell_next_cheaper(allowed_for(ordinal),
                                                                       vector.mode[ordinal]);
            if (!further.legal()) {
                switch (further.refusal) {
                case StepRefusal::AlreadyAtFloor: plan.cells_at_floor += 1; break;
                case StepRefusal::RungBytesUnknown: plan.cells_blocked_by_unknown += 1; break;
                case StepRefusal::NoByteCurrency: plan.cells_blocked_by_currency += 1; break;
                default: break;
                }
            }
        }
    }
    for (std::uint32_t ordinal = 0; ordinal < layers; ++ordinal) {
        if (demoted_per_ordinal[ordinal] != 0) { plan.demoted_layers += 1; }
        if (demoted_per_ordinal[ordinal] > plan.demoted_max_per_layer) {
            plan.demoted_max_per_layer = demoted_per_ordinal[ordinal];
        }
    }
    for (std::uint32_t b = 0; b < blocks; ++b) {
        if (demoted_per_block[b] != 0) { plan.demoted_blocks += 1; }
        if (demoted_per_block[b] > plan.demoted_max_per_block) {
            plan.demoted_max_per_block = demoted_per_block[b];
        }
    }
    report.outcome = AllocSolveOutcome::Solved;
    report.reachable_bytes = plan.total_bytes;
    report.budget_deficit_bytes = plan.total_bytes - spec.budget_bytes;
    plan.drive = DescentDrive::StoppedByBudget;
    plan.alloc_note = report.describe();
    return result;
}

// =============================================================================================
// 10. THE ONE CALL SITE THAT SELECTS THE SOLVE.
// =============================================================================================
//
// `block_alloc_plan` is the SINGLE entry point the engine calls. It selects on `spec.walk`, which is
// a field of the spec and therefore a function of the spec -- never of the ambient environment --
// and every path whose selection is not `Solve` reaches `block_descent_plan` UNCHANGED, so an
// unmodified caller is byte-identical by construction and there is no second dispatch site to keep
// in step.
[[nodiscard]] inline BlockDescentPlan block_alloc_plan(const DescentSpec& spec,
                                                       const CellAllocCostTable& table) {
    if (spec.walk != DescentWalk::Solve) { return block_descent_plan(spec); }
    return cell_alloc_solve(spec, table).plan;
}

// THE CALLER'S READER, IN THE HOUSE STYLE THE OTHER TWO KNOBS SET (`descent_step_order_from_env`,
// `ladder_ceiling_from_env`): the CALLER owns the environment, a plan is a function of its spec, and
// UNSET IS THE PRE-IMAGE. `solve` is the only value that selects the solve; anything else -- unset,
// empty, misspelled -- is `OneShot`, so a typo cannot silently change what a run measures.
inline constexpr const char* kKvDescentAllocEnv = "NINFER_KV_DESCENT_ALLOC";

[[nodiscard]] inline DescentWalk descent_walk_from_env() noexcept {
    const char* const text = std::getenv(kKvDescentAllocEnv);
    if (text == nullptr) { return DescentWalk::OneShot; }
    return std::strcmp(text, "solve") == 0 ? DescentWalk::Solve : DescentWalk::OneShot;
}

// ⚠ THE TABLE'S READER, AND IT RETURNS THE ABSENT TABLE BECAUSE THERE IS NO PRODUCER -- the same
// state `cell_frequency_absent()` records for the other measurement axis, and it is named rather
// than left out so that the collapse is a READING on the line instead of an absence. A model that
// ships brings its own table; this function is then replaced by a producer and nothing else in this
// file moves, which is the whole of section 5.
[[nodiscard]] inline CellAllocCostTable cell_alloc_costs_from_env() noexcept {
    CellAllocCostTable table = cell_alloc_costs_absent();
    table.provenance = "no-producer-exists(GAP-ALLOC-TABLE): the solve collapses by call-through";
    return table;
}

}  // namespace ninfer::product
