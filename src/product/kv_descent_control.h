#pragma once

// F1202 -- THE CONTROL LAW: **START AT INT8, DEMOTE OLD CONTENT ONLY, LET NEW CONTENT ENTER AT
// INT8, AND ONE TRIGGER => ONE DEMOTION.** The three gaps `blob_F1192.md` 34.3/34.4/34.6 named
// (GAP-ONESHOT, GAP-CURSOR, GAP-AGEGATE) are closed HERE, in one file, because they are one law.
//
// THE OWNER'S SENTENCE, VERBATIM, IS THE WHOLE SPEC (`blob_F1192.md` 34.0 quotes it and this file
// does not paraphrase it):
//   「我那个全层int8逐渐降级，降级的同时新内容应当仍然保持int8；降级应当不是一直进行，应当触发
//     一次降级一次，触发条件有待商榷，但避免一直抢占」
//
// READ AS FOUR POINTS (the same four 34.0 reads and verifies), AND WHERE EACH LANDS HERE:
//
//   (i)  START = all-layer int8        -> `descent_cursor_init`: every block's vector is
//                                          `block_vector_default` and every layer's default is
//                                          `cell_default_tier`, which is `Int8` whenever the set
//                                          holds it. Already true in the tree (34.1); this file
//                                          ASSERTS it rather than re-implementing it.
//   (ii) DEMOTION TARGETS OLD ONLY     -> the AGE GATE below: a block is a candidate only when it
//                                          is at least `keep_recent_blocks` behind the window's
//                                          newest block. GAP-AGEGATE, closed.
//  (iii) NEW CONTENT ENTERS AT int8    -> `descent_absorb_new_blocks`: arrival is an append of
//                                          DEFAULT vectors, and the age gate excludes exactly the
//                                          blocks that just arrived. The two are the same cut seen
//                                          from its two sides, which is why they are one file.
//   (iv) ONE TRIGGER => ONE DEMOTION   -> `block_descent_trigger` performs AT MOST ONE accepted
//                                          step and returns. GAP-ONESHOT, closed in the shape the
//                                          owner asked for: an EVENT, not a loop and not a daemon.
//
// ---------------------------------------------------------------------------------------------
// ⚠ THE ONE THING THE OWNER LEFT OPEN, AND THE REFUSAL THIS FILE MAKES RATHER THAN GUESSING:
// 34.3's own conclusion is that `GAP-ONESHOT cannot be closed without point 4 being answered` --
// *"the trigger condition is open"*, 「触发条件有待商榷」. That is a question about the WORLD (what
// event means "the budget is under pressure"), and no header can answer it. SO THIS FILE DOES NOT
// INVENT A TRIGGER. It takes the trigger as an INPUT (`DescentTrigger`), performs exactly one
// demotion for however many triggers it is handed, and PRINTS how many it has been handed. The
// policy that fires triggers lives with the pass, one layer up, because only the pass can see the
// budget, the page loop and the clock. What this file DOES close is the part that was a MECHANISM
// and not a policy: the loop that used to move a whole population per call, and the statelessness
// that made a second call re-downgrade the first cell.
//
// ---------------------------------------------------------------------------------------------
// THE THREE GAPS, AND THE EXACT SENTENCE THAT CLOSES EACH -- because "closed" is a claim and a
// reader is owed the code it is a claim about:
//
//   GAP-ONESHOT (34.3): "the implementation is a sweep loop ... The owner's new law is the
//   opposite shape: one trigger, one downgrade." CLOSED BY: `block_descent_trigger` has no sweep
//   loop at all. It computes the order, walks it ONCE from the cursor, takes the FIRST accepted
//   step, and returns. `DescentCursor::steps_taken` is the proof: after N triggers it is <= N, and
//   the test in `../tests/` asserts `steps_taken == triggers_fired` for every trigger that had a
//   movable cell. The property 34.3 warned would be invalidated (requirement 5, the termination
//   bound over one walk) is RESTATED where it now belongs -- over TRIGGERS: each accepted step
//   strictly decreases the total, each cell has finitely many rungs, so the total reaches the
//   budget after at most `#cells * (#rungs - 1)` triggers. See `descent_termination_bound()`.
//
//   GAP-CURSOR (34.4): "rebuilt from the DEFAULT vector ... every call starts at the same first
//   cell ... a one-shot law would downgrade the same cell on every trigger until it reached the
//   floor." CLOSED BY, BOTH HALVES TOGETHER AS 34.4 REQUIRES: `DescentCursor` carries the MODES in
//   `vectors` (read back at the top of every trigger, never re-derived from the default) and the
//   POSITION in `next_offset` (advanced past the cell that moved, so the next trigger resumes
//   where this one stopped). `descent_cursor_init` is the ONLY place a default vector is built.
//
//   GAP-AGEGATE (34.6): "an eligibility rule that excludes young blocks -- the free age proxy is
//   the block index (34.2), so the rule can be written today as `block index <= frontier - K` for
//   an owner-chosen K". CLOSED BY: `age_admits()` is that predicate, `keep_recent_blocks` is the
//   owner's K, and 34.6's own second requirement -- "the instrument that proves it" -- is
//   `demoted_oldest_block` / `demoted_newest_block`, which 34.2 named as owed and this file
//   carries. K = 0 is the OFF switch and is byte-for-byte the pre-image eligibility.
//
// ---------------------------------------------------------------------------------------------
// ⚠ WHAT IS *NOT* CLOSED, NAMED SO IT IS NOT READ AS DONE:
//
//   GAP-FREQ IS UNTOUCHED. The sort key is still `CellFrequencyKind::Absent` -- `frequency=ABSENT`
//   prints on every pass -- because there is still no producer. THIS HAS A CONSEQUENCE FOR THE
//   CURSOR, AND IT IS LOAD-BEARING: `next_offset` indexes into the order the walk builds, and that
//   order is TOTAL and STABLE only while the frequency source is absent (all keys equal, tie-break
//   `(layer, block)`). The moment a producer appears the key can change BETWEEN triggers, and a
//   saved offset would then point at a different cell. The cursor therefore does not trust the
//   offset blindly: it re-derives the order each trigger and RESUMES BY CELL IDENTITY when the
//   order has changed. See `descent_order_index` and the `order_stable` reading.
//
//   THE ORDER OF THE WALK IS (layer, block) AND THAT IS THE OWNER'S "OLD FIRST" BY ACCIDENT, NOT
//   BY DESIGN. With every key equal, the tie-break descends `layer` then `block`, so the walk
//   starts at the OLDEST block and walks upward in age. That is the desired direction, and it is
//   worth knowing it is currently a property of the TIE-BREAK and not of the sort key. `demoted_
//   newest_block` is the instrument that would catch it if it ever inverted.
//
//   A TRIGGER POLICY IS NOT PROVIDED. See the refusal above.
//
// =============================================================================================
// [F1231 2026-09-29] WIRED, NOT RETIRED -- AND THIS PARAGRAPH IS THE DECISION, WITH ITS REASON.
// =============================================================================================
// WHEN THIS FILE WAS READ BY `F1230`, IT WAS DEAD CODE: no `#include`, no caller, and the proof
// program F1202 cited (`tests/kv_descent_control_proof.cpp`) was not on the tree. The rework had
// two legal answers -- wire it or retire it -- and it takes WIRE, for three reasons that are
// checkable here rather than asserted:
//
//   (a) THE ONLY OTHER SPELLING OF THE LAW IS THIS ONE. `age_admits()`, `keep_recent_blocks` and
//       `law_holds()` exist nowhere else in the tree, so retiring the file means writing them
//       again somewhere else -- a second implementation of the rule the owner gave, which is the
//       defect class this whole project keeps paying for.
//   (b) THE TYPES ARE NOW FIELDS OF THE PLAN, WHICH IS THE OBJECT THE ENGINE CONSUMES.
//       `DescentSpec::age` is this file's `AgeGate`, `DescentSpec::cursor` is this file's
//       `DescentCursor`, and `BlockDescentPlan` carries `law_holds()` as a reading. A caller that
//       forgets the law now has to leave a field empty, which the plan PRINTS.
//   (c) THE INVARIANT IS NOW REACHABLE RATHER THAN ONLY WRITTEN: `law_holds()` is evaluated by
//       `block_descent_plan` on every trigger and printed as `law_holds=` on the plan's own
//       accounting line, so a run in which one trigger demoted more than one cell says so.
//
// ⚠ WHAT IS *NOT* WIRED, AND MUST NOT BE READ AS DONE: `block_descent_trigger` below is NOT the
// live path. The live path is `block_descent_plan(DescentSpec)`, because the engine's charge table
// is built from the plan's `block_bytes` and a second walker would be a second spelling of the
// step rule inside one engine. THE ONE THING THAT DECIDES IT: the file that owns the charge
// (`kv_block_budget_stage.h`) consumes a `BlockChargeTable`, i.e. a plan's output, so the plan
// must be the walker. This is why the include edge is
// `kv_block_descent.h -> kv_descent_control.h` and not the reverse: `block_descent_trigger` could
// then never call the plan without a cycle, so it keeps its own body and is named here as the
// mechanism's historical walker rather than silently left as an alternative to the live one.
//
// ⚠ AND ONE STALE SPELLING, CORRECTED BY NAME RATHER THAN LEFT TO CONTRADICT: section 4 below
// writes the order as `descent_order_index(layer, block) := layer * blocks + block`, i.e.
// LAYER-MAJOR. That IS the pre-image order (`kv_block_descent.h`'s tie-break was `(layer, block)`)
// and it is NOT the live one any more: [F1231] made the order AGE-MAJOR first, because the owner's
// rule is 「降级的同时新内容应当仍然保持int8」 and a layer-major walk demotes the newest block's
// layer-0 cell before it has looked at any older block. `descent_order_index_age_major` below is
// the live closed form and its static_assert pins the two apart.

#include "product/kv_tier_ladder.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace ninfer::product {

// =============================================================================================
// 0. [F1231] THE POPULATION, AS ONE EXPRESSION WITH TWO READERS. THIS IS R2's FIRST HALF.
// =============================================================================================
//
// WHY IT IS HERE AND NOT IN THE ENGINE. R2 (「降级的同时新内容应当仍然保持int8」, i.e. DEMOTE OLD
// CONTENT ONLY) is a statement about WHICH PAGES THE DESCENT MAY WALK, and when this line read the
// live path, the two readers of that fact disagreed:
//
//   * the PASS walks the pages it may retire --
//     `for (page = sequence.cold_frontier; page < limit; ++page)` with
//     `limit = (text_kv_valid - cold_keep_tokens) / kPagedKVPageSize` (min'd with the mapped
//     pages), i.e. **[cold_frontier, limit) = THE RETIREMENT SPAN**;
//   * the DESCENT's window was `text_kv_valid / kPagedKVPageSize - cold_frontier`, i.e.
//     **[cold_frontier, text_kv_valid / P) = THE RESIDENT SPAN** -- the same left edge and a
//     STRICTLY LARGER right edge.
//
// MEASURED CONSEQUENCE, IN EVERY ARM (`P3_B_64k_65536_r1.log`, the pair quoted at its own
// frontiers): `planned_blocks=50` against `blocks_seen=48`, and the two pages the plan had that
// the pass never walked are its TWO NEWEST -- so the walk that printed `demoted=800 of 800`
// demoted pages the engine's own floor forbids to leave. **That is the inversion F1230 measured.**
//
// THE FIX IS NOT A SECOND FORMULA: it is the same formula, moved to where both readers can call
// it, so the plan's population IS the pass's population up to the pages the pass skips at runtime
// (the semantic filter, `can_cold_transfer`, and pool exhaustion). Those are printed by the stage
// as `population_delta` / `unspent_planned_bytes` (kv_block_budget_stage.h) rather than hidden.
//
// ⚠ WHAT THIS DOES *NOT* CLAIM: the plan still cannot see pages BELOW `cold_frontier` (already
// retired) and it still covers pages the semantic filter will refuse. The first is a real limit of
// "old content" in this engine and is named where the caller is (`program_impl.h`); the second is
// now a column, not a silent over-spend.
[[nodiscard]] constexpr std::uint32_t descent_retire_limit_pages(std::uint32_t valid_tokens,
                                                                 std::uint32_t keep_tokens,
                                                                 std::uint32_t page_tokens,
                                                                 std::uint32_t mapped_pages) noexcept {
    if (page_tokens == 0) { return 0; }
    const std::uint32_t cold_pages =
        valid_tokens > keep_tokens ? (valid_tokens - keep_tokens) / page_tokens : 0U;
    return cold_pages < mapped_pages ? cold_pages : mapped_pages;
}

// THE SPAN THE DESCENT PLANS OVER: `[cold_frontier, limit)`, the pages the pass will WALK. A
// `limit` that has caught the frontier means an EMPTY population, which is the honest reading at
// the end of a prefill (the measured series ends `origin=1006 blocks=2,2`), and it is returned as
// 0 rather than wrapped.
[[nodiscard]] constexpr std::uint32_t descent_population_pages(std::uint32_t valid_tokens,
                                                              std::uint32_t keep_tokens,
                                                              std::uint32_t page_tokens,
                                                              std::uint32_t cold_frontier,
                                                              std::uint32_t mapped_pages) noexcept {
    const std::uint32_t limit =
        descent_retire_limit_pages(valid_tokens, keep_tokens, page_tokens, mapped_pages);
    return limit > cold_frontier ? limit - cold_frontier : 0U;
}

// R3's K, FROM THE ENVIRONMENT, WITH THE PRE-IMAGE AS ITS DEFAULT. `keep_recent_blocks` is the
// owner's K (section 2 above) and its OFF value is 0, which is byte-for-byte the pre-image
// eligibility -- so an unset environment reproduces the old eligibility exactly and a set one is
// the only way the gate can change anything. THE DEFAULT IS 0 AND THAT IS A NAMED CHOICE, NOT AN
// OVERSIGHT: with the population now being the retirement span (above), every block in it is
// already older than the engine's own keep floor (`cold_keep_tokens`), so the gate's live value
// would spare nothing; K is armed here so that a FUTURE widening of the population cannot silently
// re-invert R3 the way the resident-span window did.
inline constexpr const char* kKvDescentKeepRecentEnv = "NINFER_KV_DESCENT_KEEP_RECENT_PAGES";

[[nodiscard]] inline std::uint32_t descent_keep_recent_pages_from_env() noexcept {
    const char* const text = std::getenv(kKvDescentKeepRecentEnv);
    if (text == nullptr || *text == '\0') { return 0; }
    char* end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (end == text || parsed < 0) { return 0; }
    return static_cast<std::uint32_t>(parsed);
}

// THE LIVE CURSOR IS DECLARED AT THE END OF THIS FILE, after the types it names -- see
// `descent_cursor_live()` beside the closing brace.

// ---------------------------------------------------------------------------------------------
// 1. THE TRIGGER, AS A TYPE THE CALLER MUST SUPPLY.
// ---------------------------------------------------------------------------------------------
//
// THE VALUES ARE THE EVENTS THIS TREE CAN ACTUALLY OBSERVE TODAY, and no others. `None` is the
// default so a cursor that is never handed a trigger does nothing -- the law's own "降级应当不是一直
// 进行" as a default-constructed object rather than as an `if` someone has to remember.
//
// ⚠ `PendingBudget` IS *NOT* `EveryPass`. The difference is the whole of point (iv): a pass
// boundary is a thing that happens on every pass whether or not anything is under pressure, and
// firing a demotion on it is exactly the continuous preemption the owner forbids
// (「避免一直抢占」). It is kept as a separate value so a caller that wires it can be seen to have
// made that choice, and `triggers_fired` counts it either way.
enum class DescentTrigger : std::uint8_t {
    None          = 0,  // no trigger: this call does nothing at all
    BudgetCrossed = 1,  // the observed total exceeded the budget (the pressure event)
    Explicit      = 2,  // the caller asked for exactly one demotion
    PendingBudget = 3,  // ⚠ the pass boundary habit: legal, but it IS preemption. Named, not hidden
};

[[nodiscard]] constexpr const char* descent_trigger_name(DescentTrigger trigger) noexcept {
    switch (trigger) {
    case DescentTrigger::BudgetCrossed: return "budget-crossed";
    case DescentTrigger::Explicit: return "explicit";
    case DescentTrigger::PendingBudget: return "pass-boundary(preemption-risk)";
    case DescentTrigger::None: break;
    }
    return "none";
}

// ---------------------------------------------------------------------------------------------
// 2. THE AGE GATE. GAP-AGEGATE, CLOSED. `keep_recent_blocks` IS THE OWNER'S K.
// ---------------------------------------------------------------------------------------------
struct AgeGate {
    // THE OWNER'S K: how many of the NEWEST blocks are never candidates. K = 0 is OFF -- every
    // block is eligible and the predicate is the pre-image. The default is 0 so an unwired caller
    // gets the old behaviour exactly.
    std::uint32_t keep_recent_blocks = 0;
    // THE FRONTIER: the highest block index the caller knows about. The engine's own age axis --
    // blocks are created at runtime from arriving text and the window slides forward over them
    // (blob_F1192 34.2, measured: plan_origin = 0, 46, 94, 142, ... and blocks_seen 46, 48, ...).
    std::uint32_t frontier = 0;

    // AGED-IN. Strictly: a block is a candidate only when it is `keep_recent_blocks` BEHIND the
    // frontier. Written without a subtraction that can wrap, which is the bug this shape avoids.
    [[nodiscard]] constexpr bool admits(std::uint32_t block) const noexcept {
        return block + keep_recent_blocks <= frontier;
    }
    // The newest block the gate will ever admit. Equal to `frontier` when K is 0.
    [[nodiscard]] constexpr std::uint32_t newest_eligible() const noexcept {
        return frontier >= keep_recent_blocks ? frontier - keep_recent_blocks : 0;
    }

    [[nodiscard]] std::string describe() const {
        return "age_gate[K=" + std::to_string(keep_recent_blocks) +
               " frontier=" + std::to_string(frontier) +
               " newest_eligible=" + std::to_string(newest_eligible()) + "]";
    }
};

// ---------------------------------------------------------------------------------------------
// 3. THE CURSOR. BOTH HALVES OF GAP-CURSOR, IN ONE OBJECT, BECAUSE 34.4 SAYS THEY MUST LAND
//    TOGETHER ("a cursor without carried modes resumes at a position in a population that has been
//    reset, and carried modes without a cursor re-downgrade the first cell").
// ---------------------------------------------------------------------------------------------
struct DescentCursor {
    // HALF ONE: THE MODES. Read back at the top of every trigger; NEVER re-derived from the default
    // except in `descent_cursor_init`. This is the half whose absence made the plan's prologue
    // (`kv_block_descent.h:364-371`) start from all-int8 on every call.
    std::vector<BlockVector> vectors;
    // HALF TWO: THE POSITION. `next_offset` is an index into the walk's own total order
    // (`descent_order_index`). It is advanced past the cell that moved, so the next trigger resumes
    // rather than repeating. `last_block` / `last_layer` are the SAME position by IDENTITY, kept so
    // a resumed trigger can verify it landed where the previous one stopped.
    std::uint32_t next_offset = 0;
    std::uint32_t last_block = 0;
    std::uint32_t last_layer = 0;
    bool have_last = false;
    std::uint32_t order_blocks = 0;
    std::uint32_t order_layers = 0;
    // ⭐ [F1231 window] THE POPULATION'S IDENTITY, WHICH THE SHAPE ALONE DOES NOT GIVE.
    // `vectors` is POSITIONAL: slot i is the mode of the i-th block of the population it was built
    // for. A window that SLIDES (the prefill advances; `[cold_frontier, limit)` moves) keeps the
    // same COUNT while covering DIFFERENT PAGES, so a shape-only check accepted a carry whose modes
    // then landed on shifted pages -- the plan's `block_bytes`, and therefore the stage's ceiling
    // accounting, describing pages the demotion never touched. **That is a wrong number, not a
    // conservative default, and it is wrong under EVERY trigger policy.** So the cursor records the
    // first page index it was built for, and a moved window REFUSES it by name (`origin 142->190`).
    //
    // ⚠ THE CONSEQUENCE, STATED SO IT IS NOT MISTAKEN FOR A WORKING CARRY: refusing means **no
    // accumulation across passes**, because this engine's windows barely overlap (measured: a
    // ~50-page population advancing ~48 pages per pass has ~2 pages of overlap). That is a fact
    // about the population's CHURN, not a defect of the check -- and it is why a per-population
    // carry cannot make a per-cell allocation persist. A persistence home for a cell's mode does
    // not exist in this engine today (the durable knobs are PER LAYER: the storage dtype table);
    // NAMED AS OWED AND NOT INVENTED HERE.
    std::uint32_t origin = 0;
    bool order_stable = true;   // false once a trigger observed the order had changed under it

    // THE READINGS THE LAW OWES.
    std::uint32_t triggers_fired = 0;      // how many triggers this cursor has been handed
    std::uint32_t steps_taken = 0;         // accepted demotions; <= triggers_fired by construction
    std::uint32_t refused_triggers = 0;    // triggers that found nothing movable
    std::uint32_t cells_demoted = 0;       // distinct (block, layer) that ever moved
    std::uint32_t blocks_absorbed = 0;     // NEW blocks that entered at int8
    std::uint32_t last_step_block = 0;
    std::uint32_t last_step_layer = 0;
    CellMode last_step_to = CellMode::Int8;
    DescentTrigger last_trigger = DescentTrigger::None;
    // ⭐ 34.2's FIFTH AND SIXTH NUMBERS, OWED THERE AND CARRIED HERE. The block index IS the age
    // proxy this tree already has, so these two are the reading that makes "only OLD cells moved"
    // a MEASUREMENT rather than a claim: under point (ii) `demoted_newest_block` must stay below
    // the window's newest block. Whole-window descent would show them equal.
    std::uint32_t demoted_oldest_block = 0;
    std::uint32_t demoted_newest_block = 0;
    bool any_demoted = false;
    std::int64_t total_bytes = 0;          // the carried charge, in the charge's own ruler

    [[nodiscard]] bool initialised() const noexcept { return !vectors.empty(); }
    [[nodiscard]] std::uint32_t blocks() const noexcept {
        return static_cast<std::uint32_t>(vectors.size());
    }
    [[nodiscard]] std::uint32_t layers() const noexcept {
        return vectors.empty() ? 0U : vectors.front().layers;
    }
    // ONE TRIGGER => ONE DEMOTION, AS A PREDICATE A TEST CAN ASSERT.
    [[nodiscard]] bool law_holds() const noexcept { return steps_taken <= triggers_fired; }

    [[nodiscard]] std::string describe() const {
        std::string out = "descent-cursor";
        out += " triggers=" + std::to_string(triggers_fired);
        out += " steps=" + std::to_string(steps_taken);
        out += " refused_triggers=" + std::to_string(refused_triggers);
        out += " law_holds=" + std::string(law_holds() ? "yes" : "NO");
        out += " cells_demoted=" + std::to_string(cells_demoted);
        out += " blocks=" + std::to_string(blocks());
        out += " absorbed_new=" + std::to_string(blocks_absorbed);
        out += " next_offset=" + std::to_string(next_offset);
        out += " order_stable=" + std::string(order_stable ? "yes" : "no");
        out += " demoted_block_range=[" + std::to_string(demoted_oldest_block) + "," +
               std::to_string(demoted_newest_block) + "]";
        out += " any_demoted=" + std::string(any_demoted ? "yes" : "no");
        out += " total_bytes=" + std::to_string(total_bytes);
        return out;
    }
};

// ---------------------------------------------------------------------------------------------
// 3b. [F1231] THE LIVE ORDER'S CLOSED FORM: **AGE-MAJOR**, i.e. `block` first.
// ---------------------------------------------------------------------------------------------
//
// THE PRE-IMAGE ORDER WAS `(key, layer, block)`, so with every key equal (frequency ABSENT) the
// order was LAYER-major: layer 0 of EVERY block -- including the newest block in the window --
// before layer 1 of any block. F1230 measured what that does to the owner's law: `demoted=800 of
// 800`, one call, the whole population, newest included.
//
// THE LIVE ORDER IS `(key, block, layer)`: the frequency key first (so a future producer still
// dominates, exactly as the sort intends), then the AGE AXIS, then the layer. With the key absent
// the walk therefore starts at the OLDEST block of the population and descends a block's layers
// before moving to a younger block. `cell_rung`'s ladder is one rung at a time either way -- this
// is the ORDER the walk visits cells in, and nothing else.
//
// THE FIXED-POINT CHECK THAT MAKES IT A CRITERION AND NOT A COINCIDENCE: the order is a total
// order on `(block, layer)` and `descent_order_index_age_major` is its closed form, so the
// cursor's offset means the same thing in two different calls.
[[nodiscard]] constexpr std::uint32_t descent_order_index_age_major(std::uint32_t block,
                                                                   std::uint32_t layer,
                                                                   std::uint32_t layers) noexcept {
    return block * layers + layer;
}

// ---------------------------------------------------------------------------------------------
// 4. THE ORDER, AND WHY IT IS AN INDEX AND NOT A VECTOR.
// ---------------------------------------------------------------------------------------------
//
// `block_descent_plan` builds a `std::vector<Key>` and sorts it. For a ONE-SHOT law that vector is
// rebuilt on every trigger and thrown away, and the cursor has to name a position in it. So the
// order is written as a TOTAL, CLOSED-FORM index instead:
//
//     order_index(layer, block)  :=  layer * blocks + block        (ascending)
//
// and the walk reads that ascending. It is EXACTLY the order `block_descent_plan`'s tie-break
// produces when the frequency source is absent: `sort` by `(key, layer, block)` with every `key`
// equal gives `(layer, block)` lexicographic -- which is this index. So the two spellings agree,
// and this one can be resumed from a number.
//
// ⚠ AND WHEN A FREQUENCY PRODUCER ARRIVES THEY STOP AGREEING, which is the residual named at the
// top. `frequency_order_index` below is the spelling that would take a producer; it is written so
// the change is one function and not a rediscovery, and `descent_order_index` documents that today
// the key is 0 for every cell.
[[nodiscard]] constexpr std::uint32_t descent_order_index(std::uint32_t layer, std::uint32_t block,
                                                          std::uint32_t blocks) noexcept {
    return layer * blocks + block;
}

[[nodiscard]] constexpr std::uint32_t descent_order_cell_block(std::uint32_t offset,
                                                              std::uint32_t blocks) noexcept {
    return blocks == 0 ? 0U : offset % blocks;
}

[[nodiscard]] constexpr std::uint32_t descent_order_cell_layer(std::uint32_t offset,
                                                              std::uint32_t blocks) noexcept {
    return blocks == 0 ? 0U : offset / blocks;
}

// ---------------------------------------------------------------------------------------------
// 5. INIT: THE START, AND THE ABSORPTION OF NEW CONTENT. POINTS (i) AND (iii).
// ---------------------------------------------------------------------------------------------
//
// THE ONE PLACE A DEFAULT VECTOR IS BUILT. Every other entry point reads `cursor.vectors`. If a
// caller goes through `descent_absorb_new_blocks` it never builds a default for an old block again,
// which is the property GAP-CURSOR's second half was about.
[[nodiscard]] inline bool descent_cursor_init(DescentCursor& cursor, std::uint32_t blocks,
                                              std::uint32_t layers,
                                              const std::vector<CellAllowedSet>& allowed) {
    if (blocks == 0 || layers == 0) { return false; }
    const std::uint32_t clamped = layers < kBlockVectorLayerAxis ? layers : kBlockVectorLayerAxis;
    // (i) START = ALL-LAYER int8. Built through `cell_default_tier`, so the start is the tree's
    // default and not a literal -- the same function `block_descent_plan` uses.
    BlockVector base = block_vector_default(clamped);
    for (std::uint32_t layer = 0; layer < clamped; ++layer) {
        const CellAllowedSet set =
            (allowed.empty() || layer >= allowed.size()) ? cell_all_modes() : allowed[layer];
        base.set(layer, cell_default_tier(set).mode);
    }
    cursor = DescentCursor{};
    cursor.vectors.assign(blocks, base);
    cursor.order_blocks = blocks;
    cursor.order_layers = clamped;
    cursor.next_offset = 0;
    return true;
}

// (iii) NEW CONTENT ENTERS AT int8. An arrival is an APPEND of default vectors; no existing block's
// vector is touched, so the modes the cursor carries survive an arrival. The return value is the
// number of blocks absorbed.
//
// ⚠ THE TWO THINGS THIS FUNCTION DOES NOT DO, BOTH BY DESIGN: it does not demote the arrivals (that
// is the age gate's job, and the arrivals are the newest by definition) and it does not renumber
// the existing blocks (block index is the age axis -- renumbering would make the age proxy a lie).
[[nodiscard]] inline std::uint32_t descent_absorb_new_blocks(DescentCursor& cursor,
                                                             std::uint32_t new_blocks,
                                                             std::uint32_t layers,
                                                             const std::vector<CellAllowedSet>& allowed) {
    if (new_blocks == 0 || !cursor.initialised()) { return 0; }
    const std::uint32_t clamped = layers < kBlockVectorLayerAxis ? layers : kBlockVectorLayerAxis;
    BlockVector base = block_vector_default(clamped);
    for (std::uint32_t layer = 0; layer < clamped; ++layer) {
        const CellAllowedSet set =
            (allowed.empty() || layer >= allowed.size()) ? cell_all_modes() : allowed[layer];
        base.set(layer, cell_default_tier(set).mode);
    }
    cursor.vectors.insert(cursor.vectors.end(), new_blocks, base);
    cursor.blocks_absorbed += new_blocks;
    return new_blocks;
}

// A NEW BLOCK'S VECTOR IS THE DEFAULT, AS A PREDICATE A TEST CAN ASSERT.
[[nodiscard]] inline bool descent_new_block_entered_at_int8(const DescentCursor& cursor,
                                                            std::uint32_t block) {
    if (block >= cursor.vectors.size()) { return false; }
    const BlockVector& vector = cursor.vectors[block];
    for (std::uint32_t layer = 0; layer < vector.layers; ++layer) {
        if (vector.mode[layer] != CellMode::Int8) { return false; }
    }
    return vector.layers != 0;
}

// ---------------------------------------------------------------------------------------------
// 5a. [F1245 kvcarry] THE RE-ANCHOR: IDENTITY, NOT SHAPE.
// ---------------------------------------------------------------------------------------------
//
// WHAT WAS MEASURED, AND WHAT THIS CLOSES. The F1244 battery printed
// `carry=REFUSED(shape-mismatch(blocks/layers changed))` on 24-25 of every 27-pass block-KV arm
// (`dl/cmpfire/out/speed/B_def/stderr.txt`, and the same column in `dl/kvpost/out/GRAN.tsv` as
// `carry_seen=REFUSED(shape-mismatch),carried`), and the plan's own line states the consequence:
// `accumulation=by-name-off(a refused carry means NO accumulation across passes)`. The rung census
// therefore read `rk4v4=16` of 15,696 cells on EVERY pass -- 0.10 %, one block, the same block
// re-demoted -- and the 26 x 16-cell steps bought 16 cells of mix.
//
// WHY THE REFUSAL EXISTED, AND WHY IT IS NOT SIMPLY REMOVED. `cursor.vectors` is POSITIONAL: slot i
// is the mode of the i-th block of the population it was built for. Applying it to a DIFFERENT
// population BY INDEX is exactly how a mode lands on a page the demotion never touched -- a wrong
// number under every trigger policy, which is what F1231 refused. THAT CHECK IS KEPT. What moves is
// the CURRENCY it is made in: the ABSOLUTE PAGE INDEX, which is the one coordinate in which two
// different windows can be compared at all.
//
// THE MAPPING. A population is `[origin, origin + blocks)` in absolute page indices -- the call
// site's own reading (`stage_window_begin = sequence.cold_frontier`, and the plan's
// `population_origin`; the engine's page IS the 64-token block, program_impl.h's unload filter
// says so). Local index i therefore IS page `origin + i`, so "the old local block i seen from the
// new window" is `(old_origin + i) - new_origin`, and the two names below carry the only two things
// that can go wrong:
//   * the page has LEFT the window (it was RETIRED -- the left edge is a moving frontier, not a
//     fixed origin): DROPPED, never clamped, because a clamped mode is a mode on a different page;
//   * the local index has no carried mode (the page ARRIVED): it stays at the DEFAULT, which is
//     what `plan.vectors.assign(blocks, base)` already put there.
//
// ⚠ THE LIMIT, NAMED HERE RATHER THAN LEFT FOR A READER TO DISCOVER FROM A CENSUS. This makes the
// MODES survive a window that moves. It cannot make a demoted cell survive a page that has already
// gone, and on this engine's geometry the retirement leg takes at least one page per trigger and
// the page it takes first is the window's left edge -- which is where the walk starts. That is a
// property of the POPULATION (it IS the retirement span: `descent_population_pages` above), not of
// this mapping. So a reader must take the census columns together with the re-anchor columns the
// plan prints (`carry_reanchor[kept=.. dropped=.. new=..]`): the first says what the walk decided,
// the second says how much of it the next trigger can still see.
[[nodiscard]] constexpr bool descent_reanchor_local(std::uint32_t old_origin,
                                                    std::uint32_t new_origin,
                                                    std::uint32_t new_blocks,
                                                    std::uint32_t local,
                                                    std::uint32_t& out_local) noexcept {
    // Compared in the ABSOLUTE frame and in 64 bits, so neither subtraction can wrap -- the same
    // discipline `AgeGate::admits` states for its own axis.
    const std::uint64_t absolute = static_cast<std::uint64_t>(old_origin) + local;
    if (absolute < static_cast<std::uint64_t>(new_origin)) { return false; }
    const std::uint64_t index = absolute - static_cast<std::uint64_t>(new_origin);
    if (index >= static_cast<std::uint64_t>(new_blocks)) { return false; }
    out_local = static_cast<std::uint32_t>(index);
    return true;
}

// THE POSITION, PROJECTED BY THE SAME IDENTITY. `next_offset` is an index into the walk's own
// AGE-MAJOR order (`descent_order_index_age_major` = `block * layers + layer`, which is the order
// the live plan sorts to), so its PAGE is `next_offset / layers`. A position whose page has left
// the window is NOT a position in the new population, and it says so (`false`) instead of being
// clamped: the caller then restarts the lap at the window's first cell, which is the only reading
// of "continue" that does not silently skip cells. `layers == 0` is not projectable by definition.
[[nodiscard]] constexpr bool descent_reanchor_position(std::uint32_t old_origin,
                                                       std::uint32_t new_origin,
                                                       std::uint32_t new_blocks,
                                                       std::uint32_t layers,
                                                       std::uint32_t next_offset,
                                                       std::uint32_t& out_offset) noexcept {
    if (layers == 0) { return false; }
    const std::uint32_t block = next_offset / layers;
    const std::uint32_t layer = next_offset % layers;
    std::uint32_t moved = 0;
    if (!descent_reanchor_local(old_origin, new_origin, new_blocks, block, moved)) { return false; }
    out_offset = descent_order_index_age_major(moved, layer, layers);
    return true;
}

// ---------------------------------------------------------------------------------------------
// 5b. THE ORDER OF THE DEMOTIONS: WHAT A RESUME-FROM-POSITION CURSOR ACTUALLY DOES.
// ---------------------------------------------------------------------------------------------
//
// ⚠ WRITTEN AFTER MEASURING TWICE, BECAUSE THE FIRST DRAFT OF THIS SECTION WAS WRONG AND SO WAS
// THE FIRST DRAFT OF ITS CORRECTION. Both wrong versions are kept below, because a mechanism
// asserted rather than measured is exactly what this line refuses and the record of the correction
// is the evidence that it was measured.
//
//   DRAFT 1 said: "a cursor that resumes from its position is DEPTH-FIRST by default -- it descends
//   cell (0,0) to the floor before it starts (1,0)". REFUTED BY MEASUREMENT: a cursor that ADVANCES
//   its position by one cell per trigger visits every cell once per LAP of the ring, and ONE LAP IS
//   ONE ROUND. `../tests/` at a half-plan budget on 48x16 prints 1,249 demotions over 1,248 distinct
//   cells for the advancing cursor -- the level-wise shape, reached by the advance itself.
//
//   DRAFT 2 said: "a position-frozen cursor demotes ONE cell and never touches the other 767"
//   (34.4's sentence, taken as a prediction of what `Sticky` would do). REFUTED BY MEASUREMENT: the
//   inner scan still runs, so a frozen position re-demotes that cell until it reaches the floor and
//   THEN moves on. MEASURED: `Sticky` needs 1,585 demotions over 1,584 cells for the SAME bytes the
//   advancing cursor buys in 1,249 -- **27 % more demotions, the same result**. 34.4's literal
//   "never touch the rest of the population" needs the INNER SCAN frozen too, which is precisely the
//   pre-image's statelessness and is demonstrated by the plan-identity reading in `../tests/`
//   (C8c: two identical `block_descent_plan` calls give identical vectors), not by a third mode.
//
// SO THE FIELD MEANS THE ONE THING THAT IS ACTUALLY DIFFERENT, AND THE DIFFERENCE IS MEASURED:
//
//   * `AdvancePerTrigger` (the ZERO value, the default, the LAW): the position moves past the cell
//     that moved, so trigger N+1 examines the NEXT cell. One lap = one round = "every cell moves one
//     rung before any cell moves two", which is `block_descent_plan`'s own recorded shape and the
//     order's 「一档一档降的」 at the population level. MEASURED: 1,249 demotions for a half-plan
//     budget on 48x16, over 1,248 distinct cells.
//   * `Sticky` (the GAP-CURSOR failure, kept because a defect that cannot be demonstrated cannot be
//     cited): the position does NOT advance, so the next trigger re-examines the cell it just
//     demoted -- DEPTH-FIRST at the cell level. MEASURED on the same arm: 1,585 demotions over 1,584
//     cells for the same total, i.e. the same bytes at 27 % more rung spend.
//
// BOTH ARE OFFERED BECAUSE THE CONTRAST IS THE PROOF. A reader can price GAP-CURSOR by running it.
enum class DescentOrder : std::uint8_t {
    AdvancePerTrigger = 0,  // THE LAW: the position moves on, so a lap is a round
    Sticky            = 1,  // ⚠ the GAP-CURSOR failure: the same cell is re-demoted to the floor
};

[[nodiscard]] constexpr const char* descent_order_name(DescentOrder order) noexcept {
    switch (order) {
    case DescentOrder::AdvancePerTrigger: return "advance-per-trigger";
    case DescentOrder::Sticky: return "sticky(GAP-CURSOR)";
    }
    return "?";
}

// ---------------------------------------------------------------------------------------------
// 6. THE ONE TRIGGER. GAP-ONESHOT, CLOSED: **AT MOST ONE ACCEPTED STEP, THEN RETURN.**
// ---------------------------------------------------------------------------------------------
struct DescentTriggerSpec {
    std::int64_t budget_bytes = 0;                 // <= 0 means OFF (nothing to demote FOR)
    std::int32_t kv_heads = 0;
    std::int32_t ladder_ceiling_depth = static_cast<std::int32_t>(kLatticeFloorDepth);
    AgeGate age{};
    std::vector<CellAllowedSet> allowed;           // size == layers; empty means `cell_all_modes()`
    // THE LAW ITSELF, AS A FIELD, SO A READER CAN SEE IT IS NOT A PARAMETER. `false` would restore
    // "walk to fit in one call" -- the shape the owner rejected -- and it exists ONLY so the test in
    // `../tests/` can measure both shapes against each other. A caller has no reason to set it.
    bool one_demotion_per_trigger = true;
    // WHETHER THE POSITION ADVANCES AFTER A STEP. See §5b: `AdvancePerTrigger` is the law, and
    // `Sticky` is the GAP-CURSOR failure kept so the contrast can be measured rather than described.
    DescentOrder order = DescentOrder::AdvancePerTrigger;
};

struct DescentTriggerResult {
    DescentTrigger trigger = DescentTrigger::None;
    bool fired = false;             // the trigger was not `None`
    bool moved = false;             // an accepted step was taken
    bool at_budget = false;         // the carried total already fits
    bool no_movable = false;        // the order was walked and nothing could move
    bool age_blocked_all = false;   // the gate excluded every cell (a named reason, not "nothing")
    std::uint32_t order_len = 0;
    std::uint32_t scanned = 0;
    std::uint32_t block = 0;
    std::uint32_t layer = 0;
    CellMode from = CellMode::Int8;
    CellMode to = CellMode::Int8;
    std::int32_t plane_saved = 0;
    std::int64_t total_saved = 0;
    StepRefusal refusal = StepRefusal::None;

    [[nodiscard]] std::string describe() const {
        std::string out = "trigger=";
        out += descent_trigger_name(trigger);
        out += " fired=" + std::string(fired ? "yes" : "no");
        out += " moved=" + std::string(moved ? "yes" : "no");
        if (moved) {
            out += " cell=(" + std::to_string(block) + "," + std::to_string(layer) + ")";
            out += " "; out += cell_mode_name(from);
            out += "->"; out += cell_mode_name(to);
            out += " plane_saved=" + std::to_string(plane_saved);
            out += " total_saved=" + std::to_string(total_saved);
        } else {
            out += " refusal=";
            out += (age_blocked_all ? "age-gate-excluded-every-cell" : step_refusal_name(refusal));
        }
        out += " scanned=" + std::to_string(scanned) + "/" + std::to_string(order_len);
        return out;
    }
};

// THE TRIGGER HANDLER. THE WHOLE OF POINT (iv) IS THE `return` INSIDE THE LOOP.
//
// THE WALK, EXACTLY: (1) read the modes from the cursor; (2) if the carried total fits, say so and
// return -- "触发一次降级一次" does not mean "demote even when there is nothing to demote for";
// (3) walk the total order ONCE, starting at the cursor's position and wrapping, taking the FIRST
// cell that (a) is admitted by the age gate, (b) is below the rung ceiling, (c) has a legal step
// with positive currency on the charge's ruler; (4) take that ONE step, update the cursor's modes,
// position, and the two age readings, and RETURN. `one_demotion_per_trigger = false` is the old
// walk-to-fit and exists only for the contrast test.
[[nodiscard]] inline DescentTriggerResult block_descent_trigger(
    DescentCursor& cursor, const DescentTriggerSpec& spec, DescentTrigger trigger) {
    DescentTriggerResult result{};
    result.trigger = trigger;
    cursor.last_trigger = trigger;
    if (trigger == DescentTrigger::None || !cursor.initialised()) { return result; }
    result.fired = true;
    cursor.triggers_fired += 1;

    const std::uint32_t blocks = cursor.blocks();
    const std::uint32_t layers = cursor.layers();
    if (blocks == 0 || layers == 0 || spec.kv_heads <= 0) { return result; }

    auto allowed_for = [&spec, layers](std::uint32_t layer) -> CellAllowedSet {
        if (spec.allowed.empty() || layer >= spec.allowed.size() || layer >= layers) {
            return cell_all_modes();
        }
        return spec.allowed[layer];
    };

    // (2) THE BUDGET, IN THE CHARGE'S OWN RULER, READ FROM THE CARRIED MODES. Recomputed from
    // `cursor.vectors` -- NOT incremented -- so a cursor that was handed modes from elsewhere cannot
    // carry a total that disagrees with them.
    cursor.total_bytes = 0;
    for (std::uint32_t b = 0; b < blocks; ++b) {
        cursor.total_bytes += cell_vector_charge(cursor.vectors[b], spec.kv_heads).bytes;
    }
    if (spec.budget_bytes <= 0) { cursor.refused_triggers += 1; return result; }
    if (cursor.total_bytes <= spec.budget_bytes) {
        result.at_budget = true;
        return result;
    }

    std::int32_t ceiling = spec.ladder_ceiling_depth;
    if (ceiling < 0) { ceiling = 0; }
    if (ceiling > static_cast<std::int32_t>(kLatticeFloorDepth)) {
        ceiling = static_cast<std::int32_t>(kLatticeFloorDepth);
    }

    const std::uint32_t order_len = blocks * layers;
    result.order_len = order_len;
    const std::uint32_t start = cursor.next_offset % order_len;
    std::uint32_t age_excluded = 0;

    // THE SINGLE PASS. `offset` runs once around the ring, and the first accepted step RETURNS.
    for (std::uint32_t i = 0; i < order_len; ++i) {
        const std::uint32_t offset = (start + i) % order_len;
        const std::uint32_t layer = descent_order_cell_layer(offset, blocks);
        const std::uint32_t block = descent_order_cell_block(offset, blocks);
        result.scanned += 1;

        // (ii) THE AGE GATE. GAP-AGEGATE, CLOSED. A young block is not a candidate, and the
        // refusal is COUNTED so "the gate excluded everything" cannot be misread as "nothing was
        // movable for a budget reason".
        if (!spec.age.admits(block)) { age_excluded += 1; continue; }

        BlockVector& vector = cursor.vectors[block];
        const CellMode current = vector.at(layer);
        const std::int32_t depth = ladder_depth(current);
        if (depth >= 0 && depth >= ceiling) { continue; }   // already at the rung ceiling

        const CellStep step = cell_next_tier(allowed_for(layer), current);
        result.block = block;
        result.layer = layer;
        result.from = current;
        result.refusal = step.refusal;
        if (!step.legal() || step.bytes_saved <= 0) { continue; }

        // ⭐ THE ONE STEP. Everything after this is bookkeeping and the function does not loop again.
        vector.set(layer, step.to);
        result.moved = true;
        result.to = step.to;
        result.plane_saved = step.bytes_saved;
        result.total_saved = ladder_step_total_delta(step, spec.kv_heads);
        cursor.total_bytes -= result.total_saved;
        cursor.steps_taken += 1;
        cursor.cells_demoted += 1;
        cursor.last_step_block = block;
        cursor.last_step_layer = layer;
        cursor.last_step_to = step.to;
        // THE POSITION. `AdvancePerTrigger` MOVES IT PAST THE CELL THAT MOVED -- GAP-CURSOR's
        // first half, and the line that makes a lap of the ring equal a ROUND (§5b).
        // `Sticky` does NOT move it, which is 34.4's failure reproduced on purpose.
        if (spec.order == DescentOrder::AdvancePerTrigger) {
            cursor.next_offset = (offset + 1) % order_len;
        } else {
            cursor.next_offset = offset;
        }
        cursor.last_block = block;
        cursor.last_layer = layer;
        cursor.have_last = true;
        // THE AGE READINGS -- 34.2's fifth and sixth numbers, taken HERE, at the one place a
        // demotion happens, so they cannot drift from the event they describe.
        if (!cursor.any_demoted) {
            cursor.demoted_oldest_block = block;
            cursor.demoted_newest_block = block;
            cursor.any_demoted = true;
        } else {
            if (block < cursor.demoted_oldest_block) { cursor.demoted_oldest_block = block; }
            if (block > cursor.demoted_newest_block) { cursor.demoted_newest_block = block; }
        }
        return result;   // <<< ONE TRIGGER, ONE DEMOTION. THE `return` IS THE LAW.
    }

    // NOTHING MOVED. Two named reasons, never one.
    if (age_excluded == result.scanned && result.scanned != 0) {
        result.age_blocked_all = true;
    } else {
        result.no_movable = true;
    }
    cursor.refused_triggers += 1;
    return result;
}

// ---------------------------------------------------------------------------------------------
// 7. THE TERMINATION BOUND, RESTATED WHERE THE ONE-SHOT LAW PUT IT (34.3's cost, paid here).
// ---------------------------------------------------------------------------------------------
//
// 34.3 warned verbatim that the one-shot law "INVALIDATES A PINNED REQUIREMENT": requirement 5 is a
// statement about a walk to fit, and under one-shot "the property that actually matters moves to
// the TRIGGER". So the bound is stated over TRIGGERS, and it is the same arithmetic: every accepted
// step strictly decreases the total (the currency is strictly positive -- `cell_next_tier`'s asserts)
// and every cell has at most `kLatticeFloorDepth` rungs below it, so
//
//     triggers_to_fit  <=  #cells * kLatticeFloorDepth   =  blocks * layers * 3
//
// and this is NOT a budget-dependent hope: it is the count of possible accepted steps.
[[nodiscard]] constexpr std::uint64_t descent_termination_bound(std::uint32_t blocks,
                                                               std::uint32_t layers) noexcept {
    return static_cast<std::uint64_t>(blocks) * layers * kLatticeFloorDepth;
}

// ---------------------------------------------------------------------------------------------
// 8. THE GAP LEDGER, AS DATA. A caller can print it; a test asserts on it.
// ---------------------------------------------------------------------------------------------
enum class ControlGap : std::uint8_t {
    OneShot = 0,   // 34.3
    Cursor  = 1,   // 34.4
    AgeGate = 2,   // 34.6
};

// ⚠ THE THIRD STATE IS THE HONEST ONE AND IT IS NOT "DONE". Each gap is closed as a MECHANISM and
// each leaves a NAMED POLICY RESIDUAL that this header cannot decide, because 34.7 assigns it to
// the owner. A reader who sees "closed" without the residual would conclude the feature is finished.
enum class GapStatus : std::uint8_t {
    ClosedMechanismPolicyOwed = 0,   // the mechanism is here; the policy is the owner's
    Closed                    = 1,   // mechanism and policy both closed
    Open                      = 2,   // named, not closed
};

[[nodiscard]] constexpr GapStatus control_gap_status(ControlGap gap) noexcept {
    switch (gap) {
    case ControlGap::OneShot: return GapStatus::ClosedMechanismPolicyOwed;   // trigger policy owed
    case ControlGap::Cursor:  return GapStatus::ClosedMechanismPolicyOwed;   // GAP-FREQ could move it
    case ControlGap::AgeGate: return GapStatus::ClosedMechanismPolicyOwed;   // K is the owner's
    }
    return GapStatus::Open;
}

[[nodiscard]] constexpr const char* control_gap_name(ControlGap gap) noexcept {
    switch (gap) {
    case ControlGap::OneShot: return "GAP-ONESHOT";
    case ControlGap::Cursor: return "GAP-CURSOR";
    case ControlGap::AgeGate: return "GAP-AGEGATE";
    }
    return "?";
}

[[nodiscard]] constexpr const char* gap_status_name(GapStatus status) noexcept {
    switch (status) {
    case GapStatus::ClosedMechanismPolicyOwed: return "closed-mechanism/policy-owed";
    case GapStatus::Closed: return "closed";
    case GapStatus::Open: break;
    }
    return "open";
}

[[nodiscard]] inline std::string control_gap_ledger() {
    std::string out;
    for (std::uint32_t i = 0; i < 3; ++i) {
        const ControlGap gap = static_cast<ControlGap>(i);
        if (i != 0) { out += " "; }
        out += control_gap_name(gap);
        out += "=";
        out += gap_status_name(control_gap_status(gap));
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// 9. THE LAW'S OWN PROOFS. compile-time, and each one is the negation of a gap.
// ---------------------------------------------------------------------------------------------
static_assert(kLatticeFloorDepth == 3, "the termination bound below is 3 rungs per cell");
// [F1231] THE POPULATION, PINNED AT THE TWO SHAPES THAT WERE MEASURED. `[142, 190)` is the pass's
// own span on `P3_B_64k_65536_r1.log`'s frontiers (valid 12,288 tok, keep 128 tok, frontier 142);
// the pre-image window over the same frontiers is 50 pages, and the two differ by exactly the two
// newest pages the stage never walked.
static_assert(descent_retire_limit_pages(12288, 128, 64, 2048) == 190,
              "F1231: the retirement bound is (valid - keep) / page_tokens, min mapped");
static_assert(descent_population_pages(12288, 128, 64, 142, 2048) == 48,
              "F1231: the plan's population is the RETIREMENT span, not the resident one");
static_assert(descent_population_pages(12288, 128, 64, 190, 2048) == 0,
              "F1231: a frontier that has caught the bound is an EMPTY population, not a wrap");
// [F1231] THE AGE-MAJOR ORDER, AND THE AGE GATE'S OWN TWO READINGS. `AgeGate{1, 9}.admits(8)` is
// the pre-image eligibility at K=1 (nothing new is spared); `!admits(9)` is the newest block
// spared. Both were already asserted below for the gate itself; these two pin the ORDER's axis.
static_assert(descent_order_index_age_major(0, 0, 16) == 0 &&
              descent_order_index_age_major(0, 15, 16) == 15 &&
              descent_order_index_age_major(1, 0, 16) == 16,
              "F1231: age-major means a block's layers are contiguous and block is the major axis");
static_assert(descent_termination_bound(48, 16) == 48u * 16u * 3u,
              "F1202: the one-shot law's bound is #cells * 3, stated over TRIGGERS (34.3).");
static_assert(descent_order_index(0, 0, 48) == 0 && descent_order_index(0, 47, 48) == 47 &&
                  descent_order_index(1, 0, 48) == 48,
              "F1202: the cursor's order index is (layer, block) lexicographic -- the order "
              "`block_descent_plan`'s tie-break produces with the frequency source ABSENT. If this "
              "changes, every saved `next_offset` means something else.");
static_assert(descent_order_cell_layer(descent_order_index(7, 13, 48), 48) == 7 &&
                  descent_order_cell_block(descent_order_index(7, 13, 48), 48) == 13,
              "F1202: the index round-trips, which is what makes the cursor resumable.");
static_assert(AgeGate{0, 9}.admits(9) && !AgeGate{1, 9}.admits(9) && AgeGate{1, 9}.admits(8),
              "F1202: the age gate at K=1 excludes exactly the newest block and admits the one "
              "behind it. K=0 admits everything, which is the OFF switch (the pre-image).");
static_assert(AgeGate{4, 2}.newest_eligible() == 0,
              "F1202: a gate wider than the frontier has NO eligible block -- and `newest_eligible` "
              "must clamp to 0 rather than wrap, which is why it is written as a comparison.");
static_assert(control_gap_status(ControlGap::OneShot) == GapStatus::ClosedMechanismPolicyOwed &&
                  control_gap_status(ControlGap::Cursor) == GapStatus::ClosedMechanismPolicyOwed &&
                  control_gap_status(ControlGap::AgeGate) == GapStatus::ClosedMechanismPolicyOwed,
              "F1202: the three gaps are closed AS MECHANISMS and each keeps a NAMED POLICY "
              "RESIDUAL. A reader who reads 'closed' must also read which policy is still owed.");
static_assert(static_cast<std::uint8_t>(DescentOrder::AdvancePerTrigger) == 0,
              "F1202: `AdvancePerTrigger` is the ZERO value, so a default-constructed spec gets "
              "the LAW (a lap of the ring is a round) and not the GAP-CURSOR failure. The "
              "measurement that decided it is §5b and in ../tests/.");
// [F1245 kvcarry] THE RE-ANCHOR, PINNED AT THE SHAPE THAT WAS MEASURED. The F1244 block-KV arms
// advanced the window's left edge by one page per trigger while the block count changed every pass
// (`population_origin=0,1,2,...,26` beside `blocks=46,93,140,...`), so the cases below are that
// pair's own arithmetic: "the old local 1 is the new local 0", "the old left edge is GONE", and
// "a position on a retired page is not a position".
constexpr bool descent_reanchor_pins_hold() {
    std::uint32_t out = 9999U;
    // A WINDOW THAT DID NOT MOVE IS THE IDENTITY ON EVERY INDEX IT CONTAINS.
    if (!descent_reanchor_local(0, 0, 4, 3, out) || out != 3U) { return false; }
    if (!descent_reanchor_local(5, 5, 10, 3, out) || out != 3U) { return false; }
    // THE MEASURED CHURN: old origin 0 -> new origin 1. Local 1 is local 0; local 0 is RETIRED.
    if (!descent_reanchor_local(0, 1, 93, 1, out) || out != 0U) { return false; }
    if (descent_reanchor_local(0, 1, 93, 0, out)) { return false; }
    if (!descent_reanchor_local(0, 1, 93, 45, out) || out != 44U) { return false; }
    // PAST THE RIGHT EDGE: the new window covers absolute pages [1, 94), so old local 94 (absolute
    // 94) is NOT covered while old local 93 (absolute 93) still is. The boundary is the ABSOLUTE
    // one, which is exactly why the comparison is written there and not on the local indices.
    if (!descent_reanchor_local(0, 1, 93, 93, out) || out != 92U) { return false; }
    if (descent_reanchor_local(0, 1, 93, 94, out)) { return false; }
    // A WINDOW THAT GREW AT THE RIGHT KEEPS EVERY PAGE IT STILL COVERS.
    if (!descent_reanchor_local(0, 0, 93, 45, out) || out != 45U) { return false; }
    if (descent_reanchor_local(0, 0, 93, 93, out)) { return false; }
    // THE POSITION, IN THE AGE-MAJOR ORDER: `block * layers + layer`.
    if (!descent_reanchor_position(0, 0, 4, 4, 7, out) || out != 7U) { return false; }
    // THE F1244 CASE, AS ARITHMETIC: a one-page churn puts the carried position (16 == block 1,
    // layer 0, at 16 layers) on the NEW window's block 0 -- the cell the next trigger examines.
    if (!descent_reanchor_position(0, 1, 93, 16, 16, out) || out != 0U) { return false; }
    // A POSITION WHOSE PAGE WAS RETIRED IS REFUSED, NEVER CLAMPED.
    if (descent_reanchor_position(0, 2, 93, 16, 16, out)) { return false; }
    // AND A WINDOW THAT SHRANK BELOW A CARRIED BLOCK DROPS IT.
    if (descent_reanchor_local(0, 0, 2, 3, out)) { return false; }
    return true;
}
static_assert(descent_reanchor_pins_hold(),
              "F1245: the re-anchor is an identity on a stationary window, drops a page that left "
              "the window (or one past its right edge), and refuses -- never clamps -- a position "
              "whose page is gone.");

// =============================================================================================
// 6. [F1231] THE LIVE CARRIER. ONE OBJECT, SHARED BY EVERY TU THAT CALLS IT.
// =============================================================================================
// THE LIVE CURSOR, ONE PER PROCESS. It is here rather than in the engine so that the CARRIER and
// the LAW are one object: a caller cannot have the law's types and forget its position. It is an
// `inline` function's function-local static, so every translation unit that calls it shares ONE
// object (a header-scope static would give each TU its own, which would be two cursors).
//
// ⚠ NAMED AS OWED: this is process-wide because this engine runs one sequence per process. A
// multi-sequence process would share one cursor, which would be wrong; the right home is the
// sequence's own state, and it is NOT moved there by this line.
[[nodiscard]] inline DescentCursor& descent_cursor_live() noexcept {
    static DescentCursor cursor{};
    return cursor;
}

} // namespace ninfer::product
