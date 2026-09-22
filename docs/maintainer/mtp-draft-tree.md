# MTP draft-tree verify shape (`--draft-tree L,d`)

This is the maintainer reference for the MTP draft-tree path. It is a **contract** document: it
names the seams, the invariants each seam refuses, and the code that owns them. File-and-symbol
anchors are given instead of line numbers, because line numbers in this tree drift within a single
session; where a number is given it is marked with the revision it was taken at
(`@3944a53` = `git rev-parse --short HEAD`).

## 1. What the option means, and what "tree" changes

`--draft-tree L,d` names an **MTP tree verify shape**: `L` rank-paths per depth, verify depth `d`,
node budget `L*d`. It **replaces** the scalar draft width — `L*d` *is* the round's extent, so the
two spellings contradict each other and the CLI refuses them together:

```
"--draft-tree L,d needs --spec mtp and no --draft-tokens (the tree's node budget "
"L*d is the draft width)"
```

(`apps/cli/options.cpp`, in the cross-option validation pass.) `L,d = 1,d` must reproduce
`--draft-tokens d`.

A **chain** round accepts a column prefix: accepted node at depth `j` is verify column `j`. A
**tree** round accepts an arbitrary set of columns, so two things that were implicit stop being
true:

1. **The mask.** Visibility can no longer be "everything below the round", because a tree's sibling
   is not an ancestor.
2. **The depth order.** `alignment_ids[j] = verify_ids[j+1]` pairs index `j` with `hidden[:,j]`;
   that is depth-correct only while the accepted nodes are the column prefix.

Everything below exists to close one of those two.

## 2. The seams, in data-flow order

```
proposal side                      producing half                 receiving half
(lattice: ids [+ scores])  -->  mtp_tree_produce.h       -->  mtp_tree_publish.h
                                mtp_tree_columns_*           publish_mtp_tree()
                                                                    |
                                                                    v
                                                        Program (per-sequence slot)
                                                                    |
                    verify round  <---------------------------------+
                    (column_masks, target_column_depths)
                          |
                          v
             speculative accept (greedy + column_masks)
                          |
             accepted_columns, chain_sources
                          |
                          v
             ops::mtp_tree_commit_history  ->  K/V at frontier+1..frontier+A
                          |                     + kMtpTreeFlag* word
                          v
             ops::mtp_draft_align_hidden   ->  depth-ordered hidden for the MTP head
```

### 2.1 Receiving half — `src/targets/qwen3_6/impl/runtime/mtp_tree_publish.h`

The **single place a tree layout can enter the runtime**. Host only, engine free, standard library
only. Public surface (`@3944a53`):

| Symbol | Line | Contract |
|---|---|---|
| `struct MtpTreeColumns` | 47 | `depths` (= verify columns), `masks` (ancestor bitmask per column), `tokens` (`depths.size() - 1` entries: one per non-anchor node) |
| `mtp_tree_columns_from_ddtree(tree, node_tokens)` | 69 | builds `MtpTreeColumns` from an `ops::ddtree::Tree` via `ops::ddtree::column_depth()` / `column_ancestors()`. Throws `logic_error` on empty ddtree columns or a negative depth |
| `mtp_tree_layout_defect(columns, width)` | 96 | returns the defect as **text** (empty = valid). 12 named defects |
| `validate_mtp_tree_columns(columns, width)` | 170 | throws `logic_error("MTP draft-tree layout rejected: " + defect)` |
| `struct MtpTreePublishSlot` | 183 | the destination: arrays + widths to fill |
| `publish_mtp_tree(columns, width, slot)` | 196 | writes the slot; throws if the slot is incomplete, narrower than the layout, or has no room for the node drafts |
| `clear_mtp_tree(slot)` | 222 | zeroes the slot so "never published" is distinguishable from "published as chain" |
| `mtp_tree_required_extent(columns)` | 240 | the verify frame extent the layout needs |

`mtp_tree_layout_defect()` is deliberately **separable** from the throwing wrapper, so a host test
can assert the specific defect rather than only that something threw. The defect list checks, in
order: `depths`/`masks` size agreement, non-empty, `tokens.size() + 1 == depths.size()`, `width != 0`
and `n <= width`, `n <= 63` (the 64-bit mask), the anchor column (`depths[0] == 0 && masks[0] == 1`),
and then per non-anchor column `j`: non-zero depth, attends the anchor (`mask & 1`), attends itself
(`mask & self`), attends nothing outside this round (`mask & ~round_bits`), has exactly one ancestor
(bit `i` of `rest` counting down from `j-1`), whose mask is exactly `rest`, and whose depth is
exactly one shallower. A chain must then be the prefix: `depths[j] == j && mask == (1 << (j+1)) - 1`.

The `n <= 63` and "column 0 is the anchor" rules exist because the mask is a `std::uint64_t` and
column 0 is the round's anchor by construction; the chain-spelling rule exists because a chain
published through this seam must be *indistinguishable* from one that was never published as a tree.

### 2.2 Producing half — `src/targets/qwen3_6/impl/runtime/mtp_tree_produce.h`

Also host only / engine free / standard library only, so a CPU test pins the whole tree arithmetic
with no device, no artifact and no variant macros. There are **two lattices, and there are two
because they cost different amounts**:

**(1) Full lattice** — `MtpProposalLattice`, `mtp_lattice_defect()`,
`mtp_tree_columns_from_lattice()`. The proposal side publishes, per draft depth `s`, the top-k
candidate ids **and** the top-k x top-k predecessor x successor score grid of
`ops::ddtree::build_tree`. This is the shape a beam-`L` draft head has to produce with `L` real
forwards per depth, because row `p` is the distribution conditioned on candidate **rank** `p` at
depth `s-1`, and a rank's row cannot be inferred from another rank's line. The entry point runs the
**real reference builder**, so the layout is bit-for-bit what `ddtree::column_depth()` /
`column_ancestors()` would report for a real beam-`L` tree.

**(2) Sibling lattice** — `MtpSiblingLattice`, `mtp_sibling_lattice_defect()`,
`mtp_tree_columns_from_siblings()`. The proposal side publishes only the top-paths candidate ids per
depth — exactly what the existing per-depth proposal forward already produces, with a top-paths
extraction instead of an argmax, and **no extra forward**. The layout is: level `s` holds
`top_paths` sibling nodes, all at depth `s + 1`, all children of the level `s-1` spine node; sibling
`i == 0` is the spine and continues to level `s + 1`.

Both validators check the node budget against `ops::ddtree::kMaxNodes` and both throw
`invalid_argument("MTP ... lattice rejected: " + defect)`, again keeping the text separable.

**The one call the host hand-off makes** is `publish_mtp_tree_round(lattice, node_budget, width,
slot)`, overloaded on the lattice type (`MtpSiblingLattice` and `MtpProposalLattice`). Each overload
builds the layout, validates it through the receiving seam, writes the three layout fields and the
column-ordered node drafts, and returns the extent the round must carry (`mtp_draft_count` == the
node count). The sibling overload additionally asserts `columns.columns() == width` first: a sibling
lattice that does not fill the round's verify frame is a `logic_error`, not a partial publish.

`clear_mtp_tree_round(slot)` is the second half of the same discipline. A chain round **must not**
inherit the previous round's tree, because the verify side reads `mtp_tree_columns` unconditionally
and a stale non-zero value would make an `L <= 1` round take the tree branch. Calling this instead of
writing the field by hand keeps the invariant "a tree is published, or the field is zero" in one
place.

`kMtpLatticeTopK` is pinned to `ops::ddtree::kTopK`, and a test asserts
`ops::ddtree::kTopK == 16` — the ddtree contract header moving is a build failure, not a silent
semantic change.

### 2.3 The accept op — `include/ninfer/ops/speculative_round.h`

The greedy accept gained a `column_masks` argument (I64 `[K+1,B]` or empty). **Non-empty selects the
greedy TREE accept** and publishes the depth/column it accepted into `accepted_columns`. The
stochastic routes keep the public Op's separate definitions and **do not** take the masks: the
runtime refuses a tree round on those routes before launching, because the mask-driven accept reads
the verifier's own argmax, which only the greedy route produces. The stub
(`src/ops/launcher/speculative_round.cu`) states this in place: "The stochastic route is unreachable
with a non-empty mask (the runtime rejects a tree round on a non-greedy request), so the masks and
the accepted column are not forwarded here."

### 2.4 The commit op — `ops::mtp_tree_commit_history`

Contract in `include/ninfer/ops/mtp_round.h`; kernel `src/ops/kernel/mtp_round.cuh`;
wrapper `src/ops/wrapper/mtp_round.cpp`. It commits the accepted chain — which sits at **arbitrary
verify columns** — into the position prefix `frontier+1 .. frontier+A` that the next round reads as
history. Every layer owns its own K/V, so this is not a metadata-only operation.

Inputs: `column_masks` (I64 `[W,B]`), `column_depths` (I32 `[W,B]`), `accepted_columns` (I32 `[B]`),
`base_frontiers` (I32 `[B]`), `table_rows` (I32 `[B]`), `cache` (a `PagedKVBatchLayerView`). Inputs
are unchanged; the Op decides no frontier and keeps no state.

**The flag word.** Per-row diagnostics are written into `commit_flags`. Every non-zero bit says the
**tree metadata the draft side published for that row is inconsistent**, so the round's accepted
chain cannot be trusted:

| Bit | Constant | Meaning |
|---|---|---|
| `1 << 0` | `kMtpTreeFlagColumnOutOfRange` | `accepted_columns[b]` is not a column of this frame |
| `1 << 1` | `kMtpTreeFlagMaskMissingSelf` | `column_masks[c]` does not have its own bit `c` set |
| `1 << 2` | `kMtpTreeFlagDepthMismatch` | `popcount(column_masks[c]) - 1 != column_depths[c]` |
| `1 << 3` | `kMtpTreeFlagChainOrder` | the accepted chain's deepest column is not `c` |
| `1 << 4` | `kMtpTreeFlagPositionOutOfRange` | `frontier + A` is outside the mapped logical cache |

The runtime **turns a non-zero word into a hard error** (`program_impl.h`). That refusal used to be
a silent wrong history; it is now an engine stop. `column_depths` is the cross-check the commit runs
against the mask's own popcount: if the published depth and the mask's chain length disagree, the
metadata is inconsistent.

### 2.5 The align op — `ops::mtp_draft_align_hidden`

Contract in `include/ninfer/ops/mtp_round.h`; kernel in `src/ops/kernel/mtp_round.cuh`; wrapper in
`src/ops/wrapper/mtp_round.cpp`. **Tree rounds only** (`--draft-tree L,d` with `L > 1`).

Math, with `W` = width, `B` = batch, `chain_sources` I32 `[W,B]` holding in row `i` the verify
**column** of the accepted node at depth `i+1` (every slot beyond `A = popcount(mask) - 1` is `-1`),
and `valid_counts` I32 `[B]` = the round's `licensed_counts = A + 1`:

```
out[:,j,b] = hidden[:,j,b]                        for j == 0 or j >= valid_counts[b]
             hidden[:,chain_sources[j-1,b],b]     for 1 <= j < valid_counts[b]
```

Depth 0 keeps the anchor column (`chain_sources` has no depth-0 row). For a chain,
`chain_sources[i,b] == i+1`, so the Op is a **bit-for-bit copy** of `hidden`; the runtime does not
call it on a chain round at all. `W` must satisfy `2 <= W <= kMtpTreeMaximumWidth`, and the wrapper
**refuses `out` aliasing `hidden`**.

**Degradation, stated on purpose.** A `chain_sources` value outside `[0,W)` is **clamped to column
0** rather than used as an offset, so a malformed table can produce a wrong gather but never an
out-of-range read. The table's own validation is `mtp_tree_commit_history`'s flag word.

### 2.6 The mask re-anchor — `src/ops/kernel/gqa_attention_decode_bf16.cuh`

The visibility test must use the **round** base, not the launch's chunk base:

```cpp
const int rel0 = column_begin + key0 - first_pos;
```

`rel` is the key's column index within the round's whole row, so `rel < 0` is a key below the round —
the shared, already committed history — and is always visible, while every key of this round is
visible only where the mask says so.

With the **chunk**-relative spelling, a width above `kSmallTChunkTokens` is dispatched as several
launches with `column_begin > 0`, and the keys below a chunk are **not** all history: the earlier
chunks' columns belong to this same round block, and a tree's sibling is not an ancestor. Every
earlier chunk's column was therefore admitted unconditionally, silently over-visible-ing a tree
round's branches. For a chain the mask is the prefix `(1 << (j+1)) - 1`, so every column the old
spelling admitted is admitted by the mask too — bit-for-bit unchanged — and for `column_begin == 0`
the two spellings are the same expression.

### 2.7 The ladder snap — `mtp_ladder_round()`

`program_impl.h`. The **judgement** is made on the full decision ladder (`kMtpWindowLadder`, the
calibration domain of the cost model), but the **capture domain is the target's**:
`mtp_graph_ladder_profiles(capacity, draft_window, mtp_ladder)` captures exactly `mtp_ladder`, and
`layouts_impl.h` builds that as "every `kMtpWindowLadder` member `<= kMaximumMtpDraftTokens`".

For a target with `kMaximumMtpDraftTokens == 5` that domain is `{2,3,5}`, so a cut taken from the
full ladder (>= 6, by the ties-up rule) names a rung of 7/9/15 that **no captured graph can cover**:
`extend_mtp_graphs` cannot build a rung that is not a ladder member and returns immediately, and
`select_graph_profile` then throws "MTP batch CUDA Graph coverage is incomplete" — an engine stop
for a rung this same code chose on purpose. Snapping on the **captured** ladder keeps the criterion
and moves only the rung. Ties go **up**, the same rule as `mtp_window_ladder_index`, because at a
tie the marginal column value equals its cost and the measured failure mode on this axis is
under-drafting. If the two ladders coincide this is a no-op.

`head_floor` can still leave the captured ladder: it is `kMtpShortlistMinimumDrafts` (5) under the
Optimized proposal head, and a target whose effective ladder is `{2,3}` would take `chosen = 5`,
capture nothing for it, and stop the engine with a message naming CUDA Graph coverage instead of the
ladder. The write point therefore now refuses explicitly:

```cpp
throw std::logic_error("MTP target width is not a rung of the captured ladder");
```

No target has that ladder today, so this is an **unguarded shape rather than a reachable bug**. The
statistics/trace rung now names `mtp_ladder.back()` rather than `kMtpWindowLadderTop`, which would
report a width this target cannot run.

### 2.8 State plumbing

| Field | Owner | Role |
|---|---|---|
| `MtpDecodeIngress::target_column_depths` | `src/targets/qwen3_6/impl/runtime/schedule.h` | per-column depth of the node each verify column carries, I32 `[W,B]`; the cross-check the commit runs against the mask's popcount |
| `MtpDecodeEgress::accepted_columns` | `schedule.h`, `.../round_state.h`, `impl/state/round_state.cpp` | I32 `[B]`; the verify column the round accepted at, 0 for a chain round and for a tree round that accepted nothing |
| `MtpDecodeEgress::chain_sources` | same | I32 `[W,B]`; the accepted chain -> history map. Chain: identity (`chain_sources[i,b] == i+1`, moves no bytes) |
| `MtpDecodeEgress::tree_commit_flags` | same | I32 `[B]`; the `kMtpTreeFlag*` word |

`speculative_target_impl.h` takes the continuation state at the **accepted column**:
`frame.column_masks.data != nullptr ? frame.accepted_columns : frame.accepted_drafts`. The greedy
accept writes `accepted_columns` in **both** modes and it equals `accepted` for a chain, so the tree
gate is the only thing that changes which tensor is read — and a `--draft-tokens` round reads the
very same thing it always did.

### 2.9 The proposal-side fill — `mtp_tree_proposal_fill.h`

The producing half builds a layout **from a lattice**, and the receiving half stores one — but both are
inert until something **pushes** the lattice out of the proposal loop, and nothing did. The verify side
refuses an `L > 1` round whose tree was never published (`program_impl.h`), and the producing half's own
header records the symptom: `sequence.mtp_tree_columns` still reads `0` in a real run because "the
proposal side must publish per-column depths and ancestor masks". `mtp_tree_proposal_fill.h` is that
push. It is kept host-only and engine-free so the whole hand-off stays pinnable by a host test.

**Where the lattice comes from.** ONE device source, read on the host as two tables of the **same** round
and the **same** row:

- `ids[s * paths + i]` — `ops::mtp_proposal_topk` over the depth-`s` proposal row
  (`include/ninfer/ops/mtp_proposal_topk.h`), one extraction per depth;
- `chain[s]` — the ordinary chain draft for depth `s`, which is that same row's argmax (the
  `ops::argmax` inside `TextContext::proposal_argmax`).

They are computed by two different kernels **from the same logits row** — the same tensor, the same row
window and the same forward (`mtp_impl.h` passes `TextConfig::token_domain` as the extraction's row count
precisely because that is the window the argmax path reads). So `ids[s * paths] == chain[s]` is a
**same-source consistency** check and **not** two independent sources; the "two independent device
outputs" reading is **retired** in the header itself. What it does prove, which is not nothing: the
extraction **ran for this round** and landed in **this round's frame** — a depth whose ids block was
never written carries the previous round's numbers and would have to coincide with this round's argmax at
every depth to pass. What it does **not** prove: that the numbers *are* the extraction's output. No
host-side check can prove that, and the header names the undecidable classes instead of pretending a
rule covers them.

**The surface.** `struct MtpProposalRow` (`paths`, `steps`, `ids`, `chain`, `token_domain`),
`mtp_proposal_row_defect(row, node_budget)` returning the defect as text,
`mtp_sibling_lattice_from_row()` as the throwing wrapper, `mtp_tree_publish_slot(sequence)` (templated on
the sequence type so the field-for-field mapping lives in ONE place), and the one call a round's hand-off
makes:

```
fill_mtp_tree_round(row, node_budget, width, slot) -> the live extent the round must carry
```

It checks the lattice, builds the layout through the producing half, publishes it through the receiving
half, and re-states both invariants rather than assuming them (it refuses an extent that is not the node
count). Two readers decode the device layout for it: `mtp_proposal_row_from_egress()` (a decode round,
`lanes` = the round's batch) and `mtp_proposal_row_from_prefill()` (the bridge, one token per depth).
Both take `token_domain` as a **parameter of the read**, because the read is where the head is known.

**The egress layout.** `MtpDecodeEgress::next_proposal_ids` is laid out by the device side
(`mtp_impl.h`) as `s * depth_stride + i * lanes + t`, because `ops::mtp_proposal_topk` stores its
per-token rows with stride `tokens` and `tokens` is the round's batch. `depth_stride` is
`qwen3_6::kMtpTreeProposalDepthStride`, and `mtp_tree_proposal_index(depth, rank, lanes, lane,
depth_stride)` is the single spelling of that index — used by the readers and by the test. The three
constants live next to the array in `.../round_state.h`:

| Constant | Value | What it is |
|---|---|---|
| `kMtpTreeMaximumPaths` | = `kMtpDecodeMaximumDrafts` (15) | the per-depth path ceiling |
| `kMtpTreeProposalDepthStride` | `kMtpTreeMaximumPaths * kMaximumConcurrency` | the per-depth block stride |
| `kMtpTreeProposalEntries` | `kMtpTreeProposalDepthStride * kMtpDecodeMaximumDrafts` | the array's size |

**The planning-time landmine this header disarms.** `mtp_proposal_topk` returns **row indices** of the
table it is handed, which is a global token id **only for the full proposal head**. The optimized
(shortlist) head never even writes the round's proposal-logits frame — `TextContext::proposal_argmax`
allocates its own scratch and remaps the argmax through `proposal_head_ids_` — so publishing its indices
as tree nodes would be a silent wrong answer twice over (stale/unwritten rows **and** shortlist positions
published as token ids). The planner therefore **refuses a tree round on a shortlist head**, with the
reason spelled out in the refusal text rather than left to the reader.

**The gate's trust boundary.** The gate is given six things — `paths`, `steps`, `token_domain`, `ids[]`,
`chain[]` and the node budget — and every check it makes is decidable from them. Three classes of wrong
lattice are **not**, and the header names them where the checks are, not only in a report: (a) in-domain
provenance at rank >= 1 (an in-domain id no extraction produced is indistinguishable from a real one at
this layer; the engine's own baseline trust model is weaker still, since the chain path reads
`SequenceState::mtp_drafts` with no check of any kind); (b) a repeated token along an ancestor path —
**not** a defect and must not be refused, because the draft loop feeds depth `s-1`'s prediction back as
depth `s`'s input token, so in a repetition loop the spine repeats at every level, which is exactly where
speculative decoding pays off most; and (c) an id that repeats another depth's id. Two of the four
defects an independent audit of the earlier halves found *are* decidable here and are refused: a node
token outside `[0, token_domain)` (a provable property of the real path, hence a refusal with no false
positives) and two siblings on one level carrying the same id (the top-L merge consumes each row index it
selects).

**The compile-time pins.** `kMtpLatticeTopK` is pinned to `ops::ddtree::kTopK`, and the test asserts
`ops::ddtree::kTopK == 16` **and** `kDepthStride == 120` (`= kMtpTreeMaximumPaths * 8`), so the ddtree
contract header or the egress stride moving is a build failure rather than a silent semantic change. The
same family of pins as §2.2's.

## 3. Tests

| ctest name | File | What it pins | Needs |
|---|---|---|---|
| `ninfer_qwen3_6_mtp_tree_publish_test` | `tests/targets/qwen3_6/test_mtp_tree_publish.cpp` | layout validation and the sequence seam; uses the real ddtree reference builder | host only, no engine/CUDA/variant header |
| `ninfer_qwen3_6_mtp_tree_produce_test` | `tests/targets/qwen3_6/test_mtp_tree_produce.cpp` | the producing half; builds the verify-column layout from the candidate lattice, and for the sibling shape against `ddtree::column_depth` / `column_ancestors`, then publishes through the receiving seam | host only, no engine, no CUDA, no variant macro |
| `ninfer_qwen3_6_mtp_tree_proposal_test` | `tests/targets/qwen3_6/test_mtp_tree_proposal.cpp` | the proposal-side fill (§2.9): the lattice reader, the gate's refusals and their negative controls, the egress index against a hand-built image, and the two compile-time pins | host only, no engine, no CUDA, no variant macro |

All three are registered in `tests/CMakeLists.txt` with
`target_include_directories(... PRIVATE ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)`.

The publish test needs one thing the produce test does not: a **mirror** of the per-sequence tree
slot (`SequenceTreeMirror`), because the real `program.h` struct is variant-instantiated behind
`NINFER_QWEN36_VARIANT` and needs the engine to link. The mirror reproduces exactly the property
under test — that the field is `0` until a publisher writes it — and the test says so in place.
Read that test's mirror struct as a mirror, not as the real `SequenceTreeSlot`.

## 4. What is landed and what is not

- `mtp_proposal_topk` (`.h`, `.cuh`, `.cu`) is **additive and not yet called**: the proposal loop that
  must call it is the remaining half. Its own header says so, and cites the precedent
  (`include/ninfer/ops/dflash2_tree_walk.h`: "ADDITIVE and DEFAULT OFF ... Nothing calls it yet").
  The op writes every output entry including the tail, so a reader that wants only the first `paths`
  entries never reads an uninitialised value.
- For the **SHORTLIST** proposal head, `rows` is the shortlist width and the returned index must be
  remapped through the shortlist table. The planner **refuses a tree round on a shortlist head**
  (`layouts_impl.h`) rather than let a shortlist index be published as if it were a token id.
- The device half of the commit op and of `mtp_proposal_topk` has **not been compiled**. Four pieces
  in the landing batch put code on a device path that no one has compiled, and
  `nvcc -fsyntax-only` is not available in this environment. Treat "landed" as "the tree contains
  it and it matches a pinned post-image", not as "it runs".
- The **proposal-side fill** (§2.9) is landed and **is** called: `program_impl.h` reaches it at the
  decode hand-off and at the prefill bridge, and the egress read there is the engine's only *host*
  consumer of `MtpDecodeEgress::next_proposal_ids` (a bounded probe of seven named files —
  `program_impl.h` once, at the `mtp_proposal_row_from_egress()` call; `mtp_impl.h` once, to hand the
  device buffer to the extraction). What that does **not** mean is that a tree round has been run: the
  commit op's device half and `mtp_proposal_topk`'s are still uncompiled, and the host test pins the
  hand-off's arithmetic and its refusals, not an engine round.
- The proposal test **is** registered (`ninfer_qwen3_6_mtp_tree_proposal_test`), so unlike the
  multi-device host tests it is not in the "landed and unreachable by ctest" class — see
  [`op-development.md`](op-development.md) §6.4.

## 5. Adding a new `--draft-tree` shape

1. Add the lattice entry point in `mtp_tree_produce.h` with a `<name>_defect()` predicate that
   returns text, plus a throwing wrapper. Do not fold validation into the builder.
2. Return `MtpTreeColumns` and publish it through `publish_mtp_tree()` — never write the sequence
   slot directly. The receiving half is the only seam.
3. If the shape needs a new metadata cross-check, add a **new bit** to `kMtpTreeFlag*` in
   `include/ninfer/ops/mtp_round.h` and consume it in `program_impl.h` as a hard error. Reusing a
   bit makes two different inconsistencies indistinguishable in a log.
4. Add the shape to the ladder-snap consideration in `mtp_ladder_round()` if its node budget can
   exceed the target's captured ladder.
5. If the shape needs the proposal side to push a lattice, add the reader and the row checks in
   `mtp_tree_proposal_fill.h` next to the existing ones (§2.9), and route the hand-off through
   `fill_mtp_tree_round()`. Do not write the slot from the call site, and do not skip
   `mtp_proposal_row_defect()` — the classes it cannot decide are named there so a new rule is not
   invented for them.
6. Extend the host test. If the shape's columns are not the depth order, extend
   `mtp_draft_align_hidden`'s contract instead of post-processing the hidden in the runtime.
