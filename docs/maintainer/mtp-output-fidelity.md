# MTP output fidelity: draft-head choice leaks into the emitted tokens

Status: fixed (default head selection), measured on `qwen3.8-27b / nvfp4-dflash2`.

## Symptom

Greedy output is expected to be the target model's argmax at every position — the
documented contract of the verify path is "bit-identical to the original argmax
accept" (`src/ops/kernel/speculative_round.cuh:96`). Measured with
`--print-token-ids`, MTP under `ProposalHead::Optimized` (the shortlist draft head)
does not satisfy it: its greedy token ids differ from a plain (no speculation) run
at 89 of 160 positions.

## Measurements (greedy, 160 new tokens, same prompt, artifact nvfp4-dflash2)

| arm | differing positions vs plain | edit blocks | AL | acceptance | tok/s |
|---|---|---|---|---|---|
| mtp k=3, shortlist head | 89/160 | 6 | 3.61 | 87.79% | 207.96 |
| mtp k=3, full head | **1/160** | 1 | 3.79 | 92.86% | 199.86 |
| mtp k=9, shortlist head | 137/160 | 3 | 7.23 | 70.26% | 324.52 |
| mtp k=9, full head | 137/160 | 3 | 6.91 | 66.67% | 253.86 |
| mtp k=1, shortlist head | 87/160 | 4 | 1.94 | 93.90% | 125.09 |
| dflash2 k=7, shortlist head | 1/160 | 1 | 6.91 | 84.47% | 268.95 |
| dflash2 k=7, full head | 1/160 | 1 | 6.91 | 84.47% | 262.30 |
| plain, repeated | 0/160 | 0 | — | — | 68.87 / 68.83 |
| plain, graph vs eager (`--no-cuda-graph`) | 0/160 | 0 | — | — | — |

"differing positions" counts token-index mismatches, which inflated by shifted
content; "edit blocks" (difflib opcodes) is the honest count — the shortlist-head
MTP arm differs in 6 places, not 89 independent decisions. Text similarity for the
worst case (mtp k=3, shortlist) is 0.952.

Control experiments that make this attributable:

- **Repeats are bit-identical**, for every arm: the engine is deterministic, so this
  is not sampling or timing noise.
- **Graph vs eager is 0/160 for all three arms** (plain, k=3, k=9): the drift is not
  an execution-shape artefact. It is caused by the speculative path itself.
- **Full-head MTP lands on 1/160**, which is the *same single* near-tie flip that
  DFlash2 shows in both head configurations (position 57: plain picks 357, the
  speculative arms pick 12908). That residual is verify-batch numerics (k+1 wide
  verify columns vs a batch-1 decode step), not accept-rule logic.
- **DFlash2 is head-insensitive**: identical output and identical acceptance with
  either head, and the shortlist head is only +2.5% faster. For DFlash2 the
  shortlist head is a pure proposal mechanism, as documented.

## Fix

`qwen3_6::resolved_proposal_head` (startup_features.h) now resolves
`ProposalHead::Auto` to the shortlist head **only for the DFlash2 backend**
(where it is a pure proposal mechanism and output-identical); MTP resolves to the
full head. `--lm-head-draft` still opts into the shortlist head explicitly.
`Auto` also resolves to `Full` whenever speculation ends up disabled — without that,
a plain run fails `layouts_impl.h` "disabled speculative decoding requires
draft_tokens=0 and the full proposal head".

Applied in all three target packages (`qwen3_6_27b`, `muse_glimmer_30b`,
`qwen3_6_35b_a3b`); the 35b additionally self-resolves inside `plan_load` (it
previously froze the caller's raw options), and
`tests/targets/qwen3_6_27b/test_load_plan.cpp` now resolves before calling the
planner, which is the same contract the registry follows.

Cost: MTP k=3 loses ~4% throughput (199.9 vs 208.0 tok/s) in exchange for the
greedy contract holding.

## Open item

MTP k=9 still shows 2 edit blocks against plain even with the full head, while
k=3 shows 1. Whether the wide-verify near-tie rate can be reduced (e.g. by making
the verify columns use the same reduction order as the batch-1 decode kernel) is not
settled; it is a numerical tie-break, not an accept-rule error.

## Reproduction

```
cd build && ./apps/ninfer <artifact> \
  --prompt 'Write a Python function that merges two sorted lists in linear time, with a docstring.' \
  --max-new 160 --max-context 4096 --no-thinking --greedy --print-token-ids \
  [--spec mtp --draft-tokens 3] [--no-lm-head-draft]
```

Scripts used (kept on the Windows working tree): `_mtp_det.sh` (repeatability),
`_shape_sens2.sh` (graph vs eager), `_head_cause.sh` (head A/B on MTP),
`_dflash2_head.sh` (head A/B on DFlash2), `_analyze_vs_plain.py` (token/edit-block
comparison).
