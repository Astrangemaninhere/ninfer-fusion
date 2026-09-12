# S22 — banker's-rounding fix for the S14 parity hole C found (S18)

C's S18 (`_collab/C_s14_verify.md`) broke the S14 parity claim at 8 points, all
budget ∈ {4.125, 5.625, 6.625} × cold ≥ 8 × layers ≥ 48, root-caused to
`round()` semantics: Python banker's vs the header's half-up. Confirmed here, fixed,
re-verified. C re-runs its own harness independently — nothing here is self-certified.

## Fix (the ONLY src/ file touched this round)

`ninfer-fusion-repo/src/product/kv_bit_budget.h` (no engine TU includes it — checked in
S14; window D's build cannot see the edit):

```cpp
// before (half-up — diverges from Python exactly on binary .5 landings)
const std::int32_t budget_x100 = static_cast<std::int32_t>(budget_bits * kKvBitBudgetScale + 0.5);
// after (ties-to-even, matches Python round(); + #include <cmath>)
const std::int32_t budget_x100 = static_cast<std::int32_t>(
    std::nearbyint(budget_bits * kKvBitBudgetScale));
```
plus a comment naming the failure mode (4.125*100 = 412.5 exactly; half-up 413 vs
banker's 412; capacity shifts by `layers` 0.01-bit units; C's 8 divergences at
cold ≥ 8 × L ≥ 48 where cold's 0-bit tiers make the boundary total reachable).

## Before / after (minimal repro, raw)

Command: `wsl.exe -e bash -c "cd /tmp/kvbit_s22 && diff py_before.txt cpp_before.txt"`

```
BEFORE (diff exit 1):  py : achieved= 4.12 penalty= 7.49 mix=coldx16+e8x8+fp8x13+int8x3+nvfp4x8
                       cpp: achieved= 4.13 penalty= 7.47 mix=coldx16+e8x8+fp8x11+int8x5+nvfp4x8
                       (--layers 48 --cold-pages 16 --bits 4.125; matches C's report verbatim)
AFTER  (diff exit 0):  both sides: achieved= 4.12 penalty= 7.49
                       mix=coldx16+e8x8+fp8x13+int8x3+nvfp4x8
                       spec 0-7:e8,8-23:cold,24-26:int8,27-39:fp8,40-47:nvfp4
```

## Extended sweep (C's harness reused verbatim + phase 5 = C's full divergent set)

Harness: `_collab/A_s22_grid.sh` — PART 1 runs `_collab/C_s14_grid.sh` UNMODIFIED
(phases 1-4: L=4 x cold 0..16 x odd+midpoint budgets; L=8; L=12; L=48/64 probes),
PART 2 adds phase 5: L=48/64 x cold {0,8,16} x {3.375 4.125 5.625 6.625 7.375 8.625}
+ 0.25-multiple anchors, RAM-gated like C's phase 4. Command:

```
wsl.exe -e bash -c "bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_s22_grid.sh"
```

Raw result:

```
EXTENDED_PARITY CASES=26 LINES=327 DIVERGED=0 DIV=''
EXTENDED_GRID_ALL_MATCH          (C's own grid, now clean)
PHASE5_RAN avail=23497MB
PHASE5_PARITY CASES=8 LINES=58 DIVERGED=0
S22_ALL_MATCH
```

Every coordinate of C's 8-row divergent table is inside phase 5's cross
(L=48/64 x cold 8/16 x 4.125/5.625/6.625) — all now MATCH. No NEW divergences
surfaced anywhere in the extended sweep (item 3 answer: none beyond the rounding hole).

Rounding spot check (fixed header vs Python, same run):

```
budget  3.125 -> nearbyint = 312.0    python round spot: [312, 338, 412, 462, 562, 662, 738, 862]
budget  4.125 -> nearbyint = 412.0    (8/8 identical; 562.5->562 both sides proves
budget  5.625 -> nearbyint = 562.0     ties-to-EVEN, not floor)
... (8 values, all equal)
```

## Regression

Original S14 grid re-run against the edited header
(`wsl.exe -e bash -c "bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_kvbit_grid.sh"`):

```
PARITY_CASES=27 OUTPUT_LINES=321 DIVERGENT_CASES=0
ORIG_VS_NEW_COLD0_SPECS_IDENTICAL (30 specs)
```

0.25-multiple budgets never land on .5 (C's observation), so the original grid is
mathematically untouched by the fix — and empirically still green.

## Notes / handoff

- nearbyint honours the default FP rounding mode (FE_TONEAREST = ties-to-even); the
  engine never changes the mode. If some future TU sets a different mode, the
  `round-half-even` guarantee silently changes — noted as a theoretical caveat only.
- C's Check 3 (float-tie stress, 1640 configs) already confirmed the float-domain
  tie-break replication; the rounding hole was the only parity break found, now closed.
- Verification by C's own harness re-run is expected (board S22 row) — this report does
  not mark itself verified.
