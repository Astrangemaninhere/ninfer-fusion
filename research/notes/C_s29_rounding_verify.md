# C_s29_rounding_verify.md — S29: independent re-verification of A's S22 rounding fix

Verdict: **A's fix is VERIFIED.** Every check passed on my own terms; I could not
break it again in the directions the fix suggests. Constraints honored: plain
`g++ -O0` in `/tmp`, no CUDA tree, no GPU, no `src/` writes. Fix confirmed in
tree first: `src/product/kv_bit_budget.h:147` now
`std::nearbyint(budget_bits * kKvBitBudgetScale)` (+`<cmath>`:40, failure-mode
comment :142-144).

## Check 1 — my harness unchanged: PASS
Command: `wsl.exe -- bash -lc "bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/C_s14_grid.sh"`
Raw tail:
```
PHASE4_RAN avail=21245MB
EXTENDED_PARITY CASES=26 LINES=327 DIVERGED=0 DIV=''
EXTENDED_GRID_ALL_MATCH
```
Counts match A's claim exactly (26 cases / 327 lines / 0 divergences), phases
1-4 all ran (incl. L=48/64).

## Check 2 — sweep past A's (and mine): PASS, 0 divergences
`C_s29_sweep.sh`: layers NOT in {16,24,30,48,64} = {5,9,20,36,40,56};
cold 0..16 (full at small L, {0,4,16} at large L); budgets = exact k/8
midpoints {3.375,4.125,4.625,5.125,5.625,6.125,6.625,7.375,8.625} +
.005-neighbours that land on NON-half floats {4.115,4.135,5.615,5.635,6.615,
6.635} + 3-decimal probes {4.001,3.999,4.999,5.123,6.378} + previously-matching
anchors {3.37,4.13,4.5,5.62,6}. Raw tail:
```
AFTER_SMALL CASES=51 DIV=0
LARGE_RAN avail=21170MB
S29_SWEEP CASES=59 DIVERGED=0 DIVCASE=''
S29_SWEEP_ALL_MATCH
```
One hole closed explicitly: `L=56 C=16` hit the sweep's 300s python timeout
(PYFAIL = skipped, not counted); rerun with 850s over the dangerous midpoints:
```
PY=0 CPP=0 diff empty -> L56C16_MATCH
```
Total: 60 sweep cases, 0 divergences. First divergence: NONE.

## Check 3 — rounding cannot regress the other direction: PASS
`C_s29_rnd_dense.cpp` + `C_s29_dense_check.py`: 12,050 points (x = 3.0..9.0
step 0.0005 + the exact k/8 family), doubles cross language via %.17g
round-trip (bit-identical), comparing `std::nearbyint(x*100)` vs Python
`round(x*100)` (both half-even):
```
DENSE_CHECK points=12050 mismatches=0 first=None
```
No case where nearbyint differs from Python round exists in the reachable
budget range — so the fix cannot introduce a mirror-image hole.
Previously-matching cases unchanged: 3.37/4.13/4.5/5.62/6 anchors and the
whole phase-1 L=4 midpoint sweep (all previously matching) still match in
checks 1-2.

## Check 4 — A used my script verbatim: TRUE
`A_s22_grid.sh:13` = `bash "$COL/C_s14_grid.sh"` (unmodified invocation; I
re-ran the same file myself). Their phase 5 covers my full 8-row divergent
table (L=48/64 × cold {0,8,16} ⊇ all 8 coordinates) plus 0.25 anchors;
their only additions are phase 5 itself and an 8-value rounding spot check.
No re-run of a "modified original" was needed.

## Residual (not a divergence)
The Python tool remains the slow side at large L × cold × big budgets
(L=56/C=16 took >300 s); that is performance, not correctness.
