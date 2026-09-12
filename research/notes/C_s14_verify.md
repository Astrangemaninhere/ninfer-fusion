# C_s14_verify.md — S18 adversarial verification of A's S14 parity claim (C, 2026-09-10)

Verdict up front: **A's grid claim is true ON A'S GRID but the parity region has a hard
boundary — I found 8 divergent (layers, cold, budget) cases.** Root cause is NOT the DP;
it is budget rounding: Python `round()` is banker's (half-to-even), the C++ header does
half-up `(int)(x*100+0.5)`. They differ exactly when `budget*100` lands on a binary-exact
`.5`, i.e. budgets `k/8` with odd k (4.125, 5.625, 6.625, ...) — all NOT multiples of 0.25.
A's grid only uses 0.25-multiples ({3 3.5 4 4.5 5 6 8 10 12 16}), where the two roundings
can never disagree — so the claim survived A's own grid and falls on the extension.

Constraints honored: plain `g++` in `/tmp/kvbit_c18`, GPU + CUDA build tree untouched,
`src/**` NOT edited (C boundary; fix belongs to A). `A_kvbit_optimac_test.cpp` does not
exist in `_collab` (checked via dir; task said "if present").

## Check 1 — interface source-compatibility (item 4): PASS

Pre-cold signature was `spec(layers, budget, e8_limit=8)` (`A_kvbit_budget_h_orig.h.bak:72`).
New: `spec(layers, budget, e8_limit=8, cold_cap=0)`.

Command:
`wsl.exe -- bash -lc "cd /tmp/kvbit_c18 && g++ -std=c++20 -O0 -I<repo>/include -I<repo>/src/product C_s14_tu.cpp -o tu_iface && ./tu_iface"`
Raw output:
```
spec(8,4.5)        = 0-7:e8
spec(8,4.5,8)      = 0-7:e8
spec(8,4.5,8,0)    = 0-7:e8
solve(8,4.5).spec  = 0-7:e8
PASS source-compat: 2/3-arg calls compile, == explicit-0
spec(8,3.5)        = 0-2:e8,3-7:iso3
spec(8,3.5,8,2)    = 0-5:e8,6-7:cold
PASS cold arg live: 4-arg call changes the plan
TU_RESULT PASS
```
(Note: first TU draft wrongly expected cold to change budget 4.5 — cold penalty 0.25/layer
only beats e8 (0.08/layer) when the all-e8 plan does not fit; 3.5 is the discriminating
budget. That FAIL was my test bug, fixed; header behavior was correct both times.)

## Check 2 — extended grid (item 2): **FAIL — 8 divergences, headline below**

Harness: `_collab/C_s14_grid.sh` (layers 4/8/12 always; 48/64 RAM-gated — ran at 5.7GB
avail after the CUDA build's peak passed). Coverage: L=4 x cold 0..16 x {3.37,4.13,5.62,
12 binary-midpoint budgets}; L=8 x cold {0,2,16}; L=12 x cold {0,16}; L=48/64 probes.
Command: `wsl.exe -- bash -lc "bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/C_s14_grid.sh"`

Raw grid tail:
```
EXTENDED_PARITY CASES=26 LINES=327 DIVERGED=2 DIV='L=64 C=16 4.125'
```
Minimal repro (first divergence, raw output):
```
$ python3 tools/archkit/kv_bit_budget.py --layers 48 --cold-pages 16 --bits 4.125
bits= 4.12 -> achieved= 4.12 penalty= 7.49 mix=coldx16+e8x8+fp8x13+int8x3+nvfp4x8 \
  --kv-layer-storage 0-7:e8,8-23:cold,24-26:int8,27-39:fp8,40-47:nvfp4
$ ./kvbit_new 48 16 4.125            # src/product/kv_bit_budget.h
bits= 4.12 -> achieved= 4.13 penalty= 7.47 mix=coldx16+e8x8+fp8x11+int8x5+nvfp4x8 \
  --kv-layer-storage 0-7:e8,8-23:cold,24-28:int8,29-39:fp8,40-47:nvfp4
```
C++ is the *better* allocation (penalty 7.47 < 7.49) — Python's smaller capacity forbids
it. Full divergent set (all same mechanism):
| L | C | budget | py achieved/penalty | cpp achieved/penalty |
|---|---|--------|--------------------:|---------------------:|
| 48 | 16 | 4.125 | 4.12 / 7.49 | 4.13 / 7.47 |
| 64 | 16 | 4.125 | 4.12 / 12.78 | 4.13 / 12.75 |
| 48 | 16 | 5.625 | 5.62 / 4.06 | 5.63 / 4.04 |
| 64 | 16 | 5.625 | 5.62 / 5.57 | 5.63 / 5.54 |
| 64 |  8 | 5.625 | 5.62 / 8.59 | 5.63 / 8.57 |
| 48 | 16 | 6.625 | 6.62 / 2.75 | 6.63 / 2.73 |
| 64 | 16 | 6.625 | 6.62 / 3.81 | 6.63 / 3.78 |
| 64 |  8 | 6.625 | 6.62 / 3.82 | 6.63 / 3.79 |
Clean (MATCH) everywhere else tested: all of L=4 (cold 0..16, all midpoints), L=8, L=12,
3.375 / 7.375 / 8.625 at L=48/64 cold 8/16, and the suggested non-0.25 budgets
**3.37 / 4.13 / 5.62 everywhere**. Pattern: divergence needs binary-midpoint budget AND
cold >= 8 AND L >= 48 (cold's 0-bit tiers make the exact capacity-boundary total reachable).

Root cause micro-proof (`_collab/C_s14_rnd.cpp`, raw):
```
cpp: 4.125*100 = 412.5 (exact .5 landing: 1)
cpp: (int)(4.125*100+0.5) = 413   <-- half-up (current header)
fix: (int)std::nearbyint(4.125*100) = 412  <-- ties-to-even
fix: nearbyint spot checks: 413.5->414 562.5->562 412.99999999999994->413
py : round(412.5) = 412           <-- banker's (even)
py spot: round(413.5)= 414 round(562.5)= 562 round(4.13*100)= 413
capacity delta = layers units (e.g. L=48: py 19776 vs cpp 19824)
```
Suggested fix for A (REQUEST, src/** is A's): `kv_bit_budget.h:137`
`(int)(budget_bits * kKvBitBudgetScale + 0.5)` -> `(int)std::nearbyint(budget_bits *
kKvBitBudgetScale)` (+ `#include <cmath>`); then re-run `C_s14_grid.sh` and expect
`EXTENDED_GRID_ALL_MATCH`. nearbyint matches Python round on all spot values above.

## Check 3 — float-tie stress (item 3): ties exist, reachable, C++ matches (PASS)

Probe: `_collab/C_s14_tie_probe.py` (tracks float-domain AND exact x100-integer penalty
per final state; scans L 3..12, cold 0..3, budgets 3.00..8.25 step 0.13 = 1640 configs).
Raw output:
```
micro: (0.25+0.25)+0.30 = 0.8
micro: 0.25+(0.25+0.30) = 0.8   same-double=True
micro: 10 x +0.08 accum = 0.7999999999999999   exact-0.8-tie-with-cold+nvfp4=True
micro: grouping-dependent float sums exist: True
scan: configs=1640 exact-tie-rival-pairs(noise-broken)=4 float-changed-the-counts=0
TIE_PROBE_DONE noise_exists=True reachable_exact_ties=1 float_changed_counts=0
```
So: exact-arithmetic ties with different float sums DO exist (2x cold + nvfp4 = 0.80 =
10x e8, and the 10x e8 accumulation lands on 0.79999...), 4 such rival pairs are reachable
in the scan; in 0/1640 configs did float noise change the winner's counts vs an exact
solver; and the extended grid shows C++ == Python on all those config classes. A's
float-domain tie-break replication is CONFIRMED within the reachable range tested — the
rounding hole (Check 2) is the only parity break found.

## Files
- `_collab/C_s14_grid.sh` — extended parity grid (RAM-gated big-L phases)
- `_collab/C_s14_tie_probe.py` — exact-vs-float tie probe
- `_collab/C_s14_tu.cpp` — interface source-compat TU
- `_collab/C_s14_rnd.cpp` — rounding root-cause micro-proof
- build/run artifacts in `/tmp/kvbit_c18` only
