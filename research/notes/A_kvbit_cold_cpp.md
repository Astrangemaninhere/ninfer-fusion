# S14 — C++ port of the cold-tier KV bit-budget model + mechanical Python parity

Artifact: `ninfer-fusion-repo/src/product/kv_bit_budget.h` (extended).
Prior-round reference: `_collab/A_kvbit_cold.md` (S12, Python model).
No engine TU includes this header yet (`grep -rln kv_bit_budget src apps tests` → only the
header itself), so the CUDA build tree is untouched; everything below is plain host g++.

## 1. The 9536 constant

`kKvBitBudgetColdSlotBytes = 9536` mirrors **`ops::kEntropyNvfp4SlotBytes`**, defined at
`include/ninfer/ops/entropy_nvfp4_slot.h:14` ("320 B header + 32 streams x 256 B + 1024 B
scale tail. The INT8 tier packs its raw 9232 B layout into the same buffer"), consumed at
`src/targets/qwen3_6/impl/state/decoder_state.cpp:179` (`slot_bytes` for the cold pool).
The value is spelled out in `kv_bit_budget.h` (with file:line citation) instead of
including that header, because the latter pulls `cuda_runtime.h` and this header is
host-only.

## 2. What was added (mirror of the S12 Python semantics)

- `kv_bit_budget_solve(layers, budget_bits, e8_limit=8, cold_cap=0)` returning
  `KvBitBudgetSolution{spec, achieved_bits, penalty, counts}`; `kv_bit_budget_spec(...)`
  kept source-compatible as a thin wrapper (adds one defaulted `cold_cap` param).
- Cold pseudo-tier (index 6, outside the 6-entry hot table so engine consumers of
  `kKvBitBudgetTiers` are untouched): **0 hot bits/element**, ONE `kKvBitBudgetColdSlotBytes`
  (9536 B) page of capacity per cold layer, penalty prior 0.25, `cold_cap` = pages cap
  (cap = bytes/9536 for the by-bytes form, Python side).
- Pack order `kKvBitBudgetPackOrderCold = {e8, cold, bf16, int8, fp8, nvfp4, iso3}`:
  e8 STILL first (leading-layer constraint, stays in its verified 0-7 window), cold on the
  next-shallow block, deep layers keep the hot tail.
- `cold_cap == 0` (default): cold candidate never enumerated, cold index inert.

## 3. Mechanical parity proof (not eyeballed)

Harness: `_collab/A_kvbit_grid.sh` — builds two host binaries (new header; PRE-cold header
backup `_collab/A_kvbit_budget_h_orig.h.bak`), runs the Python tool and the C++ binary
over the grid **layers {16, 24, 30} x cold_pages {0..8} x budgets {3, 3.5, 4, 4.5, 5, 6, 8,
10, 12, 16}** (the budgets of the S12 golden sweeps), diffs every case line-by-line, and
diffs pre-cold vs new spec strings for the cold=0 column. Reproduce with:

```
wsl.exe -e bash -c "bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_kvbit_grid.sh"
```

The exact g++ commands inside it (WSL g++ 15.2.0, -std=c++20 -O1, include root
`-I<repo>/include`, header dir `-I<repo>/src/product`):

```
g++ -std=c++20 -O1 -I$REPO/include -I$REPO/src/product test_kvbit_new.cpp -o kvbit_new
g++ -std=c++20 -O1 -I$REPO/include -I$WORK test_kvbit_orig.cpp -o kvbit_orig
```

Test sources: `_collab/A_kvbit_cold_cpp_test.cpp` (prints byte-identical lines to the
Python tool), `_collab/A_kvbit_cold_cpp_orig_test.cpp` (pre-cold spec-only).

### Raw result (final)

```
BUILD_OK
PARITY_CASES=27 OUTPUT_LINES=321 DIVERGENT_CASES=0
ORIG_VS_NEW_COLD0_SPECS_IDENTICAL (30 specs)
```

Full dumps persisted: `_collab/A_kvbit_cold_cpp_grid_py.txt` / `_cpp.txt` (348 lines each,
per-case blocks), `_collab/A_kvbit_cold_cpp_orig_specs.txt` / `_new_specs.txt` (30 each).
No diverging case remains.

### A real divergence WAS found and fixed along the way (first report, verbatim)

Grid exit 3 at `L=24 C=6`, budget 4.50 — equal penalty, different allocation:

```
py : bits= 4.50 -> achieved= 4.50 penalty= 2.90 mix=coldx6+e8x8+fp8x8+int8x1+iso3x1  0-7:e8,8-13:cold,14:int8,15-22:fp8,23:iso3
cpp: bits= 4.50 -> achieved= 4.48 penalty= 2.90 mix=coldx6+e8x8+int8x8+nvfp4x2      0-7:e8,8-13:cold,14-21:int8,22-23:nvfp4
```

Diagnosis (both sides penalty 2.90; hot-bits 10797 vs 10748 of 10800 capacity): the old
C++ DP was exact-arithmetic with an explicit e8 dimension and a lowest-bits scan; the
Python reference collapses e8 into the carried counts, breaks ties by dict
first-insertion order, and — decisively — accumulates penalties as FLOATS
(0.08+0.30+0.25... in path order), so exact-arithmetic ties (2.90 == 2.90) are NOT ties
in Python's float domain. Fix in `kv_bit_budget_solve`: (a) state key (bits, cold_used)
with the surviving path's e8 count carried per state (Python counts semantics),
(b) expansion in first-insertion order over candidates ladder+["cold"] with
insert-or-strictly-improve (Python dict semantics), (c) penalties accumulated as
`penalty_x100 / 100.0` doubles in path order (bit-identical to the Python literals).
After (c) alone the case converged; (a)+(b) keep the structural mirror exact for any
future grid. Note: the pre-cold header's explicit-e8 DP agrees with all of this on the
whole cold=0 column (30/30 specs identical), so the default path is empirically unchanged.

## 4. Grid table — achieved hot bits/element (columns = cold pages 0..8)

From the parity-verified dumps (full penalty/spec detail in the dump files):

| case | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 |
|---|---|---|---|---|---|---|---|---|---|
| L=16 b=3.0 | 3.00 | 2.94 | 2.96 | 2.97 | 2.97 | 2.87 | 2.83 | 2.81 | 2.81 |
| L=16 b=4.0 | 4.00 | 4.00 | 3.95 | 3.90 | 3.86 | 3.84 | 3.84 | 3.84 | 3.84 |
| L=16 b=4.5 | 4.42 | 4.47 | 4.42 | 4.37 | 4.35 | 4.35 | 4.35 | 4.35 | 4.35 |
| L=16 b=6.0 | 5.92 | 5.90 | 5.90 | 5.90 | 5.90 | 5.90 | 5.90 | 5.90 | 5.90 |
| L=16 b=8.0 | 7.99 | 7.99 | 7.99 | 7.99 | 7.99 | 7.99 | 7.99 | 7.99 | 7.99 |
| L=24 b=3.0 | 3.00 | 2.96 | 2.97 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.95 |
| L=24 b=4.0 | 3.98 | 3.98 | 3.98 | 3.95 | 3.99 | 3.98 | 3.99 | 3.98 | 3.95 |
| L=24 b=4.5 | 4.45 | 4.48 | 4.45 | 4.49 | 4.50 | 4.49 | 4.50 | 4.45 | 4.45 |
| L=24 b=6.0 | 6.00 | 6.00 | 5.99 | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 |
| L=24 b=8.0 | 7.90 | 7.90 | 7.90 | 7.90 | 7.90 | 7.90 | 7.90 | 7.90 | 7.90 |
| L=30 b=3.0 | 3.00 | 2.97 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 |
| L=30 b=4.0 | 3.98 | 3.98 | 3.98 | 3.93 | 3.98 | 3.99 | 3.98 | 4.00 | 3.99 |
| L=30 b=4.5 | 4.46 | 4.48 | 4.46 | 4.49 | 4.50 | 4.49 | 4.48 | 4.50 | 4.50 |
| L=30 b=6.0 | 5.99 | 6.00 | 5.99 | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 |
| L=30 b=8.0 | 7.97 | 7.97 | 7.97 | 7.97 | 7.97 | 7.97 | 7.97 | 7.97 | 7.97 |

Sample full rows (identical in both dumps), L=30 cold=4:

```
bits= 3.00 -> achieved= 2.98 penalty=10.24 mix=coldx4+e8x8+iso3x16+nvfp4x2  --kv-layer-storage 0-7:e8,8-11:cold,12-13:nvfp4,14-29:iso3
bits= 3.50 -> achieved= 3.48 penalty= 8.24 mix=coldx4+e8x8+iso3x6+nvfp4x12 --kv-layer-storage 0-7:e8,8-11:cold,12-23:nvfp4,24-29:iso3
bits= 4.00 -> achieved= 3.98 penalty= 6.68 mix=coldx4+e8x8+int8x2+iso3x1+nvfp4x15 --kv-layer-storage 0-7:e8,8-11:cold,12-13:int8,14-28:nvfp4,29:iso3
```

Note the achieved column is not monotone in cold capacity: equal-penalty optima at
different bit totals are resolved by the reference tool's float-domain tie order — the
C++ port now resolves them identically by construction.

## 5. Open / handoff

- Engine wiring (board N2: `--kv-bit-budget`/cold-policy joint selection) still open; the
  header is staged and parity-proven but unreferenced by engine TUs.
- COLD_PENALTY=0.25 remains a prior (calibrate against a real `--cold-policy` A/B).
- Slot accounting stays page-granular K+V-averaged (engine reserves per
  (page, kv_head, plane)); see S12 notes.
