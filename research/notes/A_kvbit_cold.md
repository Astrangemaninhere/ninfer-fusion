# S12 — kv_bit_budget.py opt-in cold-tier cost model (agent A, round 5)

Artifact: `ninfer-fusion-repo/tools/archkit/kv_bit_budget.py` (updated).
Reference read before editing: `ninfer-fusion-repo/src/product/kv_bit_budget.h` (C++ port,
same tier table 1600/825/803/450/406/300 x100, same pack order e8-first) — tier semantics
kept identical; the cold model is Python-tool-only for now (engine wiring = board N2, open).

## 1. Model

Opt-in flags (absent/0 => cold off, allocator untouched):

- `--cold-pages N`: cold pool capacity in pages (engine `--max-cold-pages`). At most N
  layers may be cold.
- `--cold-bytes B`: same by bytes; cap = B // 9536 (`COLD_SLOT_BYTES`, mirrors engine
  `ops::kEntropyNvfp4SlotBytes`; decoder_state.cpp:174-179 "9536 B covers the rANS max").
  `--cold-pages` wins if both given.

Cold pseudo-tier "cold" (only in the DP candidate list when cap > 0):

- hot bit cost **0.00 bits/element** (layer's KV lives in host/disk cold slots, not the
  GPU pool budget);
- quality penalty **0.25** (restore/requant prior — cold slots hold int8-grade packed
  data; between e8 0.08 and nvfp4 0.30; prior, refine with cold A/B later);
- consumes 1 cold page of capacity per cold layer; DP carries `cold_used` as an explicit
  state dimension so the cap is an exact constraint (mirrors the C++ port's explicit
  e8 dimension).
- Packing (`PACK_ORDER_COLD`): **e8 still first** (the e8 limit is a LEADING-layer
  constraint — e8 must keep slots 0..count-1 inside its verified 0-7 window), then `cold`
  on the next-shallow block, deep layers keep the hot tail (deep-protection, _TODO 46/47).
  First version packed cold first; that displaced e8 to layers 5-11 (outside the verified
  window) — caught in self-review of the sample output and fixed before delivery.
- "cold" is NOT an engine `--kv-layer-storage` tier (grammar: bf16/int8/fp8/nvfp4/iso3/e8,
  serve_options.cpp:266 + kv_options.h:21). The mixed spec is the placement plan; cold
  layers deploy via `--cold-policy host|disk --max-cold-pages N`. The tool prints this
  hint line whenever cold mode is on.

Simplification (documented in the script docstring): one cold page per layer is charged
one 9536 B slot, K+V averaged, consistent with how TIERS averages K+V; the engine actually
reserves slot_bytes per (page, kv_head, plane). OPEN refinement.

## 2. No-flag path is byte-identical (evidence)

Golden captured BEFORE any edit (original script preserved as `_collab/A_kvbit_orig.py.bak`):

```
wsl.exe -e bash -c "cd /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit && python3 kv_bit_budget.py --layers 8 --bits 3.0 3.5 4.0 4.5 5 6 8 12 16 > /tmp/kvbit_golden_l8.txt && python3 kv_bit_budget.py --layers 16 --bits 4 6 10 > /tmp/kvbit_golden_l16.txt && python3 kv_bit_budget.py --layers 8 --bits 4.5 --e8-layers 0 > /tmp/kvbit_golden_e0.txt && echo GOLDEN_CAPTURED"
```
Copies: `_collab/A_kvbit_golden_l8.txt`, `A_kvbit_golden_l16.txt`, `A_kvbit_golden_e0.txt`.

After the edit, same three commands with `/tmp/kvbit_new_*` targets, then:

```
wsl.exe -e bash -c "diff /tmp/kvbit_golden_l8.txt /tmp/kvbit_new_l8.txt && diff /tmp/kvbit_golden_l16.txt /tmp/kvbit_new_l16.txt && diff /tmp/kvbit_golden_e0.txt /tmp/kvbit_new_e0.txt && echo NOFLAG_BYTE_IDENTICAL"
```
Output: `NOFLAG_BYTE_IDENTICAL` (all three diffs empty — 16 output lines total, zero bytes
differ). Also `python3 -m py_compile kv_bit_budget.py` -> `COMPILE_OK`.

Mechanism (why identity is structural, not luck): with `cold_cap == 0` the candidate list
is exactly `ORDER` (no "cold" tier ever enumerated) and the extra `cold_used` DP key
component is constantly 0 with identical insertion order and identical strict-`<`
tie-breaking, so the surviving counts and every printed byte are unchanged. Tier table,
penalties, `ORDER`, `PACK_ORDER`, and the default header line are untouched.

## 3. Sample allocations at different cold capacities

Command form (reproducible as written):
`wsl.exe -e bash -c "cd /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit && python3 kv_bit_budget.py <args>"`

(a) 8 layers, budget 3.5 bits, `--cold-pages 2` vs no-cold golden line:
```
no-cold : bits= 3.50 -> achieved= 3.40 penalty= 2.74 mix=e8x3+iso3x5       0-2:e8,3-7:iso3
cold=2  : bits= 3.50 -> achieved= 3.04 penalty= 0.98 mix=coldx2+e8x6       --kv-layer-storage 0-5:e8,6-7:cold
```
Implied hot bit cost: 6 x e8 / 8 layers = 3.04 bits/element on the GPU (2 layers' aged KV
in cold slots, 2 x 9536 B); total quality penalty drops 2.74 -> 0.98 (2.8x) at the same
GPU budget. e8 stays inside 0-5 (verified window).

(b) 16 layers, budget 4.0 bits, `--cold-bytes 76288` (= 8 pages):
```
no-cold : bits= 4.00 -> achieved= 4.00 penalty= 3.64 mix=e8x8+iso3x3+nvfp4x5   0-7:e8,8-12:nvfp4,13-15:iso3
cold=8p : bits= 4.00 -> achieved= 3.84 penalty= 1.89 mix=coldx5+e8x7+int8x4    --kv-layer-storage 0-6:e8,7-11:cold,12-15:int8
```
Implied hot bit cost: (7x4.06 + 4x8.25)/16 = 3.84 bits/element; 5 layers cold (5 x 9536 B
modeled); penalty 3.64 -> 1.89 (1.9x). Deep layers 12-15 stay HOT int8 (deep protection);
cold lands on middle block 7-11; e8 within 0-6.

(c) 8 layers, budget sweep with `--cold-pages 3` (cold used only when it pays):
```
bits= 3.50 -> achieved= 3.04 penalty= 0.98 mix=coldx2+e8x6     --kv-layer-storage 0-5:e8,6-7:cold
bits= 4.00 -> achieved= 3.55 penalty= 0.81 mix=coldx1+e8x7     --kv-layer-storage 0-6:e8,7:cold
bits= 5.00 -> achieved= 4.58 penalty= 0.58 mix=e8x7+int8x1     --kv-layer-storage 0-6:e8,7:int8
bits= 8.00 -> achieved= 7.73 penalty= 0.22 mix=e8x1+int8x7     --kv-layer-storage 0:e8,1-7:int8
```
At 3.5/4.0 bits cold buys quality (penalty 0.98/0.81); at 5+ bits the budget is loose and
the DP drops cold entirely — honest, no forced usage. Also verified `--cold-pages 16`
gives the same allocation as cap 8 (capacity is an upper bound, not a target).

## 4. Diff summary of the script

`_collab/A_kvbit_orig.py.bak` (125 lines) -> new (187 lines): docstring cold section;
`PACK_ORDER_COLD`, `COLD_BITS/COLD_PENALTY/COLD_SLOT_BYTES`; `solve(..., cold_cap=0)` with
3-tuple DP key + cold candidate/guard; `to_spec` picks pack order by presence of cold;
`main` gains `--cold-pages/--cold-bytes` + conditional `# cold:` hint line. Default header
line and per-budget output lines unchanged (section 2).

## 5. Open / handoff

- C++ port `src/product/kv_bit_budget.h` not extended (engine wiring = board N2); tier
  semantics unchanged so the port stays consistent for the no-cold path.
- COLD_PENALTY=0.25 is a prior; calibrate against a real `--cold-policy host|disk` A/B.
- Slot-bytes accounting is page-granular K+V-averaged (engine reserves per
  (page, kv_head, plane)); refine if used for exact host-RAM sizing.
