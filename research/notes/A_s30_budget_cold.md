# S30 (N2 coupling) — prepared patch: `--kv-bit-budget` resolved together with the cold capacity

Deliverable: `_collab/A_s30_budget_cold.diff` (1 file, 147 diff lines — only
`src/targets/qwen3_6/impl/runtime/layouts_impl.h`) + generator `A_s30_mkpatch.py`
(builds the diff against the N1+n1b state, proves the stack) + host check
`A_s30_check.cpp`. **NOT APPLIED** — window G owns the build; repo tree untouched.

## 1. Apply order (explicit) + stack proof

`A_n1_patch.diff` (S20/N1) → `A_n1b_cold_pages.diff` (S25) → `A_s30_budget_cold.diff`.
The generator proves the exact order (fresh copies each step, `patch -p1 --dry-run`
before every apply):

```
wsl.exe -e bash -c "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_s30_mkpatch.py"
  -> PATCH_WRITTEN ... (147 diff lines)
     STEP1_N1_DRYRUN: OK   -> apply -> STEP2_N1B_DRYRUN: OK -> apply
     STEP3_S30_DRYRUN: OK  -> apply -> STACK_FULLY_APPLIED
     PATCH_DRYRUN_OK
```

## 2. What the patch does (requirements 1 + 2)

Only `layouts_impl.h`, three hunks:

1. **`effective_cold_pages(policy, keep_tokens, explicit_pages)`** — the S25
   precedence as ONE shared helper: `None` → 0 always; `Host` → 0 until a runtime
   host-eviction consumer exists; `Window/Disk` → explicit cap when non-zero, else
   `keep_tokens / kPagedKVPageSize + 16` (page size 64, `core/paged_kv_cache.h:17`).
2. `persistent_layout`'s `.max_cold_pages` derivation now CALLS the helper (same
   semantics as S25's inline ternary, single source of truth). **The DP's cold
   capacity therefore always equals the pool the layout reserves** — the DP can
   never plan cold layers the pool cannot hold.
3. The N1 budget-resolution block resolves with the cold capacity: DP call
   `kv_bit_budget_spec(layers, bits, e8_limit, cold_cap)` where
   `cold_cap = effective_cold_pages(...)`. The plan is split — **no new storage
   tier invented**:
   - hot table: `cold` ranges are re-emitted as `nvfp4` (cold-planned layers keep a
     hot NVFP4 window — cold slots hold requantized E2M1-family data,
     decoder_state.cpp:174-179), everything else verbatim →
     `parse_kv_layer_storage` → the existing N1 override path unchanged;
   - cold hint: explicit forensics line
     `[kv-bit-budget] full_attention_layers=L bits=B cold_pages=P cold_placed=R hot=H
     (cold residency needs --cold-policy window|disk; pool sized by --max-cold-pages >= P)`.

**Host statement (out-of-scope confirmation, from evidence)**: `ColdPolicy::Host`
has zero runtime consumers (S25 grep; `program_impl.h:865/875` gate on Window/Disk),
so Host CANNOT place cold layers today — `effective_cold_pages` returns 0 for it and
the DP runs hot-only. That is stated in the patch comment, not worked around.

## 3. Byte-identity guard + S14 parity (requirement 3)

- With `cold_cap == 0` the 4-arg DP call degenerates to the N1 2-arg call (same
  default), no `cold` ranges exist, `hot_spec == spec` byte-for-byte. Proven in the
  host check against the persisted S14 goldens:
  `cold_cap=0 L=16 b=4.5 -> hot table == "0-7:e8,8:int8,9-14:nvfp4,15:iso3"`
  (S14 dump L=16 C=0 b=4.5), and L=8 b=4.5 → `0-7:e8` covered by the N1 check
  (same code path).
- Host check (`wsl.exe -e bash -c "cd /tmp/s30patch && g++ -std=c++20 -O1 -I/tmp/s30patch/a/include -I<repo>/include -I<repo>/src /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_s30_check.cpp -o s30_check && ./s30_check"`,
  compiled against the PATCHED types.h from the N1+n1b stack tree):

```
PASS cold_cap=0 L=16 b=4.5 hot table == S14 golden
PASS policy None + explicit cap -> forced hot-only (None wins)
PASS policy Host + explicit cap -> 0 (host cannot place cold today)
PASS Window derivation 128/64+16 = 18 pages
PASS L=16 b=4.0 Disk/8 -> cold 7-11, hot nvfp4 window (S14 dump)
PASS L=16 b=4.0 Disk/8 hot table slots parse
PASS L=30 b=4.5 Window/8 -> cold 8-15 (S14 dump)
ALL_CHECKS_PASS (0)
```

- S14 C++ parity re-run on the current state (S30 does not touch
  `src/product/kv_bit_budget.h`, so parity is structurally unchanged; re-run anyway
  per requirement): `bash _collab/A_s22_grid.sh` (C's `C_s14_grid.sh` verbatim +
  phase 5) → **`EXTENDED_PARITY ... DIVERGED=0` + `PHASE5_PARITY ... DIVERGED=0` +
  `S22_ALL_MATCH`** (exact line quoted in board row).

## 4. Sample resolutions (requirement 4; expectations = S14 dump values, check-verified)

**(a) L=16, bits=4.0, `--cold-policy disk --max-cold-pages 8`**

```
DP plan      : 0-6:e8,7-11:cold,12-15:int8        (mix coldx5+e8x7+int8x4)
hot table    : 0-6:e8,7-11:nvfp4,12-15:int8       (cold -> nvfp4 hot window)
cold hint    : cold_placed=7-11, needs window|disk, pool >= 8 pages
DP achieved  : 3.84 bits/element (cold layers counted at 0 hot bits)
hot footprint: 5.25 bits/element (7*406 + 5*450 + 4*825)/1600 -- the nvfp4 hot
               windows are the price of cold residency; the cold pages themselves
               live in the 9536B/slot pool (5 layers * 8 pages capacity)
```

**(b) L=16, bits=4.5, `--cold-policy disk --max-cold-pages 8`**

```
DP plan   : 0-6:e8,7-10:cold,11-15:int8   (coldx4+e8x7+int8x5, achieved 4.35)
hot table : 0-6:e8,7-10:nvfp4,11-15:int8 ; cold_placed=7-10
```

**(c) L=30, bits=4.5, `--cold-policy window --max-cold-pages 8`** (default
keep-tokens would derive 18; explicit 8 caps it):

```
DP plan   : 0-7:e8,8-15:cold,16-18:int8,19-26:fp8,27-29:nvfp4 (achieved 4.50)
hot table : 0-7:e8,8-15:nvfp4,16-18:int8,19-26:fp8,27-29:nvfp4 ; cold_placed=8-15
```

Sanity check for a human: at the same budget the no-cold table for L=16/4.0 is
`0-7:e8,8-12:nvfp4,13-15:iso3` (penalty 3.64); the coupled plan buys penalty 1.89
by moving 5 layers' aged KV into the cold pool — hot footprint rises (5.25 vs 4.00)
because cold windows run nvfp4, which is exactly the hot/cold trade the operator
should eyeball before committing the flags.

## 5. Out of scope (stated)

- Runtime calibration loop (N3, C owns).
- Any `ColdPolicy::Host` offload wiring — Host cannot place cold layers today
  (zero runtime consumers; stated in-code, not worked around).
- The nvfp4-hot-window choice for cold-planned layers is a documented default
  (cold slots hold requantized E2M1-family data); if cold A/B says another window
  tier wins, it is a one-line change at the split site.
