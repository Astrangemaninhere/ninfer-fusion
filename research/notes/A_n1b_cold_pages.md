# S25 (N1b) — prepared patch: `--max-cold-pages N` explicit cold-pool cap

Deliverable: `_collab/A_n1b_cold_pages.diff` (6 files, 126 diff lines) + generator
`_collab/A_n1b_mkpatch.py` (anchored, line-ending preserving, loud anchor checks) +
`_collab/A_n1b_stack_check.sh` (N1→N1b stacking proof). **NOT APPLIED** — window H
owns application; repo tree untouched.

## 1. The gap (code-confirmed)

`src/targets/qwen3_6/impl/runtime/layouts_impl.h:151-154` derives the cold budget:
`max_cold_pages = (Window || Disk) ? cold_keep_tokens / kPagedKVPageSize + 16 : 0`.
No CLI flag sets an explicit cap; `--cold-policy host` derives 0 pages.

## 2. Patch path (same plumbing as N1)

`--max-cold-pages N` (u64 parse, reject > uint32 max — mirrors `--cold-keep-tokens`
validation at serve_options.cpp, reject-not-clamp) → `ServeOptions.max_cold_pages`
(serve_options.h) → `EngineOptions.max_cold_pages` (types.h, after cold_keep_tokens) →
generation_service.cpp copy → `SequencePlanningInputs.max_cold_pages` +
`SequencePlanImpl.max_cold_pages` (layouts.h, both structs; inputs→impl copy at
layouts_impl.h:877-881, options→inputs at :999) → derivation at :151.

## 3. Precedence decisions (from code; task item 2)

| policy | derived (today, unchanged) | explicit `--max-cold-pages N>0` |
|---|---|---|
| none | 0 | **0 — None wins** (commented in the patch) |
| window | keep_tokens/128+16 | **N (cap wins)** |
| host | 0 | **0 — stays derived-off even with a cap** (see below) |
| disk | keep_tokens/128+16 | **N (cap wins)** |

- **None must win**: `plan_decoder_state` reserves cold device slots whenever
  `max_cold_pages != 0` regardless of policy (decoder_state.cpp:180 `if
  (spec.max_cold_pages != 0)` → add_tensor per layer), and the runtime never evicts
  under None — honoring a cap there would reserve device memory nothing consumes.
- **Host stays 0 even with an explicit cap** — determined from code:
  `grep ColdPolicy::Host` outside option parsing has ZERO consumers; the runtime cold
  machinery gates on `Window || Disk` (requant buffers, program_impl.h:865) and `Disk`
  only (spill files, :875). Host slots would be dead device memory. **What I could not
  determine from code**: whether N2's host tier should reuse these very slots (then
  this branch flips to honor the cap) or get separate host-side storage — flagged as
  the open design point; the comment in the patch says exactly this.
- **Window/Disk**: explicit cap replaces the derivation (cap is in pages, same unit
  as `max_cold_pages`).

## 4. DP number alignment (task item 3, S12/S14 semantics)

The cap is the same quantity the cold DP treats as capacity: one cold-resident layer
costs 0 hot bits and ONE page = one 9536 B slot per (page, kv_head, plane)
(`kEntropyNvfp4SlotBytes`, include/ninfer/ops/entropy_nvfp4_slot.h:14). When N2 wires
the joint selection, `kv_bit_budget_spec(layers, bits, e8_limit, cold_cap=
this max_cold_pages)` lines up by construction. Carried-over simplification (S12 doc):
the DP charges page-granular K+V-averaged slots, the engine reserves per plane.

## 5. Evidence

```
wsl.exe -e bash -c "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_n1b_mkpatch.py"
  -> PATCH_WRITTEN ... (126 diff lines)
     checking file include/ninfer/types.h / serve_options.h / serve_options.cpp /
       generation_service.cpp / layouts.h / layouts_impl.h
     PATCH_DRYRUN_OK                     (fresh copies, 6/6)

wsl.exe -e bash -c "bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_n1b_stack_check.sh"
  -> N1_APPLIED
     Hunk #1 succeeded at 192 (offset 8 lines) ... (all hunks offsets only, no fuzz)
     N1B_AFTER_N1_DRYRUN_OK
     N1B_AFTER_N1_APPLIED             (window H order proven: N1 first, then N1b)
```

## 6. What this patch does NOT do

- Does not give `ColdPolicy::Host` a runtime eviction/restore path (that is the real
  host-cold gap; this flag is the N2 prerequisite, not the fix).
- Does not wire the cap into the bit-budget DP (N2 joint selection).
- No GUI/studio surface, no usage-text updates, no docs beyond this file.
