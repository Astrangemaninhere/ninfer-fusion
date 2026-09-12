# S20 (N1) — reviewable patch: `--kv-bit-budget <bits>` engine entry point

Deliverable: `_collab/A_n1_patch.diff` (117 diff lines, 5 files). **NOT APPLIED** — repo
tree untouched (verified: `serve_options.cpp` md5 repo == pristine copy; window D owns
the build). Generator: `_collab/A_n1_mkpatch.py` (anchored insertions + `patch -p1
--dry-run` applicability proof against fresh copies of the current tree).

## 1. Parse-site analysis (the known blocker, resolved by deferral)

The CLI parses `--kv-layer-storage` at `src/serve/serve_options.cpp:266-272`, inside
`parse_serve_options` (entry at :119). Available context there: **ServeOptions only** —
plain CLI flags and strings; the artifact path is a *string* (`options.artifact_path =
argv[1]`, :140), no model is loaded, no config exists. The full-attention layer count is
a per-target compile-time constant (`TextConfig::full_attention_layers()`, alias from
`src/targets/qwen3_6/impl/runtime/instance.h:16`, consumed e.g. at
`layouts_impl.h:138`) and is **genuinely not reachable from src/serve**. Confirmed
blocker → the patch stores the raw number at parse time and resolves later.

**Deferred-resolution point (minimal, where the layer count IS known):**
`src/targets/qwen3_6/impl/runtime/layouts_impl.h`, `make_sequence_planner_impl`
(:943; post-patch block at ~:950-971). This is exactly where `options.kv_layer_storage`
is materialised into `layer_overrides` today (:949-957 pre-patch); the patch resolves
the budget into the same table type via `product::kv_bit_budget_spec(full_layers, bits)`
(same grammar) → `product::parse_kv_layer_storage` → the existing `has_override` branch
proceeds unchanged downstream. One stderr line (`[kv-bit-budget] full_attention_layers=…
bits=… -> spec`) gives startup forensics (engine has no logger at this layer; reviewers
may swap it).

## 2. Patch contents (hunks)

| file | site | change |
|---|---|---|
| `include/ninfer/types.h` | after :163 (`kv_layer_storage_explicit`) | EngineOptions += `kv_bit_budget_bits` (double, 0=off) + `kv_bit_budget_explicit` |
| `src/serve/serve_options.h` | after :47 | same two fields on ServeOptions |
| `src/serve/serve_options.cpp` | after :272 (parse chain) | `--kv-bit-budget` via existing `parse_float_in(..., 0.01f, 16.0f)` (16.0 = bf16 tier ceiling; helper at :27) |
| `src/serve/serve_options.cpp` | before :414 (`default_max_tokens_explicit` block) | cross-flag validation: budget + layer-storage both explicit → `invalid_argument` |
| `src/serve/generation_service.cpp` | after :243 | copy the two fields into engine_options (same plumbing as kv_layer_storage) |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h` | includes (:14) + `make_sequence_planner_impl` (:948) | `#include "product/kv_bit_budget.h"` + `<cstdio>`; deferred resolution block; `has_override` reads the resolved table |

Line-ending note: `types.h`/`generation_service.cpp` are CRLF, the rest LF; the
generator preserves each file's style so the diff never churns endings.

## 3. Host-only compile + behavior evidence (plain g++, /tmp only)

Commands (WSL g++ 15.2.0; `/tmp/n1patch/b` = patched copies, `-I` order puts them first):

```
wsl.exe -e bash -c "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_n1_mkpatch.py"
  -> PATCH_WRITTEN ... (117 diff lines) / PATCH_DRYRUN_OK (patch -p1 --dry-run on fresh copies)

wsl.exe -e bash -c "cd /tmp/n1patch && g++ -std=c++20 -O1 -c -I/tmp/n1patch/b/src -I<repo>/src -I<repo>/include b/src/serve/serve_options.cpp -o serve_options_patched.o && echo PATCHED_SERVE_OPTIONS_COMPILES"
  -> PATCHED_SERVE_OPTIONS_COMPILES   (the REAL patched parse TU compiles standalone)

wsl.exe -e bash -c "cd /tmp/n1patch && g++ -std=c++20 -O1 -I/tmp/n1patch/b/include -I/tmp/n1patch/b/src -I<repo>/src -I<repo>/include /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_n1_compile_check.cpp serve_options_patched.o -o n1_check && ./n1_check"
```

Output (7/7):

```
PASS parse --kv-bit-budget 4.5
PASS budget + layer-storage -> invalid_argument(mutually exclusive)
PASS --kv-bit-budget 17 out of range -> invalid_argument
PASS L=16 b=4.5 spec == python golden     (0-7:e8,8:int8,9-14:nvfp4,15:iso3  == S14 grid L=16 c=0 b=4.5)
PASS L=16 b=4.5 table slots               (E8x8 / Int8x1 / Nvfp4x6 / Iso3x1, slot 16 stays BFloat16)
PASS L=8 b=4.5 spec == python golden      (0-7:e8  == _collab/A_kvbit_golden_l8.txt line 5)
PASS no budget -> no resolution           (defaults path untouched)
ALL_CHECKS_PASS (0)
```

Check source: `_collab/A_n1_compile_check.cpp` (links the real patched parse TU; the
layouts block is lifted verbatim with a fake TextConfig because the engine TU pulls
CUDA op headers — stated scope: the NEW parsing/resolution logic compiles, not the
whole engine TU). Note the test itself caught a wrong expectation of mine first (I had
quoted the b=4.0 golden for 4.5); corrected against the persisted S14 dump.

## 4. Precedence + interactions

- `--kv-bit-budget` + explicit `--kv-layer-storage`: **hard error** at post-parse
  validation (mutually exclusive table sources; silent preference would hide user
  error). Workflow: use the budget OR paste a table from `kv_bit_budget.py`.
- `--kv-bit-budget` alone: DP with defaults e8_limit=8, **cold_cap=0** (hot-only);
  result table replaces the target default table exactly like `--kv-layer-storage`
  does (same wholesale-replacement semantics, incl. the BFloat16-inherits-kv-dtype
  caveat documented in kv_options.h).
- Global `--kv-dtype`: unchanged relationship — the resolved table wins per-layer;
  BFloat16 slots in the table inherit `--kv-dtype` (DP never emits bf16 except at
  budget >= 16 where everything is bf16 anyway).
- Cold mapping (N2, NOT in this patch): with S12/S14 semantics, cold layers cost 0 hot
  bits and ONE 9536 B page each; the engine's cold capacity is *derived* today at
  `layouts_impl.h:151-153` — `max_cold_pages = (cold_policy == Window || Disk) ?
  cold_keep_tokens / kPagedKVPageSize + 16 : 0` (there is no `--max-cold-pages` CLI
  flag; `--cold-policy host` currently derives 0 cold pages — an N2 prerequisite
  decision). N2 wiring = pass that derived value as the DP `cold_cap`, map DP-cold
  layers' hot-window slot (proposal: Nvfp4Group16, since cold slots hold requantized
  E2M1-family data per decoder_state.cpp:174-179), and surface a `--max-cold-pages`
  override if the derived value should be user-controllable.

## 5. What this patch does NOT do

- Does NOT apply anything (diff-only; repo tree clean of my writes).
- No cold coupling (that is N2, above); budget path resolves cold_cap = 0.
- No runtime calibration loop (N3): penalties remain the S12/S14 priors.
- Does not touch the OTHER parse sites: `apps/cli/options.cpp:144` (ninfer CLI) and
  `apps/perplexity/main.cpp:115` also parse `--kv-layer-storage`; the serve path is
  wired, those apps would repeat the same 8-line pattern if wanted (flagged for the
  reviewer, not silently skipped).
- No GUI/studio surface, no usage-text updates, no docs beyond this file.
- The patch context is today's working tree (which carries uncommitted collab changes,
  e.g. `M types.h`, `M generation_service.cpp` — pre-existing, not mine); the dry-run
  proves applicability to the current state, re-run `A_n1_mkpatch.py` if the tree moves.
