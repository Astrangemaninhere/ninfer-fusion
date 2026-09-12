# S24 (E3) — prepared patch: per-layer SWA window table into the KV layer views

Deliverable: `_collab/A_s24_window_table.diff` (3 files, 174 diff lines) + generator
`_collab/A_s24_mkpatch.py` (anchored, byte-level, line-ending-preserving, loud anchor
checks). **NOT APPLIED** — window D owns the build; repo tree untouched.

## 1. Verified plumbing (coordinator's points confirmed, two name corrections)

The chain mirrors `layer_residual` exactly; coordinator's "KVCacheSpec" is actually two
structs — `DecoderStateSpec` (input) and `PagedKVCacheLayout` (resolved), both in
`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h`:

| site | file:line (verified today) | residual analogue |
|---|---|---|
| table source | `layouts_impl.h` `persistent_layout` (~:133-160, where `TextConfig::full_attention_layers()` is used at :138) | `plan.kv_residual_layers` |
| spec field | `decoder_state.h` DecoderStateSpec `.layer_residual` :39 | + `.layer_sliding_windows` |
| plan_cache param+validation | `decoder_state.cpp` :21/:37 (span + shorter-than-layers reject) | + `layer_windows` span + same-style reject |
| resolved table | `decoder_state.cpp` :87 `residual_flags` + per-layer fill (:90 loop) | + `window_flags` |
| layout field | `decoder_state.cpp` :151 `.layer_residual = residual_flags` | + `.layer_sliding_windows = window_flags` |
| ctor member | `decoder_state.cpp` :202 `layer_residual_(...)` | + `layer_sliding_windows_(...)` |
| views | `decoder_state.cpp` `layer_view` :249 / `batch_layer_view` :301 (both designated-initializer, both omit `.sliding_window_tokens` today → default 0) | + `.sliding_window_tokens = window` |

View field declarations already exist: `src/core/paged_kv_cache.h:60/:87`
(`sliding_window_tokens = 0` default). Read side confirmed already wired
(`gqa_attention_decode.cu:209/388/456`, `gqa_attention_prefill.cu:132/149`, e8 variant).

## 2. `sliding_window == 0` semantics — answer from code (task item 3)

Every kernel consumer guards on positivity, e.g.
`gqa_attention_decode_iso3.cuh:132` `token_begin = (sliding_window > 0) ? window_full - sliding_window : 0;`
(same at `gqa_attention_decode_nvfp4.cuh:273`, `gqa_attention_prefill_nvfp4.cuh:1160`,
5 files total with `sliding_window > 0` guards). **0 = attend the whole cache** — it is
the correct "no window" value at every layer. Consequences, all from code:

- qwen4_exp declares `sliding_window = 0` (`config.h:18`): its table computes all-zero →
  kernel-identical to today.
- The qwen3_6 family variants (27b/35b) declare NEITHER `sliding_window` NOR
  `is_swa_attention` (grep over `src/targets/*/impl/config.h`: only muse_glimmer_30b
  and qwen4_exp have them). The patch therefore computes the table under
  `if constexpr (requires { TextConfig::sliding_window; TextConfig::is_swa_attention(0); })`
  — non-declaring variants keep the all-zero table, byte-identical behavior, and the
  branch compiles away.
- Muse (`config.h`: `sliding_window = 2048` :41, `layer_kind[52]` :59, `is_swa_attention`
  :71, full-attention count via `!= 2` :69): 39 SWA layers of 52 get 2048; kind-0 layers
  and the MTP draft cache (span `{}`) keep 0 = today's behavior.

## 3. Evidence

```
wsl.exe -e bash -c "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_s24_mkpatch.py"
  -> PATCH_WRITTEN ... (174 diff lines)
     checking file .../decoder_state.h / .../decoder_state.cpp / .../layouts_impl.h
     PATCH_DRYRUN_OK            (patch -p1 --dry-run on fresh copies, 3/3)
```
Generator self-checks: every anchor exactly-once (two view tails are byte-identical, so
the two insertions are done last-occurrence-first with a post-count assert — the first
draft double-inserted into `layer_view`; caught by the assert, fixed); patched file
verified to contain exactly 2 `window` decls + 2 `.sliding_window_tokens = window`
(lines 315 `layer_view` / 373 `batch_layer_view`).

Compile scope honestly stated: a host `g++ -fsyntax-only` of the patched TU is NOT
possible — `core/paged_kv_cache.h` includes `cuda_runtime_api.h` (attempted, fatal
error). The patch is dry-run-proven and structurally mirrors the compiling residual
path; the TU compile belongs to the window that applies it.

## 4. Acceptance test (stated up front, per task)

Apply in a build window, then Muse needle probes at 8K / 32K / 57K contexts
(`tools/archkit/longtest_57k.py` probe, §35 protocol, §47 config: shallow e8 + deep
NVFP4). **Expectation: window ON (39×2048) must NOT degrade needle hits at any depth vs
the §47 window-off baseline.** Contexts > 2048 are where behavior changes (SWA layers
stop attending beyond the window, per U4: today Muse runs SWA layers as full attention
by design defer). If any depth regresses, the finding is real (revert, then
investigate — likely window-vs-training mismatch), not to be explained away.
Additionally: qwen3.8-27b serve smoke must be output-identical (all-zero window table,
kernel args unchanged) — that is the non-SWA regression gate.

## 5. What the patch does NOT do

- No kernel changes (read side already forwards `cache.sliding_window_tokens`).
- No memory/page-count reduction: `plan_cache` still sizes pages for full context on
  SWA layers (real memory savings would need window-aware page trimming — separate
  work item; this patch restores SEMANTICS first, per U4's framing).
- No CLI/runtime window override; window comes only from the target config.
- No cold-tier or budget interaction (S25/N2 territory).
