# C_s34_closed_loop.md — S34 (N4, user Q4): hot/warm/cold automatic dynamic allocation

Status: loop DEFINED + first patch PREPARED (`patch/C_s34_first_patch.diff`,
NOT applied — window G owns the build). No `src/` writes, no GPU, no build.

## 0. What exists today (read-verified, file:line)
* Observation tap: `src/ops/common/ft_stats.h:47` `observe()` — per-layer
  attention energy (partial_l log-sum-exp proxy), called from the live nvfp4
  decode launcher (post-§111 fix placement; NOTE: the S23 TU split moves that
  launcher into `gqa_attention_decode_nvfp4.cu`, tap travels with it).
  `snapshot()` (:71) exposes cumulative `(mean_l, rounds)` in-process.
* Existing W5 loop: `src/serve/kv_auto_relayout.{h,cpp}` — `NINFER_FT_RELOAD_SECS`
  thread, `build_ft_spec()` (deep protection + energy tertiles → e8/iso3/nvfp4),
  2-cycle spec hysteresis, semantic compare, `reload_kv_storage` apply.
  **Gap 1: it never decides COLD residency.** **Gap 2: `snapshot()` is a
  cumulative mean since process start — it cannot track phase changes.**
* Static cold half (A's S30, prepared): budget × `effective_cold_pages` →
  `kv_bit_budget_spec(..., cold_cap)` DP; cold layers get an nvfp4 hot window
  + `[kv-bit-budget] ... cold_placed=R` forensics.
* KV hit/bandwidth signals: not observed per layer today (`bandwidth_governor.h`
  is global, W6); energy is the only per-layer signal — the loop starts there.

## 1. The loop (concrete)
* OBSERVE — per layer ℓ: window mean energy + observation count, derived
  EXACTLY from snapshot deltas: `(sum₁−sum₀)/(count₁−count₀)` (ft::snapshot
  never resets; no change to ft_stats.h needed — pure decision-side math).
  Window = one decision cycle (`NINFER_FT_RELOAD_SECS`, default cadence 60 s).
* SMOOTH — EWMA per layer, α=0.5 (first window initializes).
* DECIDE — cold cut = `demote_quantile` (default 25th) of EWMA over non-deep
  observed layers. Candidate = coldest ELIGIBLE layers, |cold| ≤ cold_cap
  (cold_cap = S30 `effective_cold_pages`, fed via `NINFER_FT_COLD_PAGES`).
  Eligibility gates: not deep-protected (last 20% keep nvfp4), window rounds ≥
  32 (confidence), below-cut streak ≥ 2 cycles, residency dwell ≥ 3 cycles.
* ACT — first patch: DRY ONLY (`[ft][cold:dry] would cold=[…]` + reasons);
  live path (next iteration, requires S30+N1 stack): hot spec = existing
  `build_ft_spec` with cold layers forced to their nvfp4 hot window (S30
  convention), applied through the EXISTING `decide_once` 2-cycle hysteresis +
  `reload_kv_storage` (drain-based replan, §55/§56 machinery).
* DAMPING (anti-oscillation): (a) spec-level 2-cycle hysteresis (exists);
  (b) per-layer streak ≥ 2 below cut before demotion; (c) dwell ≥ 3 cycles in
  a residency; (d) promotion is FAST (one window above cut) — quality first,
  cheap safety second; (e) EWMA α=0.5 damps single-window noise; (f) at most
  one re-layout per cycle.
* COST BOUND of a re-layout (qwen3.8-27b, 32K ctx): per layer K+V = 32768 tok
  × 4 kv_heads × 256 hd × 2 planes → bf16 128 MiB, nvfp4 32 MiB. A demotion
  moves a layer's hot window (≤32 MiB) through the requant/rANS pack into
  9536 B slots; worst action = cold_cap(8) layers ≈ 256 MiB → PCIe floor
  ≈ 16 ms at 16 GB/s + quantize kernels. The REAL stall is the drain (in-flight
  requests, ≤120 s cap in generation_service) + CUDA-graph recapture — hence
  the one-action-per-cycle bound and the dry mode.

## 2. The invariant (and its automatic test)
> A layer is demoted to cold ONLY IF its EWMA energy was at or below the cold
> cut in EACH of the last `stable_cycles` windows, it has dwelled ≥
> `min_dwell_cycles` in its current residency, it is not deep-protected, it was
> observed ≥ `min_window_rounds` this window, and the cold set still fits the
> pool. Unobserved layers are never demoted.
Machine-checked by `invariant_violations(decision, cfg)` (reads evidence the
decision captures BEFORE the dwell reset), asserted after EVERY decision —
in the property test now, and live in the tap (any violation prints
`[ft][cold] INVARIANT VIOLATION: …` and blocks the action in live mode).

## 3. First patch (`_collab/C_s34_cold_loop/patch/C_s34_first_patch.diff`)
* NEW `src/serve/kv_cold_policy.h` — the whole loop as pure host-only C++
  (window math, EWMA, gates, decision, invariant). No CUDA/engine headers.
* HUNK `src/serve/kv_auto_relayout.cpp` `loop()` — after the energy snapshot:
  `NINFER_FT_COLD_PAGES>0` enables the tap; computes window stats from
  snapshot deltas, runs `decide_cold_residency`, checks the invariant, and in
  the default DRY mode logs `[ft][cold:dry] would cold=[…]` WITHOUT touching
  the serve. `NINFER_FT_COLD_MODE=live` is reserved but intentionally NOT
  wired (needs S30's pool consumer; wiring it blind would be a silent no-op).

## 4. Evidence produced now
```
policy host test (g++ -std=c++20 -O1, /tmp/s34):            TU_RESULT PASS
  window math: exact delta mean                            PASS
  worked example (16 layers, cold tail {2,5}, L5 leaves cut):
    invariant clean every cycle 0-7                        PASS×8
    no demotion before streak+dwell gates (cycles 0-2)     PASS
    {2,5} cold by cycle 4                                  PASS
    L5 leaves the pool at cycle 5; slot refills with the
    next-coldest eligible layer (0) — pool is a cap         PASS
  property: 12 trials × 200 cycles randomized streams       PASS
    (2400 decisions, 22 demotions, 0 invariant violations,
     pool bound + deep-never-cold held throughout)
patch -p1 --dry-run (fresh shadow of kv_auto_relayout.cpp) PATCH_DRYRUN_OK
```
The worked-example numbers are SYNTHETIC (no recorded `[ft]` lines exist in
`dl/` — checked). Real energies must come from the GPU window.

## 5. What cannot be validated until training/eval finishes
Everything numerical: real energy distributions, the 25%-quantile cut sanity,
cold-cap fill rates, re-layout stall under load, and the quality matrix
(needle @32K/57K: dynamic loop vs static S30 table vs all-hot). The single
command that produces the first real evidence (GPU window, after the patch +
S30 stack are applied and built, engine serves with observation on):
```
NINFER_FT_STATS=1 NINFER_FT_RELOAD_SECS=60 NINFER_FT_COLD_PAGES=8 \
  ./apps/ninfer serve … 2>&1 | tee dl/s34_dry.log
# -> grep '\[ft\]\[cold:dry\]' dl/s34_dry.log   (decisions it WOULD take)
```
Residual compile risk: the patched TU includes ft_stats.h → cuda_runtime.h,
so full-TU syntax check is impossible host-side (same as A's S24 finding);
the new header IS fully compiled+tested host-side (std-only).
