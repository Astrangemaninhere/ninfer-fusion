# C_s38_calib_capture.md — S38 (N3 round 2): wire the KvCalibrationCapture side

Prepared patch, NOT applied (window H owns the src batch). No GPU, no build.
Consensus with M applied: (a)+(b) partial-coverage encoding (no v2 header),
stream-async revision with in-code rationale, ENFORCED token bound, ENFORCED
batched-mode refusal.

## 1. The seam (file:line, verified)
`src/targets/qwen3_6/impl/runtime/text_context_impl.h:955-993` — text-layer
attention (inside `...::schedule`): `kn` = post-rmsnorm+RoPE K view
`[head_dim, n_kv, T]` (:957, :966/968), `v` same shape, `cache_positions`
`[T]` I32 (:960-961), `fidx` = full-attention layer index, stream `s`, gated
`ph == Phase::Prefill` (:975). The existing `NINFER_KVDUMP_KV` block
(:973-993) dumps exactly these tensors as "the exact BF16 K/V (post
rmsnorm+rope) and positions the append path consumes" — proof the seam holds
everything the capture needs. ABSENT at the seam: the capture object (state
spans chunks → static singleton) and the batched-mode guard (positions arrive
`[width, batch]` there — a silent capture would produce plausible-looking
misaligned records). CHUNK BOUNDARY: the function runs once per (layer,
prefill-chunk); one call = one `.kvc` record, `record_index_` increments
naturally (same keying the KVDUMP block uses via its static counter).

## 2. The patch (`_collab/C_s38_calib/patch/C_s38_capture.diff`, 2 files)
HUNK A `kv_calibration.h`:
* `capture()` now REQUIRES `cudaStream_t`: `cudaMemcpyAsync` on the producing
  stream + one `cudaStreamSynchronize`. The in-code comment states WHY: a
  plain `cudaMemcpy` is not ordered against a non-NULL stream — a SILENT
  WRONG-DATA bug, not a slow path — so nobody "optimizes" it back.
* ENFORCED bound: cumulative per-layer token cap `NINFER_KV_CALIB_MAX_TOKENS`
  (default 4096); records beyond the cap are skipped, logged once per layer.
  A 57K-prompt runaway is impossible, not merely discouraged.
HUNK B `text_context_impl.h`: includes before the namespace-open (a header
include inside the block would nest its qualified reopens); env-gated
singleton `NINFER_KV_CALIB_DIR` (unset = one cached getenv = zero cost,
mutex-guarded); tap beside the KVDUMP_KV block: `Phase::Prefill` only,
batched-sequence prefill **throws** (`[kvcalib] batched-sequence prefill
cannot be captured (positions are [width, batch])…`), single-sequence calls
`capture(fidx, kn, v, cache_positions, s)`.

## 3. Calibration protocol (bounded, documented here as the run recipe)
Per-record bytes = 64 B header + 4·T + 2·(T·kv_heads·head_dim·2) ≈ 4100·T.
Default protocol: prompts of 2–4K tokens, cap 4096 tokens/layer → with 2K
chunks: 2 records/layer × 16 layers = 32 records ≈ 270 MB total; D2H ≈ 1.7 ms
per 8.4 MB record (pageable) — an offline mode. The file WRITE stays
synchronous `std::ofstream` deliberately: simplicity, per-record ms cost, and
the enforced token cap bounds the total number of blocking writes — the
runaway case is structurally excluded. Calibration runs are
`--no-cuda-graph` (same offline-mode precedent as NINFER_KVDUMP_DIR /
NINFER_FT_STATS; also mandatory because a blocking/ordered copy inside graph
capture is illegal).

## 4. Bake handoff + partial coverage (consensus (a)+(b))
`tools/kv_rowscale_sidecar.py` gained `records` (coverage summary) and `bake`:
reads the `.kvc` frames (same format `tools/calib/analyze_kv.py` parses —
whose docstring's `--kv-calib-dir` flag never existed; this env wiring is the
first real producer), computes per-(layer, kv_head, channel) scales —
**FIRST-CUT RMS-balance, clipped to [0.5, 2.0]; the production table was
Sinkhorn-constrained, so bake output is labeled first-cut** — and writes the
S28 sidecar. PARTIAL RUNS CANNOT BECOME TABLES: bake REFUSES unless every
layer in `[0, expect_layers)` has ≥1 record (prints the missing list), and
the sidecar's `layers` field always equals the observed set, which the S28
loader's identity gate refuses to load against a larger model. Over-claiming
is killed at write time AND load time (the O1-lie class), with no format
change.

## 5. Evidence produced now (host-only)
```
mkpatch -> patch/C_s38_capture.diff (160 diff lines)
patch -p1 --dry-run (fresh shadow of both files): PATCH_DRYRUN_OK
shadow apply: APPLIED; greps: cudaStreamSynchronize=1, MAX_TOKENS refs=2,
  loud batched refusal=1
tool e2e on synthetic frames (C_s38_kvc_frames.py, exact .kvc format):
  records: layers_observed=16 geometry=(256,4) issues=0
  bake full  -> baked.kvrs (32832 B) scale range [0.5000, 1.4609]; validate OK
  bake 3/16  -> REFUSED -- missing: 1,2,3,4,6,7,8,10,11,12,13,14,15; exit 1;
                NO_PARTIAL_FILE_WRITTEN
  corrupt magic -> records: ISSUE 0.kvc: bad magic
```

## 6. GPU-window command sequence (end-to-end, after window H builds the patch)
```
NINFER_KV_CALIB_DIR=dl/kvcalib1 NINFER_KV_CALIB_MAX_TOKENS=4096 \
  ./apps/ninfer serve <artifact> --no-cuda-graph …   # few 2-4K single-seq prompts
python3 tools/kv_rowscale_sidecar.py records --dir dl/kvcalib1
python3 tools/kv_rowscale_sidecar.py bake --dir dl/kvcalib1 \
  --expect-layers 16 --kv-heads 4 --head-dim 256 \
  --model-artifact <artifact> --tag qwen3_8_27b --out <artifact>.kvrs
python3 tools/kv_rowscale_sidecar.py validate <artifact>.kvrs \
  --expect-layers 16 --expect-kv-heads 4 --expect-head-dim 256
NINFER_KV_ROWSCALE=<artifact>.kvrs ./apps/ninfer serve …   # S28 loader applies
```
What still needs the GPU window: real `.kvc` records exist only after a
patched build runs; the first-cut bake QUALITY vs the Sinkhorn production
table (needle 32K regression: baked-sidecar vs baked-constant, expected
comparable, must be measured); the corrupt-sidecar and batched-refusal error
paths in-engine.
