# C_s28_rowscale_sidecar.md — S28 (N3 round 1): row-scale sidecar + env-gated loader

No `src/` writes (patch prepared, NOT applied), no GPU, no build. Facts from the
coordinator sanity-checked: `kGqaKvRowScaleDev` is `extern __constant__` in
`src/ops/kernel/gqa_isoquant_row_scale.cuh:18` (definition: `.cu:5`), accessor
`gqa_kv_row_scale()` (cuh:29) returns identity ONLY outside
`[0,16)×[0,4)×[0,256)` ⇒ **Muse's layers 0–15 DO read the qwen-baked table**
(confirmed in source). `kv_calibration_dir` grep: no consumers (dead, as stated).

## Sidecar format (LE, 64B header; spec also in the tool docstring)
```
 0 16 magic "NINFERKVRS1" NUL-pad     32 4 flags   bit0 = identity table
16  4 u32 version=1                    36 4 crc32  zlib-style over payload
20  4 u32 layers (table extent)        40 8 u64 model_hash sha256(artifact)[:8]
24  4 u32 kv_heads                     48 16 tag[16] NUL-pad ("qwen3_8_27b")
28  4 u32 head_dim                     64 .. payload u16[layers*kv*dim] BF16 bits
```
Identity rules (the "which model owns this table" gate):
* non-identity table: layers/kv_heads/head_dim must EQUAL the model's — the
  16-layer-table-under-52-layer-model case (today's silent Muse hazard) is
  REFUSED; payload must exactly fill the 16384-word `__constant__` symbol.
* identity table: all words verified BF16 1.0; layers may be ≤ model's.
* model_hash enforced when both sides non-zero (0 = unspecified).
Loader hard-fails (throws, field named) — no silent fallback.

## Deliverables
* `ninfer-fusion-repo/tools/kv_rowscale_sidecar.py` — dump (parse the baked
  initializer) / write / identity / validate.
* `_collab/C_s28_rowscale/` — `review/gqa_isoquant_row_scale_loader.h`
  (host-only parse+check), `review/gqa_isoquant_row_scale_loader.cu`
  (`NINFER_KV_ROWSCALE` env-gated apply: unset=off, else read→validate→
  cudaMemcpyToSymbol→`[kvrs] applied` forensics line, throw on any failure),
  `C_s28_mkpatch.py` → `patch/C_s28_loader.patch` (2 new files + ONE anchored
  hunk in `decoder_state.cpp::plan_decoder_state` where
  spec.full_attention_layers/kv_heads/attention_head_dim are visible — NOT in
  A's N1/S24/S25 file set), `C_s28_host_test.cpp`.

## Acceptance evidence (host-only, /tmp/kvrs_test, raw)
```
dump: 16384 words -> payload.bin; value range [0.5000, 2.0000], mean 1.0822
write: good.kvrs (32832 bytes, 16384 words, tag='qwen3_8_27b')
validate good.kvrs (16,4,256): OK  crc=45795edc
validate good.kvrs --expect-layers 52: FAIL identity.layers: table 16 != model 52
  (foreign-model table refused)          <- TOOL_REFUSED_52_LAYER=OK
host test (g++ -std=c++20, 12/12):
  roundtrip parse / table bytes == baked dump (byte compare) / qwen identity  PASS
  corrupt magic/version/payload-byte/truncated -> error NAMES the field       PASS×4
  16-layer table under 52-layer model REFUSED (upload never reached)          PASS
  model_hash mismatch refused (57005 != 48879)                                PASS
  identity sidecar accepted under 52-layer model, payload all BF16 1.0        PASS×2
  kv_heads / head_dim mismatch refused                                        PASS×2
  TU_RESULT PASS                                                              HOST_EXIT=0
patch -p1 --dry-run (fresh shadow of decoder_state.cpp): PATCH_DRYRUN_OK
```
Roundtrip = the byte-compare proof the brief demanded (dump→sidecar→parse ==
baked constants), so loading it cannot change qwen numerics by construction.

## What still needs the GPU window (explicit)
1. qwen 32K needle regression UNCHANGED with the table loaded from the sidecar
   (`NINFER_KV_ROWSCALE=good.kvrs` + `--kv-layer-storage`-style run): expected
   bit-identical (same table bytes), must be measured, not assumed.
2. Muse 32K/57K needle: calibrated-vs-identity comparison is round-2 QUALITY
   work (Muse was never calibrated); round 1 only makes the current silent
   foreign-table state loud (refused) instead of quiet.
3. Engine-side: the patch must be applied+compiled in a build window (with
   A's batch), then one serve run with a corrupt sidecar must show the
   readable hard error, and with a good sidecar the `[kvrs] applied` line.
Round 2 hooks (deliberately out of scope): model-hash wiring (needs the
options plumbing A owns), `--kv-calibrate off|auto|force` + `--recalibrate`
via the existing (dead) `KvCalibrationCapture` + `kv_calibration_dir`.
