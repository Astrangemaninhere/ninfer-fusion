# S36 — Muse decode corruption: unified hardcoded-256 head-dim family (audit + proven fix patch)

Artifacts: `_collab/A_s36_headdim.diff` (13 files, 94 audited site edits, 669 diff
lines, `patch -p1 --dry-run` OK) + generator `A_s36_mkpatch.py` + site dump
`A_s36_sites.txt` (via `A_s36_sites.sh`) + host numeric proof `A_s36_h1_proof.cpp`
(`H1_PROOF_OK`, both directions). NOT APPLIED — next window (H's batch already built).

## 1. Round arc (hypotheses, each with its disposition)

- **H4 (foreign row-scale table on Muse L0-15): REFUTED by arithmetic.** Baked table
  values are bf16 raw 16128-16384 = [0.5, 2.0] (`gqa_isoquant_row_scale.cu`), and
  write `×s` (`decode_nvfp4.cuh:323`, `prefill_nvfp4.cuh:531`) mirrors read `×1/s`
  (`decode_nvfp4.cuh:477`, `prefill_nvfp4.cuh:1105`) → max 2× distortion, never 1e6.
- **H3 (second unguarded LUT): refuted.** Rot table `[64][4][4]` indexed d/4 ≤ 31 for
  Muse; codecs stateless. The "out-of-range table" turned out to be the plane layout
  itself (below).
- **H2 (ISO3-V under KVHeads==2): alive but bounded (~12× nibble misread)** — cannot
  alone explain 1e6; subsumed by (iv) since V codes sit in the same misindexed layout.
- **H1 (nvfp4 256-lead vs Muse 128): CONFIRMED numerically** (proof below) — but after
  the bf16 pivot, it is one INSTANCE of the family, not the whole story.
- **Pivot (bf16 fails too, intermittently, decode-only)**: (i) SWA refuted for bf16 —
  `gqa_attention_decode_bf16.cuh` contains zero sliding-window references (read span
  `window = last_pos + 1`, :118-124); (ii) the `tokens>=128 && KVHeads==2` gates are
  prefill-only (`prefill.cu:230`, `prefill_e8.cu:50`); (iii) bookkeeping demoted.
- **(iv) UNIFIED ROOT CAUSE (the answer)**: the shared DECODE-side index helpers
  hardcode 256 — `gqa_attention_decode.cuh:22` `kGqaHeadDim = 256` feeding
  `gqa_cache_index` (:37, **the bf16/plain plane index**), `gqa_q_index`/`gqa_kv_new_index`/
  `gqa_partial_acc_index` (:43/:51/:60), reducer row/loops (:176/:208/:235), every
  per-dtype decode kernel's D and input strides, AND the nvfp4 family
  (`gqa_attention_kv_nvfp4.cuh:28-32` + `Groups` sites). The layout side is
  head_dim-correct (`decoder_state.cpp` planes), and the geometry template has carried
  `HeadDim` (128|256 assert) all along — the helpers just never used it. Explains BOTH
  dtypes (bf16 via `gqa_cache_index`, nvfp4 via the CodeLead/ScaleLead/src trio), the
  decode-only-ness (prefill is self-consistent at wrong offsets for its own attention
  read — its misdirected writes land in neighbour planes), and the intermittency
  (head-1's out-of-row writes clobber the same layer's V/K-scale planes at seams;
  which bytes survive depends on warp/block execution order → pass-to-pass variation,
  occasional "recovery", 2e8 blowups).

## 2. Numeric proof (host, no GPU) — `A_s36_h1_proof.cpp`

Replicates `paged_kv_element_offset<Lead,Heads>` (:20-27) and the layout-side
allocation exactly. Key printed results:

```
Muse(128,2): K-code plane row 8192 B/page; OLD writes 16384 B/page -> 8192 B/page land in the NEIGHBOUR plane
             head1 writes [8192,16384) = OUT OF PLANE ROW; NEW = exact bijection (8192 distinct bytes, each once)
             src head1 token0: OLD reads elements [256..512) of a 256-element K input -> ENTIRELY OUT OF BOUNDS
qwen(256,4): OLD==NEW over 110592 offset checks (codes+scales) + 65536-element rows exact
bf16 family: Muse row 16384 elem/page vs OLD 32768 written (2x + head1 out); qwen 65536 == 65536 exact
H1_PROOF_OK (0)
```

## 3. Family closure — all 105 audited sites classified (`A_s36_sites.txt`)

| category | count | disposition |
|---|---|---|
| (a) stride/shape/offset in Geometry-templated code | 94 | **patched**: `kGqaHeadDim`/`kGqaPrefillHeadDim`/`kGqaKvNvfp4HeadDim` → `Geometry::HeadDim`; `kGqaKvNvfp4CodeLead` → `(Geometry::HeadDim / 2)`; `ScaleLead`/`kGqaKvNvfp4Groups` → `(Geometry::HeadDim / kGqaKvNvfp4Group)` — across decode.cuh + 5 per-dtype decode kernels + kv_nvfp4.cuh + prefill bf16/nvfp4/common + launchers (decode.cu/impl/e8, prefill.cu incl. the reducer grid `div_up(kGqaHeadDim, kDChunk)` whose extra blocks read past 128-dim rows) |
| (b) definitions / file-scope smem upper bounds | 11 | kept 256 + warning comment ("geometry-generic code MUST use Geometry::HeadDim"): decode.cuh:22, kv_nvfp4.cuh:28-32, prefill_common.cuh:19/:25/:36-37 (smem sized for the 256 max — wasteful-not-wrong at 128) |
| (a-blocked) `gqa_attention_prefill_i8.cuh` | whole file | 256-only BY CONSTRUCTION (`static_assert(kGqaPrefillI8Groups == 4)` :49, swizzle tables) — geometry-izing it is its own work package; untouched ⇒ Muse+int8-KV prefill is no worse than today (still broken, now *documented*) |

Note: the coordinator's grep counted 81 for two identifiers; the dump covers 105
(both + the nvfp4 trio + Groups). Window H's batch landed mid-round (decode.cu
600+→112 lines); sites were re-dumped post-batch and the drift check
(`A_s36_drift_check.sh`) is clean — the patch targets the current tree.

## 4. Prefill consumers (requirement 2, precisely)

Pre-patch, prefill's misdirected writes are observable by: (1) decode kernels (the
observed corruption — same patch fixes both sides TOGETHER, so there is no
fixed-read/unfixed-write mismatch); (2) the cold-slot entropy codecs (compress
whatever bytes sit in the plane — unexercised on Muse today, correct post-patch);
(3) `NINFER_KVDUMP_DIR` forensics (misleading pre-patch, byte-true post-patch).
"Self-consistent" applied only to prefill's own attention read, and the patch removes
the misplacement everywhere rather than arguing it away.

## 5. Acceptance (stated up front, both directions)

- **Host (done)**: `H1_PROOF_OK` — for EVERY touched helper family, Muse 128 lands a
  bijection inside the allocated rows and reads in-bounds sources; qwen 256 is
  BYTE-IDENTICAL old-vs-new (compile-time-equal expressions, asserted numerically
  110592+27648 checks) — the 256 path cannot change behaviour.
- **Post-build (next window)**: (a) Muse **bf16** `--max-new 4` → **zero NaN lines**
  in headdbg (today 366) and sane text; (b) KVDUMP byte check: the bytes past each
  plane row boundary (previously the clobber landing zone) must be untouched/
  zero-filled — this TESTS the race explanation rather than assuming it; (c) qwen
  8/8 needles unchanged (regression gate); (d) falsifiable prediction on the UNFIXED
  binary if still available: failures do NOT track absolute position (`--max-new 1`
  vs `4`, prompt shifted by one token) — only race order.
- Commands: `wsl.exe -e bash -c "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_s36_mkpatch.py"`
  (regenerate + dry-run), `... g++ -std=c++20 -O1 _collab/A_s36_h1_proof.cpp -o /tmp/h1_proof && /tmp/h1_proof`;
  GPU window: `NINFER_HEADDBG=1 NINFER_KVDUMP_DIR=/tmp/kvd <engine> <muse.ninfer> --prompt '1, 2, 3, 4, 5, 6,' --max-new 4 --kv-dtype bf16 --max-context 4096` then
  byte-histogram the K-scale/V planes past row boundaries (`python3` over the dump:
  bytes ≥ 0x40 in a scale plane that should hold ~0.1-1 E4M3 = residual clobber).

## 6. What this does NOT do

- prefill_i8 geometry-ization (a-blocked above); Muse+int8 stays unsupported.
- No rebuild/GPU run (window G/H own those); the diff is dry-run-proven only —
  TU compilation happens in the next window (ccache makes it cheap).
- Does not touch the SWA semantic gap (S24) or cold/host lines.
