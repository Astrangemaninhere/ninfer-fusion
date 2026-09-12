# C_s23_tu_split.md — S23: gqa_attention_decode per-tier TU split (prepared patch set)

No `src/` writes, no build, no GPU. Everything generated under `_collab/C_s23_tu_split/`.

## Findings that shaped the design
1. **The e8 pattern already exists in production**: `gqa_attention_decode_e8.cu`
   splits the E8Kv tier out via two NON-template overloads
   (`gqa_attention_decode_e8_launch(GqaAppendInput|GqaCachedInput)`) that the
   dispatch calls at runtime. The whole split replicates that proven pattern.
2. **`gqa_attention_decode_impl.cuh` + muse/g35 TUs are DEAD CODE**: not in
   `src/CMakeLists.txt`, no wrapper callers (`gqa_attention_small_t_muse` /
   `_g35` referenced nowhere). The live build = `gqa_attention_decode.cu`
   (3 geometries x 6 tiers in ONE nvcc) + the e8 TU.
3. **The dead impl.cuh had DRIFTED from the live inline code**: its
   `ft::observe` sits in `launch_for` for every tier; the live §111 fix has it
   inside `launch_tc_partial_nvfp4` after the launch. Therefore the generator
   REGENERATES impl.cuh from decode.cu's live segments (anchored extraction),
   so the activated header is provably the production semantics; the drift is
   eliminated, not activated.

## The split (9 files)
* MODIFIED `src/ops/launcher/gqa_attention_decode_impl.cuh` — regenerated from
  the live code; `NINFER_GQA_SMALL_T_DISPATCH` macro (per-site TOKENS/WARPS x
  MultiBatch/Masked x 7 runtime dtype branches) replaced by one runtime
  `cache.dtype` chain calling extern tier launchers. Keepers verbatim: all
  `launch_tc_partial_*` templates (now instantiated only by tier TUs), split
  policy, `single_row_batch_view`, reduce section, ft::observe placement.
  Width>6 guard kept with the identical throw message.
* MODIFIED `src/ops/launcher/gqa_attention_decode.cu` — 712 -> ~110 lines:
  includes impl.cuh; keeps `gqa_attention_uses_small_t`,
  `gqa_attention_split_capacity`, both entry points (still instantiating
  `launch_for` for ALL THREE geometries -> behavior-preserving; routing Muse/35
  through the dead muse/g35 TUs is out of scope).
* NEW `gqa_attention_decode_tiers.h` — extern decls (5 tiers x 2 cache inputs).
* NEW per-tier TUs `gqa_attention_decode_{i8,nvfp4,fp8,iso3,bf16}.cu` — each:
  include impl.cuh; `launch_<tier>_for<Geometry, CacheInput>` with the width
  ladder (and MultiBatch/Masked selection; nvfp4 instead carries the
  writes_cache extraction + `v_dtype==ISO3` branch internally, same as the old
  macro); geometry dispatch (27/Muse/35); the 2 public overloads.
* MODIFIED `src/CMakeLists.txt` — 5 new source lines after `decode_e8.cu`.

## Expected win (instantiation accounting from the macro structure)
Kernel instantiations per geometry: bf16/fp8/iso3 = 6 widths x 4 (MB,Masked) x
2 CacheInputs = 48 each; i8 = 48 launchers x 1-4 schedule variants (~2.3 avg);
nvfp4 = 6 x 2 Iso3V (runtime MB/Masked/CI). Across 3 geometries:
**old decode.cu ~= 810 device-kernel instantiations in ONE nvcc** (25-35 min,
.o lands only at the end => any kill = 100% waste).
After: decode.cu ~= 48 (reduce only, <1 min); i8 TU ~= 330 (~8-12 min, worst);
nvfp4 TU = 36 (heaviest bodies, ~5-8 min); bf16/fp8/iso3 = 144 each (~3-5 min).
A kill now wastes at most the in-flight tier TU; tier-specific edits (e8/i8/
nvfp4 are all our recent edits) rebuild one ~5-10 min TU, not the 30-min one.
`-j2..-j4`: peak ptxas RSS scales with a TU's own instantiation set; the worst
new TU is ~1/3 of the old set, so **-j2 peak < old single-TU peak** (safe);
-j4 plausible combined with `--split-compile-extended=8` — validate with the
`dl/memwatch.log` curve in window H before making it default.

## apply.sh contract (coordinator requirement) + evidence
`apply.sh [--dry-run]` (override repo via `NINFER_REPO`): (1) detects the
applied state -> "already applied", exit 0; (2) pre-state anchors, then
`patch --dry-run` for ALL 9 patches BEFORE any write (abort = nothing written);
(3) one `APPLIED <file>` line per file, non-zero exit on failure; (4)
`--dry-run` writes nothing; (5) touches ONLY the 9 listed source files — never
cmake cache (window H reconfigures with
`-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache .` itself; per coordinator the split and
ccache land together in one full-rebuild window).
Evidence (fresh /tmp/s23_shadow copies of the 3 pristine files):
```
DRYRUN_OK; 9x APPLIED ...; APPLY OK (9 files; ...) EXIT_APPLY=0
re-run: "already applied" EXIT_RERUN=0; .orig/.rej count = 0
mkpatch.py re-run: md5 identical for all 9 patches (DETERMINISTIC)
applied decode.cu: launch_tc_partial references = 0 (all moved)
```
EOL note: repo is mixed (decode.cu LF, impl.cuh/CMakeLists CRLF); the generator
detects per-file EOL and emits matching patch lines (this was caught by the
shadow dry-run failing — not assumed).

## Acceptance for window H (and what needs the GPU)
Trust WITHOUT GPU: patch dry-runs + idempotency (above); after the window-H
build, `nm` partition check — `gqa_attention_decode.cu.o` contains ZERO
i8/nvfp4/bf16/fp8/iso3 partial-kernel symbols, each tier .o contains its own,
each `gqa_attention_decode_*_launch` defined exactly once; link + fresh
`apps/ninfer` mtime; REBUILT gate.
NEEDS the GPU window: `_muse_verify.sh` -> `MUSE_VERIFY_PASS`;
`_e8_postfix.sh` -> `E8_VERDICT=PASS`; qwen 32K needle regression unchanged
(identical kernels relocated — this is the behavior-identity check); tok/s
sanity vs baseline logs; memwatch peak curve to bless -j4.

## Regenerate / review
`python3 _collab/C_s23_tu_split/mkpatch.py` (anchored; any drift = named-anchor
FAIL, never a silent mis-split). Human-readable regenerated files in
`_collab/C_s23_tu_split/review/`.
