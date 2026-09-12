# S45d — prefill guard v2 (per-arm), based on pristine `b2da4c437ea412902a09d2a378dfafeb`

**Deliverable:** `_collab/E3_s45d_prefill_guard_v2.diff` (4,458 B).
Generator: `_collab/E3_s45d_mkpatch.py` (`--check` re-runs the anchors + the body-preservation proof).
No `src/**` write, no nvcc/ptxas (nothing restarted; `make ninfer -j1` untouched).

| | md5 | sha256 | lines |
|---|---|---|---|
| base = current tree | `b2da4c437ea412902a09d2a378dfafeb` | `4b8e0d6864ea028ec2f5a4c4c1014160ff828a8933cef9e1d02864ff5c25de36` | 453 |
| result | — | `ddf006deb18d94ac9b98f53ca1fb44758804250e843b6ccdce45395c7fb5d5cf` | 484 |

Shape: the three dtype branch conditions stay **exactly** as in pristine, and each arm body carries its
own `if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) { <原体> } else { throw …; }`; the attr group
keeps the no-else form you approved (one guard around the four odr-uses).

```
110     } else if (cache.dtype == DType::NVFP4) {          <- pristine line, unchanged
111-113     comment
114         if constexpr (256) {
115-166         <NVFP4 body: pristine 101-152, byte-identical>
167         } else { 168 throw …; 171 }
172     } else if (cache.dtype == DType::ISO3) {             <- pristine line, unchanged
173         if constexpr (256) { <ISO3 body: pristine 154-174> } else { 196 throw …; 199 }
200     } else if (cache.dtype == DType::FP8_E4M3FN) {       <- pristine line, unchanged
201         if constexpr (256) { <FP8 body: pristine 176-196> } else { 224 throw …; 227 }
228     } else {  <bf16 body, unguarded> }
```

## (a) `patch -p1 --dry-run` against `/home/user/ninfer-fusion` (verbatim)

```
checking file src/ops/launcher/gqa_attention_prefill.cu
rc=0
```

Shadow check: applying the diff to a pristine copy reproduces the intended bytes exactly
(`SHADOW_APPLY_OK`).

## (b) 256-equivalence: every guarded body line is unchanged (line-level diff, no claims)

Whole-file accounting (`diff pristine patched`): **`^<` = 0, `^>` = 31** — nothing removed, nothing
modified, 31 lines added (10 for the attr group, 3×7 for the arms: 3 comment lines + 1 opener + 5 closer).
The three arms were extracted with `sed` from both files and diffed:

```
$ diff <(sed -n '101,152p' src/ops/launcher/gqa_attention_prefill.cu) <(sed -n '115,166p' /tmp/e3s45d/out/gqa_attention_prefill.cu) && echo NVFP4_BODY_IDENTICAL
NVFP4_BODY_IDENTICAL
$ diff <(sed -n '154,174p' src/ops/launcher/gqa_attention_prefill.cu) <(sed -n '174,194p' /tmp/e3s45d/out/gqa_attention_prefill.cu) && echo ISO3_BODY_IDENTICAL
ISO3_BODY_IDENTICAL
$ diff <(sed -n '176,196p' src/ops/launcher/gqa_attention_prefill.cu) <(sed -n '202,222p' /tmp/e3s45d/out/gqa_attention_prefill.cu) && echo FP8_BODY_IDENTICAL
FP8_BODY_IDENTICAL
```

(`sed` diffs print nothing when identical; the generator prints the same result as
`NVFP4 IDENTICAL (52 lines at patched 115..166) / ISO3 IDENTICAL (21 lines at 174..194) / FP8 IDENTICAL
(21 lines at 202..222)` and `existing lines modified: 0`.)

So at head_dim 256 the only difference versus pristine is an extra `{ … }` nesting level per arm: same
statements, same order, same arguments, same kernels bound — no indentation changes at all (you asked for
"不重排缩进", so the bodies keep their pristine indentation).

## (c) Regression self-check: 6 dtypes × 2 head_dims, with the line each cell rests on

Lines are **patched-file** line numbers unless marked pristine.

| dtype | head_dim 256 | head_dim 128 | 依据 |
|---|---|---|---|
| **NVFP4** | enters arm 110 → guard 114 true → pristine body → `…<NVFP4, ISO3>` at 133 when `cache.v_dtype == ISO3` (132), else `…<NVFP4>` at 150 | arm 110 → guard 114 false → **throw 168-170** | arm 110, guard 114, v_dtype dispatch 132-133, launches 133/150, else-throw 167-171 |
| **ISO3** | arm 172 → guard 173 true → body → `…<ISO3>` at 178 | arm 172 → guard 173 false → **throw 196-198** | arm 172, guard 173, launch 178, else 195-199 |
| **FP8_E4M3FN** | arm 200 → guard 201 true → body → `…<FP8_E4M3FN>` at 206 | arm 200 → guard 201 false → **throw 224-226** | arm 200, guard 201, launch 206, else 223-227 |
| **BF16** | bf16 arm 228 → `gqa_attention_prefill_bf16_kernel` 231 | **same** (arm 228 is outside every guard) | arm 228, launch 231 |
| **I8** | i8 arm 85 → `gqa_attention_prefill_i8_kernel` 100 | **same as 256** — S45d does not gate this arm (see note) | arm 85, launch 100; `src/ops/launcher/gqa_attention_prefill.cu:75` pristine |
| **E8Kv** | same as I8; the e8 launchers branch inside the arm (90-91, 249-250) | **same as I8** | arm 85/248, e8 dispatch 90-91 |
| any other dtype | falls to bf16 arm 228 (pristine behaviour) | same | arm 228 |
| *attr group (all dtypes)* | four `cudaFuncSetAttribute` odr-uses 55/60/64/69 execute | guard 58 false → **skipped, no throw** (deliberate: the block is unconditional, bf16/i8 execute it too) | guard 58, attrs 55/60/64/69, closer 76-83 |

Notes on the two cells that could be misread:

* **I8/E8Kv at 128 are not gated by this diff.** They were not gated in pristine either, so S45d is
  behaviour-neutral there; `E3_s45_i8_plane_stride.diff` (S45) adds the loud gate for that arm. The two
  patches are line-disjoint and apply in either order.
* **ISO3/FP8 at 128 throw** with the same message as the NVFP4 arm; there is no arm in the packed group
  that can be reached at 128 without either throwing or being the bf16 arm.

## Why ISO3/FP8 need their own guards (unchanged argument, now per-arm)

`gqa_attention_prefill_nvfp4_kernel` is one template instantiated four times from this launcher
(`<…, NVFP4>`, `<…, NVFP4, ISO3>`, `<…, ISO3>`, `<…, FP8_E4M3FN>`). In `gqa_attention_prefill_nvfp4.cuh`:

* `:1024` `constexpr bool Mxf4QK  = KVDType == DType::NVFP4;`
* `:1025` `constexpr int  Mxf4QKKs = D / 64;` → **2** at D=128 (4 at 256)
* `:1026` `static_assert(!Mxf4QK || Mxf4QKKs == 4);` → hard compile error for the NVFP4 arms;
  `:1014-1017` (`Threads == 256`, `ProducerThreads == 128`, `QKNt == 4`, `PVKs == 2`) and
  `:1004-1008` (`Bc = 32`, `QKKs = D/16`) are the rest of the 256-shaped tile contract.

The ISO3/FP8 arms might instantiate, but the kernel body is 256-shaped (smem sizing
`kNvfp4PrefillSmemBytes`, `D`-derived tile math), i.e. at 128 they would run 256-wide tile assumptions
against 128-wide rows — silent numerical corruption of the same class S45 fixed for i8/E8. Hence all
three arms refuse loudly.

## One factual note on the S45c artifact (does not change the v2 delivery)

For the record, since it drove the revert: the S45c artifact as generated did **not** drop the ISO3/FP8
arms — it chained them (artifact lines `118 if (cache.dtype == DType::NVFP4) {`, `171 } else if (cache.dtype
== DType::ISO3) {`, `193 } else if (cache.dtype == DType::FP8_E4M3FN) {`, `215 }` closing the chain, throw
at `216-220` for the `if constexpr` else). So in that artifact an ISO3/FP8 cache at 256 reached its own
body; the silent-skip you described needs the `} else if (… ISO3) {` / `… FP8_E4M3FN) {` separators to be
absent, which they are not. It is easy to see how the *diff* reads that way, though: those two separators
never appear inside a hunk, so a diff-only reading sees just the new `if (… NVFP4) {` with no else-if —
which is exactly the ambiguity the per-arm v2 shape removes. If you have a trace from an applied S45c
build that skipped, send it and I will chase it; I am switching to v2 regardless because for a
critical-path file each arm's guard should be visible and local, and an arm must be unable to fall through
silently even if someone later reorders the chain.
