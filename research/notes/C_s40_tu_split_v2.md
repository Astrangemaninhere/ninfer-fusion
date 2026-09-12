# C_s40_tu_split_v2.md — S40: decode TU split, re-derived against the **build tree**

No `src/` writes, no build, no GPU, no nvcc. Everything below is generated/measured under
`_collab/C_s40_tu_split_v2/`. The canonical build tree `/home/user/ninfer-fusion` was read
but never written (the only `apply.sh` invocation against it is `--dry-run`).

## 0. What went wrong in S23, in one line

S23's `apply.sh` did `REPO="${NINFER_REPO:-<mirror>}"` — a *default* pointing at
`/mnt/c/.../ninfer-fusion-repo`. Its own output says it: `dry-run: 9/9 patches apply cleanly
against /mnt/c/.../ninfer-fusion-repo`. So the split landed in the tree nobody builds, and the
build tree still has the unsplit 737-line TU. **S23's patch set is superseded by this one.**

That is not just a path bug — the mirror's landed split is broken (§3). Nothing from the mirror
is reused here; every byte below is re-derived from the live build tree.

## 1. apply.sh (new contract)

```
bash apply.sh [--dry-run] <repo-root>      # repo-root is REQUIRED, there is no default
```

| guarantee | how |
|---|---|
| target is an explicit argument | `REPO_ARG="$1"`; `usage` if `$# -ne 1`. `grep -rn '/mnt/\|/home/\|ziqinzhang' apply.sh patch/` → **0 hits** |
| no writes outside the root | every hunk header is scanned first: a path that is not one of the 9 declared `TARGETS`, or that contains `/` traversal, is `REFUSED` (exit 2) **before** anything is written; `patch` then runs with `-d "$REPO"` and repo-relative names only |
| idempotent | post-marker present, pre-marker absent and all 6 new files present → `already applied`, exit 0 |
| full dry-run pre-pass | all 9 patches get `patch --dry-run` before the first write; any failure aborts with `no files were modified` |
| one line per file | `APPLIED <repo-relative file>` ×9; non-zero exit on failure |
| `--dry-run` writes nothing | proven by digest comparison of the whole shadow tree (§4.1) |
| per-file EOL preserved | `mkpatch.py` detects each target's own EOL and emits matching patch lines (CRLF for `impl.cuh`/`CMakeLists.txt`, LF for `decode.cu` and the new TUs) |
| partial state refused | `REFUSED: ... partial state (n/6 tier files present)` |

## 2. The split, re-derived from the current build tree

Live pre-state (sha256 / lines / EOL): `gqa_attention_decode.cu` `6049a50d…` 737 LF ·
`gqa_attention_decode_impl.cuh` `63892897…` 618 CRLF · `src/CMakeLists.txt` `2a1fe640…` 393 CRLF.

`mkpatch.py <repo-root>` extracts by **unique content anchor** (never by line number) and prints
the resolved ranges — drift becomes a named FAIL, not a silent mis-split:

```
i_guard 21..32   i_e8 34..49   i_anon 50..475   i_launch_for 500..510   i_reduce 609..659
i_uses_small_t 477   i_split_capacity 479..498   i_small_t_launch 662..703
i_cached_small_t_launch 705..735   i_macro 514..607
parsed ladders = tile {1:1,2:2,3:3,4:4,5:5,6:6} warps {1:2,2:4,3:4,4:4,5:4,6:4}
parsed chain   = ['I8','E8Kv','NVFP4','NVFP4','FP8_E4M3FN','ISO3']
```

The width→(TokenTile,WarpsPerCta) ladder and the dtype chain are **parsed out of the live
dispatch macro**, not transcribed, so the emitted per-tier ladders cannot disagree with it.

| file | lines | sha256 |
|---|---|---|
| `src/ops/launcher/gqa_attention_decode_impl.cuh` (MODIFY, CRLF) | 591 | `c0a302fda88f7409ee72603b5854a9eb5529c5e2a48f5d1a26d501d2ec5dff58` |
| `src/ops/launcher/gqa_attention_decode.cu` (MODIFY, LF) | 113 | `8eebd2073b11b202e6246eed425268a8bc3f2893546159a1543ef107f3d0fc91` |
| `src/CMakeLists.txt` (MODIFY, CRLF, +5/−0) | 398 | `88426a5e8847fa1bb033b517efffc1145c5190dd7d1394c99394e6752c067d8d` |
| `src/ops/launcher/gqa_attention_decode_tiers.h` (NEW, LF) | 79 | `6e228916e82ca2543b6f1f0cc13bb2c5a3adfd13ab2dcdb18885ce4c020108c0` |
| `src/ops/launcher/gqa_attention_decode_i8.cu` (NEW, LF) | 144 | `3be4a8fbdd2f31751a799d8d518c4fc0e0d12839611a3f9270234a2ac884524a` |
| `src/ops/launcher/gqa_attention_decode_nvfp4.cu` (NEW, LF) | 166 | `418e82445a471927931aee07bf069803394d04c169c5aabebb609efcc22f446b` |
| `src/ops/launcher/gqa_attention_decode_fp8.cu` (NEW, LF) | 138 | `f533a362ff415ba79b7e3c7a6c056d93884e688218fcd7d150a3e6b8665c81f8` |
| `src/ops/launcher/gqa_attention_decode_iso3.cu` (NEW, LF) | 138 | `0709f0d445a1c15afad7877aaff41877398be659182a98dce2df600e882ef50b` |
| `src/ops/launcher/gqa_attention_decode_bf16.cu` (NEW, LF) | 138 | `9f2f82e839587a76a8a2cec5557cf2444ba23dd20b34c32982bc6904383b69ad` |

Shape: each tier TU holds `launch_<tier>_ladder<Geometry,CacheInput,[MB,Masked]>` (live's
per-width switch, verbatim argument lists) + `launch_<tier>_for` (live's 4-way
`batch_size == 1` / `valid_columns` selection) + the two public overloads (live's
27→Muse→35 geometry dispatch). `impl.cuh` keeps every `launch_tc_partial_*` template, the split
policy, `single_row_batch_view`, the `require_nvfp4_geometry_dim` thrower and the reduce
section, and its `launch_for` now dispatches on `cache.dtype` to the extern tier launchers.

**The nvfp4 head-dim guard is carried over.** Live guards each of its two nvfp4 branches with
`if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) … else require_nvfp4_geometry_dim(...)`.
The split puts one such guard in `launch_nvfp4_tile`, so for Muse (HeadDim 128) the kernel is
**not instantiated at all** and the call throws with the live message, byte-identical.

## 3. Does the mirror's regenerated `impl.cuh` (576 L) carry semantic changes? — **Yes. It carries a build breakage.**

Itemised, against the live build tree (not against the old dead header). `verify.py` V4e
asserts the counts.

**Δ1 — the nvfp4 head-dim guard is gone (only real semantic delta; it is fatal).**
`require_nvfp4_geometry_dim` occurs **0 times in the entire mirror `src/`** and 3 times in the
build tree (1 definition + 2 call sites). The mirror's `gqa_attention_decode_nvfp4.cu`
instantiates `launch_nvfp4_for<GqaMuseGeometry>` from a **non-template** function
(`gqa_attention_decode_nvfp4_launch(const Tensor&, const GqaCachedInput&, …)`, mirror line 147
called from line 132). The only conditionals on that path are
`if constexpr (CacheInput::writes_cache)` (nulls two pointers) and a runtime `if (iso3_v)` —
**both arms** call `launch_tc_partial_nvfp4<…>`. So the instantiation is unconditional:
`GqaMuseGeometry = GqaGeometry<32,2,1,128>` → `D = HeadDim = 128` → `QKKs = D/64 = 2` →
`static_assert(QKKs == 4)` (`gqa_attention_decode_nvfp4.cuh:167`) fires.
Compile-arithmetic proof (`evidence/static_assert_proof.cpp`, a reduction of the kernel's own
constant expressions): with the Muse instantiation in place g++ reports
`error: static assertion failed … the comparison reduces to '(2 == 4)'` (exit 1); with it
guarded out (the live shape) exit 0. **The mirror's nvfp4 TU cannot compile.**
This is also *why* it was never noticed: it lives in a tree with no build.
Had it compiled, Muse would have run the nvfp4 tiled kernel against a hardcoded 256 — the
_TODO 126/127 out-of-bounds K-row write.

**Δ2 — the guard also protected a second thing the mirror loses:** with no `if constexpr`, the
mirror instantiates the nvfp4 kernel for a geometry whose tiling does not exist; there is no
runtime refusal either, only the (unreachable, because it would not compile) launch.

**Δ3 — width-range check rewritten** (relocation-adjacent, behaviour-preserving). Live:
`switch (invocation.width) { case 1..6 …; default: throw std::invalid_argument("…unsupported T"); }`.
Mirror: `if (width < 1 || width > 6) throw <same message>`. Same accept set, same message.

**Δ4 — geometry re-dispatch in the *cached* fallback throws where live fell through**
(input-validation only; unreachable on valid input). Live's `gqa_attention_cached_small_t_launch`
routes 27 → Muse → **unchecked** 35 fallback; the mirror's cached tier overloads add
`if (q.ne[1] != Gqa35Geometry::QHeads || q.ne[0] != Gqa35Geometry::HeadDim) throw`. Only
differs for dims the entry point's own contract already excludes.

**Not deltas (verified equal to live, i.e. the mirror's stated "regenerated from live" claim
holds on these):** the `ft::observe` placement (post-launch inside `launch_tc_partial_nvfp4`,
matching live; the *old* dead header had it in `launch_for`), the i8 dynamic-smem constant
(`4 * KeyBlock * kGqaHeadDim`, matching live), the reduce section and its `I8|NVFP4` gate, the
`single_row_batch_view` field list, the e8 handoff.

So: the mirror is a *correct regeneration of the pre-coordinator-edit* `decode.cu`. 737 − 712 =
25 lines is exactly the guard the coordinator added after S23 was generated — and the mirror is
missing precisely those 25 lines. **An unreviewed mirror rewrite must not land** (§5).

## 4. Evidence (`evidence/`)

### 4.1 `shadow_lifecycle.txt` — lifecycle + containment
```
dry-run: 9/9 patches apply cleanly against /tmp/s40_shadow/repo     dryrun_exit=0
DRYRUN_WROTE_NOTHING  (whole-tree sha256 digest unchanged)
APPLIED <9 files>                                                   apply_exit=0
APPLY OK (9 files; land in a dedicated window -- not the in-flight build)
diff -rq (whole src/, pristine vs applied): changed-file count = 9
CHANGED_SET_EXACT_MATCH (9/9, no extra, no missing)
.orig/.rej in pristine copy = 8, in applied copy = 8 → NO_NEW_BACKUPS
buildtree_dryrun_exit=0      (--dry-run against the real build tree)
already applied              rerun_exit=0
applied bytes == reviewed bytes: SAME for all 9
nl=113 LF decode.cu | 591 CRLF impl.cuh | 398 CRLF CMakeLists.txt | 79 LF tiers.h
nl=144 LF i8 | 166 LF nvfp4 | 138 LF fp8 | 138 LF iso3 | 138 LF bf16
negative controls: bad root → REFUSED exit 2; dir without src/ → REFUSED exit 2;
no argument → usage exit 1; tampered patch naming an undeclared file → REFUSED exit 2,
TAMPER_WROTE_NOTHING; path traversal `../../escape.cu` → REFUSED exit 2.
```
The 8 `.orig`/`.rej` files are **pre-existing** build-tree content copied by `cp -a`; the
applier created none.

### 4.2 `verify_output.txt` — is it a pure relocation? **33/33 PASS, `VERIFY_OK`**
The load-bearing one is V2: the multiset of `launch_tc_partial_*` instantiations is expanded
**independently** from the live macro and from the generated tier TUs and must be equal —
same tier, TokenTile, WarpsPerCta, Iso3V, MultiBatch, Masked **and the same argument list**:
```
live instantiations = 144   generated instantiations = 144   (V2 identical)
V2b  i8 24 nvfp4 48 fp8 24 iso3 24 bf16 24
V3   the 4-way (MultiBatch,Masked) set is identical per tier; nvfp4 keeps both Iso3V arms
V4a–e guard present; the ONLY nvfp4 kernel instantiation sits inside it; message
      byte-identical; thrower is [[noreturn]] inline; no guard site dropped
V5a–d dtype chain routes I8/E8Kv/NVFP4/FP8/ISO3 + bf16 fallback; ISO3-v re-keyed in the
      nvfp4 tier; e8 handoff unchanged
V6a–e decode.cu instantiates no tier kernel; impl.cuh defines 5 tiers and calls none;
      reduce kernel + its I8|NVFP4 gate intact; i8 smem constant byte-identical to live
V7   width guard message byte-identical to the live default arm
V8   CMakeLists +5/−0, tier TUs only
```
The 144 dispatch-site specs expand to **768 tier-launcher kernel instantiations** across
3 geometries × 2 `CacheInput`s (nvfp4's Muse third is discarded by the guard) — all 768 lived in
one nvcc invocation before. What stays in the split-off `decode.cu` is host dispatch plus the
reduce kernel.

### 4.3 `mkpatch_output.txt` / `mkpatch_check.txt`
Anchor ranges as above; per-file lines/sha256/EOL. `--check` → `CHECK_OK: patch/ and review/
are byte-identical to a fresh generation`; `path_audit.txt` §E → `DETERMINISTIC`.

### 4.4 `path_audit.txt`
`/mnt/`, `/home/`, `C:\`, `ziqinzhang` in `apply.sh` + `patch/` → **0 / 0 / 0 / 0**. Every path a
patch can write is listed and equals the 9 `TARGETS`; 9 patches ↔ 9 targets; each patch is a
1-file unified diff. `review/impl.cuh` still defines `gqa_attention_small_t_launch_for` and
`single_row_batch_view`, so the currently-unbuilt `gqa_attention_decode_{muse,g35}.cu` keep
compiling against it.

### 4.5 `mirror_delta.txt`
The §3 itemisation, mechanically: guard occurrences 0 vs 3; the mirror's unconditional
`launch_nvfp4_for<GqaMuseGeometry>` sites with the path's complete conditional list; the
geometry constants (`GqaMuseGeometry = GqaGeometry<32,2,1,128>`, `QKKs = D/64`,
`static_assert(QKKs == 4)`); the g++ proof (exit 1 vs exit 0); and the per-file line counts
mirror-vs-S40 (`impl.cuh` 576/591, `nvfp4.cu` 161/166, `i8.cu` 128/144, `decode.cu` 112/113,
`tiers.h` 81/79).

### 4.6 `dryrun_build_tree.txt`
```
$ bash apply.sh --dry-run /home/user/ninfer-fusion
dry-run: 9/9 patches apply cleanly against /home/user/ninfer-fusion
DRYRUN OK (no files were modified)
exit=0
```

### 4.7 `integrity.txt` — cross-tree safety
```
UNCHANGED decode.cu (737 L)   UNCHANGED impl.cuh (618 L)   UNCHANGED CMakeLists (393 L)
tier files present in the build tree (expect 0): 0
'ninfer-fusion-repo' in apply.sh/patch/review/mkpatch.py/verify.py: 0
```
The build tree's three inputs still hash to their pre-state values, so nothing in this round
wrote to it. The only file in the bundle that names the mirror is `evidence/mirror_delta.*`,
which exists precisely to audit it.

### 4.8 Two incidental findings, recorded not fixed
- The build tree's live `decode.cu` still sizes the i8 dynamic arena with
  `4 * KeyBlock * kGqaHeadDim` while the S36-fixed `gqa_attention_decode_e8.cu` uses
  `Geometry::HeadDim` for the identical construct. Relocated **verbatim** here (V6e) — a split
  must not smuggle an S36 extension. It is inert for Muse anyway: the Muse i8 ladder passes
  `DynamicArena=false`, so the constant is multiplied by 0, and for 27/35 `HeadDim == 256 ==
  kGqaHeadDim`. Flagging it as an S36 follow-up site, not changing it.
- `gqa_attention_decode.cu` carries a duplicated 2-line "Revision 2b: INT8-tier cold slots"
  comment block (live lines 241-242 == 283-284). Left verbatim.

## 5. Scope — this split is **excluded from the in-flight build**

The running window-H build compiles the **original 737-line `gqa_attention_decode.cu` at
`-j1`**; this split is not in it and must not be dropped into it. Two reasons, both concrete:
the machine has rebooted twice under memory pressure, and no new nvcc invocation should be
added to a window whose whole point is to get a link out. The split **lands as its own window
later** (that is also when `patch` should be followed by a `-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache`
reconfigure, since a split multiplies TU count).

Note for the board: line 479 currently lists `s23` among window H's 12 patches ("必需 s23").
That assumption is void for the split — S23's bundle targeted the mirror, its `impl.cuh` is the
broken one in §3, and this bundle supersedes it. The other 11 patches are unaffected.

Also: do **not** apply this with S23's `apply.sh` (it would look at the mirror again), and do not
copy the mirror's tier TUs over this bundle's — they differ by the §3 Δ1 guard.

## 6. Acceptance for the landing window (no GPU needed for 1-3)

1. `bash ~/_collab/C_s40_tu_split_v2/apply.sh --dry-run /home/user/ninfer-fusion` → `DRYRUN OK`, exit 0.
2. Apply, then `python3 ~/_collab/C_s40_tu_split_v2/verify.py /home/user/ninfer-fusion` → `VERIFY_OK`.
3. Rebuild (own window) and check the partition with `nm`:
   `gqa_attention_decode.cu.o` contains **zero** i8/nvfp4/bf16/fp8/iso3 partial-kernel symbols;
   each tier `.o` contains its own; every `gqa_attention_decode_*_launch` is defined exactly once.
   Then link + fresh `apps/ninfer` mtime.
4. GPU window: `_muse_verify.sh` → `MUSE_VERIFY_PASS`; qwen 32K needle unchanged (identical
   kernels relocated — this is the behaviour-identity check); tok/s sanity vs baseline.

## 7. Reproduce everything

```
B=/mnt/c/Users/User/Documents/ziqinzhang/_collab/C_s40_tu_split_v2
python3 $B/mkpatch.py /home/user/ninfer-fusion          # regenerate patch/ + review/
python3 $B/mkpatch.py /home/user/ninfer-fusion --check  # determinism
python3 $B/verify.py  /home/user/ninfer-fusion          # 33 equivalence checks
bash    $B/apply.sh --dry-run /home/user/ninfer-fusion   # writes nothing
bash    $B/evidence/shadow_test.sh                       # lifecycle on a /tmp copy
bash    $B/evidence/path_audit.sh                        # host-path + write-set audit
bash    $B/evidence/mirror_delta.sh                      # §3 itemisation
bash    $B/evidence/integrity.sh                         # build tree untouched + bundle complete
```
`probe.py` in this directory is a **pre-existing scratch file from a partial earlier round**
(mtime 14:40, references the mirror); it is not part of this bundle and `apply.sh`/`patch/` do
not use it. `/tmp/s40_shadow` is the kept shadow tree (delete at will).
