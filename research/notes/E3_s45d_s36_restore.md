# S45d companion — `E3_s45b_s36_restore.diff` applicability vs pristine

Optional, low priority, per your item 3. No `src/**` write, no compile.

**Q: does it also merge branch conditions (the S45c mistake)?** No. Its four changed lines are value
expressions inside grid-size computations; `grep '^[+-]' <diff> | grep -c 'if\|else\|for\|while'` = **0**.
Nothing structural is touched.

**Q: does it apply to pristine?** Yes — it was generated from the pristine text
(`sha256 4b8e0d68…`, md5 `b2da4c43…`). Verbatim:

```
$ patch -p1 --dry-run -i _collab/E3_s45b_s36_restore.diff     # cwd = /home/user/ninfer-fusion
checking file src/ops/launcher/gqa_attention_prefill.cu
rc=0
```

**Q: does it conflict with `E3_s45d_prefill_guard_v2.diff`?** No: line-disjoint (S45d touches
53-83 / 100-196; this touches 269/316/335/352) and it applies in either order. Verified by a chained
shadow apply (`S45d` then `s36_restore`) and by dry-running the pair against the live tree:

```
$ patch -p1 -s -i E3_s45d_prefill_guard_v2.diff && patch -p1 -s -i E3_s45b_s36_restore.diff   # shadow copy
BOTH_APPLIED_IN_ORDER      # 4 x if constexpr(256) guards, 3 x (Geometry::HeadDim / kGqaKvNvfp4Group)
$ patch -p1 --dry-run -i E3_s45d_prefill_guard_v2.diff && patch -p1 --dry-run -i E3_s45b_s36_restore.diff
checking file src/ops/launcher/gqa_attention_prefill.cu
checking file src/ops/launcher/gqa_attention_prefill.cu
SEQ_DRYRUN_OK
```

**Content** (pristine line numbers, one substitution each):

| pristine line | site | from → to |
|---|---|---|
| 269 | `gqa_kv_append_launch_for`, NVFP4 (+ISO3-V) fill grid | `Geometry::KVHeads * kGqaKvNvfp4Groups` → `… * (Geometry::HeadDim / kGqaKvNvfp4Group)` |
| 316 | same fn, ISO3 fill grid | same substitution |
| 335 | same fn, FP8 fill grid | same substitution |
| 352 | same fn, bf16 fill grid | `kGqaPrefillHeadDim / kFillVecElems` → `Geometry::HeadDim / kFillVecElems` |

`kGqaKvNvfp4Groups` (= `kGqaKvNvfp4HeadDim / kGqaKvNvfp4Group` = 16, `gqa_attention_kv_nvfp4.cuh:31-34`) and
`kGqaPrefillHeadDim` (= 256, `gqa_attention_prefill_common.cuh:21`) are the two 256-family *reference*
constants; at head_dim 256 the substitutions are numerically identical, at 128 they halve the four fill
grids back to the geometry's real unit count.

**Why it is worth landing anyway** (evidence that the loss is perf-only, i.e. this is not urgent): both
fill kernels recompute their unit count from the geometry and return early —
`gqa_attention_prefill_bf16.cuh:31-34` (`n = tokens*KVHeads*(Geometry::HeadDim / VecElems); if (idx >= n)
return;`) and `gqa_attention_prefill_nvfp4.cuh:736-737` (`units = tokens*KVHeads*(Geometry::HeadDim /
kGqaKvNvfp4Group); if (unit >= units) return;`). So at 128 the pre-revert grids only waste CTAs; no wrong
bytes. The bf16 line (352) is the one that sits on a live 128 path (Muse bf16 prefill); the other three sit
in the packed fill path, which refuses at 128 once S45d lands.

Restore or drop, your call — it is not part of `E3_s45d_prefill_guard_v2.diff`.
