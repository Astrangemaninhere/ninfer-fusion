# E3 / S45 — int8+E8 plane LeadingExtents are the 256 family: enumeration, correction, prefill_i8@128 verdict

Round: one, CPU only, no GPU, **no writes to `src/**`** (patch only).
Deliverables: `_collab/E3_s45_i8_plane_stride.diff` (patch), this file, board row S45.
Reproduce: `wsl.exe -- bash -lc "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/E3_s45_mkpatch.py"`
→ `ANCHORS_OK 11/11` / `SHADOW_APPLY_OK 4/4` / `LIVE_DRYRUN rc=0` / `S45_PATCH_OK`.
Probe: `wsl.exe -- bash -lc "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/E3_s45_probe.py"`.

## 0. TL;DR

1. Three index helpers in `src/ops/kernel/gqa_attention_kv_quant.cuh` (`:47`, `:54`, `:103`) and one
   source-stride helper (`:59-64`) still carry the 256 family; **fixed in the patch** (`Geometry::HeadDim`
   derived). At head_dim 256 the new expressions are **provably identical** (798,720/798,720 sampled
   offsets equal, probe part 3) ⇒ zero regression risk on qwen.
2. At Muse (head_dim 128, kv_heads 2) **every int8/E8 plane index is exactly 2×** what the pool's own
   layout says ⇒ the K-codes, i4-codes and scale writes for `(kv_head h, page p)` land on the layout slot
   of **page `h+2p`** (both kv heads, all page_offsets), and every page with `h+2p >= G` lands *outside*
   the K plane. See §2.3 for the aliased regions and the byte envelopes.
3. `prefill_i8` at 128: **gate it off loudly, do not port** (§4). The gate is placed in **both** prefill
   entry points (`append fill` *and* `attention`) so nothing is written before the throw.
4. **BLOCKER found on the way (pre-existing, not mine)**: the `if constexpr (...) else throw` guards that
   landed in `src/ops/launcher/gqa_attention_prefill.cu` at 15:07 are inserted **inside argument lists**
   — that TU cannot compile, so `PREFPILL_GUARD_OK` is false and window J2's build fails there. The
   well-formed precedent is on the decode side (`gqa_attention_decode.cu:27-32`). Evidence + exact repair
   recipe in §5. My patch deliberately mirrors the decode form.
5. Only a GPU run can confirm the end-to-end symptom set (§6). Nothing here is GPU-verified.

## 1. Enumeration of the stride-family consumers (required item 1)

Classification key: **[G]** must be geometry-derived; **[F]** legitimately a fixed
hardware/format/tile quantity; **[G\*]** geometry-derived, provably value-identical at 256.

### 1.1 The helpers themselves — `src/ops/kernel/gqa_attention_kv_quant.cuh`

| site | quantity | old | classification |
|---|---|---|---|
| `:40` | `kGqaKvQuantHeadDim = 256` | — | **[F]** now a *reference* constant (like `kGqaPrefillHeadDim`, `gqa_attention_prefill_common.cuh:19-21`); kept, must not be used as a plane stride |
| `:41-42` | `Group = 64`, `Groups = 4` | — | **[F]** group domain (matches `kKvInt8QuantGroup = 64`, `decoder_state.h:12`, and `gqa_attention.cpp:17`) |
| `:45-49` | `gqa_kv_quant_code_index` → `LeadingExtent = 256` | 256 | **[G]** → `Geometry::HeadDim` (patch) |
| `:52-56` | `gqa_kv_quant_scale_index` → `LeadingExtent = 4` | 4 | **[G]** → `Geometry::HeadDim / kGqaKvQuantGroup` (patch) |
| `:100-105` | `gqa_kv_i4_code_index` → `LeadingExtent = 128` | 128 | **[G]** → `Geometry::HeadDim / 2` (patch) |
| `:59-64` | `gqa_kv_quant_src_index` token stride = 256 | 256 | **[G]** → `Geometry::HeadDim` (patch). The dense K/V source has `ne[0] == cache.head_dim` enforced at `src/ops/wrapper/gqa_attention.cpp:486-487` (`require_shape(k, cache.head_dim, ...)`) |

Layout authority (host side, already geometry-derived — this is what makes the 256 family provably
wrong, independent of any GPU):
`src/targets/qwen3_6/impl/state/decoder_state.cpp:106-116` pushes
`KVPlaneGeometry{DType::I8, leading = head_dim, head_extent = kv_heads}` and
`{DType::FP16, leading = head_dim / group, …}`; `src/ops/wrapper/gqa_attention.cpp:90-91`
(`code_leading = head_dim`, `/2` for packed) and `:126` (`groups = head_dim / 64`) validate the paged
tensors against exactly those extents. `src/core/paged_kv_cache.cpp:71-81` turns them into
`{leading, 64, head_extent, pages}`.

### 1.2 Consumers of those helpers, by call site

| # | consumer (file:line) | helper used | path | classification |
|---|---|---|---|---|
| 1 | `gqa_attention_decode_i8.cuh:313,316` | code | decode fused append K (i8) | **[G]** (now via helper) |
| 2 | `gqa_attention_decode_i8.cuh:319,322` | code | decode fused append V (i8) | **[G]** |
| 3 | `gqa_attention_decode_i8.cuh:282,286,290,294` | i4 | decode fused append K/V (E8) | **[G]** |
| 4 | `gqa_attention_decode_i8.cuh:301,327` | scale | decode fused append scale (E8/i8) | **[G]** |
| 5 | `gqa_attention_decode_i8.cuh:462` | scale | decode `cp_async` scale row stage | **[G]** |
| 6 | `gqa_attention_decode_i8.cuh:483` | i4 | decode `cp_async` code stage (E8) | **[G]** |
| 7 | `gqa_attention_decode_i8.cuh:489` | code | decode `cp_async` code stage (i8) | **[G]** |
| 8 | `gqa_attention_prefill_i8.cuh:114-115, 253-254` | src | fill kernels: source K/V read | **[G]** (patch) |
| 9 | `gqa_attention_prefill_i8.cuh:177,181,185,189` | i4 | fill v1 (E8) | **[G]** |
| 10 | `gqa_attention_prefill_i8.cuh:200,215` | scale | fill v1 (E8/i8) | **[G]** |
| 11 | `gqa_attention_prefill_i8.cuh:208` | code | fill v1 (i8) | **[G]** |
| 12 | `gqa_attention_prefill_i8.cuh:313,317,321,325` | i4 | page fill (E8) | **[G]** |
| 13 | `gqa_attention_prefill_i8.cuh:546` | i4 | prompt attention stage (E8) | **[G]** |
| 14 | `gqa_attention_prefill_i8.cuh:551` | code | prompt attention stage (i8) | **[G]** |
| 15 | `gqa_attention_prefill_i8.cuh:525` | scale | prompt attention stage | **[G]** |
| 16 | `gqa_attention_prefill_i8.cuh:331-333` | **raw** `paged_kv_page_head_offset<kGqaKvQuantGroups,…>` | page fill (E8) scale | **[G]** (patch → helper) |
| 17 | `gqa_attention_prefill_i8.cuh:341-342` | **raw** `paged_kv_page_head_offset<kGqaKvQuantHeadDim,…>` | page fill code | **[G]** (patch → helper) |
| 18 | `gqa_attention_prefill_i8.cuh:349-351` | **raw** `paged_kv_page_head_offset<kGqaKvQuantGroups,…>` | page fill scale | **[G]** (patch → helper) |

After the patch, `grep -rn 'paged_kv_page_head_offset<kGqaKvQuant' src/` is empty: **every** plane index in
the int8/E8 family routes through a `Geometry`-derived helper. The remaining 256-only thing in that family
is the **group count / tile shape**, not a stride (§4).

Cold path (checked, unchanged): `cold_i8_kernels.cuh` is the *slot* codec, not a plane indexer; `:72-76`
`template <int Groups = 4>` + `:83 g < Groups` is the S42 fix. `decode_i8.cuh:420-421`
(`row_codes[Geometry::HeadDim]`, `row_scales[Geometry::HeadDim/64]`) already geometry-sized.

### 1.3 Legitimately fixed quantities (do **not** "fix" these)

| site | why fixed |
|---|---|
| `paged_kv_address.cuh:9-12` (`kPagedKVPageShift = 6`, `kPagedKVPageMask`) | the 64-token page is a pool/hardware format, validated at `paged_kv_cache.cpp:40` |
| `cold_i8_kernels.cuh:32-37` (16 B header / 8192 B nibbles / 1024 B E4M3) | cold-slot **format**, defined by the nvfp4 requant (`[128,64,kv_heads,pages]` nibbles + `[16,64,…]` sub-scales); the row is a 256-dim container at any head_dim |
| `gqa_attention_prefill_i8.cuh:498-499, 502, 509` (`row_codes[256]`, `row_scales[4]`) | reads a cold slot row (§ above). Consumed **only** by the 256-only prompt kernel. A 128 port must first define what the cold slot means at 128 (only the first `head_dim` channels are meaningful) |
| `gqa_attention_prefill_i8.cuh:26-51, 382-383` (`kGqaPrefillI8*`, `SmemBytes == 92672`, `GroupKc == 2`, `PVNtPerWarp == 8`) | tile/smem quantities of the 256-shaped kernel: `Wc=16` warps / 4 row-tiles, `PVNtPerWarp = D/(4*8) = 8` only holds at `D = 256`. Legitimate **because** the kernel is 256-only — which is exactly what must be gated (§4) |
| `gqa_attention_decode_i8.cuh:98` (`PageIds = 256`) | per-split page-id upper bound (the 1.01 M-key YaRN envelope = 186 pages < 256). A capacity, not a stride |
| `gqa_attention_decode_i8.cuh:63` ("256-wide PV output") | comment; the code below uses `Geometry::HeadDim` |
| `gqa_attention_decode.cu:611` `div_up(kGqaHeadDim, kDChunk)` reduce grid | **inert**: the grid is 2× oversized at 128 but every block early-returns on the (already geometry-derived) bound `d >= Geometry::HeadDim` (`gqa_attention_decode.cuh:210,237`). Cosmetic waste only — same class as the `decode.cu:254` i8-arena note in S40-d (`DynamicArena == false` on Muse ⇒ ×0) |

## 2. Corrected expressions and the arithmetic (required item 2)

### 2.1 The patch

`gqa_attention_kv_quant.cuh` (new):

```cpp
template <typename Geometry> inline constexpr int kGqaKvQuantCodeLeading  = Geometry::HeadDim;
template <typename Geometry> inline constexpr int kGqaKvQuantScaleLeading = Geometry::HeadDim / kGqaKvQuantGroup;
```

| helper | corrected expression | = at 256 | = at 128 |
|---|---|---|---|
| `gqa_kv_quant_code_index` | `paged_kv_element_offset<kGqaKvQuantCodeLeading<Geometry>, KVHeads>` | 256 | **128** |
| `gqa_kv_quant_scale_index` | `paged_kv_element_offset<kGqaKvQuantScaleLeading<Geometry>, KVHeads>` (+ `static_assert(head_dim % 64 == 0)`) | 4 | **2** |
| `gqa_kv_i4_code_index` | `paged_kv_element_offset<kGqaKvQuantCodeLeading<Geometry> / 2, KVHeads>` | 128 | **64** |
| `gqa_kv_quant_src_index` | `d + Geometry::HeadDim * (kv_head + KVHeads * token)` | 256 | **128** |

`prefill_i8.cuh` `:331/:341/:349` now call the two helpers instead of hand-rolled
`paged_kv_page_head_offset<256-family,…> + page_off * 256-family + …`.

### 2.2 Element-offset arithmetic

`offset(W, head, page, page_off, lead) = W*64*(head + KVHeads*page) + W*page_off + lead`
(plane tensor `{W, 64, KVHeads, pages}`, dim0 fastest — `paged_kv_cache.cpp:74`).

| geometry | plane | correct W | wrong W | elements per (page,head) wrong/correct | verdict |
|---|---|---|---|---|---|
| Muse 32q/2kv D=128 | i8 codes | 128 | 256 | 32768 / 16384 | **2×** |
| Muse | E8 i4 codes (bytes) | 64 | 128 | 16384 / 8192 | **2×** |
| Muse | fp16 scales | 2 | 4 | 512 / 256 | **2×** |
| Muse | src token stride | 128 | 256 | — | **2×** |
| qwen 24q/4kv D=256 | all four | — | — | identical | ok |
| qwen 16q/2kv D=256 | all four | — | — | identical | ok |

Exhaustive probe part 3 (all `page<8 × head<KVHeads × page_off<64 × lead<W`): **old == new for 798,720 /
798,720** offsets at head_dim 256 (0 mismatches) and differs for 261,888/262,144 (codes) and 4,092/4,096
(scales) at 128. So the patch is a **no-op on qwen** (byte-identical index stream) and a real fix on Muse.

### 2.3 What region was being aliased on Muse (worked example)

Layout coordinate system: `index = 8192*(2*page + kv_head) + 128*page_off + d` (Muse).
Wrong stream: `index = 16384*(2p + h) + 256*off + d`.

* Probe example `(kv_head=1, page=3, page_offset=37, d=0)`: wrong **124160**, correct **62080** (Δ 62080).
  The wrong address *is* the layout slot `(page 7, kv_head 1, page_offset 10, d 0)` (= `q = h+2p = 7`,
  `2*off = 74 = 64 + 10`).
* In general the write for `(kv_head h, page p)` addresses a contiguous **16384-element block**
  (`2 * leading * 64`) — i.e. **the whole layout page `q = h + 2p`, both kv heads, all 64 page_offsets** —
  while the correct slot is page `p`. The scale plane follows the same ×2 law (`group` row 4 vs 2).
* Because `q` is a bijection of `(h,p)` onto `q ∈ [0, 2G)` and the plane only has `q ∈ [0, G)`:
  * every page `p >= ceil((G - h)/2)` has its **K-code write outside the K plane** (it lands in the
    V-codes plane, at `q - G`);
  * K-code writes as a whole stay inside the K+V pair: max envelope `32768*G` B = exactly the end of the
    V plane (`G` = page-group count);
  * **V-code writes as a whole do not**: max envelope `V_base + 32768*G = 49152*G` B, versus the layer's
    whole 4-plane set `2*16384G + 2*512G = 33792*G` B ⇒ **`15360*G` B past the end of the layer's planes**
    (into the next layer; past the end of the pool for the last layer).
    E.g. `G=256` (16 Ki pages): K/V/scale planes 4 MiB/4 MiB/256 KiB/256 KiB = 8.25 MiB per layer, and the
    V-code envelope is 12.58 MiB — 3.9 MiB past it.
  * scale writes stay inside the layer set (max `8650751` of `8650752` B at `G=256` — one byte spare).

**Self-consistency, and why a loop test cannot see it.** Fill and decode/attention share the same wrong
stride, so a Muse i8/E8 round-trip returns the values it wrote — this is why the cache round-trip tests
that exist today pass. It stops being self-consistent as soon as (a) K and V planes interleave (the
K-writes for `q >= G` land on top of V's own writes for `q - G`), or (b) anything else consumes the
aliased pages. On a real run this is **silent numerical corruption, not a fault**, and it starts once the
committed page set passes ~half the pool.

## 3. Fix scope in the patch (files + hunk list)

1. `src/ops/kernel/gqa_attention_kv_quant.cuh` — 2 hunks: the four helpers above + the reference-constant
   comment + `kGqaKvQuantCodeLeading` / `kGqaKvQuantScaleLeading` + `static_assert(head_dim % 64 == 0)`.
2. `src/ops/kernel/gqa_attention_prefill_i8.cuh` — 3 hunks: the raw `paged_kv_page_head_offset` uses in the
   page-fill kernel now call the helpers. **Value-identical at 256** (`group * 64 + lane` == the old
   `page_off * 256 + group * 64` base at `leading == 256`), and it removes the last hand-rolled stride, so
   the family invariant "no `paged_kv_page_head_offset<kGqaKvQuant…>` anywhere" becomes mechanically
   checkable. It does **not** make the fill 128-capable: the funnels still iterate
   `kGqaPrefillI8Groups == 4` (256-only) — hence §4.
3. `src/ops/launcher/gqa_attention_prefill.cu` — 3 hunks: `require_i8_prefill_geometry_dim` +
   a gate in each of the two int8/E8 entry points.
4. `src/ops/launcher/gqa_attention_prefill_e8.cu` — 3 hunks: `#include <string>` + the same gate inside
   `attention_e8_for` / `append_e8_for`, so the E8 TU is self-defending even if a future caller bypasses
   `gqa_attention_prefill.cu`.

## 4. Verdict: `prefill_i8` at 128 — **gate off loudly, do not port** (required item 3)

Porting is not a constant swap. `gqa_attention_prefill_i8.cuh` is 256-shaped at the *tile* level:

* `:26-43` `kGqaPrefillI8Groups == 4`, `kGqaPrefillI8DB16 == 128`, `KBytes/VBytes/QBytes` all from
  `kGqaPrefillHeadDim`, `SmemBytes == 92672`;
* `:49-51` `static_assert(Groups == 4)`, `static_assert(DConsumers == 4)`, `static_assert(SmemBytes == 92672)`;
* `:382-383` `static_assert(PVNtPerWarp == 8)` — with `D = Geometry::HeadDim = 128` and 4 consumer warps
  this is `4`, so the whole 16-warp producer/consumer partition (4 row tiles × 4 d-consumers) and the
  `p_s`/`q_scale` smem layout have to be re-derived; the 4 d-consumers exist only because 4 groups must be
  covered.

That is a kernel re-tune whose only judge is a GPU run — unavailable this round. Gating is also
*consistent with what the tier can already do*: the nvfp4/FP8 prompt kernels are 256-only too
(`if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) … else throw` in the same launcher), `serve_muse.sh:25`
defaults Muse to `--kv-dtype bf16`, and the wrapper forces `E8Kv` to the prompt path
(`gqa_attention.cpp:494-497`) — so no *working* Muse configuration is lost, and the honest outcome is
"head_dim 128 supports bf16 prefill; everything else refuses".

The cited pattern's **well-formed** instance is the decode side, so I copied that one:

```cpp
// src/ops/launcher/gqa_attention_decode.cu:27-32
[[noreturn]] inline void require_nvfp4_geometry_dim(int head_dim) {
    throw std::invalid_argument("nvfp4 KV decode requires head_dim=256 but this geometry has " + …);
}
// :532-540 / :548-556
if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) { <launch> }
else { (void)nvfp4_k; (void)nvfp4_v; require_nvfp4_geometry_dim(Geometry::HeadDim); }
```

My gate uses the equivalent short form and is placed at **both** entry points:

```cpp
if (cache.dtype == DType::I8 || cache.dtype == DType::E8Kv) {
    if constexpr (Geometry::HeadDim != kGqaKvQuantHeadDim) {
        require_i8_prefill_geometry_dim(Geometry::HeadDim);
    }
    …existing body…
```

Why both: `gqa_attention_prompt_launch` (`gqa_attention_prefill.cu:473-488`) runs
`gqa_kv_append_launch_for` **then** `gqa_attention_prompt_attention_launch_for`, and both are also public
entry points called separately from `src/ops/wrapper/gqa_attention.cpp:513/:540/:564`. An
attention-only gate would let the append fill write a half-fixed plane before throwing — which is the
"half-committed state" failure mode. Note the *same argument* applies to the nvfp4/FP8 append branch
(`prefill.cu:311-395`), which is unguarded today: a Muse+nvfp4 prompt would fill, then throw in the
attention launcher. Out of my scope; flagged in §7.

Instantiation detail (why a runtime throw was not enough for nvfp4 but is acceptable here): for nvfp4 the
guard is load-bearing at *compile* time — `gqa_attention_prefill_nvfp4_kernel<GqaMuseGeometry,…>` cannot be
instantiated (`prefill_nvfp4.cuh:1026 static_assert`, `QKKs = D/64 == 2` vs `== 4`), so the branch must be
discarded. The i8 kernel *does* compile at 128 (its statics are all globals), so
`if constexpr (HeadDim != 256) throw` is a behavioural gate: it still instantiates the 256-shaped kernel,
it just refuses to run it. Stated plainly so nobody later mistakes it for de-instantiation.

## 5. Evidence, and the blocker found on the way (required item 4)

### 5.1 Mechanical evidence (host only, re-runnable)

| check | command | result |
|---|---|---|
| anchor health | `E3_s45_mkpatch.py --check` | `ANCHORS_OK 11/11` (every `old` string occurs exactly once) |
| patch reproduces the intended text | `E3_s45_mkpatch.py` | `SHADOW_APPLY_OK 4/4` — patch applied to pristine copies is **byte-identical** to the intended files |
| dry-run on the live tree | same | `LIVE_DRYRUN rc=0`, 4/4 `checking file …`, `S45_PATCH_OK` |
| 256 parity / 128 mapping | `E3_s45_probe.py` part 3 | 798,720/798,720 offsets equal at 256; 261,888/262,144 differ at 128 |
| aliasing arithmetic | `E3_s45_probe.py` part 2 | the tables in §2.2/§2.3 |
| delimiter audit | `E3_s45_probe.py` part 1 | see §5.2 |

### 5.2 BLOCKER: `src/ops/launcher/gqa_attention_prefill.cu` cannot compile

The guards landed at 15:07 (mtime) are inserted **inside argument lists**, not around statements.
Delimiter audit (a `}` whose matching `{` was opened at paren depth 0 while `()` is still open — a hard
syntax error; control files all 0):

```
src/ops/launcher/gqa_attention_prefill.cu   501 lines  8 guards  VIOLATIONS 9
  line 142: closes the block opened at line 125 at paren depth 0, but () = 1 is still open
  line 147: stray ')'   line 165: stray '}'   line 170: stray ')'   line 188: stray ')'
  line 190: stray '}'   line 217: stray ')'   line 245: stray '}'   line 501: '}' at () = 4
src/ops/launcher/gqa_attention_decode.cu    737 lines  2 guards  VIOLATIONS 0   (control)
src/ops/launcher/gqa_attention_prefill_e8.cu 236 lines 0 guards  VIOLATIONS 0   (control)
src/ops/kernel/gqa_attention_prefill_i8.cuh  889 lines               VIOLATIONS 0   (control)
src/ops/kernel/gqa_attention_kv_quant.cuh    133 lines               VIOLATIONS 0   (control)
src/ops/kernel/gqa_attention_decode_i8.cuh   815 lines               VIOLATIONS 0   (control)
```

Root cause, verbatim (lines 137-147): the closing `} else { throw … }` was inserted after the first line of
a multi-line initializer, splitting `reinterpret_cast<const std::int32_t*>(` from its argument:

```cpp
            const std::int32_t* cold_v_valid =
                cold_k_valid == nullptr ? nullptr
                    : reinterpret_cast<const std::int32_t*>(
                          reinterpret_cast<const std::uint8_t*>(cold_k_valid) +
        } else {                       // <-- line 142: block ends inside the call's argument list
            throw std::invalid_argument("nvfp4 prefill requires head_dim=256 …");
        }
                      cache.cold_slot_valid.nb[1]);      // <-- line 147: orphaned argument
```

Same shape at 165/169 (the ISO3 launch) and 222-226. The four `cudaFuncSetAttribute` guards (`:54-96`) are
fine; the four "launch" guards are not. Since `src/CMakeLists.txt:79` lists this TU, **any build including
it fails at this file**, and `PREFPILL_GUARD_OK` (brace-balance was the check) was a true-but-insufficient
signal. This is the "half-applied state" mode the protocol warns about.

Repair recipe (mechanical; the 4 insertions must move from inside the argument lists to around the three
256-only statements — `NVFP4`, `ISO3`, `FP8_E4M3FN` — in the `else if` chain):
move `if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) {` from `:125`/`:148`/`:176`/`:204` to
immediately after each branch's opening brace, and put the matching `} else { throw … }` immediately before
that branch's closing brace; the same three-statement form works for the four `attr_*` sites as written.
`/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/src/ops/launcher/gqa_attention_prefill.cu`
(26,937 B, Sep 4) is the pre-guard text and diffs to **guard-only** differences (`diff -u` shows exactly
these insertions and nothing else), so a safe repair is "take the mirror text and re-apply the guards
correctly". I did **not** put that repair in my patch: it is M's edit, it is 4 hunks of re-indentation
around lines I did not touch, and mixing it in would make my stride patch un-reviewable. Say the word and I
will deliver it as a separate patch.

## 6. What only a GPU run can confirm (required item 4c)

Nothing above is GPU-verified. Host evidence proves the *indices*; it cannot prove the end-to-end effect.
A GPU run must establish:

1. **Muse i8/E8 round-trip after the patch** — the acceptance pair is: prefill **throws**
   (`invalid_argument: int8/E8 prefill requires head_dim=256 …`) instead of writing, and the decode path
   (fused append + read at 128) produces finite, correct output. Only this distinguishes "the 2× stride is
   gone" from "it is gone in the index math but something else was also 256-shaped" (e.g. the
   `k_scale_s[Bc*Groups]` staging, the `PageIds` bound, the SWA parameter gap).
2. **The aliasing prediction** on a *pre*-patch binary (if anyone still has one): with int8/E8 KV on Muse,
   corruption should begin around half the page pool (`q = h+2p >= G`), not at a fixed absolute position.
3. **qwen 8/8 needle regression unchanged** — the only guard against a silent 256-side regression. The
   probe says old == new for every sampled offset at 256 and the patch is textually confined to
   `Geometry::HeadDim` substitutions, but the kernel is the authority: run the standard qwen 8/8 needle
   set (32K) and require the same pass/fail vector as before the patch.
4. **Muse `--kv-dtype bf16` unchanged** (the default tier): `_muse_verify.sh` → `MUSE_VERIFY_PASS`, no NaN,
   since S36/S45 touch the plane strides that bf16 does *not* use (bf16 planes are `leading = head_dim`,
   `gqa_attention_decode.cuh:37` already geometry-derived) — expectation: zero change, and a change would
   mean the patch reached further than it says.
5. Compilation of the four TUs in §3 under nvcc (host g++ cannot type-check `__device__` code here) —
   the window's job.

## 7. Residual / out of scope (flagged, not fixed)

* `gqa_attention_prefill.cu` guard syntax — **§5.2, blocking**.
* The nvfp4/FP8 **append** branch (`prefill.cu:311-395`) is unguarded at 128: a Muse+nvfp4 prompt fills
  first and throws in the attention launcher. Same "gate both entry points" argument as §4.
* `gqa_attention_decode.cu:254` (`4 * KeyBlock * kGqaHeadDim` i8-arena) and `:611`
  (`div_up(kGqaHeadDim, kDChunk)` reduce grid) are inert on Muse (`DynamicArena == false`; the reduce
  kernel is guard-bounded) — cosmetic, S40-d's follow-up list.
* The cold-slot format at head_dim 128 is undefined (the slot row is a 256-dim container). Any future 128
  port of `prefill_i8` must decide it before touching the cold path (`prefill_i8.cuh:498-513`).

---

# S45b — prefill-guard repair (separate patch, main deliverable of this round)

Deliverables: `_collab/E3_s45b_prefill_guard.diff` (required) + `_collab/E3_s45b_s36_restore.diff`
(companion, 4 lines) + generator `E3_s45b_mkpatch.py` + parser audit `E3_s45b_guard_audit.py` + syntax
check `E3_s45b_syntax_check.py` + negative control `E3_s45b_mangle_probe.py`.

## B1. The tree moved under me at 15:50:42 — read this first

| when | sha256 | lines | what |
|---|---|---|---|
| my S45 round (md §5.2) | `644a75969daf…` | 501 | build tree **with** M's 8 guards, 4 of them inside argument lists (the TU I proved cannot compile) |
| **15:50:42, window K** | `4b8e0d6864ea…` | 453 | the file was **replaced by the mirror copy** (`ninfer-fusion-repo/...`, 26,937 B, Sep 4) |

So the mis-placed guards are gone — replaced wholesale by the Sep-4 text. That unblocks the TU, and it is
why `E3_s45b_prefill_guard.diff` is now a **clean, additive landing of the intended guards** rather than a
correction of M's insertions. Two consequences to know:

1. **The revert also dropped four S36 geometry derivations** in `gqa_kv_append_launch_for`
   (`gqa_attention_prefill.cu:269, :316, :335` `* kGqaKvNvfp4Groups` → `* (Geometry::HeadDim /
   kGqaKvNvfp4Group)`, and `:352` `kGqaPrefillHeadDim / kFillVecElems` → `Geometry::HeadDim /
   kFillVecElems`). At head_dim 256 these are value-identical; at 128 the four fill grids become 2×
   oversized. **Not corruption**: both fill kernels re-derive their own unit count from the geometry and
   return early — `gqa_attention_prefill_bf16.cuh:31-34` (`n = tokens*KVHeads*(Geometry::HeadDim/VecElems);
   if (idx >= n) return;`) and `gqa_attention_prefill_nvfp4.cuh:736-737` (`units = tokens*KVHeads*
   (Geometry::HeadDim/kGqaKvNvfp4Group); if (unit >= units) return;`). So the loss is wasted CTAs, not
   wrong bytes. `E3_s45b_s36_restore.diff` puts them back (4 hunks, disjoint from the guard hunks, applies
   in either order — verified by a chained shadow apply).
2. A mirror copy is **not** a usable base for this file in general: the mirror lacks those four S36 sites,
   and S41/S40 already established that "the mirror is the authority" is false. Here the revert happened to
   be safe only because those two kernels self-guard.

## B2. The repair: two compile-time group guards, not eight statement-level ones

`E3_s45b_prefill_guard.diff`, base `4b8e0d6864ea…` → result `052aeede375c…`:

* **seam 1** — one `if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) { … } else { throw }` around the
  **four** `cudaFuncSetAttribute` odr-uses (`:54-58, :59-62, :63-67, :68-72`). Binding a kernel template's
  address instantiates it, so these must be discarded at 128 exactly like the launches.
* **seam 2** — one group guard around the whole **packed tier**:
  `} else if (cache.dtype == DType::NVFP4 || cache.dtype == DType::ISO3 || cache.dtype == DType::FP8_E4M3FN) {`
  → `if constexpr (256) { if (dtype == NVFP4) {…} else if (dtype == ISO3) {…} else if (dtype == FP8) {…} }
  else { throw … }`, with the bf16 arm left outside the guard.

Diff facts (printed by the generator): 454 → 478 lines, **453 lines unchanged**, `-1/+25`; the single
removed line is the old NVFP4 arm opener. Every added line is guard scaffolding or a comment. So
"the 256 path is unchanged" is a diff fact, not a claim: for a 256 geometry the branch resolution and the
statement sequence are exactly the mirror's, and the new condition routes each of the three dtypes to the
same arm the old `else if` chain did (`:852` says an exotic dtype still falls through to the bf16 arm,
which `!= DType::BF16` would have changed).

**Why not one function-body wrap (constraint 1).** A whole-body wrap is wrong here and would break the one
working tier: `gqa_attention_prompt_attention_launch_for<GqaMuseGeometry,…>` is the **bf16** prefill path on
Muse (`serve_muse.sh:25` defaults `--kv-dtype bf16`) and must keep working at head_dim 128. The 256-only
set is not "the function", it is the four packed kernels — so the closest correct form of the requested
single wrap is one wrap per *tier group*: 2 seams, down from 8. Both keep the 4 `cudaFuncSetAttribute`
odr-uses inside the guarded branch, as instructed.

## B3. Proof it compiles, not just balances (constraint 2)

**(a) Full `nvcc -fsyntax-only` with the build's own flags.** `E3_s45b_syntax_check.py` reads
`build/compile_commands.json`, keeps `--options-file CMakeFiles/ninfer_ops.dir/includes_CUDA.rsp -O3
-DNDEBUG -std=c++20 --generate-code=arch=compute_120a,code=[compute_120a,sm_120a] -x cu -rdc=true` and
adds `-fsyntax-only`, polling the whole process tree's RSS against a 1200 MB cap:

* **negative control** (the pre-revert text, sha256 `644a7596…`, run at 15:47): `rc=2` in **1.5 s**, 31
  diagnostics, first ones exactly at the inserted braces —
  `gqa_attention_prefill.cu(142): error: expected an expression` / `expected a ")"` / `expected a ";"`,
  `(147): error: expected a ";"`, then `identifier "attention_grid" is undefined` (151) etc. This is the
  compile-level statement that brace balance hid.
* **final artifact** (`E3_s45b_prefill_guard.diff` + `E3_s45b_s36_restore.diff`,
  sha256 `4f54264416bb…`): run launched, result appended below (single nvcc, cap 1200 MB, detached via
  `setsid`; window K's rebuild was running concurrently, so it was deliberately not re-run in parallel).

**(b) Parser-level invariants** (`E3_s45b_guard_audit.py`, the check that *would* have caught the landing):

* A: no `if constexpr` token may sit at paren/bracket depth > 0 — catches the two openers M inserted
  inside calls;
* B: no `}` may close a block opened at paren depth 0 while `(`/`[` is open — catches the four closers.

| file | A | B | note |
|---|---|---|---|
| repaired (guard) | clean | clean | balance 0/0/0 |
| repaired (guard+restore, mutated by `E3_s45b_mangle_probe.py`) | 2 (lines 246, 330) | **1 (line 135)** | re-injected defect, caught |
| build tree (as of S45b, reverted) | clean | clean | no guards |
| controls: `gqa_attention_decode.cu`, `prefill_e8.cu`, `prefill_i8.cuh` | clean | clean | no false positives |

**(c) Application evidence**: `ANCHORS_OK 4/4 + 4/4`; both patches `--dry-run` clean on the live tree;
chained shadow apply (`guard.diff` then `s36_restore.diff`) is byte-identical to the intended guard+restore
text; `E3_s45_i8_plane_stride.diff` (S45) still dry-runs clean against the reverted tree (hunks 2/3 at
offsets −24/−48).

Line ranges touched by `E3_s45b_prefill_guard.diff`: insertions at 53/72 (attrs), 100/196 (packed group);
one line rewritten (the NVFP4 arm opener). `E3_s45b_s36_restore.diff`: single-line substitutions at
269, 316, 335, 352.

## B4. Acceptance criteria, and what only the real build / a GPU can settle

* "compiles for both geometries" — **`-fsyntax-only` only proves the frontend accepts both instantiations**;
  no codegen/ptxas, and the real acceptance is window K's rebuild of this TU (the file is a build
  dependency of `ninfer_ops`, `src/CMakeLists.txt:79`).
* "256 path byte-identical in behaviour" — argued from the diff (453 unchanged lines, no re-indentation, one
  rewritten condition) but only a GPU run can confirm: qwen 8/8 needle **pass vector unchanged** and Muse
  `--kv-dtype bf16` (`_muse_verify.sh` → `MUSE_VERIFY_PASS`, no NaN).
* "a 128 geometry throws the readable geometry error instead of instantiating the kernel" — the
  *instantiation* half is structural (`if constexpr` discards the branch; the compiler error
  `Mxf4QKKs == 4` in `prefill_nvfp4.cuh:1026` is the proof that instantiation is what breaks). The *throws*
  half is a GPU/target-lib run: Muse + nvfp4/ISO3/FP8 KV prefill must now return
  `invalid_argument("nvfp4 prefill requires head_dim=256; this geometry is 128 — that kernel is not ported;
  use bf16 or int8 KV")` (wording kept verbatim from the 15:07 landing).
* Note the interaction with S45: at 128 the int8/E8 arm still needs the *separate* `E3_s45_i8_plane_stride.diff`
  gate; the two patches touch disjoint lines and can land in either order.

## B5. Superseded by S45c (same round, after M's review) — read this instead of §B2/§B3

* The guard deliverable is now **`_collab/E3_s45c_prefill_guard.diff`** (same base, md5
  `b2da4c437ea412902a09d2a378dfafeb` / sha256 `4b8e0d68…`, 453 lines → result `33514f10…`, 477 lines);
  `E3_s45b_prefill_guard.diff` is the earlier name of the same content and is superseded. Full write-up:
  **`_collab/E3_s45c_prefill_guard.md`**.
* One real fix on review: the attr group guard carries **no `else { throw }`**. Its four
  `cudaFuncSetAttribute` statements are unconditional in the function body, so an else-throw would also fire
  for `cache.dtype == DType::BF16` at head_dim 128 — i.e. it would have killed Muse's only usable prefill
  tier (the 15:07 shape had exactly that latent bug, behind the syntax error). The refusal lives in the
  packed launch group where the dtype is known.
* **No ad-hoc compiles any more** (M's instruction): my `-fsyntax-only` tree was killed by PID, window K's
  build left untouched, MemAvailable restored. Compile verification is the official `make`. The parser-level
  results in §B3(a)/(b) still stand and are what `E3_s45b_guard_audit.py` re-checks in ~1 s.
* The four S36 derivations lost in the 15:50 revert stay **out** of the guard diff (M's constraint) and ship
  as the separate `E3_s45b_s36_restore.diff`; evidence that their loss is perf-only is in §B1.
* New in this round: `_collab/E3_s45c_spec_usage.diff` (the `--spec` usage text, §7 of the S45c md).
