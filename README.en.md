# NInfer Fusion (Community Enhancement Pack)

An enhanced distribution of [Neroued/ninfer](https://github.com/Neroued/ninfer)
(Apache-2.0) that merges KV compression, cold tiers, speculative decoding and
long-context features into a single maintainable patch set, targeting the
RTX 5090 / `sm_120a`.

## How to read this file

This file states **mechanisms and state**. It deliberately carries **no performance,
precision or memory figures**: a figure is only worth the run behind it, and a run is
only worth its argv and the binary that produced it.

Three rules apply to everything below.

1. **Every standing claim has a provenance path** — either a `path:line` coordinate in
   this working tree, or a record path under `dl/<line>/`.
2. **Numbers that are exact by construction are kept** (see *Exact by construction*).
   They are `constexpr` byte counts pinned by `static_assert` in the tree: they cannot
   drift without breaking the build.
3. **Numbers that were measured once — one argv, one corpus, one geometry — are not
   here.** Where such a figure would be useful, this file names the evidence path
   instead ("machine-measured here; the paired binary sha256 and the full argv are
   recorded in `dl/<line>/`").

The figures that used to stand in this file were withdrawn on 2026-09-30. They are
preserved verbatim, each with its withdrawal reason and its evidence path (or
"evidence not found"), in
[`docs/features/withdrawn-numbers-2026-09-30.md`](docs/features/withdrawn-numbers-2026-09-30.md).
That file is an archive, not a reading.

## What this pack is

NInfer is a from-scratch C++/CUDA inference engine for explicitly registered Qwen
checkpoints on a single NVIDIA GeForce RTX 5090 (`README.md:5-8`). Prompts enter
through a local CLI or through OpenAI- and Anthropic-compatible HTTP APIs
(`README.md:6-7`). The runtime is deliberately specialized: one GPU, one resident
model, and a startup-fixed capacity of one to eight active requests (`README.md:7-8`).

This pack is our fork — `origin = Astrangemaninhere/ninfer-fusion`, branch `main`. It
adds the KV compression, cold-tier, block-descent and long-context machinery described
below, and keeps upstream's registrations intact.

## Artifacts

`apps/CMakeLists.txt` declares five executables: `ninfer` (`:1`), `ninfer-serve`
(`:13`), `ninfer-perplexity` (`:24`), `ninfer-cufree` (`:54`) and `ninfer-hostpath`
(`:118`).

Three of them are installed; the install rule is
`install(TARGETS ninfer ninfer-serve ninfer-perplexity` (`apps/CMakeLists.txt:75`),
and the two remaining executables are deliberately left out of it
(`apps/CMakeLists.txt:49-53`).

- Documentation components install to `${CMAKE_INSTALL_DOCDIR}`
  (`CMakeLists.txt:551`, `:600`).
- The configure step **generates** a self-describing `ninfer-install-manifest.txt`
  (`CMakeLists.txt:557`). It is a configure-time product, not a file in the source
  tree, so `ls` on a source checkout will not find it.

## Build

### WSL2 (primary)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j
```

Upstream's requirements (`README.md:25-28`) apply unchanged: 64-bit Linux, CUDA
Toolkit 13.1 or newer, CMake 3.28 or newer, a C++20 host compiler, Ninja,
`pkg-config`, the FFmpeg development libraries and `libcurl`. The build rejects CUDA
architectures other than `sm_120a`.

### Native Windows (adaptation branch)

See
[Astrangemaninhere/ninfer-5090-windows](https://github.com/Astrangemaninhere/ninfer-5090-windows)
(a fork of headpiece747's MSVC port, synced with this project's KV features). That is
a separate tree; it is not a build of this repository.

## Features

- **Per-layer KV storage** — `--kv-layer-storage SPEC` (`apps/cli/options.cpp:765`).
  Grammar `all:<type>` / `<type>` / `A:<type>` / `A-B:<type>`, with `<type>` one of
  `bf16, int8, fp8, nvfp4, iso4e, rk4v4, rk3v4, rk2v4` (`src/product/kv_options.h:9-22`).
  Unlisted slots stay BF16, i.e. "inherit `--kv-dtype`" (`:22-24`); the "was this slot
  written" mask is what keeps an explicit `all:bf16` distinct from an empty spec
  (`:26-30`).
- **E8Kv 4-bit KV** — E8-lattice K with an i4 V. The codec family and its plane
  geometry are `src/product/kv_e8_width.h`; the per-plane read side is a separate axis
  (`src/product/kv_cell_modes.h:667`).
- **NVFP4-tier KV** — `--kv-dtype nvfp4` (`apps/cli/options.cpp:552`) or a layer-table
  slot. E2M1 K with ISO3 V; the fusion/pure switch is `Nvfp4Mode`
  (`src/kvcfg/kv_formats.h:68`).
- **Cold pool** — `--cold-policy none|off|window|host|disk|host-then-disk`
  (`apps/cli/options.cpp:928`; usage `:399-401`). The aged-out tier's slot codec is
  derived from the layer dtype and only `int8` is reachable (`:355-358`).
- **NVMe cold tier** — `--cold-disk-path` / `--cold-disk-bytes` (usage `:399-408`):
  per-layer spill files. Off by default (`ColdPolicy::None`, `apps/cli/options.h:120`).
- **Speculative decoding** — `--spec auto|none|off|mtp|dflash|dflash2|dspark`
  (`apps/cli/options.cpp:879`); the name table is `parse_speculative_backend`
  (`src/product/speculative_options.h:11`).
- **Adaptive draft width** — `--spec mtp --draft-tokens 0` (`apps/cli/options.cpp:881`):
  the width ladder, chosen per round from a survival/cost rule
  (`src/targets/qwen3_6/impl/runtime/mtp_window_cut.h`).
- **MTP tree verification** — `--draft-tree L,d` (`apps/cli/options.cpp:892`): L
  rank-paths by d steps per round
  (`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h:99,100`).
- **DFlash2 tree/beam walk** — reached through `--spec dflash2`, with no separate flag:
  `include/ninfer/ops/dflash2_ddtree.h`, `dflash2_ddtree_beam.h`,
  `dflash2_tree_walk.h`.
- **CUDA graphs** — `--no-cuda-graph` (`apps/cli/options.cpp:1014`) and
  `--graph-capture-ceiling N` (`:1002`, usage `:411`): capture on/off, and the capture
  ladder's ceiling (default 16, `apps/cli/options.h:177`).
- **Bandwidth-governed prefill unit** — `--prefill-chunk-mode dynamic|manual`
  (`apps/cli/options.cpp:545`). The governor is
  `src/runtime/engine/bandwidth_governor.h`, and its env switch `NINFER_FT_BW_GOV` is
  read at `:250`.
- **KV bit-budget solve** — `--kv-bit-budget` / `--kv-bits` / `--kv-k-bits` with
  `--kv-v-bits` (`apps/cli/options.cpp:558,629,638,647`); the per-layer rung table is
  `src/product/kv_bit_budget.h` and `src/product/kv_kv_bits.h`.
- **KV paging preallocation** — env `NINFER_KV_PAGING_PREALLOC`, read at
  `src/targets/qwen3_6/impl/runtime/layouts_impl.h:2589`.
- **Weight offload** — `--weight-host-bytes`, `--weight-device-arena-bytes`,
  `--weight-prefetch-layers`, `--weight-span-floor-bytes`
  (`apps/cli/options.cpp:988,990,992,1000`); the budget is
  `src/product/weight_offload_budget.h`.
- **Row-scale calibration loop** — `--kv-row-scale auto|off|FILE` and `--recalibrate`
  (`apps/cli/options.cpp:839,848`): persistence and reuse of the KV row-scale table
  (`src/product/kv_rowscale_persist.h`).
- **YaRN static rope** — `--yarn` (`apps/cli/options.cpp:1016`, usage `:377`): the
  context domain is multiplied by four
  (`src/targets/qwen3_6/impl/runtime/layouts_impl.h:1333`). Off by default
  (`apps/cli/options.h:176`).
- **Plane discard** — env `NINFER_KV_DROP_LAYERS`: whole layers lose their KV planes
  (`src/targets/qwen3_6/impl/state/decoder_state.cpp:1281`, described at `:190`). A
  discarded layer is refused by name when storage is asked for it
  (`src/targets/qwen3_6/impl/runtime/layouts_impl.h:137`).
- **Auto prefix sharing** — serve default, with an opt-out flag (issue #142):
  `src/serve/serve_options.h:109`, `src/serve/openai_common.cpp:147`.
- **Auto KV relayout** — `--kv-auto-relayout SECS`, **serve only**
  (`src/serve/serve_options.cpp:582`; usage `:130`, `:214`).
- **Cross-rank / multi-device** — no switch. Host-only, with no call path:
  `src/core/shard_plan.h`, `src/core/tp_transport.h`; the contract is
  `docs/maintainer/multi-device-and-shard-plan.md`.

## Defaults

These are the shipped defaults of the two front ends: CLI defaults live in
`apps/cli/options.h`, serve-side ones in `src/serve/serve_options.h`.

- `kv_cache` (`--kv-dtype`) — **BFloat16** (`apps/cli/options.h:46`).
- `cold_policy` (`--cold-policy`) — **None** (`apps/cli/options.h:120`;
  `src/serve/serve_options.h:112`).
- `unload_watermark_pages` — **derive** (`kUnloadWatermarkDerive`,
  `src/targets/qwen3_6/impl/runtime/layouts.h:141` and `:210`). The usage text says
  `0` is off and the default derives the reserve from `--prefill-chunk`
  (`apps/cli/options.cpp:455-459`).
- `use_cuda_graph` — **true** (`apps/cli/options.h:53`; `src/serve/serve_options.h:106`).
- `spec` — **Auto** (`apps/cli/options.h:51`).
- `enable_thinking` — **true** (`apps/cli/options.h:217`;
  `src/serve/serve_options.h:150`).
- `kv_quality_weight` — **-1**, i.e. the slider is off (`apps/cli/options.h:69`;
  `src/serve/serve_options.h:60`).
- `graph_capture_ceiling` — **16** (`apps/cli/options.h:177`).

**A new mechanism's environment variable, when unset, is the pre-image.** This is a
property of the code, not of a run:

- The descent's OFF path is `budget_bytes <= 0`; it returns the default vector and the
  pre-image charge before any walk is entered — "OFF is byte-for-byte the pre-image"
  (`src/product/kv_block_descent.h:478-480`).
- The third axis' reader returns `0` for unset, empty **or any non-digit character**,
  and `0` is the pre-image shape (`src/product/kv_block_descent.h:1460-1463`). The
  pool's own `page_group_count = 0` means "this plane uses the pool's page count",
  i.e. the plane every existing run measured
  (`src/core/paged_kv_cache.h:149-150`, `:157-158`).

So **an unmodified run of this tree is a run of the original path**, plus the switches
you actually set.

## Block-KV machinery (the current work)

The block-KV machinery is **per cell**: one cell is one (64-token block, text layer)
pair, and a cell moves one rung at a time.

- **The rate ruler.** The budget is given in bits per element, through
  `NINFER_KV_BLOCK_BUDGET_RATE_X10000` — the key is defined at
  `src/product/kv_block_budget_stage.h:233` and read at `:362`. The absolute
  spellings are `NINFER_KV_BLOCK_BUDGET_BYTES` (`:176`) and `..._BLOCKS` (`:185`).
- **The trigger is derived from the rate gap, per pass.** `steps_per_call` is computed
  from `budget_bytes` against `total_bytes`
  (`src/product/kv_block_descent.h:764-788`).
- **Carry is re-anchored by page identity** —
  `carry_reanchor[kept=.. dropped=.. new=..]`
  (`src/product/kv_block_descent.h:719`, `:937`).
- **Cap/pass separation in the unload trigger.** `pass_holds`
  (`src/targets/qwen3_6/impl/runtime/program_impl.h:13627`): above the watermark the
  pass runs and prints (`reason=above-watermark`, `:14060`).
- **The third axis: a plane may be sized for its own class' pages.**
  `KVPlaneGeometry::page_group_count` (`src/core/paged_kv_cache.h:157`), knob
  `NINFER_KV_AXIS3_NARROW_PAGES` (`src/product/kv_block_descent.h:1458`), one reader
  (`:1463`) and two consumers
  (`src/targets/qwen3_6/impl/runtime/program_impl.h:13812`,
  `src/targets/qwen3_6/impl/state/decoder_state.cpp:1291`).
- **K and V are an independent pair.** `CellPair` / `cell_pair_bytes` /
  `cell_pair_read_side` (`src/product/kv_cell_modes.h:517`, `:527`, `:667`), with
  arbitrary-pair pricing `e8_kv_pair_bytes(K, V)` (`src/product/kv_e8_width.h:284`).
- **Cold tiers.** The `--cold-policy` family; the per-layer host tier is
  `src/targets/qwen3_6/impl/runtime/cold_host_tier.h`.

Two things this machinery does **not** claim:

- The third axis is a **reservation**, not a used region: the narrow class the layout
  reserves has no read-side consumer. In-tree: the `Reserved` read side at
  `src/product/kv_cell_modes.h:666-685`.
- The `e8-2bit` / `(B2,B2)` pairs are **priced** but the read side for them is
  `Reserved` (`src/product/kv_cell_modes.h:676`), and
  `e8_kv_lattice_decode_group<3>/<2>` has no caller
  (`src/product/kv_storage_dtype.h:90-91`, `:116`; `src/product/kv_block_descent.h:1425`).

## Exact by construction (the figures this file keeps)

Each line below is pinned by a `static_assert` in the tree: the value is a property of
the definition, not of a run.

- E8 K plane, W4 / W3 / W2 — 8704 / 6656 / 4608 B
  (`src/product/kv_e8_width.h:183,185,187`).
- E8 V plane, the shipped i4 plate shared by all three widths — 8704 B
  (`src/product/kv_e8_width.h:148,157`).
- E8 layer (K + V), W4 / W3 / W2 — 17408 / 15360 / 13312 B
  (`src/product/kv_e8_width.h:189,190,191`).
- `e8_kv_pair_bytes(K, V)`, the four asserted pairs: `(B4,B4)=17408`,
  `(B4,B3)=15360`, `(B2,B4)=13312`, `(B2,B2)=9216`
  (`src/product/kv_e8_width.h:319,321,333,491,494`).
- `(rk4v4, rk4v4)` = `(B4,B4)`, the shipped pair — 17,408 B
  (`src/product/kv_cell_modes.h:577`).
- `(rk3v4, rk4v4)` = `(B3,B4)` — 15,360 B (`src/product/kv_cell_modes.h:580`).
- `(e8-2bit, rk4v4)` = **`rk2v4`** = `(B2,B4)`: K at 2 bits, **V still on the i4
  plate** — **13,312 B/cell** (`src/product/kv_cell_modes.h:585`).
- `(e8-2bit, e8-2bit)` = `(B2,B2)`, the symmetric floor — **9,216 B/cell**
  (`src/product/kv_cell_modes.h:589`).
- The difference between those two pairs is one plane's `B4 - B2`; asserted at
  `src/product/kv_cell_modes.h:597`.
- KVarN record, payload and record the same width — 26880 B
  (`include/ninfer/ops/kvarn.h:36,37`).
- E8 bits per element (x100), W4 / W3 / W2 — 425 / 375 / 325
  (`src/product/kv_e8_width.h:193,194,195`).

**Read 13,312 and 9,216 as prices, not as capabilities.** `rk3v4` and `rk2v4` are in
the vocabulary and are selectable, but they are **not runnable**: no decode or append
kernel in this tree reads a 3-bit or 2-bit K code plate, so a plan that resolves to one
of them is refused by name at the resolver instead of being run through the 4-bit
reader (`src/product/kv_options.h:52-58`). The same holds for the symmetric `(B2,B2)`
pair.

## Known limitations

- **Residual planes** (`--kv-residual-layers`, case `apps/cli/options.cpp:790`, usage
  `:291`): parsed by both front ends and not on the default path. Its status is tracked
  by name in [`docs/features/unfinished.md`](docs/features/unfinished.md). This file no
  longer restates a defect claim about it — that claim came without a coordinate in
  this tree and now lives in the withdrawal archive, section 4.5.
- **Cold pool reach**: the cold slot codec is derived from the layer dtype and only
  `int8` is reachable (`apps/cli/options.cpp:355-358`), so E8Kv and NVFP4 layers are
  skipped by name rather than mishandled.
- **MTP KV has no per-layer table**: MTP layers carry no per-layer table, so the layer
  table does not reach them
  (`src/targets/qwen3_6/impl/state/decoder_state.cpp:1294`).
- **Multi-device is host-only**: the shard arithmetic exists, the call path does not
  (`CHANGELOG.md` records "host-only and has no call path";
  [`docs/features/unfinished.md`](docs/features/unfinished.md) item T19).
- **Line numbers drift.** Every `:line` above is a reading of the tree at
  `de80bd1f56b789ad5415c187721815ad1f6a963a`. Re-measure before quoting; the discipline
  is in [`docs/features/verification.md`](docs/features/verification.md) section 6.3.

## Open problems (not hidden)

Each item states **what it is** and **where the evidence lives**. No figures are quoted
here; follow the path to read them.

1. **Prefill against the unmodified engine is slower** — the one open acceptance
   blocker. The no-regression rule is "faster or equal, never slower", and the
   mechanism behind the gap is **not yet identified**. Evidence: `dl/kvspend/`,
   `dl/kvaxisA/EVIDENCE.txt` section 6, `dl/cmpfire3/blob_F1250.md`.
2. **Long context erodes the demotion mixture** — the demoted share falls as the
   context grows, a single retirement can wipe one arm's demotions, and the walk
   under-reports its own demotions. Evidence: `dl/kvplanar/EVIDENCE.txt`,
   `dl/kvplanar2/out/PLAN_EVIDENCE.txt`, `dl/kvrate2/`, `dl/kvaxis/`, `dl/cmpfire3/`.
3. **The decode column cannot be used as acceptance** — its measured instrument noise
   is larger than any effect this machinery produces. Evidence: `dl/cmpfire3/`
   (F-1250), quoted in `dl/kvaxisA/EVIDENCE.txt` section 6.
4. **The narrow region is reserved but unfilled** — the read side (per-page class
   addressing, the int8 requant driver, the block-table class label) is the missing
   link. Evidence: `dl/kvfill/STATE.md`, `dl/kvfill2/`, `dl/kvplanar/`; in-tree, the
   `Reserved` read side at `src/product/kv_cell_modes.h:666-685`.
5. **Cold refetch has never been exercised** — the census instrument says in its own
   header that a run in which nothing is refetched prints nothing at all, so "zero
   refetch" is today a sentence in a comment rather than a measurement. Evidence:
   `src/targets/qwen3_6/impl/runtime/cold_refetch_census.h:16-66`; archives
   `dl/kvmemrun/`, `dl/kvmemfix/`.
6. **1M context is blocked by three separate rings** — the native context gate (each
   variant's own `maximum_context`, `src/targets/*/impl/variant.h`; `--yarn`
   multiplies it by four, `src/targets/qwen3_6/impl/runtime/layouts_impl.h:1333`), a
   device shortfall, and a cold-tier deficit. Evidence: `dl/kv1m/EVIDENCE.txt`,
   `dl/long1m/logs/`, `dl/combo1m/REPORT.md`, `dl/1mmtp/`, `dl/ctxsweep/`.
7. **`NINFER_KV_QUALITY_WEIGHT` is not wired into the per-cell walk.** It acts only on
   the ceiling/split solver: the key is defined at `src/product/kv_kv_bits.h:446` and
   read at `:488`, and it has **no reference at all** in the per-cell walk's files
   (`src/product/kv_descent_control.h`, `src/product/kv_block_descent.h`,
   `src/product/kv_cell_alloc_solve.h`). The knob map below says so in its own row.
8. **Two mechanisms collapse to their pre-image today, by construction.**
   `NINFER_KV_DESCENT_ALLOC=solve` selects the stateless per-cell solve, but there is
   no cost-table producer, so it collapses (`src/product/kv_cell_alloc_solve.h:1102`);
   and `NINFER_KV_BUDGET_RULER_F1231` is a **compile-time `#define`**, not a knob
   (`src/product/kv_block_budget_stage.h:240`).

## Knob map for the new machinery

Every knob below is unset by default, and unset means the pre-image (see *Defaults*).

- `NINFER_KV_BLOCK_BUDGET_RATE_X10000` — the **rate** budget: bits per element this
  pass must reach.
- `NINFER_KV_BLOCK_BUDGET_BYTES` / `NINFER_KV_BLOCK_BUDGET_BLOCKS` — the same budget in
  absolute bytes / charge-blocks.
- `NINFER_KV_DESCENT_CHAIN` — the rung chain (`lattice` for the 4/3/2-bit ladder).
- `NINFER_KV_DESCENT_MAX_TIER` — how deep any cell may go (`0` = nothing may move).
- `NINFER_KV_DESCENT_ALLOC` — `solve` selects the stateless per-cell solve, which
  **collapses to the pre-image today: no cost-table producer exists**.
- `NINFER_KV_DESCENT_KEEP_RECENT_PAGES` — the age gate's K (the hot window stays int8).
- `NINFER_KV_AXIS3_NARROW_PAGES` — the third axis: pages per layer sized for the narrow
  class.
- `NINFER_KV_UNLOAD_WATERMARK_PAGES` — the unload **pass**; `0` is OFF, unset means
  "derive from `--prefill-chunk`".
- `NINFER_KV_QUALITY_WEIGHT` — the speed/quality slider of the **ceiling/split solver**
  only; it is NOT wired into the per-cell walk.

**One name in the old knob map was not a knob.** `NINFER_KV_BUDGET_RULER_F1231` is a
compile-time `#define` (`src/product/kv_block_budget_stage.h:240`), so it has been
removed from this list.

## Tools

- `tools/gui/serve_gui.py` — slider-minimal serve console (default port 8789, `:956`).
- `tools/gui/convert_gui.py` — slider-minimal model converter (port 8788, `:278`).
- `tools/gui/rag_gui.py` — slider-minimal RAG search (port 8787, `:277`).

## License

Apache-2.0.

The E8 codec lineage and the IsoQuant tables this pack carries are attributed where the
tree states them, and those are the coordinates to cite:

- E8 codec lineage (upstream `ninfer-4090` / `ninfer-3090`):
  `src/ops/kernel/e8_root_codec.cuh:2` and `:274`;
  `src/ops/softmax_attention/dense/causal_cache/small_t_nvfp4.cuh:2`.
- The 64-dim Sylvester rotation, `PR #35 lineage`:
  `src/ops/kernel/gqa_attention_kv_quant.cuh:20`.
- IsoQuant tables from the `nvfp4rtx` offline calibration:
  `src/ops/kernel/gqa_isoquant_rot.cu:12` and `gqa_isoquant_rot.cuh:4`.

There is **no `THIRD-PARTY.md` in this tree** — the previous revision of this file
linked one that does not exist. The attribution above replaces it.

## Where the withdrawn figures went

Every figure this file used to carry — the perplexity/prefill table, the capacity
table, the VRAM reference table, the third-axis memory readings, the cold-spill
readings, the prefill deltas, the erosion shares, the decode noise floor and the 1M
rings — is preserved **verbatim** in
[`docs/features/withdrawn-numbers-2026-09-30.md`](docs/features/withdrawn-numbers-2026-09-30.md),
each row with its withdrawal reason (one of four classes) and either its evidence path
in `dl/<line>/` or the words "evidence not found". Nothing was deleted; it was moved
out of the reading surface and into an archive.
