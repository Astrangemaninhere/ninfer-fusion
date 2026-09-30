# NInfer Fusion (Community Enhancement Pack)

An enhanced distribution of [Neroued/ninfer](https://github.com/Neroued/ninfer)
(Apache-2.0) that merges KV compression, cold tiers, speculative decoding and
long-context features into a single maintainable patch set, targeting the
RTX 5090 / sm_120a.

> All performance/precision figures below are measured on our RTX 5090D +
> WSL2 (CUDA 13.3, sm_120a) setup.

## Measured results (Qwen3.8-27B NVFP4, 4096 ctx, 13.3k zh corpus)

### KV schemes — ppl + prefill (ninfer-perplexity)

| config | ppl | NLL | prefill tok/s |
|---|---|---|---|
| **10L E8 + 6L NVFP4 (default)** | **1.0202** | 0.020 | **2027.7** |
| all-E8 (16 layers E8-lattice) | 1.1120 | 0.106 | 2082.7 |
| all-NVFP4 (E2M1 K + ISO3 V) | 1.7055 | 0.534 | 1949.1 |
| all-int8 (reference) | 1.5217 | 0.420 | 2017.2 |
| old default (10L NVFP4 + 6L I8) | 1.2962 | 0.259 | — |

The default 10L E8 + 6L NVFP4 mix wins on both precision and speed: the
E8-lattice K provides the lattice gain (H64 rotation aligned to the g64 scale
domain), while the NVFP4 layers' ISO3 V fits the value distribution better
than i4.

### Capacity (per head/token)

| scheme | bytes | bit/element |
|---|---|---|
| E8Kv (E8 K + i4 V, g64) | ≈260 B | ≈4.06 |
| NVFP4 (E2M1 K + ISO3 V, g16) | 288 B | 4.50 |
| int8 (reference) | 512 B | 8.00 |

### VRAM reference (RTX 5090D 32 GB, serve, 4096 ctx)

| config | VRAM |
|---|---|
| idle | 0.05 GB |
| base (int8 kv) | 20.5 GB |
| E8-mix / all-NVFP4 kv | 20.6 GB |
| + vision | 20.9 GB |
| + MTP3 | 21.4 GB |
| DFlash2 model | 20.6 GB |

Full table: [VRAM.md](VRAM.md).

## Features

| feature | switch | notes |
|---|---|---|
| Per-layer KV storage | `--kv-layer-storage` | per-layer BF16/INT8/NVFP4/E8Kv mix |
| E8Kv 4-bit KV | `--kv-layer-storage all:e8` | E8-lattice K + i4 V (H64) |
| NVFP4-tier KV | `--kv-dtype nvfp4` / layer table | E2M1 K + ISO3 V, native mxf4nvf4 QK |
| Entropy cold pool | `--cold-policy window` | rANS slots (I8 layers) |
| NVMe cold tier | `--cold-policy disk` | per-layer spill files (default off) |
| DFlash2 draft | `--spec dflash2` | bidirectional-attention draft |
| On-demand graphs | `--graph-capture-ceiling N` | decode ladder extension |
| YaRN factor-4 | `--yarn` | static 4x context |
| Auto prefix sharing | serve default | issue #142, opt-out flag |

## Known limitations

- Residual planes (`--kv-residual-layers`): block_tables corruption not yet
  located (experimental, off by default).
- Cold pool covers I8 layers only (E8Kv/NVFP4 safely skipped).
- MTP KV uses the global dtype (layer table does not affect MTP).

## Build

### WSL2 (primary)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j
```

### Native Windows (adaptation branch)

See [Astrangemaninhere/ninfer-5090-windows](https://github.com/Astrangemaninhere/ninfer-5090-windows)
(fork of headpiece747's MSVC port, synced with this project's KV features).

## Tools

- `tools/gui/serve_gui.py` — slider-minimal serve console (port 8789)
- `tools/gui/convert_gui.py` — slider-minimal model converter (port 8788)
- `tools/gui/rag_gui.py` — slider-minimal RAG search (port 8787)

## License

Apache-2.0. Third-party contributions and attribution: see
[THIRD-PARTY.md](THIRD-PARTY.md) — E8 codec lineage (PR #35 →
ninfer-4090 → ninfer-3090), IsoQuant tables (nvfp4rtx calibration).

---

## What this pack is working on (2026-09-30)

This section records the KV-allocation plan and its **current** state. Every claim below is a
reading from this working tree; nothing here is a promise. The block-KV machinery is per-cell:
one cell is one (64-token block, text layer) pair, and a cell moves one rung at a time.

### Landed and measured

| item | what it is | reading |
|---|---|---|
| Per-cell demotion with a **rate** ruler | `NINFER_KV_BLOCK_BUDGET_RATE_X10000` (`src/product/kv_block_budget_stage.h:362` reader) | 96.94% of cells movable; plan coin 4.372266 b/el (47% fewer bytes) |
| Trigger derived from the rate gap | `steps_per_call` computed per pass from `budget_bytes` vs `total_bytes` (`src/product/kv_block_descent.h`) | `TOTALS-DISAGREE` 26/27 passes -> 0/27 |
| Carry re-anchored by **page identity** | `carry_reanchor[kept=.. dropped=.. new=..]` | demotions survive a moving window |
| Cap/pass separation in the unload trigger | `program_impl.h:13411` (`pass_holds`) | above the watermark the pass now **runs and prints** (`reason=above-watermark`) |
| **Third axis**: a plane may be sized for its own class' pages | `KVPlaneGeometry::page_group_count`; knob `NINFER_KV_AXIS3_NARROW_PAGES` | `kv cache payload` 4.38 -> 3.56 GiB, `gpu sequence used` 5.71 -> 4.88 GiB, plan coin byte-identical (884,736,000 B), prefill +0.013% (indistinguishable) |
| K/V as an independent **pair** | `CellPair` / `cell_pair_bytes` / `cell_pair_read_side` (`src/product/kv_e8_width.h`) | charge and order are pair-true; arbitrary-pair pricing `e8_kv_pair_bytes(K,V)` exists (B4=8704, B3=6656, B2=4608 per plane) |
| Cold tiers (KVMem) | `--cold-policy window|host|disk|host-then-disk` | disk spill measured: 1,198,443,776 B, `write_failures=0`, `drain_ms=933`; **cold refetch is still never exercised** |

### Being worked on now

1. **Fill the narrow region** (`kvfill`): the third axis above is a **reservation**, not yet a used
   region; the read side (per-page class addressing + the int8 requant driver + the block-table
   class label) is the missing link. Acceptance is the engine's own device readings under pressure.
2. **E8 on the V plane**: today the deployed narrow form keeps V at the 128-byte i4 plate, so the
   `e8-2bit` pair costs **13,312 B/cell**; with V able to carry E8 the symmetric pair is
   **9,216 B/cell** (-30.8%). The geometry and the arbitrary-pair pricing already exist; the
   **decode-side consumer does not** (`e8_kv_lattice_decode_group<2>` has no caller).
3. **KVMem x text-prefill cooperation**: how the offload legs behave inside the prefill chunk loop,
   and what that costs per length.
4. **An un-borrowed baseline**: quality / speed / both device currencies at 8k / 64k / 128k, so any
   later mechanism is measured against our own numbers first.
5. Mechanisms evaluated by **value after overcoming the obstacles** (see `docs/features/`): a
   runtime KLT basis + reverse water-filling, a choosable entropy container for the E8 plane, and
   KVarN's channel axis. Methods whose honest value is zero are recorded as **zero**, not as "hard".

### Known standing problems (not hidden)

- Prefill against the unmodified engine is **lower** by 4.11% / 5.31% / 5.54% in three independent
  readings with disjoint intervals. The no-regression rule is "faster or equal, never slower", so
  this is the one open acceptance blocker, and its mechanism is **not yet identified**.
- Long context erodes the mixing: the demoted share is 9.32% at 8k, 2.68% at 64k and 0.60% at 128k;
  at 128k a single `retired=1024` wipes demotions (`rk4v4 672 -> 16`), and the walk under-reports its
  own demotions by 2.05x-2.41x.
- The **decode** column cannot be used as acceptance: its measured instrument noise is **30.7%**.
- 1M context is blocked by three separate rings: the native context gate (262,144, `--yarn` for 4x),
  a device shortfall of 10.43 GiB, and a cold-tier deficit of 8,457 pages (8.86 GiB) against a
  longest actually-run context of 260,096 tokens.

### Knob map for the new machinery

| knob | what it selects |
|---|---|
| `NINFER_KV_BLOCK_BUDGET_RATE_X10000` | the **rate** budget (bits per element this pass must reach) |
| `NINFER_KV_BLOCK_BUDGET_BYTES` / `_BLOCKS` | the same budget in absolute bytes / charge-blocks |
| `NINFER_KV_BUDGET_RULER_F1231` | which ruler an absolute budget is read in |
| `NINFER_KV_DESCENT_CHAIN` | the rung chain (`lattice` for the 4/3/2-bit ladder) |
| `NINFER_KV_DESCENT_MAX_TIER` | how deep any cell may go (0 = nothing may move) |
| `NINFER_KV_DESCENT_ALLOC` | `solve` selects the stateless per-cell solve (**collapses to the pre-image today: no cost-table producer exists**) |
| `NINFER_KV_DESCENT_KEEP_RECENT_PAGES` | the age gate's K (the hot window stays int8) |
| `NINFER_KV_AXIS3_NARROW_PAGES` | the third axis: pages per layer sized for the narrow class |
| `NINFER_KV_UNLOAD_WATERMARK_PAGES` | the unload **pass**; `0` is OFF, unset means "derive from `--prefill-chunk`" |
| `NINFER_KV_QUALITY_WEIGHT` | the speed/quality slider of the **ceiling/split solver** only - it is NOT wired into the per-cell walk |
