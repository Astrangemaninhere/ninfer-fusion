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
