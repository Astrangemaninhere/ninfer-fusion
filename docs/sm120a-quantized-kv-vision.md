<!--
PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
Source repo : practicorecoza/ninfer-engine
Branch      : sm120a-quantized-kv-vision
Commit      : 3a0604de0967942d95f8b061c86dc17a25a7d460
Source path : docs/sm120a-quantized-kv-vision.md
sha256(src) : 1c9e246ac3dbe1f97c80a7e209f911267775529277abec2fba10c2bf48be8fda
Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits (rk4v4-e8 = rotated K 4b, V 4b, e8 in the mix), which is a DIFFERENT axis from src/kvcfg/kv_formats.h's tier vocabulary (hot/tail/cold, E8).
-->
# SM120 quantized KV and vision

This fork preserves NInfer's native Blackwell `sm_120a` execution path while adding low-memory KV
cache layouts and a bounded vision workspace. It exists to run the Qwen3.8-27B groupwise-int
artifact with vision, MTP3 and long context inside a 21 GiB NInfer VRAM budget.

## Source lineage

The integration starts from upstream [`Neroued/ninfer`](https://github.com/Neroued/ninfer) commit
`feaf4dd0983fdaeb2ba4c06eec6da350e644fb3a`. Quantized-KV and vision-budget work was selectively
ported from [`UDPSendToFailed/ninfer-4090`](https://github.com/UDPSendToFailed/ninfer-4090), rather
than using that fork wholesale. Its W8 execution leaves target Ada and do not compile as a valid
`sm_120a` replacement.

| This branch | Reference commit | Change |
|---|---|---|
| `322f3dc` | `0a39efe3` | paged `rk8v4` foundation |
| `dd0f6d1` | `0701b973` | rotated 4-bit keys and values (`rk4v4`) |
| `ee57a7a` | `46116bf2` | E8 lattice keys and 4-bit values (`rk4v4-e8`) |
| `7f69d71` | `e4616151` | E8 cylinder/root keys and 4-bit values (`rk2v4-e8`) |
| `0869973` | `6d3fd165` | `--vision-max-tokens` workspace sizing and CLI |
| `89646ec` | `5c04b765` | apply the same vision limit in frontend processing |
| `775538c` | local | preserve configurable vision budget in test access |

Conflicts were resolved in favor of upstream's Blackwell CUDA Graph accounting and current bounded
media cache, live-byte and worker controls. The imported code contributes the KV codecs, quantized
GQA kernels, runtime layouts, option parsing and focused tests without replacing Blackwell linear
execution leaves.

## Runtime options

`--kv-dtype` accepts the upstream `bf16` and `int8` modes plus:

- `rk8v4`: rotated INT8 keys and packed 4-bit values;
- `rk4v4`: rotated packed 4-bit keys and values;
- `rk4v4-e8`: E8 lattice 4-bit keys and packed 4-bit values;
- `rk2v4-e8`: E8 cylinder/root keys and packed 4-bit values.

Vision remains opt-in with `--vision`. `--vision-max-tokens N` (alias `--vision-limit N`) bounds both
the frontend media-token budget and reserved encoder workspace. Its default is 8,192 instead of the
previous fixed 32,768-token workspace.

The production profile used by the companion deployment repository is:

```bash
ninfer-serve qwen3_8_27b.ninfer \
  --kv-dtype rk4v4-e8 \
  --max-context 145920 \
  --kv-capacity 145920 \
  --max-concurrency 2 \
  --spec mtp \
  --draft-tokens 3 \
  --lm-head-draft \
  --vision \
  --vision-max-tokens 8192
```

The context and KV capacity are startup-fixed. Changing them or the vision budget changes runtime
reservation only and does not require rebuilding NInfer.

## Build

Build for Blackwell with CUDA 13.1 or newer:

```bash
cmake -S . -B build-sm120a -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=OFF \
  -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-sm120a -j
```

The validated model-free runtime image was built from code commit `775538c` and published as:

```text
ghcr.io/kobusg/ninfer-sm120a-quant-kv@sha256:c9285445c49e655d0363268e6bcc21cd52b1b4283e7fca2da6c864d4d350cb57
```

Models are not included in the image. The deployment downloads and verifies the registered
`.ninfer` artifact at runtime.

## Validation

The profile above was measured on an RTX PRO 5000 Blackwell:

- idle NInfer residency: 21,502 MiB;
- peak during a real image request: 21,504 MiB;
- 135,311-token multimodal needle retrieval: exact image title and buried code;
- C1 decode: 103.0 tok/s;
- C2 aggregate decode: 129.3 tok/s;
- MTP acceptance: 59.6-63.9%.

The same-host INT8/MTP3 control used 18,812 MiB at 16K context. `rk4v4-e8` used 18,542 MiB and had
equivalent decode throughput within run variance. The larger production reservation spends the
savings on context while retaining the 21 GiB ceiling.

These measurements apply to `sm_120a`. This branch is not an RTX 6000 Ada (`sm_89`) or RTX A6000
build.

## Maintenance

Keep `sm120a-quantized-kv-vision` as the reproducible integration branch and preserve tested image
source commits. For a future upstream version, branch from the new upstream commit, replay the
focused commits above, resolve Blackwell runtime conflicts explicitly, build locally, and validate
on a short GPU rental before publishing a new immutable image digest. Do not move an existing image
tag or rewrite a source commit associated with measured results.
