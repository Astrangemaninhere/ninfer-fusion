<!-- Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion). -->
# Qwen3.8-Flash-Next artifact contract

This reference defines the NInfer artifact built for the single-RTX-PRO-6000 Flash-Next target.
Generic framing, numeric semantics, and physical layouts are defined by
[`artifact-container.md`](artifact-container.md), [`tensor-formats.md`](tensor-formats.md), and
[`storage-layouts.md`](storage-layouts.md). Model mathematics are defined by
[`qwen3.8-flash-next-model.md`](qwen3.8-flash-next-model.md).

## Identity and provenance

```text
filename   = qwen3_8_flash_next_mixed.ninfer
model_id   = qwen3.8-flash-next
weights_id = mixed-nvfp4-fp8-ple-int4
target_key = qwen3_8_flash_next
recipe_id  = qwen3_8_flash_next_mixed-v1
```

The converter consumes only these pinned sources:

- `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8` at
  `a4e813ed3cfbbcc61e2929699eccb864a4dfa843`, excluding its BF16 PLE table;
- `primitive-ai/Qwen3.8-Flash-Next-PLE-quant` at
  `da8b39586016d8325ac619be28ad77d6296625ec`, using only `ples_int4`;
- the six frontend resources verified byte-identical to the official repository;
- the official Qwen Community License Agreement 1.0, retained beside the local artifact.

This is a converter splice, not a Transformers-loadable checkpoint directory. It replaces exactly
128 BF16 PLE tensors and consumes all other 296502 source tensors assigned by the recipe.

## Closed object inventory

The artifact contains 1566 objects: six raw frontend resources and 1560 tensors.

| Format | Layout | Objects | Placement with all features |
|---|---|---:|---|
| `BF16` | `contiguous-le-v1` | 1237 | Device |
| `FP8_E4M3FN_ROW_F32S` | `row-scale-f32-v1` | 96 | Device |
| `NVFP4` | `expert-blockscale-k16-m128x4-v1` | 96 | Device |
| `I64` | `contiguous-le-v1` | 3 | mapped host metadata |
| `U4Z8G16_F16S` | `packed-u4-g16-v1` | 128 | mapped host PLE |
| resource | `raw-bytes-v1` | 6 | retained host bytes |

With Text, MTP, and Vision enabled, the exact binder consumes every object and plans 1429 device
objects, 131 mapped tensors, and six host resources. Device materialization capacity is
81285117440 bytes (75.70 GiB). The mapped PLE plus metadata is 29.80 GiB. The complete local file is
113298397952 bytes (105.52 GiB); it is intentionally not all-resident in VRAM.

MTP and Vision objects remain mandatory artifact members. Disabling either feature validates its
descriptors without materializing them; it does not define another artifact identity.

## Text storage boundary

Every one of 48 layers stores four BF16 tensors for the attention hyper-connection, five BF16
router/shared-expert tensors, two expert-major banks, four BF16 tensors for the MoE
hyper-connection, and seven attention- or GDN-specific tensors.

The two expert banks per layer are:

```text
text/layers/{l}/mlp/experts/gate_up [512,1280,2560] NVFP4
text/layers/{l}/mlp/experts/down    [512,2560,640]  NVFP4
```

Each expert owns packed E2M1 codes, K16 E4M3FN block scales swizzled for the Blackwell
`M128x4` consumer, and one FP32 weight-scale divisor. The gate/up rows are `[gate_640,up_640]`.

Full-attention layers store FP8 projections:

```text
attention/query_gate_key_value [13312,2560]
attention/output               [2560,6144]
```

Within each of 24 query heads the first parent preserves `[query_256,output_gate_256]`, followed by
512 key rows and 512 value rows. Each E4M3FN row owns one FP32 multiplier.

GDN layers store FP8 `[q,k,v,z] [16384,2560]` and output `[2560,6144]` parents. BF16 tails store the
four-tap convolution, `[A,B] [96,2560]`, `A_log`, `dt_bias`, and gated norm.

The global BF16 token embedding and output head are independent `[248320,2560]` objects. The final
hyper mixer is three BF16 tensors.

## PLE mapped layout

PLE support weights at `text/layers/1/ple/` remain device BF16. Its three exact int64 metadata
tensors and 128 packed table shards remain file-mapped for the lifetime of the loaded model.

Each shard is `[2500012,160]`. For a row, 80 low-nibble-first U4Z8 bytes are read from the shard's
code plane and ten FP16 group multipliers are read from its separately aligned scale plane. A value
is `(code - 8) * scale`. Runtime gathers only the 16 rows selected for a token; artifact
materialization must not pre-touch or copy the complete PLE table.

## MTP, Vision, and frontend

MTP has 29 BF16 objects, including dense `[512,...]` expert banks and one full-attention QSA layer.
Vision has 333 BF16 objects covering the patch stem, 27 blocks, and merger. The six retained
resources are tokenizer JSON, tokenizer configuration, chat template, generation configuration,
image preprocessor configuration, and video preprocessor configuration.

The active binder lives under `src/targets/qwen3_8_flash_next`; it checks every name, shape, format,
layout, and placement and fails if any artifact directory object is missing, extra, or consumed
twice. Target registration remains prohibited until the complete Program can execute this contract.

