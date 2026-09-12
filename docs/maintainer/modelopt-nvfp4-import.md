# Importing a ModelOpt NVFP4 checkpoint into a `.ninfer` artifact

This note covers only what is **not** already defined elsewhere:

* the persist/format contract itself lives in
  [`tensor-formats.md`](tensor-formats.md) (formats, semantics) and
  [`storage-layouts.md`](storage-layouts.md) (byte geometry, plane offsets, row
  views).  Read those first; nothing here restates them.
* what this note adds is the **source-flavor mapping** for a ModelOpt NVFP4
  export, the **encoder profile** it needs for the one format whose scale
  assignment the spec deliberately leaves open, and the two places where a
  faithful import is provably impossible.

Reference material: `/home/user/models/q3nvfp4` (source),
`/home/user/models/qwen3_8_27b_nvfp4.ninfer` (released artifact, "the oracle"),
`/home/user/models/q38_abl_huihui_nvfp4/model-mtp-bf16.safetensors` (MTP head).

## 1. The gap that makes an arbitrary source hard

`tools/convert/common/quantize.py` provides quantizers for the grouped
signed-integer family (`Q4G64`, `Q5G64`, `Q6G64`, `W8G32`) and nothing else.  The
two other weight formats have encoders that require finished words:

* `encode_nvfp4(packed_codes, natural_scales, weight_divisor, shape)` - and
  `tensor-formats.md` §3.3 already says the recipe **copies** those three fields
  from its source without requantizing, so no quantizer is wanted;
* `encode_fp8_row_scaled(code_words, row_scales, shape)` - and §3.4 says the
  format **does not define** how a source is assigned a scale or rounded, and that
  a recipe must therefore either preserve words exactly or **name its encoder
  profile**.

That asymmetry is the reason the shipped conversion path is dual-source: an
unquantized checkpoint supplies what must be quantized from BF16, a quantized one
supplies what can only be copied.  Importing a source that is not already in the
FP8 row-scaled form therefore means *defining an encoder profile* for it, not
adding a heuristic - and the definition has to be an exact inverse of the released
encoding.

## 2. Encoder profile for `FP8_E4M3FN_ROW_BF16S` from this source family

The profile was recovered by inverting the oracle (decode with the engine's own
`decode_fp8_row_scaled_words`, recompute the row maximum from the dequantized
values, test candidate rules) and then verified against it:

* `oracle_scale / (amax_row / 448)` was **exactly 1.000000 for all 14336 rows** of
  `text/layers/3/attention/query_key_gate_value`; `|ratio - 1| < 1e-3` held for
  **100%** of rows and `log2(ratio)` was an exact integer multiple, which rules out
  power-of-two snapping;
* `scale == bf16(fp32(amax_row / 448))` held for **100%** of rows.

Profile: `row_scale = bf16(fp32(amax_row / 448))` with the same canonical rounding
discipline as the grouped quantizer (binary64 division, explicit binary32 step,
then binary16), `code = nearest_E4M3FN(x / row_scale)`.  Implemented as
`quantize_fp8_row_matrix` / `quantize_and_encode_fp8_row` in
`tools/convert/common/quantize.py`.

Verification: re-quantizing the dequantized values of five oracle objects
reproduced the oracle's **scales and codes byte-for-byte**, 1.000000 in every case
(`text/layers/3/attention/query_key_gate_value` 14336x5120,
`text/layers/3/attention/output` 5120x6144,
`text/layers/0/gdn/query_key_value_z` 16384x5120, `text/layers/0/gdn/output`,
`text/output_head` 248320x5120), with `max |log2(scale_mine / scale_oracle)| = 0`.

A structural invariant follows and is asserted in the tests: for every non-zero row
the largest **magnitude** code word is exactly 448 (`0x7E`), because `amax/scale`
lies in [432, 448] and BF16 rounding cannot move it below 432.

## 3. ModelOpt field mapping (measured)

Every quantized Linear carries exactly four tensors:

| ModelOpt key | dtype | shape | maps to |
|---|---|---|---|
| `<key>.weight` | U8 | `(N, K/2)` | `packed_codes`, copied verbatim |
| `<key>.weight_scale` | F8_E4M3 | `(N, K/16)` | `natural_scales`, copied verbatim, **stored naturally (not swizzled)** - the swizzle is applied on write by `encode_nvfp4` |
| `<key>.weight_scale_2` | F32 | scalar | `weight_divisor = fp32(1 / weight_scale_2)`, because §3.3's `W = ... / d_w` divides while ModelOpt multiplies |
| `<key>.input_scale` | F32 | scalar | a **separate model-role tensor**, see §5 |

Measured on `text/layers/3/mlp/gate_up`: the oracle's packed codes and natural
scales are **byte-identical** to `q3nvfp4`'s `gate_proj` (upper rows) and `up_proj`
(lower rows), so the nibble order matches ModelOpt exactly and a literal copy is
valid; the fused row order is `gate ‖ up`.  The divisor relation holds exactly
(oracle word `15463.551` against `1 / 6.4668202e-05 = 15463.55`, product
`1.0000000249`), and using multiplication instead of division gives a mean relative
error of 2e8, so the direction is not a matter of taste.

## 4. The MTP head comes from a sibling file

`q3nvfp4` has no MTP tensors at all (`mtp_num_hidden_layers = 0`, zero `mtp.*`
keys), yet the oracle carries 12 `mtp/*` objects.  They are bit-exactly
reproducible from `q38_abl_huihui_nvfp4/model-mtp-bf16.safetensors`:
`tools/convert/qwen3_8_27b/mtp.py` rebuilds all 12 **byte-for-byte** in ~2 s while
streaming row segments (545 MB heap growth versus 2647 MB for a whole-matrix
quantizer; `VmHWM` 1587 MB versus 3525 MB).

The fused row orders were derived with the oracle as a byte-level judge (candidate
enumeration plus full byte comparison), and two independent implementations agree:

* `mtp/layer/attention/query_key_gate_value` (14336x5120) is
  **`query(6144) ‖ k(1024) ‖ output_gate(6144) ‖ value(1024)`**, where the 12288-row
  `q_proj` is split per head (24 heads x 512 rows) into `[0:256]` = query and
  `[256:512]` = output-gate.  A contiguous split of `q_proj` is the obvious thing
  to write and it is wrong; the next-best candidate reaches only 85.71% row
  agreement, and the first 64 code-plane bytes do not discriminate at all (9 of 15
  candidates share them).
* `mtp/layer/mlp/gate_up` (34816x5120) is **`gate ‖ up`** (`up ‖ gate` scores 1.13%).

## 5. Object accounting for `q3nvfp4`, and what is intentionally imperfect

Of the oracle's 1118 tensor objects, **1104 have a source key present**.

| class | count | treatment | fidelity |
|---|---|---|---|
| BF16 / FP32 direct | 623 | cast / upcast | **bit-exact** |
| NVFP4 (MLP, layers 0-55) | 112 | literal word copy, `d_w = 1/weight_scale_2` | **71 bit-exact**; 41 differ, see below |
| vision Q4/Q5/Q6/W8 | 111 | quantized from BF16 source | bit-exact |
| MTP | 12 | sibling file, `tools/convert/qwen3_8_27b/mtp.py` | **bit-exact** |
| FP8 (attention, GDN, `lm_head`, MLP 56-63) | 145 | dequantize NVFP4, re-encode with the §2 profile | inside the source's own NVFP4 step |
| `text/token_embedding` | 1 | as FP8 | inside tolerance |

### Deviations that are not bugs

1. **41 of the 112 NVFP4 objects (`mlp/down`, layers 15-55) cannot be reproduced
   byte-exactly.**  Their codes differ from `q3nvfp4`'s in 27.0% of elements, and
   the difference carries a rank-1 component (`sigma_1 / ||delta||_F = 0.307`
   against a same-density null of 0.051) - the signature of a different per-tensor
   global scale rather than of re-rounding.  Copy the source's own codes (identical
   to `q3nvfp4`, 6.0-6.3% RMS from the oracle) rather than dequantizing and
   re-encoding (12.6% RMS), and record the deviation.
2. **The 112 `input_scale_divisor` values must not be inferred.**  §3.3 already
   states that a site-level input divisor is a separate model-role tensor and
   cannot be derived from the weight format; this source makes that concrete - the
   oracle's values are not `1/input_scale` (ratios 0.68-1.47, no constant).  Use
   `1/input_scale`, which is self-consistent with the weights being quantized, and
   record it as the site-level calibration used.
3. **The oracle's frontend resources are themselves a mixture**: `tokenizer.json`
   and `preprocessor_config.json` match `q3nvfp4`; `chat_template.jinja` and
   `generation_config.json` match `q38_abl_huihui_bf16`;
   `video_preprocessor_config.json` matches `q38_abl_huihui_nvfp4`; and the
   oracle's `tokenizer_config.json` (17928 B) exists in no local directory.  Use
   the source's own files and record the drift - see
   `tools/convert/qwen3_6/common/frontend_policy.py`, which classifies each
   resource and refuses only genuinely missing ones.

## 6. Acceptance

`tools/convert/artifact_diff.py` compares two artifacts object by object through
the registered decoders and classifies each as `identical`, `logical`, `divergent`
(with `max_abs` and relative RMS) or `structural`; it exits non-zero on any
divergence, so it works as a gate:

```
python3 -m tools.convert.artifact_diff <reference>.ninfer <imported>.ninfer --list-divergent
```

Expected for this model: 1104 objects `identical`; the 145 FP8 objects and the
embedding object `divergent` with a relative RMS inside the NVFP4 step; the 41
`mlp/down` objects `divergent` at 6.0-6.3% RMS.  A **structural** difference means
the import is wrong, not that the reference is.
