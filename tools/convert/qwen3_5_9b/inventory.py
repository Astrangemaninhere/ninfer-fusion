"""Object inventory of the qwen3_5_9b artifact.

Every name, shape and numeric format here is the contract `src/targets/qwen3_5_9b/
impl/load/bindings.cpp` binds.  Two independent statements of one contract is the
failure mode this file is written to make visible: `tools/convert/qwen3_5_9b/
check_bindings.py` parses the C++ binder and compares it against this table, so the
two cannot drift without a red check.

The geometry is the real checkpoint's, read off
`Ornith-1.5-9B-Q4_K_M.gguf` metadata (see src/targets/qwen3_5_9b/impl/config.h for
the key-by-key provenance): hidden 4096, 32 main layers, intermediate 12288,
16 query heads x 256, 4 kv heads x 256, GDN 16x128 key / 32x128 value, conv kernel
4, vocab 248320, 8 full-attention + 24 GDN layers under `full_attention_interval=4`.

Row sources
-----------
A GGUF tensor's `ne = (fastest, ..., outermost)` is the reverse of the HF
`(out_features, in_features)` convention, and the row-major *bytes are unchanged* by
that reversal: GGUF `(k, n)` and HF `(n, k)` are the same linear run of n rows of k
elements.  So a source row range is a byte range, and no transposition is needed
anywhere in this converter.  That is the whole reason the row ranges below are
expressed as `(gguf_name, row_begin, row_count)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MODEL_ID = "qwen3.5-9b"
#: The storage scheme AND the draft-payload flavour of the artifact, stated in the artifact's
#: own identity.  `WEIGHTS_ID` is written when the source GGUF declares a draft (nextn) block
#: and `WEIGHTS_ID_NO_MTP` when it declares none, because an artifact with no `mtp/*` object
#: cannot run `--spec mtp` and must not be asked to.  The identity is the part of the artifact
#: the backend decision reads before any object is bound (`Package::resolve_weights` ->
#: `Package::resolved_auto_speculative`, src/targets/qwen3_5_9b/impl/package.cpp), and
#: `bindings.cpp` cross-checks it against the object table, so the two statements of the fact
#: cannot drift.  Precedent for a payload flavour in the weights_id: qwen3_6_27b's
#: "nvfp4-dspark" / "nvfp4-dflash2".
WEIGHTS_ID = "gguf-kquant"
WEIGHTS_ID_NO_MTP = "gguf-kquant-nomtp"
TARGET_KEY = "qwen3_5_9b"
RECIPE_ID = "qwen3_5_9b-gguf-v1"

HIDDEN = 4096
LAYERS = 32
INTERMEDIATE = 12288
FULL_ATTENTION_INTERVAL = 4
HEAD_DIM = 256
QUERY_HEADS = 16
KV_HEADS = 4
QUERY_SIZE = QUERY_HEADS * HEAD_DIM          # 4096
KV_SIZE = KV_HEADS * HEAD_DIM                # 1024
OUTPUT_ROWS = 248320
TOKEN_DOMAIN = 248077
GDN_KEY_HEADS = 16
GDN_KEY_HEAD_DIM = 128
GDN_VALUE_HEADS = 32
GDN_VALUE_HEAD_DIM = 128
KEY_DIM = GDN_KEY_HEADS * GDN_KEY_HEAD_DIM    # 2048
VALUE_DIM = GDN_VALUE_HEADS * GDN_VALUE_HEAD_DIM  # 4096
CONV_KERNEL = 4
CONV_DIM = 2 * KEY_DIM + VALUE_DIM           # 8192

#: The draft block a source that DECLARES one carries: one block, at `blk.<LAYERS>`.  A source
#: that declares none is not refused -- `plan_tensors(with_mtp=False)` leaves the block out of
#: the plan and `convert.py` writes `WEIGHTS_ID_NO_MTP` instead.
MTP_LAYERS = 1

#: The draft block's 12 object names, in binding order.  THIS literal is one statement of the
#: closure; `_mtp_specs()` below is the other, and the assert under it holds them together.
#: `check_bindings.py` compares the C++ binder's `mtp/*` region against this tuple, and
#: `convert.py` writes it into the conversion report.
MTP_OBJECT_NAMES = (
    "mtp/input_projection",
    "mtp/embedding_norm",
    "mtp/hidden_norm",
    "mtp/layer/input_norm",
    "mtp/layer/attention/query_key_gate_value",
    "mtp/layer/attention/query_norm",
    "mtp/layer/attention/key_norm",
    "mtp/layer/attention/output",
    "mtp/layer/post_attention_norm",
    "mtp/layer/mlp/gate_up",
    "mtp/layer/mlp/down",
    "mtp/final_norm",
)

FULL_LAYERS = tuple(l for l in range(LAYERS) if (l + 1) % FULL_ATTENTION_INTERVAL == 0)
GDN_LAYERS = tuple(l for l in range(LAYERS) if l not in FULL_LAYERS)
assert len(FULL_LAYERS) == 8 and len(GDN_LAYERS) == 24

BF16 = "BF16"
FP32 = "FP32"
Q4 = "Q4G64_F16S"
Q5 = "Q5G64_F16S"
Q6 = "Q6G64_F16S"

DIRECT_FORMATS = (BF16, FP32)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One artifact object: declared shape, numeric format, and where its rows come from."""

    name: str
    shape: tuple[int, ...]
    format: str
    #: (gguf_name, row_begin, row_count) concatenated in order to form this object's rows.
    #: For a direct (BF16/FP32) object the list is the flat source order.
    sources: tuple[tuple[str, int, int], ...] = field(default=())

    @property
    def kind(self) -> str:
        return "tensor"

    @property
    def is_quantized(self) -> bool:
        return self.format not in DIRECT_FORMATS


def _one(name: str, shape: tuple[int, ...], fmt: str, gguf: str) -> TensorSpec:
    rows = shape[0] if len(shape) == 2 else 1
    return TensorSpec(name, shape, fmt, ((gguf, 0, rows),))


def plan_tensors(only_layers: tuple[int, ...] | None = None,
                 *, with_mtp: bool = True) -> tuple[TensorSpec, ...]:
    """The complete ordered object inventory (resources are added by convert.py).

    `with_mtp=True` is the plan this function has always returned, including the order, so the
    MTP-declaring path is unchanged.  `with_mtp=False` is the artifact of a source that declares
    no draft block: the 12 `mtp/*` objects are absent from the plan entirely.  That absence is
    what makes MTP optional -- the closure stops being a requirement of every source and becomes
    a declaration of the source it came from.
    """

    layers = tuple(range(LAYERS)) if only_layers is None else tuple(only_layers)
    specs: list[TensorSpec] = [
        _one("text/token_embedding", (OUTPUT_ROWS, HIDDEN), Q6, "token_embd.weight"),
    ]

    for layer in layers:
        p = "text/layers/%d/" % layer
        b = "blk.%d." % layer
        specs.append(_one(p + "input_norm", (HIDDEN,), BF16, b + "attn_norm.weight"))
        if layer in FULL_LAYERS:
            specs.append(TensorSpec(
                p + "attention/query_key", (QUERY_SIZE + KV_SIZE, HIDDEN), Q4,
                ((b + "attn_q.weight", 0, QUERY_SIZE),
                 (b + "attn_k.weight", 0, KV_SIZE))))
            specs.append(TensorSpec(
                p + "attention/gate_value", (QUERY_SIZE + KV_SIZE, HIDDEN), Q5,
                ((b + "attn_q.weight", QUERY_SIZE, QUERY_SIZE),
                 (b + "attn_v.weight", 0, KV_SIZE))))
            specs.append(_one(p + "attention/query_norm", (HEAD_DIM,), BF16,
                              b + "attn_q_norm.weight"))
            specs.append(_one(p + "attention/key_norm", (HEAD_DIM,), BF16,
                              b + "attn_k_norm.weight"))
            specs.append(_one(p + "attention/output", (HIDDEN, QUERY_SIZE), Q5,
                              b + "attn_output.weight"))
        else:
            specs.append(_one(p + "gdn/a_log", (GDN_VALUE_HEADS,), FP32, b + "ssm_a"))
            specs.append(_one(p + "gdn/dt_bias", (GDN_VALUE_HEADS,), FP32, b + "ssm_dt.bias"))
            # Declared (kernel, channels); the runtime views it (channels, kernel) and the
            # payload bytes are the source's, in source order.
            specs.append(_one(p + "gdn/convolution", (CONV_KERNEL, CONV_DIM), BF16,
                              b + "ssm_conv1d.weight"))
            specs.append(_one(p + "gdn/a_projection", (GDN_VALUE_HEADS, HIDDEN), BF16,
                              b + "ssm_alpha.weight"))
            specs.append(_one(p + "gdn/b_projection", (GDN_VALUE_HEADS, HIDDEN), BF16,
                              b + "ssm_beta.weight"))
            # in_proj_qkv is [q(KEY_DIM) | k(KEY_DIM) | v(VALUE_DIM)]; in_proj_z is z.
            specs.append(TensorSpec(
                p + "gdn/query_key", (2 * KEY_DIM, HIDDEN), Q4,
                ((b + "attn_qkv.weight", 0, 2 * KEY_DIM),)))
            specs.append(TensorSpec(
                p + "gdn/value_z", (2 * VALUE_DIM, HIDDEN), Q5,
                ((b + "attn_qkv.weight", 2 * KEY_DIM, VALUE_DIM),
                 (b + "attn_gate.weight", 0, VALUE_DIM))))
            specs.append(_one(p + "gdn/norm", (GDN_VALUE_HEAD_DIM,), BF16, b + "ssm_norm.weight"))
            specs.append(_one(p + "gdn/output", (HIDDEN, VALUE_DIM), Q5, b + "ssm_out.weight"))
        specs.append(_one(p + "post_attention_norm", (HIDDEN,), BF16,
                          b + "post_attention_norm.weight"))
        specs.append(TensorSpec(
            p + "mlp/gate_up", (2 * INTERMEDIATE, HIDDEN), Q4,
            ((b + "ffn_gate.weight", 0, INTERMEDIATE),
             (b + "ffn_up.weight", 0, INTERMEDIATE))))
        specs.append(_one(p + "mlp/down", (HIDDEN, INTERMEDIATE), Q5, b + "ffn_down.weight"))

    specs.append(_one("text/final_norm", (HIDDEN,), BF16, "output_norm.weight"))
    specs.append(_one("text/output_head", (OUTPUT_ROWS, HIDDEN), Q6, "output.weight"))

    if with_mtp:
        specs.extend(_mtp_specs())

    return tuple(specs)


def _mtp_specs() -> list[TensorSpec]:
    """The draft block's 12 objects, in binding order (`MTP_OBJECT_NAMES`).

    Only reached when the source declares a draft block: `plan_tensors(with_mtp=False)` does
    not call it, so an artifact converted from a source with no MTP layer has no `mtp/*` object
    for the loader to require.
    """

    # The checkpoint's own nextn/MTP block (qwen35.block_count=33 with
    # nextn_predict_layers=1), so the draft block is blk.32.
    d = "blk.%d." % LAYERS
    mtp_prefix = "mtp"
    specs: list[TensorSpec] = []
    specs.append(_one(mtp_prefix + "/input_projection", (HIDDEN, 2 * HIDDEN), Q4,
                      d + "nextn.eh_proj.weight"))
    specs.append(_one(mtp_prefix + "/embedding_norm", (HIDDEN,), BF16, d + "nextn.enorm.weight"))
    specs.append(_one(mtp_prefix + "/hidden_norm", (HIDDEN,), BF16, d + "nextn.hnorm.weight"))
    specs.append(_one(mtp_prefix + "/layer/input_norm", (HIDDEN,), BF16, d + "attn_norm.weight"))
    # row_view order in the binder: query | key | output_gate | value.
    specs.append(TensorSpec(
        mtp_prefix + "/layer/attention/query_key_gate_value",
        (2 * QUERY_SIZE + 2 * KV_SIZE, HIDDEN), Q4,
        ((d + "attn_q.weight", 0, QUERY_SIZE),
         (d + "attn_k.weight", 0, KV_SIZE),
         (d + "attn_q.weight", QUERY_SIZE, QUERY_SIZE),
         (d + "attn_v.weight", 0, KV_SIZE))))
    specs.append(_one(mtp_prefix + "/layer/attention/query_norm", (HEAD_DIM,), BF16,
                      d + "attn_q_norm.weight"))
    specs.append(_one(mtp_prefix + "/layer/attention/key_norm", (HEAD_DIM,), BF16,
                      d + "attn_k_norm.weight"))
    specs.append(_one(mtp_prefix + "/layer/attention/output", (HIDDEN, QUERY_SIZE), Q5,
                      d + "attn_output.weight"))
    specs.append(_one(mtp_prefix + "/layer/post_attention_norm", (HIDDEN,), BF16,
                      d + "post_attention_norm.weight"))
    specs.append(TensorSpec(
        mtp_prefix + "/layer/mlp/gate_up", (2 * INTERMEDIATE, HIDDEN), Q4,
        ((d + "ffn_gate.weight", 0, INTERMEDIATE),
         (d + "ffn_up.weight", 0, INTERMEDIATE))))
    specs.append(_one(mtp_prefix + "/layer/mlp/down", (HIDDEN, INTERMEDIATE), Q5,
                      d + "ffn_down.weight"))
    specs.append(_one(mtp_prefix + "/final_norm", (HIDDEN,), BF16,
                      d + "nextn.shared_head_norm.weight"))
    return specs


#: One closure stated twice: the tuple at the top of this module and the specs `_mtp_specs()`
#: builds.  If they drift, `check_bindings.py` would compare the C++ against a name list that
#: no longer describes the plan, which is exactly the silent-drift failure this file exists to
#: prevent (see the module docstring).
assert tuple(s.name for s in _mtp_specs()) == MTP_OBJECT_NAMES, \
    "MTP_OBJECT_NAMES drifted from _mtp_specs()"


#: The six frontend resources the binder requires, in binding order.  They are raw
#: files, not tensors, so they carry a source directory rather than GGUF rows.
RESOURCE_NAMES = (
    "frontend/tokenizer.json",
    "frontend/tokenizer_config.json",
    "frontend/chat_template.jinja",
    "frontend/generation_config.json",
    "frontend/preprocessor_config.json",
    "frontend/video_preprocessor_config.json",
)

__all__ = [
    "MODEL_ID", "WEIGHTS_ID", "TARGET_KEY", "RECIPE_ID",
    "HIDDEN", "LAYERS", "INTERMEDIATE", "OUTPUT_ROWS", "TOKEN_DOMAIN",
    "FULL_LAYERS", "GDN_LAYERS", "FULL_ATTENTION_INTERVAL",
    "BF16", "FP32", "Q4", "Q5", "Q6", "DIRECT_FORMATS",
    "TensorSpec", "plan_tensors", "RESOURCE_NAMES",
    "MTP_LAYERS", "MTP_OBJECT_NAMES", "WEIGHTS_ID_NO_MTP",
]
