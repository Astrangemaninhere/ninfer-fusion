# -*- coding: utf-8 -*-
"""Source-side role inventory for a Spark-X2.5-4B artifact.

What this module is, and what it deliberately is NOT
----------------------------------------------------
It states the checkpoint's own storage roles: which source tensors exist, what each
one is, and what geometry the numbers in them must have.

It also names ENGINE objects now, and that has a date on it.  The first version of
this module refused to, in these words: "It does **not** name engine objects, because
there is no registered `spark_x2_5_4b` target in this tree, and an object contract
without a target is a guess."  `landq/sparktarget` landed that target on 2026-09-23
08:13 -- eight files under `src/targets/spark_x2_5_4b/` -- so the contract is now a
MEASUREMENT read off `impl/load/bindings.cpp:132-218` instead of a guess, and the
engine-object section below is that measurement.  The target's own header states the
consequence verbatim: once those names land, this converter's role table has to be
re-pointed at them (`bindings.cpp:9-14`, `:16`), which is what `recipe.py` does.

The one object-level decision the SOURCE side makes is the tied head, and that
decision is the same one
`tools/convert/common/source_map.py` makes mechanically for a tied checkpoint
(see `recipe.py`'s tie block for the citation).

Every number here has one of two sources, named per line:

* `models/Spark-X2.5-4B/config.json` -- the checkpoint's own declaration.
* the safetensors index/shard headers -- measured, read by `recipe.shape_check`
  and `convert.build_plan`, never restated here.

The derived quantities (per-layer role table, per-kind rotary width) are computed
from those two, so a corrected input corrects the table instead of drifting from it.

Spark-X2.5-4B in one line: 36 layers, a single attention geometry for the whole
stack (16 query / 4 KV heads at head_dim 256), sliding-window attention on 27 of
them and full attention on the 9 that are every fourth layer, per-kind rope
(theta 10000 / 5000000, partial rotary 1.0 / 0.25), a per-head sigmoid output gate,
gated GELU MLP, no qk-norm, tied word embeddings, bf16.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
MODEL_ID = "spark-x2.5-4b"
WEIGHTS_ID = "bf16"
TARGET_KEY = "spark_x2_5_4b"

# --------------------------------------------------------------------------- #
# geometry -- models/Spark-X2.5-4B/config.json, one constant per key
# --------------------------------------------------------------------------- #
HIDDEN = 2560                    # hidden_size
LAYERS = 36                      # num_hidden_layers
INTERMEDIATE = 10240             # intermediate_size
VOCAB = 131072                   # vocab_size
MAX_CTX = 1048576                # max_position_embeddings
RMS_EPS = 1e-06                  # rms_norm_eps

Q_HEADS = 16                     # num_attention_heads
KV_HEADS = 4                     # num_key_value_heads
HEAD_DIM = 256                   # head_dim

# --------------------------------------------------------------------------- #
# attention shape: one geometry for the whole stack
# --------------------------------------------------------------------------- #
# Unlike gemma4-31b (two geometries in one checkpoint) Spark declares ONE triple and
# applies it to every layer.  This is the triple the engine already registers:
# src/ops/kernel/gqa_attention_geometry.cuh:60
#     using Gqa16x4Geometry = GqaGeometry<16, 4, 1>;
# and that file's comment at :42 names this checkpoint verbatim
# ("Gqa16x4Geometry <16,4,1,256> ... CHANGES (Spark-X2.5, NOT measured)").
# `convert.geometry_gate` parses that header rather than restating the alias.
ATTENTION_GEOMETRY = {"query_heads": Q_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM}

# --------------------------------------------------------------------------- #
# layer schedule -- config.json `layer_types`, run-length compressed
# --------------------------------------------------------------------------- #
# The declared list is "3x sliding, 1x full" repeated nine times, so layer L is full
# when (L + 1) % 4 == 0.  Kept as an explicit tuple AND as the predicate, and
# `validate_schedule` in convert.py checks the predicate against the declared list
# element by element instead of trusting either side.
FULL_ATTENTION_LAYERS = tuple(l for l in range(LAYERS) if (l + 1) % 4 == 0)
SLIDING_ATTENTION_LAYERS = tuple(l for l in range(LAYERS) if l not in FULL_ATTENTION_LAYERS)
LAYER_TYPES = tuple("full_attention" if l in FULL_ATTENTION_LAYERS else "sliding_attention"
                    for l in range(LAYERS))

SLIDING_WINDOW = 512             # config.json `sliding_window`

# --------------------------------------------------------------------------- #
# per-kind rope -- config.json `rope_parameters`
# --------------------------------------------------------------------------- #
# partial_rotary_factor is a fraction of head_dim, so the engine's rotary width is
# head_dim * factor.  Whole-stack head_dim is 256, so:
#   sliding_attention -> 1.00 * 256 = 256
#   full_attention    -> 0.25 * 256 =  64
ROPE_THETA_BY_KIND = {"sliding_attention": 10000.0, "full_attention": 5000000.0}
PARTIAL_ROTARY_BY_KIND = {"sliding_attention": 1.0, "full_attention": 0.25}
ROTARY_DIM_BY_KIND = {kind: int(HEAD_DIM * factor)
                      for kind, factor in PARTIAL_ROTARY_BY_KIND.items()}

# --------------------------------------------------------------------------- #
# knobs that a single shared TextConfig has no slot for
# --------------------------------------------------------------------------- #
# These are the facts that make a new target necessary rather than a new checkout of
# an existing one; `convert.WORK_ITEMS` names where each one lands.
TIE_WORD_EMBEDDINGS = True       # config.json `tie_word_embeddings`
HIDDEN_ACTIVATION = "gelu"       # config.json `hidden_act`
ATTENTION_BIAS = False           # config.json `attention_bias`
HEADWISE_ATTN_OUTPUT_GATE = True  # config.json `headwise_attn_output_gate`
GATE_ATTN_ACT_MODE = "sigmoid"   # config.json `gate_attn_act_mode`
QK_NORM = None                   # config.json has no qk-norm knob; see qk_norm_absent()
MULTIMODAL = False               # no vision tower: 290 source tensors, all text
DTYPE = "bfloat16"               # config.json `dtype`

# --------------------------------------------------------------------------- #
# storage vocabulary (the engine's own words)
# --------------------------------------------------------------------------- #
BF16 = "BF16"
CONTIGUOUS_LAYOUT = "contiguous-le-v1"
FORMAT_NAMES = (BF16,)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT,)

# --------------------------------------------------------------------------- #
# source key layout -- measured from the safetensors index, one rule per role
# --------------------------------------------------------------------------- #
EMBED_KEY = "model.embedding.weight"
FINAL_NORM_KEY = "model.norm.weight"
LAYER_KEY_FMT = "model.layers.{layer}.{leaf}"

#: The eight leaves every layer carries, in the order the index lists them.  Spark
#: has ONE fused q/k/v projection (no q_proj / k_proj / v_proj) and a separate
#: per-head gate projection, which is the same fusion the source's own
#: modeling_spark.py:160-190 forward assumes.
LAYER_LEAVES = (
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.g_proj.weight",
    "self_attn.out_proj.weight",
    "self_attn.q_k_v_proj.weight",
)

SRC_TENSORS_PER_LAYER = len(LAYER_LEAVES)
SRC_TENSOR_COUNT = 1 + LAYERS * SRC_TENSORS_PER_LAYER + 1      # 290

# --------------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------------- #
# Derived from the geometry above, never from the header: `recipe.shape_check`
# compares these against the header, so a disagreement is reported as a conflict
# instead of being absorbed.  Row-major, `[out_features, in_features]`.
FUSED_QKV_ROWS = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM              # 6144
#: the contraction width of the attention half: q_heads * head_dim
QUERY_WIDTH = Q_HEADS * HEAD_DIM                                  # 4096
# The output gate is ONE SCALAR PER HEAD, not a per-channel projection: the measured
# shard header carries `self_attn.g_proj.weight` at `(16, 2560)` = `(Q_HEADS, HIDDEN)`,
# and 16 == num_attention_heads.  This is the shape the engine's `ops::sigmoid_mul`
# headwise-gate branch consumes.  `out_proj`, by contrast, really does contract over
# `QUERY_WIDTH` -- the header says `(2560, 4096)`.  Keeping the two apart is the whole
# point: the first draft of this table reused one constant for both and the volume gate
# caught the resulting 376,012,800-parameter overcount.
GATE_ROWS = Q_HEADS                                               # 16
EMBEDDING_SHAPE = (VOCAB, HIDDEN)
FINAL_NORM_SHAPE = (HIDDEN,)

# --------------------------------------------------------------------------- #
# the ENGINE's object contract -- read off the target that owns it
# --------------------------------------------------------------------------- #
# Every string below is a name the target's own binder looks up, so a name this
# table does not emit is refused AT LOAD TIME with
#     "required artifact tensor is missing: <name>"
# (src/targets/spark_x2_5_4b/impl/load/bindings.cpp:68 for the matrices, :78 for the
# vectors).  The binding call sites are `bind_text_layers()` at :132-171 and
# `bind_artifact()` at :190-218.
TOKEN_EMBEDDING_OBJECT = "text/token_embedding"
FINAL_NORM_OBJECT = "text/final_norm"
OUTPUT_HEAD_OBJECT = "text/output_head"
LAYER_OBJECT_DIR_FMT = "text/layers/{layer}/"
LAYER_KEY_PREFIX_FMT = "model.layers.{layer}."

#: The fused q/k/v geometry.  `model.layers.N.self_attn.q_k_v_proj.weight` is ONE
#: source tensor of `FUSED_QKV_ROWS` = 6144 rows and the engine binds THREE objects;
#: the split is the converter's job -- bindings.cpp:64-65: "The converter is the half
#: that splits the one source tensor into three objects."
KV_SIZE = KV_HEADS * HEAD_DIM                                     # 1024

#: `(object leaf inside a layer, shape)`, in the ORDER the target binds them.  The
#: order is kept because it is also the order the artifact's directory lists them, so
#: a reader can diff the two by eye; nothing structural depends on it.
LAYER_OBJECTS = (
    ("input_norm", (HIDDEN,)),
    ("attention/query", (QUERY_WIDTH, HIDDEN)),
    ("attention/key", (KV_SIZE, HIDDEN)),
    ("attention/value", (KV_SIZE, HIDDEN)),
    ("attention/gate", (GATE_ROWS, HIDDEN)),
    ("attention/output", (HIDDEN, QUERY_WIDTH)),
    ("post_attention_layernorm", (HIDDEN,)),
    ("mlp/gate", (INTERMEDIATE, HIDDEN)),
    ("mlp/up", (INTERMEDIATE, HIDDEN)),
    ("mlp/down", (HIDDEN, INTERMEDIATE)),
)

#: object leaf -> `(source leaf, first row, row after the last)`.  A `None` row range
#: is the whole tensor.  The split's ORDER and its BOUNDARIES are measured from the
#: checkpoint's own reference implementation, not chosen here --
#: models/Spark-X2.5-4B/modeling_spark.py:170-173:
#:     qkv = self.q_k_v_proj(hidden_states)
#:     q = qkv[..., :self.q_dim]
#:     k = qkv[..., self.q_dim:self.q_dim + self.kv_dim]
#:     v = qkv[..., self.q_dim + self.kv_dim:]
#: so q is rows [0, 4096), k is [4096, 5120), v is [5120, 6144) of the fused tensor.
LAYER_OBJECT_SOURCES = {
    "input_norm": ("input_layernorm.weight", None, None),
    "attention/query": ("self_attn.q_k_v_proj.weight", 0, QUERY_WIDTH),
    "attention/key": ("self_attn.q_k_v_proj.weight", QUERY_WIDTH, QUERY_WIDTH + KV_SIZE),
    "attention/value": ("self_attn.q_k_v_proj.weight", QUERY_WIDTH + KV_SIZE, None),
    "attention/gate": ("self_attn.g_proj.weight", None, None),
    "attention/output": ("self_attn.out_proj.weight", None, None),
    "post_attention_layernorm": ("post_attention_layernorm.weight", None, None),
    "mlp/gate": ("mlp.gate_proj.weight", None, None),
    "mlp/up": ("mlp.up_proj.weight", None, None),
    "mlp/down": ("mlp.down_proj.weight", None, None),
}


def layer_objects(
    layer: int,
) -> tuple[tuple[str, tuple[int, ...], str, int | None, int | None], ...]:
    """Every tensor object one layer contributes, in bind order.

    Each entry is `(object name, shape, source key, first row, row after the last)`.
    """

    prefix = LAYER_OBJECT_DIR_FMT.format(layer=layer)
    src_prefix = LAYER_KEY_PREFIX_FMT.format(layer=layer)
    out: list[tuple[str, tuple[int, ...], str, int | None, int | None]] = []
    for leaf, shape in LAYER_OBJECTS:
        source_leaf, first_row, last_row = LAYER_OBJECT_SOURCES[leaf]
        out.append((prefix + leaf, shape, src_prefix + source_leaf, first_row, last_row))
    return tuple(out)


def engine_objects() -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Every TENSOR object the artifact must carry, in the target's bind order.

    The denominator a reader should hold this against is
    `1 + LAYERS * len(LAYER_OBJECTS) + 1 + 1` = 1 + 360 + 1 + 1 = 363, and that is
    what `recipe.OBJECT_RECIPES` becomes once it is driven by this table.
    """

    out: list[tuple[str, tuple[int, ...]]] = [(TOKEN_EMBEDDING_OBJECT, EMBEDDING_SHAPE)]
    for layer in range(LAYERS):
        out.extend((name, shape) for name, shape, _key, _a, _b in layer_objects(layer))
    out.append((FINAL_NORM_OBJECT, FINAL_NORM_SHAPE))
    # The tied head.  The engine binds `text/output_head` as an INDEPENDENT object of
    # `[output_rows, hidden]` (bindings.cpp:213-214) and the index ships no
    # `lm_head.weight`, so this object's bytes ARE the embedding's.  That is the
    # decision `recipe.tie_decision()` reports; it is not taken here.
    out.append((OUTPUT_HEAD_OBJECT, EMBEDDING_SHAPE))
    return tuple(out)


def engine_object_count() -> int:
    return len(engine_objects())


def layer_shape(leaf: str) -> tuple[int, ...]:
    """The shape `leaf` must have on every layer.  Raises on an unknown leaf."""

    table = {
        "input_layernorm.weight": FINAL_NORM_SHAPE,
        "post_attention_layernorm.weight": FINAL_NORM_SHAPE,
        "mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN),
        "mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
        "mlp.down_proj.weight": (HIDDEN, INTERMEDIATE),
        "self_attn.q_k_v_proj.weight": (FUSED_QKV_ROWS, HIDDEN),
        "self_attn.g_proj.weight": (GATE_ROWS, HIDDEN),
        "self_attn.out_proj.weight": (HIDDEN, QUERY_WIDTH),
    }
    try:
        return table[leaf]
    except KeyError:                                          # pragma: no cover
        raise KeyError(f"no shape is declared for layer leaf {leaf!r}") from None


def layer_kind(layer: int) -> str:
    return "full_attention" if layer in FULL_ATTENTION_LAYERS else "sliding_attention"


def rotary_dim_of(layer: int) -> int:
    return ROTARY_DIM_BY_KIND[layer_kind(layer)]


def rope_theta_of(layer: int) -> float:
    return ROPE_THETA_BY_KIND[layer_kind(layer)]


def is_sliding(layer: int) -> bool:
    return layer not in FULL_ATTENTION_LAYERS


def qk_norm_absent() -> tuple[bool, str]:
    """Whether q and k are normalised, and the two places that say so.

    `config.json` has no qk-norm knob at all, and the fixed 290-key index has no
    `q_norm` / `k_norm` leaf under any layer.  Both are checked here so the claim is
    a measurement, not a reading of the absence of a key.
    """

    return True, ("config.json declares no qk-norm knob and the 290-key index carries "
                  "no `self_attn.q_norm.weight` / `self_attn.k_norm.weight` leaf")


# --------------------------------------------------------------------------- #
# source key enumeration
# --------------------------------------------------------------------------- #
def expected_source_keys() -> tuple[str, ...]:
    """Every source key this checkpoint must contain, in index order."""

    keys = [EMBED_KEY]
    for layer in range(LAYERS):
        keys.extend(LAYER_KEY_FMT.format(layer=layer, leaf=leaf) for leaf in LAYER_LEAVES)
    keys.append(FINAL_NORM_KEY)
    return tuple(keys)


def expected_shapes() -> dict[str, tuple[int, ...]]:
    table = {EMBED_KEY: EMBEDDING_SHAPE, FINAL_NORM_KEY: FINAL_NORM_SHAPE}
    for layer in range(LAYERS):
        for leaf in LAYER_LEAVES:
            table[LAYER_KEY_FMT.format(layer=layer, leaf=leaf)] = layer_shape(leaf)
    return table


def source_tensor_count() -> int:
    return len(expected_source_keys())


# --------------------------------------------------------------------------- #
# volumes -- arithmetic on the declared geometry, checked against the index metadata
# --------------------------------------------------------------------------- #
BYTES_PER_BF16 = 2


def tensor_bytes(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total * BYTES_PER_BF16


def payload_bytes() -> int:
    return sum(tensor_bytes(shape) for shape in expected_shapes().values())


def parameter_count() -> int:
    return sum(_prod(shape) for shape in expected_shapes().values())


def _prod(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def format_counts() -> dict[str, int]:
    # The ARTIFACT's object count, not the source's: the write path emits one BF16
    # object per engine object and splits the fused q/k/v, so this is 363 and not 290.
    return {BF16: engine_object_count()}


def layout_counts() -> dict[str, int]:
    return {CONTIGUOUS_LAYOUT: engine_object_count()}
