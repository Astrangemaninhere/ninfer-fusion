# -*- coding: utf-8 -*-
"""Convert a Spark-X2.5-4B checkpoint into one `.ninfer` artifact.

Canonical invocation::

    python -m tools.convert.spark_x2_5_4b.convert \
      --model /path/to/Spark-X2.5-4B \
      --out out/spark_x2_5_4b.ninfer \
      --resources <a dir holding frontend/tokenizer.json ...>

``--plan`` stops after the plan and its four gates, which is the mode that runs today:
it reads the checkpoint's own index and safetensors headers, the archkit spec and
generated `config.h`, and the engine's GQA alias header, and it needs neither torch
nor a GPU.

This module is also this target's answer to the import front door's contract
(`tools/convert/import_model.py`): `evaluate_targets` imports
`tools.convert.<target>.convert` and calls the `validate_config` it finds here, and the
runnable gate reads `SOURCE_QUANT_METHODS` / `SUPPLIES_FRONTEND_RESOURCES` from the
same module.  No second copy of these numbers is kept in the front door.

What this converter does now, and what it still does not
-------------------------------------------------------
IT WRITES THE ARTIFACT.  Until dl/_orch/landq/sparkwriter landed, this file's only
`open(` was the safetensors header read at :223 and the gate-green path ended by
printing `WORK_ITEMS` and returning 3 -- it said by name that it could not write rather
than writing something nothing could load.  Both halves of that reason are gone:
`landq/sparktarget` landed the C++ target on 2026-09-23 08:13 (eight files under
`src/targets/spark_x2_5_4b/`), and `landq/cmakebatch` had already landed the
`src/CMakeLists.txt` `add_subdirectory` guard.  So the object names this converter
emits are now a MEASUREMENT off `impl/load/bindings.cpp:132-218` -- carried in
`inventory.LAYER_OBJECTS` / `LAYER_OBJECT_SOURCES` -- instead of the roles the previous
version refused to pin.  The container is `tools/artifact/container.py`, the same one
`tools/convert/muse_glimmer_30b/convert.py:24, :440` uses, and the produced file is
re-opened with that module's own reader before this program reports success.

Note what the payloads ARE, because it is why this converter is CPU-only and needs no
torch: the source is unquantized BF16 row-major, which is exactly what the artifact
stores for the one format this target consumes, so every payload is a byte RANGE of a
shard.  Nothing is dequantized, cast or multiplied, and `tools/convert/common/` -- the
package that owns the tensor-maths path -- is still not imported.

What is still NOT here, and it is engine-side:

* per-kind rope.  Spark needs theta 10000 / 5000000 and rotary width 256 / 64 on the
  sliding / full layers, while the shared `TextConfig` has a single `rotary_dim` and a
  single `rope_theta` slot (`text_context.h:44`), and `ModelConfig::rope_theta_at`
  (`:108`) is called from nowhere.
* no qk-norm.  The family runtime normalises q and k unconditionally
  (`text_context_impl.h:1204-1205`), and this checkpoint carries no `q_norm` / `k_norm`
  tensor in any of its 36 layers.
* the headwise output gate's shape-selected route (`sigmoid_mul.cpp:222-225`).

Those three are why `LOAD` is refused inside the target, by name, at
`bindings.cpp:271-291` -- AFTER `plan_load` succeeds.  So the load PLAN is a real object
this artifact can be checked against, and the refusal is one call to delete on the day
the family hooks land.  The artifact itself is complete either way: every name and
shape the target binds is in it.

* the front door's own `REGISTERED_TARGETS` (`tools/convert/import_model.py:55-70`) still
  does not name this target, so `import_model.py` skips this `validate_config` and says
"no target accepts this source".  That file is outside this converter and is named in
`WORK_ITEMS` rather than edited here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in (None, ""):                              # run-by-path support
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "tools.convert.spark_x2_5_4b"

from . import inventory, recipe
# The artifact container.  Same module, same four names, same order as
# tools/convert/muse_glimmer_30b/convert.py:24-25 -- `MAGIC` is imported so the
# converter can print the bytes it wrote rather than describe them.
from tools.artifact.container import (
    MAGIC,
    RAW_BYTES_V1,
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    TensorSpec,
)
from tools.artifact.layouts import encoded_size

# THE TREE'S OWN RESTATEMENT OF THE ENGINE'S TOKENIZER GATE (F-857, line dl/sparkfix).
# The authority is src/targets/qwen3_6/impl/frontend/frontend.cpp:202-225; this module
# restates it, so a converter can assert the bytes it is about to write instead of
# asserting two keys by hand and hoping it read the same gate.  It is IMPORTED rather
# than copied because a second copy of a gate is a second thing to drift -- and the
# measured shape of this very defect is "the converter never asked the tree's own
# checker".  Path: tools/convert/qwen3_6/common/frontend_policy.py.
from tools.convert.qwen3_6.common.frontend_policy import (
    ENGINE_TOKENIZER_CONFIG_PAD_TOKEN,
    engine_tokenizer_config_reasons,
)

MODEL_ID = inventory.MODEL_ID
WEIGHTS_ID = inventory.WEIGHTS_ID
TARGET_KEY = inventory.TARGET_KEY
RECIPE_ID = "spark_x2_5_4b-v1"

#: This checkpoint is an unquantized bf16 source, so the flavour it converts is the
#: front door's own unquantized flavour -- the one `import_model.py:583-584` describes
#: as "未量化的官方源".  Saying it explicitly keeps the runnable gate honest instead of
#: inheriting a default.
#: One weights flavour, and this converter really emits it: `package.h:58-62` rules v1
#: BF16 and `accepted_weights` (:178-184) accepts exactly `("spark-x2.5-4b", "bf16")`,
#: which is the identity written below.
SOURCE_QUANT_METHODS = ("native",)
#: CHANGED False -> True by dl/_orch/landq/sparkwriter item 0004.  The write path below
#: emits the artifact's six frontend resource objects itself, so the flag that told the
#: front door "this target does not supply its frontend resources" would now be false:
#: it means "the converter synthesises those resources ... and never reads the source's
#: sidecars for them" (tools/convert/import_model.py:993-1002, which then skips the
#: source-side resource gate).
SUPPLIES_FRONTEND_RESOURCES = True

#: Read by the conversion report; the front door does not read these, but a reader of
#: the plan needs to know which family the geometry came from.
FAMILY = "spark2_5"
ARCHITECTURE = "Spark2_5ForCausalLM"

#: The archkit artifacts this target's geometry was measured into.  `--plan` reads both
#: and compares them against `inventory`, so a stale spec is a red gate rather than a
#: silent mismatch.
SPEC_PATH = Path("tools/archkit/specs/spark-x2.5-4b_spec.json")
GENERATED_CONFIG_H = Path("tools/archkit/out/spark-x2.5-4b/config.h")
GQA_GEOMETRY_HEADER = Path("src/ops/kernel/gqa_attention_geometry.cuh")

WORK_ITEMS = (
    ("front-door registration",
     "tools/convert/import_model.py:55-70 REGISTERED_TARGETS and :83-84 DECODER_FAMILIES "
     "do not name this target, so the front door skips THIS converter's validate_config "
     "and reports that no target accepts the source.  The file is outside this "
     "converter, so the row is named rather than added here."),
    ("per-kind rope",
     "the shared TextConfig has one rotary_dim and one rope_theta slot "
     "(text_context.h:44), but Spark needs 256 @ 10000 on the 27 sliding layers and "
     "64 @ 5000000 on the 9 full ones (inventory.ROTARY_DIM_BY_KIND / "
     "ROPE_THETA_BY_KIND); ModelConfig::rope_theta_at exists (:108) and is called from "
     "nowhere"),
    ("no qk-norm",
     "the family runtime normalises q and k unconditionally "
     "(text_context_impl.h:1204-1205) and this checkpoint has no q_norm / k_norm tensor "
     "in any of its 36 layers; TextConfig::qk_norm_enabled() is declared false and read "
     "from nowhere"),
    ("headwise output gate",
     "the attention-output gate is viewed {head_dim, n_q, T} (text_context_impl.h:1180) "
     "and sigmoid_mul decides its headwise route BY SHAPE (sigmoid_mul.cpp:222-225); "
     "Spark's gate has 16 rows, so the per-element route accepts the pair and reads "
     "rows this target never wrote"),
)

#: The three WORK_ITEMS entries the previous version carried and this one does NOT, with
#: where each went.  Kept as a record rather than deleted, because "the list got shorter"
#: is a claim a reader should be able to check:
#:   * "C++ target"            -> landed: landq/sparktarget, 2026-09-23 08:13, the eight
#:                               files under src/targets/spark_x2_5_4b/ this artifact's
#:                               object names are now measured off.
#:   * "src/CMakeLists.txt ..." -> landed earlier: landq/cmakebatch item 0001, the
#:                               self-activating add_subdirectory guard at :689-691.
#:   * "artifact object contract" -> THIS item: the write path below emits one object per
#:                               engine name, and `build_plan`'s `artifact_volume` gate
#:                               checks the plan against the index arithmetic.
LANDED_WORK_ITEMS = (
    "C++ target -> landq/sparktarget (8 files, 2026-09-23 08:13)",
    "src/CMakeLists.txt add_subdirectory -> landq/cmakebatch item 0001",
    "artifact object contract -> this converter's write path (item 0004)",
)

# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
#: Every scalar `config.json` must agree with `inventory` on.  Built from the
#: inventory constants so the two cannot drift: each entry is
#: (config key, inventory value, human name).
SCALAR_CONTRACT = (
    ("hidden_size", inventory.HIDDEN, "hidden"),
    ("num_hidden_layers", inventory.LAYERS, "layers"),
    ("intermediate_size", inventory.INTERMEDIATE, "intermediate"),
    ("vocab_size", inventory.VOCAB, "vocab"),
    ("max_position_embeddings", inventory.MAX_CTX, "max_ctx"),
    ("rms_norm_eps", inventory.RMS_EPS, "rms_eps"),
    ("num_attention_heads", inventory.Q_HEADS, "query_heads"),
    ("num_key_value_heads", inventory.KV_HEADS, "kv_heads"),
    ("head_dim", inventory.HEAD_DIM, "head_dim"),
    ("sliding_window", inventory.SLIDING_WINDOW, "sliding_window"),
    ("tie_word_embeddings", inventory.TIE_WORD_EMBEDDINGS, "tie_word_embeddings"),
    ("hidden_act", inventory.HIDDEN_ACTIVATION, "hidden_act"),
    ("attention_bias", inventory.ATTENTION_BIAS, "attention_bias"),
    ("headwise_attn_output_gate", inventory.HEADWISE_ATTN_OUTPUT_GATE,
     "headwise_attn_output_gate"),
    ("gate_attn_act_mode", inventory.GATE_ATTN_ACT_MODE, "gate_attn_act_mode"),
    ("dtype", inventory.DTYPE, "dtype"),
    ("model_type", FAMILY, "model_type"),
)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the checkpoint's own config against this converter's contract.

    Raises ``ValueError`` whose first line is ``checkpoint config mismatch`` -- the
    front door keys off that phrase when it formats a rejection -- and whose remaining
    lines each name one disagreeing field.
    """

    problems: list[str] = []
    for key, want, name in SCALAR_CONTRACT:
        got = config.get(key)
        if got != want:
            problems.append(f"{name}: expected {want!r} ({key}), found {got!r}")

    arch = config.get("architectures")
    if arch != [ARCHITECTURE]:
        problems.append(f"architectures: expected [{ARCHITECTURE!r}], found {arch!r}")

    if config.get("attention_dropout") not in (0.0, 0):
        problems.append(f"attention_dropout: expected 0.0, found {config.get('attention_dropout')!r}")

    declared_types = config.get("layer_types")
    if declared_types is None:
        problems.append("layer_types: absent (this checkpoint declares it)")
    elif tuple(declared_types) != inventory.LAYER_TYPES:
        first = next((i for i, (a, b) in enumerate(zip(declared_types, inventory.LAYER_TYPES))
                      if a != b), None)
        problems.append(
            f"layer_types: the declared list differs from the '(3 sliding, 1 full) x 9' "
            f"schedule at index {first} "
            f"(declared {None if first is None else declared_types[first]!r}, "
            f"derived {None if first is None else inventory.LAYER_TYPES[first]!r})")

    rope = config.get("rope_parameters")
    if not isinstance(rope, dict):
        problems.append(f"rope_parameters: expected a per-kind mapping, found {type(rope).__name__}")
    else:
        for kind in ("full_attention", "sliding_attention"):
            block = rope.get(kind)
            if not isinstance(block, dict):
                problems.append(f"rope_parameters.{kind}: absent")
                continue
            theta = block.get("rope_theta")
            if theta != inventory.ROPE_THETA_BY_KIND[kind]:
                problems.append(f"rope_parameters.{kind}.rope_theta: expected "
                                f"{inventory.ROPE_THETA_BY_KIND[kind]!r}, found {theta!r}")
            factor = block.get("partial_rotary_factor")
            if factor != inventory.PARTIAL_ROTARY_BY_KIND[kind]:
                problems.append(f"rope_parameters.{kind}.partial_rotary_factor: expected "
                                f"{inventory.PARTIAL_ROTARY_BY_KIND[kind]!r}, found {factor!r}")

    absent, why = inventory.qk_norm_absent()
    if config.get("qk_norm") not in (None, False) or not absent:
        problems.append(f"qk_norm: expected absent ({why}), found {config.get('qk_norm')!r}")

    if problems:
        raise ValueError("checkpoint config mismatch\n" + "\n".join(problems))

    return {
        "model_type": config.get("model_type"),
        "architecture": ARCHITECTURE,
        "geometry": dict(inventory.ATTENTION_GEOMETRY),
        "layer_schedule": {"full_attention": len(inventory.FULL_ATTENTION_LAYERS),
                           "sliding_attention": len(inventory.SLIDING_ATTENTION_LAYERS),
                           "full_attention_layers": list(inventory.FULL_ATTENTION_LAYERS)},
        "per_kind_rope": {kind: {"rope_theta": inventory.ROPE_THETA_BY_KIND[kind],
                                 "rotary_dim": inventory.ROTARY_DIM_BY_KIND[kind]}
                          for kind in ("sliding_attention", "full_attention")},
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "hidden_act": config.get("hidden_act"),
        "qk_norm": "absent",
        "artifact_identity": {"model_id": MODEL_ID, "weights_id": WEIGHTS_ID},
    }


# --------------------------------------------------------------------------- #
# reading the source
# --------------------------------------------------------------------------- #
def read_safetensors_header(path: Path) -> dict[str, dict]:
    """Read `path`'s safetensors header only -- never the tensor data."""

    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: too short to be a safetensors file")
        n = int.from_bytes(raw, "little")
        header = json.loads(handle.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def read_index(model_dir: Path) -> dict:
    path = model_dir / "model.safetensors.index.json"
    if not path.is_file():
        raise ValueError(f"{model_dir}: no model.safetensors.index.json")
    return json.loads(path.read_text(encoding="utf-8"))


def load_headers(model_dir: Path, weight_map: Mapping[str, str]) -> dict[str, dict]:
    """Merge every shard's header into one key -> {shape, dtype, shard, complete}."""

    merged: dict[str, dict] = {}
    per_shard: dict[str, dict] = {}
    for shard_name in sorted(set(weight_map.values())):
        path = model_dir / shard_name
        if not path.is_file():
            per_shard[shard_name] = {"complete": False}
            continue
        header = read_safetensors_header(path)
        per_shard[shard_name] = {"complete": True, "tensors": len(header)}
        for key, entry in header.items():
            merged[key] = {
                "shape": list(entry.get("shape") or ()),
                "dtype": entry.get("dtype"),
                "shard": shard_name,
                "shard_complete": True,
            }
    for key, shard in weight_map.items():
        merged.setdefault(key, {"shape": None, "dtype": None, "shard": shard,
                                "shard_complete": False})
    return merged, per_shard


# --------------------------------------------------------------------------- #
# the three cross-artifact gates
# --------------------------------------------------------------------------- #
_ALIAS_RE = re.compile(r"using\s+(\w+)\s*=\s*GqaGeometry<\s*([0-9]+)\s*,\s*([0-9]+)\s*"
                       r"(?:,\s*([0-9]+))?(?:,\s*([0-9]+))?\s*>")


def geometry_gate(repo_root: Path) -> dict:
    """Is this checkpoint's attention triple one of the engine's registered aliases?

    Parses `src/ops/kernel/gqa_attention_geometry.cuh` rather than restating its table,
    so an alias that is renamed or removed turns this gate red.
    """

    path = repo_root / GQA_GEOMETRY_HEADER
    if not path.is_file():
        return {"ok": False, "why": f"{GQA_GEOMETRY_HEADER} is not readable",
                "aliases": [], "match": None}
    text = path.read_text(encoding="utf-8")
    aliases = []
    for name, q, kv, third, fourth in _ALIAS_RE.findall(text):
        aliases.append({"alias": name, "query_heads": int(q), "kv_heads": int(kv),
                        "third": None if not third else int(third),
                        "head_dim": None if not fourth else int(fourth)})
    want_q = inventory.Q_HEADS
    want_kv = inventory.KV_HEADS
    match = next((a for a in aliases if a["query_heads"] == want_q and a["kv_heads"] == want_kv),
                 None)
    return {"ok": match is not None, "want": f"{want_q}q/{want_kv}kv@{inventory.HEAD_DIM}",
            "aliases": aliases, "match": match,
            "why": "" if match else f"no registered GQA alias is {want_q}q/{want_kv}kv"}


_CONFIG_H_INT = re.compile(r"static\s+constexpr\s+int\s+(\w+)\s*=\s*(-?[0-9]+)\s*;")
# the emitter writes the two derived layer counts as accessor functions, not as
# constants: `static constexpr int full_attention_layers() { return 9; }`
_CONFIG_H_INT_FN = re.compile(r"static\s+constexpr\s+int\s+(\w+)\s*\(\s*\)\s*\{\s*return\s+"
                              r"(-?[0-9]+)\s*;\s*\}")
_CONFIG_H_FLOAT = re.compile(r"static\s+constexpr\s+float\s+(\w+)\s*=\s*(-?[0-9.eE+]+)f?\s*;")


def generated_config_gate(repo_root: Path) -> dict:
    """Does the archkit-generated `config.h` still carry this checkpoint's numbers?"""

    path = repo_root / GENERATED_CONFIG_H
    if not path.is_file():
        return {"ok": False, "why": f"{GENERATED_CONFIG_H} is not readable", "mismatches": []}
    text = path.read_text(encoding="utf-8")
    ints = {m.group(1): int(m.group(2)) for m in _CONFIG_H_INT.finditer(text)}
    ints.update({m.group(1): int(m.group(2)) for m in _CONFIG_H_INT_FN.finditer(text)})
    floats = {m.group(1): float(m.group(2)) for m in _CONFIG_H_FLOAT.finditer(text)}
    want_ints = {"hidden": inventory.HIDDEN, "layers": inventory.LAYERS,
                 "query_heads": inventory.Q_HEADS, "kv_heads": inventory.KV_HEADS,
                 "head_dim": inventory.HEAD_DIM, "intermediate": inventory.INTERMEDIATE,
                 "vocab": inventory.VOCAB, "max_ctx": inventory.MAX_CTX,
                 "sliding_window": inventory.SLIDING_WINDOW}
    want_floats = {"rms_epsilon": inventory.RMS_EPS}
    mismatches = [f"{k}: config.h {ints.get(k)!r} != inventory {v!r}"
                  for k, v in want_ints.items() if ints.get(k) != v]
    mismatches += [f"{k}: config.h {floats.get(k)!r} != inventory {v!r}"
                   for k, v in want_floats.items() if floats.get(k) != v]
    # the two derived layer counts the emitter writes
    if ints.get("full_attention_layers") != len(inventory.FULL_ATTENTION_LAYERS):
        mismatches.append(f"full_attention_layers: config.h {ints.get('full_attention_layers')!r} "
                          f"!= {len(inventory.FULL_ATTENTION_LAYERS)}")
    if ints.get("swa_attention_layers") != len(inventory.SLIDING_ATTENTION_LAYERS):
        mismatches.append(f"swa_attention_layers: config.h {ints.get('swa_attention_layers')!r} "
                          f"!= {len(inventory.SLIDING_ATTENTION_LAYERS)}")
    return {"ok": not mismatches, "mismatches": mismatches,
            "present": sorted(ints) + sorted(floats)}


def volume_gate(derived_shapes: Mapping[str, tuple[int, ...]],
                index_parameters: int | None, index_bytes: int | None) -> dict:
    """Does arithmetic on the derived geometry reproduce the index's own metadata?

    `model.safetensors.index.json` carries `total_parameters` and `total_size`.  Both
    are derivable from the geometry and the role table, so comparing them turns a whole
    class of shape mistakes into a red gate instead of a plan that merely looks
    plausible.  The first draft of `inventory.layer_shape` got `self_attn.g_proj.weight`
    wrong by a factor of `head_dim` and this is the gate that said so.
    """

    params = sum(_prod(shape) for shape in derived_shapes.values())
    nbytes = sum(2 * _prod(shape) for shape in derived_shapes.values())
    mismatches: list[str] = []
    if index_parameters is not None and params != index_parameters:
        mismatches.append(f"derived parameters {params} != index total_parameters "
                          f"{index_parameters} (delta {params - index_parameters})")
    if index_bytes is not None and nbytes != index_bytes:
        mismatches.append(f"derived bytes {nbytes} != index total_size {index_bytes} "
                          f"(delta {nbytes - index_bytes})")
    return {"ok": not mismatches, "derived_parameters": params, "derived_bytes": nbytes,
            "index_total_parameters": index_parameters, "index_total_size": index_bytes,
            "mismatches": mismatches}


def _prod(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def artifact_tensor_bytes() -> int:
    """The artifact's tensor payload, summed from the ENGINE's object table.

    Every shape the engine binds is either a whole source tensor or a row slice of one,
    and each is a multiple of `contiguous-le-v1`'s 256-byte alignment (2560 elements x
    2 B = 5120; 4096 x 2560 x 2 = 20,971,520; 16 x 2560 x 2 = 81,920; 10240 x 2560 x 2
    = 52,428,800), so this sum has NO padding term and is exactly what the file's
    payload adds up to.
    """

    total = 0
    for _name, shape in inventory.engine_objects():
        total += encoded_size(inventory.CONTIGUOUS_LAYOUT, inventory.BF16, shape)
    return total


def artifact_volume_gate(index_bytes: int | None) -> dict:
    """Does the artifact's own object table add up to the index plus the head?

    The identity, in full: the write path emits one object per engine name; the ONLY
    object whose bytes are not a source tensor is the materialised head, whose bytes are
    the embedding's (`recipe.tie_decision()`); and no object needs padding (see
    `artifact_tensor_bytes`).  So

        sum(artifact tensor bytes) == index total_size + 2 * vocab * hidden

    must hold EXACTLY.  A split boundary that is off by one row, a shape that is wrong
    by one dimension, or an object the target binds and this table omits all move this
    number -- which is why it is a gate and not a print.  The first draft of
    `inventory.layer_shape` got `self_attn.g_proj.weight` wrong by a factor of
    `head_dim`, and this is the same class of mistake caught one level up.
    """

    artifact_bytes = artifact_tensor_bytes()
    head_bytes = inventory.tensor_bytes(inventory.EMBEDDING_SHAPE)
    expected = None if index_bytes is None else index_bytes + head_bytes
    ok = expected is None or artifact_bytes == expected
    return {
        "ok": ok,
        "artifact_tensor_bytes": artifact_bytes,
        "index_total_size": index_bytes,
        "materialised_head_bytes": head_bytes,
        "expected": expected,
        "why": "" if ok else (
            f"artifact tensor bytes {artifact_bytes} != index total_size {index_bytes} "
            f"+ materialised head {head_bytes} = {expected}"
            f" (delta {artifact_bytes - expected})"),
    }


def spec_gate(repo_root: Path) -> dict:
    """Does the archkit *spec* still agree with the inventory constants?"""

    path = repo_root / SPEC_PATH
    if not path.is_file():
        return {"ok": False, "why": f"{SPEC_PATH} is not readable", "mismatches": []}
    spec = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    geometry = spec.get("geometry") or {}
    for key, want in (("hidden", inventory.HIDDEN), ("layers", inventory.LAYERS),
                      ("intermediate", inventory.INTERMEDIATE), ("vocab", inventory.VOCAB),
                      ("max_ctx", inventory.MAX_CTX), ("query_heads", inventory.Q_HEADS),
                      ("kv_heads", inventory.KV_HEADS), ("head_dim", inventory.HEAD_DIM),
                      ("rms_eps", inventory.RMS_EPS)):
        if geometry.get(key) != want:
            mismatches.append(f"geometry.{key}: spec {geometry.get(key)!r} != inventory {want!r}")
    if spec.get("hidden_act") != inventory.HIDDEN_ACTIVATION:
        mismatches.append(f"hidden_act: spec {spec.get('hidden_act')!r} != "
                          f"inventory {inventory.HIDDEN_ACTIVATION!r}")
    if tuple(spec.get("layer_types") or ()) != inventory.LAYER_TYPES:
        mismatches.append("layer_types: the spec's list differs from the derived schedule")
    if (spec.get("rope_by_kind") or {}) != {k: v for k, v in inventory.ROPE_THETA_BY_KIND.items()}:
        mismatches.append(f"rope_by_kind: spec {spec.get('rope_by_kind')!r} != inventory "
                          f"{inventory.ROPE_THETA_BY_KIND!r}")
    knobs = spec.get("knobs") or {}
    for key, want in (("sliding_window", inventory.SLIDING_WINDOW),
                      ("tie_word_embeddings", inventory.TIE_WORD_EMBEDDINGS),
                      ("hidden_act", inventory.HIDDEN_ACTIVATION),
                      ("attention_bias", inventory.ATTENTION_BIAS),
                      ("headwise_attn_output_gate", inventory.HEADWISE_ATTN_OUTPUT_GATE),
                      ("gate_attn_act_mode", inventory.GATE_ATTN_ACT_MODE)):
        if knobs.get(key) != want:
            mismatches.append(f"knobs.{key}: spec {knobs.get(key)!r} != inventory {want!r}")
    return {"ok": not mismatches, "mismatches": mismatches,
            "gaps": [g.get("need") for g in spec.get("gaps") or []]}


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def build_plan(model_dir: Path, repo_root: Path) -> dict:
    index = read_index(model_dir)
    weight_map = index.get("weight_map") or {}
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    headers, per_shard = load_headers(model_dir, weight_map)

    coverage = recipe.validate_coverage(weight_map.keys())
    shapes = recipe.shape_check(headers)
    embed_header = headers.get(inventory.EMBED_KEY)
    if embed_header is not None and not embed_header.get("shape"):
        embed_header = None
    binding = recipe.tie_decision(weight_map.keys(), config, header=embed_header)

    derived = inventory.expected_shapes()
    bytes_planned = sum(inventory.tensor_bytes(shape) for shape in derived.values())
    index_bytes = (index.get("metadata") or {}).get("total_size")
    index_params = (index.get("metadata") or {}).get("total_parameters")

    return {
        "recipe_id": RECIPE_ID,
        "identity": {"model_id": MODEL_ID, "weights_id": WEIGHTS_ID, "target_key": TARGET_KEY},
        "shards": sorted(set(weight_map.values())),
        "shard_headers": per_shard,
        "source_tensor_count": len(weight_map),
        "object_count": len(recipe.OBJECT_RECIPES),
        "format_counts": inventory.format_counts(),
        "layout_counts": inventory.layout_counts(),
        "payload_bytes": bytes_planned,
        "payload_bytes_source_total": sum(
            inventory.tensor_bytes(tuple(e["shape"])) for e in headers.values()
            if e.get("shape")) or None,
        "index_metadata": {"total_parameters": index_params, "total_size": index_bytes},
        "derived_parameters": inventory.parameter_count(),
        "coverage": coverage,
        "shapes": shapes,
        "source_binding": {"output_head": binding.to_report()},
        "gates": {
            "geometry": geometry_gate(repo_root),
            "generated_config_h": generated_config_gate(repo_root),
            "archkit_spec": spec_gate(repo_root),
            "volume": volume_gate(derived, index_params, index_bytes),
            # The fifth gate, and the only one that reads the ENGINE's object table rather
            # than the source's: it is what makes "the plan knows what the artifact will
            # contain" a checked statement instead of an intention.
            "artifact_volume": artifact_volume_gate(index_bytes),
        },
        "artifact": {
            "identity": {"model_id": MODEL_ID, "weights_id": WEIGHTS_ID},
            "object_count": inventory.engine_object_count(),
            "object_count_rule": ("1 embedding + 36 layers x 10 objects + 1 final_norm "
                                  "+ 1 materialised head = 363 tensors, plus the 6 "
                                  "frontend resources the engine binds"),
            "tensor_payload_bytes": artifact_tensor_bytes(),
        },
        "head_implications": {
            "tied": inventory.TIE_WORD_EMBEDDINGS,
            "note": ("the engine binds text/output_head independently of "
                     "text/token_embedding, so a tied checkpoint is handled by "
                     "materialising the head at conversion time and needs no new operator"),
        },
    }


def _print_plan(plan: dict) -> None:
    ident = plan["identity"]
    print(f"  identity      : {ident['model_id']} / {ident['weights_id']}   "
          f"(target_key {ident['target_key']})")
    print(f"  shards        : {len(plan['shards'])} -> "
          f"{', '.join(plan['shards'])}")
    print(f"  source tensors: {plan['source_tensor_count']}")
    print(f"  roles         : {plan['object_count']} planned "
          f"({len(plan['format_counts'])} format, {len(plan['layout_counts'])} layout)")
    print(f"  formats       : {plan['format_counts']}")
    print(f"  layouts       : {plan['layout_counts']}")
    print(f"  payload       : {plan['payload_bytes']} B "
          f"({plan['payload_bytes'] / 2**30:.2f} GiB) planned "
          f"from derived geometry")
    meta = plan["index_metadata"]
    print(f"  index metadata: total_parameters={meta['total_parameters']} "
          f"total_size={meta['total_size']}")
    print(f"  derived check : {plan['derived_parameters']} parameters, "
          f"{plan['payload_bytes_source_total']} B of source tensors")
    cov = plan["coverage"]
    print(f"  coverage gate : {cov['consumed_key_count']}/{cov['planned_key_count']} planned "
          f"keys found; {cov['ignored_key_count']} ignored by prefix; "
          f"{len(cov['unplanned_source_keys'])} unplanned; "
          f"{len(cov['missing_source_keys'])} missing  -> {'ok' if cov['ok'] else 'FAIL'}")
    for key in cov["unplanned_source_keys"][:10]:
        print(f"    UNPLANNED  {key}")
    for key in cov["missing_source_keys"][:10]:
        print(f"    MISSING    {key}")
    shp = plan["shapes"]
    print(f"  shape gate    : {shp['checked']}/{shp['expected']} objects compared "
          f"({shp['present']} present in headers), {len(shp['conflicts'])} conflict(s)"
          f"  -> {'ok' if shp['ok'] else 'FAIL'}")
    for line in shp["conflicts"][:10]:
        print(f"    CONFLICT   {line}")
    art = plan["artifact"]
    print(f"  artifact      : {art['identity']['model_id']} / "
          f"{art['identity']['weights_id']}  {art['object_count']} tensor objects, "
          f"{art['tensor_payload_bytes']} B payload")
    print(f"                  {art['object_count_rule']}")
    b = plan["source_binding"]["output_head"]
    print(f"  tied head     : materialize={b['materialized']} from {b['materialized_from']} "
          f"transpose={b['transpose']} orientation_verified={b['orientation_verified']} "
          f"source_bytes={b['source_bytes']}")
    print(f"                  reason: {b['reason']}")
    for name, gate in plan["gates"].items():
        print(f"  gate {name:<18}: {'ok' if gate['ok'] else 'FAIL'}"
              + (f"  ({gate.get('why')})" if gate.get("why") else ""))
        if name == "geometry" and gate.get("match"):
            m = gate["match"]
            print(f"      matched alias {m['alias']} = GqaGeometry<{m['query_heads']}, "
                  f"{m['kv_heads']}, {m['third']}"
                  + (f", {m['head_dim']}" if m["head_dim"] else "") + ">")
        if name == "volume":
            print(f"      derived parameters={gate['derived_parameters']} "
                  f"vs index {gate['index_total_parameters']}; "
                  f"derived bytes={gate['derived_bytes']} "
                  f"vs index {gate['index_total_size']}")
        for line in gate.get("mismatches", [])[:10]:
            print(f"      MISMATCH   {line}")


def _resource_roots(cli_roots: Sequence[Path] | None) -> tuple[Path, ...]:
    import os
    roots: list[Path] = list(cli_roots or ())
    env = os.environ.get("NINFER_RESOURCE_ROOTS", "")
    for chunk in env.split(os.pathsep):
        if chunk:
            roots.append(Path(chunk))
    return tuple(roots)


#: The six names the engine's binder looks up, in the binder's own order --
#: src/targets/qwen3_6/impl/frontend/resources.cpp:20-33.  This target's
#: `impl/load/bindings.h:20` imports that header, so the frontend Spark binds IS
#: qwen3_6's Frontend, and the previous four-name list here (which also carried
#: `special_tokens_map.json`, a file NOTHING binds, and omitted the two preprocessor
#: configs, which are bound unconditionally) could not have produced a loadable
#: artifact.
FRONTEND_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "generation_config.json", "preprocessor_config.json",
                  "video_preprocessor_config.json")

#: The THREE of the six that no Spark checkpoint ships, and the digest each pinned
#: fallback must have.  A substitution here is FORCED BY THE ENGINE, not chosen:
#: src/targets/qwen3_6/impl/frontend/chat_template.cpp:17-26 accepts exactly TWO
#: chat-template digests -- `kThinkingToggleTemplateDigest` (e84f32a2...) and
#: `kReasoningEffortTemplateDigest` (c3cf9e34...) -- and `CompiledChatTemplate::resolve`
#: (:414-423) throws "unsupported frontend/chat_template.jinja (sha256 ...)" for
#: anything else.  Spark's OWN template (3744 B, read out of
#: models/Spark-X2.5-4B/tokenizer_config.json) hashes to neither, so shipping it would
#: produce an artifact whose frontend cannot compile.  That is named here rather than
#: hidden: see REPORT.md, "the one substitution".
#:
#: Each digest is read off a file THE TREE ALREADY SHIPS AND ALREADY USES, and each is
#: asserted before use, so a drift is a named refusal and not a silent substitution:
#:   * c3cf9e34... is tools/convert/muse_glimmer_30b/qwen_chat_template.jinja -- the very
#:     bytes the shipped muse_glimmer_30b artifact carries (8952 B, measured);
#:   * 27225450... and 7768af27... are the two preprocessor pins in
#:     tools/convert/qwen3_6/common/official_resources.py:31-36.
PINNED_FRONTEND_SHA256 = {
    "chat_template.jinja":
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "preprocessor_config.json":
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "video_preprocessor_config.json":
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
}

#: The digests the ENGINE itself accepts for `frontend/chat_template.jinja`, verbatim
#: from chat_template.cpp:17-26.  A template that is supplied by a `--resources` root is
#: held to this pair (the engine's real requirement); the pin above is what the FALLBACK
#: is held to.
ACCEPTED_CHAT_TEMPLATE_SHA256 = (
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
)

#: Where the pinned copies live.  `tools/convert/muse_glimmer_30b` is the sibling whose
#: artifact already carries these exact bytes.  The sha256 above is what makes the
#: coupling safe: if that directory changes, this converter refuses by name instead of
#: shipping different frontend bytes under the same names.
PINNED_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "muse_glimmer_30b"
PINNED_FRONTEND_FILES = {
    "chat_template.jinja": "qwen_chat_template.jinja",
    "preprocessor_config.json": "qwen_preprocessor_config.json",
    "video_preprocessor_config.json": "qwen_video_preprocessor_config.json",
}


def resolve_resources(
    model_dir: Path, roots: Sequence[Path]
) -> dict[str, tuple[bytes | None, str]]:
    """Every one of the engine's six resources, as `(bytes, provenance)` or `(None, "")`.

    Search order per resource: the checkpoint's own directory, then each `--resources`
    root, then -- for the three a Spark checkpoint never ships -- the pinned copy.
    This never raises, and deliberately: `--plan` must stay runnable without resources,
    and the digest assertion belongs where the bytes are actually used, which is
    `frontend_payloads()` in the write path.
    """

    resolved: dict[str, tuple[bytes | None, str]] = {}
    for filename in FRONTEND_FILES:
        data: bytes | None = None
        where = ""
        for root in (model_dir, *roots):
            candidate = Path(root) / filename
            if candidate.is_file() and candidate.stat().st_size:
                data = candidate.read_bytes()
                where = str(candidate)
                break
        if data is None and filename in PINNED_FRONTEND_SHA256:
            pinned = PINNED_FRONTEND_DIR / PINNED_FRONTEND_FILES[filename]
            if pinned.is_file() and pinned.stat().st_size:
                data = pinned.read_bytes()
                where = str(pinned) + " (pinned)"
        resolved[f"frontend/{filename}"] = (data, where)
    return resolved


class ConversionRefused(Exception):
    """A refusal that names the thing which is wrong, on its first line."""


class ShardReader:
    """`(shard, data_offsets)` per key -- a header and a byte range, never a tensor.

    Spark's source is unquantized BF16 laid out row-major, which is EXACTLY what the
    artifact stores for the one format this target consumes
    (`tools/artifact/numeric.py:66` BF16 = 2 bytes/element; `tools/artifact/layouts.py:142`
    contiguous-le-v1, alignment 256).  So every payload is a byte RANGE of a shard: this
    converter never dequantizes, never casts and needs neither torch nor a GPU.  That is
    also why `tools/convert/common/` is still not imported -- that package owns the
    tensor-maths path.
    """

    def __init__(self, model_dir: Path, weight_map: Mapping[str, str]):
        self._dir = Path(model_dir)
        self._map = dict(weight_map)
        self._headers: dict[str, tuple[dict, int]] = {}
        self._handles: dict[str, Any] = {}

    def _shard(self, name: str):
        if name not in self._headers:
            path = self._dir / name
            handle = open(path, "rb")
            raw = handle.read(8)
            if len(raw) != 8:
                handle.close()
                raise ConversionRefused(f"{path}: too short to be a safetensors file")
            n = int.from_bytes(raw, "little")
            try:
                header = json.loads(handle.read(n).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                handle.close()
                raise ConversionRefused(
                    f"{path}: the safetensors header is not JSON ({exc})") from None
            header.pop("__metadata__", None)
            self._headers[name] = (header, 8 + n)
            self._handles[name] = handle
        return self._headers[name]

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def span(self, key: str) -> tuple[str, int, int, int, tuple[int, ...], str]:
        shard = self._map.get(key)
        if shard is None:
            raise ConversionRefused(
                f"{key}: model.safetensors.index.json's weight_map does not name this "
                f"tensor, so no shard holds it ({len(self._map)} keys in the map)")
        header, base = self._shard(shard)
        entry = header.get(key)
        if entry is None:
            raise ConversionRefused(
                f"{shard}: the index maps {key} to this shard but the shard's safetensors "
                f"header does not carry it ({len(header)} tensors in that header)")
        begin, end = (int(value) for value in entry["data_offsets"])
        return shard, base, begin, end, tuple(entry["shape"]), str(entry["dtype"])

    def _range(self, key: str, first_row: int | None,
               last_row: int | None) -> tuple[str, int, int, int, tuple[int, ...], str]:
        """(shard, base, absolute first byte, absolute last byte, shape, dtype)."""

        shard, base, begin, end, shape, dtype = self.span(key)
        if dtype != "BF16":
            raise ConversionRefused(
                f"{key}: declared {dtype}, and this converter reads BF16 only -- BF16 is "
                f"the one format this target consumes (bindings.cpp:42)")
        whole = 2 * _prod(shape)
        if end - begin != whole:
            raise ConversionRefused(
                f"{key}: the shard header's span is {end - begin} B but {dtype} "
                f"{list(shape)} is {whole} B -- refusing to slice a tensor whose header "
                f"disagrees with its own shape")
        if first_row is None and last_row is None:
            lo, hi = begin, end
        else:
            if len(shape) != 2:
                raise ConversionRefused(
                    f"{key}: a row range needs a rank-2 tensor, got {list(shape)}")
            rows = shape[0]
            lo_row = 0 if first_row is None else int(first_row)
            hi_row = rows if last_row is None else int(last_row)
            if not 0 <= lo_row < hi_row <= rows:
                raise ConversionRefused(
                    f"{key}: row range [{lo_row}, {hi_row}) is outside its {rows} rows")
            stride = shape[1] * 2
            lo, hi = begin + lo_row * stride, begin + hi_row * stride
        return shard, base, lo, hi, shape, dtype

    def payload_size(self, key: str, first_row: int | None = None,
                     last_row: int | None = None) -> int:
        """How many bytes `payload_chunks` will yield.  Validated, never read."""

        shard, base, lo, hi, _shape, _dtype = self._range(key, first_row, last_row)
        del shard, base
        return hi - lo

    def payload_chunks(self, key: str, first_row: int | None = None,
                       last_row: int | None = None,
                       block_bytes: int = 8 << 20) -> Iterable[bytes]:
        """`key`'s bytes as BOUNDED BLOCKS -- never one allocation for the tensor.

        The first version of this reader returned one `bytes` object per object and
        the first REAL run died on it: `OSError: [Errno 12] Cannot allocate memory`, in
        `handle.read(hi - lo)`, with 3,985,771,008 B already written.  A payload is a
        FILE REGION, so there is no reason to materialise it: `ArtifactWriter.write`
        accepts any iterable of blocks and validates the TOTAL against the planned
        length (container.py:401-421), so streaming loses no check at all.
        """

        shard, base, lo, hi, _shape, _dtype = self._range(key, first_row, last_row)
        handle = self._handles[shard]
        handle.seek(base + lo)
        remaining = hi - lo
        while remaining > 0:
            chunk = handle.read(min(block_bytes, remaining))
            if not chunk:
                raise ConversionRefused(
                    f"{shard}: short read for {key}: {remaining} B still expected -- the "
                    f"shard is truncated")
            remaining -= len(chunk)
            yield chunk


# ---------------------------------------------------------------------------------
# F1227 -- THE TOKENIZER SHAPE, MADE CANONICAL IN THE CONVERTER
# ---------------------------------------------------------------------------------
# WHAT WAS WRONG, MEASURED (dl/musesparkfix, F1227; the same reading as F1219 §6.1).
# `tools/convert/spark_x2_5_4b/convert.py` shipped the checkpoint's own
# `tokenizer.json` VERBATIM as `frontend/tokenizer.json` -- 10,115,786 B, sha256
# 710cce15cf3565674c499f9413997c6e8101f2bdd96245cff8f0311fb501248c, byte-identical to
# models/Spark-X2.5-4B/tokenizer.json -- and that file is refused by the engine at the
# FRONT DOOR, after 8.28 GiB of weights are on the device. Measured shape:
#
#   131,072 `model.vocab` entries, ids 0..131,071, dense, no duplicate ids;
#   8 of them are NOT byte-level decodable (ids 0,1,2,6,7,11,12,13 -- every one
#     contains U+FF5C FULLWIDTH VERTICAL LINE, e.g. `<｜start▁of▁sentence｜>`);
#   113 `added_tokens` entries, ALL with an id inside the vocab and ALL content-identical
#     to the vocab token at that same id (113 identical, 0 differing), 22 of them
#     `special: false`, 91 `special: true`, every one with lstrip/rstrip/normalized/
#     single_word false;
#   and `added_tokens_decoder` is absent from tokenizer.json (the tokenizer_config repair
#     above writes it as `{}`).
#
# WHY THE ENGINE REFUSES IT (gates read off the authority, tokenizer.cpp):
#   * `load_added_tokens` threw on `occupied_vocab_ids.contains(token.id)` -- 113/113 --
#     and again on `occupied_vocab_tokens.contains(token.content)` -- 113/113.
#   * and the 8 non-byte-level vocab entries fell to
#     `decode_byte_level_token` (:384-399) whenever they were NOT also declared as added
#     tokens (:783-785 decodes `valid && !added` only), where the GPT-2 alphabet of
#     :372-382 has no codepoint for U+FF5C.
#   The two are ONE fact seen twice: the 113 entries are redundant re-declarations of
#   tokens the vocab already carries. Deleting the `added_tokens` half (which is what
#   dl/sparkgap tried) moves the refusal one gate further in, because it strips the 8
#   non-byte-level ids of the `added` flag that was protecting them; deleting the VOCAB
#   half is the direction that satisfies both.
#
# WHAT THIS CONVERTER DOES ABOUT IT, AND WHY IT IS NOT A VOCAB REWRITE:
#   * it ASKS the engine's gate before writing: `engine_tokenizer_json_reasons` restates
#     every gate `Tokenizer` runs on these bytes and a non-empty answer is a NAMED REFUSAL in
#     the conversion (the F-857 precedent above), not 8.9 GB and a device upload later.
#   * the engine side of the same fact changed with F1227: `load_added_tokens` now tolerates
#     an added_tokens entry that repeats a `model.vocab` entry EXACTLY (same id, same
#     content), which is what the 113 here are -- and that tolerance is not a loosening for
#     its own sake. It is what keeps the 8 non-byte-level vocab entries DECODABLE, because
#     :783-785 byte-level-decodes only the ids that are `valid && !added`; an added token at
#     the same id is exactly the declaration that protects them.
#   * `canonicalize_tokenizer_json` below is the other available direction -- move the 113
#     out of `model.vocab`, keep them in `added_tokens` -- and it is TESTED and NOT EMITTED,
#     because it is measurably not encoding-neutral for the reference implementation:
#     HF `tokenizers` resolves an added token to its declared id only while that content is
#     still a `model.vocab` entry, and APPENDS it at the end of the vocabulary otherwise.
#     Measured on this checkpoint: `<｜start▁of▁sentence｜>` 0 -> 130959, `<think>` 3 -> 130962,
#     `<｜fim▁begin｜>` 11 -> 130970, `<|System|>` 130972 -> 130973, on EVERY prompt that
#     contains one of the 113 -- while the 368- and 816-token plain-text prompts used as the
#     first neutrality control are byte-identical. That is the trap: the control could not
#     see it, and a chat prompt is made of exactly those strings.
#   * the 256 byte-level symbols are CENSUSED and reported (`missing_byte_symbols`), not
#     gated: 13 are absent from this checkpoint (0xC0, 0xC1, 0xF5..0xFF) and every one is an
#     invalid UTF-8 lead byte, so no valid prompt can reach them; a prompt that somehow did
#     would get the engine's named `byte symbol outside vocabulary` throw. Refusing here
#     would refuse this checkpoint and every one shaped like it.
#
# The shape the muse and qwen3_8_27b artifacts satisfy -- base vocab byte-level only, specials
# at ids >= vocab size -- is NOT reachable for this checkpoint without moving ids: its
# specials sit at ids 0..131,071 with embedding rows behind them. What this converter reaches
# instead is the same two properties the engine actually checks (a byte-level base vocab, and
# no unreconciled overlap between the two tables) with every id, merge and embedding row
# untouched.
def _byte_level_alphabet() -> frozenset[str]:
    """tokenizer.cpp:372-382, re-implemented exactly: byte 33-126, 161-172 and 174-255 map
    to themselves; every other byte gets a codepoint from 256 upward, in byte order."""
    alphabet: set[str] = set()
    nxt = 256
    for byte in range(256):
        visible = (33 <= byte <= 126) or (161 <= byte <= 172) or (174 <= byte <= 255)
        alphabet.add(chr(byte if visible else nxt))
        if not visible:
            nxt += 1
    return frozenset(alphabet)


BYTE_LEVEL_ALPHABET = _byte_level_alphabet()


def _outside_byte_level(token: str) -> list[str]:
    return [f"U+{ord(ch):04X}" for ch in token if ch not in BYTE_LEVEL_ALPHABET]


def _bpe_load_bearing(vocab: Mapping[str, Any], merges: Sequence[Any]) -> set[str]:
    """The tokens a BPE encoder can emit or consume: any merge's left, right or result
    (tokenizer.cpp:344-377 requires all three to be in model.vocab), plus every byte-level
    symbol (`load_byte_token_ids`, :542-553). Moving one of these out of the vocab would
    move BPE output; moving anything else provably cannot."""
    load_bearing: set[str] = set()
    for entry in merges:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            left, right = str(entry[0]), str(entry[1])
        elif isinstance(entry, str) and " " in entry:
            left, right = entry.split(" ", 1)
        else:
            continue
        load_bearing.add(left)
        load_bearing.add(right)
        load_bearing.add(left + right)
    load_bearing.update(BYTE_LEVEL_ALPHABET)
    return load_bearing


def engine_tokenizer_json_reasons(data: bytes) -> list[str]:
    """The tree's own restatement of every gate this engine runs on
    `frontend/tokenizer.json` before it will build a Tokenizer, in the order they fire.
    Authority: src/targets/qwen3_6/impl/frontend/tokenizer.cpp -- `load_vocab` (:114-145),
    `load_added_tokens` (:207-260), the byte-level decode loop with
    `decode_byte_level_token` (:384-399) and `load_bpe_merge_rules` (:344-377). Empty list
    == the engine accepts these bytes.

    A converter asserts the bytes it is ABOUT TO WRITE with this, instead of asserting two
    keys by hand and hoping it read the same gate (the F-857 precedent above, and the exact
    shape of this defect: nothing asked).
    """
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return [f"tokenizer.json is not JSON ({exc})"]
    if not isinstance(payload, Mapping):
        return ["tokenizer.json is not a JSON object"]
    model = payload.get("model")
    if not isinstance(model, Mapping):
        return ["missing field model in tokenizer.json"]
    if model.get("type") != "BPE":
        return [f"field model.type must be BPE in tokenizer.json (got {model.get('type')!r})"]
    vocab = model.get("vocab")
    if not isinstance(vocab, Mapping) or not vocab:
        return ["field model.vocab must be a non-empty object in tokenizer.json"]
    merges = model.get("merges")
    if not isinstance(merges, list):
        return ["missing field model.merges in tokenizer.json"]

    reasons: list[str] = []
    ids: dict[int, str] = {}
    for token, index in vocab.items():
        if not isinstance(index, int):
            reasons.append(f"model.vocab entry {token!r} is not an integer id")
            continue
        if index in ids:
            reasons.append(f"field model.vocab has duplicate id {index}")
            continue
        ids[index] = str(token)
    vocab_tokens = set(vocab)

    added = payload.get("added_tokens")
    if not isinstance(added, list):
        return reasons + ["missing field added_tokens in tokenizer.json"]
    added_ids: set[int] = set()
    added_contents: dict[str, int] = {}
    protected_ids: set[int] = set()
    for entry in added:
        if not isinstance(entry, Mapping):
            reasons.append("field added_tokens item must be object in tokenizer.json")
            continue
        for key in ("id", "content", "single_word", "lstrip", "rstrip", "normalized",
                    "special"):
            if key not in entry:
                reasons.append(f"missing field added_tokens.{key} in tokenizer.json")
        token_id = entry.get("id")
        content = entry.get("content")
        if not isinstance(token_id, int) or not isinstance(content, str) or not content:
            reasons.append("added_tokens entry needs an integer id and a non-empty content")
            continue
        flags = [entry.get(k) for k in ("single_word", "lstrip", "rstrip", "normalized")]
        if any(flag is not False for flag in flags):
            reasons.append(
                f"added token {content!r}: the engine supports only single_word=false, "
                f"lstrip=false, rstrip=false, normalized=false")
        repeats = vocab_tokens and ids.get(token_id) == content
        if token_id in ids and not repeats:
            reasons.append(f"field added_tokens overlaps existing id {token_id} "
                           f"(and is not an exact re-declaration of that vocab entry)")
        if token_id in added_ids:
            reasons.append(f"field added_tokens has duplicate id {token_id}")
        if content in added_contents:
            reasons.append(f"field added_tokens has duplicate content {content!r}")
        added_ids.add(token_id)
        added_contents[content] = token_id
        if token_id in ids and ids[token_id] == content:
            protected_ids.add(token_id)

    # the byte-level decode loop (:783-785 decodes `valid && !added` only)
    for index in sorted(ids):
        if index in protected_ids:
            continue
        bad = _outside_byte_level(ids[index])
        if bad:
            reasons.append(
                f"token id {index} ({ids[index]!r}) contains a character outside the "
                f"byte-level alphabet ({', '.join(bad)}) and is not declared in "
                f"added_tokens, so Tokenizer::decode would throw")
    # load_bpe_merge_rules (:344-377)
    pairs: set[tuple[str, str]] = set()
    for entry in merges:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            left, right = str(entry[0]), str(entry[1])
        elif isinstance(entry, str) and entry.count(" ") == 1:
            left, right = entry.split(" ", 1)
        else:
            reasons.append("field model.merges must contain symbol pairs in tokenizer.json")
            continue
        if (left, right) in pairs:
            reasons.append(f"duplicate merge pair {' '.join((left, right))!r}")
        pairs.add((left, right))
        for symbol in (left, right, left + right):
            if symbol not in vocab_tokens:
                reasons.append(
                    f"model.merges references {symbol!r}, which is outside model.vocab")
    # load_byte_token_ids (:542-553) is deliberately NOT a gate here: `Tokenizer::encode`
    # throws "byte symbol outside vocabulary" only for a byte the encoder actually meets, and
    # nothing checks the 256 symbols when the Tokenizer is BUILT. The census is reported by
    # `missing_byte_symbols()` instead, and the measured reading for this checkpoint is worth
    # stating where a reader will find it: 13 of the 256 byte-level symbols are absent
    # (0xC0, 0xC1 and 0xF5..0xFF), and every one of those is an INVALID UTF-8 lead byte --
    # 0xC0/0xC1 are the overlong forms of a 2-byte sequence and 0xF5..0xFF address
    # codepoints above U+10FFFF -- so no valid UTF-8 prompt can produce one. If one ever
    # did (a Latin-1 encoded prompt file), the failure is the named throw above, not silent
    # corruption. Refusing on it would refuse this checkpoint outright.
    return reasons


def _byte_symbol_codepoints() -> list[tuple[int, int]]:
    """(byte, codepoint) for all 256 symbols, in the engine's own order (:372-382)."""
    out: list[tuple[int, int]] = []
    nxt = 256
    for byte in range(256):
        visible = (33 <= byte <= 126) or (161 <= byte <= 172) or (174 <= byte <= 255)
        out.append((byte, byte if visible else nxt))
        if not visible:
            nxt += 1
    return out


def missing_byte_symbols(vocab: Mapping[str, Any]) -> list[tuple[int, str]]:
    """(byte, 'U+XXXX') for every byte-level symbol `load_byte_token_ids` cannot resolve.
    Empty in the muse / qwen3_8_27b accepted shape; 13 here, all invalid UTF-8 lead bytes."""
    return [(byte, f"U+{cp:04X}") for byte, cp in _byte_symbol_codepoints()
            if chr(cp) not in vocab]


def tokenizer_json_shape(data: bytes) -> dict[str, object]:
    """The facts a conversion report needs about a tokenizer.json, before or after."""
    payload = json.loads(data.decode("utf-8"))
    model = payload.get("model", {})
    vocab = model.get("vocab", {})
    added = payload.get("added_tokens", [])
    ids = {int(v): str(k) for k, v in vocab.items()}
    added_ids = {int(e["id"]) for e in added if isinstance(e, Mapping) and "id" in e}
    redundant = [e for e in added if isinstance(e, Mapping) and
                 ids.get(int(e.get("id", -1))) == e.get("content")]
    unprotected = [i for i, t in ids.items() if i not in added_ids and _outside_byte_level(t)]
    return {
        "vocab": len(vocab),
        "added_tokens": len(added),
        "added_inside_vocab": sum(1 for i in added_ids if i in ids),
        "redundant_exact_redeclarations": len(redundant),
        "non_byte_level_vocab": sum(1 for t in vocab if _outside_byte_level(t)),
        "non_byte_level_unprotected": len(unprotected),
        "missing_byte_symbols": [f"0x{byte:02X}" for byte, _ in missing_byte_symbols(vocab)],
        "merges": len(model.get("merges", [])),
    }


def canonicalize_tokenizer_json(data: bytes) -> tuple[bytes, str]:
    """Return `(bytes, report)`: the checkpoint's tokenizer.json with every redundant
    EXACT re-declaration removed from `model.vocab`, and a one-line report of what moved.

    ⚠ NOT CALLED BY THE WRITE PATH -- see the F1227 block above `_byte_level_alphabet`. It is
    kept, tested and named because it is the only converter-side route to the
    "base-vocab-byte-level-only" shape for a checkpoint whose specials sit inside the vocab,
    and because a future checkpoint may need it; it is not emitted because HF `tokenizers`
    renumbers the tokens it moves (measured), so it is neutral for plain text and NOT neutral
    for a chat prompt.

    It is id- and merge-preserving by construction: no id moves (the id keeps its mapping
    through the added_tokens entry), `model.merges` is untouched, and every token it moves is
    CHECKED to be unproducible by BPE (`_bpe_load_bearing`: it appears in no merge as left,
    right or result and is not a byte-level symbol), so the engine's id sequences cannot move.
    Refuses by name when the shape is one this rewrite cannot make accepted.
    """
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ConversionRefused(f"frontend/tokenizer.json: not JSON ({exc})") from None
    model = payload.get("model")
    if not isinstance(model, Mapping) or model.get("type") != "BPE":
        raise ConversionRefused(
            "frontend/tokenizer.json: this rewrite is defined for a byte-level BPE "
            f"tokenizer.json (model.type BPE); got {model.get('type') if isinstance(model, Mapping) else None!r}")
    vocab = model.get("vocab")
    added = payload.get("added_tokens")
    if not isinstance(vocab, Mapping) or not isinstance(added, list):
        raise ConversionRefused(
            "frontend/tokenizer.json: model.vocab and added_tokens are both required")

    by_id: dict[int, str] = {}
    for token, index in vocab.items():
        if not isinstance(index, int) or index in by_id:
            raise ConversionRefused(
                f"frontend/tokenizer.json: model.vocab is not a 1:1 id map (entry "
                f"{token!r} -> {index!r})")
        by_id[index] = str(token)

    move: dict[str, int] = {}
    kept_added: list[Any] = []
    for entry in added:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), int) or \
                not isinstance(entry.get("content"), str):
            raise ConversionRefused(
                "frontend/tokenizer.json: malformed added_tokens entry (id/content)")
        token_id, content = entry["id"], entry["content"]
        if by_id.get(token_id) == content:
            move[content] = token_id
        kept_added.append(entry)

    load_bearing = _bpe_load_bearing(vocab, model.get("merges") or [])
    stuck = sorted(content for content in move if content in load_bearing)
    if stuck:
        raise ConversionRefused(
            "frontend/tokenizer.json: " + str(len(stuck)) +
            " token(s) are declared in added_tokens AND are load-bearing for BPE (they "
            "appear as a merge left/right/result, or are byte-level symbols), so removing "
            "them from model.vocab would move BPE output: " + ", ".join(repr(s) for s in stuck[:8]) +
            ". This shape needs the engine-side rule, not a rewrite; refusing rather than "
            "emitting an artifact that encodes differently")

    unprotectable = sorted(
        (index, token) for index, token in by_id.items()
        if token not in move and _outside_byte_level(token)
    )
    if unprotectable:
        raise ConversionRefused(
            "frontend/tokenizer.json: " + str(len(unprotectable)) +
            " model.vocab entry/entries are NOT byte-level decodable and are NOT declared "
            "in added_tokens, so the engine's byte-level decode refuses them and this "
            "rewrite cannot protect them: " +
            ", ".join(f"{index}({token!r})" for index, token in unprotectable[:8]))

    new_vocab = {token: index for token, index in vocab.items() if str(token) not in move}
    if not new_vocab:
        raise ConversionRefused("frontend/tokenizer.json: model.vocab would be empty")
    model["vocab"] = new_vocab
    payload["added_tokens"] = kept_added

    out = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    remaining = engine_tokenizer_json_reasons(out)
    if remaining:
        raise ConversionRefused(
            "frontend/tokenizer.json: the tree's own gate mirror still refuses the bytes "
            "this converter is about to write (" + "; ".join(remaining[:6]) + ")")
    report = (
        f"{len(move)} redundant re-declaration(s) removed from model.vocab, kept in "
        f"added_tokens at the same ids; ids moved: 0; merges touched: 0; {len(vocab)} -> "
        f"{len(new_vocab)} vocab entries; "
        f"{len([t for t in move if _outside_byte_level(t)])} of them were not byte-level "
        f"decodable and are protected by their added declaration; "
        f"{len(missing_byte_symbols(new_vocab))} byte-level symbol(s) absent (encode-time, "
        f"invalid UTF-8 lead bytes)"
    )
    return out, report


def frontend_payloads(
    resolved: Mapping[str, tuple[bytes | None, str]]
) -> list[tuple[str, bytes]]:
    """The six payloads, with every digest asserted, tokenizer.json canonicalised and
    tokenizer_config patched."""

    absent = sorted(name for name, (data, _where) in resolved.items() if not data)
    if absent:
        raise ConversionRefused(
            "frontend resources absent and unrecoverable: " + ", ".join(absent) +
            " (searched the checkpoint directory, every --resources root, and the pinned "
            "copies in " + str(PINNED_FRONTEND_DIR) + ")")
    # -------------------------------------------------------------------- F1227 ----
    # THE ENGINE'S TOKENIZER GATE, RUN IN THE CONVERTER.  Two things happen here and the
    # second one is deliberately NOT a rewrite:
    #
    # (1) the bytes about to be written are asserted with `engine_tokenizer_json_reasons`,
    #     the tree's restatement of every gate `Tokenizer` runs (the F-857 precedent above).
    #     A shape the engine will refuse is a NAMED REFUSAL HERE, in the conversion, instead
    #     of 8.9 GB written, uploaded to the device, and then refused at the front door --
    #     which is exactly how dl/sparkwriter's artifact died.
    #
    # (2) the checkpoint's own `model.vocab` is NOT rewritten.  `canonicalize_tokenizer_json`
    #     below exists and is tested, and it is NOT called: measured on this checkpoint, the
    #     reshape that moves the 113 redundant re-declarations out of `model.vocab` changes
    #     the ids the HuggingFace `tokenizers` library produces for ANY text containing one
    #     of those 113 strings (it renumbers them to the end of the vocab; `<think>` 3 ->
    #     130962, `<｜start▁of▁sentence｜>` 0 -> 130959, and so on), because an added token
    #     whose content is no longer a `model.vocab` entry is appended by HF rather than
    #     resolved to its declared id.  Plain-text probes cannot see it -- which is why the
    #     first neutrality control on this defect passed -- but a CHAT prompt is exactly text
    #     made of those strings.  The engine's rule (F1227, tokenizer.cpp: an added_tokens
    #     entry that repeats a model.vocab entry EXACTLY is not a conflict) reaches the same
    #     accepted shape without moving an id, and the checkpoint's own bytes then agree with
    #     HF and with the engine at once.
    tokenizer_resource = resolved.get("frontend/tokenizer.json", (None, ""))
    if tokenizer_resource[0] is not None:
        original = tokenizer_resource[0]
        shape = tokenizer_json_shape(original)
        print("  tokenizer.json: shape "
              f"(vocab={shape['vocab']} added_tokens={shape['added_tokens']} "
              f"inside_vocab={shape['added_inside_vocab']} "
              f"exact_redundant={shape['redundant_exact_redeclarations']} "
              f"non_byte_level={shape['non_byte_level_vocab']} "
              f"unprotected={shape['non_byte_level_unprotected']} "
              f"missing_byte_symbols={len(shape['missing_byte_symbols'])} "
              f"merges={shape['merges']})")
        reasons = engine_tokenizer_json_reasons(original)
        if reasons:
            raise ConversionRefused(
                f"frontend/tokenizer.json: the engine refuses these bytes at {len(reasons)} "
                f"gate(s), so writing them would produce an artifact that cannot load. "
                f"First: {reasons[0][:200]}. "
                f"`canonicalize_tokenizer_json` can reshape a byte-level BPE vocabulary, but "
                f"it moves no ids and is not applied silently: measure what it does to the "
                f"ids your reference implementation produces first (see the F1227 note in "
                f"this file). Full list: " + "; ".join(r[:120] for r in reasons[:4]))
        if shape["added_inside_vocab"]:
            print(f"  tokenizer.json: ACCEPTED with {shape['added_inside_vocab']} "
                  f"added_tokens id(s) inside model.vocab, {shape['redundant_exact_redeclarations']} "
                  f"of them repeating that vocab entry verbatim (engine rule F1227: an exact "
                  f"re-declaration is not a conflict, because the added declaration is what "
                  f"keeps a non-byte-level entry decodable)")
        if shape["missing_byte_symbols"]:
            print(f"  tokenizer.json: ADVISORY {len(shape['missing_byte_symbols'])} byte-level "
                  f"symbol(s) absent from model.vocab "
                  f"({', '.join(shape['missing_byte_symbols'][:16])}): encode-time only, and "
                  f"every one of them is an invalid UTF-8 lead byte, so no valid prompt can "
                  f"reach them")

    template: bytes | None = None
    ordered: list[tuple[str, bytes, str]] = []
    for filename in FRONTEND_FILES:
        name = f"frontend/{filename}"
        data, where = resolved[name]
        assert data is not None
        digest = hashlib.sha256(data).hexdigest()
        pinned = PINNED_FRONTEND_SHA256.get(filename)
        if filename == "chat_template.jinja":
            # The engine's own whitelist, from wherever the bytes came.
            if digest not in ACCEPTED_CHAT_TEMPLATE_SHA256:
                raise ConversionRefused(
                    f"{name}: {where} hashes {digest}, and the engine accepts exactly "
                    f"the two digests at chat_template.cpp:17-26 "
                    f"({ACCEPTED_CHAT_TEMPLATE_SHA256[0]}, "
                    f"{ACCEPTED_CHAT_TEMPLATE_SHA256[1]}); the frontend would refuse this "
                    f"artifact with \"unsupported frontend/chat_template.jinja\"")
            template = data
        if pinned is not None and where.endswith("(pinned)") and digest != pinned:
            raise ConversionRefused(
                f"{name}: the pinned copy {where} hashes {digest}, and this converter "
                f"records {pinned} -- the pinned frontend bytes moved under it")
        ordered.append((name, data, where))
    assert template is not None
    # tokenizer_config.json must CARRY the template it ships beside:
    # src/targets/qwen3_6/impl/frontend/frontend.cpp:215-223 requires the key to be a
    # string equal to `frontend/chat_template.jinja` and refuses with "does not match
    # frontend/chat_template.jinja" otherwise.  `added_tokens_decoder` must exist; this
    # checkpoint's own tokenizer_config.json has no such key (measured: 12 keys, no
    # added_tokens_decoder).
    out: list[tuple[str, bytes]] = []
    for name, data, _where in ordered:
        if name != "frontend/tokenizer_config.json":
            out.append((name, data))
            continue
        try:
            config = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConversionRefused(
                f"{name}: not a JSON object ({exc})") from None
        config["chat_template"] = template.decode("utf-8")
        config.setdefault("added_tokens_decoder", {})
        # ------------------------------------------------------------------ F-857 ----
        # THE GATE THIS ARTIFACT WAS REFUSED BY, READ OFF THE GATE ITSELF.
        # `validate_tokenizer_config` reads BOTH booleans with `value(k, true)`:
        #     frontend.cpp:205-206
        #         if (tokenizer_config.value("add_bos_token", true) ||
        #             tokenizer_config.value("add_prefix_space", true))
        # so an ABSENT key is read as TRUE and the gate throws at :208 with
        # "tokenizer_config.json does not match Qwen3.6 tokenizer prefix semantics".
        # This checkpoint declares `add_bos_token: false` and never mentions
        # `add_prefix_space`, which is why the artifact `dl/sparkwriter` wrote was
        # refused THERE -- with 0 stdout bytes, every one of its 369 objects and its
        # whole 8.28 GiB device upload already in place.
        # The SAME function then requires `pad_token` to be the STRING "<|endoftext|>"
        #     frontend.cpp:210-213
        #         if (!tokenizer_config.contains("pad_token") ||
        #             !tokenizer_config.at("pad_token").is_string() ||
        #             tokenizer_config.at("pad_token").get<std::string>() != "<|endoftext|>")
        # while this checkpoint's own `pad_token` is an AddedToken OBJECT, so the
        # artifact was armed with a SECOND throw one line further down in the SAME
        # function: repairing only the first key moves the refusal, it does not remove
        # it.  Both are repaired HERE, and the repair is CONVERTER-SIDE on purpose --
        # the engine's validator is the authority and NOTHING in it is weakened,
        # widened, bypassed or deleted.
        #
        # The source of both properties is upstream and is NOT a converter bug either:
        # models/Spark-X2.5-4B/tokenizer_config.json (4797 B) carries `add_bos_token:
        # false`, no `add_prefix_space` key at all, and `pad_token` as an AddedToken
        # object.  The previous converter propagated it faithfully.
        #
        # What the engine does with these three keys, measured (grep -rn over src/):
        # `add_bos_token` 1 hit, `add_prefix_space` 1 hit, `pad_token` 2 hits -- ALL FIVE
        # inside validate_tokenizer_config itself.  `bos_token` / `eos_token` /
        # `unk_token` / `tokenizer_class` have ZERO hits, so the AddedToken OBJECTS this
        # config carries for those are read by nothing.  `added_tokens_decoder` has one
        # real consumer (frontend/tokenizer.cpp:245 require_object_field), which the
        # setdefault above already satisfies.  So this repair cannot move any other
        # engine behaviour, and it is not a guess about what the gate wants: the gate is
        # quoted above.
        config["add_bos_token"] = False
        config["add_prefix_space"] = False
        config["pad_token"] = ENGINE_TOKENIZER_CONFIG_PAD_TOKEN

        payload = json.dumps(config, ensure_ascii=False).encode("utf-8")
        # ACCEPTANCE, TAKEN BY THE TREE'S OWN CHECKER AND NOT BY HAND.  These are the
        # exact bytes that will be written into frontend/tokenizer_config.json, and
        # `engine_tokenizer_config_reasons` is the tree's restatement of the gate the
        # engine runs at load; `template` is the chat_template.jinja the artifact will
        # ship beside them.  A non-empty tuple here is a refusal that names itself, in
        # the conversion, instead of 8.9 GB and a full device upload later.
        remaining = engine_tokenizer_config_reasons(payload, template)
        if remaining:
            raise ConversionRefused(
                f"{name}: the tree's own gate mirror still refuses the bytes this "
                f"converter is about to write (" + "; ".join(remaining) + ")")
        out.append((name, payload))
    return out


def build_specs(resources: Sequence[tuple[str, bytes]]) -> list[Any]:
    """The complete object inventory, in the order the payloads will be written."""

    specs: list[Any] = [ResourceSpec(name, RAW_BYTES_V1, len(data))
                        for name, data in resources]
    for name, shape in inventory.engine_objects():
        specs.append(TensorSpec(name, shape, inventory.BF16, inventory.CONTIGUOUS_LAYOUT))
    return specs


def tensor_payloads(
    reader: ShardReader, graph_only: bool = False
) -> Iterable[tuple[str, bytes]]:
    """Every tensor object's bytes, in the target's bind order.

    `graph_only` is the INJECTION SWITCH landq/sparkwriter's arm RED-W3 uses: with it
    set, the fused q/k/v tensor is written whole into EVERY one of its three objects
    instead of being split, i.e. exactly the defect the split exists to avoid.  It is
    what makes "the write path refuses a wrong payload" a reproducible reading rather
    than a claim, and the refusal comes from the container's own byte-count rule.
    """

    def emit(name: str, shape: Sequence[int], key: str,
             first_row: int | None,
             last_row: int | None) -> tuple[str, Iterable[bytes]]:
        if graph_only:
            first_row, last_row = None, None
        got = reader.payload_size(key, first_row, last_row)
        want = encoded_size(inventory.CONTIGUOUS_LAYOUT, inventory.BF16, shape)
        if got != want:
            raise ConversionRefused(
                f"{name}: {key} rows [{first_row}, {last_row}) is {got} B but "
                f"{inventory.BF16}/{inventory.CONTIGUOUS_LAYOUT} says {want} B for "
                f"{list(shape)} -- the write path and the object contract disagree")
        return name, reader.payload_chunks(key, first_row, last_row)

    yield emit(inventory.TOKEN_EMBEDDING_OBJECT, inventory.EMBEDDING_SHAPE,
               inventory.EMBED_KEY, None, None)
    for layer in range(inventory.LAYERS):
        for name, shape, key, first_row, last_row in inventory.layer_objects(layer):
            yield emit(name, shape, key, first_row, last_row)
    yield emit(inventory.FINAL_NORM_OBJECT, inventory.FINAL_NORM_SHAPE,
               inventory.FINAL_NORM_KEY, None, None)
    # The materialised head: the embedding's own bytes, under its own name.  The
    # orientation is the one `recipe.tie_decision()` reported, and for this checkpoint
    # it is identity -- the embedding header is already `[131072, 2560]`.
    yield emit(inventory.OUTPUT_HEAD_OBJECT, inventory.EMBEDDING_SHAPE,
               inventory.EMBED_KEY, None, None)


def write_artifact(model_dir: Path, out_path: Path,
                   resolved: Mapping[str, tuple[bytes | None, str]],
                   plan: Mapping[str, Any], graph_only: bool = False) -> dict:
    """Write the artifact, then RE-OPEN it with the module that owns the format.

    The re-open is the point.  `ArtifactWriter` publishes by rename and validates every
    payload's length as it goes (container.py:401-436), so a truncated or over-long
    payload cannot survive to the final name; and `Artifact.open` (container.py:292-326)
    then re-parses the magic, the directory and every offset, which is the only reading
    that answers "is this a .ninfer" rather than "did the writer return".
    """

    model_dir = Path(model_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resources = frontend_payloads(resolved)
    specs = build_specs(resources)
    identity = ArtifactIdentity(MODEL_ID, WEIGHTS_ID)

    index = read_index(model_dir)
    reader = ShardReader(model_dir, index.get("weight_map") or {})
    try:
        with ArtifactWriter(out_path, identity, specs) as writer:
            for name, data in resources:
                writer.write(name, data)
            for name, data in tensor_payloads(reader, graph_only=graph_only):
                writer.write(name, data)
    finally:
        reader.close()

    return verify_written(out_path, identity, specs, plan)


def _sha256_of_file(path: Path, block: int = 1 << 20, attempts: int = 8) -> str:
    """sha256 of a file, read in bounded blocks, with RETRIES.

    MEASURED TWICE, and the second measurement is why this is not a one-liner:
    reading a freshly written 8.9 GB artifact back over /mnt/c can fail with
    `OSError: [Errno 12] Cannot allocate memory` -- once on a single 52 MiB `read()`
    and once on a 4 MiB one, both times with ~20 GiB of host memory free, so it is
    NOT an allocation-size problem.  Refusing to retry would make this program
    report failure for a file that is on disk, complete, and openable by the same
    module's reader; the offset is tracked so a retry resumes exactly where the
    failed read left off, and a genuine truncation still fails after the attempts.
    """

    digest = hashlib.sha256()
    offset = 0
    with open(path, "rb") as handle:
        while True:
            chunk = b""
            for attempt in range(attempts):
                try:
                    handle.seek(offset)
                    chunk = handle.read(block)
                    break
                except OSError as exc:
                    if exc.errno != 12 or attempt == attempts - 1:
                        raise
                    time.sleep(0.5 * (attempt + 1))
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
            offset += len(chunk)


def verify_written(path: Path, identity: ArtifactIdentity, specs: Sequence[Any],
                   plan: Mapping[str, Any]) -> dict:
    """Read the file back and report what the tree's own reader says it is."""

    digest = _sha256_of_file(path)
    with Artifact.open(path) as artifact:
        names = [obj.name for obj in artifact.objects]
        planned = [spec.name for spec in specs]
        if names != planned:
            first = next((i for i, (a, b) in enumerate(zip(names, planned))
                          if a != b), None)
            raise ConversionRefused(
                "the artifact's object list does not match the plan: first difference "
                f"at index {first} ({len(names)} objects in the file, {len(planned)} "
                "planned)")
        tensors = [obj for obj in artifact.objects if obj.kind == "tensor"]
        resources = [obj for obj in artifact.objects if obj.kind == "resource"]
        tensor_bytes = sum(obj.bytes for obj in tensors)
        index_bytes = (plan.get("index_metadata") or {}).get("total_size")
        head_bytes = inventory.tensor_bytes(inventory.EMBEDDING_SHAPE)
        expected = None if index_bytes is None else index_bytes + head_bytes
        if expected is not None and tensor_bytes != expected:
            raise ConversionRefused(
                f"the artifact's tensor payload is {tensor_bytes} B, and the checkpoint "
                f"index's total_size {index_bytes} plus the materialised head "
                f"{head_bytes} is {expected} B (delta {tensor_bytes - expected})")
        return {
            "path": str(path),
            "bytes": artifact.file_bytes,
            "sha256": digest,
            "magic": MAGIC.hex(),
            "magic_ascii": repr(MAGIC),
            "model_id": artifact.identity.model_id,
            "weights_id": artifact.identity.weights_id,
            "objects": len(artifact.objects),
            "tensors": len(tensors),
            "resources": len(resources),
            "tensor_payload_bytes": tensor_bytes,
            "payload_offset": artifact.payload_offset,
            "expected_tensor_bytes": expected,
            "formats": sorted({obj.format for obj in tensors}),
            "layouts": sorted({obj.layout for obj in tensors}),
            "encodings": sorted({obj.encoding for obj in resources}),
        }


def _print_written(written: Mapping[str, Any]) -> None:
    print(f"  wrote         : {written['path']}")
    print(f"  identity      : {written['model_id']} / {written['weights_id']}")
    print(f"  magic         : {written['magic']} ({written['magic_ascii']})")
    print(f"  bytes         : {written['bytes']}")
    print(f"  sha256        : {written['sha256']}")
    print(f"  objects       : {written['objects']} = {written['tensors']} tensors + "
          f"{written['resources']} resources (payload offset "
          f"{written['payload_offset']})")
    print(f"  tensor bytes  : {written['tensor_payload_bytes']} "
          f"(index total_size + materialised head = "
          f"{written['expected_tensor_bytes']})")
    print(f"  formats/layouts/encodings: {written['formats']} / "
          f"{written['layouts']} / {written['encodings']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resources", action="append", default=None, type=Path,
                       help="frontend resource search root (repeatable); the source "
                            "directory is searched first")
    parser.add_argument("--repo-root", type=Path, default=None,
                       help="tree to read the engine's GQA alias header and the archkit "
                            "artifacts from (default: the tree this module lives in)")
    parser.add_argument("--plan", action="store_true",
                       help="print the plan and its gates, then stop (no torch, no GPU)")
    parser.add_argument("--graph-only", action="store_true",
                       help="INJECTION PROBE (landq/sparkwriter arm RED-W3): write the "
                            "fused q/k/v tensor whole into each of its three objects "
                            "instead of splitting it; the byte-count rule must refuse")
    args = parser.parse_args(argv)

    model_dir: Path = args.model.resolve()
    repo_root: Path = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        print(f"  no config.json under {model_dir}", file=sys.stderr)
        return 2
    config = json.loads(config_path.read_text(encoding="utf-8"))

    print(f"== {MODEL_ID}: validate_config")
    try:
        summary = validate_config(config)
    except ValueError as exc:
        print(f"  REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"  ok: {summary['architecture']} / {summary['model_type']} "
          f"({summary['layer_schedule']['full_attention']} full + "
          f"{summary['layer_schedule']['sliding_attention']} sliding)")
    print(f"  attention geometry: {summary['geometry']}")
    print(f"  per-kind rope     : {summary['per_kind_rope']}")
    print(f"  tie / act / qk_norm: {summary['tie_word_embeddings']} / "
          f"{summary['hidden_act']} / {summary['qk_norm']}")

    print(f"== {MODEL_ID}: plan")
    try:
        plan = build_plan(model_dir, repo_root)
    except ValueError as exc:
        # THE RENAMED REFUSAL, and it is a repair rather than a new gate.  `read_index`
        # (':233-237') and `load_headers` already raise ValueError whose first line names
        # the file that is missing or the shard that is short -- but the call to
        # `build_plan` was NOT inside the try that surrounds `validate_config`, so a
        # checkpoint without `model.safetensors.index.json` left `main` as an UNCAUGHT
        # traceback (main:585 -> read_index:236), which names the wrong thing: a Python
        # stack says "the program broke", not "this checkpoint is missing a file".  Both
        # forms are recorded in REPORT.md, before and after.
        print(f"  REFUSED: {exc}", file=sys.stderr)
        return 2
    _print_plan(plan)

    resources = resolve_resources(model_dir, _resource_roots(args.resources))
    missing = sorted(name for name, (data, _where) in resources.items() if not data)
    print(f"  frontend      : {len(resources) - len(missing)}/{len(FRONTEND_FILES)} resolvable"
          + (f"; missing {missing}" if missing else ""))

    gates_ok = (plan["coverage"]["ok"] and plan["shapes"]["ok"]
                and all(g["ok"] for g in plan["gates"].values()))
    plan["gates_ok"] = gates_ok
    if not gates_ok:
        print("== at least one gate is red; refusing to continue", file=sys.stderr)
        return 2

    if args.plan:
        print("== --plan: plan is consistent; nothing written")
        return 0

    print(f"== {MODEL_ID}: write")
    if args.graph_only:
        print("  INJECTION --graph-only: the fused q/k/v split is DISABLED on purpose",
              file=sys.stderr)
    try:
        written = write_artifact(model_dir, args.out, resources, plan,
                                 graph_only=args.graph_only)
    except ConversionRefused as exc:
        print(f"  REFUSED: {exc}", file=sys.stderr)
        return 2
    _print_written(written)
    if WORK_ITEMS:
        print("== the artifact is complete; what the ENGINE still needs, by name:")
        for name, detail in WORK_ITEMS:
            print(f"  - [{name}] {detail}")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
