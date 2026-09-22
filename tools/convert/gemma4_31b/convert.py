# -*- coding: utf-8 -*-
"""Convert a Gemma-4-31B checkpoint into one `.ninfer` artifact.

Canonical invocation::

    python -m tools.convert.gemma4_31b.convert \
      --model /path/to/gemma4-31B \
      --out out/gemma4_31b.ninfer \
      --resources <six-resource-dir> [--resources <dir> ...]

``--plan`` stops after the plan and its two gates, which is the mode that runs
today: it reads the checkpoint's own safetensors header and checks the recipe
against it, and it needs neither torch nor a GPU.

This module is also this target's answer to the import front door's contract
(``tools/convert/import_model.py``): ``evaluate_targets`` imports
``tools.convert.<target>.convert`` and calls the ``validate_config`` it finds
here, and the runnable gate reads ``SOURCE_QUANT_METHODS`` /
``SUPPLIES_FRONTEND_RESOURCES`` from the same module.  No second copy of these
numbers is kept in the front door.

What is NOT here
----------------
The artifact cannot be written yet, and this converter says so by name instead
of writing something the engine cannot load.  ``WORK_ITEMS`` is the list, and
``main`` refuses with it.  The first three are the reason the front door's
verdict for this source is still a work item rather than ``runnable``:

* the C++ target ``src/targets/gemma4_31b/`` does not exist.  It needs the
  eight files ``archkit_target.py:121-132`` enumerates plus the three
  registration edits (``:135-141``), one of which is
  ``src/CMakeLists.txt`` ``add_subdirectory``.
* the front door's own ``REGISTERED_TARGETS`` (``import_model.py:55-70``) and
  the GGUF family set (``:83-84``) do not name this target; both live in
  ``tools/convert/import_model.py``.
* no source reader speaks this checkpoint's NVFP4 field spelling (see
  ``recipe``'s module docstring for the four-way name map onto
  ``tools/convert/dequant/modelopt.py`` + ``tools/artifact/layouts.py``).

Two further engine-side items are named here because they are *architectural*
and would otherwise be discovered as silent wrong numbers:

* two attention geometries in one checkpoint (32/16/256 on the 50 sliding
  layers, 32/4/512 on the 10 full ones).  Neither triple is in the engine's GQA
  registry -- measured, verbatim, by ``archkit_target.py check``:

      == geometry 缺口: (32 q, 16 kv, 256 head_dim) 无匹配别名
         引擎注册别名: Gqa16x4Geometry(16,4,256), Gqa27Geometry(24,4,256),
                       Gqa35Geometry(16,2,256), GqaMuseGeometry(32,2,128)
         !! QHeads=32 但 head_dim 不同的别名: ['GqaMuseGeometry'] (分派无法区分, 需扩守卫)

* ``rotary_dim`` differs by layer kind (128 on the full layers, 256 on the
  sliding ones) while the shared ``TextConfig`` has a single slot, which is why
  ``gen_full_target.py:78-82`` refuses to emit a ``config.h`` for this spec at
  all.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

#: make `python3 tools/convert/gemma4_31b/convert.py` work as well as `-m ...`
if __package__ in (None, ""):                                   # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.convert.gemma4_31b import inventory, recipe      # noqa: F401
    from tools.convert.qwen3_6.common.conversion import check_members
else:
    from . import inventory, recipe
    from ..qwen3_6.common.conversion import check_members


MODEL_ID = inventory.MODEL_ID
WEIGHTS_ID = inventory.WEIGHTS_ID
TARGET_KEY = inventory.TARGET_KEY
RECIPE_ID = "gemma4_31b-v1"

#: This converter reads the quantised source directly (its packed words are
#: passed through), so the front door may run it on a ``compressed-tensors``
#: source.  A target that declares nothing keeps the ``native``-only default.
SOURCE_QUANT_METHODS = ("compressed-tensors",)

#: The artifact's six frontend resources come from the source directory and the
#: ``--resources`` search roots, so the source-side pin comparison *is*
#: evidence here (this is the ``False`` case, unlike Muse).
SUPPLIES_FRONTEND_RESOURCES = False

#: Config members the checkpoint must agree with.  ``src/targets/gemma4_31b``
#: would compile these in, so a checkpoint that disagrees would convert into an
#: artifact the engine runs with different semantics.
_ROOT_CONFIG = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "tie_word_embeddings": True,
    # measured from the source config.json; the vision tower is present and its
    # token ids are part of what the frontend must be able to resolve.
    "image_token_id": 258880,
    "video_token_id": 258884,
    "audio_token_id": 258881,
}

_TEXT_GEOMETRY = {
    "model_type": "gemma4_text",
    "hidden_size": inventory.HIDDEN,
    "num_hidden_layers": inventory.LAYERS,
    "intermediate_size": inventory.INTERMEDIATE,
    "num_attention_heads": inventory.Q_HEADS,
    "num_key_value_heads": inventory.KV_HEADS,
    "head_dim": inventory.HEAD_DIM,
    "num_global_key_value_heads": inventory.GLOBAL_KV_HEADS,
    "global_head_dim": inventory.GLOBAL_HEAD_DIM,
    "vocab_size": inventory.VOCAB,
    "max_position_embeddings": inventory.MAX_CTX,
    "rms_norm_eps": inventory.RMS_EPS,
    "sliding_window": inventory.SLIDING_WINDOW,
    "attention_bias": False,
    "tie_word_embeddings": inventory.TIE_WORD_EMBEDDINGS,
    "hidden_activation": inventory.HIDDEN_ACTIVATION,
    "final_logit_softcapping": inventory.FINAL_LOGIT_SOFTCAPPING,
    "attention_k_eq_v": True,
}

_TEXT_ABSENT = ("num_experts", "moe_intermediate_size", "top_k_experts")

_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "quantization_status": "compressed",
}

#: Reusable lines describing what is missing, shared by ``main``'s refusal and
#: by anything that wants to print this target's open work.
WORK_ITEMS = (
    ("C++ target", "src/targets/gemma4_31b/{CMakeLists.txt,impl/config.h,impl/variant.h,"
                   "impl/variant.cpp,impl/load/bindings.h,impl/load/bindings.cpp,"
                   "impl/package.cpp,export/ninfer/targets/gemma4_31b/package.h} "
                   "(archkit_target.py:121-132), plus registry.h / registry.cpp rows "
                   "(:135-141) and src/CMakeLists.txt add_subdirectory"),
    ("front-door registration", "tools/convert/import_model.py:55-70 REGISTERED_TARGETS += "
                                "\"gemma4_31b\" (and :83-84 DECODER_FAMILIES += \"gemma4\" "
                                "for the GGUF path)"),
    ("NVFP4 source reader", "a reader for this checkpoint's field spelling "
                            "(weight_packed/weight_scale/weight_global_scale/"
                            "input_global_scale) feeding tools/artifact/layouts.py "
                            "encode_nvfp4; the modelopt reader in "
                            "tools/convert/dequant/modelopt.py speaks the same four "
                            "quantities under the other naming"),
    ("GQA geometry", "(32 q, 16 kv, 256) and (32 q, 4 kv, 512) -- both unregistered; "
                     "src/ops/kernel/gqa_attention_geometry.cuh + the four dispatch sites "
                     "listed by archkit_target.py:246-248"),
    ("per-layer rotary", "rotary_dim 128 (full) vs 256 (sliding) in one checkpoint; the "
                         "shared TextConfig has one rotary_dim slot, which is why "
                         "gen_full_target.py:78-82 refuses to emit config.h"),
    ("per-layer scalar", "layers.N.layer_scalar [1] on all 60 layers; its multiply site "
                         "is still undecided (gemma_engine_plan.md:95-96)"),
)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the registered shape summary, or raise ValueError naming each mismatch.

    Pure config: it never opens a weight shard, which is what lets the front
    door route on it.
    """

    check_members("config", config, _ROOT_CONFIG)

    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise ValueError(
            "config.json must contain text_config (Gemma-4 nests the text tower)")
    check_members("text_config", text, _TEXT_GEOMETRY)

    for name in _TEXT_ABSENT:
        if text.get(name) is not None:
            raise ValueError(
                f"text_config.{name}: expected None, got {text.get(name)!r} "
                f"(this checkpoint has no MoE block; {len(inventory.LAYERS)} dense layers)")

    expected_layer_types = tuple(
        "full_attention" if layer in inventory.FULL_ATTENTION_LAYERS else "sliding_attention"
        for layer in range(inventory.LAYERS)
    )
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != expected_layer_types:
        got = None if not isinstance(layer_types, list) else (
            sum(1 for k in layer_types if k == "full_attention"),
            sum(1 for k in layer_types if k == "sliding_attention"))
        raise ValueError(
            "text_config.layer_types does not match the registered "
            f"{inventory.LAYERS}-layer schedule "
            f"(full_attention at {', '.join(str(l) for l in inventory.FULL_ATTENTION_LAYERS)}); "
            f"expected (10 full, 50 sliding), got {got}")

    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    sliding_rope = rope.get("sliding_attention")
    full_rope = rope.get("full_attention")
    if not isinstance(sliding_rope, Mapping) or not isinstance(full_rope, Mapping):
        raise ValueError(
            "text_config.rope_parameters must name both sliding_attention and full_attention "
            "(this checkpoint's two layer kinds carry different theta)")
    if float(sliding_rope.get("rope_theta", 0.0)) != inventory.ROPE_THETA_BY_KIND["sliding"]:
        raise ValueError(
            f"text_config.rope_parameters.sliding_attention.rope_theta: expected "
            f"{inventory.ROPE_THETA_BY_KIND['sliding']!r}, got "
            f"{sliding_rope.get('rope_theta')!r}")
    if float(full_rope.get("rope_theta", 0.0)) != inventory.ROPE_THETA_BY_KIND["full"]:
        raise ValueError(
            f"text_config.rope_parameters.full_attention.rope_theta: expected "
            f"{inventory.ROPE_THETA_BY_KIND['full']!r}, got {full_rope.get('rope_theta')!r}")
    if float(full_rope.get("partial_rotary_factor", 1.0)) * inventory.GLOBAL_HEAD_DIM != \
            inventory.ROTARY_DIM_BY_KIND["full"]:
        raise ValueError(
            f"text_config.rope_parameters.full_attention.partial_rotary_factor: expected "
            f"{inventory.ROTARY_DIM_BY_KIND['full'] / inventory.GLOBAL_HEAD_DIM!r} of "
            f"global_head_dim {inventory.GLOBAL_HEAD_DIM}, got "
            f"{full_rope.get('partial_rotary_factor')!r}")

    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping):
        raise ValueError("config.json has no quantization_config (NVFP4 source expected)")
    check_members("quantization_config", quant, _QUANT_CONFIG)
    groups = quant.get("config_groups")
    if not isinstance(groups, Mapping) or len(groups) != 1:
        raise ValueError(
            f"quantization_config.config_groups: expected exactly 1 group, got "
            f"{None if not isinstance(groups, Mapping) else len(groups)}")
    group_name, group = next(iter(groups.items()))
    weights = group.get("weights") if isinstance(group, Mapping) else None
    if not isinstance(weights, Mapping):
        raise ValueError(f"quantization_config.config_groups.{group_name}.weights is missing")
    expected_weight = {"num_bits": 4, "group_size": 16, "type": "float",
                      "strategy": "tensor_group", "scale_dtype": "torch.float8_e4m3fn"}
    check_members(f"quantization_config.config_groups.{group_name}.weights",
                  weights, expected_weight)

    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "artifact_identity": {"model_id": MODEL_ID, "weights_id": WEIGHTS_ID},
        "text": {name: text[name] for name in _TEXT_GEOMETRY},
        "layer_schedule": {
            "layers": inventory.LAYERS,
            "full_attention": len(inventory.FULL_ATTENTION_LAYERS),
            "sliding_attention": len(inventory.SLIDING_ATTENTION_LAYERS),
            "full_attention_slots": list(inventory.FULL_ATTENTION_LAYERS),
        },
        "attention_geometries": {
            "sliding": {"query_heads": inventory.Q_HEADS, "kv_heads": inventory.KV_HEADS,
                        "head_dim": inventory.HEAD_DIM,
                        "rotary_dim": inventory.ROTARY_DIM_BY_KIND["sliding"]},
            "full": {"query_heads": inventory.Q_HEADS,
                     "kv_heads": inventory.GLOBAL_KV_HEADS,
                     "head_dim": inventory.GLOBAL_HEAD_DIM,
                     "rotary_dim": inventory.ROTARY_DIM_BY_KIND["full"]},
        },
        "quantization": {"quant_method": quant["quant_method"], "format": quant["format"],
                         "group": group_name, "ignore_entries": len(quant.get("ignore") or [])},
        "tie_word_embeddings": text["tie_word_embeddings"],
    }


def read_safetensors_header(path: Path) -> dict:
    """Names, dtypes and shapes only -- the payload is never read."""

    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def build_plan(model_dir: Path) -> dict:
    """The plan plus its two gates: recipe coverage and shape agreement."""

    shard = model_dir / "model.safetensors"
    if not shard.is_file():
        shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
        if not shards:
            raise SystemExit(f"no *.safetensors under {model_dir}")
        if len(shards) > 1:
            raise SystemExit(
                "this recipe reads a single model.safetensors; "
                f"{model_dir} has {len(shards)} shards {shards} -- an index-aware reader is "
                "part of the NVFP4 source-reader work item")
        shard = model_dir / shards[0]
    header = read_safetensors_header(shard)
    return {
        "shard": shard.name,
        "source_tensor_count": len(header),
        "objects": recipe.planned_objects(),
        "object_count": len(recipe.OBJECT_RECIPES),
        "format_counts": inventory.format_counts(),
        "layout_counts": inventory.layout_counts(),
        "payload_bytes": inventory.text_tensor_span_bytes(),
        "coverage": recipe.validate_coverage(header.keys()),
        "shapes": recipe.shape_check(header),
        "not_carried": inventory.nmt_objects(),
        "vision": inventory.VISION_STATUS,
    }


def _resource_roots(cli_roots: Sequence[Path] | None) -> tuple[Path, ...]:
    """``--resources`` first, then ``$NINFER_RESOURCE_ROOTS``.

    The import front door prints this converter's canonical command with the
    roots in the *environment* rather than on the command line
    (``import_model.py:960-968``: "--resource-root is THIS file's flag; the
    converter named in the command reads its six frontend resources from its own
    --resources or from $NINFER_RESOURCE_ROOTS"), so both spellings are read
    here or the printed command would not be runnable as printed.
    """

    roots: list[Path] = [Path(root) for root in (cli_roots or ())]
    for entry in os.environ.get("NINFER_RESOURCE_ROOTS", "").split(os.pathsep):
        if entry and Path(entry) not in roots:
            roots.append(Path(entry))
    return tuple(roots)


def resolve_resources(model_dir: Path, roots: Sequence[Path]) -> dict:
    """Where each of the six frontend resources would come from.

    Source directory first (the window's ``import_model.py`` ⑤ measures the same
    way), then the search roots in the order given.
    """

    resolved = {}
    for name in inventory.FRONTEND_RESOURCE_NAMES:
        filename = name.removeprefix("frontend/")
        found = None
        candidate = model_dir / filename
        if candidate.is_file() and candidate.stat().st_size:
            found = candidate
        else:
            for root in roots:
                candidate = Path(root) / filename
                if candidate.is_file() and candidate.stat().st_size:
                    found = candidate
                    break
        resolved[name] = None if found is None else str(found)
    return resolved


def _print_plan(plan: dict, resources: dict) -> None:
    print(f"  identity      : {MODEL_ID} / {WEIGHTS_ID}   (target_key {TARGET_KEY})")
    print(f"  shard         : {plan['shard']} ({plan['source_tensor_count']} source tensors)")
    print(f"  objects       : {plan['object_count']} planned "
          f"({sum(plan['format_counts'].values())} carrying bytes + "
          f"{plan['object_count'] - sum(plan['format_counts'].values())} view aliases)")
    print(f"  formats       : {plan['format_counts']}")
    print(f"  layouts       : {plan['layout_counts']}")
    print(f"  payload       : {plan['payload_bytes'] / 2**30:.2f} GiB (planned)")
    coverage = plan["coverage"]
    print(f"  coverage      : {coverage['consumed_key_count']}/{coverage['planned_key_count']} "
          f"planned keys found; {coverage['ignored_key_count']} keys ignored by prefix; "
          f"{len(coverage['unplanned_source_keys'])} unplanned; "
          f"{len(coverage['missing_source_keys'])} missing")
    for key in coverage["unplanned_source_keys"][:10]:
        print(f"    UNPLANNED  {key}")
    for key in coverage["missing_source_keys"][:10]:
        print(f"    MISSING    {key}")
    shapes = plan["shapes"]
    print(f"  shape gate    : {shapes['checked']} objects compared, "
          f"{len(shapes['conflicts'])} conflict(s)")
    for line in shapes["conflicts"][:10]:
        print(f"    CONFLICT   {line}")
    missing_resources = [name for name, where in resources.items() if where is None]
    print(f"  frontend      : {len(resources) - len(missing_resources)}/6 resolvable"
          + (f"; missing {missing_resources}" if missing_resources else ""))
    for name, where in resources.items():
        if where:
            print(f"    {name} <- {where}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resources", action="append", default=None, type=Path,
                        help="frontend resource search root (repeatable); the source "
                             "directory is searched first")
    parser.add_argument("--plan", action="store_true",
                        help="print the plan and its gates, then stop (no torch, no GPU)")
    args = parser.parse_args(argv)

    model_dir: Path = args.model.resolve()
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
    print(f"  attention geometries: {summary['attention_geometries']}")
    print(f"  quantization: {summary['quantization']}")

    print(f"== {MODEL_ID}: plan")
    plan = build_plan(model_dir)
    resources = resolve_resources(model_dir, _resource_roots(args.resources))
    _print_plan(plan, resources)

    if not plan["coverage"]["ok"] or not plan["shapes"]["ok"]:
        print("== the recipe does not match this checkpoint; refusing to continue",
              file=sys.stderr)
        return 2

    if args.plan:
        print("== --plan: plan is consistent; nothing written")
        return 0

    print("== cannot materialise this artifact yet; open work, by name:")
    for name, detail in WORK_ITEMS:
        print(f"  - [{name}] {detail}")
    print("  Neither the C++ target nor the front-door registration exists, so an "
          "artifact written now could not be loaded by any engine build.  This "
          "converter refuses instead of producing a file nothing can open.")
    return 3


if __name__ == "__main__":                                      # pragma: no cover
    raise SystemExit(main())
