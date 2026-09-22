"""Convert a *text-core-only* source of the registered 27B geometry.

Why this module exists
----------------------
`tools/convert/gguf_extract.py` writes an HF directory whose tensor names are the
text tower's (`model.language_model.*`) and whose `config.json` is the flat text
config -- for the Bonsai-27B GGUF family that is 851 BF16 tensors.  That source is
a strict prefix of this target's registered source: `tools/convert/qwen3_6_27b/
recipe.py` consumes every one of its tensors and nothing else, and
`src/targets/qwen3_6_27b/impl/config.h`'s `TextConfig` equals its geometry field
for field (hidden 5120, 64 layers, 16 full + 48 GDN, intermediate 17408,
24 query / 4 kv heads x 256, GDN 16x128 key / 48x128 value, conv 4, vocab 248320).

What it does *not* have, and what that means
--------------------------------------------
The registered artifact carries 1124 objects: 771 text core + 2 draft head + 12
MTP + 333 vision + 6 resources.  A text-core extraction has no source tensors for
any of the other 355, so the only artifact that can be *built* from it is the
text core.  This module therefore closes the object set on **what the source
actually carries** instead of on the constant:

* the 12 ``mtp/*`` objects are included iff the source carries the MTP block;
* the 2 ``text/draft_head*`` objects are included iff the source carries a
  shortlist-bearing path (this source does not);
* the 333 vision objects are included iff the source carries a vision tower.

The closure is *reported*, never silently widened: the conversion report records
`object_closure` and the measured reason for each group it left out.  A source
that carries the groups gets the registered 1124-object closure, so the
registered path is unchanged -- see `tools/convert/qwen3_6_27b/verify.py` for the
byte-level control.

Frontend resources
------------------
Resolved through ``tools.convert.qwen3_6.common.frontend_policy`` -- the same
policy ``import_model.py`` uses for its section 5 measurement -- so the converter
and the front door cannot disagree about the six files.  Search roots come from
``--resources`` or ``$NINFER_RESOURCE_ROOTS``.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as common_recipe
from tools.convert.qwen3_6.common import frontend_policy
from tools.convert.qwen3_6.common.conversion import ResourcePayload

from . import inventory, recipe


#: The registered text geometry this module accepts, as the flat root config
#: declares it.  Keys are the HF spellings; values are the registered numbers.
#: Every entry is also present in ``src/targets/qwen3_6_27b/impl/config.h``.
TEXT_ONLY_CONFIG: dict[str, object] = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "tie_word_embeddings": False,
    "rope_theta": 10000000.0,
}

#: Root keys that mean "this is the registered multimodal checkpoint, not a text core".
VISION_MARKERS: tuple[str, ...] = (
    "vision_config",
    "vision_start_token_id",
    "vision_end_token_id",
    "image_token_id",
    "video_token_id",
)


class TextCoreRefused(RuntimeError):
    """Raised with the missing or inconsistent part named."""


# --------------------------------------------------------------------------- #
# source reading
# --------------------------------------------------------------------------- #
def source_tensor_names(model_dir: str | Path) -> tuple[str, ...]:
    """Every tensor name in the source, read from the safetensors header only.

    Reading the header is enough for every decision this module makes (which
    groups exist, which layers are full attention) and touches no tensor data,
    so a refusal costs nothing.
    """

    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = family_conversion.load_json(index)["weight_map"]
        return tuple(sorted(weight_map))
    single = root / "model.safetensors"
    if not single.is_file():
        raise TextCoreRefused(
            f"{root} has neither model.safetensors nor model.safetensors.index.json")
    with single.open("rb") as handle:
        header_bytes = handle.read(8)
        if len(header_bytes) != 8:
            raise TextCoreRefused(f"{single} is shorter than its 8-byte header length")
        (length,) = struct.unpack("<Q", header_bytes)
        if length == 0 or length > 1 << 30:
            raise TextCoreRefused(f"{single} declares a {length}-byte header")
        header = json.loads(handle.read(length).decode("utf-8"))
    return tuple(sorted(name for name in header if name != "__metadata__"))


def _layer_of(name: str) -> int | None:
    marker = "model.language_model.layers."
    if not name.startswith(marker):
        return None
    rest = name[len(marker):]
    head, _, _ = rest.partition(".")
    return int(head) if head.isdigit() else None


def full_attention_layers(names: Sequence[str]) -> tuple[int, ...]:
    """The layers carrying `self_attn.q_proj.weight`, i.e. the measured schedule.

    The source's flat config declares no `layer_types`, so the registered
    schedule is a *claim* until it is checked against the tensors.  This is that
    check, and its result is what `validate_text_config` requires.
    """

    found = sorted({layer for name in names
                    if name.endswith(".self_attn.q_proj.weight")
                    and (layer := _layer_of(name)) is not None})
    return tuple(found)


@dataclass(frozen=True, slots=True)
class Closure:
    """The object set this source can actually build, with the reason per group."""

    object_specs: tuple[Any, ...]
    recipes: tuple[Any, ...]
    included: tuple[str, ...]
    omitted: tuple[tuple[str, str], ...]

    @property
    def tensor_count(self) -> int:
        return len(self.recipes)


def measure_closure(names: Sequence[str], specs_module: Any | None = None) -> Closure:
    """Decide the closure from the source's own tensor names.

    ``specs_module`` is the inventory module that owns the STORAGE ROLES.  It
    defaults to this package's own ``tools.convert.qwen3_6_27b.inventory``, so
    the registered Qwen3.6-27B call sites are unchanged.  ``tools/convert/
    qwen3_8_27b/convert.py`` passes its own module instead, because the same 27B
    text core written under the identity ``qwen3.8-27b`` + ``groupwise-int``
    records a DIFFERENT numeric format for its two vocabulary endpoints:
    ``endpoint_format(Qwen38GroupwiseInt)`` is ``W8G32_F16S``
    (``src/targets/qwen3_6_27b/impl/load/bindings.cpp:38-40``) while the
    Qwen3.6-27B row is ``Q6G64_F16S`` (``:36-37``), and the Qwen3.8 row's own
    provenance is what ``package.h:189-192`` names -- this module's default
    would store Q6 descriptors under a W8 contract, which the binder refuses
    (``src/artifact/binder.cpp:50-54``).  Only the storage role moves: the
    source tensors, the recipes and the materialized values are identical.
    """

    owner = inventory if specs_module is None else specs_module
    has_mtp = any(name.startswith("mtp.") or name.startswith("mtp/") for name in names)
    has_vision = any(name.startswith("vision_tower.") or name.startswith("vision.")
                     for name in names)

    text_recipes = recipe.TEXT_CORE_RECIPES
    specs: list[Any] = list(owner.RESOURCE_SPECS) + list(owner.TEXT_CORE_TENSOR_SPECS)
    recipes_: list[Any] = list(text_recipes)
    included = ["text_core(771)", "resources(6)"]
    omitted: list[tuple[str, str]] = []

    if has_mtp:
        specs.extend(owner.MTP_TENSOR_SPECS)
        recipes_.extend(recipe.MTP_RECIPES)
        included.append("mtp(12)")
    else:
        omitted.append(("mtp(12)", "源件没有 MTP 块（无 mtp.* 张量）"))

    # The shortlist head is a shortlist over `lm_head` rows plus a token-id table;
    # it needs no checkpoint tensor of its own, but its id table is derived from
    # the ranking dictionary *for the registered token domain* and the C++ side
    # declares it absent for this weights profile (package.cpp:104-105
    # `shortlist_head_declared(Qwen36GroupwiseInt) == false`).  It is omitted here
    # for the same reason the profile omits it, and the omission is named.
    omitted.append(("draft_head(2)", "该 weights profile 不声明短名单头"
                                     "（src/targets/qwen3_6_27b/impl/package.cpp:104-105）"))

    if has_vision:
        specs.extend(owner.VISION_TENSOR_SPECS)
        recipes_.extend(recipe.VISION_RECIPES)
        included.append("vision(333)")
    else:
        omitted.append(("vision(333)", "源件没有视觉塔（无 vision_tower.* 张量）"))

    return Closure(tuple(specs), tuple(recipes_), tuple(included), tuple(omitted))


def open_source(model_dir: str | Path) -> ShardReader:
    """A reader for either source layout, chosen from what is on disk.

    `ShardReader(model_dir)` reads `model.safetensors.index.json` and raises
    FileNotFoundError on a single-file source -- measured on
    /var/tmp/type41land/AFTER/full_q1, which has one 53.79 GB
    `model.safetensors` and no index.  `ShardReader.from_file` is the single-file
    constructor that already exists for exactly this case, so the layout is
    detected rather than assumed.
    """

    root = Path(model_dir)
    if (root / "model.safetensors.index.json").is_file():
        return ShardReader(root)
    single = root / "model.safetensors"
    if single.is_file():
        return ShardReader.from_file(single)
    raise TextCoreRefused(
        f"{root} has neither model.safetensors.index.json nor model.safetensors")


def verify_coverage(closure: Closure, names: Sequence[str]) -> dict[str, object]:
    """Require the closure to use the source exactly: nothing missing, nothing spare.

    This is the property that makes a *closed* object set safe to introduce.  A
    closure that silently used a tensor the source does not have would fail deep
    inside materialization with a message about one tensor; a closure that left a
    source tensor unconsumed would ship an artifact that quietly dropped part of
    the checkpoint.  Both are computed here from the recipes themselves (the same
    `source_requirements` the registered path uses) and both must be empty.
    """

    required = common_recipe.source_requirements(closure.recipes)
    required_names = set(required)
    present = set(names)
    missing = sorted(required_names - present)
    unconsumed = sorted(present - required_names)
    if missing or unconsumed:
        raise TextCoreRefused(
            f"closure does not match the source: {len(missing)} required tensor(s) absent "
            f"{missing[:6]}, {len(unconsumed)} source tensor(s) unconsumed {unconsumed[:6]}")
    return {"required_source_tensors": len(required_names),
            "source_tensors": len(present),
            "missing": [], "unconsumed": []}


# --------------------------------------------------------------------------- #
# config acceptance
# --------------------------------------------------------------------------- #
def looks_like_registered_multimodal(config: Mapping[str, Any]) -> bool:
    return isinstance(config.get("text_config"), Mapping) or any(
        key in config for key in VISION_MARKERS)


def validate_text_config(config: Mapping[str, Any]) -> dict[str, object]:
    """Accept the flat text-only config of the registered 27B geometry, or refuse.

    Field-by-field against `TEXT_ONLY_CONFIG`; a source that declares a
    *different* number is refused by name.  A source that declares the
    registered multimodal shell is refused here too, so the caller can hand it to
    the registered `validate_config` instead of this module guessing.
    """

    if looks_like_registered_multimodal(config):
        raise TextCoreRefused(
            "this source declares the registered multimodal shell "
            "(text_config/vision_config or vision_* token ids); it is not a text core")

    _check("config", config, TEXT_ONLY_CONFIG)
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": {name: config[name] for name in TEXT_ONLY_CONFIG
                 if name not in ("architectures", "model_type")},
    }


def _check(scope: str, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    mismatches = [
        f"{scope}.{name}: expected {value!r}, got {actual.get(name)!r}"
        for name, value in expected.items()
        if actual.get(name) != value
    ]
    if mismatches:
        raise TextCoreRefused("text-core config mismatch:\n  " + "\n  ".join(mismatches))


# --------------------------------------------------------------------------- #
# resources
# --------------------------------------------------------------------------- #
def resource_roots(explicit: Sequence[str | Path] | None) -> list[Path]:
    if explicit:
        return [Path(root) for root in explicit]
    return [Path(root) for root in
            os.environ.get("NINFER_RESOURCE_ROOTS", "").split(os.pathsep) if root]


def load_frontend_resources(
    model_dir: str | Path,
    config: Mapping[str, Any],
    roots: Sequence[str | Path] | None,
    specs_module: Any | None = None,
) -> tuple[tuple[ResourcePayload, ...], dict[str, object]]:
    """Resolve the six pinned resources through the front door's own policy."""

    owner = inventory if specs_module is None else specs_module
    search = resource_roots(roots)
    spec_names = tuple(spec.name for spec in owner.RESOURCE_SPECS)
    if spec_names != frontend_policy.FRONTEND_NAMES:
        raise TextCoreRefused(
            "the converter's resource order differs from the policy's: "
            f"{spec_names!r} vs {frontend_policy.FRONTEND_NAMES!r}")
    profile = frontend_policy.resolve_frontend_profile(model_dir, config, roots=search)
    verdict = frontend_policy.acceptability_error(profile, allow_unproven=False)
    if verdict is not None:
        raise TextCoreRefused(
            f"{verdict}；已试搜索根 {[str(r) for r in search] or '（未配置）'}；"
            "补救：--resources <dir>（可重复）或 NINFER_RESOURCE_ROOTS=<dir>[,...]")
    payloads = []
    for spec in owner.RESOURCE_SPECS:
        resolution = profile.by_name[spec.name]
        if resolution.provider is None:
            raise TextCoreRefused(f"{spec.name} resolved without a provider")
        data = resolution.provider.read_bytes()
        if not data:
            raise TextCoreRefused(f"{resolution.provider} is empty")
        payloads.append(ResourcePayload(spec.name, data))
    summary = {
        name: {"status": item.status, "provider": str(item.provider),
               "sha256": item.sha256}
        for name, item in profile.by_name.items()
    }
    return tuple(payloads), summary


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    resources: Sequence[str | Path] | None = None,
    model_id: str = inventory.MODEL_ID,
    weights_id: str = inventory.WEIGHTS_ID,
    target_key: str = inventory.TARGET_KEY,
    recipe_id: str = "qwen3_6_27b-text-core-v1",
    specs_module: Any | None = None,
) -> Path:
    # The storage roles the writer writes come from `specs_module`; see
    # `measure_closure` for why the Qwen3.8-27B row passes its own module and the
    # registered Qwen3.6-27B row keeps the default.  The writer's object plan, the
    # materialized-shape check and the encoder all read the SAME spec, so the
    # identity that gets written and the descriptors that get stored cannot come
    # from two different modules.
    owner = inventory if specs_module is None else specs_module
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)

    config = family_conversion.load_json(model / "config.json")
    config_summary = validate_text_config(config)

    names = source_tensor_names(model)
    closure = measure_closure(names, specs_module)

    coverage = verify_coverage(closure, names)

    measured_full = full_attention_layers(names)
    if measured_full != owner.FULL_ATTENTION_LAYERS:
        raise TextCoreRefused(
            "the source's measured full-attention layers "
            f"{measured_full!r} are not the registered schedule "
            f"{owner.FULL_ATTENTION_LAYERS!r}")

    payloads, resource_summary = load_frontend_resources(model, config, resources,
                                                         specs_module)
    resource_map = {payload.name: payload.data for payload in payloads}
    object_plan = family_conversion.build_object_plan(closure.object_specs, resource_map)

    print(f"text-core source: {model}")
    print(f"  tensors={len(names)}  full_attention_layers={measured_full!r}  "
          f"objects={len(object_plan.objects)}")
    print(f"  闭包包含: {', '.join(closure.included)}")
    for group, why in closure.omitted:
        print(f"  闭包排除: {group} —— {why}")
    print(f"  device={resolved_device}")

    output.parent.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, object]] = []
    # The closure adds groups (mtp, vision) that `TEXT_CORE_RECIPES_BY_NAME` does
    # not describe, so the recipe for an object is taken from the closure that
    # asked for it.  For a text-core-only source the two sets hold the same
    # objects, so this selects exactly what the old map selected.
    recipes_by_name = {item.object_name: item for item in closure.recipes}
    with open_source(model) as reader:
        with ArtifactWriter(output, ArtifactIdentity(model_id, weights_id),
                            object_plan.specs) as writer:
            if writer.objects != object_plan.objects:
                raise TextCoreRefused("writer object plan differs from the completed plan")
            for index, spec in enumerate(closure.object_specs, start=1):
                if isinstance(spec, owner.ResourceSpec):
                    payload = resource_map[spec.name]
                else:
                    tensor = recipe.materialize_recipe(
                        recipes_by_name[spec.name], reader, None)
                    if tuple(tensor.shape) != spec.shape:
                        raise TextCoreRefused(
                            f"{spec.name}: materialized shape {tuple(tensor.shape)} "
                            f"!= {spec.shape}")
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                written.append({"name": spec.name,
                                "bytes": len(payload) if isinstance(payload, (bytes, bytearray))
                                else writer.objects[index - 1].bytes})
                del payload
                if index % 25 == 0 or index == len(closure.object_specs):
                    print(f"  [{index}/{len(closure.object_specs)}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    report = {
        "recipe_id": recipe_id,
        "identity": {"model_id": model_id, "weights_id": weights_id},
        "target_key": target_key,
        "specs_module": owner.__name__,
        "source_dir": str(model),
        "source_tensors": len(names),
        "out_path": str(output),
        "out_bytes": final_bytes,
        "elapsed_seconds": elapsed,
        "device": str(resolved_device),
        "config_summary": config_summary,
        "object_closure": {
            "included": list(closure.included),
            "omitted": [{"group": group, "reason": why} for group, why in closure.omitted],
            "objects": len(object_plan.objects),
        },
        "full_attention_layers_measured": list(measured_full),
        "source_coverage": coverage,
        "frontend_resources": resource_summary,
        "objects": written,
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"complete: {final_bytes} bytes in {elapsed:.1f}s -> {output}")
    print(f"report: {report_path}")
    return report_path


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()
