"""Build a `.ninfer` artifact from a SINGLE ModelOpt-quantized HF checkpoint.

WHY THIS IS A SEPARATE ENTRY POINT
----------------------------------
``tools/convert/qwen3_8_27b/convert_nvfp4.py`` is a closed *dual source* contract:
it reads an unquantized official checkpoint (``--model``) plus a
``compressed-tensors`` mixed-precision checkpoint (``--quantized-model``) whose
config declares ``group_0`` (FP8: ``weight`` + BF16 per-row ``weight_scale``) and
``group_1`` (NVFP4: ``weight_packed`` + ``weight_scale`` + ``weight_global_scale``
+ ``input_global_scale``).  The source handled here is neither of those: it is a
single ModelOpt 0.43 directory whose declaration (``hf_quant_config.json``,
cross-checked against ``config.json.quantization_config``) is ``MIXED_PRECISION``
with a per-layer ``quant_algo``, and whose quantized ``Linear`` tensors use the
ModelOpt field vocabulary::

    NVFP4 group    weight         U8       (N, K/2)   packed E2M1 nibbles
                   weight_scale   F8_E4M3  (N, K/16)  natural (unswizzled) scales
                   weight_scale_2 F32      ()         per-tensor weight multiplier
                   input_scale    F32      ()         per-tensor activation multiplier
    FP8 group      weight         F8_E4M3  (N, K)     E4M3 words
                   weight_scale   F32      ()         per-tensor weight multiplier
                   input_scale    F32      ()         per-tensor activation multiplier

Everything else (norms, ``conv1d``, ``A_log``, ``dt_bias``, ``in_proj_a/b``,
``embed_tokens``, ``lm_head``, all of ``model.visual.*`` and ``mtp.*``) is BF16.

This module supplies the missing half for that checkpoint shape: a
**single-source object-assembly driver**.  It re-derives no numeric rule.
``tools/artifact/layouts.py`` owns the layouts and their encoders,
``tools/convert/common/quantize.py`` owns the grouped quantizers the artifact
uses for objects it re-quantizes, and ``tools/convert/common/layout_plan.py``
owns the structural layout decision.

WHAT EACH MAPPING IS JUSTIFIED BY
---------------------------------
1. **The layout of a source group** comes from ``layout_plan.select``: it keys on
   *this source's* dtype/shape relations inside one quantized group (``U8[N,K/2]``
   codes with ``F8_E4M3[N,K/16]`` scales and a code/scale column ratio of 8, or
   ``F8_E4M3[N,K]`` codes with an ``F32`` scale), cross-checked against the
   per-layer ``quant_algo`` the source itself declares.  No model name, layer name
   or key allowlist takes part in that decision; a group no layout describes is
   refused with the missing mechanism named.
2. **Which source matrices compose one artifact object** is an *artifact* fact,
   not a source fact: the fused row geometry (``mlp/gate_up`` = the rows of
   ``mlp.gate_proj`` then ``mlp.up_proj``; ``attention/query_key_gate_value`` =
   the head-major ``[q|gate]`` split of ``q_proj`` then ``k_proj`` then
   ``v_proj``) is owned by ``recipe_nvfp4`` and is reused verbatim through
   ``recipe_nvfp4.FP8_WEIGHT_RECIPES``/``NVFP4_WEIGHT_RECIPES``.  Only the *field
   vocabulary* is replaced (ModelOpt names instead of compressed-tensors names),
   and each part is re-checked structurally::

       part.source.shape == logical_shape(stored weight, declared format)

   where a declared NVFP4 group stores ``K/2`` columns and a declared FP8 group
   stores ``K``.  A part that fails that identity is refused.
3. **The numeric format of an artifact object follows the source's declaration**,
   never the historical object plan.  The registered Qwen3.8 NVFP4 inventory pins
   ``mlp/gate_up`` and ``mlp/down`` to NVFP4 for layers 0..55 and to FP8 for
   56..63; this source declares FP8 for the whole MLP of layers 0-3, 50, 52, 54
   and an FP8 ``down_proj`` at layers 21, 42, 44, 46, 48, 49, 53.  Bending the
   source to the historical split would mean re-quantizing stored FP8 words into
   NVFP4, so instead the plan follows the source and the *difference* is recorded
   as a deviation, together with the reason such an object cannot be loaded yet
   (the loader pins the split by layer index).
4. **Fields the artifact cannot carry are named, never silently dropped and never
   approximated.**  The source's static per-tensor activation scale of an FP8
   group has no slot in the artifact (``row-scale-v1`` carries weight row
   multipliers only, and the loader binds FP8 weights without an input divisor);
   that is a deviation with its consequence stated.  A fused object whose parts
   disagree about the format, or about the single trailing divisor a blockscale
   object stores, is refused outright.
5. **A payload that is not a move of stored words is charged per object.**  The
   ``official`` (vision/mtp/draft) and ``vocabulary`` routes hand plain source
   words to the artifact's own encoder (``encode_tensor_payload`` ->
   ``quantize_and_encode``) whenever the artifact object's format is not a direct
   one, and the source declares those modules outside its ``quantized_layers``.
   That re-encoding is a lossy conversion of the source's own values, so every
   affected object is recorded with ``D-RECODED-FROM-BF16`` - the same code the
   fused plain-source route already uses - instead of being quantised silently.

USAGE
-----
::

    # plan only: histograms, deviations, refusals, byte-coverage gap
    python3 -m tools.convert.qwen3_8_27b.convert_modelopt \
        --model /path/to/Qwen3.8-27B-ET-Uncensored-NVFP4/W4A4+W8A8 --plan-only

    # assemble the artifact
    python3 -m tools.convert.qwen3_8_27b.convert_modelopt \
        --model /path/to/Qwen3.8-27B-ET-Uncensored-NVFP4/W4A4+W8A8 \
        --out out/qwen3_8_27b_nvfp4_modelopt.ninfer --device cuda

Reading a *partially transferred* checkpoint is deliberate: the object plan comes
from the safetensors **headers** and the byte range each tensor claims, so the
plan - and the exact list of tensors whose bytes are not on disk yet - exists
before the transfer finishes, while ``convert()`` refuses to write an artifact
until every required tensor is present.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

_REPO_MARKER = Path("tools") / "artifact" / "layouts.py"


def _repo_root(origin: Path) -> Path:
    """The nearest ancestor that owns this repository.

    Resolved by a marker file rather than by counting parents, so a plan-only run
    from a scratch copy of this module still finds the pinned fixtures instead of
    silently reading a wrong path.
    """

    for parent in origin.parents:
        if (parent / _REPO_MARKER).is_file():
            return parent
    raise RuntimeError(
        f"{origin}: no ancestor carries {_REPO_MARKER}; run this module from inside "
        "the ninfer-fusion tree"
    )


REPO_ROOT = _repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.artifact.container import (  # noqa: E402
    ArtifactIdentity,
    ArtifactWriter,
)
from tools.artifact.layouts import (  # noqa: E402
    encode_direct,
    encode_fp8_row_scaled,
    encode_nvfp4,
    encoded_size,
    row_scale_geometry,
)
from tools.convert.common import layout_plan  # noqa: E402
from tools.convert.common.quantize import pick_device  # noqa: E402
from tools.convert.common.safetensors import TensorMetadata  # noqa: E402
from tools.convert.qwen3_6.common import conversion as family_conversion  # noqa: E402
from tools.convert.qwen3_6.common import frontend_policy  # noqa: E402
from tools.convert.qwen3_6.common import recipe as family_recipe  # noqa: E402
from tools.convert.qwen3_6_27b import convert as family_config  # noqa: E402
from tools.convert.qwen3_6_27b import draft_head  # noqa: E402
from tools.convert.qwen3_8_27b import fp8_embedding  # noqa: E402
from tools.convert.qwen3_8_27b import inventory_nvfp4 as inventory  # noqa: E402
from tools.convert.qwen3_8_27b import recipe_nvfp4 as registered_recipe  # noqa: E402


RECIPE_ID = "qwen3_8_27b_modelopt-single-source-v1"

#: This target's object plan is not the registered one: the per-layer numeric
#: format of the MLP objects follows the source's ``quant_algo``, which moves 21
#: objects (and 21 paired input divisor objects) between layouts.  The weights id
#: therefore has to differ, and the front door / loader have to learn it before
#: such an artifact can be run.
WEIGHTS_ID = "nvfp4-modelopt"
OUTPUT_BASENAME = "qwen3_8_27b_nvfp4_modelopt.ninfer"

INDEX_NAME = "model.safetensors.index.json"
SIDECAR_NAME = "hf_quant_config.json"

#: ModelOpt field vocabulary: the only naming this module knows.  Which layout a
#: field combination selects is decided by ``layout_plan``, not here.
FIELD_CODES = "weight"
FIELD_SCALE = "weight_scale"
FIELD_SCALE_2 = "weight_scale_2"
FIELD_INPUT_SCALE = "input_scale"
AUX_FIELDS = (FIELD_SCALE, FIELD_SCALE_2, FIELD_INPUT_SCALE)

ALGO_NVFP4 = "NVFP4"
ALGO_FP8 = "FP8"

#: The artifact's vocabulary endpoints.  ``text/token_embedding`` has no registered
#: matrix recipe (the registered dual-source converter streams it from BF16 too),
#: so it is routed here; ``text/output_head`` *is* a registered FP8 matrix recipe
#: whose source group is plain BF16 in this checkpoint, so the generic
#: "plain source, quantized object" rule in :func:`_requantised_entry` takes it and
#: records the re-coding as a deviation.
VOCABULARY_SOURCES = {
    "text/token_embedding": "model.language_model.embed_tokens.weight",
}

_LAYOUT_OF_ALGO = {
    ALGO_NVFP4: layout_plan.LAYOUT_BLOCKSCALE,
    ALGO_FP8: layout_plan.LAYOUT_ROW_SCALE,
}
_FORMAT_OF_LAYOUT = {
    layout_plan.LAYOUT_BLOCKSCALE: inventory.NVFP4,
    layout_plan.LAYOUT_ROW_SCALE: inventory.FP8,
}
_LAYOUT_OF_FORMAT = {
    inventory.NVFP4: layout_plan.LAYOUT_BLOCKSCALE,
    inventory.FP8: layout_plan.LAYOUT_ROW_SCALE,
}
_TORCH_DTYPE = {
    "U8": torch.uint8,
    "F8_E4M3": torch.uint8,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F16": torch.float16,
}


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Deviation:
    code: str
    scope: str
    detail: str
    consequence: str


@dataclass(frozen=True, slots=True)
class Refusal:
    code: str
    subject: str
    missing_mechanism: str
    detail: str = ""

    def render(self) -> str:
        return (
            f"[{self.code}] {self.subject}\n"
            f"      detail  : {self.detail}\n"
            f"      missing : {self.missing_mechanism}"
        )


@dataclass(slots=True)
class Ledger:
    deviations: list[Deviation] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)

    def deviate(self, code: str, scope: str, detail: str, consequence: str) -> None:
        self.deviations.append(Deviation(code, scope, detail, consequence))

    def refuse(self, code: str, subject: str, missing: str, detail: str = "") -> None:
        self.refusals.append(Refusal(code, subject, missing, detail))

    def histogram(self, attribute: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in getattr(self, attribute):
            counts[item.code] = counts.get(item.code, 0) + 1
        return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# byte-addressed source reader
# --------------------------------------------------------------------------- #
class HeaderIndex:
    """Address one safetensors shard through its header and ``data_offsets``.

    ``safetensors.safe_open`` refuses a file whose length does not exactly cover
    the header's payload span ("incomplete metadata, file not fully covered"),
    which is the normal state of a checkpoint still being transferred.  The object
    plan only needs the header, and a payload read only needs
    ``8 + header_len + data_offsets[0]``; doing it explicitly also turns the
    byte-coverage gap into a reportable number instead of a fatal error.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as handle:
            self.header_len = struct.unpack("<Q", handle.read(8))[0]
            raw = handle.read(self.header_len)
        if len(raw) != self.header_len:
            raise ValueError(f"{self.path}: header is not fully present")
        self.header: dict[str, Any] = json.loads(raw)
        self.payload_base = 8 + self.header_len
        self.payload_covered = max(0, self.path.stat().st_size - self.payload_base)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name in self.header if name != "__metadata__")

    def entry(self, name: str) -> Mapping[str, Any]:
        try:
            return self.header[name]
        except KeyError:
            raise KeyError(f"{name!r} is not a key of {self.path.name}") from None

    def meta(self, name: str) -> layout_plan.TensorMeta:
        entry = self.entry(name)
        return layout_plan.TensorMeta(str(entry["dtype"]), tuple(entry["shape"]))

    def span(self, name: str) -> tuple[int, int]:
        begin, end = self.entry(name)["data_offsets"]
        return int(begin), int(end)

    def covered(self, name: str) -> bool:
        return self.span(name)[1] <= self.payload_covered

    def shortfall(self, name: str) -> int:
        return max(0, self.span(name)[1] - self.payload_covered)

    def raw(self, name: str) -> bytes:
        begin, end = self.span(name)
        if end > self.payload_covered:
            raise EOFError(
                f"{name}: data_offsets [{begin},{end}) reach past the "
                f"{self.payload_covered} payload bytes present in {self.path.name} "
                f"({end - self.payload_covered} bytes short)"
            )
        with self.path.open("rb") as handle:
            handle.seek(self.payload_base + begin)
            return handle.read(end - begin)

    def tensor(self, name: str) -> torch.Tensor:
        meta = self.meta(name)
        dtype = _TORCH_DTYPE.get(meta.dtype)
        if dtype is None:
            raise ValueError(f"{name}: unsupported stored dtype {meta.dtype!r}")
        buffer = bytearray(self.raw(name))
        return torch.frombuffer(buffer, dtype=dtype).reshape(meta.shape)

    def scalar(self, name: str) -> float:
        meta = self.meta(name)
        if meta.dtype != "F32" or meta.shape not in ((), (1,)):
            raise TypeError(
                f"{name}: expected an FP32 scalar, got {meta.dtype}{list(meta.shape)}"
            )
        return struct.unpack("<f", self.raw(name))[0]

    def rows(self, name: str, begin: int, count: int) -> bytes:
        meta = self.meta(name)
        if len(meta.shape) != 2 or begin < 0 or meta.shape[0] < begin + count:
            raise ValueError(f"{name}: row range {begin}+{count} is outside {meta.shape}")
        first, last = self.span(name)
        row_bytes = (last - first) // meta.shape[0]
        with self.path.open("rb") as handle:
            handle.seek(self.payload_base + first + begin * row_bytes)
            return handle.read(row_bytes * count)


class ModelOptDirectory:
    """The whole source: an index, its shards, and the two declarations."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / INDEX_NAME
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path}: weight_map must be a nonempty object")
        self.weight_map: dict[str, str] = dict(weight_map)
        self.shards: dict[str, HeaderIndex] = {
            shard: HeaderIndex(self._resolve_shard(shard))
            for shard in sorted(set(self.weight_map.values()))
        }
        self._members: dict[str, dict[str, layout_plan.TensorMeta]] | None = None

    def _resolve_shard(self, shard: str) -> Path:
        """Resolve an indexed shard name.

        A published index names shards relative to the variant directory it sits
        in, and a mirror may keep that directory as a prefix
        (``W4A4+W8A8/model-nvfp4-mixed.safetensors``); both layouts are accepted.
        """

        direct = self.model_dir / shard
        if direct.is_file():
            return direct
        nested = self.model_dir.parent / shard
        if nested.is_file():
            return nested
        raise ValueError(f"{self.model_dir}: indexed shard {shard!r} is missing")

    # -------------------------------------------------------------- structure
    def has(self, name: str) -> bool:
        return name in self.weight_map

    def _shard_of(self, name: str) -> HeaderIndex:
        try:
            return self.shards[self.weight_map[name]]
        except KeyError:
            raise KeyError(f"{name!r} is not a key of this source") from None

    def meta(self, name: str) -> layout_plan.TensorMeta:
        return self._shard_of(name).meta(name)

    def tensor(self, name: str) -> torch.Tensor:
        return self._shard_of(name).tensor(name)

    def scalar(self, name: str) -> float:
        return self._shard_of(name).scalar(name)

    def rows(self, name: str, begin: int, count: int) -> bytes:
        return self._shard_of(name).rows(name, begin, count)

    def modules(self) -> dict[str, dict[str, layout_plan.TensorMeta]]:
        if self._members is None:
            self._members = layout_plan.group_by_module(
                {name: self.meta(name) for name in self.weight_map}
            )
        return self._members

    def module_members(self, module: str) -> dict[str, layout_plan.TensorMeta]:
        try:
            return self.modules()[module]
        except KeyError:
            raise KeyError(f"{module!r} is not a module of this source") from None

    def truncated(self, names: Iterable[str]) -> list[tuple[str, int]]:
        short: list[tuple[str, int]] = []
        for name in names:
            shard = self._shard_of(name)
            if not shard.covered(name):
                short.append((name, shard.shortfall(name)))
        return short

    def transfer_state(self) -> dict[str, dict[str, int]]:
        """Per shard: bytes claimed, bytes present, bytes still missing."""

        state: dict[str, dict[str, int]] = {}
        for shard, handle in sorted(self.shards.items()):
            claimed = max(
                (handle.span(name)[1] for name in handle.names), default=0
            )
            state[shard] = {
                "claimed_payload_bytes": claimed,
                "present_payload_bytes": handle.payload_covered,
                "missing_payload_bytes": max(0, claimed - handle.payload_covered),
            }
        return state

    def byte_coverage(self, names: Iterable[str]) -> dict[str, Any]:
        """How much of the local copy the plan can actually address."""

        absent = partial = present = 0
        broken: list[tuple[str, int]] = []
        for name in names:
            shard = self._shard_of(name)
            begin, end = shard.span(name)
            if end <= shard.payload_covered:
                present += 1
                continue
            if begin >= shard.payload_covered:
                absent += 1
            else:
                partial += 1
            broken.append((name, shard.shortfall(name)))
        shards = self.transfer_state()
        return {
            "required_tensors": present + absent + partial,
            "fully_present": present,
            "entirely_absent": absent,
            "partially_present": partial,
            "missing_payload_bytes": sum(
                item["missing_payload_bytes"] for item in shards.values()
            ),
            "shards": shards,
            "first_broken": [
                {"name": name, "ends_beyond_present_prefix_by": short}
                for name, short in broken[:8]
            ],
        }

    # ------------------------------------------------------------ declaration
    def declared_algos(self) -> dict[str, str]:
        declared = layout_plan.load_declared_algos(self.model_dir)
        from_config = self._declared_algos_from_config()
        if from_config and declared and from_config != declared:
            difference = sorted(set(from_config.items()) ^ set(declared.items()))
            raise ValueError(
                "the two declarations of this source disagree about quant_algo "
                f"(config.json vs {SIDECAR_NAME}); first difference {difference[:1]}"
            )
        if not declared:
            declared = from_config
        if not declared:
            raise ValueError(
                f"{self.model_dir}: neither {SIDECAR_NAME} nor "
                "config.json.quantization_config declares any quantized layer"
            )
        return declared

    def _declared_algos_from_config(self) -> dict[str, str]:
        path = self.model_dir / "config.json"
        if not path.is_file():
            return {}
        quant = json.loads(path.read_text()).get("quantization_config")
        if not isinstance(quant, Mapping):
            return {}
        layers = quant.get("quantized_layers")
        if not isinstance(layers, Mapping):
            return {}
        return {
            str(key): str(value.get("quant_algo", "")).upper()
            for key, value in layers.items()
            if isinstance(value, Mapping)
        }

    def declared_algo(self, module: str, declared: Mapping[str, str]) -> str | None:
        for key, value in declared.items():
            if module == key or module.endswith(key) or key.endswith(module):
                return value
        return None


# --------------------------------------------------------------------------- #
# numeric conventions of the source
# --------------------------------------------------------------------------- #
def logical_shape(meta: layout_plan.TensorMeta, algo: str) -> tuple[int, int]:
    """The logical ``(N, K)`` of a quantized ``Linear`` from its stored shape.

    A declared NVFP4 group stores two E2M1 nibbles per byte, so its logical K is
    exactly twice the stored column count; a declared FP8 group stores one E4M3
    word per element.  Both are shape identities, not name lookups.
    """

    if len(meta.shape) != 2:
        raise ValueError(f"a quantized Linear must be rank 2, got {list(meta.shape)}")
    rows, columns = meta.shape
    if algo == ALGO_NVFP4:
        return rows, columns * 2
    if algo == ALGO_FP8:
        return rows, columns
    raise ValueError(f"unsupported declared quant_algo {algo!r}")


def reciprocal_divisor(value: float, label: str) -> float:
    """``fp32(1 / value)`` for a positive finite FP32 multiplier.

    The artifact stores *divisors* (the engine divides by them); ModelOpt stores
    *multipliers*, so each is inverted here in binary32 - the operation
    ``tools/convert/dequant/modelopt.py`` performs - and the result must satisfy
    the layout's own finite/positive check.
    """

    word = np.float32(value)
    if not np.isfinite(word) or word <= 0:
        raise ValueError(f"{label}: multiplier must be finite and positive, got {value!r}")
    divisor = np.float32(1.0) / word
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError(f"{label}: reciprocal {divisor!r} is not finite and positive")
    return float(divisor)


def fp32_word(value: float) -> bytes:
    return struct.pack("<f", np.float32(value))


# --------------------------------------------------------------------------- #
# source-side plan: layout_plan is the authority
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SourceGroupPlan:
    module: str
    fields: tuple[str, ...]
    choice: layout_plan.Choice | None
    refusal: layout_plan.Refusal | None
    declared_algo: str | None

    @property
    def layout(self) -> str:
        if self.choice is None:
            raise ValueError(f"{self.module}: no layout was selected")
        return self.choice.layout


def plan_source(
    source: ModelOptDirectory, ledger: Ledger
) -> tuple[dict[str, SourceGroupPlan], dict[str, int], dict[str, str]]:
    """Run ``layout_plan.select`` over every module and cross-check the declaration."""

    declared = source.declared_algos()
    groups: dict[str, SourceGroupPlan] = {}
    histogram: dict[str, int] = {}
    for module, members in sorted(source.modules().items()):
        algo = source.declared_algo(module, declared)
        decision = layout_plan.select(members, algo)
        if decision.ok:
            groups[module] = SourceGroupPlan(
                module, tuple(sorted(members)), decision, None, algo
            )
            histogram[decision.layout] = histogram.get(decision.layout, 0) + 1
        else:
            groups[module] = SourceGroupPlan(
                module, tuple(sorted(members)), None, decision, algo
            )
            ledger.refuse(decision.code, module, decision.missing_mechanism, decision.reason)
    quantized = {m: g for m, g in groups.items() if g.declared_algo is not None}
    for module, group in sorted(quantized.items()):
        expected = _LAYOUT_OF_ALGO.get(str(group.declared_algo))
        if expected is None:
            ledger.refuse(
                "F-UNKNOWN-ALGO",
                module,
                "a mapping from this quant_algo to a registered layout",
                f"declared quant_algo={group.declared_algo!r}",
            )
            continue
        if group.choice is not None and group.layout != expected:
            ledger.refuse(
                "F-DECLARATION-MISMATCH",
                module,
                "a layout that satisfies both the declared quant_algo and the stored "
                "dtype/shape structure",
                f"declared {group.declared_algo} but the structure selects "
                f"{group.layout}: {group.choice.reason}",
            )
    histogram["__declared_algo__"] = len(quantized)
    return groups, histogram, declared


def unquantized_signatures(source: ModelOptDirectory) -> dict[str, tuple[str, ...]]:
    """Dtype/shape signature of every module that carries no quantisation."""

    kinds: dict[str, list[str]] = {}
    for module, members in source.modules().items():
        if any(fieldname in AUX_FIELDS for fieldname in members):
            continue
        signature = ",".join(
            f"{members[fieldname].dtype}{list(members[fieldname].shape)}"
            for fieldname in sorted(members)
        )
        kinds.setdefault(signature, []).append(module)
    return {key: tuple(value) for key, value in sorted(kinds.items())}


# --------------------------------------------------------------------------- #
# artifact-side plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MatrixPartPlan:
    """One source matrix feeding an artifact object, with its row ranges."""

    module: str
    rows: tuple[tuple[int, int], ...]
    format: str
    shape: tuple[int, int]

    @property
    def output_rows(self) -> int:
        return sum(end - begin for begin, end in self.rows)


@dataclass(frozen=True, slots=True)
class ObjectPlanEntry:
    object_name: str
    spec: inventory.TensorSpec
    route: str
    parts: tuple[MatrixPartPlan, ...] = ()
    divisor_sources: tuple[str, ...] = ()
    source_tensor: str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class ArtifactPlan:
    specs: tuple[inventory.StoredObjectSpec, ...]
    entries: tuple[ObjectPlanEntry, ...]
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    layout_histogram: dict[str, int]
    format_histogram: dict[str, int]
    route_histogram: dict[str, int]
    required_source_keys: tuple[str, ...]
    truncated: tuple[tuple[str, int], ...]
    coverage: dict[str, Any]
    narrowing: tuple[tuple[str, float, float], ...]


def _registered_parts(parts: Sequence[Any]) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    return tuple(
        (part.source.name, tuple((item.begin, item.end) for item in part.rows))
        for part in parts
    )


def _matrix_entry(
    name: str,
    shape: tuple[int, int],
    registered_parts: Sequence[Any],
    registered_format: str,
    source: ModelOptDirectory,
    groups: Mapping[str, SourceGroupPlan],
    ledger: Ledger,
) -> ObjectPlanEntry | None:
    """Resolve one matrix object; its numeric format follows the source."""

    resolved: list[MatrixPartPlan] = []
    plain = 0
    for part, (module, rows) in zip(registered_parts, _registered_parts(registered_parts)):
        expected = tuple(part.source.shape)
        group = groups.get(module)
        if group is None or group.choice is None:
            reason = (
                group.refusal.reason if group is not None and group.refusal else "no group"
            )
            ledger.refuse(
                "F-NO-SOURCE-GROUP",
                name,
                "a source group for this fused part",
                f"part module {module!r}: {reason}",
            )
            return None
        members = source.module_members(module)
        codes = members.get(FIELD_CODES)
        if codes is None:
            ledger.refuse(
                "F-NO-CODES",
                name,
                f"a {FIELD_CODES!r} tensor in {module!r}",
                f"fields present: {sorted(members)}",
            )
            return None
        if group.layout == layout_plan.LAYOUT_CONTIGUOUS:
            plain += 1
            if tuple(codes.shape) != expected:
                ledger.refuse(
                    "F-SHAPE",
                    name,
                    "a plain source matrix whose shape equals the registered part",
                    f"{module}: {codes.dtype}{list(codes.shape)} vs {expected}",
                )
                return None
            resolved.append(MatrixPartPlan(module, rows, inventory.BF16, expected))
            continue
        try:
            logical = logical_shape(codes, str(group.declared_algo))
        except ValueError as exc:
            ledger.refuse(
                "F-UNSHAPED",
                name,
                "a stored shape a declared format can be mapped from",
                f"{module}: {exc}",
            )
            return None
        if logical != expected:
            ledger.refuse(
                "F-SHAPE",
                name,
                "a source matrix whose logical shape equals the registered part",
                f"{module}: stored {codes.dtype}{list(codes.shape)} -> logical {logical}, "
                f"registered part is {expected}",
            )
            return None
        if group.layout not in _FORMAT_OF_LAYOUT:
            ledger.refuse(
                "F-UNQUANTISED-PART",
                name,
                "a quantized layout for this source group",
                f"{module}: {group.choice.reason}",
            )
            return None
        resolved.append(
            MatrixPartPlan(module, rows, _FORMAT_OF_LAYOUT[group.layout], expected)
        )

    # the registered row geometry must still reconstruct the object exactly
    rows_total = sum(part.output_rows for part in resolved)
    if rows_total != shape[0] or any(
        part.shape[1] != shape[1] or part.output_rows > part.shape[0] for part in resolved
    ):
        ledger.refuse(
            "F-FUSION-GEOMETRY",
            name,
            "a row decomposition whose parts reconstruct the artifact object",
            f"parts {[(p.module, p.output_rows, p.shape) for p in resolved]} do not "
            f"build {tuple(shape)}",
        )
        return None

    if plain == len(resolved) and resolved:
        # the source keeps this object's words plain (BF16): the artifact object
        # itself is quantized, so its payload has to be produced by a registered
        # encoder instead of moving stored codes.
        return _requantised_entry(
            name, shape, resolved, registered_format, source, ledger
        )
    if plain:
        ledger.refuse(
            "F-FUSION-PLAIN-MIXED",
            name,
            "a fused object whose parts are partly quantized and partly plain",
            "parts: " + ", ".join(f"{p.module}->{p.format}" for p in resolved),
        )
        return None

    formats = {part.format for part in resolved}
    if len(formats) != 1:
        ledger.refuse(
            "F-FUSION-FORMAT",
            name,
            "an object layout that holds a blockscale sub-matrix and a row-scale "
            "sub-matrix side by side (no registered layout does: "
            "blockscale-k16-m128x4-v1 and row-scale-v1 are whole-object layouts)",
            "fused parts declare "
            + ", ".join(f"{part.module}->{part.format}" for part in resolved),
        )
        return None
    numeric_format = formats.pop()
    if numeric_format == inventory.NVFP4:
        total_rows = sum(part.output_rows for part in resolved)
        if total_rows != shape[0] or shape[0] % 128 or shape[1] % 64:
            ledger.refuse(
                "F-LAYOUT-CONSTRAINT",
                name,
                "a padded or sub-tiled blockscale layout "
                "(blockscale-k16-m128x4-v1 requires N%128==0 and K%64==0)",
                f"resolved rows {total_rows}, shape {tuple(shape)}",
            )
            return None
    spec = inventory.tensor_spec(name, tuple(shape), numeric_format)
    if spec.format != registered_format:
        ledger.deviate(
            "D-OBJECT-FORMAT",
            name,
            f"the source declares {numeric_format} for every part of this object "
            f"({', '.join(part.module for part in resolved)}); the registered artifact "
            f"pins {registered_format}",
            "the object plan of this artifact is the source's declaration; the loader "
            "pins the registered split (src/targets/qwen3_6_27b/impl/load/bindings.cpp "
            "bind_qwen38_nvfp4_text_layers selects NVFP4 for 'layer < 56'), so this "
            "object cannot be loaded until the loader reads each object's own format",
        )
    return ObjectPlanEntry(name, spec, "matrix", tuple(resolved))


def _requantised_entry(
    name: str,
    shape: tuple[int, int],
    parts: Sequence[MatrixPartPlan],
    registered_format: str,
    source: ModelOptDirectory,
    ledger: Ledger,
) -> ObjectPlanEntry | None:
    """An artifact-quantized object whose source words are plain BF16."""

    if registered_format not in _LAYOUT_OF_FORMAT:
        ledger.refuse(
            "F-REQUANTISE-MECHANISM",
            name,
            f"a registered encoder that produces {registered_format} from plain "
            "words (tools/convert/common/quantize.py covers the grouped and "
            "row-scaled FP8 formats only)",
            f"source parts are plain BF16: {[part.module for part in parts]}",
        )
        return None
    if registered_format != inventory.FP8 or len(parts) != 1:
        ledger.refuse(
            "F-REQUANTISE-MECHANISM",
            name,
            "a streaming quantizer for this (source, artifact format) pair",
            f"source parts {[part.module for part in parts]} are plain; artifact "
            f"format is {registered_format} with {len(parts)} part(s)",
        )
        return None
    source_tensor = f"{parts[0].module}.{FIELD_CODES}"
    ledger.deviate(
        "D-RECODED-FROM-BF16",
        name,
        f"every source part of this object is plain BF16 ({source_tensor}); the "
        "registered dual-source recipe moves stored codes for it",
        "the object is quantized at import time by the registered BF16 -> FP8 "
        "row-scale profile (fp8_embedding.quantize_bf16_rows), so its words are not "
        "the source's words: lossy by the profile, not by fusion",
    )
    return ObjectPlanEntry(
        name,
        inventory.tensor_spec(name, tuple(shape), registered_format),
        "requantised_stream",
        tuple(parts),
        source_tensor=source_tensor,
        note="streamed BF16 rows through the registered row-scale FP8 profile",
    )


def _account_plain_word_quantisation(
    name: str,
    spec: inventory.TensorSpec,
    source: ModelOptDirectory,
    ledger: Ledger,
    source_names: Sequence[str],
    route: str,
) -> None:
    """Charge one object whose plain source words the encoder re-quantises.

    The ``official`` and ``vocabulary`` routes do not move stored words: their
    payload goes through ``family_conversion.encode_tensor_payload``, which sends
    every non-direct artifact format to ``quantize_and_encode``.  When the source
    tensors behind such an object are all plain - the source declares no
    quantisation for that module - that call is a real numeric conversion of the
    source's own values, so it is recorded per object exactly as the fused
    ``requantised_stream`` route records the same situation.
    """

    if spec.format in family_conversion.DIRECT_FORMATS:
        return  # a byte view of the source words: no numeric conversion
    plain: list[str] = []
    for key in source_names:
        if not source.has(key):
            continue
        meta = source.meta(key)
        plain.append(f"{key} {meta.dtype}{list(meta.shape)}")
    if not plain:
        return
    more = f"; and {len(plain) - 3} more" if len(plain) > 3 else ""
    ledger.deviate(
        "D-RECODED-FROM-BF16",
        name,
        f"the artifact stores this object as {spec.format} in {spec.layout}, while "
        f"every source tensor it reads is plain ({'; '.join(plain[:3])}{more})",
        f"the {route} route materialises those plain words and "
        "encode_tensor_payload -> quantize_and_encode re-encodes them into the "
        "artifact's format, so the stored words are not the source's words: lossy by "
        "that quantisation profile and irreversible. The source declares no "
        "quantisation for this module (it is outside its quantized_layers), so the "
        "conversion is an artifact-side decision that needs an owner: either keep "
        "the object in a direct format, or accept this recorded loss",
    )


def _divisor_entry(
    name: str,
    parent: str,
    parts: Sequence[MatrixPartPlan],
    source: ModelOptDirectory,
    fieldname: str,
    ledger: Ledger,
) -> ObjectPlanEntry | None:
    """One FP32 divisor object: ``fp32(1 / input_scale)`` of its parent's parts."""

    values: list[tuple[str, float]] = []
    for part in parts:
        key = f"{part.module}.{fieldname}"
        if not source.has(key):
            ledger.refuse(
                "F-NO-DIVISOR-FIELD",
                name,
                f"a {fieldname!r} scalar on every part of its parent object",
                f"{key} is missing",
            )
            return None
        values.append((key, source.scalar(key)))
    words = {fp32_word(reciprocal_divisor(value, key)) for key, value in values}
    if len(words) != 1:
        ledger.refuse(
            "F-FUSION-DIVISOR",
            name,
            "a per-part divisor inside one blockscale object (the layout stores a "
            "single trailing fp32 weight divisor, and the loader pairs exactly one "
            "input divisor object with it)",
            "fused parts disagree: "
            + ", ".join(f"{key}={value!r}" for key, value in values),
        )
        return None
    return ObjectPlanEntry(
        name,
        inventory.tensor_spec(name, (), inventory.FP32),
        "divisor",
        tuple(parts),
        divisor_sources=tuple(key for key, _ in values),
        note=f"fp32(1/{fieldname}) of {parent}",
    )


def build_artifact_plan(
    source: ModelOptDirectory,
    groups: Mapping[str, SourceGroupPlan],
    ledger: Ledger,
) -> ArtifactPlan:
    """Turn the registered object inventory into this source's object plan."""

    resources = family_conversion.load_resources(source.model_dir, inventory.RESOURCE_SPECS)
    entries: dict[str, ObjectPlanEntry] = {}
    dropped: set[str] = set()
    refused: set[str] = set()

    for spec in inventory.OBJECT_SPECS:
        if not isinstance(spec, inventory.TensorSpec):
            continue
        name = spec.name
        if name in VOCABULARY_SOURCES:
            continue  # handled by the vocabulary route below
        if name in registered_recipe.FP8_WEIGHTS_BY_NAME or name in registered_recipe.NVFP4_WEIGHTS_BY_NAME:
            registered = registered_recipe.FP8_WEIGHTS_BY_NAME.get(
                name
            ) or registered_recipe.NVFP4_WEIGHTS_BY_NAME[name]
            kind = (
                inventory.NVFP4
                if name in registered_recipe.NVFP4_WEIGHTS_BY_NAME
                else inventory.FP8
            )
            entry = _matrix_entry(
                name, tuple(registered.shape), registered.parts, kind, source, groups, ledger
            )
            if entry is None:
                refused.add(name)
                continue
            entries[name] = entry
            if entry.spec.format != inventory.NVFP4:
                dropped.add(name)

    for spec in inventory.OBJECT_SPECS:
        if not isinstance(spec, inventory.TensorSpec):
            continue
        registered_divisor = registered_recipe.INPUT_DIVISORS_BY_NAME.get(spec.name)
        recipe = registered_divisor
        if recipe is None:
            continue
        parents = recipe.weight_names
        if any(parent in dropped for parent in parents):
            refused.add(spec.name)
            ledger.deviate(
                "D-DIVISOR-DROPPED",
                spec.name,
                "the parent object is FP8 in this source, so the paired FP32 input "
                "divisor object has no consumer",
                "the artifact's FP8 path has no input-divisor slot (bindings.cpp binds "
                "FP8 weights through bind_weight, without a divisor), so the parent's "
                "static input_scale is dropped as well - see D-ACTIVATION-SCALE",
            )
            continue
        parts: list[MatrixPartPlan] = []
        complete = True
        for parent in parents:
            entry = entries.get(parent)
            if entry is None:
                complete = False
                break
            parts.extend(entry.parts)
        if not complete:
            continue
        divisor = _divisor_entry(
            spec.name, parents[0], parts, source, FIELD_INPUT_SCALE, ledger
        )
        if divisor is not None:
            entries[spec.name] = divisor

    for spec in inventory.OBJECT_SPECS:
        if not isinstance(spec, inventory.TensorSpec) or spec.name in entries:
            continue
        if spec.name in refused:
            continue  # already refused once; a second refusal is noise, not evidence
        name = spec.name
        if name in VOCABULARY_SOURCES:
            source_tensor = VOCABULARY_SOURCES[name]
            if spec.layout != layout_plan.LAYOUT_ROW_SCALE:
                ledger.refuse(
                    "F-VOCABULARY-FORMAT",
                    name,
                    "a vocabulary-endpoint encoder for this artifact format",
                    f"artifact format {spec.format} on a plain BF16 source",
                )
                continue
            entries[name] = ObjectPlanEntry(
                name,
                spec,
                "vocabulary",
                source_tensor=source_tensor,
                note="BF16 source rows through the registered row-scale FP8 profile",
            )
            _account_plain_word_quantisation(
                name, spec, source, ledger, (source_tensor,), "vocabulary"
            )
        elif name in registered_recipe.QUANTIZED_DIRECT_BY_NAME:
            entries[name] = ObjectPlanEntry(
                name, spec, "direct", note="verbatim direct tensor"
            )
        elif name in registered_recipe.OFFICIAL_RECIPES_BY_NAME:
            entries[name] = ObjectPlanEntry(
                name,
                spec,
                "official",
                note="vision/mtp/draft route; the artifact's own registered quantizer applies",
            )
            _account_plain_word_quantisation(
                name,
                spec,
                source,
                ledger,
                tuple(
                    requirement.name
                    for requirement in family_recipe.expression_sources(
                        registered_recipe.OFFICIAL_RECIPES_BY_NAME[name].expression
                    )
                ),
                "official",
            )
        else:
            ledger.refuse(
                "F-NO-ROUTE",
                name,
                "a route from this source to this artifact object",
                "the registered recipe names no source for it either",
            )

    specs: list[inventory.StoredObjectSpec] = []
    ordered: list[ObjectPlanEntry] = []
    for spec in inventory.OBJECT_SPECS:
        if isinstance(spec, inventory.ResourceSpec):
            specs.append(spec)
            continue
        entry = entries.get(spec.name)
        if entry is None:
            continue
        specs.append(entry.spec)
        ordered.append(entry)

    layout_histogram: dict[str, int] = {}
    format_histogram: dict[str, int] = {}
    route_histogram: dict[str, int] = {}
    for entry in ordered:
        layout_histogram[entry.spec.layout] = layout_histogram.get(entry.spec.layout, 0) + 1
        format_histogram[entry.spec.format] = format_histogram.get(entry.spec.format, 0) + 1
        route_histogram[entry.route] = route_histogram.get(entry.route, 0) + 1
    layout_histogram[inventory.RESOURCE_SPECS[0].encoding] = len(inventory.RESOURCE_SPECS)

    consumed = {part.module for entry in ordered for part in entry.parts}
    for module, group in sorted(groups.items()):
        if group.choice is None or group.declared_algo is None or module in consumed:
            continue
        ledger.refuse(
            "F-UNCONSUMED-GROUP",
            module,
            "an artifact object that consumes this quantized source group",
            f"declared {group.declared_algo}, structure selects {group.layout}, and no "
            "artifact object routes to it",
        )

    ranking = REPO_ROOT / draft_head.DEFAULT_RANKING
    if not ranking.is_file():
        ledger.refuse(
            "F-NO-RANKING",
            "text/draft_head_token_ids",
            f"the frequency ranking fixture at {ranking}",
        )
    draft = draft_head.compute_shortlist(ranking, source.model_dir)

    narrowing = _narrowing_report(ordered, source, ledger)
    required = _required_source_keys(ordered)
    return ArtifactPlan(
        specs=tuple(specs),
        entries=tuple(ordered),
        resources=resources,
        draft=draft,
        layout_histogram=layout_histogram,
        format_histogram=format_histogram,
        route_histogram=route_histogram,
        required_source_keys=required,
        truncated=tuple(source.truncated(required)),
        coverage=source.byte_coverage(required),
        narrowing=narrowing,
    )


def _narrowing_report(
    entries: Sequence[ObjectPlanEntry],
    source: ModelOptDirectory,
    ledger: Ledger,
) -> tuple[tuple[str, float, float], ...]:
    """Account for every fp32 -> bf16 row-scale narrowing the layout forces.

    ``row-scale-v1`` stores one *binary16* multiplier per row while the source
    stores one *binary32* multiplier per tensor (``weight_scale`` F32 ``[]``), so
    the scalar is narrowed once and broadcast.  The narrowing is recorded per
    affected object with its exact values, and the source's static activation
    multiplier (which the artifact has no slot for) is recorded once per group.
    """

    rows: list[tuple[str, float, float]] = []
    for entry in entries:
        if entry.route != "matrix" or entry.spec.format != inventory.FP8:
            continue
        for part in entry.parts:
            key = f"{part.module}.{FIELD_SCALE}"
            scalar = source.scalar(key)
            narrowed = float(
                torch.tensor(np.float32(scalar), dtype=torch.bfloat16).to(torch.float32)
            )
            rows.append((key, float(np.float32(scalar)), narrowed))
    distinct = len({key for key, _, _ in rows})
    if rows:
        inexact = [row for row in rows if row[1] != row[2]]
        worst = max(
            (abs(after - before) / abs(before) for _, before, after in inexact), default=0.0
        )
        ledger.deviate(
            "D-ROW-SCALE-BROADCAST",
            f"{len(rows)} matrix parts / {distinct} distinct FP8 source Linears",
            "the source stores one FP32 per-tensor multiplier "
            f"({FIELD_SCALE} F32 []) per FP8 Linear; row-scale-v1 stores one BF16 "
            f"multiplier per row, so each scalar is broadcast to that part's rows and "
            f"narrowed fp32 -> bf16 ({len(inexact)}/{len(rows)} parts inexact, worst "
            f"relative error {worst:.3e} <= the bf16 bound 2**-8 = 3.906e-03); the "
            "E4M3 codes themselves move verbatim",
            "every affected row's weight is multiplied by (1 + delta), |delta| <= "
            "2**-8: a bounded, quantified rescaling of the stored word values, and the "
            "only numerical change made to an FP8 object",
        )
    if rows:
        ledger.deviate(
            "D-ACTIVATION-SCALE",
            f"{distinct} distinct FP8 source Linears",
            f"every FP8 Linear of the source also carries a static {FIELD_INPUT_SCALE} "
            "(F32 scalar; the source's own config declares "
            "input_activations.dynamic=false)",
            "no artifact slot consumes it: row-scale-v1 carries weight row "
            "multipliers only, and the loader binds FP8 weights without an input "
            "divisor (bind_weight in src/targets/qwen3_6_27b/impl/load/bindings.cpp), "
            "so the engine quantizes FP8 activations dynamically per token instead "
            "of statically as the source declared. This is a numerics change, not an "
            "approximation of a stored value; it needs an artifact slot (or an "
            "explicit acceptance) before the source's static activation scales are "
            "reproduced",
        )
    return tuple(rows)


def _required_source_keys(entries: Sequence[ObjectPlanEntry]) -> tuple[str, ...]:
    keys: list[str] = []
    for entry in entries:
        if entry.route == "matrix":
            for part in entry.parts:
                keys.append(f"{part.module}.{FIELD_CODES}")
                keys.append(f"{part.module}.{FIELD_SCALE}")
                keys.append(f"{part.module}.{FIELD_INPUT_SCALE}")
                if part.format == inventory.NVFP4:
                    keys.append(f"{part.module}.{FIELD_SCALE_2}")
        elif entry.route == "divisor":
            keys.extend(entry.divisor_sources)
        elif entry.source_tensor is not None:
            keys.append(entry.source_tensor)
    for recipe in registered_recipe.QUANTIZED_DIRECT_RECIPES:
        keys.extend(
            requirement.name
            for requirement in family_recipe.expression_sources(recipe.expression)
        )
    for recipe in registered_recipe.OFFICIAL_RECIPES:
        for requirement in family_recipe.expression_sources(recipe.expression):
            keys.append(requirement.name)
    return tuple(dict.fromkeys(keys))


# --------------------------------------------------------------------------- #
# payload assembly
# --------------------------------------------------------------------------- #
def _bf16_row_scales(scalar: float, rows: int, label: str) -> torch.Tensor:
    """Broadcast one FP32 scalar into the layout's per-row BF16 multipliers."""

    value = np.float32(scalar)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label}: scale must be finite and positive, got {scalar!r}")
    return (
        torch.tensor(float(value), dtype=torch.bfloat16)
        .reshape(1)
        .expand(rows)
        .contiguous()
    )


def _select_rows(tensor: torch.Tensor, rows: Sequence[tuple[int, int]]) -> torch.Tensor:
    pieces = [tensor[begin:end] for begin, end in rows]
    if len(pieces) == 1 and pieces[0].shape[0] == tensor.shape[0]:
        return tensor
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def _assemble_matrix(entry: ObjectPlanEntry, source: ModelOptDirectory) -> bytes:
    """Move one artifact object's stored words into its layout."""

    shape = entry.spec.shape
    if entry.spec.format == inventory.NVFP4:
        code_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        for part in entry.parts:
            codes = source.tensor(f"{part.module}.{FIELD_CODES}")
            scales = source.tensor(f"{part.module}.{FIELD_SCALE}")
            code_parts.append(_select_rows(codes, part.rows).contiguous())
            scale_parts.append(_select_rows(scales, part.rows).contiguous())
        packed = (
            code_parts[0] if len(code_parts) == 1 else torch.cat(code_parts, dim=0)
        )
        natural = (
            scale_parts[0] if len(scale_parts) == 1 else torch.cat(scale_parts, dim=0)
        )
        divisor = reciprocal_divisor(
            source.scalar(f"{entry.parts[0].module}.{FIELD_SCALE_2}"),
            f"{entry.parts[0].module}.{FIELD_SCALE_2}",
        )
        # the layout takes the trailing divisor as its four FP32 bytes (the same
        # form ``recipe_nvfp4.materialize_nvfp4_weight`` hands it)
        return encode_nvfp4(packed, natural, fp32_word(divisor), shape)

    code_parts = []
    scale_parts = []
    for part in entry.parts:
        codes = source.tensor(f"{part.module}.{FIELD_CODES}")
        part_codes = _select_rows(codes, part.rows).contiguous()
        code_parts.append(part_codes)
        scale_parts.append(
            _bf16_row_scales(
                source.scalar(f"{part.module}.{FIELD_SCALE}"),
                part_codes.shape[0],
                f"{part.module}.{FIELD_SCALE}",
            )
        )
    codes = code_parts[0] if len(code_parts) == 1 else torch.cat(code_parts, dim=0)
    scales = scale_parts[0] if len(scale_parts) == 1 else torch.cat(scale_parts, dim=0)
    return encode_fp8_row_scaled(codes, scales, shape)


def _iter_row_scaled_payload(
    source: ModelOptDirectory,
    name: str,
    shape: Sequence[int],
    *,
    rows_per_chunk: int = 256,
) -> Iterator[bytes]:
    """Stream BF16 rows of *name* through the registered row-scale FP8 profile.

    Mirrors ``fp8_embedding.iter_reader_payload`` (code plane chunk by chunk, the
    plane-alignment padding, then the BF16 scale vector) but reads through
    :class:`HeaderIndex`, which can address a shard whose transfer is still in
    flight; the numeric transform is the registered
    ``fp8_embedding.quantize_bf16_rows``.
    """

    geometry = row_scale_geometry(inventory.FP8, shape)
    meta = source.meta(name)
    if meta.dtype != "BF16" or tuple(meta.shape) != (geometry.n, geometry.k):
        raise ValueError(
            f"{name}: source signature {meta.dtype}{list(meta.shape)} != "
            f"{('BF16', (geometry.n, geometry.k))}"
        )
    scale_words = np.empty(geometry.n, dtype=np.uint16)
    for begin in range(0, geometry.n, rows_per_chunk):
        end = min(begin + rows_per_chunk, geometry.n)
        rows = torch.frombuffer(
            bytearray(source.rows(name, begin, end - begin)), dtype=torch.bfloat16
        ).reshape(end - begin, geometry.k)
        quantized = fp8_embedding.quantize_bf16_rows(rows)
        scale_words[begin:end] = quantized.scales.view(torch.int16).numpy().view(np.uint16)
        yield quantized.codes.numpy().tobytes()
    padding = geometry.scale_plane_offset - geometry.code_plane_bytes
    if padding:
        yield bytes(padding)
    scales = torch.from_numpy(scale_words.view(np.int16)).view(torch.bfloat16)
    yield encode_direct(scales, "BF16")


class _RecipeReader:
    """``ShardReader``-compatible view over :class:`HeaderIndex`.

    ``family_recipe.materialize_recipe`` only needs ``get(name)``, and the
    requirement walkers need the mapping attributes; this is also the only reader
    that can open a shard whose transfer is still in flight.
    """

    def __init__(self, source: ModelOptDirectory) -> None:
        self.source = source
        self.model_dir = source.model_dir
        self.weight_map = source.weight_map

    def get(self, name: str) -> torch.Tensor:
        return self.source.tensor(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.weight_map))

    def has(self, name: str) -> bool:
        return self.source.has(name)

    def metadata(self, names: Iterable[str]) -> dict[str, TensorMetadata]:
        return {
            name: TensorMetadata(
                name=name,
                shard=self.weight_map[name],
                shape=tuple(self.source.meta(name).shape),
                dtype=self.source.meta(name).dtype,
            )
            for name in names
        }

    def close(self) -> None:
        return None

    def __enter__(self) -> "_RecipeReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def _payload_for(
    entry: ObjectPlanEntry,
    source: ModelOptDirectory,
    derived: Mapping[str, torch.Tensor],
    device: torch.device,
) -> bytes | Iterable[bytes]:
    if entry.route == "matrix":
        return _assemble_matrix(entry, source)
    if entry.route == "divisor":
        value = reciprocal_divisor(
            source.scalar(entry.divisor_sources[0]), entry.divisor_sources[0]
        )
        return encode_direct(
            torch.tensor(value, dtype=torch.float32).reshape(()), inventory.FP32
        )
    if entry.route == "vocabulary":
        assert entry.source_tensor is not None
        return _iter_row_scaled_payload(source, entry.source_tensor, entry.spec.shape)
    if entry.route == "requantised_stream":
        assert entry.source_tensor is not None
        return _iter_row_scaled_payload(source, entry.source_tensor, entry.spec.shape)
    if entry.route == "direct":
        recipe = registered_recipe.QUANTIZED_DIRECT_BY_NAME[entry.object_name]
        tensor = family_recipe.materialize_recipe(recipe, _RecipeReader(source))
        payload = family_conversion.encode_tensor_payload(tensor, entry.spec, device)
        del tensor
        return payload
    if entry.route == "official":
        recipe = registered_recipe.OFFICIAL_RECIPES_BY_NAME[entry.object_name]
        tensor = family_recipe.materialize_recipe(
            recipe, _RecipeReader(source), dict(derived)
        )
        payload = family_conversion.encode_tensor_payload(tensor, entry.spec, device)
        del tensor
        return payload
    raise RuntimeError(f"{entry.object_name}: unknown route {entry.route!r}")


# --------------------------------------------------------------------------- #
# preflight / conversion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Preflight:
    source: ModelOptDirectory
    groups: dict[str, SourceGroupPlan]
    source_layout_histogram: dict[str, int]
    declared: dict[str, str]
    plan: ArtifactPlan
    ledger: Ledger
    config_summary: dict[str, object]
    direct_signatures: dict[str, tuple[str, ...]]
    frontend: frontend_policy.FrontendProfile


def _check_frontend(
    source: ModelOptDirectory,
    config: Mapping[str, Any],
    ledger: Ledger,
    *,
    roots: Sequence[str | Path],
    allow_drift: bool,
) -> frontend_policy.FrontendProfile:
    """Delegate the pinned-resource decision to the registered policy.

    The pins are the repository's own harness (``official_resources``); this module
    never re-derives them, so the report and the front door cannot disagree.
    """

    profile = frontend_policy.resolve_frontend_profile(source.model_dir, config, roots=roots)
    for item in profile.resolutions:
        if item.status == "pinned":
            continue
        ledger.deviate(
            "D-FRONTEND-RESOURCE",
            item.name,
            f"status={item.status}: {item.evidence}",
            "the record of this artifact is only as strong as this resource: a "
            "'consistent' file is semantically equivalent but not the pinned bytes, "
            "and an 'unproven' one is neither",
        )
    error = frontend_policy.acceptability_error(profile, allow_unproven=allow_drift)
    if error:
        ledger.refuse(
            "F-FRONTEND-RESOURCE",
            "frontend resources",
            "a frontend resource set that satisfies the registered pins, or an "
            "explicit --allow-frontend-drift",
            error,
        )
    return profile


def preflight(
    model_dir: str | Path,
    *,
    roots: Sequence[str | Path] = (),
    allow_frontend_drift: bool = False,
) -> Preflight:
    source = ModelOptDirectory(model_dir)
    config = family_conversion.load_json(source.model_dir / "config.json")
    config_summary = family_config.validate_config(config)
    ledger = Ledger()
    groups, histogram, declared = plan_source(source, ledger)
    plan = build_artifact_plan(source, groups, ledger)
    frontend = _check_frontend(
        source, config, ledger, roots=roots, allow_drift=allow_frontend_drift
    )
    return Preflight(
        source=source,
        groups=groups,
        source_layout_histogram=histogram,
        declared=declared,
        plan=plan,
        ledger=ledger,
        config_summary=config_summary,
        direct_signatures=unquantized_signatures(source),
        frontend=frontend,
    )


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    roots: Sequence[str | Path] = (),
    allow_frontend_drift: bool = False,
) -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(
            f"the ModelOpt single-source converter output basename must be "
            f"{OUTPUT_BASENAME!r}"
        )
    requested_device = str(device)
    resolved_device = pick_device(device)
    state = preflight(model_dir, roots=roots, allow_frontend_drift=allow_frontend_drift)
    plan = state.plan

    if state.ledger.refusals:
        raise ValueError(
            f"{len(state.ledger.refusals)} refusal(s) block this import:\n  "
            + "\n  ".join(item.render() for item in state.ledger.refusals[:5])
        )
    if plan.truncated:
        raise ValueError(
            f"{len(plan.truncated)} source tensors are not fully present in the local "
            f"shards ({sum(short for _, short in plan.truncated)} bytes short); first: "
            f"{plan.truncated[0][0]} is {plan.truncated[0][1]} bytes short. Run "
            "--plan-only to list them all."
        )

    resources = {resource.name: resource.data for resource in plan.resources}
    derived = {
        draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_head.materialize_draft_head_token_ids(
            plan.draft
        )
    }
    entries = {entry.object_name: entry for entry in plan.entries}
    planned = family_conversion.build_object_plan(plan.specs, resources)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactWriter(
        output, ArtifactIdentity(inventory.MODEL_ID, WEIGHTS_ID), planned.specs
    ) as writer:
        if writer.objects != planned.objects:
            raise RuntimeError("writer object plan differs from the completed plan")
        for index, spec in enumerate(plan.specs, start=1):
            if isinstance(spec, inventory.ResourceSpec):
                writer.write(spec.name, resources[spec.name])
                continue
            entry = entries[spec.name]
            payload = _payload_for(entry, state.source, derived, resolved_device)
            expected = encoded_size(entry.spec.layout, entry.spec.format, entry.spec.shape)
            if isinstance(payload, bytes) and len(payload) != expected:
                raise RuntimeError(
                    f"{spec.name}: payload {len(payload)} bytes != {entry.spec.layout} "
                    f"geometry {expected}"
                )
            writer.write(spec.name, payload)
            del payload
            print(f"[{index}/{len(plan.specs)}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    report = _build_report(
        state=state,
        output=output,
        arguments={
            "model": str(model_dir),
            "out": str(out_path),
            "device": requested_device,
        },
        elapsed=elapsed,
        device=resolved_device,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"complete: {output.stat().st_size} bytes in {elapsed:.1f}s; report={report_path}")
    return report_path


def _dtype_counts(source: ModelOptDirectory) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in source.weight_map:
        dtype = source.meta(name).dtype
        counts[dtype] = counts.get(dtype, 0) + 1
    return counts


def _build_report(
    *,
    state: Preflight,
    output: Path,
    arguments: Mapping[str, object],
    elapsed: float,
    device: torch.device,
) -> dict[str, object]:
    plan = state.plan
    narrow = plan.narrowing
    inexact = [row for row in narrow if row[1] != row[2]]
    worst = max(
        (abs(after - before) / abs(before) for _, before, after in inexact), default=0.0
    )
    source_preflight = family_recipe.SourcePreflight(
        recipe_count=len(plan.entries),
        source_tensor_count=len(plan.required_source_keys),
        source_shard_count=len(state.source.shards),
        source_dtype_counts=_dtype_counts(state.source),
    )
    resources = {resource.name: resource.data for resource in plan.resources}
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=REPO_ROOT,
        model_dir=state.source.model_dir,
        out_path=output,
        arguments=arguments,
        config_summary=state.config_summary,
        source_preflight=source_preflight,
        objects=family_conversion.build_object_plan(plan.specs, resources).objects,
        elapsed_seconds=elapsed,
        final_bytes=output.stat().st_size,
        device=device,
        ranking_path=REPO_ROOT / draft_head.DEFAULT_RANKING,
    )
    report["source"] = {
        "layout": "single ModelOpt directory (no separate unquantized source)",
        "model_path": str(state.source.model_dir.resolve()),
        "declaration": f"{SIDECAR_NAME}, cross-checked against config.json.quantization_config",
        "declared_algo_histogram": _histogram(state.declared.values()),
        "source_layout_histogram": state.source_layout_histogram,
        "source_direct_signatures": {
            signature: len(modules)
            for signature, modules in state.direct_signatures.items()
        },
        "artifact_layout_histogram": plan.layout_histogram,
        "artifact_format_histogram": plan.format_histogram,
        "artifact_route_histogram": plan.route_histogram,
        "objects": len(plan.specs),
    }
    report["deviations"] = {
        "count": len(state.ledger.deviations),
        "by_code": state.ledger.histogram("deviations"),
        "items": [
            {
                "code": item.code,
                "scope": item.scope,
                "detail": item.detail,
                "consequence": item.consequence,
            }
            for item in state.ledger.deviations
        ],
        "fp8_row_scale_narrowing": {
            "source_matrices": len(narrow),
            "exactly_representable": len(narrow) - len(inexact),
            "inexact": len(inexact),
            "worst_relative_error": worst,
            "samples": [
                {"source": key, "fp32": before, "bf16": after} for key, before, after in narrow[:5]
            ],
        },
    }
    report["refusals"] = {
        "count": len(state.ledger.refusals),
        "by_code": state.ledger.histogram("refusals"),
        "items": [
            {
                "code": item.code,
                "subject": item.subject,
                "detail": item.detail,
                "missing_mechanism": item.missing_mechanism,
            }
            for item in state.ledger.refusals
        ],
    }
    report["byte_coverage"] = plan.coverage
    report["frontend"] = {
        "acceptability": frontend_policy.acceptability_error(
            state.frontend, allow_unproven=True
        ),
        "resources": [
            {
                "name": item.name,
                "status": item.status,
                "sha256": item.sha256,
                "evidence": item.evidence,
            }
            for item in state.frontend.resolutions
        ],
    }
    return report


def _histogram(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# plan report
# --------------------------------------------------------------------------- #
def print_plan(state: Preflight, stream=sys.stdout) -> None:
    plan = state.plan

    def out(line: str = "") -> None:
        print(line, file=stream)

    out("=== 1. source structure: layout_plan.select over every module ===")
    out(f"  modules                  : {len(state.groups)}")
    out(f"  declared quantized layers: {len(state.declared)}")
    out(f"  declared quant_algo      : {_histogram(state.declared.values())}")
    for layout, count in sorted(state.source_layout_histogram.items(), key=lambda kv: -kv[1]):
        if layout.startswith("__"):
            continue
        out(f"    {layout:<28} {count} modules")
    out("  unquantized module signatures:")
    for signature, modules in state.direct_signatures.items():
        sample = modules[0]
        out(f"    {len(modules):5d} x {signature}   e.g. {sample}")
    out()
    out("=== 2. artifact object plan (this source's declaration) ===")
    out(f"  objects                  : {len(plan.specs)}")
    for layout, count in sorted(plan.layout_histogram.items(), key=lambda kv: -kv[1]):
        out(f"    {layout:<28} {count}")
    for fmt, count in sorted(plan.format_histogram.items(), key=lambda kv: -kv[1]):
        out(f"    format {fmt:<22} {count}")
    for route, count in sorted(plan.route_histogram.items(), key=lambda kv: -kv[1]):
        out(f"    route  {route:<22} {count}")
    out()
    out("=== 3. deviations ===")
    for code, count in state.ledger.histogram("deviations").items():
        sample = next(item for item in state.ledger.deviations if item.code == code)
        out(f"  {code:<22} {count:5d}  {sample.detail[:110]}")
        out(f"      consequence: {sample.consequence[:160]}")
    out(f"  total deviations         : {len(state.ledger.deviations)}")
    out()
    out("=== 4. refusals ===")
    if not state.ledger.refusals:
        out("  none")
    else:
        for code, count in state.ledger.histogram("refusals").items():
            sample = next(item for item in state.ledger.refusals if item.code == code)
            out(f"  {code:<26} {count:5d}")
            out(f"      {sample.render()}")
    out()
    out("=== 5. frontend resources (the registered pin policy, not re-derived) ===")
    for item in state.frontend.resolutions:
        out(f"  [{item.status:<10}] {item.name}")
        out(f"               {item.evidence[:150]}")
    out(f"  acceptability (drift allowed): "
        f"{frontend_policy.acceptability_error(state.frontend, allow_unproven=True)}")
    out()
    out("=== 6. byte coverage of the local copy ===")
    coverage = plan.coverage
    out(f"  required source tensors  : {coverage['required_tensors']}")
    out(f"  fully present            : {coverage['fully_present']}")
    out(f"  entirely absent          : {coverage['entirely_absent']}")
    out(f"  partially present        : {coverage['partially_present']}")
    out(f"  payload still missing    : {coverage['missing_payload_bytes']} bytes")
    for shard, item in coverage["shards"].items():
        out(f"    {shard:<32} present {item['present_payload_bytes']} / "
            f"claimed {item['claimed_payload_bytes']}")
    for item in coverage["first_broken"]:
        out(f"    {item['name']:<70} ends {item['ends_beyond_present_prefix_by']} beyond")
    out("    ... convert() refuses while any required tensor is short")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, type=Path, help="ModelOpt source directory")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--resource-root",
        action="append",
        default=None,
        help="extra search root for the pinned frontend resources (repeatable)",
    )
    parser.add_argument(
        "--allow-frontend-drift",
        action="store_true",
        help="accept a frontend resource that cannot be proven equal to its pin",
    )
    arguments = parser.parse_args(argv)

    source = arguments.model.resolve()
    out_path = arguments.out or (source.parent / OUTPUT_BASENAME)
    roots = tuple(arguments.resource_root or ())
    state = preflight(
        source, roots=roots, allow_frontend_drift=arguments.allow_frontend_drift
    )
    if arguments.plan_only:
        print_plan(state)
        return 0
    if state.ledger.refusals:
        for item in state.ledger.refusals:
            print(item.render(), file=sys.stderr)
        print(f"\n{len(state.ledger.refusals)} refusal(s); nothing was written", file=sys.stderr)
        return 3
    convert(
        source,
        out_path,
        device=arguments.device,
        roots=roots,
        allow_frontend_drift=arguments.allow_frontend_drift,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
