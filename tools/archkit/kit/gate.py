# -*- coding: utf-8 -*-
"""kit.gate — the shared gate battery: the half that is NOT written per model.

Every gate returns a `Gate` record with a name, an `ok` flag, a one-line reading
and a tuple of named reasons.  Nothing here is model-specific: the per-model
statement lives in the declaration's object list and recipes, and these gates are
the machine check that the declaration is complete and consistent with the
checkpoint on disk.

Gate names (stable, they appear in the refusal text and in the report):

  ``source-coverage``  consumed-union-ignored == the index exactly, disjoint (R29)
  ``geometry``         each object's shape is what its source tensors actually are
  ``spec-vs-config``   the spec's geometry equals the checkpoint's config.json
  ``layer-kinds``      the layer-kind schedule is total and uses the known vocabulary
  ``plan``             `tools/artifact/container.plan_objects` accepts the plan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tools.artifact.container import ArtifactError, TensorSpec, ResourceSpec, plan_objects

from .contract import Declaration, KitRefusal, ObjectDecl, RESOURCE, TENSOR

#: the only layer kinds the shared family runtime can execute today.  This is not
#: a wish list: `src/targets/qwen3_6/impl/runtime/text_context_impl.h:1144-1203`
#: walks the schedule as a full/gdn binary split, and sliding is a modifier on the
#: full arm.  heritage's reversal table (REPORT.md 判据 3.4) names every family
#: that needs a third kind -- MoE router, Mamba/SSM, short convolution -- and this
#: set is where that boundary is written down as a machine-checkable fact.
KNOWN_LAYER_KINDS = frozenset({
    "full", "full_attention",
    "sliding", "sliding_attention",
    "gdn", "linear_attention",
})

#: dtypes a BF16/FP32/I32 pass-through object may be declared from
DIRECT_SOURCE_DTYPE = {"BF16": 2, "F32": 4, "I32": 4, "U8": 1, "F8_E4M3": 1,
                       "F8_E4M3FN": 1}


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    ok: bool
    reading: str
    reasons: tuple[str, ...] = ()

    def line(self) -> str:
        head = "OK  " if self.ok else "RED "
        body = "%s %-16s %s" % (head, self.name, self.reading)
        if self.reasons:
            body += "\n" + "\n".join("      - " + r for r in self.reasons)
        return body


def _tensor_specs(decl: Declaration) -> list:
    specs: list = []
    for obj in decl.objects:
        if obj.kind == TENSOR:
            specs.append(TensorSpec(obj.name, tuple(obj.shape), obj.format, obj.layout))
        else:
            specs.append(ResourceSpec(obj.name, obj.encoding, obj.bytes))
    return specs


# ---------------------------------------------------------------- gate: plan


def gate_plan(decl: Declaration) -> Gate:
    try:
        objects = plan_objects(_tensor_specs(decl))
    except (ArtifactError, TypeError, ValueError) as exc:
        return Gate("plan", False, "container rejected the declaration",
                    ("%s: %s" % (type(exc).__name__, exc),))
    payload = sum(o.bytes for o in objects)
    return Gate("plan", True,
                "%d objects, %d payload bytes, %d resource(s)"
                % (len(objects), payload,
                   sum(1 for o in objects if o.kind != TENSOR)))


# ------------------------------------------------------- gate: source coverage


def gate_source_coverage(decl: Declaration, source) -> Gate:
    index_keys = set(source.keys())
    #: key -> the objects that consume it, with the tied ones marked.  Two objects
    #: may share a source key only when one of them is an ALIAS (`tie_to`): that is
    #: what a tied embedding is, and it is the one duplication that is not a defect.
    consumed: dict[str, list[tuple[str, bool]]] = {}
    for obj in decl.objects:
        for key in obj.sources:
            consumed.setdefault(key, []).append((obj.name, bool(obj.tie_to)))
    reasons: list[str] = []
    for key, users in sorted(consumed.items()):
        solid = [name for name, tied in users if not tied]
        if len(solid) > 1:
            reasons.append("source key %s is consumed by %d objects that are not "
                           "aliases: %s" % (key, len(solid), solid))
    unknown = sorted(k for k in consumed if k not in index_keys)
    for key in unknown:
        reasons.append("object %s consumes %s, which the index does not carry"
                       % (",".join(n for n, _ in consumed[key]), key))
    ignored = sorted(k for k in index_keys
                     if any(k.startswith(p) for p in decl.ignored_source_prefixes))
    for prefix in decl.ignored_source_prefixes:
        if not any(k.startswith(prefix) for k in index_keys):
            reasons.append("IGNORED_SOURCE_PREFIXES entry %r matches no index key; an "
                           "ignore rule that ignores nothing is a stale claim" % prefix)
    overlap = sorted(set(consumed) & set(ignored))
    for key in overlap:
        reasons.append("source key %s is both consumed and ignored" % key)
    missing = sorted(index_keys - set(consumed) - set(ignored))
    for key in missing[:12]:
        reasons.append("source key %s is neither consumed by any object nor matched "
                       "by IGNORED_SOURCE_PREFIXES" % key)
    if len(missing) > 12:
        reasons.append("... and %d more unconsumed keys" % (len(missing) - 12))
    reading = ("objects=%d index_keys=%d consumed=%d ignored=%d consumed+ignored=%d/%d "
               "disjoint=%s"
               % (len(decl.objects), len(index_keys), len(consumed), len(ignored),
                  len(set(consumed) | set(ignored)), len(index_keys),
                  "yes" if not overlap else "NO"))
    return Gate("source-coverage", not reasons, reading, tuple(reasons))


# ------------------------------------------------------------- gate: geometry


def _product(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


def gate_geometry(decl: Declaration, source) -> Gate:
    reasons: list[str] = []
    checked = 0
    fused = 0
    for obj in decl.objects:
        if obj.kind != TENSOR or not obj.sources:
            continue
        metas = []
        for key in obj.sources:
            try:
                metas.append(source.meta(key))
            except KitRefusal as exc:
                reasons.append(str(exc))
                metas = []
                break
        if not metas:
            continue
        checked += 1
        if len(metas) == 1:
            dtype, shape, _payload = metas[0]
            if tuple(obj.shape) != shape:
                reasons.append("object %s declares shape %s; source %s holds %s"
                               % (obj.name, tuple(obj.shape), obj.sources[0], shape))
            if _product(shape) != _product(obj.shape):
                reasons.append("object %s element count %d != source %s element count %d"
                               % (obj.name, _product(obj.shape), obj.sources[0],
                                  _product(shape)))
            expected = DIRECT_SOURCE_DTYPE.get(dtype)
            if dtype == "BF16" and obj.format != "BF16":
                reasons.append("object %s declares format %s over a BF16 source; a "
                               "pass-through object must declare BF16"
                               % (obj.name, obj.format))
        else:
            fused += 1
            rows = sum(m[1][0] for m in metas)
            tails = {tuple(m[1][1:]) for m in metas}
            if len(tails) != 1:
                reasons.append("object %s fuses sources with different trailing "
                               "shapes: %s" % (obj.name, sorted(tails)))
            elif tuple(obj.shape) != (rows,) + tuple(next(iter(tails))):
                reasons.append("object %s declares fused shape %s; its %d sources "
                               "stack to shape %s"
                               % (obj.name, tuple(obj.shape), len(metas),
                                  (rows,) + tuple(next(iter(tails)))))
    return Gate("geometry", not reasons,
                "%d object(s) checked against the checkpoint headers (%d fused)"
                % (checked, fused), tuple(reasons))


# --------------------------------------------------------- gate: spec vs config


_SPEC_TO_CONFIG = (
    ("hidden", "hidden_size"),
    ("layers", "num_hidden_layers"),
    ("query_heads", "num_attention_heads"),
    ("kv_heads", "num_key_value_heads"),
    ("head_dim", "head_dim"),
    ("vocab", "vocab_size"),
    ("intermediate", "intermediate_size"),
)


def gate_spec_vs_config(decl: Declaration, source) -> Gate:
    """The spec is a claim about the family; config.json is what the source says.

    A mismatch is refused by name rather than absorbed: everything downstream
    (object shapes, the engine's static_asserts, the layer schedule) is derived
    from the spec, so a spec that disagrees with the checkpoint would build a
    structurally valid artifact for the wrong model.
    """
    cfg = source.config
    text = cfg.get("text_config") or cfg
    geometry = decl.spec.get("geometry") or {}
    reasons: list[str] = []
    checked = 0
    for spec_key, cfg_key in _SPEC_TO_CONFIG:
        want = geometry.get(spec_key)
        have = text.get(cfg_key)
        if want is None:
            reasons.append("spec geometry has no %r entry" % spec_key)
            continue
        if have is None:
            continue
        checked += 1
        if int(want) != int(have):
            reasons.append("spec geometry %s=%r but %s declares %s=%r"
                           % (spec_key, want, CONFIG_NAME_FALLBACK, cfg_key, have))
    arch = cfg.get("architectures") or []
    spec_arch = ((decl.spec.get("hf") or {}).get("architectures") or [])
    if spec_arch and arch and list(spec_arch) != list(arch):
        reasons.append("spec hf.architectures %s but config.json declares %s"
                       % (list(spec_arch), list(arch)))
    return Gate("spec-vs-config", not reasons,
                "%d geometry field(s) agree with %s; architectures=%s"
                % (checked, CONFIG_NAME_FALLBACK, arch or "?"), tuple(reasons))


CONFIG_NAME_FALLBACK = "config.json"


# --------------------------------------------------------- gate: layer kinds


def gate_layer_kinds(decl: Declaration) -> Gate:
    spec = decl.spec
    kinds = spec.get("layer_types")
    layers = (spec.get("geometry") or {}).get("layers")
    reasons: list[str] = []
    if kinds is None:
        return Gate("layer-kinds", True,
                    "no layer_types in the spec: a single-kind stack, assumed total")
    if layers is not None and len(kinds) != int(layers):
        reasons.append("the spec's layer_types has %d entries but geometry.layers=%s; "
                       "a schedule that does not cover every layer leaves the "
                       "remaining layers unexecuted and silent"
                       % (len(kinds), layers))
    unknown = sorted({k for k in kinds if k not in KNOWN_LAYER_KINDS})
    for kind in unknown:
        reasons.append("layer kind %r is outside the shared runtime's vocabulary %s"
                       % (kind, sorted(KNOWN_LAYER_KINDS)))
    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return Gate("layer-kinds", not reasons,
                "%d layers: %s" % (len(kinds),
                                   ", ".join("%s=%d" % kv for kv in sorted(counts.items()))),
                tuple(reasons))


def run_gates(decl: Declaration, source) -> tuple[Gate, ...]:
    return (
        gate_plan(decl),
        gate_spec_vs_config(decl, source),
        gate_layer_kinds(decl),
        gate_source_coverage(decl, source),
        gate_geometry(decl, source),
    )
