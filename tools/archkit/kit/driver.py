# -*- coding: utf-8 -*-
"""kit.driver — the shared driver: everything a converter does EXCEPT declare.

A model-side `convert.py` is now:

    from tools.archkit.kit import driver
    from .declaration import DECLARATION          # spec + inventory + recipe

    if __name__ == "__main__":
        raise SystemExit(driver.main(DECLARATION))

and `driver.main` owns, in this order:

  1. ``config.json``            -- refused BY NAME when absent
  2. ``model.safetensors.index.json`` -- refused BY NAME when absent
  3. the spec, through `tools/archkit/arch_spec.load_spec`  (an empty `geometry`
     is not a spec; see the note in that module)
  4. the object plan, through `tools/artifact/container.plan_objects`
  5. the gate battery in `kit.gate`
  6. the write, through `tools/artifact/container.ArtifactWriter`
  7. the re-open, through `tools/artifact/container.Artifact`, and a byte/hash
     reading of what is now on disk

Step 6 is the ONLY place in the tree that opens the output for writing.  A
converter that keeps its own `open(` for the artifact is the shape this module
replaces -- `tools/convert/spark_x2_5_4b/convert.py` said in its own docstring
that its sole `open(` was a safetensors header read, and that is the invariant
here: the driver writes, the model declares.

`--inject` is a deliberate defect injector.  It exists because "the gate refused"
is only evidence if the gate can be shown to refuse *something*: a green run proves
nothing about a battery that is inert.  Each injection names the gate that must
catch it, and `kit/test_kit.sh` asserts the name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.artifact.container import (
    MAGIC,
    Artifact,
    ArtifactError,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    TensorSpec,
)

from .contract import Declaration, EmitContext, KitRefusal, ObjectDecl, RESOURCE, TENSOR
from .gate import Gate, run_gates
from .source import SafetensorsSource

#: injections: name -> (what it corrupts, the gate that must refuse it by name)
INJECTIONS: Mapping[str, tuple[str, str]] = {
    "drop-object": (
        "drops the last planned object, so one source key stops being consumed",
        "source-coverage"),
    "wrong-shape": (
        "perturbs one tensor object's first dimension",
        "geometry"),
    "truncate-payload": (
        "writes one byte short on the first object",
        "container-writer"),
    "empty-geometry": (
        "empties the spec's geometry, so the spec is no longer a spec",
        "spec"),
}


class ConversionReport(dict):
    """The driver's result, printable as the report block."""

    def render(self) -> str:
        lines: list[str] = []
        lines.append("== plan")
        for gate in self["gates"]:
            lines.append(gate.line())
        lines.append("== R29 denominator reading")
        lines.append("   " + self["coverage"].reading)
        lines.append("== written")
        for key in ("path", "bytes", "sha256", "magic", "model_id", "weights_id",
                    "objects"):
            lines.append("   %-11s %s" % (key + ":", self[key]))
        return "\n".join(lines)


def _identity(decl: Declaration) -> ArtifactIdentity:
    return ArtifactIdentity(decl.model_id, decl.weights_id)


def _specs(decl: Declaration, objects: Sequence[ObjectDecl]) -> list:
    out: list = []
    for obj in objects:
        if obj.kind == TENSOR:
            out.append(TensorSpec(obj.name, tuple(obj.shape), obj.format, obj.layout))
        else:
            out.append(ResourceSpec(obj.name, obj.encoding, obj.bytes))
    return out


def apply_injection(decl: Declaration, injection: str) -> Declaration:
    """Return a copy of `decl` corrupted in the one named way."""
    if injection not in INJECTIONS:
        raise KitRefusal("unknown injection %r; known: %s"
                         % (injection, sorted(INJECTIONS)))
    objects = list(decl.objects)
    spec = dict(decl.spec)
    if injection == "drop-object":
        # Drop the LAST object that consumes a source key, skipping aliases and any
        # object an alias points at.  Two wrong versions were measured here first:
        # `objects[:-1]` dropped an alias (which consumes nothing, so every reading
        # stayed green -- an injection that tests nothing), and "the first consumer"
        # dropped the tie TARGET, so `Declaration`'s own validation refused it before
        # the battery ran and the gate named in INJECTIONS never fired.  The injection
        # has to survive declaration validation in order to reach the gate it claims.
        aliases = {o.tie_to for o in objects if o.tie_to}
        index = next(i for i in range(len(objects) - 1, -1, -1)
                     if objects[i].sources and not objects[i].tie_to
                     and objects[i].name not in aliases)
        objects.pop(index)
    elif injection == "wrong-shape":
        index = next(i for i, o in enumerate(objects)
                     if o.kind == TENSOR and o.sources)
        victim = objects[index]
        shape = list(victim.shape)
        shape[0] = shape[0] + 1
        objects[index] = ObjectDecl(
            name=victim.name, role=victim.role, kind=victim.kind,
            shape=tuple(shape), format=victim.format, layout=victim.layout,
            sources=victim.sources, tie_to=victim.tie_to, note=victim.note)
    elif injection == "empty-geometry":
        spec["geometry"] = {}
    return Declaration(
        model_id=decl.model_id, weights_id=decl.weights_id, family=decl.family,
        spec=spec, objects=tuple(objects), recipes=decl.recipes,
        ignored_source_prefixes=decl.ignored_source_prefixes,
        resources=decl.resources, work_items=decl.work_items)


def _resolve_spec(decl: Declaration, spec_dir: Path | None) -> Mapping[str, Any]:
    """Load the spec through the tree's own validator when a file is available."""
    if spec_dir is None:
        return decl.spec
    from tools.archkit.arch_spec import load_spec  # local import: keeps kit importable alone

    candidates = sorted(spec_dir.glob("*_spec.json"))
    stem = decl.model_id.replace("-", "_").replace(".", "_")
    match = [p for p in candidates if p.stem.replace("-", "_") == stem + "_spec"
             or p.stem.replace("-", "_") == stem]
    if not match:
        raise KitRefusal(
            "no spec file for %s under %s; a declaration without a spec file on disk "
            "cannot be re-read by a second party (candidates: %s)"
            % (decl.model_id, spec_dir, [p.name for p in candidates]))
    return load_spec(str(match[0]))


def run(
    decl: "Declaration | Callable[[SafetensorsSource], Declaration]",
    model_dir: str | Path,
    out: str | Path,
    *,
    plan_only: bool = False,
    inject: str = "",
    spec_dir: str | Path | None = None,
) -> ConversionReport:
    report = ConversionReport()
    report["inject"] = inject
    source: SafetensorsSource | None = None
    if not isinstance(decl, Declaration):
        # 1/2: the two required files, refused by name inside the reader, BEFORE the
        # builder runs -- a builder needs `source.config`, so it cannot be asked to
        # produce a declaration for a directory that has no config.json.
        try:
            source = SafetensorsSource(model_dir)
        except KitRefusal as exc:
            print("  REFUSED: %s" % exc, file=sys.stderr)
            report["refused"] = str(exc)
            raise
        decl = decl(source)
    if inject:
        decl = apply_injection(decl, inject)
        print("  INJECTION %s: %s" % (inject, INJECTIONS[inject][0]), file=sys.stderr)

    if decl.spec.get("geometry"):
        decl_spec = _resolve_spec(decl, Path(spec_dir) if spec_dir else None)
        decl = Declaration(
            model_id=decl.model_id, weights_id=decl.weights_id, family=decl.family,
            spec=decl_spec, objects=decl.objects, recipes=decl.recipes,
            ignored_source_prefixes=decl.ignored_source_prefixes,
            resources=decl.resources, work_items=decl.work_items)
    elif inject != "empty-geometry":
        raise KitRefusal(
            "spec for %s has an empty 'geometry' object; an extraction that found no "
            "geometry keys must not pass as a spec" % decl.model_id)

    # 1/2: the two required files, each refused by name inside the reader.
    if source is None:
        try:
            source = SafetensorsSource(model_dir)
        except KitRefusal as exc:
            print("  REFUSED: %s" % exc, file=sys.stderr)
            report["refused"] = str(exc)
            raise

    # 3/4/5: gates
    gates = run_gates(decl, source)
    report["gates"] = gates
    report["coverage"] = next(g for g in gates if g.name == "source-coverage")
    red = [g for g in gates if not g.ok]
    if red:
        for gate in red:
            print(gate.line(), file=sys.stderr)
        first = red[0]
        raise KitRefusal(
            "%s gate refused the declaration: %s"
            % (first.name, first.reasons[0] if first.reasons else first.reading))

    print("== %s / %s (%s)" % (decl.model_id, decl.weights_id, decl.family))
    for gate in gates:
        print(gate.line())
    print("  R29 " + report["coverage"].reading)

    planned = _specs(decl, decl.objects)

    if plan_only:
        report["path"] = "(plan only; nothing written)"
        report["bytes"] = 0
        report["sha256"] = ""
        report["magic"] = repr(MAGIC)
        report["model_id"] = decl.model_id
        report["weights_id"] = decl.weights_id
        report["objects"] = len(planned)
        return report

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    by_name = {obj.name: obj for obj in decl.objects}
    # A payload that is a BUFFER is materialised once and reused; a payload that is a
    # STREAM is not cached, because a generator is consumed by its first reader and a
    # cached exhausted generator is exactly how a tie would silently write zero bytes.
    cache: dict[str, Any] = {}

    def produce(obj: ObjectDecl) -> Any:
        recipe = decl.recipes[obj.role]
        ctx = EmitContext(declaration=decl, spec=decl.spec,
                          geometry=decl.spec.get("geometry") or {}, emit=payload_of)
        value = recipe(obj, source, ctx)
        if inject == "truncate-payload":
            def short(value=value):
                yielded = False
                for chunk in value:
                    if not yielded:
                        chunk = memoryview(chunk)[:-1]
                        yielded = True
                    yield chunk
            return short()
        return value

    def payload_of(name: str) -> Any:
        obj = by_name[name]
        if obj.tie_to:
            return payload_of(obj.tie_to)
        if name in cache:
            return cache[name]
        value = produce(obj)
        if isinstance(value, (bytes, bytearray, memoryview)):
            cache[name] = value
        return value

    # The container's own refusals are re-raised as `container-writer` refusals so
    # that every way this driver can fail is ONE exception type with a named cause in
    # its first clause.  A caller that had to distinguish `ArtifactError` from
    # `KitRefusal` would be reading the implementation instead of the contract.
    try:
        with ArtifactWriter(out, _identity(decl), planned) as writer:
            for index, obj in enumerate(decl.objects, start=1):
                if obj.tie_to:
                    print("  [%d/%d] %s <- %s (tie)" % (index, len(decl.objects),
                                                        obj.name, obj.tie_to))
                else:
                    print("  [%d/%d] %s (%s)" % (index, len(decl.objects), obj.name,
                                                 obj.role))
                writer.write(obj.name, payload_of(obj.name))
    except ArtifactError as exc:
        raise KitRefusal("container-writer: %s" % exc) from exc

    # 7: re-open with the container's own reader and hash what is on disk
    digest = hashlib.sha256()
    size = 0
    with open(out, "rb") as handle:
        while True:
            chunk = handle.read(1 << 22)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    with Artifact.open(out) as artifact:
        names = [o.name for o in artifact.objects]
        if names != [o.name for o in decl.objects]:
            raise KitRefusal(
                "re-open found %d objects in an order/name set that differs from the "
                "declaration's %d" % (len(names), len(decl.objects)))
        identity = artifact.identity
    report["path"] = str(out)
    report["bytes"] = size
    report["sha256"] = digest.hexdigest()
    report["magic"] = repr(MAGIC)
    report["model_id"] = identity.model_id
    report["weights_id"] = identity.weights_id
    report["objects"] = len(names)
    return report


def main(
    target: "Declaration | Callable[[SafetensorsSource], Declaration]",
    argv: Sequence[str] | None = None,
) -> int:
    """CLI entry.  `target` is a Declaration, or a builder that takes the source.

    The builder form is what a model-side `convert.py` uses, because the object
    plan's layer count comes from the checkpoint:

        if __name__ == "__main__":
            raise SystemExit(driver.main(build_declaration))
    """
    parser = argparse.ArgumentParser(
        description="shared archkit driver")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan", action="store_true",
                        help="run config/index/gates and stop (no payloads)")
    parser.add_argument("--inject", default="",
                        help="defect injector: " + ", ".join(sorted(INJECTIONS)))
    parser.add_argument("--spec-dir", type=Path, default=None,
                        help="directory holding <id>_spec.json")
    args = parser.parse_args(argv)

    if not isinstance(target, Declaration):
        try:
            source = SafetensorsSource(args.model)
        except KitRefusal as exc:
            print("  REFUSED: %s" % exc, file=sys.stderr)
            return 2
        declaration = target(source)
    else:
        declaration = target

    try:
        report = run(declaration, args.model, args.out, plan_only=args.plan,
                     inject=args.inject, spec_dir=args.spec_dir)
    except KitRefusal as exc:
        print("  REFUSED: %s" % exc, file=sys.stderr)
        return 2
    except ArtifactError as exc:
        print("  REFUSED: container-writer: %s" % exc, file=sys.stderr)
        return 2
    print(report.render())
    if declaration.work_items:
        print("== the artifact is complete; what the ENGINE still needs, by name:")
        for name, detail in declaration.work_items:
            print("   - [%s] %s" % (name, detail))
    return 0
