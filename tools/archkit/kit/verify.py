# -*- coding: utf-8 -*-
"""kit.verify — read a produced `.ninfer` back and say what it is, with a denominator.

This is the acceptance side of the driver.  It re-opens the artifact with
`tools/artifact/container.py`'s own reader -- never with the writer's in-memory
plan -- and compares every object against the checkpoint's own shard headers
through the DECLARATION's name mapping, so the reading is a per-object equality
with a denominator rather than a total.

    python3 -m tools.archkit.kit.verify \
      --artifact /home/user/models/minicpm5_1b_bf16.ninfer \
      --model    /mnt/c/.../MiniCPM5-1B \
      --declaration tools.convert.minicpm5_1b.declaration:build_declaration

The `--declaration` argument matters and is not a convenience.  An artifact's
object names are the ENGINE's vocabulary (`text/layers/3/attention/query`), while
the checkpoint's index is HF's (`model.layers.3.self_attn.q_proj.weight`).  A
verifier that compared the two sets directly would report 219 objects missing and
be measuring its own ignorance; the mapping lives in the declaration, and the
verification has to go through the same statement that produced the file, or it is
checking a different claim.

Readings
--------

    objects_declared N   objects_in_file N   order_equal yes
    index_keys       M   matched          M   set_equal   yes
    geometry_equal   K/K (shape AND stored byte count, per object, vs shard headers)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "tools.archkit.kit"

from tools.artifact.container import Artifact, ArtifactError

from .contract import Declaration, KitRefusal
from .source import SafetensorsSource


def resolve_builder(spec: str) -> Callable:
    if ":" not in spec:
        raise KitRefusal("--declaration must be 'module:function', got %r" % spec)
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if function is None:
        raise KitRefusal("%s has no attribute %s" % (module_name, function_name))
    return function


def sha256_of(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 22)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _product(shape) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


def verify(artifact_path, model_dir=None, declaration_spec: str | None = None) -> int:
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise KitRefusal("missing artifact %s" % artifact_path)
    digest, size = sha256_of(artifact_path)
    with open(artifact_path, "rb") as handle:
        magic = handle.read(8)

    with Artifact.open(artifact_path) as artifact:
        print("== artifact")
        print("   path        %s" % artifact_path)
        print("   bytes       %d" % size)
        print("   sha256      %s" % digest)
        print("   magic       %r" % magic)
        print("   model_id    %s" % artifact.identity.model_id)
        print("   weights_id  %s" % artifact.identity.weights_id)
        print("   objects     %d" % len(artifact.objects))
        print("   payload_at  %d" % artifact.payload_offset)
        names_in_file = [o.name for o in artifact.objects]

        if model_dir is None:
            return 0
        source = SafetensorsSource(model_dir)
        if declaration_spec is None:
            raise KitRefusal(
                "a source-equality reading needs --declaration: the artifact's object "
                "names are the engine's vocabulary and the index's keys are HF's, so "
                "without the mapping the comparison would be between two naming "
                "schemes rather than between a file and a checkpoint")

        builder = resolve_builder(declaration_spec)
        decl: Declaration = builder(source)
        declared_names = [o.name for o in decl.objects]
        order_equal = declared_names == names_in_file

        by_name = {o.name: o for o in decl.objects}
        problems: list[str] = []
        matched = 0
        geometry_equal = 0
        tensor_objects = 0
        for obj in artifact.objects:
            if obj.kind != "tensor":
                continue
            tensor_objects += 1
            decl_obj = by_name.get(obj.name)
            if decl_obj is None:
                problems.append("artifact object %s is not in the declaration" % obj.name)
                continue
            keys = list(decl_obj.sources)
            if decl_obj.tie_to:
                keys = list(by_name[decl_obj.tie_to].sources)
            if not keys:
                problems.append("object %s maps to no source key and is not an alias"
                                % obj.name)
                continue
            matched += 1
            shapes = []
            stored = 0
            for key in keys:
                _dtype, shape, payload = source.meta(key)
                shapes.append(shape)
                stored += payload
            if len(shapes) == 1:
                expect_shape = shapes[0]
            else:
                expect_shape = (sum(s[0] for s in shapes),) + tuple(shapes[0][1:])
            if tuple(obj.shape) != tuple(expect_shape):
                problems.append("object %s: artifact shape %s, source header %s"
                                % (obj.name, tuple(obj.shape), tuple(expect_shape)))
                continue
            if decl_obj.role == "matrix" and obj.bytes != stored:
                problems.append("object %s: artifact stores %d bytes, its source "
                                "tensors store %d" % (obj.name, obj.bytes, stored))
                continue
            geometry_equal += 1

        index_keys = set(source.keys())
        consumed: set[str] = set()
        for decl_obj in decl.objects:
            if decl_obj.tie_to:
                consumed.update(by_name[decl_obj.tie_to].sources)
            consumed.update(decl_obj.sources)
        ignored = sorted(k for k in index_keys
                         if any(k.startswith(p) for p in decl.ignored_source_prefixes))
        undeclared = sorted(index_keys - consumed - set(ignored))
        for key in undeclared[:8]:
            problems.append("source key %s is in the index and in no artifact object"
                            % key)

        print("== R29 denominator reading (the file vs the checkpoint's own index)")
        print("   objects_declared %d   objects_in_file %d   order_equal %s"
              % (len(decl.objects), len(names_in_file), "yes" if order_equal else "NO"))
        print("   index_keys       %d   matched          %d   set_equal   %s"
              % (len(index_keys), len(consumed), "yes" if not undeclared else "NO"))
        print("   geometry_equal   %d/%d  (shape and stored bytes, per object, against "
              "the shard headers)" % (geometry_equal, tensor_objects))
        print("   problems         %d" % len(problems))
        for line in problems:
            print("      - %s" % line)
        return 1 if problems else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=None,
                        help="checkpoint directory; without it only the container "
                             "contract is read back, not the source equality")
    parser.add_argument("--declaration", default=None,
                        help="module:function returning the Declaration for this "
                             "checkpoint, e.g. "
                             "tools.convert.minicpm5_1b.declaration:build_declaration")
    args = parser.parse_args(argv)
    try:
        return verify(args.artifact, args.model, args.declaration)
    except (KitRefusal, ArtifactError) as exc:
        print("  REFUSED: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
