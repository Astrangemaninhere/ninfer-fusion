# -*- coding: utf-8 -*-
"""kit.contract — the ONE declaration a new model writes, and its named refusals.

This module is the whole of the "what does a new model have to write" answer.  It
is deliberately data-only: no torch, no file reads, no engine knowledge.  The
shared driver (`kit.driver`) consumes a `Declaration` and owns everything else.

The four things a new model writes (nothing more):

  1. a spec              ``tools/archkit/specs/<id>_spec.json``  (geometry + layer kinds)
  2. an inventory        roles -> ordered object names  (``inventory.py``)
  3. a recipe            role -> payload expression + tie decisions  (``recipe.py``)
  4. a declaration       the two above bound together  (``convert.py``)

and the shared driver owns: config/index reading, spec validation, object
planning, offset allocation, payload streaming, the gate battery, the write, the
re-open, and every byte of the report.

A refusal is a `KitRefusal` whose message BEGINS by naming the thing that is
wrong.  That is a hard rule here: a checkpoint missing a file must produce
"missing <file> under <dir>", never a Python stack trace that says the program
broke.  `dl/sparkconvert` measured the opposite shape once (an uncaught
traceback out of `read_index`), and `tools/convert/spark_x2_5_4b/convert.py`
carries the repair in a comment.  The rule is enforced here instead of being
re-remembered per converter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

TENSOR = "tensor"
RESOURCE = "resource"


class KitRefusal(ValueError):
    """A refusal that names its cause in the first clause of the message."""


@dataclass(frozen=True, slots=True)
class ObjectDecl:
    """One planned artifact object, declared before any byte is read.

    `sources` is the load-bearing field for the R29 reading: it lists the source
    tensor keys that this object is built from, so the driver can prove that
    consumed-union-ignored equals the checkpoint index EXACTLY, one-to-one.
    A fused object (q|k|v) lists several; a pass-through object lists one.
    """

    name: str
    role: str
    kind: str = TENSOR
    shape: tuple[int, ...] = ()
    #: artifact format vocabulary; must be one `tools/artifact/layouts.py` knows
    format: str = ""
    #: artifact layout name; must be one `tools/artifact/layouts.py` knows
    layout: str = ""
    #: resource encoding (`RAW_BYTES_V1`) for `kind == RESOURCE`
    encoding: str = ""
    #: declared resource byte length, for `kind == RESOURCE`
    bytes: int = 0
    sources: tuple[str, ...] = ()
    #: declared as an alias of another object: no payload is produced for it
    tie_to: str = ""
    #: free text printed in the plan, e.g. the engine binding this object feeds
    note: str = ""


@dataclass(frozen=True, slots=True)
class Declaration:
    """What a model-side module hands the shared driver."""

    model_id: str
    weights_id: str
    family: str
    #: the loaded archkit spec (see `tools/archkit/arch_spec.py:load_spec`)
    spec: Mapping[str, Any]
    #: ORDERED: the artifact directory is written in exactly this order
    objects: tuple[ObjectDecl, ...]
    #: role -> payload expression.  Called as
    #:     recipes[role](obj, source, ctx) -> bytes | chunk iterable
    #: `ctx` is a `kit.contract.EmitContext`; `ctx.emit(name)` re-resolves another
    #: object's payload so a tie costs one function call and zero extra bytes.
    recipes: Mapping[str, Callable[..., Any]]
    #: source keys that this target deliberately does not consume.  Every prefix
    #: must account for at least one index key, and the union with `consumed`
    #: must equal the index exactly; anything else is refused by name.
    ignored_source_prefixes: tuple[str, ...] = ()
    #: (path, bytes) pairs written into the artifact directory as resources
    resources: Mapping[str, bytes] = None  # type: ignore[assignment]
    #: what the ENGINE still needs, by name; printed, never guessed at
    work_items: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.resources is None:
            object.__setattr__(self, "resources", {})
        if not self.model_id or not self.weights_id:
            raise KitRefusal("declaration is missing model_id or weights_id")
        if not self.objects:
            raise KitRefusal("declaration %s has no objects" % self.model_id)
        names = [o.name for o in self.objects]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise KitRefusal("declaration %s declares duplicate object names: %s"
                             % (self.model_id, dupes))
        known = set(self.recipes)
        for obj in self.objects:
            if obj.tie_to:
                if obj.tie_to not in names:
                    raise KitRefusal(
                        "object %s is tied to %s, which this declaration does not "
                        "declare" % (obj.name, obj.tie_to))
                continue
            if obj.role not in known:
                raise KitRefusal(
                    "object %s has role %r, which the recipe does not implement; "
                    "implemented roles: %s"
                    % (obj.name, obj.role, sorted(self.recipes)))
            if obj.kind == TENSOR and (not obj.shape or not obj.format or not obj.layout):
                raise KitRefusal(
                    "tensor object %s must declare shape, format and layout" % obj.name)
            if obj.kind == RESOURCE and (not obj.encoding or obj.bytes <= 0):
                raise KitRefusal(
                    "resource object %s must declare encoding and a positive byte "
                    "length" % obj.name)


@dataclass(frozen=True, slots=True)
class EmitContext:
    """Handed to every recipe call: the plan, the spec, and the tie resolver."""

    declaration: Declaration
    spec: Mapping[str, Any]
    geometry: Mapping[str, Any]
    emit: Callable[[str], Any]
