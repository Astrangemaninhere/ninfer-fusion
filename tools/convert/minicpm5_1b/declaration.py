# -*- coding: utf-8 -*-
"""minicpm5_1b.declaration — the binding of the spec, the inventory and the recipe.

This module is the ONLY one that knows all three, and it is deliberately a pure
function of the checkpoint: `build_declaration(source) -> Declaration`.  No file
is opened, no byte is read, nothing is written.  That is what lets the same
declaration drive

  * the real conversion of MiniCPM5-1B       (24 layers, 219 objects),
  * `kit/demo.py`'s synthetic checkpoint     (3 layers, tied embedding),
  * `kit/scaffold.py`'s generated skeleton, and
  * the negative tests, which corrupt it on purpose,

with no per-consumer copies of the geometry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tools.archkit.kit.contract import Declaration, ObjectDecl, RESOURCE, TENSOR

from . import inventory, recipe

SPEC_NAME = "minicpm5_1b_spec.json"
LAYOUT = "contiguous-le-v1"
FORMAT = "BF16"

WORK_ITEMS: tuple[tuple[str, str], ...] = (
    ("engine-family", "per-kind rope + a single rotary_dim slot: this stack needs one "
                      "theta (5000000) everywhere, so `rope_theta_at` is NOT required "
                      "here -- it is the spark stack that needs it. `config.h` must "
                      "still pin rope_theta."),
    ("engine-family", "the family runtime normalises q and k unconditionally "
                      "(text_context_impl.h:1204-1205) and this checkpoint carries no "
                      "q_norm/k_norm tensor in any of its 24 layers; the "
                      "`qk_norm_enabled` accessor that would gate it has 0 references "
                      "in the shared runtime today (see kit/refcount.py)."),
    ("registry", "src/targets/registry.h|.cpp and src/CMakeLists.txt are NOT touched by "
                 "this line: the generator prints the exact insertion points and the "
                 "batch carries only new paths, because two other lines edited "
                 "registry.h/.cpp on 2026-09-23 and an overlay is theirs to fire, not "
                 "mine."),
    ("tests", "tests/targets/minicpm5_1b/ does not exist in the tree; the scaffold emits "
              "a skeleton, and heritage names this as the thin spot in the pipeline "
              "(tests/targets/ holds four directories and spark is not one of them)."),
)


def default_spec_dir() -> Path:
    """`<repo>/tools/archkit/specs` — the tree's one spec directory.

    A spec is looked up through the tree and not through `__file__` on purpose: the
    spec is an ARCHKIT artifact with a life of its own (`arch_spec.load_spec`
    validates it, the scaffold generator consumes it, `kit/check.py` diffs it back),
    so a converter that carried a private copy would be the second statement of the
    same geometry that the joint exists to prevent.
    """
    return Path(__file__).resolve().parents[3] / "tools" / "archkit" / "specs"


def load_spec(spec_dir: str | Path | None = None) -> dict:
    from tools.archkit.kit.contract import KitRefusal

    directory = Path(spec_dir) if spec_dir is not None else default_spec_dir()
    path = directory / SPEC_NAME
    if not path.is_file():
        raise KitRefusal("missing spec file %s" % path)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_declaration(source, spec_dir: str | Path | None = None) -> Declaration:
    """One declaration, derived from the spec plus the checkpoint's own index."""
    spec = load_spec(spec_dir)
    geometry = spec["geometry"]
    layers = int(geometry["layers"])
    config: Mapping[str, Any] = source.config

    objects: list[ObjectDecl] = []
    for name, sources, role in inventory.objects(layers):
        tie = recipe.tie_of(dict(config), name)
        # A tied object still declares its full SHAPE: the artifact carries a
        # full-length object under that name, and the tie only decides where the
        # bytes come from.  Its source keys are then narrowed to the ones the
        # checkpoint actually stores -- a tied embedding whose `lm_head.weight` was
        # never written to disk is a fact about the checkpoint, and the coverage
        # gate would otherwise (correctly) report that key as unconsumed.
        present = tuple(key for key in sources if source.contains(key))
        dropped = [key for key in sources if key not in present]
        note = ""
        if tie:
            note = "alias of %s (tie_word_embeddings)" % tie
            if dropped:
                note += "; not stored by this checkpoint: %s" % (", ".join(dropped),)
        objects.append(ObjectDecl(
            name=name,
            role=role,
            kind=TENSOR,
            shape=inventory.shape_of(geometry, name),
            format=FORMAT,
            layout=LAYOUT,
            sources=present,
            tie_to=tie,
            note=note,
        ))

    return Declaration(
        model_id=str(spec["model_id"]),
        weights_id=inventory.WEIGHTS_ID,
        family=str(spec.get("family", "")),
        spec=spec,
        objects=tuple(objects),
        recipes=recipe.RECIPES,
        ignored_source_prefixes=inventory.IGNORED_SOURCE_PREFIXES,
        resources={},
        work_items=WORK_ITEMS,
    )
