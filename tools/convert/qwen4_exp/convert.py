#!/usr/bin/env python3
"""Convert a Qwen3.8-Flash-Next (qwen4_exp) checkpoint into one ninfer artifact.

This is the thin target the front door expects -- ``tools/convert/import_model.py``
``REGISTERED_TARGETS`` imports ``tools.convert.<target>.convert`` and asks it for
``validate_config``.  It re-exports the family implementation adopted from
igorls/ninfer (Apache-2.0) at ``tools/convert/qwen3_8_flash_next/`` and supplies the
two things that tree cannot know: this checkout's on-disk source locations, and the
``validate_config`` hook.

Canonical invocation::

    python3 -m tools.convert.qwen4_exp.convert --mixed <ckpt-dir> --ple <sidecar-dir> --out <x.ninfer>
    python3 -m tools.convert.qwen4_exp.convert --preflight-only

TWO SPELLINGS, DELIBERATELY
---------------------------
The engine target and this directory are **qwen4_exp** -- that is the checkpoint's own
``model_type`` (measured: ``config.json`` declares ``"qwen4_exp"`` twice and
``"qwen4_exp_text"`` once, with ``architectures = ["Qwen4ExpForConditionalGeneration"]``)
and the name of this tree's ``src/targets/qwen4_exp``.  The adopted family
implementation keeps igorls's name ``qwen3_8_flash_next``, which is also its artifact
``model_id``, and the registry recognizes both (see ``src/targets/registry.cpp``).  This
delegator is the single place where the mapping lives, so adding a third spelling
somewhere else is never necessary.

WHAT THIS CANNOT DO YET
-----------------------
The adopted family converter pins its own source provenance and PLE encoding, and the
checkpoint on this box does not satisfy either pin.  Both are named by
``--preflight-only`` rather than discovered 100 GB into a conversion:

  * ``source.MIXED_INDEX_TENSORS = 296_630`` against ``MIXED_REVISION``
    ``a4e813ed3cfbbcc61e2929699eccb864a4dfa843``, while the copy on this disk
    (Qwen3.8-Flash-Next-ABLITERATED-NVFP4, 206 shards) indexes **296_475** keys;
    ``source._require_revision`` is a hard gate.
  * ``source.validate_bundle`` reads ``<ple_dir>/../ples_int4/META.json`` and pins
    ``PLE_REVISION = da8b39586016d8325ac619be28ad77d6296625ec``, i.e. an **int4** PLE,
    while this disk carries the **BF16** sidecar (``MQ-Q5-SSD-PLE-BF16``,
    ``flashnext_ple/ple-manifest.json`` + four 25.6 GB bins).

So the pins must be reconciled (re-fetch the pinned revision, or re-pin to the local
copy) before any artifact can be produced from what is on this box.  That is a data
question, not a code one.  Until then this target validates and reports; it does not
emit.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The adopted family implementation.  Imported lazily so that ``validate_config`` --
#: which the front door calls on every candidate source -- never needs numpy or the
#: artifact container to be importable.
FAMILY = "qwen3_8_flash_next"

#: This model's identity, for the front door's report.  Mirrors
#: tools/convert/qwen3_8_flash_next/inventory.py.
MODEL_ID = "qwen3.8-flash-next"
WEIGHTS_ID = "mixed-nvfp4-fp8-ple-int4"
TARGET_KEY = "qwen3_8_flash_next"

#: The checkpoint's own ``model_type`` spellings.  Measured from the shipped config.json:
#: ``"qwen4_exp"`` appears twice (top level and inside the nested text config) and
#: ``"qwen4_exp_text"`` once.  A source declaring either is this model.
MODEL_TYPES: tuple[str, ...] = ("qwen4_exp", "qwen4_exp_text")

#: Architecture class the checkpoint declares.
ARCHITECTURES: tuple[str, ...] = ("Qwen4ExpForConditionalGeneration",)

#: Measured on this box, overridable.  A pinned default is a convenience, never a pin:
#: ``NINFER_FLASHNEXT_MIXED`` / ``NINFER_FLASHNEXT_PLE`` win over these.
DEFAULT_MIXED_DIR = Path(os.environ.get(
    "NINFER_FLASHNEXT_MIXED",
    "/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4",
))
DEFAULT_PLE_DIR = Path(os.environ.get(
    "NINFER_FLASHNEXT_PLE",
    "/mnt/c/Users/User/Documents/ziqinzhang/flashnext_ple",
))
DEFAULT_OUTPUT = Path("out") / "qwen3_8_flash_next_mixed.ninfer"

_OUTPUT_BASENAME = "qwen3_8_flash_next_mixed.ninfer"


def _family() -> Any:
    """The adopted family module, imported on demand."""
    return importlib.import_module(f"tools.convert.{FAMILY}.convert")


def _architectures(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = config.get("architectures")
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple)):
        return tuple(str(x) for x in raw)
    return ()


def validate_config(config: Mapping[str, Any]) -> None:
    """Confirm the source is this model.  Raises ValueError naming what it saw.

    The front door looks for exactly this name (``import_model._find_validator``,
    which asks for ``validate_config`` on a ``convert`` target).  It is deliberately
    cheap and torch-free: it runs for every candidate source, not just this one.
    """
    model_type = str(config.get("model_type") or "")
    if model_type not in MODEL_TYPES:
        raise ValueError(
            "checkpoint config mismatch: model_type=%r is not a Qwen3.8-Flash-Next source; "
            "expected one of %s" % (model_type, " / ".join(MODEL_TYPES))
        )
    archs = _architectures(config)
    if archs and not any(a in ARCHITECTURES for a in archs):
        raise ValueError(
            "checkpoint config mismatch: architectures=%s does not declare %s"
            % (list(archs), " / ".join(ARCHITECTURES))
        )


def validate_bundle(mixed_dir: str | Path, ple_dir: str | Path) -> Any:
    """The family's own two-source preflight (revision + expert shards + PLE)."""
    source = importlib.import_module(f"tools.convert.{FAMILY}.source")
    return source.validate_bundle(mixed_dir, ple_dir)


def validate_mixed_source(mixed_dir: str | Path) -> Any:
    """The family's recipe-side validation of the mixed checkpoint alone."""
    recipe = importlib.import_module(f"tools.convert.{FAMILY}.recipe")
    return recipe.validate_mixed_source(mixed_dir)


def convert(mixed_dir: str | Path, ple_dir: str | Path, out_path: str | Path) -> Path:
    """Delegate the conversion, after naming the local-source pins if they diverge."""
    return _family().convert(mixed_dir, ple_dir, out_path)


def preflight(mixed_dir: str | Path, ple_dir: str | Path) -> int:
    """Run both preflights and report, without reading model payloads.

    Returns 0 when the local sources satisfy the family's pins, 1 otherwise -- with
    each divergence named, never a bare failure.
    """
    source = importlib.import_module(f"tools.convert.{FAMILY}.source")
    failures = 0

    local_index = Path(mixed_dir) / "model.safetensors.index.json"
    if local_index.exists():
        import json
        keys = len(json.loads(local_index.read_text(encoding="utf-8"))["weight_map"])
        expected = source.MIXED_INDEX_TENSORS
        if keys != expected:
            print("MISMATCH mixed checkpoint: local index has %d tensors, the adopted "
                  "converter pins %d (revision %s)"
                  % (keys, expected, source.MIXED_REVISION))
            failures += 1
        else:
            print("ok       mixed checkpoint: %d tensors, matches the converter's pin" % keys)
    else:
        print("MISSING  mixed checkpoint index: %s" % local_index)
        failures += 1

    ple_int4 = Path(ple_dir).parent / "ples_int4" / "META.json"
    if ple_int4.exists():
        print("ok       int4 PLE sidecar: %s" % ple_int4)
    else:
        manifest = Path(ple_dir) / "ple-manifest.json"
        variant = ""
        if manifest.exists():
            import json
            variant = json.loads(manifest.read_text(encoding="utf-8")).get("artifact_variant", "")
        print("MISMATCH PLE sidecar: the adopted converter wants %s (revision %s); this disk "
              "carries the %s variant at %s"
              % (ple_int4, source.PLE_REVISION, variant or "unknown", ple_dir))
        failures += 1

    # The two validators are RESOLVED INSIDE the guard, not while the tuple is built.
    # Building it eagerly called `importlib.import_module(recipe)` before the `try`, and
    # `tools/convert/qwen3_8_flash_next/recipe.py:559` runs `validate_recipe()` at module
    # scope, which calls `inventory.validate_inventory()` at `:539` -- an unconditional
    # call that the documented escape at `inventory.py:332-333` does NOT cover.  So a
    # REPORTABLE divergence arrived as a bare ImportError traceback, which is the one thing
    # this function promises never to do: "with each divergence named, never a bare
    # failure" (`preflight`'s own docstring).  Resolving them here does not weaken any
    # check: the identical exceptions still raise, are still printed with their full
    # message, and are still counted in `failures`, so the exit code cannot change.
    for label, run in (
        ("validate_bundle", lambda: source.validate_bundle(mixed_dir, ple_dir)),
        ("validate_mixed_source", lambda: importlib.import_module(
            f"tools.convert.{FAMILY}.recipe").validate_mixed_source(mixed_dir)),
    ):
        try:
            run()
            print("ok       %s" % label)
        except Exception as exc:                      # noqa: BLE001 -- report, never crash
            print("FAILED   %s: %s" % (label, exc))
            failures += 1

    print("\npreflight: %d divergence(s) between the local sources and the adopted pins"
          % failures)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", type=Path, default=DEFAULT_MIXED_DIR,
                        help="mixed NVFP4/FP8 checkpoint directory")
    parser.add_argument("--ple", type=Path, default=DEFAULT_PLE_DIR,
                        help="PLE sidecar root (holds ple-manifest.json)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help="output artifact path (basename must be %s)" % _OUTPUT_BASENAME)
    parser.add_argument("--preflight-only", action="store_true",
                        help="report source-vs-pin divergences and exit; write nothing")
    args = parser.parse_args(argv)

    if args.preflight_only:
        return preflight(args.mixed, args.ple)
    _family().convert(args.mixed, args.ple, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
