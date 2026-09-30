# -*- coding: utf-8 -*-
"""kit.test_kit — the negative battery: prove the gate REFUSES, by name.

A green conversion proves nothing about a battery that is inert.  Every case below
is a deliberate defect, and each one asserts three things at once:

  1. the run FAILS (an exit status, not a warning),
  2. it fails with a `KitRefusal` whose message NAMES the missing file or the gate,
  3. nothing was produced -- a refusal that still leaves an artifact behind is the
     worst of both worlds, because the next reader finds a file and believes it.

Cases:

  ``missing-config``        delete config.json            -> "missing config.json"
  ``missing-index``         delete the index              -> "missing ...index.json"
  ``inject-drop-object``    shared driver, one object dropped   -> "source-coverage"
  ``inject-wrong-shape``    shared driver, one shape perturbed  -> "geometry"
  ``inject-truncate``       shared driver, one byte short       -> "container-writer"
  ``inject-empty-geometry`` shared driver, spec geometry emptied-> "geometry"

The last four are injections INTO THE SHARED DRIVER (`kit.driver.INJECTIONS`), which
is the point: the defect is not in a model's declaration but in the machinery every
model depends on, so a model-side test could never have found it.

    python3 -m tools.archkit.kit.test_kit
"""

from __future__ import annotations

import functools
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "tools.archkit.kit"

from . import driver
from .contract import KitRefusal
from .demo import write_synthetic_checkpoint
from tools.archkit.arch_spec import hf_to_spec
from tools.convert.minicpm5_1b.declaration import build_declaration


def _prepare(root: Path) -> Path:
    checkpoint = write_synthetic_checkpoint(root / "MiniCPM5-1B")
    hf_to_spec(str(checkpoint / "config.json"), "minicpm5-1b", family="qwen3_6",
               out=str(root / "minicpm5_1b_spec.json"))
    return checkpoint


def _run(root: Path, checkpoint: Path, out: Path, inject: str, quiet: bool):
    import contextlib
    import io as _io

    builder = functools.partial(build_declaration, spec_dir=root)
    sink = _io.StringIO() if quiet else None
    try:
        with contextlib.redirect_stdout(sink if sink is not None else sys.stdout):
            report = driver.run(builder, checkpoint, out, inject=inject, spec_dir=root)
        return 0, report
    except KitRefusal as exc:
        return 2, str(exc)
    except Exception as exc:                                    # noqa: BLE001
        return 1, "%s: %s" % (type(exc).__name__, exc)


CASES = (
    ("missing-config", "missing config.json", "delete"),
    ("missing-index", "missing model.safetensors.index.json", "delete"),
    ("inject-drop-object", "source-coverage", "inject"),
    ("inject-wrong-shape", "geometry", "inject"),
    ("inject-truncate", "container-writer", "inject"),
    ("inject-empty-geometry", "geometry", "inject"),
)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true",
                        help="let the driver's own output through (evidence, noisy)")
    args = parser.parse_args(argv)
    quiet = not args.verbose
    root = Path(tempfile.mkdtemp(prefix="archkit-test-"))
    failures = 0
    try:
        print("%-24s %-6s %-8s %-9s %s"
              % ("case", "rc", "artifact", "expected", "message (first clause)"))
        for name, expected, mode in CASES:
            work = root / name
            work.mkdir(parents=True, exist_ok=True)
            checkpoint = _prepare(work)
            out = work / "out.ninfer"
            if mode == "delete":
                if name == "missing-config":
                    (checkpoint / "config.json").unlink()
                else:
                    (checkpoint / "model.safetensors.index.json").unlink()
                rc, message = _run(work, checkpoint, out, "", quiet)
            else:
                injection = name.replace("inject-", "").replace("empty-geometry",
                                                                "empty-geometry")
                injection = {"drop-object": "drop-object",
                             "wrong-shape": "wrong-shape",
                             "truncate": "truncate-payload",
                             "empty-geometry": "empty-geometry"}[injection]
                rc, message = _run(work, checkpoint, out, injection, quiet)
            if not isinstance(message, str):
                message = "UNEXPECTED SUCCESS: %s" % message
                rc = 0
            produced = "yes" if out.exists() else "no"
            named = expected.lower() in message.lower()
            ok = (rc == 2) and named and produced == "no"
            if not ok:
                failures += 1
            print("%-24s %-6s %-8s %-9s %s"
                  % (name, rc, produced, "OK" if ok else "FAIL",
                     message.splitlines()[0][:110] if message else ""))
            if not ok:
                print("      expected %r in the message; rc=%s produced=%s"
                      % (expected, rc, produced))
        print("== %d/%d cases behave as declared" % (len(CASES) - failures, len(CASES)))
        return 1 if failures else 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
