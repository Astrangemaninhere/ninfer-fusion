# -*- coding: utf-8 -*-
"""kit.check — the ruler.  Run it on any candidate scaffold and get a verdict.

Two modes.

``--check`` (one candidate)
    Does the scaffold agree with its spec, is every `TODO(hand)` justified, how many
    places does hooking this model up really touch (against heritage's 9), which
    family hooks are zero-reference, and which of the three tiers does it reach.

``--reverse`` (an EXISTING target, the faithfulness proof)
    Take a target that already exists in the tree, generate its scaffold from its own
    spec, and diff:

      * the MECHANICAL part must agree with the real `impl/config.h`, value by value,
        and any disagreement is printed BY NAME (a disagreement is the paradigm
        leaking, not the target being wrong);
      * the HAND part must be ABSENT from the generated output -- if the generator
        emitted any name the real file declares as a hook, a per-layer table, or a
        knob, that is precisely what the prohibition exists to stop and it is
        reported BY NAME.

    This is the mode that makes "the paradigm is built in" a checkable claim rather
    than a belief: it is run against a model the generator was never told about.

    python3 -m tools.archkit.kit.check --reverse \
      --spec tools/archkit/specs/muse-glimmer-30b_spec.json \
      --id muse_glimmer_30b \
      --config src/targets/muse_glimmer_30b/impl/config.h
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "tools.archkit.kit"

from .refcount import count_references, declared_in_config, probed_by_runtime
from .scaffold import FORBIDDEN, generate

#: heritage's count, for the "how many places did this really touch" comparison
HERITAGE_PLACES = 9

#: the mechanical names the spec can settle.  Anything else a real config.h declares
#: is HAND by construction.
MECHANICAL_NAMES = frozenset({
    "hidden", "layers", "intermediate", "query_heads", "kv_heads", "head_dim",
    "vocab", "max_ctx", "rms_eps", "rope_theta", "query_size", "kv_size",
})

_CONST = re.compile(
    r"static\s+constexpr\s+(?:int|float|double|bool|std::uint32_t|std::int32_t)\s+"
    r"([a-z_][a-z0-9_]*)\s*=\s*([^;]+);")


def parse_constants(config_path: Path) -> dict[str, str]:
    from .refcount import struct_body, strip_comments

    text = strip_comments(config_path.read_text(encoding="utf-8", errors="replace"))
    body = struct_body(text, "TextConfig")
    out: dict[str, str] = {}
    for match in _CONST.finditer(body or text):
        out[match.group(1)] = match.group(2).strip()
    return out


def parse_generated(generated_path: Path) -> dict[str, str]:
    text = generated_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for match in re.finditer(r"inline constexpr \w+ ([a-z_][a-z0-9_]*)\s*=\s*([^;]+);",
                             text):
        out[match.group(1)] = match.group(2).strip()
    return out


def _spec_values(spec: Mapping[str, Any]) -> dict[str, str]:
    g = spec["geometry"]
    values = {key: str(int(g[key])) for key in
              ("hidden", "layers", "query_heads", "kv_heads", "head_dim", "vocab",
               "max_ctx", "intermediate") if key in g}
    values["query_size"] = "%d * %d" % (int(g["query_heads"]), int(g["head_dim"]))
    values["kv_size"] = "%d * %d" % (int(g["kv_heads"]), int(g["head_dim"]))
    return values


def check_candidate(scaffold_dir: Path, spec_path: Path, repo_root: Path,
                    runtime_rel: str) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    failures = 0
    print("== candidate %s" % scaffold_dir)
    print("   spec      %s" % spec_path)

    manifest = scaffold_dir / "check.json"
    if not manifest.is_file():
        print("   REFUSED: no check.json; this directory is not a scaffold")
        return 2
    verdict = json.loads(manifest.read_text(encoding="utf-8"))

    expected = [entry["path"] for entry in verdict["generated"]]
    missing = [p for p in expected if not (scaffold_dir / p).is_file()]
    print("   files     %d generated, %d missing" % (len(expected), len(missing)))
    for path in missing:
        print("      - MISSING %s" % path)
        failures += 1

    generated = parse_generated(scaffold_dir / "impl" / "spec_contract.h")
    want = _spec_values(spec)
    # Literal comparisons first, then the DERIVED ones by evaluation: the generated
    # header writes `query_size = query_heads * head_dim` and the spec settles the
    # product, so a text diff between the two is a diff between a derivation and its
    # value and would report a mismatch on a correct file (it did, first run).
    disagree: list[str] = []
    for name, value in want.items():
        if name not in generated:
            disagree.append("%s absent from the generated header" % name)
            continue
        if " " in value and "*" in value:
            continue                                    # derived; see the pass below
        if generated[name].replace(" ", "") != value.replace(" ", ""):
            disagree.append("%s: spec %r, generated %r"
                            % (name, value, generated[name]))
    env: dict[str, int] = {}
    for name, value in generated.items():
        if value.isdigit():
            env[name] = int(value)
    if env:
        for name in ("query_size", "kv_size"):
            try:
                got = eval(generated[name], {"__builtins__": {}}, dict(env))  # noqa: S307
            except Exception:                            # noqa: BLE001
                disagree.append("%s: generated expression %r does not evaluate over "
                                "the emitted constants" % (name, generated.get(name)))
                continue
            expected = int(want[name].split("*")[0]) * int(want[name].split("*")[1])
            if got != expected:
                disagree.append("%s: evaluated %d, spec %d" % (name, got, expected))
    print("   spec      %d/%d mechanical values present and equal (derived ones by "
          "evaluation over the emitted constants)"
          % (len(want) - len(disagree), len(want)))
    for line in disagree:
        print("      - MISMATCH %s" % line)
        failures += 1

    todo = scaffold_dir / "adapt_report.md"
    if todo.is_file():
        text = todo.read_text(encoding="utf-8")
        unjustified = [path for path, _ in FORBIDDEN if path not in text]
        print("   TODO      %d/%d forbidden targets named with a reason"
              % (len(FORBIDDEN) - len(unjustified), len(FORBIDDEN)))
        for path in unjustified:
            print("      - UNJUSTIFIED %s" % path)
            failures += 1

    points = verdict.get("insertion_points", [])
    found = sum(1 for p in points if p["anchor_line"] != "0")
    print("   places    %d tree-level insertion points, %d with an anchor found "
          "(heritage's floor for a new model is %d)"
          % (len(points), found, HERITAGE_PLACES))
    tier = "a"
    if (scaffold_dir / "impl" / "config.h.skeleton").is_file():
        tier = "a+scaffold"
    print("   tier      %s (the scaffold can never claim b or c: those need a "
          "converter that PRODUCED an artifact and an engine that READ it)"
          % tier)

    zero = verdict.get("zero_reference_accessors", [])
    print("   hooks     %d zero-reference accessors -- a new model does not get "
          "these for free:" % len(zero))
    for name in zero[:40]:
        print("      - %s" % name)
    if len(zero) > 40:
        print("      ... and %d more" % (len(zero) - 40))
    print("== candidate verdict: %s (%d failure(s))"
          % ("OK" if not failures else "FAIL", failures))
    return 1 if failures else 0


def reverse(spec_path: Path, target_id: str, config_path: Path, repo_root: Path,
            runtime_rel: str) -> int:
    """Generate a scaffold for an EXISTING target and diff it against the real file."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    real = parse_constants(config_path)
    if not real:
        print("REFUSED: %s declares no constants inside struct TextConfig" % config_path,
              file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="archkit-reverse-") as tmp:
        out = Path(tmp) / target_id
        verdict = generate(spec_path, target_id, out, repo_root, runtime_rel)
        generated = parse_generated(out / "impl" / "spec_contract.h")

    print("== reverse validation: %s against %s" % (target_id, config_path))
    print("   the generator was given the SPEC and never saw %s" % config_path.name)
    print("   real config.h declares %d constants inside struct TextConfig" % len(real))

    failures = 0
    agree: list[str] = []
    for name, value in sorted(real.items()):
        if name not in generated:
            continue
        if generated[name].replace(" ", "") == value.replace(" ", ""):
            agree.append(name)
        else:
            print("   MECHANICAL MISMATCH %s: real %r, generated %r"
                  % (name, value, generated[name]))
            failures += 1
    print("   MECHANICAL: %d name(s) compared, %d agree, %d disagree"
          % (len(agree) + failures, len(agree), failures))

    # Two-sided.  (1) What the SPEC settles must be emitted -- a name the spec
    # carries and the generator drops is a gap.  (2) What the real file treats as
    # HAND must NOT be emitted -- a name the generator invents is the adapt.py defect.
    # The first version of this check conflated them and reported spark's `rope_theta`
    # as a breach for the sin of declining to guess (spark's spec carries per-kind
    # theta, so declining is the correct behaviour and guessing 0.0F was the bug).
    emitted = set(generated)
    settled = set(_spec_values(spec))
    hand = sorted(name for name in real if name not in emitted)
    gaps = sorted(name for name in settled if name not in emitted)
    breaches = sorted(name for name in emitted
                      if name in real and name not in MECHANICAL_NAMES
                      and name not in settled)
    print("   HAND: %d name(s) in the real config.h that the generator did NOT emit"
          % len(hand))
    print("   GAPS: %d name(s) the spec settles that the generator did not emit%s"
          % (len(gaps), (": " + ", ".join(gaps)) if gaps else ""))
    for name in gaps:
        failures += 1
    for name in breaches:
        print("   PROHIBITION BREACH the generator emitted %s, which the real file "
              "treats as hand (%r)" % (name, real[name]))
        failures += 1
    print("   names emitted mechanically and confirmed against the real file: %d"
          % len([n for n in emitted if n in real]))
    print("   first 12 hand names (the generator must not name these): %s"
          % ", ".join(hand[:12]))
    print("== reverse verdict: %s (%d failure(s))"
          % ("OK" if not failures else "FAIL", failures))
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", type=Path, help="a candidate scaffold directory")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--id", default=None)
    parser.add_argument("--config", type=Path, help="the real impl/config.h (reverse)")
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument("--runtime", default="src/targets/qwen3_6/impl/runtime")
    args = parser.parse_args(argv)
    if args.reverse:
        if not (args.spec and args.config):
            print("REFUSED: --reverse needs --spec and --config", file=sys.stderr)
            return 2
        target_id = args.id or args.config.parent.parent.name
        return reverse(args.spec, target_id, args.config, args.repo_root, args.runtime)
    if args.check:
        if not args.spec:
            print("REFUSED: --check needs --spec", file=sys.stderr)
            return 2
        return check_candidate(args.check, args.spec, args.repo_root, args.runtime)
    print("REFUSED: pass --check <dir> or --reverse", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
