# -*- coding: utf-8 -*-
"""kit.refcount — which declared accessors does the shared runtime actually READ?

The question this answers is the one heritage reduced "what is per-model different"
to (dl/heritage/REPORT.md 判据 3.2 B): the thing that is genuinely per-model is not
the LIST of hooks a target declares, it is the set of those hooks that has ZERO
references in the shared runtime.  Everything with a nonzero count is already
general -- a new model declares it and gets it for free -- while a zero-count
accessor is a DECLARATION THE ENGINE NEVER CONSUMES, which is either dead weight or
a family-side change waiting to happen.

The denominator is stated, not assumed:

  accessors = (every `static constexpr` member and function DECLARED in the
               target's `impl/config.h` inside `struct TextConfig`)
            U (every `TC::<name>` the shared runtime PROBES with a `requires`
               expression -- the detection idioms in text_context.h)

Counting is a word-boundary textual count over the shared runtime directory, and
every file that contributes is printed with its own count, so the reading can be
audited rather than believed.

    python3 -m tools.archkit.kit.refcount --target src/targets/spark_x2_5_4b
    python3 -m tools.archkit.kit.refcount --target src/targets/muse_glimmer_30b \
                                         --target src/targets/spark_x2_5_4b
    python3 -m tools.archkit.kit.refcount --target src/targets/spark_x2_5_4b --json

No file is written anywhere and no file is modified: this is a read-only ruler.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "tools.archkit.kit"

#: the shared text runtime every text target in this tree reaches
DEFAULT_RUNTIME = "src/targets/qwen3_6/impl/runtime"
DEFAULT_CONFIG_NAMES = ("impl/config.h",)

_CONSTANT = re.compile(r"static\s+constexpr\s+[\w:<>\*&\s]+?\s+([a-z_][a-z0-9_]*)\s*[={(]")
_FUNCTION = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\([^;{)]*\)\s*(?:const\s*)?\{")
_REQUIRES_PROBE = re.compile(r"TC::([a-z_][a-z0-9_]*)")


def strip_comments(text: str) -> str:
    """Drop // and /* */ comments, keeping line count (so line numbers survive)."""
    out: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        if text.startswith("//", i):
            j = text.find("\n", i)
            if j < 0:
                break
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                break
            chunk = text[i:j + 2]
            out.append("".join("\n" if ch == "\n" else " " for ch in chunk))
            i = j + 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def struct_body(text: str, name: str) -> str:
    """The body of `struct <name> { ... }`, by brace matching."""
    match = re.search(r"\bstruct\s+%s\s*\{" % re.escape(name), text)
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return text[start:]


def declared_in_config(path: Path) -> dict[str, int]:
    text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
    body = struct_body(text, "TextConfig")
    if not body:
        return {}
    names: dict[str, int] = {}
    for kind, pattern in (("constant", _CONSTANT), ("function", _FUNCTION)):
        for match in pattern.finditer(body):
            name = match.group(1)
            line = body.count("\n", 0, match.start()) + 1
            names.setdefault(name, line)
    # the struct's own scaffolding is not an accessor
    for keyword in ("static_assert", "if", "for", "while", "return", "requires",
                    "sizeof", "decltype", "operator"):
        names.pop(keyword, None)
    return names


def probed_by_runtime(runtime: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted(runtime.rglob("*")):
        if path.suffix not in (".h", ".cpp", ".cuh"):
            continue
        text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        names.update(_REQUIRES_PROBE.findall(text))
    return names


def count_references(runtime: Path, names: Iterable[str]) -> dict[str, dict]:
    patterns = {name: re.compile(r"\b%s\b" % re.escape(name)) for name in names}
    counts: dict[str, dict] = {name: {"total": 0, "files": {}} for name in names}
    for path in sorted(runtime.rglob("*")):
        if path.suffix not in (".h", ".cpp", ".cuh"):
            continue
        text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        for name, pattern in patterns.items():
            hits = len(pattern.findall(text))
            if hits:
                counts[name]["total"] += hits
                counts[name]["files"][str(path.relative_to(runtime))] = hits
    return counts


def run(target_dir: Path, repo_root: Path, runtime_rel: str, as_json: bool) -> int:
    runtime = repo_root / runtime_rel
    if not runtime.is_dir():
        print("REFUSED: shared runtime directory %s does not exist" % runtime,
              file=sys.stderr)
        return 2
    config_path = target_dir / "impl" / "config.h"
    if not config_path.is_file():
        print("REFUSED: target %s has no impl/config.h; there is nothing declared to "
              "count" % target_dir, file=sys.stderr)
        return 2

    declared = declared_in_config(config_path)
    probed = probed_by_runtime(runtime)
    names = sorted(set(declared) | probed)
    counts = count_references(runtime, names)

    zero = [n for n in names if counts[n]["total"] == 0]
    result = {
        "target": str(target_dir),
        "config": str(config_path),
        "runtime": str(runtime.relative_to(repo_root)),
        "declared_in_config": len(declared),
        "probed_by_runtime": sorted(probed),
        "accessors": {n: counts[n]["total"] for n in names},
        "zero_reference": zero,
        "files": {n: counts[n]["files"] for n in names if counts[n]["files"]},
    }
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print("== target      %s" % target_dir)
    print("   config      %s (declares %d names inside struct TextConfig)"
          % (config_path, len(declared)))
    print("   runtime     %s/%s" % (repo_root, runtime_rel))
    print("   denominator %d accessors = %d declared in config.h U %d probed by the "
          "runtime with `requires`" % (len(names), len(declared), len(probed)))
    print("   %-34s %7s  %s" % ("accessor", "refs", "where"))
    for name in names:
        where = sorted(counts[name]["files"].items(), key=lambda kv: -kv[1])[:3]
        mark = "  <== ZERO-REFERENCE" if counts[name]["total"] == 0 else ""
        print("   %-34s %7d  %s%s"
              % (name, counts[name]["total"],
                 ", ".join("%s:%d" % kv for kv in where) or "(nowhere)", mark))
    print("== zero-reference accessors: %d/%d -- %s"
          % (len(zero), len(names), ", ".join(zero) or "(none)"))
    print("   these are the ones a NEW model does not get for free: a declaration the")
    print("   shared runtime never reads is either dead weight or a family-side change.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", action="append", required=True,
                        help="target directory, relative to the repo root or absolute "
                             "(repeatable)")
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME,
                        help="shared runtime directory, relative to --repo-root")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    status = 0
    for index, target in enumerate(args.target):
        if index:
            print()
        path = Path(target)
        if not path.is_absolute():
            path = args.repo_root / path
        status |= run(path, args.repo_root, args.runtime, args.json)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
