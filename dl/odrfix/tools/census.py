#!/usr/bin/env python3
"""Bounded census helper for the odrfix line. Reads only known build dirs.

Usage: census.py <subcommand> [args]
  macro            -> per build dir: entry count, TUs whose command carries the macro
  json-entry <cmd> -> first entry whose file basename matches <cmd>
"""
import json
import os
import sys

ROOT = "/home/user/ninfer-fusion"
BUILDS = ["build", "build-52", "build-61", "build-sim", "build-sm70"]
MACROS = ["NINFER_BUILD_CUDA_ARCHS", "NINFER_BUILD_CUDA_ARCHES", "NINFER_BUILD_CUDA_ARCH"]


def load(d):
    p = os.path.join(ROOT, d, "compile_commands.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def cmd_of(e):
    return e.get("command") or " ".join(e.get("arguments", []))


def rel(p):
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:
        return p


def macro():
    for d in BUILDS:
        cc = load(d)
        if cc is None:
            print("%-11s MISSING compile_commands.json" % d)
            continue
        print("%-11s entries=%d" % (d, len(cc)))
        for m in MACROS:
            hits = [rel(e["file"]) for e in cc if m in cmd_of(e)]
            print("    %-26s %d  %s" % (m, len(hits), hits))


def entry(pat):
    for d in BUILDS:
        cc = load(d)
        if cc is None:
            continue
        for e in cc:
            if pat in e["file"]:
                print("== %s in %s" % (d, rel(e["file"])))
                print("dir:", e.get("directory"))
                print(cmd_of(e))
                print()
                break


def includers(build, header):
    """No disk scan: derive each TU's depfile path from its own recorded -o."""
    cc = load(build)
    if cc is None:
        print("MISSING", build)
        return
    hits = []
    missing = 0
    for e in cc:
        words = cmd_of(e).split()
        obj = None
        for i, w in enumerate(words):
            if w == "-o" and i + 1 < len(words):
                obj = words[i + 1]
        if obj is None:
            continue
        dep = os.path.join(e.get("directory", ""), obj + ".d")
        if not os.path.exists(dep):
            missing += 1
            continue
        with open(dep, errors="replace") as fh:
            text = fh.read()
        if header in text:
            hits.append((rel(e["file"]), rel(dep) if dep.startswith(ROOT) else dep))
    print("%s: entries=%d depfiles_read=%d depfiles_absent=%d header=%s"
          % (build, len(cc), len(cc) - missing, missing, header))
    for e in cc:
        words = cmd_of(e).split()
        obj = None
        for i, w in enumerate(words):
            if w == "-o" and i + 1 < len(words):
                obj = words[i + 1]
        dep = os.path.join(e.get("directory", ""), (obj or "") + ".d")
        if obj is None or not os.path.exists(dep):
            print("   ABSENT-DEPFILE %s" % rel(e["file"]))
    for src, dep in sorted(hits):
        print("   %s" % src)
    print("   TOTAL TUs including %s: %d" % (header, len(hits)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "macro":
        macro()
    elif sys.argv[1] == "json-entry":
        entry(sys.argv[2])
    elif sys.argv[1] == "includers":
        includers(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
