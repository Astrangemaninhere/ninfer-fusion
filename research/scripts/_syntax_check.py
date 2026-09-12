#!/usr/bin/env python3
"""Syntax-check one TU from compile_commands.json without touching build outputs."""
import json
import os
import shlex
import subprocess
import sys

BUILD = "/home/user/ninfer-fusion/build"
TARGET = sys.argv[1] if len(sys.argv) > 1 else "src/targets/qwen3_6_27b/impl/variant.cpp"

with open(os.path.join(BUILD, "compile_commands.json"), encoding="utf-8") as f:
    entries = json.load(f)

hit = None
for e in entries:
    f = e["file"].replace("\\", "/")
    if f.endswith(TARGET):
        hit = e
        break
if hit is None:
    print("no compile command for", TARGET)
    sys.exit(2)

cmd = shlex.split(hit["command"])
out = []
i = 0
while i < len(cmd):
    a = cmd[i]
    if a == "-c":
        out.append("-fsyntax-only")
        out.append(cmd[i + 1])  # keep the source file
        i += 2
        continue
    if a == "-o":
        i += 2  # drop the object output
        continue
    if a.endswith(".o") and not a.startswith("-"):
        i += 1
        continue
    out.append(a)
    i += 1

print("cwd:", hit["directory"])
print("checking:", hit["file"])
r = subprocess.run(out, cwd=hit["directory"], capture_output=True, text=True)
if r.returncode == 0:
    print("SYNTAX_OK")
else:
    print("SYNTAX_FAIL rc=%d" % r.returncode)
    sys.stderr.write(r.stdout[-8000:])
    sys.stderr.write(r.stderr[-8000:])
    sys.exit(1)
