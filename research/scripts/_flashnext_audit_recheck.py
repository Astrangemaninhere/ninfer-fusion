#!/usr/bin/env python3
"""Independent re-run of the FlashNext contract audit (not trusting B's own numbers).

Runs `flashnext_bindings.py --audit-gguf` on the real 296,475-name inventory and
prints the fields that constitute the claim "74,804/74,804 engines, missing 0".
"""
import json
import subprocess
import sys

ARCH = "/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit"
NAMES = "/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_flashnext_names.txt"

cmd = ["python3", "flashnext_bindings.py", "--audit-gguf", NAMES]
p = subprocess.run(cmd, cwd=ARCH, capture_output=True, text=True, timeout=1800)
print("exit code:", p.returncode)
if p.returncode not in (0, 1):
    print("stderr tail:", p.stderr[-600:])

try:
    d = json.loads(p.stdout)
except Exception as exc:
    print("could not parse stdout as JSON:", exc)
    print("stdout head:", p.stdout[:400])
    sys.exit(0)

for k in ("contract_entries", "matched_sources", "engines_covered", "missing_count",
          "unmatched_count", "complete", "quant_companions", "residue_count",
          "accounting_ok"):
    if k in d:
        print("%-20s %s" % (k, d[k]))

for k, v in d.items():
    if isinstance(v, list) and v:
        print("%-20s list[%d] first=%s" % (k, len(v), json.dumps(v[0], ensure_ascii=False)[:120]))
    elif isinstance(v, dict):
        print("%-20s dict keys=%s" % (k, list(v)[:8]))
