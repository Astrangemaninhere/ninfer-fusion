#!/bin/bash
echo "=== data/Qwen3.8-27B (possible HF target) ==="
ls -la /mnt/c/Users/User/Documents/ziqinzhang/data/Qwen3.8-27B/ 2>/dev/null | head -25
echo ""
echo "=== search whole ziqinzhang for HF config.json with Qwen3.8/27B + num_hidden_layers >= 60 ==="
python3 - <<'PYEOF'
import os, json, fnmatch
root = "/mnt/c/Users/User/Documents/ziqinzhang"
found = []
for dp, dn, fn in os.walk(root):
    # skip heavy/irrelevant trees
    if any(s in dp for s in ("site-packages", ".git", "node_modules", "build/", "\\build")):
        continue
    if "config.json" in fn and dp.count(os.sep) - root.count(os.sep) <= 4:
        p = os.path.join(dp, "config.json")
        try:
            d = json.load(open(p, encoding="utf-8", errors="replace"))
        except Exception:
            continue
        nhl = d.get("num_hidden_layers")
        arch = d.get("architectures")
        if nhl and nhl >= 40:
            found.append((p, nhl, arch))
for p, n, a in sorted(found):
    print("  nhl=%-4s arch=%-40s %s" % (n, a, p))
if not found:
    print("  (none found)")
PYEOF
echo ""
echo "=== sanity: checkpoints/data dirs with HF-looking model.safetensors ==="
find /mnt/c/Users/User/Documents/ziqinzhang -maxdepth 4 -name '*.safetensors' -size +500M 2>/dev/null | head -20
