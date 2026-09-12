#!/bin/bash
# 判据：编辑 dflash2/final_norm 后逐列 draft 是否变化（判定 artifact 编辑是否生效）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
ART=/home/user/models/qwen3_8_27b_nvfp4_dflash2_refhead.ninfer
LOG=$J/dl/edit_effect.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 编辑是否生效 $(date '+%H:%M:%S') ==="

echo "--- 清零 dflash2/final_norm ---"
python3 - <<'PYEOF'
import json, struct, pathlib, hashlib
A = pathlib.Path("/home/user/models/qwen3_8_27b_nvfp4_dflash2_refhead.ninfer")
with open(A, "rb") as f:
    f.read(8); (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
payload = ((16 + jlen + 4095) // 4096) * 4096
o = {x["name"]: x for x in j["objects"]}["dflash2/final_norm"]
print("  final_norm:", {k: o.get(k) for k in ("format", "shape", "offset", "bytes")})
with open(A, "r+b") as f:
    f.seek(payload + o["offset"])
    f.write(b"\x00" * o["bytes"])
with open(A, "rb") as f:
    f.seek(payload + o["offset"]); back = f.read(o["bytes"])
print("  清零后回读全零?", all(b == 0 for b in back))
PYEOF

echo "--- 跑（带逐列探针） ---"
cd "$R/build" || exit 3
NINFER_DF2DBG=1 timeout 900 ./apps/ninfer "$ART" --prompt "$P" --max-new 24 --max-context 4096 \
  --no-thinking --greedy --spec dflash2 > $J/dl/edit_stdout.log 2>&1
echo "  rc=$?"
grep -E 'dflash2 acceptance rate|dflash2 accepted by pos' $J/dl/edit_stdout.log | sed 's/^/  /'

echo "--- 与基线（fx_spec.log，改动前）比对逐列 draft ---"
python3 - <<'PYEOF'
import re, pathlib
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
COL = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")
def drafts(p, limit=None):
    out = []
    for line in p.read_text(errors="replace").splitlines():
        m = COL.search(line)
        if m: out.append((int(m.group(2)), int(m.group(4))))
    return out[:limit] if limit else out
a = drafts(J / "fx_spec.log", 120)
b = drafts(J / "edit_stdout.log", 120)
n = min(len(a), len(b))
same = sum(1 for i in range(n) if a[i] == b[i])
print("  对比前 %d 列: 逐位相同 %d, 不同 %d" % (n, same, n - same))
print("  样例 基线:", a[:6])
print("  样例 编辑后:", b[:6])
print("  VERDICT:", "编辑生效（draft 变了）" if same < n else "编辑未生效（逐位相同）")
PYEOF
echo EDIT_EFFECT_DONE
