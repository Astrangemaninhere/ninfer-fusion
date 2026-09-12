#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 带标签的接受率（E2 字段）——修复后的真实数字 ==="
for f in /home/user/pb_dflash2.log /home/user/pb_mtp3.log; do
  echo "--- $(basename $f) ---"
  grep -E 'done .*speculative' "$f" 2>/dev/null | tail -2 | \
    grep -oE 'prompt=[0-9]+|gen=[0-9]+|finish=[a-z_]+|spec_drafted=[0-9]+|spec_accepted=[0-9]+|spec_accept_rate=[0-9.]+|spec_rounds=[0-9]+|spec_fallback_steps=[0-9]+|decode=[0-9.]+tok/s|speculative=[a-z0-9_]+ [0-9.]+tok/round' | tr '\n' ' '
  echo
done
echo
echo "=== 2) plain 对照（无 spec） ==="
grep -E 'done .*gen=' /home/user/pb_plain.log 2>/dev/null | tail -2 | \
  grep -oE 'prompt=[0-9]+|gen=[0-9]+|finish=[a-z_]+|decode=[0-9.]+tok/s' | tr '\n' ' '
echo; echo
echo "=== 3) 引擎口径换算（每轮可发 1+draft 个 token） ==="
python3 - <<'PY'
import re, pathlib
for name, drafts in (("dflash2", 7), ("mtp3", 3)):
    txt = pathlib.Path("/home/user/pb_%s.log" % name).read_text(errors="replace")
    for m in re.finditer(r"\[req (\d+)\] done .*?(spec_accept_rate=([0-9.]+))?.*?speculative=(\S+) ([\d.]+)tok/round", txt):
        pass
    rates = re.findall(r"spec_accept_rate=([0-9.]+)", txt)
    tpr = re.findall(r"speculative=\S+ ([\d.]+)tok/round", txt)
    print("  %-8s accept_rate=%s  tok/round=%s  (每轮上限 %d)" % (name, rates[-2:] or "无", tpr[-2:] or "无", drafts + 1))
PY
echo
echo "=== 4) exactness 分歧点：是平局还是真错？看分歧处两边 token 的后续是否收敛 ==="
python3 - <<'PY'
import pathlib
for pair in ("zh", "num"):
    a = pathlib.Path("/tmp/pb_text_plain_%s.txt" % pair)
    b = pathlib.Path("/tmp/pb_text_dflash2_%s.txt" % pair)
    if not (a.exists() and b.exists()):
        continue
    A, B = a.read_text(errors="replace"), b.read_text(errors="replace")
    n = min(len(A), len(B)); i = next((k for k in range(n) if A[k] != B[k]), n)
    print("[%s] 分歧在 %d；前缀长度 %d；两边长度 %d/%d" % (pair, i, i, len(A), len(B)))
    print("      plain 前 60: %r" % A[:60])
    print("      spec  前 60: %r" % B[:60])
PY
