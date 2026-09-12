#!/bin/bash
# 图模式下的列边际成本（决定"树"的性价比与 220 tok/s 目标是否可达）
# 模型：round_time(k) = a + b*k（列数 = k+1）⇒ 由 k=1..5 拟合 a、b
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/colcost.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd /home/user/ninfer-fusion/build || exit 3
pkill -9 -x ninfer 2>/dev/null; sleep 2
echo "=== 图模式列边际成本 起 $(date '+%H:%M:%S') ==="
printf "  %-6s %-8s %-8s %-11s %-10s %s\n" k AL tok/s 轮数 ms/轮 列数
for k in 1 2 3 4 5; do
  timeout 900 ./apps/ninfer "$M" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --spec mtp --draft-tokens "$k" --lm-head-draft \
    > "$J/cc_k$k.log" 2>&1
  al=$(grep -m1 'mtp acceptance length' "$J/cc_k$k.log" | grep -oE '[0-9.]+')
  sp=$(grep -m1 'decode speed' "$J/cc_k$k.log" | grep -oE '[0-9.]+')
  rd=$(grep -m1 'mtp rounds' "$J/cc_k$k.log" | grep -oE '[0-9]+')
  ms=$(awk -v s="${sp:-0}" -v a="${al:-0}" 'BEGIN{if(s>0)printf "%.2f", 1000.0*a/s; else print "?"}')
  rps=$(awk -v s="${sp:-0}" -v a="${al:-0}" 'BEGIN{if(a>0)printf "%.1f", s/a; else print "?"}')
  printf "  %-6s %-8s %-8s %-11s %-10s %s  (轮/秒=%s)\n" "$k" "${al:-?}" "${sp:-?}" "${rd:-?}" "$ms" "$((k+1))" "$rps"
done
echo
echo "=== 由 k=1..5 拟合 round_time = a + b*k（列数=k+1）==="
python3 - <<'PY'
import pathlib, re
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
pts = []
for k in range(1, 6):
    t = (J / f"cc_k{k}.log").read_text(errors="replace")
    al = re.search(r"mtp acceptance length\s+([0-9.]+)", t)
    sp = re.search(r"decode speed\s+([0-9.]+)", t)
    if al and sp and float(sp.group(1)) > 0:
        pts.append((k, 1000.0 * float(al.group(1)) / float(sp.group(1))))
if len(pts) >= 2:
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0]**2 for p in pts); sxy = sum(p[0]*p[1] for p in pts)
    b = (n*sxy - sx*sy) / (n*sxx - sx*sx)
    a = (sy - b*sx) / n
    print(f"  round_time(ms) ≈ {a:.2f} + {b:.2f}*k   （k>=1，列数=k+1）")
    print(f"  ⇒ 每多一列的成本 b = {b:.2f} ms；固定部分 a = {a:.2f} ms")
    print(f"  ⇒ 权重下界 = 19.73 GiB / 1.79 TB/s = {1000*19.73*1.073741824/1790:.2f} ms")
    for k in (3, 7, 15):
        print(f"     预测 k={k}（列数 {k+1}）: {a+b*k:.2f} ms ⇒ 轮/秒 = {1000/(a+b*k):.1f}")
else:
    print("  数据不足")
PY
echo "=== 完成 $(date '+%H:%M:%S') ==="
echo COLCOST_DONE
