#!/bin/bash
# 图模式下：--lm-head-draft（短名单头 0.34GB）vs 全词表头（1.27GB）对列边际成本 b 的影响
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/headcost.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd /home/user/ninfer-fusion/build || exit 3
pkill -9 -x ninfer 2>/dev/null; sleep 2
echo "=== 草稿头成本对照（图模式）起 $(date '+%H:%M:%S') ==="
printf "  %-14s %-6s %-8s %-9s %-9s %s\n" 臂 k AL tok/s ms/轮 轮/秒
run() {
  local lbl="$1"; shift
  timeout 900 ./apps/ninfer "$M" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --spec mtp "$@" > "$J/hc_$lbl.log" 2>&1
  local al sp
  al=$(grep -m1 'mtp acceptance length' "$J/hc_$lbl.log" | grep -oE '[0-9.]+')
  sp=$(grep -m1 'decode speed' "$J/hc_$lbl.log" | grep -oE '[0-9.]+')
  local ms rps
  ms=$(awk -v s="${sp:-0}" -v a="${al:-0}" 'BEGIN{if(s>0)printf "%.2f",1000.0*a/s; else print "?"}')
  rps=$(awk -v s="${sp:-0}" -v a="${al:-0}" 'BEGIN{if(a>0)printf "%.1f",s/a; else print "?"}')
  printf "  %-14s %-6s %-8s %-9s %-9s %s\n" "$lbl" "${1##*=}" "${al:-?}" "${sp:-?}" "$ms" "$rps"
}
for k in 1 3 5; do
  run "short_k$k" --draft-tokens "$k" --lm-head-draft
  run "full_k$k"  --draft-tokens "$k"
done
echo
echo "=== 拟合 round_time = a + b*k（看 b 是否因短头下降）==="
python3 - <<'PY'
import pathlib, re
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
def fit(prefix, ks):
    pts = []
    for k in ks:
        t = (J / f"hc_{prefix}_k{k}.log").read_text(errors="replace")
        al = re.search(r"mtp acceptance length\s+([0-9.]+)", t)
        sp = re.search(r"decode speed\s+([0-9.]+)", t)
        if al and sp and float(sp.group(1)) > 0:
            pts.append((k, 1000.0*float(al.group(1))/float(sp.group(1))))
    if len(pts) < 2: return None
    n = len(pts); sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0]**2 for p in pts); sxy = sum(p[0]*p[1] for p in pts)
    b = (n*sxy - sx*sy)/(n*sxx - sx*sx); a = (sy - b*sx)/n
    return a, b, pts
for prefix, name in (("short", "短名单头 --lm-head-draft"), ("full", "全词表头（默认）")):
    r = fit(prefix, (1, 3, 5))
    if r:
        a, b, pts = r
        print(f"  {name}: a={a:.2f} ms, b={b:.2f} ms/列   (点: {[(k, round(v,2)) for k,v in pts]})")
    else:
        print(f"  {name}: 数据不足")
PY
echo "=== 完成 $(date '+%H:%M:%S') ==="
echo HEADCOST_DONE
