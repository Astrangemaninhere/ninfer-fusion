#!/bin/bash
# 短头 vs 全词表头（干净版）：开跑前断言无 ninfer、无活跃测量服务；逐臂完整性校验；结束自清
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/headcost3.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd /home/user/ninfer-fusion/build || exit 3

# --- 守卫 ---
pkill -9 -x ninfer 2>/dev/null; sleep 3
n=$(pgrep -x ninfer | wc -l)
for u in hc hc2 hc3 pm cc m0 s2 sv bh bh2 bh3 bh4 kv3 hfdl dlr dlr2 dll dll2 rpa rb; do
  systemctl --user stop "$u.service" 2>/dev/null
done
echo "=== 前置守卫：ninfer 实例=$n（应为 0），服务已停 ==="
[ "$n" != "0" ] && { echo "ABORT: 仍有 ninfer 在跑"; exit 9; }

echo "=== 短头 vs 全词表头 起 $(date '+%H:%M:%S') ==="
printf "  %-14s %-4s %-7s %-9s %-9s %-7s %s\n" 臂 k AL tok/s ms轮 轮秒 完整
one() {
  local lbl="$1" k="$2"; shift 2
  local m=$(pgrep -x ninfer | wc -l)
  [ "$m" != "0" ] && { echo "  跳过 $lbl：有残留 ninfer"; return; }
  timeout 900 ./apps/ninfer "$M" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --spec mtp --draft-tokens "$k" "$@" > "$J/h3_$lbl.log" 2>&1
  local al sp rd ok ms rps
  al=$(grep -m1 'mtp acceptance length' "$J/h3_$lbl.log" | grep -oE '[0-9.]+')
  sp=$(grep -m1 'decode speed' "$J/h3_$lbl.log" | grep -oE '[0-9.]+')
  rd=$(grep -m1 'mtp rounds' "$J/h3_$lbl.log" | grep -oE '[0-9]+')
  ok=$([ -n "${al:-}" ] && [ -n "${sp:-}" ] && [ -n "${rd:-}" ] && echo Y || echo N)
  ms=$(awk -v s="${sp:-0}" -v a="${al:-0}" 'BEGIN{if(s>0)printf "%.2f",1000.0*a/s; else print "-"}')
  rps=$(awk -v s="${sp:-0}" -v a="${al:-0}" 'BEGIN{if(a>0)printf "%.1f",s/a; else print "-"}')
  printf "  %-14s %-4s %-7s %-9s %-9s %-7s %s\n" "$lbl" "$k" "${al:-?}" "${sp:-?}" "$ms" "$rps" "$ok"
}
for k in 1 3 5; do
  one "short_k$k" "$k" --lm-head-draft
  one "full_k$k"  "$k"
done

echo
python3 - <<'PY'
import pathlib, re
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
def fit(prefix):
    pts = []
    for k in (1, 3, 5):
        p = J / f"h3_{prefix}_k{k}.log"
        if not p.exists(): continue
        t = p.read_text(errors="replace")
        al = re.search(r"mtp acceptance length\s+([0-9.]+)", t)
        sp = re.search(r"decode speed\s+([0-9.]+)", t)
        rd = re.search(r"mtp rounds\s+([0-9]+)", t)
        if al and sp and rd and float(sp.group(1)) > 0:
            pts.append((k, 1000.0*float(al.group(1))/float(sp.group(1))))
    if len(pts) < 2: return None
    n = len(pts); sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0]**2 for p in pts); sxy = sum(p[0]*p[1] for p in pts)
    b = (n*sxy - sx*sy)/(n*sxx - sx*sx); a = (sy - b*sx)/n
    return a, b, pts
for prefix, name in (("short", "短名单头 --lm-head-draft"), ("full", "全词表头（默认）")):
    r = fit(prefix)
    print("  " + name + ": " + (f"a={r[0]:.2f} ms, b={r[1]:.2f} ms/列   点={[(k,round(v,2)) for k,v in r[2]]}" if r else "数据不足"))
print("  参考：可达读带宽 1813 GB/s ⇒ 权重下界 11.70 ms/轮；全词表 FP8 头 1.27GB≈0.70ms，短头 Q4 0.34GB≈0.19ms")
PY
pkill -9 -x ninfer 2>/dev/null
echo "=== 完成 $(date '+%H:%M:%S') ==="
echo HEADCOST3_DONE
