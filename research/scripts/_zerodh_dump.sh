#!/bin/bash
# 清零提案头 + 开候选 dump：若候选集与未清零版逐位相同 ⇒ 它不参与候选生成
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
ART=/home/user/models/qwen3_8_27b_nvfp4_dflash2_refhead.ninfer
exec > >(tee -a "$J/dl/zerodh_dump.log") 2>&1
cd "$R/build" || exit 3
echo "=== 清零头 + dump $(date '+%H:%M:%S') ==="
NINFER_DF2SEL=1 NINFER_DF2DBG=1 timeout 1200 ./apps/ninfer "$ART" --prompt "$P" \
  --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 --print-token-ids \
  > $J/dl/fx_zerodh_dump.log 2>&1
echo "  rc=$?  df2cand=$(grep -c df2cand $J/dl/fx_zerodh_dump.log)"
grep -E 'dflash2 acceptance rate|dflash2 accepted by pos' $J/dl/fx_zerodh_dump.log | sed 's/^/  /'

echo
echo "=== 与未清零版（fx_spec.log）逐位置对比候选集 ==="
python3 - <<'PY'
import re, pathlib
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
CAND = re.compile(r"\[df2cand\] s=(-?\d+) pred=(-?\d+) chosen=(-?\d+) k=(-?\d+) (.*)$")
def sets(p):
    out = []
    for line in p.read_text(errors="replace").splitlines():
        m = CAND.search(line)
        if m:
            out.append((int(m.group(1)), int(m.group(3)),
                        tuple(int(a) for a, _ in re.findall(r"(-?\d+):([-\d.]+)", m.group(5)))))
    return out
a = sets(J / "fx_spec.log")
b = sets(J / "fx_zerodh_dump.log")
print("  未清零步数 =", len(a), " 清零后步数 =", len(b))
n = min(len(a), len(b))
same_set = sum(1 for i in range(n) if a[i][2] == b[i][2])
same_chosen = sum(1 for i in range(n) if a[i][1] == b[i][1])
print("  候选集逐位相同 = %d/%d (%.1f%%)" % (same_set, n, 100.0*same_set/n if n else 0))
print("  chosen 相同     = %d/%d (%.1f%%)" % (same_chosen, n, 100.0*same_chosen/n if n else 0))
for i in range(min(3, n)):
    print("   步%d 未清零 cs=%s" % (i, a[i][2][:6]))
    print("   步%d 清零后 cs=%s" % (i, b[i][2][:6]))
PY
echo ZERODH_DUMP_DONE
