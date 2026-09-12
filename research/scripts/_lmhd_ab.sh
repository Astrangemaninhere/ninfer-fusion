#!/bin/bash
# 提案头切换对照：--lm-head-draft（4bit draft_head 域） vs 默认（output 头域）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
ART=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
exec > >(tee -a "$J/dl/lmhd.log") 2>&1
cd "$R/build" || exit 3
echo "=== --lm-head-draft 对照 $(date '+%H:%M:%S') ==="
NINFER_DF2SEL=1 NINFER_DF2DBG=1 timeout 1200 ./apps/ninfer "$ART" --prompt "$P" \
  --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 --lm-head-draft \
  --print-token-ids > $J/dl/fx_lmhd.log 2>&1
echo "  rc=$?  df2cand=$(grep -c df2cand $J/dl/fx_lmhd.log)"
grep -E 'dflash2 acceptance rate|dflash2 accepted by pos|dflash2 acceptance length|decode speed' $J/dl/fx_lmhd.log | sed 's/^/  /'
echo
echo "=== 与默认提案头（fx_spec.log）对比候选集 ==="
python3 - <<'PY'
import re, pathlib
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
CAND = re.compile(r"\[df2cand\] s=(-?\d+) pred=(-?\d+) chosen=(-?\d+) k=(-?\d+) (.*)$")
def sets(p):
    out = []
    if not p.exists(): return out
    for line in p.read_text(errors="replace").splitlines():
        m = CAND.search(line)
        if m:
            out.append((int(m.group(1)), int(m.group(3)),
                        tuple(int(a) for a, _ in re.findall(r"(-?\d+):([-\d.]+)", m.group(5)))))
    return out
a = sets(J / "fx_spec.log"); b = sets(J / "fx_lmhd.log")
print("  默认步数 =", len(a), " lm-head-draft 步数 =", len(b))
n = min(len(a), len(b))
if n:
    same = sum(1 for i in range(n) if a[i][2] == b[i][2])
    print("  候选集逐位相同 = %d/%d (%.1f%%)" % (same, n, 100.0*same/n))
    for i in range(min(3, n)):
        print("   步%d 默认  cs=%s" % (i, a[i][2][:6]))
        print("   步%d lmhd  cs=%s" % (i, b[i][2][:6]))
PY
echo LMHD_DONE
