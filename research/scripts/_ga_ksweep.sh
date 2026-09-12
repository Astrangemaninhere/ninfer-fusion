#!/bin/bash
# 判别 verify 偏差的机制：扫 K。
#   K=1（W=2，只有 1 个后续列）就偏 => verify setup 参数错（positions/cache/valid columns）
#   仅 K 大时偏              => 跨列污染（后续 draft 列被前面的列看到 = 因果掩码问题）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/ga_ksweep.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== G-A K sweep $(date '+%F %H:%M:%S') ==="
cd $R/build || exit 3
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'

one() {  # tag extra...
  local tag=$1; shift
  local log=/home/user/ks_$tag.log
  timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
      sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > /tmp/ks_$tag.ids
  local pos; pos=$(grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  echo "  [$tag] n=$(wc -w < /tmp/ks_$tag.ids) pos=[$pos]"
}

one plain
for k in 1 2 3 5 7; do one k$k --spec dflash2 --draft-tokens $k; done

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
def ids(t):
    p = pathlib.Path(f"/tmp/ks_{t}.ids")
    return p.read_text().split() if p.exists() else []
base = ids("plain")
print()
print(f"plain n={len(base)}")
for k in (1, 2, 3, 5, 7):
    B = ids(f"k{k}")
    if not B:
        print(f"  K={k}: NO DATA"); continue
    n = min(len(base), len(B))
    i = next((x for x in range(n) if base[x] != B[x]), n)
    same = (i == n and len(base) == len(B))
    if same:
        print(f"  K={k}: IDENTICAL  <-- G-A ok")
    else:
        print(f"  K={k}: DIFFER at {i}/{n}  plain={base[max(0,i-2):i+2]} spec={B[max(0,i-2):i+2]}")
PY
echo KSWEEP_DONE
