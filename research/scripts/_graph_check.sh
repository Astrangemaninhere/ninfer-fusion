#!/bin/bash
# 判别：翻转是否来自 CUDA graph 捕获/回放路径。
#   关掉 graph 后 spec 流与 plain 逐位一致  => graph 捕获/回放的状态绑定有问题
#   仍然不一致                              => 与 graph 无关（回到算术/状态本身）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/graph_check.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== graph vs no-graph $(date '+%F %H:%M:%S') ==="
cd $R/build || exit 3
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'

one() {  # tag extra...
  local tag=$1; shift
  local log=/home/user/gc_$tag.log
  timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
      sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > /tmp/gc_$tag.ids
  local pos; pos=$(grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  echo "  [$tag] n=$(wc -w < /tmp/gc_$tag.ids) pos=[$pos]"
}

one plain
one dflash2_graph    --spec dflash2
one dflash2_nograph  --spec dflash2 --no-cuda-graph
one mtp3_graph       --spec mtp --draft-tokens 3
one mtp3_nograph     --spec mtp --draft-tokens 3 --no-cuda-graph

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
def ids(t):
    p = pathlib.Path(f"/tmp/gc_{t}.ids")
    return p.read_text().split() if p.exists() else []
ref = ids("plain")
print()
print(f"plain n={len(ref)}")
for t in ("dflash2_graph", "dflash2_nograph", "mtp3_graph", "mtp3_nograph"):
    B = ids(t)
    if not B:
        print(f"  {t:>17}: NO DATA"); continue
    n = min(len(ref), len(B))
    i = next((x for x in range(n) if ref[x] != B[x]), n)
    if i == n and len(ref) == len(B):
        print(f"  {t:>17}: IDENTICAL  <-- 与 plain 逐位一致")
    else:
        print(f"  {t:>17}: DIFFER at {i}/{n}  plain={ref[max(0,i-2):i+2]} spec={B[max(0,i-2):i+2]}")
PY
echo GRAPH_CHECK_DONE
