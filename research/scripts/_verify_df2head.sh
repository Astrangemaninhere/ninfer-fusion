#!/bin/bash
# 验收：dflash2 接上专用草稿头之后
#   (1) 位置剖面是否从 [8,0,0,0,0,0,0] 变成链式（p1+ 非零）
#   (2) G-A 闸门：--lm-head-draft 不得改变生成的 token 流（接受只会 licensed
#       与 target argmax 相等的 token ⇒ 输出必须逐位相同）
#   (3) 探针数据留给 blame 对齐分析
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/verify_df2head.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== verify df2head $(date '+%F %H:%M:%S') ==="
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
cd $R/build || exit 4
ls -l --time-style=+%H:%M ./apps/ninfer

run() {  # tag extra...
  local tag=$1; shift
  local log=/home/user/vdf2_$tag.log
  timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  local rc=$?
  local pos; pos=$(grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  local acc; acc=$(grep -oE 'spec_accept_rate=[0-9.]+' "$log" | tail -1)
  local ac;  ac=$(grep -oE 'spec_accepted=[0-9]+' "$log" | tail -1)
  local dr;  dr=$(grep -oE 'spec_drafted=[0-9]+' "$log" | tail -1)
  local tpr; tpr=$(grep -oE '[0-9.]+tok/round' "$log" | tail -1)
  echo "  [$tag] rc=$rc $ac $dr $acc $tpr pos=[$pos]"
  grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
      sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > /tmp/vdf2_$tag.ids
  echo "$tag|$ac|$dr|$acc|$pos|$tpr" >> /tmp/vdf2_rows
}

: > /tmp/vdf2_rows
echo "--- (1) full head (baseline) ---"
run full   --spec dflash2
echo "--- (2) draft head (the fix) ---"
run lmhd   --spec dflash2 --lm-head-draft
echo "--- (3) draft head + probe (for alignment analysis) ---"
NINFER_DF2DBG=1 timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --spec dflash2 --lm-head-draft \
    > $J/dl/df2head_stdout.log 2> $J/dl/df2head_probe.log
echo "  probe rc=$? lines=$(wc -l < $J/dl/df2head_probe.log)"
timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --print-token-ids > $J/dl/df2head_plain.log 2>&1
echo "  plain rc=$?"

echo
echo "--- G-A: token 流是否逐位相同 ---"
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
def ids(tag):
    p = pathlib.Path(f"/tmp/vdf2_{tag}.ids")
    return p.read_text().split() if p.exists() else []
A, B = ids("full"), ids("lmhd")
n = min(len(A), len(B))
i = next((k for k in range(n) if A[k] != B[k]), n)
same = (i == n and len(A) == len(B))
print(f"full n={len(A)}  lmhd n={len(B)}  ->  "
      + ("IDENTICAL (G-A ok)" if same else f"DIFFER at {i}"))
print("full:", " ".join(A[:24]))
print("lmhd:", " ".join(B[:24]))
PY
echo VERIFY_DF2HEAD_DONE
