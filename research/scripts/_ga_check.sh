#!/bin/bash
# G-A（最根本的一条）：投机路径的 token 流必须与 plain 逐位一致。
# 历史记录 M_patchA_effect.md 3 实测两条不同草稿后端都偏离 plain ⇒ 先确认今天是否仍然偏离。
# 同一 prompt 下跑：plain / --spec dflash2 / --spec dflash(K=7)，比较 token id 流。
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/ga_check.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== G-A spec vs plain $(date '+%F %H:%M:%S') ==="
cd $R/build || exit 3
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
SHEN='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
NUM='0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9'

one() {  # tag prompt extra...
  local tag=$1 p=$2; shift 2
  local log=/home/user/ga_$tag.log
  timeout 900 ./apps/ninfer "$A" --prompt "$p" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  local rc=$?
  grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
      sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > /tmp/ga_$tag.ids
  echo "  [$tag] rc=$rc n=$(wc -w < /tmp/ga_$tag.ids)"
}

echo "--- zh prompt ---"
one zh_plain   "$SHEN"
one zh_dflash2 "$SHEN" --spec dflash2
one zh_dflash  "$SHEN" --spec dflash --draft-tokens 7
echo "--- num prompt ---"
one num_plain   "$NUM"
one num_dflash2 "$NUM" --spec dflash2

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
def ids(tag):
    p = pathlib.Path(f"/tmp/ga_{tag}.ids")
    return p.read_text().split() if p.exists() else []
def cmp(a, b):
    A, B = ids(a), ids(b)
    if not A or not B:
        return f"{a} vs {b}: NO DATA"
    n = min(len(A), len(B))
    i = next((k for k in range(n) if A[k] != B[k]), n)
    same = (i == n and len(A) == len(B))
    if same:
        return f"{a} vs {b}: IDENTICAL ({len(A)} tok)  <-- G-A ok"
    return (f"{a} vs {b}: DIFFER at {i}/{n}  plain={A[max(0,i-2):i+3]} spec={B[max(0,i-2):i+3]}")
print()
for a, b in (("zh_plain", "zh_dflash2"), ("zh_plain", "zh_dflash"),
             ("num_plain", "num_dflash2")):
    print("  " + cmp(a, b))
PY
echo GA_CHECK_DONE
