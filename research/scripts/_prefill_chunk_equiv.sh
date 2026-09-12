#!/bin/bash
# 判别"形状相关的数值不等价"是否是本引擎的既有性质（与投机无关）：
#   同一个长 prompt，只用 --prefill-chunk 改变分块大小（1 块 vs 多块），比较生成的 token 流。
#   不同 => 同数学在不同形状 kernel 上数值不等价（候选 1 成立，投机漂移至少部分是它）
#   相同 => 形状切换是比特等价的，漂移另有来源（回到 GDN/状态）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/prefill_chunk_equiv.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== prefill chunk equivalence $(date '+%F %H:%M:%S') ==="
cd $R/build || exit 3
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer

# 造一个足够长的 prompt（约 30 段重复，使分块真正生效）
python3 - <<'PY' > /tmp/long_prompt.txt
seg = "杭州地处长江三角洲南翼，浙江省北部，北靠天目山，南临钱塘江，水网密布、湖荡众多。"
print(seg * 60)
PY
P="$(cat /tmp/long_prompt.txt)"
echo "prompt chars=${#P}"

one() {  # tag chunk...
  local tag=$1; shift
  local log=/home/user/pc_$tag.log
  timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 64 --max-context 8192 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  local rc=$?
  grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
      sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > /tmp/pc_$tag.ids
  local pt; pt=$(grep -oE 'prompt tokens[[:space:]]+[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
  echo "  [$tag] rc=$rc prompt_tokens=$pt n=$(wc -w < /tmp/pc_$tag.ids)"
}

one chunk128  --prefill-chunk 128
one chunk4096 --prefill-chunk 4096

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
def ids(t):
    p = pathlib.Path(f"/tmp/pc_{t}.ids")
    return p.read_text().split() if p.exists() else []
A, B = ids("chunk128"), ids("chunk4096")
if not A or not B:
    print("NO DATA")
else:
    n = min(len(A), len(B))
    i = next((x for x in range(n) if A[x] != B[x]), n)
    if i == n and len(A) == len(B):
        print(f"chunk128 vs chunk4096: IDENTICAL ({len(A)} tok)  ⇒ 形状切换比特等价")
    else:
        print(f"chunk128 vs chunk4096: **DIFFER at {i}/{n}**  "
              f"A={A[max(0,i-2):i+2]} B={B[max(0,i-2):i+2]}")
        print("  ⇒ 同数学在不同形状 kernel 上数值不等价（与投机无关的既有性质）")
PY
echo PREFILL_CHUNK_EQUIV_DONE
