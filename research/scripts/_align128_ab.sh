#!/bin/bash
# S_A 可证伪预测：prompt 对齐到 128 整数倍后，--prefill-chunk 128 vs 4096 的首 token 是否变一致。
#   一致  => `gated_delta_net.cpp:254-261` 的 BF16 预归一化分支（末段 T<64 走 FP32 寄存器）是首因
#   仍分叉 => 主因是无条件的 tile/route 切换（按 T 选 kernel），下一步查 nvfp4_gdn_input_w4a4.cu:40
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/align128_ab.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== align128 A/B $(date '+%F %H:%M:%S') ==="
cd $R/build || exit 3
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P="$(cat /tmp/prompt_align128.txt)"
echo "prompt chars=${#P}"

one() {
  local tag=$1; shift
  local log=/home/user/a128_$tag.log
  timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 32 --max-context 8192 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
      sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > /tmp/a128_$tag.ids
  local pt; pt=$(grep -oE 'prompt tokens +[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
  echo "  [$tag] prompt_tokens=$pt n=$(wc -w < /tmp/a128_$tag.ids) first=$(head -1 /tmp/a128_$tag.ids)"
}

one chunk128  --prefill-chunk 128
one chunk4096 --prefill-chunk 4096

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
def ids(t):
    p = pathlib.Path(f"/tmp/a128_{t}.ids")
    return p.read_text().split() if p.exists() else []
A, B = ids("chunk128"), ids("chunk4096")
if not A or not B:
    print("NO DATA")
else:
    n = min(len(A), len(B))
    i = next((x for x in range(n) if A[x] != B[x]), n)
    same = (i == n and len(A) == len(B))
    print(f"chunk128 first={A[0]}  chunk4096 first={B[0]}")
    if same:
        print(f"⇒ 首 token 一致、整段 IDENTICAL（{len(A)} tok）⇒ S_A 的①BF16 预归一化分支成立")
    else:
        print(f"⇒ 仍分叉 at {i}/{n}  A={A[max(0,i-2):i+2]} B={B[max(0,i-2):i+2]}")
        print("  ⇒ 主因不是末段归一化分支，指向按 T 选 tile/route 的无条件切换")
PY
echo ALIGN128_AB_DONE
