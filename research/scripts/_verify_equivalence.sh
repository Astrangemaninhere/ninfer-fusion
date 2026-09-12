#!/bin/bash
# 判别实验：投机路径的 argmax 与 plain 路径的 argmax 到底等不等价？
#
# 为什么用 token-id 而不是文本：文本比对会被 thinking/模板/解码细节干扰（见 _TODO.md §124 的
# "空对空"教训）。CLI 支持 `--print-token-ids`，直接比 id 序列，干净且可复现。
#
# 设计（全都是贪心 temperature=0，理论上必须逐 token 相同）：
#   1) plain 跑两次           -> 自洽性基线（必须完全相同，否则实验本身无效）
#   2) dflash2 跑两次         -> 投机路径自身的确定性
#   3) plain vs dflash2       -> 第 1 个 token 就不同 ⇒ verify 的列 0 logits ≠ plain 的 next-token logits
#                               （与草稿无关：第一轮的 anchor 是最后一个 prompt token，
#                                若草稿错则 a=0，发布的正是 verify 自己算出的列 0 argmax）
#   4) plain vs mtp3          -> 同一 verify 路径的第二个草稿后端，用于交叉验证
#
# 需要在没有别的模型占用 GPU 时跑（训练/其它 serve 在跑时脚本会退出并提示）。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
CLI=$R/build/apps/ninfer
MODEL=${MODEL:-/home/user/models/qwen3_8_27b_nvfp4.ninfer}
OUT=$J/_collab/M_verify_equivalence.md
LOG=$J/dl/verify_equivalence.log
P='请把下面这句话补完，只输出补完后的整句话：今天天气很好，我想去'

if [ ! -x "$CLI" ]; then echo "CLI missing: $CLI" | tee -a "$LOG"; exit 2; fi
if pgrep -f 'train_dflash2' >/dev/null 2>&1 || pgrep -f 'ninfer-serve' >/dev/null 2>&1; then
  echo "GPU busy (training or another serve is running) - this experiment needs the GPU alone." | tee -a "$LOG"
  exit 3
fi

run() {  # label extra-args...
  local label=$1; shift
  local log=/home/user/ve_$label.log
  : > "$log"
  ( cd $R/build && timeout 600 "$CLI" "$MODEL" --prompt "$P" --max-new 24 --max-context 2048 \
      --no-thinking --print-token-ids "$@" > "$log" 2>&1 )
  local rc=$?
  local ids
  ids=$(grep -oE '^[[:space:]]*[0-9]+([[:space:]]+[0-9]+)+' "$log" | tail -1 | tr -s ' ' | sed 's/^ //')
  if [ -z "$ids" ]; then
    ids=$(grep -A2 -iE 'token ids|generated ids' "$log" | tail -1 | tr -s ' ' | sed 's/^ //')
  fi
  echo "  [$label] rc=$rc ids=${ids:0:120}"
  echo "$label|$rc|$ids" >> /tmp/ve_rows
}

: > /tmp/ve_rows
echo "=== verify-equivalence experiment $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"
run plain_a
run plain_b
run dflash2_a --spec dflash2
run dflash2_b --spec dflash2
run mtp3      --spec mtp --draft-tokens 3

python3 - <<'PY' | tee -a "$LOG"
import pathlib
rows = {}
for line in pathlib.Path("/tmp/ve_rows").read_text().splitlines():
    lab, rc, ids = (line.split("|", 2) + ["", ""])[:3]
    rows[lab] = ids.split()
def cmp(a, b):
    A, B = rows.get(a, []), rows.get(b, [])
    if not A or not B:
        return "%s vs %s: NO DATA" % (a, b)
    n = min(len(A), len(B)); i = next((k for k in range(n) if A[k] != B[k]), n)
    if i == n and len(A) == len(B):
        return "%s vs %s: IDENTICAL (%d tokens)" % (a, b, len(A))
    return ("%s vs %s: DIFFER at token %d  (A[%d]=%s  B[%d]=%s; lens %d/%d)"
            % (a, b, i, i, A[i] if i < len(A) else "-", i, B[i] if i < len(B) else "-", len(A), len(B)))
out = []
out.append("## 判别实验：verify 与 plain 的 token 级等价性 (%s)" % __import__("time").strftime("%F %H:%M"))
out.append("")
out.append("- " + cmp("plain_a", "plain_b") + "   ← 自洽性基线（必须 IDENTICAL）")
out.append("- " + cmp("dflash2_a", "dflash2_b") + "   ← 投机路径自身确定性")
out.append("- " + cmp("plain_a", "dflash2_a") + "   ← 关键：若第 0 个 token 就不同 ⇒ verify 列 0 logits ≠ plain")
out.append("- " + cmp("plain_a", "mtp3") + "   ← 交叉验证（同一 verify 路径、另一草稿后端）")
out.append("")
out.append("判读：plain vs 投机的**第 0 个 token**由 verify 自己算出的列 0 argmax 决定（首轮 a=0 时发布的正是它），")
out.append("与草稿质量无关 ⇒ 第 0 个 token 不同即为 verify 路径本身的等价性缺口。")
pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_verify_equivalence.md").write_text(
    "\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
PY
echo VERIFY_EQUIV_DONE | tee -a "$LOG"
