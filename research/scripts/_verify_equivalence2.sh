#!/bin/bash
# v2：修正了两处（v1 的判定不可信）
#   1) token id 的真实格式是 `tokens      generated ids <ids...>`（走 stderr），v1 的正则抓到了别的数字行；
#   2) `--spec dflash2` 显式指定会报 `object handle does not name a materialized tensor`
#      （服务端实测时用的是 `--spec auto`；这条差异本身作为独立线索记录）。
# 另：把生成长度与 prompt 拉长，避免只拿到 3 个 token 就下结论。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
CLI=$R/build/apps/ninfer
MODEL=${MODEL:-/home/user/models/qwen3_8_27b_nvfp4.ninfer}
OUT=$J/_collab/M_verify_equivalence.md
LOG=$J/dl/verify_equivalence2.log
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'

if pgrep -f 'train_dflash2' >/dev/null 2>&1; then
  echo "training is running - this experiment needs the GPU alone" | tee -a "$LOG"; exit 3
fi

run() {  # label extra-args...
  local label=$1; shift
  local log=/home/user/ve2_$label.log
  : > "$log"
  ( cd $R/build && timeout 900 "$CLI" "$MODEL" --prompt "$P" --max-new 48 --max-context 2048 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1 )
  local rc=$?
  local ids
  ids=$(grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//')
  if [ -z "$ids" ]; then
    ids=$(grep -oE '^error: .*' "$log" | head -1)
  fi
  echo "  [$label] rc=$rc n=$(echo $ids | wc -w) ids=${ids:0:110}"
  echo "$label|$rc|$ids" >> /tmp/ve2_rows
}

: > /tmp/ve2_rows
echo "=== verify-equivalence v2 $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"
run plain_a
run plain_b
run dflash2_auto_a --spec auto
run dflash2_auto_b --spec auto
run mtp3           --spec mtp --draft-tokens 3

python3 - <<'PY' | tee -a "$LOG"
import pathlib, time
rows = {}
for line in pathlib.Path("/tmp/ve2_rows").read_text().splitlines():
    lab, rc, ids = (line.split("|", 2) + ["", ""])[:3]
    rows[lab] = ids.split()
def cmp(a, b):
    A, B = rows.get(a, []), rows.get(b, [])
    if not A or not B:
        return "%s vs %s: NO DATA (A=%s B=%s)" % (a, b, A[:1] or "empty", B[:1] or "empty")
    n = min(len(A), len(B)); i = next((k for k in range(n) if A[k] != B[k]), n)
    if i == n and len(A) == len(B):
        return "%s vs %s: IDENTICAL (%d tokens)" % (a, b, len(A))
    return ("%s vs %s: **DIFFER at token %d**  (A[%d]=%s B[%d]=%s; lens %d/%d)"
            % (a, b, i, i, A[i] if i < len(A) else "-", i, B[i] if i < len(B) else "-", len(A), len(B)))
out = ["", "## 判别实验 v2：verify 与 plain 的 token 级等价性 (%s)" % time.strftime("%F %H:%M"), "",
       "- " + cmp("plain_a", "plain_b") + "   ← 自洽性基线（必须 IDENTICAL）",
       "- " + cmp("dflash2_auto_a", "dflash2_auto_b") + "   ← 投机路径自身确定性",
       "- " + cmp("plain_a", "dflash2_auto_a") + "   ← **关键**",
       "- " + cmp("plain_a", "mtp3") + "   ← 交叉验证（另一草稿后端）",
       "",
       "判读：首轮 a=0 时发布的正是 verify 自己算出的**列 0 argmax** ⇒ 第 0 个 token 不同即为 verify 路径的等价性缺口。",
       "已知**代码级**前提：`target_verify_batch_impl` 没有调用 `apply_final_logit_policy`（契约要求每个 lm_head 产生点都调），",
       "但对 qwen 是编译期 no-op ⇒ 不是本模型的分歧来源（已排除）；**Muse 会中招**（softcap 20 / multiplier 0.196）。",
       "另记一条独立线索：CLI 显式 `--spec dflash2` 报 `object handle does not name a materialized tensor`，而 `--spec auto` 正常 —— 待查。"]
p = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_verify_equivalence.md")
p.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
PY
echo VERIFY_EQUIV2_DONE | tee -a "$LOG"
