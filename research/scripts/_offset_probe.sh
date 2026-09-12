#!/bin/bash
# 判别：偏离点是"生成第 36 个 token"（内容/训练相关）还是"绝对位置 68"（KV 页/长度结构相关）？
# 做法：同一句结尾 prompt，前面加不同长度的填充 ⇒ 改变 prompt tokens 数；每档跑 plain 与 mtp3，
#      找 plain vs mtp3 的第一个 token 级偏离，记录 (生成序号, 绝对位置 = prompt_tokens + 序号)。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
CLI=$R/build/apps/ninfer
MODEL=${MODEL:-/home/user/models/qwen3_8_27b_nvfp4.ninfer}
LOG=$J/dl/offset_probe.log
TAIL='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
FILL='这是一段用于测试的填充文本，它本身没有特别的意义，只是为了让上下文的长度增加，从而观察模型在不同长度下的行为是否一致。'
P_SHORT="$TAIL"
P_MID="$FILL$FILL $TAIL"
P_LONG="$FILL$FILL$FILL$FILL$FILL$FILL$TAIL"

one() {  # label prompt extra...
  local label=$1 prompt=$2; shift 2
  local log=/home/user/op_$label.log
  : > "$log"
  ( cd $R/build && timeout 900 "$CLI" "$MODEL" --prompt "$prompt" --max-new 48 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1 )
  local rc=$?
  local ids pt
  ids=$(grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//')
  pt=$(grep -oE '^summary[[:space:]]+prompt tokens[[:space:]]+[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
  echo "  [$label] rc=$rc prompt_tokens=${pt:-?} n=$(echo $ids | wc -w)"
  echo "$label|${pt:-0}|$rc|$ids" >> /tmp/op_rows
}

: > /tmp/op_rows
echo "=== offset probe $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"
one s_plain "$P_SHORT"
one s_mtp   "$P_SHORT" --spec mtp --draft-tokens 3
one m_plain "$P_MID"
one m_mtp   "$P_MID" --spec mtp --draft-tokens 3
one l_plain "$P_LONG"
one l_mtp   "$P_LONG" --spec mtp --draft-tokens 3

python3 - <<'PY' | tee -a "$LOG"
import pathlib, time
rows = {}
for line in pathlib.Path("/tmp/op_rows").read_text().splitlines():
    lab, pt, rc, ids = (line.split("|", 3) + ["", "", ""])[:4]
    rows[lab] = (int(pt or 0), ids.split())
out = ["", "## 偏离点：生成序号 vs 绝对位置 (%s)" % time.strftime("%F %H:%M"), "",
       "| 档 | prompt tokens | 第一个偏离的生成序号 | 绝对位置(=prompt+序号) | 错值 |",
       "|---|---|---|---|---|"]
for tag, name in (("s", "短(32)"), ("m", "中(~120)"), ("l", "长(~260)")):
    bp, B = rows.get(tag + "_plain", (0, []))
    sp, S = rows.get(tag + "_mtp", (0, []))
    if not B or not S:
        out.append("| %s | %s | NO DATA | | |" % (name, bp)); continue
    n = min(len(B), len(S)); i = next((k for k in range(n) if B[k] != S[k]), n)
    if i == n and len(B) == len(S):
        out.append("| %s | %d | 无偏离(%d token 全同) | | |" % (name, bp, len(B)))
    else:
        out.append("| %s | %d | **%d** | **%d** | %s→%s |"
                   % (name, bp, i, bp + i, B[i] if i < len(B) else "-", S[i] if i < len(S) else "-"))
out += ["", "判读：若三档的**生成序号**都≈36 ⇒ 与内容/训练相关；若**绝对位置**都≈68 ⇒ 与 KV 页/长度结构相关。"]
p = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_offset_probe.md")
p.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
PY
echo OFFSET_PROBE_DONE | tee -a "$LOG"
