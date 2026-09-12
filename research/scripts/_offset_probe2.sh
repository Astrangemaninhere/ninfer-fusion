#!/bin/bash
# 两个补充判别：
#   A) max-new 192（同 prompt）⇒ 偏离仍在"生成第 36 个"吗？还是随 max-new 变？
#   B) 换**英文/数字** prompt（内容与中文完全不同）⇒ 错值还是 133222 吗？
#      若错值恒为同一 id ⇒ 结构性（某个特殊 token 被固定注入）；若随内容变 ⇒ 数值/上下文相关。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
CLI=$R/build/apps/ninfer
MODEL=${MODEL:-/home/user/models/qwen3_8_27b_nvfp4.ninfer}
LOG=$J/dl/offset_probe2.log
P_ZH='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
P_EN='Write a short paragraph of about two hundred words describing how the scenery of West Lake changes across the four seasons. Keep the sentences flowing and do not use bullet points.'

one() {  # label maxnew prompt extra...
  local label=$1 mx=$2 prompt=$3; shift 3
  local log=/home/user/op2_$label.log
  : > "$log"
  ( cd $R/build && timeout 900 "$CLI" "$MODEL" --prompt "$prompt" --max-new "$mx" --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1 )
  local rc=$?
  local ids pt
  ids=$(grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//')
  pt=$(grep -oE '^summary[[:space:]]+prompt tokens[[:space:]]+[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
  echo "  [$label] rc=$rc prompt=${pt:-?} n=$(echo $ids | wc -w)"
  echo "$label|${pt:-0}|$rc|$ids" >> /tmp/op2_rows
}

: > /tmp/op2_rows
echo "=== offset probe 2 $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"
one zh192_plain 192 "$P_ZH"
one zh192_mtp   192 "$P_ZH" --spec mtp --draft-tokens 3
one en48_plain   48 "$P_EN"
one en48_mtp     48 "$P_EN" --spec mtp --draft-tokens 3

python3 - <<'PY' | tee -a "$LOG"
import pathlib, time
rows = {}
for line in pathlib.Path("/tmp/op2_rows").read_text().splitlines():
    lab, pt, rc, ids = (line.split("|", 3) + ["", "", ""])[:4]
    rows[lab] = (int(pt or 0), ids.split())
out = ["", "## 补充判别 (%s)" % time.strftime("%F %H:%M"), "",
       "| 档 | prompt | 生成数 | 第一个偏离的生成序号 | 绝对位置 | 错值 |",
       "|---|---|---|---|---|---|"]
for tag, desc in (("zh192", "中文 max-new=192"), ("en48", "英文 max-new=48")):
    bp, B = rows.get(tag + "_plain", (0, [])); sp, S = rows.get(tag + "_mtp", (0, []))
    if not B or not S:
        out.append("| %s | %d | NO DATA | | | |" % (desc, bp)); continue
    n = min(len(B), len(S)); i = next((k for k in range(n) if B[k] != S[k]), n)
    if i == n and len(B) == len(S):
        out.append("| %s | %d | %d | 无偏离(全同) | | |" % (desc, bp, len(B)))
    else:
        out.append("| %s | %d | %d | **%d** | %d | %s→%s |"
                   % (desc, bp, len(B), i, bp + i, B[i] if i < len(B) else "-", S[i] if i < len(S) else "-"))
out += ["", "判读：若 192 档的偏离仍在**生成序号 36** ⇒ 固定在「第 36 个生成 token」这个计数上（与 max-new 无关）；",
        "若英文档错值仍为 **133222** ⇒ 结构性（同一 token 被固定注入），否则与内容相关的数值差。"]
p = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_offset_probe2.md")
p.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
PY
echo OFFSET_PROBE2_DONE | tee -a "$LOG"
