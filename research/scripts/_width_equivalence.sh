#!/bin/bash
# width 判别：让 verify 的 T 取 2 / 4 / 6（--draft-tokens 1 / 3 / 5），plain 作为基准。
#   若"偏离点或错值"随 T 变 ⇒ 差异是 T/batching 相关的数值噪声；
#   若三者都在**同一个 token** 偏离到**同一个值** ⇒ verify 与 plain 之间存在与 T 无关的固定差异。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
CLI=$R/build/apps/ninfer
MODEL=${MODEL:-/home/user/models/qwen3_8_27b_nvfp4.ninfer}
LOG=$J/dl/width_equiv.log
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'

if pgrep -f 'train_dflash2' >/dev/null 2>&1; then echo "training running" | tee -a "$LOG"; exit 3; fi

one() {  # label extra...
  local label=$1; shift
  local log=/home/user/we_$label.log
  : > "$log"
  ( cd $R/build && timeout 900 "$CLI" "$MODEL" --prompt "$P" --max-new 48 --max-context 2048 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1 )
  local rc=$?
  local ids; ids=$(grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
                   sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//')
  local rate; rate=$(grep -oE 'spec_accept_rate=[0-9.]+' "$log" | tail -1)
  local tpr;  tpr=$(grep -oE 'speculative=\S+ [0-9.]+tok/round' "$log" | tail -1)
  echo "  [$label] rc=$rc n=$(echo $ids | wc -w) $rate $tpr"
  echo "$label|$rc|$ids|$rate" >> /tmp/we_rows
}

: > /tmp/we_rows
echo "=== width-equivalence $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"
one plain
one mtp_d1 --spec mtp --draft-tokens 1
one mtp_d3 --spec mtp --draft-tokens 3
one mtp_d5 --spec mtp --draft-tokens 5

python3 - <<'PY' | tee -a "$LOG"
import pathlib, time
rows = {}
for line in pathlib.Path("/tmp/we_rows").read_text().splitlines():
    lab, rc, ids, rate = (line.split("|", 3) + ["", "", ""])[:4]
    rows[lab] = (ids.split(), rate)
base, _ = rows.get("plain", ([], ""))
out = ["", "## width 判别：偏离点是否随 verify 的 T 变化 (%s)" % time.strftime("%F %H:%M"), "",
       "plain 基准: %d tokens" % len(base)]
for lab in ("mtp_d1", "mtp_d3", "mtp_d5"):
    A, rate = rows.get(lab, ([], ""))
    if not A:
        out.append("- %s: NO DATA" % lab); continue
    n = min(len(base), len(A)); i = next((k for k in range(n) if base[k] != A[k]), n)
    if i == n and len(base) == len(A):
        out.append("- %-7s T=%d : IDENTICAL (%d tokens)  %s" % (lab, {"mtp_d1":2,"mtp_d3":4,"mtp_d5":6}[lab], len(A), rate))
    else:
        out.append("- %-7s T=%d : **DIFFER at token %d** base=%s spec=%s  %s"
                   % (lab, {"mtp_d1":2,"mtp_d3":4,"mtp_d5":6}[lab], i,
                      base[i] if i < len(base) else "-", A[i] if i < len(A) else "-", rate))
out += ["", "判读：三者若在**同一 token** 且错到**同一值** ⇒ verify 与 plain 之间有与 T 无关的固定差异（不是 batching 数值噪声）；",
        "若偏离点/错值随 T 变化 ⇒ 属于 T 相关的数值差（kernel 实例化）。"]
p = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_width_equivalence.md")
p.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
PY
echo WIDTH_EQUIV_DONE | tee -a "$LOG"
