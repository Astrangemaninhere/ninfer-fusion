#!/bin/bash
# 加固版前后对照（v2）。相比 v1 修了三处会白等/出假结果的缺陷：
#   (a) wait_idle 只看编译器 -> 两次 make 之间的空档会提前开跑，与 ninfer-serve 的
#       编译并发吃内存（曾把机器压到 OOM）。现在同时等构建驱动进程消失。
#   (b) 无"二进制是否真的被重建"校验 -> 若构建失败，会拿 18:08 的修复前二进制当
#       "修复后"跑，得到假结论。现在用 mtime 硬门禁（必须晚于 19:40 的备份时间）。
#   (c) 1h 超时太短（大 TU 单编 40 min + 链接 + serve 全量）-> 放宽到 2h。
# 判据与 v1 相同：dspark accept/p0 应显著上升；mtp 应逐位 IDENTICAL；df2 应基本不变。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
NEW=$R/build/apps/ninfer
OLD=/home/user/ninfer_pre_fix
REF=/home/user/ninfer_pre_fix           # 19:40 备份：新二进制必须比它新
LOG=$J/dl/post_fix_verify2.log
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== post-fix verify v2 $(date '+%F %H:%M:%S') ==="

# 自匹配陷阱：pgrep -f 的模式若与自身 cmdline 同形会永远为真，用 [x] 断开字面量。
compiler_alive() { pgrep -f 'bin/nvcc|cc1plus' >/dev/null 2>&1; }
driver_alive()   { pgrep -f '_rebuild_after_s3[5][.]sh' >/dev/null 2>&1; }

wait_idle() {
  local t0=$(date +%s)
  while compiler_alive || driver_alive; do
    if [ $(( $(date +%s) - t0 )) -gt 7200 ]; then echo "TIMEOUT 等构建 (>2h)"; return 1; fi
    sleep 20
  done
  return 0
}
wait_idle || exit 2
echo "--- 构建已停 ($(date +%H:%M:%S))，结算前静置 30s 让链接落盘 ---"
sleep 30

# 硬门禁：二进制必须比 19:40 的备份新，否则说明构建没成功 —— 绝不用旧二进制出结论。
mt_new=$(stat -c %Y "$NEW" 2>/dev/null || echo 0)
mt_ref=$(stat -c %Y "$REF")
if [ "$mt_new" -le "$mt_ref" ]; then
  echo "STALE_BINARY: $NEW 未更新 (mtime $(date -d @$mt_new '+%F %H:%M:%S') <= 备份 $(date -d @$mt_ref '+%F %H:%M:%S'))"
  echo "  -> 构建很可能失败。见 $J/dl/rebuild_after_s35.log，本对照拒绝出结论。"
  echo POST_FIX_VERIFY2_ABORTED | tee -a "$LOG"
  exit 4
fi
ls -l --time-style=+%F_%H:%M "$NEW" "$OLD" | awk '{print "  bin:", $6, $5, $NF}'
free -g | sed -n 2p

one() {  # tag bin extra...
  local tag=$1 bin=$2; shift 2
  local log=/home/user/pfv_$tag.log
  ( cd $R/build && timeout 900 "$bin" /home/user/models/${MODEL_NAME} --prompt "$P" \
      --max-new 96 --max-context 4096 --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1 )
  local ids; ids=$(grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
                   sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//')
  local rate; rate=$(grep -oE 'spec_accept_rate=[0-9.]+' "$log" | tail -1)
  local pos;  pos=$(grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  local tpr;  tpr=$(grep -oE '[0-9.]+tok/round' "$log" | tail -1)
  echo "  [$tag] n=$(echo $ids | wc -w) $rate $tpr pos=[$pos]"
  echo "$tag|$ids|$rate|$pos" >> /tmp/pfv_rows
}

: > /tmp/pfv_rows
echo
echo "--- (1) dspark：修复前 vs 修复后（判据：accept/p0 显著上升）---"
MODEL_NAME=qwen3_8_27b_nvfp4_dspark.ninfer
[ -x "$OLD" ] && one dspark_OLD "$OLD" --spec dflash --draft-tokens 7
one dspark_NEW "$NEW" --spec dflash --draft-tokens 7
echo
echo "--- (2) mtp3：等价性（判据：token id 应逐位相同）---"
MODEL_NAME=qwen3_8_27b_nvfp4.ninfer
[ -x "$OLD" ] && one mtp_OLD "$OLD" --spec mtp --draft-tokens 3
one mtp_NEW "$NEW" --spec mtp --draft-tokens 3
echo
echo "--- (3) dflash2：对照（判据：基本不变）---"
MODEL_NAME=qwen3_8_27b_nvfp4_dflash2.ninfer
[ -x "$OLD" ] && one df2_OLD "$OLD" --spec auto
one df2_NEW "$NEW" --spec auto

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
rows = {}
for line in pathlib.Path("/tmp/pfv_rows").read_text().splitlines():
    tag, ids, rate, pos = (line.split("|", 3) + ["", "", ""])[:4]
    rows[tag] = (ids.split(), rate, pos)
def cmp(a, b):
    A, ar, ap = rows.get(a, ([], "", "")); B, br, bp = rows.get(b, ([], "", ""))
    if not A or not B: return "%s vs %s: NO DATA (空输出绝不当作证据)" % (a, b)
    n = min(len(A), len(B)); i = next((k for k in range(n) if A[k] != B[k]), n)
    same = (i == n and len(A) == len(B))
    return ("%s vs %s: %s | accept %s -> %s | p0 %s -> %s"
            % (a, b, "IDENTICAL" if same else "DIFFER at %d" % i, ar or "-", br or "-",
               (ap.split(",")[0] if ap else "-"), (bp.split(",")[0] if bp else "-")))
out = ["", "## 修复前后对照 v2 (%s)" % __import__("time").strftime("%F %H:%M"), "",
       "- " + cmp("dspark_OLD", "dspark_NEW") + "   <- 判据：accept/p0 应显著上升",
       "- " + cmp("mtp_OLD", "mtp_NEW") + "   <- 判据：应 **IDENTICAL**（零风险等价改动）",
       "- " + cmp("df2_OLD", "df2_NEW") + "   <- 判据：应相同（dflash2 的列本来就对）"]
p = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_post_fix_verify.md")
p.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
PY
echo POST_FIX_VERIFY2_DONE | tee -a "$LOG"
