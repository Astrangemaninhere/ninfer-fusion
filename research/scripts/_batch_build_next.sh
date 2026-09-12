#!/bin/bash
# 批次链：等 v2 前后对照做完 -> 落 A5b + S52 -> 重编 -> 自动跑 A5b 自身对照。
#
# 为什么这样串：
#   * 不空等：v2 对照跑完的瞬间就开始落补丁，没有"等我回来再点一下"的空档；
#   * 归因干净：A5b（attention_valid 表降回 k）与已落地的列偏移修复是**一对**，
#     必须分开测。本轮对照测的是"列偏移单独"，A5b 留到本轮之后，
#     并以 /home/user/ninfer_before_a5b（= 本轮构建产物）作为它自己的 A/B 基线；
#   * 不重新 configure：只改已存在的文件 + touch 包含者。加新文件会触发 cmake
#     重新配置 -> 全量重编（今天的 2h 就是那么来的），E7 的测试文件因此不进本批。
#
# 硬门禁（任何一条不过就不出结论）：
#   G1 每个补丁先 dry-run，任一不过 -> 整体放弃，不半途落；
#   G2 编译成功且 ninfer/ninfer-serve 都被重新链接（mtime 晚于备份）；
#   G3 任一侧空输出 -> INVALID，绝不当证据。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab
R=/home/user/ninfer-fusion
B=$R/build/apps
LOG=$J/dl/batch_build_next.log
export PATH="/home/user/.local/bin:$PATH"
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== batch(A5b+S52) $(date '+%F %H:%M:%S') ==="

# 自匹配陷阱：模式与自身 cmdline 同形会让 pgrep 永远为真，用 [x] 断开字面量。
verify_running() { pgrep -f '_post_fix_verify[2][.]sh' >/dev/null 2>&1; }
compiler_alive() { pgrep -f 'bin/nvcc|cc1plus' >/dev/null 2>&1; }

# ---- 等 v2 对照结束（它是 GPU 独占的，绝不并发）----
t0=$(date +%s)
while verify_running || compiler_alive; do
  if [ $(( $(date +%s) - t0 )) -gt 10800 ]; then echo "TIMEOUT 等 v2 对照 (>3h)"; exit 2; fi
  sleep 30
done
sleep 20
echo "--- v2 对照已结束 $(date +%H:%M:%S)，开始落补丁 ---"
grep -E '^\- ' "$J/_collab/M_post_fix_verify.md" 2>/dev/null | head -5

PATCHES=("$C/A5b_attention_valid_width.diff" "$C/E9_s52_dflash2_k_slice.diff")

# ---- G1: 全量 dry-run 门禁（两种换行都能打）----
resolve() {  # $1 -> 回显可打的补丁路径，或空
  local f=$1
  patch -p1 --dry-run -d "$R" < "$f" >/dev/null 2>&1 && { echo "$f"; return 0; }
  local t=/tmp/norm_$(basename "$f")
  tr -d '\r' < "$f" > "$t"
  patch -p1 --dry-run -d "$R" < "$t" >/dev/null 2>&1 && { echo "$t"; return 0; }
  return 1
}
declare -a RESOLVED=()
for f in "${PATCHES[@]}"; do
  [ -f "$f" ] || { echo "G1 FAIL: 缺文件 $f"; exit 3; }
  r=$(resolve "$f") || { echo "G1 FAIL: $(basename "$f") 打不上（两种换行都试过）"; exit 3; }
  echo "  G1 OK: $(basename "$f") -> $r"
  RESOLVED+=("$r")
done

# ---- 备份基线二进制（= 列偏移单独修复版，A5b 的 A/B 基线）----
for b in ninfer ninfer-serve; do
  cp -f "$B/$b" "/home/user/${b}_before_a5b" || { echo "备份 $b 失败"; exit 4; }
done
ls -l --time-style=+%H:%M /home/user/ninfer_before_a5b | awk '{print "  基线:", $6, $5, $NF}'
BASE_MT=$(stat -c %Y "$B/ninfer")

# ---- 落补丁（-b 留 .orig 便于回退）----
for r in "${RESOLVED[@]}"; do
  patch -p1 -b -d "$R" < "$r" || { echo "落补丁失败: $r"; exit 5; }
  echo "  applied: $(basename "$r")"
done

# ---- 无头文件依赖跟踪：touch 所有包含这些头的源文件（不新增文件 => 不触发 configure）----
cd "$R" || exit 6
TOUCHED=$(grep -rl --include='*.cpp' --include='*.cu' \
            -e 'dflash_impl.h' -e 'dflash2_impl.h' -e 'speculative_options.h' \
            src/ 2>/dev/null | sort -u)
echo "--- touch 包含者 ($(echo "$TOUCHED" | wc -l) 个) ---"
echo "$TOUCHED" | head -12 | sed 's/^/    /'
[ -n "$TOUCHED" ] && touch $TOUCHED
# 改动到 layouts_impl.h 的 TU 也要带上（S52 的 selector scratch 尺寸在那里）
LAY=$(grep -rl --include='*.cpp' --include='*.cu' 'layouts_impl.h' src/ 2>/dev/null | sort -u)
echo "--- touch layouts 包含者 ($(echo "$LAY" | wc -l) 个) ---"
[ -n "$LAY" ] && touch $LAY

# ---- 编译（make ninfer 不产 serve，必须两个都编）----
cd "$R/build" || exit 7
for target in ninfer ninfer-serve; do
  rc=1
  for attempt in 1 2 3; do
    echo "--- make $target attempt $attempt $(date +%H:%M:%S) ---"
    free -g | sed -n 2p
    make "$target" -j1 > /tmp/batch_${target}_$attempt.log 2>&1
    rc=$?
    tail -3 /tmp/batch_${target}_$attempt.log
    [ $rc -eq 0 ] && break
    grep -E 'error:' /tmp/batch_${target}_$attempt.log | head -6 | cut -c1-170
    sleep 5
  done
  echo "  $target rc=$rc"
  [ $rc -ne 0 ] && { echo "编译失败，本批不出结论"; echo BATCH_ABORTED | tee -a "$LOG"; exit 8; }
done

# ---- G2: 两个二进制都必须真的被重链 ----
for b in ninfer ninfer-serve; do
  m=$(stat -c %Y "$B/$b")
  if [ "$m" -le "$BASE_MT" ] && [ "$b" = ninfer ]; then
    echo "G2 FAIL: $b 未重链（mtime $(date -d @$m '+%H:%M:%S') <= 基线 $(date -d @$BASE_MT '+%H:%M:%S')）"
    echo BATCH_ABORTED | tee -a "$LOG"; exit 9
  fi
  ls -l --time-style=+%H:%M "$B/$b" | awk '{print "  新:", $6, $5, $NF}'
done

# ---- A5b 自身对照：基线(列偏移单独) vs 本次(A5b+S52) ----
# 注意：S52 只放宽 K 的取值域，默认 K=7 时行为应与基线一致；A5b 才是本批的变量。
echo
echo "--- A5b 对照（dspark，基线 vs 本批）---"
PP='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
: > /tmp/a5b_rows
one5() {
  local tag=$1 bin=$2; shift 2
  local log=/home/user/a5b_$tag.log
  ( cd "$R/build" && timeout 900 "$bin" /home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer \
      --prompt "$PP" --max-new 96 --max-context 4096 --no-thinking --greedy \
      --print-token-ids --spec dflash --draft-tokens 7 "$@" > "$log" 2>&1 )
  local ids rate pos
  ids=$(grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//')
  rate=$(grep -oE 'spec_accept_rate=[0-9.]+' "$log" | tail -1)
  pos=$(grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  echo "  [$tag] n=$(echo $ids | wc -w) $rate pos=[$pos]"
  echo "$tag|$ids|$rate|$pos" >> /tmp/a5b_rows
}
one5 before_a5b /home/user/ninfer_before_a5b
one5 after_a5b  "$B/ninfer"

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
rows = {}
for line in pathlib.Path("/tmp/a5b_rows").read_text().splitlines():
    tag, ids, rate, pos = (line.split("|", 3) + ["", "", ""])[:4]
    rows[tag] = (ids.split(), rate, pos)
A, ar, ap = rows.get("before_a5b", ([], "", ""))
B, br, bp = rows.get("after_a5b", ([], "", ""))
if not A or not B:
    verdict = "INVALID：有一侧空输出（绝不当作证据）"
else:
    n = min(len(A), len(B)); i = next((k for k in range(n) if A[k] != B[k]), n)
    same = (i == n and len(A) == len(B))
    verdict = ("%s | accept %s -> %s | p0 %s -> %s"
               % ("IDENTICAL" if same else "DIFFER at %d" % i, ar or "-", br or "-",
                  (ap.split(",")[0] if ap else "-"), (bp.split(",")[0] if bp else "-")))
out = ["", "## A5b 对照（attention_valid 表保持 width）(%s)" % __import__("time").strftime("%F %H:%M"), "",
       "- before(列偏移单独) vs after(+A5b): " + verdict,
       "- 判据：A5b 应让 dspark 的 accept/p0 **再上一截**（它修的是列 k 看不到自己 KV 槽）；",
       "  若反而下降或偏离，回退：patch -R 或从 .orig 恢复 dflash_impl.h。",
       "- 注：S52 同批落地，但默认 K=7 时行为应与基线一致，故本对照的变量只有 A5b。"]
pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_a5b_verify.md").write_text(
    "\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
PY
echo BATCH_BUILD_NEXT_DONE | tee -a "$LOG"
