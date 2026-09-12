#!/bin/bash
# ① 找接受率/位置剖面的打印点与它的 gate（env 还是 flag）
# ② 找到就把开关带上，立刻跑 dspark 的 OLD(18:08) vs NEW(22:17) 对照
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH="/home/user/.local/bin:$PATH"

echo '=== ① 接受率/位置剖面的打印点 ==='
grep -rn 'accepted by pos\|spec_accept_rate\|accept_rate\|ACCEPT_STATS\|spec_stats' "$R/src" "$R/apps" 2>/dev/null | head -12 | cut -c1-150
echo
echo '=== ①b 相关 env 变量（打印点附近的 gate）==='
grep -rnoE 'NINFER_[A-Z0-9_]+' "$R/src" "$R/apps" 2>/dev/null | grep -iE 'stat|spec|accept|dbg|debug' | sort -u -t: -k3 | head -12 | cut -c1-120
echo
echo '=== ①c 打印点上下文（若上面命中）==='
for f in $(grep -rl 'accepted by pos' "$R/src" "$R/apps" 2>/dev/null | head -2); do
  echo "--- $f ---"
  n=$(grep -n 'accepted by pos' "$f" | head -1 | cut -d: -f1)
  awk -v s=$((n-12)) -v e=$((n+3)) 'NR>=s && NR<=e {printf "%5d| %s\n", NR, $0}' "$f" | cut -c1-135
done

echo
echo '=== ② dspark A/B（带上开关）==='
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
export NINFER_SPEC_STATS=1
run() {
  local tag=$1 bin=$2; shift 2
  local log=/home/user/ab2_$tag.log
  ( cd "$R/build" && timeout 900 "$bin" /home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer \
      --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy \
      --print-token-ids --spec dflash --draft-tokens 7 "$@" > "$log" 2>&1 )
  echo "  [$tag] rc=$? $(grep -oE 'decode speed +[0-9.]+ tok/s' "$log" | head -1)"
  grep -oE 'spec_accept_rate=[0-9.]+' "$log" | head -1 | sed 's/^/    /'
  grep -oE 'accepted by pos[^|]*' "$log" | head -1 | sed 's/^/    /'
  grep -oE '^tokens +generated ids.*' "$log" | head -1 | cut -c1-150 | sed 's/^/    ids: /'
}
echo '--- OLD = /home/user/ninfer_pre_fix (18:08) ---'
run OLD /home/user/ninfer_pre_fix
echo '--- NEW = build/apps/ninfer (22:17) ---'
run NEW "$R/build/apps/ninfer"
date +%H:%M:%S
