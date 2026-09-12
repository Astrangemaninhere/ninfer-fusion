#!/bin/bash
# 等编译完成 → 验证 dspark 的列偏移修复（位置剖面应显著上升）→ dflash2 作为对照（它的列本来就对，应不变）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
CLI=$R/build/apps/ninfer
LOG=$J/dl/offset_family_verify.log
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
export PATH="/home/user/.local/bin:$PATH"

exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== offset-family verify armed $(date '+%F %H:%M:%S') ==="

# 等编译结束
t0=$(date +%s)
while :; do
  busy=0
  pgrep -f 'bin/nvcc|cc1plus|ccache' >/dev/null 2>&1 && busy=1
  if [ "$busy" -eq 0 ] && grep -q 'rebuild done' "$J/dl/rebuild_after_s35.log" 2>/dev/null; then break; fi
  [ $(( $(date +%s) - t0 )) -gt 3600 ] && { echo "TIMEOUT 等编译"; break; }
  sleep 20
done
echo "编译结束 $(date +%H:%M:%S)；二进制："
ls -l --time-style=+%H:%M "$CLI" "$R/build/apps/ninfer-serve" 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
[ -x "$CLI" ] || { echo "CLI 不存在，退出"; echo OFFSET_FAMILY_VERIFY_DONE; exit 2; }

run_pos() {  # label model extra...
  local label=$1 model=$2; shift 2
  echo
  echo "--- [$label] $(date +%H:%M:%S) ---"
  ( cd $R/build && timeout 600 "$CLI" "$model" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy "$@" > /tmp/ofv_$label.log 2>&1 )
  echo "  rc=$?"
  grep -iE 'acceptance rate|accepted by pos|acceptance length|draft window|rounds|drafted tokens|tok/s' \
    /tmp/ofv_$label.log | head -10 | sed 's/^/  /'
}

run_pos dspark_fixed  /home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer  --spec dflash --draft-tokens 7
run_pos dflash2_ctrl  /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer --spec auto
run_pos mtp3_ctrl     /home/user/models/qwen3_8_27b_nvfp4.ninfer          --spec mtp --draft-tokens 3

echo
echo "对照（修复前实测）：dspark accept=11.22% / p(pos0)=15/56=26.8% / 1.39 tok/round / 35.05 tok/s"
echo "全族状态：dspark 列偏移【已修】、dflash2 列正确但训练目标偏移一行【已改默认值，需重训】、MTP 不适用该族"
echo OFFSET_FAMILY_VERIFY_DONE
