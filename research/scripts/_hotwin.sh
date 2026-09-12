#!/bin/bash
# 测"热窗太短"假设：dflash2 在 默认 / 关冷窗+大容量 / 更长上下文 下的接受率
# 判据：若放宽热窗后接受率显著上升 ⇒ 热窗太短成立
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/hotwin.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd /home/user/ninfer-fusion/build || exit 3
pkill -9 -x ninfer 2>/dev/null; sleep 2
echo "=== 热窗假设检验 起 $(date '+%m-%d %H:%M:%S') ==="
printf "  %-26s %-9s %-22s %-11s %s\n" 臂 接受率 位置剖面 解码 轮数
run() {
  local label="$1"; shift
  timeout 900 ./apps/ninfer "$M" --prompt "$P" --max-new 96 --no-thinking --greedy \
    --spec dflash2 "$@" > "$J/hw_$label.log" 2>&1
  local a p sp rd kv
  a=$(grep -m1 'dflash2 acceptance rate' "$J/hw_$label.log" | grep -oE '[0-9.]+%')
  p=$(grep -m1 'dflash2 accepted by pos' "$J/hw_$label.log" | sed 's/.*pos *//')
  sp=$(grep -m1 'decode speed' "$J/hw_$label.log" | grep -oE '[0-9.]+ tok/s')
  rd=$(grep -m1 'dflash2 rounds' "$J/hw_$label.log" | grep -oE '[0-9]+')
  kv=$(grep -m1 'kv cache payload' "$J/hw_$label.log" | awk '{print $NF}')
  printf "  %-26s %-9s %-22s %-11s %s  kv=%s\n" "$label" "${a:-?}" "${p:-?}" "${sp:-?}" "${rd:-?}" "${kv:-?}"
}
run 默认                     --max-context 4096
run 冷窗关_4096              --max-context 4096 --cold-policy none
run 冷窗关_32k               --max-context 32768 --cold-policy none
run 大容量_32k               --max-context 32768 --kv-capacity 32768
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo HOTWIN_DONE
