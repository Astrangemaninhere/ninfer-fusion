#!/bin/bash
# 单臂运行器：bash run1.sh <label> <cols> -- <ninfer args...>
set -u
label="$1"; cols="$2"; shift 2; [ "${1:-}" = "--" ] && shift
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd /home/user/ninfer-fusion/build || exit 3
timeout 600 ./apps/ninfer "$M" --prompt "$P" --max-new 96 --max-context 4096 \
  --no-thinking --greedy "$@" > "$J/dl/m0_$label.log" 2>&1
rc=$?
al=$(grep -m 1 -E 'dflash2 acceptance length|mtp acceptance length' "$J/dl/m0_$label.log" | grep -oE '[0-9.]+ tok/round')
ac=$(grep -m 1 -E 'dflash2 acceptance rate|mtp acceptance rate' "$J/dl/m0_$label.log" | grep -oE '[0-9.]+%')
p=$(grep -m 1 -E 'dflash2 accepted by pos|mtp accepted by pos' "$J/dl/m0_$label.log" | sed 's/.*pos *//')
sp=$(grep -m 1 'decode speed' "$J/dl/m0_$label.log" | grep -oE '[0-9.]+ tok/s')
rd=$(grep -m 1 -E 'dflash2 rounds|mtp rounds' "$J/dl/m0_$label.log" | grep -oE '[0-9]+')
line=$(printf "  %-10s cols=%-3s AL=%-14s rate=%-8s tok/s=%-12s rounds=%-5s pos=%s" \
  "$label" "$cols" "${al:-?}" "${ac:-?}" "${sp:-?}" "${rd:-?}" "${p:-?}")
echo "$line"
echo "$line" >> "$J/dl/mvp0_width.log"
[ "$rc" != 0 ] && echo "  (rc=$rc，见 m0_$label.log 首行)" && head -3 "$J/dl/m0_$label.log" | tail -1
exit 0
