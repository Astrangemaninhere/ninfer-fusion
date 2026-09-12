#!/bin/bash
# 判决：当前二进制下，MTP k=3 给/不给 --graph-capture-ceiling 16 的差异
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
OUT=$J/_ceiling_verdict.txt
{
  echo "=== 判决 起 $(date '+%H:%M:%S')  宿主: $(uptime | sed 's/.*load average/load/') ==="
  cd /home/user/ninfer-fusion/build || exit 3
  pkill -9 -x ninfer 2>/dev/null; sleep 2
  printf "  %-32s %-10s %-8s %s\n" 配置 tok/s AL 备注
  for cc in "" "16"; do
    flag=""
    [ -n "$cc" ] && flag="--graph-capture-ceiling $cc"
    lbl=$([ -n "$cc" ] && echo "显式 ceiling=$cc" || echo "默认（不给 flag）")
    timeout 600 ./apps/ninfer "$M" --prompt "$P" --max-new 64 --max-context 4096 \
      --no-thinking --greedy --spec mtp --draft-tokens 3 --lm-head-draft $flag \
      > "$J/cv_${cc:-def}.log" 2>&1
    rc=$?
    sp=$(grep -m1 'decode speed' "$J/cv_${cc:-def}.log" | grep -oE '[0-9.]+')
    al=$(grep -m1 'mtp acceptance length' "$J/cv_${cc:-def}.log" | grep -oE '[0-9.]+')
    note="rc=$rc"
    [ "$rc" != "0" ] && note="$note $(tail -c 60 "$J/cv_${cc:-def}.log" | tr '\n' ' ')"
    printf "  %-32s %-10s %-8s %s\n" "$lbl" "${sp:-?}" "${al:-?}" "$note"
  done
  echo "宿主结束: $(uptime | sed 's/.*load average/load/')"
} > "$OUT" 2>&1
echo written
