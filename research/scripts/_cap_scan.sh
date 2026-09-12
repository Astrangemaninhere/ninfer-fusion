#!/bin/bash
# 判据：接受率是否随 max-context/kv-capacity 变化（若变 => 命中 V2 指出的"块宽两个来源"缺陷）
cd /home/user/ninfer-fusion/build || exit 1
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
for cfg in 4096 2048 8192; do
  /home/user/ninfer_pre_fix /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
    --prompt "$P" --max-new 96 --max-context $cfg --kv-capacity $cfg \
    --no-thinking --greedy --spec dflash2 --draft-tokens 7 > /home/user/cap_$cfg.log 2>&1
  acc=$(grep -oE 'dflash2 acceptance rate +[0-9.]+%' /home/user/cap_$cfg.log | head -1)
  pos=$(grep -oE 'accepted by pos +[0-9,]+' /home/user/cap_$cfg.log | tail -1)
  tpr=$(grep -oE '[0-9.]+tok/round' /home/user/cap_$cfg.log | head -1)
  spd=$(grep -oE 'decode speed +[0-9.]+' /home/user/cap_$cfg.log | head -1)
  printf '  ctx/kv=%-5s rc=%s  %-46s  %-26s %s %s\n' "$cfg" "$?" "$acc" "$pos" "$tpr" "$spd"
done
