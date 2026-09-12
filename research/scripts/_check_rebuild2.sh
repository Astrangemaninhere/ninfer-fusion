#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
bash -n "$J/_verify_equivalence.sh" && echo "VE_SYNTAX_OK" || echo "VE_SYNTAX_FAIL"
echo "--- 重建进度 ---"
tail -5 "$J/dl/rebuild_after_s35.log" | cut -c1-140
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do
  echo "  在编 $(ps -o etime= -p $p | tr -d ' ')"
  break
done
free -g | head -2
echo "--- Muse 验收脚本还在跑？ ---"
pgrep -af '_muse_serve_accept.sh' | cut -c1-70 || echo "  已结束"
