#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
for f in _z2_extract_df2.sh _z2_inspect.sh _z2_inspect2.sh _z2_reg.sh _z2_reg2.sh; do
  echo "########## $f ($(stat -c '%y' $J/$f 2>/dev/null | cut -c1-16), $(stat -c %s $J/$f 2>/dev/null) B) ##########"
  head -28 $J/$f 2>/dev/null
  echo
done
echo "########## WSL 里是否已有 vllm 0.29 环境（已知候选路径定点查） ##########"
for d in /home/user/vllm029 /home/user/vllm-029 /home/user/venv-vllm /home/user/vllm/venv /opt/vllm /home/user/v029; do
  [ -d "$d" ] && echo "  存在: $d"
done
ls -1 /home/user/*.txt /home/user/*.log 2>/dev/null | head -8
echo "--- wh_0290 wheelhouse 内容（单目录）"
ls -l --time-style=+%m-%d_%H:%M $J/_collab/build/wh_0290 2>/dev/null | head -8
