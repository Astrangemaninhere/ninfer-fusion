#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
B=$J/_collab/build
echo "=== 1) dl_0290 轮子 ==="
ls -l --time-style=+%m-%d_%H:%M $B/dl_0290 2>/dev/null
echo
echo "=== 2) df2_src 抽取产物 ==="
ls -l --time-style=+%m-%d_%H:%M $B/df2_src 2>/dev/null
echo
echo "=== 3) build 目录下其它 z2 产物 ==="
ls -1 $B 2>/dev/null | head -20
echo
echo "=== 4) WSL 里是否已有 vllm 环境（定点候选 + pip 列表） ==="
for d in /home/user/vllm029 /home/user/venv-vllm /home/user/vllm /home/user/venv /home/user/.venv /home/user/vllm-0.29; do
  [ -d "$d" ] && echo "  存在: $d  ($(ls -1 "$d" 2>/dev/null | head -3 | tr '\n' ' '))"
done
python3 -c "import vllm, sys; print('WSL 系统 python 已有 vllm', vllm.__version__)" 2>&1 | tail -1
echo
echo "=== 5) z2 的 reg 脚本结论（是否已找到注册点） ==="
tail -25 $J/_z2_reg2.sh 2>/dev/null
