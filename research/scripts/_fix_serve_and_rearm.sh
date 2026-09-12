#!/bin/bash
# ① 取证并清掉残留的 ninfer-serve（抢 VRAM，会让后续测量失真）
# ② 重新挂上三路对照（它上次因 mtime 门禁提前开火而中止）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
echo '=== ① ninfer-serve 取证 ==='
for p in $(pgrep -x ninfer-serve); do
  echo "  PID $p  启动 $(ps -o lstart= -p $p | xargs)  已跑 $(ps -o etimes= -p $p)s"
  echo "    父进程: $(ps -o ppid= -p $p | xargs)  命令行:"
  tr '\0' ' ' < /proc/$p/cmdline | cut -c1-160 | sed 's/^/      /'
  echo
done
echo '--- 显存占用 ---'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null | head -6

# 只在确认是"孤儿/残留"时杀：父进程为 1 或父进程已不存在
KILLED=0
for p in $(pgrep -x ninfer-serve); do
  PP=$(ps -o ppid= -p $p | tr -d ' ')
  if [ "$PP" = "1" ] || ! ps -p "$PP" >/dev/null 2>&1; then
    echo "  -> PID $p 是孤儿（父=$PP），杀掉"
    kill -TERM $p 2>/dev/null; KILLED=1
  else
    echo "  -> PID $p 有活父进程 $PP（$(ps -o comm= -p $PP)），保留不杀"
  fi
done
[ $KILLED = 1 ] && sleep 3
echo '--- 处理后显存 ---'
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv 2>/dev/null

echo
echo '=== ② 重新挂三路对照（等构建停 -> mtime 门禁 -> 三路 diff）==='
setsid nohup bash /home/user/pfv2.sh >/dev/null 2>&1 &
sleep 3
pgrep -af 'pfv2[.]sh' | cut -c1-70
echo
echo '=== 构建现状 ==='
echo "  nvcc: $(pgrep -c -x nvcc 2>/dev/null || echo 0)  进度: $(grep -oE '\[[ 0-9]+%\]' "$J/dl/par_build.log" 2>/dev/null | tail -1)"
free -g | sed -n '2,3p'
date +%H:%M:%S
