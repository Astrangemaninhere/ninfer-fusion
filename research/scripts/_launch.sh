#!/bin/bash
# 标准后台发射器：setsid + nohup（与 wsl.exe 客户端完全解耦，断开不杀）
# 用法: bash launch.sh <标签> <脚本路径>
set -u
label="${1:?}"
script="${2:?}"
mkdir -p /home/user/bg
out="/home/user/bg/${label}.out"
: > "$out"
setsid nohup /bin/bash "$script" >> "$out" 2>&1 < /dev/null &
pid=$!
echo "launched label=$label pid=$pid out=$out"
sleep 2
if kill -0 "$pid" 2>/dev/null; then echo "存活 ✓"; else echo "立即退出 ✗（见 $out）"; fi
# 会话 id 与父进程（验证已脱离：应为新 sid、ppid=1 或无关进程）
ps -o pid,sid,ppid,cmd -p "$pid" 2>/dev/null | tail -1
