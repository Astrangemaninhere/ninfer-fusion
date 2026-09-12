#!/bin/bash
# ① 精确识别"当前这一轮"nvcc 正在编哪个 TU（下结论前必须有据）
# ② ccache 上限按盘容量放开（对象最大 194 MB，5 GB 必然抖动）
# ③ 记录本轮构建起点，供事后精确算单 TU 耗时
echo '=== 当前 nvcc 的精确命令行与启动时间 ==='
for p in $(pgrep -x nvcc); do
  echo "  PID $p  启动 $(ps -o lstart= -p $p | xargs)"
  tr '\0' ' ' < /proc/$p/cmdline | tr ' ' '\n' | grep -E '\.cu|\.cpp|^-o$|nvfp4|gqa|variant|engine' | head -8
  echo "  --- 其子进程（ptxas 等）---"
  ps --ppid $p -o pid,rss,etime,comm 2>/dev/null | tail -4
done
echo
echo '=== 本轮 make 的起止线索 ==='
tail -3 /mnt/c/Users/User/Documents/ziqinzhang/dl/rebuild_after_s35.log | cut -c1-120
find /home/user/ninfer-fusion/build -name '*.o' -newermt '2026-09-10 19:25' -printf '%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | awk '{printf "  %s  %8.1f MB  %s\n", substr($1,1,8), $2/1048576, $3}' | head -10
echo
echo '=== 磁盘与 ccache 上限 ==='
df -h /home | tail -1
ccache --max-size=50G 2>&1
ccache -s | grep -E 'Cache size|max_size' | head -3
ccache -p | grep -i max_size
