#!/bin/bash
# ① 清掉两个僵尸观察者（它们等的标记 BATCH_BUILD_NEXT_DONE 永远不会来了）
# ② 直接跑 make（不用看门狗：剩余活是设备链接+链接+serve，单峰内存、单步，
#    上一轮就是被 2 GB 阈值误杀在设备链接上）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
export PATH="/home/user/.local/bin:$PATH"
export NVCC_PREPEND_FLAGS="--split-compile-extended=8"

echo '=== ① 僵尸观察者取证与清理 ==='
for pat in 'probe_arm[.]sh' 'reconfig_par[.]sh' 'batch_next[.]sh' 'pfv2[.]sh'; do
  for p in $(pgrep -f "$pat"); do
    echo "  PID $p: $(ps -o args= -p $p | cut -c1-70)"
    kill $p 2>/dev/null && echo "    -> 已停"
  done
done
sleep 2
echo "  剩余观察者: $(pgrep -fc 'probe_arm|reconfig_par|batch_next|pfv2' 2>/dev/null || echo 0)"

echo
echo '=== ② 直跑 make（无看门狗）==='
cd "$R/build" || exit 3
for t in ninfer ninfer-serve; do
  echo "--- make $t -j2  $(date +%H:%M:%S) ---"
  make "$t" -j2 2>&1 | tail -25
  rc=${PIPESTATUS[0]}
  echo "  $t rc=$rc"
done
echo '--- 产物 ---'
ls -la --time-style=+%m-%d_%H:%M "$R/build/apps/ninfer" "$R/build/apps/ninfer-serve" 2>/dev/null | cut -c25-80
echo '=== make 结束 (无看门狗) ==='
