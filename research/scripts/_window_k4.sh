#!/bin/bash
# Window K4 — 重建之后重排 GPU 队列（K3 已被我停掉：它会用旧二进制跑判定）。
# 顺序：等重建完成 → 等补丁 A 实测(exactness+接受率)完成 → 四路对比 + CLI 位置剖面 → Muse → 最后续训。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
LOG=$J/dl/window_k4.log
: > "$LOG"
exec >> "$LOG" 2>&1
echo "=== window K4 start $(date +%F' '%H:%M:%S) ==="

echo "--- [1/5] 等重建结束 $(date +%H:%M:%S) ---"
t0=$(date +%s)
while ! grep -q 'rebuild done' "$J/dl/rebuild_after_s35.log" 2>/dev/null; do
  sleep 20
  [ $(( $(date +%s) - t0 )) -gt 7200 ] && { echo "TIMEOUT 等重建"; break; }
done
tail -8 "$J/dl/rebuild_after_s35.log" | cut -c1-150

echo "--- [2/5] 二进制必须新于补丁 A $(date +%H:%M:%S) ---"
BIN=$R/build/apps/ninfer-serve
PA=$(stat -c %Y $R/src/targets/qwen3_6/impl/runtime/program_impl.h)
bm=$(stat -c %Y "$BIN" 2>/dev/null || echo 0)
if [ "$bm" -gt "$PA" ]; then
  echo "  OK: serve $(date -d @$bm '+%m-%d %H:%M:%S') > 补丁A $(date -d @$PA '+%m-%d %H:%M:%S')"
else
  echo "  !! serve 仍旧（$(date -d @$bm '+%m-%d %H:%M:%S')）——测量无意义，仍继续但判读要打折"
fi

echo "--- [3/5] 等补丁 A 实测 $(date +%H:%M:%S) ---"
t0=$(date +%s)
while ! grep -qE 'POSTBUILD_(DONE|TIMEOUT)' "$J/dl/postbuild_measure.log" 2>/dev/null; do
  sleep 20
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && { echo "TIMEOUT 等实测"; break; }
done
grep -E 'IDENTICAL|DIFFERENT|SERVE_FAILED' "$J/dl/postbuild_measure.log" 2>/dev/null | tail -8
cat /home/user/pb_verdict.txt 2>/dev/null | head -16

echo "--- [4/5] 四路对比 + 32-needle 长上下文 $(date +%H:%M:%S) ---"
bash "$J/_spec_4way.sh" 2>&1 | tail -45

echo "--- [5/5] Muse 验收，然后最后续训 $(date +%H:%M:%S) ---"
bash "$J/_muse_serve_accept.sh" 2>&1 | tail -25
if [ -f "$J/_train_df2_resume.bat" ]; then
  powershell.exe -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','C:\Users\User\Documents\ziqinzhang\_train_df2_resume.bat' -WindowStyle Hidden"
  sleep 60
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { 'resumed pid ' + \$_.ProcessId }" 2>&1 | tr -d '\r'
fi
echo "=== window K4 done $(date +%F' '%H:%M:%S) ==="
echo WINDK4_DONE
