#!/bin/bash
# 18:00 汇报：只读采集（不启动构建/训练，不杀进程）
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "########## 1. 补丁 A 实测（postbuild_measure） ##########"
grep -E 'stage |chars|IDENTICAL|DIFFERENT|INVALID|POSTBUILD' "$J/dl/postbuild_measure.log" 2>/dev/null | tail -20 | cut -c1-150
echo
echo "########## 2. 接受率（pb_rows + 引擎字段） ##########"
cat /tmp/pb_rows 2>/dev/null | sed 's/^/  row: /'
for f in /home/user/pb_dflash2.log /home/user/pb_mtp3.log; do
  echo "  --- $(basename $f) ---"
  grep -E 'done .*speculative' "$f" 2>/dev/null | tail -2 | \
    grep -oE 'prompt=[0-9]+|gen=[0-9]+|finish=[a-z_]+|spec_drafted=[0-9]+|spec_accepted=[0-9]+|spec_accept_rate=[0-9.]+|spec_rounds=[0-9]+|spec_fallback_steps=[0-9]+|decode=[0-9.]+tok/s' | tr '\n' ' '
  echo
done
echo
echo "########## 3. K4 队列进度（四路 / 位置剖面 / 32-needle / Muse） ##########"
grep -E '^\-\-\- \[|needle hits|accepted by pos|VERDICT|MUSE_|resumed|WINDK4' "$J/dl/window_k4.log" 2>/dev/null | tail -18 | cut -c1-160
echo
echo "########## 4. 训练 ##########"
tail -3 "$J/dl/train-dflash2.log" 2>/dev/null | cut -c1-140 || echo "  （无训练日志）"
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { '  training pid ' + \$_.ProcessId }" 2>/dev/null | tr -d '\r'
echo
echo "########## 5. 下载 ##########"
tail -2 "$J/dl/hf-chunk.log" 2>/dev/null | cut -c1-120
tail -2 "$J/dl/spark_weights.log" 2>/dev/null | cut -c1-120
du -sh "$J/models/Spark-X2.5-4B" 2>/dev/null | sed 's/^/  Spark: /'
echo
echo "########## 6. 内存 / OOM ##########"
free -g | head -2
wsl.exe -e bash -lc "journalctl -b --no-pager 2>/dev/null | grep -c global_oom" 2>/dev/null || echo "  （OOM 计数不可用）"
echo
echo "########## 7. S55 构建修复记录 ##########"
grep -cE '^\| *[0-9]' "$J/_collab/S55_build_repair.md" 2>/dev/null | sed 's/^/  修复条目: /'
tail -6 "$J/_collab/S55_build_repair.md" 2>/dev/null | cut -c1-140
