#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo '=== R1/R2a 测试结果 ==='
cat "$J/dl/r1r2.log" 2>/dev/null | tail -14 | cut -c1-120
echo
echo '=== 重训脚本内容（看清它怎么启动、跑多久、输出到哪）==='
cat "$J/_train_df2_shift0.bat" 2>/dev/null | head -30 | cut -c1-150
echo
echo '=== 训练日志现状（是否已有进度）==='
tail -4 /home/user/dl/train-dflash2.log 2>/dev/null | cut -c1-120 || tail -4 "$J/dl/train-dflash2.log" 2>/dev/null | cut -c1-120 || echo '  （未找到）'
ls -la --time-style=+%m-%d_%H:%M "$J/dl/"*train* 2>/dev/null | cut -c25-95 | head -6
echo
echo '=== GPU 是否空闲 ==='
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv 2>/dev/null
pgrep -af 'apps/ninfer' | head -3 | cut -c1-80
