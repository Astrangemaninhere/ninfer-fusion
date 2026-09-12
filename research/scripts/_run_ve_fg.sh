#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 前置检查（脚本里会因这些条件 exit 3） ==="
printf "  train_dflash2 在跑? "; pgrep -f 'train_dflash2' >/dev/null 2>&1 && echo YES || echo no
printf "  ninfer-serve 在跑? "; pgrep -f 'ninfer-serve' >/dev/null 2>&1 && echo YES || echo no
printf "  CLI 存在? "; [ -x /home/user/ninfer-fusion/build/apps/ninfer ] && echo YES || echo "NO (!)"
printf "  模型存在? "; [ -f /home/user/models/qwen3_8_27b_nvfp4.ninfer ] && echo YES || echo "NO (!)"
echo "  显存: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo
echo "=== 直接前台跑（能看到全部输出；不重定向） ==="
bash "$J/_verify_equivalence.sh" 2>&1 | tail -30 | cut -c1-200
