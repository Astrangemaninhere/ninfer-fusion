#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== Windows 侧 models/ 下的底模 artifact ==="
ls -lt --time-style=+%m-%d_%H:%M "$J/models" 2>/dev/null | head -8
echo
echo "=== DFlash2 底模目录 ==="
ls -l --time-style=+%m-%d_%H:%M "$J/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2" 2>/dev/null | head -8
echo
echo "=== 转换脚本：权威树 vs 镜像 ==="
for d in /home/user/ninfer-fusion/tools/convert/qwen3_8_27b "$J/ninfer-fusion-repo/tools/convert/qwen3_8_27b"; do
  echo "--- $d"
  ls -1 "$d" 2>/dev/null | grep -iE 'patch|verify|roundtrip|dflash2' | head
done
echo
echo "=== 是否有 roundtrip 脚本 ==="
ls -1 $J/_dflash2_roundtrip_tmp.py $J/ninfer-fusion-repo/tools/convert/qwen3_8_27b/*.py 2>/dev/null | head -10
