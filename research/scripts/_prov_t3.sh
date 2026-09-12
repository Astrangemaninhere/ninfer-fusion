#!/bin/bash
Z=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== WSL models dir (dflash2 artifacts) ==="
ls -la /home/user/models/ 2>/dev/null | grep -iE 'dflash|dspark|nvfp4' | head -20
echo
echo "=== md5 of served artifact vs windows copy ==="
md5sum /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer 2>/dev/null
md5sum "$Z/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer" 2>/dev/null
echo
echo "=== train-dflash2-launch.log (Sep 3) args ==="
tr -d '\r' < $Z/dl/train-dflash2-launch.log 2>/dev/null | grep -iE 'steps=|target|shift|mask|anchors|python|\.py' | head -20
echo
echo "=== train-dflash2.log tail: last 25 lines ==="
tr -d '\r' < $Z/dl/train-dflash2.log 2>/dev/null | tail -25
echo
echo "=== any launcher .bat referencing train_dflash2 ==="
grep -rl 'train_dflash2' $Z/*.bat $Z/*.sh $Z/dl/*.sh 2>/dev/null | head -20
