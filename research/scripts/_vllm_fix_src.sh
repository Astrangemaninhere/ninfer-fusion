#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) dl/ 下的 wheel（单目录，找 vllm 轮子来源） ==="
ls -lt --time-style=+%m-%d_%H:%M $J/dl/*.whl 2>/dev/null | head -12
echo
echo "=== 2) venv-clean 的 pip 配置/索引（看当初从哪装） ==="
for f in pip.ini pip.conf; do
  [ -f "/mnt/c/vllm/venv-clean/$f" ] && { echo "--- $f"; cat "/mnt/c/vllm/venv-clean/$f"; }
  [ -f "/mnt/c/Users/User/AppData/Roaming/pip/$f" ] && { echo "--- AppData pip/$f"; cat "/mnt/c/Users/User/AppData/Roaming/pip/$f"; }
done
echo "--- pip config list ---"
"/mnt/c/vllm/venv-clean/Scripts/python.exe" -m pip config list 2>&1 | head -8
echo
echo "=== 3) 0.26.0+cu132 的 dist-info 里是否记录了安装来源 ==="
D=$(ls -d /mnt/c/vllm/venv-clean/Lib/site-packages/vllm-0.26.0* 2>/dev/null)
ls "$D" 2>/dev/null | head
grep -m3 -iE 'direct_url|url|index' "$D/direct_url.json" 2>/dev/null
echo
echo "=== 4) 0.29 是否真的缺编译扩展（单目录 ls） ==="
ls -1 /mnt/c/vllm/venv-clean/Lib/site-packages/vllm/*.pyd /mnt/c/vllm/venv-clean/Lib/site-packages/vllm/_C* 2>/dev/null | head -8
