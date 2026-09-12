#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 上次 pip 的真实报错 ==="
grep -nE 'ERROR|error|Could not|timed out|Timeout|No matching|SSL|proxy' $J/dl/redo2.log 2>/dev/null | head -12
echo
echo "=== 2) PyPI 可达性（探端点） ==="
CURL="/mnt/c/Windows/System32/curl.exe"
for u in https://pypi.org/simple/torch/ https://pypi.tuna.tsinghua.edu.cn/simple/torch/ https://mirrors.aliyun.com/pypi/simple/torch/; do
  code=$(timeout 25 "$CURL" -s -o /dev/null -w '%{http_code}' -L "$u" 2>/dev/null)
  echo "  $u -> $code"
done
echo
echo "=== 3) 目标模型目录（vLLM 要读的 HF 目录） ==="
ls -1 $J/models/Qwen3.8-27B-NVFP4-RTX5090 2>/dev/null | head -8
ls -1 $J/models/Qwen3.8-27B-NVFP4-RTX5090/*.json 2>/dev/null | head -4
