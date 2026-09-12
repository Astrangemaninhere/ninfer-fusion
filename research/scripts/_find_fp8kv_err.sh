#!/bin/bash
# 定位 fp8-KV 报错点及其期望的 scale 规格（定点 grep）
R=/home/user/ninfer-fusion
echo "=== 1) 报错字符串所在文件 ==="
grep -rln 'invalid NVFP4 KV cache scale dtype' $R/src 2>/dev/null | head -5
echo
echo "=== 2) 该处的上下文 ==="
F=$(grep -rl 'invalid NVFP4 KV cache scale dtype' $R/src 2>/dev/null | head -1)
if [ -n "$F" ]; then
  echo "--- $F"
  grep -n -B 25 'invalid NVFP4 KV cache scale dtype' "$F" | head -45
fi
echo
echo "=== 3) KvCacheStorage::Fp8E4M3Row256 的用法点（谁在用 fp8 KV） ==="
grep -rn 'Fp8E4M3Row256' $R/src $R/include 2>/dev/null | grep -v '\.orig' | head -12
