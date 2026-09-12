#!/bin/bash
R=/home/user/ninfer-fusion
echo '=== target_feature_layers 的消费点 ==='
grep -rn 'target_feature_layers' $R/src --include=*.h --include=*.cpp --include=*.cu 2>/dev/null | grep -v '\.orig' | cut -c1-165
echo
echo '=== 消费处上下文（取第一处定义外引用）==='
F=$(grep -rln 'target_feature_layers' $R/src --include=*.h --include=*.cpp 2>/dev/null | grep -v 'config.h' | grep -v '\.orig' | head -1)
echo "  文件: $F"
if [ -n "$F" ]; then
  N=$(grep -n 'target_feature_layers' "$F" | head -1 | cut -d: -f1)
  awk -v s=$((N-18)) -v e=$((N+12)) 'NR>=s && NR<=e {printf "%5d| %s\n", NR, $0}' "$F" | cut -c1-160
fi
echo
echo '=== dflash2 的层选择（config.h 143 之后）==='
awk 'NR>=143 && NR<=165 {printf "%4d| %s\n", NR, $0}' $R/src/targets/qwen3_6_27b/impl/config.h | cut -c1-150
