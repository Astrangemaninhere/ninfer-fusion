#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== kv_calibration_dir / env 加载器 / 旁车 ==="
grep -rnE 'kv_calibration_dir|NINFER_KV_CALIB|kv_calibration' $R/include $R/src $R/apps --include=*.h --include=*.cpp --include=*.cu 2>/dev/null | grep -v '\.orig' | head -25
echo
echo "=== kv_calibration.h 结构 ==="
grep -nE '^(class|struct|inline|void|bool|std::|template|namespace|//)' $R/src/targets/qwen3_6/impl/runtime/kv_calibration.h 2>/dev/null | head -40
echo
echo "=== 行数 ==="
wc -l $R/src/targets/qwen3_6/impl/runtime/kv_calibration.h 2>/dev/null
