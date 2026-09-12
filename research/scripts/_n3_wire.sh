#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== CLI/serve 是否有 calibration 相关旗标 ==="
grep -rnE 'calibration|calib' $R/apps/cli/options.cpp $R/src/serve/serve_options.cpp $R/src/serve/serve_options.h 2>/dev/null | head -20
echo
echo "=== EngineOptions 里的字段定义 ==="
grep -nE 'kv_calibration_dir' $R/include/ninfer/types.h $R/src/serve/serve_options.h $R/src/targets/qwen3_6/impl/runtime/layouts.h 2>/dev/null | head
echo
echo "=== 谁把 field 拷进引擎（serve/CLI 映射点） ==="
grep -rnE 'kv_calibration_dir' $R/src $R/apps 2>/dev/null | grep -v '\.orig' | head
echo
echo "=== kvcalib_capture 的调用点（prefill 路径） ==="
grep -rn 'kvcalib_capture\|kvcalib_enabled' $R/src --include=*.h --include=*.cpp 2>/dev/null | grep -v '\.orig' | head
