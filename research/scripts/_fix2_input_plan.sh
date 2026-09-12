#!/bin/bash
R=/home/user/ninfer-fusion/src/ops/gdn_input_proj
echo "=== 计划枚举与 T 分派点 ==="
grep -rnE 'ScheduleId::|tokens <=|tokens >|tokens ==' $R/nvfp4/nvfp4_gdn_input_plan.cpp $R/nvfp4/nvfp4_gdn_input_plan.h 2>/dev/null | head -20
echo "=== 文件清单（已知目录，非扫盘） ==="
ls -1 $R/nvfp4/
echo "=== 是否有 gemv/decode 路由残留 ==="
grep -rnE 'Gemv|Decode|decode' $R/nvfp4/nvfp4_gdn_input_plan.cpp 2>/dev/null | head -10
echo "=== 构建 ==="
pgrep -c nvcc
