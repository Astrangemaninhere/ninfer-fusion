#!/bin/bash
# 紧急核查：树里是否真有一个"未定义符号"的编译错误（S6 报告 R7 的警报），
# 以及我正在跑的构建会不会撞上它。
R=/home/user/ninfer-fusion
B=$R/build
J=/mnt/c/Users/User/Documents/ziqinzhang

echo '=== ① 那个符号在哪儿定义/声明 ==='
grep -rn 'kv_rowscale_sidecar_apply_from_env' "$R/src" 2>/dev/null | head -10 | cut -c1-140
echo
echo '=== ② decoder_state.cpp:165-180 现场 ==='
awk 'NR>=165 && NR<=180 {printf "%5d| %s\n", NR, $0}' "$R/src/targets/qwen3_6/impl/state/decoder_state.cpp" | cut -c1-130
echo
echo '=== ③ 该 TU 是否已被本次构建编译过（对象比源新？）==='
OBJ=$(find "$B" -name 'decoder_state.cpp.o' 2>/dev/null | head -1)
SRC=$R/src/targets/qwen3_6/impl/state/decoder_state.cpp
ls -la --time-style=+%m-%d_%H:%M:%S "$SRC" $OBJ 2>/dev/null | cut -c25-95
echo
echo '=== ④ 本次构建有没有报错 / 进度 / 是否在链接 ==='
echo "  错误条数: $(grep -cE ' error:' $J/dl/par_build.log 2>/dev/null)"
grep -E ' error:' "$J/dl/par_build.log" 2>/dev/null | head -5 | cut -c1-150
grep -oE '\[[ 0-9]+%\]' "$J/dl/par_build.log" 2>/dev/null | tail -1
echo "  nvcc: $(pgrep -c -x nvcc 2>/dev/null || echo 0)   ninfer-serve 进程: $(pgrep -c -x ninfer-serve 2>/dev/null || echo 0)"
echo
echo '=== ⑤ 树上未落地的 staged 目录（谁可能已被应用）==='
ls -la --time-style=+%H:%M "$J/_collab/build/staged/" 2>/dev/null | head -4 | cut -c25-90
ls -1 "$J/_collab/build/staged2/" 2>/dev/null | head -15
echo
echo '=== ⑥ 二进制与库的时间戳 ==='
ls -la --time-style=+%m-%d_%H:%M "$B/apps/ninfer" "$B/lib/libninfer_ops.a" "$B/lib/libninfer_engine.a" 2>/dev/null | cut -c25-90
