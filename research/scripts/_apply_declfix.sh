#!/bin/bash
# 只做必要动作：落 N3 的声明归位补丁 -> 校验 -> touch 让 decoder_state.cpp 重编 -> 重启 -j8 构建。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
export PATH="/home/user/.local/bin:$PATH"
H=$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h

echo '=== 落补丁 ==='
patch -p1 -b -d "$R" < /tmp/declfix.diff
echo "  修后 md5 $(md5sum "$H" | cut -c1-32)"

echo
echo '=== 校验：声明位置（顶格=namespace 作用域）==='
grep -n 'kv_rowscale_sidecar_apply_from_env' "$H" | cut -c1-115

echo
echo '=== 确认它不再落在 check 的函数体内 ==='
python3 -c "
import pathlib
s = pathlib.Path('$H').read_text(encoding='utf-8', errors='surrogateescape')
lines = s.splitlines()
depth = 0
for i, l in enumerate(lines, 1):
    if 'kv_rowscale_sidecar_apply_from_env' in l and l.lstrip().startswith('bool '):
        print('  行 %d 声明，所在函数体深度 = %d （0 = namespace 作用域，期望 0）' % (i, depth))
    depth += l.count('{') - l.count('}')
"
echo
echo '=== touch 触发重编 + 重启 -j8 构建 ==='
touch "$R/src/targets/qwen3_6/impl/state/decoder_state.cpp"
while pgrep -x nvcc >/dev/null 2>&1; do sleep 10; done
nohup setsid bash "$J/_par_build.sh" ninfer ninfer-serve >/dev/null 2>&1 &
sleep 25
echo "  nvcc 并行数: $(pgrep -c -x nvcc 2>/dev/null || echo 0)"
tail -4 "$J/dl/par_build.log" | cut -c1-120
date +%H:%M:%S
