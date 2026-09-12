#!/bin/bash
# 决定性实验：把 DSpark 的目标特征层改成训练脚本的口径（1-based: 2,15,30,45,58），
# 重编后复测 dspark 接受率。若从 7.63% 跳到 ~22% 即锁定。
set -u
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6_27b/impl/config.h
export PATH="/home/user/.local/bin:$PATH"

echo '=== 改前 ==='
grep -n 'target_feature_layers' "$F" | cut -c1-140
cp -f "$F" /home/user/config_h.bak_before_feature_layers

python3 - <<'PY'
import pathlib
p = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6_27b/impl/config.h")
s = p.read_text(encoding="utf-8", errors="surrogateescape")
old = "static constexpr std::array<int, feature_layers> target_feature_layers{4, 16, 28, 40, 52};"
new = ("// [EXP] aligned to train_dspark.py TARGET_LAYER_IDS = [1,14,29,44,57] (0-indexed),\n"
       "    // which vLLM collects at +1 => {2,15,30,45,58}. Was {4,16,28,40,52}.\n"
       "    static constexpr std::array<int, feature_layers> target_feature_layers{2, 15, 30, 45, 58};")
if old in s:
    p.write_text(s.replace(old, new), encoding="utf-8", errors="surrogateescape")
    print("改好了：DSpark 层号 -> {2,15,30,45,58}")
else:
    print("没找到原行，未改动")
PY

echo '=== 改后 ==='
grep -n 'target_feature_layers' "$F" | cut -c1-140

echo
echo '=== touch 包含者 + 重编 ==='
cd "$R" || exit 1
T=$(grep -rl --include='*.cpp' --include='*.cu' -e 'dflash_impl.h' src/ 2>/dev/null | sort -u)
echo "  touch $(echo "$T" | wc -l) 个"
[ -n "$T" ] && touch $T
cd "$R/build" || exit 2
MIN_FREE_GB=0 PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 setsid nohup bash /mnt/c/Users/User/Documents/ziqinzhang/_par_build.sh ninfer ninfer-serve >/dev/null 2>&1 &
sleep 30
echo "  nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)"
date +%H:%M:%S
