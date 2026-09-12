#!/bin/bash
# 等拆分编译绿 → 拷 artifact → 用项目 harness 跑两轮 A/B + 我口径的 zh 贪心三路
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
LOG=$J/dl/df2_ab_chain.log
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "=== 换头 A/B 链  $(date '+%F %H:%M:%S') ==="

for i in $(seq 1 480); do
  grep -qE 'BUILD_SPLIT_OK|BUILD_SPLIT_FAIL' $J/dl/build_split.log 2>/dev/null && break
  sleep 5
done
grep -m1 'BUILD_SPLIT_OK' $J/dl/build_split.log 2>/dev/null || { echo "编译未绿，中止"; exit 2; }
echo "编译绿: $(date '+%H:%M:%S')  $(ls -l --time-style=+%H:%M $R/build/apps/ninfer | awk '{print $5, $6}')"

mkdir -p /home/user/models
CUR=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
for CK in step_001900 step_000200; do
  S=$J/data/dflash2_ckpts/${CK}_tuned.ninfer
  D=/home/user/models/qwen3_8_27b_nvfp4_dflash2_${CK}.ninfer
  if [ ! -f "$D" ] || [ "$(stat -c %s "$D" 2>/dev/null)" != "$(stat -c %s "$S")" ]; then
    echo "拷贝 $CK ..."; cp -f "$S" "$D" || { echo 拷贝失败; exit 3; }
  fi
  echo "  $(ls -l --time-style=+%H:%M $D | awk '{print $5, $6, $7}')"
done

echo
echo "########## 项目 harness：当前 vs 新炉 step_000200 ##########"
cd "$J" || exit 4
NINFER_BIN=$R/build/apps/ninfer bash _df2_ab.sh "A=$CUR" "B=/home/user/models/qwen3_8_27b_nvfp4_dflash2_step_000200.ninfer" 2>&1 | tail -16
echo
echo "########## 项目 harness：当前 vs 老炉 step_001900 ##########"
NINFER_BIN=$R/build/apps/ninfer bash _df2_ab.sh "A=$CUR" "B=/home/user/models/qwen3_8_27b_nvfp4_dflash2_step_001900.ninfer" 2>&1 | tail -16

echo
echo "########## 我口径：zh 贪心三路（与历史 4.51/4.81% 可比） ##########"
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd "$R/build" || exit 5
for tag in cur s0200 s1900; do
  case $tag in
    cur)   ART=$CUR ;;
    s0200) ART=/home/user/models/qwen3_8_27b_nvfp4_dflash2_step_000200.ninfer ;;
    s1900) ART=/home/user/models/qwen3_8_27b_nvfp4_dflash2_step_001900.ninfer ;;
  esac
  timeout 900 ./apps/ninfer "$ART" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --spec dflash2 --print-token-ids > "$J/dl/abz_$tag.log" 2>&1
  echo "[$tag] rc=$?  $(grep -m1 'acceptance rate' $J/dl/abz_$tag.log | grep -oE '[0-9.]+%')  \
pos=$(grep -m1 'accepted by pos' $J/dl/abz_$tag.log | sed 's/.*pos *//')  \
$(grep -m1 'decode speed' $J/dl/abz_$tag.log | grep -oE '[0-9.]+ tok/s')"
done
echo DF2_AB_CHAIN_DONE
