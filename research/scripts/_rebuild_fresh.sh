#!/bin/bash
# 重编 + 新鲜度断言：杜绝"源码改了但二进制陈旧 ⇒ 静默测到旧行为"
set -u
R=/home/user/ninfer-fusion
export PATH=/home/user/.local/bin:$PATH
cd "$R/build" || exit 3
echo "=== 构建目标确认 ==="
make help 2>/dev/null | grep -iE 'ninfer|apps' | head -10
echo
echo "=== 重编 起 $(date '+%H:%M:%S') ==="
/usr/bin/time -f 'BUILD_WALL=%es' make -j8 ninfer 2>&1 | tail -25
rc=${PIPESTATUS[0]}
echo "make rc=$rc  终 $(date '+%H:%M:%S')"
echo
echo "=== 新鲜度断言（二进制必须比所有源新）==="
BIN=$R/build/apps/ninfer
newest=$(find $R/src -name '*.h' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' | xargs stat -c '%Y %n' 2>/dev/null | sort -rn | head -1)
echo "最新源: $newest"
echo "二进制: $(stat -c '%Y %y' $BIN)"
bs=$(stat -c '%Y' $BIN); ss=${newest%% *}
if [ "$bs" -ge "$ss" ]; then echo "FRESH_OK 二进制比最新源新"; else echo "FRESH_FAIL 二进制陈旧！差 $((ss-bs))s"; fi
echo
echo "=== config.h 里的有效 mask_token（DFlash2Config） ==="
sed -n '119,151p' $R/src/targets/qwen3_6_27b/impl/config.h | grep -n 'mask_token'
echo REBUILD_DONE
