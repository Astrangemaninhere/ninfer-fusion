#!/bin/bash
# 单变量 A/B：DFlash2Config::mask_token 248070 -> 248077
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
CFG=$R/src/targets/qwen3_6_27b/impl/config.h
BAK=/home/user/masktok_bak
LOG=$J/dl/mask_token_ab.log
mkdir -p "$BAK"
exec > >(tee -a "$LOG") 2>&1
echo "=== mask_token A/B $(date '+%F %H:%M:%S') ==="

echo "--- 0) 改前状态 ---"
grep -n 'mask_token' "$CFG" | sed 's/^/  /'
cp -f "$CFG" "$BAK/config.h"
echo "  backup md5=$(md5sum $BAK/config.h | cut -c1-12)"
cp -f "$CFG" "$BAK/config.h.orig2"

echo "--- 1) 只改 dflash2 那处（248070 -> 248077） ---"
python3 - <<'PYEOF'
import pathlib, hashlib
F = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6_27b/impl/config.h")
t = F.read_text()
old = "static constexpr int mask_token = 248070;"
n = t.count(old)
print("  锚点命中 =", n)
assert n == 1, "锚点必须唯一"
F.write_text(t.replace(old, "static constexpr int mask_token = 248077;"))
print("  md5 now", hashlib.md5(F.read_bytes()).hexdigest()[:12])
PYEOF
grep -n 'mask_token' "$CFG" | sed 's/^/  /'

echo "--- 2) 编译 ---"
for inc in $(grep -rl 'qwen3_6_27b/impl/config.h' $R/src 2>/dev/null | grep -v '\.orig' | sed "s|$R/||"); do
  case "$inc" in *.cpp|*.cu) touch "$R/$inc";; esac
done
cd "$R/build" || exit 4
make ninfer -j3 2>&1 | tail -3
rc=${PIPESTATUS[0]}; echo "make rc=$rc"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; cp -f $BAK/config.h "$CFG"; exit 5; }

echo "--- 3) 跑（同 prompt，DF2DBG 开） ---"
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
NINFER_DF2DBG=1 timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 \
  > $J/dl/masktok_run.log 2>&1
echo "  rc=$?"
grep -E 'dflash2 acceptance rate|dflash2 accepted by pos|dflash2 acceptance length|decode speed' $J/dl/masktok_run.log | sed 's/^/  /'

echo "--- 4) 回滚（保留结论，恢复配置）---"
cp -f $BAK/config.h "$CFG"
echo "  restored md5=$(md5sum $CFG | cut -c1-12) (期望 $(md5sum $BAK/config.h | cut -c1-12))"
echo MASK_TOKEN_AB_DONE
