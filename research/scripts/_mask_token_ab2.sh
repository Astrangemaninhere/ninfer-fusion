#!/bin/bash
# 单变量 A/B（修正版）：用正则改 dflash2 的 mask_token 248070 -> 248077，改前断言
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
CFG=$R/src/targets/qwen3_6_27b/impl/config.h
BAK=/home/user/masktok_bak
LOG=$J/dl/mask_token_ab2.log
mkdir -p "$BAK"; [ -f "$BAK/config.h" ] || cp -f "$CFG" "$BAK/config.h"
exec > >(tee -a "$LOG") 2>&1
echo "=== mask_token A/B v2 $(date '+%F %H:%M:%S') ==="
grep -n 'mask_token' "$CFG" | sed 's/^/  改前: /'

python3 - <<'PYEOF'
import pathlib, re, hashlib
F = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6_27b/impl/config.h")
t = F.read_text()
pat = re.compile(r"static constexpr int mask_token\s*=\s*248070;")
n = len(pat.findall(t))
print("  正则命中 =", n)
assert n == 1, "必须恰好命中 dflash2 那一处"
F.write_text(pat.sub("static constexpr int mask_token     = 248077;", t))
print("  md5 now", hashlib.md5(F.read_bytes()).hexdigest()[:12])
PYEOF
echo "  改后："; grep -n 'mask_token' "$CFG" | sed 's/^/    /'
n248070=$(grep -c '248070' "$CFG" || true)
echo "  仍含 248070 的行数 = $n248070 （应为 0）"
[ "$n248070" != "0" ] && { echo "补丁未生效，放弃"; exit 3; }

echo "--- 编译（touch 包含 config.h 的 TU） ---"
for inc in $(grep -rl 'qwen3_6_27b/impl/config.h' $R/src 2>/dev/null | grep -v '\.orig' | sed "s|$R/||"); do
  case "$inc" in *.cpp|*.cu) touch "$R/$inc";; esac
done
cd "$R/build" || exit 4
make ninfer -j3 2>&1 | tail -3
rc=${PIPESTATUS[0]}; echo "make rc=$rc"
if [ "$rc" -ne 0 ]; then echo BUILD_FAIL; cp -f $BAK/config.h "$CFG"; exit 5; fi

echo "--- 跑（mask=248077） ---"
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
NINFER_DF2DBG=1 timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 \
  > $J/dl/masktok_run2.log 2>&1
echo "  rc=$?"
grep -E 'dflash2 acceptance rate|dflash2 accepted by pos|dflash2 acceptance length|decode speed' $J/dl/masktok_run2.log | sed 's/^/  /'
echo "  对照基线（mask=248070）：4.81% / 20,3,0,0,0,0,0"

echo "--- 回滚 ---"
cp -f $BAK/config.h "$CFG"
echo "  restored 248070 行数 = $(grep -c '248070' $CFG)"
echo MASK_TOKEN_AB2_DONE
