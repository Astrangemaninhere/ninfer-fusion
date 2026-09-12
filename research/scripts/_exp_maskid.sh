#!/bin/bash
# 实验：把引擎的 DFlashConfig::mask_token 从 248077 改成草稿训练用的 190221，重编，复测 dspark。
set -u
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6_27b/impl/config.h
export PATH="/home/user/.local/bin:$PATH"
cp -f "$F" /home/user/config_h.bak_maskid
echo '=== 改前 ==='
grep -n 'mask_token' "$F" | cut -c1-120
python3 - <<'PY'
import pathlib
p = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6_27b/impl/config.h")
s = p.read_text(encoding="utf-8", errors="surrogateescape")
# 只改 DFlashConfig 那处（第一处，248077）
old = "    static constexpr int mask_token     = 248077;"
new = ("    // [EXP] was 248077; the dspark draft config (data/draft_model/config.json) and\n"
       "    // train_dspark.py both use 190221, so the engine was masking with the wrong id.\n"
       "    static constexpr int mask_token     = 190221;")
if s.count(old) == 1:
    p.write_text(s.replace(old, new), encoding="utf-8", errors="surrogateescape")
    print("改好：DFlashConfig.mask_token 248077 -> 190221")
else:
    print("匹配 %d 次，未改" % s.count(old))
PY
echo '=== 改后 ==='
grep -n 'mask_token' "$F" | cut -c1-120
echo
echo '=== touch + 重编 ==='
cd "$R" || exit 1
T=$(grep -rl --include='*.cpp' --include='*.cu' -e 'impl/package.h' -e 'variant.h' src/targets/ 2>/dev/null | sort -u)
[ -n "$T" ] && touch $T
cd "$R/build" || exit 2
MIN_FREE_GB=0 PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 bash /mnt/c/Users/User/Documents/ziqinzhang/_par_build.sh ninfer ninfer-serve > /tmp/maskid_build.log 2>&1
echo "  build rc=$?"
tail -3 /tmp/maskid_build.log | cut -c1-110
ls -l --time-style=+%H:%M "$R/build/apps/ninfer" | awk '{print "  ninfer:", $6, $5}'
echo
echo '=== 复测 dspark ==='
M=/home/user/models
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
for k in 7 1; do
  out=/home/user/maskid_dsp_k$k.log
  timeout 900 "$R/build/apps/ninfer" "$M/qwen3_8_27b_nvfp4_dspark.ninfer" --prompt "$P" \
    --max-new 96 --max-context 4096 --no-thinking --greedy --print-token-ids \
    --spec dflash --draft-tokens $k > "$out" 2>&1
  printf '  dspark K=%s  %s  pos=[%s]\n' "$k" \
    "$(grep -oE 'dflash acceptance rate +[0-9.]+%' "$out" | head -n 1)" \
    "$(grep -oE 'accepted by pos +[0-9,]+' "$out" | tail -n 1 | sed 's/.*pos *//')"
done
echo '  对照（mask=248077 时）: K=7 7.38% pos=[18,0,0,0,0,0,0] / K=1 22.08% pos=[17]'
