#!/bin/bash
# ① dflash2 基线（auto × 3）
# ② S52 验收：显式 --spec dflash2 --draft-tokens 1|3|7（配方要求的就是显式档）
# ③ plain 基线（不带 --spec）
set -u
R=/home/user/ninfer-fusion
M=/home/user/models
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
BIN=$R/build/apps/ninfer
cd "$R/build" || exit 1

run() {  # tag model spec-args...
  local tag=$1 model=$2; shift 2
  local out=/home/user/b2_$tag.log
  timeout 900 "$BIN" "$M/$model" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$out" 2>&1
  local rc=$?
  local pos spd ids err
  pos=$(grep -oE 'accepted by pos +[0-9,]+' "$out" | tail -1 | sed 's/.*pos *//')
  spd=$(grep -oE 'decode speed +[0-9.]+' "$out" | head -1 | grep -oE '[0-9.]+')
  ids=$(grep -oE '^tokens +generated ids.*' "$out" | head -1 | md5sum | cut -c1-8)
  err=$(grep -iE 'error|invalid|throw|refus' "$out" | head -1 | cut -c1-70)
  printf '  %-22s rc=%s pos=[%s] decode=%s ids=%s %s\n' "$tag" "$rc" "${pos:-无}" "${spd:-?}" "$ids" "${err:-}"
}

echo '=== ① dflash2 基线（--spec auto × 3）==='
for i in 1 2 3; do run "df2_auto_r$i" qwen3_8_27b_nvfp4_dflash2.ninfer --spec auto; done
echo '=== ② S52 验收：显式 --spec dflash2 的 K 扫描 ==='
for k in 1 3 7; do run "df2_k$k" qwen3_8_27b_nvfp4_dflash2.ninfer --spec dflash2 --draft-tokens $k; done
echo '=== ③ dspark 显式档对照（K=1/3/7，各 1 次）==='
for k in 1 3 7; do run "dsp_k$k" qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens $k; done
echo '=== ④ plain 基线 ==='
run plain qwen3_8_27b_nvfp4.ninfer
date +%H:%M:%S
