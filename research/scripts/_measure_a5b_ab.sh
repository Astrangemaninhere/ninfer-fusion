#!/bin/bash
# A5b 的 A/B 测量：两个只差 A5b 的二进制，跑同一档 dspark/dflash2，比接受率与剖面。
# 用户指出"开头你修了对齐的问题，但现在问题越来越严重" ⇒ 验 A5b 是否有害。
set -u
M=/home/user/models
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd /home/user/ninfer-fusion/build || exit 1

run() {  # tag binary model args...
  local tag=$1 bin=$2 model=$3; shift 3
  local out=/home/user/ab5_$tag.log
  timeout 900 "$bin" "$M/$model" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$out" 2>&1
  local rc=$? pos acc tpr
  pos=$(grep -oE 'accepted by pos +[0-9,]+' "$out" | tail -n 1 | sed 's/.*pos *//')
  acc=$(grep -oE 'dflash2? acceptance rate +[0-9.]+%' "$out" | head -n 1)
  tpr=$(grep -oE '[0-9.]+tok/round' "$out" | head -n 1)
  printf '  %-22s rc=%s %-34s pos=[%s] %s\n' "$tag" "$rc" "$acc" "${pos:-无}" "$tpr"
  # 记下 token ids 以便逐位比较
  grep -oE '^tokens +generated ids.*' "$out" | head -n 1 | md5sum | cut -c1-8 | sed "s/^/    ids_md5=/"
}

echo '=== dspark K=7：A5b 回退 vs A5b 应用 ==='
run dsp_no_a5b  /home/user/ninfer_no_a5b  qwen3_8_27b_nvfp4_dspark.ninfer  --spec dflash --draft-tokens 7
run dsp_with_a5b /home/user/ninfer_with_a5b qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 7
echo
echo '=== dspark K=1（只看 p0 质量）==='
run dsp_k1_no   /home/user/ninfer_no_a5b  qwen3_8_27b_nvfp4_dspark.ninfer  --spec dflash --draft-tokens 1
run dsp_k1_with /home/user/ninfer_with_a5b qwen3_8_27b_nvfp4_dspark.ninfer --spec dflash --draft-tokens 1
echo
echo '=== dflash2 K=7 ==='
run df2_no   /home/user/ninfer_no_a5b  qwen3_8_27b_nvfp4_dflash2.ninfer --spec dflash2 --draft-tokens 7
run df2_with /home/user/ninfer_with_a5b qwen3_8_27b_nvfp4_dflash2.ninfer --spec dflash2 --draft-tokens 7
echo
echo '=== 判读 ==='
echo '  若 no_a5b 明显高于 with_a5b ⇒ A5b 有害，应回退（用户判断成立）'
echo '  若两者相同 ⇒ A5b 与接受率无关，继续找别处'
date +%H:%M:%S
