#!/bin/bash
# 顺序重放全部补丁（树曾被二分回退），然后重编并验收
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:$PATH
echo "=== 顺序重放补丁 ==="
for f in pt.py pd3.py pw.py add_bf16head2.py patch_e.py patch_variant.py pe.py pap.py; do
  if [ -f "/home/user/$f" ]; then
    printf "  %-22s " "$f"
    python3 "/home/user/$f" > /tmp/o 2>&1 && echo "OK" || { echo "FAIL"; tail -3 /tmp/o; }
  else
    printf "  %-22s 缺失\n" "$f"
  fi
done
echo
echo "=== 触及源文件并重编 ==="
touch "$R/src/ops/linear/bf16/bf16_dispatch.cpp" \
      "$R/src/ops/linear/bf16/bf16_gemm_mma.cu" \
      "$R/apps/cli/main.cpp" "$R/apps/cli/options.h" \
      "$R/src/serve/kv_auto_relayout.cpp" 2>/dev/null
cd "$R/build" || exit 3
/usr/bin/time -f 'BUILD_WALL=%es' make -j8 ninfer 2>&1 | tail -3
echo "rc=${PIPESTATUS[0]}"
echo
echo "=== 验收 1：默认（无 flag）MTP k=3 应回到 ~130 tok/s ==="
timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt '请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。' \
  --max-new 96 --max-context 4096 --no-thinking --greedy --spec mtp --draft-tokens 3 --lm-head-draft \
  2>&1 | grep -E 'decode speed|acceptance rate|acceptance length' | sed 's/^/  /'
echo
echo "=== 验收 2：默认 dflash2（接受率应仍 4.81%/20,3,0..）==="
timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt '请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。' \
  --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 \
  2>&1 | grep -E 'decode speed|acceptance rate|accepted by pos' | sed 's/^/  /'
echo
echo "=== 验收 3：kv dtype 打印 + bf16 头 artifact 仍可加载 ==="
./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer --prompt '你好' --max-new 4 \
  --max-context 512 --no-thinking --greedy --kv-dtype nvfp4 2>&1 | grep -E 'kv cache dtype' | sed 's/^/  /'
./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2_bf16head.ninfer --prompt '你好' --max-new 4 \
  --max-context 512 --no-thinking --greedy 2>&1 | grep -E 'decode speed|error' | head -2 | sed 's/^/  /'
echo "=== 完成 $(date '+%H:%M:%S') ==="
echo REPLAY_DONE
