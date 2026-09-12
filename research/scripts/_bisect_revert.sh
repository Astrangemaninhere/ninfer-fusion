#!/bin/bash
# 二分定位 MTP k>=2 的 5~6x 性能回归：把我改动过的 10 个文件全部回退到备份，重编，测 k=3
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:$PATH
echo "=== 回退前改动清单（与备份 diff 行数）==="
for pair in "bf16head_bak:src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h" \
            "bf16head_bak:src/targets/qwen3_6_27b/impl/load/bindings.cpp" \
            "bf16head_bak:src/targets/qwen3_6_27b/impl/package.cpp" \
            "bf16head_bak:src/targets/qwen3_6_27b/impl/variant.cpp" \
            "bf16head_bak:src/ops/linear/bf16/bf16_dispatch.cpp" \
            "bf16head_bak:src/ops/linear/bf16/bf16_gemm_mma.cu" \
            "df2scores_bak:src/targets/qwen3_6/impl/runtime/dflash2_impl.h" \
            "eight_bak:apps/cli/main.cpp" \
            "eight_bak:src/targets/qwen3_6/impl/runtime/text_prefill_impl.h" \
            "eight_bak:src/serve/kv_auto_relayout.cpp"; do
  b="${pair%%:*}"; f="${pair#*:}"
  if [ -f "/home/user/$b/$f" ]; then
    n=$(diff -uw "/home/user/$b/$f" "$R/$f" 2>/dev/null | grep -cE '^[+-][^+-]')
    printf "  %-22s %s\n" "$b" "$f  差异行=$n"
  else
    printf "  %-22s %s  (无备份)\n" "$b" "$f"
  fi
done
echo
echo "=== 全部回退 ==="
for pair in "bf16head_bak:src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h" \
            "bf16head_bak:src/targets/qwen3_6_27b/impl/load/bindings.cpp" \
            "bf16head_bak:src/targets/qwen3_6_27b/impl/package.cpp" \
            "bf16head_bak:src/targets/qwen3_6_27b/impl/variant.cpp" \
            "bf16head_bak:src/ops/linear/bf16/bf16_dispatch.cpp" \
            "bf16head_bak:src/ops/linear/bf16/bf16_gemm_mma.cu" \
            "df2scores_bak:src/targets/qwen3_6/impl/runtime/dflash2_impl.h" \
            "eight_bak:apps/cli/main.cpp" \
            "eight_bak:src/targets/qwen3_6/impl/runtime/text_prefill_impl.h" \
            "eight_bak:src/serve/kv_auto_relayout.cpp"; do
  b="${pair%%:*}"; f="${pair#*:}"
  [ -f "/home/user/$b/$f" ] && cp -p "/home/user/$b/$f" "$R/$f"
done
echo "回退完成"
echo
echo "=== 重编 ==="
cd "$R/build" || exit 3
/usr/bin/time -f 'BUILD_WALL=%es' make -j8 ninfer 2>&1 | tail -4
echo "rc=${PIPESTATUS[0]}"
echo
echo "=== 回退后测 k=3（期望恢复到 ~100 tok/s）==="
timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt '请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。' \
  --max-new 96 --max-context 4096 --no-thinking --greedy --spec mtp --draft-tokens 3 --lm-head-draft \
  > "$J/revert_k3.log" 2>&1
grep -E 'decode speed|acceptance rate|acceptance length' "$J/revert_k3.log" | sed 's/^/  /'
echo "=== 完成 $(date '+%H:%M:%S') ==="
echo REVERT_DONE
