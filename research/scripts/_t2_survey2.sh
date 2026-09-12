#!/bin/bash
# T2 part 2: mtimes of all touched target files, CMakeLists grep, staged2 mapping. READ-ONLY.
T=/home/user/ninfer-fusion
cd "$T" || exit 3
echo "=== [1] mtimes of files touched by the pending/landed patches ==="
for f in \
  src/targets/qwen3_6/impl/runtime/dflash_impl.h \
  src/product/speculative_options.h \
  src/targets/qwen3_6/impl/runtime/layouts_impl.h \
  src/targets/qwen3_6/impl/runtime/dflash2_impl.h \
  src/ops/wrapper/dflash2_grouped_conv.cpp \
  src/ops/wrapper/dflash2_selector.cpp \
  src/ops/launcher/dflash2_selector.cu \
  src/ops/kernel/gqa_isoquant_row_scale_loader.h \
  src/ops/kernel/gqa_isoquant_row_scale.cuh \
  src/ops/kernel/gqa_isoquant_row_scale.cu \
  src/ops/kernel/gqa_isoquant_row_scale_loader.cu \
  src/ops/kernel/gqa_isoquant_rot.cuh \
  src/targets/qwen3_6/impl/runtime/kv_calibration.h \
  src/targets/qwen3_6/impl/runtime/text_context_impl.h \
  src/targets/qwen3_6/impl/state/decoder_state.cpp \
  src/targets/registry.cpp \
  src/ops/launcher/gqa_attention_decode.cu \
  src/ops/launcher/gqa_attention_decode_e8.cu \
  src/ops/launcher/gqa_attention_prefill.cu \
  src/ops/launcher/gqa_attention_prefill_e8.cu \
  src/ops/launcher/gqa_attention_decode_partial.cuh \
  src/ops/launcher/gqa_attention_decode_smallt.cu \
  src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu \
  src/ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh \
  apps/cli/options.cpp apps/cli/options.h apps/cli/main.cpp \
  tests/ops/test_silu_mul.cpp tests/targets/qwen3_6/test_context_store.cpp \
  src/serve/serve_options.h src/serve/serve_options.cpp \
  src/product/weight_residency.h src/ops/linear/linear.cpp \
  ; do
  if [ -f "$f" ]; then stat -c '%y  %n' "$f"; else echo "MISSING            $f"; fi
done

echo
echo "=== [2] files in tree with mtime 2026-09-10 21:4x (the 21:42:45 batch) ==="
find src apps tests include tools -type f -newermt '2026-09-10 21:40' ! -newermt '2026-09-10 21:50' -printf '%TH:%TM:%TS  %p\n' 2>/dev/null | sort | head -40

echo
echo "=== [3] CMakeLists entries for staged new files (must be 0 if not landed) ==="
echo "--- src/CMakeLists.txt:"
grep -n 'smallt\|partial.cuh\|nvfp4_w4a4_tma_arms\|nvfp4_w4a4_tma_attn\|nvfp4_w4a4_tma_gdn\|nvfp4_w4a4_tma_mlp\|nvfp4_w4a4_tma_residual\|gelu_mul\|gelu_and_mul\|gqa_attention_decode_e8_arms\|gqa_attention_prefill_arms\|gqa_attention_prefill_e8_arms\|gqa_attention_prefill_batch\|kv_append' src/CMakeLists.txt || echo '(none)'
echo "--- tests/CMakeLists.txt:"
grep -n 'gelu_mul\|test_gelu' tests/CMakeLists.txt || echo '(none)'
echo "--- src/CMakeLists.txt line count + md5:"
wc -l src/CMakeLists.txt; md5sum src/CMakeLists.txt

echo
echo "=== [4] staged vs tree: full byte diff summary ==="
S=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build
for pair in \
  "staged/gqa_attention_decode.cu.split:src/ops/launcher/gqa_attention_decode.cu" \
  "staged/nvfp4_w4a4_tma.cu.split:src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu" \
  "staged2/gqa_attention_decode_e8.cu.split:src/ops/launcher/gqa_attention_decode_e8.cu" \
  "staged2/gqa_attention_prefill.cu.split:src/ops/launcher/gqa_attention_prefill.cu" \
  "staged2/gqa_attention_prefill_e8.cu.split:src/ops/launcher/gqa_attention_prefill_e8.cu" \
  ; do
  a="$S/${pair%%:*}"; b="$T/${pair##*:}"
  if cmp -s "$a" "$b"; then echo "IDENTICAL  ${pair%%:*}  ==  ${pair##*:}"; else echo "DIFFER     ${pair%%:*}  !=  ${pair##*:}   (staged $(wc -c <"$a")B / tree $(wc -c <"$b")B)"; fi
done
echo "--- .new files: exist? byte-identical? ---"
for f in staged/gqa_attention_decode_partial.cuh.new staged/gqa_attention_decode_smallt.cu.new staged/nvfp4_w4a4_tma_arms.cuh.new staged/nvfp4_w4a4_tma_attn.cu.new staged/nvfp4_w4a4_tma_gdn.cu.new staged/nvfp4_w4a4_tma_mlp.cu.new staged/nvfp4_w4a4_tma_residual.cu.new staged2/gqa_attention_decode_e8_append_g27.cu.new staged2/gqa_attention_decode_e8_append_muse35.cu.new staged2/gqa_attention_decode_e8_arms.cuh.new staged2/gqa_attention_decode_e8_cached_g27.cu.new staged2/gqa_attention_decode_e8_cached_muse35.cu.new staged2/gqa_attention_prefill_arms.cuh.new staged2/gqa_attention_prefill_batch.cu.new staged2/gqa_attention_prefill_e8_arms.cuh.new staged2/gqa_attention_prefill_e8_kv_append.cu.new staged2/gqa_attention_prefill_e8_kv_append_single.cu.new; do
  name=${f##*/}; name=${name%.new}
  base=$(echo "$f" | sed 's|^staged2/||; s|^staged/||; s|\.new$||')
  case "$f" in
    staged/*) tgt=$(find src -name "$base" 2>/dev/null | head -1);;
    staged2/*) tgt=$(find src -name "$base" 2>/dev/null | head -1);;
  esac
  if [ -z "$tgt" ]; then echo "ABSENT-ON-TREE   $f"; else
    if cmp -s "$S/$f" "$T/$tgt"; then echo "IDENTICAL        $f  ->  $tgt"; else echo "DIFFER           $f  ->  $tgt"; fi
  fi
done

echo
echo "=== [5] who includes the staged new files already? ==="
grep -rn 'gqa_attention_decode_partial.cuh\|gqa_attention_decode_smallt\|nvfp4_w4a4_tma_arms' src --include='*.cu' --include='*.cpp' --include='*.cuh' --include='*.h' 2>/dev/null | grep -v '^src/ops/launcher/gqa_attention_decode_partial.cuh' | head -20
echo "(end)"
