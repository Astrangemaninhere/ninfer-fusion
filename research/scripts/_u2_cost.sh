#!/bin/bash
# U2: rebuild-cost forensics for the dspark A5b isolation
set -u
R=/home/user/ninfer-fusion
B=$R/build
echo "### giant gqa attention TUs: object vs source"
for t in gqa_attention_decode gqa_attention_decode_e8 gqa_attention_prefill gqa_attention_prefill_e8; do
  o=$(find "$B" -name "$t.cu.o" -printf '%TY-%Tm-%Td %TH:%TM  %10s' 2>/dev/null | head -1)
  s=$(stat -c '%y  %10s' "$R/src/ops/launcher/$t.cu" 2>/dev/null | cut -c1-21,22-40)
  printf '  %-26s OBJ[%s]  SRC[%s]\n' "$t" "${o:--}" "${s:--}"
done

echo
echo "### who includes gqa_attention_geometry.cuh / gqa_isoquant_row_scale.cuh"
grep -rl 'gqa_attention_geometry.cuh' "$R/src/" 2>/dev/null | sed "s#$R/##" | head -12
echo "  -- row_scale.cuh consumers --"
grep -rl 'gqa_isoquant_row_scale.cuh' "$R/src/" 2>/dev/null | sed "s#$R/##" | head -20
echo "  -- count --"
grep -rl 'gqa_isoquant_row_scale.cuh' "$R/src/" 2>/dev/null | wc -l

echo
echo "### newest 15 objects in build/"
find "$B" -name '*.o' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort -r | head -15 | sed "s#$B/##"

echo
echo "### engines/apps link targets + freshness"
for f in "$B/apps/ninfer" "$B/apps/ninfer-serve"; do stat -c '  %y %s %n' "$f" 2>/dev/null; done

echo
echo "### engine objects (ninfer_engine) newest 12"
find "$B/src/CMakeFiles/ninfer_engine.dir" -name '*.o' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort -r | head -12 | sed "s#$B/##"

echo
echo "### dflash_impl.h includes"
grep -n '#include' "$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h" | head -20

echo
echo "### model artifacts present"
ls -la /home/user/models/*.ninfer 2>/dev/null | head
