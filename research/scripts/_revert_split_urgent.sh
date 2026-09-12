#!/bin/bash
# 紧急：把两处 TU 拆分从"树上的半成品"退回成"链接自洽的原地版本 + 待落件"。
# 原因：拆分把符号搬到了新文件，但新文件不在生成的 Makefile 里（要重新配置才会进），
# 于是库变成"能编译、不能链接"。而武装中的批次1 是 make（不重新配置）-> 链接必炸 -> A5b 测量链被毁。
# 做法：① 保留拆分成果到 _collab/build/staged/  ② 用备份覆盖回原文件  ③ 去掉临时加的 CMake 行
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab/build
STAGE=$C/staged
mkdir -p "$STAGE"

echo '=== ① 转存拆分成果 ==='
for f in src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu \
         src/ops/launcher/gqa_attention_decode.cu; do
  b=$(basename "$f")
  if [ -f "$R/$f" ]; then
    cp -f "$R/$f" "$STAGE/$b.split" && \
      echo "  已存 $STAGE/$b.split  ($(wc -l < "$STAGE/$b.split") 行, md5 $(md5sum "$STAGE/$b.split" | cut -c1-8))"
  fi
done
# 新文件本身也留一份（它们在树上无害，但为防误加回构建，这里也存）
for f in src/ops/launcher/gqa_attention_decode_partial.cuh \
         src/ops/launcher/gqa_attention_decode_smallt.cu \
         src/ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh \
         src/ops/linear/nvfp4/nvfp4_w4a4_tma_attn.cu \
         src/ops/linear/nvfp4/nvfp4_w4a4_tma_gdn.cu \
         src/ops/linear/nvfp4/nvfp4_w4a4_tma_mlp.cu \
         src/ops/linear/nvfp4/nvfp4_w4a4_tma_residual.cu; do
  [ -f "$R/$f" ] && cp -f "$R/$f" "$STAGE/$(basename "$f").new" && echo "  已存 $(basename "$f").new"
done

echo
echo '=== ② 用备份覆盖回原文件（恢复链接自洽）==='
BK1=$C/backup/nvfp4_w4a4_tma.cu.orig
BK2=$C/backup/gqa_attention_decode.cu.orig
[ -f "$BK1" ] || BK1=/tmp/orig_tma.cu
if [ -f "$BK1" ]; then
  echo "  nvfp4 备份: $BK1  md5 $(md5sum "$BK1" | cut -c1-12)"
  cp -f "$BK1" "$R/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu" && echo "  已恢复 nvfp4_w4a4_tma.cu md5 $(md5sum "$R/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu" | cut -c1-12)"
else
  echo "  找不到 nvfp4 备份（需人工处理）"
fi
[ -f "$BK2" ] || BK2=/tmp/orig.cu
if [ -f "$BK2" ]; then
  echo "  gqa 备份: $BK2  md5 $(md5sum "$BK2" | cut -c1-12)"
  cp -f "$BK2" "$R/src/ops/launcher/gqa_attention_decode.cu" && echo "  已恢复 gqa_attention_decode.cu md5 $(md5sum "$R/src/ops/launcher/gqa_attention_decode.cu" | cut -c1-12)"
else
  echo "  找不到 gqa 备份（需人工处理）"
fi

echo
echo '=== ③ 去掉临时加的 CMake 行（生成的 Makefile 不受影响，但重新配置时必须干净）==='
CL=$R/src/CMakeLists.txt
before=$(md5sum "$CL" | cut -c1-12)
python3 - <<'PY'
import pathlib
p = pathlib.Path("/home/user/ninfer-fusion/src/CMakeLists.txt")
raw = p.read_bytes()
needle = b"  ops/launcher/gqa_attention_decode_smallt.cu\n"
if raw.count(needle) == 1:
    p.write_bytes(raw.replace(needle, b""))
    print("  已删除临时行 ops/launcher/gqa_attention_decode_smallt.cu")
else:
    print("  该行出现 %d 次，未改动（需人工确认）" % raw.count(needle))
PY
echo "  CMakeLists md5 $before -> $(md5sum "$CL" | cut -c1-12)"

echo
echo '=== ④ 自洽性检查：库该有的源都在 CMake 里 ==='
grep -n 'gqa_attention_decode\|nvfp4_w4a4_tma' "$CL" | head -8 | cut -c1-100
echo
echo '=== ⑤ 树上残留的新文件（不在 CMake，不会被编译，无害）==='
ls -la --time-style=+%H:%M "$R/src/ops/launcher/gqa_attention_decode_partial.cuh" \
   "$R/src/ops/launcher/gqa_attention_decode_smallt.cu" \
   "$R/src/ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh" 2>/dev/null | cut -c25-95
echo '=== 链条是否仍活 ==='
ps -ef | grep -E 'pfv2[.]sh|batch_next[.]sh|reconfig_next[.]sh|probe_arm[.]sh|_rebuild_after_s3[5]' | grep -v grep | awk '{print "  ", $2, $NF}'
date +%H:%M:%S
