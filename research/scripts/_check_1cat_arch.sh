#!/bin/bash
# 解包 1Cat wheel，用 cuobjdump/nvdisasm 查主扩编了哪些 GPU 架构
set -u
W=/mnt/c/Users/User/Documents/ziqinzhang/dl/1cat_vllm-1.5.0-cp312-cp312-linux_x86_64.whl
E=/tmp/1cat_whl
rm -rf "$E"; mkdir -p "$E"
python3.12 -c "
import zipfile
z = zipfile.ZipFile('$W')
for n in z.namelist():
    if n.endswith(('.so',)) and ('_C.abi3' in n or 'flash_attn_v100' in n or 'sm70' in n or '_vllm_fa2_C' in n):
        z.extract(n, '$E')
print('extracted')
"
echo "=== 解出的 so ==="
find "$E" -name '*.so' | head -10
echo
echo "=== 有无 cuobjdump ==="
which cuobjdump nvdisasm 2>/dev/null || ls /usr/local/cuda*/bin/cuobjdump 2>/dev/null || echo "  未找到 cuobjdump"
CU=$(which cuobjdump 2>/dev/null || ls /usr/local/cuda*/bin/cuobjdump 2>/dev/null | head -1)
if [ -n "$CU" ]; then
  for so in $(find "$E" -name '*.so' | head -4); do
    echo "--- $(basename $so) ---"
    "$CU" --list-elf "$so" 2>/dev/null | head -8
    echo "   [PTX 段]"
    "$CU" --list-ptx "$so" 2>/dev/null | head -5
  done
else
  echo "=== 退路：直接在二进制里搜架构标记串 ==="
  for so in $(find "$E" -name '*.so' | head -4); do
    echo "--- $(basename $so) ---"
    strings -a "$so" 2>/dev/null | grep -oE 'sm_[0-9]+[a-z]?|compute_[0-9]+[a-z]?' | sort -u | tr '\n' ' ' | head -c 300
    echo
  done
fi
