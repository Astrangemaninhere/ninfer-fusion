#!/bin/bash
# 严格确认：1Cat 扩展里是否有 PTX（PTX 可前向 JIT 到 SM120；纯 SASS 不行）
set -u
CU=/usr/local/cuda/bin/cuobjdump
E=/tmp/1cat_whl
for so in "$E/vllm/_C.abi3.so" "$E/vllm/_moe_C.abi3.so" "$E/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"; do
  [ -f "$so" ] || continue
  echo "=== $(basename $so) ==="
  n_elf=$("$CU" --list-elf "$so" 2>/dev/null | wc -l)
  n_ptx=$("$CU" --list-ptx "$so" 2>/dev/null | wc -l)
  echo "  ELF(cubin) 数 = $n_elf"
  echo "  PTX    数 = $n_ptx"
  "$CU" --list-ptx "$so" 2>&1 | head -4 | sed 's/^/    PTX: /'
  # 架构种类
  "$CU" --list-elf "$so" 2>/dev/null | grep -oE 'sm_[0-9]+[a-z]?' | sort -u | tr '\n' ' ' | sed 's/^/  架构: /'
  echo
done
echo "=== 对照：本机可跑的那份 vLLM 支持哪些架构 ==="
V=/mnt/c/vllm/venv/Lib/site-packages/vllm/_C.pyd
[ -f "$V" ] || V=$(ls /mnt/c/vllm/venv/Lib/site-packages/vllm/_C*.pyd 2>/dev/null | head -1)
echo "  文件: $V"
[ -f "$V" ] && { "$CU" --list-elf "$V" 2>/dev/null | grep -oE 'sm_[0-9]+[a-z]?' | sort -u | tr '\n' ' '; echo; "$CU" --list-elf "$V" 2>/dev/null | head -4 | sed 's/^/    /'; }
