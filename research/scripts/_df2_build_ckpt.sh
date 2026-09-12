#!/bin/bash
# 用两炉 ckpt 造 dflash2 artifact（CPU-only；走操作卡片的三闸门）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/df2_build_ckpt.log
PY="/mnt/c/Program Files/Python312/python.exe"
SRC="models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer"
TOOL="ninfer-fusion-repo/tools/convert/qwen3_8_27b"
cd "$J" || exit 3
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "=== 造 dflash2 artifact（新炉 step_000200 与老炉 step_001900）$(date '+%F %H:%M:%S') ==="
ls -l "$SRC" 2>/dev/null | tail -1

for CK in step_000200 step_001900; do
  echo
  echo "########## $CK ##########"
  OUT="data/dflash2_ckpts/${CK}_tuned.ninfer"
  if [ -f "$OUT" ]; then echo "已存在 $OUT，跳过调用"; else
    echo "--- patch ---"
    "$PY" "$TOOL/patch_dflash2.py" --src "$SRC" --ckpt "data/dflash2_ckpts/$CK.pt" --out "$OUT" 2>&1 | tail -8
    echo "patch rc=${PIPESTATUS[0]}"
  fi
  ls -l "$OUT" 2>/dev/null | tail -1
done

echo
echo "=== 编译状态 ==="
pgrep -x make >/dev/null && echo "make 仍在跑" || { echo "make 结束"; tail -5 $J/dl/build_split.log; }
echo DF2_BUILD_CKPT_DONE
