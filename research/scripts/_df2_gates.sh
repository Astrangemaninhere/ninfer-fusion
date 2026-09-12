#!/bin/bash
# 三闸门：verify_patch + roundtrip（两个 artifact），并抓 001900 的 patch 报错
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/df2_gates.log
PY="/mnt/c/Program Files/Python312/python.exe"
SRC="models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer"
TOOL="ninfer-fusion-repo/tools/convert/qwen3_8_27b"
cd "$J" || exit 3
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "=== dflash2 artifact 三闸门 $(date '+%F %H:%M:%S') ==="
echo "--- 001900 的 patch 报错（重跑抓全输出，写临时输出避免覆盖） ---"
"$PY" "$TOOL/patch_dflash2.py" --src "$SRC" --ckpt "data/dflash2_ckpts/step_001900.pt" \
   --out "data/dflash2_ckpts/step_001900_tuned2.ninfer" > /tmp/p1900.log 2>&1
echo "patch rc=$?"
grep -nE 'ERROR|error|missing|Traceback|Exception|ok:|written' /tmp/p1900.log | tail -12
rm -f "data/dflash2_ckpts/step_001900_tuned2.ninfer"

for CK in step_000200 step_001900; do
  echo
  echo "########## $CK verify 闸门 ##########"
  "$PY" "$TOOL/verify_patch.py" --src "$SRC" --out "data/dflash2_ckpts/${CK}_tuned.ninfer" \
     --ckpt "data/dflash2_ckpts/$CK.pt" 2>&1 | tail -10
  echo "verify rc=${PIPESTATUS[0]}"
  echo "########## $CK roundtrip 闸门 ##########"
  "$PY" "$J/_dflash2_roundtrip_tmp.py" "data/dflash2_ckpts/$CK.pt" \
     "data/dflash2_ckpts/${CK}_tuned.ninfer" 2>&1 | tail -6
  echo "roundtrip rc=${PIPESTATUS[0]}"
done
echo
echo "=== 编译状态 ==="
pgrep -x make >/dev/null && echo "make 仍在跑" || tail -6 $J/dl/build_split.log
echo DF2_GATES_DONE
