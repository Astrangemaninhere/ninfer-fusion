#!/bin/bash
# 定点核实：逆转化（ninfer/我们的头 -> HF）是否做过、卡在哪
J=/mnt/c/Users/User/Documents/ziqinzhang
RW=$J/ninfer-fusion-repo/tools/convert/qwen3_8_27b
echo "=== 1) 更新结果 ==="
tail -4 $J/dl/vllm_update.log 2>/dev/null
pgrep -f 'pip install' >/dev/null && echo "  pip 仍在跑" || echo "  pip 已结束"
echo
echo "=== 2) 转换工具目录里有没有"反向/导出 HF"的脚本（单目录 ls） ==="
ls -1 $RW 2>/dev/null
echo "--- 全文里提到 reverse/export/to_hf/unconvert 的行（单目录内 grep）"
grep -rniE 'reverse|to_hf|export_hf|unconvert|hf_out|--out-hf' $RW/*.py 2>/dev/null | head -12
echo
echo "=== 3) 那些 reshape note 是否可逆（(2,2,320,16) <-> (2,2,5120) 是纯 reshape） ==="
grep -nE 'reshape|view\(' $RW/patch_dflash2.py 2>/dev/null | head -10
echo
echo "=== 4) vllm 更新后是否有 Qwen3 的 dflash 草稿架构（单目录/单文件） ==="
P=/mnt/c/vllm/venv-clean/Lib/site-packages/vllm
ls -1 $P/model_executor/models 2>/dev/null | grep -iE 'dflash|dspark' | head
grep -rnE 'class .*DFlash|Qwen3.*DFlash|DFlashModel' $P/model_executor/models/qwen3_dflash.py $P/model_executor/models/dflash.py 2>/dev/null | head -6
