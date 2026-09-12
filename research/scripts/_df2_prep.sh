#!/bin/bash
# ① 把 target + draft 拷到原生盘（避免每次失败重载 6.5 分钟）
# ② 补装 flashinfer cubin / jit-cache（Linux）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029
LOG=$J/dl/df2_prep.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 准备 $(date '+%F %H:%M:%S') ==="

echo "--- ① 拷贝 target（大小 $(du -sh $J/models/Qwen3.8-27B-NVFP4-RTX5090 2>/dev/null | cut -f1)） ---"
mkdir -p /home/user/models
if [ ! -f /home/user/models/qwen3_8_27b_nvfp4_hf/config.json ]; then
  mkdir -p /home/user/models/qwen3_8_27b_nvfp4_hf
  cp -f $J/models/Qwen3.8-27B-NVFP4-RTX5090/* /home/user/models/qwen3_8_27b_nvfp4_hf/ 2>&1 | tail -2
fi
du -sh /home/user/models/qwen3_8_27b_nvfp4_hf 2>/dev/null

echo "--- ① 拷贝 draft ---"
mkdir -p /home/user/models/draft_dflash2_ref
cp -f $J/data/draft_dflash2_ref/config.json $J/data/draft_dflash2_ref/model.safetensors /home/user/models/draft_dflash2_ref/ 2>&1 | tail -2
ls -l --time-style=+%H:%M /home/user/models/draft_dflash2_ref/ | tail -3

echo "--- ② flashinfer 现状 ---"
$V/bin/python -c "import flashinfer, os; print('flashinfer', flashinfer.__version__, os.path.dirname(flashinfer.__file__))" 2>&1 | tail -2
$V/bin/pip list 2>/dev/null | grep -iE 'flashinfer|flash.attn' | head -5

echo "--- ② 装 cubin（py3-none-any，可跨平台） ---"
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
$V/bin/pip install -i $IDX flashinfer-cubin 2>&1 | tail -4
echo "--- ② 试装 jit-cache（flashinfer 官方索引，cu130/linux） ---"
$V/bin/pip install --extra-index-url https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache" 2>&1 | tail -5
$V/bin/pip list 2>/dev/null | grep -iE 'flashinfer' | head -5
echo DF2_PREP_DONE
