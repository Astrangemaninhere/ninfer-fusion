#!/bin/bash
# 清理卡住的旧脚本（先查 cmdline），再用 Windows python 续传草稿 + 建 3.12 venv 装 vllm 0.29
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 先查在跑的相关进程（铁律②） ==="
ps -eo pid,etimes,args | grep -E 'redo_dflash2|snapshot_download|hf_hub' | grep -v grep | cut -c1-120
for p in $(pgrep -f 'redo_dflash2.sh' 2>/dev/null); do
  cl=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cl" in *redo_dflash2*) echo "  杀 pid=$p ($cl)"; kill "$p" 2>/dev/null;; esac
done
pkill -f 'snapshot_download' 2>/dev/null
sleep 1
echo "  剩余: $(pgrep -fc 'redo_dflash2|snapshot_download' 2>/dev/null || echo 0)"

LOG=$J/dl/redo2.log
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "=== 再搞一次 v2 $(date '+%F %H:%M:%S') ==="
DIR_W='C:/Users/User/Documents/ziqinzhang/data/draft_dflash2_ref'
PYW='/mnt/c/vllm/venv/Scripts/python.exe'

echo "--- ① 用 Windows python 续传 incoai/Qwen3.8-27B-DFlash2 ---"
"$PYW" -c "
from huggingface_hub import snapshot_download
import os
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER','0')
p = snapshot_download('incoai/Qwen3.8-27B-DFlash2', local_dir=r'$DIR_W', max_workers=4)
print('snapshot ok ->', p)
" 2>&1 | tail -8
echo "hf rc=$?"

echo "--- ① 校验完整性（safetensors 头） ---"
python3 - "$J/data/draft_dflash2_ref/model.safetensors" <<'PY'
import sys, json, struct, pathlib
p = pathlib.Path(sys.argv[1]); sz = p.stat().st_size
with open(p,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
end = 8 + n + max((v['data_offsets'][1] for k,v in hdr.items() if k!='__metadata__'), default=0)
print("size=%d declared_end=%d => %s" % (sz, end, "COMPLETE" if end<=sz else "STILL_TRUNCATED(%.1f%%)" % (100.0*sz/end)))
PY

echo "--- ② 全新 vllm 0.29（python3.12） ---"
W=$J/_collab/build/dl_0290/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl
V=/home/user/vllm029
rm -rf "$V"; python3.12 -m venv "$V" || { echo VENV_FAIL; exit 3; }
VP=$V/bin/python
$VP -m pip install -q -U pip setuptools wheel 2>&1 | tail -2
$VP -m pip install "$W" 2>&1 | tail -10
$VP -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -2
$VP -c "import typing; from vllm.config.speculative import SpeculativeMethod; a=typing.get_args(SpeculativeMethod); print('dflash2' in a, 'dflash' in a, 'dspark' in a)" 2>&1 | tail -2
$VP -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -2
echo REDO2_DONE
