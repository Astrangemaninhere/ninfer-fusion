#!/bin/bash
# 再搞一次：① 补完 incoai/Qwen3.8-27B-DFlash2 下载（3.58GB，带完整性校验）
#            ② 用 python3.12 重装全新 vllm 0.29（本地轮子）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
W=$J/_collab/build/dl_0290/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl
DIR=$J/data/draft_dflash2_ref
LOG=$J/dl/redo_dflash2.log
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "=== 再搞一次 $(date '+%F %H:%M:%S') ==="

verify() {
python3 - "$DIR/model.safetensors" <<'PY'
import sys, json, struct, pathlib
p = pathlib.Path(sys.argv[1]); sz = p.stat().st_size
with open(p,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
end = 8 + n + max((v['data_offsets'][1] for k,v in hdr.items() if k!='__metadata__'), default=0)
print("size=%d declared_end=%d %s" % (sz, end, "COMPLETE" if end<=sz else "TRUNCATED(%.1f%%)" % (100.0*sz/end)))
PY
}
echo "--- 下载前状态 ---"; verify

echo "--- ①-1 装 huggingface_hub 并尝试官方端点 ---"
python3 -m pip install -q -U "huggingface_hub[cli]" 2>&1 | tail -2
export HF_HUB_ENABLE_HF_TRANSFER=0
timeout 3000 python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('incoai/Qwen3.8-27B-DFlash2', local_dir=r'$DIR', max_workers=4)
print('snapshot ok ->', p)
" 2>&1 | tail -6
echo "hf rc=$?"
verify
if ! python3 -c "
import sys, json, struct, pathlib
p = pathlib.Path('$DIR/model.safetensors'); sz = p.stat().st_size
with open(p,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
end = 8 + n + max((v['data_offsets'][1] for k,v in hdr.items() if k!='__metadata__'), default=0)
sys.exit(0 if end<=sz else 1)
"; then
  echo "--- ①-2 官方端点未完成，改走镜像端点 hf-mirror.com ---"
  HF_ENDPOINT=https://hf-mirror.com timeout 3000 python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('incoai/Qwen3.8-27B-DFlash2', local_dir=r'$DIR', max_workers=4)
print('snapshot ok ->', p)
" 2>&1 | tail -6
  export HF_ENDPOINT=https://hf-mirror.com
  verify
fi

echo
echo "--- ② 全新 vllm 0.29（python3.12） ---"
V=/home/user/vllm029
rm -rf "$V"
python3.12 -m venv "$V" || { echo VENV_FAIL; exit 3; }
VP=$V/bin/python
$VP -m pip install -q -U pip setuptools wheel 2>&1 | tail -2
$VP -V
$VP -m pip install "$W" 2>&1 | tail -12
echo "pip rc=${PIPESTATUS[0]}"
$VP -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -2
$VP -c "import typing; from vllm.config.speculative import SpeculativeMethod; a=typing.get_args(SpeculativeMethod); print('dflash2' in a, 'dflash' in a, 'dspark' in a)" 2>&1 | tail -2
$VP -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -2
ls -1 $V/lib/python3.12/site-packages/vllm/*.so 2>/dev/null | head -4
echo REDO_DONE
