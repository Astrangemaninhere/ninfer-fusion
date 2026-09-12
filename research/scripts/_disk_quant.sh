#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 磁盘 ==="
df -h / /mnt/c 2>/dev/null | head -4
echo
echo "=== 2) 我在 WSL 侧造的大件（单目录定点） ==="
du -sh /home/user/models/qwen3_8_27b_nvfp4_hf 2>/dev/null
du -sh /home/user/models/draft_dflash2_ref 2>/dev/null
du -sh /home/user/vllm029 2>/dev/null
du -sh /home/user/.cache/pip 2>/dev/null
du -sh /home/user/models 2>/dev/null
echo
echo "=== 3) 内存 ==="
free -g | head -2
echo
echo "=== 4) 目标模型量化配置（实证，读文件） ==="
cat $J/models/Qwen3.8-27B-NVFP4-RTX5090/hf_quant_config.json 2>/dev/null
echo "--- config.json 里的量化相关字段 ---"
grep -iE 'quant|dtype|fp8|nvfp4|int4|w4' $J/models/Qwen3.8-27B-NVFP4-RTX5090/config.json 2>/dev/null | head -10
echo
echo "=== 5) safetensors 头里的 dtype 分布（零拷贝读头） ==="
python3 - "$J/models/Qwen3.8-27B-NVFP4-RTX5090/model-00001-of-00002.safetensors" <<'PY'
import sys, json, struct, pathlib, collections
p = pathlib.Path(sys.argv[1])
with open(p,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
dt = collections.Counter(v.get('dtype','?') for k,v in hdr.items() if k != '__metadata__')
print("total tensors =", sum(dt.values()))
for d, c in dt.most_common():
    print("  dtype %-12s x%d" % (d, c))
k = [k for k in hdr if k != '__metadata__'][:6]
for kk in k:
    print("  e.g. %-60s %s %s" % (kk, hdr[kk].get('dtype'), hdr[kk].get('shape')))
PY
