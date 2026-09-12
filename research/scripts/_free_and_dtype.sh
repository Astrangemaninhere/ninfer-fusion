#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 清理我造成的大件（只删我自己的副本/缓存，源件都在 /mnt/c） ==="
echo "--- pip 缓存（12G，纯缓存，可重建） ---"
rm -rf /home/user/.cache/pip && echo "已清"
echo "--- WSL 侧 17G 模型拷贝（源在 /mnt/c/Users/User/Documents/ziqinzhang/models/...） ---"
du -sh /home/user/models/qwen3_8_27b_nvfp4_hf 2>/dev/null
rm -rf /home/user/models/qwen3_8_27b_nvfp4_hf && echo "已清（省 17G）"
echo "--- 清理结果 ---"
df -h / | tail -1
du -sh /home/user/models 2>/dev/null
echo
echo "=== 2) 目标模型 dtypes（safetensors 头，零拷贝） ==="
python3 - "$J/models/Qwen3.8-27B-NVFP4-RTX5090/model-00001-of-00002.safetensors" <<'PY'
import sys, json, struct, pathlib, collections
p = pathlib.Path(sys.argv[1])
with open(p,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
dt = collections.Counter(v.get('dtype','?') for k,v in hdr.items() if k != '__metadata__')
print("tensors =", sum(dt.values()))
for d, c in dt.most_common():
    print("  %-12s x%d" % (d, c))
keys = [k for k in hdr if k != '__metadata__']
import re
for pat in ('weight_scale', 'input_scale', 'weight_scale_2', 'kv'):
    hits = [k for k in keys if pat in k][:2]
    for h in hits:
        print("  %-70s %s %s" % (h[:70], hdr[h]['dtype'], hdr[h]['shape']))
PY
