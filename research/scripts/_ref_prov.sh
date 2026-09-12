#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) dl.err ==="; cat $J/data/draft_dflash2_ref/dl.err 2>/dev/null
echo "=== 2) dl2.err ==="; cat $J/data/draft_dflash2_ref/dl2.err 2>/dev/null
echo
echo "=== 3) 工作区脚本里提到 draft_dflash2_ref / hf download 的（单目录 glob） ==="
for f in $J/*.sh $J/*.ps1 $J/*.bat; do
  if grep -lqiE 'draft_dflash2_ref|DFlash2DraftModel|incoai|hf_transfer|huggingface-cli download|snapshot_download' "$f" 2>/dev/null; then
    echo "--- $f"; grep -niE 'draft_dflash2_ref|incoai|repo_id|download|--local-dir' "$f" | head -8
  fi
done
echo
echo "=== 4) safetensors 是否完整（读头 + 张量名，零拷贝） ==="
python3 - "$J/data/draft_dflash2_ref/model.safetensors" <<'PY'
import sys, json, struct, pathlib
p = pathlib.Path(sys.argv[1]); sz = p.stat().st_size
with open(p, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
keys = [k for k in hdr if k != '__metadata__']
print("file bytes=%d  header=%d  tensors=%d" % (sz, n, len(keys)))
print("示例键:", keys[:10])
data_end = 8 + n + max((v['data_offsets'][1] for k, v in hdr.items() if k != '__metadata__'), default=0)
print("声明数据末端=%d  文件大小=%d  => %s" % (data_end, sz, "完整" if data_end <= sz else "截断!"))
PY
echo
echo "=== 5) WSL 有哪些 python ==="
for v in 3.12 3.13 3.11; do command -v python$v >/dev/null && echo "  有 python$v: $(python$v -V 2>&1)"; done
