#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
F=$J/data/draft_dflash2_ref/model.safetensors
echo "=== 等下载收尾（最多 4 分钟） ==="
for i in $(seq 1 48); do
  s1=$(stat -c %s "$F" 2>/dev/null || echo 0); sleep 5
  s2=$(stat -c %s "$F" 2>/dev/null || echo 0)
  [ "$s1" = "$s2" ] && { echo "  大小稳定于 $s2（第 $((i*5))s）"; break; }
done
echo "--- 完整性 ---"
python3 - "$F" <<'PY'
import sys, json, struct, pathlib
p = pathlib.Path(sys.argv[1]); sz = p.stat().st_size
try:
    with open(p,'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
    end = 8 + n + max((v['data_offsets'][1] for k,v in hdr.items() if k!='__metadata__'), default=0)
    print("size=%d declared=%d => %s" % (sz, end, "COMPLETE ✓" if end<=sz else "TRUNCATED %.1f%%" % (100.0*sz/end)))
except Exception as e:
    print("读头失败:", type(e).__name__, e)
PY
echo
echo "=== 新 vllm029 环境状态 ==="
V=/home/user/vllm029
[ -x $V/bin/python ] && { $V/bin/python -V; $V/bin/python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -2; } || echo "  venv 未就绪"
tail -4 $J/dl/redo2.log 2>/dev/null
pgrep -f 'pip install' >/dev/null && echo "  pip 仍在跑" || echo "  pip 未在跑"
