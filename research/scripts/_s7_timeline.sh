set -u
W=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== dl/ dir ==="
ls -la $W/dl/ 2>/dev/null | head -20
echo "=== jit_flags_probe.log tail ==="
tail -60 $W/dl/jit_flags_probe.log 2>/dev/null || echo "no jit log"
echo "=== run _tu_timeline.py (read-only) ==="
python3 $W/_tu_timeline.py 2>&1 | head -60
