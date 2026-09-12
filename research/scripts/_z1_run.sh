#!/bin/bash
# Z1 orchestrator: guard -> run vLLM DSpark acceptance measurement on the Windows side.
# Usage: bash /tmp/z1_run.sh
set -u
OUT=/mnt/c/Users/User/Documents/ziqinzhang/_z1_run.log
WINPY=/mnt/c/vllm/venv/Scripts/python.exe
SCRIPT=/mnt/c/Users/User/Documents/ziqinzhang/_z1_measure.py

echo "================ Z1 RUN $(date -Is) ================"

# ---------------- HARD GUARD ----------------
NVCC=$(pgrep -x nvcc | wc -l)
PARB=$(pgrep -f '[_]par_build' | wc -l)
PARB_EXTRA=$(pgrep -f 'par_build[.]sh' | wc -l)
MEMAVAIL=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "GUARD: nvcc=$NVCC  _par_build=$PARB  par_build.sh=$PARB_EXTRA  MemAvailable=${MEMAVAIL}kB  GPU_used=${GPU_USED}MiB"

FAIL=0
if [ "$NVCC" -ne 0 ]; then echo "GUARD FAIL: nvcc alive"; FAIL=1; fi
if [ "$PARB" -ne 0 ]; then echo "GUARD FAIL: _par_build alive"; FAIL=1; fi
if [ "$PARB_EXTRA" -ne 0 ]; then echo "GUARD FAIL: par_build.sh alive"; FAIL=1; fi
if [ "$MEMAVAIL" -lt 8000000 ]; then echo "GUARD FAIL: MemAvailable < 8GB"; FAIL=1; fi
if [ "$FAIL" -ne 0 ]; then echo "ABORT: guard not satisfied, not touching the GPU."; exit 3; fi
echo "GUARD OK"

# ---------------- inputs ----------------
TGT=/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090
DRF=/mnt/c/Users/User/Documents/ziqinzhang/data/draft_model
for f in "$TGT/config.json" "$TGT/model.safetensors.index.json" "$TGT/model-00001-of-00002.safetensors" "$TGT/model-00002-of-00002.safetensors" "$DRF/config.json" "$DRF/model.safetensors"; do
  if [ ! -f "$f" ]; then echo "MISSING INPUT: $f"; exit 4; fi
done
echo "inputs present:"
ls -la "$TGT"/*.safetensors "$DRF"/model.safetensors

# ---------------- run ----------------
echo "--- launching Windows vLLM (log: _z1_server.log) ---"
"$WINPY" "$SCRIPT" 2>&1
RC=$?
echo "measure rc=$RC"

# ---------------- side-by-side ----------------
if [ -f /mnt/c/Users/User/Documents/ziqinzhang/_z1_result.json ]; then
  echo
  python3 /mnt/c/Users/User/Documents/ziqinzhang/_z1_compare.py 2>&1 | tee /mnt/c/Users/User/Documents/ziqinzhang/_z1_compare.txt
else
  echo "no _z1_result.json -> measurement did not complete; see _z1_server.log"
fi

echo "================ DONE $(date -Is) ================"
exit $RC
