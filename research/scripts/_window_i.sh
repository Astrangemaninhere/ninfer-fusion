#!/bin/bash
# Window I — user's ordering (2026-09-10): list first, main lines next, TRAINING LAST.
#
# Phase 1 (GPU, before the build): measurements that answer open questions and do
#   not depend on the patch batch:
#     * Muse bf16 vs nvfp4 — scope the NaN corruption to the quantized-KV path
#       (A/S36 confirmed the mechanism: hardcoded 256-dim leads vs Muse's 128).
#     * acceptance eval on the CURRENT checkpoint (step_001200) — mechanism data
#       for the user's position that low acceptance is NOT a training problem.
# Phase 2: hand over to `_window_h.sh`, which does U7 baseline -> apply the
#   11-patch batch -> ONE build (with ccache) -> judgement scripts -> and finally
#   resumes W9 training as its last step.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/window_i.log
: > "$LOG"
exec >> "$LOG" 2>&1
echo "=== window I start $(date +%F' '%H:%M:%S) ==="
echo "--- preflight: GPU must be free (training paused on purpose) ---"
"/mnt/c/Windows/System32/nvidia-smi.exe" --query-gpu=memory.used,memory.free --format=csv,noheader
if pgrep -af train_dflash2 | grep -v $$ > /dev/null; then
  echo "ABORT: training still alive — pause it deliberately first"; echo WINDI_FAIL; exit 3
fi
free -g | head -2

echo "--- [1/2] Muse bf16 vs nvfp4 comparison $(date +%H:%M:%S) ---"
bash "$J/_muse_kv_compare.sh"

echo "--- [2/2] acceptance eval on the current checkpoint $(date +%H:%M:%S) ---"
echo "(mechanism data at step_001200; the pre-registered gate is evaluated later on step_006000)"
bash "$J/_df2_accept_eval.sh" 'data\dflash2_ckpts\step_001200.pt' 300

echo "--- hand over to window H (patches -> ONE build -> judgements -> resume training LAST) $(date +%H:%M:%S) ---"
bash "$J/_window_h.sh"
echo "=== window I done $(date +%F' '%H:%M:%S) ==="
echo WINDI_DONE
