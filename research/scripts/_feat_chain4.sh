#!/bin/bash
# 加 --no-cuda-graph 重跑特征探针（图捕获期间不能 sync）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/feat4.log
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || exit 3
echo "=== --no-cuda-graph 特征探针 $(date '+%H:%M:%S') ==="
rm -f $J/dl/feat_features_*.bin $J/dl/feat_projected_*.bin $J/dl/feat_context_*.bin
NINFER_DF2FEAT=1 timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 16 --max-context 4096 --no-thinking --greedy --spec dflash2 \
  --no-cuda-graph > $J/dl/feat_stdout4.log 2>&1
echo "  rc=$?"
grep -E 'df2feat' $J/dl/feat_stdout4.log | head -9
grep -E 'dflash2 acceptance rate|dflash2 accepted by pos' $J/dl/feat_stdout4.log | sed 's/^/  /'

echo "--- 各次调用、各 tap 段范数 ---"
/home/user/vllm029/bin/python - <<'PYEOF'
import numpy as np, pathlib, json, struct
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ART = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer")
for call in range(1, 9):
    f = J / ("feat_features_%d.bin" % call)
    if not f.exists() or f.stat().st_size == 0:
        continue
    v = np.fromfile(f, dtype=np.float16).astype(np.float32)
    cols = v.size // 25600
    v = v.reshape(25600, cols)
    norms = [float(np.linalg.norm(v[i*5120:(i+1)*5120])) for i in range(5)]
    print("  call=%d cols=%d tap norms=%s total=%.4f" %
          (call, cols, ["%.3f" % x for x in norms], float(np.linalg.norm(v))))
    # 首列各 tap 的前 4 个数值，便于目视
    for i in range(5):
        print("      tap%d [0,:4] = %s" % (i, np.round(v[i*5120:i*5120+1, :4], 3).tolist()))
PYEOF
echo FEAT4_DONE
