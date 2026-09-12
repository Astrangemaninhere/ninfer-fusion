#!/bin/bash
# 修完后的统一验证：重编 → 跑特征探针（--no-cuda-graph）→ 判定 tap1..4 非零 → 复测接受率
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/verify_fix.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 修复验证 $(date '+%F %H:%M:%S') ==="

echo "--- 1) 编译 ---"
cd "$R/build" || exit 3
touch $R/src/targets/qwen3_6_27b/impl/variant.cpp $R/src/targets/muse_glimmer_30b/impl/variant.cpp
make ninfer -j3 2>&1 | tail -3
rc=${PIPESTATUS[0]}; echo "make rc=$rc"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 4; }

echo "--- 2) 特征链探针（--no-cuda-graph） ---"
rm -f $J/dl/feat_features_*.bin
NINFER_DF2FEAT=1 timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 16 --max-context 4096 --no-thinking --greedy --spec dflash2 \
  --no-cuda-graph > $J/dl/verify_feat_stdout.log 2>&1
echo "  rc=$?"

echo "--- 3) 判定 tap 范数 ---"
/home/user/vllm029/bin/python - <<'PYEOF'
import numpy as np, pathlib
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ok = False
for call in range(1, 9):
    f = J / ("feat_features_%d.bin" % call)
    if not f.exists() or f.stat().st_size == 0:
        continue
    v = np.fromfile(f, dtype=np.float16).astype(np.float32)
    cols = v.size // 25600
    v = v.reshape(25600, cols)
    norms = [float(np.linalg.norm(v[i*5120:(i+1)*5120])) for i in range(5)]
    nonzero = sum(1 for x in norms if x > 1e-6)
    print("  call=%d cols=%d tap norms=%s 非零段=%d/5" % (call, cols, ["%.2f" % x for x in norms], nonzero))
    if nonzero == 5:
        ok = True
print("  VERDICT:", "PASS（5 段全非零）" if ok else "FAIL（仍有零段）")
PYEOF

echo "--- 4) 接受率复测（与基线 4.81% / 20,3,0,0,0,0,0 对比） ---"
NINFER_DF2DBG=1 timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 \
  > $J/dl/verify_accept.log 2>&1
grep -E 'dflash2 acceptance rate|dflash2 accepted by pos|dflash2 acceptance length|decode speed' $J/dl/verify_accept.log | sed 's/^/  /'
echo VERIFY_FIX_DONE
