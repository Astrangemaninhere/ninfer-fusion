#!/bin/bash
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 重跑分析层修复（显示错误） ==="
python3 "$J/_fix_analysis_layer.py"
echo "  rc=$?"
ls -l "$J/_feat_canonical_read.py" 2>/dev/null | awk '{print "  canonical reader:", $5, $9}'
grep -c "附注（2026-09-12）" "$J/_collab/A5_dspark_rootcause.md" 2>/dev/null | sed 's/^/  A5 附注命中: /'

echo
echo "=== 2) 决定性验证：pending（sink 目的地）应为 8 列 × 5 tap 全非零 ==="
/home/user/vllm029/bin/python - <<'PYEOF'
import numpy as np, pathlib
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
for name in ("feat_pending_5.bin", "feat_features_5.bin"):
    p = J / name
    if not p.exists() or p.stat().st_size == 0:
        print("  缺:", name); continue
    raw = np.fromfile(p, dtype="<f2")
    nz = np.nonzero(raw)[0]
    span = (int(nz[0]), int(nz[-1]) + 1) if nz.size else (0, 0)
    cols = raw.size // 25600
    print("  %s: 元素=%d 非零=%d(%.1f%%) 非零span=%s 推断列数=%d" %
          (name, raw.size, nz.size, 100.0*nz.size/raw.size, span, cols))
    v = raw.astype(np.float32).reshape(cols, 25600)
    live = [c for c in range(cols) if np.any(v[c])]
    print("    非零(live)列 =", live[:8], " 共", len(live), "列")
    for c in live[:2]:
        taps = [float(np.linalg.norm(v[c, i*5120:(i+1)*5120])) for i in range(5)]
        print("    列%d 的 5 个 tap 范数 = %s" % (c, ["%.2f" % t for t in taps]))
PYEOF
echo CANON_VERIFY_DONE
