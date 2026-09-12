#!/bin/bash
# 应用 scale 探针 -> 增量重编 -> 带 NINFER_DF2SEL 跑一轮 -> 汇总 E/u/pair/uspan。
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/scale_probe.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== scale probe $(date '+%F %H:%M:%S') ==="
python3 $J/_apply_df2_scale_probe.py || { echo PATCH_FAILED; exit 2; }

cd $R/build || exit 3
make ninfer -j2 2>&1 | tail -12
rc=${PIPESTATUS[0]}
echo "make rc=$rc"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 1; }

P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
echo "--- run with NINFER_DF2SEL=1 (full head) ---"
NINFER_DF2SEL=1 timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --spec dflash2 > $J/dl/scale_probe_stdout.log 2> $J/dl/scale_probe_stderr.log
echo "rc=$? sel_lines=$(grep -c df2sel $J/dl/scale_probe_stderr.log)"

echo "--- decomposition of the first drafting round (first 16 lines) ---"
grep df2sel $J/dl/scale_probe_stderr.log | head -16

echo "--- aggregate: |pair| vs uspan over all steps ---"
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib, re, statistics
pat = re.compile(r"\[df2sel\] s=(-?\d+) pred=(-?\d+) chosen=(-?\d+) tok=(-?\d+) "
                 r"E=(-?[\d.]+) u=(-?[\d.]+) pair=(-?[\d.]+) uspan=(-?[\d.]+)")
rows = []
for line in pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl/scale_probe_stderr.log") \
        .read_text(errors="replace").splitlines():
    m = pat.search(line)
    if m:
        rows.append([float(m.group(i)) for i in range(5, 9)])
if not rows:
    print("no probe rows")
else:
    apair = [abs(r[2]) for r in rows]
    uspan = [r[3] for r in rows]
    ratio = [abs(r[2]) / r[3] if r[3] > 0 else 0.0 for r in rows]
    print(f"steps={len(rows)}")
    print(f"|pair|  median={statistics.median(apair):.3f}  max={max(apair):.3f}")
    print(f"uspan   median={statistics.median(uspan):.3f}  max={max(uspan):.3f}")
    print(f"ratio |pair|/uspan  median={statistics.median(ratio):.3f}  max={max(ratio):.3f}")
    print(f"steps where |pair| >= 0.25*uspan: {sum(1 for r in ratio if r >= 0.25)}/{len(rows)}")
    print()
    print("per-step first round:")
    for r in rows[:14]:
        print("   E=%8.3f u=%8.3f pair=%8.3f uspan=%7.3f" % tuple(r))
PY
echo SCALE_PROBE_DONE
