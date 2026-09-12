set -u
W=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== _tu_timeline.py? ==="
ls -la $W/_tu_timeline.py $W/_ccache_probe.sh $W/_par_build.sh $W/_jit_flags_probe.sh $W/_land_split.sh 2>/dev/null
echo "=== grep 171 across recent _*.py/log candidates ==="
grep -l '171' $W/_tu_timeline.py $W/_*.log 2>/dev/null | head -10
echo "=== grep 'small_t' in _tu_timeline.py ==="
grep -n 'small_t\|decode' $W/_tu_timeline.py 2>/dev/null | head -20
echo "=== any file mentioning 171 s + small_t ==="
grep -rn '171' $W/_collab/build/*.md 2>/dev/null | head -10
echo "=== build object mtimes ordered (current build) ==="
find /home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir -name '*.o' -newermt '2026-09-10 20:00' -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -25
