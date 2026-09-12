echo "=== find weight_residency.h under /home ==="
find /home -name 'weight_residency.h' 2>/dev/null | head
echo "=== find A_s32_w13_p0.md under /home ==="
find /home -name 'A_s32_w13_p0*' 2>/dev/null | head
echo "=== find mkpatch under /home ==="
find /home -name 'E9_s52_mkpatch.py' 2>/dev/null | head
echo "=== find serve_options.cpp under /home ==="
find /home -name 'serve_options.cpp' 2>/dev/null | head
echo "=== _orig_quarantine ==="
ls -la /home/user/ninfer-fusion/_orig_quarantine
echo "=== _collab-ish dirs under /home ==="
find /home -maxdepth 5 -type d -name '*collab*' 2>/dev/null | head -20
echo "=== other ninfer trees under /home/user ==="
ls -la /home/user/
