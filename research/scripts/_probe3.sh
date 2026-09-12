echo "=== find weight_residency.h ==="
find / -name 'weight_residency.h' 2>/dev/null | head
echo "=== find A_s32_w13_p0.md ==="
find / -name 'A_s32_w13_p0*' 2>/dev/null | head
echo "=== find E9_s52_mkpatch.py ==="
find / -name 'E9_s52_mkpatch.py' 2>/dev/null | head
echo "=== find serve_options.cpp ==="
find / -name 'serve_options.cpp' 2>/dev/null | head
echo "=== _orig_quarantine ==="
ls -la /home/user/ninfer-fusion/_orig_quarantine
echo "=== look for _collab-ish dirs ==="
find /home /mnt/c/Users/User/Documents -maxdepth 5 -type d \( -name '*collab*' -o -name '*_collab*' \) 2>/dev/null | head -20
