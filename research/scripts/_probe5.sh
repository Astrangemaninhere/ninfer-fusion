W=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== $W ==="
ls -la "$W"
echo "=== _collab ==="
ls -la "$W/_collab" 2>/dev/null | head -60
echo "=== _collab/build ==="
ls -la "$W/_collab/build" 2>/dev/null | head -60
echo "=== find A_s32_w13_p0 ==="
find "$W" -name 'A_s32_w13_p0*' 2>/dev/null | head
echo "=== find E9_s52_mkpatch.py ==="
find "$W" -name 'E9_s52_mkpatch.py' 2>/dev/null | head
