cd /home/user/ninfer-fusion
echo "=== top level ==="
ls -la
echo "=== find _collab anywhere ==="
find / -maxdepth 6 -type d -name '_collab' 2>/dev/null | head -20
echo "=== find ninfer dirs ==="
find /home/user -maxdepth 3 -type d 2>/dev/null | head -60
