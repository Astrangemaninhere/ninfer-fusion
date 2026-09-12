set -e
cd /home/user/ninfer-fusion
echo "=== pwd ==="
pwd
echo "=== _collab listing ==="
ls -la _collab/ | head -80
echo "=== find A_s32_w13_p0 ==="
find . -name 'A_s32_w13_p0*' 2>/dev/null
echo "=== git status short ==="
git status --short 2>/dev/null | head -40
