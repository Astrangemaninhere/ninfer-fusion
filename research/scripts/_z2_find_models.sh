#!/bin/bash
# Locate draft/target model artifacts relevant to DFlash2.
echo "=== A: config.json files under ziqinzhang mentioning dflash2/DFlash2 ==="
grep -rl -i 'dflash2' /mnt/c/Users/User/Documents/ziqinzhang --include='config.json' 2>/dev/null | head -20
echo ""
echo "=== B: config.json with DFlash2DraftModel architecture anywhere on C: (bounded) ==="
grep -rl 'DFlash2DraftModel' /mnt/c/Users/User/Documents 2>/dev/null | head -20
echo ""
echo "=== C: dirs named like dflash2 under ziqinzhang ==="
find /mnt/c/Users/User/Documents/ziqinzhang -maxdepth 4 -iname '*dflash2*' 2>/dev/null | head -30
echo ""
echo "=== D: data/ dir contents ==="
ls -la /mnt/c/Users/User/Documents/ziqinzhang/data/ 2>/dev/null | head -30
echo ""
echo "=== E: any draft* dirs ==="
find /mnt/c/Users/User/Documents/ziqinzhang -maxdepth 4 -type d -iname '*draft*' 2>/dev/null | head -20
echo ""
echo "=== F: .ninfer anywhere ==="
find /mnt/c/Users/User/Documents/ziqinzhang -maxdepth 4 -name '.ninfer' 2>/dev/null | head -10
echo ""
echo "=== G: qwen3.8-27b artifact doc ==="
find /mnt/c/Users/User/Documents/ziqinzhang /home/user/ninfer-fusion -name '*artifact*.md' 2>/dev/null | head -10
echo ""
echo "=== H: layouts.py ==="
find /mnt/c/Users/User/Documents/ziqinzhang /home/user/ninfer-fusion -name 'layouts.py' 2>/dev/null | head -10
