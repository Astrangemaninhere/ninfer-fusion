#!/bin/bash
T=/home/user/ninfer-fusion
echo "### current wrapper file (raw, numbered)"
cat -n $T/src/ops/wrapper/sigmoid_mul.cpp
echo "### current public header (raw, numbered)"
cat -n $T/include/ninfer/ops/sigmoid_mul.h
echo "### other sigmoid_mul.cpp copies anywhere reachable"
find /home/user /mnt/c/Users/User/Documents/ziqinzhang -name 'sigmoid_mul.cpp' -not -path '*/build/*' 2>/dev/null | head -20
echo "### windows-side candidate trees"
ls -d /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion* 2>/dev/null
ls -d /mnt/c/Users/User/Documents/ziqinzhang/*repo* 2>/dev/null
echo "### _collab dirs"
ls /mnt/c/Users/User/Documents/ziqinzhang/_collab/ | head -80
echo "### _collab/build"
ls /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/ 2>/dev/null | head -60
