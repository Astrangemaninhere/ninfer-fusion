#!/bin/bash
# rebuild the engine in the WSL tree (incremental)
set -u
cd /home/user/ninfer-fusion/build || exit 1
J="${1:-12}"
echo "cores=$(nproc) jobs=$J start=$(date +%H:%M:%S)"
make -j"$J" 2>&1 | tail -50
rc=${PIPESTATUS[0]}
echo "BUILD_RC=$rc end=$(date +%H:%M:%S)"
