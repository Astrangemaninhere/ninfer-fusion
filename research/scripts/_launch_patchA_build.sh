#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
bash -n "$J/_patchA_build.sh" && echo PA_BUILD_SYNTAX_OK
# make sure ccache is on PATH for this build (it lives in ~/.local/bin)
export PATH="/home/user/.local/bin:$PATH"
command -v ccache && ccache -s 2>/dev/null | head -3
setsid nohup bash "$J/_patchA_build.sh" > /dev/null 2>&1 < /dev/null &
sleep 8
pgrep -af '_patchA_build' | cut -c1-80
echo "--- log ---"
tail -12 "$J/dl/patchA_build.log"
