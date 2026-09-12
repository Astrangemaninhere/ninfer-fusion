#!/usr/bin/env python3
"""修重建脚本：显式把 ccache 所在目录加进 PATH。
（本树用 `ccache nvcc` 作为 launcher，而 ccache 装在 /home/user/.local/bin；
由不同 shell 启动 make 时 PATH 不同 ⇒ 曾出现 `ccache: not found` Error 127。）"""
import pathlib

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_rebuild_after_s35.sh")
src = P.read_text(encoding="utf-8")
line = 'export NVCC_PREPEND_FLAGS="--split-compile-extended=8"'
if "/home/user/.local/bin" in src:
    print("PATH 已修")
else:
    assert src.count(line) == 1
    src = src.replace(line, line + '\n# ccache lives in ~/.local/bin and the build launcher is `ccache nvcc`:\n# without this the make dies with "ccache: not found" (Error 127).\nexport PATH="/home/user/.local/bin:$PATH"')
    P.write_text(src, encoding="utf-8")
    print("已补 PATH")
print("ccache 可见性:", pathlib.Path("/home/user/.local/bin/ccache").exists())
