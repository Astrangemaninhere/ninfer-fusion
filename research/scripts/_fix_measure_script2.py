#!/usr/bin/env python3
"""按实际文本修两处：serve 加 --no-thinking；exactness 快照把"任一侧为空"判 INVALID。"""
import pathlib

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_post_build_measure.sh")
src = P.read_text(encoding="utf-8")

old_serve = '''  setsid nohup "$BIN" "$art" --port $PORT --max-context 4096 --no-cuda-graph "$@" \\'''
new_serve = '''  # --no-thinking is mandatory here: without it the answer lands in reasoning_content and
  # `content` comes back empty, so the exactness diff would compare "" vs "" and report
  # IDENTICAL - proving nothing. (Measured 17:44 on the new binary: content "", reasoning set.)
  setsid nohup "$BIN" "$art" --port $PORT --max-context 4096 --no-cuda-graph --no-thinking "$@" \\'''
assert src.count(old_serve) == 1, "serve 锚点 %d" % src.count(old_serve)
src = src.replace(old_serve, new_serve)

old_py = '''        if base is None or other is None:
            print("[%s %s] missing text" % (name, pair)); continue
        if base == other:'''
new_py = '''        if base is None or other is None:
            print("[%s %s] missing text" % (name, pair)); continue
        if not base or not other:
            # An empty side is NOT agreement: no content came back (thinking mode, parse
            # failure, or a rejected request), so it cannot support any conclusion.
            print("[%s %s] INVALID - empty text on at least one side (%d vs %d chars); NOT evidence"
                  % (name, pair, len(base), len(other))); continue
        if base == other:'''
assert src.count(old_py) == 1, "py 锚点 %d" % src.count(old_py)
src = src.replace(old_py, new_py)

P.write_text(src, encoding="utf-8")
print("已修：serve --no-thinking + 空文本 INVALID（两处锚点各命中一次）")
