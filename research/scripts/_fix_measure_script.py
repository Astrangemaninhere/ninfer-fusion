#!/usr/bin/env python3
"""修测量脚本的两个问题：
  1) serve 没加 `--no-thinking` ⇒ content 为空、输出全在 reasoning_content，
     双侧空文本会让 cmp 报 IDENTICAL（等于用"空比对空"证明 verifier 精确）——这是最危险的假结论；
  2) 由此加防呆：任一侧文本为空 ⇒ 报 INVALID，绝不报 IDENTICAL。
"""
import pathlib

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_post_build_measure.sh")
src = P.read_text(encoding="utf-8")

# 1) 加 --no-thinking
old_serve = '''  setsid nohup "$BIN" "$art" --port $PORT --max-context 4096 --no-cuda-graph "$@" \\'''
new_serve = '''  # --no-thinking is mandatory here: without it the answer lands in reasoning_content and
  # `content` comes back empty, so an exactness diff would compare "" vs "" and report
  # IDENTICAL - proving nothing. (Measured 17:44: response content "" with reasoning_content.)
  setsid nohup "$BIN" "$art" --port $PORT --max-context 4096 --no-cuda-graph --no-thinking "$@" \\'''
assert src.count(old_serve) == 1, "serve 锚点不唯一"
src = src.replace(old_serve, new_serve)

# 2) 防呆：空文本 = INVALID
old_cmp = '''  if [ ! -s "$a" ] || [ ! -s "$b" ]; then echo "[$name] missing text"; return; fi'''
new_cmp = '''  if [ ! -s "$a" ] || [ ! -s "$b" ]; then
    # An empty side is NOT agreement: it means the request produced no content at all
    # (thinking mode, parse failure, or a rejected request), so say INVALID loudly.
    echo "[$name] INVALID - empty text on at least one side ($(wc -c < "$a" 2>/dev/null) vs $(wc -c < "$b" 2>/dev/null) bytes); NOT evidence of anything"
    return
  fi'''
assert src.count(old_cmp) == 1, "cmp 锚点不唯一"
src = src.replace(old_cmp, new_cmp)

P.write_text(src, encoding="utf-8")
print("已修：--no-thinking + 空文本 INVALID 防呆")
