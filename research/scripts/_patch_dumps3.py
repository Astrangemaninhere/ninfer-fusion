#!/usr/bin/env python3
"""扩 dump：在同一次 selector 调用后追加 logits / proposal_hidden / projected。

动机：定位"深度 ≥2 接受精确为零"（355 次机会 0 次）——查深度 ≥1 的 hidden/logits 是否结构性退化。
纯新增；锚点断言唯一；行尾自动探测。
"""
import pathlib
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
rel = "src/targets/qwen3_6/impl/runtime/dflash2_impl.h"
p = R / rel
raw = p.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"
BAK = pathlib.Path("/home/user/df2scores_bak") / rel
BAK.parent.mkdir(parents=True, exist_ok=True)
BAK.write_bytes(raw.encode("utf-8"))
print("已备份 ->", BAK)

anchor = nl.join([
    '            dump_one("front", frontiers);',
    '            dump_one("anch", anchors);',
    "",
])
if raw.count(anchor) != 1:
    print(f"FAIL 锚点命中 {raw.count(anchor)} 次"); sys.exit(3)

repl = nl.join([
    '            dump_one("front", frontiers);',
    '            dump_one("anch", anchors);',
    '            // 深度结构探针：头的 logits、喂头的 hidden、selector 的 rank-256 hidden。',
    '            // 用于定位"深度 >=2 接受精确为零"（深度 >=1 的 hidden/logits 是否退化）。',
    '            dump_one("logits", logits);',
    '            dump_one("ph", proposal_hidden);',
    '            dump_one("proj", projected);',
    "",
])
p.write_bytes(raw.replace(anchor, repl, 1).encode("utf-8"))

chk = p.read_bytes().decode("utf-8")
assert chk.count('dump_one("logits", logits)') == 1
assert chk.count('dump_one("ph", proposal_hidden)') == 1
assert chk.count('dump_one("proj", projected)') == 1
print("OK 三个新 dump 已落")
