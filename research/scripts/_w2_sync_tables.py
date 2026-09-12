#!/usr/bin/env python3
"""把两张落地脚本的 md5 表同步到「已打 PART-B 的当前树」。
每处先断言旧值唯一命中；不一致就报错不动手。"""
import hashlib, pathlib, sys

J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang")
def md5f(p): return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()
def nlf(p): return pathlib.Path(p).read_bytes().count(b"\n")

G = pathlib.Path("/home/user/ninfer-fusion/src/ops/launcher")
cur_cu  = (md5f(G / "gqa_attention_decode.cu"),    nlf(G / "gqa_attention_decode.cu"))
cur_e8  = (md5f(G / "gqa_attention_decode_e8.cu"), nlf(G / "gqa_attention_decode_e8.cu"))
print(f"当前 gqa_attention_decode.cu    md5={cur_cu[0]}  lines={cur_cu[1]}")
print(f"当前 gqa_attention_decode_e8.cu md5={cur_e8[0]}  lines={cur_e8[1]}")

EDITS = [
  ("_land_split.sh", "e12ce1a6b9f952be898b374b398f7b65", "9c4a3ae3e71887ea2752f017135c6b7c"),
  ("_land_split.sh", "15c1794ab38ae44a085200cd73a80a52", "f709636e0b4720b7ad05e73c8ed0e09c"),
  ("_land_split.sh", f"b0a130e21e02e42a0d1f9834dfac8aae|737", f"{cur_cu[0]}|{cur_cu[1]}"),
  ("_land_s6.sh",    "42ba0988170a98acb6b6dcaab20ed00c", "5e2a0e7c34a26628cb3fad350917e23d"),
  ("_land_s6.sh",    f"da1da51f9a8902b1674493c180190beb|224", f"{cur_e8[0]}|{cur_e8[1]}"),
]

fails = 0
for fn, old, new in EDITS:
    p = J / fn
    txt = p.read_text()
    c = txt.count(old)
    if c != 1:
        print(f"  FAIL {fn}: 旧值出现 {c} 次（要求 1）: {old[:24]}"); fails += 1; continue
    txt = txt.replace(old, new)
    p.write_text(txt)
    print(f"  ok   {fn}: {old[:20]}... -> {new[:20]}...")
print("FAILS=%d" % fails)
sys.exit(1 if fails else 0)
