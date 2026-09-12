#!/usr/bin/env python3
""".ninfer 头部 objects[] 的结构总览：判断能否按原位量化布局直转到 vLLM。"""
import json, pathlib, struct
from collections import Counter, defaultdict

P = "/home/user/models/qwen3_8_27b_nvfp4.ninfer"
with open(P, "rb") as f:
    magic = f.read(8)
    n = struct.unpack("<Q", f.read(8))[0]
    j = json.loads(f.read(n).decode("utf-8", "replace"))

print("magic=%s  json=%d B" % (magic.hex(), n))
print("顶层键: %s" % ", ".join(sorted(j.keys())))
print("identity: %s" % json.dumps(j.get("identity"), ensure_ascii=False))

objs = j.get("objects")
print("\nobjects: 类型=%s 条数=%s" % (type(objs).__name__, len(objs) if hasattr(objs,'__len__') else '?'))
if isinstance(objs, dict):
    items = list(objs.items())[:3]
    print("  （dict 形态）前 3 条:")
    for k, v in items:
        print("   %s -> %s" % (k, json.dumps(v, ensure_ascii=False)[:400]))
elif isinstance(objs, list):
    print("  （list 形态）第 1 条:")
    print("   %s" % json.dumps(objs[0], ensure_ascii=False)[:600])

# 统计 dtype / layout / 量化
def walk(objs):
    rows = []
    if isinstance(objs, dict):
        it = objs.values()
    else:
        it = objs
    for o in it:
        if isinstance(o, dict):
            rows.append(o)
    return rows

rows = walk(objs)
print("\n共 %d 条对象" % len(rows))
if rows:
    keys = Counter()
    for r in rows:
        for k in r.keys():
            keys[k] += 1
    print("出现的字段（字段:条数）:")
    for k, c in keys.most_common(20):
        print("   %-22s %d" % (k, c))
    print()
    for field in ("dtype", "layout", "quantization", "quant", "format", "kind", "role"):
        c = Counter(str(r.get(field)) for r in rows if field in r)
        if c:
            print("%s 分布: %s" % (field, dict(c.most_common(10))))
    # 名字前缀分组
    names = [str(r.get("name") or r.get("id") or "") for r in rows]
    pref = Counter(n.split(".")[0] + "." + (n.split(".")[1] if n.count(".") >= 1 else "") for n in names if n)
    print("\n名字前两段分组 top15:")
    for k, c in pref.most_common(15):
        print("   %-40s %d" % (k, c))
    # 打印几条带量化信息的样本
    print("\n带量化/布局字段的样本（前 4 条）:")
    shown = 0
    for r in rows:
        if any(k in r for k in ("dtype", "layout", "quantization", "scale_dtype", "group")):
            print("   %s" % json.dumps(r, ensure_ascii=False)[:420])
            shown += 1
            if shown >= 4: break
