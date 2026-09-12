#!/usr/bin/env python3
"""S28 的调用点：在 `namespace ninfer::targets::qwen3_6` 内写 `ops::kv_rowscale_sidecar_apply_from_env(...)`，
include 回到顶层后这个名字解析不到（`ops` 既非本命名空间成员、也未在父命名空间中可见）。
改为完全限定 `::ninfer::ops::...`（最小且无歧义）。"""
import pathlib
import re
import sys

P = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/state/decoder_state.cpp")
lines = P.read_text(encoding="utf-8").splitlines(keepends=True)
hit = [i for i, l in enumerate(lines) if "kv_rowscale_sidecar_apply_from_env" in l and "ops::" in l
       and not l.strip().startswith("//")]
print("命中调用点 %d 处：" % len(hit))
for i in hit:
    print("  %d: %s" % (i + 1, lines[i].rstrip()[:120]))
assert hit, "没找到调用点"
for i in hit:
    lines[i] = lines[i].replace("ops::kv_rowscale_sidecar_apply_from_env",
                                "::ninfer::ops::kv_rowscale_sidecar_apply_from_env")
P.write_text("".join(lines), encoding="utf-8")
print("已改为完全限定名")
