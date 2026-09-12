#!/usr/bin/env python3
"""修 S35 编译错误（v2：全部按行索引，不做大字符串匹配）。
ResourceInspection 含 std::optional<Choice>，而 Choice 的移动/拷贝赋值都被 delete
(resource_manager.h:120-123)，所以 S35 引入的 `x = inspect_admission(y);` 编不过；
Choice 可移动构造 ⇒ 改成 std::optional<ResourceInspection> + emplace，使用点改 `->`，
并在 catch（必定 continue）之后插入一条不可达守卫把不变量写清楚。"""
import pathlib
import sys

P = pathlib.Path("/home/user/ninfer-fusion/src/runtime/engine/engine_core.h")
lines = P.read_text(encoding="utf-8").splitlines(keepends=True)
orig = list(lines)

def idx_of(pred, expect, what):
    hits = [i for i, l in enumerate(lines) if pred(l)]
    assert len(hits) == expect, "%s: 期望 %d，实际 %d" % (what, expect, len(hits))
    return hits

# 1) 声明 -> optional
for i in idx_of(lambda l: l.strip() in ("ResourceInspection head_inspection;",
                                       "ResourceInspection candidate_inspection;"), 2, "声明"):
    ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    name = "head_inspection" if "head_inspection" in lines[i] else "candidate_inspection"
    lines[i] = "%sstd::optional<ResourceInspection> %s;\n" % (ind, name)

# 2) 赋值 -> emplace
assign_idx = idx_of(lambda l: "= inspect_admission(" in l and l.strip().startswith(
    ("head_inspection", "candidate_inspection")), 2, "赋值")
for i in assign_idx:
    ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lhs, rhs = lines[i].split("=", 1)
    lines[i] = "%s%s.emplace(%s);\n" % (ind, lhs.strip(), rhs.strip().rstrip(";"))

# 3) 使用点改 `->`（变量名后紧跟 . 的所有行）
for i, l in enumerate(lines):
    if "head_inspection." in l or "candidate_inspection." in l:
        lines[i] = l.replace("head_inspection.", "head_inspection->").replace(
            "candidate_inspection.", "candidate_inspection->")

# 4) catch 块（必定 continue）之后插守卫：找每处赋值之后最近的 `continue;` 行的下一个 `}`
for i, name in zip(assign_idx, ("head_inspection", "candidate_inspection")):
    j = i
    while j < len(lines) and "continue;" not in lines[j]:
        j += 1
    assert j < len(lines), "%s: 找不到 continue" % name
    k = j
    while k < len(lines) and lines[k].strip() != "}":
        k += 1
    assert k < len(lines), "%s: 找不到 catch 结束括号" % name
    ind = lines[j][:len(lines[j]) - len(lines[j].lstrip())]          # continue 的缩进
    lines.insert(k + 1,
                 "%sif (!%s) {\n" % (ind, name) +
                 "%s    // unreachable: the catch above always continues\n" % ind +
                 '%s    throw std::logic_error("S35: admission inspection missing");\n' % ind +
                 "%s}\n" % ind)

P.write_text("".join(lines), encoding="utf-8")
changed = sum(1 for a, b in zip(orig, lines) if a != b)
print("已写入：改动 %d 行（原 %d 行 -> 现 %d 行）" % (changed, len(orig), len(lines)))
for i, l in enumerate(lines, 1):
    if "head_inspection" in l or "candidate_inspection" in l:
        print("  %4d %s" % (i, l.rstrip()[:120]))
