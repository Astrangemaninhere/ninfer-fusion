#!/usr/bin/env python3
"""修 S35 补丁引入的编译错误（阻塞整条测量链）。

S35 (W16 §3.4) 想让"规划期异常只失败该请求"，做法是把
    auto head_inspection = inspect_admission(head);
改成
    ResourceInspection head_inspection;
    try { ...; head_inspection = inspect_admission(head); } catch (...) { ...continue; }
但 `ResourceInspection` 含 `std::optional<Choice>`，而 `Choice::operator=(Choice&&)` 是
deleted（resource_manager.h:120-123）⇒ **拷贝/移动赋值都被删除**，`head_inspection = ...`
编不过。Choice 可移动**构造**，所以最小修法是 optional + emplace（就地构造），使用点改 `->`；
catch 分支必定 continue，因此随后取值为不变量，加一条守卫把"不可达"写清楚。"""
import pathlib
import sys

P = pathlib.Path("/home/user/ninfer-fusion/src/runtime/engine/engine_core.h")
lines = P.read_text(encoding="utf-8").splitlines(keepends=True)

def find(pred, expect, what):
    hits = [i for i, l in enumerate(lines) if pred(l)]
    assert len(hits) == expect, "%s: 期望 %d 处，实际 %d" % (what, expect, len(hits))
    return hits

# --- 1) 两处声明 + 赋值 ---
decl = find(lambda l: l.strip() in ("ResourceInspection head_inspection;",
                                    "ResourceInspection candidate_inspection;"), 2, "声明")
for i in decl:
    name = "head_inspection" if "head_inspection" in lines[i] else "candidate_inspection"
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines[i] = "%sstd::optional<ResourceInspection> %s;\n" % (indent, name)

assign = find(lambda l: "= inspect_admission(" in l and l.strip().startswith(("head_inspection", "candidate_inspection")), 2, "赋值")
for i in assign:
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lhs = lines[i].split("=")[0].strip()
    rhs = lines[i].split("=", 1)[1].strip().rstrip(";")
    lines[i] = "%s%s.emplace(%s);\n" % (indent, lhs, rhs)

# --- 2) 使用点：字段访问改 `->` ---
uses = find(lambda l: ("head_inspection." in l or "candidate_inspection." in l)
            and not l.strip().startswith("std::optional"), 0, "-")
for i in uses:
    lines[i] = lines[i].replace("head_inspection.", "head_inspection->").replace(
        "candidate_inspection.", "candidate_inspection->")

# --- 3) 两处 try/catch 之后加不变量守卫（catch 必定 continue） ---
text = "".join(lines)
for name in ("head_inspection", "candidate_inspection"):
    # 找到 catch 块的结束 `}`（紧跟 continue; 之后的那个右花括号）
    pat = ("(void)remove_pending_error(%s, std::current_exception());"
           "\n                control_progress = true;\n                continue;\n            }\n") % name
    if name == "candidate_inspection":  # 第二处缩进多 4 空格
        pat = pat.replace("\n                ", "\n                    ")
    n = text.count(pat)
    assert n == 1, "%s: catch 块锚点 %d 处" % (name, n)
    guard = pat + ("            if (!%s) {\n"
                   "                // unreachable: the catch above always continues\n"
                   "                throw std::logic_error(\"S35: admission inspection missing\");\n"
                   "            }\n") % name
    if name == "candidate_inspection":
        guard = guard.replace("\n            if (!", "\n                if (!").replace(
            "\n                throw std::logic_error", "\n                    throw std::logic_error").replace(
            "\n            }\n", "\n                }\n")
    text = text.replace(pat, guard)
P.write_text(text, encoding="utf-8")
print("已修：2 处声明->optional、2 处赋值->emplace、%d 处访问->、2 处不变量守卫" % len(uses))
