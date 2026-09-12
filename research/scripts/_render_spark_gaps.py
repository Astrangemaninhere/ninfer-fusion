#!/usr/bin/env python3
"""用真实 Spark manifest 渲染导入页的缺口表（中/英），验证 covered 等级的标签与文案。"""
import json
import pathlib
import sys

G = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui")
sys.path.insert(0, str(G))
import gui_i18n as i18n  # noqa: E402
import model_import as mi  # noqa: E402

man = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/"
                   "tools/archkit/out/spark-x2.5-4b/manifest.json")

# 找到渲染入口（名字可能是 report_* / gap_*）
cands = [n for n in dir(mi) if any(k in n.lower() for k in ("report", "gaps", "verdict", "render"))]
print("渲染相关入口:", cands)
fn = None
for name in ("gap_report", "build_report", "report", "render_report", "import_report"):
    if hasattr(mi, name):
        fn = getattr(mi, name)
        print("使用入口:", name)
        break

for lang in ("zh", "en"):
    i18n.set_lang(lang)
    print("\n" + "=" * 78)
    print("LANG =", lang)
    print("=" * 78)
    if fn is None:
        # 退而求其次：直接调用 tier 标签与单行文案渲染
        print("tier(covered) =", mi.tier_label("covered"))
        print("tier(new_op)  =", mi.tier_label("new_op"))
        print("tied 行       =", mi.gap_action_text(
            "head:tied=true", "covered",
            open(man).read() and json.loads(man.read_text())["gaps"][5]["action"]))
    else:
        try:
            out = fn(str(man))
        except TypeError:
            out = fn(json.loads(man.read_text()))
        txt = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=1)
        print(txt[:2200])
i18n.set_lang("zh")
