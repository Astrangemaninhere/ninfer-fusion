#!/usr/bin/env python3
"""改判收尾（S53 裁决的落地）：
  1) 界面上新增 `covered` 等级的中英标签（不然显示成"其他等级"）；
  2) `head:tied=true` 这行改成**按 tier 分档**：new_op 仍用旧文案（其它模型可能真没法办），
     covered 用新文案（转换期物化、无需新算子、转换器已实现 source_map.py）；
  3) 保持 --fuzz=0 的机械性：只加键、只改一行映射与一行顺序表。"""
import pathlib
import sys

G = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui")
MI = G / "model_import.py"
I18N = G / "i18n_misc.py"

# ---------- 1) model_import.py：顺序表 + tier 标签 + 按 tier 分档的 tied 文案 ----------
src = MI.read_text(encoding="utf-8")
old_order = '''TIER_ORDER = ("hook", "new_op", "post")
TIER_KEYS = {"hook": "imp.tier.hook", "new_op": "imp.tier.new_op",
             "post": "imp.tier.post"}'''
new_order = '''TIER_ORDER = ("hook", "new_op", "covered", "post")
TIER_KEYS = {"hook": "imp.tier.hook", "new_op": "imp.tier.new_op",
             "covered": "imp.tier.covered", "post": "imp.tier.post"}'''
if new_order in src:
    print("TIER 表已更新过")
else:
    assert src.count(old_order) == 1, "TIER 锚点不唯一"
    src = src.replace(old_order, new_order)
    print("TIER_ORDER/TIER_KEYS 已加 covered")

old_tied = '''    (re.compile(r"^head:tied=true$"), "imp.gap.head_tied_true", ("mb",)),'''
new_tied = '''    # adapt.py reclassified the tied head from new_op to covered once the converter
    # grew materialisation (S53): the engine still loads an independent lm_head object,
    # so "no new operator" is true, but the conversion-time cost must stay visible.
    (re.compile(r"^head:tied=true$"),
     {"default": "imp.gap.head_tied_true", "covered": "imp.gap.head_tied_true_covered"},
     ("mb",)),'''
if "imp.gap.head_tied_true_covered" in src:
    print("tied 行已分档")
else:
    assert src.count(old_tied) == 1, "tied 锚点不唯一"
    src = src.replace(old_tied, new_tied)
    print("tied 行已按 tier 分档")
MI.write_text(src, encoding="utf-8")

# ---------- 2) i18n_misc.py：补两个新键（zh+en） ----------
t = I18N.read_text(encoding="utf-8")
anchor = "  'imp.tier.other':"
if "'imp.tier.covered'" in t:
    print("i18n 键已存在")
else:
    idx = t.find(anchor)
    assert idx != -1, "找不到 imp.tier.other 锚点"
    line_end = t.find("\n", idx)
    add = (
        "\n  'imp.tier.covered': {'zh': '已覆盖（无需引擎改动）',\n"
        "                        'en': 'covered (no engine change needed)'},"
        "\n  'imp.gap.head_tied_true_covered': {\n"
        "      'zh': 'tied head：转换期把嵌入键物化为独立 head 对象（约 {mb} MB），引擎无需新算子；'\n"
        "            '转换器已实现（tools/convert/common/source_map.py）',\n"
        "      'en': 'tied head: materialise the embedding as a separate head object at conversion '\n"
        "            'time (~{mb} MB); no new engine operator is needed and the converter now '\n"
        "            'implements this (tools/convert/common/source_map.py)'},\n"
        "  'imp.gap.head_tied_true_covered.nosize': {\n"
        "      'zh': 'tied head：转换期把嵌入键物化为独立 head 对象（额外权重，体积见 manifest），'\n"
        "            '引擎无需新算子；转换器已实现（tools/convert/common/source_map.py）',\n"
        "      'en': 'tied head: materialise the embedding as a separate head object at conversion '\n"
        "            'time (extra weights; see the manifest for the size); no new engine operator '\n"
        "            'is needed and the converter now implements this '\n"
        "            '(tools/convert/common/source_map.py)'},"
    )
    t = t[:line_end] + add + t[line_end:]
    I18N.write_text(t, encoding="utf-8")
    print("i18n_misc 补入 3 个键（含 .nosize 兜底）")
print("done")
