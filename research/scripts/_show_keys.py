#!/usr/bin/env python3
"""打印现有 tier / tied 相关键的取值（给改判后的文案更新做基准）。"""
import sys

sys.path.insert(0, "/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui")
import gui_i18n as g  # noqa: E402

for k in ("imp.tier.hook", "imp.tier.new_op", "imp.tier.post", "imp.tier.other",
          "imp.gap.head_tied_true", "imp.gap.head_tied_true.nosize",
          "imp.gaps.fallback", "imp.verdict_blocked", "imp.verdict_ok"):
    v = g.STRINGS.get(k)
    if v:
        print("%-34s zh=%s\n%-34s en=%s" % (k, v.get("zh"), "", v.get("en")))
    else:
        print("%-34s <missing>" % k)
