#!/usr/bin/env python3
"""Inspect the conv.* and rag.* keys the agent already wrote."""
import sys

sys.path.insert(0, "/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui")
import gui_i18n as i18n  # noqa: E402

for area in ("conv.", "rag.", "misc."):
    ks = sorted(k for k in i18n.STRINGS if k.startswith(area))
    print("%s* keys: %d" % (area, len(ks)))
    for k in ks:
        v = i18n.STRINGS[k]
        print("  %-34s zh=%-32s en=%s" % (k, v.get("zh", "")[:32], v.get("en", "")[:40]))
    print()
