#!/bin/bash
G=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui
echo "=== GUI files touched in the last 45 min ==="
find "$G" -maxdepth 1 -type f -newermt '-45 minutes' -printf '%TH:%TM %8s %f\n' 2>/dev/null | sort
echo
echo "=== i18n_serve.py shape (first 30 lines) ==="
head -30 "$G/i18n_serve.py" | cut -c1-120
echo
echo "=== i18n_misc.py shape (first 24 lines + entry count) ==="
head -24 "$G/i18n_misc.py" | cut -c1-120
python3 - <<'PY'
import sys, pathlib
sys.path.insert(0, "/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui")
import gui_i18n as i18n
print("  merged keys: %d  coverage: %s" % (len(i18n.STRINGS), i18n.coverage()))
missing_en = [k for k, v in i18n.STRINGS.items() if 'en' not in v]
print("  entries without en: %d %s" % (len(missing_en), missing_en[:6]))
sample = list(i18n.STRINGS.items())[:4]
for k, v in sample:
    print("   %-28s zh=%s | en=%s" % (k[:28], str(v.get('zh'))[:34], str(v.get('en'))[:34]))
PY
echo
echo "=== serve_gui.py: does it consume the registry now? ==="
grep -nE 'serve_params|registry|PARAMS|groups|gui_i18n|import ' "$G/serve_gui.py" | head -14 | cut -c1-120
