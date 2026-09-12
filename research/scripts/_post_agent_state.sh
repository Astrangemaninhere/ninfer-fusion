#!/bin/bash
R=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
cd "$R" || exit 3
echo "=== working tree status (what the agents changed / added) ==="
git status --short 2>/dev/null | head -20
echo
echo "=== any new i18n tables? ==="
ls -l tools/gui/i18n_*.py tools/gui/*selftest* 2>/dev/null | awk '{print "  ", $5, $NF}'
echo
echo "=== did serve_params.json get regenerated? ==="
ls -l --time-style=+%H:%M tools/gui/serve_params.json 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
python3 -c "
import json,pathlib
p=pathlib.Path('tools/gui/serve_params.json')
d=json.loads(p.read_text())
print('  total flags in registry:', d.get('total'))
print('  groups:', {k: len(v) for k, v in d.get('groups', {}).items()})
" 2>/dev/null
echo
echo "=== Spark download progress ==="
tail -3 /mnt/c/Users/User/Documents/ziqinzhang/dl/spark_weights.log 2>/dev/null | cut -c1-130
du -sh /mnt/c/Users/User/Documents/ziqinzhang/models/Spark-X2.5-4B 2>/dev/null
echo
echo "=== build ==="
grep -oE '^\[[ 0-9]+%\] (Building|Linking)[^"]*' /tmp/pa_make_1.log | tail -2
