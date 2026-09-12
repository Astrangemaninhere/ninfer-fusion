#!/bin/bash
R=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
echo "=== spec persistence fixed? ==="
python3 - <<'PY'
import json, pathlib
s = json.loads(pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/specs/spark-x2.5-4b_spec.json").read_text())
for k in ("rope_by_kind", "partial_rotary_by_kind", "hidden_act", "knobs", "layer_kind_order"):
    v = s.get(k)
    print("  %-24s %s" % (k, (json.dumps(v, ensure_ascii=False)[:120] if v is not None else "MISSING")))
print("  attention.headwise_attn_output_gate:", s.get("attention", {}).get("headwise_attn_output_gate"))
PY
echo
echo "=== engine serve flag surface (authoritative) ==="
grep -oE '"--[a-z0-9-]+"' "$R/src/serve/serve_options.cpp" | sort -u | tr '\n' ' ' | fold -w 150
echo
echo
echo "=== GUI: which flags/params does each module already expose? ==="
for f in serve_gui.py convert_gui.py model_import.py rag_gui.py; do
  echo "--- $f ---"
  grep -oE '\-\-[a-z0-9-]+' "$R/tools/gui/$f" 2>/dev/null | sort -u | tr '\n' ' '
  echo
done
echo
echo "=== gui_tips: how many tips / what shape ==="
grep -cE "^\s*\(|^\s*'|\"" "$R/tools/gui/gui_tips.py" 2>/dev/null | sed 's/^/  tip-ish lines: /'
head -20 "$R/tools/gui/gui_tips.py" | cut -c1-120
