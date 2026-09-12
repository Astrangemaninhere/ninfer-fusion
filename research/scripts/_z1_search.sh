#!/bin/bash
for q in "Qwen3.8-27B" "Qwen3.8" "Qwen3.5-27B"; do
  echo "==================== search: $q ===================="
  out="/tmp/search_$(echo "$q" | tr -c 'a-zA-Z0-9' '_').json"
  timeout 60 curl -s "https://hf-mirror.com/api/models?search=${q}&limit=100&full=false" -o "$out"
  python3 - "$out" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("  parse error", e); sys.exit()
if not isinstance(d, list):
    print("  ->", str(d)[:300]); sys.exit()
for x in d:
    print("  %-64s dl=%-8s gated=%-5s" % (x.get('id'), x.get('downloads'), x.get('gated')))
PYEOF
done
