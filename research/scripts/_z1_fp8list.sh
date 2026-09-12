#!/bin/bash
set -e
for repo in "Qwen/Qwen3.8-27B-FP8" "Qwen/Qwen3.8-27B" "Qwen/Qwen3.8-27B-NVFP4" "Qwen/Qwen3.5-27B"; do
  out="/tmp/repo_$(echo "$repo" | tr '/' '_').json"
  code=$(timeout 60 curl -s -o "$out" -w '%{http_code}' "https://hf-mirror.com/api/models/${repo}?blobs=true" || echo TIMEOUT)
  echo "### $repo  HTTP=$code"
  if [ "$code" = "200" ]; then
    python3 - "$out" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
tot = 0
rows = []
for s in d.get('siblings', []):
    sz = s.get('size') or 0
    tot += sz
    rows.append((s['rfilename'], sz))
for n, sz in rows:
    if sz > 1024 * 1024:
        print('   %-58s %10.1f MB' % (n, sz / 2**20))
print('   TOTAL %.2f GiB  (%d files)' % (tot / 2**30, len(rows)))
PYEOF
  fi
done
echo "=== disk ==="
df -h /mnt/c /mnt/g /mnt/wsl 2>/dev/null
