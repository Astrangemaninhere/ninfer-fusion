#!/bin/bash
for repo in "$@"; do
  safe=$(echo "$repo" | tr '/' '_')
  out="/tmp/r_${safe}.json"
  code=$(timeout 90 curl -s -o "$out" -w '%{http_code}' "https://hf-mirror.com/api/models/${repo}?blobs=true")
  echo "### $repo  HTTP=$code"
  [ "$code" = "200" ] || continue
  python3 - "$out" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
sib = d.get('siblings', [])
tot = 0
for s in sib:
    sz = s.get('size') or 0
    tot += sz
print('   files=%d  TOTAL=%.2f GiB' % (len(sib), tot / 2**30))
big = sorted(((s.get('size') or 0), s['rfilename']) for s in sib)[::-1]
for sz, n in big[:14]:
    print('   %-64s %9.1f MB' % (n, sz / 2**20))
rest = [n for sz, n in big[14:] if n.endswith('.json') or n.endswith('.jinja') or n.endswith('.txt')]
print('   small/config:', rest[:12])
print('   tags:', (d.get('tags') or [])[:12])
PYEOF
done
