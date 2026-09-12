#!/bin/bash
PORT=8001
PROMPT='请详细介绍数学的发展历史'
N=$1
for i in $(seq 1 $N); do
  curl -s http://127.0.0.1:$PORT/v1/responses -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen3.8-27b\",\"input\":\"$PROMPT\",\"max_output_tokens\":300,\"stream\":false}" \
    -o /tmp/resp_$i.json &
done
wait
echo "REQUESTS_DONE=$N"
python3 - <<'PYEOF'
import json, glob
tot = 0
for p in glob.glob('/tmp/resp_*.json'):
    try:
        d = json.load(open(p))
        if 'output' in d:
            txt = ''.join(x.get('text','') for x in d['output'])
            tot += len(txt)
    except Exception:
        pass
print('total output chars:', tot)
PYEOF
