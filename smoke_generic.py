# -*- coding: utf-8 -*-
"""Model smoke: chat completion assertion (generic, model name configurable)."""
import json
import subprocess
import sys
import time

BASE = 'http://127.0.0.1:%s/v1' % sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8001/v1'
MODEL = sys.argv[2] if len(sys.argv) > 2 else 'muse-glimmer-30b'
PROMPT = sys.argv[3] if len(sys.argv) > 3 else 'What is 2+2? Answer in one word.'
MAX_NEW = int(sys.argv[4]) if len(sys.argv) > 4 else 24

payload = {
    'model': MODEL,
    'messages': [{'role': 'user', 'content': PROMPT}],
    'max_tokens': MAX_NEW,
    'temperature': 0.0,
}
t0 = time.time()
try:
    r = subprocess.run(
        ['curl', '-s', '-m', '300', '-X', 'POST', BASE + '/chat/completions',
         '-H', 'Content-Type: application/json', '-d', json.dumps(payload)],
        capture_output=True, text=True, timeout=320)
except Exception as e:
    print('CURL FAIL', e)
    sys.exit(1)
dt = time.time() - t0
print('elapsed %.1fs curl_rc=%d' % (dt, r.returncode))
body = r.stdout.strip()
print('body head:', body[:500].replace('\n', ' '))
if not body:
    sys.exit(3)
try:
    d = json.loads(body)
    content = d['choices'][0]['message']['content']
    usage = d.get('usage', {})
    print('OK content=%r tokens=%s' % (content[:300], usage))
    open('/home/user/smoke_%s.json' % MODEL.split('/')[-1], 'w').write(
        json.dumps(d, ensure_ascii=False, indent=1))
    sys.exit(0)
except Exception as e:
    print('PARSE FAIL', e)
    sys.exit(2)
