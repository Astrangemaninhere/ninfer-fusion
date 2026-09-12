# -*- coding: utf-8 -*-
"""Muse smoke: chat completion + basic assertions."""
import json
import subprocess
import sys
import time

BASE = 'http://127.0.0.1:8001/v1'
payload = {
    'model': 'muse-glimmer-30b',
    'messages': [{'role': 'user', 'content': 'What is 2+2? Answer in one word.'}],
    'max_tokens': 24,
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
print('stdout head:', r.stdout[:600].replace('\n', ' '))
if r.stdout.strip():
    try:
        d = json.loads(r.stdout)
        content = d['choices'][0]['message']['content']
        usage = d.get('usage', {})
        print('OK content=%r tokens=%s' % (content[:200], usage))
        open('/home/user/muse_smoke.json', 'w').write(json.dumps(d, ensure_ascii=False, indent=1))
        sys.exit(0)
    except Exception as e:
        print('PARSE FAIL', e)
        sys.exit(2)
sys.exit(3)
