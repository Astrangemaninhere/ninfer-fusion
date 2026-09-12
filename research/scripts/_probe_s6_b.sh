#!/usr/bin/env bash
# S6 recon probe B: timing evidence, build loop, compile flags (read-only)
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang

echo "=== patchA_build.log: e8 mentions ==="
grep -n 'gqa_attention' "$J"/dl/patchA_build.log 2>/dev/null | tail -10
ls -la "$J"/dl/patchA_build.log 2>/dev/null

echo
echo "=== ninja/make timing logs ==="
ls -la "$R"/build/.ninja_log "$R"/build/CMakeCache.txt 2>/dev/null
if [ -f "$R"/build/.ninja_log ]; then
  grep -c . "$R"/build/.ninja_log
fi

echo
echo "=== _par_build.sh ==="
cat -n "$J"/_par_build.sh

echo
echo "=== compile_commands entry for gqa_attention_decode_e8.cu ==="
python3 - "$R" <<'PY'
import json, sys, os
R = sys.argv[1]
p = os.path.join(R, 'build', 'compile_commands.json')
if not os.path.isfile(p):
    print("(no compile_commands.json)"); raise SystemExit
db = json.load(open(p))
for e in db:
    f = e.get('file', '')
    if f.endswith(('gqa_attention_decode_e8.cu', 'gqa_attention_prefill.cu', 'gqa_attention_prefill_e8.cu')):
        cmd = e.get('command') or ' '.join(e.get('arguments', []))
        print("---", os.path.relpath(f, R))
        print(cmd[:1200])
        print()
PY
