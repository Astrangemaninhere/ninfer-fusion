#!/bin/bash
# temp: replay the real compile commands for every test TU with -fsyntax-only
set -u
SRC=/home/user/ninfer-fusion
CC_JSON=$SRC/build/compile_commands.json
python3 - "$CC_JSON" <<'PY' > /tmp/test_syntax_plan.txt
import json, shlex, sys
data = json.load(open(sys.argv[1]))
for entry in data:
    f = entry.get("file", "")
    if "/tests/" not in f.replace("\\", "/"):
        continue
    cmd = entry.get("command")
    if not cmd:
        continue
    print(shlex.quote(f) + "\t" + cmd)
PY
TOTAL=0
FAILED=0
while IFS=$'\t' read -r file cmd; do
  TOTAL=$((TOTAL+1))
  # Replace the compile-and-emit step with a syntax-only pass.
  newcmd=$(echo "$cmd" | sed -e 's/ -c / -fsyntax-only /' -e 's/ -o [^ ]*\.o//g')
  out=$(cd "$SRC/build" && eval "$newcmd" 2>&1 | head -n 3)
  if [ -n "$out" ]; then
    FAILED=$((FAILED+1))
    echo "=== FAIL $file"
    echo "$out"
  fi
done < /tmp/test_syntax_plan.txt
echo "checked=$TOTAL failed=$FAILED"
