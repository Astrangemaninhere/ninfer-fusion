import re
from collections import Counter

sass = open('/home/user/bench/swiglu_full.sass').read()
start = sass.find('swiglu_decode_kernel')
if start < 0:
    raise SystemExit('kernel not found')
ops = Counter()
for ln in sass[start:].splitlines():
    m = re.search(r'\*/\s+([A-Z0-9@!._]+)', ln)
    if m:
        ops[m.group(1).split('.')[0]] += 1
total = sum(ops.values())
print('total:', total)
for op, c in ops.most_common(25):
    print(f'{op:12s} {c:6d}  {c/total*100:5.1f}%')
