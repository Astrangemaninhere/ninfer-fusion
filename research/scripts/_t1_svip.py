import re, html, sys

t = open('/home/user/svip.html', encoding='utf-8', errors='ignore').read()
t = re.sub(r'<script.*?</script>', '', t, flags=re.S)
t = re.sub(r'<style.*?</style>', '', t, flags=re.S)
t = re.sub(r'<[^>]+>', ' ', t)
t = html.unescape(t)
t = re.sub(r'\s+', ' ', t)
seen = set()
for pat in [r'algorithm', r'threshold', r'lower bound', r'entropy threshold']:
    for m in re.finditer(pat, t, flags=re.I):
        s = max(0, m.start() - 350)
        seg = t[s:m.end() + 450].strip()
        key = seg[:120]
        if key in seen:
            continue
        seen.add(key)
        print('###', pat.upper())
        print(seg)
        print()
        if len(seen) > 22:
            break
