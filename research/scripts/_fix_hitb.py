import pathlib
import subprocess

p = pathlib.Path("/home/user/hitb3.py")
s = p.read_text()
old_key = 'm_r = [r for r in rounds if r["cols"].get(0, {}).get("verify") == a0 and r["pre"] == f0]'
new_key = 'm_r = [r for r in rounds if r["cols"].get(0, {}).get("pos") == f0]'
assert old_key in s, "对齐键那一行没找到"
s = s.replace(old_key, new_key)
old_pr = 'print(f"{n:>4} | {a0:>6} | {f0:>8} | 轮{r[\'row\']:>3} (accepted={r[\'accepted\']}) | {ranks}")'
new_pr = 'print(f"{n:>4} | {a0:>6} | {f0:>8} | 轮idx={rounds.index(r):>3} (accepted={r[\'accepted\']}) | {ranks}")'
if old_pr in s:
    s = s.replace(old_pr, new_pr)
    print("打印行也已改为轮索引")
p.write_text(s)
print("patched")
print(subprocess.run(["python3", "/home/user/hitb3.py"], capture_output=True, text=True).stdout[:3000])
