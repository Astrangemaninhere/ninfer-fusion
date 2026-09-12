#!/usr/bin/env python3
# 摸清 PART-B 在"被拆文件"里的 hunk 落点，判断 staged/staged2 产物是否过期
import re, pathlib
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang")
diff = (J / "_collab/PART_B_patch.diff").read_text(errors="replace").splitlines()

targets = ["launcher/gqa_attention_decode.cu", "launcher/gqa_attention_decode_smallt.cu",
           "launcher/gqa_attention_decode_partial.cuh", "launcher/gqa_attention_decode_e8.cu"]
cur = None
out = {t: [] for t in targets}
for i, ln in enumerate(diff):
    if ln.startswith("+++ b/"):
        cur = next((t for t in targets if ln[6:].endswith(t)), None)
        if cur: out[cur].append(("FILE", ""))
    elif cur and ln.startswith("@@"):
        out[cur].append(("@@", ln))
    elif cur and (ln.startswith("+++") or ln.startswith("---")) :
        pass
    elif cur and ln[:1] in "+-" and not ln.startswith(("+++", "---")):
        out[cur].append(("CHG", ln))

for t in targets:
    print("="*70)
    print(t)
    for kind, ln in out[t]:
        if kind == "@@":
            print("  " + ln)
        elif kind == "CHG":
            print("    " + ln[:110])

# staged2 兄弟 TU 里是否含 partial_acc
print("="*70)
print("staged2 兄弟 TU 含 partial_acc 的文件：")
for p in sorted((J / "_collab/build/staged2").glob("*.new")):
    txt = p.read_text(errors="replace")
    n = txt.count("partial_acc")
    if n: print(f"  {p.name}: {n} 处")
print("staged 产物含 partial_acc：")
for p in sorted((J / "_collab/build/staged").glob("*.new")):
    n = p.read_text(errors="replace").count("partial_acc")
    if n: print(f"  {p.name}: {n} 处")
