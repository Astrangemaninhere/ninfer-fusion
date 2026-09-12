import json, struct, pathlib, re
import numpy as np

J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ART = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer")

# 草稿域映射表（draft row -> vocab id）
with open(ART, "rb") as f:
    f.read(8); (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
payload = ((16 + jlen + 4095) // 4096) * 4096
objs = {o["name"]: o for o in j["objects"]}
o = objs["text/draft_head_token_ids"]
with open(ART, "rb") as f:
    f.seek(payload + o["offset"]); table = np.frombuffer(f.read(o["bytes"]), dtype=np.int32)
print("映射表 %d 项, 范围 [%d,%d]" % (len(table), table.min(), table.max()))

ROUND = re.compile(r"row=(\d+) anchor=(-?\d+) frontier=(-?\d+) valid=(-?\d+) in_extent=(-?\d+) "
                   r"out_extent=(-?\d+) accepted=(-?\d+) count=(-?\d+)")
COL   = re.compile(r"col=(-?\d+) pos=(-?\d+) verify=(-?\d+) draft=(-?\d+) argmax=(-?\d+)")
CAND  = re.compile(r"\[df2cand\] s=(-?\d+) pred=(-?\d+) chosen=(-?\d+) k=(-?\d+) (.*)$")

rounds, cur = [], None
for line in (J / "fx_spec.log").read_text(errors="replace").splitlines():
    m = ROUND.search(line)
    if m:
        if cur: rounds.append(cur)
        cur = {"cols": [], "sel": []}; continue
    if cur is None: continue
    m = COL.search(line)
    if m:
        cur["cols"].append({"col": int(m.group(1)), "draft": int(m.group(4)), "argmax": int(m.group(5))}); continue
    m = CAND.search(line)
    if m:
        cur["sel"].append({"s": int(m.group(1)), "chosen": int(m.group(3)),
                           "cands": [int(a) for a, _ in re.findall(r"(-?\d+):([-\d.]+)", m.group(5))]})
if cur: rounds.append(cur)

print("\n=== 判据 1：候选集里 chosen 那个 id 与 [df2dbg] 该列 draft 的关系 ===")
direct = via_map = neither = 0
for r in rounds:
    for st in r["sel"]:
        if not st["cands"]: continue
        c = st["cands"][st["chosen"]] if 0 <= st["chosen"] < len(st["cands"]) else None
        if c is None: continue
        # 对应列：约定 walk step s 的产物 = 列 s+1 的 draft
        col = next((x for x in r["cols"] if x["col"] == st["s"] + 1), None)
        if col is None: continue
        if c == col["draft"]:
            direct += 1
        elif 0 <= c < len(table) and int(table[c]) == col["draft"]:
            via_map += 1
        else:
            neither += 1
print("  候选id == 该列 draft（说明 id 已是真词表）        : %d" % direct)
print("  映射表[候选id] == 该列 draft（说明 id 是草稿域）  : %d" % via_map)
print("  两者都不成立                                      : %d" % neither)
print("  ⇒ 若 direct 占绝大多数，则 '[df2cand] 打印的就是真词表 id'，我先前的 head_miss 口径成立；")
print("     若 via_map 占绝大多数，则需映射后再比，head_miss 需重算。")
