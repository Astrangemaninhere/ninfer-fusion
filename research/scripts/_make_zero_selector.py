import json, struct, shutil, pathlib, time

SRC = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\data\dflash2_ref_head.ninfer")
DST = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\data\dflash2_zeroselector.ninfer")

def align_up(x, a=4096):
    return ((x + a - 1) // a) * a

with open(SRC, "rb") as f:
    f.read(8)
    (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
payload = align_up(16 + jlen)
objs = {o["name"]: o for o in j["objects"]}
print("payload_start =", payload)

targets = ["dflash2/candidate_selector/predecessor_codebook",
           "dflash2/candidate_selector/successor_codebook"]
for t in targets:
    o = objs[t]
    print("  %s  offset=%d bytes=%d" % (t, o["offset"], o["bytes"]))

print("复制 ->", DST.name)
t0 = time.time()
shutil.copyfile(SRC, DST)
print("  复制 %.1fs" % (time.time() - t0))

with open(DST, "r+b") as f:
    for t in targets:
        o = objs[t]
        f.seek(payload + o["offset"])
        f.write(b"\x00" * o["bytes"])
print("两张码本已清零（写入 0 字节）")

# 校验：清零后回读应全 0
with open(DST, "rb") as f:
    for t in targets:
        o = objs[t]
        f.seek(payload + o["offset"])
        buf = f.read(min(o["bytes"], 1 << 20))
        print("  %s 前 1MB 全零? %s" % (t, all(b == 0 for b in buf)))
print("OUT =", DST)
print("ZERO_SELECTOR_DONE")
