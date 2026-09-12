import pathlib

p = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash2_impl.h")
raw = p.read_bytes().decode("utf-8")
print("含 CRLF:", "\r\n" in raw)
lines = raw.split("\r\n") if "\r\n" in raw else raw.split("\n")
for i, l in enumerate(lines):
    if 392 <= i + 1 <= 412:
        print(i + 1, repr(l))
