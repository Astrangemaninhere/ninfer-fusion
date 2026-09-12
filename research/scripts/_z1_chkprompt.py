import ast
import re
import sys

src = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"^PROMPT = (.*)$", src, re.M)
print("PROMPT literal:", m.group(1))
p = ast.literal_eval(m.group(1))
print("decoded      :", p)
print("n chars      :", len(p))
compile(src, "z1", "exec")
print("syntax OK")
for key in ("TARGET", "DRAFT", "RESULT", "LOG"):
    mm = re.search(r"^%s = (.*)$" % key, src, re.M)
    print(key, "=", ast.literal_eval(mm.group(1)))
