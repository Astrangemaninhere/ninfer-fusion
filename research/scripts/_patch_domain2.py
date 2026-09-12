#!/usr/bin/env python3
"""补齐两处未匹配的编辑（缩进按实际 29 空格）。"""
import pathlib, sys
R = pathlib.Path("/home/user/ninfer-fusion")
EDITS = [
    ("src/ops/launcher/dflash2_selector.h",
     "                             Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
     "                             cudaStream_t stream);\n",
     "                             Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
     "                             std::int32_t domain, cudaStream_t stream);\n"),
    ("src/ops/launcher/dflash2_selector.cu",
     "                             Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
     "                             cudaStream_t stream) {\n",
     "                             Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
     "                             std::int32_t domain, cudaStream_t stream) {\n"),
]
ok = True
for rel, old, new in EDITS:
    p = R / rel
    s = p.read_text(encoding="utf-8", errors="surrogateescape")
    n = s.count(old)
    if n != 1:
        print("FAIL %-50s 匹配 %d 次" % (rel, n)); ok = False; continue
    p.write_text(s.replace(old, new), encoding="utf-8", errors="surrogateescape")
    print("OK   %s" % rel)
print("完成" if ok else "仍有失败")
sys.exit(0 if ok else 1)
