#!/usr/bin/env python3
"""Settle the S45c question mechanically: brace depth of each branch line, and which
arm a pure-ISO3 / pure-FP8 cache reaches at head_dim 256 (vs 128)."""
import re
import sys

path = "/tmp/s45c/src/ops/launcher/gqa_attention_prefill.cu"
lines = open(path).read().splitlines()

keys = ("} else if (cache.dtype", "if (cache.dtype == DType::NVFP4)", "if constexpr (Geometry::HeadDim",
        "} else {", "throw std::invalid_argument", "gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, DType::")

depth = 0
in_str = False
for i, ln in enumerate(lines, 1):
    stripped = ln.strip()
    hit = any(k in ln for k in keys)
    if hit:
        print("line %4d depth=%2d | %s" % (i, depth, stripped[:96]))
    # brace accounting on the raw line, ignoring braces inside quotes/comments
    code = ln.split("//")[0]
    code = re.sub(r"'[^']*'", "", code)
    code = re.sub(r'"[^"]*"', "", code)
    depth += code.count("{") - code.count("}")
print("\nfinal depth at EOF: %d (0 = balanced)" % depth)
