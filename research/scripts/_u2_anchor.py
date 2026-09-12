#!/usr/bin/env python3
p = "/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash_impl.h"
s = open(p, encoding="utf-8", newline="").read()
a = ('        ops::prepare_masked_block(anchors, frontiers, attention_valid, Config::mask_token, ids,\n'
     '                                  positions, state.execution.device.stream);\n')
b = '        constexpr std::size_t source_column_offset = 1;\n'
c = '#include "core/dtype.h"\n'
print("anchor1 (prepare_masked_block 2-line) count =", s.count(a))
print("anchor2 (source_column_offset)        count =", s.count(b))
print("dtype.h include                       count =", s.count(c))
print("already has <cstdlib> :", "#include <cstdlib>" in s)
print("lines:", s.count("\n"))
