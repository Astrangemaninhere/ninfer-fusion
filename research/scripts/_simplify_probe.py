#!/usr/bin/env python3
"""把探针简化成"每次调用落盘一份编号文件"，不在 engine 里做任何 host 计算。"""
import pathlib, hashlib, re

F = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash2_impl.h")
t = F.read_text()

# 1) 去掉 host 端的 tap 范数循环块（整段替换为一行简单摘要）
start = t.find("            {\n                std::vector<std::byte> hf(fv.bytes());")
end = t.find("            }\n", t.find('std::fprintf(stderr, "\\n");', start)) 
assert start > 0 and end > start, "host 循环块定位失败"
t = t[:start] + """            std::fprintf(stderr, "[df2feat] call=%d features[%d,%d]\\n", df2feat_calls,
                         fv.ne[0], fv.ne[1]);
""" + t[end + len("            }\n"):]

# 2) 三处落盘改为按调用序号编号
t = t.replace('"/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_features.bin"',
              '("/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_features_%d.bin" % df2feat_calls)')
t = re.sub(r'dump_tensor\(state\.execution\.device\.stream,\s*\n\s*\("([^"]+)" % df2feat_calls\)',
           lambda m: 'dump_tensor(state.execution.device.stream,\n                        ' + m.group(1), t)
F.write_text(t)
print("patched, md5", hashlib.md5(F.read_bytes()).hexdigest()[:12])
txt = F.read_text()
print("仍含 host 循环:", "reinterpret_cast<const std::uint16_t*>" in txt)
print("含编号文件名:", "feat_features_%d.bin" in txt)
